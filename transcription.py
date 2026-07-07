"""
Optional Twitch stream audio transcription.

This module keeps transcription separate from chat handling: it resolves a
Twitch HLS stream with Streamlink, asks FFmpeg for 16 kHz mono PCM audio, and
feeds bounded chunks to faster-whisper in a background task.
"""

import asyncio
import os
import tempfile
import wave
from dataclasses import dataclass
from typing import Awaitable, Callable, Optional

from streamlink.options import Options
from streamlink.session import Streamlink

import config


SAMPLE_RATE = 16000
BYTES_PER_SAMPLE = 2
CHANNELS = 1


@dataclass
class AudioChunk:
    wav_path: str
    chunk_start: float
    skip_initial_seconds: float


class StreamAudioTranscriber:
    """Background transcriber for one Twitch channel."""

    def __init__(self, database):
        self.database = database
        self.channel = config.TRANSCRIPTION_CHANNEL.lower()
        self.model_name = config.TRANSCRIPTION_MODEL
        self.device = config.TRANSCRIPTION_DEVICE
        self.compute_type = config.TRANSCRIPTION_COMPUTE_TYPE
        self.language = config.TRANSCRIPTION_LANGUAGE
        self.stream_quality = config.TRANSCRIPTION_STREAM_QUALITY
        self.chunk_seconds = max(2.0, config.TRANSCRIPTION_CHUNK_SECONDS)
        self.overlap_seconds = max(0.0, min(config.TRANSCRIPTION_OVERLAP_SECONDS, self.chunk_seconds - 0.5))
        self.reconnect_delay = max(1.0, config.TRANSCRIPTION_RECONNECT_DELAY)
        self.vad_filter = config.TRANSCRIPTION_VAD_FILTER

        self._model = None
        self._task: Optional[asyncio.Task] = None
        self._transcription_task: Optional[asyncio.Task] = None
        self._stop_event = asyncio.Event()
        self._ffmpeg_process: Optional[asyncio.subprocess.Process] = None
        self._stderr_task: Optional[asyncio.Task] = None
        self._audio_chunk_queue: Optional[asyncio.Queue[AudioChunk]] = None
        self._audio_offset_seconds = 0.0
        self._last_transcript_text = ""
        self._utterance_callback: Optional[Callable[[str, str], Awaitable[bool]]] = None
        self._wait_until_llm_idle: Optional[Callable[[], Awaitable[None]]] = None
        self._pending_utterance_parts: list[str] = []
        self._pending_utterance_start: Optional[float] = None
        self._pending_utterance_end: Optional[float] = None
        self._utterance_flush_task: Optional[asyncio.Task] = None
        self._utterance_version = 0
        self.last_error: Optional[str] = None

    @property
    def is_running(self) -> bool:
        return self._task is not None and not self._task.done()

    def set_utterance_callback(self, callback: Callable[[str, str], Awaitable[bool]]) -> None:
        """Set a callback for finalized streamer speech utterances."""
        self._utterance_callback = callback

    def set_llm_idle_waiter(self, callback: Callable[[], Awaitable[None]]) -> None:
        """Set a callback used to defer Whisper while the LLM is responding."""
        self._wait_until_llm_idle = callback

    async def start(self) -> tuple[bool, str]:
        """Start the transcription background task."""
        if self.is_running:
            return True, f"Transcription is already running for #{self.channel}"

        try:
            await self._ensure_model_loaded()
        except Exception as exc:
            self.last_error = str(exc)
            return False, f"Could not start transcription: {exc}"

        self._stop_event = asyncio.Event()
        self._audio_chunk_queue = asyncio.Queue(maxsize=config.TRANSCRIPTION_AUDIO_QUEUE_MAX_SIZE)
        self.last_error = None
        self._transcription_task = asyncio.create_task(
            self._transcribe_queued_audio(),
            name=f"transcribe-worker-{self.channel}"
        )
        self._task = asyncio.create_task(self._run(), name=f"transcribe-{self.channel}")
        return True, f"Transcription started for #{self.channel}"

    async def stop(self) -> tuple[bool, str]:
        """Stop the transcription background task and FFmpeg process."""
        if not self.is_running:
            return True, "Transcription is already stopped"

        self._stop_event.set()
        await self._flush_streamer_utterance(force=True)
        await self._stop_ffmpeg()

        try:
            await asyncio.wait_for(self._task, timeout=10)
        except asyncio.TimeoutError:
            self._task.cancel()
            try:
                await self._task
            except asyncio.CancelledError:
                pass

        await self._stop_transcription_worker()
        return True, "Transcription stopped"

    def status(self) -> str:
        """Return a compact status string for chat commands."""
        state = "running" if self.is_running else "stopped"
        error = f" Last error: {self.last_error}" if self.last_error else ""
        return f"Transcription is {state} for #{self.channel} ({self.model_name}/{self.device}).{error}"

    async def _ensure_model_loaded(self) -> None:
        if self._model is not None:
            return

        try:
            from faster_whisper import WhisperModel
        except ImportError as exc:
            raise RuntimeError("faster-whisper is not installed in the venv") from exc

        print(
            f"[Transcription] Loading faster-whisper model '{self.model_name}' "
            f"on {self.device} ({self.compute_type})"
        )
        self._model = await asyncio.to_thread(
            WhisperModel,
            self.model_name,
            device=self.device,
            compute_type=self.compute_type
        )

    async def _run(self) -> None:
        print(f"[Transcription] Background transcriber ready for #{self.channel}")

        while not self._stop_event.is_set():
            try:
                await self._transcribe_stream_once()
            except asyncio.CancelledError:
                raise
            except Exception as exc:
                self.last_error = str(exc)
                print(f"[Transcription] Error: {exc}")
            finally:
                await self._stop_ffmpeg()

            if not self._stop_event.is_set():
                await asyncio.sleep(self.reconnect_delay)

        print("[Transcription] Background transcriber stopped")

    async def _transcribe_stream_once(self) -> None:
        stream_url = await asyncio.to_thread(self._resolve_stream_url)
        if not stream_url:
            self.last_error = f"No stream URL available for #{self.channel}"
            print(f"[Transcription] {self.last_error}")
            return

        command = [
            "ffmpeg",
            "-nostdin",
            "-hide_banner",
            "-loglevel",
            "warning",
            "-i",
            stream_url,
            "-vn",
            "-ac",
            str(CHANNELS),
            "-ar",
            str(SAMPLE_RATE),
            "-f",
            "s16le",
            "pipe:1"
        ]

        print(f"[Transcription] Starting FFmpeg audio pipe for #{self.channel}")
        self._ffmpeg_process = await asyncio.create_subprocess_exec(
            *command,
            stdout=asyncio.subprocess.PIPE,
            stderr=asyncio.subprocess.PIPE,
            stdin=asyncio.subprocess.DEVNULL
        )
        self._stderr_task = asyncio.create_task(self._drain_ffmpeg_stderr(self._ffmpeg_process))

        await self._read_pcm_chunks(self._ffmpeg_process)

        return_code = await self._ffmpeg_process.wait()
        if return_code != 0 and not self._stop_event.is_set():
            raise RuntimeError(f"FFmpeg exited with code {return_code}")

    def _resolve_stream_url(self) -> Optional[str]:
        channel_url = f"https://www.twitch.tv/{self.channel}"
        session = Streamlink()
        _plugin_name, plugin_class, resolved_url = session.resolve_url(channel_url)

        options = Options()
        options.set("disable-ads", True)
        options.set("low-latency", True)
        if config.STREAMLINK_OAUTH_TOKEN:
            options.set("api-header", [("Authorization", f"OAuth {config.STREAMLINK_OAUTH_TOKEN}")])

        plugin = plugin_class(session, resolved_url, options)
        streams = plugin.streams()
        if not streams:
            return None

        preferred_names = [
            self.stream_quality,
            "audio_only",
            "480p",
            "720p",
            "720p60",
            "360p",
            "worst",
            "best"
        ]
        seen_names = set()
        for name in preferred_names:
            if name in seen_names:
                continue
            seen_names.add(name)
            stream = streams.get(name)
            if stream:
                print(f"[Transcription] Using stream quality '{name}'")
                return stream.url

        return None

    async def _read_pcm_chunks(self, process: asyncio.subprocess.Process) -> None:
        if process.stdout is None:
            raise RuntimeError("FFmpeg stdout pipe was not created")

        chunk_bytes = int(self.chunk_seconds * SAMPLE_RATE * BYTES_PER_SAMPLE)
        overlap_bytes = int(self.overlap_seconds * SAMPLE_RATE * BYTES_PER_SAMPLE)
        step_bytes = max(BYTES_PER_SAMPLE, chunk_bytes - overlap_bytes)
        step_seconds = step_bytes / (SAMPLE_RATE * BYTES_PER_SAMPLE)
        minimum_final_bytes = int(2.0 * SAMPLE_RATE * BYTES_PER_SAMPLE)
        buffer = bytearray()

        while not self._stop_event.is_set():
            data = await process.stdout.read(4096)
            if not data:
                break

            buffer.extend(data)
            while len(buffer) >= chunk_bytes and not self._stop_event.is_set():
                chunk = bytes(buffer[:chunk_bytes])
                chunk_start = self._audio_offset_seconds
                skip_initial = self.overlap_seconds if chunk_start > 0 else 0.0
                await self._spool_pcm_chunk(chunk, chunk_start, skip_initial)
                del buffer[:step_bytes]
                self._audio_offset_seconds += step_seconds

        if len(buffer) >= minimum_final_bytes and not self._stop_event.is_set():
            await self._spool_pcm_chunk(bytes(buffer), self._audio_offset_seconds, 0.0)

    async def _spool_pcm_chunk(self, pcm_data: bytes, chunk_start: float, skip_initial_seconds: float) -> None:
        if not pcm_data or not self._audio_chunk_queue:
            return

        wav_path = self._write_temp_wav(pcm_data)
        chunk = AudioChunk(
            wav_path=wav_path,
            chunk_start=chunk_start,
            skip_initial_seconds=skip_initial_seconds
        )

        if self._audio_chunk_queue.full():
            try:
                dropped = self._audio_chunk_queue.get_nowait()
                self._audio_chunk_queue.task_done()
                self._delete_wav_file(dropped.wav_path)
                print("[Transcription] Dropped oldest queued audio chunk to stay realtime")
            except asyncio.QueueEmpty:
                pass

        try:
            self._audio_chunk_queue.put_nowait(chunk)
        except asyncio.QueueFull:
            self._delete_wav_file(wav_path)
            print("[Transcription] Dropped audio chunk because transcription queue is full")

    async def _transcribe_queued_audio(self) -> None:
        while not self._stop_event.is_set() or (self._audio_chunk_queue and not self._audio_chunk_queue.empty()):
            if not self._audio_chunk_queue:
                await asyncio.sleep(0.2)
                continue

            try:
                chunk = await asyncio.wait_for(self._audio_chunk_queue.get(), timeout=0.5)
            except asyncio.TimeoutError:
                continue

            try:
                if (
                    config.TRANSCRIPTION_PAUSE_WHILE_LLM_BUSY
                    and self._wait_until_llm_idle
                    and not self._stop_event.is_set()
                ):
                    await self._wait_until_llm_idle()

                await self._transcribe_wav_chunk(chunk)
            except asyncio.CancelledError:
                raise
            except Exception as exc:
                self.last_error = str(exc)
                print(f"[Transcription] Error transcribing queued audio: {exc}")
            finally:
                self._delete_wav_file(chunk.wav_path)
                self._audio_chunk_queue.task_done()

        await self._flush_streamer_utterance(force=True)

    async def _transcribe_wav_chunk(self, chunk: AudioChunk) -> None:
        if not chunk.wav_path:
            return

        segments, info = await asyncio.to_thread(self._transcribe_wav_file, chunk.wav_path)

        for segment in segments:
            if chunk.skip_initial_seconds and segment.end <= chunk.skip_initial_seconds:
                continue

            text = " ".join(segment.text.split())
            if not text or self._is_duplicate_text(text):
                continue

            start_offset = chunk.chunk_start + float(segment.start)
            end_offset = chunk.chunk_start + float(segment.end)
            language = getattr(info, "language", None)
            avg_logprob = getattr(segment, "avg_logprob", None)
            no_speech_prob = getattr(segment, "no_speech_prob", None)

            self.database.store_transcript(
                self.channel,
                text,
                start_offset=start_offset,
                end_offset=end_offset,
                language=language,
                avg_logprob=avg_logprob,
                no_speech_prob=no_speech_prob
            )
            print(f"[Transcription] #{self.channel} +{start_offset:.0f}s: {text}")
            await self._buffer_streamer_utterance(text, start_offset, end_offset)

    def _transcribe_wav_file(self, wav_path: str):
        options = {
            "vad_filter": self.vad_filter,
            "beam_size": 5,
            "condition_on_previous_text": False
        }
        if self.language:
            options["language"] = self.language
        if self.vad_filter:
            options["vad_parameters"] = {
                "threshold": 0.5,
                "min_speech_duration_ms": 250,
                "min_silence_duration_ms": 500,
                "speech_pad_ms": 400
            }

        segments, info = self._model.transcribe(wav_path, **options)
        return list(segments), info

    def _write_temp_wav(self, pcm_data: bytes) -> str:
        fd, path = tempfile.mkstemp(prefix="vaarabot_transcript_", suffix=".wav")
        os.close(fd)
        with wave.open(path, "wb") as wav_file:
            wav_file.setnchannels(CHANNELS)
            wav_file.setsampwidth(BYTES_PER_SAMPLE)
            wav_file.setframerate(SAMPLE_RATE)
            wav_file.writeframes(pcm_data)
        return path

    def _delete_wav_file(self, wav_path: str) -> None:
        try:
            os.unlink(wav_path)
        except OSError:
            pass

    def _is_duplicate_text(self, text: str) -> bool:
        normalized = " ".join(text.lower().split())
        if normalized == self._last_transcript_text:
            return True

        self._last_transcript_text = normalized
        return False

    async def _buffer_streamer_utterance(self, text: str, start_offset: float, end_offset: float) -> None:
        if not self._utterance_callback or not config.STREAMER_SPEECH_RESPONSES_ENABLED:
            return

        if self._pending_utterance_start is None:
            self._pending_utterance_start = start_offset

        self._pending_utterance_parts.append(text)
        self._pending_utterance_end = end_offset
        self._utterance_version += 1

        utterance_duration = end_offset - self._pending_utterance_start
        if utterance_duration >= config.STREAMER_SPEECH_MAX_UTTERANCE_SECONDS:
            await self._flush_streamer_utterance(force=True)
            return

        if self._utterance_flush_task and not self._utterance_flush_task.done():
            self._utterance_flush_task.cancel()

        version = self._utterance_version
        self._utterance_flush_task = asyncio.create_task(
            self._flush_streamer_utterance_after_quiet(version),
            name="streamer-utterance-flush"
        )

    async def _flush_streamer_utterance_after_quiet(self, version: int) -> None:
        try:
            await asyncio.sleep(config.STREAMER_SPEECH_QUIET_SECONDS)
            if version == self._utterance_version:
                await self._flush_streamer_utterance(force=False)
        except asyncio.CancelledError:
            pass

    async def _flush_streamer_utterance(self, force: bool) -> None:
        if not self._pending_utterance_parts:
            return

        current_task = asyncio.current_task()
        if (
            self._utterance_flush_task
            and not self._utterance_flush_task.done()
            and self._utterance_flush_task is not current_task
        ):
            self._utterance_flush_task.cancel()
            self._utterance_flush_task = None

        text = " ".join(self._pending_utterance_parts)
        word_count = len(text.split())

        self._pending_utterance_parts = []
        self._pending_utterance_start = None
        self._pending_utterance_end = None

        if not force and word_count < config.STREAMER_SPEECH_MIN_WORDS:
            return

        if word_count < config.STREAMER_SPEECH_MIN_WORDS:
            return

        if self._utterance_callback:
            queued = await self._utterance_callback(self.channel, text)
            if queued:
                print(f"[Transcription] Queued streamer utterance: {text[:120]}")

    async def _drain_ffmpeg_stderr(self, process: asyncio.subprocess.Process) -> None:
        if process.stderr is None:
            return

        while True:
            line = await process.stderr.readline()
            if not line:
                break
            if config.DEBUG_LOGGING:
                text = line.decode("utf-8", errors="ignore").strip()
                if text:
                    print(f"[Transcription FFmpeg] {text}")

    async def _stop_ffmpeg(self) -> None:
        process = self._ffmpeg_process
        if process and process.returncode is None:
            print("[Transcription] Stopping FFmpeg audio pipe")
            try:
                process.terminate()
            except ProcessLookupError:
                pass

            try:
                await asyncio.wait_for(process.wait(), timeout=5)
            except asyncio.TimeoutError:
                process.kill()
                await process.wait()

        if self._stderr_task:
            try:
                await asyncio.wait_for(self._stderr_task, timeout=2)
            except asyncio.TimeoutError:
                self._stderr_task.cancel()
            self._stderr_task = None

        self._ffmpeg_process = None

    async def _stop_transcription_worker(self) -> None:
        if not self._transcription_task:
            return

        try:
            await asyncio.wait_for(self._transcription_task, timeout=10)
        except asyncio.TimeoutError:
            self._transcription_task.cancel()
            try:
                await self._transcription_task
            except asyncio.CancelledError:
                pass

        self._transcription_task = None

        if self._audio_chunk_queue:
            while not self._audio_chunk_queue.empty():
                try:
                    chunk = self._audio_chunk_queue.get_nowait()
                except asyncio.QueueEmpty:
                    break
                self._delete_wav_file(chunk.wav_path)
                self._audio_chunk_queue.task_done()
