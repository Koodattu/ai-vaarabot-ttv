# 🤖 Vaarattu's Twitch AI Chat Bot

A witty Twitch chat bot powered by Google Gemini AI. Responds when mentioned with personality, humor, and style.

## ✨ Features

- **🎯 Mention-based** - Responds when pinged with `@botname`
- **💬 Conversation memory** - Remembers last 10 exchanges per user for context
- **🌍 Multi-language** - Detects and responds in the user's language
- **😄 Personality** - Witty, playful, and fun while staying appropriate
- **📺 Multi-channel** - Can join one or more Twitch channels
- **⏱️ Rate limiting** - Per-user cooldown and hourly message limits
- **🔐 OAuth flow** - Automatic token management with refresh support

## 🚀 Setup

### 1. Create a Twitch Application

1. Go to [Twitch Developer Console](https://dev.twitch.tv/console/apps)
2. Click **Register Your Application**
3. Set a name for your bot
4. Add `http://localhost:17563` as an **OAuth Redirect URL**
5. Set category to **Chat Bot**
6. Create and note down your **Client ID** and **Client Secret**

### 2. Get a Gemini API Key

1. Go to [Google AI Studio](https://aistudio.google.com/apikey)
2. Create an API key
3. Copy the key

### 3. Configure Environment

```bash
# Copy the example env file
cp .env.example .env

# Edit .env with your credentials
```

### 4. Install Dependencies

```bash
pip install -r requirements.txt
```

### 5. Run the Bot

```bash
python main.py
```

On first run:

1. The bot tests the Gemini API connection
2. A browser window opens for Twitch OAuth login
3. Authorize the application
4. The bot joins your channel(s) and starts listening

## 💬 Usage

Mention the bot in Twitch chat:

```
@botname what's the meaning of life?
@botname tell me a joke
@botname mikä on paras pizza? (responds in Finnish)
```

## ⚙️ Configuration

| Variable                | Description                                     | Default           |
| ----------------------- | ----------------------------------------------- | ----------------- |
| `TWITCH_APP_ID`         | Twitch application Client ID                    | _Required_        |
| `TWITCH_APP_SECRET`     | Twitch application Client Secret                | _Required_        |
| `GEMINI_API_KEY`        | Google Gemini API key                           | _Required_        |
| `TARGET_CHANNELS`       | Channels to join (comma-separated)              | Bot's own channel |
| `USER_TIMEOUT_SECONDS`  | Cooldown between responses per user             | `5`               |
| `MAX_MESSAGES_PER_HOUR` | Max responses per user per hour (0 = unlimited) | `10`              |

### Example `.env`

```env
TWITCH_APP_ID=abc123
TWITCH_APP_SECRET=secret456
GEMINI_API_KEY=AIza...
TARGET_CHANNELS=vaarattu,somechannel
USER_TIMEOUT_SECONDS=5
MAX_MESSAGES_PER_HOUR=10
```

## 🔧 How It Works

1. **Authentication** - On startup, checks for saved tokens in `tokens.json`. Validates and refreshes automatically, or opens browser for OAuth if needed.

2. **Chat Connection** - Uses TwitchAPI's IRC-based Chat module to connect to configured channels.

3. **Message Handling** - Listens to all messages, responds only when:

   - Bot is mentioned with `@botname`
   - User is not on cooldown
   - User hasn't exceeded hourly limit

4. **AI Response** - Sends messages to Gemini with:

   - Conversation history for context
   - System prompt for personality and rules
   - User's name for personalized responses

5. **Language Detection** - Automatically detects and responds in the user's language.

## 📁 Files

| File               | Description                                        |
| ------------------ | -------------------------------------------------- |
| `main.py`          | Main bot code                                      |
| `.env`             | Environment variables (create from `.env.example`) |
| `.env.example`     | Example environment file                           |
| `tokens.json`      | Saved OAuth tokens (auto-generated)                |
| `requirements.txt` | Python dependencies                                |

## 📄 License

See [LICENSE](LICENSE) for details.
