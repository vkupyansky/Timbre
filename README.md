[![Русский](https://img.shields.io/badge/%D0%A0%D1%83%D1%81%D1%81%D0%BA%D0%B8%D0%B9-181717?style=for-the-badge)](README.ru.md)

# Timbre

A Telegram bot that uses local models for Russian speech. Turns text into a voice message using the selected voice. Incoming voice messages are transcribed first, then synthesized using that voice and its settings.

## Features

- Personal voices and settings for each authorized user.
- Voice creation from 15–20 seconds of clean speech, with up to three samples of one person.
- Comparison of a new sample with the previous one before applying changes.
- Per-voice delivery, speed, instruction, and synthesis settings.
- A separate status for each job, with cancellation and delivery retry.
- Local Fish S2 Pro BF16 via MLX and Whisper-small via MPS.

## Installation and startup

Requires macOS on Apple Silicon with Metal/MPS and Docker Desktop. Prepare the native model worker once using the [installation guide](docs/architecture.md#первичная-установка) (in Russian). A Linux container cannot access the Apple GPU.

Create `.env` using [.env.example](.env.example):

| Parameter | Purpose |
| --- | --- |
| `TELEGRAM_BOT_TOKEN` | Bot token from BotFather |
| `TELEGRAM_ADMIN_USER_IDS` | Comma-separated positive numeric Telegram IDs |

The file must belong to the service user and have `0600` permissions. The application will not start without valid values. An internal key is generated automatically once.

```sh
docker compose up -d --build
```

Compose starts the application and SQLite. A macOS user service starts the models at login and waits for the application. The Mac must be on, the user must remain logged in, and Docker Desktop must be running.

Recreate the container after changing `.env`:

```sh
docker compose up -d --force-recreate
```

A prepared service can be placed separately from the source and started without `--build`. Do not delete the service directory containing models, keys, and data. See [service placement and migration](docs/architecture.md#исходники-и-установленный-сервис) (in Russian).

## Usage

1. Open a private chat with the bot and press `/start`.
2. Under "My voices" ("Мои голоса"), press "Add voice" ("Добавить голос"), send 15–20 seconds of clean Russian speech from one person, and enter a name.
3. Wait for profile preparation and listen to the sample. You can select and configure a ready voice on its card.
4. Send text or a voice message. The bot will reply with a standard voice message.

The only command, `/start`, moves the home panel to the bottom of the chat. All other controls use buttons within one panel. Each request has a separate short status message with cancellation and, when needed, delivery retry.

Voices, the selected profile, and settings are isolated per user. Only IDs in `.env` have access; the bot does not respond to others or download their files. Even administrators cannot access other users' voices through the interface.

Settings are stored with each voice. Changing samples first creates a separate candidate and preview; the previous set remains active until confirmation. Additional samples provide model context; model weights are not trained.

Add your own voice on a clean installation. Private recordings are not included in the repository. An optional local starter profile is copied into independent personal profiles; a copy deleted by its user is not recreated.

To suggest stress, capitalize the vowel: `замОк`, `зАмок`. It is best to write the letter `ё` explicitly. These are hints to the model; check the result by listening.

## Checks

Uses Python 3.12 and `uv`. Exact dependencies are pinned in [uv.lock](uv.lock) and [ml/uv.lock](ml/uv.lock); model revisions are pinned in the [manifests](models/).

```sh
uv run --frozen python -m scripts.check
```

This checks formatting, linting, strict types, tests with warnings treated as errors, dependencies, the build, and an isolated container run. No messages are sent to the production bot. Real model checks are described in the [testing guide](docs/testing.md) (in Russian).

## Limitations

Text and transcripts are limited to 2,000 characters. Incoming voice messages are limited to 120 seconds and 10 MiB; output is limited to 180 seconds and 20 MiB. Each job has a 15-minute limit; the models process jobs sequentially.

Pauses, pace, and emotions from the incoming recording are not copied. Voice similarity, pronunciation, and the effects of emotional settings require listening. A Telegram timeout may mean delivery has already succeeded; ambiguous delivery requires a user decision.

Models run locally; messages and audio pass through Telegram. Profiles and SQLite are stored in `.shift/`. Text and settings history is retained permanently; temporary recordings, results, and drafts are cleaned up within 24 hours. See the [data description](docs/data.md) (in Russian).

Payments are not connected. Authorized administrators use the bot without time deductions; accounting records for existing accounts are preserved during updates.

## Documentation

The following documents are in Russian:

- [Architecture, installation, and maintenance](docs/architecture.md)
- [Data and reports](docs/data.md)
- [Voice preparation and recognition](docs/profile-asr.md)
- [Testing](docs/testing.md)
- [Current state and limitations](STATUS.md)

## License

Built with Fish Audio. Model weights are not included in the repository and have their own licenses. Fish uses the [Fish Audio Research License](licenses/FISH-AUDIO-RESEARCH-LICENSE.md); commercial use requires separate permission. Whisper-small is distributed under Apache-2.0. Sources are listed in the model manifests, with attribution in [NOTICE](NOTICE). The model license does not license Timbre's source code.
