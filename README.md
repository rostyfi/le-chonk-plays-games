# Le Chonk plays games

AI-powered assistant for ANY card game. Takes a screenshot, sends it to
**Mistral Large 4** (`mistral-large-4`), identifies the game, analyzes
the board, and can:

- **Advise** you on the best move (terminal or desktop app)
- **Auto-play** the move by simulating mouse/keyboard input
- **Show you everything** in the desktop app: screenshot preview, analysis,
  planned actions, and execution
- **Manage prompts** — a library of titled, saved prompts (`prompts.json`)
  you can switch between and edit in the desktop app

Works with: Solitaire (Klondike, Spider, FreeCell), Poker, Blackjack, Hearts,
Spades, Bridge, Uno, MTG Arena, Balatro, Slay the Spire, and more.

## Setup

1. **Get a Mistral API key**: https://console.mistral.ai/api-keys/

2. **Set your API key**:
   ```bash
   export MISTRAL_API_KEY="your-key-here"
   ```

3. **Make the launcher executable**:
   ```bash
   chmod +x play.sh
   ```

## Usage

### Desktop app (recommended — see everything in one window)
```bash
./play.sh --gui
```
A GTK window with:
- Window selector + live screenshot preview
- AI analysis (board state, recommended move, reasoning)
- Prompt library: switch between saved prompts, edit them, save new ones
  (stored in `prompts.json`)
- Buttons: **Capture & Advise**, **Dry Run**, **Execute Move**,
  **Re-execute Last Move** (replays the last received move without a new
  screenshot or API call)
- **Nudge X/Y** controls fine-tune the aim of executed moves (screen-pixel
  correction); the screenshot preview refreshes after each executed move
- Before executing, the game window is raised to the front and focused, so
  the injected input reaches it (a minimized/closed game is refused)
- Move detection is format-robust: the model reports the two drag endpoints
  as explicit pixel centers (`CENTERS:\nsource: (584, 562)\ndest: (584, 464)`),
  which are used exactly as written; it also reports bounding boxes in the
  format recommended by the Mistral cookbook's bbox notebook
  (mistral/image_understanding/bbox_getstarted): `{"box_2d": [xmin, ymin,
  xmax, ymax], "label": ...}` dicts, coordinates normalized 0-999, with a
  concrete example in the prompt; they are used for the overlay drawing and
  as a fallback when CENTERS is missing.
  If the boxes disagree with the drag endpoints (the model sometimes emits
  pixel coordinates where normalized ones are expected), the boxes are not
  drawn or used for verification — fixed squares at the drag endpoints are
  drawn instead. The chess prompt additionally forces a coordinate protocol:
  locate the board, state its pixel bounds, and compute square centers with
  an exact formula — so the coordinates stay correct even when the screenshot
  shows extra UI (clocks, panels) around the board
- The planned move is drawn on the screenshot preview (source/destination
  boxes or endpoint squares, drag arrow, click markers) before execution
- After executing, the board around the move is compared before/after; if
  nothing changed (an app without drag support), the move is retried once
  with click-click, and the result is logged ("Move verified" / "Board
  unchanged")

### Terminal — Advice mode
```bash
./play.sh                          # Screenshot + AI advice
./play.sh --game "Balatro"         # Specify the game
./play.sh --save                   # Save the screenshot
./play.sh --quiet                  # Minimal output
```

### Terminal — Auto-play mode
```bash
./play.sh --auto --dry-run         # SAFE: shows what AI would do, no clicks
./play.sh --auto                   # Takes screenshot, AI plays one move
./play.sh --auto --loop 10         # Play 10 moves in a row
./play.sh --auto --game "Poker"    # Specify the game
./play.sh --auto --delay 2         # Wait 2s between moves (loop mode)
```

### Analyze an existing image
```bash
python3 advisor.py --image /path/to/screenshot.png
python3 advisor.py --image shot.png --game "Poker" --prompt "Call or fold?"
```

## How it works

```
┌──────────────┐     ┌──────────────────┐     ┌─────────────────────┐
│  Screenshot   │────▶│  Mistral Large 4  │────▶│  Parse + Execute     │
│  (Wayland     │     │  (mistral-large-4)│     │  (X11 XTest input)   │
│   portal)     │     │   latest)         │     │                      │
└──────────────┘     └──────────────────┘     └─────────────────────┘
```

## Safety

- **`--dry-run` / Dry Run button** — always test first. Shows planned actions
  without clicking anything.
- The AI estimates pixel coordinates from the screenshot. Simple games work
  well; complex UIs may need a custom prompt describing the game.
- Stop any loop with **Ctrl+C** in terminal.

## Files

| File | Purpose |
|---|---|
| `play.sh` | Launcher — routes to `--gui`, `--auto`, or advice mode |
| `gui.py` | Desktop app (GTK): window selector, prompt library, auto-move |
| `advisor.py` | Terminal advice mode |
| `auto_play.py` | Terminal auto-play mode |
| `screenshot.py` | Screenshot capture via xdg-desktop-portal |
| `mistral_client.py` | Mistral API client (vision models) + prompt library |
| `move_parser.py` | Parses AI response into executable actions |
| `move_detector.py` | Detects a game move on a board image via the Mistral endpoint (cookbook `box_2d` format), resolves it to from/to squares + pixel boxes, and can draw the move (source/dest boxes + drag arrow) on the picture |
| `test_move_detector.py` | Tests for `move_detector` (offline parser + stubbed end-to-end; live API test when `MISTRAL_API_KEY` is set) |
| `input_controller.py` | Mouse/keyboard injection via X11 XTest (ctypes) |
| `prompts.json` | Saved prompt library (titles + texts) |
| `chess_start.png` | Test fixture — real chess screenshot (Stockfish, start position) |
| `test_chess_photo.py` | Live Mistral API test — sends the photo to the API (chess prompt, actions format) and prints the raw reply; needs `MISTRAL_API_KEY` |

## Requirements

- Ubuntu (or any Linux with Wayland + XWayland + xdg-desktop-portal)
- Python 3 with `dbus-python` and `PyGObject`
- `libX11` and `libXtst` (already installed on any desktop Linux)
- Mistral API key
