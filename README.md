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

## Repository layout

```
├── .gitignore
├── play.sh                  # launcher: routes to --gui / --auto / advice mode
├── pyproject.toml           # package metadata + console entry points
├── LICENSE                  # MIT (c) rostyfi
├── README.md                # this file
├── src/lechonk/             # the application package
│   ├── __init__.py
│   ├── advisor.py           # terminal advice mode
│   ├── auto_play.py         # terminal auto-play mode
│   ├── gui.py               # GTK desktop app
│   ├── screenshot.py        # screen/window capture (portal + X11)
│   ├── mistral_client.py    # Mistral API client (vision)
│   ├── move_detector.py     # resolves a move to pixel coordinates on a screenshot
│   ├── move_parser.py       # parses AI responses into executable actions
│   ├── input_controller.py  # mouse/keyboard injection via X11 XTest
│   ├── overlay.py           # draws planned moves on screenshots
│   └── prompt_loader.py     # prompts.json loading/saving
├── tests/                   # test_move_detector.py (offline + live), test_chess_photo.py, test_simple.py
└── assets/                  # prompts.json (editable prompt library), chess_start.png,
                             # chess_start_large.png, Solitaire (example screenshots)
```

## Setup

1. **Clone the repo** and install the package (optional — `play.sh` works
   from a plain checkout too):
   ```bash
   git clone <repo-url> && cd le-chonk-plays-cards
   pip install .            # or: pip install -e .
   ```

2. **Install system dependencies** (not pip-installable):
   ```bash
   sudo apt install python3-dbus python3-gi libx11-6 libxtst6 xdg-desktop-portal
   ```
   Plus `Pillow` (installed automatically via pip).

3. **Get a Mistral API key**: https://console.mistral.ai/api-keys/

4. **Set your API key**:
   ```bash
   export MISTRAL_API_KEY="your-key-here"
   ```

5. **Make the launcher executable**:
   ```bash
   chmod +x play.sh
   ```

## Usage

### Desktop app (recommended — see everything in one window)
```bash
./play.sh --gui            # or, after pip install: lechonk-gui
```
A GTK window with:
- Window selector + live screenshot preview
- AI analysis (board state, recommended move, reasoning)
- Prompt library: switch between saved prompts, edit them, save new ones
  (stored in `assets/prompts.json`; the app reads and writes that file, so
  edits persist — override the location with `LECHONK_PROMPTS_FILE`)
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
./play.sh --prompt "Chess (default)"   # Pick a game prompt by title
./play.sh --save                   # Save the screenshot
./play.sh --quiet                  # Minimal output
```

### Terminal — Auto-play mode
```bash
./play.sh --auto --dry-run         # SAFE: shows what AI would do, no clicks
./play.sh --auto                   # Takes screenshot, AI plays one move
./play.sh --auto --loop 10         # Play 10 moves in a row
./play.sh --auto --prompt "Chess (default)"   # Pick a game prompt by title
./play.sh --auto --delay 2         # Wait 2s between moves (loop mode)
./play.sh --auto --reasoning max   # AI reasoning level: none, low (default), max
```

### Analyze an existing image
```bash
python3 -m lechonk.advisor --image /path/to/screenshot.png   # from a checkout
lechonk-advise --image shot.png --prompt "Chess (default)"   # after pip install
```

### Run the tests
```bash
python3 tests/test_move_detector.py                 # offline tests (no API key needed)
MISTRAL_API_KEY=sk-... python3 tests/test_move_detector.py   # also runs live API tests
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

## Requirements

- Ubuntu (or any Linux with Wayland + XWayland + xdg-desktop-portal)
- Python 3.10+ with `dbus-python` and `PyGObject` (system packages)
- `libX11` and `libXtst` (already installed on any desktop Linux)
- `Pillow` (pip)
- Mistral API key

## License

MIT — see [LICENSE](LICENSE).
