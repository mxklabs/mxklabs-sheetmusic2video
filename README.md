# sheetmusic2video

Render a piano MusicXML score as a calm, animated video. The score is engraved
as a grand staff in alternating top and bottom positions, with a cursor that
follows the music and notes that highlight as they sound.

## Requirements

- Python 3.10 or later
- The dependencies in [`requirements.txt`](requirements.txt)
- The system dependencies required by Manim on your platform

Install the Python dependencies with:

```bash
python -m pip install -r requirements.txt
```

## Usage

Render a MusicXML (`.musicxml` or `.xml`) or compressed MusicXML (`.mxl`) file:

```bash
python sheet2video.py score.musicxml
```

By default, the output is a 4K, 30 fps MP4 next to the input file, using the
same filename stem. Choose a different output path with `-o` / `--output`.

```bash
# Render a lower-resolution preview (720p, 15 fps)
python sheet2video.py score.musicxml --preview

# Render one PNG frame at 20 seconds
python sheet2video.py score.musicxml --still 20 -o frame.png
```

Additional options:

| Option | Description |
| --- | --- |
| `--fps FPS` | Set the frame rate |
| `--bpm BPM` | Force a constant tempo in quarter notes per minute |
| `--default-bpm BPM` | Tempo to use when the score has no tempo marking (default: 60) |
| `--speed MULTIPLIER` | Scale the score tempo (default: 1.0; for example, `0.8` is slower) |
| `--title TEXT` | Override the title displayed in the video |

Run `python sheet2video.py --help` to see all command-line options.

## Score support

Supported notation includes single- and multi-voice piano scores, chords,
ties, slurs, beams, flags, dots, accidentals, key and time signatures at the
start of the piece, dynamics, and tempo markings.

Grace notes, tuplet brackets, repeats, and pedal markings are not rendered.
