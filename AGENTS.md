# AGENTS.md

Default operating instructions for Copilot coding agents in this repository.

## Project Summary

`sheet2video.py` turns a piano MusicXML file into a calm, 4K Manim video: engraved grand staff, a sweeping cursor, notes that warm as they sound, and an 88-key keyboard that lights with the music.

## Tech Stack

- Python 3.14 in `.venv/` (use `.venv/bin/python`, not the system Python)
- Manim Community (`manim`), Verovio (engraving, timemap, MIDI pitches), NumPy
- ffmpeg on PATH (Homebrew); building pycairo needs `pkgconf`

## Architecture Rules

1. Engraving is done by Verovio only. Do not hand-draw notation.
2. Every colour and tuning value is a field on `Config` with a `_f(default, help)`; the CLI options are generated from it. Add new knobs there, never as hard-coded constants. Colour fields must also be listed in `COLOR_FIELDS`.
3. Defaults define the standard look. Keep them unless a visual fix requires changing one.
4. Fade notation by opacity (`System._paint`), never by mixing toward the background colour; the lit background would make notes read as dark silhouettes.
5. Z-order is by `z_index` (cursor 10, keyboard 3-7, lights and dust negative), not by add order.
6. Glows are dithered gradient images (`soft_light`), never stacked translucent discs, which show rings. Create per-note glow images lazily from a pool; copying an `ImageMobject` is slow.
7. Keep the look sophisticated and slow: muted palette, gentle motion, readable score first.
8. Parallel renders use separate Manim processes, not threads; automatic concurrency is capped at half the available CPU cores and reduced for short clips. Keep time-window rendering deterministic so independently rendered segments join cleanly.

## Testing Rules

- Never render a whole video to test. Use `--still SECONDS` (add `--preview` for 720p) and view the PNG.
- Check several times (for example 25 s, 60 s, 140 s) since layout and lighting vary.
- Stills skip manim's static-frame cache, so they can hide bugs that only show in video. The updater mobject (`driver`) must stay first by `z_index` (-100); anything ordered before it is frozen into a static image. After touching scene structure, render a few seconds of video of one system (not the whole piece) and compare frames.
- Check parallel output with a short `--start` / `--duration` clip, and verify frame count and seams against `--workers 1` before relying on a full render.
- A full 4K render takes about 12+ minutes; only run it when the user asks.

## Example Input and Commands

- Example: `~/Library/CloudStorage/Dropbox/Music/Scores/InProgress/MK16.2 - SFA/mk16.1 - 10.musicxml` (no tempo marking, so the default is 120 bpm).
- Still: `.venv/bin/python sheet2video.py <file> --still 60 --preview -o /tmp/x.png`
- Video: `.venv/bin/python sheet2video.py <file> -o output/name.mp4`

## Known Limitations

- Low bass notes with many ledger lines can approach the keyboard.
- Tie chains (from Verovio's MEI `<tie>` elements) are one continuous note: all heads light together and the key is pressed once, from the first onset to the last release.
- The video has no audio.

## Agent Response Expectations

1. State briefly what visual or behavioural change an edit makes.
2. Link changed files and say why.
3. Report what was verified (which stills) and what was not.
