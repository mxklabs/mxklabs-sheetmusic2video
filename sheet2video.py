"""Render a piano MusicXML score as a calm, 4K Manim video.

The score is engraved as a grand staff in two alternating slots (top / bottom).
A soft cursor sweeps the active system; notes warm up as they sound and then
settle into a quiet grey-blue.

    python sheet2video.py score.musicxml -o score.mp4            # 4K
    python sheet2video.py score.musicxml --preview               # 720p, fast
    python sheet2video.py score.musicxml --still 20 -o frame.png # one frame

Supported: single/multi-voice piano, chords, ties, slurs, beams, flags, dots,
accidentals, key/time signature (start of piece), dynamics, tempo markings,
compressed .mxl. Not rendered: grace notes, tuplet brackets, repeats, pedal.
"""

from __future__ import annotations

import argparse
import math
import sys
import zipfile
import xml.etree.ElementTree as ET
from dataclasses import dataclass
from pathlib import Path

import numpy as np
from manim import (
    BraceBetweenPoints,
    Circle,
    Ellipse,
    Line,
    LEFT,
    ManimColor,
    Mobject,
    Polygon,
    Scene,
    Text,
    VGroup,
    VMobject,
    tempconfig,
)

# ---------------------------------------------------------------- palette ---
BG = "#0B0E12"
UNPLAYED = "#D8D2C5"
ACTIVE = "#E6C58C"
PLAYED = "#8892A2"
STAFF = "#4B525D"
MARK = "#8C919A"
REST = "#7A808A"
TIE = "#757C87"
CURSOR = "#C9AE7B"
TITLE = "#D8D2C5"
SUBTITLE = "#7F8591"

FONT_SYMBOLS = "Apple Symbols"
FONT_TEXT = "Palatino"

STEP_INDEX = {"C": 0, "D": 1, "E": 2, "F": 3, "G": 4, "A": 5, "B": 6}
ACC_GLYPHS = {
    "sharp": "♯",
    "flat": "♭",
    "natural": "♮",
    "double-sharp": "×",
    "flat-flat": "♭♭",
    "natural-sharp": "♯",
    "natural-flat": "♭",
}
FLAGS = {"eighth": 1, "16th": 2, "32nd": 3, "64th": 4}
SHARP_POS = [8, 5, 9, 6, 3, 7, 4]  # treble-clef positions, 0 = bottom line
FLAT_POS = [4, 7, 3, 6, 2, 5, 1]


# ------------------------------------------------------------- data model ---
@dataclass
class Note:
    staff: int
    voice: str
    start: float
    dur: float
    idx: int  # diatonic index, octave * 7 + step
    pos: int  # staff position, 0 = bottom line
    alter: int
    acc: str | None
    type: str
    dots: int
    stem: str | None
    beams: dict[int, str]
    tie_start: bool
    tie_stop: bool
    measure: int
    slur_placement: str | None = None


@dataclass
class Rest:
    staff: int
    voice: str
    start: float
    dur: float
    type: str
    dots: int
    whole_measure: bool
    pos: int | None
    measure: int


@dataclass
class Measure:
    index: int
    start: float
    length: float
    fifths: int
    time: tuple[int, int]
    clefs: dict[int, str]  # staff -> "G" | "F" | "C"
    last_onset: float = 0.0


@dataclass
class Link:
    a: Note
    b: Note
    kind: str  # "tie" | "slur"
    placement: str | None


@dataclass
class Score:
    title: str
    composer: str
    measures: list[Measure]
    notes: list[Note]
    rests: list[Rest]
    links: list[Link]
    dynamics: list[tuple[float, str]]
    tempos: list[tuple[float, float]]
    staves: int


def _open_xml(path: Path) -> ET.Element:
    if zipfile.is_zipfile(path):
        with zipfile.ZipFile(path) as z:
            container = ET.fromstring(z.read("META-INF/container.xml"))
            rootfile = container.find(".//rootfile").attrib["full-path"]
            return ET.fromstring(z.read(rootfile))
    return ET.parse(path).getroot()


def _bottom_idx(sign: str, line: int) -> int:
    ref = {"G": 4 * 7 + 4, "F": 3 * 7 + 3, "C": 4 * 7}[sign]
    return ref - 2 * (line - 1)


def parse_musicxml(path: Path) -> Score:
    root = _open_xml(path)
    if root.tag != "score-partwise":
        raise ValueError("Only score-partwise MusicXML is supported")

    title = (root.findtext("work/work-title") or root.findtext("movement-title") or path.stem).strip()
    composer = ""
    for c in root.findall("identification/creator"):
        if c.get("type") == "composer" and c.text:
            composer = c.text.strip()

    notes: list[Note] = []
    rests: list[Rest] = []
    links: list[Link] = []
    dynamics: list[tuple[float, str]] = []
    tempos: list[tuple[float, float]] = []
    measures: list[Measure] = []
    max_staff = 1

    for part_no, part in enumerate(root.findall("part")):
        divisions = 1.0
        bottoms: dict[int, int] = {}
        signs: dict[int, str] = {}
        fifths = 0
        time = (4, 4)
        open_slurs: dict[str, Note] = {}
        pending_notes: list[Note] = []
        cursor_beat = 0.0

        for m_i, m in enumerate(part.findall("measure")):
            if part_no == 0:
                mstart = cursor_beat
            else:
                mstart = measures[m_i].start
            cursor = 0.0  # in divisions
            last_start = 0.0
            max_cursor = 0.0
            last_onset = 0.0

            for el in m:
                tag = el.tag
                if tag == "attributes":
                    d = el.findtext("divisions")
                    if d:
                        divisions = float(d)
                    f = el.findtext("key/fifths")
                    if f is not None:
                        fifths = int(f)
                    if el.find("time") is not None:
                        time = (int(el.findtext("time/beats")), int(el.findtext("time/beat-type")))
                    for clef in el.findall("clef"):
                        s = int(clef.get("number", "1")) + (part_no if part_no and len(root.findall("part")) > 1 else 0)
                        sign = clef.findtext("sign", "G")
                        sign = sign if sign in ("G", "F", "C") else "G"
                        line = int(clef.findtext("line", {"G": "2", "F": "4", "C": "3"}[sign]))
                        bottoms[s] = _bottom_idx(sign, line)
                        signs[s] = sign
                elif tag == "backup":
                    cursor -= float(el.findtext("duration"))
                elif tag == "forward":
                    cursor += float(el.findtext("duration"))
                    max_cursor = max(max_cursor, cursor)
                elif tag == "sound" and el.get("tempo"):
                    tempos.append((mstart + cursor / divisions, float(el.get("tempo"))))
                elif tag == "direction":
                    snd = el.find("sound")
                    if snd is not None and snd.get("tempo"):
                        tempos.append((mstart + cursor / divisions, float(snd.get("tempo"))))
                    dyn = el.find("direction-type/dynamics")
                    if dyn is not None and len(dyn):
                        dynamics.append((mstart + cursor / divisions, dyn[0].tag))
                elif tag == "note":
                    if el.find("grace") is not None:
                        continue
                    dur = float(el.findtext("duration", "0"))
                    is_chord = el.find("chord") is not None
                    start = last_start if is_chord else cursor
                    staff = int(el.findtext("staff", "1"))
                    if part_no and len(root.findall("part")) > 1:
                        staff += part_no
                    max_staff = max(max_staff, staff)
                    voice = el.findtext("voice", "1")
                    ntype = el.findtext("type", "")
                    dots = len(el.findall("dot"))
                    start_b = mstart + start / divisions
                    dur_b = dur / divisions
                    last_onset = max(last_onset, start / divisions)

                    if el.find("rest") is not None:
                        if el.get("print-object") != "no":
                            r = el.find("rest")
                            pos = None
                            if r.findtext("display-step"):
                                idx = int(r.findtext("display-octave")) * 7 + STEP_INDEX[r.findtext("display-step")]
                                pos = idx - bottoms.get(staff, _bottom_idx("G", 2))
                            rests.append(Rest(staff, voice, start_b, dur_b, ntype, dots,
                                              r.get("measure") == "yes", pos, m_i))
                    else:
                        p = el.find("pitch")
                        if p is not None:
                            idx = int(p.findtext("octave")) * 7 + STEP_INDEX[p.findtext("step")]
                            ties = {t.get("type") for t in el.findall("tie")}
                            slur_place = None
                            n = Note(
                                staff, voice, start_b, dur_b, idx,
                                idx - bottoms.get(staff, _bottom_idx("G", 2) if staff == 1 else _bottom_idx("F", 4)),
                                int(float(p.findtext("alter", "0"))),
                                el.findtext("accidental"), ntype, dots, el.findtext("stem"),
                                {int(b.get("number", "1")): (b.text or "") for b in el.findall("beam")},
                                "start" in ties, "stop" in ties, m_i,
                            )
                            notes.append(n)
                            for s in el.findall("notations/slur"):
                                key = f"{part_no}:{s.get('number', '1')}"
                                if s.get("type") == "start":
                                    open_slurs[key] = n
                                    n.slur_placement = s.get("placement")
                                elif s.get("type") == "stop" and key in open_slurs:
                                    a = open_slurs.pop(key)
                                    links.append(Link(a, n, "slur", a.slur_placement))
                            pending_notes.append(n)
                    if not is_chord:
                        last_start = cursor
                        cursor += dur
                    max_cursor = max(max_cursor, cursor)

            length = max_cursor / divisions if max_cursor else time[0] * 4 / time[1]
            if part_no == 0:
                measures.append(Measure(m_i, mstart, length, fifths, time, dict(signs), last_onset))
                cursor_beat += length
            else:
                measures[m_i].last_onset = max(measures[m_i].last_onset, last_onset)

    # Ties: link each tie-start with the note it continues into.
    for a in notes:
        if not a.tie_start:
            continue
        for b in notes:
            if (b.tie_stop and b.staff == a.staff and b.idx == a.idx and b.alter == a.alter
                    and abs(b.start - (a.start + a.dur)) < 1e-6):
                links.append(Link(a, b, "tie", None))
                break

    if not measures:
        raise ValueError("No measures found")
    tempos.sort()
    return Score(title, composer, measures, notes, rests, links, dynamics, tempos, 2 if max_staff > 1 else 1)


# ------------------------------------------------------------------ tempo ---
class TempoMap:
    """Piecewise-constant tempo; converts between quarter-note beats and seconds."""

    def __init__(self, changes: list[tuple[float, float]], default_bpm: float, speed: float, override: float | None):
        if override:
            changes = [(0.0, override)]
        elif not changes or changes[0][0] > 0:
            changes = [(0.0, changes[0][1] if changes else default_bpm)] + list(changes)
        self.beats = [c[0] for c in changes]
        self.spb = [60.0 / (c[1] * speed) for c in changes]
        self.secs = [0.0]
        for i in range(1, len(changes)):
            self.secs.append(self.secs[-1] + (self.beats[i] - self.beats[i - 1]) * self.spb[i - 1])

    def seconds(self, beat: float) -> float:
        beat = max(beat, 0.0)
        i = max(j for j, b in enumerate(self.beats) if b <= beat)
        return self.secs[i] + (beat - self.beats[i]) * self.spb[i]

    def beat(self, sec: float) -> float:
        sec = max(sec, 0.0)
        i = max(j for j, s in enumerate(self.secs) if s <= sec)
        return self.beats[i] + (sec - self.secs[i]) / self.spb[i]


# ----------------------------------------------------------------- config ---
@dataclass
class Config:
    space: float = 0.18  # staff space in scene units
    slot_width: float = 12.6
    slot_y: float = 2.0
    u_min: float = 1.15  # scene units per quarter note
    u_max: float = 1.9
    lead: float = 7.0  # seconds before the first beat
    tail: float = 3.0


# ------------------------------------------------------------- colour mix ---
def _rgb(c: str) -> np.ndarray:
    return np.array(ManimColor(c).to_rgb())


_BG, _UN, _AC, _PL = _rgb(BG), _rgb(UNPLAYED), _rgb(ACTIVE), _rgb(PLAYED)


def _col(rgb: np.ndarray) -> ManimColor:
    return ManimColor.from_rgb(tuple(float(v) for v in np.clip(rgb, 0, 1)))


def smooth(x: float) -> float:
    x = min(max(x, 0.0), 1.0)
    return x * x * (3 - 2 * x)


def ink_rgb(t: float, t_on: float, t_off: float) -> np.ndarray:
    if t < t_on:
        return _UN
    ramp = 0.22
    a_on = smooth((t - t_on) / ramp)
    if t < t_off:
        return _UN + (_AC - _UN) * a_on
    a_off = smooth((t_off - t_on) / ramp)
    start = _UN + (_AC - _UN) * a_off
    return start + (_PL - start) * smooth((t - t_off) / 1.8)


def glow_env(t: float, t_on: float, t_off: float) -> float:
    if t < t_on:
        return 0.0
    if t < t_off:
        return smooth((t - t_on) / 0.2)
    return smooth((t_off - t_on) / 0.2) * (1 - smooth((t - t_off) / 1.6))


# ----------------------------------------------------------- engraving ---
@dataclass
class Ink:
    mob: Mobject
    t_on: float
    t_off: float


@dataclass
class Event:
    staff: int
    voice: str
    start: float
    dur: float
    notes: list[Note]
    type: str
    dots: int
    stem: str | None
    beams: dict[int, str]
    dir: int = 1
    x: float = 0.0
    t_on: float = 0.0
    t_off: float = 0.0
    info: tuple[float, float, float, float] = (0.0, 0.0, 0.0, 0.0)  # stem x, start y, extreme head y, tip y


@dataclass
class SystemPlan:
    measures: list[Measure]
    b0: float
    b1: float
    header_w: float
    u: float
    first: bool
    fifths: int
    time: tuple[int, int]
    clefs: dict[int, str]


class System:
    """Visual objects and timing for one system occupying one slot."""

    def __init__(self, plan: SystemPlan, score: Score, tempo: TempoMap, cfg: Config, ox: float, oy: float, lead: float):
        self.plan, self.score, self.tempo, self.cfg = plan, score, tempo, cfg
        self.ox, self.oy, self.lead = ox, oy, lead
        self.S = cfg.space
        self.hw = 0.62 * self.S
        self.hh = 0.46 * self.S
        self.x0 = ox + plan.header_w
        self.pad_l = 0.4
        self.u = plan.u
        self.statics: list[tuple[Mobject, str]] = []
        self.inks: list[Ink] = []
        self.glow_layer = VGroup()
        self.static_layer = VGroup()
        self.ink_layer = VGroup()
        self.group = VGroup(self.glow_layer, self.static_layer, self.ink_layer)
        self.glows: list[tuple[VGroup, float, float]] = []
        self.vis = -1.0
        self.pending: list[Ink] = []
        self.active: list[Ink] = []
        self.active_glows: list[tuple[VGroup, float, float]] = []
        self.pending_glows: list[tuple[VGroup, float, float]] = []
        self.staff_ids = [1, 2] if score.staves == 2 else [1]
        self.bases = {1: oy + 2.5 * self.S, 2: oy - 6.5 * self.S} if score.staves == 2 else {1: oy - 2 * self.S}
        self.barline_x: dict[int, float] = {}
        self._build()
        self.pending = sorted(self.inks, key=lambda i: i.t_on)
        self.pending_glows.sort(key=lambda g: g[1])
        self.static_layer.set_z_index(0)

    # -- helpers
    def X(self, beat: float) -> float:
        return self.x0 + self.pad_l + (beat - self.plan.b0) * self.u

    def Y(self, staff: int, pos: float) -> float:
        return self.bases[staff] + pos * self.S / 2

    def sec(self, beat: float) -> float:
        return self.lead + self.tempo.seconds(beat)

    def _static(self, mob: Mobject, color: str) -> None:
        mob.set_color(color)
        self.statics.append((mob, color))
        self.static_layer.add(mob)

    def _ink(self, mob: Mobject, t_on: float, t_off: float) -> None:
        mob.set_color(UNPLAYED)
        self.inks.append(Ink(mob, t_on, t_off))
        self.ink_layer.add(mob)

    _glyph_cache: dict[tuple[str, str], Text] = {}

    def _glyph(self, ch: str, height: float, font: str = FONT_SYMBOLS) -> Text:
        key = (ch, font)
        if key not in System._glyph_cache:
            System._glyph_cache[key] = Text(ch, font=font)
        g = System._glyph_cache[key].copy()
        g.scale_to_fit_height(height)
        return g

    # -- build
    def _build(self) -> None:
        p, S = self.plan, self.S
        last = p.measures[-1]
        d_last = self._barline_offset(last)
        self.x_end = self.X(p.b1) - d_last
        self._staves()
        self._header()
        events = self._events()
        self._beam_groups(events)
        for ev in events:
            self._render_event(ev)
        self._rests()
        self._barlines()
        self._dynamics()
        self._links()

    def _barline_offset(self, m: Measure) -> float:
        d = max(m.start + m.length - (m.start + m.last_onset), 0.25)
        return min(d * self.u / 2, 0.8)

    def _staves(self) -> None:
        x_a = self.ox + 0.35
        for s in self.staff_ids:
            for k in range(5):
                y = self.Y(s, 2 * k)
                self._static(Line([x_a, y, 0], [self.x_end, y, 0], stroke_width=2.0), STAFF)
        if self.score.staves == 2:
            top, bot = self.Y(1, 8), self.Y(2, 0)
            self._static(Line([x_a, top, 0], [x_a, bot, 0], stroke_width=2.2), MARK)
            self._static(Line([self.x_end, top, 0], [self.x_end, bot, 0], stroke_width=2.2), MARK)
            brace = BraceBetweenPoints([x_a - 0.08, bot, 0], [x_a - 0.08, top, 0], direction=LEFT)
            brace.set_fill(opacity=1).set_stroke(width=0)
            brace.stretch_to_fit_width(0.16)
            brace.move_to([x_a - 0.08 - 0.08, (top + bot) / 2, 0])
            self._static(brace, MARK)

    def _header(self) -> None:
        p, S = self.plan, self.S
        x = self.ox + 0.35 + 0.18
        for s in self.staff_ids:
            sign = p.clefs.get(s, "G" if s == 1 else "F")
            if sign == "F":
                g = self._glyph("𝄢", 3.4 * S)
                g.move_to([x + 0.2, self.Y(s, 5.6), 0])
            else:
                g = self._glyph("𝄞", 7.6 * S)
                g.move_to([x + 0.2, self.Y(s, 3.6), 0])
            self._static(g, MARK)
        kx = x + 0.7
        n = abs(p.fifths)
        for s in self.staff_ids:
            sign = p.clefs.get(s, "G" if s == 1 else "F")
            shift = -2 if sign == "F" else 0
            table = SHARP_POS if p.fifths > 0 else FLAT_POS
            for i in range(n):
                pos = table[i] + shift
                if pos < 0:
                    pos += 7
                ch = "♯" if p.fifths > 0 else "♭"
                g = self._glyph(ch, 2.7 * S if ch == "♯" else 2.3 * S)
                g.move_to([kx + i * 1.05 * S, self.Y(s, pos) + (0.0 if ch == "♯" else 0.3 * S), 0])
                self._static(g, MARK)
        if p.first:
            tx = kx + n * 1.05 * S + 0.3
            for s in self.staff_ids:
                for txt, pos in ((str(p.time[0]), 6), (str(p.time[1]), 2)):
                    g = self._glyph(txt, 2.0 * S, FONT_TEXT)
                    g.move_to([tx, self.Y(s, pos), 0])
                    self._static(g, MARK)

    def _events(self) -> list[Event]:
        p = self.plan
        idxs = {m.index for m in p.measures}
        groups: dict[tuple[int, str, float], list[Note]] = {}
        for n in self.score.notes:
            if n.measure in idxs:
                groups.setdefault((n.staff, n.voice, round(n.start, 6)), []).append(n)
        events: list[Event] = []
        for (staff, voice, start), ns in groups.items():
            ns.sort(key=lambda n: n.pos)
            f = ns[0]
            ev = Event(staff, voice, start, f.dur, ns, f.type, f.dots, f.stem, f.beams)
            ev.x = self.X(start)
            ev.t_on = self.sec(start)
            ev.t_off = max(self.sec(start + f.dur), ev.t_on + 0.5)
            if f.stem in ("up", "down"):
                ev.dir = 1 if f.stem == "up" else -1
            else:
                mean = sum(n.pos for n in ns) / len(ns)
                ev.dir = 1 if mean < 4 else -1
            events.append(ev)
        events.sort(key=lambda e: (e.staff, e.voice, e.start))
        return events

    def _beam_groups(self, events: list[Event]) -> None:
        self.beam_groups: list[list[Event]] = []
        self.beamed: set[int] = set()
        cur: list[Event] = []
        key = None
        for ev in events:
            k = (ev.staff, ev.voice)
            state = ev.beams.get(1)
            if k != key:
                cur, key = [], k
            if state == "begin":
                cur = [ev]
            elif state in ("continue", "end") and cur:
                cur.append(ev)
                if state == "end":
                    if len(cur) > 1:
                        self.beam_groups.append(cur)
                        self.beamed.update(id(e) for e in cur)
                    cur = []
        for g in self.beam_groups:
            d = g[0].dir
            for e in g:
                e.dir = d

    def _heads(self, ev: Event) -> dict[int, float]:
        S, hw = self.S, self.hw
        offsets: dict[int, float] = {}
        ns = ev.notes
        order = ns if ev.dir == 1 else list(reversed(ns))
        prev = None
        for n in order:
            off = 0.0
            if prev is not None and abs(n.pos - prev.pos) == 1 and offsets[id(prev)] == 0.0:
                off = 2 * hw * 0.95 * ev.dir
            offsets[id(n)] = off
            prev = n
        return offsets

    def _render_event(self, ev: Event) -> None:
        S, hw, hh = self.S, self.hw, self.hh
        offsets = self._heads(ev)
        solid = ev.type not in ("whole", "half")
        accs: list[Note] = []
        for n in ev.notes:
            x = ev.x + offsets[id(n)]
            y = self.Y(ev.staff, n.pos)
            head = Ellipse(width=2 * hw, height=2 * hh).rotate(math.radians(20))
            head.move_to([x, y, 0])
            if solid:
                head.set_fill(opacity=1).set_stroke(width=0.5)
            else:
                head.set_fill(opacity=0).set_stroke(width=2.8)
            self._ink(head, ev.t_on, ev.t_off)
            glow = VGroup(*[Circle(radius=r * S).move_to([x, y, 0]) for r in (1.1, 1.8, 2.6)])
            for c in glow:
                c.set_fill(ACTIVE, opacity=0).set_stroke(width=0)
            self.glows.append((glow, ev.t_on, ev.t_off))
            self.pending_glows.append((glow, ev.t_on, ev.t_off))
            self._ledgers(ev, n, x)
            if n.type != "whole":
                for k in range(n.dots):
                    dy = 0.5 * S if n.pos % 2 == 0 else 0.0
                    dot = Circle(radius=0.2 * S).set_fill(opacity=1).set_stroke(width=0)
                    dot.move_to([x + hw + (0.55 + 0.5 * k) * S, y + dy, 0])
                    self._ink(dot, ev.t_on, ev.t_off)
            if n.acc in ACC_GLYPHS:
                accs.append(n)
        self._accidentals(ev, accs, offsets)
        if ev.type == "whole" or ev.type == "":
            return
        # Stem
        d = ev.dir
        ys = [self.Y(ev.staff, n.pos) for n in ev.notes]
        sx = ev.x + d * hw * 0.93
        start_y = min(ys) if d == 1 else max(ys)
        ext_y = max(ys) if d == 1 else min(ys)
        tip = ext_y + d * 3.5 * S
        mid = self.Y(ev.staff, 4)
        if (d == 1 and tip < mid) or (d == -1 and tip > mid):
            tip = mid
        ev_info = (sx, start_y, ext_y, tip)
        if id(ev) in self.beamed:
            ev.info = ev_info
            return
        stem = Line([sx, start_y, 0], [sx, tip, 0], stroke_width=2.4)
        self._ink(stem, ev.t_on, ev.t_off)
        for i in range(FLAGS.get(ev.type, 0)):
            fy = tip - d * i * 0.95 * S
            pts = [(0, 0), (0.9, -0.5), (1.25, -1.5), (0.75, -2.7)]
            cp = [[sx + px * S, fy + d * py * S, 0] for px, py in pts]
            flag = VMobject(stroke_width=3.4)
            flag.set_points(np.array(cp, dtype=float))
            flag.set_fill(opacity=0)
            self._ink(flag, ev.t_on, ev.t_off)

    def _ledgers(self, ev: Event, n: Note, x: float) -> None:
        S = self.S
        lows = range(-2, n.pos - 1, -2) if n.pos <= -2 else []
        highs = range(10, n.pos + 1, 2) if n.pos >= 10 else []
        for p in list(lows) + list(highs):
            y = self.Y(ev.staff, p)
            ln = Line([x - 1.7 * self.hw, y, 0], [x + 1.7 * self.hw, y, 0], stroke_width=2.6)
            self._ink(ln, ev.t_on, ev.t_off)

    def _accidentals(self, ev: Event, accs: list[Note], offsets: dict[int, float]) -> None:
        S = self.S
        cols: list[int] = []  # last pos per column
        for n in sorted(accs, key=lambda n: -n.pos):
            col = next((i for i, lp in enumerate(cols) if lp - n.pos >= 6), None)
            if col is None:
                cols.append(n.pos)
                col = len(cols) - 1
            else:
                cols[col] = n.pos
            ch = ACC_GLYPHS[n.acc]
            h = 2.7 * S if ch in ("♯", "♮", "×") else 2.2 * S
            g = self._glyph(ch, h)
            left = min(offsets[id(m)] for m in ev.notes)
            x = ev.x + left - self.hw - (0.9 + 1.0 * col) * S
            y = self.Y(ev.staff, n.pos) + (0.3 * S if ch.startswith("♭") else 0.0)
            g.move_to([x, y, 0])
            self._ink(g, ev.t_on, ev.t_off)

    def _beam_poly(self, xa: float, ya: float, xb: float, yb: float, th: float) -> Polygon:
        poly = Polygon([xa, ya, 0], [xb, yb, 0], [xb, yb - th, 0], [xa, ya - th, 0])
        poly.set_fill(opacity=1).set_stroke(width=0)
        return poly

    def _render_beams(self) -> None:
        S = self.S
        for grp in self.beam_groups:
            d = grp[0].dir
            infos = [e.info for e in grp]
            xs = [i[0] for i in infos]
            tips = [i[3] for i in infos]
            slope_y = tips[-1] - tips[0]
            lim = 1.5 * S
            ya = tips[0]
            yb = tips[0] + max(-lim, min(lim, slope_y))
            k = (yb - ya) / (xs[-1] - xs[0]) if xs[-1] != xs[0] else 0.0
            line = lambda x: ya + k * (x - xs[0])  # noqa: E731
            need = max(d * ((i[2] + d * 2.9 * S) - line(x)) for i, x in zip(infos, xs))
            if need > 0:
                ya += d * need
            line = lambda x: ya + k * (x - xs[0])  # noqa: E731
            th = 0.5 * S
            t_on, t_off = grp[0].t_on, max(e.t_off for e in grp)
            pitch = 0.78 * S
            for ev, (sx, sy, ey, tp) in zip(grp, infos):
                end = line(sx) - d * th / 2
                stem = Line([sx, sy, 0], [sx, end, 0], stroke_width=2.4)
                self._ink(stem, ev.t_on, ev.t_off)
            maxlevel = max(max(e.beams) for e in grp)
            hw = 0.012
            for L in range(1, maxlevel + 1):
                off = -d * (L - 1) * pitch
                seg_start = None
                for i, ev in enumerate(grp):
                    st = ev.beams.get(L)
                    sx = xs[i]
                    if st == "begin":
                        seg_start = i
                    elif st == "end" and seg_start is not None:
                        self._add_beam(xs[seg_start] - hw, sx + hw, line, off, th, t_on, t_off)
                        seg_start = None
                    elif st == "forward hook":
                        self._add_beam(sx - hw, sx + 0.3, line, off, th, ev.t_on, ev.t_off)
                    elif st == "backward hook":
                        self._add_beam(sx - 0.3, sx + hw, line, off, th, ev.t_on, ev.t_off)

    def _add_beam(self, a: float, b: float, line, off: float, th: float, t_on: float, t_off: float) -> None:
        poly = self._beam_poly(a, line(a) + off, b, line(b) + off, th)
        self._ink(poly, t_on, t_off)

    def _rests(self) -> None:
        S = self.S
        idxs = {m.index: m for m in self.plan.measures}
        for r in self.score.rests:
            if r.measure not in idxs:
                continue
            m = idxs[r.measure]
            if r.whole_measure or r.type == "":
                xa = self.X(m.start)
                xb = self.X(m.start + m.length) - self._barline_offset(m)
                x = (xa + xb) / 2
                self._rest_shape(r.staff, "whole", x, r.pos)
            else:
                self._rest_shape(r.staff, r.type, self.X(r.start), r.pos)

    def _rest_shape(self, staff: int, kind: str, x: float, pos: int | None) -> None:
        S = self.S
        if kind == "whole":
            y = self.Y(staff, pos if pos is not None else 6)
            r = self._rect(x, y - 0.25 * S, 1.5 * S, 0.5 * S)
        elif kind == "half":
            y = self.Y(staff, pos if pos is not None else 4)
            r = self._rect(x, y + 0.25 * S, 1.5 * S, 0.5 * S)
        elif kind == "quarter":
            y = self.Y(staff, pos if pos is not None else 4)
            pts = [(-0.3, 1.6), (0.4, 0.7), (-0.3, -0.1), (0.35, -0.9), (-0.05, -1.7)]
            r = VMobject(stroke_width=4.2)
            r.set_points_smoothly([[x + px * S, y + py * S, 0] for px, py in pts])
            r.set_fill(opacity=0)
        else:
            n = FLAGS.get(kind, 1)
            y = self.Y(staff, pos if pos is not None else 4)
            r = VGroup(Line([x + 0.35 * S, y + 1.0 * S, 0], [x - 0.3 * S, y - 1.0 * S - (n - 1) * 0.8 * S, 0],
                            stroke_width=3.0))
            for i in range(n):
                dot = Circle(radius=0.3 * S).set_fill(opacity=1).set_stroke(width=0)
                dot.move_to([x - 0.2 * S + 0.0 - i * 0.28 * S + 0.3 * S, y + 0.75 * S - i * 0.8 * S, 0])
                r.add(dot)
        self._static(r, REST)

    def _rect(self, cx: float, cy: float, w: float, h: float) -> Polygon:
        p = Polygon([cx - w / 2, cy - h / 2, 0], [cx + w / 2, cy - h / 2, 0],
                    [cx + w / 2, cy + h / 2, 0], [cx - w / 2, cy + h / 2, 0])
        p.set_fill(opacity=1).set_stroke(width=0)
        return p

    def _barlines(self) -> None:
        top, bot = self.Y(1, 8), self.Y(self.staff_ids[-1], 0)
        ms = self.plan.measures
        for i, m in enumerate(ms):
            x = self.X(m.start + m.length) - self._barline_offset(m)
            if i == len(ms) - 1:
                x = self.x_end
                if m is self.score.measures[-1]:
                    for dx, w in ((-0.09, 2.2), (0.0, 7.0)):
                        for s in self.staff_ids:
                            self._static(Line([x + dx, self.Y(s, 8), 0], [x + dx, self.Y(s, 0), 0], stroke_width=w), MARK)
                    continue
            for s in self.staff_ids:
                self._static(Line([x, self.Y(s, 8), 0], [x, self.Y(s, 0), 0], stroke_width=2.2), MARK)

    def _dynamics(self) -> None:
        b0, b1 = self.plan.b0, self.plan.b1
        for beat, name in self.score.dynamics:
            if b0 <= beat < b1:
                t = Text(name, font=FONT_TEXT, slant="ITALIC", weight="BOLD")
                t.scale_to_fit_height(1.6 * self.S if len(name) > 1 else 1.9 * self.S)
                mid = (self.Y(1, 0) + self.Y(self.staff_ids[-1], 8)) / 2 if self.score.staves == 2 else self.Y(1, -3)
                t.move_to([self.X(beat), mid, 0])
                self._static(t, "#9A948A")

    def _links(self) -> None:
        S = self.S
        idxs = {m.index for m in self.plan.measures}
        for ln in self.score.links:
            ina, inb = ln.a.measure in idxs, ln.b.measure in idxs
            if not (ina or inb):
                continue
            ya = self.Y(ln.a.staff, ln.a.pos)
            yb = self.Y(ln.b.staff, ln.b.pos)
            up = (ln.placement == "above") if ln.placement else (ln.a.stem == "down" if ln.kind == "tie" else True)
            if ln.kind == "tie" and ln.placement is None:
                up = ln.a.stem == "down" or (ln.a.stem is None and ln.a.pos >= 4)
                chord = [n.pos for n in self.score.notes
                         if n.staff == ln.a.staff and n.voice == ln.a.voice and abs(n.start - ln.a.start) < 1e-6]
                if len(chord) > 1 and ln.a.pos in (max(chord), min(chord)):
                    up = ln.a.pos == max(chord)
            sgn = 1 if up else -1
            xa = self.X(ln.a.start) + (self.hw + 0.15 * S if ina else 0)
            xb = self.X(ln.b.start) - (self.hw + 0.15 * S) if inb else self.x_end - 0.1
            if ln.kind == "slur":
                xa = self.X(ln.a.start) if ina else self.x0 - 0.1
                xb = self.X(ln.b.start) if inb else self.x_end - 0.1
            if not ina:
                xa = self.x0 + 0.05
                ya = yb
            if not inb:
                yb = ya
            gap = 0.9 * S if ln.kind == "tie" else 1.4 * S
            ya += sgn * gap
            yb += sgn * gap
            w = xb - xa
            if w <= 0.05:
                continue
            h = min(0.9 * S + 0.04 * w, 1.4 * S) if ln.kind == "tie" else min(1.2 * S + 0.16 * w, 3.2 * S)
            th = 0.22 * S
            c1 = [xa + w * 0.28, ya + sgn * h * 1.3, 0]
            c2 = [xb - w * 0.28, yb + sgn * h * 1.3, 0]
            p0, p3 = [xa, ya, 0], [xb, yb, 0]
            c1i = [c1[0], c1[1] - sgn * th * 2.0, 0]
            c2i = [c2[0], c2[1] - sgn * th * 2.0, 0]
            arc = VMobject()
            arc.set_points(np.array([p0, c1, c2, p3, p3, c2i, c1i, p0], dtype=float))
            arc.set_fill(opacity=1).set_stroke(width=0)
            self._static(arc, TIE)

    # -- per-frame
    def x_at_beat(self, beat: float) -> float:
        return self.X(beat)

    def y_extent(self) -> tuple[float, float]:
        S = self.S
        return self.Y(self.staff_ids[0], 8) + 2.2 * S, self.Y(self.staff_ids[-1], 0) - 2.2 * S

    def refresh(self, t: float, vis: float) -> None:
        """Update colours; `vis` mixes everything toward the background."""
        full = abs(vis - self.vis) > 1e-3
        if full:
            self.vis = vis
            for mob, color in self.statics:
                mob.set_color(_col(_BG + (_rgb(color) - _BG) * vis))
            for ink in self.inks:
                if ink.t_on > t:
                    ink.mob.set_color(_col(_BG + (_UN - _BG) * vis))
                elif t > ink.t_off + 1.8:
                    ink.mob.set_color(_col(_BG + (_PL - _BG) * vis))
        while self.pending and self.pending[0].t_on <= t:
            self.active.append(self.pending.pop(0))
        keep: list[Ink] = []
        for ink in self.active:
            c = ink_rgb(t, ink.t_on, ink.t_off)
            ink.mob.set_color(_col(_BG + (c - _BG) * vis))
            if t <= ink.t_off + 1.8:
                keep.append(ink)
        self.active = keep
        while self.pending_glows and self.pending_glows[0][1] <= t:
            g = self.pending_glows.pop(0)
            self.active_glows.append(g)
            self.glow_layer.add(g[0])
        keepg = []
        for g in self.active_glows:
            env = glow_env(t, g[1], g[2]) * vis
            for c, a in zip(g[0], (0.07, 0.04, 0.022)):
                c.set_fill(ACTIVE, opacity=a * env)
            if t <= g[2] + 1.6:
                keepg.append(g)
            else:
                self.glow_layer.remove(g[0])
        self.active_glows = keepg


# ------------------------------------------------------------------ plan ---
def plan_systems(score: Score, cfg: Config) -> list[SystemPlan]:
    S = cfg.space

    def header_width(first: bool, fifths: int) -> float:
        return 0.35 + 0.18 + 0.7 + abs(fifths) * 1.05 * S + (0.6 if first else 0.0) + 0.1

    def solve_u(ms: list[Measure], first: bool) -> float:
        hw = header_width(first, ms[0].fifths)
        beats = sum(m.length for m in ms)
        target = cfg.slot_width - hw - 0.4
        d = max(ms[-1].length - ms[-1].last_onset, 0.25)
        u = target / (beats - d / 2)
        if d * u / 2 > 0.8:
            u = (target + 0.8) / beats
        return u

    plans: list[SystemPlan] = []
    i = 0
    ms = score.measures
    while i < len(ms):
        first = i == 0
        j = i + 1
        while j < len(ms) and solve_u(ms[i:j + 1], first) >= cfg.u_min and ms[j].fifths == ms[i].fifths:
            j += 1
        chunk = ms[i:j]
        u = min(solve_u(chunk, first), cfg.u_max)
        plans.append(SystemPlan(
            chunk, chunk[0].start, chunk[-1].start + chunk[-1].length,
            header_width(first, chunk[0].fifths), u, first, chunk[0].fifths, chunk[0].time, chunk[0].clefs,
        ))
        i = j
    return plans


# ----------------------------------------------------------------- scene ---
class SheetScene(Scene):
    def __init__(self, score: Score, tempo: TempoMap, cfg: Config, still: float | None = None, **kw):
        super().__init__(**kw)
        self.score, self.tempo, self.cfg, self.still = score, tempo, cfg, still

    def construct(self) -> None:
        score, tempo, cfg = self.score, self.tempo, self.cfg
        S = cfg.space
        plans = plan_systems(score, cfg)
        left = -cfg.slot_width / 2
        self.systems = [
            System(p, score, tempo, cfg, left, cfg.slot_y if i % 2 == 0 else -cfg.slot_y, cfg.lead)
            for i, p in enumerate(plans)
        ]
        # Fix each system's staff-line extent to the drawn width.
        lead = cfg.lead
        T = [lead + tempo.seconds(p.b0) for p in plans] + [lead + tempo.seconds(plans[-1].b1)]
        end = T[-1]
        total = end + cfg.tail
        fade_start = end + 1.2
        self.T, self.end = T, end

        title = Text(score.title, font=FONT_TEXT, color=TITLE).scale_to_fit_height(0.5).move_to([0, 0.3, 0])
        sub = Text(score.composer, font=FONT_TEXT, color=SUBTITLE) if score.composer else None
        if sub:
            sub.scale_to_fit_height(0.26).next_to(title, direction=[0, -1, 0], buff=0.35)
        title_group = VGroup(*(m for m in (title, sub) if m))

        cursor = VGroup(
            Line([0, 0, 0], [0, 1, 0], stroke_width=2.4),
            Line([0, 0, 0], [0, 1, 0], stroke_width=14),
        )
        cursor_state = {"in": False}
        in_scene: set[int] = set()
        fade_in_at = []
        for i in range(len(plans)):
            if i == 0:
                fade_in_at.append(lead - 2.8)
            elif i == 1:
                fade_in_at.append(lead - 2.2)
            else:
                fade_in_at.append(T[i - 1] + 1.4)

        def vis_of(i: int, t: float) -> float:
            a = 0.4 * smooth((t - fade_in_at[i]) / 1.6)
            b = smooth((t - (T[i] - 1.4)) / 1.2)
            v = max(a, b)
            if i + 2 < len(plans) or True:
                v = min(v, 1 - smooth((t - T[i + 1]) / 1.2)) if i + 1 < len(T) else v
            return v

        state = {"t": 0.0}

        def update(_m: Mobject, dt: float) -> None:
            state["t"] += dt
            t = state["t"]
            g = 1 - smooth((t - fade_start) / 2.2)
            ta = smooth((t - 0.6) / 1.4) * (1 - smooth((t - (lead - 3.4)) / 1.4))
            title_group.set_opacity(ta)
            cur = None
            for i, sysm in enumerate(self.systems):
                v = vis_of(i, t) * g
                if v > 0.002 and i not in in_scene:
                    self.add(sysm.group)
                    in_scene.add(i)
                elif v <= 0.002 and i in in_scene:
                    self.remove(sysm.group)
                    in_scene.discard(i)
                if i in in_scene:
                    sysm.refresh(t, v)
                if T[i] <= t < T[i + 1]:
                    cur = i
            if cur is None:
                cur = 0 if t < T[0] else len(self.systems) - 1
            sysm = self.systems[cur]
            beat = min(max(tempo.beat(t - lead), plans[cur].b0), plans[cur].b1)
            x = sysm.X(beat)
            y_top, y_bot = sysm.y_extent()
            for ln in cursor:
                ln.put_start_and_end_on([x, y_bot, 0], [x, y_top, 0])
            a = smooth((t - (lead - 1.4)) / 1.0) * g
            cursor[0].set_stroke(CURSOR, opacity=0.7 * a)
            cursor[1].set_stroke(CURSOR, opacity=0.07 * a)
            if a > 0.002 and not cursor_state["in"]:
                self.add(cursor)
                cursor_state["in"] = True
            elif a <= 0.002 and cursor_state["in"]:
                self.remove(cursor)
                cursor_state["in"] = False

        title_group.set_opacity(0)
        self.add(title_group)
        for sysm in self.systems:
            sysm._render_beams()
        driver = Mobject()
        driver.add_updater(update)
        self.add(driver)
        self.wait(self.still if self.still is not None else total)


# ------------------------------------------------------------------- cli ---
def render(
    src: Path,
    out: Path,
    preview: bool = False,
    fps: int | None = None,
    bpm: float | None = None,
    speed: float = 1.0,
    default_bpm: float = 60.0,
    still: float | None = None,
    title: str | None = None,
) -> Path:
    score = parse_musicxml(src)
    if title:
        score.title = title
    tempo = TempoMap(score.tempos, default_bpm, speed, bpm)
    cfg = Config()
    w, h, f = (1280, 720, 15) if preview else (3840, 2160, 30)
    settings = {
        "pixel_width": w,
        "pixel_height": h,
        "frame_rate": fps or f,
        "background_color": BG,
        "media_dir": str(out.parent / ".manim_media"),
        "output_file": out.stem,
        "format": "png" if still is not None else "mp4",
        "save_last_frame": still is not None,
        "write_to_movie": still is None,
        "disable_caching": True,
        "progress_bar": "display",
        "verbosity": "WARNING",
    }
    with tempconfig(settings):
        scene = SheetScene(score, tempo, cfg, still=still)
        scene.render()
    media = out.parent / ".manim_media"
    ext = ".png" if still is not None else ".mp4"
    found = [p for p in media.rglob(out.stem + ext) if "partial_movie_files" not in p.parts]
    if not found:
        raise FileNotFoundError(f"Manim produced no {ext} output under {media}")
    out.parent.mkdir(parents=True, exist_ok=True)
    max(found, key=lambda p: p.stat().st_mtime).replace(out)
    return out


def main(argv: list[str] | None = None) -> int:
    ap = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument("musicxml", type=Path)
    ap.add_argument("-o", "--output", type=Path)
    ap.add_argument("--preview", action="store_true", help="720p / 15 fps")
    ap.add_argument("--fps", type=int)
    ap.add_argument("--bpm", type=float, help="force a constant tempo (quarter notes per minute)")
    ap.add_argument("--default-bpm", type=float, default=60.0, help="tempo when the score has none")
    ap.add_argument("--speed", type=float, default=1.0, help="tempo multiplier (0.8 = slower)")
    ap.add_argument("--still", type=float, help="save one PNG frame at this time (seconds)")
    ap.add_argument("--title", help="override the title card text")
    a = ap.parse_args(argv)
    out = a.output or a.musicxml.with_suffix(".png" if a.still is not None else ".mp4")
    print(render(a.musicxml, out, a.preview, a.fps, a.bpm, a.speed, a.default_bpm, a.still, a.title))
    return 0


if __name__ == "__main__":
    sys.exit(main())
