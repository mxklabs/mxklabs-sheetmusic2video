"""Render a piano MusicXML score as a calm, 4K Manim video.

Engraving is done by Verovio (SMuFL fonts, proper spacing, beams, slurs, dynamics).
The score is shown as a grand staff in two alternating slots (top / bottom). A soft
cursor sweeps the active system; notes warm up as they sound and then settle into a
quiet grey-blue.

    python sheet2video.py score.musicxml -o score.mp4            # 4K
    python sheet2video.py score.musicxml --preview               # 720p, fast
    python sheet2video.py score.musicxml --still 20 -o frame.png # one frame
    python sheet2video.py score.musicxml --start 40 --duration 8 --workers 2 -o excerpt.mp4
"""

from __future__ import annotations

import argparse
from concurrent.futures import FIRST_COMPLETED, ProcessPoolExecutor, wait
import math
import multiprocessing
import os
import re
import subprocess
import sys
import tempfile
import zipfile
import xml.etree.ElementTree as ET
from dataclasses import dataclass, field, fields
from pathlib import Path

import numpy as np
import verovio
from tqdm import tqdm
from manim import (
    Circle,
    Group,
    ImageMobject,
    Line,
    ManimColor,
    Mobject,
    Rectangle,
    RoundedRectangle,
    Scene,
    SVGMobject,
    Text,
    UP,
    VGroup,
    tempconfig,
)

# ---------------------------------------------------------------- palette ---

VEROVIO_STAFF_SPACE_PX = 18.0  # one staff space in Verovio's SVG pixels (sets how many measures fit per system)


def _f(default, help: str):
    return field(default=default, metadata={"help": help})


@dataclass
class Config:
    """Every field becomes a --kebab-case command-line option; defaults are the standard look."""

    # layout
    space: float = _f(0.12, "staff space in scene units (smaller = more measures per system)")
    slot_width: float = _f(12.6, "width of a system in scene units")
    slot_top: float = _f(2.35, "vertical centre of the top system")
    slot_bottom: float = _f(-0.35, "vertical centre of the bottom system")
    lead: float = _f(7.0, "seconds before the first beat (title card + fade-in)")
    tail: float = _f(3.0, "seconds after the last beat")
    # colours
    bg: str = _f("#0B0E12", "background")
    unplayed: str = _f("#D8D2C5", "notes not yet played")
    active: str = _f("#E6C58C", "notes while sounding")
    played: str = _f("#8892A2", "notes after they have sounded")
    staff: str = _f("#1e508a", "staff lines")
    mark: str = _f("#8C919A", "clefs, key/time signatures, barlines")
    rest: str = _f("#7A808A", "rests, ledger lines, pedal marks")
    tie: str = _f("#757C87", "ties and slurs")
    cursor: str = _f("#C9AE7B", "playback cursor and its light")
    title_color: str = _f("#D8D2C5", "title card text")
    subtitle_color: str = _f("#7F8591", "composer text")
    light_cool: str = _f("#3C4F73", "cool ambient light")
    light_warm: str = _f("#6B5233", "warm ambient light")
    dust_color: str = _f("#CFC6B4", "floating dust")
    key_white: str = _f("#B9B3A6", "white piano keys")
    key_black: str = _f("#12161C", "black piano keys")
    key_edge: str = _f("#1A1F27", "piano key outlines")
    # keyboard
    keyboard: int = _f(1, "show the 88-key keyboard (0 = hide)")
    keyboard_width: float = _f(13.0, "keyboard width in scene units")
    keyboard_height: float = _f(1.05, "white key height in scene units")
    keyboard_top: float = _f(-2.6, "y of the top edge of the keys")
    # note animation
    note_ramp: float = _f(0.22, "seconds for a note to warm up when it sounds")
    note_decay: float = _f(1.8, "seconds for a note to fade to the played colour")
    min_note: float = _f(0.5, "minimum seconds a note stays lit")
    glow: float = _f(1.0, "note halo strength multiplier (0 = off)")
    glow_ramp: float = _f(0.2, "seconds for a halo to appear")
    glow_decay: float = _f(1.6, "seconds for a halo to fade")
    # system transitions
    dim: float = _f(0.4, "brightness of the waiting system (notes stay hidden until it nears)")
    fade_in: float = _f(1.6, "seconds for a new waiting system to fade in")
    fade_out: float = _f(1.2, "seconds for a finished system to fade out")
    approach: float = _f(1.4, "seconds before playing that the next system brightens")
    # floating and lighting
    float_x: float = _f(0.0175, "horizontal drift amplitude (0 = none)")
    float_y: float = _f(0.0275, "vertical drift amplitude (0 = none)")
    float_period_x: float = _f(13.0, "seconds per horizontal drift cycle")
    float_period_y: float = _f(9.0, "seconds per vertical drift cycle")
    ambient: float = _f(1.0, "ambient light strength multiplier (0 = off)")
    cursor_light: float = _f(0.16, "opacity of the light following the cursor (0 = off)")
    cursor_opacity: float = _f(0.01, "opacity of the cursor line")
    cursor_width: float = _f(40.0, "stroke width of the cursor line")
    dust_count: int = _f(26, "number of dust specks (0 = none)")
    dust_opacity: float = _f(1.0, "dust opacity multiplier")
    dust_size: float = _f(1.0, "dust size multiplier")
    seed: int = _f(7, "random seed for dust placement")


@dataclass
class Meta:
    title: str
    composer: str
    has_tempo: bool


def _open_xml(path: Path) -> ET.Element:
    if zipfile.is_zipfile(path):
        with zipfile.ZipFile(path) as z:
            container = ET.fromstring(z.read("META-INF/container.xml"))
            rootfile = container.find(".//rootfile").attrib["full-path"]
            return ET.fromstring(z.read(rootfile))
    return ET.parse(path).getroot()


def read_meta(path: Path) -> Meta:
    root = _open_xml(path)
    title = (root.findtext("work/work-title") or root.findtext("movement-title") or path.stem).strip()
    composer = ""
    for c in root.findall("identification/creator"):
        if c.get("type") == "composer" and c.text:
            composer = c.text.strip()
    has_tempo = any(s.get("tempo") for s in root.iter("sound"))
    return Meta(title, composer, has_tempo)


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


# ------------------------------------------------------------- colour mix ---
def _rgb(c: str) -> np.ndarray:
    return np.array(ManimColor(c).to_rgb())


def _col(rgb: np.ndarray) -> ManimColor:
    return ManimColor.from_rgb(tuple(float(v) for v in np.clip(rgb, 0, 1)))


def smooth(x: float) -> float:
    x = min(max(x, 0.0), 1.0)
    return x * x * (3 - 2 * x)


def ink_rgb(t: float, t_on: float, t_off: float, un: np.ndarray, ac: np.ndarray, pl: np.ndarray,
            ramp: float, decay: float) -> np.ndarray:
    if t < t_on:
        return un
    a_on = smooth((t - t_on) / ramp)
    if t < t_off:
        return un + (ac - un) * a_on
    a_off = smooth((t_off - t_on) / ramp)
    start = un + (ac - un) * a_off
    return start + (pl - start) * smooth((t - t_off) / decay)


def glow_env(t: float, t_on: float, t_off: float, ramp: float, decay: float) -> float:
    if t < t_on:
        return 0.0
    if t < t_off:
        return smooth((t - t_on) / ramp)
    return smooth((t_off - t_on) / ramp) * (1 - smooth((t - t_off) / decay))


# --------------------------------------------------------------- engraving ---
@dataclass
class SystemPlan:
    svg: str
    b0: float
    b1: float
    note_ids: list[str]


@dataclass
class Engraving:
    plans: list[SystemPlan]
    on: dict[str, float]  # note id -> onset (quarter notes)
    off: dict[str, float]
    onsets: list[float]  # every onset in the piece, sorted
    pitch: dict[str, int]  # note id -> MIDI pitch (timemap notes only)
    tempos: list[tuple[float, float]]
    end: float
    tied: set[str]  # notes tied to an earlier note; they share its sounding span


def engrave(src: Path, cfg: Config) -> Engraving:
    tk = verovio.toolkit()
    tk.setOptions({
        "pageWidth": round(cfg.slot_width * VEROVIO_STAFF_SPACE_PX / cfg.space),
        "scale": 100,
        "systemMaxPerPage": 1,
        "adjustPageHeight": True,
        "breaks": "auto",
        "header": "none",
        "footer": "none",
        "pageMarginLeft": 30,
        "pageMarginRight": 30,
        "pageMarginTop": 0,
        "pageMarginBottom": 0,
    })
    if not tk.loadFile(str(src)):
        raise ValueError(f"Verovio could not read {src}")

    tm = tk.renderToTimemap({"includeMeasures": True, "includeRests": True})
    on: dict[str, float] = {}
    off: dict[str, float] = {}
    measure_q: dict[str, float] = {}
    tempos: list[tuple[float, float]] = []
    for e in tm:
        q = float(e["qstamp"])
        for i in e.get("on", []):
            on[i] = q
        for i in e.get("off", []):
            off[i] = q
        if "measureOn" in e:
            measure_q[e["measureOn"]] = q
        if "tempo" in e:
            tempos.append((q, float(e["tempo"])))
    end = max(float(e["qstamp"]) for e in tm)
    # A tie chain is one continuous note: every member shares the first onset and the last release.
    nxt = dict(re.findall(r'<tie [^>]*startid="#([\w-]+)" endid="#([\w-]+)"', tk.getMEI({})))
    tied = {b for b in nxt.values() if b in on}
    for root in (a for a in nxt if a not in tied and a in on):
        chain = [root]
        while chain[-1] in nxt and nxt[chain[-1]] in on:
            chain.append(nxt[chain[-1]])
        for m in chain:
            on[m], off[m] = on[root], off[chain[-1]]
    pitch = {i: int(tk.getMIDIValuesForElement(i)["pitch"]) for i in on}

    svgs = [tk.renderToSVG(p) for p in range(1, tk.getPageCount() + 1)]
    starts = []
    for svg in svgs:
        mids = re.findall(r'<g id="([\w-]+)" class="measure"', svg)
        starts.append(min(measure_q[m] for m in mids if m in measure_q))
    plans = [
        SystemPlan(svg, starts[i], starts[i + 1] if i + 1 < len(svgs) else end,
                   re.findall(r'<g id="([\w-]+)" class="note"', svg))
        for i, svg in enumerate(svgs)
    ]
    return Engraving(plans, on, off, sorted(set(on.values())), pitch, tempos, end, tied)


class System:
    """One Verovio system placed in a slot, with per-note colour/glow timing."""

    def __init__(self, plan: SystemPlan, eng: Engraving, tempo: TempoMap, cfg: Config,
                 ox: float, oy: float, lead: float):
        self.plan, self.cfg, self.lead, self.tempo = plan, cfg, lead, tempo
        self.S = cfg.space
        self.un, self.ac, self.pl = _rgb(cfg.unplayed), _rgb(cfg.active), _rgb(cfg.played)
        self.statics: list[tuple[Mobject, str, bool]] = []
        self.inks: list[Ink] = []
        self.glows: list[tuple[Mobject, float, float]] = []  # (head leaf, on, off); images are made lazily
        self.glow_pool: list[ImageMobject] = []
        self.vis = -1.0
        self.active: list[Ink] = []
        self.active_glows: list[tuple[Mobject, float, float]] = []
        self.glow_layer = Group()

        with tempfile.TemporaryDirectory() as d:
            path = Path(d) / "system.svg"
            path.write_text(plan.svg, encoding="utf-8")
            svg = SVGMobject(str(path), height=None, should_center=False)
        leaves = list(svg.submobjects)
        self.base = {id(m): (float(m.get_fill_opacity()), float(m.get_stroke_opacity())) for m in leaves}
        flat = [m for m in leaves if m.height < 1e-3 and m.width > 0]
        widest = max(m.width for m in flat)
        lines = [m for m in flat if m.width > 0.15 * widest]  # staff lines, not ledger lines
        ys = sorted({round(float(m.get_center()[1]), 4) for m in lines}, reverse=True)
        native_space = min(a - b for a, b in zip(ys, ys[1:]) if a - b > 1e-3)
        svg.scale(self.S / native_space, about_point=np.zeros(3))
        self.svg = svg
        left = min(m.get_left()[0] for m in lines)
        self.x_end = max(m.get_right()[0] for m in lines)
        mid = (max(m.get_top()[1] for m in lines) + min(m.get_bottom()[1] for m in lines)) / 2
        svg.shift(np.array([ox + 0.3 - left, oy - mid, 0.0]))
        self.x_end += ox + 0.3 - left
        self.y_top = max(m.get_top()[1] for m in lines) + 2.2 * self.S
        self.y_bot = min(m.get_bottom()[1] for m in lines) - 2.2 * self.S
        self.group = Group(self.glow_layer, svg)

        self._classify(leaves, lines, eng)
        self.pending = sorted(self.inks, key=lambda i: i.t_on)
        self.pending_glows = sorted(self.glows, key=lambda g: g[1])

    def sec(self, beat: float) -> float:
        return self.lead + self.tempo.seconds(beat)

    def _classify(self, leaves: list, lines: list, eng: Engraving) -> None:
        d = self.svg.id_to_vgroup_dict
        svg_text = self.plan.svg

        def leaves_of(i: str) -> list:
            return list(d[i].submobjects) if i in d else []

        def ids(cls: str) -> list[str]:
            return re.findall(rf'<g id="([\w-]+)" class="{cls}"', svg_text)

        used: set[int] = set()
        times: dict[str, tuple[float, float]] = {}
        known = [(i, eng.on[i], eng.off.get(i, eng.on[i])) for i in self.plan.note_ids if i in eng.on]
        for i, q0, q1 in known:
            times[i] = (q0, q1)
        # Tied-to notes are absent from the timemap; infer their onset from horizontal position.
        if known:
            xs = [(self._head(leaves_of(i)).get_center()[0], q0) for i, q0, _ in known]
            xs.sort()
            for i in self.plan.note_ids:
                if i in times or not leaves_of(i):
                    continue
                x = self._head(leaves_of(i)).get_center()[0]
                q0 = min(xs, key=lambda p: abs(p[0] - x))[1]
                nxt = next((o for o in eng.onsets if o > q0 + 1e-9), q0 + 1)
                times[i] = (q0, nxt)

        knots: dict[float, list[float]] = {}
        for i in self.plan.note_ids:
            ls = leaves_of(i)
            if i not in times or not ls:
                continue
            q0, q1 = times[i]
            t_on = self.sec(q0)
            t_off = max(self.sec(q1), t_on + self.cfg.min_note)
            self._add_ink(ls, t_on, t_off, used)
            head = self._head(ls)
            if i not in eng.tied:  # a tied-to head sits later than the shared onset
                knots.setdefault(q0, []).append(float(head.get_center()[0]))
            self.glows.append((head, t_on, t_off))

        # Chord stems and beams belong to several notes; colour them over the group's span.
        for cls in ("chord", "beam"):
            for i in ids(cls):
                note_leaves = {id(m) for n in ids("note") if n in times and self._inside(leaves_of(n), leaves_of(i))
                               for m in leaves_of(n)}
                own = [m for m in leaves_of(i) if id(m) not in note_leaves and id(m) not in used]
                members = [times[n] for n in ids("note") if n in times and self._inside(leaves_of(n), leaves_of(i))]
                if own and members:
                    t_on = self.sec(min(m[0] for m in members))
                    t_off = max(self.sec(max(m[1] for m in members)), t_on + self.cfg.min_note)
                    self._add_ink(own, t_on, t_off, used)

        for cls in ("rest", "mRest", "pedal"):
            for i in ids(cls):
                for m in leaves_of(i):
                    if id(m) not in used:
                        used.add(id(m))
                        self._static(m, self.cfg.rest, True)
        for cls in ("tie", "slur"):
            for i in ids(cls):
                for m in leaves_of(i):
                    if id(m) not in used:
                        used.add(id(m))
                        self._static(m, self.cfg.tie, True)
        line_ids = {id(m) for m in lines}
        flat_ids = {id(m) for m in leaves if m.height < 1e-3 and m.width > 0}
        for m in leaves:
            if id(m) not in used:
                if id(m) in line_ids:
                    self._static(m, self.cfg.staff)
                elif id(m) in flat_ids:  # ledger lines are note-level notation
                    self._static(m, self.cfg.rest, True)
                else:
                    self._static(m, self.cfg.mark)

        if not knots:
            self.knots: list[tuple[float, float]] = [(self.plan.b0, self.x_end), (self.plan.b1, self.x_end)]
            return
        pts = sorted((q, float(np.median(v))) for q, v in knots.items())
        if pts[0][0] > self.plan.b0 + 1e-6:
            pts.insert(0, (self.plan.b0, pts[0][1] - 0.3))
        if pts[-1][0] < self.plan.b1:
            pts.append((self.plan.b1, self.x_end))
        self.knots = pts

    @staticmethod
    def _inside(inner: list, outer: list) -> bool:
        s = {id(m) for m in outer}
        return bool(inner) and all(id(m) in s for m in inner)

    @staticmethod
    def _head(leaves: list):
        return max(leaves, key=lambda m: m.width)

    def _static(self, mob: Mobject, color: str, notation: bool = False) -> None:
        mob.set_color(color)
        self.statics.append((mob, color, notation))

    def _add_ink(self, leaves: list, t_on: float, t_off: float, used: set[int]) -> None:
        fresh = [m for m in leaves if id(m) not in used]
        if not fresh:
            return
        used.update(id(m) for m in fresh)
        g = VGroup(*fresh)
        g.set_color(self.cfg.unplayed)
        self.inks.append(Ink(g, t_on, t_off))

    # -- per-frame
    def X(self, beat: float) -> float:
        return float(np.interp(beat, [k[0] for k in self.knots], [k[1] for k in self.knots]))

    def y_extent(self) -> tuple[float, float]:
        return self.y_top, self.y_bot

    def _paint(self, leaves: list, color, k: float) -> None:
        # Fade by opacity, not by mixing toward BG, so notes never read darker than the lit background.
        for m in leaves:
            f, s = self.base[id(m)]
            m.set_fill(color, opacity=f * k)
            m.set_stroke(color, opacity=s * k)

    def refresh(self, t: float, vis: float) -> None:
        """Update colours; `vis` fades the system, and notation stays hidden while it only waits."""
        cfg = self.cfg
        nv = max(0.0, (vis - (cfg.dim + 0.02)) / (1 - cfg.dim - 0.02))
        if abs(vis - self.vis) > 1e-3:
            self.vis = vis
            for mob, color, notation in self.statics:
                self._paint([mob], color, nv if notation else vis)
            for ink in self.inks:
                if ink.t_on > t:
                    self._paint(ink.mob.submobjects, cfg.unplayed, nv)
                elif t > ink.t_off + cfg.note_decay:
                    self._paint(ink.mob.submobjects, cfg.played, nv)
        while self.pending and self.pending[0].t_on <= t:
            self.active.append(self.pending.pop(0))
        keep: list[Ink] = []
        for ink in self.active:
            rgb = ink_rgb(t, ink.t_on, ink.t_off, self.un, self.ac, self.pl, cfg.note_ramp, cfg.note_decay)
            self._paint(ink.mob.submobjects, _col(rgb), nv)
            if t <= ink.t_off + cfg.note_decay:
                keep.append(ink)
        self.active = keep
        while self.pending_glows and self.pending_glows[0][1] <= t:
            head, a, b = self.pending_glows.pop(0)
            img = self.glow_pool.pop() if self.glow_pool else soft_light(cfg.active, 2.6 * self.S, 0.13)
            img.move_to(head.get_center())
            self.active_glows.append((img, a, b))
            self.glow_layer.add(img)
        keepg = []
        for g in self.active_glows:
            env = glow_env(t, g[1], g[2], cfg.glow_ramp, cfg.glow_decay) * vis * cfg.glow
            set_light(g[0], env)
            if t <= g[2] + cfg.glow_decay:
                keepg.append(g)
            else:
                self.glow_layer.remove(g[0])
                self.glow_pool.append(g[0])
        self.active_glows = keepg


@dataclass
class Ink:
    mob: Mobject
    t_on: float
    t_off: float


_LIGHT_CACHE: dict[tuple, ImageMobject] = {}


def soft_light(color: str, radius: float, peak: float) -> ImageMobject:
    """Radial glow as a smooth gradient image; peak = opacity at the centre.

    The alpha is dithered so the 8-bit gradient (and its video compression) shows no rings.
    """
    px = int(min(1024, max(128, radius * 160)))
    key = (color, px, round(peak, 4))
    if key not in _LIGHT_CACHE:
        yy, xx = np.mgrid[-1:1:px * 1j, -1:1:px * 1j]
        a = np.clip(1 - (xx ** 2 + yy ** 2), 0, 1) ** 3
        noise = np.random.default_rng(1).uniform(-3, 3, a.shape) * np.clip(a * 20, 0, 1)
        arr = np.zeros((px, px, 4), dtype=np.uint8)
        arr[..., :3] = (_rgb(color) * 255).round().astype(np.uint8)
        arr[..., 3] = np.clip(np.round(a * peak * 255 + noise), 0, 255).astype(np.uint8)
        _LIGHT_CACHE[key] = ImageMobject(arr)
    g = _LIGHT_CACHE[key].copy()
    g.scale_to_fit_height(2 * radius)
    g.k = 1.0  # type: ignore[attr-defined]
    return g


def set_light(g: ImageMobject, k: float) -> None:
    k = min(max(k, 0.0), 1.0)
    if abs(k - g.k) > 1e-3:  # type: ignore[attr-defined]
        g.set_opacity(k)
        g.k = k  # type: ignore[attr-defined]


# ----------------------------------------------------------------- scene ---
class Keyboard:
    """88-key piano (A0-C8) at the bottom of the frame; keys light as notes sound."""

    FIRST, LAST = 21, 108
    WHITE_PCS = {0, 2, 4, 5, 7, 9, 11}
    BLACK_SHIFT = {1: -0.1, 3: 0.1, 6: -0.12, 8: 0.0, 10: 0.12}  # in white-key widths

    def __init__(self, cfg: Config, events: list[tuple[int, float, float]]):
        self.cfg = cfg
        self.keys: dict[int, RoundedRectangle] = {}
        self.is_black: dict[int, bool] = {}
        self.dy: dict[int, float] = {}
        self.black_h = cfg.keyboard_height * 0.62
        n_white = sum(1 for m in range(self.FIRST, self.LAST + 1) if m % 12 in self.WHITE_PCS)
        w = cfg.keyboard_width / n_white
        h, top = cfg.keyboard_height, cfg.keyboard_top
        left = -cfg.keyboard_width / 2
        whites, blacks, labels = [], [], []
        wi = -1
        for m in range(self.FIRST, self.LAST + 1):
            pc = m % 12
            if pc in self.WHITE_PCS:
                wi += 1
                k = RoundedRectangle(width=w - 0.012, height=h, corner_radius=0.03)
                k.move_to([left + (wi + 0.5) * w, top - h / 2, 0]).set_z_index(4)
                whites.append(k)
                if pc == 0:
                    lab = Text(f"C{m // 12 - 1}", font="Palatino").scale_to_fit_height(0.075)
                    lab.move_to([left + (wi + 0.5) * w, top - h + 0.11, 0]).set_z_index(5)
                    labels.append(lab)
            else:
                bh = h * 0.62
                k = RoundedRectangle(width=w * 0.58, height=bh, corner_radius=0.02)
                k.move_to([left + (wi + 1 + self.BLACK_SHIFT[pc]) * w, top - bh / 2, 0]).set_z_index(6)
                blacks.append(k)
            self.keys[m] = k
            self.is_black[m] = pc not in self.WHITE_PCS
            self.dy[m] = 0.0
        self.labels = labels
        board = Rectangle(width=cfg.keyboard_width + 0.3, height=0.08, stroke_width=0)
        board.set_fill(cfg.key_edge, opacity=1).move_to([0, top + 0.04, 0]).set_z_index(3)
        self.board = board
        self.group = Group(board, *whites, *blacks, *labels)
        self.glows: dict[int, ImageMobject] = {}
        self.lit: set[int] = set()
        self.k = -1.0
        black_active = _rgb(cfg.active) * 0.65 + _rgb(cfg.key_black) * 0.35
        self.base = {False: _rgb(cfg.key_white), True: _rgb(cfg.key_black)}
        self.hot = {False: _rgb(cfg.active), True: black_active}
        self.events = sorted(events, key=lambda e: e[1])
        self.pending = list(self.events)
        self.active: list[tuple[int, float, float]] = []

    def _paint(self, m: int, rgb: np.ndarray, k: float) -> None:
        key = self.keys[m]
        key.set_fill(_col(rgb), opacity=k)
        key.set_stroke(self.cfg.key_edge, width=1.2, opacity=k)

    def _glow(self, m: int) -> ImageMobject:
        if m not in self.glows:
            key = self.keys[m]
            g = soft_light(self.cfg.active, 0.45, 0.15).move_to([key.get_center()[0], self.cfg.keyboard_top, 0])
            set_light(g, 0.0)
            g.set_z_index(7)
            self.glows[m] = g
        return self.glows[m]

    def refresh(self, t: float, k: float) -> None:
        cfg = self.cfg
        if abs(k - self.k) > 1e-3:
            self.k = k
            for m in self.keys:
                self._paint(m, self.base[self.is_black[m]], k)
            self.board.set_fill(cfg.key_edge, opacity=k)
            for lab in self.labels:
                lab.set_fill(cfg.key_black, opacity=0.55 * k)
        while self.pending and self.pending[0][1] <= t:
            self.active.append(self.pending.pop(0))
        env: dict[int, float] = {}
        keep = []
        for ev in self.active:
            m, t_on, t_off = ev
            b = self.is_black[m]
            rgb = ink_rgb(t, t_on, t_off, self.base[b], self.hot[b], self.base[b], cfg.note_ramp, cfg.note_decay)
            self._paint(m, rgb, k)
            env[m] = max(env.get(m, 0.0), glow_env(t, t_on, t_off, cfg.note_ramp, cfg.note_decay))
            if t <= t_off + cfg.note_decay:
                keep.append(ev)
        self.active = keep
        for m in set(env) | self.lit:
            e = env.get(m, 0.0)
            dip = 0.02 * e
            if self.is_black[m]:  # keep the top edge fixed; the key lengthens downward
                h0 = self.keys[m].height
                self.keys[m].stretch((self.black_h + dip) / h0, 1, about_edge=UP)
            else:
                self.keys[m].shift([0, -(dip - self.dy[m]), 0])
            self.dy[m] = dip
            g = self._glow(m)
            set_light(g, e * k * cfg.glow)
            if e > 0 and g not in self.group.submobjects:
                self.group.add(g)
            elif e == 0 and g in self.group.submobjects:
                self.group.remove(g)
        self.lit = set(env)


class SheetScene(Scene):
    def __init__(self, meta: Meta, eng: Engraving, tempo: TempoMap, cfg: Config, still: float | None = None,
                 start: float = 0.0, duration: float | None = None,
                 progress_path: Path | None = None, progress_frames: int = 0, **kw):
        super().__init__(**kw)
        self.meta, self.eng, self.tempo, self.cfg, self.still = meta, eng, tempo, cfg, still
        self.start, self.duration = start, duration
        self.progress_path, self.progress_frames = progress_path, progress_frames

    def construct(self) -> None:
        meta, eng, tempo, cfg = self.meta, self.eng, self.tempo, self.cfg
        # Manim caches everything ordered (by z_index) before the first updater mobject as a static image.
        driver = Mobject()
        driver.set_z_index(-100)
        self.add(driver)
        plans = eng.plans
        left = -cfg.slot_width / 2
        lead = cfg.lead
        self.systems = [
            System(p, eng, tempo, cfg, left, cfg.slot_top if i % 2 == 0 else cfg.slot_bottom, lead)
            for i, p in enumerate(plans)
        ]
        T = [lead + tempo.seconds(p.b0) for p in plans] + [lead + tempo.seconds(plans[-1].b1)]
        end = T[-1]
        total = end + cfg.tail
        fade_start = end + 1.2
        self.T, self.end = T, end

        title = Text(meta.title, font="Palatino", color=cfg.title_color).scale_to_fit_height(0.5).move_to([0, 0.3, 0])
        sub = Text(meta.composer, font="Palatino", color=cfg.subtitle_color) if meta.composer else None
        if sub:
            sub.scale_to_fit_height(0.26).next_to(title, direction=[0, -1, 0], buff=0.35)
        title_group = VGroup(*(m for m in (title, sub) if m))

        cursor = VGroup(
            Line([0, 0, 0], [0, 1, 0], stroke_width=cfg.cursor_width),
            Line([0, 0, 0], [0, 1, 0], stroke_width=14),
        )
        cursor.set_z_index(10)  # systems are added later, so order alone would draw them over it
        cursor_state = {"in": False}

        # Atmosphere: two slow ambient lights, a light that follows the cursor, and drifting dust.
        ambient = [
            (soft_light(cfg.light_cool, 8.0, 0.55), (4.2, 2.0), (31.0, 43.0), 0.0),
            (soft_light(cfg.light_warm, 6.5, 0.38), (4.8, 1.6), (37.0, 29.0), 2.1),
        ]
        for light, *_ in ambient:
            light.set_z_index(-10)
        cursor_light = soft_light(cfg.cursor, 3.0, cfg.cursor_light)
        cursor_light.set_z_index(-5)
        rng = np.random.default_rng(cfg.seed)
        dust = []
        for _ in range(cfg.dust_count):
            r = float(rng.uniform(0.02, 0.05)) * cfg.dust_size
            d = Circle(radius=r).set_stroke(width=0).set_fill(cfg.dust_color, opacity=0)
            d.set_z_index(-8)
            dust.append((d, rng.uniform(-7.1, 7.1), rng.uniform(0, 8), rng.uniform(0.04, 0.11),
                         rng.uniform(0.2, 0.7), rng.uniform(0.25, 0.7), rng.uniform(0, 6.28), rng.uniform(5, 11)))
        for light, *_ in ambient:
            self.add(light)
        for d, *_ in dust:
            self.add(d)
        offsets = {}
        keyboard = None
        if cfg.keyboard:
            events = []
            for nid, q0 in eng.on.items():
                if nid in eng.pitch and nid not in eng.tied:
                    t0 = lead + tempo.seconds(q0)
                    t1 = max(lead + tempo.seconds(eng.off.get(nid, q0)), t0 + cfg.min_note)
                    events.append((eng.pitch[nid], t0, t1))
            keyboard = Keyboard(cfg, [e for e in events if Keyboard.FIRST <= e[0] <= Keyboard.LAST])
            self.add(keyboard.group)
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
            a = cfg.dim * smooth((t - fade_in_at[i]) / cfg.fade_in)
            b = smooth((t - (T[i] - cfg.approach)) / 1.2)
            v = max(a, b)
            return min(v, 1 - smooth((t - T[i + 1]) / cfg.fade_out))

        state = {"t": self.start}
        last_progress_frame = 0
        progress_interval = max(1, int(self.camera.frame_rate))

        def update(_m: Mobject, dt: float) -> None:
            nonlocal last_progress_frame
            state["t"] += dt
            t = state["t"]
            if self.progress_path is not None:
                frame = min(self.progress_frames, max(0, int((t - self.start) * self.camera.frame_rate)))
                if frame - last_progress_frame >= progress_interval:
                    self.progress_path.write_text(str(frame), encoding="utf-8")
                    last_progress_frame = frame
            g = 1 - smooth((t - fade_start) / 2.2)
            ta = smooth((t - 0.6) / 1.4) * (1 - smooth((t - (lead - 3.4)) / 1.4))
            title_group.set_opacity(ta)
            fade_in = smooth(t / 4.0)
            if keyboard:
                keyboard.refresh(t, smooth((t - (lead - 3.2)) / 1.6) * g)
            for light, (ax, ay), (px, py), ph in ambient:
                light.move_to([ax * math.sin(2 * math.pi * t / px + ph), ay * math.cos(2 * math.pi * t / py + ph), 0])
                set_light(light, fade_in * g * cfg.ambient)
            for d, x0, y0, speed, sway, peak, ph, per in dust:
                y = (y0 + speed * t) % 8.4 - 4.2
                d.move_to([x0 + sway * math.sin(2 * math.pi * t / per + ph), y, 0])
                edge = smooth(min(y + 4.2, 4.2 - y) / 1.2)
                twinkle = 0.65 + 0.35 * math.sin(2 * math.pi * t / (per * 0.6) + ph)
                d.set_fill(cfg.dust_color, opacity=peak * edge * twinkle * fade_in * g * cfg.dust_opacity)
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
                    # Slow, out-of-phase drift so the staves seem to float.
                    target = np.array([cfg.float_x * math.sin(2 * math.pi * t / cfg.float_period_x + i * 1.7),
                                       cfg.float_y * math.sin(2 * math.pi * t / cfg.float_period_y + i * 2.3), 0.0])
                    sysm.group.shift(target - offsets.get(i, np.zeros(3)))
                    offsets[i] = target
                    sysm.refresh(t, v)
                if T[i] <= t < T[i + 1]:
                    cur = i
            if cur is None:
                cur = 0 if t < T[0] else len(self.systems) - 1
            sysm = self.systems[cur]
            beat = min(max(tempo.beat(t - lead), plans[cur].b0), plans[cur].b1)
            off = offsets.get(cur, np.zeros(3))
            x = sysm.X(beat) + off[0]
            y_top, y_bot = sysm.y_extent()
            y_top, y_bot = y_top + off[1], y_bot + off[1]
            for ln in cursor:
                ln.put_start_and_end_on([x, y_bot, 0], [x, y_top, 0])
            a = smooth((t - (lead - 1.4)) / 1.0) * g
            cursor_light.move_to([x, (y_top + y_bot) / 2, 0])
            set_light(cursor_light, a)
            cursor[0].set_stroke(cfg.cursor, opacity=cfg.cursor_opacity * a)
            cursor[1].set_stroke(cfg.cursor, opacity=0.07 * a)
            if a > 0.002 and not cursor_state["in"]:
                self.add(cursor, cursor_light)
                cursor_state["in"] = True
            elif a <= 0.002 and cursor_state["in"]:
                self.remove(cursor, cursor_light)
                cursor_state["in"] = False

        title_group.set_opacity(0)
        self.add(title_group)
        driver.add_updater(update)
        self.wait(self.still if self.still is not None else (self.duration if self.duration is not None else total - self.start))


# ------------------------------------------------------------------- cli ---
@dataclass(frozen=True)
class RenderJob:
    src: Path
    workdir: Path
    index: int
    start: float
    duration: float
    width: int
    height: int
    fps: int
    bpm: float | None
    speed: float
    default_bpm: float
    title: str | None
    cfg: Config
    progress_path: Path
    frame_count: int


def _render_segment(job: RenderJob) -> Path:
    meta = read_meta(job.src)
    if job.title:
        meta.title = job.title
    eng = engrave(job.src, job.cfg)
    tempo = TempoMap(eng.tempos if meta.has_tempo else [], job.default_bpm, job.speed, job.bpm)
    name = f"part_{job.index:03d}"
    media = job.workdir / f"media_{job.index:03d}"
    settings = {
        "pixel_width": job.width,
        "pixel_height": job.height,
        "frame_rate": job.fps,
        "background_color": job.cfg.bg,
        "media_dir": str(media),
        "output_file": name,
        "format": "mp4",
        "save_last_frame": False,
        "write_to_movie": True,
        "disable_caching": True,
        "progress_bar": "none",
        "verbosity": "ERROR",
    }
    with tempconfig(settings):
        SheetScene(meta, eng, tempo, job.cfg, start=job.start, duration=job.duration,
                   progress_path=job.progress_path, progress_frames=job.frame_count).render()
    job.progress_path.write_text(str(job.frame_count), encoding="utf-8")
    found = list(media.rglob(name + ".mp4"))
    if not found:
        raise FileNotFoundError(f"Manim produced no segment {name}")
    destination = job.workdir / f"{name}.mp4"
    max(found, key=lambda p: p.stat().st_mtime).replace(destination)
    return destination


def _render_parallel(
    src: Path, out: Path, cfg: Config, start: float, duration: float, width: int, height: int, fps: int,
    bpm: float | None, speed: float, default_bpm: float, title: str | None, workers: int,
) -> Path:
    total_frames = max(1, math.ceil(duration * fps))
    count = min(workers, total_frames)
    with tempfile.TemporaryDirectory(prefix="sheet2video-") as tmp:
        workdir = Path(tmp)
        jobs = []
        for i in range(count):
            first = total_frames * i // count
            last = total_frames * (i + 1) // count
            jobs.append(RenderJob(src, workdir, i, start + first / fps, (last - first) / fps,
                                  width, height, fps, bpm, speed, default_bpm, title, cfg,
                                  workdir / f"progress_{i:03d}.txt", last - first))
        completed_frames = [0] * count
        parts: list[Path | None] = [None] * count
        with ProcessPoolExecutor(max_workers=count, mp_context=multiprocessing.get_context("spawn")) as pool, \
                tqdm(total=total_frames, desc="Rendering", unit="frame", smoothing=0.08) as progress:
            futures = {pool.submit(_render_segment, job): job for job in jobs}
            while futures:
                done, _ = wait(futures, timeout=0.25, return_when=FIRST_COMPLETED)
                for job in jobs:
                    if completed_frames[job.index] == job.frame_count:
                        continue
                    try:
                        rendered = min(job.frame_count, int(job.progress_path.read_text(encoding="utf-8")))
                    except (FileNotFoundError, ValueError):
                        continue
                    delta = max(0, rendered - completed_frames[job.index])
                    if delta:
                        progress.update(delta)
                        completed_frames[job.index] = rendered
                for future in done:
                    job = futures.pop(future)
                    parts[job.index] = future.result()
                    remaining = job.frame_count - completed_frames[job.index]
                    if remaining:
                        progress.update(remaining)
                        completed_frames[job.index] = job.frame_count
        part_paths = [p for p in parts if p is not None]
        if len(part_paths) != count:
            raise RuntimeError("Parallel render ended before all segments completed")
        concat_file = workdir / "segments.txt"
        concat_file.write_text("".join(f"file '{p.as_posix()}'\n" for p in part_paths), encoding="utf-8")
        merged = workdir / "merged.mp4"
        subprocess.run([
            "ffmpeg", "-hide_banner", "-loglevel", "error", "-y", "-f", "concat", "-safe", "0",
            "-i", str(concat_file), "-c", "copy", "-movflags", "+faststart", str(merged),
        ], check=True)
        out.parent.mkdir(parents=True, exist_ok=True)
        merged.replace(out)
    return out


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
    cfg: Config | None = None,
    workers: int | None = None,
    start: float = 0.0,
    duration: float | None = None,
) -> Path:
    meta = read_meta(src)
    if title:
        meta.title = title
    cfg = cfg or Config()
    eng = engrave(src, cfg)
    tempo = TempoMap(eng.tempos if meta.has_tempo else [], default_bpm, speed, bpm)
    w, h, f = (1280, 720, 15) if preview else (3840, 2160, 30)
    fps_value = fps or f
    total = cfg.lead + tempo.seconds(eng.end) + cfg.tail
    if start < 0 or start >= total:
        raise ValueError(f"--start must be between 0 and {total:.2f} seconds")
    window_duration = duration if duration is not None else total - start
    if window_duration <= 0:
        raise ValueError("--duration must be greater than zero")
    window_duration = min(window_duration, total - start)
    if workers is None:
        available_half = max(1, (os.cpu_count() or 1) // 2)
        worker_count = min(available_half, max(1, math.ceil(window_duration / 15)))
    else:
        worker_count = workers
    if worker_count < 1:
        raise ValueError("--workers must be at least 1")
    if still is None and worker_count > 1:
        return _render_parallel(src, out, cfg, start, window_duration, w, h, fps_value, bpm, speed,
                                 default_bpm, title, worker_count)
    settings = {
        "pixel_width": w,
        "pixel_height": h,
        "frame_rate": fps_value,
        "background_color": cfg.bg,
        "media_dir": str(out.parent / ".manim_media"),
        "output_file": out.stem,
        "format": "png" if still is not None else "mp4",
        "save_last_frame": still is not None,
        "write_to_movie": still is None,
        "disable_caching": True,
        "progress_bar": "display",
        "verbosity": "ERROR",
    }
    with tempconfig(settings):
        scene = SheetScene(meta, eng, tempo, cfg, still=still, start=start, duration=window_duration)
        scene.render()
    media = out.parent / ".manim_media"
    ext = ".png" if still is not None else ".mp4"
    found = [p for p in media.rglob(out.stem + ext) if "partial_movie_files" not in p.parts]
    if not found:
        raise FileNotFoundError(f"Manim produced no {ext} output under {media}")
    out.parent.mkdir(parents=True, exist_ok=True)
    max(found, key=lambda p: p.stat().st_mtime).replace(out)
    return out


COLOR_FIELDS = ("bg", "unplayed", "active", "played", "staff", "mark", "rest", "tie", "cursor", "title_color",
                "subtitle_color", "light_cool", "light_warm", "dust_color", "key_white", "key_black", "key_edge")


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
    ap.add_argument("--start", type=float, default=0.0, help="absolute timeline start time for a video excerpt")
    ap.add_argument("--duration", type=float, help="render only this many seconds (useful for short clips)")
    ap.add_argument("--workers", type=int, default=None,
                    help=f"parallel video render processes (default: up to half of {os.cpu_count() or 1} available cores; short clips use fewer; 1 disables parallelism)")
    ap.add_argument("--title", help="override the title card text")
    style = ap.add_argument_group("style and tuning (defaults shown)")
    defaults = Config()
    for f in fields(Config):
        d = getattr(defaults, f.name)
        style.add_argument("--" + f.name.replace("_", "-"), dest=f.name, type=type(d), default=None,
                           metavar="HEX" if f.name in COLOR_FIELDS else type(d).__name__.upper(),
                           help=f"{f.metadata['help']} [{d}]")
    a = ap.parse_args(argv)
    cfg = Config(**{f.name: getattr(a, f.name) for f in fields(Config) if getattr(a, f.name) is not None})
    for name in COLOR_FIELDS:
        ManimColor(getattr(cfg, name))  # fail early on a bad colour
    out = a.output or a.musicxml.with_suffix(".png" if a.still is not None else ".mp4")
    print(render(a.musicxml, out, a.preview, a.fps, a.bpm, a.speed, a.default_bpm, a.still, a.title,
                 cfg, a.workers, a.start, a.duration))
    return 0


if __name__ == "__main__":
    sys.exit(main())
