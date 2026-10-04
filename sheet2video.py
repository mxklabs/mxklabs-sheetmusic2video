"""Render a piano MusicXML score as a calm, 4K Manim video.

Engraving is done by Verovio (SMuFL fonts, proper spacing, beams, slurs, dynamics).
The score is shown as a grand staff in two alternating slots (top / bottom). A soft
cursor sweeps the active system; notes warm up as they sound and then settle into a
quiet grey-blue.

    python sheet2video.py score.musicxml -o score.mp4            # 4K
    python sheet2video.py score.musicxml --preview               # 720p, fast
    python sheet2video.py score.musicxml --still 20 -o frame.png # one frame
"""

from __future__ import annotations

import argparse
import math
import re
import sys
import tempfile
import zipfile
import xml.etree.ElementTree as ET
from dataclasses import dataclass, field, fields
from pathlib import Path

import numpy as np
import verovio
from manim import (
    Circle,
    Line,
    ManimColor,
    Mobject,
    Scene,
    SVGMobject,
    Text,
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
    space: float = _f(0.18, "staff space in scene units (smaller = more measures per system)")
    slot_width: float = _f(12.6, "width of a system in scene units")
    slot_y: float = _f(2.0, "vertical distance of the two system slots from the centre")
    lead: float = _f(7.0, "seconds before the first beat (title card + fade-in)")
    tail: float = _f(3.0, "seconds after the last beat")
    # colours
    bg: str = _f("#0B0E12", "background")
    unplayed: str = _f("#D8D2C5", "notes not yet played")
    active: str = _f("#E6C58C", "notes while sounding")
    played: str = _f("#8892A2", "notes after they have sounded")
    staff: str = _f("#4B525D", "staff lines")
    mark: str = _f("#8C919A", "clefs, key/time signatures, barlines")
    rest: str = _f("#7A808A", "rests, ledger lines, pedal marks")
    tie: str = _f("#757C87", "ties and slurs")
    cursor: str = _f("#C9AE7B", "playback cursor and its light")
    title_color: str = _f("#D8D2C5", "title card text")
    subtitle_color: str = _f("#7F8591", "composer text")
    light_cool: str = _f("#3C4F73", "cool ambient light")
    light_warm: str = _f("#6B5233", "warm ambient light")
    dust_color: str = _f("#CFC6B4", "floating dust")
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
    float_x: float = _f(0.035, "horizontal drift amplitude (0 = none)")
    float_y: float = _f(0.055, "vertical drift amplitude (0 = none)")
    float_period_x: float = _f(13.0, "seconds per horizontal drift cycle")
    float_period_y: float = _f(9.0, "seconds per vertical drift cycle")
    ambient: float = _f(1.0, "ambient light strength multiplier (0 = off)")
    cursor_light: float = _f(0.16, "opacity of the light following the cursor (0 = off)")
    cursor_opacity: float = _f(0.7, "opacity of the cursor line")
    cursor_width: float = _f(2.4, "stroke width of the cursor line")
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
    tempos: list[tuple[float, float]]
    end: float


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
    return Engraving(plans, on, off, sorted(set(on.values())), tempos, end)


class System:
    """One Verovio system placed in a slot, with per-note colour/glow timing."""

    def __init__(self, plan: SystemPlan, eng: Engraving, tempo: TempoMap, cfg: Config,
                 ox: float, oy: float, lead: float):
        self.plan, self.cfg, self.lead, self.tempo = plan, cfg, lead, tempo
        self.S = cfg.space
        self.un, self.ac, self.pl = _rgb(cfg.unplayed), _rgb(cfg.active), _rgb(cfg.played)
        self.statics: list[tuple[Mobject, str, bool]] = []
        self.inks: list[Ink] = []
        self.glows: list[tuple[VGroup, float, float]] = []
        self.vis = -1.0
        self.active: list[Ink] = []
        self.active_glows: list[tuple[VGroup, float, float]] = []
        self.glow_layer = VGroup()

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
        self.group = VGroup(self.glow_layer, svg)

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
            knots.setdefault(q0, []).append(float(head.get_center()[0]))
            glow = VGroup(*[Circle(radius=r * self.S).move_to(head.get_center()) for r in (1.1, 1.8, 2.6)])
            for c in glow:
                c.set_fill(self.cfg.active, opacity=0).set_stroke(width=0)
            self.glows.append((glow, t_on, t_off))

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
            g = self.pending_glows.pop(0)
            self.active_glows.append(g)
            self.glow_layer.add(g[0])
        keepg = []
        for g in self.active_glows:
            env = glow_env(t, g[1], g[2], cfg.glow_ramp, cfg.glow_decay) * vis * cfg.glow
            for c, a in zip(g[0], (0.07, 0.04, 0.022)):
                c.set_fill(cfg.active, opacity=a * env)
            if t <= g[2] + cfg.glow_decay:
                keepg.append(g)
            else:
                self.glow_layer.remove(g[0])
        self.active_glows = keepg


@dataclass
class Ink:
    mob: Mobject
    t_on: float
    t_off: float


def soft_light(color: str, radius: float, peak: float, rings: int = 9) -> VGroup:
    """Radial glow built from stacked translucent discs (peak = opacity at the centre)."""
    g = VGroup(*[Circle(radius=radius * (i + 1) / rings) for i in range(rings)])
    for c in g:
        c.set_fill(color, opacity=peak / rings).set_stroke(width=0)
    g.base = peak / rings  # type: ignore[attr-defined]
    g.color_hex = color  # type: ignore[attr-defined]
    return g


def set_light(g: VGroup, k: float) -> None:
    for c in g:
        c.set_fill(g.color_hex, opacity=g.base * k)  # type: ignore[attr-defined]


# ----------------------------------------------------------------- scene ---
class SheetScene(Scene):
    def __init__(self, meta: Meta, eng: Engraving, tempo: TempoMap, cfg: Config, still: float | None = None, **kw):
        super().__init__(**kw)
        self.meta, self.eng, self.tempo, self.cfg, self.still = meta, eng, tempo, cfg, still

    def construct(self) -> None:
        meta, eng, tempo, cfg = self.meta, self.eng, self.tempo, self.cfg
        plans = eng.plans
        left = -cfg.slot_width / 2
        lead = cfg.lead
        self.systems = [
            System(p, eng, tempo, cfg, left, cfg.slot_y if i % 2 == 0 else -cfg.slot_y, lead)
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
            (soft_light(cfg.light_cool, 8.0, 0.55, 22), (4.2, 2.0), (31.0, 43.0), 0.0),
            (soft_light(cfg.light_warm, 6.5, 0.38, 22), (4.8, 1.6), (37.0, 29.0), 2.1),
        ]
        for light, *_ in ambient:
            light.set_z_index(-10)
        cursor_light = soft_light(cfg.cursor, 3.0, cfg.cursor_light, 14)
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

        state = {"t": 0.0}

        def update(_m: Mobject, dt: float) -> None:
            state["t"] += dt
            t = state["t"]
            g = 1 - smooth((t - fade_start) / 2.2)
            ta = smooth((t - 0.6) / 1.4) * (1 - smooth((t - (lead - 3.4)) / 1.4))
            title_group.set_opacity(ta)
            fade_in = smooth(t / 4.0)
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
    cfg: Config | None = None,
) -> Path:
    meta = read_meta(src)
    if title:
        meta.title = title
    cfg = cfg or Config()
    eng = engrave(src, cfg)
    tempo = TempoMap(eng.tempos if meta.has_tempo else [], default_bpm, speed, bpm)
    w, h, f = (1280, 720, 15) if preview else (3840, 2160, 30)
    settings = {
        "pixel_width": w,
        "pixel_height": h,
        "frame_rate": fps or f,
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
        scene = SheetScene(meta, eng, tempo, cfg, still=still)
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
                "subtitle_color", "light_cool", "light_warm", "dust_color")


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
    print(render(a.musicxml, out, a.preview, a.fps, a.bpm, a.speed, a.default_bpm, a.still, a.title, cfg))
    return 0


if __name__ == "__main__":
    sys.exit(main())
