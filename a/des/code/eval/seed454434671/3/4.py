from __future__ import annotations

import math
import random

import numpy as np


BG = 0
YELLOW = 4
AZURE = 8
TEMPLATE_COLORS = (1, 2, 3, 5, 6, 7, 9)
GEN_RETRIES = 400
UNIT_CHOICES = (4, 6)


def _tick_spacing(ticks: list[int]) -> int:
    ticks = sorted(set(int(x) for x in ticks))
    diffs = [b - a for a, b in zip(ticks, ticks[1:]) if b - a > 0]
    if not diffs:
        raise ValueError("need at least two ruler ticks")
    spacing = diffs[0]
    for diff in diffs[1:]:
        spacing = math.gcd(spacing, diff)
    if spacing <= 0:
        raise ValueError("invalid tick spacing")
    return int(spacing)


def _template_bbox(zone: np.ndarray) -> tuple[int, int, int, int]:
    mask = (zone != BG) & (zone != AZURE) & (zone != YELLOW)
    pts = np.argwhere(mask)
    if pts.size == 0:
        raise ValueError("template not found")
    r0, c0 = pts.min(axis=0)
    r1, c1 = pts.max(axis=0)
    return int(r0), int(c0), int(r1), int(c1)


def _scaled_template(grid: np.ndarray, unit: int) -> np.ndarray:
    zone = np.asarray(grid[1 : 1 + unit, 1 : 1 + unit], dtype=int)
    r0, c0, r1, c1 = _template_bbox(zone)
    template = zone[r0 : r1 + 1, c0 : c1 + 1]
    th, tw = map(int, template.shape)
    if th != tw:
        raise ValueError("template must have square bbox")
    if unit % th != 0:
        raise ValueError("unit must be divisible by template side")
    scale = unit // th
    return np.repeat(np.repeat(template, scale, axis=0), scale, axis=1)


def main(grid: np.ndarray) -> np.ndarray:
    g = np.asarray(grid, dtype=int)
    if g.ndim != 2:
        raise ValueError("expected a 2D grid")

    h, w = g.shape
    top_ticks = [c for c in range(w) if int(g[0, c]) == AZURE]
    left_ticks = [r for r in range(h) if int(g[r, 0]) == AZURE]
    unit = _tick_spacing(top_ticks)
    if _tick_spacing(left_ticks) != unit:
        raise ValueError("top and left rulers disagree")

    stamp = _scaled_template(g, unit)
    sh, sw = map(int, stamp.shape)

    out = np.full_like(g, BG)
    for r, c in np.argwhere(g == YELLOW):
        rr = int(r)
        cc = int(c)
        r1 = min(h, rr + sh)
        c1 = min(w, cc + sw)
        view = stamp[: r1 - rr, : c1 - cc]
        mask = view != BG
        out[rr:r1, cc:c1][mask] = view[mask]

    return out


def _base_pattern(side: int, kind: str) -> tuple[np.ndarray, np.ndarray]:
    mask = np.zeros((side, side), dtype=bool)
    accent = np.zeros((side, side), dtype=bool)

    if kind == "L":
        mask[:, 0] = True
        mask[-1, :] = True
        accent[-1, :] = True
    elif kind == "J":
        mask[:, -1] = True
        mask[0, :] = True
        accent[:, -1] = True
    elif kind == "T":
        mask[0, :] = True
        mask[:, side // 2] = True
        accent[:, side // 2] = True
    elif kind == "PLUS":
        mask[side // 2, :] = True
        mask[:, side // 2] = True
        accent[side // 2, :] = True
    elif kind == "FRAME":
        mask[0, :] = True
        mask[-1, :] = True
        mask[:, 0] = True
        mask[:, -1] = True
        accent[:, -1] = True
    elif kind == "STAIR":
        for r in range(side):
            mask[r, : r + 1] = True
            accent[r, r] = True
    elif kind == "CROSSBAR":
        mask[:, 0] = True
        mask[side // 2, :] = True
        accent[side // 2, :] = True
    else:
        raise ValueError(kind)

    return mask, accent & mask


def _make_template(side: int, rng: random.Random) -> np.ndarray:
    kinds = ["L", "J", "T", "PLUS", "FRAME", "STAIR", "CROSSBAR"]
    kind = rng.choice(kinds)
    mask, accent = _base_pattern(side, kind)

    rot = rng.randrange(4)
    mask = np.rot90(mask, rot)
    accent = np.rot90(accent, rot)

    patch = np.full((side, side), BG, dtype=int)
    primary = rng.choice(TEMPLATE_COLORS)
    patch[mask] = primary

    use_two_colors = bool(rng.randrange(2)) and accent.any()
    if use_two_colors:
        secondary_choices = [c for c in TEMPLATE_COLORS if c != primary]
        secondary = rng.choice(secondary_choices)
        patch[accent] = secondary

    return patch


def _rects_overlap(a: tuple[int, int, int, int], b: tuple[int, int, int, int], gap: int = 0) -> bool:
    ar0, ac0, ar1, ac1 = a
    br0, bc0, br1, bc1 = b
    return not (
        ar1 + gap < br0
        or br1 + gap < ar0
        or ac1 + gap < bc0
        or bc1 + gap < ac0
    )


def _pick_markers(h: int, w: int, unit: int, count: int, rng: random.Random) -> list[tuple[int, int]] | None:
    candidates = [(r, c) for r in range(1, h - unit + 1) for c in range(1, w - unit + 1)]
    rng.shuffle(candidates)

    forbidden = [(1, 1, unit, unit)]
    chosen: list[tuple[int, int]] = []
    rects: list[tuple[int, int, int, int]] = []

    for r, c in candidates:
        rect = (r, c, r + unit - 1, c + unit - 1)
        if any(_rects_overlap(rect, other, gap=1) for other in forbidden):
            continue
        if any(_rects_overlap(rect, other, gap=1) for other in rects):
            continue
        chosen.append((r, c))
        rects.append(rect)
        if len(chosen) == count:
            return chosen

    return None


def _gen_attempt(rng: random.Random) -> np.ndarray | None:
    unit = rng.choice(UNIT_CHOICES)
    scale_branch = rng.randrange(2)
    if scale_branch == 0:
        template_side = unit
    else:
        template_side = rng.choice([d for d in range(2, unit) if unit % d == 0])

    row_units = rng.randint(4, 6)
    col_units = rng.randint(4, 6)
    h = row_units * unit + rng.randint(1, unit)
    w = col_units * unit + rng.randint(1, unit)

    grid = np.full((h, w), BG, dtype=int)
    grid[0, 0:w:unit] = AZURE
    grid[0:h:unit, 0] = AZURE

    template = _make_template(template_side, rng)
    max_off = unit - template_side
    off_r = rng.randint(0, max_off)
    off_c = rng.randint(0, max_off)
    r0 = 1 + off_r
    c0 = 1 + off_c
    grid[r0 : r0 + template_side, c0 : c0 + template_side] = template

    marker_count = rng.randint(1, 4)
    markers = _pick_markers(h, w, unit, marker_count, rng)
    if markers is None:
        return None

    for r, c in markers:
        grid[r, c] = YELLOW

    return grid


def generate_input() -> np.ndarray:
    rng = random.Random()
    for _ in range(GEN_RETRIES):
        grid = _gen_attempt(rng)
        if grid is None:
            continue
        try:
            out = main(grid)
        except Exception:
            continue
        if out.shape != grid.shape:
            continue
        if not np.any(out != BG):
            continue
        return grid
    raise RuntimeError("failed to generate a valid input")
