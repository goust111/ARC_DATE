from __future__ import annotations

from collections import deque
import random

import numpy as np


BG = 0
COLORS = tuple(range(1, 10))
SIDES = ("top", "bottom", "left", "right")
GEN_RETRIES = 500


def _neighbors4(r: int, c: int):
    yield r - 1, c
    yield r + 1, c
    yield r, c - 1
    yield r, c + 1


def _color_components(g: np.ndarray, color: int) -> list[list[tuple[int, int]]]:
    h, w = g.shape
    seen = np.zeros((h, w), dtype=bool)
    comps: list[list[tuple[int, int]]] = []
    for r, c in np.argwhere(g == color):
        r, c = int(r), int(c)
        if seen[r, c]:
            continue
        q = deque([(r, c)])
        seen[r, c] = True
        comp: list[tuple[int, int]] = []
        while q:
            cr, cc = q.popleft()
            comp.append((cr, cc))
            for nr, nc in _neighbors4(cr, cc):
                if 0 <= nr < h and 0 <= nc < w and not seen[nr, nc] and int(g[nr, nc]) == color:
                    seen[nr, nc] = True
                    q.append((nr, nc))
        comps.append(comp)
    return comps


def _boundary_cells(r0: int, c0: int, r1: int, c1: int) -> list[tuple[int, int]]:
    return [
        (r, c)
        for r in range(r0, r1 + 1)
        for c in range(c0, c1 + 1)
        if r in (r0, r1) or c in (c0, c1)
    ]


def _gap_side(gap: tuple[int, int], r0: int, c0: int, r1: int, c1: int) -> str:
    gr, gc = gap
    on_top = gr == r0
    on_bottom = gr == r1
    on_left = gc == c0
    on_right = gc == c1
    assert sum((on_top, on_bottom, on_left, on_right)) == 1, "gap must be on exactly one side"
    if on_top:
        return "top"
    if on_bottom:
        return "bottom"
    if on_left:
        return "left"
    return "right"


def _filled_sprite(
    h: int,
    w: int,
    frame_color: int,
    seed_color: int,
    gap: tuple[int, int],
    seed: tuple[int, int],
) -> tuple[np.ndarray, int]:
    sprite = np.zeros((h, w), dtype=int)
    r0, c0, r1, c1 = 0, 0, h - 1, w - 1
    for r, c in _boundary_cells(r0, c0, r1, c1):
        if (r, c) != gap:
            sprite[r, c] = frame_color

    side = _gap_side(gap, r0, c0, r1, c1)
    sr, sc = seed
    if side == "top":
        rows = range(sr, h - 1)
        cols = range(1, w - 1)
    elif side == "bottom":
        rows = range(1, sr + 1)
        cols = range(1, w - 1)
    elif side == "left":
        rows = range(1, h - 1)
        cols = range(sc, w - 1)
    else:
        rows = range(1, h - 1)
        cols = range(1, sc + 1)

    area = 0
    for r in rows:
        for c in cols:
            sprite[r, c] = seed_color
            area += 1
    return sprite, area


def _frame_from_component(g: np.ndarray, comp: list[tuple[int, int]], color: int):
    rows = [r for r, _c in comp]
    cols = [c for _r, c in comp]
    r0, r1 = min(rows), max(rows)
    c0, c1 = min(cols), max(cols)
    h, w = r1 - r0 + 1, c1 - c0 + 1
    if h < 3 or w < 3:
        return None

    boundary = set(_boundary_cells(r0, c0, r1, c1))
    comp_set = set(comp)
    if any(cell not in boundary for cell in comp_set):
        return None

    missing = [cell for cell in boundary if int(g[cell]) != color]
    if len(missing) != 1:
        return None
    expected = boundary - {missing[0]}
    if comp_set != expected:
        return None

    interior = [
        (r, c)
        for r in range(r0 + 1, r1)
        for c in range(c0 + 1, c1)
        if int(g[r, c]) != BG
    ]
    if len(interior) != 1:
        return None

    seed_r, seed_c = interior[0]
    seed_color = int(g[seed_r, seed_c])
    if seed_color == color:
        return None

    local_gap = (missing[0][0] - r0, missing[0][1] - c0)
    local_seed = (seed_r - r0, seed_c - c0)
    sprite, area = _filled_sprite(h, w, int(color), seed_color, local_gap, local_seed)
    return {
        "bbox": (r0, c0, r1, c1),
        "sprite": sprite,
        "area": int(area),
    }


def _extract_frames(g: np.ndarray) -> list[dict]:
    frames: list[dict] = []
    for color in sorted(int(v) for v in np.unique(g) if int(v) != BG):
        for comp in _color_components(g, color):
            item = _frame_from_component(g, comp, color)
            if item is not None:
                frames.append(item)
    frames.sort(key=lambda item: item["bbox"])
    assert frames, "expected at least one gapped rectangular frame"
    return frames


def main(grid: np.ndarray) -> np.ndarray:
    """
    Stack gapped rectangular frames after growing each inner seed away from its gap.

    Input structure:
    - The background is black.
    - The grid contains several one-pixel-thick monochrome rectangular frames.
    - Each frame has exactly one single-cell gap on one side and exactly one
      differently colored seed pixel in its interior.

    Rule:
    - For each frame, locate the side that contains the gap.
    - From the seed row or column, fill full interior rows or columns in the
      direction opposite the gap, using the seed color.
    - Count the filled seed-color cells as that frame's grown area.
    - Make a black canvas large enough for the largest frame dimensions, then
      paste all grown frames with top-left corners aligned, from smaller grown
      area to larger grown area; later non-black pixels cover earlier ones.

    Key insight:
    - The gap side encodes the reversed growth direction, and the resulting
      filled area gives the stacking order.
    """
    g = np.asarray(grid, dtype=int)
    assert g.ndim == 2, "expected a 2D grid"

    frames = _extract_frames(g)
    out_h = max(int(item["sprite"].shape[0]) for item in frames)
    out_w = max(int(item["sprite"].shape[1]) for item in frames)
    out = np.zeros((out_h, out_w), dtype=int)

    for item in sorted(frames, key=lambda x: (int(x["area"]), x["bbox"])):
        sprite = item["sprite"]
        h, w = sprite.shape
        mask = sprite != BG
        out[:h, :w][mask] = sprite[mask]
    return out


def _fill_area(h: int, w: int, side: str, seed: tuple[int, int]) -> int:
    sr, sc = seed
    if side == "top":
        return (h - 1 - sr) * (w - 2)
    if side == "bottom":
        return sr * (w - 2)
    if side == "left":
        return (h - 2) * (w - 1 - sc)
    return (h - 2) * sc


def _make_spec(max_h: int, max_w: int, force_max: bool = False):
    h = max_h if force_max else random.randint(4, max_h)
    w = max_w if force_max else random.randint(4, max_w)
    side = random.choice(SIDES)
    if side in ("top", "bottom"):
        gap = (0 if side == "top" else h - 1, random.randint(1, w - 2))
    else:
        gap = (random.randint(1, h - 2), 0 if side == "left" else w - 1)
    seed = (random.randint(1, h - 2), random.randint(1, w - 2))
    area = _fill_area(h, w, side, seed)
    return {"h": h, "w": w, "side": side, "gap": gap, "seed": seed, "area": area}


def _specs(n: int):
    max_h = random.randint(6, 10)
    max_w = random.randint(6, 10)
    specs = [_make_spec(max_h, max_w, force_max=True)]
    areas = {specs[0]["area"]}
    for _ in range(n - 1):
        for _attempt in range(80):
            spec = _make_spec(max_h, max_w)
            if spec["area"] not in areas:
                specs.append(spec)
                areas.add(spec["area"])
                break
        else:
            return None
    random.shuffle(specs)
    return specs


def _fits(occupied: np.ndarray, r0: int, c0: int, h: int, w: int, pad: int = 1) -> bool:
    gh, gw = occupied.shape
    rr0 = max(0, r0 - pad)
    cc0 = max(0, c0 - pad)
    rr1 = min(gh, r0 + h + pad)
    cc1 = min(gw, c0 + w + pad)
    return not bool(occupied[rr0:rr1, cc0:cc1].any())


def _place_specs(grid: np.ndarray, specs: list[dict], frame_colors: list[int], seed_colors: list[int]) -> bool:
    h, w = grid.shape
    occupied = np.zeros((h, w), dtype=bool)
    order = list(range(len(specs)))
    random.shuffle(order)
    for i in order:
        spec = specs[i]
        sh, sw = int(spec["h"]), int(spec["w"])
        positions = [
            (r, c)
            for r in range(1, h - sh)
            for c in range(1, w - sw)
            if _fits(occupied, r, c, sh, sw)
        ]
        if not positions:
            return False
        r0, c0 = random.choice(positions)
        local_gap = spec["gap"]
        gap = (r0 + local_gap[0], c0 + local_gap[1])

        for r, c in _boundary_cells(r0, c0, r0 + sh - 1, c0 + sw - 1):
            if (r, c) != gap:
                grid[r, c] = frame_colors[i]

        sr, sc = spec["seed"]
        grid[r0 + sr, c0 + sc] = seed_colors[i]
        occupied[r0 : r0 + sh, c0 : c0 + sw] = True
    return True


def _gen_attempt() -> np.ndarray | None:
    n = random.randint(2, 4)
    specs = _specs(n)
    if specs is None:
        return None

    max_h = max(int(spec["h"]) for spec in specs)
    max_w = max(int(spec["w"]) for spec in specs)
    grid_h = random.randint(max(12, max_h + 4), 30)
    grid_w = random.randint(max(12, max_w + 4), 30)
    grid = np.zeros((grid_h, grid_w), dtype=int)

    chosen = random.sample(COLORS, 2 * n)
    frame_colors = chosen[:n]
    seed_colors = chosen[n:]
    if not _place_specs(grid, specs, frame_colors, seed_colors):
        return None

    try:
        out = main(grid)
    except Exception:
        return None
    if out.ndim != 2 or max(out.shape) > 30:
        return None
    if len({int(spec["area"]) for spec in specs}) != n:
        return None
    return grid


def generate_input() -> np.ndarray:
    for _ in range(GEN_RETRIES):
        grid = _gen_attempt()
        if grid is not None:
            return grid
    raise RuntimeError("failed to generate a valid seed1030756833/3/1 input")
