from __future__ import annotations

from collections import Counter
from math import gcd
import random

import numpy as np


Coord = tuple[int, int]


def _get_bg(grid: np.ndarray) -> int:
    return int(Counter(map(int, grid.flat)).most_common(1)[0][0])


def _neighbors4(r: int, c: int) -> tuple[Coord, Coord, Coord, Coord]:
    return ((r - 1, c), (r + 1, c), (r, c - 1), (r, c + 1))


def _components(mask: np.ndarray) -> list[np.ndarray]:
    h, w = mask.shape
    seen = np.zeros((h, w), dtype=bool)
    comps: list[np.ndarray] = []
    for r in range(h):
        for c in range(w):
            if not mask[r, c] or seen[r, c]:
                continue
            stack = [(r, c)]
            seen[r, c] = True
            pts: list[Coord] = []
            while stack:
                rr, cc = stack.pop()
                pts.append((rr, cc))
                for nr, nc in _neighbors4(rr, cc):
                    if 0 <= nr < h and 0 <= nc < w and mask[nr, nc] and not seen[nr, nc]:
                        seen[nr, nc] = True
                        stack.append((nr, nc))
            comps.append(np.array(pts, dtype=int))
    return comps


def _bbox(coords: np.ndarray) -> tuple[int, int, int, int]:
    r0, c0 = coords.min(axis=0)
    r1, c1 = coords.max(axis=0)
    return int(r0), int(c0), int(r1), int(c1)


def _positive_gcd(values: list[int]) -> int:
    vals = [abs(int(v)) for v in values if int(v) != 0]
    if not vals:
        return 1
    out = vals[0]
    for v in vals[1:]:
        out = gcd(out, v)
    return int(out)


def _find_legend_points(g: np.ndarray, bg: int) -> list[tuple[int, int, int]]:
    points: list[tuple[int, int, int]] = []
    for comp in _components(g != bg):
        if len(comp) != 1:
            continue
        r, c = map(int, comp[0])
        points.append((r, c, int(g[r, c])))
    assert points, "expected legend singleton pixels"
    return points


def _find_blocks(g: np.ndarray, bg: int) -> list[tuple[int, int, int, int]]:
    blocks: list[tuple[int, int, int, int]] = []
    for comp in _components(g != bg):
        if len(comp) <= 1:
            continue
        r0, c0, r1, c1 = _bbox(comp)
        h = r1 - r0 + 1
        w = c1 - c0 + 1
        if h != w or h < 2:
            continue
        color = int(g[r0, c0])
        if color == bg:
            continue
        patch = g[r0 : r1 + 1, c0 : c1 + 1]
        if np.all(patch == color) and len(comp) == h * w:
            blocks.append((r0, c0, h, color))
    assert blocks, "expected visible square blocks"
    return blocks


def _infer_legend(legend_pts: list[tuple[int, int, int]]) -> tuple[int, int, int, dict[Coord, int]]:
    rows = sorted({r for r, _c, _v in legend_pts})
    cols = sorted({c for _r, c, _v in legend_pts})
    row_diffs = [b - a for a, b in zip(rows, rows[1:])]
    col_diffs = [b - a for a, b in zip(cols, cols[1:])]
    step = _positive_gcd(row_diffs + col_diffs)
    base_r = min(rows)
    base_c = min(cols)
    logical: dict[Coord, int] = {}
    for r, c, color in legend_pts:
        assert (r - base_r) % step == 0 and (c - base_c) % step == 0
        rr = (r - base_r) // step
        cc = (c - base_c) // step
        logical[(int(rr), int(cc))] = int(color)
    return int(base_r), int(base_c), int(step), logical


def _infer_blocks(
    blocks: list[tuple[int, int, int, int]]
) -> tuple[int, int, dict[tuple[int, int], int]]:
    size_counts = Counter(size for _r, _c, size, _color in blocks)
    block_size = int(size_counts.most_common(1)[0][0])
    chosen = [(r, c, color) for r, c, size, color in blocks if size == block_size]
    rows = sorted({r for r, _c, _color in chosen})
    cols = sorted({c for _r, c, _color in chosen})
    row_diffs = [b - a for a, b in zip(rows, rows[1:])]
    col_diffs = [b - a for a, b in zip(cols, cols[1:])]
    step = _positive_gcd(row_diffs + col_diffs)
    assert step > block_size, "block lattice needs positive gaps"
    lookup = {(int(r), int(c)): int(color) for r, c, color in chosen}
    return block_size, int(step), lookup


def _candidate_scene(
    shape: tuple[int, int],
    legend_base: Coord,
    legend_step: int,
    legend_map: dict[Coord, int],
    block_size: int,
    block_step: int,
    block_lookup: dict[Coord, int],
    base: Coord,
) -> dict[Coord, int] | None:
    h, w = shape
    base_r, base_c = base
    merged = dict(legend_map)
    overlap = 0

    for (r0, c0), color in block_lookup.items():
        dr = r0 - base_r
        dc = c0 - base_c
        if dr < 0 or dc < 0 or dr % block_step != 0 or dc % block_step != 0:
            return None
        rr = dr // block_step
        cc = dc // block_step
        key = (int(rr), int(cc))
        prev = merged.get(key)
        if prev is not None:
            if prev != color:
                return None
            overlap += 1
        merged[key] = int(color)

    if overlap == 0:
        return None

    leg_r0, leg_c0 = legend_base
    for rr, cc in merged:
        pr = leg_r0 + rr * legend_step
        pc = leg_c0 + cc * legend_step
        br = base_r + rr * block_step
        bc = base_c + cc * block_step
        if not (0 <= pr < h and 0 <= pc < w):
            return None
        if br < 0 or bc < 0 or br + block_size > h or bc + block_size > w:
            return None
    return merged


def _infer_scene(grid: np.ndarray) -> tuple[dict[Coord, int], tuple[int, int, int], tuple[int, int, int, int]]:
    g = np.asarray(grid, dtype=int)
    bg = _get_bg(g)
    legend_pts = _find_legend_points(g, bg)
    blocks = _find_blocks(g, bg)

    leg_r0, leg_c0, legend_step, legend_map = _infer_legend(legend_pts)
    block_size, block_step, block_lookup = _infer_blocks(blocks)

    by_color: dict[int, list[Coord]] = {}
    for pos, color in block_lookup.items():
        by_color.setdefault(int(color), []).append(pos)

    candidates: dict[Coord, dict[Coord, int]] = {}
    for (rr, cc), color in legend_map.items():
        for br, bc in by_color.get(int(color), []):
            base = (int(br - rr * block_step), int(bc - cc * block_step))
            if base in candidates:
                continue
            scene = _candidate_scene(
                g.shape,
                (leg_r0, leg_c0),
                legend_step,
                legend_map,
                block_size,
                block_step,
                block_lookup,
                base,
            )
            if scene is not None:
                candidates[base] = scene

    assert candidates, "no valid legend-to-block alignment found"
    best_overlap = max(
        sum(
            1
            for (rr, cc), color in legend_map.items()
            if block_lookup.get((base[0] + rr * block_step, base[1] + cc * block_step)) == color
        )
        for base in candidates
    )
    best = [
        (base, scene)
        for base, scene in candidates.items()
        if sum(
            1
            for (rr, cc), color in legend_map.items()
            if block_lookup.get((base[0] + rr * block_step, base[1] + cc * block_step)) == color
        )
        == best_overlap
    ]
    assert len(best) == 1, "alignment must be unique"
    (base_r, base_c), scene = best[0]
    return scene, (leg_r0, leg_c0, legend_step), (base_r, base_c, block_size, block_step)


def main(grid: np.ndarray) -> np.ndarray:
    """
    Complete both the top-left minimap and the main block mosaic by aligning their shared visible cells.

    Input structure:
    - The background is the most common color.
    - In the top-left corner, a regular lattice of single pixels is a minimap copy of a larger colored block layout, but some minimap cells are erased to background.
    - Elsewhere, the same layout appears as equally sized solid square blocks on a regular grid, with some blocks erased to background.

    Rule:
    - Infer the minimap lattice step and the main block size/step from the visible cells.
    - Use same-colored cells visible in both copies to determine the unique translation from minimap coordinates to block coordinates.
    - Treat every logical position visible in either copy as part of the shared layout, then fill the missing minimap pixels and missing main blocks with the corresponding colors.

    Key insight:
    - Neither representation is complete on its own; the two partial copies must first be aligned, then each one repairs the other.
    """
    g = np.asarray(grid, dtype=int)
    out = g.copy()
    scene, (leg_r0, leg_c0, legend_step), (base_r, base_c, block_size, block_step) = _infer_scene(g)

    for (rr, cc), color in scene.items():
        pr = leg_r0 + rr * legend_step
        pc = leg_c0 + cc * legend_step
        br = base_r + rr * block_step
        bc = base_c + cc * block_step
        out[pr, pc] = color
        out[br : br + block_size, bc : bc + block_size] = color
    return out


_GRID_LIMIT = 30
_LEGEND_STEP_CHOICES = (2, 3)
_ROWS_RANGE = (3, 4)
_COLS_RANGE = (3, 4)
_BLOCK_SIZE_CHOICES = (2, 3)
_SEPARATION_RANGE = (2, 4)
_MARGIN_RANGE = (1, 2)
_PALETTE_SIZE_RANGE = (3, 5)
_RETRY_LIMIT = 400


def _sample_color_layout(rows: int, cols: int, palette: list[int]) -> np.ndarray:
    while True:
        arr = np.random.choice(palette, size=(rows, cols), replace=True).astype(int)
        used = set(map(int, arr.flat))
        if len(used) < min(len(palette), 3):
            continue
        if len({int(arr[r, c]) for r in range(min(rows, 2)) for c in range(min(cols, 2))}) < 3:
            continue
        return arr


def _visibility_masks(rows: int, cols: int) -> tuple[np.ndarray, np.ndarray] | None:
    total = rows * cols
    cells = [(r, c) for r in range(rows) for c in range(cols)]
    for _ in range(120):
        states = np.random.choice(3, size=(rows, cols), p=(0.45, 0.275, 0.275))
        legend_vis = states != 2
        main_vis = states != 1
        overlap = states == 0
        if np.count_nonzero(overlap) < max(3, total // 3):
            continue
        if not np.any(states == 1) or not np.any(states == 2):
            continue
        if any(not np.any(legend_vis[r, :]) for r in range(rows)):
            continue
        if any(not np.any(legend_vis[:, c]) for c in range(cols)):
            continue
        if any(not np.any(main_vis[r, :]) for r in range(rows)):
            continue
        if any(not np.any(main_vis[:, c]) for c in range(cols)):
            continue
        if len({r for r, c in cells if overlap[r, c]}) < 2 or len({c for r, c in cells if overlap[r, c]}) < 2:
            continue
        return legend_vis, main_vis
    return None


def _render_scene(
    rows: int,
    cols: int,
    colors: np.ndarray,
    bg: int,
    legend_step: int,
    legend_origin: Coord,
    block_size: int,
    block_origin: Coord,
) -> np.ndarray:
    block_step = block_size + 1
    max_r = max(legend_origin[0] + (rows - 1) * legend_step, block_origin[0] + (rows - 1) * block_step + block_size - 1)
    max_c = max(legend_origin[1] + (cols - 1) * legend_step, block_origin[1] + (cols - 1) * block_step + block_size - 1)
    h = max_r + random.randint(*_MARGIN_RANGE) + 1
    w = max_c + random.randint(*_MARGIN_RANGE) + 1
    grid = np.full((h, w), bg, dtype=int)
    for r in range(rows):
        for c in range(cols):
            color = int(colors[r, c])
            lr = legend_origin[0] + r * legend_step
            lc = legend_origin[1] + c * legend_step
            br = block_origin[0] + r * block_step
            bc = block_origin[1] + c * block_step
            grid[lr, lc] = color
            grid[br : br + block_size, bc : bc + block_size] = color
    return grid


def _make_pair() -> tuple[np.ndarray, np.ndarray] | None:
    rows = random.randint(*_ROWS_RANGE)
    cols = random.randint(*_COLS_RANGE)
    legend_step = random.choice(_LEGEND_STEP_CHOICES)
    block_size = random.choice(_BLOCK_SIZE_CHOICES)
    block_step = block_size + 1
    legend_origin = (1, 1)

    legend_r_max = legend_origin[0] + (rows - 1) * legend_step
    legend_c_max = legend_origin[1] + (cols - 1) * legend_step
    base_r = legend_r_max + random.randint(*_SEPARATION_RANGE)
    base_c = legend_c_max + random.randint(*_SEPARATION_RANGE)

    end_r = base_r + (rows - 1) * block_step + block_size
    end_c = base_c + (cols - 1) * block_step + block_size
    if end_r + _MARGIN_RANGE[1] > _GRID_LIMIT or end_c + _MARGIN_RANGE[1] > _GRID_LIMIT:
        return None

    bg = random.randint(0, 9)
    palette_pool = [c for c in range(10) if c != bg]
    palette = random.sample(palette_pool, random.randint(*_PALETTE_SIZE_RANGE))
    colors = _sample_color_layout(rows, cols, palette)

    visibility = _visibility_masks(rows, cols)
    if visibility is None:
        return None
    legend_vis, main_vis = visibility

    out = _render_scene(rows, cols, colors, bg, legend_step, legend_origin, block_size, (base_r, base_c))
    inp = out.copy()
    for r in range(rows):
        for c in range(cols):
            lr = legend_origin[0] + r * legend_step
            lc = legend_origin[1] + c * legend_step
            br = base_r + r * block_step
            bc = base_c + c * block_step
            if not legend_vis[r, c]:
                inp[lr, lc] = bg
            if not main_vis[r, c]:
                inp[br : br + block_size, bc : bc + block_size] = bg

    if _get_bg(inp) != bg or _get_bg(out) != bg:
        return None
    if np.array_equal(inp, out):
        return None
    if np.count_nonzero(inp != bg) < 8:
        return None
    return inp, out


def generate_input() -> np.ndarray:
    for _ in range(_RETRY_LIMIT):
        pair = _make_pair()
        if pair is None:
            continue
        inp, out = pair
        try:
            pred = main(inp)
        except AssertionError:
            continue
        if pred.shape != out.shape or not np.array_equal(pred, out):
            continue
        return inp
    raise RuntimeError("failed to generate a solver-consistent sample")
