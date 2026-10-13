from __future__ import annotations

from collections import Counter
from functools import reduce
from importlib.machinery import SourceFileLoader
from math import gcd
from pathlib import Path
import random

import numpy as np


def _fallback_get_bg(grid: np.ndarray) -> int:
    return int(Counter(map(int, np.asarray(grid).ravel())).most_common(1)[0][0])


def _fallback_bbox_coords(coords: np.ndarray) -> tuple[int, int, int, int]:
    r0, c0 = coords.min(axis=0)
    r1, c1 = coords.max(axis=0)
    return int(r0), int(c0), int(r1), int(c1)


def _fallback_components_mask(mask: np.ndarray, connectivity: int = 4) -> list[np.ndarray]:
    mask = np.asarray(mask, dtype=bool)
    h, w = mask.shape
    seen = np.zeros_like(mask, dtype=bool)
    if int(connectivity) == 4:
        steps = [(-1, 0), (1, 0), (0, -1), (0, 1)]
    else:
        steps = [(dr, dc) for dr in (-1, 0, 1) for dc in (-1, 0, 1) if (dr, dc) != (0, 0)]

    comps: list[np.ndarray] = []
    for r in range(h):
        for c in range(w):
            if not mask[r, c] or seen[r, c]:
                continue
            stack = [(r, c)]
            seen[r, c] = True
            pts: list[tuple[int, int]] = []
            while stack:
                rr, cc = stack.pop()
                pts.append((rr, cc))
                for dr, dc in steps:
                    nr = rr + dr
                    nc = cc + dc
                    if 0 <= nr < h and 0 <= nc < w and mask[nr, nc] and not seen[nr, nc]:
                        seen[nr, nc] = True
                        stack.append((nr, nc))
            comps.append(np.array(pts, dtype=int))
    return comps


try:
    _arc_utils = SourceFileLoader(
        "_arc_utils_fable_8_033_3",
        str(Path("/home/lixinyu/snap/new/solvers/arc_utils.py")),
    ).load_module()
    get_bg = _arc_utils.get_bg
    bbox_coords = _arc_utils.bbox_coords
    components_mask = _arc_utils.components_mask
except Exception:
    get_bg = _fallback_get_bg
    bbox_coords = _fallback_bbox_coords
    components_mask = _fallback_components_mask


Coord = tuple[int, int]
Block = tuple[int, int, int, int]

_CORNERS = ("tl", "tr", "bl", "br")
_MAX_GEN_RETRIES = 500


def _lattice_step(row_values: list[int], col_values: list[int]) -> int:
    diffs: list[int] = []
    for values in (row_values, col_values):
        uniq = sorted(set(int(v) for v in values))
        diffs.extend(b - a for a, b in zip(uniq, uniq[1:]) if b > a)
    assert diffs, "need at least two lattice coordinates to infer a step"
    return int(reduce(gcd, diffs))


def _extract_structures(grid: np.ndarray) -> tuple[list[tuple[int, int, int]], list[Block]]:
    g = np.asarray(grid, dtype=int)
    bg = int(get_bg(g))
    singles: list[tuple[int, int, int]] = []
    blocks: list[Block] = []
    for pts in components_mask(g != bg, connectivity=4):
        if len(pts) == 1:
            r, c = map(int, pts[0])
            singles.append((r, c, int(g[r, c])))
            continue
        r0, c0, r1, c1 = bbox_coords(pts)
        hh = int(r1 - r0 + 1)
        ww = int(c1 - c0 + 1)
        if hh != ww or hh < 2:
            continue
        sub = g[r0 : r1 + 1, c0 : c1 + 1]
        color = int(sub[0, 0])
        if color == bg:
            continue
        if np.all(sub == color) and int(sub.size) == int(len(pts)):
            blocks.append((int(r0), int(c0), hh, color))
    return singles, blocks


def _legend_data(singles: list[tuple[int, int, int]]) -> tuple[list[tuple[int, int, int]], int, int, int]:
    assert singles, "expected singleton legend pixels"
    anchor_r = min(r for r, _c, _v in singles)
    anchor_c = min(c for _r, c, _v in singles)
    step = _lattice_step([r for r, _c, _v in singles], [c for _r, c, _v in singles])
    pts: list[tuple[int, int, int]] = []
    for r, c, color in singles:
        dr = int(r - anchor_r)
        dc = int(c - anchor_c)
        assert dr % step == 0 and dc % step == 0, "legend pixels must lie on a regular lattice"
        pts.append((dr // step, dc // step, int(color)))
    return pts, int(anchor_r), int(anchor_c), int(step)


def _block_data(blocks: list[Block]) -> tuple[list[Block], int, int]:
    assert blocks, "expected square blocks"
    sizes = [size for _r0, _c0, size, _color in blocks]
    size_counts = Counter(sizes)
    block_size = int(max(size_counts.items(), key=lambda kv: (kv[1], kv[0]))[0])
    chosen = [b for b in blocks if int(b[2]) == block_size]
    assert chosen, "no dominant block size found"
    step = _lattice_step(
        [r0 for r0, _c0, _size, _color in chosen],
        [c0 for _r0, c0, _size, _color in chosen],
    )
    return chosen, block_size, int(step)


def _evaluate_base(
    shape: tuple[int, int],
    legend_pts: list[tuple[int, int, int]],
    legend_anchor: Coord,
    legend_step: int,
    blocks: list[Block],
    block_lookup: dict[tuple[int, int], int],
    block_step: int,
    block_size: int,
    base: Coord,
) -> tuple[tuple[int, int], dict[Coord, int], dict[Coord, int], dict[Coord, int]] | None:
    h, w = map(int, shape)
    base_r, base_c = map(int, base)
    anchor_r, anchor_c = map(int, legend_anchor)

    legend_map: dict[Coord, int] = {}
    for rr, cc, color in legend_pts:
        key = (int(rr), int(cc))
        prev = legend_map.get(key)
        if prev is not None:
            if prev != int(color):
                return None
        else:
            legend_map[key] = int(color)

    block_map: dict[Coord, int] = {}
    for r0, c0, _size, color in blocks:
        dr = int(r0 - base_r)
        dc = int(c0 - base_c)
        if dr % block_step != 0 or dc % block_step != 0:
            return None
        key = (dr // block_step, dc // block_step)
        prev = block_map.get(key)
        if prev is not None and prev != int(color):
            return None
        block_map[key] = int(color)

    union_map = dict(legend_map)
    for key, color in block_map.items():
        prev = union_map.get(key)
        if prev is not None and prev != color:
            return None
        union_map[key] = color

    shared = set(legend_map) & set(block_map)
    if len(shared) < 2:
        return None
    if len({rr for rr, _cc in shared}) < 2 or len({cc for _rr, cc in shared}) < 2:
        return None

    for rr, cc, color in legend_pts:
        tr0 = base_r + int(rr) * block_step
        tc0 = base_c + int(cc) * block_step
        if not (0 <= tr0 <= h - block_size and 0 <= tc0 <= w - block_size):
            return None
        existing = block_lookup.get((tr0, tc0))
        if existing is not None and existing != int(color):
            return None

    for (rr, cc), color in union_map.items():
        pr = anchor_r + int(rr) * legend_step
        pc = anchor_c + int(cc) * legend_step
        tr0 = base_r + int(rr) * block_step
        tc0 = base_c + int(cc) * block_step
        if not (0 <= pr < h and 0 <= pc < w):
            return None
        if not (0 <= tr0 <= h - block_size and 0 <= tc0 <= w - block_size):
            return None
        block_color = block_lookup.get((tr0, tc0))
        if block_color is not None and block_color != color:
            return None

    return (len(shared), len(union_map)), union_map, legend_map, block_map


def _infer_completion(
    grid: np.ndarray,
) -> tuple[dict[Coord, int], Coord, int, Coord, int, int, dict[Coord, int], dict[Coord, int]]:
    g = np.asarray(grid, dtype=int)
    singles, raw_blocks = _extract_structures(g)
    assert singles, "input needs a singleton minimap"
    assert raw_blocks, "input needs square blocks"

    legend_pts, anchor_r, anchor_c, legend_step = _legend_data(singles)
    blocks, block_size, block_step = _block_data(raw_blocks)
    block_lookup = {(int(r0), int(c0)): int(color) for r0, c0, _size, color in blocks}

    by_color: dict[int, list[tuple[int, int]]] = {}
    for r0, c0, _size, color in blocks:
        by_color.setdefault(int(color), []).append((int(r0), int(c0)))

    candidates: Counter[Coord] = Counter()
    for rr, cc, color in legend_pts:
        for r0, c0 in by_color.get(int(color), []):
            candidates[(int(r0 - rr * block_step), int(c0 - cc * block_step))] += 1
    assert candidates, "shared members are required to align the two views"

    valid: list[tuple[tuple[int, int, int], Coord, dict[Coord, int], dict[Coord, int], dict[Coord, int]]] = []
    for base, votes in candidates.items():
        evaluated = _evaluate_base(
            g.shape,
            legend_pts,
            (anchor_r, anchor_c),
            legend_step,
            blocks,
            block_lookup,
            block_step,
            block_size,
            base,
        )
        if evaluated is None:
            continue
        score, union_map, legend_map, block_map = evaluated
        valid.append(((int(votes), int(score[0]), int(score[1])), base, union_map, legend_map, block_map))

    assert valid, "no valid alignment found"
    best_score = max(item[0] for item in valid)
    best = [item for item in valid if item[0] == best_score]
    assert len(best) == 1, "alignment must be uniquely determined"
    _score, base, union_map, legend_map, block_map = best[0]
    return union_map, (anchor_r, anchor_c), legend_step, base, block_step, block_size, legend_map, block_map


def _paint_completion(
    grid: np.ndarray,
    union_map: dict[Coord, int],
    legend_anchor: Coord,
    legend_step: int,
    block_base: Coord,
    block_step: int,
    block_size: int,
) -> np.ndarray:
    out = np.asarray(grid, dtype=int).copy()
    anchor_r, anchor_c = map(int, legend_anchor)
    base_r, base_c = map(int, block_base)
    for (rr, cc), color in union_map.items():
        pr = anchor_r + int(rr) * legend_step
        pc = anchor_c + int(cc) * legend_step
        tr0 = base_r + int(rr) * block_step
        tc0 = base_c + int(cc) * block_step
        out[pr, pc] = int(color)
        out[tr0 : tr0 + block_size, tc0 : tc0 + block_size] = int(color)
    return out


def main(grid: np.ndarray) -> np.ndarray:
    """
    Complete the corner minimap and the large block grid by aligning their shared cells and using each side to fill the other's gaps.

    Input structure:
    - The background is the most common color.
    - One corner contains a minimap made of isolated single pixels on a regular lattice, with some positions missing.
    - Elsewhere the same pattern appears as equally sized solid square blocks on a regular lattice, with other positions missing.
    - Every true pattern position appears in at least one of the two representations, and shared positions keep the same color.

    Rule:
    - Detect the singleton minimap pixels and the solid square blocks.
    - Use the positions that appear on both sides to infer the unique lattice offset from minimap coordinates to block coordinates.
    - Add a solid square for every minimap position whose block is missing, and add a minimap pixel for every block position whose minimap point is missing.
    - Keep all existing pixels and blocks unchanged.

    Key insight:
    - Neither representation is primary: the shared members fix the coordinate mapping, and then the union of both incomplete views gives the full pattern to paint back into both scales.
    """
    union_map, legend_anchor, legend_step, block_base, block_step, block_size, _legend_map, _block_map = _infer_completion(grid)
    return _paint_completion(grid, union_map, legend_anchor, legend_step, block_base, block_step, block_size)


def _paint_input(
    shape: tuple[int, int],
    bg: int,
    colors: dict[Coord, int],
    legend_coords: set[Coord],
    block_coords: set[Coord],
    legend_base: Coord,
    legend_step: int,
    block_base: Coord,
    block_step: int,
    block_size: int,
) -> np.ndarray:
    h, w = map(int, shape)
    grid = np.full((h, w), int(bg), dtype=int)

    for rr, cc in legend_coords:
        pr = int(legend_base[0] + rr * legend_step)
        pc = int(legend_base[1] + cc * legend_step)
        grid[pr, pc] = int(colors[(rr, cc)])

    for rr, cc in block_coords:
        tr0 = int(block_base[0] + rr * block_step)
        tc0 = int(block_base[1] + cc * block_step)
        grid[tr0 : tr0 + block_size, tc0 : tc0 + block_size] = int(colors[(rr, cc)])
    return grid


def _paint_expected(
    grid: np.ndarray,
    colors: dict[Coord, int],
    legend_base: Coord,
    legend_step: int,
    block_base: Coord,
    block_step: int,
    block_size: int,
) -> np.ndarray:
    out = np.asarray(grid, dtype=int).copy()
    for (rr, cc), color in colors.items():
        pr = int(legend_base[0] + rr * legend_step)
        pc = int(legend_base[1] + cc * legend_step)
        tr0 = int(block_base[0] + rr * block_step)
        tc0 = int(block_base[1] + cc * block_step)
        out[pr, pc] = int(color)
        out[tr0 : tr0 + block_size, tc0 : tc0 + block_size] = int(color)
    return out


def _place_legend_base(
    corner: str,
    shape: tuple[int, int],
    rr_min: int,
    rr_max: int,
    cc_min: int,
    cc_max: int,
    step: int,
) -> Coord:
    h, w = map(int, shape)
    margin_a = random.randint(1, 2)
    margin_b = random.randint(1, 2)
    if corner == "tl":
        base_r = margin_a - rr_min * step
        base_c = margin_b - cc_min * step
    elif corner == "tr":
        base_r = margin_a - rr_min * step
        base_c = (w - 1 - margin_b) - cc_max * step
    elif corner == "bl":
        base_r = (h - 1 - margin_a) - rr_max * step
        base_c = margin_b - cc_min * step
    else:
        base_r = (h - 1 - margin_a) - rr_max * step
        base_c = (w - 1 - margin_b) - cc_max * step
    return int(base_r), int(base_c)


def _place_block_base(
    corner: str,
    shape: tuple[int, int],
    rr_min: int,
    rr_max: int,
    cc_min: int,
    cc_max: int,
    step: int,
    block_size: int,
) -> Coord:
    h, w = map(int, shape)
    margin_a = random.randint(1, 2)
    margin_b = random.randint(1, 2)
    if corner == "tl":
        base_r = margin_a - rr_min * step
        base_c = margin_b - cc_min * step
    elif corner == "tr":
        base_r = margin_a - rr_min * step
        base_c = (w - margin_b - block_size) - cc_max * step
    elif corner == "bl":
        base_r = (h - margin_a - block_size) - rr_max * step
        base_c = margin_b - cc_min * step
    else:
        base_r = (h - margin_a - block_size) - rr_max * step
        base_c = (w - margin_b - block_size) - cc_max * step
    return int(base_r), int(base_c)


def _sample_pattern() -> tuple[dict[Coord, int], set[Coord], set[Coord], set[Coord]] | None:
    rows_n = random.randint(3, 4)
    cols_n = random.randint(3, 4)
    row_offset = random.randint(-2, 0)
    col_offset = random.randint(-2, 0)
    all_coords = [(row_offset + r, col_offset + c) for r in range(rows_n) for c in range(cols_n)]

    for _ in range(120):
        occupied = [coord for coord in all_coords if random.random() < 0.72]
        if len(occupied) < 7:
            continue
        rows = {rr for rr, _cc in occupied}
        cols = {cc for _rr, cc in occupied}
        if len(rows) < 3 or len(cols) < 3:
            continue
        if len(occupied) == len(all_coords) and random.random() < 0.7:
            continue

        pool = occupied[:]
        random.shuffle(pool)
        shared_count = random.randint(3, len(pool) - 4)
        legend_only_count = random.randint(2, len(pool) - shared_count - 2)
        shared = set(pool[:shared_count])
        legend_only = set(pool[shared_count : shared_count + legend_only_count])
        block_only = set(pool[shared_count + legend_only_count :])

        if len(block_only) < 2:
            continue
        legend_seen = shared | legend_only
        block_seen = shared | block_only
        if len({rr for rr, _cc in legend_seen}) < 2 or len({cc for _rr, cc in legend_seen}) < 2:
            continue
        if len({rr for rr, _cc in block_seen}) < 2 or len({cc for _rr, cc in block_seen}) < 2:
            continue
        if len({rr for rr, _cc in shared}) < 2 or len({cc for _rr, cc in shared}) < 2:
            continue
        row_min = min(rr for rr, _cc in occupied)
        row_max = max(rr for rr, _cc in occupied)
        col_min = min(cc for _rr, cc in occupied)
        col_max = max(cc for _rr, cc in occupied)
        edge_cells = {
            coord
            for coord in occupied
            if coord[0] in (row_min, row_max) or coord[1] in (col_min, col_max)
        }
        if not (legend_only & edge_cells) or not (block_only & edge_cells):
            continue

        palette_size = random.randint(2, min(4, len(pool)))
        palette = random.sample([c for c in range(10)], palette_size)
        colors: dict[Coord, int] = {}
        color_pool = palette[:]
        extra = len(pool) - len(color_pool)
        if extra > 0:
            color_pool += [random.choice(palette) for _ in range(extra)]
        random.shuffle(color_pool)
        for coord, color in zip(pool, color_pool):
            colors[coord] = int(color)

        if len(set(colors.values())) < 2:
            continue
        if max(Counter(colors.values()).values()) < 2:
            continue
        if len({colors[coord] for coord in shared}) < 2:
            continue
        return colors, shared, legend_only, block_only
    return None


def _gen_attempt() -> np.ndarray | None:
    sampled = _sample_pattern()
    if sampled is None:
        return None
    colors, shared, legend_only, block_only = sampled
    all_coords = set(colors)
    legend_coords = shared | legend_only
    block_coords = shared | block_only

    bg_choices = [c for c in range(10) if c not in set(colors.values())]
    if not bg_choices:
        return None
    bg = random.choice(bg_choices)
    legend_step = random.choice([2, 3])
    block_size = random.choice([2, 3])
    block_gap = random.choice([1, 2])
    block_step = block_size + block_gap

    rr_min = min(rr for rr, _cc in all_coords)
    rr_max = max(rr for rr, _cc in all_coords)
    cc_min = min(cc for _rr, cc in all_coords)
    cc_max = max(cc for _rr, cc in all_coords)
    legend_span_h = (rr_max - rr_min) * legend_step + 1
    legend_span_w = (cc_max - cc_min) * legend_step + 1
    block_span_h = (rr_max - rr_min) * block_step + block_size
    block_span_w = (cc_max - cc_min) * block_step + block_size

    min_h = legend_span_h + block_span_h + 4
    min_w = legend_span_w + block_span_w + 4
    if min_h > 30 or min_w > 30:
        return None
    h = random.randint(min_h, 30)
    w = random.randint(min_w, 30)

    legend_corner = random.choice(_CORNERS)
    opposite = {"tl": "br", "tr": "bl", "bl": "tr", "br": "tl"}[legend_corner]
    legend_base = _place_legend_base(legend_corner, (h, w), rr_min, rr_max, cc_min, cc_max, legend_step)
    block_base = _place_block_base(opposite, (h, w), rr_min, rr_max, cc_min, cc_max, block_step, block_size)

    inp = _paint_input(
        (h, w),
        bg,
        colors,
        legend_coords,
        block_coords,
        legend_base,
        legend_step,
        block_base,
        block_step,
        block_size,
    )
    expected = _paint_expected(inp, colors, legend_base, legend_step, block_base, block_step, block_size)

    if np.array_equal(inp, expected):
        return None

    added_legend = block_only
    added_blocks = legend_only
    if len(added_legend) < 2 or len(added_blocks) < 2:
        return None

    singles, blocks = _extract_structures(inp)
    if len(singles) != len(legend_coords):
        return None
    if len(blocks) != len(block_coords):
        return None

    try:
        solved = main(inp)
        inferred = _infer_completion(inp)
    except AssertionError:
        return None

    if not np.array_equal(solved, expected):
        return None

    union_map, legend_anchor, inf_legend_step, block_base_inf, inf_block_step, inf_block_size, legend_map, block_map = inferred
    if len(union_map) != len(all_coords):
        return None
    if len(legend_map) != len(legend_coords):
        return None
    if len(block_map) != len(block_coords):
        return None
    if inf_legend_step != legend_step or inf_block_step != block_step or inf_block_size != block_size:
        return None
    if len(set(colors.values())) == 1:
        return None

    out = solved
    legend_delta = 0
    block_delta = 0
    for rr, cc in added_legend:
        pr = int(legend_base[0] + rr * legend_step)
        pc = int(legend_base[1] + cc * legend_step)
        legend_delta += int(out[pr, pc] != inp[pr, pc])
    for rr, cc in added_blocks:
        tr0 = int(block_base[0] + rr * block_step)
        tc0 = int(block_base[1] + cc * block_step)
        block_delta += int(np.any(out[tr0 : tr0 + block_size, tc0 : tc0 + block_size] != inp[tr0 : tr0 + block_size, tc0 : tc0 + block_size]))
    if legend_delta == 0 or block_delta == 0:
        return None

    return inp


def generate_input() -> np.ndarray:
    for _ in range(_MAX_GEN_RETRIES):
        grid = _gen_attempt()
        if grid is not None:
            return grid
    raise RuntimeError("failed to generate a valid bidirectional minimap instance")
