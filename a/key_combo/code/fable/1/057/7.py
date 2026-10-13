from __future__ import annotations

import random
import sys
from collections import Counter
from pathlib import Path

import numpy as np

_ARC_UTILS_DIR = Path("/home/lixinyu/snap/new/solvers")
if _ARC_UTILS_DIR.is_dir():
    arc_utils_path = str(_ARC_UTILS_DIR)
    if arc_utils_path not in sys.path:
        sys.path.append(arc_utils_path)

try:
    from arc_utils import Color
except Exception:
    class Color:
        BLACK = 0
        BLUE = 1
        RED = 2
        GREEN = 3
        YELLOW = 4
        GREY = 5
        PINK = 6
        ORANGE = 7
        TEAL = 8
        MAROON = 9
        ALL_COLORS = [BLACK, BLUE, RED, GREEN, YELLOW, GREY, PINK, ORANGE, TEAL, MAROON]


PALETTE = list(Color.ALL_COLORS)
GRID_NS = (3, 4)
GEN_ATTEMPTS = 500
SHARED_MASK_BRANCH_PROB = 0.45
COLOR_ONLY_BRANCH_PROB = 0.6


def _pattern_key(arr: np.ndarray) -> tuple[int, ...]:
    return tuple(int(v) for v in arr.ravel())


def _mask_key(mask: np.ndarray) -> tuple[int, ...]:
    return tuple(int(v) for v in mask.astype(int).ravel())


def _mode(values: np.ndarray) -> int:
    return int(Counter(int(v) for v in values.ravel()).most_common(1)[0][0])


def _infer_layout(grid: np.ndarray) -> tuple[int, int, tuple[int, ...]]:
    g = np.asarray(grid, dtype=int)
    assert g.ndim == 2 and g.shape[0] == g.shape[1], "expected a square 2D grid"
    size = int(g.shape[0])

    candidates: list[tuple[int, int, tuple[int, ...]]] = []
    for color in np.unique(g):
        sep_rows = [r for r in range(size) if np.all(g[r, :] == color)]
        sep_cols = [c for c in range(size) if np.all(g[:, c] == color)]
        if sep_rows != sep_cols or len(sep_rows) < 3:
            continue
        n = len(sep_rows) - 1
        step = sep_rows[1] - sep_rows[0]
        expected = tuple(range(0, size, step))
        if expected[-1] != size - 1:
            continue
        if tuple(sep_rows) != expected or tuple(sep_cols) != expected:
            continue
        if step != n + 1:
            continue
        if size != n * step + 1:
            continue
        candidates.append((int(color), n, expected))

    assert len(candidates) == 1, "expected a unique separator lattice"
    return candidates[0]


def _extract_cells(grid: np.ndarray, borders: tuple[int, ...]) -> list[list[np.ndarray]]:
    n = len(borders) - 1
    cells: list[list[np.ndarray]] = []
    for i in range(n):
        row: list[np.ndarray] = []
        for j in range(n):
            cell = grid[borders[i] + 1 : borders[i + 1], borders[j] + 1 : borders[j + 1]].copy()
            assert cell.shape == (n, n), "every cell interior must be N x N"
            row.append(cell)
        cells.append(row)
    return cells


def _analyze_grid(grid: np.ndarray) -> dict:
    g = np.asarray(grid, dtype=int)
    sep_color, n, borders = _infer_layout(g)
    cells = _extract_cells(g, borders)

    pattern_counts: Counter[tuple[int, ...]] = Counter()
    pattern_arrays: dict[tuple[int, ...], np.ndarray] = {}
    cell_keys: list[list[tuple[int, ...]]] = []
    for i in range(n):
        row_keys: list[tuple[int, ...]] = []
        for j in range(n):
            key = _pattern_key(cells[i][j])
            row_keys.append(key)
            pattern_counts[key] += 1
            pattern_arrays.setdefault(key, cells[i][j])
        cell_keys.append(row_keys)

    assert len(pattern_counts) >= 2, "expected a majority pattern and at least one deviant"
    majority_count = max(pattern_counts.values())
    majority_keys = [key for key, count in pattern_counts.items() if count == majority_count]
    assert len(majority_keys) == 1, "majority pattern must be unique"
    majority_key = majority_keys[0]
    majority = pattern_arrays[majority_key]

    vals, counts = np.unique(majority, return_counts=True)
    assert int(np.sum(counts == counts.max())) == 1, "majority pattern must have a unique background color"
    cell_bg = int(vals[int(np.argmax(counts))])
    majority_mask = majority != cell_bg

    deviants = [(i, j) for i in range(n) for j in range(n) if cell_keys[i][j] != majority_key]
    assert len(deviants) >= 2, "at least two deviant cells are required"
    deviant_keys = [cell_keys[i][j] for i, j in deviants]
    assert len(deviant_keys) == len(set(deviant_keys)), "deviant patterns must all be different"

    dev_mask = np.zeros((n, n), dtype=bool)
    for i, j in deviants:
        dev_mask[i, j] = True

    keep_candidates = []
    for i, j in deviants:
        if np.array_equal(cells[i][j] != cell_bg, dev_mask):
            keep_candidates.append((i, j))
    assert len(keep_candidates) == 1, "expected exactly one self-descriptive deviant"

    return {
        "grid": g,
        "separator": sep_color,
        "n": n,
        "borders": borders,
        "cells": cells,
        "cell_keys": cell_keys,
        "majority": majority,
        "majority_key": majority_key,
        "majority_mask": majority_mask,
        "cell_bg": cell_bg,
        "deviants": deviants,
        "dev_mask": dev_mask,
        "keep": keep_candidates[0],
    }


def main(grid: np.ndarray) -> np.ndarray:
    """
    Restore every deviant cell except the one whose inner mask lists all deviant positions.

    Input structure:
    - The canvas is partitioned by a one-pixel separator lattice into N x N cells.
    - Each cell interior is itself an N x N pixel patch.
    - Most cell interiors are exactly identical; the remaining deviant cells all differ from that majority pattern and from each other.

    Rule:
    - Identify the exact majority cell pattern and mark every coarse-grid cell whose interior differs from it.
    - This produces an N x N deviant-position map on the coarse cell lattice.
    - Among the deviant cells, find the unique one whose non-background pixel positions inside its own N x N interior exactly equal that coarse deviant-position map.
    - Rewrite every other deviant cell back to the majority pattern, while leaving the self-descriptive deviant and all separator lines unchanged.

    Key insight:
    - Cell interiors and coarse cell positions share the same N x N coordinate system, so one deviant cell can serve as a cross-scale roster of all deviant locations, including itself.
    """
    info = _analyze_grid(np.asarray(grid, dtype=int))
    out = info["grid"].copy()
    n = info["n"]
    majority = info["majority"]
    keep = info["keep"]

    for i, j in info["deviants"]:
        if (i, j) == keep:
            continue
        r0 = info["borders"][i] + 1
        c0 = info["borders"][j] + 1
        out[r0 : r0 + n, c0 : c0 + n] = majority
    return out


def _render_grid(n: int, separator: int, cells: list[list[np.ndarray]]) -> np.ndarray:
    step = n + 1
    size = n * step + 1
    grid = np.full((size, size), separator, dtype=int)
    for i in range(n):
        for j in range(n):
            r0 = i * step + 1
            c0 = j * step + 1
            grid[r0 : r0 + n, c0 : c0 + n] = cells[i][j]
    return grid


def _random_mask(
    n: int,
    *,
    active_count: int,
    forbidden: set[tuple[int, ...]] | None = None,
) -> np.ndarray:
    forbidden = forbidden or set()
    total = n * n
    for _ in range(400):
        chosen = random.sample(range(total), active_count)
        mask = np.zeros((n, n), dtype=bool)
        mask.flat[chosen] = True
        if _mask_key(mask) not in forbidden:
            return mask
    raise RuntimeError("failed to sample a unique mask")


def _mask_pattern(mask: np.ndarray, bg_color: int, ink_color: int) -> np.ndarray:
    arr = np.full(mask.shape, bg_color, dtype=int)
    arr[mask] = ink_color
    return arr


def _paint_mask(
    mask: np.ndarray,
    bg_color: int,
    ink_pool: list[int],
    *,
    avoid: set[tuple[int, ...]],
    prefer_multicolor: bool,
) -> np.ndarray:
    coords = [tuple(coord) for coord in np.argwhere(mask)]
    assert coords, "mask must have at least one active pixel"

    for _ in range(400):
        arr = np.full(mask.shape, bg_color, dtype=int)
        multicolor = prefer_multicolor or (len(ink_pool) > 1 and len(coords) > 1 and random.random() < 0.55)
        if multicolor:
            colors = [random.choice(ink_pool) for _ in coords]
            if len(set(colors)) == 1 and len(ink_pool) > 1:
                idx = random.randrange(len(colors))
                alternatives = [color for color in ink_pool if color != colors[idx]]
                colors[idx] = random.choice(alternatives)
        else:
            colors = [random.choice(ink_pool)] * len(coords)

        for (r, c), color in zip(coords, colors):
            arr[r, c] = color

        key = _pattern_key(arr)
        if key not in avoid:
            return arr
    raise RuntimeError("failed to paint a unique pattern")


def _deviant_count_bounds(n: int) -> tuple[int, int]:
    if n == 3:
        return 2, 4
    return 2, 6


def _majority_active_bounds(n: int) -> tuple[int, int]:
    return 1, max(2, (n * n) // 3)


def _sample_case() -> tuple[np.ndarray, dict]:
    n = random.choice(GRID_NS)
    sep_color, bg_color, majority_ink = random.sample(PALETTE, 3)
    extra_inks = random.sample([color for color in PALETTE if color not in {sep_color, bg_color, majority_ink}], 3)
    ink_pool = [majority_ink] + extra_inks

    all_positions = [(i, j) for i in range(n) for j in range(n)]
    keep = random.choice(all_positions)
    dev_lo, dev_hi = _deviant_count_bounds(n)
    deviant_count = random.randint(dev_lo, dev_hi)
    deviant_positions = {keep, *random.sample([pos for pos in all_positions if pos != keep], deviant_count - 1)}
    dev_mask = np.zeros((n, n), dtype=bool)
    for i, j in deviant_positions:
        dev_mask[i, j] = True
    roster_active_count = int(dev_mask.sum())

    shared_mask_branch = random.random() < SHARED_MASK_BRANCH_PROB
    if shared_mask_branch:
        majority_mask = dev_mask.copy()
    else:
        maj_lo, maj_hi = _majority_active_bounds(n)
        majority_active_count = random.randint(maj_lo, maj_hi)
        majority_mask = _random_mask(n, active_count=majority_active_count, forbidden={_mask_key(dev_mask)})

    majority = _mask_pattern(majority_mask, bg_color, majority_ink)
    cells = [[majority.copy() for _ in range(n)] for _ in range(n)]

    used_patterns = {_pattern_key(majority)}
    used_deviant_masks = {_mask_key(dev_mask)}
    meta = {
        "n": n,
        "separator": sep_color,
        "bg": bg_color,
        "majority": majority,
        "majority_mask": majority_mask,
        "keep": keep,
        "deviants": tuple(sorted(deviant_positions)),
        "dev_mask": dev_mask,
        "shared_mask_branch": shared_mask_branch,
        "color_only_branch": False,
    }

    keep_pattern = _paint_mask(
        dev_mask,
        bg_color,
        ink_pool,
        avoid=used_patterns,
        prefer_multicolor=shared_mask_branch or random.random() < 0.5,
    )
    cells[keep[0]][keep[1]] = keep_pattern
    used_patterns.add(_pattern_key(keep_pattern))

    others = [pos for pos in sorted(deviant_positions) if pos != keep]
    same_count_pos = others[0]
    same_count_mask = _random_mask(
        n,
        active_count=roster_active_count,
        forbidden={_mask_key(dev_mask), _mask_key(majority_mask)},
    )
    same_count_pattern = _paint_mask(
        same_count_mask,
        bg_color,
        ink_pool,
        avoid=used_patterns,
        prefer_multicolor=random.random() < 0.5,
    )
    cells[same_count_pos[0]][same_count_pos[1]] = same_count_pattern
    used_patterns.add(_pattern_key(same_count_pattern))
    used_deviant_masks.add(_mask_key(same_count_mask))

    remaining = others[1:]
    if remaining and not shared_mask_branch and random.random() < COLOR_ONLY_BRANCH_PROB:
        pos = remaining.pop(0)
        color_only_pattern = _paint_mask(
            majority_mask,
            bg_color,
            ink_pool,
            avoid=used_patterns,
            prefer_multicolor=True,
        )
        cells[pos[0]][pos[1]] = color_only_pattern
        used_patterns.add(_pattern_key(color_only_pattern))
        used_deviant_masks.add(_mask_key(majority_mask))
        meta["color_only_branch"] = True

    for pos in remaining:
        lo = 1
        hi = max(2, min(n * n - 1, n + 2))
        active_count = random.randint(lo, hi)
        mask = _random_mask(
            n,
            active_count=active_count,
            forbidden=used_deviant_masks | {_mask_key(majority_mask)},
        )
        pattern = _paint_mask(
            mask,
            bg_color,
            ink_pool,
            avoid=used_patterns,
            prefer_multicolor=random.random() < 0.5,
        )
        cells[pos[0]][pos[1]] = pattern
        used_patterns.add(_pattern_key(pattern))
        used_deviant_masks.add(_mask_key(mask))

    inp = _render_grid(n, sep_color, cells)
    meta["input"] = inp
    return inp, meta


def _expected_output(inp: np.ndarray, keep: tuple[int, int]) -> np.ndarray:
    info = _analyze_grid(inp)
    out = info["grid"].copy()
    n = info["n"]
    for i, j in info["deviants"]:
        if (i, j) == keep:
            continue
        r0 = info["borders"][i] + 1
        c0 = info["borders"][j] + 1
        out[r0 : r0 + n, c0 : c0 + n] = info["majority"]
    return out


def _validate_case(inp: np.ndarray, meta: dict) -> bool:
    info = _analyze_grid(inp)
    keep = tuple(meta["keep"])
    expected = _expected_output(inp, keep)
    actual = main(inp)
    if not np.array_equal(actual, expected):
        return False
    if np.array_equal(inp, expected):
        return False

    if info["n"] != meta["n"] or info["keep"] != keep:
        return False
    if set(info["deviants"]) != set(meta["deviants"]):
        return False
    if not np.array_equal(info["dev_mask"], meta["dev_mask"]):
        return False

    roster_count = int(info["dev_mask"].sum())
    same_count_deviants = sum(
        int((info["cells"][i][j] != info["cell_bg"]).sum()) == roster_count for i, j in info["deviants"]
    )
    if same_count_deviants < 2:
        return False

    if meta["shared_mask_branch"]:
        if not np.array_equal(info["majority_mask"], info["dev_mask"]):
            return False
        all_mask_matches = [
            (i, j)
            for i in range(info["n"])
            for j in range(info["n"])
            if np.array_equal(info["cells"][i][j] != info["cell_bg"], info["dev_mask"])
        ]
        if len(all_mask_matches) <= 1:
            return False

    if meta["color_only_branch"]:
        found = False
        for i, j in info["deviants"]:
            if (i, j) == keep:
                continue
            same_mask = np.array_equal(info["cells"][i][j] != info["cell_bg"], info["majority_mask"])
            if same_mask and not np.array_equal(info["cells"][i][j], info["majority"]):
                found = True
                break
        if not found:
            return False

    for i in range(info["n"]):
        for j in range(info["n"]):
            r0 = info["borders"][i] + 1
            c0 = info["borders"][j] + 1
            block = actual[r0 : r0 + info["n"], c0 : c0 + info["n"]]
            if (i, j) == keep:
                if not np.array_equal(block, info["cells"][i][j]):
                    return False
            else:
                if not np.array_equal(block, info["majority"]):
                    return False
    return True


def generate_input() -> np.ndarray:
    for _ in range(GEN_ATTEMPTS):
        inp, meta = _sample_case()
        if _validate_case(inp, meta):
            return inp
    raise RuntimeError("failed to generate a valid input for puzzle 7")
