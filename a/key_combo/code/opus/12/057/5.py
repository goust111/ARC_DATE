from __future__ import annotations

from collections import Counter, deque
import math
import random

import numpy as np


BLACK = 0
BLUE = 1
RED = 2
YELLOW = 4
ORANGE = 7
TEAL = 8

LEGEND_STEP = 2
GEN_ATTEMPTS = 600

DIRS4 = [(-1, 0), (1, 0), (0, -1), (0, 1)]


def _neighbors4(r: int, c: int):
    for dr, dc in DIRS4:
        yield r + dr, c + dc


def _components(mask: np.ndarray) -> list[list[tuple[int, int]]]:
    mask = np.asarray(mask, dtype=bool)
    h, w = mask.shape
    seen = np.zeros((h, w), dtype=bool)
    comps: list[list[tuple[int, int]]] = []
    for r in range(h):
        for c in range(w):
            if not mask[r, c] or seen[r, c]:
                continue
            q = deque([(r, c)])
            seen[r, c] = True
            comp: list[tuple[int, int]] = []
            while q:
                rr, cc = q.popleft()
                comp.append((rr, cc))
                for nr, nc in _neighbors4(rr, cc):
                    if 0 <= nr < h and 0 <= nc < w and mask[nr, nc] and not seen[nr, nc]:
                        seen[nr, nc] = True
                        q.append((nr, nc))
            comps.append(comp)
    return comps


def _bbox(coords: list[tuple[int, int]]) -> tuple[int, int, int, int]:
    rows = [r for r, _ in coords]
    cols = [c for _, c in coords]
    return min(rows), min(cols), max(rows), max(cols)


def _get_bg(grid: np.ndarray) -> int:
    return int(Counter(map(int, grid.ravel())).most_common(1)[0][0])


def _extract_objects(grid: np.ndarray) -> tuple[list[tuple[int, int, int]], list[tuple[int, int, int, int]]]:
    singles: list[tuple[int, int, int]] = []
    blocks: list[tuple[int, int, int, int]] = []
    for comp in _components(grid != BLACK):
        r0, c0, r1, c1 = _bbox(comp)
        color = int(grid[comp[0][0], comp[0][1]])
        if len(comp) == 1:
            singles.append((r0, c0, color))
            continue
        height = r1 - r0 + 1
        width = c1 - c0 + 1
        if height == width and 3 <= height <= 5:
            sub = grid[r0 : r1 + 1, c0 : c1 + 1]
            if np.all(sub == color) and len(comp) == height * width:
                blocks.append((r0, c0, height, color))
                continue
        raise AssertionError("unexpected non-background component")
    return singles, blocks


def _legend_from_singles(grid: np.ndarray, singles: list[tuple[int, int, int]]) -> tuple[np.ndarray, tuple[int, int, int, int]]:
    legend_singles = [(r, c, color) for r, c, color in singles if r < 12 and c < 12]
    assert legend_singles, "expected legend pixels in the top-left corner"

    rows = sorted({r for r, _, _ in legend_singles})
    cols = sorted({c for _, c, _ in legend_singles})
    row_diffs = [rows[i + 1] - rows[i] for i in range(len(rows) - 1) if rows[i + 1] > rows[i]]
    col_diffs = [cols[i + 1] - cols[i] for i in range(len(cols) - 1) if cols[i + 1] > cols[i]]
    diffs = row_diffs + col_diffs
    step = diffs[0]
    for value in diffs[1:]:
        step = math.gcd(step, value)
    assert step > 0

    r_min = min(r for r, _, _ in legend_singles)
    c_min = min(c for _, c, _ in legend_singles)
    logical = [((r - r_min) // step, (c - c_min) // step, color) for r, c, color in legend_singles]
    legend_h = max(rr for rr, _, _ in logical) + 1
    legend_w = max(cc for _, cc, _ in logical) + 1
    legend = np.zeros((legend_h, legend_w), dtype=int)
    for rr, cc, color in logical:
        legend[rr, cc] = int(color)
    return legend, (r_min, c_min, r_min + (legend_h - 1) * step, c_min + (legend_w - 1) * step)


def _mapping_candidates(
    grid: np.ndarray,
    legend: np.ndarray,
    blocks: list[tuple[int, int, int, int]],
) -> list[dict]:
    h, w = grid.shape
    nonzero = [(rr, cc, int(legend[rr, cc])) for rr, cc in np.argwhere(legend != BLACK)]
    sizes = sorted({size for _, _, size, _ in blocks})
    candidates: list[dict] = []

    for size in sizes:
        size_blocks = [(r0, c0, color) for r0, c0, block_size, color in blocks if block_size == size]
        if len(size_blocks) < 3:
            continue
        max_step = min(10, h, w)
        for step in range(size + 1, max_step + 1):
            bases = set()
            for rr, cc, color in nonzero:
                for br, bc, bcolor in size_blocks:
                    if color == bcolor:
                        bases.add((br - rr * step, bc - cc * step))
            for base_r, base_c in bases:
                present: dict[tuple[int, int], tuple[int, int, int]] = {}
                ok = True
                for br, bc, color in size_blocks:
                    dr = br - base_r
                    dc = bc - base_c
                    if dr % step or dc % step:
                        ok = False
                        break
                    rr = dr // step
                    cc = dc // step
                    if not (0 <= rr < legend.shape[0] and 0 <= cc < legend.shape[1]):
                        ok = False
                        break
                    if int(legend[rr, cc]) != color:
                        ok = False
                        break
                    key = (rr, cc)
                    if key in present:
                        ok = False
                        break
                    sub = grid[br : br + size, bc : bc + size]
                    if sub.shape != (size, size) or not np.all(sub == color):
                        ok = False
                        break
                    present[key] = (br, bc, color)
                if not ok:
                    continue

                missing = []
                for rr, cc, color in nonzero:
                    tr = base_r + rr * step
                    tc = base_c + cc * step
                    if not (0 <= tr <= h - size and 0 <= tc <= w - size):
                        ok = False
                        break
                    sub = grid[tr : tr + size, tc : tc + size]
                    if (rr, cc) in present:
                        if not np.all(sub == color):
                            ok = False
                            break
                    else:
                        if not np.all(sub == BLACK):
                            ok = False
                            break
                        missing.append((rr, cc, color))
                if not ok or not missing:
                    continue
                candidates.append(
                    {
                        "size": size,
                        "step": step,
                        "base_r": base_r,
                        "base_c": base_c,
                        "present": present,
                        "missing": missing,
                    }
                )
    return candidates


def _infer_mapping(
    grid: np.ndarray,
) -> tuple[np.ndarray, tuple[int, int, int, int], dict]:
    singles, blocks = _extract_objects(grid)
    assert blocks, "expected main-area blocks"
    legend, legend_bbox = _legend_from_singles(grid, singles)
    candidates = _mapping_candidates(grid, legend, blocks)
    assert candidates, "failed to infer a valid minimap mapping"

    best_score = max(len(cand["present"]) for cand in candidates)
    best = [cand for cand in candidates if len(cand["present"]) == best_score]
    unique: list[dict] = []
    seen = set()
    for cand in best:
        key = (
            cand["size"],
            cand["step"],
            cand["base_r"],
            cand["base_c"],
            tuple(sorted(cand["present"])),
        )
        if key not in seen:
            unique.append(cand)
            seen.add(key)
    assert len(unique) == 1, "mapping must be uniquely determined"
    return legend, legend_bbox, unique[0]


def main(grid: np.ndarray) -> np.ndarray:
    """
    Complete the scaled block layout from a top-left minimap, then recolor one color group using the completion count.

    Input structure:
    - Black background with a sparse minimap sprite of singleton blue, red, and teal pixels in the top-left corner.
    - Elsewhere, the main area contains several separated NxN solid blocks, with N between 3 and 5.
    - The main blocks correspond to non-black minimap pixels under one shared scale-and-offset mapping, but some mapped positions are still empty.

    Rule:
    - Infer the block size, lattice step, and offset by aligning the existing main-area blocks with same-colored minimap pixels.
    - For every non-black minimap pixel whose mapped NxN block is absent, place a new block of that pixel's color and count how many blocks were added.
    - If the number of added blocks is odd, recolor every teal main-area block to yellow.
    - If the number of added blocks is even, recolor every red main-area block to orange.
    - Blue blocks are never recolored, and the minimap pixels themselves stay unchanged.

    Key insight:
    - The minimap first determines where completion must happen, and only that completion count reveals which entire block color family in the main area must be recolored.
    """
    g = np.asarray(grid, dtype=int)
    legend, _legend_bbox, mapping = _infer_mapping(g)
    out = g.copy()

    added = 0
    for rr, cc in np.argwhere(legend != BLACK):
        color = int(legend[rr, cc])
        tr = mapping["base_r"] + int(rr) * mapping["step"]
        tc = mapping["base_c"] + int(cc) * mapping["step"]
        if (int(rr), int(cc)) not in mapping["present"]:
            out[tr : tr + mapping["size"], tc : tc + mapping["size"]] = color
            added += 1

    from_color, to_color = (TEAL, YELLOW) if added % 2 == 1 else (RED, ORANGE)
    for rr, cc in np.argwhere(legend == from_color):
        tr = mapping["base_r"] + int(rr) * mapping["step"]
        tc = mapping["base_c"] + int(cc) * mapping["step"]
        out[tr : tr + mapping["size"], tc : tc + mapping["size"]] = to_color
    return out


def _physical_legend_size(rows: int, cols: int) -> tuple[int, int]:
    return 1 + (rows - 1) * LEGEND_STEP, 1 + (cols - 1) * LEGEND_STEP


def _make_legend(rng: random.Random, rows: int, cols: int) -> np.ndarray | None:
    total = rows * cols
    fill_count = rng.randint(max(7, total - 5), total)
    cells = [(r, c) for r in range(rows) for c in range(cols)]
    rng.shuffle(cells)
    filled = set(cells[:fill_count])

    if any(all((r, c) not in filled for c in range(cols)) for r in range(rows)):
        return None
    if any(all((r, c) not in filled for r in range(rows)) for c in range(cols)):
        return None

    legend = np.zeros((rows, cols), dtype=int)
    mandatory = [TEAL, TEAL, RED, RED, BLUE]
    fill_list = list(filled)
    rng.shuffle(fill_list)
    if len(fill_list) < len(mandatory):
        return None
    for cell, color in zip(fill_list[: len(mandatory)], mandatory):
        legend[cell] = color
    for cell in fill_list[len(mandatory) :]:
        legend[cell] = rng.choice([BLUE, RED, TEAL])
    return legend


def _choose_missing(
    rng: random.Random,
    legend: np.ndarray,
    want_odd: bool,
) -> set[tuple[int, int]] | None:
    nonzero = [(int(r), int(c), int(legend[r, c])) for r, c in np.argwhere(legend != BLACK)]
    target = TEAL if want_odd else RED
    allowed_sizes = [k for k in range(1, len(nonzero) - 2) if (k % 2 == 1) == want_odd]
    rng.shuffle(allowed_sizes)

    for missing_size in allowed_sizes:
        for _ in range(250):
            sample = rng.sample(nonzero, missing_size)
            missing = {(r, c) for r, c, _ in sample}
            missing_colors = {color for _, _, color in sample}
            present_colors = {
                color for r, c, color in nonzero if (r, c) not in missing
            }
            if target not in missing_colors:
                continue
            if target not in present_colors:
                continue
            if TEAL not in present_colors or RED not in present_colors or BLUE not in present_colors:
                continue
            return missing
    return None


def _expected_from_params(inp: np.ndarray, params: dict) -> np.ndarray:
    out = inp.copy()
    for rr, cc in params["missing"]:
        color = int(params["legend"][rr, cc])
        tr = params["base_r"] + rr * params["step"]
        tc = params["base_c"] + cc * params["step"]
        out[tr : tr + params["size"], tc : tc + params["size"]] = color

    from_color, to_color = (TEAL, YELLOW) if len(params["missing"]) % 2 == 1 else (RED, ORANGE)
    for rr, cc in np.argwhere(params["legend"] == from_color):
        tr = params["base_r"] + int(rr) * params["step"]
        tc = params["base_c"] + int(cc) * params["step"]
        out[tr : tr + params["size"], tc : tc + params["size"]] = to_color
    return out


def _validate_case(inp: np.ndarray, params: dict) -> bool:
    if inp.ndim != 2 or max(inp.shape) > 30:
        return False
    if len(params["present"]) < 3:
        return False
    if {TEAL, RED} - {params["legend"][r, c] for r, c in params["present"]}:
        return False

    target = TEAL if len(params["missing"]) % 2 == 1 else RED
    if not any(int(params["legend"][r, c]) == target for r, c in params["missing"]):
        return False
    if not any(int(params["legend"][r, c]) == target for r, c in params["present"]):
        return False

    try:
        legend, _legend_bbox, mapping = _infer_mapping(inp)
    except AssertionError:
        return False
    if not np.array_equal(legend, params["legend"]):
        return False
    if mapping["size"] != params["size"] or mapping["step"] != params["step"]:
        return False
    if mapping["base_r"] != params["base_r"] or mapping["base_c"] != params["base_c"]:
        return False

    expected = _expected_from_params(inp, params)
    actual = main(inp)
    if not np.array_equal(actual, expected):
        return False

    unchanged_target = RED if target == TEAL else TEAL
    if not any(int(params["legend"][r, c]) == unchanged_target for r, c in params["present"]):
        return False
    return True


def _gen_attempt(rng: random.Random) -> tuple[np.ndarray, dict] | None:
    size = rng.randint(3, 5)
    gap = rng.randint(1, 2)
    step = size + gap
    rows = rng.randint(3, 5)
    cols = rng.randint(4, 5)
    phys_h, phys_w = _physical_legend_size(rows, cols)

    grid_h = rng.randint(22, 30)
    grid_w = rng.randint(22, 30)
    min_base_r = phys_h + 2
    min_base_c = phys_w + 3
    max_base_r = grid_h - ((rows - 1) * step + size)
    max_base_c = grid_w - ((cols - 1) * step + size)
    if min_base_r > max_base_r or min_base_c > max_base_c:
        return None

    legend = _make_legend(rng, rows, cols)
    if legend is None:
        return None

    want_odd = bool(rng.randrange(2))
    missing = _choose_missing(rng, legend, want_odd)
    if missing is None:
        return None

    present = {(int(r), int(c)) for r, c in np.argwhere(legend != BLACK)} - missing
    present_rows = {r for r, _ in present}
    present_cols = {c for _, c in present}
    if len(present_rows) < 2 or len(present_cols) < 2:
        return None

    base_r = rng.randint(min_base_r, max_base_r)
    base_c = rng.randint(min_base_c, max_base_c)
    grid = np.zeros((grid_h, grid_w), dtype=int)

    origin_r = 1
    origin_c = 1
    for rr, cc in np.argwhere(legend != BLACK):
        grid[origin_r + int(rr) * LEGEND_STEP, origin_c + int(cc) * LEGEND_STEP] = int(legend[rr, cc])

    for rr, cc in sorted(present):
        color = int(legend[rr, cc])
        tr = base_r + rr * step
        tc = base_c + cc * step
        grid[tr : tr + size, tc : tc + size] = color

    params = {
        "legend": legend,
        "size": size,
        "step": step,
        "base_r": base_r,
        "base_c": base_c,
        "present": present,
        "missing": missing,
    }
    return grid, params


def generate_input() -> np.ndarray:
    rng = random.Random()
    for _ in range(GEN_ATTEMPTS):
        result = _gen_attempt(rng)
        if result is None:
            continue
        grid, params = result
        if _validate_case(grid, params):
            return grid
    raise RuntimeError("failed to generate a valid input")
