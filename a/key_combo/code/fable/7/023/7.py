from __future__ import annotations

import itertools
from collections import Counter

import numpy as np


ALL_COLORS = tuple(range(10))
MAX_GENERATION_ATTEMPTS = 400


def get_bg(grid: np.ndarray) -> int:
    """Background color = most common value in the grid."""
    return int(Counter(map(int, np.asarray(grid).ravel())).most_common(1)[0][0])


def colors_present(grid: np.ndarray, *, exclude: set[int] | None = None) -> list[int]:
    """Return sorted unique colors present in the grid."""
    colors = set(map(int, np.asarray(grid).ravel()))
    if exclude is not None:
        colors -= set(exclude)
    return sorted(colors)


def _pair_candidate_info(grid: np.ndarray, bg: int, color_a: int, color_b: int) -> dict | None:
    mask = (grid == color_a) | (grid == color_b)
    coords = np.argwhere(mask)
    clue_count = int(coords.shape[0])
    if clue_count < 6:
        return None

    rows = sorted(map(int, np.unique(coords[:, 0])))
    cols = sorted(map(int, np.unique(coords[:, 1])))
    row_count = len(rows)
    col_count = len(cols)
    if row_count < 3 or col_count < 3:
        return None

    area = row_count * col_count
    if clue_count >= area or clue_count * 2 < area:
        return None

    row_to_index = {r: i for i, r in enumerate(rows)}
    col_to_index = {c: j for j, c in enumerate(cols)}
    row_hits = Counter()
    col_hits = Counter()
    valid_parities = []

    for a_on_parity in (0, 1):
        ok = True
        for r, c in coords:
            r = int(r)
            c = int(c)
            parity = (row_to_index[r] + col_to_index[c]) % 2
            expected = color_a if parity == a_on_parity else color_b
            if int(grid[r, c]) != expected:
                ok = False
                break
        if ok:
            valid_parities.append(a_on_parity)

    if len(valid_parities) != 1:
        return None

    for r, c in coords:
        row_hits[int(r)] += 1
        col_hits[int(c)] += 1
    if min(row_hits.values()) < 1 or min(col_hits.values()) < 1:
        return None

    for r in rows:
        for c in cols:
            value = int(grid[r, c])
            if value not in (bg, color_a, color_b):
                return None

    seen_colors = set(map(int, grid[mask].ravel()))
    if seen_colors != {color_a, color_b}:
        return None

    density = clue_count / area
    return {
        "colors": (int(color_a), int(color_b)),
        "rows": rows,
        "cols": cols,
        "row_to_index": row_to_index,
        "col_to_index": col_to_index,
        "a_on_parity": int(valid_parities[0]),
        "clue_count": clue_count,
        "area": area,
        "density": density,
    }


def _all_candidates(grid: np.ndarray) -> list[dict]:
    bg = get_bg(grid)
    candidates = []
    for color_a, color_b in itertools.combinations(colors_present(grid, exclude={bg}), 2):
        info = _pair_candidate_info(grid, bg, color_a, color_b)
        if info is not None:
            candidates.append(info)
    return candidates


def _find_target_candidate(grid: np.ndarray) -> dict:
    candidates = _all_candidates(grid)
    assert candidates, "expected exactly one recoverable two-color lattice"
    candidates.sort(key=lambda info: (info["density"], info["area"], info["clue_count"]), reverse=True)
    best = candidates[0]
    if len(candidates) > 1:
        best_key = (best["density"], best["area"], best["clue_count"])
        next_key = (candidates[1]["density"], candidates[1]["area"], candidates[1]["clue_count"])
        assert best_key > next_key, "expected a unique best lattice candidate"
    return best


def _color_for(candidate: dict, r: int, c: int) -> int:
    a, b = candidate["colors"]
    parity = (candidate["row_to_index"][r] + candidate["col_to_index"][c]) % 2
    return a if parity == candidate["a_on_parity"] else b


def main(input_grid: np.ndarray) -> np.ndarray:
    """
    Reconstruct the hidden active-row x active-column lattice and complete every missing A/B intersection.

    Input structure:
    - A dominant background contains sparse pixels of exactly one meaningful color pair A and B.
    - Those A/B pixels lie only on intersections of some hidden active rows and active columns, while
      other colors are distractor pixels placed away from the true intersections.
    - In active-row order i and active-column order j, the true lattice follows a checkerboard rule
      based on the parity of i + j, but some intersections are still missing.

    Rule:
    - Search for the unique color pair whose occupied rows and columns define a partially drawn row/column product.
    - Use only that pair to recover the active-row set, active-column set, and which parity corresponds to A vs B.
    - Fill every missing intersection of the recovered product lattice with its checkerboard color, leaving
      distractors and all non-target cells unchanged.

    Key insight:
    - The checkerboard law lives in row/column index space, not in geometric spacing, so the solver must
      rebuild the hidden lattice ordering before painting any missing pixels.
    """
    grid = np.asarray(input_grid, dtype=int)
    candidate = _find_target_candidate(grid)
    out = grid.copy()
    for r in candidate["rows"]:
        for c in candidate["cols"]:
            out[r, c] = _color_for(candidate, r, c)
    return out


def _has_irregular_gaps(positions: list[int]) -> bool:
    gaps = np.diff(positions)
    return len(gaps) >= 2 and len(set(map(int, gaps))) >= 2


def _sample_positions(rng: np.random.Generator, size: int, count: int) -> list[int] | None:
    choices = np.arange(1, size - 1)
    if len(choices) < count:
        return None
    for _ in range(200):
        positions = sorted(map(int, rng.choice(choices, size=count, replace=False)))
        if _has_irregular_gaps(positions):
            return positions
    return None


def _build_visible_subset(
    rng: np.random.Generator, row_count: int, col_count: int
) -> tuple[set[tuple[int, int]], int] | None:
    area = row_count * col_count
    min_visible = max(row_count + col_count - 1, (area + 1) // 2, 6)
    if min_visible >= area:
        return None

    target_count = int(rng.integers(min_visible, area))
    visible: set[tuple[int, int]] = set()

    for ri in range(row_count):
        visible.add((ri, int(rng.integers(col_count))))
    for cj in range(col_count):
        visible.add((int(rng.integers(row_count)), cj))

    while len(visible) < target_count:
        visible.add((int(rng.integers(row_count)), int(rng.integers(col_count))))

    parities = {(ri + cj) % 2 for ri, cj in visible}
    if len(parities) != 2 or len(visible) >= area:
        return None
    return visible, area


def _available_distractor_cells(
    shape: tuple[int, int], rows: list[int], cols: list[int]
) -> dict[str, list[tuple[int, int]]]:
    height, width = shape
    active_rows = set(rows)
    active_cols = set(cols)
    cells = {"row": [], "col": [], "neither": []}
    for r in range(height):
        for c in range(width):
            if r in active_rows and c in active_cols:
                continue
            if r in active_rows:
                cells["row"].append((r, c))
            elif c in active_cols:
                cells["col"].append((r, c))
            else:
                cells["neither"].append((r, c))
    return cells


def _place_distractors(
    rng: np.random.Generator,
    grid: np.ndarray,
    rows: list[int],
    cols: list[int],
    palette: list[int],
) -> list[dict]:
    available = _available_distractor_cells(grid.shape, rows, cols)
    modes = [mode for mode, cells in available.items() if cells]
    assert modes, "expected some non-intersection cells"

    mandatory = []
    for mode in ("row", "col"):
        if available[mode]:
            mandatory.append(mode)
    if available["neither"] and rng.random() < 0.6:
        mandatory.append("neither")

    pool = sum(len(cells) for cells in available.values())
    count = int(rng.integers(len(mandatory), min(8, pool) + 1))
    chosen_modes = list(mandatory)
    while len(chosen_modes) < count:
        chosen_modes.append(str(rng.choice(modes)))

    distractors = []
    used = set()
    for mode in chosen_modes:
        cells = [cell for cell in available[mode] if cell not in used]
        if not cells:
            continue
        idx = int(rng.integers(len(cells)))
        r, c = cells[idx]
        color = int(rng.choice(palette))
        grid[r, c] = color
        used.add((r, c))
        distractors.append({"pos": (r, c), "mode": mode, "color": color})
    return distractors


def _expected_output(
    input_grid: np.ndarray,
    rows: list[int],
    cols: list[int],
    color_a: int,
    color_b: int,
    a_on_parity: int,
) -> np.ndarray:
    out = np.asarray(input_grid, dtype=int).copy()
    row_to_index = {r: i for i, r in enumerate(rows)}
    col_to_index = {c: j for j, c in enumerate(cols)}
    for r in rows:
        for c in cols:
            parity = (row_to_index[r] + col_to_index[c]) % 2
            out[r, c] = color_a if parity == a_on_parity else color_b
    return out


def _generate_case() -> dict | None:
    rng = np.random.default_rng()
    height = int(rng.integers(12, 25))
    width = int(rng.integers(12, 25))
    row_count = int(rng.integers(3, 6))
    col_count = int(rng.integers(3, 7))

    rows = _sample_positions(rng, height, row_count)
    cols = _sample_positions(rng, width, col_count)
    if rows is None or cols is None:
        return None

    visible_info = _build_visible_subset(rng, row_count, col_count)
    if visible_info is None:
        return None
    visible, area = visible_info

    bg = int(rng.choice(ALL_COLORS))
    palette = [color for color in ALL_COLORS if color != bg]
    rng.shuffle(palette)
    color_a, color_b = map(int, palette[:2])
    distractor_palette = list(map(int, palette[2:]))
    a_on_parity = int(rng.integers(2))

    grid = np.full((height, width), bg, dtype=int)
    shown_cells = set()
    for ri, cj in visible:
        r = rows[ri]
        c = cols[cj]
        shown_cells.add((r, c))
        parity = (ri + cj) % 2
        grid[r, c] = color_a if parity == a_on_parity else color_b

    distractors = _place_distractors(rng, grid, rows, cols, distractor_palette)
    expected = _expected_output(grid, rows, cols, color_a, color_b, a_on_parity)

    return {
        "input": grid,
        "output": expected,
        "bg": bg,
        "rows": rows,
        "cols": cols,
        "colors": (color_a, color_b),
        "a_on_parity": a_on_parity,
        "shown_cells": shown_cells,
        "missing_cells": {(r, c) for r in rows for c in cols} - shown_cells,
        "distractors": distractors,
        "area": area,
    }


def _validate_case(case: dict) -> bool:
    grid = case["input"]
    expected = case["output"]
    rows = case["rows"]
    cols = case["cols"]
    color_a, color_b = case["colors"]
    solved = main(grid)

    if not np.array_equal(solved, expected):
        return False
    if max(grid.shape) > 30 or max(expected.shape) > 30:
        return False
    if not case["missing_cells"]:
        return False
    if not _has_irregular_gaps(rows) or not _has_irregular_gaps(cols):
        return False

    changed = {tuple(map(int, pos)) for pos in np.argwhere(grid != expected)}
    if changed != case["missing_cells"]:
        return False

    for distractor in case["distractors"]:
        r, c = distractor["pos"]
        if (r, c) in changed:
            return False

    candidates = _all_candidates(grid)
    if not candidates:
        return False
    candidates.sort(key=lambda info: (info["density"], info["area"], info["clue_count"]), reverse=True)
    best = candidates[0]
    if set(best["colors"]) != {color_a, color_b}:
        return False
    if best["rows"] != rows or best["cols"] != cols:
        return False
    parity_probe = _color_for(best, rows[0], cols[0])
    expected_probe = expected[rows[0], cols[0]]
    if int(parity_probe) != int(expected_probe):
        return False
    if len(candidates) > 1:
        best_key = (best["density"], best["area"], best["clue_count"])
        next_key = (candidates[1]["density"], candidates[1]["area"], candidates[1]["clue_count"])
        if best_key <= next_key:
            return False

    modes = {d["mode"] for d in case["distractors"]}
    if "row" not in modes or "col" not in modes:
        return False

    return set(map(int, grid[tuple(np.array(list(case["shown_cells"])).T)])) == {color_a, color_b}


def generate_input() -> np.ndarray:
    for _ in range(MAX_GENERATION_ATTEMPTS):
        case = _generate_case()
        if case is not None and _validate_case(case):
            return case["input"]
    raise RuntimeError("failed to generate a valid ARC sample")
