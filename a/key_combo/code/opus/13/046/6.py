from __future__ import annotations

import random

import numpy as np


BLACK = 0
BLUE = 1
RED = 2
GREEN = 3
GREY = 5
CONTROL_COLORS = (BLUE, RED, GREEN)

MIN_CONTROLS = 2
MAX_CONTROLS = 3
MIN_RECTS = 3
MAX_RECTS = 5
RECT_H_RANGE = (2, 4)
RECT_W_RANGE = (3, 6)
GRID_W_RANGE = (14, 18)
GRID_H_CAP = 28
CONTROL_MIN_GAP = 5
GEN_ATTEMPTS = 400


def _neighbors4(r: int, c: int) -> tuple[tuple[int, int], ...]:
    return ((r - 1, c), (r + 1, c), (r, c - 1), (r, c + 1))


def _components_mask(mask: np.ndarray) -> list[np.ndarray]:
    h, w = mask.shape
    seen = np.zeros_like(mask, dtype=bool)
    comps: list[np.ndarray] = []
    for r in range(h):
        for c in range(w):
            if not bool(mask[r, c]) or bool(seen[r, c]):
                continue
            stack = [(r, c)]
            seen[r, c] = True
            coords: list[tuple[int, int]] = []
            while stack:
                rr, cc = stack.pop()
                coords.append((rr, cc))
                for nr, nc in _neighbors4(rr, cc):
                    if 0 <= nr < h and 0 <= nc < w and bool(mask[nr, nc]) and not bool(seen[nr, nc]):
                        seen[nr, nc] = True
                        stack.append((nr, nc))
            comps.append(np.asarray(coords, dtype=int))
    return comps


def _bbox_coords(coords: np.ndarray) -> tuple[int, int, int, int]:
    r0, c0 = coords.min(axis=0)
    r1, c1 = coords.max(axis=0)
    return int(r0), int(c0), int(r1), int(c1)


def _extract_controls(grid: np.ndarray) -> list[tuple[int, int]]:
    controls = [(int(c), int(grid[0, c])) for c in range(grid.shape[1]) if int(grid[0, c]) != BLACK]
    assert MIN_CONTROLS <= len(controls) <= MAX_CONTROLS
    assert len({c for c, _ in controls}) == len(controls)
    assert all(color in CONTROL_COLORS for _, color in controls)
    assert not np.any((grid[1:, :] != BLACK) & (grid[1:, :] != GREY))
    return controls


def _erase_recolor(grid: np.ndarray) -> tuple[np.ndarray, list[np.ndarray], list[np.ndarray]]:
    g = np.asarray(grid, dtype=int)
    controls = _extract_controls(g)
    grey_comps = _components_mask(g == GREY)
    small = [comp for comp in grey_comps if int(comp.shape[0]) < 3]
    large = [comp for comp in grey_comps if int(comp.shape[0]) >= 3]
    assert MIN_RECTS <= len(large) <= MAX_RECTS

    out = g.copy()
    out[g == GREY] = BLACK
    for comp in large:
        _r0, c0, _r1, c1 = _bbox_coords(comp)
        matches = [color for col, color in controls if c0 <= col <= c1]
        assert len(matches) == 1
        out[comp[:, 0], comp[:, 1]] = matches[0]
    return out, large, small


def _compress_rows(grid: np.ndarray) -> np.ndarray:
    keep = [0]
    keep.extend(r for r in range(1, grid.shape[0]) if bool(np.any(grid[r] != BLACK)))
    return grid[np.asarray(keep, dtype=int), :].copy()


def main(grid: np.ndarray) -> np.ndarray:
    """
    Erase tiny grey noise, recolor the surviving grey rectangles by top-row marker span, then compress the non-empty rows.

    Input structure:
    - Black background with 2-3 colored singleton control pixels in the top row.
    - Below the controls are several grey 4-connected rectangular objects plus smaller grey noise components.
    - Every surviving rectangle has exactly one control-pixel column inside its bounding-box column span.

    Rule:
    - Find all grey 4-connected components; erase every component of area 1 or 2 to black.
    - For each remaining grey component, compute its bounding-box column range and recolor the whole component with the unique top-row control color whose column lies inside that range.
    - Keep the top-row control pixels unchanged.
    - Starting from the second row, discard every all-black row and stack the remaining rows directly under the top row.

    Key insight:
    - Tiny grey components are disposable noise: only the larger grey rectangles determine the recoloring, and removing the noise also changes which rows survive the final vertical compression.
    """
    recolored, _large, _small = _erase_recolor(np.asarray(grid, dtype=int))
    return _compress_rows(recolored)


def _large_components(grid: np.ndarray) -> list[np.ndarray]:
    return [comp for comp in _components_mask(grid == GREY) if int(comp.shape[0]) >= 3]


def _spans_for_control(width: int, control_cols: list[int], idx: int) -> list[tuple[int, int]]:
    col = control_cols[idx]
    left_bound = 0 if idx == 0 else control_cols[idx - 1] + 1
    right_bound = width - 1 if idx == len(control_cols) - 1 else control_cols[idx + 1] - 1
    spans: list[tuple[int, int]] = []
    for rect_w in range(RECT_W_RANGE[0], RECT_W_RANGE[1] + 1):
        for c0 in range(left_bound, right_bound - rect_w + 2):
            c1 = c0 + rect_w - 1
            if c0 <= col <= c1:
                spans.append((c0, c1))
    return spans


def _sample_control_columns(width: int, count: int) -> list[int] | None:
    cols = list(range(1, width - 1))
    for _ in range(200):
        picked = sorted(random.sample(cols, count))
        if min((b - a for a, b in zip(picked, picked[1:])), default=CONTROL_MIN_GAP) < CONTROL_MIN_GAP:
            continue
        if all(_spans_for_control(width, picked, idx) for idx in range(count)):
            return picked
    return None


def _place_rows(heights: list[int]) -> tuple[list[int], int] | None:
    tops: list[int] = []
    cursor = 2
    for h in heights:
        cursor += random.randint(0, 2)
        tops.append(cursor)
        cursor += h
        cursor += random.randint(1, 3)
    total_h = cursor + random.randint(0, 2)
    if total_h > GRID_H_CAP:
        return None
    return tops, total_h


def _special_side_options(control_cols: list[int], rect_span: tuple[int, int], idx: int) -> list[tuple[str, int]]:
    c0, c1 = rect_span
    out: list[tuple[str, int]] = []
    if idx > 0 and c0 - control_cols[idx - 1] >= 2:
        out.append(("left", control_cols[idx - 1]))
    if idx + 1 < len(control_cols) and control_cols[idx + 1] - c1 >= 2:
        out.append(("right", control_cols[idx + 1]))
    return out


def _can_place_noise(grid: np.ndarray, cells: list[tuple[int, int]]) -> bool:
    h, w = grid.shape
    cell_set = set(cells)
    if not cells:
        return False
    for r, c in cells:
        if not (1 <= r < h and 0 <= c < w):
            return False
        if int(grid[r, c]) != BLACK:
            return False
    for r, c in cells:
        for nr, nc in _neighbors4(r, c):
            if (nr, nc) in cell_set:
                continue
            if 0 <= nr < h and 0 <= nc < w and int(grid[nr, nc]) == GREY:
                return False
    return True


def _place_noise_component(grid: np.ndarray, cells: list[tuple[int, int]]) -> bool:
    if not _can_place_noise(grid, cells):
        return False
    for r, c in cells:
        grid[r, c] = GREY
    return True


def _rows_occupied_by_rects(rects: list[dict]) -> set[int]:
    rows: set[int] = set()
    for rect in rects:
        rows.update(range(rect["r0"], rect["r1"] + 1))
    return rows


def _misleading_span_exists(grid: np.ndarray) -> bool:
    controls = _extract_controls(grid)
    large = [comp for comp in _components_mask(grid == GREY) if int(comp.shape[0]) >= 3]
    small = [comp for comp in _components_mask(grid == GREY) if int(comp.shape[0]) < 3]
    for comp in large:
        r0, c0, r1, c1 = _bbox_coords(comp)
        true_matches = [col for col, _color in controls if c0 <= col <= c1]
        if len(true_matches) != 1:
            continue
        true_col = true_matches[0]
        row_set = set(range(r0, r1 + 1))
        for noise in small:
            nr0, nc0, nr1, nc1 = _bbox_coords(noise)
            if not row_set.intersection(range(nr0, nr1 + 1)):
                continue
            union_c0 = min(c0, nc0)
            union_c1 = max(c1, nc1)
            union_cols = [col for col, _color in controls if union_c0 <= col <= union_c1]
            if any(col != true_col for col in union_cols):
                return True
    return False


def _no_erase_variant(grid: np.ndarray) -> np.ndarray:
    g = np.asarray(grid, dtype=int)
    out = g.copy()
    large = _large_components(g)
    controls = _extract_controls(g)
    for comp in large:
        _r0, c0, _r1, c1 = _bbox_coords(comp)
        matches = [color for col, color in controls if c0 <= col <= c1]
        if len(matches) == 1:
            out[comp[:, 0], comp[:, 1]] = matches[0]
    return _compress_rows(out)


def _no_compress_variant(grid: np.ndarray) -> np.ndarray:
    recolored, _large, _small = _erase_recolor(grid)
    return recolored


def _validate_generated(grid: np.ndarray) -> bool:
    if not isinstance(grid, np.ndarray) or grid.ndim != 2:
        return False
    if max(grid.shape) > 30:
        return False
    try:
        controls = _extract_controls(grid)
        recolored, large, small = _erase_recolor(grid)
        out = main(grid)
    except Exception:
        return False

    if len(controls) < MIN_CONTROLS or len(controls) > MAX_CONTROLS:
        return False
    if len(large) < MIN_RECTS or len(large) > MAX_RECTS:
        return False
    if not small:
        return False

    for comp in large:
        r0, c0, r1, c1 = _bbox_coords(comp)
        if int(comp.shape[0]) != (r1 - r0 + 1) * (c1 - c0 + 1):
            return False

    erased_only_noise_rows = [
        r
        for r in range(1, grid.shape[0])
        if bool(np.any(grid[r] == GREY)) and not bool(np.any(recolored[r] != BLACK))
    ]
    if not erased_only_noise_rows:
        return False
    if not _misleading_span_exists(grid):
        return False
    if np.array_equal(out, _no_erase_variant(grid)):
        return False
    if np.array_equal(out, _no_compress_variant(grid)):
        return False
    return True


def _gen_attempt() -> np.ndarray | None:
    width = random.randint(*GRID_W_RANGE)
    control_count = random.randint(MIN_CONTROLS, MAX_CONTROLS)
    control_cols = _sample_control_columns(width, control_count)
    if control_cols is None:
        return None
    control_colors = random.sample(list(CONTROL_COLORS), control_count)

    rect_count = random.randint(MIN_RECTS, MAX_RECTS)
    assignments = list(range(control_count))
    while len(assignments) < rect_count:
        assignments.append(random.randrange(control_count))
    random.shuffle(assignments)

    special_rect_idx = random.randrange(rect_count)
    special_control_idx = assignments[special_rect_idx]

    rect_specs: list[dict] = []
    special_noise_col = None
    for rect_idx, control_idx in enumerate(assignments):
        spans = _spans_for_control(width, control_cols, control_idx)
        if not spans:
            return None
        random.shuffle(spans)

        chosen_span = None
        chosen_side = None
        if rect_idx == special_rect_idx:
            for span in spans:
                side_opts = _special_side_options(control_cols, span, control_idx)
                if side_opts:
                    chosen_span = span
                    chosen_side, special_noise_col = random.choice(side_opts)
                    break
            if chosen_span is None:
                return None
        else:
            chosen_span = spans[0]

        c0, c1 = chosen_span
        rect_specs.append(
            {
                "control_idx": control_idx,
                "c0": c0,
                "c1": c1,
                "w": c1 - c0 + 1,
                "h": random.randint(*RECT_H_RANGE),
                "special_side": chosen_side,
            }
        )

    heights = [spec["h"] for spec in rect_specs]
    placed = _place_rows(heights)
    if placed is None:
        return None
    tops, height = placed

    grid = np.full((height, width), BLACK, dtype=int)
    for col, color in zip(control_cols, control_colors):
        grid[0, col] = color

    for spec, r0 in zip(rect_specs, tops):
        spec["r0"] = r0
        spec["r1"] = r0 + spec["h"] - 1
        grid[r0 : spec["r1"] + 1, spec["c0"] : spec["c1"] + 1] = GREY

    occupied_rows = _rows_occupied_by_rects(rect_specs)
    special_rect = rect_specs[special_rect_idx]
    if special_noise_col is None:
        return None

    candidate_rows = list(range(special_rect["r0"], special_rect["r1"] + 1))
    random.shuffle(candidate_rows)
    special_cells = None
    for row in candidate_rows:
        area_two = random.random() < 0.5 and row + 1 <= special_rect["r1"]
        cells = [(row, special_noise_col)]
        if area_two:
            cells.append((row + 1, special_noise_col))
        if _place_noise_component(grid, cells):
            special_cells = cells
            break
    if special_cells is None:
        return None

    empty_rows = [r for r in range(1, height) if r not in occupied_rows]
    random.shuffle(empty_rows)
    row_only_noise_done = False
    for row in empty_rows:
        cols = list(range(width))
        random.shuffle(cols)
        for col in cols:
            if _place_noise_component(grid, [(row, col)]):
                row_only_noise_done = True
                break
        if row_only_noise_done:
            break
    if not row_only_noise_done:
        return None

    extra_components = random.randint(1, 4)
    for _ in range(extra_components):
        for _try in range(60):
            col = random.randrange(width)
            row = random.randrange(1, height)
            if random.random() < 0.45 and row + 1 < height:
                cells = [(row, col), (row + 1, col)]
            elif random.random() < 0.45 and col + 1 < width:
                cells = [(row, col), (row, col + 1)]
            else:
                cells = [(row, col)]
            if _place_noise_component(grid, cells):
                break

    if not _validate_generated(grid):
        return None
    return grid


def generate_input() -> np.ndarray:
    for _ in range(GEN_ATTEMPTS):
        grid = _gen_attempt()
        if grid is not None:
            return grid
    raise RuntimeError("failed to generate a valid input for puzzle 6")
