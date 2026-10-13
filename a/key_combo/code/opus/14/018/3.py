from __future__ import annotations

import random
from collections import defaultdict, deque

import numpy as np


BG = 0
RED = 2
GREY = 5
CONTROL_COLORS = (1, 3, 4, 6, 7, 8, 9)
GEN_ATTEMPTS = 800


def _neighbors4(r: int, c: int):
    yield r - 1, c
    yield r + 1, c
    yield r, c - 1
    yield r, c + 1


def _components(mask: np.ndarray) -> list[list[tuple[int, int]]]:
    mask = np.asarray(mask, dtype=bool)
    h, w = mask.shape
    seen = np.zeros((h, w), dtype=bool)
    out: list[list[tuple[int, int]]] = []
    for r in range(h):
        for c in range(w):
            if not mask[r, c] or seen[r, c]:
                continue
            seen[r, c] = True
            q = deque([(r, c)])
            comp: list[tuple[int, int]] = []
            while q:
                rr, cc = q.popleft()
                comp.append((rr, cc))
                for nr, nc in _neighbors4(rr, cc):
                    if 0 <= nr < h and 0 <= nc < w and mask[nr, nc] and not seen[nr, nc]:
                        seen[nr, nc] = True
                        q.append((nr, nc))
            out.append(comp)
    return out


def _bbox(coords: list[tuple[int, int]] | tuple[tuple[int, int], ...]) -> tuple[int, int, int, int]:
    rows = [r for r, _ in coords]
    cols = [c for _, c in coords]
    return min(rows), min(cols), max(rows), max(cols)


def _is_solid_rectangle(coords: list[tuple[int, int]] | tuple[tuple[int, int], ...]) -> bool:
    r0, c0, r1, c1 = _bbox(coords)
    return len(coords) == (r1 - r0 + 1) * (c1 - c0 + 1)


def _is_hollow_rectangle(coords: list[tuple[int, int]] | tuple[tuple[int, int], ...]) -> bool:
    r0, c0, r1, c1 = _bbox(coords)
    if r1 - r0 + 1 < 3 or c1 - c0 + 1 < 3:
        return False
    want = set()
    for c in range(c0, c1 + 1):
        want.add((r0, c))
        want.add((r1, c))
    for r in range(r0, r1 + 1):
        want.add((r, c0))
        want.add((r, c1))
    return set(coords) == want


def _row_span_interior(grid: np.ndarray, coords: list[tuple[int, int]]) -> tuple[tuple[int, int], ...]:
    occupied = set(coords)
    by_row: dict[int, list[int]] = defaultdict(list)
    for r, c in coords:
        by_row[r].append(c)

    interior: list[tuple[int, int]] = []
    for r in sorted(by_row):
        cols = sorted(by_row[r])
        if len(cols) < 2:
            continue
        left = cols[0]
        right = cols[-1]
        for c in range(left + 1, right):
            if (r, c) not in occupied and int(grid[r, c]) == BG:
                interior.append((r, c))
    return tuple(interior)


def _outline_infos(grid: np.ndarray) -> list[dict]:
    infos = []
    for comp in _components(grid == GREY):
        box = _bbox(comp)
        interior = _row_span_interior(grid, comp)
        infos.append(
            {
                "coords": tuple(comp),
                "bbox": box,
                "interior": interior,
                "area": len(interior),
            }
        )
    return infos


def _control_pixels(grid: np.ndarray) -> list[tuple[int, int]]:
    return [(c, int(grid[0, c])) for c in range(grid.shape[1]) if int(grid[0, c]) != BG]


def _analyze_input(grid: np.ndarray) -> dict:
    g = np.asarray(grid, dtype=int)
    assert g.ndim == 2
    h, w = g.shape
    assert 4 <= h <= 30 and 4 <= w <= 30
    assert np.all((0 <= g) & (g <= 9)), "ARC colors must be in 0..9"

    controls = _control_pixels(g)
    assert 2 <= len(controls) <= 3, "expected 2 or 3 top-row control pixels"
    control_cols = [c for c, _ in controls]
    control_colors = [color for _, color in controls]
    assert len(set(control_cols)) == len(control_cols)
    assert len(set(control_colors)) == len(control_colors), "control colors must be distinct"
    assert all(color not in (BG, RED, GREY) for color in control_colors)

    red_components = _components(g == RED)
    assert len(red_components) == 1, "expected exactly one red component"
    red_comp = red_components[0]
    assert _is_solid_rectangle(red_comp), "red object must be a solid rectangle"
    red_area = len(red_comp)
    red_box = _bbox(red_comp)

    outlines = _outline_infos(g)
    assert 3 <= len(outlines) <= 4, "expected 3 or 4 grey outlines"
    areas = [info["area"] for info in outlines]
    assert len(areas) == len(set(areas)), "outline row-span areas must be distinct"

    target_matches = [info for info in outlines if info["area"] == red_area]
    assert len(target_matches) == 1, "expected exactly one area-matching outline"
    target = target_matches[0]

    for info in outlines:
        r0, c0, r1, c1 = info["bbox"]
        assert r1 - r0 + 1 >= 3 and c1 - c0 + 1 >= 3
        assert _is_hollow_rectangle(list(info["coords"])), "grey objects must be hollow rectangles"
        hits = [control for control in controls if c0 <= control[0] <= c1]
        assert len(hits) <= 1, "each outline may intersect at most one control column"
        info["hits"] = hits

    assert len(target["hits"]) == 1, "the area-matched outline must intersect exactly one control column"
    fill_col = target["hits"][0][1]

    rc0, rc1 = red_box[1], red_box[3]
    assert all(not (rc0 <= c <= rc1) for c in control_cols), "control columns must avoid the red rectangle"

    return {
        "controls": controls,
        "control_map": {c: color for c, color in controls},
        "red_area": red_area,
        "red_box": red_box,
        "outlines": outlines,
        "target": target,
        "fill_color": fill_col,
    }


def main(grid: np.ndarray) -> np.ndarray:
    """
    Use red-rectangle area to choose one grey outline, then use the intersecting control ray color to fill it.

    Input structure:
    - The background is black.
    - The top row contains 2 or 3 distinct non-black control pixels, each defining one vertical ray column and color.
    - Lower rows contain 3 or 4 grey hollow rectangles with distinct row-span interior areas, plus one red solid rectangle.

    Rule:
    - Count the red rectangle pixels to get N.
    - For each grey outline, compute its row-span interior: on every occupied row, count the black cells strictly between the leftmost and rightmost grey pixels.
    - Select the unique outline whose row-span interior area equals N.
    - Extend every top-row control downward in its own column, recoloring only black cells and leaving grey or red cells unchanged.
    - The unique control column intersecting the selected outline's column span supplies the fill color; repaint that outline's row-span interior with that color and erase the red rectangle to black.

    Key insight:
    - The target location and the fill color come from two independent channels: area matching chooses the outline, while ray-column intersection chooses the color.
    """
    g = np.asarray(grid, dtype=int)
    info = _analyze_input(g)
    out = g.copy()
    h, _ = out.shape

    for c, color in info["controls"]:
        for r in range(1, h):
            if int(out[r, c]) == BG:
                out[r, c] = color

    for r, c in info["target"]["interior"]:
        out[r, c] = info["fill_color"]

    out[g == RED] = BG
    return out


OUTLINE_OPTIONS_BY_AREA: dict[int, list[tuple[int, int]]] = defaultdict(list)
for outer_h in range(3, 6):
    for outer_w in range(4, 9):
        OUTLINE_OPTIONS_BY_AREA[(outer_h - 2) * (outer_w - 2)].append((outer_h, outer_w))

RED_OPTIONS_BY_AREA: dict[int, list[tuple[int, int]]] = defaultdict(list)
for rect_h in range(2, 6):
    for rect_w in range(2, 6):
        RED_OPTIONS_BY_AREA[rect_h * rect_w].append((rect_h, rect_w))

TARGET_AREAS = sorted(set(OUTLINE_OPTIONS_BY_AREA) & set(RED_OPTIONS_BY_AREA))
DISTRACTOR_AREAS = sorted(OUTLINE_OPTIONS_BY_AREA)


def _pick_control_columns(width: int, count: int) -> list[int] | None:
    candidates = list(range(1, width - 1))
    chosen: list[int] = []
    for _ in range(count):
        valid = [c for c in candidates if all(abs(c - prev) >= 3 for prev in chosen)]
        if not valid:
            return None
        chosen.append(random.choice(valid))
    return sorted(chosen)


def _pack_rows(heights: list[int], height: int) -> list[int] | None:
    need = sum(heights) + (len(heights) - 1)
    free = (height - 1) - need
    if free < 0:
        return None
    extras = [0] * (len(heights) + 1)
    for _ in range(free):
        extras[random.randrange(len(extras))] += 1
    rows = []
    cur = 1 + extras[0]
    for i, h in enumerate(heights):
        rows.append(cur)
        cur += h
        if i + 1 < len(heights):
            cur += 1 + extras[i + 1]
    return rows


def _free_segments(width: int, control_cols: list[int]) -> list[tuple[int, int]]:
    segments = []
    prev = -1
    for col in control_cols + [width]:
        start = prev + 1
        end = col - 1
        if start <= end:
            segments.append((start, end))
        prev = col
    return segments


def _column_options(
    width: int,
    control_cols: list[int],
    span_w: int,
    assigned_control: int | None,
    mode: str,
) -> list[int]:
    if assigned_control is None:
        options: list[int] = []
        for start, end in _free_segments(width, control_cols):
            if end - start + 1 >= span_w:
                for c0 in range(start, end - span_w + 2):
                    options.append(c0)
        return options

    control_col = control_cols[assigned_control]
    left_bound = control_cols[assigned_control - 1] + 1 if assigned_control > 0 else 0
    right_bound = control_cols[assigned_control + 1] - 1 if assigned_control + 1 < len(control_cols) else width - 1

    if mode == "left":
        c0 = control_col
        return [c0] if c0 + span_w - 1 <= right_bound else []
    if mode == "right":
        c0 = control_col - span_w + 1
        return [c0] if c0 >= left_bound else []

    start_lo = max(left_bound, control_col - (span_w - 2))
    start_hi = min(control_col - 1, right_bound - span_w + 1)
    return [c0 for c0 in range(start_lo, start_hi + 1) if c0 < control_col < c0 + span_w - 1]


def _draw_outline(grid: np.ndarray, top: int, left: int, outer_h: int, outer_w: int) -> None:
    bottom = top + outer_h - 1
    right = left + outer_w - 1
    grid[top, left : right + 1] = GREY
    grid[bottom, left : right + 1] = GREY
    grid[top : bottom + 1, left] = GREY
    grid[top : bottom + 1, right] = GREY


def _expanded_blocked(mask: np.ndarray) -> np.ndarray:
    h, w = mask.shape
    out = mask.copy()
    points = np.argwhere(mask)
    for r, c in points:
        r0 = max(0, int(r) - 1)
        r1 = min(h, int(r) + 2)
        c0 = max(0, int(c) - 1)
        c1 = min(w, int(c) + 2)
        out[r0:r1, c0:c1] = True
    return out


def _place_red_rectangle(
    grid: np.ndarray,
    red_h: int,
    red_w: int,
    control_cols: list[int],
) -> tuple[int, int] | None:
    h, w = grid.shape
    blocked = _expanded_blocked(grid != BG)
    options = []
    for r0 in range(1, h - red_h + 1):
        for c0 in range(0, w - red_w + 1):
            c1 = c0 + red_w - 1
            if any(c0 <= col <= c1 for col in control_cols):
                continue
            if np.any(blocked[r0 : r0 + red_h, c0 : c0 + red_w]):
                continue
            options.append((r0, c0))
    return random.choice(options) if options else None


def _build_expected_cellwise(grid: np.ndarray) -> np.ndarray:
    info = _analyze_input(grid)
    target_cells = set(info["target"]["interior"])
    control_map = info["control_map"]
    out = np.asarray(grid, dtype=int).copy()
    h, w = out.shape
    for r in range(h):
        for c in range(w):
            if r == 0 or int(grid[r, c]) == GREY:
                continue
            if int(grid[r, c]) == RED:
                out[r, c] = BG
                continue
            if (r, c) in target_cells:
                out[r, c] = info["fill_color"]
                continue
            if c in control_map and int(grid[r, c]) == BG:
                out[r, c] = control_map[c]
    return out


def _validate_generated_case(grid: np.ndarray) -> None:
    info = _analyze_input(grid)
    out = main(grid)
    assert out.shape == grid.shape
    assert not np.array_equal(grid, out), "output must differ from input"
    assert np.array_equal(out, _build_expected_cellwise(grid))

    target_cells = set(info["target"]["interior"])
    ray_cols = set(info["control_map"])
    for outline in info["outlines"]:
        for r, c in outline["interior"]:
            if (r, c) in target_cells:
                assert int(out[r, c]) == info["fill_color"]
            elif c in ray_cols:
                assert int(out[r, c]) == info["control_map"][c]
            else:
                assert int(out[r, c]) == BG

    assert np.count_nonzero(out == RED) == 0
    for c, color in info["controls"]:
        painted = sum(1 for r in range(1, grid.shape[0]) if int(grid[r, c]) == BG and int(out[r, c]) == color)
        assert painted > 0, "every control ray should visibly paint some background"


def _generate_one() -> np.ndarray | None:
    width = random.randint(15, 19)
    control_count = random.randint(2, 3)
    control_cols = _pick_control_columns(width, control_count)
    if control_cols is None:
        return None

    outline_count = random.choice((3, 3, 4))
    target_area = random.choice(TARGET_AREAS)
    other_areas = random.sample([a for a in DISTRACTOR_AREAS if a != target_area], outline_count - 1)
    all_areas = other_areas + [target_area]
    random.shuffle(all_areas)
    target_index = all_areas.index(target_area)

    outline_dims = [random.choice(OUTLINE_OPTIONS_BY_AREA[area]) for area in all_areas]
    heights = [dims[0] for dims in outline_dims]
    min_height = 1 + sum(heights) + (outline_count - 1)
    if min_height > 19:
        return None
    height = random.randint(max(15, min_height), 19)
    row_starts = _pack_rows(heights, height)
    if row_starts is None:
        return None

    control_colors = random.sample(CONTROL_COLORS, control_count)
    target_control = random.randrange(control_count)

    assignments: list[int | None] = [None] * outline_count
    assignments[target_index] = target_control
    other_indices = [i for i in range(outline_count) if i != target_index]
    random.shuffle(other_indices)
    for idx in other_indices:
        if random.random() < 0.45:
            assignments[idx] = random.randrange(control_count)

    lefts: list[int] = []
    for idx, (outer_h, outer_w) in enumerate(outline_dims):
        assigned = assignments[idx]
        modes = ["inside", "left", "right"] if assigned is not None else ["inside"]
        if assigned is None:
            options = _column_options(width, control_cols, outer_w, None, "inside")
        else:
            random.shuffle(modes)
            options = []
            for mode in modes:
                options = _column_options(width, control_cols, outer_w, assigned, mode)
                if options:
                    break
        if not options:
            return None
        lefts.append(random.choice(options))

    red_h, red_w = random.choice(RED_OPTIONS_BY_AREA[target_area])
    grid = np.zeros((height, width), dtype=int)
    grid[0, control_cols] = np.array(control_colors, dtype=int)

    for top, left, (outer_h, outer_w) in zip(row_starts, lefts, outline_dims):
        _draw_outline(grid, top, left, outer_h, outer_w)

    red_pos = _place_red_rectangle(grid, red_h, red_w, control_cols)
    if red_pos is None:
        return None
    rr, rc = red_pos
    grid[rr : rr + red_h, rc : rc + red_w] = RED

    try:
        _validate_generated_case(grid)
    except AssertionError:
        return None
    return grid


def generate_input() -> np.ndarray:
    for _ in range(GEN_ATTEMPTS):
        grid = _generate_one()
        if grid is not None:
            return grid
    raise RuntimeError("failed to generate a valid input for puzzle 3")
