from __future__ import annotations

from collections import Counter, deque
import random

import numpy as np


Coord = tuple[int, int]
BBox = tuple[int, int, int, int]

BG = 0
MARKER_COLOR = 5
ACTIVE_COLORS = (1, 2, 3, 4, 6, 7, 8, 9)
DIRS4 = ((1, 0), (-1, 0), (0, 1), (0, -1))


def _get_bg(grid: np.ndarray) -> int:
    return int(Counter(map(int, grid.flat)).most_common(1)[0][0])


def _colors_present(grid: np.ndarray, exclude: set[int] | None = None) -> list[int]:
    colors = sorted(set(map(int, grid.flat)))
    if exclude is None:
        return colors
    return [color for color in colors if color not in exclude]


def _coords_to_bbox(coords: np.ndarray) -> BBox:
    r0, c0 = coords.min(axis=0)
    r1, c1 = coords.max(axis=0)
    return int(r0), int(c0), int(r1), int(c1)


def _bbox_area(box: BBox) -> int:
    r0, c0, r1, c1 = box
    return int(r1 - r0 + 1) * int(c1 - c0 + 1)


def _touches_any_edge(box: BBox, shape: tuple[int, int]) -> bool:
    r0, c0, r1, c1 = box
    h, w = shape
    return bool(r0 == 0 or c0 == 0 or r1 == h - 1 or c1 == w - 1)


def _touches_move_edge(box: BBox, shape: tuple[int, int], axis: int) -> bool:
    h, w = shape
    if axis == 0:
        return bool(box[0] == 0 or box[2] == h - 1)
    return bool(box[1] == 0 or box[3] == w - 1)


def _find_components_of_color(grid: np.ndarray, color: int) -> list[dict]:
    h, w = grid.shape
    seen = np.zeros((h, w), dtype=bool)
    components: list[dict] = []
    for r in range(h):
        for c in range(w):
            if seen[r, c] or int(grid[r, c]) != int(color):
                continue
            queue = deque([(r, c)])
            seen[r, c] = True
            coords: list[Coord] = []
            while queue:
                rr, cc = queue.popleft()
                coords.append((rr, cc))
                for dr, dc in DIRS4:
                    nr, nc = rr + dr, cc + dc
                    if 0 <= nr < h and 0 <= nc < w and not seen[nr, nc] and int(grid[nr, nc]) == int(color):
                        seen[nr, nc] = True
                        queue.append((nr, nc))
            arr = np.array(coords, dtype=int)
            components.append(
                {
                    "color": int(color),
                    "coords": arr,
                    "bbox": _coords_to_bbox(arr),
                    "area": int(len(coords)),
                }
            )
    return components


def _find_marker_line(grid: np.ndarray, bg: int) -> tuple[int, int, int, list[int]]:
    candidates: list[tuple[int, int, int, list[int]]] = []
    for color in _colors_present(grid, exclude={bg}):
        coords = np.argwhere(grid == color)
        if len(coords) < 3:
            continue
        rows = {int(v) for v in coords[:, 0]}
        cols = {int(v) for v in coords[:, 1]}
        if (len(rows) == 1) ^ (len(cols) == 1):
            axis = 0 if len(rows) == 1 else 1
            line = int(coords[0, axis])
            positions = sorted(int(v) for v in coords[:, 1 - axis])
            candidates.append((int(color), axis, line, positions))
    assert len(candidates) == 1, "expected exactly one sparse marker row or column"
    return candidates[0]


def _translated_coords(coords: np.ndarray, axis: int, delta: int) -> np.ndarray:
    moved = coords.copy()
    moved[:, axis] += int(delta)
    return moved


def _solid_rectangle_bbox(coords: set[Coord]) -> BBox | None:
    arr = np.array(sorted(coords), dtype=int)
    box = _coords_to_bbox(arr)
    return box if _bbox_area(box) == len(coords) else None


def _completion_candidates(
    grid: np.ndarray,
    source_coords: np.ndarray,
    anchor_coords: np.ndarray,
    move_axis: int,
    bg: int,
) -> list[dict]:
    h, w = grid.shape
    span = h if move_axis == 0 else w
    source_min = int(source_coords[:, move_axis].min())
    source_max = int(source_coords[:, move_axis].max())

    blocked = {tuple(map(int, rc)) for rc in np.argwhere(grid != bg)}
    source_set = {tuple(map(int, rc)) for rc in source_coords}
    anchor_set = {tuple(map(int, rc)) for rc in anchor_coords}
    blocked -= source_set

    candidates: list[dict] = []
    for delta in range(-source_min, span - source_max):
        if delta == 0:
            continue
        moved = _translated_coords(source_coords, move_axis, delta)
        moved_set = [tuple(map(int, rc)) for rc in moved]
        if any(cell in blocked for cell in moved_set):
            continue
        if len(set(moved_set)) != len(moved_set):
            continue
        if set(moved_set) & anchor_set:
            continue
        union = set(moved_set) | anchor_set
        box = _solid_rectangle_bbox(union)
        if box is None:
            continue
        candidates.append({"delta": int(delta), "bbox": box, "moved": moved})
    return candidates


def _apply_completions(grid: np.ndarray) -> tuple[np.ndarray, list[BBox], int, int, int, list[int]]:
    g = np.asarray(grid, dtype=int)
    bg = _get_bg(g)
    marker_color, marker_axis, marker_line, marker_positions = _find_marker_line(g, bg)
    move_axis = marker_axis

    active_colors = _colors_present(g, exclude={bg, marker_color})
    assert 1 <= len(active_colors) <= 2, "expected one or two active colors"

    out = g.copy()
    rectangles: list[BBox] = []
    for color in active_colors:
        comps = _find_components_of_color(g, color)
        assert len(comps) == 2, "each active color must have exactly two components"
        edge_parts = [comp for comp in comps if _touches_move_edge(comp["bbox"], g.shape, move_axis)]
        anchor_parts = [comp for comp in comps if not _touches_any_edge(comp["bbox"], g.shape)]
        assert len(edge_parts) == 1, "expected one edge fragment per active color"
        assert len(anchor_parts) == 1, "expected one interior anchor fragment per active color"
        source = edge_parts[0]
        anchor = anchor_parts[0]
        assert source is not anchor

        candidates = _completion_candidates(g, source["coords"], anchor["coords"], move_axis, bg)
        assert len(candidates) == 1, "expected a unique rectangle-completing translation"
        cand = candidates[0]

        for r, c in source["coords"]:
            out[int(r), int(c)] = bg
        for r, c in cand["moved"]:
            rr, cc = int(r), int(c)
            assert int(out[rr, cc]) == bg, "translated fragment must land on background"
            out[rr, cc] = int(color)
        rectangles.append(cand["bbox"])

    return out, rectangles, marker_color, marker_axis, marker_line, marker_positions


def _apply_mirrors(
    grid: np.ndarray,
    rectangles: list[BBox],
    marker_color: int,
    marker_axis: int,
    marker_line: int,
    bg: int,
) -> np.ndarray:
    out = grid.copy()
    h, w = out.shape
    for r0, c0, r1, c1 in rectangles:
        for r in range(r0, r1 + 1):
            for c in range(c0, c1 + 1):
                if marker_axis == 1:
                    rr, cc = r, 2 * marker_line - c
                else:
                    rr, cc = 2 * marker_line - r, c
                if 0 <= rr < h and 0 <= cc < w and int(out[rr, cc]) == int(bg):
                    out[rr, cc] = int(marker_color)
    return out


def _mirror_without_move(grid: np.ndarray) -> np.ndarray:
    g = np.asarray(grid, dtype=int)
    bg = _get_bg(g)
    marker_color, marker_axis, marker_line, _ = _find_marker_line(g, bg)
    out = g.copy()
    for color in _colors_present(g, exclude={bg, marker_color}):
        for r, c in np.argwhere(g == color):
            rr, cc = int(r), int(c)
            if marker_axis == 1:
                mr, mc = rr, 2 * marker_line - cc
            else:
                mr, mc = 2 * marker_line - rr, cc
            if 0 <= mr < g.shape[0] and 0 <= mc < g.shape[1] and int(out[mr, mc]) == int(bg):
                out[mr, mc] = int(marker_color)
    return out


def main(grid: np.ndarray) -> np.ndarray:
    """
    Translate each edge fragment to complete its anchored rectangle, then mirror each completed rectangle across the sparse marker line.

    Input structure:
    - The grid has black background, one sparse marker row or column, and one or two active foreground colors.
    - For each active color there are exactly two 4-connected fragments: one interior anchor fragment and one same-colored fragment touching the grid edge parallel to the marker line.
    - A unique translation of the edge fragment along the marker-line axis makes the two same-colored fragments form one filled axis-aligned rectangle.

    Rule:
    - For each active color, erase the edge fragment from its original location and translate it by the unique amount that completes a solid rectangle together with the anchor fragment.
    - Keep the completed rectangles and the sparse marker line in place.
    - Reflect every pixel of each completed rectangle across the marker row or column, painting only background landing cells with the marker color and discarding out-of-bounds reflected pixels.

    Key insight:
    - The mirror source is not a pre-existing object: it is the rectangle produced only after the fragment-completion step, so solving requires the move first and the reflection second.
    """
    assembled, rectangles, marker_color, marker_axis, marker_line, _marker_positions = _apply_completions(grid)
    bg = _get_bg(np.asarray(grid, dtype=int))
    return _apply_mirrors(assembled, rectangles, marker_color, marker_axis, marker_line, bg)


def _sample_widths(height: int, width: int) -> list[int] | None:
    for _ in range(200):
        widths = [random.randint(1, width - 1) for _ in range(height)]
        source_area = sum(widths)
        anchor_area = height * width - source_area
        if not (3 <= source_area <= 6 and 3 <= anchor_area <= 8):
            continue
        if len(set(widths)) == 1:
            continue
        return widths
    return None


def _build_case() -> tuple[np.ndarray, np.ndarray] | None:
    n_colors = random.choice((1, 2))
    transpose_case = random.choice((False, True))
    clip_branch = random.random() < 0.35

    h = random.randint(12, 18)
    w = random.randint(13, 19)
    grid_in = np.full((h, w), BG, dtype=int)
    grid_out = np.full((h, w), BG, dtype=int)

    colors = random.sample(ACTIVE_COLORS, n_colors)

    rect_specs = []
    heights = []
    gaps = []
    for idx in range(n_colors):
        rect_h = random.randint(2, 4)
        rect_w = random.randint(3, 5)
        widths = _sample_widths(rect_h, rect_w)
        if widths is None:
            return None
        heights.append(rect_h)
        if idx:
            gaps.append(random.randint(1, 2))
        rect_specs.append({"h": rect_h, "w": rect_w, "widths": widths})

    total_height = sum(heights) + sum(gaps)
    if total_height > h - 2:
        return None

    top_margin = random.randint(1, h - 1 - total_height)
    cur_top = top_margin
    for idx, spec in enumerate(rect_specs):
        spec["top"] = cur_top
        cur_top += spec["h"]
        if idx < len(gaps):
            cur_top += gaps[idx]

    max_rect_w = max(spec["w"] for spec in rect_specs)
    min_line = max_rect_w + 3
    max_line = w - 3
    if min_line > max_line:
        return None
    line_col = random.randint(min_line, max_line)

    fit_threshold = max(2, 2 * line_col - (w - 1))
    clip_index = random.randrange(n_colors) if clip_branch else -1
    clip_realized = False

    for idx, spec in enumerate(rect_specs):
        left_min = 2
        left_max = line_col - spec["w"] - 1
        if left_min > left_max:
            return None
        if idx == clip_index:
            clip_max = min(left_max, fit_threshold - 1)
            if left_min <= clip_max:
                spec["left"] = random.randint(left_min, clip_max)
                clip_realized = True
            else:
                spec["left"] = random.randint(max(left_min, fit_threshold), left_max)
        else:
            fit_min = max(left_min, fit_threshold)
            if fit_min > left_max:
                return None
            spec["left"] = random.randint(fit_min, left_max)

    if clip_branch and not clip_realized:
        return None

    marker_count = random.randint(4, min(7, h))
    marker_rows = sorted(random.sample(range(h), marker_count))
    if marker_rows[-1] - marker_rows[0] < 3:
        return None
    if all(b - a == 1 for a, b in zip(marker_rows, marker_rows[1:])):
        return None

    grid_in[marker_rows, line_col] = MARKER_COLOR
    grid_out[marker_rows, line_col] = MARKER_COLOR

    rectangles: list[BBox] = []
    for color, spec in zip(colors, rect_specs):
        top, left, rect_h, rect_w = spec["top"], spec["left"], spec["h"], spec["w"]
        widths = spec["widths"]
        rectangles.append((top, left, top + rect_h - 1, left + rect_w - 1))
        grid_out[top : top + rect_h, left : left + rect_w] = color

        for offset, width_i in enumerate(widths):
            row = top + offset
            anchor_start = left + width_i
            grid_in[row, anchor_start : left + rect_w] = color
            grid_in[row, 0:width_i] = color

    mirrored = _apply_mirrors(grid_out, rectangles, MARKER_COLOR, 1, line_col, BG)
    grid_out = mirrored

    if transpose_case:
        grid_in = grid_in.T.copy()
        grid_out = grid_out.T.copy()

    return grid_in, grid_out


def _validate_generated(inp: np.ndarray, expected: np.ndarray) -> bool:
    if inp.ndim != 2 or expected.ndim != 2:
        return False
    if max(inp.shape) > 30 or max(expected.shape) > 30:
        return False

    try:
        produced = main(inp)
    except AssertionError:
        return False
    if not np.array_equal(produced, expected):
        return False

    try:
        assembled, rectangles, marker_color, marker_axis, marker_line, marker_positions = _apply_completions(inp)
    except AssertionError:
        return False
    if marker_color != MARKER_COLOR:
        return False
    if marker_axis not in (0, 1):
        return False
    if len(marker_positions) < 4:
        return False
    if np.array_equal(assembled, expected):
        return False
    if np.array_equal(_mirror_without_move(inp), expected):
        return False

    bg = _get_bg(inp)
    active_colors = _colors_present(inp, exclude={bg, marker_color})
    if len(active_colors) not in (1, 2):
        return False

    for color in active_colors:
        comps = _find_components_of_color(inp, color)
        if len(comps) != 2:
            return False
        edge_parts = [comp for comp in comps if _touches_move_edge(comp["bbox"], inp.shape, marker_axis)]
        anchor_parts = [comp for comp in comps if not _touches_any_edge(comp["bbox"], inp.shape)]
        if len(edge_parts) != 1 or len(anchor_parts) != 1:
            return False
        if len(_completion_candidates(inp, edge_parts[0]["coords"], anchor_parts[0]["coords"], marker_axis, bg)) != 1:
            return False

    for box in rectangles:
        if _bbox_area(box) < 6:
            return False

    return True


def generate_input() -> np.ndarray:
    for _ in range(500):
        built = _build_case()
        if built is None:
            continue
        inp, expected = built
        if _validate_generated(inp, expected):
            return inp
    raise RuntimeError("failed to generate a valid puzzle instance")
