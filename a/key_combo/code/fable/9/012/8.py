from __future__ import annotations

from collections import Counter, deque
import random

import numpy as np


Coord = tuple[int, int]
BBox = tuple[int, int, int, int]
DIR4 = ((1, 0), (-1, 0), (0, 1), (0, -1))
PALETTE = tuple(range(1, 10))
BG = 0
MAX_GEN_ATTEMPTS = 400


def _get_bg(grid: np.ndarray) -> int:
    return int(Counter(map(int, grid.ravel())).most_common(1)[0][0])


def _bbox_coords(coords: np.ndarray) -> BBox:
    r0, c0 = coords.min(axis=0)
    r1, c1 = coords.max(axis=0)
    return int(r0), int(c0), int(r1), int(c1)


def _components_mask(mask: np.ndarray) -> list[np.ndarray]:
    h, w = mask.shape
    seen = np.zeros((h, w), dtype=bool)
    components: list[np.ndarray] = []
    for r in range(h):
        for c in range(w):
            if not mask[r, c] or seen[r, c]:
                continue
            q = deque([(r, c)])
            seen[r, c] = True
            coords: list[Coord] = []
            while q:
                rr, cc = q.popleft()
                coords.append((rr, cc))
                for dr, dc in DIR4:
                    nr, nc = rr + dr, cc + dc
                    if 0 <= nr < h and 0 <= nc < w and mask[nr, nc] and not seen[nr, nc]:
                        seen[nr, nc] = True
                        q.append((nr, nc))
            components.append(np.array(coords, dtype=int))
    return components


def _find_components(grid: np.ndarray, bg: int) -> list[dict]:
    out: list[dict] = []
    for color in sorted(int(c) for c in np.unique(grid) if int(c) != bg):
        for coords in _components_mask(grid == color):
            out.append(
                {
                    "color": color,
                    "coords": coords,
                    "bbox": _bbox_coords(coords),
                    "area": int(coords.shape[0]),
                }
            )
    return out


def _bracket_shape(bbox: BBox, orientation: str) -> set[Coord]:
    r0, c0, r1, c1 = bbox
    cells = {(r, c0) for r in range(r0, r1 + 1)} | {(r, c1) for r in range(r0, r1 + 1)}
    if orientation == "down":
        cells |= {(r0, c) for c in range(c0, c1 + 1)}
    else:
        cells |= {(r1, c) for c in range(c0, c1 + 1)}
    return cells


def _classify_bracket(component: dict) -> str | None:
    coords = {tuple(map(int, p)) for p in component["coords"]}
    r0, c0, r1, c1 = component["bbox"]
    h = r1 - r0 + 1
    w = c1 - c0 + 1
    if h < 3 or w < 3:
        return None
    if component["area"] != w + 2 * (h - 1):
        return None
    if coords == _bracket_shape(component["bbox"], "down"):
        return "down"
    if coords == _bracket_shape(component["bbox"], "up"):
        return "up"
    return None


def _marker_inside(grid: np.ndarray, bg: int, bracket: dict) -> Coord:
    bracket_cells = {tuple(map(int, p)) for p in bracket["coords"]}
    r0, c0, r1, c1 = bracket["bbox"]
    markers = [
        (r, c)
        for r in range(r0, r1 + 1)
        for c in range(c0, c1 + 1)
        if (r, c) not in bracket_cells and int(grid[r, c]) != bg
    ]
    assert len(markers) == 1, "each bracket interior must contain exactly one marker"
    return markers[0]


def _selectors(grid: np.ndarray) -> list[dict]:
    bg = _get_bg(grid)
    selectors: list[dict] = []
    for component in _find_components(grid, bg):
        orientation = _classify_bracket(component)
        if orientation is None:
            continue
        marker = _marker_inside(grid, bg, component)
        marker_color = int(grid[marker])
        assert marker_color not in (bg, int(component["color"]))
        selectors.append(
            {
                "orientation": orientation,
                "color": int(component["color"]),
                "coords": component["coords"],
                "bbox": component["bbox"],
                "marker": marker,
                "marker_color": marker_color,
            }
        )
    assert len(selectors) == 2, "expected exactly two bracket selectors"
    assert {sel["orientation"] for sel in selectors} == {"down", "up"}
    assert len({sel["color"] for sel in selectors}) == 2
    assert len({sel["marker_color"] for sel in selectors}) == 2
    return selectors


def _protected_mask(shape: tuple[int, int], selectors: list[dict]) -> np.ndarray:
    mask = np.zeros(shape, dtype=bool)
    for selector in selectors:
        coords = selector["coords"]
        mask[coords[:, 0], coords[:, 1]] = True
        mr, mc = selector["marker"]
        mask[mr, mc] = True
    return mask


def _transform_with_roles(
    grid: np.ndarray,
    selectors: list[dict],
    erase_color: int,
    anchor_color: int,
    *,
    use_erasure: bool,
    use_anchor: bool,
) -> np.ndarray:
    out = np.asarray(grid, dtype=int).copy()
    protected = _protected_mask(out.shape, selectors)
    stack_mask = (out != BG) & ~protected
    out[stack_mask] = BG
    h, w = out.shape

    for c in range(w):
        stack_rows = np.flatnonzero(stack_mask[:, c])
        if stack_rows.size == 0:
            continue
        top = int(stack_rows[0])
        bottom = int(stack_rows[-1])
        assert bottom == h - 1, "stack columns must reach the canvas bottom"
        expected_rows = np.arange(top, h)
        assert np.array_equal(stack_rows, expected_rows), "stack columns must be contiguous bottom-up"

        anchor_rows = [
            int(r) for r in stack_rows
            if use_anchor and int(grid[r, c]) == anchor_color
        ]
        for r in anchor_rows:
            out[r, c] = anchor_color

        segment_starts = [top] + [r + 1 for r in anchor_rows]
        segment_ends = [r - 1 for r in anchor_rows] + [h - 1]
        for start, end in zip(segment_starts, segment_ends):
            if start > end:
                continue
            values = []
            for r in range(start, end + 1):
                value = int(grid[r, c])
                if use_anchor and value == anchor_color:
                    continue
                if use_erasure and value == erase_color:
                    continue
                values.append(value)
            for r, value in zip(range(end - len(values) + 1, end + 1), values):
                out[r, c] = value

    return out


def _solve(grid: np.ndarray) -> np.ndarray:
    selectors = _selectors(grid)
    erase_color = next(sel["marker_color"] for sel in selectors if sel["orientation"] == "down")
    anchor_color = next(sel["marker_color"] for sel in selectors if sel["orientation"] == "up")
    return _transform_with_roles(
        grid,
        selectors,
        erase_color,
        anchor_color,
        use_erasure=True,
        use_anchor=True,
    )


def main(grid: np.ndarray) -> np.ndarray:
    """
    Remove the color named by the downward bracket, keep the upward bracket's color anchored, and let the other stack cells collapse by column.

    Input structure:
    - A black background contains a bottom-reaching stack made of several continuous colored columns; each column is a contiguous suffix ending at the bottom edge.
    - Away from the stack sit two monochrome three-sided brackets of different colors: one opens downward and one opens upward.
    - Each bracket's bounding box contains exactly one non-background marker pixel outside the bracket cells; the two marker colors are distinct and both also occur in the stack.

    Rule:
    - The marker inside the downward-opening bracket names the extraction color: erase every stack cell of that color.
    - The marker inside the upward-opening bracket names the anchor color: every stack cell of that color stays fixed in place.
    - Then process each stack column independently. Within every segment between the stack top and the next anchor, or between consecutive anchors, all remaining non-anchor cells keep their top-to-bottom order and fall as far downward as possible, stopping at the column bottom or directly above the next anchor.
    - Preserve both brackets and both markers unchanged.

    Key insight:
    - The two selectors have the same geometry but opposite semantics, and the semantics come only from the opening direction: the downward bracket chooses what vanishes, while the upward bracket chooses the cells that become supports for the columnwise stable gravity.
    """
    g = np.asarray(grid, dtype=int)
    assert g.ndim == 2
    return _solve(g)


def _draw_selector(
    grid: np.ndarray,
    bbox: BBox,
    orientation: str,
    color: int,
    marker_color: int,
) -> tuple[set[Coord], Coord]:
    cells = _bracket_shape(bbox, orientation)
    for r, c in cells:
        grid[r, c] = color
    r0, c0, r1, c1 = bbox
    interior = [
        (r, c)
        for r in range(r0, r1 + 1)
        for c in range(c0 + 1, c1)
        if (r, c) not in cells
    ]
    marker = random.choice(interior)
    grid[marker] = marker_color
    return cells, marker


def _random_column(height: int, colors: list[int], erase_color: int, anchor_color: int) -> list[int]:
    values = [random.choice(colors) for _ in range(height)]
    if anchor_color not in values and random.random() < 0.45:
        values[random.randrange(height)] = anchor_color
    if erase_color not in values and random.random() < 0.55:
        values[random.randrange(height)] = erase_color
    return values


def _stack_column_top_row(height: int, stack_height: int) -> int:
    return height - stack_height


def _paint_stack(
    grid: np.ndarray,
    c0: int,
    columns: list[list[int]],
) -> None:
    h = grid.shape[0]
    for offset, values in enumerate(columns):
        top = _stack_column_top_row(h, len(values))
        for idx, value in enumerate(values):
            grid[top + idx, c0 + offset] = value


def _touches_stack(grid: np.ndarray, selectors: list[dict]) -> bool:
    protected = _protected_mask(grid.shape, selectors)
    stack_mask = (grid != BG) & ~protected
    h, w = grid.shape
    for r, c in np.argwhere(protected):
        for dr, dc in DIR4:
            nr, nc = int(r) + dr, int(c) + dc
            if 0 <= nr < h and 0 <= nc < w and stack_mask[nr, nc]:
                return True
    return False


def _column_feature_counts(
    grid: np.ndarray,
    selectors: list[dict],
    erase_color: int,
    anchor_color: int,
) -> dict[str, int]:
    protected = _protected_mask(grid.shape, selectors)
    stack_mask = (grid != BG) & ~protected
    result = {
        "anchor_columns": 0,
        "double_anchor_columns": 0,
        "columns_with_erase": 0,
        "columns_above_anchor": 0,
        "erase_cells": 0,
        "anchor_cells": 0,
    }
    for c in range(grid.shape[1]):
        rows = np.flatnonzero(stack_mask[:, c])
        if rows.size == 0:
            continue
        colors = [int(grid[r, c]) for r in rows]
        anchor_rows = [int(r) for r in rows if int(grid[r, c]) == anchor_color]
        erase_count = sum(value == erase_color for value in colors)
        if anchor_rows:
            result["anchor_columns"] += 1
            result["anchor_cells"] += len(anchor_rows)
            if len(anchor_rows) >= 2:
                result["double_anchor_columns"] += 1
            first_anchor = anchor_rows[0]
            if any(int(grid[r, c]) not in (erase_color, anchor_color) for r in rows if int(r) < first_anchor):
                result["columns_above_anchor"] += 1
        if erase_count:
            result["columns_with_erase"] += 1
            result["erase_cells"] += erase_count
    return result


def _validate_instance(grid: np.ndarray) -> bool:
    if grid.ndim != 2 or max(grid.shape) > 30:
        return False
    if _get_bg(grid) != BG:
        return False

    try:
        selectors = _selectors(grid)
    except AssertionError:
        return False
    if _touches_stack(grid, selectors):
        return False

    erase_color = next(sel["marker_color"] for sel in selectors if sel["orientation"] == "down")
    anchor_color = next(sel["marker_color"] for sel in selectors if sel["orientation"] == "up")
    protected = _protected_mask(grid.shape, selectors)
    stack_mask = (grid != BG) & ~protected
    if not stack_mask.any():
        return False

    features = _column_feature_counts(grid, selectors, erase_color, anchor_color)
    if features["anchor_columns"] < 2:
        return False
    if features["double_anchor_columns"] < 1:
        return False
    if features["columns_above_anchor"] < 1:
        return False
    if features["columns_with_erase"] < 2:
        return False
    if features["erase_cells"] < 3 or features["anchor_cells"] < 4:
        return False

    try:
        expected = _solve(grid)
        no_anchor = _transform_with_roles(
            grid,
            selectors,
            erase_color,
            anchor_color,
            use_erasure=True,
            use_anchor=False,
        )
        no_erase = _transform_with_roles(
            grid,
            selectors,
            erase_color,
            anchor_color,
            use_erasure=False,
            use_anchor=True,
        )
        swapped = _transform_with_roles(
            grid,
            selectors,
            anchor_color,
            erase_color,
            use_erasure=True,
            use_anchor=True,
        )
    except AssertionError:
        return False

    if np.array_equal(expected, grid):
        return False
    if np.array_equal(expected, no_anchor):
        return False
    if np.array_equal(expected, no_erase):
        return False
    if np.array_equal(expected, swapped):
        return False

    changed = int(np.count_nonzero(expected != grid))
    return changed >= 4


def _generate_attempt() -> np.ndarray | None:
    height = random.randint(18, 26)
    width = random.randint(18, 28)

    down_h = random.randint(4, 6)
    down_w = random.randint(4, 6)
    up_h = random.randint(4, 6)
    up_w = random.randint(4, 6)
    down_r0 = random.randint(1, 3)
    up_r0 = random.randint(1, 3)
    down_c0 = random.randint(1, 4)
    up_c0_min = max(width - up_w - 5, down_c0 + down_w + 4)
    up_c0_max = width - up_w - 1
    if up_c0_min > up_c0_max:
        return None
    up_c0 = random.randint(up_c0_min, up_c0_max)

    bracket_bottom = max(down_r0 + down_h - 1, up_r0 + up_h - 1)
    max_stack_height = min(12, height - bracket_bottom - 3)
    if max_stack_height < 7:
        return None

    bracket_colors = random.sample(list(PALETTE), 2)
    stack_palette = [c for c in PALETTE if c not in bracket_colors]
    erase_color, anchor_color = random.sample(stack_palette, 2)
    other_colors = [c for c in stack_palette if c not in (erase_color, anchor_color)]
    extra_count = random.randint(2, min(4, len(other_colors)))
    stack_colors = [erase_color, anchor_color] + random.sample(other_colors, extra_count)

    stack_width = random.randint(6, min(10, width - 4))
    stack_c0 = random.randint(2, width - stack_width - 2)
    column_count = stack_width

    columns: list[list[int]] = []
    multi_anchor_height = random.randint(7, max_stack_height)
    a, b, c = random.sample(other_colors, 3)
    multi_anchor = [a, anchor_color, b, anchor_color, c]
    while len(multi_anchor) < multi_anchor_height:
        insert_at = random.randint(0, len(multi_anchor))
        multi_anchor.insert(insert_at, random.choice(stack_colors))
    columns.append(multi_anchor)

    erase_anchor_height = random.randint(6, max_stack_height)
    d, e, f = random.sample(other_colors, 3)
    erase_anchor = [d, erase_color, e, anchor_color, f]
    while len(erase_anchor) < erase_anchor_height:
        erase_anchor.insert(random.randint(0, len(erase_anchor)), random.choice(stack_colors))
    columns.append(erase_anchor)

    plain_height = random.randint(5, max_stack_height)
    g, hcol = random.sample(other_colors, 2)
    plain = [g, erase_color, hcol, random.choice(other_colors)]
    while len(plain) < plain_height:
        plain.insert(random.randint(0, len(plain)), random.choice([erase_color] + other_colors))
    columns.append(plain)

    for _ in range(column_count - len(columns)):
        col_height = random.randint(5, max_stack_height)
        columns.append(_random_column(col_height, stack_colors, erase_color, anchor_color))
    random.shuffle(columns)

    grid = np.full((height, width), BG, dtype=int)
    _paint_stack(grid, stack_c0, columns)

    down_bbox = (down_r0, down_c0, down_r0 + down_h - 1, down_c0 + down_w - 1)
    up_bbox = (up_r0, up_c0, up_r0 + up_h - 1, up_c0 + up_w - 1)
    _draw_selector(grid, down_bbox, "down", bracket_colors[0], erase_color)
    _draw_selector(grid, up_bbox, "up", bracket_colors[1], anchor_color)

    if not _validate_instance(grid):
        return None
    return grid


def generate_input() -> np.ndarray:
    for _ in range(MAX_GEN_ATTEMPTS):
        grid = _generate_attempt()
        if grid is not None:
            return grid
    raise RuntimeError(f"generate_input: failed after {MAX_GEN_ATTEMPTS} attempts")
