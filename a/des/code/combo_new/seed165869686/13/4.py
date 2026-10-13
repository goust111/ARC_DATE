from __future__ import annotations

import random

import numpy as np


BG = 0
GREY = 5
PALETTE = [1, 2, 3, 4, 6, 7, 8, 9]
DIRS4 = ((1, 0), (-1, 0), (0, 1), (0, -1))

GEN_RETRIES = 400
GRID_H_RANGE = (20, 30)
GRID_W_RANGE = (20, 30)
LOOKUP_OFFSET_RANGE = (0, 1)
MAX_EXTRA_BLOCKS = 2

TOUCHSETS_BY_COUNT = {
    0: [frozenset()],
    1: [frozenset([edge]) for edge in ("top", "bottom", "left", "right")],
    2: [
        frozenset(["top", "left"]),
        frozenset(["top", "right"]),
        frozenset(["bottom", "left"]),
        frozenset(["bottom", "right"]),
        frozenset(["top", "bottom"]),
        frozenset(["left", "right"]),
    ],
    3: [
        frozenset(["top", "bottom", "left"]),
        frozenset(["top", "bottom", "right"]),
        frozenset(["top", "left", "right"]),
        frozenset(["bottom", "left", "right"]),
    ],
}
MAX_FREE_DIM_BY_COUNT = {0: 5, 1: 7, 2: 6, 3: 4}


def _neighbors4(r: int, c: int) -> tuple[tuple[int, int], ...]:
    return ((r + 1, c), (r - 1, c), (r, c + 1), (r, c - 1))


def _bbox_from_coords(coords: list[tuple[int, int]]) -> tuple[int, int, int, int]:
    rows = [r for r, _c in coords]
    cols = [c for _r, c in coords]
    return min(rows), min(cols), max(rows), max(cols)


def _all_components(grid: np.ndarray, exclude_mask: np.ndarray | None = None) -> list[dict]:
    g = np.asarray(grid, dtype=int)
    h, w = g.shape
    seen = np.zeros((h, w), dtype=bool)
    if exclude_mask is not None:
        seen |= np.asarray(exclude_mask, dtype=bool)

    comps: list[dict] = []
    for r in range(h):
        for c in range(w):
            if seen[r, c] or int(g[r, c]) == BG:
                continue

            color = int(g[r, c])
            stack = [(r, c)]
            seen[r, c] = True
            coords: list[tuple[int, int]] = []

            while stack:
                rr, cc = stack.pop()
                coords.append((int(rr), int(cc)))
                for nr, nc in _neighbors4(rr, cc):
                    if 0 <= nr < h and 0 <= nc < w and not seen[nr, nc] and int(g[nr, nc]) == color:
                        seen[nr, nc] = True
                        stack.append((int(nr), int(nc)))

            bbox = _bbox_from_coords(coords)
            comps.append(
                {
                    "color": color,
                    "coords": coords,
                    "bbox": bbox,
                    "area": len(coords),
                }
            )

    return comps


def _components_of_color(grid: np.ndarray, color: int) -> list[dict]:
    g = np.asarray(grid, dtype=int)
    h, w = g.shape
    seen = np.zeros((h, w), dtype=bool)
    comps: list[dict] = []

    for r in range(h):
        for c in range(w):
            if seen[r, c] or int(g[r, c]) != color:
                continue

            stack = [(r, c)]
            seen[r, c] = True
            coords: list[tuple[int, int]] = []

            while stack:
                rr, cc = stack.pop()
                coords.append((int(rr), int(cc)))
                for nr, nc in _neighbors4(rr, cc):
                    if 0 <= nr < h and 0 <= nc < w and not seen[nr, nc] and int(g[nr, nc]) == color:
                        seen[nr, nc] = True
                        stack.append((int(nr), int(nc)))

            bbox = _bbox_from_coords(coords)
            comps.append(
                {
                    "color": color,
                    "coords": coords,
                    "bbox": bbox,
                    "area": len(coords),
                }
            )

    return comps


def _is_frame_component(component: dict) -> bool:
    r0, c0, r1, c1 = component["bbox"]
    h = r1 - r0 + 1
    w = c1 - c0 + 1
    if h < 3 or w < 3:
        return False

    mask = np.zeros((h, w), dtype=bool)
    for r, c in component["coords"]:
        mask[r - r0, c - c0] = True

    expected = np.zeros((h, w), dtype=bool)
    expected[0, :] = True
    expected[-1, :] = True
    expected[:, 0] = True
    expected[:, -1] = True
    return np.array_equal(mask, expected)


def _parse_lookup_frame(grid: np.ndarray, bbox: tuple[int, int, int, int]) -> dict[int, int] | None:
    r0, c0, r1, c1 = bbox
    interior = np.asarray(grid[r0 + 1 : r1, c0 + 1 : c1], dtype=int)
    mapping: dict[int, int] = {}
    nonempty_rows = 0

    for row in interior:
        nz = np.flatnonzero(row != BG)
        if len(nz) == 0:
            continue

        nonempty_rows += 1
        if len(nz) != int(nz[-1] - nz[0] + 1):
            return None

        segment = row[nz[0] : nz[-1] + 1]
        count = len(segment) - 1
        if not 1 <= count <= 4:
            return None
        if not np.all(segment[:-1] == GREY):
            return None

        color = int(segment[-1])
        if color in (BG, GREY) or count in mapping:
            return None
        mapping[count] = color

    if nonempty_rows != 4 or set(mapping) != {1, 2, 3, 4}:
        return None
    return mapping


def _find_lookup(grid: np.ndarray) -> tuple[tuple[int, int, int, int], dict[int, int]]:
    candidates: list[tuple[tuple[int, int, int, int], dict[int, int]]] = []
    for component in _components_of_color(grid, GREY):
        if not _is_frame_component(component):
            continue
        mapping = _parse_lookup_frame(grid, component["bbox"])
        if mapping is not None:
            candidates.append((component["bbox"], mapping))

    assert len(candidates) == 1, "expected exactly one valid lookup table"
    return candidates[0]


def _is_solid_rectangle(grid: np.ndarray, component: dict) -> bool:
    r0, c0, r1, c1 = component["bbox"]
    color = int(component["color"])
    return int(component["area"]) == (r1 - r0 + 1) * (c1 - c0 + 1) and np.all(
        np.asarray(grid[r0 : r1 + 1, c0 : c1 + 1], dtype=int) == color
    )


def _touch_count(bbox: tuple[int, int, int, int], h: int, w: int) -> int:
    r0, c0, r1, c1 = bbox
    return int(r0 == 0) + int(r1 == h - 1) + int(c0 == 0) + int(c1 == w - 1)


def main(grid: np.ndarray) -> np.ndarray:
    """
    Recolor each solid rectangle by looking up the number of grid borders it touches.

    Input structure:
    - The background is black.
    - Several solid monochrome rectangles are scattered around the grid; some touch one or more outer borders and some stay fully internal.
    - One corner region contains a grey-framed lookup table whose four rows show 1-4 grey marker cells followed immediately by one colored pixel.

    Rule:
    - Detect the lookup table and read its mapping from marker-count 1, 2, 3, and 4 to replacement colors.
    - Ignore the lookup table itself, then find every other 4-connected monochrome component and treat each one as a solid rectangle.
    - Count how many of the four outer grid borders each rectangle touches; keep rectangles with count 0 unchanged, and recolor every rectangle with count >= 1 to the lookup color for that count.

    Key insight:
    - The recolor key is not the rectangle's size or position alone but the hidden count of distinct outer borders it meets, which must then be translated through the legend.
    """
    g = np.asarray(grid, dtype=int)
    assert g.ndim == 2, "expected a 2D grid"
    h, w = g.shape

    lookup_bbox, mapping = _find_lookup(g)
    r0, c0, r1, c1 = lookup_bbox

    exclude = np.zeros((h, w), dtype=bool)
    exclude[r0 : r1 + 1, c0 : c1 + 1] = True

    out = g.copy()
    for component in _all_components(g, exclude_mask=exclude):
        assert _is_solid_rectangle(g, component), "all non-lookup components must be solid rectangles"
        edge_count = _touch_count(component["bbox"], h, w)
        if edge_count >= 1:
            br0, bc0, br1, bc1 = component["bbox"]
            out[br0 : br1 + 1, bc0 : bc1 + 1] = mapping[edge_count]

    return out


def _expanded_bbox(
    bbox: tuple[int, int, int, int], h: int, w: int, pad: int = 1
) -> tuple[int, int, int, int]:
    r0, c0, r1, c1 = bbox
    return max(0, r0 - pad), max(0, c0 - pad), min(h - 1, r1 + pad), min(w - 1, c1 + pad)


def _mark_blocked(blocked: np.ndarray, bbox: tuple[int, int, int, int], pad: int = 1) -> None:
    h, w = blocked.shape
    r0, c0, r1, c1 = _expanded_bbox(bbox, h, w, pad=pad)
    blocked[r0 : r1 + 1, c0 : c1 + 1] = True


def _sample_interval(
    length: int, touch_start: bool, touch_end: bool, max_free_dim: int, rng: random.Random
) -> tuple[int, int] | None:
    if touch_start and touch_end:
        return 0, length - 1

    if touch_start or touch_end:
        upper = min(max_free_dim, length - 1)
        if upper < 2:
            return None
        size = rng.randint(2, upper)
        return (0, size - 1) if touch_start else (length - size, length - 1)

    upper = min(max_free_dim, length - 2)
    if upper < 2:
        return None
    size = rng.randint(2, upper)
    start = rng.randint(1, length - size - 1)
    return start, start + size - 1


def _try_place_block(blocked: np.ndarray, touch_count: int, rng: random.Random) -> tuple[int, int, int, int] | None:
    h, w = blocked.shape
    touchsets = list(TOUCHSETS_BY_COUNT[touch_count])
    rng.shuffle(touchsets)

    for touchset in touchsets:
        for _ in range(80):
            row_interval = _sample_interval(
                h, "top" in touchset, "bottom" in touchset, MAX_FREE_DIM_BY_COUNT[touch_count], rng
            )
            col_interval = _sample_interval(
                w, "left" in touchset, "right" in touchset, MAX_FREE_DIM_BY_COUNT[touch_count], rng
            )
            if row_interval is None or col_interval is None:
                continue

            r0, r1 = row_interval
            c0, c1 = col_interval
            if blocked[r0 : r1 + 1, c0 : c1 + 1].any():
                continue

            return r0, c0, r1, c1

    return None


def _choose_lookup_layout(
    h: int, w: int, rng: random.Random
) -> tuple[tuple[int, int, int, int], dict[int, int], int, list[tuple[int, int]], list[int]]:
    left_pad = rng.randint(1, 2)
    right_pad = rng.randint(1, 2)
    top_pad = rng.randint(1, 2)
    bottom_pad = rng.randint(1, 2)
    row_gaps = [rng.randint(0, 1) for _ in range(3)]

    row_positions: list[int] = []
    row = 1 + top_pad
    for i in range(4):
        row_positions.append(row)
        if i < 3:
            row += 1 + row_gaps[i]

    outer_h = 1 + top_pad + 4 + sum(row_gaps) + bottom_pad + 1
    outer_w = 1 + left_pad + 5 + right_pad + 1

    corner = rng.choice(["top_left", "top_right", "bottom_left", "bottom_right"])
    row_offset = rng.randint(*LOOKUP_OFFSET_RANGE)
    col_offset = rng.randint(*LOOKUP_OFFSET_RANGE)

    if corner.startswith("top"):
        r0 = row_offset
    else:
        r0 = h - outer_h - row_offset
    if corner.endswith("left"):
        c0 = col_offset
    else:
        c0 = w - outer_w - col_offset

    bbox = (r0, c0, r0 + outer_h - 1, c0 + outer_w - 1)
    marker_start_col = c0 + 1 + left_pad
    abs_rows = [r0 + rel_row for rel_row in row_positions]

    lookup_colors = rng.sample(PALETTE, 4)
    mapping = {count: color for count, color in zip((1, 2, 3, 4), lookup_colors)}
    row_counts = rng.sample([1, 2, 3, 4], 4)

    row_specs = [(abs_rows[i], row_counts[i]) for i in range(4)]
    return bbox, mapping, marker_start_col, row_specs, [color for color in PALETTE if color not in lookup_colors]


def _draw_lookup(
    grid: np.ndarray,
    bbox: tuple[int, int, int, int],
    mapping: dict[int, int],
    marker_start_col: int,
    row_specs: list[tuple[int, int]],
) -> None:
    r0, c0, r1, c1 = bbox
    grid[r0, c0 : c1 + 1] = GREY
    grid[r1, c0 : c1 + 1] = GREY
    grid[r0 : r1 + 1, c0] = GREY
    grid[r0 : r1 + 1, c1] = GREY

    for row, count in row_specs:
        grid[row, marker_start_col : marker_start_col + count] = GREY
        grid[row, marker_start_col + count] = mapping[count]


def _render_rect(grid: np.ndarray, bbox: tuple[int, int, int, int], color: int) -> None:
    r0, c0, r1, c1 = bbox
    grid[r0 : r1 + 1, c0 : c1 + 1] = color


def _expected_output(grid: np.ndarray, mapping: dict[int, int], blocks: list[dict]) -> np.ndarray:
    out = np.asarray(grid, dtype=int).copy()
    h, w = out.shape
    for block in blocks:
        bbox = block["bbox"]
        edge_count = _touch_count(bbox, h, w)
        if edge_count >= 1:
            r0, c0, r1, c1 = bbox
            out[r0 : r1 + 1, c0 : c1 + 1] = mapping[edge_count]
    return out


def _gen_attempt(rng: random.Random) -> np.ndarray | None:
    h = rng.randint(*GRID_H_RANGE)
    w = rng.randint(*GRID_W_RANGE)
    grid = np.zeros((h, w), dtype=int)
    blocked = np.zeros((h, w), dtype=bool)

    lookup_bbox, mapping, marker_start_col, row_specs, source_palette = _choose_lookup_layout(h, w, rng)
    _draw_lookup(grid, lookup_bbox, mapping, marker_start_col, row_specs)
    _mark_blocked(blocked, lookup_bbox, pad=1)

    if len(source_palette) < 4:
        return None

    blocks: list[dict] = []
    source_colors = rng.sample(source_palette, 4)
    for touch_count, color in zip((3, 2, 1, 0), source_colors):
        bbox = _try_place_block(blocked, touch_count, rng)
        if bbox is None or _touch_count(bbox, h, w) != touch_count:
            return None
        _render_rect(grid, bbox, color)
        _mark_blocked(blocked, bbox, pad=1)
        blocks.append({"bbox": bbox, "color": color})

    for _ in range(rng.randint(0, MAX_EXTRA_BLOCKS)):
        touch_count = rng.choice([0, 1, 2, 3])
        bbox = _try_place_block(blocked, touch_count, rng)
        if bbox is None:
            break
        color = rng.choice(source_palette)
        _render_rect(grid, bbox, color)
        _mark_blocked(blocked, bbox, pad=1)
        blocks.append({"bbox": bbox, "color": color})

    expected = _expected_output(grid, mapping, blocks)
    try:
        actual = main(grid)
    except Exception:
        return None
    if not np.array_equal(actual, expected):
        return None

    return grid


def generate_input() -> np.ndarray:
    rng = random.Random()
    for _ in range(GEN_RETRIES):
        grid = _gen_attempt(rng)
        if grid is not None:
            return grid
    raise RuntimeError("failed to generate a valid input")
