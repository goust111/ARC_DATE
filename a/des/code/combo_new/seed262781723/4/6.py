import numpy as np


_DIRS4 = [(-1, 0), (1, 0), (0, -1), (0, 1)]
_DIRS8 = [
    (-1, -1), (-1, 0), (-1, 1),
    (0, -1),           (0, 1),
    (1, -1),  (1, 0),  (1, 1),
]


def _connected_components(mask: np.ndarray) -> list[np.ndarray]:
    h, w = mask.shape
    seen = np.zeros((h, w), dtype=bool)
    comps = []
    for r in range(h):
        for c in range(w):
            if not mask[r, c] or seen[r, c]:
                continue
            stack = [(r, c)]
            seen[r, c] = True
            cells = []
            while stack:
                cr, cc = stack.pop()
                cells.append((cr, cc))
                for dr, dc in _DIRS4:
                    nr, nc = cr + dr, cc + dc
                    if 0 <= nr < h and 0 <= nc < w and mask[nr, nc] and not seen[nr, nc]:
                        seen[nr, nc] = True
                        stack.append((nr, nc))
            comps.append(np.array(cells, dtype=int))
    return comps


def _bbox(coords: np.ndarray) -> tuple[int, int, int, int]:
    return (
        int(coords[:, 0].min()),
        int(coords[:, 1].min()),
        int(coords[:, 0].max()),
        int(coords[:, 1].max()),
    )


def _touches_border(coords: np.ndarray, h: int, w: int) -> bool:
    return bool(
        np.any(coords[:, 0] == 0)
        or np.any(coords[:, 0] == h - 1)
        or np.any(coords[:, 1] == 0)
        or np.any(coords[:, 1] == w - 1)
    )


def _normalize(coords: np.ndarray) -> np.ndarray:
    r0, c0, _, _ = _bbox(coords)
    norm = coords.copy()
    norm[:, 0] -= r0
    norm[:, 1] -= c0
    order = np.lexsort((norm[:, 1], norm[:, 0]))
    return norm[order]


def _shift(coords: np.ndarray, dr: int, dc: int) -> np.ndarray:
    moved = coords.copy()
    moved[:, 0] += dr
    moved[:, 1] += dc
    return moved


def _collect_pairs(grid: np.ndarray) -> list[tuple[int, np.ndarray, np.ndarray]]:
    g = np.asarray(grid, dtype=int)
    h, w = g.shape
    pairs = []
    for color in np.unique(g):
        color = int(color)
        if color == 0:
            continue
        comps = _connected_components(g == color)
        border = [comp for comp in comps if _touches_border(comp, h, w)]
        inner = [comp for comp in comps if not _touches_border(comp, h, w)]
        assert len(border) == 1, f"color {color} must have exactly one border segment"
        assert len(inner) == 1, f"color {color} must have exactly one interior object"
        pairs.append((color, border[0], inner[0]))
    return pairs


def main(grid: np.ndarray) -> np.ndarray:
    """Project each interior object onto its same-color edge track by sliding along the axis perpendicular to its long side.

    Input structure:
    - The background is black.
    - Several colors each form one 1-pixel-thick horizontal or vertical line segment on the outer border.
    - For every such color, exactly one same-color connected object of 2-5 pixels sits strictly inside the grid.

    Rule:
    - Pair every interior object with the unique border segment of the same color.
    - If the object's bounding-box width is at least its height, treat it as horizontal and translate it vertically until the side facing the border lies on the segment's row.
    - Otherwise treat it as vertical and translate it horizontally until the side facing the border lies on the segment's column.
    - Erase the original object position, keep the border segment, and draw the translated object on top of any overlapped segment cells.

    Key insight:
    - The matching cue is color, but the movement axis comes from the object's own long-axis orientation rather than from a generic nearest-edge rule.
    """
    g = np.asarray(grid, dtype=int)
    h, w = g.shape
    out = g.copy()

    for color, segment, obj in _collect_pairs(g):
        out[obj[:, 0], obj[:, 1]] = 0

        r0, c0, r1, c1 = _bbox(obj)
        obj_h = r1 - r0 + 1
        obj_w = c1 - c0 + 1
        sr0, sc0, sr1, sc1 = _bbox(segment)

        if obj_w >= obj_h:
            assert sr0 == sr1, f"horizontal object color {color} requires a horizontal segment"
            if sr0 == 0:
                dr = -r0
            else:
                assert sr0 == h - 1
                dr = (h - 1) - r1
            moved = _shift(obj, dr, 0)
        else:
            assert sc0 == sc1, f"vertical object color {color} requires a vertical segment"
            if sc0 == 0:
                dc = -c0
            else:
                assert sc0 == w - 1
                dc = (w - 1) - c1
            moved = _shift(obj, 0, dc)

        assert np.all((0 <= moved[:, 0]) & (moved[:, 0] < h))
        assert np.all((0 <= moved[:, 1]) & (moved[:, 1] < w))
        out[moved[:, 0], moved[:, 1]] = color

    return out


PALETTE = np.array([1, 2, 3, 4, 5, 6, 7, 8, 9], dtype=int)
GRID_HEIGHT_RANGE = (14, 27)
GRID_WIDTH_RANGE = (14, 27)
OBJECT_COUNT_RANGE = (2, 5)
MAX_GENERATION_ATTEMPTS = 400
MAX_PLACEMENT_ATTEMPTS = 120
MAX_SHAPE_ATTEMPTS = 120
SEGMENT_PAD_MAX = 2
INNER_MARGIN = 2


def _to_set(coords: np.ndarray) -> set[tuple[int, int]]:
    return {(int(r), int(c)) for r, c in coords}


def _inflate(cells: set[tuple[int, int]], h: int, w: int) -> set[tuple[int, int]]:
    expanded = set()
    for r, c in cells:
        for dr, dc in _DIRS8:
            nr, nc = r + dr, c + dc
            if 0 <= nr < h and 0 <= nc < w:
                expanded.add((nr, nc))
        expanded.add((r, c))
    return expanded


def _random_shape(rng: np.random.Generator, horizontal: bool) -> np.ndarray:
    for _ in range(MAX_SHAPE_ATTEMPTS):
        if horizontal:
            box_h = int(rng.integers(1, 4))
            box_w = int(rng.integers(2, 5))
        else:
            box_h = int(rng.integers(2, 5))
            box_w = int(rng.integers(1, 4))
        area = int(rng.integers(2, min(5, box_h * box_w) + 1))

        start = (int(rng.integers(box_h)), int(rng.integers(box_w)))
        cells = {start}
        while len(cells) < area:
            frontier = []
            for r, c in cells:
                for dr, dc in _DIRS4:
                    nr, nc = r + dr, c + dc
                    if 0 <= nr < box_h and 0 <= nc < box_w and (nr, nc) not in cells:
                        frontier.append((nr, nc))
            if not frontier:
                break
            pick = frontier[int(rng.integers(len(frontier)))]
            cells.add(pick)

        coords = np.array(sorted(cells), dtype=int)
        if coords.shape[0] != area:
            continue
        coords = _normalize(coords)
        r0, c0, r1, c1 = _bbox(coords)
        height = r1 - r0 + 1
        width = c1 - c0 + 1
        if horizontal and width >= height:
            return coords
        if not horizontal and height > width:
            return coords
    raise RuntimeError("failed to sample a connected shape with the requested orientation")


def _build_expected(
    shape: np.ndarray,
    color: int,
    side: int,
    r0: int,
    c0: int,
    h: int,
    w: int,
    pad_before: int,
    pad_after: int,
) -> tuple[set[tuple[int, int]], set[tuple[int, int]], set[tuple[int, int]]]:
    obj = _shift(shape, r0, c0)
    br0, bc0, br1, bc1 = _bbox(obj)

    if side in (0, 1):
        seg_row = 0 if side == 0 else h - 1
        seg_start = bc0 - pad_before
        seg_end = bc1 + pad_after
        segment = {(seg_row, cc) for cc in range(seg_start, seg_end + 1)}
        dr = -br0 if side == 0 else (h - 1) - br1
        final_obj = _to_set(_shift(obj, dr, 0))
    else:
        seg_col = 0 if side == 2 else w - 1
        seg_start = br0 - pad_before
        seg_end = br1 + pad_after
        segment = {(rr, seg_col) for rr in range(seg_start, seg_end + 1)}
        dc = -bc0 if side == 2 else (w - 1) - bc1
        final_obj = _to_set(_shift(obj, 0, dc))

    return _to_set(obj), segment, final_obj


def _orientation_schedule(rng: np.random.Generator, count: int) -> list[bool]:
    if count == 2:
        flags = [True, False]
    elif count == 3:
        flags = [True, True, False] if int(rng.integers(2)) == 0 else [True, False, False]
    else:
        flags = [True, True, False, False]
    rng.shuffle(flags)
    return flags


def _gen_attempt(rng: np.random.Generator) -> np.ndarray | None:
    h = int(rng.integers(*GRID_HEIGHT_RANGE))
    w = int(rng.integers(*GRID_WIDTH_RANGE))
    count = int(rng.integers(*OBJECT_COUNT_RANGE))
    colors = [int(c) for c in rng.choice(PALETTE, size=count, replace=False)]
    orientations = _orientation_schedule(rng, count)

    placements = []
    input_exact: set[tuple[int, int]] = set()
    input_object_buffer: set[tuple[int, int]] = set()
    output_exact: set[tuple[int, int]] = set()
    output_object_buffer: set[tuple[int, int]] = set()

    for color, horizontal in zip(colors, orientations):
        placed = False
        for _ in range(MAX_PLACEMENT_ATTEMPTS):
            shape = _random_shape(rng, horizontal)
            sr0, sc0, sr1, sc1 = _bbox(shape)
            sh = sr1 - sr0 + 1
            sw = sc1 - sc0 + 1

            if horizontal:
                side = int(rng.integers(2))  # 0 top, 1 bottom
                min_r = INNER_MARGIN
                max_r = h - sh - INNER_MARGIN - 1
                min_c = INNER_MARGIN
                max_c = w - sw - INNER_MARGIN
            else:
                side = 2 + int(rng.integers(2))  # 2 left, 3 right
                min_r = INNER_MARGIN
                max_r = h - sh - INNER_MARGIN
                min_c = INNER_MARGIN
                max_c = w - sw - INNER_MARGIN - 1

            if max_r < min_r or max_c < min_c:
                return None

            r0 = int(rng.integers(min_r, max_r + 1))
            c0 = int(rng.integers(min_c, max_c + 1))

            pad_before = int(rng.integers(0, SEGMENT_PAD_MAX + 1))
            pad_after = int(rng.integers(0, SEGMENT_PAD_MAX + 1))

            obj_cells, segment_cells, final_cells = _build_expected(
                shape, color, side, r0, c0, h, w, pad_before, pad_after
            )

            if any(r in (0, h - 1) or c in (0, w - 1) for r, c in obj_cells):
                continue
            if any(not (0 <= r < h and 0 <= c < w) for r, c in segment_cells | final_cells):
                continue

            if obj_cells & input_exact:
                continue
            if obj_cells & input_object_buffer:
                continue
            if segment_cells & input_exact:
                continue

            candidate_output = segment_cells | final_cells
            if candidate_output & output_exact:
                continue
            if final_cells & output_object_buffer:
                continue

            placements.append(
                {
                    "color": color,
                    "object": obj_cells,
                    "segment": segment_cells,
                    "final": final_cells,
                }
            )
            input_exact |= obj_cells | segment_cells
            input_object_buffer |= _inflate(obj_cells, h, w)
            output_exact |= candidate_output
            output_object_buffer |= _inflate(final_cells, h, w)
            placed = True
            break
        if not placed:
            return None

    grid = np.zeros((h, w), dtype=int)
    expected = np.zeros((h, w), dtype=int)

    for placement in placements:
        color = placement["color"]
        for r, c in placement["segment"]:
            grid[r, c] = color
            expected[r, c] = color
        for r, c in placement["object"]:
            grid[r, c] = color
    for placement in placements:
        color = placement["color"]
        for r, c in placement["final"]:
            expected[r, c] = color

    try:
        pairs = _collect_pairs(grid)
    except AssertionError:
        return None
    if len(pairs) != count:
        return None
    if not np.array_equal(main(grid), expected):
        return None
    return grid


def generate_input() -> np.ndarray:
    rng = np.random.default_rng()
    for _ in range(MAX_GENERATION_ATTEMPTS):
        grid = _gen_attempt(rng)
        if grid is not None:
            return grid
    raise RuntimeError("failed to generate a valid sample for puzzle 6")
