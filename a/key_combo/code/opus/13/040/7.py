import numpy as np


BG = 0
LINE = 1
MARKER = 2
RETRY_LIMIT = 500


def _bbox(mask: np.ndarray) -> tuple[int, int, int, int]:
    coords = np.argwhere(mask)
    if len(coords) == 0:
        raise AssertionError("expected non-empty mask")
    r0, c0 = coords.min(axis=0)
    r1, c1 = coords.max(axis=0)
    return int(r0), int(c0), int(r1), int(c1)


def _bar_union_mask(
    shape: tuple[int, int],
    box: tuple[int, int, int, int],
    h_rows: list[int],
    v_cols: list[int],
) -> np.ndarray:
    r0, c0, r1, c1 = box
    mask = np.zeros(shape, dtype=bool)
    for row in h_rows:
        mask[row, c0 : c1 + 1] = True
    for col in v_cols:
        mask[r0 : r1 + 1, col] = True
    return mask


def _extract_bar_structure(mask: np.ndarray) -> tuple[tuple[int, int, int, int], list[int], list[int]]:
    box = _bbox(mask)
    r0, c0, r1, c1 = box
    h_rows = [r for r in range(r0, r1 + 1) if np.all(mask[r, c0 : c1 + 1])]
    v_cols = [c for c in range(c0, c1 + 1) if np.all(mask[r0 : r1 + 1, c])]
    assert h_rows and v_cols, "expected full horizontal and vertical bars"
    rebuilt = _bar_union_mask(mask.shape, box, h_rows, v_cols)
    assert np.array_equal(mask, rebuilt), "line drawing must be a union of full bars"
    assert h_rows[0] == r0 and h_rows[-1] == r1, "outer horizontal bars missing"
    assert v_cols[0] == c0 and v_cols[-1] == c1, "outer vertical bars missing"
    assert len(h_rows) in (2, 3) and len(v_cols) in (2, 3), "unexpected bar count"
    return box, h_rows, v_cols


def _count_side_markers(grid: np.ndarray, box: tuple[int, int, int, int]) -> dict[str, int]:
    r0, c0, r1, c1 = box
    counts = {"top": 0, "right": 0, "bottom": 0, "left": 0}
    for r, c in np.argwhere(grid == MARKER):
        r = int(r)
        c = int(c)
        if r == r0 - 1 and c0 - 1 <= c <= c1 + 1:
            counts["top"] += 1
        elif r == r1 + 1 and c0 - 1 <= c <= c1 + 1:
            counts["bottom"] += 1
        elif c == c0 - 1 and r0 <= r <= r1:
            counts["left"] += 1
        elif c == c1 + 1 and r0 <= r <= r1:
            counts["right"] += 1
        else:
            raise AssertionError("marker must be exactly one cell outside one side")
    return counts


def _stretched_positions(
    box: tuple[int, int, int, int],
    h_rows: list[int],
    v_cols: list[int],
    counts: dict[str, int],
) -> tuple[tuple[int, int, int, int], list[int], list[int]]:
    r0, c0, r1, c1 = box
    new_box = (
        r0 - counts["top"],
        c0 - counts["left"],
        r1 + counts["bottom"],
        c1 + counts["right"],
    )
    new_r0, new_c0, new_r1, new_c1 = new_box
    new_h_rows = [
        new_r0 if row == r0 else new_r1 if row == r1 else row
        for row in h_rows
    ]
    new_v_cols = [
        new_c0 if col == c0 else new_c1 if col == c1 else col
        for col in v_cols
    ]
    assert len(new_h_rows) == len(set(new_h_rows)), "horizontal bars collapsed"
    assert len(new_v_cols) == len(set(new_v_cols)), "vertical bars collapsed"
    return new_box, new_h_rows, new_v_cols


def _connected4(mask: np.ndarray) -> bool:
    coords = np.argwhere(mask)
    if len(coords) == 0:
        return False
    todo = [tuple(map(int, coords[0]))]
    seen = {todo[0]}
    while todo:
        r, c = todo.pop()
        for nr, nc in ((r - 1, c), (r + 1, c), (r, c - 1), (r, c + 1)):
            if 0 <= nr < mask.shape[0] and 0 <= nc < mask.shape[1] and mask[nr, nc] and (nr, nc) not in seen:
                seen.add((nr, nc))
                todo.append((nr, nc))
    return len(seen) == int(mask.sum())


def main(grid: np.ndarray) -> np.ndarray:
    """
    Stretch the blue rectilinear line drawing by distances encoded as red marker counts on each side.

    Input structure:
    - The grid uses black background, one blue one-pixel-thick rectilinear line drawing, and red single-pixel control markers.
    - The blue object is a connected union of full horizontal and vertical bars inside one bounding box, with the outermost rows and columns occupied.
    - Every red marker lies exactly one cell outside that blue bounding box, on the adjacent ring around one side.

    Rule:
    - Count how many red markers belong to the top, right, bottom, and left sides of the blue bounding box.
    - Markers on the top or bottom adjacent rows count for those row-sides first, including corner positions; remaining markers on the left or right adjacent columns count for those column-sides.
    - Move the outer horizontal and vertical bars outward by those side counts, keep interior bar rows and columns fixed, redraw the stretched bar union, and erase all red markers.

    Key insight:
    - The red pixels are count tokens rather than destination markers: each side's marker count determines how far that side of the shared line drawing must expand.
    """
    g = np.asarray(grid, dtype=int)
    assert g.ndim == 2, "expected a 2D grid"
    assert np.all(np.isin(g, [BG, LINE, MARKER])), "unexpected colors"

    line_mask = g == LINE
    assert line_mask.any(), "expected a blue line drawing"
    box, h_rows, v_cols = _extract_bar_structure(line_mask)
    counts = _count_side_markers(g, box)

    new_box, new_h_rows, new_v_cols = _stretched_positions(box, h_rows, v_cols, counts)
    new_r0, new_c0, new_r1, new_c1 = new_box
    assert 0 <= new_r0 <= new_r1 < g.shape[0], "stretched rows leave the grid"
    assert 0 <= new_c0 <= new_c1 < g.shape[1], "stretched cols leave the grid"

    out = np.zeros_like(g)
    out[_bar_union_mask(g.shape, new_box, new_h_rows, new_v_cols)] = LINE
    return out


def _randint(lo: int, hi: int) -> int:
    return int(np.random.randint(lo, hi + 1))


def _sample_spread(values: list[int], count: int) -> list[int]:
    if count == 0:
        return []
    if count == 1:
        return [int(values[_randint(0, len(values) - 1)])]

    picks: list[int] = []
    used: set[int] = set()
    edges = np.linspace(0, len(values), count + 1, dtype=int)
    for i in range(count):
        lo = int(edges[i])
        hi = int(edges[i + 1])
        if hi <= lo:
            hi = lo + 1
        pool = [int(values[j]) for j in range(lo, min(hi, len(values))) if int(values[j]) not in used]
        if not pool:
            pool = [int(v) for v in values if int(v) not in used]
        pick = int(pool[_randint(0, len(pool) - 1)])
        picks.append(pick)
        used.add(pick)
    picks.sort()
    return picks


def _sample_side_positions(
    count: int,
    slots: list[int],
    *,
    allow_corner_bias: bool,
) -> list[int]:
    chosen = _sample_spread(slots, count)
    if allow_corner_bias and count > 0 and len(slots) >= 2 and np.random.rand() < 0.65:
        edge = int(slots[0] if np.random.rand() < 0.5 else slots[-1])
        if edge not in chosen:
            replace_i = _randint(0, len(chosen) - 1)
            chosen[replace_i] = edge
            chosen = sorted(set(chosen))
            while len(chosen) < count:
                remaining = [int(v) for v in slots if int(v) not in chosen]
                chosen.append(int(remaining[_randint(0, len(remaining) - 1)]))
                chosen.sort()
    return chosen


def _choose_inner_offset(size: int, side_a: int, side_b: int) -> int | None:
    blocked = {int(side_a), int(size - 1 - side_b)}
    candidates = [offset for offset in range(1, size - 1) if offset not in blocked]
    if not candidates:
        return None
    return int(candidates[_randint(0, len(candidates) - 1)])


def _render_case(
    shape: tuple[int, int],
    box: tuple[int, int, int, int],
    h_rows: list[int],
    v_cols: list[int],
    counts: dict[str, int],
    marker_cells: list[tuple[int, int]],
) -> tuple[np.ndarray, np.ndarray]:
    inp = np.zeros(shape, dtype=int)
    inp[_bar_union_mask(shape, box, h_rows, v_cols)] = LINE
    for r, c in marker_cells:
        assert inp[r, c] == BG, "marker overlaps line drawing"
        inp[r, c] = MARKER

    new_box, new_h_rows, new_v_cols = _stretched_positions(box, h_rows, v_cols, counts)
    out = np.zeros(shape, dtype=int)
    out[_bar_union_mask(shape, new_box, new_h_rows, new_v_cols)] = LINE
    return inp, out


def _semantic_checks(
    inp: np.ndarray,
    expected: np.ndarray,
    counts: dict[str, int],
    shape_kind: str,
) -> bool:
    try:
        observed = main(inp)
    except Exception:
        return False
    if not np.array_equal(observed, expected):
        return False
    if np.array_equal(inp, expected):
        return False

    in_box, in_h_rows, in_v_cols = _extract_bar_structure(inp == LINE)
    out_box, out_h_rows, out_v_cols = _extract_bar_structure(expected == LINE)
    got_counts = _count_side_markers(inp, in_box)
    if got_counts != counts:
        return False
    if sorted(got_counts.values()) != [0, 1, 2, 3]:
        return False

    dr = (in_box[0] - out_box[0], out_box[2] - in_box[2])
    dc = (in_box[1] - out_box[1], out_box[3] - in_box[3])
    if dr != (counts["top"], counts["bottom"]):
        return False
    if dc != (counts["left"], counts["right"]):
        return False
    if len(in_h_rows) != len(out_h_rows) or len(in_v_cols) != len(out_v_cols):
        return False
    if shape_kind == "frame" and (len(in_h_rows) != 2 or len(in_v_cols) != 2):
        return False
    if shape_kind == "cross" and (len(in_h_rows) != 3 or len(in_v_cols) != 3):
        return False
    if not _connected4(inp == LINE) or not _connected4(expected == LINE):
        return False
    return True


def _gen_attempt() -> np.ndarray | None:
    height = _randint(18, 24)
    width = _randint(18, 24)

    counts_perm = list(np.random.permutation([0, 1, 2, 3]))
    counts = {
        "top": int(counts_perm[0]),
        "right": int(counts_perm[1]),
        "bottom": int(counts_perm[2]),
        "left": int(counts_perm[3]),
    }

    old_h = _randint(5, 7)
    old_w = _randint(5, 7)
    new_h = old_h + counts["top"] + counts["bottom"]
    new_w = old_w + counts["left"] + counts["right"]
    if new_h > height - 2 or new_w > width - 2:
        return None

    new_r0 = _randint(1, height - new_h - 1)
    new_c0 = _randint(1, width - new_w - 1)
    old_r0 = new_r0 + counts["top"]
    old_c0 = new_c0 + counts["left"]
    old_box = (old_r0, old_c0, old_r0 + old_h - 1, old_c0 + old_w - 1)

    shape_kind = "cross" if np.random.rand() < 0.5 else "frame"
    h_rows = [old_box[0], old_box[2]]
    v_cols = [old_box[1], old_box[3]]
    if shape_kind == "cross":
        mid_r = _choose_inner_offset(old_h, counts["top"], counts["bottom"])
        mid_c = _choose_inner_offset(old_w, counts["left"], counts["right"])
        if mid_r is None or mid_c is None:
            return None
        h_rows.insert(1, old_box[0] + mid_r)
        v_cols.insert(1, old_box[1] + mid_c)

    top_slots = list(range(old_box[1] - 1, old_box[3] + 2))
    bottom_slots = list(range(old_box[1] - 1, old_box[3] + 2))
    left_slots = list(range(old_box[0], old_box[2] + 1))
    right_slots = list(range(old_box[0], old_box[2] + 1))

    marker_cells: list[tuple[int, int]] = []
    for c in _sample_side_positions(counts["top"], top_slots, allow_corner_bias=True):
        marker_cells.append((old_box[0] - 1, int(c)))
    for c in _sample_side_positions(counts["bottom"], bottom_slots, allow_corner_bias=True):
        marker_cells.append((old_box[2] + 1, int(c)))
    for r in _sample_side_positions(counts["left"], left_slots, allow_corner_bias=False):
        marker_cells.append((int(r), old_box[1] - 1))
    for r in _sample_side_positions(counts["right"], right_slots, allow_corner_bias=False):
        marker_cells.append((int(r), old_box[3] + 1))

    if len(marker_cells) != len(set(marker_cells)):
        return None

    inp, out = _render_case((height, width), old_box, h_rows, v_cols, counts, marker_cells)
    if not _semantic_checks(inp, out, counts, shape_kind):
        return None

    corner_markers = sum(
        1
        for r, c in marker_cells
        if r in (old_box[0] - 1, old_box[2] + 1) and c in (old_box[1] - 1, old_box[3] + 1)
    )
    if shape_kind == "cross" and np.random.rand() < 0.35 and corner_markers == 0:
        return None
    return inp


def generate_input() -> np.ndarray:
    for _ in range(RETRY_LIMIT):
        grid = _gen_attempt()
        if grid is not None:
            return grid
    raise RuntimeError("failed to generate a valid puzzle instance")
