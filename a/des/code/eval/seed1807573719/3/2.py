from __future__ import annotations

import random

import numpy as np


MAROON = 9
DIRS4 = ((1, 0), (-1, 0), (0, 1), (0, -1))


def _most_common_color(grid: np.ndarray) -> int:
    values, counts = np.unique(grid, return_counts=True)
    return int(values[int(np.argmax(counts))])


def _find_components(grid: np.ndarray, color: int) -> list[np.ndarray]:
    h, w = grid.shape
    mask = grid == int(color)
    seen = np.zeros((h, w), dtype=bool)
    comps: list[np.ndarray] = []

    for r, c in np.argwhere(mask):
        r = int(r)
        c = int(c)
        if seen[r, c]:
            continue
        stack = [(r, c)]
        seen[r, c] = True
        coords: list[tuple[int, int]] = []
        while stack:
            cr, cc = stack.pop()
            coords.append((int(cr), int(cc)))
            for dr, dc in DIRS4:
                nr = int(cr + dr)
                nc = int(cc + dc)
                if 0 <= nr < h and 0 <= nc < w and mask[nr, nc] and not seen[nr, nc]:
                    seen[nr, nc] = True
                    stack.append((nr, nc))
        comps.append(np.asarray(coords, dtype=int))

    return comps


def _normalize(coords: np.ndarray | set[tuple[int, int]]) -> set[tuple[int, int]]:
    if isinstance(coords, np.ndarray):
        rows = coords[:, 0]
        cols = coords[:, 1]
        r0 = int(rows.min())
        c0 = int(cols.min())
        return {(int(r - r0), int(c - c0)) for r, c in coords}

    rows = [r for r, _ in coords]
    cols = [c for _, c in coords]
    r0 = min(rows)
    c0 = min(cols)
    return {(int(r - r0), int(c - c0)) for r, c in coords}


def _bbox(cells: set[tuple[int, int]]) -> tuple[int, int, int, int]:
    rows = [r for r, _ in cells]
    cols = [c for _, c in cells]
    return min(rows), min(cols), max(rows), max(cols)


def _dims(cells: set[tuple[int, int]]) -> tuple[int, int]:
    r0, c0, r1, c1 = _bbox(cells)
    return int(r1 - r0 + 1), int(c1 - c0 + 1)


def _row_intervals(cells: set[tuple[int, int]], height: int) -> list[tuple[int, int] | None] | None:
    by_row: list[list[int]] = [[] for _ in range(height)]
    for r, c in cells:
        by_row[int(r)].append(int(c))

    intervals: list[tuple[int, int] | None] = []
    for cols in by_row:
        if not cols:
            intervals.append(None)
            continue
        lo = min(cols)
        hi = max(cols)
        if hi - lo + 1 != len(cols):
            return None
        intervals.append((int(lo), int(hi)))
    return intervals


def _col_intervals(cells: set[tuple[int, int]], width: int) -> list[tuple[int, int] | None] | None:
    by_col: list[list[int]] = [[] for _ in range(width)]
    for r, c in cells:
        by_col[int(c)].append(int(r))

    intervals: list[tuple[int, int] | None] = []
    for rows in by_col:
        if not rows:
            intervals.append(None)
            continue
        lo = min(rows)
        hi = max(rows)
        if hi - lo + 1 != len(rows):
            return None
        intervals.append((int(lo), int(hi)))
    return intervals


def _is_horizontal_partition(
    a: set[tuple[int, int]], b: set[tuple[int, int]], height: int, width: int
) -> bool:
    rows_a = _row_intervals(a, height)
    rows_b = _row_intervals(b, height)
    if rows_a is None or rows_b is None:
        return False

    full = (0, int(width - 1))
    for a_left in (True, False):
        shared_row = False
        ok = True
        for ia, ib in zip(rows_a, rows_b):
            if ia is None and ib is None:
                ok = False
                break
            if ia is None:
                if ib != full:
                    ok = False
                    break
                continue
            if ib is None:
                if ia != full:
                    ok = False
                    break
                continue

            shared_row = True
            if a_left:
                if ia[0] != 0 or ib[1] != width - 1 or ia[1] + 1 != ib[0]:
                    ok = False
                    break
            else:
                if ib[0] != 0 or ia[1] != width - 1 or ib[1] + 1 != ia[0]:
                    ok = False
                    break
        if ok and shared_row:
            return True

    return False


def _is_vertical_partition(
    a: set[tuple[int, int]], b: set[tuple[int, int]], height: int, width: int
) -> bool:
    cols_a = _col_intervals(a, width)
    cols_b = _col_intervals(b, width)
    if cols_a is None or cols_b is None:
        return False

    full = (0, int(height - 1))
    for a_top in (True, False):
        shared_col = False
        ok = True
        for ia, ib in zip(cols_a, cols_b):
            if ia is None and ib is None:
                ok = False
                break
            if ia is None:
                if ib != full:
                    ok = False
                    break
                continue
            if ib is None:
                if ia != full:
                    ok = False
                    break
                continue

            shared_col = True
            if a_top:
                if ia[0] != 0 or ib[1] != height - 1 or ia[1] + 1 != ib[0]:
                    ok = False
                    break
            else:
                if ib[0] != 0 or ia[1] != height - 1 or ib[1] + 1 != ia[0]:
                    ok = False
                    break
        if ok and shared_col:
            return True

    return False


def _rectangle_solutions(
    coords_a: np.ndarray | set[tuple[int, int]], coords_b: np.ndarray | set[tuple[int, int]]
) -> tuple[set[tuple[int, int]], set[tuple[int, int]]]:
    a = _normalize(coords_a)
    b = _normalize(coords_b)
    ha, wa = _dims(a)
    hb, wb = _dims(b)
    dr_limit = ha + hb
    dc_limit = wa + wb

    horizontal: set[tuple[int, int]] = set()
    vertical: set[tuple[int, int]] = set()

    for dr in range(-dr_limit, dr_limit + 1):
        for dc in range(-dc_limit, dc_limit + 1):
            shifted_b = {(int(r + dr), int(c + dc)) for r, c in b}
            if a & shifted_b:
                continue

            union = a | shifted_b
            r0, c0, r1, c1 = _bbox(union)
            height = int(r1 - r0 + 1)
            width = int(c1 - c0 + 1)
            if height * width != len(union):
                continue

            norm_a = {(int(r - r0), int(c - c0)) for r, c in a}
            norm_b = {(int(r - r0), int(c - c0)) for r, c in shifted_b}

            if _is_horizontal_partition(norm_a, norm_b, height, width):
                horizontal.add((height, width))
            if _is_vertical_partition(norm_a, norm_b, height, width):
                vertical.add((height, width))

    return horizontal, vertical


def _infer_pair(coords_a: np.ndarray, coords_b: np.ndarray) -> tuple[str, tuple[int, int]]:
    horizontal, vertical = _rectangle_solutions(coords_a, coords_b)

    if len(horizontal) == 1 and not vertical:
        return "horizontal", next(iter(horizontal))
    if len(vertical) == 1 and not horizontal:
        return "vertical", next(iter(vertical))

    raise AssertionError(
        f"pair must have exactly one orientation, got horizontal={horizontal}, vertical={vertical}"
    )


def _pack_width(items: list[dict]) -> int:
    if not items:
        return 0
    return sum(int(item["dims"][1]) for item in items) + len(items) - 1


def _build_output(records: list[dict], bg: int, baseline_color: int) -> np.ndarray:
    left = sorted(
        (item for item in records if item["orientation"] == "horizontal"),
        key=lambda item: (int(item["area"]), int(item["dims"][0]), int(item["dims"][1])),
    )
    right = sorted(
        (item for item in records if item["orientation"] == "vertical"),
        key=lambda item: (int(item["area"]), int(item["dims"][0]), int(item["dims"][1])),
    )

    max_rect_h = max(int(item["dims"][0]) for item in records)
    out_h = int(max_rect_h + 1)
    left_w = _pack_width(left)
    right_w = _pack_width(right)
    center_gap = 1 if left and right else 0
    out_w = int(left_w + center_gap + right_w)

    out = np.full((out_h, out_w), bg, dtype=int)
    out[-1, :] = int(baseline_color)

    col = 0
    for item in left:
        height, width = map(int, item["dims"])
        out[out_h - 1 - height : out_h - 1, col : col + width] = int(item["color"])
        col += width + 1

    col = left_w + center_gap
    for item in right:
        height, width = map(int, item["dims"])
        out[out_h - 1 - height : out_h - 1, col : col + width] = int(item["color"])
        col += width + 1

    return out


def main(grid: np.ndarray) -> np.ndarray:
    """
    Pair same-color fragments, infer whether each pair joins left-right or top-bottom,
    then repack the completed rectangles above a maroon baseline.
    """
    g = np.asarray(grid, dtype=int)
    assert g.ndim == 2 and g.shape[0] >= 2 and g.shape[1] >= 2
    assert bool(np.all(g[-1] == g[-1, 0])), "bottom row must be the baseline"

    baseline_color = int(g[-1, 0])
    bg = _most_common_color(g)
    assert baseline_color != bg

    records: list[dict] = []
    for color in map(int, np.unique(g)):
        if color in (bg, baseline_color):
            continue
        comps = _find_components(g[:-1], color)
        assert len(comps) == 2, f"expected exactly two fragments for color {color}"
        orientation, dims = _infer_pair(comps[0], comps[1])
        height, width = map(int, dims)
        records.append(
            {
                "color": int(color),
                "orientation": orientation,
                "dims": (height, width),
                "area": int(height * width),
            }
        )

    assert records
    return _build_output(records, bg, baseline_color)


RECT_H_RANGE = (2, 6)
RECT_W_RANGE = (2, 6)
PAIR_COUNT_CHOICES = (1, 2)
MAX_OUTPUT_WIDTH = 26
INPUT_H_RANGE = (16, 28)
INPUT_W_MAX = 30
INPUT_EXTRA_W_RANGE = (4, 10)
PLACEMENT_PADDING = 1
MAX_PAIR_ATTEMPTS = 120
MAX_LAYOUT_ATTEMPTS = 120
MAX_GENERATION_ATTEMPTS = 300


def _random_sequence(length: int, low: int, high: int) -> list[int]:
    seq = [random.randint(low, high) for _ in range(length)]
    if high > low and length >= 2 and len(set(seq)) == 1 and random.random() < 0.85:
        idx = random.randrange(length)
        alternatives = [v for v in range(low, high + 1) if v != seq[idx]]
        seq[idx] = random.choice(alternatives)
    return seq


def _make_horizontal_pair(height: int, width: int) -> tuple[set[tuple[int, int]], set[tuple[int, int]]]:
    split = _random_sequence(height, 1, width - 1)
    left = {(r, c) for r in range(height) for c in range(split[r])}
    right = {(r, c) for r in range(height) for c in range(split[r], width)}
    if random.random() < 0.5:
        return right, left
    return left, right


def _make_vertical_pair(height: int, width: int) -> tuple[set[tuple[int, int]], set[tuple[int, int]]]:
    split = _random_sequence(width, 1, height - 1)
    top = {(r, c) for c in range(width) for r in range(split[c])}
    bottom = {(r, c) for c in range(width) for r in range(split[c], height)}
    if random.random() < 0.5:
        return bottom, top
    return top, bottom


def _cells_to_array(cells: set[tuple[int, int]]) -> np.ndarray:
    arr = np.asarray(sorted(cells), dtype=int)
    return arr


def _sample_pair_spec(
    orientation: str,
    color: int,
    used_areas: set[int],
) -> dict | None:
    for _ in range(MAX_PAIR_ATTEMPTS):
        height = random.randint(*RECT_H_RANGE)
        width = random.randint(*RECT_W_RANGE)
        area = int(height * width)
        if area in used_areas:
            continue
        if height == 1 or width == 1:
            continue

        if orientation == "horizontal":
            piece_a, piece_b = _make_horizontal_pair(height, width)
        else:
            piece_a, piece_b = _make_vertical_pair(height, width)

        arr_a = _cells_to_array(piece_a)
        arr_b = _cells_to_array(piece_b)

        try:
            inferred_orientation, inferred_dims = _infer_pair(arr_a, arr_b)
        except AssertionError:
            continue

        if inferred_orientation != orientation or tuple(map(int, inferred_dims)) != (height, width):
            continue

        used_areas.add(area)
        return {
            "color": int(color),
            "orientation": orientation,
            "dims": (int(height), int(width)),
            "area": area,
            "pieces": (arr_a, arr_b),
        }

    return None


def _can_place(
    occupied: np.ndarray, cells: np.ndarray, top: int, left: int
) -> bool:
    h, w = occupied.shape
    for r, c in cells:
        rr = int(top + r)
        cc = int(left + c)
        if not (0 <= rr < h and 0 <= cc < w):
            return False
        r_lo = max(0, rr - PLACEMENT_PADDING)
        r_hi = min(h, rr + PLACEMENT_PADDING + 1)
        c_lo = max(0, cc - PLACEMENT_PADDING)
        c_hi = min(w, cc + PLACEMENT_PADDING + 1)
        if bool(occupied[r_lo:r_hi, c_lo:c_hi].any()):
            return False
    return True


def _stamp(grid: np.ndarray, occupied: np.ndarray, cells: np.ndarray, top: int, left: int, color: int) -> None:
    for r, c in cells:
        rr = int(top + r)
        cc = int(left + c)
        grid[rr, cc] = int(color)
        occupied[rr, cc] = True


def _render_input(specs: list[dict], bg: int, baseline_color: int) -> np.ndarray | None:
    fragments: list[tuple[int, np.ndarray]] = []
    for spec in specs:
        for piece in spec["pieces"]:
            fragments.append((int(spec["color"]), piece))
    random.shuffle(fragments)

    max_piece_h = max(int(piece[:, 0].max()) + 1 for _color, piece in fragments)
    min_width = max(
        18,
        _build_output(specs, bg, baseline_color).shape[1] + random.randint(*INPUT_EXTRA_W_RANGE),
    )
    max_width = INPUT_W_MAX
    if min_width > max_width:
        return None

    for _ in range(MAX_LAYOUT_ATTEMPTS):
        height = random.randint(max(16, max_piece_h + 4), INPUT_H_RANGE[1])
        width = random.randint(min_width, max_width)
        grid = np.full((height, width), bg, dtype=int)
        grid[-1, :] = int(baseline_color)

        occupied = np.zeros((height, width), dtype=bool)
        occupied[-1, :] = True
        ok = True

        for color, piece in fragments:
            piece_h = int(piece[:, 0].max()) + 1
            piece_w = int(piece[:, 1].max()) + 1
            positions: list[tuple[int, int]] = []
            for top in range(0, height - 1 - piece_h + 1):
                for left in range(0, width - piece_w + 1):
                    if _can_place(occupied, piece, top, left):
                        positions.append((int(top), int(left)))
            if not positions:
                ok = False
                break
            top, left = random.choice(positions)
            _stamp(grid, occupied, piece, top, left, color)

        if ok and _most_common_color(grid) == bg:
            return grid

    return None


def _gen_attempt() -> np.ndarray | None:
    bg = random.choice([c for c in range(9) if c != MAROON])
    baseline_color = MAROON
    n_horizontal = random.choice(PAIR_COUNT_CHOICES)
    n_vertical = random.choice(PAIR_COUNT_CHOICES)
    total_pairs = n_horizontal + n_vertical

    available_colors = [c for c in range(10) if c not in (bg, baseline_color)]
    if len(available_colors) < total_pairs:
        return None
    colors = random.sample(available_colors, total_pairs)

    specs: list[dict] = []
    used_horizontal_areas: set[int] = set()
    used_vertical_areas: set[int] = set()

    orientations = ["horizontal"] * n_horizontal + ["vertical"] * n_vertical
    random.shuffle(orientations)

    for color, orientation in zip(colors, orientations):
        used = used_horizontal_areas if orientation == "horizontal" else used_vertical_areas
        spec = _sample_pair_spec(orientation, color, used)
        if spec is None:
            return None
        specs.append(spec)

    expected = _build_output(specs, bg, baseline_color)
    if expected.shape[1] > MAX_OUTPUT_WIDTH:
        return None

    inp = _render_input(specs, bg, baseline_color)
    if inp is None:
        return None

    try:
        out = main(inp)
    except Exception:
        return None

    if not np.array_equal(out, expected):
        return None
    return inp


def generate_input() -> np.ndarray:
    for _ in range(MAX_GENERATION_ATTEMPTS):
        grid = _gen_attempt()
        if grid is not None:
            return grid
    raise RuntimeError("failed to generate a valid input")
