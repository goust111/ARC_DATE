from __future__ import annotations

import random

import numpy as np


DIRS4 = {
    "up": (-1, 0),
    "down": (1, 0),
    "left": (0, -1),
    "right": (0, 1),
}
SIDE_OPPOSITE = {"up": "down", "down": "up", "left": "right", "right": "left"}
DIR_ORDER = {"left": 0, "up": 1, "right": 2, "down": 3}

CHECKER_EVEN = 0
CHECKER_ODD = 5
FOREGROUND_COLORS = (1, 2, 3, 4, 6, 7, 8, 9)

GEN_ROWS = (20, 30)
GEN_COLS = (20, 30)
GEN_PAIR_COUNTS = (1, 2)
GEN_SINGLE_COUNTS = (0, 1)
GEN_GAP_MIN = 2
GEN_GAP_MAX = 6
GEN_RETRIES = 400


def _mask(rows: tuple[str, ...]) -> np.ndarray:
    return np.array([[ch == "1" for ch in row] for row in rows], dtype=bool)


ANCHOR_LEFT_TEMPLATES = (
    _mask(("110", "111", "111", "111", "111", "110")),
    _mask(("110", "111", "111", "111", "111", "111", "110")),
    _mask(("1100", "1111", "1111", "1111", "1111", "1100")),
    _mask(("1100", "1111", "1111", "1111", "1111", "1111", "1100")),
)
SEEKER_LEFT_TEMPLATES = (
    _mask(("10", "11", "10")),
    _mask(("100", "111", "100")),
    _mask(("110", "100", "110")),
)


def _checkerboard(shape: tuple[int, int], even_color: int, odd_color: int) -> np.ndarray:
    rows, cols = np.indices(shape)
    return np.where((rows + cols) % 2 == 0, int(even_color), int(odd_color))


def _mode(values: np.ndarray) -> int:
    colors, counts = np.unique(values, return_counts=True)
    return int(colors[np.argmax(counts)])


def _infer_checker_colors(grid: np.ndarray) -> tuple[int, int]:
    rows, cols = np.indices(grid.shape)
    parity = (rows + cols) % 2
    even_color = _mode(grid[parity == 0])
    odd_color = _mode(grid[parity == 1])
    assert even_color != odd_color, "expected a two-color checkerboard background"
    return even_color, odd_color


def _background_mask(grid: np.ndarray, even_color: int, odd_color: int) -> np.ndarray:
    rows, cols = np.indices(grid.shape)
    expected = np.where((rows + cols) % 2 == 0, int(even_color), int(odd_color))
    return grid == expected


def _find_components(grid: np.ndarray, bg_mask: np.ndarray) -> list[dict]:
    h, w = grid.shape
    seen = np.zeros((h, w), dtype=bool)
    comps: list[dict] = []

    for r in range(h):
        for c in range(w):
            if bg_mask[r, c] or seen[r, c]:
                continue
            color = int(grid[r, c])
            stack = [(r, c)]
            seen[r, c] = True
            coords: list[tuple[int, int]] = []
            while stack:
                rr, cc = stack.pop()
                coords.append((rr, cc))
                for dr, dc in DIRS4.values():
                    nr, nc = rr + dr, cc + dc
                    if not (0 <= nr < h and 0 <= nc < w):
                        continue
                    if seen[nr, nc] or bg_mask[nr, nc] or int(grid[nr, nc]) != color:
                        continue
                    seen[nr, nc] = True
                    stack.append((nr, nc))

            arr = np.array(coords, dtype=int)
            r0, c0 = arr.min(axis=0)
            r1, c1 = arr.max(axis=0)
            comps.append(
                {
                    "color": color,
                    "coords": arr,
                    "area": int(arr.shape[0]),
                    "bbox": (int(r0), int(c0), int(r1), int(c1)),
                    "direction": _heavy_direction(arr),
                }
            )
    return comps


def _heavy_direction(coords: np.ndarray) -> str:
    rows = coords[:, 0].astype(float)
    cols = coords[:, 1].astype(float)
    r0, c0 = coords.min(axis=0)
    r1, c1 = coords.max(axis=0)
    row_offset = float(rows.mean() - (r0 + r1) / 2.0)
    col_offset = float(cols.mean() - (c0 + c1) / 2.0)

    if abs(col_offset) > abs(row_offset):
        assert abs(col_offset) > 1e-9, "object must have a horizontal or vertical bias"
        return "right" if col_offset > 0 else "left"
    if abs(row_offset) > abs(col_offset):
        assert abs(row_offset) > 1e-9, "object must have a horizontal or vertical bias"
        return "down" if row_offset > 0 else "up"
    if abs(col_offset) > 1e-9:
        return "right" if col_offset > 0 else "left"
    assert abs(row_offset) > 1e-9, "object must have a horizontal or vertical bias"
    return "down" if row_offset > 0 else "up"


def _distance_to_edge(coords: np.ndarray, direction: str, shape: tuple[int, int]) -> int:
    h, w = shape
    if direction == "left":
        return int(coords[:, 1].min())
    if direction == "right":
        return int(w - 1 - coords[:, 1].max())
    if direction == "up":
        return int(coords[:, 0].min())
    return int(h - 1 - coords[:, 0].max())


def _settle_order(obj: dict, shape: tuple[int, int]) -> tuple[int, int, int, int]:
    coords = obj["coords"]
    rows = coords[:, 0]
    cols = coords[:, 1]
    return (
        _distance_to_edge(coords, obj["direction"], shape),
        DIR_ORDER[obj["direction"]],
        int(rows.min()),
        int(cols.min()),
    )


def _has_same_color_neighbor(coords: np.ndarray, color_grid: np.ndarray, color: int) -> bool:
    h, w = color_grid.shape
    for r, c in coords:
        for dr, dc in DIRS4.values():
            nr, nc = int(r) + dr, int(c) + dc
            if 0 <= nr < h and 0 <= nc < w and int(color_grid[nr, nc]) == color:
                return True
    return False


def _settle_objects(objects: list[dict], shape: tuple[int, int]) -> list[np.ndarray]:
    h, w = shape
    occ = np.full((h, w), -1, dtype=int)
    colors = np.full((h, w), -1, dtype=int)
    positions = [obj["coords"].copy() for obj in objects]

    for idx, obj in enumerate(objects):
        rr = obj["coords"][:, 0]
        cc = obj["coords"][:, 1]
        occ[rr, cc] = idx
        colors[rr, cc] = int(obj["color"])

    order = sorted(range(len(objects)), key=lambda idx: _settle_order(objects[idx], shape))
    for idx in order:
        color = int(objects[idx]["color"])
        coords = positions[idx]
        occ[coords[:, 0], coords[:, 1]] = -1
        colors[coords[:, 0], coords[:, 1]] = -1

        if _has_same_color_neighbor(coords, colors, color):
            placed = coords
        else:
            dr, dc = DIRS4[objects[idx]["direction"]]
            placed = coords
            while True:
                candidate = placed + np.array([dr, dc], dtype=int)
                if (
                    int(candidate[:, 0].min()) < 0
                    or int(candidate[:, 0].max()) >= h
                    or int(candidate[:, 1].min()) < 0
                    or int(candidate[:, 1].max()) >= w
                ):
                    break
                if np.any(occ[candidate[:, 0], candidate[:, 1]] != -1):
                    break
                placed = candidate
                if _has_same_color_neighbor(placed, colors, color):
                    break

        positions[idx] = placed
        occ[placed[:, 0], placed[:, 1]] = idx
        colors[placed[:, 0], placed[:, 1]] = color

    return positions


def _contact_info(large_coords: np.ndarray, small_coords: np.ndarray) -> tuple[str, list[int]] | None:
    small = {tuple(map(int, cell)) for cell in small_coords.tolist()}
    contacts: dict[str, list[int]] = {"left": [], "right": [], "up": [], "down": []}

    for r, c in large_coords:
        rr = int(r)
        cc = int(c)
        if (rr, cc - 1) in small:
            contacts["left"].append(rr)
        if (rr, cc + 1) in small:
            contacts["right"].append(rr)
        if (rr - 1, cc) in small:
            contacts["up"].append(cc)
        if (rr + 1, cc) in small:
            contacts["down"].append(cc)

    sides = [(side, values) for side, values in contacts.items() if values]
    if len(sides) != 1:
        return None
    side, values = sides[0]
    return side, values


def _contiguous_runs(values: list[int]) -> list[tuple[int, int]]:
    if not values:
        return []
    runs: list[tuple[int, int]] = []
    start = prev = values[0]
    for value in values[1:]:
        if value == prev + 1:
            prev = value
            continue
        runs.append((start, prev))
        start = prev = value
    runs.append((start, prev))
    return runs


def _extension_cells(large_coords: np.ndarray, small_coords: np.ndarray, side: str, count: int) -> np.ndarray:
    large = {tuple(map(int, cell)) for cell in large_coords.tolist()}
    info = _contact_info(large_coords, small_coords)
    assert info is not None and info[0] == side
    _, contacts = info
    r0, c0 = large_coords.min(axis=0)
    r1, c1 = large_coords.max(axis=0)

    if side in {"left", "right"}:
        edge_col = int(c0 if side == "left" else c1)
        ext_col = edge_col - 1 if side == "left" else edge_col + 1
        support = sorted({r for r, c in large if c == edge_col})
    else:
        edge_row = int(r0 if side == "up" else r1)
        ext_row = edge_row - 1 if side == "up" else edge_row + 1
        support = sorted({c for r, c in large if r == edge_row})

    runs = [run for run in _contiguous_runs(support) if run[1] - run[0] + 1 >= count]
    assert runs, "contact side must have enough support for the absorbed area"
    target_center = float(sum(contacts)) / float(len(contacts))
    start_run, end_run = min(
        runs,
        key=lambda run: (abs((run[0] + run[1]) / 2.0 - target_center), run[0], run[1]),
    )
    start = int(round(target_center - (count - 1) / 2.0))
    start = max(start_run, min(start, end_run - count + 1))

    if side in {"left", "right"}:
        return np.array([(r, ext_col) for r in range(start, start + count)], dtype=int)
    return np.array([(ext_row, c) for c in range(start, start + count)], dtype=int)


def main(grid: np.ndarray) -> np.ndarray:
    """Slide each asymmetric colored object in its heavier direction, then absorb smaller same-color neighbors into the larger edge.

    Input structure:
    - The background is a two-color checkerboard.
    - Several non-background monochrome 4-connected objects sit on top of that checkerboard.
    - Each object is visually unbalanced toward one cardinal direction inside its own bounding box.

    Rule:
    - For every object, compare its pixel centroid with the center of its tight bounding box to infer a heavier direction: left/right from the column offset, up/down from the row offset, using the larger-magnitude axis when both exist.
    - Slide the whole object cell-by-cell in that direction until the next step would leave the grid or overlap another object; if it reaches 4-neighbor contact with a same-color object, stop at that touching position.
    - Whenever two same-color objects are touching, erase the smaller one and convert its area into a same-color added row or column attached just outside the larger object's contact side. The checkerboard background elsewhere stays unchanged.

    Key insight:
    - The checkerboard is a distractor; the real cue is the object's internal mass imbalance, which determines motion, and the motion then exposes which same-color pair should merge.
    """
    g = np.asarray(grid, dtype=int)
    assert g.ndim == 2, "expected a 2D grid"

    even_color, odd_color = _infer_checker_colors(g)
    bg = _checkerboard(g.shape, even_color, odd_color)
    objects = _find_components(g, _background_mask(g, even_color, odd_color))
    assert objects, "expected at least one foreground object"

    settled = _settle_objects(objects, g.shape)
    out = bg.copy()
    for obj, coords in zip(objects, settled, strict=True):
        out[coords[:, 0], coords[:, 1]] = int(obj["color"])

    groups: dict[int, list[int]] = {}
    for idx, obj in enumerate(objects):
        groups.setdefault(int(obj["color"]), []).append(idx)

    removed: set[int] = set()
    for ids in groups.values():
        for i in range(len(ids)):
            for j in range(i + 1, len(ids)):
                a = ids[i]
                b = ids[j]
                if a in removed or b in removed:
                    continue
                area_a = int(objects[a]["area"])
                area_b = int(objects[b]["area"])
                if area_a == area_b:
                    continue

                if area_a > area_b:
                    large, small = a, b
                else:
                    large, small = b, a

                info = _contact_info(settled[large], settled[small])
                if info is None:
                    continue
                side, _ = info
                small_coords = settled[small]
                out[small_coords[:, 0], small_coords[:, 1]] = bg[small_coords[:, 0], small_coords[:, 1]]
                ext = _extension_cells(settled[large], settled[small], side, int(objects[small]["area"]))
                out[ext[:, 0], ext[:, 1]] = int(objects[large]["color"])
                removed.add(small)
    return out


def _orient_left_mask(mask: np.ndarray, direction: str) -> np.ndarray:
    if direction == "left":
        return mask.copy()
    if direction == "right":
        return np.fliplr(mask).copy()
    if direction == "up":
        return mask.T.copy()
    assert direction == "down"
    return np.flipud(mask.T).copy()


def _mask_coords(mask: np.ndarray, r0: int, c0: int) -> np.ndarray:
    coords = np.argwhere(mask)
    return coords + np.array([int(r0), int(c0)], dtype=int)


def _coords_set(coords: np.ndarray) -> set[tuple[int, int]]:
    return {tuple(map(int, cell)) for cell in coords.tolist()}


def _paint_coords(grid: np.ndarray, coords: np.ndarray, color: int) -> None:
    grid[coords[:, 0], coords[:, 1]] = int(color)


def _leading_indices(mask: np.ndarray, direction: str) -> list[int]:
    coords = np.argwhere(mask)
    if direction == "left":
        edge = int(coords[:, 1].min())
        return sorted({int(r) for r, c in coords if int(c) == edge})
    if direction == "right":
        edge = int(coords[:, 1].max())
        return sorted({int(r) for r, c in coords if int(c) == edge})
    if direction == "up":
        edge = int(coords[:, 0].min())
        return sorted({int(c) for r, c in coords if int(r) == edge})
    edge = int(coords[:, 0].max())
    return sorted({int(c) for r, c in coords if int(r) == edge})


def _support_indices(mask: np.ndarray, side: str) -> list[int]:
    coords = np.argwhere(mask)
    if side == "left":
        edge = int(coords[:, 1].min())
        return sorted({int(r) for r, c in coords if int(c) == edge})
    if side == "right":
        edge = int(coords[:, 1].max())
        return sorted({int(r) for r, c in coords if int(c) == edge})
    if side == "up":
        edge = int(coords[:, 0].min())
        return sorted({int(c) for r, c in coords if int(r) == edge})
    edge = int(coords[:, 0].max())
    return sorted({int(c) for r, c in coords if int(r) == edge})


def _max_run_length(values: list[int]) -> int:
    runs = _contiguous_runs(values)
    return max((end - start + 1 for start, end in runs), default=0)


def _cells_union(*coord_arrays: np.ndarray) -> set[tuple[int, int]]:
    cells: set[tuple[int, int]] = set()
    for coords in coord_arrays:
        cells |= _coords_set(coords)
    return cells


def _path_cells(mask: np.ndarray, start_r: int, start_c: int, direction: str, steps: int) -> set[tuple[int, int]]:
    dr, dc = DIRS4[direction]
    cells: set[tuple[int, int]] = set()
    for step in range(steps + 1):
        coords = _mask_coords(mask, start_r + step * dr, start_c + step * dc)
        cells |= _coords_set(coords)
    return cells


def _expand_cells(cells: set[tuple[int, int]], shape: tuple[int, int]) -> set[tuple[int, int]]:
    h, w = shape
    expanded: set[tuple[int, int]] = set()
    for r, c in cells:
        for dr in (-1, 0, 1):
            for dc in (-1, 0, 1):
                nr, nc = r + dr, c + dc
                if 0 <= nr < h and 0 <= nc < w:
                    expanded.add((nr, nc))
    return expanded


def _heavy_direction_matches(mask: np.ndarray, direction: str) -> bool:
    return _heavy_direction(np.argwhere(mask)) == direction


def _place_pair(
    h: int,
    w: int,
    reserved: set[tuple[int, int]],
    color: int,
    direction: str,
) -> tuple[list[tuple[int, np.ndarray]], list[tuple[int, np.ndarray]], set[tuple[int, int]]] | None:
    seeker_base = random.choice(SEEKER_LEFT_TEMPLATES)
    seeker_mask = _orient_left_mask(seeker_base, direction)
    seeker_area = int(seeker_mask.sum())
    anchor_options = []
    for base in ANCHOR_LEFT_TEMPLATES:
        anchor_mask = _orient_left_mask(base, direction)
        if _max_run_length(_support_indices(anchor_mask, SIDE_OPPOSITE[direction])) >= seeker_area:
            anchor_options.append(anchor_mask)
    if not anchor_options:
        return None
    anchor_mask = random.choice(anchor_options)

    assert _heavy_direction_matches(anchor_mask, direction)
    assert _heavy_direction_matches(seeker_mask, direction)

    ah, aw = anchor_mask.shape
    sh, sw = seeker_mask.shape
    support = _support_indices(anchor_mask, SIDE_OPPOSITE[direction])
    leading = _leading_indices(seeker_mask, direction)

    if direction in {"left", "right"}:
        min_anchor_row = max(0, -support[0] + leading[0])
        max_anchor_row = min(h - ah, h - 1 - support[-1] - (sh - 1 - leading[-1]))
        if min_anchor_row > max_anchor_row:
            return None
        anchor_row = random.randint(min_anchor_row, max_anchor_row)
        anchor_col = 0 if direction == "left" else w - aw
        if direction == "left":
            seeker_final_row = anchor_row + support[0] - leading[0]
            seeker_final_col = anchor_col + aw
            max_gap = w - (seeker_final_col + sw)
        else:
            seeker_final_row = anchor_row + support[0] - leading[0]
            seeker_final_col = anchor_col - sw
            max_gap = seeker_final_col
        if max_gap < GEN_GAP_MIN:
            return None
        gap = random.randint(GEN_GAP_MIN, min(GEN_GAP_MAX, max_gap))
    else:
        min_anchor_col = max(0, -support[0] + leading[0])
        max_anchor_col = min(w - aw, w - 1 - support[-1] - (sw - 1 - leading[-1]))
        if min_anchor_col > max_anchor_col:
            return None
        anchor_col = random.randint(min_anchor_col, max_anchor_col)
        anchor_row = 0 if direction == "up" else h - ah
        if direction == "up":
            seeker_final_row = anchor_row + ah
            seeker_final_col = anchor_col + support[0] - leading[0]
            max_gap = h - (seeker_final_row + sh)
        else:
            seeker_final_row = anchor_row - sh
            seeker_final_col = anchor_col + support[0] - leading[0]
            max_gap = seeker_final_row
        if max_gap < GEN_GAP_MIN:
            return None
        gap = random.randint(GEN_GAP_MIN, min(GEN_GAP_MAX, max_gap))

    dr, dc = DIRS4[direction]
    seeker_start_row = seeker_final_row - gap * dr
    seeker_start_col = seeker_final_col - gap * dc

    anchor_coords = _mask_coords(anchor_mask, anchor_row, anchor_col)
    seeker_start = _mask_coords(seeker_mask, seeker_start_row, seeker_start_col)
    seeker_final = _mask_coords(seeker_mask, seeker_final_row, seeker_final_col)
    if (
        int(seeker_start[:, 0].min()) < 0
        or int(seeker_start[:, 0].max()) >= h
        or int(seeker_start[:, 1].min()) < 0
        or int(seeker_start[:, 1].max()) >= w
    ):
        return None

    contact = _contact_info(anchor_coords, seeker_final)
    if contact is None or contact[0] != SIDE_OPPOSITE[direction]:
        return None
    merged_extra = _extension_cells(anchor_coords, seeker_final, SIDE_OPPOSITE[direction], seeker_area)

    footprint = _cells_union(anchor_coords, seeker_start, seeker_final, merged_extra)
    footprint |= _path_cells(seeker_mask, seeker_start_row, seeker_start_col, direction, gap)
    expanded = _expand_cells(footprint, (h, w))
    if expanded & reserved:
        return None

    input_objects = [(color, anchor_coords), (color, seeker_start)]
    output_objects = [(color, anchor_coords), (color, merged_extra)]
    return input_objects, output_objects, expanded


def _place_single(
    h: int,
    w: int,
    reserved: set[tuple[int, int]],
    color: int,
    direction: str,
) -> tuple[list[tuple[int, np.ndarray]], list[tuple[int, np.ndarray]], set[tuple[int, int]]] | None:
    mask = _orient_left_mask(random.choice(SEEKER_LEFT_TEMPLATES), direction)
    assert _heavy_direction_matches(mask, direction)
    oh, ow = mask.shape

    if direction in {"left", "right"}:
        final_row = random.randint(0, h - oh)
        final_col = 0 if direction == "left" else w - ow
        max_gap = w - ow - final_col if direction == "left" else final_col
        if max_gap < GEN_GAP_MIN:
            return None
        gap = random.randint(GEN_GAP_MIN, min(GEN_GAP_MAX, max_gap))
    else:
        final_col = random.randint(0, w - ow)
        final_row = 0 if direction == "up" else h - oh
        max_gap = h - oh - final_row if direction == "up" else final_row
        if max_gap < GEN_GAP_MIN:
            return None
        gap = random.randint(GEN_GAP_MIN, min(GEN_GAP_MAX, max_gap))

    dr, dc = DIRS4[direction]
    start_row = final_row - gap * dr
    start_col = final_col - gap * dc
    start_coords = _mask_coords(mask, start_row, start_col)
    final_coords = _mask_coords(mask, final_row, final_col)
    if (
        int(start_coords[:, 0].min()) < 0
        or int(start_coords[:, 0].max()) >= h
        or int(start_coords[:, 1].min()) < 0
        or int(start_coords[:, 1].max()) >= w
    ):
        return None

    footprint = _cells_union(start_coords, final_coords) | _path_cells(mask, start_row, start_col, direction, gap)
    expanded = _expand_cells(footprint, (h, w))
    if expanded & reserved:
        return None

    input_objects = [(color, start_coords)]
    output_objects = [(color, final_coords)]
    return input_objects, output_objects, expanded


def _gen_attempt() -> np.ndarray | None:
    h = random.randint(*GEN_ROWS)
    w = random.randint(*GEN_COLS)
    pair_count = random.randint(*GEN_PAIR_COUNTS)
    single_count = random.randint(*GEN_SINGLE_COUNTS)
    colors = random.sample(FOREGROUND_COLORS, pair_count + single_count)

    reserved: set[tuple[int, int]] = set()
    input_objects: list[tuple[int, np.ndarray]] = []
    output_objects: list[tuple[int, np.ndarray]] = []

    for color in colors[:pair_count]:
        placed = None
        for _ in range(80):
            direction = random.choice(tuple(DIRS4))
            placed = _place_pair(h, w, reserved, color, direction)
            if placed is not None:
                break
        if placed is None:
            return None
        new_in, new_out, block = placed
        input_objects.extend(new_in)
        output_objects.extend(new_out)
        reserved |= block

    for color in colors[pair_count:]:
        placed = None
        for _ in range(80):
            direction = random.choice(tuple(DIRS4))
            placed = _place_single(h, w, reserved, color, direction)
            if placed is not None:
                break
        if placed is None:
            return None
        new_in, new_out, block = placed
        input_objects.extend(new_in)
        output_objects.extend(new_out)
        reserved |= block

    inp = _checkerboard((h, w), CHECKER_EVEN, CHECKER_ODD)
    out = _checkerboard((h, w), CHECKER_EVEN, CHECKER_ODD)
    for color, coords in input_objects:
        _paint_coords(inp, coords, color)
    for color, coords in output_objects:
        _paint_coords(out, coords, color)

    found = _find_components(inp, _background_mask(inp, CHECKER_EVEN, CHECKER_ODD))
    if len(found) != len(input_objects):
        return None
    solved = main(inp)
    if not np.array_equal(solved, out):
        return None
    return inp


def generate_input() -> np.ndarray:
    for _ in range(GEN_RETRIES):
        grid = _gen_attempt()
        if grid is not None:
            return grid
    raise RuntimeError("failed to generate a valid sliding-merge checkerboard sample")
