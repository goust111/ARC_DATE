import random

import numpy as np


BG = 0
WALL = 5
PATTERN_COLORS = (1, 2, 3, 4, 6, 7, 8, 9)

ROOM_COUNT_RANGE = (2, 4)
ROOM_SIZE_RANGE = (3, 6)
PAIR_COUNT_CAP = 4
MAX_GENERATION_ATTEMPTS = 400


def _line_rows(grid: np.ndarray, color: int) -> list[int]:
    return [
        r
        for r in range(grid.shape[0])
        if np.any((grid[r, :-1] == color) & (grid[r, 1:] == color))
    ]


def _line_cols(grid: np.ndarray, color: int) -> list[int]:
    return [
        c
        for c in range(grid.shape[1])
        if np.any((grid[:-1, c] == color) & (grid[1:, c] == color))
    ]


def _find_wall_color(grid: np.ndarray, bg: int = BG) -> int:
    h, w = grid.shape
    candidates: list[tuple[int, int]] = []
    for color in sorted(int(v) for v in np.unique(grid) if int(v) != int(bg)):
        rows = _line_rows(grid, color)
        cols = _line_cols(grid, color)
        if len(rows) < 2 or len(cols) < 2:
            continue
        if 0 not in rows or h - 1 not in rows or 0 not in cols or w - 1 not in cols:
            continue
        rowset = set(rows)
        colset = set(cols)
        coords = np.argwhere(grid == color)
        if all(int(r) in rowset or int(c) in colset for r, c in coords):
            candidates.append((int(coords.shape[0]), color))
    assert candidates, "expected a wall color"
    return max(candidates)[1]


def _room_bbox(
    wall_rows: list[int],
    wall_cols: list[int],
    room_r: int,
    room_c: int,
) -> tuple[int, int, int, int]:
    return (
        wall_rows[room_r] + 1,
        wall_rows[room_r + 1] - 1,
        wall_cols[room_c] + 1,
        wall_cols[room_c + 1] - 1,
    )


def _extract_rooms(
    grid: np.ndarray,
    wall_rows: list[int],
    wall_cols: list[int],
    wall_color: int,
    bg: int = BG,
) -> dict[tuple[int, int], dict[str, object]]:
    rooms: dict[tuple[int, int], dict[str, object]] = {}
    for rr in range(len(wall_rows) - 1):
        for cc in range(len(wall_cols) - 1):
            r0, r1, c0, c1 = _room_bbox(wall_rows, wall_cols, rr, cc)
            patch = grid[r0 : r1 + 1, c0 : c1 + 1]
            mask = (patch != int(bg)) & (patch != int(wall_color))
            coords = np.argwhere(mask)
            if coords.size == 0:
                pattern_color = None
                abs_coords = np.empty((0, 2), dtype=int)
            else:
                vals = {int(patch[int(r), int(c)]) for r, c in coords}
                assert len(vals) == 1, "room patterns must be monochrome"
                pattern_color = vals.pop()
                abs_coords = coords + np.array([r0, c0], dtype=int)
            rooms[(rr, cc)] = {
                "bbox": (r0, r1, c0, c1),
                "pattern_color": pattern_color,
                "pattern_coords": abs_coords,
            }
    return rooms


def _gap_moves(
    grid: np.ndarray,
    rooms: dict[tuple[int, int], dict[str, object]],
    wall_rows: list[int],
    wall_cols: list[int],
    bg: int = BG,
) -> list[tuple[tuple[int, int], tuple[int, int], str]]:
    n_room_rows = len(wall_rows) - 1
    n_room_cols = len(wall_cols) - 1
    moves: list[tuple[tuple[int, int], tuple[int, int], str]] = []

    for rr in range(n_room_rows):
        for cc in range(n_room_cols - 1):
            wall_c = wall_cols[cc + 1]
            r0 = wall_rows[rr] + 1
            r1 = wall_rows[rr + 1] - 1
            gap_rows = [r for r in range(r0, r1 + 1) if int(grid[r, wall_c]) == int(bg)]
            if not gap_rows:
                continue
            assert len(gap_rows) == 1, "expected at most one gap per vertical wall segment"
            left = (rr, cc)
            right = (rr, cc + 1)
            left_has = rooms[left]["pattern_color"] is not None
            right_has = rooms[right]["pattern_color"] is not None
            if left_has ^ right_has:
                if left_has:
                    moves.append((left, right, "right"))
                else:
                    moves.append((right, left, "left"))

    for rr in range(n_room_rows - 1):
        for cc in range(n_room_cols):
            wall_r = wall_rows[rr + 1]
            c0 = wall_cols[cc] + 1
            c1 = wall_cols[cc + 1] - 1
            gap_cols = [c for c in range(c0, c1 + 1) if int(grid[wall_r, c]) == int(bg)]
            if not gap_cols:
                continue
            assert len(gap_cols) == 1, "expected at most one gap per horizontal wall segment"
            top = (rr, cc)
            bottom = (rr + 1, cc)
            top_has = rooms[top]["pattern_color"] is not None
            bottom_has = rooms[bottom]["pattern_color"] is not None
            if top_has ^ bottom_has:
                if top_has:
                    moves.append((top, bottom, "down"))
                else:
                    moves.append((bottom, top, "up"))

    return moves


def _translate_to_target(
    coords: np.ndarray,
    src_bbox: tuple[int, int, int, int],
    dst_bbox: tuple[int, int, int, int],
    direction: str,
) -> tuple[int, int]:
    pr0 = int(coords[:, 0].min())
    pr1 = int(coords[:, 0].max())
    pc0 = int(coords[:, 1].min())
    pc1 = int(coords[:, 1].max())
    tr0, tr1, tc0, tc1 = dst_bbox
    if direction == "right":
        return 0, int(tc1 - pc1)
    if direction == "left":
        return 0, int(tc0 - pc0)
    if direction == "down":
        return int(tr1 - pr1), 0
    assert direction == "up"
    return int(tr0 - pr0), 0


def main(grid: np.ndarray) -> np.ndarray:
    """
    Push each room's colored pattern through its wall gap into the paired blank room.

    Input structure:
    - The background is black.
    - One wall color forms a one-pixel orthogonal room network with an outer border.
    - Rooms are axis-aligned rectangles between consecutive wall rows and wall columns.
    - Some internal wall segments contain a single black gap connecting two adjacent rooms.
    - A non-empty room contains one monochrome multi-pixel 4-connected pattern; its paired room across the gap is empty.

    Rule:
    - Detect the wall rows and wall columns, then recover every rectangular room.
    - For each gap, find which side contains the pattern and which side is the blank target room.
    - Erase the pattern from its source room.
    - Translate that pattern straight across the gap into the target room so its leading edge becomes flush with the target room wall opposite the gap.
    - Leave every wall pixel and every gap pixel unchanged.

    Key insight:
    - The black gap is not part of a shape: it identifies both the paired neighboring room and the movement direction, while the destination offset is determined by the opposite wall of that target room.
    """
    g = np.asarray(grid, dtype=int)
    assert g.ndim == 2

    bg = BG
    wall_color = _find_wall_color(g, bg=bg)
    wall_rows = _line_rows(g, wall_color)
    wall_cols = _line_cols(g, wall_color)
    rooms = _extract_rooms(g, wall_rows, wall_cols, wall_color, bg=bg)
    moves = _gap_moves(g, rooms, wall_rows, wall_cols, bg=bg)

    patterned_rooms = {
        room_id for room_id, info in rooms.items() if info["pattern_color"] is not None
    }
    source_ids = [src for src, _dst, _direction in moves]
    assert len(source_ids) == len(set(source_ids)), "each source room must have one gap"
    assert patterned_rooms == set(source_ids), "every patterned room must pair with one blank room"

    out = g.copy()
    for src, dst, direction in moves:
        src_info = rooms[src]
        dst_info = rooms[dst]
        color = int(src_info["pattern_color"])
        coords = np.asarray(src_info["pattern_coords"], dtype=int)
        dr, dc = _translate_to_target(
            coords,
            src_info["bbox"],
            dst_info["bbox"],
            direction,
        )
        out[coords[:, 0], coords[:, 1]] = int(bg)
        tr0, tr1, tc0, tc1 = dst_info["bbox"]
        for r, c in coords:
            rr = int(r + dr)
            cc = int(c + dc)
            assert tr0 <= rr <= tr1 and tc0 <= cc <= tc1
            assert int(out[rr, cc]) == int(bg)
            out[rr, cc] = color
    return out


def _room_sizes(count: int) -> list[int]:
    return [random.randint(*ROOM_SIZE_RANGE) for _ in range(count)]


def _wall_positions(sizes: list[int]) -> list[int]:
    pos = [0]
    cursor = 0
    for size in sizes:
        cursor += size + 1
        pos.append(cursor)
    return pos


def _adjacent_edges(nr: int, nc: int) -> list[tuple[tuple[int, int], tuple[int, int], str]]:
    edges: list[tuple[tuple[int, int], tuple[int, int], str]] = []
    for rr in range(nr):
        for cc in range(nc):
            if cc + 1 < nc:
                edges.append(((rr, cc), (rr, cc + 1), "horizontal"))
            if rr + 1 < nr:
                edges.append(((rr, cc), (rr + 1, cc), "vertical"))
    return edges


def _sample_pairs(nr: int, nc: int) -> list[tuple[tuple[int, int], tuple[int, int], str]] | None:
    edges = _adjacent_edges(nr, nc)
    max_pairs = min(PAIR_COUNT_CAP, (nr * nc) // 2)
    pair_count = random.randint(1, max_pairs)
    for _ in range(100):
        unused = {(rr, cc) for rr in range(nr) for cc in range(nc)}
        pairs: list[tuple[tuple[int, int], tuple[int, int], str]] = []
        while unused and len(pairs) < pair_count:
            candidates = [edge for edge in edges if edge[0] in unused and edge[1] in unused]
            if not candidates:
                break
            edge = random.choice(candidates)
            pairs.append(edge)
            unused.remove(edge[0])
            unused.remove(edge[1])
        if len(pairs) != pair_count:
            continue
        return pairs
    return None


def _grow_connected_shape(height: int, width: int, area: int) -> np.ndarray:
    cells = {(random.randrange(height), random.randrange(width))}
    while len(cells) < area:
        frontier = []
        for r, c in cells:
            for dr, dc in ((1, 0), (-1, 0), (0, 1), (0, -1)):
                rr = r + dr
                cc = c + dc
                if 0 <= rr < height and 0 <= cc < width and (rr, cc) not in cells:
                    frontier.append((rr, cc))
        cells.add(random.choice(frontier))
    patch = np.zeros((height, width), dtype=bool)
    for r, c in cells:
        patch[r, c] = True
    return patch


def _shape_patch(src_h: int, src_w: int, dst_h: int, dst_w: int, direction: str) -> np.ndarray:
    max_h = min(src_h, dst_h)
    max_w = min(src_w, dst_w)
    if direction in {"left", "right"} and max_w >= 3:
        max_w -= 1
    if direction in {"up", "down"} and max_h >= 3:
        max_h -= 1
    max_h = max(2, max_h)
    max_w = max(2, max_w)
    patch_h = random.randint(2, max_h)
    patch_w = random.randint(2, max_w)
    max_area = patch_h * patch_w
    area_hi = max(2, max_area - (1 if max_area > 3 else 0))
    area = random.randint(2, area_hi)
    return _grow_connected_shape(patch_h, patch_w, area)


def _place_pattern(
    grid: np.ndarray,
    room_bbox: tuple[int, int, int, int],
    patch: np.ndarray,
    color: int,
) -> np.ndarray:
    r0, r1, c0, c1 = room_bbox
    ph, pw = patch.shape
    top = random.randint(r0, r1 - ph + 1)
    left = random.randint(c0, c1 - pw + 1)
    coords = np.argwhere(patch) + np.array([top, left], dtype=int)
    for r, c in coords:
        grid[int(r), int(c)] = int(color)
    return coords


def _gap_cell(
    wall_rows: list[int],
    wall_cols: list[int],
    a: tuple[int, int],
    b: tuple[int, int],
) -> tuple[int, int]:
    if a[0] == b[0]:
        rr = a[0]
        wall_c = wall_cols[min(a[1], b[1]) + 1]
        top = wall_rows[rr] + 1
        bottom = wall_rows[rr + 1] - 1
        return random.randint(top, bottom), wall_c
    cc = a[1]
    wall_r = wall_rows[min(a[0], b[0]) + 1]
    left = wall_cols[cc] + 1
    right = wall_cols[cc + 1] - 1
    return wall_r, random.randint(left, right)


def _move_patch_to_target(
    coords: np.ndarray,
    dst_bbox: tuple[int, int, int, int],
    direction: str,
) -> np.ndarray:
    dr, dc = _translate_to_target(coords, (0, 0, 0, 0), dst_bbox, direction)
    return coords + np.array([dr, dc], dtype=int)


def _build_base_grid(room_heights: list[int], room_widths: list[int]) -> tuple[np.ndarray, list[int], list[int]]:
    wall_rows = _wall_positions(room_heights)
    wall_cols = _wall_positions(room_widths)
    height = wall_rows[-1] + 1
    width = wall_cols[-1] + 1
    grid = np.full((height, width), BG, dtype=int)
    for r in wall_rows:
        grid[r, :] = WALL
    for c in wall_cols:
        grid[:, c] = WALL
    return grid, wall_rows, wall_cols


def _gen_attempt() -> np.ndarray | None:
    nr = random.randint(*ROOM_COUNT_RANGE)
    nc = random.randint(*ROOM_COUNT_RANGE)
    room_heights = _room_sizes(nr)
    room_widths = _room_sizes(nc)
    inp, wall_rows, wall_cols = _build_base_grid(room_heights, room_widths)
    out = inp.copy()

    pairs = _sample_pairs(nr, nc)
    if pairs is None:
        return None
    colors = random.sample(PATTERN_COLORS, len(pairs))

    for color, (room_a, room_b, orientation) in zip(colors, pairs):
        if orientation == "horizontal":
            if random.random() < 0.5:
                src, dst, direction = room_a, room_b, "right"
            else:
                src, dst, direction = room_b, room_a, "left"
        else:
            if random.random() < 0.5:
                src, dst, direction = room_a, room_b, "down"
            else:
                src, dst, direction = room_b, room_a, "up"

        src_bbox = _room_bbox(wall_rows, wall_cols, src[0], src[1])
        dst_bbox = _room_bbox(wall_rows, wall_cols, dst[0], dst[1])
        src_h = src_bbox[1] - src_bbox[0] + 1
        src_w = src_bbox[3] - src_bbox[2] + 1
        dst_h = dst_bbox[1] - dst_bbox[0] + 1
        dst_w = dst_bbox[3] - dst_bbox[2] + 1
        patch = _shape_patch(src_h, src_w, dst_h, dst_w, direction)

        src_coords = _place_pattern(inp, src_bbox, patch, color)
        moved_coords = _move_patch_to_target(src_coords, dst_bbox, direction)
        for r, c in src_coords:
            out[int(r), int(c)] = BG
        for r, c in moved_coords:
            out[int(r), int(c)] = int(color)

        gap_r, gap_c = _gap_cell(wall_rows, wall_cols, room_a, room_b)
        inp[gap_r, gap_c] = BG
        out[gap_r, gap_c] = BG

    try:
        actual = main(inp)
    except Exception:
        return None
    if not np.array_equal(actual, out):
        return None
    return inp


def generate_input() -> np.ndarray:
    for _ in range(MAX_GENERATION_ATTEMPTS):
        grid = _gen_attempt()
        if grid is not None:
            return grid
    raise RuntimeError("failed to generate a valid room-gap ARC input")
