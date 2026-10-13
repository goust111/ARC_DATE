from __future__ import annotations

from collections import deque
import random

import numpy as np


BLACK = 0
GREY = 5
DIR4 = ((-1, 0), (1, 0), (0, -1), (0, 1))
COLOR_POOL = (1, 2, 3, 4, 6, 7, 8, 9)
SIDE_SETS = (("right", "top"), ("right", "bottom"), ("right", "top", "bottom"))
GENERATE_ATTEMPTS = 500


def _border_mask(height: int, width: int) -> np.ndarray:
    mask = np.zeros((height, width), dtype=bool)
    mask[0, :] = True
    mask[-1, :] = True
    mask[:, 0] = True
    mask[:, -1] = True
    return mask


def _box_overlap(a: tuple[int, int, int, int], b: tuple[int, int, int, int], pad: int = 0) -> bool:
    ar0, ac0, ah, aw = a
    br0, bc0, bh, bw = b
    return not (
        ar0 + ah - 1 + pad < br0
        or br0 + bh - 1 + pad < ar0
        or ac0 + aw - 1 + pad < bc0
        or bc0 + bw - 1 + pad < ac0
    )


def _carve_line(cells: set[tuple[int, int]], r0: int, c0: int, r1: int, c1: int) -> None:
    if r0 == r1:
        step = 1 if c1 >= c0 else -1
        for c in range(c0, c1 + step, step):
            cells.add((int(r0), int(c)))
        return
    if c0 == c1:
        step = 1 if r1 >= r0 else -1
        for r in range(r0, r1 + step, step):
            cells.add((int(r), int(c0)))
        return
    raise ValueError("lines must be axis-aligned")


def _adjacent_black_cells(grid: np.ndarray, box: tuple[int, int, int, int]) -> list[tuple[int, int]]:
    r0, c0, height, width = box
    h, w = grid.shape
    cells: set[tuple[int, int]] = set()
    r1 = r0 + height - 1
    c1 = c0 + width - 1
    for c in range(c0, c1 + 1):
        for rr in (r0, r1):
            for dr, dc in DIR4:
                nr, nc = rr + dr, c + dc
                if 0 <= nr < h and 0 <= nc < w and grid[nr, nc] == BLACK:
                    cells.add((nr, nc))
    for r in range(r0, r1 + 1):
        for cc in (c0, c1):
            for dr, dc in DIR4:
                nr, nc = r + dr, cc + dc
                if 0 <= nr < h and 0 <= nc < w and grid[nr, nc] == BLACK:
                    cells.add((nr, nc))
    return sorted(cells)


def _shortest_boundary_distance(
    grid: np.ndarray,
    source_box: tuple[int, int, int, int],
    target_box: tuple[int, int, int, int],
) -> int:
    starts = _adjacent_black_cells(grid, source_box)
    targets = set(_adjacent_black_cells(grid, target_box))
    assert starts and targets

    dist = np.full(grid.shape, -1, dtype=int)
    queue = deque()
    for r, c in starts:
        dist[r, c] = 1
        queue.append((r, c))

    while queue:
        r, c = queue.popleft()
        if (r, c) in targets:
            return int(dist[r, c] + 1)
        for dr, dc in DIR4:
            nr, nc = r + dr, c + dc
            if 0 <= nr < grid.shape[0] and 0 <= nc < grid.shape[1] and grid[nr, nc] == BLACK and dist[nr, nc] < 0:
                dist[nr, nc] = dist[r, c] + 1
                queue.append((nr, nc))

    raise AssertionError("template and window must be connected through black corridors")


def _shortest_black_path_stats(
    grid: np.ndarray,
    start: tuple[int, int],
    goal: tuple[int, int],
) -> tuple[int, int]:
    dist = {start: 0}
    ways = {start: 1}
    queue = deque([start])
    while queue:
        r, c = queue.popleft()
        for dr, dc in DIR4:
            nr, nc = r + dr, c + dc
            nxt = (nr, nc)
            if not (0 <= nr < grid.shape[0] and 0 <= nc < grid.shape[1]) or grid[nr, nc] != BLACK:
                continue
            nd = dist[(r, c)] + 1
            if nxt not in dist:
                dist[nxt] = nd
                ways[nxt] = ways[(r, c)]
                queue.append(nxt)
            elif dist[nxt] == nd:
                ways[nxt] = min(2, ways[nxt] + ways[(r, c)])
    assert goal in dist
    return int(dist[goal]), int(min(2, ways[goal]))


def _black_component(grid: np.ndarray, start: tuple[int, int]) -> set[tuple[int, int]]:
    seen = {start}
    queue = deque([start])
    while queue:
        r, c = queue.popleft()
        for dr, dc in DIR4:
            nr, nc = r + dr, c + dc
            nxt = (nr, nc)
            if 0 <= nr < grid.shape[0] and 0 <= nc < grid.shape[1] and grid[nr, nc] == BLACK and nxt not in seen:
                seen.add(nxt)
                queue.append(nxt)
    return seen


def _find_template(grid: np.ndarray) -> tuple[int, int, int, int, int, int]:
    h, w = grid.shape
    candidates: list[tuple[int, int, int, int, int, int]] = []
    for height in range(4, h + 1):
        for width in range(4, w + 1):
            border = _border_mask(height, width)
            for r0 in range(h - height + 1):
                for c0 in range(w - width + 1):
                    sub = grid[r0 : r0 + height, c0 : c0 + width]
                    frame_color = int(sub[0, 0])
                    if frame_color in (BLACK, GREY):
                        continue
                    if not np.all(sub[border] == frame_color):
                        continue
                    interior = sub[1:-1, 1:-1]
                    values = np.unique(interior)
                    if len(values) != 1:
                        continue
                    fill_color = int(values[0])
                    if fill_color in (BLACK, GREY) or fill_color == frame_color:
                        continue
                    candidates.append((r0, c0, height, width, frame_color, fill_color))
    assert len(candidates) == 1, f"expected one template, found {len(candidates)}"
    return candidates[0]


def _find_windows(
    grid: np.ndarray,
    height: int,
    width: int,
    source_origin: tuple[int, int],
) -> list[tuple[int, int, int, int]]:
    border = _border_mask(height, width)
    windows: list[tuple[int, int, int, int]] = []
    for r0 in range(grid.shape[0] - height + 1):
        for c0 in range(grid.shape[1] - width + 1):
            if (r0, c0) == source_origin:
                continue
            sub = grid[r0 : r0 + height, c0 : c0 + width]
            if np.all(sub[border] == GREY) and np.all(sub[~border] == BLACK):
                windows.append((r0, c0, height, width))
    return windows


def _paint_border(grid: np.ndarray, box: tuple[int, int, int, int], color: int) -> None:
    r0, c0, height, width = box
    border = _border_mask(height, width)
    view = grid[r0 : r0 + height, c0 : c0 + width]
    view[border] = int(color)


def _make_expected_output(
    grid: np.ndarray,
    template_box: tuple[int, int, int, int],
    window_boxes: list[tuple[int, int, int, int]],
    frame_color: int,
    fill_color: int,
) -> tuple[np.ndarray, list[int]]:
    out = grid.copy()
    parities: list[int] = []
    for box in window_boxes:
        distance = _shortest_boundary_distance(grid, template_box, box)
        parities.append(distance % 2)
        _paint_border(out, box, frame_color if distance % 2 == 1 else fill_color)
    return out, parities


def main(grid: np.ndarray) -> np.ndarray:
    """
    Recolor each empty matching window by using shortest-path parity to choose which template color is transferred.

    Input structure:
    - The background is black, with one unique non-grey colored rectangular template frame whose interior is uniformly filled by a second non-grey color.
    - The grid also contains two or three same-size empty windows with grey borders and black interiors.
    - Grey cells form maze walls, while black corridor cells are the only traversable cells between the template and each window.

    Rule:
    - Find the unique colored template frame and record both its frame color and its interior fill color.
    - Find every grey-bordered empty window with the same outer size as the template.
    - For each window, compute the shortest 4-connected path length from any template border pixel to any window border pixel, with only black cells allowed inside the path.
    - If that length is odd, recolor the window border with the template frame color; if it is even, recolor the window border with the template interior color. Leave interiors, corridors, and the template unchanged.

    Key insight:
    - The template carries two candidate colors, and the maze path contributes the deciding signal: only the parity of the shortest boundary-to-boundary route tells you which color to transmit.
    """
    g = np.asarray(grid, dtype=int)
    assert g.ndim == 2

    r0, c0, height, width, frame_color, fill_color = _find_template(g)
    template_box = (r0, c0, height, width)
    windows = _find_windows(g, height, width, (r0, c0))
    assert 2 <= len(windows) <= 3, f"expected 2 or 3 windows, found {len(windows)}"

    out, _ = _make_expected_output(g, template_box, windows, frame_color, fill_color)
    return out


def _object_box(
    side: str,
    maze_r0: int,
    maze_c0: int,
    maze_h: int,
    maze_w: int,
    obj_h: int,
    obj_w: int,
    present_sides: set[str],
) -> tuple[int, int, int, int]:
    if side == "left":
        row_lo = maze_r0 + 1
        row_hi = maze_r0 + maze_h - obj_h - 1
        if "top" in present_sides:
            row_lo = max(row_lo, maze_r0 + obj_h + 2)
        if "bottom" in present_sides:
            row_hi = min(row_hi, maze_r0 + maze_h - 2 * obj_h - 2)
        r0 = random.randint(row_lo, row_hi)
        c0 = maze_c0 + 1
        return (r0, c0, obj_h, obj_w)
    if side == "right":
        row_lo = maze_r0 + 1
        row_hi = maze_r0 + maze_h - obj_h - 1
        if "top" in present_sides:
            row_lo = max(row_lo, maze_r0 + obj_h + 2)
        if "bottom" in present_sides:
            row_hi = min(row_hi, maze_r0 + maze_h - 2 * obj_h - 2)
        r0 = random.randint(row_lo, row_hi)
        c0 = maze_c0 + maze_w - obj_w - 1
        return (r0, c0, obj_h, obj_w)
    if side == "top":
        col_lo = maze_c0 + obj_w + 2
        col_hi = maze_c0 + maze_w - 2 * obj_w - 2
        c0 = random.randint(col_lo, col_hi)
        r0 = maze_r0 + 1
        return (r0, c0, obj_h, obj_w)
    if side == "bottom":
        col_lo = maze_c0 + obj_w + 2
        col_hi = maze_c0 + maze_w - 2 * obj_w - 2
        c0 = random.randint(col_lo, col_hi)
        r0 = maze_r0 + maze_h - obj_h - 1
        return (r0, c0, obj_h, obj_w)
    raise ValueError(side)


def _door_for_box(side: str, box: tuple[int, int, int, int]) -> tuple[int, int]:
    r0, c0, height, width = box
    if side == "left":
        return (r0 + height // 2, c0 + width)
    if side == "right":
        return (r0 + height // 2, c0 - 1)
    if side == "top":
        return (r0 + height, c0 + width // 2)
    if side == "bottom":
        return (r0 - 1, c0 + width // 2)
    raise ValueError(side)


def _stamp_template(grid: np.ndarray, box: tuple[int, int, int, int], frame_color: int, fill_color: int) -> None:
    r0, c0, height, width = box
    sub = grid[r0 : r0 + height, c0 : c0 + width]
    sub[:, :] = fill_color
    sub[_border_mask(height, width)] = frame_color


def _stamp_window(grid: np.ndarray, box: tuple[int, int, int, int]) -> None:
    r0, c0, height, width = box
    sub = grid[r0 : r0 + height, c0 : c0 + width]
    sub[:, :] = BLACK
    sub[_border_mask(height, width)] = GREY


def _validate_instance(
    grid: np.ndarray,
    template_box: tuple[int, int, int, int],
    window_boxes: list[tuple[int, int, int, int]],
    source_door: tuple[int, int],
    window_doors: list[tuple[int, int]],
    frame_color: int,
    fill_color: int,
) -> bool:
    try:
        found = _find_template(grid)
        if found[:4] != template_box or found[4] != frame_color or found[5] != fill_color:
            return False

        found_windows = sorted(_find_windows(grid, template_box[2], template_box[3], template_box[:2]))
        if found_windows != sorted(window_boxes):
            return False

        component = _black_component(grid, source_door)
        if [cell for cell in _adjacent_black_cells(grid, template_box) if cell in component] != [source_door]:
            return False
        for box, door in zip(window_boxes, window_doors):
            if [cell for cell in _adjacent_black_cells(grid, box) if cell in component] != [door]:
                return False
        if any(door not in component for door in window_doors):
            return False

        edge_count = 0
        max_degree = 0
        for r, c in component:
            degree = sum((r + dr, c + dc) in component for dr, dc in DIR4)
            edge_count += degree
            max_degree = max(max_degree, degree)
        edge_count //= 2
        if edge_count != len(component) - 1:
            return False
        if max_degree < 3:
            return False

        parities: list[int] = []
        for door, box in zip(window_doors, window_boxes):
            dist, ways = _shortest_black_path_stats(grid, source_door, door)
            if ways != 1:
                return False
            boundary_dist = dist + 2
            if boundary_dist != _shortest_boundary_distance(grid, template_box, box):
                return False
            parities.append(boundary_dist % 2)

        if not (any(p == 0 for p in parities) and any(p == 1 for p in parities)):
            return False

        expected, _ = _make_expected_output(grid, template_box, window_boxes, frame_color, fill_color)
        if np.array_equal(expected, grid):
            return False
        if not np.array_equal(main(grid), expected):
            return False
        return True
    except AssertionError:
        return False


def _gen_attempt() -> np.ndarray | None:
    obj_h = random.randint(4, 5)
    obj_w = random.randint(4, 5)
    sides = random.choice(SIDE_SETS)

    min_maze_h = 3 * obj_h + 4 if "bottom" in sides and "top" in sides else 2 * obj_h + 5
    min_maze_w = 3 * obj_w + 4
    maze_h = random.randint(min_maze_h, min(24, min_maze_h + 4))
    maze_w = random.randint(min_maze_w, min(24, min_maze_w + 4))

    outer_h = maze_h + random.randint(2, min(6, 30 - maze_h))
    outer_w = maze_w + random.randint(2, min(6, 30 - maze_w))
    if outer_h > 30 or outer_w > 30:
        return None

    maze_r0 = random.randint(1, outer_h - maze_h - 1)
    maze_c0 = random.randint(1, outer_w - maze_w - 1)

    present_sides = set(sides)
    template_box = _object_box("left", maze_r0, maze_c0, maze_h, maze_w, obj_h, obj_w, present_sides)
    right_box = _object_box("right", maze_r0, maze_c0, maze_h, maze_w, obj_h, obj_w, present_sides)
    boxes = {"template": template_box, "right": right_box}
    for side in sides:
        if side != "right":
            boxes[side] = _object_box(side, maze_r0, maze_c0, maze_h, maze_w, obj_h, obj_w, present_sides)

    all_boxes = list(boxes.values())
    for i in range(len(all_boxes)):
        for j in range(i + 1, len(all_boxes)):
            if _box_overlap(all_boxes[i], all_boxes[j], pad=1):
                return None

    source_door = _door_for_box("left", template_box)
    right_door = _door_for_box("right", right_box)
    top_door = _door_for_box("top", boxes["top"]) if "top" in boxes else None
    bottom_door = _door_for_box("bottom", boxes["bottom"]) if "bottom" in boxes else None

    jr_lo = min(source_door[0], right_door[0])
    jr_hi = max(source_door[0], right_door[0])
    if top_door is not None:
        jr_lo = max(jr_lo, top_door[0] + 2)
    if bottom_door is not None:
        jr_hi = min(jr_hi, bottom_door[0] - 2)
    if jr_lo > jr_hi:
        return None
    jr = random.randint(jr_lo, jr_hi)

    jc_lo = source_door[1] + 2
    jc_hi = right_door[1] - 2
    if jc_lo > jc_hi:
        return None
    jc = random.randint(jc_lo, jc_hi)

    corridor: set[tuple[int, int]] = set()
    _carve_line(corridor, source_door[0], source_door[1], source_door[0], jc)
    _carve_line(corridor, source_door[0], jc, jr, jc)

    _carve_line(corridor, right_door[0], right_door[1], right_door[0], jc)
    _carve_line(corridor, right_door[0], jc, jr, jc)

    window_boxes = [right_box]
    window_doors = [right_door]
    if top_door is not None:
        _carve_line(corridor, top_door[0], top_door[1], jr, top_door[1])
        _carve_line(corridor, jr, top_door[1], jr, jc)
        window_boxes.append(boxes["top"])
        window_doors.append(top_door)
    if bottom_door is not None:
        _carve_line(corridor, bottom_door[0], bottom_door[1], jr, bottom_door[1])
        _carve_line(corridor, jr, bottom_door[1], jr, jc)
        window_boxes.append(boxes["bottom"])
        window_doors.append(bottom_door)

    frame_color, fill_color = random.sample(COLOR_POOL, 2)
    grid = np.zeros((outer_h, outer_w), dtype=int)
    grid[maze_r0 : maze_r0 + maze_h, maze_c0 : maze_c0 + maze_w] = GREY
    for r, c in corridor:
        grid[r, c] = BLACK

    _stamp_template(grid, template_box, frame_color, fill_color)
    for box in window_boxes:
        _stamp_window(grid, box)

    if not _validate_instance(grid, template_box, window_boxes, source_door, window_doors, frame_color, fill_color):
        return None
    return grid


def generate_input() -> np.ndarray:
    for _ in range(GENERATE_ATTEMPTS):
        grid = _gen_attempt()
        if grid is not None:
            return grid
    raise RuntimeError("failed to generate a valid parity-transfer maze instance")
