from __future__ import annotations

from collections import deque

import numpy as np


BLUE = 1
RED = 2
PAIR_COLORS = [3, 4, 5, 6, 7, 8, 9]
DIRS = [(-1, 0), (0, 1), (1, 0), (0, -1)]


def _neighbors4(r: int, c: int) -> tuple[tuple[int, int], ...]:
    return ((r - 1, c), (r + 1, c), (r, c - 1), (r, c + 1))


def _most_common_color(grid: np.ndarray) -> int:
    values, counts = np.unique(grid, return_counts=True)
    return int(values[np.argmax(counts)])


def _components(mask: np.ndarray) -> list[list[tuple[int, int]]]:
    h, w = mask.shape
    seen = np.zeros_like(mask, dtype=bool)
    out: list[list[tuple[int, int]]] = []

    for r in range(h):
        for c in range(w):
            if not mask[r, c] or seen[r, c]:
                continue
            q = deque([(r, c)])
            seen[r, c] = True
            cells: list[tuple[int, int]] = []
            while q:
                cr, cc = q.popleft()
                cells.append((cr, cc))
                for nr, nc in _neighbors4(cr, cc):
                    if 0 <= nr < h and 0 <= nc < w and mask[nr, nc] and not seen[nr, nc]:
                        seen[nr, nc] = True
                        q.append((nr, nc))
            out.append(cells)
    return out


def _bbox(cells: list[tuple[int, int]]) -> tuple[int, int, int, int]:
    rows = [r for r, _ in cells]
    cols = [c for _, c in cells]
    return min(rows), min(cols), max(rows), max(cols)


def _is_solid_rectangle(grid: np.ndarray, cells: list[tuple[int, int]], color: int) -> bool:
    r0, c0, r1, c1 = _bbox(cells)
    return (
        len(cells) == (r1 - r0 + 1) * (c1 - c0 + 1)
        and np.all(grid[r0 : r1 + 1, c0 : c1 + 1] == color)
    )


def _middle_values(lo: int, hi: int) -> list[int]:
    length = hi - lo + 1
    if length % 2 == 1:
        return [lo + length // 2]
    mid = lo + length // 2
    return [mid - 1, mid]


def _side_centers(box: tuple[int, int, int, int], dir_idx: int) -> list[tuple[int, int]]:
    r0, c0, r1, c1 = box
    if dir_idx == 0:
        return [(r0, c) for c in _middle_values(c0, c1)]
    if dir_idx == 1:
        return [(r, c1) for r in _middle_values(r0, r1)]
    if dir_idx == 2:
        return [(r1, c) for c in _middle_values(c0, c1)]
    return [(r, c0) for r in _middle_values(r0, r1)]


def _center(box: tuple[int, int, int, int]) -> tuple[float, float]:
    r0, c0, r1, c1 = box
    return (r0 + r1) / 2.0, (c0 + c1) / 2.0


def _dirs_toward(a_box: tuple[int, int, int, int], b_box: tuple[int, int, int, int]) -> list[int]:
    ar, ac = _center(a_box)
    br, bc = _center(b_box)
    candidates: list[tuple[float, int]] = []
    if br < ar:
        candidates.append((abs(br - ar), 0))
    elif br > ar:
        candidates.append((abs(br - ar), 2))
    if bc > ac:
        candidates.append((abs(bc - ac), 1))
    elif bc < ac:
        candidates.append((abs(bc - ac), 3))
    candidates.sort(reverse=True)
    return [dir_idx for _, dir_idx in candidates]


def _follow_path(
    grid: np.ndarray,
    bg: int,
    source_mask: np.ndarray,
    target_mask: np.ndarray,
    start: tuple[int, int],
    dir_idx: int,
) -> tuple[list[tuple[int, int]], int] | None:
    h, w = grid.shape
    path = [start]
    cur = start
    cur_dir = dir_idx
    seen_states = {(start[0], start[1], cur_dir)}

    for _ in range(h * w * 4):
        dr, dc = DIRS[cur_dir]
        nr, nc = cur[0] + dr, cur[1] + dc
        if not (0 <= nr < h and 0 <= nc < w):
            return None
        if source_mask[nr, nc]:
            return None
        if target_mask[nr, nc]:
            path.append((nr, nc))
            return path, cur_dir

        value = int(grid[nr, nc])
        if value not in (bg, RED, BLUE):
            return None

        cur = (nr, nc)
        path.append(cur)
        if value == RED:
            cur_dir = (cur_dir + 1) % 4
        elif value == BLUE:
            cur_dir = (cur_dir - 1) % 4

        state = (cur[0], cur[1], cur_dir)
        if state in seen_states:
            return None
        seen_states.add(state)
    return None


def _pair_rectangles(grid: np.ndarray) -> tuple[int, dict[int, list[dict[str, object]]]]:
    bg = _most_common_color(grid)
    pair_colors = [int(color) for color in np.unique(grid) if color not in (bg, BLUE, RED)]
    rectangles: dict[int, list[dict[str, object]]] = {}

    for color in pair_colors:
        comps = _components(grid == color)
        assert len(comps) == 2, "each pair color must form exactly two rectangles"
        info_list: list[dict[str, object]] = []
        for cells in comps:
            assert _is_solid_rectangle(grid, cells, color), "objects must be solid rectangles"
            mask = np.zeros_like(grid, dtype=bool)
            for r, c in cells:
                mask[r, c] = True
            info_list.append({"bbox": _bbox(cells), "mask": mask})
        rectangles[color] = info_list

    return bg, rectangles


def _solve_paths(grid: np.ndarray) -> list[tuple[int, list[tuple[int, int]]]]:
    g = np.asarray(grid, dtype=int)
    bg, rectangles = _pair_rectangles(g)
    solved: list[tuple[int, list[tuple[int, int]]]] = []

    for color in sorted(rectangles):
        rects = rectangles[color]
        unique_paths: dict[tuple[tuple[int, int], ...], list[tuple[int, int]]] = {}

        for src_idx, tgt_idx in ((0, 1), (1, 0)):
            source = rects[src_idx]
            target = rects[tgt_idx]
            source_box = source["bbox"]
            target_box = target["bbox"]
            source_mask = source["mask"]
            target_mask = target["mask"]

            for dir_idx in _dirs_toward(source_box, target_box):
                for start in _side_centers(source_box, dir_idx):
                    result = _follow_path(
                        g,
                        bg,
                        source_mask,
                        target_mask,
                        start,
                        dir_idx,
                    )
                    if result is None:
                        continue
                    path, enter_dir = result
                    end_side = set(_side_centers(target_box, (enter_dir + 2) % 4))
                    if path[-1] not in end_side:
                        continue
                    unique_paths[tuple(path)] = path

        assert len(unique_paths) == 1, "each pair must have a unique valid marker-guided path"
        solved.append((color, next(iter(unique_paths.values()))))

    return solved


def main(grid: np.ndarray) -> np.ndarray:
    """
    Connect each same-color rectangle pair by following guideposts.

    Red guideposts turn the path right, blue guideposts turn it left.
    The drawn line uses the pair color, while rectangles and guideposts stay unchanged.
    """
    g = np.asarray(grid, dtype=int)
    out = g.copy()
    for color, path in _solve_paths(g):
        for r, c in path:
            if int(out[r, c]) in (RED, BLUE):
                continue
            out[r, c] = color
    return out


def _turn_color(turn: int) -> int:
    return RED if turn == 1 else BLUE


def _trace_polyline(
    start: tuple[int, int],
    start_dir: int,
    turns: list[int],
    lengths: list[int],
) -> tuple[list[tuple[int, int]], dict[tuple[int, int], int], int] | None:
    path = [start]
    turn_colors: dict[tuple[int, int], int] = {}
    visited = {start}
    r, c = start
    cur_dir = start_dir

    for seg_idx, steps in enumerate(lengths):
        dr, dc = DIRS[cur_dir]
        for _ in range(steps):
            r += dr
            c += dc
            if (r, c) in visited:
                return None
            visited.add((r, c))
            path.append((r, c))
        if seg_idx < len(turns):
            turn_colors[(r, c)] = _turn_color(turns[seg_idx])
            cur_dir = (cur_dir + turns[seg_idx]) % 4

    return path, turn_colors, cur_dir


def _sample_rect_for_side(rng: np.random.Generator, side_dir: int) -> tuple[int, int]:
    span = int(rng.choice([1, 3]))
    depth = int(rng.choice([2, 3]))
    if side_dir in (1, 3):
        return span, depth
    return depth, span


def _rect_from_edge(
    edge: tuple[int, int],
    side_dir: int,
    shape: tuple[int, int],
) -> tuple[int, int, int, int]:
    r, c = edge
    height, width = shape
    if side_dir == 1:
        half = height // 2
        return r - half, c - width + 1, r + half, c
    if side_dir == 3:
        half = height // 2
        return r - half, c, r + half, c + width - 1
    if side_dir == 0:
        half = width // 2
        return r, c - half, r + height - 1, c + half
    half = width // 2
    return r - height + 1, c - half, r, c + half


def _cells_in_box(box: tuple[int, int, int, int]) -> set[tuple[int, int]]:
    r0, c0, r1, c1 = box
    return {(r, c) for r in range(r0, r1 + 1) for c in range(c0, c1 + 1)}


def _box_in_bounds(box: tuple[int, int, int, int], h: int, w: int) -> bool:
    r0, c0, r1, c1 = box
    return 0 <= r0 <= r1 < h and 0 <= c0 <= c1 < w


def _expand_cells(cells: set[tuple[int, int]], h: int, w: int) -> set[tuple[int, int]]:
    expanded = set(cells)
    for r, c in list(cells):
        for nr, nc in _neighbors4(r, c):
            if 0 <= nr < h and 0 <= nc < w:
                expanded.add((nr, nc))
    return expanded


def _pair_count_for_shape(h: int, w: int, rng: np.random.Generator) -> int:
    if min(h, w) <= 13:
        return int(rng.choice([1, 2], p=[0.45, 0.55]))
    return int(rng.choice([1, 2, 3], p=[0.25, 0.5, 0.25]))


def _paint_box(grid: np.ndarray, box: tuple[int, int, int, int], color: int) -> None:
    r0, c0, r1, c1 = box
    grid[r0 : r1 + 1, c0 : c1 + 1] = color


def _build_expected(
    h: int,
    w: int,
    plans: list[dict[str, object]],
) -> tuple[np.ndarray, np.ndarray]:
    out = np.zeros((h, w), dtype=int)
    inp = np.zeros((h, w), dtype=int)

    for plan in plans:
        color = int(plan["color"])
        source_box = plan["source_box"]
        target_box = plan["target_box"]
        path = plan["path"]
        turn_colors = plan["turn_colors"]
        source_cells = _cells_in_box(source_box)
        target_cells = _cells_in_box(target_box)

        _paint_box(out, source_box, color)
        _paint_box(out, target_box, color)
        _paint_box(inp, source_box, color)
        _paint_box(inp, target_box, color)

        for cell, marker_color in turn_colors.items():
            out[cell] = marker_color
            inp[cell] = marker_color

        for r, c in path:
            if (r, c) in turn_colors:
                continue
            out[r, c] = color
            if (r, c) not in source_cells and (r, c) not in target_cells:
                inp[r, c] = 0

    return inp, out


def _sample_plan(
    h: int,
    w: int,
    color: int,
    blocked: set[tuple[int, int]],
    rng: np.random.Generator,
) -> dict[str, object] | None:
    turn_count = int(rng.choice([1, 2, 3], p=[0.35, 0.4, 0.25]))
    turns = [int(rng.choice([-1, 1])) for _ in range(turn_count)]
    lengths = [int(rng.integers(2, 6)) for _ in range(turn_count + 1)]
    start_dir = int(rng.integers(0, 4))
    start = (int(rng.integers(0, h)), int(rng.integers(0, w)))

    traced = _trace_polyline(start, start_dir, turns, lengths)
    if traced is None:
        return None
    path, turn_colors, end_dir = traced

    if any(not (0 <= r < h and 0 <= c < w) for r, c in path):
        return None

    source_shape = _sample_rect_for_side(rng, start_dir)
    target_shape = _sample_rect_for_side(rng, (end_dir + 2) % 4)
    source_box = _rect_from_edge(path[0], start_dir, source_shape)
    target_box = _rect_from_edge(path[-1], (end_dir + 2) % 4, target_shape)

    if not _box_in_bounds(source_box, h, w) or not _box_in_bounds(target_box, h, w):
        return None

    source_cells = _cells_in_box(source_box)
    target_cells = _cells_in_box(target_box)
    path_cells = set(path)

    if source_cells & (path_cells - {path[0]}):
        return None
    if target_cells & (path_cells - {path[-1]}):
        return None
    if (source_cells - {path[0]}) & target_cells:
        return None

    occupied = source_cells | target_cells | path_cells
    if occupied & blocked:
        return None

    return {
        "color": color,
        "source_box": source_box,
        "target_box": target_box,
        "path": path,
        "turn_colors": turn_colors,
        "occupied": occupied,
    }


def _valid_generated_input(inp: np.ndarray, out: np.ndarray) -> bool:
    if _most_common_color(inp) != 0:
        return False
    if np.count_nonzero(inp == RED) + np.count_nonzero(inp == BLUE) == 0:
        return False
    try:
        actual = main(inp)
    except Exception:
        return False
    return np.array_equal(actual, out)


def _attempt_generate(rng: np.random.Generator) -> np.ndarray | None:
    h = int(rng.integers(12, 19))
    w = int(rng.integers(12, 19))
    pair_count = _pair_count_for_shape(h, w, rng)
    colors = [int(c) for c in rng.choice(PAIR_COLORS, size=pair_count, replace=False)]

    blocked: set[tuple[int, int]] = set()
    plans: list[dict[str, object]] = []

    for color in colors:
        accepted = None
        for _ in range(200):
            plan = _sample_plan(h, w, color, blocked, rng)
            if plan is None:
                continue
            accepted = plan
            break
        if accepted is None:
            return None
        plans.append(accepted)
        blocked |= _expand_cells(accepted["occupied"], h, w)

    inp, out = _build_expected(h, w, plans)
    if not _valid_generated_input(inp, out):
        return None
    return inp


def generate_input() -> np.ndarray:
    rng = np.random.default_rng()
    for _ in range(500):
        grid = _attempt_generate(rng)
        if grid is not None:
            return grid
    raise RuntimeError("failed to generate a valid marker-guided connector input")
