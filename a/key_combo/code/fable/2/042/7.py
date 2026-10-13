from __future__ import annotations

import random

import numpy as np


DIR4 = ((-1, 0), (1, 0), (0, -1), (0, 1))
GREY = 5
ARC_COLORS = tuple(range(10))

GRID_H_RANGE = (10, 18)
GRID_W_RANGE = (10, 18)
PALETTE_LEN_RANGE = (3, 6)
STRAIGHT_PROBABILITY = 0.15
MAX_GEN_ATTEMPTS = 700
MAX_PATH_ATTEMPTS = 120


def _get_bg(grid: np.ndarray) -> int:
    counts = np.bincount(np.asarray(grid, dtype=int).ravel(), minlength=10)
    return int(np.argmax(counts))


def _in_bounds(shape: tuple[int, int], r: int, c: int) -> bool:
    return 0 <= int(r) < int(shape[0]) and 0 <= int(c) < int(shape[1])


def _neighbors4(shape: tuple[int, int], r: int, c: int):
    for dr, dc in DIR4:
        nr = int(r) + dr
        nc = int(c) + dc
        if _in_bounds(shape, nr, nc):
            yield nr, nc


def _mask_components(mask: np.ndarray) -> list[list[tuple[int, int]]]:
    h, w = mask.shape
    seen = np.zeros((h, w), dtype=bool)
    comps: list[list[tuple[int, int]]] = []
    for r in range(h):
        for c in range(w):
            if not bool(mask[r, c]) or bool(seen[r, c]):
                continue
            stack = [(r, c)]
            seen[r, c] = True
            coords: list[tuple[int, int]] = []
            while stack:
                rr, cc = stack.pop()
                coords.append((int(rr), int(cc)))
                for nr, nc in _neighbors4(mask.shape, rr, cc):
                    if bool(mask[nr, nc]) and not bool(seen[nr, nc]):
                        seen[nr, nc] = True
                        stack.append((nr, nc))
            comps.append(coords)
    return comps


def _top_palette(row: np.ndarray, bg: int) -> tuple[list[int], tuple[int, int]]:
    idxs = [int(c) for c, value in enumerate(row) if int(value) != int(bg)]
    assert idxs, "expected a palette segment on the top row"
    assert idxs == list(range(idxs[0], idxs[-1] + 1)), "palette must be one contiguous top-row segment"
    palette = [int(v) for v in row[idxs[0] : idxs[-1] + 1]]
    assert len(palette) >= 3, "palette must contain at least three colors"
    assert len(set(palette)) == len(palette), "palette colors must be distinct"
    assert int(bg) not in palette and GREY not in palette, "palette must exclude background and gray"
    return palette, (idxs[0], idxs[-1])


def _corner_flags(path: list[tuple[int, int]]) -> list[bool]:
    flags = [False] * len(path)
    for i in range(1, len(path) - 1):
        prev_dir = (path[i][0] - path[i - 1][0], path[i][1] - path[i - 1][1])
        next_dir = (path[i + 1][0] - path[i][0], path[i + 1][1] - path[i][1])
        flags[i] = prev_dir != next_dir
    return flags


def _path_colors(
    palette: list[int],
    start_index: int,
    corner_flags: list[bool],
    *,
    mode: str = "at_corner",
) -> list[int]:
    phase = int(start_index)
    colors = [int(palette[phase])]
    for i in range(1, len(corner_flags)):
        step = 1
        if mode == "at_corner" and bool(corner_flags[i]):
            step = 2
        elif mode == "after_corner" and bool(corner_flags[i - 1]):
            step = 2
        elif mode == "never":
            step = 1
        elif mode not in {"at_corner", "after_corner", "never"}:
            raise ValueError(f"unknown coloring mode: {mode}")
        phase = (phase + step) % len(palette)
        colors.append(int(palette[phase]))
    return colors


def _order_path(coords: list[tuple[int, int]], start: tuple[int, int]) -> list[tuple[int, int]]:
    coord_set = set(coords)
    ordered = [start]
    prev: tuple[int, int] | None = None
    cur = start
    while True:
        nxts = [nbr for nbr in ((cur[0] - 1, cur[1]), (cur[0] + 1, cur[1]), (cur[0], cur[1] - 1), (cur[0], cur[1] + 1)) if nbr in coord_set and nbr != prev]
        if not nxts:
            break
        assert len(nxts) == 1, "path must be non-branching"
        prev, cur = cur, nxts[0]
        ordered.append(cur)
    assert len(ordered) == len(coord_set), "path ordering must visit every cell exactly once"
    return ordered


def _parse_instance(grid: np.ndarray) -> dict:
    g = np.asarray(grid, dtype=int)
    assert g.ndim == 2, "expected a 2D grid"
    h, w = g.shape
    assert h >= 4 and w >= 4, "grid is too small"

    bg = _get_bg(g)
    palette, palette_span = _top_palette(g[0], bg)
    palette_set = set(palette)

    lower_mask = np.zeros_like(g, dtype=bool)
    lower_mask[1:, :] = g[1:, :] != bg
    comps = _mask_components(lower_mask)
    assert len(comps) == 1, "expected exactly one path component below the palette row"
    coords = comps[0]
    assert len(coords) >= len(palette) + 2, "path must be long enough to show palette cycling"

    coord_set = set(coords)
    degrees = {
        cell: sum((nbr in coord_set) for nbr in ((cell[0] - 1, cell[1]), (cell[0] + 1, cell[1]), (cell[0], cell[1] - 1), (cell[0], cell[1] + 1)))
        for cell in coord_set
    }
    endpoints = [cell for cell, degree in degrees.items() if degree == 1]
    assert len(endpoints) == 2, "path must have exactly two endpoints"
    assert all(degree in {1, 2} for degree in degrees.values()), "path must be one-pixel-wide and unbranched"

    colored_path_cells = [cell for cell in coords if int(g[cell]) in palette_set]
    assert len(colored_path_cells) == 1, "expected exactly one palette-colored path endpoint"
    start = colored_path_cells[0]
    assert start in endpoints, "the palette-colored cell must be an endpoint"
    start_color = int(g[start])
    start_index = palette.index(start_color)

    gray_cells = [cell for cell in coords if cell != start]
    assert gray_cells, "path must contain gray cells beyond the colored start"
    assert all(int(g[cell]) == GREY for cell in gray_cells), "all non-start path cells must be gray"

    path = _order_path(coords, start)
    corners = _corner_flags(path)
    return {
        "grid": g,
        "bg": int(bg),
        "palette": palette,
        "palette_span": palette_span,
        "path": path,
        "corners": corners,
        "start": start,
        "start_color": start_color,
        "start_index": int(start_index),
    }


def _apply_rule(parsed: dict) -> np.ndarray:
    out = parsed["grid"].copy()
    path = parsed["path"]
    colors = _path_colors(parsed["palette"], parsed["start_index"], parsed["corners"], mode="at_corner")
    for (r, c), color in zip(path, colors):
        out[r, c] = int(color)
    return out


def main(grid: np.ndarray) -> np.ndarray:
    """
    Recolor the gray path from its colored endpoint using the palette cycle, skipping one extra phase at each turn.

    Input structure:
    - The top row contains one contiguous palette segment made of distinct colors; its left-to-right order defines the repeating cycle.
    - Below that row there is exactly one 4-connected, one-cell-thick, non-branching gray path.
    - Exactly one endpoint of the path is already colored with one of the palette colors.

    Rule:
    - Keep the palette row unchanged and walk along the path starting from the colored endpoint.
    - Color the start cell with its existing color, then advance one palette step for each next path cell.
    - If the newly reached path cell is a corner where the walking direction changes, advance two palette steps instead of one for that cell.

    Key insight:
    - The path is colored by distance along a single ordered walk, but each corner consumes an extra palette phase, so the color pattern changes only at turn cells.
    """
    return _apply_rule(_parse_instance(np.asarray(grid, dtype=int)))


def _valid_next(
    shape: tuple[int, int],
    occupied: set[tuple[int, int]],
    tail: tuple[int, int],
    candidate: tuple[int, int],
) -> bool:
    r, c = candidate
    if not _in_bounds(shape, r, c) or int(r) < 2 or candidate in occupied:
        return False
    occupied_neighbors = [
        nbr
        for nbr in ((r - 1, c), (r + 1, c), (r, c - 1), (r, c + 1))
        if nbr in occupied
    ]
    return occupied_neighbors == [tail] or (len(occupied_neighbors) == 1 and occupied_neighbors[0] == tail)


def _build_path(shape: tuple[int, int], target_len: int, target_corners: int) -> list[tuple[int, int]] | None:
    h, w = shape
    starts = [(r, c) for r in range(2, h) for c in range(w)]
    random.shuffle(starts)
    for start in starts[: min(len(starts), MAX_PATH_ATTEMPTS)]:
        occupied = {start}
        path = [start]

        def dfs(last_dir: tuple[int, int] | None, corners: int) -> bool:
            if len(path) == target_len:
                return corners == target_corners

            tail = path[-1]
            dirs = list(DIR4)
            random.shuffle(dirs)
            if last_dir is not None:
                turns = [d for d in dirs if d != last_dir]
                straights = [d for d in dirs if d == last_dir]
                need_more_turns = target_corners - corners
                if need_more_turns > 0 and random.random() < 0.7:
                    dirs = turns + straights
                elif need_more_turns <= 0:
                    dirs = straights + turns
                else:
                    dirs = straights + turns if random.random() < 0.5 else turns + straights

            for direction in dirs:
                cand = (tail[0] + direction[0], tail[1] + direction[1])
                if not _valid_next(shape, occupied, tail, cand):
                    continue
                new_corners = corners + int(last_dir is not None and direction != last_dir)
                if new_corners > target_corners:
                    continue
                remaining_cells = target_len - len(path) - 1
                if new_corners + remaining_cells < target_corners:
                    continue
                occupied.add(cand)
                path.append(cand)
                if dfs(direction, new_corners):
                    return True
                path.pop()
                occupied.remove(cand)
            return False

        if dfs(None, 0):
            return path.copy()
    return None


def _render_input(
    shape: tuple[int, int],
    bg: int,
    palette: list[int],
    palette_col: int,
    path: list[tuple[int, int]],
    start_index: int,
) -> np.ndarray:
    grid = np.full(shape, int(bg), dtype=int)
    grid[0, palette_col : palette_col + len(palette)] = np.asarray(palette, dtype=int)
    for r, c in path[1:]:
        grid[r, c] = GREY
    sr, sc = path[0]
    grid[sr, sc] = int(palette[start_index])
    return grid


def _semantic_ok(grid: np.ndarray) -> bool:
    try:
        parsed = _parse_instance(grid)
    except AssertionError:
        return False

    path_len = len(parsed["path"])
    corner_count = sum(parsed["corners"])
    if path_len < len(parsed["palette"]) + 2:
        return False

    correct = _path_colors(parsed["palette"], parsed["start_index"], parsed["corners"], mode="at_corner")
    if corner_count == 0:
        return path_len >= len(parsed["palette"]) + 3

    no_skip = _path_colors(parsed["palette"], parsed["start_index"], parsed["corners"], mode="never")
    after_corner = _path_colors(parsed["palette"], parsed["start_index"], parsed["corners"], mode="after_corner")
    if sum(a != b for a, b in zip(correct, no_skip)) < 2:
        return False
    if sum(a != b for a, b in zip(correct, after_corner)) < 2:
        return False

    first_corner = next((i for i, is_corner in enumerate(parsed["corners"]) if is_corner), None)
    if first_corner is None or path_len - first_corner < 3:
        return False
    if parsed["start_index"] != 0:
        start_zero = _path_colors(parsed["palette"], 0, parsed["corners"], mode="at_corner")
        if start_zero == correct:
            return False
    return True


def _gen_attempt() -> np.ndarray | None:
    h = random.randint(*GRID_H_RANGE)
    w = random.randint(*GRID_W_RANGE)
    bg = random.choice([c for c in ARC_COLORS if c != GREY])

    palette_len = random.randint(*PALETTE_LEN_RANGE)
    usable_colors = [c for c in ARC_COLORS if c not in {bg, GREY}]
    if len(usable_colors) < palette_len:
        return None
    palette = random.sample(usable_colors, palette_len)
    palette_col = random.randint(0, w - palette_len)

    base_min_len = max(7, palette_len + 3)
    base_max_len = min(18, max(base_min_len, (h - 2) * w // 3))
    if base_max_len < base_min_len:
        return None

    turn_mode = random.random() >= STRAIGHT_PROBABILITY
    if turn_mode:
        max_corners = min(5, base_max_len - 4)
        if max_corners < 1:
            return None
        target_corners = random.choices(
            list(range(1, max_corners + 1)),
            weights=[3 if k in {2, 3} else 2 for k in range(1, max_corners + 1)],
            k=1,
        )[0]
        min_len = max(base_min_len, target_corners + 5)
    else:
        target_corners = 0
        min_len = base_min_len
    if base_max_len < min_len:
        return None

    target_len = random.randint(min_len, base_max_len)
    path = _build_path((h, w), target_len, target_corners)
    if path is None:
        return None

    start_choices = list(range(palette_len))
    if palette_len > 1 and random.random() < 0.7:
        start_choices = [idx for idx in range(palette_len) if idx != 0]
    start_index = random.choice(start_choices)

    grid = _render_input((h, w), bg, palette, palette_col, path, start_index)
    if not _semantic_ok(grid):
        return None

    try:
        out = main(grid)
    except AssertionError:
        return None
    if np.array_equal(out, grid):
        return None
    return grid


def generate_input() -> np.ndarray:
    for _ in range(MAX_GEN_ATTEMPTS):
        grid = _gen_attempt()
        if grid is not None:
            return grid
    raise RuntimeError("failed to generate a valid corner-phase ARC input")
