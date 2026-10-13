import random
from collections import Counter, deque

import numpy as np


BG = 0
FRAME = 5
MAX_GENERATION_ATTEMPTS = 500
PATH_COLOR_POOL = [1, 2, 3, 4, 6, 7, 8, 9]


def _neighbors4(r: int, c: int) -> tuple[tuple[int, int], ...]:
    return ((r - 1, c), (r + 1, c), (r, c - 1), (r, c + 1))


def _most_common_color(grid: np.ndarray) -> int:
    values, counts = np.unique(grid, return_counts=True)
    return int(values[np.argmax(counts)])


def _components(mask: np.ndarray) -> list[list[tuple[int, int]]]:
    h, w = mask.shape
    seen = np.zeros((h, w), dtype=bool)
    comps: list[list[tuple[int, int]]] = []
    for r in range(h):
        for c in range(w):
            if not mask[r, c] or seen[r, c]:
                continue
            comp: list[tuple[int, int]] = []
            q = deque([(r, c)])
            seen[r, c] = True
            while q:
                cr, cc = q.popleft()
                comp.append((cr, cc))
                for nr, nc in _neighbors4(cr, cc):
                    if 0 <= nr < h and 0 <= nc < w and mask[nr, nc] and not seen[nr, nc]:
                        seen[nr, nc] = True
                        q.append((nr, nc))
            comps.append(comp)
    return comps


def _find_frames(grid: np.ndarray, bg: int) -> list[dict]:
    h, w = grid.shape
    frames: list[dict] = []
    for color in sorted(int(v) for v in np.unique(grid) if int(v) != int(bg)):
        for comp in _components(grid == color):
            rows = [r for r, _ in comp]
            cols = [c for _, c in comp]
            r0, r1 = min(rows), max(rows)
            c0, c1 = min(cols), max(cols)
            if r1 - r0 != 2 or c1 - c0 < 2:
                continue

            expected = {
                (r0, c) for c in range(c0, c1 + 1)
            } | {
                (r1, c) for c in range(c0, c1 + 1)
            } | {
                (r0 + 1, c0), (r0 + 1, c1)
            }
            comp_set = set(comp)
            if comp_set != expected:
                continue
            if not np.all(grid[r0 + 1, c0 + 1:c1] == bg):
                continue
            if r1 + 1 >= h:
                continue
            key_cols = [c for c in range(c0, c1 + 1) if int(grid[r1 + 1, c]) != int(bg)]
            if len(key_cols) != 1:
                continue
            key_c = key_cols[0]
            path_color = int(grid[r1 + 1, key_c])
            if path_color == color:
                continue
            frames.append(
                {
                    "frame_color": int(color),
                    "r0": int(r0),
                    "c0": int(c0),
                    "r1": int(r1),
                    "c1": int(c1),
                    "key_pos": (int(r1 + 1), int(key_c)),
                    "path_color": int(path_color),
                    "width": int(c1 - c0 - 1),
                    "cells": comp_set,
                }
            )
    return frames


def _trace_marker_sequence(
    grid: np.ndarray,
    path_color: int,
    path_colors: set[int],
    frame_colors: set[int],
    frame_cells: set[tuple[int, int]],
    key_positions: set[tuple[int, int]],
    bg: int,
) -> list[int]:
    h, w = grid.shape
    base_cells = {
        (r, c)
        for r in range(h)
        for c in range(w)
        if int(grid[r, c]) == int(path_color) and (r, c) not in key_positions
    }
    assert base_cells, "path color must appear in the main area"

    comp = set(base_cells)
    q = deque(base_cells)
    while q:
        r, c = q.popleft()
        for nr, nc in _neighbors4(r, c):
            if not (0 <= nr < h and 0 <= nc < w):
                continue
            if (nr, nc) in comp or (nr, nc) in frame_cells or (nr, nc) in key_positions:
                continue
            value = int(grid[nr, nc])
            if value == int(path_color):
                comp.add((nr, nc))
                q.append((nr, nc))
                continue
            if value == int(bg) or value in frame_colors or value in path_colors:
                continue
            comp.add((nr, nc))
            q.append((nr, nc))

    neighbors = {
        cell: [n for n in _neighbors4(*cell) if n in comp]
        for cell in comp
    }
    endpoints = [cell for cell, nbrs in neighbors.items() if len(nbrs) == 1]
    assert len(endpoints) == 2, "reconstructed path must have exactly two endpoints"
    assert all(len(nbrs) in (1, 2) for nbrs in neighbors.values()), "path must be simple"

    start = min(endpoints, key=lambda p: (p[0], p[1]))
    order: list[tuple[int, int]] = []
    prev: tuple[int, int] | None = None
    cur = start
    while True:
        order.append(cur)
        nxt = [n for n in neighbors[cur] if n != prev]
        if not nxt:
            break
        assert len(nxt) == 1, "path traversal must be unambiguous"
        prev, cur = cur, nxt[0]
    assert len(order) == len(comp), "traversal must cover the full path"
    return [int(grid[r, c]) for r, c in order if int(grid[r, c]) != int(path_color)]


def main(grid: np.ndarray) -> np.ndarray:
    """
    Reconstruct each colored path and write its encountered sign colors into the matching bottom frame.

    Input structure:
    - The background is the dominant color.
    - Several logical paths are encoded by one-pixel-wide horizontal and vertical fragments of distinct path colors.
    - Some path cells are overwritten by sign pixels of other colors, which break a path into same-color fragments.
    - A row of empty grey rectangular frames sits at the bottom; each frame has one colored key pixel directly below it matching one path color.

    Rule:
    - Detect the bottom frames and use the key pixel below each one to determine which path color it belongs to.
    - For each path color, reconstruct the full path by joining its same-color fragments through the non-background sign pixels that connect them.
    - The directed start of the path is the lexicographically smaller endpoint among the two degree-1 endpoints; traverse the path from that start and record each sign color encountered in order.
    - Fill the interior of the matching bottom frame from left to right with that recorded sign-color sequence, leaving all other cells unchanged.

    Key insight:
    - The same-color pixels alone are incomplete; the real path is the simple 4-connected chain formed by path-color cells plus the overwritten sign cells, while the bottom key pixel provides the only reliable path-to-frame correspondence.
    """
    g = np.asarray(grid, dtype=int)
    bg = _most_common_color(g)

    frames = _find_frames(g, bg)
    assert frames, "at least one bottom frame must be present"

    frame_cells = set().union(*(frame["cells"] for frame in frames))
    key_positions = {frame["key_pos"] for frame in frames}
    path_colors = {int(frame["path_color"]) for frame in frames}
    frame_colors = {int(frame["frame_color"]) for frame in frames}

    out = g.copy()
    for frame in frames:
        sequence = _trace_marker_sequence(
            g,
            int(frame["path_color"]),
            path_colors,
            frame_colors,
            frame_cells,
            key_positions,
            bg,
        )
        assert len(sequence) == int(frame["width"]), "frame width must match sign count"
        out[int(frame["r0"]) + 1, int(frame["c0"]) + 1:int(frame["c1"])] = np.array(sequence, dtype=int)

    return out


def _can_add_cell(
    cell: tuple[int, int],
    prev: tuple[int, int],
    used: set[tuple[int, int]],
    height: int,
    width: int,
) -> bool:
    r, c = cell
    if not (0 <= r < height and 0 <= c < width):
        return False
    if cell in used:
        return False
    for nr, nc in _neighbors4(r, c):
        if (nr, nc) in used and (nr, nc) != prev:
            return False
    return True


def _max_extension(
    start: tuple[int, int],
    direction: tuple[int, int],
    used: set[tuple[int, int]],
    height: int,
    width: int,
) -> list[tuple[int, int]]:
    dr, dc = direction
    prev = start
    r, c = start
    run: list[tuple[int, int]] = []
    while True:
        nr, nc = r + dr, c + dc
        cell = (nr, nc)
        if not _can_add_cell(cell, prev, used | set(run), height, width):
            break
        run.append(cell)
        prev = cell
        r, c = nr, nc
    return run


def _has_future_turn(
    end: tuple[int, int],
    next_horizontal: bool,
    temp_used: set[tuple[int, int]],
    height: int,
    width: int,
) -> bool:
    dirs = ((0, 1), (0, -1)) if next_horizontal else ((1, 0), (-1, 0))
    return any(len(_max_extension(end, direction, temp_used, height, width)) >= 2 for direction in dirs)


def _build_polyline(height: int, width: int, rng: random.Random) -> list[tuple[int, int]] | None:
    directions = {
        True: ((0, 1), (0, -1)),
        False: ((1, 0), (-1, 0)),
    }
    for _ in range(300):
        segment_count = rng.randint(1, 4)
        horizontal = bool(rng.randint(0, 1))
        start = (rng.randrange(height), rng.randrange(width))
        path = [start]
        used = {start}
        ok = True

        for seg_idx in range(segment_count):
            need_turn = seg_idx < segment_count - 1
            candidates: list[tuple[int, int, int]] = []
            for dr, dc in directions[horizontal]:
                run = _max_extension(path[-1], (dr, dc), used, height, width)
                for length in range(2, len(run) + 1):
                    segment_cells = run[:length]
                    if need_turn:
                        temp_used = used | set(segment_cells)
                        if not _has_future_turn(segment_cells[-1], not horizontal, temp_used, height, width):
                            continue
                    candidates.append((dr, dc, length))
            if not candidates:
                ok = False
                break

            rng.shuffle(candidates)
            candidates.sort(key=lambda item: item[2], reverse=True)
            dr, dc, length = candidates[rng.randrange(min(4, len(candidates)))]
            prev = path[-1]
            for _ in range(length):
                nxt = (prev[0] + dr, prev[1] + dc)
                path.append(nxt)
                used.add(nxt)
                prev = nxt
            if need_turn:
                horizontal = not horizontal

        if not ok or len(path) < 4:
            continue
        return path
    return None


def _pick_marker_indices(path: list[tuple[int, int]], rng: random.Random) -> list[int] | None:
    index_by_cell = {cell: i for i, cell in enumerate(path)}
    mandatory = {
        index_by_cell[path[i]]
        for i in range(1, len(path) - 1)
        if path[i - 1][0] != path[i + 1][0] and path[i - 1][1] != path[i + 1][1]
    }
    if len(mandatory) > 4:
        return None

    chosen = set(mandatory)
    candidates = list(range(1, len(path) - 1))
    rng.shuffle(candidates)
    filtered = []
    for idx in candidates:
        if idx in chosen:
            continue
        if any(abs(idx - other) == 1 for other in chosen):
            continue
        filtered.append(idx)

    if not chosen and not filtered:
        return None

    max_extra = 4 - len(chosen)
    if max_extra < 0:
        return None
    min_extra = 1 if not chosen else 0
    max_feasible = 0
    scratch = set(chosen)
    for idx in filtered:
        if any(abs(idx - other) == 1 for other in scratch):
            continue
        scratch.add(idx)
        max_feasible += 1
    if max_feasible < min_extra:
        return None

    extra_target = rng.randint(min_extra, min(max_extra, max_feasible))
    extras: list[int] = []
    for idx in filtered:
        if len(extras) == extra_target:
            break
        if any(abs(idx - other) == 1 for other in chosen):
            continue
        extras.append(idx)
        chosen.add(idx)
    if len(extras) != extra_target:
        return None
    return sorted(chosen)


def _distribute_extra(total: int, slots: int, rng: random.Random) -> list[int]:
    extra = [0] * slots
    for _ in range(total):
        extra[rng.randrange(slots)] += 1
    return extra


def _layout_starts(widths: list[int], base_gap: int, total_width: int, rng: random.Random) -> list[int] | None:
    minimum = sum(widths) + base_gap * (len(widths) + 1)
    if minimum > total_width:
        return None
    bonuses = _distribute_extra(total_width - minimum, len(widths) + 1, rng)
    starts: list[int] = []
    x = base_gap + bonuses[0]
    for i, width in enumerate(widths):
        starts.append(x)
        x += width + base_gap + bonuses[i + 1]
    return starts


def _draw_frame(grid: np.ndarray, top: int, left: int, width: int, key_color: int, rng: random.Random) -> dict:
    grid[top, left:left + width] = FRAME
    grid[top + 2, left:left + width] = FRAME
    grid[top + 1, left] = FRAME
    grid[top + 1, left + width - 1] = FRAME
    key_col = rng.randint(left, left + width - 1)
    grid[top + 3, key_col] = int(key_color)
    return {"r0": top, "c0": left, "width": width, "key_col": key_col}


def _bg_is_dominant(grid: np.ndarray) -> bool:
    counts = Counter(map(int, grid.ravel()))
    bg_count = counts[int(BG)]
    return all(color == int(BG) or bg_count > count for color, count in counts.items())


def _make_attempt(rng: random.Random) -> np.ndarray | None:
    path_count = rng.randint(2, 4)
    strip_ranges = {
        2: (8, 11),
        3: (6, 8),
        4: (5, 6),
    }
    top_height = rng.randint(10, 16)
    usable_height = top_height - 2
    if usable_height < 4:
        return None

    colors = rng.sample(PATH_COLOR_POOL, len(PATH_COLOR_POOL))
    path_colors = colors[:path_count]
    marker_palette = colors[path_count:path_count + rng.randint(2, 3)]
    if len(marker_palette) < 2:
        return None

    strips = [rng.randint(*strip_ranges[path_count]) for _ in range(path_count)]
    path_specs: list[dict] = []
    for strip_width, path_color in zip(strips, path_colors):
        usable_width = strip_width - 2
        if usable_width < 3:
            return None
        local_path = _build_polyline(usable_height, usable_width, rng)
        if local_path is None:
            return None
        marker_indices = _pick_marker_indices(local_path, rng)
        if marker_indices is None:
            return None

        endpoints = (local_path[0], local_path[-1])
        ordered_cells = local_path if endpoints[0] <= endpoints[1] else list(reversed(local_path))
        marker_cells = {local_path[idx] for idx in marker_indices}
        ordered_marker_cells = [cell for cell in ordered_cells if cell in marker_cells]
        marker_sequence = [rng.choice(marker_palette) for _ in ordered_marker_cells]
        cell_to_marker = {
            cell: color for cell, color in zip(ordered_marker_cells, marker_sequence)
        }
        path_specs.append(
            {
                "path_color": int(path_color),
                "strip_width": int(strip_width),
                "path": local_path,
                "cell_to_marker": cell_to_marker,
                "marker_sequence": marker_sequence,
            }
        )

    frame_order = list(range(path_count))
    rng.shuffle(frame_order)
    frame_widths = [len(path_specs[i]["marker_sequence"]) + 2 for i in frame_order]

    top_gap = rng.randint(1, 2)
    bottom_gap = rng.randint(1, 2)
    top_min_width = sum(strips) + top_gap * (path_count + 1)
    bottom_min_width = sum(frame_widths) + bottom_gap * (path_count + 1)
    total_width = max(top_min_width, bottom_min_width) + rng.randint(0, 3)
    if total_width > 30:
        return None

    height = top_height + 4
    if height > 30:
        return None

    strip_starts = _layout_starts(strips, top_gap, total_width, rng)
    frame_starts = _layout_starts(frame_widths, bottom_gap, total_width, rng)
    if strip_starts is None or frame_starts is None:
        return None

    inp = np.full((height, total_width), BG, dtype=int)
    expected = inp.copy()

    for spec, strip_left in zip(path_specs, strip_starts):
        for local_r, local_c in spec["path"]:
            r = 1 + int(local_r)
            c = int(strip_left) + 1 + int(local_c)
            value = spec["cell_to_marker"].get((local_r, local_c), int(spec["path_color"]))
            inp[r, c] = int(value)
            expected[r, c] = int(value)

    frame_top = top_height
    for idx, left, width in zip(frame_order, frame_starts, frame_widths):
        spec = path_specs[idx]
        key_col = rng.randint(left, left + width - 1)
        for canvas in (inp, expected):
            canvas[frame_top, left:left + width] = FRAME
            canvas[frame_top + 2, left:left + width] = FRAME
            canvas[frame_top + 1, left] = FRAME
            canvas[frame_top + 1, left + width - 1] = FRAME
            canvas[frame_top + 3, key_col] = int(spec["path_color"])
        sequence = spec["marker_sequence"]
        expected[frame_top + 1, left + 1:left + 1 + len(sequence)] = np.array(sequence, dtype=int)

    if not _bg_is_dominant(inp):
        return None

    try:
        actual = main(inp)
    except Exception:
        return None
    if not np.array_equal(actual, expected):
        return None
    return inp


def generate_input() -> np.ndarray:
    rng = random.Random()
    for _ in range(MAX_GENERATION_ATTEMPTS):
        grid = _make_attempt(rng)
        if grid is not None:
            return grid
    raise RuntimeError("failed to generate a valid input")
