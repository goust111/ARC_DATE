from __future__ import annotations

import random
from collections import deque

import numpy as np


BLACK = 0
GREY = 5
DIR4 = [(-1, 0), (1, 0), (0, -1), (0, 1)]
MAX_GENERATION_ATTEMPTS = 500


def _neighbors4(r: int, c: int):
    for dr, dc in DIR4:
        yield r + dr, c + dc


def _find_separator_col(grid: np.ndarray) -> int:
    cols = [c for c in range(grid.shape[1]) if np.all(grid[:, c] == GREY)]
    valid = [c for c in cols if 0 < c < grid.shape[1] - 1]
    assert len(valid) == 1, "expected one interior solid grey separator column"
    return int(valid[0])


def _path_cells(panel: np.ndarray) -> set[tuple[int, int]]:
    return {
        (int(r), int(c))
        for r, c in np.argwhere((panel != BLACK) & (panel != GREY))
    }


def _trace_simple_path(panel: np.ndarray) -> list[tuple[int, int]]:
    cells = _path_cells(panel)
    assert cells, "expected a non-empty snake path"

    visited = set()
    q = deque([next(iter(cells))])
    while q:
        cur = q.popleft()
        if cur in visited:
            continue
        visited.add(cur)
        r, c = cur
        for nr, nc in _neighbors4(r, c):
            if (nr, nc) in cells and (nr, nc) not in visited:
                q.append((nr, nc))
    assert visited == cells, "path pixels must form one 4-connected component"

    degree = {}
    for r, c in cells:
        degree[(r, c)] = sum((nr, nc) in cells for nr, nc in _neighbors4(r, c))
    endpoints = [cell for cell, deg in degree.items() if deg == 1]
    assert len(endpoints) == 2, "snake must have exactly two endpoints"
    assert all(deg <= 2 for deg in degree.values()), "snake must have no branches"

    start = min(endpoints)
    order: list[tuple[int, int]] = []
    prev: tuple[int, int] | None = None
    cur = start
    while True:
        order.append(cur)
        next_cells = [n for n in _neighbors4(*cur) if n in cells and n != prev]
        assert len(next_cells) <= 1, "simple path cannot branch during tracing"
        if not next_cells:
            break
        prev, cur = cur, next_cells[0]

    assert len(order) == len(cells), "trace must visit every path pixel exactly once"
    return order


def _segment_lengths(panel: np.ndarray) -> tuple[int, int]:
    order = _trace_simple_path(panel)
    colors = [int(panel[r, c]) for r, c in order]
    segments: list[int] = []
    current = colors[0]
    run = 1
    for color in colors[1:]:
        if color == current:
            run += 1
        else:
            segments.append(run)
            current = color
            run = 1
    segments.append(run)
    assert len(segments) == 2, "expected exactly two contiguous color segments"
    assert colors[0] != colors[-1], "the two path segments must use different colors"
    return int(segments[0]), int(segments[1])


def main(grid: np.ndarray) -> np.ndarray:
    """
    Decode two cyclic shift amounts from the colored snake path and roll the payload panel.

    Input structure:
    - A solid grey separator column splits the grid into a left control panel and a right payload panel.
    - The left panel contains one non-black, non-grey 4-connected snake with exactly two endpoints and no branches.
    - Traversed from the top-left-most endpoint, the snake consists of exactly two contiguous same-color runs.
    - The right panel is a full rectangular payload with no black, no grey, and no path colors.

    Rule:
    - Find the separator column and isolate the left and right panels.
    - Trace the left snake from the endpoint closer to the top-left corner and measure the two contiguous color-run lengths.
    - Let the first run length be V and the second run length be H.
    - Roll the right panel down by V and then right by H, both cyclically, and output only the rolled payload.

    Key insight:
    - The snake is not copied; its traversal order encodes the vertical and horizontal cyclic shifts applied to the payload.
    """
    g = np.asarray(grid, dtype=int)
    assert g.ndim == 2, "expected a 2D grid"

    sep_col = _find_separator_col(g)
    left = g[:, :sep_col]
    right = g[:, sep_col + 1 :]
    assert left.size > 0 and right.size > 0, "separator must split two non-empty panels"

    v_shift, h_shift = _segment_lengths(left)
    assert v_shift < right.shape[0], "vertical shift must fit payload height"
    assert h_shift < right.shape[1], "horizontal shift must fit payload width"

    path_colors = {int(left[r, c]) for r, c in _trace_simple_path(left)}
    assert not np.any(right == BLACK), "payload must fill the entire right panel"
    assert not np.any(right == GREY), "payload must not contain grey"
    assert not any(color in set(map(int, right.ravel())) for color in path_colors), (
        "payload must not reuse path colors"
    )

    return np.roll(right, shift=(v_shift, h_shift), axis=(0, 1))


def _turn_count(path: list[tuple[int, int]]) -> int:
    turns = 0
    for i in range(2, len(path)):
        dr1 = path[i - 1][0] - path[i - 2][0]
        dc1 = path[i - 1][1] - path[i - 2][1]
        dr2 = path[i][0] - path[i - 1][0]
        dc2 = path[i][1] - path[i - 1][1]
        if (dr1, dc1) != (dr2, dc2):
            turns += 1
    return turns


def _candidate_steps(
    current: tuple[int, int],
    occupied: set[tuple[int, int]],
    height: int,
    width: int,
) -> list[tuple[int, int]]:
    candidates = []
    for nr, nc in _neighbors4(*current):
        if not (1 <= nr < height - 1 and 1 <= nc < width - 1):
            continue
        if (nr, nc) in occupied:
            continue
        adjacency = sum((ar, ac) in occupied for ar, ac in _neighbors4(nr, nc))
        if adjacency == 1:
            candidates.append((nr, nc))
    random.shuffle(candidates)
    return candidates


def _search_path(
    path: list[tuple[int, int]],
    occupied: set[tuple[int, int]],
    target_len: int,
    min_turns: int,
    height: int,
    width: int,
) -> list[tuple[int, int]] | None:
    if len(path) == target_len:
        if min(path[0], path[-1]) != path[0]:
            return None
        if _turn_count(path) < min_turns:
            return None
        return path.copy()

    prev_dir = None
    if len(path) >= 2:
        prev_dir = (path[-1][0] - path[-2][0], path[-1][1] - path[-2][1])

    remaining = target_len - len(path)
    if _turn_count(path) + max(0, remaining - 1) < min_turns:
        return None

    for nxt in _candidate_steps(path[-1], occupied, height, width):
        new_dir = (nxt[0] - path[-1][0], nxt[1] - path[-1][1])
        if prev_dir is not None and random.random() < 0.35 and new_dir == prev_dir:
            # Mild anti-straight bias so the path family covers bendy cases more often.
            pass
        path.append(nxt)
        occupied.add(nxt)
        result = _search_path(path, occupied, target_len, min_turns, height, width)
        if result is not None:
            return result
        occupied.remove(nxt)
        path.pop()
    return None


def _build_random_path(height: int, width: int, length: int, min_turns: int) -> list[tuple[int, int]] | None:
    row_choices = list(range(1, max(2, height // 2 + 1)))
    col_choices = list(range(1, max(2, width // 2 + 1)))
    starts = [(r, c) for r in row_choices for c in col_choices]
    random.shuffle(starts)
    for start in starts:
        path = [start]
        result = _search_path(path, {start}, length, min_turns, height, width)
        if result is not None:
            return result
    return None


def _choose_payload_palette(forbidden: set[int]) -> list[int]:
    available = [color for color in range(1, 10) if color not in forbidden and color != GREY]
    palette_size = random.randint(3, min(5, len(available)))
    return random.sample(available, palette_size)


def _make_payload(height: int, width: int, palette: list[int], v_shift: int, h_shift: int) -> np.ndarray | None:
    area = height * width
    while True:
        flat = [random.choice(palette) for _ in range(area)]
        for idx, color in enumerate(palette):
            flat[idx] = color
        random.shuffle(flat)
        payload = np.array(flat, dtype=int).reshape(height, width)

        rolled = np.roll(payload, shift=(v_shift, h_shift), axis=(0, 1))
        row_only = np.roll(payload, shift=(v_shift, 0), axis=(0, 1))
        col_only = np.roll(payload, shift=(0, h_shift), axis=(0, 1))
        swapped = np.roll(payload, shift=(h_shift % height, v_shift % width), axis=(0, 1))

        if np.array_equal(rolled, payload):
            continue
        if np.array_equal(rolled, row_only):
            continue
        if np.array_equal(rolled, col_only):
            continue
        if np.array_equal(rolled, swapped):
            continue
        if len(set(map(int, payload.ravel()))) < 3:
            continue
        return payload


def _render_left_panel(
    height: int,
    width: int,
    path: list[tuple[int, int]],
    first_len: int,
    colors: tuple[int, int],
) -> np.ndarray:
    panel = np.full((height, width), BLACK, dtype=int)
    first_color, second_color = colors
    for idx, (r, c) in enumerate(path):
        panel[r, c] = first_color if idx < first_len else second_color
    return panel


def _validate_instance(
    grid: np.ndarray,
    expected_v: int,
    expected_h: int,
    expected_path: list[tuple[int, int]],
    expected_payload: np.ndarray,
) -> bool:
    try:
        sep_col = _find_separator_col(grid)
        left = grid[:, :sep_col]
        right = grid[:, sep_col + 1 :]
        order = _trace_simple_path(left)
        if order != expected_path:
            return False
        if _segment_lengths(left) != (expected_v, expected_h):
            return False
        if not np.array_equal(right, expected_payload):
            return False

        out = main(grid)
        target = np.roll(expected_payload, shift=(expected_v, expected_h), axis=(0, 1))
        if not np.array_equal(out, target):
            return False

        if np.array_equal(out, expected_payload):
            return False
        if np.array_equal(out, np.roll(expected_payload, shift=(expected_v, 0), axis=(0, 1))):
            return False
        if np.array_equal(out, np.roll(expected_payload, shift=(0, expected_h), axis=(0, 1))):
            return False
        if np.array_equal(
            out,
            np.roll(
                expected_payload,
                shift=(expected_h % expected_payload.shape[0], expected_v % expected_payload.shape[1]),
                axis=(0, 1),
            ),
        ):
            return False
    except Exception:
        return False
    return True


def _gen_attempt() -> np.ndarray | None:
    grid_h = random.randint(8, 13)
    right_w = random.randint(4, 9)
    v_shift = random.randint(1, min(8, grid_h - 2))
    h_shift = random.randint(1, right_w - 1)
    if v_shift == h_shift and right_w > 2:
        h_shift = (h_shift % (right_w - 1)) + 1
        if h_shift == v_shift:
            return None

    total_len = v_shift + h_shift
    left_w = random.randint(6, 11)
    if total_len > (grid_h - 2) * (left_w - 2):
        return None

    max_turns = min(5, total_len - 2)
    min_turns = random.randint(0, max_turns) if max_turns > 0 else 0
    path = _build_random_path(grid_h, left_w, total_len, min_turns)
    if path is None:
        return None

    path_colors = random.sample([c for c in range(1, 10) if c != GREY], 2)
    left = _render_left_panel(grid_h, left_w, path, v_shift, (path_colors[0], path_colors[1]))
    payload_palette = _choose_payload_palette(set(path_colors) | {BLACK, GREY})
    payload = _make_payload(grid_h, right_w, payload_palette, v_shift, h_shift)
    if payload is None:
        return None

    grid = np.full((grid_h, left_w + 1 + right_w), BLACK, dtype=int)
    grid[:, :left_w] = left
    grid[:, left_w] = GREY
    grid[:, left_w + 1 :] = payload

    if not _validate_instance(grid, v_shift, h_shift, path, payload):
        return None
    return grid


def generate_input() -> np.ndarray:
    for _ in range(MAX_GENERATION_ATTEMPTS):
        grid = _gen_attempt()
        if grid is not None:
            return grid
    raise RuntimeError("failed to generate a valid snake-roll ARC input")
