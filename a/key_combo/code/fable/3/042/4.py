from __future__ import annotations

from collections import Counter
import random

import numpy as np


BG = 0
PATH = 5
BEAD_COLORS = [1, 2, 3, 4, 6, 7, 8, 9]
GEN_RETRY_LIMIT = 500


def _neighbors4(r: int, c: int):
    yield r - 1, c
    yield r + 1, c
    yield r, c - 1
    yield r, c + 1


def _path_order(g: np.ndarray) -> list[tuple[int, int]]:
    mask = g != BG
    coords = [tuple(map(int, rc)) for rc in np.argwhere(mask)]
    assert coords, "expected a visible path"

    coord_set = set(coords)
    degrees: dict[tuple[int, int], int] = {}
    for r, c in coords:
        degrees[(r, c)] = sum((nr, nc) in coord_set for nr, nc in _neighbors4(r, c))

    endpoints = sorted([pos for pos, degree in degrees.items() if degree == 1])
    assert len(endpoints) == 2, "expected exactly two path endpoints"
    assert all(degree in (1, 2) for degree in degrees.values()), "path must not branch or touch itself"

    order = [endpoints[0]]
    prev: tuple[int, int] | None = None
    cur = endpoints[0]
    while cur != endpoints[1]:
        choices = [
            (nr, nc)
            for nr, nc in _neighbors4(*cur)
            if (nr, nc) in coord_set and (nr, nc) != prev
        ]
        assert len(choices) == 1, "path must have a unique next cell"
        prev, cur = cur, choices[0]
        order.append(cur)

    assert len(order) == len(coord_set), "path traversal must visit every path cell"
    return order


def _bead_indices(g: np.ndarray, path: list[tuple[int, int]]) -> list[int]:
    beads = [
        i
        for i, (r, c) in enumerate(path)
        if int(g[r, c]) not in (BG, PATH)
    ]
    assert beads and len(beads) % 2 == 1, "expected an odd positive number of beads"
    return beads


def main(grid: np.ndarray) -> np.ndarray:
    """Mirror the beads on each side of the hidden balanced bead along a snake path.

    Input structure
    - A black background contains one grey, one-cell-wide, 4-connected,
      non-branching snake path whose two endpoints each have one path neighbor.
    - Some path cells are occupied by colored beads; the number of beads is odd
      and bead colors may repeat.
    - A globally unique bead color, when present, is only a decoy and does not
      identify the axis bead.

    Rule
    - Linearize the snake from one endpoint to the other and sort the beads by
      this path order.
    - The middle bead in that order is the axis bead because it has the same
      number of beads on both sides; keep it fixed.
    - On each side separately, reverse bead positions within the side's full
      path-cell sequence from the axis neighbor to the endpoint, carrying each
      bead's color to its mirrored arclength position and restoring vacated
      bead cells to grey path.

    Key insight
    - The transformation is not a visual left-right reflection and not a color
      cue: the beads are mirrored in the one-dimensional arclength coordinate
      obtained by unfolding the snake path.
    """
    g = np.asarray(grid, dtype=int)
    assert g.ndim == 2, "expected a 2D grid"

    path = _path_order(g)
    bead_indices = _bead_indices(g, path)
    axis_index = bead_indices[len(bead_indices) // 2]

    out = g.copy()
    for idx in bead_indices:
        r, c = path[idx]
        out[r, c] = PATH

    path_len = len(path)
    for idx in bead_indices:
        r, c = path[idx]
        color = int(g[r, c])
        if idx == axis_index:
            target = idx
        elif idx < axis_index:
            target = axis_index - 1 - idx
        else:
            target = path_len + axis_index - idx
        tr, tc = path[target]
        out[tr, tc] = color

    return out


# --------------- generator ---------------


def _coarse_walk(height: int, width: int, steps: int) -> list[tuple[int, int]] | None:
    start = (random.randrange(height), random.randrange(width))
    path = [start]
    used = {start}
    for _ in range(steps):
        r, c = path[-1]
        choices = []
        for dr, dc in random.sample([(-1, 0), (1, 0), (0, -1), (0, 1)], 4):
            nr, nc = r + dr, c + dc
            if 0 <= nr < height and 0 <= nc < width and (nr, nc) not in used:
                choices.append((nr, nc))
        if not choices:
            return None
        nxt = random.choice(choices)
        path.append(nxt)
        used.add(nxt)
    return path


def _turn_count(vertices: list[tuple[int, int]]) -> int:
    turns = 0
    for a, b, c in zip(vertices, vertices[1:], vertices[2:]):
        d1 = (b[0] - a[0], b[1] - a[1])
        d2 = (c[0] - b[0], c[1] - b[1])
        if d1 != d2:
            turns += 1
    return turns


def _expand_vertices(vertices: list[tuple[int, int]]) -> list[tuple[int, int]]:
    out = [(2 * vertices[0][0], 2 * vertices[0][1])]
    for (r0, c0), (r1, c1) in zip(vertices, vertices[1:]):
        dr, dc = r1 - r0, c1 - c0
        last_r, last_c = out[-1]
        out.append((last_r + dr, last_c + dc))
        out.append((last_r + 2 * dr, last_c + 2 * dc))
    min_r = min(r for r, _c in out)
    min_c = min(c for _r, c in out)
    return [(r - min_r, c - min_c) for r, c in out]


def _gen_path() -> list[tuple[int, int]] | None:
    coarse_h = random.randint(5, 12)
    coarse_w = random.randint(5, 12)
    max_steps = min(coarse_h * coarse_w - 1, 34)
    min_steps = min(12, max_steps)
    if min_steps > max_steps:
        return None

    steps = random.randint(min_steps, max_steps)
    vertices = _coarse_walk(coarse_h, coarse_w, steps)
    if vertices is None or _turn_count(vertices) < 4:
        return None

    path = _expand_vertices(vertices)
    rows = [r for r, _c in path]
    cols = [c for _r, c in path]
    if max(rows) >= 28 or max(cols) >= 28:
        return None
    return path


def _mirror_targets(indices: list[int], axis: int, path_len: int) -> set[int]:
    out = set()
    for idx in indices:
        if idx < axis:
            out.add(axis - 1 - idx)
        elif idx > axis:
            out.add(path_len + axis - idx)
    return out


def _side_sets_change(left: list[int], right: list[int], axis: int, path_len: int) -> bool:
    left_targets = _mirror_targets(left, axis, path_len)
    right_targets = _mirror_targets(right, axis, path_len)
    return set(left) != left_targets and set(right) != right_targets


def _color_counts(total: int, unique_decoy: bool) -> list[int]:
    if unique_decoy:
        return [1] + [2] * ((total - 1) // 2)
    return [3] + [2] * ((total - 3) // 2)


def _assign_colors(
    bead_positions: list[int], axis_index: int, unique_decoy: bool
) -> dict[int, int] | None:
    total = len(bead_positions)
    counts = _color_counts(total, unique_decoy)
    if len(counts) > len(BEAD_COLORS):
        return None

    colors = random.sample(BEAD_COLORS, len(counts))
    assignment: dict[int, int] = {}

    if unique_decoy:
        decoy_color = colors[0]
        repeated_colors = colors[1:]
        decoy_candidates = [idx for idx in bead_positions if idx != axis_index]
        decoy_pos = random.choice(decoy_candidates)
        axis_color = random.choice(repeated_colors)
        assignment[decoy_pos] = decoy_color
        assignment[axis_index] = axis_color

        remaining_colors = []
        for color in repeated_colors:
            count = 2 - (1 if color == axis_color else 0)
            remaining_colors.extend([color] * count)
    else:
        repeated_colors = colors
        axis_color = random.choice(repeated_colors)
        assignment[axis_index] = axis_color

        remaining_colors = []
        for color, count in zip(repeated_colors, counts):
            count_left = count - (1 if color == axis_color else 0)
            remaining_colors.extend([color] * count_left)

    remaining_positions = [idx for idx in bead_positions if idx not in assignment]
    if len(remaining_positions) != len(remaining_colors):
        return None
    random.shuffle(remaining_colors)
    for idx, color in zip(remaining_positions, remaining_colors):
        assignment[idx] = int(color)
    return assignment


def _place_path_on_canvas(path: list[tuple[int, int]]) -> tuple[np.ndarray, list[tuple[int, int]]]:
    path_h = max(r for r, _c in path) + 1
    path_w = max(c for _r, c in path) + 1

    min_h = path_h + 2
    min_w = path_w + 2
    height = random.randint(min_h, 30)
    width = random.randint(min_w, 30)
    row_offset = random.randint(1, height - path_h - 1)
    col_offset = random.randint(1, width - path_w - 1)

    shifted = [(r + row_offset, c + col_offset) for r, c in path]
    grid = np.full((height, width), BG, dtype=int)
    for r, c in shifted:
        grid[r, c] = PATH
    return grid, shifted


def _gen_attempt() -> np.ndarray | None:
    path = _gen_path()
    if path is None:
        return None
    grid, shifted_path = _place_path_on_canvas(path)
    path_len = len(shifted_path)

    max_beads = min(11, path_len // 2 * 2 - 1)
    bead_options = [n for n in (5, 7, 9, 11) if n <= max_beads]
    if not bead_options:
        return None
    total_beads = random.choice(bead_options)
    k = total_beads // 2

    candidate_axes = [
        idx
        for idx in range(k + 3, path_len - k - 3)
        if abs(idx - (path_len - 1 - idx)) >= 3
    ]
    if not candidate_axes:
        candidate_axes = list(range(k + 3, path_len - k - 3))
    if not candidate_axes:
        return None
    axis_index = random.choice(candidate_axes)

    left = sorted(random.sample(range(axis_index), k))
    right = sorted(random.sample(range(axis_index + 1, path_len), k))
    if not _side_sets_change(left, right, axis_index, path_len):
        return None

    bead_positions = left + [axis_index] + right
    unique_decoy = total_beads >= 7 and random.random() < 0.45
    colors = _assign_colors(bead_positions, axis_index, unique_decoy)
    if colors is None:
        return None

    for idx, color in colors.items():
        r, c = shifted_path[idx]
        grid[r, c] = color

    try:
        ordered_path = _path_order(grid)
        bead_indices = _bead_indices(grid, ordered_path)
        out = main(grid)
    except AssertionError:
        return None

    found_axis = bead_indices[len(bead_indices) // 2]
    if ordered_path[found_axis] != shifted_path[axis_index]:
        return None
    if np.array_equal(out, grid):
        return None

    counts = Counter(int(grid[r, c]) for idx in bead_positions for r, c in [shifted_path[idx]])
    unique_colors = [color for color, count in counts.items() if count == 1]
    axis_color = int(grid[shifted_path[axis_index]])
    if unique_decoy:
        if len(unique_colors) != 1 or unique_colors[0] == axis_color:
            return None
    elif unique_colors:
        return None

    left_coords = [shifted_path[i] for i in range(axis_index)]
    right_coords = [shifted_path[i] for i in range(axis_index + 1, path_len)]
    if not any(grid[pos] != out[pos] for pos in left_coords):
        return None
    if not any(grid[pos] != out[pos] for pos in right_coords):
        return None

    return grid


def generate_input() -> np.ndarray:
    for _ in range(GEN_RETRY_LIMIT):
        grid = _gen_attempt()
        if grid is not None:
            return grid
    raise RuntimeError("failed to generate a valid bead-chain mirror input")
