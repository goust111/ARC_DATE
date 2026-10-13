from __future__ import annotations

from collections import Counter
import random

import numpy as np


ARC_COLORS = list(range(10))
GEN_ATTEMPTS = 400


def _get_bg(grid: np.ndarray) -> int:
    return int(Counter(map(int, grid.flat)).most_common(1)[0][0])


def _dominant_fg_color(grid: np.ndarray, bg: int) -> int:
    counts = Counter(map(int, grid.flat))
    counts.pop(int(bg), None)
    assert counts, "expected at least one foreground color"
    return int(counts.most_common(1)[0][0])


def _neighbors4(r: int, c: int) -> list[tuple[int, int]]:
    return [(r - 1, c), (r + 1, c), (r, c - 1), (r, c + 1)]


def _mirror_col(c: int, axis2: int) -> int:
    return int(axis2 - c)


def _is_connected(cells: set[tuple[int, int]]) -> bool:
    if not cells:
        return False
    start = next(iter(cells))
    seen = {start}
    stack = [start]
    while stack:
        r, c = stack.pop()
        for nb in _neighbors4(r, c):
            if nb in cells and nb not in seen:
                seen.add(nb)
                stack.append(nb)
    return len(seen) == len(cells)


def _axis_room(axis2: int, width: int) -> int:
    if axis2 % 2 == 1:
        k = axis2 // 2
        return min(k + 1, width - k - 1)
    k = axis2 // 2
    return min(k, width - k - 1)


def _candidate_axes(width: int) -> list[int]:
    return [axis2 for axis2 in range(2 * width - 1) if _axis_room(axis2, width) >= 2]


def _marker_pairs_for_axis(
    grid: np.ndarray, bg: int, main_color: int, axis2: int
) -> tuple[set[tuple[tuple[int, int], tuple[int, int]]], set[tuple[int, int]]]:
    h, w = grid.shape
    marker_mask = (grid != bg) & (grid != main_color)
    pairs: set[tuple[tuple[int, int], tuple[int, int]]] = set()
    for r, c in np.argwhere(marker_mask):
        rr = int(r)
        cc = int(c)
        mc = _mirror_col(cc, axis2)
        if mc == cc or not (0 <= mc < w):
            continue
        if int(grid[rr, mc]) != int(grid[rr, cc]):
            continue
        pair = tuple(sorted(((rr, cc), (rr, mc))))
        pairs.add(pair)
    paired_cells = {coord for pair in pairs for coord in pair}
    return pairs, paired_cells


def _completion_gain(grid: np.ndarray, bg: int, main_color: int, axis2: int) -> int:
    h, w = grid.shape
    gain = 0
    for r, c in np.argwhere(grid == main_color):
        rr = int(r)
        cc = int(c)
        mc = _mirror_col(cc, axis2)
        if 0 <= mc < w and int(grid[rr, mc]) == bg:
            gain += 1
    return gain


def _infer_axis(grid: np.ndarray, bg: int, main_color: int) -> tuple[int, set[tuple[int, int]]]:
    best_score: tuple[int, int] | None = None
    best_axes: list[int] = []
    best_cells: set[tuple[int, int]] = set()
    for axis2 in _candidate_axes(grid.shape[1]):
        pairs, paired_cells = _marker_pairs_for_axis(grid, bg, main_color, axis2)
        score = (len(pairs), _completion_gain(grid, bg, main_color, axis2))
        if best_score is None or score > best_score:
            best_score = score
            best_axes = [axis2]
            best_cells = paired_cells
        elif score == best_score:
            best_axes.append(axis2)
    assert best_score is not None and best_score[0] >= 1, "expected at least one mirrored marker pair"
    assert len(best_axes) == 1, "expected a unique best axis"
    return best_axes[0], best_cells


def main(grid: np.ndarray) -> np.ndarray:
    """
    Complete the dominant foreground object using the vertical axis pinned by mirrored marker pairs, and erase every unpaired marker.

    Input structure:
    - One background color fills most cells.
    - The most frequent non-background color forms a partial foreground object.
    - Other foreground colors appear only as isolated singleton marker pixels.

    Rule:
    - Find the unique vertical reflection axis for which same-colored marker pixels form mirrored pairs.
    - Keep those paired markers, but change every unpaired marker pixel to the background color.
    - Reflect every dominant-color object pixel across that axis and paint its mirror only when the mirrored input cell is background.

    Key insight:
    - The clutter pixels are measurement pins: their same-row mirror pairings determine the hidden axis, and that same test also separates real anchors from deletable noise.
    """
    g = np.asarray(grid, dtype=int)
    assert g.ndim == 2

    bg = _get_bg(g)
    main_color = _dominant_fg_color(g, bg)
    axis2, paired_cells = _infer_axis(g, bg, main_color)

    out = g.copy()
    marker_mask = (g != bg) & (g != main_color)
    for r, c in np.argwhere(marker_mask):
        coord = (int(r), int(c))
        if coord not in paired_cells:
            out[coord] = bg

    h, w = g.shape
    for r, c in np.argwhere(g == main_color):
        rr = int(r)
        cc = int(c)
        mc = _mirror_col(cc, axis2)
        if 0 <= mc < w and int(g[rr, mc]) == bg:
            out[rr, mc] = main_color
    return out


def _random_connected_half(height: int, width: int, target: int) -> np.ndarray | None:
    for _ in range(80):
        start = (random.randrange(height), width - 1)
        cells = {start}
        frontier = {nb for nb in _neighbors4(*start) if 0 <= nb[0] < height and 0 <= nb[1] < width}
        while frontier and len(cells) < target:
            cell = random.choice(tuple(frontier))
            frontier.remove(cell)
            cells.add(cell)
            for nb in _neighbors4(*cell):
                if 0 <= nb[0] < height and 0 <= nb[1] < width and nb not in cells:
                    frontier.add(nb)
            if random.random() < 0.25 and frontier and len(cells) < target:
                extra = random.choice(tuple(frontier))
                frontier.remove(extra)
                cells.add(extra)
                for nb in _neighbors4(*extra):
                    if 0 <= nb[0] < height and 0 <= nb[1] < width and nb not in cells:
                        frontier.add(nb)
        mask = np.zeros((height, width), dtype=bool)
        for r, c in cells:
            mask[r, c] = True
        rows = np.where(mask.any(axis=1))[0]
        cols = np.where(mask.any(axis=0))[0]
        if len(rows) >= 3 and len(cols) >= 2 and mask[:, -1].any():
            return mask
    return None


def _expanded_blocked(cells: set[tuple[int, int]], shape: tuple[int, int]) -> set[tuple[int, int]]:
    h, w = shape
    blocked: set[tuple[int, int]] = set()
    for r, c in cells:
        for dr in (-1, 0, 1):
            for dc in (-1, 0, 1):
                rr = r + dr
                cc = c + dc
                if 0 <= rr < h and 0 <= cc < w:
                    blocked.add((rr, cc))
    return blocked


def _choose_axis(width: int) -> int:
    axes = _candidate_axes(width)
    assert axes
    center_axis = width - 1
    off_center = [axis2 for axis2 in axes if axis2 != center_axis]
    if off_center and random.random() < 0.85:
        axes = off_center
    parity = random.choice([0, 1])
    parity_axes = [axis2 for axis2 in axes if axis2 % 2 == parity]
    if parity_axes:
        axes = parity_axes
    return int(random.choice(axes))


def _make_object(
    shape: tuple[int, int], axis2: int
) -> tuple[set[tuple[int, int]], set[tuple[int, int]], set[tuple[int, int]]] | None:
    h, w = shape
    room = _axis_room(axis2, w)
    if room < 2:
        return None

    obj_h = random.randint(5, min(10, h - 2))
    side_w = random.randint(2, min(6, room))
    r0 = random.randint(1, h - obj_h - 1)
    target = random.randint(max(7, obj_h), min(obj_h * side_w, 18))
    local = _random_connected_half(obj_h, side_w, target)
    if local is None:
        return None

    if axis2 % 2 == 1:
        left_c0 = axis2 // 2 - side_w + 1
        axis_cells: set[tuple[int, int]] = set()
    else:
        left_c0 = axis2 // 2 - side_w
        axis_col = axis2 // 2
        axis_rows = np.where(local[:, -1])[0]
        axis_cells = {(r0 + int(rr), axis_col) for rr in axis_rows}

    left_cells = {
        (r0 + int(rr), left_c0 + int(cc))
        for rr, cc in np.argwhere(local)
    }
    right_cells = {(r, _mirror_col(c, axis2)) for r, c in left_cells}
    full_cells = left_cells | right_cells | axis_cells
    if not _is_connected(full_cells):
        return None

    visible_side = random.choice(["left", "right"])
    visible_cells = set(left_cells if visible_side == "left" else right_cells) | axis_cells
    if not _is_connected(visible_cells):
        return None
    return full_cells, visible_cells, axis_cells


def _place_pairs(
    shape: tuple[int, int], axis2: int, blocked: set[tuple[int, int]], pair_count: int
) -> list[tuple[tuple[int, int], tuple[int, int]]] | None:
    h, w = shape
    pairs: list[tuple[tuple[int, int], tuple[int, int]]] = []
    rows = list(range(h))
    random.shuffle(rows)
    for row in rows:
        options = []
        for c in range(w):
            mc = _mirror_col(c, axis2)
            if mc <= c or not (0 <= mc < w):
                continue
            left = (row, c)
            right = (row, mc)
            if left in blocked or right in blocked:
                continue
            options.append((left, right))
        if not options:
            continue
        choice = random.choice(options)
        pairs.append(choice)
        blocked |= _expanded_blocked({choice[0], choice[1]}, shape)
        if len(pairs) == pair_count:
            return pairs
    return None


def _place_noise(
    shape: tuple[int, int], axis2: int, blocked: set[tuple[int, int]], noise_count: int
) -> list[tuple[int, int]] | None:
    h, w = shape
    noise: list[tuple[int, int]] = []
    rows = list(range(h))
    random.shuffle(rows)
    for row in rows:
        options = []
        for c in range(w):
            if _mirror_col(c, axis2) == c:
                continue
            coord = (row, c)
            if coord in blocked:
                continue
            options.append(coord)
        if not options:
            continue
        choice = random.choice(options)
        noise.append(choice)
        blocked |= _expanded_blocked({choice}, shape)
        if len(noise) == noise_count:
            return noise
    return None


def _expected_output(
    shape: tuple[int, int],
    bg: int,
    main_color: int,
    full_cells: set[tuple[int, int]],
    paired_markers: dict[tuple[int, int], int],
) -> np.ndarray:
    out = np.full(shape, bg, dtype=int)
    for r, c in full_cells:
        out[r, c] = main_color
    for (r, c), color in paired_markers.items():
        out[r, c] = color
    return out


def _generate_attempt() -> np.ndarray | None:
    h = random.randint(12, 24)
    w = random.randint(12, 24)
    bg = random.choice(ARC_COLORS)
    main_color = random.choice([c for c in ARC_COLORS if c != bg])
    axis2 = _choose_axis(w)

    made = _make_object((h, w), axis2)
    if made is None:
        return None
    full_cells, visible_cells, _axis_cells = made

    blocked = _expanded_blocked(full_cells, (h, w))
    pair_count = random.randint(2, 4)
    noise_count = random.randint(1, 3)
    pairs = _place_pairs((h, w), axis2, blocked, pair_count)
    if pairs is None:
        return None
    noise = _place_noise((h, w), axis2, blocked, noise_count)
    if noise is None:
        return None

    marker_palette = [c for c in ARC_COLORS if c not in {bg, main_color}]
    base_color_count = random.randint(2, min(4, len(marker_palette)))
    base_colors = random.sample(marker_palette, base_color_count)

    grid = np.full((h, w), bg, dtype=int)
    for r, c in visible_cells:
        grid[r, c] = main_color

    paired_markers: dict[tuple[int, int], int] = {}
    color_counts: Counter[int] = Counter()
    for left, right in pairs:
        color = random.choice(base_colors)
        paired_markers[left] = color
        paired_markers[right] = color
        color_counts[color] += 2
    for coord in noise:
        color = random.choice(base_colors)
        grid[coord] = color
        color_counts[color] += 1
    for coord, color in paired_markers.items():
        grid[coord] = color

    main_count = int(np.count_nonzero(grid == main_color))
    if not color_counts or main_count <= max(color_counts.values()):
        return None

    if _get_bg(grid) != bg or _dominant_fg_color(grid, bg) != main_color:
        return None

    solved_axis2, paired_cells = _infer_axis(grid, bg, main_color)
    if solved_axis2 != axis2:
        return None
    if paired_cells != set(paired_markers):
        return None

    expected = _expected_output((h, w), bg, main_color, full_cells, paired_markers)
    actual = main(grid)
    if not np.array_equal(actual, expected):
        return None
    return grid


def generate_input() -> np.ndarray:
    for _ in range(GEN_ATTEMPTS):
        grid = _generate_attempt()
        if grid is not None:
            return grid
    raise RuntimeError(f"failed to generate a valid instance after {GEN_ATTEMPTS} attempts")
