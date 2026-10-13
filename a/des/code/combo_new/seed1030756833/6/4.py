from __future__ import annotations

import random
from collections import Counter

import numpy as np


AXIS_COLOR = 5
COLOR_COUNT = 10
NEIGHBORS4 = ((-1, 0), (1, 0), (0, -1), (0, 1))
MAX_GENERATION_ATTEMPTS = 300


def _background_color(g: np.ndarray) -> int:
    values, counts = np.unique(g, return_counts=True)
    return int(values[int(np.argmax(counts))])


def _vertical_mismatch_count(g: np.ndarray, axis_c: int) -> int:
    h, w = g.shape
    max_d = min(axis_c, w - 1 - axis_c)
    mismatches = 0
    for r in range(h - 1):
        for d in range(1, max_d + 1):
            if int(g[r, axis_c - d]) != int(g[r, axis_c + d]):
                mismatches += 1
    return mismatches


def _horizontal_mismatch_count(g: np.ndarray, axis_r: int) -> int:
    h, w = g.shape
    max_d = min(axis_r, h - 1 - axis_r)
    mismatches = 0
    for c in range(w - 1):
        for d in range(1, max_d + 1):
            if int(g[axis_r - d, c]) != int(g[axis_r + d, c]):
                mismatches += 1
    return mismatches


def _find_axis(g: np.ndarray, bg: int) -> tuple[str, int, int]:
    h, w = g.shape
    candidates: list[tuple[int, int, int, str, int]] = []

    if h >= 3 and np.all(g[h - 1, :] == bg):
        for c in range(1, w - 1):
            line = g[: h - 1, c]
            color = int(line[0])
            if color == bg:
                continue
            if np.all(line == color) and int(g[h - 1, c]) == bg:
                mismatches = _vertical_mismatch_count(g, c)
                extra = int(np.count_nonzero(g == color) - (h - 1))
                candidates.append((0 if color == AXIS_COLOR else 1, mismatches, extra, "vertical", c))

    if w >= 3 and np.all(g[:, w - 1] == bg):
        for r in range(1, h - 1):
            line = g[r, : w - 1]
            color = int(line[0])
            if color == bg:
                continue
            if np.all(line == color) and int(g[r, w - 1]) == bg:
                mismatches = _horizontal_mismatch_count(g, r)
                extra = int(np.count_nonzero(g == color) - (w - 1))
                candidates.append((0 if color == AXIS_COLOR else 1, mismatches, extra, "horizontal", r))

    assert candidates, "expected one full grey symmetry axis with a blank signature edge"
    candidates.sort()
    _axis_priority, mismatches, _extra, orientation, index = candidates[0]
    assert mismatches <= 5, "axis should have only a few asymmetric pixel pairs"
    return orientation, int(index), int(mismatches)


def _is_axis_cell(r: int, c: int, orientation: str, axis: int, h: int, w: int) -> bool:
    if orientation == "vertical":
        return c == axis and r < h - 1
    return r == axis and c < w - 1


def _is_signature_cell(r: int, c: int, orientation: str, h: int, w: int) -> bool:
    if orientation == "vertical":
        return r == h - 1
    return c == w - 1


def _locally_inconsistent(g: np.ndarray, r: int, c: int, orientation: str, axis: int) -> bool:
    h, w = g.shape
    color = int(g[r, c])
    neighbors: list[int] = []
    for dr, dc in NEIGHBORS4:
        nr, nc = r + dr, c + dc
        if not (0 <= nr < h and 0 <= nc < w):
            continue
        if _is_axis_cell(nr, nc, orientation, axis, h, w):
            continue
        if _is_signature_cell(nr, nc, orientation, h, w):
            continue
        neighbors.append(int(g[nr, nc]))
    if not neighbors:
        return False

    counts = Counter(neighbors)
    majority_color, majority_count = max(counts.items(), key=lambda item: (item[1], -item[0]))
    same_count = counts.get(color, 0)
    return majority_color != color and majority_count >= 2 and same_count <= 1


def _find_repairs(g: np.ndarray, orientation: str, axis: int) -> list[tuple[int, int, int, int, int]]:
    h, w = g.shape
    repairs: list[tuple[int, int, int, int, int]] = []

    if orientation == "vertical":
        max_d = min(axis, w - 1 - axis)
        for r in range(h - 1):
            for d in range(1, max_d + 1):
                left = (r, axis - d)
                right = (r, axis + d)
                left_color = int(g[left])
                right_color = int(g[right])
                if left_color == right_color:
                    continue
                bad_left = _locally_inconsistent(g, left[0], left[1], orientation, axis)
                bad_right = _locally_inconsistent(g, right[0], right[1], orientation, axis)
                assert bad_left != bad_right, "each asymmetric pair must identify exactly one local anomaly"
                if bad_left:
                    repairs.append((r, d, left[0], left[1], right_color))
                else:
                    repairs.append((r, d, right[0], right[1], left_color))
    else:
        max_d = min(axis, h - 1 - axis)
        for c in range(w - 1):
            for d in range(1, max_d + 1):
                top = (axis - d, c)
                bottom = (axis + d, c)
                top_color = int(g[top])
                bottom_color = int(g[bottom])
                if top_color == bottom_color:
                    continue
                bad_top = _locally_inconsistent(g, top[0], top[1], orientation, axis)
                bad_bottom = _locally_inconsistent(g, bottom[0], bottom[1], orientation, axis)
                assert bad_top != bad_bottom, "each asymmetric pair must identify exactly one local anomaly"
                if bad_top:
                    repairs.append((c, d, top[0], top[1], bottom_color))
                else:
                    repairs.append((c, d, bottom[0], bottom[1], top_color))

    repairs.sort()
    return repairs


def main(grid: np.ndarray) -> np.ndarray:
    """
    Repair tampered mirror pixels and write their former colors as an edge signature.

    Input structure:
    - The background is the dominant color.
    - A unique grey line forms a complete vertical or horizontal symmetry axis,
      stopping before the reserved blank signature edge.
    - On the two sides of the axis is a nearly mirror-symmetric multicolor
      pattern, with at most five single pixels replaced by wrong colors.
    - The bottom row is blank for a vertical axis, and the right column is blank
      for a horizontal axis.

    Rule:
    - Identify the grey axis and compare every pair of reflected cells on its
      two sides.
    - For each unequal reflected pair, the side whose four-neighborhood majority
      disagrees with its current color is the tampered pixel.
    - Replace each tampered pixel by the color of its mirror mate.
    - Sort repaired pixels by their projection along the axis, then write their
      pre-repair colors into the blank signature edge from the start.

    Key insight: The mirror comparison locates suspicious pairs, local
    neighborhood consistency chooses which side is wrong, and the discarded
    colors become the ordered edge signature.
    """
    g = np.asarray(grid, dtype=int)
    assert g.ndim == 2, "grid must be two-dimensional"
    bg = _background_color(g)
    orientation, axis, _mismatches = _find_axis(g, bg)
    repairs = _find_repairs(g, orientation, axis)

    out = g.copy()
    signature: list[int] = []
    for _projection, _distance, r, c, new_color in repairs:
        signature.append(int(out[r, c]))
        out[r, c] = int(new_color)

    if orientation == "vertical":
        out[-1, :] = bg
        for c, color in enumerate(signature):
            out[-1, c] = int(color)
    else:
        out[:, -1] = bg
        for r, color in enumerate(signature):
            out[r, -1] = int(color)

    return out


def _free_rect(occupied: np.ndarray, r0: int, c0: int, rh: int, rw: int) -> bool:
    r1 = r0 + rh
    c1 = c0 + rw
    pr0 = max(0, r0 - 1)
    pc0 = max(0, c0 - 1)
    pr1 = min(occupied.shape[0], r1 + 1)
    pc1 = min(occupied.shape[1], c1 + 1)
    return not bool(occupied[pr0:pr1, pc0:pc1].any())


def _paint_rect_pair(
    grid: np.ndarray,
    orientation: str,
    axis: int,
    rect: tuple[int, int, int, int],
    color: int,
) -> None:
    r0, c0, r1, c1 = rect
    grid[r0 : r1 + 1, c0 : c1 + 1] = color
    if orientation == "vertical":
        mc0 = 2 * axis - c1
        mc1 = 2 * axis - c0
        grid[r0 : r1 + 1, mc0 : mc1 + 1] = color
    else:
        mr0 = 2 * axis - r1
        mr1 = 2 * axis - r0
        grid[mr0 : mr1 + 1, c0 : c1 + 1] = color


def _make_clean_grid(orientation: str, bg: int, colors: list[int]) -> tuple[np.ndarray, list[tuple]]:
    if orientation == "vertical":
        side = random.randint(6, 14)
        height = random.randint(12, 26)
        width = 2 * side + 1
        axis = side
        grid = np.full((height, width), bg, dtype=int)
        grid[: height - 1, axis] = AXIS_COLOR
        source_shape = (height - 1, side)
    else:
        side = random.randint(6, 14)
        width = random.randint(12, 26)
        height = 2 * side + 1
        axis = side
        grid = np.full((height, width), bg, dtype=int)
        grid[axis, : width - 1] = AXIS_COLOR
        source_shape = (side, width - 1)

    occupied = np.zeros(source_shape, dtype=bool)
    rectangles: list[tuple] = []
    target_count = random.randint(2, 5)
    for _ in range(200):
        if len(rectangles) >= target_count:
            break
        max_h = min(6, source_shape[0])
        max_w = min(6, source_shape[1])
        if max_h < 3 or max_w < 3:
            return grid, []
        rh = random.randint(3, max_h)
        rw = random.randint(3, max_w)
        r0 = random.randint(0, source_shape[0] - rh)
        c0 = random.randint(0, source_shape[1] - rw)
        if not _free_rect(occupied, r0, c0, rh, rw):
            continue
        occupied[r0 : r0 + rh, c0 : c0 + rw] = True
        rect = (r0, c0, r0 + rh - 1, c0 + rw - 1)
        color = random.choice(colors)
        _paint_rect_pair(grid, orientation, axis, rect, color)
        rectangles.append((rect, color, axis))

    if len(rectangles) < 2:
        return grid, []
    return grid, rectangles


def _candidate_pairs(orientation: str, rectangles: list[tuple]) -> list[tuple[int, tuple[int, int], tuple[int, int], int]]:
    pairs: list[tuple[int, tuple[int, int], tuple[int, int], int]] = []
    for rect, color, axis in rectangles:
        r0, c0, r1, c1 = rect
        for r in range(r0 + 1, r1):
            for c in range(c0 + 1, c1):
                if orientation == "vertical":
                    left = (r, c)
                    right = (r, 2 * axis - c)
                    projection = r
                else:
                    left = (r, c)
                    right = (2 * axis - r, c)
                    projection = c
                pairs.append((projection, left, right, int(color)))
    random.shuffle(pairs)
    return pairs


def _near_any(coord: tuple[int, int], selected: set[tuple[int, int]]) -> bool:
    r, c = coord
    for sr, sc in selected:
        if abs(r - sr) + abs(c - sc) <= 1:
            return True
    return False


def _add_tamper_pixels(
    clean: np.ndarray,
    orientation: str,
    rectangles: list[tuple],
    colors: list[int],
) -> tuple[np.ndarray, np.ndarray] | None:
    inp = clean.copy()
    expected = clean.copy()
    pairs = _candidate_pairs(orientation, rectangles)
    if not pairs:
        return None

    want = random.randint(1, min(5, len(pairs)))
    used_projections: set[int] = set()
    selected_cells: set[tuple[int, int]] = set()
    signature_items: list[tuple[int, int]] = []
    chosen = 0

    for projection, first, second, original in pairs:
        if chosen >= want:
            break
        if projection in used_projections:
            continue
        if _near_any(first, selected_cells) or _near_any(second, selected_cells):
            continue
        wrong_choices = [color for color in colors if color != original]
        if not wrong_choices:
            continue
        bad = first if random.random() < 0.5 else second
        wrong = random.choice(wrong_choices)
        inp[bad] = wrong
        signature_items.append((int(projection), int(wrong)))
        used_projections.add(int(projection))
        selected_cells.add(first)
        selected_cells.add(second)
        chosen += 1

    if chosen < want:
        return None

    signature_items.sort()
    if orientation == "vertical":
        expected[-1, :] = _background_color(clean)
        for c, (_projection, color) in enumerate(signature_items):
            expected[-1, c] = color
    else:
        expected[:, -1] = _background_color(clean)
        for r, (_projection, color) in enumerate(signature_items):
            expected[r, -1] = color
    return inp, expected


def _gen_attempt() -> np.ndarray | None:
    orientation = random.choice(("vertical", "horizontal"))
    bg = random.choice([color for color in range(COLOR_COUNT) if color != AXIS_COLOR])
    colors = [color for color in range(COLOR_COUNT) if color not in (bg, AXIS_COLOR)]
    random.shuffle(colors)
    colors = colors[: random.randint(3, min(6, len(colors)))]

    clean, rectangles = _make_clean_grid(orientation, bg, colors)
    if not rectangles:
        return None
    made = _add_tamper_pixels(clean, orientation, rectangles, colors)
    if made is None:
        return None
    inp, expected = made
    if max(inp.shape) > 30 or max(expected.shape) > 30:
        return None
    if _background_color(inp) != bg:
        return None
    try:
        solved = main(inp)
    except AssertionError:
        return None
    if not np.array_equal(solved, expected):
        return None
    return inp


def generate_input() -> np.ndarray:
    for _ in range(MAX_GENERATION_ATTEMPTS):
        grid = _gen_attempt()
        if grid is not None:
            return grid
    raise RuntimeError("failed to generate a valid input")
