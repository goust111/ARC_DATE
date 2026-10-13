import numpy as np

from arc_utils import get_bg
import random


def main(grid: np.ndarray) -> np.ndarray:
    """Move each interior marker to the cell just inside the matching-color frame side.

    Input structure
    The grid has background-colored corners, a one-cell rectangular frame, and a
    background interior containing colored marker pixels.

    Rule
    Clear the frame interior. If an interior marker has the top, bottom, left, or
    right frame-side color, repaint it just inside that side while preserving its
    original column or row. Interior colors not used by any frame side disappear.

    Key insight
    The four side colors are directional labels, so each marker's color selects
    the target side and its orthogonal coordinate supplies the placement.
    """
    g = np.asarray(grid, dtype=int)
    h, w = g.shape
    bg = get_bg(g)

    assert h >= 3 and w >= 3
    assert g[0, 0] == g[0, w - 1] == g[h - 1, 0] == g[h - 1, w - 1] == bg

    top = int(g[0, 1])
    bottom = int(g[h - 1, 1])
    left = int(g[1, 0])
    right = int(g[1, w - 1])
    side_colors = {top, bottom, left, right}

    assert bg not in side_colors
    assert len(side_colors) == 4
    assert np.all(g[0, 1 : w - 1] == top)
    assert np.all(g[h - 1, 1 : w - 1] == bottom)
    assert np.all(g[1 : h - 1, 0] == left)
    assert np.all(g[1 : h - 1, w - 1] == right)

    out = g.copy()
    out[1 : h - 1, 1 : w - 1] = bg

    placements: dict[tuple[int, int], int] = {}
    for rr, cc in np.argwhere(g[1 : h - 1, 1 : w - 1] != bg):
        r = int(rr) + 1
        c = int(cc) + 1
        color = int(g[r, c])

        if color == top:
            target = (1, c)
        elif color == bottom:
            target = (h - 2, c)
        elif color == left:
            target = (r, 1)
        elif color == right:
            target = (r, w - 2)
        else:
            continue

        assert target not in placements or placements[target] == color
        placements[target] = color

    for (r, c), color in placements.items():
        out[r, c] = color

    return out

# --------------- generator (o5x) ---------------

# Canvas and palette.
GRID_MIN, GRID_MAX = 10, 17
BACKGROUND, COLOR_MIN, COLOR_MAX = 0, 1, 9
# Marker count constraints.
SIDE_COUNT, MAX_MARKERS_PER_SIDE, MIN_ACTIVE_SIDES = 4, 3, 3
MATCHING_COUNT_MIN, MATCHING_COUNT_MAX = 5, 10
DISTRACTOR_COUNT_MIN, DISTRACTOR_COUNT_MAX = 1, 3
# Placement and retry policy.
MARKER_GAP, GENERATION_RETRIES = 1, 300
SIDE_NAMES = ("top", "bottom", "left", "right")
HORIZONTAL_SIDES = {"top", "bottom"}

def _gen_count_tuple(total):
    choices = [
        counts
        for counts in np.ndindex(*((MAX_MARKERS_PER_SIDE + 1,) * SIDE_COUNT))
        if sum(counts) == total and sum(v > 0 for v in counts) >= MIN_ACTIVE_SIDES
    ]
    return list(random.choice(choices))


def _gen_choose_targets(h, w, counts):
    side_specs = [
        (SIDE_NAMES[0], 1, range(1, w - 1)),
        (SIDE_NAMES[1], h - 2, range(1, w - 1)),
        (SIDE_NAMES[2], 1, range(1, h - 1)),
        (SIDE_NAMES[3], w - 2, range(1, h - 1)),
    ]
    order = list(range(SIDE_COUNT))
    random.shuffle(order)
    occupied = set()
    by_side = [[] for _ in range(SIDE_COUNT)]

    for idx in order:
        side, fixed, coords = side_specs[idx]
        available = []
        for coord in coords:
            cell = (fixed, coord) if side in HORIZONTAL_SIDES else (coord, fixed)
            if cell not in occupied:
                available.append(coord)
        if len(available) < counts[idx]:
            return None
        chosen = random.sample(available, counts[idx])
        for coord in chosen:
            cell = (fixed, coord) if side in HORIZONTAL_SIDES else (coord, fixed)
            occupied.add(cell)
            by_side[idx].append((side, coord, cell))

    return by_side


def _framed_grid(h, w, side_colors):
    top, bottom, left, right = side_colors
    grid = np.full((h, w), BACKGROUND, dtype=int)
    grid[0, 1 : w - 1] = top
    grid[h - 1, 1 : w - 1] = bottom
    grid[1 : h - 1, 0] = left
    grid[1 : h - 1, w - 1] = right
    return grid


def _is_free_with_gap(cell, occupied):
    r, c = cell
    return cell not in occupied and all(
        max(abs(r - rr), abs(c - cc)) > MARKER_GAP for rr, cc in occupied
    )


def _gen_place_input_marker(inp, marker, color, occupied):
    side, coord, _target = marker
    h, w = inp.shape
    if side in HORIZONTAL_SIDES:
        candidates = [
            (r, coord)
            for r in range(1, h - 1)
            if _is_free_with_gap((r, coord), occupied)
        ]
    else:
        candidates = [
            (coord, c)
            for c in range(1, w - 1)
            if _is_free_with_gap((coord, c), occupied)
        ]
    if not candidates:
        return False
    r, c = random.choice(candidates)
    inp[r, c] = color
    occupied.add((r, c))
    return True


def _render_output_markers(out, targets, side_colors):
    all_markers = []
    for side_idx, markers in enumerate(targets):
        color = side_colors[side_idx]
        for marker in markers:
            r, c = marker[2]
            out[r, c] = color
            all_markers.append((marker, color))
    return all_markers


def _place_matching_input_markers(inp, all_markers):
    occupied = set()
    random.shuffle(all_markers)
    for marker, color in all_markers:
        if not _gen_place_input_marker(inp, marker, color, occupied):
            return None
    return occupied


def _place_distractors(inp, side_colors, occupied):
    h, w = inp.shape
    distractor_colors = [
        c for c in range(COLOR_MIN, COLOR_MAX + 1) if c not in side_colors
    ]
    distractor_count = random.randint(DISTRACTOR_COUNT_MIN, DISTRACTOR_COUNT_MAX)
    for _ in range(distractor_count):
        free_cells = [
            (r, c)
            for r in range(1, h - 1)
            for c in range(1, w - 1)
            if _is_free_with_gap((r, c), occupied)
        ]
        if not free_cells:
            return False
        r, c = random.choice(free_cells)
        inp[r, c] = random.choice(distractor_colors)
        occupied.add((r, c))
    return True


def _gen_attempt():
    h, w = random.randint(GRID_MIN, GRID_MAX), random.randint(GRID_MIN, GRID_MAX)
    side_colors = random.sample(range(COLOR_MIN, COLOR_MAX + 1), SIDE_COUNT)

    total = random.randint(MATCHING_COUNT_MIN, MATCHING_COUNT_MAX)
    counts = _gen_count_tuple(total)
    targets = _gen_choose_targets(h, w, counts)
    if targets is None:
        return None

    out = _framed_grid(h, w, side_colors)
    all_markers = _render_output_markers(out, targets, side_colors)

    inp = _framed_grid(h, w, side_colors)
    occupied = _place_matching_input_markers(inp, all_markers)
    if occupied is None:
        return None

    if not _place_distractors(inp, side_colors, occupied):
        return None

    if not np.array_equal(main(inp), out):
        return None
    return inp


def generate_input():
    for _ in range(GENERATION_RETRIES):
        grid = _gen_attempt()
        if grid is not None:
            return grid
    raise RuntimeError("failed to generate valid d687bc17 input")
