from __future__ import annotations

import random

import numpy as np


BLACK = 0
GREY = 5
FIELD_SIZE = 5
SLOT_WIDTH = FIELD_SIZE + 1
SLOT_GAP = 2
ATLAS_HEIGHT = FIELD_SIZE
PALETTE = [1, 2, 3, 4, 6, 7, 8, 9]


def _normalize(coords: list[tuple[int, int]]) -> tuple[tuple[int, int], ...]:
    arr = np.array(coords, dtype=int)
    arr[:, 0] -= int(arr[:, 0].min())
    arr[:, 1] -= int(arr[:, 1].min())
    return tuple(sorted((int(r), int(c)) for r, c in arr))


def _orientations(coords: list[tuple[int, int]]) -> list[tuple[tuple[int, int], ...]]:
    pts = [(int(r), int(c)) for r, c in coords]
    oriented = []
    seen = set()
    for swap in (False, True):
        for sr in (-1, 1):
            for sc in (-1, 1):
                if swap:
                    transformed = [(sr * c, sc * r) for r, c in pts]
                else:
                    transformed = [(sr * r, sc * c) for r, c in pts]
                key = _normalize(transformed)
                if key not in seen:
                    seen.add(key)
                    oriented.append(key)
    return oriented


def _shape_key(coords: list[tuple[int, int]]) -> tuple[tuple[int, int], ...]:
    return min(_orientations(coords))


def _round_half_up(values: np.ndarray) -> int:
    return int(np.floor(float(np.mean(values)) + 0.5))


def _atlas_top(g: np.ndarray) -> int:
    grey = np.argwhere(g == GREY)
    assert len(grey) > 0, "expected a grey atlas"
    return int(grey[:, 0].min())


def _atlas_entries(g: np.ndarray) -> dict[tuple[tuple[int, int], ...], int]:
    top = _atlas_top(g)
    atlas = g[top:, :]

    label_coords = []
    for rr, cc in np.argwhere((atlas != BLACK) & (atlas != GREY)):
        r, c = int(rr + top), int(cc)
        label_coords.append((r, c, int(g[r, c])))
    assert label_coords, "expected colored atlas labels"

    assigned: list[list[tuple[int, int]]] = [[] for _ in label_coords]
    for r, c in ((int(r), int(c)) for r, c in np.argwhere(g == GREY)):
        choices = []
        for i, (lr, lc, _color) in enumerate(label_coords):
            if lc > c:
                choices.append((lc - c, abs(lr - r), i))
        assert choices, "each grey reference pixel must have a label to its right"
        _dist, _row_gap, idx = min(choices)
        assigned[idx].append((r, c))

    entries: dict[tuple[tuple[int, int], ...], int] = {}
    for (lr, lc, color), coords in zip(label_coords, assigned):
        assert coords, f"label at {(lr, lc)} has no grey reference pixels"
        key = _shape_key(coords)
        assert key not in entries, "reference constellation shapes must be unique"
        entries[key] = color
    return entries


def main(grid: np.ndarray) -> np.ndarray:
    """
    Mark each scattered same-color constellation with the atlas label for its relative shape.

    Input structure:
    - The background is Black.
    - Above the bottom atlas, each non-Black, non-Grey color appears as one 3-5 pixel constellation.
    - The bottom atlas contains small Grey reference constellations; each reference has a one-cell colored label immediately to its right.

    Rule:
    - For each foreground color above the atlas, normalize the relative positions of all pixels of that color.
    - Match that normalized constellation to a Grey atlas reference, allowing rotations and flips.
    - Compute the visual center of the foreground constellation by rounding the average row and column.
    - Clear the foreground pixels and the atlas, then put the matched label color at that center.

    Key insight: The original color and absolute position are distractors; the constellation's relative geometry is the lookup key.
    """
    g = np.asarray(grid, dtype=int)
    assert g.ndim == 2
    top = _atlas_top(g)
    atlas = _atlas_entries(g)

    out = np.zeros_like(g)
    upper = g[:top, :]
    colors = [int(c) for c in np.unique(upper) if int(c) not in (BLACK, GREY)]
    assert colors, "expected at least one foreground constellation"

    used_centers: set[tuple[int, int]] = set()
    for color in colors:
        coords_arr = np.argwhere(upper == color)
        assert 3 <= len(coords_arr) <= 5, "each constellation must contain 3-5 pixels"
        coords = [(int(r), int(c)) for r, c in coords_arr]
        key = _shape_key(coords)
        assert key in atlas, "constellation shape must appear in the atlas"

        rr = _round_half_up(coords_arr[:, 0])
        cc = _round_half_up(coords_arr[:, 1])
        assert (rr, cc) not in used_centers, "constellation centers must be distinct"
        used_centers.add((rr, cc))
        out[rr, cc] = atlas[key]

    return out


def _bbox_size(coords: tuple[tuple[int, int], ...]) -> tuple[int, int]:
    rows = [r for r, _c in coords]
    cols = [c for _r, c in coords]
    return max(rows) + 1, max(cols) + 1


def _random_constellation() -> tuple[tuple[int, int], ...]:
    for _ in range(200):
        count = random.randint(3, 5)
        height = random.randint(3, FIELD_SIZE)
        width = random.randint(3, FIELD_SIZE)
        coords = random.sample([(r, c) for r in range(height) for c in range(width)], count)
        norm = _normalize(coords)
        rows = [r for r, _c in norm]
        cols = [c for _r, c in norm]
        if max(rows) < 2 or max(cols) < 2:
            continue
        return norm
    raise RuntimeError("failed to make a constellation")


def _pick_shapes(n: int) -> list[tuple[tuple[int, int], ...]]:
    shapes = []
    used = set()
    for _ in range(n):
        for _attempt in range(500):
            shape = _random_constellation()
            key = _shape_key(list(shape))
            if key in used:
                continue
            used.add(key)
            shapes.append(shape)
            break
        else:
            return []
    return shapes


def _random_orientation(shape: tuple[tuple[int, int], ...]) -> tuple[tuple[int, int], ...]:
    return random.choice(_orientations(list(shape)))


def _center(coords: list[tuple[int, int]]) -> tuple[int, int]:
    arr = np.array(coords, dtype=int)
    return _round_half_up(arr[:, 0]), _round_half_up(arr[:, 1])


def _can_place(blocked: np.ndarray, coords: list[tuple[int, int]], center: tuple[int, int]) -> bool:
    h, w = blocked.shape
    cells = set(coords)
    for r, c in coords:
        if blocked[r, c]:
            return False
    cr, cc = center
    if not (0 <= cr < h and 0 <= cc < w):
        return False
    if blocked[cr, cc] and (cr, cc) not in cells:
        return False
    return True


def _mark_blocked(blocked: np.ndarray, coords: list[tuple[int, int]], center: tuple[int, int]) -> None:
    h, w = blocked.shape
    for r, c in coords:
        for rr in range(max(0, r - 1), min(h, r + 2)):
            for cc in range(max(0, c - 1), min(w, c + 2)):
                blocked[rr, cc] = True
    cr, cc = center
    blocked[cr, cc] = True


def _place_foreground(
    grid: np.ndarray,
    shapes: list[tuple[tuple[int, int], ...]],
    colors: list[int],
    label_colors: list[int],
    atlas_top: int,
) -> np.ndarray | None:
    star_h = atlas_top - 1
    width = grid.shape[1]
    blocked = np.zeros((star_h, width), dtype=bool)
    expected = np.zeros_like(grid)

    order = list(range(len(shapes)))
    random.shuffle(order)
    for i in order:
        shape = _random_orientation(shapes[i])
        ph, pw = _bbox_size(shape)
        if ph > star_h or pw > width:
            return None

        for _attempt in range(500):
            r0 = random.randint(0, star_h - ph)
            c0 = random.randint(0, width - pw)
            coords = [(r0 + r, c0 + c) for r, c in shape]
            center = _center(coords)
            if not _can_place(blocked, coords, center):
                continue
            for r, c in coords:
                grid[r, c] = colors[i]
            expected[center] = label_colors[i]
            _mark_blocked(blocked, coords, center)
            break
        else:
            return None

    return expected


def _draw_atlas(
    grid: np.ndarray,
    shapes: list[tuple[tuple[int, int], ...]],
    label_colors: list[int],
    atlas_top: int,
) -> None:
    n = len(shapes)
    total_width = n * SLOT_WIDTH + (n - 1) * SLOT_GAP
    start_col = random.randint(0, grid.shape[1] - total_width)

    for i, shape in enumerate(shapes):
        c0 = start_col + i * (SLOT_WIDTH + SLOT_GAP)
        oriented = _random_orientation(shape)
        ph, pw = _bbox_size(oriented)
        roff = random.randint(0, FIELD_SIZE - ph)
        coff = random.randint(0, FIELD_SIZE - pw)
        for r, c in oriented:
            grid[atlas_top + roff + r, c0 + coff + c] = GREY
        grid[atlas_top + FIELD_SIZE // 2, c0 + FIELD_SIZE] = label_colors[i]


def _gen_attempt() -> np.ndarray | None:
    width = random.randint(22, 30)
    height = random.randint(18, 30)
    atlas_top = height - ATLAS_HEIGHT
    max_cards = (width + SLOT_GAP) // (SLOT_WIDTH + SLOT_GAP)
    if max_cards < 3:
        return None
    n_shapes = random.randint(3, min(4, max_cards))

    shapes = _pick_shapes(n_shapes)
    if not shapes:
        return None

    colors = random.sample(PALETTE, 2 * n_shapes)
    fg_colors = colors[:n_shapes]
    label_colors = colors[n_shapes:]

    grid = np.full((height, width), BLACK, dtype=int)
    expected = _place_foreground(grid, shapes, fg_colors, label_colors, atlas_top)
    if expected is None:
        return None
    _draw_atlas(grid, shapes, label_colors, atlas_top)

    try:
        actual = main(grid)
    except Exception:
        return None
    if not np.array_equal(actual, expected):
        return None
    return grid


def generate_input() -> np.ndarray:
    for _ in range(300):
        grid = _gen_attempt()
        if grid is not None:
            return grid
    raise RuntimeError("generate_input failed after 300 attempts")
