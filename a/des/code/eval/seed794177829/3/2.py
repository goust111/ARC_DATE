from __future__ import annotations

import random

import numpy as np


_RNG = random.Random()
_BG = 0
_COLOR_POOL = tuple(range(1, 10))
_MAX_GENERATION_ATTEMPTS = 4000
_ORIENTATIONS = ("horizontal", "vertical")


def _get_bg(grid: np.ndarray) -> int:
    values = np.bincount(np.asarray(grid, dtype=int).ravel())
    return int(np.argmax(values))


def _bbox_from_mask(mask: np.ndarray) -> tuple[int, int, int, int]:
    coords = np.argwhere(mask)
    assert coords.size > 0, "expected a non-empty mask"
    rows = coords[:, 0]
    cols = coords[:, 1]
    return int(rows.min()), int(cols.min()), int(rows.max()), int(cols.max())


def _neighbors4(shape: tuple[int, int], r: int, c: int) -> list[tuple[int, int]]:
    h, w = shape
    out: list[tuple[int, int]] = []
    if r > 0:
        out.append((r - 1, c))
    if r + 1 < h:
        out.append((r + 1, c))
    if c > 0:
        out.append((r, c - 1))
    if c + 1 < w:
        out.append((r, c + 1))
    return out


def _same_neighbor_count(grid: np.ndarray, r: int, c: int) -> int:
    color = int(grid[r, c])
    return sum(int(grid[nr, nc]) == color for nr, nc in _neighbors4(grid.shape, r, c))


def _find_marker_line(grid: np.ndarray, bg: int) -> tuple[int, str]:
    h, w = grid.shape
    candidates: list[tuple[int, str]] = []
    for color in map(int, np.unique(grid)):
        if color == bg:
            continue
        coords = np.argwhere(grid == color)
        rows = coords[:, 0]
        cols = coords[:, 1]

        if np.all(rows == rows[0]) and int(rows[0]) in (0, h - 1):
            c0 = int(cols.min())
            c1 = int(cols.max())
            if coords.shape[0] >= 2 and coords.shape[0] == c1 - c0 + 1:
                candidates.append((color, "horizontal"))

        if np.all(cols == cols[0]) and int(cols[0]) in (0, w - 1):
            r0 = int(rows.min())
            r1 = int(rows.max())
            if coords.shape[0] >= 2 and coords.shape[0] == r1 - r0 + 1:
                candidates.append((color, "vertical"))

    unique = list(dict.fromkeys(candidates))
    assert len(unique) == 1, "expected exactly one edge marker line"
    return unique[0]


def _iter_mirror_pairs(
    bbox: tuple[int, int, int, int], orientation: str
) -> list[tuple[tuple[int, int], tuple[int, int]]]:
    r0, c0, r1, c1 = bbox
    pairs: list[tuple[tuple[int, int], tuple[int, int]]] = []
    if orientation == "horizontal":
        axis_sum = r0 + r1
        for r in range(r0, r1 + 1):
            mr = axis_sum - r
            if r >= mr:
                continue
            for c in range(c0, c1 + 1):
                pairs.append(((r, c), (mr, c)))
    else:
        axis_sum = c0 + c1
        for c in range(c0, c1 + 1):
            mc = axis_sum - c
            if c >= mc:
                continue
            for r in range(r0, r1 + 1):
                pairs.append(((r, c), (r, mc)))
    return pairs


def main(grid: np.ndarray) -> np.ndarray:
    """
    Repair a reflection-symmetric pattern by locating the polluted side of each mismatch.

    The edge marker line gives the axis orientation. The pattern's bounding box
    gives the geometric midline. For every mirrored pair that disagrees, the
    pixel with fewer same-colored 4-neighbors is the pollutant and is recolored
    to match its mirror partner.
    """
    g = np.asarray(grid, dtype=int)
    assert g.ndim == 2, "expected a 2D grid"

    bg = _get_bg(g)
    marker_color, orientation = _find_marker_line(g, bg)

    pattern_mask = (g != bg) & (g != marker_color)
    bbox = _bbox_from_mask(pattern_mask)

    replacements: dict[tuple[int, int], int] = {}
    for pos_a, pos_b in _iter_mirror_pairs(bbox, orientation):
        ra, ca = pos_a
        rb, cb = pos_b
        color_a = int(g[ra, ca])
        color_b = int(g[rb, cb])
        if color_a == color_b:
            continue

        support_a = _same_neighbor_count(g, ra, ca)
        support_b = _same_neighbor_count(g, rb, cb)
        assert support_a != support_b, "mismatched mirror pair must have a unique pollutant"

        if support_a < support_b:
            replacements[(ra, ca)] = color_b
        else:
            replacements[(rb, cb)] = color_a

    out = g.copy()
    for (r, c), color in replacements.items():
        out[r, c] = color
    return out


def _random_palette(count: int, exclude: set[int] | None = None) -> list[int]:
    exclude = exclude or set()
    choices = [color for color in _COLOR_POOL if color not in exclude]
    return _RNG.sample(choices, count)


def _smooth_array(arr: np.ndarray) -> np.ndarray:
    rows, cols = arr.shape
    steps = _RNG.randint(rows * cols, rows * cols * 2)
    for _ in range(steps):
        r = _RNG.randrange(rows)
        c = _RNG.randrange(cols)
        neighbors = [(r, c)]
        if r > 0:
            neighbors.append((r - 1, c))
        if r + 1 < rows:
            neighbors.append((r + 1, c))
        if c > 0:
            neighbors.append((r, c - 1))
        if c + 1 < cols:
            neighbors.append((r, c + 1))
        nr, nc = _RNG.choice(neighbors)
        arr[nr, nc] = arr[r, c]

    for _ in range(_RNG.randint(1, 3)):
        color = int(_RNG.choice(arr.ravel()))
        h = _RNG.randint(1, max(1, rows // 2))
        w = _RNG.randint(1, max(1, cols // 2))
        r0 = _RNG.randint(0, rows - h)
        c0 = _RNG.randint(0, cols - w)
        arr[r0 : r0 + h, c0 : c0 + w] = color
    return arr


def _make_symmetric_patch(orientation: str, palette: list[int]) -> np.ndarray:
    cell_h = _RNG.randint(2, 3)
    cell_w = _RNG.randint(2, 3)

    if orientation == "horizontal":
        top_rows = _RNG.randint(2, 3)
        center_rows = _RNG.randint(0, 1)
        cols = _RNG.randint(4, 6)
        seed = np.array(_RNG.choices(palette, k=(top_rows + center_rows) * cols), dtype=int)
        seed = seed.reshape(top_rows + center_rows, cols)
        seed = _smooth_array(seed)
        top = seed[:top_rows]
        middle = seed[top_rows:] if center_rows else np.empty((0, cols), dtype=int)
        coarse = np.vstack([top, middle, top[::-1]])
    else:
        rows = _RNG.randint(4, 6)
        left_cols = _RNG.randint(2, 3)
        center_cols = _RNG.randint(0, 1)
        seed = np.array(_RNG.choices(palette, k=rows * (left_cols + center_cols)), dtype=int)
        seed = seed.reshape(rows, left_cols + center_cols)
        seed = _smooth_array(seed)
        left = seed[:, :left_cols]
        middle = seed[:, left_cols:] if center_cols else np.empty((rows, 0), dtype=int)
        coarse = np.hstack([left, middle, left[:, ::-1]])

    if coarse.size >= len(palette):
        slots = _RNG.sample(range(coarse.size), len(palette))
        flat = coarse.ravel()
        for idx, color in zip(slots, palette):
            flat[idx] = color
        coarse = flat.reshape(coarse.shape)

    return np.repeat(np.repeat(coarse, cell_h, axis=0), cell_w, axis=1)


def _place_marker(grid: np.ndarray, orientation: str, marker_color: int) -> None:
    h, w = grid.shape
    if orientation == "horizontal":
        row = _RNG.choice([0, h - 1])
        length = _RNG.randint(3, w - 2)
        start = _RNG.randint(1, w - length - 1)
        grid[row, start : start + length] = marker_color
    else:
        col = _RNG.choice([0, w - 1])
        length = _RNG.randint(3, h - 2)
        start = _RNG.randint(1, h - length - 1)
        grid[start : start + length, col] = marker_color


def _pair_clear_of_used(
    pair: tuple[tuple[int, int], tuple[int, int]], used_positions: list[tuple[int, int]]
) -> bool:
    for r, c in pair:
        for ur, uc in used_positions:
            if abs(r - ur) + abs(c - uc) <= 1:
                return False
    return True


def _pollution_options(
    grid: np.ndarray, pos: tuple[int, int], original: int, palette: list[int], bg: int
) -> list[int]:
    neighbor_colors = {int(grid[nr, nc]) for nr, nc in _neighbors4(grid.shape, *pos)}
    options = [color for color in palette if color != original and color not in neighbor_colors]
    if bg != original and bg not in neighbor_colors and _RNG.random() < 0.35:
        options.append(bg)
    _RNG.shuffle(options)
    return options


def _candidate_pairs(
    clean: np.ndarray, bbox: tuple[int, int, int, int], orientation: str
) -> list[tuple[tuple[int, int], tuple[int, int]]]:
    r0, c0, r1, c1 = bbox
    pairs = []
    for pos_a, pos_b in _iter_mirror_pairs(bbox, orientation):
        ra, ca = pos_a
        rb, cb = pos_b
        if min(ra - r0, r1 - ra, ca - c0, c1 - ca) < 1:
            continue
        if min(rb - r0, r1 - rb, cb - c0, c1 - cb) < 1:
            continue
        if int(clean[ra, ca]) == _BG or int(clean[rb, cb]) == _BG:
            continue
        if _same_neighbor_count(clean, ra, ca) < 2:
            continue
        if _same_neighbor_count(clean, rb, cb) < 2:
            continue
        pairs.append((pos_a, pos_b))
    _RNG.shuffle(pairs)
    return pairs


def _gen_attempt() -> np.ndarray | None:
    orientation = _RNG.choice(_ORIENTATIONS)
    n_pattern_colors = _RNG.randint(3, 5)
    palette = _random_palette(n_pattern_colors, exclude={_BG})
    marker_color = _random_palette(1, exclude={_BG, *palette})[0]

    patch = _make_symmetric_patch(orientation, palette)
    patch_h, patch_w = patch.shape

    pad_top = _RNG.randint(3, 4)
    pad_bottom = _RNG.randint(3, 4)
    pad_left = _RNG.randint(3, 4)
    pad_right = _RNG.randint(3, 4)
    grid_h = patch_h + pad_top + pad_bottom
    grid_w = patch_w + pad_left + pad_right

    clean = np.full((grid_h, grid_w), _BG, dtype=int)
    r0 = pad_top
    c0 = pad_left
    clean[r0 : r0 + patch_h, c0 : c0 + patch_w] = patch
    _place_marker(clean, orientation, marker_color)

    if np.count_nonzero(clean == _BG) <= clean.size // 2:
        return None

    bbox = (r0, c0, r0 + patch_h - 1, c0 + patch_w - 1)
    pairs = _candidate_pairs(clean, bbox, orientation)
    if not pairs:
        return None

    target_count = _RNG.randint(1, min(4, len(pairs)))
    polluted = clean.copy()
    used_positions: list[tuple[int, int]] = []
    placed = 0

    for pair in pairs:
        if placed >= target_count:
            break
        if not _pair_clear_of_used(pair, used_positions):
            continue

        pos_a, pos_b = pair
        sides = [(pos_a, pos_b), (pos_b, pos_a)]
        _RNG.shuffle(sides)
        committed = False

        for bad_pos, good_pos in sides:
            original = int(clean[bad_pos])
            for new_color in _pollution_options(polluted, bad_pos, original, palette, _BG):
                trial = polluted.copy()
                trial[bad_pos] = new_color
                bad_support = _same_neighbor_count(trial, *bad_pos)
                good_support = _same_neighbor_count(trial, *good_pos)
                if bad_support >= good_support:
                    continue
                polluted = trial
                used_positions.extend([pos_a, pos_b])
                placed += 1
                committed = True
                break
            if committed:
                break

    if placed != target_count:
        return None

    try:
        repaired = main(polluted)
    except AssertionError:
        return None
    if not np.array_equal(repaired, clean):
        return None
    return polluted


def generate_input() -> np.ndarray:
    for _ in range(_MAX_GENERATION_ATTEMPTS):
        grid = _gen_attempt()
        if grid is not None:
            return grid
    raise RuntimeError("failed to generate a valid symmetric-pollution ARC sample")
