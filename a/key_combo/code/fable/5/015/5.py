from __future__ import annotations

import random
from collections import Counter, deque

import numpy as np


BLACK = 0
PALETTE = tuple(range(1, 10))
GENERATION_ATTEMPTS = 400


def _components(mask: np.ndarray, connectivity: int = 8) -> list[np.ndarray]:
    """Return connected components as arrays of (row, column) coordinates."""
    h, w = mask.shape
    seen = np.zeros_like(mask, dtype=bool)
    if connectivity == 8:
        steps = [(dr, dc) for dr in (-1, 0, 1) for dc in (-1, 0, 1)
                 if (dr, dc) != (0, 0)]
    else:
        steps = [(-1, 0), (1, 0), (0, -1), (0, 1)]

    result: list[np.ndarray] = []
    for r, c in np.argwhere(mask):
        r, c = int(r), int(c)
        if seen[r, c]:
            continue
        seen[r, c] = True
        queue = deque([(r, c)])
        cells: list[tuple[int, int]] = []
        while queue:
            rr, cc = queue.popleft()
            cells.append((rr, cc))
            for dr, dc in steps:
                nr, nc = rr + dr, cc + dc
                if (0 <= nr < h and 0 <= nc < w and mask[nr, nc]
                        and not seen[nr, nc]):
                    seen[nr, nc] = True
                    queue.append((nr, nc))
        result.append(np.asarray(cells, dtype=int))
    return result


def _bbox(coords: np.ndarray) -> tuple[int, int, int, int]:
    """Inclusive bbox of (row, column) coordinates."""
    lo = coords.min(axis=0)
    hi = coords.max(axis=0)
    return int(lo[0]), int(lo[1]), int(hi[0]), int(hi[1])


def _rectangle_perimeter(h: int, w: int) -> np.ndarray:
    mask = np.zeros((h, w), dtype=bool)
    mask[0, :] = mask[-1, :] = True
    mask[:, 0] = mask[:, -1] = True
    return mask


def _find_target(g: np.ndarray, bg: int) -> tuple[int, int, int, int]:
    """Find the unique large monochrome frame with an empty interior."""
    candidates: list[tuple[int, int, int, int]] = []
    for color in set(map(int, g.ravel())) - {int(bg)}:
        for coords in _components(g == color, connectivity=8):
            r0, c0, r1, c1 = _bbox(coords)
            h, w = r1 - r0 + 1, c1 - c0 + 1
            if h < 7 or w < 7:
                continue
            sub = g[r0:r1 + 1, c0:c1 + 1]
            perimeter = _rectangle_perimeter(h, w)
            if (coords.shape[0] == int(perimeter.sum())
                    and np.all(sub[perimeter] == color)
                    and np.all(sub[~perimeter] == bg)):
                candidates.append((r0, c0, r1, c1))
    assert len(candidates) == 1, "expected one uniquely identifiable empty target frame"
    return candidates[0]


def _find_legend(g: np.ndarray) -> tuple[tuple[int, int, int, int], list[int]]:
    """Find the black legend patch and read keys down its sole non-black column."""
    candidates: list[tuple[tuple[int, int, int, int], list[int]]] = []
    for coords in _components(g == BLACK, connectivity=8):
        r0, c0, r1, c1 = _bbox(coords)
        h, w = r1 - r0 + 1, c1 - c0 + 1
        if h < 5 or w < 4:
            continue
        sub = g[r0:r1 + 1, c0:c1 + 1]
        nonblack = np.argwhere(sub != BLACK)
        if nonblack.shape[0] < 2:
            continue
        columns = set(map(int, nonblack[:, 1]))
        if len(columns) != 1:
            continue
        key_col = next(iter(columns))
        if not (0 < key_col < w - 1):
            continue
        keys = [int(sub[r, key_col]) for r in range(h)
                if int(sub[r, key_col]) != BLACK]
        if len(keys) == len(set(keys)) and all(color != BLACK for color in keys):
            candidates.append(((r0, c0, r1, c1), keys))
    assert len(candidates) == 1, "expected one uniquely identifiable black legend"
    return candidates[0]


def _counts_outside_structures(
    g: np.ndarray,
    keys: list[int],
    legend: tuple[int, int, int, int],
    target: tuple[int, int, int, int],
) -> list[int]:
    outside = np.ones(g.shape, dtype=bool)
    for r0, c0, r1, c1 in (legend, target):
        outside[r0:r1 + 1, c0:c1 + 1] = False
    counts = [len(_components((g == color) & outside, connectivity=8))
              for color in keys]
    assert all(count > 0 for count in counts), "every key color needs an outside object"
    return counts


def _band_color(depth: int, keys: list[int], counts: list[int]) -> int:
    position = int(depth) % sum(counts)
    for color, thickness in zip(keys, counts):
        if position < thickness:
            return int(color)
        position -= thickness
    raise AssertionError("unreachable band position")


def main(grid: np.ndarray) -> np.ndarray:
    """
    Count outside key-color objects and paint their counts as repeating concentric band thicknesses.

    Input structure:
    - A dominant-color background contains one black legend patch whose only
      non-black cells are distinct key colors in one column, read top to bottom.
    - A separate one-cell monochrome rectangular frame encloses an entirely
      background-colored target region.
    - Outside both rectangles, mutually separated monochrome objects use the
      key colors.

    Rule:
    - Count the 8-connected objects of each key color outside the legend and
      target rectangles.  Starting at the target frame's inner boundary, paint
      the key colors in legend row order, using each count as that color's band
      thickness.  Repeat the thickness sequence inward and truncate the final
      band when the center is reached; preserve every cell outside the interior.

    Key insight:
    - Legend position supplies band order while component count independently
      supplies band thickness, so both clues are required to fill the target.
    """
    g = np.asarray(grid, dtype=int)
    assert g.ndim == 2, "expected a two-dimensional ARC grid"
    bg = int(Counter(map(int, g.ravel())).most_common(1)[0][0])
    assert bg != BLACK, "black is reserved for the legend"

    target = _find_target(g, bg)
    legend, keys = _find_legend(g)
    assert set(keys).isdisjoint({bg}), "key colors must differ from background"
    counts = _counts_outside_structures(g, keys, legend, target)

    out = g.copy()
    r0, c0, r1, c1 = target
    for r in range(r0 + 1, r1):
        for c in range(c0 + 1, c1):
            depth = min(r - (r0 + 1), (r1 - 1) - r,
                        c - (c0 + 1), (c1 - 1) - c)
            out[r, c] = _band_color(depth, keys, counts)
    return out


# ------------------------------- generator ---------------------------------

_SPRITES = (
    np.array([[1]], dtype=int),
    np.array([[1, 1]], dtype=int),
    np.array([[1], [1]], dtype=int),
    np.array([[1, 1], [1, 0]], dtype=int),
    np.array([[1, 1], [0, 1]], dtype=int),
    np.array([[1, 0], [1, 1]], dtype=int),
    np.array([[0, 1], [1, 1]], dtype=int),
)


def _sample_thickness_plan(rng: random.Random) -> tuple[list[int], int, bool]:
    """Choose counts and target depth, covering exact and truncated centers."""
    truncate = bool(rng.getrandbits(1))
    for _ in range(100):
        k = rng.choice((2, 3, 4))
        counts = [rng.choice((1, 2)) for _ in range(k)]
        cycle = sum(counts)
        if cycle > 6 or 2 not in counts:
            continue
        if truncate:
            choices = []
            prefix = 0
            for width in counts:
                if width == 2 and cycle + prefix + 1 <= 8:
                    choices.append(cycle + prefix + 1)
                prefix += width
        else:
            prefix = 0
            choices = []
            for width in counts:
                prefix += width
                if cycle + prefix <= 8:
                    choices.append(cycle + prefix)
        if choices:
            return counts, rng.choice(choices), truncate
    raise RuntimeError("could not sample a valid thickness plan")


def _place_structures(
    rng: random.Random,
    target_shape: tuple[int, int],
    legend_shape: tuple[int, int],
) -> tuple[tuple[int, int], tuple[int, int], tuple[int, int]]:
    """Lay out the two rectangles side-by-side or stacked with visible spacing."""
    th, tw = target_shape
    lh, lw = legend_shape
    gap = 3
    horizontal_min = tw + lw + gap + 2
    vertical_min = th + lh + gap + 2
    orientations = []
    if horizontal_min <= 30:
        orientations.append(True)
    if vertical_min <= 30:
        orientations.append(False)
    if not orientations:
        raise ValueError("structures do not fit in an ARC-sized canvas")
    horizontal = rng.choice(orientations)
    if horizontal:
        h = max(th, lh) + 2 + rng.randint(1, max(1, 28 - max(th, lh)))
        w = tw + lw + gap + 2 + rng.randint(0, max(0, 30 - (tw + lw + gap + 2)))
        h = min(h, 30)
        target_first = bool(rng.getrandbits(1))
        left_shape, right_shape = ((th, tw), (lh, lw)) if target_first else ((lh, lw), (th, tw))
        left_r = rng.randint(1, h - left_shape[0] - 1)
        right_r = rng.randint(1, h - right_shape[0] - 1)
        slack = w - (left_shape[1] + gap + right_shape[1])
        left_c = rng.randint(1, max(1, slack - 1))
        right_c = left_c + left_shape[1] + gap
        if target_first:
            return (h, w), (left_r, left_c), (right_r, right_c)
        return (h, w), (right_r, right_c), (left_r, left_c)

    h = th + lh + gap + 2 + rng.randint(0, max(0, 30 - (th + lh + gap + 2)))
    w = max(tw, lw) + 2 + rng.randint(1, max(1, 28 - max(tw, lw)))
    h, w = min(h, 30), min(w, 30)
    target_first = bool(rng.getrandbits(1))
    top_shape, bottom_shape = ((th, tw), (lh, lw)) if target_first else ((lh, lw), (th, tw))
    top_c = rng.randint(1, w - top_shape[1] - 1)
    bottom_c = rng.randint(1, w - bottom_shape[1] - 1)
    slack = h - (top_shape[0] + gap + bottom_shape[0])
    top_r = rng.randint(1, max(1, slack - 1))
    bottom_r = top_r + top_shape[0] + gap
    if target_first:
        return (h, w), (top_r, top_c), (bottom_r, bottom_c)
    return (h, w), (bottom_r, bottom_c), (top_r, top_c)


def _dilate(mask: np.ndarray) -> np.ndarray:
    padded = np.pad(mask, 1, constant_values=False)
    out = np.zeros_like(mask)
    for dr in range(3):
        for dc in range(3):
            out |= padded[dr:dr + mask.shape[0], dc:dc + mask.shape[1]]
    return out


def _place_objects(
    rng: random.Random,
    grid: np.ndarray,
    occupied: np.ndarray,
    keys: list[int],
    counts: list[int],
) -> tuple[bool, list[int]]:
    """Place 8-separated connected sprites; return their per-color pixel totals."""
    queue: list[tuple[int, np.ndarray]] = []
    for color, count in zip(keys, counts):
        # Every color has a multi-cell object, defeating pixel-count shortcuts.
        shapes = list(_SPRITES[1:])
        chosen = [rng.choice(shapes)]
        chosen.extend(rng.choice(_SPRITES) for _ in range(count - 1))
        queue.extend((color, shape.copy()) for shape in chosen)
    rng.shuffle(queue)

    pixel_totals = {color: 0 for color in keys}
    for color, sprite in queue:
        sh, sw = sprite.shape
        candidates: list[tuple[int, int]] = []
        for r in range(1, grid.shape[0] - sh):
            for c in range(1, grid.shape[1] - sw):
                local = sprite.astype(bool)
                if np.any(occupied[r:r + sh, c:c + sw] & local):
                    continue
                trial = np.zeros_like(occupied)
                trial[r:r + sh, c:c + sw] = local
                if np.any(_dilate(trial) & occupied):
                    continue
                candidates.append((r, c))
        if not candidates:
            return False, []
        r, c = rng.choice(candidates)
        local = sprite.astype(bool)
        grid[r:r + sh, c:c + sw][local] = color
        occupied[r:r + sh, c:c + sw] |= local
        pixel_totals[color] += int(local.sum())
    return True, [pixel_totals[color] for color in keys]


def _expected_fill(
    grid: np.ndarray,
    target: tuple[int, int, int, int],
    keys: list[int],
    counts: list[int],
) -> np.ndarray:
    """Generator oracle expressed with cumulative depth intervals."""
    out = grid.copy()
    cycle = sum(counts)
    ends = np.cumsum(counts)
    r0, c0, r1, c1 = target
    rows = np.arange(r1 - r0 - 1)[:, None]
    cols = np.arange(c1 - c0 - 1)[None, :]
    depth = np.minimum(np.minimum(rows, rows[::-1]), np.minimum(cols, cols[:, ::-1]))
    phase = depth % cycle
    interior = np.empty(depth.shape, dtype=int)
    start = 0
    for color, end in zip(keys, ends):
        interior[(phase >= start) & (phase < end)] = color
        start = int(end)
    out[r0 + 1:r1, c0 + 1:c1] = interior
    return out


def _gen_attempt(rng: random.Random) -> np.ndarray | None:
    counts, depth_layers, truncated = _sample_thickness_plan(rng)
    k = len(counts)

    colors = list(PALETTE)
    bg = rng.choice(colors)
    colors.remove(bg)
    keys = rng.sample(colors, k)
    for color in keys:
        colors.remove(color)
    border = rng.choice(colors)

    min_interior = 2 * depth_layers - rng.choice((0, 1))
    extra = rng.randint(0, 2)
    if rng.getrandbits(1):
        interior_shape = (min_interior, min_interior + extra)
    else:
        interior_shape = (min_interior + extra, min_interior)
    target_shape = (interior_shape[0] + 2, interior_shape[1] + 2)
    legend_shape = (2 * k + 1, rng.choice((4, 5)))

    try:
        canvas_shape, target_pos, legend_pos = _place_structures(
            rng, target_shape, legend_shape)
    except ValueError:
        return None
    if max(canvas_shape) > 30:
        return None
    grid = np.full(canvas_shape, bg, dtype=int)
    occupied = np.zeros(canvas_shape, dtype=bool)

    tr, tc = target_pos
    th, tw = target_shape
    target_mask = _rectangle_perimeter(th, tw)
    target_patch = np.full(target_shape, bg, dtype=int)
    target_patch[target_mask] = border
    grid[tr:tr + th, tc:tc + tw] = target_patch
    occupied[tr:tr + th, tc:tc + tw] = True
    target_bbox = (tr, tc, tr + th - 1, tc + tw - 1)

    lr, lc = legend_pos
    lh, lw = legend_shape
    legend_patch = np.full(legend_shape, BLACK, dtype=int)
    for i, color in enumerate(keys):
        legend_patch[1 + 2 * i, 1] = color
    grid[lr:lr + lh, lc:lc + lw] = legend_patch
    occupied[lr:lr + lh, lc:lc + lw] = True
    legend_bbox = (lr, lc, lr + lh - 1, lc + lw - 1)

    placed, pixel_totals = _place_objects(rng, grid, occupied, keys, counts)
    if not placed or any(pixels <= count for pixels, count in zip(pixel_totals, counts)):
        return None

    # Independent semantic checks, not merely main(generate_input()) closure.
    if Counter(map(int, grid.ravel())).most_common(1)[0][0] != bg:
        return None
    try:
        found_target = _find_target(grid, bg)
        found_legend, found_keys = _find_legend(grid)
        found_counts = _counts_outside_structures(grid, found_keys, found_legend, found_target)
    except AssertionError:
        return None
    if (found_target != target_bbox or found_legend != legend_bbox
            or found_keys != keys or found_counts != counts):
        return None

    expected = _expected_fill(grid, target_bbox, keys, counts)
    try:
        solved = main(grid)
    except (AssertionError, ValueError):
        return None
    if not np.array_equal(solved, expected) or np.array_equal(solved, grid):
        return None

    # Both legend order and every thickness must materially affect the target.
    reversed_fill = _expected_fill(grid, target_bbox, list(reversed(keys)), list(reversed(counts)))
    if np.array_equal(expected, reversed_fill):
        return None
    for i in range(k):
        altered = counts.copy()
        altered[i] += 1
        if np.array_equal(expected, _expected_fill(grid, target_bbox, keys, altered)):
            return None

    cycle = sum(counts)
    if depth_layers <= cycle:
        return None
    remainder = depth_layers - cycle
    boundaries = set(np.cumsum(counts).tolist())
    if truncated != (remainder not in boundaries):
        return None
    return grid


def generate_input() -> np.ndarray:
    """Generate a diverse, structurally unambiguous instance of the puzzle."""
    rng = random.Random()
    for _ in range(GENERATION_ATTEMPTS):
        grid = _gen_attempt(rng)
        if grid is not None:
            return grid
    raise RuntimeError(f"failed to generate a valid instance after {GENERATION_ATTEMPTS} attempts")
