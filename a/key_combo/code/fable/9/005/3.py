from __future__ import annotations

import itertools
import random

import numpy as np


BLACK = 0
BLUE = 1
GEN_ATTEMPTS = 400
PLACE_ATTEMPTS = 80


def _scale_sprite(sprite: np.ndarray, factor: int) -> np.ndarray:
    k = int(factor)
    if k <= 0:
        raise ValueError("scale factor must be positive")
    return np.repeat(np.repeat(np.asarray(sprite, dtype=int), k, axis=0), k, axis=1)


def _color_components(grid: np.ndarray, color: int) -> list[np.ndarray]:
    mask = np.asarray(grid == color, dtype=bool)
    h, w = mask.shape
    seen = np.zeros_like(mask, dtype=bool)
    comps: list[np.ndarray] = []
    for r in range(h):
        for c in range(w):
            if not mask[r, c] or seen[r, c]:
                continue
            stack = [(r, c)]
            seen[r, c] = True
            coords = []
            while stack:
                rr, cc = stack.pop()
                coords.append((rr, cc))
                for nr, nc in ((rr - 1, cc), (rr + 1, cc), (rr, cc - 1), (rr, cc + 1)):
                    if 0 <= nr < h and 0 <= nc < w and mask[nr, nc] and not seen[nr, nc]:
                        seen[nr, nc] = True
                        stack.append((nr, nc))
            comps.append(np.array(coords, dtype=int))
    return comps


def _find_frame(grid: np.ndarray) -> tuple[int, int, int, int]:
    g = np.asarray(grid, dtype=int)
    candidates = []
    for coords in _color_components(g, BLUE):
        r0, c0 = coords.min(axis=0)
        r1, c1 = coords.max(axis=0)
        height = int(r1 - r0 + 1)
        width = int(c1 - c0 + 1)
        if height < 3 or width < 3:
            continue
        if coords.shape[0] != 2 * (height + width) - 4:
            continue
        box = g[r0 : r1 + 1, c0 : c1 + 1]
        if (
            np.all(box[0, :] == BLUE)
            and np.all(box[-1, :] == BLUE)
            and np.all(box[:, 0] == BLUE)
            and np.all(box[:, -1] == BLUE)
            and np.all(box[1:-1, 1:-1] == BLACK)
        ):
            candidates.append((int(r0), int(c0), int(r1), int(c1)))

    assert len(candidates) == 1, "expected exactly one blue frame with black cavity"
    return candidates[0]


def _find_hidden_image(grid: np.ndarray, cavity_h: int, cavity_w: int) -> tuple[np.ndarray, int]:
    g = np.asarray(grid, dtype=int)
    colors = sorted(set(map(int, g.ravel())) - {BLUE})
    candidates = []
    for a, b in itertools.combinations(colors, 2):
        mask = np.isin(g, (a, b))
        rows = mask.any(axis=1)
        cols = mask.any(axis=0)
        h = int(rows.sum())
        w = int(cols.sum())
        if h == 0 or w == 0:
            continue
        if int(mask.sum()) != h * w:
            continue
        if cavity_h % h != 0 or cavity_w % w != 0:
            continue
        scale_h = cavity_h // h
        scale_w = cavity_w // w
        if scale_h != scale_w:
            continue
        sub = g[np.ix_(rows, cols)]
        values = set(map(int, sub.ravel()))
        if values != {a, b}:
            continue
        candidates.append((sub.copy(), int(scale_h)))

    assert len(candidates) == 1, "expected exactly one complete non-blue color pair"
    return candidates[0]


def main(grid: np.ndarray) -> np.ndarray:
    """
    Extract the unique complete two-color image and enlarge it to fill the blue frame cavity.

    Input structure:
    - The grid contains a single blue one-pixel-thick rectangular frame whose cavity is all black.
    - Elsewhere in the clutter, one non-blue color pair occupies every cell of the cross product of its active rows and active columns; compressing to those rows and columns reveals a hidden two-color image.
    - Another non-blue color pair is a decoy that is almost complete but misses one cell.

    Rule:
    - Find the blue frame and read its inner cavity size.
    - Among non-blue color pairs, find the unique pair whose pixels form a complete row-column product rectangle after deleting unused rows and columns.
    - Use the cavity dimensions to infer the integer nearest-neighbor scale factor, enlarge the hidden image by that factor, and place it inside the frame while keeping the blue border unchanged.

    Key insight:
    - The hidden picture is identified by structural completeness, not by color or location, and the frame cavity is the only ruler that reveals the required upscale factor.
    """
    g = np.asarray(grid, dtype=int)
    assert g.ndim == 2, "expected a 2D grid"

    r0, c0, r1, c1 = _find_frame(g)
    cavity_h = r1 - r0 - 1
    cavity_w = c1 - c0 - 1
    hidden, scale = _find_hidden_image(g, cavity_h, cavity_w)

    out = np.full((cavity_h + 2, cavity_w + 2), BLUE, dtype=int)
    out[1:-1, 1:-1] = _scale_sprite(hidden, scale)
    return out


def _make_connected_mask(h: int, w: int) -> np.ndarray:
    area = h * w
    target = random.randint(max(1, area // 4), min(area - 1, max(2, (3 * area) // 5)))
    seed = (random.randrange(h), random.randrange(w))
    coords = {seed}
    while len(coords) < target:
        frontier = []
        for r, c in coords:
            for nr, nc in ((r - 1, c), (r + 1, c), (r, c - 1), (r, c + 1)):
                if 0 <= nr < h and 0 <= nc < w and (nr, nc) not in coords:
                    frontier.append((nr, nc))
        coords.add(random.choice(frontier))
    mask = np.zeros((h, w), dtype=bool)
    pts = np.array(tuple(coords), dtype=int)
    mask[pts[:, 0], pts[:, 1]] = True
    return mask


def _make_binary_pattern(h: int, w: int) -> np.ndarray:
    branch = random.randrange(3)
    rr, cc = np.indices((h, w))
    if branch == 0:
        offset = random.randrange(2)
        mask = (rr + cc + offset) % 2 == 0
    elif branch == 1:
        mode = random.randrange(3)
        band = random.randint(1, 2)
        if mode == 0:
            mask = ((rr // band) + random.randrange(2)) % 2 == 0
        elif mode == 1:
            mask = ((cc // band) + random.randrange(2)) % 2 == 0
        else:
            mask = (((rr + cc) // band) + random.randrange(2)) % 2 == 0
    else:
        mask = _make_connected_mask(h, w)

    if mask.all() or (~mask).all():
        mask = _make_connected_mask(h, w)
    return mask


def _make_image(h: int, w: int, colors: tuple[int, int]) -> np.ndarray:
    a, b = colors
    mask = _make_binary_pattern(h, w)
    return np.where(mask, a, b).astype(int)


def _sparse_positions(count: int) -> list[int]:
    if count == 1:
        return [0]
    gaps = [random.randint(0, 2) for _ in range(count - 1)]
    if max(gaps) == 0:
        gaps[random.randrange(count - 1)] = 1
    pos = [0]
    for gap in gaps:
        pos.append(pos[-1] + 1 + gap)
    return pos


def _boxes_overlap(a: tuple[int, int, int, int], b: tuple[int, int, int, int], pad: int = 0) -> bool:
    return not (
        a[2] + pad < b[0]
        or b[2] + pad < a[0]
        or a[3] + pad < b[1]
        or b[3] + pad < a[1]
    )


def _place_box(
    grid_h: int,
    grid_w: int,
    box_h: int,
    box_w: int,
    taken: list[tuple[int, int, int, int]],
    pad: int = 1,
) -> tuple[int, int] | None:
    if box_h > grid_h or box_w > grid_w:
        return None
    for _ in range(PLACE_ATTEMPTS):
        r0 = random.randint(0, grid_h - box_h)
        c0 = random.randint(0, grid_w - box_w)
        box = (r0, c0, r0 + box_h - 1, c0 + box_w - 1)
        if all(not _boxes_overlap(box, other, pad) for other in taken):
            return r0, c0
    return None


def _full_and_almost_pairs(grid: np.ndarray) -> tuple[list[tuple[int, int]], list[tuple[int, int]]]:
    g = np.asarray(grid, dtype=int)
    colors = sorted(set(map(int, g.ravel())) - {BLUE})
    full = []
    almost = []
    for a, b in itertools.combinations(colors, 2):
        mask = np.isin(g, (a, b))
        rows = mask.any(axis=1)
        cols = mask.any(axis=0)
        h = int(rows.sum())
        w = int(cols.sum())
        if h == 0 or w == 0:
            continue
        cells = int(mask.sum())
        if cells == h * w:
            full.append((a, b))
        elif h >= 2 and w >= 2 and cells == h * w - 1:
            sub = g[np.ix_(rows, cols)]
            miss = int(np.sum(~np.isin(sub, (a, b))))
            if miss == 1:
                almost.append((a, b))
    return full, almost


def _choose_dims_and_scale() -> tuple[int, int, int]:
    options = []
    for h in range(2, 6):
        for w in range(2, 6):
            for scale in (2, 3, 4):
                if scale * h + 2 <= 16 and scale * w + 2 <= 16:
                    options.append((h, w, scale))
    return random.choice(options)


def _apply_sparse_image(
    grid: np.ndarray,
    top: int,
    left: int,
    rows: list[int],
    cols: list[int],
    image: np.ndarray,
) -> None:
    for i, rr in enumerate(rows):
        for j, cc in enumerate(cols):
            grid[top + rr, left + cc] = int(image[i, j])


def _make_filler(grid_h: int, grid_w: int, colors: tuple[int, int, int]) -> np.ndarray:
    coeffs = random.choice(((1, 1), (1, 2), (2, 1), (2, 2)))
    offset = random.randrange(3)
    rr, cc = np.indices((grid_h, grid_w))
    idx = (coeffs[0] * rr + coeffs[1] * cc + offset) % 3
    palette = np.array(colors, dtype=int)
    return palette[idx]


def _draw_frame(grid: np.ndarray, top: int, left: int, cavity_h: int, cavity_w: int) -> None:
    bottom = top + cavity_h + 1
    right = left + cavity_w + 1
    grid[top : bottom + 1, left : right + 1] = BLUE
    grid[top + 1 : bottom, left + 1 : right] = BLACK


def _gen_attempt() -> np.ndarray | None:
    hidden_h, hidden_w, scale = _choose_dims_and_scale()
    cavity_h = hidden_h * scale
    cavity_w = hidden_w * scale
    frame_h = cavity_h + 2
    frame_w = cavity_w + 2

    color_pool = [2, 3, 4, 5, 6, 7, 8, 9]
    chosen = random.sample(color_pool, 7)
    hidden_colors = (chosen[0], chosen[1])
    decoy_colors = (chosen[2], chosen[3])
    filler_colors = (chosen[4], chosen[5], chosen[6])

    hidden_rows = _sparse_positions(hidden_h)
    hidden_cols = _sparse_positions(hidden_w)
    hidden_box_h = hidden_rows[-1] + 1
    hidden_box_w = hidden_cols[-1] + 1

    decoy_h = random.randint(2, 5)
    decoy_w = random.randint(2, 5)
    decoy_rows = _sparse_positions(decoy_h)
    decoy_cols = _sparse_positions(decoy_w)
    decoy_box_h = decoy_rows[-1] + 1
    decoy_box_w = decoy_cols[-1] + 1

    min_h = max(18, frame_h, hidden_box_h, decoy_box_h)
    min_w = max(18, frame_w, hidden_box_w, decoy_box_w)
    if min_h > 30 or min_w > 30:
        return None
    grid_h = random.randint(min_h, 30)
    grid_w = random.randint(min_w, 30)

    taken: list[tuple[int, int, int, int]] = []
    frame_loc = _place_box(grid_h, grid_w, frame_h, frame_w, taken, pad=1)
    if frame_loc is None:
        return None
    fr, fc = frame_loc
    taken.append((fr, fc, fr + frame_h - 1, fc + frame_w - 1))

    hidden_loc = _place_box(grid_h, grid_w, hidden_box_h, hidden_box_w, taken, pad=1)
    if hidden_loc is None:
        return None
    hr, hc = hidden_loc
    taken.append((hr, hc, hr + hidden_box_h - 1, hc + hidden_box_w - 1))

    decoy_loc = _place_box(grid_h, grid_w, decoy_box_h, decoy_box_w, taken, pad=1)
    if decoy_loc is None:
        return None
    dr, dc = decoy_loc

    grid = _make_filler(grid_h, grid_w, filler_colors)

    hidden_image = _make_image(hidden_h, hidden_w, hidden_colors)
    _apply_sparse_image(grid, hr, hc, hidden_rows, hidden_cols, hidden_image)

    decoy_image = _make_image(decoy_h, decoy_w, decoy_colors)
    hole_r = random.randrange(decoy_h)
    hole_c = random.randrange(decoy_w)
    for i, rr in enumerate(decoy_rows):
        for j, cc in enumerate(decoy_cols):
            if i == hole_r and j == hole_c:
                continue
            grid[dr + rr, dc + cc] = int(decoy_image[i, j])

    _draw_frame(grid, fr, fc, cavity_h, cavity_w)

    full_pairs, almost_pairs = _full_and_almost_pairs(grid)
    hidden_pair = tuple(sorted(hidden_colors))
    decoy_pair = tuple(sorted(decoy_colors))
    if sorted(map(tuple, map(sorted, full_pairs))) != [hidden_pair]:
        return None
    if decoy_pair not in [tuple(sorted(pair)) for pair in almost_pairs]:
        return None
    if len({tuple(sorted(pair)) for pair in almost_pairs}) != 1:
        return None

    try:
        solved = main(grid)
    except AssertionError:
        return None

    expected = np.full((frame_h, frame_w), BLUE, dtype=int)
    expected[1:-1, 1:-1] = _scale_sprite(hidden_image, scale)
    if not np.array_equal(solved, expected):
        return None
    return grid


def generate_input() -> np.ndarray:
    for _ in range(GEN_ATTEMPTS):
        grid = _gen_attempt()
        if grid is not None:
            return grid
    raise RuntimeError("failed to generate a valid input")
