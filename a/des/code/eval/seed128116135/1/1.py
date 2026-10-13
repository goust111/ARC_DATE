import random
from collections import Counter

import numpy as np


RED = 2
GREEN = 3
GREY = 5
COLOR_POOL = (0, 1, 4, 6, 7, 8, 9)
MAX_ATTEMPTS = 400


def _get_bg(grid: np.ndarray) -> int:
    return Counter(map(int, grid.flat)).most_common(1)[0][0]


def _orient_from_canonical(arr: np.ndarray, quad: tuple[int, int]) -> np.ndarray:
    if quad == (0, 0):
        return arr.copy()
    if quad == (0, 1):
        return np.fliplr(arr)
    if quad == (1, 0):
        return np.flipud(arr)
    if quad == (1, 1):
        return np.flipud(np.fliplr(arr))
    raise AssertionError(f"invalid quadrant {quad}")


def _find_dividers(grid: np.ndarray) -> tuple[int, int]:
    h, w = grid.shape
    row_hits = [r for r in range(h) if np.all(grid[r, :] == GREY)]
    col_hits = [c for c in range(w) if np.all(grid[:, c] == GREY)]
    assert len(row_hits) == 1 and len(col_hits) == 1, "expected one full grey row and one full grey column"
    mid_r = row_hits[0]
    mid_c = col_hits[0]
    assert mid_r == h - mid_r - 1 and mid_c == w - mid_c - 1, "divider must split the grid evenly"
    return mid_r, mid_c


def _extract_quadrants(grid: np.ndarray, mid_r: int, mid_c: int) -> dict[tuple[int, int], np.ndarray]:
    return {
        (0, 0): grid[:mid_r, :mid_c],
        (0, 1): grid[:mid_r, mid_c + 1 :],
        (1, 0): grid[mid_r + 1 :, :mid_c],
        (1, 1): grid[mid_r + 1 :, mid_c + 1 :],
    }


def _paste_quadrant(grid: np.ndarray, mid_r: int, mid_c: int, quad: tuple[int, int], patch: np.ndarray) -> None:
    if quad == (0, 0):
        grid[:mid_r, :mid_c] = patch
    elif quad == (0, 1):
        grid[:mid_r, mid_c + 1 :] = patch
    elif quad == (1, 0):
        grid[mid_r + 1 :, :mid_c] = patch
    elif quad == (1, 1):
        grid[mid_r + 1 :, mid_c + 1 :] = patch
    else:
        raise AssertionError(f"invalid quadrant {quad}")


def _orient_pos_from_canonical(pos: tuple[int, int], shape: tuple[int, int], quad: tuple[int, int]) -> tuple[int, int]:
    r, c = pos
    h, w = shape
    if quad == (0, 0):
        return r, c
    if quad == (0, 1):
        return r, w - 1 - c
    if quad == (1, 0):
        return h - 1 - r, c
    if quad == (1, 1):
        return h - 1 - r, w - 1 - c
    raise AssertionError(f"invalid quadrant {quad}")


def main(grid: np.ndarray) -> np.ndarray:
    g = np.asarray(grid, dtype=int)
    assert g.ndim == 2
    bg = _get_bg(g)
    assert bg not in (RED, GREEN, GREY)

    mid_r, mid_c = _find_dividers(g)
    quadrants = _extract_quadrants(g, mid_r, mid_c)

    empties = [quad for quad, patch in quadrants.items() if np.all(patch == bg)]
    assert len(empties) == 1, "expected exactly one empty quadrant"
    empty_quad = empties[0]

    canonical_views = []
    for quad, patch in quadrants.items():
        if quad == empty_quad:
            continue
        canonical_views.append(_orient_from_canonical(patch, quad))

    stack = np.stack(canonical_views, axis=0)
    qh, qw = stack.shape[1:]

    base = np.empty((qh, qw), dtype=int)
    marker_pos = None
    total_red = 0
    for r in range(qh):
        for c in range(qw):
            vals = [int(v) for v in stack[:, r, c]]
            cnt = Counter(vals)
            total_red += cnt.get(RED, 0)
            if len(cnt) == 1:
                base[r, c] = vals[0]
                continue
            assert len(cnt) == 2 and RED in cnt and cnt[RED] == 1, "only one transformed red marker is allowed"
            majority = [color for color, amount in cnt.items() if amount == 2]
            assert len(majority) == 1 and majority[0] != RED
            assert marker_pos is None, "expected a unique marker position"
            marker_pos = (r, c)
            base[r, c] = majority[0]

    assert total_red == 1, "input must contain exactly one red marker cell"
    assert marker_pos is not None, "marker position not found"

    fill = base.copy()
    fill[marker_pos] = GREEN
    restored = _orient_from_canonical(fill, empty_quad)

    out = g.copy()
    _paste_quadrant(out, mid_r, mid_c, empty_quad, restored)
    return out


def _neighbors(cell: tuple[int, int], h: int, w: int) -> list[tuple[int, int]]:
    r, c = cell
    out = []
    if r > 0:
        out.append((r - 1, c))
    if r + 1 < h:
        out.append((r + 1, c))
    if c > 0:
        out.append((r, c - 1))
    if c + 1 < w:
        out.append((r, c + 1))
    return out


def _grow_connected_mask(h: int, w: int, area: int) -> np.ndarray:
    cells = {(random.randrange(h), random.randrange(w))}
    frontier = set(_neighbors(next(iter(cells)), h, w))
    while len(cells) < area:
        if not frontier:
            return np.zeros((h, w), dtype=bool)
        cell = random.choice(tuple(frontier))
        frontier.remove(cell)
        cells.add(cell)
        for nb in _neighbors(cell, h, w):
            if nb not in cells:
                frontier.add(nb)
    mask = np.zeros((h, w), dtype=bool)
    for r, c in cells:
        mask[r, c] = True
    return mask


def _colorize_mask(mask: np.ndarray, colors: tuple[int, ...]) -> np.ndarray:
    coords = [tuple(map(int, rc)) for rc in np.argwhere(mask)]
    seeds = random.sample(coords, len(colors))
    patch = np.zeros(mask.shape, dtype=int)
    for r, c in coords:
        idx = min(
            range(len(seeds)),
            key=lambda i: (abs(r - seeds[i][0]) + abs(c - seeds[i][1]), i),
        )
        patch[r, c] = colors[idx]
    return patch


def _build_canonical_motif(qh: int, qw: int, bg: int) -> tuple[np.ndarray, tuple[int, int]] | None:
    box_h = random.randint(max(3, qh - 4), qh)
    box_w = random.randint(max(3, qw - 4), qw)
    r0 = random.randint(0, qh - box_h)
    c0 = random.randint(0, qw - box_w)

    area_lo = max(5, (box_h * box_w) // 5)
    area_hi = max(area_lo, (box_h * box_w * 2) // 5)
    area = random.randint(area_lo, area_hi)
    if area >= box_h * box_w:
        area = box_h * box_w - 1
    if area < 5:
        return None

    inner_mask = _grow_connected_mask(box_h, box_w, area)
    if inner_mask.sum() != area:
        return None

    rows, cols = np.where(inner_mask)
    if np.ptp(rows) < 1 or np.ptp(cols) < 1:
        return None

    motif = np.full((qh, qw), bg, dtype=int)
    body_colors = tuple(random.sample([c for c in COLOR_POOL if c != bg], random.randint(2, 4)))
    motif[r0 : r0 + box_h, c0 : c0 + box_w] = np.where(
        inner_mask,
        _colorize_mask(inner_mask, body_colors),
        bg,
    )

    if np.array_equal(motif, np.fliplr(motif)) or np.array_equal(motif, np.flipud(motif)):
        return None

    coords = [tuple(map(int, rc)) for rc in np.argwhere(motif != bg)]
    marker_pos = random.choice(coords)
    return motif, marker_pos


def _expected_output_from_parts(
    grid: np.ndarray,
    empty_quad: tuple[int, int],
    motif: np.ndarray,
    marker_pos: tuple[int, int],
) -> np.ndarray:
    out = grid.copy()
    mid_r, mid_c = _find_dividers(grid)
    fill = motif.copy()
    fill[marker_pos] = GREEN
    _paste_quadrant(out, mid_r, mid_c, empty_quad, _orient_from_canonical(fill, empty_quad))
    return out


def _gen_attempt() -> np.ndarray | None:
    qh = random.randint(5, 9)
    qw = random.randint(5, 9)
    bg = random.choice(COLOR_POOL)

    built = _build_canonical_motif(qh, qw, bg)
    if built is None:
        return None
    motif, marker_pos = built

    h = 2 * qh + 1
    w = 2 * qw + 1
    grid = np.full((h, w), bg, dtype=int)
    grid[qh, :] = GREY
    grid[:, qw] = GREY

    all_quads = [(0, 0), (0, 1), (1, 0), (1, 1)]
    empty_quad = random.choice(all_quads)
    non_empty = [quad for quad in all_quads if quad != empty_quad]
    marker_quad = random.choice(non_empty)

    for quad in non_empty:
        patch = _orient_from_canonical(motif, quad)
        _paste_quadrant(grid, qh, qw, quad, patch)

    mr, mc = _orient_pos_from_canonical(marker_pos, (qh, qw), marker_quad)
    target = _extract_quadrants(grid, qh, qw)[marker_quad].copy()
    target[mr, mc] = RED
    _paste_quadrant(grid, qh, qw, marker_quad, target)

    if int((grid == RED).sum()) != 1:
        return None

    try:
        predicted = main(grid)
    except AssertionError:
        return None

    expected = _expected_output_from_parts(grid, empty_quad, motif, marker_pos)
    if not np.array_equal(predicted, expected):
        return None
    if np.array_equal(predicted, grid):
        return None
    return grid


def generate_input() -> np.ndarray:
    for _ in range(MAX_ATTEMPTS):
        grid = _gen_attempt()
        if grid is not None:
            return grid
    raise RuntimeError("failed to generate a valid input")
