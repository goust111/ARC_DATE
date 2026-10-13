from collections import Counter, deque
from typing import Iterable

import numpy as np


BG = 0
FILL_SEQUENCE = [2, 3, 4, 7, 6]  # red, green, yellow, orange, magenta
INPUT_FG_CHOICES = [1, 5, 8, 9]
DIR4 = [(-1, 0), (1, 0), (0, -1), (0, 1)]

_RNG = np.random.default_rng()

GEN_ATTEMPTS = 400
GRID_SIDE_MIN = 15
GRID_SIDE_MAX = 30
RING_COUNT_VALUES = [2, 3, 4, 5]
RING_COUNT_PROBS = [0.16, 0.24, 0.25, 0.35]
RING_PAD = 1
DISTRACTOR_PAD = 1


def _get_bg(grid: np.ndarray) -> int:
    return Counter(int(v) for v in grid.ravel()).most_common(1)[0][0]


def _components(mask: np.ndarray) -> list[list[tuple[int, int]]]:
    h, w = mask.shape
    seen = np.zeros_like(mask, dtype=bool)
    comps: list[list[tuple[int, int]]] = []
    for r in range(h):
        for c in range(w):
            if not bool(mask[r, c]) or bool(seen[r, c]):
                continue
            q = deque([(r, c)])
            seen[r, c] = True
            comp: list[tuple[int, int]] = []
            while q:
                cr, cc = q.popleft()
                comp.append((cr, cc))
                for dr, dc in DIR4:
                    nr, nc = cr + dr, cc + dc
                    if 0 <= nr < h and 0 <= nc < w and bool(mask[nr, nc]) and not bool(seen[nr, nc]):
                        seen[nr, nc] = True
                        q.append((nr, nc))
            comps.append(comp)
    return comps


def _enclosed_background_regions(grid: np.ndarray, bg: int) -> list[list[tuple[int, int]]]:
    h, w = grid.shape
    enclosed: list[list[tuple[int, int]]] = []
    for comp in _components(grid == bg):
        if any(r == 0 or c == 0 or r == h - 1 or c == w - 1 for r, c in comp):
            continue
        has_boundary = False
        for r, c in comp:
            for dr, dc in DIR4:
                nr, nc = r + dr, c + dc
                if 0 <= nr < h and 0 <= nc < w and int(grid[nr, nc]) != bg:
                    has_boundary = True
                    break
            if has_boundary:
                break
        if has_boundary:
            enclosed.append(comp)
    return enclosed


def main(grid: np.ndarray) -> np.ndarray:
    """
    Fill each enclosed background region inside a closed foreground ring.

    Regions are sorted by enclosed area from small to large and filled with
    red, green, yellow, orange, magenta in that order. Non-enclosing
    foreground distractors stay unchanged.
    """
    g = np.asarray(grid, dtype=int)
    bg = _get_bg(g)
    out = g.copy()

    cavities = _enclosed_background_regions(g, bg)
    cavities.sort(key=lambda comp: (len(comp), min(comp)))

    for idx, comp in enumerate(cavities):
        fill = FILL_SEQUENCE[idx % len(FILL_SEQUENCE)]
        for r, c in comp:
            out[r, c] = fill

    return out


def _normalize(coords: Iterable[tuple[int, int]]) -> list[tuple[int, int]]:
    pts = list(coords)
    min_r = min(r for r, _ in pts)
    min_c = min(c for _, c in pts)
    return sorted((r - min_r, c - min_c) for r, c in pts)


def _shape_dims(coords: list[tuple[int, int]]) -> tuple[int, int]:
    return max(r for r, _ in coords) + 1, max(c for _, c in coords) + 1


def _shape_connected(coords: list[tuple[int, int]]) -> bool:
    if not coords:
        return False
    pts = set(coords)
    q = deque([coords[0]])
    seen = {coords[0]}
    while q:
        r, c = q.popleft()
        for dr, dc in DIR4:
            nxt = (r + dr, c + dc)
            if nxt in pts and nxt not in seen:
                seen.add(nxt)
                q.append(nxt)
    return len(seen) == len(pts)


def _shape_hole_areas(coords: list[tuple[int, int]]) -> list[int]:
    sh, sw = _shape_dims(coords)
    grid = np.zeros((sh + 2, sw + 2), dtype=int)
    for r, c in coords:
        grid[r + 1, c + 1] = 1
    return sorted(len(comp) for comp in _enclosed_background_regions(grid, 0))


def _ring_shape(rng: np.random.Generator) -> tuple[list[tuple[int, int]], int]:
    inner_h = int(rng.integers(1, 7))
    inner_w = int(rng.integers(1, 7))
    top = int(rng.integers(1, 3))
    bottom = int(rng.integers(1, 3))
    left = int(rng.integers(1, 3))
    right = int(rng.integers(1, 3))

    outer_h = inner_h + top + bottom
    outer_w = inner_w + left + right
    coords = {
        (r, c)
        for r in range(outer_h)
        for c in range(outer_w)
        if not (top <= r < top + inner_h and left <= c < left + inner_w)
    }

    if rng.random() < 0.45:
        side = int(rng.integers(0, 4))
        if side in (0, 1):
            span = int(rng.integers(1, outer_w + 1))
            start = int(rng.integers(0, outer_w - span + 1))
            depth = int(rng.integers(1, 3))
            base_row = -depth if side == 0 else outer_h
            rows = range(base_row, 0) if side == 0 else range(outer_h, outer_h + depth)
            for r in rows:
                for c in range(start, start + span):
                    coords.add((r, c))
        else:
            span = int(rng.integers(1, outer_h + 1))
            start = int(rng.integers(0, outer_h - span + 1))
            depth = int(rng.integers(1, 3))
            base_col = -depth if side == 2 else outer_w
            cols = range(base_col, 0) if side == 2 else range(outer_w, outer_w + depth)
            for r in range(start, start + span):
                for c in cols:
                    coords.add((r, c))

    pts = _normalize(coords)
    area = inner_h * inner_w
    assert _shape_connected(pts)
    assert _shape_hole_areas(pts) == [area]
    return pts, area


def _line_shape(rng: np.random.Generator) -> list[tuple[int, int]]:
    length = int(rng.integers(2, 7))
    if rng.random() < 0.5:
        return [(0, c) for c in range(length)]
    return [(r, 0) for r in range(length)]


def _l_shape(rng: np.random.Generator) -> list[tuple[int, int]]:
    arm_h = int(rng.integers(2, 6))
    arm_w = int(rng.integers(2, 6))
    coords = {(r, 0) for r in range(arm_h)} | {(0, c) for c in range(arm_w)}
    orient = int(rng.integers(0, 4))
    if orient == 1:
        coords = {(r, arm_w - 1 - c) for r, c in coords}
    elif orient == 2:
        coords = {(arm_h - 1 - r, c) for r, c in coords}
    elif orient == 3:
        coords = {(arm_h - 1 - r, arm_w - 1 - c) for r, c in coords}
    return _normalize(coords)


def _open_rect_shape(rng: np.random.Generator) -> list[tuple[int, int]]:
    height = int(rng.integers(3, 7))
    width = int(rng.integers(3, 7))
    coords = {
        (r, c)
        for r in range(height)
        for c in range(width)
        if r in (0, height - 1) or c in (0, width - 1)
    }

    side = int(rng.integers(0, 4))
    if side in (0, 1):
        side_len = width
        gap_len = 1 if side_len <= 4 else int(rng.integers(1, 3))
        start = int(rng.integers(1, side_len - gap_len))
        row = 0 if side == 0 else height - 1
        for c in range(start, start + gap_len):
            coords.discard((row, c))
    else:
        side_len = height
        gap_len = 1 if side_len <= 4 else int(rng.integers(1, 3))
        start = int(rng.integers(1, side_len - gap_len))
        col = 0 if side == 2 else width - 1
        for r in range(start, start + gap_len):
            coords.discard((r, col))

    pts = _normalize(coords)
    assert _shape_connected(pts)
    assert _shape_hole_areas(pts) == []
    return pts


def _distractor_shape(rng: np.random.Generator) -> list[tuple[int, int]]:
    kind = int(rng.integers(0, 4))
    if kind == 0:
        pts = [(0, 0)]
    elif kind == 1:
        pts = _line_shape(rng)
    elif kind == 2:
        pts = _l_shape(rng)
    else:
        pts = _open_rect_shape(rng)
    assert _shape_connected(pts)
    assert _shape_hole_areas(pts) == []
    return pts


def _can_place(
    coords: list[tuple[int, int]],
    top: int,
    left: int,
    blocked: np.ndarray,
    pad: int,
) -> bool:
    h, w = blocked.shape
    sh, sw = _shape_dims(coords)
    if top < 0 or left < 0 or top + sh > h or left + sw > w:
        return False
    r0 = max(0, top - pad)
    c0 = max(0, left - pad)
    r1 = min(h, top + sh + pad)
    c1 = min(w, left + sw + pad)
    return not bool(blocked[r0:r1, c0:c1].any())


def _place(
    grid: np.ndarray,
    blocked: np.ndarray,
    coords: list[tuple[int, int]],
    top: int,
    left: int,
    color: int,
    pad: int,
) -> None:
    sh, sw = _shape_dims(coords)
    for r, c in coords:
        grid[top + r, left + c] = color
    r0 = max(0, top - pad)
    c0 = max(0, left - pad)
    r1 = min(grid.shape[0], top + sh + pad)
    c1 = min(grid.shape[1], left + sw + pad)
    blocked[r0:r1, c0:c1] = True


def _try_place_shape(
    grid: np.ndarray,
    blocked: np.ndarray,
    coords: list[tuple[int, int]],
    color: int,
    pad: int,
    rng: np.random.Generator,
    tries: int = 80,
) -> bool:
    sh, sw = _shape_dims(coords)
    h, w = grid.shape
    if sh > h or sw > w:
        return False
    for _ in range(tries):
        top = int(rng.integers(0, h - sh + 1))
        left = int(rng.integers(0, w - sw + 1))
        if _can_place(coords, top, left, blocked, pad):
            _place(grid, blocked, coords, top, left, color, pad)
            return True
    return False


def _expected_output(grid: np.ndarray) -> tuple[np.ndarray, list[int]] | None:
    cavities = _enclosed_background_regions(grid, BG)
    cavity_sizes = sorted(len(comp) for comp in cavities)
    if not cavity_sizes:
        return None
    out = main(grid)
    return out, cavity_sizes


def _gen_attempt(rng: np.random.Generator) -> np.ndarray | None:
    n_rings = int(rng.choice(RING_COUNT_VALUES, p=RING_COUNT_PROBS))
    min_side = max(GRID_SIDE_MIN, 8 + 3 * n_rings)
    height = int(rng.integers(min_side, GRID_SIDE_MAX + 1))
    width = int(rng.integers(min_side, GRID_SIDE_MAX + 1))
    fg = int(rng.choice(INPUT_FG_CHOICES))

    grid = np.full((height, width), BG, dtype=int)
    blocked = np.zeros((height, width), dtype=bool)

    target_areas: list[int] = []
    for _ in range(n_rings):
        placed = False
        for _ in range(120):
            coords, area = _ring_shape(rng)
            if area in target_areas:
                continue
            if _try_place_shape(grid, blocked, coords, fg, RING_PAD, rng):
                target_areas.append(area)
                placed = True
                break
        if not placed:
            return None

    distractor_count = 0 if rng.random() < 0.2 else int(rng.integers(1, 7))
    for _ in range(distractor_count):
        coords = _distractor_shape(rng)
        _try_place_shape(grid, blocked, coords, fg, DISTRACTOR_PAD, rng, tries=60)

    if int((grid != BG).sum()) * 2 >= grid.size:
        return None

    non_bg = sorted(int(v) for v in np.unique(grid) if int(v) != BG)
    if non_bg != [fg]:
        return None

    expected = _expected_output(grid)
    if expected is None:
        return None
    out, cavity_sizes = expected
    if cavity_sizes != sorted(target_areas):
        return None

    ranked = sorted(_enclosed_background_regions(grid, BG), key=lambda comp: (len(comp), min(comp)))
    for idx, comp in enumerate(ranked):
        want = FILL_SEQUENCE[idx % len(FILL_SEQUENCE)]
        if any(int(out[r, c]) != want for r, c in comp):
            return None

    return grid


def generate_input() -> np.ndarray:
    for _ in range(GEN_ATTEMPTS):
        grid = _gen_attempt(_RNG)
        if grid is not None:
            return grid
    raise RuntimeError("failed to generate a valid ring-sorting input")
