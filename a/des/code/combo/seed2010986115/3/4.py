from __future__ import annotations

import random

import numpy as np


BG = 0
LINE = 9
CROSS = 4
OBJECT_COLOR_POOL = [1, 2, 3, 5, 6, 7, 8]

QUADRANT_SIZE_RANGE = (5, 9)
MAX_OBJECT_BBOX = 5
GEN_ATTEMPTS = 400
SHAPE_ATTEMPTS = 120

DIRS4 = ((1, 0), (-1, 0), (0, 1), (0, -1))


def _bbox(coords: np.ndarray) -> tuple[int, int, int, int]:
    rows = coords[:, 0]
    cols = coords[:, 1]
    return int(rows.min()), int(cols.min()), int(rows.max()), int(cols.max())


def _components(mask: np.ndarray) -> list[np.ndarray]:
    h, w = mask.shape
    seen = np.zeros_like(mask, dtype=bool)
    comps: list[np.ndarray] = []

    for r in range(h):
        for c in range(w):
            if not mask[r, c] or seen[r, c]:
                continue
            stack = [(r, c)]
            seen[r, c] = True
            comp: list[tuple[int, int]] = []
            while stack:
                rr, cc = stack.pop()
                comp.append((rr, cc))
                for dr, dc in DIRS4:
                    nr, nc = rr + dr, cc + dc
                    if 0 <= nr < h and 0 <= nc < w and mask[nr, nc] and not seen[nr, nc]:
                        seen[nr, nc] = True
                        stack.append((nr, nc))
            comps.append(np.array(comp, dtype=int))
    return comps


def _transform_patch(patch: np.ndarray, quadrant: str) -> np.ndarray:
    if quadrant == "tl":
        return patch.copy()
    if quadrant == "tr":
        return np.fliplr(patch)
    if quadrant == "bl":
        return np.flipud(patch)
    if quadrant == "br":
        return np.rot90(patch, 2)
    raise AssertionError(f"unknown quadrant: {quadrant}")


def _find_cross(g: np.ndarray) -> tuple[int, int, int, int]:
    h, w = g.shape
    row_candidates: list[tuple[int, int, int, int]] = []
    col_candidates: list[tuple[int, int, int, int]] = []

    for r in range(h):
        colors, counts = np.unique(g[r, :], return_counts=True)
        if len(colors) != 2 or sorted(int(x) for x in counts) != [1, w - 1]:
            continue
        lone_idx = int(np.argmin(counts))
        cross_color = int(colors[lone_idx])
        line_color = int(colors[1 - lone_idx])
        c = int(np.flatnonzero(g[r, :] == cross_color)[0])
        row_candidates.append((r, c, line_color, cross_color))

    for c in range(w):
        colors, counts = np.unique(g[:, c], return_counts=True)
        if len(colors) != 2 or sorted(int(x) for x in counts) != [1, h - 1]:
            continue
        lone_idx = int(np.argmin(counts))
        cross_color = int(colors[lone_idx])
        line_color = int(colors[1 - lone_idx])
        r = int(np.flatnonzero(g[:, c] == cross_color)[0])
        col_candidates.append((r, c, line_color, cross_color))

    matches = [cand for cand in row_candidates if cand in col_candidates]
    assert len(matches) == 1, "expected exactly one divider cross"
    return matches[0]


def _infer_background(g: np.ndarray, line_color: int, cross_color: int) -> int:
    colors, counts = np.unique(g, return_counts=True)
    remaining = [
        (int(count), int(color))
        for color, count in zip(colors.tolist(), counts.tolist(), strict=False)
        if int(color) not in {line_color, cross_color}
    ]
    assert remaining, "expected background/object colors besides divider colors"
    return max(remaining)[1]


def _extract_object_patch(quadrant: np.ndarray, bg: int) -> tuple[int, int, np.ndarray]:
    mask = quadrant != bg
    assert mask.any(), "each quadrant must contain an object"
    comps = _components(mask)
    assert len(comps) == 1, "each quadrant must contain exactly one 4-connected object"
    r0, c0, r1, c1 = _bbox(comps[0])
    return r0, c0, quadrant[r0 : r1 + 1, c0 : c1 + 1].copy()


def main(grid: np.ndarray) -> np.ndarray:
    """
    Remove the divider cross, stitch the quadrants together, and transform the
    single object in each quadrant according to its position:
    TL = identity, TR = horizontal flip, BL = vertical flip, BR = 180 rotation.
    """
    g = np.asarray(grid, dtype=int)
    assert g.ndim == 2 and min(g.shape) >= 3, "expected a non-empty 2D grid"

    mid_r, mid_c, line_color, cross_color = _find_cross(g)
    bg = _infer_background(g, line_color, cross_color)

    quadrants = {
        "tl": g[:mid_r, :mid_c],
        "tr": g[:mid_r, mid_c + 1 :],
        "bl": g[mid_r + 1 :, :mid_c],
        "br": g[mid_r + 1 :, mid_c + 1 :],
    }
    out = np.full((g.shape[0] - 1, g.shape[1] - 1), bg, dtype=int)
    out_quadrants = {
        "tl": out[:mid_r, :mid_c],
        "tr": out[:mid_r, mid_c:],
        "bl": out[mid_r:, :mid_c],
        "br": out[mid_r:, mid_c:],
    }

    for key, quadrant in quadrants.items():
        r0, c0, patch = _extract_object_patch(quadrant, bg)
        transformed = _transform_patch(patch, key)
        h, w = patch.shape
        out_quadrants[key][r0 : r0 + h, c0 : c0 + w] = transformed

    return out


def _grow_connected_mask(height: int, width: int, area: int) -> np.ndarray | None:
    seed = (random.randrange(height), random.randrange(width))
    cells = {seed}
    frontier = set()
    for dr, dc in DIRS4:
        nr, nc = seed[0] + dr, seed[1] + dc
        if 0 <= nr < height and 0 <= nc < width:
            frontier.add((nr, nc))

    while len(cells) < area:
        if not frontier:
            return None
        cell = random.choice(tuple(frontier))
        frontier.remove(cell)
        cells.add(cell)
        for dr, dc in DIRS4:
            nr, nc = cell[0] + dr, cell[1] + dc
            if 0 <= nr < height and 0 <= nc < width and (nr, nc) not in cells:
                frontier.add((nr, nc))

    mask = np.zeros((height, width), dtype=bool)
    for r, c in cells:
        mask[r, c] = True

    r0, c0, r1, c1 = _bbox(np.argwhere(mask))
    return mask[r0 : r1 + 1, c0 : c1 + 1]


def _has_forbidden_symmetry(mask: np.ndarray) -> bool:
    return (
        np.array_equal(mask, np.fliplr(mask))
        or np.array_equal(mask, np.flipud(mask))
        or np.array_equal(mask, np.rot90(mask, 2))
    )


def _random_object_mask(max_h: int, max_w: int) -> np.ndarray | None:
    h_hi = min(MAX_OBJECT_BBOX, max_h)
    w_hi = min(MAX_OBJECT_BBOX, max_w)
    if h_hi < 2 or w_hi < 2:
        return None

    for _ in range(SHAPE_ATTEMPTS):
        height = random.randint(2, h_hi)
        width = random.randint(2, w_hi)
        max_area = height * width - 1
        min_area = max(3, (height * width + 1) // 2)
        if min_area > max_area:
            continue
        area = random.randint(min_area, max_area)
        mask = _grow_connected_mask(height, width, area)
        if mask is None or min(mask.shape) < 2:
            continue
        if mask.sum() == mask.size:
            continue
        if _has_forbidden_symmetry(mask):
            continue
        if random.random() < 0.5:
            mask = np.fliplr(mask)
        mask = np.rot90(mask, random.randrange(4))
        return mask.astype(bool)
    return None


def _make_quadrant(height: int, width: int, color: int, quadrant: str) -> tuple[np.ndarray, np.ndarray] | None:
    mask = _random_object_mask(height, width)
    if mask is None:
        return None

    patch = np.full(mask.shape, BG, dtype=int)
    patch[mask] = color
    ph, pw = patch.shape
    r0 = random.randint(0, height - ph)
    c0 = random.randint(0, width - pw)

    quadrant_in = np.full((height, width), BG, dtype=int)
    quadrant_out = np.full((height, width), BG, dtype=int)
    quadrant_in[r0 : r0 + ph, c0 : c0 + pw] = patch
    quadrant_out[r0 : r0 + ph, c0 : c0 + pw] = _transform_patch(patch, quadrant)
    return quadrant_in, quadrant_out


def _generate_case() -> tuple[np.ndarray, np.ndarray] | None:
    top = random.randint(*QUADRANT_SIZE_RANGE)
    bottom = random.randint(*QUADRANT_SIZE_RANGE)
    left = random.randint(*QUADRANT_SIZE_RANGE)
    right = random.randint(*QUADRANT_SIZE_RANGE)

    grid = np.full((top + bottom + 1, left + right + 1), BG, dtype=int)
    grid[top, :] = LINE
    grid[:, left] = LINE
    grid[top, left] = CROSS

    expected = np.full((top + bottom, left + right), BG, dtype=int)
    colors = random.sample(OBJECT_COLOR_POOL, 4)
    specs = [
        ("tl", top, left, (slice(0, top), slice(0, left)), (slice(0, top), slice(0, left))),
        ("tr", top, right, (slice(0, top), slice(left + 1, left + 1 + right)), (slice(0, top), slice(left, left + right))),
        ("bl", bottom, left, (slice(top + 1, top + 1 + bottom), slice(0, left)), (slice(top, top + bottom), slice(0, left))),
        ("br", bottom, right, (slice(top + 1, top + 1 + bottom), slice(left + 1, left + 1 + right)), (slice(top, top + bottom), slice(left, left + right))),
    ]

    for color, (key, height, width, in_slice, out_slice) in zip(colors, specs, strict=False):
        built = _make_quadrant(height, width, color, key)
        if built is None:
            return None
        quadrant_in, quadrant_out = built
        grid[in_slice] = quadrant_in
        expected[out_slice] = quadrant_out

    try:
        actual = main(grid)
    except Exception:
        return None
    if not np.array_equal(actual, expected):
        return None
    return grid, expected


def generate_input() -> np.ndarray:
    for _ in range(GEN_ATTEMPTS):
        case = _generate_case()
        if case is not None:
            return case[0]
    raise RuntimeError("failed to generate a valid input")
