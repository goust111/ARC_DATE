import random
from collections import deque

import numpy as np


BLACK = 0
GREY = 5
RULER_COLORS = (1, 2, 3, 4, 6, 7, 8, 9)
GEN_MAX_ATTEMPTS = 400


def _bbox(mask: np.ndarray) -> tuple[int, int, int, int]:
    rows, cols = np.where(mask)
    assert rows.size > 0
    return int(rows.min()), int(cols.min()), int(rows.max()), int(cols.max())


def _crop_nonblack(grid: np.ndarray) -> np.ndarray:
    mask = grid != BLACK
    r0, c0, r1, c1 = _bbox(mask)
    return grid[r0 : r1 + 1, c0 : c1 + 1]


def _is_connected(mask: np.ndarray) -> bool:
    cells = np.argwhere(mask)
    if cells.size == 0:
        return False
    h, w = mask.shape
    start = tuple(map(int, cells[0]))
    seen = {start}
    q = deque([start])
    while q:
        r, c = q.popleft()
        for dr, dc in ((1, 0), (-1, 0), (0, 1), (0, -1)):
            rr = r + dr
            cc = c + dc
            if 0 <= rr < h and 0 <= cc < w and mask[rr, cc] and (rr, cc) not in seen:
                seen.add((rr, cc))
                q.append((rr, cc))
    return len(seen) == int(mask.sum())


def _extract_ruler(grid: np.ndarray) -> tuple[list[int], list[int]]:
    row = np.asarray(grid[0], dtype=int)
    cols = np.flatnonzero(row != BLACK)
    assert cols.size > 0
    left = int(cols[0])
    right = int(cols[-1])
    assert np.all(row[left : right + 1] != BLACK)
    assert np.all(row[:left] == BLACK)
    assert np.all(row[right + 1 :] == BLACK)

    colors: list[int] = []
    widths: list[int] = []
    c = left
    while c <= right:
        color = int(row[c])
        end = c
        while end <= right and int(row[end]) == color:
            end += 1
        colors.append(color)
        widths.append(end - c)
        c = end

    assert len(colors) >= 2
    assert len(colors) == len(set(colors))
    return colors, widths


def main(grid: np.ndarray) -> np.ndarray:
    """Slice a grey object by the top ruler, reorder slices by segment width, and recolor them."""
    g = np.asarray(grid, dtype=int)
    assert g.ndim == 2

    colors, widths = _extract_ruler(g)
    shape_mask = g == GREY
    assert shape_mask.any()
    assert _is_connected(shape_mask)

    r0, c0, r1, c1 = _bbox(shape_mask)
    assert r0 >= 1
    bbox_mask = shape_mask[r0 : r1 + 1, c0 : c1 + 1]
    assert bbox_mask.shape[1] == sum(widths)

    recolored_slices = []
    offset = 0
    for index, (color, width) in enumerate(zip(colors, widths, strict=False)):
        slice_mask = bbox_mask[:, offset : offset + width]
        patch = np.zeros(slice_mask.shape, dtype=int)
        patch[slice_mask] = int(color)
        recolored_slices.append((int(width), int(index), patch))
        offset += width

    pieces = [
        patch
        for _width, _index, patch in sorted(
            recolored_slices, key=lambda item: (-item[0], item[1])
        )
    ]
    out = np.concatenate(pieces, axis=1)
    return _crop_nonblack(out)


def _sort_permutation(widths: list[int]) -> list[int]:
    return sorted(range(len(widths)), key=lambda i: (-widths[i], i))


def _sample_widths() -> list[int]:
    for _ in range(200):
        n_segments = random.randint(2, 5)
        want_ties = n_segments >= 3 and random.random() < 0.5

        if want_ties:
            widths = [random.randint(1, 5) for _ in range(n_segments)]
            if len(set(widths)) == 1 or len(set(widths)) == len(widths):
                continue
        else:
            pool = list(range(1, 7))
            random.shuffle(pool)
            widths = pool[:n_segments]

        if not 6 <= sum(widths) <= 16:
            continue
        if _sort_permutation(widths) == list(range(n_segments)):
            continue
        return widths
    raise RuntimeError("failed to sample ruler widths")


def _shape_valid(mask: np.ndarray, widths: list[int]) -> bool:
    h, w = mask.shape
    if not mask.any() or not _is_connected(mask):
        return False
    if not np.all(mask.any(axis=0)) or not np.all(mask.any(axis=1)):
        return False
    if int(mask.sum()) < max(8, (h * w * 11) // 20):
        return False

    offset = 0
    for width in widths:
        if not mask[:, offset : offset + width].any():
            return False
        offset += width
    return True


def _random_boundary_cut(h: int, w: int) -> np.ndarray | None:
    cut = np.zeros((h, w), dtype=bool)
    side = random.choice(("top", "bottom", "left", "right"))

    if side in ("top", "bottom"):
        if h < 3 or w < 2:
            return None
        depth = random.randint(1, max(1, h // 2))
        span = random.randint(1, max(1, w - 1))
        c0 = random.randint(0, w - span)
        rows = slice(0, depth) if side == "top" else slice(h - depth, h)
        cut[rows, c0 : c0 + span] = True
    else:
        if w < 3 or h < 2:
            return None
        depth = random.randint(1, max(1, w // 2))
        span = random.randint(1, max(1, h - 1))
        r0 = random.randint(0, h - span)
        cols = slice(0, depth) if side == "left" else slice(w - depth, w)
        cut[r0 : r0 + span, cols] = True
    return cut


def _random_hole(h: int, w: int) -> np.ndarray | None:
    if h < 5 or w < 5:
        return None
    hole = np.zeros((h, w), dtype=bool)
    hole_h = random.randint(1, max(1, h // 3))
    hole_w = random.randint(1, max(1, w // 3))
    if hole_h >= h - 1 or hole_w >= w - 1:
        return None
    r0 = random.randint(1, h - hole_h - 1)
    c0 = random.randint(1, w - hole_w - 1)
    hole[r0 : r0 + hole_h, c0 : c0 + hole_w] = True
    return hole


def _build_shape(height: int, widths: list[int]) -> np.ndarray:
    width = sum(widths)
    for _ in range(300):
        mask = np.ones((height, width), dtype=bool)
        changed = 0
        target_notches = random.randint(1, 4)
        target_holes = random.randint(0, 2)

        for _ in range(target_notches):
            for _ in range(20):
                cut = _random_boundary_cut(height, width)
                if cut is None:
                    break
                candidate = mask & (~cut)
                if _shape_valid(candidate, widths):
                    mask = candidate
                    changed += 1
                    break

        for _ in range(target_holes):
            for _ in range(20):
                hole = _random_hole(height, width)
                if hole is None:
                    break
                candidate = mask & (~hole)
                if _shape_valid(candidate, widths):
                    mask = candidate
                    changed += 1
                    break

        if changed >= 1 and _shape_valid(mask, widths):
            return mask
    raise RuntimeError("failed to build shape")


def _render_input(
    grid_h: int,
    grid_w: int,
    ruler_colors: list[int],
    widths: list[int],
    shape_mask: np.ndarray,
) -> np.ndarray:
    bbox_h, bbox_w = shape_mask.shape
    grid = np.zeros((grid_h, grid_w), dtype=int)

    ruler_c0 = random.randint(0, grid_w - bbox_w)
    grid[0, ruler_c0 : ruler_c0 + bbox_w] = np.concatenate(
        [np.full(width, color, dtype=int) for color, width in zip(ruler_colors, widths, strict=False)]
    )

    shape_r0 = random.randint(2, grid_h - bbox_h - 1)
    shape_c0 = random.randint(1, grid_w - bbox_w - 1)
    target = grid[shape_r0 : shape_r0 + bbox_h, shape_c0 : shape_c0 + bbox_w]
    target[shape_mask] = GREY
    return grid


def _gen_attempt() -> np.ndarray | None:
    widths = _sample_widths()
    bbox_w = sum(widths)
    bbox_h = random.randint(4, 10)
    shape_mask = _build_shape(bbox_h, widths)
    ruler_colors = random.sample(RULER_COLORS, len(widths))

    grid_h = bbox_h + random.randint(3, 8)
    grid_w = bbox_w + random.randint(2, 9)
    if grid_h < bbox_h + 3 or grid_w < bbox_w + 2:
        return None

    inp = _render_input(grid_h, grid_w, ruler_colors, widths, shape_mask)
    try:
        out = main(inp)
    except Exception:
        return None

    if out.shape != shape_mask.shape:
        return None
    if set(map(int, np.unique(out))) - {BLACK} != set(ruler_colors):
        return None
    if np.array_equal(inp, out):
        return None
    return inp


def generate_input() -> np.ndarray:
    for _ in range(GEN_MAX_ATTEMPTS):
        inp = _gen_attempt()
        if inp is not None:
            return inp
    raise RuntimeError("failed to generate input")
