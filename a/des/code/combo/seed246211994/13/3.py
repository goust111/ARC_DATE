from __future__ import annotations

import random
from collections import deque

import numpy as np


RED = 2
MAX_RETRIES = 400
NEIGHBORS4 = ((1, 0), (-1, 0), (0, 1), (0, -1))


def _mode_color(grid: np.ndarray) -> int:
    vals, counts = np.unique(grid, return_counts=True)
    return int(vals[np.argmax(counts)])


def _components(mask: np.ndarray) -> list[list[tuple[int, int]]]:
    h, w = mask.shape
    seen = np.zeros((h, w), dtype=bool)
    comps: list[list[tuple[int, int]]] = []
    for r in range(h):
        for c in range(w):
            if not mask[r, c] or seen[r, c]:
                continue
            q = deque([(r, c)])
            seen[r, c] = True
            comp: list[tuple[int, int]] = []
            while q:
                rr, cc = q.popleft()
                comp.append((rr, cc))
                for dr, dc in NEIGHBORS4:
                    nr, nc = rr + dr, cc + dc
                    if 0 <= nr < h and 0 <= nc < w and mask[nr, nc] and not seen[nr, nc]:
                        seen[nr, nc] = True
                        q.append((nr, nc))
            comps.append(comp)
    return comps


def _bbox(coords: list[tuple[int, int]]) -> tuple[int, int, int, int]:
    rows = [r for r, _ in coords]
    cols = [c for _, c in coords]
    return min(rows), min(cols), max(rows), max(cols)


def _find_dividers(grid: np.ndarray, bg: int) -> tuple[int, int]:
    h, w = grid.shape
    rows = [r for r in range(h) if int(grid[r, 0]) != bg and np.all(grid[r, :] == grid[r, 0])]
    cols = [c for c in range(w) if int(grid[0, c]) != bg and np.all(grid[:, c] == grid[0, c])]
    matches = [
        (r, c)
        for r in rows
        for c in cols
        if int(grid[r, 0]) == int(grid[0, c])
    ]
    assert len(matches) == 1, "expected one matching full-row/full-col divider"
    return int(matches[0][0]), int(matches[0][1])


def _resize_nn(img: np.ndarray, new_h: int, new_w: int) -> np.ndarray:
    src_h, src_w = img.shape
    assert new_h > 0 and new_w > 0
    rr = np.clip((np.arange(new_h) * src_h // new_h).astype(int), 0, src_h - 1)
    cc = np.clip((np.arange(new_w) * src_w // new_w).astype(int), 0, src_w - 1)
    return img[rr[:, None], cc[None, :]]


def _quadrant_bounds(shape: tuple[int, int], mid_r: int, mid_c: int) -> dict[tuple[int, int], tuple[int, int, int, int]]:
    h, w = shape
    return {
        (0, 0): (0, mid_r - 1, 0, mid_c - 1),
        (0, 1): (0, mid_r - 1, mid_c + 1, w - 1),
        (1, 0): (mid_r + 1, h - 1, 0, mid_c - 1),
        (1, 1): (mid_r + 1, h - 1, mid_c + 1, w - 1),
    }


def _find_anchors(grid: np.ndarray, mid_r: int, mid_c: int) -> dict[tuple[int, int], tuple[int, int, int]]:
    anchors: dict[tuple[int, int], tuple[int, int, int]] = {}
    for comp in _components(grid == RED):
        r0, c0, r1, c1 = _bbox(comp)
        side_r = r1 - r0 + 1
        side_c = c1 - c0 + 1
        assert side_r == side_c == len(comp) ** 0.5, "red anchors must be filled squares"
        center_r = (r0 + r1) // 2
        center_c = (c0 + c1) // 2
        key = (0 if center_r < mid_r else 1, 0 if center_c < mid_c else 1)
        assert key not in anchors, "expected one red anchor per quadrant"
        anchors[key] = (int(r0), int(c0), int(side_r))
    assert set(anchors) == {(0, 0), (0, 1), (1, 0), (1, 1)}, "expected four quadrant anchors"
    return anchors


def _source_patch(grid: np.ndarray, bg: int, mid_r: int, mid_c: int) -> np.ndarray:
    tl = grid[:mid_r, :mid_c]
    for comp in _components(tl != bg):
        if any(int(tl[r, c]) == RED for r, c in comp):
            r0, c0, r1, c1 = _bbox(comp)
            return tl[r0 : r1 + 1, c0 : c1 + 1].copy()
    raise AssertionError("top-left quadrant must contain one motif attached to the red anchor")


def _transform_patch(patch: np.ndarray, target_size: int, source_size: int, flip_v: bool, flip_h: bool) -> np.ndarray:
    out = patch
    if flip_h:
        out = np.fliplr(out)
    if flip_v:
        out = np.flipud(out)
    new_h = max(1, int(round(out.shape[0] * target_size / source_size)))
    new_w = max(1, int(round(out.shape[1] * target_size / source_size)))
    return _resize_nn(out, new_h, new_w)


def main(grid: np.ndarray) -> np.ndarray:
    """
    Copy the top-left motif to the other quadrants using red-anchor scale and quadrant reflection.
    """
    g = np.asarray(grid, dtype=int)
    bg = _mode_color(g)
    mid_r, mid_c = _find_dividers(g, bg)
    bounds = _quadrant_bounds(g.shape, mid_r, mid_c)
    anchors = _find_anchors(g, mid_r, mid_c)

    src_r0, src_c0, src_size = anchors[(0, 0)]
    patch = _source_patch(g, bg, mid_r, mid_c)

    out = g.copy()
    for key, (tr0, tc0, target_size) in anchors.items():
        if key == (0, 0):
            continue
        qb, qr = key
        patch_t = _transform_patch(
            patch,
            target_size=target_size,
            source_size=src_size,
            flip_v=bool(qb),
            flip_h=bool(qr),
        )
        red_cells = np.argwhere(patch_t == RED)
        assert red_cells.size > 0, "transformed patch must contain red anchor pixels"
        red_r0, red_c0 = red_cells.min(axis=0)
        place_r0 = int(tr0) - int(red_r0)
        place_c0 = int(tc0) - int(red_c0)

        q_r0, q_r1, q_c0, q_c1 = bounds[key]
        nz = np.argwhere(patch_t != bg)
        for rr, cc in nz:
            gr = place_r0 + int(rr)
            gc = place_c0 + int(cc)
            if not (q_r0 <= gr <= q_r1 and q_c0 <= gc <= q_c1):
                continue
            out[gr, gc] = int(patch_t[int(rr), int(cc)])

    return out


def _grow_connected(height: int, width: int, seed: tuple[int, int], target_area: int) -> set[tuple[int, int]]:
    cells = {seed}
    frontier = set()

    def add_frontier(r: int, c: int) -> None:
        for dr, dc in NEIGHBORS4:
            nr, nc = r + dr, c + dc
            if 0 <= nr < height and 0 <= nc < width and (nr, nc) not in cells:
                frontier.add((nr, nc))

    add_frontier(*seed)
    while len(cells) < target_area:
        if not frontier:
            return set()
        choice = random.choice(list(frontier))
        frontier.remove(choice)
        cells.add(choice)
        add_frontier(*choice)
    return cells


def _logical_patch() -> tuple[np.ndarray, int] | None:
    src_size = random.randint(1, 3)
    h = random.randint(2, 4)
    w = random.randint(2, 4)
    anchor = (random.randrange(h), random.randrange(w))
    min_fill = max(3, (h * w + 1) // 2)
    max_fill = min(h * w, max(min_fill, int(np.ceil(h * w * 0.85))))
    filled = _grow_connected(h, w, anchor, random.randint(min_fill, max_fill))
    if not filled:
        return None
    if {r for r, _ in filled} != set(range(h)) or {c for _, c in filled} != set(range(w)):
        return None

    extra_count = len(filled) - 1
    if extra_count < 2:
        return None
    color_pool = [c for c in range(10) if c != RED]
    bg = random.choice(color_pool)
    color_pool.remove(bg)
    if len(color_pool) < 2:
        return None
    n_extra_colors = random.randint(2, min(4, len(color_pool)))
    extras = random.sample(color_pool, n_extra_colors)

    logical = np.full((h, w), bg, dtype=int)
    logical[anchor] = RED
    others = [cell for cell in filled if cell != anchor]
    random.shuffle(others)
    for i, (r, c) in enumerate(others):
        logical[r, c] = extras[i % len(extras)]

    patch = np.repeat(np.repeat(logical, src_size, axis=0), src_size, axis=1)
    return patch, bg


def _anchor_ranges(q_bounds: tuple[int, int, int, int], patch: np.ndarray) -> tuple[tuple[int, int], tuple[int, int]] | None:
    q_r0, q_r1, q_c0, q_c1 = q_bounds
    red_cells = np.argwhere(patch == RED)
    red_r0, red_c0 = red_cells.min(axis=0)
    r_lo = q_r0 + int(red_r0)
    r_hi = q_r1 - patch.shape[0] + 1 + int(red_r0)
    c_lo = q_c0 + int(red_c0)
    c_hi = q_c1 - patch.shape[1] + 1 + int(red_c0)
    if r_lo > r_hi or c_lo > c_hi:
        return None
    return (int(r_lo), int(r_hi)), (int(c_lo), int(c_hi))


def _place_patch(canvas: np.ndarray, patch: np.ndarray, anchor_r: int, anchor_c: int, bg: int) -> None:
    red_cells = np.argwhere(patch == RED)
    red_r0, red_c0 = red_cells.min(axis=0)
    top = int(anchor_r) - int(red_r0)
    left = int(anchor_c) - int(red_c0)
    nz = np.argwhere(patch != bg)
    for rr, cc in nz:
        canvas[top + int(rr), left + int(cc)] = int(patch[int(rr), int(cc)])


def _gen_attempt() -> np.ndarray | None:
    source = _logical_patch()
    if source is None:
        return None
    source_patch, bg = source

    src_red = np.argwhere(source_patch == RED)
    src_size = int(src_red[:, 0].max() - src_red[:, 0].min() + 1)
    divider_choices = [c for c in range(10) if c not in {bg, RED}]
    divider = random.choice(divider_choices)

    target_sizes = {
        (0, 0): src_size,
        (0, 1): random.randint(1, 4),
        (1, 0): random.randint(1, 4),
        (1, 1): random.randint(1, 4),
    }
    if all(size == src_size for key, size in target_sizes.items() if key != (0, 0)):
        key = random.choice([(0, 1), (1, 0), (1, 1)])
        target_sizes[key] = 1 if src_size != 1 else 4

    transformed = {
        key: _transform_patch(
            source_patch,
            target_size=target_sizes[key],
            source_size=src_size,
            flip_v=bool(key[0]),
            flip_h=bool(key[1]),
        )
        for key in target_sizes
    }

    margin = random.randint(1, 4)
    top_h = max(transformed[(0, 0)].shape[0], transformed[(0, 1)].shape[0]) + margin
    bottom_h = max(transformed[(1, 0)].shape[0], transformed[(1, 1)].shape[0]) + margin
    left_w = max(transformed[(0, 0)].shape[1], transformed[(1, 0)].shape[1]) + margin
    right_w = max(transformed[(0, 1)].shape[1], transformed[(1, 1)].shape[1]) + margin

    mid_r = top_h
    mid_c = left_w
    h = top_h + 1 + bottom_h
    w = left_w + 1 + right_w
    bounds = _quadrant_bounds((h, w), mid_r, mid_c)

    anchors: dict[tuple[int, int], tuple[int, int]] = {}
    for key, patch in transformed.items():
        ranges = _anchor_ranges(bounds[key], patch)
        if ranges is None:
            return None
        (r_lo, r_hi), (c_lo, c_hi) = ranges
        anchors[key] = (random.randint(r_lo, r_hi), random.randint(c_lo, c_hi))

    out = np.full((h, w), bg, dtype=int)
    out[mid_r, :] = divider
    out[:, mid_c] = divider

    for key, patch in transformed.items():
        anchor_r, anchor_c = anchors[key]
        _place_patch(out, patch, anchor_r, anchor_c, bg)

    inp = out.copy()
    for key in [(0, 1), (1, 0), (1, 1)]:
        q_r0, q_r1, q_c0, q_c1 = bounds[key]
        inp[q_r0 : q_r1 + 1, q_c0 : q_c1 + 1] = bg
        size = target_sizes[key]
        ar, ac = anchors[key]
        inp[ar : ar + size, ac : ac + size] = RED

    if np.count_nonzero(inp == RED) <= src_size * src_size:
        return None
    if not np.array_equal(main(inp), out):
        return None
    return inp


def generate_input() -> np.ndarray:
    for _ in range(MAX_RETRIES):
        grid = _gen_attempt()
        if grid is not None:
            return grid
    raise RuntimeError("failed to generate a valid input")
