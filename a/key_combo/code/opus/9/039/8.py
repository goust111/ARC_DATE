import random
from collections import Counter, deque

import numpy as np


BLACK = 0
BLUE = 1
RED = 2
GREY = 5

GRID_ROWS_RANGE = (15, 19)
GRID_COLS_RANGE = (15, 19)
RECT_DIM_CHOICES = [(2, 3), (2, 4), (3, 2), (3, 4), (2, 2), (3, 3)]
RECT_DIM_WEIGHTS = [4, 4, 4, 4, 1, 1]
MAX_GENERATION_ATTEMPTS = 400


def _get_bg(grid: np.ndarray) -> int:
    return int(Counter(map(int, grid.ravel())).most_common(1)[0][0])


def _bbox(mask: np.ndarray) -> tuple[int, int, int, int] | None:
    coords = np.argwhere(mask)
    if coords.size == 0:
        return None
    r0, c0 = coords.min(axis=0)
    r1, c1 = coords.max(axis=0)
    return int(r0), int(c0), int(r1), int(c1)


def _components(mask: np.ndarray) -> list[np.ndarray]:
    H, W = mask.shape
    seen = np.zeros((H, W), dtype=bool)
    comps: list[np.ndarray] = []
    for r in range(H):
        for c in range(W):
            if not mask[r, c] or seen[r, c]:
                continue
            q = deque([(r, c)])
            seen[r, c] = True
            coords: list[tuple[int, int]] = []
            while q:
                rr, cc = q.popleft()
                coords.append((rr, cc))
                for nr, nc in ((rr - 1, cc), (rr + 1, cc), (rr, cc - 1), (rr, cc + 1)):
                    if 0 <= nr < H and 0 <= nc < W and mask[nr, nc] and not seen[nr, nc]:
                        seen[nr, nc] = True
                        q.append((nr, nc))
            comps.append(np.array(coords, dtype=int))
    return comps


def _stamp_positions(
    shape: tuple[int, int], rect_top_left: tuple[int, int], rect_hw: tuple[int, int]
) -> list[tuple[int, int]]:
    H, W = shape
    r0, c0 = rect_top_left
    h, w = rect_hw
    seen = {(int(r0), int(c0))}
    for sr in (-1, 1):
        for sc in (-1, 1):
            k = 1
            while True:
                rr = int(r0 + sr * k * h)
                cc = int(c0 + sc * k * w)
                if rr >= H or rr + h <= 0 or cc >= W or cc + w <= 0:
                    break
                seen.add((rr, cc))
                k += 1
    return sorted(seen)


def _stamp_mask(
    shape: tuple[int, int], rect_top_left: tuple[int, int], rect_hw: tuple[int, int]
) -> np.ndarray:
    H, W = shape
    h, w = rect_hw
    mask = np.zeros((H, W), dtype=bool)
    for rr, cc in _stamp_positions(shape, rect_top_left, rect_hw):
        rs = max(0, rr)
        re = min(H, rr + h)
        cs = max(0, cc)
        ce = min(W, cc + w)
        if rs < re and cs < ce:
            mask[rs:re, cs:ce] = True
    return mask


def _mask_symmetry_ok(mask: np.ndarray) -> bool:
    patch = mask.astype(int)
    return not (
        np.array_equal(patch, np.fliplr(patch))
        or np.array_equal(patch, np.flipud(patch))
        or np.array_equal(patch, np.rot90(patch, 2))
    )


def _min_manhattan(a: np.ndarray, b: np.ndarray) -> int:
    best = 10**9
    for r0, c0 in a:
        d = np.abs(b[:, 0] - int(r0)) + np.abs(b[:, 1] - int(c0))
        best = min(best, int(d.min()))
    return best


def _build_chain_mask(
    shape: tuple[int, int],
    starts: list[tuple[int, int]],
    branch: str,
    rect_hw: tuple[int, int],
    clipped: bool,
    variant: int,
    add_tail: bool,
) -> np.ndarray | None:
    H, W = shape
    h, w = rect_hw
    coords: set[tuple[int, int]] = set()

    def add_global(base: tuple[int, int], local_cells: list[tuple[int, int]]) -> bool:
        br, bc = base
        for dr, dc in local_cells:
            rr = br + dr
            cc = bc + dc
            if not (0 <= rr < H and 0 <= cc < W):
                return False
            coords.add((rr, cc))
        return True

    if branch == "right":
        first = [(h - 1, w - 1), (h - 2, w - 1)] if variant == 0 else [(h - 1, w - 1), (h - 1, w - 2)]
        middle = (
            [(0, j) for j in range(w)] + [(i, w - 1) for i in range(1, h)]
            if variant == 0
            else [(i, 0) for i in range(h)] + [(h - 1, j) for j in range(1, w)]
        )
        last = [(0, 0), (1, 0)] if clipped or variant == 0 else [(0, 0), (0, 1)]
        bridges = [
            (starts[0][0] + h, starts[0][1] + w - 1),
            (starts[1][0] + h, starts[1][1] + w - 1),
        ]
        tail = (bridges[0][0], bridges[0][1] - 1) if variant == 0 else (bridges[1][0], bridges[1][1] - 1)
    else:
        first = [(h - 1, 0), (h - 2, 0)] if variant == 0 else [(h - 1, 0), (h - 1, 1)]
        middle = (
            [(0, j) for j in range(w)] + [(i, 0) for i in range(1, h)]
            if variant == 0
            else [(i, w - 1) for i in range(h)] + [(h - 1, j) for j in range(w - 1)]
        )
        last = [(0, w - 1), (1, w - 1)] if clipped or variant == 0 else [(0, w - 1), (0, w - 2)]
        bridges = [
            (starts[0][0] + h, starts[0][1]),
            (starts[1][0] + h, starts[1][1]),
        ]
        tail = (bridges[0][0], bridges[0][1] + 1) if variant == 0 else (bridges[1][0], bridges[1][1] + 1)

    if not add_global(starts[0], first):
        return None
    if not add_global(starts[1], middle):
        return None
    if not add_global(starts[2], last):
        return None
    for rr, cc in bridges:
        if not (0 <= rr < H and 0 <= cc < W):
            return None
        coords.add((rr, cc))
    if add_tail:
        rr, cc = tail
        if 0 <= rr < H and 0 <= cc < W:
            coords.add((rr, cc))

    mask = np.zeros((H, W), dtype=bool)
    for rr, cc in coords:
        mask[rr, cc] = True
    return mask


def _rect_and_mask(grid: np.ndarray) -> tuple[tuple[int, int], tuple[int, int], np.ndarray]:
    g = np.asarray(grid, dtype=int)
    blue = g == BLUE
    grey = g == GREY

    blue_comps = _components(blue)
    grey_comps = _components(grey)
    assert len(blue_comps) == 1, "expected one blue component"
    assert len(grey_comps) == 1, "expected one grey mask component"

    box = _bbox(blue)
    assert box is not None
    r0, c0, r1, c1 = box
    h = r1 - r0 + 1
    w = c1 - c0 + 1
    assert int(blue.sum()) == h * w, "blue object must be a solid rectangle"
    assert np.all(g[r0 : r1 + 1, c0 : c1 + 1] == BLUE), "bbox must be fully blue"
    return (r0, c0), (h, w), grey


def main(grid: np.ndarray) -> np.ndarray:
    """
    Keep only the overlap between the diagonal stamp copies of the blue rectangle and the grey mask, recolored red.

    Input structure:
    - A black background contains exactly one blue solid rectangle in the upper part of the grid.
    - A separate grey 4-connected irregular mask sits lower in the grid.
    - The blue rectangle and the grey mask do not overlap in the input.

    Rule:
    - Let the blue rectangle have height h and width w.
    - Stamp copies of that rectangle from its top-left corner at all integer offsets on the four diagonal rays: (+h,+w), (+h,-w), (-h,+w), (-h,-w), plus the original rectangle.
    - Clip any stamp that leaves the grid.
    - Keep only pixels that are both stamped blue pixels and grey mask pixels, recolor those surviving pixels red, and turn every other pixel black.

    Key insight:
    - The output is the intersection of two independent spatial patterns: the strict diagonal stamp lattice determined by the rectangle dimensions, and the irregular grey mask supplied in the input.
    """
    g = np.asarray(grid, dtype=int)
    assert g.ndim == 2
    rect_top_left, rect_hw, grey = _rect_and_mask(g)
    stamped = _stamp_mask(g.shape, rect_top_left, rect_hw)
    out = np.zeros_like(g)
    out[stamped & grey] = RED
    return out


def _contributing_stamps(
    shape: tuple[int, int], rect_top_left: tuple[int, int], rect_hw: tuple[int, int], grey: np.ndarray
) -> dict[tuple[int, int], int]:
    H, W = shape
    h, w = rect_hw
    counts: dict[tuple[int, int], int] = {}
    for rr, cc in _stamp_positions(shape, rect_top_left, rect_hw):
        rs = max(0, rr)
        re = min(H, rr + h)
        cs = max(0, cc)
        ce = min(W, cc + w)
        if rs >= re or cs >= ce:
            continue
        hit = int(grey[rs:re, cs:ce].sum())
        if hit:
            counts[(rr, cc)] = hit
    return counts


def _validate_semantics(grid: np.ndarray) -> bool:
    try:
        out = main(grid)
    except AssertionError:
        return False

    rect_top_left, rect_hw, grey = _rect_and_mask(grid)
    h, w = rect_hw
    H, W = grid.shape
    grey_coords = np.argwhere(grey)
    blue_coords = np.argwhere(grid == BLUE)
    out_coords = np.argwhere(out == RED)

    if grey_coords.shape[0] < 8 or grey_coords.shape[0] > 15:
        return False
    if out_coords.shape[0] < 5 or out_coords.shape[0] > 10:
        return False
    if np.any((out != BLACK) & (out != RED)):
        return False
    if np.array_equal(out > 0, grey):
        return False
    if not _mask_symmetry_ok(grey):
        return False
    if len(_components(grey)) != 1:
        return False
    mask_box = _bbox(grey)
    if mask_box is None:
        return False
    mr0, mc0, mr1, mc1 = mask_box
    if int(grey.sum()) == (mr1 - mr0 + 1) * (mc1 - mc0 + 1):
        return False
    if _min_manhattan(blue_coords, grey_coords) < 2:
        return False
    if rect_top_left[0] + h > H // 2:
        return False
    if grey_coords[:, 0].mean() < H * 0.48:
        return False

    stamped = _stamp_mask(grid.shape, rect_top_left, rect_hw)
    off_stamp = grey & ~stamped
    if int(off_stamp.sum()) < 2:
        return False

    contrib = _contributing_stamps(grid.shape, rect_top_left, rect_hw, grey)
    if len(contrib) < 3:
        return False
    if not any(0 < count < h * w for count in contrib.values()):
        return False

    full_red_rects = 0
    for rr, cc in contrib:
        rs = max(0, rr)
        re = min(H, rr + h)
        cs = max(0, cc)
        ce = min(W, cc + w)
        if rs >= re or cs >= ce:
            continue
        visible_area = (re - rs) * (ce - cs)
        red_area = int(np.count_nonzero(out[rs:re, cs:ce] == RED))
        if red_area == visible_area:
            full_red_rects += 1
    if full_red_rects == len(contrib):
        return False

    return True


def _gen_attempt() -> np.ndarray | None:
    H = random.randint(*GRID_ROWS_RANGE)
    W = random.randint(*GRID_COLS_RANGE)
    h, w = random.choices(RECT_DIM_CHOICES, weights=RECT_DIM_WEIGHTS, k=1)[0]
    branch = random.choice(["left", "right"])
    clipped = random.random() < 0.35

    max_r0 = min(H // 2 - h, H - 5 * h)
    if max_r0 < 0:
        return None
    r_candidates = list(range(max_r0 + 1))
    if not r_candidates:
        return None

    if branch == "left":
        if clipped:
            c_min = 3 * w
            c_max = min(4 * w - 1, W - w)
        else:
            c_min = 4 * w
            c_max = W - w
    else:
        if clipped:
            c_min = max(0, W - 5 * w + 1)
            c_max = min(W - 4 * w, W - w)
        else:
            c_min = 0
            c_max = W - 5 * w
    if c_min > c_max:
        return None

    r0 = random.choice(r_candidates)
    c0 = random.randint(c_min, c_max)

    if branch == "left":
        starts = [(r0 + 2 * h, c0 - 2 * w), (r0 + 3 * h, c0 - 3 * w), (r0 + 4 * h, c0 - 4 * w)]
    else:
        starts = [(r0 + 2 * h, c0 + 2 * w), (r0 + 3 * h, c0 + 3 * w), (r0 + 4 * h, c0 + 4 * w)]

    grey_mask = _build_chain_mask(
        (H, W),
        starts,
        branch,
        (h, w),
        clipped=clipped,
        variant=random.randint(0, 1),
        add_tail=random.random() < 0.5,
    )
    if grey_mask is None:
        return None

    grid = np.zeros((H, W), dtype=int)
    grid[r0 : r0 + h, c0 : c0 + w] = BLUE
    grid[grey_mask] = GREY

    if not _validate_semantics(grid):
        return None
    return grid


def generate_input() -> np.ndarray:
    for _ in range(MAX_GENERATION_ATTEMPTS):
        grid = _gen_attempt()
        if grid is not None:
            return grid
    raise RuntimeError("failed to generate a valid input for puzzle 8")
