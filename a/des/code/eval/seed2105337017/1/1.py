import numpy as np


BLACK = 0
GREY = 5
SEED_COLORS = (1, 2, 3, 4, 6, 7, 8, 9)
SIDES = ("top", "right", "bottom", "left")
CLOCKWISE = {
    "top": "right",
    "right": "bottom",
    "bottom": "left",
    "left": "top",
}

_RNG = np.random.default_rng()

GRID_H_RANGE = (16, 31)
GRID_W_RANGE = (16, 31)
REGION_COUNT_RANGE = (1, 5)
OUTER_H_RANGE = (6, 13)
OUTER_W_RANGE = (6, 13)
PLACEMENT_PADDING = 2
PLACEMENT_TRIES = 120
GENERATE_ATTEMPTS = 300


def _components(mask: np.ndarray) -> list[list[tuple[int, int]]]:
    h, w = mask.shape
    seen = np.zeros((h, w), dtype=bool)
    comps: list[list[tuple[int, int]]] = []
    for r in range(h):
        for c in range(w):
            if not mask[r, c] or seen[r, c]:
                continue
            stack = [(r, c)]
            seen[r, c] = True
            comp: list[tuple[int, int]] = []
            while stack:
                cr, cc = stack.pop()
                comp.append((cr, cc))
                for dr, dc in ((1, 0), (-1, 0), (0, 1), (0, -1)):
                    nr, nc = cr + dr, cc + dc
                    if 0 <= nr < h and 0 <= nc < w and mask[nr, nc] and not seen[nr, nc]:
                        seen[nr, nc] = True
                        stack.append((nr, nc))
            comps.append(comp)
    return comps


def _frame_bboxes(grid: np.ndarray) -> list[tuple[int, int, int, int]]:
    frames: list[tuple[int, int, int, int]] = []
    for comp in _components(grid == GREY):
        rows = [r for r, _ in comp]
        cols = [c for _, c in comp]
        r0, r1 = min(rows), max(rows)
        c0, c1 = min(cols), max(cols)
        assert r1 - r0 >= 2 and c1 - c0 >= 2, "frame must enclose a non-empty interior"

        expected = set()
        for c in range(c0, c1 + 1):
            expected.add((r0, c))
            expected.add((r1, c))
        for r in range(r0, r1 + 1):
            expected.add((r, c0))
            expected.add((r, c1))
        assert set(comp) == expected, "grey component must be a 1-pixel rectangular frame"
        frames.append((r0, c0, r1, c1))
    frames.sort()
    return frames


def _ordered_seeds(
    grid: np.ndarray, r0: int, c0: int, r1: int, c1: int
) -> tuple[int, int, str]:
    interior = grid[r0 + 1 : r1, c0 + 1 : c1]
    seed_rc = np.argwhere((interior != BLACK) & (interior != GREY))
    assert seed_rc.shape[0] == 2, "each frame interior must contain exactly two seeds"
    assert not np.any(interior == GREY), "grey should appear only on the border"

    seeds = []
    for rr, cc in seed_rc:
        ar, ac = r0 + 1 + int(rr), c0 + 1 + int(cc)
        seeds.append((ar, ac, int(grid[ar, ac])))

    assert seeds[0][2] != seeds[1][2], "seed colors must be distinct"
    assert abs(seeds[0][0] - seeds[1][0]) + abs(seeds[0][1] - seeds[1][1]) == 1, (
        "seeds must be adjacent"
    )

    if seeds[0][0] == seeds[1][0]:
        left, right = sorted(seeds, key=lambda item: item[1])
        dist_left = left[1] - (c0 + 1)
        dist_right = (c1 - 1) - right[1]
        if dist_left < dist_right:
            return left[2], right[2], "right"
        if dist_right < dist_left:
            return right[2], left[2], "left"

        pair_mid = (left[1] + right[1]) / 2.0
        box_mid = (c0 + c1) / 2.0
        if pair_mid <= box_mid:
            return left[2], right[2], "right"
        return right[2], left[2], "left"

    top, bottom = sorted(seeds, key=lambda item: item[0])
    dist_top = top[0] - (r0 + 1)
    dist_bottom = (r1 - 1) - bottom[0]
    if dist_top < dist_bottom:
        return top[2], bottom[2], "bottom"
    if dist_bottom < dist_top:
        return bottom[2], top[2], "top"

    pair_mid = (top[0] + bottom[0]) / 2.0
    box_mid = (r0 + r1) / 2.0
    if pair_mid <= box_mid:
        return top[2], bottom[2], "bottom"
    return bottom[2], top[2], "top"


def _paint_spiral(
    out: np.ndarray,
    r0: int,
    c0: int,
    r1: int,
    c1: int,
    color_a: int,
    color_b: int,
    start_side: str,
) -> None:
    top, left = r0 + 1, c0 + 1
    bottom, right = r1 - 1, c1 - 1
    side = start_side
    step = 0

    while top <= bottom and left <= right:
        color = color_a if step % 2 == 0 else color_b
        if side == "top":
            out[top, left : right + 1] = color
            top += 1
        elif side == "right":
            out[top : bottom + 1, right] = color
            right -= 1
        elif side == "bottom":
            out[bottom, left : right + 1] = color
            bottom -= 1
        else:
            out[top : bottom + 1, left] = color
            left += 1
        step += 1
        side = CLOCKWISE[side]


def main(grid: np.ndarray) -> np.ndarray:
    """
    Fill each grey rectangular frame by peeling one-cell-wide boundary strips
    in clockwise order, starting from the side implied by the seed pair.

    The first strip uses the outward seed's color; the second uses the inward
    seed's color; then the two colors alternate as the rectangle contracts.
    """
    g = np.asarray(grid, dtype=int)
    assert g.ndim == 2, "expected a 2D grid"

    out = g.copy()
    for r0, c0, r1, c1 in _frame_bboxes(g):
        color_a, color_b, start_side = _ordered_seeds(g, r0, c0, r1, c1)
        _paint_spiral(out, r0, c0, r1, c1, color_a, color_b, start_side)
    return out


def _draw_frame(grid: np.ndarray, r0: int, c0: int, r1: int, c1: int) -> None:
    grid[r0, c0 : c1 + 1] = GREY
    grid[r1, c0 : c1 + 1] = GREY
    grid[r0 : r1 + 1, c0] = GREY
    grid[r0 : r1 + 1, c1] = GREY


def _place_seed_pair(
    grid: np.ndarray,
    r0: int,
    c0: int,
    r1: int,
    c1: int,
    start_side: str,
    colors: tuple[int, int],
    rng: np.random.Generator,
) -> None:
    top, left = r0 + 1, c0 + 1
    bottom, right = r1 - 1, c1 - 1
    inner_h = bottom - top + 1
    inner_w = right - left + 1
    color_a, color_b = colors

    if start_side == "right":
        row = int(rng.integers(top, bottom + 1))
        max_offset = (inner_w - 3) // 2
        offset = int(rng.integers(0, max_offset + 1))
        a = (row, left + offset)
        b = (row, left + offset + 1)
    elif start_side == "left":
        row = int(rng.integers(top, bottom + 1))
        max_offset = (inner_w - 3) // 2
        offset = int(rng.integers(0, max_offset + 1))
        a = (row, right - offset)
        b = (row, right - offset - 1)
    elif start_side == "bottom":
        col = int(rng.integers(left, right + 1))
        max_offset = (inner_h - 3) // 2
        offset = int(rng.integers(0, max_offset + 1))
        a = (top + offset, col)
        b = (top + offset + 1, col)
    else:
        col = int(rng.integers(left, right + 1))
        max_offset = (inner_h - 3) // 2
        offset = int(rng.integers(0, max_offset + 1))
        a = (bottom - offset, col)
        b = (bottom - offset - 1, col)

    grid[a] = color_a
    grid[b] = color_b


def _can_place(used: np.ndarray, r0: int, c0: int, r1: int, c1: int) -> bool:
    pr0 = max(0, r0 - PLACEMENT_PADDING)
    pc0 = max(0, c0 - PLACEMENT_PADDING)
    pr1 = min(used.shape[0] - 1, r1 + PLACEMENT_PADDING)
    pc1 = min(used.shape[1] - 1, c1 + PLACEMENT_PADDING)
    return not used[pr0 : pr1 + 1, pc0 : pc1 + 1].any()


def _mark_used(used: np.ndarray, r0: int, c0: int, r1: int, c1: int) -> None:
    pr0 = max(0, r0 - PLACEMENT_PADDING)
    pc0 = max(0, c0 - PLACEMENT_PADDING)
    pr1 = min(used.shape[0] - 1, r1 + PLACEMENT_PADDING)
    pc1 = min(used.shape[1] - 1, c1 + PLACEMENT_PADDING)
    used[pr0 : pr1 + 1, pc0 : pc1 + 1] = True


def _gen_attempt(rng: np.random.Generator) -> np.ndarray | None:
    h = int(rng.integers(*GRID_H_RANGE))
    w = int(rng.integers(*GRID_W_RANGE))
    region_count = int(rng.integers(*REGION_COUNT_RANGE))

    grid = np.full((h, w), BLACK, dtype=int)
    used = np.zeros((h, w), dtype=bool)
    side_order = list(rng.permutation(np.array(SIDES)))
    if region_count > len(side_order):
        extra = list(rng.choice(SIDES, size=region_count - len(side_order), replace=True))
        side_order.extend(extra)

    frames: list[tuple[int, int, int, int, str, tuple[int, int]]] = []
    for idx in range(region_count):
        start_side = side_order[idx]
        placed = False
        for _ in range(PLACEMENT_TRIES):
            outer_h = int(rng.integers(*OUTER_H_RANGE))
            outer_w = int(rng.integers(*OUTER_W_RANGE))
            if start_side in ("left", "right") and outer_w < 6:
                continue
            if start_side in ("top", "bottom") and outer_h < 6:
                continue
            if outer_h >= h - 1 or outer_w >= w - 1:
                continue

            r0 = int(rng.integers(1, h - outer_h))
            c0 = int(rng.integers(1, w - outer_w))
            r1 = r0 + outer_h - 1
            c1 = c0 + outer_w - 1
            if not _can_place(used, r0, c0, r1, c1):
                continue

            colors = tuple(int(v) for v in rng.choice(SEED_COLORS, size=2, replace=False))
            frames.append((r0, c0, r1, c1, start_side, colors))
            _mark_used(used, r0, c0, r1, c1)
            placed = True
            break
        if not placed:
            return None

    for r0, c0, r1, c1, start_side, colors in frames:
        _draw_frame(grid, r0, c0, r1, c1)
        _place_seed_pair(grid, r0, c0, r1, c1, start_side, colors, rng)

    out = main(grid)
    if np.array_equal(out, grid):
        return None
    return grid


def generate_input() -> np.ndarray:
    for _ in range(GENERATE_ATTEMPTS):
        grid = _gen_attempt(_RNG)
        if grid is not None:
            return grid
    raise RuntimeError("failed to generate a valid grid")
