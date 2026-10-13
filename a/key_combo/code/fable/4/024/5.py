from __future__ import annotations

import random

import numpy as np


BG = 0
GRAY = 5
GEN_ATTEMPTS = 500
GEN_SCENARIOS = (
    (True, True),
    (True, False),
    (False, True),
    (False, False),
)
GEN_SCENARIO_WEIGHTS = (4, 2, 2, 1)
GEN_MOTIF_COLORS = (1, 2, 3, 4, 6, 7, 8, 9)


def _get_bg(grid: np.ndarray) -> int:
    values, counts = np.unique(np.asarray(grid, dtype=int), return_counts=True)
    return int(values[np.argmax(counts)])


def _extract_motifs(grid: np.ndarray) -> tuple[int, dict, dict]:
    g = np.asarray(grid, dtype=int)
    bg = _get_bg(g)
    colors = [int(color) for color in np.unique(g) if int(color) != bg]
    assert len(colors) == 2, "expected exactly two foreground colors"

    motifs = []
    for color in colors:
        coords = np.argwhere(g == color)
        assert coords.size > 0, "foreground color must have pixels"
        (r0, c0), (r1, c1) = coords.min(axis=0), coords.max(axis=0)
        r0, c0, r1, c1 = map(int, (r0, c0, r1, c1))
        mask = np.zeros((r1 - r0 + 1, c1 - c0 + 1), dtype=bool)
        mask[coords[:, 0] - r0, coords[:, 1] - c0] = True
        motifs.append(
            {
                "color": color,
                "r0": r0,
                "c0": c0,
                "height": int(mask.shape[0]),
                "width": int(mask.shape[1]),
                "mask": mask,
            }
        )

    motifs.sort(key=lambda motif: motif["c0"])
    left, right = motifs
    height, width = map(int, g.shape)
    assert 0 < left["c0"] and left["c0"] + left["width"] < width
    assert 0 < right["c0"] and right["c0"] + right["width"] < width
    assert left["c0"] + left["width"] <= right["c0"], "motifs must not overlap in the input"
    return bg, left, right


def _stamp_arrivals(shape: tuple[int, int], motif: dict, direction: int) -> tuple[np.ndarray, bool]:
    height, width = map(int, shape)
    mask_coords = np.argwhere(motif["mask"])
    rows = motif["r0"] + mask_coords[:, 0]
    rel_cols = mask_coords[:, 1]
    step = int(motif["width"])

    arrivals = np.full((height, width), -1, dtype=int)
    used_partial_copy = False
    k = 0

    while True:
        start_c = int(motif["c0"] + direction * k * step)
        end_c = start_c + step - 1
        if end_c < 0 or start_c >= width:
            break

        if start_c < 0 or end_c >= width:
            used_partial_copy = True

        cols = start_c + rel_cols
        keep = (0 <= cols) & (cols < width)
        if np.any(keep):
            rr = rows[keep]
            cc = cols[keep]
            unclaimed = arrivals[rr, cc] == -1
            arrivals[rr[unclaimed], cc[unclaimed]] = k
        k += 1

    return arrivals, used_partial_copy


def _solve_race(grid: np.ndarray) -> tuple[np.ndarray, dict]:
    g = np.asarray(grid, dtype=int)
    assert g.ndim == 2, "expected a 2D grid"

    bg, left, right = _extract_motifs(g)
    left_arrivals, left_partial = _stamp_arrivals(g.shape, left, direction=1)
    right_arrivals, right_partial = _stamp_arrivals(g.shape, right, direction=-1)

    out = np.full_like(g, bg)
    left_seen = left_arrivals >= 0
    right_seen = right_arrivals >= 0
    overlap = left_seen & right_seen

    left_only = left_seen & ~right_seen
    right_only = right_seen & ~left_seen
    left_first = overlap & (left_arrivals < right_arrivals)
    right_first = overlap & (right_arrivals < left_arrivals)
    tied = overlap & (left_arrivals == right_arrivals)

    out[left_only | left_first] = int(left["color"])
    out[right_only | right_first] = int(right["color"])
    out[tied] = GRAY

    meta = {
        "left": left,
        "right": right,
        "partials": (left_partial, right_partial),
        "overlap_count": int(np.count_nonzero(overlap)),
        "gray_count": int(np.count_nonzero(tied)),
    }
    return out, meta


def main(grid: np.ndarray) -> np.ndarray:
    """
    Repeat the left and right motifs toward each other by their own widths, letting the earlier copy keep each contested cell.

    Input structure:
    - A black background contains one monochrome motif on the left and one on the right.
    - The two motifs use different colors, leave a wide black gap between them, and do not touch the left or right border.
    - Each motif is interpreted through its tight bounding-box width, while its internal pixel mask may be irregular.

    Rule:
    - Copy the left motif to the right in steps of its own width, and copy the right motif to the left in steps of its own width.
    - Keep every full or truncated copy whose translated bbox still intersects the grid.
    - For cells covered by both sides, compare the copy indices k needed to reach that cell: the smaller k wins, and equal k values become gray.

    Key insight:
    - Travel time is measured in copy count, not geometric distance, so a wider motif can arrive sooner because each step advances by more columns.
    """
    return _solve_race(grid)[0]


def _connected(mask: np.ndarray) -> bool:
    coords = np.argwhere(mask)
    if coords.size == 0:
        return False

    start = tuple(map(int, coords[0]))
    seen = {start}
    stack = [start]
    height, width = mask.shape
    while stack:
        r, c = stack.pop()
        for dr, dc in ((1, 0), (-1, 0), (0, 1), (0, -1)):
            nr, nc = r + dr, c + dc
            if 0 <= nr < height and 0 <= nc < width and mask[nr, nc] and (nr, nc) not in seen:
                seen.add((nr, nc))
                stack.append((nr, nc))
    return len(seen) == int(mask.sum())


def _generate_mask(height: int, width: int) -> np.ndarray:
    area = height * width
    min_area = max(height + width - 1, (area * 45 + 99) // 100)
    cells = [(r, c) for r in range(height) for c in range(width)]

    for _ in range(80):
        target = random.randint(min_area, area)
        mask = np.ones((height, width), dtype=bool)
        while int(mask.sum()) > target:
            removed = False
            random.shuffle(cells)
            for r, c in cells:
                if not mask[r, c]:
                    continue
                mask[r, c] = False
                if mask[r].any() and mask[:, c].any() and _connected(mask):
                    removed = True
                    if int(mask.sum()) == target:
                        break
                else:
                    mask[r, c] = True
            if int(mask.sum()) == target:
                break
            if not removed:
                break
        if int(mask.sum()) == target:
            return mask
    return np.ones((height, width), dtype=bool)


def _choose_rows(grid_h: int, left_h: int, right_h: int) -> tuple[int, int] | None:
    for _ in range(80):
        left_r0 = random.randint(0, grid_h - left_h)
        right_r0 = random.randint(0, grid_h - right_h)
        overlap = min(left_r0 + left_h, right_r0 + right_h) - max(left_r0, right_r0)
        if overlap >= 1:
            return left_r0, right_r0
    return None


def _paint_motif(grid: np.ndarray, mask: np.ndarray, r0: int, c0: int, color: int) -> None:
    coords = np.argwhere(mask)
    grid[r0 + coords[:, 0], c0 + coords[:, 1]] = int(color)


def _candidate_grid(
    grid_h: int,
    grid_w: int,
    left_mask: np.ndarray,
    right_mask: np.ndarray,
    left_r0: int,
    right_r0: int,
    left_c0: int,
    right_c0: int,
    left_color: int,
    right_color: int,
) -> np.ndarray:
    grid = np.full((grid_h, grid_w), BG, dtype=int)
    _paint_motif(grid, left_mask, left_r0, left_c0, left_color)
    _paint_motif(grid, right_mask, right_r0, right_c0, right_color)
    return grid


def _accept_example(grid: np.ndarray, out: np.ndarray, meta: dict, want_tie: bool, want_partial: bool) -> bool:
    left_color = int(meta["left"]["color"])
    right_color = int(meta["right"]["color"])
    gray_count = int(meta["gray_count"])
    overlap_count = int(meta["overlap_count"])
    partial = any(meta["partials"])

    if np.count_nonzero(grid == BG) <= grid.size // 2:
        return False
    if overlap_count == 0:
        return False
    if not np.any(out == left_color) or not np.any(out == right_color):
        return False
    if want_tie:
        if gray_count < 2:
            return False
    elif gray_count != 0:
        return False
    if want_partial:
        if not partial:
            return False
    elif partial:
        return False
    if np.count_nonzero(out != BG) >= int(out.size * 0.7):
        return False
    return True


def _gen_attempt() -> np.ndarray | None:
    want_tie, want_partial = random.choices(GEN_SCENARIOS, weights=GEN_SCENARIO_WEIGHTS, k=1)[0]
    prefer_equal_widths = want_tie and random.random() < 0.45

    left_h = random.randint(2, 6)
    right_h = random.randint(2, 6)
    left_w = random.randint(2, 6)
    if prefer_equal_widths:
        right_w = left_w
    else:
        right_w = random.randint(2, 6)
        if random.random() < 0.75:
            while right_w == left_w:
                right_w = random.randint(2, 6)

    grid_h = random.randint(max(left_h, right_h) + 2, min(max(left_h, right_h) + 10, 20))
    min_w = left_w + right_w + 8
    grid_w = random.randint(max(min_w, 14), 30)

    rows = _choose_rows(grid_h, left_h, right_h)
    if rows is None:
        return None
    left_r0, right_r0 = rows

    left_mask = _generate_mask(left_h, left_w)
    right_mask = _generate_mask(right_h, right_w)
    left_color, right_color = random.sample(GEN_MOTIF_COLORS, 2)

    left_margin_max = min(5, grid_w - left_w - right_w - 4)
    right_margin_max = min(5, grid_w - left_w - right_w - 4)
    if left_margin_max < 1 or right_margin_max < 1:
        return None

    for _ in range(120):
        left_c0 = random.randint(1, left_margin_max)
        right_margin = random.randint(1, right_margin_max)
        right_c0 = grid_w - right_margin - right_w
        gap = right_c0 - (left_c0 + left_w)
        if gap < max(3, min(left_w, right_w) // 2 + 2):
            continue

        grid = _candidate_grid(
            grid_h,
            grid_w,
            left_mask,
            right_mask,
            left_r0,
            right_r0,
            left_c0,
            right_c0,
            left_color,
            right_color,
        )
        out, meta = _solve_race(grid)
        if not np.array_equal(main(grid), out):
            continue
        if _accept_example(grid, out, meta, want_tie=want_tie, want_partial=want_partial):
            return grid
    return None


def generate_input() -> np.ndarray:
    for _ in range(GEN_ATTEMPTS):
        grid = _gen_attempt()
        if grid is not None:
            return grid
    raise RuntimeError("failed to generate a valid input for puzzle 5")
