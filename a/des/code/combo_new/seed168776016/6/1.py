from __future__ import annotations

import numpy as np


BG = 0
SEPARATOR_COLOR = 5
PALETTE = np.array([1, 2, 3, 4, 6, 7, 8, 9], dtype=int)
MAX_ATTEMPTS = 500

TWO_OBJECT_COUNTS = (
    (2, 2),
    (2, 3),
    (2, 4),
    (2, 5),
    (3, 3),
    (3, 4),
)
THREE_OBJECT_COUNTS = (
    (2, 2, 2),
    (2, 2, 3),
)


def _bbox(mask: np.ndarray) -> tuple[int, int, int, int] | None:
    coords = np.argwhere(mask)
    if coords.size == 0:
        return None
    r0, c0 = coords.min(axis=0)
    r1, c1 = coords.max(axis=0)
    return int(r0), int(c0), int(r1), int(c1)


def _count_enclosed_regions(walls: np.ndarray) -> int:
    h, w = map(int, walls.shape)
    seen = np.zeros_like(walls, dtype=bool)
    total = 0

    for sr in range(h):
        for sc in range(w):
            if walls[sr, sc] or seen[sr, sc]:
                continue
            stack = [(int(sr), int(sc))]
            seen[sr, sc] = True
            touches_edge = False
            while stack:
                r, c = stack.pop()
                if r == 0 or r == h - 1 or c == 0 or c == w - 1:
                    touches_edge = True
                for nr, nc in ((r - 1, c), (r + 1, c), (r, c - 1), (r, c + 1)):
                    if 0 <= nr < h and 0 <= nc < w and not walls[nr, nc] and not seen[nr, nc]:
                        seen[nr, nc] = True
                        stack.append((int(nr), int(nc)))
            if not touches_edge:
                total += 1
    return int(total)


def _find_separator_column(grid: np.ndarray) -> tuple[int, int]:
    candidates: list[tuple[int, int]] = []
    for c in range(grid.shape[1]):
        vals = np.unique(grid[:, c])
        if vals.size == 1 and int(vals[0]) != BG:
            candidates.append((int(c), int(vals[0])))
    assert len(candidates) == 1, "expected one full-height separator column"
    return candidates[0]


def _count_rooms_for_color(region: np.ndarray, color: int) -> int:
    mask = region == int(color)
    box = _bbox(mask)
    assert box is not None, "frame color must exist in the left region"
    r0, c0, r1, c1 = box
    walls = region[r0 : r1 + 1, c0 : c1 + 1] == int(color)
    return _count_enclosed_regions(walls)


def _find_anchor(global_grid: np.ndarray, sep_col: int, color: int) -> tuple[int, int]:
    right = global_grid[:, sep_col + 1 :]
    mask = right == int(color)
    box = _bbox(mask)
    assert box is not None, "anchor color must exist in the right region"
    r0, c0, r1, c1 = box
    patch = right[r0 : r1 + 1, c0 : c1 + 1]
    assert patch.shape == (3, 3), "anchor must be a 3x3 square"
    assert int(mask.sum()) == 9, "anchor must be solid"
    assert np.all(patch == int(color)), "anchor must be monochrome"
    return int(r0 + 1), int(sep_col + 1 + c0 + 1)


def _draw_nested_rectangles(out: np.ndarray, center_r: int, center_c: int, layers: int, color: int) -> None:
    out[center_r - 1 : center_r + 2, center_c - 1 : center_c + 2] = int(color)
    for layer in range(2, int(layers) + 1):
        radius = 2 * int(layer) - 1
        r0 = int(center_r - radius)
        r1 = int(center_r + radius)
        c0 = int(center_c - radius)
        c1 = int(center_c + radius)
        assert 0 <= r0 < r1 < out.shape[0]
        assert 0 <= c0 < c1 < out.shape[1]
        out[r0, c0 : c1 + 1] = int(color)
        out[r1, c0 : c1 + 1] = int(color)
        out[r0 : r1 + 1, c0] = int(color)
        out[r0 : r1 + 1, c1] = int(color)


def main(grid: np.ndarray) -> np.ndarray:
    """
    Count each partitioned frame's interior regions and draw that many concentric rectangles around the matching anchor.

    Input structure:
    - Black background with one full-height grey separator column splitting the grid into left and right regions.
    - The left region contains several differently colored rectangular frames whose interiors are subdivided by same-color horizontal and/or vertical wall segments.
    - The right region contains one solid 3x3 anchor square for each frame color, with enough empty space around each anchor.

    Rule:
    - For every frame color on the left, count the number of 4-connected black regions enclosed inside that frame's colored walls.
    - Find the right-side anchor of the same color and keep its 3x3 square as layer 1.
    - Add layers 2..N as same-color concentric rectangular borders centered on that anchor, with one black cell between successive borders.

    Key insight:
    - The frame's internal partitions encode a region count, and that count becomes the nesting depth of the same-colored rectangle pattern around the paired anchor.
    """
    g = np.asarray(grid, dtype=int)
    sep_col, sep_color = _find_separator_column(g)

    left = g[:, :sep_col]
    frame_colors = sorted(int(v) for v in np.unique(left) if int(v) not in {BG, sep_color})

    out = g.copy()
    for color in frame_colors:
        layers = _count_rooms_for_color(left, int(color))
        center_r, center_c = _find_anchor(g, sep_col, int(color))
        _draw_nested_rectangles(out, center_r, center_c, int(layers), int(color))
    return out


def _build_partition(num_regions: int, rng: np.random.Generator) -> tuple[int, int, list[tuple[str, int, int, int, int, int]]]:
    dims = [
        (rows, cols)
        for rows in range(1, 4)
        for cols in range(1, 4)
        if rows * cols >= int(num_regions)
    ]
    for _ in range(200):
        rows, cols = dims[int(rng.integers(0, len(dims)))]
        leaves = [(0, rows - 1, 0, cols - 1)]
        splits: list[tuple[str, int, int, int, int, int]] = []

        for _step in range(int(num_regions) - 1):
            candidates: list[tuple[int, str]] = []
            for idx, (r0, r1, c0, c1) in enumerate(leaves):
                if r1 > r0:
                    candidates.append((int(idx), "h"))
                if c1 > c0:
                    candidates.append((int(idx), "v"))
            if not candidates:
                break

            pick = candidates[int(rng.integers(0, len(candidates)))]
            idx, orient = pick
            r0, r1, c0, c1 = leaves.pop(idx)
            if orient == "h":
                cut = int(rng.integers(r0, r1))
                leaves.extend([(r0, cut, c0, c1), (cut + 1, r1, c0, c1)])
                splits.append(("h", cut, r0, r1, c0, c1))
            else:
                cut = int(rng.integers(c0, c1))
                leaves.extend([(r0, r1, c0, cut), (r0, r1, cut + 1, c1)])
                splits.append(("v", cut, r0, r1, c0, c1))

        if len(leaves) == int(num_regions):
            return int(rows), int(cols), splits
    raise RuntimeError("failed to build a partition tree")


def _render_partition_walls(
    rows: int,
    cols: int,
    splits: list[tuple[str, int, int, int, int, int]],
    rng: np.random.Generator,
) -> np.ndarray:
    row_sizes = rng.integers(1, 3, size=int(rows))
    col_sizes = rng.integers(1, 3, size=int(cols))

    row_bounds = [0]
    for size in row_sizes:
        row_bounds.append(int(row_bounds[-1] + int(size) + 1))
    col_bounds = [0]
    for size in col_sizes:
        col_bounds.append(int(col_bounds[-1] + int(size) + 1))

    h = int(row_bounds[-1] + 1)
    w = int(col_bounds[-1] + 1)
    walls = np.zeros((h, w), dtype=bool)
    walls[0, :] = True
    walls[-1, :] = True
    walls[:, 0] = True
    walls[:, -1] = True

    for orient, cut, r0, r1, c0, c1 in splits:
        if orient == "h":
            rr = int(row_bounds[int(cut) + 1])
            cc0 = int(col_bounds[int(c0)])
            cc1 = int(col_bounds[int(c1) + 1])
            walls[rr, cc0 : cc1 + 1] = True
        else:
            cc = int(col_bounds[int(cut) + 1])
            rr0 = int(row_bounds[int(r0)])
            rr1 = int(row_bounds[int(r1) + 1])
            walls[rr0 : rr1 + 1, cc] = True
    return walls


def _make_frame_patch(num_regions: int, color: int, rng: np.random.Generator) -> np.ndarray:
    for _ in range(200):
        rows, cols, splits = _build_partition(int(num_regions), rng)
        walls = _render_partition_walls(rows, cols, splits, rng)
        if walls.shape[0] > 10 or walls.shape[1] > 9:
            continue
        if _count_enclosed_regions(walls) != int(num_regions):
            continue
        return np.where(walls, int(color), BG)
    raise RuntimeError("failed to create a valid frame patch")


def _sample_counts(rng: np.random.Generator) -> list[int]:
    if bool(rng.integers(0, 2)):
        counts = list(TWO_OBJECT_COUNTS[int(rng.integers(0, len(TWO_OBJECT_COUNTS)))])
    else:
        counts = list(THREE_OBJECT_COUNTS[int(rng.integers(0, len(THREE_OBJECT_COUNTS)))])
    rng.shuffle(counts)
    return [int(v) for v in counts]


def _stack_height(counts: list[int]) -> int:
    sides = [4 * int(v) - 1 for v in counts]
    return int(sum(sides) + max(0, len(sides) - 1))


def _place_anchor_footprints(
    height: int,
    width: int,
    counts: list[int],
    rng: np.random.Generator,
) -> list[tuple[int, int]] | None:
    order = sorted(range(len(counts)), key=lambda i: -counts[i])
    occupied = np.zeros((int(height), int(width)), dtype=bool)
    centers: list[tuple[int, int] | None] = [None] * len(counts)

    for idx in order:
        radius = 2 * int(counts[idx]) - 1
        candidates: list[tuple[int, int]] = []
        for cr in range(radius, int(height) - radius):
            for cc in range(radius, int(width) - radius):
                rr0 = max(0, cr - radius - 1)
                rr1 = min(int(height), cr + radius + 2)
                cc0 = max(0, cc - radius - 1)
                cc1 = min(int(width), cc + radius + 2)
                if not occupied[rr0:rr1, cc0:cc1].any():
                    candidates.append((int(cr), int(cc)))
        if not candidates:
            return None
        cr, cc = candidates[int(rng.integers(0, len(candidates)))]
        centers[idx] = (int(cr), int(cc))
        rr0 = max(0, cr - radius - 1)
        rr1 = min(int(height), cr + radius + 2)
        cc0 = max(0, cc - radius - 1)
        cc1 = min(int(width), cc + radius + 2)
        occupied[rr0:rr1, cc0:cc1] = True

    assert all(center is not None for center in centers)
    return [(int(r), int(c)) for r, c in centers if r is not None]


def _draw_expected_output(
    inp: np.ndarray,
    sep_col: int,
    colors: list[int],
    counts: list[int],
    centers: list[tuple[int, int]],
) -> np.ndarray:
    out = inp.copy()
    for color, count, (cr, cc) in zip(colors, counts, centers):
        _draw_nested_rectangles(out, int(cr), int(sep_col + 1 + cc), int(count), int(color))
    return out


def _generate_candidate(rng: np.random.Generator) -> np.ndarray | None:
    counts = _sample_counts(rng)
    n = len(counts)
    colors = [int(v) for v in rng.choice(PALETTE, size=n, replace=False)]

    frame_patches = [_make_frame_patch(int(count), int(color), rng) for count, color in zip(counts, colors)]
    frame_heights = [int(p.shape[0]) for p in frame_patches]
    frame_widths = [int(p.shape[1]) for p in frame_patches]

    gaps = [int(rng.integers(1, 3)) for _ in range(max(0, n - 1))]
    left_needed_h = int(sum(frame_heights) + sum(gaps))
    left_min_w = int(max(frame_widths) + 1)
    right_min_w = int(max(4 * count - 1 for count in counts))
    min_total_w = int(left_min_w + 1 + right_min_w)
    if min_total_w > 30:
        return None

    stack_needed_h = _stack_height(counts)
    base_h = int(max(left_needed_h, stack_needed_h))
    if base_h > 30:
        return None

    width_slack = int(30 - min_total_w)
    extra_left = int(rng.integers(0, width_slack + 1))
    extra_right = int(rng.integers(0, width_slack - extra_left + 1))
    left_w = int(left_min_w + extra_left)
    right_w = int(right_min_w + extra_right)
    width = int(left_w + 1 + right_w)

    extra_h = int(rng.integers(0, 31 - base_h))
    height = int(base_h + extra_h)
    sep_col = int(left_w)

    centers = _place_anchor_footprints(height, right_w, counts, rng)
    if centers is None:
        return None

    grid = np.full((height, width), BG, dtype=int)
    grid[:, sep_col] = SEPARATOR_COLOR

    free_vertical = int(height - left_needed_h)
    top = int(rng.integers(0, free_vertical + 1))
    y = top
    for idx, patch in enumerate(frame_patches):
        x = int(rng.integers(0, left_w - patch.shape[1] + 1))
        ph, pw = patch.shape
        grid[y : y + ph, x : x + pw][patch != BG] = patch[patch != BG]
        y += int(ph)
        if idx < len(gaps):
            y += int(gaps[idx])

    for color, (cr, cc) in zip(colors, centers):
        grid[cr - 1 : cr + 2, sep_col + 1 + cc - 1 : sep_col + 1 + cc + 2] = int(color)

    expected = _draw_expected_output(grid, sep_col, colors, counts, centers)
    try:
        actual = main(grid)
    except Exception:
        return None
    if not np.array_equal(actual, expected):
        return None
    return grid


def generate_input() -> np.ndarray:
    rng = np.random.default_rng()
    for _ in range(MAX_ATTEMPTS):
        candidate = _generate_candidate(rng)
        if candidate is not None:
            return candidate
    raise RuntimeError(f"failed to generate a valid input after {MAX_ATTEMPTS} attempts")
