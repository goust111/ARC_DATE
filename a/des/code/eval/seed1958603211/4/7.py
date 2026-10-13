import random
from collections import Counter, deque

import numpy as np


def _most_common_color(values: np.ndarray) -> int:
    vals, counts = np.unique(np.asarray(values, dtype=int), return_counts=True)
    return int(vals[int(np.argmax(counts))])


def _components(mask: np.ndarray) -> list[list[tuple[int, int]]]:
    h, w = mask.shape
    seen = np.zeros((h, w), dtype=bool)
    comps: list[list[tuple[int, int]]] = []
    for r in range(h):
        for c in range(w):
            if not mask[r, c] or seen[r, c]:
                continue
            seen[r, c] = True
            q = deque([(r, c)])
            pts: list[tuple[int, int]] = []
            while q:
                cr, cc = q.popleft()
                pts.append((cr, cc))
                for nr, nc in ((cr - 1, cc), (cr + 1, cc), (cr, cc - 1), (cr, cc + 1)):
                    if 0 <= nr < h and 0 <= nc < w and mask[nr, nc] and not seen[nr, nc]:
                        seen[nr, nc] = True
                        q.append((nr, nc))
            comps.append(pts)
    return comps


def _bbox(points: list[tuple[int, int]]) -> tuple[int, int, int, int]:
    rows = [r for r, _ in points]
    cols = [c for _, c in points]
    return min(rows), min(cols), max(rows), max(cols)


def _find_board_boxes(grid: np.ndarray, bg: int) -> list[tuple[int, int, int, int]]:
    groups: dict[tuple[int, int], list[tuple[int, int, int, int]]] = {}
    for pts in _components(grid != bg):
        r0, c0, r1, c1 = _bbox(pts)
        h = r1 - r0 + 1
        w = c1 - c0 + 1
        if h < 4 or w < 4 or len(pts) != h * w:
            continue
        sub = grid[r0 : r1 + 1, c0 : c1 + 1]
        if np.unique(sub).size < 2:
            continue
        groups.setdefault((h, w), []).append((r0, c0, r1, c1))

    assert groups, "expected at least one solid rectangle group"
    shape, boxes = max(
        groups.items(),
        key=lambda item: (len(item[1]) * item[0][0] * item[0][1], len(item[1]), item[0][0] * item[0][1]),
    )
    del shape
    assert len(boxes) >= 2, "expected multiple mask boards"
    return sorted(boxes)


def _mode_color(colors: list[int]) -> int:
    counts = Counter(colors).most_common()
    assert counts, "each hidden cell must be visible at least once"
    if len(counts) > 1:
        assert counts[0][1] > counts[1][1], "visible colors must have a unique winner"
    return int(counts[0][0])


def _reconstruct_pattern(boards: list[np.ndarray], mask_color: int) -> np.ndarray:
    h, w = boards[0].shape
    out = np.zeros((h, w), dtype=int)
    for r in range(h):
        for c in range(w):
            visible = [int(board[r, c]) for board in boards if int(board[r, c]) != mask_color]
            out[r, c] = _mode_color(visible)
    return out


def _legend_mapping(
    grid: np.ndarray,
    bg: int,
    board_boxes: list[tuple[int, int, int, int]],
) -> dict[int, int]:
    free = np.ones(grid.shape, dtype=bool)
    for r0, c0, r1, c1 in board_boxes:
        free[r0 : r1 + 1, c0 : c1 + 1] = False

    mapping: dict[int, int] = {}
    for pts in _components((grid != bg) & free):
        r0, c0, r1, c1 = _bbox(pts)
        if len(pts) != 2 or r0 != r1 or c1 != c0 + 1:
            continue
        left = int(grid[r0, c0])
        right = int(grid[r0, c1])
        mapping[left] = right

    assert mapping, "expected at least one legend pair"
    return mapping


def main(grid: np.ndarray) -> np.ndarray:
    """
    Rebuild the hidden shared pattern from multiple grey mask boards, keep the
    majority color version, then recolor it with the legend pairs below.
    """
    g = np.asarray(grid, dtype=int)
    bg = _most_common_color(g)
    board_boxes = _find_board_boxes(g, bg)
    boards = [g[r0 : r1 + 1, c0 : c1 + 1] for r0, c0, r1, c1 in board_boxes]

    board_pixels = np.concatenate([board.ravel() for board in boards])
    mask_color = _most_common_color(board_pixels)
    pattern = _reconstruct_pattern(boards, mask_color)
    mapping = _legend_mapping(g, bg, board_boxes)

    base_colors = set(map(int, np.unique(pattern)))
    assert base_colors <= set(mapping), "legend must map every base color"

    out = pattern.copy()
    for src, dst in mapping.items():
        out[pattern == int(src)] = int(dst)
    return out


def _voronoi_pattern(h: int, w: int, k: int, rng: random.Random) -> np.ndarray:
    area = h * w
    rr, cc = np.indices((h, w))
    min_cells = max(2, area // (k * 4))
    cells = [(r, c) for r in range(h) for c in range(w)]

    for _ in range(120):
        seeds = np.array(rng.sample(cells, k), dtype=int)
        row_weight = rng.randint(1, 3)
        col_weight = rng.randint(1, 3)
        jitter = np.array([rng.random() * 0.05 for _ in range(k)], dtype=float)[:, None, None]
        dists = (
            row_weight * np.abs(rr[None, :, :] - seeds[:, 0][:, None, None])
            + col_weight * np.abs(cc[None, :, :] - seeds[:, 1][:, None, None])
            + jitter
        )
        regions = np.argmin(dists, axis=0).astype(int)
        counts = np.bincount(regions.ravel(), minlength=k)
        if int(counts.min()) < min_cells:
            continue
        if rng.random() < 0.5:
            regions = np.flipud(regions)
        if rng.random() < 0.5:
            regions = np.fliplr(regions)
        return regions
    raise RuntimeError("failed to build a balanced hidden pattern")


def _minority_window_mask(h: int, w: int, rng: random.Random) -> np.ndarray:
    area = h * w
    low = max(1, area // 6)
    high = max(low, (3 * area) // 4)
    for _ in range(80):
        mask = np.zeros((h, w), dtype=bool)
        for _ in range(rng.randint(1, 3)):
            rh = rng.randint(1, h)
            rw = rng.randint(1, w)
            r0 = rng.randint(0, h - rh)
            c0 = rng.randint(0, w - rw)
            mask[r0 : r0 + rh, c0 : c0 + rw] = True
        visible = int(mask.sum())
        if low <= visible <= high and visible < area:
            return mask
    raise RuntimeError("failed to build a minority mask window layout")


def _render_board(pattern: np.ndarray, palette: list[int], window_mask: np.ndarray, mask_color: int) -> np.ndarray:
    colored = np.take(np.asarray(palette, dtype=int), pattern)
    board = np.full(pattern.shape, int(mask_color), dtype=int)
    board[window_mask] = colored[window_mask]
    return board


def _legend_row(base_colors: list[int], target_colors: list[int], bg: int, rng: random.Random) -> np.ndarray:
    pairs = list(zip(base_colors, target_colors))
    rng.shuffle(pairs)
    gap = rng.randint(1, 2)
    width = len(pairs) * 2 + (len(pairs) - 1) * gap
    row = np.full((1, width), int(bg), dtype=int)
    c = 0
    for i, (src, dst) in enumerate(pairs):
        row[0, c] = int(src)
        row[0, c + 1] = int(dst)
        c += 2
        if i + 1 < len(pairs):
            c += gap
    return row


def _compose_scene(
    boards: list[np.ndarray],
    legend: np.ndarray,
    bg: int,
    rng: random.Random,
) -> np.ndarray:
    bh, bw = boards[0].shape
    row1_count = rng.choice([2, 3])
    row2_count = len(boards) - row1_count
    gap_x = rng.randint(1, 2)
    gap_y = rng.randint(1, 2)
    top_margin = rng.randint(1, 3)
    side_margin = rng.randint(1, 3)
    lower_gap = rng.randint(2, 4)
    bottom_margin = rng.randint(1, 3)

    row1_w = row1_count * bw + (row1_count - 1) * gap_x
    row2_w = row2_count * bw + (row2_count - 1) * gap_x
    content_w = max(row1_w, row2_w, legend.shape[1])
    width = content_w + 2 * side_margin + rng.randint(0, 3)

    row1_y = top_margin
    row2_y = row1_y + bh + gap_y
    legend_y = row2_y + bh + lower_gap
    height = legend_y + legend.shape[0] + bottom_margin

    grid = np.full((height, width), int(bg), dtype=int)
    order = boards[:]
    rng.shuffle(order)

    starts = [
        rng.randint(side_margin, width - side_margin - row1_w),
        rng.randint(side_margin, width - side_margin - row2_w),
    ]
    counts = [row1_count, row2_count]
    ys = [row1_y, row2_y]
    idx = 0
    for row_idx in range(2):
        x = starts[row_idx]
        for _ in range(counts[row_idx]):
            grid[ys[row_idx] : ys[row_idx] + bh, x : x + bw] = order[idx]
            idx += 1
            x += bw + gap_x

    legend_x = rng.randint(side_margin, width - side_margin - legend.shape[1])
    grid[legend_y : legend_y + legend.shape[0], legend_x : legend_x + legend.shape[1]] = legend
    return grid


def _gen_attempt() -> np.ndarray | None:
    rng = random.Random()
    mask_color = 5
    bg = rng.choice([c for c in range(10) if c != mask_color])
    h = rng.randint(5, 8)
    w = rng.randint(5, 8)
    k = rng.choice([2, 3])

    base_pool = [c for c in range(10) if c not in {bg, mask_color}]
    base_colors = rng.sample(base_pool, k)
    target_pool = [c for c in base_pool if c not in base_colors]
    target_colors = rng.sample(target_pool, k)
    alt_colors = rng.sample(target_pool, k)

    regions = _voronoi_pattern(h, w, k, rng)

    r_cut = rng.randint(1, h - 1)
    c_cut = rng.randint(1, w - 1)
    majority_masks = []

    top = np.zeros((h, w), dtype=bool)
    top[:r_cut, :] = True
    majority_masks.append(top)

    bottom = np.zeros((h, w), dtype=bool)
    bottom[r_cut:, :] = True
    majority_masks.append(bottom)

    left = np.zeros((h, w), dtype=bool)
    left[:, :c_cut] = True
    majority_masks.append(left)

    right = np.zeros((h, w), dtype=bool)
    right[:, c_cut:] = True
    majority_masks.append(right)

    boards = [_render_board(regions, base_colors, mask, mask_color) for mask in majority_masks]
    minority_mask = _minority_window_mask(h, w, rng)
    boards.append(_render_board(regions, alt_colors, minority_mask, mask_color))

    legend = _legend_row(base_colors, target_colors, bg, rng)
    grid = _compose_scene(boards, legend, bg, rng)

    try:
        out = main(grid)
    except Exception:
        return None

    expected = np.take(np.asarray(target_colors, dtype=int), regions)
    if np.array_equal(out, expected):
        return grid
    return None


def generate_input() -> np.ndarray:
    for _ in range(300):
        grid = _gen_attempt()
        if grid is not None:
            return grid
    raise RuntimeError("failed to generate a valid masked-pattern puzzle")
