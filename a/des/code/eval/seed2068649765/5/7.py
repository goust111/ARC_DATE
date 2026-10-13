from __future__ import annotations

import random

import numpy as np


BG = 0
MAROON = 9
NON_BLACK_COLORS = tuple(range(1, 10))
MAX_GENERATION_ATTEMPTS = 400


def _merge_arrays(base: np.ndarray, overlay: np.ndarray, priority_rank: dict[int, int]) -> np.ndarray:
    out = np.array(base, copy=True)
    default_rank = len(priority_rank) + 10
    rows, cols = out.shape
    for r in range(rows):
        for c in range(cols):
            a = int(base[r, c])
            b = int(overlay[r, c])
            if a == BG:
                out[r, c] = b
            elif b == BG or a == b:
                out[r, c] = a
            else:
                ra = priority_rank.get(a, default_rank)
                rb = priority_rank.get(b, default_rank)
                out[r, c] = a if ra <= rb else b
    return out


def _read_priority_line(block: np.ndarray) -> list[int] | None:
    h, w = block.shape
    candidates: list[tuple[int, list[int]]] = []

    if h == 2 and w >= 3:
        for r in range(2):
            vals = [int(v) for v in block[r, :].tolist()]
            if BG in vals:
                continue
            uniq = len(set(vals))
            if uniq == len(vals):
                candidates.append((uniq, vals))
    elif w == 2 and h >= 3:
        for c in range(2):
            vals = [int(v) for v in block[:, c].tolist()]
            if BG in vals:
                continue
            uniq = len(set(vals))
            if uniq == len(vals):
                candidates.append((uniq, vals))

    if not candidates:
        return None
    candidates.sort(key=lambda item: item[0], reverse=True)
    return candidates[0][1]


def _corner_scan_bbox(grid: np.ndarray, corner: str) -> tuple[int, int, int, int] | None:
    h, w = grid.shape
    top = corner in {"tl", "tr"}
    left = corner in {"tl", "bl"}
    r0 = 0 if top else h - 1
    c0 = 0 if left else w - 1
    if int(grid[r0, c0]) == BG:
        return None

    r1 = r0 + (1 if top else -1)
    c1 = c0 + (1 if left else -1)
    if not (0 <= r1 < h and 0 <= c1 < w):
        return None

    width = 0
    while True:
        c = c0 + (width if left else -width)
        if not (0 <= c < w):
            break
        if int(grid[r0, c]) == BG or int(grid[r1, c]) == BG:
            break
        width += 1

    height = 0
    while True:
        r = r0 + (height if top else -height)
        if not (0 <= r < h):
            break
        if int(grid[r, c0]) == BG or int(grid[r, c1]) == BG:
            break
        height += 1

    if width >= 3 and height == 2:
        rr = sorted((r0, r1))
        cc = sorted((c0, c0 + (width - 1 if left else -(width - 1))))
        rb, cb = rr[1], cc[1]
        if np.any(grid[rr[0] : rb + 1, cc[0] : cb + 1] == BG):
            return None
        inward_r = rb + 1 if top else rr[0] - 1
        inward_c = cb + 1 if left else cc[0] - 1
        if 0 <= inward_r < h and np.any(grid[inward_r, cc[0] : cb + 1] != BG):
            return None
        if 0 <= inward_c < w and np.any(grid[rr[0] : rb + 1, inward_c] != BG):
            return None
        return rr[0], cc[0], rb, cb

    if height >= 3 and width == 2:
        rr = sorted((r0, r0 + (height - 1 if top else -(height - 1))))
        cc = sorted((c0, c1))
        rb, cb = rr[1], cc[1]
        if np.any(grid[rr[0] : rb + 1, cc[0] : cb + 1] == BG):
            return None
        inward_r = rb + 1 if top else rr[0] - 1
        inward_c = cb + 1 if left else cc[0] - 1
        if 0 <= inward_r < h and np.any(grid[inward_r, cc[0] : cb + 1] != BG):
            return None
        if 0 <= inward_c < w and np.any(grid[rr[0] : rb + 1, inward_c] != BG):
            return None
        return rr[0], cc[0], rb, cb

    return None


def _detect_legend(grid: np.ndarray) -> tuple[tuple[int, int, int, int], list[int]]:
    candidates: list[tuple[int, tuple[int, int, int, int], list[int]]] = []
    for corner in ("tl", "tr", "bl", "br"):
        bbox = _corner_scan_bbox(grid, corner)
        if bbox is None:
            continue
        r0, c0, r1, c1 = bbox
        priority = _read_priority_line(grid[r0 : r1 + 1, c0 : c1 + 1])
        if priority is None:
            continue
        area = (r1 - r0 + 1) * (c1 - c0 + 1)
        candidates.append((area, bbox, priority))

    assert candidates, "expected one corner legend"
    candidates.sort(key=lambda item: item[0], reverse=True)
    best = candidates[0]
    if len(candidates) > 1:
        assert best[0] > candidates[1][0], "legend detection is ambiguous"
    return best[1], best[2]


def _find_fold_line(grid: np.ndarray) -> tuple[str, int]:
    maroons = np.argwhere(grid == MAROON)
    assert maroons.size > 0, "expected maroon hinge markers"

    row_hits: dict[int, list[int]] = {}
    col_hits: dict[int, list[int]] = {}
    for r, c in maroons:
        row_hits.setdefault(int(r), []).append(int(c))
        col_hits.setdefault(int(c), []).append(int(r))

    candidates: list[tuple[int, str, int]] = []
    for r, cols in row_hits.items():
        if len(cols) == 2:
            cols.sort()
            candidates.append((cols[1] - cols[0], "h", r))
    for c, rows in col_hits.items():
        if len(rows) == 2:
            rows.sort()
            candidates.append((rows[1] - rows[0], "v", c))

    assert candidates, "failed to locate the hinge line"
    candidates.sort(reverse=True)
    best = candidates[0]
    if len(candidates) > 1:
        assert best[0] > candidates[1][0], "hinge line is ambiguous"
    return best[1], best[2]


def _fold_horizontal(grid: np.ndarray, fold_r: int, priority_rank: dict[int, int]) -> np.ndarray:
    top = grid[:fold_r, :]
    bottom = grid[fold_r + 1 :, :]
    if top.shape[0] <= bottom.shape[0]:
        out = np.array(bottom, copy=True)
        overlay = top[::-1, :]
        out[: overlay.shape[0], :] = _merge_arrays(out[: overlay.shape[0], :], overlay, priority_rank)
        return out

    out = np.array(top, copy=True)
    overlay = bottom[::-1, :]
    out[-overlay.shape[0] :, :] = _merge_arrays(out[-overlay.shape[0] :, :], overlay, priority_rank)
    return out


def _fold_vertical(grid: np.ndarray, fold_c: int, priority_rank: dict[int, int]) -> np.ndarray:
    left = grid[:, :fold_c]
    right = grid[:, fold_c + 1 :]
    if left.shape[1] <= right.shape[1]:
        out = np.array(right, copy=True)
        overlay = left[:, ::-1]
        out[:, : overlay.shape[1]] = _merge_arrays(out[:, : overlay.shape[1]], overlay, priority_rank)
        return out

    out = np.array(left, copy=True)
    overlay = right[:, ::-1]
    out[:, -overlay.shape[1] :] = _merge_arrays(out[:, -overlay.shape[1] :], overlay, priority_rank)
    return out


def main(grid: np.ndarray) -> np.ndarray:
    """
    Fold the grid along the maroon-marked hinge line and merge overlaps by legend priority.

    Input structure
    - The background is black.
    - An isolated 2xN or Nx2 legend sits in one corner. One row/column lists colors from
      highest priority to lowest priority.
    - Outside the legend, exactly one row or one column contains two maroon pixels that
      mark the folding hinge.

    Rule
    - Remove the corner legend from the working canvas after reading the priority order.
    - Locate the hinge from the two maroon pixels, discard the hinge row/column itself,
      and fold the shorter side onto the longer side.
    - In overlapped cells, non-black beats black; if both are non-black and different,
      keep the color that appears earlier in the legend's priority line.
    """
    g = np.asarray(grid, dtype=int)
    assert g.ndim == 2, "expected a 2D grid"

    legend_bbox, priority = _detect_legend(g)
    priority_rank = {int(color): idx for idx, color in enumerate(priority)}

    work = np.array(g, copy=True)
    r0, c0, r1, c1 = legend_bbox
    work[r0 : r1 + 1, c0 : c1 + 1] = BG

    axis, hinge = _find_fold_line(work)
    if axis == "h":
        return _fold_horizontal(work, hinge, priority_rank)
    return _fold_vertical(work, hinge, priority_rank)


# --------------- generator ---------------

HEIGHT_RANGE = (14, 22)
WIDTH_RANGE = (14, 22)
PRIORITY_COLOR_COUNT = (4, 5)  # includes maroon
FILL_TARGET = (0.18, 0.32)
SHAPE_ATTEMPTS = 120


def _legend_block(priority: list[int], accent: int, orientation: str, sequence_first: bool) -> np.ndarray:
    if orientation == "h":
        block = np.full((2, len(priority)), accent, dtype=int)
        seq_row = 0 if sequence_first else 1
        block[seq_row, :] = np.asarray(priority, dtype=int)
        return block

    block = np.full((len(priority), 2), accent, dtype=int)
    seq_col = 0 if sequence_first else 1
    block[:, seq_col] = np.asarray(priority, dtype=int)
    return block


def _place_corner_block(grid: np.ndarray, block: np.ndarray, corner: str) -> tuple[int, int, int, int]:
    h, w = grid.shape
    bh, bw = block.shape
    if corner == "tl":
        r0, c0 = 0, 0
    elif corner == "tr":
        r0, c0 = 0, w - bw
    elif corner == "bl":
        r0, c0 = h - bh, 0
    elif corner == "br":
        r0, c0 = h - bh, w - bw
    else:
        raise ValueError(corner)

    grid[r0 : r0 + bh, c0 : c0 + bw] = block
    return r0, c0, r0 + bh - 1, c0 + bw - 1


def _legend_forbidden(shape: tuple[int, int], bbox: tuple[int, int, int, int]) -> np.ndarray:
    h, w = shape
    r0, c0, r1, c1 = bbox
    mask = np.zeros((h, w), dtype=bool)
    rr0 = max(0, r0 - 1)
    cc0 = max(0, c0 - 1)
    rr1 = min(h, r1 + 2)
    cc1 = min(w, c1 + 2)
    mask[rr0:rr1, cc0:cc1] = True
    return mask


def _side_masks(shape: tuple[int, int], axis: str, hinge: int) -> tuple[np.ndarray, np.ndarray]:
    h, w = shape
    a = np.zeros((h, w), dtype=bool)
    b = np.zeros((h, w), dtype=bool)
    if axis == "h":
        a[:hinge, :] = True
        b[hinge + 1 :, :] = True
    else:
        a[:, :hinge] = True
        b[:, hinge + 1 :] = True
    return a, b


def _mirror_cell(r: int, c: int, axis: str, hinge: int) -> tuple[int, int]:
    if axis == "h":
        return 2 * hinge - r, c
    return r, 2 * hinge - c


def _pick_sizes(axis: str, legend_shape: tuple[int, int], corner: str) -> tuple[int, int, int, str] | None:
    leg_h, leg_w = legend_shape

    if axis == "h":
        min_top = 5
        min_bottom = 5
        if corner in {"tl", "tr"}:
            min_top = max(min_top, leg_h + 2)
        else:
            min_bottom = max(min_bottom, leg_h + 2)

        for _ in range(100):
            top_short = random.choice([True, False])
            short_len = random.randint(4, 8)
            long_len = short_len + random.randint(1, 5)
            top_len = short_len if top_short else long_len
            bottom_len = long_len if top_short else short_len
            height = top_len + bottom_len + 1
            if not (HEIGHT_RANGE[0] <= height <= HEIGHT_RANGE[1]):
                continue
            if top_len < min_top or bottom_len < min_bottom:
                continue
            short_side = "top" if top_short else "bottom"
            return height, random.randint(max(12, leg_w + 4), WIDTH_RANGE[1]), top_len, short_side
        return None

    min_left = 5
    min_right = 5
    if corner in {"tl", "bl"}:
        min_left = max(min_left, leg_w + 2)
    else:
        min_right = max(min_right, leg_w + 2)

    for _ in range(100):
        left_short = random.choice([True, False])
        short_len = random.randint(4, 8)
        long_len = short_len + random.randint(1, 5)
        left_len = short_len if left_short else long_len
        right_len = long_len if left_short else short_len
        width = left_len + right_len + 1
        if not (WIDTH_RANGE[0] <= width <= WIDTH_RANGE[1]):
            continue
        if left_len < min_left or right_len < min_right:
            continue
        short_side = "left" if left_short else "right"
        return random.randint(max(12, leg_h + 4), HEIGHT_RANGE[1]), width, left_len, short_side
    return None


def _sample_block_cells(
    shape: tuple[int, int],
    axis: str,
    short_side: str,
    hinge: int,
    allowed: np.ndarray,
    block_h: int,
    block_w: int,
    grid: np.ndarray,
) -> list[tuple[int, int]] | None:
    h, w = shape
    for _ in range(200):
        if axis == "h":
            max_c0 = w - block_w
            if max_c0 < 0:
                return None
            c0 = random.randint(0, max_c0)
            if short_side == "top":
                max_r0 = hinge - block_h
                if max_r0 < 0:
                    return None
                r0 = random.randint(0, max_r0)
            else:
                min_r0 = hinge + 1
                max_r0 = h - block_h
                if min_r0 > max_r0:
                    return None
                r0 = random.randint(min_r0, max_r0)
        else:
            max_r0 = h - block_h
            if max_r0 < 0:
                return None
            r0 = random.randint(0, max_r0)
            if short_side == "left":
                max_c0 = hinge - block_w
                if max_c0 < 0:
                    return None
                c0 = random.randint(0, max_c0)
            else:
                min_c0 = hinge + 1
                max_c0 = w - block_w
                if min_c0 > max_c0:
                    return None
                c0 = random.randint(min_c0, max_c0)

        cells = [(r0 + dr, c0 + dc) for dr in range(block_h) for dc in range(block_w)]
        mirrored = [_mirror_cell(r, c, axis, hinge) for r, c in cells]
        if any(not allowed[r, c] for r, c in cells):
            continue
        if any(not allowed[r, c] for r, c in mirrored):
            continue
        if any(int(grid[r, c]) != BG for r, c in cells):
            continue
        if any(int(grid[r, c]) != BG for r, c in mirrored):
            continue
        return cells
    return None


def _paint_cells(grid: np.ndarray, cells: list[tuple[int, int]], color: int) -> None:
    for r, c in cells:
        grid[r, c] = color


def _place_conflict(
    grid: np.ndarray,
    allowed: np.ndarray,
    axis: str,
    short_side: str,
    hinge: int,
    short_color: int,
    long_color: int,
) -> bool:
    if axis == "h":
        block_h = random.randint(1, 2)
        block_w = random.randint(2, 4)
    else:
        block_h = random.randint(2, 4)
        block_w = random.randint(1, 2)

    cells = _sample_block_cells(grid.shape, axis, short_side, hinge, allowed, block_h, block_w, grid)
    if cells is None:
        return False

    mirrored = [_mirror_cell(r, c, axis, hinge) for r, c in cells]
    _paint_cells(grid, cells, short_color)
    _paint_cells(grid, mirrored, long_color)
    return True


def _place_seed_patch(
    grid: np.ndarray,
    allowed_mask: np.ndarray,
    color: int,
    horizontal_bias: bool,
) -> bool:
    for _ in range(100):
        if horizontal_bias:
            block_h = random.randint(1, 2)
            block_w = random.randint(1, 3)
        else:
            block_h = random.randint(1, 3)
            block_w = random.randint(1, 2)
        rows, cols = grid.shape
        if block_h > rows or block_w > cols:
            return False
        r0 = random.randint(0, rows - block_h)
        c0 = random.randint(0, cols - block_w)
        sub_mask = allowed_mask[r0 : r0 + block_h, c0 : c0 + block_w]
        sub_grid = grid[r0 : r0 + block_h, c0 : c0 + block_w]
        if np.all(sub_mask) and np.all(sub_grid == BG):
            sub_grid[:, :] = color
            return True
    return False


def _fill_with_shapes(grid: np.ndarray, allowed: np.ndarray, colors: list[int], target_fraction: float) -> None:
    usable = int(np.sum(allowed))
    if usable == 0:
        return
    target = int(target_fraction * usable)
    for _ in range(SHAPE_ATTEMPTS):
        filled = int(np.sum((grid != BG) & allowed))
        if filled >= target:
            return
        use_horizontal = random.random() < 0.5
        block_h = random.randint(1, 3 if use_horizontal else 2)
        block_w = random.randint(1, 2 if use_horizontal else 4)
        rows, cols = grid.shape
        if block_h > rows or block_w > cols:
            continue
        r0 = random.randint(0, rows - block_h)
        c0 = random.randint(0, cols - block_w)
        sub_mask = allowed[r0 : r0 + block_h, c0 : c0 + block_w]
        sub_grid = grid[r0 : r0 + block_h, c0 : c0 + block_w]
        if not np.all(sub_mask):
            continue
        if not np.all(sub_grid == BG):
            continue
        sub_grid[:, :] = random.choice(colors)


def _add_hinge_markers(grid: np.ndarray, axis: str, hinge: int, forbidden: np.ndarray, paint_colors: list[int]) -> bool:
    h, w = grid.shape
    if axis == "h":
        choices = [c for c in range(1, w - 1) if not forbidden[hinge, c]]
        if len(choices) < 2:
            return False
        left, right = sorted(random.sample(choices, 2))
        grid[hinge, left] = MAROON
        grid[hinge, right] = MAROON
        extras = [c for c in choices if c not in {left, right}]
        for c in random.sample(extras, k=min(len(extras), random.randint(0, 3))):
            grid[hinge, c] = random.choice(paint_colors)
        return True

    choices = [r for r in range(1, h - 1) if not forbidden[r, hinge]]
    if len(choices) < 2:
        return False
    top, bottom = sorted(random.sample(choices, 2))
    grid[top, hinge] = MAROON
    grid[bottom, hinge] = MAROON
    extras = [r for r in choices if r not in {top, bottom}]
    for r in random.sample(extras, k=min(len(extras), random.randint(0, 3))):
        grid[r, hinge] = random.choice(paint_colors)
    return True


def _gen_attempt() -> np.ndarray | None:
    axis = random.choice(["h", "v"])
    corner = random.choice(["tl", "tr", "bl", "br"])
    legend_orientation = random.choice(["h", "v"])
    sequence_first = random.choice([True, False])

    priority_size = random.randint(*PRIORITY_COLOR_COUNT)
    priority_colors = random.sample([c for c in NON_BLACK_COLORS if c != MAROON], priority_size - 1)
    priority_colors.append(MAROON)
    random.shuffle(priority_colors)

    remaining = [c for c in NON_BLACK_COLORS if c not in priority_colors]
    accent = random.choice(remaining)
    legend = _legend_block(priority_colors, accent, legend_orientation, sequence_first)

    size_info = _pick_sizes(axis, legend.shape, corner)
    if size_info is None:
        return None

    if axis == "h":
        height, width, hinge, short_side = size_info
    else:
        height, width, hinge, short_side = size_info

    grid = np.full((height, width), BG, dtype=int)
    legend_bbox = _place_corner_block(grid, legend, corner)
    forbidden = _legend_forbidden(grid.shape, legend_bbox)

    interior = np.zeros_like(forbidden)
    interior[1:-1, 1:-1] = True
    allowed = interior & (~forbidden)
    if axis == "h":
        allowed[hinge, :] = False
    else:
        allowed[:, hinge] = False

    side_a, side_b = _side_masks(grid.shape, axis, hinge)
    if short_side in {"top", "left"}:
        short_mask = side_a
        long_mask = side_b
    else:
        short_mask = side_b
        long_mask = side_a

    short_allowed = allowed & short_mask
    long_allowed = allowed & long_mask
    if np.sum(short_allowed) < 12 or np.sum(long_allowed) < 18:
        return None

    paint_colors = [c for c in priority_colors if c != MAROON]
    rank = {color: idx for idx, color in enumerate(priority_colors)}
    sorted_paint = sorted(paint_colors, key=lambda color: rank[color])
    if len(sorted_paint) < 3:
        return None

    if not _place_conflict(grid, allowed, axis, short_side, hinge, sorted_paint[0], sorted_paint[-1]):
        return None
    if not _place_conflict(grid, allowed, axis, short_side, hinge, sorted_paint[-1], sorted_paint[1]):
        return None

    used_colors = {int(v) for v in np.unique(grid) if int(v) not in {BG, MAROON, accent}}
    for color in paint_colors:
        if color in used_colors:
            continue
        side_choice = short_allowed if random.random() < 0.5 else long_allowed
        if not _place_seed_patch(grid, side_choice, color, axis == "h"):
            if not _place_seed_patch(grid, short_allowed | long_allowed, color, axis == "h"):
                return None

    _fill_with_shapes(grid, short_allowed, paint_colors, random.uniform(*FILL_TARGET))
    _fill_with_shapes(grid, long_allowed, paint_colors, random.uniform(*FILL_TARGET))

    if not _add_hinge_markers(grid, axis, hinge, forbidden, paint_colors):
        return None

    try:
        out = main(grid)
    except Exception:
        return None

    if axis == "h":
        expected_rows = max(hinge, height - hinge - 1)
        if out.shape != (expected_rows, width):
            return None
    else:
        expected_cols = max(hinge, width - hinge - 1)
        if out.shape != (height, expected_cols):
            return None

    # Ensure the priority rule is visibly needed: the first conflict must let the short side win somewhere.
    work = np.array(grid, copy=True)
    r0, c0, r1, c1 = legend_bbox
    work[r0 : r1 + 1, c0 : c1 + 1] = BG
    simple = work[hinge + 1 :, :] if axis == "h" and short_side == "top" else None
    if axis == "h" and short_side == "bottom":
        simple = work[:hinge, :]
    if axis == "v" and short_side == "left":
        simple = work[:, hinge + 1 :]
    if axis == "v" and short_side == "right":
        simple = work[:, :hinge]
    if simple is None or np.array_equal(out, simple):
        return None

    return grid


def generate_input() -> np.ndarray:
    for _ in range(MAX_GENERATION_ATTEMPTS):
        grid = _gen_attempt()
        if grid is not None:
            return grid
    raise RuntimeError("failed to generate a valid fold-with-priority puzzle")
