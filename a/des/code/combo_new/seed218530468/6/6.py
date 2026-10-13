import random

import numpy as np


BG = 0
GREY = 5
MAX_SIZE = 30
GEN_ATTEMPTS = 400
PAIR_COLORS = (1, 2, 3, 4, 6, 7, 8, 9)


def _neighbors4(r: int, c: int, shape: tuple[int, int]):
    h, w = shape
    if r > 0:
        yield r - 1, c
    if r + 1 < h:
        yield r + 1, c
    if c > 0:
        yield r, c - 1
    if c + 1 < w:
        yield r, c + 1


def _bbox_overlap(a: tuple[int, int, int, int], b: tuple[int, int, int, int], pad: int = 0) -> bool:
    ar0, ac0, ar1, ac1 = a
    br0, bc0, br1, bc1 = b
    return not (ar1 + pad < br0 or br1 + pad < ar0 or ac1 + pad < bc0 or bc1 + pad < ac0)


def _rect_area(rect: tuple[int, int, int, int]) -> int:
    r0, c0, r1, c1 = rect
    return (r1 - r0 + 1) * (c1 - c0 + 1)


def _border_cells(rect: tuple[int, int, int, int]) -> set[tuple[int, int]]:
    r0, c0, r1, c1 = rect
    cells: set[tuple[int, int]] = set()
    for c in range(c0, c1 + 1):
        cells.add((r0, c))
        cells.add((r1, c))
    for r in range(r0, r1 + 1):
        cells.add((r, c0))
        cells.add((r, c1))
    return cells


def _crop_non_bg(arr: np.ndarray) -> np.ndarray:
    pts = np.argwhere(arr != BG)
    if pts.size == 0:
        return arr.copy()
    r0, c0 = pts.min(axis=0)
    r1, c1 = pts.max(axis=0)
    return arr[int(r0) : int(r1) + 1, int(c0) : int(c1) + 1].copy()


def _draw_segment(arr: np.ndarray, p0: tuple[int, int], p1: tuple[int, int], color: int) -> None:
    r0, c0 = p0
    r1, c1 = p1
    if r0 == r1:
        step = 1 if c1 >= c0 else -1
        for c in range(c0, c1 + step, step):
            arr[r0, c] = color
        return
    if c0 == c1:
        step = 1 if r1 >= r0 else -1
        for r in range(r0, r1 + step, step):
            arr[r, c0] = color
        return
    raise ValueError("segments must be axis-aligned")


def _draw_polyline(arr: np.ndarray, points: list[tuple[int, int]], color: int) -> None:
    for p0, p1 in zip(points, points[1:]):
        _draw_segment(arr, p0, p1, color)


def _draw_frame(
    arr: np.ndarray,
    top: int,
    left: int,
    inner_h: int,
    inner_w: int,
    color: int,
    pattern: np.ndarray | None,
) -> None:
    outer_h = inner_h + 2
    outer_w = inner_w + 2
    arr[top, left : left + outer_w] = color
    arr[top + outer_h - 1, left : left + outer_w] = color
    arr[top : top + outer_h, left] = color
    arr[top : top + outer_h, left + outer_w - 1] = color
    if pattern is not None:
        inner = arr[top + 1 : top + 1 + inner_h, left + 1 : left + 1 + inner_w]
        inner[pattern] = GREY


def _find_rectangles_for_color(grid: np.ndarray, color: int) -> list[tuple[int, int, int, int]]:
    h, w = grid.shape
    rects: list[tuple[int, int, int, int]] = []
    for r0 in range(h - 2):
        for c0 in range(w - 2):
            if int(grid[r0, c0]) != color:
                continue
            for r1 in range(r0 + 2, h):
                if int(grid[r1, c0]) != color:
                    continue
                for c1 in range(c0 + 2, w):
                    if int(grid[r0, c1]) != color or int(grid[r1, c1]) != color:
                        continue
                    if not np.all(grid[r0, c0 : c1 + 1] == color):
                        continue
                    if not np.all(grid[r1, c0 : c1 + 1] == color):
                        continue
                    if not np.all(grid[r0 : r1 + 1, c0] == color):
                        continue
                    if not np.all(grid[r0 : r1 + 1, c1] == color):
                        continue
                    if np.any(grid[r0 + 1 : r1, c0 + 1 : c1] == color):
                        continue
                    rects.append((r0, c0, r1, c1))
    return rects


def _choose_pair_rectangles(grid: np.ndarray, color: int) -> tuple[tuple[int, int, int, int], tuple[int, int, int, int]] | None:
    rects = _find_rectangles_for_color(grid, color)
    if len(rects) < 2:
        return None

    best = None
    best_score = None
    for i in range(len(rects)):
        for j in range(i + 1, len(rects)):
            a = rects[i]
            b = rects[j]
            if _bbox_overlap(a, b):
                continue
            a_shape = (a[2] - a[0] - 1, a[3] - a[1] - 1)
            b_shape = (b[2] - b[0] - 1, b[3] - b[1] - 1)
            if a_shape != b_shape:
                continue
            score = _rect_area(a) + _rect_area(b)
            if best_score is None or score > best_score:
                best = (a, b)
                best_score = score
    return best


def _find_attachment(
    grid: np.ndarray,
    color: int,
    rect: tuple[int, int, int, int],
    border_union: set[tuple[int, int]],
) -> tuple[int, int]:
    border = _border_cells(rect)
    attachments: list[tuple[int, int]] = []
    for cell in sorted(border):
        r, c = cell
        outside_neighbors = [
            nbr
            for nbr in _neighbors4(r, c, grid.shape)
            if int(grid[nbr]) == color and nbr not in border_union
        ]
        if outside_neighbors:
            if len(outside_neighbors) != 1:
                raise AssertionError("attachment must connect to a single external path cell")
            attachments.append(cell)
    if len(attachments) != 1:
        raise AssertionError("each rectangle must expose exactly one attachment point")
    return attachments[0]


def _count_turns_along_path(
    grid: np.ndarray,
    color: int,
    rect_a: tuple[int, int, int, int],
    rect_b: tuple[int, int, int, int],
) -> int:
    border_union = _border_cells(rect_a) | _border_cells(rect_b)
    start = _find_attachment(grid, color, rect_a, border_union)
    end = _find_attachment(grid, color, rect_b, border_union)

    allowed = {tuple(coord) for coord in np.argwhere(grid == color)}
    allowed.difference_update(border_union)
    allowed.add(start)
    allowed.add(end)

    dirs: list[tuple[int, int]] = []
    prev = None
    cur = start
    budget = len(allowed) + 2
    while cur != end and budget > 0:
        budget -= 1
        nbrs = [nbr for nbr in _neighbors4(cur[0], cur[1], grid.shape) if nbr in allowed and nbr != prev]
        if len(nbrs) != 1:
            raise AssertionError("path between paired rectangles must be a simple chain")
        nxt = nbrs[0]
        dirs.append((nxt[0] - cur[0], nxt[1] - cur[1]))
        prev, cur = cur, nxt
    if cur != end:
        raise AssertionError("failed to reach target attachment")
    return sum(dirs[i] != dirs[i - 1] for i in range(1, len(dirs)))


def main(grid: np.ndarray) -> np.ndarray:
    """
    Pair same-colored framed regions, read path turns, and rotate the source pattern into the matching target.

    Input structure:
    - The background is black.
    - Several colors each form exactly two one-pixel-thick rectangular frames of the same size, plus a same-colored orthogonal path connecting the two frames.
    - For each color pair, one frame interior contains a grey asymmetric pattern while the other interior is empty black.

    Rule:
    - For every border color, identify which frame is the grey-pattern source and which empty frame is the target.
    - Trace the connecting path from source to target and count its right-angle turns T.
    - Rotate the source's grey interior pattern clockwise by (T mod 4) quarter-turns, then copy that rotated grey mask into the target interior.
    - Keep every frame, path, and source pattern pixel unchanged.

    Key insight:
    - The border color solves the pairing, but the path geometry is the hidden parameter: its turn count encodes the rotation that must be applied before filling the matching target frame.
    """
    g = np.asarray(grid, dtype=int)
    out = g.copy()

    for color in sorted(int(c) for c in np.unique(g) if int(c) not in {BG, GREY}):
        pair = _choose_pair_rectangles(g, color)
        if pair is None:
            continue

        rect_a, rect_b = pair
        a_inner = g[rect_a[0] + 1 : rect_a[2], rect_a[1] + 1 : rect_a[3]]
        b_inner = g[rect_b[0] + 1 : rect_b[2], rect_b[1] + 1 : rect_b[3]]
        a_has_grey = bool(np.any(a_inner == GREY))
        b_has_grey = bool(np.any(b_inner == GREY))
        if a_has_grey == b_has_grey:
            raise AssertionError("exactly one frame of each color must contain grey source pixels")

        source_rect, target_rect = (rect_a, rect_b) if a_has_grey else (rect_b, rect_a)
        turns = _count_turns_along_path(g, color, source_rect, target_rect)
        rotation = turns % 4

        source_mask = g[source_rect[0] + 1 : source_rect[2], source_rect[1] + 1 : source_rect[3]] == GREY
        rotated = np.rot90(source_mask, -rotation)
        target_view = out[target_rect[0] + 1 : target_rect[2], target_rect[1] + 1 : target_rect[3]]
        if rotated.shape != target_view.shape:
            raise AssertionError("rotated source pattern must match the target interior shape")
        target_view[:] = BG
        target_view[rotated] = GREY

    return out


def _pattern_ok(mask: np.ndarray) -> bool:
    h, w = mask.shape
    if int(mask.sum()) < 2 or int(mask.sum()) >= h * w:
        return False
    if not mask[0].any() or not mask[-1].any() or not mask[:, 0].any() or not mask[:, -1].any():
        return False
    if np.array_equal(mask, np.rot90(mask, 2)):
        return False
    if h == w and (np.array_equal(mask, np.rot90(mask, 1)) or np.array_equal(mask, np.rot90(mask, 3))):
        return False
    return True


def _make_pattern(rng: random.Random, h: int, w: int) -> np.ndarray:
    for _ in range(200):
        density = rng.uniform(0.35, 0.65)
        mask = np.array([[rng.random() < density for _ in range(w)] for _ in range(h)], dtype=bool)
        if _pattern_ok(mask):
            return mask
    raise RuntimeError("failed to generate a non-degenerate source pattern")


def _pair_canvas_valid(arr: np.ndarray, color: int, turns: int) -> bool:
    pair = _choose_pair_rectangles(arr, color)
    if pair is None:
        return False
    rect_a, rect_b = pair
    a_has_grey = bool(np.any(arr[rect_a[0] + 1 : rect_a[2], rect_a[1] + 1 : rect_a[3]] == GREY))
    b_has_grey = bool(np.any(arr[rect_b[0] + 1 : rect_b[2], rect_b[1] + 1 : rect_b[3]] == GREY))
    if a_has_grey == b_has_grey:
        return False
    try:
        actual_turns = _count_turns_along_path(arr, color, rect_a, rect_b)
    except AssertionError:
        return False
    return actual_turns == turns


def _build_pair_canvas_once(color: int, turns: int, rng: random.Random) -> np.ndarray:
    if turns % 2 == 1:
        inner_h = inner_w = rng.randint(2, 4)
    else:
        inner_h = rng.randint(2, 4)
        inner_w = rng.randint(2, 4)
    pattern = _make_pattern(rng, inner_h, inner_w)

    outer_h = inner_h + 2
    outer_w = inner_w + 2

    if turns == 0:
        gap = rng.randint(2, 5)
        attach = rng.randint(1, inner_h)
        src_top, src_left = 0, 0
        dst_top, dst_left = 0, outer_w + gap
        points = [
            (src_top + attach, src_left + outer_w - 1),
            (dst_top + attach, dst_left),
        ]
    elif turns == 1:
        gap_h = rng.randint(2, 5)
        gap_v = rng.randint(2, 5)
        attach_row = rng.randint(1, inner_h)
        attach_col = rng.randint(1, inner_w)
        src_top, src_left = 0, 0
        dst_top, dst_left = outer_h + gap_v, outer_w + gap_h
        end_col = dst_left + attach_col
        points = [
            (src_top + attach_row, src_left + outer_w - 1),
            (src_top + attach_row, end_col),
            (dst_top, end_col),
        ]
    elif turns == 2:
        gap_h = rng.randint(2, 6)
        src_top, src_left = 0, 0
        dst_top, dst_left = 0, outer_w + gap_h
        src_row = rng.randint(1, inner_h)
        dst_row = rng.randint(1, inner_h)
        while dst_row == src_row:
            dst_row = rng.randint(1, inner_h)
        mid_col = rng.randint(src_left + outer_w, dst_left - 1)
        points = [
            (src_top + src_row, src_left + outer_w - 1),
            (src_top + src_row, mid_col),
            (dst_top + dst_row, mid_col),
            (dst_top + dst_row, dst_left),
        ]
    else:
        gap_h = rng.randint(2, 5)
        gap_v = rng.randint(2, 5)
        attach_row = rng.randint(1, inner_h)
        attach_col = rng.randint(1, inner_w)
        src_top, src_left = 0, 0
        dst_top, dst_left = outer_h + gap_v, outer_w + gap_h
        start_row = src_top + attach_row
        end_col = dst_left + attach_col
        mid_col = rng.randint(src_left + outer_w, dst_left - 1)
        mid_row = rng.randint(start_row + 1, dst_top - 1)
        points = [
            (start_row, src_left + outer_w - 1),
            (start_row, mid_col),
            (mid_row, mid_col),
            (mid_row, end_col),
            (dst_top, end_col),
        ]

    max_r = max(max(r for r, _ in points), src_top + outer_h - 1, dst_top + outer_h - 1)
    max_c = max(max(c for _, c in points), src_left + outer_w - 1, dst_left + outer_w - 1)
    arr = np.full((max_r + 1, max_c + 1), BG, dtype=int)

    _draw_frame(arr, src_top, src_left, inner_h, inner_w, color, pattern)
    _draw_frame(arr, dst_top, dst_left, inner_h, inner_w, color, None)
    _draw_polyline(arr, points, color)

    if rng.random() < 0.5:
        arr = np.fliplr(arr)
    arr = np.rot90(arr, rng.randrange(4))
    return _crop_non_bg(arr)


def _make_pair_canvas(color: int, turns: int, rng: random.Random) -> np.ndarray:
    for _ in range(200):
        arr = _build_pair_canvas_once(color, turns, rng)
        if _pair_canvas_valid(arr, color, turns):
            return arr
    raise RuntimeError("failed to generate an unambiguous pair canvas")


def _place_pieces(pieces: list[np.ndarray], rng: random.Random) -> np.ndarray | None:
    canvas = np.full((MAX_SIZE, MAX_SIZE), BG, dtype=int)
    boxes: list[tuple[int, int, int, int]] = []

    for piece in sorted(pieces, key=lambda p: int((p != BG).sum()), reverse=True):
        h, w = piece.shape
        row_lo = 1 if h <= MAX_SIZE - 2 else 0
        row_hi = MAX_SIZE - h - 1 if h <= MAX_SIZE - 2 else MAX_SIZE - h
        col_lo = 1 if w <= MAX_SIZE - 2 else 0
        col_hi = MAX_SIZE - w - 1 if w <= MAX_SIZE - 2 else MAX_SIZE - w
        if row_hi < row_lo or col_hi < col_lo:
            return None

        placed = False
        for _ in range(300):
            top = rng.randint(row_lo, row_hi)
            left = rng.randint(col_lo, col_hi)
            box = (top, left, top + h - 1, left + w - 1)
            if any(_bbox_overlap(box, other, pad=1) for other in boxes):
                continue
            mask = piece != BG
            patch = canvas[top : top + h, left : left + w]
            if np.any(patch[mask] != BG):
                continue
            patch[mask] = piece[mask]
            boxes.append(box)
            placed = True
            break
        if not placed:
            return None

    return canvas


def _crop_with_margin(canvas: np.ndarray, rng: random.Random) -> np.ndarray:
    pts = np.argwhere(canvas != BG)
    if pts.size == 0:
        return canvas.copy()
    r0, c0 = pts.min(axis=0)
    r1, c1 = pts.max(axis=0)
    top_pad = rng.randint(0, min(3, int(r0)))
    left_pad = rng.randint(0, min(3, int(c0)))
    bottom_pad = rng.randint(0, min(3, int(canvas.shape[0] - 1 - r1)))
    right_pad = rng.randint(0, min(3, int(canvas.shape[1] - 1 - c1)))
    return canvas[int(r0) - top_pad : int(r1) + bottom_pad + 1, int(c0) - left_pad : int(c1) + right_pad + 1].copy()


def _gen_attempt() -> np.ndarray | None:
    rng = random.Random()
    pair_count = rng.choice((1, 2, 3))
    turns = rng.sample((0, 1, 2, 3), pair_count)
    colors = rng.sample(PAIR_COLORS, pair_count)

    pieces = [_make_pair_canvas(color, turn, rng) for color, turn in zip(colors, turns)]
    canvas = _place_pieces(pieces, rng)
    if canvas is None:
        return None

    grid = _crop_with_margin(canvas, rng)
    if grid.ndim != 2 or max(grid.shape) > MAX_SIZE:
        return None

    out = main(grid)
    if out.shape != grid.shape:
        return None
    if np.count_nonzero(out == GREY) <= np.count_nonzero(grid == GREY):
        return None
    return grid


def generate_input() -> np.ndarray:
    for _ in range(GEN_ATTEMPTS):
        grid = _gen_attempt()
        if grid is not None:
            return grid
    raise RuntimeError("failed to generate a valid turn-path rotation puzzle")
