import random

import numpy as np


BG = 0
FRAME = 5
COLOR_POOL = (1, 2, 3, 4, 6, 7, 8, 9)
GEN_RETRIES = 500


def _bbox_from_coords(coords: np.ndarray) -> tuple[int, int, int, int]:
    rows = coords[:, 0]
    cols = coords[:, 1]
    return int(rows.min()), int(cols.min()), int(rows.max()), int(cols.max())


def _boxes_overlap(
    a: tuple[int, int, int, int], b: tuple[int, int, int, int], pad: int = 0
) -> bool:
    ar0, ac0, ar1, ac1 = a
    br0, bc0, br1, bc1 = b
    return not (
        ar1 + pad < br0 or br1 + pad < ar0 or ac1 + pad < bc0 or bc1 + pad < ac0
    )


def _empty_canvas(height: int, width: int) -> np.ndarray:
    out = np.full((height, width), BG, dtype=int)
    out[0, :] = FRAME
    out[-1, :] = FRAME
    out[:, 0] = FRAME
    out[:, -1] = FRAME
    return out


def _find_canvas_bbox(grid: np.ndarray) -> tuple[int, int, int, int]:
    coords = np.argwhere(grid == FRAME)
    if coords.size == 0:
        raise ValueError("missing grey frame")
    r0, c0, r1, c1 = _bbox_from_coords(coords)
    if r1 - r0 < 2 or c1 - c0 < 2:
        raise ValueError("frame too small")
    if not np.all(grid[r0, c0 : c1 + 1] == FRAME):
        raise ValueError("invalid frame top edge")
    if not np.all(grid[r1, c0 : c1 + 1] == FRAME):
        raise ValueError("invalid frame bottom edge")
    if not np.all(grid[r0 : r1 + 1, c0] == FRAME):
        raise ValueError("invalid frame left edge")
    if not np.all(grid[r0 : r1 + 1, c1] == FRAME):
        raise ValueError("invalid frame right edge")
    return r0, c0, r1, c1


def _find_legend_order(grid: np.ndarray) -> tuple[int, list[int]]:
    h, w = grid.shape
    for r in range(h - 1, -1, -1):
        cols = [
            c
            for c in range(w)
            if int(grid[r, c]) not in (BG, FRAME)
        ]
        if len(cols) < 2:
            continue
        colors = [int(grid[r, c]) for c in cols]
        if len(set(colors)) != len(colors):
            continue
        isolated_vertically = True
        for c in cols:
            if r > 0 and int(grid[r - 1, c]) != BG:
                isolated_vertically = False
                break
            if r + 1 < h and int(grid[r + 1, c]) != BG:
                isolated_vertically = False
                break
        if isolated_vertically:
            return r, colors
    raise ValueError("missing legend row")


def _find_marker_position(
    grid: np.ndarray, canvas_bbox: tuple[int, int, int, int], color: int
) -> tuple[int, int]:
    r0, c0, r1, c1 = canvas_bbox
    interior = grid[r0 + 1 : r1, c0 + 1 : c1]
    coords = np.argwhere(interior == color)
    if coords.shape[0] != 1:
        raise ValueError(f"expected one marker for color {color}")
    return int(coords[0, 0]) + 1, int(coords[0, 1]) + 1


def _find_block_size(
    grid: np.ndarray, canvas_bbox: tuple[int, int, int, int], legend_row: int, color: int
) -> tuple[int, int]:
    mask = grid == color
    r0, c0, r1, c1 = canvas_bbox
    mask[r0 : r1 + 1, c0 : c1 + 1] = False
    mask[legend_row, :] = False
    coords = np.argwhere(mask)
    if coords.size == 0:
        raise ValueError(f"missing source block for color {color}")
    br0, bc0, br1, bc1 = _bbox_from_coords(coords)
    block = grid[br0 : br1 + 1, bc0 : bc1 + 1]
    if not np.all(block == color):
        raise ValueError(f"source block for color {color} is not solid")
    return br1 - br0 + 1, bc1 - bc0 + 1


def main(grid: np.ndarray) -> np.ndarray:
    """
    Stack scattered colored rectangles onto the framed canvas.

    The grey rectangle on the left is the output canvas.  Inside it, one marker pixel
    per color gives the target top-left corner for the matching rectangle found outside
    the canvas.  The bottom legend lists the paint order from back to front, so later
    legend colors overwrite earlier ones in overlaps.
    """
    g = np.asarray(grid, dtype=int)
    canvas_bbox = _find_canvas_bbox(g)
    legend_row, legend_colors = _find_legend_order(g)
    r0, c0, r1, c1 = canvas_bbox

    out = _empty_canvas(r1 - r0 + 1, c1 - c0 + 1)
    out_h, out_w = out.shape

    for color in legend_colors:
        top, left = _find_marker_position(g, canvas_bbox, int(color))
        block_h, block_w = _find_block_size(g, canvas_bbox, legend_row, int(color))

        rr0 = max(1, top)
        cc0 = max(1, left)
        rr1 = min(out_h - 1, top + block_h)
        cc1 = min(out_w - 1, left + block_w)
        if rr0 < rr1 and cc0 < cc1:
            out[rr0:rr1, cc0:cc1] = int(color)

    return out


def _render_expected(
    inner_h: int,
    inner_w: int,
    sizes_by_color: dict[int, tuple[int, int]],
    markers_by_color: dict[int, tuple[int, int]],
    legend_order: list[int],
) -> np.ndarray:
    out = _empty_canvas(inner_h + 2, inner_w + 2)
    for color in legend_order:
        block_h, block_w = sizes_by_color[color]
        top, left = markers_by_color[color]
        out[1 + top : 1 + top + block_h, 1 + left : 1 + left + block_w] = color
    return out


def _place_non_overlapping(
    sizes: list[tuple[int, int]], area_h: int, area_w: int, pad: int, rng: random.Random
) -> list[tuple[int, int]] | None:
    order = sorted(
        range(len(sizes)),
        key=lambda i: sizes[i][0] * sizes[i][1],
        reverse=True,
    )
    for _ in range(250):
        positions: list[tuple[int, int] | None] = [None] * len(sizes)
        boxes: list[tuple[int, int, int, int]] = []
        ok = True
        for idx in order:
            block_h, block_w = sizes[idx]
            if block_h > area_h or block_w > area_w:
                return None
            placed = False
            for _ in range(200):
                r = rng.randrange(area_h - block_h + 1)
                c = rng.randrange(area_w - block_w + 1)
                box = (r, c, r + block_h - 1, c + block_w - 1)
                if any(_boxes_overlap(box, other, pad=pad) for other in boxes):
                    continue
                positions[idx] = (r, c)
                boxes.append(box)
                placed = True
                break
            if not placed:
                ok = False
                break
        if ok:
            return [(int(r), int(c)) for r, c in positions if r is not None and c is not None]
    return None


def _place_with_overlap(
    sizes: list[tuple[int, int]], inner_h: int, inner_w: int, rng: random.Random
) -> list[tuple[int, int]] | None:
    if len(sizes) < 2:
        return None

    i, j = rng.sample(range(len(sizes)), 2)
    hi, wi = sizes[i]
    hj, wj = sizes[j]

    for _ in range(300):
        ri = rng.randrange(inner_h - hi + 1)
        ci = rng.randrange(inner_w - wi + 1)

        r_lo = max(0, ri - hj + 1)
        r_hi = min(inner_h - hj, ri + hi - 1)
        c_lo = max(0, ci - wj + 1)
        c_hi = min(inner_w - wj, ci + wi - 1)
        if r_lo > r_hi or c_lo > c_hi:
            continue

        rj = rng.randint(r_lo, r_hi)
        cj = rng.randint(c_lo, c_hi)
        if (ri, ci) == (rj, cj):
            continue

        box_i = (ri, ci, ri + hi - 1, ci + wi - 1)
        box_j = (rj, cj, rj + hj - 1, cj + wj - 1)
        if not _boxes_overlap(box_i, box_j):
            continue

        i_contains_j = ri <= rj and ci <= cj and ri + hi >= rj + hj and ci + wi >= cj + wj
        j_contains_i = rj <= ri and cj <= ci and rj + hj >= ri + hi and cj + wj >= ci + wi
        if i_contains_j or j_contains_i:
            continue

        positions: list[tuple[int, int] | None] = [None] * len(sizes)
        positions[i] = (ri, ci)
        positions[j] = (rj, cj)

        used = {(ri, ci), (rj, cj)}
        ok = True
        for k, (hk, wk) in enumerate(sizes):
            if positions[k] is not None:
                continue
            placed = False
            for _ in range(150):
                rk = rng.randrange(inner_h - hk + 1)
                ck = rng.randrange(inner_w - wk + 1)
                if (rk, ck) in used:
                    continue
                positions[k] = (rk, ck)
                used.add((rk, ck))
                placed = True
                break
            if not placed:
                ok = False
                break
        if ok:
            return [(int(r), int(c)) for r, c in positions if r is not None and c is not None]
    return None


def _gen_attempt(rng: random.Random) -> np.ndarray | None:
    count = rng.randint(2, 5)
    overlap_mode = rng.random() < 0.5
    colors = rng.sample(COLOR_POOL, count)
    legend_order = rng.sample(colors, len(colors))

    inner_h = rng.randint(8, 13)
    inner_w = rng.randint(9, 14)
    sizes: list[tuple[int, int]] = []
    for _ in range(count):
        block_h = rng.randint(2, min(5, inner_h - 1))
        block_w = rng.randint(2, min(5, inner_w - 1))
        sizes.append((block_h, block_w))

    total_area = sum(h * w for h, w in sizes)
    capacity = inner_h * inner_w
    if overlap_mode and total_area > int(capacity * 0.82):
        return None
    if not overlap_mode and total_area > int(capacity * 0.60):
        return None

    if overlap_mode:
        marker_positions = _place_with_overlap(sizes, inner_h, inner_w, rng)
    else:
        marker_positions = _place_non_overlapping(sizes, inner_h, inner_w, pad=0, rng=rng)
    if marker_positions is None or len(marker_positions) != count:
        return None

    frame_h = inner_h + 2
    frame_w = inner_w + 2
    max_block_h = max(h for h, _ in sizes)
    max_block_w = max(w for _, w in sizes)
    field_h = max(frame_h, max_block_h + 2) + rng.randint(1, 4)
    strip_w = max(10, max_block_w + 2, (sum(w for _, w in sizes) + 2 * count + 1) // 2)
    strip_w += rng.randint(2, 5)

    scattered_positions = _place_non_overlapping(sizes, field_h, strip_w, pad=1, rng=rng)
    if scattered_positions is None or len(scattered_positions) != count:
        return None

    left_margin = rng.randint(1, 2)
    gap = rng.randint(2, 4)
    right_margin = rng.randint(1, 3)
    width = left_margin + frame_w + gap + strip_w + right_margin
    height = field_h + 2

    frame_top = rng.randrange(field_h - frame_h + 1)
    frame_left = left_margin
    strip_left = frame_left + frame_w + gap

    grid = np.full((height, width), BG, dtype=int)
    grid[frame_top, frame_left : frame_left + frame_w] = FRAME
    grid[frame_top + frame_h - 1, frame_left : frame_left + frame_w] = FRAME
    grid[frame_top : frame_top + frame_h, frame_left] = FRAME
    grid[frame_top : frame_top + frame_h, frame_left + frame_w - 1] = FRAME

    sizes_by_color = {color: size for color, size in zip(colors, sizes)}
    markers_by_color = {color: pos for color, pos in zip(colors, marker_positions)}
    for color, (top, left) in markers_by_color.items():
        grid[frame_top + 1 + top, frame_left + 1 + left] = color

    for color, (top, left) in zip(colors, scattered_positions):
        block_h, block_w = sizes_by_color[color]
        grid[top : top + block_h, strip_left + left : strip_left + left + block_w] = color

    legend_start = rng.randrange(width - count + 1)
    for idx, color in enumerate(legend_order):
        grid[height - 1, legend_start + idx] = color

    expected = _render_expected(inner_h, inner_w, sizes_by_color, markers_by_color, legend_order)
    if not np.array_equal(main(grid), expected):
        return None

    return grid


def generate_input() -> np.ndarray:
    rng = random.Random()
    for _ in range(GEN_RETRIES):
        try:
            grid = _gen_attempt(rng)
        except Exception:
            continue
        if grid is not None:
            return grid
    raise RuntimeError("failed to generate a valid input")
