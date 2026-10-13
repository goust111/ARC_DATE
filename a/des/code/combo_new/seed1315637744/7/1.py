from collections import Counter, deque
import random

import numpy as np


BG = 0
FRAME = 5
LEGEND_POOL = (1, 2, 3, 4, 6, 7, 8, 9)
GEN_ATTEMPTS = 300
DIR4 = ((1, 0), (-1, 0), (0, 1), (0, -1))


def _components(mask: np.ndarray) -> list[list[tuple[int, int]]]:
    h, w = mask.shape
    seen = np.zeros_like(mask, dtype=bool)
    comps: list[list[tuple[int, int]]] = []
    for r in range(h):
        for c in range(w):
            if not bool(mask[r, c]) or bool(seen[r, c]):
                continue
            seen[r, c] = True
            q = deque([(r, c)])
            pts: list[tuple[int, int]] = []
            while q:
                cr, cc = q.popleft()
                pts.append((cr, cc))
                for dr, dc in DIR4:
                    nr = cr + dr
                    nc = cc + dc
                    if 0 <= nr < h and 0 <= nc < w and bool(mask[nr, nc]) and not bool(seen[nr, nc]):
                        seen[nr, nc] = True
                        q.append((nr, nc))
            comps.append(pts)
    return comps


def _bbox(points: list[tuple[int, int]]) -> tuple[int, int, int, int]:
    rows = [r for r, _ in points]
    cols = [c for _, c in points]
    return min(rows), min(cols), max(rows), max(cols)


def _is_rectangular_ring(points: list[tuple[int, int]]) -> bool:
    point_set = set(points)
    r0, c0, r1, c1 = _bbox(points)
    h = r1 - r0 + 1
    w = c1 - c0 + 1
    if len(point_set) != 2 * (h + w) - 4:
        return False
    for r in range(r0, r1 + 1):
        if (r, c0) not in point_set or (r, c1) not in point_set:
            return False
    for c in range(c0, c1 + 1):
        if (r0, c) not in point_set or (r1, c) not in point_set:
            return False
    return all(r in (r0, r1) or c in (c0, c1) for r, c in point_set)


def _read_legend(grid: np.ndarray, bg: int) -> tuple[list[int], np.ndarray]:
    h, w = grid.shape
    top_positions = [c for c in range(w) if int(grid[0, c]) != bg]
    left_positions = [r for r in range(h) if int(grid[r, 0]) != bg]

    def _valid(pos: list[int]) -> bool:
        return len(pos) >= 2 and all(pos[i + 1] - pos[i] >= 2 for i in range(len(pos) - 1))

    top_valid = _valid(top_positions)
    left_valid = _valid(left_positions)
    assert top_valid or left_valid, "legend must appear on the top edge or left edge"

    legend_mask = np.zeros_like(grid, dtype=bool)
    if top_valid and (not left_valid or len(top_positions) >= len(left_positions)):
        for c in top_positions:
            legend_mask[0, c] = True
        return [int(grid[0, c]) for c in top_positions], legend_mask

    for r in left_positions:
        legend_mask[r, 0] = True
    return [int(grid[r, 0]) for r in left_positions], legend_mask


def _frame_color(grid: np.ndarray, bg: int, legend_mask: np.ndarray) -> int:
    colors = [int(v) for v in grid[~legend_mask].flat if int(v) != bg]
    assert colors, "grid must contain framed objects"
    return Counter(colors).most_common(1)[0][0]


def main(grid: np.ndarray) -> np.ndarray:
    """
    Fill each framed rectangle according to how many tiny enclosed windows it contains.

    Input structure:
    - The background is black.
    - A legend of isolated single colored pixels sits on the left edge or the top edge, with one black gap between neighboring legend cells.
    - Elsewhere, grey pixels form several closed rectangular outer frames, and each frame may contain several smaller grey 3×3 rings whose centers are 1×1 black window pixels.

    Rule:
    - Read the edge legend in order: top-to-bottom on the left edge or left-to-right on the top edge, so the 1st legend color means 1 window, the 2nd means 2 windows, and so on.
    - For each grey outer frame, count how many 1×1 black window centers are enclosed by small grey rings inside it.
    - Replace every black pixel inside that outer frame, including the window centers, with the legend color indexed by the window count.
    - Keep all grey frame pixels unchanged.

    Key insight:
    The visible count of enclosed 1×1 windows is not itself the answer color; it must first be used as an index into the separate edge legend.
    """
    g = np.asarray(grid, dtype=int)
    bg = int(Counter([int(g[0, 0]), int(g[0, -1]), int(g[-1, 0]), int(g[-1, -1])]).most_common(1)[0][0])

    legend, legend_mask = _read_legend(g, bg)
    frame_color = _frame_color(g, bg, legend_mask)

    comps = _components((g == int(frame_color)) & ~legend_mask)
    outer_boxes: list[tuple[int, int, int, int]] = []
    window_boxes: list[tuple[int, int, int, int]] = []
    for comp in comps:
        assert _is_rectangular_ring(comp), "grey components must be rectangular rings"
        box = _bbox(comp)
        h = box[2] - box[0] + 1
        w = box[3] - box[1] + 1
        if h == 3 and w == 3:
            window_boxes.append(box)
        else:
            assert h >= 5 and w >= 5, "outer frames must be larger than windows"
            outer_boxes.append(box)

    assert outer_boxes, "expected at least one outer frame"
    assert window_boxes, "expected at least one inner window"

    for wr0, wc0, wr1, wc1 in window_boxes:
        owners = [
            box
            for box in outer_boxes
            if box[0] < wr0 and wr1 < box[2] and box[1] < wc0 and wc1 < box[3]
        ]
        assert len(owners) == 1, "each window must belong to exactly one outer frame"

    out = g.copy()
    for r0, c0, r1, c1 in outer_boxes:
        count = sum(
            1
            for wr0, wc0, wr1, wc1 in window_boxes
            if r0 < wr0 and wr1 < r1 and c0 < wc0 and wc1 < c1
        )
        assert 1 <= count <= len(legend), "window count must index the legend"
        fill_color = int(legend[count - 1])
        interior = out[r0 + 1 : r1, c0 + 1 : c1]
        interior[interior == bg] = fill_color

    return out


def _slot_candidates(count: int) -> list[tuple[int, int]]:
    candidates: list[tuple[int, int, int, int]] = []
    for rows in range(1, 4):
        for cols in range(1, 4):
            if rows * cols >= count:
                candidates.append((rows, cols, rows * cols, abs(rows - cols)))
    candidates.sort(key=lambda item: (item[2], item[3], item[0] + item[1]))
    best_area = candidates[0][2]
    return [(r, c) for r, c, area, _ in candidates if area <= best_area + 1]


def _make_rectangle(window_count: int, rng: random.Random) -> np.ndarray:
    slot_rows, slot_cols = rng.choice(_slot_candidates(window_count))
    pad_top = rng.randint(1, 2)
    pad_bottom = rng.randint(1, 2)
    pad_left = rng.randint(1, 2)
    pad_right = rng.randint(1, 2)

    inner_h = pad_top + pad_bottom + 3 + 4 * (slot_rows - 1)
    inner_w = pad_left + pad_right + 3 + 4 * (slot_cols - 1)
    patch = np.full((inner_h + 2, inner_w + 2), BG, dtype=int)
    patch[0, :] = FRAME
    patch[-1, :] = FRAME
    patch[:, 0] = FRAME
    patch[:, -1] = FRAME

    slots = [(sr, sc) for sr in range(slot_rows) for sc in range(slot_cols)]
    for sr, sc in rng.sample(slots, window_count):
        r0 = 1 + pad_top + 4 * sr
        c0 = 1 + pad_left + 4 * sc
        patch[r0 : r0 + 3, c0 : c0 + 3] = FRAME
        patch[r0 + 1, c0 + 1] = BG

    return patch


def _choose_counts(legend_len: int, rect_count: int, rng: random.Random) -> list[int]:
    base = list(range(1, legend_len + 1))
    rng.shuffle(base)
    counts = base[: min(rect_count, legend_len)]
    while len(counts) < rect_count:
        counts.append(rng.randint(1, legend_len))
    rng.shuffle(counts)
    return counts


def _layout_rectangles(
    rects: list[np.ndarray], orientation: str, legend_len: int, rng: random.Random
) -> tuple[np.ndarray, list[np.ndarray]] | None:
    shuffled = rects[:]
    rng.shuffle(shuffled)

    col_options = list(range(1, min(3, len(shuffled)) + 1))
    rng.shuffle(col_options)
    for cols in col_options:
        rows = (len(shuffled) + cols - 1) // cols
        row_heights = [0] * rows
        col_widths = [0] * cols
        for idx, rect in enumerate(shuffled):
            rr = idx // cols
            cc = idx % cols
            h, w = rect.shape
            row_heights[rr] = max(row_heights[rr], int(h))
            col_widths[cc] = max(col_widths[cc], int(w))

        row_gaps = [rng.randint(1, 3) for _ in range(rows - 1)]
        col_gaps = [rng.randint(1, 3) for _ in range(cols - 1)]
        top_margin = rng.randint(1, 2) + (2 if orientation == "top" else 0)
        left_margin = rng.randint(1, 2) + (2 if orientation == "left" else 0)
        bottom_margin = rng.randint(1, 2)
        right_margin = rng.randint(1, 2)

        total_h = top_margin + sum(row_heights) + sum(row_gaps) + bottom_margin
        total_w = left_margin + sum(col_widths) + sum(col_gaps) + right_margin
        legend_span = 2 * legend_len - 1
        if orientation == "left":
            total_h = max(total_h, legend_span + 2)
        else:
            total_w = max(total_w, legend_span + 2)
        if total_h > 30 or total_w > 30:
            continue

        grid = np.full((total_h, total_w), BG, dtype=int)
        row_starts: list[int] = []
        col_starts: list[int] = []

        cur = top_margin
        for rr in range(rows):
            row_starts.append(cur)
            cur += row_heights[rr]
            if rr < rows - 1:
                cur += row_gaps[rr]

        cur = left_margin
        for cc in range(cols):
            col_starts.append(cur)
            cur += col_widths[cc]
            if cc < cols - 1:
                cur += col_gaps[cc]

        for idx, rect in enumerate(shuffled):
            rr = idx // cols
            cc = idx % cols
            cell_h = row_heights[rr]
            cell_w = col_widths[cc]
            h, w = rect.shape
            r0 = row_starts[rr] + rng.randint(0, cell_h - int(h))
            c0 = col_starts[cc] + rng.randint(0, cell_w - int(w))
            grid[r0 : r0 + h, c0 : c0 + w] = rect

        return grid, shuffled

    return None


def _place_legend(grid: np.ndarray, colors: list[int], orientation: str, rng: random.Random) -> None:
    span = 2 * len(colors) - 1
    if orientation == "left":
        start = rng.randint(1, grid.shape[0] - span)
        for idx, color in enumerate(colors):
            grid[start + 2 * idx, 0] = int(color)
    else:
        start = rng.randint(1, grid.shape[1] - span)
        for idx, color in enumerate(colors):
            grid[0, start + 2 * idx] = int(color)


def _gen_attempt(rng: random.Random) -> np.ndarray | None:
    orientation = rng.choice(["left", "top"])
    legend_len = rng.randint(2, 4)
    rect_count = rng.randint(1, min(4, legend_len + 1))
    counts = _choose_counts(legend_len, rect_count, rng)
    legend_colors = list(rng.sample(LEGEND_POOL, legend_len))
    rects = [_make_rectangle(count, rng) for count in counts]

    laid_out = _layout_rectangles(rects, orientation, legend_len, rng)
    if laid_out is None:
        return None
    grid, _ = laid_out
    _place_legend(grid, legend_colors, orientation, rng)

    try:
        out = main(grid)
    except AssertionError:
        return None

    if not bool((out != grid).any()):
        return None
    return grid


def generate_input() -> np.ndarray:
    rng = random.Random()
    for _ in range(GEN_ATTEMPTS):
        grid = _gen_attempt(rng)
        if grid is not None:
            return grid
    raise RuntimeError("failed to generate a valid input grid")
