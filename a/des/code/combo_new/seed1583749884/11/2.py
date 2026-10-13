from __future__ import annotations

from collections import defaultdict

import numpy as np


BG = 8
FRAME = 9
LINE = 5
STRIPE = 4
DIR4 = ((1, 0), (-1, 0), (0, 1), (0, -1))
MAX_GEN_ATTEMPTS = 400


def _cc4(mask: np.ndarray) -> list[list[tuple[int, int]]]:
    h, w = mask.shape
    seen = np.zeros_like(mask, dtype=bool)
    comps: list[list[tuple[int, int]]] = []
    for r, c in np.argwhere(mask):
        r = int(r)
        c = int(c)
        if seen[r, c]:
            continue
        stack = [(r, c)]
        seen[r, c] = True
        comp: list[tuple[int, int]] = []
        while stack:
            rr, cc = stack.pop()
            comp.append((rr, cc))
            for dr, dc in DIR4:
                nr = rr + dr
                nc = cc + dc
                if 0 <= nr < h and 0 <= nc < w and mask[nr, nc] and not seen[nr, nc]:
                    seen[nr, nc] = True
                    stack.append((nr, nc))
        comps.append(comp)
    return comps


def _bbox(points: list[tuple[int, int]]) -> tuple[int, int, int, int]:
    rows = [r for r, _ in points]
    cols = [c for _, c in points]
    return min(rows), min(cols), max(rows), max(cols)


def _perimeter_set(r0: int, c0: int, r1: int, c1: int) -> set[tuple[int, int]]:
    pts = set()
    for c in range(c0, c1 + 1):
        pts.add((r0, c))
        pts.add((r1, c))
    for r in range(r0, r1 + 1):
        pts.add((r, c0))
        pts.add((r, c1))
    return pts


def _find_frames(grid: np.ndarray) -> tuple[list[dict[str, object]], np.ndarray]:
    candidates: list[dict[str, object]] = []
    for comp in _cc4(grid == FRAME):
        r0, c0, r1, c1 = _bbox(comp)
        if r1 - r0 + 1 < 3 or c1 - c0 + 1 < 3:
            continue
        perimeter = _perimeter_set(r0, c0, r1, c1)
        comp_set = set(comp)
        if comp_set != perimeter:
            continue
        candidates.append(
            {
                "bbox": (r0, c0, r1, c1),
                "points": comp,
            }
        )

    candidates.sort(key=lambda item: item["bbox"])
    labels = np.full(grid.shape, -1, dtype=int)
    frames: list[dict[str, object]] = []
    for idx, item in enumerate(candidates):
        r0, c0, r1, c1 = item["bbox"]
        h = r1 - r0 + 1
        w = c1 - c0 + 1
        frame = {
            "bbox": (r0, c0, r1, c1),
            "height": h,
            "width": w,
        }
        frames.append(frame)
        for r, c in item["points"]:
            labels[r, c] = idx
    return frames, labels


def _find_incoming_lengths(grid: np.ndarray, frame_labels: np.ndarray) -> dict[int, int]:
    h, w = grid.shape
    incoming: dict[int, int] = defaultdict(int)

    for comp in _cc4(grid == LINE):
        if not comp:
            continue

        if len(comp) == 1:
            r, c = comp[0]
            left_id = int(frame_labels[r, c - 1]) if c - 1 >= 0 else -1
            right_id = int(frame_labels[r, c + 1]) if c + 1 < w else -1
            top_id = int(frame_labels[r - 1, c]) if r - 1 >= 0 else -1
            bottom_id = int(frame_labels[r + 1, c]) if r + 1 < h else -1
            horizontal = left_id >= 0 and right_id >= 0 and left_id != right_id
            vertical = top_id >= 0 and bottom_id >= 0 and top_id != bottom_id
            if horizontal and not vertical:
                incoming[right_id] = max(incoming[right_id], 1)
            elif vertical and not horizontal:
                incoming[bottom_id] = max(incoming[bottom_id], 1)
            continue

        rows = sorted({r for r, _ in comp})
        cols = sorted({c for _, c in comp})
        if len(rows) == 1:
            row = rows[0]
            seg_cols = sorted(c for _, c in comp)
            if seg_cols != list(range(seg_cols[0], seg_cols[-1] + 1)):
                continue
            left_id = int(frame_labels[row, seg_cols[0] - 1]) if seg_cols[0] - 1 >= 0 else -1
            right_id = int(frame_labels[row, seg_cols[-1] + 1]) if seg_cols[-1] + 1 < w else -1
            if left_id >= 0 and right_id >= 0 and left_id != right_id:
                incoming[right_id] = max(incoming[right_id], len(seg_cols))
            continue

        if len(cols) == 1:
            col = cols[0]
            seg_rows = sorted(r for r, _ in comp)
            if seg_rows != list(range(seg_rows[0], seg_rows[-1] + 1)):
                continue
            top_id = int(frame_labels[seg_rows[0] - 1, col]) if seg_rows[0] - 1 >= 0 else -1
            bottom_id = int(frame_labels[seg_rows[-1] + 1, col]) if seg_rows[-1] + 1 < h else -1
            if top_id >= 0 and bottom_id >= 0 and top_id != bottom_id:
                incoming[bottom_id] = max(incoming[bottom_id], len(seg_rows))

    return dict(incoming)


def main(grid: np.ndarray) -> np.ndarray:
    """
    Encode each connector length as the number of horizontal stripes inside the box it points into.

    Input structure:
    - The background is Azure.
    - Several one-pixel-thick hollow Maroon rectangular frames appear in the grid.
    - Some pairs of frames are joined by one-pixel-wide Grey horizontal or vertical straight connectors.
    - Each connector touches one frame at its left/top end and another frame at its right/bottom end.

    Rule:
    - Keep the background, all frames, and all connectors unchanged.
    - For every Grey connector, measure its pixel length L.
    - Treat the frame at the connector's right end (for horizontal lines) or bottom end
      (for vertical lines) as the target frame.
    - Inside each target frame, draw L Yellow horizontal stripes starting from the top
      interior row, leaving one background row between consecutive stripes.
    - If multiple connectors point into the same frame, use only the longest connector.

    Key insight:
    The connector is not merely linking two boxes; its visible length is the code that
    determines how many evenly spaced horizontal stripes must appear inside the destination box.
    """
    g = np.asarray(grid, dtype=int)
    out = g.copy()
    frames, frame_labels = _find_frames(g)
    if not frames:
        return out

    incoming = _find_incoming_lengths(g, frame_labels)
    for frame_id, length in incoming.items():
        r0, c0, r1, c1 = frames[frame_id]["bbox"]
        for k in range(length):
            rr = r0 + 1 + 2 * k
            if rr > r1 - 1:
                break
            out[rr, c0 + 1 : c1] = STRIPE
    return out


def _draw_frame(grid: np.ndarray, r0: int, c0: int, h: int, w: int) -> None:
    grid[r0, c0 : c0 + w] = FRAME
    grid[r0 + h - 1, c0 : c0 + w] = FRAME
    grid[r0 : r0 + h, c0] = FRAME
    grid[r0 : r0 + h, c0 + w - 1] = FRAME


def _height_choices(n_rows: int) -> tuple[int, ...]:
    if n_rows == 1:
        return (5, 7, 9, 11)
    if n_rows == 2:
        return (5, 7, 9, 11)
    if n_rows == 3:
        return (5, 7, 9)
    return (5, 7)


def _width_choices(n_cols: int) -> tuple[int, ...]:
    if n_cols <= 2:
        return (4, 5, 6, 7, 8)
    if n_cols == 3:
        return (4, 5, 6)
    return (4, 5)


def _choose_mode(rng: np.random.Generator) -> tuple[int, int]:
    mode = int(rng.integers(0, 3))
    if mode == 0:
        return 1, int(rng.integers(2, 5))
    if mode == 1:
        return int(rng.integers(2, 5)), 1
    shapes = ((2, 2), (2, 3), (3, 2))
    return shapes[int(rng.integers(0, len(shapes)))]


def _expected_output(
    grid: np.ndarray,
    boxes: dict[tuple[int, int], tuple[int, int, int, int]],
    incoming: dict[tuple[int, int], int],
) -> np.ndarray:
    out = grid.copy()
    for key, length in incoming.items():
        r0, c0, h, w = boxes[key]
        for k in range(length):
            rr = r0 + 1 + 2 * k
            if rr > r0 + h - 2:
                break
            out[rr, c0 + 1 : c0 + w - 1] = STRIPE
    return out


def _gen_attempt(rng: np.random.Generator) -> np.ndarray | None:
    n_rows, n_cols = _choose_mode(rng)
    row_heights = [int(rng.choice(_height_choices(n_rows))) for _ in range(n_rows)]
    col_widths = [int(rng.choice(_width_choices(n_cols))) for _ in range(n_cols)]
    row_caps = [(h - 1) // 2 for h in row_heights]

    if n_cols > 1:
        horiz_cap = row_caps[0] if n_rows == 1 else min(row_caps)
        if horiz_cap < 1:
            return None
        gaps_x = [int(rng.integers(1, min(5, horiz_cap) + 1)) for _ in range(n_cols - 1)]
    else:
        gaps_x = []

    if n_rows > 1:
        gaps_y = [int(rng.integers(1, min(5, row_caps[i + 1]) + 1)) for i in range(n_rows - 1)]
    else:
        gaps_y = []

    body_h = sum(row_heights) + sum(gaps_y)
    body_w = sum(col_widths) + sum(gaps_x)
    if body_h > 28 or body_w > 28:
        return None

    free_h = 30 - body_h
    free_w = 30 - body_w
    if free_h < 2 or free_w < 2:
        return None

    extra_h = int(rng.integers(2, free_h + 1))
    extra_w = int(rng.integers(2, free_w + 1))
    top = int(rng.integers(1, extra_h))
    left = int(rng.integers(1, extra_w))
    bottom = extra_h - top
    right = extra_w - left
    if bottom < 1 or right < 1:
        return None

    h_edges = np.zeros((n_rows, max(n_cols - 1, 0)), dtype=bool)
    v_edges = np.zeros((max(n_rows - 1, 0), n_cols), dtype=bool)

    if n_cols > 1:
        prob = 0.7 if n_rows == 1 else 0.45
        h_edges = rng.random(h_edges.shape) < prob
        if not np.any(h_edges):
            rr = int(rng.integers(0, n_rows))
            cc = int(rng.integers(0, n_cols - 1))
            h_edges[rr, cc] = True

    if n_rows > 1:
        prob = 0.7 if n_cols == 1 else 0.45
        v_edges = rng.random(v_edges.shape) < prob
        if not np.any(v_edges):
            rr = int(rng.integers(0, n_rows - 1))
            cc = int(rng.integers(0, n_cols))
            v_edges[rr, cc] = True

    if n_rows > 1 and n_cols > 1:
        target_r = int(rng.integers(1, n_rows))
        target_c = int(rng.integers(1, n_cols))
        h_edges[target_r, target_c - 1] = True
        v_edges[target_r - 1, target_c] = True

    row_tops = [top]
    for i in range(1, n_rows):
        row_tops.append(row_tops[-1] + row_heights[i - 1] + gaps_y[i - 1])
    col_lefts = [left]
    for j in range(1, n_cols):
        col_lefts.append(col_lefts[-1] + col_widths[j - 1] + gaps_x[j - 1])

    grid = np.full((top + body_h + bottom, left + body_w + right), BG, dtype=int)
    boxes: dict[tuple[int, int], tuple[int, int, int, int]] = {}

    for i in range(n_rows):
        for j in range(n_cols):
            r0 = row_tops[i]
            c0 = col_lefts[j]
            h = row_heights[i]
            w = col_widths[j]
            boxes[(i, j)] = (r0, c0, h, w)
            _draw_frame(grid, r0, c0, h, w)

    incoming: dict[tuple[int, int], int] = {}

    for i in range(n_rows):
        for j in range(n_cols - 1):
            if not h_edges[i, j]:
                continue
            r0, c0, h, w = boxes[(i, j)]
            next_r0, next_c0, _nh, _nw = boxes[(i, j + 1)]
            row = r0 + h // 2
            c_start = c0 + w
            c_end = next_c0 - 1
            if c_start > c_end:
                return None
            if np.any(grid[row, c_start : c_end + 1] != BG):
                return None
            grid[row, c_start : c_end + 1] = LINE
            incoming[(i, j + 1)] = max(incoming.get((i, j + 1), 0), c_end - c_start + 1)

    for i in range(n_rows - 1):
        for j in range(n_cols):
            if not v_edges[i, j]:
                continue
            r0, c0, h, w = boxes[(i, j)]
            next_r0, next_c0, _nh, _nw = boxes[(i + 1, j)]
            col = c0 + w // 2
            r_start = r0 + h
            r_end = next_r0 - 1
            if r_start > r_end:
                return None
            if np.any(grid[r_start : r_end + 1, col] != BG):
                return None
            grid[r_start : r_end + 1, col] = LINE
            incoming[(i + 1, j)] = max(incoming.get((i + 1, j), 0), r_end - r_start + 1)

    if not incoming:
        return None

    expected = _expected_output(grid, boxes, incoming)
    if not np.array_equal(main(grid), expected):
        return None
    return grid


def generate_input() -> np.ndarray:
    rng = np.random.default_rng()
    for _ in range(MAX_GEN_ATTEMPTS):
        grid = _gen_attempt(rng)
        if grid is not None:
            return grid
    raise RuntimeError("failed to generate a valid ARC sample")
