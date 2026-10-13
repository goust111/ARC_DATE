import random
from collections import Counter, deque

import numpy as np


BG = 0
FRAME_COLOR = 5
MARKER_COLORS = [1, 2, 3, 4, 6, 7, 8, 9]
SIDES = ("top", "bottom", "left", "right")

MIN_GRID_SIZE = 14
MAX_GRID_SIZE = 30
FRAME_COUNT_RANGE = (1, 4)
FRAME_OUTER_H_RANGE = (5, 9)
FRAME_OUTER_W_RANGE = (5, 9)
PLACEMENT_RETRIES = 120
GENERATION_RETRIES = 400


def _corner_background(grid: np.ndarray) -> int:
    corners = [
        int(grid[0, 0]),
        int(grid[0, -1]),
        int(grid[-1, 0]),
        int(grid[-1, -1]),
    ]
    return Counter(corners).most_common(1)[0][0]


def _components(mask: np.ndarray) -> list[list[tuple[int, int]]]:
    h, w = mask.shape
    seen = np.zeros((h, w), dtype=bool)
    comps: list[list[tuple[int, int]]] = []
    for r in range(h):
        for c in range(w):
            if not bool(mask[r, c]) or bool(seen[r, c]):
                continue
            q = deque([(r, c)])
            seen[r, c] = True
            comp: list[tuple[int, int]] = []
            while q:
                cr, cc = q.popleft()
                comp.append((int(cr), int(cc)))
                for dr, dc in ((1, 0), (-1, 0), (0, 1), (0, -1)):
                    nr = cr + dr
                    nc = cc + dc
                    if 0 <= nr < h and 0 <= nc < w and bool(mask[nr, nc]) and not bool(seen[nr, nc]):
                        seen[nr, nc] = True
                        q.append((int(nr), int(nc)))
            comps.append(comp)
    return comps


def _border_positions(r0: int, c0: int, r1: int, c1: int) -> set[tuple[int, int]]:
    border = set()
    for c in range(c0, c1 + 1):
        border.add((r0, c))
        border.add((r1, c))
    for r in range(r0, r1 + 1):
        border.add((r, c0))
        border.add((r, c1))
    return border


def _frame_info(
    grid: np.ndarray, color: int, comp: list[tuple[int, int]], bg: int
) -> dict[str, object] | None:
    rows = [r for r, _ in comp]
    cols = [c for _, c in comp]
    r0, r1 = min(rows), max(rows)
    c0, c1 = min(cols), max(cols)
    if r1 - r0 < 2 or c1 - c0 < 2:
        return None

    border = _border_positions(r0, c0, r1, c1)
    comp_set = set(comp)
    if not comp_set <= border:
        return None

    missing = [pos for pos in border if pos not in comp_set]
    if len(missing) != 1:
        return None

    gap_r, gap_c = missing[0]
    side_candidates = []
    if gap_r == r0 and c0 < gap_c < c1:
        side_candidates.append("top")
    if gap_r == r1 and c0 < gap_c < c1:
        side_candidates.append("bottom")
    if gap_c == c0 and r0 < gap_r < r1:
        side_candidates.append("left")
    if gap_c == c1 and r0 < gap_r < r1:
        side_candidates.append("right")
    if len(side_candidates) != 1:
        return None

    if int(grid[gap_r, gap_c]) != bg:
        return None

    interior = grid[r0 + 1 : r1, c0 + 1 : c1]
    marker_mask = (interior != bg) & (interior != color)
    marker_vals = sorted(set(map(int, interior[marker_mask].tolist())))
    if len(marker_vals) != 1:
        return None

    return {
        "color": int(color),
        "bbox": (int(r0), int(c0), int(r1), int(c1)),
        "side": side_candidates[0],
        "gap": (int(gap_r), int(gap_c)),
        "marker_color": int(marker_vals[0]),
        "count": int(marker_mask.sum()),
    }


def _find_frames(grid: np.ndarray, bg: int) -> list[dict[str, object]]:
    frames: list[dict[str, object]] = []
    for color in sorted(set(map(int, np.unique(grid))) - {bg}):
        for comp in _components(grid == color):
            info = _frame_info(grid, color, comp, bg)
            if info is not None:
                frames.append(info)
    frames.sort(key=lambda item: tuple(item["bbox"]))  # deterministic update order
    return frames


def _paint_bar(
    out: np.ndarray,
    bbox: tuple[int, int, int, int],
    side: str,
    gap: tuple[int, int],
    color: int,
    length: int,
) -> None:
    r0, c0, r1, c1 = bbox
    gap_r, gap_c = gap
    if side == "top":
        steps = min(length, r1 - r0 - 1)
        for k in range(steps):
            out[r1 - 1 - k, gap_c] = color
    elif side == "bottom":
        steps = min(length, r1 - r0 - 1)
        for k in range(steps):
            out[r0 + 1 + k, gap_c] = color
    elif side == "left":
        steps = min(length, c1 - c0 - 1)
        for k in range(steps):
            out[gap_r, c1 - 1 - k] = color
    else:
        steps = min(length, c1 - c0 - 1)
        for k in range(steps):
            out[gap_r, c0 + 1 + k] = color


def main(grid: np.ndarray) -> np.ndarray:
    """
    Grow a colored tally bar inside each gapped rectangular frame from the side opposite the gap.

    Input structure:
    - The background is black.
    - Several grey rectangular frames appear on the grid, and each frame has exactly one single-cell gap on one edge.
    - Each frame contains scattered marker pixels of one non-black, non-grey color, and different frames use different marker colors.
    - Outside the frames there are no foreground pixels.

    Rule:
    - Detect each rectangular frame, its gap side, and the row or column aligned with the gap cell.
    - Count the marker pixels inside that frame and erase the original scattered markers back to black.
    - Starting from the edge opposite the gap, draw a one-pixel-wide bar of the marker color toward the gap, aligned with the gap row or column.
    - The bar length equals the number of original markers, while the grey frame and the black gap stay unchanged.

    Key insight:
    - The marker positions are only a counter; the gap simultaneously determines the growth direction, the aligned row or column, and the opposite-edge starting side.
    """
    g = np.asarray(grid, dtype=int)
    out = g.copy()
    bg = _corner_background(g)

    for frame in _find_frames(g, bg):
        r0, c0, r1, c1 = frame["bbox"]
        frame_color = int(frame["color"])
        marker_color = int(frame["marker_color"])
        count = int(frame["count"])
        inner_out = out[r0 + 1 : r1, c0 + 1 : c1]
        marker_mask = (inner_out != bg) & (inner_out != frame_color)
        inner_out[marker_mask] = bg
        _paint_bar(
            out,
            (r0, c0, r1, c1),
            str(frame["side"]),
            tuple(frame["gap"]),
            marker_color,
            count,
        )

    return out


def _target_cells(
    bbox: tuple[int, int, int, int], side: str, gap: tuple[int, int]
) -> list[tuple[int, int]]:
    r0, c0, r1, c1 = bbox
    gap_r, gap_c = gap
    if side == "top":
        return [(r, gap_c) for r in range(r1 - 1, r0, -1)]
    if side == "bottom":
        return [(r, gap_c) for r in range(r0 + 1, r1)]
    if side == "left":
        return [(gap_r, c) for c in range(c1 - 1, c0, -1)]
    return [(gap_r, c) for c in range(c0 + 1, c1)]


def _sample_marker_positions(
    rng: random.Random,
    bbox: tuple[int, int, int, int],
    side: str,
    gap: tuple[int, int],
    count: int,
) -> list[tuple[int, int]]:
    r0, c0, r1, c1 = bbox
    interior = [(r, c) for r in range(r0 + 1, r1) for c in range(c0 + 1, c1)]
    target = _target_cells(bbox, side, gap)
    target_set = set(target)

    for _ in range(80):
        picks = rng.sample(interior, count)
        if count > 1 and all(pos in target_set for pos in picks):
            continue
        if set(picks) == set(target[:count]):
            continue
        return picks
    return rng.sample(interior, count)


def _draw_frame(
    grid: np.ndarray,
    bbox: tuple[int, int, int, int],
    side: str,
    gap: tuple[int, int],
    marker_color: int,
    count: int,
    rng: random.Random,
) -> None:
    r0, c0, r1, c1 = bbox
    grid[r0, c0 : c1 + 1] = FRAME_COLOR
    grid[r1, c0 : c1 + 1] = FRAME_COLOR
    grid[r0 : r1 + 1, c0] = FRAME_COLOR
    grid[r0 : r1 + 1, c1] = FRAME_COLOR
    grid[gap] = BG

    for r, c in _sample_marker_positions(rng, bbox, side, gap, count):
        grid[r, c] = marker_color


def _can_place(used: np.ndarray, bbox: tuple[int, int, int, int]) -> bool:
    r0, c0, r1, c1 = bbox
    h, w = used.shape
    rr0 = max(0, r0 - 1)
    cc0 = max(0, c0 - 1)
    rr1 = min(h, r1 + 2)
    cc1 = min(w, c1 + 2)
    return not bool(used[rr0:rr1, cc0:cc1].any())


def _reserve(used: np.ndarray, bbox: tuple[int, int, int, int]) -> None:
    r0, c0, r1, c1 = bbox
    h, w = used.shape
    rr0 = max(0, r0 - 1)
    cc0 = max(0, c0 - 1)
    rr1 = min(h, r1 + 2)
    cc1 = min(w, c1 + 2)
    used[rr0:rr1, cc0:cc1] = True


def _gen_attempt(rng: random.Random) -> np.ndarray | None:
    h = rng.randint(MIN_GRID_SIZE, MAX_GRID_SIZE)
    w = rng.randint(MIN_GRID_SIZE, MAX_GRID_SIZE)
    grid = np.full((h, w), BG, dtype=int)
    used = np.zeros((h, w), dtype=bool)

    frame_count = rng.randint(FRAME_COUNT_RANGE[0], FRAME_COUNT_RANGE[1])
    marker_colors = rng.sample(MARKER_COLORS, frame_count)
    sides = rng.sample(SIDES, frame_count)

    for idx in range(frame_count):
        side = sides[idx]
        marker_color = marker_colors[idx]
        placed = False

        for _ in range(PLACEMENT_RETRIES):
            outer_h = rng.randint(FRAME_OUTER_H_RANGE[0], FRAME_OUTER_H_RANGE[1])
            outer_w = rng.randint(FRAME_OUTER_W_RANGE[0], FRAME_OUTER_W_RANGE[1])
            if outer_h >= h - 1 or outer_w >= w - 1:
                continue

            r0 = rng.randint(1, h - outer_h - 1)
            c0 = rng.randint(1, w - outer_w - 1)
            bbox = (r0, c0, r0 + outer_h - 1, c0 + outer_w - 1)
            if not _can_place(used, bbox):
                continue

            r1, c1 = bbox[2], bbox[3]
            if side in ("top", "bottom"):
                gap = (r0 if side == "top" else r1, rng.randint(c0 + 1, c1 - 1))
                max_count = outer_h - 2
            else:
                gap = (rng.randint(r0 + 1, r1 - 1), c0 if side == "left" else c1)
                max_count = outer_w - 2

            if max_count <= 0:
                continue

            count = rng.randint(1, max_count)
            _draw_frame(grid, bbox, side, gap, marker_color, count, rng)
            _reserve(used, bbox)
            placed = True
            break

        if not placed:
            return None

    # Reject accidental ambiguities or degenerate samples.
    if len(_find_frames(grid, BG)) != frame_count:
        return None
    if np.array_equal(main(grid), grid):
        return None
    return grid


def generate_input() -> np.ndarray:
    rng = random.Random()
    for _ in range(GENERATION_RETRIES):
        grid = _gen_attempt(rng)
        if grid is not None:
            return grid
    raise RuntimeError("generate_input: exhausted retries")
