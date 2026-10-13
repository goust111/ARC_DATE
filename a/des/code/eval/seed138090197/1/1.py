from __future__ import annotations

import random
from collections import Counter

import numpy as np


BG_COLOR = 0
FRAME_COLOR = 5
MARKER_COLORS = (1, 2, 3, 4, 6, 7, 8, 9)

GRID_H_RANGE = (16, 26)
GRID_W_RANGE = (16, 26)
RECT_COUNT_RANGE = (1, 4)
OUTER_H_RANGE = (4, 8)
OUTER_W_RANGE = (4, 8)
DISTRACTOR_COUNT_RANGE = (1, 2)
GEN_ATTEMPTS = 400
PLACEMENT_ATTEMPTS = 250

DIR4 = [(-1, 0), (1, 0), (0, -1), (0, 1)]
SIDE_TO_DIR = {
    "top": (-1, 0),
    "bottom": (1, 0),
    "left": (0, -1),
    "right": (0, 1),
}


def get_bg(grid: np.ndarray) -> int:
    counts = Counter(map(int, np.asarray(grid).ravel()))
    return int(counts.most_common(1)[0][0])


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
                for dr, dc in DIR4:
                    nr, nc = cr + dr, cc + dc
                    if 0 <= nr < h and 0 <= nc < w and mask[nr, nc] and not seen[nr, nc]:
                        seen[nr, nc] = True
                        stack.append((nr, nc))
            comps.append(comp)
    return comps


def _rectangle_border(r0: int, c0: int, r1: int, c1: int) -> list[tuple[int, int]]:
    border: list[tuple[int, int]] = []
    for c in range(c0, c1 + 1):
        border.append((r0, c))
    for r in range(r0 + 1, r1):
        border.append((r, c1))
    for c in range(c1, c0 - 1, -1):
        border.append((r1, c))
    for r in range(r1 - 1, r0, -1):
        border.append((r, c0))
    return border


def _gap_direction(
    gap: tuple[int, int], bbox: tuple[int, int, int, int]
) -> tuple[int, int] | None:
    r0, c0, r1, c1 = bbox
    gr, gc = gap
    if gr == r0 and c0 < gc < c1:
        return SIDE_TO_DIR["top"]
    if gr == r1 and c0 < gc < c1:
        return SIDE_TO_DIR["bottom"]
    if gc == c0 and r0 < gr < r1:
        return SIDE_TO_DIR["left"]
    if gc == c1 and r0 < gr < r1:
        return SIDE_TO_DIR["right"]
    return None


def _recognize_frame(
    grid: np.ndarray, comp: list[tuple[int, int]], color: int, bg: int
) -> dict[str, object] | None:
    rows = [r for r, _ in comp]
    cols = [c for _, c in comp]
    r0, r1 = min(rows), max(rows)
    c0, c1 = min(cols), max(cols)
    if r1 - r0 < 2 or c1 - c0 < 2:
        return None

    border = _rectangle_border(r0, c0, r1, c1)
    border_set = set(border)
    comp_set = set(comp)
    if len(comp_set) != len(border) - 1:
        return None
    if not comp_set <= border_set:
        return None

    missing = [cell for cell in border if cell not in comp_set]
    if len(missing) != 1:
        return None
    gap = missing[0]
    direction = _gap_direction(gap, (r0, c0, r1, c1))
    if direction is None or int(grid[gap]) != bg:
        return None

    inner = grid[r0 + 1 : r1, c0 + 1 : c1]
    if inner.size == 0 or np.any(inner == color):
        return None
    marker_mask = (inner != bg) & (inner != color)
    if int(marker_mask.sum()) != 1:
        return None
    if int(np.count_nonzero(inner != bg)) != 1:
        return None

    marker_rel = np.argwhere(marker_mask)[0]
    marker_r = int(r0 + 1 + marker_rel[0])
    marker_c = int(c0 + 1 + marker_rel[1])
    marker_color = int(grid[marker_r, marker_c])
    ray_length = int(np.count_nonzero(inner == bg))

    return {
        "bbox": (r0, c0, r1, c1),
        "gap": gap,
        "direction": direction,
        "marker": (marker_r, marker_c),
        "marker_color": marker_color,
        "ray_length": ray_length,
        "frame_color": color,
    }


def _find_frames(grid: np.ndarray) -> list[dict[str, object]]:
    g = np.asarray(grid, dtype=int)
    bg = get_bg(g)
    frames: list[dict[str, object]] = []
    for color in map(int, np.unique(g)):
        if color == bg:
            continue
        for comp in _components(g == color):
            frame = _recognize_frame(g, comp, color, bg)
            if frame is not None:
                frames.append(frame)
    frames.sort(key=lambda item: (item["bbox"], item["gap"]))  # deterministic ray order
    return frames


def main(grid: np.ndarray) -> np.ndarray:
    """
    Emit a ray through each rectangular frame's gap.

    The ray direction is determined by the side containing the unique border gap.
    The ray color comes from the unique interior marker.
    The ray length equals the number of background cells inside the frame.
    """
    g = np.asarray(grid, dtype=int)
    bg = get_bg(g)
    out = g.copy()

    for frame in _find_frames(g):
        gap_r, gap_c = frame["gap"]
        dr, dc = frame["direction"]
        color = int(frame["marker_color"])
        ray_length = int(frame["ray_length"])
        for step in range(ray_length):
            rr = gap_r + dr * step
            cc = gap_c + dc * step
            if not (0 <= rr < g.shape[0] and 0 <= cc < g.shape[1]):
                break
            if g[rr, cc] == bg and out[rr, cc] == bg:
                out[rr, cc] = color

    return out


def _mark_blocked_box(
    blocked: np.ndarray, r0: int, c0: int, r1: int, c1: int, pad: int = 0
) -> None:
    h, w = blocked.shape
    rr0 = max(0, r0 - pad)
    cc0 = max(0, c0 - pad)
    rr1 = min(h - 1, r1 + pad)
    cc1 = min(w - 1, c1 + pad)
    blocked[rr0 : rr1 + 1, cc0 : cc1 + 1] = True


def _ray_cells(
    gap: tuple[int, int], direction: tuple[int, int], length: int, h: int, w: int
) -> list[tuple[int, int]]:
    cells: list[tuple[int, int]] = []
    gr, gc = gap
    dr, dc = direction
    for step in range(length):
        rr = gr + dr * step
        cc = gc + dc * step
        if not (0 <= rr < h and 0 <= cc < w):
            break
        cells.append((rr, cc))
    return cells


def _sample_gap(
    rng: random.Random, side: str, r0: int, c0: int, r1: int, c1: int
) -> tuple[int, int]:
    if side == "top":
        return (r0, rng.randint(c0 + 1, c1 - 1))
    if side == "bottom":
        return (r1, rng.randint(c0 + 1, c1 - 1))
    if side == "left":
        return (rng.randint(r0 + 1, r1 - 1), c0)
    return (rng.randint(r0 + 1, r1 - 1), c1)


def _try_place_frame(
    rng: random.Random,
    grid: np.ndarray,
    blocked: np.ndarray,
    marker_color: int,
    side: str,
) -> dict[str, object] | None:
    h, w = grid.shape
    max_outer_h = min(OUTER_H_RANGE[1], h - 2)
    max_outer_w = min(OUTER_W_RANGE[1], w - 2)
    if max_outer_h < OUTER_H_RANGE[0] or max_outer_w < OUTER_W_RANGE[0]:
        return None

    for _ in range(PLACEMENT_ATTEMPTS):
        outer_h = rng.randint(OUTER_H_RANGE[0], max_outer_h)
        outer_w = rng.randint(OUTER_W_RANGE[0], max_outer_w)
        r0 = rng.randint(1, h - outer_h - 1)
        c0 = rng.randint(1, w - outer_w - 1)
        r1 = r0 + outer_h - 1
        c1 = c0 + outer_w - 1
        gap = _sample_gap(rng, side, r0, c0, r1, c1)
        marker = (rng.randint(r0 + 1, r1 - 1), rng.randint(c0 + 1, c1 - 1))
        ray_length = (outer_h - 2) * (outer_w - 2) - 1
        if ray_length <= 0:
            continue

        box_cells = [(r, c) for r in range(r0, r1 + 1) for c in range(c0, c1 + 1)]
        if any(blocked[r, c] for r, c in box_cells):
            continue

        direction = SIDE_TO_DIR[side]
        ray_cells = _ray_cells(gap, direction, ray_length, h, w)
        if len(ray_cells) < 2:
            continue
        if any(blocked[r, c] for r, c in ray_cells):
            continue

        for r, c in _rectangle_border(r0, c0, r1, c1):
            if (r, c) != gap:
                grid[r, c] = FRAME_COLOR
        grid[marker] = marker_color

        _mark_blocked_box(blocked, r0, c0, r1, c1, pad=1)
        for r, c in ray_cells:
            blocked[r, c] = True

        return {
            "bbox": (r0, c0, r1, c1),
            "gap": gap,
            "direction": direction,
            "marker": marker,
            "marker_color": marker_color,
            "ray_length": ray_length,
        }

    return None


def _place_distractors(
    rng: random.Random, grid: np.ndarray, blocked: np.ndarray, color: int
) -> bool:
    h, w = grid.shape
    count = rng.randint(DISTRACTOR_COUNT_RANGE[0], DISTRACTOR_COUNT_RANGE[1])
    placed = 0
    for _ in range(PLACEMENT_ATTEMPTS * 3):
        if placed == count:
            return True
        r = rng.randrange(h)
        c = rng.randrange(w)
        if blocked[r, c] or int(grid[r, c]) != BG_COLOR:
            continue
        grid[r, c] = color
        _mark_blocked_box(blocked, r, c, r, c, pad=1)
        placed += 1
    return placed == count


def _gen_attempt(rng: random.Random) -> np.ndarray | None:
    h = rng.randint(GRID_H_RANGE[0], GRID_H_RANGE[1])
    w = rng.randint(GRID_W_RANGE[0], GRID_W_RANGE[1])
    grid = np.full((h, w), BG_COLOR, dtype=int)
    blocked = np.zeros((h, w), dtype=bool)

    rect_count = rng.randint(RECT_COUNT_RANGE[0], RECT_COUNT_RANGE[1])
    marker_colors = rng.sample(list(MARKER_COLORS), rect_count)

    sides = list(SIDE_TO_DIR)
    rng.shuffle(sides)
    while len(sides) < rect_count:
        sides.append(rng.choice(tuple(SIDE_TO_DIR)))

    frames = []
    for idx in range(rect_count):
        frame = _try_place_frame(rng, grid, blocked, marker_colors[idx], sides[idx])
        if frame is None:
            return None
        frames.append(frame)

    for color in marker_colors:
        if not _place_distractors(rng, grid, blocked, color):
            return None

    if get_bg(grid) != BG_COLOR:
        return None
    found = _find_frames(grid)
    if len(found) != rect_count:
        return None

    out = main(grid)
    if np.array_equal(out, grid):
        return None

    return grid


def generate_input() -> np.ndarray:
    rng = random.Random()
    for _ in range(GEN_ATTEMPTS):
        grid = _gen_attempt(rng)
        if grid is not None:
            return grid
    raise RuntimeError("generate_input: exhausted attempts")
