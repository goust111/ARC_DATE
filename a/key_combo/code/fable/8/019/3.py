from __future__ import annotations

import random
from collections import deque

import numpy as np


BG = 0
FRAME = 5
CONTROL_COLORS = (1, 2, 3, 4, 6, 7, 8, 9)
SIDES = ("top", "bottom", "left", "right")


def _neighbors4(r: int, c: int):
    yield r - 1, c
    yield r + 1, c
    yield r, c - 1
    yield r, c + 1


def _in_bounds(shape: tuple[int, int], r: int, c: int) -> bool:
    return 0 <= r < shape[0] and 0 <= c < shape[1]


def _get_bg(grid: np.ndarray) -> int:
    values, counts = np.unique(grid, return_counts=True)
    return int(values[np.argmax(counts)])


def _color_components(grid: np.ndarray, color: int) -> list[list[tuple[int, int]]]:
    h, w = grid.shape
    seen = np.zeros((h, w), dtype=bool)
    comps: list[list[tuple[int, int]]] = []
    for r in range(h):
        for c in range(w):
            if seen[r, c] or int(grid[r, c]) != color:
                continue
            seen[r, c] = True
            q = deque([(r, c)])
            comp = []
            while q:
                rr, cc = q.popleft()
                comp.append((rr, cc))
                for nr, nc in _neighbors4(rr, cc):
                    if _in_bounds(grid.shape, nr, nc) and not seen[nr, nc] and int(grid[nr, nc]) == color:
                        seen[nr, nc] = True
                        q.append((nr, nc))
            comps.append(comp)
    return comps


def _border_cells(r0: int, c0: int, r1: int, c1: int) -> set[tuple[int, int]]:
    cells = set()
    for c in range(c0, c1 + 1):
        cells.add((r0, c))
        cells.add((r1, c))
    for r in range(r0, r1 + 1):
        cells.add((r, c0))
        cells.add((r, c1))
    return cells


def _frame_from_component(comp: list[tuple[int, int]]) -> dict | None:
    rs = [r for r, _ in comp]
    cs = [c for _, c in comp]
    r0, r1 = min(rs), max(rs)
    c0, c1 = min(cs), max(cs)
    if r1 - r0 + 1 < 3 or c1 - c0 + 1 < 3:
        return None

    cells = set(comp)
    border = _border_cells(r0, c0, r1, c1)
    if not cells <= border:
        return None

    gap_cells = sorted(border - cells)
    if len(gap_cells) != 1:
        return None
    gap = gap_cells[0]

    if (gap[0] in (r0, r1)) and (gap[1] in (c0, c1)):
        return None

    if gap[0] == r0:
        step = (-1, 0)
    elif gap[0] == r1:
        step = (1, 0)
    elif gap[1] == c0:
        step = (0, -1)
    else:
        step = (0, 1)

    return {
        "bbox": (r0, c0, r1, c1),
        "gap": gap,
        "step": step,
    }


def _find_frames(grid: np.ndarray, frame_color: int = FRAME) -> list[dict]:
    frames = []
    for comp in _color_components(grid, frame_color):
        frame = _frame_from_component(comp)
        if frame is not None:
            frames.append(frame)
    frames.sort(key=lambda item: item["bbox"])
    return frames


def _ray_match_color(grid: np.ndarray, gap: tuple[int, int], step: tuple[int, int], bg: int) -> int | None:
    r, c = gap[0] + step[0], gap[1] + step[1]
    while _in_bounds(grid.shape, r, c):
        value = int(grid[r, c])
        if value != bg:
            if value != FRAME:
                return value
            return None
        r += step[0]
        c += step[1]
    return None


def main(grid: np.ndarray) -> np.ndarray:
    """
    Fill each hollow grey frame with the color of the single control pixel that can see through its one-cell gap.

    Input structure:
    - The background is black, with several one-pixel-thick grey hollow rectangles.
    - Each grey frame is missing exactly one non-corner border cell, creating a gap to the outside.
    - Colored single-pixel control points lie outside the frames and may share rows or columns with gaps.

    Rule:
    - Detect every grey frame by its bounding box and unique missing border cell.
    - From each frame's gap, look only straight outward; if the first non-background cell on that ray is a colored control point, fill the frame's whole interior with that color.
    - If the outward ray reaches the boundary without a control point, or if another grey frame wall is encountered first, leave that interior black.

    Key insight:
    - The decisive relation is not geometric proximity but unobstructed cardinal line of sight from a control point to a frame's gap; the first non-background cell seen from the gap determines whether the frame fills and with which color.
    """
    g = np.asarray(grid, dtype=int)
    bg = _get_bg(g)
    out = g.copy()

    for frame in _find_frames(g, frame_color=FRAME):
        color = _ray_match_color(g, frame["gap"], frame["step"], bg)
        if color is None:
            continue
        r0, c0, r1, c1 = frame["bbox"]
        if r1 - r0 > 1 and c1 - c0 > 1:
            out[r0 + 1 : r1, c0 + 1 : c1] = color
    return out


def _ray_cells(shape: tuple[int, int], gap: tuple[int, int], step: tuple[int, int]) -> list[tuple[int, int]]:
    cells = []
    r, c = gap[0] + step[0], gap[1] + step[1]
    while _in_bounds(shape, r, c):
        cells.append((r, c))
        r += step[0]
        c += step[1]
    return cells


def _frame_cells(r0: int, c0: int, h: int, w: int, side: str, offset: int) -> tuple[set[tuple[int, int]], tuple[int, int], tuple[int, int]]:
    r1, c1 = r0 + h - 1, c0 + w - 1
    cells = _border_cells(r0, c0, r1, c1)
    if side == "top":
        gap = (r0, c0 + offset)
        step = (-1, 0)
    elif side == "bottom":
        gap = (r1, c0 + offset)
        step = (1, 0)
    elif side == "left":
        gap = (r0 + offset, c0)
        step = (0, -1)
    else:
        gap = (r0 + offset, c1)
        step = (0, 1)
    cells.remove(gap)
    return cells, gap, step


def _reserve_blocked(blocked: np.ndarray, cells: list[tuple[int, int]]) -> None:
    for r, c in cells:
        if _in_bounds(blocked.shape, r, c):
            blocked[r, c] = True


def _expanded_bbox_cells(shape: tuple[int, int], r0: int, c0: int, r1: int, c1: int, pad: int = 1) -> list[tuple[int, int]]:
    cells = []
    for r in range(max(0, r0 - pad), min(shape[0], r1 + pad + 1)):
        for c in range(max(0, c0 - pad), min(shape[1], c1 + pad + 1)):
            cells.append((r, c))
    return cells


def _choose_color(used: set[int], exclude: set[int] | None = None) -> int:
    banned = set(used)
    if exclude is not None:
        banned |= set(exclude)
    choices = [color for color in CONTROL_COLORS if color not in banned]
    if not choices:
        choices = [color for color in CONTROL_COLORS if exclude is None or color not in exclude]
    return random.choice(choices)


def _place_frame(
    grid: np.ndarray,
    blocked: np.ndarray,
    status: str,
    used_colors: set[int],
) -> dict | None:
    h_lim, w_lim = grid.shape
    for _ in range(300):
        h = random.randint(4, 8)
        w = random.randint(4, 8)
        if h >= h_lim - 2 or w >= w_lim - 2:
            continue
        r0 = random.randint(1, h_lim - h - 1)
        c0 = random.randint(1, w_lim - w - 1)
        r1, c1 = r0 + h - 1, c0 + w - 1
        bbox_cells = _expanded_bbox_cells(grid.shape, r0, c0, r1, c1, pad=1)
        if any(blocked[r, c] for r, c in bbox_cells):
            continue

        side = random.choice(SIDES)
        offset = random.randint(1, (w - 2) if side in ("top", "bottom") else (h - 2))
        frame_cells, gap, step = _frame_cells(r0, c0, h, w, side, offset)
        ray = _ray_cells(grid.shape, gap, step)
        if not ray:
            continue

        need_clear = []
        controls: list[tuple[tuple[int, int], int]] = []
        if status == "hit":
            if len(ray) < 2:
                continue
            near_idx = random.randint(1, min(len(ray), 6)) - 1
            need_clear = ray[: near_idx + 1]
            if any(blocked[r, c] for r, c in need_clear):
                continue
            color = _choose_color(used_colors)
            controls.append((ray[near_idx], color))
            used_colors.add(color)
        elif status == "hit_with_decoy":
            if len(ray) < 4:
                continue
            near_idx = random.randint(1, min(len(ray) - 2, 4)) - 1
            far_idx = random.randint(near_idx + 2, min(len(ray), near_idx + 6)) - 1
            need_clear = ray[: far_idx + 1]
            if any(blocked[r, c] for r, c in need_clear):
                continue
            near_color = _choose_color(used_colors)
            far_color = _choose_color(used_colors, exclude={near_color})
            controls.append((ray[near_idx], near_color))
            controls.append((ray[far_idx], far_color))
            used_colors.add(near_color)
            used_colors.add(far_color)
        else:
            if any(blocked[r, c] or int(grid[r, c]) != BG for r, c in ray):
                continue

        for r, c in frame_cells:
            grid[r, c] = FRAME
        for (r, c), color in controls:
            grid[r, c] = color

        _reserve_blocked(blocked, bbox_cells)
        _reserve_blocked(blocked, need_clear)
        return {
            "bbox": (r0, c0, r1, c1),
            "gap": gap,
            "step": step,
            "status": status,
            "controls": controls,
        }
    return None


def _valid_generated_case(grid: np.ndarray, expected_frames: int) -> bool:
    frames = _find_frames(grid)
    if len(frames) != expected_frames:
        return False

    hits = 0
    misses = 0
    for frame in frames:
        if _ray_match_color(grid, frame["gap"], frame["step"], BG) is None:
            misses += 1
        else:
            hits += 1

    if hits == 0 or misses == 0:
        return False

    colors = {int(grid[r, c]) for r, c in np.argwhere((grid != BG) & (grid != FRAME))}
    if len(colors) < 2:
        return False

    out = main(grid)
    if np.array_equal(out, grid):
        return False
    return out.shape == grid.shape


def _attempt_generate() -> np.ndarray | None:
    height = random.randint(18, 30)
    width = random.randint(18, 30)
    grid = np.full((height, width), BG, dtype=int)
    blocked = np.zeros((height, width), dtype=bool)
    used_colors: set[int] = set()

    frame_count = random.randint(2, 4)
    statuses = ["hit", "empty"]
    if frame_count >= 3:
        statuses.append("hit_with_decoy")
    while len(statuses) < frame_count:
        statuses.append(random.choices(["hit", "empty", "hit_with_decoy"], weights=[3, 2, 2])[0])

    placed = []
    for status in [s for s in statuses if s != "empty"] + [s for s in statuses if s == "empty"]:
        spec = _place_frame(grid, blocked, status, used_colors)
        if spec is None:
            return None
        placed.append(spec)

    if not _valid_generated_case(grid, len(placed)):
        return None
    return grid


def generate_input() -> np.ndarray:
    for _ in range(400):
        grid = _attempt_generate()
        if grid is not None:
            return grid
    raise RuntimeError("failed to generate valid input for puzzle 3")
