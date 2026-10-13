from __future__ import annotations

import random

import numpy as np


BG = 0
SIDES = ("top", "bottom", "left", "right")
CARDINAL4 = ((-1, 0), (1, 0), (0, -1), (0, 1))

# Generator bounds.
GRID_MIN = 18
GRID_MAX = 28
FRAME_COUNT_RANGE = (1, 4)
FRAME_SIZE_RANGE = (5, 9)
PATTERN_CELLS_MAX = 8
MAX_FRAME_PLACEMENT_ATTEMPTS = 160
MAX_GENERATION_ATTEMPTS = 400


def _components(mask: np.ndarray) -> list[np.ndarray]:
    h, w = mask.shape
    seen = np.zeros((h, w), dtype=bool)
    comps: list[np.ndarray] = []
    for r in range(h):
        for c in range(w):
            if not mask[r, c] or seen[r, c]:
                continue
            stack = [(r, c)]
            seen[r, c] = True
            coords: list[tuple[int, int]] = []
            while stack:
                cr, cc = stack.pop()
                coords.append((cr, cc))
                for dr, dc in CARDINAL4:
                    nr, nc = cr + dr, cc + dc
                    if 0 <= nr < h and 0 <= nc < w and mask[nr, nc] and not seen[nr, nc]:
                        seen[nr, nc] = True
                        stack.append((nr, nc))
            comps.append(np.asarray(coords, dtype=int))
    return comps


def _bbox(coords: np.ndarray) -> tuple[int, int, int, int]:
    rows = coords[:, 0]
    cols = coords[:, 1]
    return int(rows.min()), int(cols.min()), int(rows.max()), int(cols.max())


def _edge_colors(grid: np.ndarray) -> dict[str, int]:
    h, w = grid.shape
    assert h >= 5 and w >= 5

    def unique_non_bg(arr: np.ndarray) -> int:
        vals = {int(v) for v in np.unique(arr) if int(v) != BG}
        assert len(vals) == 1, "each border strip must use one visible color"
        return vals.pop()

    colors = {
        "top": unique_non_bg(grid[0, 1:-1]),
        "bottom": unique_non_bg(grid[-1, 1:-1]),
        "left": unique_non_bg(grid[:, 0]),
        "right": unique_non_bg(grid[:, -1]),
    }
    assert len(set(colors.values())) == 4, "border strip colors must be distinct"
    return colors


def _frame_spec(component: np.ndarray, shape: tuple[int, int]) -> dict | None:
    h, w = shape
    r0, c0, r1, c1 = _bbox(component)
    if r0 <= 0 or c0 <= 0 or r1 >= h - 1 or c1 >= w - 1:
        return None

    rect_h = r1 - r0 + 1
    rect_w = c1 - c0 + 1
    if rect_h < 4 or rect_w < 4:
        return None

    coords = {(int(r), int(c)) for r, c in component}
    perimeter = set()
    for c in range(c0, c1 + 1):
        perimeter.add((r0, c))
        perimeter.add((r1, c))
    for r in range(r0, r1 + 1):
        perimeter.add((r, c0))
        perimeter.add((r, c1))

    if not coords <= perimeter:
        return None

    missing = list(perimeter - coords)
    if len(missing) != 1:
        return None
    gap_r, gap_c = missing[0]
    if (gap_r in (r0, r1)) and (gap_c in (c0, c1)):
        return None

    if len(coords) != len(perimeter) - 1:
        return None

    if gap_r == r0:
        gap_side = "top"
    elif gap_r == r1:
        gap_side = "bottom"
    elif gap_c == c0:
        gap_side = "left"
    else:
        gap_side = "right"

    return {
        "bbox": (r0, c0, r1, c1),
        "gap_side": gap_side,
    }


def _find_frames(grid: np.ndarray) -> list[dict]:
    frames: list[dict] = []
    for color in np.unique(grid):
        color = int(color)
        if color == BG:
            continue
        for comp in _components(grid == color):
            spec = _frame_spec(comp, grid.shape)
            if spec is not None:
                frames.append(spec)
    frames.sort(key=lambda item: item["bbox"])
    assert frames, "expected at least one gapped rectangular frame"
    return frames


def _project_cells(cells: list[tuple[int, int]], side: str, shape: tuple[int, int]) -> list[tuple[int, int]]:
    rows = [r for r, _ in cells]
    cols = [c for _, c in cells]
    h, w = shape

    if side == "top":
        dr, dc = 1 - min(rows), 0
    elif side == "bottom":
        dr, dc = (h - 2) - max(rows), 0
    elif side == "left":
        dr, dc = 0, 1 - min(cols)
    else:
        dr, dc = 0, (w - 2) - max(cols)

    projected = [(r + dr, c + dc) for r, c in cells]
    assert all(0 < r < h - 1 and 0 < c < w - 1 for r, c in projected)
    return projected


def main(grid: np.ndarray) -> np.ndarray:
    """
    Project each frame's interior pattern to the grid edge indicated by the frame gap, recoloring the copy with that edge strip's color.

    Input structure:
    - The background is black.
    - The grid contains several one-pixel-thick rectangular colored frames; each frame border has exactly one one-pixel gap on one side.
    - Each frame encloses a small non-black pattern, and the four outer sides of the grid carry one-pixel-wide color strips with different colors.

    Rule:
    - For each frame, detect which side contains the unique gap: top, bottom, left, or right.
    - Copy every non-black cell inside that frame and translate the copy toward the matching grid edge, keeping the perpendicular coordinates unchanged and moving the copy so it sits immediately inside the edge strip.
    - Recolor the translated copy with the target edge strip's color, while leaving the original frame and its interior pattern unchanged.

    Key insight:
    - The frame gap is a directional pointer: it selects both the destination edge and the visible color of the projected copy.
    """
    g = np.asarray(grid, dtype=int)
    assert g.ndim == 2

    edge_colors = _edge_colors(g)
    frames = _find_frames(g)
    out = g.copy()

    for frame in frames:
        r0, c0, r1, c1 = frame["bbox"]
        interior = g[r0 + 1 : r1, c0 + 1 : c1]
        pattern_coords = [
            (r0 + 1 + int(r), c0 + 1 + int(c))
            for r, c in np.argwhere(interior != BG)
        ]
        assert pattern_coords, "each frame must contain a non-black interior pattern"
        projected = _project_cells(pattern_coords, frame["gap_side"], g.shape)
        color = edge_colors[frame["gap_side"]]
        for r, c in projected:
            out[r, c] = color

    return out


def _paint_border_strips(grid: np.ndarray, edge_colors: dict[str, int]) -> None:
    grid[0, 1:-1] = edge_colors["top"]
    grid[-1, 1:-1] = edge_colors["bottom"]
    grid[:, 0] = edge_colors["left"]
    grid[:, -1] = edge_colors["right"]


def _random_connected_pattern(height: int, width: int) -> np.ndarray:
    area = height * width
    target = random.randint(2, min(PATTERN_CELLS_MAX, area))
    start = (random.randrange(height), random.randrange(width))
    chosen = {start}

    while len(chosen) < target:
        frontier: list[tuple[int, int]] = []
        for r, c in chosen:
            for dr, dc in CARDINAL4:
                nr, nc = r + dr, c + dc
                if 0 <= nr < height and 0 <= nc < width and (nr, nc) not in chosen:
                    frontier.append((nr, nc))
        if not frontier:
            break
        chosen.add(random.choice(frontier))

    rows = {r for r, _ in chosen}
    cols = {c for _, c in chosen}
    if len(rows) == 1 and len(cols) == 1:
        rr, cc = next(iter(chosen))
        for dr, dc in CARDINAL4:
            nr, nc = rr + dr, cc + dc
            if 0 <= nr < height and 0 <= nc < width:
                chosen.add((nr, nc))
                break

    mask = np.zeros((height, width), dtype=bool)
    for r, c in chosen:
        mask[r, c] = True
    return mask


def _pattern_extent(mask: np.ndarray) -> tuple[int, int]:
    coords = np.argwhere(mask)
    rr0, cc0 = coords.min(axis=0)
    rr1, cc1 = coords.max(axis=0)
    return int(rr1 - rr0 + 1), int(cc1 - cc0 + 1)


def _frame_cells(r0: int, c0: int, rect_h: int, rect_w: int, side: str, gap_idx: int) -> list[tuple[int, int]]:
    cells: list[tuple[int, int]] = []
    for c in range(c0, c0 + rect_w):
        cells.append((r0, c))
        cells.append((r0 + rect_h - 1, c))
    for r in range(r0 + 1, r0 + rect_h - 1):
        cells.append((r, c0))
        cells.append((r, c0 + rect_w - 1))

    if side == "top":
        gap = (r0, c0 + gap_idx)
    elif side == "bottom":
        gap = (r0 + rect_h - 1, c0 + gap_idx)
    elif side == "left":
        gap = (r0 + gap_idx, c0)
    else:
        gap = (r0 + gap_idx, c0 + rect_w - 1)

    return [cell for cell in cells if cell != gap]


def _bbox_clear(mask: np.ndarray, r0: int, c0: int, r1: int, c1: int, pad: int = 0) -> bool:
    rr0 = max(0, r0 - pad)
    cc0 = max(0, c0 - pad)
    rr1 = min(mask.shape[0] - 1, r1 + pad)
    cc1 = min(mask.shape[1] - 1, c1 + pad)
    return not bool(mask[rr0 : rr1 + 1, cc0 : cc1 + 1].any())


def _attempt_generate() -> np.ndarray | None:
    h = random.randint(GRID_MIN, GRID_MAX)
    w = random.randint(GRID_MIN, GRID_MAX)
    grid = np.full((h, w), BG, dtype=int)

    all_colors = list(range(1, 10))
    random.shuffle(all_colors)
    edge_colors = {
        "top": all_colors[0],
        "bottom": all_colors[1],
        "left": all_colors[2],
        "right": all_colors[3],
    }
    _paint_border_strips(grid, edge_colors)

    remaining_colors = [c for c in range(1, 10) if c not in edge_colors.values()]
    frame_count = random.randint(*FRAME_COUNT_RANGE)
    directions = random.sample(SIDES, k=frame_count)
    frame_colors = random.sample(remaining_colors, k=frame_count)

    occupied = grid != BG
    reserved_projection = np.zeros_like(grid, dtype=bool)
    projections: list[tuple[str, list[tuple[int, int]]]] = []

    for side, frame_color in zip(directions, frame_colors, strict=False):
        placed = False
        for _ in range(MAX_FRAME_PLACEMENT_ATTEMPTS):
            rect_h = random.randint(*FRAME_SIZE_RANGE)
            rect_w = random.randint(*FRAME_SIZE_RANGE)
            if rect_h > h - 4 or rect_w > w - 4:
                continue

            pattern_mask = _random_connected_pattern(rect_h - 2, rect_w - 2)
            pat_h, pat_w = _pattern_extent(pattern_mask)

            min_r0 = 2
            max_r0 = h - rect_h - 2
            min_c0 = 2
            max_c0 = w - rect_w - 2

            if side == "top":
                min_r0 = max(min_r0, pat_h + 1)
            elif side == "bottom":
                max_r0 = min(max_r0, h - pat_h - rect_h - 1)
            elif side == "left":
                min_c0 = max(min_c0, pat_w + 1)
            else:
                max_c0 = min(max_c0, w - pat_w - rect_w - 1)

            if min_r0 > max_r0 or min_c0 > max_c0:
                continue

            r0 = random.randint(min_r0, max_r0)
            c0 = random.randint(min_c0, max_c0)
            r1 = r0 + rect_h - 1
            c1 = c0 + rect_w - 1

            combined_blockers = occupied | reserved_projection
            if not _bbox_clear(combined_blockers, r0, c0, r1, c1, pad=1):
                continue

            gap_idx = random.randint(1, rect_w - 2) if side in ("top", "bottom") else random.randint(1, rect_h - 2)
            frame_cells = _frame_cells(r0, c0, rect_h, rect_w, side, gap_idx)
            pattern_cells = [
                (r0 + 1 + int(pr), c0 + 1 + int(pc))
                for pr, pc in np.argwhere(pattern_mask)
            ]
            projected = _project_cells(pattern_cells, side, grid.shape)

            if set(projected) & set(frame_cells):
                continue
            if set(projected) & set(pattern_cells):
                continue
            if any(occupied[r, c] or reserved_projection[r, c] for r, c in projected):
                continue

            pattern_color_choices = [c for c in remaining_colors if c != frame_color]
            pattern_color = random.choice(pattern_color_choices)

            for r, c in frame_cells:
                grid[r, c] = frame_color
                occupied[r, c] = True
            for r, c in pattern_cells:
                grid[r, c] = pattern_color
                occupied[r, c] = True
            for r, c in projected:
                reserved_projection[r, c] = True

            projections.append((side, projected))
            placed = True
            break

        if not placed:
            return None

    expected = grid.copy()
    for side, cells in projections:
        expected[tuple(np.array(cells).T)] = edge_colors[side]

    if not np.array_equal(main(grid), expected):
        return None
    return grid


def generate_input() -> np.ndarray:
    for _ in range(MAX_GENERATION_ATTEMPTS):
        grid = _attempt_generate()
        if grid is not None:
            return grid
    raise RuntimeError("failed to generate a valid input for puzzle 6")
