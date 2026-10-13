from __future__ import annotations

import random

import numpy as np


BLACK = 0
GREY = 5
MAROON = 9

DIRS4 = [(1, 0), (-1, 0), (0, 1), (0, -1)]
DIRS8 = DIRS4 + [(1, 1), (1, -1), (-1, 1), (-1, -1)]
DIRECTION_ORDER = ("up", "down", "left", "right")
ARROW_PATTERNS = {
    "up": {(0, 1), (1, 0), (1, 2)},
    "down": {(2, 1), (1, 0), (1, 2)},
    "left": {(1, 0), (0, 1), (2, 1)},
    "right": {(1, 2), (0, 1), (2, 1)},
}
ENTRY_H = 3
ENTRY_W = 6
ENTRY_GAP = 1


def _components(mask: np.ndarray, connectivity: int = 4) -> list[list[tuple[int, int]]]:
    h, w = mask.shape
    seen = np.zeros((h, w), dtype=bool)
    steps = DIRS4 if connectivity == 4 else DIRS8
    out: list[list[tuple[int, int]]] = []
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
                for dr, dc in steps:
                    nr = cr + dr
                    nc = cc + dc
                    if 0 <= nr < h and 0 <= nc < w and mask[nr, nc] and not seen[nr, nc]:
                        seen[nr, nc] = True
                        stack.append((nr, nc))
            out.append(comp)
    return out


def _bbox(points: list[tuple[int, int]]) -> tuple[int, int, int, int]:
    rs = [r for r, _ in points]
    cs = [c for _, c in points]
    return min(rs), min(cs), max(rs), max(cs)


def _frame_components(grid: np.ndarray) -> list[tuple[int, int, int, int]]:
    frames: list[tuple[int, int, int, int]] = []
    for comp in _components(grid == MAROON, connectivity=4):
        r0, c0, r1, c1 = _bbox(comp)
        side_h = r1 - r0 + 1
        side_w = c1 - c0 + 1
        if side_h != side_w or not (5 <= side_h <= 7):
            continue
        patch = grid[r0 : r1 + 1, c0 : c1 + 1]
        border = np.zeros_like(patch, dtype=bool)
        border[0, :] = True
        border[-1, :] = True
        border[:, 0] = True
        border[:, -1] = True
        if int(border.sum()) != len(comp):
            continue
        if not np.all(patch[border] == MAROON):
            continue
        if np.any(patch[1:-1, 1:-1] == MAROON):
            continue
        frames.append((r0, c0, r1, c1))
    assert frames, "expected at least one maroon frame"
    return frames


def _read_arrow(zone: np.ndarray) -> str | None:
    if zone.shape != (3, 3):
        return None
    grey_pts = {(int(r), int(c)) for r, c in np.argwhere(zone == GREY)}
    if len(grey_pts) != 3:
        return None
    if np.any((zone != BLACK) & (zone != GREY)):
        return None
    for direction, pattern in ARROW_PATTERNS.items():
        if grey_pts == pattern:
            return direction
    return None


def _legend_mapping(grid: np.ndarray) -> dict[str, int]:
    h, w = grid.shape
    mapping: dict[str, int] = {}
    seen_entries: set[tuple[int, int]] = set()

    for r in range(h - ENTRY_H + 1):
        for c in range(w - ENTRY_W + 1):
            if r != 0 and c != 0:
                continue
            swatch = grid[r : r + 2, c : c + 2]
            color = int(swatch[0, 0])
            if color in (BLACK, GREY, MAROON):
                continue
            if not np.all(swatch == color):
                continue
            if np.any(grid[r : r + ENTRY_H, c + 2] != BLACK):
                continue
            if np.any(grid[r + 2, c : c + 2] != BLACK):
                continue
            direction = _read_arrow(grid[r : r + 3, c + 3 : c + 6])
            if direction is None:
                continue
            seen_entries.add((r, c))
            mapping[direction] = color

    assert len(seen_entries) >= 4, "expected legend entries on the top or left edge"
    assert set(mapping) >= set(DIRECTION_ORDER), "legend must cover all directions"
    return mapping


def _pattern_direction(mask: np.ndarray) -> str:
    extra_lr = mask & ~np.fliplr(mask)
    extra_ud = mask & ~np.flipud(mask)
    has_lr = bool(extra_lr.any())
    has_ud = bool(extra_ud.any())
    assert has_lr != has_ud, "pattern must have exactly one reflection defect axis"
    if has_lr:
        cols = np.argwhere(extra_lr)[:, 1]
        mid = (mask.shape[1] - 1) / 2.0
        return "left" if float(cols.mean()) < mid else "right"
    rows = np.argwhere(extra_ud)[:, 0]
    mid = (mask.shape[0] - 1) / 2.0
    return "up" if float(rows.mean()) < mid else "down"


def main(grid: np.ndarray) -> np.ndarray:
    g = np.asarray(grid, dtype=int)
    assert g.ndim == 2
    assert np.all((g >= 0) & (g <= 9))

    direction_to_color = _legend_mapping(g)
    frames = _frame_components(g)

    out = g.copy()
    for r0, c0, r1, c1 in frames:
        interior = g[r0 + 1 : r1, c0 + 1 : c1]
        colors = set(map(int, np.unique(interior))) - {BLACK}
        assert len(colors) == 1, "each frame interior must contain exactly one foreground color"
        pattern_color = int(next(iter(colors)))
        direction = _pattern_direction(interior == pattern_color)

        frame_patch = out[r0 : r1 + 1, c0 : c1 + 1]
        border = np.zeros_like(frame_patch, dtype=bool)
        border[0, :] = True
        border[-1, :] = True
        border[:, 0] = True
        border[:, -1] = True
        frame_patch[border] = direction_to_color[direction]
    return out


def _orbit_groups(n: int) -> list[list[tuple[int, int]]]:
    groups: list[list[tuple[int, int]]] = []
    seen: set[tuple[int, int]] = set()
    for r in range(n):
        for c in range(n):
            if (r, c) in seen:
                continue
            orbit = {
                (r, c),
                (r, n - 1 - c),
                (n - 1 - r, c),
                (n - 1 - r, n - 1 - c),
            }
            group = sorted(orbit)
            seen.update(group)
            groups.append(group)
    return groups


def _defect_candidates(mask: np.ndarray, direction: str) -> list[list[tuple[int, int]]]:
    n = mask.shape[0]
    half_rows = range(max(1, n // 2))
    half_cols = range(max(1, n // 2))
    candidates: list[list[tuple[int, int]]] = []
    for r in half_rows:
        for c in half_cols:
            left_pair = [(r, c), (n - 1 - r, c)]
            right_pair = [(r, n - 1 - c), (n - 1 - r, n - 1 - c)]
            top_pair = [(r, c), (r, n - 1 - c)]
            bottom_pair = [(n - 1 - r, c), (n - 1 - r, n - 1 - c)]

            if direction == "left":
                need = left_pair
                forbid = right_pair
            elif direction == "right":
                need = right_pair
                forbid = left_pair
            elif direction == "up":
                need = top_pair
                forbid = bottom_pair
            else:
                need = bottom_pair
                forbid = top_pair

            need_set = sorted(set(need))
            forbid_set = sorted(set(forbid))
            if any(mask[rr, cc] for rr, cc in need_set):
                continue
            if any(mask[rr, cc] for rr, cc in forbid_set):
                continue
            candidates.append(need_set)
    return candidates


def _make_pattern(n: int, direction: str) -> np.ndarray | None:
    groups = _orbit_groups(n)
    for _ in range(80):
        mask = np.zeros((n, n), dtype=bool)
        random.shuffle(groups)
        fill_prob = random.uniform(0.25, 0.6)
        for group in groups:
            if random.random() < fill_prob:
                for r, c in group:
                    mask[r, c] = True
        if int(mask.sum()) < 2 or int(mask.sum()) > n * n - 3:
            continue
        candidates = _defect_candidates(mask, direction)
        if not candidates:
            continue
        for r, c in random.choice(candidates):
            mask[r, c] = True
        if _pattern_direction(mask) != direction:
            continue
        return mask
    return None


def _entry_tile(color: int, direction: str) -> np.ndarray:
    tile = np.zeros((ENTRY_H, ENTRY_W), dtype=int)
    tile[0:2, 0:2] = color
    for r, c in ARROW_PATTERNS[direction]:
        tile[r, c + 3] = GREY
    return tile


def _place_legend(
    grid: np.ndarray, direction_to_color: dict[str, int], orientation: str
) -> tuple[int, int, int, int]:
    h, w = grid.shape
    order = list(DIRECTION_ORDER)
    random.shuffle(order)

    if orientation == "top":
        total_w = len(order) * ENTRY_W + (len(order) - 1) * ENTRY_GAP
        start_c = random.randint(0, w - total_w)
        for i, direction in enumerate(order):
            c0 = start_c + i * (ENTRY_W + ENTRY_GAP)
            grid[0:ENTRY_H, c0 : c0 + ENTRY_W] = _entry_tile(
                direction_to_color[direction], direction
            )
        return (0, start_c, ENTRY_H - 1, start_c + total_w - 1)

    total_h = len(order) * ENTRY_H + (len(order) - 1) * ENTRY_GAP
    start_r = random.randint(0, h - total_h)
    for i, direction in enumerate(order):
        r0 = start_r + i * (ENTRY_H + ENTRY_GAP)
        grid[r0 : r0 + ENTRY_H, 0:ENTRY_W] = _entry_tile(
            direction_to_color[direction], direction
        )
    return (start_r, 0, start_r + total_h - 1, ENTRY_W - 1)


def _boxes_overlap(a: tuple[int, int, int, int], b: tuple[int, int, int, int]) -> bool:
    return a[0] <= b[2] and b[0] <= a[2] and a[1] <= b[3] and b[1] <= a[3]


def _paint_frame(
    grid: np.ndarray,
    r0: int,
    c0: int,
    side: int,
    pattern_color: int,
    direction: str,
) -> None:
    r1 = r0 + side - 1
    c1 = c0 + side - 1
    grid[r0, c0 : c1 + 1] = MAROON
    grid[r1, c0 : c1 + 1] = MAROON
    grid[r0 : r1 + 1, c0] = MAROON
    grid[r0 : r1 + 1, c1] = MAROON

    pattern = _make_pattern(side - 2, direction)
    assert pattern is not None
    interior = grid[r0 + 1 : r1, c0 + 1 : c1]
    interior[pattern] = pattern_color


def _sample_grid_shape(orientation: str) -> tuple[int, int]:
    if orientation == "top":
        return random.randint(18, 28), random.randint(27, 34)
    return random.randint(18, 30), random.randint(20, 30)


def _place_frames(
    grid: np.ndarray,
    legend_box: tuple[int, int, int, int],
    direction_to_color: dict[str, int],
) -> bool:
    h, w = grid.shape
    frame_count = random.randint(2, 4)
    placed: list[tuple[int, int, int, int]] = [legend_box]
    other_colors = [1, 2, 3, 4, 6, 7, 8]

    if legend_box[0] == 0:
        min_r = legend_box[2] + 2
        min_c = 1
    else:
        min_r = 1
        min_c = legend_box[3] + 2

    for _ in range(frame_count):
        side = random.randint(5, 7)
        if min_r + side > h - 1 or min_c + side > w - 1:
            return False
        direction = random.choice(DIRECTION_ORDER)
        pattern_choices = [c for c in other_colors if c != direction_to_color[direction]]
        pattern_color = random.choice(pattern_choices)

        placed_here = False
        for _attempt in range(120):
            r0 = random.randint(min_r, h - side - 1)
            c0 = random.randint(min_c, w - side - 1)
            box = (r0 - 1, c0 - 1, r0 + side, c0 + side)
            if any(_boxes_overlap(box, prev) for prev in placed):
                continue
            _paint_frame(grid, r0, c0, side, pattern_color, direction)
            placed.append(box)
            placed_here = True
            break
        if not placed_here:
            return False
    return True


def _generate_candidate() -> np.ndarray | None:
    orientation = random.choice(("top", "left"))
    h, w = _sample_grid_shape(orientation)
    grid = np.zeros((h, w), dtype=int)

    legend_colors = random.sample([1, 2, 3, 4, 6, 7, 8], 4)
    random.shuffle(legend_colors)
    direction_to_color = {
        direction: color for direction, color in zip(DIRECTION_ORDER, legend_colors, strict=False)
    }

    legend_box = _place_legend(grid, direction_to_color, orientation)
    if not _place_frames(grid, legend_box, direction_to_color):
        return None

    try:
        out = main(grid)
    except Exception:
        return None
    if np.array_equal(out, grid):
        return None
    return grid


def generate_input() -> np.ndarray:
    for _ in range(300):
        grid = _generate_candidate()
        if grid is not None:
            return grid
    raise RuntimeError("failed to generate a valid puzzle instance")
