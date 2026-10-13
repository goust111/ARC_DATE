import random
from collections import deque

import numpy as np


DIR_VECS = {
    "up": (-1, 0),
    "down": (1, 0),
    "left": (0, -1),
    "right": (0, 1),
}

ARROW_PATTERNS = {
    "up": frozenset({(1, 0), (1, 1), (1, 2), (0, 1)}),
    "down": frozenset({(0, 0), (0, 1), (0, 2), (1, 1)}),
    "left": frozenset({(0, 1), (1, 1), (2, 1), (1, 0)}),
    "right": frozenset({(0, 0), (1, 0), (2, 0), (1, 1)}),
}

ARROW_TIPS = {
    "up": (0, 1),
    "down": (1, 1),
    "left": (1, 0),
    "right": (1, 1),
}

ARROW_SIZES = {
    direction: (
        1 + max(r for r, _ in pattern),
        1 + max(c for _, c in pattern),
    )
    for direction, pattern in ARROW_PATTERNS.items()
}


def _get_bg(grid: np.ndarray) -> int:
    values, counts = np.unique(grid, return_counts=True)
    return int(values[np.argmax(counts)])


def _components_of_value(grid: np.ndarray, value: int) -> list[list[tuple[int, int]]]:
    h, w = grid.shape
    seen = np.zeros((h, w), dtype=bool)
    components: list[list[tuple[int, int]]] = []
    for r in range(h):
        for c in range(w):
            if seen[r, c] or int(grid[r, c]) != value:
                continue
            seen[r, c] = True
            queue = deque([(r, c)])
            comp: list[tuple[int, int]] = []
            while queue:
                cr, cc = queue.popleft()
                comp.append((cr, cc))
                for dr, dc in ((1, 0), (-1, 0), (0, 1), (0, -1)):
                    nr, nc = cr + dr, cc + dc
                    if 0 <= nr < h and 0 <= nc < w and not seen[nr, nc] and int(grid[nr, nc]) == value:
                        seen[nr, nc] = True
                        queue.append((nr, nc))
            components.append(comp)
    return components


def _normalize(cells: list[tuple[int, int]]) -> frozenset[tuple[int, int]]:
    r0 = min(r for r, _ in cells)
    c0 = min(c for _, c in cells)
    return frozenset((r - r0, c - c0) for r, c in cells)


def _detect_arrow_direction(cells: list[tuple[int, int]]) -> str | None:
    if len(cells) != 4:
        return None
    shape = _normalize(cells)
    for direction, pattern in ARROW_PATTERNS.items():
        if shape == pattern:
            return direction
    return None


def _rectangle_boundary(r0: int, c0: int, r1: int, c1: int) -> set[tuple[int, int]]:
    boundary: set[tuple[int, int]] = set()
    for c in range(c0, c1 + 1):
        boundary.add((r0, c))
        boundary.add((r1, c))
    for r in range(r0, r1 + 1):
        boundary.add((r, c0))
        boundary.add((r, c1))
    return boundary


def _detect_frames(grid: np.ndarray, bg: int) -> list[dict]:
    frames: list[dict] = []
    for color in sorted(int(v) for v in np.unique(grid) if int(v) != bg):
        for comp in _components_of_value(grid, color):
            r0 = min(r for r, _ in comp)
            c0 = min(c for _, c in comp)
            r1 = max(r for r, _ in comp)
            c1 = max(c for _, c in comp)
            if r1 - r0 < 2 or c1 - c0 < 2:
                continue
            boundary = _rectangle_boundary(r0, c0, r1, c1)
            comp_set = set(comp)
            if not comp_set <= boundary:
                continue
            missing = boundary - comp_set
            if len(missing) != 1:
                continue
            gap_r, gap_c = next(iter(missing))
            if (gap_r in (r0, r1)) and (gap_c in (c0, c1)):
                continue
            if gap_r == r0:
                gap_side = "up"
            elif gap_r == r1:
                gap_side = "down"
            elif gap_c == c0:
                gap_side = "left"
            else:
                gap_side = "right"
            frames.append(
                {
                    "color": color,
                    "bbox": (r0, c0, r1, c1),
                    "border_cells": comp_set,
                    "gap_side": gap_side,
                    "gap_cell": (gap_r, gap_c),
                }
            )
    frames.sort(key=lambda frame: frame["bbox"])
    return frames


def _frame_arrow_cells(grid: np.ndarray, frame: dict, bg: int) -> list[tuple[int, int]]:
    r0, c0, r1, c1 = frame["bbox"]
    cells: list[tuple[int, int]] = []
    for r in range(r0 + 1, r1):
        for c in range(c0 + 1, c1):
            if int(grid[r, c]) != bg:
                cells.append((r, c))
    return cells


def _exit_shift(frame_bbox: tuple[int, int, int, int], arrow_cells: list[tuple[int, int]], direction: str) -> int:
    r0, c0, r1, c1 = frame_bbox
    if direction == "right":
        return c1 - min(c for _, c in arrow_cells) + 1
    if direction == "left":
        return max(c for _, c in arrow_cells) - c0 + 1
    if direction == "down":
        return r1 - min(r for r, _ in arrow_cells) + 1
    return max(r for r, _ in arrow_cells) - r0 + 1


def _shift_cells(cells: list[tuple[int, int]], dr: int, dc: int, steps: int) -> list[tuple[int, int]]:
    return [(r + dr * steps, c + dc * steps) for r, c in cells]


def _valid_projection(cells: list[tuple[int, int]], h: int, w: int, border_mask: np.ndarray) -> bool:
    for r, c in cells:
        if not (0 <= r < h and 0 <= c < w):
            return False
        if border_mask[r, c]:
            return False
    return True


def main(grid: np.ndarray) -> np.ndarray:
    g = np.asarray(grid, dtype=int)
    h, w = g.shape
    bg = _get_bg(g)
    frames = _detect_frames(g, bg)
    border_mask = np.zeros((h, w), dtype=bool)
    for frame in frames:
        for r, c in frame["border_cells"]:
            border_mask[r, c] = True

    out = g.copy()
    for frame in frames:
        arrow_cells = _frame_arrow_cells(g, frame, bg)
        if not arrow_cells:
            continue
        arrow_colors = {int(g[r, c]) for r, c in arrow_cells}
        if len(arrow_colors) != 1 or frame["color"] in arrow_colors:
            continue
        direction = _detect_arrow_direction(arrow_cells)
        if direction is None or direction != frame["gap_side"]:
            continue

        dr, dc = DIR_VECS[direction]
        steps = _exit_shift(frame["bbox"], arrow_cells, direction)
        final_cells: list[tuple[int, int]] | None = None
        while True:
            shifted = _shift_cells(arrow_cells, dr, dc, steps)
            if not _valid_projection(shifted, h, w, border_mask):
                break
            final_cells = shifted
            steps += 1
        if final_cells is None:
            continue
        for r, c in final_cells:
            out[r, c] = frame["color"]
    return out


GEN_BG = 0
GEN_ARROW = 5
GEN_FRAME_COLORS = [1, 2, 3, 4, 6, 7, 8, 9]
GRID_H_RANGE = (22, 31)
GRID_W_RANGE = (22, 31)
FRAME_H_RANGE = (6, 10)
FRAME_W_RANGE = (6, 10)
FRAME_COUNT_CHOICES = [2, 3, 4]
PLACEMENT_TRIES = 300
GEN_RETRIES = 300


def _all_cells_free(mask: np.ndarray, cells: set[tuple[int, int]]) -> bool:
    return all(not mask[r, c] for r, c in cells)


def _expanded_box_cells(h: int, w: int, r0: int, c0: int, r1: int, c1: int, pad: int = 1) -> set[tuple[int, int]]:
    cells: set[tuple[int, int]] = set()
    for r in range(max(0, r0 - pad), min(h, r1 + pad + 1)):
        for c in range(max(0, c0 - pad), min(w, c1 + pad + 1)):
            cells.add((r, c))
    return cells


def _arrow_cells_from_local(r0: int, c0: int, direction: str, local_r: int, local_c: int) -> list[tuple[int, int]]:
    return [(r0 + 1 + local_r + dr, c0 + 1 + local_c + dc) for dr, dc in ARROW_PATTERNS[direction]]


def _tip_cell(r0: int, c0: int, direction: str, local_r: int, local_c: int) -> tuple[int, int]:
    tip_r, tip_c = ARROW_TIPS[direction]
    return (r0 + 1 + local_r + tip_r, c0 + 1 + local_c + tip_c)


def _gap_cell_for_spec(
    r0: int,
    c0: int,
    height: int,
    width: int,
    gap_side: str,
    tip: tuple[int, int] | None,
    rng: random.Random,
) -> tuple[int, int]:
    if tip is not None:
        tr, tc = tip
        if gap_side == "up":
            return (r0, tc)
        if gap_side == "down":
            return (r0 + height - 1, tc)
        if gap_side == "left":
            return (tr, c0)
        return (tr, c0 + width - 1)
    if gap_side in ("up", "down"):
        return (r0 if gap_side == "up" else r0 + height - 1, rng.randint(c0 + 1, c0 + width - 2))
    return (rng.randint(r0 + 1, r0 + height - 2), c0 if gap_side == "left" else c0 + width - 1)


def _frame_border_cells(
    r0: int,
    c0: int,
    height: int,
    width: int,
    gap_cell: tuple[int, int],
) -> set[tuple[int, int]]:
    cells = _rectangle_boundary(r0, c0, r0 + height - 1, c0 + width - 1)
    cells.remove(gap_cell)
    return cells


def _boundary_final_shift(grid_shape: tuple[int, int], arrow_cells: list[tuple[int, int]], direction: str) -> int:
    h, w = grid_shape
    if direction == "right":
        return (w - 1) - max(c for _, c in arrow_cells)
    if direction == "left":
        return min(c for _, c in arrow_cells)
    if direction == "down":
        return (h - 1) - max(r for r, _ in arrow_cells)
    return min(r for r, _ in arrow_cells)


def _projection_path(
    grid_shape: tuple[int, int],
    frame_bbox: tuple[int, int, int, int],
    arrow_cells: list[tuple[int, int]],
    direction: str,
) -> tuple[set[tuple[int, int]], set[tuple[int, int]]] | None:
    dr, dc = DIR_VECS[direction]
    start = _exit_shift(frame_bbox, arrow_cells, direction)
    end = _boundary_final_shift(grid_shape, arrow_cells, direction)
    if end < start + 2:
        return None
    path: set[tuple[int, int]] = set()
    for steps in range(start, end + 1):
        path.update(_shift_cells(arrow_cells, dr, dc, steps))
    final_cells = set(_shift_cells(arrow_cells, dr, dc, end))
    return path, final_cells


def _frame_position_ranges(
    grid_shape: tuple[int, int],
    height: int,
    width: int,
    direction: str | None,
    matched: bool,
) -> tuple[range, range]:
    h, w = grid_shape
    row_min = 1
    row_max = h - height - 1
    col_min = 1
    col_max = w - width - 1
    if matched and direction == "up":
        row_min = max(row_min, 4)
    if matched and direction == "down":
        row_max = min(row_max, h - height - 5)
    if matched and direction == "left":
        col_min = max(col_min, 4)
    if matched and direction == "right":
        col_max = min(col_max, w - width - 5)
    if row_min > row_max or col_min > col_max:
        return range(0), range(0)
    return range(row_min, row_max + 1), range(col_min, col_max + 1)


def _place_spec(
    grid: np.ndarray,
    blocked: np.ndarray,
    spec: dict,
    color: int,
    rng: random.Random,
) -> bool:
    h, w = grid.shape
    arrow_h, arrow_w = ARROW_SIZES[spec["arrow_dir"]]
    for _ in range(PLACEMENT_TRIES):
        height = rng.randint(*FRAME_H_RANGE)
        width = rng.randint(*FRAME_W_RANGE)
        if height - 2 < arrow_h or width - 2 < arrow_w:
            continue
        row_range, col_range = _frame_position_ranges((h, w), height, width, spec["arrow_dir"], spec["matched"])
        if not row_range or not col_range:
            continue
        r0 = rng.choice(list(row_range))
        c0 = rng.choice(list(col_range))
        box_cells = _expanded_box_cells(h, w, r0, c0, r0 + height - 1, c0 + width - 1, pad=1)
        if not _all_cells_free(blocked, box_cells):
            continue

        local_r = rng.randint(0, height - 2 - arrow_h)
        local_c = rng.randint(0, width - 2 - arrow_w)
        arrow_cells = _arrow_cells_from_local(r0, c0, spec["arrow_dir"], local_r, local_c)
        tip = _tip_cell(r0, c0, spec["arrow_dir"], local_r, local_c) if spec["matched"] else None
        gap_cell = _gap_cell_for_spec(r0, c0, height, width, spec["gap_side"], tip if spec["gap_side"] == spec["arrow_dir"] else None, rng)
        border_cells = _frame_border_cells(r0, c0, height, width, gap_cell)
        if gap_cell in border_cells:
            continue
        if not _all_cells_free(blocked, border_cells | set(arrow_cells)):
            continue

        extra_block: set[tuple[int, int]] = set()
        if spec["matched"]:
            path_info = _projection_path((h, w), (r0, c0, r0 + height - 1, c0 + width - 1), arrow_cells, spec["arrow_dir"])
            if path_info is None:
                continue
            path_cells, final_cells = path_info
            if not _all_cells_free(blocked, path_cells):
                continue
            extra_block |= path_cells
            extra_block |= _expanded_box_cells(
                h,
                w,
                min(r for r, _ in final_cells),
                min(c for _, c in final_cells),
                max(r for r, _ in final_cells),
                max(c for _, c in final_cells),
                pad=1,
            )

        for r, c in border_cells:
            grid[r, c] = color
        for r, c in arrow_cells:
            grid[r, c] = GEN_ARROW

        for r, c in box_cells | extra_block:
            blocked[r, c] = True
        return True
    return False


def _sample_specs(rng: random.Random) -> list[dict]:
    count = rng.choice(FRAME_COUNT_CHOICES)
    matched_count = rng.choice([count // 2, (count + 1) // 2])
    matched_count = max(1, min(count - 1, matched_count))

    matched_dirs = rng.sample(list(DIR_VECS.keys()), matched_count)
    specs = [{"matched": True, "arrow_dir": direction, "gap_side": direction} for direction in matched_dirs]
    for _ in range(count - matched_count):
        arrow_dir = rng.choice(list(DIR_VECS.keys()))
        gap_side = rng.choice([side for side in DIR_VECS if side != arrow_dir])
        specs.append({"matched": False, "arrow_dir": arrow_dir, "gap_side": gap_side})
    return specs


def _gen_attempt(rng: random.Random) -> np.ndarray | None:
    h = rng.randint(*GRID_H_RANGE)
    w = rng.randint(*GRID_W_RANGE)
    grid = np.full((h, w), GEN_BG, dtype=int)
    blocked = np.zeros((h, w), dtype=bool)

    specs = _sample_specs(rng)
    colors = rng.sample(GEN_FRAME_COLORS, len(specs))
    placed_specs: list[tuple[dict, int]] = []

    matched = [(spec, color) for spec, color in zip(specs, colors) if spec["matched"]]
    mismatched = [(spec, color) for spec, color in zip(specs, colors) if not spec["matched"]]
    rng.shuffle(matched)
    rng.shuffle(mismatched)

    for spec, color in matched + mismatched:
        if not _place_spec(grid, blocked, spec, color, rng):
            return None
        placed_specs.append((spec, color))

    try:
        out = main(grid)
    except Exception:
        return None

    if np.array_equal(out, grid):
        return None

    frame_colors = {color for _spec, color in placed_specs}
    diff_mask = out != grid
    if int(diff_mask.sum()) != 4 * sum(1 for spec, _color in placed_specs if spec["matched"]):
        return None
    if not all(int(out[r, c]) in frame_colors for r, c in np.argwhere(diff_mask)):
        return None
    return grid


def generate_input() -> np.ndarray:
    rng = random.Random()
    for _ in range(GEN_RETRIES):
        candidate = _gen_attempt(rng)
        if candidate is not None:
            return candidate
    raise RuntimeError("failed to generate a valid puzzle instance")
