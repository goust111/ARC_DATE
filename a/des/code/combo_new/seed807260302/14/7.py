import random
from typing import Dict, List, Optional, Sequence, Tuple

import numpy as np


BG = 0
WALL = 5
PALETTE = [1, 2, 3, 4, 6, 7, 8, 9]
DIRS = ("up", "down", "left", "right")
GEN_MAX_ATTEMPTS = 300


def _mode(values: np.ndarray) -> int:
    vals, counts = np.unique(np.asarray(values, dtype=int), return_counts=True)
    return int(vals[np.argmax(counts)])


def _wall_color(grid: np.ndarray) -> int:
    g = np.asarray(grid, dtype=int)
    h, w = g.shape
    best_score: Optional[Tuple[int, int]] = None
    best_color = int(g[0, 0])
    row_threshold = max(1, (3 * w + 3) // 4)
    col_threshold = max(1, (3 * h + 3) // 4)
    for color in map(int, np.unique(g)):
        row_hits = sum(int(np.count_nonzero(g[r, :] == color) >= row_threshold) for r in range(h))
        col_hits = sum(int(np.count_nonzero(g[:, c] == color) >= col_threshold) for c in range(w))
        score = (row_hits + col_hits, int(np.count_nonzero(g == color)))
        if best_score is None or score > best_score:
            best_score = score
            best_color = color
    return best_color


def _line_indices(grid: np.ndarray, wall: int, axis: int) -> List[int]:
    g = np.asarray(grid, dtype=int)
    h, w = g.shape
    row_threshold = max(1, (3 * w + 3) // 4)
    col_threshold = max(1, (3 * h + 3) // 4)
    if axis == 0:
        return [r for r in range(h) if np.count_nonzero(g[r, :] == wall) >= row_threshold]
    return [c for c in range(w) if np.count_nonzero(g[:, c] == wall) >= col_threshold]


def _marker_info(
    grid: np.ndarray, r0: int, r1: int, c0: int, c1: int, wall: int
) -> Optional[Tuple[str, int]]:
    g = np.asarray(grid, dtype=int)
    markers: List[Tuple[str, int]] = []

    sides = {
        "up": g[r0, c0 + 1 : c1],
        "down": g[r1, c0 + 1 : c1],
        "left": g[r0 + 1 : r1, c0],
        "right": g[r0 + 1 : r1, c1],
    }
    for side, cells in sides.items():
        idxs = np.flatnonzero(cells != wall)
        if len(idxs) > 1:
            return None
        if len(idxs) == 1:
            markers.append((side, int(cells[int(idxs[0])])))

    if len(markers) != 1:
        return None
    return markers[0]


def _shift_patch(patch: np.ndarray, bg: int, side: str, fill_color: int) -> np.ndarray:
    src = np.asarray(patch, dtype=int)
    fg = src != bg
    if not bool(fg.any()):
        return src.copy()

    coords = np.argwhere(fg)
    dr = 0
    dc = 0
    if side == "up":
        dr = -int(coords[:, 0].min())
    elif side == "down":
        dr = int(src.shape[0] - 1 - coords[:, 0].max())
    elif side == "left":
        dc = -int(coords[:, 1].min())
    else:
        dc = int(src.shape[1] - 1 - coords[:, 1].max())

    out = src.copy()
    out[fg] = bg
    moved = np.zeros_like(fg)
    for r, c in coords:
        rr = int(r + dr)
        cc = int(c + dc)
        out[rr, cc] = int(src[r, c])
        moved[rr, cc] = True

    vacated = fg & (~moved)
    out[vacated] = fill_color
    return out


def main(grid: np.ndarray) -> np.ndarray:
    """
    Push each marked panel's interior figure against the marked side and paint the vacated trail with the marker color.

    Input structure:
    - The background is black and grey one-pixel lines partition the grid into equal-sized rectangular panels.
    - Each panel interior contains a small foreground figure on the black background.
    - Some panels have exactly one colored marker pixel replacing one grey border pixel on one side of that panel.

    Rule:
    - Identify every panel whose border contains exactly one non-grey marker.
    - The marker side gives the push direction: top means push upward, bottom downward, left leftward, and right rightward.
    - Translate the entire interior figure rigidly in that direction until its leading edge touches the panel's inner boundary.
    - Fill cells that were occupied before the push but are no longer occupied afterward with the marker color.
    - Panels without a marker stay unchanged, and every marker pixel is restored to grey in the output border.

    Key insight:
    - The marker position and the marker color encode two independent signals: side chooses the translation direction, while color chooses the trailing fill left behind by that translation.
    """
    g = np.asarray(grid, dtype=int)
    h, w = g.shape
    if h == 0 or w == 0:
        return g.copy()

    wall = _wall_color(g)
    line_rows = _line_indices(g, wall, axis=0)
    line_cols = _line_indices(g, wall, axis=1)
    if len(line_rows) < 2 or len(line_cols) < 2:
        return g.copy()

    interior_mask = np.ones((h, w), dtype=bool)
    interior_mask[line_rows, :] = False
    interior_mask[:, line_cols] = False
    if not bool(interior_mask.any()):
        return g.copy()
    bg = _mode(g[interior_mask])

    out = g.copy()
    for r_idx in range(len(line_rows) - 1):
        r0 = int(line_rows[r_idx])
        r1 = int(line_rows[r_idx + 1])
        if r1 - r0 <= 1:
            continue
        for c_idx in range(len(line_cols) - 1):
            c0 = int(line_cols[c_idx])
            c1 = int(line_cols[c_idx + 1])
            if c1 - c0 <= 1:
                continue
            marker = _marker_info(g, r0, r1, c0, c1, wall)
            if marker is None:
                continue
            side, fill_color = marker
            patch = g[r0 + 1 : r1, c0 + 1 : c1]
            out[r0 + 1 : r1, c0 + 1 : c1] = _shift_patch(patch, bg, side, fill_color)

    out[line_rows, :] = wall
    out[:, line_cols] = wall
    return out


def _panel_sides(panel_r: int, panel_c: int, panel_rows: int, panel_cols: int) -> Tuple[str, ...]:
    sides: List[str] = []
    if panel_r == 0:
        sides.append("up")
    if panel_r == panel_rows - 1:
        sides.append("down")
    if panel_c == 0:
        sides.append("left")
    if panel_c == panel_cols - 1:
        sides.append("right")
    return tuple(sides)


def _outer_panels(panel_rows: int, panel_cols: int) -> List[Tuple[int, int]]:
    panels: List[Tuple[int, int]] = []
    for panel_r in range(panel_rows):
        for panel_c in range(panel_cols):
            if panel_r in (0, panel_rows - 1) or panel_c in (0, panel_cols - 1):
                panels.append((panel_r, panel_c))
    return panels


def _balanced_directions(count: int, rng: random.Random) -> List[str]:
    directions: List[str] = []
    while len(directions) < count:
        cycle = list(DIRS)
        rng.shuffle(cycle)
        directions.extend(cycle)
    return directions[:count]


def _assign_markers(
    panel_rows: int, panel_cols: int, rng: random.Random
) -> Optional[Dict[Tuple[int, int], str]]:
    outer = _outer_panels(panel_rows, panel_cols)
    if len(outer) < 2:
        return None

    max_markers = min(len(outer) - 1, 6)
    mark_count = rng.randint(1, max_markers)
    directions = _balanced_directions(mark_count, rng)

    candidates: Dict[str, List[Tuple[int, int]]] = {
        direction: [panel for panel in outer if direction in _panel_sides(panel[0], panel[1], panel_rows, panel_cols)]
        for direction in DIRS
    }
    ordered_dirs = sorted(directions, key=lambda direction: len(candidates[direction]))
    assignment: Dict[Tuple[int, int], str] = {}
    used: set[Tuple[int, int]] = set()

    def backtrack(idx: int) -> bool:
        if idx == len(ordered_dirs):
            return True
        direction = ordered_dirs[idx]
        options = [panel for panel in candidates[direction] if panel not in used]
        rng.shuffle(options)
        for panel in options:
            used.add(panel)
            assignment[panel] = direction
            if backtrack(idx + 1):
                return True
            used.remove(panel)
            del assignment[panel]
        return False

    if not backtrack(0):
        return None
    return assignment


def _walk_shape(inner_h: int, inner_w: int, max_cells: int, rng: random.Random) -> np.ndarray:
    target = rng.randint(1, max_cells)
    cells = {(rng.randrange(inner_h), rng.randrange(inner_w))}
    while len(cells) < target:
        expandable = []
        for r, c in cells:
            neigh = []
            for dr, dc in ((1, 0), (-1, 0), (0, 1), (0, -1)):
                rr = r + dr
                cc = c + dc
                if 0 <= rr < inner_h and 0 <= cc < inner_w and (rr, cc) not in cells:
                    neigh.append((rr, cc))
            if neigh:
                expandable.append(((r, c), neigh))
        if not expandable:
            break
        (_, _), neigh = rng.choice(expandable)
        cells.add(rng.choice(neigh))

    mask = np.zeros((inner_h, inner_w), dtype=bool)
    for r, c in cells:
        mask[r, c] = True
    return mask


def _rect_shape(inner_h: int, inner_w: int, max_cells: int, rng: random.Random) -> np.ndarray:
    max_h = min(inner_h, max_cells)
    rect_h = rng.randint(1, max_h)
    max_w = min(inner_w, max(1, max_cells // rect_h))
    rect_w = rng.randint(1, max_w)
    top = rng.randint(0, inner_h - rect_h)
    left = rng.randint(0, inner_w - rect_w)
    mask = np.zeros((inner_h, inner_w), dtype=bool)
    mask[top : top + rect_h, left : left + rect_w] = True
    return mask


def _l_shape(inner_h: int, inner_w: int, max_cells: int, rng: random.Random) -> np.ndarray:
    if inner_h == 1 or inner_w == 1:
        return _walk_shape(inner_h, inner_w, max_cells, rng)

    arm_h = rng.randint(2, inner_h)
    arm_w = rng.randint(2, inner_w)
    if arm_h + arm_w - 1 > max_cells:
        return _walk_shape(inner_h, inner_w, max_cells, rng)

    top = rng.randint(0, inner_h - arm_h)
    left = rng.randint(0, inner_w - arm_w)
    mask = np.zeros((inner_h, inner_w), dtype=bool)
    variant = rng.choice((0, 1, 2, 3))
    if variant == 0:
        mask[top : top + arm_h, left] = True
        mask[top + arm_h - 1, left : left + arm_w] = True
    elif variant == 1:
        mask[top : top + arm_h, left + arm_w - 1] = True
        mask[top + arm_h - 1, left : left + arm_w] = True
    elif variant == 2:
        mask[top, left : left + arm_w] = True
        mask[top : top + arm_h, left] = True
    else:
        mask[top, left : left + arm_w] = True
        mask[top : top + arm_h, left + arm_w - 1] = True
    return mask


def _make_inner_shape(inner_h: int, inner_w: int, rng: random.Random) -> np.ndarray:
    area = inner_h * inner_w
    max_cells = max(1, min(area, max(2, area // 2 + 1)))
    kind = rng.choice(("walk", "rect", "l"))
    if kind == "rect":
        return _rect_shape(inner_h, inner_w, max_cells, rng)
    if kind == "l":
        return _l_shape(inner_h, inner_w, max_cells, rng)
    return _walk_shape(inner_h, inner_w, max_cells, rng)


def _panel_patch(height: int, width: int, color: int, rng: random.Random) -> np.ndarray:
    patch = np.full((height, width), BG, dtype=int)
    inner = patch[1 : height - 1, 1 : width - 1]
    shape = _make_inner_shape(height - 2, width - 2, rng)
    inner[shape] = color
    return patch


def _gen_attempt(rng: random.Random) -> Optional[np.ndarray]:
    panel_rows = rng.randint(2, 4)
    panel_cols = rng.randint(2, 4)
    panel_h_max = min(6, (30 - (panel_rows + 1)) // panel_rows)
    panel_w_max = min(6, (30 - (panel_cols + 1)) // panel_cols)
    if panel_h_max < 3 or panel_w_max < 3:
        return None

    panel_h = rng.randint(3, panel_h_max)
    panel_w = rng.randint(3, panel_w_max)
    height = panel_rows * panel_h + (panel_rows + 1)
    width = panel_cols * panel_w + (panel_cols + 1)
    if height > 30 or width > 30:
        return None

    markers = _assign_markers(panel_rows, panel_cols, rng)
    if markers is None:
        return None

    grid = np.full((height, width), BG, dtype=int)
    line_rows = [idx * (panel_h + 1) for idx in range(panel_rows + 1)]
    line_cols = [idx * (panel_w + 1) for idx in range(panel_cols + 1)]
    grid[line_rows, :] = WALL
    grid[:, line_cols] = WALL

    for panel_r in range(panel_rows):
        for panel_c in range(panel_cols):
            color = rng.choice(PALETTE)
            patch = _panel_patch(panel_h, panel_w, color, rng)
            r0 = line_rows[panel_r] + 1
            c0 = line_cols[panel_c] + 1
            grid[r0 : r0 + panel_h, c0 : c0 + panel_w] = patch

    for (panel_r, panel_c), side in markers.items():
        r0 = line_rows[panel_r]
        r1 = line_rows[panel_r + 1]
        c0 = line_cols[panel_c]
        c1 = line_cols[panel_c + 1]
        panel_patch = grid[r0 + 1 : r1, c0 + 1 : c1]
        figure_colors = [color for color in np.unique(panel_patch) if color != BG]
        if len(figure_colors) != 1:
            return None
        figure_color = int(figure_colors[0])
        marker_choices = [color for color in PALETTE if color != figure_color]
        marker_color = rng.choice(marker_choices)
        if side == "up":
            grid[r0, rng.randint(c0 + 1, c1 - 1)] = marker_color
        elif side == "down":
            grid[r1, rng.randint(c0 + 1, c1 - 1)] = marker_color
        elif side == "left":
            grid[rng.randint(r0 + 1, r1 - 1), c0] = marker_color
        else:
            grid[rng.randint(r0 + 1, r1 - 1), c1] = marker_color

    interior_mask = np.ones((height, width), dtype=bool)
    interior_mask[line_rows, :] = False
    interior_mask[:, line_cols] = False
    if _mode(grid[interior_mask]) != BG:
        return None

    try:
        output = main(grid)
    except Exception:
        return None
    if np.array_equal(grid, output):
        return None
    return grid


def generate_input() -> np.ndarray:
    rng = random.Random()
    for _ in range(GEN_MAX_ATTEMPTS):
        grid = _gen_attempt(rng)
        if grid is not None:
            return grid
    raise RuntimeError("generate_input failed to produce a valid panel-push puzzle")
