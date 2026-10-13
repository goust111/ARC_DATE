import random
from collections import deque

import numpy as np


BLACK = 0
AZURE = 8
MAROON = 9
PATTERN_COLORS = tuple(color for color in range(10) if color not in {BLACK, AZURE, MAROON})
WINDOW_SIDES = ("top", "bottom", "left", "right")
RNG = random.Random()

PATTERN_HEIGHT_RANGE = (4, 7)
PATTERN_WIDTH_RANGE = (4, 7)
COLOR_COUNT_CHOICES = (2, 3, 4)
SHAPE_TYPES = ("rect", "hline", "vline", "lshape")
SHAPES_PER_PATTERN_RANGE = (4, 9)
HOLE_COUNT_RANGE = (1, 3)
MAX_GENERATION_ATTEMPTS = 400
MAX_PATTERN_ATTEMPTS = 120
MAX_HOLE_ATTEMPTS = 120


def _neighbors4(r: int, c: int) -> list[tuple[int, int]]:
    return [(r - 1, c), (r + 1, c), (r, c - 1), (r, c + 1)]


def _components_of_color(grid: np.ndarray, color: int) -> list[list[tuple[int, int]]]:
    h, w = grid.shape
    seen = np.zeros((h, w), dtype=bool)
    components: list[list[tuple[int, int]]] = []
    for r in range(h):
        for c in range(w):
            if seen[r, c] or int(grid[r, c]) != color:
                continue
            q = deque([(r, c)])
            seen[r, c] = True
            cells: list[tuple[int, int]] = []
            while q:
                cr, cc = q.popleft()
                cells.append((int(cr), int(cc)))
                for nr, nc in _neighbors4(cr, cc):
                    if 0 <= nr < h and 0 <= nc < w and not seen[nr, nc] and int(grid[nr, nc]) == color:
                        seen[nr, nc] = True
                        q.append((nr, nc))
            components.append(cells)
    return components


def _bbox(cells: list[tuple[int, int]]) -> tuple[int, int, int, int]:
    rows = [r for r, _ in cells]
    cols = [c for _, c in cells]
    return min(rows), min(cols), max(rows), max(cols)


def _outline_cells(r0: int, c0: int, r1: int, c1: int) -> set[tuple[int, int]]:
    cells: set[tuple[int, int]] = set()
    for c in range(c0, c1 + 1):
        cells.add((r0, c))
        cells.add((r1, c))
    for r in range(r0, r1 + 1):
        cells.add((r, c0))
        cells.add((r, c1))
    return cells


def _parse_window(grid: np.ndarray, component: list[tuple[int, int]]) -> dict | None:
    r0, c0, r1, c1 = _bbox(component)
    if r1 - r0 < 2 or c1 - c0 < 2:
        return None

    outline = _outline_cells(r0, c0, r1, c1)
    maroon_cells = set(component)
    if maroon_cells - outline:
        return None

    missing = [(r, c) for r, c in outline if int(grid[r, c]) != MAROON]
    if len(missing) != 1:
        return None
    label_r, label_c = missing[0]
    label_color = int(grid[label_r, label_c])
    if label_color in {BLACK, AZURE, MAROON}:
        return None
    if maroon_cells != outline - {(label_r, label_c)}:
        return None

    inner = grid[r0 + 1 : r1, c0 + 1 : c1]
    return {
        "bbox": (r0, c0, r1, c1),
        "inner_shape": (r1 - r0 - 1, c1 - c0 - 1),
        "label_color": label_color,
        "inner": inner,
    }


def _find_windows(grid: np.ndarray) -> list[dict]:
    windows: list[dict] = []
    for component in _components_of_color(grid, MAROON):
        window = _parse_window(grid, component)
        if window is not None:
            windows.append(window)
    assert windows, "expected at least one filter window"

    inner_shapes = {window["inner_shape"] for window in windows}
    assert len(inner_shapes) == 1, "all filter windows must share one inner size"

    labels = [int(window["label_color"]) for window in windows]
    assert len(labels) == len(set(labels)), "filter window colors must be unique"
    return windows


def _find_pattern_bbox(grid: np.ndarray, pattern_shape: tuple[int, int], windows: list[dict]) -> tuple[int, int, int, int]:
    ph, pw = pattern_shape
    h, w = grid.shape
    blocked = np.zeros((h, w), dtype=bool)
    for window in windows:
        r0, c0, r1, c1 = window["bbox"]
        blocked[r0 : r1 + 1, c0 : c1 + 1] = True

    candidates: list[tuple[int, int, int, int]] = []
    for r0 in range(h - ph + 1):
        for c0 in range(w - pw + 1):
            patch = grid[r0 : r0 + ph, c0 : c0 + pw]
            if np.any(blocked[r0 : r0 + ph, c0 : c0 + pw]):
                continue
            if np.any(patch == BLACK):
                continue

            r1 = r0 + ph - 1
            c1 = c0 + pw - 1
            if r0 > 0 and np.all(grid[r0 - 1, c0 : c1 + 1] != BLACK):
                continue
            if r1 + 1 < h and np.all(grid[r1 + 1, c0 : c1 + 1] != BLACK):
                continue
            if c0 > 0 and np.all(grid[r0 : r1 + 1, c0 - 1] != BLACK):
                continue
            if c1 + 1 < w and np.all(grid[r0 : r1 + 1, c1 + 1] != BLACK):
                continue
            candidates.append((r0, c0, r1, c1))

    assert len(candidates) == 1, "expected a unique large pattern rectangle"
    return candidates[0]


def _reconstruct_pattern(grid: np.ndarray, windows: list[dict], pattern_shape: tuple[int, int]) -> np.ndarray:
    ph, pw = pattern_shape
    pattern = np.full((ph, pw), AZURE, dtype=int)
    for window in windows:
        label_color = int(window["label_color"])
        inner = np.asarray(window["inner"], dtype=int)
        mask = inner != BLACK
        conflict = mask & (pattern != AZURE) & (pattern != label_color)
        assert not np.any(conflict), "filter windows must not conflict"
        pattern[mask] = label_color
    return pattern


def main(grid: np.ndarray) -> np.ndarray:
    """
    Repair the erased cells of a central pattern by overlaying color-separated filter windows.

    Input structure:
    - The background is black.
    - One large solid rectangle sits away from the border; its cells are all non-black, with azure used as the pattern's internal background and other colors forming the visible motif.
    - Several smaller border-adjacent windows are framed by a one-cell maroon outline with exactly one non-maroon border pixel acting as a color label.
    - Each window's interior has the same height and width as the large rectangle and shows only one pattern color on black.

    Rule:
    - Find every maroon-framed window and read its label color from the unique non-maroon border cell.
    - The window interiors are aligned copies of the large pattern rectangle, but each one keeps only a single color and turns every other position black.
    - Reconstruct the complete pattern by overlaying those single-color interiors: a non-black window pixel restores its label color at the matching position, while positions that stay black in every window remain azure.
    - Replace the large rectangle with this reconstructed pattern and leave the rest of the grid unchanged.

    Key insight:
    - The side windows are color channels of the same hidden image; once their shared geometry is matched to the large rectangle, repairing the erased area is just channel recomposition.
    """
    g = np.asarray(grid, dtype=int)
    windows = _find_windows(g)
    pattern_shape = windows[0]["inner_shape"]
    pattern_bbox = _find_pattern_bbox(g, pattern_shape, windows)
    reconstructed = _reconstruct_pattern(g, windows, pattern_shape)

    out = g.copy()
    r0, c0, r1, c1 = pattern_bbox
    original = out[r0 : r1 + 1, c0 : c1 + 1]
    stable = original != AZURE
    assert np.all(reconstructed[stable] == original[stable]), "visible pattern pixels must already agree with the filters"
    out[r0 : r1 + 1, c0 : c1 + 1] = reconstructed
    return out


def _shape_cells(shape_type: str, ph: int, pw: int, rng: random.Random) -> list[tuple[int, int]]:
    if shape_type == "rect":
        height = rng.randint(1, max(1, min(3, ph)))
        width = rng.randint(1, max(1, min(3, pw)))
        r0 = rng.randint(0, ph - height)
        c0 = rng.randint(0, pw - width)
        return [(r, c) for r in range(r0, r0 + height) for c in range(c0, c0 + width)]

    if shape_type == "hline":
        row = rng.randrange(ph)
        width = rng.randint(2, pw)
        c0 = rng.randint(0, pw - width)
        return [(row, c) for c in range(c0, c0 + width)]

    if shape_type == "vline":
        col = rng.randrange(pw)
        height = rng.randint(2, ph)
        r0 = rng.randint(0, ph - height)
        return [(r, col) for r in range(r0, r0 + height)]

    corner_r = rng.randrange(ph)
    corner_c = rng.randrange(pw)
    vert_len = rng.randint(2, ph)
    hori_len = rng.randint(2, pw)
    if corner_r + vert_len > ph:
        corner_r = ph - vert_len
    if corner_c + hori_len > pw:
        corner_c = pw - hori_len
    cells = {(corner_r + dr, corner_c) for dr in range(vert_len)}
    cells |= {(corner_r, corner_c + dc) for dc in range(hori_len)}
    return sorted(cells)


def _make_pattern(ph: int, pw: int, colors: list[int], rng: random.Random) -> np.ndarray:
    for _ in range(MAX_PATTERN_ATTEMPTS):
        pattern = np.full((ph, pw), AZURE, dtype=int)
        stroke_plan = list(colors)
        extra_count = rng.randint(SHAPES_PER_PATTERN_RANGE[0], SHAPES_PER_PATTERN_RANGE[1])
        while len(stroke_plan) < extra_count:
            stroke_plan.append(rng.choice(colors))
        rng.shuffle(stroke_plan)

        for color in stroke_plan:
            shape_type = rng.choice(SHAPE_TYPES)
            cells = _shape_cells(shape_type, ph, pw, rng)
            for r, c in cells:
                pattern[r, c] = color

        if any(not np.any(pattern == color) for color in colors):
            continue
        azure_count = int(np.sum(pattern == AZURE))
        if azure_count == 0 or azure_count == ph * pw:
            continue
        return pattern

    raise RuntimeError("failed to generate a non-degenerate full pattern")


def _choose_holes(pattern: np.ndarray, rng: random.Random) -> np.ndarray:
    ph, pw = pattern.shape
    hole_mask = np.zeros((ph, pw), dtype=bool)
    hole_target = rng.randint(HOLE_COUNT_RANGE[0], HOLE_COUNT_RANGE[1])
    holes_made = 0
    attempts = 0
    while holes_made < hole_target and attempts < MAX_HOLE_ATTEMPTS:
        attempts += 1
        height = rng.randint(1, max(1, min(3, ph)))
        width = rng.randint(1, max(1, min(3, pw)))
        r0 = rng.randint(0, ph - height)
        c0 = rng.randint(0, pw - width)
        rect = hole_mask[r0 : r0 + height, c0 : c0 + width]
        if np.any(rect):
            continue
        patch = pattern[r0 : r0 + height, c0 : c0 + width]
        if not np.any(patch != AZURE):
            continue
        hole_mask[r0 : r0 + height, c0 : c0 + width] = True
        holes_made += 1

    if holes_made == 0:
        raise RuntimeError("failed to place any repair hole")
    return hole_mask


def _make_window(pattern: np.ndarray, color: int, rng: random.Random) -> np.ndarray:
    ph, pw = pattern.shape
    window = np.zeros((ph + 2, pw + 2), dtype=int)
    window[0, :] = MAROON
    window[-1, :] = MAROON
    window[:, 0] = MAROON
    window[:, -1] = MAROON
    window[1:-1, 1:-1] = np.where(pattern == color, color, BLACK)

    label_side = rng.choice(WINDOW_SIDES)
    if label_side == "top":
        label_r, label_c = 0, rng.randint(1, pw)
    elif label_side == "bottom":
        label_r, label_c = ph + 1, rng.randint(1, pw)
    elif label_side == "left":
        label_r, label_c = rng.randint(1, ph), 0
    else:
        label_r, label_c = rng.randint(1, ph), pw + 1
    window[label_r, label_c] = color
    return window


def _gen_attempt(rng: random.Random) -> np.ndarray | None:
    ph = rng.randint(PATTERN_HEIGHT_RANGE[0], PATTERN_HEIGHT_RANGE[1])
    pw = rng.randint(PATTERN_WIDTH_RANGE[0], PATTERN_WIDTH_RANGE[1])
    color_count = rng.choice(COLOR_COUNT_CHOICES)
    colors = rng.sample(list(PATTERN_COLORS), color_count)
    side_assignment = dict(zip(rng.sample(list(WINDOW_SIDES), color_count), colors))

    wh, ww = ph + 2, pw + 2
    gap = rng.randint(1, 2)
    top_reserved = wh + gap if "top" in side_assignment else 1
    bottom_reserved = wh + gap if "bottom" in side_assignment else 1
    left_reserved = ww + gap if "left" in side_assignment else 1
    right_reserved = ww + gap if "right" in side_assignment else 1

    min_h = top_reserved + ph + bottom_reserved
    min_w = left_reserved + pw + right_reserved
    if min_h > 30 or min_w > 30:
        return None

    grid_h = rng.randint(min_h, min(30, min_h + 4))
    grid_w = rng.randint(min_w, min(30, min_w + 4))

    pattern_r0 = rng.randint(top_reserved, grid_h - bottom_reserved - ph)
    pattern_c0 = rng.randint(left_reserved, grid_w - right_reserved - pw)

    window_positions: dict[str, tuple[int, int]] = {}
    if "top" in side_assignment:
        min_c = ww if "left" in side_assignment else 0
        max_c = grid_w - ww - (ww if "right" in side_assignment else 0)
        if min_c > max_c:
            return None
        window_positions["top"] = (0, rng.randint(min_c, max_c))
    if "bottom" in side_assignment:
        min_c = ww if "left" in side_assignment else 0
        max_c = grid_w - ww - (ww if "right" in side_assignment else 0)
        if min_c > max_c:
            return None
        window_positions["bottom"] = (grid_h - wh, rng.randint(min_c, max_c))
    if "left" in side_assignment:
        min_r = wh if "top" in side_assignment else 0
        max_r = grid_h - wh - (wh if "bottom" in side_assignment else 0)
        if min_r > max_r:
            return None
        window_positions["left"] = (rng.randint(min_r, max_r), 0)
    if "right" in side_assignment:
        min_r = wh if "top" in side_assignment else 0
        max_r = grid_h - wh - (wh if "bottom" in side_assignment else 0)
        if min_r > max_r:
            return None
        window_positions["right"] = (rng.randint(min_r, max_r), grid_w - ww)

    occupied = np.zeros((grid_h, grid_w), dtype=bool)
    occupied[pattern_r0 : pattern_r0 + ph, pattern_c0 : pattern_c0 + pw] = True
    for side, (r0, c0) in window_positions.items():
        if np.any(occupied[r0 : r0 + wh, c0 : c0 + ww]):
            return None
        occupied[r0 : r0 + wh, c0 : c0 + ww] = True

    full_pattern = _make_pattern(ph, pw, colors, rng)
    hole_mask = _choose_holes(full_pattern, rng)
    damaged = full_pattern.copy()
    damaged[hole_mask] = AZURE
    if np.array_equal(damaged, full_pattern):
        return None

    grid = np.zeros((grid_h, grid_w), dtype=int)
    grid[pattern_r0 : pattern_r0 + ph, pattern_c0 : pattern_c0 + pw] = damaged
    for side, color in side_assignment.items():
        window = _make_window(full_pattern, color, rng)
        wr0, wc0 = window_positions[side]
        grid[wr0 : wr0 + wh, wc0 : wc0 + ww] = window

    solved = main(grid)
    if not np.array_equal(solved[pattern_r0 : pattern_r0 + ph, pattern_c0 : pattern_c0 + pw], full_pattern):
        return None
    return grid


def generate_input() -> np.ndarray:
    for _ in range(MAX_GENERATION_ATTEMPTS):
        grid = _gen_attempt(RNG)
        if grid is not None:
            return grid
    raise RuntimeError("failed to generate a valid filter-window puzzle")
