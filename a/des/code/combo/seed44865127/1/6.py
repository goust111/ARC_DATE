from __future__ import annotations

import random
from collections import Counter

import numpy as np


TOP_COLOR = 2      # red
BOTTOM_COLOR = 1   # blue
LEFT_COLOR = 3     # green
RIGHT_COLOR = 4    # yellow

RAY_COLORS = {
    "top": TOP_COLOR,
    "bottom": BOTTOM_COLOR,
    "left": LEFT_COLOR,
    "right": RIGHT_COLOR,
}
RAY_DIRS = {
    "top": (-1, 0),
    "bottom": (1, 0),
    "left": (0, -1),
    "right": (0, 1),
}
DIR4 = [(-1, 0), (1, 0), (0, -1), (0, 1)]


def _dominant_color(grid: np.ndarray) -> int:
    values, counts = np.unique(grid, return_counts=True)
    return int(values[np.argmax(counts)])


def _components_by_color(grid: np.ndarray, bg: int) -> list[dict]:
    h, w = map(int, grid.shape)
    visited = np.zeros((h, w), dtype=bool)
    components: list[dict] = []

    for r in range(h):
        for c in range(w):
            if visited[r, c] or int(grid[r, c]) == bg:
                continue
            color = int(grid[r, c])
            stack = [(r, c)]
            visited[r, c] = True
            coords: list[tuple[int, int]] = []

            while stack:
                cr, cc = stack.pop()
                coords.append((int(cr), int(cc)))
                for dr, dc in DIR4:
                    nr, nc = cr + dr, cc + dc
                    if not (0 <= nr < h and 0 <= nc < w):
                        continue
                    if visited[nr, nc] or int(grid[nr, nc]) != color:
                        continue
                    visited[nr, nc] = True
                    stack.append((nr, nc))

            arr = np.array(coords, dtype=int)
            components.append(
                {
                    "color": color,
                    "coords": arr,
                    "coord_set": set(coords),
                    "bbox": (
                        int(arr[:, 0].min()),
                        int(arr[:, 1].min()),
                        int(arr[:, 0].max()),
                        int(arr[:, 1].max()),
                    ),
                }
            )

    return components


def _border_coords(r0: int, c0: int, r1: int, c1: int) -> set[tuple[int, int]]:
    border: set[tuple[int, int]] = set()
    for c in range(c0, c1 + 1):
        border.add((int(r0), int(c)))
        border.add((int(r1), int(c)))
    for r in range(r0, r1 + 1):
        border.add((int(r), int(c0)))
        border.add((int(r), int(c1)))
    return border


def _frame_info(component: dict) -> dict:
    r0, c0, r1, c1 = component["bbox"]
    if r1 - r0 < 2 or c1 - c0 < 2:
        raise AssertionError("frame bbox must be at least 3x3")

    border = _border_coords(r0, c0, r1, c1)
    coords = component["coord_set"]
    if not coords.issubset(border):
        raise AssertionError("frame component must lie on its bbox border")

    missing = border - coords
    if len(missing) != 1 or len(coords) != len(border) - 1:
        raise AssertionError("expected a one-pixel border gap")

    gap = next(iter(missing))
    gr, gc = gap
    side_matches = []
    if gr == r0:
        side_matches.append("top")
    if gr == r1:
        side_matches.append("bottom")
    if gc == c0:
        side_matches.append("left")
    if gc == c1:
        side_matches.append("right")
    if len(side_matches) != 1:
        raise AssertionError("gap must lie on exactly one side, not a corner")

    side = side_matches[0]
    if side in ("top", "bottom") and not (c0 < gc < c1):
        raise AssertionError("horizontal-side gap cannot be on a corner")
    if side in ("left", "right") and not (r0 < gr < r1):
        raise AssertionError("vertical-side gap cannot be on a corner")

    return {
        "color": int(component["color"]),
        "bbox": (int(r0), int(c0), int(r1), int(c1)),
        "gap": (int(gr), int(gc)),
        "side": side,
        "perimeter": int(len(border)),
    }


def _ray_cells(grid: np.ndarray, bg: int, frame: dict) -> list[tuple[int, int]]:
    h, w = map(int, grid.shape)
    r, c = frame["gap"]
    dr, dc = RAY_DIRS[frame["side"]]
    cells: list[tuple[int, int]] = []

    for _ in range(int(frame["perimeter"])):
        if not (0 <= r < h and 0 <= c < w):
            break
        if int(grid[r, c]) != bg:
            break
        cells.append((int(r), int(c)))
        r += dr
        c += dc

    return cells


def main(grid: np.ndarray) -> np.ndarray:
    """
    Cast a colored ray outward from each frame's single border gap.

    Input structure:
    - The background is the dominant color.
    - Foreground objects are one-pixel-thick hollow rectangular frames.
    - Each frame has exactly one single-cell gap on one side.

    Rule:
    - Recover each frame's missing border cell and the side that contains it.
    - The ray direction is the outward normal of that side.
    - The ray length equals the complete rectangular border perimeter.
    - Gap on top/bottom/left/right maps to red/blue/green/yellow respectively.
    - Paint only background cells; stop at a non-background cell or the grid edge.
    - Preserve every original frame pixel.
    """
    g = np.asarray(grid, dtype=int)
    assert g.ndim == 2, "expected a 2D grid"

    bg = _dominant_color(g)
    components = _components_by_color(g, bg)
    assert components, "expected at least one frame"
    frames = [_frame_info(component) for component in components]

    out = g.copy()
    for frame in frames:
        ray_color = RAY_COLORS[frame["side"]]
        for r, c in _ray_cells(g, bg, frame):
            out[r, c] = int(ray_color)

    return out


# --------------- generator ---------------

GRID_MIN = 12
GRID_MAX = 24
FRAME_COUNT_RANGE = (1, 4)
FRAME_HEIGHT_RANGE = (3, 8)
FRAME_WIDTH_RANGE = (3, 8)
FRAME_COLORS = (0, 5, 6, 7, 8, 9)
SIDES = ("top", "bottom", "left", "right")
GEN_ATTEMPTS = 400


def _rectangles_too_close(a: tuple[int, int, int, int], b: tuple[int, int, int, int]) -> bool:
    ar0, ac0, ar1, ac1 = a
    br0, bc0, br1, bc1 = b
    return not (ar1 + 1 < br0 or br1 + 1 < ar0 or ac1 + 1 < bc0 or bc1 + 1 < ac0)


def _choose_gap(r0: int, c0: int, r1: int, c1: int, side: str) -> tuple[int, int]:
    if side == "top":
        return int(r0), random.randint(c0 + 1, c1 - 1)
    if side == "bottom":
        return int(r1), random.randint(c0 + 1, c1 - 1)
    if side == "left":
        return random.randint(r0 + 1, r1 - 1), int(c0)
    return random.randint(r0 + 1, r1 - 1), int(c1)


def _frame_pixels(frame: dict) -> set[tuple[int, int]]:
    r0, c0, r1, c1 = frame["bbox"]
    gap = frame["gap"]
    return _border_coords(r0, c0, r1, c1) - {gap}


def _sample_frame_bbox(h: int, w: int) -> tuple[int, int, int, int]:
    fh = random.randint(FRAME_HEIGHT_RANGE[0], min(FRAME_HEIGHT_RANGE[1], h - 2))
    fw = random.randint(FRAME_WIDTH_RANGE[0], min(FRAME_WIDTH_RANGE[1], w - 2))
    r0 = random.randint(1, h - fh - 1)
    c0 = random.randint(1, w - fw - 1)
    return int(r0), int(c0), int(r0 + fh - 1), int(c0 + fw - 1)


def _sample_layout() -> tuple[np.ndarray, list[dict]] | None:
    frame_count = random.randint(*FRAME_COUNT_RANGE)
    min_size = max(GRID_MIN, 10 + 2 * frame_count)
    h = random.randint(min_size, GRID_MAX)
    w = random.randint(min_size, GRID_MAX)

    bg = random.choice(FRAME_COLORS)
    usable_frame_colors = [color for color in FRAME_COLORS if color != bg]
    if len(usable_frame_colors) < frame_count:
        return None
    colors = random.sample(usable_frame_colors, frame_count)
    sides = random.sample(SIDES, frame_count)

    frames: list[dict] = []
    for color, side in zip(colors, sides):
        placed = False
        for _ in range(120):
            bbox = _sample_frame_bbox(h, w)
            if any(_rectangles_too_close(bbox, other["bbox"]) for other in frames):
                continue
            r0, c0, r1, c1 = bbox
            gap = _choose_gap(r0, c0, r1, c1, side)
            frames.append(
                {
                    "color": int(color),
                    "bbox": bbox,
                    "gap": gap,
                    "side": side,
                    "perimeter": int(2 * ((r1 - r0 + 1) + (c1 - c0 + 1)) - 4),
                }
            )
            placed = True
            break
        if not placed:
            return None

    grid = np.full((h, w), int(bg), dtype=int)
    for frame in frames:
        for r, c in _frame_pixels(frame):
            grid[r, c] = int(frame["color"])

    return grid, frames


def _paths_are_clear(grid: np.ndarray, bg: int, frames: list[dict]) -> bool:
    path_sets: list[set[tuple[int, int]]] = []
    for i, frame in enumerate(frames):
        cells = set(_ray_cells(grid, bg, frame))
        others = [other for j, other in enumerate(frames) if j != i]
        for r, c in cells:
            for other in others:
                r0, c0, r1, c1 = other["bbox"]
                if r0 <= r <= r1 and c0 <= c <= c1:
                    return False
        path_sets.append(cells)

    for i in range(len(path_sets)):
        for j in range(i + 1, len(path_sets)):
            if path_sets[i] & path_sets[j]:
                return False

    return True


def _input_is_valid(grid: np.ndarray, frames: list[dict]) -> bool:
    bg = _dominant_color(grid)
    if Counter(map(int, grid.ravel())).most_common(1)[0][0] != bg:
        return False
    if not _paths_are_clear(grid, bg, frames):
        return False

    try:
        parsed = [_frame_info(component) for component in _components_by_color(grid, bg)]
        if len(parsed) != len(frames):
            return False
        out = main(grid)
    except AssertionError:
        return False

    if np.array_equal(out, grid):
        return False
    return True


def generate_input() -> np.ndarray:
    for _ in range(GEN_ATTEMPTS):
        sampled = _sample_layout()
        if sampled is None:
            continue
        grid, frames = sampled
        if _input_is_valid(grid, frames):
            return grid
    raise RuntimeError("failed to generate a valid frame-gap ray puzzle input")
