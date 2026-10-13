from __future__ import annotations

import random
from typing import Iterable

import numpy as np


BLACK = 0
BLUE = 1
RED = 2
GREEN = 3
YELLOW = 4
ORANGE = 7
MAROON = 9

CONTAINER_COLORS = (BLUE, GREEN, YELLOW, RED)
DEPTH_COLORS = {
    2: (BLUE, RED),
    3: (BLUE, GREEN, RED),
    4: (BLUE, GREEN, YELLOW, RED),
}
ARROW_TEMPLATES = {
    "up": {(0, 1), (1, 1), (2, 0), (2, 1), (2, 2)},
    "down": {(0, 0), (0, 1), (0, 2), (1, 1), (2, 1)},
    "left": {(1, 0), (1, 1), (0, 2), (1, 2), (2, 2)},
    "right": {(0, 0), (1, 0), (2, 0), (1, 1), (1, 2)},
}
BASE_SHAPES = (
    {(0, 0), (1, 0), (1, 1)},
    {(0, 0), (1, 0), (2, 0), (2, 1)},
    {(0, 1), (1, 0), (1, 1), (1, 2)},
    {(0, 1), (0, 2), (1, 0), (1, 1)},
    {(0, 0), (1, 0), (1, 1), (2, 1), (2, 2)},
    {(0, 0), (0, 1), (1, 1), (2, 1), (2, 2)},
    {(0, 1), (1, 0), (1, 1), (1, 2), (2, 1)},
)
GEN_RETRIES = 300


def _neighbors4(r: int, c: int) -> Iterable[tuple[int, int]]:
    yield r - 1, c
    yield r + 1, c
    yield r, c - 1
    yield r, c + 1


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
                for nr, nc in _neighbors4(cr, cc):
                    if 0 <= nr < h and 0 <= nc < w and mask[nr, nc] and not seen[nr, nc]:
                        seen[nr, nc] = True
                        stack.append((nr, nc))
            comps.append(comp)
    return comps


def _bbox(points: Iterable[tuple[int, int]]) -> tuple[int, int, int, int]:
    rows = [r for r, _ in points]
    cols = [c for _, c in points]
    return min(rows), min(cols), max(rows), max(cols)


def _bbox_area(box: tuple[int, int, int, int]) -> int:
    r0, c0, r1, c1 = box
    return (r1 - r0 + 1) * (c1 - c0 + 1)


def _shift_bbox(box: tuple[int, int, int, int], dr: int, dc: int) -> tuple[int, int, int, int]:
    r0, c0, r1, c1 = box
    return r0 + dr, c0 + dc, r1 + dr, c1 + dc


def _draw_frame(grid: np.ndarray, box: tuple[int, int, int, int], color: int) -> None:
    r0, c0, r1, c1 = box
    grid[r0, c0 : c1 + 1] = color
    grid[r1, c0 : c1 + 1] = color
    grid[r0 : r1 + 1, c0] = color
    grid[r0 : r1 + 1, c1] = color


def _is_frame_component(grid: np.ndarray, comp: list[tuple[int, int]], color: int) -> bool:
    r0, c0, r1, c1 = _bbox(comp)
    if r1 - r0 < 2 or c1 - c0 < 2:
        return False
    expected = 2 * ((r1 - r0 + 1) + (c1 - c0 + 1)) - 4
    if len(comp) != expected:
        return False
    comp_set = set(comp)
    for r in range(r0, r1 + 1):
        for c in range(c0, c1 + 1):
            on_border = r in (r0, r1) or c in (c0, c1)
            if on_border != ((r, c) in comp_set):
                return False
            if on_border and int(grid[r, c]) != color:
                return False
    return True


def _find_frames(grid: np.ndarray) -> list[dict[str, object]]:
    frames: list[dict[str, object]] = []
    for color in CONTAINER_COLORS:
        for comp in _components(grid == color):
            if not _is_frame_component(grid, comp, color):
                continue
            frames.append({"color": color, "bbox": _bbox(comp)})
    return frames


def _find_arrow_direction(grid: np.ndarray) -> str:
    comps = _components(grid == MAROON)
    for comp in comps:
        if len(comp) != 5:
            continue
        r0, c0, r1, c1 = _bbox(comp)
        if (r1 - r0, c1 - c0) != (2, 2):
            continue
        norm = {(r - r0, c - c0) for r, c in comp}
        for name, template in ARROW_TEMPLATES.items():
            if norm == template:
                return name
    raise AssertionError("expected a unique 3x3 maroon arrow")


def _contains(parent: tuple[int, int, int, int], child: tuple[int, int, int, int]) -> bool:
    return (
        parent[0] < child[0]
        and parent[1] < child[1]
        and parent[2] > child[2]
        and parent[3] > child[3]
    )


def _point_inside(box: tuple[int, int, int, int], r: int, c: int) -> bool:
    return box[0] < r < box[2] and box[1] < c < box[3]


def main(grid: np.ndarray) -> np.ndarray:
    g = np.asarray(grid, dtype=int)
    assert g.ndim == 2

    direction = _find_arrow_direction(g)
    frames = _find_frames(g)
    assert frames

    orange_comps = _components(g == ORANGE)
    assert orange_comps

    pattern_cells: list[tuple[int, int]] | None = None
    red_frame: dict[str, object] | None = None
    for comp in orange_comps:
        candidates = [
            frame
            for frame in frames
            if frame["color"] == RED and all(_point_inside(frame["bbox"], r, c) for r, c in comp)
        ]
        if not candidates:
            continue
        red_frame = min(candidates, key=lambda frame: _bbox_area(frame["bbox"]))
        pattern_cells = comp
        break

    assert pattern_cells is not None and red_frame is not None

    chain = [red_frame]
    while True:
        current_box = chain[-1]["bbox"]
        parents = [frame for frame in frames if _contains(frame["bbox"], current_box)]
        if not parents:
            break
        parent = min(parents, key=lambda frame: _bbox_area(frame["bbox"]))
        chain.append(parent)

    assert len(chain) >= 2
    chain.reverse()

    out = g.copy()
    for r, c in pattern_cells:
        out[r, c] = BLACK

    current_cells = list(pattern_cells)
    current_color = ORANGE
    for idx in range(len(chain) - 1, 0, -1):
        current_frame = chain[idx]
        parent = chain[idx - 1]
        pr0, pc0, pr1, pc1 = _bbox(current_cells)
        r0, c0, r1, c1 = parent["bbox"]
        if direction == "up":
            dr, dc = r0 + 1 - pr0, 0
        elif direction == "down":
            dr, dc = r1 - 1 - pr1, 0
        elif direction == "left":
            dr, dc = 0, c0 + 1 - pc0
        else:
            dr, dc = 0, c1 - 1 - pc1
        current_cells = [(r + dr, c + dc) for r, c in current_cells]
        current_color = int(current_frame["color"])

    for r, c in current_cells:
        out[r, c] = current_color
    return out


def _normalize_shape(cells: Iterable[tuple[int, int]]) -> list[tuple[int, int]]:
    box = _bbox(cells)
    r0, c0, _, _ = box
    return sorted((r - r0, c - c0) for r, c in cells)


def _transform_shape(
    shape: Iterable[tuple[int, int]],
    rotations: int,
    flip: bool,
) -> list[tuple[int, int]]:
    pts = [(r, c) for r, c in shape]
    if flip:
        pts = [(r, -c) for r, c in pts]
    for _ in range(rotations % 4):
        pts = [(c, -r) for r, c in pts]
    return _normalize_shape(pts)


def _sample_shape(rng: random.Random) -> list[tuple[int, int]]:
    base = rng.choice(BASE_SHAPES)
    return _transform_shape(base, rng.randrange(4), rng.random() < 0.5)


def _shape_size(shape: Iterable[tuple[int, int]]) -> tuple[int, int]:
    r0, c0, r1, c1 = _bbox(shape)
    return r1 - r0 + 1, c1 - c0 + 1


def _build_local_layout(
    rng: random.Random,
    colors: tuple[int, ...],
    direction: str,
    shape: list[tuple[int, int]],
) -> tuple[list[dict[str, object]], list[tuple[int, int]], int, int]:
    shape_h, shape_w = _shape_size(shape)

    pad_top = rng.randint(1, 3)
    pad_bottom = rng.randint(1, 3)
    pad_left = rng.randint(1, 3)
    pad_right = rng.randint(1, 3)

    canvas_h = shape_h + pad_top + pad_bottom + 2
    canvas_w = shape_w + pad_left + pad_right + 2
    pattern = [(1 + pad_top + r, 1 + pad_left + c) for r, c in shape]
    frames = [{"color": colors[-1], "bbox": (0, 0, canvas_h - 1, canvas_w - 1)}]

    needed_span = shape_h if direction in ("up", "down") else shape_w
    for color in reversed(colors[:-1]):
        top_gap = rng.randint(1, 4)
        bottom_gap = rng.randint(1, 4)
        left_gap = rng.randint(1, 4)
        right_gap = rng.randint(1, 4)
        big_gap = rng.randint(needed_span, needed_span + 3)
        if direction == "up":
            top_gap = big_gap
        elif direction == "down":
            bottom_gap = big_gap
        elif direction == "left":
            left_gap = big_gap
        else:
            right_gap = big_gap

        shift_r = 1 + top_gap
        shift_c = 1 + left_gap
        pattern = [(r + shift_r, c + shift_c) for r, c in pattern]
        frames = [
            {"color": color, "bbox": (0, 0, canvas_h + top_gap + bottom_gap + 1, canvas_w + left_gap + right_gap + 1)}
        ] + [
            {"color": int(frame["color"]), "bbox": _shift_bbox(frame["bbox"], shift_r, shift_c)}
            for frame in frames
        ]
        canvas_h = canvas_h + top_gap + bottom_gap + 2
        canvas_w = canvas_w + left_gap + right_gap + 2

    return frames, pattern, canvas_h, canvas_w


def _place_arrow(grid: np.ndarray, top: int, left: int, direction: str) -> None:
    for dr, dc in ARROW_TEMPLATES[direction]:
        grid[top + dr, left + dc] = MAROON


def _gen_attempt(rng: random.Random) -> np.ndarray | None:
    depth = rng.choice((2, 3, 4))
    colors = DEPTH_COLORS[depth]
    direction = rng.choice(tuple(ARROW_TEMPLATES))
    shape = _sample_shape(rng)

    frames, pattern, local_h, local_w = _build_local_layout(rng, colors, direction, shape)

    top_margin = rng.randint(1, 4)
    left_margin = rng.randint(1, 4)
    right_margin = rng.randint(1, 4)
    gap_above_arrow = rng.randint(2, 5)

    grid_h = top_margin + local_h + gap_above_arrow + 3
    grid_w = left_margin + local_w + right_margin
    if grid_w < 5:
        return None

    grid = np.full((grid_h, grid_w), BLACK, dtype=int)
    for frame in frames:
        box = _shift_bbox(frame["bbox"], top_margin, left_margin)
        _draw_frame(grid, box, int(frame["color"]))

    for r, c in pattern:
        grid[top_margin + r, left_margin + c] = ORANGE

    arrow_top = grid_h - 3
    arrow_left = rng.randint(1, grid_w - 4)
    _place_arrow(grid, arrow_top, arrow_left, direction)

    try:
        out = main(grid)
    except AssertionError:
        return None
    if out.shape != grid.shape or np.array_equal(out, grid):
        return None
    if np.any(out == ORANGE):
        return None
    return grid


def generate_input() -> np.ndarray:
    rng = random.Random()
    for _ in range(GEN_RETRIES):
        grid = _gen_attempt(rng)
        if grid is not None:
            return grid
    raise RuntimeError("failed to generate valid nested-container input")
