from __future__ import annotations

from collections import deque
from dataclasses import dataclass

import numpy as np


BG = 0
ARROW_COLOR = 5
MAX_DIM = 30
DIR4 = [(-1, 0), (1, 0), (0, -1), (0, 1)]
PALETTE = [1, 2, 3, 4, 6, 7, 8, 9]
GENERATE_ATTEMPTS = 400

ARROW_RIGHT = np.array(
    [
        [1, 0, 0],
        [1, 1, 1],
        [1, 0, 0],
    ],
    dtype=bool,
)
ARROW_DOWN = np.array(
    [
        [1, 1, 1],
        [0, 1, 0],
        [0, 1, 0],
    ],
    dtype=bool,
)


@dataclass(frozen=True)
class Component:
    color: int
    coords: tuple[tuple[int, int], ...]
    bbox: tuple[int, int, int, int]
    area: int
    center: tuple[float, float]


def _find_components(grid: np.ndarray, bg: int = BG) -> list[Component]:
    h, w = grid.shape
    seen = np.zeros((h, w), dtype=bool)
    components: list[Component] = []

    for r in range(h):
        for c in range(w):
            if seen[r, c] or int(grid[r, c]) == bg:
                continue
            color = int(grid[r, c])
            queue = deque([(r, c)])
            seen[r, c] = True
            cells: list[tuple[int, int]] = []

            while queue:
                cr, cc = queue.popleft()
                cells.append((cr, cc))
                for dr, dc in DIR4:
                    nr = cr + dr
                    nc = cc + dc
                    if not (0 <= nr < h and 0 <= nc < w):
                        continue
                    if seen[nr, nc] or int(grid[nr, nc]) != color:
                        continue
                    seen[nr, nc] = True
                    queue.append((nr, nc))

            rows = [cell[0] for cell in cells]
            cols = [cell[1] for cell in cells]
            coords = tuple(sorted(cells))
            components.append(
                Component(
                    color=color,
                    coords=coords,
                    bbox=(min(rows), min(cols), max(rows), max(cols)),
                    area=len(coords),
                    center=((min(rows) + max(rows)) / 2.0, (min(cols) + max(cols)) / 2.0),
                )
            )
    return components


def _analyze_scene(grid: np.ndarray) -> tuple[Component, Component, list[Component], str]:
    g = np.asarray(grid, dtype=int)
    assert g.ndim == 2, "grid must be 2D"
    assert g.shape[0] <= MAX_DIM and g.shape[1] <= MAX_DIM, "grid must fit ARC size limits"

    components = _find_components(g, bg=BG)
    assert components, "expected foreground components"

    arrows = [comp for comp in components if comp.color == ARROW_COLOR and comp.area > 1]
    assert len(arrows) == 1, f"expected one gray arrow sprite, got {len(arrows)}"
    arrow = arrows[0]

    sources = [comp for comp in components if comp.color != ARROW_COLOR and comp.area == 1]
    assert len(sources) == 1, f"expected one source pixel, got {len(sources)}"
    source = sources[0]

    objects = [comp for comp in components if comp is not arrow and comp is not source]
    assert len(objects) >= 2, "expected at least two recolorable objects"
    assert all(comp.area > 1 for comp in objects), "objects must be multi-pixel"

    object_colors = [comp.color for comp in objects]
    assert len(set(object_colors)) == len(object_colors), "object colors must be unique"
    assert source.color not in object_colors, "source color must be distinct"

    row_span = max(comp.center[0] for comp in objects) - min(comp.center[0] for comp in objects)
    col_span = max(comp.center[1] for comp in objects) - min(comp.center[1] for comp in objects)
    axis = "horizontal" if col_span >= row_span else "vertical"

    if axis == "horizontal":
        ordered = sorted(objects, key=lambda comp: (comp.center[1], comp.center[0]))
        assert arrow.bbox[3] < ordered[0].bbox[1], "horizontal arrow must sit before the first object"
        assert source.coords[0][1] > ordered[-1].bbox[3], "source pixel must sit after the last object"
    else:
        ordered = sorted(objects, key=lambda comp: (comp.center[0], comp.center[1]))
        assert arrow.bbox[2] < ordered[0].bbox[0], "vertical arrow must sit above the first object"
        assert source.coords[0][0] > ordered[-1].bbox[2], "source pixel must sit below the last object"

    return arrow, source, ordered, axis


def main(grid: np.ndarray) -> np.ndarray:
    """
    Shift a row or column of object colors along the arrow direction, injecting the trailing source color.

    Input structure:
    - A black background contains several separated monochrome objects arranged in one horizontal row or one vertical column.
    - A single gray arrow sprite sits just outside the first object and points along the arrangement direction, which is either rightward or downward.
    - Beyond the last object there is one isolated source pixel whose color is different from every object color.

    Rule:
    - Read the objects in arrow order from the arrow end toward the source pixel.
    - Replace each object's color with the color of the next position along that order.
    - The last object receives the isolated source pixel's color, while the first object's original color disappears.
    - Object shapes and positions stay fixed, and both the gray arrow sprite and the isolated source pixel are removed in the output.

    Key insight:
    - The arrow defines a color-propagation order rather than physical motion, so the task is a one-step queue shift of colors through an unchanged object chain.
    """
    g = np.asarray(grid, dtype=int)
    arrow, source, objects, _axis = _analyze_scene(g)

    out = g.copy()
    for r, c in arrow.coords:
        out[r, c] = BG
    sr, sc = source.coords[0]
    out[sr, sc] = BG

    shifted_colors = [comp.color for comp in objects[1:]] + [source.color]
    for comp, new_color in zip(objects, shifted_colors):
        for r, c in comp.coords:
            out[r, c] = int(new_color)

    return out


def _crop_mask(mask: np.ndarray) -> np.ndarray:
    rows, cols = np.where(mask)
    return mask[rows.min() : rows.max() + 1, cols.min() : cols.max() + 1]


def _random_object_mask(rng: np.random.Generator) -> np.ndarray:
    for _ in range(100):
        raw_h = int(rng.integers(2, 5))
        raw_w = int(rng.integers(2, 5))
        target_area = int(rng.integers(3, raw_h * raw_w + 1))
        seed = (int(rng.integers(raw_h)), int(rng.integers(raw_w)))
        cells = {seed}

        while len(cells) < target_area:
            frontier: list[tuple[int, int]] = []
            for r, c in cells:
                for dr, dc in DIR4:
                    nr = r + dr
                    nc = c + dc
                    if 0 <= nr < raw_h and 0 <= nc < raw_w and (nr, nc) not in cells:
                        frontier.append((nr, nc))
            assert frontier, "connected-growth frontier unexpectedly empty"
            pick = frontier[int(rng.integers(len(frontier)))]
            cells.add(pick)

        mask = np.zeros((raw_h, raw_w), dtype=bool)
        for r, c in cells:
            mask[r, c] = True
        mask = _crop_mask(mask)
        if min(mask.shape) >= 2:
            return mask

    raise RuntimeError("failed to sample a non-degenerate object mask")


def _paint_mask(grid: np.ndarray, top: int, left: int, mask: np.ndarray, color: int) -> None:
    h, w = mask.shape
    patch = grid[top : top + h, left : left + w]
    patch[mask] = int(color)


def _gen_horizontal(rng: np.random.Generator, masks: list[np.ndarray], colors: list[int], source_color: int) -> np.ndarray | None:
    gaps = [int(rng.integers(1, 3)) for _ in range(len(masks) - 1)]
    left_margin = int(rng.integers(1, 4))
    right_margin = int(rng.integers(1, 4))
    top_margin = int(rng.integers(1, 5))
    bottom_margin = int(rng.integers(1, 5))
    source_gap = int(rng.integers(2, 4))

    band_h = max(3, max(mask.shape[0] for mask in masks))
    total_w = left_margin + 3 + sum(mask.shape[1] for mask in masks) + sum(gaps) + source_gap + 1 + right_margin
    total_h = top_margin + band_h + bottom_margin
    if total_h > MAX_DIM or total_w > MAX_DIM:
        return None

    grid = np.zeros((total_h, total_w), dtype=int)

    arrow_top = top_margin + (band_h - 3) // 2
    arrow_left = left_margin
    _paint_mask(grid, arrow_top, arrow_left, ARROW_RIGHT, ARROW_COLOR)

    cursor = left_margin + 3
    for index, (mask, color) in enumerate(zip(masks, colors)):
        obj_top = top_margin + (band_h - mask.shape[0]) // 2
        _paint_mask(grid, obj_top, cursor, mask, color)
        cursor += mask.shape[1]
        if index + 1 < len(masks):
            cursor += gaps[index]

    source_row = top_margin + band_h // 2
    source_col = cursor + source_gap - 1
    grid[source_row, source_col] = int(source_color)
    return grid


def _gen_vertical(rng: np.random.Generator, masks: list[np.ndarray], colors: list[int], source_color: int) -> np.ndarray | None:
    gaps = [int(rng.integers(1, 3)) for _ in range(len(masks) - 1)]
    left_margin = int(rng.integers(1, 5))
    right_margin = int(rng.integers(1, 5))
    top_margin = int(rng.integers(1, 4))
    bottom_margin = int(rng.integers(1, 4))
    source_gap = int(rng.integers(2, 4))

    band_w = max(3, max(mask.shape[1] for mask in masks))
    total_h = top_margin + 3 + sum(mask.shape[0] for mask in masks) + sum(gaps) + source_gap + 1 + bottom_margin
    total_w = left_margin + band_w + right_margin
    if total_h > MAX_DIM or total_w > MAX_DIM:
        return None

    grid = np.zeros((total_h, total_w), dtype=int)

    arrow_top = top_margin
    arrow_left = left_margin + (band_w - 3) // 2
    _paint_mask(grid, arrow_top, arrow_left, ARROW_DOWN, ARROW_COLOR)

    cursor = top_margin + 3
    for index, (mask, color) in enumerate(zip(masks, colors)):
        obj_left = left_margin + (band_w - mask.shape[1]) // 2
        _paint_mask(grid, cursor, obj_left, mask, color)
        cursor += mask.shape[0]
        if index + 1 < len(masks):
            cursor += gaps[index]

    source_row = cursor + source_gap - 1
    source_col = left_margin + band_w // 2
    grid[source_row, source_col] = int(source_color)
    return grid


def _gen_attempt(rng: np.random.Generator) -> np.ndarray | None:
    object_count = int(rng.integers(3, 7))
    if object_count + 1 > len(PALETTE):
        return None

    colors = list(rng.choice(PALETTE, size=object_count + 1, replace=False))
    object_colors = colors[:-1]
    source_color = colors[-1]
    masks = [_random_object_mask(rng) for _ in range(object_count)]
    direction = "horizontal" if int(rng.integers(2)) == 0 else "vertical"

    if direction == "horizontal":
        grid = _gen_horizontal(rng, masks, object_colors, source_color)
    else:
        grid = _gen_vertical(rng, masks, object_colors, source_color)
    if grid is None:
        return None

    try:
        arrow, source, objects, axis = _analyze_scene(grid)
    except AssertionError:
        return None

    if axis != direction:
        return None
    if len(objects) != object_count:
        return None
    if arrow.color != ARROW_COLOR or source.color != source_color:
        return None

    return grid


def generate_input() -> np.ndarray:
    rng = np.random.default_rng()
    for _ in range(GENERATE_ATTEMPTS):
        grid = _gen_attempt(rng)
        if grid is not None:
            return grid
    raise RuntimeError(f"failed to generate a valid input after {GENERATE_ATTEMPTS} attempts")
