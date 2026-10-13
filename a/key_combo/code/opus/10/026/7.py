from __future__ import annotations

import random
from collections import Counter

import numpy as np


BG = 5
TEMPLATE_COLOR = 0
INDICATOR_COLORS = [1, 2, 3, 4, 6, 7, 8, 9]
CARDINAL_STEPS = {
    "up": (-1, 0),
    "down": (1, 0),
    "left": (0, -1),
    "right": (0, 1),
}
SIDE_CONTACTS = {
    "up": (0, 1),
    "down": (2, 1),
    "left": (1, 0),
    "right": (1, 2),
}
INDICATOR_OFFSETS = {
    "up": (-1, 1),
    "down": (3, 1),
    "left": (1, -1),
    "right": (1, 3),
}
SHAPES = {
    "plus": {(0, 1), (1, 0), (1, 1), (1, 2), (2, 1)},
    "left_t": {(0, 1), (1, 0), (1, 1), (2, 1)},
    "right_t": {(0, 1), (1, 1), (1, 2), (2, 1)},
}
DIRECTION_SETS = [
    ("up", "down"),
    ("up", "down", "left"),
    ("up", "down", "right"),
]
RETRY_LIMIT = 500


def _in_bounds(shape: tuple[int, int], r: int, c: int) -> bool:
    h, w = shape
    return 0 <= r < h and 0 <= c < w


def _components_for_color(grid: np.ndarray, color: int) -> list[list[tuple[int, int]]]:
    mask = grid == color
    h, w = grid.shape
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
                rr, cc = stack.pop()
                comp.append((rr, cc))
                for nr, nc in ((rr - 1, cc), (rr + 1, cc), (rr, cc - 1), (rr, cc + 1)):
                    if _in_bounds((h, w), nr, nc) and mask[nr, nc] and not seen[nr, nc]:
                        seen[nr, nc] = True
                        stack.append((nr, nc))
            comps.append(comp)
    return comps


def _bbox(coords: list[tuple[int, int]]) -> tuple[int, int, int, int]:
    rows = [r for r, _ in coords]
    cols = [c for _, c in coords]
    return min(rows), min(cols), max(rows), max(cols)


def _extract_scene(
    grid: np.ndarray,
) -> tuple[list[tuple[int, int]], tuple[int, int, int, int], list[tuple[int, tuple[int, int], str]]]:
    g = np.asarray(grid, dtype=int)
    assert g.ndim == 2, "expected a 2D grid"
    assert int(Counter(map(int, g.ravel())).most_common(1)[0][0]) == BG, "expected grey background"

    template_components = _components_for_color(g, TEMPLATE_COLOR)
    assert len(template_components) == 1, "expected exactly one black template"
    template_coords = template_components[0]
    template_bbox = _bbox(template_coords)
    r0, c0, r1, c1 = template_bbox
    center_r = (r0 + r1) / 2.0
    center_c = (c0 + c1) / 2.0
    template_mask = np.zeros_like(g, dtype=bool)
    for r, c in template_coords:
        template_mask[r, c] = True

    indicators: list[tuple[int, tuple[int, int], str]] = []
    for color in sorted(set(map(int, g.ravel())) - {BG, TEMPLATE_COLOR}):
        components = _components_for_color(g, color)
        for comp in components:
            assert len(comp) == 1, "indicators must be single pixels"
            r, c = comp[0]
            assert any(
                _in_bounds(g.shape, nr, nc) and template_mask[nr, nc]
                for nr, nc in ((r - 1, c), (r + 1, c), (r, c - 1), (r, c + 1))
            ), "indicators must touch the template"
            dr = r - center_r
            dc = c - center_c
            if abs(dr) > abs(dc):
                direction = "down" if dr > 0 else "up"
            elif abs(dc) > abs(dr):
                direction = "right" if dc > 0 else "left"
            elif r < r0:
                direction = "up"
            elif r > r1:
                direction = "down"
            elif c < c0:
                direction = "left"
            else:
                direction = "right"
            indicators.append((color, (r, c), direction))

    assert 2 <= len(indicators) <= 3, "expected 2 or 3 indicators"
    indicator_dirs = [direction for _, _, direction in indicators]
    assert "up" in indicator_dirs or any(direction in {"left", "right"} for direction in indicator_dirs)
    return template_coords, template_bbox, indicators


def _render_tiled_state(
    grid: np.ndarray,
) -> tuple[np.ndarray, np.ndarray, list[tuple[int, tuple[int, int], str]]]:
    g = np.asarray(grid, dtype=int)
    h, w = g.shape
    template_coords, template_bbox, indicators = _extract_scene(g)
    r0, c0, r1, c1 = template_bbox
    bbox_h = r1 - r0 + 1
    bbox_w = c1 - c0 + 1
    rel_coords = [(r - r0, c - c0) for r, c in template_coords]

    tiled = np.full((h, w), BG, dtype=int)
    sources = np.full((h, w), -1, dtype=int)
    for r, c in template_coords:
        tiled[r, c] = TEMPLATE_COLOR
        sources[r, c] = 0

    for source_id, (color, _pos, direction) in enumerate(indicators, start=1):
        step_r, step_c = CARDINAL_STEPS[direction]
        stride = bbox_h + 1 if direction in {"up", "down"} else bbox_w + 1
        k = 1
        while True:
            top = r0 + step_r * stride * k
            left = c0 + step_c * stride * k
            if top > h - 1 or top + bbox_h - 1 < 0 or left > w - 1 or left + bbox_w - 1 < 0:
                break
            for rr, cc in rel_coords:
                tr = top + rr
                tc = left + cc
                if _in_bounds((h, w), tr, tc):
                    assert sources[tr, tc] == -1, "tiled copies should not overlap before gravity"
                    tiled[tr, tc] = color
                    sources[tr, tc] = source_id
            k += 1

    return tiled, sources, indicators


def _apply_gravity(grid: np.ndarray, bg: int = BG) -> np.ndarray:
    g = np.asarray(grid, dtype=int)
    h, w = g.shape
    out = np.full((h, w), bg, dtype=int)
    for c in range(w):
        column = [int(g[r, c]) for r in range(h - 1, -1, -1) if int(g[r, c]) != bg]
        for offset, color in enumerate(column):
            out[h - 1 - offset, c] = color
    return out


def _apply_gravity_to_sources(sources: np.ndarray, colors: np.ndarray, bg: int = BG) -> np.ndarray:
    h, w = colors.shape
    out = np.full((h, w), -1, dtype=int)
    for c in range(w):
        column = [int(sources[r, c]) for r in range(h - 1, -1, -1) if int(colors[r, c]) != bg]
        for offset, source_id in enumerate(column):
            out[h - 1 - offset, c] = source_id
    return out


def main(grid: np.ndarray) -> np.ndarray:
    """
    Tile the black template in each indicator direction, then let every non-grey pixel fall straight down as an independent column stack.

    Input structure:
    - The grid has a grey background, one 4-connected black template, and 2-3 distinct single-pixel indicators touching the template.
    - Each indicator color is non-grey and non-black, and its position relative to the template center defines one cardinal direction.
    - The template occupies a small upper-half region; repeated translated copies may extend partially beyond the canvas.

    Rule:
    - Erase the indicator pixels and keep the original black template.
    - For each indicator, recolor the template to that indicator color and repeat it outward in the indicator direction with stride equal to the template bbox size along that axis plus one.
    - After all directional copies are placed, apply downward gravity independently in each column to every non-grey pixel, preserving bottom-to-top order within each column.

    Key insight:
    - The answer is not the tiled layout itself: all copies from every direction must be created first, and only then globally compacted downward so different directional copies collide and stack.
    """
    tiled, _sources, _indicators = _render_tiled_state(np.asarray(grid, dtype=int))
    return _apply_gravity(tiled, bg=BG)


def _shape_supports(shape: set[tuple[int, int]], directions: tuple[str, ...]) -> bool:
    return all(SIDE_CONTACTS[direction] in shape for direction in directions)


def _build_candidate_input() -> np.ndarray:
    h = random.randint(18, 24)
    w = random.randint(18, 24)
    directions = random.choice(DIRECTION_SETS)
    colors = random.sample(INDICATOR_COLORS, k=len(directions))

    compatible_shapes = [shape for shape in SHAPES.values() if _shape_supports(shape, directions)]
    shape = random.choice(compatible_shapes)

    row_hi = min(h // 2 - 2, h - 7)
    row_lo = 4
    assert row_lo <= row_hi
    r0 = random.randint(row_lo, row_hi)
    c0 = random.randint(1, w - 4)

    grid = np.full((h, w), BG, dtype=int)
    for rr, cc in shape:
        grid[r0 + rr, c0 + cc] = TEMPLATE_COLOR
    for color, direction in zip(colors, directions):
        dr, dc = INDICATOR_OFFSETS[direction]
        grid[r0 + dr, c0 + dc] = color

    if random.random() < 0.5:
        grid = np.rot90(grid, 2).copy()
    return grid


def _passes_semantic_checks(grid: np.ndarray) -> bool:
    try:
        tiled, sources, indicators = _render_tiled_state(grid)
        out = main(grid)
    except AssertionError:
        return False

    if np.array_equal(tiled, out):
        return False

    directions = {direction for _color, _pos, direction in indicators}
    if "up" not in directions and not directions.intersection({"left", "right"}):
        return False

    moved = False
    for c in range(grid.shape[1]):
        before = np.flatnonzero(tiled[:, c] != BG)
        after = np.flatnonzero(out[:, c] != BG)
        if before.size != after.size:
            return False
        if before.size and not np.array_equal(before, after):
            moved = True
    if not moved:
        return False

    source_after = _apply_gravity_to_sources(sources, tiled, bg=BG)
    collided_directions = False
    strong_stack = False
    for c in range(grid.shape[1]):
        source_ids = [int(source_after[r, c]) for r in range(grid.shape[0]) if source_after[r, c] > 0]
        colors = [int(out[r, c]) for r in range(grid.shape[0]) if int(out[r, c]) != BG]
        if len(set(source_ids)) >= 2:
            collided_directions = True
        if len(colors) >= 4 and len(set(colors)) >= 3:
            strong_stack = True
    if not collided_directions or not strong_stack:
        return False

    non_bg_colors = set(map(int, out.ravel())) - {BG}
    if TEMPLATE_COLOR not in non_bg_colors or len(non_bg_colors) < 3:
        return False

    return True


def generate_input() -> np.ndarray:
    for _ in range(RETRY_LIMIT):
        grid = _build_candidate_input()
        if _passes_semantic_checks(grid):
            return grid
    raise RuntimeError("failed to generate a valid input for puzzle 7")
