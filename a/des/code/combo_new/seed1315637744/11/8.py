from __future__ import annotations

from collections import deque

import numpy as np


BG = 0
COLOR_POOL = tuple(range(1, 10))
MAX_GENERATION_ATTEMPTS = 500
LAYER_COUNT_RANGE = (2, 6)
CORE_H_RANGE = (1, 6)
CORE_W_RANGE = (1, 6)
MARGIN_CHOICES = (2, 3)
MAX_EXTRA_PADDING = 6


def _bbox(coords: np.ndarray) -> tuple[int, int, int, int]:
    rows = coords[:, 0]
    cols = coords[:, 1]
    return int(rows.min()), int(cols.min()), int(rows.max()), int(cols.max())


def _bbox_area(box: tuple[int, int, int, int]) -> int:
    r0, c0, r1, c1 = box
    return (r1 - r0 + 1) * (c1 - c0 + 1)


def _strictly_contains(
    outer: tuple[int, int, int, int], inner: tuple[int, int, int, int]
) -> bool:
    return (
        outer[0] < inner[0]
        and outer[1] < inner[1]
        and outer[2] > inner[2]
        and outer[3] > inner[3]
    )


def _contains_with_gap(
    outer: tuple[int, int, int, int], inner: tuple[int, int, int, int], gap: int = 1
) -> bool:
    pad = gap + 1
    return (
        outer[0] + pad <= inner[0]
        and outer[1] + pad <= inner[1]
        and outer[2] - pad >= inner[2]
        and outer[3] - pad >= inner[3]
    )


def _find_components(grid: np.ndarray) -> list[dict]:
    h, w = grid.shape
    seen = np.zeros((h, w), dtype=bool)
    comps: list[dict] = []
    for r in range(h):
        for c in range(w):
            color = int(grid[r, c])
            if color == BG or seen[r, c]:
                continue
            q = deque([(r, c)])
            seen[r, c] = True
            cells: list[tuple[int, int]] = []
            while q:
                rr, cc = q.popleft()
                cells.append((rr, cc))
                for dr, dc in ((1, 0), (-1, 0), (0, 1), (0, -1)):
                    nr, nc = rr + dr, cc + dc
                    if 0 <= nr < h and 0 <= nc < w and not seen[nr, nc] and int(grid[nr, nc]) == color:
                        seen[nr, nc] = True
                        q.append((nr, nc))
            coords = np.array(cells, dtype=int)
            comps.append(
                {
                    "color": color,
                    "coords": coords,
                    "bbox": _bbox(coords),
                    "area": int(len(cells)),
                }
            )
    return comps


def _component_mask(comp: dict) -> np.ndarray:
    r0, c0, r1, c1 = comp["bbox"]
    mask = np.zeros((r1 - r0 + 1, c1 - c0 + 1), dtype=bool)
    coords = comp["coords"]
    mask[coords[:, 0] - r0, coords[:, 1] - c0] = True
    return mask


def _is_rect_frame(comp: dict) -> bool:
    r0, c0, r1, c1 = comp["bbox"]
    h = r1 - r0 + 1
    w = c1 - c0 + 1
    if h < 3 or w < 3:
        return False
    border = np.zeros((h, w), dtype=bool)
    border[0, :] = True
    border[-1, :] = True
    border[:, 0] = True
    border[:, -1] = True
    return comp["area"] == int(border.sum()) and np.array_equal(_component_mask(comp), border)


def _is_solid_rect(comp: dict) -> bool:
    r0, c0, r1, c1 = comp["bbox"]
    h = r1 - r0 + 1
    w = c1 - c0 + 1
    return comp["area"] == h * w and _component_mask(comp).all()


def _paint_frame(grid: np.ndarray, box: tuple[int, int, int, int], color: int) -> None:
    r0, c0, r1, c1 = box
    grid[r0, c0 : c1 + 1] = color
    grid[r1, c0 : c1 + 1] = color
    grid[r0 : r1 + 1, c0] = color
    grid[r0 : r1 + 1, c1] = color


def _extract_structure(grid: np.ndarray) -> tuple[list[dict], dict]:
    comps = _find_components(grid)
    assert len(comps) >= 2, "expected at least one ring and one center rectangle"

    frames = [comp for comp in comps if _is_rect_frame(comp)]
    solids = [comp for comp in comps if _is_solid_rect(comp)]
    assert frames, "expected rectangular rings"
    assert solids, "expected a solid center rectangle"

    rings = sorted(frames, key=lambda comp: _bbox_area(comp["bbox"]), reverse=True)
    for outer, inner in zip(rings, rings[1:]):
        assert _contains_with_gap(outer["bbox"], inner["bbox"]), "rings must be nested with a black gap"

    inner_ring = rings[-1]
    center_candidates = [
        comp
        for comp in solids
        if _contains_with_gap(inner_ring["bbox"], comp["bbox"])
        and all(_strictly_contains(ring["bbox"], comp["bbox"]) for ring in rings)
    ]
    assert len(center_candidates) == 1, "expected a unique center rectangle"
    center = center_candidates[0]

    assert len(comps) == len(rings) + 1, "unexpected extra foreground components"
    return rings, center


def main(grid: np.ndarray) -> np.ndarray:
    """Shift nested frame colors outward so each ring inherits the original color of the next inner object.

    Input structure:
    - The background is black.
    - One nesting chain of one-pixel-thick hollow rectangular rings appears on the grid, with black gaps between consecutive rings.
    - The innermost region contains one solid rectangle whose color differs from every ring color.

    Rule:
    - Detect the rings ordered from outermost to innermost and detect the solid center rectangle.
    - Recolor the innermost ring to the center rectangle color.
    - Recolor every other ring to the original color of the next inner ring, so colors shift outward by one layer.
    - Leave the solid center rectangle unchanged and keep all black background pixels unchanged.

    Key insight:
    The geometry of nested rectangular containment determines a layer order; once that order is known, the task is a one-step outward rotation of the visible colors, with the center color injected into the innermost ring.
    """
    g = np.asarray(grid, dtype=int)
    assert g.ndim == 2, "expected a 2D grid"

    rings, center = _extract_structure(g)
    out = g.copy()
    shifted_colors = [ring["color"] for ring in rings[1:]] + [int(center["color"])]
    for ring, new_color in zip(rings, shifted_colors):
        coords = ring["coords"]
        out[coords[:, 0], coords[:, 1]] = int(new_color)
    return out


def _sample_inner_to_outer_boxes(
    rng: np.random.Generator, core_h: int, core_w: int, n_layers: int
) -> tuple[tuple[int, int, int, int], list[tuple[int, int, int, int]]]:
    core = (0, 0, core_h - 1, core_w - 1)
    boxes = []
    current = core
    for _ in range(n_layers):
        top = int(rng.choice(MARGIN_CHOICES))
        left = int(rng.choice(MARGIN_CHOICES))
        bottom = int(rng.choice(MARGIN_CHOICES))
        right = int(rng.choice(MARGIN_CHOICES))
        current = (
            current[0] - top,
            current[1] - left,
            current[2] + bottom,
            current[3] + right,
        )
        boxes.append(current)
    return core, boxes


def _shift_boxes(
    boxes: list[tuple[int, int, int, int]], dr: int, dc: int
) -> list[tuple[int, int, int, int]]:
    return [(r0 + dr, c0 + dc, r1 + dr, c1 + dc) for r0, c0, r1, c1 in boxes]


def _expected_output(
    inp: np.ndarray,
    ring_boxes_outer_to_inner: list[tuple[int, int, int, int]],
    ring_colors_outer_to_inner: list[int],
    center_color: int,
) -> np.ndarray:
    out = inp.copy()
    shifted_colors = ring_colors_outer_to_inner[1:] + [center_color]
    for box, color in zip(ring_boxes_outer_to_inner, shifted_colors):
        _paint_frame(out, box, int(color))
    return out


def _gen_attempt(rng: np.random.Generator) -> np.ndarray | None:
    n_layers = int(rng.integers(*LAYER_COUNT_RANGE))
    core_h = int(rng.integers(*CORE_H_RANGE))
    core_w = int(rng.integers(*CORE_W_RANGE))
    core_box, ring_boxes_inner_to_outer = _sample_inner_to_outer_boxes(rng, core_h, core_w, n_layers)
    outer_box = ring_boxes_inner_to_outer[-1]
    total_h = outer_box[2] - outer_box[0] + 1
    total_w = outer_box[3] - outer_box[1] + 1

    max_extra_h = min(MAX_EXTRA_PADDING, 30 - total_h)
    max_extra_w = min(MAX_EXTRA_PADDING, 30 - total_w)
    if max_extra_h < 2 or max_extra_w < 2:
        return None

    extra_h = int(rng.integers(2, max_extra_h + 1))
    extra_w = int(rng.integers(2, max_extra_w + 1))
    top_pad = int(rng.integers(1, extra_h))
    left_pad = int(rng.integers(1, extra_w))
    bottom_pad = extra_h - top_pad
    right_pad = extra_w - left_pad
    if min(top_pad, left_pad, bottom_pad, right_pad) < 1:
        return None

    grid_h = total_h + extra_h
    grid_w = total_w + extra_w
    if grid_h > 30 or grid_w > 30:
        return None

    shift_r = top_pad - outer_box[0]
    shift_c = left_pad - outer_box[1]
    shifted_core = _shift_boxes([core_box], shift_r, shift_c)[0]
    shifted_rings_inner_to_outer = _shift_boxes(ring_boxes_inner_to_outer, shift_r, shift_c)
    shifted_rings_outer_to_inner = list(reversed(shifted_rings_inner_to_outer))

    palette = rng.choice(np.array(COLOR_POOL, dtype=int), size=n_layers + 1, replace=False)
    center_color = int(palette[0])
    ring_colors_inner_to_outer = [int(c) for c in palette[1:]]
    ring_colors_outer_to_inner = list(reversed(ring_colors_inner_to_outer))

    grid = np.zeros((grid_h, grid_w), dtype=int)
    for box, color in zip(shifted_rings_inner_to_outer, ring_colors_inner_to_outer):
        _paint_frame(grid, box, color)

    r0, c0, r1, c1 = shifted_core
    grid[r0 : r1 + 1, c0 : c1 + 1] = center_color

    expected = _expected_output(grid, shifted_rings_outer_to_inner, ring_colors_outer_to_inner, center_color)
    try:
        actual = main(grid)
    except AssertionError:
        return None
    if not np.array_equal(actual, expected):
        return None
    return grid


def generate_input() -> np.ndarray:
    rng = np.random.default_rng()
    for _ in range(MAX_GENERATION_ATTEMPTS):
        grid = _gen_attempt(rng)
        if grid is not None:
            return grid
    raise RuntimeError("failed to generate a valid nested-ring input")
