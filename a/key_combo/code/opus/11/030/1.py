from __future__ import annotations

import random
from functools import lru_cache

import numpy as np


BACKGROUND = 0
RING_COLORS = (1, 2, 3, 4, 5)
MIN_LAYERS = 2
MAX_LAYERS = 5
MAX_SIDE = 30
GEN_RETRIES = 400


def _bbox_from_coords(coords: list[tuple[int, int]]) -> tuple[int, int, int, int]:
    rows = [r for r, _ in coords]
    cols = [c for _, c in coords]
    return min(rows), min(cols), max(rows), max(cols)


def _components4(mask: np.ndarray) -> list[list[tuple[int, int]]]:
    h, w = mask.shape
    seen = np.zeros_like(mask, dtype=bool)
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
                for nr, nc in ((cr - 1, cc), (cr + 1, cc), (cr, cc - 1), (cr, cc + 1)):
                    if 0 <= nr < h and 0 <= nc < w and mask[nr, nc] and not seen[nr, nc]:
                        seen[nr, nc] = True
                        stack.append((nr, nc))
            comps.append(comp)
    return comps


def _ring_mask(side: int) -> np.ndarray:
    mask = np.zeros((side, side), dtype=bool)
    mask[0, :] = True
    mask[-1, :] = True
    mask[:, 0] = True
    mask[:, -1] = True
    return mask


def _corner_specs(bbox: tuple[int, int, int, int]) -> list[tuple[tuple[int, int], tuple[int, int]]]:
    r0, c0, r1, c1 = bbox
    return [
        ((r0, c0), (-1, -1)),
        ((r0, c1), (-1, 1)),
        ((r1, c0), (1, -1)),
        ((r1, c1), (1, 1)),
    ]


def _extract_rings(grid: np.ndarray) -> list[dict[str, object]]:
    g = np.asarray(grid, dtype=int)
    assert g.ndim == 2, "expected a 2D grid"

    components = _components4(g != BACKGROUND)
    assert MIN_LAYERS <= len(components) <= MAX_LAYERS, "expected 2-5 foreground rings"

    rings: list[dict[str, object]] = []
    for comp in components:
        bbox = _bbox_from_coords(comp)
        r0, c0, r1, c1 = bbox
        height = r1 - r0 + 1
        width = c1 - c0 + 1
        assert height == width >= 3, "each layer must be a square ring"

        colors = {int(g[r, c]) for r, c in comp}
        assert len(colors) == 1, "each ring must be monochromatic"
        color = next(iter(colors))

        patch = g[r0 : r1 + 1, c0 : c1 + 1]
        assert np.array_equal(patch == color, _ring_mask(height)), "component must be a 1-pixel square ring"

        rings.append({"bbox": bbox, "color": color, "side": height})

    rings.sort(key=lambda ring: int(ring["side"]), reverse=True)
    actual_colors = [int(ring["color"]) for ring in rings]
    assert actual_colors == list(RING_COLORS[: len(rings)]), "ring colors must follow the specified outer-to-inner order"

    outer_bbox = rings[0]["bbox"]
    h, w = g.shape
    r0, c0, r1, c1 = outer_bbox
    assert r0 == h - 1 - r1 and c0 == w - 1 - c1, "outer ring must be centered in the grid"

    for outer, inner in zip(rings, rings[1:]):
        or0, oc0, or1, oc1 = outer["bbox"]
        ir0, ic0, ir1, ic1 = inner["bbox"]
        assert (ir0, ic0, ir1, ic1) == (or0 + 2, oc0 + 2, or1 - 2, oc1 - 2), "rings must be concentric with a 1-cell gap"

    return rings


def _paint_rays(
    grid: np.ndarray,
    bbox: tuple[int, int, int, int],
    ray_len: int,
    color: int,
) -> np.ndarray:
    out = np.array(grid, copy=True)
    h, w = out.shape
    for (r, c), (dr, dc) in _corner_specs(bbox):
        for step in range(1, ray_len + 1):
            nr = r + dr * step
            nc = c + dc * step
            assert 0 <= nr < h and 0 <= nc < w, "ray must stay in bounds"
            assert out[nr, nc] == BACKGROUND, "rays may only paint background"
            out[nr, nc] = int(color)
    return out


def main(grid: np.ndarray) -> np.ndarray:
    """
    Count concentric square-ring layers and emit diagonal corner rays whose length equals that layer count.

    Input structure:
    - The grid has black background and one centered set of concentric square rings.
    - There are 2 to 5 monochromatic rings, each 1 pixel thick, with a 1-pixel black gap between successive rings.
    - From outermost to innermost, the ring colors are blue, red, green, yellow, and gray prefixes.

    Rule:
    - Detect the nested ring layers and count their total number N.
    - Locate the four corners of the outermost ring's bounding box.
    - From each outer corner, extend a diagonal ray outward for exactly N cells, excluding the corner itself.
    - Keep all original pixels unchanged and draw the new ray cells in the outer ring's color.

    Key insight:
    - The ray length is not marked explicitly; it is encoded by the number of nested rings, while the ray origins come from the geometry of the outermost square.
    """
    rings = _extract_rings(grid)
    outer = rings[0]
    return _paint_rays(grid, outer["bbox"], len(rings), outer["color"])


@lru_cache(maxsize=None)
def _generation_configs(layer_count: int) -> tuple[tuple[int, int, int], ...]:
    configs: list[tuple[int, int, int]] = []
    for inner_side in range(3, MAX_SIDE + 1, 2):
        outer_side = inner_side + 4 * (layer_count - 1)
        if outer_side > MAX_SIDE:
            continue
        for margin_r in range(layer_count, layer_count + 5):
            for margin_c in range(layer_count, layer_count + 5):
                height = outer_side + 2 * margin_r
                width = outer_side + 2 * margin_c
                if max(height, width) <= MAX_SIDE:
                    configs.append((inner_side, margin_r, margin_c))
    return tuple(configs)


def _build_input(layer_count: int, inner_side: int, margin_r: int, margin_c: int) -> np.ndarray:
    outer_side = inner_side + 4 * (layer_count - 1)
    height = outer_side + 2 * margin_r
    width = outer_side + 2 * margin_c
    grid = np.zeros((height, width), dtype=int)

    r0 = margin_r
    c0 = margin_c
    for depth in range(layer_count):
        side = outer_side - 4 * depth
        color = RING_COLORS[depth]
        rr = r0 + 2 * depth
        cc = c0 + 2 * depth
        grid[rr, cc : cc + side] = color
        grid[rr + side - 1, cc : cc + side] = color
        grid[rr : rr + side, cc] = color
        grid[rr : rr + side, cc + side - 1] = color
    return grid


def _validate_case(grid: np.ndarray) -> np.ndarray:
    rings = _extract_rings(grid)
    layer_count = len(rings)
    outer_bbox = rings[0]["bbox"]
    outer_color = int(rings[0]["color"])
    output = _paint_rays(grid, outer_bbox, layer_count, outer_color)

    delta = output != grid
    assert int(delta.sum()) == 4 * layer_count, "output must add exactly four rays of length N"
    assert np.all(grid[grid != BACKGROUND] == output[grid != BACKGROUND]), "original non-background pixels must stay unchanged"
    assert np.all(output[delta] == outer_color), "every new pixel must use the outer ring color"

    if layer_count > MIN_LAYERS - 1:
        shorter = _paint_rays(grid, outer_bbox, layer_count - 1, outer_color)
        assert not np.array_equal(shorter, output), "layer count must affect the result"

        second_bbox = rings[1]["bbox"]
        blocked = False
        for (r, c), (dr, dc) in _corner_specs(second_bbox):
            for step in range(1, layer_count + 1):
                nr = r + dr * step
                nc = c + dc * step
                if not (0 <= nr < grid.shape[0] and 0 <= nc < grid.shape[1]) or grid[nr, nc] != BACKGROUND:
                    blocked = True
                    break
            if blocked:
                break
        assert blocked, "using an inner ring as the ray origin must be invalid"

    assert np.array_equal(main(grid), output), "solver must agree with the structural oracle"
    return output


def _sample_layer_count() -> int:
    return random.choice([2, 2, 3, 4, 5, 5])


def generate_input() -> np.ndarray:
    for _ in range(GEN_RETRIES):
        layer_count = _sample_layer_count()
        configs = _generation_configs(layer_count)
        if not configs:
            continue
        inner_side, margin_r, margin_c = random.choice(configs)
        grid = _build_input(layer_count, inner_side, margin_r, margin_c)
        _validate_case(grid)
        return grid
    raise RuntimeError("failed to generate a valid nested-ring ray puzzle")
