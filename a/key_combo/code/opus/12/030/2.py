from __future__ import annotations

import random
from typing import Iterable

import numpy as np


PALETTE = [1, 2, 3, 4, 5, 6, 7, 8, 9]


def _neighbors4(r: int, c: int) -> Iterable[tuple[int, int]]:
    yield r - 1, c
    yield r + 1, c
    yield r, c - 1
    yield r, c + 1


def _component_coords(mask: np.ndarray) -> list[np.ndarray]:
    h, w = mask.shape
    seen = np.zeros_like(mask, dtype=bool)
    comps: list[np.ndarray] = []
    for r in range(h):
        for c in range(w):
            if not mask[r, c] or seen[r, c]:
                continue
            stack = [(r, c)]
            seen[r, c] = True
            coords: list[tuple[int, int]] = []
            while stack:
                rr, cc = stack.pop()
                coords.append((rr, cc))
                for nr, nc in _neighbors4(rr, cc):
                    if 0 <= nr < h and 0 <= nc < w and mask[nr, nc] and not seen[nr, nc]:
                        seen[nr, nc] = True
                        stack.append((nr, nc))
            comps.append(np.array(coords, dtype=int))
    return comps


def _bbox_from_coords(coords: np.ndarray) -> tuple[int, int, int, int]:
    r0, c0 = coords.min(axis=0)
    r1, c1 = coords.max(axis=0)
    return int(r0), int(c0), int(r1), int(c1)


def _crop_mask(mask: np.ndarray) -> np.ndarray:
    coords = np.argwhere(mask)
    if coords.size == 0:
        return np.zeros((0, 0), dtype=bool)
    r0, c0, r1, c1 = _bbox_from_coords(coords)
    return mask[r0 : r1 + 1, c0 : c1 + 1]


def _is_rot90_symmetric(mask: np.ndarray) -> bool:
    cropped = _crop_mask(mask)
    if cropped.size == 0 or cropped.shape[0] != cropped.shape[1]:
        return False
    return bool(np.array_equal(cropped, np.rot90(cropped)))


def _extract_layers(grid: np.ndarray) -> list[dict]:
    g = np.asarray(grid, dtype=int)
    layers: list[dict] = []
    for color in sorted(int(c) for c in np.unique(g) if int(c) != 0):
        comps = _component_coords(g == color)
        for coords in comps:
            bbox = _bbox_from_coords(coords)
            r0, c0, r1, c1 = bbox
            layers.append(
                {
                    "color": color,
                    "coords": coords,
                    "bbox": bbox,
                    "bbox_area": (r1 - r0 + 1) * (c1 - c0 + 1),
                    "area": int(coords.shape[0]),
                }
            )
    layers.sort(key=lambda item: (item["bbox_area"], item["area"]), reverse=True)
    return layers


def main(grid: np.ndarray) -> np.ndarray:
    """
    Filter nested layers by 90-degree rotational symmetry and compress the survivors into nested square rings.

    Input structure:
    - The grid has black background and one central nested stack of monochrome layers.
    - Each non-black layer is 4-connected, uses a distinct color, and the layers are ordered from outermost to innermost by decreasing bounding-box size.
    - Some layers are invariant under a quarter-turn about their own center and others are not.

    Rule:
    - Extract the nested colored layers and sort them from outermost to innermost.
    - Test each layer's pixel set for 90-degree rotational symmetry around the layer's center, using the layer alone.
    - Discard every layer that fails the symmetry test.
    - Build a (2K-1) x (2K-1) output, where K is the number of surviving layers, and fill nested square rings from outermost to innermost with the surviving layer colors in their original order.

    Key insight:
    - The output size is not fixed by the input layer count; it is determined only after filtering layers by rotational symmetry, so the solver must identify layers first and compress only the survivors.
    """
    g = np.asarray(grid, dtype=int)
    assert g.ndim == 2

    survivors = []
    for layer in _extract_layers(g):
        mask = np.zeros_like(g, dtype=bool)
        rr, cc = layer["coords"][:, 0], layer["coords"][:, 1]
        mask[rr, cc] = True
        if _is_rot90_symmetric(mask):
            survivors.append(layer["color"])

    if not survivors:
        return np.zeros((1, 1), dtype=int)

    size = 2 * len(survivors) - 1
    out = np.zeros((size, size), dtype=int)
    for i, color in enumerate(survivors):
        out[i : size - i, i : size - i] = int(color)
    return out


def _shape_mask(radius: int, limits: list[int], canvas_radius: int) -> np.ndarray:
    side = 2 * canvas_radius + 1
    center = canvas_radius
    rr, cc = np.indices((side, side))
    dr = rr - center
    dc = cc - center
    ad_r = np.abs(dr)
    ad_c = np.abs(dc)
    cheb = np.maximum(ad_r, ad_c)
    manhattan = ad_r + ad_c

    quad_limits = np.where(
        (dr <= 0) & (dc >= 0),
        limits[0],
        np.where(
            (dr >= 0) & (dc >= 0),
            limits[1],
            np.where((dr >= 0) & (dc <= 0), limits[2], limits[3]),
        ),
    )
    return (cheb <= radius) & (manhattan <= quad_limits)


def _build_layer_masks(
    radii: list[int],
    limit_sets: list[list[int]],
) -> list[np.ndarray]:
    canvas_radius = radii[0]
    filled_masks = [_shape_mask(radius, limits, canvas_radius) for radius, limits in zip(radii, limit_sets)]
    layers: list[np.ndarray] = []
    for idx, mask in enumerate(filled_masks):
        if idx == len(filled_masks) - 1:
            layers.append(mask.copy())
        else:
            layers.append(mask & ~filled_masks[idx + 1])
    return layers


def _layer_centers(layers: list[dict]) -> set[tuple[float, float]]:
    centers = set()
    for layer in layers:
        r0, c0, r1, c1 = layer["bbox"]
        centers.add(((r0 + r1) / 2.0, (c0 + c1) / 2.0))
    return centers


def _generate_nested_structure() -> tuple[list[np.ndarray], list[int]]:
    for _ in range(800):
        n_layers = random.choice([4, 5, 5, 6, 6])
        inner_radius = random.randint(2, 3)
        radii = [inner_radius]
        for _ in range(n_layers - 1):
            radii.append(radii[-1] + random.randint(1, 2))
        radii.reverse()

        values = [0] * n_layers
        values[-1] = radii[-1] + random.randint(1, min(3, radii[-1] - 1))
        ok = True
        for idx in range(n_layers - 2, -1, -1):
            low = max(values[idx + 1], radii[idx] + 1)
            high = min(2 * radii[idx] - 1, values[idx + 1] + (radii[idx] - radii[idx + 1]) + 1)
            if low > high:
                ok = False
                break
            values[idx] = random.randint(low, high)
        if not ok or values[0] >= 2 * radii[0] - 1:
            continue

        limit_sets = [[v, v, v, v] for v in values]
        outer_diff = random.randint(1, min(2, 2 * radii[0] - 1 - values[0]))
        limit_sets[0] = [values[0], values[0] + outer_diff, values[0], values[0] + outer_diff]

        if n_layers >= 5 and random.random() < 0.25:
            idx = random.randint(1, n_layers - 2)
            cap = min(2 * radii[idx] - 1, values[idx - 1])
            if values[idx] < cap:
                diff = random.randint(1, min(2, cap - values[idx]))
                limit_sets[idx] = [values[idx], values[idx] + diff, values[idx], values[idx] + diff]

        layer_masks = _build_layer_masks(radii, limit_sets)
        if any(int(mask.sum()) < 8 for mask in layer_masks):
            continue
        if any(len(_component_coords(mask)) != 1 for mask in layer_masks):
            continue

        sym_flags = [_is_rot90_symmetric(mask) for mask in layer_masks]
        if sym_flags.count(True) < 2 or sym_flags.count(False) < 1:
            continue

        colors = random.sample(PALETTE, n_layers)
        return layer_masks, colors

    raise RuntimeError("failed to generate a valid nested structure")


def _render_grid(layer_masks: list[np.ndarray], colors: list[int]) -> np.ndarray:
    local_side = layer_masks[0].shape[0]
    extra_rows = random.randint(2, max(2, 30 - local_side))
    extra_cols = random.randint(2, max(2, 30 - local_side))
    height = local_side + extra_rows
    width = local_side + extra_cols

    top_min = max(1, extra_rows // 2 - 1)
    top_max = min(extra_rows - 1, extra_rows // 2 + 1)
    left_min = max(1, extra_cols // 2 - 1)
    left_max = min(extra_cols - 1, extra_cols // 2 + 1)
    top = random.randint(top_min, top_max)
    left = random.randint(left_min, left_max)

    grid = np.zeros((height, width), dtype=int)
    for mask, color in zip(layer_masks, colors):
        patch = np.where(mask, int(color), 0)
        grid[top : top + local_side, left : left + local_side][mask] = patch[mask]
    return grid


def _validate_generated_grid(grid: np.ndarray) -> bool:
    g = np.asarray(grid, dtype=int)
    if g.ndim != 2 or max(g.shape) > 30:
        return False

    layers = _extract_layers(g)
    if not (4 <= len(layers) <= 6):
        return False
    if len({layer["color"] for layer in layers}) != len(layers):
        return False
    if any(layer["area"] < 8 for layer in layers):
        return False
    if len(_layer_centers(layers)) != 1:
        return False

    sym_flags = []
    bbox_areas = []
    for layer in layers:
        mask = np.zeros_like(g, dtype=bool)
        rr, cc = layer["coords"][:, 0], layer["coords"][:, 1]
        mask[rr, cc] = True
        if len(_component_coords(mask)) != 1:
            return False
        sym_flags.append(_is_rot90_symmetric(mask))
        bbox_areas.append(layer["bbox_area"])

    if sym_flags.count(True) < 2 or sym_flags.count(False) < 1:
        return False
    if any(a <= b for a, b in zip(bbox_areas, bbox_areas[1:])):
        return False

    out = main(g)
    expected = 2 * sym_flags.count(True) - 1
    if out.shape != (expected, expected):
        return False
    if max(out.shape) > 30:
        return False
    return True


def generate_input() -> np.ndarray:
    for _ in range(400):
        layer_masks, colors = _generate_nested_structure()
        grid = _render_grid(layer_masks, colors)
        if _validate_generated_grid(grid):
            return grid
    raise RuntimeError("failed to generate a valid puzzle instance")
