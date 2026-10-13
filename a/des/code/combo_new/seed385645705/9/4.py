from __future__ import annotations

from dataclasses import dataclass
import random

import numpy as np


BG = 0
PALETTE = tuple(range(1, 10))

# --------------- solver ---------------


@dataclass(frozen=True)
class _Layer:
    color: int
    r0: int
    c0: int
    r1: int
    c1: int


def _rect_mask(shape: tuple[int, int], layer: _Layer) -> np.ndarray:
    mask = np.zeros(shape, dtype=bool)
    mask[layer.r0 : layer.r1 + 1, layer.c0 : layer.c1 + 1] = True
    return mask


def _compose(shape: tuple[int, int], layers: list[_Layer], mirror_even: bool) -> np.ndarray:
    out = np.full(shape, BG, dtype=int)
    width = shape[1]
    for idx, layer in enumerate(layers, start=1):
        c0, c1 = layer.c0, layer.c1
        if mirror_even and idx % 2 == 0:
            c0, c1 = width - 1 - c1, width - 1 - c0
        out[layer.r0 : layer.r1 + 1, c0 : c1 + 1] = int(layer.color)
    return out


def _row_order(g: np.ndarray, row: int, fg_colors: set[int]) -> list[int] | None:
    cols = np.flatnonzero(g[row] != BG)
    if len(cols) != len(fg_colors):
        return None
    colors = [int(g[row, c]) for c in cols]
    if len(set(colors)) != len(fg_colors) or set(colors) != fg_colors:
        return None
    return colors


def _reconstruct_layers(g: np.ndarray, strip_row: int, order: list[int]) -> list[_Layer] | None:
    layers: list[_Layer] = []
    for color in order:
        mask = g == int(color)
        mask[strip_row, :] = False
        coords = np.argwhere(mask)
        if coords.size == 0:
            return None
        r0, c0 = coords.min(axis=0)
        r1, c1 = coords.max(axis=0)
        layers.append(_Layer(int(color), int(r0), int(c0), int(r1), int(c1)))
    return layers


def _valid_strip_rows(g: np.ndarray) -> list[tuple[int, list[int]]]:
    fg_colors = {int(v) for v in g.flat if int(v) != BG}
    assert len(fg_colors) >= 2, "expected at least two layer colors"

    valid: list[tuple[int, list[int]]] = []
    off_masks = np.ones(g.shape, dtype=bool)
    for row in range(g.shape[0]):
        order = _row_order(g, row, fg_colors)
        if order is None:
            continue
        layers = _reconstruct_layers(g, row, order)
        if layers is None:
            continue
        base = _compose(g.shape, layers, mirror_even=False)
        off_masks[:] = True
        off_masks[row, :] = False
        if np.array_equal(base[off_masks], g[off_masks]):
            valid.append((row, order))
    return valid


def main(grid: np.ndarray) -> np.ndarray:
    """
    Separate stacked solid rectangles using a one-row layer-order strip, mirror every even layer horizontally, and recompose the scene.

    Input structure:
    - The background is black.
    - Each foreground color belongs to exactly one solid axis-aligned rectangular layer.
    - Layers overlap in a top-view stack, so lower rectangles may be partially hidden.
    - One special row contains exactly one pixel of every foreground color from left to right, encoding the layer order from bottom to top.

    Rule:
    - Identify the layer-order strip row and read its non-black pixels from left to right.
    - For each color, ignore the strip row and take the bounding box of all remaining visible pixels as the full rectangle.
    - Reconstruct every rectangle as a solid block of its color.
    - Keep odd-numbered layers in place and mirror every even-numbered layer across the grid's vertical midline.
    - Draw the reconstructed layers from bottom to top on a black canvas; the strip row is not copied into the output.

    Key insight:
    - The single-pixel color strip gives the hidden stacking order, while each color's visible fragments still preserve the true rectangle bbox needed to repair occluded parts before the parity-based mirror step.
    """
    g = np.asarray(grid, dtype=int)
    assert g.ndim == 2

    candidates = _valid_strip_rows(g)
    assert candidates, "failed to identify the layer-order strip"

    edge_candidates = [item for item in candidates if item[0] in {0, g.shape[0] - 1}]
    strip_row, order = (edge_candidates or candidates)[0]
    layers = _reconstruct_layers(g, strip_row, order)
    assert layers is not None
    return _compose(g.shape, layers, mirror_even=True)


# --------------- generator ---------------

# Grid and layer bounds.
GRID_H_RANGE = (12, 24)
GRID_W_RANGE = (12, 26)
LAYER_COUNT_RANGE = (3, 6)
RECT_H_RANGE = (2, 8)
RECT_W_RANGE = (2, 9)

# Placement and retry limits.
PLACE_TRIES_PER_LAYER = 40
GENERATION_RETRIES = 400
OVERLAP_PROBABILITY = 0.7


def _overlap_area(a: _Layer, b: _Layer) -> int:
    height = min(a.r1, b.r1) - max(a.r0, b.r0) + 1
    width = min(a.c1, b.c1) - max(a.c0, b.c0) + 1
    if height <= 0 or width <= 0:
        return 0
    return int(height * width)


def _visible_masks(shape: tuple[int, int], layers: list[_Layer]) -> list[np.ndarray]:
    covered = np.zeros(shape, dtype=bool)
    visible: list[np.ndarray] = [np.zeros(shape, dtype=bool) for _ in layers]
    for idx in range(len(layers) - 1, -1, -1):
        full = _rect_mask(shape, layers[idx])
        visible[idx] = full & ~covered
        covered |= full
    return visible


def _bbox_preserved(layer: _Layer, visible: np.ndarray) -> bool:
    if not visible.any():
        return False
    top = visible[layer.r0, layer.c0 : layer.c1 + 1].any()
    bottom = visible[layer.r1, layer.c0 : layer.c1 + 1].any()
    left = visible[layer.r0 : layer.r1 + 1, layer.c0].any()
    right = visible[layer.r0 : layer.r1 + 1, layer.c1].any()
    return bool(top and bottom and left and right)


def _layout_ok(shape: tuple[int, int], layers: list[_Layer]) -> bool:
    visible = _visible_masks(shape, layers)
    return all(_bbox_preserved(layer, vis) for layer, vis in zip(layers, visible))


def _sample_layer(
    color: int,
    width: int,
    row_lo: int,
    row_hi: int,
    prior: list[_Layer],
    force_overlap: bool,
) -> _Layer | None:
    max_h = min(RECT_H_RANGE[1], row_hi - row_lo + 1)
    max_w = min(RECT_W_RANGE[1], width - 2)
    if max_h < RECT_H_RANGE[0] or max_w < RECT_W_RANGE[0]:
        return None

    for _ in range(PLACE_TRIES_PER_LAYER):
        h = random.randint(RECT_H_RANGE[0], max_h)
        w = random.randint(RECT_W_RANGE[0], max_w)
        r0_lo = row_lo
        r0_hi = row_hi - h + 1
        c0_lo = 1
        c0_hi = width - w - 1
        if r0_lo > r0_hi or c0_lo > c0_hi:
            continue

        target = None
        if prior and (force_overlap or random.random() < OVERLAP_PROBABILITY):
            target = random.choice(prior)
            r0_lo = max(r0_lo, target.r0 - h + 1)
            r0_hi = min(r0_hi, target.r1)
            c0_lo = max(c0_lo, target.c0 - w + 1)
            c0_hi = min(c0_hi, target.c1)
            if r0_lo > r0_hi or c0_lo > c0_hi:
                target = None
                r0_lo = row_lo
                r0_hi = row_hi - h + 1
                c0_lo = 1
                c0_hi = width - w - 1

        r0 = random.randint(r0_lo, r0_hi)
        c0 = random.randint(c0_lo, c0_hi)
        layer = _Layer(int(color), int(r0), int(c0), int(r0 + h - 1), int(c0 + w - 1))
        if target is not None and _overlap_area(layer, target) == 0:
            continue
        return layer
    return None


def _sample_strip_positions(width: int, count: int) -> list[int]:
    usable = list(range(1, width - 1))
    return sorted(random.sample(usable, count))


def _gen_attempt() -> np.ndarray | None:
    height = random.randint(*GRID_H_RANGE)
    width = random.randint(*GRID_W_RANGE)
    strip_on_top = random.random() < 0.5
    strip_row = 0 if strip_on_top else height - 1
    row_lo = 2 if strip_on_top else 1
    row_hi = height - 2 if strip_on_top else height - 3
    if row_hi - row_lo + 1 < RECT_H_RANGE[0]:
        return None

    layer_count = random.randint(LAYER_COUNT_RANGE[0], min(LAYER_COUNT_RANGE[1], len(PALETTE), width - 2))
    colors = random.sample(PALETTE, layer_count)

    layers: list[_Layer] = []
    for idx, color in enumerate(colors):
        force_overlap = idx > 0 and idx == 1
        layer = _sample_layer(color, width, row_lo, row_hi, layers, force_overlap)
        if layer is None:
            return None
        trial_layers = layers + [layer]
        if not _layout_ok((height, width), trial_layers):
            # A few immediate rejections avoid hopeless dense layouts.
            for _ in range(PLACE_TRIES_PER_LAYER):
                layer = _sample_layer(color, width, row_lo, row_hi, layers, force_overlap)
                if layer is None:
                    break
                trial_layers = layers + [layer]
                if _layout_ok((height, width), trial_layers):
                    break
            else:
                return None
            if not _layout_ok((height, width), trial_layers):
                return None
        layers = trial_layers

    visible = _visible_masks((height, width), layers)
    occluded = [
        idx
        for idx, (layer, vis) in enumerate(zip(layers, visible))
        if idx < len(layers) - 1 and int(vis.sum()) < (layer.r1 - layer.r0 + 1) * (layer.c1 - layer.c0 + 1)
    ]
    if not occluded:
        return None

    base = _compose((height, width), layers, mirror_even=False)
    transformed = _compose((height, width), layers, mirror_even=True)
    if not np.any(base != transformed):
        return None

    strip_cols = _sample_strip_positions(width, layer_count)
    inp = base.copy()
    inp[strip_row, :] = BG
    for col, color in zip(strip_cols, colors):
        inp[strip_row, col] = int(color)

    valid_rows = _valid_strip_rows(inp)
    if len(valid_rows) != 1 or valid_rows[0][0] != strip_row:
        return None

    try:
        got = main(inp)
    except Exception:
        return None
    if not np.array_equal(got, transformed):
        return None
    return inp


def generate_input() -> np.ndarray:
    for _ in range(GENERATION_RETRIES):
        grid = _gen_attempt()
        if grid is not None:
            return grid
    raise RuntimeError("failed to generate valid layered-rectangle input")
