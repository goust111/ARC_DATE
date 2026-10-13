from __future__ import annotations

import random

import numpy as np


PALETTE = tuple(range(1, 10))
MAX_GENERATION_ATTEMPTS = 400


def _ring_coords(shape: tuple[int, int], layer: int) -> list[tuple[int, int]]:
    height, width = map(int, shape)
    r0 = c0 = int(layer)
    r1 = height - 1 - int(layer)
    c1 = width - 1 - int(layer)
    assert 0 <= r0 <= r1 < height
    assert 0 <= c0 <= c1 < width

    coords = [(r0, c) for c in range(c0, c1 + 1)]
    coords.extend((r, c1) for r in range(r0 + 1, r1 + 1))
    if r1 > r0:
        coords.extend((r1, c) for c in range(c1 - 1, c0 - 1, -1))
    if c1 > c0:
        coords.extend((r, c0) for r in range(r1 - 1, r0, -1))
    return coords


def _ring_noncorner_coords(shape: tuple[int, int], layer: int) -> list[tuple[int, int]]:
    height, width = map(int, shape)
    r0 = c0 = int(layer)
    r1 = height - 1 - int(layer)
    c1 = width - 1 - int(layer)
    corners = {(r0, c0), (r0, c1), (r1, c0), (r1, c1)}
    return [coord for coord in _ring_coords(shape, layer) if coord not in corners]


def _num_layers(shape: tuple[int, int]) -> int:
    return (min(map(int, shape)) + 1) // 2


def _first_occurrence_order(colors: list[int]) -> list[int]:
    order: list[int] = []
    for color in colors:
        if color not in order:
            order.append(int(color))
    return order


def _layer_info(grid: np.ndarray, layer: int) -> dict[str, object]:
    coords = _ring_coords(grid.shape, layer)
    values = [int(grid[r, c]) for r, c in coords]
    nonblack = sorted({value for value in values if value != 0})
    has_gap = any(value == 0 for value in values)

    if has_gap:
        assert len(nonblack) <= 1, "a gapped ring may only contain one surviving non-black color"
        color = nonblack[0] if nonblack else 0
    else:
        assert len(nonblack) == 1, "a complete ring must be monochromatic"
        color = nonblack[0]

    return {
        "coords": coords,
        "color": int(color),
        "has_gap": bool(has_gap),
        "gap_count": int(sum(value == 0 for value in values)),
    }


def _apply_rule(grid: np.ndarray) -> np.ndarray:
    g = np.asarray(grid, dtype=int)
    out = g.copy()
    infos = [_layer_info(g, layer) for layer in range(_num_layers(g.shape))]

    cycle_order: list[int] = []
    for info in infos:
        if not info["has_gap"]:
            color = int(info["color"])
            if color not in cycle_order:
                cycle_order.append(color)

    mapping = {color: cycle_order[i - 1] for i, color in enumerate(cycle_order)}

    for info in infos:
        coords = info["coords"]
        if info["has_gap"]:
            for r, c in coords:
                out[r, c] = 0
        else:
            new_color = mapping[int(info["color"])]
            for r, c in coords:
                out[r, c] = new_color

    return out


def main(grid: np.ndarray) -> np.ndarray:
    """
    Erase every concentric ring that contains a black gap, then cyclically recolor the surviving rings by their outside-in distinct-color order.

    Input structure:
    - The whole grid is tiled by one-cell-thick concentric rectangular rings defined by layer index min(r, c, H-1-r, W-1-c).
    - Each ring is monochromatic except that 1-2 rings may contain a few black gap pixels.
    - Gap pixels never define a separate object; they only break the completeness of their own ring.

    Rule:
    - Inspect each ring in the layer-index coordinate system.
    - If any pixel on a ring is black, erase that entire ring to black.
    - Among the remaining complete rings, read the distinct non-black colors in first outside-to-inside appearance order and map each color to the previous color in that cyclic list.
    - Recolor every surviving ring with that cyclic predecessor while leaving erased rings black.

    Key insight:
    - The black gaps only decide which layers survive; the color cycle is computed after that pruning, so removing one ring can change the recoloring of every surviving ring.
    """
    g = np.asarray(grid, dtype=int)
    assert g.ndim == 2 and g.size > 0, "expected a non-empty 2D grid"
    return _apply_rule(g)


def _choose_shape() -> tuple[int, int]:
    num_layers = random.choice((5, 6, 6, 7, 7))
    side_a = random.choice((2 * num_layers - 1, 2 * num_layers))
    side_b = random.choice((2 * num_layers - 1, 2 * num_layers))
    return side_a, side_b


def _choose_gap_layers(shape: tuple[int, int]) -> list[int]:
    num_layers = _num_layers(shape)
    eligible = [layer for layer in range(num_layers) if _ring_noncorner_coords(shape, layer)]
    assert 0 in eligible, "the outer ring must support non-corner gaps"

    gap_layers = [0]
    second_choices = [layer for layer in eligible if layer >= 2]
    if second_choices and random.random() < 0.5:
        gap_layers.append(random.choice(second_choices))
    return sorted(gap_layers)


def _build_complete_color_sequence(count: int) -> list[int]:
    assert count >= 3
    if count == 3:
        unique_count = 2
    elif count == 4:
        unique_count = random.choice((2, 3))
    else:
        unique_count = random.choice((3, 4))
    unique_count = min(unique_count, count - 1)

    unique_colors = random.sample(PALETTE, unique_count)
    offset = random.randrange(unique_count)
    base_order = unique_colors[offset:] + unique_colors[:offset]
    return [int(base_order[i % unique_count]) for i in range(count)]


def _assign_ring_colors(shape: tuple[int, int], gap_layers: list[int]) -> list[int]:
    num_layers = _num_layers(shape)
    complete_positions = [layer for layer in range(num_layers) if layer not in gap_layers]
    complete_colors = _build_complete_color_sequence(len(complete_positions))
    complete_order = _first_occurrence_order(complete_colors)
    assert len(complete_order) >= 2

    ring_colors = [-1] * num_layers
    for layer, color in zip(complete_positions, complete_colors, strict=False):
        ring_colors[layer] = int(color)

    ring_colors[0] = int(complete_order[-1])
    for layer in gap_layers:
        if layer == 0:
            continue
        extra_choices = [color for color in PALETTE if color not in complete_order]
        if extra_choices and random.random() < 0.5:
            pool = extra_choices
        else:
            pool = complete_order
        forbidden = {
            ring_colors[neighbor]
            for neighbor in (layer - 1, layer + 1)
            if 0 <= neighbor < num_layers and ring_colors[neighbor] != -1
        }
        filtered = [color for color in pool if color not in forbidden]
        ring_colors[layer] = int(random.choice(filtered or pool))

    assert all(color in PALETTE for color in ring_colors)
    return ring_colors


def _paint_rings(shape: tuple[int, int], ring_colors: list[int]) -> np.ndarray:
    grid = np.zeros(shape, dtype=int)
    for layer, color in enumerate(ring_colors):
        for r, c in _ring_coords(shape, layer):
            grid[r, c] = int(color)
    return grid


def _choose_gap_pixels(candidates: list[tuple[int, int]], count: int) -> list[tuple[int, int]]:
    shuffled = candidates[:]
    random.shuffle(shuffled)
    chosen: list[tuple[int, int]] = []

    for cell in shuffled:
        if all(abs(cell[0] - other[0]) + abs(cell[1] - other[1]) >= 2 for other in chosen):
            chosen.append(cell)
            if len(chosen) == count:
                return chosen

    for cell in shuffled:
        if cell not in chosen:
            chosen.append(cell)
            if len(chosen) == count:
                return chosen

    raise AssertionError("not enough candidate gap cells")


def _carve_gaps(grid: np.ndarray, gap_layers: list[int]) -> None:
    for layer in gap_layers:
        candidates = _ring_noncorner_coords(grid.shape, layer)
        assert candidates, "every chosen gap layer must have non-corner cells"
        gap_count = random.randint(1, min(3, len(candidates)))
        for r, c in _choose_gap_pixels(candidates, gap_count):
            grid[r, c] = 0


def _validate_generated(grid: np.ndarray) -> None:
    infos = [_layer_info(grid, layer) for layer in range(_num_layers(grid.shape))]
    assert 5 <= len(infos) <= 7

    gap_layers = [layer for layer, info in enumerate(infos) if info["has_gap"]]
    complete_colors = [int(info["color"]) for info in infos if not info["has_gap"]]
    gap_colors = [int(info["color"]) for info in infos if info["has_gap"]]

    assert 1 <= len(gap_layers) <= 2
    assert all(gap_layers[i + 1] - gap_layers[i] >= 2 for i in range(len(gap_layers) - 1))
    assert len(set(complete_colors)) >= 2
    assert len(complete_colors) > len(set(complete_colors))
    assert set(gap_colors) & set(complete_colors)

    overall_order = _first_occurrence_order([int(info["color"]) for info in infos])
    surviving_order = _first_occurrence_order(complete_colors)
    assert overall_order != surviving_order
    assert all(
        int(infos[layer]["color"]) != int(infos[layer + 1]["color"])
        for layer in range(len(infos) - 1)
    )

    for layer, info in enumerate(infos):
        if info["has_gap"]:
            assert 1 <= int(info["gap_count"]) <= 3
            gap_cells = [
                coord for coord in _ring_coords(grid.shape, layer) if int(grid[coord]) == 0
            ]
            assert all(coord in _ring_noncorner_coords(grid.shape, layer) for coord in gap_cells)
        else:
            assert int(info["gap_count"]) == 0

    output = _apply_rule(grid)
    assert output.shape == grid.shape
    assert np.any(output != grid), "the sample should exhibit a visible transformation"


def _attempt_generate() -> np.ndarray | None:
    shape = _choose_shape()
    gap_layers = _choose_gap_layers(shape)
    ring_colors = _assign_ring_colors(shape, gap_layers)
    grid = _paint_rings(shape, ring_colors)
    _carve_gaps(grid, gap_layers)

    try:
        _validate_generated(grid)
    except AssertionError:
        return None
    return grid


def generate_input() -> np.ndarray:
    for _ in range(MAX_GENERATION_ATTEMPTS):
        grid = _attempt_generate()
        if grid is not None:
            return grid
    raise RuntimeError("failed to generate a valid missing-ring color-cycle input")
