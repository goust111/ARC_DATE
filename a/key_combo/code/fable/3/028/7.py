from __future__ import annotations

from collections import Counter
import random

import numpy as np


Coord = tuple[int, int]


def _num_layers(shape: tuple[int, int]) -> int:
    h, w = map(int, shape)
    assert h > 0 and w > 0
    return (min(h, w) + 1) // 2


def _ring_mask(shape: tuple[int, int], layer: int) -> np.ndarray:
    h, w = map(int, shape)
    k = int(layer)
    r0, c0 = k, k
    r1, c1 = h - 1 - k, w - 1 - k
    assert r0 <= r1 and c0 <= c1

    mask = np.zeros((h, w), dtype=bool)
    mask[r0, c0 : c1 + 1] = True
    mask[r1, c0 : c1 + 1] = True
    mask[r0 : r1 + 1, c0] = True
    mask[r0 : r1 + 1, c1] = True
    return mask


def _ring_cells(shape: tuple[int, int], layer: int) -> list[Coord]:
    return [tuple(map(int, rc)) for rc in np.argwhere(_ring_mask(shape, layer))]


def _analyze_layers(g: np.ndarray) -> tuple[list[np.ndarray], list[int], dict[int, int]]:
    masks: list[np.ndarray] = []
    main_colors: list[int] = []
    marker_colors: dict[int, int] = {}

    for layer in range(_num_layers(g.shape)):
        mask = _ring_mask(g.shape, layer)
        values = [int(v) for v in g[mask]]
        counts = Counter(values)
        ranked = counts.most_common()

        if len(ranked) == 1:
            main_color = int(ranked[0][0])
        else:
            assert len(ranked) == 2, "a participating ring must contain exactly one marker color"
            (main_color, main_count), (marker_color, marker_count) = ranked
            assert main_count > marker_count, "ring main color must be the unique majority"
            assert marker_count == 1, "each participating ring must contain one marker pixel"
            assert main_count == len(values) - 1, "all non-marker cells in a ring must share one color"
            main_color = int(main_color)
            marker_colors[layer] = int(marker_color)

        masks.append(mask)
        main_colors.append(main_color)

    assert len(set(main_colors)) == len(main_colors), "ring main colors must be distinct"
    return masks, main_colors, marker_colors


def _arrows_from_markers(main_colors: list[int], marker_colors: dict[int, int]) -> dict[int, int]:
    color_to_layer = {color: layer for layer, color in enumerate(main_colors)}
    arrows: dict[int, int] = {}

    for src, marker_color in marker_colors.items():
        assert marker_color in color_to_layer, "marker color must be another ring's main color"
        dst = color_to_layer[marker_color]
        assert dst != src, "a marker must name a different ring"
        arrows[int(src)] = int(dst)

    assert arrows, "expected at least one directed cycle"
    sources = set(arrows)
    targets = set(arrows.values())
    assert len(targets) == len(arrows), "each participating ring must have one incoming arrow"
    assert sources == targets, "participating arrows must close into disjoint directed cycles"

    seen: set[int] = set()
    for start in sorted(sources):
        if start in seen:
            continue
        path: list[int] = []
        cur = start
        while cur not in path:
            assert cur in arrows, "cycle left the participating set"
            path.append(cur)
            cur = arrows[cur]
        assert cur == start, "each arrow component must be a directed cycle"
        seen.update(path)

    assert seen == sources
    return arrows


def main(grid: np.ndarray) -> np.ndarray:
    """Pass colors backward around marker-defined cycles on concentric rectangular rings.

    Input structure:
    - The grid is fully tiled by one-cell-thick concentric rectangular layers.
    - Each layer has a distinct majority main color.
    - Some layers contain exactly one minority marker pixel whose color is the
      main color of another layer; layers without such a marker do not
      participate.

    Rule:
    - A marker in layer X with layer Y's color means X sends its old main color
      to Y.
    - The marker arrows form disjoint directed cycles, so every participating
      layer has exactly one incoming arrow.
    - Repaint every participating layer with the old main color of its incoming
      predecessor; repaint every non-participating layer with its own old main
      color, removing all marker pixels.

    Key insight:
    - The minority pixel names the outgoing destination, not the replacement
      color; only following the whole directed cycle and taking colors from the
      inverse arrow gives the correct simultaneous recoloring.
    """
    g = np.asarray(grid, dtype=int)
    assert g.ndim == 2, "expected a 2D grid"

    masks, main_colors, marker_colors = _analyze_layers(g)
    arrows = _arrows_from_markers(main_colors, marker_colors)
    incoming = {dst: src for src, dst in arrows.items()}

    out = g.copy()
    for layer, mask in enumerate(masks):
        if layer in incoming:
            new_color = main_colors[incoming[layer]]
        else:
            new_color = main_colors[layer]
        out[mask] = int(new_color)
    return out


# --------------- generator (o5x) ---------------

ARC_COLORS = tuple(range(10))
LAYER_RANGE = (6, 9)
EXTRA_SIDE_RANGE = (0, 12)
LONG_CYCLE_MAX = 5
GENERATION_RETRIES = 300


def _paint_pure_layers(shape: tuple[int, int], colors: list[int]) -> np.ndarray:
    grid = np.full(shape, -1, dtype=int)
    for layer, color in enumerate(colors):
        grid[_ring_mask(shape, layer)] = int(color)
    assert np.all(grid >= 0)
    return grid


def _make_shape(layer_count: int) -> tuple[int, int]:
    short_side = 2 * int(layer_count)
    long_side = short_side + random.randint(*EXTRA_SIDE_RANGE)
    long_side = min(30, long_side)
    if random.random() < 0.5:
        return short_side, long_side
    return long_side, short_side


def _make_cycles(layer_count: int) -> list[list[int]]:
    layers = list(range(layer_count))
    random.shuffle(layers)

    max_long = min(LONG_CYCLE_MAX, layer_count - 3)
    long_len = random.randint(3, max_long)
    cycles = [layers[:2], layers[2 : 2 + long_len]]
    used = 2 + long_len

    spare = layers[used:]
    if len(spare) >= 3 and random.random() < 0.35:
        extra_len = random.randint(2, len(spare) - 1)
        cycles.append(spare[:extra_len])

    for cycle in cycles:
        random.shuffle(cycle)

    assert any(len(cycle) == 2 for cycle in cycles)
    assert any(len(cycle) >= 3 for cycle in cycles)
    assert sum(len(cycle) for cycle in cycles) < layer_count
    return cycles


def _arrows_from_cycles(cycles: list[list[int]]) -> dict[int, int]:
    arrows: dict[int, int] = {}
    for cycle in cycles:
        for i, src in enumerate(cycle):
            arrows[int(src)] = int(cycle[(i + 1) % len(cycle)])
    return arrows


def _marker_cells(shape: tuple[int, int], layer: int) -> list[Coord]:
    h, w = shape
    k = int(layer)
    r0, c0 = k, k
    r1, c1 = h - 1 - k, w - 1 - k

    preferred: list[Coord] = []
    if c1 - c0 >= 4:
        preferred.extend((r0, c) for c in range(c0 + 1, c1))
        preferred.extend((r1, c) for c in range(c0 + 1, c1))
    if r1 - r0 >= 4:
        preferred.extend((r, c0) for r in range(r0 + 1, r1))
        preferred.extend((r, c1) for r in range(r0 + 1, r1))

    if preferred:
        return preferred
    return _ring_cells(shape, layer)


def _place_markers(grid: np.ndarray, colors: list[int], arrows: dict[int, int]) -> None:
    for src, dst in arrows.items():
        r, c = random.choice(_marker_cells(grid.shape, src))
        grid[r, c] = int(colors[dst])


def _expected_from_plan(
    shape: tuple[int, int],
    colors: list[int],
    arrows: dict[int, int],
) -> np.ndarray:
    incoming = {dst: src for src, dst in arrows.items()}
    out = np.full(shape, -1, dtype=int)
    for layer, old_color in enumerate(colors):
        color = colors[incoming[layer]] if layer in incoming else old_color
        out[_ring_mask(shape, layer)] = int(color)
    assert np.all(out >= 0)
    return out


def _direct_marker_recolor(
    shape: tuple[int, int],
    colors: list[int],
    arrows: dict[int, int],
) -> np.ndarray:
    out = np.full(shape, -1, dtype=int)
    for layer, old_color in enumerate(colors):
        color = colors[arrows[layer]] if layer in arrows else old_color
        out[_ring_mask(shape, layer)] = int(color)
    assert np.all(out >= 0)
    return out


def _gen_attempt() -> np.ndarray | None:
    layer_count = random.randint(*LAYER_RANGE)
    shape = _make_shape(layer_count)
    if max(shape) > 30 or _num_layers(shape) != layer_count:
        return None

    colors = random.sample(ARC_COLORS, layer_count)
    cycles = _make_cycles(layer_count)
    arrows = _arrows_from_cycles(cycles)
    if len(arrows) >= layer_count:
        return None

    grid = _paint_pure_layers(shape, colors)
    _place_markers(grid, colors, arrows)

    expected = _expected_from_plan(shape, colors, arrows)
    direct = _direct_marker_recolor(shape, colors, arrows)
    if np.array_equal(direct, expected):
        return None
    if np.array_equal(grid, expected):
        return None

    try:
        actual = main(grid)
    except Exception:
        return None
    if not np.array_equal(actual, expected):
        return None

    masks, found_colors, marker_colors = _analyze_layers(grid)
    found_arrows = _arrows_from_markers(found_colors, marker_colors)
    if found_arrows != arrows:
        return None
    if not all(np.count_nonzero(mask) >= 4 for mask in masks):
        return None

    return grid


def generate_input() -> np.ndarray:
    for _ in range(GENERATION_RETRIES):
        grid = _gen_attempt()
        if grid is not None:
            return grid
    raise RuntimeError("failed to generate a valid ring transport-cycle input")
