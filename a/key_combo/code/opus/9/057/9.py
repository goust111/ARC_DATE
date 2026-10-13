from __future__ import annotations

from collections import Counter
import random

import numpy as np


BLACK = 0
BLUE = 1
RED = 2
GREEN = 3
YELLOW = 4
GREY = 5
PALETTE_POOL = (BLUE, GREEN, YELLOW, GREY)


def _in_bounds(shape: tuple[int, int], r: int, c: int) -> bool:
    return 0 <= r < shape[0] and 0 <= c < shape[1]


def _layer_positions(shape: tuple[int, int], center: tuple[int, int], radius: int) -> list[tuple[int, int]]:
    cr, cc = center
    pts: set[tuple[int, int]] = set()
    for dc in range(-radius, radius + 1):
        for rr in (cr - radius, cr + radius):
            cc2 = cc + dc
            if _in_bounds(shape, rr, cc2):
                pts.add((rr, cc2))
    for dr in range(-radius + 1, radius):
        for cc2 in (cc - radius, cc + radius):
            rr = cr + dr
            if _in_bounds(shape, rr, cc2):
                pts.add((rr, cc2))
    return sorted(pts)


def _nearest_layer(chebyshev_dist: int, spacing: int) -> int:
    return int(np.floor(chebyshev_dist / spacing + 0.5))


def _seed_positions(center: tuple[int, int], spacing: int, slope: int) -> list[tuple[int, int]]:
    cr, cc = center
    return [
        (cr - spacing, cc - slope * spacing),
        (cr, cc),
        (cr + spacing, cc + slope * spacing),
    ]


def _extract_seed_geometry(grid: np.ndarray) -> tuple[tuple[int, int], int]:
    seeds = [tuple(map(int, p)) for p in np.argwhere(grid == RED)]
    assert len(seeds) == 3, "expected exactly three red seed pixels"

    for center in seeds:
        others = [p for p in seeds if p != center]
        if len(others) != 2:
            continue
        (r0, c0), (r1, c1) = others
        cr, cc = center
        if r0 + r1 != 2 * cr or c0 + c1 != 2 * cc:
            continue
        d0 = max(abs(r0 - cr), abs(c0 - cc))
        d1 = max(abs(r1 - cr), abs(c1 - cc))
        if d0 == d1 and d0 > 0 and abs(r0 - cr) == abs(c0 - cc) and abs(r1 - cr) == abs(c1 - cc):
            return center, d0

    raise AssertionError("red seeds do not form one equally spaced 45-degree diagonal triplet")


def _visible_layer_major_colors(grid: np.ndarray, center: tuple[int, int], spacing: int) -> dict[int, int]:
    majors: dict[int, int] = {}
    for k in sorted(
        {
            _nearest_layer(max(abs(int(r) - center[0]), abs(int(c) - center[1])), spacing)
            for r, c in np.argwhere((grid != BLACK) & (grid != RED))
        }
    ):
        if k <= 0:
            continue
        colors = [
            int(grid[r, c])
            for r, c in np.argwhere((grid != BLACK) & (grid != RED))
            if _nearest_layer(max(abs(int(r) - center[0]), abs(int(c) - center[1])), spacing) == k
        ]
        assert colors, "each visible layer must retain at least one non-seed pixel"
        majors[k] = Counter(colors).most_common(1)[0][0]
    assert majors, "expected at least one visible square layer"
    return majors


def _infer_period(major_by_layer: dict[int, int]) -> tuple[int, dict[int, int]]:
    max_layer = max(major_by_layer)
    for period in range(1, max_layer + 1):
        residue_to_color: dict[int, int] = {}
        ok = True
        for k, color in sorted(major_by_layer.items()):
            residue = k % period
            if residue in residue_to_color and residue_to_color[residue] != color:
                ok = False
                break
            residue_to_color[residue] = color
        if ok and len(residue_to_color) == period:
            return period, residue_to_color
    raise AssertionError("no consistent minimal color period found")


def main(grid: np.ndarray) -> np.ndarray:
    """Repair erased pixels on concentric square borders by inferring the color cycle over layer indices.

    Input structure:
    - A mostly black grid contains exactly three red seed pixels on one equally spaced 45-degree diagonal.
    - The middle red seed is the center of several concentric square borders, and the outer two seeds fix the spacing.
    - Non-seed border pixels use colors from a small repeating palette, but some of those border pixels are erased to black.

    Rule:
    - Use the three red seeds to recover the center and spacing d.
    - Group every visible non-red border pixel by its layer index k, where k is the nearest multiple of d under Chebyshev distance from the center.
    - Infer the smallest color period whose residues k mod P explain the visible layer-majority colors.
    - For every observed layer, restore black pixels on that full square border with the residue color, while keeping the three red seed pixels red.

    Key insight:
    - Black cells are ambiguous until geometry identifies which square-border layer they belong to; only then can the repeating color rule over layer numbers recover the missing color.
    """
    g = np.asarray(grid, dtype=int)
    assert g.ndim == 2

    center, spacing = _extract_seed_geometry(g)
    major_by_layer = _visible_layer_major_colors(g, center, spacing)
    period, residue_to_color = _infer_period(major_by_layer)

    out = g.copy()
    for k in range(1, max(major_by_layer) + 1):
        color = residue_to_color[k % period]
        for r, c in _layer_positions(g.shape, center, k * spacing):
            if out[r, c] == BLACK:
                out[r, c] = color

    for r, c in np.argwhere(g == RED):
        out[int(r), int(c)] = RED
    return out


def _choose_geometry() -> tuple[tuple[int, int], tuple[int, int], int, int, int]:
    while True:
        spacing = random.choices([1, 2, 3], weights=[1, 4, 2], k=1)[0]
        max_layers = min(7, 14 // spacing)
        layers = random.randint(4, max_layers)
        outer_radius = layers * spacing
        min_side = max(17, 2 * outer_radius + 1)
        if min_side > 29:
            continue
        height = random.randint(min_side, min(29, min_side + 4))
        width = random.randint(min_side, min(29, min_side + 4))
        center = (
            random.randint(outer_radius, height - outer_radius - 1),
            random.randint(outer_radius, width - outer_radius - 1),
        )
        slope = random.choice([-1, 1])
        return (height, width), center, spacing, layers, slope


def _residue_colors(period: int) -> dict[int, int]:
    seq = random.sample(PALETTE_POOL, period)
    return {k % period: seq[k - 1] for k in range(1, period + 1)}


def _build_output(
    shape: tuple[int, int],
    center: tuple[int, int],
    spacing: int,
    layers: int,
    residue_to_color: dict[int, int],
    slope: int,
) -> np.ndarray:
    grid = np.zeros(shape, dtype=int)
    period = len(residue_to_color)
    for k in range(1, layers + 1):
        color = residue_to_color[k % period]
        for r, c in _layer_positions(shape, center, k * spacing):
            grid[r, c] = color
    for r, c in _seed_positions(center, spacing, slope):
        grid[r, c] = RED
    return grid


def _erase_layer_pixels(
    grid: np.ndarray,
    center: tuple[int, int],
    spacing: int,
    layer: int,
    protected: set[tuple[int, int]],
) -> int:
    radius = layer * spacing
    positions = [p for p in _layer_positions(grid.shape, center, radius) if p not in protected]
    keep_min = max(1, int(np.ceil(0.30 * len(positions))))
    erase_max = len(positions) - keep_min
    if erase_max <= 0:
        return 0

    if random.random() < 0.2:
        erase_count = 0
    else:
        erase_count = random.randint(max(1, len(positions) // 8), max(1, min(erase_max, (3 * len(positions)) // 5)))

    if erase_count == 0:
        return 0

    side_groups: list[list[tuple[int, int]]] = []
    cr, cc = center
    side_groups.append([(r, c) for r, c in positions if r == cr - radius])
    side_groups.append([(r, c) for r, c in positions if r == cr + radius])
    side_groups.append([(r, c) for r, c in positions if c == cc - radius and r not in (cr - radius, cr + radius)])
    side_groups.append([(r, c) for r, c in positions if c == cc + radius and r not in (cr - radius, cr + radius)])
    side_groups = [group for group in side_groups if group]
    random.shuffle(side_groups)

    chosen: set[tuple[int, int]] = set()
    for group in side_groups:
        if len(chosen) >= erase_count:
            break
        random.shuffle(group)
        take = min(len(group), random.randint(0, max(1, min(len(group), erase_count - len(chosen)))))
        for cell in group[:take]:
            chosen.add(cell)
            if len(chosen) >= erase_count:
                break

    leftovers = [p for p in positions if p not in chosen]
    random.shuffle(leftovers)
    for cell in leftovers:
        if len(chosen) >= erase_count:
            break
        chosen.add(cell)

    for r, c in chosen:
        grid[r, c] = BLACK
    return len(chosen)


def _validate_instance(
    inp: np.ndarray,
    out: np.ndarray,
    center: tuple[int, int],
    spacing: int,
    layers: int,
    period: int,
    residue_to_color: dict[int, int],
) -> bool:
    if not np.array_equal(main(inp), out):
        return False
    if np.array_equal(inp, out):
        return False

    seed_count = int(np.count_nonzero(inp == RED))
    if seed_count != 3:
        return False
    if _extract_seed_geometry(inp) != (center, spacing):
        return False

    major_by_layer = _visible_layer_major_colors(inp, center, spacing)
    inferred_period, inferred_map = _infer_period(major_by_layer)
    if inferred_period != period:
        return False
    if inferred_map != residue_to_color:
        return False

    seen_residues = {k % period for k in major_by_layer}
    if seen_residues != set(residue_to_color):
        return False

    for layer in range(1, layers + 1):
        border = _layer_positions(inp.shape, center, layer * spacing)
        nonseed = [p for p in border if out[p] != RED]
        visible = sum(inp[p] != BLACK for p in nonseed)
        if visible < int(np.ceil(0.30 * len(nonseed))):
            return False
        layer_colors = {int(out[p]) for p in nonseed}
        if layer_colors != {residue_to_color[layer % period]}:
            return False
        if not any(inp[p] != BLACK for p in nonseed):
            return False

    input_colors = set(map(int, np.unique(inp)))
    if not input_colors.issubset({BLACK, RED, *PALETTE_POOL}):
        return False
    return True


def generate_input() -> np.ndarray:
    for _ in range(400):
        shape, center, spacing, layers, slope = _choose_geometry()
        period = random.choice([2, 3])
        residue_to_color = _residue_colors(period)
        out = _build_output(shape, center, spacing, layers, residue_to_color, slope)
        inp = out.copy()

        protected = set(_seed_positions(center, spacing, slope))
        erased_total = 0
        for layer in range(1, layers + 1):
            erased_total += _erase_layer_pixels(inp, center, spacing, layer, protected)

        if erased_total == 0:
            continue
        if _validate_instance(inp, out, center, spacing, layers, period, residue_to_color):
            return inp

    raise RuntimeError("failed to generate a valid concentric-period-repair instance")
