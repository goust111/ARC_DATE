from __future__ import annotations

import random
import sys
from functools import lru_cache
from itertools import product
from pathlib import Path

import numpy as np

UTILS_DIR = Path("/home/lixinyu/snap/new/solvers")
if str(UTILS_DIR) not in sys.path:
    sys.path.append(str(UTILS_DIR))

from arc_utils import Color, in_bounds, neighbors4


MASK_COLOR = Color.PINK
RING_PALETTE = [
    Color.BLUE,
    Color.RED,
    Color.GREEN,
    Color.YELLOW,
    Color.ORANGE,
    Color.TEAL,
    Color.MAROON,
]

ACTUAL_WIDTH_SEQUENCES = [
    (1, 2, 1, 1, 2),
    (2, 1, 2, 1, 2),
    (1, 2, 2, 1, 2),
    (2, 2, 1, 2, 2),
    (1, 1, 2, 1, 1, 2),
    (1, 2, 1, 2, 1, 2),
]

GEN_RETRY_LIMIT = 400


def _depth(size: int, r: int, c: int) -> int:
    return min(r, c, size - 1 - r, size - 1 - c)


@lru_cache(maxsize=None)
def _depth_to_layer(widths: tuple[int, ...]) -> tuple[int, ...]:
    mapping: list[int] = []
    for layer, width in enumerate(widths):
        mapping.extend([layer] * int(width))
    return tuple(mapping)


def _layer_for_depth(widths: tuple[int, ...], depth: int) -> int:
    return int(_depth_to_layer(widths)[depth])


def _minimal_period(widths: tuple[int, ...]) -> int:
    n = len(widths)
    for period in range(1, n + 1):
        if all(widths[i] == widths[i % period] for i in range(n)):
            return period
    return n


@lru_cache(maxsize=None)
def _candidate_widths(depth_count: int, periodic_only: bool) -> tuple[tuple[int, ...], ...]:
    out: list[tuple[int, ...]] = []
    for layer_count in range(5, 8):
        for widths in product((1, 2), repeat=layer_count):
            if sum(widths) != depth_count:
                continue
            if periodic_only and _minimal_period(widths) not in (2, 3):
                continue
            out.append(tuple(int(v) for v in widths))
    return tuple(out)


def _render_rings(size: int, widths: tuple[int, ...], colors: tuple[int, ...]) -> np.ndarray:
    grid = np.zeros((size, size), dtype=int)
    layer_of_depth = _depth_to_layer(widths)
    for r in range(size):
        for c in range(size):
            layer = layer_of_depth[_depth(size, r, c)]
            grid[r, c] = int(colors[layer])
    return grid


def _cross_cells(size: int, center: tuple[int, int]) -> list[tuple[int, int]]:
    r, c = center
    cells = [(r, c)]
    for rr, cc in neighbors4(r, c):
        if not in_bounds((size, size), rr, cc):
            raise ValueError("cross must stay inside the grid")
        cells.append((rr, cc))
    return cells


def _mask_with_crosses(clean: np.ndarray, centers: list[tuple[int, int]]) -> np.ndarray:
    out = clean.copy()
    for center in centers:
        for r, c in _cross_cells(clean.shape[0], center):
            out[r, c] = MASK_COLOR
    return out


def _infer_colors_for_widths(
    grid: np.ndarray,
    widths: tuple[int, ...],
    *,
    require_all_observed: bool,
) -> tuple[int | None, ...] | None:
    size = int(grid.shape[0])
    layer_of_depth = _depth_to_layer(widths)
    observed: list[set[int]] = [set() for _ in widths]

    for r in range(size):
        for c in range(size):
            color = int(grid[r, c])
            if color == MASK_COLOR:
                continue
            layer = layer_of_depth[_depth(size, r, c)]
            observed[layer].add(color)
            if len(observed[layer]) > 1:
                return None

    used: set[int] = set()
    colors: list[int | None] = []
    for seen in observed:
        if require_all_observed and not seen:
            return None
        if len(seen) == 1:
            color = next(iter(seen))
            if color == MASK_COLOR or color in used:
                return None
            used.add(color)
            colors.append(int(color))
        else:
            colors.append(None)
    return tuple(colors)


def _unique_periodic_solution(grid: np.ndarray) -> tuple[tuple[int, ...], tuple[int, ...]]:
    g = np.asarray(grid, dtype=int)
    assert g.ndim == 2 and g.shape[0] == g.shape[1], "expected a square grid"
    size = int(g.shape[0])
    assert size % 2 == 1, "expected an odd-sized grid"
    depth_count = size // 2 + 1

    candidates: list[tuple[tuple[int, ...], tuple[int, ...]]] = []
    for widths in _candidate_widths(depth_count, True):
        colors = _infer_colors_for_widths(g, widths, require_all_observed=True)
        if colors is None or any(color is None for color in colors):
            continue
        candidates.append((widths, tuple(int(color) for color in colors)))

    assert len(candidates) == 1, f"expected one periodic ring explanation, got {len(candidates)}"
    return candidates[0]


def main(grid: np.ndarray) -> np.ndarray:
    """
    Repair magenta cross damage in concentric rectangular rings by inferring the repeating ring-width cycle.

    Input structure:
    - The input is an odd square grid filled completely by nested concentric rectangular rings centered on the grid.
    - Each ring layer has one uniform color, while 2 to 3 magenta 5-cell crosses overwrite some original ring pixels.
    - Ring widths are 1 or 2 cells and the width sequence has 5 to 7 layers with a minimal repeat period of 2 or 3.

    Rule:
    - Treat magenta cells as unknown damage.
    - Enumerate the possible periodic ring-width sequences that fit the grid depth, and keep the unique one whose layer
      assignment is consistent with every visible non-magenta pixel.
    - Use the recovered layer colors together with the inferred widths to reconstruct the full concentric-ring pattern.

    Key insight:
    - The hidden information is geometric, not chromatic: a masked center can still be ambiguous after exact-depth color
      lookup, but the repeating width cycle fixes which depths belong to the same ring layer.
    """
    widths, colors = _unique_periodic_solution(np.asarray(grid, dtype=int))
    return _render_rings(int(grid.shape[0]), widths, colors)


def _visible_count_per_layer(clean: np.ndarray, masked: np.ndarray, widths: tuple[int, ...]) -> list[int]:
    size = int(clean.shape[0])
    counts = [0 for _ in widths]
    for r in range(size):
        for c in range(size):
            if int(masked[r, c]) == MASK_COLOR:
                continue
            layer = _layer_for_depth(widths, _depth(size, r, c))
            counts[layer] += 1
    return counts


def _boundary_cross_candidates(clean: np.ndarray, widths: tuple[int, ...]) -> list[tuple[int, int]]:
    size = int(clean.shape[0])
    max_depth = size // 2
    candidates: list[tuple[int, int]] = []
    for r in range(1, size - 1):
        for c in range(1, size - 1):
            if _depth(size, r, c) >= max_depth - 1:
                continue
            cells = _cross_cells(size, (r, c))
            layers = {_layer_for_depth(widths, _depth(size, rr, cc)) for rr, cc in cells}
            if len(layers) >= 2:
                candidates.append((r, c))
    random.shuffle(candidates)
    return candidates


def _crosses_overlap(size: int, a: tuple[int, int], b: tuple[int, int]) -> bool:
    return bool(set(_cross_cells(size, a)) & set(_cross_cells(size, b)))


def _choose_cross_centers(clean: np.ndarray, widths: tuple[int, ...]) -> list[tuple[int, int]] | None:
    size = int(clean.shape[0])
    center = (size // 2, size // 2)
    chosen = [center]
    target_count = random.choice((2, 3))
    for candidate in _boundary_cross_candidates(clean, widths):
        if any(_crosses_overlap(size, candidate, existing) for existing in chosen):
            continue
        chosen.append(candidate)
        if len(chosen) == target_count:
            return chosen
    return None


def _alt_center_split_exists(grid: np.ndarray, widths: tuple[int, ...], colors: tuple[int, ...]) -> bool:
    if widths[-1] != 2 or len(widths) >= 7:
        return False
    alt_widths = tuple(list(widths[:-1]) + [1, 1])
    alt_seen = _infer_colors_for_widths(grid, alt_widths, require_all_observed=False)
    if alt_seen is None:
        return False
    used = {int(color) for color in alt_seen if color is not None}
    unused = [color for color in RING_PALETTE if color not in used]
    if not unused:
        return False
    return int(colors[-1]) in used and alt_seen[-1] is None


def _exact_depth_shortcut_fails(masked: np.ndarray) -> bool:
    size = int(masked.shape[0])
    center_depth = size // 2
    for r in range(size):
        for c in range(size):
            if int(masked[r, c]) != MASK_COLOR:
                continue
            d = _depth(size, r, c)
            visible_same_depth = [
                (rr, cc)
                for rr in range(size)
                for cc in range(size)
                if _depth(size, rr, cc) == d and int(masked[rr, cc]) != MASK_COLOR
            ]
            if not visible_same_depth:
                return True
            if d == center_depth:
                return False
    return False


def _is_boundary_cross(clean: np.ndarray, center: tuple[int, int]) -> bool:
    colors = {int(clean[r, c]) for r, c in _cross_cells(int(clean.shape[0]), center)}
    return len(colors) >= 2


def _build_valid_example() -> np.ndarray | None:
    widths = random.choice(ACTUAL_WIDTH_SEQUENCES)
    size = 2 * sum(widths) - 1
    colors = tuple(random.sample(RING_PALETTE, len(widths)))
    clean = _render_rings(size, widths, colors)

    centers = _choose_cross_centers(clean, widths)
    if centers is None:
        return None

    masked = _mask_with_crosses(clean, centers)
    if not any(_is_boundary_cross(clean, center) for center in centers[1:]):
        return None

    if min(_visible_count_per_layer(clean, masked, widths)) <= 0:
        return None
    if not _exact_depth_shortcut_fails(masked):
        return None
    if not _alt_center_split_exists(masked, widths, colors):
        return None

    periodic_widths, periodic_colors = _unique_periodic_solution(masked)
    if periodic_widths != widths or periodic_colors != colors:
        return None
    if not np.array_equal(main(masked), clean):
        return None

    return masked


def generate_input() -> np.ndarray:
    for _ in range(GEN_RETRY_LIMIT):
        grid = _build_valid_example()
        if grid is not None:
            return grid
    raise RuntimeError("failed to generate a valid periodic-ring repair input")
