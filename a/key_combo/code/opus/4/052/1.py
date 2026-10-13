from __future__ import annotations

from importlib.machinery import SourceFileLoader
from pathlib import Path
import random

import numpy as np


_ARC_UTILS_PATH = Path("/home/lixinyu/snap/new/solvers/arc_utils.py")
_arc_utils = SourceFileLoader("_arc_utils_052_1", str(_ARC_UTILS_PATH)).load_module()
find_components = _arc_utils.find_components
paste = _arc_utils.paste
random_free_location = _arc_utils.random_free_location


BG = 0
PALETTE = (1, 2, 3, 4, 5, 6, 7)
GRID_SIDES = (11, 13, 15, 17)
K_RANGE_BY_SIDE = {
    11: (3, 4),
    13: (3, 5),
    15: (4, 6),
    17: (4, 7),
}
AREA_POOL = tuple(range(3, 13))
NOISE_COUNT_RANGE = (2, 5)
GEN_ATTEMPTS = 400

_USED_OUTPUT_SPECS: set[tuple[int, tuple[int, ...]]] = set()


def _load_components(grid: np.ndarray) -> tuple[list[dict], list[dict]]:
    comps = find_components(
        np.asarray(grid, dtype=int),
        bg=BG,
        connectivity=4,
        monochromatic=True,
        return_format="dict",
    )
    valid = [comp for comp in comps if int(comp["area"]) > 1]
    noise = [comp for comp in comps if int(comp["area"]) == 1]
    valid.sort(key=lambda comp: (int(comp["area"]), comp["bbox"][0], comp["bbox"][1]))
    noise.sort(key=lambda comp: (comp["bbox"][0], comp["bbox"][1], int(comp["color"])))
    return valid, noise


def _ring_output(side: int, colors: list[int]) -> np.ndarray:
    out = np.zeros((side, side), dtype=int)
    center = side // 2
    rr, cc = np.indices((side, side))
    dist = np.maximum(np.abs(rr - center), np.abs(cc - center))
    for d, color in enumerate(colors, start=1):
        out[dist == d] = int(color)
    return out


def _component_centroid_distance(comp: dict, center: int) -> float:
    coords = np.asarray(comp["coords"], dtype=int)
    centroid = coords.mean(axis=0)
    return float(max(abs(centroid[0] - center), abs(centroid[1] - center)))


def _all_component_output(side: int, valid: list[dict], noise: list[dict]) -> np.ndarray:
    comps = noise + valid
    comps.sort(key=lambda comp: (int(comp["area"]), comp["bbox"][0], comp["bbox"][1]))
    colors = [int(comp["color"]) for comp in comps]
    return _ring_output(side, colors)


def _is_irregular(coords: np.ndarray) -> bool:
    rs = coords[:, 0]
    cs = coords[:, 1]
    h = int(rs.max() - rs.min() + 1)
    w = int(cs.max() - cs.min() + 1)
    area = int(coords.shape[0])
    return h > 1 and w > 1 and area < h * w


def _make_irregular_patch(area: int, color: int) -> np.ndarray:
    for _ in range(300):
        cells = {(0, 0), (0, 1), (1, 0)}
        while len(cells) < area:
            frontier = []
            for r, c in cells:
                for dr, dc in ((1, 0), (-1, 0), (0, 1), (0, -1)):
                    nr, nc = r + dr, c + dc
                    if (nr, nc) in cells or abs(nr) > 4 or abs(nc) > 4:
                        continue
                    touch = sum((nr + rr, nc + cc) in cells for rr, cc in ((1, 0), (-1, 0), (0, 1), (0, -1)))
                    frontier.extend([(nr, nc)] * max(1, touch))
            if not frontier:
                break
            cells.add(random.choice(frontier))

        coords = np.array(sorted(cells), dtype=int)
        coords -= coords.min(axis=0)
        if not _is_irregular(coords):
            continue
        h = int(coords[:, 0].max() + 1)
        w = int(coords[:, 1].max() + 1)
        if h > 6 or w > 6:
            continue
        patch = np.zeros((h, w), dtype=int)
        for r, c in coords:
            patch[int(r), int(c)] = int(color)
        k = random.randrange(4)
        patch = np.rot90(patch, k)
        if random.random() < 0.5:
            patch = np.fliplr(patch)
        if random.random() < 0.5:
            patch = np.flipud(patch)
        fg = np.argwhere(patch == int(color))
        if _is_irregular(fg):
            return patch
    raise RuntimeError(f"failed to build irregular patch for area={area}")


def _pick_output_spec() -> tuple[int, list[int]]:
    side_choices = list(GRID_SIDES)
    random.shuffle(side_choices)
    for side in side_choices:
        k_min, k_max = K_RANGE_BY_SIDE[side]
        ks = list(range(k_min, k_max + 1))
        random.shuffle(ks)
        for k in ks:
            colors = random.sample(PALETTE, k)
            if colors == sorted(colors) or colors == sorted(colors, reverse=True):
                continue
            spec = (side, tuple(colors))
            if spec not in _USED_OUTPUT_SPECS:
                return side, colors
    raise RuntimeError("exhausted output specs")


def _place_patch(grid: np.ndarray, patch: np.ndarray) -> None:
    r, c = random_free_location(
        grid,
        patch,
        transparent=BG,
        border=1,
        padding=1,
        padding_connectivity=8,
    )
    paste(grid, patch, r, c, transparent=BG)


def _make_noise_colors(valid_colors: list[int]) -> list[int]:
    noise_count = random.randint(*NOISE_COUNT_RANGE)
    colors = [random.choice(valid_colors)]
    while len(colors) < noise_count:
        if random.random() < 0.6:
            colors.append(random.choice(valid_colors))
        else:
            colors.append(random.choice(PALETTE))
    random.shuffle(colors)
    return colors


def _semantic_checks(grid: np.ndarray) -> None:
    valid, noise = _load_components(grid)
    side = int(grid.shape[0])
    assert grid.shape == (side, side)
    assert side % 2 == 1
    assert noise, "generator must include singleton noise"
    assert 3 <= len(valid) <= min((side - 1) // 2, len(PALETTE))

    valid_colors = [int(comp["color"]) for comp in valid]
    areas = [int(comp["area"]) for comp in valid]
    assert len(set(valid_colors)) == len(valid)
    assert len(set(areas)) == len(valid)
    assert all(_is_irregular(np.asarray(comp["coords"], dtype=int)) for comp in valid)

    bbox_order = [
        int(comp["color"])
        for comp in sorted(valid, key=lambda comp: (comp["bbox"][0], comp["bbox"][1]))
    ]
    center = side // 2
    radial_order = [
        int(comp["color"])
        for comp in sorted(valid, key=lambda comp: (_component_centroid_distance(comp, center), comp["bbox"][0], comp["bbox"][1]))
    ]
    descending = [int(comp["color"]) for comp in sorted(valid, key=lambda comp: (-int(comp["area"]), comp["bbox"][0], comp["bbox"][1]))]

    assert valid_colors != bbox_order, "area order must differ from row-major placement order"
    assert valid_colors != radial_order, "area order must differ from centroid distance order"
    assert valid_colors != descending, "ascending and descending area orders must differ"

    expected = _ring_output(side, valid_colors)
    assert np.array_equal(main(grid), expected)
    assert not np.array_equal(expected, _all_component_output(side, valid, noise))
    assert not np.array_equal(expected, _ring_output(side, bbox_order))
    assert not np.array_equal(expected, _ring_output(side, descending))


def main(grid: np.ndarray) -> np.ndarray:
    """Map area-sorted non-noise components to center-aligned concentric square rings.

    Input structure:
    - The input is an odd-sized square grid with black background.
    - Several scattered 4-connected monochrome foreground components have distinct colors and distinct areas greater than 1.
    - Additional isolated single foreground pixels may appear as noise, possibly reusing component colors.

    Rule:
    - Find all monochrome 4-connected foreground components and discard the ones with area 1.
    - Sort the remaining components by area from small to large, breaking ties by bbox top-left row then column.
    - Return an all-black grid of the same size, and for ring distance d = 1..K fill every cell at Chebyshev distance d from the grid center with the color of the d-th sorted component.

    Key insight:
    - The scattered objects are only a hidden ordering device: after filtering singleton noise, component area becomes the index that determines each concentric square ring color.
    """
    g = np.asarray(grid, dtype=int)
    assert g.ndim == 2 and g.shape[0] == g.shape[1] and g.shape[0] % 2 == 1
    valid, _noise = _load_components(g)
    colors = [int(comp["color"]) for comp in valid]
    return _ring_output(int(g.shape[0]), colors)


def _attempt_generate() -> np.ndarray | None:
    side, ring_colors = _pick_output_spec()
    k = len(ring_colors)
    areas = sorted(random.sample(AREA_POOL, k))
    grid = np.zeros((side, side), dtype=int)

    for color, area in zip(ring_colors, areas):
        try:
            patch = _make_irregular_patch(area, color)
            _place_patch(grid, patch)
        except (RuntimeError, ValueError):
            return None

    for color in _make_noise_colors(ring_colors):
        patch = np.array([[int(color)]], dtype=int)
        try:
            _place_patch(grid, patch)
        except ValueError:
            return None

    try:
        _semantic_checks(grid)
    except AssertionError:
        return None

    _USED_OUTPUT_SPECS.add((side, tuple(ring_colors)))
    return grid


def generate_input() -> np.ndarray:
    for _ in range(GEN_ATTEMPTS):
        grid = _attempt_generate()
        if grid is not None:
            return grid
    raise RuntimeError("failed to generate a valid input")
