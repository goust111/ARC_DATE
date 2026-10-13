from __future__ import annotations

from collections import defaultdict, deque
import random

import numpy as np


BLACK = 0
BLUE = 1
RED = 2
GREEN = 3
YELLOW = 4
GREY = 5
ORANGE = 7
MARKER_COLORS = (BLUE, RED, GREEN)

ORTHO_DIRS = [(-1, 0), (1, 0), (0, -1), (0, 1)]
DIAG_DIRS = [(-1, -1), (-1, 1), (1, -1), (1, 1)]


def _in_bounds(shape: tuple[int, int], r: int, c: int) -> bool:
    return 0 <= r < shape[0] and 0 <= c < shape[1]


def _neighbors4(r: int, c: int):
    for dr, dc in ORTHO_DIRS:
        yield r + dr, c + dc


def _black_components(grid: np.ndarray) -> tuple[list[list[tuple[int, int]]], np.ndarray]:
    h, w = grid.shape
    labels = np.full((h, w), -1, dtype=int)
    components: list[list[tuple[int, int]]] = []
    comp_id = 0

    for r in range(h):
        for c in range(w):
            if grid[r, c] != BLACK or labels[r, c] >= 0:
                continue
            q = deque([(r, c)])
            labels[r, c] = comp_id
            coords: list[tuple[int, int]] = []
            while q:
                rr, cc = q.popleft()
                coords.append((rr, cc))
                for nr, nc in _neighbors4(rr, cc):
                    if _in_bounds(grid.shape, nr, nc) and grid[nr, nc] == BLACK and labels[nr, nc] < 0:
                        labels[nr, nc] = comp_id
                        q.append((nr, nc))
            components.append(coords)
            comp_id += 1
    return components, labels


def _touches_border(coords: list[tuple[int, int]], shape: tuple[int, int]) -> bool:
    h, w = shape
    return any(r == 0 or c == 0 or r == h - 1 or c == w - 1 for r, c in coords)


def _closed_regions(grid: np.ndarray) -> list[dict]:
    components, labels = _black_components(grid)
    regions: list[dict] = []

    for comp_id, coords in enumerate(components):
        if _touches_border(coords, grid.shape):
            continue

        marker_candidates: set[tuple[int, int]] = set()
        for r, c in coords:
            for nr, nc in _neighbors4(r, c):
                if not _in_bounds(grid.shape, nr, nc):
                    continue
                color = int(grid[nr, nc])
                if color in MARKER_COLORS:
                    marker_candidates.add((nr, nc))

        assert len(marker_candidates) == 1, "each enclosed region must contain exactly one marker"
        marker = next(iter(marker_candidates))
        mr, mc = marker
        assert all(
            _in_bounds(grid.shape, mr + dr, mc + dc) and labels[mr + dr, mc + dc] == comp_id
            for dr, dc in ORTHO_DIRS
        ), "marker must be surrounded by its own enclosed black region"

        regions.append(
            {
                "comp_id": comp_id,
                "coords": coords,
                "mask": {(r, c) for r, c in coords},
                "area": len(coords),
                "marker": marker,
                "marker_color": int(grid[marker]),
            }
        )

    return regions


def main(grid: np.ndarray) -> np.ndarray:
    """
    Mark each enclosed-region marker with diagonal or orthogonal neighbors based on the region's black-area parity.

    Input structure:
    - The grid is a black background with several non-overlapping gray hollow rectangles.
    - Each rectangle encloses one 4-connected black interior region that contains exactly one red, blue, or green marker pixel.
    - Markers are separated from the gray border by at least one black cell, and only black regions not touching the grid edge are relevant.

    Rule:
    - Detect every enclosed 4-connected black region and count its black pixels, excluding the marker itself.
    - If a marker's region area is odd, color the marker's in-region diagonal neighbors yellow.
    - If a marker's region area is even, color the marker's in-region orthogonal neighbors orange; keep gray borders, markers, and exterior black unchanged.

    Key insight:
    - Marker color is a distractor; the controlling signal is the enclosed region's global black-area parity, which must be found by region detection before choosing the neighborhood geometry.
    """
    g = np.asarray(grid, dtype=int)
    assert g.ndim == 2, "expected a 2D grid"

    out = g.copy()
    for region in _closed_regions(g):
        marker_r, marker_c = region["marker"]
        odd_area = region["area"] % 2 == 1
        deltas = DIAG_DIRS if odd_area else ORTHO_DIRS
        paint = YELLOW if odd_area else ORANGE
        for dr, dc in deltas:
            nr, nc = marker_r + dr, marker_c + dc
            if (nr, nc) in region["mask"]:
                out[nr, nc] = paint
    return out


GRID_MIN = 15
GRID_MAX = 22
MAX_ATTEMPTS = 400
MIN_FRAME_GAP = 2
EXTRA_REGION_WEIGHTS = (0, 1, 1)
ODD_BLACK_SHAPES = [(3, 4), (4, 4), (3, 6), (4, 5), (5, 6)]
EVEN_BLACK_SHAPES = [(3, 3), (3, 5), (5, 5)]


def _choose_shape(parity: str) -> tuple[int, int]:
    if parity == "odd":
        return random.choice(ODD_BLACK_SHAPES)
    if parity == "even":
        return random.choice(EVEN_BLACK_SHAPES)
    raise ValueError(f"unknown parity {parity}")


def _sample_region_specs() -> list[dict]:
    shared_color = random.choice(MARKER_COLORS)
    specs = [
        {"parity": "odd", "inner_h": 0, "inner_w": 0, "marker_color": shared_color},
        {"parity": "even", "inner_h": 0, "inner_w": 0, "marker_color": shared_color},
    ]

    for _ in range(random.choice(EXTRA_REGION_WEIGHTS)):
        specs.append(
            {
                "parity": random.choice(("odd", "even")),
                "inner_h": 0,
                "inner_w": 0,
                "marker_color": random.choice(MARKER_COLORS),
            }
        )

    for spec in specs:
        spec["inner_h"], spec["inner_w"] = _choose_shape(spec["parity"])
    random.shuffle(specs)
    return specs


def _separated(a: tuple[int, int, int, int], b: tuple[int, int, int, int]) -> bool:
    ar0, ac0, ar1, ac1 = a
    br0, bc0, br1, bc1 = b
    row_gap = max(ar0 - br1, br0 - ar1, 0)
    col_gap = max(ac0 - bc1, bc0 - ac1, 0)
    return max(row_gap, col_gap) >= MIN_FRAME_GAP


def _place_specs(height: int, width: int, specs: list[dict]) -> list[dict] | None:
    placements: list[dict] = []
    order = sorted(specs, key=lambda spec: (spec["inner_h"] + 2) * (spec["inner_w"] + 2), reverse=True)

    for spec in order:
        inner_h = int(spec["inner_h"])
        inner_w = int(spec["inner_w"])
        outer_h = inner_h + 2
        outer_w = inner_w + 2
        valid: list[tuple[int, int]] = []

        for r0 in range(1, height - outer_h):
            for c0 in range(1, width - outer_w):
                bbox = (r0, c0, r0 + outer_h - 1, c0 + outer_w - 1)
                if all(_separated(bbox, placed["bbox"]) for placed in placements):
                    valid.append((r0, c0))

        if not valid:
            return None

        r0, c0 = random.choice(valid)
        inner_r = random.randint(1, inner_h - 2)
        inner_c = random.randint(1, inner_w - 2)
        placements.append(
            {
                **spec,
                "top": r0,
                "left": c0,
                "bbox": (r0, c0, r0 + outer_h - 1, c0 + outer_w - 1),
                "marker": (r0 + 1 + inner_r, c0 + 1 + inner_c),
            }
        )
    return placements


def _draw_input(height: int, width: int, placements: list[dict]) -> np.ndarray:
    grid = np.full((height, width), BLACK, dtype=int)
    for placed in placements:
        r0 = int(placed["top"])
        c0 = int(placed["left"])
        inner_h = int(placed["inner_h"])
        inner_w = int(placed["inner_w"])
        outer_h = inner_h + 2
        outer_w = inner_w + 2

        grid[r0, c0 : c0 + outer_w] = GREY
        grid[r0 + outer_h - 1, c0 : c0 + outer_w] = GREY
        grid[r0 : r0 + outer_h, c0] = GREY
        grid[r0 : r0 + outer_h, c0 + outer_w - 1] = GREY

        mr, mc = placed["marker"]
        grid[mr, mc] = int(placed["marker_color"])
    return grid


def _expected_from_placements(grid: np.ndarray, placements: list[dict]) -> np.ndarray:
    out = grid.copy()
    for placed in placements:
        mr, mc = placed["marker"]
        black_area = int(placed["inner_h"]) * int(placed["inner_w"]) - 1
        odd_area = black_area % 2 == 1
        deltas = DIAG_DIRS if odd_area else ORTHO_DIRS
        color = YELLOW if odd_area else ORANGE
        for dr, dc in deltas:
            out[mr + dr, mc + dc] = color
    return out


def _semantics_hold(grid: np.ndarray, placements: list[dict]) -> bool:
    regions = _closed_regions(grid)
    if len(regions) != len(placements):
        return False

    observed = sorted((region["area"], region["marker_color"]) for region in regions)
    expected = sorted(
        (int(placed["inner_h"]) * int(placed["inner_w"]) - 1, int(placed["marker_color"]))
        for placed in placements
    )
    if observed != expected:
        return False

    parities = {region["area"] % 2 for region in regions}
    if parities != {0, 1}:
        return False

    color_to_parities: dict[int, set[int]] = defaultdict(set)
    for region in regions:
        color_to_parities[int(region["marker_color"])].add(region["area"] % 2)
    if not any(parity_set == {0, 1} for parity_set in color_to_parities.values()):
        return False

    return True


def _gen_attempt() -> np.ndarray | None:
    specs = _sample_region_specs()
    height = random.randint(GRID_MIN, GRID_MAX)
    width = random.randint(GRID_MIN, GRID_MAX)
    placements = _place_specs(height, width, specs)
    if placements is None:
        return None

    grid = _draw_input(height, width, placements)
    if not _semantics_hold(grid, placements):
        return None

    expected = _expected_from_placements(grid, placements)
    try:
        if not np.array_equal(main(grid), expected):
            return None
    except Exception:
        return None
    return grid


def generate_input() -> np.ndarray:
    for _ in range(MAX_ATTEMPTS):
        grid = _gen_attempt()
        if grid is not None:
            return grid
    raise RuntimeError("failed to generate a valid enclosed-parity ARC input")
