from __future__ import annotations

from collections import deque

import numpy as np


BACKGROUND = 0
BLUE = 1
RED = 2
GREEN = 3

GEN_RETRIES = 800
PLACEMENT_RETRIES = 300
GRID_SIDE_RANGE = (15, 23)  # upper bound is exclusive


def _bbox_from_coords(coords: np.ndarray) -> tuple[int, int, int, int]:
    r0, c0 = coords.min(axis=0)
    r1, c1 = coords.max(axis=0)
    return int(r0), int(c0), int(r1), int(c1)


def _find_components(mask: np.ndarray) -> list[np.ndarray]:
    h, w = mask.shape
    seen = np.zeros_like(mask, dtype=bool)
    comps: list[np.ndarray] = []

    for r in range(h):
        for c in range(w):
            if not mask[r, c] or seen[r, c]:
                continue
            q = deque([(r, c)])
            seen[r, c] = True
            cells: list[tuple[int, int]] = []
            while q:
                rr, cc = q.popleft()
                cells.append((rr, cc))
                for dr in (-1, 0, 1):
                    for dc in (-1, 0, 1):
                        if dr == 0 and dc == 0:
                            continue
                        nr = rr + dr
                        nc = cc + dc
                        if 0 <= nr < h and 0 <= nc < w and mask[nr, nc] and not seen[nr, nc]:
                            seen[nr, nc] = True
                            q.append((nr, nc))
            comps.append(np.array(cells, dtype=int))

    comps.sort(key=lambda coords: _bbox_from_coords(coords))
    return comps


def _rect_cells(top: int, left: int, size: int, shape: tuple[int, int]) -> set[tuple[int, int]]:
    h, w = shape
    cells: set[tuple[int, int]] = set()
    for r in range(top, top + size):
        for c in range(left, left + size):
            if 0 <= r < h and 0 <= c < w:
                cells.add((r, c))
    return cells


def _halo_cells(top: int, left: int, size: int, shape: tuple[int, int]) -> set[tuple[int, int]]:
    h, w = shape
    halo: set[tuple[int, int]] = set()
    for r in range(top - 1, top + size + 1):
        for c in range(left - 1, left + size + 1):
            if not (0 <= r < h and 0 <= c < w):
                continue
            if top <= r < top + size and left <= c < left + size:
                continue
            halo.add((r, c))
    return halo


def _block_anchors(r0: int, c0: int, size: int, orientation: str) -> list[tuple[int, int]]:
    if orientation == "nwse":
        return [(r0 - size, c0 + 2 * size), (r0 + 2 * size, c0 - size)]
    if orientation == "nesw":
        return [(r0 - size, c0 - size), (r0 + 2 * size, c0 + 2 * size)]
    raise ValueError(f"unknown orientation: {orientation}")


def _object_details(
    r0: int, c0: int, size: int, orientation: str, shape: tuple[int, int]
) -> dict[str, object]:
    green_cells = set()
    green_cells |= _rect_cells(r0, c0, size, shape)
    green_cells |= _rect_cells(r0 + size, c0 + size, size, shape)
    if orientation == "nesw":
        green_cells = set()
        green_cells |= _rect_cells(r0, c0 + size, size, shape)
        green_cells |= _rect_cells(r0 + size, c0, size, shape)

    anchors = _block_anchors(r0, c0, size, orientation)
    block_cells: set[tuple[int, int]] = set()
    halo_cells: set[tuple[int, int]] = set()
    reserve_cells = set(green_cells)
    visible_targets: list[int] = []

    for top, left in anchors:
        block = _rect_cells(top, left, size, shape)
        halo = _halo_cells(top, left, size, shape)
        block_cells |= block
        halo_cells |= halo
        reserve_cells |= block
        reserve_cells |= halo
        visible_targets.append(len(block))

    color = BLUE if orientation == "nwse" else RED
    bbox = (r0, c0, r0 + 2 * size - 1, c0 + 2 * size - 1)
    return {
        "bbox": bbox,
        "size": size,
        "orientation": orientation,
        "green_cells": green_cells,
        "block_cells": block_cells,
        "halo_cells": halo_cells,
        "reserve_cells": reserve_cells,
        "color": color,
        "visible_targets": visible_targets,
    }


def _parse_object(grid: np.ndarray, coords: np.ndarray) -> dict[str, object]:
    r0, c0, r1, c1 = _bbox_from_coords(coords)
    side_r = r1 - r0 + 1
    side_c = c1 - c0 + 1
    assert side_r == side_c and side_r % 2 == 0, "each object must have an even square bbox"
    size = side_r // 2

    sub = grid[r0 : r1 + 1, c0 : c1 + 1]
    occupied: set[tuple[int, int]] = set()
    for br in range(2):
        for bc in range(2):
            block = sub[br * size : (br + 1) * size, bc * size : (bc + 1) * size]
            if np.all(block == GREEN):
                occupied.add((br, bc))
            else:
                assert np.all(block == BACKGROUND), "sub-blocks must be fully green or fully black"

    if occupied == {(0, 0), (1, 1)}:
        orientation = "nwse"
    else:
        assert occupied == {(0, 1), (1, 0)}, "expected one of the two diagonal fill patterns"
        orientation = "nesw"

    return _object_details(r0, c0, size, orientation, grid.shape)


def _bbox_clearance(a: tuple[int, int, int, int], b: tuple[int, int, int, int]) -> int:
    ar0, ac0, ar1, ac1 = a
    br0, bc0, br1, bc1 = b
    row_gap = max(0, br0 - ar1 - 1, ar0 - br1 - 1)
    col_gap = max(0, bc0 - ac1 - 1, ac0 - bc1 - 1)
    return max(row_gap, col_gap)


def _expected_overlay(grid: np.ndarray) -> tuple[list[dict[str, object]], dict[tuple[int, int], int]]:
    comps = _find_components(grid == GREEN)
    assert 3 <= len(comps) <= 4, "expected 3-4 green objects"

    parsed = [_parse_object(grid, comp) for comp in comps]
    orientations = [obj["orientation"] for obj in parsed]
    assert "nwse" in orientations and "nesw" in orientations, "both diagonal directions must be present"

    for i, left in enumerate(parsed):
        for right in parsed[i + 1 :]:
            assert _bbox_clearance(left["bbox"], right["bbox"]) >= 3, "object bboxes are too close"

    expected: dict[tuple[int, int], int] = {}
    for obj in parsed:
        color = int(obj["color"])
        for cell in obj["block_cells"]:
            assert grid[cell] == BACKGROUND, "generator must keep complement blocks empty in the input"
            prev = expected.get(cell)
            assert prev is None or prev == color
            expected[cell] = color
        for cell in obj["halo_cells"]:
            if grid[cell] != BACKGROUND:
                continue
            prev = expected.get(cell)
            assert prev is None or prev == color
            expected[cell] = color

    return parsed, expected


def main(grid: np.ndarray) -> np.ndarray:
    """
    Place colored complement blocks outside each diagonal motif and expand them with a 1-pixel halo.

    Input structure:
    - The grid is black background with 3-4 green motifs.
    - Each motif is one 8-connected diagonal pair of equal solid square blocks inside an even square bbox.
    - Its 2x2 sub-block layout is either northwest+southeast filled or northeast+southwest filled.

    Rule:
    - Preserve every original green pixel.
    - For a northwest+southeast motif, place equal-size blue blocks at the exterior northeast and southwest diagonal corners.
    - For a northeast+southwest motif, place equal-size red blocks at the exterior northwest and southeast diagonal corners.
    - Around each new block, fill the Chebyshev-distance-1 halo in the same color, but only on black background cells; clip at grid edges.

    Key insight:
    - The motif's diagonal fill direction simultaneously determines both where the complement blocks go and which color the entire added region uses.
    """
    g = np.asarray(grid, dtype=int)
    assert g.ndim == 2
    assert set(np.unique(g)).issubset({BACKGROUND, GREEN}), "input must contain only black and green"

    out = g.copy()
    objects, _ = _expected_overlay(g)

    block_jobs: list[tuple[set[tuple[int, int]], int]] = []
    halo_jobs: list[tuple[set[tuple[int, int]], int]] = []
    for obj in objects:
        block_jobs.append((obj["block_cells"], int(obj["color"])))
        halo_jobs.append((obj["halo_cells"], int(obj["color"])))

    for cells, color in block_jobs:
        for r, c in cells:
            if out[r, c] == BACKGROUND:
                out[r, c] = color

    for cells, color in halo_jobs:
        for r, c in cells:
            if out[r, c] == BACKGROUND:
                out[r, c] = color

    return out


def _random_orientation_list(rng: np.random.Generator, count: int) -> list[str]:
    orientations = ["nwse", "nesw"]
    extras = rng.choice(np.array(["nwse", "nesw"]), size=count - 2, replace=True)
    orientations.extend(str(x) for x in extras.tolist())
    rng.shuffle(orientations)
    return orientations


def _random_sizes(rng: np.random.Generator, count: int, side_limit: int) -> list[int]:
    max_big = 0
    if side_limit >= 21 and count == 3:
        max_big = 2
    elif side_limit >= 18:
        max_big = 1
    big_count = int(rng.integers(0, max_big + 1))
    sizes = [3] * big_count + [2] * (count - big_count)
    rng.shuffle(sizes)
    return sizes


def _sample_position(
    rng: np.random.Generator, h: int, w: int, size: int
) -> tuple[int, int]:
    r_positions = list(range(0, h - 2 * size + 1))
    c_positions = list(range(0, w - 2 * size + 1))

    def _biased_pick(positions: list[int], limit: int) -> int:
        if rng.random() < 0.45:
            edge = [p for p in positions if p < size or p > limit - 3 * size]
            if edge:
                return int(rng.choice(np.array(edge)))
        return int(rng.choice(np.array(positions)))

    return _biased_pick(r_positions, h), _biased_pick(c_positions, w)


def _valid_candidate(
    details: dict[str, object],
    objects: list[dict[str, object]],
    occupied_green: set[tuple[int, int]],
    occupied_reserve: set[tuple[int, int]],
) -> bool:
    if any(cell in occupied_green for cell in details["green_cells"]):
        return False
    if any(cell in occupied_reserve for cell in details["reserve_cells"]):
        return False
    if any(v == 0 for v in details["visible_targets"]):
        return False
    if not details["block_cells"]:
        return False
    for other in objects:
        if _bbox_clearance(details["bbox"], other["bbox"]) < 3:
            return False
    return True


def _generate_attempt(rng: np.random.Generator) -> np.ndarray | None:
    h = int(rng.integers(*GRID_SIDE_RANGE))
    w = int(rng.integers(*GRID_SIDE_RANGE))
    grid = np.full((h, w), BACKGROUND, dtype=int)

    count = int(rng.integers(3, 5))
    orientations = _random_orientation_list(rng, count)
    sizes = _random_sizes(rng, count, min(h, w))

    objects: list[dict[str, object]] = []
    occupied_green: set[tuple[int, int]] = set()
    occupied_reserve: set[tuple[int, int]] = set()

    for orientation, size in zip(orientations, sizes):
        placed = False
        for _ in range(PLACEMENT_RETRIES):
            r0, c0 = _sample_position(rng, h, w, size)
            details = _object_details(r0, c0, size, orientation, grid.shape)
            if not _valid_candidate(details, objects, occupied_green, occupied_reserve):
                continue
            for r, c in details["green_cells"]:
                grid[r, c] = GREEN
            objects.append(details)
            occupied_green |= set(details["green_cells"])
            occupied_reserve |= set(details["reserve_cells"])
            placed = True
            break
        if not placed:
            return None

    try:
        _validate_generated(grid)
    except AssertionError:
        return None
    return grid


def _validate_generated(grid: np.ndarray) -> None:
    assert isinstance(grid, np.ndarray)
    assert grid.ndim == 2 and max(grid.shape) <= 30
    assert set(np.unique(grid)).issubset({BACKGROUND, GREEN})

    objects, expected = _expected_overlay(grid)
    out = main(grid)

    assert out.shape == grid.shape
    assert np.array_equal(out[grid == GREEN], grid[grid == GREEN]), "green source pixels must stay unchanged"
    assert BLUE in set(expected.values()) and RED in set(expected.values()), "both color branches must appear"

    actual_added = {(int(r), int(c)): int(out[r, c]) for r, c in np.argwhere(out != grid)}
    assert actual_added == expected, "output additions must exactly match complement blocks plus valid halo cells"

    for obj in objects:
        assert len(obj["block_cells"]) >= 2, "each object must contribute visible complement pixels"


def generate_input() -> np.ndarray:
    rng = np.random.default_rng()
    for _ in range(GEN_RETRIES):
        grid = _generate_attempt(rng)
        if grid is not None:
            return grid
    raise RuntimeError("failed to generate a valid diagonal-complement halo puzzle input")
