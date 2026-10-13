from __future__ import annotations

import random

import numpy as np


BACKGROUND = 0
ANCHOR_COLOR = 5
MARKER_COLORS = (1, 2, 3, 4, 6, 7, 8, 9)

GEN_RETRIES = 400
GRID_SIZE_RANGE = (14, 26)
ANCHOR_DISTANCE_MIN = 6
RING_COUNT_OPTIONS = ((1, 2), (2, 1), (2, 2), (1, 3), (3, 1), (2, 3), (3, 2))
MIN_VISIBLE_RING_CELLS = 5
MAX_VISIBLE_RING_CELLS = 36
TARGET_COVERAGE_RANGE = (0.10, 0.42)


def _chebyshev(a: tuple[int, int], b: tuple[int, int]) -> int:
    return max(abs(int(a[0]) - int(b[0])), abs(int(a[1]) - int(b[1])))


def _background_color(grid: np.ndarray) -> int:
    values, counts = np.unique(grid, return_counts=True)
    return int(values[np.argmax(counts)])


def _ring_cells(shape: tuple[int, int], center: tuple[int, int], radius: int) -> list[tuple[int, int]]:
    h, w = map(int, shape)
    cr, cc = map(int, center)
    cells = []
    r0 = max(0, cr - radius)
    r1 = min(h - 1, cr + radius)
    c0 = max(0, cc - radius)
    c1 = min(w - 1, cc + radius)
    for r in range(r0, r1 + 1):
        for c in range(c0, c1 + 1):
            if max(abs(r - cr), abs(c - cc)) == radius:
                cells.append((r, c))
    return cells


def _is_clipped(shape: tuple[int, int], center: tuple[int, int], radius: int) -> bool:
    h, w = map(int, shape)
    cr, cc = map(int, center)
    return not (cr - radius >= 0 and cr + radius < h and cc - radius >= 0 and cc + radius < w)


def _parse_input(grid: np.ndarray) -> tuple[int, list[tuple[int, int]], list[tuple[tuple[int, int], int]]]:
    g = np.asarray(grid, dtype=int)
    bg = _background_color(g)
    anchors = [tuple(map(int, coord)) for coord in np.argwhere(g == ANCHOR_COLOR)]
    markers = []
    for r, c in np.argwhere((g != bg) & (g != ANCHOR_COLOR)):
        markers.append(((int(r), int(c)), int(g[r, c])))
    return bg, anchors, markers


def main(grid: np.ndarray) -> np.ndarray:
    """
    Expand each colored marker into the full Chebyshev square ring around its nearer grey anchor.

    Input structure:
    - The grid has a black background, exactly two grey anchor pixels, and several non-grey single-pixel colored markers.
    - Each colored marker is strictly nearer to one anchor than the other in Chebyshev distance.
    - Markers assigned to the same anchor sit at distinct Chebyshev radii from that anchor.

    Rule:
    - For each colored marker, choose the nearer grey anchor and compute its Chebyshev distance d to that anchor.
    - Paint every in-bounds cell whose Chebyshev distance to that anchor equals d with the marker's color.
    - Keep both grey anchors unchanged and leave every cell not covered by any ring as background.

    Key insight:
    - A marker supplies both a color and a radius; the only extra decision is which anchor owns it, determined by nearest-anchor assignment.
    """
    g = np.asarray(grid, dtype=int)
    assert g.ndim == 2, "expected a 2D grid"

    bg, anchors, markers = _parse_input(g)
    assert len(anchors) == 2, "expected exactly two grey anchors"
    assert markers, "expected at least one colored marker"

    out = np.full(g.shape, bg, dtype=int)
    for anchor in anchors:
        out[anchor] = ANCHOR_COLOR

    for coord, color in markers:
        d0 = _chebyshev(coord, anchors[0])
        d1 = _chebyshev(coord, anchors[1])
        assert d0 != d1, "marker must have a unique nearer anchor"
        anchor = anchors[0] if d0 < d1 else anchors[1]
        radius = min(d0, d1)
        for rr, cc in _ring_cells(g.shape, anchor, radius):
            out[rr, cc] = color

    for anchor in anchors:
        out[anchor] = ANCHOR_COLOR
    return out


def _sample_anchor_pair(height: int, width: int, prefer_edge: bool) -> tuple[tuple[int, int], tuple[int, int]] | None:
    def interior_coord() -> tuple[int, int]:
        return random.randrange(1, height - 1), random.randrange(1, width - 1)

    def edge_coord() -> tuple[int, int]:
        side = random.choice(("top", "bottom", "left", "right"))
        if side == "top":
            return 0, random.randrange(width)
        if side == "bottom":
            return height - 1, random.randrange(width)
        if side == "left":
            return random.randrange(height), 0
        return random.randrange(height), width - 1

    for _ in range(60):
        mode = random.choice(("horizontal", "vertical", "diagonal", "mixed"))
        if prefer_edge and random.random() < 0.7:
            first = edge_coord()
            second = interior_coord()
        elif mode == "horizontal":
            first = (random.randrange(height), random.randrange(0, max(1, width // 3)))
            second = (random.randrange(height), random.randrange(max(1, 2 * width // 3), width))
        elif mode == "vertical":
            first = (random.randrange(0, max(1, height // 3)), random.randrange(width))
            second = (random.randrange(max(1, 2 * height // 3), height), random.randrange(width))
        elif mode == "diagonal":
            if random.random() < 0.5:
                first = (random.randrange(0, max(1, height // 3)), random.randrange(0, max(1, width // 3)))
                second = (
                    random.randrange(max(1, 2 * height // 3), height),
                    random.randrange(max(1, 2 * width // 3), width),
                )
            else:
                first = (random.randrange(0, max(1, height // 3)), random.randrange(max(1, 2 * width // 3), width))
                second = (random.randrange(max(1, 2 * height // 3), height), random.randrange(0, max(1, width // 3)))
        else:
            first = interior_coord()
            second = interior_coord()

        if first != second and _chebyshev(first, second) >= ANCHOR_DISTANCE_MIN:
            return first, second
    return None


def _ring_candidate(
    shape: tuple[int, int],
    anchor: tuple[int, int],
    other_anchor: tuple[int, int],
    anchor_index: int,
    radius: int,
) -> dict | None:
    ring = _ring_cells(shape, anchor, radius)
    if len(ring) < MIN_VISIBLE_RING_CELLS or len(ring) > MAX_VISIBLE_RING_CELLS:
        return None
    if other_anchor in ring:
        return None

    valid_markers = [cell for cell in ring if _chebyshev(cell, other_anchor) > radius]
    if len(valid_markers) < 2:
        return None

    return {
        "anchor_index": anchor_index,
        "radius": radius,
        "ring": set(ring),
        "valid_markers": valid_markers,
        "clipped": _is_clipped(shape, anchor, radius),
    }


def _build_candidates(
    shape: tuple[int, int], anchors: list[tuple[int, int]]
) -> tuple[list[dict], list[dict]]:
    candidates = []
    for anchor_index, anchor in enumerate(anchors):
        other_anchor = anchors[1 - anchor_index]
        max_radius = max(
            anchor[0],
            shape[0] - 1 - anchor[0],
            anchor[1],
            shape[1] - 1 - anchor[1],
        )
        anchor_candidates = []
        for radius in range(1, max_radius + 1):
            candidate = _ring_candidate(shape, anchor, other_anchor, anchor_index, radius)
            if candidate is not None:
                anchor_candidates.append(candidate)
        candidates.append(anchor_candidates)
    return candidates[0], candidates[1]


def _choose_rings(
    candidates_by_anchor: tuple[list[dict], list[dict]],
    counts: tuple[int, int],
    require_clipped: bool,
) -> list[dict] | None:
    need = [0] * counts[0] + [1] * counts[1]
    random.shuffle(need)
    chosen: list[dict] = []
    occupied: set[tuple[int, int]] = set()

    def dfs(idx: int) -> bool:
        if idx == len(need):
            return (not require_clipped) or any(item["clipped"] for item in chosen)

        anchor_index = need[idx]
        candidates = candidates_by_anchor[anchor_index][:]
        random.shuffle(candidates)
        if require_clipped and not any(item["clipped"] for item in chosen):
            candidates.sort(key=lambda item: 0 if item["clipped"] else 1)

        for candidate in candidates:
            if any(
                item["anchor_index"] == anchor_index and item["radius"] == candidate["radius"]
                for item in chosen
            ):
                continue
            if not occupied.isdisjoint(candidate["ring"]):
                continue
            if require_clipped and idx == len(need) - 1 and not candidate["clipped"] and not any(
                item["clipped"] for item in chosen
            ):
                continue

            chosen.append(candidate)
            for cell in candidate["ring"]:
                occupied.add(cell)
            if dfs(idx + 1):
                return True
            for cell in candidate["ring"]:
                occupied.remove(cell)
            chosen.pop()
        return False

    return chosen.copy() if dfs(0) else None


def _validate_generated_input(grid: np.ndarray) -> bool:
    bg, anchors, markers = _parse_input(grid)
    if bg != BACKGROUND or len(anchors) != 2 or not markers:
        return False

    seen_colors = set()
    used_radii = {0: set(), 1: set()}
    occupied_rings: list[set[tuple[int, int]]] = []

    for coord, color in markers:
        if color in seen_colors or color in (BACKGROUND, ANCHOR_COLOR):
            return False
        seen_colors.add(color)

        d0 = _chebyshev(coord, anchors[0])
        d1 = _chebyshev(coord, anchors[1])
        if d0 == d1:
            return False
        owner = 0 if d0 < d1 else 1
        radius = min(d0, d1)
        if radius in used_radii[owner]:
            return False
        used_radii[owner].add(radius)

        ring = set(_ring_cells(grid.shape, anchors[owner], radius))
        if any(ring & other for other in occupied_rings):
            return False
        occupied_rings.append(ring)

    out = main(grid)
    coverage = float(np.count_nonzero(out != BACKGROUND)) / float(out.size)
    if not (TARGET_COVERAGE_RANGE[0] <= coverage <= TARGET_COVERAGE_RANGE[1]):
        return False
    return not np.array_equal(out, grid)


def _gen_attempt() -> np.ndarray | None:
    height = random.randint(*GRID_SIZE_RANGE)
    width = random.randint(*GRID_SIZE_RANGE)
    counts = random.choice(RING_COUNT_OPTIONS)
    require_clipped = random.random() < 0.45

    anchors_pair = _sample_anchor_pair(height, width, prefer_edge=require_clipped)
    if anchors_pair is None:
        return None
    anchors = [anchors_pair[0], anchors_pair[1]]

    candidates = _build_candidates((height, width), anchors)
    if len(candidates[0]) < counts[0] or len(candidates[1]) < counts[1]:
        return None
    if require_clipped and not any(item["clipped"] for group in candidates for item in group):
        return None

    chosen = _choose_rings(candidates, counts, require_clipped=require_clipped)
    if chosen is None:
        return None

    random.shuffle(chosen)
    colors = random.sample(MARKER_COLORS, len(chosen))
    grid = np.full((height, width), BACKGROUND, dtype=int)
    for anchor in anchors:
        grid[anchor] = ANCHOR_COLOR

    for candidate, color in zip(chosen, colors):
        marker = random.choice(candidate["valid_markers"])
        grid[marker] = color

    anchor_counts = [0, 0]
    clipped_count = 0
    for candidate in chosen:
        anchor_counts[candidate["anchor_index"]] += 1
        clipped_count += int(candidate["clipped"])
    if min(anchor_counts) < 1:
        return None
    if require_clipped and clipped_count < 1:
        return None
    if not _validate_generated_input(grid):
        return None
    return grid


def generate_input() -> np.ndarray:
    for _ in range(GEN_RETRIES):
        grid = _gen_attempt()
        if grid is not None:
            return grid
    raise RuntimeError("failed to generate a valid pixel-ring input")
