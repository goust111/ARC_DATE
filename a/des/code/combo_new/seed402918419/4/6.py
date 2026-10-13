from __future__ import annotations

from collections import deque
import random

import numpy as np


BG = 0
TRACK = 5
MARKER = 7
BEAD_COLORS = [1, 2, 3, 4, 6, 8, 9]

RING_COUNT_CHOICES = [1, 2, 3, 4]
RING_COUNT_WEIGHTS = [3, 4, 2, 1]
MIN_GRID_SIDE = {1: 10, 2: 12, 3: 16, 4: 18}
MAX_GRID_SIDE = 30
RING_MAX_SIDE = {1: 12, 2: 10, 3: 8, 4: 7}
PLACEMENT_RETRIES = 200
GEN_RETRIES = 400


def _components4(mask: np.ndarray) -> list[list[tuple[int, int]]]:
    h, w = mask.shape
    seen = np.zeros_like(mask, dtype=bool)
    comps: list[list[tuple[int, int]]] = []
    for r in range(h):
        for c in range(w):
            if not mask[r, c] or seen[r, c]:
                continue
            q = deque([(r, c)])
            seen[r, c] = True
            comp: list[tuple[int, int]] = []
            while q:
                cr, cc = q.popleft()
                comp.append((cr, cc))
                for dr, dc in ((1, 0), (-1, 0), (0, 1), (0, -1)):
                    nr, nc = cr + dr, cc + dc
                    if 0 <= nr < h and 0 <= nc < w and mask[nr, nc] and not seen[nr, nc]:
                        seen[nr, nc] = True
                        q.append((nr, nc))
            comps.append(comp)
    return comps


def _bbox(points: list[tuple[int, int]]) -> tuple[int, int, int, int]:
    rows = [r for r, _ in points]
    cols = [c for _, c in points]
    return min(rows), min(cols), max(rows), max(cols)


def _perimeter_positions(r0: int, c0: int, r1: int, c1: int) -> list[tuple[int, int]]:
    pts = [(r0, c) for c in range(c0, c1 + 1)]
    pts.extend((r, c1) for r in range(r0 + 1, r1 + 1))
    pts.extend((r1, c) for c in range(c1 - 1, c0 - 1, -1))
    pts.extend((r, c0) for r in range(r1 - 1, r0, -1))
    return pts


def _ring_from_component(points: list[tuple[int, int]]) -> dict[str, object] | None:
    r0, c0, r1, c1 = _bbox(points)
    h = r1 - r0 + 1
    w = c1 - c0 + 1
    if h < 3 or w < 3:
        return None
    perimeter = _perimeter_positions(r0, c0, r1, c1)
    if set(points) != set(perimeter):
        return None
    return {"bbox": (r0, c0, r1, c1), "perimeter": perimeter}


def _find_rings(grid: np.ndarray) -> list[dict[str, object]]:
    mask = np.asarray(grid, dtype=int) != BG
    rings: list[dict[str, object]] = []
    for comp in _components4(mask):
        ring = _ring_from_component(comp)
        if ring is not None:
            rings.append(ring)
    rings.sort(key=lambda ring: tuple(ring["bbox"]))  # type: ignore[arg-type]
    return rings


def main(grid: np.ndarray) -> np.ndarray:
    """
    Roll colored beads clockwise around rectangular loop tracks by the count of interior markers.

    Input structure:
    - The background is black.
    - One or more single-pixel-wide rectangular loop tracks appear as hollow rectangles.
    - Track cells are gray except where single-pixel colored beads sit directly on the track.
    - The empty interior of each loop may contain orange single-pixel marker dots.

    Rule:
    - Detect each rectangular loop track from its closed perimeter, treating bead cells as part of the track.
    - Count the orange marker dots strictly inside that loop's hollow interior; call this count N.
    - Move every bead on that loop N steps clockwise along the perimeter order:
      top edge left-to-right, right edge top-to-bottom, bottom edge right-to-left, left edge bottom-to-top.
    - After moving the beads, restore every other track cell to gray and erase all interior orange markers to black.

    Key insight:
    The colored beads are embedded on the loop itself, while the orange dots inside the loop encode a shared clockwise rotation amount for that loop.
    """
    g = np.asarray(grid, dtype=int)
    out = g.copy()
    rings = _find_rings(g)
    assert rings, "expected at least one rectangular loop track"

    for ring in rings:
        r0, c0, r1, c1 = ring["bbox"]  # type: ignore[misc]
        perimeter = ring["perimeter"]  # type: ignore[misc]
        shift = int(np.count_nonzero(g[r0 + 1 : r1, c0 + 1 : c1] == MARKER))
        beads = [(idx, int(g[r, c])) for idx, (r, c) in enumerate(perimeter) if int(g[r, c]) != TRACK]

        interior = g[r0 + 1 : r1, c0 + 1 : c1] == MARKER
        out[r0 + 1 : r1, c0 + 1 : c1][interior] = BG
        for r, c in perimeter:
            out[r, c] = TRACK

        loop_len = len(perimeter)
        for idx, color in beads:
            nr, nc = perimeter[(idx + shift) % loop_len]
            out[nr, nc] = color

    return out


def _sample_ring_dims(rng: random.Random, ring_count: int) -> list[tuple[int, int]]:
    dims: list[tuple[int, int]] = []
    max_side = RING_MAX_SIDE[ring_count]
    for _ in range(ring_count):
        min_side = 3 if rng.random() < 0.2 else 4
        h = rng.randint(min_side, max_side)
        w = rng.randint(min_side, max_side)
        dims.append((h, w))
    dims.sort(key=lambda hw: hw[0] * hw[1], reverse=True)
    return dims


def _place_rings(
    rng: random.Random, height: int, width: int, dims: list[tuple[int, int]]
) -> list[tuple[int, int, int, int]] | None:
    blocked = np.zeros((height, width), dtype=bool)
    placed: list[tuple[int, int, int, int]] = []
    for ring_h, ring_w in dims:
        if height - ring_h - 1 < 1 or width - ring_w - 1 < 1:
            return None
        found = False
        for _ in range(PLACEMENT_RETRIES):
            r0 = rng.randint(1, height - ring_h - 1)
            c0 = rng.randint(1, width - ring_w - 1)
            if blocked[r0 - 1 : r0 + ring_h + 1, c0 - 1 : c0 + ring_w + 1].any():
                continue
            blocked[r0 - 1 : r0 + ring_h + 1, c0 - 1 : c0 + ring_w + 1] = True
            placed.append((r0, c0, ring_h, ring_w))
            found = True
            break
        if not found:
            return None
    return placed


def _interior_positions(r0: int, c0: int, r1: int, c1: int) -> list[tuple[int, int]]:
    return [(r, c) for r in range(r0 + 1, r1) for c in range(c0 + 1, c1)]


def _sample_marker_count(
    rng: random.Random, interior_area: int, loop_len: int, force_zero: bool
) -> int:
    if force_zero:
        return 0
    if rng.random() < 0.25:
        return 0

    upper = min(interior_area, min(loop_len + 2, 10))
    if upper <= 0:
        return 0
    if rng.random() < 0.4:
        return rng.randint(1, min(upper, 3))
    return rng.randint(1, upper)


def _gen_attempt(rng: random.Random) -> np.ndarray | None:
    ring_count = rng.choices(RING_COUNT_CHOICES, weights=RING_COUNT_WEIGHTS, k=1)[0]
    min_side = MIN_GRID_SIDE[ring_count]
    height = rng.randint(min_side, MAX_GRID_SIDE)
    width = rng.randint(min_side, MAX_GRID_SIDE)

    placements = _place_rings(rng, height, width, _sample_ring_dims(rng, ring_count))
    if placements is None:
        return None

    grid = np.full((height, width), BG, dtype=int)
    ring_specs: list[
        tuple[list[tuple[int, int]], list[tuple[int, int]], list[int], list[tuple[int, int]]]
    ] = []

    zero_ring_index = rng.randrange(ring_count) if ring_count > 1 and rng.random() < 0.5 else None
    moved_any = False

    for idx, (r0, c0, ring_h, ring_w) in enumerate(placements):
        r1 = r0 + ring_h - 1
        c1 = c0 + ring_w - 1
        perimeter = _perimeter_positions(r0, c0, r1, c1)
        loop_len = len(perimeter)
        interior_cells = _interior_positions(r0, c0, r1, c1)
        marker_count = _sample_marker_count(rng, len(interior_cells), loop_len, idx == zero_ring_index)
        bead_max = min(7, max(1, loop_len // 4 + 1))
        bead_count = rng.randint(1, bead_max)
        bead_positions = rng.sample(perimeter, bead_count)
        bead_colors = [rng.choice(BEAD_COLORS) for _ in range(bead_count)]
        marker_positions = rng.sample(interior_cells, marker_count) if marker_count else []
        ring_specs.append((perimeter, bead_positions, bead_colors, marker_positions))
        if marker_count % loop_len != 0:
            moved_any = True

    if not moved_any:
        return None

    for perimeter, bead_positions, bead_colors, marker_positions in ring_specs:
        for r, c in perimeter:
            grid[r, c] = TRACK
        for (r, c), color in zip(bead_positions, bead_colors):
            grid[r, c] = color
        for r, c in marker_positions:
                grid[r, c] = MARKER

    return grid


def generate_input() -> np.ndarray:
    rng = random.Random()
    for _ in range(GEN_RETRIES):
        grid = _gen_attempt(rng)
        if grid is None:
            continue
        try:
            out = main(grid)
        except Exception:
            continue
        if not isinstance(out, np.ndarray):
            continue
        if grid.ndim != 2 or out.ndim != 2:
            continue
        if max(grid.shape) > 30 or max(out.shape) > 30:
            continue
        if np.array_equal(grid, out):
            continue
        return grid
    raise RuntimeError("failed to generate a valid loop-rotation input")
