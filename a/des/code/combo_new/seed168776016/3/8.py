from __future__ import annotations

import random
from typing import Iterable

import numpy as np


BG = 0
RED = 2
GEN_RETRIES = 400
PALETTE = [1, 3, 4, 5, 6, 7, 8, 9]
DIR_TO_STEP = {
    "up": (-1, 0),
    "down": (1, 0),
    "left": (0, -1),
    "right": (0, 1),
}


def _components(mask: np.ndarray) -> list[np.ndarray]:
    h, w = mask.shape
    seen = np.zeros((h, w), dtype=bool)
    comps: list[np.ndarray] = []
    for r in range(h):
        for c in range(w):
            if not mask[r, c] or seen[r, c]:
                continue
            stack = [(r, c)]
            seen[r, c] = True
            pts: list[tuple[int, int]] = []
            while stack:
                cr, cc = stack.pop()
                pts.append((cr, cc))
                for dr, dc in ((1, 0), (-1, 0), (0, 1), (0, -1)):
                    nr = cr + dr
                    nc = cc + dc
                    if 0 <= nr < h and 0 <= nc < w and mask[nr, nc] and not seen[nr, nc]:
                        seen[nr, nc] = True
                        stack.append((nr, nc))
            comps.append(np.array(pts, dtype=int))
    return comps


def _bbox(coords: np.ndarray) -> tuple[int, int, int, int]:
    r0, c0 = coords.min(axis=0)
    r1, c1 = coords.max(axis=0)
    return int(r0), int(c0), int(r1), int(c1)


def _full_border_cells(r0: int, c0: int, r1: int, c1: int) -> set[tuple[int, int]]:
    cells = set()
    for c in range(c0, c1 + 1):
        cells.add((r0, c))
        cells.add((r1, c))
    for r in range(r0, r1 + 1):
        cells.add((r, c0))
        cells.add((r, c1))
    return cells


def _missing_side_cells(r0: int, c0: int, r1: int, c1: int) -> dict[str, set[tuple[int, int]]]:
    return {
        "up": {(r0, c) for c in range(c0 + 1, c1)},
        "down": {(r1, c) for c in range(c0 + 1, c1)},
        "left": {(r, c0) for r in range(r0 + 1, r1)},
        "right": {(r, c1) for r in range(r0 + 1, r1)},
    }


def _classify_ring(coords: np.ndarray) -> dict[str, int | str] | None:
    r0, c0, r1, c1 = _bbox(coords)
    height = r1 - r0 + 1
    width = c1 - c0 + 1
    if height < 3 or width < 3:
        return None
    pts = {tuple(map(int, rc)) for rc in coords}
    if any(r not in (r0, r1) and c not in (c0, c1) for r, c in pts):
        return None
    border = _full_border_cells(r0, c0, r1, c1)
    for direction, missing in _missing_side_cells(r0, c0, r1, c1).items():
        if pts == border - missing:
            return {
                "r0": r0,
                "c0": c0,
                "r1": r1,
                "c1": c1,
                "area": height * width,
                "direction": direction,
            }
    return None


def _find_marker(g: np.ndarray) -> tuple[int, int]:
    marker_comps = [comp for comp in _components(g == RED) if len(comp) == 1]
    assert len(marker_comps) == 1
    r, c = marker_comps[0][0]
    return int(r), int(c)


def _find_rings(g: np.ndarray) -> list[dict[str, int | str]]:
    rings: list[dict[str, int | str]] = []
    for color in sorted(int(v) for v in np.unique(g) if int(v) != BG):
        for comp in _components(g == color):
            if color == RED and len(comp) == 1:
                continue
            ring = _classify_ring(comp)
            assert ring is not None
            rings.append(ring)
    areas = [int(ring["area"]) for ring in rings]
    assert areas and len(areas) == len(set(areas))
    rings.sort(key=lambda ring: int(ring["area"]))
    return rings


def main(grid: np.ndarray) -> np.ndarray:
    """
    Move a red marker by reading C-ring openings from smallest ring to largest ring.

    Input structure:
    - The background is black.
    - Several non-touching monochrome C-shaped rings appear as one-pixel rectangular borders with exactly one missing side interior.
    - A single red marker pixel sits on an otherwise black cell.

    Rule:
    - Detect every C-shaped ring and compute its bounding-box area.
    - Sort the rings from smallest bounding-box area to largest; each ring's missing side encodes one step direction.
    - Move the red marker one cell per ring in that direction, clamping at the grid boundary when a step would leave the grid.
    - Keep every ring unchanged, remove the original red pixel, and place one red pixel at the final marker location.

    Key insight:
    - Each ring contributes a direction via its opening, while the ring size determines the order in which those directions are executed.
    """
    g = np.asarray(grid, dtype=int)
    h, w = g.shape
    mr, mc = _find_marker(g)
    rings = _find_rings(g)

    out = g.copy()
    out[mr, mc] = BG
    cr, cc = mr, mc
    for ring in rings:
        dr, dc = DIR_TO_STEP[str(ring["direction"])]
        nr = min(max(cr + dr, 0), h - 1)
        nc = min(max(cc + dc, 0), w - 1)
        cr, cc = nr, nc
    out[cr, cc] = RED
    return out


def _ring_cells(r0: int, c0: int, height: int, width: int, direction: str) -> set[tuple[int, int]]:
    r1 = r0 + height - 1
    c1 = c0 + width - 1
    border = _full_border_cells(r0, c0, r1, c1)
    return border - _missing_side_cells(r0, c0, r1, c1)[direction]


def _simulate_path(
    start: tuple[int, int],
    moves: Iterable[tuple[int, int]],
    h: int,
    w: int,
) -> tuple[list[tuple[int, int]], int, int]:
    positions = [start]
    moved_steps = 0
    clamped_steps = 0
    cr, cc = start
    for dr, dc in moves:
        nr = min(max(cr + dr, 0), h - 1)
        nc = min(max(cc + dc, 0), w - 1)
        if (nr, nc) == (cr, cc):
            clamped_steps += 1
        else:
            moved_steps += 1
        cr, cc = nr, nc
        positions.append((cr, cc))
    return positions, moved_steps, clamped_steps


def _place_rings(
    rng: random.Random,
    h: int,
    w: int,
    ring_specs: list[tuple[int, int, str, int]],
) -> tuple[np.ndarray, np.ndarray] | None:
    occupied = np.zeros((h, w), dtype=bool)
    bbox_blocked = np.zeros((h, w), dtype=bool)
    grid = np.full((h, w), BG, dtype=int)

    # Larger boxes first makes random packing more reliable.
    for height, width, direction, color in sorted(ring_specs, key=lambda item: item[0] * item[1], reverse=True):
        placed = False
        positions = [
            (r0, c0)
            for r0 in range(1, h - height)
            for c0 in range(1, w - width)
        ]
        rng.shuffle(positions)
        for r0, c0 in positions:
            r1 = r0 + height - 1
            c1 = c0 + width - 1
            if occupied[r0 - 1 : r1 + 2, c0 - 1 : c1 + 2].any():
                continue
            cells = _ring_cells(r0, c0, height, width, direction)
            for rr, cc in cells:
                grid[rr, cc] = color
                occupied[rr, cc] = True
            bbox_blocked[r0 : r1 + 1, c0 : c1 + 1] = True
            placed = True
            break
        if not placed:
            return None
    return grid, bbox_blocked


def _choose_marker_start(
    rng: random.Random,
    grid: np.ndarray,
    bbox_blocked: np.ndarray,
    moves: list[tuple[int, int]],
    force_clamp: bool,
) -> tuple[int, int] | None:
    h, w = grid.shape
    candidates = [
        (r, c)
        for r in range(h)
        for c in range(w)
        if grid[r, c] == BG and not bbox_blocked[r, c]
    ]
    rng.shuffle(candidates)
    for start in candidates:
        positions, moved_steps, clamped_steps = _simulate_path(start, moves, h, w)
        if force_clamp:
            if clamped_steps == 0 or moved_steps == 0:
                continue
        else:
            if clamped_steps != 0:
                continue
        if any(bbox_blocked[r, c] for r, c in positions):
            continue
        return start
    return None


def _gen_attempt(rng: random.Random) -> np.ndarray | None:
    h = rng.randint(12, 30)
    w = rng.randint(12, 30)
    max_rings = 5 if h * w >= 280 else 4
    ring_count = rng.randint(2, max_rings)

    dims = [(hh, ww) for hh in range(3, 9) for ww in range(3, 9)]
    rng.shuffle(dims)
    used_areas: set[int] = set()
    ring_specs: list[tuple[int, int, str, int]] = []
    colors = rng.sample(PALETTE, ring_count)
    directions = ["up", "down", "left", "right"]
    for color in colors:
        picked = None
        for height, width in dims:
            area = height * width
            if area in used_areas:
                continue
            picked = (height, width)
            used_areas.add(area)
            break
        if picked is None:
            return None
        direction = directions[rng.randrange(4)]
        ring_specs.append((picked[0], picked[1], direction, color))

    placed = _place_rings(rng, h, w, ring_specs)
    if placed is None:
        return None
    grid, bbox_blocked = placed

    ordered = sorted(ring_specs, key=lambda item: item[0] * item[1])
    moves = [DIR_TO_STEP[direction] for _height, _width, direction, _color in ordered]
    force_clamp = bool(rng.randrange(2))
    start = _choose_marker_start(rng, grid, bbox_blocked, moves, force_clamp)
    if start is None:
        return None

    positions, _moved_steps, _clamped_steps = _simulate_path(start, moves, h, w)
    if len(positions) < 2:
        return None
    sr, sc = start
    fr, fc = positions[-1]
    if bbox_blocked[fr, fc]:
        return None
    grid[sr, sc] = RED

    expected = grid.copy()
    expected[sr, sc] = BG
    expected[fr, fc] = RED
    if not np.array_equal(main(grid), expected):
        return None
    return grid


def generate_input() -> np.ndarray:
    rng = random.Random()
    for _ in range(GEN_RETRIES):
        grid = _gen_attempt(rng)
        if grid is not None:
            return grid
    raise RuntimeError(f"generate_input: failed after {GEN_RETRIES} attempts")
