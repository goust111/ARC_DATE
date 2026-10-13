import random
from collections import deque
from typing import Dict, List, Optional, Sequence, Set, Tuple

import numpy as np


BG_COLOR = 0
FRAME_COLOR = 5
PALETTE = (1, 2, 3, 4, 6, 7, 8, 9)
SIDES = ("left", "right", "top", "bottom")
SIDE_TO_STEP = {
    "left": (0, 1),
    "right": (0, -1),
    "top": (1, 0),
    "bottom": (-1, 0),
}

# Generator configuration.
MIN_GRID = 16
MAX_GRID = 30
MIN_OUTER = 5
MAX_OUTER = 9
MAX_FRAMES = 4
MAX_OUTSIDE_LEN = 7
GEN_RETRIES = 400
PLACEMENT_TRIES = 140


def _neighbors4(r: int, c: int) -> Sequence[Tuple[int, int]]:
    return ((r - 1, c), (r + 1, c), (r, c - 1), (r, c + 1))


def _components(mask: np.ndarray) -> List[List[Tuple[int, int]]]:
    h, w = mask.shape
    seen = np.zeros((h, w), dtype=bool)
    components: List[List[Tuple[int, int]]] = []
    for r in range(h):
        for c in range(w):
            if not mask[r, c] or seen[r, c]:
                continue
            comp: List[Tuple[int, int]] = []
            q: deque[Tuple[int, int]] = deque([(r, c)])
            seen[r, c] = True
            while q:
                cr, cc = q.popleft()
                comp.append((cr, cc))
                for nr, nc in _neighbors4(cr, cc):
                    if 0 <= nr < h and 0 <= nc < w and mask[nr, nc] and not seen[nr, nc]:
                        seen[nr, nc] = True
                        q.append((nr, nc))
            components.append(comp)
    return components


def _ring_bbox(component: Sequence[Tuple[int, int]]) -> Optional[Tuple[int, int, int, int]]:
    rows = [r for r, _ in component]
    cols = [c for _, c in component]
    r0, r1 = min(rows), max(rows)
    c0, c1 = min(cols), max(cols)
    if r1 - r0 < 2 or c1 - c0 < 2:
        return None

    pts = set(component)
    expected_area = 2 * ((r1 - r0 + 1) + (c1 - c0 + 1)) - 4
    if len(pts) != expected_area:
        return None

    for c in range(c0, c1 + 1):
        if (r0, c) not in pts or (r1, c) not in pts:
            return None
    for r in range(r0, r1 + 1):
        if (r, c0) not in pts or (r, c1) not in pts:
            return None
    return r0, c0, r1, c1


def _paint_segment(out: np.ndarray, start: Tuple[int, int], end: Tuple[int, int], color: int) -> None:
    r0, c0 = start
    r1, c1 = end
    if r0 == r1:
        lo, hi = sorted((c0, c1))
        out[r0, lo : hi + 1] = color
        return
    if c0 == c1:
        lo, hi = sorted((r0, r1))
        out[lo : hi + 1, c0] = color
        return
    raise AssertionError("segment must be orthogonal")


def _detect_frames(grid: np.ndarray) -> Tuple[List[Dict[str, Tuple[int, int] | int]], Dict[int, Dict[str, Tuple[int, int] | int]]]:
    frames: List[Dict[str, Tuple[int, int] | int]] = []
    by_color: Dict[int, Dict[str, Tuple[int, int] | int]] = {}
    for component in _components(grid == FRAME_COLOR):
        bbox = _ring_bbox(component)
        if bbox is None:
            continue
        r0, c0, r1, c1 = bbox
        interior = grid[r0 + 1 : r1, c0 + 1 : c1]
        marker_rel = np.argwhere((interior != BG_COLOR) & (interior != FRAME_COLOR))
        assert len(marker_rel) == 1, "each frame must contain exactly one colored marker"
        mr = int(marker_rel[0, 0] + r0 + 1)
        mc = int(marker_rel[0, 1] + c0 + 1)
        color = int(grid[mr, mc])
        assert color not in by_color, "marker colors must identify frames uniquely"
        frame = {"bbox": (r0, c0, r1, c1), "marker": (mr, mc), "color": color}
        frames.append(frame)
        by_color[color] = frame
    assert frames, "expected at least one gray frame"
    return frames, by_color


def main(grid: np.ndarray) -> np.ndarray:
    """
    Draw straight colored injections from matching exterior seeds through gray frames to the interior marker.

    Input structure:
    - The background is black and one-pixel-thick gray rectangular frames sit on it.
    - Each gray frame encloses exactly one non-black, non-gray marker pixel.
    - Outside the frames, single colored seed pixels reuse those marker colors and align with the marker row or column.

    Rule:
    - Match every exterior seed to the unique frame whose interior marker has the same color.
    - From the seed, draw a straight orthogonal line to the aligned frame side, recoloring the contacted gray wall pixel to open an entrance.
    - Continue in the same direction inside the frame until the line reaches the interior marker pixel.
    - Leave all other pixels unchanged.

    Key insight:
    - The marker color resolves which frame a seed belongs to, while row/column alignment fixes the unique side and direction of the straight-through path.
    """
    g = np.asarray(grid, dtype=int)
    out = g.copy()
    frames, by_color = _detect_frames(g)

    marker_cells = {tuple(frame["marker"]) for frame in frames}
    colored = np.argwhere((g != BG_COLOR) & (g != FRAME_COLOR))
    for r, c in colored:
        coord = (int(r), int(c))
        if coord in marker_cells:
            continue
        color = int(g[coord])
        assert color in by_color, "every seed must match a frame marker color"
        frame = by_color[color]
        r0, c0, r1, c1 = frame["bbox"]
        mr, mc = frame["marker"]

        candidates = []
        if coord[0] == mr and coord[1] < c0:
            candidates.append((c0 - coord[1], (mr, c0), SIDE_TO_STEP["left"]))
        if coord[0] == mr and coord[1] > c1:
            candidates.append((coord[1] - c1, (mr, c1), SIDE_TO_STEP["right"]))
        if coord[1] == mc and coord[0] < r0:
            candidates.append((r0 - coord[0], (r0, mc), SIDE_TO_STEP["top"]))
        if coord[1] == mc and coord[0] > r1:
            candidates.append((coord[0] - r1, (r1, mc), SIDE_TO_STEP["bottom"]))
        assert candidates, "each seed must align with its frame and marker"

        candidates.sort(key=lambda item: item[0])
        if len(candidates) > 1:
            assert candidates[0][0] < candidates[1][0], "seed-to-frame alignment must be unambiguous"
        _, entry, step = candidates[0]
        dr, dc = step

        if dr == 0:
            assert entry[0] == mr and (mc - entry[1]) * dc >= 1
        else:
            assert entry[1] == mc and (mr - entry[0]) * dr >= 1

        _paint_segment(out, coord, entry, color)
        _paint_segment(out, entry, (mr, mc), color)

    return out


def _expanded(cells: Set[Tuple[int, int]], h: int, w: int) -> Set[Tuple[int, int]]:
    expanded: Set[Tuple[int, int]] = set()
    for r, c in cells:
        for nr in range(r - 1, r + 2):
            for nc in range(c - 1, c + 2):
                if 0 <= nr < h and 0 <= nc < w:
                    expanded.add((nr, nc))
    return expanded


def _frame_cells(r0: int, c0: int, r1: int, c1: int) -> Set[Tuple[int, int]]:
    return {(r, c) for r in range(r0, r1 + 1) for c in range(c0, c1 + 1)}


def _choose_seed_sides(rng: random.Random) -> List[str]:
    count = 1 if rng.random() < 0.6 else 2
    return rng.sample(list(SIDES), count)


def _candidate_seed_positions(
    side: str,
    marker: Tuple[int, int],
    bbox: Tuple[int, int, int, int],
    shape: Tuple[int, int],
    rng: random.Random,
) -> Optional[Tuple[Tuple[int, int], Set[Tuple[int, int]]]]:
    h, w = shape
    rm, mc = marker
    r0, c0, r1, c1 = bbox

    if side == "left":
        lo = max(0, c0 - MAX_OUTSIDE_LEN)
        choices = list(range(lo, c0 - 1))
        if not choices:
            return None
        sc = rng.choice(choices)
        seed = (rm, sc)
        corridor = {(rm, c) for c in range(sc, c0)}
        return seed, corridor
    if side == "right":
        hi = min(w - 1, c1 + MAX_OUTSIDE_LEN)
        choices = list(range(c1 + 2, hi + 1))
        if not choices:
            return None
        sc = rng.choice(choices)
        seed = (rm, sc)
        corridor = {(rm, c) for c in range(c1 + 1, sc + 1)}
        return seed, corridor
    if side == "top":
        lo = max(0, r0 - MAX_OUTSIDE_LEN)
        choices = list(range(lo, r0 - 1))
        if not choices:
            return None
        sr = rng.choice(choices)
        seed = (sr, mc)
        corridor = {(r, mc) for r in range(sr, r0)}
        return seed, corridor

    hi = min(h - 1, r1 + MAX_OUTSIDE_LEN)
    choices = list(range(r1 + 2, hi + 1))
    if not choices:
        return None
    sr = rng.choice(choices)
    seed = (sr, mc)
    corridor = {(r, mc) for r in range(r1 + 1, sr + 1)}
    return seed, corridor


def _try_place_frame(
    grid: np.ndarray,
    reserved: Set[Tuple[int, int]],
    color: int,
    rng: random.Random,
) -> bool:
    h, w = grid.shape
    for _ in range(PLACEMENT_TRIES):
        outer_h = rng.randint(MIN_OUTER, min(MAX_OUTER, h - 4))
        outer_w = rng.randint(MIN_OUTER, min(MAX_OUTER, w - 4))
        max_r0 = h - outer_h - 2
        max_c0 = w - outer_w - 2
        if max_r0 < 2 or max_c0 < 2:
            return False
        r0 = rng.randint(2, max_r0)
        c0 = rng.randint(2, max_c0)
        r1 = r0 + outer_h - 1
        c1 = c0 + outer_w - 1
        rm = rng.randint(r0 + 1, r1 - 1)
        mc = rng.randint(c0 + 1, c1 - 1)
        sides = _choose_seed_sides(rng)

        seeds: List[Tuple[int, int]] = []
        corridor: Set[Tuple[int, int]] = set()
        valid = True
        for side in sides:
            placed = _candidate_seed_positions(side, (rm, mc), (r0, c0, r1, c1), (h, w), rng)
            if placed is None:
                valid = False
                break
            seed, cells = placed
            seeds.append(seed)
            corridor |= cells
        if not valid:
            continue

        footprint = _frame_cells(r0, c0, r1, c1) | corridor
        claim = _expanded(footprint, h, w)
        if not claim.isdisjoint(reserved):
            continue

        grid[r0, c0 : c1 + 1] = FRAME_COLOR
        grid[r1, c0 : c1 + 1] = FRAME_COLOR
        grid[r0 : r1 + 1, c0] = FRAME_COLOR
        grid[r0 : r1 + 1, c1] = FRAME_COLOR
        grid[rm, mc] = color
        for sr, sc in seeds:
            grid[sr, sc] = color
        reserved |= claim
        return True
    return False


def _sample_frame_count(h: int, w: int, rng: random.Random) -> int:
    area_based = max(1, min(MAX_FRAMES, h * w // 150 + 1))
    return rng.randint(1, area_based)


def _gen_attempt(rng: random.Random) -> Optional[np.ndarray]:
    h = rng.randint(MIN_GRID, MAX_GRID)
    w = rng.randint(MIN_GRID, MAX_GRID)
    grid = np.zeros((h, w), dtype=int)
    reserved: Set[Tuple[int, int]] = set()

    n_frames = _sample_frame_count(h, w, rng)
    colors = rng.sample(list(PALETTE), n_frames)
    for color in colors:
        if not _try_place_frame(grid, reserved, color, rng):
            return None

    try:
        out = main(grid)
    except Exception:
        return None
    if np.array_equal(out, grid):
        return None
    return grid


def generate_input() -> np.ndarray:
    rng = random.Random()
    for _ in range(GEN_RETRIES):
        grid = _gen_attempt(rng)
        if grid is not None:
            return grid
    raise RuntimeError("failed to generate a valid input")
