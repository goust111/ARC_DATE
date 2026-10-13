from __future__ import annotations

from collections import deque
from dataclasses import dataclass
import random

import numpy as np


BG = 0
BLUE = 1
RED = 2
GREEN = 3
YELLOW = 4
ORANGE = 7

DIR_BY_COLOR = {
    RED: (1, 0),
    GREEN: (-1, 0),
    YELLOW: (0, 1),
    ORANGE: (0, -1),
}

DIR_COLORS = tuple(DIR_BY_COLOR)
GEN_ATTEMPTS = 300
MIN_GRID = 18
MAX_GRID = 28
MIN_LOCAL_SHAPE = 12
MIN_INTERIOR_CELLS = 24
MIN_RAY_LEN = 2


@dataclass(frozen=True)
class Candidate:
    pos: tuple[int, int]
    color: int
    ray: tuple[tuple[int, int], ...]
    ray_set: frozenset[tuple[int, int]]


def main(grid: np.ndarray) -> np.ndarray:
    """
    Grow colored rays from interior markers until they hit the enclosing blue outline.

    Input structure:
    - The grid has a black background and one large closed blue outline with thickness one cell.
    - The outline encloses a black interior region containing several isolated colored marker pixels.
    - Marker colors are limited to red, green, yellow, and orange, each encoding one cardinal direction.

    Rule:
    - Keep the blue outline and every exterior cell unchanged.
    - Detect the cells enclosed by the blue outline by flood-filling from the grid border through non-blue cells.
    - Scan enclosed marker pixels from top to bottom, then left to right, and extend a same-color ray from each marker:
      red grows downward, green upward, yellow rightward, and orange leftward.
    - Each ray fills every traversed non-blue cell up to but not including the first blue outline cell; later rays overwrite earlier crossings, and all original marker pixels are restored afterward.

    Key insight:
    - The semantic information is entirely encoded by marker color plus membership inside the unique closed outline; once the enclosed region is isolated, the task reduces to ordered directional line growth.
    """
    g = np.asarray(grid, dtype=int)
    h, w = g.shape

    outside = _outside_mask(g, BLUE)
    inside = (g != BLUE) & (~outside)

    markers: list[tuple[int, int, int]] = []
    for r in range(h):
        for c in range(w):
            color = int(g[r, c])
            if inside[r, c] and color in DIR_BY_COLOR:
                markers.append((r, c, color))

    out = g.copy()
    for r, c, color in markers:
        dr, dc = DIR_BY_COLOR[color]
        rr, cc = r + dr, c + dc
        while 0 <= rr < h and 0 <= cc < w and int(g[rr, cc]) != BLUE:
            out[rr, cc] = color
            rr += dr
            cc += dc

    for r, c, color in markers:
        out[r, c] = color
    return out


def _outside_mask(grid: np.ndarray, wall: int) -> np.ndarray:
    h, w = grid.shape
    seen = np.zeros((h, w), dtype=bool)
    q: deque[tuple[int, int]] = deque()

    for r in range(h):
        for c in (0, w - 1):
            if not seen[r, c] and int(grid[r, c]) != wall:
                seen[r, c] = True
                q.append((r, c))
    for c in range(w):
        for r in (0, h - 1):
            if not seen[r, c] and int(grid[r, c]) != wall:
                seen[r, c] = True
                q.append((r, c))

    while q:
        r, c = q.popleft()
        for dr, dc in ((1, 0), (-1, 0), (0, 1), (0, -1)):
            rr, cc = r + dr, c + dc
            if 0 <= rr < h and 0 <= cc < w and not seen[rr, cc] and int(grid[rr, cc]) != wall:
                seen[rr, cc] = True
                q.append((rr, cc))
    return seen


def _boundary_of(mask: np.ndarray) -> np.ndarray:
    h, w = mask.shape
    boundary = np.zeros_like(mask, dtype=bool)
    coords = np.argwhere(mask)
    for r, c in coords:
        r = int(r)
        c = int(c)
        if r == 0 or c == 0 or r == h - 1 or c == w - 1:
            boundary[r, c] = True
            continue
        if not (mask[r - 1, c] and mask[r + 1, c] and mask[r, c - 1] and mask[r, c + 1]):
            boundary[r, c] = True
    return boundary


def _component(mask: np.ndarray, start: tuple[int, int]) -> set[tuple[int, int]]:
    h, w = mask.shape
    comp = {start}
    q: deque[tuple[int, int]] = deque([start])
    while q:
        r, c = q.popleft()
        for dr, dc in ((1, 0), (-1, 0), (0, 1), (0, -1)):
            rr, cc = r + dr, c + dc
            if 0 <= rr < h and 0 <= cc < w and mask[rr, cc] and (rr, cc) not in comp:
                comp.add((rr, cc))
                q.append((rr, cc))
    return comp


def _trim(mask: np.ndarray) -> np.ndarray:
    rows = np.flatnonzero(mask.any(axis=1))
    cols = np.flatnonzero(mask.any(axis=0))
    return mask[rows[0] : rows[-1] + 1, cols[0] : cols[-1] + 1]


def _make_shape_mask(rng: random.Random, h: int, w: int) -> np.ndarray | None:
    mask = np.zeros((h, w), dtype=bool)

    base_h = rng.randint(max(7, h // 2), h - 4)
    base_w = rng.randint(max(7, w // 2), w - 4)
    base_top = rng.randint(2, h - base_h - 2)
    base_left = rng.randint(2, w - base_w - 2)
    mask[base_top : base_top + base_h, base_left : base_left + base_w] = True

    arm_count = rng.randint(1, 4)
    for _ in range(arm_count):
        feasible: list[str] = []
        if base_top >= 2:
            feasible.append("top")
        if h - (base_top + base_h) >= 2:
            feasible.append("bottom")
        if base_left >= 2:
            feasible.append("left")
        if w - (base_left + base_w) >= 2:
            feasible.append("right")
        if not feasible:
            break
        side = rng.choice(feasible)
        if side == "top":
            arm_h = rng.randint(2, base_top)
            arm_w = rng.randint(3, base_w - 1)
            arm_left = rng.randint(base_left, base_left + base_w - arm_w)
            mask[base_top - arm_h : base_top, arm_left : arm_left + arm_w] = True
        elif side == "bottom":
            max_h = h - (base_top + base_h)
            arm_h = rng.randint(2, max_h)
            arm_w = rng.randint(3, base_w - 1)
            arm_left = rng.randint(base_left, base_left + base_w - arm_w)
            mask[base_top + base_h : base_top + base_h + arm_h, arm_left : arm_left + arm_w] = True
        elif side == "left":
            arm_w = rng.randint(2, base_left)
            arm_h = rng.randint(3, base_h - 1)
            arm_top = rng.randint(base_top, base_top + base_h - arm_h)
            mask[arm_top : arm_top + arm_h, base_left - arm_w : base_left] = True
        else:
            max_w = w - (base_left + base_w)
            arm_w = rng.randint(2, max_w)
            arm_h = rng.randint(3, base_h - 1)
            arm_top = rng.randint(base_top, base_top + base_h - arm_h)
            mask[arm_top : arm_top + arm_h, base_left + base_w : base_left + base_w + arm_w] = True

    mask = _trim(mask)
    boundary = _boundary_of(mask)
    interior = mask & (~boundary)
    if interior.sum() < MIN_INTERIOR_CELLS:
        return None

    start = tuple(map(int, np.argwhere(mask)[0]))
    if len(_component(mask, start)) != int(mask.sum()):
        return None

    interior_pts = np.argwhere(interior)
    interior_start = tuple(map(int, interior_pts[0]))
    if len(_component(interior, interior_start)) != int(interior.sum()):
        return None
    return mask


def _ray_cells(grid: np.ndarray, pos: tuple[int, int], color: int) -> tuple[tuple[int, int], ...]:
    h, w = grid.shape
    r, c = pos
    dr, dc = DIR_BY_COLOR[color]
    ray: list[tuple[int, int]] = []
    rr, cc = r + dr, c + dc
    while 0 <= rr < h and 0 <= cc < w and int(grid[rr, cc]) != BLUE:
        ray.append((rr, cc))
        rr += dr
        cc += dc
    if not (0 <= rr < h and 0 <= cc < w):
        return ()
    return tuple(ray)


def _all_candidates(grid: np.ndarray, interior_cells: list[tuple[int, int]]) -> dict[int, list[Candidate]]:
    by_color = {color: [] for color in DIR_COLORS}
    for pos in interior_cells:
        for color in DIR_COLORS:
            ray = _ray_cells(grid, pos, color)
            if len(ray) >= MIN_RAY_LEN:
                by_color[color].append(Candidate(pos, color, ray, frozenset(ray)))
    return by_color


def _pick_candidate(
    rng: random.Random,
    pool: list[Candidate],
    marker_positions: set[tuple[int, int]],
    ray_cells: set[tuple[int, int]],
) -> Candidate | None:
    order = pool[:]
    rng.shuffle(order)
    for cand in order:
        if cand.pos in marker_positions:
            continue
        if cand.pos in ray_cells:
            continue
        if cand.ray_set & marker_positions:
            continue
        return cand
    return None


def _find_crossing_pairs(candidates: dict[int, list[Candidate]]) -> list[tuple[Candidate, Candidate]]:
    vertical = candidates[RED] + candidates[GREEN]
    horizontal = candidates[YELLOW] + candidates[ORANGE]
    pairs: list[tuple[Candidate, Candidate]] = []
    for vcand in vertical:
        for hcand in horizontal:
            if vcand.pos == hcand.pos:
                continue
            if hcand.pos in vcand.ray_set or vcand.pos in hcand.ray_set:
                continue
            if vcand.ray_set & hcand.ray_set:
                pairs.append((vcand, hcand))
    return pairs


def _place_markers(
    rng: random.Random,
    grid: np.ndarray,
    interior_cells: list[tuple[int, int]],
) -> list[Candidate] | None:
    candidates = _all_candidates(grid, interior_cells)
    available_colors = [color for color in DIR_COLORS if candidates[color]]
    if len(available_colors) < 2:
        return None

    chosen: list[Candidate] = []
    marker_positions: set[tuple[int, int]] = set()
    ray_cells: set[tuple[int, int]] = set()

    if len(available_colors) == 4 and rng.random() < 0.5:
        distinct_target = 4
    else:
        distinct_target = rng.randint(2, len(available_colors))
    required_colors = rng.sample(available_colors, distinct_target)

    if rng.random() < 0.6:
        crossing_pairs = _find_crossing_pairs(candidates)
        rng.shuffle(crossing_pairs)
        for a, b in crossing_pairs:
            if a.pos == b.pos:
                continue
            if a.pos in ray_cells or b.pos in ray_cells:
                continue
            if a.ray_set & marker_positions or b.ray_set & marker_positions:
                continue
            chosen.extend([a, b])
            marker_positions.update((a.pos, b.pos))
            ray_cells.update(a.ray_set)
            ray_cells.update(b.ray_set)
            break

    for color in required_colors:
        if any(cand.color == color for cand in chosen):
            continue
        cand = _pick_candidate(rng, candidates[color], marker_positions, ray_cells)
        if cand is None:
            return None
        chosen.append(cand)
        marker_positions.add(cand.pos)
        ray_cells.update(cand.ray_set)

    max_markers = min(6, max(2, len(interior_cells) // 12 + 2))
    target_count = rng.randint(max(distinct_target, 2), max_markers)
    all_pool = [cand for color in available_colors for cand in candidates[color]]
    while len(chosen) < target_count:
        cand = _pick_candidate(rng, all_pool, marker_positions, ray_cells)
        if cand is None:
            break
        chosen.append(cand)
        marker_positions.add(cand.pos)
        ray_cells.update(cand.ray_set)

    if len(chosen) < 2:
        return None
    return chosen


def _gen_attempt(rng: random.Random) -> np.ndarray | None:
    h = rng.randint(MIN_GRID, MAX_GRID)
    w = rng.randint(MIN_GRID, MAX_GRID)

    local_h = rng.randint(MIN_LOCAL_SHAPE, h - 2)
    local_w = rng.randint(MIN_LOCAL_SHAPE, w - 2)
    shape = _make_shape_mask(rng, local_h, local_w)
    if shape is None:
        return None

    boundary = _boundary_of(shape)
    interior = shape & (~boundary)
    sh, sw = shape.shape
    top = rng.randint(1, h - sh - 1)
    left = rng.randint(1, w - sw - 1)

    grid = np.full((h, w), BG, dtype=int)
    grid[top : top + sh, left : left + sw][boundary] = BLUE

    interior_cells = [
        (top + int(r), left + int(c))
        for r, c in np.argwhere(interior)
    ]
    markers = _place_markers(rng, grid, interior_cells)
    if markers is None:
        return None

    for cand in markers:
        r, c = cand.pos
        grid[r, c] = cand.color

    out = main(grid)
    if np.array_equal(out, grid):
        return None
    return grid


def generate_input() -> np.ndarray:
    rng = random.Random()
    for _ in range(GEN_ATTEMPTS):
        grid = _gen_attempt(rng)
        if grid is not None:
            return grid
    raise RuntimeError("failed to generate a valid input")
