from __future__ import annotations

from collections import deque
from dataclasses import dataclass

import numpy as np


BG = 0
RED = 2
GREEN = 3
ORANGE = 7

DIRS = {
    "up": (-1, 0),
    "down": (1, 0),
    "left": (0, -1),
    "right": (0, 1),
}

RECT_H_RANGE = (3, 6)
RECT_W_RANGE = (3, 6)
GRID_H_RANGE = (20, 31)
GRID_W_RANGE = (20, 31)
CHAIN_LEN_RANGE = (2, 5)
DISTRACTOR_LEN_RANGE = (2, 4)
GAP_RANGE = (2, 7)
GEN_ATTEMPTS = 400
EDGE_ATTEMPTS = 120


@dataclass
class Rect:
    bbox: tuple[int, int, int, int]
    base_color: int
    indicator: tuple[int, int] | None = None
    direction: tuple[int, int] | None = None
    cells: list[tuple[int, int]] | None = None
    body_cells: list[tuple[int, int]] | None = None


def _in_bounds(shape: tuple[int, int], r: int, c: int) -> bool:
    h, w = shape
    return 0 <= r < h and 0 <= c < w


def _bbox_cells(bbox: tuple[int, int, int, int]) -> list[tuple[int, int]]:
    r0, c0, r1, c1 = bbox
    return [(r, c) for r in range(r0, r1 + 1) for c in range(c0, c1 + 1)]


def _side_positions(bbox: tuple[int, int, int, int], direction: tuple[int, int]) -> list[tuple[int, int]]:
    r0, c0, r1, c1 = bbox
    if direction == DIRS["up"]:
        return [(r0, c) for c in range(c0 + 1, c1)]
    if direction == DIRS["down"]:
        return [(r1, c) for c in range(c0 + 1, c1)]
    if direction == DIRS["left"]:
        return [(r, c0) for r in range(r0 + 1, r1)]
    if direction == DIRS["right"]:
        return [(r, c1) for r in range(r0 + 1, r1)]
    raise ValueError(direction)


def _components(mask: np.ndarray) -> list[list[tuple[int, int]]]:
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
                rr, cc = q.popleft()
                comp.append((rr, cc))
                for dr, dc in DIRS.values():
                    nr, nc = rr + dr, cc + dc
                    if _in_bounds(mask.shape, nr, nc) and mask[nr, nc] and not seen[nr, nc]:
                        seen[nr, nc] = True
                        q.append((nr, nc))
            comps.append(comp)
    return comps


def _parse_rectangles(grid: np.ndarray) -> list[Rect]:
    rects: list[Rect] = []
    for comp in _components(grid != BG):
        rows = [r for r, _ in comp]
        cols = [c for _, c in comp]
        r0, r1 = min(rows), max(rows)
        c0, c1 = min(cols), max(cols)
        bbox = (r0, c0, r1, c1)
        area = (r1 - r0 + 1) * (c1 - c0 + 1)
        assert len(comp) == area, "each object must be a solid rectangle with one indicator pixel"

        red_cells = [(r, c) for r, c in comp if int(grid[r, c]) == RED]
        assert len(red_cells) == 1, "each rectangle must have exactly one red indicator"
        indicator = red_cells[0]

        colors = {int(grid[r, c]) for r, c in comp if int(grid[r, c]) != RED}
        assert len(colors) == 1 and colors <= {GREEN, ORANGE}, "rectangle body must be green or orange"
        base_color = next(iter(colors))

        side_dirs = []
        ir, ic = indicator
        if ir == r0 and c0 < ic < c1:
            side_dirs.append(DIRS["up"])
        if ir == r1 and c0 < ic < c1:
            side_dirs.append(DIRS["down"])
        if ic == c0 and r0 < ir < r1:
            side_dirs.append(DIRS["left"])
        if ic == c1 and r0 < ir < r1:
            side_dirs.append(DIRS["right"])
        assert len(side_dirs) == 1, "indicator must lie on exactly one non-corner side"

        body_cells = [(r, c) for r, c in comp if (r, c) != indicator]
        rects.append(
            Rect(
                bbox=bbox,
                base_color=base_color,
                indicator=indicator,
                direction=side_dirs[0],
                cells=comp,
                body_cells=body_cells,
            )
        )
    assert rects, "expected at least one rectangle"
    return rects


def _owner_grid(shape: tuple[int, int], rects: list[Rect]) -> np.ndarray:
    owner = np.full(shape, -1, dtype=int)
    for idx, rect in enumerate(rects):
        assert rect.cells is not None
        for r, c in rect.cells:
            owner[r, c] = idx
    return owner


def _trace_ray(
    start: tuple[int, int], direction: tuple[int, int], owner: np.ndarray
) -> tuple[list[tuple[int, int]], int | None]:
    dr, dc = direction
    r, c = start[0] + dr, start[1] + dc
    path: list[tuple[int, int]] = []
    while _in_bounds(owner.shape, r, c):
        hit = int(owner[r, c])
        if hit >= 0:
            return path, hit
        path.append((r, c))
        r += dr
        c += dc
    return path, None


def main(grid: np.ndarray) -> np.ndarray:
    """
    Ignite the directed rectangle chain reachable from the unique orange seed and draw only those fired rays.

    Input structure:
    - Black background with several solid axis-aligned rectangles.
    - Each rectangle has exactly one red border pixel on a non-corner side, which indicates the outward ray direction.
    - Exactly one rectangle is orange (the seed); every other rectangle body is green.

    Rule:
    - Start from the unique orange seed rectangle.
    - Every ignited rectangle emits one red ray straight outward from its red indicator pixel until the ray exits the grid or first hits another rectangle.
    - A hit green rectangle turns orange, becomes ignited, and later emits its own ray; unreachable green rectangles stay green and never emit.

    Key insight:
    - Each rectangle defines one directed edge candidate, but the output is only the reachable closure from the seed, not the result of firing every rectangle.
    """
    g = np.asarray(grid, dtype=int)
    assert g.ndim == 2, "expected a 2D grid"

    rects = _parse_rectangles(g)
    owner = _owner_grid(g.shape, rects)

    seed_ids = [idx for idx, rect in enumerate(rects) if rect.base_color == ORANGE]
    assert len(seed_ids) == 1, "expected exactly one orange seed rectangle"
    seed = seed_ids[0]

    out = g.copy()
    ignited = {seed}
    q = deque([seed])

    while q:
        idx = q.popleft()
        rect = rects[idx]
        assert rect.indicator is not None and rect.direction is not None
        path, hit = _trace_ray(rect.indicator, rect.direction, owner)
        for r, c in path:
            out[r, c] = RED
        if hit is None or hit in ignited:
            continue

        ignited.add(hit)
        target = rects[hit]
        assert target.body_cells is not None
        for r, c in target.body_cells:
            out[r, c] = ORANGE
        q.append(hit)

    return out


def _mark_rect_masks(
    bbox: tuple[int, int, int, int], actual: np.ndarray, padded: np.ndarray
) -> None:
    h, w = actual.shape
    r0, c0, r1, c1 = bbox
    actual[r0 : r1 + 1, c0 : c1 + 1] = True
    pr0 = max(0, r0 - 1)
    pc0 = max(0, c0 - 1)
    pr1 = min(h - 1, r1 + 1)
    pc1 = min(w - 1, c1 + 1)
    padded[pr0 : pr1 + 1, pc0 : pc1 + 1] = True


def _bbox_clear(
    bbox: tuple[int, int, int, int],
    padded: np.ndarray,
    forbidden: np.ndarray,
) -> bool:
    r0, c0, r1, c1 = bbox
    return not padded[r0 : r1 + 1, c0 : c1 + 1].any() and not forbidden[r0 : r1 + 1, c0 : c1 + 1].any()


def _ray_path_between(
    source: tuple[int, int], direction: tuple[int, int], target_bbox: tuple[int, int, int, int]
) -> list[tuple[int, int]]:
    r0, c0, r1, c1 = target_bbox
    sr, sc = source
    if direction == DIRS["up"]:
        return [(r, sc) for r in range(sr - 1, r1, -1)]
    if direction == DIRS["down"]:
        return [(r, sc) for r in range(sr + 1, r0)]
    if direction == DIRS["left"]:
        return [(sr, c) for c in range(sc - 1, c1, -1)]
    if direction == DIRS["right"]:
        return [(sr, c) for c in range(sc + 1, c0)]
    raise ValueError(direction)


def _path_hits_existing(path: list[tuple[int, int]], actual: np.ndarray) -> bool:
    return any(actual[r, c] for r, c in path)


def _miss_candidates(
    bbox: tuple[int, int, int, int], actual: np.ndarray
) -> list[tuple[tuple[int, int], tuple[int, int], list[tuple[int, int]]]]:
    h, w = actual.shape
    candidates = []
    for direction in DIRS.values():
        for indicator in _side_positions(bbox, direction):
            dr, dc = direction
            r, c = indicator[0] + dr, indicator[1] + dc
            path: list[tuple[int, int]] = []
            ok = True
            while _in_bounds((h, w), r, c):
                if actual[r, c]:
                    ok = False
                    break
                path.append((r, c))
                r += dr
                c += dc
            if ok and path:
                candidates.append((indicator, direction, path))
    return candidates


def _extend_from_rect(
    bbox: tuple[int, int, int, int],
    actual: np.ndarray,
    padded: np.ndarray,
    forbidden: np.ndarray,
    rng: np.random.Generator,
) -> tuple[tuple[int, int, int, int], tuple[int, int], tuple[int, int], list[tuple[int, int]]] | None:
    h, w = actual.shape
    directions = list(DIRS.values())
    rng.shuffle(directions)
    for _ in range(EDGE_ATTEMPTS):
        direction = directions[int(rng.integers(0, len(directions)))]
        side_positions = _side_positions(bbox, direction)
        if not side_positions:
            continue
        indicator = side_positions[int(rng.integers(0, len(side_positions)))]
        rect_h = int(rng.integers(*RECT_H_RANGE))
        rect_w = int(rng.integers(*RECT_W_RANGE))
        gap = int(rng.integers(*GAP_RANGE))
        sr, sc = indicator

        if direction == DIRS["up"]:
            target_r1 = sr - gap - 1
            target_r0 = target_r1 - rect_h + 1
            c_lo = max(1, sc - rect_w + 1)
            c_hi = min(sc, w - rect_w - 1)
            if target_r0 < 1 or c_lo > c_hi:
                continue
            target_c0 = int(rng.integers(c_lo, c_hi + 1))
            target_bbox = (target_r0, target_c0, target_r1, target_c0 + rect_w - 1)
        elif direction == DIRS["down"]:
            target_r0 = sr + gap + 1
            target_r1 = target_r0 + rect_h - 1
            c_lo = max(1, sc - rect_w + 1)
            c_hi = min(sc, w - rect_w - 1)
            if target_r1 > h - 2 or c_lo > c_hi:
                continue
            target_c0 = int(rng.integers(c_lo, c_hi + 1))
            target_bbox = (target_r0, target_c0, target_r1, target_c0 + rect_w - 1)
        elif direction == DIRS["left"]:
            target_c1 = sc - gap - 1
            target_c0 = target_c1 - rect_w + 1
            r_lo = max(1, sr - rect_h + 1)
            r_hi = min(sr, h - rect_h - 1)
            if target_c0 < 1 or r_lo > r_hi:
                continue
            target_r0 = int(rng.integers(r_lo, r_hi + 1))
            target_bbox = (target_r0, target_c0, target_r0 + rect_h - 1, target_c1)
        else:
            target_c0 = sc + gap + 1
            target_c1 = target_c0 + rect_w - 1
            r_lo = max(1, sr - rect_h + 1)
            r_hi = min(sr, h - rect_h - 1)
            if target_c1 > w - 2 or r_lo > r_hi:
                continue
            target_r0 = int(rng.integers(r_lo, r_hi + 1))
            target_bbox = (target_r0, target_c0, target_r0 + rect_h - 1, target_c1)

        if not _bbox_clear(target_bbox, padded, forbidden):
            continue
        path = _ray_path_between(indicator, direction, target_bbox)
        if not path or _path_hits_existing(path, actual):
            continue
        return target_bbox, indicator, direction, path
    return None


def _place_random_root(
    actual: np.ndarray,
    padded: np.ndarray,
    forbidden: np.ndarray,
    base_color: int,
    rng: np.random.Generator,
) -> Rect | None:
    h, w = actual.shape
    for _ in range(EDGE_ATTEMPTS):
        rect_h = int(rng.integers(*RECT_H_RANGE))
        rect_w = int(rng.integers(*RECT_W_RANGE))
        if h - rect_h - 1 <= 1 or w - rect_w - 1 <= 1:
            return None
        r0 = int(rng.integers(1, h - rect_h))
        c0 = int(rng.integers(1, w - rect_w))
        bbox = (r0, c0, r0 + rect_h - 1, c0 + rect_w - 1)
        if _bbox_clear(bbox, padded, forbidden):
            _mark_rect_masks(bbox, actual, padded)
            return Rect(bbox=bbox, base_color=base_color)
    return None


def _build_chain(
    length: int,
    first_color: int,
    actual: np.ndarray,
    padded: np.ndarray,
    global_forbidden: np.ndarray,
    rng: np.random.Generator,
    reserve_globally: bool,
) -> list[Rect] | None:
    local_reserved = np.zeros_like(actual, dtype=bool)
    forbidden = global_forbidden.copy()
    root = _place_random_root(actual, padded, forbidden, first_color, rng)
    if root is None:
        return None
    rects = [root]

    for _ in range(length - 1):
        forbidden = global_forbidden | local_reserved
        edge = _extend_from_rect(rects[-1].bbox, actual, padded, forbidden, rng)
        if edge is None:
            return None
        target_bbox, indicator, direction, path = edge
        rects[-1].indicator = indicator
        rects[-1].direction = direction
        mark_mask = global_forbidden if reserve_globally else local_reserved
        for r, c in path:
            mark_mask[r, c] = True
        target = Rect(bbox=target_bbox, base_color=GREEN)
        _mark_rect_masks(target_bbox, actual, padded)
        rects.append(target)

    miss_forbidden = global_forbidden | local_reserved
    candidates = _miss_candidates(rects[-1].bbox, actual)
    candidates = [cand for cand in candidates if all(not miss_forbidden[r, c] for r, c in cand[2])]
    if not candidates:
        return None
    indicator, direction, path = candidates[int(rng.integers(0, len(candidates)))]
    rects[-1].indicator = indicator
    rects[-1].direction = direction
    mark_mask = global_forbidden if reserve_globally else local_reserved
    for r, c in path:
        mark_mask[r, c] = True

    return rects


def _build_input_grid(rects: list[Rect], shape: tuple[int, int]) -> np.ndarray:
    grid = np.full(shape, BG, dtype=int)
    for rect in rects:
        r0, c0, r1, c1 = rect.bbox
        grid[r0 : r1 + 1, c0 : c1 + 1] = rect.base_color
    for rect in rects:
        assert rect.indicator is not None
        grid[rect.indicator] = RED
    return grid


def _valid_scene(inp: np.ndarray, out: np.ndarray) -> bool:
    if not np.any(inp == ORANGE):
        return False
    if not np.any(inp == GREEN):
        return False
    if np.count_nonzero(out == RED) <= np.count_nonzero(inp == RED):
        return False
    return np.any((inp == GREEN) & (out == GREEN))


def _gen_attempt() -> np.ndarray | None:
    rng = np.random.default_rng()
    h = int(rng.integers(*GRID_H_RANGE))
    w = int(rng.integers(*GRID_W_RANGE))
    actual = np.zeros((h, w), dtype=bool)
    padded = np.zeros((h, w), dtype=bool)
    reserved_reachable = np.zeros((h, w), dtype=bool)

    reachable_len = int(rng.integers(*CHAIN_LEN_RANGE))
    distractor_len = int(rng.integers(*DISTRACTOR_LEN_RANGE))

    reachable = _build_chain(
        reachable_len,
        ORANGE,
        actual,
        padded,
        reserved_reachable,
        rng,
        reserve_globally=True,
    )
    if reachable is None:
        return None

    distractors = _build_chain(
        distractor_len,
        GREEN,
        actual,
        padded,
        reserved_reachable,
        rng,
        reserve_globally=False,
    )
    if distractors is None:
        return None

    all_rects = reachable + distractors
    inp = _build_input_grid(all_rects, (h, w))
    try:
        out = main(inp)
    except Exception:
        return None
    if not _valid_scene(inp, out):
        return None
    return inp


def generate_input() -> np.ndarray:
    for _ in range(GEN_ATTEMPTS):
        grid = _gen_attempt()
        if grid is not None:
            return grid
    raise RuntimeError("failed to generate valid input")
