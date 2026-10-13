from __future__ import annotations

import random

import numpy as np


BLACK = 0
BLUE = 1
RED = 2
YELLOW = 4
TEAL = 8

GRID_RANGE = (15, 19)
RECT_HEIGHTS = (2, 3)
RECT_WIDTHS = (3, 4)
MARKER_COUNTS = (2, 3, 4)
OFFSETS = (-1, 0, 1)
GEN_RETRIES = 400


def _in_bounds(shape: tuple[int, int], r: int, c: int) -> bool:
    h, w = shape
    return 0 <= r < h and 0 <= c < w


def _find_source_bbox(grid: np.ndarray) -> tuple[int, int, int, int]:
    teal = np.argwhere(grid == TEAL)
    assert teal.size > 0, "expected a teal source rectangle"
    r0, c0 = teal.min(axis=0)
    r1, c1 = teal.max(axis=0)
    r0, c0, r1, c1 = map(int, (r0, c0, r1, c1))
    assert np.all(grid[r0 : r1 + 1, c0 : c1 + 1] == TEAL), "source must be solid"
    assert int((grid == TEAL).sum()) == (r1 - r0 + 1) * (c1 - c0 + 1), "source must be rectangular"
    return r0, c0, r1, c1


def _marker_list(grid: np.ndarray) -> list[tuple[int, int]]:
    return [tuple(map(int, rc)) for rc in np.argwhere(grid == RED)]


def _blocker_mask(shape: tuple[int, int], markers: list[tuple[int, int]]) -> np.ndarray:
    mask = np.zeros(shape, dtype=bool)
    for r, c in markers:
        for rr in range(r - 1, r + 2):
            for cc in range(c - 1, c + 2):
                if _in_bounds(shape, rr, cc):
                    mask[rr, cc] = True
    return mask


def _ray_masks(
    shape: tuple[int, int],
    bbox: tuple[int, int, int, int],
    blocker_mask: np.ndarray,
) -> tuple[np.ndarray, np.ndarray, list[dict[str, int | str]]]:
    h, w = shape
    r0, c0, r1, c1 = bbox
    horizontal = np.zeros(shape, dtype=bool)
    vertical = np.zeros(shape, dtype=bool)
    infos: list[dict[str, int | str]] = []

    for r in range(r0, r1 + 1):
        for direction, start_c in (("left", c0 - 1), ("right", c1 + 1)):
            dc = -1 if direction == "left" else 1
            c = start_c
            painted = 0
            while 0 <= c < w and not blocker_mask[r, c]:
                horizontal[r, c] = True
                painted += 1
                c += dc
            infos.append(
                {
                    "axis": "h",
                    "direction": direction,
                    "index": r,
                    "painted": painted,
                    "reason": "edge" if not (0 <= c < w) else "blocker",
                }
            )

    for c in range(c0, c1 + 1):
        for direction, start_r in (("up", r0 - 1), ("down", r1 + 1)):
            dr = -1 if direction == "up" else 1
            r = start_r
            painted = 0
            while 0 <= r < h and not blocker_mask[r, c]:
                vertical[r, c] = True
                painted += 1
                r += dr
            infos.append(
                {
                    "axis": "v",
                    "direction": direction,
                    "index": c,
                    "painted": painted,
                    "reason": "edge" if not (0 <= r < h) else "blocker",
                }
            )

    return horizontal, vertical, infos


def _solve_from_parts(grid: np.ndarray) -> np.ndarray:
    bbox = _find_source_bbox(grid)
    markers = _marker_list(grid)
    blockers = _blocker_mask(grid.shape, markers)
    horizontal, vertical, _infos = _ray_masks(grid.shape, bbox, blockers)

    out = np.array(grid, dtype=int, copy=True)
    out[horizontal | vertical] = YELLOW
    out[bbox[0] : bbox[2] + 1, bbox[1] : bbox[3] + 1] = TEAL
    out[blockers & (out == BLACK)] = BLUE
    out[grid == RED] = RED
    return out


def main(grid: np.ndarray) -> np.ndarray:
    """
    Cast cross-shaped yellow rays from the teal rectangle, stopping them at the red markers' implicit 3x3 halo blockers, then reveal the halos in blue.

    Input structure:
    - A black background contains one solid teal rectangle and 2-4 isolated red marker pixels.
    - Blue and yellow do not appear in the input.
    - The red markers are separated so their Chebyshev-radius-1 halos do not overlap the teal rectangle or each other.

    Rule:
    - Around every red marker, compute an implicit Chebyshev-distance-1 blocking zone (its full 3x3 halo) without drawing it yet.
    - From every row of the teal rectangle, cast yellow rays left and right; from every column, cast yellow rays up and down.
    - Each ray paints cells until the next cell would enter a blocking halo or leave the grid; the blocking cell itself is not painted.
    - After all rays are drawn, color every still-black cell inside any blocking halo blue, while keeping the red markers and the teal rectangle unchanged.

    Key insight:
    - The visible blue halos are rendered only at the end, but the rays already stop at those larger hidden 3x3 halos rather than at the red centers themselves.
    """
    g = np.asarray(grid, dtype=int)
    assert g.ndim == 2
    assert max(g.shape) <= 30
    assert set(map(int, np.unique(g))) <= {BLACK, RED, TEAL}
    return _solve_from_parts(g)


def _naive_red_stop_mask(shape: tuple[int, int], bbox: tuple[int, int, int, int], markers: list[tuple[int, int]]) -> np.ndarray:
    red_mask = np.zeros(shape, dtype=bool)
    for r, c in markers:
        red_mask[r, c] = True
    horizontal, vertical, _infos = _ray_masks(shape, bbox, red_mask)
    return horizontal | vertical


def _chebyshev(a: tuple[int, int], b: tuple[int, int]) -> int:
    return max(abs(a[0] - b[0]), abs(a[1] - b[1]))


def _marker_candidates(
    shape: tuple[int, int],
    bbox: tuple[int, int, int, int],
    side: str,
    used: list[tuple[int, int]],
) -> list[dict[str, int | str | tuple[int, int]]]:
    h, w = shape
    r0, c0, r1, c1 = bbox
    candidates: list[dict[str, int | str | tuple[int, int]]] = []

    if side in ("left", "right"):
        base_values = list(range(r0, r1 + 1))
        line_values = list(range(1, c0 - 1)) if side == "left" else list(range(c1 + 2, w - 1))
        for base in base_values:
            for offset in OFFSETS:
                r = base + offset
                if not (0 <= r < h):
                    continue
                for c in line_values:
                    pos = (r, c)
                    if all(_chebyshev(pos, other) >= 3 for other in used):
                        candidates.append(
                            {
                                "side": side,
                                "base": base,
                                "offset": offset,
                                "pos": pos,
                                "distance": (c0 - c) if side == "left" else (c - c1),
                            }
                        )
    else:
        base_values = list(range(c0, c1 + 1))
        line_values = list(range(1, r0 - 1)) if side == "up" else list(range(r1 + 2, h - 1))
        for base in base_values:
            for offset in OFFSETS:
                c = base + offset
                if not (0 <= c < w):
                    continue
                for r in line_values:
                    pos = (r, c)
                    if all(_chebyshev(pos, other) >= 3 for other in used):
                        candidates.append(
                            {
                                "side": side,
                                "base": base,
                                "offset": offset,
                                "pos": pos,
                                "distance": (r0 - r) if side == "up" else (r - r1),
                            }
                        )
    return candidates


def _choose_marker(
    shape: tuple[int, int],
    bbox: tuple[int, int, int, int],
    side: str,
    used: list[tuple[int, int]],
    require_nonzero_offset: bool,
) -> dict[str, int | str | tuple[int, int]] | None:
    candidates = _marker_candidates(shape, bbox, side, used)
    if require_nonzero_offset:
        candidates = [cand for cand in candidates if int(cand["offset"]) != 0]
    if not candidates:
        return None

    preferred_distances = [2, 3, 4, 5, 6]
    random.shuffle(preferred_distances)
    for dist in preferred_distances:
        dist_matches = [cand for cand in candidates if int(cand["distance"]) == dist]
        if dist_matches:
            return random.choice(dist_matches)
    return random.choice(candidates)


def _sample_bbox(shape: tuple[int, int]) -> tuple[int, int, int, int]:
    h, w = shape
    rect_h = random.choice(RECT_HEIGHTS)
    rect_w = random.choice(RECT_WIDTHS)
    r0 = random.randint(4, h - rect_h - 5)
    c0 = random.randint(4, w - rect_w - 5)
    return r0, c0, r0 + rect_h - 1, c0 + rect_w - 1


def _render_input(
    shape: tuple[int, int],
    bbox: tuple[int, int, int, int],
    markers: list[tuple[int, int]],
) -> np.ndarray:
    grid = np.zeros(shape, dtype=int)
    r0, c0, r1, c1 = bbox
    grid[r0 : r1 + 1, c0 : c1 + 1] = TEAL
    for r, c in markers:
        grid[r, c] = RED
    return grid


def _ray_blocked_by_marker(
    marker: tuple[int, int],
    bbox: tuple[int, int, int, int],
    blockers: np.ndarray,
    infos: list[dict[str, int | str]],
) -> bool:
    mr, mc = marker
    r0, c0, r1, c1 = bbox
    halo_rows = range(max(0, mr - 1), min(blockers.shape[0], mr + 2))
    halo_cols = range(max(0, mc - 1), min(blockers.shape[1], mc + 2))

    for info in infos:
        if info["reason"] != "blocker":
            continue
        if info["axis"] == "h":
            r = int(info["index"])
            if r not in halo_rows:
                continue
            if info["direction"] == "left":
                stop_c = c0 - int(info["painted"]) - 1
            else:
                stop_c = c1 + int(info["painted"]) + 1
            if 0 <= stop_c < blockers.shape[1] and blockers[r, stop_c] and stop_c in halo_cols:
                return True
        else:
            c = int(info["index"])
            if c not in halo_cols:
                continue
            if info["direction"] == "up":
                stop_r = r0 - int(info["painted"]) - 1
            else:
                stop_r = r1 + int(info["painted"]) + 1
            if 0 <= stop_r < blockers.shape[0] and blockers[stop_r, c] and stop_r in halo_rows:
                return True
    return False


def _validate_instance(grid: np.ndarray, marker_specs: list[dict[str, int | str | tuple[int, int]]]) -> bool:
    if grid.ndim != 2 or max(grid.shape) > 30:
        return False
    if set(map(int, np.unique(grid))) != {BLACK, RED, TEAL}:
        return False

    bbox = _find_source_bbox(grid)
    r0, c0, r1, c1 = bbox
    if (r1 - r0 + 1) not in RECT_HEIGHTS or (c1 - c0 + 1) not in RECT_WIDTHS:
        return False

    markers = _marker_list(grid)
    if len(markers) not in MARKER_COUNTS:
        return False
    if len(set(markers)) != len(markers):
        return False
    if not any(int(spec["offset"]) != 0 for spec in marker_specs):
        return False

    for i, marker in enumerate(markers):
        mr, mc = marker
        if r0 - 1 <= mr <= r1 + 1 and c0 - 1 <= mc <= c1 + 1:
            return False
        for other in markers[i + 1 :]:
            if _chebyshev(marker, other) < 3:
                return False

    blockers = _blocker_mask(grid.shape, markers)
    if blockers[r0 : r1 + 1, c0 : c1 + 1].any():
        return False

    horizontal, vertical, infos = _ray_masks(grid.shape, bbox, blockers)
    yellow = horizontal | vertical
    if yellow.sum() < 8:
        return False
    if not horizontal.any() or not vertical.any():
        return False
    if not any(info["reason"] == "blocker" for info in infos):
        return False
    if not any(info["reason"] == "edge" for info in infos):
        return False

    naive = _naive_red_stop_mask(grid.shape, bbox, markers)
    if np.array_equal(naive, yellow):
        return False

    for marker in markers:
        reduced = [m for m in markers if m != marker]
        reduced_blockers = _blocker_mask(grid.shape, reduced)
        reduced_h, reduced_v, _ = _ray_masks(grid.shape, bbox, reduced_blockers)
        if np.array_equal(reduced_h | reduced_v, yellow):
            return False
        if not _ray_blocked_by_marker(marker, bbox, blockers, infos):
            return False

    out = main(grid)
    if int((out == BLUE).sum()) < 8:
        return False
    if int((out == YELLOW).sum()) < 8:
        return False
    if np.any(out[grid == RED] != RED):
        return False
    if np.any(out[grid == TEAL] != TEAL):
        return False
    return True


def _attempt_generate() -> np.ndarray | None:
    shape = (random.randint(*GRID_RANGE), random.randint(*GRID_RANGE))
    bbox = _sample_bbox(shape)
    marker_count = random.choice(MARKER_COUNTS)

    horizontal_side = random.choice(["left", "right"])
    vertical_side = random.choice(["up", "down"])
    sides = [horizontal_side, vertical_side]
    leftovers = [side for side in ["left", "right", "up", "down"] if side not in sides]
    random.shuffle(leftovers)
    while len(sides) < marker_count:
        sides.append(leftovers.pop())
    random.shuffle(sides)

    forced_nonzero_side = random.choice(sides)
    marker_specs: list[dict[str, int | str | tuple[int, int]]] = []
    used: list[tuple[int, int]] = []
    for side in sides:
        spec = _choose_marker(
            shape,
            bbox,
            side,
            used,
            require_nonzero_offset=(side == forced_nonzero_side),
        )
        if spec is None:
            return None
        pos = tuple(spec["pos"])
        marker_specs.append(spec)
        used.append(pos)

    markers = [tuple(spec["pos"]) for spec in marker_specs]
    grid = _render_input(shape, bbox, markers)
    if not _validate_instance(grid, marker_specs):
        return None
    return grid


def generate_input() -> np.ndarray:
    for _ in range(GEN_RETRIES):
        grid = _attempt_generate()
        if grid is not None:
            return grid
    raise RuntimeError("failed to generate a valid hidden-halo ray puzzle input")
