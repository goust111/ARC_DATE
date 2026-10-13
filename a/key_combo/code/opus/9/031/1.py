from __future__ import annotations

import random
from collections import deque

import numpy as np


BLACK = 0
WALL = 1
MARKER_COLORS = (2, 3, 4)
START = 6
DIAGONALS = ((-1, -1), (-1, 1), (1, -1), (1, 1))
GEN_ATTEMPTS = 400


def _in_bounds(shape: tuple[int, int], r: int, c: int) -> bool:
    h, w = shape
    return 0 <= r < h and 0 <= c < w


def _neighbors4(r: int, c: int):
    yield r - 1, c
    yield r + 1, c
    yield r, c - 1
    yield r, c + 1


def _flood_reachable(grid: np.ndarray, start: tuple[int, int]) -> np.ndarray:
    """Reachable black cells from the corner marker using 4-connectivity."""
    h, w = grid.shape
    reach = np.zeros((h, w), dtype=bool)
    q = deque([start])
    seen = {start}
    while q:
        r, c = q.popleft()
        for nr, nc in _neighbors4(r, c):
            if not _in_bounds((h, w), nr, nc) or (nr, nc) in seen:
                continue
            seen.add((nr, nc))
            if int(grid[nr, nc]) != BLACK:
                continue
            reach[nr, nc] = True
            q.append((nr, nc))
    return reach


def _components(mask: np.ndarray) -> list[np.ndarray]:
    h, w = mask.shape
    seen = np.zeros((h, w), dtype=bool)
    out: list[np.ndarray] = []
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
                for nr, nc in _neighbors4(rr, cc):
                    if _in_bounds((h, w), nr, nc) and mask[nr, nc] and not seen[nr, nc]:
                        seen[nr, nc] = True
                        q.append((nr, nc))
            out.append(np.array(comp, dtype=int))
    return out


def _ray_cells(grid: np.ndarray, marker: tuple[int, int], step: tuple[int, int]) -> tuple[list[tuple[int, int]], str]:
    dr, dc = step
    r, c = marker
    rr, cc = r + 2 * dr, c + 2 * dc
    cells: list[tuple[int, int]] = []
    while _in_bounds(grid.shape, rr, cc) and int(grid[rr, cc]) == BLACK:
        cells.append((rr, cc))
        rr += dr
        cc += dc
    reason = "edge" if not _in_bounds(grid.shape, rr, cc) else "block"
    return cells, reason


def _oriented_pair_for_color(grid: np.ndarray, color: int) -> tuple[tuple[int, int], tuple[int, int], tuple[int, int]]:
    coords = [tuple(map(int, rc)) for rc in np.argwhere(grid == color)]
    assert len(coords) == 2, f"color {color} must occupy exactly two cells"
    a, b = coords
    assert abs(a[0] - b[0]) == 1 and abs(a[1] - b[1]) == 1, f"color {color} cells must be diagonal neighbors"

    candidates = []
    for marker, indicator in ((a, b), (b, a)):
        dr = indicator[0] - marker[0]
        dc = indicator[1] - marker[1]
        back = (marker[0] - dr, marker[1] - dc)
        front = (indicator[0] + dr, indicator[1] + dc)
        if not _in_bounds(grid.shape, back[0], back[1]):
            continue
        if int(grid[back]) == BLACK:
            continue
        if not _in_bounds(grid.shape, front[0], front[1]):
            continue
        if int(grid[front]) != BLACK:
            continue
        candidates.append((marker, indicator, (dr, dc)))

    assert len(candidates) == 1, f"color {color} orientation must be unique"
    return candidates[0]


def main(grid: np.ndarray) -> np.ndarray:
    """
    Fire only the diagonally indicated rays whose marker pixels lie in the corner-reachable region.

    Input structure:
    - The background is black (0), and blue (1) wall segments partition the grid into regions.
    - A single magenta pixel (6) sits on one corner and acts as the flood-fill start.
    - Each used marker color from {2,3,4} appears as exactly one diagonal pair of same-colored pixels inside one region.

    Rule:
    - Flood-fill 4-connected black cells starting from the corner marker position; blue walls and non-black cells block reachability.
    - For each colored diagonal pair, treat the uniquely oriented cell as the marker and the other as its direction indicator; the marker is active iff at least one of its 4-neighbors is a reachable black cell.
    - Every active marker extends a ray beyond its indicator in the same diagonal direction, painting only black cells until the ray would leave the grid or hit a non-black cell; inactive pairs are erased to black.
    - Finally erase the corner marker, while keeping active markers and their direction indicators.

    Key insight:
    The wall layout determines both which region is reachable from the marked corner and which diagonal pair is allowed to emit a ray, so reachability acts as the gate that decides whether a direction cue matters or gets erased.
    """
    g = np.asarray(grid, dtype=int)
    assert g.ndim == 2, "expected a 2D grid"

    h, w = g.shape
    corners = [(0, 0), (0, w - 1), (h - 1, 0), (h - 1, w - 1)]
    start = None
    for rc in corners:
        if int(g[rc]) == START:
            start = rc
            break
    assert start is not None, "expected a magenta corner marker"

    reach = _flood_reachable(g, start)
    out = g.copy()

    for color in MARKER_COLORS:
        if not np.any(g == color):
            continue
        marker, indicator, step = _oriented_pair_for_color(g, color)
        active = any(_in_bounds(g.shape, nr, nc) and reach[nr, nc] for nr, nc in _neighbors4(*marker))
        if not active:
            out[marker] = BLACK
            out[indicator] = BLACK
            continue
        for rr, cc in _ray_cells(g, marker, step)[0]:
            out[rr, cc] = color

    out[start] = BLACK
    return out


def _canonical_components(grid: np.ndarray, start: tuple[int, int]) -> tuple[np.ndarray, list[np.ndarray]]:
    reach = _flood_reachable(grid, start)
    comps = _components((grid == BLACK) & ~reach)
    return reach, comps


def _candidate_pairs(
    grid: np.ndarray,
    region_mask: np.ndarray,
    *,
    want_stop: str | None = None,
) -> list[dict]:
    coords = [tuple(map(int, rc)) for rc in np.argwhere(region_mask)]
    candidates: list[dict] = []
    for marker in coords:
        if not any(_in_bounds(grid.shape, nr, nc) and region_mask[nr, nc] for nr, nc in _neighbors4(*marker)):
            continue
        for step in DIAGONALS:
            dr, dc = step
            indicator = (marker[0] + dr, marker[1] + dc)
            back = (marker[0] - dr, marker[1] - dc)
            if not _in_bounds(grid.shape, indicator[0], indicator[1]) or not region_mask[indicator]:
                continue
            if not _in_bounds(grid.shape, back[0], back[1]) or int(grid[back]) == BLACK:
                continue
            path, stop_reason = _ray_cells(grid, marker, step)
            if len(path) < 1:
                continue
            if want_stop is not None and stop_reason != want_stop:
                continue
            candidates.append(
                {
                    "marker": marker,
                    "indicator": indicator,
                    "step": step,
                    "path": path,
                    "stop_reason": stop_reason,
                }
            )
    return candidates


def _place_pair(grid: np.ndarray, color: int, info: dict) -> None:
    grid[info["marker"]] = color
    grid[info["indicator"]] = color


def _make_base_layout(rng: random.Random) -> tuple[np.ndarray, tuple[int, int], int, int]:
    h = rng.randint(13, 18)
    w = rng.randint(13, 18)
    grid = np.zeros((h, w), dtype=int)
    main_c = rng.randint(4, w - 5)
    grid[:, main_c] = WALL

    split_r = -1
    if rng.random() < 0.55:
        split_r = rng.randint(4, h - 5)
        grid[split_r, main_c:] = WALL

    start = (h - 1, 0)
    grid[start] = START
    return grid, start, main_c, split_r


def _rotate(grid: np.ndarray, k: int) -> np.ndarray:
    return np.rot90(grid, k=k)


def _expected_from_plan(inp: np.ndarray, placements: list[dict], active_index: int, start: tuple[int, int]) -> np.ndarray:
    out = inp.copy()
    for idx, info in enumerate(placements):
        color = info["color"]
        if idx != active_index:
            out[info["marker"]] = BLACK
            out[info["indicator"]] = BLACK
            continue
        for rr, cc in info["path"]:
            out[rr, cc] = color
    out[start] = BLACK
    return out


def _gen_attempt() -> tuple[np.ndarray, np.ndarray] | None:
    rng = random.Random()
    grid, start, _main_c, _split_r = _make_base_layout(rng)
    reach, unreachable = _canonical_components(grid, start)
    if len(unreachable) not in (1, 2):
        return None

    used_colors = rng.sample(list(MARKER_COLORS), 1 + len(unreachable))
    placements: list[dict] = []

    active_pref = "wall" if rng.random() < 0.5 else "edge"
    active_candidates = _candidate_pairs(grid, reach, want_stop=active_pref)
    if not active_candidates:
        active_candidates = _candidate_pairs(grid, reach)
    if not active_candidates:
        return None
    active = rng.choice(active_candidates)
    _place_pair(grid, used_colors[0], active)
    placements.append({**active, "color": used_colors[0]})

    for color, comp in zip(used_colors[1:], unreachable):
        region = np.zeros_like(grid, dtype=bool)
        region[comp[:, 0], comp[:, 1]] = True
        region &= (grid == BLACK)
        want = rng.choice(["wall", "edge", None])
        candidates = _candidate_pairs(grid, region, want_stop=want)
        if not candidates:
            candidates = _candidate_pairs(grid, region)
        if not candidates:
            return None
        info = rng.choice(candidates)
        _place_pair(grid, color, info)
        placements.append({**info, "color": color})

    expected = _expected_from_plan(grid, placements, 0, start)

    solved = main(grid)
    if not np.array_equal(solved, expected):
        return None

    remaining_colors = {int(v) for v in np.unique(expected)} & set(MARKER_COLORS)
    if remaining_colors != {used_colors[0]}:
        return None
    if sum(np.count_nonzero(expected == color) for color in used_colors[1:]) != 0:
        return None
    if len(placements[0]["path"]) < 1:
        return None

    k = rng.randrange(4)
    return _rotate(grid, k), _rotate(expected, k)


def generate_input() -> np.ndarray:
    for _ in range(GEN_ATTEMPTS):
        built = _gen_attempt()
        if built is None:
            continue
        inp, out = built
        try:
            solved = main(inp)
        except Exception:
            continue
        if np.array_equal(solved, out):
            return inp
    raise RuntimeError("failed to generate a valid input for reachability-gated diagonal rays")
