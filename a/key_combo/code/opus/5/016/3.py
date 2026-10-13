from __future__ import annotations

from collections import Counter, defaultdict
from importlib.machinery import SourceFileLoader
import random

import numpy as np


_ARC_UTILS = SourceFileLoader(
    "_arc_utils_shared_for_opus_5_016_3",
    "/home/lixinyu/snap/new/solvers/arc_utils.py",
).load_module()
Color = _ARC_UTILS.Color


_INTERIOR_OFFSETS = (
    (1, 2),
    (1, 3),
    (2, 1),
    (2, 2),
    (2, 3),
    (2, 4),
    (3, 1),
    (3, 2),
    (3, 3),
    (3, 4),
    (4, 2),
    (4, 3),
)
_INTERIOR = set(_INTERIOR_OFFSETS)

_GRID_SIZE = 23
_LATTICE = Color.TEAL
_EMPTY = Color.BLACK
_MARKER_COLORS = (
    Color.BLUE,
    Color.RED,
    Color.GREEN,
    Color.YELLOW,
    Color.PINK,
    Color.ORANGE,
    Color.MAROON,
)
_GEN_RETRIES = 300


def _chebyshev(a: tuple[int, int], b: tuple[int, int]) -> int:
    return max(abs(int(a[0]) - int(b[0])), abs(int(a[1]) - int(b[1])))


def _periodic_value(r: int, c: int, lattice: int, phase_r: int, phase_c: int) -> int:
    dr = (int(r) - int(phase_r)) % 5
    dc = (int(c) - int(phase_c)) % 5
    if dr == 0 and dc == 0:
        return _EMPTY
    if (dr, dc) in _INTERIOR:
        return _EMPTY
    return int(lattice)


def _infer_lattice_color(grid: np.ndarray) -> int:
    vals, counts = np.unique(grid, return_counts=True)
    non_black = [(int(v), int(c)) for v, c in zip(vals, counts) if int(v) != _EMPTY]
    assert non_black, "expected a non-black lattice color"
    lattice, _count = max(non_black, key=lambda item: item[1])
    return int(lattice)


def _matches_full_tile(grid: np.ndarray, r0: int, c0: int, lattice: int) -> bool:
    H, W = grid.shape
    if r0 < 0 or c0 < 0 or r0 + 4 >= H or c0 + 4 >= W:
        return False
    if int(grid[r0, c0]) == int(lattice):
        return False
    for dr in range(5):
        for dc in range(5):
            if (dr, dc) == (0, 0):
                continue
            expected = _EMPTY if (dr, dc) in _INTERIOR else int(lattice)
            if int(grid[r0 + dr, c0 + dc]) != expected:
                return False
    return True


def _extract_structure(grid: np.ndarray) -> dict:
    g = np.asarray(grid, dtype=int)
    H, W = g.shape
    assert H == W and H % 2 == 1, "task expects an odd square grid"
    center = (H // 2, W // 2)
    lattice = _infer_lattice_color(g)

    matches = [(r, c) for r in range(H - 4) for c in range(W - 4) if _matches_full_tile(g, r, c, lattice)]
    assert matches, "expected at least one complete 5x5 lattice tile"

    phase_r = Counter(r % 5 for r, _c in matches).most_common(1)[0][0]
    phase_c = Counter(c % 5 for _r, c in matches).most_common(1)[0][0]

    full_anchors: list[tuple[int, int]] = []
    for r in range(int(phase_r), H - 4, 5):
        for c in range(int(phase_c), W - 4, 5):
            if _matches_full_tile(g, r, c, lattice):
                full_anchors.append((int(r), int(c)))

    assert full_anchors, "expected fillable chambers"

    chambers = []
    chambers_by_dist: dict[int, list[dict]] = defaultdict(list)
    full_anchor_set = set(full_anchors)
    for r0, c0 in full_anchors:
        coords = [(int(r0 + dr), int(c0 + dc)) for dr, dc in _INTERIOR_OFFSETS]
        assert all(int(g[r, c]) == _EMPTY for r, c in coords), "input chambers must be empty"
        center_pixel = (int(r0 + 2), int(c0 + 2))
        dist = _chebyshev(center_pixel, center)
        chamber = {
            "anchor": (int(r0), int(c0)),
            "coords": coords,
            "center": center_pixel,
            "dist": int(dist),
        }
        chambers.append(chamber)
        chambers_by_dist[int(dist)].append(chamber)

    markers = []
    gap_positions = []
    for r in range(int(phase_r), H, 5):
        for c in range(int(phase_c), W, 5):
            gap_positions.append((int(r), int(c)))
            value = int(g[r, c])
            if value not in (_EMPTY, int(lattice)):
                markers.append(
                    {
                        "pos": (int(r), int(c)),
                        "color": value,
                        "dist": _chebyshev((int(r), int(c)), center),
                        "has_local_tile": (int(r), int(c)) in full_anchor_set,
                    }
                )

    assert markers, "expected at least one marker"
    return {
        "grid": g,
        "center": center,
        "lattice": int(lattice),
        "phase": (int(phase_r), int(phase_c)),
        "gap_positions": gap_positions,
        "markers": markers,
        "chambers": chambers,
        "chambers_by_dist": dict(chambers_by_dist),
    }


def _solve(grid: np.ndarray) -> np.ndarray:
    info = _extract_structure(grid)
    out = np.asarray(grid, dtype=int).copy()

    marker_dists = [int(marker["dist"]) for marker in info["markers"]]
    assert len(marker_dists) == len(set(marker_dists)), "marker distances must be unique"

    for marker in info["markers"]:
        r, c = marker["pos"]
        out[int(r), int(c)] = _EMPTY

    for marker in info["markers"]:
        targets = info["chambers_by_dist"].get(int(marker["dist"]), [])
        assert targets, "every marker distance must match at least one cavity"
        for chamber in targets:
            for r, c in chamber["coords"]:
                out[int(r), int(c)] = int(marker["color"])

    return out


def main(grid: np.ndarray) -> np.ndarray:
    """
    Recolor lattice cavities whose center-aligned Chebyshev distance matches scattered marker gaps.

    Input structure:
    - The grid is an odd square containing a repeated teal 5x5 lattice pattern with black corner gaps.
    - Each complete tile encloses one 12-pixel black cavity in the 17b866bd lattice shape.
    - Three or more colored marker pixels sit on some lattice corner gaps; all other gaps remain black.

    Rule:
    - Use the grid center as the reference point for every distance measurement.
    - For each marker gap, compute its Chebyshev distance to the grid center.
    - For each complete cavity, use the lattice tile's center-aligned cavity representative pixel as that cavity's center, then compute the same Chebyshev distance.
    - Fill every cavity whose distance equals the marker's distance with the marker color, and erase all original markers back to black.

    Key insight:
    - The marker does not control its adjacent cavity. It remotely colors every cavity on the same Chebyshev-distance ring around the grid center.
    """
    return _solve(grid)


def _base_lattice(size: int = _GRID_SIZE, *, phase_r: int = 0, phase_c: int = 0, lattice: int = _LATTICE) -> np.ndarray:
    grid = np.zeros((size, size), dtype=int)
    for r in range(size):
        for c in range(size):
            grid[r, c] = _periodic_value(r, c, int(lattice), int(phase_r), int(phase_c))
    return grid


def _generator_blueprint(size: int = _GRID_SIZE) -> dict:
    center = (size // 2, size // 2)
    full_anchors = [(r, c) for r in range(0, size - 4, 5) for c in range(0, size - 4, 5)]
    chambers_by_dist: dict[int, list[list[tuple[int, int]]]] = defaultdict(list)
    for r0, c0 in full_anchors:
        coords = [(int(r0 + dr), int(c0 + dc)) for dr, dc in _INTERIOR_OFFSETS]
        dist = _chebyshev((int(r0 + 2), int(c0 + 2)), center)
        chambers_by_dist[int(dist)].append(coords)

    gap_positions_by_dist: dict[int, list[tuple[int, int]]] = defaultdict(list)
    edge_gap_positions_by_dist: dict[int, list[tuple[int, int]]] = defaultdict(list)
    for r in range(0, size, 5):
        for c in range(0, size, 5):
            dist = _chebyshev((r, c), center)
            if dist not in chambers_by_dist:
                continue
            gap_positions_by_dist[int(dist)].append((int(r), int(c)))
            if r > size - 5 or c > size - 5:
                edge_gap_positions_by_dist[int(dist)].append((int(r), int(c)))

    return {
        "center": center,
        "chambers_by_dist": dict(chambers_by_dist),
        "gap_positions_by_dist": dict(gap_positions_by_dist),
        "edge_gap_positions_by_dist": dict(edge_gap_positions_by_dist),
        "usable_dists": tuple(sorted(chambers_by_dist)),
    }


_BLUEPRINT = _generator_blueprint()


def _choose_marker_distances() -> list[int]:
    choices = [
        [1, 4, 6],
        [1, 4, 9],
        [1, 6, 9],
        [4, 6, 9],
        [1, 4, 6, 9],
    ]
    return list(random.choice(choices))


def _expected_from_spec(base: np.ndarray, dist_to_marker: dict[int, tuple[tuple[int, int], int]]) -> np.ndarray:
    out = base.copy()
    for dist, (_pos, color) in dist_to_marker.items():
        for chamber in _BLUEPRINT["chambers_by_dist"][int(dist)]:
            for r, c in chamber:
                out[int(r), int(c)] = int(color)
    return out


def _adjacent_baseline(grid: np.ndarray) -> np.ndarray:
    info = _extract_structure(grid)
    out = np.asarray(grid, dtype=int).copy()
    chamber_at_anchor = {tuple(chamber["anchor"]): chamber["coords"] for chamber in info["chambers"]}
    for marker in info["markers"]:
        r, c = marker["pos"]
        out[int(r), int(c)] = _EMPTY
        coords = chamber_at_anchor.get((int(r), int(c)))
        if coords is None:
            continue
        for rr, cc in coords:
            out[int(rr), int(cc)] = int(marker["color"])
    return out


def _validate_generated_case(
    grid: np.ndarray,
    expected: np.ndarray,
    dist_to_marker: dict[int, tuple[tuple[int, int], int]],
) -> bool:
    try:
        info = _extract_structure(grid)
    except Exception:
        return False

    if len(info["markers"]) != len(dist_to_marker):
        return False
    if len(info["markers"]) < 3:
        return False

    marker_dists = [int(marker["dist"]) for marker in info["markers"]]
    if len(marker_dists) != len(set(marker_dists)):
        return False

    if set(marker_dists) != set(map(int, dist_to_marker)):
        return False

    if not any(not bool(marker["has_local_tile"]) for marker in info["markers"]):
        return False

    if not any(len(info["chambers_by_dist"][dist]) > 1 for dist in marker_dists):
        return False

    if len(info["markers"]) == len(info["chambers_by_dist"]):
        return False

    try:
        solved = main(grid)
    except Exception:
        return False
    if not np.array_equal(solved, expected):
        return False

    baseline = _adjacent_baseline(grid)
    if np.array_equal(baseline, expected):
        return False

    return True


def _gen_attempt() -> np.ndarray | None:
    base = _base_lattice()
    marker_dists = _choose_marker_distances()
    marker_colors = random.sample(_MARKER_COLORS, len(marker_dists))
    random.shuffle(marker_colors)

    dist_to_marker: dict[int, tuple[tuple[int, int], int]] = {}
    for dist, color in zip(marker_dists, marker_colors):
        positions = list(_BLUEPRINT["gap_positions_by_dist"][int(dist)])
        if int(dist) == 9 and _BLUEPRINT["edge_gap_positions_by_dist"][int(dist)]:
            positions = list(_BLUEPRINT["edge_gap_positions_by_dist"][int(dist)])
        pos = random.choice(positions)
        dist_to_marker[int(dist)] = (pos, int(color))

    inp = base.copy()
    for _dist, (pos, color) in dist_to_marker.items():
        inp[int(pos[0]), int(pos[1])] = int(color)

    expected = _expected_from_spec(base, dist_to_marker)
    if _validate_generated_case(inp, expected, dist_to_marker):
        return inp
    return None


def generate_input() -> np.ndarray:
    for _ in range(_GEN_RETRIES):
        grid = _gen_attempt()
        if grid is not None:
            return grid
    raise RuntimeError(f"failed to generate a valid input after {_GEN_RETRIES} attempts")
