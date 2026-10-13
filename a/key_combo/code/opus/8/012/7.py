from __future__ import annotations

import random
import sys
from importlib.machinery import SourceFileLoader
from pathlib import Path

import numpy as np

_ARC_UTILS_PATH = Path("/home/lixinyu/snap/new/solvers/arc_utils.py")
if str(_ARC_UTILS_PATH.parent) not in sys.path:
    sys.path.append(str(_ARC_UTILS_PATH.parent))
try:
    from arc_utils import Color, bbox, find_components
except ModuleNotFoundError:
    _arc_utils = SourceFileLoader("_arc_utils_shared", str(_ARC_UTILS_PATH)).load_module()
    Color = _arc_utils.Color
    bbox = _arc_utils.bbox
    find_components = _arc_utils.find_components


PATTERN_COLORS = (Color.BLUE, Color.RED, Color.YELLOW)
CARDINALS = ((-1, 0), (0, 1), (1, 0), (0, -1))
GRID_SIZES = (11, 13, 15, 17)
MISSING_SIZES = (3, 4, 5)
EDGE_BAND = 2
GENERATION_RETRIES = 400
PAIR_TRIALS = 220
SINGLE_TRIALS = 80


def _d4_orbit(r: int, c: int, n: int) -> tuple[tuple[int, int], ...]:
    last = n - 1
    return (
        (r, c),
        (r, last - c),
        (last - r, c),
        (last - r, last - c),
        (c, r),
        (c, last - r),
        (last - c, r),
        (last - c, last - r),
    )


def _clockwise(dr: int, dc: int) -> tuple[int, int]:
    return dc, -dr


def _edge_distance(r: int, c: int, n: int) -> int:
    last = n - 1
    return min(r, c, last - r, last - c)


def _manhattan(a: tuple[int, int], b: tuple[int, int]) -> int:
    return abs(a[0] - b[0]) + abs(a[1] - b[1])


def _build_pattern(n: int) -> np.ndarray:
    center = n // 2
    palette = list(PATTERN_COLORS)
    random.shuffle(palette)

    for _ in range(60):
        mode = random.randrange(4)
        params = {
            "a": random.randint(0, 2),
            "b": random.randint(1, 3),
            "c": random.randint(0, 2),
            "d": random.randint(0, 2),
            "e": random.randint(0, 2),
            "shift": random.randint(0, 2),
            "band_u": random.randint(1, max(1, center // 2)),
            "band_v": random.randint(1, max(1, center // 2)),
        }
        grid = np.zeros((n, n), dtype=int)
        for r in range(n):
            for c in range(n):
                dr = abs(r - center)
                dc = abs(c - center)
                u = min(dr, dc)
                v = max(dr, dc)
                if mode == 0:
                    idx = (
                        params["a"] * u
                        + params["b"] * v
                        + params["c"] * ((u + v) & 1)
                        + params["d"] * (u == 0)
                        + params["e"] * (u == v)
                        + params["shift"]
                    ) % 3
                elif mode == 1:
                    idx = (
                        params["a"] * (u % 3)
                        + params["b"] * (v % 3)
                        + params["c"] * ((v - u) % 3)
                        + params["d"] * ((u + v) & 1)
                        + params["shift"]
                    ) % 3
                elif mode == 2:
                    idx = (
                        params["a"] * (u // params["band_u"])
                        + params["b"] * (v // params["band_v"])
                        + params["c"] * ((v - u) & 1)
                        + params["d"] * (u == 0)
                        + params["shift"]
                    ) % 3
                else:
                    idx = (
                        params["a"] * (u % 2)
                        + params["b"] * (v % 2)
                        + params["c"] * ((u + v) % 3)
                        + params["d"] * (u == v)
                        + params["e"] * (u == 0)
                        + params["shift"]
                    ) % 3
                grid[r, c] = palette[idx]
        if set(map(int, np.unique(grid))) == set(PATTERN_COLORS):
            return grid
    raise RuntimeError("failed to build a 3-color D4 pattern")


def _choose_missing_bbox(n: int) -> tuple[int, int, int, int]:
    center = n // 2
    h = random.choice(MISSING_SIZES)
    w = random.choice(MISSING_SIZES)
    quadrant = random.choice(("tl", "tr", "bl", "br"))

    if quadrant == "tl":
        r0 = random.randint(0, center - h)
        c0 = random.randint(0, center - w)
    elif quadrant == "tr":
        r0 = random.randint(0, center - h)
        c0 = random.randint(center + 1, n - w)
    elif quadrant == "bl":
        r0 = random.randint(center + 1, n - h)
        c0 = random.randint(0, center - w)
    else:
        r0 = random.randint(center + 1, n - h)
        c0 = random.randint(center + 1, n - w)
    return r0, c0, r0 + h - 1, c0 + w - 1


def _cells_in_bbox(box: tuple[int, int, int, int]) -> list[tuple[int, int]]:
    r0, c0, r1, c1 = box
    return [(r, c) for r in range(r0, r1 + 1) for c in range(c0, c1 + 1)]


def _enumerate_candidates(
    n: int,
    missing_mask: np.ndarray,
) -> list[dict]:
    missing_area = int(missing_mask.sum())
    candidates = []
    for pr in range(n):
        for pc in range(n):
            if missing_mask[pr, pc] or _edge_distance(pr, pc, n) > EDGE_BAND:
                continue
            for dr, dc in CARDINALS:
                for length in (2, 3, 4):
                    source = []
                    valid = True
                    for k in range(1, length + 1):
                        sr = pr + k * dr
                        sc = pc + k * dc
                        if not (0 <= sr < n and 0 <= sc < n):
                            valid = False
                            break
                        if missing_mask[sr, sc] or _edge_distance(sr, sc, n) > EDGE_BAND:
                            valid = False
                            break
                        source.append((sr, sc))
                    if not valid:
                        continue

                    rot_dr, rot_dc = _clockwise(dr, dc)
                    rotated = []
                    inside = []
                    for k in range(1, length + 1):
                        rr = pr + k * rot_dr
                        rc = pc + k * rot_dc
                        if not (0 <= rr < n and 0 <= rc < n):
                            valid = False
                            break
                        rotated.append((rr, rc))
                        if missing_mask[rr, rc]:
                            inside.append((rr, rc))
                    if not valid or not inside or len(inside) == missing_area:
                        continue

                    candidates.append(
                        {
                            "pivot": (pr, pc),
                            "dir": (dr, dc),
                            "length": length,
                            "source": tuple(source),
                            "rotated": tuple(rotated),
                            "inside": tuple(inside),
                        }
                    )
    random.shuffle(candidates)
    return candidates


def _compatible_segments(segments: list[dict], missing_area: int) -> bool:
    source_cells = set()
    pivots = []
    covered = set()
    for seg in segments:
        pivot = seg["pivot"]
        if pivot in source_cells:
            return False
        if pivot in pivots:
            return False
        pivots.append(pivot)
        for cell in seg["source"]:
            if cell in source_cells or cell == pivot:
                return False
            source_cells.add(cell)
        for cell in seg["inside"]:
            if cell in covered:
                return False
            covered.add(cell)

    if not covered or len(covered) == missing_area:
        return False

    for i, seg_a in enumerate(segments):
        own_first = (seg_a["pivot"][0] + seg_a["dir"][0], seg_a["pivot"][1] + seg_a["dir"][1])
        for cell in source_cells:
            if cell != own_first and _manhattan(seg_a["pivot"], cell) == 1:
                return False
        for j in range(i + 1, len(segments)):
            seg_b = segments[j]
            for a in seg_a["source"]:
                for b in seg_b["source"]:
                    if _manhattan(a, b) == 1:
                        return False
    return True


def _render_input(
    full: np.ndarray,
    missing_bbox: tuple[int, int, int, int],
    segments: list[dict],
) -> np.ndarray:
    inp = full.copy()
    r0, c0, r1, c1 = missing_bbox
    inp[r0 : r1 + 1, c0 : c1 + 1] = Color.BLACK
    for seg in segments:
        pr, pc = seg["pivot"]
        inp[pr, pc] = Color.GREY
        for r, c in seg["source"]:
            inp[r, c] = Color.GREEN
    return inp


def _render_expected(
    full: np.ndarray,
    missing_bbox: tuple[int, int, int, int],
    segments: list[dict],
) -> np.ndarray:
    out = full.copy()
    for seg in segments:
        pr, pc = seg["pivot"]
        out[pr, pc] = Color.GREY
        for r, c in seg["source"]:
            out[r, c] = Color.ORANGE
        for r, c in seg["inside"]:
            out[r, c] = Color.GREEN
    return out


def _visible_orbit_values(
    full: np.ndarray,
    missing_mask: np.ndarray,
    blocked: set[tuple[int, int]],
    cell: tuple[int, int],
) -> set[int]:
    n = full.shape[0]
    values = set()
    for rr, cc in _d4_orbit(cell[0], cell[1], n):
        if missing_mask[rr, cc] or (rr, cc) in blocked:
            continue
        values.add(int(full[rr, cc]))
    return values


def _validate_case(
    inp: np.ndarray,
    full: np.ndarray,
    missing_bbox: tuple[int, int, int, int],
    segments: list[dict],
) -> np.ndarray | None:
    expected = _render_expected(full, missing_bbox, segments)
    missing_mask = np.zeros_like(inp, dtype=bool)
    r0, c0, r1, c1 = missing_bbox
    missing_mask[r0 : r1 + 1, c0 : c1 + 1] = True

    blocked = {seg["pivot"] for seg in segments}
    for seg in segments:
        blocked.update(seg["source"])
    covered = {cell for seg in segments for cell in seg["inside"]}
    missing_cells = set(_cells_in_bbox(missing_bbox))
    remaining = missing_cells - covered

    if not covered or not remaining:
        return None

    symmetry_only = full.copy()
    for seg in segments:
        pr, pc = seg["pivot"]
        symmetry_only[pr, pc] = Color.GREY
        for r, c in seg["source"]:
            symmetry_only[r, c] = Color.ORANGE
    if np.array_equal(symmetry_only, expected):
        return None

    for cell in remaining:
        values = _visible_orbit_values(full, missing_mask, blocked, cell)
        if values != {int(full[cell])}:
            return None

    rotation_only = inp.copy()
    for seg in segments:
        for r, c in seg["source"]:
            rotation_only[r, c] = Color.ORANGE
        for r, c in seg["inside"]:
            rotation_only[r, c] = Color.GREEN
    if not np.any(rotation_only[r0 : r1 + 1, c0 : c1 + 1] == Color.BLACK):
        return None

    solved = main(inp)
    if not np.array_equal(solved, expected):
        return None
    return expected


def _pick_segments(
    full: np.ndarray,
    missing_bbox: tuple[int, int, int, int],
) -> tuple[np.ndarray, np.ndarray] | None:
    n = full.shape[0]
    missing_mask = np.zeros((n, n), dtype=bool)
    r0, c0, r1, c1 = missing_bbox
    missing_mask[r0 : r1 + 1, c0 : c1 + 1] = True
    missing_area = int(missing_mask.sum())

    candidates = _enumerate_candidates(n, missing_mask)
    if not candidates:
        return None

    prefer_two = random.random() < 0.55
    order = (2, 1) if prefer_two else (1, 2)
    for wanted in order:
        if wanted == 1:
            trials = min(SINGLE_TRIALS, len(candidates))
            for seg in random.sample(candidates, trials):
                segments = [seg]
                if not _compatible_segments(segments, missing_area):
                    continue
                inp = _render_input(full, missing_bbox, segments)
                expected = _validate_case(inp, full, missing_bbox, segments)
                if expected is not None:
                    return inp, expected
        else:
            if len(candidates) < 2:
                continue
            for _ in range(PAIR_TRIALS):
                seg_a, seg_b = random.sample(candidates, 2)
                segments = [seg_a, seg_b]
                if not _compatible_segments(segments, missing_area):
                    continue
                inp = _render_input(full, missing_bbox, segments)
                expected = _validate_case(inp, full, missing_bbox, segments)
                if expected is not None:
                    return inp, expected
    return None


def _extract_segments(grid: np.ndarray) -> list[dict]:
    components = find_components(
        grid,
        bg=Color.BLACK,
        connectivity=4,
        monochromatic=True,
        colors=[Color.GREEN],
    )
    assert 1 <= len(components) <= 2
    pivots = {tuple(map(int, coord)) for coord in np.argwhere(grid == Color.GREY)}
    assert len(pivots) == len(components)

    segments = []
    h, w = grid.shape
    for comp in components:
        coords = {tuple(map(int, coord)) for coord in comp["coords"]}
        adjacent_pivots = set()
        for r, c in coords:
            for nr, nc in ((r - 1, c), (r + 1, c), (r, c - 1), (r, c + 1)):
                if 0 <= nr < h and 0 <= nc < w and int(grid[nr, nc]) == Color.GREY:
                    adjacent_pivots.add((nr, nc))
        assert len(adjacent_pivots) == 1
        pivot = adjacent_pivots.pop()

        touching = [cell for cell in coords if _manhattan(cell, pivot) == 1]
        assert len(touching) == 1
        dr = touching[0][0] - pivot[0]
        dc = touching[0][1] - pivot[1]
        length = len(coords)
        assert 2 <= length <= 4
        expected_source = {
            (pivot[0] + k * dr, pivot[1] + k * dc)
            for k in range(1, length + 1)
        }
        assert coords == expected_source

        rot_dr, rot_dc = _clockwise(dr, dc)
        rotated = tuple(
            (pivot[0] + k * rot_dr, pivot[1] + k * rot_dc)
            for k in range(1, length + 1)
        )
        segments.append(
            {
                "pivot": pivot,
                "dir": (dr, dc),
                "length": length,
                "source": tuple(sorted(coords)),
                "rotated": rotated,
            }
        )
    segments.sort(key=lambda seg: (seg["pivot"], seg["source"]))
    return segments


def main(grid: np.ndarray) -> np.ndarray:
    """
    Fill a missing rectangle by rotating edge bars into it and completing the rest from D4-symmetric copies.

    Input structure:
    - A square odd-sized grid is fully covered by a blue/red/yellow D4-symmetric pattern except for one solid black rectangular missing region.
    - One or two green straight bars of length 2 to 4 lie near the outer border, each attached at one endpoint to a grey pivot cell.
    - The missing rectangle stays fully inside one quadrant and does not cross the center row or center column.

    Rule:
    - For each green bar, find the adjacent grey pivot, infer the bar direction away from the pivot, and recolor the original green cells orange.
    - Rotate that bar 90 degrees clockwise around the pivot and paint only the rotated cells that land inside the black missing rectangle as green.
    - For every still-black cell in the missing rectangle, inspect its D4 orbit about the grid center and copy the unique visible blue/red/yellow value from that orbit.

    Key insight:
    - The hole is repaired by two non-interchangeable mechanisms: rotated bars claim specific cells with green priority, and D4 symmetry restores only the untouched remainder.
    """
    g = np.asarray(grid, dtype=int)
    assert g.ndim == 2
    n, m = g.shape
    assert n == m and n % 2 == 1

    missing = g == Color.BLACK
    missing_bbox = bbox(missing)
    assert missing_bbox is not None
    r0, c0, r1, c1 = missing_bbox
    assert int(missing.sum()) == (r1 - r0 + 1) * (c1 - c0 + 1)
    center = n // 2
    assert r1 < center or r0 > center
    assert c1 < center or c0 > center

    out = g.copy()
    segments = _extract_segments(g)
    for seg in segments:
        for r, c in seg["source"]:
            out[r, c] = Color.ORANGE

    for seg in segments:
        for r, c in seg["rotated"]:
            if missing[r, c]:
                out[r, c] = Color.GREEN

    for r in range(r0, r1 + 1):
        for c in range(c0, c1 + 1):
            if int(out[r, c]) != Color.BLACK:
                continue
            values = {
                int(out[rr, cc])
                for rr, cc in _d4_orbit(r, c, n)
                if int(out[rr, cc]) in PATTERN_COLORS
            }
            assert len(values) == 1
            out[r, c] = values.pop()

    assert not np.any(out[r0 : r1 + 1, c0 : c1 + 1] == Color.BLACK)
    return out


def generate_input() -> np.ndarray:
    for _ in range(GENERATION_RETRIES):
        n = random.choice(GRID_SIZES)
        full = _build_pattern(n)
        missing_bbox = _choose_missing_bbox(n)
        picked = _pick_segments(full, missing_bbox)
        if picked is None:
            continue
        inp, _expected = picked
        return inp
    raise RuntimeError("failed to generate a valid input")
