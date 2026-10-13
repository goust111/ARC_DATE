from __future__ import annotations

import random
from collections import deque

import numpy as np


BLACK = 0
GREEN = 3
TEAL = 8


def _in_bounds(shape: tuple[int, int], r: int, c: int) -> bool:
    h, w = shape
    return 0 <= int(r) < h and 0 <= int(c) < w


def _neighbors8(r: int, c: int):
    for dr in (-1, 0, 1):
        for dc in (-1, 0, 1):
            if dr or dc:
                yield r + dr, c + dc


def _components8(mask: np.ndarray) -> list[list[tuple[int, int]]]:
    seen = np.zeros(mask.shape, dtype=bool)
    out: list[list[tuple[int, int]]] = []
    for r0, c0 in np.argwhere(mask):
        r0 = int(r0)
        c0 = int(c0)
        if seen[r0, c0]:
            continue
        q = deque([(r0, c0)])
        seen[r0, c0] = True
        comp: list[tuple[int, int]] = []
        while q:
            r, c = q.popleft()
            comp.append((r, c))
            for nr, nc in _neighbors8(r, c):
                if _in_bounds(mask.shape, nr, nc) and mask[nr, nc] and not seen[nr, nc]:
                    seen[nr, nc] = True
                    q.append((nr, nc))
        out.append(sorted(comp))
    return out


def _line_info(coords: list[tuple[int, int]], idx: int) -> dict:
    assert len(coords) >= 2, "green objects must be diagonal fragments, not isolated pixels"
    rows = sorted(r for r, _ in coords)
    assert len(set(rows)) == len(rows), "a diagonal fragment has one pixel in each row"
    assert rows == list(range(rows[0], rows[-1] + 1)), "diagonal fragments must be contiguous"

    diffs = {c - r for r, c in coords}
    sums = {c + r for r, c in coords}
    if len(diffs) == 1:
        slope = 1
        const = next(iter(diffs))
    elif len(sums) == 1:
        slope = -1
        const = next(iter(sums))
    else:
        raise AssertionError("green objects must be straight 45-degree fragments")

    by_row = {r: c for r, c in coords}
    return {
        "idx": idx,
        "coords": tuple(coords),
        "rows": tuple(rows),
        "slope": slope,
        "const": int(const),
        "by_row": by_row,
    }


def _intersection(a: dict, b: dict) -> tuple[int, int] | None:
    m1, k1 = int(a["slope"]), int(a["const"])
    m2, k2 = int(b["slope"]), int(b["const"])
    denom = m1 - m2
    if denom == 0:
        return None
    num_r = k2 - k1
    if num_r % denom != 0:
        return None
    r = num_r // denom
    c_num = m1 * r + k1
    return int(r), int(c_num)


def _pair_wedge(g: np.ndarray, a: dict, b: dict) -> dict | None:
    if int(a["slope"]) == int(b["slope"]):
        return None
    if tuple(a["rows"]) != tuple(b["rows"]):
        return None

    row_sums = {int(a["by_row"][r]) + int(b["by_row"][r]) for r in a["rows"]}
    if len(row_sums) != 1:
        return None
    doubled_axis = row_sums.pop()
    if doubled_axis % 2 != 0:
        return None
    axis = doubled_axis // 2

    hit = _intersection(a, b)
    if hit is None:
        return None
    apex_r, apex_c = hit
    if apex_c != axis:
        return None
    if not _in_bounds(g.shape, apex_r, apex_c):
        return None
    if int(g[apex_r, apex_c]) == GREEN:
        return None

    rows = tuple(int(r) for r in a["rows"])
    if min(rows) > apex_r:
        row_dir = 1
    elif max(rows) < apex_r:
        row_dir = -1
    else:
        return None

    distances = [abs(r - apex_r) for r in rows]
    if min(distances) < 1:
        return None
    for line in (a, b):
        for r, c in line["coords"]:
            if abs(int(r) - apex_r) != abs(int(c) - axis):
                return None
    height = max(distances)
    return {
        "apex": (int(apex_r), int(axis)),
        "row_dir": int(row_dir),
        "height": int(height),
        "pair": (int(a["idx"]), int(b["idx"])),
    }


def _find_wedges(g: np.ndarray) -> list[dict]:
    comps = _components8(g == GREEN)
    infos = [_line_info(comp, i) for i, comp in enumerate(comps)]
    candidates: list[dict] = []
    for i in range(len(infos)):
        for j in range(i + 1, len(infos)):
            wedge = _pair_wedge(g, infos[i], infos[j])
            if wedge is not None:
                candidates.append(wedge)

    paired: dict[int, dict] = {}
    for wedge in candidates:
        for idx in wedge["pair"]:
            assert idx not in paired, "mirror pairing must be unique"
            paired[idx] = wedge
    assert candidates, "expected at least one mirrored pair of green fragments"
    return sorted(candidates, key=lambda w: (w["apex"], w["row_dir"], w["height"]))


def _wedge_cells(
    apex_r: int,
    apex_c: int,
    row_dir: int,
    height: int,
    shape: tuple[int, int],
) -> list[tuple[int, int]]:
    cells: list[tuple[int, int]] = []
    for d in range(int(height) + 1):
        r = int(apex_r) + int(row_dir) * d
        for c in range(int(apex_c) - d, int(apex_c) + d + 1):
            if _in_bounds(shape, r, c):
                cells.append((r, c))
    return cells


def main(grid: np.ndarray) -> np.ndarray:
    """
    Extend mirrored green diagonal fragments to invisible apexes and fill the wedges with teal.

    Input structure:
    - The grid has a black background and green 45-degree diagonal line fragments.
    - Some green fragments form exact mirror pairs across a vertical axis; other fragments have no mirrored partner.
    - For every real pair, the two diagonal lines meet at an in-bounds black apex cell that is not drawn in green.

    Rule:
    - Find each uniquely mirrored pair by matching opposite-slope diagonal fragments with the same row span and one vertical mirror axis.
    - Extend the two fragment lines toward that axis to locate their unseen intersection, then use the farthest paired green pixel to set the wedge height.
    - Paint every non-green cell inside and on the boundary of each triangular wedge teal, preserving all green fragments; unpaired fragments do not create wedges.

    Key insight:
    - The wedge tip is absent from the input, so the fill must be inferred by mirror pairing and diagonal extrapolation rather than by treating every green fragment as a boundary.
    """
    g = np.asarray(grid, dtype=int)
    assert g.ndim == 2, "expected a 2D grid"
    colors = set(map(int, np.unique(g)))
    assert GREEN in colors, "expected green diagonal fragments"
    assert colors <= {BLACK, GREEN}, "expected only black background and green fragments"

    out = g.copy()
    for wedge in _find_wedges(g):
        apex_r, apex_c = wedge["apex"]
        for r, c in _wedge_cells(apex_r, apex_c, wedge["row_dir"], wedge["height"], g.shape):
            if int(out[r, c]) != GREEN:
                out[r, c] = TEAL
    return out


# --------------- generator ---------------

GEN_ATTEMPTS = 500
HEIGHT_RANGE = (13, 24)
WIDTH_RANGE = (13, 28)
PAIR_COUNT_CHOICES = (1, 2, 2, 3)
DISTRACTOR_COUNT_RANGE = (1, 3)
WEDGE_HEIGHT_RANGE = (3, 6)
OBJECT_GAP = 1


def _padded(cells: set[tuple[int, int]], shape: tuple[int, int], gap: int = OBJECT_GAP) -> set[tuple[int, int]]:
    out: set[tuple[int, int]] = set()
    for r, c in cells:
        for rr in range(r - gap, r + gap + 1):
            for cc in range(c - gap, c + gap + 1):
                if _in_bounds(shape, rr, cc):
                    out.add((rr, cc))
    return out


def _pair_line_cells(apex_r: int, axis: int, row_dir: int, height: int, start: int) -> set[tuple[int, int]]:
    cells: set[tuple[int, int]] = set()
    for d in range(start, height + 1):
        r = apex_r + row_dir * d
        cells.add((r, axis - d))
        cells.add((r, axis + d))
    return cells


def _valid_pair_specs(shape: tuple[int, int]) -> list[tuple[int, int, int, int, int]]:
    h, w = shape
    specs: list[tuple[int, int, int, int, int]] = []
    for height in range(WEDGE_HEIGHT_RANGE[0], WEDGE_HEIGHT_RANGE[1] + 1):
        for row_dir in (-1, 1):
            for apex_r in range(h):
                far_r = apex_r + row_dir * height
                if not (1 <= far_r < h - 1 and 1 <= apex_r < h - 1):
                    continue
                for axis in range(height + 1, w - height - 1):
                    for start in range(2, height):
                        specs.append((apex_r, axis, row_dir, height, start))
    random.shuffle(specs)
    return specs


def _region_for_spec(shape: tuple[int, int], spec: tuple[int, int, int, int, int]) -> set[tuple[int, int]]:
    apex_r, axis, row_dir, height, _start = spec
    return set(_wedge_cells(apex_r, axis, row_dir, height, shape))


def _place_pairs(
    grid: np.ndarray,
    count: int,
) -> tuple[list[tuple[int, int, int, int, int]], set[tuple[int, int]]] | None:
    shape = grid.shape
    used: set[tuple[int, int]] = set()
    placed: list[tuple[int, int, int, int, int]] = []
    specs = _valid_pair_specs(shape)
    for _ in range(count):
        choices = []
        for spec in specs:
            region = _region_for_spec(shape, spec)
            if not (_padded(region, shape) & used):
                choices.append((spec, region))
        if not choices:
            return None
        spec, region = random.choice(choices)
        line_cells = _pair_line_cells(*spec)
        if any(grid[r, c] != BLACK for r, c in line_cells):
            return None
        for r, c in line_cells:
            grid[r, c] = GREEN
        used.update(_padded(region, shape))
        placed.append(spec)
    return placed, used


def _line_segment_cells(shape: tuple[int, int], r: int, c: int, slope: int, length: int) -> set[tuple[int, int]] | None:
    cells = {(r + i, c + slope * i) for i in range(length)}
    if all(_in_bounds(shape, rr, cc) for rr, cc in cells):
        return cells
    return None


def _place_distractors(
    grid: np.ndarray,
    used: set[tuple[int, int]],
    count: int,
) -> set[tuple[int, int]] | None:
    shape = grid.shape
    h, w = shape
    distractors: set[tuple[int, int]] = set()
    for _ in range(count):
        candidates: list[set[tuple[int, int]]] = []
        for length in range(2, 6):
            for slope in (-1, 1):
                for r in range(1, h - length):
                    for c in range(1, w - 1):
                        cells = _line_segment_cells(shape, r, c, slope, length)
                        if cells is None:
                            continue
                        if any(grid[rr, cc] != BLACK for rr, cc in cells):
                            continue
                        if _padded(cells, shape) & used:
                            continue
                        candidates.append(cells)
        if not candidates:
            return None
        cells = random.choice(candidates)
        for r, c in cells:
            grid[r, c] = GREEN
        used.update(_padded(cells, shape))
        distractors.update(cells)
    return distractors


def _expected_from_specs(grid: np.ndarray, specs: list[tuple[int, int, int, int, int]]) -> np.ndarray:
    out = grid.copy()
    for apex_r, axis, row_dir, height, _start in specs:
        for r, c in _wedge_cells(apex_r, axis, row_dir, height, grid.shape):
            if int(out[r, c]) != GREEN:
                out[r, c] = TEAL
    return out


def _wedge_signature(wedge: dict) -> tuple[int, int, int, int]:
    apex_r, axis = wedge["apex"]
    return int(apex_r), int(axis), int(wedge["row_dir"]), int(wedge["height"])


def _spec_signature(spec: tuple[int, int, int, int, int]) -> tuple[int, int, int, int]:
    apex_r, axis, row_dir, height, _start = spec
    return int(apex_r), int(axis), int(row_dir), int(height)


def _gen_attempt() -> np.ndarray | None:
    h = random.randint(*HEIGHT_RANGE)
    w = random.randint(*WIDTH_RANGE)
    grid = np.full((h, w), BLACK, dtype=int)

    pair_count = random.choice(PAIR_COUNT_CHOICES)
    placed = _place_pairs(grid, pair_count)
    if placed is None:
        return None
    specs, used = placed

    distractor_count = random.randint(*DISTRACTOR_COUNT_RANGE)
    distractors = _place_distractors(grid, used, distractor_count)
    if distractors is None:
        return None

    try:
        found = _find_wedges(grid)
    except AssertionError:
        return None
    if sorted(_wedge_signature(w) for w in found) != sorted(_spec_signature(s) for s in specs):
        return None

    expected = _expected_from_specs(grid, specs)
    try:
        actual = main(grid)
    except AssertionError:
        return None
    if not np.array_equal(actual, expected):
        return None

    # Semantic checks beyond closure: unseen apexes become teal, the deliberate
    # one-cell gap to each apex is filled, and all unpaired fragments stay green.
    for apex_r, axis, row_dir, _height, _start in specs:
        if int(grid[apex_r, axis]) != BLACK or int(actual[apex_r, axis]) != TEAL:
            return None
        gap_cells = {(apex_r + row_dir, axis - 1), (apex_r + row_dir, axis + 1)}
        if any(int(grid[r, c]) != BLACK or int(actual[r, c]) != TEAL for r, c in gap_cells):
            return None
    if not distractors or any(int(actual[r, c]) != GREEN for r, c in distractors):
        return None
    if int(np.count_nonzero(actual == TEAL)) < 8 * len(specs):
        return None

    return grid


def generate_input() -> np.ndarray:
    for _ in range(GEN_ATTEMPTS):
        grid = _gen_attempt()
        if grid is not None:
            return grid
    raise RuntimeError("failed to generate a valid invisible-apex mirror wedge input")
