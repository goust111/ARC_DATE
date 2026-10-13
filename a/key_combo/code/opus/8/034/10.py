from __future__ import annotations

from collections import Counter
import random

import numpy as np


BLACK = 0
BLUE = 1
RED = 2
GREEN = 3

QUADRANTS = {
    "tl": (-1, -1),
    "tr": (-1, 1),
    "bl": (1, -1),
    "br": (1, 1),
}

GRID_SIZE_MIN = 11
GRID_SIZE_MAX = 16
AXIS_MARGIN = 3
PATTERN_BOX_MAX = 5
MIN_PATTERN_PIXELS = 5
MAX_GENERATION_ATTEMPTS = 500


def _find_axes(grid: np.ndarray) -> tuple[int, int]:
    red_rows = np.flatnonzero(np.all(grid == RED, axis=1))
    red_cols = np.flatnonzero(np.all(grid == RED, axis=0))
    assert red_rows.size == 1, "expected one full red horizontal line"
    assert red_cols.size == 1, "expected one full red vertical line"
    return int(red_rows[0]), int(red_cols[0])


def _quadrant_bounds(
    h: int,
    w: int,
    hr: int,
    vc: int,
) -> dict[str, tuple[tuple[int, int], tuple[int, int]]]:
    return {
        "tl": ((0, hr), (0, vc)),
        "tr": ((0, hr), (vc + 1, w)),
        "bl": ((hr + 1, h), (0, vc)),
        "br": ((hr + 1, h), (vc + 1, w)),
    }


def _quadrant_counts(grid: np.ndarray, hr: int, vc: int) -> dict[str, int]:
    h, w = grid.shape
    counts: dict[str, int] = {}
    for name, ((r0, r1), (c0, c1)) in _quadrant_bounds(h, w, hr, vc).items():
        patch = grid[r0:r1, c0:c1]
        counts[name] = int(np.count_nonzero((patch != BLACK) & (patch != RED)))
    return counts


def _pick_source_quadrant(grid: np.ndarray, hr: int, vc: int) -> str:
    counts = _quadrant_counts(grid, hr, vc)
    ordered = sorted(counts.items(), key=lambda item: item[1], reverse=True)
    assert ordered[0][1] > 0, "expected one non-empty pattern quadrant"
    assert ordered[0][1] > ordered[1][1], "pattern quadrant must be uniquely dominant"
    return ordered[0][0]


def _orbit_positions(
    hr: int,
    vc: int,
    dr: int,
    dc: int,
    shape: tuple[int, int],
) -> list[tuple[int, int]]:
    h, w = shape
    pts = []
    for row_sign in (-1, 1):
        for col_sign in (-1, 1):
            r = hr + row_sign * dr
            c = vc + col_sign * dc
            if 0 <= r < h and 0 <= c < w:
                pts.append((int(r), int(c)))
    return sorted(set(pts))


def _pre_vote_expand(grid: np.ndarray, hr: int, vc: int, source_name: str) -> np.ndarray:
    work = np.asarray(grid, dtype=int).copy()
    h, w = work.shape
    (row_sign, col_sign) = QUADRANTS[source_name]
    ((r0, r1), (c0, c1)) = _quadrant_bounds(h, w, hr, vc)[source_name]

    for r in range(r0, r1):
        for c in range(c0, c1):
            color = int(work[r, c])
            if color in (BLACK, RED):
                continue
            dr = abs(r - hr)
            dc = abs(c - vc)
            assert row_sign == -1 if r < hr else row_sign == 1
            assert col_sign == -1 if c < vc else col_sign == 1
            for rr, cc in _orbit_positions(hr, vc, dr, dc, (h, w)):
                if work[rr, cc] == BLACK:
                    work[rr, cc] = color
    return work


def _vote_orbits(grid: np.ndarray, hr: int, vc: int) -> np.ndarray:
    g = np.asarray(grid, dtype=int)
    h, w = g.shape
    out = g.copy()
    seen: set[tuple[int, int]] = set()

    for r in range(h):
        for c in range(w):
            if r == hr or c == vc:
                continue
            key = (abs(r - hr), abs(c - vc))
            if key in seen:
                continue
            seen.add(key)
            orbit = _orbit_positions(hr, vc, key[0], key[1], (h, w))
            colors = [int(g[rr, cc]) for rr, cc in orbit if int(g[rr, cc]) not in (BLACK, RED)]
            if not colors:
                continue
            counts = Counter(colors)
            best = max(counts.values())
            chosen = min(color for color, cnt in counts.items() if cnt == best)
            for rr, cc in orbit:
                out[rr, cc] = chosen
    return out


def main(grid: np.ndarray) -> np.ndarray:
    """
    Mirror the dominant quadrant across the red cross, then resolve orbit conflicts by majority vote.

    Input structure:
    - A black grid is split into four quadrants by one full red horizontal line and one full red vertical line.
    - One quadrant contains the main blue/green pattern, while the other quadrants may contain at most a few sparse blue/green clue pixels.
    - No non-red pixel lies on either red separator line.

    Rule:
    - Detect the red row and red column, split the grid into quadrants, and choose the uniquely most-populated quadrant as the source pattern.
    - Reflect every non-black source pixel across the horizontal axis, vertical axis, and both axes to fill its full four-position symmetry orbit, but do not overwrite already visible clue pixels.
    - For each four-position orbit away from the red axes, collect all visible non-background colors in that orbit.
    - Recolor every position in the orbit to the most frequent visible color; if two colors tie, keep the smaller color id.
    - Preserve the red separator lines and keep untouched orbits black.

    Key insight:
    - The dense quadrant supplies the intended mirrored motif, while sparse off-quadrant clues create local disagreements that can only be repaired after expansion by orbit-wise majority voting.
    """
    g = np.asarray(grid, dtype=int)
    assert g.ndim == 2, "expected a 2D grid"

    hr, vc = _find_axes(g)
    source_name = _pick_source_quadrant(g, hr, vc)
    expanded = _pre_vote_expand(g, hr, vc, source_name)
    out = _vote_orbits(expanded, hr, vc)
    out[hr, :] = RED
    out[:, vc] = RED
    return out


def _in_bounds(shape: tuple[int, int], r: int, c: int) -> bool:
    return 0 <= r < shape[0] and 0 <= c < shape[1]


def _has_adjacent(coords: list[tuple[int, int]]) -> bool:
    pts = set(coords)
    for r, c in coords:
        if (r + 1, c) in pts or (r - 1, c) in pts or (r, c + 1) in pts or (r, c - 1) in pts:
            return True
    return False


def _render_local_patch(
    box_h: int,
    box_w: int,
    occupied: dict[tuple[int, int], int],
) -> np.ndarray:
    patch = np.zeros((box_h, box_w), dtype=int)
    for (rr, cc), color in occupied.items():
        patch[rr, cc] = color
    return patch


def _random_pattern(box_h: int, box_w: int) -> dict[tuple[int, int], int] | None:
    cells = [(r, c) for r in range(box_h) for c in range(box_w)]
    max_count = min(len(cells), max(MIN_PATTERN_PIXELS + 3, len(cells) - 1))
    if max_count < MIN_PATTERN_PIXELS:
        return None

    for _ in range(120):
        count = random.randint(MIN_PATTERN_PIXELS, max_count)
        chosen = random.sample(cells, count)
        if not _has_adjacent(chosen):
            continue

        occupied = {cell: random.choice((BLUE, GREEN)) for cell in chosen}
        colors = set(occupied.values())
        if colors != {BLUE, GREEN}:
            continue

        patch = _render_local_patch(box_h, box_w, occupied)
        if np.array_equal(patch, np.flipud(patch)):
            continue
        if np.array_equal(patch, np.fliplr(patch)):
            continue
        return occupied
    return None


def _absolute_position(source_name: str, hr: int, vc: int, dr: int, dc: int) -> tuple[int, int]:
    row_sign, col_sign = QUADRANTS[source_name]
    return hr + row_sign * dr, vc + col_sign * dc


def _non_source_positions(
    source_name: str,
    hr: int,
    vc: int,
    dr: int,
    dc: int,
    shape: tuple[int, int],
) -> list[tuple[int, int]]:
    source_pos = _absolute_position(source_name, hr, vc, dr, dc)
    return [pos for pos in _orbit_positions(hr, vc, dr, dc, shape) if pos != source_pos]


def _build_expected(
    shape: tuple[int, int],
    hr: int,
    vc: int,
    offsets_to_colors: dict[tuple[int, int], int],
) -> np.ndarray:
    out = np.zeros(shape, dtype=int)
    out[hr, :] = RED
    out[:, vc] = RED
    for (dr, dc), color in offsets_to_colors.items():
        for r, c in _orbit_positions(hr, vc, dr, dc, shape):
            out[r, c] = color
    out[hr, :] = RED
    out[:, vc] = RED
    return out


def _build_input(
    shape: tuple[int, int],
    hr: int,
    vc: int,
    source_name: str,
    offsets_to_colors: dict[tuple[int, int], int],
    noise: list[tuple[int, int, int]],
) -> np.ndarray:
    inp = np.zeros(shape, dtype=int)
    inp[hr, :] = RED
    inp[:, vc] = RED
    for (dr, dc), color in offsets_to_colors.items():
        r, c = _absolute_position(source_name, hr, vc, dr, dc)
        inp[r, c] = color
    for r, c, color in noise:
        inp[r, c] = color
    inp[hr, :] = RED
    inp[:, vc] = RED
    return inp


def _generate_attempt() -> tuple[np.ndarray, np.ndarray] | None:
    h = random.randint(GRID_SIZE_MIN, GRID_SIZE_MAX)
    w = random.randint(GRID_SIZE_MIN, GRID_SIZE_MAX)
    hr = random.randint(AXIS_MARGIN, h - AXIS_MARGIN - 1)
    vc = random.randint(AXIS_MARGIN, w - AXIS_MARGIN - 1)
    reach_r = min(hr, h - hr - 1)
    reach_c = min(vc, w - vc - 1)
    if reach_r < 3 or reach_c < 3:
        return None

    dims = [
        (bh, bw)
        for bh in range(2, min(PATTERN_BOX_MAX, reach_r) + 1)
        for bw in range(2, min(PATTERN_BOX_MAX, reach_c) + 1)
        if bh * bw >= 6
    ]
    if not dims:
        return None

    box_h, box_w = random.choice(dims)
    row_start = random.randint(1, reach_r - box_h + 1)
    col_start = random.randint(1, reach_c - box_w + 1)
    local_pattern = _random_pattern(box_h, box_w)
    if local_pattern is None:
        return None

    offsets_to_colors = {
        (row_start + rr, col_start + cc): color
        for (rr, cc), color in local_pattern.items()
    }
    source_name = random.choice(tuple(QUADRANTS))
    expected = _build_expected((h, w), hr, vc, offsets_to_colors)

    blue_offsets = [offset for offset, color in offsets_to_colors.items() if color == BLUE]
    noise: list[tuple[int, int, int]] = []

    use_tie = bool(blue_offsets) and random.random() < 0.45
    if use_tie:
        dr, dc = random.choice(blue_offsets)
        candidates = _non_source_positions(source_name, hr, vc, dr, dc, (h, w))
        chosen = random.sample(candidates, 2)
        noise.extend((r, c, GREEN) for r, c in chosen)
    else:
        noisy_orbits = random.sample(list(offsets_to_colors), k=random.randint(1, 2))
        for dr, dc in noisy_orbits:
            color = offsets_to_colors[(dr, dc)]
            wrong = GREEN if color == BLUE else BLUE
            candidates = _non_source_positions(source_name, hr, vc, dr, dc, (h, w))
            r, c = random.choice(candidates)
            noise.append((r, c, wrong))

    inp = _build_input((h, w), hr, vc, source_name, offsets_to_colors, noise)
    if _verify_generated(inp, expected, source_name):
        return inp, expected
    return None


def _verify_generated(inp: np.ndarray, expected: np.ndarray, source_name: str) -> bool:
    if inp.ndim != 2 or expected.ndim != 2:
        return False
    if inp.shape != expected.shape or max(inp.shape) > 30:
        return False

    try:
        hr, vc = _find_axes(inp)
    except AssertionError:
        return False

    counts = _quadrant_counts(inp, hr, vc)
    if max(counts.values()) < MIN_PATTERN_PIXELS:
        return False
    if counts[source_name] != max(counts.values()):
        return False
    if list(counts.values()).count(counts[source_name]) != 1:
        return False
    if sum(counts[name] for name in counts if name != source_name) not in (1, 2):
        return False

    source_patch = _quadrant_slice(inp, hr, vc, source_name)
    if not {BLUE, GREEN}.issubset(set(map(int, np.unique(source_patch)))):
        return False

    expanded = _pre_vote_expand(inp, hr, vc, source_name)
    if np.array_equal(expanded, expected):
        return False

    conflict_found = False
    seen: set[tuple[int, int]] = set()
    h, w = inp.shape
    for r in range(h):
        for c in range(w):
            if r == hr or c == vc:
                continue
            key = (abs(r - hr), abs(c - vc))
            if key in seen:
                continue
            seen.add(key)
            orbit = _orbit_positions(hr, vc, key[0], key[1], inp.shape)
            colors = {
                int(expanded[rr, cc])
                for rr, cc in orbit
                if int(expanded[rr, cc]) not in (BLACK, RED)
            }
            if len(colors) > 1:
                conflict_found = True
                break
        if conflict_found:
            break
    if not conflict_found:
        return False

    for r in range(h):
        for c in range(w):
            if r == hr or c == vc:
                if expected[r, c] != RED:
                    return False
                continue
            orbit = _orbit_positions(hr, vc, abs(r - hr), abs(c - vc), inp.shape)
            values = {int(expected[rr, cc]) for rr, cc in orbit}
            if len(values) > 1:
                return False

    try:
        actual = main(inp)
    except AssertionError:
        return False
    return np.array_equal(actual, expected)


def _quadrant_slice(grid: np.ndarray, hr: int, vc: int, name: str) -> np.ndarray:
    h, w = grid.shape
    ((r0, r1), (c0, c1)) = _quadrant_bounds(h, w, hr, vc)[name]
    return grid[r0:r1, c0:c1]


def generate_input() -> np.ndarray:
    for _ in range(MAX_GENERATION_ATTEMPTS):
        built = _generate_attempt()
        if built is not None:
            return built[0]
    raise RuntimeError("failed to generate a valid mirrored-majority ARC input")
