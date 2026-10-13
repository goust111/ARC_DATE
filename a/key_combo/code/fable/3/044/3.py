from __future__ import annotations

import random
from collections import Counter

import numpy as np


def _get_bg(grid: np.ndarray) -> int:
    return int(Counter(map(int, grid.ravel())).most_common(1)[0][0])


def _in_bounds(shape: tuple[int, int], r: int, c: int) -> bool:
    h, w = shape
    return 0 <= int(r) < h and 0 <= int(c) < w


def _foreground_points(g: np.ndarray, bg: int) -> list[tuple[int, int, int]]:
    return [(int(r), int(c), int(g[r, c])) for r, c in np.argwhere(g != bg)]


def _reflection_targets(
    r: int, c: int, axis_r: int, axis_c: int
) -> tuple[tuple[int, int], tuple[int, int], tuple[int, int]]:
    return (
        (2 * axis_r - r, c),
        (r, 2 * axis_c - c),
        (2 * axis_r - r, 2 * axis_c - c),
    )


def _orbit_positions(
    shape: tuple[int, int], r: int, c: int, axis_r: int, axis_c: int
) -> tuple[tuple[int, int], ...]:
    coords = (
        (r, c),
        (2 * axis_r - r, c),
        (r, 2 * axis_c - c),
        (2 * axis_r - r, 2 * axis_c - c),
    )
    out: list[tuple[int, int]] = []
    seen: set[tuple[int, int]] = set()
    for rr, cc in coords:
        pos = (int(rr), int(cc))
        if _in_bounds(shape, *pos) and pos not in seen:
            out.append(pos)
            seen.add(pos)
    return tuple(out)


def _empty_axis_candidates(g: np.ndarray, bg: int) -> tuple[list[int], list[int]]:
    rows = [int(r) for r in range(g.shape[0]) if np.all(g[r, :] == bg)]
    cols = [int(c) for c in range(g.shape[1]) if np.all(g[:, c] == bg)]
    assert rows and cols, "the hidden axes must be empty background lines"
    return rows, cols


def _same_color_pair_score(
    points: list[tuple[int, int, int]],
    color_at: dict[tuple[int, int], int],
    shape: tuple[int, int],
    axis_r: int,
    axis_c: int,
) -> int:
    pairs: set[tuple[tuple[int, int], tuple[int, int]]] = set()
    for r, c, color in points:
        p = (r, c)
        for rr, cc in _reflection_targets(r, c, axis_r, axis_c):
            q = (int(rr), int(cc))
            if q == p or not _in_bounds(shape, *q):
                continue
            if color_at.get(q) == color:
                pairs.add((p, q) if p < q else (q, p))
    return len(pairs)


def _find_best_axes(g: np.ndarray, bg: int) -> tuple[int, int]:
    points = _foreground_points(g, bg)
    assert points, "expected sparse foreground markers"
    color_at = {(r, c): color for r, c, color in points}

    rows, cols = _empty_axis_candidates(g, bg)
    scored: list[tuple[int, int, int]] = []
    for axis_r in rows:
        for axis_c in cols:
            score = _same_color_pair_score(points, color_at, g.shape, axis_r, axis_c)
            scored.append((score, axis_r, axis_c))

    best_score = max(score for score, _r, _c in scored)
    assert best_score > 0, "at least one same-color mirror pair is required"
    best = [(r, c) for score, r, c in scored if score == best_score]
    assert len(best) == 1, "hidden row and column axes must be uniquely determined"
    return best[0]


def _has_same_color_partner(
    r: int,
    c: int,
    color: int,
    color_at: dict[tuple[int, int], int],
    shape: tuple[int, int],
    axis_r: int,
    axis_c: int,
) -> bool:
    for rr, cc in _reflection_targets(r, c, axis_r, axis_c):
        q = (int(rr), int(cc))
        if q != (r, c) and _in_bounds(shape, *q) and color_at.get(q) == color:
            return True
    return False


def main(grid: np.ndarray) -> np.ndarray:
    """
    Infer the hidden horizontal and vertical mirror axes, complete confirmed marker quartets, and erase lone noise.

    Input structure:
    - A dominant background contains sparse one-pixel colored markers.
    - The mirror axes are not drawn; they are one empty row and one empty column.
    - Most markers have a same-color partner under reflection across the row axis, the column axis, or both axes.
    - A few remaining markers have no same-color partner in that hidden axis system.

    Rule:
    - Search all empty row and empty column pairs and choose the unique pair with the most same-color reflected marker pairs.
    - For every marker that has at least one same-color reflected partner, paint all in-bounds positions in its four-position mirror orbit.
    - Treat out-of-bounds reflected positions as clipped, and erase every unpartnered marker back to the background.

    Key insight:
    - The axes are latent generators inferred from partial same-color pair evidence; partnership, not mere foreground status, separates the true pattern from noise.
    """
    g = np.asarray(grid, dtype=int)
    assert g.ndim == 2, "expected a 2D grid"
    bg = _get_bg(g)
    axis_r, axis_c = _find_best_axes(g, bg)

    points = _foreground_points(g, bg)
    color_at = {(r, c): color for r, c, color in points}
    out = np.full_like(g, bg)

    for r, c, color in points:
        assert r != axis_r and c != axis_c, "markers must not lie on the hidden axes"
        if not _has_same_color_partner(r, c, color, color_at, g.shape, axis_r, axis_c):
            continue
        for rr, cc in _orbit_positions(g.shape, r, c, axis_r, axis_c):
            existing = int(g[rr, cc])
            assert existing in (bg, color), "completed marker orbits must not cross other colors"
            already = int(out[rr, cc])
            assert already in (bg, color), "completed marker orbits must not conflict"
            out[rr, cc] = color

    return out


# --------------- generator ---------------

_GEN_MIN_SIZE = 13
_GEN_MAX_SIZE = 25
_GEN_MAX_ATTEMPTS = 800
_GEN_MIN_MARKER_GAP = 2


def _gen_axis_options(n: int) -> list[int]:
    opts: list[int] = []
    for axis in range(2, n - 2):
        near = min(axis, n - 1 - axis)
        far = max(axis, n - 1 - axis)
        if near >= 2 and far >= near + 2:
            opts.append(axis)
    return opts


def _gen_orbits(
    h: int, w: int, axis_r: int, axis_c: int
) -> tuple[list[tuple[tuple[int, int], ...]], list[tuple[tuple[int, int], ...]]]:
    full: list[tuple[tuple[int, int], ...]] = []
    clipped: list[tuple[tuple[int, int], ...]] = []
    seen: set[frozenset[tuple[int, int]]] = set()
    for r in range(h):
        if r == axis_r:
            continue
        for c in range(w):
            if c == axis_c:
                continue
            orbit = _orbit_positions((h, w), r, c, axis_r, axis_c)
            if len(orbit) < 2:
                continue
            key = frozenset(orbit)
            if key in seen:
                continue
            seen.add(key)
            ordered = tuple(sorted(orbit))
            if len(ordered) == 4:
                full.append(ordered)
            elif len(ordered) == 2:
                clipped.append(ordered)
    return full, clipped


def _full_pair(
    orbit: tuple[tuple[int, int], ...], axis_r: int, axis_c: int, kind: str
) -> list[tuple[int, int]]:
    by_side: dict[tuple[int, int], tuple[int, int]] = {}
    for r, c in orbit:
        by_side[(1 if r > axis_r else -1, 1 if c > axis_c else -1)] = (r, c)
    assert len(by_side) == 4
    top_left = by_side[(-1, -1)]
    top_right = by_side[(-1, 1)]
    bottom_left = by_side[(1, -1)]
    bottom_right = by_side[(1, 1)]
    if kind == "row":
        return [top_left, bottom_left]
    if kind == "col":
        return [top_left, top_right]
    if kind == "diag":
        return [top_left, bottom_right]
    coords = [top_left, top_right, bottom_left, bottom_right]
    random.shuffle(coords)
    return coords[: random.choice((2, 3))]


def _spaced(coords: list[tuple[int, int]], min_gap: int = _GEN_MIN_MARKER_GAP) -> bool:
    for i, (r0, c0) in enumerate(coords):
        for r1, c1 in coords[i + 1 :]:
            if max(abs(r0 - r1), abs(c0 - c1)) < min_gap:
                return False
    return True


def _gen_expected(
    shape: tuple[int, int],
    bg: int,
    groups: list[tuple[int, tuple[tuple[int, int], ...], list[tuple[int, int]]]],
) -> np.ndarray:
    out = np.full(shape, bg, dtype=int)
    for color, orbit, _observed in groups:
        for r, c in orbit:
            out[r, c] = color
    return out


def _gen_attempt() -> np.ndarray | None:
    h = random.randint(_GEN_MIN_SIZE, _GEN_MAX_SIZE)
    w = random.randint(_GEN_MIN_SIZE, _GEN_MAX_SIZE)
    row_options = _gen_axis_options(h)
    col_options = _gen_axis_options(w)
    if not row_options or not col_options:
        return None

    axis_r = random.choice(row_options)
    axis_c = random.choice(col_options)
    full_orbits, clipped_orbits = _gen_orbits(h, w, axis_r, axis_c)
    if len(full_orbits) < 3 or not clipped_orbits:
        return None

    random.shuffle(full_orbits)
    random.shuffle(clipped_orbits)
    full_count = random.randint(3, min(5, len(full_orbits)))
    clip_count = random.randint(1, min(2, len(clipped_orbits)))
    selected_full = full_orbits[:full_count]
    selected_clipped = clipped_orbits[:clip_count]

    bg = random.randrange(10)
    palette = [color for color in range(10) if color != bg]
    group_count = full_count + clip_count
    max_noise = min(2, len(palette) - group_count)
    if max_noise < 1:
        return None
    noise_count = random.randint(1, max_noise)
    colors = random.sample(palette, group_count + noise_count)

    groups: list[tuple[int, tuple[tuple[int, int], ...], list[tuple[int, int]]]] = []
    required_kinds = ["diag", "row", "col"]
    for i, orbit in enumerate(selected_full):
        kind = required_kinds[i] if i < len(required_kinds) else "random"
        observed = _full_pair(orbit, axis_r, axis_c, kind)
        groups.append((colors[i], orbit, observed))

    offset = len(selected_full)
    for j, orbit in enumerate(selected_clipped):
        groups.append((colors[offset + j], orbit, list(orbit)))

    marker_coords: list[tuple[int, int]] = []
    used: set[tuple[int, int]] = set()
    true_output_cells: set[tuple[int, int]] = set()
    grid = np.full((h, w), bg, dtype=int)
    for color, orbit, observed in groups:
        true_output_cells.update(orbit)
        for r, c in observed:
            if (r, c) in used or r == axis_r or c == axis_c:
                return None
            used.add((r, c))
            marker_coords.append((r, c))
            grid[r, c] = color

    if not _spaced(marker_coords):
        return None

    noise_colors = colors[group_count:]
    noise_coords: list[tuple[int, int]] = []
    for color in noise_colors:
        choices = [
            (r, c)
            for r in range(h)
            for c in range(w)
            if r != axis_r
            and c != axis_c
            and (r, c) not in used
            and (r, c) not in true_output_cells
            and _spaced(marker_coords + noise_coords + [(r, c)])
        ]
        if not choices:
            return None
        r, c = random.choice(choices)
        grid[r, c] = color
        used.add((r, c))
        noise_coords.append((r, c))

    expected = _gen_expected((h, w), bg, groups)
    input_marker_count = int(np.count_nonzero(grid != bg))
    true_marker_count = input_marker_count - len(noise_coords)
    if true_marker_count <= len(noise_coords):
        return None
    if not np.all(grid[axis_r, :] == bg) or not np.all(grid[:, axis_c] == bg):
        return None
    if not any(len(orbit) == 2 for _color, orbit, _observed in groups):
        return None
    if not any(len(orbit) > len(observed) for _color, orbit, observed in groups):
        return None
    if not all(expected[r, c] == bg for r, c in noise_coords):
        return None
    if not np.any((grid == bg) & (expected != bg)):
        return None

    try:
        found_axes = _find_best_axes(grid, bg)
        actual = main(grid)
    except Exception:
        return None
    if found_axes != (axis_r, axis_c):
        return None
    if not np.array_equal(actual, expected):
        return None

    return grid


def generate_input() -> np.ndarray:
    for _ in range(_GEN_MAX_ATTEMPTS):
        grid = _gen_attempt()
        if grid is not None:
            return grid
    raise RuntimeError("failed to generate a valid hidden-biaxial marker input")
