# Combo variant of 0934a4d8 (index 9)
# combo_spec: D7/2+D10/1
# Name: 小网格+大标记

from __future__ import annotations

from collections import deque

import numpy as np

from arc_utils import bbox


# Design analysis for the fused variant:
# 1. Concrete trace:
#    The original task used a 30x30 canvas with a medium rectangle. Here the grid is only
#    10-15 wide, while the marker covers strictly more than a quarter of the whole grid.
#    To keep transpose fallback physically possible on such tiny boards, the free top/left
#    border is widened adaptively (lo=4, and lo=5 only for N=15). Example: on a 12x12 grid
#    with lo=4, a 5x8 marker hides 40/144 cells. A cell on the top border of that marker may
#    have no visible reflection mate at all, so the solver must fall back to g[c, r].
# 2. Complexity calibration:
#    A literal "two direct reflections hidden but the double reflection visible" state is
#    impossible for an axis-aligned solid rectangle: if (r,c), (r,c'), and (r',c) are all
#    inside the rectangle, then (r',c') is inside it too. So the practical large-marker
#    effect is not new orbit topology, but orbit sparsity: many cells have only one visible
#    orbit peer, and some have zero visible peers and therefore require transpose fallback.
# 3. Solver assumptions that break:
#    The original code assumes a large amount of visible evidence and trusts the first lo
#    whose mirrored strips are consistent. With a huge marker on a tiny board, smaller fake
#    lo values can be accidentally consistent. The new solver enumerates candidate lo values
#    and keeps only those that can recover every marker cell without ambiguity.
# 4. Placement legality:
#    Large markers are accepted only when area > N*N/4, when they touch the widened free
#    top/left border (so fully hidden orbits can use transpose fallback), and when every
#    fallback transpose cell lies outside the marker.
# 5. Visual observability:
#    Each sample uses only 4-5 non-marker colors even though the global palette is larger.
#    This keeps the tiny grids readable, makes the mirrored structure visible by eye, and
#    prevents the fused variant from degenerating into noisy random color texture.


_PALETTE = np.array([1, 2, 3, 4, 5, 6, 7, 9], dtype=int)
_MARKER = 8

_GRID_MIN = 10
_GRID_MAX = 15
_MAX_RETRIES = 2000

_RNG = np.random.default_rng()
_SIZE_CACHE: dict[int, list[tuple[int, int]]] = {}
_PLACEMENT_CACHE: dict[tuple[int, int, int, int], list[dict[str, object]]] = {}


def _solid_rectangle_marker(g: np.ndarray) -> tuple[int, tuple[int, int, int, int]]:
    candidates: list[tuple[int, int, tuple[int, int, int, int]]] = []
    for color in np.unique(g):
        mask = g == int(color)
        b = bbox(mask)
        assert b is not None
        r0, c0, r1, c1 = b
        area = int((r1 - r0 + 1) * (c1 - c0 + 1))
        if area > 1 and int(mask.sum()) == area and bool(np.all(mask[r0 : r1 + 1, c0 : c1 + 1])):
            candidates.append((area, int(color), b))

    assert len(candidates) == 1
    _area, color, b = candidates[0]
    return color, b


def _candidate_strip_starts(g: np.ndarray, marker: int) -> list[int]:
    n, m = map(int, g.shape)
    assert n == m
    hi = n - 1
    out: list[int] = []

    for lo in range(n // 2 + 1):
        strip = g[:, lo : hi + 1]
        strip_reflected = strip[:, ::-1]
        known = (strip != marker) & (strip_reflected != marker)
        if bool(np.any((strip != strip_reflected) & known)):
            continue

        strip = g[lo : hi + 1, :]
        strip_reflected = strip[::-1, :]
        known = (strip != marker) & (strip_reflected != marker)
        if bool(np.any((strip != strip_reflected) & known)):
            continue

        out.append(lo)

    return out


def _border_width(n: int) -> int:
    return 5 if n >= 15 else 4


def _reflection_orbit_values(g: np.ndarray, r: int, c: int, lo: int, marker: int) -> set[int]:
    n = int(g.shape[0])
    hi = n - 1
    seen: set[tuple[int, int]] = set()
    values: set[int] = set()
    q: deque[tuple[int, int]] = deque([(int(r), int(c))])

    while q:
        rr, cc = q.popleft()
        if (rr, cc) in seen:
            continue
        seen.add((rr, cc))

        value = int(g[rr, cc])
        if value != marker:
            values.add(value)

        if lo <= cc <= hi:
            q.append((rr, lo + hi - cc))
        if lo <= rr <= hi:
            q.append((lo + hi - rr, cc))

    return values


def _recover_cell(g: np.ndarray, r: int, c: int, lo: int, marker: int) -> int:
    values = _reflection_orbit_values(g, r, c, lo, marker)
    if values:
        assert len(values) == 1
        return values.pop()

    assert r < lo or c < lo
    value = int(g[c, r])
    assert value != marker
    return value


def _choose_reflected_strip_start(
    g: np.ndarray, marker: int, region: tuple[int, int, int, int]
) -> int:
    r0, c0, r1, c1 = region
    valid: list[int] = []

    for lo in _candidate_strip_starts(g, marker):
        try:
            for r in range(r0, r1 + 1):
                for c in range(c0, c1 + 1):
                    _recover_cell(g, r, c, lo, marker)
        except Exception:
            continue
        valid.append(lo)

    assert len(valid) == 1
    return valid[0]


def main(grid: np.ndarray) -> np.ndarray:
    """
    Reconstruct the large rectangle hidden by the marker inside a much smaller symmetric board.

    Input structure:
    - A square grid of side 10-15 containing one solid rectangular marker color.
    - Outside the marker, rows/cols from index lo onward still obey bilateral left-right and
      up-down symmetry, while the top/left lo-wide border remains the free transpose backup zone.
    - Because the marker is huge, many cells expose only one visible orbit mate, and a few border
      cells expose none at all.

    Rule:
    - Detect the unique solid-color rectangle and treat its footprint as the output window.
    - Infer the symmetry strip start by testing every mirror-consistent lo candidate and keeping
      the unique one that can recover every hidden cell without contradiction.
    - For each marker cell, read the single non-marker value in its reflection orbit; if the whole
      orbit is hidden, use the transposed backup cell g[c, r].

    Key insight:
    The fused variant does not change what determines each hidden color; it changes how little of
    that evidence stays visible. The rectangle is recovered by exploiting the same symmetry orbit,
    but now under much tighter visibility and with fallback cells appearing more often.
    """
    g = np.asarray(grid, dtype=int)
    marker, (r0, c0, r1, c1) = _solid_rectangle_marker(g)
    lo = _choose_reflected_strip_start(g, marker, (r0, c0, r1, c1))

    h = int(r1 - r0 + 1)
    w = int(c1 - c0 + 1)
    out = np.empty((h, w), dtype=int)

    for rr in range(h):
        for cc in range(w):
            out[rr, cc] = _recover_cell(g, int(r0 + rr), int(c0 + cc), lo, marker)

    return out


def _mirror_index(i: int, lo: int, hi: int) -> int:
    return int(lo + hi - i)


def _orbit(r: int, c: int, lo: int, hi: int) -> set[tuple[int, int]]:
    rows = {int(r)}
    cols = {int(c)}
    if lo <= r <= hi:
        rows.add(_mirror_index(r, lo, hi))
    if lo <= c <= hi:
        cols.add(_mirror_index(c, lo, hi))
    return {(rr, cc) for rr in rows for cc in cols}


def _in_rect(r: int, c: int, r0: int, c0: int, h: int, w: int) -> bool:
    return r0 <= r < r0 + h and c0 <= c < c0 + w


def _size_pairs(n: int) -> list[tuple[int, int]]:
    if n not in _SIZE_CACHE:
        min_area = n * n // 4 + 1
        pairs = [
            (h, w)
            for h in range(3, n + 1)
            for w in range(3, n + 1)
            if min_area <= h * w < n * n
        ]
        _SIZE_CACHE[n] = pairs
    return _SIZE_CACHE[n]


def _placement_profile(n: int, lo: int, r0: int, c0: int, h: int, w: int) -> dict[str, object] | None:
    area = int(h * w)
    if area * 4 <= n * n:
        return None
    if not (r0 < lo or c0 < lo):
        return None

    hi = n - 1
    fallback: list[tuple[int, int]] = []
    visible_total = 0
    sparse_cells = 0

    for r in range(r0, r0 + h):
        for c in range(c0, c0 + w):
            orbit = _orbit(r, c, lo, hi)
            visible = [(rr, cc) for rr, cc in orbit if not _in_rect(rr, cc, r0, c0, h, w)]
            peer_count = len(visible)
            visible_total += peer_count
            if peer_count <= 1:
                sparse_cells += 1

            if peer_count == 0:
                if r >= lo and c >= lo:
                    return None
                if _in_rect(c, r, r0, c0, h, w):
                    return None
                fallback.append((r, c))

    if not fallback:
        return None
    if sparse_cells * 4 < area * 3:
        return None
    if visible_total * 2 > area * 3:
        return None

    score = float(area * 3 - visible_total + 4 * len(fallback))
    return {
        "r0": r0,
        "c0": c0,
        "h": h,
        "w": w,
        "area": area,
        "fallback": fallback,
        "visible_total": visible_total,
        "sparse_cells": sparse_cells,
        "score": score,
    }


def _valid_placements(n: int, lo: int, h: int, w: int) -> list[dict[str, object]]:
    key = (n, lo, h, w)
    if key not in _PLACEMENT_CACHE:
        placements: list[dict[str, object]] = []
        for r0 in range(n - h + 1):
            for c0 in range(n - w + 1):
                profile = _placement_profile(n, lo, r0, c0, h, w)
                if profile is not None:
                    placements.append(profile)
        _PLACEMENT_CACHE[key] = placements
    return _PLACEMENT_CACHE[key]


def _choose_palette() -> np.ndarray:
    k = int(_RNG.integers(4, 6))
    return np.sort(_RNG.choice(_PALETTE, size=k, replace=False)).astype(int)


def _build_clean(n: int, lo: int, palette: np.ndarray) -> np.ndarray:
    hi = n - 1
    clean = np.full((n, n), -1, dtype=int)

    clean[:lo, :lo] = _RNG.choice(palette, size=(lo, lo))

    for r in range(lo):
        row_colors: dict[int, int] = {}
        for c in range(lo, n):
            rep = min(c, _mirror_index(c, lo, hi))
            row_colors.setdefault(rep, int(_RNG.choice(palette)))
            clean[r, c] = row_colors[rep]

    for c in range(lo):
        col_colors: dict[int, int] = {}
        for r in range(lo, n):
            rep = min(r, _mirror_index(r, lo, hi))
            col_colors.setdefault(rep, int(_RNG.choice(palette)))
            clean[r, c] = col_colors[rep]

    core_colors: dict[tuple[int, int], int] = {}
    for r in range(lo, n):
        rr = min(r, _mirror_index(r, lo, hi))
        for c in range(lo, n):
            cc = min(c, _mirror_index(c, lo, hi))
            key = (rr, cc)
            core_colors.setdefault(key, int(_RNG.choice(palette)))
            clean[r, c] = core_colors[key]

    assert np.all(clean >= 0)
    return clean


def _unify_groups(clean: np.ndarray, groups: list[set[tuple[int, int]]], palette: np.ndarray) -> None:
    parent: dict[tuple[int, int], tuple[int, int]] = {}

    def find(x: tuple[int, int]) -> tuple[int, int]:
        parent.setdefault(x, x)
        while parent[x] != x:
            parent[x] = parent[parent[x]]
            x = parent[x]
        return x

    def union(a: tuple[int, int], b: tuple[int, int]) -> None:
        ra, rb = find(a), find(b)
        if ra != rb:
            parent[rb] = ra

    for group in groups:
        coords = list(group)
        if not coords:
            continue
        for coord in coords[1:]:
            union(coords[0], coord)

    merged: dict[tuple[int, int], list[tuple[int, int]]] = {}
    for coord in parent:
        merged.setdefault(find(coord), []).append(coord)

    for coords in merged.values():
        color = int(_RNG.choice(palette))
        for r, c in coords:
            clean[r, c] = color


def _behavior_metrics(grid: np.ndarray) -> dict[str, float]:
    g = np.asarray(grid, dtype=int)
    marker, (r0, c0, r1, c1) = _solid_rectangle_marker(g)
    lo = _choose_reflected_strip_start(g, marker, (r0, c0, r1, c1))
    h = int(r1 - r0 + 1)
    w = int(c1 - c0 + 1)
    area = h * w
    hi = int(g.shape[0] - 1)

    fallback = 0
    sparse = 0
    visible_total = 0
    for r in range(r0, r1 + 1):
        for c in range(c0, c1 + 1):
            orbit = _orbit(r, c, lo, hi)
            peer_count = sum(not _in_rect(rr, cc, r0, c0, h, w) for rr, cc in orbit)
            visible_total += int(peer_count)
            if peer_count <= 1:
                sparse += 1
            if peer_count == 0:
                fallback += 1

    return {
        "n": float(g.shape[0]),
        "lo": float(lo),
        "area_ratio": area / float(g.shape[0] * g.shape[1]),
        "fallback_cells": float(fallback),
        "sparse_ratio": sparse / float(area),
        "avg_visible_peers": visible_total / float(area),
    }


def _gen_attempt() -> np.ndarray | None:
    n = int(_RNG.integers(_GRID_MIN, _GRID_MAX + 1))
    lo = _border_width(n)
    palette = _choose_palette()
    clean = _build_clean(n, lo, palette)

    options: list[dict[str, object]] = []
    for h, w in _size_pairs(n):
        options.extend(_valid_placements(n, lo, h, w))
    if not options:
        return None

    weights = np.array([float(opt["score"]) for opt in options], dtype=float)
    weights = weights / weights.sum()
    choice = options[int(_RNG.choice(len(options), p=weights))]

    r0 = int(choice["r0"])
    c0 = int(choice["c0"])
    h = int(choice["h"])
    w = int(choice["w"])
    fallback = list(choice["fallback"])

    groups = [_orbit(r, c, lo, n - 1) | _orbit(c, r, lo, n - 1) for r, c in fallback]
    _unify_groups(clean, groups, palette)

    expected = clean[r0 : r0 + h, c0 : c0 + w].copy()
    inp = clean.copy()
    inp[r0 : r0 + h, c0 : c0 + w] = _MARKER

    try:
        marker, region = _solid_rectangle_marker(inp)
        if marker != _MARKER or region != (r0, c0, r0 + h - 1, c0 + w - 1):
            return None
        if not np.array_equal(main(inp), expected):
            return None
        metrics = _behavior_metrics(inp)
    except Exception:
        return None

    if metrics["area_ratio"] <= 0.25:
        return None
    if metrics["fallback_cells"] < 1:
        return None
    if metrics["sparse_ratio"] < 0.75:
        return None
    if metrics["avg_visible_peers"] > 1.5:
        return None

    return inp


def generate_input() -> np.ndarray:
    for _ in range(_MAX_RETRIES):
        grid = _gen_attempt()
        if grid is not None:
            return grid
    raise RuntimeError("failed to generate a valid 0934a4d8 combo input")
