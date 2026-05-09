# Combo variant of 0934a4d8 (index 20)
# combo_spec: D1/1+D4/3+D14/1
# Name: C4旋转+180度恢复+跨旋转中心标记

from __future__ import annotations

import numpy as np

from arc_utils import bbox


# Design analysis for the combo variant
#
# a) Concrete behavior and trace:
# - The clean board keeps a 30x30 layout with a unique solid marker rectangle, but the
#   recoverable structure is now the C4-symmetric square [lo..N-1] x [lo..N-1] rather
#   than the original LR+UD mirror scaffold.
# - Recovery is no longer "collect any reflection-orbit witness". The intended first
#   source is the 180-degree counterpart `(lo+hi-r, lo+hi-c)`.
# - The marker is forced to cover the rotation center, so some cells lose that direct
#   point-symmetric witness because the 180-degree partner is also inside the marker.
#   Those cells then fall back to the 90/270-degree positions from the same C4 orbit.
#   Example with lo=2, hi=29: for hidden `(15,14)`, the 180 partner `(16,17)` can also
#   be hidden by a center-spanning rectangle, while the quarter-turn partners `(14,16)`
#   and `(17,15)` remain visible and expose the same value.
# - In an even-sized C4 square, the exact central 2x2 orbit becomes fully buried as soon
#   as a rectangle spans both center lines. To keep the fused variant physically valid,
#   the generator copies that unique 2x2 orbit into the otherwise free top-left 2x2
#   corner, giving the solver one visible proxy only for that degenerate all-hidden case.
# - In the original task the same cell would have been solved from LR/UD reflections or
#   a transpose backup. Here the geometry is specifically "try point symmetry first,
#   then exploit quarter-turn symmetry when point symmetry is blocked by the centered
#   marker itself".
#
# b) Complexity calibration:
# - If the rectangle merely crossed one center line, 180-degree recovery would usually
#   stay visible and the task would collapse into a plain D4/3 point-symmetry puzzle.
#   To make the fusion real, valid placements must cross both the row-center pair and
#   the column-center pair, i.e. truly cover the rotation center.
# - If the rectangle were allowed to be a C4-stable centered square, all four rotated
#   copies of some cells could be hidden at once. The generator filters those cases out.
# - The inner pattern is built from one random quadrant and its 90/180/270 rotations,
#   so the board visibly reads as rotational rather than reflective. We also reject
#   accidental LR/UD-looking boards to keep the visual change obvious.
#
# c) Hidden-assumption audit against the original main():
# - `_find_reflected_strip_start` depended on bilateral mirrors; that assumption fails
#   completely and must become a 90-degree rotation consistency check on the square
#   suffix block.
# - `_reflection_orbit_values` assumed an unordered reflection orbit. The combo needs a
#   prioritized recovery chain: 180-degree mate first, then 90/270-degree witnesses only
#   when the 180-degree mate is hidden.
# - The old transpose fallback is invalid here. Once the marker is centered, the relevant
#   hidden ambiguity lives inside the rotational square, so the solver must rely on C4
#   structure; only the unavoidable buried center 2x2 uses the dedicated corner proxy.
#
# d) Placement legality:
# - The marker is restricted to the rotational square [lo..N-1]^2, so every hidden cell
#   participates in the same C4 geometry and there is no unsupported border case.
# - A placement is legal only if it crosses both center lines and every marker cell has
#   either a visible 180-degree partner or, failing that, at least one visible 90/270
#   partner. The generator computes these counts before accepting a placement.
#
# e) Visual observability:
# - Across samples, a human can see two stable clues: the large lower-right square is a
#   quarter-turn tiling, and the marker always straddles the exact rotational center.
# - Because the output is still the hidden rectangle, examples reveal that some central
#   cells must come from sideways quarter-turn copies after their point-symmetric mates
#   disappear under the same marker.
#
# f) Visual cleanliness:
# - The scene still contains one explicit object: the solid marker rectangle. No legends,
#   overlays, or color remapping are added.
# - Keeping the marker inside the main C4 square avoids mixing unrelated border logic
#   with the new centered-rotation rule.
#
# g) Color inference:
# - There is no color-id mapping. All recovery is positional, and the generator only
#   enforces equality along visible C4 orbits plus the explicit center-to-corner proxy.
#
# Post-change audit of new main() assumptions:
# - `main()` assumes the unique solid rectangle exists and spans the rotation center;
#   generator-side placement filtering and end-to-end self-checks guarantee this.
# - `main()` assumes the smallest rotated-consistent suffix is the true `lo=2`; the
#   generator rejects accidental `lo=0/1` boards with `_find_rotated_strip_start`.
# - `main()` assumes any cell whose 180-degree mate is hidden either has a unique visible
#   90/270-degree witness or belongs to the special central 2x2 proxy orbit; generation
#   enforces exactly that.


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


def _find_rotated_strip_start(g: np.ndarray, marker: int) -> int:
    n, m = map(int, g.shape)
    assert n == m

    for lo in range(n // 2 + 1):
        block = g[lo:n, lo:n]
        rotated = np.rot90(block, -1)
        known = (block != marker) & (rotated != marker)
        if bool(np.any((block != rotated) & known)):
            continue
        return lo

    raise AssertionError("no rotated strip found")


def _axis_pair(lo: int, n: int) -> tuple[int, int]:
    hi = n - 1
    left = (lo + hi) // 2
    return left, left + 1


def _crosses_axis(start: int, end: int, lo: int, n: int) -> bool:
    axis_lo, axis_hi = _axis_pair(lo, n)
    return int(start) <= axis_lo and int(end) >= axis_hi


def _covers_rotation_center(r0: int, c0: int, h: int, w: int, lo: int, n: int) -> bool:
    return _crosses_axis(r0, r0 + h - 1, lo, n) and _crosses_axis(c0, c0 + w - 1, lo, n)


def _rotate90(r: int, c: int, lo: int, hi: int) -> tuple[int, int]:
    return int(c), int(lo + hi - r)


def _rotate180(r: int, c: int, lo: int, hi: int) -> tuple[int, int]:
    return int(lo + hi - r), int(lo + hi - c)


def _rotate270(r: int, c: int, lo: int, hi: int) -> tuple[int, int]:
    return int(lo + hi - c), int(r)


def _inside_rect(r: int, c: int, r0: int, c0: int, h: int, w: int) -> bool:
    return r0 <= int(r) < r0 + h and c0 <= int(c) < c0 + w


def _recovery_mode_counts(
    r0: int, c0: int, h: int, w: int, lo: int, hi: int
) -> tuple[int, int, int] | None:
    direct_180 = 0
    via_c4 = 0
    via_corner = 0
    axis_lo, axis_hi = _axis_pair(lo, hi + 1)
    for r in range(r0, r0 + h):
        for c in range(c0, c0 + w):
            r180, c180 = _rotate180(r, c, lo, hi)
            if not _inside_rect(r180, c180, r0, c0, h, w):
                direct_180 += 1
                continue

            visible_quarter_turns = 0
            for rr, cc in (_rotate90(r, c, lo, hi), _rotate270(r, c, lo, hi)):
                if not _inside_rect(rr, cc, r0, c0, h, w):
                    visible_quarter_turns += 1
            if visible_quarter_turns > 0:
                via_c4 += 1
            elif axis_lo <= r <= axis_hi and axis_lo <= c <= axis_hi:
                via_corner += 1
            else:
                return None

    return direct_180, via_c4, via_corner


def _center_proxy_value(g: np.ndarray, r: int, c: int, lo: int, marker: int) -> int:
    axis_lo, axis_hi = _axis_pair(lo, int(g.shape[0]))
    assert axis_lo <= r <= axis_hi and axis_lo <= c <= axis_hi
    value = int(g[r - axis_lo, c - axis_lo])
    assert value != marker
    return value


def _recover_value(g: np.ndarray, r: int, c: int, lo: int, marker: int) -> int:
    hi = int(g.shape[0] - 1)
    r180, c180 = _rotate180(r, c, lo, hi)
    value = int(g[r180, c180])
    if value != marker:
        return value

    values: set[int] = set()
    for rr, cc in (_rotate90(r, c, lo, hi), _rotate270(r, c, lo, hi)):
        candidate = int(g[rr, cc])
        if candidate != marker:
            values.add(candidate)

    if values:
        assert len(values) == 1
        return values.pop()
    return _center_proxy_value(g, r, c, lo, marker)


def main(grid: np.ndarray) -> np.ndarray:
    """
    Reveal the rectangle hidden by the solid marker by combining point symmetry with quarter-turn recovery.

    Input structure:
    - A square colored grid contains one solid rectangular marker that hides the answer patch.
    - The large suffix square [lo..N-1] x [lo..N-1] is C4-symmetric: its clean content is invariant under 90-degree rotation.
    - The marker lies inside that rotational square and spans both center lines, so it covers the rotation center itself.
    - Because that centered placement fully hides the exact central 2x2 orbit, the free top-left 2x2 corner shows a visible copy of that orbit.

    Rule:
    - Find the unique solid rectangle color and use its bounding box as the output window.
    - Infer the smallest `lo` whose suffix square is 90-degree rotation-consistent when marker pixels are ignored.
    - For each hidden cell, first read its 180-degree counterpart. If that point-symmetric mate is also hidden by the center-spanning marker, recover the value from the visible 90/270-degree cells in the same C4 orbit; only the fully buried center 2x2 falls back to its copied top-left proxy.

    Key insight:
    The marker blocks the most direct point-symmetry witness exactly where it crosses the rotation center, but a true C4 pattern still leaves the same value visible one quarter-turn away.
    """
    g = np.asarray(grid, dtype=int)
    marker, (r0, c0, r1, c1) = _solid_rectangle_marker(g)
    lo = _find_rotated_strip_start(g, marker)
    n = int(g.shape[0])

    assert _covers_rotation_center(r0, c0, r1 - r0 + 1, c1 - c0 + 1, lo, n)
    assert r0 >= lo and c0 >= lo

    h = int(r1 - r0 + 1)
    w = int(c1 - c0 + 1)
    out = np.empty((h, w), dtype=int)

    for rr in range(h):
        for cc in range(w):
            out[rr, cc] = _recover_value(g, int(r0 + rr), int(c0 + cc), lo, marker)

    return out


# --------------- generator (o5x) ---------------

# --- CFG: canvas & structure ---
_N = 30
_LO = 2
_HI = _N - 1
_AXIS_LO, _AXIS_HI = _axis_pair(_LO, _N)

# --- CFG: colors ---
_PALETTE = np.array([1, 2, 3, 4, 5, 6, 7, 9], dtype=int)
_MARKER = 8

# --- CFG: marker rectangle size ---
_MARKER_DIM_LO = 3
_MARKER_DIM_HI = 9
_MARKER_AREA_LO = 16
_MARKER_AREA_HI = 36

# --- CFG: generation retries ---
_MAX_RETRIES = 300

_RNG = np.random.default_rng()
_SIZE_PAIRS = [
    (h, w)
    for h in range(_MARKER_DIM_LO, _MARKER_DIM_HI + 1)
    for w in range(_MARKER_DIM_LO, _MARKER_DIM_HI + 1)
    if _MARKER_AREA_LO <= h * w <= _MARKER_AREA_HI
]
_PLACEMENT_CACHE: dict[tuple[int, int], list[tuple[int, int, int, int, int]]] = {}


def _valid_placements(h: int, w: int) -> list[tuple[int, int, int, int, int]]:
    key = (h, w)
    if key not in _PLACEMENT_CACHE:
        placements: list[tuple[int, int, int, int, int]] = []
        for r0 in range(_LO, _N - h + 1):
            for c0 in range(_LO, _N - w + 1):
                if not _covers_rotation_center(r0, c0, h, w, _LO, _N):
                    continue
                stats = _recovery_mode_counts(r0, c0, h, w, _LO, _HI)
                if stats is None:
                    continue
                direct_180, via_c4, via_corner = stats
                if direct_180 > 0 and via_c4 > 0 and via_corner > 0:
                    placements.append((r0, c0, direct_180, via_c4, via_corner))
        _PLACEMENT_CACHE[key] = placements
    return _PLACEMENT_CACHE[key]


def _build_clean() -> np.ndarray:
    """Build a clean NxN grid with a C4-symmetric inner square."""
    half = (_N - _LO) // 2
    clean = _RNG.choice(_PALETTE, size=(_N, _N))

    seed = _RNG.choice(_PALETTE, size=(half, half))
    top = np.concatenate([seed, np.rot90(seed, -1)], axis=1)
    bottom = np.concatenate([np.rot90(seed, 1), np.rot90(seed, 2)], axis=1)
    clean[_LO:_N, _LO:_N] = np.concatenate([top, bottom], axis=0)
    # The exact central 2x2 orbit is always hidden by a center-spanning rectangle, so
    # expose one visible proxy for that orbit in the otherwise free corner.
    clean[:_LO, :_LO] = clean[_AXIS_LO : _AXIS_HI + 1, _AXIS_LO : _AXIS_HI + 1]
    return clean


def _visible_reflection_mismatches(clean: np.ndarray) -> int:
    block = clean[_LO:_N, _LO:_N]
    lr = int(np.count_nonzero(block != block[:, ::-1]))
    ud = int(np.count_nonzero(block != block[::-1, :]))
    return lr + ud


def _gen_attempt() -> np.ndarray | None:
    clean = _build_clean()
    if _visible_reflection_mismatches(clean) == 0:
        return None

    h, w = _SIZE_PAIRS[int(_RNG.integers(len(_SIZE_PAIRS)))]
    placements = _valid_placements(h, w)
    if not placements:
        return None
    r0, c0, direct_180, via_c4, via_corner = placements[int(_RNG.integers(len(placements)))]
    assert direct_180 > 0 and via_c4 > 0 and via_corner > 0

    expected = clean[r0 : r0 + h, c0 : c0 + w].copy()
    inp = clean.copy()
    inp[r0 : r0 + h, c0 : c0 + w] = _MARKER

    try:
        if _find_rotated_strip_start(inp, _MARKER) != _LO:
            return None
        if np.array_equal(main(inp), expected):
            return inp
    except Exception:
        return None
    return None


def generate_input() -> np.ndarray:
    for _ in range(_MAX_RETRIES):
        grid = _gen_attempt()
        if grid is not None:
            return grid
    raise RuntimeError("failed to generate a valid 0934a4d8 combo input")
