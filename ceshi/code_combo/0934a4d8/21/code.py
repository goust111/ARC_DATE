# Combo variant of 0934a4d8 (index 21)
# combo_spec: D8/1+D13/0
# Name: 对称性破坏识别+两层嵌套恢复

from __future__ import annotations

from collections import deque
from pathlib import Path
import sys

import numpy as np

sys.path.append(str(Path(__file__).resolve().parents[4] / "solvers"))
from arc_utils import bbox


# Design analysis for the combo variant
# a) Concrete behavior and trace:
#    - The clean grid keeps the source task's LR+UD symmetry on the [lo..N-1] strip plus transpose fallback.
#    - The visible occluder is nested: an outer color-8 rectangular ring contains a solid inner color-0 rectangle.
#    - Solving is now iterative and both identification steps use symmetry-breaking, not the old filled-rectangle cue:
#      1. test each visible color as "ignore this and bilateral symmetry becomes valid";
#         only the outer ring color works on the raw input.
#      2. restore that outer ring while treating both marker colors as hidden.
#      3. repeat the same color-test on the partially restored grid; now only the inner color works.
#      4. restore the inner rectangle and output it.
#    - Small trace:
#      original: unique solid rectangle -> recover once -> output that rectangle.
#      combo: raw input has a visible inner solid rectangle, but its mismatches are completely shielded by the outer ring,
#      so the first symmetry-restoring color is the ring. After repairing the ring, the inner rectangle becomes the new
#      and only symmetry-breaking color, so the same "ignore one color and re-check symmetry" logic must be run again.
# b) Complexity calibration:
#    - The variant adds exactly one extra occlusion layer and one extra identification pass.
#    - No extra object families or hidden mappings are introduced; the difficulty increase comes from sequential reuse of
#      the same symmetry test under a new partially restored state.
# c) Implicit assumptions audit against the original main():
#    - "the marker is a unique solid rectangle" no longer holds globally: the first layer is a ring.
#    - "find lo after one marker is known" changes to "discover the active hidden color by trying each color as wildcard".
#    - "recover directly from the raw input" is invalid for the inner layer because some of its witnesses are still hidden
#      under the outer ring during stage 1.
#    - Any fallback read g[c,r] now depends on the generator guaranteeing that transpose witnesses for stage 1 lie outside
#      the whole outer bbox, and for stage 2 lie outside the inner bbox.
# d) Placement legality:
#    - The inner rectangle is strictly inside the outer bbox, and the outer visible color occupies exactly the surrounding ring.
#    - The outer bbox is only accepted when stage-1 fallback legality holds as if the whole outer bbox were hidden.
#    - The inner bbox is only accepted when stage-2 fallback legality holds, every direct LR/UD mirror mismatch is hidden by
#      the outer ring, and at least one inner cell still needs a witness from the restored ring in step 2.
# e) Visual observability:
#    - Across examples, the input visibly shows a nested two-color marker, while the output size matches the inner rectangle.
#    - The repeated pattern is observable: first the larger surrounding layer must be removed, then the smaller inner block.
#      The solver-side novelty is that the active layer is determined by which color explains the current symmetry break.
# f) Visual cleanliness:
#    - There is only one nested marker pair over the same symmetric background as the source task.
#    - Because the inner marker is fully contained inside one ring and no extra distractor objects are added, the scene stays
#      readable even though the reasoning is two-stage.
# g) Rule inferability:
#    - Reserved colors 8 and 0 denote marker layers; the clean pattern never uses them.
#    - No color arithmetic or ordering is used. The repeatable clue is structural: whichever visible color must be ignored to
#      make the grid symmetric is the current hidden layer.
# 9) New solver assumptions guaranteed by the generator:
#    - On the raw input, the unique symmetry-restoring color is the outer ring color 8.
#    - After stage-1 restoration, the unique symmetry-restoring color is the inner rectangle color 0.
#    - At least one inner cell requires a witness from the restored ring, so the task does not collapse to a single-step solve.


_OUTER_MARKER = 8
_INNER_MARKER = 0


def _in_rect(r: int, c: int, rect: tuple[int, int, int, int]) -> bool:
    r0, c0, r1, c1 = rect
    return r0 <= r <= r1 and c0 <= c <= c1


def _solid_rectangle_bbox(g: np.ndarray, color: int) -> tuple[int, int, int, int]:
    mask = g == int(color)
    b = bbox(mask)
    assert b is not None
    r0, c0, r1, c1 = b
    area = int((r1 - r0 + 1) * (c1 - c0 + 1))
    assert area > 1
    assert int(mask.sum()) == area
    assert bool(np.all(mask[r0 : r1 + 1, c0 : c1 + 1]))
    return int(r0), int(c0), int(r1), int(c1)


def _try_reflected_strip_start_mask(g: np.ndarray, hidden: np.ndarray) -> int | None:
    n, m = map(int, g.shape)
    assert n == m
    hi = n - 1

    for lo in range(n // 2 + 1):
        strip = g[:, lo : hi + 1]
        strip_hidden = hidden[:, lo : hi + 1]
        strip_reflected = strip[:, ::-1]
        strip_hidden_reflected = strip_hidden[:, ::-1]
        known = (~strip_hidden) & (~strip_hidden_reflected)
        if bool(np.any((strip != strip_reflected) & known)):
            continue

        strip = g[lo : hi + 1, :]
        strip_hidden = hidden[lo : hi + 1, :]
        strip_reflected = strip[::-1, :]
        strip_hidden_reflected = strip_hidden[::-1, :]
        known = (~strip_hidden) & (~strip_hidden_reflected)
        if bool(np.any((strip != strip_reflected) & known)):
            continue

        return lo

    return None


def _try_reflected_strip_start(g: np.ndarray, hidden_colors: set[int]) -> int | None:
    return _try_reflected_strip_start_mask(g, np.isin(g, list(hidden_colors)))


def _symmetry_breaking_candidates(g: np.ndarray) -> list[tuple[int, int]]:
    candidates: list[tuple[int, int]] = []
    for color in np.unique(g):
        lo = _try_reflected_strip_start(g, {int(color)})
        if lo is not None:
            candidates.append((int(color), int(lo)))
    return candidates


def _unique_symmetry_breaking_color(g: np.ndarray) -> tuple[int, int]:
    candidates = _symmetry_breaking_candidates(g)
    assert len(candidates) == 1
    return candidates[0]


def _nested_markers_from_outer_color(
    g: np.ndarray, outer_color: int
) -> tuple[tuple[int, int, int, int], int, tuple[int, int, int, int]]:
    outer_mask = g == int(outer_color)
    outer = bbox(outer_mask)
    assert outer is not None
    or0, oc0, or1, oc1 = map(int, outer)

    view = g[or0 : or1 + 1, oc0 : oc1 + 1]
    inner_colors = [int(color) for color in np.unique(view) if int(color) != outer_color]
    assert len(inner_colors) == 1
    inner_color = inner_colors[0]
    inner = _solid_rectangle_bbox(g, inner_color)
    ir0, ic0, ir1, ic1 = inner

    assert or0 < ir0 <= ir1 < or1
    assert oc0 < ic0 <= ic1 < oc1

    expected = np.full(view.shape, outer_color, dtype=int)
    expected[ir0 - or0 : ir1 - or0 + 1, ic0 - oc0 : ic1 - oc0 + 1] = inner_color
    assert np.array_equal(view, expected)
    return (or0, oc0, or1, oc1), inner_color, inner


def _reflection_orbit_values(
    g: np.ndarray, r: int, c: int, lo: int, hidden_colors: set[int]
) -> set[int]:
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
        if value not in hidden_colors:
            values.add(value)

        if lo <= cc <= hi:
            q.append((rr, lo + hi - cc))
        if lo <= rr <= hi:
            q.append((lo + hi - rr, cc))

    return values


def _recover_cell(g: np.ndarray, r: int, c: int, lo: int, hidden_colors: set[int]) -> int:
    values = _reflection_orbit_values(g, r, c, lo, hidden_colors)
    if values:
        assert len(values) == 1
        return values.pop()

    assert r < lo or c < lo
    assert 0 <= c < g.shape[0] and 0 <= r < g.shape[1]
    value = int(g[c, r])
    assert value not in hidden_colors
    return value


def _restore_outer_layer(
    g: np.ndarray,
    outer: tuple[int, int, int, int],
    inner: tuple[int, int, int, int],
    lo: int,
    outer_color: int,
    inner_color: int,
) -> np.ndarray:
    stage1 = g.copy()
    or0, oc0, or1, oc1 = outer
    hidden = {int(outer_color), int(inner_color)}
    for r in range(or0, or1 + 1):
        for c in range(oc0, oc1 + 1):
            if _in_rect(r, c, inner):
                continue
            stage1[r, c] = _recover_cell(g, r, c, lo, hidden)
    return stage1


def main(grid: np.ndarray) -> np.ndarray:
    """
    Recover a nested two-layer occlusion by repeatedly identifying the current symmetry-breaking color.

    Input structure:
    - A square colored grid follows the source task's left-right and up-down reflection law on a large lower/right strip,
      with the same transpose fallback near the top or left border.
    - The visible marker is nested: an outer rectangular ring of one reserved color surrounds a smaller solid rectangle of
      another reserved color.
    - In the raw input, the outer ring hides the visible LR/UD witnesses of the inner rectangle, so only the outer color is
      an immediate explanation for the broken symmetry.

    Rule:
    - Test each visible color as the possible "ignore this color and symmetry is restored" hypothesis.
    - The unique successful color on the raw input is the outer layer; recover that ring while treating both marker colors as
      hidden.
    - Run the same symmetry test again on the partially restored grid; the unique successful color is now the inner layer.
      Recover the inner rectangle from reflections or transpose fallback and output it.

    Key insight:
    The task reuses the same symmetry-detection mechanism twice. The inner marker is already visible at the start, but its
    evidence is masked by the outer ring, so it does not become the active symmetry-breaking color until the first layer has
    been repaired.
    """
    g = np.asarray(grid, dtype=int)

    outer_color, lo1 = _unique_symmetry_breaking_color(g)
    outer, inner_color, inner = _nested_markers_from_outer_color(g, outer_color)
    stage1 = _restore_outer_layer(g, outer, inner, lo1, outer_color, inner_color)

    recovered_inner_color, lo2 = _unique_symmetry_breaking_color(stage1)
    assert recovered_inner_color == inner_color
    inner_rect = _solid_rectangle_bbox(stage1, recovered_inner_color)
    assert inner_rect == inner

    ir0, ic0, ir1, ic1 = inner
    h = int(ir1 - ir0 + 1)
    w = int(ic1 - ic0 + 1)
    out = np.empty((h, w), dtype=int)
    for rr in range(h):
        for cc in range(w):
            r = int(ir0 + rr)
            c = int(ic0 + cc)
            out[rr, cc] = _recover_cell(stage1, r, c, lo2, {int(inner_color)})

    return out


# --------------- generator (o5x) ---------------

# --- CFG: canvas & structure ---
_N = 30
_LO = 2
_HI = _N - 1

# --- CFG: colors ---
_PALETTE = np.array([1, 2, 3, 4, 5, 6, 7, 9], dtype=int)

# --- CFG: top-strip family ---
_TOP_INNER_W_LO = 4
_TOP_INNER_W_HI = 7
_TOP_INNER_C0_CHOICES = np.array([6, 7, 8, 9, 10], dtype=int)
_TOP_OUTER_R1_CHOICES = np.array([5, 6, 7, 8], dtype=int)

# --- CFG: left-strip family ---
_LEFT_INNER_H_LO = 4
_LEFT_INNER_H_HI = 7
_LEFT_INNER_R0_CHOICES = np.array([6, 7, 8, 9, 10], dtype=int)
_LEFT_OUTER_C1_CHOICES = np.array([5, 6, 7, 8], dtype=int)

# --- CFG: generation retries ---
_MAX_RETRIES = 200

_RNG = np.random.default_rng()


def _orbit(r: int, c: int) -> set[tuple[int, int]]:
    rows = {r}
    cols = {c}
    if _LO <= r <= _HI:
        rows.add(_LO + _HI - r)
    if _LO <= c <= _HI:
        cols.add(_LO + _HI - c)
    return {(rr, cc) for rr in rows for cc in cols}


def _fallback_cells(r0: int, c0: int, h: int, w: int) -> list[tuple[int, int]] | None:
    r1, c1 = r0 + h, c0 + w
    cells: list[tuple[int, int]] = []
    for r in range(r0, r1):
        for c in range(c0, c1):
            if all(r0 <= rr < r1 and c0 <= cc < c1 for rr, cc in _orbit(r, c)):
                if r >= _LO and c >= _LO:
                    return None
                if r0 <= c < r1 and c0 <= r < c1:
                    return None
                cells.append((r, c))
    return cells


def _dependent_inner_cells(
    outer_r0: int, outer_c0: int, outer_h: int, outer_w: int, inner_r0: int, inner_c0: int, inner_h: int, inner_w: int
) -> list[tuple[int, int]]:
    outer = (outer_r0, outer_c0, outer_r0 + outer_h - 1, outer_c0 + outer_w - 1)
    inner = (inner_r0, inner_c0, inner_r0 + inner_h - 1, inner_c0 + inner_w - 1)
    dependent: list[tuple[int, int]] = []
    for r in range(inner_r0, inner_r0 + inner_h):
        for c in range(inner_c0, inner_c0 + inner_w):
            outside_inner = [(rr, cc) for rr, cc in _orbit(r, c) if not _in_rect(rr, cc, inner)]
            if outside_inner and all(_in_rect(rr, cc, outer) for rr, cc in outside_inner):
                dependent.append((r, c))
    return dependent


def _inner_symmetry_shielded(
    outer_r0: int, outer_c0: int, outer_h: int, outer_w: int, inner_r0: int, inner_c0: int, inner_h: int, inner_w: int
) -> bool:
    outer = (outer_r0, outer_c0, outer_r0 + outer_h - 1, outer_c0 + outer_w - 1)
    inner = (inner_r0, inner_c0, inner_r0 + inner_h - 1, inner_c0 + inner_w - 1)
    for r in range(inner_r0, inner_r0 + inner_h):
        for c in range(inner_c0, inner_c0 + inner_w):
            if c >= _LO:
                mate = (r, _LO + _HI - c)
                if not _in_rect(mate[0], mate[1], inner) and not _in_rect(mate[0], mate[1], outer):
                    return False
            if r >= _LO:
                mate = (_LO + _HI - r, c)
                if not _in_rect(mate[0], mate[1], inner) and not _in_rect(mate[0], mate[1], outer):
                    return False
    return True


def _unify_groups(clean: np.ndarray, groups: list[set[tuple[int, int]]]) -> None:
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
        for coord in coords:
            parent.setdefault(coord, coord)
        for coord in coords[1:]:
            union(coords[0], coord)

    merged: dict[tuple[int, int], list[tuple[int, int]]] = {}
    for coord in parent:
        merged.setdefault(find(coord), []).append(coord)

    for coords in merged.values():
        color = int(_RNG.choice(_PALETTE))
        for r, c in coords:
            clean[r, c] = color


def _build_clean() -> np.ndarray:
    half = (_N - _LO) // 2
    clean = np.empty((_N, _N), dtype=int)

    q = _RNG.choice(_PALETTE, size=(half, half))
    top = np.concatenate([q, q[:, ::-1]], axis=1)
    clean[_LO:_N, _LO:_N] = np.concatenate([top, top[::-1, :]], axis=0)

    for r in range(_LO):
        vals = _RNG.choice(_PALETTE, size=half)
        clean[r, _LO : _LO + half] = vals
        clean[r, _LO + half : _N] = vals[::-1]

    for c in range(_LO):
        vals = _RNG.choice(_PALETTE, size=half)
        clean[_LO : _LO + half, c] = vals
        clean[_LO + half : _N, c] = vals[::-1]

    clean[:_LO, :_LO] = _RNG.choice(_PALETTE, size=(_LO, _LO))
    return clean


def _sample_top_layout() -> tuple[tuple[int, int, int, int], tuple[int, int, int, int], list[tuple[int, int]], list[tuple[int, int]]] | None:
    inner_w = int(_RNG.integers(_TOP_INNER_W_LO, _TOP_INNER_W_HI + 1))
    inner_r0 = 1
    inner_c0 = int(_RNG.choice(_TOP_INNER_C0_CHOICES))
    inner_r1 = inner_r0
    inner_c1 = inner_c0 + inner_w - 1
    if inner_c1 >= _N:
        return None

    mirror_c0 = _LO + _HI - inner_c1
    mirror_c1 = _LO + _HI - inner_c0
    outer = (
        0,
        min(inner_c0, mirror_c0) - 1,
        int(_RNG.choice(_TOP_OUTER_R1_CHOICES)),
        max(inner_c1, mirror_c1) + 1,
    )
    outer_r0, outer_c0, outer_r1, outer_c1 = outer
    if outer_c0 < 0 or outer_c1 >= _N:
        return None

    inner = (inner_r0, inner_c0, inner_r1, inner_c1)
    outer_fallback = _fallback_cells(outer_r0, outer_c0, outer_r1 - outer_r0 + 1, outer_c1 - outer_c0 + 1)
    inner_fallback = _fallback_cells(inner_r0, inner_c0, inner_r1 - inner_r0 + 1, inner_c1 - inner_c0 + 1)
    if outer_fallback is None or inner_fallback is None:
        return None
    if not _inner_symmetry_shielded(
        outer_r0, outer_c0, outer_r1 - outer_r0 + 1, outer_c1 - outer_c0 + 1, inner_r0, inner_c0, 1, inner_w
    ):
        return None
    return outer, inner, outer_fallback, inner_fallback


def _sample_left_layout() -> tuple[tuple[int, int, int, int], tuple[int, int, int, int], list[tuple[int, int]], list[tuple[int, int]]] | None:
    inner_h = int(_RNG.integers(_LEFT_INNER_H_LO, _LEFT_INNER_H_HI + 1))
    inner_r0 = int(_RNG.choice(_LEFT_INNER_R0_CHOICES))
    inner_c0 = 1
    inner_r1 = inner_r0 + inner_h - 1
    inner_c1 = inner_c0
    if inner_r1 >= _N:
        return None

    mirror_r0 = _LO + _HI - inner_r1
    mirror_r1 = _LO + _HI - inner_r0
    outer = (
        min(inner_r0, mirror_r0) - 1,
        0,
        max(inner_r1, mirror_r1) + 1,
        int(_RNG.choice(_LEFT_OUTER_C1_CHOICES)),
    )
    outer_r0, outer_c0, outer_r1, outer_c1 = outer
    if outer_r0 < 0 or outer_r1 >= _N:
        return None

    inner = (inner_r0, inner_c0, inner_r1, inner_c1)
    outer_fallback = _fallback_cells(outer_r0, outer_c0, outer_r1 - outer_r0 + 1, outer_c1 - outer_c0 + 1)
    inner_fallback = _fallback_cells(inner_r0, inner_c0, inner_r1 - inner_r0 + 1, inner_c1 - inner_c0 + 1)
    if outer_fallback is None or inner_fallback is None:
        return None
    if not _inner_symmetry_shielded(
        outer_r0, outer_c0, outer_r1 - outer_r0 + 1, outer_c1 - outer_c0 + 1, inner_r0, inner_c0, inner_h, 1
    ):
        return None
    return outer, inner, outer_fallback, inner_fallback


def _sample_layout() -> tuple[tuple[int, int, int, int], tuple[int, int, int, int], list[tuple[int, int]], list[tuple[int, int]]] | None:
    if int(_RNG.integers(2)) == 0:
        return _sample_top_layout()
    return _sample_left_layout()


def _variant_metrics(g: np.ndarray) -> dict[str, int]:
    raw_candidates = _symmetry_breaking_candidates(g)
    raw_is_outer = int(len(raw_candidates) == 1 and raw_candidates[0][0] == _OUTER_MARKER)
    metrics = {
        "raw_candidate_count": len(raw_candidates),
        "raw_is_outer": raw_is_outer,
        "stage2_candidate_count": -1,
        "stage2_is_inner": 0,
        "dependent_count": -1,
    }
    if not raw_is_outer:
        return metrics

    _outer_color, lo1 = raw_candidates[0]
    try:
        outer, inner_color, inner = _nested_markers_from_outer_color(g, _outer_color)
    except AssertionError:
        return metrics
    stage1 = _restore_outer_layer(g, outer, inner, lo1, _OUTER_MARKER, inner_color)
    stage2_candidates = _symmetry_breaking_candidates(stage1)
    metrics["stage2_candidate_count"] = len(stage2_candidates)
    metrics["stage2_is_inner"] = int(
        len(stage2_candidates) == 1 and stage2_candidates[0][0] == _INNER_MARKER
    )
    or0, oc0, or1, oc1 = outer
    ir0, ic0, ir1, ic1 = inner
    dependent = _dependent_inner_cells(or0, oc0, or1 - or0 + 1, oc1 - oc0 + 1, ir0, ic0, ir1 - ir0 + 1, ic1 - ic0 + 1)
    metrics["dependent_count"] = len(dependent)
    return metrics


def _gen_attempt() -> np.ndarray | None:
    clean = _build_clean()

    layout = _sample_layout()
    if layout is None:
        return None
    outer, inner, outer_fallback, inner_fallback = layout
    outer_r0, outer_c0, outer_r1, outer_c1 = outer
    inner_r0, inner_c0, inner_r1, inner_c1 = inner
    outer_h = int(outer_r1 - outer_r0 + 1)
    outer_w = int(outer_c1 - outer_c0 + 1)
    inner_h = int(inner_r1 - inner_r0 + 1)
    inner_w = int(inner_c1 - inner_c0 + 1)
    dependent = _dependent_inner_cells(outer_r0, outer_c0, outer_h, outer_w, inner_r0, inner_c0, inner_h, inner_w)
    if not dependent:
        return None

    groups = [_orbit(r, c) | _orbit(c, r) for r, c in outer_fallback]
    groups += [_orbit(r, c) | _orbit(c, r) for r, c in inner_fallback]
    _unify_groups(clean, groups)

    expected = clean[inner_r0 : inner_r0 + inner_h, inner_c0 : inner_c0 + inner_w].copy()

    inp = clean.copy()
    inp[outer_r0 : outer_r0 + outer_h, outer_c0 : outer_c0 + outer_w] = _OUTER_MARKER
    inp[inner_r0 : inner_r0 + inner_h, inner_c0 : inner_c0 + inner_w] = _INNER_MARKER

    try:
        if not np.array_equal(main(inp), expected):
            return None
    except Exception:
        return None

    metrics = _variant_metrics(inp)
    if metrics["raw_candidate_count"] != 1 or metrics["raw_is_outer"] != 1:
        return None
    if metrics["stage2_candidate_count"] != 1 or metrics["stage2_is_inner"] != 1:
        return None
    if metrics["dependent_count"] <= 0 or not dependent:
        return None
    return inp


def generate_input() -> np.ndarray:
    for _ in range(_MAX_RETRIES):
        grid = _gen_attempt()
        if grid is not None:
            return grid
    raise RuntimeError("failed to generate a valid 0934a4d8 combo input")
