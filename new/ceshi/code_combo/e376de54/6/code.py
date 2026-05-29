# Combo variant of e376de54 (index 6)
# combo_spec: D3/2+D13/2
# Name: 中心对齐+居中排列

# Design analysis for the combo variant D3/2 + D13/2
# - Concrete object model:
#   Plain line segments are not enough here. If we only changed the input to a centered "diamond"
#   arrangement (D13/2) and kept bare segments, then the D3/2 output rule "match the reference by
#   center and resize symmetrically" would collapse to the same body footprint as the original
#   endpoint-based task, because the reference center and reference endpoints define the same final
#   interval. To make the center-based reconstruction visible, each odd-length segment body carries
#   one same-color nub attached one perpendicular step away from one endpoint.
# - Behavior trace:
#   Example: three horizontal bodies of lengths 3, 5, 9 share one center column, so the input body
#   silhouette is a diamond. Their nubs all sit off the positive endpoint on the same side. The
#   median-offset body (length 5) is the reference. In the fused rule, every object keeps its line
#   offset, is rebuilt around the reference center, and is cropped/extended symmetrically to length 5;
#   the nub is then redrawn at that rebuilt endpoint. Compared with the original task, the body
#   extent still becomes the reference extent, but the nub movement proves the solver is reconstructing
#   the full centered object rather than leaving endpoint-side details behind.
# - Complexity calibration:
#   Lengths are restricted to odd values so every body has one exact center cell. The nub has only one
#   form: same color, one cell perpendicular from an endpoint, with one common endpoint-side choice
#   per puzzle. That keeps the visual vocabulary small while still making the centered geometry
#   readable from 2-4 examples.
# - Hidden assumptions audited against the original solver:
#   1. The original solver assumed every relevant foreground cell lies on a straight run; that fails
#      because nubs are single pixels that belong to the object and must be cleared/redrawn.
#   2. The original generator assumed all segments share a start or end. D13/2 breaks that: inputs now
#      share one center instead, so starts and ends fan out symmetrically.
#   3. The original draw step aligned starts. In this combo the body must be reconstructed from
#      (reference_center, reference_length), and the nub must be reattached to the rebuilt endpoint.
# - Placement legality:
#   The generator is output-first. It first places the rectangular target state, then derives a
#   centered diamond input by varying odd body lengths around the same center. Perpendicular gaps keep
#   bodies and nubs disjoint, all cells are checked for bounds, and each sample is validated by
#   `main(input) == output`.
# - Visual observability and cleanliness:
#   Every sample forces at least one shorter and one longer non-reference body, so the input visibly
#   contains both inward and outward ragged endpoints. Because the nub is attached to the active
#   endpoint, at least one nub moves inward and one moves outward in the output; this is the key
#   observable signal that the fused rule is active rather than a plain restatement of the original.
# - Quantified non-degeneracy:
#   The generator rejects samples unless a legacy body-only rewrite (which ignores nubs and only snaps
#   the straight runs to the reference extent) differs from the fused output. Operationally this means
#   at least one nub must move.
# - Color reasoning:
#   Colors are preserved; they only label object identity. No ordering or rule branch depends on the
#   numeric color ids.

from __future__ import annotations

import random
import sys

import numpy as np

_SOLVER_UTILS = "/home/lixinyu/snap/new/solvers"
if _SOLVER_UTILS not in sys.path:
    sys.path.append(_SOLVER_UTILS)

from arc_utils import get_bg

_DIR_CANDIDATES = [(0, 1), (1, 0), (1, 1), (1, -1)]  # horiz, vert, diag \, diag /


def _pick_main_dir(g: np.ndarray, bg: int) -> tuple[int, int]:
    counts = []
    h, w = g.shape
    for dr, dc in _DIR_CANDIDATES:
        cnt = 0
        for r in range(h):
            for c in range(w):
                v = int(g[r, c])
                if v == bg:
                    continue
                rr, cc = r + dr, c + dc
                if 0 <= rr < h and 0 <= cc < w and int(g[rr, cc]) == v:
                    cnt += 1
        counts.append(((dr, dc), cnt))

    best_cnt = max(cnt for _, cnt in counts)
    assert best_cnt > 0, "main direction must be detectable"
    best_dirs = [d for d, cnt in counts if cnt == best_cnt]
    assert len(best_dirs) == 1, "main direction must be unique"
    return best_dirs[0]


def _dot(p: tuple[int, int], d: tuple[int, int]) -> int:
    return p[0] * d[0] + p[1] * d[1]


def _perp_dot(p: tuple[int, int], d: tuple[int, int]) -> int:
    return p[0] * d[1] - p[1] * d[0]


def _step(d: tuple[int, int]) -> int:
    dr, dc = d
    return dr * dr + dc * dc


def _cell_from_ts(d: tuple[int, int], t: int, s: int, h: int, w: int) -> tuple[int, int] | None:
    if d == (0, 1):
        r, c = s, t
    elif d == (1, 0):
        r, c = t, -s
    elif d == (1, 1):
        if (t + s) % 2 or (t - s) % 2:
            return None
        r, c = (t + s) // 2, (t - s) // 2
    else:
        if (t - s) % 2 or (-s - t) % 2:
            return None
        r, c = (t - s) // 2, (-s - t) // 2
    if 0 <= r < h and 0 <= c < w:
        return r, c
    return None


def _segment_cells(
    d: tuple[int, int], t0: int, s: int, length: int, h: int, w: int
) -> list[tuple[int, int]] | None:
    delta = _step(d)
    cells = []
    for k in range(length):
        cell = _cell_from_ts(d, t0 + k * delta, s, h, w)
        if cell is None:
            return None
        cells.append(cell)
    return cells


def _centered_segment_cells(
    d: tuple[int, int], center_t: int, s: int, length: int, h: int, w: int
) -> list[tuple[int, int]] | None:
    assert length % 2 == 1, "centered variant uses odd lengths"
    half = length // 2
    return _segment_cells(d, center_t - half * _step(d), s, length, h, w)


def _nub_cell_for_object(
    d: tuple[int, int],
    center_t: int,
    s: int,
    length: int,
    anchor: str,
    side: int,
    h: int,
    w: int,
) -> tuple[int, int] | None:
    delta = _step(d)
    half = length // 2
    if anchor == "start":
        t_anchor = center_t - half * delta
    else:
        t_anchor = center_t + half * delta
    return _cell_from_ts(d, t_anchor, s + side * delta, h, w)


def _extract_segments(g: np.ndarray, bg: int, d: tuple[int, int]) -> list[dict]:
    h, w = g.shape
    dr, dc = d
    segments = []
    colors = [int(c) for c in np.unique(g) if int(c) != bg]
    perp_steps = ((1, (dc, -dr)), (-1, (-dc, dr)))

    for col in colors:
        mask = g == col
        for r in range(h):
            for c in range(w):
                if not mask[r, c]:
                    continue
                pr, pc = r - dr, c - dc
                if 0 <= pr < h and 0 <= pc < w and mask[pr, pc]:
                    continue
                cells = []
                rr, cc = r, c
                while 0 <= rr < h and 0 <= cc < w and mask[rr, cc]:
                    cells.append((rr, cc))
                    rr += dr
                    cc += dc
                if len(cells) < 2:
                    continue
                assert len(cells) % 2 == 1, "each segment body must have odd length"
                start = cells[0]
                end = cells[-1]
                nub_cands = []
                for anchor_name, anchor_cell in (("start", start), ("end", end)):
                    for side, (qr, qc) in perp_steps:
                        nr, nc = anchor_cell[0] + qr, anchor_cell[1] + qc
                        if 0 <= nr < h and 0 <= nc < w and mask[nr, nc]:
                            nub_cands.append((anchor_name, side, (nr, nc)))
                assert len(nub_cands) == 1, "each segment must have exactly one endpoint nub"
                center = cells[len(cells) // 2]
                segments.append(
                    {
                        "color": col,
                        "cells": cells,
                        "nub": nub_cands[0][2],
                        "nub_anchor": nub_cands[0][0],
                        "nub_side": nub_cands[0][1],
                        "t_start": _dot(start, d),
                        "t_end": _dot(end, d),
                        "t_center": _dot(center, d),
                        "s": _perp_dot(start, d),
                        "len": len(cells),
                    }
                )
    return segments


def _legacy_like_output(g: np.ndarray, bg: int, d: tuple[int, int], segments: list[dict], ref: dict) -> np.ndarray:
    out = g.copy()
    target_len = ref["len"]
    ref_t0 = ref["t_start"]
    delta = _step(d)
    dr, dc = d

    for s in segments:
        for r, c in s["cells"]:
            out[r, c] = bg

    for s in segments:
        dt = ref_t0 - s["t_start"]
        assert dt % delta == 0
        k = dt // delta
        r0 = s["cells"][0][0] + k * dr
        c0 = s["cells"][0][1] + k * dc
        rr, cc = r0, c0
        for _ in range(target_len):
            assert 0 <= rr < g.shape[0] and 0 <= cc < g.shape[1]
            out[rr, cc] = s["color"]
            rr += dr
            cc += dc
    return out


def main(grid: np.ndarray) -> np.ndarray:
    """
    Rebuild centered parallel segment objects so they all match the reference object's centered extent.

    Input structure:
    - The background is the most frequent color.
    - Several odd-length line segments run in one dominant direction on distinct parallel offsets.
    - Their centers are already aligned, so the bodies form a centered diamond-like stack, and each
      body carries one same-color nub attached perpendicular to one endpoint.

    Rule:
    - Detect the common direction and extract each segment body together with its endpoint nub.
    - Choose the reference as the segment on the median perpendicular offset.
    - Keep every segment on its original parallel offset, but reconstruct its body around the
      reference center and resize it symmetrically to the reference length; then reattach the nub to
      the same endpoint-side of the rebuilt body.
    - Leave unrelated isolated pixels untouched and keep the grid size unchanged.

    Key insight:
    The input no longer shares a start or end; it shares a center. The solver therefore cannot reuse
    start-alignment logic. It must recover each object's center-based geometry and redraw the whole
    object from the reference center and reference odd length.
    """
    g = np.asarray(grid, dtype=int)
    h, w = g.shape
    bg = get_bg(g)

    d = _pick_main_dir(g, bg)
    delta_t = _step(d)
    segments = _extract_segments(g, bg, d)
    assert segments, "at least one multi-cell segment must exist"

    s_vals = sorted(s["s"] for s in segments)
    if delta_t == 2:
        for i in range(len(s_vals) - 1):
            assert s_vals[i + 1] - s_vals[i] >= 4
    assert len(s_vals) % 2 == 1, "segment count must have a single median"
    assert len(s_vals) == len(set(s_vals)), "segment line coordinates must be unique"
    s_ref = s_vals[len(s_vals) // 2]
    ref_cands = [s for s in segments if s["s"] == s_ref]
    assert len(ref_cands) == 1, "median reference segment must be unique"
    ref = ref_cands[0]
    assert ref["len"] % 2 == 1, "reference length must be odd"
    target_len = ref["len"]
    target_center = ref["t_center"]

    out = g.copy()
    for s in segments:
        for r, c in s["cells"]:
            out[r, c] = bg
        nr, nc = s["nub"]
        out[nr, nc] = bg

    for s in segments:
        cells = _centered_segment_cells(d, target_center, s["s"], target_len, h, w)
        nub = _nub_cell_for_object(
            d, target_center, s["s"], target_len, s["nub_anchor"], s["nub_side"], h, w
        )
        assert cells is not None and nub is not None, "rebuilt centered object must stay in bounds"
        for r, c in cells:
            out[r, c] = s["color"]
        out[nub[0], nub[1]] = s["color"]

    return out


# --------------- generator (combo D3/2 + D13/2) ---------------

_H, _W = 16, 16
_BG = 7

_DIRS = [(0, 1), (1, 0), (1, 1), (1, -1)]
_COLORS = [1, 2, 3, 5, 6, 8, 9]

_SEGMENT_COUNTS = [3, 5]
_REF_LEN_OPTIONS = [5, 7]
_ODD_LENGTHS = [3, 5, 7, 9, 11]
_MAX_DISTINCT_COLORS = 3

_PERP_GAP_ORTHO = [2, 3, 4, 5, 6]
_PERP_GAP_DIAG = [4, 6]

_T_SCAN_RANGE = (-30, 31)
_GEN_RETRIES = 500
_SAME_LEN_AVOID_PROB = 0.8


def _s_limits(d: tuple[int, int]) -> tuple[int, int]:
    if d == (0, 1):
        return 0, 15
    if d == (1, 0):
        return -15, 0
    if d == (1, 1):
        return -15, 15
    return -30, 0


def _gap_patterns(d: tuple[int, int], n: int) -> list[list[int]]:
    allowed = _PERP_GAP_DIAG if d in [(1, 1), (1, -1)] else _PERP_GAP_ORTHO
    lo, hi = _s_limits(d)
    max_span = hi - lo
    patterns = [[]]
    for _ in range(n - 1):
        patterns = [p + [g] for p in patterns for g in allowed]
    return [
        p
        for p in patterns
        if sum(p) <= max_span and (n == 3 and p[0] != p[1] or n == 5 and len(set(p)) > 1)
    ]


def _choose_offsets(d: tuple[int, int], n: int) -> tuple[int, ...]:
    lo, hi = _s_limits(d)
    gaps = random.choice(_gap_patterns(d, n))
    s0 = random.randint(lo, hi - sum(gaps))
    s_vals = [s0]
    for gap in gaps:
        s_vals.append(s_vals[-1] + gap)
    return tuple(s_vals)


def _object_cells(
    d: tuple[int, int],
    center_t: int,
    s: int,
    length: int,
    nub_anchor: str,
    nub_side: int,
    h: int,
    w: int,
) -> list[tuple[int, int]] | None:
    body = _centered_segment_cells(d, center_t, s, length, h, w)
    nub = _nub_cell_for_object(d, center_t, s, length, nub_anchor, nub_side, h, w)
    if body is None or nub is None:
        return None
    return body + [nub]


def _max_centered_len(d: tuple[int, int], center_t: int, s: int, h: int, w: int) -> int:
    delta = _step(d)
    half = 0
    while True:
        next_half = half + 1
        left_ok = _cell_from_ts(d, center_t - next_half * delta, s, h, w) is not None
        right_ok = _cell_from_ts(d, center_t + next_half * delta, s, h, w) is not None
        if not (left_ok and right_ok):
            break
        half = next_half
    return 2 * half + 1


def _feasible_ref_centers(
    d: tuple[int, int], s_vals: tuple[int, ...], ref_len: int, nub_anchor: str, nub_side: int
) -> list[int]:
    centers = []
    for center_t in range(*_T_SCAN_RANGE):
        if all(
            _object_cells(d, center_t, s, ref_len, nub_anchor, nub_side, _H, _W) is not None for s in s_vals
        ):
            centers.append(center_t)
    return centers


def _choose_colors(s_vals: tuple[int, ...]) -> tuple[int, ...]:
    n_colors = random.randint(1, min(_MAX_DISTINCT_COLORS, len(s_vals)))
    palette = random.sample(_COLORS, n_colors)
    colors = [random.choice(palette) for _ in s_vals]
    while len(set(colors)) < n_colors:
        colors[random.randrange(len(colors))] = random.choice(palette)
    return tuple(colors)


def _choose_lengths(ref_len: int, caps: list[int], n: int) -> list[int] | None:
    ref_idx = n // 2
    lengths = [ref_len] * n
    nonrefs = [i for i in range(n) if i != ref_idx]
    longable = [i for i in nonrefs if caps[i] > ref_len]
    if not longable or ref_len <= 3:
        return None

    long_i = random.choice(longable)
    short_i = random.choice([i for i in nonrefs if i != long_i])
    long_options = [x for x in _ODD_LENGTHS if ref_len < x <= caps[long_i]]
    short_options = [x for x in _ODD_LENGTHS if 3 <= x < ref_len and x <= caps[short_i]]
    if not long_options or not short_options:
        return None

    lengths[long_i] = random.choice(long_options)
    lengths[short_i] = random.choice(short_options)

    for i in nonrefs:
        if i in (long_i, short_i):
            continue
        options = [x for x in _ODD_LENGTHS if 3 <= x <= caps[i]]
        if not options:
            return None
        if len(options) > 1 and ref_len in options and random.random() < _SAME_LEN_AVOID_PROB:
            alt = [x for x in options if x != ref_len]
            if alt:
                options = alt
        lengths[i] = random.choice(options)
    return lengths


def _paint(
    grid: np.ndarray,
    d: tuple[int, int],
    s_vals: tuple[int, ...],
    center_t: int,
    lengths: list[int],
    colors: tuple[int, ...],
    nub_anchor: str,
    nub_side: int,
) -> bool:
    used = set()
    objects = []
    for s, length, color in zip(s_vals, lengths, colors):
        coords = _object_cells(d, center_t, s, length, nub_anchor, nub_side, _H, _W)
        if coords is None:
            return False
        for cell in coords:
            if cell in used:
                return False
            used.add(cell)
        objects.append((coords, color))
    for coords, color in objects:
        for r, c in coords:
            grid[r, c] = color
    return True


def _fusion_metric(inp: np.ndarray, out: np.ndarray, d: tuple[int, int], ref_len: int) -> tuple[int, int]:
    bg = get_bg(inp)
    segments = _extract_segments(inp, bg, d)
    s_ref = sorted(s["s"] for s in segments)[len(segments) // 2]
    ref = [s for s in segments if s["s"] == s_ref][0]
    legacy = _legacy_like_output(inp, bg, d, segments, ref)
    moved_nubs = sum(1 for s in segments if s["len"] != ref_len)
    legacy_diff = int(np.count_nonzero(out != legacy))
    return moved_nubs, legacy_diff


def _gen_attempt() -> np.ndarray | None:
    d = random.choice(_DIRS)
    n = random.choice(_SEGMENT_COUNTS)
    ref_len = random.choice(_REF_LEN_OPTIONS)
    s_vals = _choose_offsets(d, n)
    nub_anchor = "end"
    nub_side = random.choice([-1, 1])

    ref_centers = _feasible_ref_centers(d, s_vals, ref_len, nub_anchor, nub_side)
    if not ref_centers:
        return None
    ref_center = random.choice(ref_centers)

    caps = [min(_ODD_LENGTHS[-1], _max_centered_len(d, ref_center, s, _H, _W)) for s in s_vals]
    if any(cap < ref_len for cap in caps):
        return None
    # Generator guarantees the extra assumptions introduced by the fused solver:
    # - all bodies are odd, so `t_center` is an actual cell;
    # - every body has exactly one nub, always attached to the same endpoint family in the sample;
    # - all input centers already equal the reference center, so the visible action is the symmetric
    #   resize itself plus the nub relocation to the rebuilt endpoint.
    lengths = _choose_lengths(ref_len, caps, n)
    if lengths is None:
        return None
    colors = _choose_colors(s_vals)

    out = np.full((_H, _W), _BG, dtype=int)
    if not _paint(out, d, s_vals, ref_center, [ref_len] * n, colors, nub_anchor, nub_side):
        return None

    inp = np.full((_H, _W), _BG, dtype=int)
    if not _paint(inp, d, s_vals, ref_center, lengths, colors, nub_anchor, nub_side):
        return None

    try:
        produced = main(inp)
    except Exception:
        return None
    if not np.array_equal(produced, out):
        return None

    moved_nubs, legacy_diff = _fusion_metric(inp, out, d, ref_len)
    if moved_nubs < 2 or legacy_diff == 0:
        return None
    return inp


def generate_input() -> np.ndarray:
    for _ in range(_GEN_RETRIES):
        grid = _gen_attempt()
        if grid is not None:
            return grid
    raise RuntimeError("failed to generate valid input")
