# Combo variant of e376de54 (index 19)
# combo_spec: D7/1+D11/2
# Name: 全同色+双像素宽

from __future__ import annotations

# Design analysis for combo D7/1 + D11/2
# =======================================
# a) Concrete behavior trace
#    - Original task: several 1-pixel-wide parallel segments, usually easy to separate because
#      colors differ. The solver groups by color, traces one thin run per color/offset, chooses
#      the median perpendicular offset as the reference, then aligns all starts to that reference
#      start and redraws every segment at the reference length.
#    - Fused variant:
#      1. D7/1 removes the color cue entirely: every segment uses the same foreground color.
#      2. D11/2 thickens each logical segment from one lane to a 2-pixel-wide ribbon.
#         Horizontal/vertical lines become 2-cell-thick rectangles. Diagonals become paired
#         adjacent diagonal lanes, rendered as a slanted 2-pixel-wide band.
#    - Concrete horizontal example:
#         original input:
#           row 2: 2 2 2 2 2
#           row 5: 8 8 8
#           row 7: 6 6 6 6
#         fused input with one common color 3 and width 2:
#           rows 2-3: 3 3 3 3 3
#           rows 5-6: 3 3 3
#           rows 7-8: 3 3 3 3
#      The transformation is still "align all ribbons to the reference span", but the solver can
#      no longer separate objects by color and can no longer treat one occupied row as one object.
#      It must recover each logical ribbon from geometry alone.
#
# b) Complexity calibration
#    - The design is not simplified back to the original task:
#      * all non-background cells share one color in every sample;
#      * each logical segment contributes two parallel lanes, so the old thin-segment extractor
#        would split each ribbon into two objects even if color were ignored.
#    - The design is also kept learnable:
#      * segment count stays 3 or 5, preserving a unique median reference;
#      * width is fixed at exactly 2, never mixed widths;
#      * the thickening rule is globally consistent for all four directions.
#    - To keep the grids legible, the generator preserves the original irregular offset spacing.
#      Adjacent ribbons may touch because width increased, but they are never allowed to share the
#      same lane, so the geometry is recoverable from repeated lane-pair structure.
#
# c) Implicit-assumption audit against the original main()
#    - Original `colors = np.unique(g) - {bg}` and per-color tracing assume one color identifies
#      one segment family. Broken: now all segments share one foreground color.
#    - Original "walk while same-color cells continue along d" assumes 1-pixel width. Broken:
#      under width 2, one logical segment yields two separate same-direction lane runs.
#    - Original `len(cells)` equals logical segment length. Broken for thick segments: each lane
#      has that length, but the object has twice as many pixels.
#    - Original clearing/redrawing loop is still conceptually valid after reconstruction, provided
#      the solver first merges each matched lane pair into one ribbon object.
#    - New solver assumption: for the true direction, every occupied cell belongs to exactly one
#      lane run, and every logical ribbon is exactly a pair of lane runs related by one fixed
#      width-offset vector. The generator enforces this by construction.
#
# d) Placement legality
#    - Generator remains output-first. It first places the aligned target ribbons, then derives the
#      input by anchoring all ribbons at a common start or end and varying only their lengths.
#    - Every candidate placement is checked with the exact thick-ribbon renderer, so no band exits
#      the board.
#    - Base offsets stay distinct. Different ribbons may touch after thickening, but they never
#      reuse the same lane, preventing ambiguous overpainting.
#
# e) Visual observability
#    - Human-visible cue for D7/1: all segments share one color, so identity must come from
#      position alone.
#    - Human-visible cue for D11/2: each segment is visibly a 2-pixel-wide band rather than a
#      1-pixel stroke; outputs preserve that thickness after alignment.
#    - Across 2-4 examples, the repeated pattern is still obvious: parallel bands on ordered
#      offsets all snap to the middle band's span. No hidden numeric color ordering is used.
#
# f) Visual cleanliness
#    - There is only one object family on the board: same-color parallel ribbons.
#    - No extra markers are added.
#    - The generator keeps the original small segment count and irregular spacing, so even when two
#      ribbons touch, the board still reads as a stack of parallel bars rather than noise.
#
# g) Post-change solver assumptions checked against the generator
#    - Exactly one non-background color is used -> enforced by `_choose_fg_color`.
#    - Every logical segment has width exactly 2 -> enforced by `_band_cells`.
#    - All logical segments share one direction -> enforced by choosing a single `d`.
#    - Segment count is odd and offsets are unique -> enforced by `_SEGMENT_COUNTS` and
#      `_choose_offsets`.
#    - For the true direction, each ribbon is recoverable as one lane pair with a fixed
#      `(delta_s, delta_t)` relation -> enforced by the fixed width-offset convention.

from pathlib import Path
import random
import sys

import numpy as np

_ROOT = Path(__file__).resolve().parents[4]
_SOLVER_DIR = _ROOT / "solvers"
if str(_SOLVER_DIR) not in sys.path:
    sys.path.append(str(_SOLVER_DIR))

from arc_utils import get_bg

_DIR_CANDIDATES = [(0, 1), (1, 0), (1, 1), (1, -1)]

# --------------- shared geometry ---------------

_H, _W = 16, 16
_BG = 7
_FG_COLORS = [1, 2, 3, 5, 6, 8, 9]

_SEGMENT_COUNTS = [3, 5]
_REF_LEN_RANGE = (3, 8)
_MAX_SEG_LEN = 12

_PERP_GAP_ORTHO = [2, 3, 4, 5, 6]
_PERP_GAP_DIAG = [4, 6]

_T_SCAN_RANGE = (-30, 31)
_GEN_RETRIES = 400


def _dot(p: tuple[int, int], d: tuple[int, int]) -> int:
    return p[0] * d[0] + p[1] * d[1]


def _perp_dot(p: tuple[int, int], d: tuple[int, int]) -> int:
    return p[0] * d[1] - p[1] * d[0]


def _step(d: tuple[int, int]) -> int:
    dr, dc = d
    return dr * dr + dc * dc


def _cell_from_ts(d: tuple[int, int], t: int, s: int) -> tuple[int, int] | None:
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
    if 0 <= r < _H and 0 <= c < _W:
        return r, c
    return None


def _width_offset(d: tuple[int, int]) -> tuple[int, int]:
    if d == (0, 1):
        return (1, 0)
    if d == (1, 0):
        return (0, 1)
    return (1, 0)


def _band_relation(d: tuple[int, int]) -> tuple[int, int]:
    w = _width_offset(d)
    return _perp_dot(w, d), _dot(w, d)


def _band_cells(
    d: tuple[int, int], t0: int, s_base: int, length: int
) -> list[tuple[int, int]] | None:
    delta = _step(d)
    wr, wc = _width_offset(d)
    cells: list[tuple[int, int]] = []
    for k in range(length):
        base = _cell_from_ts(d, t0 + k * delta, s_base)
        if base is None:
            return None
        r, c = base
        rr, cc = r + wr, c + wc
        if not (0 <= rr < _H and 0 <= cc < _W):
            return None
        cells.append((r, c))
        cells.append((rr, cc))
    return cells


def _choose_fg_color() -> int:
    return random.choice(_FG_COLORS)


# --------------- solver ---------------

def _extract_lane_runs(g: np.ndarray, bg: int, d: tuple[int, int]) -> list[dict]:
    delta = _step(d)
    buckets: dict[int, list[int]] = {}
    h, w = g.shape
    for r in range(h):
        for c in range(w):
            if int(g[r, c]) == bg:
                continue
            s = _perp_dot((r, c), d)
            t = _dot((r, c), d)
            buckets.setdefault(s, []).append(t)

    runs: list[dict] = []
    next_id = 0
    for s, t_vals in buckets.items():
        vals = sorted(set(t_vals))
        if not vals:
            continue
        start_idx = 0
        while start_idx < len(vals):
            end_idx = start_idx
            while end_idx + 1 < len(vals) and vals[end_idx + 1] - vals[end_idx] == delta:
                end_idx += 1
            run_vals = vals[start_idx : end_idx + 1]
            if len(run_vals) >= 2:
                cells = []
                ok = True
                for t in run_vals:
                    cell = _cell_from_ts(d, t, s)
                    if cell is None or int(g[cell]) == bg:
                        ok = False
                        break
                    cells.append(cell)
                if ok:
                    runs.append(
                        {
                            "id": next_id,
                            "s": s,
                            "t_start": run_vals[0],
                            "t_end": run_vals[-1],
                            "len": len(run_vals),
                            "cells": cells,
                            "start": cells[0],
                        }
                    )
                    next_id += 1
            start_idx = end_idx + 1
    return runs


def _segments_for_dir(g: np.ndarray, bg: int, d: tuple[int, int]) -> tuple[list[dict], int]:
    runs = _extract_lane_runs(g, bg, d)
    ds, dt = _band_relation(d)
    run_map = {(run["s"], run["t_start"], run["len"]): run for run in runs}

    segments: list[dict] = []
    used_run_ids: set[int] = set()
    covered_cells: set[tuple[int, int]] = set()

    ordered = sorted(runs, key=lambda run: (run["t_start"], run["s"]))
    for run in ordered:
        partner = run_map.get((run["s"] + ds, run["t_start"] + dt, run["len"]))
        if partner is None:
            continue
        if run["id"] in used_run_ids or partner["id"] in used_run_ids:
            continue
        if partner["t_end"] != run["t_end"] + dt:
            continue

        band_cells = _band_cells(d, run["t_start"], run["s"], run["len"])
        if band_cells is None:
            continue
        band_cell_set = set(band_cells)
        if band_cell_set != set(run["cells"]) | set(partner["cells"]):
            continue

        used_run_ids.add(run["id"])
        used_run_ids.add(partner["id"])
        covered_cells.update(band_cell_set)
        segments.append(
            {
                "cells": band_cells,
                "start": run["start"],
                "t_start": run["t_start"],
                "t_end": run["t_end"],
                "s": run["s"],
                "len": run["len"],
            }
        )

    return segments, len(covered_cells)


def _pick_main_dir(g: np.ndarray, bg: int) -> tuple[tuple[int, int], list[dict]]:
    non_bg_total = int(np.count_nonzero(g != bg))
    scored = []
    for d in _DIR_CANDIDATES:
        segments, covered = _segments_for_dir(g, bg, d)
        score = (covered, len(segments), sum(seg["len"] for seg in segments))
        scored.append((score, d, segments))

    best_score = max(score for score, _d, _segments in scored)
    best = [(d, segments) for score, d, segments in scored if score == best_score]
    assert best_score[0] > 0, "main direction must be detectable"
    assert len(best) == 1, "main direction must be unique"
    d, segments = best[0]
    assert best_score[0] == non_bg_total, "all non-background cells must belong to thick segments"
    return d, segments


def main(grid: np.ndarray) -> np.ndarray:
    """
    Align same-color thick parallel ribbons so they all match the reference ribbon's span.

    Input structure:
    - The background is the most common color.
    - Every non-background object belongs to one family of parallel ribbons that all share a
      single foreground color.
    - Each logical segment has width 2: horizontal/vertical segments are 2-cell-thick bars, and
      diagonal segments are 2-pixel-wide slanted bands made from two adjacent diagonal lanes.

    Rule:
    - Detect which of the four candidate directions explains all foreground cells as width-2
      ribbons.
    - Reconstruct each logical ribbon by pairing its two parallel lanes using only their geometric
      offset relation.
    - Use the ribbon at the median perpendicular offset as the reference.
    - Shift every ribbon so its start aligns with the reference start, then redraw it at exactly
      the reference length while keeping the same uniform color and width 2.

    Key insight:
    Color no longer separates objects, so the solver must infer each segment from repeated
    two-lane geometry: one logical ribbon is not one connected 1-pixel chain but a matched pair of
    parallel lanes whose combined band is then aligned to the median-offset reference.
    """
    g = np.asarray(grid, dtype=int)
    h, w = g.shape
    bg = get_bg(g)

    d, segments = _pick_main_dir(g, bg)
    dr, dc = d
    delta_t = _step(d)

    assert segments, "at least one thick segment must exist"
    s_vals = sorted(seg["s"] for seg in segments)
    if delta_t == 2:
        for i in range(len(s_vals) - 1):
            assert s_vals[i + 1] - s_vals[i] >= 4
    assert len(s_vals) % 2 == 1, "segment count must be odd"
    assert len(s_vals) == len(set(s_vals)), "base offsets must be unique"

    s_ref = s_vals[len(s_vals) // 2]
    ref_cands = [seg for seg in segments if seg["s"] == s_ref]
    assert len(ref_cands) == 1, "reference segment must be unique"
    ref = ref_cands[0]
    ref_t0 = ref["t_start"]
    target_len = ref["len"]

    fg_colors = [int(c) for c in np.unique(g) if int(c) != bg]
    assert len(fg_colors) == 1, "all thick segments must share one foreground color"
    fg = fg_colors[0]

    out = g.copy()
    for seg in segments:
        for r, c in seg["cells"]:
            out[r, c] = bg

    for seg in segments:
        dt = ref_t0 - seg["t_start"]
        assert dt % delta_t == 0, "segment shift must follow the direction lattice"
        k = dt // delta_t
        r0 = seg["start"][0] + k * dr
        c0 = seg["start"][1] + k * dc
        shifted_t0 = _dot((r0, c0), d)
        shifted_cells = _band_cells(d, shifted_t0, seg["s"], target_len)
        assert shifted_cells is not None, "shifted thick segment must stay in bounds"
        for r, c in shifted_cells:
            assert 0 <= r < h and 0 <= c < w
            out[r, c] = fg

    return out


# --------------- generator ---------------

def _max_len_from_band(d: tuple[int, int], t0: int, s: int) -> int:
    length = 0
    while _band_cells(d, t0, s, length + 1) is not None:
        length += 1
    return length


def _max_len_to_band(d: tuple[int, int], t1: int, s: int) -> int:
    delta = _step(d)
    length = 0
    while True:
        cand = length + 1
        t0 = t1 - (cand - 1) * delta
        if _band_cells(d, t0, s, cand) is None:
            return length
        length = cand


def _s_limits(d: tuple[int, int]) -> tuple[int, int]:
    if d == (0, 1):
        return 0, 15
    if d == (1, 0):
        return -15, 0
    if d == (1, 1):
        return -15, 15
    return -30, 0


def _gap_patterns(d: tuple[int, int], n: int) -> list[list[int]]:
    allowed = _PERP_GAP_DIAG if d in {(1, 1), (1, -1)} else _PERP_GAP_ORTHO
    lo, hi = _s_limits(d)
    max_span = hi - lo
    patterns = [[]]
    for _ in range(n - 1):
        patterns = [p + [gap] for p in patterns for gap in allowed]
    return [
        p
        for p in patterns
        if sum(p) <= max_span and ((n == 3 and p[0] != p[1]) or (n == 5 and len(set(p)) > 1))
    ]


def _choose_offsets(d: tuple[int, int], n: int) -> tuple[int, ...]:
    lo, hi = _s_limits(d)
    gaps = random.choice(_gap_patterns(d, n))
    s0 = random.randint(lo, hi - sum(gaps))
    s_vals = [s0]
    for gap in gaps:
        s_vals.append(s_vals[-1] + gap)
    return tuple(s_vals)


def _feasible_placements(d: tuple[int, int], s_vals: tuple[int, ...], ref_len: int) -> list[tuple[int, str, list[int]]]:
    delta = _step(d)
    ref_idx = len(s_vals) // 2
    placements: list[tuple[int, str, list[int]]] = []
    for t0 in range(*_T_SCAN_RANGE):
        if any(_band_cells(d, t0, s, ref_len) is None for s in s_vals):
            continue
        t1 = t0 + (ref_len - 1) * delta

        start_caps = [_max_len_from_band(d, t0, s) for s in s_vals]
        if any(start_caps[i] > ref_len for i in range(len(s_vals)) if i != ref_idx):
            placements.append((t0, "start", start_caps))

        end_caps = [_max_len_to_band(d, t1, s) for s in s_vals]
        if any(end_caps[i] > ref_len for i in range(len(s_vals)) if i != ref_idx):
            placements.append((t0, "end", end_caps))
    return placements


def _choose_lengths(ref_len: int, caps: list[int], n: int) -> list[int] | None:
    ref_idx = n // 2
    lengths = [ref_len] * n
    nonrefs = [i for i in range(n) if i != ref_idx]
    longable = [i for i in nonrefs if caps[i] > ref_len]
    if not longable or ref_len <= 2:
        return None

    long_i = random.choice(longable)
    short_i = random.choice([i for i in nonrefs if i != long_i])
    lengths[long_i] = random.randint(ref_len + 1, min(_MAX_SEG_LEN, caps[long_i]))
    lengths[short_i] = random.randint(2, ref_len - 1)

    for i in nonrefs:
        if i in (long_i, short_i):
            continue
        hi = min(_MAX_SEG_LEN, caps[i])
        if hi < 2:
            return None
        options = list(range(2, hi + 1))
        if len(options) > 1 and ref_len in options and random.random() < 0.75:
            options.remove(ref_len)
        lengths[i] = random.choice(options)
    return lengths


def _paint_bands(
    grid: np.ndarray,
    d: tuple[int, int],
    s_vals: tuple[int, ...],
    starts: list[int],
    lengths: list[int],
    fg: int,
) -> bool:
    for s, t0, length in zip(s_vals, starts, lengths):
        cells = _band_cells(d, t0, s, length)
        if cells is None:
            return False
        for r, c in cells:
            grid[r, c] = fg
    return True


def _gen_attempt() -> np.ndarray | None:
    d = random.choice(_DIR_CANDIDATES)
    n = random.choice(_SEGMENT_COUNTS)
    ref_len = random.randint(*_REF_LEN_RANGE)
    s_vals = _choose_offsets(d, n)
    placements = _feasible_placements(d, s_vals, ref_len)
    if not placements:
        return None

    t0, anchor, caps = random.choice(placements)
    lengths = _choose_lengths(ref_len, caps, n)
    if lengths is None:
        return None

    fg = _choose_fg_color()

    out = np.full((_H, _W), _BG, dtype=int)
    if not _paint_bands(out, d, s_vals, [t0] * n, [ref_len] * n, fg):
        return None

    delta = _step(d)
    t1 = t0 + (ref_len - 1) * delta
    if anchor == "start":
        starts = [t0] * n
    else:
        starts = [t1 - (length - 1) * delta for length in lengths]

    inp = np.full((_H, _W), _BG, dtype=int)
    if not _paint_bands(inp, d, s_vals, starts, lengths, fg):
        return None

    try:
        if np.array_equal(main(inp), out):
            return inp
    except Exception:
        return None
    return None


def generate_input() -> np.ndarray:
    for _ in range(_GEN_RETRIES):
        grid = _gen_attempt()
        if grid is not None:
            return grid
    raise RuntimeError("failed to generate valid input for combo D7/1+D11/2")
