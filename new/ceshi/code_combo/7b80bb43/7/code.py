# Combo variant of 7b80bb43 (index 7)
# combo_spec: D8/0+D9/1
# Name: 独立锚定线段

from __future__ import annotations

from pathlib import Path
import random
import sys

import numpy as np

_SOLVER_DIR = Path("/home/lixinyu/snap/new/solvers")
if str(_SOLVER_DIR) not in sys.path:
    sys.path.append(str(_SOLVER_DIR))

from arc_utils import draw_line, get_bg


"""
design_analysis:
  a_specific_behavior:
    original:
      The seed task repairs short collinear gaps almost purely from local evidence:
      remove diagonal-only hint pixels, then reconnect any short row/column break
      that sits next to such a hint. T-junction topology and "one free end may come
      from the border" are generator assumptions, not solver-level validators.
    fused_variant:
      The clean output is now a set of mutually independent segments with no
      junctions at all. Every surviving segment must have both endpoints on the
      grid boundary. In practice this yields two visible segment families:
      1. full-span interior lines that touch opposite borders;
      2. partial runs that lie directly on one border row/column, so both ends are
         still edge pixels.
      A concrete small example:
        - True target row: `..########..` on row 0 is a legal border-hugging
          segment; if cells 4..6 are deleted and a 2-pixel diagonal hint is drawn
          inward, the row should be restored.
        - Floating decoy: `..##...##..` on row 5 with the same diagonal hint pattern
          looks locally repairable, but both repaired ends would stay in the
          interior, so it is not part of the output and must disappear.
      Difference:
        The local "short gap + nearby diagonal" cue is no longer sufficient. The
        solver must also validate that the repaired segment would be edge-anchored
        and non-intersecting. This makes D9/1 active instead of descriptive, while
        D8/0 removes all true crossings so the validation target is clean.
  b_complexity_calibration:
    - The output still uses one fg color and only axis-aligned 1-pixel-thick lines,
      so the visual language stays close to the seed task.
    - The no-junction change is made obvious by generating only isolated components.
      There are no T nodes to reason about anywhere.
    - To keep the puzzle from collapsing into a trivial "erase diagonals" task, the
      input also contains 1-2 floating fake breaks that mimic the same local hint
      pattern but fail the edge-anchoring test. Humans can see the repeated contrast:
      border/full-span lines survive, interior mini-lines do not.
    - Freedom is intentionally limited to two repeated anchor templates
      (full-span vs border-hugging), so 2-4 examples are enough to infer the rule.
  c_hidden_assumptions_audit:
    - `main()` in the seed assumes any short collinear break beside a diagonal hint
      is a true gap. This is broken by the fused variant because decoy floating
      fragments intentionally create false positives unless anchoring is checked.
    - The seed generator assumes T-specific blocked positions near junctions. Those
      assumptions are removed entirely because the fused output contains no junctions.
    - The seed pruning stage only trims leaf pixels near diagonal hints. That is not
      enough once floating straight fragments are allowed as decoys, so the new
      solver reconstructs the output from legal anchored templates and discards every
      other residue.
    - Independence matters globally: a legal repaired segment may not cross any
      already-kept segment. The generator therefore spaces components so the solver's
      overlap rejection is never ambiguous on valid inputs.
  d_legality:
    - Output-first generation places either:
        * several full-width interior horizontal lines plus border vertical segments,
          or
        * several full-height interior vertical lines plus border horizontal segments.
      These sets are mixed only when their spans are chosen to avoid every crossing.
    - True gaps are always 3 cells wide and stay at least 2 cells from segment ends.
      Diagonal hints extend into empty space and remain diagonal-only.
    - Floating decoys are placed strictly inside the grid, away from all true
      segments and away from borders, so they never accidentally satisfy the anchor
      template.
  e_visual_observability:
    - The output visibly contains only independent edge-anchored segments.
    - The input repeatedly shows two classes of broken objects with the same local
      diagonal clue: anchored ones are restored, floating ones vanish. This makes the
      fused rule readable by eye instead of relying on hidden metadata.
  f_visual_cleanliness:
    - Components are separated by at least one background moat.
    - The generator uses only a few long lines plus at most two decoys, avoiding a
      cluttered interior.
    - No extra colors or stacked overlays are introduced.
  g_color_inference:
    - No color mapping is involved. The task remains binary: identify fg structure,
      reason about geometry, and reconstruct only the legally anchored subset.
"""


GRID_H_RANGE = (16, 29)
GRID_W_RANGE = (17, 30)
N_COLORS = 10

GAP_SIZE = 3
CHAIN_LENGTH = 2
GAP_END_INSET = 2

TRUE_GAP_CHOICES = (2, 3)
DECOY_COUNT_CHOICES = (1, 2)

FULL_SPAN_GAP = 3
SIDE_INTERVAL_GAP = 2
FULL_COUNT_GAP = 3

FULL_LINE_COUNT_CHOICES = (2, 3)
TOTAL_SEGMENTS_BY_MODE = {
    "horizontal": (6, 7),
    "vertical": (6, 7),
}

BORDER_SEG_LEN_MIN = 4
BORDER_SEG_LEN_MAX = 8
DECOY_ARM_MIN = 2
DECOY_ARM_MAX = 3

GEN_RETRIES = 600


def main(grid: np.ndarray) -> np.ndarray:
    """
    Restore only the clean anchored segments from diagonal gap hints.

    Input structure:
    - One foreground color forms axis-aligned 1-pixel lines on a single background.
    - True objects are independent segments with both endpoints on the grid boundary:
      either full-span interior lines touching opposite sides, or partial runs drawn
      directly on a border row/column.
    - The input may also contain short floating broken stubs that imitate the same
      diagonal hint pattern but are not edge-anchored.

    Rule:
    - Remove every diagonal-only hint pixel.
    - Reconnect a short row/column gap only when the repaired result matches a legal
      anchored segment template and the removed diagonal hint sits next to that gap.
    - Keep the resulting non-intersecting anchored segments and discard all remaining
      floating fragments.

    Key insight:
    - The diagonal hint is only a local clue; the decisive test is global anchoring.
      If repairing a break would not place both segment ends on the border, that
      structure is a decoy and should vanish instead of being completed.
    """
    g = np.array(grid, dtype=int)
    bg = int(get_bg(g))
    colors = {int(x) for x in np.unique(g)}
    colors.discard(bg)
    assert colors, "expected one foreground color"
    fg = int(next(iter(colors)))

    mask = g == fg
    h, w = g.shape

    def deg4(m: np.ndarray, r: int, c: int) -> int:
        d = 0
        if r > 0 and bool(m[r - 1, c]):
            d += 1
        if r + 1 < h and bool(m[r + 1, c]):
            d += 1
        if c > 0 and bool(m[r, c - 1]):
            d += 1
        if c + 1 < w and bool(m[r, c + 1]):
            d += 1
        return d

    def has_diag(m: np.ndarray, r: int, c: int) -> bool:
        for dr in (-1, 1):
            for dc in (-1, 1):
                rr = r + dr
                cc = c + dc
                if 0 <= rr < h and 0 <= cc < w and bool(m[rr, cc]):
                    return True
        return False

    def row_hint_near(r: int, c0: int, c1: int, diag: np.ndarray) -> bool:
        rr0 = max(0, r - 1)
        rr1 = min(h - 1, r + 1)
        cc0 = max(0, c0 - 1)
        cc1 = min(w - 1, c1 + 1)
        return bool(diag[rr0 : rr1 + 1, cc0 : cc1 + 1].any())

    def col_hint_near(c: int, r0: int, r1: int, diag: np.ndarray) -> bool:
        rr0 = max(0, r0 - 1)
        rr1 = min(h - 1, r1 + 1)
        cc0 = max(0, c - 1)
        cc1 = min(w - 1, c + 1)
        return bool(diag[rr0 : rr1 + 1, cc0 : cc1 + 1].any())

    def runs_1d(vals: np.ndarray) -> list[tuple[int, int]]:
        if vals.size == 0:
            return []
        out: list[tuple[int, int]] = []
        start = prev = int(vals[0])
        for x in vals[1:]:
            x = int(x)
            if x == prev + 1:
                prev = x
            else:
                out.append((start, prev))
                start = prev = x
        out.append((start, prev))
        return out

    def merge_runs(
        runs: list[tuple[int, int]],
        gap_ok,
    ) -> list[tuple[int, int]]:
        if not runs:
            return []
        merged: list[tuple[int, int]] = []
        cur0, cur1 = runs[0]
        for nxt0, nxt1 in runs[1:]:
            gap0 = cur1 + 1
            gap1 = nxt0 - 1
            gap_len = gap1 - gap0 + 1
            if 1 <= gap_len <= GAP_SIZE and gap_ok(gap0, gap1):
                cur1 = nxt1
            else:
                merged.append((cur0, cur1))
                cur0, cur1 = nxt0, nxt1
        merged.append((cur0, cur1))
        return merged

    diag = np.zeros((h, w), dtype=bool)
    for r in range(h):
        for c in range(w):
            if bool(mask[r, c]) and deg4(mask, r, c) == 0 and has_diag(mask, r, c):
                diag[r, c] = True

    cleaned = mask & (~diag)
    out_mask = np.zeros_like(cleaned, dtype=bool)

    def place_row(r: int, c0: int, c1: int) -> None:
        if c1 < c0 or c1 - c0 + 1 < 2:
            return
        if bool(out_mask[r, c0 : c1 + 1].any()):
            return
        out_mask[r, c0 : c1 + 1] = True

    def place_col(c: int, r0: int, r1: int) -> None:
        if r1 < r0 or r1 - r0 + 1 < 2:
            return
        if bool(out_mask[r0 : r1 + 1, c].any()):
            return
        out_mask[r0 : r1 + 1, c] = True

    for r in range(h):
        cols = np.where(cleaned[r, :])[0]
        row_runs = runs_1d(cols)
        if not row_runs:
            continue
        merged = merge_runs(row_runs, lambda g0, g1, rr=r: row_hint_near(rr, g0, g1, diag))
        if r in (0, h - 1):
            for c0, c1 in merged:
                place_row(r, c0, c1)
        elif len(merged) == 1 and merged[0] == (0, w - 1):
            place_row(r, 0, w - 1)

    for c in range(w):
        rows = np.where(cleaned[:, c])[0]
        col_runs = runs_1d(rows)
        if not col_runs:
            continue
        merged = merge_runs(col_runs, lambda g0, g1, cc=c: col_hint_near(cc, g0, g1, diag))
        if c in (0, w - 1):
            for r0, r1 in merged:
                place_col(c, r0, r1)
        elif len(merged) == 1 and merged[0] == (0, h - 1):
            place_col(c, 0, h - 1)

    out = np.full((h, w), bg, dtype=int)
    out[out_mask] = fg
    return out


def _pick_distinct(rng: random.Random, values: list[int], k: int, gap: int) -> list[int]:
    vals = values[:]
    rng.shuffle(vals)
    picked: list[int] = []
    for v in vals:
        if all(abs(v - p) >= gap for p in picked):
            picked.append(v)
            if len(picked) == k:
                return sorted(picked)
    return []


def _sample_intervals(
    rng: random.Random,
    size: int,
    count: int,
    forbidden_points: set[int],
) -> list[tuple[int, int]]:
    intervals: list[tuple[int, int]] = []
    for _ in range(400):
        if len(intervals) == count:
            break
        length = rng.randint(BORDER_SEG_LEN_MIN, BORDER_SEG_LEN_MAX)
        if size - length - 2 <= 1:
            return []
        start = rng.randint(1, size - length - 2)
        end = start + length - 1
        if any(start <= p <= end for p in forbidden_points):
            continue
        if any(not (end + SIDE_INTERVAL_GAP < a or b + SIDE_INTERVAL_GAP < start) for a, b in intervals):
            continue
        intervals.append((start, end))
    if len(intervals) != count:
        return []
    return sorted(intervals)


def _reserve_neighborhood(cells: set[tuple[int, int]], coords: list[tuple[int, int]]) -> None:
    for r, c in coords:
        for dr in (-1, 0, 1):
            for dc in (-1, 0, 1):
                cells.add((r + dr, c + dc))


def _cells_are_color(grid: np.ndarray, coords: list[tuple[int, int]], color: int) -> bool:
    h, w = grid.shape
    for r, c in coords:
        if not (0 <= r < h and 0 <= c < w):
            return False
        if int(grid[r, c]) != int(color):
            return False
    return True


def _validate_chain(
    grid: np.ndarray,
    coords: list[tuple[int, int]],
    fg: int,
    occupied: set[tuple[int, int]],
) -> bool:
    h, w = grid.shape
    coord_set = set(coords)
    if any(cell in occupied for cell in coords):
        return False
    for r, c in coords:
        for rr, cc in ((r - 1, c), (r + 1, c), (r, c - 1), (r, c + 1)):
            if (rr, cc) in coord_set:
                return False
            if 0 <= rr < h and 0 <= cc < w and int(grid[rr, cc]) == int(fg):
                return False
        ok = False
        for rr, cc in ((r - 1, c - 1), (r - 1, c + 1), (r + 1, c - 1), (r + 1, c + 1)):
            if (rr, cc) in coord_set:
                ok = True
                break
            if 0 <= rr < h and 0 <= cc < w and int(grid[rr, cc]) == int(fg):
                ok = True
                break
        if not ok:
            return False
    return True


def _h_chain_coords(row: int, gap_start: int, anchor: str, dr: int) -> list[tuple[int, int]]:
    if anchor == "left":
        return [(row + dr * i, gap_start - 1 + i) for i in range(1, CHAIN_LENGTH + 1)]
    return [(row + dr * i, gap_start + GAP_SIZE - i) for i in range(1, CHAIN_LENGTH + 1)]


def _v_chain_coords(col: int, gap_start: int, anchor: str, dc: int) -> list[tuple[int, int]]:
    if anchor == "top":
        return [(gap_start + i, col + dc * (i + 1)) for i in range(CHAIN_LENGTH)]
    return [(gap_start + GAP_SIZE - 1 - i, col + dc * (i + 1)) for i in range(CHAIN_LENGTH)]


def _segment_cells(seg: dict) -> list[tuple[int, int]]:
    if seg["orient"] == "h":
        return [(seg["row"], c) for c in range(seg["c0"], seg["c1"] + 1)]
    return [(r, seg["col"]) for r in range(seg["r0"], seg["r1"] + 1)]


def _h_gap_options(
    seg: dict,
    grid: np.ndarray,
    fg: int,
    bg: int,
    occupied: set[tuple[int, int]],
) -> list[tuple[int, str, int]]:
    row, c0, c1 = seg["row"], seg["c0"], seg["c1"]
    drs = seg["drs"]
    opts: list[tuple[int, str, int]] = []
    for g in range(c0 + GAP_END_INSET, c1 - GAP_SIZE):
        temp = grid.copy()
        if not np.all(temp[row, g : g + GAP_SIZE] == fg):
            continue
        temp[row, g : g + GAP_SIZE] = bg
        for anchor in ("left", "right"):
            for dr in drs:
                coords = _h_chain_coords(row, g, anchor, dr)
                if not _cells_are_color(temp, coords, bg):
                    continue
                if _validate_chain(temp, coords, fg, occupied):
                    opts.append((g, anchor, dr))
    return opts


def _v_gap_options(
    seg: dict,
    grid: np.ndarray,
    fg: int,
    bg: int,
    occupied: set[tuple[int, int]],
) -> list[tuple[int, str, int]]:
    col, r0, r1 = seg["col"], seg["r0"], seg["r1"]
    dcs = seg["dcs"]
    opts: list[tuple[int, str, int]] = []
    for g in range(r0 + GAP_END_INSET, r1 - GAP_SIZE):
        temp = grid.copy()
        if not np.all(temp[g : g + GAP_SIZE, col] == fg):
            continue
        temp[g : g + GAP_SIZE, col] = bg
        for anchor in ("top", "bottom"):
            for dc in dcs:
                coords = _v_chain_coords(col, g, anchor, dc)
                if not _cells_are_color(temp, coords, bg):
                    continue
                if _validate_chain(temp, coords, fg, occupied):
                    opts.append((g, anchor, dc))
    return opts


def _decoy_options(
    grid: np.ndarray,
    fg: int,
    bg: int,
    forbidden: set[tuple[int, int]],
) -> list[dict]:
    h, w = grid.shape
    opts: list[dict] = []

    for r in range(2, h - 2):
        for left_len in range(DECOY_ARM_MIN, DECOY_ARM_MAX + 1):
            for right_len in range(DECOY_ARM_MIN, DECOY_ARM_MAX + 1):
                total = left_len + GAP_SIZE + right_len
                for c0 in range(2, w - total - 1):
                    c1 = c0 + total - 1
                    left = [(r, c) for c in range(c0, c0 + left_len)]
                    right = [(r, c) for c in range(c0 + left_len + GAP_SIZE, c1 + 1)]
                    body = left + right
                    if any(cell in forbidden for cell in body):
                        continue
                    temp = grid.copy()
                    for rr, cc in body:
                        temp[rr, cc] = fg
                    gap_start = c0 + left_len
                    for dr in (-1, 1):
                        for anchor in ("left", "right"):
                            chain = _h_chain_coords(r, gap_start, anchor, dr)
                            if any(cell in forbidden for cell in chain):
                                continue
                            if not _cells_are_color(temp, chain, bg):
                                continue
                            if _validate_chain(temp, chain, fg, set()):
                                opts.append(
                                    {
                                        "cells": body,
                                        "chain": chain,
                                    }
                                )

    for c in range(2, w - 2):
        for top_len in range(DECOY_ARM_MIN, DECOY_ARM_MAX + 1):
            for bottom_len in range(DECOY_ARM_MIN, DECOY_ARM_MAX + 1):
                total = top_len + GAP_SIZE + bottom_len
                for r0 in range(2, h - total - 1):
                    r1 = r0 + total - 1
                    top = [(r, c) for r in range(r0, r0 + top_len)]
                    bottom = [(r, c) for r in range(r0 + top_len + GAP_SIZE, r1 + 1)]
                    body = top + bottom
                    if any(cell in forbidden for cell in body):
                        continue
                    temp = grid.copy()
                    for rr, cc in body:
                        temp[rr, cc] = fg
                    gap_start = r0 + top_len
                    for dc in (-1, 1):
                        for anchor in ("top", "bottom"):
                            chain = _v_chain_coords(c, gap_start, anchor, dc)
                            if any(cell in forbidden for cell in chain):
                                continue
                            if not _cells_are_color(temp, chain, bg):
                                continue
                            if _validate_chain(temp, chain, fg, set()):
                                opts.append(
                                    {
                                        "cells": body,
                                        "chain": chain,
                                    }
                                )
    return opts


def _build_output(rng: random.Random, h: int, w: int, bg: int, fg: int) -> tuple[np.ndarray, list[dict]] | None:
    out = np.full((h, w), bg, dtype=int)
    segments: list[dict] = []
    mode = rng.choice(["horizontal", "vertical"])
    total = rng.choice(TOTAL_SEGMENTS_BY_MODE[mode])
    full_count = rng.choice(FULL_LINE_COUNT_CHOICES)
    side_count = total - full_count
    if side_count < 2 or side_count > 4:
        return None

    if mode == "horizontal":
        rows = _pick_distinct(rng, list(range(2, h - 2)), full_count, FULL_COUNT_GAP)
        if len(rows) != full_count:
            return None
        for r in rows:
            draw_line(out, r, 0, end_r=r, end_c=w - 1, color=fg)
            segments.append({"kind": "full_h", "orient": "h", "row": r, "c0": 0, "c1": w - 1, "drs": (-1, 1)})

        left_count = side_count // 2
        right_count = side_count - left_count
        forbidden_rows = set(rows)
        left_intervals = _sample_intervals(rng, h, left_count, forbidden_rows)
        right_intervals = _sample_intervals(rng, h, right_count, forbidden_rows)
        if len(left_intervals) != left_count or len(right_intervals) != right_count:
            return None
        for r0, r1 in left_intervals:
            draw_line(out, r0, 0, end_r=r1, end_c=0, color=fg)
            segments.append({"kind": "border_v", "orient": "v", "col": 0, "r0": r0, "r1": r1, "dcs": (1,)})
        for r0, r1 in right_intervals:
            draw_line(out, r0, w - 1, end_r=r1, end_c=w - 1, color=fg)
            segments.append({"kind": "border_v", "orient": "v", "col": w - 1, "r0": r0, "r1": r1, "dcs": (-1,)})
    else:
        cols = _pick_distinct(rng, list(range(2, w - 2)), full_count, FULL_COUNT_GAP)
        if len(cols) != full_count:
            return None
        for c in cols:
            draw_line(out, 0, c, end_r=h - 1, end_c=c, color=fg)
            segments.append({"kind": "full_v", "orient": "v", "col": c, "r0": 0, "r1": h - 1, "dcs": (-1, 1)})

        top_count = side_count // 2
        bottom_count = side_count - top_count
        forbidden_cols = set(cols)
        top_intervals = _sample_intervals(rng, w, top_count, forbidden_cols)
        bottom_intervals = _sample_intervals(rng, w, bottom_count, forbidden_cols)
        if len(top_intervals) != top_count or len(bottom_intervals) != bottom_count:
            return None
        for c0, c1 in top_intervals:
            draw_line(out, 0, c0, end_r=0, end_c=c1, color=fg)
            segments.append({"kind": "border_h", "orient": "h", "row": 0, "c0": c0, "c1": c1, "drs": (1,)})
        for c0, c1 in bottom_intervals:
            draw_line(out, h - 1, c0, end_r=h - 1, end_c=c1, color=fg)
            segments.append({"kind": "border_h", "orient": "h", "row": h - 1, "c0": c0, "c1": c1, "drs": (-1,)})

    return out, segments


def _apply_true_gaps(
    rng: random.Random,
    out: np.ndarray,
    segments: list[dict],
    fg: int,
    bg: int,
) -> np.ndarray | None:
    inp = out.copy()
    occupied: set[tuple[int, int]] = set()
    ready: list[tuple[dict, list[tuple[int, str, int]]]] = []
    for seg in segments:
        if seg["orient"] == "h":
            opts = _h_gap_options(seg, inp, fg, bg, occupied)
        else:
            opts = _v_gap_options(seg, inp, fg, bg, occupied)
        if opts:
            ready.append((seg, opts))
    if len(ready) < min(TRUE_GAP_CHOICES):
        return None

    true_gap_count = rng.choice(TRUE_GAP_CHOICES)
    if len(ready) < true_gap_count:
        return None

    full_ready = [item for item in ready if item[0]["kind"].startswith("full_")]
    border_ready = [item for item in ready if item[0]["kind"].startswith("border_")]
    chosen: list[tuple[dict, list[tuple[int, str, int]]]] = []
    if full_ready:
        chosen.append(rng.choice(full_ready))
    if border_ready and len(chosen) < true_gap_count:
        pick = rng.choice(border_ready)
        if all(pick[0] is not seg for seg, _ in chosen):
            chosen.append(pick)
    pool = ready[:]
    rng.shuffle(pool)
    for item in pool:
        if len(chosen) >= true_gap_count:
            break
        if any(item[0] is seg for seg, _ in chosen):
            continue
        chosen.append(item)
    if len(chosen) != true_gap_count:
        return None

    for seg, _opts in chosen:
        opts = _h_gap_options(seg, inp, fg, bg, occupied) if seg["orient"] == "h" else _v_gap_options(seg, inp, fg, bg, occupied)
        if not opts:
            return None
        gap_start, anchor, delta = rng.choice(opts)
        if seg["orient"] == "h":
            row = seg["row"]
            inp[row, gap_start : gap_start + GAP_SIZE] = bg
            chain = _h_chain_coords(row, gap_start, anchor, delta)
        else:
            col = seg["col"]
            inp[gap_start : gap_start + GAP_SIZE, col] = bg
            chain = _v_chain_coords(col, gap_start, anchor, delta)
        for r, c in chain:
            inp[r, c] = fg
        occupied.update(chain)
    return inp


def _place_decoys(rng: random.Random, inp: np.ndarray, fg: int, bg: int) -> np.ndarray | None:
    forbidden: set[tuple[int, int]] = set()
    fg_cells = [tuple(map(int, rc)) for rc in np.argwhere(inp == fg)]
    _reserve_neighborhood(forbidden, fg_cells)

    decoy_count = rng.choice(DECOY_COUNT_CHOICES)
    placed = 0
    while placed < decoy_count:
        opts = _decoy_options(inp, fg, bg, forbidden)
        if not opts:
            break
        choice = rng.choice(opts)
        for r, c in choice["cells"]:
            inp[r, c] = fg
        for r, c in choice["chain"]:
            inp[r, c] = fg
        _reserve_neighborhood(forbidden, choice["cells"] + choice["chain"])
        placed += 1
    if placed == 0:
        return None
    return inp


def _gen_attempt() -> np.ndarray | None:
    rng = random.Random()
    h = rng.randint(*GRID_H_RANGE)
    w = rng.randint(*GRID_W_RANGE)
    bg = rng.randrange(N_COLORS)
    fg = rng.choice([c for c in range(N_COLORS) if c != bg])

    built = _build_output(rng, h, w, bg, fg)
    if built is None:
        return None
    out, segments = built

    inp = _apply_true_gaps(rng, out, segments, fg, bg)
    if inp is None:
        return None

    inp = _place_decoys(rng, inp, fg, bg)
    if inp is None:
        return None

    try:
        if not np.array_equal(main(inp), out):
            return None
    except Exception:
        return None
    return inp


def generate_input() -> np.ndarray:
    for _ in range(GEN_RETRIES):
        grid = _gen_attempt()
        if grid is not None:
            return grid
    raise RuntimeError("failed to generate valid input for 7b80bb43 combo D8/0+D9/1")
