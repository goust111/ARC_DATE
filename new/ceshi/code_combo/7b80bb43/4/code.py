# Combo variant of 7b80bb43 (index 4)
# combo_spec: D1/0+D14/2
# Name: 对称性精确匹配填充

from __future__ import annotations

import random

import numpy as np

from arc_utils import components_mask, get_bg

# Design analysis for the fused variant
# - Concrete behavior: the original task filled any short H/V gap near a diagonal hint.
#   Here each diagonal chain is inspected independently. A nearby gap is symmetry-repaired
#   only when chain_length == gap_length - 1. The repair does not fill the gap; instead it
#   trims unmatched excess cells from the longer arm so the remaining two arms mirror each
#   other around the gap center.
#   Example: a 3-cell horizontal gap with a 2-pixel diagonal chain and arm lengths 5 vs 3
#   becomes 3 vs 3 by deleting the two far-left excess pixels; the same gap with a 1-pixel
#   chain stays unchanged apart from deleting the diagonal hint itself.
# - Complexity calibration: every generated input contains three visible case types:
#   exact+asymmetry (repair by deletion), mismatch+asymmetry (no repair), and another
#   exact+asymmetry case in the other orientation. This prevents degeneration into
#   "always trim asymmetry" or "only count chain length".
# - Hidden assumptions changed from the source solver:
#   * global MAXG no longer applies; matching is per-component and exact.
#   * proximity alone is insufficient; local arm balance must be checked.
#   * multiple hints are not allowed to compete for the same broken segment in the
#     generator, so the solver can resolve each chain locally and independently.
# - Placement legality: the generator reserves Chebyshev-1 buffers between objects, keeps
#   all chains in background, and validates that chain pixels have no orthogonal fg
#   neighbors while still touching the target line diagonally.
# - Visual observability: the output difference is easy to read. Some broken segments lose
#   far-end pixels to become balanced, while others keep their longer arm because the chain
#   length does not match.


def _deg4(mask: np.ndarray, r: int, c: int) -> int:
    h, w = mask.shape
    total = 0
    if r > 0 and bool(mask[r - 1, c]):
        total += 1
    if r + 1 < h and bool(mask[r + 1, c]):
        total += 1
    if c > 0 and bool(mask[r, c - 1]):
        total += 1
    if c + 1 < w and bool(mask[r, c + 1]):
        total += 1
    return total


def _has_diag(mask: np.ndarray, r: int, c: int) -> bool:
    h, w = mask.shape
    for dr, dc in ((-1, -1), (-1, 1), (1, -1), (1, 1)):
        rr, cc = r + dr, c + dc
        if 0 <= rr < h and 0 <= cc < w and bool(mask[rr, cc]):
            return True
    return False


def _diag_mask(mask: np.ndarray) -> np.ndarray:
    h, w = mask.shape
    diag = np.zeros((h, w), dtype=bool)
    for r in range(h):
        for c in range(w):
            if bool(mask[r, c]) and _deg4(mask, r, c) == 0 and _has_diag(mask, r, c):
                diag[r, c] = True
    return diag


def _collect_gap_candidates(mask: np.ndarray) -> list[dict]:
    h, w = mask.shape
    candidates: list[dict] = []

    for r in range(h):
        cols = np.where(mask[r])[0]
        if cols.size < 2:
            continue
        start = int(cols[0])
        prev = int(cols[0])
        segs: list[tuple[int, int]] = []
        for c in cols[1:]:
            c = int(c)
            if c == prev + 1:
                prev = c
            else:
                segs.append((start, prev))
                start = prev = c
        segs.append((start, prev))
        for (l0, l1), (r0, r1) in zip(segs, segs[1:]):
            gap0 = l1 + 1
            gap1 = r0 - 1
            if gap1 < gap0:
                continue
            left_len = l1 - l0 + 1
            right_len = r1 - r0 + 1
            candidates.append(
                {
                    "kind": "h",
                    "line": r,
                    "gap0": gap0,
                    "gap1": gap1,
                    "size": gap1 - gap0 + 1,
                    "left_len": left_len,
                    "right_len": right_len,
                    "symmetric": left_len == right_len,
                    "imbalance": abs(left_len - right_len),
                    "score": min(left_len, right_len),
                }
            )

    for c in range(w):
        rows = np.where(mask[:, c])[0]
        if rows.size < 2:
            continue
        start = int(rows[0])
        prev = int(rows[0])
        segs: list[tuple[int, int]] = []
        for r in rows[1:]:
            r = int(r)
            if r == prev + 1:
                prev = r
            else:
                segs.append((start, prev))
                start = prev = r
        segs.append((start, prev))
        for (t0, t1), (b0, b1) in zip(segs, segs[1:]):
            gap0 = t1 + 1
            gap1 = b0 - 1
            if gap1 < gap0:
                continue
            top_len = t1 - t0 + 1
            bottom_len = b1 - b0 + 1
            candidates.append(
                {
                    "kind": "v",
                    "line": c,
                    "gap0": gap0,
                    "gap1": gap1,
                    "size": gap1 - gap0 + 1,
                    "left_len": top_len,
                    "right_len": bottom_len,
                    "symmetric": top_len == bottom_len,
                    "imbalance": abs(top_len - bottom_len),
                    "score": min(top_len, bottom_len),
                }
            )

    return candidates


def _candidate_is_near_component(candidate: dict, coords: np.ndarray) -> bool:
    chain = {(int(r), int(c)) for r, c in coords}
    chain_len = len(chain)
    if candidate["kind"] == "h":
        row = int(candidate["line"])
        gap0 = int(candidate["gap0"])
        gap_size = int(candidate["size"])
        return chain in (
            set(_h_chain_coords(row, gap0, gap_size, chain_len, "left")),
            set(_h_chain_coords(row, gap0, gap_size, chain_len, "right")),
        )
    col = int(candidate["line"])
    gap0 = int(candidate["gap0"])
    gap_size = int(candidate["size"])
    return chain in (
        set(_v_chain_coords(col, gap0, gap_size, chain_len, "top", -1)),
        set(_v_chain_coords(col, gap0, gap_size, chain_len, "top", 1)),
        set(_v_chain_coords(col, gap0, gap_size, chain_len, "bottom", -1)),
        set(_v_chain_coords(col, gap0, gap_size, chain_len, "bottom", 1)),
    )


def _repair_candidate(mask: np.ndarray, candidate: dict) -> None:
    keep = int(min(candidate["left_len"], candidate["right_len"]))
    if candidate["kind"] == "h":
        r = int(candidate["line"])
        gap0 = int(candidate["gap0"])
        gap1 = int(candidate["gap1"])
        left_len = int(candidate["left_len"])
        right_len = int(candidate["right_len"])
        if left_len > keep:
            left_clear0 = gap0 - left_len
            left_clear1 = gap0 - keep
            mask[r, left_clear0:left_clear1] = False
        if right_len > keep:
            right_clear0 = gap1 + 1 + keep
            right_clear1 = gap1 + 1 + right_len
            mask[r, right_clear0:right_clear1] = False
        return

    c = int(candidate["line"])
    gap0 = int(candidate["gap0"])
    gap1 = int(candidate["gap1"])
    top_len = int(candidate["left_len"])
    bottom_len = int(candidate["right_len"])
    if top_len > keep:
        top_clear0 = gap0 - top_len
        top_clear1 = gap0 - keep
        mask[top_clear0:top_clear1, c] = False
    if bottom_len > keep:
        bottom_clear0 = gap1 + 1 + keep
        bottom_clear1 = gap1 + 1 + bottom_len
        mask[bottom_clear0:bottom_clear1, c] = False


def analyze_cases(grid: np.ndarray) -> list[dict]:
    g = np.array(grid, dtype=int)
    bg = int(get_bg(g))
    colors = {int(x) for x in np.unique(g)}
    colors.discard(bg)
    if not colors:
        return []
    fg = int(next(iter(colors)))

    raw_mask = g == fg
    diag = _diag_mask(raw_mask)
    base_mask = raw_mask & (~diag)
    candidates = _collect_gap_candidates(base_mask)

    summary: list[dict] = []
    for coords in components_mask(diag, connectivity=8):
        coords = np.array(coords, dtype=int)
        comp_len = int(len(coords))
        near = [cand for cand in candidates if _candidate_is_near_component(cand, coords)]
        exact = [cand for cand in near if int(cand["size"]) == comp_len + 1]
        repairable_exact = [cand for cand in exact if int(cand["imbalance"]) > 0]
        balanced_exact = [cand for cand in exact if int(cand["imbalance"]) == 0]
        summary.append(
            {
                "chain_len": comp_len,
                "near_count": len(near),
                "exact_count": len(exact),
                "exact_repairable_count": len(repairable_exact),
                "exact_balanced_count": len(balanced_exact),
                "best_exact_imbalance": max((cand["imbalance"] for cand in repairable_exact), default=0),
            }
        )
    return summary


def main(grid: np.ndarray) -> np.ndarray:
    """
    Remove diagonal hint chains and symmetry-repair only exact-match gaps.

    Input structure:
    - A single foreground color draws horizontal and vertical line pieces on one background.
    - Some breaks in those line pieces have diagonal staircase hints nearby; chain lengths can differ.

    Rule:
    - Delete every diagonal-only foreground chain.
    - For each deleted chain, inspect nearby H/V gaps in the cleaned linework.
    - Only when the chain length is exactly one less than the gap size, trim unmatched
      excess cells from the longer surviving arm so the two arms become locally mirror
      balanced around the gap. The gap itself is never filled.

    Key insight: The diagonal hint is no longer a generic "repair here" marker. It only
    authorizes a local symmetry repair when its count matches the missing span; the repair
    is deletion-based, not completion-based.
    """
    g = np.array(grid, dtype=int)
    bg = int(get_bg(g))

    colors = {int(x) for x in np.unique(g)}
    colors.discard(bg)
    assert colors, "grid must contain a foreground color"
    fg = int(next(iter(colors)))

    raw_mask = g == fg
    diag = _diag_mask(raw_mask)
    out_mask = raw_mask & (~diag)
    base_mask = out_mask.copy()
    candidates = _collect_gap_candidates(base_mask)

    for coords in components_mask(diag, connectivity=8):
        coords = np.array(coords, dtype=int)
        comp_len = int(len(coords))
        matches = [
            cand
            for cand in candidates
            if int(cand["size"]) == comp_len + 1
            and _candidate_is_near_component(cand, coords)
        ]
        if not matches:
            continue
        matches.sort(
            key=lambda cand: (
                int(cand["imbalance"]),
                int(cand["score"]),
                -int(cand["gap0"]),
                0 if cand["kind"] == "h" else 1,
            ),
            reverse=True,
        )
        _repair_candidate(out_mask, matches[0])

    out = np.full(g.shape, bg, dtype=int)
    out[out_mask] = fg
    return out


GRID_H_RANGE = (16, 29)
GRID_W_RANGE = (17, 30)
N_COLORS = 10
GEN_RETRIES = 400


def _cheb1(coords: set[tuple[int, int]]) -> set[tuple[int, int]]:
    out: set[tuple[int, int]] = set()
    for r, c in coords:
        for dr in (-1, 0, 1):
            for dc in (-1, 0, 1):
                out.add((r + dr, c + dc))
    return out


def _h_chain_coords(row: int, gap_start: int, gap_size: int, length: int, anchor: str) -> list[tuple[int, int]]:
    if anchor == "left":
        return [(row + i, gap_start - 1 + i) for i in range(1, length + 1)]
    return [(row + i, gap_start + gap_size - i) for i in range(1, length + 1)]


def _v_chain_coords(
    col: int, gap_start: int, gap_size: int, length: int, anchor: str, dc: int
) -> list[tuple[int, int]]:
    if anchor == "top":
        return [(gap_start + i, col + dc * (i + 1)) for i in range(length)]
    return [(gap_start + gap_size - 1 - i, col + dc * (i + 1)) for i in range(length)]


def _chain_valid(
    h: int, w: int, base_fg: set[tuple[int, int]], chain: list[tuple[int, int]]
) -> bool:
    chain_set = set(chain)
    if len(chain_set) != len(chain):
        return False
    for r, c in chain:
        if not (0 <= r < h and 0 <= c < w):
            return False
        if (r, c) in base_fg:
            return False
        for rr, cc in ((r - 1, c), (r + 1, c), (r, c - 1), (r, c + 1)):
            if (rr, cc) in chain_set or (rr, cc) in base_fg:
                return False
        has_diag = False
        for rr, cc in ((r - 1, c - 1), (r - 1, c + 1), (r + 1, c - 1), (r + 1, c + 1)):
            if (rr, cc) in chain_set or (rr, cc) in base_fg:
                has_diag = True
                break
        if not has_diag:
            return False
    return True


def _write_cells(grid: np.ndarray, cells: set[tuple[int, int]], color: int) -> None:
    for r, c in cells:
        grid[r, c] = color


def _repaired_horizontal_cells(row: int, gap0: int, gap_size: int, left_arm: int, right_arm: int) -> set[tuple[int, int]]:
    keep = min(left_arm, right_arm)
    cells = {(row, c) for c in range(gap0 - keep, gap0)}
    cells.update((row, c) for c in range(gap0 + gap_size, gap0 + gap_size + keep))
    return cells


def _repaired_vertical_cells(col: int, gap0: int, gap_size: int, top_arm: int, bottom_arm: int) -> set[tuple[int, int]]:
    keep = min(top_arm, bottom_arm)
    cells = {(r, col) for r in range(gap0 - keep, gap0)}
    cells.update((r, col) for r in range(gap0 + gap_size, gap0 + gap_size + keep))
    return cells


def _place_horizontal_case(
    rng: random.Random,
    out: np.ndarray,
    inp: np.ndarray,
    forbidden: set[tuple[int, int]],
    fg: int,
    *,
    row_lo: int,
    row_hi: int,
    col_lo: int,
    col_hi: int,
    left_arm: int,
    right_arm: int,
    gap_size: int,
    chain_len: int,
    output_style: str,
) -> bool:
    h, w = out.shape
    total = left_arm + gap_size + right_arm
    if row_lo > row_hi or col_hi - col_lo + 1 < total:
        return False

    anchors = ["left", "right"]
    rows = list(range(row_lo, row_hi + 1))
    rng.shuffle(rows)
    for row in rows:
        c_options = list(range(col_lo, col_hi - total + 2))
        rng.shuffle(c_options)
        for c0 in c_options:
            gap0 = c0 + left_arm
            line_out = {(row, c) for c in range(c0, c0 + total)}
            line_in = line_out - {(row, c) for c in range(gap0, gap0 + gap_size)}
            if output_style == "repair":
                line_target = _repaired_horizontal_cells(row, gap0, gap_size, left_arm, right_arm)
            else:
                line_target = line_in
            local_fg = {(int(r), int(c)) for r, c in np.argwhere(inp == fg)} | line_in
            rng.shuffle(anchors)
            for anchor in anchors:
                chain = _h_chain_coords(row, gap0, gap_size, chain_len, anchor)
                chain_set = set(chain)
                if not _chain_valid(h, w, local_fg - chain_set, chain):
                    continue
                actual = set(line_out) | set(line_in) | chain_set
                if any(cell in forbidden for cell in actual):
                    continue
                reserve = _cheb1(actual)
                if any(cell in forbidden for cell in reserve):
                    continue
                _write_cells(out, line_target, fg)
                _write_cells(inp, line_in, fg)
                _write_cells(inp, chain_set, fg)
                forbidden.update(reserve)
                return True
    return False


def _place_vertical_case(
    rng: random.Random,
    out: np.ndarray,
    inp: np.ndarray,
    forbidden: set[tuple[int, int]],
    fg: int,
    *,
    row_lo: int,
    row_hi: int,
    col_lo: int,
    col_hi: int,
    top_arm: int,
    bottom_arm: int,
    gap_size: int,
    chain_len: int,
    output_style: str,
) -> bool:
    h, w = out.shape
    total = top_arm + gap_size + bottom_arm
    if col_lo > col_hi or row_hi - row_lo + 1 < total:
        return False

    cols = list(range(col_lo, col_hi + 1))
    rng.shuffle(cols)
    for col in cols:
        r_options = list(range(row_lo, row_hi - total + 2))
        rng.shuffle(r_options)
        for r0 in r_options:
            gap0 = r0 + top_arm
            line_out = {(r, col) for r in range(r0, r0 + total)}
            line_in = line_out - {(r, col) for r in range(gap0, gap0 + gap_size)}
            if output_style == "repair":
                line_target = _repaired_vertical_cells(col, gap0, gap_size, top_arm, bottom_arm)
            else:
                line_target = line_in
            base_fg = {(int(r), int(c)) for r, c in np.argwhere(inp == fg)} | line_in
            options = [("top", -1), ("top", 1), ("bottom", -1), ("bottom", 1)]
            rng.shuffle(options)
            for anchor, dc in options:
                chain = _v_chain_coords(col, gap0, gap_size, chain_len, anchor, dc)
                chain_set = set(chain)
                if not _chain_valid(h, w, base_fg - chain_set, chain):
                    continue
                actual = set(line_out) | set(line_in) | chain_set
                if any(cell in forbidden for cell in actual):
                    continue
                reserve = _cheb1(actual)
                if any(cell in forbidden for cell in reserve):
                    continue
                _write_cells(out, line_target, fg)
                _write_cells(inp, line_in, fg)
                _write_cells(inp, chain_set, fg)
                forbidden.update(reserve)
                return True
    return False


def _place_plain_segment(
    rng: random.Random,
    out: np.ndarray,
    inp: np.ndarray,
    forbidden: set[tuple[int, int]],
    fg: int,
    *,
    orientation: str,
    row_lo: int,
    row_hi: int,
    col_lo: int,
    col_hi: int,
    length_lo: int,
    length_hi: int,
) -> bool:
    h, w = out.shape
    for _ in range(80):
        length = rng.randint(length_lo, length_hi)
        if orientation == "h":
            if row_lo > row_hi or col_hi - col_lo + 1 < length:
                return False
            row = rng.randint(row_lo, row_hi)
            c0 = rng.randint(col_lo, col_hi - length + 1)
            cells = {(row, c) for c in range(c0, c0 + length)}
        else:
            if col_lo > col_hi or row_hi - row_lo + 1 < length:
                return False
            col = rng.randint(col_lo, col_hi)
            r0 = rng.randint(row_lo, row_hi - length + 1)
            cells = {(r, col) for r in range(r0, r0 + length)}
        reserve = _cheb1(cells)
        if any(cell in forbidden for cell in reserve):
            continue
        _write_cells(out, cells, fg)
        _write_cells(inp, cells, fg)
        forbidden.update(reserve)
        return True
    return False


def _gen_attempt() -> np.ndarray | None:
    rng = random.Random()
    h = rng.randint(*GRID_H_RANGE)
    w = rng.randint(*GRID_W_RANGE)
    bg = rng.randrange(N_COLORS)
    fg = rng.choice([c for c in range(N_COLORS) if c != bg])

    out = np.full((h, w), bg, dtype=int)
    inp = out.copy()
    forbidden: set[tuple[int, int]] = set()

    repair_gap = rng.choice([2, 3])
    repair_short = rng.randint(2, 3)
    repair_long = rng.randint(repair_short + 2, repair_short + 4)
    repair_left, repair_right = (
        (repair_short, repair_long) if rng.random() < 0.5 else (repair_long, repair_short)
    )
    if not _place_horizontal_case(
        rng,
        out,
        inp,
        forbidden,
        fg,
        row_lo=2,
        row_hi=max(2, min(h // 4, h - 6)),
        col_lo=4,
        col_hi=w - 5,
        left_arm=repair_left,
        right_arm=repair_right,
        gap_size=repair_gap,
        chain_len=repair_gap - 1,
        output_style="repair",
    ):
        return None

    mis_gap = rng.choice([2, 3])
    mis_short = rng.randint(2, 3)
    mis_long = rng.randint(mis_short + 2, mis_short + 4)
    mis_top, mis_bottom = (
        (mis_short, mis_long) if rng.random() < 0.5 else (mis_long, mis_short)
    )
    if not _place_vertical_case(
        rng,
        out,
        inp,
        forbidden,
        fg,
        row_lo=2,
        row_hi=h - 5,
        col_lo=max(w // 2, 4),
        col_hi=w - 4,
        top_arm=mis_top,
        bottom_arm=mis_bottom,
        gap_size=mis_gap,
        chain_len=mis_gap,
        output_style="same_gap",
    ):
        return None

    asym_gap = rng.choice([2, 3])
    short_arm = rng.randint(2, 3)
    long_arm = rng.randint(short_arm + 2, short_arm + 4)
    asym_top, asym_bottom = (short_arm, long_arm) if rng.random() < 0.5 else (long_arm, short_arm)
    if not _place_vertical_case(
        rng,
        out,
        inp,
        forbidden,
        fg,
        row_lo=max(4, h // 3),
        row_hi=h - 4,
        col_lo=2,
        col_hi=max(3, min(w // 3, w - 4)),
        top_arm=asym_top,
        bottom_arm=asym_bottom,
        gap_size=asym_gap,
        chain_len=asym_gap - 1,
        output_style="repair",
    ):
        if not _place_vertical_case(
            rng,
            out,
            inp,
            forbidden,
            fg,
            row_lo=max(4, h // 3),
            row_hi=h - 4,
            col_lo=2,
            col_hi=max(3, min(w // 3, w - 4)),
            top_arm=asym_bottom,
            bottom_arm=asym_top,
            gap_size=asym_gap,
            chain_len=asym_gap - 1,
            output_style="repair",
        ):
            return None

    _place_plain_segment(
        rng,
        out,
        inp,
        forbidden,
        fg,
        orientation="v",
        row_lo=max(h // 3, 4),
        row_hi=h - 3,
        col_lo=1,
        col_hi=min(3, w - 2),
        length_lo=4,
        length_hi=min(7, h - 4),
    )
    _place_plain_segment(
        rng,
        out,
        inp,
        forbidden,
        fg,
        orientation="h",
        row_lo=max(4, h // 3),
        row_hi=min(h - 3, max(4, h // 3 + 2)),
        col_lo=max(w // 2, 4),
        col_hi=w - 2,
        length_lo=4,
        length_hi=min(7, w - 4),
    )

    try:
        if not np.array_equal(main(inp), out):
            return None
    except Exception:
        return None

    stats = analyze_cases(inp)
    if len(stats) != 3 or any(s["near_count"] != 1 for s in stats):
        return None
    positive = sum(1 for s in stats if s["exact_repairable_count"] > 0)
    mismatch_only = sum(1 for s in stats if s["exact_count"] == 0 and s["near_count"] > 0)
    if positive < 2 or mismatch_only < 1:
        return None

    return inp


def generate_input() -> np.ndarray:
    for _ in range(GEN_RETRIES):
        grid = _gen_attempt()
        if grid is not None:
            return grid
    raise RuntimeError("failed to generate valid input for fused variant 7b80bb43/4")
