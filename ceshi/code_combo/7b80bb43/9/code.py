# Combo variant of 7b80bb43 (index 9)
# combo_spec: D2/1+D6/1+D7/3
# Name: 颜色标记抗干扰递归修剪

from __future__ import annotations

import random

import numpy as np

from arc_utils import draw_line, get_bg


# Design analysis for the fused variant:
# - Concrete behavior: start from a loop-based fg backbone. Cut exact 3-cell gaps on some loop edges.
#   Near each real gap, place a 2-pixel red diagonal chain. Also add 3-5 same-color diagonal fg
#   distractor chains elsewhere, plus several orthogonal fg branches attached to the loops. The solver
#   must use red-only chains to decide which 3-cell gaps to heal, then delete every diagonal pixel
#   (red hints + fg distractors), and finally run global recursive pruning so every attached branch
#   collapses, leaving only the loop backbone.
# - Why the original design breaks: the seed solver assumed a single non-bg color, treated every
#   diagonal-only pixel as a functional hint, and only pruned local degree-1 leftovers near those hints.
#   Under D2/1 that would wrongly let same-color distractors trigger fills; under D7/3 the original
#   tree-like output would prune to empty. The fused generator therefore changes the clean target into a
#   non-empty 2-core (rectangular cycles, sometimes joined by a corridor).
# - Complexity calibration: the task uses exactly three visible roles per sample: bg, one fg, and a
#   stable marker color (red). There are only 2-3 true gaps, but 3-5 same-color distractor diagonals,
#   so color is the only reliable way to identify gap hints. Recursive pruning is made visible by 2-4
#   long orthogonal branches that reach toward the border and disappear completely.
# - Legality / observability guarantees: markers and distractors are placed only on free background,
#   never overlap, and stay orthogonally disconnected from the backbone. Branch anchors are unique and
#   avoid marker neighborhoods. The solver assumes exactly one fg color plus one marker color, true gaps
#   of length 3, and marker chains lying within a radius-2 neighborhood of their matching gap; the
#   generator enforces all of those assumptions.


MARK = 2
GAP_SIZE = 3
MARK_CHAIN_LEN = 2

GRID_H_RANGE = (18, 29)
GRID_W_RANGE = (18, 30)
N_COLORS = 10

TRUE_GAP_COUNT_RANGE = (2, 3)
DISTRACTOR_COUNT_RANGE = (3, 5)
DISTRACTOR_LENGTHS = (2, 3)
BRANCH_COUNT_RANGE = (2, 4)

GEN_RETRIES = 400


def main(grid: np.ndarray) -> np.ndarray:
    """
    Remove mixed-color diagonal clutter, restore marked line breaks, then peel off every tree branch.

    Input structure:
    - bg is the dominant color. One non-bg color is the line network; red is the marker color.
    - Real gap hints are short red diagonal chains placed next to 3-cell breaks on straight fg segments.
    - 3-5 same-color fg diagonal chains are unrelated distractors, and extra fg branches may hang off the core.

    Rule:
    - Use only the red markers to decide which 3-cell horizontal/vertical gaps must be filled.
    - Delete all diagonal pixels: every red marker pixel and every diagonal-only fg distractor pixel.
    - After healing the marked gaps, recursively prune all remaining fg pixels of degree <= 1 until stable.

    Key insight:
    Red identifies semantics, not geometry: many diagonal chains exist, but only the red ones authorize gap repair.
    The final output is the 2-core of the repaired fg network, so all acyclic branches disappear completely.
    """
    g = np.array(grid, dtype=int)
    bg = int(get_bg(g))
    H, W = g.shape

    colors = [int(c) for c in np.unique(g) if int(c) != int(bg)]
    assert len(colors) == 2
    counts = {c: int(np.count_nonzero(g == int(c))) for c in colors}
    fg = max(colors, key=lambda c: counts[int(c)])
    mark = min(colors, key=lambda c: counts[int(c)])

    def deg4(mask: np.ndarray, r: int, c: int) -> int:
        d = 0
        if r > 0 and bool(mask[r - 1, c]):
            d += 1
        if r + 1 < H and bool(mask[r + 1, c]):
            d += 1
        if c > 0 and bool(mask[r, c - 1]):
            d += 1
        if c + 1 < W and bool(mask[r, c + 1]):
            d += 1
        return d

    def has_diag(mask: np.ndarray, r: int, c: int) -> bool:
        for dr, dc in ((-1, -1), (-1, 1), (1, -1), (1, 1)):
            rr = r + dr
            cc = c + dc
            if 0 <= rr < H and 0 <= cc < W and bool(mask[rr, cc]):
                return True
        return False

    marker = g == int(mark)
    fg_mask = g == int(fg)

    diag_noise = np.zeros((H, W), dtype=bool)
    for r in range(H):
        for c in range(W):
            if bool(fg_mask[r, c]) and deg4(fg_mask, r, c) == 0 and has_diag(fg_mask, r, c):
                diag_noise[r, c] = True

    cleaned = g.copy()
    cleaned[marker | diag_noise] = int(bg)
    line = cleaned == int(fg)

    def marker_near_gap(r0: int, c0: int, r1: int, c1: int) -> bool:
        rr0 = max(0, int(r0 - 2))
        rr1 = min(int(H - 1), int(r1 + 2))
        cc0 = max(0, int(c0 - 2))
        cc1 = min(int(W - 1), int(c1 + 2))
        return bool(marker[rr0 : rr1 + 1, cc0 : cc1 + 1].any())

    def row_segments(mask: np.ndarray, r: int) -> list[tuple[int, int]]:
        cols = np.where(mask[r])[0]
        if cols.size == 0:
            return []
        segs: list[tuple[int, int]] = []
        start = prev = int(cols[0])
        for c in cols[1:]:
            c = int(c)
            if c == prev + 1:
                prev = c
            else:
                segs.append((start, prev))
                start = prev = c
        segs.append((start, prev))
        return segs

    def col_segments(mask: np.ndarray, c: int) -> list[tuple[int, int]]:
        rows = np.where(mask[:, c])[0]
        if rows.size == 0:
            return []
        segs: list[tuple[int, int]] = []
        start = prev = int(rows[0])
        for r in rows[1:]:
            r = int(r)
            if r == prev + 1:
                prev = r
            else:
                segs.append((start, prev))
                start = prev = r
        segs.append((start, prev))
        return segs

    for r in range(H):
        for (a1, b1), (a2, b2) in zip(row_segments(line, r), row_segments(line, r)[1:]):
            if b1 - a1 + 1 < 2 or b2 - a2 + 1 < 2:
                continue
            gap0 = int(b1 + 1)
            gap1 = int(a2 - 1)
            if gap1 - gap0 + 1 != GAP_SIZE:
                continue
            if marker_near_gap(r, gap0, r, gap1):
                line[r, gap0 : gap1 + 1] = True

    for c in range(W):
        for (a1, b1), (a2, b2) in zip(col_segments(line, c), col_segments(line, c)[1:]):
            if b1 - a1 + 1 < 2 or b2 - a2 + 1 < 2:
                continue
            gap0 = int(b1 + 1)
            gap1 = int(a2 - 1)
            if gap1 - gap0 + 1 != GAP_SIZE:
                continue
            if marker_near_gap(gap0, c, gap1, c):
                line[gap0 : gap1 + 1, c] = True

    changed = True
    while changed:
        changed = False
        to_clear: list[tuple[int, int]] = []
        for r in range(H):
            for c in range(W):
                if bool(line[r, c]) and deg4(line, r, c) <= 1:
                    to_clear.append((r, c))
        if to_clear:
            changed = True
            for r, c in to_clear:
                line[r, c] = False

    out = np.full((H, W), int(bg), dtype=int)
    out[line] = int(fg)
    return out


def _draw_rect(grid: np.ndarray, top: int, left: int, bottom: int, right: int, color: int) -> None:
    draw_line(grid, top, left, end_r=top, end_c=right, color=color)
    draw_line(grid, bottom, left, end_r=bottom, end_c=right, color=color)
    draw_line(grid, top, left, end_r=bottom, end_c=left, color=color)
    draw_line(grid, top, right, end_r=bottom, end_c=right, color=color)


def _reserve(cells: set[tuple[int, int]], coords: list[tuple[int, int]], radius: int = 1) -> None:
    for r, c in coords:
        for dr in range(-radius, radius + 1):
            for dc in range(-radius, radius + 1):
                cells.add((int(r + dr), int(c + dc)))


def _h_chain_coords(row: int, gap_start: int, anchor: str, vdir: int) -> list[tuple[int, int]]:
    if anchor == "left":
        return [(int(row + vdir * i), int(gap_start - 1 + i)) for i in range(1, MARK_CHAIN_LEN + 1)]
    return [(int(row + vdir * i), int(gap_start + GAP_SIZE - i)) for i in range(1, MARK_CHAIN_LEN + 1)]


def _v_chain_coords(col: int, gap_start: int, anchor: str, hdir: int) -> list[tuple[int, int]]:
    if anchor == "top":
        return [(int(gap_start + i), int(col + hdir * (i + 1))) for i in range(MARK_CHAIN_LEN)]
    return [(int(gap_start + GAP_SIZE - 1 - i), int(col + hdir * (i + 1))) for i in range(MARK_CHAIN_LEN)]


def _in_bounds(shape: tuple[int, int], r: int, c: int) -> bool:
    H, W = shape
    return 0 <= int(r) < H and 0 <= int(c) < W


def _validate_diag_chain(grid: np.ndarray, coords: list[tuple[int, int]], bg: int) -> bool:
    H, W = grid.shape
    coord_set = set(coords)
    for r, c in coords:
        if not _in_bounds((H, W), r, c) or int(grid[r, c]) != int(bg):
            return False
        for rr, cc in ((r - 1, c), (r + 1, c), (r, c - 1), (r, c + 1)):
            if (rr, cc) in coord_set:
                return False
            if _in_bounds((H, W), rr, cc) and int(grid[rr, cc]) != int(bg):
                return False
        has_diag = False
        for rr, cc in ((r - 1, c - 1), (r - 1, c + 1), (r + 1, c - 1), (r + 1, c + 1)):
            if (rr, cc) in coord_set:
                has_diag = True
                break
            if _in_bounds((H, W), rr, cc) and int(grid[rr, cc]) != int(bg):
                has_diag = True
                break
        if not has_diag:
            return False
    return True


def _build_core(rng: random.Random, h: int, w: int, bg: int, fg: int) -> tuple[np.ndarray, list[dict], int | None] | None:
    out = np.full((h, w), bg, dtype=int)
    rects: list[dict] = []
    bridge_row: int | None = None

    if w >= 24 and h >= 11 and rng.random() < 0.7:
        top = rng.randint(2, h - 9)
        rect_h = rng.randint(7, min(9, h - top - 2))
        bottom = top + rect_h - 1
        left0 = rng.randint(2, 4)
        right1 = w - 1 - rng.randint(2, 4)
        left_w = rng.randint(7, 9)
        right_w = rng.randint(7, 9)
        left1 = left0 + left_w - 1
        right0 = right1 - right_w + 1
        if right0 - left1 < 4:
            return None
        rects = [
            {"top": top, "left": left0, "bottom": bottom, "right": left1, "outer_lr": "left"},
            {"top": top, "left": right0, "bottom": bottom, "right": right1, "outer_lr": "right"},
        ]
        for rect in rects:
            _draw_rect(out, rect["top"], rect["left"], rect["bottom"], rect["right"], fg)
        bridge_row = rng.randint(top + 2, bottom - 2)
        draw_line(out, bridge_row, left1, end_r=bridge_row, end_c=right0, color=fg)
        return out, rects, bridge_row

    rect_h = rng.randint(7, min(10, h - 4))
    rect_w = rng.randint(9, min(14, w - 4))
    top = rng.randint(2, h - rect_h - 2)
    left = rng.randint(2, w - rect_w - 2)
    bottom = top + rect_h - 1
    right = left + rect_w - 1
    rect = {"top": top, "left": left, "bottom": bottom, "right": right, "outer_lr": "both"}
    _draw_rect(out, top, left, bottom, right, fg)
    return out, [rect], None


def _gap_candidates(rects: list[dict], out: np.ndarray, bg: int) -> tuple[list[dict], list[dict]]:
    h_cands: list[dict] = []
    v_cands: list[dict] = []
    for rect in rects:
        top = int(rect["top"])
        left = int(rect["left"])
        bottom = int(rect["bottom"])
        right = int(rect["right"])

        for row, vdir in ((top, -1), (bottom, 1)):
            max_start = right - GAP_SIZE - 1
            for g in range(left + 2, max_start + 1):
                for anchor in ("left", "right"):
                    chain = _h_chain_coords(row, g, anchor, vdir)
                    temp = out.copy()
                    temp[row, g : g + GAP_SIZE] = bg
                    if _validate_diag_chain(temp, chain, bg):
                        h_cands.append(
                            {
                                "kind": "h",
                                "row": row,
                                "gap_start": g,
                                "anchor": anchor,
                                "vdir": vdir,
                                "chain": chain,
                                "cells": [(row, c) for c in range(g, g + GAP_SIZE)],
                            }
                        )

        for col, hdir in ((left, -1), (right, 1)):
            max_start = bottom - GAP_SIZE - 1
            for g in range(top + 2, max_start + 1):
                for anchor in ("top", "bottom"):
                    chain = _v_chain_coords(col, g, anchor, hdir)
                    temp = out.copy()
                    temp[g : g + GAP_SIZE, col] = bg
                    if _validate_diag_chain(temp, chain, bg):
                        v_cands.append(
                            {
                                "kind": "v",
                                "col": col,
                                "gap_start": g,
                                "anchor": anchor,
                                "hdir": hdir,
                                "chain": chain,
                                "cells": [(r, col) for r in range(g, g + GAP_SIZE)],
                            }
                        )

    return h_cands, v_cands


def _choose_gaps(rng: random.Random, h_cands: list[dict], v_cands: list[dict]) -> tuple[list[dict], set[tuple[int, int]]] | None:
    if not h_cands or not v_cands:
        return None

    selected: list[dict] = []
    reserved: set[tuple[int, int]] = set()

    def footprint(cand: dict) -> list[tuple[int, int]]:
        return list(cand["cells"]) + list(cand["chain"])

    def can_use(cand: dict) -> bool:
        return all((int(r), int(c)) not in reserved for r, c in footprint(cand))

    h_pool = h_cands[:]
    v_pool = v_cands[:]
    rng.shuffle(h_pool)
    rng.shuffle(v_pool)

    first_h = next((cand for cand in h_pool if can_use(cand)), None)
    if first_h is None:
        return None
    selected.append(first_h)
    _reserve(reserved, footprint(first_h), radius=2)

    first_v = next((cand for cand in v_pool if can_use(cand)), None)
    if first_v is None:
        return None
    selected.append(first_v)
    _reserve(reserved, footprint(first_v), radius=2)

    target = rng.randint(*TRUE_GAP_COUNT_RANGE)
    extra_pool = h_pool + v_pool
    rng.shuffle(extra_pool)
    for cand in extra_pool:
        if len(selected) >= target:
            break
        if can_use(cand):
            selected.append(cand)
            _reserve(reserved, footprint(cand), radius=2)

    if len(selected) < 2:
        return None
    return selected, reserved


def _branch_candidates(rects: list[dict], shape: tuple[int, int], bridge_row: int | None) -> list[list[tuple[int, int]]]:
    H, W = shape
    cands: list[list[tuple[int, int]]] = []
    for idx, rect in enumerate(rects):
        top = int(rect["top"])
        left = int(rect["left"])
        bottom = int(rect["bottom"])
        right = int(rect["right"])

        if top >= 2:
            for c in range(left + 1, right):
                if bridge_row is not None and c in (left, right):
                    continue
                cands.append([(r, c) for r in range(0, top)])
        if H - bottom - 1 >= 2:
            for c in range(left + 1, right):
                cands.append([(r, c) for r in range(bottom + 1, H)])

        outer = rect["outer_lr"]
        if outer in ("left", "both") and left >= 2:
            for r in range(top + 1, bottom):
                if bridge_row is not None and idx == 0 and r == bridge_row:
                    continue
                cands.append([(r, c) for c in range(0, left)])
        if outer in ("right", "both") and W - right - 1 >= 2:
            for r in range(top + 1, bottom):
                if bridge_row is not None and idx == len(rects) - 1 and r == bridge_row:
                    continue
                cands.append([(r, c) for c in range(right + 1, W)])
    return cands


def _distractor_options(inp: np.ndarray, bg: int, forbidden: set[tuple[int, int]]) -> list[list[tuple[int, int]]]:
    H, W = inp.shape
    opts: list[list[tuple[int, int]]] = []
    for length in DISTRACTOR_LENGTHS:
        for r0 in range(1, H - length):
            for c0 in range(1, W - 1):
                for dc in (-1, 1):
                    coords = [(int(r0 + i), int(c0 + dc * i)) for i in range(length)]
                    if any(not _in_bounds((H, W), r, c) for r, c in coords):
                        continue
                    if any((r, c) in forbidden for r, c in coords):
                        continue
                    if _validate_diag_chain(inp, coords, bg):
                        opts.append(coords)
    return opts


def _gen_attempt() -> np.ndarray | None:
    rng = random.Random()
    h = rng.randint(*GRID_H_RANGE)
    w = rng.randint(*GRID_W_RANGE)

    bg = rng.choice([c for c in range(N_COLORS) if c != MARK])
    fg = rng.choice([c for c in range(N_COLORS) if c not in (bg, MARK)])

    built = _build_core(rng, h, w, bg, fg)
    if built is None:
        return None
    out, rects, bridge_row = built

    gaps = _choose_gaps(rng, *_gap_candidates(rects, out, bg))
    if gaps is None:
        return None
    selected_gaps, reserved = gaps

    inp = out.copy()
    for cand in selected_gaps:
        for r, c in cand["cells"]:
            inp[int(r), int(c)] = int(bg)
        for r, c in cand["chain"]:
            inp[int(r), int(c)] = int(MARK)

    branch_pool = _branch_candidates(rects, inp.shape, bridge_row)
    rng.shuffle(branch_pool)
    branch_target = rng.randint(*BRANCH_COUNT_RANGE)
    branches_added = 0
    for cells in branch_pool:
        if branches_added >= branch_target:
            break
        if any((int(r), int(c)) in reserved for r, c in cells):
            continue
        for r, c in cells:
            inp[int(r), int(c)] = int(fg)
        _reserve(reserved, cells, radius=1)
        branches_added += 1
    if branches_added < 2:
        return None

    forbidden: set[tuple[int, int]] = set(reserved)
    nonbg = [(int(r), int(c)) for r, c in np.argwhere(inp != int(bg))]
    _reserve(forbidden, nonbg, radius=1)
    distractor_pool = _distractor_options(inp, bg, forbidden)
    rng.shuffle(distractor_pool)
    distractor_target = rng.randint(*DISTRACTOR_COUNT_RANGE)
    placed = 0
    for coords in distractor_pool:
        if placed >= distractor_target:
            break
        if any((int(r), int(c)) in forbidden for r, c in coords):
            continue
        for r, c in coords:
            inp[int(r), int(c)] = int(fg)
        _reserve(forbidden, coords, radius=1)
        placed += 1
    if placed < 3:
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
    raise RuntimeError("failed to generate valid input for combo variant 7b80bb43/9")
