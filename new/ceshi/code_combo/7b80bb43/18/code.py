# Combo variant of 7b80bb43 (index 18)
# combo_spec: D7/1+D12/1
# Name: 度一检测全局修剪

from __future__ import annotations

import random

import numpy as np

from arc_utils import draw_line, get_bg

# Design analysis for the fused variant
# -------------------------------------
# a) Concrete behavior and trace
#    - Original task:
#      * detect diagonal-only cue pixels (`deg4==0` and diagonal contact),
#      * delete them,
#      * use those deleted diagonal hints to fill nearby 3-cell gaps,
#      * then prune only a small local area near the hints.
#    - Fused task:
#      * detect every foreground leaf pixel with exactly one orthogonal neighbor,
#      * use those deleted degree-1 leaves as the cue set,
#      * fill nearby 3-cell horizontal/vertical gaps,
#      * then prune degree-1 leaves globally until no leaves remain.
#    - Small trace:
#      * clean core output: a rectangle loop has no degree-1 cells.
#      * corrupted input: cut a 3-cell gap on the top edge, then add one leaf
#        pixel just above each gap endpoint. Those two leaves are the only
#        `deg4==1` pixels around the break, so they get deleted first; the line
#        endpoints stay degree-2 in the input because of those temporary leaves.
#      * gap fill sees deleted leaves inside the gap neighborhood and restores
#        the missing 3 cells.
#      * elsewhere, a length-3 branch attached to the loop contributes exactly
#        one initial leaf at its tip. After the first deletion, its new tip
#        becomes degree-1, so the later global pruning peels the whole branch.
#    - Difference from the source task:
#      * the cue geometry is no longer a detached diagonal staircase;
#      * the post-fill cleanup is no longer local, so distant branches vanish.
#
# b) Complexity calibration
#    - The output is still a one-color orthogonal line network, but it is now
#      constrained to be a leafless "core": rectangles with optional internal
#      chords. This keeps the puzzle visually close to the source while making
#      the global 2-core-style pruning meaningful.
#    - Every sample contains 2-3 real 3-cell gaps and at least one extra branch
#      of length 2-4 far from any gap. That guarantees both dimensions of the
#      combo are exercised in the same grid.
#    - The cue vocabulary stays small:
#      * gap repair cues are single-pixel leaves near gap endpoints,
#      * global-pruning-only noise is a small number of straight branches,
#      * optional floating distractors are only 2-pixel dominoes.
#      This keeps the visual categories repeated and learnable.
#
# c) Hidden-assumption audit against the original main()
#    - Original `diag` detection assumed real cues are 4-disconnected from the
#      network. Broken here: the informative cue pixels are now attached leaves,
#      so the detector must look at `deg4==1`, not diagonal isolation.
#    - Original gap filling relied on deleting diagonal chains while leaving the
#      broken line endpoints untouched. Broken here unless the generator keeps
#      those endpoints at degree 2 in the input. The generator therefore adds a
#      leaf on each side of every gap so the endpoints are not misdetected.
#    - Original local pruning assumed only hint-adjacent clutter should shrink.
#      Broken here by design: far-away branches must also disappear, so pruning
#      must scan the whole grid repeatedly.
#    - New solver assumption:
#      * after deleting the initial degree-1 hints, any remaining true structure
#        that should survive must be recoverable by filling only 1-3 cell gaps.
#      Generator guarantee:
#      * all intended repairs are exactly 3 cells long and marked by nearby
#        deleted leaves; no other short accidental gaps are created.
#
# d) Placement legality
#    - Components are placed as non-overlapping rectangles with a safety margin
#      that leaves room for outward leaves, branches, and distractors.
#    - Gap cues use only in-bounds background cells just outside a component.
#    - Extra branches grow outward from component edges and are kept away from
#      gap neighborhoods so they cannot act as false repair cues.
#    - Floating domino distractors are placed in open background with a blank
#      Chebyshev-1 moat, so they do not touch the core or each other.
#
# e) Visual observability
#    - A human can see three stable phenomena across a few examples:
#      * endpoint-adjacent leaf marks disappear and a 3-cell straight break is
#        restored,
#      * unrelated branches elsewhere are shaved off completely,
#      * the final output contains only the compact leafless orthogonal core.
#    - No invisible ordering by color id is used; everything is driven by degree
#      and visible local geometry.
#
# f) Visual cleanliness
#    - The core is built from rectangles and straight chords only, so the output
#      remains tidy.
#    - Gap cues sit immediately outside a broken side; branches and floating
#      distractors are sparse and separated, avoiding crowded overlays.
#
# g) New solver assumptions checked against the generator
#    - Assumption: the puzzle has one foreground color. Guaranteed.
#    - Assumption: every intended repair gap is horizontal or vertical and has
#      length at most 3. Guaranteed.
#    - Assumption: at least one non-gap branch survives the initial hint deletion
#      long enough for global pruning to matter. Guaranteed by forcing one or
#      more branches of length >= 2 outside all cue zones.

GRID_H_RANGE = (18, 29)
GRID_W_RANGE = (18, 30)
N_COLORS = 10

COMPONENT_COUNT_CHOICES = (2, 3)
RECT_H_RANGE = (7, 10)
RECT_W_RANGE = (7, 11)
OUTER_MARGIN = 4
COMPONENT_SPACING = 3

GAP_SIZE = 3
GAP_COUNT_CHOICES = (2, 3)

BRANCH_LEN_RANGE = (2, 3)
BRANCH_COUNT_CHOICES = (1, 2, 3)

FLOATING_COUNT_CHOICES = (0, 1, 2)
GEN_RETRIES = 800


def _deg4(mask: np.ndarray, r: int, c: int) -> int:
    h, w = mask.shape
    deg = 0
    if r > 0 and bool(mask[r - 1, c]):
        deg += 1
    if r + 1 < h and bool(mask[r + 1, c]):
        deg += 1
    if c > 0 and bool(mask[r, c - 1]):
        deg += 1
    if c + 1 < w and bool(mask[r, c + 1]):
        deg += 1
    return int(deg)


def _runs(indices: np.ndarray) -> list[tuple[int, int]]:
    if indices.size == 0:
        return []
    vals = [int(x) for x in indices]
    out: list[tuple[int, int]] = []
    start = vals[0]
    prev = vals[0]
    for cur in vals[1:]:
        if cur == prev + 1:
            prev = cur
        else:
            out.append((int(start), int(prev)))
            start = prev = cur
    out.append((int(start), int(prev)))
    return out


def _hint_mask(mask: np.ndarray) -> np.ndarray:
    h, w = mask.shape
    hint = np.zeros((h, w), dtype=bool)
    for r in range(h):
        for c in range(w):
            if bool(mask[r, c]) and _deg4(mask, int(r), int(c)) == 1:
                hint[r, c] = True
    return hint


def _fill_guided_gaps(base: np.ndarray, hint: np.ndarray) -> np.ndarray:
    h, w = base.shape
    out = base.copy()

    for r in range(h):
        segs = _runs(np.where(base[r, :])[0])
        for (_a1, b1), (a2, _b2) in zip(segs, segs[1:]):
            gap0 = int(b1 + 1)
            gap1 = int(a2 - 1)
            if gap1 < gap0:
                continue
            if int(gap1 - gap0 + 1) > int(GAP_SIZE):
                continue
            rr0 = max(0, int(r - 1))
            rr1 = min(int(h - 1), int(r + 1))
            cc0 = max(0, int(gap0 - 1))
            cc1 = min(int(w - 1), int(gap1 + 1))
            if bool(hint[rr0 : rr1 + 1, cc0 : cc1 + 1].any()):
                out[r, gap0 : gap1 + 1] = True

    for c in range(w):
        segs = _runs(np.where(base[:, c])[0])
        for (_a1, b1), (a2, _b2) in zip(segs, segs[1:]):
            gap0 = int(b1 + 1)
            gap1 = int(a2 - 1)
            if gap1 < gap0:
                continue
            if int(gap1 - gap0 + 1) > int(GAP_SIZE):
                continue
            rr0 = max(0, int(gap0 - 1))
            rr1 = min(int(h - 1), int(gap1 + 1))
            cc0 = max(0, int(c - 1))
            cc1 = min(int(w - 1), int(c + 1))
            if bool(hint[rr0 : rr1 + 1, cc0 : cc1 + 1].any()):
                out[gap0 : gap1 + 1, c] = True

    return out


def _global_prune_deg1(mask: np.ndarray) -> np.ndarray:
    m = mask.copy()
    h, w = m.shape
    changed = True
    while changed:
        changed = False
        to_clear: list[tuple[int, int]] = []
        for r in range(h):
            for c in range(w):
                if bool(m[r, c]) and _deg4(m, int(r), int(c)) == 1:
                    to_clear.append((int(r), int(c)))
        if to_clear:
            changed = True
            for r, c in to_clear:
                m[r, c] = False
    return m


def _solve_mask(mask: np.ndarray) -> np.ndarray:
    hint = _hint_mask(mask)
    base = mask & (~hint)
    repaired = _fill_guided_gaps(base, hint)
    return _global_prune_deg1(repaired)


def main(grid: np.ndarray) -> np.ndarray:
    """
    Delete degree-1 cue pixels, repair the short orthogonal gaps they signal, and then peel all remaining leaves globally.

    Input structure:
    - One background color and one foreground color.
    - The foreground forms orthogonal loop-like cores, plus some temporary degree-1 leaves and short outward branches.
    - Some 3-cell breaks on straight sides are marked by nearby leaf pixels instead of detached diagonal staircases.

    Rule:
    - Mark every foreground pixel with exactly one orthogonal foreground neighbor as a cue/noise leaf and delete it.
    - Fill horizontal or vertical gaps of length 1-3 when such deleted leaves appear in the gap's immediate neighborhood.
    - After repairing those gaps, iteratively prune degree-1 foreground pixels across the entire grid until no leaves remain.

    Key insight:
    - In this variant, the useful hints are not diagonal-only islands; they are the degree-1 leaves themselves, and the final answer is the repaired 2-core-like skeleton of the network.
    """
    g = np.array(grid, dtype=int)
    bg = int(get_bg(g))

    colors = {int(x) for x in np.unique(g)}
    colors.discard(int(bg))
    assert colors, "input must contain a foreground color"
    fg = int(next(iter(colors)))

    solved = _solve_mask(g == int(fg))
    out = np.full(g.shape, int(bg), dtype=int)
    out[solved] = int(fg)
    return out


def _draw_component(grid: np.ndarray, fg: int, comp: dict) -> None:
    r0 = int(comp["r0"])
    c0 = int(comp["c0"])
    r1 = int(comp["r1"])
    c1 = int(comp["c1"])
    draw_line(grid, r0, c0, end_r=r0, end_c=c1, color=fg)
    draw_line(grid, r1, c0, end_r=r1, end_c=c1, color=fg)
    draw_line(grid, r0, c0, end_r=r1, end_c=c0, color=fg)
    draw_line(grid, r0, c1, end_r=r1, end_c=c1, color=fg)
    for col in comp["v_chords"]:
        draw_line(grid, r0, int(col), end_r=r1, end_c=int(col), color=fg)
    for row in comp["h_chords"]:
        draw_line(grid, int(row), c0, end_r=int(row), end_c=c1, color=fg)


def _boxes_far(a: tuple[int, int, int, int], b: tuple[int, int, int, int], pad: int) -> bool:
    ar0, ac0, ar1, ac1 = a
    br0, bc0, br1, bc1 = b
    return ar1 + pad < br0 or br1 + pad < ar0 or ac1 + pad < bc0 or bc1 + pad < ac0


def _make_component(rng: random.Random, rmin: int, rmax: int, cmin: int, cmax: int, force_chord: bool) -> dict | None:
    region_h = int(rmax - rmin + 1)
    region_w = int(cmax - cmin + 1)
    if region_h < RECT_H_RANGE[0] or region_w < RECT_W_RANGE[0]:
        return None
    hh = rng.randint(RECT_H_RANGE[0], min(RECT_H_RANGE[1], region_h))
    ww = rng.randint(RECT_W_RANGE[0], min(RECT_W_RANGE[1], region_w))
    r0 = rng.randint(rmin, rmax - hh + 1)
    c0 = rng.randint(cmin, cmax - ww + 1)
    r1 = int(r0 + hh - 1)
    c1 = int(c0 + ww - 1)

    v_chords: list[int] = []
    h_chords: list[int] = []
    if ww >= 9 and (force_chord or rng.random() < 0.6):
        v_chords.append(rng.randint(c0 + 2, c1 - 2))
    if hh >= 9 and rng.random() < 0.5:
        row = rng.randint(r0 + 2, r1 - 2)
        if row not in h_chords:
            h_chords.append(row)
    return {
        "r0": int(r0),
        "c0": int(c0),
        "r1": int(r1),
        "c1": int(c1),
        "v_chords": sorted(int(x) for x in v_chords),
        "h_chords": sorted(int(x) for x in h_chords),
    }


def _sample_components(rng: random.Random, h: int, w: int) -> list[dict] | None:
    split_mode = rng.choice(["horizontal", "vertical"])
    force_chord = True

    if split_mode == "horizontal":
        top = _make_component(
            rng,
            OUTER_MARGIN,
            max(OUTER_MARGIN, h // 2 - COMPONENT_SPACING - 1),
            OUTER_MARGIN,
            w - OUTER_MARGIN - 1,
            force_chord,
        )
        if top is None:
            return None
        force_chord = not (top["v_chords"] or top["h_chords"])
        bottom = _make_component(
            rng,
            min(h - OUTER_MARGIN - 1, h // 2 + COMPONENT_SPACING + 1),
            h - OUTER_MARGIN - 1,
            OUTER_MARGIN,
            w - OUTER_MARGIN - 1,
            force_chord,
        )
        if bottom is None:
            return None
        comps = [top, bottom]
    else:
        left = _make_component(
            rng,
            OUTER_MARGIN,
            h - OUTER_MARGIN - 1,
            OUTER_MARGIN,
            max(OUTER_MARGIN, w // 2 - COMPONENT_SPACING - 1),
            force_chord,
        )
        if left is None:
            return None
        force_chord = not (left["v_chords"] or left["h_chords"])
        right = _make_component(
            rng,
            OUTER_MARGIN,
            h - OUTER_MARGIN - 1,
            min(w - OUTER_MARGIN - 1, w // 2 + COMPONENT_SPACING + 1),
            w - OUTER_MARGIN - 1,
            force_chord,
        )
        if right is None:
            return None
        comps = [left, right]

    if not _boxes_far(
        (int(comps[0]["r0"]), int(comps[0]["c0"]), int(comps[0]["r1"]), int(comps[0]["c1"])),
        (int(comps[1]["r0"]), int(comps[1]["c0"]), int(comps[1]["r1"]), int(comps[1]["c1"])),
        COMPONENT_SPACING,
    ):
        return None
    if force_chord and comps and comps[0]["c1"] - comps[0]["c0"] >= 4:
        comps[0]["v_chords"] = [int((comps[0]["c0"] + comps[0]["c1"]) // 2)]
    return comps


def _component_mask(h: int, w: int, comps: list[dict]) -> np.ndarray:
    grid = np.zeros((h, w), dtype=bool)
    for comp in comps:
        tmp = np.zeros((h, w), dtype=int)
        _draw_component(tmp, 1, comp)
        grid |= tmp == 1
    return grid


def _gap_candidates(comp: dict, mask: np.ndarray) -> list[dict]:
    r0 = int(comp["r0"])
    c0 = int(comp["c0"])
    r1 = int(comp["r1"])
    c1 = int(comp["c1"])
    v_chords = set(int(x) for x in comp["v_chords"])
    h_chords = set(int(x) for x in comp["h_chords"])
    out: list[dict] = []

    for start in range(c0 + 2, c1 - GAP_SIZE):
        if any(start - 1 <= vc <= start + GAP_SIZE for vc in v_chords):
            continue
        left = int(start - 1)
        right = int(start + GAP_SIZE)
        cues = [(r0 - 1, left), (r0 - 1, right)]
        if all(not bool(mask[rr, cc]) for rr, cc in cues):
            out.append({"kind": "top", "start": int(start), "cues": cues})
        cues = [(r1 + 1, left), (r1 + 1, right)]
        if all(not bool(mask[rr, cc]) for rr, cc in cues):
            out.append({"kind": "bottom", "start": int(start), "cues": cues})

    for start in range(r0 + 2, r1 - GAP_SIZE):
        if any(start - 1 <= hr <= start + GAP_SIZE for hr in h_chords):
            continue
        top = int(start - 1)
        bottom = int(start + GAP_SIZE)
        cues = [(top, c0 - 1), (bottom, c0 - 1)]
        if all(not bool(mask[rr, cc]) for rr, cc in cues):
            out.append({"kind": "left", "start": int(start), "cues": cues})
        cues = [(top, c1 + 1), (bottom, c1 + 1)]
        if all(not bool(mask[rr, cc]) for rr, cc in cues):
            out.append({"kind": "right", "start": int(start), "cues": cues})

    return out


def _apply_gap(inp: np.ndarray, fg: int, bg: int, comp: dict, gap: dict) -> set[tuple[int, int]]:
    r0 = int(comp["r0"])
    c0 = int(comp["c0"])
    r1 = int(comp["r1"])
    c1 = int(comp["c1"])
    start = int(gap["start"])
    reserved: set[tuple[int, int]] = set()

    if gap["kind"] == "top":
        inp[r0, start : start + GAP_SIZE] = int(bg)
    elif gap["kind"] == "bottom":
        inp[r1, start : start + GAP_SIZE] = int(bg)
    elif gap["kind"] == "left":
        inp[start : start + GAP_SIZE, c0] = int(bg)
    else:
        inp[start : start + GAP_SIZE, c1] = int(bg)

    for rr, cc in gap["cues"]:
        inp[int(rr), int(cc)] = int(fg)
    for rr, cc in gap["cues"]:
        for dr in (-2, -1, 0, 1, 2):
            for dc in (-2, -1, 0, 1, 2):
                reserved.add((int(rr + dr), int(cc + dc)))
    return reserved


def _branch_candidates(comp: dict, mask: np.ndarray, reserved: set[tuple[int, int]]) -> list[list[tuple[int, int]]]:
    r0 = int(comp["r0"])
    c0 = int(comp["c0"])
    r1 = int(comp["r1"])
    c1 = int(comp["c1"])
    out: list[list[tuple[int, int]]] = []

    def clear(cells: list[tuple[int, int]]) -> bool:
        h, w = mask.shape
        for rr, cc in cells:
            if not (0 <= rr < h and 0 <= cc < w):
                return False
            if bool(mask[rr, cc]):
                return False
            if (int(rr), int(cc)) in reserved:
                return False
        return True

    for col in range(c0 + 2, c1 - 1):
        for length in range(BRANCH_LEN_RANGE[0], BRANCH_LEN_RANGE[1] + 1):
            cells = [(r0 - k, col) for k in range(1, length + 1)]
            if clear(cells):
                out.append([(int(r), int(c)) for r, c in cells])
            cells = [(r1 + k, col) for k in range(1, length + 1)]
            if clear(cells):
                out.append([(int(r), int(c)) for r, c in cells])

    for row in range(r0 + 2, r1 - 1):
        for length in range(BRANCH_LEN_RANGE[0], BRANCH_LEN_RANGE[1] + 1):
            cells = [(row, c0 - k) for k in range(1, length + 1)]
            if clear(cells):
                out.append([(int(r), int(c)) for r, c in cells])
            cells = [(row, c1 + k) for k in range(1, length + 1)]
            if clear(cells):
                out.append([(int(r), int(c)) for r, c in cells])

    return out


def _branch_conflicts(candidate: list[tuple[int, int]], branches: list[list[tuple[int, int]]]) -> bool:
    cand_rows = {int(r) for r, _c in candidate}
    cand_cols = {int(c) for _r, c in candidate}
    cand_vertical = len(cand_cols) == 1

    if cand_vertical:
        cand_col = int(next(iter(cand_cols)))
        cand_r0 = min(cand_rows)
        cand_r1 = max(cand_rows)
        for other in branches:
            other_cols = {int(c) for _r, c in other}
            if len(other_cols) != 1:
                continue
            other_rows = {int(r) for r, _c in other}
            other_col = int(next(iter(other_cols)))
            if max(cand_r0, min(other_rows)) > min(cand_r1, max(other_rows)):
                continue
            if abs(cand_col - other_col) - 1 <= GAP_SIZE:
                return True
        return False

    cand_row = int(next(iter(cand_rows)))
    cand_c0 = min(cand_cols)
    cand_c1 = max(cand_cols)
    for other in branches:
        other_rows = {int(r) for r, _c in other}
        if len(other_rows) != 1:
            continue
        other_cols = {int(c) for _r, c in other}
        other_row = int(next(iter(other_rows)))
        if max(cand_c0, min(other_cols)) > min(cand_c1, max(other_cols)):
            continue
        if abs(cand_row - other_row) - 1 <= GAP_SIZE:
            return True
    return False


def _reserve_moat(cells: list[tuple[int, int]], reserved: set[tuple[int, int]]) -> None:
    for r, c in cells:
        for dr in (-1, 0, 1):
            for dc in (-1, 0, 1):
                reserved.add((int(r + dr), int(c + dc)))


def _floating_domino_candidates(mask: np.ndarray, reserved: set[tuple[int, int]]) -> list[list[tuple[int, int]]]:
    h, w = mask.shape
    out: list[list[tuple[int, int]]] = []
    for r in range(1, h - 1):
        for c in range(1, w - 1):
            for cells in ([(r, c), (r, c + 1)], [(r, c), (r + 1, c)]):
                ok = True
                cell_set = {(int(rr), int(cc)) for rr, cc in cells}
                for rr, cc in cells:
                    if bool(mask[int(rr), int(cc)]) or (int(rr), int(cc)) in reserved:
                        ok = False
                        break
                    for dr in (-1, 0, 1):
                        for dc in (-1, 0, 1):
                            pos = (int(rr + dr), int(cc + dc))
                            if pos in cell_set:
                                continue
                            if 0 <= pos[0] < h and 0 <= pos[1] < w and bool(mask[pos[0], pos[1]]):
                                ok = False
                                break
                        if not ok:
                            break
                    if not ok:
                        break
                if ok:
                    out.append([(int(rr), int(cc)) for rr, cc in cells])
    return out


def _output_has_no_leaves(mask: np.ndarray) -> bool:
    for r, c in np.argwhere(mask):
        if _deg4(mask, int(r), int(c)) < 2:
            return False
    return True


def _gen_attempt(debug: bool = False) -> tuple[np.ndarray, np.ndarray] | str | None:
    def fail(reason: str) -> str | None:
        if debug:
            return reason
        return None

    rng = random.Random()
    h = rng.randint(*GRID_H_RANGE)
    w = rng.randint(*GRID_W_RANGE)
    bg = rng.randrange(N_COLORS)
    fg = rng.choice([c for c in range(N_COLORS) if c != bg])

    comps = _sample_components(rng, h, w)
    if comps is None:
        return fail("sample_components")

    out = np.full((h, w), int(bg), dtype=int)
    for comp in comps:
        _draw_component(out, fg, comp)
    out_mask = out == int(fg)
    if not _output_has_no_leaves(out_mask):
        return fail("output_has_leaves")

    inp = out.copy()
    gap_count = rng.choice(GAP_COUNT_CHOICES)
    gap_pool: list[tuple[int, dict]] = []
    for idx, comp in enumerate(comps):
        for gap in _gap_candidates(comp, out_mask):
            gap_pool.append((int(idx), gap))
    rng.shuffle(gap_pool)
    chosen_gaps: list[tuple[int, dict]] = []
    used_components: dict[int, int] = {}
    for idx, gap in gap_pool:
        kind = str(gap["kind"])
        start = int(gap["start"])
        comp = comps[idx]
        conflict = False
        for prev_idx, prev_gap in chosen_gaps:
            if idx != prev_idx:
                continue
            if kind == prev_gap["kind"] and abs(start - int(prev_gap["start"])) <= GAP_SIZE + 2:
                conflict = True
                break
        if conflict:
            continue
        chosen_gaps.append((int(idx), gap))
        used_components[idx] = used_components.get(idx, 0) + 1
        if len(chosen_gaps) == gap_count:
            break
    if len(chosen_gaps) != gap_count:
        return fail("chosen_gaps")

    reserved: set[tuple[int, int]] = set()
    for idx, gap in chosen_gaps:
        reserved |= _apply_gap(inp, fg, bg, comps[int(idx)], gap)

    cur_mask = inp == int(fg)
    branch_count = rng.choice(BRANCH_COUNT_CHOICES)
    branch_pool: list[list[tuple[int, int]]] = []
    for comp in comps:
        branch_pool.extend(_branch_candidates(comp, cur_mask, reserved))
    rng.shuffle(branch_pool)
    branches: list[list[tuple[int, int]]] = []
    for cells in branch_pool:
        if any(cell in reserved for cell in cells):
            continue
        if _branch_conflicts(cells, branches):
            continue
        branches.append(cells)
        _reserve_moat(cells, reserved)
        for r, c in cells:
            inp[int(r), int(c)] = int(fg)
        if len(branches) == branch_count:
            break
    if len(branches) == 0:
        return fail("branches")

    cur_mask = inp == int(fg)
    float_count = rng.choice(FLOATING_COUNT_CHOICES)
    if float_count > 0:
        float_pool = _floating_domino_candidates(cur_mask, reserved)
        rng.shuffle(float_pool)
        placed_float = 0
        for cells in float_pool:
            if any(cell in reserved for cell in cells):
                continue
            for r, c in cells:
                inp[int(r), int(c)] = int(fg)
            _reserve_moat(cells, reserved)
            placed_float += 1
            if placed_float == float_count:
                break

    in_mask = inp == int(fg)
    hint = _hint_mask(in_mask)
    base = in_mask & (~hint)
    repaired = _fill_guided_gaps(base, hint)
    final_mask = _global_prune_deg1(repaired)

    # Fusion-behavior guards:
    # - at least one gap was actually refilled by degree-1 hints
    # - at least one extra pixel was removed only by the later global pruning
    if int(np.count_nonzero(repaired & (~base))) < GAP_SIZE:
        return fail("no_gap_refill")
    if int(np.count_nonzero(repaired)) <= int(np.count_nonzero(final_mask)):
        return fail("no_global_prune")

    if not np.array_equal(final_mask, out_mask):
        return fail("final_mask_mismatch")
    if not np.array_equal(main(inp), out):
        return fail("main_mismatch")
    return inp, out


def generate_input() -> np.ndarray:
    for _ in range(GEN_RETRIES):
        built = _gen_attempt()
        if built is not None:
            inp, _out = built
            return inp
    raise RuntimeError("failed to generate valid input for 7b80bb43 combo D7/1+D12/1")
