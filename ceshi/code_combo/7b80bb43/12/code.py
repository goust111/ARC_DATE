# Combo variant of 7b80bb43 (index 12)
# combo_spec: D11/1+D17/0
# Name: 多色图例导航网络

from __future__ import annotations

import random

import numpy as np

from arc_utils import components_mask, draw_line, find_components, get_bg


# Design analysis for the fused variant
# -------------------------------------
# a) Concrete behavior and trace
#    - Original task: one foreground color does double duty. The same-color 45-degree
#      staircase both marks a broken line and also has to be recognized as removable
#      noise from topology alone.
#    - Fused task: the top edge now contains a 2-row legend. Each legend entry is a
#      vertical color pair: top block = hint color, bottom block = line color. Below a
#      blank separator row sits the multi-color line network.
#    - Local trace:
#      original: a blue-less world, gap on one line, nearby same-color diagonal chain,
#      solver infers "these diagonal fg pixels belong to the broken line".
#      fused: suppose the legend shows pink -> yellow and teal -> red. In the core
#      region, a pink staircase near a yellow line gap repairs yellow only; a teal
#      staircase repairs red only. The stair color itself disappears in the output.
#    - Difference: the repair trigger is no longer "same color as the line". It is
#      "match the line color indicated by the legend for this hint color".
#
# b) Complexity calibration
#    - The topology, gap size, and distractor style stay close to the source task, so
#      the puzzle still reads as broken line restoration rather than a new geometry task.
#    - To make the color mapping learnable without becoming noisy, the generator uses
#      exactly one legend row per connected component. That yields 3-4 color pairs:
#      enough repetition to see a legend, but not so many that the board turns into a
#      palette dump.
#    - Every legend entry is exercised by at least one real gap, and the chosen gaps
#      include both horizontal and vertical repairs across the sample set.
#    - At least one extra staircase per sample intentionally uses the "wrong" legend
#      color next to another component's gap. That gap stays open in the output, which
#      makes the legend operational rather than decorative.
#
# c) Hidden-assumption audit against the original solver
#    - `fg = next(iter(colors))` assumed a single semantic foreground color. Broken:
#      here colors split into line colors and hint colors plus legend colors.
#    - `diag` was inferred from 4-neighbor degree inside the line color. Broken:
#      hint chains are now explicitly tagged by their own color, so topology-based hint
#      detection is unnecessary and would miss them entirely.
#    - The old solver processed the whole grid uniformly. Broken: the top 2 rows are
#      instruction symbols, not part of the line network, so they must be masked out and
#      preserved unchanged.
#    - The old generator could reuse one fg for all components. Broken: if two distant
#      components shared a line color, the per-color repair pass could create ambiguous
#      interactions. This fused generator therefore gives each 4-connected component a
#      unique line color and a unique hint color.
#
# d) Placement legality
#    - The generator still builds a clean axis-aligned network first using the original
#      output-first layout routine, so edge anchoring and T-junction constraints remain
#      valid.
#    - One legal gap plan is selected per component. Each chosen hint chain stays in
#      bounds, sits on background, is orthogonally separated from all lines, and its
#      Chebyshev-1 neighborhood is reserved so hint chains do not visually collide.
#    - Distractor chains are placed only in open background away from the network and
#      away from the legend band.
#
# e) Visual observability
#    - Humans can directly read the top legend as a repeated table of color pairs.
#    - The same hint color reappears near a broken line of the mapped line color, and
#      one wrong-color staircase is also shown beside a different broken line. Across
#      examples, that contrast makes the legend visibly do work.
#    - Because the output preserves the legend while deleting hint chains and repairing
#      only the mapped line colors, the control flow is visible in 2-3 examples.
#
# f) Cleanliness
#    - The legend is confined to the top-left edge and separated from the network by one
#      blank row, so symbols do not overlap the main geometry.
#    - Hint chains and distractors use thin diagonals only; all repaired structure in
#      the output remains rectilinear.
#
# g) New solver assumptions and generator guarantees
#    - Assumption: the legend occupies the top two rows and each entry is a contiguous
#      run whose top row is constant hint color and bottom row is constant line color.
#      Guarantee: `_wrap_with_legend` renders exactly that format.
#    - Assumption: every non-background color in the core region is either a legend-mapped
#      line color or a legend-mapped hint color. Guarantee: the generator uses no other
#      core colors.
#    - Assumption: one repair color pair should not overlap another in the output.
#      Guarantee: every component has a unique line color and components are disjoint.


LEGEND_ROWS = 2
SEP_ROWS = 1
TOP_MARGIN = LEGEND_ROWS + SEP_ROWS
LEGEND_ENTRY_WIDTH = 2
LEGEND_ENTRY_GAP = 1

GRID_H_RANGE = (16, 29)
GRID_W_RANGE = (17, 30)
N_COLORS = 10

VALID_CONFIGS = [
    {"segments": 6, "h": 2, "v": 4, "junctions": 2},  # 4 components
    {"segments": 6, "h": 3, "v": 3, "junctions": 2},  # 4 components
    {"segments": 6, "h": 3, "v": 3, "junctions": 3},  # 3 components
    {"segments": 7, "h": 3, "v": 4, "junctions": 3},  # 4 components
    {"segments": 7, "h": 4, "v": 3, "junctions": 3},
]
T_SLOTS = ("LT", "RT", "LB", "RB")

EDGE_MARGIN = 3
ROW_COL_GAP = 2
T_BRANCH_OFFSET_MIN = 3
T_BRANCH_OFFSET_MAX = 9
H_SEG_END_MARGIN = 4
H_SEG_MAX_END = 25

SIMPLE_H_LEN_MIN = 7
SIMPLE_H_LEN_MAX = 11
SIMPLE_V_MARGIN_NEAR = 7
SIMPLE_V_MARGIN_FAR = 3

GAP_SIZE = 3
CHAIN_LENGTH = 2
GAP_BLOCKED_LO = 2
GAP_BLOCKED_HI = 4
GAP_SEG_INSET = 2

DISTRACTOR_LENGTHS = (3, 2)
DISTRACTOR_MARGIN = 1
GEN_RETRIES = 1500


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


def _has_diag(mask: np.ndarray, r: int, c: int) -> bool:
    h, w = mask.shape
    for dr in (-1, 1):
        for dc in (-1, 1):
            rr = int(r + dr)
            cc = int(c + dc)
            if 0 <= rr < h and 0 <= cc < w and bool(mask[rr, cc]):
                return True
    return False


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


def _legend_width(n_entries: int) -> int:
    return int(n_entries * LEGEND_ENTRY_WIDTH + max(0, n_entries - 1) * LEGEND_ENTRY_GAP)


def _parse_legend(g: np.ndarray, bg: int) -> list[tuple[int, int]]:
    pairs: list[tuple[int, int]] = []
    c = 0
    w = g.shape[1]
    while c < w:
        top = int(g[0, c])
        bot = int(g[1, c])
        if top == int(bg) or bot == int(bg):
            c += 1
            continue
        c1 = int(c)
        while c1 + 1 < w and int(g[0, c1 + 1]) == top and int(g[1, c1 + 1]) == bot:
            c1 += 1
        pairs.append((int(top), int(bot)))
        c = int(c1 + 1)
    assert pairs, "legend missing"
    hints = [h for h, _l in pairs]
    lines = [l for _h, l in pairs]
    assert len(set(hints)) == len(hints), "hint colors must be unique in legend"
    assert len(set(lines)) == len(lines), "line colors must be unique in legend"
    return pairs


def _solve_color_pair(core: np.ndarray, bg: int, line_color: int, hint_color: int) -> np.ndarray:
    line_mask = core == int(line_color)
    hint_mask = core == int(hint_color)
    h, w = core.shape

    near = np.zeros((h, w), dtype=bool)
    for r, c in np.argwhere(hint_mask):
        for dr in (-1, 0, 1):
            for dc in (-1, 0, 1):
                rr = int(r + dr)
                cc = int(c + dc)
                if 0 <= rr < h and 0 <= cc < w:
                    near[rr, cc] = True

    out = line_mask.copy()
    max_gap = GAP_SIZE

    for r in range(h):
        segs = _runs(np.where(line_mask[r, :])[0])
        for (_a1, b1), (a2, _b2) in zip(segs, segs[1:]):
            gap0 = int(b1 + 1)
            gap1 = int(a2 - 1)
            if gap1 < gap0 or int(gap1 - gap0 + 1) > int(max_gap):
                continue
            rr0 = max(0, int(r - 1))
            rr1 = min(int(h - 1), int(r + 1))
            cc0 = max(0, int(gap0 - 1))
            cc1 = min(int(w - 1), int(gap1 + 1))
            if bool(hint_mask[rr0 : rr1 + 1, cc0 : cc1 + 1].any()):
                out[r, gap0 : gap1 + 1] = True

    for c in range(w):
        segs = _runs(np.where(line_mask[:, c])[0])
        for (_a1, b1), (a2, _b2) in zip(segs, segs[1:]):
            gap0 = int(b1 + 1)
            gap1 = int(a2 - 1)
            if gap1 < gap0 or int(gap1 - gap0 + 1) > int(max_gap):
                continue
            rr0 = max(0, int(gap0 - 1))
            rr1 = min(int(h - 1), int(gap1 + 1))
            cc0 = max(0, int(c - 1))
            cc1 = min(int(w - 1), int(c + 1))
            if bool(hint_mask[rr0 : rr1 + 1, cc0 : cc1 + 1].any()):
                out[gap0 : gap1 + 1, c] = True

    changed = True
    while changed:
        changed = False
        to_clear: list[tuple[int, int]] = []
        for r in range(h):
            for c in range(w):
                if not bool(out[r, c]) or not bool(near[r, c]):
                    continue
                if _deg4(out, int(r), int(c)) != 1:
                    continue
                if not _has_diag(out, int(r), int(c)):
                    continue
                to_clear.append((int(r), int(c)))
        if to_clear:
            changed = True
            for r, c in to_clear:
                out[r, c] = False

    return out


def main(grid: np.ndarray) -> np.ndarray:
    """
    Decode a color legend, erase diagonal hint chains, and restore clean multi-color line networks.

    Input structure:
    - The top edge contains a 2-row legend: each entry's upper block is a hint color and
      the lower block is the matching line color; one blank separator row sits below it.
    - The core region contains 3-4 disjoint axis-aligned line components in multiple colors.
    - Gaps of length 3 are marked by short diagonal hint chains whose color may differ from
      the target line color; some wrong-color staircases may sit beside other broken lines
      and must be ignored except for being deleted.

    Rule:
    - Read the legend as hint-color -> line-color mappings.
    - In the core region, delete every hint-color staircase and repair only the horizontal or
      vertical gaps in the mapped line color that sit next to that hint chain.
    - After filling those short gaps, prune any residual diagonal-adjacent leaf pixels near
      former hint sites, while leaving the legend itself unchanged.

    Key insight:
    - The diagonal pixels are no longer semantically tied to the line by sharing its color.
      Their role is specified explicitly by the edge legend, so the puzzle is solved by
      combining color decoding with the original local gap-repair geometry.
    """
    g = np.array(grid, dtype=int)
    bg = int(get_bg(g))
    assert g.shape[0] > TOP_MARGIN, "grid too short for legend + core"

    legend_pairs = _parse_legend(g, bg)
    core = g[TOP_MARGIN:, :]

    out = g.copy()
    out[:TOP_MARGIN, :] = g[:TOP_MARGIN, :]
    out_core = np.full(core.shape, int(bg), dtype=int)

    for hint_color, line_color in legend_pairs:
        solved_mask = _solve_color_pair(core, bg, int(line_color), int(hint_color))
        assert not np.any(solved_mask & (out_core != int(bg))), "line colors overlap unexpectedly"
        out_core[solved_mask] = int(line_color)

    out[TOP_MARGIN:, :] = out_core
    return out


def _pick_distinct(rng: random.Random, values: list[int], k: int, gap: int = ROW_COL_GAP) -> list[int]:
    vals = values[:]
    rng.shuffle(vals)
    picked: list[int] = []
    for v in vals:
        if all(abs(v - p) >= gap for p in picked):
            picked.append(int(v))
            if len(picked) == k:
                return sorted(picked)
    return []


def _chain_coords(row: int, gap_start: int, anchor: str) -> list[tuple[int, int]]:
    if anchor == "left":
        return [(int(row + i), int(gap_start - 1 + i)) for i in range(1, CHAIN_LENGTH + 1)]
    return [(int(row + i), int(gap_start + GAP_SIZE - i)) for i in range(1, CHAIN_LENGTH + 1)]


def _v_chain_coords(col: int, gap_start: int, anchor: str, dc: int) -> list[tuple[int, int]]:
    if anchor == "top":
        return [(int(gap_start + i), int(col + dc * (i + 1))) for i in range(CHAIN_LENGTH)]
    return [(int(gap_start + GAP_SIZE - 1 - i), int(col + dc * (i + 1))) for i in range(CHAIN_LENGTH)]


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
    coord_set = {(int(r), int(c)) for r, c in coords}
    if any((int(r), int(c)) in occupied for r, c in coord_set):
        return False
    for r, c in coord_set:
        for rr, cc in ((r - 1, c), (r + 1, c), (r, c - 1), (r, c + 1)):
            if (rr, cc) in coord_set:
                return False
            if 0 <= rr < h and 0 <= cc < w and int(grid[rr, cc]) == int(fg):
                return False
        has_diag = False
        for rr, cc in ((r - 1, c - 1), (r - 1, c + 1), (r + 1, c - 1), (r + 1, c + 1)):
            if (rr, cc) in coord_set or (0 <= rr < h and 0 <= cc < w and int(grid[rr, cc]) == int(fg)):
                has_diag = True
                break
        if not has_diag:
            return False
    return True


def _place_t_component(out: np.ndarray, fg: int, slot: str, row: int, branch_col: int, end_col: int) -> tuple[dict, dict]:
    h, w = out.shape
    h_c0 = 0 if slot[0] == "L" else end_col
    h_c1 = end_col if slot[0] == "L" else w - 1
    v_r0 = 0 if slot[1] == "T" else row
    v_r1 = row if slot[1] == "T" else h - 1
    draw_line(out, row, h_c0, end_r=row, end_c=h_c1, color=fg)
    draw_line(out, v_r0, branch_col, end_r=v_r1, end_c=branch_col, color=fg)
    h_seg = {"row": int(row), "c0": int(h_c0), "c1": int(h_c1), "blocked": {int(branch_col)}}
    v_seg = {"col": int(branch_col), "r0": int(v_r0), "r1": int(v_r1), "blocked": {int(row)}}
    return h_seg, v_seg


def _place_simple_h(out: np.ndarray, fg: int, side: str, row: int, length: int) -> dict:
    w = out.shape[1]
    c0 = 0 if side == "left" else w - length
    c1 = length - 1 if side == "left" else w - 1
    draw_line(out, row, c0, end_r=row, end_c=c1, color=fg)
    return {"row": int(row), "c0": int(c0), "c1": int(c1), "blocked": set()}


def _place_simple_v(out: np.ndarray, fg: int, side: str, col: int, length: int) -> dict:
    h = out.shape[0]
    r0 = 0 if side == "top" else h - length
    r1 = length - 1 if side == "top" else h - 1
    draw_line(out, r0, col, end_r=r1, end_c=col, color=fg)
    return {"col": int(col), "r0": int(r0), "r1": int(r1), "blocked": set()}


def _h_gap_options(seg: dict, out: np.ndarray, fg: int, bg: int, occupied: set[tuple[int, int]]) -> list[tuple[int, str]]:
    row, c0, c1 = int(seg["row"]), int(seg["c0"]), int(seg["c1"])
    blocked = {int(x) for x in seg["blocked"]}
    options: list[tuple[int, str]] = []
    for g0 in range(c0 + GAP_SEG_INSET, c1 - GAP_SIZE - 1):
        if any(int(g0 - GAP_BLOCKED_LO) <= b <= int(g0 + GAP_BLOCKED_HI) for b in blocked):
            continue
        temp = out.copy()
        temp[row, g0 : g0 + GAP_SIZE] = int(bg)
        for anchor in ("left", "right"):
            coords = _chain_coords(row, g0, anchor)
            if not _cells_are_color(temp, coords, bg):
                continue
            if not _validate_chain(temp, coords, fg, occupied):
                continue
            options.append((int(g0), anchor))
    return options


def _v_gap_options(seg: dict, out: np.ndarray, fg: int, bg: int, occupied: set[tuple[int, int]]) -> list[tuple[int, str, int]]:
    col, r0, r1 = int(seg["col"]), int(seg["r0"]), int(seg["r1"])
    blocked = {int(x) for x in seg["blocked"]}
    options: list[tuple[int, str, int]] = []
    for g0 in range(r0 + GAP_SEG_INSET, r1 - GAP_SIZE - 1):
        if any(int(g0 - GAP_BLOCKED_LO) <= b <= int(g0 + GAP_BLOCKED_HI) for b in blocked):
            continue
        temp = out.copy()
        temp[g0 : g0 + GAP_SIZE, col] = int(bg)
        for anchor in ("top", "bottom"):
            for dc in (-1, 1):
                coords = _v_chain_coords(col, g0, anchor, dc)
                if not _cells_are_color(temp, coords, bg):
                    continue
                if not _validate_chain(temp, coords, fg, occupied):
                    continue
                options.append((int(g0), anchor, int(dc)))
    return options


def _reserve_neighborhood(cells: set[tuple[int, int]], coords: list[tuple[int, int]]) -> None:
    for r, c in coords:
        for dr in (-1, 0, 1):
            for dc in (-1, 0, 1):
                cells.add((int(r + dr), int(c + dc)))


def _apply_gap(grid: np.ndarray, opt: dict, bg: int) -> None:
    if opt["kind"] == "h":
        row = int(opt["seg"]["row"])
        gap_start = int(opt["gap_start"])
        grid[row, gap_start : gap_start + GAP_SIZE] = int(bg)
    else:
        col = int(opt["seg"]["col"])
        gap_start = int(opt["gap_start"])
        grid[gap_start : gap_start + GAP_SIZE, col] = int(bg)


def _gap_signature(opt: dict) -> tuple[int, int, int]:
    if opt["kind"] == "h":
        return (0, int(opt["seg"]["row"]), int(opt["gap_start"]))
    return (1, int(opt["seg"]["col"]), int(opt["gap_start"]))


def _legend_blind_output(grid: np.ndarray, bg: int) -> np.ndarray:
    g = np.array(grid, dtype=int)
    legend_pairs = _parse_legend(g, bg)
    core = g[TOP_MARGIN:, :]
    line_colors = [int(line_color) for _hint_color, line_color in legend_pairs]
    hint_colors = [int(hint_color) for hint_color, _line_color in legend_pairs]
    any_hint = np.isin(core, hint_colors)

    out = g.copy()
    out_core = np.full(core.shape, int(bg), dtype=int)

    for line_color in line_colors:
        line_mask = core == int(line_color)
        solved = line_mask.copy()
        h, w = core.shape

        for r in range(h):
            segs = _runs(np.where(line_mask[r, :])[0])
            for (_a1, b1), (a2, _b2) in zip(segs, segs[1:]):
                gap0 = int(b1 + 1)
                gap1 = int(a2 - 1)
                if gap1 < gap0 or int(gap1 - gap0 + 1) > int(GAP_SIZE):
                    continue
                rr0 = max(0, int(r - 1))
                rr1 = min(int(h - 1), int(r + 1))
                cc0 = max(0, int(gap0 - 1))
                cc1 = min(int(w - 1), int(gap1 + 1))
                if bool(any_hint[rr0 : rr1 + 1, cc0 : cc1 + 1].any()):
                    solved[r, gap0 : gap1 + 1] = True

        for c in range(w):
            segs = _runs(np.where(line_mask[:, c])[0])
            for (_a1, b1), (a2, _b2) in zip(segs, segs[1:]):
                gap0 = int(b1 + 1)
                gap1 = int(a2 - 1)
                if gap1 < gap0 or int(gap1 - gap0 + 1) > int(GAP_SIZE):
                    continue
                rr0 = max(0, int(gap0 - 1))
                rr1 = min(int(h - 1), int(gap1 + 1))
                cc0 = max(0, int(c - 1))
                cc1 = min(int(w - 1), int(c + 1))
                if bool(any_hint[rr0 : rr1 + 1, cc0 : cc1 + 1].any()):
                    solved[gap0 : gap1 + 1, c] = True

        out_core[solved] = int(line_color)

    out[TOP_MARGIN:, :] = out_core
    return out


def _distractor_options(inp: np.ndarray, bg: int, forbidden: set[tuple[int, int]]) -> list[list[tuple[int, int]]]:
    h, w = inp.shape
    options: list[list[tuple[int, int]]] = []
    for length in DISTRACTOR_LENGTHS:
        for r in range(DISTRACTOR_MARGIN, h - length):
            for c in range(DISTRACTOR_MARGIN, w - DISTRACTOR_MARGIN):
                for direction in (-1, 1):
                    coords = [(int(r + i), int(c + direction * i)) for i in range(length)]
                    if not _cells_are_color(inp, coords, bg):
                        continue
                    if any((int(rr), int(cc)) in forbidden for rr, cc in coords):
                        continue
                    options.append(coords)
    return options


def _choose_decoy_hint(
    rng: random.Random,
    plan: list[dict],
    per_comp: dict[int, list[dict]],
    reserved: set[tuple[int, int]],
    comp_to_hint: dict[int, int],
) -> dict | None:
    used_gaps = {_gap_signature(opt) for opt in plan}
    candidates: list[dict] = []
    for target_comp, opts in per_comp.items():
        wrong_hint_colors = [int(color) for comp_id, color in comp_to_hint.items() if int(comp_id) != int(target_comp)]
        if not wrong_hint_colors:
            continue
        for opt in opts:
            if _gap_signature(opt) in used_gaps:
                continue
            if any((int(r), int(c)) in reserved for r, c in opt["chain"]):
                continue
            decoy = dict(opt)
            decoy["hint_color"] = int(rng.choice(wrong_hint_colors))
            candidates.append(decoy)
    if not candidates:
        return None
    return rng.choice(candidates)


def _build_output(rng: random.Random, h: int, w: int, bg: int, fg: int, cfg: dict) -> tuple[np.ndarray, list[dict], list[dict]] | None:
    out = np.full((h, w), int(bg), dtype=int)
    h_segments: list[dict] = []
    v_segments: list[dict] = []

    h_simple = int(cfg["h"] - cfg["junctions"])
    v_simple = int(cfg["v"] - cfg["junctions"])
    t_slots = rng.sample(list(T_SLOTS), int(cfg["junctions"]))

    top_slots = [s for s in t_slots if s[1] == "T"]
    bottom_slots = [s for s in t_slots if s[1] == "B"]
    top_pool = list(range(EDGE_MARGIN, max(EDGE_MARGIN + 1, h // 2 - 2)))
    bottom_pool = list(range(max(h // 2 + 1, EDGE_MARGIN + 2), h - EDGE_MARGIN))
    top_rows = _pick_distinct(rng, top_pool, len(top_slots))
    bottom_rows = _pick_distinct(rng, bottom_pool, len(bottom_slots))
    if len(top_rows) != len(top_slots) or len(bottom_rows) != len(bottom_slots):
        return None
    row_by_slot = dict(zip(top_slots, top_rows)) | dict(zip(bottom_slots, bottom_rows))

    branch_cols = _pick_distinct(rng, list(range(EDGE_MARGIN, w - EDGE_MARGIN)), len(t_slots))
    if len(branch_cols) != len(t_slots):
        return None
    rng.shuffle(branch_cols)
    branch_by_slot = dict(zip(t_slots, branch_cols))

    for slot in t_slots:
        branch_col = int(branch_by_slot[slot])
        row = int(row_by_slot[slot])
        if slot[0] == "L":
            min_end = max(SIMPLE_H_LEN_MIN + 1, branch_col + T_BRANCH_OFFSET_MIN)
            max_end = min(w - H_SEG_END_MARGIN, branch_col + T_BRANCH_OFFSET_MAX, H_SEG_MAX_END)
        else:
            min_end = max(2, branch_col - T_BRANCH_OFFSET_MAX, w - H_SEG_MAX_END - 1)
            max_end = min(w - SIMPLE_H_LEN_MIN - 1, branch_col - T_BRANCH_OFFSET_MIN)
        if min_end > max_end:
            return None
        end_col = rng.randint(min_end, max_end)
        h_seg, v_seg = _place_t_component(out, fg, slot, row, branch_col, end_col)
        h_segments.append(h_seg)
        v_segments.append(v_seg)

    center_start = max(top_rows + [1]) + 2
    center_end = min(bottom_rows + [h - 2]) - 2
    if h_simple > 0 and center_start > center_end:
        return None

    simple_h_rows = _pick_distinct(rng, list(range(center_start, center_end + 1)), h_simple) if h_simple else []
    if len(simple_h_rows) != h_simple:
        return None
    if h_simple == 0:
        h_sides: list[str] = []
    elif h_simple == 1:
        h_sides = [rng.choice(["left", "right"])]
    else:
        h_sides = ["left", "right"]
        rng.shuffle(h_sides)

    simple_v_cols: list[int] = []
    used_branch_cols = set(branch_by_slot.values())
    if v_simple:
        has_left = "left" in h_sides
        has_right = "right" in h_sides
        if has_left and has_right:
            col_pool = list(range(SIMPLE_V_MARGIN_NEAR, w - SIMPLE_V_MARGIN_NEAR))
        elif has_left:
            col_pool = list(range(SIMPLE_V_MARGIN_NEAR, w - SIMPLE_V_MARGIN_FAR))
        elif has_right:
            col_pool = list(range(2, w - SIMPLE_V_MARGIN_NEAR))
        else:
            col_pool = list(range(2, w - 2))
        col_pool = [int(c) for c in col_pool if int(c) not in used_branch_cols]
        simple_v_cols = _pick_distinct(rng, col_pool, v_simple)
        if len(simple_v_cols) != v_simple:
            return None

    for row, side in zip(simple_h_rows, h_sides):
        if side == "left":
            max_len = (min(simple_v_cols) if simple_v_cols else w - EDGE_MARGIN) - 1
        else:
            max_len = w - (max(simple_v_cols) if simple_v_cols else 2) - 2
        max_len = min(SIMPLE_H_LEN_MAX, max_len)
        if max_len < SIMPLE_H_LEN_MIN:
            return None
        h_segments.append(_place_simple_h(out, fg, side, int(row), rng.randint(SIMPLE_H_LEN_MIN, max_len)))

    top_v = min(2, v_simple)
    top_v_cols = simple_v_cols[:top_v]
    bottom_v_cols = simple_v_cols[top_v:]
    top_limit = min(min(top_rows) - 2 if top_rows else center_start - 2, center_start - 2)
    bottom_start = max(max(bottom_rows) + 2 if bottom_rows else center_end + 2, center_end + 2)
    for col in top_v_cols:
        if top_limit < 2:
            return None
        v_segments.append(_place_simple_v(out, fg, "top", int(col), rng.randint(2, top_limit)))
    for col in bottom_v_cols:
        max_len = h - bottom_start
        if max_len < 2:
            return None
        v_segments.append(_place_simple_v(out, fg, "bottom", int(col), rng.randint(2, max_len)))

    comps = find_components(out, bg=bg, connectivity=4, monochromatic=True, return_format="dict")
    n_comp = len(comps)
    if n_comp < 3 or n_comp > 4:
        return None
    if n_comp != int(cfg["segments"] - cfg["junctions"]):
        return None
    return out, h_segments, v_segments


def _component_labels(clean: np.ndarray, bg: int) -> tuple[list[dict], np.ndarray]:
    comps = find_components(clean, bg=bg, connectivity=4, monochromatic=True, return_format="dict")
    comps = sorted(
        comps,
        key=lambda comp: (
            int(comp["bbox"][0]),
            int(comp["bbox"][1]),
            int(comp["bbox"][2]),
            int(comp["bbox"][3]),
        ),
    )
    labels = np.zeros(clean.shape, dtype=int)
    for idx, comp in enumerate(comps, start=1):
        for r, c in comp["coords"]:
            labels[int(r), int(c)] = int(idx)
    return comps, labels


def _segment_component_id(seg: dict, kind: str, labels: np.ndarray) -> int:
    if kind == "h":
        vals = set(int(x) for x in labels[int(seg["row"]), int(seg["c0"]) : int(seg["c1"]) + 1] if int(x) != 0)
    else:
        vals = set(int(x) for x in labels[int(seg["r0"]) : int(seg["r1"]) + 1, int(seg["col"])] if int(x) != 0)
    assert len(vals) == 1, "segment must belong to exactly one component"
    return int(next(iter(vals)))


def _choose_gap_plan(rng: random.Random, per_comp: dict[int, list[dict]]) -> list[dict] | None:
    comp_ids = sorted(per_comp, key=lambda cid: (len(per_comp[cid]), cid))

    def rec(idx: int, reserved: set[tuple[int, int]], chosen: list[dict], n_h: int, n_v: int) -> list[dict] | None:
        if idx == len(comp_ids):
            if n_h >= 1 and n_v >= 1:
                return chosen[:]
            return None

        comp_id = int(comp_ids[idx])
        opts = per_comp[comp_id][:]
        rng.shuffle(opts)
        for opt in opts:
            if any((int(r), int(c)) in reserved for r, c in opt["chain"]):
                continue
            next_reserved = set(reserved)
            _reserve_neighborhood(next_reserved, opt["chain"])
            chosen.append(opt)
            found = rec(
                idx + 1,
                next_reserved,
                chosen,
                n_h + (1 if opt["kind"] == "h" else 0),
                n_v + (1 if opt["kind"] == "v" else 0),
            )
            if found is not None:
                return found
            chosen.pop()
        return None

    return rec(0, set(), [], 0, 0)


def _wrap_with_legend(core: np.ndarray, bg: int, pairs: list[tuple[int, int]]) -> np.ndarray:
    full = np.full((int(core.shape[0] + TOP_MARGIN), int(core.shape[1])), int(bg), dtype=int)
    c = 0
    for hint_color, line_color in pairs:
        full[0, c : c + LEGEND_ENTRY_WIDTH] = int(hint_color)
        full[1, c : c + LEGEND_ENTRY_WIDTH] = int(line_color)
        c += LEGEND_ENTRY_WIDTH + LEGEND_ENTRY_GAP
    full[TOP_MARGIN:, :] = core
    return full


def _gen_attempt() -> np.ndarray | None:
    rng = random.Random()
    core_h = rng.randint(*GRID_H_RANGE)
    core_w = rng.randint(*GRID_W_RANGE)
    bg = rng.randrange(N_COLORS)
    fg = rng.choice([c for c in range(N_COLORS) if c != bg])
    cfg = rng.choice(VALID_CONFIGS)

    built = _build_output(rng, core_h, core_w, bg, fg, cfg)
    if built is None:
        return None
    clean_single, h_segments, v_segments = built

    comps, labels = _component_labels(clean_single, bg)
    n_comp = len(comps)
    if n_comp < 3 or n_comp > 4:
        return None
    if core_w < _legend_width(n_comp):
        return None

    palette = [c for c in range(N_COLORS) if c != bg]
    if len(palette) < 2 * n_comp:
        return None
    chosen_colors = rng.sample(palette, 2 * n_comp)
    line_colors = chosen_colors[:n_comp]
    hint_colors = chosen_colors[n_comp:]

    pairs = [(int(hint_colors[i]), int(line_colors[i])) for i in range(n_comp)]
    comp_to_line = {idx + 1: int(line_colors[idx]) for idx in range(n_comp)}
    comp_to_hint = {idx + 1: int(hint_colors[idx]) for idx in range(n_comp)}

    out_core = np.full(clean_single.shape, int(bg), dtype=int)
    for idx, comp in enumerate(comps, start=1):
        out_core[labels == int(idx)] = int(comp_to_line[idx])

    for seg in h_segments:
        seg["component"] = _segment_component_id(seg, "h", labels)
    for seg in v_segments:
        seg["component"] = _segment_component_id(seg, "v", labels)

    occupied: set[tuple[int, int]] = set()
    per_comp: dict[int, list[dict]] = {idx: [] for idx in range(1, n_comp + 1)}
    for seg in h_segments:
        opts = _h_gap_options(seg, clean_single, fg, bg, occupied)
        for gap_start, anchor in opts:
            per_comp[int(seg["component"])].append(
                {
                    "component": int(seg["component"]),
                    "kind": "h",
                    "seg": seg,
                    "gap_start": int(gap_start),
                    "anchor": anchor,
                    "chain": _chain_coords(int(seg["row"]), int(gap_start), anchor),
                }
            )
    for seg in v_segments:
        opts = _v_gap_options(seg, clean_single, fg, bg, occupied)
        for gap_start, anchor, dc in opts:
            per_comp[int(seg["component"])].append(
                {
                    "component": int(seg["component"]),
                    "kind": "v",
                    "seg": seg,
                    "gap_start": int(gap_start),
                    "anchor": anchor,
                    "dc": int(dc),
                    "chain": _v_chain_coords(int(seg["col"]), int(gap_start), anchor, int(dc)),
                }
            )
    if any(not opts for opts in per_comp.values()):
        return None

    plan = _choose_gap_plan(rng, per_comp)
    if plan is None:
        return None

    reserved: set[tuple[int, int]] = set()
    for opt in plan:
        _reserve_neighborhood(reserved, opt["chain"])

    decoy = _choose_decoy_hint(rng, plan, per_comp, reserved, comp_to_hint)
    if decoy is None:
        return None

    inp_core = out_core.copy()
    for opt in plan:
        comp_id = int(opt["component"])
        hint_color = int(comp_to_hint[comp_id])
        _apply_gap(inp_core, opt, bg)
        for r, c in opt["chain"]:
            inp_core[int(r), int(c)] = int(hint_color)

    _apply_gap(inp_core, decoy, bg)
    _apply_gap(out_core, decoy, bg)
    for r, c in decoy["chain"]:
        inp_core[int(r), int(c)] = int(decoy["hint_color"])
    _reserve_neighborhood(reserved, decoy["chain"])

    distractor_count = rng.randint(0, 2)
    forbidden: set[tuple[int, int]] = set()
    _reserve_neighborhood(forbidden, list(map(tuple, np.argwhere(out_core != int(bg)))))
    _reserve_neighborhood(forbidden, [cell for opt in plan for cell in opt["chain"]])
    _reserve_neighborhood(forbidden, decoy["chain"])
    distractors = _distractor_options(inp_core, bg, forbidden)
    rng.shuffle(distractors)
    placed = 0
    for coords in distractors:
        if placed >= distractor_count:
            break
        if any((int(r), int(c)) in forbidden for r, c in coords):
            continue
        color = int(rng.choice(hint_colors))
        for r, c in coords:
            inp_core[int(r), int(c)] = int(color)
        _reserve_neighborhood(forbidden, coords)
        placed += 1

    out = _wrap_with_legend(out_core, bg, pairs)
    inp = _wrap_with_legend(inp_core, bg, pairs)

    try:
        solved = main(inp)
    except Exception:
        return None
    if not np.array_equal(solved, out):
        return None
    if np.array_equal(_legend_blind_output(inp, bg), out):
        return None
    return inp


def generate_input() -> np.ndarray:
    for _ in range(GEN_RETRIES):
        grid = _gen_attempt()
        if grid is not None:
            return grid
    raise RuntimeError("failed to generate valid input for 7b80bb43 combo D11/1+D17/0")
