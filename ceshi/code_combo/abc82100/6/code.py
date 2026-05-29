# Combo variant of abc82100 (index 6)
# combo_spec: D2/1+D9/2
# Name: 训练对编码+规则链式应用

from __future__ import annotations

import random

import numpy as np

from arc_utils import get_bg


# Design analysis for the combo variant
# a) Concrete realization and behavior trace:
#    The original task taught a rule with a local colored pair plus a nearby teal stencil, then applied that
#    one-step rewrite independently to scattered source pixels. Here the instruction area becomes a chain of
#    framed training pairs. Each legend row has two 5x5 teal hollow frames with 3x3 interiors:
#    the left interior is a monochrome source sprite, and the right interior is the corresponding monochrome
#    target sprite. Consecutive legends are linked because rule i's right sprite is exactly rule i+1's left
#    sprite, including both shape and color. A concrete trace is:
#    - Original: isolated red pixel + nearby teal stencil example means "red dot -> blue copied stencil".
#    - Combo step 1: framed example shows [red dot] -> [blue L].
#    - Combo step 2: next framed example shows [blue L] -> [yellow T].
#    - Application: each scattered red dot becomes a blue L after step 1, then that blue L becomes a yellow T
#      after step 2. The output therefore contains only the final yellow T sprites, not the intermediate blue L.
#    The main difference is that the rule must be learned from analogy panels and then composed across steps.
# b) Complexity calibration:
#    The chain length is 2-3, which is large enough that the second step materially matters but still small
#    enough for humans to track. Every sprite lives inside a 3x3 board and is monochrome, so the rule space is
#    visually structured rather than arbitrary. The first stage is always a single-pixel source, preserving a
#    recognizable link to the original task, while later stages use nontrivial connected sprites so the chain is
#    visibly different from a one-step recolor-and-stamp puzzle.
# c) Implicit assumption audit against the original main():
#    The old solver depended on three assumptions that no longer hold:
#    - A rule is not encoded by an exposed adjacent source-target color pair anymore.
#    - Teal no longer marks the stencil itself; teal now only marks hollow instruction frames.
#    - Multiple rules are not parallel color mappings; they must be ordered and executed sequentially.
#    The new solver therefore assumes:
#    - teal appears only as 5x5 hollow frames around 3x3 training boards;
#    - each left/right board is monochrome and contains the center cell, which is the shared anchor;
#    - the rule order is discoverable from exact stage continuity: previous output signature == next input
#      signature, not from scan order alone;
#    - live sprites outside the legend occupy isolated 3x3 neighborhoods, so exact board matching is possible.
# d) Placement legality:
#    Generator guarantees that legend clusters do not overlap, that their gap column is preserved, and that
#    scattered anchors are placed so the full 3x3 neighborhood of every intermediate and final sprite stays
#    in-bounds and disjoint from other sprites and from legends. This prevents accidental overlaps that would
#    break later chain steps. Bystanders are also kept away from all eventual sprite footprints.
# e) Visual observability:
#    A human can read the chain directly: the right training board of one row reappears as the left training
#    board of the next row. That makes the execution order visible from the pixels themselves rather than from a
#    hidden convention. Seeing two or three rows is enough to infer "apply these local rewrites in sequence to
#    every scattered start object".
# f) Visual cleanliness:
#    The instruction area is a compact vertical strip of framed pairs, while the scattered objects occupy a
#    separate free area. Because each active object is confined to its own 3x3 neighborhood, the scene remains
#    easy to parse and the intermediate logic is not muddied by overlapping shapes.
# g) Change inferability:
#    Colors are limited to a short stage chain c0 -> c1 -> c2 (-> c3). The color change is always accompanied
#    by a visible shape change in the training panels, so the solver never has to infer arbitrary color mapping
#    from color alone. Bystander colors are drawn from colors unused by the chain, which makes their inert role
#    visually evident.
# h) New solver assumptions and generator guarantees:
#    - Assumption: horizontally adjacent equal-sized teal frames with one blank column between them form one
#      training pair. Guarantee: generator uses exactly that layout.
#    - Assumption: the anchor is the center of the 3x3 board. Guarantee: every sampled sprite contains center.
#    - Assumption: chain order is unique. Guarantee: stage signatures are unique except for the intended
#      target-to-next-source links.
#    - Assumption: source matches are exact 3x3 local-board matches. Guarantee: scattered objects and all
#      intermediate states live in isolated 3x3 windows with background elsewhere in those windows.
# i) Degeneracy check target:
#    The critical anti-degeneration metric is "chain effect pixels": compare the full sequential output against
#    the output produced by applying only the first learned rule. If that difference were always zero, the task
#    would have collapsed back to a one-step variant. The generator is designed so later stages change both
#    shape and color, making that metric strictly positive on valid samples.


FRAME_COLOR = 8
NON_BG_NON_TEAL_COLORS = [1, 2, 3, 4, 5, 6, 7, 9]

INNER_SIZE = 3
FRAME_SIZE = INNER_SIZE + 2
PAIR_GAP = 1
CLUSTER_H = FRAME_SIZE
CLUSTER_W = FRAME_SIZE * 2 + PAIR_GAP
CENTER = (1, 1)

GRID_DIM_MAX = 20
MIN_DIMS_BY_RULES = {
    2: (12, 15),
    3: (16, 15),
}
RULE_COUNT_WEIGHTS = [2, 2, 3, 3]

SCATTERED_MIN = 3
SCATTERED_MAX_CAP = 5
SCATTERED_AREA_DIV = 55
BYSTANDER_MAX = 4

SCATTERED_RETRIES_PER_PIXEL = 80
BYSTANDER_RETRIES = 80
OUTER_RETRIES = 400

_DIRS8 = [
    (-1, -1),
    (-1, 0),
    (-1, 1),
    (0, -1),
    (0, 1),
    (1, -1),
    (1, 0),
    (1, 1),
]


def _is_connected_8(cells: set[tuple[int, int]]) -> bool:
    todo = [next(iter(cells))]
    seen: set[tuple[int, int]] = set()
    while todo:
        cur = todo.pop()
        if cur in seen:
            continue
        seen.add(cur)
        r, c = cur
        for dr, dc in _DIRS8:
            nxt = (r + dr, c + dc)
            if nxt in cells and nxt not in seen:
                todo.append(nxt)
    return seen == cells


def _build_pattern_library() -> list[tuple[tuple[int, int], ...]]:
    others = [(r, c) for r in range(-1, 2) for c in range(-1, 2) if (r, c) != (0, 0)]
    patterns: list[tuple[tuple[int, int], ...]] = []
    for mask in range(1 << len(others)):
        cells = {(0, 0)}
        for i, pos in enumerate(others):
            if (mask >> i) & 1:
                cells.add(pos)
        if not (1 <= len(cells) <= 5):
            continue
        if not _is_connected_8(cells):
            continue
        rel = tuple(sorted(cells))
        patterns.append(rel)
    patterns.sort(key=lambda rel: (len(rel), rel))
    return patterns


_PATTERN_LIBRARY = _build_pattern_library()
_NONTRIVIAL_PATTERNS = [rel for rel in _PATTERN_LIBRARY if len(rel) >= 2]
_SINGLETON_PATTERN = ((0, 0),)


def _rect_cells(r0: int, c0: int, h: int, w: int) -> set[tuple[int, int]]:
    return {(r, c) for r in range(r0, r0 + h) for c in range(c0, c0 + w)}


def _frame_border_cells(r0: int, c0: int, r1: int, c1: int) -> set[tuple[int, int]]:
    cells = set()
    for c in range(c0, c1 + 1):
        cells.add((r0, c))
        cells.add((r1, c))
    for r in range(r0, r1 + 1):
        cells.add((r, c0))
        cells.add((r, c1))
    return cells


def _expanded(cells: set[tuple[int, int]], radius: int = 1) -> set[tuple[int, int]]:
    out = set(cells)
    for r, c in list(cells):
        for dr in range(-radius, radius + 1):
            for dc in range(-radius, radius + 1):
                out.add((r + dr, c + dc))
    return out


def _box3_cells(center_r: int, center_c: int) -> set[tuple[int, int]]:
    return {(center_r + dr, center_c + dc) for dr in range(-1, 2) for dc in range(-1, 2)}


def _stamp_cells(anchor: tuple[int, int], rel: tuple[tuple[int, int], ...]) -> set[tuple[int, int]]:
    ar, ac = anchor
    return {(ar + dr, ac + dc) for dr, dc in rel}


def _draw_sprite(board: np.ndarray, anchor_r: int, anchor_c: int, rel, color: int) -> None:
    for dr, dc in rel:
        board[anchor_r + dr, anchor_c + dc] = int(color)


def _find_frames(grid: np.ndarray, frame_color: int = FRAME_COLOR) -> list[dict]:
    g = np.asarray(grid, dtype=int)
    h, w = g.shape
    candidates: list[dict] = []
    for r0 in range(h - 2):
        for c0 in range(w - 2):
            if int(g[r0, c0]) != frame_color:
                continue
            for r1 in range(r0 + 2, h):
                if int(g[r1, c0]) != frame_color:
                    continue
                for c1 in range(c0 + 2, w):
                    if int(g[r0, c1]) != frame_color or int(g[r1, c1]) != frame_color:
                        continue
                    if not (
                        np.all(g[r0, c0 : c1 + 1] == frame_color)
                        and np.all(g[r1, c0 : c1 + 1] == frame_color)
                        and np.all(g[r0 : r1 + 1, c0] == frame_color)
                        and np.all(g[r0 : r1 + 1, c1] == frame_color)
                    ):
                        continue
                    if np.any(g[r0 + 1 : r1, c0 + 1 : c1] == frame_color):
                        continue
                    candidates.append(
                        {
                            "r0": int(r0),
                            "c0": int(c0),
                            "r1": int(r1),
                            "c1": int(c1),
                            "h": int(r1 - r0 + 1),
                            "w": int(c1 - c0 + 1),
                        }
                    )
    candidates.sort(key=lambda fr: (-(fr["h"] * fr["w"]), fr["r0"], fr["c0"]))
    picked: list[dict] = []
    used: set[tuple[int, int]] = set()
    for fr in candidates:
        area = _rect_cells(fr["r0"], fr["c0"], fr["h"], fr["w"])
        if area & used:
            continue
        fr["area_cells"] = area
        picked.append(fr)
        used.update(area)
    picked.sort(key=lambda fr: (fr["r0"], fr["c0"]))
    return picked


def _extract_board_signature(inner: np.ndarray, bg: int) -> tuple[int, tuple[tuple[int, int], ...]] | None:
    colors = sorted(set(int(x) for x in inner.ravel() if int(x) != int(bg)))
    if len(colors) != 1:
        return None
    color = colors[0]
    coords = np.argwhere(inner == color)
    rel = tuple(sorted((int(r) - CENTER[0], int(c) - CENTER[1]) for r, c in coords))
    if (0, 0) not in rel:
        return None
    return color, rel


def _discover_rules(grid: np.ndarray, bg: int, frame_color: int = FRAME_COLOR) -> list[dict]:
    g = np.asarray(grid, dtype=int)
    frames = _find_frames(g, frame_color=frame_color)
    by_key = {(fr["r0"], fr["c0"], fr["r1"], fr["c1"]): fr for fr in frames}
    rules: list[dict] = []

    for fr in frames:
        right_key = (fr["r0"], fr["c1"] + PAIR_GAP + 1, fr["r1"], fr["c1"] + PAIR_GAP + fr["w"])
        right = by_key.get(right_key)
        if right is None or right["h"] != fr["h"] or right["w"] != fr["w"]:
            continue

        left_inner = g[fr["r0"] + 1 : fr["r1"], fr["c0"] + 1 : fr["c1"]]
        right_inner = g[right["r0"] + 1 : right["r1"], right["c0"] + 1 : right["c1"]]
        src_sig = _extract_board_signature(left_inner, bg=bg)
        tgt_sig = _extract_board_signature(right_inner, bg=bg)
        if src_sig is None or tgt_sig is None:
            continue

        legend_cells = set(fr["area_cells"]) | set(right["area_cells"]) | {
            (r, right["c0"] - 1) for r in range(fr["r0"], fr["r1"] + 1)
        }
        source_color, source_rel = src_sig
        target_color, target_rel = tgt_sig
        rules.append(
            {
                "source_color": int(source_color),
                "source_rel": tuple(source_rel),
                "target_color": int(target_color),
                "target_rel": tuple(target_rel),
                "source_sig": (int(source_color), tuple(source_rel)),
                "target_sig": (int(target_color), tuple(target_rel)),
                "legend_cells": legend_cells,
                "top": int(fr["r0"]),
                "left": int(fr["c0"]),
            }
        )

    return rules


def _order_rules(rules: list[dict]) -> list[dict]:
    if not rules:
        return []
    by_source = {rule["source_sig"]: rule for rule in rules}
    target_sigs = {rule["target_sig"] for rule in rules}
    starts = [rule for rule in rules if rule["source_sig"] not in target_sigs]
    if len(starts) != 1:
        return sorted(rules, key=lambda rule: (rule["top"], rule["left"]))

    ordered = [starts[0]]
    seen = {id(starts[0])}
    while ordered[-1]["target_sig"] in by_source:
        nxt = by_source[ordered[-1]["target_sig"]]
        if id(nxt) in seen:
            break
        ordered.append(nxt)
        seen.add(id(nxt))
    if len(ordered) != len(rules):
        return sorted(rules, key=lambda rule: (rule["top"], rule["left"]))
    return ordered


def _matches_sprite(grid: np.ndarray, anchor_r: int, anchor_c: int, rel, color: int, bg: int) -> bool:
    h, w = grid.shape
    if not (1 <= anchor_r < h - 1 and 1 <= anchor_c < w - 1):
        return False
    rel_set = set(rel)
    for dr in range(-1, 2):
        for dc in range(-1, 2):
            rr = anchor_r + dr
            cc = anchor_c + dc
            expected = int(color) if (dr, dc) in rel_set else int(bg)
            if int(grid[rr, cc]) != expected:
                return False
    return True


def _apply_rule_once(grid: np.ndarray, rule: dict, bg: int) -> tuple[np.ndarray, list[tuple[int, int]]]:
    cur = np.asarray(grid, dtype=int)
    h, w = cur.shape
    anchors: list[tuple[int, int]] = []
    for r in range(1, h - 1):
        for c in range(1, w - 1):
            if _matches_sprite(cur, r, c, rule["source_rel"], rule["source_color"], bg):
                anchors.append((int(r), int(c)))

    nxt = cur.copy()
    for ar, ac in anchors:
        for dr, dc in rule["source_rel"]:
            nxt[ar + dr, ac + dc] = int(bg)
    for ar, ac in anchors:
        for dr, dc in rule["target_rel"]:
            nxt[ar + dr, ac + dc] = int(rule["target_color"])
    return nxt, anchors


def _apply_chain(grid: np.ndarray, rules: list[dict], bg: int) -> np.ndarray:
    cur = np.asarray(grid, dtype=int).copy()
    for rule in rules:
        cur, _anchors = _apply_rule_once(cur, rule, bg=bg)
    return cur


def _apply_first_rule_only(grid: np.ndarray, rules: list[dict], bg: int) -> np.ndarray:
    if not rules:
        return np.asarray(grid, dtype=int).copy()
    cur = np.asarray(grid, dtype=int).copy()
    cur, _anchors = _apply_rule_once(cur, rules[0], bg=bg)
    return cur


def _chain_effect_pixels_from_input(grid: np.ndarray) -> int:
    g = np.asarray(grid, dtype=int)
    bg = get_bg(g)
    rules = _order_rules(_discover_rules(g, bg=bg, frame_color=FRAME_COLOR))
    legend_cells: set[tuple[int, int]] = set()
    for rule in rules:
        legend_cells.update(rule["legend_cells"])
    work = g.copy()
    for r, c in legend_cells:
        work[r, c] = int(bg)
    full = _apply_chain(work, rules, bg=bg)
    first_only = _apply_first_rule_only(work, rules, bg=bg)
    return int(np.count_nonzero(full != first_only))


def main(grid: np.ndarray) -> np.ndarray:
    """
    Learn a sequence of framed local rewrites, then execute the whole chain on every scattered start object.

    Input structure:
    - Black background with teal (8) hollow frames used only as instruction panels.
    - Each legend row is a horizontally adjacent pair of same-sized teal frames with a one-column gap.
      The left 3x3 inner board shows a monochrome source sprite, and the right 3x3 inner board shows the
      corresponding monochrome target sprite.
    - Consecutive legend rows form a chain because one row's target board exactly matches the next row's
      source board. Away from the legends, scattered objects match only the first source board of the chain.

    Rule:
    - Parse every framed pair as a local rewrite "source 3x3 board -> target 3x3 board", using the center
      of the board as the anchor.
    - Order the learned rules by stage continuity: the output signature of one step is the input signature
      of the next step.
    - Remove the entire legend area, then repeatedly apply the rules in that order to every exact matching
      sprite elsewhere in the grid. Uninvolved bystanders stay unchanged.

    Key insight: This is no longer a direct color-to-stencil lookup. The legends are miniature training
    examples, and the right board of one example becoming the left board of the next reveals that the
    transformation must be composed across multiple stages before reaching the final output.
    """
    g = np.asarray(grid, dtype=int)
    bg = get_bg(g)

    rules = _order_rules(_discover_rules(g, bg=bg, frame_color=FRAME_COLOR))
    legend_cells: set[tuple[int, int]] = set()
    for rule in rules:
        legend_cells.update(rule["legend_cells"])

    work = g.copy()
    for r, c in legend_cells:
        work[r, c] = int(bg)

    return _apply_chain(work, rules, bg=bg)


def _build_chain(num_rules: int, area: int) -> tuple[list[dict], set[int], int]:
    colors = random.sample(NON_BG_NON_TEAL_COLORS, num_rules + 1)
    patterns = [_SINGLETON_PATTERN]
    chosen_shapes: set[tuple[tuple[int, int], ...]] = set(patterns)
    while len(patterns) < num_rules + 1:
        cand = random.choice(_NONTRIVIAL_PATTERNS)
        if cand in chosen_shapes:
            continue
        patterns.append(cand)
        chosen_shapes.add(cand)

    sc_max = min(SCATTERED_MAX_CAP, max(SCATTERED_MIN, area // SCATTERED_AREA_DIV))
    n_scattered = random.randint(SCATTERED_MIN, sc_max)

    rules = []
    used_colors = set(colors)
    for i in range(num_rules):
        rules.append(
            {
                "source_color": int(colors[i]),
                "source_rel": patterns[i],
                "target_color": int(colors[i + 1]),
                "target_rel": patterns[i + 1],
            }
        )
    return rules, used_colors, n_scattered


def _draw_legend(rule: dict, inp: np.ndarray, top: int, left: int) -> set[tuple[int, int]]:
    left_r0, left_c0 = top, left
    left_r1, left_c1 = top + FRAME_SIZE - 1, left + FRAME_SIZE - 1
    right_r0, right_c0 = top, left + FRAME_SIZE + PAIR_GAP
    right_r1, right_c1 = right_r0 + FRAME_SIZE - 1, right_c0 + FRAME_SIZE - 1

    for r, c in _frame_border_cells(left_r0, left_c0, left_r1, left_c1):
        inp[r, c] = FRAME_COLOR
    for r, c in _frame_border_cells(right_r0, right_c0, right_r1, right_c1):
        inp[r, c] = FRAME_COLOR

    left_anchor_r = left_r0 + 1 + CENTER[0]
    left_anchor_c = left_c0 + 1 + CENTER[1]
    right_anchor_r = right_r0 + 1 + CENTER[0]
    right_anchor_c = right_c0 + 1 + CENTER[1]
    _draw_sprite(inp, left_anchor_r, left_anchor_c, rule["source_rel"], rule["source_color"])
    _draw_sprite(inp, right_anchor_r, right_anchor_c, rule["target_rel"], rule["target_color"])

    cluster_cells = _rect_cells(top, left, CLUSTER_H, CLUSTER_W)
    gap_col = left + FRAME_SIZE
    cluster_cells.update((r, gap_col) for r in range(top, top + CLUSTER_H))
    return cluster_cells


def _sample_starts(total: int, block: int, count: int) -> list[int] | None:
    slack = total - block * count
    if slack < 0:
        return None
    parts = [0] * (count + 1)
    for _ in range(slack):
        parts[random.randint(0, count)] += 1
    starts = []
    cur = parts[0]
    for i in range(count):
        starts.append(cur)
        cur += block + parts[i + 1]
    return starts


def _place_legends(rules: list[dict], inp: np.ndarray) -> tuple[bool, set[tuple[int, int]], set[tuple[int, int]]]:
    h, w = inp.shape
    if h < CLUSTER_H or w < CLUSTER_W:
        return False, set(), set()
    starts = _sample_starts(h, CLUSTER_H, len(rules))
    if starts is None:
        return False, set(), set()
    left = random.randint(0, w - CLUSTER_W)

    occupied: set[tuple[int, int]] = set()
    reserved: set[tuple[int, int]] = set()
    for rule, top in zip(rules, starts):
        cluster = _draw_legend(rule, inp, top, left)
        occupied.update(cluster)
        reserved.update(_expanded(cluster, radius=1))
    return True, occupied, reserved


def _place_scattered(
    first_rule: dict,
    n_scattered: int,
    inp: np.ndarray,
    occupied: set[tuple[int, int]],
    reserved: set[tuple[int, int]],
) -> list[tuple[int, int]] | None:
    h, w = inp.shape
    anchors: list[tuple[int, int]] = []
    for _ in range(n_scattered * SCATTERED_RETRIES_PER_PIXEL):
        if len(anchors) >= n_scattered:
            break
        r = random.randint(1, h - 2)
        c = random.randint(1, w - 2)
        box = _box3_cells(r, c)
        if box & reserved:
            continue
        anchors.append((r, c))
        occupied.update(box)
        reserved.update(_expanded(box, radius=0))
        _draw_sprite(inp, r, c, first_rule["source_rel"], first_rule["source_color"])
    if len(anchors) < SCATTERED_MIN:
        return None
    return anchors


def _all_stage_footprints(rules: list[dict], anchors: list[tuple[int, int]]) -> set[tuple[int, int]]:
    cells = set()
    stage_rels = [rules[0]["source_rel"]] + [rule["target_rel"] for rule in rules]
    for anchor in anchors:
        for rel in stage_rels:
            cells.update(_stamp_cells(anchor, rel))
    return cells


def _place_bystanders(
    used_colors: set[int],
    future_footprints: set[tuple[int, int]],
    inp: np.ndarray,
    occupied: set[tuple[int, int]],
    reserved: set[tuple[int, int]],
) -> list[tuple[int, int, int]]:
    bystander_colors = [c for c in NON_BG_NON_TEAL_COLORS if c not in used_colors]
    if not bystander_colors:
        return []
    h, w = inp.shape
    n = random.randint(0, min(BYSTANDER_MAX, len(bystander_colors) * 2))
    placed = []
    for _ in range(n):
        for _ in range(BYSTANDER_RETRIES):
            r = random.randint(0, h - 1)
            c = random.randint(0, w - 1)
            if (r, c) in occupied or (r, c) in reserved or (r, c) in future_footprints:
                continue
            color = random.choice(bystander_colors)
            inp[r, c] = color
            placed.append((r, c, color))
            occupied.add((r, c))
            reserved.update(_expanded({(r, c)}, radius=1))
            break
    return placed


def _build_expected(rules: list[dict], anchors: list[tuple[int, int]], bystanders, h: int, w: int) -> np.ndarray:
    out = np.zeros((h, w), dtype=int)
    for r, c, color in bystanders:
        out[r, c] = color
    final_rel = rules[-1]["target_rel"]
    final_color = rules[-1]["target_color"]
    for ar, ac in anchors:
        for dr, dc in final_rel:
            out[ar + dr, ac + dc] = final_color
    return out


def _gen_attempt() -> np.ndarray | None:
    num_rules = random.choice(RULE_COUNT_WEIGHTS)
    min_h, min_w = MIN_DIMS_BY_RULES[num_rules]
    h = random.randint(min_h, GRID_DIM_MAX)
    w = random.randint(min_w, GRID_DIM_MAX)

    rules, used_colors, n_scattered = _build_chain(num_rules, h * w)
    inp = np.zeros((h, w), dtype=int)

    ok, occupied, reserved = _place_legends(rules, inp)
    if not ok:
        return None

    anchors = _place_scattered(rules[0], n_scattered, inp, occupied, reserved)
    if anchors is None:
        return None

    future_footprints = _all_stage_footprints(rules, anchors)
    bystanders = _place_bystanders(used_colors, future_footprints, inp, occupied, reserved)
    expected = _build_expected(rules, anchors, bystanders, h, w)

    try:
        actual = main(inp)
    except Exception:
        return None
    if not np.array_equal(actual, expected):
        return None
    if _chain_effect_pixels_from_input(inp) <= 0:
        return None
    return inp


def generate_input() -> np.ndarray:
    for _ in range(OUTER_RETRIES):
        result = _gen_attempt()
        if result is not None:
            return result
    raise RuntimeError(f"Failed to generate valid input after {OUTER_RETRIES} attempts")
