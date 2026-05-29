# Variant of abc82100
# Dimension: D2 — 规则示例的编码格式 (symbol_instructions)
# Change: 规则示例由相邻的源色-目标色像素对加上附近的teal模板图案构成，颜色对直接编码映射关系... → 规则示例使用训练对方式编码

from __future__ import annotations

import random

import numpy as np

from arc_utils import get_bg


# Design analysis for the D2 variant
# a) Concrete realization and behavior trace:
#    Each rule is no longer encoded by a local "src pixel -> tgt pixel + nearby teal stencil" motif.
#    Instead, the legend becomes a mini training pair made of two teal hollow frames placed side by side.
#    The left frame is the training input: a 3x3 inner board with exactly one centered source-color pixel.
#    The right frame is the training output: a same-size board showing the target-colored stamp shape,
#    aligned to that same center cell. Away from the legends, the grid still contains scattered isolated
#    source pixels plus uninvolved bystanders. The output erases all legend frames and replaces each
#    scattered source pixel by the learned target-colored shape anchored at that pixel.
#    Small trace:
#    - Original logic: read adjacent colors A,B near teal component T, infer "A becomes recolored T in B".
#    - Variant logic: read framed pair [centered A] -> [centered B-shape], infer the same anchored rewrite.
#    - Difference: the mapping is now expressed as an example input/output board that must be parsed and
#      aligned, not as an immediately adjacent color pair with an external stencil component.
# b) Complexity calibration:
#    The legends remain local and low-arity: one training pair per rule. This keeps the task human-solvable,
#    but the solver must detect frames, pair them, align source/output coordinates, and extract a shape from
#    the training output panel. Shapes are chosen from connected asymmetric-or-near-asymmetric 3x3 patterns
#    with 2-5 cells, so the visual change is obvious and different rules remain distinguishable.
# c) Implicit assumption audit against the original main():
#    The old solver assumed every rule could be found by scanning adjacent colored pairs and then searching
#    for a nearby teal component. That assumption is invalid here because:
#    - legend information is now boxed inside teal frames, not exposed as a free colored pair;
#    - the target color is read from the training output frame, not from a neighboring pixel;
#    - the shape is read from the output panel itself, not from a teal component in the main scene.
#    The new main() therefore depends on different invariants:
#    - teal only appears as hollow training frames;
#    - each left frame has exactly one non-frame non-bg pixel at the center;
#    - each right frame has exactly one non-frame non-bg color, whose footprint is the stamp.
# d) Placement legality:
#    Generator guarantees that frame rectangles do not overlap, that each rule occupies a whole 5x11 legend
#    cluster (left frame, one-column gap, right frame), and that scattered source pixels are placed outside
#    these clusters with 4-neighbor separation. Output stamps may overlap each other or clip at borders, as
#    in the original task, but bystanders are placed outside all future stamp footprints so they stay visible.
# e) Visual observability:
#    Humans can compare two framed mini-grids directly: a centered source dot on the left and the resulting
#    target-colored shape on the right. Because the source anchor stays at the center and also belongs to the
#    learned shape, the transformation is visible from 2-3 examples without relying on hidden metadata.
# f) Visual cleanliness:
#    The legends live in compact framed regions, while scattered pixels stay isolated. This separates
#    "instruction area" from "application area" and avoids mixing frame borders with active stamped shapes.
# g) Color inference:
#    Each rule still uses a unique source color and a unique target color from the non-bg non-teal palette.
#    Bystanders use only colors outside the rule palette, so a human can safely infer that pixels in source
#    colors are the ones to transform and that other colors are inert.
# h) New solver assumptions and generator guarantees:
#    - Assumption: a horizontally adjacent pair of same-sized teal frames with one blank column between them
#      is a training pair. Guarantee: generator only creates legends in that exact format.
#    - Assumption: the source anchor is the centered pixel of the left frame. Guarantee: generator always
#      places the left-frame source pixel at that center.
#    - Assumption: the right frame contains one monochrome stamp. Guarantee: generator samples exactly one
#      target color and one connected footprint for each right frame.


FRAME_COLOR = 8
NON_BG_NON_TEAL_COLORS = [1, 2, 3, 4, 5, 6, 7, 9]

# 3x3 inner training boards, hence 5x5 outer frames and 5x11 legend clusters.
INNER_SIZE = 3
FRAME_SIZE = INNER_SIZE + 2
PAIR_GAP = 1
CLUSTER_H = FRAME_SIZE
CLUSTER_W = FRAME_SIZE * 2 + PAIR_GAP
CENTER = (1, 1)

GRID_DIM_MAX = 20
MIN_DIM_BY_RULES = {1: 11, 2: 12, 3: 16, 4: 20}
RULE_COUNT_WEIGHTS = [1, 1, 2, 2, 3, 3, 4]

SCATTERED_MIN = 3
SCATTERED_MAX_CAP = 6
SCATTERED_AREA_DIV = 45
BYSTANDER_MAX = 5

SCATTERED_RETRIES_PER_PIXEL = 60
BYSTANDER_RETRIES = 80
OUTER_RETRIES = 300

_DIRS4 = [(-1, 0), (1, 0), (0, -1), (0, 1)]
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
    seen = set()
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
    for mask in range(1, 1 << len(others)):
        cells = {(0, 0)}
        for i, pos in enumerate(others):
            if (mask >> i) & 1:
                cells.add(pos)
        if not (2 <= len(cells) <= 5):
            continue
        if not _is_connected_8(cells):
            continue
        rel = tuple(sorted(cells))
        patterns.append(rel)
    patterns.sort(key=lambda rel: (len(rel), rel))
    return patterns


_PATTERN_LIBRARY = _build_pattern_library()


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


def _reserve_4neighbors(reserved: set[tuple[int, int]], cells) -> None:
    for r, c in cells:
        for dr, dc in _DIRS4:
            reserved.add((r + dr, c + dc))


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

        left_coords = np.argwhere((left_inner != int(bg)) & (left_inner != frame_color))
        if left_coords.shape[0] != 1:
            continue
        src_lr, src_lc = map(int, left_coords[0])
        source_color = int(left_inner[src_lr, src_lc])

        tgt_colors = sorted(set(int(x) for x in right_inner.ravel() if int(x) not in (int(bg), frame_color)))
        if len(tgt_colors) != 1:
            continue
        target_color = tgt_colors[0]
        tgt_coords = np.argwhere(right_inner == target_color)
        if tgt_coords.size == 0:
            continue

        rel = tuple(sorted((int(r) - src_lr, int(c) - src_lc) for r, c in tgt_coords))
        legend_cells = set(fr["area_cells"]) | set(right["area_cells"]) | {
            (r, right["c0"] - 1) for r in range(fr["r0"], fr["r1"] + 1)
        }
        rules.append(
            {
                "source_color": source_color,
                "target_color": target_color,
                "rel": rel,
                "legend_cells": legend_cells,
            }
        )

    return rules


def main(grid: np.ndarray) -> np.ndarray:
    """
    Learn rewrite rules from framed training pairs, then stamp the learned outputs onto matching source pixels.

    Input structure:
    - Black background with scattered isolated colored pixels, plus teal (8) hollow frames.
    - Each rule legend is two same-sized teal frames placed side by side: the left frame contains a tiny
      training input (a centered source-color pixel), and the right frame contains the corresponding tiny
      training output (a target-colored shape on the same local board).

    Rule:
    - For each framed pair, infer which source color is being transformed, which target color is produced,
      and which relative shape is drawn around the source anchor.
    - Apply that learned rewrite to every other pixel of the same source color elsewhere in the grid:
      replace the single pixel by the learned target-colored shape anchored at that pixel.
    - Remove all legend frames and their contents from the final output; preserve uninvolved bystanders.

    Key insight: The instruction is an analogy panel rather than a color pair. The left mini-board shows
    the local input state, the right mini-board shows the local output state, and the same local rewrite
    must be generalized to all scattered occurrences of the source color.
    """
    g = np.asarray(grid, dtype=int)
    bg = get_bg(g)
    h, w = g.shape

    rules = _discover_rules(g, bg=bg, frame_color=FRAME_COLOR)
    legend_cells: set[tuple[int, int]] = set()
    source_colors: set[int] = set()
    for rule in rules:
        legend_cells.update(rule["legend_cells"])
        source_colors.add(rule["source_color"])

    out = np.full_like(g, int(bg))

    for r, c in map(tuple, np.argwhere((g != int(bg)) & (g != FRAME_COLOR))):
        r = int(r)
        c = int(c)
        color = int(g[r, c])
        if (r, c) in legend_cells or color in source_colors:
            continue
        out[r, c] = color

    for rule in rules:
        for sr, sc in map(tuple, np.argwhere(g == rule["source_color"])):
            sr = int(sr)
            sc = int(sc)
            if (sr, sc) in legend_cells:
                continue
            rows = np.array([sr + rr for rr, _ in rule["rel"]], dtype=int)
            cols = np.array([sc + cc for _, cc in rule["rel"]], dtype=int)
            ok = (0 <= rows) & (rows < h) & (0 <= cols) & (cols < w)
            out[rows[ok], cols[ok]] = rule["target_color"]

    return out


def _build_rules(num_rules: int, area: int):
    colors = NON_BG_NON_TEAL_COLORS[:]
    random.shuffle(colors)
    sc_max = min(SCATTERED_MAX_CAP, max(SCATTERED_MIN, area // SCATTERED_AREA_DIV))
    patterns = random.sample(_PATTERN_LIBRARY, num_rules)

    rules = []
    used_colors = set()
    for i in range(num_rules):
        src_c, tgt_c = colors[2 * i], colors[2 * i + 1]
        used_colors.update((src_c, tgt_c))
        rules.append(
            {
                "src": src_c,
                "tgt": tgt_c,
                "rel": patterns[i],
                "n_sc": random.randint(SCATTERED_MIN, sc_max),
            }
        )
    return rules, used_colors


def _draw_legend(rule: dict, inp: np.ndarray, top: int, left: int) -> tuple[set[tuple[int, int]], set[tuple[int, int]]]:
    border_cells = set()
    occupied_cells = set()

    left_r0, left_c0 = top, left
    left_r1, left_c1 = top + FRAME_SIZE - 1, left + FRAME_SIZE - 1
    right_r0, right_c0 = top, left + FRAME_SIZE + PAIR_GAP
    right_r1, right_c1 = right_r0 + FRAME_SIZE - 1, right_c0 + FRAME_SIZE - 1

    for r, c in _frame_border_cells(left_r0, left_c0, left_r1, left_c1):
        inp[r, c] = FRAME_COLOR
        border_cells.add((r, c))
        occupied_cells.add((r, c))
    for r, c in _frame_border_cells(right_r0, right_c0, right_r1, right_c1):
        inp[r, c] = FRAME_COLOR
        border_cells.add((r, c))
        occupied_cells.add((r, c))

    src_r = left_r0 + 1 + CENTER[0]
    src_c = left_c0 + 1 + CENTER[1]
    inp[src_r, src_c] = rule["src"]
    occupied_cells.add((src_r, src_c))

    anchor_r = right_r0 + 1 + CENTER[0]
    anchor_c = right_c0 + 1 + CENTER[1]
    for rr, cc in rule["rel"]:
        nr = anchor_r + rr
        nc = anchor_c + cc
        inp[nr, nc] = rule["tgt"]
        occupied_cells.add((nr, nc))

    cluster_cells = _rect_cells(top, left, CLUSTER_H, CLUSTER_W)
    gap_col = left + FRAME_SIZE
    cluster_cells.update((r, gap_col) for r in range(top, top + CLUSTER_H))
    return border_cells, cluster_cells | occupied_cells


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


def _place_legends(rules, inp, occupied, reserved):
    h, w = inp.shape
    if h < CLUSTER_H or w < CLUSTER_W:
        return False
    starts = _sample_starts(h, CLUSTER_H, len(rules))
    if starts is None:
        return False
    left = random.randint(0, w - CLUSTER_W)
    for rule, top in zip(rules, starts):
        cluster_cells = _rect_cells(top, left, CLUSTER_H, CLUSTER_W)
        if cluster_cells & occupied:
            return False
        border_cells, all_cells = _draw_legend(rule, inp, top, left)
        occupied.update(all_cells)
        _reserve_4neighbors(reserved, border_cells)
    return True


def _place_scattered(rules, inp, occupied, reserved):
    h, w = inp.shape
    for rule in rules:
        positions = []
        tries = rule["n_sc"] * SCATTERED_RETRIES_PER_PIXEL
        for _ in range(tries):
            if len(positions) >= rule["n_sc"]:
                break
            r = random.randint(0, h - 1)
            c = random.randint(0, w - 1)
            if (r, c) in occupied or (r, c) in reserved:
                continue
            positions.append((r, c))
            occupied.add((r, c))
            _reserve_4neighbors(reserved, [(r, c)])
            inp[r, c] = rule["src"]
        if len(positions) < SCATTERED_MIN:
            return False
        rule["scattered"] = positions
    return True


def _stamp_footprint(rules, h: int, w: int) -> set[tuple[int, int]]:
    cells = set()
    for rule in rules:
        for sr, sc in rule.get("scattered", []):
            for rr, cc in rule["rel"]:
                nr = sr + rr
                nc = sc + cc
                if 0 <= nr < h and 0 <= nc < w:
                    cells.add((nr, nc))
    return cells


def _place_bystanders(used_colors, rules, inp, occupied, reserved):
    bystander_colors = [c for c in NON_BG_NON_TEAL_COLORS if c not in used_colors]
    if not bystander_colors:
        return []
    h, w = inp.shape
    n = random.randint(0, min(BYSTANDER_MAX, len(bystander_colors) * 2))
    forbidden = _stamp_footprint(rules, h, w)
    placed = []
    for _ in range(n):
        for _ in range(BYSTANDER_RETRIES):
            r = random.randint(0, h - 1)
            c = random.randint(0, w - 1)
            if (r, c) in occupied or (r, c) in reserved or (r, c) in forbidden:
                continue
            color = random.choice(bystander_colors)
            inp[r, c] = color
            placed.append((r, c, color))
            occupied.add((r, c))
            _reserve_4neighbors(reserved, [(r, c)])
            break
    return placed


def _build_output(rules, bystanders, h: int, w: int) -> np.ndarray:
    out = np.zeros((h, w), dtype=int)
    for r, c, color in bystanders:
        out[r, c] = color
    for rule in rules:
        for sr, sc in rule["scattered"]:
            for rr, cc in rule["rel"]:
                nr = sr + rr
                nc = sc + cc
                if 0 <= nr < h and 0 <= nc < w:
                    out[nr, nc] = rule["tgt"]
    return out


def _gen_attempt():
    num_rules = random.choice(RULE_COUNT_WEIGHTS)
    min_dim = MIN_DIM_BY_RULES[num_rules]
    h = random.randint(min_dim, GRID_DIM_MAX)
    w = random.randint(min_dim, GRID_DIM_MAX)

    rules, used_colors = _build_rules(num_rules, h * w)
    inp = np.zeros((h, w), dtype=int)
    occupied: set[tuple[int, int]] = set()
    reserved: set[tuple[int, int]] = set()

    if not _place_legends(rules, inp, occupied, reserved):
        return None
    if not _place_scattered(rules, inp, occupied, reserved):
        return None
    bystanders = _place_bystanders(used_colors, rules, inp, occupied, reserved)

    expected = _build_output(rules, bystanders, h, w)
    try:
        if np.array_equal(main(inp), expected):
            return inp
    except Exception:
        pass
    return None


def generate_input() -> np.ndarray:
    for _ in range(OUTER_RETRIES):
        result = _gen_attempt()
        if result is not None:
            return result
    raise RuntimeError(f"Failed to generate valid input after {OUTER_RETRIES} attempts")
