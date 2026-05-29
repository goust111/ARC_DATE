# Combo variant of abc82100 (index 41)
# combo_spec: D1/1+D2/0+D10/1
# Name: 行列对齐匹配+图例编码+边缘规则

from __future__ import annotations

import random
from functools import lru_cache
from itertools import combinations

import numpy as np

from arc_utils import get_bg, label_mask

# Design analysis for the combo variant
# a) Concrete behavior and trace:
#    - Original abc82100 learns a full rule from one local scene fragment:
#      source color, target color, and a nearby teal stencil all live together,
#      and scattered pixels choose a rule by matching source color.
#    - This combo splits that information across two edge legends:
#        * left edge row labels store "shared source marker -> target color"
#        * top edge column labels store "shared source marker + teal stencil"
#      Every interior marker pixel sits at the intersection of one active row and
#      one active column. Its row chooses the output color; its column chooses the
#      output shape. The solver therefore reads the grid like a coordinate table.
#    - Small trace:
#        * Suppose the left edge says row 7 maps marker->yellow and row 11 maps
#          marker->red.
#        * The top edge says column 6 uses a T-shaped stencil and column 12 uses
#          an L-shaped stencil.
#        * Then the source pixel at (7, 6) becomes a yellow T, while the source
#          pixel at (11, 12) becomes a red L, even though both input pixels have
#          the same marker color.
# b) Complexity calibration:
#    - The generator always uses at least two active rows and two active columns,
#      so both axes repeat and neither can be ignored.
#    - All scattered source pixels share one marker color. This blocks fallback to
#      the original color-matching strategy.
#    - Distinct target colors vary by row and distinct stencil shapes vary by
#      column, giving repeated visual support for both control variables.
# c) Implicit-assumption audit against the original main():
#    - The original "_discover_rules()" assumed full rules appear as local objects
#      anywhere in the scene. That is false here, so the combo solver decodes two
#      structured edge legends instead of searching globally.
#    - The original preservation logic assumed "source colors" identify all
#      consumable pixels. That is false here because every consumable pixel shares
#      the same source color with every rule. The combo solver now consumes only
#      marker pixels that lie on an active row and an active column.
#    - The original stamping loop used one rule's stored anchor for every matching
#      source color. Here the anchor is fixed inside each column legend's 3x3 mini
#      stencil, so translation is defined by the active column rather than by a
#      local example position in the work area.
# d) Placement legality:
#    - Row legends live only in the left band, column legends only in the top band,
#      and source pixels only in the central work area.
#    - Active rows and columns are spaced apart so source pixels are isolated and
#      top legend stencil windows do not overlap.
#    - Bystanders are placed off active rows/columns and outside every future stamp
#      footprint, so they remain unchanged in the output.
# e) Visual observability:
#    - A human can see a clean axis system: colored row labels on the left, teal
#      shape labels on the top, and identical marker dots in the interior at their
#      intersections.
#    - Across 2-4 samples, "same row => same color family" and "same column =>
#      same shape family" are both directly visible without using hidden metadata.
# f) Visual cleanliness:
#    - The input is partitioned into three zones: top legend band, left legend
#      band, and central work area. The separator row/columns keep these roles
#      visually distinct.
#    - The work area uses a sparse Cartesian layout instead of scattered clutter.
# g) Change inferability:
#    - Target-color change is explicit on the left axis and stencil-shape change is
#      explicit on the top axis. Because every source marker reuses the same color,
#      the only explainable way to account for the outputs is to combine row and
#      column information.
# h) Anti-degeneration strategy:
#    - If a solver ignored row alignment, it could not explain multiple target
#      colors produced from the same input marker color.
#    - If a solver ignored column alignment, it could not explain multiple stencil
#      shapes produced from the same input marker color.
#    - If a solver collapsed back to the original one-rule-per-source-color logic,
#      it would predict at most one (shape, color) pair, while each generated
#      sample contains a Cartesian product of multiple row colors and column shapes.
# i) New solver assumptions guaranteed by the generator:
#    - All active row labels use the same source marker in column 0 and unique
#      target colors in column 1.
#    - All active column labels place the same source marker in row 0 and a valid
#      3x3 teal stencil directly below it, centered on that column.
#    - Every interior source marker lies exactly at an active-row/active-column
#      intersection, and bystanders never use the shared source marker color.


PATTERN_COLOR = 8
NON_BG_NON_TEAL_COLORS = [1, 2, 3, 4, 5, 6, 7, 9]

GRID_DIM_MAX = 20
LAYOUT_CHOICES = [(2, 2), (2, 3), (3, 2), (3, 3)]
LAYOUT_WEIGHTS = [3, 2, 2, 1]

TOP_MARKER_ROW = 0
TOP_TEMPLATE_R0 = 1
TEMPLATE_H = 3
TEMPLATE_W = 3
ANCHOR_LOCAL = (1, 1)
TOP_BAND_H = TOP_TEMPLATE_R0 + TEMPLATE_H
TOP_SEPARATOR = 1
WORK_R0 = TOP_BAND_H + TOP_SEPARATOR

LEFT_TARGET_COL = 1
LEFT_BAND_W = 2
LEFT_SEPARATOR = 2
WORK_C0 = LEFT_BAND_W + LEFT_SEPARATOR

ROW_GAP = 3
COL_GAP = 4
SCATTERED_MARKER_COLOR_COUNT_MIN = 4
BYSTANDER_MAX = 4
BYSTANDER_RETRIES = 80
POSITION_RETRIES = 200
OUTER_RETRIES = 400

_DIRS4 = [(-1, 0), (1, 0), (0, -1), (0, 1)]


def _reserve_4neighbors(reserved: set[tuple[int, int]], cells: list[tuple[int, int]]) -> None:
    for r, c in cells:
        for dr, dc in _DIRS4:
            reserved.add((int(r) + dr, int(c) + dc))


@lru_cache(maxsize=None)
def _valid_stencil_masks() -> tuple[np.ndarray, ...]:
    masks: list[np.ndarray] = []
    anchor_index = ANCHOR_LOCAL[0] * TEMPLATE_W + ANCHOR_LOCAL[1]
    for area in range(3, 6):
        for cells in combinations(range(TEMPLATE_H * TEMPLATE_W), area):
            if anchor_index not in cells:
                continue
            mask = np.zeros((TEMPLATE_H, TEMPLATE_W), dtype=bool)
            for cell in cells:
                mask[cell // TEMPLATE_W, cell % TEMPLATE_W] = True
            if not mask[0, :].any() or not mask[-1, :].any():
                continue
            _labels, count = label_mask(mask, connectivity=8)
            if int(count) == 1:
                masks.append(mask)
    return tuple(masks)


def _decode_row_rules(grid: np.ndarray, bg: int) -> tuple[int, dict[int, int]]:
    row_rules: dict[int, int] = {}
    source_marker: int | None = None
    h, _w = grid.shape
    for r in range(WORK_R0, h):
        src = int(grid[int(r), 0])
        tgt = int(grid[int(r), LEFT_TARGET_COL])
        if src == int(bg) and tgt == int(bg):
            continue
        if src in (int(bg), int(PATTERN_COLOR)) or tgt in (int(bg), int(PATTERN_COLOR)) or src == tgt:
            raise ValueError("invalid left-edge row legend")
        if source_marker is None:
            source_marker = src
        elif int(source_marker) != src:
            raise ValueError("row legends must share one marker color")
        row_rules[int(r)] = tgt
    if source_marker is None or not row_rules:
        raise ValueError("row legends not found")
    return int(source_marker), row_rules


def _decode_col_rules(grid: np.ndarray, bg: int, source_marker: int) -> dict[int, np.ndarray]:
    col_rules: dict[int, np.ndarray] = {}
    _h, w = grid.shape
    for c in range(WORK_C0 + 1, w - 1):
        if int(grid[TOP_MARKER_ROW, int(c)]) == int(bg):
            continue
        if int(grid[TOP_MARKER_ROW, int(c)]) != int(source_marker):
            raise ValueError("top-edge marker mismatch")
        window = grid[
            TOP_TEMPLATE_R0 : TOP_TEMPLATE_R0 + TEMPLATE_H,
            int(c) - 1 : int(c) + 2,
        ]
        mask = window == int(PATTERN_COLOR)
        if not bool(mask[ANCHOR_LOCAL]):
            raise ValueError("column legend stencil missing fixed anchor")
        rel = np.argwhere(mask).astype(int) - np.array(ANCHOR_LOCAL, dtype=int)
        col_rules[int(c)] = np.asarray(rel, dtype=int)
    if not col_rules:
        raise ValueError("column legends not found")
    return col_rules


def _work_mask(shape: tuple[int, int]) -> np.ndarray:
    h, w = map(int, shape)
    mask = np.zeros((h, w), dtype=bool)
    mask[WORK_R0:, WORK_C0:] = True
    return mask


def main(grid: np.ndarray) -> np.ndarray:
    """
    Read color labels from the left edge and stencil labels from the top edge, then fuse them at interior intersections.

    Input structure:
    - A black grid is partitioned into a top legend band, a left legend band, and a central work area.
    - Each active work row has a left-edge legend cell pair: shared source marker in column 0 and that row's target color in column 1.
    - Each active work column has a top-edge legend: the same source marker in row 0 and a 3x3 teal mini-stencil directly underneath it.
    - Interior source pixels are the shared marker color and appear at intersections of active rows and active columns.

    Rule:
    - Decode every active row into a target-color rule and every active column into a stencil-shape rule.
    - For each interior marker pixel on an active row and active column, stamp the column's teal stencil centered on that pixel and recolor the stamp to the row's target color.
    - Remove the entire legend system and all source markers; preserve unrelated colored bystanders in the work area.

    Key insight: A source pixel is not resolved by its own color at all. Its row chooses the output color, its column chooses the output shape, and only the combination of both edge legends explains the result.
    """
    g = np.asarray(grid, dtype=int)
    bg = get_bg(g)
    h, w = g.shape

    source_marker, row_rules = _decode_row_rules(g, bg=bg)
    col_rules = _decode_col_rules(g, bg=bg, source_marker=source_marker)

    out = np.full_like(g, int(bg))
    work = _work_mask(g.shape)

    for r, c in map(tuple, np.argwhere(work & (g != int(bg)) & (g != int(PATTERN_COLOR)))):
        r = int(r)
        c = int(c)
        color = int(g[r, c])
        if color == int(source_marker) and r in row_rules and c in col_rules:
            continue
        out[r, c] = color

    for r, target_color in row_rules.items():
        for c, rel in col_rules.items():
            if int(g[int(r), int(c)]) != int(source_marker):
                continue
            rows = int(r) + rel[:, 0]
            cols = int(c) + rel[:, 1]
            ok = (0 <= rows) & (rows < h) & (0 <= cols) & (cols < w)
            out[rows[ok], cols[ok]] = int(target_color)

    return out


def _choose_spaced_positions(
    candidates: list[int],
    count: int,
    gap: int,
    retries: int = POSITION_RETRIES,
) -> list[int] | None:
    for _ in range(retries):
        chosen: list[int] = []
        order = candidates[:]
        random.shuffle(order)
        for value in order:
            if all(abs(int(value) - int(prev)) >= gap for prev in chosen):
                chosen.append(int(value))
                if len(chosen) == count:
                    return sorted(chosen)
    return None


def _build_rules(row_count: int, col_count: int) -> tuple[int, list[dict], list[dict]]:
    palette = NON_BG_NON_TEAL_COLORS[:]
    random.shuffle(palette)
    source_marker = int(palette[0])
    row_targets = [int(c) for c in palette[1 : 1 + row_count]]

    masks = list(_valid_stencil_masks())
    random.shuffle(masks)
    col_masks = masks[:col_count]

    row_rules = [{"target": color} for color in row_targets]
    col_rules = []
    for mask in col_masks:
        rel = np.argwhere(mask).astype(int) - np.array(ANCHOR_LOCAL, dtype=int)
        col_rules.append({"mask": mask.copy(), "rel": np.asarray(rel, dtype=int)})
    return source_marker, row_rules, col_rules


def _place_legends(
    inp: np.ndarray,
    source_marker: int,
    row_rules: list[dict],
    col_rules: list[dict],
    active_rows: list[int],
    active_cols: list[int],
) -> None:
    for row, rule in zip(active_rows, row_rules):
        inp[int(row), 0] = int(source_marker)
        inp[int(row), LEFT_TARGET_COL] = int(rule["target"])
        rule["row"] = int(row)

    for col, rule in zip(active_cols, col_rules):
        inp[TOP_MARKER_ROW, int(col)] = int(source_marker)
        window = inp[
            TOP_TEMPLATE_R0 : TOP_TEMPLATE_R0 + TEMPLATE_H,
            int(col) - 1 : int(col) + 2,
        ]
        window[rule["mask"]] = int(PATTERN_COLOR)
        rule["col"] = int(col)


def _place_sources(
    inp: np.ndarray,
    source_marker: int,
    row_rules: list[dict],
    col_rules: list[dict],
) -> list[tuple[int, int]]:
    positions: list[tuple[int, int]] = []
    for row_rule in row_rules:
        row = int(row_rule["row"])
        for col_rule in col_rules:
            col = int(col_rule["col"])
            inp[row, col] = int(source_marker)
            positions.append((row, col))
    if len(positions) < SCATTERED_MARKER_COLOR_COUNT_MIN:
        raise ValueError("not enough source intersections")
    return positions


def _stamp_cells(center_r: int, center_c: int, rel: np.ndarray, shape: tuple[int, int]) -> set[tuple[int, int]]:
    h, w = map(int, shape)
    cells: set[tuple[int, int]] = set()
    for rr, cc in np.asarray(rel, dtype=int):
        nr = int(center_r) + int(rr)
        nc = int(center_c) + int(cc)
        if 0 <= nr < h and 0 <= nc < w:
            cells.add((nr, nc))
    return cells


def _stamp_footprint(
    row_rules: list[dict],
    col_rules: list[dict],
    shape: tuple[int, int],
) -> set[tuple[int, int]]:
    cells: set[tuple[int, int]] = set()
    for row_rule in row_rules:
        for col_rule in col_rules:
            cells |= _stamp_cells(int(row_rule["row"]), int(col_rule["col"]), col_rule["rel"], shape)
    return cells


def _place_bystanders(
    inp: np.ndarray,
    source_marker: int,
    row_rules: list[dict],
    col_rules: list[dict],
) -> list[tuple[int, int, int]]:
    h, w = inp.shape
    active_rows = {int(rule["row"]) for rule in row_rules}
    active_cols = {int(rule["col"]) for rule in col_rules}
    used_colors = {int(source_marker)} | {int(rule["target"]) for rule in row_rules}
    bystander_colors = [c for c in NON_BG_NON_TEAL_COLORS if c not in used_colors]
    if not bystander_colors:
        return []

    forbidden = _stamp_footprint(row_rules, col_rules, inp.shape)
    occupied = {(int(r), int(c)) for r in active_rows for c in active_cols}
    reserved: set[tuple[int, int]] = set()
    _reserve_4neighbors(reserved, list(occupied))

    candidates = [
        (r, c)
        for r in range(WORK_R0, h)
        for c in range(WORK_C0, w)
        if r not in active_rows and c not in active_cols
    ]
    random.shuffle(candidates)

    placed: list[tuple[int, int, int]] = []
    want = random.randint(0, min(BYSTANDER_MAX, len(bystander_colors)))
    for r, c in candidates:
        if len(placed) >= want:
            break
        pos = (int(r), int(c))
        if pos in occupied or pos in reserved or pos in forbidden:
            continue
        color = int(random.choice(bystander_colors))
        inp[pos] = color
        placed.append((int(r), int(c), color))
        occupied.add(pos)
        _reserve_4neighbors(reserved, [pos])
    return placed


def _build_output(
    row_rules: list[dict],
    col_rules: list[dict],
    bystanders: list[tuple[int, int, int]],
    shape: tuple[int, int],
) -> np.ndarray:
    out = np.zeros(shape, dtype=int)
    for r, c, color in bystanders:
        out[int(r), int(c)] = int(color)
    for row_rule in row_rules:
        row = int(row_rule["row"])
        target = int(row_rule["target"])
        for col_rule in col_rules:
            col = int(col_rule["col"])
            for nr, nc in _stamp_cells(row, col, col_rule["rel"], shape):
                out[int(nr), int(nc)] = target
    return out


def _gen_attempt() -> np.ndarray | None:
    row_count, col_count = random.choices(LAYOUT_CHOICES, weights=LAYOUT_WEIGHTS, k=1)[0]
    min_h = WORK_R0 + 3 + (row_count - 1) * ROW_GAP
    min_w = WORK_C0 + 3 + (col_count - 1) * COL_GAP
    h = random.randint(min_h, GRID_DIM_MAX)
    w = random.randint(min_w, GRID_DIM_MAX)

    row_candidates = list(range(WORK_R0 + 1, h - 1))
    col_candidates = list(range(WORK_C0 + 1, w - 1))
    active_rows = _choose_spaced_positions(row_candidates, row_count, ROW_GAP)
    active_cols = _choose_spaced_positions(col_candidates, col_count, COL_GAP)
    if active_rows is None or active_cols is None:
        return None

    source_marker, row_rules, col_rules = _build_rules(row_count, col_count)
    inp = np.zeros((h, w), dtype=int)

    _place_legends(inp, source_marker, row_rules, col_rules, active_rows, active_cols)
    _place_sources(inp, source_marker, row_rules, col_rules)
    bystanders = _place_bystanders(inp, source_marker, row_rules, col_rules)

    expected = _build_output(row_rules, col_rules, bystanders, inp.shape)
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
