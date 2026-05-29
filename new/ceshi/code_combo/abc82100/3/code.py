# Combo variant of abc82100 (index 3)
# combo_spec: D1/1+D10/1
# Name: 行列对齐匹配+边缘规则分布

from __future__ import annotations

import random

import numpy as np

from arc_utils import components_mask, get_bg

# Design analysis for the combo variant
# 1. Concrete fused behavior:
#    The original task matches scattered source pixels by color. This combo removes
#    that cue and relocates legends onto the border. A legend on the left or right
#    edge governs its whole row; a legend on the top or bottom edge governs its
#    whole column. All scattered source pixels reuse one shared marker color, so
#    the solver must read which border ruler they align with.
#    Small trace:
#    - Original: a scattered pixel of color 1 always uses the 1->4 legend, even if
#      another nearby row carries a 1->7 legend.
#    - Fused variant: if row 5 has a left-edge legend marker->4 and column 9 has a
#      top-edge legend marker->7, then a center pixel at (5, 12) uses the row rule
#      and a center pixel at (11, 9) uses the column rule. Their shared input color
#      is intentionally non-informative.
# 2. Complexity calibration:
#    Shared source color is necessary; otherwise the puzzle collapses back to the
#    original color-matching rule. At the same time, target colors stay distinct so
#    the output still gives repeated visual evidence that different aligned lines
#    produce different recolored stamps.
# 3. Implicit assumption audit against the original main():
#    - _discover_rules() assumes legends are still local source->target pairs with a
#      teal stencil beyond the target. That remains true, so legend discovery can be
#      preserved unchanged.
#    - The original preservation logic treated every pixel of a learned source color
#      as consumable. That is now false because one shared marker color appears under
#      multiple rules. The new solver instead consumes only colored pixels aligned to
#      exactly one active border legend line.
#    - The original stamping loop iterated by source color. That would stamp every
#      rule onto every marker pixel. The fused solver therefore selects a rule from
#      border position: row for left/right legends, column for top/bottom legends.
# 4. New combined assumptions introduced by the fusion:
#    - Every usable legend source cell lies on exactly one non-corner border cell.
#    - Each active row or active column has exactly one discovered legend.
#    - Every consumable center source pixel aligns with exactly one active rule line;
#      intersections of active rows and active columns are kept empty.
#    - Bystanders never sit on active rows or active columns, because those lines are
#      interpreted as source-bearing rulers.
# 5. Placement legality:
#    Legends use border source cells, inward target/anchor cells, and compact teal
#    stencils that fully fit in-bounds. Scattered sources are placed only inside a
#    central band and avoid occupied cells, 4-neighbors, and row/column-rule
#    intersections. This prevents accidental extra legends and guarantees the solver's
#    alignment test has a single answer per source.
# 6. Visual observability:
#    The border placement creates explicit rulers: a human can see that edge legends
#    label particular rows or columns, while the center contains the pixels to be
#    transformed. Repeating the same source color across multiple legend lines makes
#    it visually obvious that position, not color, controls which stencil/color pair
#    applies.
# 7. Visual cleanliness:
#    Source pixels stay in the middle of the board, legends stay on the perimeter,
#    and active lines are spaced apart. That keeps the grid segmented into readable
#    zones and avoids a cluttered mixture of legends, sources, and bystanders.
# 8. Color-change inferability:
#    Only one source marker color is reused; target colors differ per legend line.
#    This keeps the color story simple enough for ARC: the solver only has to infer
#    which aligned legend owns a source, then copy that legend's target color and
#    stencil.
# 9. Anti-degeneration check to enforce in generation:
#    Multi-rule samples must exist, rules must share the same source marker color,
#    and legends must live on edges while sources stay in the center. Under those
#    conditions, the original color-based solver and the fused alignment solver
#    disagree on the same inputs, so the combo does not collapse back to the seed.


def _chebyshev(a: tuple[int, int], b: tuple[int, int]) -> int:
    return max(abs(int(a[0]) - int(b[0])), abs(int(a[1]) - int(b[1])))


def _anchor_distance(pattern: np.ndarray, anchor: tuple[int, int]) -> int:
    return min(_chebyshev((int(r), int(c)), anchor) for r, c in pattern)


def _discover_rules(grid: np.ndarray, bg: int, pattern_color: int = 8) -> list[dict]:
    h, w = grid.shape
    patterns = components_mask(grid == pattern_color, connectivity=8)
    rules: list[dict] = []
    seen: set[tuple[tuple[int, int], tuple[int, int], int]] = set()

    for r in range(h):
        for c in range(w):
            source_color = int(grid[r, c])
            if source_color in (int(bg), int(pattern_color)):
                continue

            for dr, dc in ((-1, 0), (1, 0), (0, -1), (0, 1)):
                tr = r + dr
                tc = c + dc
                if not (0 <= tr < h and 0 <= tc < w):
                    continue

                target_color = int(grid[tr, tc])
                if target_color in (int(bg), int(pattern_color)) or target_color == source_color:
                    continue

                anchor = (tr + dr, tc + dc)
                candidates: list[tuple[int, int, np.ndarray]] = []
                for i, pattern in enumerate(patterns):
                    dist = _anchor_distance(pattern, anchor)
                    if dist <= 1:
                        candidates.append((int(dist), int(i), pattern))

                if not candidates:
                    continue

                candidates.sort(key=lambda item: (item[0], len(item[2])))
                _dist, pattern_index, pattern = candidates[0]
                key = ((int(r), int(c)), (int(tr), int(tc)), int(pattern_index))
                if key in seen:
                    continue
                seen.add(key)
                rules.append(
                    {
                        "source_color": int(source_color),
                        "target_color": int(target_color),
                        "source_pos": (int(r), int(c)),
                        "target_pos": (int(tr), int(tc)),
                        "anchor": (int(anchor[0]), int(anchor[1])),
                        "pattern": np.asarray(pattern, dtype=int),
                    }
                )

    return rules


def _classify_edge_rule(rule: dict, h: int, w: int) -> tuple[str, int] | None:
    sr, sc = map(int, rule["source_pos"])
    if sr in (0, h - 1) and 0 < sc < w - 1:
        return ("col", int(sc))
    if sc in (0, w - 1) and 0 < sr < h - 1:
        return ("row", int(sr))
    return None


def main(grid: np.ndarray) -> np.ndarray:
    """
    Use border legends as row/column rulers for stamping their teal templates.

    Input structure:
    - Black background with teal (8) stencil components attached to local source->target legend pairs.
    - Every usable legend is anchored on a non-corner border cell: left/right edge legends define rows, top/bottom edge legends define columns.
    - Extra colored pixels are concentrated in the center and may all share the same color, so color alone does not identify a rule.

    Rule:
    - Discover each legend's target color, anchor, and teal stencil exactly as in the seed task.
    - A non-legend colored pixel uses the unique legend aligned with it: same row for a left/right border legend, or same column for a top/bottom border legend.
    - Translate that legend's stencil so its anchor lands on the aligned pixel, recolor it to the legend target color, and erase all legend/stencil pixels.

    Key insight: The border legends act like spatial rulers; matching depends on row/column alignment with an edge legend, not on the source pixel's own color.
    """
    g = np.asarray(grid, dtype=int)
    bg = get_bg(g)
    pattern_color = 8
    h, w = g.shape

    discovered = _discover_rules(g, bg=bg, pattern_color=pattern_color)
    legend_pixels: set[tuple[int, int]] = set()
    row_buckets: dict[int, list[dict]] = {}
    col_buckets: dict[int, list[dict]] = {}
    for rule in discovered:
        legend_pixels.add(tuple(map(int, rule["source_pos"])))
        legend_pixels.add(tuple(map(int, rule["target_pos"])))
        edge_info = _classify_edge_rule(rule, h, w)
        if edge_info is None:
            continue
        axis, key = edge_info
        if axis == "row":
            row_buckets.setdefault(int(key), []).append(rule)
        else:
            col_buckets.setdefault(int(key), []).append(rule)

    row_rules = {row: rules[0] for row, rules in row_buckets.items() if len(rules) == 1}
    col_rules = {col: rules[0] for col, rules in col_buckets.items() if len(rules) == 1}

    out = np.full_like(g, int(bg))

    for r, c in map(tuple, np.argwhere((g != int(bg)) & (g != pattern_color))):
        r = int(r)
        c = int(c)
        if (r, c) in legend_pixels:
            continue

        matches: list[dict] = []
        if r in row_rules:
            matches.append(row_rules[r])
        if c in col_rules:
            matches.append(col_rules[c])

        if len(matches) != 1:
            out[r, c] = int(g[r, c])
            continue

        rule = matches[0]
        ar, ac = map(int, rule["anchor"])
        pattern = np.asarray(rule["pattern"], dtype=int)
        rows = pattern[:, 0] + (r - ar)
        cols = pattern[:, 1] + (c - ac)
        ok = (0 <= rows) & (rows < h) & (0 <= cols) & (cols < w)
        out[rows[ok], cols[ok]] = int(rule["target_color"])

    return out


# --------------- generator (combo D1/1 + D10/1) ---------------

GRID_DIM_MAX = 20
MIN_DIM_BY_RULES = {2: 11, 3: 15, 4: 18}
RULE_COUNT_WEIGHTS = [2, 2, 3, 3, 4]
NON_BG_NON_TEAL_COLORS = [1, 2, 3, 4, 5, 6, 7, 9]

STENCIL_MAX_CAP = 6
STENCIL_MIN_AREA_DIV = 120
STENCIL_MAX_AREA_DIV = 40
STENCIL_MAX_HEIGHT = 4
STENCIL_MAX_WIDTH = 4

SCATTERED_MIN = 2
SCATTERED_MAX_CAP = 4
BYSTANDER_MAX = 3

CENTER_MARGIN = 3
LINE_GAP = 2
LAYOUT_RETRIES = 200
STENCIL_RETRIES = 80
LEGEND_RETRIES = 8
SCATTER_RETRIES = 120
BYSTANDER_RETRIES = 80
OUTER_RETRIES = 600

_DIRS4 = [(-1, 0), (1, 0), (0, -1), (0, 1)]
_DIRS8 = [(-1, -1), (-1, 0), (-1, 1), (0, -1), (0, 1), (1, -1), (1, 0), (1, 1)]


def _reserve_4neighbors(reserved: set[tuple[int, int]], cells) -> None:
    for r, c in cells:
        for dr, dc in _DIRS4:
            reserved.add((int(r) + dr, int(c) + dc))


def _make_stencil(min_pixels: int, max_pixels: int):
    for _ in range(STENCIL_RETRIES):
        n = random.randint(min_pixels, max_pixels)
        coords = {(0, 0)}
        for _ in range(n - 1):
            frontier = set()
            for r, c in coords:
                for dr, dc in _DIRS8:
                    p = (r + dr, c + dc)
                    if p not in coords:
                        frontier.add(p)
            coords.add(random.choice(list(frontier)))
        rows = [r for r, _ in coords]
        cols = [c for _, c in coords]
        if max(rows) - min(rows) + 1 > STENCIL_MAX_HEIGHT:
            continue
        if max(cols) - min(cols) + 1 > STENCIL_MAX_WIDTH:
            continue
        coords_list = sorted(coords)
        anchor = random.choice(coords_list)
        return coords_list, anchor
    raise RuntimeError("failed to make compact stencil")


def _sample_spaced(candidates: list[int], count: int, gap: int) -> list[int] | None:
    for _ in range(LAYOUT_RETRIES):
        order = candidates[:]
        random.shuffle(order)
        chosen: list[int] = []
        for value in order:
            if all(abs(int(value) - int(prev)) >= gap for prev in chosen):
                chosen.append(int(value))
                if len(chosen) == count:
                    return sorted(chosen)
    return None


def _build_rules(num_rules: int, area: int):
    palette = NON_BG_NON_TEAL_COLORS[:]
    random.shuffle(palette)
    source_marker = int(palette[0])
    target_colors = [int(c) for c in palette[1 : 1 + num_rules]]

    max_s = min(STENCIL_MAX_CAP, max(2, area // STENCIL_MAX_AREA_DIV))
    min_s = min(max(1, area // STENCIL_MIN_AREA_DIV), max_s)

    rules = []
    for target_color in target_colors:
        stencil, anchor = _make_stencil(min_s, max_s)
        rel = [(int(r - anchor[0]), int(c - anchor[1])) for r, c in stencil]
        rules.append({"src": int(source_marker), "tgt": int(target_color), "rel": rel})
    used_colors = {int(source_marker), *target_colors}
    return rules, used_colors


def _choose_layout(h: int, w: int, num_rules: int):
    row_candidates = list(range(CENTER_MARGIN, h - CENTER_MARGIN))
    col_candidates = list(range(CENTER_MARGIN, w - CENTER_MARGIN))
    if not row_candidates or not col_candidates:
        return None

    max_rows = (len(row_candidates) + LINE_GAP - 1) // LINE_GAP
    max_cols = (len(col_candidates) + LINE_GAP - 1) // LINE_GAP
    row_lo = max(1, num_rules - max_cols)
    row_hi = min(max_rows, num_rules - 1)
    if row_lo > row_hi:
        return None

    for _ in range(LAYOUT_RETRIES):
        row_count = random.randint(row_lo, row_hi)
        col_count = num_rules - row_count
        rows = _sample_spaced(row_candidates, row_count, LINE_GAP)
        cols = _sample_spaced(col_candidates, col_count, LINE_GAP)
        if rows is None or cols is None:
            continue
        return rows, cols
    return None


def _legend_triplet(rule: dict, h: int, w: int, side: str):
    if rule["axis"] == "row":
        row = int(rule["line"])
        if side == "left":
            return (row, 0), (row, 1), (row, 2)
        return (row, w - 1), (row, w - 2), (row, w - 3)

    col = int(rule["line"])
    if side == "top":
        return (0, col), (1, col), (2, col)
    return (h - 1, col), (h - 2, col), (h - 3, col)


def _place_legends(rules, inp, occupied, reserved):
    h, w = inp.shape
    for rule in rules:
        sides = ["left", "right"] if rule["axis"] == "row" else ["top", "bottom"]
        random.shuffle(sides)
        placed = False
        for _ in range(LEGEND_RETRIES):
            for side in sides:
                source_pos, target_pos, anchor = _legend_triplet(rule, h, w, side)
                teal_cells = set()
                ok = True
                for rr, cc in rule["rel"]:
                    nr = int(anchor[0]) + int(rr)
                    nc = int(anchor[1]) + int(cc)
                    if not (0 <= nr < h and 0 <= nc < w):
                        ok = False
                        break
                    teal_cells.add((int(nr), int(nc)))
                if not ok:
                    continue
                if source_pos in teal_cells or target_pos in teal_cells:
                    continue
                dr = int(target_pos[0]) - int(source_pos[0])
                dc = int(target_pos[1]) - int(source_pos[1])
                reverse_anchor = (int(source_pos[0]) - dr, int(source_pos[1]) - dc)
                if any(_chebyshev(cell, reverse_anchor) <= 1 for cell in teal_cells):
                    continue
                all_cells = {source_pos, target_pos, anchor} | teal_cells
                if all_cells & occupied or all_cells & reserved:
                    continue

                inp[source_pos] = int(rule["src"])
                inp[target_pos] = int(rule["tgt"])
                for cell in teal_cells:
                    inp[cell] = 8
                occupied.update(all_cells)
                _reserve_4neighbors(reserved, all_cells)
                rule["side"] = side
                rule["source_pos"] = source_pos
                rule["target_pos"] = target_pos
                rule["anchor"] = anchor
                placed = True
                break
            if placed:
                break
        if not placed:
            return False
    return True


def _place_scattered(rules, active_rows, active_cols, inp, occupied, reserved):
    h, w = inp.shape
    center_rows = list(range(CENTER_MARGIN, h - CENTER_MARGIN))
    center_cols = list(range(CENTER_MARGIN, w - CENTER_MARGIN))
    active_row_set = set(map(int, active_rows))
    active_col_set = set(map(int, active_cols))

    row_capacity = [c for c in center_cols if c not in active_col_set]
    col_capacity = [r for r in center_rows if r not in active_row_set]
    if (len(row_capacity) + 1) // 2 < SCATTERED_MIN:
        return False
    if (len(col_capacity) + 1) // 2 < SCATTERED_MIN:
        return False

    row_max = min(SCATTERED_MAX_CAP, max(SCATTERED_MIN, (len(row_capacity) + 1) // 2))
    col_max = min(SCATTERED_MAX_CAP, max(SCATTERED_MIN, (len(col_capacity) + 1) // 2))

    for rule in rules:
        positions = []
        if rule["axis"] == "row":
            candidates = [c for c in center_cols if c not in active_col_set]
            n_sc = random.randint(SCATTERED_MIN, row_max)
            for _ in range(SCATTER_RETRIES):
                chosen = _sample_spaced(candidates, n_sc, 2)
                if chosen is None:
                    continue
                trial = [(int(rule["line"]), int(c)) for c in chosen]
                if any(pos in occupied or pos in reserved for pos in trial):
                    continue
                positions = trial
                break
        else:
            candidates = [r for r in center_rows if r not in active_row_set]
            n_sc = random.randint(SCATTERED_MIN, col_max)
            for _ in range(SCATTER_RETRIES):
                chosen = _sample_spaced(candidates, n_sc, 2)
                if chosen is None:
                    continue
                trial = [(int(r), int(rule["line"])) for r in chosen]
                if any(pos in occupied or pos in reserved for pos in trial):
                    continue
                positions = trial
                break

        if len(positions) < SCATTERED_MIN:
            return False

        rule["scattered"] = positions
        for pos in positions:
            inp[pos] = int(rule["src"])
            occupied.add(pos)
        _reserve_4neighbors(reserved, positions)
    return True


def _place_bystanders(used_colors, active_rows, active_cols, inp, occupied, reserved):
    bystander_colors = [int(c) for c in NON_BG_NON_TEAL_COLORS if c not in used_colors]
    if not bystander_colors:
        return []

    h, w = inp.shape
    active_row_set = set(map(int, active_rows))
    active_col_set = set(map(int, active_cols))
    placed = []
    n = random.randint(0, BYSTANDER_MAX)
    for _ in range(n):
        for _ in range(BYSTANDER_RETRIES):
            r = random.randint(1, h - 2)
            c = random.randint(1, w - 2)
            pos = (int(r), int(c))
            if r in active_row_set or c in active_col_set:
                continue
            if pos in occupied or pos in reserved:
                continue
            color = random.choice(bystander_colors)
            inp[pos] = int(color)
            placed.append((int(r), int(c), int(color)))
            occupied.add(pos)
            _reserve_4neighbors(reserved, [pos])
            break
    return placed


def _build_output(rules, bystanders, h: int, w: int):
    out = np.zeros((h, w), dtype=int)
    for r, c, color in bystanders:
        out[int(r), int(c)] = int(color)
    for rule in rules:
        for sr, sc in rule["scattered"]:
            for rr, cc in rule["rel"]:
                nr = int(sr) + int(rr)
                nc = int(sc) + int(cc)
                if 0 <= nr < h and 0 <= nc < w:
                    out[nr, nc] = int(rule["tgt"])
    return out


def _gen_attempt():
    num_rules = random.choice(RULE_COUNT_WEIGHTS)
    min_dim = MIN_DIM_BY_RULES[num_rules]
    h = random.randint(min_dim, GRID_DIM_MAX)
    w = random.randint(min_dim, GRID_DIM_MAX)

    layout = _choose_layout(h, w, num_rules)
    if layout is None:
        return None
    rows, cols = layout

    rules, used_colors = _build_rules(num_rules, h * w)
    slots = []
    for row in rows:
        slots.append({"axis": "row", "line": int(row)})
    for col in cols:
        slots.append({"axis": "col", "line": int(col)})
    random.shuffle(slots)
    for rule, slot in zip(rules, slots):
        rule.update(slot)

    occupied: set[tuple[int, int]] = set()
    reserved: set[tuple[int, int]] = set()
    inp = np.zeros((h, w), dtype=int)

    if not _place_legends(rules, inp, occupied, reserved):
        return None
    if not _place_scattered(rules, rows, cols, inp, occupied, reserved):
        return None

    bystanders = _place_bystanders(used_colors, rows, cols, inp, occupied, reserved)
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
