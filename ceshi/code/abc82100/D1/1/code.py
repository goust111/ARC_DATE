# Variant of abc82100
# Dimension: D1 — 源像素与规则的匹配条件 (match_cond)
# Change: s.color == d.source_color：源像素的颜色必须等于规则示例中源色像素的颜色... → 空间位置匹配（行/列对齐）

from __future__ import annotations

import random

import numpy as np

from arc_utils import components_mask, get_bg

# Design analysis for the D1 variant
# 1. Concrete behavior:
#    This variant instantiates the "row/column alignment" idea as a row-alignment
#    task. Each rule legend still looks like a local source->target pair followed
#    by an anchor touching a teal stencil, but all scattered source pixels for that
#    rule now sit on the same row as the legend. Their own colors are intentionally
#    uninformative: every legend source pixel and every scattered source pixel uses
#    one shared marker color, while different rows carry different target colors.
#    Small trace:
#    - Original: if legends are 1->4 and 6->7, then a scattered 1 gets stencil-4
#      and a scattered 6 gets stencil-7, regardless of row.
#    - Variant: if row 3 legend is 1->4 and row 9 legend is 1->7, then any colored
#      non-legend pixel on row 3 gets stencil-4 and any colored non-legend pixel on
#      row 9 gets stencil-7, even though both rows use the same source color 1.
#    The changed step is exactly the rule-selection step after legend discovery.
# 2. Complexity calibration:
#    If scattered colors were still unique per rule, a human could keep solving by
#    color and miss the spatial change. To avoid that degeneration, the generator
#    reuses one source color across all rules and gives each row a distinct target
#    color. That creates repeated evidence that color alone is insufficient, while
#    still keeping the number of active rule rows small (1-4).
# 3. Implicit assumptions from the original solver:
#    - _discover_rules() assumes legends are still local colored pairs near teal.
#      That remains valid, so discovery is kept unchanged.
#    - The original preservation loop assumed "color in source_colors" was enough to
#      identify consumable scattered pixels. That is false now because the same color
#      appears under multiple rules. The new solver instead treats rows with legends
#      as consumable rows.
#    - The original stamping loop iterated over every pixel of rule["source_color"].
#      That would incorrectly stamp all rows when source colors repeat, so it is
#      replaced by an iteration over colored non-legend pixels on the rule's row.
# 4. Placement legality:
#    The generator enforces one rule per row, horizontal legends, and 4-neighbor
#    spacing between objects. Bystanders are never placed on rule rows, because the
#    solver interprets every non-legend colored pixel on a rule row as a source.
#    Legend stencils must fit fully in-bounds and must not overlap the source-target
#    pair. Output stamps may still clip at borders, matching the original task.
# 5. Visual observability:
#    The key visual cue is "same row, same result". Each training sample shows a
#    legend at the left part of a row and several same-row dots elsewhere; another
#    row reuses the same source color but produces a different target-colored stamp.
#    That makes the row grouping visible from 2-4 examples without hidden numeric
#    ordering or color-id tricks.
# 6. Visual cleanliness:
#    Stencils are kept compact, rule rows are separated, and scattered sources are
#    isolated along their rows. This avoids mixing multiple legends or bystanders in
#    the same row, which would make ownership ambiguous.
# 7. Color-change inferability:
#    Only target colors vary by rule; source colors are deliberately shared. That
#    keeps the color story simple: source color is not the key, row membership is.
# 8. Anti-degeneration strategy:
#    Reusing one source color across all rules forces the new row-based control
#    variable to matter. On the same input, the original color-matching solver would
#    merge rules that share a source color and therefore produce a different output.
# 9. New solver assumptions to satisfy:
#    - Every consumable non-legend colored pixel lies on exactly one rule row.
#    - Each rule row contains exactly one legend pair discovered by _discover_rules().
#    - Bystanders appear only off rule rows.
#    The generator constructs inputs to guarantee all three.


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


def main(grid: np.ndarray) -> np.ndarray:
    """
    Use each row's local legend to stamp its teal stencil onto aligned source pixels.

    Input structure:
    - Black background with horizontal source->target legend pairs and nearby teal (8) stencil components.
    - Each legend remains a 4-neighbor colored pair whose continued direction reaches an anchor touching a teal stencil.
    - Extra colored pixels appear elsewhere on legend rows; their colors may repeat across multiple rows.

    Rule:
    - Discover each legend's target color and teal stencil exactly as in the original task.
    - For a discovered legend, every non-legend colored pixel on that same row uses that rule, regardless of its color.
    - Translate the stencil so the legend anchor lands on the aligned pixel, recolor it to the legend target color, and erase all legend/stencil pixels.

    Key insight: Rule selection is controlled by row alignment with a legend, not by matching the legend's source color.
    """
    g = np.asarray(grid, dtype=int)
    bg = get_bg(g)
    pattern_color = 8
    h, w = g.shape

    rules = _discover_rules(g, bg=bg, pattern_color=pattern_color)
    legend_pixels: set[tuple[int, int]] = set()
    row_rules: dict[int, dict] = {}
    for rule in rules:
        legend_pixels.add(rule["source_pos"])
        legend_pixels.add(rule["target_pos"])
        row_rules[int(rule["source_pos"][0])] = rule

    out = np.full_like(g, int(bg))

    # Colored pixels off the rule rows are ordinary bystanders and survive unchanged.
    for r, c in map(tuple, np.argwhere((g != int(bg)) & (g != pattern_color))):
        r = int(r)
        c = int(c)
        if (r, c) in legend_pixels or r in row_rules:
            continue
        out[r, c] = int(g[r, c])

    for row, rule in row_rules.items():
        ar, ac = rule["anchor"]
        pattern = rule["pattern"]
        row_shift = int(row) - int(ar)
        for sc in range(w):
            if (int(row), int(sc)) in legend_pixels:
                continue
            color = int(g[int(row), int(sc)])
            if color in (int(bg), int(pattern_color)):
                continue
            col_shift = int(sc) - int(ac)
            rows = pattern[:, 0] + row_shift
            cols = pattern[:, 1] + col_shift
            ok = (0 <= rows) & (rows < h) & (0 <= cols) & (cols < w)
            out[rows[ok], cols[ok]] = int(rule["target_color"])

    return out


# --------------- generator (D1 row-alignment variant) ---------------

GRID_DIM_MAX = 20
MIN_DIM_BY_RULES = {1: 8, 2: 10, 3: 14, 4: 18}
RULE_COUNT_WEIGHTS = [1, 1, 2, 2, 3, 3, 4]
NON_BG_NON_TEAL_COLORS = [1, 2, 3, 4, 5, 6, 7, 9]

STENCIL_MAX_CAP = 8
STENCIL_MIN_AREA_DIV = 100
STENCIL_MAX_AREA_DIV = 35
STENCIL_MAX_HEIGHT = 4
STENCIL_MAX_WIDTH = 4

SCATTERED_MIN = 3
SCATTERED_MAX_CAP = 6
BYSTANDER_MAX = 4

RULE_ROW_GAP = 3
LEGEND_RETRIES = 300
STENCIL_RETRIES = 80
SCATTERED_RETRIES_PER_PIXEL = 80
BYSTANDER_RETRIES = 60
ROW_SELECTION_RETRIES = 200
OUTER_RETRIES = 400

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


def _choose_rule_rows(h: int, num_rules: int) -> list[int] | None:
    candidates = list(range(h))
    for _ in range(ROW_SELECTION_RETRIES):
        chosen: list[int] = []
        order = candidates[:]
        random.shuffle(order)
        for row in order:
            if all(abs(int(row) - int(prev)) >= RULE_ROW_GAP for prev in chosen):
                chosen.append(int(row))
                if len(chosen) == num_rules:
                    return sorted(chosen)
    return None


def _build_rules(num_rules: int, area: int, width: int):
    palette = NON_BG_NON_TEAL_COLORS[:]
    random.shuffle(palette)
    source_marker = palette[0]
    target_colors = palette[1 : 1 + num_rules]

    max_s = min(STENCIL_MAX_CAP, max(2, area // STENCIL_MAX_AREA_DIV))
    min_s = min(max(1, area // STENCIL_MIN_AREA_DIV), max_s)
    max_scattered = min(SCATTERED_MAX_CAP, max(SCATTERED_MIN, (width - 2) // 3))

    rules = []
    for i in range(num_rules):
        stencil, anchor = _make_stencil(min_s, max_s)
        rel = [(int(r - anchor[0]), int(c - anchor[1])) for r, c in stencil]
        rules.append(
            {
                "src": int(source_marker),
                "tgt": int(target_colors[i]),
                "rel": rel,
                "n_sc": random.randint(SCATTERED_MIN, max_scattered),
            }
        )
    used_colors = {int(source_marker), *map(int, target_colors)}
    return rules, used_colors


def _place_legends(rules, rule_rows, inp, occupied, reserved):
    h, w = inp.shape
    for rule, row in zip(rules, rule_rows):
        placed = False
        for _ in range(LEGEND_RETRIES):
            sc = random.randint(0, w - 3)
            sr = int(row)
            tr, tc = sr, sc + 1
            ar, ac = sr, sc + 2

            teal_cells = set()
            ok = True
            for rr, cc in rule["rel"]:
                nr, nc = ar + int(rr), ac + int(cc)
                if not (0 <= nr < h and 0 <= nc < w):
                    ok = False
                    break
                teal_cells.add((int(nr), int(nc)))
            if not ok:
                continue
            if (sr, sc) in teal_cells or (tr, tc) in teal_cells:
                continue

            all_cells = {(sr, sc), (tr, tc), (ar, ac)} | teal_cells
            if all_cells & occupied or all_cells & reserved:
                continue

            inp[sr, sc] = int(rule["src"])
            inp[tr, tc] = int(rule["tgt"])
            for nr, nc in teal_cells:
                inp[nr, nc] = 8
            occupied.update(all_cells)
            _reserve_4neighbors(reserved, all_cells)
            rule["row"] = int(row)
            rule["legend_source"] = (sr, sc)
            rule["legend_target"] = (tr, tc)
            rule["anchor"] = (ar, ac)
            placed = True
            break
        if not placed:
            return False
    return True


def _place_scattered(rules, inp, occupied, reserved):
    h, w = inp.shape
    del h
    for rule in rules:
        row = int(rule["row"])
        positions = []
        for _ in range(int(rule["n_sc"]) * SCATTERED_RETRIES_PER_PIXEL):
            if len(positions) >= int(rule["n_sc"]):
                break
            c = random.randint(0, w - 1)
            pos = (row, int(c))
            if pos in occupied or pos in reserved:
                continue
            positions.append(pos)
            occupied.add(pos)
            _reserve_4neighbors(reserved, [pos])
            inp[pos] = int(rule["src"])
        if len(positions) < SCATTERED_MIN:
            return False
        rule["scattered"] = positions
    return True


def _place_bystanders(used_colors, rule_rows, inp, occupied, reserved):
    bystander_colors = [c for c in NON_BG_NON_TEAL_COLORS if c not in used_colors]
    if not bystander_colors:
        return []
    h, w = inp.shape
    allowed_rows = [r for r in range(h) if r not in set(map(int, rule_rows))]
    if not allowed_rows:
        return []
    n = random.randint(0, min(BYSTANDER_MAX, len(bystander_colors)))
    placed = []
    for _ in range(n):
        for _ in range(BYSTANDER_RETRIES):
            r = random.choice(allowed_rows)
            c = random.randint(0, w - 1)
            pos = (int(r), int(c))
            if pos in occupied or pos in reserved:
                continue
            color = random.choice(bystander_colors)
            inp[pos] = int(color)
            placed.append((int(r), int(c), int(color)))
            occupied.add(pos)
            _reserve_4neighbors(reserved, [pos])
            break
    return placed


def _build_output(rules, bystander_pos, h: int, w: int):
    output = np.zeros((h, w), dtype=int)
    for r, c, color in bystander_pos:
        output[int(r), int(c)] = int(color)
    for rule in rules:
        for sr, sc in rule["scattered"]:
            for rr, cc in rule["rel"]:
                nr, nc = int(sr) + int(rr), int(sc) + int(cc)
                if 0 <= nr < h and 0 <= nc < w:
                    output[nr, nc] = int(rule["tgt"])
    return output


def _gen_attempt():
    num_rules = random.choice(RULE_COUNT_WEIGHTS)
    min_dim = MIN_DIM_BY_RULES[num_rules]
    h = random.randint(min_dim, GRID_DIM_MAX)
    w = random.randint(max(8, min_dim), GRID_DIM_MAX)

    rules, used_colors = _build_rules(num_rules, h * w, w)
    rule_rows = _choose_rule_rows(h, num_rules)
    if rule_rows is None:
        return None

    occupied: set[tuple[int, int]] = set()
    reserved: set[tuple[int, int]] = set()
    inp = np.zeros((h, w), dtype=int)

    if not _place_legends(rules, rule_rows, inp, occupied, reserved):
        return None
    if not _place_scattered(rules, inp, occupied, reserved):
        return None

    bystander_pos = _place_bystanders(used_colors, rule_rows, inp, occupied, reserved)
    output = _build_output(rules, bystander_pos, h, w)
    try:
        if np.array_equal(main(inp), output):
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
