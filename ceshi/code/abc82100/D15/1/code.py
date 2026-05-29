# Variant of abc82100
# Dimension: D15 — 边界处理规则 (meta)
# Change: 模板放置时超出网格边界的像素被截断/忽略... → 超界部分环绕到对面边缘

from __future__ import annotations

import random

import numpy as np

from arc_utils import components_mask, get_bg


# Design analysis
# ---
# source_task: none
# concretization:
#   input still contains local rule legends of the form source -> target -> anchor,
#   with a nearby teal component acting as the stencil. The only semantic change is
#   what happens when that stencil is translated onto a scattered source pixel near
#   an edge. In the original task, off-grid cells disappear. In this variant, any
#   translated cell that exits one side of the grid re-enters from the opposite side.
#   The output therefore shows target-colored stencil fragments split across edges.
# example_trace:
#   Let a rule have anchor-relative cells {(0,0), (0,1), (1,0)}. If a scattered
#   source sits at column W-1, then the translated cell at (0,1) would land at
#   column W in the original task and be clipped away; here it lands at column 0.
#   The same input thus changes from an L-shape missing one arm to an L-shape whose
#   rightmost arm wraps onto the left edge.
# complexity_calibration:
#   The legend-discovery mechanism stays unchanged so the puzzle still hinges on
#   matching colors to nearby teal stencils. To make D15 observable, the generator
#   forces at least one wrapped scattered stamp per rule and uses stencils of at
#   least two cells, avoiding degenerate samples where wraparound never matters.
# implicit_assumption_review:
#   - _discover_rules assumes legends are ordinary in-bounds 4-neighbor examples.
#     This remains valid because only copied stamps wrap, not the legend itself.
#   - The original main() computed rows/cols and then filtered with an in-bounds
#     mask. That line encoded the clipping behavior and must change to modulo writes.
#   - Preserving non-rule colored pixels still works because only source-colored
#     scattered pixels are consumed; generator keeps bystanders off all wrapped
#     stamp footprints so they remain visibly preserved.
# placement_legality:
#   Legends are still placed fully in bounds, with their teal components not
#   wrapping, so rule extraction is unambiguous. Scattered source pixels remain
#   isolated in the input via 4-neighbor reservation. Output stamps may overlap,
#   matching the original task, but bystanders are placed outside every wrapped
#   footprint to keep the preservation rule intact.
# visual_observability:
#   Each generated instance has repeated support for the new rule because every
#   rule receives one wrapped source placement. The output therefore shows target
#   patterns reappearing on the opposite border instead of simply disappearing.
# cleanliness:
#   The only cross-edge interaction is within stamped output. Input legends remain
#   spatially separated, so the periodic-boundary effect is noticeable rather than
#   being confused with rule discovery artifacts.


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


def _wrapped_stamp_cells(
    rel: list[tuple[int, int]], sr: int, sc: int, h: int, w: int
) -> tuple[set[tuple[int, int]], bool, bool]:
    cells: set[tuple[int, int]] = set()
    wraps_row = False
    wraps_col = False
    for rr, cc in rel:
        raw_r = int(sr + rr)
        raw_c = int(sc + cc)
        if not (0 <= raw_r < h):
            wraps_row = True
        if not (0 <= raw_c < w):
            wraps_col = True
        cells.add((raw_r % h, raw_c % w))
    return cells, wraps_row, wraps_col


def _pattern_to_rel(pattern: np.ndarray, anchor: tuple[int, int]) -> list[tuple[int, int]]:
    ar, ac = anchor
    return [(int(r) - int(ar), int(c) - int(ac)) for r, c in pattern]


def _rule_metrics(rule: dict, h: int, w: int) -> dict[str, int]:
    wrapped_sources = 0
    horizontal_wrap_sources = 0
    vertical_wrap_sources = 0
    wrapped_cells = 0
    for sr, sc in rule["scattered"]:
        cells, wraps_row, wraps_col = _wrapped_stamp_cells(rule["rel"], int(sr), int(sc), h, w)
        if wraps_row or wraps_col:
            wrapped_sources += 1
            wrapped_cells += len(cells)
        if wraps_col:
            horizontal_wrap_sources += 1
        if wraps_row:
            vertical_wrap_sources += 1
    return {
        "wrapped_sources": wrapped_sources,
        "horizontal_wrap_sources": horizontal_wrap_sources,
        "vertical_wrap_sources": vertical_wrap_sources,
        "wrapped_cells": wrapped_cells,
    }


def _grid_wrap_metrics(grid: np.ndarray) -> dict[str, int]:
    g = np.asarray(grid, dtype=int)
    bg = get_bg(g)
    rules = _discover_rules(g, bg=bg, pattern_color=8)
    metrics = {
        "rules": len(rules),
        "wrapped_rules": 0,
        "wrapped_sources": 0,
        "horizontal_wrap_sources": 0,
        "vertical_wrap_sources": 0,
        "wrapped_cells": 0,
    }
    for rule in rules:
        legend_pixels = {rule["source_pos"], rule["target_pos"]}
        rel = _pattern_to_rel(rule["pattern"], rule["anchor"])
        scattered = [
            (int(sr), int(sc))
            for sr, sc in map(tuple, np.argwhere(g == rule["source_color"]))
            if (int(sr), int(sc)) not in legend_pixels
        ]
        ruled = {"rel": rel, "scattered": scattered}
        rule_metrics = _rule_metrics(ruled, *g.shape)
        if rule_metrics["wrapped_sources"] > 0:
            metrics["wrapped_rules"] += 1
        for key in ("wrapped_sources", "horizontal_wrap_sources", "vertical_wrap_sources", "wrapped_cells"):
            metrics[key] += rule_metrics[key]
    return metrics


def main(grid: np.ndarray) -> np.ndarray:
    """
    Learn local color-to-color stencil rules and replay them with toroidal wrapping.

    Input structure:
    - A black background contains isolated colored pixels plus teal (8) connected
      components that serve as reusable stencil templates.
    - Each rule example is a 4-neighbor source/target color pair, and continuing
      one more step in the same direction reaches an anchor touching that teal stencil.

    Rule:
    - For every discovered rule, stamp the associated teal component onto every
      other pixel of the same source color and recolor the stamp to the target color.
    - The legend anchor is translated onto the scattered source pixel, and any
      stamp cells that leave the grid wrap around to the opposite edge.
    - Erase the teal templates and the colored pixels that were used only to teach
      the rules; preserve unrelated bystander pixels.

    Key insight: The adjacent color pair tells you which color changes into which,
    while the nearby teal component provides a stencil whose translated copies obey
    periodic boundary conditions instead of being clipped at the border.
    """
    g = np.asarray(grid, dtype=int)
    bg = get_bg(g)
    pattern_color = 8
    h, w = g.shape

    rules = _discover_rules(g, bg=bg, pattern_color=pattern_color)
    legend_pixels: set[tuple[int, int]] = set()
    source_colors: set[int] = set()
    for rule in rules:
        legend_pixels.add(rule["source_pos"])
        legend_pixels.add(rule["target_pos"])
        source_colors.add(rule["source_color"])

    out = np.full_like(g, int(bg))

    for r, c in map(tuple, np.argwhere((g != int(bg)) & (g != pattern_color))):
        r = int(r)
        c = int(c)
        color = int(g[r, c])
        if (r, c) in legend_pixels or color in source_colors:
            continue
        out[r, c] = color

    for rule in rules:
        ar, ac = rule["anchor"]
        pattern = rule["pattern"]
        for sr, sc in map(tuple, np.argwhere(g == rule["source_color"])):
            sr = int(sr)
            sc = int(sc)
            if (sr, sc) in legend_pixels:
                continue
            dr = sr - ar
            dc = sc - ac
            rows = (pattern[:, 0] + dr) % h
            cols = (pattern[:, 1] + dc) % w
            out[rows, cols] = rule["target_color"]

    return out


# --------------- generator (c) ---------------

# Grid dimensions
GRID_DIM_MAX = 20
MIN_DIM_BY_RULES = {1: 5, 2: 10, 3: 14, 4: 18}

# Rule sampling
RULE_COUNT_WEIGHTS = [1, 1, 2, 2, 3, 3, 4]
NON_BG_NON_TEAL_COLORS = [1, 2, 3, 4, 5, 6, 7, 9]

# Stencil size
STENCIL_MAX_CAP = 12
STENCIL_MIN_AREA_DIV = 100
STENCIL_MAX_AREA_DIV = 25

# Scattered source pixels per rule
SCATTERED_MIN = 3
SCATTERED_MAX_CAP = 8
SCATTERED_AREA_DIV = 30
WRAPPED_SCATTERED_PER_RULE = 1

# Bystander pixels
BYSTANDER_MAX = 5

# Placement retry budgets
LEGEND_RETRIES = 500
SCATTERED_RETRIES_PER_PIXEL = 50
BYSTANDER_RETRIES = 50
OUTER_RETRIES = 300

_DIRS4 = [(-1, 0), (1, 0), (0, -1), (0, 1)]
_DIRS8 = [(-1, -1), (-1, 0), (-1, 1), (0, -1), (0, 1), (1, -1), (1, 0), (1, 1)]


def _reserve_4neighbors(reserved: set[tuple[int, int]], cells) -> None:
    for r, c in cells:
        for dr, dc in _DIRS4:
            reserved.add((int(r) + dr, int(c) + dc))


def _make_stencil(min_pixels: int, max_pixels: int):
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
    coords_list = sorted(coords)
    return coords_list, random.choice(coords_list)


def _build_rules(num_rules: int, area: int):
    colors = NON_BG_NON_TEAL_COLORS[:]
    random.shuffle(colors)

    max_s = min(STENCIL_MAX_CAP, max(2, area // STENCIL_MAX_AREA_DIV))
    min_s = min(max(2, area // STENCIL_MIN_AREA_DIV), max_s)
    sc_max = min(SCATTERED_MAX_CAP, max(SCATTERED_MIN, area // SCATTERED_AREA_DIV))

    rules, used = [], set()
    for i in range(num_rules):
        src_c, tgt_c = colors[2 * i], colors[2 * i + 1]
        used.update((src_c, tgt_c))
        stencil, anchor = _make_stencil(min_s, max_s)
        rel = [(int(r) - int(anchor[0]), int(c) - int(anchor[1])) for r, c in stencil]
        rules.append(
            {
                "src": src_c,
                "tgt": tgt_c,
                "rel": rel,
                "n_sc": random.randint(SCATTERED_MIN, sc_max),
            }
        )
    return rules, used


def _place_legends(rules, inp, occupied, reserved):
    h, w = inp.shape
    for rule in rules:
        rel = rule["rel"]
        placed = False
        for _ in range(LEGEND_RETRIES):
            dr, dc = random.choice(_DIRS4)
            sr, sc = random.randint(0, h - 1), random.randint(0, w - 1)
            tr, tc = sr + dr, sc + dc
            ar, ac = sr + 2 * dr, sc + 2 * dc
            if not (0 <= tr < h and 0 <= tc < w and 0 <= ar < h and 0 <= ac < w):
                continue

            teal_cells = set()
            ok = True
            for rr, cc in rel:
                nr, nc = ar + rr, ac + cc
                if not (0 <= nr < h and 0 <= nc < w):
                    ok = False
                    break
                teal_cells.add((nr, nc))
            if not ok:
                continue

            all_cells = {(sr, sc), (tr, tc), (ar, ac)} | teal_cells
            if all_cells & (occupied | reserved):
                continue

            inp[sr, sc] = rule["src"]
            inp[tr, tc] = rule["tgt"]
            for nr, nc in teal_cells:
                inp[nr, nc] = 8
            occupied.update(all_cells)
            _reserve_4neighbors(reserved, all_cells)
            placed = True
            break
        if not placed:
            return False
    return True


def _candidate_positions(h: int, w: int, rel: list[tuple[int, int]], require_wrap: bool) -> list[tuple[int, int]]:
    positions = []
    for r in range(h):
        for c in range(w):
            _cells, wraps_row, wraps_col = _wrapped_stamp_cells(rel, r, c, h, w)
            if require_wrap and not (wraps_row or wraps_col):
                continue
            positions.append((r, c))
    random.shuffle(positions)
    return positions


def _place_scattered(rules, inp, occupied, reserved):
    h, w = inp.shape
    for rule in rules:
        positions: list[tuple[int, int]] = []
        wrap_needed = WRAPPED_SCATTERED_PER_RULE
        wrapped_candidates = _candidate_positions(h, w, rule["rel"], require_wrap=True)
        plain_candidates = _candidate_positions(h, w, rule["rel"], require_wrap=False)

        candidate_pools = [wrapped_candidates] * wrap_needed + [plain_candidates] * max(0, rule["n_sc"] - wrap_needed)
        for candidates in candidate_pools:
            placed = False
            limit = min(len(candidates), max(1, rule["n_sc"] * SCATTERED_RETRIES_PER_PIXEL))
            for r, c in candidates[:limit]:
                if (r, c) in occupied or (r, c) in reserved:
                    continue
                positions.append((r, c))
                occupied.add((r, c))
                _reserve_4neighbors(reserved, [(r, c)])
                placed = True
                break
            if not placed:
                return False

        if len(positions) < SCATTERED_MIN:
            return False
        rule["scattered"] = positions
        for r, c in positions:
            inp[r, c] = rule["src"]
    return True


def _all_stamp_cells(rules, h: int, w: int) -> set[tuple[int, int]]:
    cells: set[tuple[int, int]] = set()
    for rule in rules:
        for sr, sc in rule["scattered"]:
            stamp_cells, _wraps_row, _wraps_col = _wrapped_stamp_cells(rule["rel"], int(sr), int(sc), h, w)
            cells.update(stamp_cells)
    return cells


def _place_bystanders(used_colors, inp, occupied, reserved, forbidden_output_cells):
    bystander_colors = [c for c in NON_BG_NON_TEAL_COLORS if c not in used_colors]
    if not bystander_colors:
        return []
    h, w = inp.shape
    n = random.randint(0, min(BYSTANDER_MAX, len(bystander_colors) * 2))
    placed = []
    for _ in range(n):
        for _ in range(BYSTANDER_RETRIES):
            r, c = random.randint(0, h - 1), random.randint(0, w - 1)
            if (r, c) in occupied or (r, c) in reserved or (r, c) in forbidden_output_cells:
                continue
            bc = random.choice(bystander_colors)
            inp[r, c] = bc
            placed.append((r, c, bc))
            occupied.add((r, c))
            _reserve_4neighbors(reserved, [(r, c)])
            break
    return placed


def _build_output(rules, bystander_pos, h: int, w: int):
    output = np.zeros((h, w), dtype=int)
    for r, c, bc in bystander_pos:
        output[int(r), int(c)] = int(bc)
    for rule in rules:
        for sr, sc in rule["scattered"]:
            stamp_cells, _wraps_row, _wraps_col = _wrapped_stamp_cells(rule["rel"], int(sr), int(sc), h, w)
            for nr, nc in stamp_cells:
                output[nr, nc] = int(rule["tgt"])
    return output


def _build_clipped_output(rules, bystander_pos, h: int, w: int):
    output = np.zeros((h, w), dtype=int)
    for r, c, bc in bystander_pos:
        output[int(r), int(c)] = int(bc)
    for rule in rules:
        for sr, sc in rule["scattered"]:
            for rr, cc in rule["rel"]:
                nr = int(sr + rr)
                nc = int(sc + cc)
                if 0 <= nr < h and 0 <= nc < w:
                    output[nr, nc] = int(rule["tgt"])
    return output


def _gen_attempt():
    num_rules = random.choice(RULE_COUNT_WEIGHTS)
    min_dim = MIN_DIM_BY_RULES[num_rules]
    h = random.randint(min_dim, GRID_DIM_MAX)
    w = random.randint(min_dim, GRID_DIM_MAX)

    rules, used_colors = _build_rules(num_rules, h * w)
    occupied: set[tuple[int, int]] = set()
    reserved: set[tuple[int, int]] = set()
    inp = np.zeros((h, w), dtype=int)

    if not _place_legends(rules, inp, occupied, reserved):
        return None
    if not _place_scattered(rules, inp, occupied, reserved):
        return None

    output_cells = _all_stamp_cells(rules, h, w)
    bystander_pos = _place_bystanders(used_colors, inp, occupied, reserved, output_cells)
    output = _build_output(rules, bystander_pos, h, w)
    clipped_output = _build_clipped_output(rules, bystander_pos, h, w)

    wrapped_rules = 0
    for rule in rules:
        if _rule_metrics(rule, h, w)["wrapped_sources"] > 0:
            wrapped_rules += 1
    if wrapped_rules < len(rules):
        return None
    if np.array_equal(output, clipped_output):
        return None

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
