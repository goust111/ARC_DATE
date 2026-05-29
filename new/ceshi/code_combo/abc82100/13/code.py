# Combo variant of abc82100 (index 13)
# combo_spec: D5/3+D12/1
# Name: 模板叠加+重叠冲突变色

from __future__ import annotations

import random
from collections import Counter

import numpy as np

from arc_utils import components_mask, get_bg


# ---------------------------------------------------------------------------
# Design analysis for the fused overlay+conflict variant
#
# a) Concrete behavior and trace
# - Seed task: erase the legend, then stamp each translated teal stencil as a
#   plain overwrite, so the working canvas starts empty.
# - D5/3 contribution: do not clear non-legend colored pixels. Scattered source
#   pixels and bystanders survive as a lower layer, and recolored templates are
#   painted on top of that layer.
# - D12/1 contribution: incompatible overlaps do not use write order. They turn
#   into one fixed conflict color instead.
# - Small trace:
#   * Rule example says blue(1) -> red(2) with an anchor cell inside the teal
#     prototype. A scattered blue source at (6,6) remains on the canvas.
#   * Seed behavior: the translated prototype would simply write red onto its
#     footprint, so the source cell becomes red.
#   * Fused behavior: when the red stamp reaches (6,6), it lands on existing
#     blue, so that cell becomes conflict grey(5). Empty footprint cells still
#     become red.
#   * If a yellow(4) stamp from another rule also reaches one of those red
#     cells, that shared cell becomes grey(5) as well. Same-color overlaps stay
#     unchanged, so the output shows uncovered canvas, singly covered template
#     paint, and multi-layer conflict zones.
#
# b) Complexity calibration
# - Conflict color is fixed to grey(5) and is excluded from input rules, so it
#   is a stable visible marker rather than a hidden numeric convention.
# - Every stencil anchor is one of the teal cells, guaranteeing that each
#   scattered source produces at least one overlay conflict at its own location.
# - The generator also forces at least one cross-rule overlap between different
#   target colors, so the task cannot collapse into "only source cells turn
#   grey" or back into the overwrite behavior of the seed task.
#
# c) Hidden assumptions invalidated in the seed solver
# - Rebuilding output from all-background was only valid when templates replaced
#   the scene. The fused rule needs an existing canvas layer.
# - Removing every source-colored pixel from the preserved canvas was only valid
#   when source pixels were consumed before stamping. Here they must remain so
#   the overlay can visibly collide with them.
# - Direct assignment `out[...] = target` relied on last-write-wins. That is
#   invalid because overlap outcome is now semantic, not accidental ordering.
# - New solver assumption: conflict cells are absorbing. Once a cell is grey,
#   later stamps leave it grey. The generator respects this by treating grey as
#   a dedicated output-only marker, never as an input rule color.
#
# d) Placement legality guaranteed by the generator
# - Legend triplets and teal prototypes keep a 4-neighbor buffer and also avoid
#   the reverse-direction anchor zone, so rule discovery stays one-way and
#   unambiguous.
# - Scattered sources and bystanders are never 4-adjacent to existing colored
#   cells, preventing accidental extra source-target pairs.
# - Forced cross-rule conflict is created by choosing one relative stencil cell
#   from each of two rules and solving for two distinct in-bounds source
#   positions that make those cells coincide.
#
# e) Visual observability
# - Across examples, the local colored pair still explicitly teaches which
#   source color emits which recolored stencil.
# - Grey only appears in outputs, and it appears exactly where overlay creates a
#   disagreement. Because source cells themselves become grey and at least one
#   extra cross-rule conflict is forced, a human can infer both "templates are
#   overlaid onto existing content" and "different overlaps become grey".
#
# f) Visual cleanliness
# - Input legend areas stay separated; only output footprints interact.
# - The generator forces conflict but keeps bystanders sparse and overlap count
#   modest, so outputs show a few crisp grey cells/patches instead of a dense
#   noisy carpet.
#
# g) Change inferability
# - Color substitution still comes from repeated local examples, not from hidden
#   color ordering.
# - Grey does not depend on which specific colors collide; it depends only on
#   the visible structural event "different non-background layers meet here".
# ---------------------------------------------------------------------------


BG_COLOR = 0
PATTERN_COLOR = 8
CONFLICT_COLOR = 5

# Grid dimensions
GRID_DIM_MAX = 20
MIN_DIM_BY_RULES = {2: 10, 3: 14}

# Rule sampling
RULE_COUNT_WEIGHTS = [2, 2, 3, 3]
NON_BG_NON_SPECIAL_COLORS = [1, 2, 3, 4, 6, 7, 9]

# Stencil size
STENCIL_MIN_PIXELS = 3
STENCIL_MAX_CAP = 12
STENCIL_MIN_AREA_DIV = 90
STENCIL_MAX_AREA_DIV = 22

# Scattered source pixels per rule
SCATTERED_MIN = 3
SCATTERED_MAX_CAP = 7
SCATTERED_AREA_DIV = 32

# Bystanders
BYSTANDER_MAX = 4

# Retry budgets
LEGEND_RETRIES = 500
CONFLICT_PAIR_RETRIES = 900
SCATTERED_RETRIES_PER_PIXEL = 80
BYSTANDER_RETRIES = 120
OUTER_RETRIES = 500

_DIRS4 = [(-1, 0), (1, 0), (0, -1), (0, 1)]
_DIRS8 = [(-1, -1), (-1, 0), (-1, 1), (0, -1), (0, 1), (1, -1), (1, 0), (1, 1)]


def _chebyshev(a: tuple[int, int], b: tuple[int, int]) -> int:
    return max(abs(int(a[0]) - int(b[0])), abs(int(a[1]) - int(b[1])))


def _anchor_distance(pattern: np.ndarray, anchor: tuple[int, int]) -> int:
    return min(_chebyshev((int(r), int(c)), anchor) for r, c in pattern)


def _discover_rules(grid: np.ndarray, bg: int, pattern_color: int = PATTERN_COLOR) -> list[dict]:
    h, w = grid.shape
    patterns = components_mask(grid == pattern_color, connectivity=8)
    rules: list[dict] = []
    seen: set[tuple[tuple[int, int], tuple[int, int], int]] = set()

    for r in range(h):
        for c in range(w):
            source_color = int(grid[r, c])
            if source_color in (int(bg), int(pattern_color)):
                continue

            for dr, dc in _DIRS4:
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


def _merge_color(current: int, incoming: int, bg: int, conflict_color: int = CONFLICT_COLOR) -> int:
    current = int(current)
    incoming = int(incoming)
    if current == int(bg):
        return int(incoming)
    if current == int(incoming) or current == int(conflict_color):
        return int(current)
    return int(conflict_color)


def main(grid: np.ndarray) -> np.ndarray:
    """
    Learn local source-target-template examples, keep the ordinary scene as a
    base layer, then overlay recolored stencil copies and mark clashes in grey.

    Input structure:
    - Black background with isolated colored pixels plus teal (8) prototype
      components attached to local rule examples.
    - Each rule example is a 4-neighbor source->target pair whose direction
      continues to an anchor touching the matching teal prototype.

    Rule:
    - Remove the legend pair cells and all teal prototype cells from the output,
      but preserve every other non-teal colored pixel as the starting canvas.
    - For each non-legend source pixel of a learned source color, translate the
      matched teal prototype so its anchor lands on that source pixel and
      recolor the translated footprint to the rule's target color.
    - Overlay the footprint cell by cell: background accepts the new color,
      same-color overlaps stay unchanged, and any different non-background
      overlap becomes the fixed conflict color grey (5).

    Key insight:
    - The local colored pair still identifies which stencil to copy and what
      color it should emit, but the copied stencil behaves like paint over an
      existing canvas, and disagreement between layers is shown explicitly.
    """
    g = np.asarray(grid, dtype=int)
    bg = get_bg(g)
    h, w = g.shape

    rules = _discover_rules(g, bg=bg, pattern_color=PATTERN_COLOR)
    legend_pixels: set[tuple[int, int]] = set()
    for rule in rules:
        legend_pixels.add(rule["source_pos"])
        legend_pixels.add(rule["target_pos"])

    out = np.full_like(g, int(bg))

    for r, c in map(tuple, np.argwhere((g != int(bg)) & (g != PATTERN_COLOR))):
        r = int(r)
        c = int(c)
        if (r, c) in legend_pixels:
            continue
        out[r, c] = int(g[r, c])

    for rule in rules:
        ar, ac = rule["anchor"]
        pattern = rule["pattern"]
        target_color = int(rule["target_color"])
        for sr, sc in map(tuple, np.argwhere(g == rule["source_color"])):
            sr = int(sr)
            sc = int(sc)
            if (sr, sc) in legend_pixels:
                continue
            dr = sr - ar
            dc = sc - ac
            rows = pattern[:, 0] + dr
            cols = pattern[:, 1] + dc
            ok = (0 <= rows) & (rows < h) & (0 <= cols) & (cols < w)
            for nr, nc in zip(rows[ok], cols[ok]):
                nr = int(nr)
                nc = int(nc)
                out[nr, nc] = _merge_color(out[nr, nc], target_color, bg=bg)

    return out


def _reserve_4neighbors(reserved: set[tuple[int, int]], cells) -> None:
    for r, c in cells:
        for dr, dc in _DIRS4:
            reserved.add((int(r) + dr, int(c) + dc))


def _reserve_chebyshev1(reserved: set[tuple[int, int]], cells) -> None:
    for r, c in cells:
        for dr in (-1, 0, 1):
            for dc in (-1, 0, 1):
                reserved.add((int(r) + dr, int(c) + dc))


def _has_adjacent_4(cell: tuple[int, int], occupied: set[tuple[int, int]]) -> bool:
    r, c = int(cell[0]), int(cell[1])
    return any((r + dr, c + dc) in occupied for dr, dc in _DIRS4)


def _can_place_colored(
    cell: tuple[int, int],
    occupied: set[tuple[int, int]],
    reserved: set[tuple[int, int]],
) -> bool:
    r, c = int(cell[0]), int(cell[1])
    if (r, c) in occupied or (r, c) in reserved:
        return False
    if _has_adjacent_4((r, c), occupied):
        return False
    return True


def _make_stencil(min_pixels: int, max_pixels: int) -> tuple[list[tuple[int, int]], tuple[int, int]]:
    n = random.randint(min_pixels, max_pixels)
    coords = {(0, 0)}
    while len(coords) < n:
        frontier = set()
        for r, c in coords:
            for dr, dc in _DIRS8:
                frontier.add((r + dr, c + dc))
        frontier -= coords
        coords.add(random.choice(list(frontier)))
    coords_list = sorted(coords)
    anchor = random.choice(coords_list)
    return coords_list, anchor


def _build_rules(num_rules: int, area: int) -> tuple[list[dict], set[int]]:
    colors = NON_BG_NON_SPECIAL_COLORS[:]
    random.shuffle(colors)

    max_s = min(STENCIL_MAX_CAP, max(STENCIL_MIN_PIXELS, area // STENCIL_MAX_AREA_DIV))
    min_s = min(max(STENCIL_MIN_PIXELS, area // STENCIL_MIN_AREA_DIV), max_s)
    sc_max = min(SCATTERED_MAX_CAP, max(SCATTERED_MIN, area // SCATTERED_AREA_DIV))

    rules: list[dict] = []
    used: set[int] = set()
    for i in range(num_rules):
        src_c, tgt_c = colors[2 * i], colors[2 * i + 1]
        used.update((int(src_c), int(tgt_c)))
        stencil, anchor = _make_stencil(min_s, max_s)
        rel = [(int(r - anchor[0]), int(c - anchor[1])) for r, c in stencil]
        rules.append(
            {
                "src": int(src_c),
                "tgt": int(tgt_c),
                "rel": rel,
                "n_sc": random.randint(SCATTERED_MIN, sc_max),
                "scattered": [],
            }
        )
    return rules, used


def _stamp_cells(rel: list[tuple[int, int]], sr: int, sc: int, h: int, w: int) -> list[tuple[int, int]]:
    cells: list[tuple[int, int]] = []
    for rr, cc in rel:
        nr = int(sr + rr)
        nc = int(sc + cc)
        if 0 <= nr < h and 0 <= nc < w:
            cells.append((nr, nc))
    return cells


def _place_legends(
    rules: list[dict],
    inp: np.ndarray,
    occupied: set[tuple[int, int]],
    reserved: set[tuple[int, int]],
) -> bool:
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

            teal_cells: set[tuple[int, int]] = set()
            ok = True
            for rr, cc in rel:
                nr, nc = ar + rr, ac + cc
                if not (0 <= nr < h and 0 <= nc < w):
                    ok = False
                    break
                teal_cells.add((int(nr), int(nc)))
            if not ok:
                continue

            reverse_anchor = (int(sr - dr), int(sc - dc))
            teal_arr = np.asarray(sorted(teal_cells), dtype=int)
            if _anchor_distance(teal_arr, reverse_anchor) <= 1:
                continue
            reverse_zone = {
                (reverse_anchor[0] + rr, reverse_anchor[1] + cc)
                for rr in (-1, 0, 1)
                for cc in (-1, 0, 1)
            }
            all_cells = {(int(sr), int(sc)), (int(tr), int(tc)), (int(ar), int(ac))} | teal_cells
            if all_cells & (occupied | reserved):
                continue
            if reverse_zone & (occupied | reserved):
                continue

            inp[sr, sc] = int(rule["src"])
            inp[tr, tc] = int(rule["tgt"])
            for nr, nc in teal_cells:
                inp[nr, nc] = PATTERN_COLOR
            occupied.update(all_cells)
            _reserve_4neighbors(reserved, all_cells)
            _reserve_chebyshev1(reserved, [reverse_anchor])
            placed = True
            break
        if not placed:
            return False
    return True


def _add_scattered(
    rule: dict,
    pos: tuple[int, int],
    inp: np.ndarray,
    occupied: set[tuple[int, int]],
    reserved: set[tuple[int, int]],
) -> bool:
    r, c = int(pos[0]), int(pos[1])
    if not _can_place_colored((r, c), occupied, reserved):
        return False
    rule["scattered"].append((r, c))
    inp[r, c] = int(rule["src"])
    occupied.add((r, c))
    _reserve_4neighbors(reserved, [(r, c)])
    return True


def _place_conflict_pair(
    rules: list[dict],
    inp: np.ndarray,
    occupied: set[tuple[int, int]],
    reserved: set[tuple[int, int]],
) -> bool:
    h, w = inp.shape
    ia, ib = random.sample(range(len(rules)), 2)
    rule_a = rules[int(ia)]
    rule_b = rules[int(ib)]

    rel_a = rule_a["rel"][:]
    rel_b = rule_b["rel"][:]
    random.shuffle(rel_a)
    random.shuffle(rel_b)

    for _ in range(CONFLICT_PAIR_RETRIES):
        rr_a, cc_a = random.choice(rel_a)
        rr_b, cc_b = random.choice(rel_b)
        overlap_r = random.randint(0, h - 1)
        overlap_c = random.randint(0, w - 1)
        src_a = (int(overlap_r - rr_a), int(overlap_c - cc_a))
        src_b = (int(overlap_r - rr_b), int(overlap_c - cc_b))

        if not (0 <= src_a[0] < h and 0 <= src_a[1] < w):
            continue
        if not (0 <= src_b[0] < h and 0 <= src_b[1] < w):
            continue
        if src_a == src_b:
            continue
        if abs(src_a[0] - src_b[0]) + abs(src_a[1] - src_b[1]) == 1:
            continue
        if not _can_place_colored(src_a, occupied, reserved):
            continue
        temp_occupied = set(occupied)
        temp_occupied.add(src_a)
        if not _can_place_colored(src_b, temp_occupied, reserved):
            continue
        if not _add_scattered(rule_a, src_a, inp, occupied, reserved):
            continue
        if not _add_scattered(rule_b, src_b, inp, occupied, reserved):
            raise RuntimeError("prevalidated conflict placement failed unexpectedly")
        return True

    return False


def _place_scattered(
    rules: list[dict],
    inp: np.ndarray,
    occupied: set[tuple[int, int]],
    reserved: set[tuple[int, int]],
) -> bool:
    for rule in rules:
        rule["scattered"] = []

    if len(rules) < 2 or not _place_conflict_pair(rules, inp, occupied, reserved):
        return False

    h, w = inp.shape
    for rule in rules:
        for _ in range(rule["n_sc"] * SCATTERED_RETRIES_PER_PIXEL):
            if len(rule["scattered"]) >= rule["n_sc"]:
                break
            r = random.randint(0, h - 1)
            c = random.randint(0, w - 1)
            _add_scattered(rule, (r, c), inp, occupied, reserved)
        if len(rule["scattered"]) < SCATTERED_MIN:
            return False

    return True


def _collect_stamp_footprint(rules: list[dict], h: int, w: int) -> Counter:
    counts: Counter = Counter()
    for rule in rules:
        for sr, sc in rule["scattered"]:
            for cell in _stamp_cells(rule["rel"], int(sr), int(sc), h, w):
                counts[cell] += 1
    return counts


def _place_bystanders(
    rules: list[dict],
    used_colors: set[int],
    inp: np.ndarray,
    occupied: set[tuple[int, int]],
    reserved: set[tuple[int, int]],
) -> list[tuple[int, int, int]]:
    bystander_colors = [c for c in NON_BG_NON_SPECIAL_COLORS if c not in used_colors]
    if not bystander_colors:
        return []

    h, w = inp.shape
    footprint = _collect_stamp_footprint(rules, h, w)
    candidates = [
        cell
        for cell, _count in footprint.most_common()
        if cell not in occupied and cell not in reserved and not _has_adjacent_4(cell, occupied)
    ]

    placed: list[tuple[int, int, int]] = []
    if candidates:
        r, c = candidates[0]
        bc = random.choice(bystander_colors)
        inp[r, c] = int(bc)
        placed.append((int(r), int(c), int(bc)))
        occupied.add((int(r), int(c)))
        _reserve_4neighbors(reserved, [(int(r), int(c))])

    n_extra = random.randint(0, max(0, BYSTANDER_MAX - len(placed)))
    for _ in range(n_extra):
        for _ in range(BYSTANDER_RETRIES):
            r = random.randint(0, h - 1)
            c = random.randint(0, w - 1)
            if (r, c) in occupied or (r, c) in reserved or _has_adjacent_4((r, c), occupied):
                continue
            bc = random.choice(bystander_colors)
            inp[r, c] = int(bc)
            placed.append((int(r), int(c), int(bc)))
            occupied.add((int(r), int(c)))
            _reserve_4neighbors(reserved, [(int(r), int(c))])
            break

    return placed


def _build_fused_output(
    rules: list[dict],
    bystander_pos: list[tuple[int, int, int]],
    h: int,
    w: int,
    bg: int = BG_COLOR,
) -> tuple[np.ndarray, dict]:
    output = np.full((h, w), int(bg), dtype=int)

    source_cells: list[tuple[int, int]] = []
    bystander_map = {(int(r), int(c)): int(color) for r, c, color in bystander_pos}
    source_owner: dict[tuple[int, int], int] = {}
    for rule_index, rule in enumerate(rules):
        for sr, sc in rule["scattered"]:
            sr = int(sr)
            sc = int(sc)
            source_cells.append((sr, sc))
            source_owner[(sr, sc)] = rule_index
            output[sr, sc] = int(rule["src"])
    for (r, c), color in bystander_map.items():
        output[r, c] = int(color)

    target_sets: dict[tuple[int, int], set[int]] = {}
    source_hit_sets: dict[tuple[int, int], set[int]] = {}
    bystander_hit_sets: dict[tuple[int, int], set[int]] = {}
    coverage: Counter = Counter()

    for rule_index, rule in enumerate(rules):
        target_color = int(rule["tgt"])
        for sr, sc in rule["scattered"]:
            cells = _stamp_cells(rule["rel"], int(sr), int(sc), h, w)
            for cell in cells:
                coverage[cell] += 1
                target_sets.setdefault(cell, set()).add(target_color)
                if cell in source_owner:
                    source_hit_sets.setdefault(cell, set()).add(rule_index)
                if cell in bystander_map:
                    bystander_hit_sets.setdefault(cell, set()).add(rule_index)
                r, c = cell
                output[r, c] = _merge_color(output[r, c], target_color, bg=bg)

    conflict_cells = {(int(r), int(c)) for r, c in np.argwhere(output == CONFLICT_COLOR)}
    cross_rule_conflict_cells = {
        cell for cell, colors in target_sets.items() if len(colors) >= 2 and cell in conflict_cells
    }
    self_conflict_cells = {
        cell for cell, hitters in source_hit_sets.items() if len(hitters) >= 1 and cell in conflict_cells
    }
    bystander_conflict_cells = {
        cell for cell, hitters in bystander_hit_sets.items() if len(hitters) >= 1 and cell in conflict_cells
    }
    metrics = {
        "source_count": len(source_cells),
        "conflict_cells": len(conflict_cells),
        "cross_rule_conflict_cells": len(cross_rule_conflict_cells),
        "self_conflict_cells": len(self_conflict_cells),
        "bystander_conflict_cells": len(bystander_conflict_cells),
        "overlap_cells": sum(count >= 2 for count in coverage.values()),
    }
    return output, metrics


def _build_seed_output(
    rules: list[dict],
    bystander_pos: list[tuple[int, int, int]],
    h: int,
    w: int,
) -> np.ndarray:
    output = np.zeros((h, w), dtype=int)
    for r, c, color in bystander_pos:
        output[int(r), int(c)] = int(color)
    for rule in rules:
        for sr, sc in rule["scattered"]:
            for nr, nc in _stamp_cells(rule["rel"], int(sr), int(sc), h, w):
                output[nr, nc] = int(rule["tgt"])
    return output


def _build_d12_only_output(
    rules: list[dict],
    bystander_pos: list[tuple[int, int, int]],
    h: int,
    w: int,
) -> np.ndarray:
    output = np.zeros((h, w), dtype=int)
    for r, c, color in bystander_pos:
        output[int(r), int(c)] = int(color)

    stamp_color = np.zeros((h, w), dtype=int)
    conflict_mask = np.zeros((h, w), dtype=bool)
    for rule in rules:
        target_color = int(rule["tgt"])
        for sr, sc in rule["scattered"]:
            for nr, nc in _stamp_cells(rule["rel"], int(sr), int(sc), h, w):
                current = int(stamp_color[nr, nc])
                if current == BG_COLOR:
                    stamp_color[nr, nc] = target_color
                elif current != target_color:
                    conflict_mask[nr, nc] = True

    stamp_mask = stamp_color != BG_COLOR
    output[stamp_mask] = stamp_color[stamp_mask]
    output[conflict_mask] = CONFLICT_COLOR
    return output


def _gen_attempt() -> np.ndarray | None:
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
    bystander_pos = _place_bystanders(rules, used_colors, inp, occupied, reserved)

    output, metrics = _build_fused_output(rules, bystander_pos, h, w)
    seed_output = _build_seed_output(rules, bystander_pos, h, w)
    d12_only_output = _build_d12_only_output(rules, bystander_pos, h, w)

    if metrics["cross_rule_conflict_cells"] < 1:
        return None
    if metrics["self_conflict_cells"] < metrics["source_count"]:
        return None
    if np.array_equal(output, seed_output):
        return None
    if np.array_equal(output, d12_only_output):
        return None

    try:
        if np.array_equal(main(inp), output):
            return inp
    except Exception:
        return None
    return None


def generate_input() -> np.ndarray:
    for _ in range(OUTER_RETRIES):
        result = _gen_attempt()
        if result is not None:
            return result
    raise RuntimeError(f"Failed to generate valid input after {OUTER_RETRIES} attempts")


def sample_metrics(samples: int = 50) -> dict[str, float]:
    """Lightweight helper for post-hoc validation of generated behavior."""
    cross_rule = 0
    self_conflict = 0
    bystander_conflict = 0
    total_conflict = 0
    for _ in range(samples):
        grid = generate_input()
        rules = _discover_rules(grid, bg=BG_COLOR, pattern_color=PATTERN_COLOR)
        h, w = grid.shape
        bystander_pos = [
            (int(r), int(c), int(grid[r, c]))
            for r, c in map(tuple, np.argwhere((grid != BG_COLOR) & (grid != PATTERN_COLOR)))
            if (int(r), int(c)) not in {rule["source_pos"] for rule in rules}
            and (int(r), int(c)) not in {rule["target_pos"] for rule in rules}
            and all(int(grid[r, c]) != int(rule["source_color"]) for rule in rules)
        ]
        rebuilt_rules: list[dict] = []
        for rule in rules:
            scattered = [
                (int(r), int(c))
                for r, c in map(tuple, np.argwhere(grid == int(rule["source_color"])))
                if (int(r), int(c)) != rule["source_pos"]
            ]
            rebuilt_rules.append(
                {
                    "src": int(rule["source_color"]),
                    "tgt": int(rule["target_color"]),
                    "rel": [
                        (int(r - rule["anchor"][0]), int(c - rule["anchor"][1]))
                        for r, c in rule["pattern"]
                    ],
                    "scattered": scattered,
                }
            )
        _out, metrics = _build_fused_output(rebuilt_rules, bystander_pos, h, w)
        cross_rule += int(metrics["cross_rule_conflict_cells"] > 0)
        self_conflict += int(metrics["self_conflict_cells"] > 0)
        bystander_conflict += int(metrics["bystander_conflict_cells"] > 0)
        total_conflict += int(metrics["conflict_cells"] > 0)
    return {
        "cross_rule_rate": cross_rule / samples,
        "self_conflict_rate": self_conflict / samples,
        "bystander_conflict_rate": bystander_conflict / samples,
        "any_conflict_rate": total_conflict / samples,
    }
