# Combo variant of abc82100 (index 29)
# combo_spec: D5/3+D7/0+D12/0
# Name: 模板叠加+保留规则+OR合并

from __future__ import annotations

import random
from collections import Counter

import numpy as np

from arc_utils import components_mask, get_bg


# ---------------------------------------------------------------------------
# Design analysis for the fused variant
#
# a) Concrete behavior and trace
# - The seed task learned local source->target->teal legends, erased both the
#   legends and the teal prototypes, and then redrew only recolored stencil
#   copies on an empty canvas.
# - This fusion changes three stages at once:
#   1. only teal prototype cells are removed,
#   2. every other colored pixel stays as a lower layer,
#   3. translated stencil copies are overlaid with an order-independent merge.
# - Because ARC colors are human-visible categories rather than visible numbers,
#   "OR merge" is implemented as union of painted support, not hidden bitwise OR:
#   * background + target -> target
#   * same color + target -> unchanged
#   * different non-background colors meeting in one cell -> teal(8)
#   This is observable and still does not depend on stamping order.
# - Small trace:
#   suppose a legend says blue source -> yellow target with a teal stencil whose
#   anchor lies inside the stencil. A scattered blue source pixel remains on the
#   output canvas. When the yellow stencil is translated onto it, the anchor cell
#   now lands on an occupied blue cell, so that cell becomes teal instead of
#   yellow. If another rule's red stencil also reaches a nearby yellow cell, that
#   overlap also becomes teal. The legend pair itself is not blanked first; it is
#   simply another preserved layer.
#
# b) Complexity calibration
# - The generator keeps 2-4 rules so multi-template interaction is actually
#   visible; one-rule boards would make D12 inert.
# - Every stencil contains its anchor, so each scattered source creates one
#   guaranteed self-overlay at its own location.
# - Samples are rejected unless they contain at least one overlap cell hit by
#   two different target colors and unless the fused output differs from a naive
#   overwrite baseline. That prevents regression to the seed task or to a weak
#   "overlay in name only" dataset.
# - The legend region is buffered away from scattered sources so rule discovery
#   stays unambiguous and the output does not become unreadably crowded.
#
# c) Implicit-assumption audit against the original solver
# - `out = full(bg)` is no longer valid: the output now starts from the input
#   with only teal removed.
# - Skipping every learned source-color pixel is no longer valid: scattered
#   sources survive as lower-layer content and must be present before stamping.
# - Direct assignment `out[...] = target` is no longer valid: overlap handling is
#   the point of the variant and must be order-independent.
# - Restoring legend pixels after stamping is also wrong for the fused rule:
#   legends are preserved because they are never cleared, not because they are
#   immune to later overlay. They participate in the same visible merge if a
#   translated stencil reaches them.
#
# d) Placement legality
# - Rule legends and their teal prototype components are placed with a 4-neighbor
#   exclusion buffer, so no accidental extra legend can be formed nearby.
# - Scattered source pixels are placed off that buffer and are kept non-4-adjacent
#   to other occupied cells, preventing spurious source-target pairs.
# - Stencils are generated as 8-connected components whose anchor is one of their
#   cells, matching the solver's overlay assumption.
# - Stamps may clip at the border, but the generator never writes the input
#   outside bounds and validates every produced sample with `main(input)`.
#
# e) Visual observability
# - Across a few I/O examples, a human can see three stable cues:
#   1. the local colored legends are still present in output instead of vanishing,
#   2. scattered sources also remain instead of being replaced wholesale,
#   3. cells where layers meet turn teal, revealing order-independent overlay.
# - The new behavior is expressed by visible geometry and visible colors, not by
#   hidden numeric color ids or processing order.
#
# f) Visual cleanliness
# - The generator keeps the scattered sources in one compact working cluster so
#   overlaps are visible without spraying the whole board with collisions.
# - Legends stay in separate buffered zones, and bystanders are sparse. This
#   preserves the original "small local examples + remote applications" look.
#
# g) Inferability of color changes
# - The source->target recoloring rule is still taught by repeated local legends,
#   so the new task does not introduce an unhinted color lookup.
# - Teal has one consistent observable meaning in the output: a cell belongs to
#   more than one incompatible visible layer. The generator enforces repeated
#   examples of that effect before accepting a sample.
#
# Post-change solver assumptions audited against the generator
# - Solver assumption: the discovered teal component includes the anchor cell.
#   Generator guarantee: `_make_stencil()` always chooses the anchor from inside
#   the component.
# - Solver assumption: every non-legend source-color pixel is a real stamp origin.
#   Generator guarantee: scattered sources are isolated from legend buffers and
#   validated against rediscovery.
# - Solver assumption: order-independent overlay, not overwrite, defines the
#   final cell value at overlaps.
#   Generator guarantee: samples are rejected when fused output equals a naive
#   overwrite baseline or when no multi-target overlap cell exists.
# - Solver assumption: legends are retained because they are not erased, but may
#   still be touched by overlay if geometry happens to reach them.
#   Generator guarantee: legends are usually buffered away, keeping them legible,
#   while the solver still handles the general case.
# ---------------------------------------------------------------------------


PATTERN_COLOR = 8
COLLISION_COLOR = 8


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


def _merge_color(current: int, incoming: int, bg: int, collision_color: int = COLLISION_COLOR) -> int:
    current = int(current)
    incoming = int(incoming)
    bg = int(bg)
    if current == bg:
        return incoming
    if current == incoming or current == int(collision_color):
        return current
    return int(collision_color)


def _legend_pixels(rules: list[dict]) -> set[tuple[int, int]]:
    pixels: set[tuple[int, int]] = set()
    for rule in rules:
        pixels.add(tuple(rule["source_pos"]))
        pixels.add(tuple(rule["target_pos"]))
    return pixels


def _stamp_cells(pattern: np.ndarray, anchor: tuple[int, int], sr: int, sc: int, h: int, w: int) -> list[tuple[int, int]]:
    ar, ac = anchor
    dr = int(sr) - int(ar)
    dc = int(sc) - int(ac)
    rows = pattern[:, 0] + dr
    cols = pattern[:, 1] + dc
    ok = (0 <= rows) & (rows < h) & (0 <= cols) & (cols < w)
    return [(int(r), int(c)) for r, c in zip(rows[ok], cols[ok])]


def _simulate(
    grid: np.ndarray,
    rules: list[dict],
    bg: int,
    *,
    overwrite: bool,
) -> tuple[np.ndarray, dict]:
    g = np.asarray(grid, dtype=int)
    h, w = g.shape
    legend_pixels = _legend_pixels(rules)

    out = np.array(g, dtype=int, copy=True)
    out[out == PATTERN_COLOR] = int(bg)

    target_hits: dict[tuple[int, int], set[int]] = {}
    hit_counts: Counter[tuple[int, int]] = Counter()
    source_cells: set[tuple[int, int]] = set()

    for rule in rules:
        target_color = int(rule["target_color"])
        pattern = np.asarray(rule["pattern"], dtype=int)
        anchor = tuple(rule["anchor"])
        for sr, sc in map(tuple, np.argwhere(g == int(rule["source_color"]))):
            sr = int(sr)
            sc = int(sc)
            if (sr, sc) in legend_pixels:
                continue
            source_cells.add((sr, sc))
            for nr, nc in _stamp_cells(pattern, anchor, sr, sc, h, w):
                cell = (int(nr), int(nc))
                hit_counts[cell] += 1
                target_hits.setdefault(cell, set()).add(target_color)
                if overwrite:
                    out[nr, nc] = target_color
                else:
                    out[nr, nc] = _merge_color(out[nr, nc], target_color, bg=bg)

    collision_cells = {(int(r), int(c)) for r, c in np.argwhere(out == COLLISION_COLOR)}
    multi_target_overlap_cells = {
        cell for cell, colors in target_hits.items() if len(colors) >= 2
    }
    metrics = {
        "source_count": len(source_cells),
        "source_collision_cells": sum(cell in collision_cells for cell in source_cells),
        "collision_cells": len(collision_cells),
        "extra_collision_cells": len(collision_cells) - len(source_cells),
        "template_overlap_cells": sum(count >= 2 for count in hit_counts.values()),
        "multi_target_overlap_cells": len(multi_target_overlap_cells),
        "multi_target_collision_cells": sum(cell in collision_cells for cell in multi_target_overlap_cells),
        "legend_non_bg_cells": sum(int(out[r, c]) != int(bg) for r, c in legend_pixels),
        "legend_changed_cells": sum(int(out[r, c]) != int(g[r, c]) for r, c in legend_pixels),
    }
    return out, metrics


def behavior_metrics(grid: np.ndarray) -> dict:
    """
    Quantify whether a sample exhibits the fused behavior rather than regressing
    to the seed overwrite task.
    """
    g = np.asarray(grid, dtype=int)
    bg = get_bg(g)
    rules = _discover_rules(g, bg=bg, pattern_color=PATTERN_COLOR)
    fused, metrics = _simulate(g, rules, bg, overwrite=False)
    overwrite_out, _overwrite_metrics = _simulate(g, rules, bg, overwrite=True)
    metrics = dict(metrics)
    metrics["overwrite_gap_cells"] = int(np.count_nonzero(fused != overwrite_out))
    return metrics


def main(grid: np.ndarray) -> np.ndarray:
    """
    Learn local source-target-stencil legends, keep the non-teal scene, and overlay recolored stencil copies onto it.

    Input structure:
    - A black-background grid contains several local legends made of a colored source pixel, an adjacent colored target
      pixel, and a nearby teal (8) stencil component.
    - Elsewhere, isolated pixels of the legend source colors act as the real trigger points for translated stencil copies.
    - The legend pair itself is ordinary content now; only the teal prototype cells are disposable.

    Rule:
    - Discover each rule from a 4-neighbor source->target pair whose continued direction reaches an anchor touching the
      matching teal component, then translate that component so the anchor lands on every non-legend source pixel of the
      same color.
    - Start the output from the full input with only teal removed, so legends, scattered source pixels, and bystanders all
      remain as lower layers.
    - Recolor each translated stencil to the rule target color and overlay it cell by cell: background accepts the new
      color, equal colors stay as they are, and any different non-background overlap becomes teal.

    Key insight:
    - The local legend still teaches which source color emits which stencil, but the scene is no longer rebuilt from
      scratch. The final picture is the visible superposition of the preserved input content and every translated stencil,
      with teal marking mixed overlaps in an order-independent way.
    """
    g = np.asarray(grid, dtype=int)
    bg = get_bg(g)
    rules = _discover_rules(g, bg=bg, pattern_color=PATTERN_COLOR)
    out, _metrics = _simulate(g, rules, bg, overwrite=False)
    return out


# --------------- generator ---------------

# Grid dimensions
GRID_DIM_MAX = 20
MIN_DIM_BY_RULES = {2: 10, 3: 14, 4: 18}

# Rule sampling
RULE_COUNT_WEIGHTS = [2, 2, 3, 3, 4]
SOURCE_COLOR_POOL = [1, 2, 3, 4, 5, 6]
VALID_TARGETS = {
    1: [2, 4, 6],
    2: [1, 4, 5],
    3: [4, 5, 6],
    4: [1, 2, 3],
    5: [2, 3, 6],
    6: [1, 3, 5],
}
BYSTANDER_COLOR_POOL = [1, 2, 3, 4, 5, 6, 7]

# Stencil size and scattered counts
STENCIL_MIN_PIXELS = 3
STENCIL_MAX_CAP = 12
STENCIL_MIN_AREA_DIV = 90
STENCIL_MAX_AREA_DIV = 22
SCATTERED_MIN = 3
SCATTERED_MAX_CAP = 6
SCATTERED_AREA_DIV = 34
BYSTANDER_MAX = 3

# Placement budgets
LEGEND_RETRIES = 600
SCATTER_CLUSTER_RETRIES = 160
BYSTANDER_RETRIES = 160
OUTER_RETRIES = 500

_DIRS4 = [(-1, 0), (1, 0), (0, -1), (0, 1)]
_DIRS8 = [(-1, -1), (-1, 0), (-1, 1), (0, -1), (0, 1), (1, -1), (1, 0), (1, 1)]


def _reserve_4neighbors(reserved: set[tuple[int, int]], cells) -> None:
    for r, c in cells:
        for dr, dc in _DIRS4:
            reserved.add((int(r) + dr, int(c) + dc))


def _has_adjacent_4(cell: tuple[int, int], occupied: set[tuple[int, int]]) -> bool:
    r, c = cell
    return any((int(r) + dr, int(c) + dc) in occupied for dr, dc in _DIRS4)


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


def _choose_rule_colors(num_rules: int) -> list[tuple[int, int]] | None:
    for _ in range(200):
        sources = random.sample(SOURCE_COLOR_POOL, num_rules)
        targets: list[int] = []
        used_targets: set[int] = set()
        ok = True
        for src in sources:
            candidates = VALID_TARGETS[src][:]
            random.shuffle(candidates)
            pick = next((t for t in candidates if t not in used_targets), None)
            if pick is None:
                pick = candidates[0] if candidates else None
            if pick is None:
                ok = False
                break
            used_targets.add(int(pick))
            targets.append(int(pick))
        if not ok:
            continue
        if len(set(targets)) < 2:
            continue
        return list(zip(sources, targets))
    return None


def _build_rules(num_rules: int, area: int) -> tuple[list[dict], set[int]] | tuple[None, None]:
    pairs = _choose_rule_colors(num_rules)
    if pairs is None:
        return None, None

    max_s = min(STENCIL_MAX_CAP, max(STENCIL_MIN_PIXELS, area // STENCIL_MAX_AREA_DIV))
    min_s = min(max(STENCIL_MIN_PIXELS, area // STENCIL_MIN_AREA_DIV), max_s)
    sc_max = min(SCATTERED_MAX_CAP, max(SCATTERED_MIN, area // SCATTERED_AREA_DIV))

    rules: list[dict] = []
    used_colors: set[int] = set()
    for src_c, tgt_c in pairs:
        used_colors.add(int(src_c))
        stencil, anchor = _make_stencil(min_s, max_s)
        rel = [(int(r - anchor[0]), int(c - anchor[1])) for r, c in stencil]
        rules.append(
            {
                "src": int(src_c),
                "tgt": int(tgt_c),
                "rel": rel,
                "n_sc": random.randint(SCATTERED_MIN, sc_max),
            }
        )
    return rules, used_colors


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
            teal_cells = set()
            ok = True
            for rr, cc in rel:
                nr, nc = ar + rr, ac + cc
                if not (0 <= nr < h and 0 <= nc < w):
                    ok = False
                    break
                teal_cells.add((int(nr), int(nc)))
            if not ok:
                continue
            all_cells = {(int(sr), int(sc)), (int(tr), int(tc)), (int(ar), int(ac))} | teal_cells
            if all_cells & occupied:
                continue
            if all_cells & reserved:
                continue
            inp[sr, sc] = int(rule["src"])
            inp[tr, tc] = int(rule["tgt"])
            for nr, nc in teal_cells:
                inp[nr, nc] = PATTERN_COLOR
            occupied.update(all_cells)
            _reserve_4neighbors(reserved, all_cells)
            rule["source_pos"] = (int(sr), int(sc))
            rule["target_pos"] = (int(tr), int(tc))
            placed = True
            break
        if not placed:
            return False
    return True


def _place_scattered(
    rules: list[dict],
    inp: np.ndarray,
    occupied: set[tuple[int, int]],
    reserved: set[tuple[int, int]],
) -> bool:
    h, w = inp.shape
    total_needed = sum(int(rule["n_sc"]) for rule in rules)
    max_radius = 3
    if total_needed < len(rules) * SCATTERED_MIN:
        return False

    for _ in range(SCATTER_CLUSTER_RETRIES):
        radius = random.choice((2, 3, 3))
        cr = random.randint(radius, h - 1 - radius) if h > 2 * radius else random.randint(0, h - 1)
        cc = random.randint(radius, w - 1 - radius) if w > 2 * radius else random.randint(0, w - 1)
        candidates = [
            (int(r), int(c))
            for r in range(max(0, cr - radius), min(h, cr + radius + 1))
            for c in range(max(0, cc - radius), min(w, cc + radius + 1))
            if (r, c) not in occupied and (r, c) not in reserved
        ]
        random.shuffle(candidates)
        blocked = set(occupied)
        picks_by_rule: dict[int, list[tuple[int, int]]] = {i: [] for i in range(len(rules))}

        progressed = True
        while progressed:
            progressed = False
            for i, rule in enumerate(rules):
                if len(picks_by_rule[i]) >= int(rule["n_sc"]):
                    continue
                for cell in candidates:
                    if cell in blocked:
                        continue
                    if _has_adjacent_4(cell, blocked):
                        continue
                    picks_by_rule[i].append(cell)
                    blocked.add(cell)
                    progressed = True
                    break

        if all(len(picks_by_rule[i]) >= int(rules[i]["n_sc"]) for i in range(len(rules))):
            for i, rule in enumerate(rules):
                rule["scattered"] = picks_by_rule[i]
                for r, c in picks_by_rule[i]:
                    inp[r, c] = int(rule["src"])
                    occupied.add((int(r), int(c)))
            return True

        if radius < max_radius:
            continue

    return False


def _stamp_cells_rel(rel: list[tuple[int, int]], sr: int, sc: int, h: int, w: int) -> list[tuple[int, int]]:
    cells: list[tuple[int, int]] = []
    for rr, cc in rel:
        nr = int(sr + rr)
        nc = int(sc + cc)
        if 0 <= nr < h and 0 <= nc < w:
            cells.append((nr, nc))
    return cells


def _place_bystanders(
    rules: list[dict],
    used_source_colors: set[int],
    inp: np.ndarray,
    occupied: set[tuple[int, int]],
    reserved: set[tuple[int, int]],
) -> list[tuple[int, int, int]]:
    bystander_colors = [c for c in BYSTANDER_COLOR_POOL if c not in used_source_colors]
    if not bystander_colors:
        return []

    h, w = inp.shape
    footprint_cells: list[tuple[int, int]] = []
    for rule in rules:
        for sr, sc in rule["scattered"]:
            footprint_cells.extend(_stamp_cells_rel(rule["rel"], sr, sc, h, w))
    counts = Counter(footprint_cells)

    placed: list[tuple[int, int, int]] = []
    targeted_candidates = [
        cell
        for cell, _count in counts.most_common()
        if cell not in occupied
        and cell not in reserved
        and not _has_adjacent_4(cell, occupied)
    ]
    if targeted_candidates and random.random() < 0.7:
        r, c = targeted_candidates[0]
        bc = random.choice(bystander_colors)
        inp[r, c] = int(bc)
        placed.append((int(r), int(c), int(bc)))
        occupied.add((int(r), int(c)))

    remaining = max(0, random.randint(0, BYSTANDER_MAX) - len(placed))
    for _ in range(remaining):
        for _ in range(BYSTANDER_RETRIES):
            r, c = random.randint(0, h - 1), random.randint(0, w - 1)
            if (r, c) in occupied or (r, c) in reserved or _has_adjacent_4((r, c), occupied):
                continue
            bc = random.choice(bystander_colors)
            inp[r, c] = int(bc)
            placed.append((int(r), int(c), int(bc)))
            occupied.add((int(r), int(c)))
            break

    return placed


def _build_output(inp: np.ndarray, rules: list[dict], h: int, w: int, bg: int = 0) -> tuple[np.ndarray, dict]:
    translated_rules = []
    for rule in rules:
        anchor = (0, 0)
        pattern = np.asarray([(rr, cc) for rr, cc in rule["rel"]], dtype=int)
        translated_rules.append(
            {
                "source_color": int(rule["src"]),
                "target_color": int(rule["tgt"]),
                "source_pos": tuple(rule["source_pos"]),
                "target_pos": tuple(rule["target_pos"]),
                "anchor": anchor,
                "pattern": pattern,
            }
        )

    g = np.asarray(inp, dtype=int)
    legend_pixels = _legend_pixels(translated_rules)
    out = np.array(g, dtype=int, copy=True)
    out[out == PATTERN_COLOR] = int(bg)

    target_hits: dict[tuple[int, int], set[int]] = {}
    hit_counts: Counter[tuple[int, int]] = Counter()
    source_cells = {tuple(pos) for rule in rules for pos in rule["scattered"]}

    for rule in rules:
        for sr, sc in rule["scattered"]:
            for nr, nc in _stamp_cells_rel(rule["rel"], sr, sc, h, w):
                cell = (int(nr), int(nc))
                hit_counts[cell] += 1
                target_hits.setdefault(cell, set()).add(int(rule["tgt"]))
                out[nr, nc] = _merge_color(out[nr, nc], int(rule["tgt"]), bg=bg)

    collision_cells = {(int(r), int(c)) for r, c in np.argwhere(out == COLLISION_COLOR)}
    multi_target_overlap_cells = {
        cell for cell, colors in target_hits.items() if len(colors) >= 2
    }
    metrics = {
        "source_count": len(source_cells),
        "source_collision_cells": sum(cell in collision_cells for cell in source_cells),
        "collision_cells": len(collision_cells),
        "extra_collision_cells": len(collision_cells) - len(source_cells),
        "template_overlap_cells": sum(count >= 2 for count in hit_counts.values()),
        "multi_target_overlap_cells": len(multi_target_overlap_cells),
        "multi_target_collision_cells": sum(cell in collision_cells for cell in multi_target_overlap_cells),
        "legend_non_bg_cells": sum(int(out[r, c]) != int(bg) for r, c in legend_pixels),
        "legend_changed_cells": sum(int(out[r, c]) != int(g[r, c]) for r, c in legend_pixels),
    }
    return out, metrics


def _build_overwrite_output(inp: np.ndarray, rules: list[dict], h: int, w: int, bg: int = 0) -> np.ndarray:
    out = np.array(inp, dtype=int, copy=True)
    out[out == PATTERN_COLOR] = int(bg)
    for rule in rules:
        for sr, sc in rule["scattered"]:
            for nr, nc in _stamp_cells_rel(rule["rel"], sr, sc, h, w):
                out[nr, nc] = int(rule["tgt"])
    return out


def _gen_attempt() -> np.ndarray | None:
    num_rules = random.choice(RULE_COUNT_WEIGHTS)
    min_dim = MIN_DIM_BY_RULES[num_rules]
    h = random.randint(min_dim, GRID_DIM_MAX)
    w = random.randint(min_dim, GRID_DIM_MAX)

    built = _build_rules(num_rules, h * w)
    if built == (None, None):
        return None
    rules, used_source_colors = built
    assert rules is not None
    assert used_source_colors is not None

    occupied: set[tuple[int, int]] = set()
    reserved: set[tuple[int, int]] = set()
    inp = np.zeros((h, w), dtype=int)

    if not _place_legends(rules, inp, occupied, reserved):
        return None
    if not _place_scattered(rules, inp, occupied, reserved):
        return None
    _place_bystanders(rules, used_source_colors, inp, occupied, reserved)

    output, metrics = _build_output(inp, rules, h, w)
    overwrite = _build_overwrite_output(inp, rules, h, w)

    if metrics["source_collision_cells"] < metrics["source_count"]:
        return None
    if metrics["multi_target_overlap_cells"] < 1:
        return None
    if metrics["multi_target_collision_cells"] < 1:
        return None
    if metrics["legend_non_bg_cells"] < 2 * num_rules:
        return None
    if np.array_equal(output, overwrite):
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
