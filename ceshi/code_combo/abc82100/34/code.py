# Combo variant of abc82100 (index 34)
# combo_spec: D6/0+D12/0
# Name: OR合并重叠+多色模板保留

from __future__ import annotations

import random
from collections import Counter, defaultdict

import numpy as np

from arc_utils import components_mask, get_bg

# Design analysis for combo D6/0 + D12/0
# ---------------------------------------
# a) Concrete behavior and trace
#   - Baseline abc82100 learns local legends of the form:
#       source pixel -> target pixel -> nearby teal stencil.
#     Every scattered source pixel of that color receives a translated copy of the
#     teal stencil recolored uniformly to the target color.
#   - D6/0 changes the copied object: the template now has a teal backbone plus
#     1-3 fixed accent cells. When copied, teal cells become the rule's target
#     color, but accent cells keep their own colors.
#   - D12/0 changes overlap semantics from "later write overwrites earlier write"
#     to a genuine order-free merge. For this combo, we make the OR semantics
#     visible and inferable by splitting template cells into two roles:
#       * recolored teal cells = mutable layer
#       * non-teal accent cells = fixed layer
#     If multiple stamps touch the same output cell, any fixed-layer claim wins
#     over mutable-layer claims; within one layer we only generate same-color
#     overlaps. Example:
#       rule A legend shows red->yellow next to a teal zig-zag with one blue
#       accent. rule B legend shows green->orange next to another template. If a
#       scattered A stamp places its blue accent on a cell that a scattered B
#       stamp also paints orange via teal, the output keeps blue there. The union
#       is therefore order-independent and visibly different from overwrite.
#
# b) Complexity calibration
#   - Every rule has at least one accent cell, so the task cannot silently reduce
#     to the monochrome original.
#   - Each sample is forced to contain at least one accent-over-target overlap, so
#     OR merging is not merely theoretical.
#   - Accent count stays in 1-3 and teal stencil size stays small, keeping the
#     legends readable while still making overlap behavior visually obvious.
#
# c) Hidden assumptions broken by the combo
#   - Original rule discovery stored only teal component coordinates; that loses
#     accent cells, so the solver must extract the full multicolor template.
#   - Original preservation logic kept every non-bg non-teal pixel outside legend
#     pairs. Under D6 this would wrongly preserve legend accent cells in output.
#   - Original stamping line `out[...] = target_color` encoded overwrite order.
#     Under D12/0 the solver must accumulate claims first and resolve them after
#     all stamps have been considered.
#
# d) Placement legality
#   - Legends keep a Chebyshev-1 halo around their whole template region, so all
#     cells within distance 1 of a teal component belong only to that template.
#   - Scattered source pixels stay pairwise non-4-adjacent and outside legend
#     halos, preventing accidental fake legends.
#   - Overlaps are constrained to the intended family: fixed-vs-target or same-
#     color same-layer. The generator forbids fixed-vs-fixed color conflicts and
#     forbids different target colors on the same cell.
#
# e) Visual observability
#   - Humans can read the legend as "teal changes, accent stays". The output then
#     shows that when translated copies collide, the fixed accent survives.
#   - Because every sample includes at least one mixed overlap and also several
#     ordinary non-overlapping placements, the difference from both the original
#     task and pure D6 is visible from a few examples.
#
# f) Visual cleanliness
#   - Legend regions are separated by halos; only output stamps are allowed to
#     overlap.
#   - Bystanders are forbidden from final stamp footprints, so preserved pixels do
#     not get entangled with merged templates.
#
# g) Color-change inferability
#   - The only semantic color split is structural and locally shown in the legend:
#     teal cells are the mutable positions, non-teal cells are fixed positions.
#   - No priority depends on hidden color ids. The visible priority is role-based:
#     fixed template color beats recolored teal whenever both claim the same cell.
#
# main() assumption audit after the combo
#   - _extract_template assumes every non-background cell inside the teal
#     component's Chebyshev-1 halo belongs to that template. Generator enforces
#     this via legend halos.
#   - _resolve_bucket assumes same-layer overlaps never need a meaningful tie-break
#     between different colors. Generator enforces this by rejecting such cases.
#   - The source-color-to-rule mapping is still unique across rules. Generator
#     preserves the original pair selection regime.
#   - Preserved bystanders are assumed to lie outside all translated template
#     footprints. Generator enforces that directly.

PATTERN_COLOR = 8

# --------------- solver ---------------


def _chebyshev(a: tuple[int, int], b: tuple[int, int]) -> int:
    return max(abs(int(a[0]) - int(b[0])), abs(int(a[1]) - int(b[1])))


def _manhattan(a: tuple[int, int], b: tuple[int, int]) -> int:
    return abs(int(a[0]) - int(b[0])) + abs(int(a[1]) - int(b[1]))


def _anchor_distance(pattern: np.ndarray, anchor: tuple[int, int]) -> int:
    return min(_chebyshev((int(r), int(c)), anchor) for r, c in pattern)


def _extract_template(
    grid: np.ndarray,
    pattern: np.ndarray,
    anchor: tuple[int, int],
    excluded: set[tuple[int, int]],
    bg: int,
) -> tuple[list[tuple[int, int, int]], list[tuple[int, int]]]:
    """Collect all legend template cells inside the teal component halo."""
    rows = pattern[:, 0]
    cols = pattern[:, 1]
    r0 = max(0, int(rows.min()) - 1)
    r1 = min(grid.shape[0] - 1, int(rows.max()) + 1)
    c0 = max(0, int(cols.min()) - 1)
    c1 = min(grid.shape[1] - 1, int(cols.max()) + 1)

    ar, ac = anchor
    template: list[tuple[int, int, int]] = []
    template_cells: list[tuple[int, int]] = []
    for r in range(r0, r1 + 1):
        for c in range(c0, c1 + 1):
            if (r, c) in excluded:
                continue
            color = int(grid[r, c])
            if color == int(bg):
                continue
            if _anchor_distance(pattern, (r, c)) > 1:
                continue
            template.append((int(r - ar), int(c - ac), color))
            template_cells.append((int(r), int(c)))

    template.sort()
    template_cells.sort()
    return template, template_cells


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
                template, template_cells = _extract_template(
                    grid,
                    pattern=pattern,
                    anchor=anchor,
                    excluded={(int(r), int(c)), (int(tr), int(tc))},
                    bg=bg,
                )
                rules.append(
                    {
                        "source_color": int(source_color),
                        "target_color": int(target_color),
                        "source_pos": (int(r), int(c)),
                        "target_pos": (int(tr), int(tc)),
                        "anchor": (int(anchor[0]), int(anchor[1])),
                        "template": template,
                        "template_cells": template_cells,
                    }
                )

    return rules


def _resolve_color_counter(counter: Counter[int]) -> int:
    best = max(counter.values())
    winners = [color for color, count in counter.items() if count == best]
    return int(min(winners))


def _resolve_bucket(bucket: dict[str, Counter[int]], bg: int) -> int:
    fixed_counter = bucket["fixed"]
    if fixed_counter:
        return _resolve_color_counter(fixed_counter)
    target_counter = bucket["target"]
    if target_counter:
        return _resolve_color_counter(target_counter)
    return int(bg)


def _record_claim(
    claims: dict[tuple[int, int], dict[str, Counter[int]]],
    row: int,
    col: int,
    kind: str,
    color: int,
) -> None:
    claims[(int(row), int(col))][kind][int(color)] += 1


def main(grid: np.ndarray) -> np.ndarray:
    """
    Learn local source-target-template examples, then stamp multicolor motifs with
    order-free overlap merging.

    Input structure:
    - Black background with isolated colored source pixels and local legends.
    - Each legend is a 4-neighbor source->target pair whose forward direction
      reaches an anchor beside a teal-backed template; the template may also
      contain non-teal accent pixels inside the teal component's halo.

    Rule:
    - For every non-legend pixel of a learned source color, translate that rule's
      whole template so the legend anchor lands on the source pixel.
    - Cells that were teal in the legend become the rule's target color; cells
      that were already non-teal keep their original template color.
    - Merge all translated copies by OR semantics: any fixed non-teal template
      color overrides recolored teal on overlapping cells, while same-layer
      overlaps are treated as one color claim. Remove the original legends and
      their template drawings from the output.

    Key insight: The legend now teaches two things at once: which source color
    activates which target color, and which template cells are mutable (`teal`)
    versus fixed (already-colored accents). Overlap is resolved by that visible
    role split, not by the order in which sources are processed.
    """
    g = np.asarray(grid, dtype=int)
    bg = get_bg(g)
    h, w = g.shape

    rules = _discover_rules(g, bg=bg, pattern_color=PATTERN_COLOR)
    legend_pixels: set[tuple[int, int]] = set()
    template_pixels: set[tuple[int, int]] = set()
    source_colors: set[int] = set()
    for rule in rules:
        legend_pixels.add(rule["source_pos"])
        legend_pixels.add(rule["target_pos"])
        template_pixels.update(rule["template_cells"])
        source_colors.add(rule["source_color"])

    out = np.full_like(g, int(bg))

    for r, c in map(tuple, np.argwhere((g != int(bg)) & (g != PATTERN_COLOR))):
        r = int(r)
        c = int(c)
        color = int(g[r, c])
        if (r, c) in legend_pixels or (r, c) in template_pixels or color in source_colors:
            continue
        out[r, c] = color

    claims: dict[tuple[int, int], dict[str, Counter[int]]] = defaultdict(
        lambda: {"fixed": Counter(), "target": Counter()}
    )
    for rule in rules:
        for sr, sc in map(tuple, np.argwhere(g == rule["source_color"])):
            sr = int(sr)
            sc = int(sc)
            if (sr, sc) in legend_pixels:
                continue
            for rr, cc, color in rule["template"]:
                nr = int(sr + rr)
                nc = int(sc + cc)
                if 0 <= nr < h and 0 <= nc < w:
                    if color == PATTERN_COLOR:
                        _record_claim(claims, nr, nc, "target", int(rule["target_color"]))
                    else:
                        _record_claim(claims, nr, nc, "fixed", int(color))

    for (r, c), bucket in claims.items():
        out[r, c] = _resolve_bucket(bucket, int(bg))

    return out


# --------------- generator ---------------

# Grid dimensions
GRID_DIM_MAX = 20
MIN_DIM_BY_RULES = {1: 5, 2: 10, 3: 14, 4: 18}

# Rule sampling
RULE_COUNT_WEIGHTS = [1, 1, 2, 2, 3, 3, 4]
NON_BG_NON_TEAL_COLORS = [1, 2, 3, 4, 5, 6, 7, 9]

# Template size
STENCIL_MAX_CAP = 12
STENCIL_MIN_AREA_DIV = 100
STENCIL_MAX_AREA_DIV = 25
ACCENT_MIN = 1
ACCENT_MAX = 3
TEMPLATE_GEN_RETRIES = 120

# Scattered source pixels per rule
SCATTERED_MIN = 3
SCATTERED_MAX_CAP = 8
SCATTERED_AREA_DIV = 30

# Bystanders
BYSTANDER_MAX = 5

# Placement retry budgets
LEGEND_RETRIES = 500
SCATTERED_RETRIES_PER_PIXEL = 80
FORCED_OVERLAP_RETRIES = 1600
BYSTANDER_RETRIES = 80
OUTER_RETRIES = 500

_DIRS4 = [(-1, 0), (1, 0), (0, -1), (0, 1)]
_DIRS8 = [(-1, -1), (-1, 0), (-1, 1), (0, -1), (0, 1), (1, -1), (1, 0), (1, 1)]


def _reserve_4neighbors(reserved: set[tuple[int, int]], cells) -> None:
    for r, c in cells:
        for dr, dc in _DIRS4:
            reserved.add((int(r + dr), int(c + dc)))


def _reserve_8neighbors(reserved: set[tuple[int, int]], cells) -> None:
    for r, c in cells:
        for dr, dc in _DIRS8:
            reserved.add((int(r + dr), int(c + dc)))


def _make_teal_stencil(min_pixels: int, max_pixels: int):
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


def _make_multicolor_template(min_pixels: int, max_pixels: int, src_c: int, tgt_c: int):
    accent_pool = [c for c in NON_BG_NON_TEAL_COLORS if c not in (src_c, tgt_c)]
    for _ in range(TEMPLATE_GEN_RETRIES):
        teal_coords, anchor = _make_teal_stencil(min_pixels, max_pixels)
        teal_set = set(teal_coords)
        candidates: set[tuple[int, int]] = set()
        for r, c in teal_coords:
            for dr in (-1, 1):
                for dc in (-1, 1):
                    p = (r + dr, c + dc)
                    if p not in teal_set:
                        candidates.add(p)
        if not candidates:
            continue

        accent_count = min(random.randint(ACCENT_MIN, ACCENT_MAX), len(candidates))
        shuffled = list(candidates)
        random.shuffle(shuffled)

        accents: list[tuple[int, int, int]] = []
        for pos in shuffled:
            if any(_manhattan(pos, (ar, ac)) == 1 for ar, ac, _ in accents):
                continue
            accents.append((int(pos[0]), int(pos[1]), int(random.choice(accent_pool))))
            if len(accents) >= accent_count:
                break
        if not accents:
            continue

        ar, ac = anchor
        template = [(int(r - ar), int(c - ac), PATTERN_COLOR) for r, c in teal_coords]
        template.extend((int(r - ar), int(c - ac), int(color)) for r, c, color in accents)
        template.sort()
        return template
    raise RuntimeError("Failed to generate multicolor template")


def _build_rules(num_rules: int, area: int):
    colors = NON_BG_NON_TEAL_COLORS[:]
    random.shuffle(colors)

    max_s = min(STENCIL_MAX_CAP, max(2, area // STENCIL_MAX_AREA_DIV))
    min_s = min(max(1, area // STENCIL_MIN_AREA_DIV), max_s)
    sc_max = min(SCATTERED_MAX_CAP, max(SCATTERED_MIN, area // SCATTERED_AREA_DIV))

    rules: list[dict] = []
    used_colors: set[int] = set()
    for i in range(num_rules):
        src_c, tgt_c = colors[2 * i], colors[2 * i + 1]
        template = _make_multicolor_template(min_s, max_s, src_c, tgt_c)
        accents = [(rr, cc, color) for rr, cc, color in template if color != PATTERN_COLOR]
        teal_offsets = [(rr, cc) for rr, cc, color in template if color == PATTERN_COLOR]
        rules.append(
            {
                "src": int(src_c),
                "tgt": int(tgt_c),
                "template": template,
                "accent_entries": accents,
                "teal_offsets": teal_offsets,
                "n_sc": random.randint(SCATTERED_MIN, sc_max),
            }
        )
        used_colors.update((int(src_c), int(tgt_c)))
        used_colors.update(int(color) for _rr, _cc, color in accents)
    return rules, used_colors


def _absolute_template_cells(rule, anchor: tuple[int, int]):
    ar, ac = anchor
    teal_cells: set[tuple[int, int]] = set()
    accent_cells: list[tuple[int, int, int]] = []
    for rr, cc, color in rule["template"]:
        nr = int(ar + rr)
        nc = int(ac + cc)
        if color == PATTERN_COLOR:
            teal_cells.add((nr, nc))
        else:
            accent_cells.append((nr, nc, int(color)))
    return teal_cells, accent_cells


def _place_legends(rules, inp, occupied, reserved):
    h, w = inp.shape
    for rule in rules:
        placed = False
        for _ in range(LEGEND_RETRIES):
            dr, dc = random.choice(_DIRS4)
            sr, sc = random.randint(0, h - 1), random.randint(0, w - 1)
            tr, tc = sr + dr, sc + dc
            ar, ac = sr + 2 * dr, sc + 2 * dc
            if not (0 <= tr < h and 0 <= tc < w and 0 <= ar < h and 0 <= ac < w):
                continue

            teal_cells, accent_cells = _absolute_template_cells(rule, (ar, ac))
            template_cells = set(teal_cells) | {(r, c) for r, c, _ in accent_cells}
            if not template_cells:
                continue
            if any(not (0 <= r < h and 0 <= c < w) for r, c in template_cells):
                continue
            if any(
                _manhattan((r, c), (sr, sc)) == 1 or _manhattan((r, c), (tr, tc)) == 1
                for r, c, _color in accent_cells
            ):
                continue

            all_cells = {(sr, sc), (tr, tc)} | template_cells
            if all_cells & (occupied | reserved):
                continue

            inp[sr, sc] = rule["src"]
            inp[tr, tc] = rule["tgt"]
            for nr, nc in teal_cells:
                inp[nr, nc] = PATTERN_COLOR
            for nr, nc, color in accent_cells:
                inp[nr, nc] = color

            occupied.update(all_cells)
            _reserve_4neighbors(reserved, all_cells)
            _reserve_8neighbors(reserved, all_cells)
            rule["source_pos"] = (int(sr), int(sc))
            rule["target_pos"] = (int(tr), int(tc))
            rule["anchor"] = (int(ar), int(ac))
            placed = True
            break
        if not placed:
            return False
    return True


def _stamp_claims(rule, source_pos: tuple[int, int], h: int, w: int):
    sr, sc = source_pos
    claims: list[tuple[int, int, str, int]] = []
    for rr, cc, color in rule["template"]:
        nr = int(sr + rr)
        nc = int(sc + cc)
        if 0 <= nr < h and 0 <= nc < w:
            if color == PATTERN_COLOR:
                claims.append((nr, nc, "target", int(rule["tgt"])))
            else:
                claims.append((nr, nc, "fixed", int(color)))
    return claims


def _can_use_source(
    pos: tuple[int, int],
    occupied: set[tuple[int, int]],
    reserved: set[tuple[int, int]],
    pending: list[tuple[int, int]],
) -> bool:
    if pos in occupied or pos in reserved:
        return False
    if pos in pending:
        return False
    return all(_manhattan(pos, other) > 1 for other in pending)


def _claims_compatible(claim_map, new_claims) -> bool:
    scratch: dict[tuple[int, int], dict[str, set[int]]] = {}
    for row, col, kind, color in new_claims:
        cell = (int(row), int(col))
        if cell not in scratch:
            existing = claim_map.get(cell)
            scratch[cell] = {
                "fixed": set() if existing is None else set(existing["fixed"]),
                "target": set() if existing is None else set(existing["target"]),
            }
        scratch[cell][kind].add(int(color))
        if len(scratch[cell]["fixed"]) > 1 or len(scratch[cell]["target"]) > 1:
            return False
    return True


def _commit_claims(claim_map, claims) -> None:
    for row, col, kind, color in claims:
        claim_map[(int(row), int(col))][kind].add(int(color))


def _force_overlap_pair(rules, h, w, occupied, reserved, claim_map):
    rule_order = list(range(len(rules)))
    random.shuffle(rule_order)

    # Prefer a same-rule overlap whose row-major source order makes overwrite
    # semantics diverge from the fixed-over-target OR rule.
    for i in rule_order:
        rule = rules[i]
        if not rule["accent_entries"] or not rule["teal_offsets"]:
            continue
        for _ in range(FORCED_OVERLAP_RETRIES):
            arr, acc, accent_color = random.choice(rule["accent_entries"])
            trr, tcc = random.choice(rule["teal_offsets"])
            if int(accent_color) == int(rule["tgt"]):
                continue
            overlap_r = random.randint(0, h - 1)
            overlap_c = random.randint(0, w - 1)
            src_fixed = (int(overlap_r - arr), int(overlap_c - acc))
            src_target = (int(overlap_r - trr), int(overlap_c - tcc))
            if not (0 <= src_fixed[0] < h and 0 <= src_fixed[1] < w):
                continue
            if not (0 <= src_target[0] < h and 0 <= src_target[1] < w):
                continue
            if not (src_fixed < src_target):
                continue
            if not _can_use_source(src_fixed, occupied, reserved, []):
                continue
            if not _can_use_source(src_target, occupied, reserved, [src_fixed]):
                continue

            claims_fixed = _stamp_claims(rule, src_fixed, h, w)
            claims_target = _stamp_claims(rule, src_target, h, w)
            combined = claims_fixed + claims_target
            if not _claims_compatible(claim_map, combined):
                continue

            local: dict[tuple[int, int], dict[str, set[int]]] = defaultdict(
                lambda: {"fixed": set(), "target": set()}
            )
            for row, col, kind, color in combined:
                local[(int(row), int(col))][kind].add(int(color))
            bucket = local[(int(overlap_r), int(overlap_c))]
            if bucket["fixed"] and bucket["target"]:
                protected = {
                    cell
                    for cell, cell_bucket in local.items()
                    if cell_bucket["fixed"] and cell_bucket["target"]
                }
                return [(i, src_fixed, claims_fixed), (i, src_target, claims_target)], protected

    return None


def _place_scattered(rules, inp, occupied, reserved):
    h, w = inp.shape
    claim_map: dict[tuple[int, int], dict[str, set[int]]] = defaultdict(
        lambda: {"fixed": set(), "target": set()}
    )

    for rule in rules:
        rule["scattered"] = []

    forced = _force_overlap_pair(rules, h, w, occupied, reserved, claim_map)
    if forced is None:
        return False
    forced_placements, protected_cells = forced

    for idx, pos, claims in forced_placements:
        rules[idx]["scattered"].append(pos)
        occupied.add(pos)
        _reserve_4neighbors(reserved, [pos])
        _commit_claims(claim_map, claims)

    for rule in rules:
        needed = rule["n_sc"] - len(rule["scattered"])
        tries = max(needed * SCATTERED_RETRIES_PER_PIXEL, SCATTERED_MIN * SCATTERED_RETRIES_PER_PIXEL)
        for _ in range(tries):
            if len(rule["scattered"]) >= rule["n_sc"]:
                break
            pos = (random.randint(0, h - 1), random.randint(0, w - 1))
            if not _can_use_source(pos, occupied, reserved, []):
                continue
            claims = _stamp_claims(rule, pos, h, w)
            if any((int(row), int(col)) in protected_cells for row, col, _kind, _color in claims):
                continue
            if not _claims_compatible(claim_map, claims):
                continue
            rule["scattered"].append((int(pos[0]), int(pos[1])))
            occupied.add((int(pos[0]), int(pos[1])))
            _reserve_4neighbors(reserved, [pos])
            _commit_claims(claim_map, claims)
        if len(rule["scattered"]) < SCATTERED_MIN:
            return False
        for r, c in rule["scattered"]:
            inp[r, c] = rule["src"]

    return True


def _place_bystanders(used_colors, inp, occupied, reserved, rules):
    bystander_colors = [c for c in NON_BG_NON_TEAL_COLORS if c not in used_colors]
    if not bystander_colors:
        return []

    h, w = inp.shape
    forbidden_output_cells: set[tuple[int, int]] = set()
    for rule in rules:
        for sr, sc in rule["scattered"]:
            for rr, cc, _color in rule["template"]:
                nr = int(sr + rr)
                nc = int(sc + cc)
                if 0 <= nr < h and 0 <= nc < w:
                    forbidden_output_cells.add((nr, nc))

    n = random.randint(0, min(BYSTANDER_MAX, len(bystander_colors) * 2))
    placed = []
    for _ in range(n):
        for _ in range(BYSTANDER_RETRIES):
            r, c = random.randint(0, h - 1), random.randint(0, w - 1)
            if (r, c) in occupied or (r, c) in reserved or (r, c) in forbidden_output_cells:
                continue
            color = int(random.choice(bystander_colors))
            inp[r, c] = color
            placed.append((int(r), int(c), color))
            occupied.add((int(r), int(c)))
            _reserve_4neighbors(reserved, [(r, c)])
            break
    return placed


def _build_output(rules, bystander_pos, h, w):
    output = np.zeros((h, w), dtype=int)
    for r, c, color in bystander_pos:
        output[r, c] = int(color)

    claims: dict[tuple[int, int], dict[str, Counter[int]]] = defaultdict(
        lambda: {"fixed": Counter(), "target": Counter()}
    )
    for rule in rules:
        for sr, sc in rule["scattered"]:
            for nr, nc, kind, color in _stamp_claims(rule, (sr, sc), h, w):
                _record_claim(claims, nr, nc, kind, color)

    for (r, c), bucket in claims.items():
        output[r, c] = _resolve_bucket(bucket, 0)
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
    bystander_pos = _place_bystanders(used_colors, inp, occupied, reserved, rules)

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
