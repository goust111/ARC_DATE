# Variant of abc82100
# Dimension: D6 — 颜色重着色规则 (transform)
# Change: teal模板像素统一重着色为规则的目标色（单一颜色替换）... → 模板保留多色（teal标记非teal保留）

from __future__ import annotations

import random

import numpy as np

from arc_utils import components_mask, get_bg

# Design analysis for the D6 variant
# ----------------------------------
# a) Concrete behavior
# spec_variant:
#   grid:
#     same canvas regime as the source task: 5-20 x 5-20, black background
#   rule_example:
#     still a 4-neighbor source -> target pair followed by an anchor cell in the
#     same direction, with a nearby teal component that identifies the rule
#   template:
#     the copied motif is no longer "all teal"; each rule now has:
#       - a teal 8-connected backbone
#       - 1-3 extra accent cells of non-teal colors, each within Chebyshev 1 of
#         the teal backbone, usually diagonal-only attachments
#     when the motif is stamped elsewhere:
#       - teal cells become the rule's target color
#       - accent cells keep their original colors
#   example_trace:
#     old:
#       legend shows red -> yellow next to a teal zig-zag; every scattered red
#       pixel becomes a fully yellow zig-zag
#     new:
#       legend shows red -> yellow next to a motif whose teal zig-zag also has,
#       say, one grey and one blue accent pixel; every scattered red pixel becomes
#       the same motif, but only the teal zig-zag turns yellow while grey/blue stay
#       grey/blue
#
# b) Complexity calibration
#   - Each rule is still anchored by a visible local legend, so the new freedom is
#     only inside the template coloring, not in the matching logic.
#   - Every rule gets at least one non-teal accent cell. This prevents silent
#     collapse back to the source task.
#   - Accent count is capped at 3 so the motif stays readable and the legend does
#     not become cluttered.
#
# c) Old assumptions that break under D6
#   - Old _discover_rules stored only teal coordinates. That loses the accent cells.
#   - Old preservation logic copied every non-bg non-teal pixel unless it was a
#     legend pair or a consumed source color. Under D6 that would wrongly preserve
#     accent cells from the legend/template area into the output.
#   - Old stamping logic wrote one flat target color onto the whole template.
#     Under D6 that would destroy fixed accent colors.
#
# d) Placement legality
#   - Generator keeps a full Chebyshev-1 halo around each legend/template region.
#     That guarantees "cells within distance 1 of a teal component" belong only to
#     that template, so solver-side template extraction is unambiguous.
#   - Accent cells are forbidden from being 4-neighbors of the legend source or
#     target cells, and accent-accent 4-neighbor pairs are also forbidden. This
#     avoids spurious source->target rule discoveries.
#
# e) Visual observability
#   - Humans can see the rule because each legend explicitly shows which cells are
#     changeable (teal) and which are fixed (other colors).
#   - The output visibly differs from the source task: repeated motifs are no
#     longer monochrome blocks.
#
# f) Visual cleanliness
#   - Accents stay local to the teal backbone and legends are isolated by halos, so
#     bystanders and scattered sources do not visually fuse with the templates.
#
# g) Color-change inferability
#   - No hidden ordering by color id is used.
#   - The only color transform is local and visible: teal positions adopt the
#     adjacent target color; non-teal motif positions keep the color already shown
#     in the legend.
#
# main() assumption audit after the change
#   - _discover_rules assumes each true legend source pixel has exactly one nearby
#     teal component within anchor distance <= 1 that is isolated from other
#     non-background cells by a Chebyshev-1 halo. Generator enforces this.
#   - _extract_template assumes every non-background, non-legend cell within
#     Chebyshev 1 of that teal component belongs to the template. Generator enforces
#     this by keeping unrelated objects away from the halo.
#   - Stamping assumes source colors are unique across rules. Generator preserves
#     the original task's unique source/target pairing scheme.


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
    pattern_color: int = 8,
) -> tuple[list[tuple[int, int, int]], list[tuple[int, int]]]:
    """Collect the full multicolor template around a teal component."""
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
                template, template_cells = _extract_template(
                    grid,
                    pattern=pattern,
                    anchor=anchor,
                    excluded={(int(r), int(c)), (int(tr), int(tc))},
                    bg=bg,
                    pattern_color=pattern_color,
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


def main(grid: np.ndarray) -> np.ndarray:
    """
    Learn local source-target-template examples and stamp multicolor motifs.

    Input structure:
    - Black background with isolated colored source pixels and rule legends.
    - Each legend is a 4-neighbor source->target pair whose forward direction
      reaches an anchor beside a teal-backed template that may also contain
      non-teal accent pixels.

    Rule:
    - For every other pixel of the legend's source color, translate that rule's
      template so the legend anchor lands on the source pixel being replaced.
    - Cells that were teal in the legend become the legend's target color; cells
      that were any other color keep that original color.
    - Remove the rule legends and the original template drawings from the output.

    Key insight: The adjacent source/target pixels still encode which source color
    triggers which target color, but the nearby template now separates "changeable"
    teal cells from fixed accent cells that must be copied unchanged.
    """
    g = np.asarray(grid, dtype=int)
    bg = get_bg(g)
    pattern_color = 8
    h, w = g.shape

    rules = _discover_rules(g, bg=bg, pattern_color=pattern_color)
    legend_pixels: set[tuple[int, int]] = set()
    template_pixels: set[tuple[int, int]] = set()
    source_colors: set[int] = set()
    for rule in rules:
        legend_pixels.add(rule["source_pos"])
        legend_pixels.add(rule["target_pos"])
        template_pixels.update(rule["template_cells"])
        source_colors.add(rule["source_color"])

    out = np.full_like(g, int(bg))

    # Preserve only unrelated colored pixels; legend/template cells are erased.
    for r, c in map(tuple, np.argwhere((g != int(bg)) & (g != pattern_color))):
        r = int(r)
        c = int(c)
        color = int(g[r, c])
        if (r, c) in legend_pixels or (r, c) in template_pixels or color in source_colors:
            continue
        out[r, c] = color

    for rule in rules:
        ar, ac = rule["anchor"]
        template = rule["template"]
        for sr, sc in map(tuple, np.argwhere(g == rule["source_color"])):
            sr = int(sr)
            sc = int(sc)
            if (sr, sc) in legend_pixels:
                continue
            dr = sr - ar
            dc = sc - ac
            for rr, cc, color in template:
                nr = int(ar + rr + dr)
                nc = int(ac + cc + dc)
                if 0 <= nr < h and 0 <= nc < w:
                    out[nr, nc] = rule["target_color"] if color == pattern_color else color

    return out


# --------------- generator (c) ---------------

# ── CFG: tunable generation parameters ──────────────────────────────
GRID_DIM_MAX = 20
MIN_DIM_BY_RULES = {1: 5, 2: 10, 3: 14, 4: 18}

RULE_COUNT_WEIGHTS = [1, 1, 2, 2, 3, 3, 4]
NON_BG_NON_TEAL_COLORS = [1, 2, 3, 4, 5, 6, 7, 9]

STENCIL_MAX_CAP = 12
STENCIL_MIN_AREA_DIV = 100
STENCIL_MAX_AREA_DIV = 25

ACCENT_MIN = 1
ACCENT_MAX = 3
TEMPLATE_GEN_RETRIES = 80

SCATTERED_MIN = 3
SCATTERED_MAX_CAP = 8
SCATTERED_AREA_DIV = 30

BYSTANDER_MAX = 5

LEGEND_RETRIES = 500
SCATTERED_RETRIES_PER_PIXEL = 50
BYSTANDER_RETRIES = 50
OUTER_RETRIES = 400

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
        random_candidates = list(candidates)
        random.shuffle(random_candidates)

        accents: list[tuple[int, int, int]] = []
        for pos in random_candidates:
            if any(_manhattan(pos, (ar, ac)) == 1 for ar, ac, _ in accents):
                continue
            color = random.choice(accent_pool)
            accents.append((int(pos[0]), int(pos[1]), int(color)))
            if len(accents) >= accent_count:
                break
        if not accents:
            continue

        ar, ac = anchor
        rel_template = [(int(r - ar), int(c - ac), 8) for r, c in teal_coords]
        rel_template.extend((int(r - ar), int(c - ac), int(color)) for r, c, color in accents)
        rel_template.sort()
        return rel_template
    raise RuntimeError("Failed to generate multicolor template")


def _build_rules(num_rules: int, area: int):
    colors = NON_BG_NON_TEAL_COLORS[:]
    random.shuffle(colors)

    max_s = min(STENCIL_MAX_CAP, max(2, area // STENCIL_MAX_AREA_DIV))
    min_s = min(max(1, area // STENCIL_MIN_AREA_DIV), max_s)
    sc_max = min(SCATTERED_MAX_CAP, max(SCATTERED_MIN, area // SCATTERED_AREA_DIV))

    rules, used = [], set()
    for i in range(num_rules):
        src_c, tgt_c = colors[2 * i], colors[2 * i + 1]
        used.update((src_c, tgt_c))
        template = _make_multicolor_template(min_s, max_s, src_c, tgt_c)
        rules.append(
            {
                "src": src_c,
                "tgt": tgt_c,
                "template": template,
                "n_sc": random.randint(SCATTERED_MIN, sc_max),
            }
        )
    return rules, used


def _absolute_template_cells(rule, anchor: tuple[int, int]):
    ar, ac = anchor
    teal_cells: set[tuple[int, int]] = set()
    accent_cells: list[tuple[int, int, int]] = []
    for rr, cc, color in rule["template"]:
        nr = int(ar + rr)
        nc = int(ac + cc)
        if color == 8:
            teal_cells.add((nr, nc))
        else:
            accent_cells.append((nr, nc, int(color)))
    return teal_cells, accent_cells


def _place_legends(rules, inp, occupied, reserved):
    H, W = inp.shape
    for rule in rules:
        placed = False
        for _ in range(LEGEND_RETRIES):
            dr, dc = random.choice(_DIRS4)
            sr, sc = random.randint(0, H - 1), random.randint(0, W - 1)
            tr, tc = sr + dr, sc + dc
            ar, ac = sr + 2 * dr, sc + 2 * dc
            if not (0 <= tr < H and 0 <= tc < W and 0 <= ar < H and 0 <= ac < W):
                continue

            teal_cells, accent_cells = _absolute_template_cells(rule, (ar, ac))
            all_template_cells = set(teal_cells) | {(r, c) for r, c, _ in accent_cells}
            if not all_template_cells:
                continue
            if any(not (0 <= r < H and 0 <= c < W) for r, c in all_template_cells):
                continue
            if any(_manhattan((r, c), (sr, sc)) == 1 or _manhattan((r, c), (tr, tc)) == 1 for r, c, _ in accent_cells):
                continue

            all_cells = {(sr, sc), (tr, tc)} | all_template_cells
            if all_cells & (occupied | reserved):
                continue

            inp[sr, sc] = rule["src"]
            inp[tr, tc] = rule["tgt"]
            for nr, nc in teal_cells:
                inp[nr, nc] = 8
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


def _place_scattered(rules, inp, occupied, reserved):
    H, W = inp.shape
    for rule in rules:
        positions = []
        for _ in range(rule["n_sc"] * SCATTERED_RETRIES_PER_PIXEL):
            if len(positions) >= rule["n_sc"]:
                break
            r, c = random.randint(0, H - 1), random.randint(0, W - 1)
            if (r, c) in occupied or (r, c) in reserved:
                continue
            positions.append((int(r), int(c)))
            occupied.add((int(r), int(c)))
            _reserve_4neighbors(reserved, [(r, c)])
        if len(positions) < SCATTERED_MIN:
            return False
        rule["scattered"] = positions
        for r, c in positions:
            inp[r, c] = rule["src"]
    return True


def _place_bystanders(used_colors, inp, occupied, reserved):
    bystander_colors = [c for c in NON_BG_NON_TEAL_COLORS if c not in used_colors]
    if not bystander_colors:
        return []
    H, W = inp.shape
    n = random.randint(0, min(BYSTANDER_MAX, len(bystander_colors) * 2))
    placed = []
    for _ in range(n):
        for _ in range(BYSTANDER_RETRIES):
            r, c = random.randint(0, H - 1), random.randint(0, W - 1)
            if (r, c) not in occupied and (r, c) not in reserved:
                bc = random.choice(bystander_colors)
                inp[r, c] = bc
                placed.append((int(r), int(c), int(bc)))
                occupied.add((int(r), int(c)))
                _reserve_4neighbors(reserved, [(r, c)])
                break
    return placed


def _build_output(rules, bystander_pos, H, W):
    output = np.zeros((H, W), dtype=int)
    for r, c, bc in bystander_pos:
        output[r, c] = bc
    for rule in sorted(rules, key=lambda item: item["source_pos"]):
        for sr, sc in rule["scattered"]:
            for rr, cc, color in rule["template"]:
                nr = int(sr + rr)
                nc = int(sc + cc)
                if 0 <= nr < H and 0 <= nc < W:
                    output[nr, nc] = rule["tgt"] if color == 8 else color
    return output


def _gen_attempt():
    num_rules = random.choice(RULE_COUNT_WEIGHTS)
    min_dim = MIN_DIM_BY_RULES[num_rules]
    H = random.randint(min_dim, GRID_DIM_MAX)
    W = random.randint(min_dim, GRID_DIM_MAX)

    rules, used_colors = _build_rules(num_rules, H * W)
    occupied, reserved = set(), set()
    inp = np.zeros((H, W), dtype=int)

    if not _place_legends(rules, inp, occupied, reserved):
        return None
    if not _place_scattered(rules, inp, occupied, reserved):
        return None
    bystander_pos = _place_bystanders(used_colors, inp, occupied, reserved)

    output = _build_output(rules, bystander_pos, H, W)
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
    raise RuntimeError("Failed to generate valid input after %d attempts" % OUTER_RETRIES)
