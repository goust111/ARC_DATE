# Combo variant of abc82100 (index 7)
# combo_spec: D2/2+D6/0
# Name: 模板色=目标色+多色模板保留

from __future__ import annotations

import random
import sys

import numpy as np

if "/home/lixinyu/snap/new/solvers" not in sys.path:
    sys.path.append("/home/lixinyu/snap/new/solvers")

from arc_utils import components_mask, get_bg


# Design analysis for the combo variant
# analysis:
#   a_concrete_behavior_and_trace:
#     original_task:
#       - a legend is "source pixel -> target pixel -> teal stencil"
#       - solver can globally segment every stencil by reading color 8 components
#       - stamping recolors the whole stencil to target_color
#     combo_task:
#       - the stencil backbone is no longer teal; it is drawn directly in target_color
#       - the same template may also include 1-3 fixed accent cells of other colors
#       - stamping now copies the whole multicolor template unchanged, because the
#         changeable backbone is already shown in target_color
#     mini_trace:
#       - old legend: red, then yellow, then a teal L-shape
#       - combo legend: red, then yellow, then a yellow L-shape plus e.g. one grey diagonal accent
#       - old solver action: detect teal component, recolor it yellow at each scattered red
#       - combo solver action: remove the single yellow legend-target cell from the yellow mask,
#         recover the nearby remaining yellow component, then collect every non-black cell in its
#         Chebyshev-1 halo as the full template; stamp that template as-is at each scattered red
#     key_difference:
#       - "which yellow cell is the rule's target pixel?" and "which nearby cells belong to the
#         copied template?" are no longer separated by a dedicated marker color, so the solver has
#         to separate them structurally rather than chromatically
#   b_complexity_calibration:
#     - every rule still has a unique source_color and target_color pair, preserving learnability
#     - every template is forced to include at least one non-target accent cell, so D6/0 is always visible
#     - color 8 never appears, so D2/2 is always visible and the instance cannot silently collapse to the source task
#     - accents are capped at 3 and kept local to the backbone so the legends stay readable
#   c_implicit_assumptions_audit:
#     - original _discover_rules assumed all template cells are color 8; invalid now
#     - original preservation loop skipped only teal and source colors; invalid now because legend templates
#       are ordinary colored cells and must be removed explicitly
#     - original stamping recolored the entire stencil to target_color; invalid now because accent colors
#       must survive unchanged
#     - new solver assumption 1: after removing the legend target cell from the target-color mask, exactly one
#       nearby same-colored component corresponds to the rule backbone; generator isolates legends to guarantee this
#     - new solver assumption 2: every non-background cell within Chebyshev distance 1 of that backbone belongs
#       to the same template; generator enforces a full halo around each legend/template region
#     - new solver assumption 3: source colors remain unique across rules; generator keeps the original unique pairing
#   d_placement_legality:
#     - legend geometry remains source -> target -> anchor in one cardinal direction
#     - anchor is always a cell of the target-colored backbone
#     - no template cell may be 4-neighbor to the legend source except the legend target
#     - no accent cell may be 4-neighbor to the legend target, preventing spurious alternative legends
#     - legends, scattered sources, and bystanders are separated by 4/8-neighbor reservations
#   e_visual_observability:
#     - humans can still read the local rule because the only adjacent two-cell color pair is source -> target
#     - the copied motif is visibly multicolor, and the absence of teal is visible immediately
#     - repeated scattered sources provide multiple demonstrations that the same multicolor shape is reused
#   f_visual_cleanliness:
#     - accents are attached only within Chebyshev 1 of the target-colored backbone and avoid 4-neighbor chains
#     - legend regions receive an 8-neighbor halo so they do not visually fuse with unrelated objects
#     - bystanders avoid all source/target colors and stay isolated
#   g_change_inferability:
#     - the color mapping is still directly shown by the adjacent source/target pair
#     - the template copy rule is local and structural: copy the entire nearby motif, not just same-colored cells
#     - no ordering by color id or any hidden non-visual feature is used


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


def _chebyshev(a: tuple[int, int], b: tuple[int, int]) -> int:
    return max(abs(int(a[0]) - int(b[0])), abs(int(a[1]) - int(b[1])))


def _manhattan(a: tuple[int, int], b: tuple[int, int]) -> int:
    return abs(int(a[0]) - int(b[0])) + abs(int(a[1]) - int(b[1]))


def _anchor_distance(pattern: np.ndarray, anchor: tuple[int, int]) -> int:
    return min(_chebyshev((int(r), int(c)), anchor) for r, c in pattern)


def _component_signature(pattern: np.ndarray) -> tuple[tuple[int, int], ...]:
    return tuple((int(r), int(c)) for r, c in np.asarray(pattern, dtype=int))


def _extract_template(
    grid: np.ndarray,
    backbone: np.ndarray,
    anchor: tuple[int, int],
    excluded: set[tuple[int, int]],
    bg: int,
) -> tuple[list[tuple[int, int, int]], list[tuple[int, int]]]:
    """Collect the full multicolor template around a target-colored backbone."""
    rows = backbone[:, 0]
    cols = backbone[:, 1]
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
            if _anchor_distance(backbone, (r, c)) > 1:
                continue
            template.append((int(r - ar), int(c - ac), color))
            template_cells.append((int(r), int(c)))

    template.sort()
    template_cells.sort()
    return template, template_cells


def _discover_rules(grid: np.ndarray, bg: int) -> list[dict]:
    h, w = grid.shape
    rules: list[dict] = []
    seen: set[tuple[tuple[int, int], tuple[int, int], tuple[tuple[int, int], ...]]] = set()

    for r in range(h):
        for c in range(w):
            source_color = int(grid[r, c])
            if source_color == int(bg):
                continue

            for dr, dc in _DIRS4:
                tr = r + dr
                tc = c + dc
                ar = tr + dr
                ac = tc + dc
                if not (0 <= tr < h and 0 <= tc < w and 0 <= ar < h and 0 <= ac < w):
                    continue

                target_color = int(grid[tr, tc])
                if target_color in (int(bg), source_color):
                    continue

                target_mask = grid == target_color
                target_mask[tr, tc] = False
                candidates: list[tuple[int, int, np.ndarray]] = []
                for backbone in components_mask(target_mask, connectivity=8):
                    dist = _anchor_distance(backbone, (ar, ac))
                    if dist <= 1:
                        candidates.append((int(dist), len(backbone), backbone))

                if not candidates:
                    continue

                candidates.sort(key=lambda item: (item[0], item[1]))
                _dist, _size, backbone = candidates[0]
                signature = _component_signature(backbone)
                key = ((int(r), int(c)), (int(tr), int(tc)), signature)
                if key in seen:
                    continue
                seen.add(key)

                template, template_cells = _extract_template(
                    grid,
                    backbone=backbone,
                    anchor=(ar, ac),
                    excluded={(int(r), int(c)), (int(tr), int(tc))},
                    bg=bg,
                )
                rules.append(
                    {
                        "source_color": int(source_color),
                        "target_color": int(target_color),
                        "source_pos": (int(r), int(c)),
                        "target_pos": (int(tr), int(tc)),
                        "anchor": (int(ar), int(ac)),
                        "backbone": np.asarray(backbone, dtype=int),
                        "template": template,
                        "template_cells": template_cells,
                    }
                )

    return rules


def main(grid: np.ndarray) -> np.ndarray:
    """
    Learn local source-target-template examples and stamp the recovered motifs.

    Input structure:
    - Black background with isolated colored source pixels and local rule legends.
    - Each legend is a 4-neighbor source->target pair whose forward direction
      reaches a target-colored backbone component; the same template may also
      include nearby accent cells of other colors.

    Rule:
    - For every other pixel of the legend's source color, translate that rule's
      entire nearby multicolor template so the legend anchor lands on the source
      pixel being replaced.
    - Target-colored backbone cells stay target-colored, and non-target accent
      cells keep exactly the colors shown in the legend.
    - Remove the legend pair and remove the original legend template itself.

    Key insight: the adjacent source/target cells still encode the color mapping,
    but the copied motif is hidden inside an ordinary target-colored blob plus
    nearby accents, so the legend target cell must be separated from the rest of
    that blob before the full multicolor template can be extracted.
    """
    g = np.asarray(grid, dtype=int)
    bg = get_bg(g)
    h, w = g.shape

    rules = _discover_rules(g, bg=bg)
    legend_pixels: set[tuple[int, int]] = set()
    template_pixels: set[tuple[int, int]] = set()
    source_colors: set[int] = set()
    for rule in rules:
        legend_pixels.add(rule["source_pos"])
        legend_pixels.add(rule["target_pos"])
        template_pixels.update(rule["template_cells"])
        source_colors.add(rule["source_color"])

    out = np.full_like(g, int(bg))

    for r, c in map(tuple, np.argwhere(g != int(bg))):
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
                    out[nr, nc] = int(color)

    return out


# --------------- generator (c) ---------------

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


def _reserve_4neighbors(reserved: set[tuple[int, int]], cells) -> None:
    for r, c in cells:
        for dr, dc in _DIRS4:
            reserved.add((int(r + dr), int(c + dc)))


def _reserve_8neighbors(reserved: set[tuple[int, int]], cells) -> None:
    for r, c in cells:
        for dr, dc in _DIRS8:
            reserved.add((int(r + dr), int(c + dc)))


def _make_backbone(min_pixels: int, max_pixels: int):
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
        backbone_coords, anchor = _make_backbone(min_pixels, max_pixels)
        backbone_set = set(backbone_coords)
        candidates: set[tuple[int, int]] = set()
        for r, c in backbone_coords:
            for dr in (-1, 1):
                for dc in (-1, 1):
                    p = (r + dr, c + dc)
                    if p not in backbone_set:
                        candidates.add(p)
        if not candidates:
            continue

        accent_goal = min(random.randint(ACCENT_MIN, ACCENT_MAX), len(candidates))
        shuffled = list(candidates)
        random.shuffle(shuffled)

        accents: list[tuple[int, int, int]] = []
        for pos in shuffled:
            if any(_manhattan(pos, cell) == 1 for cell in backbone_coords):
                continue
            if any(_manhattan(pos, (ar, ac)) == 1 for ar, ac, _ in accents):
                continue
            accents.append((int(pos[0]), int(pos[1]), int(random.choice(accent_pool))))
            if len(accents) >= accent_goal:
                break
        if not accents:
            continue

        ar, ac = anchor
        template = [(int(r - ar), int(c - ac), int(tgt_c)) for r, c in backbone_coords]
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
        src_c = colors[2 * i]
        tgt_c = colors[2 * i + 1]
        used_colors.update((src_c, tgt_c))
        rules.append(
            {
                "src": int(src_c),
                "tgt": int(tgt_c),
                "template": _make_multicolor_template(min_s, max_s, src_c, tgt_c),
                "n_sc": random.randint(SCATTERED_MIN, sc_max),
            }
        )
    return rules, used_colors


def _absolute_template_cells(rule: dict, anchor: tuple[int, int]):
    ar, ac = anchor
    cells: list[tuple[int, int, int]] = []
    for rr, cc, color in rule["template"]:
        cells.append((int(ar + rr), int(ac + cc), int(color)))
    return cells


def _legend_is_valid(
    sr: int,
    sc: int,
    tr: int,
    tc: int,
    template_cells: list[tuple[int, int, int]],
    target_color: int,
) -> bool:
    occupied = {(int(r), int(c)) for r, c, _ in template_cells}
    if (sr, sc) in occupied or (tr, tc) in occupied:
        return False
    target_cells = {(int(r), int(c)) for r, c, color in template_cells if int(color) == int(target_color)}
    for r, c, color in template_cells:
        if _manhattan((r, c), (sr, sc)) == 1:
            return False
        if int(color) != int(target_color) and _manhattan((r, c), (tr, tc)) == 1:
            return False
        if int(color) != int(target_color) and any(_manhattan((r, c), cell) == 1 for cell in target_cells):
            return False
    return True


def _place_legends(rules, inp, occupied, reserved):
    H, W = inp.shape
    for rule in rules:
        placed = False
        for _ in range(LEGEND_RETRIES):
            dr, dc = random.choice(_DIRS4)
            sr = random.randint(0, H - 1)
            sc = random.randint(0, W - 1)
            tr, tc = sr + dr, sc + dc
            ar, ac = sr + 2 * dr, sc + 2 * dc
            if not (0 <= tr < H and 0 <= tc < W and 0 <= ar < H and 0 <= ac < W):
                continue

            template_cells = _absolute_template_cells(rule, (ar, ac))
            if not template_cells:
                continue
            if any(not (0 <= r < H and 0 <= c < W) for r, c, _ in template_cells):
                continue
            if not _legend_is_valid(sr, sc, tr, tc, template_cells, rule["tgt"]):
                continue

            all_cells = {(int(sr), int(sc)), (int(tr), int(tc))}
            all_cells.update((int(r), int(c)) for r, c, _ in template_cells)
            if all_cells & occupied:
                continue
            if all_cells & reserved:
                continue

            inp[sr, sc] = rule["src"]
            inp[tr, tc] = rule["tgt"]
            for r, c, color in template_cells:
                inp[r, c] = color

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
        positions: list[tuple[int, int]] = []
        for _ in range(rule["n_sc"] * SCATTERED_RETRIES_PER_PIXEL):
            if len(positions) >= rule["n_sc"]:
                break
            r = random.randint(0, H - 1)
            c = random.randint(0, W - 1)
            if (r, c) in occupied or (r, c) in reserved:
                continue
            positions.append((int(r), int(c)))
            occupied.add((int(r), int(c)))
            _reserve_4neighbors(reserved, [(r, c)])
        if len(positions) < SCATTERED_MIN:
            return False
        positions.sort()
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
    placed: list[tuple[int, int, int]] = []
    for _ in range(n):
        for _ in range(BYSTANDER_RETRIES):
            r = random.randint(0, H - 1)
            c = random.randint(0, W - 1)
            if (r, c) in occupied or (r, c) in reserved:
                continue
            color = int(random.choice(bystander_colors))
            inp[r, c] = color
            placed.append((int(r), int(c), color))
            occupied.add((int(r), int(c)))
            _reserve_4neighbors(reserved, [(r, c)])
            break
    return placed


def _build_output(rules, bystander_pos, H: int, W: int):
    output = np.zeros((H, W), dtype=int)
    for r, c, color in bystander_pos:
        output[r, c] = int(color)
    for rule in sorted(rules, key=lambda item: item["source_pos"]):
        for sr, sc in sorted(rule["scattered"]):
            for rr, cc, color in rule["template"]:
                nr = int(sr + rr)
                nc = int(sc + cc)
                if 0 <= nr < H and 0 <= nc < W:
                    output[nr, nc] = int(color)
    return output


def _gen_attempt():
    num_rules = random.choice(RULE_COUNT_WEIGHTS)
    min_dim = MIN_DIM_BY_RULES[num_rules]
    H = random.randint(min_dim, GRID_DIM_MAX)
    W = random.randint(min_dim, GRID_DIM_MAX)

    rules, used_colors = _build_rules(num_rules, H * W)
    occupied: set[tuple[int, int]] = set()
    reserved: set[tuple[int, int]] = set()
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
    raise RuntimeError(f"Failed to generate valid input after {OUTER_RETRIES} attempts")
