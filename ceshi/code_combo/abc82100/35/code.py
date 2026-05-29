# Combo variant of abc82100 (index 35)
# combo_spec: D7/3+D13/2
# Name: 恒等映射+规则区域特殊标记

from __future__ import annotations

import random

import numpy as np

from arc_utils import components_mask, get_bg


# Design analysis for combo D7/3 + D13/2
# =======================================
# a) Concreteization and behavior trace
#    - Baseline abc82100:
#      1. A legend is a local source pixel -> adjacent target pixel -> nearby teal stencil.
#      2. Every scattered pixel of the same source color is replaced by a translated copy
#         of that stencil recolored to the target color.
#      3. The whole legend area disappears to background in the output.
#    - Combo behavior:
#      1. Some legends are identity mappings, e.g. red -> red. They still define a real
#         stencil placement rule; the scattered red singletons expand into red shapes.
#      2. The legend region itself no longer vanishes. Instead, the bounding box covering
#         the legend pair and its teal stencil becomes a solid grey marker block.
#      3. Example trace: suppose a rule is red -> red with a 3-cell L-shaped stencil.
#         In the input we see "red red teal-L". In the output the legend zone becomes a
#         grey rectangle, while each scattered red source becomes a red L-shape. Color did
#         not change, but shape did. The grey legend marker proves that this region was a
#         consumed rule rather than ordinary background.
#    - Difference from baseline:
#      1. `source == target` is now valid and must still trigger stamping.
#      2. Rule cleanup is no longer "erase to background"; it becomes a visibly marked area.
#
# b) Complexity calibration
#    - At least one identity rule is mandatory in every generated sample.
#    - If multiple rules exist, at least one non-identity rule is also kept so the solver
#      sees both "changed color" and "same color but expanded shape" evidence.
#    - Identity stencils always have at least 2 cells and are placed fully in bounds. This
#      avoids degenerate cases where stamping a single cell would look identical to doing
#      nothing.
#    - Legend regions are filled as bbox blocks rather than sparse pixel leftovers. That
#      makes D7/3 visually obvious without adding another symbolic sub-language.
#
# c) Hidden-assumption review of the original main()
#    - Original `_discover_rules()` rejected `target_color == source_color`. That assumption
#      is broken by D13/2, so the filter must be removed.
#    - Original output started from background and only restored bystanders because legend
#      pixels and teal templates were supposed to disappear. Under D7/3, we still start from
#      background, but after stamping we explicitly paint each legend bbox with grey.
#    - Original preservation logic skipped any pixel whose color was one of the learned
#      source colors. That remains valid only if generator never uses source colors for
#      bystanders. The combo generator enforces that.
#    - Original stamping loop read from the immutable input grid, which is still correct here.
#      It matters more now because an identity rule may stamp the same color it reads.
#
# d) Placement legality
#    - Legends are placed with one unique teal component each, full in bounds, and with their
#      source/target cells outside the stencil itself.
#    - Generator reserves every legend bbox plus a one-cell margin, so scattered sources and
#      bystanders never appear inside the future grey marker zones.
#    - Scattered source pixels are isolated in the input and their full future stamp footprint
#      is also reserved in the output. This prevents accidental overlaps with grey legend
#      zones, bystanders, or other stamps.
#
# e) Visual observability
#    - A human sees that the rule zones all become the same grey rectangles, distinguishing
#      "consumed rule area" from ordinary black background.
#    - Identity rules remain inferable because several same-colored scattered source pixels
#      grow into repeated multi-cell shapes even though their color stays unchanged.
#
# f) Visual cleanliness
#    - Full in-bounds, non-overlapping stamps keep the output legible.
#    - Grey legend bboxes are separated from stamp footprints, so the two signals do not mix.
#
# g) New solver assumptions and how generator satisfies them
#    - Assumption: each usable legend has exactly one nearest teal component within the
#      anchor-distance threshold. Guaranteed by wide spacing between legends.
#    - Assumption: source colors are unique across rules. Guaranteed by construction.
#    - Assumption: no unrelated pixel uses a source color. Guaranteed by the bystander palette.
#    - Assumption: legend bboxes do not overlap. Guaranteed by bbox+margin reservation.


PATTERN_COLOR = 8
MARKER_COLOR = 5
ACTIVE_COLORS = [1, 2, 3, 4, 6, 7, 9]

GRID_DIM_MAX = 20
MIN_DIM_BY_RULES = {1: 7, 2: 10, 3: 14, 4: 18}

RULE_COUNT_WEIGHTS = [1, 1, 2, 2, 3, 3, 4]

STENCIL_MAX_CAP = 8
STENCIL_MIN_AREA_DIV = 120
STENCIL_MAX_AREA_DIV = 35

SCATTERED_MIN = 3
SCATTERED_MAX_CAP = 6
SCATTERED_AREA_DIV = 45

BYSTANDER_MAX = 4

LEGEND_RETRIES = 700
SCATTERED_RETRIES_PER_PIXEL = 120
BYSTANDER_RETRIES = 80
OUTER_RETRIES = 500

_DIRS4 = [(-1, 0), (1, 0), (0, -1), (0, 1)]
_DIRS8 = [
    (-1, -1), (-1, 0), (-1, 1),
    (0, -1),           (0, 1),
    (1, -1),  (1, 0),  (1, 1),
]


def _chebyshev(a: tuple[int, int], b: tuple[int, int]) -> int:
    return max(abs(int(a[0]) - int(b[0])), abs(int(a[1]) - int(b[1])))


def _anchor_distance(pattern: np.ndarray, anchor: tuple[int, int]) -> int:
    return min(_chebyshev((int(r), int(c)), anchor) for r, c in pattern)


def _bbox_from_cells(cells: set[tuple[int, int]]) -> tuple[int, int, int, int]:
    rows = [int(r) for r, _ in cells]
    cols = [int(c) for _, c in cells]
    return min(rows), min(cols), max(rows), max(cols)


def _bbox_cells(bbox: tuple[int, int, int, int], pad: int = 0) -> set[tuple[int, int]]:
    r0, c0, r1, c1 = bbox
    return {
        (r, c)
        for r in range(int(r0) - int(pad), int(r1) + int(pad) + 1)
        for c in range(int(c0) - int(pad), int(c1) + int(pad) + 1)
    }


def _translated_cells(base_r: int, base_c: int, rel) -> set[tuple[int, int]]:
    return {(int(base_r) + int(rr), int(base_c) + int(cc)) for rr, cc in rel}


def _all_in_bounds(shape: tuple[int, int], cells: set[tuple[int, int]]) -> bool:
    h, w = map(int, shape)
    return all(0 <= int(r) < h and 0 <= int(c) < w for r, c in cells)


def _reserve_4neighbors(reserved: set[tuple[int, int]], cells) -> None:
    for r, c in cells:
        r = int(r)
        c = int(c)
        for dr, dc in _DIRS4:
            reserved.add((r + dr, c + dc))


def _make_stencil(min_pixels: int, max_pixels: int) -> tuple[list[tuple[int, int]], tuple[int, int]]:
    n = random.randint(int(min_pixels), int(max_pixels))
    coords = {(0, 0)}
    for _ in range(n - 1):
        frontier = set()
        for r, c in coords:
            for dr, dc in _DIRS8:
                p = (int(r + dr), int(c + dc))
                if p not in coords:
                    frontier.add(p)
        coords.add(random.choice(list(frontier)))
    coords_list = sorted(coords)
    return coords_list, random.choice(coords_list)


def _discover_rules(grid: np.ndarray, bg: int, pattern_color: int = PATTERN_COLOR) -> list[dict]:
    h, w = grid.shape
    patterns = components_mask(grid == int(pattern_color), connectivity=8)
    rules: list[dict] = []
    seen: set[tuple[tuple[int, int], tuple[int, int], int]] = set()

    for r in range(h):
        for c in range(w):
            source_color = int(grid[r, c])
            if source_color in (int(bg), int(pattern_color)):
                continue

            for dr, dc in _DIRS4:
                tr = int(r + dr)
                tc = int(c + dc)
                if not (0 <= tr < h and 0 <= tc < w):
                    continue

                target_color = int(grid[tr, tc])
                if target_color in (int(bg), int(pattern_color)):
                    continue

                anchor = (int(tr + dr), int(tc + dc))
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

                legend_cells = {(int(r), int(c)), (int(tr), int(tc)), tuple(anchor)}
                legend_cells.update((int(pr), int(pc)) for pr, pc in np.asarray(pattern, dtype=int))
                rules.append(
                    {
                        "source_color": int(source_color),
                        "target_color": int(target_color),
                        "source_pos": (int(r), int(c)),
                        "target_pos": (int(tr), int(tc)),
                        "anchor": (int(anchor[0]), int(anchor[1])),
                        "pattern": np.asarray(pattern, dtype=int),
                        "legend_bbox": _bbox_from_cells(legend_cells),
                    }
                )

    return rules


def main(grid: np.ndarray) -> np.ndarray:
    """
    Learn local color-pair legends with nearby teal stencils, then stamp those stencils elsewhere.

    Input structure:
    - The grid is mostly background, with isolated colored pixels and several teal (8)
      stencil components.
    - Each usable legend is a 4-neighbor source-target pair whose forward continuation
      reaches an anchor touching one teal stencil.
    - Some legends are identity mappings: source and target may have the same color.

    Rule:
    - For every non-legend pixel whose color matches a legend's source color, translate
      that legend's teal stencil so the legend anchor lands on the source pixel, then paint
      the stencil in the legend's target color.
    - After all stamping, replace each legend's whole rectangular rule region with a fixed
      grey marker color instead of erasing it to background.
    - Preserve unrelated bystander pixels that do not use any learned source color.

    Key insight: the adjacent colored pair still tells you which stencil to reuse, but now
    "same color" legends are valid rules too, and the output marks consumed legend areas as
    grey blocks so a rule can act even when its color mapping looks unchanged.
    """
    g = np.asarray(grid, dtype=int)
    bg = get_bg(g)
    h, w = g.shape

    rules = _discover_rules(g, bg=bg, pattern_color=PATTERN_COLOR)
    legend_pixels: set[tuple[int, int]] = set()
    source_colors: set[int] = set()
    legend_regions: list[set[tuple[int, int]]] = []
    source_positions: dict[int, list[tuple[int, int]]] = {}

    for rule in rules:
        legend_pixels.add(tuple(rule["source_pos"]))
        legend_pixels.add(tuple(rule["target_pos"]))
        source_color = int(rule["source_color"])
        source_colors.add(source_color)
        source_positions.setdefault(source_color, [])
        legend_regions.append(_bbox_cells(rule["legend_bbox"]))

    for r, c in map(tuple, np.argwhere((g != int(bg)) & (g != int(PATTERN_COLOR)))):
        r = int(r)
        c = int(c)
        color = int(g[r, c])
        if (r, c) in legend_pixels or color not in source_positions:
            continue
        source_positions[color].append((r, c))

    out = np.full_like(g, int(bg))

    # Keep bystanders and any other non-source colored pixels.
    for r, c in map(tuple, np.argwhere((g != int(bg)) & (g != int(PATTERN_COLOR)))):
        r = int(r)
        c = int(c)
        color = int(g[r, c])
        if (r, c) in legend_pixels or color in source_colors:
            continue
        out[r, c] = color

    for rule in rules:
        ar, ac = rule["anchor"]
        pattern = rule["pattern"]
        target_color = int(rule["target_color"])
        for sr, sc in source_positions.get(int(rule["source_color"]), []):
            dr = int(sr) - int(ar)
            dc = int(sc) - int(ac)
            rows = pattern[:, 0] + dr
            cols = pattern[:, 1] + dc
            ok = (0 <= rows) & (rows < h) & (0 <= cols) & (cols < w)
            out[rows[ok], cols[ok]] = target_color

    for region in legend_regions:
        for r, c in region:
            if 0 <= int(r) < h and 0 <= int(c) < w:
                out[int(r), int(c)] = int(MARKER_COLOR)

    return out


def _build_rules(num_rules: int, area: int) -> tuple[list[dict], set[int]]:
    source_colors = random.sample(ACTIVE_COLORS, int(num_rules))
    identity_indices = {random.randrange(int(num_rules))}
    if num_rules >= 3 and random.random() < 0.35:
        extra = random.choice([i for i in range(num_rules) if i not in identity_indices])
        identity_indices.add(int(extra))

    max_s = min(STENCIL_MAX_CAP, max(3, area // STENCIL_MAX_AREA_DIV))
    min_s = min(max(2, area // STENCIL_MIN_AREA_DIV), max_s)
    sc_max = min(SCATTERED_MAX_CAP, max(SCATTERED_MIN, area // SCATTERED_AREA_DIV))

    rules: list[dict] = []
    used_colors = set()
    for i, src in enumerate(source_colors):
        is_identity = int(i) in identity_indices
        if is_identity:
            tgt = int(src)
        else:
            tgt = int(random.choice([c for c in ACTIVE_COLORS if c != int(src)]))
        stencil, anchor = _make_stencil(min_s, max_s)
        rel = [(int(r - anchor[0]), int(c - anchor[1])) for r, c in stencil]
        rules.append(
            {
                "src": int(src),
                "tgt": int(tgt),
                "identity": bool(is_identity),
                "rel": rel,
                "n_sc": random.randint(SCATTERED_MIN, sc_max),
            }
        )
        used_colors.add(int(src))
        used_colors.add(int(tgt))

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
            sr = random.randint(0, h - 1)
            sc = random.randint(0, w - 1)
            tr = int(sr + dr)
            tc = int(sc + dc)
            ar = int(sr + 2 * dr)
            ac = int(sc + 2 * dc)
            if not (0 <= tr < h and 0 <= tc < w and 0 <= ar < h and 0 <= ac < w):
                continue

            teal_cells = _translated_cells(ar, ac, rel)
            if not _all_in_bounds((h, w), teal_cells):
                continue
            if (sr, sc) in teal_cells or (tr, tc) in teal_cells:
                continue

            all_cells = {(sr, sc), (tr, tc)} | teal_cells
            bbox = _bbox_from_cells(all_cells)
            region = _bbox_cells(bbox)
            guard = _bbox_cells(bbox, pad=1)
            if all_cells & occupied:
                continue
            if region & occupied:
                continue
            if guard & reserved:
                continue

            inp[sr, sc] = int(rule["src"])
            inp[tr, tc] = int(rule["tgt"])
            for r, c in teal_cells:
                inp[int(r), int(c)] = int(PATTERN_COLOR)

            occupied.update(all_cells)
            reserved.update(guard)
            rule["source_pos"] = (int(sr), int(sc))
            rule["target_pos"] = (int(tr), int(tc))
            rule["anchor"] = (int(ar), int(ac))
            rule["legend_bbox"] = bbox
            rule["legend_region"] = region
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
    output_reserved: set[tuple[int, int]],
) -> bool:
    h, w = inp.shape
    for rule in rules:
        positions: list[tuple[int, int]] = []
        attempts = int(rule["n_sc"]) * SCATTERED_RETRIES_PER_PIXEL
        for _ in range(attempts):
            if len(positions) >= int(rule["n_sc"]):
                break
            r = random.randint(0, h - 1)
            c = random.randint(0, w - 1)
            if (r, c) in occupied or (r, c) in reserved:
                continue

            stamp_cells = _translated_cells(r, c, rule["rel"])
            if not _all_in_bounds((h, w), stamp_cells):
                continue
            if stamp_cells & output_reserved:
                continue

            positions.append((int(r), int(c)))
            occupied.add((int(r), int(c)))
            _reserve_4neighbors(reserved, [(int(r), int(c))])
            output_reserved.update(stamp_cells)
            inp[int(r), int(c)] = int(rule["src"])

        if len(positions) < SCATTERED_MIN:
            return False
        rule["scattered"] = positions
    return True


def _place_bystanders(
    used_colors: set[int],
    inp: np.ndarray,
    occupied: set[tuple[int, int]],
    reserved: set[tuple[int, int]],
    output_reserved: set[tuple[int, int]],
) -> list[tuple[int, int, int]]:
    bystander_colors = [c for c in ACTIVE_COLORS if c not in used_colors]
    if not bystander_colors:
        return []

    h, w = inp.shape
    placed: list[tuple[int, int, int]] = []
    n = random.randint(0, min(BYSTANDER_MAX, len(bystander_colors) * 2))
    for _ in range(n):
        for _ in range(BYSTANDER_RETRIES):
            r = random.randint(0, h - 1)
            c = random.randint(0, w - 1)
            if (r, c) in occupied or (r, c) in reserved or (r, c) in output_reserved:
                continue
            color = int(random.choice(bystander_colors))
            inp[int(r), int(c)] = color
            placed.append((int(r), int(c), color))
            occupied.add((int(r), int(c)))
            output_reserved.add((int(r), int(c)))
            _reserve_4neighbors(reserved, [(int(r), int(c))])
            break
    return placed


def _build_output(rules: list[dict], bystanders: list[tuple[int, int, int]], h: int, w: int) -> np.ndarray:
    out = np.zeros((h, w), dtype=int)
    for r, c, color in bystanders:
        out[int(r), int(c)] = int(color)
    for rule in rules:
        for sr, sc in rule["scattered"]:
            for rr, cc in rule["rel"]:
                out[int(sr + rr), int(sc + cc)] = int(rule["tgt"])
    for rule in rules:
        r0, c0, r1, c1 = rule["legend_bbox"]
        out[int(r0):int(r1) + 1, int(c0):int(c1) + 1] = int(MARKER_COLOR)
    return out


def _gen_attempt() -> np.ndarray | None:
    num_rules = random.choice(RULE_COUNT_WEIGHTS)
    min_dim = MIN_DIM_BY_RULES[int(num_rules)]
    h = random.randint(min_dim, GRID_DIM_MAX)
    w = random.randint(min_dim, GRID_DIM_MAX)

    rules, used_colors = _build_rules(int(num_rules), h * w)
    occupied: set[tuple[int, int]] = set()
    reserved: set[tuple[int, int]] = set()
    inp = np.zeros((h, w), dtype=int)

    if not _place_legends(rules, inp, occupied, reserved):
        return None

    output_reserved: set[tuple[int, int]] = set()
    for rule in rules:
        output_reserved.update(rule["legend_region"])

    if not _place_scattered(rules, inp, occupied, reserved, output_reserved):
        return None

    bystanders = _place_bystanders(used_colors, inp, occupied, reserved, output_reserved)
    output = _build_output(rules, bystanders, h, w)

    identity_rules = [rule for rule in rules if rule["identity"]]
    if not identity_rules:
        return None
    if not any(len(rule["rel"]) >= 2 and rule["scattered"] for rule in identity_rules):
        return None
    if not np.any(output == int(MARKER_COLOR)):
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
    raise RuntimeError(f"Failed to generate valid combo sample after {OUTER_RETRIES} attempts")
