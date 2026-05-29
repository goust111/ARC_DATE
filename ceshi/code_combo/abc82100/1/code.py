# Combo variant of abc82100 (index 1)
# combo_spec: D1/0+D3/0
# Name: 形状匹配+多像素色块源

from __future__ import annotations

import random

import numpy as np

from arc_utils import components_mask, find_components, get_bg


# Design analysis for the combo variant
# -------------------------------------
# a) Concrete behavior and trace
#    - Original task: a 1-cell legend source next to a 1-cell target color teaches
#      "every other pixel of this source color becomes this teal stencil recolored
#      to the target color".
#    - Combo task: the legend source is now a small 4-connected monochrome sprite.
#      Exactly one cell of that sprite touches the target singleton; this contact
#      cell becomes the alignment anchor for all other copies of the same shape.
#    - Scattered sources are also multi-cell sprites, but their colors are no longer
#      informative. Example:
#        old: blue pixel -> red pixel + teal shape => every other blue pixel stamps red
#        combo: green L-shape -> red pixel + teal shape => every other L-shape stamps red,
#        even when those L-shapes are yellow or maroon.
#    - The two modified dimensions interact multiplicatively: the solver must first
#      group pixels into connected components, then compare component geometry to the
#      legend sprite, then align the stencil using the corresponding contact cell.
#
# b) Complexity calibration
#    - Rule sprites use 3-5 cells and must span at least a 2x2 bounding box, so the
#      source objects visibly differ from the original single-pixel regime.
#    - Rule shapes are unique across one grid, preventing ambiguous shape->target
#      mappings.
#    - Scattered source colors come from a tiny shared palette and are reused across
#      different rule shapes. This makes color-only matching fail while remaining
#      visually understandable from 2-4 examples.
#
# c) Hidden assumptions from the original solver that break here
#    - Scanning `g == source_color` cell-by-cell no longer works: a source object is a
#      component, not a pixel.
#    - Using source color as the rule key no longer works: multiple rule shapes may
#      share the same scattered color.
#    - Preserving "all pixels whose color is not a source color" no longer works:
#      unrelated bystanders may legitimately share colors with source sprites.
#    - Therefore the new solver must operate component-by-component and decide consume
#      vs preserve from geometry and legend positions, not from color membership alone.
#
# d) Placement legality guarantees required from the generator
#    - Every legend source sprite has exactly one cell 4-adjacent to its target pixel.
#    - Rule legends, scattered sprites, and bystanders are separated by a 4-neighbor
#      moat, except for the intended legend source->target adjacency.
#    - Rule shapes are pairwise distinct after translation-normalization.
#    - Bystanders are chosen so their shapes do not match any rule sprite.
#
# e) Visual observability
#    - The source objects are visibly blobs, not isolated dots.
#    - The same scattered color is intentionally reused by multiple different rule
#      shapes, so humans can see that color alone is insufficient.
#    - Legend source colors often differ from scattered source colors, reinforcing
#      that the transferable feature is the sprite silhouette.
#
# f) Visual cleanliness
#    - Input objects never overlap and keep local padding, so legends and scattered
#      sprites remain separable by eye.
#    - Output stamps may overlap each other or clip at borders, matching the original
#      family, but bystanders are placed outside all predicted stamp footprints so
#      they stay clearly preserved.
#
# g) New solver assumptions and generator support
#    - Assumption: a legend target is a singleton colored component.
#      Guarantee: generator places target as an isolated pixel.
#    - Assumption: a matching scattered source has the same normalized shape and the
#      same relative contact cell as the legend source.
#      Guarantee: generator scatters translated copies of the exact legend shape only.
#    - Assumption: no unmatched component accidentally shares a rule shape.
#      Guarantee: bystanders are single pixels and rule shapes use area >= 3.


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

GRID_DIM_MAX = 20
MIN_DIM_BY_RULES = {1: 6, 2: 10, 3: 14, 4: 20}

RULE_COUNT_WEIGHTS = [1, 1, 2, 2, 3, 3, 4]
NON_BG_NON_TEAL_COLORS = [1, 2, 3, 4, 5, 6, 7, 9]

STENCIL_MAX_CAP = 12
STENCIL_MIN_AREA_DIV = 100
STENCIL_MAX_AREA_DIV = 25

SOURCE_MIN_CELLS = 3
SOURCE_MAX_CELLS = 5

SCATTERED_MIN = 3
SCATTERED_MAX_CAP = 6
SCATTERED_AREA_DIV = 45

BYSTANDER_MAX = 5

LEGEND_RETRIES = 500
SCATTERED_RETRIES_PER_OBJECT = 100
BYSTANDER_RETRIES = 80
OUTER_RETRIES = 600
SOURCE_SHAPE_RETRIES = 300


def _chebyshev(a: tuple[int, int], b: tuple[int, int]) -> int:
    return max(abs(int(a[0]) - int(b[0])), abs(int(a[1]) - int(b[1])))


def _anchor_distance(pattern: np.ndarray, anchor: tuple[int, int]) -> int:
    return min(_chebyshev((int(r), int(c)), anchor) for r, c in pattern)


def _normalize_coords(coords) -> tuple[tuple[int, int], ...]:
    coords = [(int(r), int(c)) for r, c in coords]
    r0 = min(r for r, _ in coords)
    c0 = min(c for _, c in coords)
    return tuple(sorted((r - r0, c - c0) for r, c in coords))


def _shape_info(coords: np.ndarray) -> dict:
    coords = np.asarray(coords, dtype=int)
    r0 = int(coords[:, 0].min())
    c0 = int(coords[:, 1].min())
    r1 = int(coords[:, 0].max())
    c1 = int(coords[:, 1].max())
    return {
        "bbox": (r0, c0, r1, c1),
        "rel": tuple(sorted((int(r - r0), int(c - c0)) for r, c in coords)),
        "coord_set": {(int(r), int(c)) for r, c in coords},
        "area": int(coords.shape[0]),
    }


def _colored_components(grid: np.ndarray, bg: int, pattern_color: int = 8) -> list[dict]:
    raw = find_components(grid, bg=bg, connectivity=4, monochromatic=True, return_format="dict")
    comps: list[dict] = []
    for comp in raw:
        color = int(comp["color"])
        if color in (int(bg), int(pattern_color)):
            continue
        coords = np.asarray(comp["coords"], dtype=int)
        info = _shape_info(coords)
        comps.append(
            {
                "index": len(comps),
                "color": color,
                "coords": coords,
                "bbox": info["bbox"],
                "rel": info["rel"],
                "coord_set": info["coord_set"],
                "area": info["area"],
            }
        )
    return comps


def _discover_rules(grid: np.ndarray, bg: int, pattern_color: int = 8) -> list[dict]:
    h, w = grid.shape
    patterns = components_mask(grid == pattern_color, connectivity=8)
    comps = _colored_components(grid, bg=bg, pattern_color=pattern_color)
    cell_to_comp: dict[tuple[int, int], int] = {}
    for comp in comps:
        for r, c in comp["coord_set"]:
            cell_to_comp[(int(r), int(c))] = int(comp["index"])

    rules: list[dict] = []
    seen: set[tuple[int, tuple[int, int], int]] = set()

    for comp in comps:
        if comp["area"] < 2:
            continue
        r0, c0, _r1, _c1 = comp["bbox"]
        for r, c in map(tuple, comp["coords"]):
            r = int(r)
            c = int(c)
            for dr, dc in _DIRS4:
                tr = r + dr
                tc = c + dc
                if not (0 <= tr < h and 0 <= tc < w):
                    continue

                target_color = int(grid[tr, tc])
                if target_color in (int(bg), int(pattern_color), int(comp["color"])):
                    continue

                target_comp_index = cell_to_comp.get((int(tr), int(tc)))
                if target_comp_index is None or comps[target_comp_index]["area"] != 1:
                    continue

                adjacency_count = sum(1 for rr, cc in comp["coord_set"] if abs(rr - tr) + abs(cc - tc) == 1)
                if adjacency_count != 1:
                    continue

                anchor = (tr + dr, tc + dc)
                candidates: list[tuple[int, int, np.ndarray]] = []
                for pattern_index, pattern in enumerate(patterns):
                    dist = _anchor_distance(pattern, anchor)
                    if dist <= 1:
                        candidates.append((int(dist), int(pattern_index), pattern))
                if not candidates:
                    continue

                candidates.sort(key=lambda item: (item[0], len(item[2])))
                _dist, pattern_index, pattern = candidates[0]
                key = (int(comp["index"]), (int(tr), int(tc)), int(pattern_index))
                if key in seen:
                    continue
                seen.add(key)
                rules.append(
                    {
                        "source_comp_index": int(comp["index"]),
                        "source_shape": comp["rel"],
                        "source_contact_rel": (int(r - r0), int(c - c0)),
                        "target_color": int(target_color),
                        "target_pos": (int(tr), int(tc)),
                        "anchor": (int(anchor[0]), int(anchor[1])),
                        "pattern": np.asarray(pattern, dtype=int),
                    }
                )

    return rules


def main(grid: np.ndarray) -> np.ndarray:
    """
    Learn local sprite-to-stencil rules, then apply them to matching source blobs.

    Input structure:
    - Black background with small colored sprite components, isolated colored target
      pixels, and teal (8) stencil components.
    - A rule example is a multi-cell source sprite whose unique boundary contact cell
      touches a target pixel; extending that direction reaches an anchor touching the
      rule's teal stencil.

    Rule:
    - Extract every non-background, non-teal connected component and identify which
      ones are legend sources by their adjacency to a singleton target pixel and a
      nearby teal stencil.
    - For each other component, compare its normalized shape to the learned legend
      shapes. When there is exactly one match, align that rule's teal stencil so its
      anchor lands on the corresponding contact cell of the component, then recolor
      the stencil to the legend target color.
    - Remove teal pixels and all legend cells; keep only unmatched bystanders.

    Key insight: Neither color nor individual pixels identify the replaceable objects
    anymore. The transferable invariant is a whole component's geometry plus the one
    relative cell that faces the target pixel in the legend.
    """
    g = np.asarray(grid, dtype=int)
    bg = get_bg(g)
    pattern_color = 8
    h, w = g.shape

    rules = _discover_rules(g, bg=bg, pattern_color=pattern_color)
    comps = _colored_components(g, bg=bg, pattern_color=pattern_color)

    legend_source_indices = {int(rule["source_comp_index"]) for rule in rules}
    legend_target_positions = {tuple(rule["target_pos"]) for rule in rules}
    shape_to_rules: dict[tuple[tuple[int, int], ...], list[dict]] = {}
    for rule in rules:
        shape_to_rules.setdefault(rule["source_shape"], []).append(rule)

    out = np.full_like(g, int(bg))

    for comp in comps:
        if comp["index"] in legend_source_indices:
            continue

        if comp["area"] == 1:
            only_cell = next(iter(comp["coord_set"]))
            if only_cell in legend_target_positions:
                continue

        candidates = shape_to_rules.get(comp["rel"], [])
        if len(candidates) == 1:
            rule = candidates[0]
            cr0, cc0, _cr1, _cc1 = comp["bbox"]
            contact_r = int(cr0 + rule["source_contact_rel"][0])
            contact_c = int(cc0 + rule["source_contact_rel"][1])
            ar, ac = rule["anchor"]
            rows = rule["pattern"][:, 0] + (contact_r - int(ar))
            cols = rule["pattern"][:, 1] + (contact_c - int(ac))
            ok = (0 <= rows) & (rows < h) & (0 <= cols) & (cols < w)
            out[rows[ok], cols[ok]] = int(rule["target_color"])
            continue

        for r, c in comp["coord_set"]:
            out[int(r), int(c)] = int(comp["color"])

    return out


def _reserve_4neighbors(reserved: set[tuple[int, int]], cells) -> None:
    for r, c in cells:
        for dr, dc in _DIRS4:
            reserved.add((int(r + dr), int(c + dc)))


def _translate(rel, top: int, left: int) -> set[tuple[int, int]]:
    return {(int(top + r), int(left + c)) for r, c in rel}


def _make_stencil(min_pixels: int, max_pixels: int) -> tuple[tuple[tuple[int, int], ...], tuple[int, int]]:
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
    anchor = random.choice(coords_list)
    rel = tuple(sorted((r - anchor[0], c - anchor[1]) for r, c in coords_list))
    return rel, anchor


def _valid_contacts(rel: tuple[tuple[int, int], ...]) -> list[tuple[tuple[int, int], tuple[int, int]]]:
    rel_set = set(rel)
    out: list[tuple[tuple[int, int], tuple[int, int]]] = []
    for r, c in rel:
        for dr, dc in _DIRS4:
            target = (r + dr, c + dc)
            if target in rel_set:
                continue
            adjacency_count = sum(1 for rr, cc in rel_set if abs(rr - target[0]) + abs(cc - target[1]) == 1)
            if adjacency_count == 1:
                out.append(((int(r), int(c)), (int(dr), int(dc))))
    return out


def _make_source_shape(used_shapes: set[tuple[tuple[int, int], ...]]):
    for _ in range(SOURCE_SHAPE_RETRIES):
        n = random.randint(SOURCE_MIN_CELLS, SOURCE_MAX_CELLS)
        coords = {(0, 0)}
        for _ in range(n - 1):
            frontier = set()
            for r, c in coords:
                for dr, dc in _DIRS4:
                    p = (r + dr, c + dc)
                    if p not in coords:
                        frontier.add(p)
            coords.add(random.choice(list(frontier)))

        rel = _normalize_coords(coords)
        rows = {r for r, _ in rel}
        cols = {c for _, c in rel}
        if len(rows) < 2 or len(cols) < 2:
            continue
        if rel in used_shapes:
            continue

        contacts = _valid_contacts(rel)
        if not contacts:
            continue

        used_shapes.add(rel)
        contact_rel, direction = random.choice(contacts)
        return rel, contact_rel, direction

    raise RuntimeError("failed to generate a distinct source sprite shape")


def _build_rules(num_rules: int, area: int):
    colors = NON_BG_NON_TEAL_COLORS[:]
    random.shuffle(colors)

    max_stencil = min(STENCIL_MAX_CAP, max(2, area // STENCIL_MAX_AREA_DIV))
    min_stencil = min(max(1, area // STENCIL_MIN_AREA_DIV), max_stencil)
    max_instances = min(SCATTERED_MAX_CAP, max(SCATTERED_MIN, area // SCATTERED_AREA_DIV))
    max_instances = min(max_instances, max(SCATTERED_MIN, 7 - max(0, num_rules - 1)))

    used_shapes: set[tuple[tuple[int, int], ...]] = set()
    rules = []
    used_colors = set()

    legend_sources = [int(colors[2 * i]) for i in range(num_rules)]
    dominant_candidates = [c for c in NON_BG_NON_TEAL_COLORS if c not in legend_sources]
    dominant_scatter_color = int(random.choice(dominant_candidates or NON_BG_NON_TEAL_COLORS))
    alternate_pool = [c for c in NON_BG_NON_TEAL_COLORS if c != dominant_scatter_color]

    for i in range(num_rules):
        legend_src = int(colors[2 * i])
        target = int(colors[2 * i + 1])
        used_colors.update((legend_src, target))
        source_rel, contact_rel, direction = _make_source_shape(used_shapes)
        teal_rel, _anchor = _make_stencil(min_stencil, max_stencil)

        scatter_palette = [dominant_scatter_color]
        if alternate_pool:
            scatter_palette.append(int(random.choice(alternate_pool)))
        rules.append(
            {
                "legend_src": legend_src,
                "target": target,
                "source_rel": source_rel,
                "contact_rel": contact_rel,
                "direction": direction,
                "teal_rel": teal_rel,
                "n_scattered": random.randint(SCATTERED_MIN, max_instances),
                "scatter_palette": tuple(scatter_palette),
            }
        )

    return rules, used_colors, dominant_scatter_color


def _place_legends(rules, inp: np.ndarray, occupied: set, reserved: set) -> bool:
    h, w = inp.shape
    for rule in rules:
        placed = False
        for _ in range(LEGEND_RETRIES):
            top = random.randint(0, h - 1)
            left = random.randint(0, w - 1)
            source_cells = _translate(rule["source_rel"], top, left)
            if not source_cells:
                continue
            if min(r for r, _ in source_cells) < 0 or max(r for r, _ in source_cells) >= h:
                continue
            if min(c for _, c in source_cells) < 0 or max(c for _, c in source_cells) >= w:
                continue

            contact_r = int(top + rule["contact_rel"][0])
            contact_c = int(left + rule["contact_rel"][1])
            dr, dc = rule["direction"]
            target = (int(contact_r + dr), int(contact_c + dc))
            anchor = (int(target[0] + dr), int(target[1] + dc))
            if not (0 <= target[0] < h and 0 <= target[1] < w and 0 <= anchor[0] < h and 0 <= anchor[1] < w):
                continue

            teal_cells = _translate(rule["teal_rel"], anchor[0], anchor[1])
            if min(r for r, _ in teal_cells) < 0 or max(r for r, _ in teal_cells) >= h:
                continue
            if min(c for _, c in teal_cells) < 0 or max(c for _, c in teal_cells) >= w:
                continue

            all_cells = set(source_cells) | {target} | set(teal_cells)
            if all_cells & (occupied | reserved):
                continue

            for r, c in source_cells:
                inp[r, c] = int(rule["legend_src"])
            inp[target[0], target[1]] = int(rule["target"])
            for r, c in teal_cells:
                inp[r, c] = 8

            occupied.update(all_cells)
            _reserve_4neighbors(reserved, all_cells)
            rule["legend_top_left"] = (int(top), int(left))
            rule["target_pos"] = target
            rule["anchor"] = anchor
            placed = True
            break

        if not placed:
            return False

    return True


def _choose_scatter_color(rule: dict, num_rules: int, index: int) -> int:
    palette = list(rule["scatter_palette"])
    if num_rules == 1:
        color = int(palette[index % len(palette)])
        if color == int(rule["legend_src"]) and len(palette) > 1:
            return int(palette[(index + 1) % len(palette)])
        return color
    return int(palette[0])


def _place_scattered(rules, inp: np.ndarray, occupied: set, reserved: set) -> bool:
    h, w = inp.shape
    num_rules = len(rules)
    for rule in rules:
        objects = []
        target_count = int(rule["n_scattered"])
        for attempt in range(target_count * SCATTERED_RETRIES_PER_OBJECT):
            if len(objects) >= target_count:
                break
            top = random.randint(0, h - 1)
            left = random.randint(0, w - 1)
            cells = _translate(rule["source_rel"], top, left)
            if min(r for r, _ in cells) < 0 or max(r for r, _ in cells) >= h:
                continue
            if min(c for _, c in cells) < 0 or max(c for _, c in cells) >= w:
                continue
            if cells & (occupied | reserved):
                continue

            color = _choose_scatter_color(rule, num_rules, len(objects))
            for r, c in cells:
                inp[r, c] = color
            occupied.update(cells)
            _reserve_4neighbors(reserved, cells)
            objects.append({"top_left": (int(top), int(left)), "color": int(color)})

        if len(objects) < SCATTERED_MIN:
            return False
        rule["scattered"] = objects

    return True


def _stamp_cells(rule: dict, obj: dict, h: int, w: int) -> set[tuple[int, int]]:
    top, left = obj["top_left"]
    contact_r = int(top + rule["contact_rel"][0])
    contact_c = int(left + rule["contact_rel"][1])
    cells = set()
    for rr, cc in rule["teal_rel"]:
        nr = int(contact_r + rr)
        nc = int(contact_c + cc)
        if 0 <= nr < h and 0 <= nc < w:
            cells.add((nr, nc))
    return cells


def _place_bystanders(rules, used_colors, inp: np.ndarray, occupied: set, reserved: set):
    h, w = inp.shape
    forbidden = set()
    for rule in rules:
        for obj in rule.get("scattered", []):
            forbidden.update(_stamp_cells(rule, obj, h, w))

    bystander_colors = NON_BG_NON_TEAL_COLORS[:]
    n = random.randint(0, BYSTANDER_MAX)
    placed = []
    for _ in range(n):
        for _ in range(BYSTANDER_RETRIES):
            r = random.randint(0, h - 1)
            c = random.randint(0, w - 1)
            if (r, c) in occupied or (r, c) in reserved or (r, c) in forbidden:
                continue
            color = int(random.choice(bystander_colors))
            inp[r, c] = color
            placed.append((int(r), int(c), color))
            occupied.add((int(r), int(c)))
            _reserve_4neighbors(reserved, [(int(r), int(c))])
            break
    return placed


def _build_output(rules, bystander_pos, h: int, w: int) -> np.ndarray:
    out = np.zeros((h, w), dtype=int)
    for r, c, color in bystander_pos:
        out[int(r), int(c)] = int(color)

    for rule in rules:
        for obj in rule["scattered"]:
            for r, c in _stamp_cells(rule, obj, h, w):
                out[int(r), int(c)] = int(rule["target"])

    return out


def _gen_attempt():
    num_rules = random.choice(RULE_COUNT_WEIGHTS)
    min_dim = MIN_DIM_BY_RULES[num_rules]
    h = random.randint(min_dim, GRID_DIM_MAX)
    w = random.randint(min_dim, GRID_DIM_MAX)

    rules, used_colors, _dominant_scatter_color = _build_rules(num_rules, h * w)
    occupied: set[tuple[int, int]] = set()
    reserved: set[tuple[int, int]] = set()
    inp = np.zeros((h, w), dtype=int)

    if not _place_legends(rules, inp, occupied, reserved):
        return None
    if not _place_scattered(rules, inp, occupied, reserved):
        return None
    bystander_pos = _place_bystanders(rules, used_colors, inp, occupied, reserved)

    expected = _build_output(rules, bystander_pos, h, w)
    try:
        if np.array_equal(main(inp), expected):
            return inp
    except Exception:
        pass
    return None


def generate_input() -> np.ndarray:
    for _ in range(OUTER_RETRIES):
        grid = _gen_attempt()
        if grid is not None:
            return grid
    raise RuntimeError(f"Failed to generate valid input after {OUTER_RETRIES} attempts")
