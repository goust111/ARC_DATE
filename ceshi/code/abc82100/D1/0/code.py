# Variant of abc82100
# Dimension: D1 — 源像素与规则的匹配条件 (match_cond)
# Change: s.color == d.source_color：源像素的颜色必须等于规则示例中源色像素的颜色... → 形状匹配而非颜色匹配

from __future__ import annotations

import random

import numpy as np

from arc_utils import components_mask, find_components, get_bg


# Design analysis for the D1 variant
# ----------------------------------
# a) Concrete behavior and trace
#    - Each rule legend is still local and linear: source sprite -> target pixel -> teal stencil anchor.
#    - The old source was a single pixel, so the legend key was just its color.
#    - The new source is a small 4-connected monochrome sprite. One boundary cell of that
#      sprite is the unique cell touching the target pixel; this becomes the sprite's
#      "contact cell". The same relative cell inside a scattered copy is where the teal
#      stencil must be aligned.
#    - Small trace:
#        old: a blue pixel next to a red pixel means "every other blue pixel becomes a red stencil".
#        new: an L-shaped sprite next to a red pixel means "every other L-shaped sprite becomes
#             a red stencil", even if those scattered L-shapes are green, yellow, or maroon.
#    - So the only changed inference step is the rule match: color equality is replaced by
#      exact shape equality (translation only, no rotation/flip).
#
# b) Complexity calibration
#    - Source sprites are 3-5 cells, 4-connected, and forced to occupy at least a 2x2 bbox,
#      so they are visibly not single pixels or straight 1D bars.
#    - Different rules always use different normalized shapes; otherwise a solver could not
#      know which target color belongs to which scattered object.
#    - Scattered object colors are drawn from a small shared palette and intentionally need
#      not match the legend color. This makes the D1 change visible in only 2-3 examples.
#
# c) Old hidden assumptions that break here
#    - Original `_discover_rules` scanned individual cells; that assumes every source object
#      is a pixel. In this variant we must scan whole components and remember the contact cell.
#    - Original replacement loop iterated over `g == source_color`; this assumes color fully
#      identifies all replaceable objects. Now colors are irrelevant, so we must iterate over
#      components and compare normalized shapes instead.
#    - Original preservation logic skipped every pixel whose color was any rule source color.
#      That would incorrectly erase unrelated components here, because scattered objects may
#      use arbitrary colors. Preservation must therefore be positional/component-based.
#
# d) Placement legality guarantees used by the generator
#    - Every legend source sprite has exactly one cell adjacent to its target pixel.
#    - All rule legends, scattered sprites, and bystanders are separated by a 4-neighbor moat,
#      except for the intended source->target adjacency inside each legend.
#    - Teal stencils may later overlap after stamping, but the input objects do not overlap.
#
# e) Visual observability and cleanliness
#    - The solver can only succeed by noticing shape repetition: same-colored objects can map
#      to different outputs if their shapes differ, and differently-colored objects can map to
#      the same output if their shapes match.
#    - Bystanders remain single pixels, while source objects are always multi-cell sprites,
#      so the scene stays readable and unmatched leftovers are easy to spot.
#
# f) New solver assumptions and how the generator satisfies them
#    - Assumption: each source legend is a multi-cell monochrome component.
#      Guarantee: source sprites are generated with area 3-5.
#    - Assumption: each rule shape is unique.
#      Guarantee: generator rejects duplicate normalized shapes.
#    - Assumption: the contact cell is uniquely recoverable from legend adjacency.
#      Guarantee: generator only accepts source-shape / direction pairs with exactly one
#      source cell adjacent to the target cell.


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
    rel = tuple(sorted((int(r - r0), int(c - c0)) for r, c in coords))
    return {
        "bbox": (r0, c0, r1, c1),
        "rel": rel,
        "coord_set": set((int(r), int(c)) for r, c in coords),
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
    cell_to_comp = {}
    for comp in comps:
        for r, c in comp["coord_set"]:
            cell_to_comp[(int(r), int(c))] = comp["index"]

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

                target_comp_idx = cell_to_comp.get((int(tr), int(tc)))
                if target_comp_idx is None or comps[target_comp_idx]["area"] != 1:
                    continue

                if sum(1 for rr, cc in comp["coord_set"] if abs(rr - tr) + abs(cc - tc) == 1) != 1:
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
                key = (int(comp["index"]), (int(tr), int(tc)), int(pattern_index))
                if key in seen:
                    continue
                seen.add(key)
                rules.append(
                    {
                        "source_comp_index": int(comp["index"]),
                        "source_shape": comp["rel"],
                        "source_color": int(comp["color"]),
                        "source_anchor_rel": (int(r - r0), int(c - c0)),
                        "target_color": int(target_color),
                        "target_pos": (int(tr), int(tc)),
                        "anchor": (int(anchor[0]), int(anchor[1])),
                        "pattern": np.asarray(pattern, dtype=int),
                    }
                )

    return rules


def main(grid: np.ndarray) -> np.ndarray:
    """
    Use local sprite-target-pattern examples to replace matching sprite components.

    Input structure:
    - Black background with several small colored sprites, isolated colored pixels,
      and teal (8) pattern components.
    - A rule example is a multi-cell source sprite with one adjacent target pixel;
      extending that direction reaches an anchor touching a teal pattern component.

    Rule:
    - Learn the source sprite's exact shape and which sprite cell touches the target.
    - For every other component with the same shape, align that rule's teal component
      so the rule anchor lands on the matching contact cell, then recolor it to the
      target pixel's color.
    - Remove teal components and all cells belonging to rule examples or matched
      source sprites; keep unrelated colored bystanders unchanged.

    Key insight: The legend no longer keys off the source color. The repeated invariant
    is the source sprite's geometry plus its unique contact cell next to the target.
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
            contact_r = int(cr0 + rule["source_anchor_rel"][0])
            contact_c = int(cc0 + rule["source_anchor_rel"][1])
            ar, ac = rule["anchor"]
            pattern = rule["pattern"]
            dr = contact_r - int(ar)
            dc = contact_c - int(ac)
            rows = pattern[:, 0] + dr
            cols = pattern[:, 1] + dc
            ok = (0 <= rows) & (rows < h) & (0 <= cols) & (cols < w)
            out[rows[ok], cols[ok]] = int(rule["target_color"])
            continue

        for r, c in comp["coord_set"]:
            out[int(r), int(c)] = int(comp["color"])

    return out


# --------------- generator ---------------

# Adapted capacities: source objects are now multi-cell sprites, so instance counts and
# minimum sizes are slightly tightened to keep dense 4-rule cases placeable and legible.
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
SCATTERED_RETRIES_PER_OBJECT = 80
BYSTANDER_RETRIES = 80
OUTER_RETRIES = 500
SOURCE_SHAPE_RETRIES = 300


def _reserve_4neighbors(reserved: set[tuple[int, int]], cells) -> None:
    for r, c in cells:
        for dr, dc in _DIRS4:
            reserved.add((int(r + dr), int(c + dc)))


def _translate(rel, top: int, left: int) -> set[tuple[int, int]]:
    return {(int(top + r), int(left + c)) for r, c in rel}


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
    anchor = random.choice(coords_list)
    rel = tuple(sorted((r - anchor[0], c - anchor[1]) for r, c in coords_list))
    return rel


def _valid_contacts(rel: tuple[tuple[int, int], ...]) -> list[tuple[tuple[int, int], tuple[int, int]]]:
    rel_set = set(rel)
    out: list[tuple[tuple[int, int], tuple[int, int]]] = []
    for r, c in rel:
        for dr, dc in _DIRS4:
            target = (r + dr, c + dc)
            if target in rel_set:
                continue
            if sum(1 for rr, cc in rel_set if abs(rr - target[0]) + abs(cc - target[1]) == 1) == 1:
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

    max_s = min(STENCIL_MAX_CAP, max(2, area // STENCIL_MAX_AREA_DIV))
    min_s = min(max(1, area // STENCIL_MIN_AREA_DIV), max_s)
    sc_max = min(SCATTERED_MAX_CAP, max(SCATTERED_MIN, area // SCATTERED_AREA_DIV))
    sc_max = min(sc_max, max(SCATTERED_MIN, 7 - max(0, num_rules - 1)))

    shared_scatter_colors = random.sample(NON_BG_NON_TEAL_COLORS, k=random.randint(2, 3))
    used_shapes: set[tuple[tuple[int, int], ...]] = set()
    rules = []
    used_colors = set()

    for i in range(num_rules):
        src_legend = colors[2 * i]
        tgt = colors[2 * i + 1]
        used_colors.update((src_legend, tgt))
        source_rel, contact_rel, direction = _make_source_shape(used_shapes)
        teal_rel = _make_stencil(min_s, max_s)
        rules.append(
            {
                "src_legend": int(src_legend),
                "tgt": int(tgt),
                "source_rel": source_rel,
                "contact_rel": contact_rel,
                "dir": direction,
                "teal_rel": teal_rel,
                "n_sc": random.randint(SCATTERED_MIN, sc_max),
                "scatter_palette": [c for c in shared_scatter_colors if c != int(src_legend)] or [c for c in NON_BG_NON_TEAL_COLORS if c != int(src_legend)],
            }
        )
    return rules, used_colors


def _place_legends(rules, inp: np.ndarray, occupied: set, reserved: set) -> bool:
    h, w = inp.shape
    for rule in rules:
        placed = False
        for _ in range(LEGEND_RETRIES):
            source_rel = rule["source_rel"]
            sr0 = random.randint(0, h - 1)
            sc0 = random.randint(0, w - 1)
            source_cells = _translate(source_rel, sr0, sc0)
            if not source_cells:
                continue
            if min(r for r, _ in source_cells) < 0 or max(r for r, _ in source_cells) >= h:
                continue
            if min(c for _, c in source_cells) < 0 or max(c for _, c in source_cells) >= w:
                continue

            contact_r = sr0 + rule["contact_rel"][0]
            contact_c = sc0 + rule["contact_rel"][1]
            dr, dc = rule["dir"]
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
                inp[r, c] = int(rule["src_legend"])
            inp[target[0], target[1]] = int(rule["tgt"])
            for r, c in teal_cells:
                inp[r, c] = 8

            occupied.update(all_cells)
            _reserve_4neighbors(reserved, all_cells)
            rule["legend_top_left"] = (int(sr0), int(sc0))
            rule["target_pos"] = target
            rule["anchor"] = anchor
            placed = True
            break

        if not placed:
            return False

    return True


def _place_scattered(rules, inp: np.ndarray, occupied: set, reserved: set) -> bool:
    h, w = inp.shape
    for rule in rules:
        objects = []
        for _ in range(rule["n_sc"] * SCATTERED_RETRIES_PER_OBJECT):
            if len(objects) >= rule["n_sc"]:
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

            color = int(random.choice(rule["scatter_palette"]))
            for r, c in cells:
                inp[r, c] = color
            occupied.update(cells)
            _reserve_4neighbors(reserved, cells)
            objects.append({"top_left": (int(top), int(left)), "color": color})

        if len(objects) < SCATTERED_MIN:
            return False
        rule["scattered"] = objects

    return True


def _place_bystanders(used_colors, inp: np.ndarray, occupied: set, reserved: set):
    bystander_colors = [c for c in NON_BG_NON_TEAL_COLORS if c not in used_colors]
    if not bystander_colors:
        bystander_colors = NON_BG_NON_TEAL_COLORS[:]

    h, w = inp.shape
    n = random.randint(0, BYSTANDER_MAX)
    placed = []
    for _ in range(n):
        for _ in range(BYSTANDER_RETRIES):
            r = random.randint(0, h - 1)
            c = random.randint(0, w - 1)
            if (r, c) in occupied or (r, c) in reserved:
                continue
            color = int(random.choice(bystander_colors))
            inp[r, c] = color
            placed.append((int(r), int(c), color))
            occupied.add((int(r), int(c)))
            _reserve_4neighbors(reserved, [(int(r), int(c))])
            break
    return placed


def _build_output(rules, bystander_pos, h: int, w: int) -> np.ndarray:
    output = np.zeros((h, w), dtype=int)
    for r, c, color in bystander_pos:
        output[int(r), int(c)] = int(color)

    for rule in rules:
        for obj in rule["scattered"]:
            top, left = obj["top_left"]
            contact_r = int(top + rule["contact_rel"][0])
            contact_c = int(left + rule["contact_rel"][1])
            for rr, cc in rule["teal_rel"]:
                nr = int(contact_r + rr)
                nc = int(contact_c + cc)
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
    bystander_pos = _place_bystanders(used_colors, inp, occupied, reserved)

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
