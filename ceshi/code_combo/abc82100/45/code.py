# Combo variant of abc82100 (index 45)
# combo_spec: D4/3+D5/0+D12/2
# Name: 距离排序处理+中心对齐+模板缩放

from __future__ import annotations

import random

import numpy as np

from arc_utils import components_mask, get_bg


# Design analysis for combo D4/3 + D5/0 + D12/2
# a) Concrete behavior and trace:
# - The original task learns local legends of the form source -> target -> anchor+teal
#   and then copies the teal component onto every scattered source pixel by anchor
#   translation at one fixed size.
# - This combo keeps the same legend discovery, but changes the scattered stage in
#   one unified way around the grid center:
#   1) each scattered source pixel is assigned a center-distance d;
#   2) d also chooses the scale: inner-ring sources use scale 1, outer-ring
#      sources use scale 3;
#   3) the copied prototype is centered on the source pixel, not anchor-aligned;
#   4) all scattered instances across all rules are processed in increasing d, so
#      farther instances are stamped later and overwrite nearer ones on overlaps.
# - Example: if a rule has a 3-cell asymmetric prototype and two scattered sources,
#   one near the grid center and one in the outer ring, then the near source
#   produces a small centered copy first, while the far source produces a larger
#   centered copy later. If the large copy overlaps the earlier one, the overlap
#   region ends in the farther source's target color.
# - Relative to the source task, the changed stages are now coupled instead of
#   independent: distance determines both ordering and size, while centering
#   changes where that scaled footprint lands.
# b) Complexity calibration:
# - Scale is restricted to two visible odd factors {1, 3}. Odd factors keep the
#   centered geometry exact on cell grids; using many factors would make the
#   samples noisy and would weaken repeated support inside 2-4 demonstrations.
# - Every generated rule gets at least one inner-ring and one outer-ring source,
#   so the same color mapping is observed at multiple sizes.
# - Every generated sample uses at least two rules and includes one intentional
#   cross-rule overlap between a near small stamp and a farther large stamp, so
#   distance ordering is not merely nominal.
# c) Hidden assumptions audit against the original main():
# - `_discover_rules(...)` still assumes a 4-neighbor source/target pair whose
#   forward anchor touches a teal component. The combo preserves that exactly;
#   legends still identify which prototype belongs to which color substitution.
# - The original stamping formula `source - anchor` is invalid now. The solver
#   must compute prototype coordinates relative to the prototype center, then
#   scale those offsets, and finally center the scaled result on the source.
# - The original solver treated every source-colored non-legend cell as an
#   independent instance with no global interaction. That breaks because output
#   now depends on global distance ordering across all rules.
# - The old source-color removal logic still holds because scattered instances are
#   still plain source-colored pixels and bystanders never reuse source colors.
# d) Placement legality enforced by the generator:
# - Prototype legends are placed after scattered sources and are forbidden from
#   touching input objects or any final output footprint.
# - Scattered source pixels remain 4-neighbor isolated to avoid creating false
#   local source->target legends.
# - Most final stamps are non-overlapping for cleanliness; exactly one designated
#   cross-rule overlap is forced to make the ordering rule observable.
# - Distances of scattered source pixels are kept unique inside a sample, so the
#   distance sort is deterministic and does not rely on hidden tie-breaking.
# e) Visual observability:
# - A human sees that source pixels nearer the geometric center yield smaller
#   target motifs, while farther ones yield larger motifs of the same shape.
# - Because the same center-distance also governs overwrite order, overlap colors
#   can be explained spatially without referring to color ids or hidden numbers.
# - Non-symmetric prototypes with anchor != center make center alignment visibly
#   different from the original anchor-aligned task.
# f) Visual cleanliness:
# - Bystanders stay outside both input objects and output footprints.
# - Only one overlap hotspot is intentionally introduced; all other stamps are
#   kept separate, so the sample remains readable.
# g) Change inferability / generator rejection tests:
# - The generator rejects any sample where the combo output equals the output of
#   any of these ablations:
#   1) center-aligned but unscaled stamping,
#   2) anchor-aligned but distance-scaled stamping,
#   3) center-aligned scaled stamping in reverse distance order.
# - This directly checks that scaling, centering, and ordering all affect the
#   produced sample rather than being present only in the code.


def _chebyshev(a: tuple[int, int], b: tuple[int, int]) -> int:
    return max(abs(int(a[0]) - int(b[0])), abs(int(a[1]) - int(b[1])))


def _anchor_distance(pattern: np.ndarray, anchor: tuple[int, int]) -> int:
    return min(_chebyshev((int(r), int(c)), anchor) for r, c in pattern)


def _pattern_center(pattern: np.ndarray) -> tuple[int, int]:
    rows = pattern[:, 0]
    cols = pattern[:, 1]
    return (
        int((int(rows.min()) + int(rows.max())) // 2),
        int((int(cols.min()) + int(cols.max())) // 2),
    )


def _center_distance_key(h: int, w: int, r: int, c: int) -> int:
    return max(abs(2 * int(r) - (int(h) - 1)), abs(2 * int(c) - (int(w) - 1)))


def _scale_for_source(h: int, w: int, r: int, c: int) -> int:
    return 1 if _center_distance_key(h, w, r, c) * 2 <= max(h - 1, w - 1) else 3


def _scaled_cells_from_rel(
    rel_coords: list[tuple[int, int]] | np.ndarray,
    source: tuple[int, int],
    scale: int,
    h: int,
    w: int,
) -> set[tuple[int, int]]:
    sr, sc = int(source[0]), int(source[1])
    k = int(scale)
    half = k // 2
    cells: set[tuple[int, int]] = set()
    for rr, cc in rel_coords:
        base_r = sr + int(rr) * k
        base_c = sc + int(cc) * k
        for dr in range(-half, half + 1):
            for dc in range(-half, half + 1):
                nr = base_r + dr
                nc = base_c + dc
                if 0 <= nr < h and 0 <= nc < w:
                    cells.add((int(nr), int(nc)))
    return cells


def _span_for_rel(rel_coords: list[tuple[int, int]], scale: int) -> tuple[int, int, int, int]:
    k = int(scale)
    half = k // 2
    rows: list[int] = []
    cols: list[int] = []
    for rr, cc in rel_coords:
        base_r = int(rr) * k
        base_c = int(cc) * k
        for dr in range(-half, half + 1):
            for dc in range(-half, half + 1):
                rows.append(base_r + dr)
                cols.append(base_c + dc)
    return min(rows), max(rows), min(cols), max(cols)


def _fits_full(rel_coords: list[tuple[int, int]], scale: int, r: int, c: int, h: int, w: int) -> bool:
    min_r, max_r, min_c, max_c = _span_for_rel(rel_coords, scale)
    return 0 <= int(r) + min_r and int(r) + max_r < h and 0 <= int(c) + min_c and int(c) + max_c < w


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
                pattern_arr = np.asarray(pattern, dtype=int)
                center = _pattern_center(pattern_arr)
                rules.append(
                    {
                        "source_color": int(source_color),
                        "target_color": int(target_color),
                        "source_pos": (int(r), int(c)),
                        "target_pos": (int(tr), int(tc)),
                        "anchor": (int(anchor[0]), int(anchor[1])),
                        "center": center,
                        "pattern": pattern_arr,
                        "rel_center": [
                            (int(pr - center[0]), int(pc - center[1])) for pr, pc in pattern_arr.tolist()
                        ],
                    }
                )

    return rules


def main(grid: np.ndarray) -> np.ndarray:
    """
    Learn local color legends and stamp centered scaled motifs in center-distance order.

    Input structure:
    - Black background with local source->target legend pairs and nearby teal (8)
      prototype components.
    - Away from the legends, each scattered source is a single isolated pixel of
      a learned source color; its position relative to the grid center determines
      both its stamp size and when it is applied.

    Rule:
    - Discover each rule from an adjacent colored source/target pair whose forward
      anchor touches a teal prototype component.
    - For every non-legend source pixel of that color, choose scale 1 for inner
      center-distance positions and scale 3 for outer ones, then stamp a scaled
      copy of the prototype recolored to the rule's target color with the scaled
      prototype center aligned to the source pixel.
    - Apply all such stamps globally from nearest-to-center to farthest, so later
      farther stamps overwrite earlier nearer ones on overlaps; erase legends and
      all teal prototype pixels from the output.

    Key insight: The legend still teaches color substitution and shape, but the
    scattered source's center distance now controls two linked things at once:
    how large the centered stamp becomes and whether it is overwritten by a later,
    farther stamp in overlap regions.
    """
    g = np.asarray(grid, dtype=int)
    bg = get_bg(g)
    pattern_color = 8
    h, w = g.shape

    rules = _discover_rules(g, bg=bg, pattern_color=pattern_color)
    legend_pixels: set[tuple[int, int]] = set()
    source_colors: set[int] = set()
    instances: list[tuple[int, int, int, int, list[tuple[int, int]]]] = []

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
        rel_center = rule["rel_center"]
        for sr, sc in map(tuple, np.argwhere(g == rule["source_color"])):
            sr = int(sr)
            sc = int(sc)
            if (sr, sc) in legend_pixels:
                continue
            dist = _center_distance_key(h, w, sr, sc)
            instances.append((int(dist), int(sr), int(sc), int(rule["target_color"]), rel_center))

    instances.sort(key=lambda item: (item[0], item[1], item[2], item[3]))
    for _dist, sr, sc, target_color, rel_center in instances:
        scale = _scale_for_source(h, w, sr, sc)
        for rr, cc in _scaled_cells_from_rel(rel_center, (sr, sc), scale, h, w):
            out[rr, cc] = int(target_color)

    return out


# --------------- generator (combo) ---------------

GRID_DIM_MAX = 20
MIN_DIM_BY_RULES = {2: 12, 3: 16}

RULE_COUNT_WEIGHTS = [2, 2, 2, 3, 3]
NON_BG_NON_TEAL_COLORS = [1, 2, 3, 4, 5, 6, 7, 9]

PROTOTYPE_MIN_CELLS = 3
PROTOTYPE_MAX_CELLS = 5
PROTOTYPE_MAX_SPAN = 3
PROTOTYPE_RETRIES = 300

SCATTERED_PER_RULE = 3
BYSTANDER_MAX = 4

PAIR_RETRIES = 300
INSTANCE_RETRIES = 400
LEGEND_RETRIES = 700
BYSTANDER_RETRIES = 100
OUTER_RETRIES = 500

_DIRS4 = [(-1, 0), (1, 0), (0, -1), (0, 1)]
_DIRS8 = [(-1, -1), (-1, 0), (-1, 1), (0, -1), (0, 1), (1, -1), (1, 0), (1, 1)]


def _reserve_4neighbors(reserved: set[tuple[int, int]], cells) -> None:
    for r, c in cells:
        for dr, dc in _DIRS4:
            reserved.add((int(r) + dr, int(c) + dc))


def _normalize_coords(coords: set[tuple[int, int]]) -> list[tuple[int, int]]:
    min_r = min(r for r, _ in coords)
    min_c = min(c for _, c in coords)
    return sorted((int(r - min_r), int(c - min_c)) for r, c in coords)


def _is_connected(coords: set[tuple[int, int]]) -> bool:
    todo = [next(iter(coords))]
    seen = {todo[0]}
    while todo:
        r, c = todo.pop()
        for dr, dc in _DIRS8:
            nxt = (int(r) + dr, int(c) + dc)
            if nxt in coords and nxt not in seen:
                seen.add(nxt)
                todo.append(nxt)
    return len(seen) == len(coords)


def _bbox_center_local(coords: list[tuple[int, int]]) -> tuple[int, int]:
    rows = [r for r, _ in coords]
    cols = [c for _, c in coords]
    return ((min(rows) + max(rows)) // 2, (min(cols) + max(cols)) // 2)


def _is_centrally_symmetric(coords: list[tuple[int, int]]) -> bool:
    rows = [r for r, _ in coords]
    cols = [c for _, c in coords]
    max_r = max(rows)
    max_c = max(cols)
    coords_set = set(coords)
    return all((max_r - r, max_c - c) in coords_set for r, c in coords)


def _make_prototype() -> tuple[list[tuple[int, int]], tuple[int, int], tuple[int, int]]:
    for _ in range(PROTOTYPE_RETRIES):
        n = random.randint(PROTOTYPE_MIN_CELLS, PROTOTYPE_MAX_CELLS)
        coords = {(0, 0)}
        while len(coords) < n:
            r, c = random.choice(list(coords))
            dr, dc = random.choice(_DIRS8)
            coords.add((int(r) + dr, int(c) + dc))

        coords_list = _normalize_coords(coords)
        rows = [r for r, _ in coords_list]
        cols = [c for _, c in coords_list]
        height = max(rows) + 1
        width = max(cols) + 1
        if height > PROTOTYPE_MAX_SPAN or width > PROTOTYPE_MAX_SPAN:
            continue
        if height % 2 == 0 or width % 2 == 0:
            continue
        if not _is_connected(set(coords_list)):
            continue
        if _is_centrally_symmetric(coords_list):
            continue

        center = _bbox_center_local(coords_list)
        if center not in set(coords_list):
            continue
        anchor_candidates = [p for p in coords_list if p != center]
        if not anchor_candidates:
            continue
        anchor = max(
            anchor_candidates,
            key=lambda p: (_chebyshev(p, center), abs(p[0] - center[0]) + abs(p[1] - center[1])),
        )
        return coords_list, anchor, center

    raise RuntimeError("Failed to build compact asymmetric centered prototype")


def _build_rules(num_rules: int):
    colors = NON_BG_NON_TEAL_COLORS[:]
    random.shuffle(colors)

    rules: list[dict] = []
    used_colors: set[int] = set()
    for i in range(num_rules):
        src_c, tgt_c = colors[2 * i], colors[2 * i + 1]
        used_colors.update((src_c, tgt_c))
        stencil, anchor, center = _make_prototype()
        rel_anchor = sorted((int(r - anchor[0]), int(c - anchor[1])) for r, c in stencil)
        rel_center = sorted((int(r - center[0]), int(c - center[1])) for r, c in stencil)
        rules.append(
            {
                "src": int(src_c),
                "tgt": int(tgt_c),
                "rel_anchor": rel_anchor,
                "rel_center": rel_center,
                "scattered": [],
                "bands": [],
            }
        )
    return rules, used_colors


def _band_name(h: int, w: int, r: int, c: int) -> str:
    return "near" if _scale_for_source(h, w, r, c) == 1 else "far"


def _iter_cells(h: int, w: int) -> list[tuple[int, int]]:
    cells = [(r, c) for r in range(h) for c in range(w)]
    random.shuffle(cells)
    return cells


def _place_source_instance(
    rule: dict,
    inp: np.ndarray,
    occupied: set[tuple[int, int]],
    reserved: set[tuple[int, int]],
    output_footprint: set[tuple[int, int]],
    used_distances: set[int],
    *,
    required_band: str,
    prefer_full: bool,
) -> bool:
    h, w = inp.shape
    rel_center = rule["rel_center"]
    cells = _iter_cells(h, w)

    for require_full in ([prefer_full, False] if prefer_full else [False]):
        for r, c in cells:
            if (r, c) in occupied or (r, c) in reserved:
                continue
            if _band_name(h, w, r, c) != required_band:
                continue
            dist = _center_distance_key(h, w, r, c)
            if dist in used_distances:
                continue
            scale = _scale_for_source(h, w, r, c)
            if require_full and not _fits_full(rel_center, scale, r, c, h, w):
                continue
            stamp = _scaled_cells_from_rel(rel_center, (r, c), scale, h, w)
            if not stamp or (stamp & output_footprint):
                continue

            inp[r, c] = int(rule["src"])
            occupied.add((int(r), int(c)))
            _reserve_4neighbors(reserved, [(int(r), int(c))])
            output_footprint.update(stamp)
            used_distances.add(int(dist))
            rule["scattered"].append((int(r), int(c)))
            rule["bands"].append(required_band)
            return True

    return False


def _place_overlap_pair(
    rules: list[dict],
    inp: np.ndarray,
    occupied: set[tuple[int, int]],
    reserved: set[tuple[int, int]],
    output_footprint: set[tuple[int, int]],
    used_distances: set[int],
) -> bool:
    h, w = inp.shape
    rule_near = rules[0]
    rule_far = rules[1]
    near_cells = _iter_cells(h, w)
    far_cells = _iter_cells(h, w)

    for _ in range(PAIR_RETRIES):
        for nr, nc in near_cells:
            if (nr, nc) in occupied or (nr, nc) in reserved:
                continue
            if _band_name(h, w, nr, nc) != "near":
                continue
            near_dist = _center_distance_key(h, w, nr, nc)
            if near_dist in used_distances:
                continue
            if not _fits_full(rule_near["rel_center"], 1, nr, nc, h, w):
                continue
            near_stamp = _scaled_cells_from_rel(rule_near["rel_center"], (nr, nc), 1, h, w)

            for fr, fc in far_cells:
                if (fr, fc) in occupied or (fr, fc) in reserved:
                    continue
                if abs(int(fr) - int(nr)) + abs(int(fc) - int(nc)) <= 1:
                    continue
                if _band_name(h, w, fr, fc) != "far":
                    continue
                far_dist = _center_distance_key(h, w, fr, fc)
                if far_dist in used_distances or far_dist <= near_dist:
                    continue
                far_stamp = _scaled_cells_from_rel(rule_far["rel_center"], (fr, fc), 3, h, w)
                inter = near_stamp & far_stamp
                if not inter:
                    continue
                if len(inter) >= len(near_stamp) or len(inter) >= len(far_stamp):
                    continue

                inp[nr, nc] = int(rule_near["src"])
                inp[fr, fc] = int(rule_far["src"])
                occupied.update({(int(nr), int(nc)), (int(fr), int(fc))})
                _reserve_4neighbors(reserved, [(int(nr), int(nc)), (int(fr), int(fc))])
                output_footprint.update(near_stamp)
                output_footprint.update(far_stamp)
                used_distances.update({int(near_dist), int(far_dist)})
                rule_near["scattered"].append((int(nr), int(nc)))
                rule_near["bands"].append("near")
                rule_far["scattered"].append((int(fr), int(fc)))
                rule_far["bands"].append("far")
                return True

    return False


def _place_remaining_sources(
    rules: list[dict],
    inp: np.ndarray,
    occupied: set[tuple[int, int]],
    reserved: set[tuple[int, int]],
    output_footprint: set[tuple[int, int]],
    used_distances: set[int],
) -> bool:
    for rule in rules:
        if "near" not in rule["bands"]:
            if not _place_source_instance(
                rule,
                inp,
                occupied,
                reserved,
                output_footprint,
                used_distances,
                required_band="near",
                prefer_full=True,
            ):
                return False
        if "far" not in rule["bands"]:
            if not _place_source_instance(
                rule,
                inp,
                occupied,
                reserved,
                output_footprint,
                used_distances,
                required_band="far",
                prefer_full=False,
            ):
                return False

    for idx, rule in enumerate(rules):
        while len(rule["scattered"]) < SCATTERED_PER_RULE:
            preferred_band = "far" if idx % 2 == 0 else "near"
            if not _place_source_instance(
                rule,
                inp,
                occupied,
                reserved,
                output_footprint,
                used_distances,
                required_band=preferred_band,
                prefer_full=(preferred_band == "near"),
            ):
                other_band = "near" if preferred_band == "far" else "far"
                if not _place_source_instance(
                    rule,
                    inp,
                    occupied,
                    reserved,
                    output_footprint,
                    used_distances,
                    required_band=other_band,
                    prefer_full=(other_band == "near"),
                ):
                    return False

    return True


def _place_bystanders(
    used_colors: set[int],
    inp: np.ndarray,
    occupied: set[tuple[int, int]],
    reserved: set[tuple[int, int]],
    output_footprint: set[tuple[int, int]],
) -> list[tuple[int, int, int]]:
    bystander_colors = [c for c in NON_BG_NON_TEAL_COLORS if c not in used_colors]
    if not bystander_colors:
        return []

    h, w = inp.shape
    n = random.randint(0, BYSTANDER_MAX)
    placed: list[tuple[int, int, int]] = []
    for _ in range(n):
        for r, c in _iter_cells(h, w):
            if (r, c) in occupied or (r, c) in reserved or (r, c) in output_footprint:
                continue
            color = int(random.choice(bystander_colors))
            inp[r, c] = color
            placed.append((int(r), int(c), color))
            occupied.add((int(r), int(c)))
            _reserve_4neighbors(reserved, [(int(r), int(c))])
            break
    return placed


def _place_legends(
    rules: list[dict],
    inp: np.ndarray,
    occupied: set[tuple[int, int]],
    reserved: set[tuple[int, int]],
    output_footprint: set[tuple[int, int]],
) -> bool:
    h, w = inp.shape
    for rule in rules:
        rel = rule["rel_anchor"]
        placed = False
        for _ in range(LEGEND_RETRIES):
            dr, dc = random.choice(_DIRS4)
            sr = random.randint(0, h - 1)
            sc = random.randint(0, w - 1)
            tr, tc = sr + dr, sc + dc
            ar, ac = sr + 2 * dr, sc + 2 * dc
            if not (0 <= tr < h and 0 <= tc < w and 0 <= ar < h and 0 <= ac < w):
                continue

            teal_cells = {(int(ar + rr), int(ac + cc)) for rr, cc in rel}
            if any(not (0 <= r < h and 0 <= c < w) for r, c in teal_cells):
                continue
            reverse_anchor = (int(sr - dr), int(sc - dc))
            if any(_chebyshev(reverse_anchor, cell) <= 1 for cell in teal_cells):
                continue

            all_cells = {(int(sr), int(sc)), (int(tr), int(tc)), (int(ar), int(ac))} | teal_cells
            if all_cells & occupied:
                continue
            if all_cells & reserved:
                continue
            if all_cells & output_footprint:
                continue

            inp[sr, sc] = int(rule["src"])
            inp[tr, tc] = int(rule["tgt"])
            for r, c in teal_cells:
                inp[r, c] = 8
            occupied.update(all_cells)
            _reserve_4neighbors(reserved, all_cells)
            placed = True
            break
        if not placed:
            return False
    return True


def _ordered_instances(rules: list[dict], h: int, w: int, *, reverse: bool = False) -> list[tuple[int, int, int, dict]]:
    items: list[tuple[int, int, int, dict]] = []
    for idx, rule in enumerate(rules):
        for sr, sc in rule["scattered"]:
            dist = _center_distance_key(h, w, sr, sc)
            items.append((int(dist), int(sr), int(sc), {"rule_index": idx}))
    items.sort(key=lambda item: (item[0], item[1], item[2], item[3]["rule_index"]), reverse=reverse)
    return items


def _build_output(
    rules: list[dict],
    bystander_pos: list[tuple[int, int, int]],
    h: int,
    w: int,
    *,
    rel_key: str,
    scaled: bool,
    reverse_order: bool,
) -> np.ndarray:
    output = np.zeros((h, w), dtype=int)
    for r, c, color in bystander_pos:
        output[int(r), int(c)] = int(color)

    for dist, sr, sc, meta in _ordered_instances(rules, h, w, reverse=reverse_order):
        del dist
        rule = rules[int(meta["rule_index"])]
        rel = rule[rel_key]
        scale = _scale_for_source(h, w, sr, sc) if scaled else 1
        for r, c in _scaled_cells_from_rel(rel, (sr, sc), scale, h, w):
            output[r, c] = int(rule["tgt"])

    return output


def _gen_attempt() -> np.ndarray | None:
    num_rules = random.choice(RULE_COUNT_WEIGHTS)
    min_dim = MIN_DIM_BY_RULES[num_rules]
    h = random.randint(min_dim, GRID_DIM_MAX)
    w = random.randint(min_dim, GRID_DIM_MAX)

    rules, used_colors = _build_rules(num_rules)
    inp = np.zeros((h, w), dtype=int)
    occupied: set[tuple[int, int]] = set()
    reserved: set[tuple[int, int]] = set()
    output_footprint: set[tuple[int, int]] = set()
    used_distances: set[int] = set()

    if not _place_overlap_pair(rules, inp, occupied, reserved, output_footprint, used_distances):
        return None
    if not _place_remaining_sources(rules, inp, occupied, reserved, output_footprint, used_distances):
        return None
    bystander_pos = _place_bystanders(used_colors, inp, occupied, reserved, output_footprint)
    if not _place_legends(rules, inp, occupied, reserved, output_footprint):
        return None

    discovered = _discover_rules(inp, bg=get_bg(inp), pattern_color=8)
    expected_pairs = {(int(rule["src"]), int(rule["tgt"])) for rule in rules}
    found_pairs = {(int(rule["source_color"]), int(rule["target_color"])) for rule in discovered}
    if found_pairs != expected_pairs or len(discovered) != len(rules):
        return None

    combo_output = _build_output(rules, bystander_pos, h, w, rel_key="rel_center", scaled=True, reverse_order=False)
    no_scale_output = _build_output(rules, bystander_pos, h, w, rel_key="rel_center", scaled=False, reverse_order=False)
    anchor_scaled_output = _build_output(
        rules, bystander_pos, h, w, rel_key="rel_anchor", scaled=True, reverse_order=False
    )
    reverse_order_output = _build_output(
        rules, bystander_pos, h, w, rel_key="rel_center", scaled=True, reverse_order=True
    )

    if np.array_equal(combo_output, no_scale_output):
        return None
    if np.array_equal(combo_output, anchor_scaled_output):
        return None
    if np.array_equal(combo_output, reverse_order_output):
        return None

    try:
        if np.array_equal(main(inp), combo_output):
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
