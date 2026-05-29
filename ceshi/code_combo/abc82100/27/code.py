# Combo variant of abc82100 (index 27)
# combo_spec: D3/3+D4/2+D16/2
# Name: 方向标记源+模板旋转+颜色对方向编码

from __future__ import annotations

import random

import numpy as np

from arc_utils import components_mask, get_bg


# Design analysis for the combo variant
# -------------------------------------
# a) Concrete behavior and trace
#    - The base task had one legend per rule:
#        source -> target -> anchor -> nearby teal stencil
#      and every scattered source pixel simply received that stencil translated so
#      the legend anchor landed on the source pixel.
#    - Here every scattered source is a 2-cell object:
#        {source-colored center} + {fixed marker-colored neighbor}
#      so the marker side supplies a visible base direction and also becomes the
#      placement anchor.
#    - The legend's source->target direction still identifies the rule, but now it
#      also contributes an extra quarter-turn code. We keep the D16 mapping:
#        source left of target  -> extra turn 0
#        source above target    -> extra turn 1
#        source right of target -> extra turn 2
#        source below target    -> extra turn 3
#    - To integrate D4, the final transform is not "just rotate the raw offsets".
#      Instead the stencil is stored in a local (forward, left) frame. At stamp
#      time the marker direction gives the base forward axis; the legend direction
#      rotates that axis by 0/90/180/270; directions 2 and 3 also activate a
#      mirror across the final forward axis. Therefore the same marker direction
#      can lead to different absolute outputs depending on the legend, and the
#      same legend can still produce several outputs because different markers
#      change the base axis.
#    - Example:
#      Suppose the local stencil relative to anchor is {(0,0), (1,0), (1,1)} in
#      (forward,left) coordinates. If a scattered source has its marker on the
#      east side, then:
#        * legend pair east  -> use forward=east, no mirror
#        * legend pair south -> use forward=south, no mirror
#        * legend pair west  -> use forward=west, mirror
#      All three come from the same source+marker object type, but the legend adds
#      an extra transform layer beyond the D3 shift.
# b) Complexity calibration
#    - The marker has only four possible sides and the legend direction has only
#      four possible directions. That is enough to express the intended combined
#      transform while staying human-trackable.
#    - To keep D4 visible, the generator only emits stencil shapes whose 8 dihedral
#      variants are all distinct. Symmetric stencils would collapse the new logic.
#    - Each rule gets several scattered source+marker objects and uses at least two
#      marker directions; grids with multiple rules also use multiple legend
#      directions, including mirrored cases when possible.
# c) Hidden assumptions broken in the base solver
#    - Base rule discovery treated every non-bg/non-teal color as a possible legend
#      member. That fails once marker pixels are introduced, so marker_color must be
#      excluded from legend discovery.
#    - Base stamping aligned anchor -> source cell. That is no longer true: anchor
#      must align to the marker cell.
#    - Base stamping also used raw absolute stencil coordinates. Here that would
#      destroy the rotation/flip semantics, so rules must first convert stencil
#      cells into a legend-relative local frame.
#    - Base preservation copied any non-legend non-source colored cell. Marker
#      pixels would leak into the output unless they are explicitly removed.
#    - A naive fusion of D3 and D16 could still reduce to pure rotation. The combo
#      solver therefore includes a mirror bit derived from the legend direction, and
#      the generator guarantees asymmetric stencils so this extra degree of freedom
#      visibly matters.
# d) Placement legality
#    - Legend examples are placed with full spacing so no scattered source can form
#      a false source->target pair that accidentally points to a teal component.
#    - Each source+marker object is placed atomically, both cells must fit, and its
#      future output stencil must also fit fully in-bounds.
#    - Future output stamps are kept disjoint from one another and from occupied
#      input structures. This is stricter than the base task but keeps the fused
#      logic legible and avoids ambiguous overlaps.
# e) Visual observability
#    - The input visibly contains two directional cues: the marker side on each
#      scattered source and the direction of each legend color pair.
#    - A human can compare repeated source+marker objects of the same color and see
#      that marker direction changes orientation, then compare different legends and
#      see that the legend direction adds another transform on top of that.
#    - No rule depends on invisible facts such as numeric color ordering.
# f) Visual cleanliness
#    - Only one extra marker color is introduced, teal remains legend-only, and
#      scattered objects are sparse 2-cell motifs rather than denser seeds.
#    - Output stamps are separated and in-bounds, so the final recolored shapes are
#      easy to inspect instead of merging into clutter.
# g) Color-change inferability
#    - Recoloring is still taught locally by legend source/target pairs.
#    - Targets are chosen from colors that never act as sources, so seeing a source
#      color in the input always means "this belongs to a rule", while the marker
#      color has a single global meaning.
# New solver assumptions checked against the generator
#    - Every non-legend source occurrence has exactly one 4-neighbor marker.
#    - Marker color never appears inside legends or among bystanders.
#    - The nearby teal component chosen for each legend actually contains the anchor.
#    - Stencil shapes are dihedrally asymmetric, so rotation and mirror decisions
#      change the produced output instead of accidentally matching each other.


MARKER_COLOR = 2
PATTERN_COLOR = 8

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
_STEP_BY_ROT = {
    0: (0, 1),
    1: (1, 0),
    2: (0, -1),
    3: (-1, 0),
}
_ROT_BY_STEP = {step: rot for rot, step in _STEP_BY_ROT.items()}


def _left(vec: tuple[int, int]) -> tuple[int, int]:
    return -int(vec[1]), int(vec[0])


def _rotate_dir(vec: tuple[int, int], turns: int) -> tuple[int, int]:
    dr, dc = int(vec[0]), int(vec[1])
    for _ in range(int(turns) % 4):
        dr, dc = dc, -dr
    return int(dr), int(dc)


def _rot90_local(coords: tuple[tuple[int, int], ...]) -> tuple[tuple[int, int], ...]:
    return tuple(sorted((-int(b), int(a)) for a, b in coords))


def _flip_local(coords: tuple[tuple[int, int], ...]) -> tuple[tuple[int, int], ...]:
    return tuple(sorted((int(a), -int(b)) for a, b in coords))


def _variant_signatures(local_coords: list[tuple[int, int]]) -> set[tuple[tuple[int, int], ...]]:
    base = tuple(sorted((int(a), int(b)) for a, b in local_coords))
    seen: set[tuple[tuple[int, int], ...]] = set()
    cur = base
    for _ in range(4):
        seen.add(cur)
        seen.add(_flip_local(cur))
        cur = _rot90_local(cur)
    return seen


def _chebyshev(a: tuple[int, int], b: tuple[int, int]) -> int:
    return max(abs(int(a[0]) - int(b[0])), abs(int(a[1]) - int(b[1])))


def _anchor_distance(pattern: np.ndarray, anchor: tuple[int, int]) -> int:
    return min(_chebyshev((int(r), int(c)), anchor) for r, c in pattern)


def _apply_local_pattern(
    local_pattern: np.ndarray,
    anchor: tuple[int, int],
    forward: tuple[int, int],
    flip: bool,
) -> np.ndarray:
    anchor_r, anchor_c = int(anchor[0]), int(anchor[1])
    forward_r, forward_c = int(forward[0]), int(forward[1])
    left_r, left_c = _left(forward)
    side_sign = -1 if flip else 1

    rows = anchor_r + local_pattern[:, 0] * forward_r + side_sign * local_pattern[:, 1] * left_r
    cols = anchor_c + local_pattern[:, 0] * forward_c + side_sign * local_pattern[:, 1] * left_c
    return np.stack([rows, cols], axis=1).astype(int)


def _discover_rules(
    grid: np.ndarray,
    bg: int,
    pattern_color: int = PATTERN_COLOR,
    marker_color: int = MARKER_COLOR,
) -> list[dict]:
    h, w = grid.shape
    patterns = components_mask(grid == pattern_color, connectivity=8)
    rules: list[dict] = []
    seen: set[tuple[tuple[int, int], tuple[int, int], int]] = set()

    for r in range(h):
        for c in range(w):
            source_color = int(grid[r, c])
            if source_color in (int(bg), int(pattern_color), int(marker_color)):
                continue

            for dr, dc in _DIRS4:
                tr = r + dr
                tc = c + dc
                if not (0 <= tr < h and 0 <= tc < w):
                    continue

                target_color = int(grid[tr, tc])
                if target_color in (int(bg), int(pattern_color), int(marker_color)):
                    continue
                if target_color == source_color:
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

                legend_forward = (-int(dr), -int(dc))
                legend_left = _left(legend_forward)
                local_pattern: list[tuple[int, int]] = []
                for pr, pc in map(tuple, pattern):
                    delta_r = int(pr) - int(anchor[0])
                    delta_c = int(pc) - int(anchor[1])
                    a = delta_r * legend_forward[0] + delta_c * legend_forward[1]
                    b = delta_r * legend_left[0] + delta_c * legend_left[1]
                    local_pattern.append((int(a), int(b)))

                extra_turns = int(_ROT_BY_STEP[(int(dr), int(dc))])
                rules.append(
                    {
                        "source_color": int(source_color),
                        "target_color": int(target_color),
                        "source_pos": (int(r), int(c)),
                        "target_pos": (int(tr), int(tc)),
                        "anchor": (int(anchor[0]), int(anchor[1])),
                        "local_pattern": np.asarray(local_pattern, dtype=int),
                        "extra_turns": extra_turns,
                        "flip": bool(extra_turns >= 2),
                    }
                )

    return rules


def _find_marker(
    grid: np.ndarray,
    source_pos: tuple[int, int],
    marker_color: int = MARKER_COLOR,
) -> tuple[int, int] | None:
    sr, sc = int(source_pos[0]), int(source_pos[1])
    hits = []
    for dr, dc in _DIRS4:
        mr = sr + dr
        mc = sc + dc
        if 0 <= mr < grid.shape[0] and 0 <= mc < grid.shape[1] and int(grid[mr, mc]) == int(marker_color):
            hits.append((int(mr), int(mc)))
    if len(hits) != 1:
        return None
    return hits[0]


def _find_scattered_objects(
    grid: np.ndarray,
    source_color: int,
    legend_pixels: set[tuple[int, int]],
    bg: int,
    marker_color: int = MARKER_COLOR,
) -> list[dict]:
    h, w = grid.shape
    objects: list[dict] = []
    used_markers: set[tuple[int, int]] = set()

    for sr, sc in map(tuple, np.argwhere(grid == int(source_color))):
        sr = int(sr)
        sc = int(sc)
        if (sr, sc) in legend_pixels:
            continue

        marker = _find_marker(grid, (sr, sc), marker_color=marker_color)
        if marker is None:
            continue
        if marker in legend_pixels or marker in used_markers:
            continue

        mr, mc = marker
        marker_dir = (int(mr - sr), int(mc - sc))
        extra_neighbor = False
        for dr, dc in _DIRS4:
            rr = sr + dr
            cc = sc + dc
            if not (0 <= rr < h and 0 <= cc < w):
                continue
            if (rr, cc) == marker:
                continue
            color = int(grid[rr, cc])
            if color != int(bg):
                extra_neighbor = True
                break
        if extra_neighbor:
            continue

        objects.append(
            {
                "source": (sr, sc),
                "marker": marker,
                "marker_dir": marker_dir,
                "cells": {(sr, sc), marker},
            }
        )
        used_markers.add(marker)

    return objects


def _placed_cells_for_object(rule: dict, marker: tuple[int, int], marker_dir: tuple[int, int]) -> np.ndarray:
    final_forward = _rotate_dir(marker_dir, int(rule["extra_turns"]))
    return _apply_local_pattern(
        np.asarray(rule["local_pattern"], dtype=int),
        anchor=marker,
        forward=final_forward,
        flip=bool(rule["flip"]),
    )


def _behavior_counts(grid: np.ndarray) -> dict[str, int]:
    g = np.asarray(grid, dtype=int)
    bg = get_bg(g)
    rules = _discover_rules(g, bg=bg, pattern_color=PATTERN_COLOR, marker_color=MARKER_COLOR)
    legend_pixels = {tuple(rule["source_pos"]) for rule in rules} | {tuple(rule["target_pos"]) for rule in rules}

    objects_total = 0
    changed_vs_d3 = 0
    changed_vs_base = 0
    mirrored_objects = 0
    rotated_objects = 0

    for rule in rules:
        objects = _find_scattered_objects(
            g,
            source_color=int(rule["source_color"]),
            legend_pixels=legend_pixels,
            bg=bg,
            marker_color=MARKER_COLOR,
        )
        for obj in objects:
            objects_total += 1
            marker = obj["marker"]
            marker_dir = obj["marker_dir"]

            combo = {
                tuple(cell)
                for cell in _placed_cells_for_object(rule, marker=marker, marker_dir=marker_dir)
            }
            d3_only = {
                tuple(cell)
                for cell in _apply_local_pattern(
                    np.asarray(rule["local_pattern"], dtype=int),
                    anchor=marker,
                    forward=marker_dir,
                    flip=False,
                )
            }
            base_like = {
                tuple(cell)
                for cell in _apply_local_pattern(
                    np.asarray(rule["local_pattern"], dtype=int),
                    anchor=obj["source"],
                    forward=_rotate_dir(marker_dir, int(rule["extra_turns"])),
                    flip=bool(rule["flip"]),
                )
            }

            if combo != d3_only:
                changed_vs_d3 += 1
            if combo != base_like:
                changed_vs_base += 1
            if bool(rule["flip"]):
                mirrored_objects += 1
            if int(rule["extra_turns"]) % 4 != 0:
                rotated_objects += 1

    return {
        "objects": int(objects_total),
        "changed_vs_d3": int(changed_vs_d3),
        "changed_vs_base": int(changed_vs_base),
        "mirrored_objects": int(mirrored_objects),
        "rotated_objects": int(rotated_objects),
    }


def main(grid: np.ndarray) -> np.ndarray:
    """
    Learn legend stencils, then stamp transformed copies using both marker direction and legend direction.

    Input structure:
    - Black background with teal (8) legend stencils, local source->target examples,
      and scattered 2-cell source objects made of a source-colored pixel plus one
      adjacent fixed marker pixel.
    - Each legend is a 4-neighbor source/target pair whose continued direction
      reaches an anchor touching a teal stencil component.
    - The marker side on each scattered source gives a base orientation, while the
      legend pair direction supplies an additional transform code.

    Rule:
    - For each legend, learn the source color, target color, and the nearby teal
      stencil in a local coordinate frame tied to the legend line.
    - For every non-legend source object of that color, place the stencil with its
      anchor on the marker cell. Start from the marker direction as the base axis,
      rotate that axis according to the legend pair direction, and mirror the
      stencil for the two legend directions that encode the reflected cases.
    - Recolor the placed stencil to the legend target color, and erase legends,
      teal pixels, source pixels, and marker pixels.

    Key insight: Solving the puzzle now needs a three-way fusion: the legend still
    teaches the color mapping and canonical stencil, the side marker tells you the
    base facing of each scattered source, and the legend pair orientation modifies
    that facing into the final rotated or mirrored stencil copy.
    """
    g = np.asarray(grid, dtype=int)
    bg = get_bg(g)
    h, w = g.shape

    rules = _discover_rules(g, bg=bg, pattern_color=PATTERN_COLOR, marker_color=MARKER_COLOR)
    legend_pixels: set[tuple[int, int]] = set()
    source_colors: set[int] = set()
    for rule in rules:
        legend_pixels.add(tuple(rule["source_pos"]))
        legend_pixels.add(tuple(rule["target_pos"]))
        source_colors.add(int(rule["source_color"]))

    objects_by_rule: list[list[dict]] = []
    object_pixels: set[tuple[int, int]] = set()
    for rule in rules:
        objects = _find_scattered_objects(
            g,
            source_color=int(rule["source_color"]),
            legend_pixels=legend_pixels,
            bg=bg,
            marker_color=MARKER_COLOR,
        )
        objects_by_rule.append(objects)
        for obj in objects:
            object_pixels.update(obj["cells"])

    out = np.full_like(g, int(bg))

    for r, c in map(tuple, np.argwhere((g != int(bg)) & (g != PATTERN_COLOR))):
        r = int(r)
        c = int(c)
        color = int(g[r, c])
        if color == int(MARKER_COLOR):
            continue
        if (r, c) in legend_pixels or (r, c) in object_pixels or color in source_colors:
            continue
        out[r, c] = color

    for rule, objects in zip(rules, objects_by_rule):
        for obj in objects:
            placed = _placed_cells_for_object(rule, marker=obj["marker"], marker_dir=obj["marker_dir"])
            rows = placed[:, 0]
            cols = placed[:, 1]
            ok = (0 <= rows) & (rows < h) & (0 <= cols) & (cols < w)
            out[rows[ok], cols[ok]] = int(rule["target_color"])

    return out


# --------------- generator (combo) ---------------

GRID_DIM_MAX = 20
MIN_DIM_BY_RULES = {1: 8, 2: 12, 3: 16, 4: 19}

RULE_COUNT_WEIGHTS = [1, 1, 2, 2, 2, 3, 3, 4]
NON_BG_NON_TEAL_COLORS = [1, 3, 4, 5, 6, 7, 9]

STENCIL_MAX_CAP = 7
STENCIL_MIN_AREA_DIV = 90
STENCIL_MAX_AREA_DIV = 45
ASYMMETRIC_STENCIL_RETRIES = 300

SCATTERED_MIN = 3
SCATTERED_MAX_CAP = 6
SCATTERED_AREA_DIV = 45

BYSTANDER_MAX = 4

LEGEND_RETRIES = 600
SCATTERED_RETRIES_PER_OBJECT = 160
BYSTANDER_RETRIES = 80
OUTER_RETRIES = 500


def _reserve_4neighbors(reserved: set[tuple[int, int]], cells: set[tuple[int, int]]) -> None:
    for r, c in cells:
        reserved.add((int(r), int(c)))
        for dr, dc in _DIRS4:
            reserved.add((int(r) + dr, int(c) + dc))


def _make_local_pattern(min_pixels: int, max_pixels: int) -> np.ndarray:
    lo = max(4, int(min_pixels))
    hi = max(lo, int(max_pixels))
    for _ in range(ASYMMETRIC_STENCIL_RETRIES):
        n = random.randint(lo, hi)
        coords = {(0, 0)}
        while len(coords) < n:
            frontier = set()
            for a, b in coords:
                for da, db in _DIRS8:
                    cand = (int(a) + da, int(b) + db)
                    if cand not in coords:
                        frontier.add(cand)
            coords.add(random.choice(list(frontier)))

        local_coords = sorted(coords)
        if len(_variant_signatures(local_coords)) != 8:
            continue
        if not any(a != 0 for a, _b in local_coords):
            continue
        if not any(b != 0 for _a, b in local_coords):
            continue
        return np.asarray(local_coords, dtype=int)

    return np.asarray([(0, 0), (1, 0), (1, 1), (2, 0)], dtype=int)


def _choose_rotations(num_rules: int) -> list[int]:
    if num_rules == 1:
        return [random.choice([1, 2, 3])]

    pool = [0, 1, 2, 3]
    random.shuffle(pool)
    chosen = [random.choice([0, 1]), random.choice([2, 3])]
    for rot in pool:
        if rot not in chosen:
            chosen.append(rot)
        if len(chosen) >= num_rules:
            break
    random.shuffle(chosen)
    return chosen[:num_rules]


def _marker_plan(n_sc: int) -> list[tuple[int, int]]:
    dirs = _DIRS4[:]
    random.shuffle(dirs)
    chosen = [dirs[0], dirs[1]]
    while len(chosen) < n_sc:
        chosen.append(random.choice(dirs))
    random.shuffle(chosen)
    return chosen


def _build_rules(num_rules: int, area: int) -> tuple[list[dict], set[int]]:
    colors = NON_BG_NON_TEAL_COLORS[:]
    random.shuffle(colors)

    src_colors = colors[:num_rules]
    tgt_pool = [c for c in colors if c not in src_colors]
    max_s = min(STENCIL_MAX_CAP, max(4, area // STENCIL_MAX_AREA_DIV))
    min_s = min(max(4, area // STENCIL_MIN_AREA_DIV), max_s)
    sc_max = min(SCATTERED_MAX_CAP, max(SCATTERED_MIN, area // SCATTERED_AREA_DIV))
    rotations = _choose_rotations(num_rules)

    rules: list[dict] = []
    used_colors: set[int] = {MARKER_COLOR}
    for src_c, rot in zip(src_colors, rotations):
        tgt_c = random.choice(tgt_pool)
        local_pattern = _make_local_pattern(min_s, max_s)
        rules.append(
            {
                "src": int(src_c),
                "tgt": int(tgt_c),
                "local_pattern": local_pattern,
                "extra_turns": int(rot),
                "flip": bool(rot >= 2),
                "pair_step": _STEP_BY_ROT[int(rot)],
                "n_sc": random.randint(SCATTERED_MIN, sc_max),
            }
        )
        used_colors.update((int(src_c), int(tgt_c)))
    return rules, used_colors


def _legend_teal_cells(rule: dict, anchor: tuple[int, int]) -> set[tuple[int, int]]:
    forward = (-int(rule["pair_step"][0]), -int(rule["pair_step"][1]))
    return {
        tuple(cell)
        for cell in _apply_local_pattern(
            np.asarray(rule["local_pattern"], dtype=int),
            anchor=anchor,
            forward=forward,
            flip=False,
        )
    }


def _object_stamp_cells(rule: dict, source: tuple[int, int], marker_dir: tuple[int, int]) -> set[tuple[int, int]]:
    marker = (int(source[0] + marker_dir[0]), int(source[1] + marker_dir[1]))
    return {
        tuple(cell)
        for cell in _placed_cells_for_object(rule, marker=marker, marker_dir=marker_dir)
    }


def _place_legends(
    rules: list[dict],
    inp: np.ndarray,
    occupied: set[tuple[int, int]],
    reserved: set[tuple[int, int]],
) -> bool:
    h, w = inp.shape
    for rule in rules:
        dr, dc = int(rule["pair_step"][0]), int(rule["pair_step"][1])
        placed = False
        for _ in range(LEGEND_RETRIES):
            sr = random.randint(0, h - 1)
            sc = random.randint(0, w - 1)
            tr, tc = sr + dr, sc + dc
            ar, ac = sr + 2 * dr, sc + 2 * dc
            if not (0 <= tr < h and 0 <= tc < w and 0 <= ar < h and 0 <= ac < w):
                continue

            teal_cells = _legend_teal_cells(rule, anchor=(ar, ac))
            all_cells = {(int(sr), int(sc)), (int(tr), int(tc))} | teal_cells
            if any(not (0 <= r < h and 0 <= c < w) for r, c in all_cells):
                continue
            if all_cells & (occupied | reserved):
                continue

            inp[sr, sc] = int(rule["src"])
            inp[tr, tc] = int(rule["tgt"])
            for r, c in teal_cells:
                inp[r, c] = PATTERN_COLOR
            occupied.update(all_cells)
            _reserve_4neighbors(reserved, all_cells)
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
    future_output: set[tuple[int, int]],
) -> bool:
    h, w = inp.shape
    for rule in rules:
        objects: list[dict] = []
        for marker_dir in _marker_plan(int(rule["n_sc"])):
            placed = False
            for _ in range(SCATTERED_RETRIES_PER_OBJECT):
                sr = random.randint(0, h - 1)
                sc = random.randint(0, w - 1)
                source = (int(sr), int(sc))
                marker = (int(sr + marker_dir[0]), int(sc + marker_dir[1]))
                cells = {source, marker}
                if any(not (0 <= r < h and 0 <= c < w) for r, c in cells):
                    continue
                if cells & (occupied | reserved):
                    continue

                stamp = _object_stamp_cells(rule, source=source, marker_dir=marker_dir)
                if any(not (0 <= r < h and 0 <= c < w) for r, c in stamp):
                    continue
                if stamp & future_output:
                    continue
                if stamp & occupied:
                    continue

                inp[source] = int(rule["src"])
                inp[marker] = MARKER_COLOR
                occupied.update(cells)
                _reserve_4neighbors(reserved, cells)
                future_output.update(stamp)
                objects.append(
                    {
                        "source": source,
                        "marker": marker,
                        "marker_dir": marker_dir,
                    }
                )
                placed = True
                break
            if not placed:
                return False
        if len(objects) < SCATTERED_MIN:
            return False
        rule["objects"] = objects
    return True


def _place_bystanders(
    used_colors: set[int],
    inp: np.ndarray,
    occupied: set[tuple[int, int]],
    reserved: set[tuple[int, int]],
    future_output: set[tuple[int, int]],
) -> list[tuple[int, int, int]]:
    bystander_colors = [c for c in NON_BG_NON_TEAL_COLORS if c not in used_colors]
    if not bystander_colors:
        return []

    h, w = inp.shape
    placed: list[tuple[int, int, int]] = []
    n = random.randint(0, min(BYSTANDER_MAX, len(bystander_colors) * 2))
    for _ in range(n):
        for _ in range(BYSTANDER_RETRIES):
            r = random.randint(0, h - 1)
            c = random.randint(0, w - 1)
            if (r, c) in occupied or (r, c) in reserved or (r, c) in future_output:
                continue
            color = random.choice(bystander_colors)
            inp[r, c] = int(color)
            placed.append((int(r), int(c), int(color)))
            occupied.add((int(r), int(c)))
            _reserve_4neighbors(reserved, {(int(r), int(c))})
            break
    return placed


def _build_output(rules: list[dict], bystanders: list[tuple[int, int, int]], h: int, w: int) -> np.ndarray:
    out = np.zeros((h, w), dtype=int)
    for r, c, color in bystanders:
        out[int(r), int(c)] = int(color)

    for rule in rules:
        for obj in rule["objects"]:
            stamp = _placed_cells_for_object(rule, marker=obj["marker"], marker_dir=obj["marker_dir"])
            out[stamp[:, 0], stamp[:, 1]] = int(rule["tgt"])
    return out


def _expected_rule_signature(rules: list[dict]) -> dict[int, tuple[int, int, bool]]:
    return {
        int(rule["src"]): (int(rule["tgt"]), int(rule["extra_turns"]), bool(rule["flip"]))
        for rule in rules
    }


def _discovered_rule_signature(discovered: list[dict]) -> dict[int, tuple[int, int, bool]]:
    return {
        int(rule["source_color"]): (
            int(rule["target_color"]),
            int(rule["extra_turns"]),
            bool(rule["flip"]),
        )
        for rule in discovered
    }


def _gen_attempt() -> np.ndarray | None:
    num_rules = random.choice(RULE_COUNT_WEIGHTS)
    min_dim = MIN_DIM_BY_RULES[num_rules]
    h = random.randint(min_dim, GRID_DIM_MAX)
    w = random.randint(min_dim, GRID_DIM_MAX)

    rules, used_colors = _build_rules(num_rules, h * w)
    inp = np.zeros((h, w), dtype=int)
    occupied: set[tuple[int, int]] = set()
    reserved: set[tuple[int, int]] = set()
    future_output: set[tuple[int, int]] = set()

    if not _place_legends(rules, inp, occupied, reserved):
        return None
    if not _place_scattered(rules, inp, occupied, reserved, future_output):
        return None
    bystanders = _place_bystanders(used_colors, inp, occupied, reserved, future_output)

    output = _build_output(rules, bystanders, h, w)
    try:
        discovered = _discover_rules(inp, bg=get_bg(inp), pattern_color=PATTERN_COLOR, marker_color=MARKER_COLOR)
        if len(discovered) != num_rules:
            return None
        if _discovered_rule_signature(discovered) != _expected_rule_signature(rules):
            return None

        stats = _behavior_counts(inp)
        if stats["objects"] < num_rules * SCATTERED_MIN:
            return None
        if stats["changed_vs_base"] != stats["objects"]:
            return None
        if stats["changed_vs_d3"] <= 0:
            return None
        if num_rules > 1 and stats["mirrored_objects"] <= 0:
            return None

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
