# Combo variant of abc82100 (index 10)
# combo_spec: D3/3+D4/2
# Name: 带方向标记源+模板旋转/翻转

from __future__ import annotations

import random
import sys

import numpy as np

try:
    from arc_utils import components_mask, get_bg
except ModuleNotFoundError:
    sys.path.append("/home/lixinyu/snap/new/solvers")
    from arc_utils import components_mask, get_bg


# Design analysis for the combo variant
# -------------------------------------
# a) Concrete behavior and trace
#    - The legend stays recognizable from the base task: a 4-neighbor src->tgt pair
#      followed by a teal stencil whose anchor lies one more step in the same
#      direction. That still teaches "source color -> target color" plus one
#      canonical stencil shape.
#    - The changed scattered object is now exactly two cells:
#      {source-colored center} + {fixed marker-color cell in one cardinal neighbor}.
#      The marker no longer acts like another legend color; it is a direction token.
#    - Execution is a causal chain:
#      marker side -> choose stencil orientation -> stamp recolored stencil.
#      Example with a local stencil {(0,0), (1,0), (1,1)} in the legend frame:
#      * base task on source (8,8): stamp stays {(8,8), (9,8), (9,9)}
#      * this combo, marker above source: rotate so "forward" points up, keep it
#        unflipped, then stamp on the source cell
#      * this combo, marker right of source: rotate so "forward" points right, and
#        also mirror because horizontal markers use the mirrored family
#      The visible difference is therefore not a mere shift; the same color can
#      yield different output orientations depending on the nearby marker.
# b) Complexity calibration
#    - Only four marker states exist (up/down/left/right), so the control signal is
#      small and human-visible.
#    - To make the D4 part observable, the generator uses asymmetric stencils and
#      forces each rule to appear with both vertical and horizontal markers. Vertical
#      markers use the unmirrored family; horizontal markers use the mirrored family.
#      This gives repeated evidence for both rotation and reflection without creating
#      eight arbitrary cases.
# c) Hidden assumptions broken in the base solver
#    - Base `_discover_rules` treated every non-bg/non-teal pixel as a possible
#      legend participant. That would misread marker pixels as legend colors, so the
#      fixed marker color must be excluded from legend discovery.
#    - Base rule storage used absolute teal coordinates. That fails once different
#      scattered objects request different orientations, so the learned stencil must
#      be converted into a local frame derived from the legend direction.
#    - Base stamping aligned a raw copied stencil to every source-colored pixel.
#      That is invalid now: only non-legend source pixels with exactly one adjacent
#      marker and no other colored 4-neighbors are true scattered objects.
#    - Base cleanup would preserve the new marker pixels. The combo solver must erase
#      markers together with the source objects and legend region.
# d) Placement legality
#    - The generator places each source+marker pair atomically, checks bounds for
#      both cells, and reserves 4-neighbor padding so no accidental colored adjacency
#      creates false legends or ambiguous source objects.
#    - It also tracks the future output footprint of every transformed stamp, keeping
#      transformed results inside bounds and non-overlapping for visual clarity.
# e) Visual observability
#    - A human sees the same legend-derived stencil reappear around several sources,
#      but the orientation varies only with the marker side.
#    - Because the marker has a fixed dedicated color and repeats across objects, the
#      extra control variable is explicit rather than hidden in color id arithmetic.
# f) Visual cleanliness
#    - Inputs stay sparse: legends are isolated, source objects are two-cell pairs,
#      and bystanders avoid both the marker color and all future output footprints.
#      Outputs stay readable because transformed stamps do not overlap one another.
# g) Color-change inferability
#    - Color substitution is still taught locally by the src->tgt legend pair. The
#      marker changes only orientation, not recoloring, so the new difficulty is
#      structural and remains inferable from a few examples.
# h) Quantitative anti-degeneration target
#    - A sample is considered healthy only if each rule uses both a vertical marker
#      and a horizontal marker, and at least one scattered object for that rule
#      produces a footprint different from the base-style "no transform, just
#      translate the legend stencil onto the source" baseline.
# i) New solver assumptions that the generator must guarantee
#    - Marker color appears only as direction markers.
#    - Every scattered source has exactly one adjacent marker and no other colored
#      4-neighbors.
#    - Source colors are unique across rules, so source_color alone picks the rule.
#    - Learned stencils are asymmetric enough that horizontal mirrored placement is
#      visibly different from vertical unmirrored placement.


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


def _left(vec: tuple[int, int]) -> tuple[int, int]:
    return (-int(vec[1]), int(vec[0]))


def _chebyshev(a: tuple[int, int], b: tuple[int, int]) -> int:
    return max(abs(int(a[0]) - int(b[0])), abs(int(a[1]) - int(b[1])))


def _anchor_distance(pattern: np.ndarray, anchor: tuple[int, int]) -> int:
    return min(_chebyshev((int(r), int(c)), anchor) for r, c in pattern)


def _marker_flip(marker_dir: tuple[int, int]) -> bool:
    # Vertical markers keep the canonical chirality; horizontal markers use the
    # mirrored family so one visible extra degree of freedom survives in the combo.
    return int(marker_dir[1]) != 0


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
    patterns = components_mask(grid == int(pattern_color), connectivity=8)
    rules: list[dict] = []
    seen: set[tuple[tuple[int, int], tuple[int, int], int]] = set()

    for r in range(h):
        for c in range(w):
            source_color = int(grid[r, c])
            if source_color in (int(bg), int(pattern_color), int(marker_color)):
                continue

            for dr, dc in _DIRS4:
                tr = int(r + dr)
                tc = int(c + dc)
                if not (0 <= tr < h and 0 <= tc < w):
                    continue

                target_color = int(grid[tr, tc])
                if target_color in (int(bg), int(pattern_color), int(marker_color)):
                    continue
                if target_color == source_color:
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
                key = ((int(r), int(c)), (tr, tc), int(pattern_index))
                if key in seen:
                    continue

                seen.add(key)
                forward = (-int(dr), -int(dc))
                lateral = _left(forward)
                local_pattern: list[tuple[int, int]] = []
                for pr, pc in map(tuple, pattern):
                    delta_r = int(pr) - int(anchor[0])
                    delta_c = int(pc) - int(anchor[1])
                    a = delta_r * forward[0] + delta_c * forward[1]
                    b = delta_r * lateral[0] + delta_c * lateral[1]
                    local_pattern.append((int(a), int(b)))

                rules.append(
                    {
                        "source_color": int(source_color),
                        "target_color": int(target_color),
                        "source_pos": (int(r), int(c)),
                        "target_pos": (tr, tc),
                        "anchor": (int(anchor[0]), int(anchor[1])),
                        "local_pattern": np.asarray(local_pattern, dtype=int),
                        "legend_forward": forward,
                    }
                )

    return rules


def _find_source_object(
    grid: np.ndarray,
    source_pos: tuple[int, int],
    bg: int,
    marker_color: int = MARKER_COLOR,
) -> dict | None:
    h, w = grid.shape
    sr, sc = map(int, source_pos)
    marker_dirs: list[tuple[int, int]] = []
    extra_neighbor = False
    for dr, dc in _DIRS4:
        rr = int(sr + dr)
        cc = int(sc + dc)
        if not (0 <= rr < h and 0 <= cc < w):
            continue
        color = int(grid[rr, cc])
        if color == int(marker_color):
            marker_dirs.append((int(dr), int(dc)))
        elif color != int(bg):
            extra_neighbor = True

    if extra_neighbor or len(marker_dirs) != 1:
        return None

    marker_dir = marker_dirs[0]
    marker = (int(sr + marker_dir[0]), int(sc + marker_dir[1]))
    return {
        "center": (sr, sc),
        "marker": marker,
        "forward": marker_dir,
        "flip": _marker_flip(marker_dir),
        "cells": {(sr, sc), marker},
    }


def _collect_rule_objects(
    grid: np.ndarray,
    rule: dict,
    legend_pixels: set[tuple[int, int]],
    bg: int,
) -> list[dict]:
    objects: list[dict] = []
    consumed: set[tuple[int, int]] = set()
    for sr, sc in map(tuple, np.argwhere(grid == int(rule["source_color"]))):
        sr = int(sr)
        sc = int(sc)
        if (sr, sc) in legend_pixels or (sr, sc) in consumed:
            continue
        obj = _find_source_object(grid, (sr, sc), bg=bg, marker_color=MARKER_COLOR)
        if obj is None:
            continue
        if obj["cells"] & legend_pixels or obj["cells"] & consumed:
            continue
        objects.append(obj)
        consumed.update(obj["cells"])
    return objects


def main(grid: np.ndarray) -> np.ndarray:
    """
    Learn legend stencils, then orient each copy from a side marker before stamping it.

    Input structure:
    - Black background with teal (8) stencil components, local src->tgt legend pairs,
      and scattered 2-cell source objects.
    - A legend example is a 4-neighbor source->target pair whose continued direction
      reaches the anchor of a teal stencil component.
    - Every scattered source object is a source-colored pixel with one adjacent fixed
      marker pixel; the marker side gives the requested facing direction.

    Rule:
    - Each legend teaches a source color, the target color to paint with, and a
      canonical teal stencil expressed in the legend's own orientation.
    - For every non-legend source object of that color, rotate the stencil so its
      forward axis points toward the marker; if the marker is horizontal, mirror the
      rotated stencil before stamping.
    - Stamp the transformed stencil on the source pixel, recolor it to the learned
      target color, and erase all legend pixels, teal pixels, sources, and markers.

    Key insight: The legend still teaches "what shape and what color", but the extra
    adjacent marker is now an instruction token that selects which transformed version
    of that legend stencil gets executed at each source.
    """
    g = np.asarray(grid, dtype=int)
    bg = get_bg(g)
    h, w = g.shape

    rules = _discover_rules(g, bg=bg, pattern_color=PATTERN_COLOR, marker_color=MARKER_COLOR)
    legend_pixels: set[tuple[int, int]] = set()
    source_colors: set[int] = set()
    for rule in rules:
        legend_pixels.add(rule["source_pos"])
        legend_pixels.add(rule["target_pos"])
        source_colors.add(int(rule["source_color"]))

    objects_by_rule: list[list[dict]] = []
    object_pixels: set[tuple[int, int]] = set()
    for rule in rules:
        objects = _collect_rule_objects(g, rule=rule, legend_pixels=legend_pixels, bg=bg)
        objects_by_rule.append(objects)
        for obj in objects:
            object_pixels.update(obj["cells"])

    out = np.full_like(g, int(bg))

    for r, c in map(tuple, np.argwhere((g != int(bg)) & (g != int(PATTERN_COLOR)))):
        r = int(r)
        c = int(c)
        color = int(g[r, c])
        if color == int(MARKER_COLOR):
            continue
        if (r, c) in legend_pixels or (r, c) in object_pixels or color in source_colors:
            continue
        out[r, c] = color

    for rule, objects in zip(rules, objects_by_rule):
        local_pattern = rule["local_pattern"]
        for obj in objects:
            placed = _apply_local_pattern(
                local_pattern,
                anchor=obj["center"],
                forward=obj["forward"],
                flip=obj["flip"],
            )
            rows = placed[:, 0]
            cols = placed[:, 1]
            ok = (0 <= rows) & (rows < h) & (0 <= cols) & (cols < w)
            out[rows[ok], cols[ok]] = int(rule["target_color"])

    return out


# --------------- generator (combo) ---------------

GRID_DIM_MAX = 20
MIN_DIM_BY_RULES = {1: 8, 2: 12, 3: 16, 4: 19}

RULE_COUNT_WEIGHTS = [1, 1, 2, 2, 3, 3, 4]
NON_BG_NON_TEAL_COLORS = [1, 3, 4, 5, 6, 7, 9]

STENCIL_MAX_CAP = 7
STENCIL_MIN_AREA_DIV = 90
STENCIL_MAX_AREA_DIV = 45
STENCIL_RETRIES = 400

SCATTERED_MIN = 4
SCATTERED_MAX_CAP = 6
SCATTERED_AREA_DIV = 45

BYSTANDER_MAX = 4

LEGEND_RETRIES = 500
SCATTERED_RETRIES_PER_OBJECT = 140
BYSTANDER_RETRIES = 60
OUTER_RETRIES = 500


def _reserve_4neighbors(reserved: set[tuple[int, int]], cells: set[tuple[int, int]]) -> None:
    for r, c in cells:
        reserved.add((int(r), int(c)))
        for dr, dc in _DIRS4:
            reserved.add((int(r) + dr, int(c) + dc))


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


def _make_local_pattern(min_pixels: int, max_pixels: int) -> np.ndarray:
    for _ in range(STENCIL_RETRIES):
        n = random.randint(max(4, min_pixels), max(max(4, min_pixels), max_pixels))
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
    raise RuntimeError("failed to create asymmetric stencil")


def _build_rules(num_rules: int, area: int) -> tuple[list[dict], set[int]]:
    colors = NON_BG_NON_TEAL_COLORS[:]
    random.shuffle(colors)
    src_colors = colors[:num_rules]
    tgt_pool = [c for c in colors if c not in src_colors]

    max_s = min(STENCIL_MAX_CAP, max(4, area // STENCIL_MAX_AREA_DIV))
    min_s = min(max(4, area // STENCIL_MIN_AREA_DIV), max_s)
    sc_max = min(SCATTERED_MAX_CAP, max(SCATTERED_MIN, area // SCATTERED_AREA_DIV))

    rules: list[dict] = []
    used_colors: set[int] = {MARKER_COLOR}
    used_pairs: set[tuple[int, int]] = set()
    for src_c in src_colors:
        targets = [c for c in tgt_pool if c != int(src_c) and (int(src_c), int(c)) not in used_pairs]
        tgt_c = int(random.choice(targets))
        used_pairs.add((int(src_c), int(tgt_c)))
        used_colors.update((int(src_c), int(tgt_c)))
        rules.append(
            {
                "src": int(src_c),
                "tgt": int(tgt_c),
                "local_pattern": _make_local_pattern(min_s, max_s),
                "n_sc": random.randint(SCATTERED_MIN, sc_max),
            }
        )
    return rules, used_colors


def _direction_plan(n_sc: int) -> list[tuple[int, int]]:
    dirs = _DIRS4[:]
    random.shuffle(dirs)
    plan = dirs[:]
    while len(plan) < n_sc:
        plan.append(random.choice(_DIRS4))
    random.shuffle(plan)
    return plan[:n_sc]


def _place_legends(
    rules: list[dict],
    inp: np.ndarray,
    occupied: set[tuple[int, int]],
    reserved: set[tuple[int, int]],
) -> bool:
    h, w = inp.shape
    for rule in rules:
        placed = False
        for _ in range(LEGEND_RETRIES):
            dr, dc = random.choice(_DIRS4)
            sr = random.randint(0, h - 1)
            sc = random.randint(0, w - 1)
            tr, tc = int(sr + dr), int(sc + dc)
            ar, ac = int(sr + 2 * dr), int(sc + 2 * dc)
            if not (0 <= tr < h and 0 <= tc < w and 0 <= ar < h and 0 <= ac < w):
                continue

            teal_cells = {
                tuple(cell)
                for cell in _apply_local_pattern(
                    rule["local_pattern"],
                    anchor=(ar, ac),
                    forward=(-int(dr), -int(dc)),
                    flip=False,
                )
            }
            all_cells = {(int(sr), int(sc)), (tr, tc)} | teal_cells
            if any(not (0 <= r < h and 0 <= c < w) for r, c in all_cells):
                continue
            if all_cells & (occupied | reserved):
                continue

            inp[int(sr), int(sc)] = int(rule["src"])
            inp[tr, tc] = int(rule["tgt"])
            for r, c in teal_cells:
                inp[r, c] = int(PATTERN_COLOR)
            occupied.update(all_cells)
            _reserve_4neighbors(reserved, all_cells)
            rule["legend_forward"] = (-int(dr), -int(dc))
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
        for marker_dir in _direction_plan(rule["n_sc"]):
            placed = False
            for _ in range(SCATTERED_RETRIES_PER_OBJECT):
                sr = random.randint(0, h - 1)
                sc = random.randint(0, w - 1)
                center = (int(sr), int(sc))
                marker = (int(sr + marker_dir[0]), int(sc + marker_dir[1]))
                cells = {center, marker}
                if any(not (0 <= r < h and 0 <= c < w) for r, c in cells):
                    continue
                if cells & (occupied | reserved):
                    continue

                stamp = {
                    tuple(cell)
                    for cell in _apply_local_pattern(
                        rule["local_pattern"],
                        anchor=center,
                        forward=marker_dir,
                        flip=_marker_flip(marker_dir),
                    )
                }
                if any(not (0 <= r < h and 0 <= c < w) for r, c in stamp):
                    continue
                if stamp & future_output:
                    continue
                if {p for p in stamp if p not in cells} & occupied:
                    continue

                inp[center] = int(rule["src"])
                inp[marker] = int(MARKER_COLOR)
                occupied.update(cells)
                _reserve_4neighbors(reserved, cells)
                future_output.update(stamp)
                objects.append(
                    {
                        "center": center,
                        "marker": marker,
                        "forward": marker_dir,
                        "flip": _marker_flip(marker_dir),
                    }
                )
                placed = True
                break
            if not placed:
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
            color = int(random.choice(bystander_colors))
            inp[r, c] = color
            placed.append((int(r), int(c), color))
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
            stamp = _apply_local_pattern(
                rule["local_pattern"],
                anchor=obj["center"],
                forward=obj["forward"],
                flip=obj["flip"],
            )
            out[stamp[:, 0], stamp[:, 1]] = int(rule["tgt"])
    return out


def _rule_health(rule: dict) -> dict[str, int]:
    dirs = {tuple(obj["forward"]) for obj in rule["objects"]}
    vertical = sum(1 for obj in rule["objects"] if int(obj["forward"][0]) != 0)
    horizontal = sum(1 for obj in rule["objects"] if int(obj["forward"][1]) != 0)
    changed = 0
    for obj in rule["objects"]:
        baseline = {
            tuple(cell)
            for cell in _apply_local_pattern(
                rule["local_pattern"],
                anchor=obj["center"],
                forward=rule["legend_forward"],
                flip=False,
            )
        }
        actual = {
            tuple(cell)
            for cell in _apply_local_pattern(
                rule["local_pattern"],
                anchor=obj["center"],
                forward=obj["forward"],
                flip=obj["flip"],
            )
        }
        if actual != baseline:
            changed += 1
    return {
        "unique_dirs": len(dirs),
        "vertical": vertical,
        "horizontal": horizontal,
        "changed": changed,
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

    for rule in rules:
        health = _rule_health(rule)
        if health["vertical"] == 0 or health["horizontal"] == 0:
            return None
        if health["changed"] == 0:
            return None

    output = _build_output(rules, bystanders, h, w)
    try:
        discovered = _discover_rules(inp, bg=get_bg(inp), pattern_color=PATTERN_COLOR, marker_color=MARKER_COLOR)
        if len(discovered) != num_rules:
            return None
        discovered_pairs = {(int(rule["source_color"]), int(rule["target_color"])) for rule in discovered}
        intended_pairs = {(int(rule["src"]), int(rule["tgt"])) for rule in rules}
        if discovered_pairs != intended_pairs:
            return None
        if not np.array_equal(main(inp), output):
            return None
    except Exception:
        return None
    return inp


def generate_input() -> np.ndarray:
    for _ in range(OUTER_RETRIES):
        result = _gen_attempt()
        if result is not None:
            return result
    raise RuntimeError(f"Failed to generate valid input after {OUTER_RETRIES} attempts")
