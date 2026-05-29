# Combo variant of abc82100 (index 11)
# combo_spec: D4/2+D14/1
# Name: 模板旋转/翻转+特殊标记锚点

from __future__ import annotations

import random

import numpy as np

from arc_utils import components_mask, get_bg


# Design analysis for the combo variant
# -------------------------------------
# concretization:
# - The legend keeps the original local "source -> target" color pair, but the nearby
#   template is now a mixed 8-connected component: teal(8) cells plus exactly one
#   blue(1) marker cell. The continuation cell after the target is guaranteed to be a
#   teal cell, not the marker.
# - Each scattered occurrence is no longer a lone source pixel. It is a 3-cell seed:
#   the center is source-colored, one same-color arm indicates the forward direction,
#   and one target-colored arm sits on either the left or right side. Those two arms
#   choose a dihedral transform of the legend template.
# - The anchor for stamping is the marker cell inside the legend template. The teal
#   cells are stored in a canonical coordinate frame centered on that marker and
#   oriented by the legend line. When a scattered seed is processed, the same local
#   coordinates are rotated and optionally mirrored before being stamped.
#
# behavior trace:
# - Suppose a legend has source at (5,5), target at (5,6), probe teal at (5,7), and
#   marker at (4,7). Another teal cell is at (4,8). In the base task, the anchor would
#   be the continuation/probe cell (5,7), so stamping onto a scattered source at (10,10)
#   would place that extra teal cell at (9,11).
# - In this combo variant, the true anchor is the marker (4,7). Relative to the marker,
#   the extra teal cell is one step to the right in the legend's local frame. If the
#   scattered seed at (10,10) points south with its side arm on the west, the template
#   is first rotated so "forward" points south, then mirrored so the side is on the
#   opposite side, and only then aligned so the marker would land on (10,10).
# - The key difference is twofold and coupled: the solver must read the marker before
#   it knows the correct local coordinates, and it must rotate/flip those marker-based
#   coordinates according to the seed before stamping.
#
# complexity calibration:
# - A symmetric template would hide the D4 change, so the generator only emits teal
#   shapes whose eight dihedral transforms are all distinct.
# - The marker is never placed on the probe cell; `touch_rel != (0, 0)` is enforced.
#   That prevents the D14 change from collapsing back to the original "anchor is the
#   continuation cell" behavior.
# - Each rule gets multiple scattered seeds and the generator forces at least two
#   distinct orientation states, including both mirror parities when possible. That
#   gives repeated evidence that orientation is controlled by local geometry, not by
#   color id.
# - Source colors are unique across rules. Target colors may repeat because the marker
#   consumes one palette slot, but source uniqueness keeps rule lookup unambiguous.
#
# implicit-assumption audit for main():
# - The old solver assumed pure-teal components; now rule discovery must inspect mixed
#   teal+marker components, verify exactly one marker exists, and only let legend probes
#   attach through teal cells.
# - The old solver anchored on the continuation cell. That is invalid now: we must find
#   the mixed component via the probe, then read the marker position, then re-express
#   teal cells in a local frame around that marker.
# - The old stamping loop treated every non-legend source-colored pixel as a real source.
#   That breaks once scattered seeds contain an extra same-color arm, so we explicitly
#   parse source centers with one same-color neighbor and one perpendicular target-color
#   neighbor and reject any center with extra colored cardinal neighbors.
# - The old preserve-loop only removed teal and legend pixels. Now it must also remove
#   marker pixels and both seed arms; otherwise the input cues would leak into output.
# - New hidden assumptions introduced by the combo solver:
#   * every valid mixed template has exactly one marker and at least one teal cell;
#   * the probe cell after a legend pair belongs to teal, never to the marker;
#   * each scattered seed is isolated enough that its three cells are parsed as one
#     object and do not share cells with another seed or legend;
#   * bystanders do not use any source color, because the solver erases all pixels of
#     learned source colors that are not part of preserved bystanders.
#   The generator below enforces all of these.
#
# placement legality:
# - All colored objects reserve their king-neighborhoods after placement. This prevents
#   mixed templates from merging diagonally and keeps seeds visually isolated.
# - Legend placement solves for the marker location from a chosen probe teal cell, then
#   checks the entire transformed template plus marker stays in-bounds and disjoint.
# - Scattered seeds are only accepted when their transformed output stamp is fully
#   in-bounds and does not overlap any earlier stamp or input object. This is stricter
#   than the base task's clipping allowance, but it keeps the transformation readable.
#
# visual observability and cleanliness:
# - Humans see the same blue anchor dot repeated inside every legend template, so the
#   marker has an obvious persistent role.
# - Humans also see repeated L-shaped seeds of the same two colors. Comparing several
#   seeds of one rule reveals that the target shape changes with the arms' orientation,
#   not with hidden numeric properties of the colors.
# - Because output stamps do not overlap and bystanders avoid future stamp cells, the
#   generated grids stay clean enough to parse from 2-4 examples.
#
# change inferability:
# - Color mapping is still taught by the local source->target legend pair.
# - Rotation/flip is taught structurally by repeated seeds of the same rule using
#   different arm orientations.
# - Anchor choice is taught structurally by the offset between the probe teal cell and
#   the embedded marker; this offset is visible in every legend and affects all stamps.


MARKER_COLOR = 1
PATTERN_COLOR = 8

PAIR_COLORS = [2, 3, 4, 5, 6, 7, 9]

GRID_DIM_MAX = 20
MIN_DIM_BY_RULES = {1: 8, 2: 12, 3: 16, 4: 19}

RULE_COUNT_WEIGHTS = [1, 1, 2, 2, 3, 3, 4]

# Total mixed-template size counts teal cells + one marker.
STENCIL_MAX_CAP = 8
STENCIL_MIN_AREA_DIV = 80
STENCIL_MAX_AREA_DIV = 40

SCATTERED_MIN = 3
SCATTERED_MAX_CAP = 6
SCATTERED_AREA_DIV = 45

BYSTANDER_MAX = 4

LEGEND_RETRIES = 600
SCATTERED_RETRIES_PER_SEED = 150
BYSTANDER_RETRIES = 80
OUTER_RETRIES = 500
STENCIL_RETRIES = 400

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


def _right(vec: tuple[int, int]) -> tuple[int, int]:
    lr, lc = _left(vec)
    return (-lr, -lc)


def _local_offset(
    local_coord: tuple[int, int],
    forward: tuple[int, int],
    flip: bool,
) -> tuple[int, int]:
    a = int(local_coord[0])
    b = int(local_coord[1])
    fr = int(forward[0])
    fc = int(forward[1])
    lr, lc = _left(forward)
    side_sign = -1 if flip else 1
    return (a * fr + side_sign * b * lr, a * fc + side_sign * b * lc)


def _apply_local_pattern(
    local_pattern: np.ndarray,
    anchor: tuple[int, int],
    forward: tuple[int, int],
    flip: bool,
) -> np.ndarray:
    ar = int(anchor[0])
    ac = int(anchor[1])
    rows: list[int] = []
    cols: list[int] = []
    for a, b in map(tuple, local_pattern):
        dr, dc = _local_offset((int(a), int(b)), forward, flip)
        rows.append(ar + dr)
        cols.append(ac + dc)
    return np.stack([rows, cols], axis=1).astype(int)


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


def _reserve_neighbors(reserved: set[tuple[int, int]], cells: set[tuple[int, int]]) -> None:
    for r, c in cells:
        rr = int(r)
        cc = int(c)
        reserved.add((rr, cc))
        for dr, dc in _DIRS8:
            reserved.add((rr + dr, cc + dc))


def _discover_pattern_components(
    grid: np.ndarray,
    pattern_color: int = PATTERN_COLOR,
    marker_color: int = MARKER_COLOR,
) -> tuple[list[dict], dict[tuple[int, int], int]]:
    mask = (grid == int(pattern_color)) | (grid == int(marker_color))
    comps = components_mask(mask, connectivity=8)
    patterns: list[dict] = []
    teal_cell_to_pattern: dict[tuple[int, int], int] = {}

    for pts in comps:
        teal_cells: list[tuple[int, int]] = []
        marker_cells: list[tuple[int, int]] = []
        for r, c in map(tuple, pts):
            rr = int(r)
            cc = int(c)
            color = int(grid[rr, cc])
            if color == int(pattern_color):
                teal_cells.append((rr, cc))
            elif color == int(marker_color):
                marker_cells.append((rr, cc))
        if len(marker_cells) != 1 or not teal_cells:
            continue

        idx = len(patterns)
        marker = marker_cells[0]
        teal_arr = np.asarray(sorted(teal_cells), dtype=int)
        patterns.append({"marker": marker, "teal": teal_arr})
        for cell in teal_cells:
            teal_cell_to_pattern[(int(cell[0]), int(cell[1]))] = idx

    return patterns, teal_cell_to_pattern


def _discover_rules(grid: np.ndarray, bg: int) -> list[dict]:
    h, w = grid.shape
    patterns, teal_cell_to_pattern = _discover_pattern_components(grid)
    rules: list[dict] = []
    seen: set[tuple[tuple[int, int], tuple[int, int], int]] = set()

    for r in range(h):
        for c in range(w):
            source_color = int(grid[r, c])
            if source_color in (int(bg), PATTERN_COLOR, MARKER_COLOR):
                continue

            for dr, dc in _DIRS4:
                tr = r + dr
                tc = c + dc
                pr = tr + dr
                pc = tc + dc
                if not (0 <= tr < h and 0 <= tc < w and 0 <= pr < h and 0 <= pc < w):
                    continue

                target_color = int(grid[tr, tc])
                if target_color in (int(bg), PATTERN_COLOR, MARKER_COLOR) or target_color == source_color:
                    continue

                pattern_index = teal_cell_to_pattern.get((int(pr), int(pc)))
                if pattern_index is None:
                    continue

                key = ((int(r), int(c)), (int(tr), int(tc)), int(pattern_index))
                if key in seen:
                    continue
                seen.add(key)

                pattern = patterns[pattern_index]
                marker = pattern["marker"]
                forward = (-int(dr), -int(dc))
                lateral = _left(forward)
                local_pattern: list[tuple[int, int]] = []
                for rr, cc in map(tuple, pattern["teal"]):
                    delta_r = int(rr) - int(marker[0])
                    delta_c = int(cc) - int(marker[1])
                    a = delta_r * forward[0] + delta_c * forward[1]
                    b = delta_r * lateral[0] + delta_c * lateral[1]
                    local_pattern.append((int(a), int(b)))

                rules.append(
                    {
                        "source_color": int(source_color),
                        "target_color": int(target_color),
                        "source_pos": (int(r), int(c)),
                        "target_pos": (int(tr), int(tc)),
                        "probe_pos": (int(pr), int(pc)),
                        "anchor": (int(marker[0]), int(marker[1])),
                        "local_pattern": np.asarray(local_pattern, dtype=int),
                    }
                )

    return rules


def _find_rule_seeds(
    grid: np.ndarray,
    rule: dict,
    legend_pixels: set[tuple[int, int]],
    bg: int,
) -> list[dict]:
    h, w = grid.shape
    src = int(rule["source_color"])
    tgt = int(rule["target_color"])
    seeds: list[dict] = []
    consumed: set[tuple[int, int]] = set()

    for sr, sc in map(tuple, np.argwhere(grid == src)):
        sr = int(sr)
        sc = int(sc)
        if (sr, sc) in legend_pixels:
            continue

        same_dirs: list[tuple[int, int]] = []
        tgt_dirs: list[tuple[int, int]] = []
        extra_neighbor = False
        for dr, dc in _DIRS4:
            rr = sr + dr
            cc = sc + dc
            if not (0 <= rr < h and 0 <= cc < w):
                continue
            color = int(grid[rr, cc])
            if color == src:
                same_dirs.append((int(dr), int(dc)))
            elif color == tgt:
                tgt_dirs.append((int(dr), int(dc)))
            elif color != int(bg):
                extra_neighbor = True

        if extra_neighbor or len(same_dirs) != 1 or len(tgt_dirs) != 1:
            continue

        forward = same_dirs[0]
        side = tgt_dirs[0]
        if side == _left(forward):
            flip = False
        elif side == _right(forward):
            flip = True
        else:
            continue

        cells = {
            (sr, sc),
            (sr + forward[0], sc + forward[1]),
            (sr + side[0], sc + side[1]),
        }
        if cells & legend_pixels or cells & consumed:
            continue

        seeds.append(
            {
                "center": (sr, sc),
                "forward": forward,
                "flip": bool(flip),
                "cells": cells,
            }
        )
        consumed.update(cells)

    return seeds


def main(grid: np.ndarray) -> np.ndarray:
    """
    Use local color-pair legends plus marked templates to stamp rotated or mirrored copies.

    Input structure:
    - Black background with mixed legend templates made of teal (8) cells and one blue
      marker cell, plus scattered 3-cell source seeds.
    - A rule legend is a 4-neighbor source->target pair whose continued direction lands
      on a teal probe cell belonging to one mixed template component.
    - Each scattered seed has a source-colored center, a same-color forward arm, and a
      target-colored side arm on the left or right.

    Rule:
    - Learn the source->target color substitution from the legend pair, then find the
      touched mixed template and treat its blue marker as the true anchor.
    - Record the teal cells in the legend's local frame around that marker. For every
      scattered seed of the same source color, rotate the template so its forward axis
      matches the seed's same-color arm, and mirror it if the side arm is on the
      opposite side.
    - Stamp only the teal cells in the target color, aligning the rotated marker
      position to the seed center, and erase all legend/template/seed cue pixels.

    Key insight: The probe after the legend pair only identifies which template to use;
    the actual offset comes from the embedded marker, and the seed arms choose which
    rotated or reflected version of that marker-centered template gets copied.
    """
    g = np.asarray(grid, dtype=int)
    bg = get_bg(g)
    h, w = g.shape

    rules = _discover_rules(g, bg=bg)
    legend_pixels: set[tuple[int, int]] = set()
    source_colors: set[int] = set()
    for rule in rules:
        legend_pixels.add(rule["source_pos"])
        legend_pixels.add(rule["target_pos"])
        source_colors.add(rule["source_color"])

    seeds_by_rule: list[list[dict]] = []
    seed_pixels: set[tuple[int, int]] = set()
    for rule in rules:
        seeds = _find_rule_seeds(g, rule=rule, legend_pixels=legend_pixels, bg=bg)
        seeds_by_rule.append(seeds)
        for seed in seeds:
            seed_pixels.update(seed["cells"])

    out = np.full_like(g, int(bg))

    keep_mask = (g != int(bg)) & (g != PATTERN_COLOR) & (g != MARKER_COLOR)
    for r, c in map(tuple, np.argwhere(keep_mask)):
        rr = int(r)
        cc = int(c)
        color = int(g[rr, cc])
        if (rr, cc) in legend_pixels or (rr, cc) in seed_pixels or color in source_colors:
            continue
        out[rr, cc] = color

    for rule, seeds in zip(rules, seeds_by_rule):
        for seed in seeds:
            placed = _apply_local_pattern(
                rule["local_pattern"],
                anchor=seed["center"],
                forward=seed["forward"],
                flip=seed["flip"],
            )
            rows = placed[:, 0]
            cols = placed[:, 1]
            ok = (0 <= rows) & (rows < h) & (0 <= cols) & (cols < w)
            out[rows[ok], cols[ok]] = int(rule["target_color"])

    return out


def _make_stencil(min_cells: int, max_cells: int) -> tuple[np.ndarray, tuple[int, int]]:
    for _ in range(STENCIL_RETRIES):
        total_cells = random.randint(max(5, min_cells), max(max(5, min_cells), max_cells))
        coords = {(0, 0)}
        while len(coords) < total_cells:
            frontier = set()
            for a, b in coords:
                for da, db in _DIRS8:
                    cand = (int(a) + da, int(b) + db)
                    if cand not in coords:
                        frontier.add(cand)
            coords.add(random.choice(list(frontier)))

        teal_rel = sorted(cell for cell in coords if cell != (0, 0))
        if len(_variant_signatures(teal_rel)) != 8:
            continue
        if not any(a != 0 for a, _b in teal_rel):
            continue
        if not any(b != 0 for _a, b in teal_rel):
            continue

        touch_rel = random.choice(teal_rel)
        return np.asarray(teal_rel, dtype=int), (int(touch_rel[0]), int(touch_rel[1]))

    raise RuntimeError("failed to create asymmetric marked template")


def _build_rules(num_rules: int, area: int) -> tuple[list[dict], set[int]]:
    max_s = min(STENCIL_MAX_CAP, max(5, area // STENCIL_MAX_AREA_DIV))
    min_s = min(max(5, area // STENCIL_MIN_AREA_DIV), max_s)
    sc_max = min(SCATTERED_MAX_CAP, max(SCATTERED_MIN, area // SCATTERED_AREA_DIV))

    source_colors = random.sample(PAIR_COLORS, num_rules)
    rules: list[dict] = []
    used: set[int] = {MARKER_COLOR}
    for src_c in source_colors:
        tgt_choices = [c for c in PAIR_COLORS if c != src_c]
        tgt_c = random.choice(tgt_choices)
        used.update((int(src_c), int(tgt_c)))
        local_pattern, touch_rel = _make_stencil(min_s, max_s)
        rules.append(
            {
                "src": int(src_c),
                "tgt": int(tgt_c),
                "local_pattern": local_pattern,
                "touch_rel": touch_rel,
                "n_sc": random.randint(SCATTERED_MIN, sc_max),
            }
        )
    return rules, used


def _seed_variants(n_sc: int) -> list[tuple[tuple[int, int], bool]]:
    dirs = _DIRS4[:]
    random.shuffle(dirs)
    variants = [(dirs[0], False), (dirs[1], True)]
    pool = [(d, f) for d in _DIRS4 for f in (False, True)]
    random.shuffle(pool)
    for item in pool:
        if item not in variants:
            variants.append(item)
        if len(variants) >= n_sc:
            break
    return variants[:n_sc]


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
            tr = sr + dr
            tc = sc + dc
            pr = tr + dr
            pc = tc + dc
            if not (0 <= tr < h and 0 <= tc < w and 0 <= pr < h and 0 <= pc < w):
                continue

            forward = (-int(dr), -int(dc))
            probe_offset = _local_offset(rule["touch_rel"], forward, False)
            marker = (int(pr) - int(probe_offset[0]), int(pc) - int(probe_offset[1]))
            mr = int(marker[0])
            mc = int(marker[1])
            if not (0 <= mr < h and 0 <= mc < w):
                continue

            teal_cells = {
                tuple(cell)
                for cell in _apply_local_pattern(
                    rule["local_pattern"],
                    anchor=marker,
                    forward=forward,
                    flip=False,
                )
            }
            if (int(pr), int(pc)) not in teal_cells:
                continue
            reverse_probe = (int(sr - dr), int(sc - dc))
            if reverse_probe in teal_cells:
                continue

            template_cells = set(teal_cells) | {(mr, mc)}
            all_cells = {(int(sr), int(sc)), (int(tr), int(tc))} | template_cells
            if any(not (0 <= r < h and 0 <= c < w) for r, c in all_cells):
                continue
            if all_cells & (occupied | reserved):
                continue

            inp[sr, sc] = int(rule["src"])
            inp[tr, tc] = int(rule["tgt"])
            inp[mr, mc] = MARKER_COLOR
            for r, c in teal_cells:
                inp[r, c] = PATTERN_COLOR
            occupied.update(all_cells)
            _reserve_neighbors(reserved, all_cells)
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
        seeds: list[dict] = []
        for forward, flip in _seed_variants(rule["n_sc"]):
            side = _right(forward) if flip else _left(forward)
            placed = False
            for _ in range(SCATTERED_RETRIES_PER_SEED):
                sr = random.randint(0, h - 1)
                sc = random.randint(0, w - 1)
                center = (int(sr), int(sc))
                src_arm = (int(sr + forward[0]), int(sc + forward[1]))
                tgt_arm = (int(sr + side[0]), int(sc + side[1]))
                cells = {center, src_arm, tgt_arm}
                if any(not (0 <= r < h and 0 <= c < w) for r, c in cells):
                    continue
                if cells & (occupied | reserved):
                    continue

                stamp = {
                    tuple(cell)
                    for cell in _apply_local_pattern(
                        rule["local_pattern"],
                        anchor=center,
                        forward=forward,
                        flip=flip,
                    )
                }
                if any(not (0 <= r < h and 0 <= c < w) for r, c in stamp):
                    continue
                if stamp & future_output:
                    continue
                if stamp & occupied:
                    continue

                inp[center] = int(rule["src"])
                inp[src_arm] = int(rule["src"])
                inp[tgt_arm] = int(rule["tgt"])
                occupied.update(cells)
                _reserve_neighbors(reserved, cells)
                future_output.update(stamp)
                seeds.append(
                    {
                        "center": center,
                        "forward": forward,
                        "flip": bool(flip),
                    }
                )
                placed = True
                break
            if not placed:
                return False
        rule["seeds"] = seeds
    return True


def _place_bystanders(
    used_colors: set[int],
    inp: np.ndarray,
    occupied: set[tuple[int, int]],
    reserved: set[tuple[int, int]],
    future_output: set[tuple[int, int]],
) -> list[tuple[int, int, int]]:
    bystander_colors = [c for c in PAIR_COLORS if c not in used_colors]
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
            _reserve_neighbors(reserved, {(int(r), int(c))})
            break
    return placed


def _build_output(rules: list[dict], bystanders: list[tuple[int, int, int]], h: int, w: int) -> np.ndarray:
    out = np.zeros((h, w), dtype=int)
    for r, c, color in bystanders:
        out[int(r), int(c)] = int(color)

    for rule in rules:
        for seed in rule["seeds"]:
            stamp = _apply_local_pattern(
                rule["local_pattern"],
                anchor=seed["center"],
                forward=seed["forward"],
                flip=seed["flip"],
            )
            out[stamp[:, 0], stamp[:, 1]] = int(rule["tgt"])
    return out


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
