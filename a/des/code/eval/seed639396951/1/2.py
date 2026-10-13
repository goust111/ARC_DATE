from __future__ import annotations

from fractions import Fraction
import random

import numpy as np


BG = 0
GREY = 5
DIRS = {
    "up": (-1, 0),
    "down": (1, 0),
    "left": (0, -1),
    "right": (0, 1),
}
PERP = {
    "up": (0, 1),
    "down": (0, 1),
    "left": (1, 0),
    "right": (1, 0),
}
NEIGHBORS8 = [
    (-1, -1),
    (-1, 0),
    (-1, 1),
    (0, -1),
    (0, 1),
    (1, -1),
    (1, 0),
    (1, 1),
]


def _bbox(coords: set[tuple[int, int]]) -> tuple[int, int, int, int]:
    rows = [r for r, _c in coords]
    cols = [c for _r, c in coords]
    return min(rows), min(cols), max(rows), max(cols)


def _centroid(coords: set[tuple[int, int]]) -> tuple[Fraction, Fraction]:
    n = len(coords)
    return (
        Fraction(sum(r for r, _c in coords), n),
        Fraction(sum(c for _r, c in coords), n),
    )


def _shift_coords(
    coords: set[tuple[int, int]], dr: int, dc: int
) -> set[tuple[int, int]]:
    return {(r + dr, c + dc) for r, c in coords}


def _connected_components(mask: np.ndarray) -> list[set[tuple[int, int]]]:
    h, w = mask.shape
    seen = np.zeros((h, w), dtype=bool)
    comps: list[set[tuple[int, int]]] = []
    for r in range(h):
        for c in range(w):
            if not mask[r, c] or seen[r, c]:
                continue
            stack = [(r, c)]
            seen[r, c] = True
            comp: set[tuple[int, int]] = set()
            while stack:
                rr, cc = stack.pop()
                comp.add((rr, cc))
                for dr, dc in NEIGHBORS8:
                    nr = rr + dr
                    nc = cc + dc
                    if 0 <= nr < h and 0 <= nc < w and mask[nr, nc] and not seen[nr, nc]:
                        seen[nr, nc] = True
                        stack.append((nr, nc))
            comps.append(comp)
    return comps


def _decode_arrow(
    coords: set[tuple[int, int]]
) -> tuple[str, tuple[int, int]] | None:
    if not 3 <= len(coords) <= 5:
        return None

    matches: list[tuple[str, tuple[int, int]]] = []
    for tip in coords:
        tr, tc = tip
        for name, (dr, dc) in DIRS.items():
            pr, pc = PERP[name]
            forward_scores = [(r - tr) * dr + (c - tc) * dc for r, c in coords]
            if max(forward_scores) != 0 or forward_scores.count(0) != 1:
                continue

            near_wings = {
                (tr - dr + pr, tc - dc + pc),
                (tr - dr - pr, tc - dc - pc),
            }
            tail1 = (tr - dr, tc - dc)
            near_need = {tip, *near_wings}
            near_extra = coords - near_need
            if near_need.issubset(coords) and near_extra in [set(), {tail1}]:
                matches.append((name, tip))
                continue

            far_wings = {
                (tr - 2 * dr + pr, tc - 2 * dc + pc),
                (tr - 2 * dr - pr, tc - 2 * dc - pc),
            }
            tail2 = (tr - 2 * dr, tc - 2 * dc)
            far_need = {tip, tail1, tail2, *far_wings}
            if coords == far_need:
                matches.append((name, tip))

    if len(matches) != 1:
        return None
    return matches[0]


def _nearest_anchor(
    anchors: list[set[tuple[int, int]]],
    tip: tuple[int, int],
    direction: str,
) -> set[tuple[int, int]]:
    tr, tc = tip
    best: tuple[int, set[tuple[int, int]]] | None = None
    for anchor in anchors:
        r0, c0, r1, c1 = _bbox(anchor)
        if direction == "right" and c0 > tc and r0 <= tr <= r1:
            dist = c0 - tc
        elif direction == "left" and c1 < tc and r0 <= tr <= r1:
            dist = tc - c1
        elif direction == "down" and r0 > tr and c0 <= tc <= c1:
            dist = r0 - tr
        elif direction == "up" and r1 < tr and c0 <= tc <= c1:
            dist = tr - r1
        else:
            continue

        if best is None or dist < best[0]:
            best = (dist, anchor)

    assert best is not None, "each arrow must point to a grey anchor"
    return best[1]


def _place_shape_at_centroid(
    coords: set[tuple[int, int]],
    target_centroid: tuple[Fraction, Fraction],
) -> set[tuple[int, int]]:
    src_r, src_c = _centroid(coords)
    dr = target_centroid[0] - src_r
    dc = target_centroid[1] - src_c
    assert dr.denominator == 1 and dc.denominator == 1, "translation must be integral"
    return _shift_coords(coords, dr.numerator, dc.numerator)


def main(grid: np.ndarray) -> np.ndarray:
    """
    Group same-color fragments, decode which fragment is the arrow, and move the
    companion shape so its centroid lands on the arrow-indicated grey anchor.
    """
    g = np.asarray(grid, dtype=int)
    h, w = g.shape

    grey_comps = _connected_components(g == GREY)
    assert grey_comps, "expected at least one grey anchor"

    colors = sorted(int(x) for x in np.unique(g) if int(x) not in {BG, GREY})
    out = np.full((h, w), BG, dtype=int)

    for color in colors:
        comps = _connected_components(g == color)
        assert len(comps) == 2, "each color must appear in exactly two groups"

        decoded = []
        for comp in comps:
            arrow = _decode_arrow(comp)
            decoded.append((comp, arrow))

        arrow_candidates = [(comp, arrow) for comp, arrow in decoded if arrow is not None]
        assert len(arrow_candidates) == 1, "each color must have exactly one arrow group"

        arrow_comp, (direction, tip) = arrow_candidates[0]
        pattern_comp = comps[0] if comps[1] is arrow_comp else comps[1]

        anchor = _nearest_anchor(grey_comps, tip, direction)
        moved = _place_shape_at_centroid(pattern_comp, _centroid(anchor))

        for r, c in moved:
            assert 0 <= r < h and 0 <= c < w, "translated pattern must stay in bounds"
            assert out[r, c] == BG, "translated patterns must not overlap"
            out[r, c] = color

    return out


# --------------- generator ---------------

GRID_H_RANGE = (18, 31)
GRID_W_RANGE = (18, 31)
GROUP_COUNT_RANGE = (2, 4)
COLORS = [1, 2, 3, 4, 6, 7, 8, 9]
PAD = 1
PLACEMENT_ATTEMPTS = 150
GEN_ATTEMPTS = 400

INT_PATTERNS = [
    {(0, -1), (0, 0), (0, 1)},
    {(-1, -1), (0, 0), (1, 1)},
    {(-1, 0), (0, -1), (0, 1), (1, 0)},
    {(-1, -1), (-1, 0), (0, 0), (1, 0), (1, 1)},
    {(-1, -1), (-1, 1), (0, 0), (1, -1), (1, 1)},
    {(-1, -1), (0, -1), (0, 0), (0, 1), (1, 1)},
]
HALF_PATTERN = {(0, 0), (0, 1), (1, 0), (1, 1)}


def _ceil_fraction(x: Fraction) -> int:
    return -((-x.numerator) // x.denominator)


def _rotate_int_pattern(coords: set[tuple[int, int]]) -> set[tuple[int, int]]:
    return {(-c, r) for r, c in coords}


def _reflect_int_pattern(coords: set[tuple[int, int]]) -> set[tuple[int, int]]:
    return {(r, -c) for r, c in coords}


def _random_int_pattern() -> set[tuple[int, int]]:
    while True:
        coords = set(random.choice(INT_PATTERNS))
        for _ in range(random.randrange(4)):
            coords = _rotate_int_pattern(coords)
        if random.choice([False, True]):
            coords = _reflect_int_pattern(coords)
        if _decode_arrow(coords) is None:
            return coords


def _arrow_cells(
    tip: tuple[int, int], direction: str, arrow_size: int
) -> set[tuple[int, int]]:
    tr, tc = tip
    dr, dc = DIRS[direction]
    pr, pc = PERP[direction]
    if arrow_size == 3:
        return {
            tip,
            (tr - dr + pr, tc - dc + pc),
            (tr - dr - pr, tc - dc - pc),
        }
    if arrow_size == 4:
        return {
            tip,
            (tr - dr + pr, tc - dc + pc),
            (tr - dr - pr, tc - dc - pc),
            (tr - dr, tc - dc),
        }
    assert arrow_size == 5
    return {
        tip,
        (tr - dr, tc - dc),
        (tr - 2 * dr, tc - 2 * dc),
        (tr - 2 * dr + pr, tc - 2 * dc + pc),
        (tr - 2 * dr - pr, tc - 2 * dc - pc),
    }


def _anchor_cells(size: int) -> set[tuple[int, int]]:
    return {(r, c) for r in range(size) for c in range(size)}


def _choose_pattern_for_anchor(
    anchor_size: int,
) -> tuple[set[tuple[int, int]], tuple[Fraction, Fraction]]:
    if anchor_size == 1:
        coords = _random_int_pattern()
    else:
        coords = set(HALF_PATTERN)
    return coords, _centroid(coords)


def _build_group(color: int) -> dict | None:
    direction = random.choice(list(DIRS))
    anchor_size = random.choice([1, 2])
    pattern_coords, pattern_centroid = _choose_pattern_for_anchor(anchor_size)
    anchor_coords = _anchor_cells(anchor_size)
    anchor_centroid = _centroid(anchor_coords)

    sep_arrow = random.randint(2, 4)
    sep_pattern = random.randint(2, 4)
    arrow_size = random.choice([3, 4, 5])

    if direction in {"left", "right"}:
        lane = 0 if anchor_size == 1 else random.choice([0, 1])
        tip = (lane, -sep_arrow) if direction == "right" else (lane, anchor_size - 1 + sep_arrow)

        rel_cols = [Fraction(c) - pattern_centroid[1] for _r, c in pattern_coords]
        if direction == "right":
            min_rel = min(rel_cols)
            need = Fraction(anchor_size - 1 + sep_pattern) - anchor_centroid[1] - min_rel
            offset = _ceil_fraction(need) + random.randint(0, 2)
            input_target = (anchor_centroid[0], anchor_centroid[1] + offset)
        else:
            max_rel = max(rel_cols)
            need = anchor_centroid[1] + max_rel + sep_pattern
            offset = _ceil_fraction(need) + random.randint(0, 2)
            input_target = (anchor_centroid[0], anchor_centroid[1] - offset)
    else:
        lane = 0 if anchor_size == 1 else random.choice([0, 1])
        tip = (-sep_arrow, lane) if direction == "down" else (anchor_size - 1 + sep_arrow, lane)

        rel_rows = [Fraction(r) - pattern_centroid[0] for r, _c in pattern_coords]
        if direction == "down":
            min_rel = min(rel_rows)
            need = Fraction(anchor_size - 1 + sep_pattern) - anchor_centroid[0] - min_rel
            offset = _ceil_fraction(need) + random.randint(0, 2)
            input_target = (anchor_centroid[0] + offset, anchor_centroid[1])
        else:
            max_rel = max(rel_rows)
            need = anchor_centroid[0] + max_rel + sep_pattern
            offset = _ceil_fraction(need) + random.randint(0, 2)
            input_target = (anchor_centroid[0] - offset, anchor_centroid[1])

    arrow_coords = _arrow_cells(tip, direction, arrow_size)
    assert _decode_arrow(arrow_coords) is not None

    placed_pattern = _place_shape_at_centroid(pattern_coords, input_target)
    output_pattern = _place_shape_at_centroid(pattern_coords, anchor_centroid)

    if arrow_coords & anchor_coords:
        return None
    if arrow_coords & placed_pattern:
        return None
    if anchor_coords & placed_pattern:
        return None

    all_coords = arrow_coords | anchor_coords | placed_pattern | output_pattern
    min_r, min_c, max_r, max_c = _bbox(all_coords)
    shift_r = -min_r
    shift_c = -min_c

    arrow_coords = _shift_coords(arrow_coords, shift_r, shift_c)
    anchor_coords = _shift_coords(anchor_coords, shift_r, shift_c)
    placed_pattern = _shift_coords(placed_pattern, shift_r, shift_c)
    output_pattern = _shift_coords(output_pattern, shift_r, shift_c)

    return {
        "color": color,
        "direction": direction,
        "anchor_size": anchor_size,
        "input_cells": (
            [(r, c, color) for r, c in arrow_coords]
            + [(r, c, GREY) for r, c in anchor_coords]
            + [(r, c, color) for r, c in placed_pattern]
        ),
        "output_cells": [(r, c, color) for r, c in output_pattern],
        "shape": (max_r - min_r + 1, max_c - min_c + 1),
    }


def _bbox_region_clear(
    occupied: np.ndarray, r0: int, c0: int, h: int, w: int
) -> bool:
    rr0 = max(0, r0 - PAD)
    cc0 = max(0, c0 - PAD)
    rr1 = min(occupied.shape[0], r0 + h + PAD)
    cc1 = min(occupied.shape[1], c0 + w + PAD)
    return not occupied[rr0:rr1, cc0:cc1].any()


def _mark_bbox(occupied: np.ndarray, r0: int, c0: int, h: int, w: int) -> None:
    rr0 = max(0, r0 - PAD)
    cc0 = max(0, c0 - PAD)
    rr1 = min(occupied.shape[0], r0 + h + PAD)
    cc1 = min(occupied.shape[1], c0 + w + PAD)
    occupied[rr0:rr1, cc0:cc1] = True


def _assemble_scene(
    groups: list[dict], height: int, width: int
) -> tuple[np.ndarray, np.ndarray] | None:
    inp = np.full((height, width), BG, dtype=int)
    out = np.full((height, width), BG, dtype=int)
    occupied = np.zeros((height, width), dtype=bool)

    for group in sorted(groups, key=lambda g: g["shape"][0] * g["shape"][1], reverse=True):
        gh, gw = group["shape"]
        if gh > height or gw > width:
            return None

        placed = False
        for _ in range(PLACEMENT_ATTEMPTS):
            r0 = random.randint(0, height - gh)
            c0 = random.randint(0, width - gw)
            if not _bbox_region_clear(occupied, r0, c0, gh, gw):
                continue

            for r, c, value in group["input_cells"]:
                assert inp[r0 + r, c0 + c] == BG
                inp[r0 + r, c0 + c] = value
            for r, c, value in group["output_cells"]:
                assert out[r0 + r, c0 + c] == BG
                out[r0 + r, c0 + c] = value

            _mark_bbox(occupied, r0, c0, gh, gw)
            placed = True
            break

        if not placed:
            return None

    return inp, out


def _gen_attempt() -> np.ndarray | None:
    height = random.randint(*GRID_H_RANGE)
    width = random.randint(*GRID_W_RANGE)
    n_groups = random.randint(*GROUP_COUNT_RANGE)
    colors = random.sample(COLORS, n_groups)

    groups = []
    for color in colors:
        group = _build_group(color)
        if group is None:
            return None
        groups.append(group)

    scene = _assemble_scene(groups, height, width)
    if scene is None:
        return None
    inp, oracle = scene

    if np.array_equal(inp, oracle):
        return None
    if not np.array_equal(main(inp), oracle):
        return None
    return inp


def generate_input() -> np.ndarray:
    for _ in range(GEN_ATTEMPTS):
        sample = _gen_attempt()
        if sample is not None:
            return sample
    raise RuntimeError("failed to generate a valid ARC sample")
