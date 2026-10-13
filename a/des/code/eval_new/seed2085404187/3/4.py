from __future__ import annotations

import random
from typing import Iterable

import numpy as np


BG = 0
GREY = 5
PALETTE = (1, 2, 3, 4, 6, 7, 8, 9)
GEN_RETRIES = 400

DIR4 = [(-1, 0), (1, 0), (0, -1), (0, 1)]
DIR8 = DIR4 + [(-1, -1), (-1, 1), (1, -1), (1, 1)]
DIR_BY_NAME = {
    "up": (-1, 0),
    "down": (1, 0),
    "left": (0, -1),
    "right": (0, 1),
}

ARROW_PATTERNS = {
    frozenset({(1, 0), (0, 1), (2, 1)}): ("right", (1, 0), ((0, 1), (2, 1))),
    frozenset({(0, 0), (2, 0), (1, 1)}): ("left", (1, 1), ((0, 0), (2, 0))),
    frozenset({(0, 0), (0, 2), (1, 1)}): ("up", (1, 1), ((0, 0), (0, 2))),
    frozenset({(0, 1), (1, 0), (1, 2)}): ("down", (0, 1), ((1, 0), (1, 2))),
}

BASE_SHAPES = (
    ((0, 0), (0, 1), (1, 0), (1, 1)),
    ((0, 0), (0, 1), (0, 2), (1, 0), (1, 1), (1, 2)),
    ((0, 0), (0, 1), (1, 0), (1, 1), (2, 0), (2, 1)),
    ((0, 0), (1, 0), (2, 0), (2, 1)),
    ((0, 0), (0, 1), (1, 0), (2, 0)),
    ((0, 1), (1, 0), (1, 1), (1, 2)),
    ((0, 0), (0, 1), (1, 1), (1, 2)),
    ((1, 0), (0, 1), (1, 1), (2, 1), (1, 2)),
)


def _normalize_shape(shape: Iterable[tuple[int, int]]) -> tuple[tuple[int, int], ...]:
    pts = {(int(r), int(c)) for r, c in shape}
    r0 = min(r for r, _ in pts)
    c0 = min(c for _, c in pts)
    return tuple(sorted((r - r0, c - c0) for r, c in pts))


def _rot90_shape(shape: Iterable[tuple[int, int]]) -> tuple[tuple[int, int], ...]:
    pts = {(int(r), int(c)) for r, c in shape}
    w = max(c for _, c in pts) + 1
    return _normalize_shape((c, w - 1 - r) for r, c in pts)


def _flip_h_shape(shape: Iterable[tuple[int, int]]) -> tuple[tuple[int, int], ...]:
    pts = {(int(r), int(c)) for r, c in shape}
    w = max(c for _, c in pts) + 1
    return _normalize_shape((r, w - 1 - c) for r, c in pts)


def _flip_v_shape(shape: Iterable[tuple[int, int]]) -> tuple[tuple[int, int], ...]:
    pts = {(int(r), int(c)) for r, c in shape}
    h = max(r for r, _ in pts) + 1
    return _normalize_shape((h - 1 - r, c) for r, c in pts)


def _shape_dims(shape: Iterable[tuple[int, int]]) -> tuple[int, int]:
    pts = list(shape)
    return max(r for r, _ in pts) + 1, max(c for _, c in pts) + 1


def _bbox(points: Iterable[tuple[int, int]]) -> tuple[int, int, int, int]:
    pts = list(points)
    return (
        min(r for r, _ in pts),
        min(c for _, c in pts),
        max(r for r, _ in pts),
        max(c for _, c in pts),
    )


def _components(mask: np.ndarray, conn8: bool) -> list[list[tuple[int, int]]]:
    h, w = mask.shape
    seen = np.zeros((h, w), dtype=bool)
    dirs = DIR8 if conn8 else DIR4
    comps: list[list[tuple[int, int]]] = []
    for r in range(h):
        for c in range(w):
            if not bool(mask[r, c]) or bool(seen[r, c]):
                continue
            stack = [(r, c)]
            seen[r, c] = True
            comp: list[tuple[int, int]] = []
            while stack:
                rr, cc = stack.pop()
                comp.append((rr, cc))
                for dr, dc in dirs:
                    nr, nc = rr + dr, cc + dc
                    if 0 <= nr < h and 0 <= nc < w and bool(mask[nr, nc]) and not bool(seen[nr, nc]):
                        seen[nr, nc] = True
                        stack.append((nr, nc))
            comps.append(comp)
    return comps


def _extract_cargos(grid: np.ndarray) -> list[dict]:
    cargos: list[dict] = []
    for color in sorted(int(v) for v in np.unique(grid) if int(v) not in (BG, GREY)):
        comps = _components(grid == color, conn8=False)
        assert len(comps) == 1, "each cargo color must form exactly one 4-connected component"
        cells = {tuple(p) for p in comps[0]}
        r0, c0, r1, c1 = _bbox(cells)
        cargos.append(
            {
                "color": int(color),
                "cells": cells,
                "bbox": (r0, c0, r1, c1),
            }
        )
    cargos.sort(key=lambda cargo: (cargo["bbox"][0], cargo["bbox"][1]))
    return cargos


def _detect_arrows(grid: np.ndarray) -> list[dict]:
    arrows: list[dict] = []
    for comp in _components(grid == GREY, conn8=True):
        if len(comp) != 3:
            raise AssertionError("every grey marker must be a 3-pixel arrow")
        r0, c0, r1, c1 = _bbox(comp)
        norm = frozenset((r - r0, c - c0) for r, c in comp)
        assert norm in ARROW_PATTERNS, "unknown grey marker shape"
        direction, shaft_rel, tip_rels = ARROW_PATTERNS[norm]
        arrows.append(
            {
                "cells": {tuple(p) for p in comp},
                "bbox": (r0, c0, r1, c1),
                "dir": direction,
                "shaft": (r0 + shaft_rel[0], c0 + shaft_rel[1]),
                "tips": tuple((r0 + tr, c0 + tc) for tr, tc in tip_rels),
            }
        )
    arrows.sort(key=lambda arrow: (arrow["bbox"][0], arrow["bbox"][1]))
    return arrows


def _intervals_overlap(a0: int, a1: int, b0: int, b1: int) -> bool:
    return max(int(a0), int(b0)) <= min(int(a1), int(b1))


def _pair_arrows_with_cargos(arrows: list[dict], cargos: list[dict]) -> list[dict]:
    matches: list[dict] = []
    seen_colors: set[int] = set()
    for arrow in arrows:
        ar0, ac0, ar1, ac1 = arrow["bbox"]
        candidates: list[tuple[int, int, int, dict]] = []
        for cargo in cargos:
            r0, c0, r1, c1 = cargo["bbox"]
            if arrow["dir"] == "right":
                gap = c0 - ac1 - 1
                ok = c0 > ac1 and _intervals_overlap(r0, r1, ar0, ar1)
                axis_dist = c0 - ac1
                cross_dist = abs((r0 + r1) - (ar0 + ar1))
            elif arrow["dir"] == "left":
                gap = ac0 - c1 - 1
                ok = c1 < ac0 and _intervals_overlap(r0, r1, ar0, ar1)
                axis_dist = ac0 - c1
                cross_dist = abs((r0 + r1) - (ar0 + ar1))
            elif arrow["dir"] == "down":
                gap = r0 - ar1 - 1
                ok = r0 > ar1 and _intervals_overlap(c0, c1, ac0, ac1)
                axis_dist = r0 - ar1
                cross_dist = abs((c0 + c1) - (ac0 + ac1))
            else:
                gap = ar0 - r1 - 1
                ok = r1 < ar0 and _intervals_overlap(c0, c1, ac0, ac1)
                axis_dist = ar0 - r1
                cross_dist = abs((c0 + c1) - (ac0 + ac1))
            if ok and gap in (1, 2):
                candidates.append((gap, axis_dist, cross_dist, cargo))
        assert candidates, "each arrow must have a unique cargo in front of it"
        candidates.sort(key=lambda item: (item[0], item[1], item[2], item[3]["bbox"][0], item[3]["bbox"][1]))
        gap, _, _, cargo = candidates[0]
        assert cargo["color"] not in seen_colors, "a cargo cannot be controlled by multiple arrows"
        seen_colors.add(int(cargo["color"]))
        matches.append(
            {
                "color": int(cargo["color"]),
                "cells": set(cargo["cells"]),
                "bbox": cargo["bbox"],
                "dir": arrow["dir"],
                "gap": int(gap),
            }
        )
    assert len(matches) == len(cargos) == len(arrows), "scene must be one arrow per cargo"
    return matches


def _can_shift(
    cells: set[tuple[int, int]],
    dr: int,
    dc: int,
    step: int,
    blockers: set[tuple[int, int]],
    h: int,
    w: int,
) -> bool:
    for r, c in cells:
        nr = r + dr * step
        nc = c + dc * step
        if nr < 0 or nr >= h or nc < 0 or nc >= w:
            return False
        if (nr, nc) in blockers:
            return False
    return True


def _simulate_moves(items: list[dict], h: int, w: int, order: list[int] | None = None) -> tuple[np.ndarray, dict[int, int]]:
    out = np.full((h, w), BG, dtype=int)
    blockers: set[tuple[int, int]] = set()
    by_idx = {idx: item for idx, item in enumerate(items)}
    for item in items:
        for r, c in item["cells"]:
            out[r, c] = int(item["color"])
            blockers.add((r, c))

    if order is None:
        move_order = sorted(
            range(len(items)),
            key=lambda idx: (items[idx]["gap"], items[idx]["bbox"][0], items[idx]["bbox"][1]),
        )
    else:
        move_order = list(order)

    steps_by_color: dict[int, int] = {}
    for idx in move_order:
        item = by_idx[idx]
        cells = set(item["cells"])
        blockers.difference_update(cells)
        dr, dc = DIR_BY_NAME[item["dir"]]
        step = 0
        while _can_shift(cells, dr, dc, step + 1, blockers, h, w):
            step += 1
        for t in range(step + 1):
            for r, c in cells:
                out[r + dr * t, c + dc * t] = int(item["color"])
        final_cells = {(r + dr * step, c + dc * step) for r, c in cells}
        blockers.update(final_cells)
        steps_by_color[int(item["color"])] = int(step)
    return out, steps_by_color


def main(grid: np.ndarray) -> np.ndarray:
    """
    Move each colored cargo along its grey command arrow and print its swept shape as a trail.

    Input structure:
    - The background is black.
    - Each non-grey color forms one connected cargo block, with colors unique across cargos.
    - Each cargo has one nearby grey 3-pixel arrow, read as an 8-connected chevron whose 2-pixel side is the arrowhead.
    - The arrow sits behind its cargo, with a visible gap of 1 or 2 cells before the cargo, and grid edges or other cargos can block motion.

    Rule:
    - Decode each grey chevron as one of four directions: `>`, `<`, `^`, or `v`.
    - Match every arrow to the unique cargo directly in front of it; the gap between arrowhead side and cargo is either 1 or 2.
    - Move all gap-1 cargos first and gap-2 cargos second; after a cargo stops, only its final position acts as a blocker for later cargos.
    - For each moving cargo, paint every translated copy of its shape from the start through the final stop, then remove all grey arrow pixels.

    Key insight:
    - The puzzle layers three cues: chevron orientation gives direction, the swept union of the cargo shape gives the trail pattern, and the small arrow-to-cargo gap encodes movement priority.
    """
    g = np.asarray(grid, dtype=int)
    h, w = g.shape
    cargos = _extract_cargos(g)
    arrows = _detect_arrows(g)
    items = _pair_arrows_with_cargos(arrows, cargos)
    out, _ = _simulate_moves(items, h, w)
    return out


def _random_shape(rng: random.Random) -> tuple[tuple[int, int], ...]:
    shape = _normalize_shape(rng.choice(BASE_SHAPES))
    for _ in range(rng.randrange(4)):
        shape = _rot90_shape(shape)
    if rng.random() < 0.5:
        shape = _flip_h_shape(shape)
    if rng.random() < 0.5:
        shape = _flip_v_shape(shape)
    return _normalize_shape(shape)


def _arrow_cells_for_bbox(bbox: tuple[int, int, int, int], direction: str, gap: int) -> set[tuple[int, int]]:
    r0, c0, r1, c1 = bbox
    if direction == "right":
        mid = (r0 + r1) // 2
        shaft_c = c0 - gap - 2
        return {(mid, shaft_c), (mid - 1, shaft_c + 1), (mid + 1, shaft_c + 1)}
    if direction == "left":
        mid = (r0 + r1) // 2
        head_c = c1 + gap + 1
        return {(mid - 1, head_c), (mid + 1, head_c), (mid, head_c + 1)}
    if direction == "down":
        mid = (c0 + c1) // 2
        shaft_r = r0 - gap - 2
        return {(shaft_r, mid), (shaft_r + 1, mid - 1), (shaft_r + 1, mid + 1)}
    mid = (c0 + c1) // 2
    head_r = r1 + gap + 1
    return {(head_r, mid - 1), (head_r, mid + 1), (head_r + 1, mid)}


def _render_specs(h: int, w: int, specs: list[dict]) -> np.ndarray | None:
    grid = np.full((h, w), BG, dtype=int)
    grey_cells: set[tuple[int, int]] = set()
    occupied: set[tuple[int, int]] = set()

    for spec in specs:
        color = int(spec["color"])
        top = int(spec["top"])
        left = int(spec["left"])
        cells = {(top + r, left + c) for r, c in spec["shape"]}
        bbox = _bbox(cells)
        arrow = _arrow_cells_for_bbox(bbox, spec["dir"], int(spec["gap"]))

        if any(r < 0 or r >= h or c < 0 or c >= w for r, c in cells | arrow):
            return None
        if occupied & cells:
            return None
        if occupied & arrow:
            return None
        for r, c in arrow:
            for dr, dc in DIR8:
                nr, nc = r + dr, c + dc
                if (nr, nc) in grey_cells:
                    return None

        for r, c in cells:
            grid[r, c] = color
        for r, c in arrow:
            grid[r, c] = GREY

        occupied.update(cells)
        occupied.update(arrow)
        grey_cells.update(arrow)

    return grid


def _spec_to_item(spec: dict) -> dict:
    top = int(spec["top"])
    left = int(spec["left"])
    cells = {(top + r, left + c) for r, c in spec["shape"]}
    return {
        "color": int(spec["color"]),
        "cells": cells,
        "bbox": _bbox(cells),
        "dir": spec["dir"],
        "gap": int(spec["gap"]),
        "role": spec["role"],
    }


def _choose_pair_positions(
    rng: random.Random,
    h: int,
    w: int,
    shape_a: tuple[tuple[int, int], ...],
    shape_b: tuple[tuple[int, int], ...],
) -> tuple[int, int, int] | None:
    ha, wa = _shape_dims(shape_a)
    hb, wb = _shape_dims(shape_b)
    band_h = max(ha, hb)
    top_hi = h - band_h - 1
    if top_hi < 1:
        return None
    top = rng.randint(1, top_hi)

    leader_right_margin = rng.randint(1, 3)
    leader_left = w - wa - leader_right_margin
    if leader_left < 7:
        return None

    sep_min = max(3, rng.randint(3, 6))
    follower_left_hi = leader_left - wb - sep_min
    if follower_left_hi < 4:
        return None
    follower_left = rng.randint(4, follower_left_hi)
    return top, leader_left, follower_left


def _add_optional_extra(
    rng: random.Random,
    h: int,
    w: int,
    specs: list[dict],
    used_colors: set[int],
) -> list[dict]:
    if rng.random() < 0.5:
        return specs

    shape = _random_shape(rng)
    hh, ww = _shape_dims(shape)
    band1_top = min(int(spec["top"]) for spec in specs)
    band1_bot = max(int(spec["top"]) + _shape_dims(spec["shape"])[0] - 1 for spec in specs)

    top_candidates: list[int] = []
    for top in range(1, h - hh):
        bottom = top + hh - 1
        if bottom < band1_top - 3 or top > band1_bot + 3:
            top_candidates.append(top)
    if not top_candidates:
        return specs

    left_hi = w - ww - rng.randint(1, 3)
    if left_hi < 4:
        return specs

    color = rng.choice([c for c in PALETTE if c not in used_colors])
    extra = {
        "role": "extra",
        "shape": shape,
        "top": rng.choice(top_candidates),
        "left": rng.randint(4, left_hi),
        "gap": rng.choice((1, 2)),
        "dir": "right",
        "color": color,
    }
    specs2 = specs + [extra]
    return specs2 if _render_specs(h, w, specs2) is not None else specs


def _rotate_scene(grid: np.ndarray, turns: int) -> np.ndarray:
    return np.rot90(grid, int(turns))


def _gen_attempt(rng: random.Random) -> np.ndarray | None:
    h = rng.randint(12, 20)
    w = rng.randint(16, 24)

    colors = list(rng.sample(PALETTE, 3))
    shape_leader = _random_shape(rng)
    shape_follower = _random_shape(rng)
    pos = _choose_pair_positions(rng, h, w, shape_leader, shape_follower)
    if pos is None:
        return None
    top, leader_left, follower_left = pos

    specs = [
        {
            "role": "leader",
            "shape": shape_leader,
            "top": top,
            "left": leader_left,
            "gap": 1,
            "dir": "right",
            "color": colors[0],
        },
        {
            "role": "follower",
            "shape": shape_follower,
            "top": top,
            "left": follower_left,
            "gap": 2,
            "dir": "right",
            "color": colors[1],
        },
    ]
    specs = _add_optional_extra(rng, h, w, specs, {colors[0], colors[1]})

    grid = _render_specs(h, w, specs)
    if grid is None:
        return None

    items = [_spec_to_item(spec) for spec in specs]
    correct_out, steps = _simulate_moves(items, h, w)
    if steps[colors[0]] <= 0 or steps[colors[1]] <= 0:
        return None

    leader_idx = next(i for i, item in enumerate(items) if item["role"] == "leader")
    follower_idx = next(i for i, item in enumerate(items) if item["role"] == "follower")
    other_idxs = [i for i in range(len(items)) if i not in (leader_idx, follower_idx)]
    wrong_order = [follower_idx, leader_idx] + sorted(
        other_idxs,
        key=lambda idx: (items[idx]["gap"], items[idx]["bbox"][0], items[idx]["bbox"][1]),
    )
    wrong_out, _ = _simulate_moves(items, h, w, order=wrong_order)
    if np.array_equal(correct_out, wrong_out):
        return None

    turns = rng.randrange(4)
    grid = _rotate_scene(grid, turns)

    parsed = _pair_arrows_with_cargos(_detect_arrows(grid), _extract_cargos(grid))
    gaps = sorted(item["gap"] for item in parsed)
    if gaps[:2] != [1, 2]:
        return None

    out = main(grid)
    if out.shape != grid.shape or max(out.shape) > 30:
        return None
    return grid


def generate_input() -> np.ndarray:
    rng = random.Random()
    for _ in range(GEN_RETRIES):
        grid = _gen_attempt(rng)
        if grid is not None:
            return grid
    raise RuntimeError("generate_input: failed to create a valid non-degenerate sample")
