from __future__ import annotations

from collections import Counter, deque

import numpy as np


BG = 0
GREY = 5
ARROW_COLORS = (1, 2, 3, 4, 6, 7, 8, 9)
DIRS = ((-1, 0), (1, 0), (0, -1), (0, 1))
VERTICAL_DIRS = ((-1, 0), (1, 0))
HORIZONTAL_DIRS = ((0, -1), (0, 1))
GEN_ATTEMPTS = 400
PLACE_ATTEMPTS = 250
STANDARD_ARM = 1
STANDARD_STEM = 2


def main(grid: np.ndarray) -> np.ndarray:
    """
    Cast a color ray from each arrow tip and paint ray crossings grey.

    Input structure:
    - The background is black and the foreground is split into separated monochrome arrow objects.
    - Each arrow is a 4-connected T-shaped component: one junction pixel, two opposite side arms, and one stem ending at a single tip.
    - Arrow colors are distinct from the background and from grey, and arrows do not touch each other.

    Rule:
    - Keep every arrow in place.
    - For each arrow, infer its pointing direction from the unique stem that leaves the T junction.
    - Starting one cell beyond the tip, extend a straight ray in that direction for as many cells as the arrow's pixel area, stopping early at the grid boundary.
    - Paint empty ray cells in the arrow color; whenever two or more rays occupy the same cell, paint that cell grey.

    Key insight:
    - The ray length is not a memorized constant: it is the area of the arrow component, so the solver must identify each arrow as an object and count its pixels after recovering its direction.
    """
    g = np.asarray(grid, dtype=int)
    assert g.ndim == 2, "expected a 2D grid"
    bg = _get_bg(g)

    arrows = []
    for comp in _components(g, bg):
        arrows.append(_infer_arrow(comp["coords"], int(comp["color"])))
    arrows.sort(key=lambda arrow: (arrow["tip"][0], arrow["tip"][1], arrow["color"]))

    out = g.copy()
    protected = np.zeros_like(g, dtype=bool)
    for arrow in arrows:
        for r, c in arrow["cells"]:
            protected[r, c] = True

    owner = np.full(g.shape, -1, dtype=int)
    for idx, arrow in enumerate(arrows):
        tip_r, tip_c = arrow["tip"]
        dr, dc = arrow["direction"]
        for step in range(1, int(arrow["area"]) + 1):
            r = tip_r + dr * step
            c = tip_c + dc * step
            if not (0 <= r < g.shape[0] and 0 <= c < g.shape[1]):
                break
            if protected[r, c]:
                continue
            if owner[r, c] == -1:
                owner[r, c] = idx
                out[r, c] = int(arrow["color"])
            elif owner[r, c] != idx:
                owner[r, c] = -2
                out[r, c] = GREY

    return out


def generate_input() -> np.ndarray:
    rng = np.random.default_rng()
    for _ in range(GEN_ATTEMPTS):
        grid = _gen_attempt(rng)
        if grid is None:
            continue
        try:
            out = main(grid)
        except AssertionError:
            continue
        if np.array_equal(out, grid):
            continue
        return grid
    raise RuntimeError("failed to generate a valid input")


def _get_bg(grid: np.ndarray) -> int:
    colors, counts = np.unique(grid, return_counts=True)
    return int(colors[np.argmax(counts)])


def _components(grid: np.ndarray, bg: int) -> list[dict[str, object]]:
    h, w = grid.shape
    visited = np.zeros((h, w), dtype=bool)
    comps: list[dict[str, object]] = []
    for r in range(h):
        for c in range(w):
            if visited[r, c] or int(grid[r, c]) == bg:
                continue
            color = int(grid[r, c])
            q = deque([(r, c)])
            visited[r, c] = True
            coords: list[tuple[int, int]] = []
            while q:
                cr, cc = q.popleft()
                coords.append((int(cr), int(cc)))
                for nr, nc in _neighbors4(cr, cc):
                    if 0 <= nr < h and 0 <= nc < w and not visited[nr, nc] and int(grid[nr, nc]) == color:
                        visited[nr, nc] = True
                        q.append((nr, nc))
            comps.append({"color": color, "coords": coords})
    return comps


def _neighbors4(r: int, c: int) -> tuple[tuple[int, int], ...]:
    return ((r - 1, c), (r + 1, c), (r, c - 1), (r, c + 1))


def _infer_arrow(coords: list[tuple[int, int]], color: int) -> dict[str, object]:
    cells = {tuple(map(int, cell)) for cell in coords}
    degrees = {
        cell: sum((nbr in cells) for nbr in _neighbors4(*cell))
        for cell in cells
    }
    edge_count = sum(degrees.values()) // 2
    assert edge_count == len(cells) - 1, "arrow must be a tree"

    junctions = [cell for cell, degree in degrees.items() if degree == 3]
    leaves = [cell for cell, degree in degrees.items() if degree == 1]
    assert len(junctions) == 1, "expected one T junction"
    assert len(leaves) == 3, "expected three T endpoints"

    junction = junctions[0]
    jr, jc = junction

    horizontal_leaves = [leaf for leaf in leaves if leaf[0] == jr and leaf[1] != jc]
    if len(horizontal_leaves) == 2:
        cross_cols = sorted(leaf[1] for leaf in horizontal_leaves)
        assert cross_cols[0] < jc < cross_cols[1], "junction must sit between the side leaves"
        tip = next(leaf for leaf in leaves if leaf not in horizontal_leaves)
        assert tip[1] == jc and tip[0] != jr, "tip must continue the stem"
        direction = (_sign(tip[0] - jr), 0)
        _assert_t_shape(cells, junction, horizontal_leaves, tip)
        return {
            "color": color,
            "cells": cells,
            "junction": junction,
            "tip": tip,
            "direction": direction,
            "area": len(cells),
        }

    vertical_leaves = [leaf for leaf in leaves if leaf[1] == jc and leaf[0] != jr]
    assert len(vertical_leaves) == 2, "expected vertical side leaves for a horizontal arrow"
    cross_rows = sorted(leaf[0] for leaf in vertical_leaves)
    assert cross_rows[0] < jr < cross_rows[1], "junction must sit between the side leaves"
    tip = next(leaf for leaf in leaves if leaf not in vertical_leaves)
    assert tip[0] == jr and tip[1] != jc, "tip must continue the stem"
    direction = (0, _sign(tip[1] - jc))
    _assert_t_shape(cells, junction, vertical_leaves, tip)
    return {
        "color": color,
        "cells": cells,
        "junction": junction,
        "tip": tip,
        "direction": direction,
        "area": len(cells),
    }


def _assert_t_shape(
    cells: set[tuple[int, int]],
    junction: tuple[int, int],
    cross_leaves: list[tuple[int, int]],
    tip: tuple[int, int],
) -> None:
    jr, jc = junction
    if tip[1] == jc:
        c0 = min(leaf[1] for leaf in cross_leaves)
        c1 = max(leaf[1] for leaf in cross_leaves)
        step = _sign(tip[0] - jr)
        expected = {(jr, c) for c in range(c0, c1 + 1)}
        for r in range(jr, tip[0] + step, step):
            expected.add((r, jc))
    else:
        r0 = min(leaf[0] for leaf in cross_leaves)
        r1 = max(leaf[0] for leaf in cross_leaves)
        step = _sign(tip[1] - jc)
        expected = {(r, jc) for r in range(r0, r1 + 1)}
        for c in range(jc, tip[1] + step, step):
            expected.add((jr, c))
    assert cells == expected, "component must be a straight T shape"


def _sign(value: int) -> int:
    assert value != 0
    return 1 if value > 0 else -1


def _gen_attempt(rng: np.random.Generator) -> np.ndarray | None:
    h = int(rng.integers(12, 29))
    w = int(rng.integers(12, 29))
    want_intersection = bool(rng.integers(0, 2))
    want_truncation = bool(rng.integers(0, 2))
    min_arrows = 2 + int(want_truncation and want_intersection)
    n_arrows = int(rng.integers(min_arrows, 6))
    colors = rng.choice(np.array(ARROW_COLORS, dtype=int), size=n_arrows, replace=False).tolist()

    specs: list[dict[str, object]] = []
    color_idx = 0

    if want_intersection:
        pair = _sample_intersection_pair(rng, colors[color_idx], colors[color_idx + 1], h, w)
        if pair is None:
            return None
        specs.extend(pair)
        color_idx += 2

    if want_truncation:
        if not _append_arrow(
            rng,
            specs,
            colors[color_idx],
            h,
            w,
            require_truncation=True,
            avoid_truncation=False,
            target_intersections=1 if want_intersection else 0,
        ):
            return None
        color_idx += 1

    while len(specs) < n_arrows:
        if not _append_arrow(
            rng,
            specs,
            colors[color_idx],
            h,
            w,
            require_truncation=False,
            avoid_truncation=not want_truncation,
            target_intersections=1 if want_intersection else 0,
        ):
            return None
        color_idx += 1

    stats = _scene_stats(specs, h, w)
    if want_intersection:
        if len(stats["intersections"]) != 1:
            return None
    elif stats["intersections"]:
        return None

    if want_truncation:
        if int(stats["truncated_count"]) == 0:
            return None
    elif int(stats["truncated_count"]) != 0:
        return None

    if bool(stats["ray_hits_arrow"]):
        return None

    grid = np.full((h, w), BG, dtype=int)
    for spec in specs:
        color = int(spec["color"])
        for r, c in spec["cells"]:
            grid[r, c] = color
    return grid


def _sample_intersection_pair(
    rng: np.random.Generator,
    color_a: int,
    color_b: int,
    h: int,
    w: int,
) -> list[dict[str, object]] | None:
    for _ in range(PLACE_ATTEMPTS):
        cross_r = int(rng.integers(4, h - 4))
        cross_c = int(rng.integers(4, w - 4))
        vertical = VERTICAL_DIRS[int(rng.integers(0, len(VERTICAL_DIRS)))]
        horizontal = HORIZONTAL_DIRS[int(rng.integers(0, len(HORIZONTAL_DIRS)))]
        step_a = int(rng.integers(1, 6))
        step_b = int(rng.integers(1, 6))

        tip_a = (cross_r - vertical[0] * step_a, cross_c - vertical[1] * step_a)
        tip_b = (cross_r - horizontal[0] * step_b, cross_c - horizontal[1] * step_b)
        spec_a = _spec_from_tip(color_a, tip_a, vertical, STANDARD_ARM, STANDARD_STEM)
        spec_b = _spec_from_tip(color_b, tip_b, horizontal, STANDARD_ARM, STANDARD_STEM)
        if spec_a is None or spec_b is None:
            continue
        if not _cells_in_bounds(spec_a["cells"], h, w):
            continue
        if not _cells_in_bounds(spec_b["cells"], h, w):
            continue
        if _available_steps(spec_a["tip"], spec_a["direction"], h, w) < int(spec_a["area"]):
            continue
        if _available_steps(spec_b["tip"], spec_b["direction"], h, w) < int(spec_b["area"]):
            continue
        if not _arrows_separated(spec_a, [spec_b], h, w):
            continue
        stats = _scene_stats([spec_a, spec_b], h, w)
        if len(stats["intersections"]) != 1:
            continue
        if int(stats["truncated_count"]) != 0:
            continue
        if bool(stats["ray_hits_arrow"]):
            continue
        return [spec_a, spec_b]
    return None


def _append_arrow(
    rng: np.random.Generator,
    specs: list[dict[str, object]],
    color: int,
    h: int,
    w: int,
    require_truncation: bool,
    avoid_truncation: bool,
    target_intersections: int,
) -> bool:
    for _ in range(PLACE_ATTEMPTS):
        spec = _sample_arrow_spec(rng, color, h, w, require_truncation, avoid_truncation)
        if spec is None:
            continue
        if not _arrows_separated(spec, specs, h, w):
            continue
        trial = specs + [spec]
        stats = _scene_stats(trial, h, w)
        if len(stats["intersections"]) != target_intersections:
            continue
        if bool(stats["ray_hits_arrow"]):
            continue
        specs.append(spec)
        return True
    return False


def _sample_arrow_spec(
    rng: np.random.Generator,
    color: int,
    h: int,
    w: int,
    require_truncation: bool,
    avoid_truncation: bool,
) -> dict[str, object] | None:
    for _ in range(PLACE_ATTEMPTS):
        direction = DIRS[int(rng.integers(0, len(DIRS)))]
        center = (int(rng.integers(1, h - 1)), int(rng.integers(1, w - 1)))
        spec = _spec_from_center(color, center, direction, STANDARD_ARM, STANDARD_STEM)
        if not _cells_in_bounds(spec["cells"], h, w):
            continue
        available = _available_steps(spec["tip"], spec["direction"], h, w)
        if available < 1:
            continue
        if require_truncation and not (1 <= available < int(spec["area"])):
            continue
        if avoid_truncation and available < int(spec["area"]):
            continue
        return spec
    return None


def _spec_from_center(
    color: int,
    center: tuple[int, int],
    direction: tuple[int, int],
    arm_len: int,
    stem_len: int,
) -> dict[str, object]:
    dr, dc = direction
    pr, pc = -dc, dr
    cells = {(int(center[0]), int(center[1]))}
    for offset in range(1, arm_len + 1):
        cells.add((int(center[0] + pr * offset), int(center[1] + pc * offset)))
        cells.add((int(center[0] - pr * offset), int(center[1] - pc * offset)))
    for offset in range(1, stem_len + 1):
        cells.add((int(center[0] + dr * offset), int(center[1] + dc * offset)))
    tip = (int(center[0] + dr * stem_len), int(center[1] + dc * stem_len))
    return {
        "color": int(color),
        "center": (int(center[0]), int(center[1])),
        "direction": (int(dr), int(dc)),
        "arm_len": int(arm_len),
        "stem_len": int(stem_len),
        "tip": tip,
        "cells": cells,
        "area": len(cells),
    }


def _spec_from_tip(
    color: int,
    tip: tuple[int, int],
    direction: tuple[int, int],
    arm_len: int,
    stem_len: int,
) -> dict[str, object] | None:
    dr, dc = direction
    center = (int(tip[0] - dr * stem_len), int(tip[1] - dc * stem_len))
    return _spec_from_center(color, center, direction, arm_len, stem_len)


def _cells_in_bounds(cells: set[tuple[int, int]], h: int, w: int) -> bool:
    return all(0 <= r < h and 0 <= c < w for r, c in cells)


def _available_steps(tip: tuple[int, int], direction: tuple[int, int], h: int, w: int) -> int:
    r, c = tip
    dr, dc = direction
    if dr == -1:
        return r
    if dr == 1:
        return h - 1 - r
    if dc == -1:
        return c
    return w - 1 - c


def _arrows_separated(
    candidate: dict[str, object],
    others: list[dict[str, object]],
    h: int,
    w: int,
) -> bool:
    occupied = set()
    buffered = set()
    for spec in others:
        for r, c in spec["cells"]:
            occupied.add((int(r), int(c)))
            for rr in range(r - 1, r + 2):
                for cc in range(c - 1, c + 2):
                    if 0 <= rr < h and 0 <= cc < w:
                        buffered.add((int(rr), int(cc)))
    for r, c in candidate["cells"]:
        if (int(r), int(c)) in buffered or (int(r), int(c)) in occupied:
            return False
    return True


def _scene_stats(specs: list[dict[str, object]], h: int, w: int) -> dict[str, object]:
    protected = set()
    for spec in specs:
        protected.update(spec["cells"])

    intersections: set[tuple[int, int]] = set()
    occupancy: dict[tuple[int, int], int] = {}
    truncated_count = 0
    ray_hits_arrow = False

    for idx, spec in enumerate(specs):
        tip_r, tip_c = spec["tip"]
        dr, dc = spec["direction"]
        truncated = False
        for step in range(1, int(spec["area"]) + 1):
            r = tip_r + dr * step
            c = tip_c + dc * step
            if not (0 <= r < h and 0 <= c < w):
                truncated = True
                break
            pos = (int(r), int(c))
            if pos in protected:
                ray_hits_arrow = True
                continue
            prev = occupancy.get(pos)
            if prev is None:
                occupancy[pos] = idx
            elif prev != idx:
                intersections.add(pos)
                occupancy[pos] = -1
        if truncated:
            truncated_count += 1

    return {
        "intersections": intersections,
        "truncated_count": truncated_count,
        "ray_hits_arrow": ray_hits_arrow,
    }
