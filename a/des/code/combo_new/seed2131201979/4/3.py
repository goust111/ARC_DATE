from __future__ import annotations

import random

import numpy as np


DIRS = {
    "up": (-1, 0),
    "down": (1, 0),
    "left": (0, -1),
    "right": (0, 1),
}

ARROW_PATTERNS = {
    "up": np.array(
        [
            [1, 1, 1],
            [0, 1, 0],
            [0, 1, 0],
        ],
        dtype=int,
    ),
    "down": np.array(
        [
            [0, 1, 0],
            [0, 1, 0],
            [1, 1, 1],
        ],
        dtype=int,
    ),
    "left": np.array(
        [
            [1, 0, 0],
            [1, 1, 1],
            [1, 0, 0],
        ],
        dtype=int,
    ),
    "right": np.array(
        [
            [0, 0, 1],
            [1, 1, 1],
            [0, 0, 1],
        ],
        dtype=int,
    ),
}

HEAD_CENTER_OFFSETS = {
    "up": (0, 1),
    "down": (2, 1),
    "left": (1, 0),
    "right": (1, 2),
}

ARROW_OFFSETS = {
    "up": [(0, -1), (0, 0), (0, 1), (1, 0), (2, 0)],
    "down": [(0, -1), (0, 0), (0, 1), (-1, 0), (-2, 0)],
    "left": [(-1, 0), (0, 0), (1, 0), (0, 1), (0, 2)],
    "right": [(-1, 0), (0, 0), (1, 0), (0, -1), (0, -2)],
}

GEN_ATTEMPTS = 500
GRID_MAX = 28
ARROW_COUNT_RANGE = (1, 4)
DIST_MIN = 1
DIST_MAX = 8


def _most_common_color(grid: np.ndarray) -> int:
    vals, cnts = np.unique(grid, return_counts=True)
    return int(vals[np.argmax(cnts)])


def _components(grid: np.ndarray, bg: int) -> list[dict]:
    h, w = grid.shape
    seen = np.zeros((h, w), dtype=bool)
    out: list[dict] = []

    for r in range(h):
        for c in range(w):
            if seen[r, c] or int(grid[r, c]) == bg:
                continue

            color = int(grid[r, c])
            stack = [(r, c)]
            seen[r, c] = True
            pts: list[tuple[int, int]] = []
            while stack:
                cr, cc = stack.pop()
                pts.append((int(cr), int(cc)))
                for nr, nc in ((cr - 1, cc), (cr + 1, cc), (cr, cc - 1), (cr, cc + 1)):
                    if 0 <= nr < h and 0 <= nc < w and not seen[nr, nc] and int(grid[nr, nc]) == color:
                        seen[nr, nc] = True
                        stack.append((nr, nc))

            coords = np.array(pts, dtype=int)
            r0, c0 = coords.min(axis=0)
            r1, c1 = coords.max(axis=0)
            out.append(
                {
                    "color": color,
                    "coords": coords,
                    "bbox": (int(r0), int(c0), int(r1), int(c1)),
                    "area": int(len(pts)),
                }
            )

    return out


def _detect_arrow(coords: np.ndarray) -> dict | None:
    if int(coords.shape[0]) != 5:
        return None

    r0, c0 = coords.min(axis=0)
    r1, c1 = coords.max(axis=0)
    if int(r1 - r0 + 1) != 3 or int(c1 - c0 + 1) != 3:
        return None

    patch = np.zeros((3, 3), dtype=int)
    patch[coords[:, 0] - int(r0), coords[:, 1] - int(c0)] = 1

    for direction, pattern in ARROW_PATTERNS.items():
        if np.array_equal(patch, pattern):
            dr, dc = HEAD_CENTER_OFFSETS[direction]
            return {
                "direction": direction,
                "head_center": (int(r0 + dr), int(c0 + dc)),
            }
    return None


def _find_anchor_for_arrow(arrow: dict, anchors: list[dict]) -> dict:
    hr, hc = arrow["head_center"]
    direction = arrow["direction"]
    matches: list[tuple[int, dict]] = []
    for anchor in anchors:
        ar, ac = anchor["pos"]
        if direction == "up" and ac == hc and ar < hr:
            matches.append((int(hr - ar), anchor))
        elif direction == "down" and ac == hc and ar > hr:
            matches.append((int(ar - hr), anchor))
        elif direction == "left" and ar == hr and ac < hc:
            matches.append((int(hc - ac), anchor))
        elif direction == "right" and ar == hr and ac > hc:
            matches.append((int(ac - hc), anchor))

    assert len(matches) == 1, "each arrow must face exactly one anchor on its center ray"
    return matches[0][1]


def _paint_rectangle(out: np.ndarray, direction: str, head_center: tuple[int, int], anchor_pos: tuple[int, int], color: int) -> None:
    hr, hc = head_center
    ar, ac = anchor_pos
    if direction == "up":
        out[ar : hr + 1, hc - 1 : hc + 2] = color
    elif direction == "down":
        out[hr : ar + 1, hc - 1 : hc + 2] = color
    elif direction == "left":
        out[hr - 1 : hr + 2, ac : hc + 1] = color
    else:
        out[hr - 1 : hr + 2, hc : ac + 1] = color


def main(grid: np.ndarray) -> np.ndarray:
    """
    Stretch each T-shaped arrow forward to its anchor and recolor the swept rectangle with the anchor color.

    Input structure:
    - The most common color is the background.
    - Every foreground component is either a one-pixel anchor or a five-pixel T-shaped arrow inside a 3x3 box.
    - An arrow's three-pixel bar marks its forward-facing head, and a two-pixel shaft extends backward from the head center.
    - Each arrow's forward center ray contains exactly one anchor, and anchor colors differ from arrow colors.

    Rule:
    - Detect every arrow and infer its orientation from which side of its 3x3 box contains the three-pixel head.
    - For each arrow, find the unique anchor lying on its forward center ray.
    - Erase the original arrow body.
    - Fill the 3-cell-wide rectangle from the arrow head to the anchor, inclusive, using the anchor's color.

    Key insight:
    - The T shape determines motion direction, but the final paint color is supplied by the separate anchor hit along that direction rather than by the arrow itself.
    """
    g = np.asarray(grid, dtype=int)
    assert g.ndim == 2, "expected a 2D grid"

    bg = _most_common_color(g)
    comps = _components(g, bg)
    assert comps, "expected arrows and anchors"

    arrows: list[dict] = []
    anchors: list[dict] = []
    for comp in comps:
        area = int(comp["area"])
        if area == 1:
            r, c = map(int, comp["coords"][0])
            anchors.append({"pos": (r, c), "color": int(comp["color"])})
            continue

        arrow = _detect_arrow(comp["coords"])
        assert arrow is not None, "every non-anchor foreground component must be a T-shaped arrow"
        arrow["coords"] = comp["coords"]
        arrow["color"] = int(comp["color"])
        arrows.append(arrow)

    assert arrows, "expected at least one arrow"
    assert anchors, "expected at least one anchor"

    out = g.copy()
    for arrow in arrows:
        coords = arrow["coords"]
        out[coords[:, 0], coords[:, 1]] = bg

    used_anchors: set[tuple[int, int]] = set()
    for arrow in arrows:
        anchor = _find_anchor_for_arrow(arrow, anchors)
        assert anchor["pos"] not in used_anchors, "anchors should be matched one-to-one with arrows"
        used_anchors.add(anchor["pos"])
        _paint_rectangle(out, arrow["direction"], arrow["head_center"], anchor["pos"], int(anchor["color"]))

    assert used_anchors == {anchor["pos"] for anchor in anchors}, "all anchors must be used"
    return out


def _arrow_cells(direction: str, head_center: tuple[int, int]) -> set[tuple[int, int]]:
    hr, hc = head_center
    return {(int(hr + dr), int(hc + dc)) for dr, dc in ARROW_OFFSETS[direction]}


def _anchor_pos(direction: str, head_center: tuple[int, int], dist: int) -> tuple[int, int]:
    hr, hc = head_center
    dr, dc = DIRS[direction]
    return int(hr + dr * dist), int(hc + dc * dist)


def _rectangle_cells(direction: str, head_center: tuple[int, int], anchor_pos: tuple[int, int]) -> set[tuple[int, int]]:
    hr, hc = head_center
    ar, ac = anchor_pos

    if direction == "up":
        return {(r, c) for r in range(ar, hr + 1) for c in range(hc - 1, hc + 2)}
    if direction == "down":
        return {(r, c) for r in range(hr, ar + 1) for c in range(hc - 1, hc + 2)}
    if direction == "left":
        return {(r, c) for r in range(hr - 1, hr + 2) for c in range(ac, hc + 1)}
    return {(r, c) for r in range(hr - 1, hr + 2) for c in range(hc, ac + 1)}


def _halo(cells: set[tuple[int, int]], h: int, w: int) -> set[tuple[int, int]] | None:
    halo: set[tuple[int, int]] = set()
    for r, c in cells:
        for dr in (-1, 0, 1):
            for dc in (-1, 0, 1):
                rr = int(r + dr)
                cc = int(c + dc)
                if not (0 <= rr < h and 0 <= cc < w):
                    return None
                halo.add((rr, cc))
    return halo


def _sample_head_and_dist(h: int, w: int, direction: str, rng: random.Random) -> tuple[tuple[int, int], int] | None:
    if direction in ("up", "down"):
        max_dist = min(DIST_MAX, h - 5)
    else:
        max_dist = min(DIST_MAX, w - 5)
    if max_dist < DIST_MIN:
        return None

    dist = rng.randint(DIST_MIN, max_dist)
    if direction == "up":
        hr = rng.randint(1 + dist, h - 4)
        hc = rng.randint(2, w - 3)
    elif direction == "down":
        hr = rng.randint(3, h - 2 - dist)
        hc = rng.randint(2, w - 3)
    elif direction == "left":
        hr = rng.randint(2, h - 3)
        hc = rng.randint(1 + dist, w - 4)
    else:
        hr = rng.randint(2, h - 3)
        hc = rng.randint(3, w - 2 - dist)

    return (int(hr), int(hc)), int(dist)


def _build_expected(grid: np.ndarray, placements: list[dict], bg: int) -> np.ndarray:
    out = np.array(grid, copy=True)
    for spec in placements:
        arrow_cells = np.array(sorted(spec["arrow_cells"]), dtype=int)
        out[arrow_cells[:, 0], arrow_cells[:, 1]] = bg
    for spec in placements:
        _paint_rectangle(out, spec["direction"], spec["head_center"], spec["anchor_pos"], int(spec["anchor_color"]))
    return out


def _gen_attempt(rng: random.Random) -> np.ndarray | None:
    arrow_count = rng.randint(*ARROW_COUNT_RANGE)
    min_side = 10 + 3 * arrow_count
    if min_side > GRID_MAX:
        return None

    h = rng.randint(min_side, GRID_MAX)
    w = rng.randint(min_side, GRID_MAX)
    all_colors = list(range(10))
    rng.shuffle(all_colors)
    bg = int(all_colors[0])
    arrow_colors = [int(c) for c in all_colors[1 : 1 + arrow_count]]
    anchor_colors = [int(c) for c in all_colors[1 + arrow_count : 1 + 2 * arrow_count]]

    directions = rng.sample(list(DIRS.keys()), arrow_count)
    grid = np.full((h, w), bg, dtype=int)
    reserved: set[tuple[int, int]] = set()
    anchor_rows: set[int] = set()
    anchor_cols: set[int] = set()
    placements: list[dict] = []

    for idx in range(arrow_count):
        direction = directions[idx]
        placed = False
        for _ in range(300):
            sampled = _sample_head_and_dist(h, w, direction, rng)
            if sampled is None:
                return None
            head_center, dist = sampled
            anchor_pos = _anchor_pos(direction, head_center, dist)
            if anchor_pos[0] in anchor_rows or anchor_pos[1] in anchor_cols:
                continue

            arrow_cells = _arrow_cells(direction, head_center)
            rect_cells = _rectangle_cells(direction, head_center, anchor_pos)
            base_cells = set(arrow_cells) | set(rect_cells) | {anchor_pos}
            halo = _halo(base_cells, h, w)
            if halo is None or halo & reserved:
                continue

            placements.append(
                {
                    "direction": direction,
                    "head_center": head_center,
                    "anchor_pos": anchor_pos,
                    "arrow_color": int(arrow_colors[idx]),
                    "anchor_color": int(anchor_colors[idx]),
                    "arrow_cells": arrow_cells,
                    "rect_cells": rect_cells,
                }
            )
            reserved |= halo
            anchor_rows.add(int(anchor_pos[0]))
            anchor_cols.add(int(anchor_pos[1]))
            placed = True
            break
        if not placed:
            return None

    for spec in placements:
        for r, c in spec["arrow_cells"]:
            grid[r, c] = int(spec["arrow_color"])
        ar, ac = spec["anchor_pos"]
        grid[ar, ac] = int(spec["anchor_color"])

    expected = _build_expected(grid, placements, bg)
    try:
        predicted = main(grid)
    except Exception:
        return None

    if not np.array_equal(predicted, expected):
        return None
    if np.array_equal(expected, grid):
        return None
    if _most_common_color(grid) != bg:
        return None

    comps = _components(grid, bg)
    if len(comps) != 2 * arrow_count:
        return None
    if sum(1 for comp in comps if int(comp["area"]) == 1) != arrow_count:
        return None
    if sum(1 for comp in comps if int(comp["area"]) == 5 and _detect_arrow(comp["coords"]) is not None) != arrow_count:
        return None

    return grid


def generate_input() -> np.ndarray:
    rng = random.Random()
    for _ in range(GEN_ATTEMPTS):
        grid = _gen_attempt(rng)
        if grid is not None:
            return grid
    raise RuntimeError("failed to generate a valid arrow-anchor ARC input")
