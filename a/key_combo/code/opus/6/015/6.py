"""ARC task 6: reflection-gated ray emission from hole-forming composites."""

from __future__ import annotations

from collections import deque
import random

import numpy as np


BG = 0
RED = 2
GREEN = 3
BLUE = 1
INPUT_COLORS = (4, 5, 6, 7, 8, 9)
GEN_ATTEMPTS = 700
GROUP_PLACE_ATTEMPTS = 180


def _neighbors4(r: int, c: int) -> tuple[tuple[int, int], ...]:
    return ((r - 1, c), (r + 1, c), (r, c - 1), (r, c + 1))


def _bbox(coords: list[tuple[int, int]]) -> tuple[int, int, int, int]:
    rows = [r for r, _ in coords]
    cols = [c for _, c in coords]
    return min(rows), min(cols), max(rows), max(cols)


def _components(mask: np.ndarray) -> list[list[tuple[int, int]]]:
    h, w = mask.shape
    seen = np.zeros_like(mask, dtype=bool)
    comps: list[list[tuple[int, int]]] = []
    for r in range(h):
        for c in range(w):
            if not mask[r, c] or seen[r, c]:
                continue
            q = deque([(r, c)])
            seen[r, c] = True
            comp: list[tuple[int, int]] = []
            while q:
                cr, cc = q.popleft()
                comp.append((cr, cc))
                for nr, nc in _neighbors4(cr, cc):
                    if 0 <= nr < h and 0 <= nc < w and mask[nr, nc] and not seen[nr, nc]:
                        seen[nr, nc] = True
                        q.append((nr, nc))
            comps.append(sorted(comp))
    comps.sort(key=_bbox)
    return comps


def _rotate_coords(coords: tuple[tuple[int, int], ...], turns: int) -> tuple[tuple[int, int], ...]:
    out = list(coords)
    for _ in range(turns % 4):
        out = [(-c, r) for r, c in out]
    return tuple(sorted(out))


def _near_marker(payload: list[tuple[int, int]], marker: list[tuple[int, int]]) -> bool:
    for pr, pc in payload:
        for mr, mc in marker:
            if max(abs(pr - mr), abs(pc - mc)) <= 1:
                return True
    return False


def _marker_kind(marker: list[tuple[int, int]]) -> tuple[str, int]:
    if len(marker) == 1:
        return "point", 1
    rows = sorted({r for r, _ in marker})
    cols = sorted({c for _, c in marker})
    if len(rows) == 1:
        row = rows[0]
        ordered = sorted(c for _, c in marker)
        if ordered[-1] - ordered[0] + 1 == len(ordered):
            return "row", row
    if len(cols) == 1:
        col = cols[0]
        ordered = sorted(r for r, _ in marker)
        if ordered[-1] - ordered[0] + 1 == len(ordered):
            return "col", col
    raise ValueError("ambiguous red marker geometry")


def _reflect_payload(
    payload: list[tuple[int, int]],
    marker: list[tuple[int, int]],
) -> tuple[str, list[tuple[int, int]]]:
    kind, axis = _marker_kind(marker)
    if kind == "point":
        pr, pc = marker[0]
        return kind, [(2 * pr - r, 2 * pc - c) for r, c in payload]
    if kind == "row":
        return kind, [(2 * axis - r, c) for r, c in payload]
    return kind, [(r, 2 * axis - c) for r, c in payload]


def _holes_in_composite(
    composite: set[tuple[int, int]],
) -> list[list[tuple[int, int]]]:
    coords = list(composite)
    r0, c0, r1, c1 = _bbox(coords)
    h = r1 - r0 + 1
    w = c1 - c0 + 1
    occ = np.zeros((h, w), dtype=bool)
    for r, c in coords:
        occ[r - r0, c - c0] = True
    holes: list[list[tuple[int, int]]] = []
    for comp in _components(~occ):
        if any(rr == 0 or rr == h - 1 or cc == 0 or cc == w - 1 for rr, cc in comp):
            continue
        holes.append([(rr + r0, cc + c0) for rr, cc in comp])
    return holes


def _ray_from_hole_row(
    base: np.ndarray,
    composite: set[tuple[int, int]],
    row: int,
    cols: list[int],
) -> tuple[int, list[tuple[int, int]]]:
    dirs = set()
    for c in cols:
        left = c
        right = base.shape[1] - 1 - c
        if left == right:
            raise ValueError("hole row has a center-tie ray direction")
        dirs.add(-1 if left < right else 1)
    if len(dirs) != 1:
        raise ValueError("hole row is not uniquely directed")
    step = dirs.pop()
    row_cells = sorted(c for r, c in composite if r == row)
    if step == -1:
        cur = row_cells[0] - 1
    else:
        cur = row_cells[-1] + 1
    cells: list[tuple[int, int]] = []
    while 0 <= cur < base.shape[1] and int(base[row, cur]) == BG:
        cells.append((row, cur))
        cur += step
    return step, cells


def _analyze(grid: np.ndarray, *, strict: bool) -> dict[str, object]:
    g = np.asarray(grid, dtype=int)
    h, w = g.shape

    payload_components = _components((g != BG) & (g != RED))
    red_components = _components(g == RED)
    used_red: set[int] = set()
    reflection_claims: dict[tuple[int, int], int] = {}
    groups: list[dict[str, object]] = []

    for payload in payload_components:
        payload_color = int(g[payload[0]])
        if any(int(g[r, c]) != payload_color for r, c in payload):
            if strict:
                raise ValueError("payload component mixes colors")
            continue

        candidates = [i for i, marker in enumerate(red_components) if _near_marker(payload, marker)]
        if strict and len(candidates) != 1:
            raise ValueError("payload must have exactly one nearby red marker")
        if not candidates:
            continue
        marker_index = candidates[0]
        if strict and marker_index in used_red:
            raise ValueError("red marker reused by two payloads")
        used_red.add(marker_index)
        marker = red_components[marker_index]

        mirror_kind, reflected = _reflect_payload(payload, marker)
        valid_reflected: list[tuple[int, int]] = []
        for rr, cc in reflected:
            if not (0 <= rr < h and 0 <= cc < w):
                if strict:
                    raise ValueError("reflection leaves the grid")
                continue
            if int(g[rr, cc]) == BG:
                if strict:
                    other = reflection_claims.get((rr, cc))
                    if other is not None and other != payload_color:
                        raise ValueError("two reflections target the same background cell")
                reflection_claims[(rr, cc)] = payload_color
                valid_reflected.append((rr, cc))
            elif strict:
                raise ValueError("reflection must land on background only")

        composite = set(payload) | set(marker) | set(valid_reflected)
        holes = _holes_in_composite(composite)
        if strict and not holes:
            raise ValueError("every generated group must create at least one hole")

        groups.append(
            {
                "payload": payload,
                "marker": marker,
                "payload_color": payload_color,
                "mirror_kind": mirror_kind,
                "reflected": valid_reflected,
                "composite": composite,
                "composite_size": len(composite),
                "holes": holes,
                "marker_length": len(marker),
            }
        )

    if strict and (len(used_red) != len(red_components) or len(groups) != len(payload_components)):
        raise ValueError("every red marker and payload must pair bijectively")

    base = g.copy()
    for group in groups:
        color = int(group["payload_color"])
        for rr, cc in group["reflected"]:
            base[rr, cc] = color

    rays: list[dict[str, object]] = []
    ray_directions: set[int] = set()
    axis_orientations: set[str] = set()
    for group in groups:
        mirror_kind = str(group["mirror_kind"])
        if mirror_kind == "row":
            axis_orientations.add("horizontal")
        elif mirror_kind == "col":
            axis_orientations.add("vertical")

        ray_color = GREEN if mirror_kind in {"row", "col"} else BLUE
        for hole in group["holes"]:
            rows = sorted({r for r, _ in hole})
            for row in rows:
                cols = sorted(c for r, c in hole if r == row)
                step, cells = _ray_from_hole_row(base, set(group["composite"]), row, cols)
                ray_directions.add(step)
                if strict and len(cells) < 2:
                    raise ValueError("rays must visibly extend beyond the hole")
                rays.append(
                    {
                        "cells": cells,
                        "color": ray_color,
                        "size": int(group["composite_size"]),
                        "source_kind": mirror_kind,
                    }
                )

    winners: dict[tuple[int, int], tuple[int, int]] = {}
    for ray in rays:
        color = int(ray["color"])
        size = int(ray["size"])
        for cell in ray["cells"]:
            current = winners.get(cell)
            if current is None or size > current[0]:
                winners[cell] = (size, color)
            elif strict and size == current[0] and color != current[1]:
                raise ValueError("same-sized conflicting rays are ambiguous")

    out = base.copy()
    for (r, c), (_size, color) in winners.items():
        out[r, c] = color

    return {
        "groups": groups,
        "base": base,
        "output": out,
        "rays": rays,
        "ray_directions": ray_directions,
        "axis_orientations": axis_orientations,
    }


def main(grid: np.ndarray) -> np.ndarray:
    """
    Reflect each colored sprite according to its red marker type, then fire color-coded rays from any holes created by the reflected composite.

    Input structure:
    - The background is black (0).
    - There are 3-5 separated non-red sprites, each a single 4-connected component of one unique color.
    - Each sprite has one nearby red marker: either a horizontal/vertical red line segment or a single red pixel.

    Rule:
    - For each sprite, inspect the nearby red marker geometry.
    - A red line segment reflects the sprite's non-red pixels across that row/column; a single red pixel point-reflects the sprite by 180 degrees around that pixel.
    - Keep the original sprite and red marker, add only the reflected copy where the input cell was background, then detect any 4-connected background holes enclosed by that composite.
    - For each hole row, emit a horizontal ray from the composite's outside edge on that row toward the nearer grid edge; axis-reflection holes emit green (3), point-reflection holes emit blue (1), and if two rays overlap the larger composite wins.

    Key insight:
    - The visible geometry of the red marker is the code: it selects the reflection type, which determines the composite enclosure pattern, which in turn determines both whether rays appear and which ray color they use.
    """
    return np.asarray(_analyze(np.asarray(grid, dtype=int), strict=False)["output"], dtype=int)


AXIS_TEMPLATES = (
    {
        "payload": ((-1, -2), (-1, -1), (0, -2), (1, -2), (1, -1)),
        "marker": ((-1, 0), (0, 0)),
    },
    {
        "payload": ((-1, -2), (-1, -1), (0, -2), (1, -2), (1, -1)),
        "marker": ((-1, 0), (0, 0), (1, 0)),
    },
    {
        "payload": ((-2, -3), (-2, -2), (-2, -1), (-1, -2), (0, -2), (0, -1)),
        "marker": ((-1, 0), (0, 0)),
    },
)

POINT_TEMPLATES = (
    {
        "payload": ((-2, -1), (-2, 0), (-2, 1), (-1, -1), (-1, 1)),
        "marker": ((0, 0),),
    },
    {
        "payload": ((-1, -2), (-1, -1), (-1, 0), (-1, 1), (0, -2)),
        "marker": ((0, 0),),
    },
    {
        "payload": ((-2, -2), (-2, -1), (-2, 0), (-2, 1), (-1, -1), (-1, 1)),
        "marker": ((0, 0),),
    },
)


def _sample_template(kind: str) -> dict[str, tuple[tuple[int, int], ...]]:
    pool = AXIS_TEMPLATES if kind == "axis" else POINT_TEMPLATES
    base = random.choice(pool)
    turns = random.randrange(4)
    return {
        "payload": _rotate_coords(tuple(base["payload"]), turns),
        "marker": _rotate_coords(tuple(base["marker"]), turns),
    }


def _translate(coords: tuple[tuple[int, int], ...], dr: int, dc: int) -> list[tuple[int, int]]:
    return [(r + dr, c + dc) for r, c in coords]


def _box_far_enough(
    a: tuple[int, int, int, int],
    b: tuple[int, int, int, int],
    gap: int = 3,
) -> bool:
    return a[2] + gap < b[0] or b[2] + gap < a[0] or a[3] + gap < b[1] or b[3] + gap < a[1]


def _rows_disjoint(
    a: tuple[int, int, int, int],
    b: tuple[int, int, int, int],
) -> bool:
    return a[2] < b[0] or b[2] < a[0]


def _place_group(
    grid: np.ndarray,
    boxes: list[tuple[int, int, int, int]],
    kind: str,
    color: int,
    preferred_side: str,
) -> bool:
    h, w = grid.shape
    for _ in range(GROUP_PLACE_ATTEMPTS):
        template = _sample_template(kind)
        payload_rel = template["payload"]
        marker_rel = template["marker"]
        coords_rel = list(payload_rel + marker_rel)
        r0, c0, r1, c1 = _bbox(coords_rel)
        min_r = 4 - r0
        max_r = h - 5 - r1
        if min_r > max_r:
            continue
        all_cols = range(4 - c0, w - 4 - c1)
        if preferred_side == "left":
            preferred = [c for c in all_cols if c + (c0 + c1) // 2 <= (w // 2) - 2]
        else:
            preferred = [c for c in all_cols if c + (c0 + c1) // 2 >= (w // 2) + 1]
        candidate_cols = preferred or list(all_cols)
        if not candidate_cols:
            continue

        dr = random.randint(min_r, max_r)
        dc = random.choice(candidate_cols)
        payload = _translate(payload_rel, dr, dc)
        marker = _translate(marker_rel, dr, dc)
        box = _bbox(payload + marker)
        if any((not _box_far_enough(box, other)) or (not _rows_disjoint(box, other)) for other in boxes):
            continue
        if any(int(grid[r, c]) != BG for r, c in payload + marker):
            continue

        for r, c in payload:
            grid[r, c] = color
        for r, c in marker:
            grid[r, c] = RED
        boxes.append(box)
        return True
    return False


def _validate_generated(grid: np.ndarray, expected_groups: int) -> bool:
    try:
        info = _analyze(grid, strict=True)
    except ValueError:
        return False

    groups = info["groups"]
    if len(groups) != expected_groups:
        return False
    kinds = {str(group["mirror_kind"]) for group in groups}
    if not ({"row", "col"} & kinds) or "point" not in kinds:
        return False
    if any(len(group["reflected"]) != len(group["payload"]) for group in groups):
        return False
    if any(any(not (1 <= len(hole) <= 2) for hole in group["holes"]) for group in groups):
        return False
    if info["ray_directions"] != {-1, 1}:
        return False

    axis_groups = [group for group in groups if str(group["mirror_kind"]) in {"row", "col"}]
    if len(axis_groups) >= 2 and len(info["axis_orientations"]) < 2:
        return False

    out = np.asarray(info["output"], dtype=int)
    if not np.any(out == GREEN) or not np.any(out == BLUE):
        return False
    if np.array_equal(out, grid):
        return False

    payload_boxes = [_bbox(group["payload"] + group["marker"]) for group in groups]
    for i, box_a in enumerate(payload_boxes):
        for box_b in payload_boxes[i + 1 :]:
            if not _box_far_enough(box_a, box_b):
                return False
    return True


def _generate_attempt() -> np.ndarray | None:
    h = random.randint(21, 26)
    w = random.randint(20, 26)
    group_count = random.randint(3, 5)
    grid = np.zeros((h, w), dtype=int)
    colors = random.sample(INPUT_COLORS, group_count)

    kinds = ["axis", "point"]
    while len(kinds) < group_count:
        kinds.append(random.choice(("axis", "point")))
    random.shuffle(kinds)

    preferred_sides = ["left", "right"]
    while len(preferred_sides) < group_count:
        preferred_sides.append(random.choice(("left", "right")))
    random.shuffle(preferred_sides)

    boxes: list[tuple[int, int, int, int]] = []
    for color, kind, side in zip(colors, kinds, preferred_sides):
        if not _place_group(grid, boxes, kind, color, side):
            return None

    return grid if _validate_generated(grid, group_count) else None


def generate_input() -> np.ndarray:
    for _ in range(GEN_ATTEMPTS):
        grid = _generate_attempt()
        if grid is not None:
            return grid
    raise RuntimeError("generate_input failed to build a valid reflection-ray puzzle")
