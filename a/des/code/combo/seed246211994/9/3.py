from __future__ import annotations

from collections import Counter, defaultdict
import random

import numpy as np


SIDES = {
    "top": (-1, 0),
    "bottom": (1, 0),
    "left": (0, -1),
    "right": (0, 1),
}

# Local attachment shapes are expressed in (depth_from_plus, lateral_offset)
# coordinates, with the anchor cell adjacent to the plus at (0, 0).
LOCAL_SHAPES = (
    frozenset({(0, 0)}),
    frozenset({(0, 0), (1, 0)}),
    frozenset({(0, 0), (0, -1)}),
    frozenset({(0, 0), (0, 1)}),
    frozenset({(0, 0), (1, 0), (2, 0)}),
    frozenset({(0, -1), (0, 0), (0, 1)}),
    frozenset({(0, 0), (1, 0), (0, -1)}),
    frozenset({(0, 0), (1, 0), (0, 1)}),
    frozenset({(0, 0), (1, 0), (1, -1)}),
    frozenset({(0, 0), (1, 0), (1, 1)}),
)

GRID_MIN = 12
GRID_MAX = 21
PLUS_COUNT_MIN = 1
PLUS_COUNT_MAX = 3
ARM_LEN_MIN = 1
ARM_LEN_MAX = 3
MAX_ATTACHMENTS = 8
MIN_BG_FRACTION = 0.55
PLACEMENT_TRIES = 80
GENERATION_RETRIES = 400


def _most_common_color(grid: np.ndarray) -> int:
    values, counts = np.unique(grid, return_counts=True)
    return int(values[np.argmax(counts)])


def _find_components(grid: np.ndarray, bg: int) -> list[dict]:
    h, w = grid.shape
    seen = np.zeros((h, w), dtype=bool)
    components: list[dict] = []

    for r in range(h):
        for c in range(w):
            color = int(grid[r, c])
            if seen[r, c] or color == bg:
                continue

            stack = [(r, c)]
            seen[r, c] = True
            coords: list[tuple[int, int]] = []
            while stack:
                rr, cc = stack.pop()
                coords.append((rr, cc))
                for dr, dc in ((1, 0), (-1, 0), (0, 1), (0, -1)):
                    nr, nc = rr + dr, cc + dc
                    if 0 <= nr < h and 0 <= nc < w and not seen[nr, nc] and int(grid[nr, nc]) == color:
                        seen[nr, nc] = True
                        stack.append((nr, nc))

            rows = [rr for rr, _ in coords]
            cols = [cc for _, cc in coords]
            components.append(
                {
                    "color": color,
                    "coords": frozenset(coords),
                    "area": len(coords),
                    "bbox": (min(rows), min(cols), max(rows), max(cols)),
                }
            )

    return components


def _parse_plus(component: dict) -> dict | None:
    coords = component["coords"]
    centers = []
    for r, c in coords:
        if all((r + dr, c + dc) in coords for dr, dc in SIDES.values()):
            centers.append((r, c))
    if len(centers) != 1:
        return None

    center_r, center_c = centers[0]
    arms: dict[str, int] = {}
    for side, (dr, dc) in SIDES.items():
        steps = 0
        rr, cc = center_r + dr, center_c + dc
        while (rr, cc) in coords:
            steps += 1
            rr += dr
            cc += dc
        if steps == 0:
            return None
        arms[side] = steps

    expected = {(center_r, center_c)}
    for offset in range(1, arms["top"] + 1):
        expected.add((center_r - offset, center_c))
    for offset in range(1, arms["bottom"] + 1):
        expected.add((center_r + offset, center_c))
    for offset in range(1, arms["left"] + 1):
        expected.add((center_r, center_c - offset))
    for offset in range(1, arms["right"] + 1):
        expected.add((center_r, center_c + offset))
    if coords != expected:
        return None

    ends = {
        "top": (center_r - arms["top"], center_c),
        "bottom": (center_r + arms["bottom"], center_c),
        "left": (center_r, center_c - arms["left"]),
        "right": (center_r, center_c + arms["right"]),
    }
    return {
        "color": int(component["color"]),
        "coords": coords,
        "area": int(component["area"]),
        "center": (center_r, center_c),
        "arms": arms,
        "ends": ends,
    }


def _infer_pluses(components: list[dict]) -> tuple[int, list[dict]]:
    parsed_by_color: dict[int, list[dict]] = defaultdict(list)
    counts_by_color: Counter[int] = Counter()
    all_by_color: dict[int, list[dict]] = defaultdict(list)

    for comp in components:
        color = int(comp["color"])
        all_by_color[color].append(comp)
        parsed = _parse_plus(comp)
        if parsed is not None:
            parsed_by_color[color].append(parsed)
            counts_by_color[color] += int(parsed["area"])

    assert counts_by_color, "expected at least one plus object"
    ordered = counts_by_color.most_common()
    plus_color, plus_area = ordered[0]
    if len(ordered) > 1:
        assert plus_area > ordered[1][1], "plus color must be uniquely dominant among plus-shaped colors"
    assert len(parsed_by_color[plus_color]) == len(all_by_color[plus_color]), "plus color must contain only plus components"
    return int(plus_color), parsed_by_color[plus_color]


def _attachment_matches_side(coords: frozenset[tuple[int, int]], plus: dict, side: str) -> bool:
    end_r, end_c = plus["ends"][side]
    if side == "top":
        return (end_r - 1, end_c) in coords and all(r < end_r for r, _ in coords)
    if side == "bottom":
        return (end_r + 1, end_c) in coords and all(r > end_r for r, _ in coords)
    if side == "left":
        return (end_r, end_c - 1) in coords and all(c < end_c for _, c in coords)
    return (end_r, end_c + 1) in coords and all(c > end_c for _, c in coords)


def _attachment_period(coords: frozenset[tuple[int, int]], plus: dict, side: str) -> int:
    end_r, end_c = plus["ends"][side]
    if side == "top":
        distances = [end_r - r for r, _ in coords]
    elif side == "bottom":
        distances = [r - end_r for r, _ in coords]
    elif side == "left":
        distances = [end_c - c for _, c in coords]
    else:
        distances = [c - end_c for _, c in coords]
    assert min(distances) >= 1
    return max(distances)


def main(grid: np.ndarray) -> np.ndarray:
    """
    Repeat each colored block attached outside a plus arm periodically away from the plus.

    Input structure:
    - The background is the dominant color.
    - One dominant foreground color forms several '+' objects.
    - Small colored 4-connected blocks (1-3 cells) are attached just beyond some arm ends.

    Rule:
    - Determine which side of which plus each block is attached to.
    - Repeat the block by translating it away from that plus with period equal to
      the block's own depth along that direction.
    - Continue until the grid boundary, without overwriting the plus pixels.
    """
    g = np.asarray(grid, dtype=int)
    assert g.ndim == 2

    bg = _most_common_color(g)
    components = _find_components(g, bg)
    plus_color, pluses = _infer_pluses(components)

    out = g.copy()
    attachment_components = [comp for comp in components if int(comp["color"]) != plus_color]
    assert attachment_components, "expected at least one attached block"

    matched = 0
    h, w = g.shape
    for comp in attachment_components:
        coords = comp["coords"]
        candidates: list[tuple[dict, str]] = []
        for plus in pluses:
            for side in SIDES:
                if _attachment_matches_side(coords, plus, side):
                    candidates.append((plus, side))
        assert len(candidates) == 1, "each block must attach to exactly one plus side"

        plus, side = candidates[0]
        period = _attachment_period(coords, plus, side)
        dr, dc = SIDES[side]
        shift_r = dr * period
        shift_c = dc * period
        color = int(comp["color"])

        step = 1
        while True:
            any_in_bounds = False
            for r, c in coords:
                rr = r + step * shift_r
                cc = c + step * shift_c
                if 0 <= rr < h and 0 <= cc < w:
                    any_in_bounds = True
                    if int(out[rr, cc]) != plus_color:
                        out[rr, cc] = color
            if not any_in_bounds:
                break
            step += 1
        matched += 1

    assert matched == len(attachment_components)
    return out


def _shape_depth(shape: frozenset[tuple[int, int]]) -> int:
    return max(t for t, _ in shape) + 1


def _plus_cells(center_r: int, center_c: int, arm_len: int) -> set[tuple[int, int]]:
    cells = {(center_r, center_c)}
    for offset in range(1, arm_len + 1):
        cells.add((center_r - offset, center_c))
        cells.add((center_r + offset, center_c))
        cells.add((center_r, center_c - offset))
        cells.add((center_r, center_c + offset))
    return cells


def _end_cell(center_r: int, center_c: int, arm_len: int, side: str) -> tuple[int, int]:
    if side == "top":
        return center_r - arm_len, center_c
    if side == "bottom":
        return center_r + arm_len, center_c
    if side == "left":
        return center_r, center_c - arm_len
    return center_r, center_c + arm_len


def _absolute_attachment_cells(
    center_r: int,
    center_c: int,
    arm_len: int,
    side: str,
    shape: frozenset[tuple[int, int]],
    repeat_index: int = 0,
) -> set[tuple[int, int]]:
    end_r, end_c = _end_cell(center_r, center_c, arm_len, side)
    depth = _shape_depth(shape)
    extra = repeat_index * depth
    cells = set()

    for t, s in shape:
        tt = t + extra
        if side == "top":
            cells.add((end_r - 1 - tt, end_c + s))
        elif side == "bottom":
            cells.add((end_r + 1 + tt, end_c + s))
        elif side == "left":
            cells.add((end_r + s, end_c - 1 - tt))
        else:
            cells.add((end_r + s, end_c + 1 + tt))
    return cells


def _all_repeated_cells(
    center_r: int,
    center_c: int,
    arm_len: int,
    side: str,
    shape: frozenset[tuple[int, int]],
    h: int,
    w: int,
) -> set[tuple[int, int]]:
    cells = set()
    repeat_index = 0
    while True:
        repeat_cells = _absolute_attachment_cells(center_r, center_c, arm_len, side, shape, repeat_index)
        in_bounds = {(r, c) for r, c in repeat_cells if 0 <= r < h and 0 <= c < w}
        if not in_bounds:
            break
        cells |= in_bounds
        repeat_index += 1
    return cells


def _chebyshev_halo(cells: set[tuple[int, int]], h: int, w: int) -> set[tuple[int, int]]:
    halo = set()
    for r, c in cells:
        for dr in (-1, 0, 1):
            for dc in (-1, 0, 1):
                rr, cc = r + dr, c + dc
                if 0 <= rr < h and 0 <= cc < w:
                    halo.add((rr, cc))
    return halo


def _center_bounds(
    h: int,
    w: int,
    arm_len: int,
    plan: dict[str, frozenset[tuple[int, int]]],
) -> tuple[int, int, int, int] | None:
    row_min = arm_len
    row_max = h - 1 - arm_len
    col_min = arm_len
    col_max = w - 1 - arm_len

    for side, shape in plan.items():
        depth = _shape_depth(shape)
        laterals = [s for _, s in shape]
        min_lat = min(laterals)
        max_lat = max(laterals)

        if side == "top":
            row_min = max(row_min, arm_len + 2 * depth)
            col_min = max(col_min, -min_lat)
            col_max = min(col_max, w - 1 - max_lat)
        elif side == "bottom":
            row_max = min(row_max, h - 1 - arm_len - 2 * depth)
            col_min = max(col_min, -min_lat)
            col_max = min(col_max, w - 1 - max_lat)
        elif side == "left":
            col_min = max(col_min, arm_len + 2 * depth)
            row_min = max(row_min, -min_lat)
            row_max = min(row_max, h - 1 - max_lat)
        else:
            col_max = min(col_max, w - 1 - arm_len - 2 * depth)
            row_min = max(row_min, -min_lat)
            row_max = min(row_max, h - 1 - max_lat)

    if row_min > row_max or col_min > col_max:
        return None
    return row_min, row_max, col_min, col_max


def _sample_attachment_plan(max_attachments_left: int) -> dict[str, frozenset[tuple[int, int]]] | None:
    max_sides = min(4, max_attachments_left)
    if max_sides <= 0:
        return None
    side_count = random.randint(1, max_sides)
    sides = random.sample(list(SIDES), side_count)
    return {side: random.choice(LOCAL_SHAPES) for side in sides}


def _try_place_plus(
    h: int,
    w: int,
    blocked_input: set[tuple[int, int]],
    output_occ: set[tuple[int, int]],
    max_attachments_left: int,
) -> dict | None:
    for _ in range(PLACEMENT_TRIES):
        arm_len = random.randint(ARM_LEN_MIN, ARM_LEN_MAX)
        plan = _sample_attachment_plan(max_attachments_left)
        if not plan:
            return None
        bounds = _center_bounds(h, w, arm_len, plan)
        if bounds is None:
            continue

        row_min, row_max, col_min, col_max = bounds
        center_r = random.randint(row_min, row_max)
        center_c = random.randint(col_min, col_max)
        plus_cells = _plus_cells(center_r, center_c, arm_len)

        package_input = set(plus_cells)
        package_output = set(plus_cells)
        attachments = []
        valid = True
        for side, shape in plan.items():
            original = _absolute_attachment_cells(center_r, center_c, arm_len, side, shape, repeat_index=0)
            second = _absolute_attachment_cells(center_r, center_c, arm_len, side, shape, repeat_index=1)
            full = _all_repeated_cells(center_r, center_c, arm_len, side, shape, h, w)

            if not all(0 <= r < h and 0 <= c < w for r, c in original):
                valid = False
                break
            if not all(0 <= r < h and 0 <= c < w for r, c in second):
                valid = False
                break
            if plus_cells & original or plus_cells & full:
                valid = False
                break
            if package_input & original or package_output & full:
                valid = False
                break

            attachments.append({"side": side, "shape": shape, "original": original, "full": full})
            package_input |= original
            package_output |= full

        if not valid:
            continue
        if package_input & blocked_input:
            continue
        if package_output & output_occ:
            continue

        return {
            "center": (center_r, center_c),
            "arm_len": arm_len,
            "plus_cells": plus_cells,
            "attachments": attachments,
            "package_input": package_input,
            "package_output": package_output,
        }

    return None


def _count_colors(grid: np.ndarray) -> Counter[int]:
    return Counter(int(v) for v in grid.ravel())


def _gen_attempt() -> np.ndarray | None:
    h = random.randint(GRID_MIN, GRID_MAX)
    w = random.randint(GRID_MIN, GRID_MAX)
    bg, plus_color = random.sample(range(10), 2)

    n_pluses = random.randint(PLUS_COUNT_MIN, PLUS_COUNT_MAX)
    packages = []
    blocked_input: set[tuple[int, int]] = set()
    output_occ: set[tuple[int, int]] = set()
    attachment_total = 0

    for plus_index in range(n_pluses):
        pluses_left = n_pluses - plus_index
        max_left = MAX_ATTACHMENTS - attachment_total - (pluses_left - 1)
        package = _try_place_plus(h, w, blocked_input, output_occ, max_left)
        if package is None:
            return None
        packages.append(package)
        attachment_total += len(package["attachments"])
        blocked_input |= _chebyshev_halo(package["package_input"], h, w)
        output_occ |= package["package_output"]

    if attachment_total == 0 or attachment_total > 8:
        return None

    input_grid = np.full((h, w), bg, dtype=int)
    output_grid = np.full((h, w), bg, dtype=int)

    attachment_colors = random.sample([c for c in range(10) if c not in {bg, plus_color}], attachment_total)
    color_index = 0
    for package in packages:
        for r, c in package["plus_cells"]:
            input_grid[r, c] = plus_color
            output_grid[r, c] = plus_color
        for attachment in package["attachments"]:
            color = attachment_colors[color_index]
            color_index += 1
            for r, c in attachment["original"]:
                input_grid[r, c] = color
            for r, c in attachment["full"]:
                output_grid[r, c] = color

    counts = _count_colors(input_grid)
    if counts[bg] / input_grid.size < MIN_BG_FRACTION:
        return None
    if counts[plus_color] <= max(count for color, count in counts.items() if color not in {bg, plus_color}):
        return None

    try:
        solved = main(input_grid)
    except Exception:
        return None
    if not np.array_equal(solved, output_grid):
        return None
    return input_grid


def generate_input() -> np.ndarray:
    for _ in range(GENERATION_RETRIES):
        grid = _gen_attempt()
        if grid is not None:
            return grid
    raise RuntimeError(f"failed to generate valid input after {GENERATION_RETRIES} attempts")
