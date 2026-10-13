import random
from collections import deque

import numpy as np


DIRS = {
    "up": (-1, 0),
    "right": (0, 1),
    "down": (1, 0),
    "left": (0, -1),
}
OPPOSITE = {
    "up": "down",
    "right": "left",
    "down": "up",
    "left": "right",
}
ARROW_OFFSETS = {
    "up": [(0, 0), (-1, 0), (0, -1), (0, 1)],
    "right": [(0, 0), (-1, 0), (1, 0), (0, 1)],
    "down": [(0, 0), (1, 0), (0, -1), (0, 1)],
    "left": [(0, 0), (-1, 0), (1, 0), (0, -1)],
}

PALETTE = tuple(range(10))
RECT_DIMS = (
    (1, 2),
    (1, 3),
    (2, 1),
    (3, 1),
    (2, 2),
    (2, 3),
    (3, 2),
    (3, 3),
)
PAIR_COUNT_OPTIONS = (1, 2, 3, 4)
PAIR_COUNT_WEIGHTS = (2, 4, 4, 2)
GRID_MIN = 12
GRID_MAX = 28
GAP_MIN = 2
GAP_MAX = 8
MAX_GENERATION_ATTEMPTS = 500
PAIR_PLACEMENT_ATTEMPTS = 200


def _get_bg(grid: np.ndarray) -> int:
    values, counts = np.unique(grid, return_counts=True)
    return int(values[np.argmax(counts)])


def _neighbors4(r: int, c: int):
    for dr, dc in DIRS.values():
        yield r + dr, c + dc


def _find_components(grid: np.ndarray, bg: int):
    h, w = grid.shape
    seen = np.zeros((h, w), dtype=bool)
    components = []

    for r in range(h):
        for c in range(w):
            if seen[r, c] or int(grid[r, c]) == bg:
                continue
            color = int(grid[r, c])
            q = deque([(r, c)])
            seen[r, c] = True
            coords = []
            while q:
                cr, cc = q.popleft()
                coords.append((int(cr), int(cc)))
                for nr, nc in _neighbors4(cr, cc):
                    if 0 <= nr < h and 0 <= nc < w and not seen[nr, nc] and int(grid[nr, nc]) == color:
                        seen[nr, nc] = True
                        q.append((nr, nc))

            arr = np.array(coords, dtype=int)
            r0, c0 = arr.min(axis=0)
            r1, c1 = arr.max(axis=0)
            components.append(
                {
                    "color": color,
                    "coords": arr,
                    "set": set(coords),
                    "bbox": (int(r0), int(c0), int(r1), int(c1)),
                    "area": int(len(coords)),
                }
            )
    return components


def _is_solid_rectangle(component: dict) -> bool:
    r0, c0, r1, c1 = component["bbox"]
    return int(component["area"]) == (r1 - r0 + 1) * (c1 - c0 + 1)


def _parse_arrow(component: dict):
    if int(component["area"]) != 4:
        return None

    coords = component["set"]
    degrees = {
        cell: sum((cell[0] + dr, cell[1] + dc) in coords for dr, dc in DIRS.values())
        for cell in coords
    }
    centers = [cell for cell, degree in degrees.items() if degree == 3]
    leaves = [cell for cell, degree in degrees.items() if degree == 1]
    if len(centers) != 1 or len(leaves) != 3:
        return None

    center = centers[0]
    present_dirs = [
        name
        for name, (dr, dc) in DIRS.items()
        if (center[0] + dr, center[1] + dc) in coords
    ]
    if len(present_dirs) != 3:
        return None

    missing = next(name for name in DIRS if name not in present_dirs)
    direction = OPPOSITE[missing]
    dr, dc = DIRS[direction]
    tip = (center[0] + dr, center[1] + dc)
    return {
        "color": int(component["color"]),
        "center": center,
        "direction": direction,
        "step": (dr, dc),
        "tip": tip,
        "coords": component["coords"],
    }


def _ray_distance(arrow: dict, target: dict) -> int | None:
    tr, tc = arrow["tip"]
    r0, c0, r1, c1 = target["bbox"]
    direction = arrow["direction"]

    if direction == "right" and r0 <= tr <= r1 and c0 > tc:
        return int(c0 - tc)
    if direction == "left" and r0 <= tr <= r1 and c1 < tc:
        return int(tc - c1)
    if direction == "down" and c0 <= tc <= c1 and r0 > tr:
        return int(r0 - tr)
    if direction == "up" and c0 <= tc <= c1 and r1 < tr:
        return int(tr - r1)
    return None


def _path_cells(arrow: dict, distance: int):
    tr, tc = arrow["tip"]
    dr, dc = arrow["step"]
    return [(tr + dr * k, tc + dc * k) for k in range(1, distance)]


def _shifted_coords(target: dict, arrow: dict, distance: int) -> np.ndarray:
    dr, dc = arrow["step"]
    shift_steps = 1 - distance
    delta = np.array([dr * shift_steps, dc * shift_steps], dtype=int)
    return target["coords"] + delta


def main(grid: np.ndarray) -> np.ndarray:
    """
    Pull each same-color target block up to its arrowhead and paint the traveled corridor in that arrow's color.

    Input structure:
    - The background is the most frequent color.
    - Foreground components are monochrome arrows and monochrome rectangular target blocks.
    - Each arrow is a 4-cell cross-with-one-missing-arm shape whose missing arm is opposite its pointing direction.
    - For every arrow, exactly one same-color rectangular target lies ahead on the arrowhead's row or column.

    Rule:
    - Detect every arrow's center, missing arm, and pointing direction.
    - For each arrow, find the unique same-color rectangle intersected by the forward ray from its tip.
    - Erase that target from its original location, shift it along the ray until its nearest edge occupies the cell immediately after the arrow tip, and repaint it there.
    - On the cells between the arrow tip and the target's original near edge, fill every originally background cell with the arrow color while leaving the arrow itself unchanged.

    Key insight:
    - The T-shaped 4-cell arrow encodes direction via its missing arm, while color matches the arrow to its target; solving the puzzle requires combining those two independent cues.
    """
    g = np.asarray(grid, dtype=int)
    bg = _get_bg(g)
    components = _find_components(g, bg)

    arrows = []
    targets = []
    for component in components:
        arrow = _parse_arrow(component)
        if arrow is not None:
            arrows.append(arrow)
        elif _is_solid_rectangle(component) and max(component["bbox"][2] - component["bbox"][0] + 1, component["bbox"][3] - component["bbox"][1] + 1) >= 2:
            targets.append(component)

    assert arrows, "expected at least one arrow"
    assert targets, "expected at least one target block"

    plans = []
    used_targets = set()
    for arrow in arrows:
        candidates = []
        for index, target in enumerate(targets):
            if int(target["color"]) != int(arrow["color"]):
                continue
            distance = _ray_distance(arrow, target)
            if distance is not None:
                candidates.append((distance, index, target))

        assert len(candidates) == 1, "each arrow must see exactly one same-color target"
        distance, index, target = candidates[0]
        assert index not in used_targets, "a target cannot serve multiple arrows"
        used_targets.add(index)

        path = _path_cells(arrow, distance)
        for r, c in path:
            assert int(g[r, c]) == bg, "ray to target must pass only through background"

        shifted = _shifted_coords(target, arrow, distance)
        plans.append(
            {
                "color": int(arrow["color"]),
                "target": target,
                "path": path,
                "shifted": shifted,
            }
        )

    out = g.copy()
    for plan in plans:
        coords = plan["target"]["coords"]
        out[coords[:, 0], coords[:, 1]] = bg

    for plan in plans:
        for r, c in plan["path"]:
            if int(g[r, c]) == bg:
                out[r, c] = int(plan["color"])
        shifted = plan["shifted"]
        out[shifted[:, 0], shifted[:, 1]] = int(plan["color"])

    return out


def _arrow_cells(center: tuple[int, int], direction: str):
    return {(center[0] + dr, center[1] + dc) for dr, dc in ARROW_OFFSETS[direction]}


def _rect_cells(r0: int, c0: int, h: int, w: int):
    return {(r, c) for r in range(r0, r0 + h) for c in range(c0, c0 + w)}


def _expand(cells: set[tuple[int, int]], shape: tuple[int, int]):
    h, w = shape
    expanded = set()
    for r, c in cells:
        for dr in (-1, 0, 1):
            for dc in (-1, 0, 1):
                nr, nc = r + dr, c + dc
                if 0 <= nr < h and 0 <= nc < w:
                    expanded.add((nr, nc))
    return expanded


def _sample_pair_placement(h: int, w: int, direction: str, rect_h: int, rect_w: int, gap: int):
    if direction in ("left", "right"):
        hit_index = random.randrange(rect_h)
        row_lo = max(1, hit_index)
        row_hi = min(h - 2, h - rect_h + hit_index)
        if row_lo > row_hi:
            return None

        if direction == "right":
            col_lo, col_hi = 1, w - gap - rect_w - 2
        else:
            col_lo, col_hi = gap + rect_w + 1, w - 2
        if col_lo > col_hi:
            return None

        center = (random.randint(row_lo, row_hi), random.randint(col_lo, col_hi))
        rect_r0 = center[0] - hit_index
        if direction == "right":
            rect_c0 = center[1] + 2 + gap
        else:
            rect_c0 = center[1] - gap - rect_w - 1
    else:
        hit_index = random.randrange(rect_w)
        col_lo = max(1, hit_index)
        col_hi = min(w - 2, w - rect_w + hit_index)
        if col_lo > col_hi:
            return None

        if direction == "down":
            row_lo, row_hi = 1, h - gap - rect_h - 2
        else:
            row_lo, row_hi = gap + rect_h + 1, h - 2
        if row_lo > row_hi:
            return None

        center = (random.randint(row_lo, row_hi), random.randint(col_lo, col_hi))
        rect_c0 = center[1] - hit_index
        if direction == "down":
            rect_r0 = center[0] + 2 + gap
        else:
            rect_r0 = center[0] - gap - rect_h - 1

    return center, rect_r0, rect_c0


def _build_pair_footprint(center: tuple[int, int], direction: str, rect_r0: int, rect_c0: int, rect_h: int, rect_w: int):
    arrow = _arrow_cells(center, direction)
    target = _rect_cells(rect_r0, rect_c0, rect_h, rect_w)

    if direction == "right":
        path = {(center[0], c) for c in range(center[1] + 2, rect_c0)}
        final_target = _rect_cells(rect_r0, center[1] + 2, rect_h, rect_w)
    elif direction == "left":
        path = {(center[0], c) for c in range(rect_c0 + rect_w, center[1] - 1)}
        final_target = _rect_cells(rect_r0, center[1] - rect_w - 1, rect_h, rect_w)
    elif direction == "down":
        path = {(r, center[1]) for r in range(center[0] + 2, rect_r0)}
        final_target = _rect_cells(center[0] + 2, rect_c0, rect_h, rect_w)
    else:
        path = {(r, center[1]) for r in range(rect_r0 + rect_h, center[0] - 1)}
        final_target = _rect_cells(center[0] - rect_h - 1, rect_c0, rect_h, rect_w)

    return arrow, target, path, final_target


def _gen_attempt() -> np.ndarray | None:
    n_pairs = random.choices(PAIR_COUNT_OPTIONS, weights=PAIR_COUNT_WEIGHTS, k=1)[0]
    min_dim = max(GRID_MIN, 9 + 4 * (n_pairs - 1))
    h = random.randint(min_dim, GRID_MAX)
    w = random.randint(min_dim, GRID_MAX)

    bg = random.choice(PALETTE)
    colors = random.sample([color for color in PALETTE if color != bg], n_pairs)
    directions = list(DIRS)
    random.shuffle(directions)
    directions = directions[:n_pairs]

    grid = np.full((h, w), bg, dtype=int)
    blocked = set()

    for color, direction in zip(colors, directions):
        placed = False
        for _ in range(PAIR_PLACEMENT_ATTEMPTS):
            rect_h, rect_w = random.choice(RECT_DIMS)
            gap = random.randint(GAP_MIN, GAP_MAX)
            placement = _sample_pair_placement(h, w, direction, rect_h, rect_w, gap)
            if placement is None:
                continue

            center, rect_r0, rect_c0 = placement
            arrow, target, path, final_target = _build_pair_footprint(
                center,
                direction,
                rect_r0,
                rect_c0,
                rect_h,
                rect_w,
            )
            footprint = arrow | target | path | final_target
            if any(not (0 <= r < h and 0 <= c < w) for r, c in footprint):
                continue

            protected = _expand(footprint, (h, w))
            if protected & blocked:
                continue

            for r, c in arrow | target:
                grid[r, c] = color
            blocked |= protected
            placed = True
            break

        if not placed:
            return None

    out = main(grid)
    if out.shape != grid.shape or np.array_equal(out, grid):
        return None
    return grid


def generate_input() -> np.ndarray:
    for _ in range(MAX_GENERATION_ATTEMPTS):
        grid = _gen_attempt()
        if grid is not None:
            return grid
    raise RuntimeError("failed to generate valid input for puzzle 2")
