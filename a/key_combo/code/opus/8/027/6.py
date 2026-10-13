from __future__ import annotations

import random
import sys
from pathlib import Path

import numpy as np

_ARC_UTILS_DIR = Path("/home/lixinyu/snap/new/solvers")
if str(_ARC_UTILS_DIR) not in sys.path:
    sys.path.append(str(_ARC_UTILS_DIR))

from arc_utils import Color, find_components


Coord = tuple[int, int]
Shape = tuple[Coord, ...]

MOVE_COLORS = (Color.BLUE, Color.RED, Color.GREEN, Color.YELLOW, Color.ORANGE)
AREA_SUBSETS = (
    (2, 3, 4),
    (3, 4, 5),
    (2, 3, 4, 5),
)
GEN_RETRIES = 400

AREA_SHAPES: dict[int, tuple[Shape, ...]] = {
    2: (
        ((0, 0), (0, 1)),
        ((0, 0), (1, 0)),
    ),
    3: (
        ((0, 0), (1, 0), (2, 0)),
    ),
    4: (
        ((0, 0), (0, 1), (1, 0), (1, 1)),
        ((0, 0), (1, 0), (1, 1), (2, 1)),
        ((0, 1), (1, 0), (1, 1), (2, 0)),
    ),
    5: (
        ((0, 1), (1, 0), (1, 1), (1, 2), (2, 1)),
        ((0, 0), (0, 1), (0, 2), (1, 1), (2, 1)),
        ((0, 1), (1, 0), (1, 1), (1, 2), (2, 0)),
    ),
}


def _shape_dims(shape: Shape) -> tuple[int, int]:
    rows = [r for r, _ in shape]
    cols = [c for _, c in shape]
    return max(rows) + 1, max(cols) + 1


def _shift_shape(shape: Shape, top: int, left: int) -> np.ndarray:
    return np.array([(top + r, left + c) for r, c in shape], dtype=int)


def _component_sort_key(comp: dict) -> tuple[int, int, int]:
    coords = comp["coords"]
    return (
        int(comp["area"]),
        int(coords[:, 0].min()),
        int(coords[:, 1].min()),
    )


def _movable_components(grid: np.ndarray) -> list[dict]:
    comps = find_components(
        grid,
        bg=Color.BLACK,
        connectivity=4,
        monochromatic=True,
        return_format="dict",
    )
    return [comp for comp in comps if int(comp["color"]) not in {Color.BLACK, Color.GREY}]


def _drop_components(
    grid: np.ndarray,
    components: list[dict],
    *,
    use_settled: bool,
) -> tuple[np.ndarray, list[dict]]:
    g = np.asarray(grid, dtype=int)
    out = g.copy()
    details: list[dict] = []

    base_mask = g == Color.GREY
    occupied = base_mask.copy()
    for comp in components:
        coords = np.array(comp["coords"], dtype=int)
        out[coords[:, 0], coords[:, 1]] = Color.BLACK

    settled_mask = np.zeros_like(base_mask, dtype=bool)
    for comp in components:
        coords = np.array(comp["coords"], dtype=int)
        color = int(comp["color"])
        drop = 0
        reason = "floor"
        stop_obstacle = False
        stop_settled = False

        while True:
            next_coords = coords + np.array([drop + 1, 0], dtype=int)
            if int(next_coords[:, 0].max()) >= g.shape[0]:
                reason = "floor"
                break

            next_rows = next_coords[:, 0]
            next_cols = next_coords[:, 1]
            stop_obstacle = bool(base_mask[next_rows, next_cols].any())
            stop_settled = bool(settled_mask[next_rows, next_cols].any()) if use_settled else False
            if stop_obstacle or stop_settled:
                if stop_settled and not stop_obstacle:
                    reason = "stack"
                elif stop_obstacle and not stop_settled:
                    reason = "obstacle"
                else:
                    reason = "mixed"
                break
            drop += 1

        final_coords = coords + np.array([drop, 0], dtype=int)
        out[final_coords[:, 0], final_coords[:, 1]] = color
        settled_mask[final_coords[:, 0], final_coords[:, 1]] = True
        if use_settled:
            occupied[final_coords[:, 0], final_coords[:, 1]] = True
        details.append(
            {
                "color": color,
                "area": int(comp["area"]),
                "start": coords,
                "final": final_coords,
                "drop": int(drop),
                "reason": reason,
            }
        )

    return out, details


def main(grid: np.ndarray) -> np.ndarray:
    """
    Drop the colored components downward in ascending area order until they stack on the grey obstacle.

    Input structure:
    - Black background fills the grid.
    - One grey connected obstacle sits in the lower half with an uneven upper contour.
    - Three or four small colored 4-connected components lie strictly above the obstacle.

    Rule:
    - Find every non-grey, non-black 4-connected component and compute its area.
    - Process the components from smallest area to largest.
    - For each one, translate it straight downward until one more row would make any pixel overlap the grey obstacle,
      a previously settled component, or the grid bottom; keep obstacle and all other pixels unchanged.

    Key insight:
    - Area determines the falling order, so earlier small components become new support surfaces that change where
      later larger components can stop.
    """
    g = np.asarray(grid, dtype=int)
    assert g.ndim == 2, "expected a 2D grid"

    components = sorted(_movable_components(g), key=_component_sort_key)
    out, _ = _drop_components(g, components, use_settled=True)
    return out


def _make_obstacle(rows: int, cols: int) -> tuple[np.ndarray, int, int, int]:
    peak = rows - 4
    width = random.randint(6, 8)
    left = random.randint(2, cols - width - 2)
    extra = width - 5
    left_extra = random.randint(0, extra)
    right_extra = extra - left_extra

    profile = [peak + random.randint(0, 1) for _ in range(left_extra)]
    profile.extend([peak, peak + 1, peak + 2, peak + 1, peak])
    profile.extend([peak + random.randint(0, 1) for _ in range(right_extra)])
    profile = [min(rows - 2, value) for value in profile]

    grid = np.zeros((rows, cols), dtype=int)
    for dc, top in enumerate(profile):
        grid[top:, left + dc] = Color.GREY

    pit_col = left + left_extra + 2
    return grid, left, left + width - 1, pit_col


def _choose_shape(area: int) -> Shape:
    return random.choice(AREA_SHAPES[area])


def _area_specs(
    areas: tuple[int, ...],
    obstacle_left: int,
    obstacle_right: int,
    pit_col: int,
) -> list[dict]:
    specs: list[dict] = []

    if 2 in areas:
        side = random.choice(["left", "right"])
        shape = _choose_shape(2)
        _h, w = _shape_dims(shape)
        if side == "left":
            left = random.randint(obstacle_left, max(obstacle_left, pit_col - w))
        else:
            left = random.randint(min(obstacle_right - w + 1, pit_col + 1), obstacle_right - w + 1)
        specs.append({"area": 2, "shape": shape, "left": left})

    if 3 in areas:
        specs.append({"area": 3, "shape": AREA_SHAPES[3][0], "left": pit_col})

    if 4 in areas:
        shape = _choose_shape(4)
        _h, w = _shape_dims(shape)
        side = random.choice(["left", "right"])
        if side == "left":
            left = pit_col - (w - 1)
        else:
            left = pit_col
        specs.append({"area": 4, "shape": shape, "left": left})

    if 5 in areas:
        shape = _choose_shape(5)
        _h, w = _shape_dims(shape)
        left = pit_col - (w // 2)
        specs.append({"area": 5, "shape": shape, "left": left})

    return specs


def _place_components(
    rows: int,
    cols: int,
    peak_row: int,
    specs: list[dict],
    colors: list[int],
) -> list[dict] | None:
    row_slots = [0, 3, 6, 9]
    if len(specs) == 3:
        row_slots = random.sample(row_slots, 3)
    else:
        random.shuffle(row_slots)

    placements: list[dict] = []
    occupied = np.zeros((rows, cols), dtype=bool)
    for spec, top in zip(random.sample(specs, len(specs)), row_slots):
        area = int(spec["area"])
        shape = spec["shape"]
        left = int(spec["left"])
        color = colors.pop()
        h, w = _shape_dims(shape)
        if left < 0 or top < 0 or top + h > peak_row or left + w > occupied.shape[1]:
            return None
        coords = _shift_shape(shape, top, left)
        if occupied[coords[:, 0], coords[:, 1]].any():
            return None
        occupied[coords[:, 0], coords[:, 1]] = True
        placements.append({"area": area, "color": color, "coords": coords})
    return placements


def _render_grid(
    obstacle: np.ndarray,
    placements: list[dict],
) -> np.ndarray:
    grid = obstacle.copy()
    for item in placements:
        coords = item["coords"]
        grid[coords[:, 0], coords[:, 1]] = int(item["color"])
    return grid


def _component_columns(coords: np.ndarray) -> set[int]:
    return {int(c) for c in coords[:, 1]}


def _validate_input(grid: np.ndarray) -> bool:
    if grid.ndim != 2 or max(grid.shape) > 30:
        return False

    obstacle_comps = find_components(
        grid,
        bg=Color.BLACK,
        connectivity=8,
        monochromatic=True,
        colors=[Color.GREY],
        return_format="dict",
    )
    if len(obstacle_comps) != 1:
        return False

    obstacle = obstacle_comps[0]
    obstacle_top = int(obstacle["coords"][:, 0].min())
    obstacle_cols = _component_columns(obstacle["coords"])
    top_profile = []
    for col in sorted(obstacle_cols):
        rows = obstacle["coords"][obstacle["coords"][:, 1] == col][:, 0]
        top_profile.append(int(rows.min()))

    movable = _movable_components(grid)
    if len(movable) not in {3, 4}:
        return False
    colors = [int(comp["color"]) for comp in movable]
    areas = [int(comp["area"]) for comp in movable]
    if len(colors) != len(set(colors)) or len(areas) != len(set(areas)):
        return False
    if any(area < 2 or area > 5 for area in areas):
        return False
    if len(set(top_profile)) < 3:
        return False
    if not any(
        top_profile[i] > top_profile[i - 1] and top_profile[i] > top_profile[i + 1]
        for i in range(1, len(top_profile) - 1)
    ):
        return False

    for comp in movable:
        coords = comp["coords"]
        if int(coords[:, 0].max()) >= obstacle_top:
            return False
        if not (_component_columns(coords) & obstacle_cols):
            return False

    canonical_order = sorted(movable, key=_component_sort_key)
    reverse_order = list(reversed(canonical_order))
    out_main = main(grid)
    out_can, details = _drop_components(grid, canonical_order, use_settled=True)
    out_rev, _ = _drop_components(grid, reverse_order, use_settled=True)
    out_nostack, _ = _drop_components(grid, canonical_order, use_settled=False)

    if not np.array_equal(out_main, out_can):
        return False
    if np.array_equal(out_can, grid):
        return False
    if np.array_equal(out_can, out_rev):
        return False
    if np.array_equal(out_can, out_nostack):
        return False
    if not any(detail["reason"] in {"stack", "mixed"} for detail in details):
        return False
    if sum(int(detail["drop"]) > 0 for detail in details) < 2:
        return False
    if len({int(detail["final"][:, 0].min()) for detail in details}) < 2:
        return False

    out_movable = _movable_components(out_can)
    if sorted(int(comp["area"]) for comp in out_movable) != sorted(areas):
        return False
    return True


def _attempt_generate() -> np.ndarray | None:
    rows = random.randint(16, 19)
    cols = random.randint(14, 19)
    obstacle, obstacle_left, obstacle_right, pit_col = _make_obstacle(rows, cols)
    peak_row = int(np.argwhere(obstacle == Color.GREY)[:, 0].min())

    areas = random.choice(AREA_SUBSETS)
    colors = random.sample(list(MOVE_COLORS), len(areas))
    specs = _area_specs(areas, obstacle_left, obstacle_right, pit_col)
    if len(specs) != len(areas):
        return None

    placements = _place_components(rows, cols, peak_row, specs, colors)
    if placements is None:
        return None

    grid = _render_grid(obstacle, placements)
    return grid if _validate_input(grid) else None


def generate_input() -> np.ndarray:
    for _ in range(GEN_RETRIES):
        grid = _attempt_generate()
        if grid is not None:
            return grid
    raise RuntimeError("failed to generate a valid area-sorted gravity-stacking input")
