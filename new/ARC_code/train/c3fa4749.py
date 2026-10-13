from __future__ import annotations

import numpy as np
from scipy.ndimage import binary_fill_holes

from arc_utils import bbox_coords, colors_present, components_mask
import random


def _mask_from_coords(shape: tuple[int, int], coords: np.ndarray) -> np.ndarray:
    mask = np.zeros(shape, dtype=bool)
    mask[coords[:, 0], coords[:, 1]] = True
    return mask


def _single_side_marker(region: np.ndarray, obj_mask: np.ndarray) -> int | None:
    side_values: list[int] = []
    touched = 0

    if obj_mask[0, :].any():
        touched += 1
        side_values.extend(map(int, region[0, obj_mask[0, :]]))
    if obj_mask[-1, :].any():
        touched += 1
        side_values.extend(map(int, region[-1, obj_mask[-1, :]]))
    if obj_mask[:, 0].any():
        touched += 1
        side_values.extend(map(int, region[obj_mask[:, 0], 0]))
    if obj_mask[:, -1].any():
        touched += 1
        side_values.extend(map(int, region[obj_mask[:, -1], -1]))

    if touched != 1:
        return None
    if len(side_values) != 1:
        return None

    assert side_values, "a side-touching object must contribute at least one marker cell"
    marker = side_values[0]
    if any(value != marker for value in side_values):
        return None
    return marker


def _looks_like_panel(coords: np.ndarray) -> bool:
    r0, c0, r1, c1 = bbox_coords(coords)
    height = r1 - r0 + 1
    width = c1 - c0 + 1
    area = len(coords)
    return height >= 5 and width >= 5 and area >= 20 and area / (height * width) >= 0.35


def _has_multicolor_object(region: np.ndarray, panel_color: int) -> bool:
    for obj_coords in components_mask(region != panel_color, connectivity=4):
        obj_colors = {int(region[r, c]) for r, c in obj_coords}
        if len(obj_colors) >= 2:
            return True
    return False


def _assert_each_panel_has_multicolor_object(g: np.ndarray) -> None:
    panels = 0
    for panel_color in colors_present(g):
        for panel_coords in components_mask(g == panel_color, connectivity=4):
            if not _looks_like_panel(panel_coords):
                continue

            panels += 1
            r0, c0, r1, c1 = bbox_coords(panel_coords)
            region = g[r0 : r1 + 1, c0 : c1 + 1]
            assert _has_multicolor_object(region, int(panel_color)), (
                "each monochrome panel must contain a 4-connected multicolor object"
            )

    assert panels > 0, "expected at least one monochrome panel"


def main(grid: np.ndarray) -> np.ndarray:
    """
    Recolor each side-marked multicolor object embedded in a monochrome panel to its marker color.

    Input structure:
    - Same-color 4-connected components form monochrome panel regions.
    - The bounding box of a panel may contain 4-connected non-panel-color objects.
    - A target object touches exactly one side of its panel bbox, and all of its
      pixels on that side have one marker color.

    Rule:
    - For every target object, fill any panel-colored holes enclosed by the object.
    - Recolor the object and filled holes to the side marker color.
    - Leave all other pixels unchanged.

    Key insight:
    - A multicolor object's single contact side identifies both which object should
      be normalized and which color should replace it.
    """
    g = np.asarray(grid, dtype=int)
    assert g.ndim == 2, "expected a 2D grid"
    _assert_each_panel_has_multicolor_object(g)

    planned: dict[tuple[int, int], int] = {}
    selected_objects = 0

    for panel_color in colors_present(g):
        for panel_coords in components_mask(g == panel_color, connectivity=4):
            r0, c0, r1, c1 = bbox_coords(panel_coords)
            region = g[r0 : r1 + 1, c0 : c1 + 1]

            for obj_coords in components_mask(region != panel_color, connectivity=4):
                obj_mask = _mask_from_coords(region.shape, obj_coords)
                marker = _single_side_marker(region, obj_mask)
                if marker is None:
                    continue

                selected_objects += 1
                paint_mask = binary_fill_holes(obj_mask).astype(bool)
                for rr, cc in np.argwhere(paint_mask):
                    coord = (int(r0 + rr), int(c0 + cc))
                    previous = planned.get(coord)
                    assert previous is None or previous == marker, "conflicting side-marker recolors"
                    planned[coord] = int(marker)

    assert selected_objects > 0, "expected at least one side-marked object"
    assert any(int(g[r, c]) != color for (r, c), color in planned.items()), "expected at least one visible recolor"

    out = g.copy()
    for (r, c), color in planned.items():
        out[r, c] = color
    return out

# --------------- generator (att1) ---------------
COLOR_COUNT = 10
GRID_HEIGHT_RANGE = (20, 30)
GRID_WIDTH_RANGE = (20, 30)
PANEL_COUNT_RANGE = (2, 4)
PANEL_HEIGHT_RANGE = (5, 12)
PANEL_WIDTH_RANGE = (5, 10)
PANEL_PLACEMENT_TRIES = 50
TARGET_COUNT_CHOICES = [1, 2, 3, 4]
TARGET_COUNT_WEIGHTS = [2, 3, 3, 2]
MAX_TARGETS_PER_PANEL = 2
DOUBLE_TARGET_PROB = 0.25
TARGET_SIZE_RANGE = (3, 13)
GENERATION_ATTEMPTS = 300
NEIGHBORS_4 = [(-1, 0), (1, 0), (0, -1), (0, 1)]
SIDE_TOP, SIDE_BOTTOM, SIDE_LEFT, SIDE_RIGHT = range(4)


def _on_side(r, c, ph, pw, side):
    if side == SIDE_TOP:
        return r == 0
    if side == SIDE_BOTTOM:
        return r == ph - 1
    if side == SIDE_LEFT:
        return c == 0
    return c == pw - 1


def _on_other_side(r, c, ph, pw, side):
    return (
        (side != SIDE_TOP and r == 0)
        or (side != SIDE_BOTTOM and r == ph - 1)
        or (side != SIDE_LEFT and c == 0)
        or (side != SIDE_RIGHT and c == pw - 1)
    )


def _can_grow_into(r, c, ph, pw, side, blob, available_mask):
    return (
        0 <= r < ph
        and 0 <= c < pw
        and (r, c) not in blob
        and not _on_other_side(r, c, ph, pw, side)
        and not _on_side(r, c, ph, pw, side)
        and available_mask[r, c]
    )


def _extend_frontier(frontier, r, c, ph, pw, side, blob, available_mask):
    for dr, dc in NEIGHBORS_4:
        nr, nc = r + dr, c + dc
        if _can_grow_into(nr, nc, ph, pw, side, blob, available_mask):
            frontier.append((nr, nc))


def _grow_blob_constrained(ph, pw, contact_cells, target_size, side, available_mask):
    """Grow a 4-connected blob from contact_cells avoiding other sides."""
    blob = set(tuple(x) for x in contact_cells)
    frontier = []
    for r, c in blob:
        _extend_frontier(frontier, r, c, ph, pw, side, blob, available_mask)

    while len(blob) < target_size and frontier:
        idx = random.randint(0, len(frontier) - 1)
        cell = frontier[idx]
        frontier[idx] = frontier[-1]
        frontier.pop()
        if cell in blob:
            continue
        blob.add(cell)
        r, c = cell
        _extend_frontier(frontier, r, c, ph, pw, side, blob, available_mask)

    if len(blob) < 2:
        return None
    return blob


def _make_noise_grid(H, W, avoid_colors_at=None):
    """Create random noise where no two 4-adjacent cells share a color."""
    grid = np.zeros((H, W), dtype=int)
    if avoid_colors_at is None:
        avoid_colors_at = {}
    for r in range(H):
        for c in range(W):
            neighbors = set()
            if r > 0: neighbors.add(int(grid[r-1, c]))
            if c > 0: neighbors.add(int(grid[r, c-1]))
            extra = avoid_colors_at.get((r, c), set())
            forbidden = neighbors | extra
            available = [x for x in range(COLOR_COUNT) if x not in forbidden]
            if not available:
                available = [x for x in range(COLOR_COUNT) if x not in neighbors]
            grid[r, c] = random.choice(available)
    return grid


def _panel_has_multicolor_object(grid, panel_color, r0, c0, ph, pw):
    region = grid[r0:r0+ph, c0:c0+pw]
    for obj_coords in components_mask(region != panel_color, connectivity=4):
        obj_colors = {int(region[r, c]) for r, c in obj_coords}
        if len(obj_colors) >= 2:
            return True
    return False


def _place_multicolor_corner_distractor(grid, output_grid, panel_color, r0, c0, ph, pw, available):
    color_choices = [c for c in range(COLOR_COUNT) if c != panel_color]
    if len(color_choices) < 2:
        return False

    corner_shapes = [
        [(0, 0), (0, 1), (1, 0), (1, 1)],
        [(0, pw - 2), (0, pw - 1), (1, pw - 2), (1, pw - 1)],
        [(ph - 2, 0), (ph - 2, 1), (ph - 1, 0), (ph - 1, 1)],
        [(ph - 2, pw - 2), (ph - 2, pw - 1), (ph - 1, pw - 2), (ph - 1, pw - 1)],
    ]
    random.shuffle(corner_shapes)

    for cells in corner_shapes:
        if not all(0 <= r < ph and 0 <= c < pw and available[r, c] for r, c in cells):
            continue

        cell_set = set(cells)
        isolated = True
        for r, c in cells:
            for dr, dc in NEIGHBORS_4:
                nr, nc = r + dr, c + dc
                if 0 <= nr < ph and 0 <= nc < pw and (nr, nc) not in cell_set:
                    if int(grid[r0 + nr, c0 + nc]) != panel_color:
                        isolated = False
                        break
            if not isolated:
                break
        if not isolated:
            continue

        colors = random.sample(color_choices, 2)
        for idx, (r, c) in enumerate(cells):
            value = colors[idx % 2]
            grid[r0 + r, c0 + c] = value
            output_grid[r0 + r, c0 + c] = value
            available[r, c] = False
        return True

    return False


def _place_panels(H, W, panel_colors):
    panels_info = []
    occupied = np.zeros((H, W), dtype=bool)

    for pc in panel_colors:
        ph = random.randint(*PANEL_HEIGHT_RANGE)
        pw = random.randint(*PANEL_WIDTH_RANGE)
        placed = False
        for _ in range(PANEL_PLACEMENT_TRIES):
            r0 = random.randint(0, H - ph)
            c0 = random.randint(0, W - pw)
            if not occupied[r0:r0+ph, c0:c0+pw].any():
                placed = True
                break
        if not placed:
            return None
        occupied[r0:r0+ph, c0:c0+pw] = True
        panels_info.append((pc, r0, c0, ph, pw))
    return panels_info, occupied


def _build_panel_avoidance(H, W, panels_info, occupied):
    avoid_at = {}
    for pc, r0, c0, ph, pw in panels_info:
        for r in range(r0, r0+ph):
            for c in [c0-1, c0+pw]:
                if 0 <= r < H and 0 <= c < W and not occupied[r, c]:
                    avoid_at.setdefault((r, c), set()).add(pc)
        for c in range(c0, c0+pw):
            for r in [r0-1, r0+ph]:
                if 0 <= r < H and 0 <= c < W and not occupied[r, c]:
                    avoid_at.setdefault((r, c), set()).add(pc)
    return avoid_at


def _assign_target_counts(num_panels):
    total_desired = random.choices(TARGET_COUNT_CHOICES, weights=TARGET_COUNT_WEIGHTS, k=1)[0]
    total_desired = min(total_desired, num_panels * MAX_TARGETS_PER_PANEL)
    panel_target_count = [0] * num_panels
    shuffled = list(range(num_panels))
    random.shuffle(shuffled)
    remaining = total_desired
    for pi in shuffled:
        if remaining <= 0:
            break
        assign = min(
            remaining,
            MAX_TARGETS_PER_PANEL if random.random() < DOUBLE_TARGET_PROB and remaining >= 2 else 1,
        )
        panel_target_count[pi] = assign
        remaining -= assign
    return panel_target_count


def _available_side_cells(available, ph, pw, side):
    if side == SIDE_TOP:
        return [(0, c) for c in range(1, pw - 1) if available[0, c]]
    if side == SIDE_BOTTOM:
        return [(ph - 1, c) for c in range(1, pw - 1) if available[ph - 1, c]]
    if side == SIDE_LEFT:
        return [(r, 0) for r in range(1, ph - 1) if available[r, 0]]
    return [(r, pw - 1) for r in range(1, ph - 1) if available[r, pw - 1]]


def _paint_target(grid, output_grid, available, panel, side):
    pc, r0, c0, ph, pw = panel
    marker = random.choice([c for c in range(COLOR_COUNT) if c != pc])
    side_cells = _available_side_cells(available, ph, pw, side)
    if not side_cells:
        return False

    contact_cells = [random.choice(side_cells)]
    target_size = random.randint(*TARGET_SIZE_RANGE)
    blob = _grow_blob_constrained(ph, pw, contact_cells, target_size, side, available)
    if blob is None:
        return False

    contact_on_side = {(r, c) for r, c in blob if _on_side(r, c, ph, pw, side)}
    interior = blob - contact_on_side
    if not interior:
        return False

    blob_mask = np.zeros((ph, pw), dtype=bool)
    for r, c in blob:
        blob_mask[r, c] = True
    filled_mask = binary_fill_holes(blob_mask)
    hole_cells = set(zip(*np.where(filled_mask & ~blob_mask)))

    for r, c in blob:
        available[r, c] = False
    for r, c in hole_cells:
        available[r, c] = False

    for lr, lc in blob:
        output_grid[r0+lr, c0+lc] = marker
    for lr, lc in hole_cells:
        output_grid[r0+lr, c0+lc] = marker

    for lr, lc in contact_on_side:
        grid[r0+lr, c0+lc] = marker
    non_panel_non_marker = [c for c in range(COLOR_COUNT) if c != pc and c != marker]
    if not non_panel_non_marker:
        non_panel_non_marker = [c for c in range(COLOR_COUNT) if c != pc]
    for lr, lc in interior:
        grid[r0+lr, c0+lc] = random.choice(non_panel_non_marker)
    return True


def _paint_targets(grid, output_grid, panels_info, panel_target_count, panel_available):
    total_targets_placed = 0
    for pi in range(len(panels_info)):
        if panel_target_count[pi] == 0:
            continue
        panel = panels_info[pi]
        _, _, _, ph, pw = panel
        available = panel_available[pi]
        used_sides = set()

        for _ in range(panel_target_count[pi]):
            available_sides = [s for s in range(4) if s not in used_sides]
            if not available_sides:
                break
            side = random.choice(available_sides)
            used_sides.add(side)

            if _paint_target(grid, output_grid, available, panel, side):
                total_targets_placed += 1
    return total_targets_placed


def _ensure_multicolor_objects(grid, output_grid, panels_info, panel_available):
    for pi, (pc, r0, c0, ph, pw) in enumerate(panels_info):
        if _panel_has_multicolor_object(grid, pc, r0, c0, ph, pw):
            continue
        if not _place_multicolor_corner_distractor(
            grid, output_grid, pc, r0, c0, ph, pw, panel_available[pi]
        ):
            return False
    return True


def _verified_input(grid, output_grid):
    try:
        result = main(grid)
        if np.array_equal(result, output_grid):
            return grid
    except (AssertionError, Exception):
        pass
    return None


def _gen_attempt():
    H = random.randint(*GRID_HEIGHT_RANGE)
    W = random.randint(*GRID_WIDTH_RANGE)

    num_panels = random.randint(*PANEL_COUNT_RANGE)
    panel_colors = random.sample(range(COLOR_COUNT), num_panels)

    placed = _place_panels(H, W, panel_colors)
    if placed is None:
        return None
    panels_info, occupied = placed

    grid = _make_noise_grid(H, W, _build_panel_avoidance(H, W, panels_info, occupied))
    for pc, r0, c0, ph, pw in panels_info:
        grid[r0:r0+ph, c0:c0+pw] = pc

    output_grid = grid.copy()
    panel_available = [np.ones((ph, pw), dtype=bool) for _, _, _, ph, pw in panels_info]
    panel_target_count = _assign_target_counts(num_panels)

    if _paint_targets(grid, output_grid, panels_info, panel_target_count, panel_available) == 0:
        return None
    if not _ensure_multicolor_objects(grid, output_grid, panels_info, panel_available):
        return None
    return _verified_input(grid, output_grid)


def generate_input() -> np.ndarray:
    for _ in range(GENERATION_ATTEMPTS):
        result = _gen_attempt()
        if result is not None:
            return result
    raise RuntimeError(f"Failed to generate valid input after {GENERATION_ATTEMPTS} attempts")
