import random
from collections import deque

import numpy as np


BLACK = 0
UP = 0
RIGHT = 1
DOWN = 2
LEFT = 3
DIRS = {
    UP: (-1, 0),
    RIGHT: (0, 1),
    DOWN: (1, 0),
    LEFT: (0, -1),
}
PORTAL_COLORS = tuple(range(1, 10))

GEN_RETRIES = 500
PAIR_COUNT_CHOICES = (1, 2, 3)
GRID_MIN = 14
GRID_MAX = 30
PORTAL_LEN_RANGE = (2, 4)
PATTERN_AREA_RANGE = (2, 6)
MAX_PATTERN_SIDE = 3
MAX_PLACE_ATTEMPTS = 120


def neighbors4(row: int, col: int) -> tuple[tuple[int, int], ...]:
    return ((row - 1, col), (row + 1, col), (row, col - 1), (row, col + 1))


def components_mask(mask: np.ndarray) -> tuple[np.ndarray, list[list[tuple[int, int]]]]:
    height, width = mask.shape
    labels = np.zeros((height, width), dtype=int)
    components: list[list[tuple[int, int]]] = []
    next_label = 1

    for row in range(height):
        for col in range(width):
            if not mask[row, col] or labels[row, col] != 0:
                continue
            queue = deque([(row, col)])
            labels[row, col] = next_label
            coords: list[tuple[int, int]] = []
            while queue:
                r, c = queue.popleft()
                coords.append((int(r), int(c)))
                for nr, nc in neighbors4(r, c):
                    if 0 <= nr < height and 0 <= nc < width and mask[nr, nc] and labels[nr, nc] == 0:
                        labels[nr, nc] = next_label
                        queue.append((nr, nc))
            components.append(coords)
            next_label += 1

    return labels, components


def bbox_coords(coords: list[tuple[int, int]]) -> tuple[int, int, int, int]:
    rows = [r for r, _ in coords]
    cols = [c for _, c in coords]
    return min(rows), min(cols), max(rows), max(cols)


def bbox_center(bbox: tuple[int, int, int, int]) -> tuple[float, float]:
    r0, c0, r1, c1 = bbox
    return ((r0 + r1) / 2.0, (c0 + c1) / 2.0)


def crop_nonzero(patch: np.ndarray) -> np.ndarray:
    rows, cols = np.where(patch != BLACK)
    assert rows.size > 0, "patch must contain foreground"
    return patch[rows.min() : rows.max() + 1, cols.min() : cols.max() + 1].copy()


def rotate_patch_cw(patch: np.ndarray, turns: int) -> np.ndarray:
    rotated = np.rot90(np.asarray(patch, dtype=int), -int(turns % 4))
    return crop_nonzero(rotated)


def rotate_vector_cw(vec: tuple[float, float], turns: int) -> tuple[float, float]:
    dr, dc = vec
    for _ in range(turns % 4):
        dr, dc = dc, -dr
    return dr, dc


def round_half_up(value: float) -> int:
    return int(np.floor(value + 0.5))


def portal_info_from_coords(coords: list[tuple[int, int]], color: int) -> dict[str, object] | None:
    if not (2 <= len(coords) <= 4):
        return None

    rows = sorted(r for r, _ in coords)
    cols = sorted(c for _, c in coords)
    row_set = sorted(set(rows))
    col_set = sorted(set(cols))

    if len(row_set) == 1:
        c0 = cols[0]
        if cols != list(range(c0, c0 + len(cols))):
            return None
        row = row_set[0]
        cells = [(row, c) for c in range(c0, c0 + len(cols))]
        return {
            "color": int(color),
            "axis": "h",
            "row": int(row),
            "c0": int(c0),
            "c1": int(c0 + len(cols) - 1),
            "cells": cells,
            "bbox": (int(row), int(c0), int(row), int(c0 + len(cols) - 1)),
        }

    if len(col_set) == 1:
        r0 = rows[0]
        if rows != list(range(r0, r0 + len(rows))):
            return None
        col = col_set[0]
        cells = [(r, col) for r in range(r0, r0 + len(rows))]
        return {
            "color": int(color),
            "axis": "v",
            "col": int(col),
            "r0": int(r0),
            "r1": int(r0 + len(rows) - 1),
            "cells": cells,
            "bbox": (int(r0), int(col), int(r0 + len(rows) - 1), int(col)),
        }

    return None


def portal_center(portal: dict[str, object]) -> tuple[float, float]:
    if portal["axis"] == "h":
        return float(portal["row"]), (int(portal["c0"]) + int(portal["c1"])) / 2.0
    return (int(portal["r0"]) + int(portal["r1"])) / 2.0, float(portal["col"])


def axis_vector(portal: dict[str, object]) -> tuple[float, float]:
    return (0.0, 1.0) if portal["axis"] == "h" else (1.0, 0.0)


def build_pattern_patch(grid: np.ndarray, coords: list[tuple[int, int]]) -> tuple[np.ndarray, tuple[int, int, int, int]]:
    bbox = bbox_coords(coords)
    r0, c0, r1, c1 = bbox
    patch = np.zeros((r1 - r0 + 1, c1 - c0 + 1), dtype=int)
    for row, col in coords:
        patch[row - r0, col - c0] = int(grid[row, col])
    return patch, bbox


def parse_scene(grid: np.ndarray) -> tuple[dict[int, list[dict[str, object]]], dict[int, dict[str, object]]]:
    g = np.asarray(grid, dtype=int)
    assert g.ndim == 2, "expected a 2D grid"

    portal_by_color: dict[int, list[dict[str, object]]] = {}
    portal_mask = np.zeros_like(g, dtype=bool)

    for color in sorted(int(c) for c in np.unique(g) if int(c) != BLACK):
        _, mono_components = components_mask(g == int(color))
        if len(mono_components) != 2:
            continue
        infos = []
        for coords in mono_components:
            info = portal_info_from_coords(coords, color)
            if info is None:
                infos = []
                break
            infos.append(info)
        if not infos:
            continue
        portal_by_color[int(color)] = infos
        for info in infos:
            for row, col in info["cells"]:
                portal_mask[row, col] = True

    assert portal_by_color, "expected at least one portal pair"

    labels, pattern_components = components_mask((g != BLACK) & (~portal_mask))
    pattern_info: dict[int, dict[str, object]] = {}
    for idx, coords in enumerate(pattern_components, start=1):
        patch, bbox = build_pattern_patch(g, coords)
        pattern_info[idx] = {"coords": coords, "bbox": bbox, "patch": patch}

    used_pattern_ids: set[int] = set()
    for color in sorted(portal_by_color):
        for portal in portal_by_color[color]:
            side_dirs = (UP, DOWN) if portal["axis"] == "h" else (LEFT, RIGHT)
            touching: dict[int, set[int]] = {}
            for direction in side_dirs:
                dr, dc = DIRS[direction]
                ids = set()
                for row, col in portal["cells"]:
                    nr, nc = row + dr, col + dc
                    if 0 <= nr < g.shape[0] and 0 <= nc < g.shape[1] and labels[nr, nc] != 0:
                        ids.add(int(labels[nr, nc]))
                touching[direction] = ids

            nonempty = [(direction, ids) for direction, ids in touching.items() if ids]
            assert len(nonempty) == 1, "each portal must have exactly one adjacent pattern side"
            pattern_side, ids = nonempty[0]
            assert len(ids) == 1, "portal must touch exactly one pattern component"
            other_side = DOWN if pattern_side == UP else UP if pattern_side == DOWN else RIGHT if pattern_side == LEFT else LEFT
            assert not touching[other_side], "the portal entrance side must be empty"

            pattern_id = next(iter(ids))
            used_pattern_ids.add(int(pattern_id))
            portal["pattern_side"] = int(pattern_side)
            portal["entrance"] = int((pattern_side + 2) % 4)
            portal["pattern_id"] = int(pattern_id)

    assert used_pattern_ids == set(pattern_info), "every non-portal component must belong to exactly one portal"
    return portal_by_color, pattern_info


def place_copy(
    out: np.ndarray,
    portal: dict[str, object],
    patch: np.ndarray,
    along_offset: float,
) -> None:
    patch = crop_nonzero(np.asarray(patch, dtype=int))
    height, width = patch.shape
    center_row, center_col = portal_center(portal)

    if portal["axis"] == "v":
        top = round_half_up(center_row + along_offset - (height - 1) / 2.0)
        left = int(portal["col"]) + 1 if int(portal["entrance"]) == RIGHT else int(portal["col"]) - width
    else:
        top = int(portal["row"]) + 1 if int(portal["entrance"]) == DOWN else int(portal["row"]) - height
        left = round_half_up(center_col + along_offset - (width - 1) / 2.0)

    assert 0 <= top and top + height <= out.shape[0], "teleported copy must stay in bounds"
    assert 0 <= left and left + width <= out.shape[1], "teleported copy must stay in bounds"

    for row, col in np.argwhere(patch != BLACK):
        rr = top + int(row)
        cc = left + int(col)
        assert out[rr, cc] in (BLACK, int(patch[row, col])), "teleported copy must not overwrite another object"
        out[rr, cc] = int(patch[row, col])


def main(grid: np.ndarray) -> np.ndarray:
    """
    Connect each same-colored portal pair and stamp every adjacent pattern beside the partner portal with the required rotation.

    Input structure:
    - Black background fills the grid.
    - Several colors appear as pairs of straight portal segments, each segment being a 2-4 cell horizontal or vertical line.
    - Every portal segment has exactly one small multicolor 4-connected pattern touching one side, while the opposite side is empty background.

    Rule:
    - Pair portal segments by color.
    - For each portal, read its entrance direction as the side opposite its adjacent pattern.
    - Copy that portal's pattern to the entrance side of the same-colored partner portal, keeping the original portals and patterns unchanged.
    - Rotate the copied pattern clockwise by the entrance-direction difference from source portal to target portal.
    - Place the rotated copy flush against the target entrance side and preserve its offset along the portal axis.

    Key insight:
    - The repeated color reveals the portal pairing, the pattern-touching side reveals each portal's orientation, and the clockwise gap between entrance directions determines the copy rotation.
    """
    g = np.asarray(grid, dtype=int)
    portal_by_color, pattern_info = parse_scene(g)

    out = g.copy()
    for color in sorted(portal_by_color):
        first, second = portal_by_color[color]
        for source, target in ((first, second), (second, first)):
            pattern = pattern_info[int(source["pattern_id"])]
            turns = (int(target["entrance"]) - int(source["entrance"])) % 4
            rotated = rotate_patch_cw(np.asarray(pattern["patch"], dtype=int), turns)

            src_pattern_center = bbox_center(pattern["bbox"])
            src_portal_center = portal_center(source)
            delta = (
                src_pattern_center[0] - src_portal_center[0],
                src_pattern_center[1] - src_portal_center[1],
            )
            delta_rot = rotate_vector_cw(delta, turns)
            axis_dr, axis_dc = axis_vector(target)
            along_offset = delta_rot[0] * axis_dr + delta_rot[1] * axis_dc

            place_copy(out, target, rotated, along_offset)

    return out


def make_canonical_pattern(rng: random.Random, palette: list[int]) -> np.ndarray:
    area = rng.randint(*PATTERN_AREA_RANGE)
    dims = [
        (height, width)
        for height in range(1, MAX_PATTERN_SIDE + 1)
        for width in range(1, MAX_PATTERN_SIDE + 1)
        if area <= height * width
    ]
    height, width = rng.choice(dims)

    cells = {(rng.randrange(height), width - 1)}
    while len(cells) < area:
        frontier = []
        for row, col in cells:
            for nr, nc in neighbors4(row, col):
                if 0 <= nr < height and 0 <= nc < width and (nr, nc) not in cells:
                    frontier.append((nr, nc))
        cells.add(rng.choice(frontier))

    rows = [r for r, _ in cells]
    cols = [c for _, c in cells]
    r0, r1 = min(rows), max(rows)
    c0, c1 = min(cols), max(cols)

    coords = sorted((r - r0, c - c0) for r, c in cells)
    patch = np.zeros((r1 - r0 + 1, c1 - c0 + 1), dtype=int)

    color_count = 2 if area == 2 else rng.randint(2, min(3, area))
    colors = rng.sample(palette, color_count)
    rng.shuffle(coords)
    for idx, (row, col) in enumerate(coords):
        color = colors[idx] if idx < color_count else rng.choice(colors)
        patch[row, col] = int(color)

    return crop_nonzero(patch)


def attachment_positions(patch: np.ndarray, direction: int) -> list[int]:
    if direction == RIGHT:
        return [int(row) for row in np.where(patch[:, -1] != BLACK)[0]]
    if direction == LEFT:
        return [int(row) for row in np.where(patch[:, 0] != BLACK)[0]]
    if direction == UP:
        return [int(col) for col in np.where(patch[0, :] != BLACK)[0]]
    return [int(col) for col in np.where(patch[-1, :] != BLACK)[0]]


def update_blocked(blocked: np.ndarray, cells: set[tuple[int, int]]) -> None:
    height, width = blocked.shape
    for row, col in cells:
        for dr in (-1, 0, 1):
            for dc in (-1, 0, 1):
                nr, nc = row + dr, col + dc
                if 0 <= nr < height and 0 <= nc < width:
                    blocked[nr, nc] = True


def place_one_portal(
    spec: dict[str, object],
    height: int,
    width: int,
    blocked: np.ndarray,
    rng: random.Random,
) -> dict[str, object] | None:
    patch = np.asarray(spec["patch"], dtype=int)
    patch_h, patch_w = patch.shape
    boundary_positions = attachment_positions(patch, int(spec["entrance"]))
    assert boundary_positions, "pattern must expose at least one portal-facing boundary cell"

    for _ in range(MAX_PLACE_ATTEMPTS):
        if spec["axis"] == "v":
            start_row = rng.randint(1, height - int(spec["length"]) - 2)
            col = rng.randint(3, width - 4)
            portal_cells = {(start_row + offset, col) for offset in range(int(spec["length"]))}
            anchor_patch = rng.choice(boundary_positions)
            anchor_portal = start_row + rng.randrange(int(spec["length"]))
            top = anchor_portal - anchor_patch
            left = col - patch_w if int(spec["entrance"]) == RIGHT else col + 1
        else:
            row = rng.randint(3, height - 4)
            start_col = rng.randint(1, width - int(spec["length"]) - 2)
            portal_cells = {(row, start_col + offset) for offset in range(int(spec["length"]))}
            anchor_patch = rng.choice(boundary_positions)
            anchor_portal = start_col + rng.randrange(int(spec["length"]))
            top = row + 1 if int(spec["entrance"]) == UP else row - patch_h
            left = anchor_portal - anchor_patch

        if top < 0 or left < 0 or top + patch_h > height or left + patch_w > width:
            continue

        pattern_cells = {
            (top + int(row), left + int(col))
            for row, col in np.argwhere(patch != BLACK)
        }
        all_cells = portal_cells | pattern_cells
        if any(blocked[row, col] for row, col in all_cells):
            continue

        placement = {
            "pair_idx": int(spec["pair_idx"]),
            "portal_color": int(spec["portal_color"]),
            "entrance": int(spec["entrance"]),
            "axis": spec["axis"],
            "length": int(spec["length"]),
            "patch": patch.copy(),
            "pattern_size": int(spec["pattern_size"]),
            "portal_cells": portal_cells,
            "pattern_cells": pattern_cells,
        }
        return placement

    return None


def draws_ambiguous_non_portal_pair(grid: np.ndarray, portal_colors: set[int]) -> bool:
    for color in sorted(int(c) for c in np.unique(grid) if int(c) not in portal_colors and int(c) != BLACK):
        _, components = components_mask(grid == int(color))
        if len(components) != 2:
            continue
        if all(portal_info_from_coords(coords, color) is not None for coords in components):
            return True
    return False


def make_specs(pair_count: int, rng: random.Random) -> tuple[list[dict[str, object]], list[int]]:
    portal_colors = rng.sample(list(PORTAL_COLORS), pair_count)
    pattern_palette = [color for color in PORTAL_COLORS if color not in portal_colors]
    specs: list[dict[str, object]] = []

    for pair_idx, portal_color in enumerate(portal_colors):
        for _ in range(2):
            entrance = rng.randrange(4)
            axis = "v" if entrance in (LEFT, RIGHT) else "h"
            pattern = make_canonical_pattern(rng, pattern_palette)
            turns = (entrance - RIGHT) % 4
            oriented = rotate_patch_cw(pattern, turns)
            specs.append(
                {
                    "pair_idx": int(pair_idx),
                    "portal_color": int(portal_color),
                    "entrance": int(entrance),
                    "axis": axis,
                    "length": int(rng.randint(*PORTAL_LEN_RANGE)),
                    "patch": oriented,
                    "pattern_size": int(np.count_nonzero(oriented)),
                }
            )

    rng.shuffle(specs)
    return specs, portal_colors


def place_patch_on_grid(grid: np.ndarray, cells: set[tuple[int, int]], patch: np.ndarray) -> None:
    rows = [r for r, _ in cells]
    cols = [c for _, c in cells]
    top, left = min(rows), min(cols)
    for row, col in np.argwhere(patch != BLACK):
        grid[top + int(row), left + int(col)] = int(patch[row, col])


def generate_candidate(rng: random.Random) -> np.ndarray | None:
    pair_count = rng.choice(PAIR_COUNT_CHOICES)
    min_side = max(GRID_MIN, 10 + 4 * pair_count)
    height = rng.randint(min_side, GRID_MAX)
    width = rng.randint(min_side, GRID_MAX)
    specs, portal_colors = make_specs(pair_count, rng)

    blocked = np.zeros((height, width), dtype=bool)
    placements: list[dict[str, object]] = []
    for spec in specs:
        placement = place_one_portal(spec, height, width, blocked, rng)
        if placement is None:
            return None
        placements.append(placement)
        update_blocked(blocked, placement["portal_cells"] | placement["pattern_cells"])

    grid = np.zeros((height, width), dtype=int)
    for placement in placements:
        for row, col in placement["portal_cells"]:
            grid[row, col] = int(placement["portal_color"])
        place_patch_on_grid(grid, placement["pattern_cells"], np.asarray(placement["patch"], dtype=int))

    if draws_ambiguous_non_portal_pair(grid, set(portal_colors)):
        return None

    expected_added = sum(int(placement["pattern_size"]) for placement in placements)
    try:
        out = main(grid)
    except AssertionError:
        return None

    if np.array_equal(out, grid):
        return None
    if int(np.count_nonzero(out != grid)) != expected_added:
        return None
    return grid


def generate_input() -> np.ndarray:
    for _ in range(GEN_RETRIES):
        seed = random.randrange(1 << 30)
        rng = random.Random(seed)
        grid = generate_candidate(rng)
        if grid is not None:
            return grid
    raise RuntimeError("failed to generate a valid portal-rotation input")
