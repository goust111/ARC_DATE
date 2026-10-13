from __future__ import annotations

import random

import numpy as np


BACKGROUND = 0
MARKER = 5
SHAPE_COLORS = (1, 2, 3, 4, 6, 7, 8, 9)
DIR4 = ((1, 0), (-1, 0), (0, 1), (0, -1))


def _bbox_coords(coords: np.ndarray) -> tuple[int, int, int, int]:
    r0, c0 = coords.min(axis=0)
    r1, c1 = coords.max(axis=0)
    return int(r0), int(c0), int(r1), int(c1)


def _bbox(mask: np.ndarray) -> tuple[int, int, int, int] | None:
    coords = np.argwhere(mask)
    if coords.size == 0:
        return None
    return _bbox_coords(coords)


def _crop_bbox(grid: np.ndarray, box: tuple[int, int, int, int]) -> np.ndarray:
    r0, c0, r1, c1 = box
    return grid[r0 : r1 + 1, c0 : c1 + 1].copy()


def _crop_mask(mask: np.ndarray) -> np.ndarray:
    box = _bbox(mask)
    assert box is not None, "expected a non-empty mask"
    return _crop_bbox(mask, box)


def _find_monochromatic_components(grid: np.ndarray, valid_mask: np.ndarray) -> list[dict]:
    g = np.asarray(grid, dtype=int)
    seen = np.zeros(g.shape, dtype=bool)
    components: list[dict] = []
    height, width = g.shape

    for r in range(height):
        for c in range(width):
            if seen[r, c] or not valid_mask[r, c]:
                continue
            color = int(g[r, c])
            stack = [(r, c)]
            seen[r, c] = True
            coords: list[tuple[int, int]] = []

            while stack:
                rr, cc = stack.pop()
                coords.append((rr, cc))
                for dr, dc in DIR4:
                    nr, nc = rr + dr, cc + dc
                    if (
                        0 <= nr < height
                        and 0 <= nc < width
                        and not seen[nr, nc]
                        and valid_mask[nr, nc]
                        and int(g[nr, nc]) == color
                    ):
                        seen[nr, nc] = True
                        stack.append((nr, nc))

            coord_array = np.asarray(coords, dtype=int)
            components.append(
                {
                    "color": color,
                    "coords": coord_array,
                    "bbox": _bbox_coords(coord_array),
                    "area": int(coord_array.shape[0]),
                }
            )

    return components


def _component_mask(component: dict) -> np.ndarray:
    r0, c0, r1, c1 = component["bbox"]
    local = component["coords"] - np.array([r0, c0])
    mask = np.zeros((r1 - r0 + 1, c1 - c0 + 1), dtype=bool)
    mask[local[:, 0], local[:, 1]] = True
    return mask


def _component_center(box: tuple[int, int, int, int]) -> tuple[float, float]:
    r0, c0, r1, c1 = box
    return (r0 + r1) / 2.0, (c0 + c1) / 2.0


def _rotated_mask(mask: np.ndarray, turns: int) -> np.ndarray:
    rotated = np.rot90(mask, -int(turns) % 4)
    return _crop_mask(rotated)


def _analyze_layers(grid: np.ndarray) -> tuple[list[dict], tuple[int, int]]:
    g = np.asarray(grid, dtype=int)
    assert g.ndim == 2 and g.size > 0, "expected a non-empty 2D grid"

    shape_mask = (g != BACKGROUND) & (g != MARKER)
    components = _find_monochromatic_components(g, shape_mask)
    assert components, "expected at least one colored shape"

    marker_coords = np.argwhere(g == MARKER)
    centers = [_component_center(comp["bbox"]) for comp in components]
    marker_counts = [0] * len(components)
    for mr, mc in marker_coords:
        distances = [(mr - cr) ** 2 + (mc - cc) ** 2 for cr, cc in centers]
        best = min(distances)
        owners = [idx for idx, dist in enumerate(distances) if abs(dist - best) < 1e-9]
        assert len(owners) == 1, "marker ownership must be unambiguous"
        marker_counts[owners[0]] += 1

    layers: list[dict] = []
    max_height = 0
    max_width = 0
    for component, turns in zip(components, marker_counts):
        assert 0 <= turns <= 3, "each shape may have only 0 to 3 markers"
        base_mask = _component_mask(component)
        rotated = _rotated_mask(base_mask, turns)
        height, width = map(int, rotated.shape)
        max_height = max(max_height, height)
        max_width = max(max_width, width)
        layers.append(
            {
                "color": int(component["color"]),
                "mask": rotated,
                "bbox_area": int(height * width),
                "input_key": (int(component["bbox"][0]), int(component["bbox"][1])),
            }
        )

    layers.sort(key=lambda layer: (-layer["bbox_area"], layer["input_key"]))
    return layers, (max_height, max_width)


def main(grid: np.ndarray) -> np.ndarray:
    """
    Rotate each colored shape by its nearby gray-marker count, then stack the rotated shapes at the upper-left corner.

    Input structure:
    - The grid has a black background, 3-5 scattered monochrome 4-connected shapes, and optional gray marker pixels.
    - Each shape uses a distinct non-black, non-gray color and has a non-square bounding box.
    - Gray pixels are isolated control markers placed near shapes; each marker is uniquely closest to one shape's bounding-box center.

    Rule:
    - Count how many gray marker pixels belong to each shape.
    - Rotate every shape clockwise by that many quarter-turns around its bounding-box center, then recrop it to its tight bounding box.
    - Create a black output whose height and width match the largest rotated bounding-box height and width.
    - Paint all rotated shapes at output origin (0,0), ordered by decreasing rotated bounding-box area, breaking ties by the input bounding-box top-left coordinate; later shapes overwrite earlier ones.

    Key insight:
    - The gray pixels do not move to the output; they only encode how far each shape rotates before all rotated shapes are restacked on the same upper-left anchor.
    """
    layers, (out_height, out_width) = _analyze_layers(grid)
    out = np.full((out_height, out_width), BACKGROUND, dtype=int)
    for layer in layers:
        mask = layer["mask"]
        height, width = mask.shape
        out[:height, :width][mask] = int(layer["color"])
    return out


def _is_connected(mask: np.ndarray) -> bool:
    coords = np.argwhere(mask)
    if coords.size == 0:
        return False
    start = tuple(map(int, coords[0]))
    seen = {start}
    stack = [start]
    height, width = mask.shape

    while stack:
        r, c = stack.pop()
        for dr, dc in DIR4:
            nr, nc = r + dr, c + dc
            if 0 <= nr < height and 0 <= nc < width and mask[nr, nc] and (nr, nc) not in seen:
                seen.add((nr, nc))
                stack.append((nr, nc))
    return len(seen) == int(coords.shape[0])


def _random_shape_mask(required_turns: int) -> np.ndarray | None:
    for _ in range(600):
        height = random.randint(2, 4)
        width = random.randint(2, 5)
        if height == width:
            continue

        area = random.randint(max(3, max(height, width)), min(8, height * width))
        mask = np.zeros((height, width), dtype=bool)
        seed_r = random.randrange(height)
        seed_c = random.randrange(width)
        mask[seed_r, seed_c] = True

        while int(mask.sum()) < area:
            frontier: list[tuple[int, int]] = []
            frontier_seen = set()
            for r, c in np.argwhere(mask):
                for dr, dc in DIR4:
                    nr, nc = int(r + dr), int(c + dc)
                    if 0 <= nr < height and 0 <= nc < width and not mask[nr, nc] and (nr, nc) not in frontier_seen:
                        frontier_seen.add((nr, nc))
                        frontier.append((nr, nc))
            if not frontier:
                break
            pick_r, pick_c = random.choice(frontier)
            mask[pick_r, pick_c] = True

        if int(mask.sum()) != area or not _is_connected(mask):
            continue
        if not (mask[0].any() and mask[-1].any() and mask[:, 0].any() and mask[:, -1].any()):
            continue
        if np.all(mask):
            continue

        mask = _crop_mask(mask)
        rotated = _rotated_mask(mask, required_turns)
        if required_turns and np.array_equal(mask, rotated):
            continue
        return mask

    return None


def _choose_rotation_counts(n_shapes: int) -> list[int]:
    counts = [0, 1, 2, 3]
    if n_shapes == 3:
        counts.remove(random.choice(counts))
    elif n_shapes == 5:
        counts.append(random.choice((0, 1, 2, 3)))
    random.shuffle(counts)
    return counts


def _choose_slots(height: int, width: int, n_shapes: int) -> list[tuple[int, int, int, int]]:
    if n_shapes <= 4:
        slot_rows, slot_cols = 2, 2
    else:
        slot_rows, slot_cols = 2, 3

    slots = []
    for rr in range(slot_rows):
        for cc in range(slot_cols):
            r0 = rr * height // slot_rows
            r1 = (rr + 1) * height // slot_rows - 1
            c0 = cc * width // slot_cols
            c1 = (cc + 1) * width // slot_cols - 1
            slots.append((r0, c0, r1, c1))
    random.shuffle(slots)
    return slots[:n_shapes]


def _marker_candidates(
    box: tuple[int, int, int, int], occupied_shapes: np.ndarray
) -> list[tuple[int, int]]:
    r0, c0, r1, c1 = box
    height, width = occupied_shapes.shape

    candidates: list[tuple[int, int]] = []
    for c in range(c0, c1 + 1):
        candidates.append((r0 - 2, c))
        candidates.append((r1 + 2, c))
    for r in range(r0, r1 + 1):
        candidates.append((r, c0 - 2))
        candidates.append((r, c1 + 2))

    valid: list[tuple[int, int]] = []
    seen = set()
    for r, c in candidates:
        if (r, c) in seen or not (0 <= r < height and 0 <= c < width):
            continue
        if occupied_shapes[r, c]:
            continue
        if any(
            0 <= r + dr < height and 0 <= c + dc < width and occupied_shapes[r + dr, c + dc]
            for dr, dc in DIR4
        ):
            continue
        seen.add((r, c))
        valid.append((r, c))

    random.shuffle(valid)
    return valid


def _overlap_cells(layers: list[dict], out_shape: tuple[int, int]) -> int:
    occupancy = np.zeros(out_shape, dtype=int)
    for layer in layers:
        mask = layer["mask"]
        height, width = mask.shape
        occupancy[:height, :width] += mask.astype(int)
    return int(np.sum(occupancy >= 2))


def _attempt_generate() -> np.ndarray | None:
    n_shapes = random.choices((3, 4, 5), weights=(1, 2, 2), k=1)[0]
    if n_shapes <= 4:
        height = random.randint(18, 26)
        width = random.randint(18, 26)
    else:
        height = random.randint(18, 24)
        width = random.randint(24, 30)

    grid = np.full((height, width), BACKGROUND, dtype=int)
    occupied_shapes = np.zeros((height, width), dtype=bool)
    counts = _choose_rotation_counts(n_shapes)
    colors = random.sample(SHAPE_COLORS, n_shapes)
    slots = _choose_slots(height, width, n_shapes)

    placed_shapes: list[dict] = []
    for slot, turns, color in zip(slots, counts, colors):
        mask = _random_shape_mask(turns)
        if mask is None:
            return None

        shape_h, shape_w = mask.shape
        slot_r0, slot_c0, slot_r1, slot_c1 = slot
        row_choices = list(range(slot_r0 + 2, slot_r1 - shape_h + 1))
        col_choices = list(range(slot_c0 + 2, slot_c1 - shape_w + 1))
        if not row_choices or not col_choices:
            return None

        top = random.choice(row_choices)
        left = random.choice(col_choices)
        window = occupied_shapes[top : top + shape_h, left : left + shape_w]
        if np.any(window & mask):
            return None

        grid[top : top + shape_h, left : left + shape_w][mask] = int(color)
        occupied_shapes[top : top + shape_h, left : left + shape_w] |= mask
        placed_shapes.append(
            {
                "bbox": (top, left, top + shape_h - 1, left + shape_w - 1),
                "turns": int(turns),
                "mask": mask,
            }
        )

    centers = [_component_center(shape["bbox"]) for shape in placed_shapes]
    used_markers: list[tuple[int, int]] = []
    any_rotational_change = False

    for index, shape in enumerate(placed_shapes):
        turns = shape["turns"]
        if turns:
            rotated = _rotated_mask(shape["mask"], turns)
            if not np.array_equal(rotated, shape["mask"]):
                any_rotational_change = True

        need = turns
        if need == 0:
            continue

        chosen: list[tuple[int, int]] = []
        for r, c in _marker_candidates(shape["bbox"], occupied_shapes):
            if any(abs(r - mr) + abs(c - mc) <= 1 for mr, mc in used_markers):
                continue

            distances = [(r - cr) ** 2 + (c - cc) ** 2 for cr, cc in centers]
            best = min(distances)
            owners = [idx for idx, dist in enumerate(distances) if abs(dist - best) < 1e-9]
            if owners != [index]:
                continue

            chosen.append((r, c))
            used_markers.append((r, c))
            if len(chosen) == need:
                break

        if len(chosen) != need:
            return None
        for r, c in chosen:
            grid[r, c] = MARKER

    if not any_rotational_change:
        return None

    try:
        out = main(grid)
    except AssertionError:
        return None

    if max(grid.shape) > 30 or max(out.shape) > 30:
        return None
    if len(np.unique(out[out != BACKGROUND])) < 2:
        return None

    layers, out_shape = _analyze_layers(grid)
    if _overlap_cells(layers, out_shape) == 0:
        return None
    if int(np.count_nonzero(out)) >= sum(int(layer["mask"].sum()) for layer in layers):
        return None

    return grid


def generate_input() -> np.ndarray:
    for _ in range(600):
        grid = _attempt_generate()
        if grid is not None:
            return grid
    raise RuntimeError("failed to generate a valid input for puzzle 1")
