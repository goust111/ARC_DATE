from __future__ import annotations

import random
import sys
from collections import Counter
from itertools import combinations
from pathlib import Path

import numpy as np


def _ensure_arc_utils_path() -> None:
    here = Path(__file__).resolve()
    for parent in here.parents:
        candidate = parent / "new" / "solvers"
        if candidate.is_dir():
            path = str(candidate)
            if path not in sys.path:
                sys.path.append(path)
            return

    fallback = Path("/home/lixinyu/snap/new/solvers")
    if fallback.is_dir():
        path = str(fallback)
        if path not in sys.path:
            sys.path.append(path)
        return

    raise ImportError("could not locate new/solvers/arc_utils.py")


_ensure_arc_utils_path()

from arc_utils import Color, bbox_coords, components_mask  # noqa: E402


BG = int(Color.BLACK)
RED = int(Color.RED)
BLUE = int(Color.BLUE)
GREEN = int(Color.GREEN)

OBJECT_COUNT_RANGE = (5, 7)
GRID_SIDE_MIN = 15
GRID_SIDE_MAX = 24
BBOX_GAP = 2
SHAPE_SIZE_OPTIONS = ((3, 3), (3, 4), (4, 3))
MAX_GENERATION_ATTEMPTS = 400
CLIPPED_OUTLINE_PROB = 0.45


def _mask_key(mask: np.ndarray) -> tuple[tuple[int, ...], ...]:
    return tuple(tuple(int(v) for v in row) for row in mask.astype(int))


def _coords_to_mask(coords: np.ndarray) -> np.ndarray:
    r0, c0, r1, c1 = bbox_coords(coords)
    mask = np.zeros((r1 - r0 + 1, c1 - c0 + 1), dtype=bool)
    shifted = coords - np.array([r0, c0], dtype=int)
    mask[shifted[:, 0], shifted[:, 1]] = True
    return mask


def _component_records(grid: np.ndarray) -> list[dict]:
    components = []
    for coords in components_mask(grid == RED, connectivity=4):
        coords = coords.astype(int)
        bbox = bbox_coords(coords)
        components.append(
            {
                "coords": coords,
                "bbox": bbox,
                "mask": _coords_to_mask(coords),
                "area": int(coords.shape[0]),
            }
        )
    components.sort(key=lambda comp: (comp["bbox"][0], comp["bbox"][1]))
    return components


def _expanded_perimeter_cells(shape: tuple[int, int], bbox: tuple[int, int, int, int]) -> list[tuple[int, int]]:
    h, w = shape
    r0, c0, r1, c1 = bbox
    er0 = max(0, r0 - 1)
    ec0 = max(0, c0 - 1)
    er1 = min(h - 1, r1 + 1)
    ec1 = min(w - 1, c1 + 1)
    cells = []
    for r in range(er0, er1 + 1):
        for c in range(ec0, ec1 + 1):
            if r in (er0, er1) or c in (ec0, ec1):
                cells.append((r, c))
    return cells


def _paint_outline(out: np.ndarray, bbox: tuple[int, int, int, int]) -> None:
    for r, c in _expanded_perimeter_cells(out.shape, bbox):
        if int(out[r, c]) == BG:
            out[r, c] = GREEN


def _paint_interior_fill(out: np.ndarray, bbox: tuple[int, int, int, int]) -> None:
    r0, c0, r1, c1 = bbox
    for r in range(r0 + 1, r1):
        for c in range(c0 + 1, c1):
            if int(out[r, c]) == BG:
                out[r, c] = BLUE


def main(grid: np.ndarray) -> np.ndarray:
    """
    Choose between outlining and bbox-interior filling based on how often each red object shape appears.

    Input structure:
    - The grid is a black background containing only red 4-connected foreground objects.
    - Each object has area 4 to 8 and a tight bounding box of at least 3 by 3.
    - Up to translation within its tight bounding box, exactly one shape appears once and one other shape appears multiple times.

    Rule:
    - Find every red 4-connected object and normalize its shape by cropping to its tight bounding box.
    - Group objects by that translation-only normalized mask and count each shape frequency.
    - For the singleton shape, expand its bounding box outward by one cell, clip to the grid, and paint green on the expanded perimeter wherever the cell is black.
    - For the repeated shape, fill every black cell strictly inside each instance's tight bounding box with blue.
    - Keep every original red pixel unchanged and process objects in top-left bbox order.

    Key insight:
    - The operation choice is global: an object's shape must be compared against all other objects before deciding whether it gets an outline or an interior fill.
    """
    g = np.asarray(grid, dtype=int)
    assert g.ndim == 2 and g.size > 0
    assert set(map(int, np.unique(g))).issubset({BG, RED})

    components = _component_records(g)
    assert components

    shape_keys = [_mask_key(comp["mask"]) for comp in components]
    counts = Counter(shape_keys)
    singleton_keys = {key for key, count in counts.items() if count == 1}
    repeated_keys = {key for key, count in counts.items() if count > 1}
    assert len(singleton_keys) == 1
    assert len(repeated_keys) == 1

    out = g.copy()
    for comp, key in zip(components, shape_keys):
        if counts[key] == 1:
            _paint_outline(out, comp["bbox"])
        else:
            _paint_interior_fill(out, comp["bbox"])
    return out


def _mask_is_valid(mask: np.ndarray) -> bool:
    h, w = mask.shape
    area = int(mask.sum())
    if not (4 <= area <= 8):
        return False
    if h < 3 or w < 3 or bool(mask.all()):
        return False
    if mask[1 : h - 1, 1 : w - 1].size == 0 or bool(mask[1 : h - 1, 1 : w - 1].all()):
        return False
    coords = np.argwhere(mask)
    if len(components_mask(mask, connectivity=4)) != 1:
        return False
    return bbox_coords(coords) == (0, 0, h - 1, w - 1)


def _dihedral_keys(mask: np.ndarray) -> set[tuple[tuple[int, ...], ...]]:
    keys = set()
    for k in range(4):
        rotated = np.rot90(mask, k)
        keys.add(_mask_key(rotated))
        keys.add(_mask_key(np.fliplr(rotated)))
    return keys


def _build_shape_pools() -> tuple[dict[tuple[int, int, int], list[np.ndarray]], list[tuple[int, int, int]]]:
    pools: dict[tuple[int, int, int], list[np.ndarray]] = {}
    pairable: list[tuple[int, int, int]] = []

    for h, w in SHAPE_SIZE_OPTIONS:
        cell_count = h * w
        for area in range(4, min(8, cell_count) + 1):
            masks = []
            for picks in combinations(range(cell_count), area):
                mask = np.zeros((h, w), dtype=bool)
                for idx in picks:
                    mask[idx // w, idx % w] = True
                if _mask_is_valid(mask):
                    masks.append(mask)
            if len(masks) < 2:
                continue
            pools[(h, w, area)] = masks

            for i, first in enumerate(masks):
                forbidden = _dihedral_keys(first)
                if any(_mask_key(second) not in forbidden for second in masks[i + 1 :]):
                    pairable.append((h, w, area))
                    break

    return pools, pairable


SHAPE_POOLS, PAIRABLE_SIGNATURES = _build_shape_pools()
assert PAIRABLE_SIGNATURES


def _sample_shape_pair() -> tuple[np.ndarray, np.ndarray]:
    h, w, area = random.choice(PAIRABLE_SIGNATURES)
    pool = SHAPE_POOLS[(h, w, area)]
    first = random.choice(pool)
    forbidden = _dihedral_keys(first)
    second_choices = [mask for mask in pool if _mask_key(mask) not in forbidden]
    assert second_choices
    second = random.choice(second_choices)
    return first.copy(), second.copy()


def _random_composition(total: int, parts: int, zero_index: int | None = None) -> list[int]:
    values = [0] * parts
    choices = [i for i in range(parts) if i != zero_index]
    for _ in range(total):
        values[random.choice(choices)] += 1
    return values


def _starts_from_margins(lengths: int, item_size: int, gap: int, margins: list[int]) -> list[int]:
    starts = []
    pos = margins[0]
    for idx in range(lengths):
        starts.append(pos)
        pos += item_size
        if idx < lengths - 1:
            pos += gap + margins[idx + 1]
    return starts


def _bbox_from_pos(r: int, c: int, h: int, w: int) -> tuple[int, int, int, int]:
    return r, c, r + h - 1, c + w - 1


def _bboxes_are_separated(a: tuple[int, int, int, int], b: tuple[int, int, int, int], gap: int = BBOX_GAP) -> bool:
    return (
        a[2] + gap < b[0]
        or b[2] + gap < a[0]
        or a[3] + gap < b[1]
        or b[3] + gap < a[1]
    )


def _edge_sides(slot: tuple[int, int], slot_rows: int, slot_cols: int) -> list[str]:
    sr, sc = slot
    sides = []
    if sr == 0:
        sides.append("top")
    if sr == slot_rows - 1:
        sides.append("bottom")
    if sc == 0:
        sides.append("left")
    if sc == slot_cols - 1:
        sides.append("right")
    return sides


def _layout_positions(
    count: int,
    bbox_hw: tuple[int, int],
    require_clipped_outline: bool,
) -> tuple[tuple[int, int], list[tuple[int, int]], tuple[int, int]] | None:
    h, w = bbox_hw
    slot_options = []
    for slot_rows in range(2, count + 1):
        slot_cols = (count + slot_rows - 1) // slot_rows
        min_rows = slot_rows * h + (slot_rows - 1) * BBOX_GAP
        min_cols = slot_cols * w + (slot_cols - 1) * BBOX_GAP
        if min_rows <= GRID_SIDE_MAX and min_cols <= GRID_SIDE_MAX:
            slot_options.append((slot_rows, slot_cols, min_rows, min_cols))
    if not slot_options:
        return None

    slot_rows, slot_cols, min_rows, min_cols = random.choice(slot_options)
    rows = random.randint(max(GRID_SIDE_MIN, min_rows), GRID_SIDE_MAX)
    cols = random.randint(max(GRID_SIDE_MIN, min_cols), GRID_SIDE_MAX)
    row_slack = rows - min_rows
    col_slack = cols - min_cols

    all_slots = [(sr, sc) for sr in range(slot_rows) for sc in range(slot_cols)]
    boundary_slots = [slot for slot in all_slots if _edge_sides(slot, slot_rows, slot_cols)]
    if require_clipped_outline and not boundary_slots:
        return None

    unique_slot = random.choice(boundary_slots if require_clipped_outline else all_slots)
    remaining_slots = [slot for slot in all_slots if slot != unique_slot]
    chosen_slots = [unique_slot] + random.sample(remaining_slots, count - 1)

    row_zero = None
    col_zero = None
    if require_clipped_outline:
        side = random.choice(_edge_sides(unique_slot, slot_rows, slot_cols))
        if side == "top":
            row_zero = 0
        elif side == "bottom":
            row_zero = slot_rows
        elif side == "left":
            col_zero = 0
        else:
            col_zero = slot_cols

    row_margins = _random_composition(row_slack, slot_rows + 1, zero_index=row_zero)
    col_margins = _random_composition(col_slack, slot_cols + 1, zero_index=col_zero)
    row_starts = _starts_from_margins(slot_rows, h, BBOX_GAP, row_margins)
    col_starts = _starts_from_margins(slot_cols, w, BBOX_GAP, col_margins)

    positions = [(row_starts[sr], col_starts[sc]) for sr, sc in chosen_slots]
    random.shuffle(positions[1:])
    return positions[0], positions, (rows, cols)


def _render_objects(
    grid_shape: tuple[int, int],
    unique_mask: np.ndarray,
    repeated_mask: np.ndarray,
    positions: list[tuple[int, int]],
    unique_pos: tuple[int, int],
) -> np.ndarray:
    h, w = unique_mask.shape
    grid = np.full(grid_shape, BG, dtype=int)
    for pos in positions:
        mask = unique_mask if pos == unique_pos else repeated_mask
        r0, c0 = pos
        patch = np.where(mask, RED, BG).astype(int)
        grid[r0 : r0 + h, c0 : c0 + w] = patch
    return grid


def _oracle_output(grid: np.ndarray) -> np.ndarray:
    components = _component_records(grid)
    shape_keys = [_mask_key(comp["mask"]) for comp in components]
    counts = Counter(shape_keys)
    out = grid.copy()
    for comp, key in zip(components, shape_keys):
        if counts[key] == 1:
            _paint_outline(out, comp["bbox"])
        else:
            _paint_interior_fill(out, comp["bbox"])
    return out


def _bbox_cells(box: tuple[int, int, int, int]) -> set[tuple[int, int]]:
    r0, c0, r1, c1 = box
    return {(r, c) for r in range(r0, r1 + 1) for c in range(c0, c1 + 1)}


def _validate_instance(grid: np.ndarray, require_clipped_outline: bool) -> bool:
    if not (isinstance(grid, np.ndarray) and grid.ndim == 2):
        return False
    if max(grid.shape) > 30 or not set(map(int, np.unique(grid))).issubset({BG, RED}):
        return False

    components = _component_records(grid)
    if not (OBJECT_COUNT_RANGE[0] <= len(components) <= OBJECT_COUNT_RANGE[1]):
        return False

    for idx, comp in enumerate(components):
        box = comp["bbox"]
        if not (4 <= comp["area"] <= 8):
            return False
        if (box[2] - box[0] + 1) < 3 or (box[3] - box[1] + 1) < 3:
            return False
        for other in components[idx + 1 :]:
            if not _bboxes_are_separated(box, other["bbox"]):
                return False

    shape_keys = [_mask_key(comp["mask"]) for comp in components]
    counts = Counter(shape_keys)
    if len(counts) != 2:
        return False
    if sorted(counts.values()) != [1, len(components) - 1]:
        return False

    unique_key = next(key for key, count in counts.items() if count == 1)
    repeated_key = next(key for key, count in counts.items() if count > 1)
    unique_comp = next(comp for comp, key in zip(components, shape_keys) if key == unique_key)
    repeated_comp = next(comp for comp, key in zip(components, shape_keys) if key == repeated_key)

    if unique_comp["mask"].shape != repeated_comp["mask"].shape:
        return False
    if unique_comp["area"] != repeated_comp["area"]:
        return False
    if _mask_key(repeated_comp["mask"]) in _dihedral_keys(unique_comp["mask"]):
        return False

    unique_box = unique_comp["bbox"]
    unique_touches_edge = (
        unique_box[0] == 0
        or unique_box[1] == 0
        or unique_box[2] == grid.shape[0] - 1
        or unique_box[3] == grid.shape[1] - 1
    )
    if require_clipped_outline and not unique_touches_edge:
        return False

    frame_cells = set(_expanded_perimeter_cells(grid.shape, unique_box))
    if not any(grid[r, c] == BG for r, c in frame_cells):
        return False
    for comp in components:
        if comp is unique_comp:
            continue
        if frame_cells & _bbox_cells(comp["bbox"]):
            return False

    expected = _oracle_output(grid)
    if np.array_equal(expected, grid):
        return False

    green_count = sum(1 for r, c in frame_cells if expected[r, c] == GREEN)
    if green_count == 0:
        return False

    for comp, key in zip(components, shape_keys):
        if key != repeated_key:
            continue
        r0, c0, r1, c1 = comp["bbox"]
        interior = expected[r0 + 1 : r1, c0 + 1 : c1]
        if interior.size == 0 or not np.any(interior == BLUE):
            return False

    try:
        actual = main(grid)
    except Exception:
        return False
    return np.array_equal(actual, expected)


def _gen_attempt() -> np.ndarray | None:
    total_objects = random.randint(*OBJECT_COUNT_RANGE)
    repeated_mask, unique_mask = _sample_shape_pair()
    require_clipped_outline = random.random() < CLIPPED_OUTLINE_PROB

    layout = _layout_positions(total_objects, repeated_mask.shape, require_clipped_outline)
    if layout is None:
        return None
    unique_pos, positions, grid_shape = layout
    grid = _render_objects(grid_shape, unique_mask, repeated_mask, positions, unique_pos)

    if random.random() < 0.5:
        grid = np.fliplr(grid)
    grid = np.rot90(grid, random.randint(0, 3)).copy()

    if not _validate_instance(grid, require_clipped_outline):
        return None
    return grid


def generate_input() -> np.ndarray:
    for _ in range(MAX_GENERATION_ATTEMPTS):
        grid = _gen_attempt()
        if grid is not None:
            return grid
    raise RuntimeError("failed to generate a valid frequency-based shape transform input")
