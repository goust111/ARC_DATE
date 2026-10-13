from __future__ import annotations

import random
from collections import deque

import numpy as np


BG_COLOR = 0
BAR_COLOR = 5
OBJECT_COLORS = [1, 2, 3, 4, 7]

GRID_HEIGHT_RANGE = (12, 17)
GRID_WIDTH_RANGE = (13, 19)
OBJECT_COUNT_RANGE = (3, 5)
OBJECT_HEIGHT_MAX = 7
OBJECT_WIDTH_RANGE = (2, 4)
OBJECT_AREA_RANGE = (4, 8)
RETRY_LIMIT = 600
CLIP_PROBABILITY = 0.35
PLACEMENT_RETRIES = 120


def _neighbors4(r: int, c: int) -> list[tuple[int, int]]:
    return [(r - 1, c), (r + 1, c), (r, c - 1), (r, c + 1)]


def _background_color(grid: np.ndarray, bar_color: int) -> int:
    colors, counts = np.unique(grid, return_counts=True)
    best_color = None
    best_count = -1
    for color, count in zip(colors.tolist(), counts.tolist()):
        if color == bar_color:
            continue
        if count > best_count:
            best_color = color
            best_count = count
    if best_color is None:
        raise AssertionError("expected a background color distinct from the bar")
    return int(best_color)


def _crop_mask(mask: np.ndarray) -> np.ndarray:
    rows = np.any(mask, axis=1)
    cols = np.any(mask, axis=0)
    return mask[rows][:, cols]


def _components(grid: np.ndarray, bg_color: int, bar_color: int) -> list[tuple[int, np.ndarray]]:
    h, w = grid.shape
    visited = np.zeros((h, w), dtype=bool)
    out: list[tuple[int, np.ndarray]] = []

    for r in range(h):
        for c in range(w):
            color = int(grid[r, c])
            if color in (bg_color, bar_color) or visited[r, c]:
                continue
            q = deque([(r, c)])
            visited[r, c] = True
            coords: list[tuple[int, int]] = []
            while q:
                rr, cc = q.popleft()
                coords.append((rr, cc))
                for nr, nc in _neighbors4(rr, cc):
                    if 0 <= nr < h and 0 <= nc < w and not visited[nr, nc] and int(grid[nr, nc]) == color:
                        visited[nr, nc] = True
                        q.append((nr, nc))
            out.append((color, np.array(coords, dtype=int)))
    return out


def _sheared_coords(coords: np.ndarray, bottom_row: int, width: int) -> list[tuple[int, int]]:
    shifted: list[tuple[int, int]] = []
    for r, c in coords.tolist():
        new_c = int(c) - (bottom_row - int(r))
        if 0 <= new_c < width:
            shifted.append((int(r), new_c))
    return shifted


def main(grid: np.ndarray) -> np.ndarray:
    """
    Shear each colored object left by its own height-above-bottom and resolve overlaps by distance to the bottom edge bar.

    Input structure:
    - The background is black, and the bottom row is a full-width gray edge bar.
    - Above the bar sit 3-5 single-color 4-connected objects in distinct colors.
    - Each object's bottom row is different, so its distance to the gray bar is unique.

    Rule:
    - Keep the gray bottom bar unchanged.
    - For each foreground object, let its own bottommost row be the local shear anchor.
    - Move every object pixel at (r, c) to (r, c - (bottom_object - r)); pixels shifted past the left edge are discarded.
    - Draw the transformed objects from farthest-from-bar to nearest-to-bar, so nearer objects overwrite earlier ones where they overlap.

    Key insight:
    - The left shear is applied per object, but the visible layering is controlled by the original vertical distance from each object's bottom edge to the common gray bar.
    """
    g = np.asarray(grid, dtype=int)
    assert g.ndim == 2 and g.shape[0] >= 2 and g.shape[1] >= 2
    bar_row = g[-1, :]
    assert np.all(bar_row == bar_row[0]), "expected a solid bottom edge bar"
    bar_color = int(bar_row[0])
    bg_color = _background_color(g, bar_color)

    objects = _components(g[:-1, :], bg_color, bar_color)
    out = np.full_like(g, bg_color)
    out[-1, :] = bar_color

    bar_r = g.shape[0] - 1
    layered: list[tuple[int, int, list[tuple[int, int]]]] = []
    for color, coords in objects:
        bottom = int(coords[:, 0].max())
        distance = bar_r - bottom
        shifted = _sheared_coords(coords, bottom, g.shape[1])
        layered.append((distance, color, shifted))

    for _distance, color, shifted in sorted(layered, key=lambda item: item[0], reverse=True):
        for r, c in shifted:
            out[r, c] = color
    return out


def _make_shape(height: int, width: int, area: int) -> np.ndarray | None:
    for _ in range(80):
        mask = np.zeros((height, width), dtype=bool)
        col = random.randrange(width)
        mask[height - 1, col] = True
        for r in range(height - 2, -1, -1):
            options = [nc for nc in (col - 1, col, col + 1) if 0 <= nc < width]
            col = random.choice(options)
            mask[r, col] = True

        while int(mask.sum()) < area:
            frontier: list[tuple[int, int]] = []
            for r in range(height):
                for c in range(width):
                    if mask[r, c]:
                        continue
                    if any(0 <= nr < height and 0 <= nc < width and mask[nr, nc] for nr, nc in _neighbors4(r, c)):
                        frontier.append((r, c))
            if not frontier:
                break
            rr, cc = random.choice(frontier)
            mask[rr, cc] = True

        if int(mask.sum()) != area:
            continue
        mask = _crop_mask(mask)
        if mask.shape[0] != height:
            continue
        if mask.shape[1] < 2:
            continue
        return mask
    return None


def _coords_from_mask(mask: np.ndarray, top: int, left: int) -> list[tuple[int, int]]:
    rows, cols = np.where(mask)
    return [(top + int(r), left + int(c)) for r, c in zip(rows.tolist(), cols.tolist())]


def _coords_well_separated(occupied: np.ndarray, coords: list[tuple[int, int]]) -> bool:
    h, w = occupied.shape
    for r, c in coords:
        if not (0 <= r < h and 0 <= c < w):
            return False
        if occupied[r, c]:
            return False
        for nr, nc in _neighbors4(r, c):
            if 0 <= nr < h and 0 <= nc < w and occupied[nr, nc]:
                return False
    return True


def _render_input(height: int, width: int, objects: list[dict]) -> np.ndarray | None:
    grid = np.full((height, width), BG_COLOR, dtype=int)
    grid[-1, :] = BAR_COLOR
    occupied = np.zeros((height, width), dtype=bool)
    occupied[-1, :] = True

    for obj in objects:
        coords = obj["input_coords"]
        if not _coords_well_separated(occupied, coords):
            return None
        for r, c in coords:
            grid[r, c] = obj["color"]
            occupied[r, c] = True
    return grid


def _overlap_profile(objects: list[dict], width: int) -> tuple[int, int]:
    cell_counts: dict[tuple[int, int], set[int]] = {}
    for idx, obj in enumerate(objects):
        shifted = _sheared_coords(np.array(obj["input_coords"], dtype=int), obj["bottom"], width)
        obj["shifted_coords"] = shifted
        for cell in shifted:
            cell_counts.setdefault(cell, set()).add(idx)
    overlap_cells = sum(1 for owners in cell_counts.values() if len(owners) >= 2)
    overlap_pairs = len(
        {
            tuple(sorted((a, b)))
            for owners in cell_counts.values()
            if len(owners) >= 2
            for a in owners
            for b in owners
            if a < b
        }
    )
    return overlap_cells, overlap_pairs


def _attempt_generate() -> np.ndarray | None:
    height = random.randint(*GRID_HEIGHT_RANGE)
    width = random.randint(*GRID_WIDTH_RANGE)
    object_count = random.randint(*OBJECT_COUNT_RANGE)

    shared_row_max = height - 8
    if shared_row_max < 2:
        return None
    shared_row = random.randint(2, shared_row_max)

    bottom_choices = list(range(shared_row + 1, min(height - 2, shared_row + 7)))
    if len(bottom_choices) < object_count:
        return None
    bottoms = sorted(random.sample(bottom_choices, object_count))
    colors = random.sample(OBJECT_COLORS, object_count)

    objects: list[dict] = []
    for bottom, color in zip(bottoms, colors):
        min_height = bottom - shared_row + 1
        max_height = min(OBJECT_HEIGHT_MAX, bottom + 1)
        if min_height > max_height:
            return None

        shape = None
        for _ in range(80):
            shape_h = random.randint(min_height, max_height)
            shape_w = random.randint(*OBJECT_WIDTH_RANGE)
            min_area = max(OBJECT_AREA_RANGE[0], shape_h)
            max_area = min(OBJECT_AREA_RANGE[1], shape_h * shape_w)
            if min_area > max_area:
                continue
            area = random.randint(min_area, max_area)
            shape = _make_shape(shape_h, shape_w, area)
            if shape is None:
                continue
            shared_local_row = shape.shape[0] - 1 - (bottom - shared_row)
            if shape[shared_local_row].any() and int(shape[shared_local_row].sum()) <= 2:
                break
        if shape is None:
            return None

        shared_local_row = shape.shape[0] - 1 - (bottom - shared_row)
        shared_cols = np.flatnonzero(shape[shared_local_row])
        ref_col = int(random.choice(shared_cols.tolist()))
        objects.append(
            {
                "color": color,
                "bottom": bottom,
                "shape": shape,
                "ref_col": ref_col,
                "ref_dr": bottom - shared_row,
            }
        )

    t_lo = 0
    t_hi = width - 1
    for obj in objects:
        shape = obj["shape"]
        ref_col = obj["ref_col"]
        ref_dr = obj["ref_dr"]
        cols = np.where(shape)[1]
        lo = ref_col - ref_dr - int(cols.min())
        hi = (width - 1) - int(cols.max()) + ref_col - ref_dr
        t_lo = max(t_lo, lo)
        t_hi = min(t_hi, hi)
    if t_lo > t_hi:
        return None

    clip_mode = random.random() < CLIP_PROBABILITY
    inp = None
    for _ in range(PLACEMENT_RETRIES):
        if clip_mode:
            base = random.randint(t_lo, min(t_hi, t_lo + 2))
        else:
            base = random.randint(t_lo, t_hi)

        for obj in objects:
            target = random.randint(max(t_lo, base - 1), min(t_hi, base + 2))
            shape = obj["shape"]
            left = target - obj["ref_col"] + obj["ref_dr"]
            top = obj["bottom"] - shape.shape[0] + 1
            obj["input_coords"] = _coords_from_mask(shape, top, left)

        candidate = _render_input(height, width, objects)
        if candidate is None:
            continue

        overlap_cells, overlap_pairs = _overlap_profile(objects, width)
        if overlap_cells < 1 or overlap_pairs < 1 or overlap_cells > 8:
            continue
        if any(len(obj["shifted_coords"]) < 3 for obj in objects):
            continue
        if clip_mode and not any(len(obj["shifted_coords"]) < len(obj["input_coords"]) for obj in objects):
            continue
        inp = candidate
        break
    if inp is None:
        return None

    out = main(inp)
    if np.array_equal(out, inp):
        return None

    occluded = False
    for obj in objects:
        visible = int(np.count_nonzero(out == obj["color"]))
        if visible < 2:
            return None
        if visible < len(obj["shifted_coords"]):
            occluded = True
    if not occluded:
        return None

    return inp


def generate_input() -> np.ndarray:
    for _ in range(RETRY_LIMIT):
        grid = _attempt_generate()
        if grid is not None:
            return grid
    raise RuntimeError("failed to generate a valid input for puzzle 2")
