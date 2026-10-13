from __future__ import annotations

import importlib.util
import random
from pathlib import Path

import numpy as np

try:
    from arc_utils import find_components, get_bg
except Exception:  # pragma: no cover - import path fallback for direct SourceFileLoader use
    _LOCAL_UTILS = None
    for _parent in Path(__file__).resolve().parents:
        _candidate = _parent / "new" / "solvers" / "arc_utils.py"
        if _candidate.exists():
            _LOCAL_UTILS = _candidate
            break
    if _LOCAL_UTILS is None:
        raise ImportError("failed to locate arc_utils module")
    _UTILS_SPEC = importlib.util.spec_from_file_location("_arc_utils", str(_LOCAL_UTILS))
    if _UTILS_SPEC is None or _UTILS_SPEC.loader is None:
        raise ImportError("failed to locate arc_utils module")
    _ARC_UTILS = importlib.util.module_from_spec(_UTILS_SPEC)
    _UTILS_SPEC.loader.exec_module(_ARC_UTILS)
    find_components = _ARC_UTILS.find_components
    get_bg = _ARC_UTILS.get_bg


_DIRECTIONS4 = ((-1, 0), (1, 0), (0, -1), (0, 1))
_BG_CHOICES = (0, 1, 2, 3, 4, 5, 6, 7, 8, 9)


def _occupancy(shape: tuple[int, int], objects: list[list[np.ndarray]]) -> np.ndarray:
    occ = np.full(shape, -1, dtype=int)
    for idx, (coords, _values) in enumerate(objects):
        assert np.all(occ[coords[:, 0], coords[:, 1]] == -1), "objects must be non-overlapping"
        occ[coords[:, 0], coords[:, 1]] = idx
    return occ


def _objects_that_can_shift_right(
    objects: list[list[np.ndarray]],
    occupancy: np.ndarray,
) -> set[int]:
    h, w = occupancy.shape
    movable = set(range(len(objects)))
    changed = True

    while changed:
        changed = False
        for idx in list(movable):
            coords, _values = objects[idx]
            next_cols = coords[:, 1] + 1
            if np.any(next_cols >= w):
                movable.remove(idx)
                changed = True
                continue

            labels = occupancy[coords[:, 0], next_cols]
            blockers = {int(label) for label in labels if int(label) not in (-1, idx)}
            if any(blocker not in movable for blocker in blockers):
                movable.remove(idx)
                changed = True

    return movable


def _simulate_slide(g: np.ndarray) -> tuple[list[list[np.ndarray]], list[int], list[tuple[int, int]], int]:
    bg = int(get_bg(g))
    h, w = g.shape
    components = find_components(
        g,
        bg=bg,
        connectivity=4,
        monochromatic=True,
        return_format="dict",
    )
    if not components:
        return [], [], [], bg

    objects: list[list[np.ndarray]] = []
    anchors: list[tuple[int, int]] = []
    for comp in components:
        coords = np.asarray(comp["coords"], dtype=int)
        assert coords.ndim == 2 and coords.shape[1] == 2
        values = g[coords[:, 0], coords[:, 1]].astype(int)
        objects.append([coords, values])
        anchors.append((int(coords[:, 0].min()), int(coords[:, 1].min())))

    displacement = [0 for _ in objects]

    for _step in range(w + 1):
        occ = _occupancy((h, w), objects)
        movable = _objects_that_can_shift_right(objects, occ)
        if not movable:
            break
        for idx in movable:
            objects[idx][0][:, 1] += 1
            displacement[idx] += 1
    else:
        raise AssertionError("rightward slide did not settle within grid width")

    return objects, displacement, anchors, bg


def main(grid: np.ndarray) -> np.ndarray:
    """
    Generate a one-column coded bar recording each sprite's actual rightward displacement.

    Input structure:
    - The grid has a dominant background color.
    - There are several disjoint monochrome, 4-connected sprites with distinct colors.
    - Sprites may overlap in row ranges, so a right sprite can stop a left sprite.

    Rule:
    - Simultaneously move all sprites one column right whenever their next cells are inside the grid and not blocked by a sprite that is already immobile.
    - Repeat until no sprite can move.
    - Record each sprite's final displacement d in top-to-bottom order by source top row, then left column; write d cells of that sprite color in one output column.

    Key insight:
    - The output is a run-length history of synchronized motion, so each sprite must be simulated with full blocking interactions.
    """
    g = np.asarray(grid, dtype=int)
    assert g.ndim == 2 and g.size > 0, "expected a non-empty 2D grid"

    objects, displacement, anchors, bg = _simulate_slide(g)
    if not objects:
        return np.full((g.shape[0], 1), bg, dtype=int)

    order = sorted(range(len(objects)), key=lambda i: (anchors[i][0], anchors[i][1]))
    out = np.full((g.shape[0], 1), bg, dtype=int)
    row = 0
    for idx in order:
        d = int(displacement[idx])
        if d <= 0:
            continue
        color = int(objects[idx][1][0]) if objects[idx][1].size else bg
        assert row + d <= out.shape[0], "encoded displacement exceeds output height"
        out[row : row + d, 0] = color
        row += d

    return out


def _normalize_coords(coords: np.ndarray) -> np.ndarray:
    out = coords.copy()
    out[:, 0] -= int(out[:, 0].min())
    out[:, 1] -= int(out[:, 1].min())
    return out


def _random_connected_shape(area: int) -> np.ndarray:
    area = max(1, min(int(area), 8))
    cells = [(0, 0)]
    cell_set = {(0, 0)}
    frontier = {(1, 0), (-1, 0), (0, 1), (0, -1)}
    while len(cells) < area:
        r, c = random.choice(tuple(frontier))
        frontier.remove((r, c))
        if (r, c) in cell_set:
            continue
        cells.append((r, c))
        cell_set.add((r, c))
        for dr, dc in _DIRECTIONS4:
            nxt = (r + dr, c + dc)
            if nxt not in cell_set:
                frontier.add(nxt)

    return _normalize_coords(np.array(cells, dtype=int))


def _randomized_shape(area: int) -> np.ndarray:
    coords = _random_connected_shape(area)
    if random.random() < 0.5:
        coords = coords[:, [1, 0]]
    if random.random() < 0.5:
        coords[:, 0] = -coords[:, 0]
    if random.random() < 0.5:
        coords[:, 1] = -coords[:, 1]
    return _normalize_coords(coords)


def _can_place(
    occupied: np.ndarray, shape_coords: np.ndarray, top: int, left: int, h: int, w: int
) -> bool:
    rows = shape_coords[:, 0] + int(top)
    cols = shape_coords[:, 1] + int(left)
    if rows.min() < 0 or cols.min() < 0 or rows.max() >= occupied.shape[0] or cols.max() >= occupied.shape[1]:
        return False
    return not occupied[rows, cols].any()


def _rand_inclusive(lo: int, hi: int) -> int:
    lo = int(lo)
    hi = int(hi)
    if lo > hi:
        return hi
    return random.randint(lo, hi)


def _place_shape(
    grid: np.ndarray,
    occupied: np.ndarray,
    shape_coords: np.ndarray,
    top: int,
    left: int,
    color: int,
) -> None:
    rows = shape_coords[:, 0] + int(top)
    cols = shape_coords[:, 1] + int(left)
    grid[rows, cols] = int(color)
    occupied[rows, cols] = True


def _try_generate_candidate() -> np.ndarray | None:
    h = random.randint(8, 30)
    w = random.randint(8, 30)
    bg = random.choice(_BG_CHOICES)
    sprite_count = random.randint(1, min(6, len(_BG_CHOICES) - 1))
    palette = [c for c in _BG_CHOICES if c != bg]
    colors = random.sample(palette, sprite_count)

    mode = random.choices(("chain", "overlap", "scattered"), weights=(0.35, 0.35, 0.30), k=1)[0]
    grid = np.full((h, w), int(bg), dtype=int)
    occupied = np.zeros_like(grid, dtype=bool)

    placed_meta: list[tuple[int, int, int, int]] = []

    for idx, color in enumerate(colors):
        area = random.randint(1, 8)
        shape = _randomized_shape(area)
        shape_h = int(shape[:, 0].max()) + 1
        shape_w = int(shape[:, 1].max()) + 1
        if shape_h > h or shape_w > w:
            return None

        placed = False
        for _ in range(180):
            if mode == "chain":
                if idx == 0:
                    left = _rand_inclusive(max(0, w - shape_w - 2), w - shape_w)
                    top = random.randint(0, h - shape_h)
                else:
                    ref_top, ref_left, ref_h, ref_w = random.choice(placed_meta)
                    top_low = max(0, ref_top - 2)
                    top_high = min(h - shape_h, ref_top + ref_h + 1)
                    if top_low > top_high:
                        top = random.randint(0, h - shape_h)
                    else:
                        top = random.randint(top_low, top_high)

                    left_low = max(0, ref_left - random.randint(1, 6))
                    left_high = min(w - shape_w, ref_left + random.randint(0, 2))
                    if left_low > left_high:
                        left_low, left_high = 0, w - shape_w
                    left = _rand_inclusive(left_low, left_high)
            elif mode == "overlap":
                if placed_meta:
                    ref_top, ref_left, ref_h, ref_w = random.choice(placed_meta)
                    top_low = max(0, ref_top - 1)
                    top_high = min(h - shape_h, ref_top + ref_h - 1)
                    if top_low > top_high:
                        top = random.randint(0, h - shape_h)
                    else:
                        top = random.randint(top_low, top_high)
                    left = _rand_inclusive(max(0, w // 2), w - shape_w)
                else:
                    top = random.randint(0, h - shape_h)
                    left = _rand_inclusive(max(0, w // 2), w - shape_w)
            else:
                top = random.randint(0, h - shape_h)
                left = _rand_inclusive(0, w - shape_w)

            if not _can_place(occupied, shape, top, left, h, w):
                continue
            _place_shape(grid, occupied, shape, top, left, color)
            placed_meta.append((top, left, shape_h, shape_w))
            placed = True
            break

        if not placed:
            return None

    try:
        _, displacement, _anchors, _bg = _simulate_slide(grid)
    except AssertionError:
        return None

    if not displacement:
        return None
    total_move = int(sum(displacement))
    if total_move == 0:
        return None
    if total_move > h:
        return None

    if random.random() < 0.4 and all(value > 0 for value in displacement):
        return None
    if random.random() < 0.3 and all(value == 0 for value in displacement):
        return None

    return grid


def generate_input() -> np.ndarray:
    for _ in range(1200):
        grid = _try_generate_candidate()
        if grid is None:
            continue

        out = main(grid)
        if out.shape != (grid.shape[0], 1):
            continue
        if out.shape[0] > 30:
            continue
        if out.shape[1] > 30:
            continue
        return grid

    raise RuntimeError("failed to generate a valid input for task 3")
