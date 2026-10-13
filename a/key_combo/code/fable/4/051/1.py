from __future__ import annotations

import random
from collections import Counter

import numpy as np


BLACK = 0
YELLOW = 4
NON_BLACK_NON_YELLOW = [1, 2, 3, 5, 6, 7, 8, 9]

GRID_HEIGHT_RANGE = (18, 27)
GRID_WIDTH_RANGE = (18, 27)
COUNT_OPTIONS = [(1, 2), (2, 1), (2, 2)]
VERTICAL_HEIGHT_RANGE = (3, 8)
VERTICAL_WIDTH_OPTIONS = (3, 5, 7, 9)
HORIZONTAL_HEIGHT_OPTIONS = (3, 5, 7, 9)
HORIZONTAL_WIDTH_RANGE = (3, 8)
SINGLETON_COUNT_RANGE = (2, 7)
GENERATION_RETRIES = 400


def _get_bg(grid: np.ndarray) -> int:
    return int(Counter(map(int, np.asarray(grid).ravel())).most_common(1)[0][0])


def _component_mask(coords: np.ndarray, bbox: tuple[int, int, int, int]) -> np.ndarray:
    r0, c0, r1, c1 = bbox
    mask = np.zeros((r1 - r0 + 1, c1 - c0 + 1), dtype=bool)
    mask[coords[:, 0] - r0, coords[:, 1] - c0] = True
    return mask


def _find_components(grid: np.ndarray, bg: int) -> list[dict]:
    g = np.asarray(grid, dtype=int)
    h, w = g.shape
    seen = np.zeros((h, w), dtype=bool)
    objects: list[dict] = []

    for r in range(h):
        for c in range(w):
            if seen[r, c] or int(g[r, c]) == bg:
                continue
            color = int(g[r, c])
            stack = [(r, c)]
            seen[r, c] = True
            coords = []
            while stack:
                rr, cc = stack.pop()
                coords.append((rr, cc))
                for nr, nc in ((rr - 1, cc), (rr + 1, cc), (rr, cc - 1), (rr, cc + 1)):
                    if 0 <= nr < h and 0 <= nc < w and not seen[nr, nc] and int(g[nr, nc]) == color:
                        seen[nr, nc] = True
                        stack.append((nr, nc))
            arr = np.array(coords, dtype=int)
            r0, c0 = arr.min(axis=0)
            r1, c1 = arr.max(axis=0)
            objects.append(
                {
                    "color": color,
                    "coords": arr,
                    "bbox": (int(r0), int(c0), int(r1), int(c1)),
                    "area": int(arr.shape[0]),
                }
            )
    return objects


def _classify_symmetry(mask: np.ndarray) -> tuple[str, int]:
    vertical = mask.shape[1] % 2 == 1 and bool(np.array_equal(mask, mask[:, ::-1]))
    horizontal = mask.shape[0] % 2 == 1 and bool(np.array_equal(mask, mask[::-1, :]))
    assert vertical ^ horizontal, "each non-singleton object must have exactly one axial symmetry"
    if vertical:
        return "vertical", mask.shape[1] // 2
    return "horizontal", mask.shape[0] // 2


def main(grid: np.ndarray) -> np.ndarray:
    """
    Extend each object's unique reflection axis across the canvas, erase singleton noise, and mark uncovered crossings yellow.

    Input structure:
    - A black background contains several 4-connected foreground objects in distinct colors.
    - Every non-singleton object has exactly one axial reflection symmetry: vertical or horizontal.
    - Extra isolated single pixels of arbitrary colors act as noise.

    Rule:
    - Remove every singleton noise pixel.
    - For each vertically symmetric object, draw a full-height line of the same color on its symmetry-axis column.
    - For each horizontally symmetric object, draw a full-width line of the same color on its symmetry-axis row.
    - Draw the lines underneath the original objects, then recolor every uncovered vertical-horizontal line crossing yellow.

    Key insight:
    - Object symmetry is only a switch that chooses whether to emit a row line or a column line, while the original object layer occludes both the lines and any potential yellow intersections.
    """
    g = np.asarray(grid, dtype=int)
    assert g.ndim == 2, "expected a 2D grid"
    bg = int(_get_bg(g))
    assert bg == BLACK, "expected black background"

    objects = _find_components(g, bg=bg)
    out = np.full_like(g, bg)
    covered = np.zeros_like(g, dtype=bool)
    vertical_lines: list[tuple[int, int]] = []
    horizontal_lines: list[tuple[int, int]] = []

    for obj in objects:
        if int(obj["area"]) == 1:
            continue
        mask = _component_mask(obj["coords"], obj["bbox"])
        kind, axis_local = _classify_symmetry(mask)
        r0, c0, _r1, _c1 = obj["bbox"]
        if kind == "vertical":
            vertical_lines.append((c0 + axis_local, int(obj["color"])))
        else:
            horizontal_lines.append((r0 + axis_local, int(obj["color"])))
        covered[obj["coords"][:, 0], obj["coords"][:, 1]] = True

    vertical_axes = [col for col, _color in vertical_lines]
    horizontal_axes = [row for row, _color in horizontal_lines]
    assert len(vertical_axes) == len(set(vertical_axes)), "vertical line columns must be unique"
    assert len(horizontal_axes) == len(set(horizontal_axes)), "horizontal line rows must be unique"

    for col, color in vertical_lines:
        out[:, col] = color
    for row, color in horizontal_lines:
        out[row, :] = color

    for row, _color_h in horizontal_lines:
        for col, _color_v in vertical_lines:
            if not covered[row, col]:
                out[row, col] = YELLOW

    for obj in objects:
        if int(obj["area"]) > 1:
            out[obj["coords"][:, 0], obj["coords"][:, 1]] = int(obj["color"])

    return out


def _vertical_mask() -> np.ndarray:
    for _ in range(200):
        h = random.randint(*VERTICAL_HEIGHT_RANGE)
        w = random.choice(VERTICAL_WIDTH_OPTIONS)
        arm_max = w // 2
        arms = np.array([random.randint(0, arm_max) for _ in range(h)], dtype=int)
        if int(arms.max()) == 0:
            continue
        if np.array_equal(arms, arms[::-1]):
            continue
        center = w // 2
        mask = np.zeros((h, w), dtype=bool)
        for r, arm in enumerate(arms):
            mask[r, center - arm : center + arm + 1] = True
        if _classify_symmetry(mask)[0] == "vertical":
            return mask
    raise RuntimeError("failed to generate a vertical-symmetric mask")


def _horizontal_mask() -> np.ndarray:
    for _ in range(200):
        h = random.choice(HORIZONTAL_HEIGHT_OPTIONS)
        w = random.randint(*HORIZONTAL_WIDTH_RANGE)
        arm_max = h // 2
        arms = np.array([random.randint(0, arm_max) for _ in range(w)], dtype=int)
        if int(arms.max()) == 0:
            continue
        if np.array_equal(arms, arms[::-1]):
            continue
        center = h // 2
        mask = np.zeros((h, w), dtype=bool)
        for c, arm in enumerate(arms):
            mask[center - arm : center + arm + 1, c] = True
        if _classify_symmetry(mask)[0] == "horizontal":
            return mask
    raise RuntimeError("failed to generate a horizontal-symmetric mask")


def _bbox_includes_col(bbox: tuple[int, int, int, int], col: int) -> bool:
    return bbox[1] <= int(col) <= bbox[3]


def _bbox_includes_row(bbox: tuple[int, int, int, int], row: int) -> bool:
    return bbox[0] <= int(row) <= bbox[2]


def _candidate_positions(
    shape: tuple[int, int],
    mask: np.ndarray,
    kind: str,
    blocked: np.ndarray,
    used_vertical_axes: set[int],
    used_horizontal_axes: set[int],
    predicate=None,
) -> list[dict]:
    grid_h, grid_w = shape
    h, w = mask.shape
    local = np.argwhere(mask)
    row_start = 1
    row_stop = grid_h - h
    col_start = 1
    col_stop = grid_w - w
    candidates: list[dict] = []

    for r0 in range(row_start, row_stop + 1):
        for c0 in range(col_start, col_stop + 1):
            r1 = r0 + h - 1
            c1 = c0 + w - 1
            if blocked[r0 : r1 + 1, c0 : c1 + 1].any():
                continue
            axis = c0 + w // 2 if kind == "vertical" else r0 + h // 2
            if kind == "vertical" and axis in used_vertical_axes:
                continue
            if kind == "horizontal" and axis in used_horizontal_axes:
                continue
            info = {
                "r0": r0,
                "c0": c0,
                "bbox": (r0, c0, r1, c1),
                "axis": axis,
                "coords": local + np.array([r0, c0]),
            }
            if predicate is None or predicate(info):
                candidates.append(info)
    return candidates


def _place_object(
    grid: np.ndarray,
    blocked: np.ndarray,
    mask: np.ndarray,
    kind: str,
    color: int,
    used_vertical_axes: set[int],
    used_horizontal_axes: set[int],
    predicate=None,
) -> dict | None:
    candidates = _candidate_positions(
        grid.shape, mask, kind, blocked, used_vertical_axes, used_horizontal_axes, predicate=predicate
    )
    if not candidates:
        return None
    chosen = random.choice(candidates)
    coords = chosen["coords"]
    grid[coords[:, 0], coords[:, 1]] = color

    r0, c0, r1, c1 = chosen["bbox"]
    rr0 = max(0, r0 - 1)
    cc0 = max(0, c0 - 1)
    rr1 = min(grid.shape[0] - 1, r1 + 1)
    cc1 = min(grid.shape[1] - 1, c1 + 1)
    blocked[rr0 : rr1 + 1, cc0 : cc1 + 1] = True

    placed = {
        "kind": kind,
        "color": color,
        "bbox": chosen["bbox"],
        "axis": int(chosen["axis"]),
        "coords": coords,
        "mask": mask,
    }
    if kind == "vertical":
        used_vertical_axes.add(int(chosen["axis"]))
    else:
        used_horizontal_axes.add(int(chosen["axis"]))
    return placed


def _random_object(mask: np.ndarray, kind: str, color: int) -> dict:
    return {"mask": mask, "kind": kind, "color": color}


def _place_noise(grid: np.ndarray, blocked: np.ndarray, count: int) -> bool:
    colors = NON_BLACK_NON_YELLOW
    for _ in range(count):
        choices = np.argwhere(~blocked)
        if len(choices) == 0:
            return False
        r, c = random.choice(choices.tolist())
        grid[r, c] = random.choice(colors)
        rr0 = max(0, r - 1)
        cc0 = max(0, c - 1)
        rr1 = min(grid.shape[0] - 1, r + 1)
        cc1 = min(grid.shape[1] - 1, c + 1)
        blocked[rr0 : rr1 + 1, cc0 : cc1 + 1] = True
    return True


def _count_crossing_types(grid: np.ndarray) -> tuple[int, int]:
    bg = BLACK
    objects = [obj for obj in _find_components(grid, bg=bg) if int(obj["area"]) > 1]
    covered = np.zeros_like(grid, dtype=bool)
    vertical_axes = []
    horizontal_axes = []
    for obj in objects:
        covered[obj["coords"][:, 0], obj["coords"][:, 1]] = True
        kind, axis_local = _classify_symmetry(_component_mask(obj["coords"], obj["bbox"]))
        r0, c0, _r1, _c1 = obj["bbox"]
        if kind == "vertical":
            vertical_axes.append(c0 + axis_local)
        else:
            horizontal_axes.append(r0 + axis_local)

    yellow_like = 0
    occluded = 0
    for row in horizontal_axes:
        for col in vertical_axes:
            if covered[row, col]:
                occluded += 1
            else:
                yellow_like += 1
    return yellow_like, occluded


def _generate_layout() -> np.ndarray | None:
    grid_h = random.randint(*GRID_HEIGHT_RANGE)
    grid_w = random.randint(*GRID_WIDTH_RANGE)
    grid = np.zeros((grid_h, grid_w), dtype=int)
    blocked = np.zeros((grid_h, grid_w), dtype=bool)
    used_vertical_axes: set[int] = set()
    used_horizontal_axes: set[int] = set()

    n_vertical, n_horizontal = random.choice(COUNT_OPTIONS)
    total = n_vertical + n_horizontal
    colors = random.sample(NON_BLACK_NON_YELLOW, total)

    vertical_specs = [_random_object(_vertical_mask(), "vertical", colors[i]) for i in range(n_vertical)]
    horizontal_specs = [
        _random_object(_horizontal_mask(), "horizontal", colors[n_vertical + i]) for i in range(n_horizontal)
    ]

    placed: list[dict] = []

    v0 = _place_object(
        grid,
        blocked,
        vertical_specs[0]["mask"],
        "vertical",
        vertical_specs[0]["color"],
        used_vertical_axes,
        used_horizontal_axes,
    )
    if v0 is None:
        return None
    placed.append(v0)

    first_mode = random.choice(["occluded", "background"])
    h0 = _place_object(
        grid,
        blocked,
        horizontal_specs[0]["mask"],
        "horizontal",
        horizontal_specs[0]["color"],
        used_vertical_axes,
        used_horizontal_axes,
        predicate=lambda info: (
            not _bbox_includes_row(v0["bbox"], info["axis"])
            and (_bbox_includes_col(info["bbox"], v0["axis"]) if first_mode == "occluded" else not _bbox_includes_col(info["bbox"], v0["axis"]))
        ),
    )
    if h0 is None:
        return None
    placed.append(h0)

    if n_horizontal >= 2:
        second_mode = "background" if first_mode == "occluded" else "occluded"
        h1 = _place_object(
            grid,
            blocked,
            horizontal_specs[1]["mask"],
            "horizontal",
            horizontal_specs[1]["color"],
            used_vertical_axes,
            used_horizontal_axes,
            predicate=lambda info: (
                not _bbox_includes_row(v0["bbox"], info["axis"])
                and (
                    _bbox_includes_col(info["bbox"], v0["axis"])
                    if second_mode == "occluded"
                    else not _bbox_includes_col(info["bbox"], v0["axis"])
                )
            ),
        )
        if h1 is None:
            return None
        placed.append(h1)
        remaining_vertical = vertical_specs[1:]
        remaining_horizontal = horizontal_specs[2:]
    else:
        second_mode = "background" if first_mode == "occluded" else "occluded"
        v1 = _place_object(
            grid,
            blocked,
            vertical_specs[1]["mask"],
            "vertical",
            vertical_specs[1]["color"],
            used_vertical_axes,
            used_horizontal_axes,
            predicate=lambda info: (
                not _bbox_includes_col(h0["bbox"], info["axis"])
                and (
                    _bbox_includes_row(info["bbox"], h0["axis"])
                    if second_mode == "occluded"
                    else not _bbox_includes_row(info["bbox"], h0["axis"])
                )
            ),
        )
        if v1 is None:
            return None
        placed.append(v1)
        remaining_vertical = vertical_specs[2:]
        remaining_horizontal = horizontal_specs[1:]

    remaining = remaining_vertical + remaining_horizontal
    random.shuffle(remaining)
    for spec in remaining:
        obj = _place_object(
            grid,
            blocked,
            spec["mask"],
            spec["kind"],
            spec["color"],
            used_vertical_axes,
            used_horizontal_axes,
        )
        if obj is None:
            return None
        placed.append(obj)

    if not _place_noise(grid, blocked.copy(), random.randint(*SINGLETON_COUNT_RANGE)):
        return None

    yellow_crossings, occluded_crossings = _count_crossing_types(grid)
    if yellow_crossings == 0 or occluded_crossings == 0:
        return None

    try:
        out = main(grid)
    except AssertionError:
        return None
    if out.shape != grid.shape:
        return None
    if np.array_equal(out, grid):
        return None
    return grid


def generate_input() -> np.ndarray:
    for _ in range(GENERATION_RETRIES):
        grid = _generate_layout()
        if grid is not None:
            return grid
    raise RuntimeError("failed to generate a valid input")
