from __future__ import annotations

import random
import sys
from pathlib import Path

import numpy as np

_ARC_UTILS_DIR = Path("/home/lixinyu/snap/new/solvers")
if str(_ARC_UTILS_DIR) not in sys.path:
    sys.path.append(str(_ARC_UTILS_DIR))

from arc_utils import bbox_h, bbox_w, crop_bbox, find_components, paste


PALETTE = (1, 2, 3, 4, 5, 6)
MAX_ATTEMPTS = 400
INPUT_SIZE_RANGE = (18, 24)


def _rows_to_patch(rows: tuple[str, ...]) -> np.ndarray:
    return np.array([[1 if ch == "#" else 0 for ch in row] for row in rows], dtype=int)


CONTAINER_PATCH = _rows_to_patch(
    (
        "...#",
        "###.",
        "#...",
        "#...",
        "#...",
    )
)

TARGET_PATCH = _rows_to_patch(
    (
        ".#.",
        "###",
        ".#.",
    )
)

WIDE_PATCHES = (
    _rows_to_patch(
        (
            "###.",
            "..##",
        )
    ),
    _rows_to_patch(
        (
            "##.##",
            ".###.",
        )
    ),
    _rows_to_patch(
        (
            ".###",
            "##..",
            ".#..",
        )
    ),
    _rows_to_patch(
        (
            "####",
            ".#..",
            ".#..",
        )
    ),
)

SQUARE_PATCHES = (
    TARGET_PATCH,
    _rows_to_patch(
        (
            "##.",
            ".##",
            "..#",
        )
    ),
    _rows_to_patch(
        (
            "#.#",
            ".#.",
            "###",
        )
    ),
)


def _bbox_contains(outer: tuple[int, int, int, int], inner: tuple[int, int, int, int]) -> bool:
    return outer[0] <= inner[0] and outer[1] <= inner[1] and inner[2] <= outer[2] and inner[3] <= outer[3]


def _strict_bbox_contains(outer: tuple[int, int, int, int], inner: tuple[int, int, int, int]) -> bool:
    return _bbox_contains(outer, inner) and not _bbox_contains(inner, outer)


def _object_patch(grid: np.ndarray, obj: dict, bg: int) -> np.ndarray:
    patch = crop_bbox(grid, obj["bbox"])
    patch[patch != int(obj["color"])] = int(bg)
    return patch


def _mirror_all(grid: np.ndarray) -> np.ndarray:
    g = np.asarray(grid, dtype=int)
    bg = 0
    out = g.copy()
    objects = find_components(g, bg=bg, connectivity=8, monochromatic=True, return_format="dict")
    for obj in objects:
        patch = _object_patch(g, obj, bg)
        r0, c0, r1, c1 = map(int, obj["bbox"])
        h = bbox_h(obj["bbox"])
        w = bbox_w(obj["bbox"])
        if w > h:
            paste(out, np.flipud(patch), r1 + 1, c0, transparent=bg, clip=True)
        elif h > w:
            paste(out, np.fliplr(patch), r0, c1 + 1, transparent=bg, clip=True)
    return out


def _contained_objects(grid: np.ndarray) -> list[dict]:
    mirrored = _mirror_all(grid)
    objects = find_components(mirrored, bg=0, connectivity=8, monochromatic=True, return_format="dict")
    victims: list[dict] = []
    for obj in objects:
        if any(
            other is not obj
            and int(other["color"]) != int(obj["color"])
            and _strict_bbox_contains(other["bbox"], obj["bbox"])
            for other in objects
        ):
            victims.append(obj)
    victims.sort(key=lambda obj: (int(obj["area"]), int(obj["bbox"][0]), int(obj["bbox"][1])))
    return victims


def main(grid: np.ndarray) -> np.ndarray:
    """
    Mirror each object along an axis chosen by its bounding-box aspect ratio, then erase foreign-color objects newly enclosed by another mirrored bbox.

    Input structure:
    - The background is black.
    - The foreground consists of several non-overlapping 8-connected monochromatic objects, each with a distinct color.
    - Object shapes are irregular and their bounding boxes may be wide, tall, or square.

    Rule:
    - For each object, inspect its original bounding box.
    - If width > height, keep the object and append a vertically flipped copy immediately below its bbox.
    - If height > width, keep the object and append a horizontally flipped copy immediately to the right of its bbox.
    - If width = height, leave the object unchanged; mirrored pixels that would leave the grid are clipped away.
    - After all objects finish mirroring, recompute monochromatic 8-connected objects; erase every object whose bbox is strictly contained in the bbox of a different-colored object, processing victims from smaller area to larger area.

    Key insight:
    - The decisive containment relation does not exist in the input; it is created only after the aspect-ratio-dependent mirror doubles some objects and enlarges their bounding boxes.
    """
    mirrored = _mirror_all(grid)
    out = mirrored.copy()
    for obj in _contained_objects(grid):
        coords = obj["coords"]
        out[coords[:, 0], coords[:, 1]] = 0
    return out


def _transform_local_coords(patch: np.ndarray) -> set[tuple[int, int]]:
    coords = {(int(r), int(c)) for r, c in np.argwhere(patch != 0)}
    h, w = patch.shape
    out = set(coords)
    if w > h:
        out |= {(2 * h - 1 - r, c) for r, c in coords}
    elif h > w:
        out |= {(r, 2 * w - 1 - c) for r, c in coords}
    return out


def _patch_coords(patch: np.ndarray, top: int, left: int) -> set[tuple[int, int]]:
    return {(int(top + r), int(left + c)) for r, c in np.argwhere(patch != 0)}


def _transformed_coords(patch: np.ndarray, top: int, left: int) -> set[tuple[int, int]]:
    return {(int(top + r), int(left + c)) for r, c in _transform_local_coords(patch)}


def _update_forbidden(cells: set[tuple[int, int]], forbidden: set[tuple[int, int]]) -> None:
    for r, c in cells:
        for dr in (-1, 0, 1):
            for dc in (-1, 0, 1):
                forbidden.add((int(r + dr), int(c + dc)))


def _place_patch(grid: np.ndarray, patch: np.ndarray, color: int, top: int, left: int) -> None:
    grid[top : top + patch.shape[0], left : left + patch.shape[1]][patch != 0] = int(color)


def _random_oriented_patch(kind: str) -> np.ndarray:
    if kind == "wide":
        patch = random.choice(WIDE_PATCHES).copy()
        if random.random() < 0.5:
            patch = np.flipud(patch)
        if random.random() < 0.5:
            patch = np.fliplr(patch)
        return patch
    if kind == "tall":
        patch = np.transpose(random.choice(WIDE_PATCHES)).copy()
        if random.random() < 0.5:
            patch = np.flipud(patch)
        if random.random() < 0.5:
            patch = np.fliplr(patch)
        return patch
    patch = random.choice(SQUARE_PATCHES).copy()
    patch = np.rot90(patch, random.randrange(4))
    if random.random() < 0.5:
        patch = np.fliplr(patch)
    return patch


def _can_place(
    patch: np.ndarray,
    top: int,
    left: int,
    shape: tuple[int, int],
    forbidden_input: set[tuple[int, int]],
    occupied_final: set[tuple[int, int]],
) -> bool:
    h, w = patch.shape
    if top < 0 or left < 0 or top + h > shape[0] or left + w > shape[1]:
        return False
    cells = _patch_coords(patch, top, left)
    if cells & forbidden_input:
        return False
    final_cells = _transformed_coords(patch, top, left)
    if any(not (0 <= r < shape[0] and 0 <= c < shape[1]) for r, c in final_cells):
        return False
    return not bool(final_cells & occupied_final)


def _analysis_ok(grid: np.ndarray) -> bool:
    objects = find_components(grid, bg=0, connectivity=8, monochromatic=True, return_format="dict")
    if not (4 <= len(objects) <= 6):
        return False
    if len({int(obj["color"]) for obj in objects}) != len(objects):
        return False
    for i, obj in enumerate(objects):
        h = bbox_h(obj["bbox"])
        w = bbox_w(obj["bbox"])
        if not (4 <= int(obj["area"]) <= 8):
            return False
        if h == w:
            if not any(hh == ww for hh, ww in [(h, w)]):
                return False
        if h * w == int(obj["area"]):
            return False
        for other in objects[i + 1 :]:
            if _bbox_contains(obj["bbox"], other["bbox"]) or _bbox_contains(other["bbox"], obj["bbox"]):
                return False
    victims = _contained_objects(grid)
    if not victims:
        return False
    out = main(grid)
    return np.any(grid != out)


def _attempt_generate() -> np.ndarray | None:
    h = random.randint(*INPUT_SIZE_RANGE)
    w = random.randint(*INPUT_SIZE_RANGE)
    n_objects = random.randint(4, 6)
    colors = random.sample(PALETTE, n_objects)

    grid = np.zeros((h, w), dtype=int)
    forbidden_input: set[tuple[int, int]] = set()
    occupied_final: set[tuple[int, int]] = set()

    container_top = random.randint(0, h - CONTAINER_PATCH.shape[0])
    container_left = random.randint(0, w - 8)
    target_top = container_top + 2
    target_left = container_left + 4

    if not _can_place(CONTAINER_PATCH, container_top, container_left, grid.shape, forbidden_input, occupied_final):
        return None
    container_cells = _patch_coords(CONTAINER_PATCH, container_top, container_left)
    _place_patch(grid, CONTAINER_PATCH, colors[0], container_top, container_left)
    _update_forbidden(container_cells, forbidden_input)
    occupied_final |= _transformed_coords(CONTAINER_PATCH, container_top, container_left)

    if not _can_place(TARGET_PATCH, target_top, target_left, grid.shape, forbidden_input, occupied_final):
        return None
    target_cells = _patch_coords(TARGET_PATCH, target_top, target_left)
    _place_patch(grid, TARGET_PATCH, colors[1], target_top, target_left)
    _update_forbidden(target_cells, forbidden_input)
    occupied_final |= _transformed_coords(TARGET_PATCH, target_top, target_left)

    required_kinds = ["wide"]
    while len(required_kinds) + 2 < n_objects:
        required_kinds.append(random.choice(["wide", "tall", "square"]))

    for color, kind in zip(colors[2:], required_kinds):
        placed = False
        for _ in range(120):
            patch = _random_oriented_patch(kind)
            top = random.randint(0, h - patch.shape[0])
            left = random.randint(0, w - patch.shape[1])
            if not _can_place(patch, top, left, grid.shape, forbidden_input, occupied_final):
                continue
            cells = _patch_coords(patch, top, left)
            _place_patch(grid, patch, color, top, left)
            _update_forbidden(cells, forbidden_input)
            occupied_final |= _transformed_coords(patch, top, left)
            placed = True
            break
        if not placed:
            return None

    if not _analysis_ok(grid):
        return None
    if not np.array_equal(main(grid), main(grid.copy())):
        return None
    return grid


def generate_input() -> np.ndarray:
    for _ in range(MAX_ATTEMPTS):
        grid = _attempt_generate()
        if grid is not None:
            return grid
    raise RuntimeError("failed to generate valid input for puzzle 041/2")
