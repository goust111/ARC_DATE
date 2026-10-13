from __future__ import annotations

from importlib.machinery import SourceFileLoader
import random

import numpy as np


_ARC = SourceFileLoader(
    "_arc_utils_shared_017_7",
    "/home/lixinyu/snap/new/solvers/arc_utils.py",
).load_module()

Color = _ARC.Color
bbox_coords = _ARC.bbox_coords
components_mask = _ARC.components_mask
get_bg = _ARC.get_bg


PALETTE = (Color.BLUE, Color.RED, Color.GREEN, Color.YELLOW)
SMALL_TEMPLATES = (
    ((0, 0),),
    ((0, 0), (0, 1)),
    ((0, 0), (1, 0)),
    ((0, 0), (0, 1), (0, 2)),
    ((0, 0), (1, 0), (2, 0)),
    ((0, 0), (0, 1), (1, 0)),
    ((0, 0), (0, 1), (1, 1)),
)
GEN_RETRY_LIMIT = 500


def _normalize(coords: set[tuple[int, int]] | list[tuple[int, int]]) -> tuple[tuple[int, int], ...]:
    rows = [r for r, _ in coords]
    cols = [c for _, c in coords]
    r0 = min(rows)
    c0 = min(cols)
    return tuple(sorted((r - r0, c - c0) for r, c in coords))


def _shape_dims(shape: tuple[tuple[int, int], ...]) -> tuple[int, int]:
    return max(r for r, _ in shape) + 1, max(c for _, c in shape) + 1


def _find_objects(grid: np.ndarray) -> list[dict]:
    g = np.asarray(grid, dtype=int)
    bg = int(get_bg(g))
    objects: list[dict] = []
    for color in sorted(int(c) for c in np.unique(g) if int(c) != bg):
        for coords in components_mask(g == color, connectivity=4):
            coords = np.asarray(coords, dtype=int)
            r0, c0, r1, c1 = bbox_coords(coords)
            objects.append(
                {
                    "color": int(color),
                    "coords": coords,
                    "bbox": (int(r0), int(c0), int(r1), int(c1)),
                    "area": int(coords.shape[0]),
                    # Store twice-the-center coordinates to keep Manhattan distances exact.
                    "center2": (int(r0 + r1), int(c0 + c1)),
                }
            )
    return objects


def _mirror_shift(obj: dict, width: int) -> tuple[int, int]:
    r0, c0, r1, c1 = obj["bbox"]
    box_w = c1 - c0 + 1
    new_c0 = width - box_w - c0
    return 0, int(new_c0 - c0)


def _distance2(a: tuple[int, int], b: tuple[int, int]) -> int:
    return abs(a[0] - b[0]) + abs(a[1] - b[1])


def _classify(objects: list[dict]) -> tuple[list[dict], list[dict]]:
    small = [obj for obj in objects if obj["area"] <= 3]
    large = [obj for obj in objects if obj["area"] >= 5]
    return small, large


def _solve(grid: np.ndarray, *, use_moved_small_centers: bool) -> np.ndarray:
    g = np.asarray(grid, dtype=int)
    assert g.ndim == 2, "expected a 2D grid"
    bg = int(get_bg(g))
    objects = _find_objects(g)
    small, large = _classify(objects)
    assert small and large, "expected both small and large objects"

    out = g.copy()
    H, W = g.shape

    small_info = []
    for obj in small:
        coords = obj["coords"]
        out[coords[:, 0], coords[:, 1]] = bg
        dr, dc = _mirror_shift(obj, W)
        moved = np.column_stack((coords[:, 0] + dr, coords[:, 1] + dc))
        assert np.all((0 <= moved[:, 0]) & (moved[:, 0] < H))
        assert np.all((0 <= moved[:, 1]) & (moved[:, 1] < W))
        if np.any(out[moved[:, 0], moved[:, 1]] != bg):
            raise AssertionError("mirrored small objects must land on background")
        out[moved[:, 0], moved[:, 1]] = obj["color"]

        moved_bbox = (
            obj["bbox"][0] + dr,
            obj["bbox"][1] + dc,
            obj["bbox"][2] + dr,
            obj["bbox"][3] + dc,
        )
        small_info.append(
            {
                "color": obj["color"],
                "center2_orig": obj["center2"],
                "center2_moved": (moved_bbox[0] + moved_bbox[2], moved_bbox[1] + moved_bbox[3]),
            }
        )

    for obj in large:
        best_color = None
        best_dist = None
        tied = False
        for info in small_info:
            center = info["center2_moved"] if use_moved_small_centers else info["center2_orig"]
            dist = _distance2(obj["center2"], center)
            if best_dist is None or dist < best_dist:
                best_dist = dist
                best_color = info["color"]
                tied = False
            elif dist == best_dist:
                tied = True
        assert not tied, "nearest small object must be unique"
        coords = obj["coords"]
        out[coords[:, 0], coords[:, 1]] = int(best_color)

    return out


def main(grid: np.ndarray) -> np.ndarray:
    """
    Move each small object to its horizontal mirror position, then recolor each large object from its nearest pre-move small object.

    Input structure:
    - The background is black.
    - The foreground contains 6-8 disconnected 4-connected single-color objects in blue, red, green, or yellow.
    - Each object has area 1-3 or at least 5, so the scene splits cleanly into small and large objects.

    Rule:
    - Detect every non-background monochromatic 4-connected object and record its area, color, bounding box, and bounding-box center.
    - Small objects (area <= 3) are erased from their original cells and translated to the horizontal mirror position of their bounding box; rows stay fixed and the internal pixel arrangement is not flipped.
    - Large objects (area >= 5) stay in place, but each one is recolored to the color of the unique nearest small object measured by Manhattan distance between bounding-box centers recorded before any small-object movement.

    Key insight:
    - The nearest-small lookup is locked to the original small-object positions, so solving in the wrong order changes which color a large object should borrow.
    """
    return _solve(grid, use_moved_small_centers=False)


def _random_big_shape() -> tuple[tuple[int, int], ...]:
    target_area = random.randint(5, 8)
    for _ in range(100):
        cells = {(0, 0)}
        while len(cells) < target_area:
            frontier = set()
            for r, c in cells:
                frontier.update(((r - 1, c), (r + 1, c), (r, c - 1), (r, c + 1)))
            frontier.difference_update(cells)
            nr, nc = random.choice(tuple(frontier))
            cells.add((nr, nc))
        shape = _normalize(cells)
        h, w = _shape_dims(shape)
        if h <= 4 and w <= 4:
            return shape
    # Fallback keeps generation deterministic even if random growth is unlucky.
    return random.choice(
        (
            ((0, 0), (0, 1), (0, 2), (1, 1), (2, 1)),
            ((0, 0), (1, 0), (2, 0), (2, 1), (2, 2)),
            ((0, 0), (0, 1), (1, 1), (1, 2), (2, 2), (2, 3)),
            ((0, 0), (0, 1), (1, 0), (1, 1), (2, 0), (2, 1)),
        )
    )


def _placeable(coords: list[tuple[int, int]], reserved: np.ndarray) -> bool:
    H, W = reserved.shape
    for r, c in coords:
        if not (0 <= r < H and 0 <= c < W):
            return False
        if reserved[r, c]:
            return False
    return True


def _reserve_cells(reserved: np.ndarray, coords: list[tuple[int, int]]) -> None:
    H, W = reserved.shape
    for r, c in coords:
        for dr in (-1, 0, 1):
            for dc in (-1, 0, 1):
                rr = r + dr
                cc = c + dc
                if 0 <= rr < H and 0 <= cc < W:
                    reserved[rr, cc] = True


def _translated(shape: tuple[tuple[int, int], ...], r0: int, c0: int) -> list[tuple[int, int]]:
    return [(r0 + r, c0 + c) for r, c in shape]


def _paint(grid: np.ndarray, coords: list[tuple[int, int]], color: int) -> None:
    for r, c in coords:
        grid[r, c] = int(color)


def _validate_grid(grid: np.ndarray) -> bool:
    objects = _find_objects(grid)
    if len(objects) < 6 or len(objects) > 8:
        return False
    small, large = _classify(objects)
    if len(small) not in (3, 4) or len(large) not in (3, 4):
        return False
    if any(obj["area"] == 4 for obj in objects):
        return False
    if len({obj["color"] for obj in small}) < 2:
        return False

    width = grid.shape[1]
    for obj in small:
        _, dc = _mirror_shift(obj, width)
        if dc == 0:
            return False

    assigned_colors = []
    changed_big = False
    for obj in large:
        pairs = sorted(
            (_distance2(obj["center2"], small_obj["center2"]), small_obj["color"])
            for small_obj in small
        )
        if len(pairs) < 2 or pairs[0][0] == pairs[1][0]:
            return False
        assigned_colors.append(int(pairs[0][1]))
        if int(pairs[0][1]) != int(obj["color"]):
            changed_big = True

    if not changed_big:
        return False
    if len(set(assigned_colors)) < 2:
        return False

    try:
        correct = main(grid)
        wrong = _solve(grid, use_moved_small_centers=True)
    except Exception:
        return False

    if np.array_equal(correct, grid):
        return False
    if np.array_equal(correct, wrong):
        return False
    return True


def _gen_attempt() -> np.ndarray | None:
    height = random.randint(14, 18)
    width = random.randint(15, 20)
    grid = np.full((height, width), Color.BLACK, dtype=int)
    reserved = np.zeros((height, width), dtype=bool)

    small_count, large_count = random.choice(((3, 3), (3, 4), (4, 3), (4, 4)))

    # Place large objects first; their geometry is the stable frame of reference.
    for _ in range(large_count):
        color = random.choice(PALETTE)
        shape = _random_big_shape()
        h, w = _shape_dims(shape)
        placed = False
        candidates = [
            (r0, c0)
            for r0 in range(height - h + 1)
            for c0 in range(width - w + 1)
        ]
        random.shuffle(candidates)
        for r0, c0 in candidates:
            coords = _translated(shape, r0, c0)
            if not _placeable(coords, reserved):
                continue
            _paint(grid, coords, color)
            _reserve_cells(reserved, coords)
            placed = True
            break
        if not placed:
            return None

    # Each small object must fit both at its original location and at its mirrored destination.
    for _ in range(small_count):
        color = random.choice(PALETTE)
        shape = random.choice(SMALL_TEMPLATES)
        h, w = _shape_dims(shape)
        placed = False
        candidates = [
            (r0, c0)
            for r0 in range(height - h + 1)
            for c0 in range(width - w + 1)
        ]
        random.shuffle(candidates)
        for r0, c0 in candidates:
            mirror_c0 = width - w - c0
            if mirror_c0 == c0:
                continue
            coords = _translated(shape, r0, c0)
            mirror_coords = _translated(shape, r0, mirror_c0)
            if not _placeable(coords, reserved):
                continue
            if not _placeable(mirror_coords, reserved):
                continue
            _paint(grid, coords, color)
            _reserve_cells(reserved, coords)
            _reserve_cells(reserved, mirror_coords)
            placed = True
            break
        if not placed:
            return None

    return grid if _validate_grid(grid) else None


def generate_input() -> np.ndarray:
    for _ in range(GEN_RETRY_LIMIT):
        grid = _gen_attempt()
        if grid is not None:
            return grid
    raise RuntimeError("failed to generate a valid input")
