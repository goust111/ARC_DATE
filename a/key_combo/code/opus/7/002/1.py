from __future__ import annotations

import random
from collections import deque

import numpy as np


BLACK = 0
NEIGHBORS4 = ((-1, 0), (1, 0), (0, -1), (0, 1))
PALETTE = (2, 3, 4, 6, 7, 8, 9)
MAX_ATTEMPTS = 600
SLOT_SPAN = 5

# Stable under the one-pass denoise rule: every shape pixel has at least two
# same-color 4-neighbors before denoising.
SHAPE_LIBRARY = (
    np.array([[1, 1], [1, 1]], dtype=bool),
    np.array([[1, 1, 1], [1, 1, 1]], dtype=bool),
    np.array([[1, 1], [1, 1], [1, 1]], dtype=bool),
    np.array([[1, 1, 0], [1, 1, 1], [0, 1, 1]], dtype=bool),
    np.array([[0, 1, 1], [1, 1, 1], [1, 1, 0]], dtype=bool),
)


def _in_bounds(shape: tuple[int, int], r: int, c: int) -> bool:
    return 0 <= r < shape[0] and 0 <= c < shape[1]


def _bbox_from_coords(coords: np.ndarray) -> tuple[int, int, int, int]:
    r0, c0 = coords.min(axis=0)
    r1, c1 = coords.max(axis=0)
    return int(r0), int(c0), int(r1), int(c1)


def _same_color_degree(grid: np.ndarray, r: int, c: int) -> int:
    color = int(grid[r, c])
    total = 0
    for dr, dc in NEIGHBORS4:
        rr, cc = r + dr, c + dc
        if _in_bounds(grid.shape, rr, cc) and int(grid[rr, cc]) == color:
            total += 1
    return total


def _denoise(grid: np.ndarray) -> np.ndarray:
    out = np.asarray(grid, dtype=int).copy()
    remove: list[tuple[int, int]] = []
    for r, c in np.argwhere(out != BLACK):
        rr = int(r)
        cc = int(c)
        if _same_color_degree(out, rr, cc) < 2:
            remove.append((rr, cc))
    for r, c in remove:
        out[r, c] = BLACK
    return out


def _find_components(grid: np.ndarray) -> list[dict]:
    g = np.asarray(grid, dtype=int)
    seen = np.zeros(g.shape, dtype=bool)
    components: list[dict] = []

    for r, c in np.argwhere(g != BLACK):
        r0 = int(r)
        c0 = int(c)
        if seen[r0, c0]:
            continue
        color = int(g[r0, c0])
        q: deque[tuple[int, int]] = deque([(r0, c0)])
        seen[r0, c0] = True
        cells: list[tuple[int, int]] = []

        while q:
            rr, cc = q.popleft()
            cells.append((rr, cc))
            for dr, dc in NEIGHBORS4:
                nr, nc = rr + dr, cc + dc
                if not _in_bounds(g.shape, nr, nc) or seen[nr, nc]:
                    continue
                if int(g[nr, nc]) != color:
                    continue
                seen[nr, nc] = True
                q.append((nr, nc))

        coords = np.asarray(cells, dtype=int)
        components.append(
            {
                "color": color,
                "coords": coords,
                "area": int(coords.shape[0]),
                "bbox": _bbox_from_coords(coords),
            }
        )

    components.sort(key=lambda comp: (comp["bbox"][0], comp["bbox"][1], comp["color"]))
    return components


def _draw_frame(out: np.ndarray, bbox: tuple[int, int, int, int], color: int) -> None:
    r0, c0, r1, c1 = bbox
    out[r0, c0 : c1 + 1] = color
    out[r1, c0 : c1 + 1] = color
    out[r0 : r1 + 1, c0] = color
    out[r0 : r1 + 1, c1] = color


def _slide_and_frame(grid: np.ndarray) -> np.ndarray:
    g = np.asarray(grid, dtype=int)
    out = np.zeros_like(g)
    height = g.shape[0]

    for comp in _find_components(g):
        target = np.asarray(comp["coords"], dtype=int).copy()
        target[:, 0] += int(comp["area"])
        visible = target[(0 <= target[:, 0]) & (target[:, 0] < height)]
        if visible.size == 0:
            continue
        _draw_frame(out, _bbox_from_coords(visible), int(comp["color"]))

    return out


def _slide_only(grid: np.ndarray) -> np.ndarray:
    g = np.asarray(grid, dtype=int)
    out = np.zeros_like(g)
    height = g.shape[0]
    width = g.shape[1]

    for comp in _find_components(g):
        target = np.asarray(comp["coords"], dtype=int).copy()
        target[:, 0] += int(comp["area"])
        visible = target[
            (0 <= target[:, 0])
            & (target[:, 0] < height)
            & (0 <= target[:, 1])
            & (target[:, 1] < width)
        ]
        if visible.size == 0:
            continue
        out[visible[:, 0], visible[:, 1]] = int(comp["color"])

    return out


def _raw_pipeline_without_denoise(grid: np.ndarray) -> np.ndarray:
    return _slide_and_frame(np.asarray(grid, dtype=int))


def main(grid: np.ndarray) -> np.ndarray:
    """
    Denoise same-color specks, drop each surviving shape by its area, and keep only the fallen bbox frame.

    Input structure:
    - A black background grid contains 3-5 small 4-connected monochromatic shapes in
      the upper half, with different colors and nearby same-color noise pixels.
    - Real shapes remain 4-connected after denoising; each noise pixel has fewer than
      two same-color 4-neighbors and is not 4-connected to the real shape.
    - The lower half is otherwise empty, leaving room for area-driven downward motion.

    Rule:
    - First, remove every non-black pixel whose same-color 4-neighbor count is less
      than two, using the original input as the reference for that count.
    - For each surviving monochromatic 4-connected shape, compute its area A and move
      the entire shape downward by A rows, clipping anything below the bottom edge.
    - At the moved location, draw the 1-pixel-thick rectangle border of the moved
      shape's bbox in the same color, then clear the interior so only the hollow frame remains.

    Key insight:
    - The same-color specks are not extra objects to keep; they must be erased first,
      because only the denoised shapes determine the area-based fall and the final bbox shadow.
    """
    g = np.asarray(grid, dtype=int)
    assert g.ndim == 2
    denoised = _denoise(g)
    return _slide_and_frame(denoised)


def _shape_choices(n_objects: int) -> list[np.ndarray]:
    shapes = [
        SHAPE_LIBRARY[3].copy(),
        random.choice(SHAPE_LIBRARY[:3]).copy(),
    ]
    while len(shapes) < n_objects:
        shapes.append(random.choice(SHAPE_LIBRARY).copy())
    random.shuffle(shapes)

    areas = [int(mask.sum()) for mask in shapes]
    if len(set(areas)) < 2:
        return _shape_choices(n_objects)
    if not any(mask.shape[0] >= 3 and mask.shape[1] >= 3 for mask in shapes):
        return _shape_choices(n_objects)
    return shapes


def _shape_coords(mask: np.ndarray, top: int, left: int) -> np.ndarray:
    coords = np.argwhere(mask)
    coords = coords + np.array([int(top), int(left)])
    return coords.astype(int)


def _touches_any_8(grid: np.ndarray, r: int, c: int) -> bool:
    for dr in (-1, 0, 1):
        for dc in (-1, 0, 1):
            if dr == 0 and dc == 0:
                continue
            rr, cc = r + dr, c + dc
            if _in_bounds(grid.shape, rr, cc) and int(grid[rr, cc]) != BLACK:
                return True
    return False


def _place_noise(
    grid: np.ndarray,
    *,
    color: int,
    shape_bbox: tuple[int, int, int, int],
    slot_left: int,
    slot_right: int,
    top_band: int,
) -> list[tuple[int, int]] | None:
    r0, c0, r1, c1 = shape_bbox
    candidates: list[tuple[int, int]] = []
    row_lo = max(0, r0 - 2)
    row_hi = min(top_band - 1, r1 + 2)
    col_lo = max(slot_left, c0 - 2)
    col_hi = min(slot_right, c1 + 2)

    for r in range(row_lo, row_hi + 1):
        for c in range(col_lo, col_hi + 1):
            if int(grid[r, c]) != BLACK:
                continue
            if r0 <= r <= r1 and c0 <= c <= c1:
                continue
            if _touches_any_8(grid, r, c):
                continue
            distance = min(abs(r - sr) + abs(c - sc) for sr in range(r0, r1 + 1) for sc in range(c0, c1 + 1))
            if 2 <= distance <= 4:
                candidates.append((r, c))

    if not candidates:
        return None

    random.shuffle(candidates)
    noise_count = random.randint(1, min(3, len(candidates)))
    chosen: list[tuple[int, int]] = []
    for r, c in candidates:
        if any(abs(r - rr) <= 1 and abs(c - cc) <= 1 for rr, cc in chosen):
            continue
        chosen.append((r, c))
        if len(chosen) == noise_count:
            break

    if len(chosen) != noise_count:
        return None

    for r, c in chosen:
        grid[r, c] = color
    return chosen


def _rectangles_overlap(a: tuple[int, int, int, int], b: tuple[int, int, int, int]) -> bool:
    return not (a[2] < b[0] or b[2] < a[0] or a[3] < b[1] or b[3] < a[1])


def _build_scene() -> tuple[np.ndarray, list[dict]] | None:
    n_objects = random.randint(3, 5)
    shapes = _shape_choices(n_objects)
    colors = random.sample(PALETTE, n_objects)

    min_width = n_objects * SLOT_SPAN
    width = random.randint(min_width, min(30, min_width + 4))
    height = random.randint(16, 22)
    left_margin = random.randint(0, width - min_width)
    top_band = height // 2
    grid = np.zeros((height, width), dtype=int)

    objects: list[dict] = []
    fallen_bboxes: list[tuple[int, int, int, int]] = []

    for idx, (mask, color) in enumerate(zip(shapes, colors)):
        area = int(mask.sum())
        shape_h, shape_w = mask.shape
        slot_left = left_margin + idx * SLOT_SPAN
        slot_right = slot_left + SLOT_SPAN - 1
        max_left = slot_right - shape_w + 1
        if max_left < slot_left:
            return None

        max_top = min(top_band - shape_h, height - area - shape_h)
        if max_top < 1:
            return None
        top = random.randint(1, max_top)
        left = random.randint(slot_left, max_left)
        coords = _shape_coords(mask, top, left)
        grid[coords[:, 0], coords[:, 1]] = color
        shape_bbox = _bbox_from_coords(coords)

        fallen_bbox = (shape_bbox[0] + area, shape_bbox[1], shape_bbox[2] + area, shape_bbox[3])
        if any(_rectangles_overlap(fallen_bbox, other) for other in fallen_bboxes):
            return None
        fallen_bboxes.append(fallen_bbox)

        noise = _place_noise(
            grid,
            color=color,
            shape_bbox=shape_bbox,
            slot_left=slot_left,
            slot_right=slot_right,
            top_band=top_band,
        )
        if noise is None:
            return None

        objects.append(
            {
                "color": color,
                "mask": mask.copy(),
                "coords": coords.copy(),
                "bbox": shape_bbox,
                "area": area,
                "noise": list(noise),
                "fallen_bbox": fallen_bbox,
            }
        )

    if np.any(grid[top_band:, :] != BLACK):
        return None
    return grid, objects


def _expected_from_scene(scene: np.ndarray, objects: list[dict]) -> np.ndarray:
    out = np.zeros_like(scene)
    for obj in objects:
        _draw_frame(out, obj["fallen_bbox"], int(obj["color"]))
    return out


def _validate_scene(scene: np.ndarray, objects: list[dict]) -> bool:
    if scene.ndim != 2 or max(scene.shape) > 30:
        return False
    if not (3 <= len(objects) <= 5):
        return False
    if len({int(obj["color"]) for obj in objects}) != len(objects):
        return False
    if len({int(obj["area"]) for obj in objects}) < 2:
        return False
    if not any(obj["mask"].shape[0] >= 3 and obj["mask"].shape[1] >= 3 for obj in objects):
        return False

    denoised = _denoise(scene)
    comps = _find_components(denoised)
    if len(comps) != len(objects):
        return False

    by_color = {int(comp["color"]): comp for comp in comps}
    for obj in objects:
        color = int(obj["color"])
        if color not in by_color:
            return False
        comp = by_color[color]
        if int(comp["area"]) != int(obj["area"]):
            return False
        if comp["bbox"] != obj["bbox"]:
            return False
        if set(map(tuple, comp["coords"])) != set(map(tuple, obj["coords"])):
            return False
        for r, c in obj["coords"]:
            if _same_color_degree(scene, int(r), int(c)) < 2:
                return False
        for r, c in obj["noise"]:
            if _same_color_degree(scene, int(r), int(c)) >= 2:
                return False
            if denoised[int(r), int(c)] != BLACK:
                return False

    expected = _expected_from_scene(scene, objects)
    solved = main(scene)
    if not np.array_equal(solved, expected):
        return False
    if np.array_equal(solved, _raw_pipeline_without_denoise(scene)):
        return False
    if np.array_equal(solved, _slide_only(denoised)):
        return False
    if not any(
        obj["fallen_bbox"][2] - obj["fallen_bbox"][0] + 1 >= 3
        and obj["fallen_bbox"][3] - obj["fallen_bbox"][1] + 1 >= 3
        for obj in objects
    ):
        return False
    if any(
        _rectangles_overlap(objects[i]["fallen_bbox"], objects[j]["fallen_bbox"])
        for i in range(len(objects))
        for j in range(i + 1, len(objects))
    ):
        return False
    return True


def generate_input() -> np.ndarray:
    for _ in range(MAX_ATTEMPTS):
        built = _build_scene()
        if built is None:
            continue
        scene, objects = built
        if _validate_scene(scene, objects):
            return scene
    raise RuntimeError("failed to generate a valid input for ARC puzzle 1")
