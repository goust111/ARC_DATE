from __future__ import annotations

import random
from collections import deque

import numpy as np


BG = 0
OUTLINE = 5
FILL_COLORS = (1, 2, 3)

MAX_GEN_ATTEMPTS = 400
NEIGHBORS4 = [(-1, 0), (1, 0), (0, -1), (0, 1)]
NEIGHBORS8 = [
    (-1, -1), (-1, 0), (-1, 1),
    (0, -1),           (0, 1),
    (1, -1),  (1, 0),  (1, 1),
]


def _neighbors(connectivity: int) -> list[tuple[int, int]]:
    return NEIGHBORS4 if int(connectivity) == 4 else NEIGHBORS8


def _components(mask: np.ndarray, connectivity: int) -> list[np.ndarray]:
    h, w = mask.shape
    seen = np.zeros((h, w), dtype=bool)
    comps: list[np.ndarray] = []
    steps = _neighbors(connectivity)
    for r in range(h):
        for c in range(w):
            if not mask[r, c] or seen[r, c]:
                continue
            q: deque[tuple[int, int]] = deque([(r, c)])
            seen[r, c] = True
            cells: list[tuple[int, int]] = []
            while q:
                rr, cc = q.popleft()
                cells.append((rr, cc))
                for dr, dc in steps:
                    nr, nc = rr + dr, cc + dc
                    if 0 <= nr < h and 0 <= nc < w and mask[nr, nc] and not seen[nr, nc]:
                        seen[nr, nc] = True
                        q.append((nr, nc))
            comps.append(np.array(cells, dtype=int))
    return comps


def _bbox_from_coords(coords: np.ndarray) -> tuple[int, int, int, int]:
    r0, c0 = coords.min(axis=0)
    r1, c1 = coords.max(axis=0)
    return int(r0), int(c0), int(r1), int(c1)


def _crop_bbox(grid: np.ndarray, box: tuple[int, int, int, int]) -> np.ndarray:
    r0, c0, r1, c1 = box
    return grid[r0 : r1 + 1, c0 : c1 + 1].copy()


def _tight_crop_nonzero(grid: np.ndarray) -> np.ndarray:
    coords = np.argwhere(grid != BG)
    assert coords.size > 0
    return _crop_bbox(grid, _bbox_from_coords(coords))


def _enclosed_holes(patch: np.ndarray) -> list[np.ndarray]:
    holes = []
    for comp in _components(patch == BG, connectivity=4):
        r0, c0, r1, c1 = _bbox_from_coords(comp)
        if r0 == 0 or c0 == 0 or r1 == patch.shape[0] - 1 or c1 == patch.shape[1] - 1:
            continue
        holes.append(comp)
    return holes


def _hole_area(patch: np.ndarray) -> int:
    return int(sum(len(comp) for comp in _enclosed_holes(patch)))


def _internal_hole_mask(patch: np.ndarray) -> np.ndarray:
    mask = np.zeros(patch.shape, dtype=bool)
    for comp in _enclosed_holes(patch):
        mask[comp[:, 0], comp[:, 1]] = True
    return mask


def _fill_target_from_transpose(src_patch: np.ndarray, dst_patch: np.ndarray) -> np.ndarray:
    transposed = src_patch.T.copy()
    out = dst_patch.copy()
    fill_mask = (out == BG) & np.isin(transposed, FILL_COLORS)
    out[fill_mask] = transposed[fill_mask]
    return out


def _find_objects(grid: np.ndarray) -> list[np.ndarray]:
    comps = _components(grid != BG, connectivity=8)
    assert len(comps) == 2, f"expected 2 objects, found {len(comps)}"
    objects = []
    for comp in comps:
        objects.append(_crop_bbox(grid, _bbox_from_coords(comp)))
    return objects


def _random_connected_shape(area: int) -> np.ndarray:
    h = random.randint(2, 3)
    w = random.randint(2, 3)
    cells = [(r, c) for r in range(h) for c in range(w)]
    start = random.choice(cells)
    chosen = {start}
    frontier = {n for n in _adjacent_cells(start, h, w)}
    while len(chosen) < area:
        if not frontier:
            return _random_connected_shape(area)
        cell = random.choice(tuple(frontier))
        frontier.remove(cell)
        chosen.add(cell)
        for nb in _adjacent_cells(cell, h, w):
            if nb not in chosen:
                frontier.add(nb)
    mask = np.zeros((h, w), dtype=bool)
    for r, c in chosen:
        mask[r, c] = True
    coords = np.argwhere(mask)
    r0, c0, r1, c1 = _bbox_from_coords(coords)
    return mask[r0 : r1 + 1, c0 : c1 + 1]


def _adjacent_cells(cell: tuple[int, int], h: int, w: int) -> list[tuple[int, int]]:
    r, c = cell
    out = []
    for dr, dc in NEIGHBORS4:
        nr, nc = r + dr, c + dc
        if 0 <= nr < h and 0 <= nc < w:
            out.append((nr, nc))
    return out


def _notch_cells(size: int) -> np.ndarray:
    side = random.choice(["top", "bottom", "left", "right"])
    span = random.randint(1, min(3, size - 3))
    depth = random.randint(1, min(3, size - 4))
    start = random.randint(1, size - span - 1)
    mask = np.zeros((size, size), dtype=bool)
    if side == "top":
        mask[0:depth, start : start + span] = True
    elif side == "bottom":
        mask[size - depth : size, start : start + span] = True
    elif side == "left":
        mask[start : start + span, 0:depth] = True
    else:
        mask[start : start + span, size - depth : size] = True
    return mask


def _edge_covered(mask: np.ndarray) -> bool:
    return bool(mask[0].any() and mask[-1].any() and mask[:, 0].any() and mask[:, -1].any())


def _place_holes(size: int, notches: np.ndarray, hole_count: int) -> list[tuple[np.ndarray, int, int, int]]:
    placed: list[tuple[np.ndarray, int, int, int]] = []
    carved = notches.copy()
    for color in random.sample(FILL_COLORS, hole_count):
        shape = _random_connected_shape(random.randint(2, 4))
        hh, ww = shape.shape
        positions = [
            (r0, c0)
            for r0 in range(2, size - hh - 1)
            for c0 in range(2, size - ww - 1)
        ]
        random.shuffle(positions)
        placed_this = False
        for r0, c0 in positions:
            local = np.zeros((size, size), dtype=bool)
            local[r0 : r0 + hh, c0 : c0 + ww] = shape
            rr0 = max(0, r0 - 1)
            cc0 = max(0, c0 - 1)
            rr1 = min(size, r0 + hh + 1)
            cc1 = min(size, c0 + ww + 1)
            expanded = np.zeros((size, size), dtype=bool)
            expanded[rr0:rr1, cc0:cc1] = True
            if bool((local & carved).any()):
                continue
            if bool((expanded & carved).any()):
                continue
            carved |= local
            placed.append((shape, r0, c0, int(color)))
            placed_this = True
            break
        if not placed_this:
            raise ValueError("failed to place hole")
    return placed


def _build_patches() -> tuple[np.ndarray, np.ndarray, np.ndarray]:
    size = random.randint(8, 11)
    hole_count = random.randint(2, 3)

    notches = np.zeros((size, size), dtype=bool)
    notch_count = random.randint(2, 3)
    for _ in range(notch_count):
        notches |= _notch_cells(size)

    holes = _place_holes(size, notches, hole_count)
    solid = np.ones((size, size), dtype=bool)
    solid[notches] = False
    for shape, r0, c0, _color in holes:
        hh, ww = shape.shape
        solid[r0 : r0 + hh, c0 : c0 + ww][shape] = False

    if not _edge_covered(solid):
        raise ValueError("solid lost square bbox")
    if len(_components(solid, connectivity=8)) != 1:
        raise ValueError("solid disconnected")
    if len(_enclosed_holes(np.where(solid, OUTLINE, BG))) != hole_count:
        raise ValueError("unexpected hole count")

    src = np.where(solid, OUTLINE, BG).astype(int)
    for shape, r0, c0, color in holes:
        hh, ww = shape.shape
        patch = src[r0 : r0 + hh, c0 : c0 + ww]
        patch[shape] = int(color)

    dst = np.where(solid.T, OUTLINE, BG).astype(int)
    expected = _fill_target_from_transpose(src, dst)
    return src, dst, expected


def _variant_arrays(src: np.ndarray) -> dict[str, np.ndarray]:
    return {
        "identity": src.copy(),
        "rot90": np.rot90(src, 1),
        "rot180": np.rot90(src, 2),
        "rot270": np.rot90(src, 3),
        "flipud": np.flipud(src),
        "fliplr": np.fliplr(src),
        "antidiag": np.fliplr(np.flipud(src)).T,
        "transpose": src.T.copy(),
    }


def _reject_degenerate(src: np.ndarray, dst: np.ndarray, expected: np.ndarray) -> bool:
    variants = _variant_arrays(src)
    if np.array_equal(src, variants["transpose"]):
        return True
    hole_mask = _internal_hole_mask(dst)
    if not hole_mask.any():
        return True
    if not np.all(np.isin(src.T[hole_mask], FILL_COLORS)):
        return True
    for name, arr in variants.items():
        if name == "transpose":
            continue
        if np.array_equal(arr == OUTLINE, dst == OUTLINE):
            return True
    for name, arr in variants.items():
        if name == "transpose":
            continue
        alt = dst.copy()
        fill_mask = (alt == BG) & np.isin(arr, FILL_COLORS)
        alt[fill_mask] = arr[fill_mask]
        if np.array_equal(alt, expected):
            return True
    return False


def _place_on_canvas(src: np.ndarray, dst: np.ndarray) -> np.ndarray:
    size = src.shape[0]
    gap = random.randint(2, 5)
    h = random.randint(size + 2, min(24, size + 9))
    min_w = 2 * size + gap + 2
    w = random.randint(min_w, min(30, min_w + 6))
    canvas = np.zeros((h, w), dtype=int)

    src_r = random.randint(1, h - size - 1)
    dst_r = random.randint(1, h - size - 1)
    src_c_max = w - 2 * size - gap - 1
    src_c = random.randint(1, src_c_max)
    dst_c = random.randint(src_c + size + gap, w - size - 1)

    canvas[src_r : src_r + size, src_c : src_c + size] = src
    canvas[dst_r : dst_r + size, dst_c : dst_c + size] = dst
    return canvas


def _constructed_expected(grid: np.ndarray) -> np.ndarray:
    objects = _find_objects(grid)
    areas = [_hole_area(obj) for obj in objects]
    target = objects[int(np.argmax(areas))]
    source = objects[1 - int(np.argmax(areas))]
    return _tight_crop_nonzero(_fill_target_from_transpose(source, target))


def main(grid: np.ndarray) -> np.ndarray:
    """
    Fill the larger-hole transposed target object with colors copied from the source after main-diagonal transposition.

    Input structure:
    - Black (0) is background. Exactly two 8-connected foreground objects appear on the canvas.
    - Both objects use grey (5) for the shared outline/support shape and occupy square tight bounding boxes.
    - The source object contains 2-3 small colored subsprites inside enclosed cavities; the target object is the source support shape reflected across the main diagonal, but its corresponding cavities are empty black holes.

    Rule:
    - Find the two foreground objects and measure each object's enclosed black-hole area inside its tight bounding box.
    - The object with the larger enclosed hole area is the target; the other is the source.
    - Transpose the entire source bounding box across its main diagonal, align that transposed patch to the target bounding box, and recolor each black target-hole cell with the color found at the same aligned location in the transposed source.
    - Keep grey outline cells unchanged and return the tight crop of the filled target object.

    Key insight:
    - The source-to-target geometric relation is neither a rotation nor a flip: only a main-diagonal transpose places the source subsprites onto the target holes correctly.
    """
    g = np.asarray(grid, dtype=int)
    assert g.ndim == 2, "expected a 2D grid"

    objects = _find_objects(g)
    areas = [_hole_area(obj) for obj in objects]
    target_idx = int(np.argmax(areas))
    source_idx = 1 - target_idx
    assert areas[target_idx] > areas[source_idx], "target must have more hole area than source"

    source = objects[source_idx]
    target = objects[target_idx].copy()
    assert source.shape == target.shape, "source/target bounding boxes must stay square and aligned by transpose"
    assert source.shape[0] == source.shape[1], "expected square object bounding boxes"

    filled = _fill_target_from_transpose(source, target)
    hole_mask = _internal_hole_mask(target)
    assert np.all(filled[hole_mask] != BG), "every target hole cell must receive a color"
    return _tight_crop_nonzero(filled)


def generate_input() -> np.ndarray:
    for _ in range(MAX_GEN_ATTEMPTS):
        try:
            src, dst, expected = _build_patches()
        except ValueError:
            continue
        if _reject_degenerate(src, dst, expected):
            continue
        inp = _place_on_canvas(src, dst)
        try:
            objects = _find_objects(inp)
        except AssertionError:
            continue
        areas = sorted(_hole_area(obj) for obj in objects)
        if not (areas[0] < areas[1]):
            continue
        if not np.array_equal(_constructed_expected(inp), expected):
            continue
        try:
            out = main(inp)
        except AssertionError:
            continue
        if not np.array_equal(out, expected):
            continue
        return inp
    raise RuntimeError("failed to generate a valid diagonal-transpose hole-filling input")
