from __future__ import annotations

import random
from collections import deque

import numpy as np


BLACK = 0
BLUE = 1
RED = 2

BBox = tuple[int, int, int, int]
Coord = tuple[int, int]


def _neighbors4(r: int, c: int) -> tuple[Coord, Coord, Coord, Coord]:
    return ((r - 1, c), (r + 1, c), (r, c - 1), (r, c + 1))


def _components_mask(mask: np.ndarray) -> list[np.ndarray]:
    mask = np.asarray(mask, dtype=bool)
    h, w = mask.shape
    seen = np.zeros((h, w), dtype=bool)
    comps: list[np.ndarray] = []
    for r in range(h):
        for c in range(w):
            if not mask[r, c] or seen[r, c]:
                continue
            q = deque([(r, c)])
            seen[r, c] = True
            cells: list[Coord] = []
            while q:
                rr, cc = q.popleft()
                cells.append((rr, cc))
                for nr, nc in _neighbors4(rr, cc):
                    if 0 <= nr < h and 0 <= nc < w and mask[nr, nc] and not seen[nr, nc]:
                        seen[nr, nc] = True
                        q.append((nr, nc))
            comps.append(np.array(cells, dtype=int))
    return comps


def _bbox_coords(coords: np.ndarray) -> BBox:
    r0, c0 = coords.min(axis=0)
    r1, c1 = coords.max(axis=0)
    return int(r0), int(c0), int(r1), int(c1)


def _count_holes(mask: np.ndarray) -> int:
    mask = np.asarray(mask, dtype=bool)
    padded_bg = ~np.pad(mask, 1, constant_values=False)
    seen = np.zeros_like(padded_bg, dtype=bool)
    q = deque([(0, 0)])
    seen[0, 0] = True
    while q:
        r, c = q.popleft()
        for nr, nc in _neighbors4(r, c):
            if 0 <= nr < padded_bg.shape[0] and 0 <= nc < padded_bg.shape[1]:
                if padded_bg[nr, nc] and not seen[nr, nc]:
                    seen[nr, nc] = True
                    q.append((nr, nc))
    holes = padded_bg & ~seen
    return len(_components_mask(holes))


def _mirror_mask(mask: np.ndarray, hole_count: int) -> np.ndarray:
    if hole_count % 2 == 1:
        return np.flipud(mask)
    return np.fliplr(mask)


def _recolor_only(mask: np.ndarray, hole_count: int) -> np.ndarray:
    h, w = mask.shape
    out = np.zeros((h, w), dtype=int)
    if hole_count % 2 == 1:
        split = h // 2
        out[:split][mask[:split]] = BLUE
        out[split:][mask[split:]] = RED
    else:
        split = w // 2
        out[:, :split][mask[:, :split]] = BLUE
        out[:, split:][mask[:, split:]] = RED
    return out


def _transform_patch(mask: np.ndarray, hole_count: int) -> np.ndarray:
    mask = np.asarray(mask, dtype=bool)
    h, w = mask.shape
    assert h % 2 == 0 and w % 2 == 0
    mirrored = _mirror_mask(mask, hole_count)
    out = np.zeros((h, w), dtype=int)
    if hole_count % 2 == 1:
        split = h // 2
        out[:split][mirrored[:split]] = BLUE
        out[split:][mirrored[split:]] = RED
    else:
        split = w // 2
        out[:, :split][mirrored[:, :split]] = BLUE
        out[:, split:][mirrored[:, split:]] = RED
    return out


def _sorted_blue_components(grid: np.ndarray) -> list[tuple[BBox, np.ndarray]]:
    comps = _components_mask(grid == BLUE)
    objects: list[tuple[BBox, np.ndarray]] = []
    for coords in comps:
        box = _bbox_coords(coords)
        r0, c0, r1, c1 = box
        patch = np.zeros((r1 - r0 + 1, c1 - c0 + 1), dtype=bool)
        patch[coords[:, 0] - r0, coords[:, 1] - c0] = True
        objects.append((box, patch))
    objects.sort(key=lambda item: (item[0][0], item[0][1]))
    return objects


def main(grid: np.ndarray) -> np.ndarray:
    """Count each blue object's enclosed holes, choose the mirror axis from parity, then recolor the mirrored lower/right half red.

    Input structure:
    - A black background contains 2 to 4 separated blue 4-connected objects.
    - Each object has an even-height and even-width bounding box, and some objects contain enclosed black holes.
    - Holes are black 4-connected regions fully surrounded by blue inside one object's bbox.

    Rule:
    - Process each blue object independently in bbox row-major order.
    - Count the object's enclosed holes by flood-filling bbox background from outside the object; remaining black components are holes.
    - If the hole count is odd, mirror the blue mask vertically inside its bbox and recolor the mirrored lower half red.
    - If the hole count is even, mirror the blue mask horizontally inside its bbox and recolor the mirrored right half red.
    - Background and hole pixels remain black.

    Key insight:
    - The visible red half is not chosen directly from shape position; it is coupled to a hidden topological feature, the parity of the object's enclosed-hole count, which first determines the mirror axis.
    """
    g = np.asarray(grid, dtype=int)
    assert g.ndim == 2

    out = np.zeros_like(g)
    for (r0, c0, r1, c1), patch in _sorted_blue_components(g):
        hole_count = _count_holes(patch)
        transformed = _transform_patch(patch, hole_count)
        region = out[r0 : r1 + 1, c0 : c1 + 1]
        assert not np.any((region != BLACK) & (transformed != BLACK))
        region[transformed != BLACK] = transformed[transformed != BLACK]
    return out


RECT_DIMS = ((4, 4), (4, 6), (6, 4))
MAX_GEN_ATTEMPTS = 400


def _enumerate_closed_hole_candidates(target_holes: int) -> list[np.ndarray]:
    candidates: list[np.ndarray] = []
    for h, w in RECT_DIMS:
        interior_h = h - 2
        interior_w = w - 2
        total = interior_h * interior_w
        for bits in range(1 << total):
            mask = np.ones((h, w), dtype=bool)
            interior = np.ones(total, dtype=bool)
            for i in range(total):
                if (bits >> i) & 1:
                    interior[i] = False
            mask[1:-1, 1:-1] = interior.reshape(interior_h, interior_w)
            if not (10 <= int(mask.sum()) <= 25):
                continue
            if len(_components_mask(mask)) != 1:
                continue
            if _count_holes(mask) != target_holes:
                continue
            if target_holes % 2 == 1:
                if np.array_equal(mask, np.flipud(mask)):
                    continue
            else:
                if np.array_equal(mask, np.fliplr(mask)):
                    continue
            transformed = _transform_patch(mask, target_holes)
            if np.array_equal(transformed, _recolor_only(mask, target_holes)):
                continue
            if not np.any(transformed == BLUE) or not np.any(transformed == RED):
                continue
            candidates.append(mask)
    return candidates


def _make_open_notch_object() -> np.ndarray | None:
    h, w = random.choice(RECT_DIMS)
    mask = np.ones((h, w), dtype=bool)
    side = random.choice(("left", "right"))
    depth = random.randint(1, min(2, w - 2))
    notch_h = random.randint(1, h - 2)
    start_r = random.randint(1, h - notch_h - 1)
    if side == "left":
        mask[start_r : start_r + notch_h, :depth] = False
    else:
        mask[start_r : start_r + notch_h, w - depth :] = False
    if not (10 <= int(mask.sum()) <= 25):
        return None
    if len(_components_mask(mask)) != 1:
        return None
    if _count_holes(mask) != 0:
        return None
    if np.array_equal(mask, np.fliplr(mask)):
        return None
    transformed = _transform_patch(mask, 0)
    if np.array_equal(transformed, _recolor_only(mask, 0)):
        return None
    if not np.any(transformed == BLUE) or not np.any(transformed == RED):
        return None
    return mask


ODD_ONE_CANDIDATES = _enumerate_closed_hole_candidates(1)
EVEN_TWO_CANDIDATES = _enumerate_closed_hole_candidates(2)
ODD_THREE_CANDIDATES = _enumerate_closed_hole_candidates(3)


def _sample_object_mask(hole_count: int) -> np.ndarray:
    if hole_count == 0:
        for _ in range(100):
            mask = _make_open_notch_object()
            if mask is not None:
                return mask
        raise RuntimeError("failed to build a zero-hole object")
    if hole_count == 1:
        return random.choice(ODD_ONE_CANDIDATES).copy()
    if hole_count == 2:
        return random.choice(EVEN_TWO_CANDIDATES).copy()
    if hole_count == 3:
        return random.choice(ODD_THREE_CANDIDATES).copy()
    raise ValueError(f"unsupported hole count: {hole_count}")


def _candidate_locations(blocked: np.ndarray, shape: tuple[int, int]) -> list[Coord]:
    h, w = shape
    H, W = blocked.shape
    spots: list[Coord] = []
    for r in range(1, H - h):
        for c in range(1, W - w):
            if not blocked[r - 1 : r + h + 1, c - 1 : c + w + 1].any():
                spots.append((r, c))
    return spots


def _paint_object(grid: np.ndarray, blocked: np.ndarray, mask: np.ndarray, top_left: Coord) -> None:
    r0, c0 = top_left
    h, w = mask.shape
    grid[r0 : r0 + h, c0 : c0 + w][mask] = BLUE
    blocked[r0 - 1 : r0 + h + 1, c0 - 1 : c0 + w + 1] = True


def _validate_generated_grid(grid: np.ndarray) -> None:
    objects = _sorted_blue_components(grid)
    assert 2 <= len(objects) <= 4
    hole_counts: list[int] = []
    out = main(grid)
    assert out.shape == grid.shape
    for (r0, c0, r1, c1), patch in objects:
        h, w = patch.shape
        assert h % 2 == 0 and w % 2 == 0
        assert 10 <= int(patch.sum()) <= 25
        hole_count = _count_holes(patch)
        assert 0 <= hole_count <= 3
        hole_counts.append(hole_count)

        transformed = _transform_patch(patch, hole_count)
        assert np.array_equal(out[r0 : r1 + 1, c0 : c1 + 1], transformed)
        assert not np.array_equal(transformed, _recolor_only(patch, hole_count))
        if hole_count % 2 == 1:
            assert not np.array_equal(patch, np.flipud(patch))
        else:
            assert not np.array_equal(patch, np.fliplr(patch))
        assert np.any(transformed == BLUE) and np.any(transformed == RED)

    assert len(set(hole_counts)) == len(hole_counts)
    assert any(count % 2 == 1 for count in hole_counts)
    assert any(count % 2 == 0 for count in hole_counts)


def generate_input() -> np.ndarray:
    for _ in range(MAX_GEN_ATTEMPTS):
        height = random.randint(18, 24)
        width = random.randint(18, 24)
        grid = np.zeros((height, width), dtype=int)
        blocked = np.zeros((height, width), dtype=bool)

        object_count = random.randint(2, 4)
        hole_counts = random.sample((0, 1, 2, 3), k=object_count)
        if not any(count % 2 == 1 for count in hole_counts):
            continue
        if not any(count % 2 == 0 for count in hole_counts):
            continue

        masks = [(count, _sample_object_mask(count)) for count in hole_counts]
        masks.sort(key=lambda item: max(item[1].shape), reverse=True)

        ok = True
        for _, mask in masks:
            spots = _candidate_locations(blocked, mask.shape)
            if not spots:
                ok = False
                break
            _paint_object(grid, blocked, mask, random.choice(spots))

        if not ok:
            continue

        try:
            _validate_generated_grid(grid)
        except AssertionError:
            continue
        return grid
    raise RuntimeError("failed to generate a valid puzzle input")
