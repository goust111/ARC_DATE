from __future__ import annotations

from collections import Counter, deque
from typing import Iterable

import numpy as np


DIRS4 = [(-1, 0), (1, 0), (0, -1), (0, 1)]
GEN_RETRIES = 300
RNG = np.random.default_rng()


def _get_bg(grid: np.ndarray) -> int:
    values, counts = np.unique(grid, return_counts=True)
    return int(values[np.argmax(counts)])


def _trim_mask(mask: np.ndarray) -> np.ndarray:
    pts = np.argwhere(mask)
    if len(pts) == 0:
        return np.zeros((0, 0), dtype=bool)
    r0, c0 = pts.min(axis=0)
    r1, c1 = pts.max(axis=0)
    return np.asarray(mask[r0 : r1 + 1, c0 : c1 + 1], dtype=bool)


def _neighbors4(r: int, c: int, h: int, w: int) -> Iterable[tuple[int, int]]:
    for dr, dc in DIRS4:
        nr, nc = r + dr, c + dc
        if 0 <= nr < h and 0 <= nc < w:
            yield nr, nc


def _components(mask: np.ndarray) -> list[list[tuple[int, int]]]:
    h, w = mask.shape
    seen = np.zeros((h, w), dtype=bool)
    result: list[list[tuple[int, int]]] = []
    for r in range(h):
        for c in range(w):
            if not mask[r, c] or seen[r, c]:
                continue
            seen[r, c] = True
            q = deque([(r, c)])
            pts: list[tuple[int, int]] = []
            while q:
                rr, cc = q.popleft()
                pts.append((int(rr), int(cc)))
                for nr, nc in _neighbors4(rr, cc, h, w):
                    if mask[nr, nc] and not seen[nr, nc]:
                        seen[nr, nc] = True
                        q.append((nr, nc))
            result.append(pts)
    return result


def _extract_objects(grid: np.ndarray, bg: int) -> list[dict]:
    objects: list[dict] = []
    for color in sorted(set(map(int, grid.flatten())) - {int(bg)}):
        color_mask = grid == int(color)
        for pts in _components(color_mask):
            arr = np.asarray(pts, dtype=int)
            r0, c0 = arr.min(axis=0)
            r1, c1 = arr.max(axis=0)
            mask = np.zeros((int(r1 - r0 + 1), int(c1 - c0 + 1)), dtype=bool)
            for rr, cc in pts:
                mask[int(rr - r0), int(cc - c0)] = True
            objects.append(
                {
                    "color": int(color),
                    "bbox": (int(r0), int(c0), int(r1), int(c1)),
                    "mask": mask,
                }
            )
    return objects


def _classify_mask(mask: np.ndarray) -> str:
    m = _trim_mask(np.asarray(mask, dtype=bool))
    if m.size == 0:
        return "none"
    sym_h = np.array_equal(m, m[::-1, :])
    sym_v = np.array_equal(m, m[:, ::-1])
    if sym_h and sym_v:
        return "both"
    if sym_h:
        return "horizontal"
    if sym_v:
        return "vertical"
    return "none"


def _paint_mask(grid: np.ndarray, mask: np.ndarray, r0: int, c0: int, color: int) -> None:
    pts = np.argwhere(mask)
    if len(pts) == 0:
        return
    rr, cc = pts.T
    grid[r0 + rr, c0 + cc] = int(color)


def _fold_horizontal(mask: np.ndarray) -> np.ndarray:
    h, _w = mask.shape
    keep_h = h // 2
    top = mask[:keep_h, :]
    bottom = mask[h - keep_h :, :]
    return np.asarray(top | bottom[::-1, :], dtype=bool)


def _fold_vertical(mask: np.ndarray) -> np.ndarray:
    _h, w = mask.shape
    keep_w = w // 2
    left = mask[:, :keep_w]
    right = mask[:, w - keep_w :]
    return np.asarray(left | right[:, ::-1], dtype=bool)


def main(grid: np.ndarray) -> np.ndarray:
    """
    Classify each foreground object by its own reflection symmetry, then keep, fold, or erase it.

    Input structure:
    - The background is the dominant color.
    - The grid contains several disconnected monochrome 4-connected foreground objects.
    - Each object is considered inside its own tight bounding box and may be doubly symmetric, symmetric on exactly one axis, or fully asymmetric.

    Rule:
    - Detect horizontal-axis symmetry by comparing the object mask with its top-bottom reflection, and vertical-axis symmetry by comparing with its left-right reflection.
    - If an object has both symmetries, keep it unchanged.
    - If it has only horizontal-axis symmetry, fold the lower half onto the upper half, keep the merged result in the upper half of the original bbox, and clear the lower half.
    - If it has only vertical-axis symmetry, fold the right half onto the left half, keep the merged result in the left half of the original bbox, and clear the right half.
    - If it has neither symmetry, erase the whole object to background.

    Key insight:
    - The transformation is chosen independently for each object from its local symmetry type, not from its color, size, or absolute position.
    """
    g = np.asarray(grid, dtype=int)
    assert g.ndim == 2

    bg = _get_bg(g)
    out = g.copy()

    for obj in _extract_objects(g, bg):
        color = int(obj["color"])
        r0, c0, r1, c1 = obj["bbox"]
        mask = np.asarray(obj["mask"], dtype=bool)
        kind = _classify_mask(mask)

        out[r0 : r1 + 1, c0 : c1 + 1] = int(bg)

        if kind == "both":
            _paint_mask(out, mask, r0, c0, color)
        elif kind == "horizontal":
            folded = _fold_horizontal(mask)
            _paint_mask(out, folded, r0, c0, color)
        elif kind == "vertical":
            folded = _fold_vertical(mask)
            _paint_mask(out, folded, r0, c0, color)

    return out


def _is_connected(mask: np.ndarray) -> bool:
    m = np.asarray(mask, dtype=bool)
    return m.any() and len(_components(m)) == 1


def _grow_connected_mask(h: int, w: int, target_cells: int) -> np.ndarray | None:
    if h <= 0 or w <= 0 or target_cells <= 0 or target_cells > h * w:
        return None

    mask = np.zeros((h, w), dtype=bool)
    start = (int(RNG.integers(h)), int(RNG.integers(w)))
    mask[start] = True
    frontier = {cell for cell in _neighbors4(start[0], start[1], h, w)}

    while int(mask.sum()) < int(target_cells):
        if not frontier:
            return None
        cell = list(frontier)[int(RNG.integers(len(frontier)))]
        frontier.remove(cell)
        if mask[cell]:
            continue
        mask[cell] = True
        for nbr in _neighbors4(cell[0], cell[1], h, w):
            if not mask[nbr]:
                frontier.add(nbr)

    return mask


def _generate_both_axis_mask() -> np.ndarray:
    for _ in range(100):
        mode = str(RNG.choice(["rect", "frame", "plus", "diamond"]))
        if mode == "rect":
            h = int(RNG.integers(2, 8))
            w = int(RNG.integers(2, 8))
            mask = np.ones((h, w), dtype=bool)
        elif mode == "frame":
            h = int(RNG.integers(4, 8))
            w = int(RNG.integers(4, 8))
            mask = np.zeros((h, w), dtype=bool)
            mask[0, :] = True
            mask[-1, :] = True
            mask[:, 0] = True
            mask[:, -1] = True
        elif mode == "plus":
            h = int(RNG.choice([5, 7, 9]))
            w = int(RNG.choice([5, 7, 9]))
            mask = np.zeros((h, w), dtype=bool)
            row_half = 0 if h < 7 or RNG.random() < 0.65 else 1
            col_half = 0 if w < 7 or RNG.random() < 0.65 else 1
            cr = h // 2
            cc = w // 2
            mask[cr - row_half : cr + row_half + 1, :] = True
            mask[:, cc - col_half : cc + col_half + 1] = True
        else:
            k = int(RNG.integers(2, 5))
            size = 2 * k + 1
            mask = np.zeros((size, size), dtype=bool)
            for r in range(size):
                for c in range(size):
                    if abs(r - k) + abs(c - k) <= k:
                        mask[r, c] = True

        mask = _trim_mask(mask)
        if _classify_mask(mask) == "both" and _is_connected(mask):
            return mask

    return np.array(
        [
            [0, 1, 0],
            [1, 1, 1],
            [0, 1, 0],
        ],
        dtype=bool,
    )


def _generate_horizontal_only_mask() -> np.ndarray:
    fallback = np.array(
        [
            [1, 1, 1, 0],
            [0, 1, 0, 0],
            [0, 1, 0, 0],
            [1, 1, 1, 0],
        ],
        dtype=bool,
    )
    for _ in range(150):
        half_h = int(RNG.integers(2, 5))
        w = int(RNG.integers(3, 8))
        area = half_h * w
        target = int(RNG.integers(max(2, area // 3), area + 1))
        top = _grow_connected_mask(half_h, w, target)
        if top is None:
            continue
        mask = np.vstack([top, top[::-1, :]])
        mask = _trim_mask(mask)
        if _classify_mask(mask) == "horizontal" and _is_connected(mask):
            return mask
    return fallback


def _generate_vertical_only_mask() -> np.ndarray:
    mask = _generate_horizontal_only_mask().T
    if _classify_mask(mask) == "vertical" and _is_connected(mask):
        return np.asarray(mask, dtype=bool)

    return np.array(
        [
            [1, 0, 0, 1],
            [1, 1, 1, 1],
            [1, 0, 0, 1],
        ],
        dtype=bool,
    )


def _generate_asymmetric_mask() -> np.ndarray:
    fallback = np.array(
        [
            [1, 1, 0],
            [0, 1, 0],
            [0, 1, 1],
        ],
        dtype=bool,
    )
    for _ in range(150):
        h = int(RNG.integers(3, 8))
        w = int(RNG.integers(3, 8))
        area = h * w
        target = int(RNG.integers(max(3, area // 3), area))
        mask = _grow_connected_mask(h, w, target)
        if mask is None:
            continue
        mask = _trim_mask(mask)
        if _classify_mask(mask) == "none" and _is_connected(mask):
            return mask
    return fallback


def _generate_mask(kind: str) -> np.ndarray:
    if kind == "both":
        return _generate_both_axis_mask()
    if kind == "horizontal":
        return _generate_horizontal_only_mask()
    if kind == "vertical":
        return _generate_vertical_only_mask()
    return _generate_asymmetric_mask()


def _rect_overlap(a: tuple[int, int, int, int], b: tuple[int, int, int, int]) -> bool:
    ar0, ac0, ar1, ac1 = a
    br0, bc0, br1, bc1 = b
    return not (ar1 <= br0 or br1 <= ar0 or ac1 <= bc0 or bc1 <= ac0)


def _validate_generated_grid(grid: np.ndarray, expected_counts: Counter) -> bool:
    if grid.ndim != 2 or max(grid.shape) > 30:
        return False
    bg = _get_bg(grid)
    found = Counter(_classify_mask(obj["mask"]) for obj in _extract_objects(grid, bg))
    if found != expected_counts:
        return False
    out = main(grid)
    if out.shape != grid.shape:
        return False
    if np.array_equal(out, grid):
        return False
    return True


def _gen_attempt() -> np.ndarray | None:
    bg = int(RNG.integers(0, 10))
    base_kinds = ["both", "horizontal", "vertical", "none"]
    extra_count = int(RNG.integers(0, 3))
    extras = [str(RNG.choice(base_kinds)) for _ in range(extra_count)]
    kinds = base_kinds + extras
    RNG.shuffle(kinds)

    palette = [color for color in range(10) if color != bg]
    RNG.shuffle(palette)
    pool_size = int(RNG.integers(3, min(7, len(palette)) + 1))
    color_pool = palette[:pool_size]

    objects = []
    used_colors: list[int] = []
    for kind in kinds:
        mask = _generate_mask(kind)
        if used_colors and RNG.random() < 0.35:
            color = int(used_colors[int(RNG.integers(len(used_colors)))])
        else:
            color = int(color_pool[int(RNG.integers(len(color_pool)))])
        used_colors.append(color)
        objects.append({"kind": kind, "mask": mask, "color": color})

    objects.sort(key=lambda item: item["mask"].shape[0] * item["mask"].shape[1], reverse=True)

    total_bbox_area = sum(int(obj["mask"].shape[0] * obj["mask"].shape[1]) for obj in objects)
    max_h = max(int(obj["mask"].shape[0]) for obj in objects)
    max_w = max(int(obj["mask"].shape[1]) for obj in objects)

    side = int(np.ceil(np.sqrt(total_bbox_area * 3.4)))
    low_h = max(18, max_h + 2, side - 3)
    low_w = max(18, max_w + 2, side - 3)
    high_h = min(30, side + 4)
    high_w = min(30, side + 4)
    if low_h > high_h or low_w > high_w:
        return None

    h = int(RNG.integers(low_h, high_h + 1))
    w = int(RNG.integers(low_w, high_w + 1))
    grid = np.full((h, w), bg, dtype=int)

    placed_rects: list[tuple[int, int, int, int]] = []
    for obj in objects:
        mh, mw = obj["mask"].shape
        if mh > h or mw > w:
            return None

        candidates: list[tuple[int, int, tuple[int, int, int, int]]] = []
        for _ in range(250):
            r0 = int(RNG.integers(0, h - mh + 1))
            c0 = int(RNG.integers(0, w - mw + 1))
            rect = (
                max(0, r0 - 1),
                max(0, c0 - 1),
                min(h, r0 + mh + 1),
                min(w, c0 + mw + 1),
            )
            if any(_rect_overlap(rect, prev) for prev in placed_rects):
                continue
            candidates.append((r0, c0, rect))
            if len(candidates) >= 25:
                break

        if not candidates:
            return None

        r0, c0, rect = candidates[int(RNG.integers(len(candidates)))]
        _paint_mask(grid, obj["mask"], r0, c0, int(obj["color"]))
        placed_rects.append(rect)

    expected = Counter(obj["kind"] for obj in objects)
    if not _validate_generated_grid(grid, expected):
        return None
    return grid


def generate_input() -> np.ndarray:
    for _ in range(GEN_RETRIES):
        grid = _gen_attempt()
        if grid is not None:
            return grid
    raise RuntimeError("failed to generate valid symmetry-fold puzzle input")
