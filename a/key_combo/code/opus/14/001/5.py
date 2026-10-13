from __future__ import annotations

from collections import deque
import random

import numpy as np


def _neighbors4(r: int, c: int):
    yield r - 1, c
    yield r + 1, c
    yield r, c - 1
    yield r, c + 1


def _get_bg(grid: np.ndarray) -> int:
    values, counts = np.unique(grid, return_counts=True)
    return int(values[np.argmax(counts)])


def _find_components(grid: np.ndarray, bg: int) -> list[dict]:
    h, w = grid.shape
    seen = np.zeros((h, w), dtype=bool)
    comps: list[dict] = []

    for r in range(h):
        for c in range(w):
            if seen[r, c] or int(grid[r, c]) == bg:
                continue
            color = int(grid[r, c])
            q = deque([(r, c)])
            seen[r, c] = True
            coords = []
            while q:
                rr, cc = q.popleft()
                coords.append((rr, cc))
                for nr, nc in _neighbors4(rr, cc):
                    if 0 <= nr < h and 0 <= nc < w and not seen[nr, nc] and int(grid[nr, nc]) == color:
                        seen[nr, nc] = True
                        q.append((nr, nc))
            arr = np.array(coords, dtype=int)
            r0 = int(arr[:, 0].min())
            c0 = int(arr[:, 1].min())
            r1 = int(arr[:, 0].max())
            c1 = int(arr[:, 1].max())
            comps.append(
                {
                    "color": color,
                    "coords": arr,
                    "bbox": (r0, c0, r1, c1),
                    "area": int(arr.shape[0]),
                }
            )
    return comps


def _mirror_half_component(comp: dict, grid_shape: tuple[int, int]) -> tuple[np.ndarray, tuple[int, int]]:
    rows = comp["coords"][:, 0]
    cols = comp["coords"][:, 1]
    r0 = int(rows.min())
    r1 = int(rows.max())
    c0 = int(cols.min())
    axis_left = int(cols.max())
    full_c1 = 2 * axis_left + 1 - c0
    if full_c1 >= grid_shape[1]:
        raise ValueError("mirrored object exceeds grid width")

    mask = np.zeros((r1 - r0 + 1, full_c1 - c0 + 1), dtype=bool)
    for r, c in comp["coords"]:
        rr = int(r) - r0
        cc = int(c) - c0
        mc = 2 * axis_left + 1 - int(c) - c0
        mask[rr, cc] = True
        mask[rr, mc] = True
    return mask, (r0, c0)


def _background_regions(mask: np.ndarray) -> tuple[np.ndarray, np.ndarray]:
    h, w = mask.shape
    padded = np.zeros((h + 2, w + 2), dtype=bool)
    padded[1:-1, 1:-1] = mask

    exterior = np.zeros_like(padded, dtype=bool)
    q = deque()
    for r in range(h + 2):
        for c in (0, w + 1):
            if not padded[r, c] and not exterior[r, c]:
                exterior[r, c] = True
                q.append((r, c))
    for c in range(w + 2):
        for r in (0, h + 1):
            if not padded[r, c] and not exterior[r, c]:
                exterior[r, c] = True
                q.append((r, c))

    while q:
        r, c = q.popleft()
        for nr, nc in _neighbors4(r, c):
            if 0 <= nr < h + 2 and 0 <= nc < w + 2 and not padded[nr, nc] and not exterior[nr, nc]:
                exterior[nr, nc] = True
                q.append((nr, nc))

    holes = (~padded) & (~exterior)
    return holes[1:-1, 1:-1], exterior


def _outer_boundary(mask: np.ndarray) -> np.ndarray:
    h, w = mask.shape
    _, exterior = _background_regions(mask)
    boundary = np.zeros_like(mask, dtype=bool)
    for r in range(h):
        for c in range(w):
            if not mask[r, c]:
                continue
            pr = r + 1
            pc = c + 1
            for nr, nc in ((pr - 1, pc), (pr + 1, pc), (pr, pc - 1), (pr, pc + 1)):
                if exterior[nr, nc]:
                    boundary[r, c] = True
                    break
    return boundary


def _paint_mask(dst: np.ndarray, mask: np.ndarray, top_left: tuple[int, int], color: int) -> None:
    r0, c0 = top_left
    rows, cols = np.where(mask)
    dst[rows + r0, cols + c0] = int(color)


def main(grid: np.ndarray) -> np.ndarray:
    """Mirror-complete each half-shape, classify the completed topology, then invert its morphology.

    Input structure:
    - The background is black or another dominant canvas color.
    - Each foreground component is a monochromatic left-half shape whose rightmost column sits just left of a vertical half-column symmetry axis.
    - Components are separated from each other, and each one becomes a full horizontally symmetric shape after mirroring across its own axis.

    Rule:
    - For every foreground component, take its global rightmost occupied column as `col_R`, mirror every pixel to column `2*col_R + 1 - c`, and keep the original pixels.
    - On the mirrored full shape, test whether there exists any 4-connected background region enclosed by the shape and unreachable from the outside.
    - If the completed shape is hollow, fill all enclosed holes with the same color; if it is hole-free, erase every non-boundary pixel and keep only the 4-connected outer boundary.

    Key insight:
    - The visible half does not reveal the final topology by itself; only after mirror completion can each object be classified as hollow or solid, and that classification determines the opposite morphological action.
    """
    g = np.asarray(grid, dtype=int)
    if g.ndim != 2:
        raise ValueError("expected a 2D grid")

    bg = _get_bg(g)
    out = np.full_like(g, bg)
    comps = _find_components(g, bg)

    for comp in comps:
        full_mask, top_left = _mirror_half_component(comp, g.shape)
        holes, _ = _background_regions(full_mask)
        if holes.any():
            transformed = full_mask | holes
        else:
            transformed = _outer_boundary(full_mask)
        _paint_mask(out, transformed, top_left, comp["color"])

    return out


def _is_connected(mask: np.ndarray) -> bool:
    coords = np.argwhere(mask)
    if coords.size == 0:
        return False
    h, w = mask.shape
    seen = np.zeros((h, w), dtype=bool)
    start = tuple(map(int, coords[0]))
    q = deque([start])
    seen[start] = True
    count = 0
    while q:
        r, c = q.popleft()
        count += 1
        for nr, nc in _neighbors4(r, c):
            if 0 <= nr < h and 0 <= nc < w and mask[nr, nc] and not seen[nr, nc]:
                seen[nr, nc] = True
                q.append((nr, nc))
    return count == int(mask.sum())


def _mirror_half_mask(half: np.ndarray) -> np.ndarray:
    h, q = half.shape
    full = np.zeros((h, 2 * q), dtype=bool)
    for r in range(h):
        for c in range(q):
            if half[r, c]:
                full[r, c] = True
                full[r, 2 * q - 1 - c] = True
    return full


def _mask_from_intervals(outer: list[int], inner: list[int], q: int) -> np.ndarray:
    h = len(outer)
    mask = np.zeros((h, q), dtype=bool)
    for r in range(h):
        if outer[r] <= inner[r]:
            mask[r, outer[r] : inner[r] + 1] = True
    return mask


def _half_bbox_width(mask: np.ndarray) -> int:
    cols = np.where(mask.any(axis=0))[0]
    return 0 if cols.size == 0 else int(cols[-1] - cols[0] + 1)


def _generate_hollow_half() -> np.ndarray | None:
    for _ in range(200):
        h = random.randint(4, 8)
        q = random.randint(3, 6)
        top_cap = 1 if h == 4 else random.randint(1, min(2, h - 3))
        bottom_cap_max = min(2, h - top_cap - 2)
        if bottom_cap_max < 1:
            continue
        bottom_cap = random.randint(1, bottom_cap_max)
        interior = h - top_cap - bottom_cap
        if interior < 2:
            continue

        max_gap = min(3, q - 1)
        gap = random.randint(1, max_gap)
        gaps = []
        for i in range(interior):
            if i > 0:
                gap = max(1, min(max_gap, gap + random.choice((-1, 0, 1))))
            gaps.append(gap)
        inner = [q - 1] * top_cap + [q - 1 - g for g in gaps] + [q - 1] * bottom_cap

        outer = [random.randint(0, min(1, inner[0]))]
        for r in range(1, h):
            limit = inner[r]
            candidate = outer[-1] + random.choice((-1, 0, 1))
            outer.append(max(0, min(limit, candidate)))

        half = _mask_from_intervals(outer, inner, q)
        if not _is_connected(half):
            continue
        if not (8 <= int(half.sum()) <= 20):
            continue
        if not (3 <= _half_bbox_width(half) <= 6):
            continue

        full = _mirror_half_mask(half)
        holes, _ = _background_regions(full)
        if not holes.any():
            continue
        return half
    return None


def _generate_solid_half() -> np.ndarray | None:
    for _ in range(200):
        h = random.randint(4, 8)
        q = random.randint(3, 6)
        outer = [random.randint(0, max(0, q - 3))]
        for _r in range(1, h):
            candidate = outer[-1] + random.choice((-1, 0, 1))
            outer.append(max(0, min(q - 1, candidate)))
        half = _mask_from_intervals(outer, [q - 1] * h, q)

        if not _is_connected(half):
            continue
        if not (8 <= int(half.sum()) <= 20):
            continue
        if not (3 <= _half_bbox_width(half) <= 6):
            continue

        full = _mirror_half_mask(half)
        holes, _ = _background_regions(full)
        if holes.any():
            continue
        boundary = _outer_boundary(full)
        if int(boundary.sum()) == int(full.sum()):
            continue
        return half
    return None


def _paint_half(grid: np.ndarray, half: np.ndarray, r0: int, c0: int, color: int) -> None:
    rows, cols = np.where(half)
    grid[rows + r0, cols + c0] = int(color)


def _validate_input(grid: np.ndarray) -> bool:
    if grid.ndim != 2 or max(grid.shape) > 30:
        return False
    comps = _find_components(grid, 0)
    if len(comps) != 2:
        return False
    if len({comp["color"] for comp in comps}) != 2:
        return False

    by_row = sorted(comps, key=lambda comp: comp["bbox"][0])
    if by_row[0]["bbox"][2] >= by_row[1]["bbox"][0]:
        return False

    occupied = np.zeros_like(grid, dtype=bool)
    hole_flags = []
    for comp in comps:
        r0, c0, r1, c1 = comp["bbox"]
        if not (4 <= r1 - r0 + 1 <= 8):
            return False
        if not (3 <= c1 - c0 + 1 <= 6):
            return False
        if not (8 <= comp["area"] <= 20):
            return False

        full_mask, top_left = _mirror_half_component(comp, grid.shape)
        rows, cols = np.where(full_mask)
        rr = rows + top_left[0]
        cc = cols + top_left[1]
        if occupied[rr, cc].any():
            return False
        occupied[rr, cc] = True

        holes, _ = _background_regions(full_mask)
        hole_flags.append(bool(holes.any()))

    if sum(hole_flags) != 1:
        return False

    out = main(grid)
    if np.array_equal(out, grid):
        return False
    return True


def _attempt_generate() -> np.ndarray | None:
    hollow = _generate_hollow_half()
    solid = _generate_solid_half()
    if hollow is None or solid is None:
        return None

    top_half, bottom_half = (hollow, solid) if random.random() < 0.5 else (solid, hollow)
    top_h, top_q = top_half.shape
    bottom_h, bottom_q = bottom_half.shape
    top_full_w = 2 * top_q
    bottom_full_w = 2 * bottom_q

    gap = random.randint(2, 5)
    min_h = top_h + bottom_h + gap
    grid_h = random.randint(max(16, min_h), min(24, max(16, min_h) + 5))
    min_w = max(top_full_w, bottom_full_w)
    grid_w = random.randint(max(16, min_w + 1), min(24, max(16, min_w + 6)))

    top_r0 = random.randint(0, grid_h - (top_h + gap + bottom_h))
    bottom_r0 = random.randint(top_r0 + top_h + gap, grid_h - bottom_h)
    top_c0 = random.randint(0, grid_w - top_full_w)
    bottom_c0 = random.randint(0, grid_w - bottom_full_w)

    colors = random.sample([1, 2, 3, 4, 6, 7, 8, 9], 2)
    grid = np.zeros((grid_h, grid_w), dtype=int)
    _paint_half(grid, top_half, top_r0, top_c0, colors[0])
    _paint_half(grid, bottom_half, bottom_r0, bottom_c0, colors[1])

    if not _validate_input(grid):
        return None
    return grid


def generate_input() -> np.ndarray:
    for _ in range(300):
        grid = _attempt_generate()
        if grid is not None:
            return grid
    raise RuntimeError("failed to generate valid puzzle input")
