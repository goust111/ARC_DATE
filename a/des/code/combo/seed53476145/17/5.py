from __future__ import annotations

import random
from collections import Counter, deque

import numpy as np


def _find_components(grid: np.ndarray) -> list[dict]:
    """Return 4-connected monochrome foreground components with tight bbox patches."""
    g = np.asarray(grid, dtype=int)
    h, w = g.shape
    seen = np.zeros((h, w), dtype=bool)
    components: list[dict] = []

    for r in range(h):
        for c in range(w):
            color = int(g[r, c])
            if color == 0 or seen[r, c]:
                continue

            queue = deque([(r, c)])
            seen[r, c] = True
            coords: list[tuple[int, int]] = []

            while queue:
                cr, cc = queue.popleft()
                coords.append((cr, cc))
                for dr, dc in ((1, 0), (-1, 0), (0, 1), (0, -1)):
                    nr, nc = cr + dr, cc + dc
                    if not (0 <= nr < h and 0 <= nc < w):
                        continue
                    if seen[nr, nc] or int(g[nr, nc]) != color:
                        continue
                    seen[nr, nc] = True
                    queue.append((nr, nc))

            rows = [rr for rr, _ in coords]
            cols = [cc for _, cc in coords]
            r0, r1 = min(rows), max(rows)
            c0, c1 = min(cols), max(cols)
            box_h = r1 - r0 + 1
            box_w = c1 - c0 + 1
            patch = np.zeros((box_h, box_w), dtype=int)
            for rr, cc in coords:
                patch[rr - r0, cc - c0] = color

            components.append(
                {
                    "color": color,
                    "bbox": (r0, c0, r1, c1),
                    "patch": patch,
                    "area": len(coords),
                    "projection": 2 * (box_h + box_w),
                }
            )

    return components


def _paste_centered(out: np.ndarray, patch: np.ndarray) -> None:
    h, w = patch.shape
    dst_r = (out.shape[0] - h) // 2
    dst_c = (out.shape[1] - w) // 2
    mask = patch != 0
    target = out[dst_r : dst_r + h, dst_c : dst_c + w]
    target[mask] = patch[mask]


def main(grid: np.ndarray) -> np.ndarray:
    """Sort sprites by bbox perimeter and redraw them with aligned bbox centers."""
    g = np.asarray(grid, dtype=int)
    assert g.ndim == 2

    components = _find_components(g)
    assert components, "expected at least one foreground sprite"

    components.sort(key=lambda comp: (-comp["projection"], comp["bbox"][0], comp["bbox"][1]))

    out = np.zeros_like(g)
    for comp in components:
        _paste_centered(out, comp["patch"])
    return out


# --------------- generator ---------------

_COLOR_POOL = tuple(range(1, 10))
_GRID_H_RANGE = (16, 24)
_GRID_W_RANGE = (16, 24)
_OBJECT_COUNT_RANGE = (3, 5)
_SMALL_SIDE_RANGE = (3, 5)
_GROWTH_RANGE = (1, 3)
_PLACEMENT_PADDING = 1
_MAX_GENERATION_ATTEMPTS = 400


def _sample_bbox_dims(n_objects: int, grid_h: int, grid_w: int) -> list[tuple[int, int]] | None:
    """Sample strictly nestable bbox sizes with unique perimeters."""
    dims = [
        (
            random.randint(*_SMALL_SIDE_RANGE),
            random.randint(*_SMALL_SIDE_RANGE),
        )
    ]

    for _ in range(1, n_objects):
        prev_h, prev_w = dims[-1]
        dims.append(
            (
                prev_h + random.randint(*_GROWTH_RANGE),
                prev_w + random.randint(*_GROWTH_RANGE),
            )
        )

    outer_h, outer_w = dims[-1]
    if outer_h > grid_h - 4 or outer_w > grid_w - 4:
        return None
    return list(reversed(dims))


def _random_path(start: tuple[int, int], end: tuple[int, int]) -> list[tuple[int, int]]:
    """Return one monotone Manhattan path inside the bbox."""
    r, c = start
    er, ec = end
    path = [(r, c)]
    while (r, c) != (er, ec):
        candidates = []
        if r < er:
            candidates.append((r + 1, c))
        elif r > er:
            candidates.append((r - 1, c))
        if c < ec:
            candidates.append((r, c + 1))
        elif c > ec:
            candidates.append((r, c - 1))
        r, c = random.choice(candidates)
        path.append((r, c))
    return path


def _frontier(mask: np.ndarray) -> list[tuple[int, int]]:
    h, w = mask.shape
    cells = []
    for r in range(h):
        for c in range(w):
            if mask[r, c]:
                continue
            for dr, dc in ((1, 0), (-1, 0), (0, 1), (0, -1)):
                nr, nc = r + dr, c + dc
                if 0 <= nr < h and 0 <= nc < w and mask[nr, nc]:
                    cells.append((r, c))
                    break
    return cells


def _make_mask(height: int, width: int, density: float) -> np.ndarray:
    """Build a random 4-connected mask whose tight bbox is exactly (height, width)."""
    mask = np.zeros((height, width), dtype=bool)
    hub = (random.randrange(height), random.randrange(width))
    anchors = [
        (0, random.randrange(width)),
        (height - 1, random.randrange(width)),
        (random.randrange(height), 0),
        (random.randrange(height), width - 1),
    ]
    for anchor in anchors:
        for rr, cc in _random_path(hub, anchor):
            mask[rr, cc] = True

    max_extra = height * width - int(mask.sum())
    target_extra = int(round(density * max_extra))
    target_extra = max(0, min(max_extra, target_extra + random.randint(-2, 2)))
    for _ in range(target_extra):
        options = _frontier(mask)
        if not options:
            break
        rr, cc = random.choice(options)
        mask[rr, cc] = True

    return mask


def _make_patch(height: int, width: int, color: int, density: float) -> np.ndarray:
    patch = np.zeros((height, width), dtype=int)
    patch[_make_mask(height, width, density)] = color
    return patch


def _rect_clear(
    placed: list[tuple[int, int, int, int]],
    top: int,
    left: int,
    height: int,
    width: int,
    padding: int,
) -> bool:
    r1 = top + height - 1
    c1 = left + width - 1
    for pr0, pc0, pr1, pc1 in placed:
        if not (
            r1 < pr0 - padding
            or pr1 < top - padding
            or c1 < pc0 - padding
            or pc1 < left - padding
        ):
            return False
    return True


def _place_patches(
    patches: list[np.ndarray], grid_h: int, grid_w: int
) -> np.ndarray | None:
    grid = np.zeros((grid_h, grid_w), dtype=int)
    placed: list[tuple[int, int, int, int]] = []

    for patch in sorted(patches, key=lambda arr: arr.shape[0] * arr.shape[1], reverse=True):
        h, w = patch.shape
        positions = [
            (r, c)
            for r in range(grid_h - h + 1)
            for c in range(grid_w - w + 1)
            if _rect_clear(placed, r, c, h, w, _PLACEMENT_PADDING)
        ]
        if not positions:
            return None
        top, left = random.choice(positions)
        mask = patch != 0
        grid[top : top + h, left : left + w][mask] = patch[mask]
        placed.append((top, left, top + h - 1, left + w - 1))

    return grid


def _has_perimeter_area_inversion(grid: np.ndarray) -> bool:
    comps = _find_components(grid)
    by_projection = sorted(comps, key=lambda comp: -comp["projection"])
    by_area = sorted(comps, key=lambda comp: -comp["area"])
    return [comp["color"] for comp in by_projection] != [comp["color"] for comp in by_area]


def _is_valid_input(grid: np.ndarray, expected_count: int) -> bool:
    comps = _find_components(grid)
    if len(comps) != expected_count:
        return False
    projections = [comp["projection"] for comp in comps]
    if len(set(projections)) != len(projections):
        return False
    if not _has_perimeter_area_inversion(grid):
        return False

    colors = [comp["color"] for comp in comps]
    if len(set(colors)) != len(colors):
        return False

    out = main(grid)
    out_colors = set(map(int, out.ravel())) - {0}
    return out_colors == set(colors)


def _density_schedule(n_objects: int) -> list[float]:
    levels = [0.08, 0.18, 0.32, 0.5, 0.7, 0.85]
    return random.sample(levels, n_objects)


def _gen_attempt() -> np.ndarray | None:
    grid_h = random.randint(*_GRID_H_RANGE)
    grid_w = random.randint(*_GRID_W_RANGE)
    n_objects = random.randint(*_OBJECT_COUNT_RANGE)

    dims = _sample_bbox_dims(n_objects, grid_h, grid_w)
    if dims is None:
        return None

    colors = random.sample(_COLOR_POOL, n_objects)
    densities = _density_schedule(n_objects)
    patches = [
        _make_patch(height, width, color, density)
        for (height, width), color, density in zip(dims, colors, densities)
    ]

    grid = _place_patches(patches, grid_h, grid_w)
    if grid is None:
        return None
    if not _is_valid_input(grid, n_objects):
        return None
    return grid


def generate_input() -> np.ndarray:
    for _ in range(_MAX_GENERATION_ATTEMPTS):
        grid = _gen_attempt()
        if grid is not None:
            return grid
    raise RuntimeError("failed to generate a valid input")

