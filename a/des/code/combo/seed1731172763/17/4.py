from __future__ import annotations

import numpy as np


BACKGROUND = 0
BLUE = 1
RED = 2
GREEN = 3
YELLOW = 4

INPUT_COLORS = (5, 6, 7, 8, 9)
KIND_TO_OUTPUT = {
    ("cold", "deep"): BLUE,
    ("cold", "light"): GREEN,
    ("warm", "deep"): RED,
    ("warm", "light"): YELLOW,
}
KIND_ORDER = tuple(KIND_TO_OUTPUT)

SHAPE_ATTEMPTS = 120
PLACEMENT_ATTEMPTS = 200
GENERATION_ATTEMPTS = 400


def _neighbors4(r: int, c: int, h: int, w: int):
    if r > 0:
        yield r - 1, c
    if r + 1 < h:
        yield r + 1, c
    if c > 0:
        yield r, c - 1
    if c + 1 < w:
        yield r, c + 1


def _find_components(grid: np.ndarray) -> list[tuple[int, np.ndarray]]:
    h, w = grid.shape
    seen = np.zeros((h, w), dtype=bool)
    components: list[tuple[int, np.ndarray]] = []

    for r in range(h):
        for c in range(w):
            color = int(grid[r, c])
            if color == BACKGROUND or seen[r, c]:
                continue

            stack = [(r, c)]
            seen[r, c] = True
            coords: list[tuple[int, int]] = []
            while stack:
                rr, cc = stack.pop()
                coords.append((rr, cc))
                for nr, nc in _neighbors4(rr, cc, h, w):
                    if not seen[nr, nc] and int(grid[nr, nc]) == color:
                        seen[nr, nc] = True
                        stack.append((nr, nc))

            components.append((color, np.array(coords, dtype=int)))

    return components


def _bbox_from_coords(coords: np.ndarray) -> tuple[int, int, int, int]:
    rows = coords[:, 0]
    cols = coords[:, 1]
    return int(rows.min()), int(cols.min()), int(rows.max()), int(cols.max())


def _classify_coords(coords: np.ndarray) -> int:
    r0, c0, r1, c1 = _bbox_from_coords(coords)
    center_r = (r0 + r1) / 2.0
    center_c = (c0 + c1) / 2.0
    centroid_r = float(coords[:, 0].mean())
    centroid_c = float(coords[:, 1].mean())

    horizontal = "cold" if centroid_c < center_c else "warm"
    vertical = "deep" if centroid_r < center_r else "light"
    return KIND_TO_OUTPUT[(horizontal, vertical)]


def main(grid: np.ndarray) -> np.ndarray:
    """
    Recolor each foreground object by the direction of the offset between its
    pixel centroid and the center of its bounding box.

    Cold + deep  -> blue
    Cold + light -> green
    Warm + deep  -> red
    Warm + light -> yellow
    """
    g = np.asarray(grid, dtype=int)
    assert g.ndim == 2, "expected a 2D grid"

    out = np.full(g.shape, BACKGROUND, dtype=int)
    for _color, coords in _find_components(g):
        out_color = _classify_coords(coords)
        out[coords[:, 0], coords[:, 1]] = out_color
    return out


def _mask_has_symmetry(mask: np.ndarray) -> bool:
    if np.array_equal(mask, np.flipud(mask)):
        return True
    if np.array_equal(mask, np.fliplr(mask)):
        return True
    if np.array_equal(mask, np.rot90(mask, 2)):
        return True
    if mask.shape[0] == mask.shape[1]:
        if np.array_equal(mask, mask.T):
            return True
        if np.array_equal(mask, np.rot90(mask, 2).T):
            return True
    return False


def _direction_weights(h: int, w: int, kind: tuple[str, str]) -> np.ndarray:
    horizontal, vertical = kind
    row_weights = np.arange(h, 0, -1, dtype=float) if vertical == "deep" else np.arange(1, h + 1, dtype=float)
    col_weights = np.arange(w, 0, -1, dtype=float) if horizontal == "cold" else np.arange(1, w + 1, dtype=float)
    return row_weights[:, None] * col_weights[None, :]


def _weighted_choice_index(rng: np.random.Generator, weights: np.ndarray) -> int:
    total = float(weights.sum())
    assert total > 0.0
    pick = float(rng.random() * total)
    running = 0.0
    for idx, weight in enumerate(weights):
        running += float(weight)
        if pick <= running:
            return idx
    return len(weights) - 1


def _frontier(mask: np.ndarray) -> list[tuple[int, int]]:
    h, w = mask.shape
    cells: set[tuple[int, int]] = set()
    for r, c in np.argwhere(mask):
        rr = int(r)
        cc = int(c)
        for nr, nc in _neighbors4(rr, cc, h, w):
            if not mask[nr, nc]:
                cells.add((nr, nc))
    return list(cells)


def _mask_coords(mask: np.ndarray) -> np.ndarray:
    return np.argwhere(mask).astype(int)


def _matches_kind(mask: np.ndarray, kind: tuple[str, str]) -> bool:
    coords = _mask_coords(mask)
    r0, c0, r1, c1 = _bbox_from_coords(coords)
    center_r = (r0 + r1) / 2.0
    center_c = (c0 + c1) / 2.0
    centroid_r = float(coords[:, 0].mean())
    centroid_c = float(coords[:, 1].mean())
    horizontal, vertical = kind
    if horizontal == "cold" and not (centroid_c < center_c):
        return False
    if horizontal == "warm" and not (centroid_c > center_c):
        return False
    if vertical == "deep" and not (centroid_r < center_r):
        return False
    if vertical == "light" and not (centroid_r > center_r):
        return False
    return True


def _generate_object_mask(rng: np.random.Generator, kind: tuple[str, str]) -> np.ndarray | None:
    horizontal, vertical = kind
    for _ in range(SHAPE_ATTEMPTS):
        h = int(rng.integers(4, 8))
        w = int(rng.integers(4, 8))
        band_h = int(rng.integers(1, min(3, h - 1) + 1))
        band_w = int(rng.integers(1, min(3, w - 1) + 1))

        mask = np.zeros((h, w), dtype=bool)
        if vertical == "deep":
            mask[:band_h, :] = True
        else:
            mask[h - band_h :, :] = True

        if horizontal == "cold":
            mask[:, :band_w] = True
        else:
            mask[:, w - band_w :] = True

        weights = _direction_weights(h, w, kind)
        spare = h * w - int(mask.sum())
        extra_steps = int(rng.integers(1, max(2, min(spare + 1, h + w))))
        for _ in range(extra_steps):
            frontier = _frontier(mask)
            if not frontier:
                break
            frontier_weights = np.array([weights[r, c] for r, c in frontier], dtype=float)
            idx = _weighted_choice_index(rng, frontier_weights)
            rr, cc = frontier[idx]
            mask[rr, cc] = True

        fill_ratio = float(mask.mean())
        if fill_ratio > 0.82:
            continue
        if min(mask.shape) < 3:
            continue
        if _mask_has_symmetry(mask):
            continue
        if not _matches_kind(mask, kind):
            continue
        return mask
    return None


def _mark_forbidden(forbidden: np.ndarray, top: int, left: int, mask: np.ndarray) -> None:
    h, w = forbidden.shape
    for r, c in np.argwhere(mask):
        rr = top + int(r)
        cc = left + int(c)
        for dr in (-1, 0, 1):
            for dc in (-1, 0, 1):
                nr = rr + dr
                nc = cc + dc
                if 0 <= nr < h and 0 <= nc < w:
                    forbidden[nr, nc] = True


def _place_objects(
    rng: np.random.Generator,
    shape: tuple[int, int],
    objects: list[tuple[np.ndarray, int]],
) -> np.ndarray | None:
    h, w = shape
    grid = np.full((h, w), BACKGROUND, dtype=int)
    forbidden = np.zeros((h, w), dtype=bool)

    order = sorted(range(len(objects)), key=lambda idx: int(objects[idx][0].sum()), reverse=True)
    for idx in order:
        mask, color = objects[idx]
        obj_h, obj_w = mask.shape
        row_lo = 1 if h - obj_h >= 2 else 0
        row_hi = h - obj_h - 1 if h - obj_h >= 2 else h - obj_h
        col_lo = 1 if w - obj_w >= 2 else 0
        col_hi = w - obj_w - 1 if w - obj_w >= 2 else w - obj_w
        if row_hi < row_lo or col_hi < col_lo:
            return None

        placed = False
        for _ in range(PLACEMENT_ATTEMPTS):
            top = int(rng.integers(row_lo, row_hi + 1))
            left = int(rng.integers(col_lo, col_hi + 1))
            window = forbidden[top : top + obj_h, left : left + obj_w]
            if np.any(window & mask):
                continue
            patch = grid[top : top + obj_h, left : left + obj_w]
            patch[mask] = int(color)
            _mark_forbidden(forbidden, top, left, mask)
            placed = True
            break
        if not placed:
            return None

    return grid


def _kind_schedule(rng: np.random.Generator, count: int) -> list[tuple[str, str]]:
    kinds = list(KIND_ORDER)
    rng.shuffle(kinds)
    if count <= len(kinds):
        chosen = kinds[:count]
    else:
        chosen = kinds + [kinds[int(rng.integers(0, len(kinds)))]]
    rng.shuffle(chosen)
    return chosen


def _grid_bounds(objects: list[tuple[np.ndarray, int]]) -> tuple[int, int]:
    heights = [obj.shape[0] for obj, _ in objects]
    widths = [obj.shape[1] for obj, _ in objects]
    count = len(objects)
    min_h = max(12, max(heights) + 2, (sum(heights) // 2) + count + 2)
    min_w = max(12, max(widths) + 2, (sum(widths) // 2) + count + 2)
    return min_h, min_w


def _gen_attempt(rng: np.random.Generator) -> np.ndarray | None:
    count = int(rng.integers(2, 6))
    kinds = _kind_schedule(rng, count)
    colors = [int(color) for color in rng.choice(np.array(INPUT_COLORS, dtype=int), size=count, replace=False)]

    objects: list[tuple[np.ndarray, int]] = []
    for kind, color in zip(kinds, colors):
        mask = _generate_object_mask(rng, kind)
        if mask is None:
            return None
        objects.append((mask, color))

    min_h, min_w = _grid_bounds(objects)
    max_h = max(min_h + 1, 25)
    max_w = max(min_w + 1, 25)
    grid_h = int(rng.integers(min_h, max_h))
    grid_w = int(rng.integers(min_w, max_w))

    grid = _place_objects(rng, (grid_h, grid_w), objects)
    if grid is None:
        return None

    try:
        main(grid)
    except Exception:
        return None
    return grid


def generate_input() -> np.ndarray:
    rng = np.random.default_rng()
    for _ in range(GENERATION_ATTEMPTS):
        grid = _gen_attempt(rng)
        if grid is not None:
            return grid
    raise RuntimeError("failed to generate a valid input")
