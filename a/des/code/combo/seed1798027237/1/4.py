from __future__ import annotations

from collections import defaultdict
import random

import numpy as np


DIR4 = ((1, 0), (-1, 0), (0, 1), (0, -1))


def _components_of_mask(mask: np.ndarray) -> list[np.ndarray]:
    """Return 4-connected components of a boolean mask as coordinate arrays."""
    h, w = map(int, mask.shape)
    seen = np.zeros((h, w), dtype=bool)
    comps: list[np.ndarray] = []
    for r in range(h):
        for c in range(w):
            if not mask[r, c] or seen[r, c]:
                continue
            stack = [(r, c)]
            seen[r, c] = True
            pts: list[tuple[int, int]] = []
            while stack:
                cr, cc = stack.pop()
                pts.append((cr, cc))
                for dr, dc in DIR4:
                    nr, nc = cr + dr, cc + dc
                    if 0 <= nr < h and 0 <= nc < w and mask[nr, nc] and not seen[nr, nc]:
                        seen[nr, nc] = True
                        stack.append((nr, nc))
            comps.append(np.asarray(pts, dtype=int))
    return comps


def _color_components(grid: np.ndarray) -> list[dict[str, object]]:
    """Return all non-black monochrome 4-connected components."""
    g = np.asarray(grid, dtype=int)
    h, w = map(int, g.shape)
    seen = np.zeros((h, w), dtype=bool)
    comps: list[dict[str, object]] = []
    for r in range(h):
        for c in range(w):
            color = int(g[r, c])
            if color == 0 or seen[r, c]:
                continue
            stack = [(r, c)]
            seen[r, c] = True
            pts: list[tuple[int, int]] = []
            while stack:
                cr, cc = stack.pop()
                pts.append((cr, cc))
                for dr, dc in DIR4:
                    nr, nc = cr + dr, cc + dc
                    if 0 <= nr < h and 0 <= nc < w and not seen[nr, nc] and int(g[nr, nc]) == color:
                        seen[nr, nc] = True
                        stack.append((nr, nc))
            comps.append({"color": color, "coords": np.asarray(pts, dtype=int)})
    return comps


def _bbox(coords: np.ndarray) -> tuple[int, int, int, int]:
    rows = coords[:, 0]
    cols = coords[:, 1]
    return int(rows.min()), int(cols.min()), int(rows.max()), int(cols.max())


def _count_holes_for_component(grid: np.ndarray, coords: np.ndarray, color: int) -> int:
    """Count enclosed black 4-components fully surrounded by this component."""
    r0, c0, r1, c1 = _bbox(coords)
    sub = np.asarray(grid[r0 : r1 + 1, c0 : c1 + 1], dtype=int)
    holes = 0
    for pts in _components_of_mask(sub == 0):
        rows = pts[:, 0]
        cols = pts[:, 1]
        touches_box = bool(
            np.any((rows == 0) | (rows == sub.shape[0] - 1) | (cols == 0) | (cols == sub.shape[1] - 1))
        )
        if touches_box:
            continue
        adjacent_colors: set[int] = set()
        for rr, cc in pts:
            for dr, dc in DIR4:
                nr, nc = int(rr + dr), int(cc + dc)
                if 0 <= nr < sub.shape[0] and 0 <= nc < sub.shape[1]:
                    v = int(sub[nr, nc])
                    if v != 0:
                        adjacent_colors.add(v)
        if adjacent_colors == {int(color)}:
            holes += 1
    return holes


def _boundary_pixels(grid: np.ndarray, coords: np.ndarray) -> list[tuple[int, int]]:
    """Return component pixels 4-adjacent to black."""
    g = np.asarray(grid, dtype=int)
    h, w = map(int, g.shape)
    boundary: list[tuple[int, int]] = []
    for r, c in coords:
        rr, cc = int(r), int(c)
        for dr, dc in DIR4:
            nr, nc = rr + dr, cc + dc
            if 0 <= nr < h and 0 <= nc < w and int(g[nr, nc]) == 0:
                boundary.append((rr, cc))
                break
    return boundary


def main(grid: np.ndarray) -> np.ndarray:
    """
    Pair monochrome regions by enclosed-hole count, then swap their boundary colors.

    Input structure:
    - Black background.
    - An even number of disconnected single-color 4-connected regions.
    - Each hole count appears exactly twice across regions.

    Rule:
    - Count enclosed black holes inside each region.
    - Pair the two regions with the same hole count.
    - Repaint each region's black-adjacent boundary pixels using its partner's color.
    """
    g = np.asarray(grid, dtype=int)
    out = g.copy()

    groups: dict[int, list[dict[str, object]]] = defaultdict(list)
    comps = _color_components(g)
    assert len(comps) % 2 == 0 and comps, "expected a non-empty even number of regions"

    for comp in comps:
        color = int(comp["color"])
        coords = np.asarray(comp["coords"], dtype=int)
        groups[_count_holes_for_component(g, coords, color)].append(
            {
                "color": color,
                "boundary": _boundary_pixels(g, coords),
            }
        )

    for hole_count, pair in groups.items():
        assert len(pair) == 2, f"hole count {hole_count} must appear exactly twice"
        first, second = pair
        for r, c in first["boundary"]:
            out[r, c] = int(second["color"])
        for r, c in second["boundary"]:
            out[r, c] = int(first["color"])

    return out


# --------------- generator ---------------

GRID_MIN = 20
GRID_MAX = 30
PAIR_COUNT_RANGE = (2, 4)
MAX_HOLE_COUNT = 3
SHAPE_ATTEMPTS = 200
PLACEMENT_ATTEMPTS = 200
GENERATION_ATTEMPTS = 400
MIN_SHAPE_AREA = 18

_SHAPE_POOLS: dict[int, list[np.ndarray]] = {}


def _count_mask_holes(mask: np.ndarray) -> int:
    h, w = map(int, mask.shape)
    holes = 0
    for pts in _components_of_mask(~mask):
        rows = pts[:, 0]
        cols = pts[:, 1]
        touches_border = bool(
            np.any((rows == 0) | (rows == h - 1) | (cols == 0) | (cols == w - 1))
        )
        if not touches_border:
            holes += 1
    return holes


def _is_connected(mask: np.ndarray) -> bool:
    comps = _components_of_mask(mask)
    return len(comps) == 1


def _core_pixel_count(mask: np.ndarray) -> int:
    h, w = map(int, mask.shape)
    if h < 3 or w < 3:
        return 0
    core = mask[1:-1, 1:-1].copy()
    core &= mask[:-2, 1:-1]
    core &= mask[2:, 1:-1]
    core &= mask[1:-1, :-2]
    core &= mask[1:-1, 2:]
    return int(core.sum())


def _crop_mask(mask: np.ndarray) -> np.ndarray:
    rows = np.where(mask.any(axis=1))[0]
    cols = np.where(mask.any(axis=0))[0]
    return mask[rows[0] : rows[-1] + 1, cols[0] : cols[-1] + 1]


def _exterior_background(mask: np.ndarray) -> np.ndarray:
    h, w = map(int, mask.shape)
    bg = ~mask
    ext = np.zeros((h, w), dtype=bool)
    stack: list[tuple[int, int]] = []
    for r in range(h):
        for c in (0, w - 1):
            if bg[r, c] and not ext[r, c]:
                ext[r, c] = True
                stack.append((r, c))
    for c in range(w):
        for r in (0, h - 1):
            if bg[r, c] and not ext[r, c]:
                ext[r, c] = True
                stack.append((r, c))
    while stack:
        r, c = stack.pop()
        for dr, dc in DIR4:
            nr, nc = r + dr, c + dc
            if 0 <= nr < h and 0 <= nc < w and bg[nr, nc] and not ext[nr, nc]:
                ext[nr, nc] = True
                stack.append((nr, nc))
    return ext


def _outer_boundary_cells(mask: np.ndarray) -> list[tuple[int, int]]:
    h, w = map(int, mask.shape)
    ext = _exterior_background(mask)
    cells: list[tuple[int, int]] = []
    for r in range(h):
        for c in range(w):
            if not mask[r, c]:
                continue
            for dr, dc in DIR4:
                nr, nc = r + dr, c + dc
                if not (0 <= nr < h and 0 <= nc < w):
                    cells.append((r, c))
                    break
                if ext[nr, nc]:
                    cells.append((r, c))
                    break
    return cells


def _exterior_growth_cells(mask: np.ndarray) -> list[tuple[int, int]]:
    h, w = map(int, mask.shape)
    ext = _exterior_background(mask)
    cells: list[tuple[int, int]] = []
    for r in range(h):
        for c in range(w):
            if not ext[r, c]:
                continue
            for dr, dc in DIR4:
                nr, nc = r + dr, c + dc
                if 0 <= nr < h and 0 <= nc < w and mask[nr, nc]:
                    cells.append((r, c))
                    break
    return cells


def _pick_hole_positions(height: int, width: int, hole_count: int) -> list[tuple[int, int]] | None:
    if hole_count == 0:
        return []
    candidates = [(r, c) for r in range(2, height - 2) for c in range(2, width - 2)]
    random.shuffle(candidates)
    chosen: list[tuple[int, int]] = []
    for r, c in candidates:
        if all(max(abs(r - rr), abs(c - cc)) >= 2 for rr, cc in chosen):
            chosen.append((r, c))
            if len(chosen) == hole_count:
                return chosen
    return None


def _random_transform(mask: np.ndarray) -> np.ndarray:
    out = np.rot90(mask, random.randint(0, 3))
    if random.random() < 0.5:
        out = np.fliplr(out)
    if random.random() < 0.5:
        out = np.flipud(out)
    return np.asarray(out, dtype=bool)


def _build_shape(hole_count: int) -> np.ndarray:
    for _ in range(SHAPE_ATTEMPTS):
        min_dim = 6 + hole_count
        if hole_count > 0:
            min_dim = max(min_dim, 7)
        height = random.randint(min_dim, min_dim + 3)
        width = random.randint(min_dim, min_dim + 3)
        holes = _pick_hole_positions(height, width, hole_count)
        if holes is None:
            continue

        mask = np.ones((height, width), dtype=bool)
        for r, c in holes:
            mask[r, c] = False
        if _count_mask_holes(mask) != hole_count:
            continue

        mask = np.pad(mask, 1, constant_values=False)
        cuts_done = 0
        target_cuts = random.randint(2 + hole_count, 6 + 2 * hole_count)
        for _ in range(40):
            candidates = _outer_boundary_cells(mask)
            random.shuffle(candidates)
            changed = False
            for r, c in candidates:
                test = mask.copy()
                test[r, c] = False
                if int(test.sum()) < MIN_SHAPE_AREA:
                    continue
                if not _is_connected(test):
                    continue
                if _count_mask_holes(test) != hole_count:
                    continue
                if _core_pixel_count(test) < 1:
                    continue
                mask = test
                cuts_done += 1
                changed = True
                break
            if not changed or cuts_done >= target_cuts:
                break

        adds_done = 0
        target_adds = random.randint(1, 4)
        for _ in range(20):
            candidates = _exterior_growth_cells(mask)
            random.shuffle(candidates)
            changed = False
            for r, c in candidates:
                test = mask.copy()
                test[r, c] = True
                if _count_mask_holes(test) != hole_count:
                    continue
                if _core_pixel_count(test) < 1:
                    continue
                mask = test
                adds_done += 1
                changed = True
                break
            if not changed or adds_done >= target_adds:
                break

        mask = _crop_mask(mask)
        if int(mask.sum()) < MIN_SHAPE_AREA:
            continue
        if not _is_connected(mask):
            continue
        if _count_mask_holes(mask) != hole_count:
            continue
        if _core_pixel_count(mask) < 1:
            continue
        if cuts_done + adds_done < 2:
            continue
        if hole_count == 0 and np.all(mask):
            continue
        return mask
    raise RuntimeError(f"failed to generate shape with {hole_count} holes")


def _generate_shape(hole_count: int) -> np.ndarray:
    pool = _SHAPE_POOLS.setdefault(hole_count, [])
    target_pool = 10
    while len(pool) < target_pool:
        pool.append(_build_shape(hole_count))
    if random.random() < 0.15 and len(pool) < 20:
        pool.append(_build_shape(hole_count))
    return _random_transform(np.asarray(random.choice(pool), dtype=bool))


def _place_shapes(items: list[dict[str, object]], height: int, width: int) -> np.ndarray | None:
    grid = np.zeros((height, width), dtype=int)
    reserved = np.zeros((height, width), dtype=bool)
    order = sorted(items, key=lambda item: -int(np.asarray(item["mask"], dtype=bool).sum()))

    for item in order:
        mask = np.asarray(item["mask"], dtype=bool)
        color = int(item["color"])
        sh, sw = map(int, mask.shape)
        if sh + 2 > height or sw + 2 > width:
            return None

        placed = False
        for _ in range(PLACEMENT_ATTEMPTS):
            top = random.randint(1, height - sh - 1)
            left = random.randint(1, width - sw - 1)
            r0 = top - 1
            c0 = left - 1
            r1 = top + sh + 1
            c1 = left + sw + 1
            if reserved[r0:r1, c0:c1].any():
                continue
            window = grid[top : top + sh, left : left + sw]
            if np.any(window[mask] != 0):
                continue
            window[mask] = color
            reserved[r0:r1, c0:c1] = True
            placed = True
            break
        if not placed:
            return None

    return grid


def _valid_pairing_structure(grid: np.ndarray) -> bool:
    comps = _color_components(grid)
    if not comps or len(comps) % 2 != 0:
        return False
    by_count: dict[int, int] = defaultdict(int)
    for comp in comps:
        color = int(comp["color"])
        coords = np.asarray(comp["coords"], dtype=int)
        by_count[_count_holes_for_component(grid, coords, color)] += 1
    return all(v == 2 for v in by_count.values())


def _gen_attempt() -> np.ndarray | None:
    pair_count = random.randint(*PAIR_COUNT_RANGE)
    hole_counts = random.sample(range(MAX_HOLE_COUNT + 1), pair_count)
    colors = random.sample(range(1, 10), 2 * pair_count)

    items: list[dict[str, object]] = []
    for hole_count in hole_counts:
        for _ in range(2):
            items.append(
                {
                    "holes": hole_count,
                    "mask": _generate_shape(hole_count),
                }
            )
    random.shuffle(items)
    for item, color in zip(items, colors):
        item["color"] = color

    low = max(GRID_MIN, 16 + 2 * pair_count)
    height = random.randint(low, GRID_MAX)
    width = random.randint(low, GRID_MAX)
    grid = _place_shapes(items, height, width)
    if grid is None:
        return None
    if not _valid_pairing_structure(grid):
        return None
    try:
        out = main(grid)
    except Exception:
        return None
    if np.array_equal(out, grid):
        return None
    return grid


def generate_input() -> np.ndarray:
    for _ in range(GENERATION_ATTEMPTS):
        grid = _gen_attempt()
        if grid is not None:
            return grid
    raise RuntimeError("failed to generate valid input")
