from __future__ import annotations

import random
import numpy as np


BG = 0
DIR4 = [(-1, 0), (1, 0), (0, -1), (0, 1)]
PALETTE = [1, 2, 3, 4, 5, 6, 7, 8, 9]
GEN_RETRIES = 400
OBJECT_RETRIES = 120
PLACEMENT_RETRIES = 120


def _components(mask: np.ndarray) -> list[list[tuple[int, int]]]:
    h, w = mask.shape
    seen = np.zeros((h, w), dtype=bool)
    comps: list[list[tuple[int, int]]] = []
    for r in range(h):
        for c in range(w):
            if not mask[r, c] or seen[r, c]:
                continue
            stack = [(r, c)]
            seen[r, c] = True
            comp: list[tuple[int, int]] = []
            while stack:
                cr, cc = stack.pop()
                comp.append((cr, cc))
                for dr, dc in DIR4:
                    nr, nc = cr + dr, cc + dc
                    if 0 <= nr < h and 0 <= nc < w and mask[nr, nc] and not seen[nr, nc]:
                        seen[nr, nc] = True
                        stack.append((nr, nc))
            comps.append(comp)
    return comps


def _bbox_from_points(points: list[tuple[int, int]]) -> tuple[int, int, int, int]:
    rows = [r for r, _ in points]
    cols = [c for _, c in points]
    return min(rows), min(cols), max(rows), max(cols)


def _crop_mask(mask: np.ndarray) -> np.ndarray:
    coords = np.argwhere(mask)
    assert coords.size > 0
    r0, c0 = coords.min(axis=0)
    r1, c1 = coords.max(axis=0)
    return mask[int(r0) : int(r1) + 1, int(c0) : int(c1) + 1]


def _count_holes(mask: np.ndarray) -> int:
    inv = ~mask
    holes = 0
    h, w = mask.shape
    for comp in _components(inv):
        if any(r == 0 or c == 0 or r == h - 1 or c == w - 1 for r, c in comp):
            continue
        holes += 1
    return holes


def _is_connected(mask: np.ndarray) -> bool:
    if not bool(mask.any()):
        return False
    return len(_components(mask)) == 1


def _exterior_background(mask: np.ndarray) -> np.ndarray:
    h, w = mask.shape
    ext = np.zeros((h, w), dtype=bool)
    stack: list[tuple[int, int]] = []
    for r in range(h):
        for c in (0, w - 1):
            if not mask[r, c] and not ext[r, c]:
                ext[r, c] = True
                stack.append((r, c))
    for c in range(w):
        for r in (0, h - 1):
            if not mask[r, c] and not ext[r, c]:
                ext[r, c] = True
                stack.append((r, c))
    while stack:
        r, c = stack.pop()
        for dr, dc in DIR4:
            nr, nc = r + dr, c + dc
            if 0 <= nr < h and 0 <= nc < w and not mask[nr, nc] and not ext[nr, nc]:
                ext[nr, nc] = True
                stack.append((nr, nc))
    return ext


def _object_descriptors(grid: np.ndarray) -> list[dict[str, int]]:
    h, w = grid.shape
    seen = np.zeros((h, w), dtype=bool)
    objects: list[dict[str, int]] = []
    for r in range(h):
        for c in range(w):
            color = int(grid[r, c])
            if color == BG or seen[r, c]:
                continue
            stack = [(r, c)]
            seen[r, c] = True
            pts: list[tuple[int, int]] = []
            while stack:
                cr, cc = stack.pop()
                pts.append((cr, cc))
                for dr, dc in DIR4:
                    nr, nc = cr + dr, cc + dc
                    if 0 <= nr < h and 0 <= nc < w and not seen[nr, nc] and int(grid[nr, nc]) == color:
                        seen[nr, nc] = True
                        stack.append((nr, nc))
            r0, c0, r1, c1 = _bbox_from_points(pts)
            patch = np.zeros((r1 - r0 + 1, c1 - c0 + 1), dtype=bool)
            for pr, pc in pts:
                patch[pr - r0, pc - c0] = True
            objects.append(
                {
                    "color": color,
                    "area": len(pts),
                    "holes": _count_holes(patch),
                }
            )
    return objects


def _render_nested(colors: list[int]) -> np.ndarray:
    assert colors
    n_layers = len(colors)
    side = 2 * n_layers - 1
    out = np.full((side, side), int(colors[0]), dtype=int)
    for depth, color in enumerate(colors[1:], start=1):
        out[depth : side - depth, depth : side - depth] = int(color)
    return out


def main(grid: np.ndarray) -> np.ndarray:
    """
    Rebuild the scene as nested square rings ordered by each object's enclosed-hole count.

    Input structure:
    - The background is black.
    - Several scattered monochrome 4-connected objects appear on the grid, each in a distinct color.
    - Some objects contain one or more black 4-connected holes fully enclosed by that object, while others contain none.

    Rule:
    - Count the number of enclosed black hole components inside every colored object.
    - Group objects by hole count and sort those groups from more holes to fewer holes.
    - Produce a new nested square output whose number of layers equals the number of distinct hole counts.
    - Each hole-count group contributes one 1-pixel-thick square layer; when multiple objects share a hole count, the largest-area object's color wins that entire layer.
    - The innermost group fills the center cell, so a zero-hole group becomes the center when present.

    Key insight:
    - The original shapes and positions do not matter; only the hidden topological property "how many enclosed holes this object has" determines the new nesting order.
    """
    g = np.asarray(grid, dtype=int)
    assert g.ndim == 2, "expected a 2D grid"

    objects = _object_descriptors(g)
    assert objects, "expected at least one colored object"

    best_by_holes: dict[int, dict[str, int]] = {}
    for obj in objects:
        holes = int(obj["holes"])
        chosen = best_by_holes.get(holes)
        if chosen is None or int(obj["area"]) > int(chosen["area"]):
            best_by_holes[holes] = obj

    layer_colors = [int(best_by_holes[h]["color"]) for h in sorted(best_by_holes, reverse=True)]
    return _render_nested(layer_colors)


def _isolated_hole_seeds(h: int, w: int, count: int) -> list[tuple[int, int]] | None:
    cells = [(r, c) for r in range(1, h - 1) for c in range(1, w - 1)]
    random.shuffle(cells)
    seeds: list[tuple[int, int]] = []
    for r, c in cells:
        if all(max(abs(r - rr), abs(c - cc)) > 1 for rr, cc in seeds):
            seeds.append((r, c))
            if len(seeds) == count:
                return seeds
    return None


def _base_object(hole_count: int, size_tier: int) -> np.ndarray | None:
    core_h = random.randint(4 + hole_count + size_tier, 5 + hole_count + size_tier)
    core_w = random.randint(4 + hole_count + size_tier, 5 + hole_count + size_tier)
    mask = np.ones((core_h, core_w), dtype=bool)

    if hole_count == 0:
        return mask

    seeds = _isolated_hole_seeds(core_h, core_w, hole_count)
    if seeds is None:
        return None

    holes: list[set[tuple[int, int]]] = [{seed} for seed in seeds]
    occupied = set(seeds)
    for r, c in seeds:
        mask[r, c] = False

    max_growth = 1 + size_tier + hole_count
    for idx, comp in enumerate(holes):
        for _ in range(random.randint(0, max_growth)):
            frontier: list[tuple[int, int]] = []
            for r, c in comp:
                for dr, dc in DIR4:
                    nr, nc = r + dr, c + dc
                    if not (1 <= nr < core_h - 1 and 1 <= nc < core_w - 1):
                        continue
                    if (nr, nc) in occupied:
                        continue
                    blocked = False
                    for rr in range(nr - 1, nr + 2):
                        for cc in range(nc - 1, nc + 2):
                            if (rr, cc) in occupied and (rr, cc) not in comp:
                                blocked = True
                                break
                        if blocked:
                            break
                    if not blocked:
                        frontier.append((nr, nc))
            if not frontier:
                break
            nr, nc = random.choice(frontier)
            comp.add((nr, nc))
            occupied.add((nr, nc))
            mask[nr, nc] = False

    if _count_holes(mask) != hole_count:
        return None
    return mask


def _irregularize(mask: np.ndarray, hole_count: int, size_tier: int) -> np.ndarray | None:
    pad = 3
    canvas = np.zeros((mask.shape[0] + 2 * pad, mask.shape[1] + 2 * pad), dtype=bool)
    canvas[pad : pad + mask.shape[0], pad : pad + mask.shape[1]] = mask

    growth_steps = random.randint(2 + size_tier, 4 + hole_count + size_tier)
    for _ in range(growth_steps):
        ext = _exterior_background(canvas)
        frontier: list[tuple[int, int]] = []
        h, w = canvas.shape
        for r in range(1, h - 1):
            for c in range(1, w - 1):
                if not ext[r, c]:
                    continue
                if any(canvas[r + dr, c + dc] for dr, dc in DIR4):
                    frontier.append((r, c))
        if not frontier:
            break
        r, c = random.choice(frontier)
        canvas[r, c] = True

    carve_steps = random.randint(0, 1 + size_tier)
    for _ in range(carve_steps):
        ext = _exterior_background(canvas)
        candidates: list[tuple[int, int]] = []
        h, w = canvas.shape
        for r in range(1, h - 1):
            for c in range(1, w - 1):
                if not canvas[r, c]:
                    continue
                if not any(ext[r + dr, c + dc] for dr, dc in DIR4):
                    continue
                trial = canvas.copy()
                trial[r, c] = False
                if _is_connected(trial) and _count_holes(trial) == hole_count:
                    candidates.append((r, c))
        if not candidates:
            break
        r, c = random.choice(candidates)
        canvas[r, c] = False

    cropped = _crop_mask(canvas)
    if not _is_connected(cropped):
        return None
    if _count_holes(cropped) != hole_count:
        return None
    return cropped


def _make_object_mask(hole_count: int, size_tier: int) -> np.ndarray | None:
    for _ in range(OBJECT_RETRIES):
        base = _base_object(hole_count, size_tier)
        if base is None:
            continue
        mask = _irregularize(base, hole_count, size_tier)
        if mask is None:
            continue
        if int(mask.sum()) < 6 + hole_count:
            continue
        return mask
    return None


def _choose_hole_counts() -> list[int]:
    n_layers = random.randint(2, 4)
    positive = random.sample([1, 2, 3], n_layers - 1)
    return sorted(positive + [0], reverse=True)


def _group_sizes(hole_counts: list[int]) -> dict[int, int]:
    sizes = {hole_count: 1 for hole_count in hole_counts}
    if random.random() < 0.5:
        dup_groups = 1 if len(hole_counts) <= 2 or random.random() < 0.7 else 2
        for hole_count in random.sample(hole_counts, dup_groups):
            sizes[hole_count] = 2
    return sizes


def _masks_for_group(hole_count: int, count: int) -> list[np.ndarray] | None:
    if count == 1:
        tier = random.randint(0, 2)
        mask = _make_object_mask(hole_count, tier)
        return None if mask is None else [mask]

    tiers = [0, 2]
    random.shuffle(tiers)
    masks: list[np.ndarray] = []
    used_areas: set[int] = set()
    for tier in tiers[:count]:
        placed = False
        for _ in range(OBJECT_RETRIES):
            mask = _make_object_mask(hole_count, tier)
            if mask is None:
                continue
            area = int(mask.sum())
            if area in used_areas:
                continue
            used_areas.add(area)
            masks.append(mask)
            placed = True
            break
        if not placed:
            return None
    return masks


def _place_objects(items: list[dict[str, object]]) -> np.ndarray | None:
    bbox_load = sum((mask.shape[0] + 2) * (mask.shape[1] + 2) for mask in (item["mask"] for item in items))
    max_h = max(int(item["mask"].shape[0]) for item in items)
    max_w = max(int(item["mask"].shape[1]) for item in items)
    side_hint = int(np.ceil(np.sqrt(bbox_load)))
    h_lo = max(max_h + 2, side_hint - 2, 14)
    w_lo = max(max_w + 2, side_hint - 2, 14)
    h_hi = min(30, h_lo + 7)
    w_hi = min(30, w_lo + 7)
    if h_lo > h_hi or w_lo > w_hi:
        return None

    ordered = list(items)
    random.shuffle(ordered)
    ordered.sort(key=lambda item: int(np.sum(item["mask"])), reverse=True)

    for _ in range(PLACEMENT_RETRIES):
        h = random.randint(h_lo, h_hi)
        w = random.randint(w_lo, w_hi)
        grid = np.full((h, w), BG, dtype=int)
        occupied = np.zeros((h, w), dtype=bool)
        ok = True
        for item in ordered:
            mask = item["mask"]
            color = int(item["color"])
            mh, mw = mask.shape
            positions: list[tuple[int, int]] = []
            for r in range(h - mh + 1):
                for c in range(w - mw + 1):
                    r0 = max(0, r - 1)
                    c0 = max(0, c - 1)
                    r1 = min(h, r + mh + 1)
                    c1 = min(w, c + mw + 1)
                    if occupied[r0:r1, c0:c1].any():
                        continue
                    positions.append((r, c))
            if not positions:
                ok = False
                break
            r, c = random.choice(positions)
            grid[r : r + mh, c : c + mw][mask] = color
            occupied[max(0, r - 1) : min(h, r + mh + 1), max(0, c - 1) : min(w, c + mw + 1)] = True
        if ok:
            return grid
    return None


def _expected_output(items: list[dict[str, object]]) -> np.ndarray:
    best: dict[int, tuple[int, int]] = {}
    for item in items:
        holes = int(item["holes"])
        area = int(np.sum(item["mask"]))
        color = int(item["color"])
        if holes not in best or area > best[holes][0]:
            best[holes] = (area, color)
    colors = [best[holes][1] for holes in sorted(best, reverse=True)]
    return _render_nested(colors)


def _gen_attempt() -> np.ndarray | None:
    hole_counts = _choose_hole_counts()
    group_sizes = _group_sizes(hole_counts)

    group_masks: dict[int, list[np.ndarray]] = {}
    for hole_count in hole_counts:
        masks = _masks_for_group(hole_count, group_sizes[hole_count])
        if masks is None:
            return None
        group_masks[hole_count] = masks

    total_objects = sum(len(masks) for masks in group_masks.values())
    colors = random.sample(PALETTE, total_objects)
    random.shuffle(colors)

    items: list[dict[str, object]] = []
    color_idx = 0
    for hole_count in hole_counts:
        masks = list(group_masks[hole_count])
        random.shuffle(masks)
        for mask in masks:
            items.append({"holes": hole_count, "mask": mask, "color": colors[color_idx]})
            color_idx += 1

    grid = _place_objects(items)
    if grid is None:
        return None

    expected = _expected_output(items)
    if expected.shape[0] > 30 or expected.shape[1] > 30:
        return None
    if not np.array_equal(main(grid), expected):
        return None
    return grid


def generate_input() -> np.ndarray:
    for _ in range(GEN_RETRIES):
        grid = _gen_attempt()
        if grid is not None:
            return grid
    raise RuntimeError("failed to generate valid input")
