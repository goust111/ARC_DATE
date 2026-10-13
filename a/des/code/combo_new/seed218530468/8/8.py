from __future__ import annotations

import random

import numpy as np


BG = 0
SEP = 5
OBJECT_COLORS = (1, 2, 3, 4, 6, 7, 8, 9)
GRID_MAX = 30
GEN_ATTEMPTS = 400
SHAPE_ATTEMPTS = 120
PLACE_ATTEMPTS = 120


def _neighbors4(r: int, c: int) -> tuple[tuple[int, int], ...]:
    return ((r - 1, c), (r + 1, c), (r, c - 1), (r, c + 1))


def _components(mask: np.ndarray) -> list[list[tuple[int, int]]]:
    h, w = mask.shape
    seen = np.zeros((h, w), dtype=bool)
    comps: list[list[tuple[int, int]]] = []
    for r in range(h):
        for c in range(w):
            if not bool(mask[r, c]) or bool(seen[r, c]):
                continue
            stack = [(r, c)]
            seen[r, c] = True
            comp: list[tuple[int, int]] = []
            while stack:
                rr, cc = stack.pop()
                comp.append((rr, cc))
                for nr, nc in _neighbors4(rr, cc):
                    if 0 <= nr < h and 0 <= nc < w and bool(mask[nr, nc]) and not bool(seen[nr, nc]):
                        seen[nr, nc] = True
                        stack.append((nr, nc))
            comps.append(comp)
    return comps


def _crop_mask(mask: np.ndarray) -> np.ndarray:
    coords = np.argwhere(mask)
    assert coords.size > 0
    r0, c0 = coords.min(axis=0)
    r1, c1 = coords.max(axis=0)
    return mask[int(r0) : int(r1) + 1, int(c0) : int(c1) + 1].copy()


def _count_holes(mask: np.ndarray) -> int:
    bg_components = _components(~mask)
    h, w = mask.shape
    holes = 0
    for comp in bg_components:
        if any(r == 0 or c == 0 or r == h - 1 or c == w - 1 for r, c in comp):
            continue
        holes += 1
    return holes


def _majority_color(grid: np.ndarray) -> int:
    vals, cnts = np.unique(grid, return_counts=True)
    return int(vals[np.argmax(cnts)])


def _extract_objects(grid: np.ndarray, bg: int) -> list[dict]:
    g = np.asarray(grid, dtype=int)
    objects: list[dict] = []
    for comp in _components(g != bg):
        rows = [r for r, _c in comp]
        cols = [c for _r, c in comp]
        r0, r1 = min(rows), max(rows)
        c0, c1 = min(cols), max(cols)
        sprite = np.full((r1 - r0 + 1, c1 - c0 + 1), bg, dtype=int)
        for r, c in comp:
            sprite[r - r0, c - c0] = int(g[r, c])
        mask = sprite != bg
        objects.append(
            {
                "sprite": sprite,
                "mask": mask,
                "holes": _count_holes(mask),
                "area": int(mask.sum()),
                "origin": (r0, c0),
            }
        )
    return objects


def main(grid: np.ndarray) -> np.ndarray:
    """
    Group disconnected colored objects by how many enclosed holes they contain, then restack each group in a separate horizontal band.

    Input structure:
    - The background is black.
    - Several disconnected colored objects are scattered across the grid.
    - Each object is a single 4-connected shape, and some objects contain 0, 1, or 2 enclosed black holes.
    - Objects are separated from one another and do not share bounding boxes.

    Rule:
    - For each object, count its enclosed black holes by looking for background regions fully surrounded by the object.
    - Partition the objects into the hole-count groups H=0, H=1, and H=2.
    - Emit the non-empty groups from top to bottom in that order; inside each group, place the original object sprites left-to-right sorted by increasing area, with one black column between neighbors.
    - Each group gets a band as tall as its tallest object, objects are bottom-aligned within that band, and consecutive non-empty groups are separated by a full-width grey row.

    Key insight:
    - Color and exact silhouette are distractions: the stable feature that controls the layout is the number of enclosed holes inside each disconnected object.
    """
    g = np.asarray(grid, dtype=int)
    assert g.ndim == 2
    bg = BG if np.any(g == BG) else _majority_color(g)
    objects = _extract_objects(g, bg)
    assert objects

    groups = {0: [], 1: [], 2: []}
    for obj in objects:
        assert 0 <= int(obj["holes"]) <= 2
        groups[int(obj["holes"])].append(obj)

    active_groups = [hole_count for hole_count in (0, 1, 2) if groups[hole_count]]
    assert active_groups

    band_heights: dict[int, int] = {}
    band_widths: dict[int, int] = {}
    for hole_count in active_groups:
        groups[hole_count].sort(
            key=lambda obj: (
                int(obj["area"]),
                int(obj["sprite"].shape[0]),
                int(obj["sprite"].shape[1]),
                int(obj["origin"][0]),
                int(obj["origin"][1]),
            )
        )
        band_heights[hole_count] = max(int(obj["sprite"].shape[0]) for obj in groups[hole_count])
        band_widths[hole_count] = sum(int(obj["sprite"].shape[1]) for obj in groups[hole_count]) + len(
            groups[hole_count]
        ) - 1

    out_h = sum(band_heights.values()) + len(active_groups) - 1
    out_w = max(band_widths.values())
    out = np.full((out_h, out_w), bg, dtype=int)

    row0 = 0
    for idx, hole_count in enumerate(active_groups):
        band_h = band_heights[hole_count]
        col0 = 0
        for obj in groups[hole_count]:
            sprite = obj["sprite"]
            mask = obj["mask"]
            obj_h, obj_w = sprite.shape
            top = row0 + band_h - obj_h
            patch = out[top : top + obj_h, col0 : col0 + obj_w]
            patch[mask] = sprite[mask]
            col0 += obj_w + 1
        row0 += band_h
        if idx != len(active_groups) - 1:
            out[row0, :] = SEP
            row0 += 1

    return out


def _random_orient(mask: np.ndarray) -> np.ndarray:
    out = mask.copy()
    if random.random() < 0.5:
        out = np.flipud(out)
    if random.random() < 0.5:
        out = np.fliplr(out)
    if random.random() < 0.35:
        out = out.T
    return _crop_mask(out)


def _attach_tabs(mask: np.ndarray, max_tabs: int) -> np.ndarray:
    canvas = np.pad(mask.astype(bool), 2, constant_values=False)
    top = 2
    left = 2
    bottom = top + mask.shape[0] - 1
    right = left + mask.shape[1] - 1
    for _ in range(random.randint(0, max_tabs)):
        side = random.choice(("top", "bottom", "left", "right"))
        if side in ("top", "bottom"):
            length = random.randint(1, max(1, mask.shape[1] - 1))
            start = random.randint(left, right - length + 1)
            row = top - 1 if side == "top" else bottom + 1
            canvas[row, start : start + length] = True
        else:
            length = random.randint(1, max(1, mask.shape[0] - 1))
            start = random.randint(top, bottom - length + 1)
            col = left - 1 if side == "left" else right + 1
            canvas[start : start + length, col] = True
    return _crop_mask(canvas)


def _grow_hole_free_polyomino() -> np.ndarray:
    for _ in range(SHAPE_ATTEMPTS):
        h = random.randint(2, 6)
        w = random.randint(2, 6)
        target_area = random.randint(4, min(14, h * w))
        cells = {(random.randrange(h), random.randrange(w))}
        while len(cells) < target_area:
            frontier: list[tuple[int, int]] = []
            for r, c in cells:
                for nr, nc in _neighbors4(r, c):
                    if 0 <= nr < h and 0 <= nc < w and (nr, nc) not in cells:
                        frontier.append((nr, nc))
            if not frontier:
                break
            cells.add(random.choice(frontier))
        mask = np.zeros((h, w), dtype=bool)
        for r, c in cells:
            mask[r, c] = True
        mask = _crop_mask(mask)
        if _count_holes(mask) == 0:
            return _random_orient(mask)
    raise RuntimeError("failed to sample a hole-free object")


def _make_hole0_mask() -> np.ndarray:
    if random.random() < 0.3:
        mask = np.ones((random.randint(2, 5), random.randint(2, 6)), dtype=bool)
        return _random_orient(mask)
    return _grow_hole_free_polyomino()


def _make_hole1_mask() -> np.ndarray:
    for _ in range(SHAPE_ATTEMPTS):
        thickness = random.choice((1, 1, 2))
        outer_h = random.randint(4 + 2 * (thickness - 1), 7 + 2 * (thickness - 1))
        outer_w = random.randint(4 + 2 * (thickness - 1), 8 + 2 * (thickness - 1))
        inner_h = outer_h - 2 * thickness
        inner_w = outer_w - 2 * thickness
        if inner_h < 1 or inner_w < 1:
            continue
        mask = np.ones((outer_h, outer_w), dtype=bool)
        mask[thickness : outer_h - thickness, thickness : outer_w - thickness] = False
        mask = _attach_tabs(mask, max_tabs=2)
        mask = _random_orient(mask)
        if _count_holes(mask) == 1 and max(mask.shape) <= 9:
            return mask
    raise RuntimeError("failed to sample a one-hole object")


def _make_hole2_mask() -> np.ndarray:
    for _ in range(SHAPE_ATTEMPTS):
        thickness = random.choice((1, 1, 2))
        vertical_split = random.random() < 0.5
        if vertical_split:
            outer_h = random.randint(5 + 2 * (thickness - 1), 8 + 2 * (thickness - 1))
            outer_w = random.randint(6 + 2 * (thickness - 1), 10 + 2 * (thickness - 1))
        else:
            outer_h = random.randint(6 + 2 * (thickness - 1), 10 + 2 * (thickness - 1))
            outer_w = random.randint(5 + 2 * (thickness - 1), 8 + 2 * (thickness - 1))

        inner_h = outer_h - 2 * thickness
        inner_w = outer_w - 2 * thickness
        if vertical_split and inner_w < 3:
            continue
        if (not vertical_split) and inner_h < 3:
            continue

        mask = np.ones((outer_h, outer_w), dtype=bool)
        mask[thickness : outer_h - thickness, thickness : outer_w - thickness] = False
        if vertical_split:
            split_c = random.randint(thickness + 1, outer_w - thickness - 2)
            mask[thickness : outer_h - thickness, split_c] = True
        else:
            split_r = random.randint(thickness + 1, outer_h - thickness - 2)
            mask[split_r, thickness : outer_w - thickness] = True
        mask = _attach_tabs(mask, max_tabs=2)
        mask = _random_orient(mask)
        if _count_holes(mask) == 2 and max(mask.shape) <= 10:
            return mask
    raise RuntimeError("failed to sample a two-hole object")


def _make_mask_with_holes(hole_count: int) -> np.ndarray:
    if hole_count == 0:
        return _make_hole0_mask()
    if hole_count == 1:
        return _make_hole1_mask()
    if hole_count == 2:
        return _make_hole2_mask()
    raise ValueError(f"unsupported hole count {hole_count}")


def _sample_group_objects(hole_count: int, count: int) -> list[np.ndarray]:
    objs: list[np.ndarray] = []
    used_areas: set[int] = set()
    for _ in range(SHAPE_ATTEMPTS * 3):
        if len(objs) == count:
            break
        mask = _make_mask_with_holes(hole_count)
        area = int(mask.sum())
        if area in used_areas:
            continue
        objs.append(mask)
        used_areas.add(area)
    if len(objs) != count:
        raise RuntimeError("failed to build a group with distinct areas")
    return objs


def _sample_colors(n: int) -> list[int]:
    colors: list[int] = []
    for _ in range(n):
        if colors and random.random() < 0.35:
            colors.append(random.choice(colors))
        else:
            colors.append(random.choice(OBJECT_COLORS))
    if len(set(colors)) == 1:
        idx = random.randrange(n)
        choices = [color for color in OBJECT_COLORS if color != colors[idx]]
        colors[idx] = random.choice(choices)
    return colors


def _output_size_ok(masks_by_group: dict[int, list[np.ndarray]]) -> bool:
    widths = []
    heights = []
    for hole_count in (0, 1, 2):
        masks = masks_by_group[hole_count]
        widths.append(sum(int(mask.shape[1]) for mask in masks) + len(masks) - 1)
        heights.append(max(int(mask.shape[0]) for mask in masks))
    return max(widths) <= GRID_MAX and sum(heights) + 2 <= GRID_MAX


def _sample_input_shape(masks: list[np.ndarray]) -> tuple[int, int]:
    bbox_area = sum(int(mask.shape[0] * mask.shape[1]) for mask in masks)
    max_h = max(int(mask.shape[0]) for mask in masks)
    max_w = max(int(mask.shape[1]) for mask in masks)
    base = int(np.ceil(np.sqrt(bbox_area * 2.5)))
    h_lo = max(max_h + 2, base - 3, 12)
    w_lo = max(max_w + 2, base - 3, 12)
    h_hi = min(GRID_MAX, max(h_lo, base + 5))
    w_hi = min(GRID_MAX, max(w_lo, base + 5))
    return random.randint(h_lo, h_hi), random.randint(w_lo, w_hi)


def _place_masks_randomly(masks: list[np.ndarray], colors: list[int]) -> np.ndarray | None:
    order = sorted(range(len(masks)), key=lambda i: masks[i].shape[0] * masks[i].shape[1], reverse=True)
    for _ in range(PLACE_ATTEMPTS):
        h, w = _sample_input_shape(masks)
        grid = np.full((h, w), BG, dtype=int)
        blocked = np.zeros((h, w), dtype=bool)
        ok = True
        for idx in order:
            mask = masks[idx]
            mh, mw = mask.shape
            candidates: list[tuple[int, int]] = []
            for r in range(h - mh + 1):
                for c in range(w - mw + 1):
                    r0 = max(0, r - 1)
                    r1 = min(h, r + mh + 1)
                    c0 = max(0, c - 1)
                    c1 = min(w, c + mw + 1)
                    if bool(blocked[r0:r1, c0:c1].any()):
                        continue
                    candidates.append((r, c))
            if not candidates:
                ok = False
                break
            r, c = random.choice(candidates)
            grid[r : r + mh, c : c + mw][mask] = int(colors[idx])
            blocked[max(0, r - 1) : min(h, r + mh + 1), max(0, c - 1) : min(w, c + mw + 1)] = True
        if ok and float(np.mean(grid == BG)) >= 0.45:
            return grid
    return None


def _build_sample() -> np.ndarray | None:
    group_counts = {hole_count: random.choice((1, 2)) for hole_count in (0, 1, 2)}
    masks_by_group = {
        hole_count: _sample_group_objects(hole_count, group_counts[hole_count]) for hole_count in (0, 1, 2)
    }
    if not _output_size_ok(masks_by_group):
        return None

    masks: list[np.ndarray] = []
    for hole_count in random.sample((0, 1, 2), 3):
        masks.extend(masks_by_group[hole_count])

    colors = _sample_colors(len(masks))
    grid = _place_masks_randomly(masks, colors)
    if grid is None:
        return None

    objects = _extract_objects(grid, BG)
    if len(objects) != len(masks):
        return None
    if sorted(int(obj["holes"]) for obj in objects) != sorted([0] * group_counts[0] + [1] * group_counts[1] + [2] * group_counts[2]):
        return None

    out = main(grid)
    if max(grid.shape) > GRID_MAX or max(out.shape) > GRID_MAX:
        return None
    if sum(np.all(out == SEP, axis=1)) != 2:
        return None
    return grid


def generate_input() -> np.ndarray:
    for _ in range(GEN_ATTEMPTS):
        grid = _build_sample()
        if grid is not None:
            return grid
    raise RuntimeError(f"failed to generate a valid puzzle after {GEN_ATTEMPTS} attempts")
