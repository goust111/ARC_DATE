from __future__ import annotations

from collections import deque
import random

import numpy as np


MAROON = 9
DIR4 = [(-1, 0), (1, 0), (0, -1), (0, 1)]


def _most_common_color(grid: np.ndarray) -> int:
    values, counts = np.unique(grid, return_counts=True)
    return int(values[np.argmax(counts)])


def _components(mask: np.ndarray) -> list[list[tuple[int, int]]]:
    h, w = mask.shape
    seen = np.zeros((h, w), dtype=bool)
    parts: list[list[tuple[int, int]]] = []

    for r in range(h):
        for c in range(w):
            if not mask[r, c] or seen[r, c]:
                continue
            seen[r, c] = True
            queue = deque([(r, c)])
            coords: list[tuple[int, int]] = []
            while queue:
                rr, cc = queue.popleft()
                coords.append((int(rr), int(cc)))
                for dr, dc in DIR4:
                    nr = rr + dr
                    nc = cc + dc
                    if 0 <= nr < h and 0 <= nc < w and mask[nr, nc] and not seen[nr, nc]:
                        seen[nr, nc] = True
                        queue.append((nr, nc))
            parts.append(coords)
    return parts


def _foreground_components(
    grid: np.ndarray, bg: int
) -> tuple[list[dict[str, object]], np.ndarray]:
    h, w = grid.shape
    seen = np.zeros((h, w), dtype=bool)
    comp_index = np.full((h, w), -1, dtype=int)
    objects: list[dict[str, object]] = []

    for r in range(h):
        for c in range(w):
            color = int(grid[r, c])
            if color == bg or seen[r, c]:
                continue
            seen[r, c] = True
            queue = deque([(r, c)])
            coords: list[tuple[int, int]] = []
            while queue:
                rr, cc = queue.popleft()
                coords.append((int(rr), int(cc)))
                comp_index[rr, cc] = len(objects)
                for dr, dc in DIR4:
                    nr = rr + dr
                    nc = cc + dc
                    if (
                        0 <= nr < h
                        and 0 <= nc < w
                        and not seen[nr, nc]
                        and int(grid[nr, nc]) == color
                    ):
                        seen[nr, nc] = True
                        queue.append((nr, nc))
            objects.append({"color": color, "coords": coords})
    return objects, comp_index


def _analyze_objects(
    grid: np.ndarray, bg: int
) -> tuple[list[dict[str, object]], list[int], np.ndarray]:
    h, w = grid.shape
    objects, comp_index = _foreground_components(grid, bg)
    hole_counts = [0 for _ in objects]
    outside_bg = np.zeros((h, w), dtype=bool)
    seen_bg = np.zeros((h, w), dtype=bool)

    for r in range(h):
        for c in range(w):
            if int(grid[r, c]) != bg or seen_bg[r, c]:
                continue
            seen_bg[r, c] = True
            queue = deque([(r, c)])
            region: list[tuple[int, int]] = []
            touches_border = False
            adjacent_objects: set[int] = set()

            while queue:
                rr, cc = queue.popleft()
                region.append((int(rr), int(cc)))
                if rr == 0 or cc == 0 or rr == h - 1 or cc == w - 1:
                    touches_border = True
                for dr, dc in DIR4:
                    nr = rr + dr
                    nc = cc + dc
                    if not (0 <= nr < h and 0 <= nc < w):
                        continue
                    if int(grid[nr, nc]) == bg:
                        if not seen_bg[nr, nc]:
                            seen_bg[nr, nc] = True
                            queue.append((nr, nc))
                    else:
                        comp_id = int(comp_index[nr, nc])
                        if comp_id >= 0:
                            adjacent_objects.add(comp_id)

            if touches_border:
                for rr, cc in region:
                    outside_bg[rr, cc] = True
            elif len(adjacent_objects) == 1:
                hole_counts[next(iter(adjacent_objects))] += 1

    return objects, hole_counts, outside_bg


def _expansion_layers(
    grid: np.ndarray,
    bg: int,
    outside_bg: np.ndarray,
    obj_coords: list[tuple[int, int]],
    layers: int,
) -> list[set[tuple[int, int]]]:
    if layers <= 0:
        return []

    h, w = grid.shape
    frontier: set[tuple[int, int]] = set()
    for r, c in obj_coords:
        for dr, dc in DIR4:
            nr = r + dr
            nc = c + dc
            if (
                0 <= nr < h
                and 0 <= nc < w
                and outside_bg[nr, nc]
                and int(grid[nr, nc]) == bg
            ):
                frontier.add((int(nr), int(nc)))

    visited = set(frontier)
    result: list[set[tuple[int, int]]] = []
    current = frontier

    for _depth in range(1, layers + 1):
        if not current:
            break
        result.append(current)
        next_frontier: set[tuple[int, int]] = set()
        for r, c in current:
            for dr, dc in DIR4:
                nr = r + dr
                nc = c + dc
                cell = (int(nr), int(nc))
                if (
                    0 <= nr < h
                    and 0 <= nc < w
                    and outside_bg[nr, nc]
                    and int(grid[nr, nc]) == bg
                    and cell not in visited
                ):
                    visited.add(cell)
                    next_frontier.add(cell)
        current = next_frontier

    return result


def main(grid: np.ndarray) -> np.ndarray:
    """
    Expand each object's exterior contour by as many rings as the number of enclosed holes.

    Input structure:
    - The background is the dominant color.
    - There are several separated 4-connected monochrome foreground objects.
    - Some objects contain closed background-color holes.
    - The input never uses maroon (9); maroon is reserved for the added rings.

    Rule:
    - Count each object's enclosed background holes.
    - Grow that object's outside contour outward by N rings, where N is the hole count.
    - Ring 1 is maroon, ring 2 is the object's original color, then alternate.
    - Paint only background cells; never overwrite existing foreground objects.
    - Objects with zero holes remain unchanged.
    """
    g = np.asarray(grid, dtype=int)
    bg = _most_common_color(g)
    objects, hole_counts, outside_bg = _analyze_objects(g, bg)

    out = g.copy()
    for obj, hole_count in zip(objects, hole_counts, strict=False):
        if hole_count <= 0:
            continue
        color = int(obj["color"])
        coords = obj["coords"]
        assert isinstance(coords, list)
        layers = _expansion_layers(g, bg, outside_bg, coords, hole_count)
        for depth, ring in enumerate(layers, start=1):
            paint = MAROON if depth % 2 == 1 else color
            for r, c in ring:
                if int(out[r, c]) == bg:
                    out[r, c] = paint
    return out


def _crop_mask(mask: np.ndarray) -> np.ndarray:
    rows = np.where(mask.any(axis=1))[0]
    cols = np.where(mask.any(axis=0))[0]
    if len(rows) == 0 or len(cols) == 0:
        raise ValueError("mask must contain at least one filled cell")
    return mask[rows[0] : rows[-1] + 1, cols[0] : cols[-1] + 1]


def _mask_hole_count(mask: np.ndarray) -> int:
    return sum(1 for comp in _components(~mask) if not any(
        r == 0 or c == 0 or r == mask.shape[0] - 1 or c == mask.shape[1] - 1
        for r, c in comp
    ))


def _mask_connected(mask: np.ndarray) -> bool:
    return len(_components(mask)) == 1


def _attach_arms(mask: np.ndarray) -> np.ndarray:
    pad = 3
    h, w = mask.shape
    canvas = np.zeros((h + 2 * pad, w + 2 * pad), dtype=bool)
    canvas[pad : pad + h, pad : pad + w] = mask

    for _ in range(random.randint(0, 3)):
        rows = np.where(canvas.any(axis=1))[0]
        cols = np.where(canvas.any(axis=0))[0]
        r0, r1 = int(rows[0]), int(rows[-1])
        c0, c1 = int(cols[0]), int(cols[-1])
        side = random.choice(["top", "bottom", "left", "right"])

        if side in ("top", "bottom"):
            span = random.randint(1, max(1, min(3, c1 - c0 + 1)))
            start = random.randint(c0, c1 - span + 1)
            depth = random.randint(1, 2)
            if side == "top" and r0 - depth >= 0:
                canvas[r0 - depth : r0, start : start + span] = True
            if side == "bottom" and r1 + depth < canvas.shape[0]:
                canvas[r1 + 1 : r1 + 1 + depth, start : start + span] = True
        else:
            span = random.randint(1, max(1, min(3, r1 - r0 + 1)))
            start = random.randint(r0, r1 - span + 1)
            depth = random.randint(1, 2)
            if side == "left" and c0 - depth >= 0:
                canvas[start : start + span, c0 - depth : c0] = True
            if side == "right" and c1 + depth < canvas.shape[1]:
                canvas[start : start + span, c1 + 1 : c1 + 1 + depth] = True

    return _crop_mask(canvas)


def _solid_mask() -> np.ndarray:
    for _ in range(80):
        h = random.randint(4, 8)
        w = random.randint(4, 8)
        mask = np.zeros((h, w), dtype=bool)

        seed_h = random.randint(2, min(3, h))
        seed_w = random.randint(2, min(3, w))
        seed_r = random.randint(0, h - seed_h)
        seed_c = random.randint(0, w - seed_w)
        mask[seed_r : seed_r + seed_h, seed_c : seed_c + seed_w] = True

        target_area = random.randint(max(6, (h * w) // 2), min(h * w, (3 * h * w) // 4))
        while int(mask.sum()) < target_area:
            frontier: set[tuple[int, int]] = set()
            for r, c in zip(*np.where(mask), strict=False):
                for dr, dc in DIR4:
                    nr = int(r + dr)
                    nc = int(c + dc)
                    if 0 <= nr < h and 0 <= nc < w and not mask[nr, nc]:
                        frontier.add((nr, nc))
            if not frontier:
                break
            nr, nc = random.choice(tuple(frontier))
            mask[nr, nc] = True

        mask = _attach_arms(_crop_mask(mask))
        if (
            _mask_connected(mask)
            and _mask_hole_count(mask) == 0
            and mask.shape[0] >= 2
            and mask.shape[1] >= 2
        ):
            return mask
    raise RuntimeError("failed to create a solid object")


def _window_mask(n_holes: int) -> np.ndarray:
    assert n_holes >= 1
    for _ in range(80):
        orientation = random.choice(["vertical", "horizontal"])
        if orientation == "vertical":
            chamber_widths = [random.randint(1, 3) for _ in range(n_holes)]
            inner_w = sum(chamber_widths) + (n_holes - 1)
            h = random.randint(4, 8)
            w = inner_w + 2
            mask = np.zeros((h, w), dtype=bool)
            mask[0, :] = True
            mask[-1, :] = True
            mask[:, 0] = True
            mask[:, -1] = True
            x = 1
            for width in chamber_widths[:-1]:
                x += width
                mask[1:-1, x] = True
                x += 1
        else:
            chamber_heights = [random.randint(1, 3) for _ in range(n_holes)]
            inner_h = sum(chamber_heights) + (n_holes - 1)
            h = inner_h + 2
            w = random.randint(4, 8)
            mask = np.zeros((h, w), dtype=bool)
            mask[0, :] = True
            mask[-1, :] = True
            mask[:, 0] = True
            mask[:, -1] = True
            y = 1
            for height in chamber_heights[:-1]:
                y += height
                mask[y, 1:-1] = True
                y += 1

        mask = _attach_arms(mask)
        if _mask_connected(mask) and _mask_hole_count(mask) == n_holes:
            return mask
    raise RuntimeError(f"failed to create an object with {n_holes} holes")


def _rectangles_overlap(a: tuple[int, int, int, int], b: tuple[int, int, int, int]) -> bool:
    ar0, ac0, ar1, ac1 = a
    br0, bc0, br1, bc1 = b
    return not (ar1 < br0 or br1 < ar0 or ac1 < bc0 or bc1 < ac0)


def _place_objects(
    h: int,
    w: int,
    bg: int,
    object_specs: list[dict[str, object]],
) -> np.ndarray | None:
    grid = np.full((h, w), bg, dtype=int)
    reserved_boxes: list[tuple[int, int, int, int]] = []

    ordered = sorted(
        object_specs,
        key=lambda spec: (
            -(spec["mask"].shape[0] + 2 * (int(spec["holes"]) + 1))
            * (spec["mask"].shape[1] + 2 * (int(spec["holes"]) + 1))
        ),
    )

    for spec in ordered:
        mask = spec["mask"]
        color = int(spec["color"])
        holes = int(spec["holes"])
        mh, mw = mask.shape
        pad = holes + 1

        min_top = pad
        max_top = h - mh - pad
        min_left = pad
        max_left = w - mw - pad
        if min_top > max_top or min_left > max_left:
            return None

        placed = False
        for _ in range(120):
            top = random.randint(min_top, max_top)
            left = random.randint(min_left, max_left)
            box = (top - pad, left - pad, top + mh - 1 + pad, left + mw - 1 + pad)
            if any(_rectangles_overlap(box, other) for other in reserved_boxes):
                continue

            patch = grid[top : top + mh, left : left + mw]
            if np.any(patch[mask] != bg):
                continue

            patch[mask] = color
            reserved_boxes.append(box)
            placed = True
            break

        if not placed:
            return None

    return grid


def _object_specs(bg: int, n_objects: int) -> list[dict[str, object]]:
    usable_colors = [c for c in range(10) if c not in (bg, MAROON)]
    colors = random.sample(usable_colors, n_objects)

    hole_counts = [0, random.randint(1, 3)]
    while len(hole_counts) < n_objects:
        hole_counts.append(random.randint(0, 3))
    random.shuffle(hole_counts)

    specs: list[dict[str, object]] = []
    for color, holes in zip(colors, hole_counts, strict=False):
        mask = _solid_mask() if holes == 0 else _window_mask(holes)
        specs.append({"color": color, "holes": holes, "mask": mask})
    return specs


def _validate_generated(grid: np.ndarray) -> bool:
    bg = _most_common_color(grid)
    if bg == MAROON:
        return False
    if np.any(grid == MAROON):
        return False

    objects, hole_counts, outside_bg = _analyze_objects(grid, bg)
    if len(objects) < 2:
        return False
    if not any(count == 0 for count in hole_counts):
        return False
    if not any(count > 0 for count in hole_counts):
        return False

    painted: set[tuple[int, int]] = set()
    for obj, holes in zip(objects, hole_counts, strict=False):
        coords = obj["coords"]
        assert isinstance(coords, list)
        layers = _expansion_layers(grid, bg, outside_bg, coords, holes)
        if len(layers) != holes:
            return False
        for ring in layers:
            for cell in ring:
                if cell in painted:
                    return False
                painted.add(cell)

    out = main(grid)
    return np.any(out != grid)


def generate_input() -> np.ndarray:
    for _ in range(400):
        h = random.randint(18, 30)
        w = random.randint(18, 30)
        bg = random.choice([c for c in range(9) if c != MAROON])
        n_objects = random.randint(2, 4)
        specs = _object_specs(bg, n_objects)
        grid = _place_objects(h, w, bg, specs)
        if grid is None:
            continue
        if _most_common_color(grid) != bg:
            continue
        if _validate_generated(grid):
            return grid
    raise RuntimeError("failed to generate a valid input")
