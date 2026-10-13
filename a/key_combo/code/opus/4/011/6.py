from __future__ import annotations

from collections import Counter, deque
import random

import numpy as np


def neighbors4(r: int, c: int):
    yield r - 1, c
    yield r + 1, c
    yield r, c - 1
    yield r, c + 1


def bbox_coords(coords: list[tuple[int, int]]) -> tuple[int, int, int, int]:
    rows = [r for r, _ in coords]
    cols = [c for _, c in coords]
    return min(rows), min(cols), max(rows), max(cols)


def crop_bbox(grid: np.ndarray, bbox: tuple[int, int, int, int]) -> np.ndarray:
    r0, c0, r1, c1 = bbox
    return grid[r0 : r1 + 1, c0 : c1 + 1].copy()


def paste(dst: np.ndarray, patch: np.ndarray, r0: int, c0: int, *, transparent: int = 0) -> None:
    h, w = patch.shape
    view = dst[r0 : r0 + h, c0 : c0 + w]
    mask = patch != int(transparent)
    view[mask] = patch[mask]


def component_coords(mask: np.ndarray) -> list[list[tuple[int, int]]]:
    h, w = mask.shape
    seen = np.zeros_like(mask, dtype=bool)
    comps: list[list[tuple[int, int]]] = []
    for r in range(h):
        for c in range(w):
            if not mask[r, c] or seen[r, c]:
                continue
            q = deque([(r, c)])
            seen[r, c] = True
            coords: list[tuple[int, int]] = []
            while q:
                rr, cc = q.popleft()
                coords.append((rr, cc))
                for nr, nc in neighbors4(rr, cc):
                    if 0 <= nr < h and 0 <= nc < w and mask[nr, nc] and not seen[nr, nc]:
                        seen[nr, nc] = True
                        q.append((nr, nc))
            comps.append(coords)
    return comps


def normalize_cells(cells: set[tuple[int, int]]) -> set[tuple[int, int]]:
    r0, c0, _r1, _c1 = bbox_coords(list(cells))
    return {(r - r0, c - c0) for r, c in cells}


def transform_cells(cells: set[tuple[int, int]]) -> set[tuple[int, int]]:
    arr = np.array(sorted(cells), dtype=int)
    k = random.randrange(4)
    for _ in range(k):
        arr = np.stack([arr[:, 1], -arr[:, 0]], axis=1)
    if random.random() < 0.5:
        arr[:, 1] *= -1
    if random.random() < 0.5:
        arr[:, 0] *= -1
    out = {(int(r), int(c)) for r, c in arr}
    return normalize_cells(out)


def cells_to_patch(cells: set[tuple[int, int]], color: int) -> np.ndarray:
    r0, c0, r1, c1 = bbox_coords(list(cells))
    patch = np.zeros((r1 - r0 + 1, c1 - c0 + 1), dtype=int)
    for r, c in cells:
        patch[r - r0, c - c0] = int(color)
    return patch


def hole_free(cells: set[tuple[int, int]]) -> bool:
    r0, c0, r1, c1 = bbox_coords(list(cells))
    h = r1 - r0 + 3
    w = c1 - c0 + 3
    occupied = np.zeros((h, w), dtype=bool)
    for r, c in cells:
        occupied[r - r0 + 1, c - c0 + 1] = True
    seen = np.zeros_like(occupied, dtype=bool)
    q = deque([(0, 0)])
    seen[0, 0] = True
    while q:
        rr, cc = q.popleft()
        for nr, nc in neighbors4(rr, cc):
            if 0 <= nr < h and 0 <= nc < w and not occupied[nr, nc] and not seen[nr, nc]:
                seen[nr, nc] = True
                q.append((nr, nc))
    return not np.any((~occupied) & (~seen))


def interior_cells(cells: set[tuple[int, int]]) -> list[tuple[int, int]]:
    interior: list[tuple[int, int]] = []
    for r, c in cells:
        if all((nr, nc) in cells for nr, nc in neighbors4(r, c)):
            interior.append((r, c))
    return interior


def enclosed_zero_area(sprite: np.ndarray) -> int:
    h, w = sprite.shape
    seen = np.zeros((h, w), dtype=bool)
    q = deque()
    for r in range(h):
        for c in (0, w - 1):
            if sprite[r, c] == 0 and not seen[r, c]:
                seen[r, c] = True
                q.append((r, c))
    for c in range(w):
        for r in (0, h - 1):
            if sprite[r, c] == 0 and not seen[r, c]:
                seen[r, c] = True
                q.append((r, c))
    while q:
        rr, cc = q.popleft()
        for nr, nc in neighbors4(rr, cc):
            if 0 <= nr < h and 0 <= nc < w and sprite[nr, nc] == 0 and not seen[nr, nc]:
                seen[nr, nc] = True
                q.append((nr, nc))
    return int(np.count_nonzero((sprite == 0) & (~seen)))


def outline_sprite_from_component(
    grid: np.ndarray,
    coords: list[tuple[int, int]],
    marker_color: int,
) -> np.ndarray:
    outline = np.zeros_like(grid)
    occupied = set(coords)
    h, w = grid.shape
    for r, c in coords:
        for nr, nc in neighbors4(r, c):
            if 0 <= nr < h and 0 <= nc < w and grid[nr, nc] == 0:
                outline[r, c] = int(marker_color)
                break
            if (nr, nc) not in occupied and not (0 <= nr < h and 0 <= nc < w):
                continue
    b = bbox_coords([(r, c) for r, c in np.argwhere(outline != 0)])
    return crop_bbox(outline, b)


def analyze_objects(grid: np.ndarray) -> list[dict]:
    g = np.asarray(grid, dtype=int)
    comps = component_coords(g != 0)
    objects: list[dict] = []
    for coords in comps:
        vals = [int(g[r, c]) for r, c in coords]
        counts = Counter(vals)
        assert len(counts) == 2, "each object must contain one body color and one marker color"
        body_color, body_count = max(counts.items(), key=lambda kv: (kv[1], kv[0]))
        marker_color, marker_count = min(counts.items(), key=lambda kv: (kv[1], kv[0]))
        assert body_count > marker_count >= 1
        marker_pixels = [(r, c) for r, c in coords if int(g[r, c]) == int(marker_color)]
        assert len(marker_pixels) == 1, "expected exactly one marker pixel"
        sprite = outline_sprite_from_component(g, coords, int(marker_color))
        inner_area = enclosed_zero_area(sprite)
        objects.append(
            {
                "coords": coords,
                "body_color": int(body_color),
                "marker_color": int(marker_color),
                "marker_pixel": marker_pixels[0],
                "area": int(len(coords)),
                "outline_sprite": sprite,
                "inner_area": int(inner_area),
            }
        )
    return objects


def pack_bottom_aligned(sprites: list[np.ndarray]) -> np.ndarray:
    height = max(sprite.shape[0] for sprite in sprites)
    width = sum(sprite.shape[1] for sprite in sprites) + len(sprites) - 1
    out = np.zeros((height, width), dtype=int)
    c = 0
    for sprite in sprites:
        h, w = sprite.shape
        paste(out, sprite, height - h, c, transparent=0)
        c += w + 1
    return out


def main(grid: np.ndarray) -> np.ndarray:
    """
    Replace each filled marked object with its marker-colored outline, then sort by enclosed area and pack bottom-aligned.

    Input structure:
    - The grid is about 20x20 with black background.
    - There are 3-5 separated 4-connected foreground objects; each object is mostly one body color and contains exactly one different interior marker pixel.
    - The marker pixel lies strictly inside its object, so it is 4-neighbor surrounded by object pixels and each object has no hole.

    Rule:
    - For each non-black connected object, identify the single marker color carried by its interior marker pixel.
    - Erase the filled object and keep only those original object cells that are 4-neighbor adjacent to black background; these surviving boundary cells form the object's outline and are recolored to the marker color.
    - Crop each outline to its bounding box, compute the number of black cells enclosed by the outline inside that crop, sort the cropped outlines by this enclosed area from small to large, and place them left-to-right with one black column gap and all bottom edges aligned.

    Key insight:
    - The marker color chooses the repaint color, but the sorting key is neither original pixel count nor bounding-box area: it is the enclosed hollow area created only after the erase-to-outline transformation.
    """
    objects = analyze_objects(np.asarray(grid, dtype=int))
    assert 3 <= len(objects) <= 5
    inner_areas = [obj["inner_area"] for obj in objects]
    assert len(inner_areas) == len(set(inner_areas)), "outline enclosed areas must be distinct"
    ordered = sorted(objects, key=lambda obj: obj["inner_area"])
    sprites = [obj["outline_sprite"] for obj in ordered]
    return pack_bottom_aligned(sprites)


BASE_RECTS = {
    1: [(3, 3)],
    2: [(3, 4)],
    3: [(3, 5)],
    4: [(4, 4), (3, 6)],
    6: [(4, 5)],
}
INTERIOR_CHOICES = sorted(BASE_RECTS)
BODY_COLORS = [1, 2, 3, 5, 6, 7]
ALL_NONZERO = [1, 2, 3, 4, 5, 6, 7, 8, 9]
GRID_SIZE_RANGE = (18, 24)
GEN_ATTEMPTS = 500


def rectangle_cells(h: int, w: int) -> set[tuple[int, int]]:
    return {(r, c) for r in range(h) for c in range(w)}


def outside_ring_candidates(cells: set[tuple[int, int]]) -> list[tuple[int, int]]:
    candidates = set()
    for r, c in cells:
        for nr, nc in neighbors4(r, c):
            if (nr, nc) not in cells:
                candidates.add((nr, nc))
    return sorted(candidates)


def random_shape(target_inner_area: int) -> set[tuple[int, int]] | None:
    for _ in range(120):
        h, w = random.choice(BASE_RECTS[target_inner_area])
        cells = rectangle_cells(h, w)
        corners = [(0, 0), (0, w - 1), (h - 1, 0), (h - 1, w - 1)]
        removable = [corner for corner in corners if len(cells) - 1 >= 6]
        remove_count = random.choices([0, 1, 2], weights=[2, 5, 3])[0]
        remove_count = min(remove_count, len(removable))
        for corner in random.sample(removable, remove_count):
            cells.remove(corner)

        ring = outside_ring_candidates(cells)
        max_extra = min(20 - len(cells), len(ring), 6)
        if max_extra < 0:
            continue
        if max_extra == 0:
            extra_count = 0
        else:
            extra_min = 0 if remove_count > 0 else 1
            extra_count = random.randint(extra_min, max_extra)
        if extra_count:
            side_groups = {
                "top": [p for p in ring if p[0] < 0],
                "bottom": [p for p in ring if p[0] >= h],
                "left": [p for p in ring if p[1] < 0],
                "right": [p for p in ring if p[1] >= w],
            }
            chosen_sides = random.sample(list(side_groups), random.randint(1, min(3, len(side_groups))))
            preferred = [p for side in chosen_sides for p in side_groups[side]]
            random.shuffle(preferred)
            pool = preferred + [p for p in ring if p not in preferred]
            for p in pool[:extra_count]:
                cells.add(p)

        cells = transform_cells(cells)
        if not (6 <= len(cells) <= 20):
            continue
        if not hole_free(cells):
            continue
        inner = interior_cells(cells)
        if len(inner) != target_inner_area:
            continue
        return cells
    return None


def object_patch(
    cells: set[tuple[int, int]],
    body_color: int,
    marker_color: int,
) -> tuple[np.ndarray, int]:
    inner = interior_cells(cells)
    assert inner
    marker_cell = random.choice(inner)
    patch = cells_to_patch(cells, int(body_color))
    patch[marker_cell] = int(marker_color)
    return patch, int(len(inner))


def candidate_positions(
    canvas: np.ndarray,
    patch: np.ndarray,
    *,
    padding: int,
) -> list[tuple[int, int]]:
    h, w = canvas.shape
    ph, pw = patch.shape
    occupied = canvas != 0
    candidates: list[tuple[int, int]] = []
    for r0 in range(1, h - ph):
        for c0 in range(1, w - pw):
            ok = True
            for rr in range(ph):
                for cc in range(pw):
                    if patch[rr, cc] == 0:
                        continue
                    gr = r0 + rr
                    gc = c0 + cc
                    r_lo = max(0, gr - padding)
                    r_hi = min(h, gr + padding + 1)
                    c_lo = max(0, gc - padding)
                    c_hi = min(w, gc + padding + 1)
                    if np.any(occupied[r_lo:r_hi, c_lo:c_hi]):
                        ok = False
                        break
                if not ok:
                    break
            if ok:
                candidates.append((r0, c0))
    return candidates


def sort_orders_differ(objects: list[dict]) -> bool:
    by_area = [obj["id"] for obj in sorted(objects, key=lambda obj: obj["area"])]
    by_inner = [obj["id"] for obj in sorted(objects, key=lambda obj: obj["inner_area"])]
    return by_area != by_inner


def validate_generated(grid: np.ndarray) -> None:
    objects = analyze_objects(grid)
    assert 3 <= len(objects) <= 5
    inner_areas = [obj["inner_area"] for obj in objects]
    assert len(set(inner_areas)) == len(inner_areas)
    for obj in objects:
        coords = set(obj["coords"])
        assert 6 <= obj["area"] <= 20
        assert hole_free(normalize_cells(coords))
        r, c = obj["marker_pixel"]
        assert all((nr, nc) in coords for nr, nc in neighbors4(r, c))
    tagged = [{"id": idx, **obj} for idx, obj in enumerate(objects)]
    assert sort_orders_differ(tagged), "pixel-area order must differ from enclosed-area order"
    out = main(grid)
    assert out.ndim == 2 and max(out.shape) <= 30
    expected_width = sum(obj["outline_sprite"].shape[1] for obj in sorted(objects, key=lambda obj: obj["inner_area"])) + len(objects) - 1
    assert out.shape[1] == expected_width


def generate_input() -> np.ndarray:
    for _ in range(GEN_ATTEMPTS):
        n_objects = random.randint(3, 5)
        h = random.randint(*GRID_SIZE_RANGE)
        w = random.randint(*GRID_SIZE_RANGE)
        inner_targets = random.sample(INTERIOR_CHOICES, n_objects)
        body_colors = random.sample(BODY_COLORS, n_objects)
        marker_perm = random.sample(ALL_NONZERO, n_objects)
        marker_colors = []
        for body_color, marker_color in zip(body_colors, marker_perm):
            if marker_color == body_color:
                alternatives = [c for c in ALL_NONZERO if c != body_color and c not in marker_colors]
                marker_color = random.choice(alternatives)
            marker_colors.append(marker_color)

        specs = []
        failed = False
        for idx, (inner_target, body_color, marker_color) in enumerate(
            zip(inner_targets, body_colors, marker_colors)
        ):
            cells = random_shape(inner_target)
            if cells is None:
                failed = True
                break
            patch, actual_inner = object_patch(cells, body_color, marker_color)
            specs.append(
                {
                    "id": idx,
                    "patch": patch,
                    "inner_area": actual_inner,
                    "area": int(np.count_nonzero(patch)),
                }
            )
        if failed:
            continue

        if len({spec["inner_area"] for spec in specs}) != len(specs):
            continue
        if not sort_orders_differ(specs):
            continue

        output_width = sum(spec["patch"].shape[1] for spec in specs) + len(specs) - 1
        output_height = max(spec["patch"].shape[0] for spec in specs)
        if output_width > 30 or output_height > 30:
            continue

        canvas = np.zeros((h, w), dtype=int)
        placement_order = sorted(specs, key=lambda spec: max(spec["patch"].shape), reverse=True)
        for spec in placement_order:
            padding = 1 if random.random() < 0.5 else 2
            positions = candidate_positions(canvas, spec["patch"], padding=padding)
            if not positions:
                failed = True
                break
            r0, c0 = random.choice(positions)
            paste(canvas, spec["patch"], r0, c0, transparent=0)
        if failed:
            continue

        try:
            validate_generated(canvas)
        except AssertionError:
            continue
        return canvas
    raise RuntimeError("failed to generate valid input for puzzle 6")
