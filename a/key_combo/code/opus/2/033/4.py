from __future__ import annotations

from collections import deque
import random

import numpy as np


BG = 0
WALL = 9


def _neighbors4(r: int, c: int):
    yield r - 1, c
    yield r + 1, c
    yield r, c - 1
    yield r, c + 1


def _bbox_coords(coords: np.ndarray) -> tuple[int, int, int, int]:
    r0, c0 = coords.min(axis=0)
    r1, c1 = coords.max(axis=0)
    return int(r0), int(c0), int(r1), int(c1)


def _components(mask: np.ndarray) -> list[np.ndarray]:
    mask = np.asarray(mask, dtype=bool)
    h, w = mask.shape
    seen = np.zeros_like(mask, dtype=bool)
    comps: list[np.ndarray] = []
    for r in range(h):
        for c in range(w):
            if not mask[r, c] or seen[r, c]:
                continue
            q = deque([(r, c)])
            seen[r, c] = True
            pts: list[tuple[int, int]] = []
            while q:
                rr, cc = q.popleft()
                pts.append((rr, cc))
                for nr, nc in _neighbors4(rr, cc):
                    if 0 <= nr < h and 0 <= nc < w and mask[nr, nc] and not seen[nr, nc]:
                        seen[nr, nc] = True
                        q.append((nr, nc))
            comps.append(np.array(pts, dtype=int))
    return comps


def _count_enclosed_bg(subgrid: np.ndarray) -> int:
    zeros = np.asarray(subgrid == BG, dtype=bool)
    h, w = zeros.shape
    outside = np.zeros_like(zeros, dtype=bool)
    q: deque[tuple[int, int]] = deque()

    for r in range(h):
        for c in (0, w - 1):
            if zeros[r, c] and not outside[r, c]:
                outside[r, c] = True
                q.append((r, c))
    for c in range(w):
        for r in (0, h - 1):
            if zeros[r, c] and not outside[r, c]:
                outside[r, c] = True
                q.append((r, c))

    while q:
        r, c = q.popleft()
        for nr, nc in _neighbors4(r, c):
            if 0 <= nr < h and 0 <= nc < w and zeros[nr, nc] and not outside[nr, nc]:
                outside[nr, nc] = True
                q.append((nr, nc))

    return int(np.count_nonzero(zeros & ~outside))


def _hole_area_in_bbox(grid: np.ndarray, bbox: tuple[int, int, int, int]) -> int:
    r0, c0, r1, c1 = bbox
    return _count_enclosed_bg(grid[r0 : r1 + 1, c0 : c1 + 1])


def _walk_ray(
    grid: np.ndarray, start_r: int, start_c: int, dr: int, dc: int
) -> tuple[list[tuple[int, int]], str]:
    h, w = grid.shape
    r, c = int(start_r), int(start_c)
    cells: list[tuple[int, int]] = []
    while 0 <= r < h and 0 <= c < w:
        if r == h - 1:
            return cells, "edge"
        value = int(grid[r, c])
        if value == WALL:
            return cells, "wall"
        if value != BG:
            return cells, "shape"
        cells.append((r, c))
        r += dr
        c += dc
    return cells, "edge"


def _ray_segments_for_bbox(
    grid: np.ndarray, bbox: tuple[int, int, int, int]
) -> list[tuple[list[tuple[int, int]], str]]:
    r0, c0, r1, c1 = bbox
    rays: list[tuple[list[tuple[int, int]], str]] = []
    for r in range(r0, r1 + 1):
        rays.append(_walk_ray(grid, r, c0 - 1, 0, -1))
        rays.append(_walk_ray(grid, r, c1 + 1, 0, 1))
    for c in range(c0, c1 + 1):
        rays.append(_walk_ray(grid, r0 - 1, c, -1, 0))
        rays.append(_walk_ray(grid, r1 + 1, c, 1, 0))
    return rays


def _extract_scene(grid: np.ndarray) -> dict:
    g = np.asarray(grid, dtype=int)
    assert g.ndim == 2, "expected a 2D grid"
    assert g.shape[0] >= 4 and g.shape[1] >= 4, "grid is too small"

    palette = [int(v) for v in g[-1] if int(v) != BG]
    assert palette, "expected a palette on the bottom row"
    assert len(palette) == len(set(palette)), "palette colors must be distinct"

    work = g[:-1]
    assert np.any(work == WALL), "expected a maroon wall above the palette row"
    shape_colors = sorted(set(map(int, work.ravel())) - {BG, WALL})

    shapes = []
    for color in shape_colors:
        for coords in _components(work == color):
            bbox = _bbox_coords(coords)
            shapes.append(
                {
                    "color": int(color),
                    "bbox": bbox,
                    "hole_area": _hole_area_in_bbox(work, bbox),
                }
            )

    assert shapes, "expected at least one closed shape"
    assert len(shapes) == len(palette), "shape count must match palette length"
    shapes.sort(key=lambda item: (item["hole_area"], item["bbox"][0], item["bbox"][1], item["color"]))
    return {"palette": palette, "shapes": shapes}


def _apply_rays(grid: np.ndarray, scene: dict, *, return_meta: bool = False):
    g = np.asarray(grid, dtype=int)
    out = g.copy()
    coverage = np.zeros(g.shape, dtype=int)
    stop_types: set[str] = set()
    shape_paints = []

    for rank, shape in enumerate(scene["shapes"]):
        paint_color = int(scene["palette"][rank])
        covered: set[tuple[int, int]] = set()
        local_stops: set[str] = set()
        for cells, stopper in _ray_segments_for_bbox(g, shape["bbox"]):
            if cells:
                covered.update(cells)
                local_stops.add(stopper)
        for r, c in covered:
            out[r, c] = paint_color
            coverage[r, c] += 1
        stop_types.update(local_stops)
        shape_paints.append(
            {
                "paint_color": paint_color,
                "covered": covered,
                "hole_area": int(shape["hole_area"]),
                "stop_types": local_stops,
            }
        )

    out[-1, :] = g[-1, :]
    if not return_meta:
        return out

    visible_counts = {color: int(np.count_nonzero(out[:-1] == color)) for color in scene["palette"]}
    for stat in shape_paints:
        stat["cell_count"] = len(stat["covered"])
        stat["visible_count"] = visible_counts[stat["paint_color"]]

    return out, {
        "shape_paints": shape_paints,
        "stop_types": stop_types,
        "painted_count": int(np.count_nonzero(coverage > 0)),
        "overlap_count": int(np.count_nonzero(coverage > 1)),
    }


def main(grid: np.ndarray) -> np.ndarray:
    """
    Rank closed shapes by enclosed black area, then color their cross-projection rays with the bottom palette.

    Input structure:
    - Black is the background and the bottom row is a palette row with one distinct color per shape.
    - Above the palette are 3-4 separated monochrome closed shapes with different enclosed black areas.
    - A maroon solid rectangle acts as a wall; shapes and the wall are disjoint.

    Rule:
    - For each shape, count the black cells inside its bounding box that are 4-connectedly enclosed away from the box border.
    - Sort the shapes from smallest enclosed area to largest and map rank k to the k-th non-black palette color from left to right.
    - From every bbox row extend left/right rays, and from every bbox column extend up/down rays, stopping before the first wall cell, another shape cell, or the palette/border edge.
    - Paint the traversed background cells in rank order so larger enclosed-area ranks overwrite smaller ones; keep the original shapes, wall, and palette row unchanged.

    Key insight:
    - The ray geometry comes from each shape's bounding box, but the ray color comes from the shape's enclosed-hole area rank rather than from the shape's own color.
    """
    g = np.asarray(grid, dtype=int)
    scene = _extract_scene(g)
    return _apply_rays(g, scene, return_meta=False)


# --------------- generator ---------------

GRID_HEIGHT_RANGE = (20, 26)
GRID_WIDTH_RANGE = (20, 26)
OBJECT_COUNT_OPTIONS = (3, 4)
GEN_RETRIES = 500
SHAPE_COLOR_POOL = tuple(range(1, 9))
WALL_SPECS = ((2, 2), (2, 3), (3, 2), (3, 3))
SHAPE_BBOXES = ((4, 4), (4, 5), (5, 4), (5, 5), (5, 6), (6, 5))


def _make_shape_mask(h: int, w: int, vertical: bool, horizontal: bool) -> np.ndarray:
    mask = np.zeros((h, w), dtype=bool)
    mask[0, :] = True
    mask[-1, :] = True
    mask[:, 0] = True
    mask[:, -1] = True
    if vertical:
        mask[1:-1, w // 2] = True
    if horizontal:
        mask[h // 2, 1:-1] = True
    return mask


def _build_shape_library() -> list[dict]:
    specs = []
    for h, w in SHAPE_BBOXES:
        for vertical, horizontal in ((False, False), (True, False), (False, True), (True, True)):
            mask = _make_shape_mask(h, w, vertical, horizontal)
            hole_area = _count_enclosed_bg(mask.astype(int))
            if hole_area <= 0:
                continue
            specs.append(
                {
                    "h": h,
                    "w": w,
                    "vertical": vertical,
                    "horizontal": horizontal,
                    "mask": mask,
                    "hole_area": hole_area,
                }
            )
    return specs


SHAPE_LIBRARY = _build_shape_library()
SPECS_BY_BBOX: dict[tuple[int, int], list[dict]] = {}
for _spec in SHAPE_LIBRARY:
    SPECS_BY_BBOX.setdefault((_spec["h"], _spec["w"]), []).append(_spec)


def _sample_shape_specs(n_shapes: int) -> list[dict]:
    chosen: list[dict] = []
    used_areas: set[int] = set()

    if random.random() < 0.8:
        candidate_groups = [
            group for group in SPECS_BY_BBOX.values() if len({int(spec["hole_area"]) for spec in group}) >= 2
        ]
        group = random.choice(candidate_groups)
        by_area: dict[int, list[dict]] = {}
        for spec in group:
            by_area.setdefault(int(spec["hole_area"]), []).append(spec)
        area_a, area_b = random.sample(sorted(by_area), 2)
        chosen.append(random.choice(by_area[area_a]))
        chosen.append(random.choice(by_area[area_b]))
        used_areas.update((area_a, area_b))

    while len(chosen) < n_shapes:
        pool = [spec for spec in SHAPE_LIBRARY if int(spec["hole_area"]) not in used_areas]
        spec = random.choice(pool)
        chosen.append(spec)
        used_areas.add(int(spec["hole_area"]))

    random.shuffle(chosen)
    return chosen


def _can_place_bbox(used: np.ndarray, top: int, left: int, h: int, w: int, pad: int = 1) -> bool:
    r0 = max(0, top - pad)
    c0 = max(0, left - pad)
    r1 = min(used.shape[0], top + h + pad)
    c1 = min(used.shape[1], left + w + pad)
    return not bool(used[r0:r1, c0:c1].any())


def _mark_bbox(used: np.ndarray, top: int, left: int, h: int, w: int, pad: int = 1) -> None:
    r0 = max(0, top - pad)
    c0 = max(0, left - pad)
    r1 = min(used.shape[0], top + h + pad)
    c1 = min(used.shape[1], left + w + pad)
    used[r0:r1, c0:c1] = True


def _random_bbox_position(used: np.ndarray, h: int, w: int, tries: int = 150) -> tuple[int, int] | None:
    grid_h_plus_palette, grid_w = used.shape[0] + 1, used.shape[1]
    top_min, top_max = 1, grid_h_plus_palette - h - 2
    left_min, left_max = 1, grid_w - w - 2
    if top_min > top_max or left_min > left_max:
        return None
    for _ in range(tries):
        top = random.randint(top_min, top_max)
        left = random.randint(left_min, left_max)
        if _can_place_bbox(used, top, left, h, w):
            return top, left
    return None


def _place_aligned_pair(
    used: np.ndarray, spec_a: dict, spec_b: dict, orientation: str
) -> list[tuple[int, int]] | None:
    h1, w1 = int(spec_a["h"]), int(spec_a["w"])
    h2, w2 = int(spec_b["h"]), int(spec_b["w"])
    grid_h_plus_palette, grid_w = used.shape[0] + 1, used.shape[1]

    for _ in range(120):
        gap = random.randint(2, 4)
        if orientation == "horizontal":
            top_max = grid_h_plus_palette - max(h1, h2) - 2
            left_max = grid_w - (w1 + gap + w2) - 1
            if top_max < 1 or left_max < 1:
                return None
            top = random.randint(1, top_max)
            left = random.randint(1, left_max)
            pos_a = (top, left)
            pos_b = (top, left + w1 + gap)
        else:
            top_max = grid_h_plus_palette - (h1 + gap + h2) - 1
            left_max = grid_w - max(w1, w2) - 2
            if top_max < 1 or left_max < 1:
                return None
            top = random.randint(1, top_max)
            left = random.randint(1, left_max)
            pos_a = (top, left)
            pos_b = (top + h1 + gap, left)

        if _can_place_bbox(used, pos_a[0], pos_a[1], h1, w1) and _can_place_bbox(
            used, pos_b[0], pos_b[1], h2, w2
        ):
            _mark_bbox(used, pos_a[0], pos_a[1], h1, w1)
            _mark_bbox(used, pos_b[0], pos_b[1], h2, w2)
            return [pos_a, pos_b]
    return None


def _place_wall(used: np.ndarray, shapes: list[dict]) -> tuple[int, int, int, int] | None:
    grid_h_plus_palette, grid_w = used.shape[0] + 1, used.shape[1]
    directions = ("left", "right", "up", "down")

    for _ in range(160):
        wall_h, wall_w = random.choice(WALL_SPECS)
        target = random.choice(shapes)
        direction = random.choice(directions)

        if direction in ("left", "right"):
            top_lo = max(1, target["top"] - wall_h + 1)
            top_hi = min(grid_h_plus_palette - wall_h - 2, target["top"] + target["h"] - 1)
            if top_lo > top_hi:
                continue
            top = random.randint(top_lo, top_hi)
            if direction == "right":
                left_lo = target["left"] + target["w"] + 2
                left_hi = grid_w - wall_w - 2
            else:
                left_lo = 1
                left_hi = target["left"] - wall_w - 2
            if left_lo > left_hi:
                continue
            left = random.randint(left_lo, left_hi)
        else:
            left_lo = max(1, target["left"] - wall_w + 1)
            left_hi = min(grid_w - wall_w - 2, target["left"] + target["w"] - 1)
            if left_lo > left_hi:
                continue
            left = random.randint(left_lo, left_hi)
            if direction == "down":
                top_lo = target["top"] + target["h"] + 2
                top_hi = grid_h_plus_palette - wall_h - 2
            else:
                top_lo = 1
                top_hi = target["top"] - wall_h - 2
            if top_lo > top_hi:
                continue
            top = random.randint(top_lo, top_hi)

        if _can_place_bbox(used, top, left, wall_h, wall_w):
            _mark_bbox(used, top, left, wall_h, wall_w)
            return top, left, wall_h, wall_w

    for wall_h, wall_w in random.sample(list(WALL_SPECS), len(WALL_SPECS)):
        pos = _random_bbox_position(used, wall_h, wall_w, tries=80)
        if pos is not None:
            top, left = pos
            _mark_bbox(used, top, left, wall_h, wall_w)
            return top, left, wall_h, wall_w
    return None


def _palette_columns(width: int, n_colors: int) -> list[int]:
    slots = list(range(1, width - 1))
    for _ in range(80):
        cols = sorted(random.sample(slots, n_colors))
        if all(b - a >= 2 for a, b in zip(cols, cols[1:])):
            return cols
    return sorted(random.sample(slots, n_colors))


def _render_scene(
    height: int,
    width: int,
    shapes: list[dict],
    wall_bbox: tuple[int, int, int, int],
    palette_colors: list[int],
) -> np.ndarray:
    grid = np.zeros((height, width), dtype=int)

    for shape in shapes:
        mask = np.asarray(shape["spec"]["mask"], dtype=bool)
        patch = np.zeros(mask.shape, dtype=int)
        patch[mask] = int(shape["color"])
        top, left = int(shape["top"]), int(shape["left"])
        h, w = patch.shape
        grid[top : top + h, left : left + w] = patch

    wall_top, wall_left, wall_h, wall_w = wall_bbox
    grid[wall_top : wall_top + wall_h, wall_left : wall_left + wall_w] = WALL

    for c, color in zip(_palette_columns(width, len(palette_colors)), palette_colors):
        grid[-1, c] = int(color)
    return grid


def _valid_generated_grid(grid: np.ndarray, require_shape_stop: bool) -> bool:
    try:
        scene = _extract_scene(grid)
        out, meta = _apply_rays(grid, scene, return_meta=True)
    except AssertionError:
        return False

    if len(scene["shapes"]) not in OBJECT_COUNT_OPTIONS:
        return False
    if len({int(shape["hole_area"]) for shape in scene["shapes"]}) != len(scene["shapes"]):
        return False
    if not np.any(out[:-1] != grid[:-1]):
        return False

    painted = int(meta["painted_count"])
    overlap = int(meta["overlap_count"])
    if painted < 20:
        return False
    if painted > int(0.75 * (grid.shape[0] - 1) * grid.shape[1]):
        return False
    if overlap > max(20, int(0.45 * painted)):
        return False

    stop_types = set(meta["stop_types"])
    if "wall" not in stop_types or len(stop_types) < 2:
        return False
    if require_shape_stop and "shape" not in stop_types:
        return False

    for color in scene["palette"]:
        if int(np.count_nonzero(out[:-1] == color)) < 3:
            return False
    return True


def _gen_attempt() -> np.ndarray | None:
    n_shapes = 3 if random.random() < 0.65 else 4
    min_side = 20 if n_shapes == 3 else 22
    height = random.randint(min_side, GRID_HEIGHT_RANGE[1])
    width = random.randint(min_side, GRID_WIDTH_RANGE[1])

    specs = _sample_shape_specs(n_shapes)
    colors = random.sample(list(SHAPE_COLOR_POOL), 2 * n_shapes)
    shape_colors = colors[:n_shapes]
    palette_colors = colors[n_shapes:]

    used = np.zeros((height - 1, width), dtype=bool)
    shapes: list[dict] = []

    require_shape_stop = random.random() < 0.75
    if require_shape_stop:
        aligned = _place_aligned_pair(used, specs[0], specs[1], random.choice(("horizontal", "vertical")))
        if aligned is None:
            return None
        for idx, (top, left) in enumerate(aligned):
            spec = specs[idx]
            shapes.append(
                {
                    "spec": spec,
                    "color": shape_colors[idx],
                    "top": top,
                    "left": left,
                    "h": int(spec["h"]),
                    "w": int(spec["w"]),
                }
            )
        start_index = 2
    else:
        start_index = 0

    for idx in range(start_index, n_shapes):
        spec = specs[idx]
        pos = _random_bbox_position(used, int(spec["h"]), int(spec["w"]))
        if pos is None:
            return None
        _mark_bbox(used, pos[0], pos[1], int(spec["h"]), int(spec["w"]))
        shapes.append(
            {
                "spec": spec,
                "color": shape_colors[idx],
                "top": pos[0],
                "left": pos[1],
                "h": int(spec["h"]),
                "w": int(spec["w"]),
            }
        )

    wall_bbox = _place_wall(used, shapes)
    if wall_bbox is None:
        return None

    grid = _render_scene(height, width, shapes, wall_bbox, palette_colors)
    if not _valid_generated_grid(grid, require_shape_stop=require_shape_stop):
        return None
    return grid


def generate_input() -> np.ndarray:
    for _ in range(GEN_RETRIES):
        grid = _gen_attempt()
        if grid is not None:
            return grid
    raise RuntimeError("failed to generate a valid input for puzzle 4")
