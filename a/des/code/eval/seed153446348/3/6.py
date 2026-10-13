import numpy as np


BG = 8
MARKER = 9
BASE_PALETTE = np.array([0, 1, 2, 3, 4, 5, 6, 7], dtype=int)
DIRS = ("up", "down", "left", "right")

GEN_GRID_H_RANGE = (16, 25)
GEN_GRID_W_RANGE = (16, 25)
GEN_SHAPE_COUNT_RANGE = (4, 6)
GEN_SHAPE_SIZE_RANGE = (3, 7)
GEN_ATTEMPTS = 400
GEN_SHAPE_ATTEMPTS = 200
GEN_PLACEMENT_ATTEMPTS = 250


def _neighbors4(r: int, c: int) -> tuple[tuple[int, int], ...]:
    return ((r - 1, c), (r + 1, c), (r, c - 1), (r, c + 1))


def _connected_components(mask: np.ndarray) -> list[list[tuple[int, int]]]:
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
                rr, cc = stack.pop()
                comp.append((int(rr), int(cc)))
                for nr, nc in _neighbors4(rr, cc):
                    if 0 <= nr < h and 0 <= nc < w and mask[nr, nc] and not seen[nr, nc]:
                        seen[nr, nc] = True
                        stack.append((int(nr), int(nc)))
            comps.append(comp)
    return comps


def _is_connected(mask: np.ndarray) -> bool:
    total = int(mask.sum())
    if total == 0:
        return False
    return len(_connected_components(mask)) == 1


def _side_candidates(mask: np.ndarray) -> dict[str, list[tuple[int, int]]]:
    h, w = mask.shape
    return {
        "up": [(0, c) for c in range(1, w - 1) if mask[0, c]],
        "down": [(h - 1, c) for c in range(1, w - 1) if mask[h - 1, c]],
        "left": [(r, 0) for r in range(1, h - 1) if mask[r, 0]],
        "right": [(r, w - 1) for r in range(1, h - 1) if mask[r, w - 1]],
    }


def _all_sides_available(mask: np.ndarray) -> bool:
    return all(_side_candidates(mask)[name] for name in DIRS)


def _shift_from_direction(direction: str, height: int, width: int) -> tuple[int, int]:
    if direction == "up":
        return -width, 0
    if direction == "down":
        return width, 0
    if direction == "left":
        return 0, -height
    assert direction == "right"
    return 0, height


def _parse_objects(grid: np.ndarray) -> list[dict[str, object]]:
    mask = grid != BG
    components = _connected_components(mask)
    objects: list[dict[str, object]] = []
    for index, comp in enumerate(components):
        rows = np.array([r for r, _ in comp], dtype=int)
        cols = np.array([c for _, c in comp], dtype=int)
        values = np.array([int(grid[r, c]) for r, c in comp], dtype=int)
        marker_locs = [(r, c) for r, c in comp if int(grid[r, c]) == MARKER]
        assert len(marker_locs) == 1, "each region must contain exactly one marker pixel"
        marker_r, marker_c = marker_locs[0]
        base_colors = sorted(set(map(int, values.tolist())) - {MARKER})
        assert len(base_colors) == 1, "each region must otherwise be monochrome"
        base_color = int(base_colors[0])

        r0, r1 = int(rows.min()), int(rows.max())
        c0, c1 = int(cols.min()), int(cols.max())
        height = int(r1 - r0 + 1)
        width = int(c1 - c0 + 1)

        direction_hits = []
        if marker_r == r0:
            direction_hits.append("up")
        if marker_r == r1:
            direction_hits.append("down")
        if marker_c == c0:
            direction_hits.append("left")
        if marker_c == c1:
            direction_hits.append("right")
        assert len(direction_hits) == 1, "marker must touch exactly one bbox side"

        objects.append(
            {
                "index": int(index),
                "coords": [(int(r), int(c)) for r, c in comp],
                "base_color": int(base_color),
                "marker": (int(marker_r), int(marker_c)),
                "direction": direction_hits[0],
                "bbox": (int(r0), int(c0), int(r1), int(c1)),
                "height": int(height),
                "width": int(width),
                "area": int(len(comp)),
            }
        )
    return objects


def main(grid: np.ndarray) -> np.ndarray:
    """
    Move each marked connected region by the bbox-span rule encoded on its maroon side marker.

    The maroon marker vanishes after movement, so every translated shape is drawn entirely in its base color.
    When translated shapes overlap, larger original regions stay on top.
    """
    g = np.asarray(grid, dtype=int)
    h, w = g.shape
    out = np.full((h, w), BG, dtype=int)
    objects = _parse_objects(g)

    for obj in sorted(objects, key=lambda item: (int(item["area"]), int(item["index"]))):
        dr, dc = _shift_from_direction(
            str(obj["direction"]),
            int(obj["height"]),
            int(obj["width"]),
        )
        color = int(obj["base_color"])
        for r, c in obj["coords"]:
            rr = int(r) + int(dr)
            cc = int(c) + int(dc)
            if 0 <= rr < h and 0 <= cc < w:
                out[rr, cc] = color
    return out


def _random_shape(rng: np.random.Generator, direction: str, used_areas: set[int]) -> dict[str, object] | None:
    for _ in range(GEN_SHAPE_ATTEMPTS):
        height = int(rng.integers(*GEN_SHAPE_SIZE_RANGE))
        width = int(rng.integers(*GEN_SHAPE_SIZE_RANGE))
        mask = np.ones((height, width), dtype=bool)

        protected = {
            (0, int(rng.integers(1, width - 1))),
            (height - 1, int(rng.integers(1, width - 1))),
            (int(rng.integers(1, height - 1)), 0),
            (int(rng.integers(1, height - 1)), width - 1),
        }

        min_area = max(5, int(np.ceil(0.55 * height * width)))
        max_remove = max(0, int(np.floor(0.35 * height * width)))
        target_remove = int(rng.integers(0, max_remove + 1))
        cells = [(r, c) for r in range(height) for c in range(width) if (r, c) not in protected]
        rng.shuffle(cells)

        removed = 0
        for r, c in cells:
            if removed >= target_remove or int(mask.sum()) - 1 < min_area:
                break
            mask[r, c] = False
            if not _is_connected(mask) or not _all_sides_available(mask):
                mask[r, c] = True
                continue
            removed += 1

        area = int(mask.sum())
        if area in used_areas or not _all_sides_available(mask):
            continue

        candidates = _side_candidates(mask)[direction]
        if not candidates:
            continue

        marker = candidates[int(rng.integers(0, len(candidates)))]
        cells_local = [tuple(map(int, cell)) for cell in np.argwhere(mask)]
        return {
            "mask": mask,
            "cells": cells_local,
            "height": int(height),
            "width": int(width),
            "area": int(area),
            "direction": direction,
            "marker_local": (int(marker[0]), int(marker[1])),
        }
    return None


def _output_range(shape: dict[str, object], grid_h: int, grid_w: int) -> tuple[int, int, int, int]:
    dr, dc = _shift_from_direction(
        str(shape["direction"]),
        int(shape["height"]),
        int(shape["width"]),
    )
    return (
        int(dr),
        int(grid_h - int(shape["height"]) + dr),
        int(dc),
        int(grid_w - int(shape["width"]) + dc),
    )


def _input_origin(shape: dict[str, object], out_r0: int, out_c0: int) -> tuple[int, int]:
    dr, dc = _shift_from_direction(
        str(shape["direction"]),
        int(shape["height"]),
        int(shape["width"]),
    )
    return int(out_r0 - dr), int(out_c0 - dc)


def _visible_output_cells(
    shape: dict[str, object],
    out_r0: int,
    out_c0: int,
    grid_h: int,
    grid_w: int,
) -> set[tuple[int, int]]:
    visible: set[tuple[int, int]] = set()
    for r, c in shape["cells"]:
        rr = int(out_r0) + int(r)
        cc = int(out_c0) + int(c)
        if 0 <= rr < grid_h and 0 <= cc < grid_w:
            visible.add((int(rr), int(cc)))
    return visible


def _sample_inside_origin(
    rng: np.random.Generator,
    shape: dict[str, object],
    grid_h: int,
    grid_w: int,
    center_bias: bool = False,
) -> tuple[int, int] | None:
    rmin, rmax, cmin, cmax = _output_range(shape, grid_h, grid_w)
    rlo = max(0, rmin)
    rhi = min(grid_h - int(shape["height"]), rmax)
    clo = max(0, cmin)
    chi = min(grid_w - int(shape["width"]), cmax)
    if rlo > rhi or clo > chi:
        return None

    if center_bias:
        rspan = rhi - rlo + 1
        cspan = chi - clo + 1
        band_r0 = rlo + rspan // 4
        band_r1 = rhi - rspan // 4
        band_c0 = clo + cspan // 4
        band_c1 = chi - cspan // 4
        if band_r0 <= band_r1:
            rlo, rhi = band_r0, band_r1
        if band_c0 <= band_c1:
            clo, chi = band_c0, band_c1

    out_r0 = int(rng.integers(rlo, rhi + 1))
    out_c0 = int(rng.integers(clo, chi + 1))
    return out_r0, out_c0


def _sample_clip_origin(
    rng: np.random.Generator,
    shape: dict[str, object],
    grid_h: int,
    grid_w: int,
) -> tuple[int, int] | None:
    direction = str(shape["direction"])
    height = int(shape["height"])
    width = int(shape["width"])
    rmin, rmax, cmin, cmax = _output_range(shape, grid_h, grid_w)

    rlo = max(0, rmin)
    rhi = min(grid_h - height, rmax)
    clo = max(0, cmin)
    chi = min(grid_w - width, cmax)

    if direction == "up":
        clip_r0 = max(rmin, -height + 1)
        clip_r1 = min(rmax, -1)
        if clip_r0 > clip_r1 or clo > chi:
            return None
        return int(rng.integers(clip_r0, clip_r1 + 1)), int(rng.integers(clo, chi + 1))
    if direction == "down":
        clip_r0 = max(rmin, grid_h - height + 1)
        clip_r1 = min(rmax, grid_h - 1)
        if clip_r0 > clip_r1 or clo > chi:
            return None
        return int(rng.integers(clip_r0, clip_r1 + 1)), int(rng.integers(clo, chi + 1))
    if direction == "left":
        clip_c0 = max(cmin, -width + 1)
        clip_c1 = min(cmax, -1)
        if clip_c0 > clip_c1 or rlo > rhi:
            return None
        return int(rng.integers(rlo, rhi + 1)), int(rng.integers(clip_c0, clip_c1 + 1))

    clip_c0 = max(cmin, grid_w - width + 1)
    clip_c1 = min(cmax, grid_w - 1)
    if clip_c0 > clip_c1 or rlo > rhi:
        return None
    return int(rng.integers(rlo, rhi + 1)), int(rng.integers(clip_c0, clip_c1 + 1))


def _sample_overlap_origin(
    rng: np.random.Generator,
    shape: dict[str, object],
    grid_h: int,
    grid_w: int,
    anchor_bbox: tuple[int, int, int, int],
) -> tuple[int, int] | None:
    height = int(shape["height"])
    width = int(shape["width"])
    rmin, rmax, cmin, cmax = _output_range(shape, grid_h, grid_w)
    ar0, ac0, ah, aw = anchor_bbox

    rlo = max(0, rmin, int(ar0) - height + 1)
    rhi = min(grid_h - height, rmax, int(ar0) + int(ah) - 1)
    clo = max(0, cmin, int(ac0) - width + 1)
    chi = min(grid_w - width, cmax, int(ac0) + int(aw) - 1)
    if rlo > rhi or clo > chi:
        return None

    out_r0 = int(rng.integers(rlo, rhi + 1))
    out_c0 = int(rng.integers(clo, chi + 1))
    return out_r0, out_c0


def _can_place_input_bbox(
    blocked: np.ndarray,
    in_r0: int,
    in_c0: int,
    height: int,
    width: int,
) -> bool:
    grid_h, grid_w = blocked.shape
    if in_r0 < 0 or in_c0 < 0 or in_r0 + height > grid_h or in_c0 + width > grid_w:
        return False
    r0 = max(0, in_r0 - 1)
    r1 = min(grid_h, in_r0 + height + 1)
    c0 = max(0, in_c0 - 1)
    c1 = min(grid_w, in_c0 + width + 1)
    return not bool(blocked[r0:r1, c0:c1].any())


def _build_grid(
    shapes: list[dict[str, object]],
    colors: list[int],
    placements: dict[int, dict[str, object]],
    grid_h: int,
    grid_w: int,
) -> np.ndarray:
    grid = np.full((grid_h, grid_w), BG, dtype=int)
    for idx, shape in enumerate(shapes):
        in_r0, in_c0 = placements[idx]["input_origin"]
        color = int(colors[idx])
        for r, c in shape["cells"]:
            grid[int(in_r0) + int(r), int(in_c0) + int(c)] = color
        mr, mc = shape["marker_local"]
        grid[int(in_r0) + int(mr), int(in_c0) + int(mc)] = MARKER
    return grid


def _generate_attempt(rng: np.random.Generator) -> np.ndarray | None:
    grid_h = int(rng.integers(*GEN_GRID_H_RANGE))
    grid_w = int(rng.integers(*GEN_GRID_W_RANGE))
    shape_count = int(rng.integers(*GEN_SHAPE_COUNT_RANGE))

    directions = list(DIRS)
    while len(directions) < shape_count:
        directions.append(str(rng.choice(DIRS)))
    rng.shuffle(directions)

    shapes: list[dict[str, object]] = []
    used_areas: set[int] = set()
    for direction in directions:
        shape = _random_shape(rng, direction, used_areas)
        if shape is None:
            return None
        used_areas.add(int(shape["area"]))
        shapes.append(shape)

    colors = [int(x) for x in rng.choice(BASE_PALETTE, size=shape_count, replace=False)]

    indices = list(range(shape_count))
    sorted_by_area = sorted(indices, key=lambda idx: int(shapes[idx]["area"]))
    anchor_idx = sorted_by_area[-1]
    partner_idx = sorted_by_area[0]

    force_overlap = bool(rng.random() < 0.55)
    force_clip = bool(rng.random() < 0.40)

    remaining = [idx for idx in indices if idx not in {anchor_idx, partner_idx}]
    clip_idx = None
    if force_clip and remaining:
        clip_idx = int(rng.choice(remaining))
    elif force_clip:
        clip_idx = partner_idx

    blocked = np.zeros((grid_h, grid_w), dtype=bool)
    placements: dict[int, dict[str, object]] = {}
    output_cells: dict[int, set[tuple[int, int]]] = {}

    order: list[int] = []
    if force_overlap:
        order.extend([anchor_idx, partner_idx])
    if clip_idx is not None and clip_idx not in order:
        order.append(int(clip_idx))
    rest = [idx for idx in indices if idx not in order]
    rng.shuffle(rest)
    order.extend(rest)

    for idx in order:
        shape = shapes[idx]
        placed = False
        for _ in range(GEN_PLACEMENT_ATTEMPTS):
            origin: tuple[int, int] | None
            if force_overlap and idx == anchor_idx:
                origin = _sample_inside_origin(rng, shape, grid_h, grid_w, center_bias=True)
            elif force_overlap and idx == partner_idx:
                anchor_place = placements.get(anchor_idx)
                if anchor_place is None:
                    continue
                origin = _sample_overlap_origin(
                    rng,
                    shape,
                    grid_h,
                    grid_w,
                    anchor_place["output_bbox"],
                )
            elif clip_idx is not None and idx == clip_idx:
                origin = _sample_clip_origin(rng, shape, grid_h, grid_w)
            else:
                origin = _sample_inside_origin(rng, shape, grid_h, grid_w, center_bias=False)

            if origin is None:
                continue

            out_r0, out_c0 = origin
            in_r0, in_c0 = _input_origin(shape, out_r0, out_c0)
            if not _can_place_input_bbox(
                blocked,
                in_r0,
                in_c0,
                int(shape["height"]),
                int(shape["width"]),
            ):
                continue

            visible = _visible_output_cells(shape, out_r0, out_c0, grid_h, grid_w)
            area = int(shape["area"])
            if len(visible) < max(3, area // 3):
                continue

            if force_overlap and idx == partner_idx:
                shared = visible & output_cells[anchor_idx]
                if len(shared) < 2:
                    continue

            if clip_idx is not None and idx == clip_idx and len(visible) == area:
                continue

            blocked[
                int(in_r0) : int(in_r0) + int(shape["height"]),
                int(in_c0) : int(in_c0) + int(shape["width"]),
            ] = True
            placements[idx] = {
                "input_origin": (int(in_r0), int(in_c0)),
                "output_origin": (int(out_r0), int(out_c0)),
                "output_bbox": (
                    int(out_r0),
                    int(out_c0),
                    int(shape["height"]),
                    int(shape["width"]),
                ),
            }
            output_cells[idx] = visible
            placed = True
            break
        if not placed:
            return None

    grid = _build_grid(shapes, colors, placements, grid_h, grid_w)

    try:
        parsed = _parse_objects(grid)
        if len(parsed) != shape_count:
            return None
        out = main(grid)
    except AssertionError:
        return None

    if out.shape != grid.shape or np.array_equal(out, grid):
        return None

    overlaps = any(
        bool(output_cells[i] & output_cells[j])
        for i in range(shape_count)
        for j in range(i + 1, shape_count)
    )
    clipped = any(len(output_cells[idx]) < int(shapes[idx]["area"]) for idx in range(shape_count))
    if force_overlap and not overlaps:
        return None
    if force_clip and not clipped:
        return None

    return grid


def generate_input() -> np.ndarray:
    rng = np.random.default_rng()
    for _ in range(GEN_ATTEMPTS):
        grid = _generate_attempt(rng)
        if grid is not None:
            return grid
    raise RuntimeError("failed to generate a valid input")
