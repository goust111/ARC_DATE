from __future__ import annotations

import numpy as np


RANK_COLORS = [2, 1, 3, 4, 6]  # red, blue, green, yellow, magenta
NEIGHBORS4 = [(-1, 0), (1, 0), (0, -1), (0, 1)]
RNG = np.random.default_rng()

# Generator controls.
MIN_GRID_SIZE = 16
MAX_GRID_SIZE = 28
MIN_OBJECTS = 2
MAX_OBJECTS = 5
MIN_DENSITY = 0.55
MAX_GENERATION_RETRIES = 400
MASK_GENERATION_RETRIES = 200
PLACEMENT_RETRIES = 500
RECT_MARGIN = 1

HORIZONTAL_DIMS = [(h, w) for h in range(2, 6) for w in range(h + 1, 9)]
VERTICAL_DIMS = [(h, w) for w in range(2, 6) for h in range(w + 1, 9)]


def _get_bg(grid: np.ndarray) -> int:
    colors, counts = np.unique(grid, return_counts=True)
    return int(colors[np.argmax(counts)])


def _find_components(grid: np.ndarray, bg: int) -> list[dict]:
    h, w = grid.shape
    seen = np.zeros((h, w), dtype=bool)
    components: list[dict] = []

    for r in range(h):
        for c in range(w):
            color = int(grid[r, c])
            if color == bg or seen[r, c]:
                continue

            stack = [(r, c)]
            seen[r, c] = True
            coords: list[tuple[int, int]] = []

            while stack:
                rr, cc = stack.pop()
                coords.append((rr, cc))
                for dr, dc in NEIGHBORS4:
                    nr = rr + dr
                    nc = cc + dc
                    if 0 <= nr < h and 0 <= nc < w and not seen[nr, nc] and int(grid[nr, nc]) == color:
                        seen[nr, nc] = True
                        stack.append((nr, nc))

            rows = [rr for rr, _ in coords]
            cols = [cc for _, cc in coords]
            r0, r1 = min(rows), max(rows)
            c0, c1 = min(cols), max(cols)
            mask = np.zeros((r1 - r0 + 1, c1 - c0 + 1), dtype=bool)
            for rr, cc in coords:
                mask[rr - r0, cc - c0] = True

            components.append(
                {
                    "color": color,
                    "area": len(coords),
                    "bbox": (r0, c0, r1, c1),
                    "mask": mask,
                }
            )

    return components


def main(grid: np.ndarray) -> np.ndarray:
    """Flip each object along its bbox long axis, then recolor by descending area rank."""
    g = np.asarray(grid, dtype=int)
    assert g.ndim == 2

    bg = _get_bg(g)
    components = _find_components(g, bg)
    assert 1 <= len(components) <= len(RANK_COLORS)

    ordered = sorted(
        components,
        key=lambda comp: (-int(comp["area"]), int(comp["bbox"][0]), int(comp["bbox"][1])),
    )

    out = np.full(g.shape, int(bg), dtype=int)
    for rank, comp in enumerate(ordered):
        r0, c0, r1, c1 = comp["bbox"]
        mask = np.asarray(comp["mask"], dtype=bool)
        if mask.shape[1] > mask.shape[0]:
            flipped = np.fliplr(mask)
        else:
            flipped = np.flipud(mask)
        patch = out[r0 : r1 + 1, c0 : c1 + 1]
        patch[flipped] = int(RANK_COLORS[rank])

    return out


def _is_connected(mask: np.ndarray) -> bool:
    coords = np.argwhere(mask)
    if len(coords) == 0:
        return False

    seen = np.zeros_like(mask, dtype=bool)
    start_r, start_c = map(int, coords[0])
    stack = [(start_r, start_c)]
    seen[start_r, start_c] = True
    visited = 0

    while stack:
        r, c = stack.pop()
        visited += 1
        for dr, dc in NEIGHBORS4:
            nr = r + dr
            nc = c + dc
            if 0 <= nr < mask.shape[0] and 0 <= nc < mask.shape[1] and mask[nr, nc] and not seen[nr, nc]:
                seen[nr, nc] = True
                stack.append((nr, nc))

    return visited == int(mask.sum())


def _touches_all_edges(mask: np.ndarray) -> bool:
    return bool(mask[0, :].any() and mask[-1, :].any() and mask[:, 0].any() and mask[:, -1].any())


def _is_boundary_cell(mask: np.ndarray, r: int, c: int) -> bool:
    for dr, dc in NEIGHBORS4:
        nr = r + dr
        nc = c + dc
        if not (0 <= nr < mask.shape[0] and 0 <= nc < mask.shape[1]):
            return True
        if not mask[nr, nc]:
            return True
    return False


def _build_mask(height: int, width: int, area: int, orientation: str) -> np.ndarray | None:
    for _ in range(MASK_GENERATION_RETRIES):
        mask = np.ones((height, width), dtype=bool)

        while int(mask.sum()) > area:
            boundary_cells = [(int(r), int(c)) for r, c in np.argwhere(mask) if _is_boundary_cell(mask, int(r), int(c))]
            if not boundary_cells:
                break

            removable: list[tuple[int, int]] = []
            order = RNG.permutation(len(boundary_cells))
            for idx in order:
                r, c = boundary_cells[int(idx)]
                mask[r, c] = False
                if _touches_all_edges(mask) and _is_connected(mask):
                    removable.append((r, c))
                mask[r, c] = True

            if not removable:
                break

            r, c = removable[int(RNG.integers(0, len(removable)))]
            mask[r, c] = False

        if int(mask.sum()) != area:
            continue
        if not _touches_all_edges(mask):
            continue
        if not _is_connected(mask):
            continue

        flipped = np.fliplr(mask) if orientation == "H" else np.flipud(mask)
        if np.array_equal(mask, flipped):
            continue
        return mask

    return None


def _make_object(orientation: str, color: int, used_areas: set[int]) -> dict | None:
    dims_pool = HORIZONTAL_DIMS if orientation == "H" else VERTICAL_DIMS

    for dim_idx in RNG.permutation(len(dims_pool)):
        height, width = dims_pool[int(dim_idx)]
        min_area = max(height + width - 1, int(np.ceil(MIN_DENSITY * height * width)))
        max_area = height * width - 1
        if min_area > max_area:
            continue

        area_choices = [area for area in range(min_area, max_area + 1) if area not in used_areas]
        if not area_choices:
            continue

        for area_idx in RNG.permutation(len(area_choices)):
            area = int(area_choices[int(area_idx)])
            mask = _build_mask(height, width, area, orientation)
            if mask is None:
                continue
            return {
                "color": int(color),
                "area": int(area),
                "mask": mask,
                "height": int(height),
                "width": int(width),
            }

    return None


def _sample_orientations(count: int) -> list[str]:
    if count == 2:
        orientations = ["H", "V"]
    else:
        extra_horizontal = int(RNG.integers(0, 2)) if count % 2 else 0
        horizontal_count = count // 2 + extra_horizontal
        vertical_count = count - horizontal_count
        orientations = ["H"] * horizontal_count + ["V"] * vertical_count
    RNG.shuffle(orientations)
    return orientations


def _rect_conflicts(a: tuple[int, int, int, int], b: tuple[int, int, int, int], margin: int) -> bool:
    ar0, ac0, ah, aw = a
    br0, bc0, bh, bw = b
    ar1, ac1 = ar0 + ah - 1, ac0 + aw - 1
    br1, bc1 = br0 + bh - 1, bc0 + bw - 1
    separated = ar1 + margin < br0 or br1 + margin < ar0 or ac1 + margin < bc0 or bc1 + margin < ac0
    return not separated


def _sample_canvas(objects: list[dict]) -> tuple[int, int] | None:
    total_area = sum(int(obj["area"]) for obj in objects)
    padded_area = sum((int(obj["height"]) + 2) * (int(obj["width"]) + 2) for obj in objects)
    min_canvas_area = max(3 * total_area, padded_area)
    min_side = int(np.ceil(np.sqrt(min_canvas_area)))
    min_side = max(
        min_side,
        max(int(obj["height"]) for obj in objects) + 2,
        max(int(obj["width"]) for obj in objects) + 2,
        MIN_GRID_SIZE,
    )
    if min_side > MAX_GRID_SIZE:
        return None

    for _ in range(50):
        height = int(RNG.integers(min_side, MAX_GRID_SIZE + 1))
        width = int(RNG.integers(min_side, MAX_GRID_SIZE + 1))
        if height * width >= min_canvas_area:
            return height, width
    return None


def _place_objects(objects: list[dict], height: int, width: int) -> bool:
    placed: list[tuple[int, int, int, int]] = []
    order = sorted(range(len(objects)), key=lambda idx: objects[idx]["height"] * objects[idx]["width"], reverse=True)

    for idx in order:
        obj = objects[idx]
        obj_h = int(obj["height"])
        obj_w = int(obj["width"])
        found = False

        for _ in range(PLACEMENT_RETRIES):
            r0 = int(RNG.integers(0, height - obj_h + 1))
            c0 = int(RNG.integers(0, width - obj_w + 1))
            rect = (r0, c0, obj_h, obj_w)
            if any(_rect_conflicts(rect, other, RECT_MARGIN) for other in placed):
                continue
            obj["row"] = r0
            obj["col"] = c0
            placed.append(rect)
            found = True
            break

        if not found:
            return False

    return True


def _render_input(objects: list[dict], height: int, width: int, bg: int) -> np.ndarray:
    grid = np.full((height, width), int(bg), dtype=int)
    for obj in objects:
        r0 = int(obj["row"])
        c0 = int(obj["col"])
        patch = grid[r0 : r0 + int(obj["height"]), c0 : c0 + int(obj["width"])]
        patch[np.asarray(obj["mask"], dtype=bool)] = int(obj["color"])
    return grid


def _expected_output(objects: list[dict], height: int, width: int, bg: int) -> np.ndarray:
    out = np.full((height, width), int(bg), dtype=int)
    ordered = sorted(objects, key=lambda obj: (-int(obj["area"]), int(obj["row"]), int(obj["col"])))
    for rank, obj in enumerate(ordered):
        r0 = int(obj["row"])
        c0 = int(obj["col"])
        mask = np.asarray(obj["mask"], dtype=bool)
        flipped = np.fliplr(mask) if int(obj["width"]) > int(obj["height"]) else np.flipud(mask)
        patch = out[r0 : r0 + int(obj["height"]), c0 : c0 + int(obj["width"])]
        patch[flipped] = int(RANK_COLORS[rank])
    return out


def _gen_attempt() -> np.ndarray | None:
    bg = int(RNG.integers(0, 10))
    object_count = int(RNG.integers(MIN_OBJECTS, MAX_OBJECTS + 1))
    palette = [color for color in range(10) if color != bg]
    colors = list(map(int, RNG.choice(palette, size=object_count, replace=False)))
    RNG.shuffle(colors)

    objects: list[dict] = []
    used_areas: set[int] = set()
    for orientation, color in zip(_sample_orientations(object_count), colors):
        obj = _make_object(orientation, color, used_areas)
        if obj is None:
            return None
        used_areas.add(int(obj["area"]))
        objects.append(obj)

    canvas = _sample_canvas(objects)
    if canvas is None:
        return None
    height, width = canvas
    if not _place_objects(objects, height, width):
        return None

    grid = _render_input(objects, height, width, bg)
    if _get_bg(grid) != bg:
        return None

    expected = _expected_output(objects, height, width, bg)
    try:
        if not np.array_equal(main(grid), expected):
            return None
    except AssertionError:
        return None

    components = _find_components(grid, bg)
    if len(components) != object_count:
        return None
    if len({int(comp["area"]) for comp in components}) != object_count:
        return None
    if not ({comp["mask"].shape[1] > comp["mask"].shape[0] for comp in components} == {False, True}):
        return None

    return grid


def generate_input() -> np.ndarray:
    for _ in range(MAX_GENERATION_RETRIES):
        grid = _gen_attempt()
        if grid is not None:
            return grid
    raise RuntimeError(f"failed to generate valid input after {MAX_GENERATION_RETRIES} attempts")
