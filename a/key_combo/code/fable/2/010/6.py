from __future__ import annotations

from collections import Counter
from pathlib import Path
import random
import sys

import numpy as np
from scipy.ndimage import binary_fill_holes

_ARC_UTILS_DIR = Path("/home/lixinyu/snap/new/solvers")
if str(_ARC_UTILS_DIR) not in sys.path:
    sys.path.append(str(_ARC_UTILS_DIR))

from arc_utils import (
    bbox_coords,
    bbox_h,
    bbox_w,
    colors_present,
    components_mask,
    crop_bbox,
    find_components,
    get_bg,
    paste,
)


DIRS4 = ((-1, 0), (1, 0), (0, -1), (0, 1))
GEN_ATTEMPTS = 500
ALL_COLORS = tuple(range(10))
SIDE_DIRECTIONS = ((1, 0), (-1, 0), (0, 1), (0, -1))


def _dominant_object(grid: np.ndarray, bg: int) -> dict:
    counts = Counter(map(int, grid[grid != bg].flat))
    assert counts, "expected non-background colors"
    object_color, object_count = counts.most_common(1)[0]
    assert sum(count == object_count for count in counts.values()) == 1, "object color must be unique"

    components = find_components(
        grid,
        bg=bg,
        connectivity=4,
        monochromatic=True,
        colors=[int(object_color)],
    )
    assert components, "expected an object component"
    areas = [int(component["area"]) for component in components]
    largest_area = max(areas)
    assert areas.count(largest_area) == 1, "expected one dominant object component"
    return max(components, key=lambda component: int(component["area"]))


def _attachment_dirs(coords: np.ndarray, object_mask: np.ndarray) -> set[tuple[int, int]]:
    height, width = object_mask.shape
    dirs: set[tuple[int, int]] = set()
    for r, c in coords:
        rr = int(r)
        cc = int(c)
        for dr, dc in DIRS4:
            nr = rr + dr
            nc = cc + dc
            if 0 <= nr < height and 0 <= nc < width and bool(object_mask[nr, nc]):
                dirs.add((int(dr), int(dc)))
    return dirs


def _extract_patch(grid: np.ndarray, bg: int, object_color: int, object_mask: np.ndarray) -> dict:
    patch_mask = (grid != int(bg)) & (grid != int(object_color))
    patch_components = components_mask(patch_mask, connectivity=4)
    assert patch_components, "expected a colored patch"

    candidates = []
    for coords in patch_components:
        dirs = _attachment_dirs(coords, object_mask)
        if not dirs:
            continue
        box = bbox_coords(coords)
        r0, c0, r1, c1 = box
        if int(coords.shape[0]) != bbox_h(box) * bbox_w(box):
            continue
        if not bool(np.all(patch_mask[r0 : r1 + 1, c0 : c1 + 1])):
            continue
        candidates.append(
            {
                "coords": coords,
                "bbox": box,
                "tile": crop_bbox(grid, box),
                "dirs": dirs,
            }
        )

    assert len(candidates) == 1, "expected exactly one rectangular patch touching the object"
    patch = candidates[0]
    assert len(patch["dirs"]) == 1, "patch must attach along one side"
    return patch


def _check_full_attachment(object_mask: np.ndarray, patch_bbox: tuple[int, int, int, int], direction: tuple[int, int]) -> None:
    r0, c0, r1, c1 = patch_bbox
    dr, dc = direction
    if dr == 1:
        assert bool(np.all(object_mask[r1 + 1, c0 : c1 + 1]))
    elif dr == -1:
        assert bool(np.all(object_mask[r0 - 1, c0 : c1 + 1]))
    elif dc == 1:
        assert bool(np.all(object_mask[r0 : r1 + 1, c1 + 1]))
    else:
        assert bool(np.all(object_mask[r0 : r1 + 1, c0 - 1]))


def _hole_mask(grid: np.ndarray, object_mask: np.ndarray, bg: int) -> np.ndarray:
    filled = binary_fill_holes(object_mask)
    holes = filled & ~object_mask
    return holes & (grid == int(bg))


def _cell_in_band(r: int, c: int, patch_bbox: tuple[int, int, int, int], direction: tuple[int, int]) -> bool:
    r0, c0, r1, c1 = patch_bbox
    dr, _dc = direction
    if dr != 0:
        return c0 <= int(c) <= c1
    return r0 <= int(r) <= r1


def _relative_patch_bbox(
    object_bbox: tuple[int, int, int, int],
    patch_bbox: tuple[int, int, int, int],
) -> tuple[int, int, int, int]:
    obj_r0, obj_c0, _obj_r1, _obj_c1 = object_bbox
    patch_r0, patch_c0, patch_r1, patch_c1 = patch_bbox
    return (
        int(patch_r0 - obj_r0),
        int(patch_c0 - obj_c0),
        int(patch_r1 - obj_r0),
        int(patch_c1 - obj_c0),
    )


def _sample_tile_color(tile: np.ndarray, patch_bbox: tuple[int, int, int, int], r: int, c: int) -> int:
    r0, c0, _r1, _c1 = patch_bbox
    tr = (int(r) - int(r0)) % tile.shape[0]
    tc = (int(c) - int(c0)) % tile.shape[1]
    return int(tile[tr, tc])


def _analyze_scene(grid: np.ndarray) -> dict:
    g = np.asarray(grid, dtype=int)
    bg = int(get_bg(g))

    obj = _dominant_object(g, bg)
    object_color = int(obj["color"])
    object_mask = np.zeros_like(g, dtype=bool)
    coords = obj["coords"]
    object_mask[coords[:, 0], coords[:, 1]] = True

    patch = _extract_patch(g, bg, object_color, object_mask)
    direction = next(iter(patch["dirs"]))
    _check_full_attachment(object_mask, patch["bbox"], direction)

    holes = _hole_mask(g, object_mask, bg)
    assert bool(holes.any()), "expected enclosed holes"

    return {
        "grid": g,
        "bg": bg,
        "object_color": object_color,
        "object_mask": object_mask,
        "patch_bbox": patch["bbox"],
        "tile": patch["tile"],
        "direction": direction,
        "holes": holes,
    }


def main(grid: np.ndarray) -> np.ndarray:
    """
    Reveal a periodic patch pattern only through the enclosed holes of the large monochrome object.

    Input structure
    - The background is the dominant color.
    - One large monochrome component contains one or more enclosed background holes.
    - A small multicolor rectangular patch is attached flush to exactly one outer side of that object.

    Rule
    - Use the patch as one full period of a hidden infinite strip that continues perpendicular to the touched side.
    - Keep the patch and the large object unchanged.
    - For each background hole cell that lies inside the strip, recolor it with the periodic patch color at the same phase; hole cells outside the strip stay background.

    Key insight
    - The output does not copy a nearby row or column from the patch: each visible hole cell samples one coordinate from the same hidden periodic wallpaper anchored by the patch itself.
    """
    scene = _analyze_scene(np.asarray(grid, dtype=int))
    out = scene["grid"].copy()

    holes = scene["holes"]
    patch_bbox = scene["patch_bbox"]
    direction = scene["direction"]
    tile = scene["tile"]

    for r, c in np.argwhere(holes):
        rr = int(r)
        cc = int(c)
        if _cell_in_band(rr, cc, patch_bbox, direction):
            out[rr, cc] = _sample_tile_color(tile, patch_bbox, rr, cc)

    return out


def _base_hole_shapes() -> list[np.ndarray]:
    return [
        np.array([[1]], dtype=bool),
        np.array([[1, 1]], dtype=bool),
        np.array([[1], [1]], dtype=bool),
        np.array([[1, 1, 1]], dtype=bool),
        np.array([[1], [1], [1]], dtype=bool),
        np.array([[1, 1], [1, 1]], dtype=bool),
        np.array([[1, 1, 1], [1, 1, 1]], dtype=bool),
        np.array([[1, 1], [1, 1], [1, 1]], dtype=bool),
        np.array([[1, 1, 1, 1]], dtype=bool),
        np.array([[1], [1], [1], [1]], dtype=bool),
        np.array([[1, 0], [1, 1]], dtype=bool),
        np.array([[0, 1], [1, 1]], dtype=bool),
        np.array([[1, 1], [1, 0]], dtype=bool),
        np.array([[1, 1], [0, 1]], dtype=bool),
        np.array([[1, 1, 1], [0, 1, 0]], dtype=bool),
        np.array([[0, 1, 0], [1, 1, 1]], dtype=bool),
        np.array([[1, 0], [1, 1], [1, 0]], dtype=bool),
        np.array([[0, 1], [1, 1], [0, 1]], dtype=bool),
    ]


def _unique_shape_variants() -> list[np.ndarray]:
    variants: dict[bytes, np.ndarray] = {}
    for base in _base_hole_shapes():
        for k in range(4):
            rot = np.rot90(base, k)
            key = rot.astype(np.uint8).tobytes() + str(rot.shape).encode()
            variants.setdefault(key, rot.astype(bool))
    return list(variants.values())


HOLE_SHAPES = _unique_shape_variants()


def _make_patch_tile(height: int, width: int, palette: list[int]) -> np.ndarray | None:
    for _ in range(100):
        tile = np.array(random.choices(palette, k=height * width), dtype=int).reshape(height, width)
        if len(set(map(int, tile.flat))) < 2:
            continue
        if len({tuple(map(int, row)) for row in tile}) < 2:
            continue
        if len({tuple(map(int, tile[:, c])) for c in range(width)}) < 2:
            continue
        return tile
    return None


def _sample_layout(direction: tuple[int, int]) -> dict | None:
    patch_h = random.randint(2, 4)
    patch_w = random.randint(2, 5)
    if direction[0] != 0:
        obj_h = random.randint(10, 17)
        obj_w = random.randint(max(8, patch_w + 4), 18)
    else:
        obj_h = random.randint(max(8, patch_h + 4), 18)
        obj_w = random.randint(10, 17)

    top = random.randint(1, 4)
    bottom = random.randint(1, 4)
    left = random.randint(1, 4)
    right = random.randint(1, 4)

    if direction == (1, 0):
        height = top + patch_h + obj_h + bottom
        width = left + obj_w + right
        obj_r0 = top + patch_h
        obj_c0 = left
        patch_r0 = top
        patch_c0 = left + random.randint(1, obj_w - patch_w - 1)
    elif direction == (-1, 0):
        height = top + obj_h + patch_h + bottom
        width = left + obj_w + right
        obj_r0 = top
        obj_c0 = left
        patch_r0 = obj_r0 + obj_h
        patch_c0 = left + random.randint(1, obj_w - patch_w - 1)
    elif direction == (0, 1):
        height = top + obj_h + bottom
        width = left + patch_w + obj_w + right
        obj_r0 = top
        obj_c0 = left + patch_w
        patch_r0 = top + random.randint(1, obj_h - patch_h - 1)
        patch_c0 = left
    else:
        height = top + obj_h + bottom
        width = left + obj_w + patch_w + right
        obj_r0 = top
        obj_c0 = left
        patch_r0 = top + random.randint(1, obj_h - patch_h - 1)
        patch_c0 = obj_c0 + obj_w

    if max(height, width) > 30:
        return None

    obj_bbox = (obj_r0, obj_c0, obj_r0 + obj_h - 1, obj_c0 + obj_w - 1)
    patch_bbox = (patch_r0, patch_c0, patch_r0 + patch_h - 1, patch_c0 + patch_w - 1)
    return {
        "shape": (height, width),
        "direction": direction,
        "object_bbox": obj_bbox,
        "patch_bbox": patch_bbox,
        "object_shape": (obj_h, obj_w),
        "patch_shape": (patch_h, patch_w),
    }


def _band_relation(cells: list[tuple[int, int]], direction: tuple[int, int], patch_bbox: tuple[int, int, int, int]) -> str:
    inside = [bool(_cell_in_band(r, c, patch_bbox, direction)) for r, c in cells]
    if not any(inside):
        return "outside"
    if all(inside):
        return "full"
    return "partial"


def _shape_cells(shape: np.ndarray, r0: int, c0: int) -> list[tuple[int, int]]:
    return [(int(r0 + r), int(c0 + c)) for r, c in np.argwhere(shape)]


def _hole_candidates(
    object_shape: tuple[int, int],
    shape: np.ndarray,
    role: str,
    patch_bbox: tuple[int, int, int, int],
    direction: tuple[int, int],
    forbidden: np.ndarray,
) -> list[list[tuple[int, int]]]:
    obj_h, obj_w = object_shape
    sh, sw = shape.shape
    candidates: list[list[tuple[int, int]]] = []

    for rr in range(1, obj_h - sh):
        for cc in range(1, obj_w - sw):
            cells = _shape_cells(shape, rr, cc)
            if any(bool(forbidden[r, c]) for r, c in cells):
                continue
            if _band_relation(cells, direction, patch_bbox) != role:
                continue
            candidates.append(cells)
    return candidates


def _mark_forbidden(forbidden: np.ndarray, cells: list[tuple[int, int]]) -> None:
    obj_h, obj_w = forbidden.shape
    for r, c in cells:
        for nr in range(max(1, r - 1), min(obj_h - 1, r + 2)):
            for nc in range(max(1, c - 1), min(obj_w - 1, c + 2)):
                forbidden[nr, nc] = True


def _sample_hole_roles() -> list[str]:
    primary = random.choice(["full", "partial"])
    roles = [primary, "outside"]
    if random.random() < 0.8:
        roles.append("partial" if primary == "full" else "full")
    if random.random() < 0.45:
        roles.append(random.choice(["outside", "full", "partial"]))
    return roles


def _place_holes(layout: dict) -> list[list[tuple[int, int]]] | None:
    obj_h, obj_w = layout["object_shape"]
    patch_bbox = _relative_patch_bbox(layout["object_bbox"], layout["patch_bbox"])
    direction = layout["direction"]
    forbidden = np.zeros((obj_h, obj_w), dtype=bool)
    placed: list[list[tuple[int, int]]] = []

    roles = _sample_hole_roles()
    role_order = sorted(range(len(roles)), key=lambda idx: {"partial": 0, "full": 1, "outside": 2}[roles[idx]])
    holes_by_index: list[list[tuple[int, int]] | None] = [None] * len(roles)

    for idx in role_order:
        role = roles[idx]
        shapes = random.sample(HOLE_SHAPES, len(HOLE_SHAPES))
        random.shuffle(shapes)
        candidates: list[list[tuple[int, int]]] = []
        for shape in shapes:
            candidates.extend(_hole_candidates((obj_h, obj_w), shape, role, patch_bbox, direction, forbidden))
        if not candidates:
            return None
        cells = random.choice(candidates)
        holes_by_index[idx] = cells
        placed.append(cells)
        _mark_forbidden(forbidden, cells)

    holes = [hole for hole in holes_by_index if hole is not None]
    assert len(holes) == len(roles)
    return holes


def _render_case(case: dict) -> tuple[np.ndarray, np.ndarray]:
    height, width = case["shape"]
    bg = int(case["bg"])
    object_color = int(case["object_color"])
    grid = np.full((height, width), bg, dtype=int)

    obj_r0, obj_c0, obj_r1, obj_c1 = case["object_bbox"]
    patch_r0, patch_c0, patch_r1, patch_c1 = case["patch_bbox"]
    tile = case["tile"]

    grid[obj_r0 : obj_r1 + 1, obj_c0 : obj_c1 + 1] = object_color
    paste(grid, tile, patch_r0, patch_c0)

    hole_cells_abs: list[tuple[int, int]] = []
    for hole in case["holes"]:
        for r, c in hole:
            rr = obj_r0 + int(r)
            cc = obj_c0 + int(c)
            grid[rr, cc] = bg
            hole_cells_abs.append((rr, cc))

    out = grid.copy()
    for r, c in hole_cells_abs:
        if _cell_in_band(r, c, case["patch_bbox"], case["direction"]):
            out[r, c] = _sample_tile_color(tile, case["patch_bbox"], r, c)

    return grid, out


def _generate_case() -> tuple[np.ndarray, np.ndarray] | None:
    direction = random.choice(SIDE_DIRECTIONS)
    layout = _sample_layout(direction)
    if layout is None:
        return None

    bg = random.choice(ALL_COLORS)
    object_color = random.choice([color for color in ALL_COLORS if color != bg])
    patch_palette = random.sample([color for color in ALL_COLORS if color not in {bg, object_color}], random.randint(2, 4))
    tile = _make_patch_tile(layout["patch_shape"][0], layout["patch_shape"][1], patch_palette)
    if tile is None:
        return None

    holes = _place_holes(layout)
    if holes is None:
        return None

    case = {
        **layout,
        "bg": int(bg),
        "object_color": int(object_color),
        "tile": tile,
        "holes": holes,
    }
    grid, expected = _render_case(case)

    counts = Counter(map(int, grid.flat))
    if counts[int(bg)] <= max(count for color, count in counts.items() if color != int(bg)):
        return None

    relative_patch_bbox = _relative_patch_bbox(case["object_bbox"], case["patch_bbox"])
    band_hits = 0
    band_misses = 0
    for hole in holes:
        relation = _band_relation(hole, direction, relative_patch_bbox)
        if relation == "outside":
            band_misses += 1
        else:
            band_hits += 1
    if band_hits == 0 or band_misses == 0:
        return None

    changed = np.argwhere(grid != expected)
    if changed.size == 0:
        return None

    try:
        actual = main(grid)
    except Exception:
        return None
    if not np.array_equal(actual, expected):
        return None

    scene = _analyze_scene(grid)
    if not np.array_equal(scene["holes"], binary_fill_holes(scene["object_mask"]) & ~scene["object_mask"] & (grid == scene["bg"])):
        return None

    return grid, expected


def generate_input() -> np.ndarray:
    for _ in range(GEN_ATTEMPTS):
        case = _generate_case()
        if case is None:
            continue
        grid, _expected = case
        return grid
    raise RuntimeError("failed to generate a valid periodic-hole-reveal input")
