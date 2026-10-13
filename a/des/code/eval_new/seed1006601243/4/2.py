from __future__ import annotations

import random

import numpy as np


BG = 8  # azure
MARKER_COLOR = 0
FRAME_COLOR_POOL = [1, 2, 3, 4, 5, 6, 7, 9]
OBJECT_COLOR_POOL = [1, 2, 3, 4, 5, 6, 7, 9]
FRAME_SIDE_CHOICES = ("left", "right")
FRAME_HEIGHT = 5
FRAME_GAP = 1
FRAME_CAPACITY_MAX = 3
COUNT_TRIPLES = [
    (a, b, c)
    for a in range(1, FRAME_CAPACITY_MAX + 1)
    for b in range(1, FRAME_CAPACITY_MAX + 1)
    for c in range(1, FRAME_CAPACITY_MAX + 1)
    if 3 <= a + b + c <= 5
]
GEN_ATTEMPTS = 300
CLASS_ORDER = ("rot4", "rot2", "none")


def _background_color(grid: np.ndarray) -> int:
    values, counts = np.unique(np.asarray(grid, dtype=int), return_counts=True)
    return int(values[int(np.argmax(counts))])


def _neighbors4(r: int, c: int) -> list[tuple[int, int]]:
    return [(r - 1, c), (r + 1, c), (r, c - 1), (r, c + 1)]


def _connected(mask: np.ndarray) -> bool:
    points = np.argwhere(mask)
    if points.size == 0:
        return False
    H, W = mask.shape
    start = tuple(map(int, points[0]))
    seen = {start}
    stack = [start]
    while stack:
        r, c = stack.pop()
        for nr, nc in _neighbors4(r, c):
            if 0 <= nr < H and 0 <= nc < W and mask[nr, nc] and (nr, nc) not in seen:
                seen.add((nr, nc))
                stack.append((nr, nc))
    return len(seen) == int(points.shape[0])


def _rotation_class(mask: np.ndarray) -> str:
    arr = np.asarray(mask, dtype=bool)
    if np.array_equal(arr, np.rot90(arr)):
        return "rot4"
    if np.array_equal(arr, np.rot90(arr, 2)):
        return "rot2"
    return "none"


def _enumerate_shape_catalog() -> dict[str, list[np.ndarray]]:
    catalog: dict[str, list[np.ndarray]] = {name: [] for name in CLASS_ORDER}
    for bits in range(1, 1 << 9):
        mask = np.array([(bits >> i) & 1 for i in range(9)], dtype=bool).reshape(3, 3)
        if not np.all(mask.any(axis=0)) or not np.all(mask.any(axis=1)):
            continue
        if not _connected(mask):
            continue
        catalog[_rotation_class(mask)].append(mask)
    return catalog


SHAPE_CATALOG = _enumerate_shape_catalog()


def _component_list(grid: np.ndarray, bg: int) -> list[dict]:
    g = np.asarray(grid, dtype=int)
    H, W = g.shape
    visited = np.zeros((H, W), dtype=bool)
    components: list[dict] = []

    for r in range(H):
        for c in range(W):
            if visited[r, c] or int(g[r, c]) == bg:
                continue
            color = int(g[r, c])
            stack = [(r, c)]
            visited[r, c] = True
            coords: list[tuple[int, int]] = []
            while stack:
                cr, cc = stack.pop()
                coords.append((cr, cc))
                for nr, nc in _neighbors4(cr, cc):
                    if 0 <= nr < H and 0 <= nc < W and not visited[nr, nc] and int(g[nr, nc]) == color:
                        visited[nr, nc] = True
                        stack.append((nr, nc))

            arr = np.array(coords, dtype=int)
            r0 = int(arr[:, 0].min())
            r1 = int(arr[:, 0].max())
            c0 = int(arr[:, 1].min())
            c1 = int(arr[:, 1].max())
            patch = np.full((r1 - r0 + 1, c1 - c0 + 1), bg, dtype=int)
            patch[arr[:, 0] - r0, arr[:, 1] - c0] = color
            components.append(
                {
                    "color": color,
                    "coords": arr,
                    "bbox": (r0, c0, r1, c1),
                    "patch": patch,
                    "area": int(arr.shape[0]),
                }
            )
    return components


def _is_border_rectangle(component: dict, H: int, W: int) -> bool:
    r0, c0, r1, c1 = component["bbox"]
    h = int(r1 - r0 + 1)
    w = int(c1 - c0 + 1)
    if h < FRAME_HEIGHT or w < 8:
        return False
    if r0 != 0 and c0 != 0 and r1 != H - 1 and c1 != W - 1:
        return False
    local = np.zeros((h, w), dtype=bool)
    coords = component["coords"]
    local[coords[:, 0] - r0, coords[:, 1] - c0] = True
    border = np.zeros((h, w), dtype=bool)
    border[0, :] = True
    border[-1, :] = True
    border[:, 0] = True
    border[:, -1] = True
    return int(coords.shape[0]) == int(border.sum()) and np.array_equal(local, border)


def _decode_marker(component: dict) -> str:
    patch = component["patch"]
    h, w = patch.shape
    area = int(component["area"])
    if area == 4 and (h, w) == (2, 2):
        return "rot4"
    if area == 2 and (h, w) == (1, 2):
        return "rot2"
    if area == 1 and (h, w) == (1, 1):
        return "none"
    raise AssertionError("unsupported frame marker")


def _extract_frames_and_objects(grid: np.ndarray) -> tuple[int, dict[str, dict], list[dict]]:
    g = np.asarray(grid, dtype=int)
    bg = _background_color(g)
    comps = _component_list(g, bg)
    H, W = g.shape

    frame_components = [comp for comp in comps if _is_border_rectangle(comp, H, W)]
    assert len(frame_components) == 3, "expected exactly three storage frames"

    used_ids = {id(comp) for comp in frame_components}
    frames: dict[str, dict] = {}
    for frame in frame_components:
        r0, c0, r1, c1 = frame["bbox"]
        inside = [
            comp
            for comp in comps
            if id(comp) not in used_ids
            and r0 < comp["bbox"][0] <= comp["bbox"][2] < r1
            and c0 < comp["bbox"][1] <= comp["bbox"][3] < c1
        ]
        assert len(inside) == 1, "each frame needs one internal marker"
        marker = inside[0]
        used_ids.add(id(marker))
        cls = _decode_marker(marker)
        assert cls not in frames, "marker classes must be unique"
        frames[cls] = {"bbox": frame["bbox"], "color": frame["color"], "marker": marker}

    assert set(frames) == set(CLASS_ORDER)

    objects = [comp for comp in comps if id(comp) not in used_ids]
    for obj in objects:
        r0, c0, r1, c1 = obj["bbox"]
        assert (r1 - r0 + 1, c1 - c0 + 1) == (3, 3), "objects must use 3x3 bounding boxes"
    return bg, frames, objects


def _paste(out: np.ndarray, patch: np.ndarray, top: int, left: int, bg: int) -> None:
    h, w = patch.shape
    mask = patch != bg
    target = out[top : top + h, left : left + w]
    target[mask] = patch[mask]


def main(grid: np.ndarray) -> np.ndarray:
    """
    Sort each scattered 3x3 sprite into the frame whose marker encodes the same rotation-symmetry class.

    Input structure:
    - The background is azure.
    - Three hollow monochrome storage frames sit on the grid edge, each containing one tiny marker: a solid 2x2 block, a 1x2 horizontal bar, or a single pixel.
    - Elsewhere on the canvas are several scattered monochrome objects, each occupying a tight 3x3 bounding box.

    Rule:
    - Decode the frame markers as rotation classes: solid 2x2 -> 4-fold symmetry, horizontal 1x2 bar -> 2-fold symmetry, single pixel -> no nontrivial rotational symmetry.
    - Classify every scattered object by comparing its 3x3 mask with its 90-degree and 180-degree rotations.
    - Erase the objects from their original positions, then place each one into the matching frame in row-major order of appearance, keeping the original color and orientation.
    - Inside each frame, the collected objects are packed left-to-right in a single row with one azure column between neighbors, while the marker stays visible at the right side.

    Key insight:
    - The tiny symbols inside the frames are not decoration; they encode which rotational symmetry class each 3x3 shape belongs to.
    """
    g = np.asarray(grid, dtype=int)
    bg, frames, objects = _extract_frames_and_objects(g)

    ordered = sorted(objects, key=lambda comp: (int(comp["bbox"][0]), int(comp["bbox"][1])))
    grouped = {name: [] for name in CLASS_ORDER}
    for obj in ordered:
        grouped[_rotation_class(obj["patch"] != bg)].append(obj)

    out = g.copy()
    for obj in objects:
        coords = obj["coords"]
        out[coords[:, 0], coords[:, 1]] = bg

    for cls, frame in frames.items():
        r0, c0, _r1, _c1 = frame["bbox"]
        cursor = int(c0) + 1
        for obj in grouped[cls]:
            _paste(out, obj["patch"], int(r0) + 1, cursor, bg)
            cursor += obj["patch"].shape[1] + 1

    return out


def _frame_outer_width(count: int) -> int:
    return 4 * int(count) + 4


def _draw_frame(grid: np.ndarray, bbox: tuple[int, int, int, int], color: int, cls: str) -> None:
    r0, c0, r1, c1 = bbox
    grid[r0 : r1 + 1, c0] = color
    grid[r0 : r1 + 1, c1] = color
    grid[r0, c0 : c1 + 1] = color
    grid[r1, c0 : c1 + 1] = color

    marker_r = r0 + 1
    marker_c = c1 - 2
    if cls == "rot4":
        grid[marker_r : marker_r + 2, marker_c : marker_c + 2] = MARKER_COLOR
    elif cls == "rot2":
        grid[marker_r, marker_c : marker_c + 2] = MARKER_COLOR
    else:
        grid[marker_r, marker_c] = MARKER_COLOR


def _place_objects_randomly(
    grid: np.ndarray,
    objects: list[dict],
    col_lo: int,
    col_hi: int,
) -> bool:
    H, W = grid.shape
    blocked = np.zeros((H, W), dtype=bool)
    blocked[:, :col_lo] = True
    blocked[:, col_hi:] = True

    for obj in objects:
        choices: list[tuple[int, int]] = []
        for r0 in range(0, H - 2):
            for c0 in range(col_lo, col_hi - 2):
                if blocked[r0 : r0 + 3, c0 : c0 + 3].any():
                    continue
                choices.append((r0, c0))
        if not choices:
            return False
        r0, c0 = random.choice(choices)
        patch = obj["patch"]
        mask = patch != BG
        grid[r0 : r0 + 3, c0 : c0 + 3][mask] = obj["color"]

        rr0 = max(0, r0 - 1)
        rr1 = min(H, r0 + 4)
        cc0 = max(0, c0 - 1)
        cc1 = min(W, c0 + 4)
        blocked[rr0:rr1, cc0:cc1] = True
    return True


def _sample_objects(counts: dict[str, int], colors: list[int]) -> list[dict]:
    objects: list[dict] = []
    color_iter = iter(colors)
    for cls in CLASS_ORDER:
        masks = random.sample(SHAPE_CATALOG[cls], counts[cls])
        for mask in masks:
            color = next(color_iter)
            patch = np.full((3, 3), BG, dtype=int)
            patch[mask] = color
            objects.append({"class": cls, "color": color, "patch": patch})
    random.shuffle(objects)
    return objects


def _render_expected_output(inp: np.ndarray) -> np.ndarray:
    return main(inp)


def _gen_attempt() -> np.ndarray | None:
    frame_side = random.choice(FRAME_SIDE_CHOICES)
    class_rows = random.sample(list(CLASS_ORDER), len(CLASS_ORDER))
    triple = random.choice(COUNT_TRIPLES)
    counts = {cls: count for cls, count in zip(CLASS_ORDER, triple)}

    max_frame_width = max(_frame_outer_width(counts[cls]) for cls in CLASS_ORDER)
    grid_h = random.randint(17, 22)
    grid_w = max_frame_width + random.randint(8, 10)
    if grid_w > 30:
        return None

    top_margin = random.randint(0, grid_h - (3 * FRAME_HEIGHT + 2 * FRAME_GAP))
    row_positions = [top_margin + i * (FRAME_HEIGHT + FRAME_GAP) for i in range(3)]

    frame_colors = random.sample(FRAME_COLOR_POOL, 3)
    remaining_colors = [color for color in OBJECT_COLOR_POOL if color not in frame_colors]
    object_count = sum(counts.values())
    if object_count > len(remaining_colors):
        return None
    object_colors = random.sample(remaining_colors, object_count)
    objects = _sample_objects(counts, object_colors)

    grid = np.full((grid_h, grid_w), BG, dtype=int)
    frame_map: dict[str, tuple[int, int, int, int]] = {}
    frame_color_map = {cls: frame_colors[i] for i, cls in enumerate(class_rows)}
    for row_index, cls in enumerate(class_rows):
        outer_w = _frame_outer_width(counts[cls])
        r0 = row_positions[row_index]
        c0 = 0 if frame_side == "left" else grid_w - outer_w
        bbox = (r0, c0, r0 + FRAME_HEIGHT - 1, c0 + outer_w - 1)
        frame_map[cls] = bbox
        _draw_frame(grid, bbox, frame_color_map[cls], cls)

    if frame_side == "left":
        col_lo = max_frame_width + 1
        col_hi = grid_w
    else:
        col_lo = 0
        col_hi = grid_w - max_frame_width - 1
    if col_hi - col_lo < 3:
        return None

    if not _place_objects_randomly(grid, objects, col_lo, col_hi):
        return None

    try:
        out = _render_expected_output(grid)
        assert out.shape == grid.shape
    except AssertionError:
        return None
    return grid


def generate_input() -> np.ndarray:
    for _ in range(GEN_ATTEMPTS):
        grid = _gen_attempt()
        if grid is not None:
            return grid
    raise RuntimeError("failed to generate a valid symmetry-sorting puzzle")
