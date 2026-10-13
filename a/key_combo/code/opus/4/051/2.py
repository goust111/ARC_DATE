from __future__ import annotations

import random
from collections import Counter

import numpy as np


BACKGROUND = 5
TRAIL_COLOR = 7
CROSS_COLOR = 2
INPUT_COLOR_POOL = [1, 3, 4, 6, 8, 9]
EDGES = ("top", "bottom", "left", "right")
DIRS4 = [(-1, 0), (1, 0), (0, -1), (0, 1)]


def get_bg(grid: np.ndarray) -> int:
    return int(Counter(map(int, np.asarray(grid).ravel())).most_common(1)[0][0])


def _bbox(coords: np.ndarray) -> tuple[int, int, int, int]:
    r0, c0 = coords.min(axis=0)
    r1, c1 = coords.max(axis=0)
    return int(r0), int(c0), int(r1), int(c1)


def _bbox_hw(box: tuple[int, int, int, int]) -> tuple[int, int]:
    r0, c0, r1, c1 = box
    return int(r1 - r0 + 1), int(c1 - c0 + 1)


def _components(mask: np.ndarray) -> list[np.ndarray]:
    h, w = map(int, mask.shape)
    seen = np.zeros((h, w), dtype=bool)
    comps: list[np.ndarray] = []
    for r in range(h):
        for c in range(w):
            if not mask[r, c] or seen[r, c]:
                continue
            stack = [(r, c)]
            seen[r, c] = True
            cells: list[tuple[int, int]] = []
            while stack:
                rr, cc = stack.pop()
                cells.append((rr, cc))
                for dr, dc in DIRS4:
                    nr, nc = rr + dr, cc + dc
                    if 0 <= nr < h and 0 <= nc < w and mask[nr, nc] and not seen[nr, nc]:
                        seen[nr, nc] = True
                        stack.append((nr, nc))
            comps.append(np.array(cells, dtype=int))
    return comps


def _find_objects(grid: np.ndarray, bg: int) -> list[dict]:
    objects: list[dict] = []
    for coords in _components(grid != bg):
        colors = {int(grid[r, c]) for r, c in coords}
        if len(colors) != 1:
            raise AssertionError("each object must be monochrome")
        color = next(iter(colors))
        objects.append(
            {
                "color": int(color),
                "coords": coords,
                "bbox": _bbox(coords),
                "area": int(coords.shape[0]),
            }
        )
    return objects


def _touching_edges(box: tuple[int, int, int, int], shape: tuple[int, int]) -> list[str]:
    h, w = map(int, shape)
    r0, c0, r1, c1 = box
    edges: list[str] = []
    if r0 == 0:
        edges.append("top")
    if r1 == h - 1:
        edges.append("bottom")
    if c0 == 0:
        edges.append("left")
    if c1 == w - 1:
        edges.append("right")
    return edges


def _is_solid_rectangle(obj: dict) -> bool:
    box = obj["bbox"]
    h, w = _bbox_hw(box)
    return int(obj["area"]) == h * w and (h >= 2 or w >= 2)


def _trail_and_destination(
    box: tuple[int, int, int, int], edge: str, shape: tuple[int, int]
) -> tuple[tuple[int, int, int, int], tuple[int, int, int, int]]:
    h, w = map(int, shape)
    r0, c0, r1, c1 = box
    obj_h, obj_w = _bbox_hw(box)
    if edge == "top":
        return (0, c0, h - 1, c1), (h - obj_h, c0, h - 1, c1)
    if edge == "bottom":
        return (0, c0, h - 1, c1), (0, c0, obj_h - 1, c1)
    if edge == "left":
        return (r0, 0, r1, w - 1), (r0, w - obj_w, r1, w - 1)
    if edge == "right":
        return (r0, 0, r1, w - 1), (r0, 0, r1, obj_w - 1)
    raise ValueError(f"unknown edge {edge}")


def _paint_box(grid: np.ndarray, box: tuple[int, int, int, int], value: int) -> None:
    r0, c0, r1, c1 = box
    grid[r0 : r1 + 1, c0 : c1 + 1] = int(value)


def _add_box(counts: np.ndarray, box: tuple[int, int, int, int]) -> None:
    r0, c0, r1, c1 = box
    counts[r0 : r1 + 1, c0 : c1 + 1] += 1


def _transform(grid: np.ndarray, *, solid_only: bool) -> np.ndarray:
    g = np.asarray(grid, dtype=int)
    bg = get_bg(g)
    out = np.full_like(g, int(bg))
    trail_counts = np.zeros(g.shape, dtype=int)
    destinations: list[tuple[tuple[int, int, int, int], int]] = []

    for obj in _find_objects(g, bg):
        edges = _touching_edges(obj["bbox"], g.shape)
        if len(edges) != 1:
            continue
        if solid_only and not _is_solid_rectangle(obj):
            continue
        trail_box, dst_box = _trail_and_destination(obj["bbox"], edges[0], g.shape)
        _add_box(trail_counts, trail_box)
        destinations.append((dst_box, int(obj["color"])))

    out[trail_counts > 0] = TRAIL_COLOR
    out[trail_counts > 1] = CROSS_COLOR
    for dst_box, color in destinations:
        _paint_box(out, dst_box, color)
    return out


def main(grid: np.ndarray) -> np.ndarray:
    """
    Slide only the edge-touching solid rectangles to the opposite side, leaving orange trails and red crossing marks.

    Input structure:
    - A grey background contains several separated monochrome objects, and each object touches exactly one outer edge.
    - Some edge objects are filled axis-aligned rectangles, while others are irregular 4-connected shapes that do not fill their bounding boxes.
    - The objects use distinct non-grey colors and include at least one rectangle from a horizontal edge and one from a vertical edge.

    Rule:
    - Identify each edge-touching object whose pixels exactly fill its bounding box; erase every non-rectangle to the grey background.
    - Slide each surviving rectangle straight to the opposite edge: top to bottom, bottom to top, left to right, right to left.
    - Paint the full swept stripe of every moving rectangle orange, recolor stripe overlaps red, then draw each destination rectangle in its original color over the stripes.

    Key insight:
    - Shape classification is the gate: only true filled rectangles move, and the red cells are not explicit objects but the overlap of horizontal and vertical sweep stripes.
    """
    return _transform(grid, solid_only=True)


def _make_rect_patch(height: int, width: int) -> np.ndarray:
    return np.ones((int(height), int(width)), dtype=bool)


def _make_nonrect_patch(edge: str) -> np.ndarray:
    height = random.randint(3, 5)
    width = random.randint(3, 5)
    patch = np.ones((height, width), dtype=bool)
    bite_h = random.randint(1, min(2, height - 1))
    bite_w = random.randint(1, min(2, width - 1))
    side = random.choice(["a", "b"])
    if edge == "top":
        if side == "a":
            patch[height - bite_h :, :bite_w] = False
        else:
            patch[height - bite_h :, width - bite_w :] = False
    elif edge == "bottom":
        if side == "a":
            patch[:bite_h, :bite_w] = False
        else:
            patch[:bite_h, width - bite_w :] = False
    elif edge == "left":
        if side == "a":
            patch[:bite_h, width - bite_w :] = False
        else:
            patch[height - bite_h :, width - bite_w :] = False
    else:
        if side == "a":
            patch[:bite_h, :bite_w] = False
        else:
            patch[height - bite_h :, :bite_w] = False
    return patch


def _candidate_positions(
    shape: tuple[int, int], patch_shape: tuple[int, int], edge: str
) -> list[tuple[int, int]]:
    h, w = map(int, shape)
    ph, pw = map(int, patch_shape)
    if ph >= h or pw >= w:
        return []
    if edge == "top":
        return [(0, c) for c in range(1, w - pw)]
    if edge == "bottom":
        return [(h - ph, c) for c in range(1, w - pw)]
    if edge == "left":
        return [(r, 0) for r in range(1, h - ph)]
    return [(r, w - pw) for r in range(1, h - ph)]


def _can_place(occupied: np.ndarray, patch: np.ndarray, r0: int, c0: int) -> bool:
    h, w = occupied.shape
    ph, pw = patch.shape
    for rr in range(ph):
        for cc in range(pw):
            if not patch[rr, cc]:
                continue
            gr, gc = r0 + rr, c0 + cc
            if occupied[gr, gc]:
                return False
            for nr in range(max(0, gr - 1), min(h, gr + 2)):
                for nc in range(max(0, gc - 1), min(w, gc + 2)):
                    if occupied[nr, nc]:
                        return False
    return True


def _place_patch(grid: np.ndarray, patch: np.ndarray, color: int, r0: int, c0: int) -> dict:
    coords = np.argwhere(patch).astype(int)
    coords[:, 0] += int(r0)
    coords[:, 1] += int(c0)
    grid[coords[:, 0], coords[:, 1]] = int(color)
    return {
        "color": int(color),
        "coords": coords,
        "bbox": _bbox(coords),
        "area": int(coords.shape[0]),
    }


def _boxes_overlap(a: tuple[int, int, int, int], b: tuple[int, int, int, int]) -> bool:
    return not (a[2] < b[0] or b[2] < a[0] or a[3] < b[1] or b[3] < a[1])


def _visible_overlap_exists(rectangles: list[dict], shape: tuple[int, int]) -> bool:
    trail_counts = np.zeros(shape, dtype=int)
    cover = np.zeros(shape, dtype=bool)
    for obj in rectangles:
        edge = _touching_edges(obj["bbox"], shape)[0]
        trail_box, dst_box = _trail_and_destination(obj["bbox"], edge, shape)
        _add_box(trail_counts, trail_box)
        r0, c0, r1, c1 = dst_box
        cover[r0 : r1 + 1, c0 : c1 + 1] = True
    return bool(np.any((trail_counts > 1) & ~cover))


def _semantic_ok(grid: np.ndarray) -> bool:
    if not isinstance(grid, np.ndarray) or grid.ndim != 2:
        return False
    if max(grid.shape) > 30 or get_bg(grid) != BACKGROUND:
        return False

    objects = _find_objects(grid, BACKGROUND)
    if len(objects) not in (5, 6):
        return False
    if len({int(obj["color"]) for obj in objects}) != len(objects):
        return False

    rectangles: list[dict] = []
    nonrectangles: list[dict] = []
    horizontal_count = 0
    vertical_count = 0
    for obj in objects:
        edges = _touching_edges(obj["bbox"], grid.shape)
        if len(edges) != 1:
            return False
        if _is_solid_rectangle(obj):
            rectangles.append(obj)
            if edges[0] in ("top", "bottom"):
                horizontal_count += 1
            else:
                vertical_count += 1
        else:
            nonrectangles.append(obj)

    if horizontal_count < 1 or vertical_count < 1 or len(nonrectangles) < 1:
        return False
    if len(rectangles) < 2:
        return False
    if not _visible_overlap_exists(rectangles, grid.shape):
        return False

    out = main(grid)
    if not np.any(out == CROSS_COLOR):
        return False
    if np.array_equal(out, _transform(grid, solid_only=False)):
        return False

    no_cross = out.copy()
    no_cross[out == CROSS_COLOR] = TRAIL_COLOR
    if np.array_equal(out, no_cross):
        return False

    for obj in nonrectangles:
        coords = obj["coords"]
        if np.any(out[coords[:, 0], coords[:, 1]] != BACKGROUND):
            return False

    return True


def _sample_primary_rectangles(shape: tuple[int, int]) -> list[tuple[np.ndarray, str]]:
    h, w = map(int, shape)
    h_edge = random.choice(["top", "bottom"])
    v_edge = random.choice(["left", "right"])
    for _ in range(200):
        h_h = random.randint(2, 4)
        h_w = random.randint(2, 5)
        v_h = random.randint(2, 5)
        v_w = random.randint(2, 4)
        if h_h >= h or h_w >= w or v_h >= h or v_w >= w:
            continue

        if h_edge == "top":
            row_min, row_max = 2, h - h_h - v_h
        else:
            row_min, row_max = h_h, h - v_h - 2
        if v_edge == "left":
            col_min, col_max = 2, w - v_w - h_w
        else:
            col_min, col_max = v_w, w - h_w - 2
        if row_min > row_max or col_min > col_max:
            continue
        return [(_make_rect_patch(h_h, h_w), h_edge), (_make_rect_patch(v_h, v_w), v_edge)]
    raise RuntimeError("failed to sample primary rectangles")


def _try_place(
    grid: np.ndarray,
    occupied: np.ndarray,
    patch: np.ndarray,
    color: int,
    edge: str,
    *,
    rectangles: list[dict],
    row_range: tuple[int, int] | None = None,
    col_range: tuple[int, int] | None = None,
    require_destination_free: bool = False,
) -> dict | None:
    positions = _candidate_positions(grid.shape, patch.shape, edge)
    if row_range is not None:
        positions = [pos for pos in positions if row_range[0] <= pos[0] <= row_range[1]]
    if col_range is not None:
        positions = [pos for pos in positions if col_range[0] <= pos[1] <= col_range[1]]
    random.shuffle(positions)
    for r0, c0 in positions:
        if not _can_place(occupied, patch, r0, c0):
            continue
        candidate = _place_patch(grid.copy(), patch, color, r0, c0)
        if require_destination_free:
            edge_name = _touching_edges(candidate["bbox"], grid.shape)[0]
            _trail_box, dst_box = _trail_and_destination(candidate["bbox"], edge_name, grid.shape)
            if any(_boxes_overlap(dst_box, _trail_and_destination(obj["bbox"], _touching_edges(obj["bbox"], grid.shape)[0], grid.shape)[1]) for obj in rectangles):
                continue
        placed = _place_patch(grid, patch, color, r0, c0)
        occupied[placed["coords"][:, 0], placed["coords"][:, 1]] = True
        return placed
    return None


def _gen_attempt() -> np.ndarray | None:
    h = random.randint(14, 18)
    w = random.randint(14, 18)
    object_count = random.choice([5, 6])
    rectangle_count = random.choice([2, 3])
    nonrect_count = object_count - rectangle_count
    colors = random.sample(INPUT_COLOR_POOL, object_count)

    grid = np.full((h, w), BACKGROUND, dtype=int)
    occupied = np.zeros((h, w), dtype=bool)
    rectangles: list[dict] = []
    objects: list[dict] = []

    try:
        primary_specs = _sample_primary_rectangles((h, w))
    except RuntimeError:
        return None

    h_patch, h_edge = primary_specs[0]
    v_patch, v_edge = primary_specs[1]
    h_h, h_w = h_patch.shape
    v_h, v_w = v_patch.shape

    if h_edge == "top":
        row_range = (2, h - h_h - v_h)
    else:
        row_range = (h_h, h - v_h - 2)
    if v_edge == "left":
        col_range = (2, w - v_w - h_w)
    else:
        col_range = (v_w, w - h_w - 2)

    first = _try_place(
        grid,
        occupied,
        h_patch,
        colors[0],
        h_edge,
        rectangles=rectangles,
        col_range=col_range,
        require_destination_free=True,
    )
    if first is None:
        return None
    rectangles.append(first)
    objects.append(first)

    second = _try_place(
        grid,
        occupied,
        v_patch,
        colors[1],
        v_edge,
        rectangles=rectangles,
        row_range=row_range,
        require_destination_free=True,
    )
    if second is None:
        return None
    rectangles.append(second)
    objects.append(second)

    extra_color_index = 2
    if rectangle_count == 3:
        for _ in range(80):
            edge = random.choice(EDGES)
            patch = _make_rect_patch(random.randint(2, 4), random.randint(2, 4))
            extra = _try_place(
                grid,
                occupied,
                patch,
                colors[extra_color_index],
                edge,
                rectangles=rectangles,
                require_destination_free=True,
            )
            if extra is None:
                continue
            rectangles.append(extra)
            objects.append(extra)
            extra_color_index += 1
            break
        else:
            return None

    for _ in range(nonrect_count):
        placed = None
        for _attempt in range(120):
            edge = random.choice(EDGES)
            patch = _make_nonrect_patch(edge)
            placed = _try_place(
                grid,
                occupied,
                patch,
                colors[extra_color_index],
                edge,
                rectangles=rectangles,
                require_destination_free=False,
            )
            if placed is not None:
                objects.append(placed)
                extra_color_index += 1
                break
        if placed is None:
            return None

    if len(objects) != object_count:
        return None
    if not _semantic_ok(grid):
        return None
    return grid


def generate_input() -> np.ndarray:
    for _ in range(500):
        grid = _gen_attempt()
        if grid is not None:
            return grid
    raise RuntimeError("failed to generate a valid input")
