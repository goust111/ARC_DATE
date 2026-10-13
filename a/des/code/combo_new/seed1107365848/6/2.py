from __future__ import annotations

from collections import deque

import numpy as np


BG = 0
NEIGHBORS4 = ((1, 0), (-1, 0), (0, 1), (0, -1))
RNG = np.random.default_rng()

# Generator bounds.
GEN_ATTEMPTS = 400
SHAPE_ATTEMPTS = 200
BUNDLE_ATTEMPTS = 120
PLACEMENT_ATTEMPTS = 250
GRID_MIN = 18
GRID_MAX = 30
COLOR_COUNT_RANGE = (2, 4)  # 2-3 colors
NO_WRAP_COUNT_RANGE = (2, 4)  # 2-3 fragments
WRAP_COUNT_RANGE = (3, 5)  # 3-4 fragments
AREA_CHOICES = np.arange(3, 8)  # unique areas per color
SHAPE_MAX_H = 4
SHAPE_MAX_W = 4
INTERIOR_MAX_H = 9
INTERIOR_MAX_W = 12


def _connected_components(mask: np.ndarray) -> list[dict]:
    mask = np.asarray(mask, dtype=bool)
    h, w = mask.shape
    seen = np.zeros((h, w), dtype=bool)
    components: list[dict] = []
    for r in range(h):
        for c in range(w):
            if not mask[r, c] or seen[r, c]:
                continue
            q = deque([(int(r), int(c))])
            seen[r, c] = True
            cells: list[tuple[int, int]] = []
            while q:
                rr, cc = q.popleft()
                cells.append((int(rr), int(cc)))
                for dr, dc in NEIGHBORS4:
                    nr = int(rr + dr)
                    nc = int(cc + dc)
                    if 0 <= nr < h and 0 <= nc < w and mask[nr, nc] and not seen[nr, nc]:
                        seen[nr, nc] = True
                        q.append((nr, nc))
            coords = np.array(cells, dtype=int)
            r0, c0 = coords.min(axis=0)
            r1, c1 = coords.max(axis=0)
            components.append(
                {
                    "coords": coords,
                    "bbox": (int(r0), int(c0), int(r1), int(c1)),
                    "area": int(coords.shape[0]),
                }
            )
    return components


def _coords_to_mask(coords: np.ndarray) -> np.ndarray:
    coords = np.asarray(coords, dtype=int)
    r0, c0 = coords.min(axis=0)
    norm = coords - np.array([int(r0), int(c0)], dtype=int)
    h = int(norm[:, 0].max()) + 1
    w = int(norm[:, 1].max()) + 1
    mask = np.zeros((h, w), dtype=bool)
    mask[norm[:, 0], norm[:, 1]] = True
    return mask


def _cells_to_mask(cells: set[tuple[int, int]]) -> np.ndarray:
    coords = np.array(sorted(cells), dtype=int)
    return _coords_to_mask(coords)


def _is_rect_frame_mask(mask: np.ndarray) -> bool:
    mask = np.asarray(mask, dtype=bool)
    h, w = mask.shape
    if h < 3 or w < 3:
        return False
    expected = np.zeros((h, w), dtype=bool)
    expected[0, :] = True
    expected[-1, :] = True
    expected[:, 0] = True
    expected[:, -1] = True
    return bool(np.array_equal(mask, expected))


def _is_rect_frame_component(component: dict) -> bool:
    coords = component["coords"]
    mask = _coords_to_mask(coords)
    return _is_rect_frame_mask(mask)


def _pack_shapes(shapes: list[np.ndarray], inner_h: int, inner_w: int) -> dict | None:
    row = 0
    next_col = 0
    row_h = 0
    rows_used = 1 if shapes else 0
    positions: list[tuple[int, int]] = []
    for idx, shape in enumerate(shapes):
        h, w = map(int, shape.shape)
        if idx > 0 and next_col + w > inner_w:
            row += row_h
            next_col = 0
            row_h = 0
            rows_used += 1
        if next_col + w > inner_w:
            return None
        if row + h > inner_h:
            return None
        positions.append((int(row), int(next_col)))
        next_col += w + 1
        row_h = max(int(row_h), int(h))
    used_h = int(row + row_h) if shapes else 0
    return {"positions": positions, "used_height": used_h, "rows_used": rows_used}


def _draw_mask(grid: np.ndarray, top: int, left: int, mask: np.ndarray, color: int) -> None:
    rr, cc = np.nonzero(mask)
    grid[top + rr, left + cc] = int(color)


def _draw_frame(grid: np.ndarray, top: int, left: int, inner_h: int, inner_w: int, color: int) -> None:
    h = int(inner_h + 2)
    w = int(inner_w + 2)
    grid[top, left : left + w] = int(color)
    grid[top + h - 1, left : left + w] = int(color)
    grid[top : top + h, left] = int(color)
    grid[top : top + h, left + w - 1] = int(color)


def main(grid: np.ndarray) -> np.ndarray:
    """
    Gather each color's scattered fragments into the matching framed container, ordered by decreasing area.

    Input structure:
    - The background is black.
    - Some colors form one-pixel-thick hollow rectangular containers, with exactly one container per color.
    - Outside the containers, each such color also appears as one or more separate 4-connected monochrome fragments.

    Rule:
    - For each color, identify its rectangular container and all same-colored fragments outside it.
    - Normalize every fragment by translating it to the top-left of its own bounding box, then sort the fragments by area from largest to smallest.
    - Keep the container border, clear every outside fragment, and repack the normalized fragments inside the container from the interior's top-left, placing them left-to-right with one black column between neighbors and wrapping to the next row when needed.

    Key insight:
    - Color decides which container receives a fragment, while area decides the fragment order within that container.
    """
    g = np.asarray(grid, dtype=int)
    out = np.zeros_like(g)

    colors = [int(color) for color in np.unique(g) if int(color) != BG]
    for color in colors:
        components = _connected_components(g == int(color))
        if not components:
            continue
        frames = [component for component in components if _is_rect_frame_component(component)]
        if len(frames) != 1:
            continue

        frame = frames[0]
        fragments = [component for component in components if component is not frame]

        coords = frame["coords"]
        out[coords[:, 0], coords[:, 1]] = int(color)

        r0, c0, r1, c1 = map(int, frame["bbox"])
        inner_h = int(r1 - r0 - 1)
        inner_w = int(c1 - c0 - 1)
        ordered = sorted(fragments, key=lambda comp: (-int(comp["area"]), int(comp["bbox"][0]), int(comp["bbox"][1])))
        shapes = [_coords_to_mask(component["coords"]) for component in ordered]
        packed = _pack_shapes(shapes, inner_h, inner_w)
        if packed is None:
            continue

        for shape, (rr, cc) in zip(shapes, packed["positions"]):
            _draw_mask(out, int(r0 + 1 + rr), int(c0 + 1 + cc), shape, int(color))

    return out


def _random_transform(mask: np.ndarray) -> np.ndarray:
    out = np.array(mask, dtype=bool)
    out = np.rot90(out, int(RNG.integers(0, 4)))
    if RNG.random() < 0.5:
        out = np.fliplr(out)
    if RNG.random() < 0.5:
        out = np.flipud(out)
    return np.array(out, dtype=bool)


def _random_connected_shape(area: int) -> np.ndarray:
    for _ in range(SHAPE_ATTEMPTS):
        cells: set[tuple[int, int]] = {(0, 0)}
        while len(cells) < area:
            frontier: list[tuple[int, int]] = []
            for r, c in cells:
                for dr, dc in NEIGHBORS4:
                    nr = int(r + dr)
                    nc = int(c + dc)
                    if (nr, nc) in cells:
                        continue
                    test = set(cells)
                    test.add((nr, nc))
                    mask = _cells_to_mask(test)
                    if mask.shape[0] <= SHAPE_MAX_H and mask.shape[1] <= SHAPE_MAX_W:
                        frontier.append((nr, nc))
            if not frontier:
                break
            cells.add(frontier[int(RNG.integers(0, len(frontier)))])

        if len(cells) != area:
            continue

        mask = _random_transform(_cells_to_mask(cells))
        if int(mask.sum()) != area:
            continue
        if min(mask.shape) < 2:
            continue
        if int(mask.sum()) == int(mask.size):
            continue
        if _is_rect_frame_mask(mask):
            continue
        return np.array(mask, dtype=bool)

    raise RuntimeError(f"failed to generate a connected shape of area {area}")


def _choose_layout(shapes: list[np.ndarray], want_wrap: bool) -> dict | None:
    max_w = max(int(shape.shape[1]) for shape in shapes)
    total_w = sum(int(shape.shape[1]) for shape in shapes) + len(shapes) - 1
    candidates: list[tuple[int, dict]] = []
    upper_w = min(INTERIOR_MAX_W, total_w)
    for width in range(max_w, upper_w + 1):
        packed = _pack_shapes(shapes, INTERIOR_MAX_H, width)
        if packed is None:
            continue
        wrapped = int(packed["rows_used"]) > 1
        if wrapped != want_wrap:
            continue
        candidates.append((int(width), packed))
    if not candidates:
        return None

    width, packed = candidates[int(RNG.integers(0, len(candidates)))]
    used_h = int(packed["used_height"])
    extra_h = int(RNG.integers(0, min(3, INTERIOR_MAX_H - used_h + 1)))
    inner_h = int(used_h + extra_h)
    return {
        "inner_h": inner_h,
        "inner_w": int(width),
        "pack": _pack_shapes(shapes, inner_h, int(width)),
        "wraps": bool(want_wrap),
    }


def _build_bundle(color: int, want_wrap: bool) -> dict | None:
    count_low, count_high = WRAP_COUNT_RANGE if want_wrap else NO_WRAP_COUNT_RANGE
    for _ in range(BUNDLE_ATTEMPTS):
        count = int(RNG.integers(count_low, count_high))
        areas = sorted(RNG.choice(AREA_CHOICES, size=count, replace=False).tolist(), reverse=True)
        shapes = [_random_connected_shape(int(area)) for area in areas]
        layout = _choose_layout(shapes, want_wrap)
        if layout is None:
            continue
        return {
            "color": int(color),
            "areas": [int(area) for area in areas],
            "shapes": shapes,
            "inner_h": int(layout["inner_h"]),
            "inner_w": int(layout["inner_w"]),
            "pack": layout["pack"],
            "wraps": bool(layout["wraps"]),
        }
    return None


def _choose_grid_shape(boxes: list[tuple[int, int]]) -> tuple[int, int] | None:
    min_h = max(GRID_MIN, max(int(h) + 2 for h, _w in boxes))
    min_w = max(GRID_MIN, max(int(w) + 2 for _h, w in boxes))
    need = sum((int(h) + 2) * (int(w) + 2) for h, w in boxes)
    if need > GRID_MAX * GRID_MAX:
        return None

    for _ in range(80):
        h = int(RNG.integers(min_h, GRID_MAX + 1))
        w = int(RNG.integers(min_w, GRID_MAX + 1))
        if h * w >= need:
            return h, w

    h = min(GRID_MAX, max(min_h, int(np.ceil(np.sqrt(float(need))))))
    w = min(GRID_MAX, max(min_w, int(np.ceil(float(need) / float(h)))))
    if h <= GRID_MAX and w <= GRID_MAX and h * w >= need:
        return int(h), int(w)
    return None


def _box_clear(reserved: np.ndarray, top: int, left: int, h: int, w: int) -> bool:
    grid_h, grid_w = reserved.shape
    r0 = max(0, int(top - 1))
    c0 = max(0, int(left - 1))
    r1 = min(grid_h, int(top + h + 1))
    c1 = min(grid_w, int(left + w + 1))
    return not bool(reserved[r0:r1, c0:c1].any())


def _place_box(reserved: np.ndarray, h: int, w: int) -> tuple[int, int] | None:
    grid_h, grid_w = reserved.shape
    if h > grid_h or w > grid_w:
        return None

    for _ in range(PLACEMENT_ATTEMPTS):
        top = int(RNG.integers(0, grid_h - h + 1))
        left = int(RNG.integers(0, grid_w - w + 1))
        if _box_clear(reserved, top, left, h, w):
            reserved[top : top + h, left : left + w] = True
            return top, left

    rows = RNG.permutation(grid_h - h + 1)
    cols = RNG.permutation(grid_w - w + 1)
    for top in rows:
        for left in cols:
            if _box_clear(reserved, int(top), int(left), h, w):
                reserved[int(top) : int(top) + h, int(left) : int(left) + w] = True
                return int(top), int(left)
    return None


def _gen_attempt() -> np.ndarray | None:
    color_count = int(RNG.integers(*COLOR_COUNT_RANGE))
    colors = RNG.choice(np.arange(1, 10), size=color_count, replace=False).tolist()
    wrap_mode = bool(RNG.integers(0, 2))
    forced_wrap_index = int(RNG.integers(0, color_count)) if wrap_mode else -1

    bundles: list[dict] = []
    for idx, color in enumerate(colors):
        want_wrap = bool(wrap_mode and idx == forced_wrap_index)
        bundle = _build_bundle(int(color), want_wrap)
        if bundle is None:
            return None
        bundles.append(bundle)

    if wrap_mode and not any(bundle["wraps"] for bundle in bundles):
        return None

    boxes: list[tuple[int, int]] = []
    for bundle in bundles:
        boxes.append((int(bundle["inner_h"] + 2), int(bundle["inner_w"] + 2)))
        for shape in bundle["shapes"]:
            boxes.append(tuple(map(int, shape.shape)))

    grid_shape = _choose_grid_shape(boxes)
    if grid_shape is None:
        return None
    h, w = grid_shape

    inp = np.zeros((h, w), dtype=int)
    expected = np.zeros((h, w), dtype=int)
    reserved = np.zeros((h, w), dtype=bool)

    for bundle in sorted(bundles, key=lambda item: -int((item["inner_h"] + 2) * (item["inner_w"] + 2))):
        frame_h = int(bundle["inner_h"] + 2)
        frame_w = int(bundle["inner_w"] + 2)
        pos = _place_box(reserved, frame_h, frame_w)
        if pos is None:
            return None
        top, left = pos
        bundle["container_pos"] = (int(top), int(left))
        _draw_frame(inp, int(top), int(left), int(bundle["inner_h"]), int(bundle["inner_w"]), int(bundle["color"]))
        _draw_frame(expected, int(top), int(left), int(bundle["inner_h"]), int(bundle["inner_w"]), int(bundle["color"]))

        packed = bundle["pack"]
        if packed is None:
            return None
        for shape, (rr, cc) in zip(bundle["shapes"], packed["positions"]):
            _draw_mask(expected, int(top + 1 + rr), int(left + 1 + cc), shape, int(bundle["color"]))

    for bundle in bundles:
        order = RNG.permutation(len(bundle["shapes"]))
        for idx in order:
            shape = bundle["shapes"][int(idx)]
            sh, sw = map(int, shape.shape)
            pos = _place_box(reserved, sh, sw)
            if pos is None:
                return None
            top, left = pos
            _draw_mask(inp, int(top), int(left), shape, int(bundle["color"]))

    if not np.array_equal(main(inp), expected):
        return None
    return inp


def generate_input() -> np.ndarray:
    for _ in range(GEN_ATTEMPTS):
        grid = _gen_attempt()
        if grid is not None:
            return grid
    raise RuntimeError("failed to generate a valid packed-fragments ARC input")
