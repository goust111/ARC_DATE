from __future__ import annotations

import math
import random

import numpy as np


BBox = tuple[int, int, int, int]


def _bbox_from_coords(coords: np.ndarray) -> BBox:
    rows = coords[:, 0]
    cols = coords[:, 1]
    return int(rows.min()), int(cols.min()), int(rows.max()), int(cols.max())


def _color_components(grid: np.ndarray, color: int) -> list[dict]:
    mask = grid == color
    seen = np.zeros_like(mask, dtype=bool)
    height, width = map(int, grid.shape)
    components: list[dict] = []

    for start_r, start_c in np.argwhere(mask):
        r = int(start_r)
        c = int(start_c)
        if seen[r, c]:
            continue

        stack = [(r, c)]
        seen[r, c] = True
        cells: list[tuple[int, int]] = []
        while stack:
            cr, cc = stack.pop()
            cells.append((cr, cc))
            for nr, nc in ((cr - 1, cc), (cr + 1, cc), (cr, cc - 1), (cr, cc + 1)):
                if 0 <= nr < height and 0 <= nc < width and mask[nr, nc] and not seen[nr, nc]:
                    seen[nr, nc] = True
                    stack.append((nr, nc))

        coords = np.array(cells, dtype=int)
        components.append(
            {
                "color": int(color),
                "coords": coords,
                "area": int(coords.shape[0]),
                "bbox": _bbox_from_coords(coords),
            }
        )

    return components


def _frame_cell_count(bbox: BBox) -> int:
    r0, c0, r1, c1 = bbox
    height = r1 - r0 + 1
    width = c1 - c0 + 1
    return 2 * height + 2 * width - 4


def _is_rectangular_frame(component: dict) -> bool:
    r0, c0, r1, c1 = component["bbox"]
    height = r1 - r0 + 1
    width = c1 - c0 + 1
    if height < 3 or width < 3:
        return False
    if int(component["area"]) != _frame_cell_count(component["bbox"]):
        return False

    patch = np.zeros((height, width), dtype=bool)
    coords = component["coords"]
    patch[coords[:, 0] - r0, coords[:, 1] - c0] = True
    border = np.zeros_like(patch)
    border[0, :] = True
    border[-1, :] = True
    border[:, 0] = True
    border[:, -1] = True
    return bool(np.array_equal(patch, border))


def _draw_frame(grid: np.ndarray, bbox: BBox, color: int) -> None:
    r0, c0, r1, c1 = bbox
    grid[r0, c0 : c1 + 1] = color
    grid[r1, c0 : c1 + 1] = color
    grid[r0 : r1 + 1, c0] = color
    grid[r0 : r1 + 1, c1] = color


def _interior_seed(grid: np.ndarray, bbox: BBox, bg: int, frame_color: int) -> tuple[int, int, int]:
    r0, c0, r1, c1 = bbox
    interior = grid[r0 + 1 : r1, c0 + 1 : c1]
    assert interior.size > 0, "frame interior must be non-empty"

    seed_local = np.argwhere(interior != bg)
    assert len(seed_local) == 1, "frame interior must contain exactly one seed pixel"
    seed_r = int(r0 + 1 + seed_local[0, 0])
    seed_c = int(c0 + 1 + seed_local[0, 1])
    seed_color = int(grid[seed_r, seed_c])
    assert seed_color != frame_color, "seed color must differ from its frame color"
    return seed_r, seed_c, seed_color


def _boxes_intersect(a: BBox, b: BBox, pad: int = 0) -> bool:
    return not (
        a[2] + pad < b[0]
        or b[2] + pad < a[0]
        or a[3] + pad < b[1]
        or b[3] + pad < a[1]
    )


def main(grid: np.ndarray) -> np.ndarray:
    """
    Replace each framed room by a centered solid square whose size comes from the room's interior area.

    Input structure:
    - One color is the dominant background.
    - Several one-cell-thick closed rectangular room frames appear in distinct colors.
    - Each room contains exactly one isolated single-pixel seed whose color differs from the room border.

    Rule:
    - For each rectangular room, measure the full interior area inside the border, including the seed cell and excluding the border cells.
    - Take the square root of that interior area to obtain the output square side length.
    - Erase every frame and seed, then draw a solid square centered on the seed's original position.
    - Paint the new square with the room border color, not the seed color.

    Key insight:
    - The seed contributes position only, while the surrounding frame contributes color and the frame's enclosed area determines the square size through a square-root step.
    """
    g = np.asarray(grid, dtype=int)
    assert g.ndim == 2, "expected a 2D grid"
    height, width = map(int, g.shape)

    colors, counts = np.unique(g, return_counts=True)
    bg = int(colors[int(np.argmax(counts))])
    bg_count = int(counts.max())
    other_counts = [int(count) for color, count in zip(colors, counts) if int(color) != bg]
    assert not other_counts or bg_count > max(other_counts), "background must be uniquely dominant"

    components: list[dict] = []
    for color in colors:
        color_int = int(color)
        if color_int == bg:
            continue
        components.extend(_color_components(g, color_int))

    frames = [component for component in components if _is_rectangular_frame(component)]
    assert frames, "expected at least one framed room"

    out = np.full_like(g, bg)
    covered = np.zeros_like(g, dtype=bool)
    painted = np.zeros_like(g, dtype=bool)

    for frame in frames:
        frame_color = int(frame["color"])
        bbox = frame["bbox"]
        r0, c0, r1, c1 = bbox
        seed_r, seed_c, _seed_color = _interior_seed(g, bbox, bg, frame_color)

        covered[frame["coords"][:, 0], frame["coords"][:, 1]] = True
        covered[seed_r, seed_c] = True

        interior_h = r1 - r0 - 1
        interior_w = c1 - c0 - 1
        interior_area = interior_h * interior_w
        side = math.isqrt(interior_area)
        assert side * side == interior_area, "room interior area must be a perfect square"

        top = seed_r - side // 2
        left = seed_c - side // 2
        bottom = top + side - 1
        right = left + side - 1
        assert 0 <= top <= bottom < height and 0 <= left <= right < width, "output square must stay in-bounds"
        square_bbox = (top, left, bottom, right)
        assert not np.any(painted[top : bottom + 1, left : right + 1]), "output squares must not overlap"

        out[top : bottom + 1, left : right + 1] = frame_color
        painted[top : bottom + 1, left : right + 1] = True

    assert np.array_equal(covered, g != bg), "all non-background pixels must belong to framed rooms or seeds"
    return out


# --------------- generator ---------------

GRID_MIN = 15
GRID_MAX = 30
ROOM_COUNT_CHOICES = [1, 2, 2, 3, 3, 4]
SIDE_OPTIONS_BY_COUNT = {
    1: [3, 5, 7],
    2: [3, 5, 5, 7],
    3: [3, 3, 5, 5],
    4: [3, 3, 3, 5],
}
PLACEMENT_TRIES = 300
MAX_ATTEMPTS = 600
PALETTE = list(range(10))


def _factor_pairs(area: int, max_h: int, max_w: int) -> list[tuple[int, int]]:
    pairs: list[tuple[int, int]] = []
    for h in range(1, int(math.isqrt(area)) + 1):
        if area % h != 0:
            continue
        w = area // h
        if h <= max_h and w <= max_w:
            pairs.append((h, w))
        if w != h and w <= max_h and h <= max_w:
            pairs.append((w, h))
    return pairs


def _build_spec(height: int, width: int, room_count: int, frame_color: int, seed_color: int) -> dict | None:
    side = random.choice(SIDE_OPTIONS_BY_COUNT[room_count])
    pairs = _factor_pairs(side * side, height - 2, width - 2)
    if not pairs:
        return None
    interior_h, interior_w = random.choice(pairs)
    return {
        "side": int(side),
        "interior_h": int(interior_h),
        "interior_w": int(interior_w),
        "frame_color": int(frame_color),
        "seed_color": int(seed_color),
    }


def _place_rooms(height: int, width: int, specs: list[dict]) -> list[dict] | None:
    ordered = sorted(
        specs,
        key=lambda spec: max(
            (spec["interior_h"] + 2) * (spec["interior_w"] + 2),
            spec["side"] * spec["side"],
        ),
        reverse=True,
    )
    placed: list[dict] = []
    room_boxes: list[BBox] = []
    square_boxes: list[BBox] = []

    for spec in ordered:
        interior_h = int(spec["interior_h"])
        interior_w = int(spec["interior_w"])
        frame_h = interior_h + 2
        frame_w = interior_w + 2
        side = int(spec["side"])
        half = side // 2
        done = False

        for _ in range(PLACEMENT_TRIES):
            r0 = random.randint(0, height - frame_h)
            c0 = random.randint(0, width - frame_w)
            seed_local_r = random.randint(0, interior_h - 1)
            seed_local_c = random.randint(0, interior_w - 1)
            seed_r = r0 + 1 + seed_local_r
            seed_c = c0 + 1 + seed_local_c

            top = seed_r - half
            left = seed_c - half
            bottom = top + side - 1
            right = left + side - 1
            if top < 0 or left < 0 or bottom >= height or right >= width:
                continue

            room_bbox = (r0, c0, r0 + frame_h - 1, c0 + frame_w - 1)
            square_bbox = (top, left, bottom, right)
            if any(_boxes_intersect(room_bbox, other, pad=1) for other in room_boxes):
                continue
            if any(_boxes_intersect(square_bbox, other, pad=1) for other in square_boxes):
                continue

            placed.append(
                {
                    **spec,
                    "room_bbox": room_bbox,
                    "seed": (seed_r, seed_c),
                }
            )
            room_boxes.append(room_bbox)
            square_boxes.append(square_bbox)
            done = True
            break

        if not done:
            return None

    return placed


def _expected_output(height: int, width: int, bg: int, rooms: list[dict]) -> np.ndarray:
    out = np.full((height, width), bg, dtype=int)
    for room in rooms:
        seed_r, seed_c = room["seed"]
        side = int(room["side"])
        half = side // 2
        out[seed_r - half : seed_r + half + 1, seed_c - half : seed_c + half + 1] = int(room["frame_color"])
    return out


def _build_input(height: int, width: int, bg: int, rooms: list[dict]) -> np.ndarray:
    grid = np.full((height, width), bg, dtype=int)
    for room in rooms:
        _draw_frame(grid, room["room_bbox"], int(room["frame_color"]))
        seed_r, seed_c = room["seed"]
        grid[seed_r, seed_c] = int(room["seed_color"])
    return grid


def _gen_attempt() -> np.ndarray | None:
    height = random.randint(GRID_MIN, GRID_MAX)
    width = random.randint(GRID_MIN, GRID_MAX)
    room_count = random.choice(ROOM_COUNT_CHOICES)

    bg = random.choice(PALETTE)
    non_bg = [color for color in PALETTE if color != bg]
    if 2 * room_count > len(non_bg):
        return None
    chosen = random.sample(non_bg, 2 * room_count)
    frame_colors = chosen[:room_count]
    seed_colors = chosen[room_count:]

    specs: list[dict] = []
    for frame_color, seed_color in zip(frame_colors, seed_colors):
        spec = _build_spec(height, width, room_count, frame_color, seed_color)
        if spec is None:
            return None
        specs.append(spec)

    rooms = _place_rooms(height, width, specs)
    if rooms is None:
        return None

    inp = _build_input(height, width, bg, rooms)
    colors, counts = np.unique(inp, return_counts=True)
    bg_count = int(counts[np.argmax(colors == bg)])
    other_counts = [int(count) for color, count in zip(colors, counts) if int(color) != bg]
    if other_counts and bg_count <= max(other_counts):
        return None

    expected = _expected_output(height, width, bg, rooms)
    try:
        if np.array_equal(main(inp), expected):
            return inp
    except Exception:
        return None
    return None


def generate_input() -> np.ndarray:
    for _ in range(MAX_ATTEMPTS):
        grid = _gen_attempt()
        if grid is not None:
            return grid
    raise RuntimeError("failed to generate a valid input")
