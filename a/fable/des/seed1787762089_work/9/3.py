import random
from typing import Iterable

import numpy as np


BLACK = 0
BLUE = 1
RED = 2
GREEN = 3
YELLOW = 4
MAGENTA = 6
AZURE = 8

FRAME_COLORS = (BLUE, YELLOW, AZURE, MAGENTA)
GRID_SIZE_RANGE = (19, 23)
FRAME_SIZE_VALUES = (5, 7)
MAX_DOTS_PER_COLOR = 5
MAX_RAY = 5
GENERATE_RETRIES = 400
PLACEMENT_RETRIES = 200


def _components_mask(mask: np.ndarray) -> list[np.ndarray]:
    mask = np.asarray(mask, dtype=bool)
    height, width = mask.shape
    seen = np.zeros_like(mask, dtype=bool)
    components: list[list[tuple[int, int]]] = []

    for r in range(height):
        for c in range(width):
            if not mask[r, c] or seen[r, c]:
                continue
            stack = [(r, c)]
            seen[r, c] = True
            coords: list[tuple[int, int]] = []
            while stack:
                cr, cc = stack.pop()
                coords.append((cr, cc))
                for nr, nc in ((cr - 1, cc), (cr + 1, cc), (cr, cc - 1), (cr, cc + 1)):
                    if 0 <= nr < height and 0 <= nc < width and mask[nr, nc] and not seen[nr, nc]:
                        seen[nr, nc] = True
                        stack.append((nr, nc))
            components.append(coords)

    return [np.array(coords, dtype=int) for coords in components]


def _bbox(coords: np.ndarray) -> tuple[int, int, int, int]:
    rows = coords[:, 0]
    cols = coords[:, 1]
    return int(rows.min()), int(cols.min()), int(rows.max()), int(cols.max())


def _frame_bbox(grid: np.ndarray, color: int, coords: np.ndarray) -> tuple[int, int, int, int] | None:
    if coords.shape[0] < 16:
        return None
    r0, c0, r1, c1 = _bbox(coords)
    height = r1 - r0 + 1
    width = c1 - c0 + 1
    if min(height, width) < 5 or height % 2 == 0 or width % 2 == 0:
        return None
    if r0 == 0 or c0 == 0 or r1 == grid.shape[0] - 1 or c1 == grid.shape[1] - 1:
        return None
    if coords.shape[0] != 2 * (height + width) - 4:
        return None

    window = grid[r0 : r1 + 1, c0 : c1 + 1]
    if not np.all(window[0, :] == color):
        return None
    if not np.all(window[-1, :] == color):
        return None
    if not np.all(window[:, 0] == color):
        return None
    if not np.all(window[:, -1] == color):
        return None
    if np.any(window[1:-1, 1:-1] == color):
        return None
    return r0, c0, r1, c1


def _parse_scene(grid: np.ndarray) -> list[dict]:
    height, width = grid.shape
    objects: list[dict] = []

    for color in map(int, np.unique(grid)):
        if color in (BLACK, RED, GREEN):
            continue

        frame_boxes: list[tuple[int, int, int, int]] = []
        markers: list[tuple[int, int]] = []
        for coords in _components_mask(grid == color):
            frame = _frame_bbox(grid, color, coords)
            if frame is not None:
                frame_boxes.append(frame)
            elif coords.shape[0] == 1:
                r, c = map(int, coords[0])
                if r in (0, height - 1) or c in (0, width - 1):
                    markers.append((r, c))

        if not frame_boxes:
            continue

        assert len(frame_boxes) == 1, "each frame color must define exactly one container"
        assert len(markers) == 1, "each frame color must define exactly one boundary marker"

        r0, c0, r1, c1 = frame_boxes[0]
        center_r = (r0 + r1) // 2
        center_c = (c0 + c1) // 2
        marker_r, marker_c = markers[0]

        if marker_r == center_r and marker_c in (0, width - 1):
            axis = "h"
            toward = (0, -1) if marker_c == 0 else (0, 1)
        elif marker_c == center_c and marker_r in (0, height - 1):
            axis = "v"
            toward = (-1, 0) if marker_r == 0 else (1, 0)
        else:
            raise AssertionError("marker must align with the container center row or column")

        objects.append(
            {
                "color": color,
                "bbox": (r0, c0, r1, c1),
                "axis": axis,
                "toward": toward,
            }
        )

    assert 2 <= len(objects) <= 3, "expected two or three containers"
    return objects


def _draw_ray(
    out: np.ndarray,
    bbox: tuple[int, int, int, int],
    axis: str,
    direction: tuple[int, int],
    length: int,
    color: int,
) -> None:
    if length <= 0:
        return

    r0, c0, r1, c1 = bbox
    dr, dc = direction
    if axis == "h":
        row = (r0 + r1) // 2
        col = c1 + 1 if dc > 0 else c0 - 1
        start_r, start_c = row, col
    else:
        col = (c0 + c1) // 2
        row = r1 + 1 if dr > 0 else r0 - 1
        start_r, start_c = row, col

    height, width = out.shape
    for step in range(length):
        rr = start_r + dr * step
        cc = start_c + dc * step
        assert 0 <= rr < height and 0 <= cc < width
        assert out[rr, cc] == BLACK, "ray path must stay on empty background"
        out[rr, cc] = color


def _transform(grid: np.ndarray) -> np.ndarray:
    g = np.asarray(grid, dtype=int)
    assert g.ndim == 2, "expected a 2D grid"

    out = g.copy()
    for obj in _parse_scene(g):
        r0, c0, r1, c1 = obj["bbox"]
        inner = g[r0 + 1 : r1, c0 + 1 : c1]
        assert set(map(int, np.unique(inner))) <= {BLACK, RED, GREEN}

        green_count = int(np.count_nonzero(inner == GREEN))
        red_count = int(np.count_nonzero(inner == RED))
        delta = green_count - red_count
        if delta == 0:
            continue

        toward = obj["toward"]
        move_dir = toward if delta > 0 else (-toward[0], -toward[1])
        ray_color = GREEN if delta > 0 else RED
        _draw_ray(out, obj["bbox"], obj["axis"], move_dir, abs(delta), ray_color)

    return out


def main(grid: np.ndarray) -> np.ndarray:
    return _transform(grid)


def _sample_counts(sign: int) -> tuple[int, int]:
    diff = random.randint(1, MAX_RAY)
    if sign > 0:
        red_count = random.randint(0, MAX_DOTS_PER_COLOR - diff)
        green_count = red_count + diff
    else:
        green_count = random.randint(0, MAX_DOTS_PER_COLOR - diff)
        red_count = green_count + diff
    return green_count, red_count


def _size_options(total_dots: int) -> list[tuple[int, int]]:
    options = []
    for height in FRAME_SIZE_VALUES:
        for width in FRAME_SIZE_VALUES:
            if (height - 2) * (width - 2) >= total_dots:
                options.append((height, width))
    return options


def _bbox_ranges(height: int, width: int, frame_h: int, frame_w: int, direction: str) -> tuple[range, range]:
    if direction in ("left", "right"):
        row_range = range(1, height - frame_h)
        col_range = range(MAX_RAY + 1, width - frame_w - MAX_RAY)
    else:
        row_range = range(MAX_RAY + 1, height - frame_h - MAX_RAY)
        col_range = range(1, width - frame_w)
    return row_range, col_range


def _marker_for_bbox(height: int, width: int, bbox: tuple[int, int, int, int], direction: str) -> tuple[int, int]:
    r0, c0, r1, c1 = bbox
    center_r = (r0 + r1) // 2
    center_c = (c0 + c1) // 2
    if direction == "left":
        return center_r, 0
    if direction == "right":
        return center_r, width - 1
    if direction == "up":
        return 0, center_c
    return height - 1, center_c


def _reserved_cells(height: int, width: int, bbox: tuple[int, int, int, int], direction: str) -> set[tuple[int, int]]:
    r0, c0, r1, c1 = bbox
    center_r = (r0 + r1) // 2
    center_c = (c0 + c1) // 2
    cells: set[tuple[int, int]] = set()

    for rr in range(max(0, r0 - 1), min(height, r1 + 2)):
        for cc in range(max(0, c0 - 1), min(width, c1 + 2)):
            cells.add((rr, cc))

    if direction == "left":
        for cc in range(1, c0):
            cells.add((center_r, cc))
        for cc in range(c1 + 1, min(width - 1, c1 + MAX_RAY + 1)):
            cells.add((center_r, cc))
    elif direction == "right":
        for cc in range(c1 + 1, width - 1):
            cells.add((center_r, cc))
        for cc in range(max(1, c0 - MAX_RAY), c0):
            cells.add((center_r, cc))
    elif direction == "up":
        for rr in range(1, r0):
            cells.add((rr, center_c))
        for rr in range(r1 + 1, min(height - 1, r1 + MAX_RAY + 1)):
            cells.add((rr, center_c))
    else:
        for rr in range(r1 + 1, height - 1):
            cells.add((rr, center_c))
        for rr in range(max(1, r0 - MAX_RAY), r0):
            cells.add((rr, center_c))

    cells.add(_marker_for_bbox(height, width, bbox, direction))
    return cells


def _too_close_to_markers(marker: tuple[int, int], others: Iterable[tuple[int, int]]) -> bool:
    mr, mc = marker
    for orow, ocol in others:
        if max(abs(mr - orow), abs(mc - ocol)) <= 1:
            return True
    return False


def _direction_to_vector(direction: str) -> tuple[int, int]:
    if direction == "left":
        return 0, -1
    if direction == "right":
        return 0, 1
    if direction == "up":
        return -1, 0
    return 1, 0


def _render_expected(grid: np.ndarray, placed: list[dict]) -> np.ndarray:
    out = grid.copy()
    for spec in placed:
        delta = spec["green_count"] - spec["red_count"]
        toward = _direction_to_vector(spec["direction"])
        move_dir = toward if delta > 0 else (-toward[0], -toward[1])
        ray_color = GREEN if delta > 0 else RED
        _draw_ray(out, spec["bbox"], spec["axis"], move_dir, abs(delta), ray_color)
    return out


def _gen_attempt() -> np.ndarray | None:
    height = random.randint(*GRID_SIZE_RANGE)
    width = random.randint(*GRID_SIZE_RANGE)
    container_count = random.choice((2, 3))

    colors = random.sample(FRAME_COLORS, container_count)
    signs = [1, -1]
    if container_count == 3:
        signs.append(random.choice((-1, 1)))
    random.shuffle(signs)

    axes = ["h", "v"]
    if container_count == 3:
        axes.append(random.choice(("h", "v")))
    random.shuffle(axes)

    specs: list[dict] = []
    for color, sign, axis in zip(colors, signs, axes):
        green_count, red_count = _sample_counts(sign)
        direction = random.choice(("left", "right")) if axis == "h" else random.choice(("up", "down"))
        sizes = _size_options(green_count + red_count)
        if not sizes:
            return None
        frame_h, frame_w = random.choice(sizes)
        specs.append(
            {
                "color": color,
                "axis": axis,
                "direction": direction,
                "green_count": green_count,
                "red_count": red_count,
                "frame_h": frame_h,
                "frame_w": frame_w,
            }
        )

    specs.sort(key=lambda spec: (spec["frame_h"] * spec["frame_w"], spec["axis"] == "v"), reverse=True)

    occupied = np.zeros((height, width), dtype=bool)
    markers: list[tuple[int, int]] = []
    placed: list[dict] = []

    for spec in specs:
        frame_h = spec["frame_h"]
        frame_w = spec["frame_w"]
        row_range, col_range = _bbox_ranges(height, width, frame_h, frame_w, spec["direction"])
        if not row_range or not col_range:
            return None

        found = False
        for _ in range(PLACEMENT_RETRIES):
            r0 = random.choice(row_range)
            c0 = random.choice(col_range)
            bbox = (r0, c0, r0 + frame_h - 1, c0 + frame_w - 1)
            marker = _marker_for_bbox(height, width, bbox, spec["direction"])
            reserved = _reserved_cells(height, width, bbox, spec["direction"])
            if _too_close_to_markers(marker, markers):
                continue
            if any(occupied[r, c] for r, c in reserved):
                continue

            for r, c in reserved:
                occupied[r, c] = True
            spec = dict(spec)
            spec["bbox"] = bbox
            placed.append(spec)
            markers.append(marker)
            found = True
            break

        if not found:
            return None

    grid = np.full((height, width), BLACK, dtype=int)
    for spec in placed:
        color = spec["color"]
        r0, c0, r1, c1 = spec["bbox"]
        grid[r0, c0 : c1 + 1] = color
        grid[r1, c0 : c1 + 1] = color
        grid[r0 : r1 + 1, c0] = color
        grid[r0 : r1 + 1, c1] = color
        marker_r, marker_c = _marker_for_bbox(height, width, spec["bbox"], spec["direction"])
        grid[marker_r, marker_c] = color

        interior_cells = [(r, c) for r in range(r0 + 1, r1) for c in range(c0 + 1, c1)]
        random.shuffle(interior_cells)
        total = spec["green_count"] + spec["red_count"]
        chosen = interior_cells[:total]
        for r, c in chosen[: spec["green_count"]]:
            grid[r, c] = GREEN
        for r, c in chosen[spec["green_count"] :]:
            grid[r, c] = RED

    expected = _render_expected(grid, placed)
    try:
        actual = main(grid)
    except AssertionError:
        return None
    if not np.array_equal(actual, expected):
        return None
    return grid


def generate_input() -> np.ndarray:
    for _ in range(GENERATE_RETRIES):
        grid = _gen_attempt()
        if grid is not None:
            return grid
    raise RuntimeError("failed to generate a valid input")
