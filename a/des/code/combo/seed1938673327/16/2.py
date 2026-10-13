import random
from collections import deque

import numpy as np


NEIGHBORS4 = ((-1, 0), (1, 0), (0, -1), (0, 1))


def _background_color(grid: np.ndarray) -> int:
    values, counts = np.unique(grid, return_counts=True)
    if 0 in values:
        return 0
    return int(values[np.argmax(counts)])


def _find_objects(grid: np.ndarray, bg: int) -> list[dict]:
    h, w = grid.shape
    seen = np.zeros((h, w), dtype=bool)
    objects: list[dict] = []

    for r in range(h):
        for c in range(w):
            if seen[r, c] or int(grid[r, c]) == bg:
                continue

            color = int(grid[r, c])
            queue = deque([(r, c)])
            seen[r, c] = True
            cells: list[tuple[int, int]] = []

            while queue:
                rr, cc = queue.popleft()
                cells.append((rr, cc))
                for dr, dc in NEIGHBORS4:
                    nr, nc = rr + dr, cc + dc
                    if not (0 <= nr < h and 0 <= nc < w):
                        continue
                    if seen[nr, nc] or int(grid[nr, nc]) != color:
                        continue
                    seen[nr, nc] = True
                    queue.append((nr, nc))

            coords = np.array(cells, dtype=int)
            r0, c0 = coords.min(axis=0)
            r1, c1 = coords.max(axis=0)
            objects.append(
                {
                    "color": color,
                    "coords": coords,
                    "bbox": (int(r0), int(c0), int(r1), int(c1)),
                    "area": int(coords.shape[0]),
                }
            )

    return objects


def _rank_key(obj: dict) -> tuple[int, int, int, int]:
    r0, c0, r1, c1 = obj["bbox"]
    return (-int(obj["area"]), int(r0), int(c0), (r1 - r0 + 1) * (c1 - c0 + 1))


def main(grid: np.ndarray) -> np.ndarray:
    """Swap colors between horizontal and vertical objects rank-matched by area."""
    g = np.asarray(grid, dtype=int)
    bg = _background_color(g)
    objects = _find_objects(g, bg)

    horizontal: list[dict] = []
    vertical: list[dict] = []
    for obj in objects:
        r0, c0, r1, c1 = obj["bbox"]
        height = r1 - r0 + 1
        width = c1 - c0 + 1
        if width > height:
            horizontal.append(obj)
        elif height > width:
            vertical.append(obj)

    horizontal.sort(key=_rank_key)
    vertical.sort(key=_rank_key)

    new_colors = {id(obj): int(obj["color"]) for obj in objects}
    for idx in range(min(len(horizontal), len(vertical))):
        h_obj = horizontal[idx]
        v_obj = vertical[idx]
        new_colors[id(h_obj)] = int(v_obj["color"])
        new_colors[id(v_obj)] = int(h_obj["color"])

    out = g.copy()
    for obj in objects:
        coords = obj["coords"]
        out[coords[:, 0], coords[:, 1]] = new_colors[id(obj)]
    return out


# --------------- generator ---------------

BG_COLOR = 0
COLOR_POOL = list(range(1, 10))
GROUP_MODES = ("equal", "horizontal_extra", "vertical_extra")

MIN_BBOX_DIM = 2
MAX_BBOX_DIM = 6
MAX_OBJECT_AREA = 24
OBJECT_GAP = 1

GRID_SIDE_MIN = 16
GRID_SIDE_MAX = 30
GRID_EXTRA_SIDE = 6

MAX_GENERATION_ATTEMPTS = 400


def _build_feasible_specs() -> dict[str, dict[int, list[tuple[int, int]]]]:
    result = {"h": {}, "v": {}}
    for height in range(MIN_BBOX_DIM, MAX_BBOX_DIM + 1):
        for width in range(MIN_BBOX_DIM, MAX_BBOX_DIM + 1):
            if width == height:
                continue
            orientation = "h" if width > height else "v"
            min_area = height + width - 1
            max_area = min(height * width, MAX_OBJECT_AREA)
            for area in range(min_area, max_area + 1):
                result[orientation].setdefault(area, []).append((height, width))
    return result


FEASIBLE_SPECS = _build_feasible_specs()


def _sample_group_counts() -> tuple[int, int]:
    mode = random.choice(GROUP_MODES)
    if mode == "equal":
        count = random.randint(2, 4)
        return count, count
    if mode == "horizontal_extra":
        vertical = random.randint(1, 3)
        horizontal = vertical + random.randint(1, 2)
        return horizontal, vertical
    horizontal = random.randint(1, 3)
    vertical = horizontal + random.randint(1, 2)
    return horizontal, vertical


def _make_connected_mask(height: int, width: int, area: int) -> np.ndarray:
    mode = random.choice(("top", "bottom", "left", "right"))
    mask = np.zeros((height, width), dtype=bool)

    if mode == "top":
        col = random.randrange(width)
        mask[0, :] = True
        mask[:, col] = True
    elif mode == "bottom":
        col = random.randrange(width)
        mask[-1, :] = True
        mask[:, col] = True
    elif mode == "left":
        row = random.randrange(height)
        mask[:, 0] = True
        mask[row, :] = True
    else:
        row = random.randrange(height)
        mask[:, -1] = True
        mask[row, :] = True

    while int(mask.sum()) < area:
        frontier: list[tuple[int, int]] = []
        for r in range(height):
            for c in range(width):
                if mask[r, c]:
                    continue
                if any(
                    0 <= r + dr < height and 0 <= c + dc < width and mask[r + dr, c + dc]
                    for dr, dc in NEIGHBORS4
                ):
                    frontier.append((r, c))
        r, c = random.choice(frontier)
        mask[r, c] = True

    return mask


def _sample_specs(count: int, orientation: str) -> list[dict] | None:
    area_pool = list(FEASIBLE_SPECS[orientation].keys())
    if len(area_pool) < count:
        return None

    areas = random.sample(area_pool, count)
    specs = []
    for area in areas:
        height, width = random.choice(FEASIBLE_SPECS[orientation][area])
        specs.append(
            {
                "orientation": orientation,
                "area": int(area),
                "height": int(height),
                "width": int(width),
                "mask": _make_connected_mask(height, width, int(area)),
            }
        )
    return specs


def _size_bounds(specs: list[dict]) -> tuple[int, int] | None:
    padded_area = sum((spec["height"] + OBJECT_GAP) * (spec["width"] + OBJECT_GAP) for spec in specs)
    min_side = max(
        GRID_SIDE_MIN,
        max(spec["height"] for spec in specs) + 2,
        max(spec["width"] for spec in specs) + 2,
        int(np.ceil(np.sqrt(padded_area))) + 2,
    )
    max_side = min(GRID_SIDE_MAX, min_side + GRID_EXTRA_SIDE)
    if min_side > max_side:
        return None
    return min_side, max_side


def _valid_positions(occupied: np.ndarray, height: int, width: int) -> list[tuple[int, int]]:
    h, w = occupied.shape
    positions = []
    for r in range(h - height + 1):
        r0 = max(0, r - OBJECT_GAP)
        r1 = min(h, r + height + OBJECT_GAP)
        for c in range(w - width + 1):
            c0 = max(0, c - OBJECT_GAP)
            c1 = min(w, c + width + OBJECT_GAP)
            if not occupied[r0:r1, c0:c1].any():
                positions.append((r, c))
    return positions


def _mark_occupied(occupied: np.ndarray, top: int, left: int, height: int, width: int) -> None:
    h, w = occupied.shape
    r0 = max(0, top - OBJECT_GAP)
    r1 = min(h, top + height + OBJECT_GAP)
    c0 = max(0, left - OBJECT_GAP)
    c1 = min(w, left + width + OBJECT_GAP)
    occupied[r0:r1, c0:c1] = True


def _place_specs(specs: list[dict]) -> np.ndarray | None:
    side_bounds = _size_bounds(specs)
    if side_bounds is None:
        return None
    min_side, max_side = side_bounds

    height = random.randint(min_side, max_side)
    width = random.randint(min_side, max_side)
    grid = np.full((height, width), BG_COLOR, dtype=int)
    occupied = np.zeros((height, width), dtype=bool)

    for spec in sorted(specs, key=lambda item: item["height"] * item["width"], reverse=True):
        positions = _valid_positions(occupied, spec["height"], spec["width"])
        if not positions:
            return None
        top, left = random.choice(positions)
        coords = np.argwhere(spec["mask"]) + np.array([top, left], dtype=int)
        grid[coords[:, 0], coords[:, 1]] = int(spec["color"])
        spec["top"] = int(top)
        spec["left"] = int(left)
        spec["coords"] = coords
        _mark_occupied(occupied, top, left, spec["height"], spec["width"])

    return grid


def _expected_output(grid: np.ndarray, specs: list[dict]) -> np.ndarray:
    out = grid.copy()
    horizontal = sorted(
        [spec for spec in specs if spec["orientation"] == "h"],
        key=lambda spec: (-int(spec["area"]), int(spec["top"]), int(spec["left"])),
    )
    vertical = sorted(
        [spec for spec in specs if spec["orientation"] == "v"],
        key=lambda spec: (-int(spec["area"]), int(spec["top"]), int(spec["left"])),
    )

    new_colors = {spec["index"]: int(spec["color"]) for spec in specs}
    for idx in range(min(len(horizontal), len(vertical))):
        h_spec = horizontal[idx]
        v_spec = vertical[idx]
        new_colors[h_spec["index"]] = int(v_spec["color"])
        new_colors[v_spec["index"]] = int(h_spec["color"])

    for spec in specs:
        coords = spec["coords"]
        out[coords[:, 0], coords[:, 1]] = new_colors[spec["index"]]
    return out


def _gen_attempt() -> np.ndarray | None:
    n_horizontal, n_vertical = _sample_group_counts()
    horizontal_specs = _sample_specs(n_horizontal, "h")
    vertical_specs = _sample_specs(n_vertical, "v")
    if horizontal_specs is None or vertical_specs is None:
        return None

    specs = horizontal_specs + vertical_specs
    colors = random.sample(COLOR_POOL, len(specs))
    random.shuffle(specs)
    for idx, (spec, color) in enumerate(zip(specs, colors)):
        spec["index"] = idx
        spec["color"] = int(color)

    grid = _place_specs(specs)
    if grid is None:
        return None

    expected = _expected_output(grid, specs)
    if np.array_equal(expected, grid):
        return None
    if not np.array_equal(main(grid), expected):
        return None
    return grid


def generate_input() -> np.ndarray:
    for _ in range(MAX_GENERATION_ATTEMPTS):
        grid = _gen_attempt()
        if grid is not None:
            return grid
    raise RuntimeError(f"failed to generate valid input after {MAX_GENERATION_ATTEMPTS} attempts")
