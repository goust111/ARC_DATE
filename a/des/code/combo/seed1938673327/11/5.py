from __future__ import annotations

import numpy as np


BG = 0
WALL = 1
OBJECT_COLORS = tuple(range(2, 10))
NEIGHBORS4 = ((-1, 0), (1, 0), (0, -1), (0, 1))
SIDES = ("up", "down", "left", "right")
GEN_ATTEMPTS = 400

_RNG = np.random.default_rng()


def _label_components(mask: np.ndarray) -> tuple[np.ndarray, list[np.ndarray]]:
    mask = np.asarray(mask, dtype=bool)
    h, w = mask.shape
    labels = np.zeros((h, w), dtype=int)
    components: list[np.ndarray] = []
    comp_id = 0

    for r in range(h):
        for c in range(w):
            if not mask[r, c] or labels[r, c] != 0:
                continue
            comp_id += 1
            stack = [(r, c)]
            labels[r, c] = comp_id
            coords: list[tuple[int, int]] = []
            while stack:
                rr, cc = stack.pop()
                coords.append((rr, cc))
                for dr, dc in NEIGHBORS4:
                    nr, nc = rr + dr, cc + dc
                    if 0 <= nr < h and 0 <= nc < w and mask[nr, nc] and labels[nr, nc] == 0:
                        labels[nr, nc] = comp_id
                        stack.append((nr, nc))
            components.append(np.asarray(coords, dtype=int))
    return labels, components


def _extract_objects(grid: np.ndarray) -> list[dict[str, object]]:
    objects: list[dict[str, object]] = []
    for color in np.unique(grid):
        color = int(color)
        if color in {BG, WALL}:
            continue
        _, components = _label_components(grid == color)
        for coords in components:
            r0 = int(coords[:, 0].min())
            r1 = int(coords[:, 0].max())
            c0 = int(coords[:, 1].min())
            c1 = int(coords[:, 1].max())
            crop = np.full((r1 - r0 + 1, c1 - c0 + 1), BG, dtype=int)
            crop[coords[:, 0] - r0, coords[:, 1] - c0] = color
            objects.append(
                {
                    "color": color,
                    "coords": coords,
                    "bbox": (r0, c0, r1, c1),
                    "crop": crop,
                }
            )
    objects.sort(key=lambda obj: (obj["bbox"][0], obj["bbox"][1]))
    return objects


def _touched_wall_ids(coords: np.ndarray, wall_labels: np.ndarray) -> set[int]:
    h, w = wall_labels.shape
    touched: set[int] = set()
    for r, c in coords:
        rr = int(r)
        cc = int(c)
        for dr, dc in NEIGHBORS4:
            nr, nc = rr + dr, cc + dc
            if 0 <= nr < h and 0 <= nc < w:
                wall_id = int(wall_labels[nr, nc])
                if wall_id:
                    touched.add(wall_id)
    return touched


def _repeat_crop(crop: np.ndarray, copies: int) -> np.ndarray:
    h, w = crop.shape
    out = np.full((h, copies * w + (copies - 1)), BG, dtype=int)
    for i in range(copies):
        start = i * (w + 1)
        out[:, start : start + w] = crop
    return out


def _stack_rows(rows: list[np.ndarray]) -> np.ndarray:
    if not rows:
        return np.full((1, 1), BG, dtype=int)
    out_h = sum(row.shape[0] for row in rows) + len(rows) - 1
    out_w = max(row.shape[1] for row in rows)
    out = np.full((out_h, out_w), BG, dtype=int)
    y = 0
    for idx, row in enumerate(rows):
        h, w = row.shape
        out[y : y + h, :w] = row
        y += h
        if idx + 1 < len(rows):
            y += 1
    return out


def main(grid: np.ndarray) -> np.ndarray:
    """Replicate each object by the number of distinct blue wall segments it touches."""
    g = np.asarray(grid, dtype=int)
    assert g.ndim == 2

    wall_labels, _ = _label_components(g == WALL)
    objects = _extract_objects(g)
    assert objects, "expected at least one foreground object"

    rows: list[np.ndarray] = []
    for obj in objects:
        touched = _touched_wall_ids(np.asarray(obj["coords"], dtype=int), wall_labels)
        assert touched, "every object must touch at least one wall segment"
        rows.append(_repeat_crop(np.asarray(obj["crop"], dtype=int), len(touched)))
    return _stack_rows(rows)


def _randint(low: int, high: int) -> int:
    return int(_RNG.integers(low, high + 1))


def _random_polyomino(area: int) -> np.ndarray:
    cells = {(0, 0)}
    while len(cells) < area:
        frontier: set[tuple[int, int]] = set()
        for r, c in cells:
            for dr, dc in NEIGHBORS4:
                pos = (r + dr, c + dc)
                if pos not in cells:
                    frontier.add(pos)
        nr, nc = tuple(list(frontier)[_randint(0, len(frontier) - 1)])
        cells.add((nr, nc))

    coords = np.asarray(sorted(cells), dtype=int)
    coords[:, 0] -= coords[:, 0].min()
    coords[:, 1] -= coords[:, 1].min()
    h = int(coords[:, 0].max()) + 1
    w = int(coords[:, 1].max()) + 1
    mask = np.zeros((h, w), dtype=bool)
    mask[coords[:, 0], coords[:, 1]] = True

    if _randint(0, 1):
        mask = np.fliplr(mask)
    if _randint(0, 1):
        mask = np.flipud(mask)
    mask = np.rot90(mask, _randint(0, 3))
    return np.asarray(mask, dtype=bool)


def _boundary_cells(mask: np.ndarray) -> dict[str, np.ndarray]:
    coords = np.argwhere(mask)
    return {
        "up": coords[coords[:, 0] == coords[:, 0].min()],
        "down": coords[coords[:, 0] == coords[:, 0].max()],
        "left": coords[coords[:, 1] == coords[:, 1].min()],
        "right": coords[coords[:, 1] == coords[:, 1].max()],
    }


def _make_object_block(color: int, touch_count: int) -> dict[str, object]:
    area = _randint(1, 6)
    mask = _random_polyomino(area)
    obj_h, obj_w = mask.shape
    touched_sides = tuple(_RNG.choice(np.asarray(SIDES), size=touch_count, replace=False).tolist())
    wall_lengths = {side: _randint(2, 5) for side in touched_sides}

    top_margin = wall_lengths.get("up", 0) + 1
    bottom_margin = wall_lengths.get("down", 0) + 1
    left_margin = wall_lengths.get("left", 0) + 1
    right_margin = wall_lengths.get("right", 0) + 1

    block = np.full((top_margin + obj_h + bottom_margin, left_margin + obj_w + right_margin), BG, dtype=int)
    obj_r0 = top_margin
    obj_c0 = left_margin

    crop = np.full(mask.shape, BG, dtype=int)
    crop[mask] = color
    obj_view = block[obj_r0 : obj_r0 + obj_h, obj_c0 : obj_c0 + obj_w]
    obj_view[mask] = color

    boundary = _boundary_cells(mask)
    for side in touched_sides:
        candidates = boundary[side]
        idx = _randint(0, len(candidates) - 1)
        cell_r, cell_c = map(int, candidates[idx])
        length = wall_lengths[side]

        if side == "up":
            col = obj_c0 + cell_c
            for step in range(length):
                block[obj_r0 + cell_r - 1 - step, col] = WALL
        elif side == "down":
            col = obj_c0 + cell_c
            for step in range(length):
                block[obj_r0 + cell_r + 1 + step, col] = WALL
        elif side == "left":
            row = obj_r0 + cell_r
            for step in range(length):
                block[row, obj_c0 + cell_c - 1 - step] = WALL
        else:
            row = obj_r0 + cell_r
            for step in range(length):
                block[row, obj_c0 + cell_c + 1 + step] = WALL

    return {
        "color": color,
        "touch_count": touch_count,
        "crop": crop,
        "block": block,
        "obj_top": obj_r0,
        "obj_left": obj_c0,
    }


def _expected_from_specs(specs: list[dict[str, object]]) -> np.ndarray:
    rows = [_repeat_crop(np.asarray(spec["crop"], dtype=int), int(spec["touch_count"])) for spec in specs]
    return _stack_rows(rows)


def _validate_scene(grid: np.ndarray, specs: list[dict[str, object]]) -> bool:
    objects = _extract_objects(grid)
    if len(objects) != len(specs):
        return False

    wall_labels, wall_components = _label_components(grid == WALL)
    if len(wall_components) != sum(int(spec["touch_count"]) for spec in specs):
        return False

    by_color = {int(spec["color"]): spec for spec in specs}
    if set(by_color) != {int(obj["color"]) for obj in objects}:
        return False

    for obj in objects:
        color = int(obj["color"])
        spec = by_color[color]
        if not np.array_equal(np.asarray(obj["crop"], dtype=int), np.asarray(spec["crop"], dtype=int)):
            return False
        if len(_touched_wall_ids(np.asarray(obj["coords"], dtype=int), wall_labels)) != int(spec["touch_count"]):
            return False
    return True


def _gen_attempt() -> np.ndarray | None:
    object_count = _randint(2, 5)
    colors = _RNG.choice(np.asarray(OBJECT_COLORS), size=object_count, replace=False)
    touch_counts = [_randint(1, 4) for _ in range(object_count)]
    specs = [_make_object_block(int(color), int(touch_count)) for color, touch_count in zip(colors, touch_counts)]

    gaps = [_randint(1, 3) for _ in range(object_count - 1)]
    top_margin = _randint(1, 3)
    bottom_margin = _randint(1, 3)
    grid_h = top_margin + bottom_margin + sum(int(spec["block"].shape[0]) for spec in specs) + sum(gaps)

    max_block_w = max(int(spec["block"].shape[1]) for spec in specs)
    grid_w = max_block_w + _randint(2, 10)
    if grid_h > 35 or grid_w > 35:
        return None

    grid = np.full((grid_h, grid_w), BG, dtype=int)
    y = top_margin
    ordered_specs: list[dict[str, object]] = []
    for idx, spec in enumerate(specs):
        block = np.asarray(spec["block"], dtype=int)
        block_h, block_w = block.shape
        x = _randint(1, grid_w - block_w - 1)
        sub = grid[y : y + block_h, x : x + block_w]
        if np.any((sub != BG) & (block != BG)):
            return None
        sub[block != BG] = block[block != BG]

        ordered_specs.append(
            {
                "color": int(spec["color"]),
                "crop": np.asarray(spec["crop"], dtype=int),
                "touch_count": int(spec["touch_count"]),
                "row_key": y + int(spec["obj_top"]),
                "col_key": x + int(spec["obj_left"]),
            }
        )
        y += block_h
        if idx + 1 < object_count:
            y += gaps[idx]

    ordered_specs.sort(key=lambda spec: (int(spec["row_key"]), int(spec["col_key"])))
    if not _validate_scene(grid, ordered_specs):
        return None

    expected = _expected_from_specs(ordered_specs)
    if not np.array_equal(main(grid), expected):
        return None
    return grid


def generate_input() -> np.ndarray:
    for _ in range(GEN_ATTEMPTS):
        grid = _gen_attempt()
        if grid is not None:
            return grid
    raise RuntimeError("failed to generate a valid input")
