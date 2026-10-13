from __future__ import annotations

import random

import numpy as np


BLACK = 0
GREY = 5
DRAW_COLORS = [1, 2, 3, 4, 6, 7, 8, 9]


def _trim_mask(mask: np.ndarray) -> np.ndarray:
    coords = np.argwhere(mask)
    r0, c0 = coords.min(axis=0)
    r1, c1 = coords.max(axis=0)
    return mask[r0 : r1 + 1, c0 : c1 + 1].copy()


def _find_monochrome_components(grid: np.ndarray, *, bg: int = BLACK) -> list[dict]:
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
                for dr in (-1, 0, 1):
                    for dc in (-1, 0, 1):
                        if dr == 0 and dc == 0:
                            continue
                        nr = rr + dr
                        nc = cc + dc
                        if 0 <= nr < h and 0 <= nc < w and not seen[nr, nc] and int(grid[nr, nc]) == color:
                            seen[nr, nc] = True
                            stack.append((nr, nc))

            arr = np.array(coords, dtype=int)
            r0, c0 = arr.min(axis=0)
            r1, c1 = arr.max(axis=0)
            components.append(
                {
                    "color": color,
                    "coords": arr,
                    "bbox": (int(r0), int(c0), int(r1), int(c1)),
                    "area": int(arr.shape[0]),
                }
            )

    return components


def main(grid: np.ndarray) -> np.ndarray:
    """
    Recolor each lower object by the color of the last top-row schedule cell it consumes.

    Input structure:
    - Row 0 is a color schedule whose cells are read from left to right.
    - Row 1 is a full grey separator row.
    - Below the separator, monochrome objects are scattered on a black background.

    Rule:
    - Sort the lower objects in reading order by their topmost row and then leftmost column.
    - Starting from the left end of the top row, let each object consume as many consecutive
      schedule cells as its pixel area.
    - Recolor the whole object with the color of the last schedule cell it consumed.
    - Keep the top row, the separator row, and every object's shape and position unchanged.

    Key insight:
    - The top row is a consumable tape, not a one-token-per-object legend: cumulative area
      determines which schedule cell finishes each object.
    """
    g = np.asarray(grid, dtype=int)
    assert g.ndim == 2 and 3 <= g.shape[0] <= 30 and 1 <= g.shape[1] <= 30
    assert np.all(g[1] == GREY), "expected a full grey separator row"

    schedule = g[0]
    scene = g[2:]
    objects = _find_monochrome_components(scene, bg=BLACK)
    assert objects, "expected at least one lower object"

    objects.sort(key=lambda obj: (obj["bbox"][0], obj["bbox"][1]))
    total_area = sum(obj["area"] for obj in objects)
    assert total_area <= schedule.size, "top row must cover the total consumed area"

    out = g.copy()
    consumed = 0
    for obj in objects:
        consumed += obj["area"]
        paint = int(schedule[consumed - 1])
        rr, cc = obj["coords"].T
        out[rr + 2, cc] = paint
    return out


def _random_polyomino(area: int) -> np.ndarray:
    cells = {(0, 0)}
    while len(cells) < area:
        frontier = []
        for r, c in cells:
            for dr, dc in ((-1, 0), (1, 0), (0, -1), (0, 1)):
                nxt = (r + dr, c + dc)
                if nxt not in cells:
                    frontier.append(nxt)
        cells.add(random.choice(frontier))

    coords = np.array(list(cells), dtype=int)
    coords -= coords.min(axis=0)
    h, w = coords.max(axis=0) + 1
    mask = np.zeros((int(h), int(w)), dtype=bool)
    mask[coords[:, 0], coords[:, 1]] = True

    mask = np.rot90(mask, random.randrange(4))
    if random.choice([False, True]):
        mask = np.fliplr(mask)
    if random.choice([False, True]):
        mask = np.flipud(mask)
    return _trim_mask(mask)


def _sample_areas(n_objects: int) -> list[int]:
    while True:
        areas = [1, random.randint(2, 6)]
        while len(areas) < n_objects:
            areas.append(random.randint(1, 6))
        random.shuffle(areas)
        if sum(areas) <= 24 and 1 in areas and any(area >= 3 for area in areas):
            return areas


def _can_place(occupied: np.ndarray, mask: np.ndarray, r0: int, c0: int) -> bool:
    h, w = occupied.shape
    mh, mw = mask.shape
    if r0 < 0 or c0 < 0 or r0 + mh > h or c0 + mw > w:
        return False

    for dr, dc in np.argwhere(mask):
        rr = int(r0 + dr)
        cc = int(c0 + dc)
        if occupied[max(0, rr - 1) : min(h, rr + 2), max(0, cc - 1) : min(w, cc + 2)].any():
            return False
    return True


def _place_objects(scene_h: int, width: int, masks: list[np.ndarray]) -> list[dict] | None:
    occupied = np.zeros((scene_h, width), dtype=bool)
    placements: list[dict | None] = [None] * len(masks)
    order = sorted(range(len(masks)), key=lambda i: masks[i].shape[0] * masks[i].shape[1], reverse=True)

    for idx in order:
        mask = masks[idx]
        mh, mw = mask.shape
        row_border = 1 if scene_h - mh >= 2 else 0
        col_border = 1 if width - mw >= 2 else 0
        positions = [
            (r0, c0)
            for r0 in range(row_border, scene_h - mh - row_border + 1)
            for c0 in range(col_border, width - mw - col_border + 1)
        ]
        random.shuffle(positions)

        chosen = None
        for r0, c0 in positions:
            if _can_place(occupied, mask, r0, c0):
                chosen = (r0, c0)
                break
        if chosen is None:
            return None

        r0, c0 = chosen
        rr, cc = np.argwhere(mask).T
        occupied[rr + r0, cc + c0] = True
        placements[idx] = {"mask": mask, "r0": int(r0), "c0": int(c0), "area": int(mask.sum())}

    return [item for item in placements if item is not None]


def _chunk_sizes(total: int, min_chunks: int, max_chunks: int) -> list[int]:
    if total == 0:
        return []
    low = max(1, min(min_chunks, total))
    high = max(low, min(max_chunks, total))
    n_chunks = random.randint(low, high)
    if n_chunks == 1:
        return [total]
    cuts = sorted(random.sample(range(1, total), n_chunks - 1))
    out = []
    prev = 0
    for cut in cuts + [total]:
        out.append(cut - prev)
        prev = cut
    return out


def _chunked_color_row(length: int, palette: list[int], *, min_chunks: int) -> list[int]:
    row: list[int] = []
    prev = None
    for size in _chunk_sizes(length, min_chunks=min_chunks, max_chunks=6):
        choices = [color for color in palette if color != prev]
        color = random.choice(choices)
        row.extend([color] * size)
        prev = color
    return row


def _build_schedule(width: int, ordered_areas: list[int]) -> tuple[np.ndarray, list[int]]:
    total_area = sum(ordered_areas)
    while True:
        palette = random.sample(DRAW_COLORS, 3)
        consumed_prefix = _chunked_color_row(total_area, palette, min_chunks=max(3, len(ordered_areas)))
        tail = _chunked_color_row(width - total_area, palette, min_chunks=1)
        schedule = np.array(consumed_prefix + tail, dtype=int)

        consumed = 0
        last_colors = []
        for area in ordered_areas:
            consumed += area
            last_colors.append(int(schedule[consumed - 1]))

        if len(set(last_colors)) >= 2 and len(set(consumed_prefix)) >= 2:
            return schedule, palette


def _render_grid(scene_h: int, width: int, objects: list[dict], schedule: np.ndarray, object_colors: list[int]) -> np.ndarray:
    grid = np.full((scene_h + 2, width), BLACK, dtype=int)
    grid[0] = schedule
    grid[1] = GREY

    for obj, color in zip(objects, object_colors):
        view = grid[2 + obj["r0"] : 2 + obj["r0"] + obj["mask"].shape[0], obj["c0"] : obj["c0"] + obj["mask"].shape[1]]
        view[obj["mask"]] = int(color)
    return grid


def _expected_output(inp: np.ndarray, ordered_objects: list[dict], schedule: np.ndarray) -> np.ndarray:
    out = inp.copy()
    consumed = 0
    for obj in ordered_objects:
        consumed += obj["area"]
        color = int(schedule[consumed - 1])
        view = out[2 + obj["r0"] : 2 + obj["r0"] + obj["mask"].shape[0], obj["c0"] : obj["c0"] + obj["mask"].shape[1]]
        view[obj["mask"]] = color
    return out


def _generate_candidate() -> np.ndarray | None:
    n_objects = random.randint(3, 5)
    areas = _sample_areas(n_objects)
    masks = [_random_polyomino(area) for area in areas]

    total_area = sum(areas)
    width_min = max(total_area, max(12, max(mask.shape[1] for mask in masks) + 2))
    width_max = min(30, total_area + 6)
    if width_min > width_max:
        return None
    width = random.randint(width_min, width_max)

    height_min = max(6, max(mask.shape[0] for mask in masks) + 1)
    height_max = min(14, height_min + 5)
    scene_h = random.randint(height_min, height_max)

    placed = _place_objects(scene_h, width, masks)
    if placed is None:
        return None

    placed.sort(key=lambda obj: (obj["r0"], obj["c0"]))
    ordered_areas = [obj["area"] for obj in placed]
    schedule, schedule_palette = _build_schedule(width, ordered_areas)

    object_palette = [color for color in DRAW_COLORS if color not in schedule_palette]
    if len(object_palette) < len(placed):
        return None
    object_colors = random.sample(object_palette, len(placed))

    inp = _render_grid(scene_h, width, placed, schedule, object_colors)
    expected = _expected_output(inp, placed, schedule)

    scene_components = _find_monochrome_components(inp[2:], bg=BLACK)
    scene_components.sort(key=lambda obj: (obj["bbox"][0], obj["bbox"][1]))
    if [obj["area"] for obj in scene_components] != ordered_areas:
        return None

    try:
        actual = main(inp)
    except AssertionError:
        return None
    if not np.array_equal(actual, expected):
        return None
    return inp


def generate_input() -> np.ndarray:
    for _ in range(300):
        grid = _generate_candidate()
        if grid is not None:
            return grid
    raise RuntimeError("failed to generate a valid input for puzzle 9")
