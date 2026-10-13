import random
from collections import deque

import numpy as np


BLACK = 0
GREY = 5
COLOR_POOL = (1, 2, 3, 4, 6, 7, 8, 9)
NEIGHBORS4 = ((1, 0), (-1, 0), (0, 1), (0, -1))
GEN_RETRIES = 400


def _components(mask: np.ndarray) -> list[np.ndarray]:
    h, w = mask.shape
    seen = np.zeros((h, w), dtype=bool)
    comps: list[np.ndarray] = []
    for r in range(h):
        for c in range(w):
            if not mask[r, c] or seen[r, c]:
                continue
            q = deque([(r, c)])
            seen[r, c] = True
            coords: list[tuple[int, int]] = []
            while q:
                rr, cc = q.popleft()
                coords.append((rr, cc))
                for dr, dc in NEIGHBORS4:
                    nr, nc = rr + dr, cc + dc
                    if 0 <= nr < h and 0 <= nc < w and mask[nr, nc] and not seen[nr, nc]:
                        seen[nr, nc] = True
                        q.append((nr, nc))
            arr = np.array(coords, dtype=int)
            comps.append(arr)
    comps.sort(key=lambda arr: (int(arr[:, 0].min()), int(arr[:, 1].min()), int(arr.shape[0])))
    return comps


def _bbox(coords: np.ndarray) -> tuple[int, int, int, int]:
    return (
        int(coords[:, 0].min()),
        int(coords[:, 1].min()),
        int(coords[:, 0].max()),
        int(coords[:, 1].max()),
    )


def _parse_rooms(grid: np.ndarray) -> list[dict]:
    g = np.asarray(grid, dtype=int)
    rooms: list[dict] = []
    for coords in _components(g != GREY):
        colors = [int(g[r, c]) for r, c in coords if int(g[r, c]) != BLACK]
        assert colors, "every room must contain a non-black color"
        unique = set(colors)
        assert len(unique) == 1, "every room must have exactly one non-black color"
        color = unique.pop()
        r0, c0, r1, c1 = _bbox(coords)
        local_mask = np.zeros((r1 - r0 + 1, c1 - c0 + 1), dtype=bool)
        local_mask[coords[:, 0] - r0, coords[:, 1] - c0] = True
        rooms.append(
            {
                "coords": coords,
                "bbox": (r0, c0, r1, c1),
                "mask": local_mask,
                "color": int(color),
                "count": int(len(colors)),
            }
        )
    rooms.sort(key=lambda room: (room["bbox"][0], room["bbox"][1], room["coords"].shape[0]))
    return rooms


def _solve(grid: np.ndarray) -> np.ndarray:
    g = np.asarray(grid, dtype=int)
    rooms = _parse_rooms(g)

    templates: dict[int, np.ndarray] = {}
    legend_ids: set[int] = set()
    for idx, room in enumerate(rooms):
        if room["count"] <= 1:
            continue
        color = int(room["color"])
        assert color not in templates, "legend colors must be unique"
        r0, c0, r1, c1 = room["bbox"]
        pattern = np.full((r1 - r0 + 1, c1 - c0 + 1), BLACK, dtype=int)
        for r, c in room["coords"]:
            if int(g[r, c]) == color:
                pattern[r - r0, c - c0] = color
        templates[color] = pattern
        legend_ids.add(idx)

    assert templates, "expected at least one legend template"

    out = g.copy()
    ordinary_rooms = 0
    for idx, room in enumerate(rooms):
        if idx in legend_ids:
            continue
        ordinary_rooms += 1
        assert room["count"] == 1, "non-legend rooms must contain exactly one seed"
        color = int(room["color"])
        assert color in templates, "every seed color must appear in the legend"
        pattern = templates[color]
        ph, pw = pattern.shape
        r0, c0, _r1, _c1 = room["bbox"]
        for r, c in room["coords"]:
            out[r, c] = int(pattern[(r - r0) % ph, (c - c0) % pw])

    assert ordinary_rooms > 0, "expected at least one non-legend room"
    return out


def main(grid: np.ndarray) -> np.ndarray:
    """
    Fill each grey-walled room by tiling the legend pattern keyed by its seed color.

    Input structure:
    - The background is black, and one-cell-wide grey walls partition the grid into enclosed rooms.
    - Most non-legend rooms are black inside except for exactly one non-black, non-grey seed pixel.
    - A corner legend area contains several small grey-walled rooms, each showing a monochrome 2x2 or 3x3 template in a unique color.

    Rule:
    - Treat every maximal non-grey connected area as one room.
    - Rooms with multiple colored pixels are legend templates; their color identifies the template to store.
    - For every other room, use its seed color to choose the matching legend template, then tile that template from the room bbox's top-left corner across the room and clip it to the room shape.

    Key insight:
    - Grey walls define the true fill domain, while the single seed color inside each ordinary room is only a lookup key into the corner legend before any tiling happens.
    """
    return _solve(np.asarray(grid, dtype=int))


def _band_starts(sizes: list[int]) -> list[int]:
    starts: list[int] = []
    pos = 1
    for size in sizes:
        starts.append(pos)
        pos += size + 1
    return starts


def _draw_grid(heights: list[int], widths: list[int]) -> tuple[np.ndarray, list[int], list[int]]:
    h = sum(heights) + len(heights) + 1
    w = sum(widths) + len(widths) + 1
    grid = np.full((h, w), BLACK, dtype=int)
    grid[0, :] = GREY
    grid[-1, :] = GREY
    grid[:, 0] = GREY
    grid[:, -1] = GREY

    row_starts = _band_starts(heights)
    col_starts = _band_starts(widths)
    for r0, height in zip(row_starts, heights, strict=False):
        grid[r0 + height, :] = GREY
    for c0, width in zip(col_starts, widths, strict=False):
        grid[:, c0 + width] = GREY
    return grid, row_starts, col_starts


def _cell_bbox(
    row_starts: list[int],
    heights: list[int],
    col_starts: list[int],
    widths: list[int],
    row_idx: int,
    col_idx: int,
) -> tuple[int, int, int, int]:
    r0 = row_starts[row_idx]
    c0 = col_starts[col_idx]
    return r0, c0, r0 + heights[row_idx] - 1, c0 + widths[col_idx] - 1


def _legend_cells(
    row_count: int,
    col_count: int,
    orientation: str,
    corner: str,
    template_count: int,
) -> list[tuple[int, int]]:
    top = corner[0] == "T"
    left = corner[1] == "L"
    if orientation == "horizontal":
        row_idx = 0 if top else row_count - 1
        cols = list(range(template_count)) if left else list(range(col_count - template_count, col_count))
        return [(row_idx, col_idx) for col_idx in cols]
    col_idx = 0 if left else col_count - 1
    rows = list(range(template_count)) if top else list(range(row_count - template_count, row_count))
    return [(row_idx, col_idx) for row_idx in rows]


def _remove_wall(
    grid: np.ndarray,
    row_starts: list[int],
    heights: list[int],
    col_starts: list[int],
    widths: list[int],
    wall: tuple[str, int, int],
) -> None:
    kind, row_idx, col_idx = wall
    if kind == "v":
        r0 = row_starts[row_idx]
        c = col_starts[col_idx] + widths[col_idx]
        grid[r0 : r0 + heights[row_idx], c] = BLACK
    else:
        r = row_starts[row_idx] + heights[row_idx]
        c0 = col_starts[col_idx]
        grid[r, c0 : c0 + widths[col_idx]] = BLACK


def _pattern_library(size: int) -> list[np.ndarray]:
    total = size * size
    patterns: list[np.ndarray] = []
    for bits in range(1 << total):
        if bits & 1 == 0:
            continue
        colored = bits.bit_count()
        if colored < 2 or colored >= total:
            continue
        mask = np.zeros((size, size), dtype=bool)
        for idx in range(total):
            if (bits >> idx) & 1:
                mask[idx // size, idx % size] = True
        patterns.append(mask)
    return patterns


def _paint_room_pattern(grid: np.ndarray, coords: np.ndarray, pattern: np.ndarray, color: int) -> None:
    r0, c0, r1, c1 = _bbox(coords)
    patch = grid[r0 : r1 + 1, c0 : c1 + 1]
    patch[:] = BLACK
    patch[pattern] = color


def _classify_blank_rooms(grid: np.ndarray, legend_mask: np.ndarray) -> tuple[list[np.ndarray], list[np.ndarray]]:
    legend_rooms: list[np.ndarray] = []
    ordinary_rooms: list[np.ndarray] = []
    for coords in _components(grid != GREY):
        if np.all(legend_mask[coords[:, 0], coords[:, 1]]):
            legend_rooms.append(coords)
        else:
            ordinary_rooms.append(coords)
    legend_rooms.sort(key=lambda coords: _bbox(coords))
    ordinary_rooms.sort(key=lambda coords: _bbox(coords))
    return legend_rooms, ordinary_rooms


def _validate_generated(grid: np.ndarray, template_colors: list[int], legend_count: int) -> None:
    rooms = _parse_rooms(grid)
    legend_rooms = [room for room in rooms if room["count"] > 1]
    ordinary_rooms = [room for room in rooms if room["count"] == 1]

    assert len(legend_rooms) == legend_count, "generator must create the configured number of legend rooms"
    assert len(ordinary_rooms) >= legend_count >= 2, "generator must create enough ordinary rooms"
    assert {room["color"] for room in legend_rooms} == set(template_colors), "legend colors must match the template palette"
    assert set(template_colors) <= {room["color"] for room in ordinary_rooms}, "every template color must be referenced by at least one seed"

    out = _solve(grid)
    assert out.shape == grid.shape
    assert np.any(out != grid), "generated task should have a nontrivial output"

    for room in legend_rooms:
        coords = room["coords"]
        assert np.array_equal(out[coords[:, 0], coords[:, 1]], grid[coords[:, 0], coords[:, 1]]), "legend must stay unchanged"


def generate_input() -> np.ndarray:
    for _ in range(GEN_RETRIES):
        template_count = random.randint(2, 4)
        template_size = random.choice((2, 3))
        orientation = random.choice(("horizontal", "vertical"))
        corner = random.choice(("TL", "TR", "BL", "BR"))

        if orientation == "horizontal":
            row_count = random.randint(3, 5)
            col_count = random.randint(template_count, 6)
        else:
            row_count = random.randint(template_count, 6)
            col_count = random.randint(3, 5)

        heights = [random.randint(2, 4) for _ in range(row_count)]
        widths = [random.randint(2, 4) for _ in range(col_count)]
        legend_cells = _legend_cells(row_count, col_count, orientation, corner, template_count)
        legend_set = set(legend_cells)

        if orientation == "horizontal":
            legend_row = legend_cells[0][0]
            heights[legend_row] = template_size
            for _row_idx, col_idx in legend_cells:
                widths[col_idx] = template_size
        else:
            legend_col = legend_cells[0][1]
            widths[legend_col] = template_size
            for row_idx, _col_idx in legend_cells:
                heights[row_idx] = template_size

        grid, row_starts, col_starts = _draw_grid(heights, widths)
        if max(grid.shape) > 30:
            continue

        legend_mask = np.zeros_like(grid, dtype=bool)
        for row_idx, col_idx in legend_cells:
            r0, c0, r1, c1 = _cell_bbox(row_starts, heights, col_starts, widths, row_idx, col_idx)
            legend_mask[r0 : r1 + 1, c0 : c1 + 1] = True

        candidates: list[tuple[str, int, int]] = []
        for row_idx in range(row_count):
            for col_idx in range(col_count - 1):
                if (row_idx, col_idx) in legend_set or (row_idx, col_idx + 1) in legend_set:
                    continue
                candidates.append(("v", row_idx, col_idx))
        for row_idx in range(row_count - 1):
            for col_idx in range(col_count):
                if (row_idx, col_idx) in legend_set or (row_idx + 1, col_idx) in legend_set:
                    continue
                candidates.append(("h", row_idx, col_idx))

        merge_prob = random.choice((0.0, 0.2, 0.4, 0.6))
        opened = [wall for wall in candidates if random.random() < merge_prob]
        if candidates and merge_prob > 0.0 and not opened and random.random() < 0.8:
            opened = [random.choice(candidates)]
        for wall in opened:
            _remove_wall(grid, row_starts, heights, col_starts, widths, wall)

        legend_rooms, ordinary_rooms = _classify_blank_rooms(grid, legend_mask)
        if len(legend_rooms) != template_count:
            continue
        if len(ordinary_rooms) < template_count:
            continue

        template_colors = random.sample(COLOR_POOL, template_count)
        pattern_choices = random.sample(_pattern_library(template_size), template_count)
        for coords, color, pattern in zip(legend_rooms, template_colors, pattern_choices, strict=False):
            r0, c0, r1, c1 = _bbox(coords)
            if (r1 - r0 + 1, c1 - c0 + 1) != pattern.shape:
                break
            _paint_room_pattern(grid, coords, pattern, color)
        else:
            room_colors = template_colors.copy()
            while len(room_colors) < len(ordinary_rooms):
                room_colors.append(random.choice(template_colors))
            random.shuffle(room_colors)
            for coords, color in zip(ordinary_rooms, room_colors, strict=False):
                seed_r, seed_c = map(int, coords[random.randrange(coords.shape[0])])
                grid[seed_r, seed_c] = color

            try:
                _validate_generated(grid, template_colors, template_count)
            except AssertionError:
                continue
            return grid

    raise RuntimeError("failed to generate a valid ARC sample")
