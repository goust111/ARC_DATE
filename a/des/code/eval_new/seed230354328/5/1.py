from __future__ import annotations

import random

import numpy as np


BACKGROUND = 0
SIDES = ("top", "bottom", "left", "right")
SIDE_STEPS = {
    "top": (-1, 0),
    "bottom": (1, 0),
    "left": (0, -1),
    "right": (0, 1),
}

# Generator configuration.
GRID_MIN = 18
GRID_MAX = 30
ROOM_HEIGHT_RANGE = (6, 10)
ROOM_WIDTH_RANGE = (6, 10)
MAX_SEEDS_PER_ROOM = 5
GENERATE_ATTEMPTS = 400
ROOM_PLACEMENT_ATTEMPTS = 200
SEED_LAYOUT_ATTEMPTS = 80


def _room_from_bbox(
    grid: np.ndarray, wall_color: int, r0: int, c0: int, r1: int, c1: int
) -> dict | None:
    height = r1 - r0 + 1
    width = c1 - c0 + 1
    if height < 4 or width < 4:
        return None

    interior = grid[r0 + 1 : r1, c0 + 1 : c1]
    if not np.all(interior != wall_color):
        return None

    corners = ((r0, c0), (r0, c1), (r1, c0), (r1, c1))
    if any(int(grid[r, c]) != wall_color for r, c in corners):
        return None

    top_inner = grid[r0, c0 + 1 : c1]
    bottom_inner = grid[r1, c0 + 1 : c1]
    left_inner = grid[r0 + 1 : r1, c0]
    right_inner = grid[r0 + 1 : r1, c1]

    top_gap_idx = np.flatnonzero(top_inner != wall_color)
    bottom_gap_idx = np.flatnonzero(bottom_inner != wall_color)
    left_gap_idx = np.flatnonzero(left_inner != wall_color)
    right_gap_idx = np.flatnonzero(right_inner != wall_color)
    top_gaps = [int(c0 + 1 + idx) for idx in top_gap_idx]
    bottom_gaps = [int(c0 + 1 + idx) for idx in bottom_gap_idx]
    left_gaps = [int(r0 + 1 + idx) for idx in left_gap_idx]
    right_gaps = [int(r0 + 1 + idx) for idx in right_gap_idx]
    if not (len(top_gaps) == len(bottom_gaps) == len(left_gaps) == len(right_gaps) == 1):
        return None

    if int(grid[r0, top_gaps[0]]) != BACKGROUND:
        return None
    if int(grid[r1, bottom_gaps[0]]) != BACKGROUND:
        return None
    if int(grid[left_gaps[0], c0]) != BACKGROUND:
        return None
    if int(grid[right_gaps[0], c1]) != BACKGROUND:
        return None

    if not np.any((interior != BACKGROUND) & (interior != wall_color)):
        return None

    return {
        "bbox": (int(r0), int(c0), int(r1), int(c1)),
        "wall_color": int(wall_color),
        "gaps": {
            "top": (int(r0), int(top_gaps[0])),
            "bottom": (int(r1), int(bottom_gaps[0])),
            "left": (int(left_gaps[0]), int(c0)),
            "right": (int(right_gaps[0]), int(c1)),
        },
    }


def _extract_rooms(grid: np.ndarray) -> list[dict]:
    g = np.asarray(grid, dtype=int)
    rooms: list[dict] = []
    for color in map(int, np.unique(g)):
        if color == BACKGROUND:
            continue
        positions = np.argwhere(g == color)
        if positions.size == 0:
            continue
        rows = sorted(set(map(int, positions[:, 0])))
        cols = sorted(set(map(int, positions[:, 1])))
        for row_i, r0 in enumerate(rows):
            for r1 in rows[row_i + 1 :]:
                if r1 - r0 + 1 < 4:
                    continue
                for col_i, c0 in enumerate(cols):
                    for c1 in cols[col_i + 1 :]:
                        if c1 - c0 + 1 < 4:
                            continue
                        if (
                            int(g[r0, c0]) != color
                            or int(g[r0, c1]) != color
                            or int(g[r1, c0]) != color
                            or int(g[r1, c1]) != color
                        ):
                            continue
                        room = _room_from_bbox(g, color, r0, c0, r1, c1)
                        if room is not None:
                            rooms.append(room)
    unique = {
        (room["wall_color"], *room["bbox"]): room
        for room in rooms
    }
    rooms = sorted(unique.values(), key=lambda room: room["bbox"])
    assert rooms, "expected at least one room"
    return rooms


def _room_seed_info(grid: np.ndarray, room: dict) -> tuple[np.ndarray, int]:
    r0, c0, r1, c1 = room["bbox"]
    interior = grid[r0 + 1 : r1, c0 + 1 : c1]
    seed_mask = (interior != BACKGROUND) & (interior != room["wall_color"])
    coords = np.argwhere(seed_mask)
    assert coords.size > 0, "each room must contain colored seed pixels"

    seed_values = interior[seed_mask]
    colors = np.unique(seed_values)
    assert colors.size == 1, "all seed pixels inside a room must share one color"

    abs_coords = coords + np.array([[r0 + 1, c0 + 1]])
    return abs_coords.astype(int), int(colors[0])


def _dominant_side(seed_coords: np.ndarray, room: dict) -> str:
    r0, c0, r1, c1 = room["bbox"]
    rows = seed_coords[:, 0].astype(float)
    cols = seed_coords[:, 1].astype(float)
    scores = {
        "top": float(np.mean(rows - (r0 + 1))),
        "bottom": float(np.mean((r1 - 1) - rows)),
        "left": float(np.mean(cols - (c0 + 1))),
        "right": float(np.mean((c1 - 1) - cols)),
    }
    ordered = sorted(scores.items(), key=lambda item: item[1])
    assert ordered[1][1] - ordered[0][1] > 1e-9, "seed cluster must prefer a unique side"
    return ordered[0][0]


def main(grid: np.ndarray) -> np.ndarray:
    """
    Activate the wall gap nearest the seed cluster inside each room, then grow an outward branch whose length equals the seed count.

    Input structure:
    - The background is black.
    - One or more monochrome rectangular rooms appear as one-cell-thick borders, and each side of every room has exactly one missing wall pixel.
    - Each room contains one or more interior seed pixels of a single non-black, non-wall color.

    Rule:
    - Keep every room border, every inactive gap, and every interior seed pixel unchanged.
    - For each room, determine which side the seed cluster is biased toward by comparing its average distance to the top, bottom, left, and right interior walls.
    - Use the missing wall pixel on that side as the active gap, and draw a straight line from that gap outward, perpendicular to the wall, in the seed color.
    - The branch length in pixels equals the number of seed pixels inside that room, counting the gap cell itself as the first branch pixel.

    Key insight:
    - The missing-wall positions are only candidates; the actual branch direction is encoded by which side the room's seed cluster sits closest to, while the branch length is encoded by the seed count.
    """
    g = np.asarray(grid, dtype=int)
    assert g.ndim == 2, "expected a 2D grid"
    h, w = g.shape
    out = g.copy()

    for room in _extract_rooms(g):
        seed_coords, seed_color = _room_seed_info(g, room)
        side = _dominant_side(seed_coords, room)
        gap_r, gap_c = room["gaps"][side]
        dr, dc = SIDE_STEPS[side]
        length = int(seed_coords.shape[0])

        for step in range(length):
            rr = gap_r + dr * step
            cc = gap_c + dc * step
            assert 0 <= rr < h and 0 <= cc < w, "branch must stay inside the grid"
            assert int(g[rr, cc]) == BACKGROUND, "branch corridor must be empty in the input"
            assert int(out[rr, cc]) == BACKGROUND, "branches must not overlap"
            out[rr, cc] = seed_color

    return out


def _dilate_cells(
    cells: set[tuple[int, int]], height: int, width: int, radius: int = 1
) -> set[tuple[int, int]]:
    expanded: set[tuple[int, int]] = set()
    for r, c in cells:
        for dr in range(-radius, radius + 1):
            for dc in range(-radius, radius + 1):
                nr, nc = r + dr, c + dc
                if 0 <= nr < height and 0 <= nc < width:
                    expanded.add((int(nr), int(nc)))
    return expanded


def _sample_seed_layout(interior_h: int, interior_w: int, side: str) -> list[tuple[int, int]] | None:
    for _ in range(SEED_LAYOUT_ATTEMPTS):
        if side in ("top", "bottom"):
            band_depth = random.randint(1, min(2, interior_h))
            row_range = range(band_depth) if side == "top" else range(interior_h - band_depth, interior_h)
            trim = 1 if interior_w >= 4 else 0
            col_range = range(trim, interior_w - trim) if interior_w - 2 * trim > 0 else range(interior_w)
        else:
            band_depth = random.randint(1, min(2, interior_w))
            col_range = range(band_depth) if side == "left" else range(interior_w - band_depth, interior_w)
            trim = 1 if interior_h >= 4 else 0
            row_range = range(trim, interior_h - trim) if interior_h - 2 * trim > 0 else range(interior_h)

        pool = [(r, c) for r in row_range for c in col_range]
        if len(pool) < 2:
            continue

        count = random.randint(2, min(MAX_SEEDS_PER_ROOM, len(pool)))
        chosen = random.sample(pool, count)

        if side in ("top", "bottom"):
            if len({c for _r, c in chosen}) < min(2, count):
                continue
        else:
            if len({r for r, _c in chosen}) < min(2, count):
                continue

        probe_room = {"bbox": (0, 0, interior_h + 1, interior_w + 1)}
        probe_coords = np.array([(r + 1, c + 1) for r, c in chosen], dtype=int)
        if _dominant_side(probe_coords, probe_room) == side:
            return chosen

    return None


def _relative_gap_offsets(height: int, width: int) -> dict[str, int]:
    return {
        "top": random.randint(1, width - 2),
        "bottom": random.randint(1, width - 2),
        "left": random.randint(1, height - 2),
        "right": random.randint(1, height - 2),
    }


def _branch_path(
    r0: int,
    c0: int,
    height: int,
    width: int,
    gaps: dict[str, int],
    side: str,
    length: int,
) -> list[tuple[int, int]]:
    if side == "top":
        start = (r0, c0 + gaps["top"])
    elif side == "bottom":
        start = (r0 + height - 1, c0 + gaps["bottom"])
    elif side == "left":
        start = (r0 + gaps["left"], c0)
    else:
        start = (r0 + gaps["right"], c0 + width - 1)

    dr, dc = SIDE_STEPS[side]
    return [(start[0] + dr * step, start[1] + dc * step) for step in range(length)]


def _draw_room(
    grid: np.ndarray,
    r0: int,
    c0: int,
    height: int,
    width: int,
    wall_color: int,
    gaps: dict[str, int],
    seed_cells: list[tuple[int, int]],
    seed_color: int,
) -> None:
    r1 = r0 + height - 1
    c1 = c0 + width - 1
    grid[r0, c0 : c1 + 1] = wall_color
    grid[r1, c0 : c1 + 1] = wall_color
    grid[r0 : r1 + 1, c0] = wall_color
    grid[r0 : r1 + 1, c1] = wall_color

    grid[r0, c0 + gaps["top"]] = BACKGROUND
    grid[r1, c0 + gaps["bottom"]] = BACKGROUND
    grid[r0 + gaps["left"], c0] = BACKGROUND
    grid[r0 + gaps["right"], c1] = BACKGROUND

    for sr, sc in seed_cells:
        grid[r0 + 1 + sr, c0 + 1 + sc] = seed_color


def _sample_room_specs(grid_h: int, grid_w: int) -> list[dict] | None:
    max_rooms = 4 if min(grid_h, grid_w) >= 24 else 3
    room_count = random.randint(1, max_rooms)

    wall_color = random.randint(1, 9)
    seed_palette = [color for color in range(1, 10) if color != wall_color]
    random.shuffle(seed_palette)

    specs: list[dict] = []
    reserved: set[tuple[int, int]] = set()
    for room_index in range(room_count):
        seed_color = seed_palette[room_index % len(seed_palette)]
        placed = False

        for _ in range(ROOM_PLACEMENT_ATTEMPTS):
            height = random.randint(*ROOM_HEIGHT_RANGE)
            width = random.randint(*ROOM_WIDTH_RANGE)
            if height >= grid_h - 2 or width >= grid_w - 2:
                continue

            side = random.choice(SIDES)
            seed_cells = _sample_seed_layout(height - 2, width - 2, side)
            if seed_cells is None:
                continue

            length = len(seed_cells)
            min_r0 = 1 + (length - 1 if side == "top" else 0)
            max_r0 = grid_h - height - 2 - (length - 1 if side == "bottom" else 0)
            min_c0 = 1 + (length - 1 if side == "left" else 0)
            max_c0 = grid_w - width - 2 - (length - 1 if side == "right" else 0)
            if min_r0 > max_r0 or min_c0 > max_c0:
                continue

            r0 = random.randint(min_r0, max_r0)
            c0 = random.randint(min_c0, max_c0)
            gaps = _relative_gap_offsets(height, width)
            branch = _branch_path(r0, c0, height, width, gaps, side, length)

            footprint = {
                (r, c)
                for r in range(r0, r0 + height)
                for c in range(c0, c0 + width)
            }
            footprint.update(branch)
            safety = _dilate_cells(footprint, grid_h, grid_w, radius=1)
            if footprint & reserved or safety & reserved:
                continue

            reserved.update(safety)
            specs.append(
                {
                    "r0": int(r0),
                    "c0": int(c0),
                    "height": int(height),
                    "width": int(width),
                    "wall_color": int(wall_color),
                    "gaps": gaps,
                    "seed_cells": seed_cells,
                    "seed_color": int(seed_color),
                    "side": side,
                    "branch": branch,
                }
            )
            placed = True
            break

        if not placed:
            return None

    return specs


def _expected_from_specs(grid: np.ndarray, specs: list[dict]) -> np.ndarray:
    out = grid.copy()
    for spec in specs:
        for r, c in spec["branch"]:
            out[r, c] = spec["seed_color"]
    return out


def _valid_generated_pair(grid: np.ndarray, specs: list[dict]) -> bool:
    rooms = _extract_rooms(grid)
    if len(rooms) != len(specs):
        return False

    side_hist = {side: 0 for side in SIDES}
    for room in rooms:
        coords, _color = _room_seed_info(grid, room)
        side_hist[_dominant_side(coords, room)] += 1

    if sum(side_hist.values()) != len(specs):
        return False

    expected = _expected_from_specs(grid, specs)
    try:
        actual = main(grid)
    except Exception:
        return False
    return np.array_equal(actual, expected)


def generate_input() -> np.ndarray:
    for _ in range(GENERATE_ATTEMPTS):
        grid_h = random.randint(GRID_MIN, GRID_MAX)
        grid_w = random.randint(GRID_MIN, GRID_MAX)
        specs = _sample_room_specs(grid_h, grid_w)
        if specs is None:
            continue

        grid = np.zeros((grid_h, grid_w), dtype=int)
        for spec in specs:
            _draw_room(
                grid,
                spec["r0"],
                spec["c0"],
                spec["height"],
                spec["width"],
                spec["wall_color"],
                spec["gaps"],
                spec["seed_cells"],
                spec["seed_color"],
            )

        if _valid_generated_pair(grid, specs):
            return grid

    raise RuntimeError("failed to generate a valid input")
