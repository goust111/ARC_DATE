from __future__ import annotations

import random

import numpy as np


BG = 0
WALL = 5
COLOR_POOL = [1, 2, 3, 4, 6, 7, 8, 9]
MAX_GENERATION_ATTEMPTS = 400


def _components(mask: np.ndarray) -> list[list[tuple[int, int]]]:
    h, w = mask.shape
    seen = np.zeros_like(mask, dtype=bool)
    comps: list[list[tuple[int, int]]] = []
    for r in range(h):
        for c in range(w):
            if not mask[r, c] or seen[r, c]:
                continue
            stack = [(r, c)]
            seen[r, c] = True
            comp: list[tuple[int, int]] = []
            while stack:
                cr, cc = stack.pop()
                comp.append((cr, cc))
                for nr, nc in ((cr - 1, cc), (cr + 1, cc), (cr, cc - 1), (cr, cc + 1)):
                    if 0 <= nr < h and 0 <= nc < w and mask[nr, nc] and not seen[nr, nc]:
                        seen[nr, nc] = True
                        stack.append((nr, nc))
            comps.append(comp)
    return comps


def _rect_wall_bbox(comp: list[tuple[int, int]]) -> tuple[int, int, int, int] | None:
    rows = [r for r, _ in comp]
    cols = [c for _, c in comp]
    r0, r1 = min(rows), max(rows)
    c0, c1 = min(cols), max(cols)
    if r1 - r0 < 2 or c1 - c0 < 2:
        return None

    expected = set()
    for c in range(c0, c1 + 1):
        expected.add((r0, c))
        expected.add((r1, c))
    for r in range(r0, r1 + 1):
        expected.add((r, c0))
        expected.add((r, c1))

    if set(comp) != expected:
        return None
    return r0, c0, r1, c1


def _find_rooms(grid: np.ndarray) -> tuple[list[dict[str, int | tuple[int, int, int, int]]], np.ndarray]:
    g = np.asarray(grid, dtype=int)
    rooms: list[dict[str, int | tuple[int, int, int, int]]] = []
    interior_mask = np.zeros_like(g, dtype=bool)

    for comp in _components(g == WALL):
        bbox = _rect_wall_bbox(comp)
        assert bbox is not None, "each grey component must be a single-pixel rectangular wall"
        r0, c0, r1, c1 = bbox
        interior = g[r0 + 1 : r1, c0 + 1 : c1]
        assert interior.size > 0, "rooms must have nonempty interiors"
        assert not np.any(interior == WALL), "room interiors must not contain grey wall pixels"

        seeds = np.argwhere((interior != BG) & (interior != WALL))
        assert len(seeds) == 1, "each room must contain exactly one seed pixel"
        sr_local, sc_local = map(int, seeds[0])
        seed_color = int(interior[sr_local, sc_local])
        assert np.all(
            (interior == BG)
            | ((np.arange(interior.shape[0])[:, None] == sr_local) & (np.arange(interior.shape[1])[None, :] == sc_local))
        ), "room interiors must otherwise be background"

        interior_mask[r0 + 1 : r1, c0 + 1 : c1] = True
        rooms.append(
            {
                "r0": r0,
                "c0": c0,
                "r1": r1,
                "c1": c1,
                "seed_r": r0 + 1 + sr_local,
                "seed_c": c0 + 1 + sc_local,
                "seed_local_r": sr_local,
                "seed_color": seed_color,
            }
        )

    assert rooms, "expected at least one rectangular room"
    rooms.sort(key=lambda room: (int(room["r0"]), int(room["c0"])))
    return rooms, interior_mask


def _find_legend(grid: np.ndarray, interior_mask: np.ndarray) -> dict[int, tuple[int, int]]:
    g = np.asarray(grid, dtype=int)
    legend_pixels = np.argwhere((g != BG) & (g != WALL) & (~interior_mask))
    assert len(legend_pixels) >= 3, "legend pixels must exist outside the rooms"

    by_row: dict[int, list[int]] = {}
    for r, c in legend_pixels:
        by_row.setdefault(int(r), []).append(int(c))

    legend: dict[int, tuple[int, int]] = {}
    for r in sorted(by_row):
        cols = sorted(by_row[r])
        assert len(cols) == 3, "each legend row must contain exactly three pixels"
        key, fill_a, fill_b = (int(g[r, c]) for c in cols)
        assert key not in legend, "legend key colors must be unique"
        legend[key] = (fill_a, fill_b)

    assert legend, "expected at least one legend entry"
    return legend


def main(grid: np.ndarray) -> np.ndarray:
    """
    Fill each grey-walled room from a top-left legend, using stripes or checkerboard based on seed height.

    Input structure:
    - The background is black, and each room is enclosed by a single-pixel grey rectangular wall.
    - Every room interior contains exactly one non-black, non-grey seed pixel.
    - A small legend sits in the top-left area; each legend row has three colored pixels:
      seed color key, fill color A, and fill color B.

    Rule:
    - Read the legend rows to build a mapping from each seed color to its two fill colors.
    - For each room, locate its seed color and fetch that room's fill colors A and B.
    - If the seed lies in the room's upper half, or on the middle interior row, fill the room
      with horizontal row stripes starting with A on the first interior row.
    - If the seed lies strictly below the middle interior row, fill the room with a checkerboard
      where cells with even local row+column parity use A and odd parity uses B.
    - The seed itself is overwritten, while the grey walls and the legend stay unchanged.

    Key insight:
    - Two independent clues must be combined: the legend chooses the pair of fill colors, and the
      seed's vertical position inside its room chooses which of two different fill patterns to apply.
    """
    g = np.asarray(grid, dtype=int)
    assert g.ndim == 2 and g.size > 0, "expected a nonempty 2D grid"

    rooms, interior_mask = _find_rooms(g)
    legend = _find_legend(g, interior_mask)

    out = g.copy()
    for room in rooms:
        seed_color = int(room["seed_color"])
        assert seed_color in legend, "every room seed must appear in the legend"
        fill_a, fill_b = legend[seed_color]
        r0, c0, r1, c1 = (int(room["r0"]), int(room["c0"]), int(room["r1"]), int(room["c1"]))
        interior_h = r1 - r0 - 1
        interior_w = c1 - c0 - 1
        seed_local_r = int(room["seed_local_r"])
        stripe_mode = seed_local_r <= (interior_h - 1) // 2

        for rr in range(interior_h):
            for cc in range(interior_w):
                if stripe_mode:
                    color = fill_a if rr % 2 == 0 else fill_b
                else:
                    color = fill_a if (rr + cc) % 2 == 0 else fill_b
                out[r0 + 1 + rr, c0 + 1 + cc] = color

    return out


def _can_place(occupied: np.ndarray, top: int, left: int, height: int, width: int, pad: int = 1) -> bool:
    h, w = occupied.shape
    r0 = max(0, top - pad)
    r1 = min(h, top + height + pad)
    c0 = max(0, left - pad)
    c1 = min(w, left + width + pad)
    return not bool(occupied[r0:r1, c0:c1].any())


def _mark_occupied(occupied: np.ndarray, top: int, left: int, height: int, width: int, pad: int = 1) -> None:
    h, w = occupied.shape
    r0 = max(0, top - pad)
    r1 = min(h, top + height + pad)
    c0 = max(0, left - pad)
    c1 = min(w, left + width + pad)
    occupied[r0:r1, c0:c1] = True


def _draw_room(
    grid: np.ndarray,
    top: int,
    left: int,
    inner_h: int,
    inner_w: int,
    seed_color: int,
    stripe_mode: bool,
    rng: random.Random,
) -> None:
    outer_h = inner_h + 2
    outer_w = inner_w + 2
    grid[top, left : left + outer_w] = WALL
    grid[top + outer_h - 1, left : left + outer_w] = WALL
    grid[top : top + outer_h, left] = WALL
    grid[top : top + outer_h, left + outer_w - 1] = WALL

    mid = (inner_h - 1) // 2
    if stripe_mode:
        seed_local_r = rng.randint(0, mid)
    else:
        seed_local_r = rng.randint(mid + 1, inner_h - 1)
    seed_local_c = rng.randrange(inner_w)
    grid[top + 1 + seed_local_r, left + 1 + seed_local_c] = seed_color


def _room_specs(rng: random.Random, room_count: int, keys: list[int]) -> list[dict[str, int | bool]]:
    stripe_count = (room_count + 1) // 2
    checker_count = room_count // 2
    branches = [True] * stripe_count + [False] * checker_count
    rng.shuffle(branches)

    seed_colors = keys.copy()
    while len(seed_colors) < room_count:
        seed_colors.append(rng.choice(keys))
    rng.shuffle(seed_colors)

    specs: list[dict[str, int | bool]] = []
    for idx in range(room_count):
        specs.append(
            {
                "inner_h": rng.randint(3, 7),
                "inner_w": rng.randint(3, 7),
                "seed_color": seed_colors[idx],
                "stripe_mode": branches[idx],
            }
        )
    specs.sort(key=lambda spec: (int(spec["inner_h"]) + 2) * (int(spec["inner_w"]) + 2), reverse=True)
    return specs


def _legend_entries(rng: random.Random, keys: list[int]) -> dict[int, tuple[int, int]]:
    entries: dict[int, tuple[int, int]] = {}
    for key in keys:
        fill_choices = [color for color in COLOR_POOL if color != key]
        fill_a, fill_b = rng.sample(fill_choices, 2)
        entries[key] = (fill_a, fill_b)
    return entries


def _place_legend(
    grid: np.ndarray,
    occupied: np.ndarray,
    legend: dict[int, tuple[int, int]],
    rng: random.Random,
) -> bool:
    rows = len(legend)
    row_step = rng.randint(1, 2)
    col_step_1 = rng.randint(1, 2)
    col_step_2 = rng.randint(1, 2)
    start_r = rng.randint(0, 2)
    start_c = rng.randint(0, 2)

    legend_h = (rows - 1) * row_step + 1
    legend_w = col_step_1 + col_step_2 + 1
    if start_r + legend_h > grid.shape[0] or start_c + legend_w > grid.shape[1]:
        return False
    if not _can_place(occupied, start_r, start_c, legend_h, legend_w, pad=2):
        return False

    for idx, key in enumerate(sorted(legend)):
        r = start_r + idx * row_step
        c0 = start_c
        c1 = start_c + col_step_1
        c2 = start_c + col_step_1 + col_step_2
        fill_a, fill_b = legend[key]
        grid[r, c0] = key
        grid[r, c1] = fill_a
        grid[r, c2] = fill_b

    _mark_occupied(occupied, start_r, start_c, legend_h, legend_w, pad=2)
    return True


def _gen_attempt(rng: random.Random) -> np.ndarray | None:
    height = rng.randint(15, 30)
    width = rng.randint(15, 30)
    grid = np.full((height, width), BG, dtype=int)
    occupied = np.zeros((height, width), dtype=bool)

    room_count = rng.randint(2, 5)
    key_count = rng.randint(2, min(4, room_count, len(COLOR_POOL)))
    keys = rng.sample(COLOR_POOL, key_count)
    legend = _legend_entries(rng, keys)
    if not _place_legend(grid, occupied, legend, rng):
        return None

    specs = _room_specs(rng, room_count, keys)
    for spec in specs:
        inner_h = int(spec["inner_h"])
        inner_w = int(spec["inner_w"])
        outer_h = inner_h + 2
        outer_w = inner_w + 2

        placed = False
        for _ in range(120):
            top = rng.randint(0, height - outer_h)
            left = rng.randint(0, width - outer_w)
            if not _can_place(occupied, top, left, outer_h, outer_w, pad=1):
                continue
            _draw_room(
                grid,
                top,
                left,
                inner_h,
                inner_w,
                int(spec["seed_color"]),
                bool(spec["stripe_mode"]),
                rng,
            )
            _mark_occupied(occupied, top, left, outer_h, outer_w, pad=1)
            placed = True
            break
        if not placed:
            return None

    rooms, interior_mask = _find_rooms(grid)
    parsed_legend = _find_legend(grid, interior_mask)
    if len(rooms) != room_count or set(parsed_legend) != set(legend):
        return None

    _ = main(grid)
    return grid


def generate_input() -> np.ndarray:
    rng = random.Random()
    for _ in range(MAX_GENERATION_ATTEMPTS):
        grid = _gen_attempt(rng)
        if grid is not None:
            return grid
    raise RuntimeError("failed to generate a valid input")
