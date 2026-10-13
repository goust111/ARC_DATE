import random
from collections import deque

import numpy as np


BLACK = 0
GREY = 5
MAROON = 9
PALETTE_POOL = (1, 2, 3, 4, 6, 7, 8)

GRID_H_RANGE = (18, 30)
GRID_W_RANGE = (18, 30)
ROOM_COUNT_OPTIONS = (2, 3, 4)
ROOM_COUNT_WEIGHTS = (2, 5, 2)
INNER_H_RANGE = (1, 5)
INNER_W_RANGE = (1, 6)
GEN_ATTEMPTS = 400
ROOM_PLACEMENT_TRIES = 120


def _components(mask: np.ndarray) -> list[list[tuple[int, int]]]:
    h, w = mask.shape
    seen = np.zeros((h, w), dtype=bool)
    comps: list[list[tuple[int, int]]] = []
    for r in range(h):
        for c in range(w):
            if not mask[r, c] or seen[r, c]:
                continue
            seen[r, c] = True
            q = deque([(r, c)])
            pts: list[tuple[int, int]] = []
            while q:
                cr, cc = q.popleft()
                pts.append((cr, cc))
                for dr, dc in ((1, 0), (-1, 0), (0, 1), (0, -1)):
                    nr, nc = cr + dr, cc + dc
                    if 0 <= nr < h and 0 <= nc < w and mask[nr, nc] and not seen[nr, nc]:
                        seen[nr, nc] = True
                        q.append((nr, nc))
            comps.append(pts)
    return comps


def _bbox(points: list[tuple[int, int]]) -> tuple[int, int, int, int]:
    rows = [r for r, _ in points]
    cols = [c for _, c in points]
    return min(rows), min(cols), max(rows), max(cols)


def _parse_palette(grid: np.ndarray) -> tuple[int, list[int]]:
    special = ~np.isin(grid, (BLACK, GREY, MAROON))
    rows = np.flatnonzero(special.any(axis=1))
    assert rows.size > 0, "palette row not found"
    row = int(rows[-1])
    colors = [int(grid[row, c]) for c in range(grid.shape[1]) if int(grid[row, c]) != BLACK]
    assert len(colors) >= 3, "palette must contain at least three colors"
    assert len(colors) == len(set(colors)), "palette colors must be unique"
    return row, colors


def _extract_rooms(grid: np.ndarray) -> list[dict]:
    mask = (grid == GREY) | (grid == MAROON)
    rooms: list[dict] = []
    for pts in _components(mask):
        r0, c0, r1, c1 = _bbox(pts)
        outer_h = r1 - r0 + 1
        outer_w = c1 - c0 + 1
        assert outer_h >= 3 and outer_w >= 3, "room must have an interior"

        expected_perimeter = 2 * (outer_h + outer_w) - 4
        assert len(pts) == expected_perimeter, "room wall must be a full rectangle"

        windows: list[tuple[int, int, str]] = []
        for r in range(r0, r1 + 1):
            for c in range(c0, c1 + 1):
                on_edge = r in (r0, r1) or c in (c0, c1)
                value = int(grid[r, c])
                if on_edge:
                    assert value in (GREY, MAROON), "room boundary must be grey or maroon"
                    if (r in (r0, r1)) and (c in (c0, c1)):
                        assert value == GREY, "room corners cannot be windows"
                    elif value == MAROON:
                        if r == r0:
                            side = "top"
                        elif r == r1:
                            side = "bottom"
                        elif c == c0:
                            side = "left"
                        else:
                            side = "right"
                        windows.append((r, c, side))
                else:
                    assert value == BLACK, "room interior must be black"

        assert 1 <= len(windows) <= 3, "each room must have 1..3 windows"
        rooms.append(
            {
                "bbox": (r0, c0, r1, c1),
                "inner_h": outer_h - 2,
                "inner_w": outer_w - 2,
                "windows": sorted(windows),
                "window_count": len(windows),
            }
        )
    rooms.sort(key=lambda room: (room["bbox"][0], room["bbox"][1]))
    return rooms


def _beam_cells(grid: np.ndarray, room: dict, window: tuple[int, int, str]) -> tuple[list[tuple[int, int]], int]:
    h, w = grid.shape
    r, c, side = window
    if side == "top":
        dr, dc = -1, 0
        nominal = int(room["inner_h"])
    elif side == "bottom":
        dr, dc = 1, 0
        nominal = int(room["inner_h"])
    elif side == "left":
        dr, dc = 0, -1
        nominal = int(room["inner_w"])
    else:
        dr, dc = 0, 1
        nominal = int(room["inner_w"])

    cells: list[tuple[int, int]] = []
    for step in range(1, nominal + 1):
        nr = r + dr * step
        nc = c + dc * step
        if not (0 <= nr < h and 0 <= nc < w):
            break
        if int(grid[nr, nc]) != BLACK:
            break
        cells.append((nr, nc))
    return cells, nominal


def _apply_rule(grid: np.ndarray, with_meta: bool = False):
    g = np.asarray(grid, dtype=int)
    palette_row, palette = _parse_palette(g)
    rooms = _extract_rooms(g)
    assert rooms, "at least one room is required"
    assert len(palette) >= max(room["window_count"] for room in rooms), "palette too short"

    out = g.copy()
    beam_infos: list[dict] = []
    for room in rooms:
        color = int(palette[room["window_count"] - 1])
        for window in room["windows"]:
            cells, nominal = _beam_cells(g, room, window)
            for r, c in cells:
                out[r, c] = color
            beam_infos.append(
                {
                    "cells": cells,
                    "nominal": nominal,
                    "actual": len(cells),
                    "color": color,
                    "side": window[2],
                    "room_bbox": room["bbox"],
                }
            )

    if with_meta:
        return out, beam_infos, palette_row, rooms, palette
    return out


def main(grid: np.ndarray) -> np.ndarray:
    """
    Color each room's beams by its window count, using the bottom palette as a 1-based lookup.
    """
    return _apply_rule(grid, with_meta=False)


def _sample_window_counts(rng: random.Random, room_count: int) -> list[int]:
    if room_count == 2:
        counts = list(rng.sample((1, 2, 3), 2))
    else:
        counts = [1, 2, 3]
        while len(counts) < room_count:
            counts.append(rng.randint(1, 3))
    rng.shuffle(counts)
    return counts


def _sample_room_spec(rng: random.Random, window_count: int) -> dict:
    inner_h = rng.randint(*INNER_H_RANGE)
    inner_w = rng.randint(*INNER_W_RANGE)
    if inner_h + inner_w > 9 and rng.random() < 0.6:
        if inner_w >= inner_h and inner_w > INNER_W_RANGE[0]:
            inner_w -= 1
        elif inner_h > INNER_H_RANGE[0]:
            inner_h -= 1

    outer_h = inner_h + 2
    outer_w = inner_w + 2
    candidates: list[tuple[int, int, str]] = []
    for c in range(1, outer_w - 1):
        candidates.append((0, c, "top"))
        candidates.append((outer_h - 1, c, "bottom"))
    for r in range(1, outer_h - 1):
        candidates.append((r, 0, "left"))
        candidates.append((r, outer_w - 1, "right"))

    windows = rng.sample(candidates, window_count)
    return {
        "inner_h": inner_h,
        "inner_w": inner_w,
        "outer_h": outer_h,
        "outer_w": outer_w,
        "windows_local": windows,
        "window_count": window_count,
    }


def _rects_too_close(a: tuple[int, int, int, int], b: tuple[int, int, int, int]) -> bool:
    ar0, ac0, ar1, ac1 = a
    br0, bc0, br1, bc1 = b
    separated = ar1 < br0 - 1 or ar0 > br1 + 1 or ac1 < bc0 - 1 or ac0 > bc1 + 1
    return not separated


def _eligible_truncation_windows(specs: list[dict]) -> list[tuple[int, tuple[int, int, str]]]:
    eligible: list[tuple[int, tuple[int, int, str]]] = []
    for idx, spec in enumerate(specs):
        for window in spec["windows_local"]:
            side = window[2]
            nominal = spec["inner_h"] if side in ("top", "bottom") else spec["inner_w"]
            if side in ("top", "left", "right") and nominal >= 2:
                eligible.append((idx, window))
    return eligible


def _place_rooms(specs: list[dict], h: int, w: int, rng: random.Random, want_truncation: bool) -> list[dict] | None:
    max_room_top = h - 5
    if max_room_top < 0:
        return None

    designated: tuple[int, tuple[int, int, str]] | None = None
    if want_truncation:
        eligible = _eligible_truncation_windows(specs)
        if not eligible:
            return None
        designated = rng.choice(eligible)

    areas = [spec["outer_h"] * spec["outer_w"] for spec in specs]
    order = sorted(range(len(specs)), key=lambda idx: (-areas[idx], idx))
    if designated is not None:
        first = designated[0]
        order = [first] + [idx for idx in order if idx != first]

    placed_by_idx: dict[int, dict] = {}
    placed_bboxes: list[tuple[int, int, int, int]] = []

    for idx in order:
        spec = specs[idx]
        outer_h = int(spec["outer_h"])
        outer_w = int(spec["outer_w"])
        top_limit = h - outer_h - 3
        left_limit = w - outer_w
        if top_limit < 0 or left_limit < 0:
            return None

        for _ in range(ROOM_PLACEMENT_TRIES):
            if designated is not None and idx == designated[0]:
                _, (_, _, side) = designated
                if side == "top":
                    top_cap = min(spec["inner_h"] - 1, top_limit)
                    if top_cap < 1:
                        return None
                    top = rng.randint(1, top_cap)
                    left = rng.randint(0, left_limit)
                elif side == "left":
                    left_cap = min(spec["inner_w"] - 1, left_limit)
                    if left_cap < 1:
                        return None
                    left = rng.randint(1, left_cap)
                    top = rng.randint(0, top_limit)
                else:
                    gap_cap = min(spec["inner_w"] - 1, left_limit)
                    if gap_cap < 1:
                        return None
                    gap = rng.randint(1, gap_cap)
                    left = w - outer_w - gap
                    top = rng.randint(0, top_limit)
            else:
                top = rng.randint(0, top_limit)
                left = rng.randint(0, left_limit)

            bbox = (top, left, top + outer_h - 1, left + outer_w - 1)
            if any(_rects_too_close(bbox, other) for other in placed_bboxes):
                continue

            windows_abs = [(top + wr, left + wc, side) for wr, wc, side in spec["windows_local"]]
            placed_by_idx[idx] = {
                "bbox": bbox,
                "inner_h": spec["inner_h"],
                "inner_w": spec["inner_w"],
                "windows": sorted(windows_abs),
                "window_count": spec["window_count"],
            }
            placed_bboxes.append(bbox)
            break
        else:
            return None

    return [placed_by_idx[idx] for idx in range(len(specs))]


def _palette_positions(width: int, count: int, rng: random.Random) -> list[int] | None:
    gaps = [rng.randint(1, 2) for _ in range(count - 1)]
    positions = [0]
    for gap in gaps:
        positions.append(positions[-1] + gap + 1)
    span = positions[-1] + 1
    if span > width:
        return None
    start = rng.randint(0, width - span)
    return [start + pos for pos in positions]


def _draw_input(h: int, w: int, palette: list[int], palette_cols: list[int], rooms: list[dict]) -> np.ndarray:
    grid = np.full((h, w), BLACK, dtype=int)
    palette_row = h - 1
    for color, col in zip(palette, palette_cols):
        grid[palette_row, col] = int(color)

    for room in rooms:
        r0, c0, r1, c1 = room["bbox"]
        grid[r0, c0 : c1 + 1] = GREY
        grid[r1, c0 : c1 + 1] = GREY
        grid[r0 : r1 + 1, c0] = GREY
        grid[r0 : r1 + 1, c1] = GREY
        for wr, wc, _ in room["windows"]:
            grid[wr, wc] = MAROON
    return grid


def _valid_generated_grid(grid: np.ndarray, expected_rooms: int, want_truncation: bool) -> bool:
    try:
        _, beam_infos, palette_row, rooms, _ = _apply_rule(grid, with_meta=True)
    except AssertionError:
        return False

    if len(rooms) != expected_rooms:
        return False
    if len({room["window_count"] for room in rooms}) < 2:
        return False

    used_cells: set[tuple[int, int]] = set()
    has_full = False
    has_truncated = False
    for info in beam_infos:
        if info["actual"] == 0:
            return False
        if info["actual"] == info["nominal"]:
            has_full = True
        if info["actual"] < info["nominal"]:
            has_truncated = True
        for cell in info["cells"]:
            if cell[0] == palette_row:
                return False
            if cell in used_cells:
                return False
            used_cells.add(cell)

    if not used_cells or not has_full:
        return False
    if want_truncation and not has_truncated:
        return False
    return True


def _generate_attempt(rng: random.Random) -> np.ndarray | None:
    h = rng.randint(*GRID_H_RANGE)
    w = rng.randint(*GRID_W_RANGE)

    room_count = rng.choices(ROOM_COUNT_OPTIONS, weights=ROOM_COUNT_WEIGHTS, k=1)[0]
    if h < 21 or w < 21:
        room_count = min(room_count, 3)

    palette = rng.sample(PALETTE_POOL, 3)
    palette_cols = _palette_positions(w, len(palette), rng)
    if palette_cols is None:
        return None

    want_truncation = rng.random() < 0.45
    counts = _sample_window_counts(rng, room_count)
    specs = [_sample_room_spec(rng, count) for count in counts]
    rooms = _place_rooms(specs, h, w, rng, want_truncation)
    if rooms is None:
        return None

    grid = _draw_input(h, w, palette, palette_cols, rooms)
    if not _valid_generated_grid(grid, room_count, want_truncation):
        return None
    return grid


def generate_input() -> np.ndarray:
    rng = random.Random()
    for _ in range(GEN_ATTEMPTS):
        grid = _generate_attempt(rng)
        if grid is not None:
            return grid
    raise RuntimeError("generate_input: failed to construct a valid puzzle")
