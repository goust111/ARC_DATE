from __future__ import annotations

import random
from collections import deque

import numpy as np


BLACK = 0
GREY = 5
SEED_COLORS = (1, 2, 3, 4, 6, 7, 8, 9)
DIR4 = ((-1, 0), (1, 0), (0, -1), (0, 1))
WALL_DIRS = {
    "up": (-1, 0),
    "down": (1, 0),
    "left": (0, -1),
    "right": (0, 1),
}
SCENARIOS = ("forward", "wrap", "truncate")
MAX_GENERATION_ATTEMPTS = 300


def _components(mask: np.ndarray) -> list[np.ndarray]:
    h, w = mask.shape
    seen = np.zeros((h, w), dtype=bool)
    comps: list[np.ndarray] = []
    for r in range(h):
        for c in range(w):
            if not mask[r, c] or seen[r, c]:
                continue
            q: deque[tuple[int, int]] = deque([(r, c)])
            seen[r, c] = True
            cells: list[tuple[int, int]] = []
            while q:
                cr, cc = q.popleft()
                cells.append((cr, cc))
                for dr, dc in DIR4:
                    nr, nc = cr + dr, cc + dc
                    if 0 <= nr < h and 0 <= nc < w and mask[nr, nc] and not seen[nr, nc]:
                        seen[nr, nc] = True
                        q.append((nr, nc))
            comps.append(np.array(cells, dtype=int))
    return comps


def _adjacent_wall(grid: np.ndarray, r: int, c: int) -> str:
    h, w = grid.shape
    touches: list[str] = []
    for name, (dr, dc) in WALL_DIRS.items():
        nr, nc = r + dr, c + dc
        if 0 <= nr < h and 0 <= nc < w and int(grid[nr, nc]) == GREY:
            touches.append(name)
    assert len(touches) == 1, "each seed must touch exactly one wall"
    return touches[0]


def _grow_segment(
    out: np.ndarray,
    src: np.ndarray,
    r: int,
    c: int,
    length: int,
    color: int,
    horizontal: bool,
) -> None:
    out[r, c] = color
    remaining = int(length) - 1
    if horizontal:
        nc = c + 1
        while remaining > 0 and nc < src.shape[1] and int(src[r, nc]) != GREY:
            out[r, nc] = color
            remaining -= 1
            nc += 1
        nc = c - 1
        while remaining > 0 and nc >= 0 and int(src[r, nc]) != GREY:
            out[r, nc] = color
            remaining -= 1
            nc -= 1
    else:
        nr = r + 1
        while remaining > 0 and nr < src.shape[0] and int(src[nr, c]) != GREY:
            out[nr, c] = color
            remaining -= 1
            nr += 1
        nr = r - 1
        while remaining > 0 and nr >= 0 and int(src[nr, c]) != GREY:
            out[nr, c] = color
            remaining -= 1
            nr -= 1


def main(grid: np.ndarray) -> np.ndarray:
    """
    Grow same-color wall-hugging line segments from each room seed, with segment length set by that room's seed count.

    Input structure:
    - The background is black and the walls are grey one-pixel-thick horizontal
      and vertical separators that partition the canvas into closed rectangular rooms.
    - Room interiors are black except for colored single-pixel seeds.
    - Every seed touches exactly one wall by 4-neighbor adjacency, and all seeds
      inside the same room share one color.

    Rule:
    - Treat each non-grey connected region as one room and count how many seeds it contains.
    - For each seed, detect which wall it touches; seeds touching the top or bottom
      wall grow horizontally, while seeds touching the left or right wall grow vertically.
    - Draw a same-color segment of that room's seed count, including the seed itself,
      preferring the increasing row/column direction first and using the opposite
      direction only if the preferred side reaches a wall early; truncate if the room is too short.

    Key insight:
    - The room-wide seed count controls segment length, while each seed's local
      wall contact independently determines the segment's axis.
    """
    g = np.asarray(grid, dtype=int)
    assert g.ndim == 2, "expected a 2D grid"

    out = g.copy()
    for room in _components(g != GREY):
        seed_cells = [(int(r), int(c)) for r, c in room if int(g[r, c]) not in (BLACK, GREY)]
        if not seed_cells:
            continue
        colors = {int(g[r, c]) for r, c in seed_cells}
        assert len(colors) == 1, "seeds in one room must share a color"
        color = colors.pop()
        length = len(seed_cells)
        for r, c in seed_cells:
            wall = _adjacent_wall(g, r, c)
            _grow_segment(out, g, r, c, length, color, horizontal=wall in {"up", "down"})
    return out


def _build_partition(heights: list[int], widths: list[int]) -> tuple[np.ndarray, list[tuple[int, int, int, int]]]:
    h = sum(heights) + len(heights) + 1
    w = sum(widths) + len(widths) + 1
    grid = np.full((h, w), GREY, dtype=int)
    rooms: list[tuple[int, int, int, int]] = []
    r0 = 1
    for room_h in heights:
        c0 = 1
        for room_w in widths:
            r1 = r0 + room_h - 1
            c1 = c0 + room_w - 1
            grid[r0 : r1 + 1, c0 : c1 + 1] = BLACK
            rooms.append((r0, c0, r1, c1))
            c0 = c1 + 2
        r0 = r1 + 2
    return grid, rooms


def _room_shape(room: tuple[int, int, int, int]) -> tuple[int, int]:
    r0, c0, r1, c1 = room
    return r1 - r0 + 1, c1 - c0 + 1


def _wall_candidates(room: tuple[int, int, int, int], wall: str) -> list[tuple[int, int]]:
    r0, c0, r1, c1 = room
    if wall == "top":
        return [(r0, c) for c in range(c0 + 1, c1)]
    if wall == "bottom":
        return [(r1, c) for c in range(c0 + 1, c1)]
    if wall == "left":
        return [(r, c0) for r in range(r0 + 1, r1)]
    if wall == "right":
        return [(r, c1) for r in range(r0 + 1, r1)]
    raise ValueError(f"unknown wall: {wall}")


def _all_seed_candidates(room: tuple[int, int, int, int]) -> list[tuple[int, int]]:
    cells: list[tuple[int, int]] = []
    for wall in ("top", "bottom", "left", "right"):
        cells.extend(_wall_candidates(room, wall))
    return cells


def _parallel_length(room: tuple[int, int, int, int], wall: str) -> int:
    room_h, room_w = _room_shape(room)
    return room_w if wall in {"top", "bottom"} else room_h


def _seed_offset(room: tuple[int, int, int, int], wall: str, cell: tuple[int, int]) -> int:
    r0, c0, _r1, _c1 = room
    r, c = cell
    return c - c0 if wall in {"top", "bottom"} else r - r0


def _scenario_options(
    room: tuple[int, int, int, int],
    scenario: str,
) -> list[tuple[str, int, tuple[int, int]]]:
    total_candidates = len(_all_seed_candidates(room))
    options: list[tuple[str, int, tuple[int, int]]] = []
    for wall in ("top", "bottom", "left", "right"):
        cells = _wall_candidates(room, wall)
        if not cells:
            continue
        axis_len = _parallel_length(room, wall)
        if scenario == "truncate":
            for n_seeds in range(max(2, axis_len + 1), 5):
                if n_seeds > total_candidates:
                    continue
                for cell in cells:
                    options.append((wall, n_seeds, cell))
        elif scenario == "forward":
            for n_seeds in range(2, min(4, axis_len) + 1):
                if n_seeds > total_candidates:
                    continue
                for cell in cells:
                    if _seed_offset(room, wall, cell) <= axis_len - n_seeds:
                        options.append((wall, n_seeds, cell))
        elif scenario == "wrap":
            for n_seeds in range(3, min(4, axis_len) + 1):
                if n_seeds > total_candidates:
                    continue
                for cell in cells:
                    if _seed_offset(room, wall, cell) > axis_len - n_seeds:
                        options.append((wall, n_seeds, cell))
        else:
            raise ValueError(f"unknown scenario: {scenario}")
    return options


def _place_room_seeds(
    grid: np.ndarray,
    room: tuple[int, int, int, int],
    n_seeds: int,
    color: int,
    rng: random.Random,
    forced_cell: tuple[int, int] | None = None,
) -> bool:
    by_wall = {wall: _wall_candidates(room, wall)[:] for wall in ("top", "bottom", "left", "right")}
    for cells in by_wall.values():
        rng.shuffle(cells)

    chosen: list[tuple[int, int]] = []
    used: set[tuple[int, int]] = set()
    if forced_cell is not None:
        chosen.append(forced_cell)
        used.add(forced_cell)
        for cells in by_wall.values():
            if forced_cell in cells:
                cells.remove(forced_cell)

    wall_order = list(by_wall)
    rng.shuffle(wall_order)
    while len(chosen) < n_seeds:
        progress = False
        for wall in wall_order:
            while by_wall[wall] and by_wall[wall][-1] in used:
                by_wall[wall].pop()
            if not by_wall[wall]:
                continue
            cell = by_wall[wall].pop()
            chosen.append(cell)
            used.add(cell)
            progress = True
            if len(chosen) == n_seeds:
                break
        if not progress:
            return False

    for r, c in chosen:
        grid[r, c] = color
    return True


def _gen_attempt(rng: random.Random) -> np.ndarray | None:
    n_room_rows = rng.randint(2, 4)
    n_room_cols = rng.randint(2, 4)
    heights = [rng.randint(3, 6) for _ in range(n_room_rows)]
    widths = [rng.randint(3, 6) for _ in range(n_room_cols)]
    grid, rooms = _build_partition(heights, widths)

    scenario = rng.choice(SCENARIOS)
    focus_options: list[tuple[int, str, int, tuple[int, int]]] = []
    for room_idx, room in enumerate(rooms):
        for wall, n_seeds, cell in _scenario_options(room, scenario):
            focus_options.append((room_idx, wall, n_seeds, cell))
    if not focus_options:
        return None

    focus_room_idx, _focus_wall, focus_n, focus_cell = rng.choice(focus_options)
    remaining_room_ids = [idx for idx in range(len(rooms)) if idx != focus_room_idx]
    if not remaining_room_ids:
        return None

    singleton_room_idx = rng.choice(remaining_room_ids)
    extra_candidates = [idx for idx in remaining_room_ids if idx != singleton_room_idx]
    rng.shuffle(extra_candidates)
    max_extra = min(3, max(0, len(extra_candidates) - 1))
    extra_count = rng.randint(0, max_extra) if max_extra > 0 else 0
    active_room_ids = [focus_room_idx, singleton_room_idx] + extra_candidates[:extra_count]

    room_specs: dict[int, tuple[int, tuple[int, int] | None]] = {
        focus_room_idx: (focus_n, focus_cell),
        singleton_room_idx: (1, None),
    }
    for room_idx in active_room_ids:
        if room_idx in room_specs:
            continue
        room_specs[room_idx] = (rng.randint(1, 4), None)

    colors = list(SEED_COLORS)
    rng.shuffle(colors)
    for room_idx, color in zip(active_room_ids, colors):
        n_seeds, forced_cell = room_specs[room_idx]
        if not _place_room_seeds(grid, rooms[room_idx], n_seeds, color, rng, forced_cell=forced_cell):
            return None

    return grid


def generate_input() -> np.ndarray:
    rng = random.Random()
    for _ in range(MAX_GENERATION_ATTEMPTS):
        grid = _gen_attempt(rng)
        if grid is not None:
            return grid
    raise RuntimeError("failed to generate a valid room-seed input")
