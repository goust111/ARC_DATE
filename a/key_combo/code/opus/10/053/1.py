from __future__ import annotations

from collections import deque
from importlib.machinery import SourceFileLoader
from pathlib import Path
import random

import numpy as np

try:
    from arc_utils import Color, components_mask
except ModuleNotFoundError:
    _ARC_UTILS = SourceFileLoader(
        "arc_utils_shared", "/home/lixinyu/snap/new/solvers/arc_utils.py"
    ).load_module()
    Color = _ARC_UTILS.Color
    components_mask = _ARC_UTILS.components_mask


BACKGROUND = Color.BLACK
WALL = Color.GREY
MARKER_COLORS = (Color.BLUE, Color.RED, Color.GREEN, Color.YELLOW, Color.PINK, Color.ORANGE, Color.TEAL, Color.MAROON)

GRID_HEIGHT_RANGE = (17, 24)
GRID_WIDTH_RANGE = (17, 24)
ROOM_COUNT_RANGE = (2, 4)
ROOM_INTERIOR_HEIGHT_RANGE = (4, 7)
ROOM_INTERIOR_WIDTH_RANGE = (4, 7)
ROOM_MARGIN_TOP = 2
ROOM_MARGIN_SIDE = 1
ROOM_MARGIN_BOTTOM = 1
ROOM_GAP = 2
MAX_GENERATION_ATTEMPTS = 400


def _exterior_mask(passable: np.ndarray) -> np.ndarray:
    h, w = passable.shape
    seen = np.zeros_like(passable, dtype=bool)
    q: deque[tuple[int, int]] = deque()

    for r in range(h):
        for c in (0, w - 1):
            if passable[r, c] and not seen[r, c]:
                seen[r, c] = True
                q.append((r, c))
    for c in range(w):
        for r in (0, h - 1):
            if passable[r, c] and not seen[r, c]:
                seen[r, c] = True
                q.append((r, c))

    while q:
        r, c = q.popleft()
        for nr, nc in ((r - 1, c), (r + 1, c), (r, c - 1), (r, c + 1)):
            if 0 <= nr < h and 0 <= nc < w and passable[nr, nc] and not seen[nr, nc]:
                seen[nr, nc] = True
                q.append((nr, nc))
    return seen


def _enclosed_components(grid: np.ndarray) -> list[np.ndarray]:
    passable = grid != WALL
    exterior = _exterior_mask(passable)
    return components_mask(passable & ~exterior, connectivity=4)


def main(grid: np.ndarray) -> np.ndarray:
    """Fill each enclosed region on the marker-selected parity lattice while preserving walls and exterior.

    Input structure
    - A black background is partitioned by grey wall pixels into an exterior and several enclosed regions.
    - Each enclosed region contains exactly one non-black, non-grey marker pixel.
    - Marker coordinates use the standard grid reference frame `(row, col)`.

    Rule
    - Keep every grey wall pixel unchanged.
    - Keep every exterior cell connected to the border unchanged.
    - For each enclosed region, find its unique marker; use the marker's color as the paint color.
    - Recolor each black cell in that region iff its row parity and column parity both match the marker's parity; keep the marker itself unchanged.

    Key insight
    - The fill pattern is local to each wall-bounded region: walls decide the region, while the marker's own coordinate parity decides the checkerboard phase inside that region.
    """
    g = np.asarray(grid, dtype=int)
    assert g.ndim == 2

    out = g.copy()
    for coords in _enclosed_components(g):
        colors = {
            int(g[r, c])
            for r, c in coords
            if int(g[r, c]) not in (BACKGROUND, WALL)
        }
        assert len(colors) == 1, "each enclosed region must contain exactly one marker color"
        marker_color = colors.pop()

        marker_coords = [
            (int(r), int(c))
            for r, c in coords
            if int(g[r, c]) == marker_color
        ]
        assert len(marker_coords) == 1, "each enclosed region must contain exactly one marker pixel"
        marker_r, marker_c = marker_coords[0]
        row_parity = marker_r % 2
        col_parity = marker_c % 2

        for r, c in coords:
            rr = int(r)
            cc = int(c)
            if int(g[rr, cc]) == BACKGROUND and rr % 2 == row_parity and cc % 2 == col_parity:
                out[rr, cc] = marker_color

    return out


def _boxes_separated(a: tuple[int, int, int, int], b: tuple[int, int, int, int], gap: int = ROOM_GAP) -> bool:
    ar0, ac0, ar1, ac1 = a
    br0, bc0, br1, bc1 = b
    return ar1 + gap < br0 or br1 + gap < ar0 or ac1 + gap < bc0 or bc1 + gap < ac0


def _sample_rooms(h: int, w: int, room_count: int) -> list[dict] | None:
    rooms: list[dict] = []
    for _ in range(200):
        rooms.clear()
        ok = True
        for _room_index in range(room_count):
            interior_h = random.randint(*ROOM_INTERIOR_HEIGHT_RANGE)
            interior_w = random.randint(*ROOM_INTERIOR_WIDTH_RANGE)
            outer_h = interior_h + 2
            outer_w = interior_w + 2

            r0_min = ROOM_MARGIN_TOP
            r0_max = h - ROOM_MARGIN_BOTTOM - outer_h
            c0_min = ROOM_MARGIN_SIDE
            c0_max = w - ROOM_MARGIN_SIDE - outer_w
            if r0_min > r0_max or c0_min > c0_max:
                return None

            placed = False
            for _ in range(120):
                r0 = random.randint(r0_min, r0_max)
                c0 = random.randint(c0_min, c0_max)
                box = (r0, c0, r0 + outer_h - 1, c0 + outer_w - 1)
                if all(_boxes_separated(box, other["box"]) for other in rooms):
                    rooms.append(
                        {
                            "box": box,
                            "inner": (r0 + 1, c0 + 1, r0 + outer_h - 2, c0 + outer_w - 2),
                        }
                    )
                    placed = True
                    break
            if not placed:
                ok = False
                break
        if ok:
            return [dict(room) for room in rooms]
    return None


def _available_stem_columns(rooms: list[dict], room_index: int) -> list[int]:
    r0, c0, _r1, c1 = rooms[room_index]["box"]
    options = []
    for col in range(c0 + 1, c1):
        blocked = False
        for j, other in enumerate(rooms):
            if j == room_index:
                continue
            or0, oc0, or1, oc1 = other["box"]
            if oc0 <= col <= oc1 and or0 <= r0 - 1:
                blocked = True
                break
        if not blocked:
            options.append(col)
    return options


def _draw_rooms_and_walls(h: int, w: int, rooms: list[dict]) -> np.ndarray | None:
    grid = np.zeros((h, w), dtype=int)
    grid[0, :] = WALL

    for room in rooms:
        r0, c0, r1, c1 = room["box"]
        grid[r0 : r1 + 1, c0] = WALL
        grid[r0 : r1 + 1, c1] = WALL
        grid[r0, c0 : c1 + 1] = WALL
        grid[r1, c0 : c1 + 1] = WALL

    for idx, room in enumerate(rooms):
        options = _available_stem_columns(rooms, idx)
        if not options:
            return None
        stem_col = random.choice(options)
        room["stem_col"] = stem_col
        r0 = room["box"][0]
        grid[: r0 + 1, stem_col] = WALL

    return grid


def _safe_marker_cells(room: dict) -> list[tuple[int, int]]:
    ir0, ic0, ir1, ic1 = room["inner"]
    return [
        (r, c)
        for r in range(ir0 + 1, ir1)
        for c in range(ic0 + 1, ic1)
    ]


def _choose_markers(grid: np.ndarray, rooms: list[dict]) -> bool:
    all_parities: list[tuple[int, int]] = []
    all_colors: list[int] = []

    marker_colors = random.sample(MARKER_COLORS, k=min(len(MARKER_COLORS), max(2, len(rooms))))
    if len(rooms) >= 3 and random.random() < 0.6:
        marker_colors = marker_colors[: len(rooms) - 1] + [random.choice(marker_colors[: len(rooms) - 1])]
    while len(marker_colors) < len(rooms):
        marker_colors.append(random.choice(MARKER_COLORS))
    random.shuffle(marker_colors)

    for room, color in zip(rooms, marker_colors):
        safe_cells = _safe_marker_cells(room)
        if not safe_cells:
            return False
        marker_r, marker_c = random.choice(safe_cells)
        room["marker"] = (marker_r, marker_c)
        room["color"] = int(color)
        grid[marker_r, marker_c] = int(color)
        all_parities.append((marker_r % 2, marker_c % 2))
        all_colors.append(int(color))

    if len(set(all_parities)) < 2:
        return False
    if len(set(all_colors)) < 2:
        return False
    return True


def _expected_from_rooms(grid: np.ndarray, rooms: list[dict]) -> np.ndarray:
    out = grid.copy()
    for room in rooms:
        ir0, ic0, ir1, ic1 = room["inner"]
        marker_r, marker_c = room["marker"]
        color = room["color"]
        for r in range(ir0, ir1 + 1):
            for c in range(ic0, ic1 + 1):
                if out[r, c] == BACKGROUND and r % 2 == marker_r % 2 and c % 2 == marker_c % 2:
                    out[r, c] = color
    return out


def _validate_semantics(grid: np.ndarray, rooms: list[dict]) -> bool:
    grey_comps = components_mask(grid == WALL, connectivity=4)
    if len(grey_comps) != 1:
        return False

    enclosed = _enclosed_components(grid)
    if len(enclosed) != len(rooms):
        return False

    seen_markers: set[tuple[int, int]] = set()
    parity_classes: set[tuple[int, int]] = set()
    recovered_areas = sorted(len(coords) for coords in enclosed)
    intended_areas = sorted((room["inner"][2] - room["inner"][0] + 1) * (room["inner"][3] - room["inner"][1] + 1) for room in rooms)
    if recovered_areas != intended_areas:
        return False

    for coords in enclosed:
        marker_cells = [
            (int(r), int(c))
            for r, c in coords
            if int(grid[r, c]) not in (BACKGROUND, WALL)
        ]
        if len(marker_cells) != 1:
            return False
        marker_r, marker_c = marker_cells[0]
        seen_markers.add((marker_r, marker_c))
        parity_classes.add((marker_r % 2, marker_c % 2))

        for nr, nc in ((marker_r - 1, marker_c), (marker_r + 1, marker_c), (marker_r, marker_c - 1), (marker_r, marker_c + 1)):
            if int(grid[nr, nc]) == WALL:
                return False

    room_markers = {tuple(room["marker"]) for room in rooms}
    if seen_markers != room_markers:
        return False
    if len(parity_classes) < 2:
        return False

    expected = _expected_from_rooms(grid, rooms)
    try:
        solved = main(grid)
    except AssertionError:
        return False
    return bool(np.array_equal(expected, solved))


def _gen_attempt() -> np.ndarray | None:
    h = random.randint(*GRID_HEIGHT_RANGE)
    w = random.randint(*GRID_WIDTH_RANGE)
    room_count = random.randint(*ROOM_COUNT_RANGE)

    rooms = _sample_rooms(h, w, room_count)
    if rooms is None:
        return None

    grid = _draw_rooms_and_walls(h, w, rooms)
    if grid is None:
        return None

    if not _choose_markers(grid, rooms):
        return None
    if not _validate_semantics(grid, rooms):
        return None
    return grid


def generate_input() -> np.ndarray:
    for _ in range(MAX_GENERATION_ATTEMPTS):
        grid = _gen_attempt()
        if grid is not None:
            return grid
    raise RuntimeError("failed to generate a valid enclosed parity-fill input")

