from __future__ import annotations

import random

import numpy as np


SIDE_ORDER = ("top", "bottom", "left", "right")
SIDE_PRIORITY = {side: idx for idx, side in enumerate(SIDE_ORDER)}


def _dominant_color(grid: np.ndarray) -> int:
    values, counts = np.unique(grid, return_counts=True)
    return int(values[np.argmax(counts)])


def _color_components(grid: np.ndarray, color: int) -> list[np.ndarray]:
    mask = grid == color
    visited = np.zeros_like(mask, dtype=bool)
    height, width = grid.shape
    components: list[np.ndarray] = []

    for r, c in np.argwhere(mask):
        rr = int(r)
        cc = int(c)
        if visited[rr, cc]:
            continue
        stack = [(rr, cc)]
        visited[rr, cc] = True
        coords: list[tuple[int, int]] = []
        while stack:
            cr, cc = stack.pop()
            coords.append((cr, cc))
            for dr, dc in ((-1, 0), (1, 0), (0, -1), (0, 1)):
                nr = cr + dr
                nc = cc + dc
                if 0 <= nr < height and 0 <= nc < width and mask[nr, nc] and not visited[nr, nc]:
                    visited[nr, nc] = True
                    stack.append((nr, nc))
        components.append(np.asarray(coords, dtype=int))
    return components


def _bbox(coords: np.ndarray) -> tuple[int, int, int, int]:
    r0, c0 = coords.min(axis=0)
    r1, c1 = coords.max(axis=0)
    return int(r0), int(c0), int(r1), int(c1)


def _find_antennas(grid: np.ndarray, bg: int) -> tuple[list[dict], np.ndarray]:
    antennas: list[dict] = []
    antenna_mask = np.zeros_like(grid, dtype=bool)
    antenna_color: int | None = None

    for raw_color in np.unique(grid):
        color = int(raw_color)
        if color == bg:
            continue
        for coords in _color_components(grid, color):
            if len(coords) != 9:
                continue
            r0, c0, r1, c1 = _bbox(coords)
            if r1 - r0 != 2 or c1 - c0 != 2:
                continue
            if not np.all(grid[r0 : r1 + 1, c0 : c1 + 1] == color):
                continue
            if antenna_color is None:
                antenna_color = color
            else:
                assert antenna_color == color, "all antennas must share one color"
            antennas.append({"bbox": (r0, c0, r1, c1), "color": color})
            antenna_mask[r0 : r1 + 1, c0 : c1 + 1] = True

    assert antennas, "expected at least one antenna"
    assert antenna_color is not None
    assert np.array_equal(grid == antenna_color, antenna_mask), "antenna color must belong only to antennas"
    antennas.sort(key=lambda ant: (ant["bbox"][0], ant["bbox"][1]))
    return antennas, antenna_mask


def _paint_side(out: np.ndarray, bbox: tuple[int, int, int, int], side: str, color: int) -> None:
    r0, c0, r1, c1 = bbox
    if side == "top":
        out[r0, c0 : c1 + 1] = color
    elif side == "bottom":
        out[r1, c0 : c1 + 1] = color
    elif side == "left":
        out[r0 : r1 + 1, c0] = color
    else:
        assert side == "right"
        out[r0 : r1 + 1, c1] = color


def main(grid: np.ndarray) -> np.ndarray:
    """
    Propagate each signal color onto the corresponding face of the nearest aligned 3x3 antenna.
    """
    g = np.asarray(grid, dtype=int)
    assert g.ndim == 2, "expected a 2D grid"

    bg = _dominant_color(g)
    antennas, antenna_mask = _find_antennas(g, bg)

    out = np.full_like(g, bg)
    for antenna in antennas:
        r0, c0, r1, c1 = antenna["bbox"]
        out[r0 : r1 + 1, c0 : c1 + 1] = int(antenna["color"])

    best_signals: list[dict[str, tuple[int, int]]] = [dict() for _ in antennas]
    signal_cells = np.argwhere((g != bg) & ~antenna_mask)
    assert len(signal_cells) > 0, "expected at least one signal pixel"

    for r, c in signal_cells:
        rr = int(r)
        cc = int(c)
        color = int(g[rr, cc])
        candidates: list[tuple[int, int, str]] = []

        for idx, antenna in enumerate(antennas):
            r0, c0, r1, c1 = antenna["bbox"]
            if c0 <= cc <= c1:
                if rr < r0:
                    candidates.append((r0 - rr, idx, "top"))
                elif rr > r1:
                    candidates.append((rr - r1, idx, "bottom"))
            if r0 <= rr <= r1:
                if cc < c0:
                    candidates.append((c0 - cc, idx, "left"))
                elif cc > c1:
                    candidates.append((cc - c1, idx, "right"))

        assert candidates, "every signal must align to at least one antenna"
        dist, antenna_idx, side = min(
            candidates,
            key=lambda item: (
                item[0],
                antennas[item[1]]["bbox"][0],
                antennas[item[1]]["bbox"][1],
                SIDE_PRIORITY[item[2]],
            ),
        )
        previous = best_signals[antenna_idx].get(side)
        if previous is None or dist < previous[0]:
            best_signals[antenna_idx][side] = (dist, color)

    for antenna, face_map in zip(antennas, best_signals):
        bbox = antenna["bbox"]
        for side in SIDE_ORDER:
            if side in face_map:
                _paint_side(out, bbox, side, face_map[side][1])
        r0, c0, _r1, _c1 = bbox
        out[r0 + 1, c0 + 1] = int(antenna["color"])

    return out


# --------------- generator ---------------

GEN_BG = 8
GEN_ANTENNA = 5
ANTENNA_SIZE = 3
SIGNAL_COLORS = [1, 2, 3, 4, 6, 7, 9]
GENERATION_ATTEMPTS = 400

GRID_MIN = 16
GRID_MAX = 26
SAFE_MARGIN = 3
SLOT_STEP = 4
MAX_ANTENNAS = 4
MAX_SIGNALS_PER_SIDE = 3

SINGLE_MODE = "single"
OPPOSITE_MODE = "opposite"
MULTI_SAME_MODE = "multi_same"


def _slot_starts(length: int) -> list[int]:
    last = length - ANTENNA_SIZE - SAFE_MARGIN
    return list(range(SAFE_MARGIN, last + 1, SLOT_STEP))


def _max_distance(bbox: tuple[int, int, int, int], side: str, height: int, width: int) -> int:
    r0, c0, r1, c1 = bbox
    if side == "top":
        return r0
    if side == "bottom":
        return height - 1 - r1
    if side == "left":
        return c0
    assert side == "right"
    return width - 1 - c1


def _other_signal_color(base_color: int) -> int:
    options = [color for color in SIGNAL_COLORS if color != base_color]
    return random.choice(options)


def _pick_active_sides() -> tuple[str, ...]:
    mode = random.choice((SINGLE_MODE, OPPOSITE_MODE, MULTI_SAME_MODE))
    if mode == SINGLE_MODE:
        return mode, (random.choice(SIDE_ORDER),)
    if mode == OPPOSITE_MODE:
        return mode, random.choice((("top", "bottom"), ("left", "right")))
    size = random.randint(2, 4)
    return mode, tuple(random.sample(SIDE_ORDER, size))


def _face_colors(mode: str, active_sides: tuple[str, ...]) -> dict[str, int]:
    if mode == MULTI_SAME_MODE:
        shared = random.choice(SIGNAL_COLORS)
        return {side: shared for side in active_sides}
    if len(active_sides) == 1:
        return {active_sides[0]: random.choice(SIGNAL_COLORS)}
    colors = random.sample(SIGNAL_COLORS, len(active_sides))
    return {side: color for side, color in zip(active_sides, colors)}


def _signal_positions(
    bbox: tuple[int, int, int, int], side: str, distances: list[int]
) -> list[tuple[int, int]]:
    r0, c0, r1, c1 = bbox
    offsets = random.sample(range(ANTENNA_SIZE), len(distances))
    positions: list[tuple[int, int]] = []
    for dist, offset in zip(distances, offsets):
        if side == "top":
            positions.append((r0 - dist, c0 + offset))
        elif side == "bottom":
            positions.append((r1 + dist, c0 + offset))
        elif side == "left":
            positions.append((r0 + offset, c0 - dist))
        else:
            assert side == "right"
            positions.append((r0 + offset, c1 + dist))
    return positions


def _add_antenna_signals(
    grid: np.ndarray, bbox: tuple[int, int, int, int]
) -> dict[str, int] | None:
    height, width = grid.shape
    mode, active_sides = _pick_active_sides()
    face_map = _face_colors(mode, active_sides)

    for side in active_sides:
        max_dist = _max_distance(bbox, side, height, width)
        if max_dist < 1:
            return None
        signal_count = random.randint(1, min(MAX_SIGNALS_PER_SIDE, max_dist))
        distances = random.sample(range(1, max_dist + 1), signal_count)
        nearest = min(distances)
        positions = _signal_positions(bbox, side, distances)
        for dist, (r, c) in zip(distances, positions):
            if grid[r, c] != GEN_BG:
                return None
            grid[r, c] = face_map[side] if dist == nearest else _other_signal_color(face_map[side])

    return face_map


def _expected_output(
    shape: tuple[int, int], antennas: list[tuple[int, int, int, int]], face_maps: list[dict[str, int]]
) -> np.ndarray:
    out = np.full(shape, GEN_BG, dtype=int)
    for bbox, face_map in zip(antennas, face_maps):
        r0, c0, r1, c1 = bbox
        out[r0 : r1 + 1, c0 : c1 + 1] = GEN_ANTENNA
        for side in SIDE_ORDER:
            if side in face_map:
                _paint_side(out, bbox, side, face_map[side])
        out[r0 + 1, c0 + 1] = GEN_ANTENNA
    return out


def _gen_attempt() -> np.ndarray | None:
    height = random.randint(GRID_MIN, GRID_MAX)
    width = random.randint(GRID_MIN, GRID_MAX)

    row_slots = _slot_starts(height)
    col_slots = _slot_starts(width)
    max_antennas = min(len(row_slots), len(col_slots), MAX_ANTENNAS)
    if max_antennas < 1:
        return None

    antenna_count = random.randint(1, max_antennas)
    rows = random.sample(row_slots, antenna_count)
    cols = random.sample(col_slots, antenna_count)
    random.shuffle(cols)

    antennas = [(r, c, r + ANTENNA_SIZE - 1, c + ANTENNA_SIZE - 1) for r, c in zip(rows, cols)]
    grid = np.full((height, width), GEN_BG, dtype=int)
    for r0, c0, r1, c1 in antennas:
        grid[r0 : r1 + 1, c0 : c1 + 1] = GEN_ANTENNA

    face_maps: list[dict[str, int]] = []
    for bbox in antennas:
        face_map = _add_antenna_signals(grid, bbox)
        if face_map is None:
            return None
        face_maps.append(face_map)

    expected = _expected_output(grid.shape, antennas, face_maps)
    try:
        result = main(grid)
    except Exception:
        return None
    if not np.array_equal(result, expected):
        return None
    return grid


def generate_input() -> np.ndarray:
    for _ in range(GENERATION_ATTEMPTS):
        grid = _gen_attempt()
        if grid is not None:
            return grid
    raise RuntimeError(f"failed to generate valid input after {GENERATION_ATTEMPTS} attempts")
