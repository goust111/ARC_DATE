from __future__ import annotations

from collections import Counter, deque
from itertools import combinations
import random

import numpy as np


DIRS4 = ((-1, 0), (1, 0), (0, -1), (0, 1))
COLOR_IDS = tuple(range(10))
LAYOUTS = ((2, 2), (2, 3), (3, 2))
ROOM_SIZE_RANGE = (4, 7)
PALETTE_SIZE_RANGE = (4, 6)
MAX_ROOM_SET_SIZE = 3
MAX_GENERATION_ATTEMPTS = 300
SET_ASSIGNMENT_ATTEMPTS = 80


def _most_common_color(grid: np.ndarray) -> int:
    values, counts = np.unique(grid, return_counts=True)
    return int(values[int(np.argmax(counts))])


def _largest_non_bg_color(grid: np.ndarray, bg: int) -> int:
    values, counts = np.unique(grid, return_counts=True)
    best_color = None
    best_count = -1
    for value, count in zip(values, counts):
        color = int(value)
        if color == bg:
            continue
        if int(count) > best_count:
            best_color = color
            best_count = int(count)
    assert best_color is not None, "expected a wall color"
    return int(best_color)


def _label_components(mask: np.ndarray) -> tuple[np.ndarray, list[list[tuple[int, int]]]]:
    height, width = mask.shape
    labels = np.full((height, width), -1, dtype=int)
    components: list[list[tuple[int, int]]] = []

    for r in range(height):
        for c in range(width):
            if not bool(mask[r, c]) or labels[r, c] >= 0:
                continue

            comp_id = len(components)
            coords: list[tuple[int, int]] = []
            queue: deque[tuple[int, int]] = deque([(r, c)])
            labels[r, c] = comp_id
            while queue:
                cr, cc = queue.popleft()
                coords.append((cr, cc))
                for dr, dc in DIRS4:
                    nr = cr + dr
                    nc = cc + dc
                    if 0 <= nr < height and 0 <= nc < width and bool(mask[nr, nc]) and labels[nr, nc] < 0:
                        labels[nr, nc] = comp_id
                        queue.append((nr, nc))
            components.append(coords)

    return labels, components


def _room_marker_sets(
    grid: np.ndarray,
    room_labels: np.ndarray,
    rooms: list[list[tuple[int, int]]],
    bg: int,
) -> list[set[int]]:
    marker_sets: list[set[int]] = []
    for room_id, coords in enumerate(rooms):
        colors = {
            int(grid[r, c])
            for r, c in coords
            if room_labels[r, c] == room_id and int(grid[r, c]) != bg
        }
        marker_sets.append(colors)
    return marker_sets


def _shared_wall_segments(grid: np.ndarray, room_labels: np.ndarray, wall: int) -> dict[tuple[str, int, int], list[tuple[int, int]]]:
    height, width = grid.shape
    segments: dict[tuple[str, int, int], list[tuple[int, int]]] = {}

    for r in range(height):
        for c in range(width):
            if int(grid[r, c]) != wall:
                continue

            if 0 < c < width - 1:
                left = int(room_labels[r, c - 1])
                right = int(room_labels[r, c + 1])
                if left >= 0 and right >= 0 and left != right:
                    a, b = sorted((left, right))
                    segments.setdefault(("V", a, b), []).append((r, c))
                    continue

            if 0 < r < height - 1:
                top = int(room_labels[r - 1, c])
                bottom = int(room_labels[r + 1, c])
                if top >= 0 and bottom >= 0 and top != bottom:
                    a, b = sorted((top, bottom))
                    segments.setdefault(("H", a, b), []).append((r, c))

    return segments


def _door_position(orientation: str, coords: list[tuple[int, int]]) -> tuple[int, int]:
    assert coords, "expected a nonempty wall segment"
    if orientation == "V":
        col = coords[0][1]
        rows = sorted(r for r, _ in coords)
        return int(rows[(len(rows) - 1) // 2]), int(col)
    row = coords[0][0]
    cols = sorted(c for _, c in coords)
    return int(row), int(cols[(len(cols) - 1) // 2])


def main(grid: np.ndarray) -> np.ndarray:
    """
    Open doors on shared walls when adjacent rooms have exactly one common marker color.
    """
    g = np.asarray(grid, dtype=int)
    assert g.ndim == 2, "expected a 2D grid"

    bg = _most_common_color(g)
    wall = _largest_non_bg_color(g, bg)

    room_labels, rooms = _label_components(g != wall)
    assert rooms, "expected at least one room"
    room_markers = _room_marker_sets(g, room_labels, rooms, bg)
    segments = _shared_wall_segments(g, room_labels, wall)

    out = g.copy()
    for (orientation, room_a, room_b), coords in segments.items():
        common = room_markers[room_a] & room_markers[room_b]
        if len(common) != 1:
            continue
        door_r, door_c = _door_position(orientation, coords)
        out[door_r, door_c] = next(iter(common))
    return out


def _room_id(row: int, col: int, room_cols: int) -> int:
    return row * room_cols + col


def _room_rects(heights: list[int], widths: list[int]) -> dict[int, tuple[int, int, int, int]]:
    rects: dict[int, tuple[int, int, int, int]] = {}
    room_cols = len(widths)
    row_start = 1
    for i, height in enumerate(heights):
        col_start = 1
        for j, width in enumerate(widths):
            rects[_room_id(i, j, room_cols)] = (row_start, col_start, int(height), int(width))
            col_start += int(width) + 1
        row_start += int(height) + 1
    return rects


def _adjacency_edges(heights: list[int], widths: list[int]) -> list[dict[str, int | str]]:
    room_rows = len(heights)
    room_cols = len(widths)
    edges: list[dict[str, int | str]] = []

    for i in range(room_rows):
        for j in range(room_cols - 1):
            edges.append(
                {
                    "a": _room_id(i, j, room_cols),
                    "b": _room_id(i, j + 1, room_cols),
                    "orientation": "V",
                }
            )
    for i in range(room_rows - 1):
        for j in range(room_cols):
            edges.append(
                {
                    "a": _room_id(i, j, room_cols),
                    "b": _room_id(i + 1, j, room_cols),
                    "orientation": "H",
                }
            )
    return edges


def _choose_active_edges(edges: list[dict[str, int | str]]) -> set[int] | None:
    vertical = [i for i, edge in enumerate(edges) if edge["orientation"] == "V"]
    horizontal = [i for i, edge in enumerate(edges) if edge["orientation"] == "H"]

    for _ in range(40):
        active = {i for i in range(len(edges)) if random.random() < 0.5}
        if not active or len(active) == len(edges):
            continue
        if vertical and not any(i in active for i in vertical):
            continue
        if horizontal and not any(i in active for i in horizontal):
            continue
        return active
    return None


def _candidate_room_sets(palette: list[int]) -> list[frozenset[int]]:
    candidates: list[frozenset[int]] = []
    for size in range(1, min(MAX_ROOM_SET_SIZE, len(palette)) + 1):
        for subset in combinations(palette, size):
            candidates.append(frozenset(int(color) for color in subset))
    return candidates


def _assign_room_sets(room_count: int, edges: list[dict[str, int | str]], active_edges: set[int], palette: list[int]) -> list[set[int]] | None:
    incident: list[list[tuple[int, int, int]]] = [[] for _ in range(room_count)]
    degrees = [0] * room_count
    for edge_idx, edge in enumerate(edges):
        a = int(edge["a"])
        b = int(edge["b"])
        need = 1 if edge_idx in active_edges else 0
        incident[a].append((b, need, edge_idx))
        incident[b].append((a, need, edge_idx))
        degrees[a] += 1
        degrees[b] += 1

    order = sorted(range(room_count), key=lambda room: (-degrees[room], room))
    candidates = _candidate_room_sets(palette)
    assignment: list[frozenset[int] | None] = [None] * room_count

    def backtrack(pos: int) -> bool:
        if pos == len(order):
            return True

        room = order[pos]
        local_candidates = candidates[:]
        random.shuffle(local_candidates)
        for subset in local_candidates:
            ok = True
            for neighbor, need, _edge_idx in incident[room]:
                other = assignment[neighbor]
                if other is None:
                    continue
                if len(subset & other) != need:
                    ok = False
                    break
            if not ok:
                continue
            assignment[room] = subset
            if backtrack(pos + 1):
                return True
            assignment[room] = None
        return False

    if not backtrack(0):
        return None

    result = [set() for _ in range(room_count)]
    for room, subset in enumerate(assignment):
        assert subset is not None
        result[room] = set(int(color) for color in subset)

    if max(len(colors) for colors in result) < 2:
        return None
    if len(set().union(*result)) < 3:
        return None
    return result


def _interior_wall_positions(height: int, width: int) -> list[tuple[int, int]]:
    positions: list[tuple[int, int]] = []
    for c in range(1, width - 1):
        positions.append((0, c))
        positions.append((height - 1, c))
    for r in range(1, height - 1):
        positions.append((r, 0))
        positions.append((r, width - 1))
    return positions


def _place_room_markers(
    grid: np.ndarray,
    rect: tuple[int, int, int, int],
    colors: set[int],
) -> None:
    r0, c0, height, width = rect
    eligible = _interior_wall_positions(height, width)
    min_needed = len(colors)
    extra_cap = max(0, min(2, len(eligible) - min_needed))
    marker_total = min_needed + random.randint(0, extra_cap)
    marker_colors = list(colors)
    marker_colors.extend(random.choice(tuple(colors)) for _ in range(marker_total - min_needed))
    random.shuffle(marker_colors)

    chosen = random.sample(eligible, len(marker_colors))
    for (dr, dc), color in zip(chosen, marker_colors):
        grid[r0 + dr, c0 + dc] = int(color)


def _expected_output(
    grid: np.ndarray,
    room_sets: list[set[int]],
    heights: list[int],
    widths: list[int],
    edges: list[dict[str, int | str]],
) -> np.ndarray:
    out = grid.copy()
    rects = _room_rects(heights, widths)
    for edge in edges:
        a = int(edge["a"])
        b = int(edge["b"])
        common = room_sets[a] & room_sets[b]
        if len(common) != 1:
            continue
        r0, c0, height, width = rects[a]
        if edge["orientation"] == "V":
            door_r = r0 + (height - 1) // 2
            door_c = c0 + width
        else:
            door_r = r0 + height
            door_c = c0 + (width - 1) // 2
        out[door_r, door_c] = next(iter(common))
    return out


def _counts_are_valid(grid: np.ndarray, bg: int, wall: int) -> bool:
    counts = Counter(map(int, grid.reshape(-1)))
    if counts[bg] != max(counts.values()):
        return False
    if list(counts.values()).count(counts[bg]) != 1:
        return False

    non_bg_counts = [count for color, count in counts.items() if color != bg]
    if not non_bg_counts or counts[wall] != max(non_bg_counts):
        return False
    if sum(1 for color, count in counts.items() if color != bg and count == counts[wall]) != 1:
        return False
    return True


def _generate_layout() -> tuple[list[int], list[int]]:
    room_rows, room_cols = random.choice(LAYOUTS)
    heights = [random.randint(*ROOM_SIZE_RANGE) for _ in range(room_rows)]
    widths = [random.randint(*ROOM_SIZE_RANGE) for _ in range(room_cols)]
    return heights, widths


def _build_candidate() -> np.ndarray | None:
    heights, widths = _generate_layout()
    room_count = len(heights) * len(widths)
    edges = _adjacency_edges(heights, widths)
    active_edges = _choose_active_edges(edges)
    if active_edges is None:
        return None

    bg, wall = random.sample(COLOR_IDS, 2)
    marker_palette_size = random.randint(
        PALETTE_SIZE_RANGE[0],
        min(PALETTE_SIZE_RANGE[1], len(COLOR_IDS) - 2),
    )
    marker_palette = random.sample([c for c in COLOR_IDS if c not in {bg, wall}], marker_palette_size)

    room_sets = None
    for _ in range(SET_ASSIGNMENT_ATTEMPTS):
        room_sets = _assign_room_sets(room_count, edges, active_edges, marker_palette)
        if room_sets is not None:
            break
    if room_sets is None:
        return None

    height = sum(heights) + len(heights) + 1
    width = sum(widths) + len(widths) + 1
    grid = np.full((height, width), wall, dtype=int)
    rects = _room_rects(heights, widths)
    for room_id, rect in rects.items():
        r0, c0, h, w = rect
        grid[r0 : r0 + h, c0 : c0 + w] = bg
        _place_room_markers(grid, rect, room_sets[room_id])

    if not _counts_are_valid(grid, bg, wall):
        return None

    expected = _expected_output(grid, room_sets, heights, widths, edges)
    if np.array_equal(grid, expected):
        return None
    if not np.array_equal(main(grid), expected):
        return None
    return grid


def generate_input() -> np.ndarray:
    for _ in range(MAX_GENERATION_ATTEMPTS):
        grid = _build_candidate()
        if grid is not None:
            return grid
    raise RuntimeError("failed to generate a valid room-door ARC input")
