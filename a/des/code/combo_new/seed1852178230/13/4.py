from __future__ import annotations

import random

import numpy as np


BLACK = 0
GREY = 5
AZURE = 8
COLOR_POOL = (1, 2, 3, 4, 6, 7, 9)
SIDES = ("top", "bottom", "left", "right")
OPPOSITE = {"top": "bottom", "bottom": "top", "left": "right", "right": "left"}
MAX_GENERATION_ATTEMPTS = 400
ROOM_COUNT_CHOICES = (2, 3, 4)
ROOM_SIZE_RANGE = (3, 6)


def _components(mask: np.ndarray) -> list[list[tuple[int, int]]]:
    h, w = mask.shape
    seen = np.zeros((h, w), dtype=bool)
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


def _room_side_values(grid: np.ndarray, bbox: tuple[int, int, int, int]) -> dict[str, np.ndarray]:
    r0, c0, r1, c1 = bbox
    return {
        "top": grid[r0 - 1, c0 : c1 + 1],
        "bottom": grid[r1 + 1, c0 : c1 + 1],
        "left": grid[r0 : r1 + 1, c0 - 1],
        "right": grid[r0 : r1 + 1, c1 + 1],
    }


def _find_rooms(grid: np.ndarray) -> list[dict]:
    non_wall = (grid != GREY) & (grid != AZURE)
    rooms: list[dict] = []
    for comp in _components(non_wall):
        rs = [r for r, _ in comp]
        cs = [c for _, c in comp]
        r0, r1 = min(rs), max(rs)
        c0, c1 = min(cs), max(cs)

        if r0 == 0 or c0 == 0 or r1 == grid.shape[0] - 1 or c1 == grid.shape[1] - 1:
            continue
        if not np.all(non_wall[r0 : r1 + 1, c0 : c1 + 1]):
            continue

        side_arrays = _room_side_values(grid, (r0, c0, r1, c1))
        mirror_sides: list[str] = []
        valid = True
        for side, arr in side_arrays.items():
            values = {int(v) for v in np.asarray(arr).ravel()}
            if values == {GREY}:
                continue
            if values == {AZURE}:
                mirror_sides.append(side)
                continue
            valid = False
            break
        if not valid or len(mirror_sides) != 1:
            continue

        colored: list[tuple[int, int, int]] = []
        for rr, cc in np.argwhere((grid[r0 : r1 + 1, c0 : c1 + 1] != BLACK)):
            color = int(grid[r0 + rr, c0 + cc])
            if color in {GREY, AZURE}:
                valid = False
                break
            colored.append((r0 + int(rr), c0 + int(cc), color))
        if not valid or not colored:
            continue

        rooms.append(
            {
                "bbox": (r0, c0, r1, c1),
                "side": mirror_sides[0],
                "colored": colored,
            }
        )

    rooms.sort(key=lambda room: room["bbox"])
    assert rooms, "expected at least one room"
    return rooms


def _mirror_position(r: int, c: int, bbox: tuple[int, int, int, int], side: str) -> tuple[int, int]:
    r0, c0, r1, c1 = bbox
    if side in {"top", "bottom"}:
        return r0 + r1 - r, c
    return r, c0 + c1 - c


def main(grid: np.ndarray) -> np.ndarray:
    """
    Reflect each room's colored pixels across the room axis indicated by its unique azure wall.

    Input structure:
    - The background is black.
    - Grey one-pixel walls partition the grid into disjoint rectangular rooms, and each room has exactly one wall segment recolored azure.
    - Each room interior is otherwise black except for 1-3 sparse colored pixels whose colors are neither black, grey, nor azure.

    Rule:
    - Detect every rectangular room interior as a non-wall rectangle bounded on all four sides by wall segments.
    - Read the room's unique azure wall: a top or bottom azure wall means mirror the colored pixels to the vertically opposite positions inside the same room, while a left or right azure wall means mirror them horizontally.
    - Keep the original pixels and all walls unchanged, and paint same-colored copies at the mirrored interior coordinates.

    Key insight:
    - The only local cue is the single grey-to-azure wall change on each room border; that wall does not add a new object, it encodes which reflection axis to use inside the room.
    """
    g = np.asarray(grid, dtype=int)
    assert g.ndim == 2, "expected a 2D grid"

    out = g.copy()
    for room in _find_rooms(g):
        bbox = room["bbox"]
        side = room["side"]
        r0, c0, r1, c1 = bbox
        for r, c, color in room["colored"]:
            rr, cc = _mirror_position(r, c, bbox, side)
            if r0 <= rr <= r1 and c0 <= cc <= c1:
                out[rr, cc] = int(color)
    return out


def _solve_wall_assignment(room_rows: int, room_cols: int) -> dict[tuple[int, int], str]:
    cells = [(r, c) for r in range(room_rows) for c in range(room_cols)]
    index = {cell: idx for idx, cell in enumerate(cells)}
    neighbors: list[list[tuple[str, int]]] = [[] for _ in cells]
    boundaries: list[list[str]] = [[] for _ in cells]

    for r, c in cells:
        idx = index[(r, c)]
        if r > 0:
            neighbors[idx].append(("top", index[(r - 1, c)]))
        if r + 1 < room_rows:
            neighbors[idx].append(("bottom", index[(r + 1, c)]))
        if c > 0:
            neighbors[idx].append(("left", index[(r, c - 1)]))
        if c + 1 < room_cols:
            neighbors[idx].append(("right", index[(r, c + 1)]))

        if r == 0:
            boundaries[idx].append("top")
        if r == room_rows - 1:
            boundaries[idx].append("bottom")
        if c == 0:
            boundaries[idx].append("left")
        if c == room_cols - 1:
            boundaries[idx].append("right")

    full_mask = (1 << len(cells)) - 1

    def dfs(mask: int, side_map: dict[int, str], failed: set[int]) -> dict[int, str] | None:
        if mask == full_mask:
            return side_map.copy()
        if mask in failed:
            return None

        best_idx = -1
        best_options = None
        for idx in range(len(cells)):
            if (mask >> idx) & 1:
                continue
            pair_count = sum(((mask >> nidx) & 1) == 0 for _side, nidx in neighbors[idx])
            option_count = pair_count + len(boundaries[idx])
            if option_count == 0:
                failed.add(mask)
                return None
            if best_options is None or option_count < best_options:
                best_idx = idx
                best_options = option_count

        pair_options = [(side, nidx) for side, nidx in neighbors[best_idx] if ((mask >> nidx) & 1) == 0]
        boundary_options = list(boundaries[best_idx])
        random.shuffle(pair_options)
        random.shuffle(boundary_options)
        use_pairs_first = random.random() < 0.75
        ordered_options: list[tuple[str, str, int | None]] = []
        if use_pairs_first:
            ordered_options.extend(("pair", side, nidx) for side, nidx in pair_options)
            ordered_options.extend(("boundary", side, None) for side in boundary_options)
        else:
            ordered_options.extend(("boundary", side, None) for side in boundary_options)
            ordered_options.extend(("pair", side, nidx) for side, nidx in pair_options)

        for kind, side, other_idx in ordered_options:
            if kind == "pair":
                assert other_idx is not None
                side_map[best_idx] = side
                side_map[other_idx] = OPPOSITE[side]
                result = dfs(mask | (1 << best_idx) | (1 << other_idx), side_map, failed)
                if result is not None:
                    return result
                del side_map[best_idx]
                del side_map[other_idx]
            else:
                side_map[best_idx] = side
                result = dfs(mask | (1 << best_idx), side_map, failed)
                if result is not None:
                    return result
                del side_map[best_idx]

        failed.add(mask)
        return None

    for _ in range(120):
        solution = dfs(0, {}, set())
        if solution is None:
            continue
        by_cell = {cells[idx]: side for idx, side in solution.items()}
        labels = set(by_cell.values())
        if labels & {"top", "bottom"} and labels & {"left", "right"}:
            return by_cell
    raise RuntimeError("failed to assign one mirror wall per room")


def _build_layout(room_rows: int, room_cols: int) -> tuple[list[tuple[int, int]], list[tuple[int, int]], int, int]:
    heights = [random.randint(*ROOM_SIZE_RANGE) for _ in range(room_rows)]
    widths = [random.randint(*ROOM_SIZE_RANGE) for _ in range(room_cols)]

    row_bounds: list[tuple[int, int]] = []
    cursor = 0
    for height in heights:
        r0 = cursor + 1
        r1 = r0 + height - 1
        row_bounds.append((r0, r1))
        cursor = r1 + 1
    height_total = cursor + 1

    col_bounds: list[tuple[int, int]] = []
    cursor = 0
    for width in widths:
        c0 = cursor + 1
        c1 = c0 + width - 1
        col_bounds.append((c0, c1))
        cursor = c1 + 1
    width_total = cursor + 1

    return row_bounds, col_bounds, height_total, width_total


def _paint_walls(grid: np.ndarray, row_bounds: list[tuple[int, int]], col_bounds: list[tuple[int, int]]) -> None:
    wall_rows = [0] + [r1 + 1 for _r0, r1 in row_bounds]
    wall_cols = [0] + [c1 + 1 for _c0, c1 in col_bounds]
    grid[wall_rows, :] = GREY
    grid[:, wall_cols] = GREY


def _room_bbox(row_bounds: list[tuple[int, int]], col_bounds: list[tuple[int, int]], room: tuple[int, int]) -> tuple[int, int, int, int]:
    rr, cc = room
    r0, r1 = row_bounds[rr]
    c0, c1 = col_bounds[cc]
    return r0, c0, r1, c1


def _paint_azure_wall(grid: np.ndarray, bbox: tuple[int, int, int, int], side: str) -> None:
    r0, c0, r1, c1 = bbox
    if side == "top":
        grid[r0 - 1, c0 : c1 + 1] = AZURE
    elif side == "bottom":
        grid[r1 + 1, c0 : c1 + 1] = AZURE
    elif side == "left":
        grid[r0 : r1 + 1, c0 - 1] = AZURE
    else:
        grid[r0 : r1 + 1, c1 + 1] = AZURE


def _candidate_positions(bbox: tuple[int, int, int, int], side: str) -> list[tuple[int, int]]:
    r0, c0, r1, c1 = bbox
    candidates: list[tuple[int, int]] = []
    for r in range(r0, r1 + 1):
        for c in range(c0, c1 + 1):
            rr, cc = _mirror_position(r, c, bbox, side)
            if (r, c) == (rr, cc):
                continue
            if side == "top" and r < rr:
                candidates.append((r, c))
            elif side == "bottom" and r > rr:
                candidates.append((r, c))
            elif side == "left" and c < cc:
                candidates.append((r, c))
            elif side == "right" and c > cc:
                candidates.append((r, c))
    return candidates


def _generate_attempt() -> np.ndarray | None:
    room_rows = random.choice(ROOM_COUNT_CHOICES)
    room_cols = random.choice(ROOM_COUNT_CHOICES)
    row_bounds, col_bounds, height_total, width_total = _build_layout(room_rows, room_cols)
    if height_total > 30 or width_total > 30:
        return None

    side_map = _solve_wall_assignment(room_rows, room_cols)

    grid = np.full((height_total, width_total), BLACK, dtype=int)
    expected = grid.copy()
    _paint_walls(grid, row_bounds, col_bounds)
    _paint_walls(expected, row_bounds, col_bounds)

    for room, side in side_map.items():
        bbox = _room_bbox(row_bounds, col_bounds, room)
        _paint_azure_wall(grid, bbox, side)
        _paint_azure_wall(expected, bbox, side)

    for room, side in side_map.items():
        bbox = _room_bbox(row_bounds, col_bounds, room)
        candidates = _candidate_positions(bbox, side)
        if not candidates:
            return None
        count = random.randint(1, min(3, len(candidates)))
        chosen_positions = random.sample(candidates, count)
        colors = random.sample(COLOR_POOL, count)
        for (r, c), color in zip(chosen_positions, colors):
            rr, cc = _mirror_position(r, c, bbox, side)
            grid[r, c] = color
            expected[r, c] = color
            expected[rr, cc] = color

    if not np.array_equal(main(grid), expected):
        return None
    return grid


def generate_input() -> np.ndarray:
    for _ in range(MAX_GENERATION_ATTEMPTS):
        grid = _generate_attempt()
        if grid is not None:
            return grid
    raise RuntimeError("failed to generate a valid mirror-wall room puzzle")
