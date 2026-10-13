import random
from collections import deque

import numpy as np


BLACK = 0
GREY = 5
VISIBLE_COLORS = (1, 2, 3, 4, 6, 7, 8, 9)
GEN_RETRIES = 400
_RNG = random.Random()


def _components(mask: np.ndarray) -> list[np.ndarray]:
    h, w = mask.shape
    seen = np.zeros((h, w), dtype=bool)
    comps: list[np.ndarray] = []
    for r in range(h):
        for c in range(w):
            if not mask[r, c] or seen[r, c]:
                continue
            queue = deque([(r, c)])
            seen[r, c] = True
            pts: list[tuple[int, int]] = []
            while queue:
                rr, cc = queue.popleft()
                pts.append((rr, cc))
                for dr, dc in ((1, 0), (-1, 0), (0, 1), (0, -1)):
                    nr, nc = rr + dr, cc + dc
                    if 0 <= nr < h and 0 <= nc < w and mask[nr, nc] and not seen[nr, nc]:
                        seen[nr, nc] = True
                        queue.append((nr, nc))
            comps.append(np.array(pts, dtype=int))
    return comps


def _count_enclosed_black_regions(area: np.ndarray) -> int:
    mask = area == BLACK
    h, w = mask.shape
    if h == 0 or w == 0:
        return 0
    seen = np.zeros((h, w), dtype=bool)
    count = 0
    for r in range(h):
        for c in range(w):
            if not mask[r, c] or seen[r, c]:
                continue
            queue = deque([(r, c)])
            seen[r, c] = True
            touches_edge = False
            while queue:
                rr, cc = queue.popleft()
                if rr in (0, h - 1) or cc in (0, w - 1):
                    touches_edge = True
                for dr, dc in ((1, 0), (-1, 0), (0, 1), (0, -1)):
                    nr, nc = rr + dr, cc + dc
                    if 0 <= nr < h and 0 <= nc < w and mask[nr, nc] and not seen[nr, nc]:
                        seen[nr, nc] = True
                        queue.append((nr, nc))
            if not touches_edge:
                count += 1
    return count


def _extract_legend(grid: np.ndarray) -> tuple[list[int], int]:
    h, _ = grid.shape
    legend_row = None
    for r in range(h - 1, -1, -1):
        if np.any(grid[r] != BLACK):
            legend_row = r
            break
    assert legend_row is not None, "expected a non-empty legend row"
    cols = np.flatnonzero(grid[legend_row] != BLACK)
    assert len(cols) >= 2, "legend must contain at least two colors"
    assert np.all(np.diff(cols) >= 2), "legend colors must be separated by black cells"
    return [int(grid[legend_row, c]) for c in cols], int(legend_row)


def _find_frames(grid: np.ndarray, legend_row: int) -> list[tuple[int, int, int, int]]:
    search = grid[:legend_row, :]
    frames: list[tuple[int, int, int, int]] = []
    for pts in _components(search == GREY):
        r0 = int(pts[:, 0].min())
        r1 = int(pts[:, 0].max())
        c0 = int(pts[:, 1].min())
        c1 = int(pts[:, 1].max())
        if r1 - r0 < 2 or c1 - c0 < 2:
            continue
        box = search[r0 : r1 + 1, c0 : c1 + 1]
        perimeter = 2 * (box.shape[0] + box.shape[1]) - 4
        if len(pts) != perimeter:
            continue
        if not np.all(box[0, :] == GREY):
            continue
        if not np.all(box[-1, :] == GREY):
            continue
        if not np.all(box[:, 0] == GREY):
            continue
        if not np.all(box[:, -1] == GREY):
            continue
        if box.shape[0] <= 2 or box.shape[1] <= 2:
            continue
        if np.any(box[1:-1, 1:-1] == GREY):
            continue
        frames.append((r0, c0, r1, c1))
    return frames


def main(grid: np.ndarray) -> np.ndarray:
    """
    Recolor each grey-framed region by looking up the number of internal holes in the bottom legend.

    Input structure:
    - The background is black.
    - The bottom row contains a legend of distinct single-pixel colors ordered left-to-right, with one black cell between neighboring legend pixels.
    - Above the legend are several axis-aligned grey rectangular frames whose interiors are filled with non-black, non-grey colors.
    - Some frame interiors contain one or more black holes fully enclosed by the interior fill.

    Rule:
    - Keep the black background and every grey border unchanged.
    - For each grey rectangle, count how many enclosed black hole-components its interior contains.
    - Use that count as a zero-based index into the bottom legend.
    - Recolor every non-black interior pixel of the rectangle to the selected legend color, while preserving the black holes.

    Key insight:
    - The holes are the lookup key: their count indexes the ordered legend, which supplies the replacement color for the whole framed region.
    """
    g = np.asarray(grid, dtype=int)
    legend, legend_row = _extract_legend(g)
    out = g.copy()
    for r0, c0, r1, c1 in _find_frames(g, legend_row):
        interior = g[r0 + 1 : r1, c0 + 1 : c1]
        hole_count = _count_enclosed_black_regions(interior)
        assert 0 <= hole_count < len(legend), "hole count must index the legend"
        target = legend[hole_count]
        dest = out[r0 + 1 : r1, c0 + 1 : c1]
        dest[dest != BLACK] = target
    return out


HOLE_SHAPES = (
    np.array([(0, 0)], dtype=int),
    np.array([(0, 0), (0, 1)], dtype=int),
    np.array([(0, 0), (1, 0)], dtype=int),
    np.array([(0, 0), (0, 1), (1, 0)], dtype=int),
    np.array([(0, 0), (0, 1), (1, 0), (1, 1)], dtype=int),
)


def _fill_color_for(target: int, rng: random.Random) -> int:
    choices = [color for color in VISIBLE_COLORS if color != target]
    return int(rng.choice(choices))


def _try_place_holes(inner_h: int, inner_w: int, hole_count: int, rng: random.Random) -> list[np.ndarray] | None:
    if hole_count == 0:
        return []
    for _ in range(80):
        blocked = np.zeros((inner_h, inner_w), dtype=bool)
        placed: list[np.ndarray] = []
        success = True
        for _hole_idx in range(hole_count):
            candidates: list[np.ndarray] = []
            shapes = list(HOLE_SHAPES)
            rng.shuffle(shapes)
            for shape in shapes:
                sh = int(shape[:, 0].max()) + 1
                sw = int(shape[:, 1].max()) + 1
                if inner_h - sh < 2 or inner_w - sw < 2:
                    continue
                for top in range(1, inner_h - sh):
                    for left in range(1, inner_w - sw):
                        cells = shape + np.array([top, left], dtype=int)
                        if any(blocked[r, c] for r, c in cells):
                            continue
                        candidates.append(cells)
            if not candidates:
                success = False
                break
            cells = rng.choice(candidates)
            placed.append(cells.copy())
            for r, c in cells:
                for nr in range(r - 1, r + 2):
                    for nc in range(c - 1, c + 2):
                        if 0 <= nr < inner_h and 0 <= nc < inner_w:
                            blocked[nr, nc] = True
        if success:
            return placed
    return None


def _make_room(hole_count: int, target_color: int, rng: random.Random) -> np.ndarray | None:
    for _ in range(120):
        if hole_count == 0:
            inner_h = rng.randint(2, 7)
            inner_w = rng.randint(2, 8)
        else:
            inner_h = rng.randint(5, 8)
            inner_w = rng.randint(5, 9)
        holes = _try_place_holes(inner_h, inner_w, hole_count, rng)
        if holes is None:
            continue
        fill = _fill_color_for(target_color, rng)
        room = np.full((inner_h + 2, inner_w + 2), fill, dtype=int)
        room[0, :] = GREY
        room[-1, :] = GREY
        room[:, 0] = GREY
        room[:, -1] = GREY
        for cells in holes:
            for r, c in cells:
                room[r + 1, c + 1] = BLACK
        if _count_enclosed_black_regions(room[1:-1, 1:-1]) != hole_count:
            continue
        return room
    return None


def _place_rooms(
    grid_h: int, grid_w: int, rooms: list[np.ndarray], rng: random.Random
) -> list[tuple[int, int]] | None:
    top_h = grid_h - 2
    blocked = np.zeros((top_h, grid_w), dtype=bool)
    placements: list[tuple[int, int] | None] = [None] * len(rooms)
    order = sorted(
        range(len(rooms)),
        key=lambda idx: int(rooms[idx].shape[0] * rooms[idx].shape[1]),
        reverse=True,
    )
    for idx in order:
        room = rooms[idx]
        rh, rw = map(int, room.shape)
        if rh > top_h or rw > grid_w:
            return None
        candidates: list[tuple[int, int]] = []
        all_positions = [(r, c) for r in range(top_h - rh + 1) for c in range(grid_w - rw + 1)]
        rng.shuffle(all_positions)
        for r, c in all_positions:
            rr0 = max(0, r - 1)
            rr1 = min(top_h, r + rh + 1)
            cc0 = max(0, c - 1)
            cc1 = min(grid_w, c + rw + 1)
            if blocked[rr0:rr1, cc0:cc1].any():
                continue
            candidates.append((r, c))
            if len(candidates) >= 16:
                break
        if not candidates:
            return None
        r, c = rng.choice(candidates)
        placements[idx] = (r, c)
        rr0 = max(0, r - 1)
        rr1 = min(top_h, r + rh + 1)
        cc0 = max(0, c - 1)
        cc1 = min(grid_w, c + rw + 1)
        blocked[rr0:rr1, cc0:cc1] = True
    if any(pos is None for pos in placements):
        return None
    return [(int(pos[0]), int(pos[1])) for pos in placements]


def _gen_attempt(rng: random.Random) -> np.ndarray | None:
    legend_len = rng.randint(3, 5)
    legend_colors = rng.sample(VISIBLE_COLORS, legend_len)
    rooms = []
    for hole_count, target_color in enumerate(legend_colors):
        room = _make_room(hole_count, int(target_color), rng)
        if room is None:
            return None
        rooms.append(room)

    legend_width = 2 * legend_len - 1
    padded_area = sum((room.shape[0] + 2) * (room.shape[1] + 2) for room in rooms)
    max_room_h = max(int(room.shape[0]) for room in rooms)
    max_room_w = max(int(room.shape[1]) for room in rooms)
    side_hint = int(np.ceil(np.sqrt(padded_area)))
    min_h = max(max_room_h + 2, side_hint // 2 + 6)
    min_w = max(max_room_w, legend_width, side_hint // 2 + 8)
    max_h = min(30, max(min_h, side_hint + 6))
    max_w = min(30, max(min_w, side_hint + 8))
    if min_h > max_h or min_w > max_w:
        return None

    for _ in range(100):
        grid_h = rng.randint(min_h, max_h)
        grid_w = rng.randint(min_w, max_w)
        placements = _place_rooms(grid_h, grid_w, rooms, rng)
        if placements is None:
            continue
        grid = np.zeros((grid_h, grid_w), dtype=int)
        legend_start = rng.randint(0, grid_w - legend_width)
        for idx, color in enumerate(legend_colors):
            grid[grid_h - 1, legend_start + 2 * idx] = int(color)
        for room, (r, c) in zip(rooms, placements):
            rh, rw = room.shape
            grid[r : r + rh, c : c + rw] = room
        solved = main(grid)
        if np.array_equal(solved, grid):
            continue
        return grid
    return None


def generate_input() -> np.ndarray:
    for _ in range(GEN_RETRIES):
        grid = _gen_attempt(_RNG)
        if grid is not None:
            return grid
    raise RuntimeError("failed to generate a valid puzzle instance")
