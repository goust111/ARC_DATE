import random
from collections import deque

import numpy as np


BLACK = 0
GREEN = 3
ANCHOR_COLORS = (1, 2, 4, 6, 7, 8, 9)
DIRS4 = [(-1, 0), (1, 0), (0, -1), (0, 1)]

GEN_ROWS_RANGE = (19, 24)
GEN_COLS_RANGE = (19, 24)
GEN_FRAME_COUNT_OPTIONS = (2, 3)
GEN_FRAME_HEIGHT_RANGE = (6, 10)
GEN_FRAME_WIDTH_RANGE = (6, 10)
GEN_FRAME_MARGIN = 1
GEN_FRAME_GAP = 1
GEN_TOTAL_ANCHORS_RANGE = (4, 6)
GEN_RETRIES = 400


def _in_bounds(shape: tuple[int, int], r: int, c: int) -> bool:
    rows, cols = shape
    return 0 <= r < rows and 0 <= c < cols


def _neighbors4(r: int, c: int):
    for dr, dc in DIRS4:
        yield r + dr, c + dc


def _components(mask: np.ndarray) -> list[np.ndarray]:
    rows, cols = mask.shape
    seen = np.zeros_like(mask, dtype=bool)
    parts: list[np.ndarray] = []
    for r in range(rows):
        for c in range(cols):
            if not mask[r, c] or seen[r, c]:
                continue
            q = deque([(r, c)])
            seen[r, c] = True
            coords = []
            while q:
                cr, cc = q.popleft()
                coords.append((cr, cc))
                for nr, nc in _neighbors4(cr, cc):
                    if _in_bounds(mask.shape, nr, nc) and mask[nr, nc] and not seen[nr, nc]:
                        seen[nr, nc] = True
                        q.append((nr, nc))
            parts.append(np.asarray(coords, dtype=int))
    return parts


def _bbox(coords: np.ndarray) -> tuple[int, int, int, int]:
    r0, c0 = coords.min(axis=0)
    r1, c1 = coords.max(axis=0)
    return int(r0), int(c0), int(r1), int(c1)


def _perimeter_cells(frame: tuple[int, int, int, int]) -> set[tuple[int, int]]:
    r0, c0, r1, c1 = frame
    cells = {(r0, c) for c in range(c0, c1 + 1)}
    cells |= {(r1, c) for c in range(c0, c1 + 1)}
    cells |= {(r, c0) for r in range(r0, r1 + 1)}
    cells |= {(r, c1) for r in range(r0, r1 + 1)}
    return cells


def _find_frames(grid: np.ndarray) -> list[tuple[int, int, int, int]]:
    green_parts = _components(grid == GREEN)
    frames = []
    for coords in green_parts:
        frame = _bbox(coords)
        r0, c0, r1, c1 = frame
        assert r1 - r0 >= 2 and c1 - c0 >= 2, "green component must be a hollow rectangle"
        got = {tuple(map(int, cell)) for cell in coords}
        expect = _perimeter_cells(frame)
        assert got == expect, "green pixels must form complete 1-pixel frame borders"
        assert not np.any(grid[r0 + 1 : r1, c0 + 1 : c1] == GREEN), "frame interior must stay empty"
        frames.append(frame)
    assert frames, "expected at least one green frame"
    return sorted(frames)


def _anchors(grid: np.ndarray) -> list[tuple[int, int, int]]:
    coords = np.argwhere((grid != BLACK) & (grid != GREEN))
    anchors = [(int(r), int(c), int(grid[r, c])) for r, c in coords]
    assert anchors, "expected colored anchor pixels"
    return anchors


def _containing_frame(r: int, c: int, frames: list[tuple[int, int, int, int]]):
    owners = [frame for frame in frames if frame[0] < r < frame[2] and frame[1] < c < frame[3]]
    assert len(owners) <= 1, "frames must not overlap"
    return owners[0] if owners else None


def _edge_steps(shape: tuple[int, int], r: int, c: int, dr: int, dc: int) -> int:
    rows, cols = shape
    if dr == -1:
        return r
    if dr == 1:
        return rows - 1 - r
    if dc == -1:
        return c
    return cols - 1 - c


def _decay_reach(edge_steps: int) -> int:
    # Distance d is drawable while moving d cells plus one extra decay cost
    # every second step still fits inside the available travel budget.
    reach = 0
    while True:
        nxt = reach + 1
        cost = nxt + (nxt - 1) // 2
        if cost > edge_steps:
            return reach
        reach = nxt


def _inside_barrier_distance(frame: tuple[int, int, int, int], r: int, c: int, dr: int, dc: int) -> int:
    r0, c0, r1, c1 = frame
    if dr == -1:
        return r - r0 - 1
    if dr == 1:
        return r1 - r - 1
    if dc == -1:
        return c - c0 - 1
    return c1 - c - 1


def _outside_barrier_distance(grid: np.ndarray, r: int, c: int, dr: int, dc: int) -> float:
    steps = 1
    while True:
        nr = r + dr * steps
        nc = c + dc * steps
        if not _in_bounds(grid.shape, nr, nc):
            return float("inf")
        if grid[nr, nc] == GREEN:
            return float(steps - 1)
        steps += 1


def _ray_limit(
    grid: np.ndarray,
    r: int,
    c: int,
    dr: int,
    dc: int,
    frame: tuple[int, int, int, int] | None,
) -> int:
    reach = _decay_reach(_edge_steps(grid.shape, r, c, dr, dc))
    if frame is None:
        barrier = _outside_barrier_distance(grid, r, c, dr, dc)
    else:
        barrier = float(_inside_barrier_distance(frame, r, c, dr, dc))
    return int(min(reach, barrier))


def _classify_anchor_behaviour(
    grid: np.ndarray,
    r: int,
    c: int,
    frame: tuple[int, int, int, int] | None,
) -> tuple[bool, bool]:
    hit_wall = False
    free_decay = False
    for dr, dc in DIRS4:
        reach = _decay_reach(_edge_steps(grid.shape, r, c, dr, dc))
        if frame is None:
            barrier = _outside_barrier_distance(grid, r, c, dr, dc)
        else:
            barrier = float(_inside_barrier_distance(frame, r, c, dr, dc))
        if barrier < reach:
            hit_wall = True
        if reach <= barrier and reach > 0:
            free_decay = True
    return hit_wall, free_decay


def main(grid: np.ndarray) -> np.ndarray:
    """Emit four shrinking cardinal rays from each anchor, with green frames acting as containment-gated barriers.

    Input structure:
    - The background is black.
    - The grid contains 2-3 hollow green rectangular frames with empty black interiors.
    - Several non-green, non-black single-pixel anchors appear; some lie inside a frame and some lie outside every frame.

    Rule:
    - Detect every green rectangular frame and decide for each anchor whether it is enclosed by one frame.
    - From each anchor, draw four 1-pixel-wide rays in the cardinal directions.
    - A ray loses one unit of travel budget every two steps, so its drawable extent shrinks as it moves away from the anchor.
    - For anchors inside a frame, the containing frame wall truncates the ray before the wall pixel.
    - For anchors outside all frames, the ray continues until its decay budget ends, the grid edge is reached, or a green wall is encountered; it never paints onto a wall pixel.
    - When different anchors paint the same cell, keep the color from the nearer anchor by Manhattan distance; equal distances prefer the anchor with the smaller row.

    Key insight:
    - The same shrinking-ray rule produces two visual regimes: enclosed anchors are cut short by their frame walls, while exterior anchors produce longer fading tails unless a wall blocks their path.
    """
    g = np.asarray(grid, dtype=int)
    assert g.ndim == 2, "expected a 2D grid"

    frames = _find_frames(g)
    anchors = _anchors(g)

    out = g.copy()
    best: dict[tuple[int, int], tuple[tuple[int, int, int, int], int]] = {}
    for anchor_index, (r, c, color) in enumerate(anchors):
        frame = _containing_frame(r, c, frames)
        for dr, dc in DIRS4:
            limit = _ray_limit(g, r, c, dr, dc, frame)
            for dist in range(limit + 1):
                rr = r + dr * dist
                cc = c + dc * dist
                if g[rr, cc] == GREEN:
                    break
                score = (dist, r, c, anchor_index)
                key = (rr, cc)
                if key not in best or score < best[key][0]:
                    best[key] = (score, color)

    for (r, c), (_score, color) in best.items():
        out[r, c] = color
    return out


def _boxes_have_gap(a: tuple[int, int, int, int], b: tuple[int, int, int, int], gap: int = GEN_FRAME_GAP) -> bool:
    return (
        a[2] + gap < b[0]
        or b[2] + gap < a[0]
        or a[3] + gap < b[1]
        or b[3] + gap < a[1]
    )


def _place_frames(rows: int, cols: int, count: int) -> list[tuple[int, int, int, int]] | None:
    sizes = [
        (
            random.randint(*GEN_FRAME_HEIGHT_RANGE),
            random.randint(*GEN_FRAME_WIDTH_RANGE),
        )
        for _ in range(count)
    ]
    sizes.sort(key=lambda hw: hw[0] * hw[1], reverse=True)
    frames: list[tuple[int, int, int, int]] = []

    for height, width in sizes:
        candidates = []
        for r0 in range(GEN_FRAME_MARGIN, rows - height - GEN_FRAME_MARGIN + 1):
            for c0 in range(GEN_FRAME_MARGIN, cols - width - GEN_FRAME_MARGIN + 1):
                box = (r0, c0, r0 + height - 1, c0 + width - 1)
                if all(_boxes_have_gap(box, other) for other in frames):
                    candidates.append(box)
        if not candidates:
            return None
        frames.append(random.choice(candidates))
    return frames


def _draw_frames(shape: tuple[int, int], frames: list[tuple[int, int, int, int]]) -> np.ndarray:
    grid = np.zeros(shape, dtype=int)
    for frame in frames:
        for r, c in _perimeter_cells(frame):
            grid[r, c] = GREEN
    return grid


def _frame_wall_cells(frames: list[tuple[int, int, int, int]]) -> list[tuple[int, int]]:
    cells: set[tuple[int, int]] = set()
    for frame in frames:
        cells |= _perimeter_cells(frame)
    return sorted(cells)


def _outside_of_all_frames(r: int, c: int, frames: list[tuple[int, int, int, int]]) -> bool:
    return all(not (frame[0] < r < frame[2] and frame[1] < c < frame[3]) for frame in frames)


def _min_wall_distance(r: int, c: int, wall_cells: list[tuple[int, int]]) -> int:
    return min(abs(r - wr) + abs(c - wc) for wr, wc in wall_cells)


def _far_from_anchors(r: int, c: int, anchors: list[tuple[int, int, int]]) -> bool:
    return all(max(abs(r - ar), abs(c - ac)) >= 2 for ar, ac, _ in anchors)


def _inside_candidates(frame: tuple[int, int, int, int]) -> list[tuple[int, int]]:
    r0, c0, r1, c1 = frame
    return [(r, c) for r in range(r0 + 2, r1 - 1) for c in range(c0 + 2, c1 - 1)]


def _pick_inside_anchors(frames: list[tuple[int, int, int, int]]) -> list[tuple[int, int]]:
    frame_ids = list(range(len(frames)))
    random.shuffle(frame_ids)

    chosen: list[tuple[int, int]] = []
    per_frame = [0] * len(frames)

    required_frames = frame_ids[: min(2, len(frame_ids))]
    for idx in required_frames:
        candidates = _inside_candidates(frames[idx])
        random.shuffle(candidates)
        for r, c in candidates:
            if _far_from_anchors(r, c, [(ar, ac, 1) for ar, ac in chosen]):
                chosen.append((r, c))
                per_frame[idx] += 1
                break
        else:
            return []

    extra = random.randint(0, min(2, 2 * len(frames) - len(chosen)))
    for _ in range(extra):
        frame_order = frame_ids[:]
        random.shuffle(frame_order)
        placed = False
        for idx in frame_order:
            if per_frame[idx] >= 2:
                continue
            candidates = _inside_candidates(frames[idx])
            random.shuffle(candidates)
            for r, c in candidates:
                if _far_from_anchors(r, c, [(ar, ac, 1) for ar, ac in chosen]):
                    chosen.append((r, c))
                    per_frame[idx] += 1
                    placed = True
                    break
            if placed:
                break
        if not placed:
            break
    return chosen


def _blocked_outside_candidates(
    shape: tuple[int, int], frames: list[tuple[int, int, int, int]], wall_cells: list[tuple[int, int]]
) -> list[tuple[int, int]]:
    rows, cols = shape
    out: set[tuple[int, int]] = set()
    for r0, c0, r1, c1 in frames:
        for dist in (3, 4, 5, 6):
            for c in range(c0 + 1, c1):
                for r in (r0 - dist, r1 + dist):
                    if 0 <= r < rows:
                        out.add((r, c))
            for r in range(r0 + 1, r1):
                for c in (c0 - dist, c1 + dist):
                    if 0 <= c < cols:
                        out.add((r, c))
    return [
        (r, c)
        for r, c in out
        if _outside_of_all_frames(r, c, frames) and _min_wall_distance(r, c, wall_cells) >= 3
    ]


def _general_outside_candidates(
    shape: tuple[int, int], frames: list[tuple[int, int, int, int]], wall_cells: list[tuple[int, int]]
) -> list[tuple[int, int]]:
    rows, cols = shape
    candidates = []
    for r in range(rows):
        for c in range(cols):
            if not _outside_of_all_frames(r, c, frames):
                continue
            if _min_wall_distance(r, c, wall_cells) < 3:
                continue
            candidates.append((r, c))
    return candidates


def _choose_outside_anchors(
    grid_frames: np.ndarray,
    frames: list[tuple[int, int, int, int]],
    anchors: list[tuple[int, int]],
) -> list[tuple[int, int]] | None:
    wall_cells = _frame_wall_cells(frames)
    blocked_pool = _blocked_outside_candidates(grid_frames.shape, frames, wall_cells)
    general_pool = _general_outside_candidates(grid_frames.shape, frames, wall_cells)

    blocked = []
    unblocked = []
    for r, c in general_pool:
        if not _far_from_anchors(r, c, [(ar, ac, 1) for ar, ac in anchors]):
            continue
        hits_wall, free_decay = _classify_anchor_behaviour(grid_frames, r, c, None)
        if hits_wall:
            blocked.append((r, c))
        if free_decay:
            unblocked.append((r, c))

    random.shuffle(blocked_pool)
    random.shuffle(blocked)
    random.shuffle(unblocked)

    chosen: list[tuple[int, int]] = []
    outside_target = 1 if random.random() < 0.35 else 2

    blocked_candidates = [pos for pos in blocked_pool if pos in set(blocked)] + [pos for pos in blocked if pos not in blocked_pool]
    for pos in blocked_candidates:
        r, c = pos
        if _far_from_anchors(r, c, [(ar, ac, 1) for ar, ac in anchors + chosen]):
            chosen.append(pos)
            break
    else:
        return None

    if outside_target >= 2:
        for pos in unblocked:
            r, c = pos
            if pos in chosen:
                continue
            if _far_from_anchors(r, c, [(ar, ac, 1) for ar, ac in anchors + chosen]):
                chosen.append(pos)
                break
        else:
            return None

    return chosen


def _random_transform(grid: np.ndarray) -> np.ndarray:
    turns = random.randint(0, 3)
    out = np.rot90(grid, turns)
    if random.random() < 0.5:
        out = np.fliplr(out)
    if random.random() < 0.3:
        out = np.flipud(out)
    return out.copy()


def _attempt() -> np.ndarray | None:
    rows = random.randint(*GEN_ROWS_RANGE)
    cols = random.randint(*GEN_COLS_RANGE)
    frame_count = random.choice(GEN_FRAME_COUNT_OPTIONS)
    frames = _place_frames(rows, cols, frame_count)
    if frames is None:
        return None

    inside = _pick_inside_anchors(frames)
    if len(inside) < 2:
        return None

    grid = _draw_frames((rows, cols), frames)
    outside = _choose_outside_anchors(grid, frames, inside)
    if outside is None or len(outside) < 1:
        return None

    all_anchor_positions = inside + outside
    total_anchors = random.randint(max(4, len(all_anchor_positions)), min(6, len(all_anchor_positions) + 1))
    if len(all_anchor_positions) < total_anchors:
        wall_cells = _frame_wall_cells(frames)
        extra_pool = _general_outside_candidates(grid.shape, frames, wall_cells)
        random.shuffle(extra_pool)
        for r, c in extra_pool:
            if (r, c) in all_anchor_positions:
                continue
            if not _far_from_anchors(r, c, [(ar, ac, 1) for ar, ac in all_anchor_positions]):
                continue
            all_anchor_positions.append((r, c))
            if len(all_anchor_positions) == total_anchors:
                break
    if not (4 <= len(all_anchor_positions) <= 6):
        return None

    colors = random.sample(ANCHOR_COLORS, len(all_anchor_positions))
    for (r, c), color in zip(all_anchor_positions, colors):
        grid[r, c] = color

    grid = _random_transform(grid)
    try:
        out = main(grid)
    except AssertionError:
        return None

    frames_after = _find_frames(grid)
    anchors_after = _anchors(grid)
    inside_count = sum(_containing_frame(r, c, frames_after) is not None for r, c, _ in anchors_after)
    outside_count = len(anchors_after) - inside_count
    if inside_count < 2 or outside_count < 1:
        return None

    outside_behaviours = [
        _classify_anchor_behaviour(grid, r, c, None)
        for r, c, _ in anchors_after
        if _containing_frame(r, c, frames_after) is None
    ]
    if not any(hit for hit, _free in outside_behaviours):
        return None
    if not any(free for _hit, free in outside_behaviours):
        return None

    inside_behaviours = [
        _classify_anchor_behaviour(grid, r, c, _containing_frame(r, c, frames_after))
        for r, c, _ in anchors_after
        if _containing_frame(r, c, frames_after) is not None
    ]
    if not all(hit for hit, _free in inside_behaviours):
        return None

    added = int(np.count_nonzero(out != grid))
    foreground_ratio = float(np.count_nonzero(out) / out.size)
    if added < len(anchors_after) or foreground_ratio > 0.42:
        return None
    return grid


def generate_input() -> np.ndarray:
    for _ in range(GEN_RETRIES):
        grid = _attempt()
        if grid is not None:
            return grid
    raise RuntimeError("failed to generate a valid shrinking-ray containment example")
