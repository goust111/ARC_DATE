from __future__ import annotations

import random
from collections import deque

import numpy as np


BLACK = 0
BLUE = 1
RED = 2
GREY = 5
ORANGE = 7
TEAL = 8

DIRS = {
    "U": (-1, 0),
    "D": (1, 0),
    "L": (0, -1),
    "R": (0, 1),
}

ARM_COLOR = {
    "U": {( -1, 0): TEAL, (1, 0): BLUE, (0, -1): BLUE, (0, 1): BLUE},
    "D": {( -1, 0): BLUE, (1, 0): TEAL, (0, -1): BLUE, (0, 1): BLUE},
    "L": {( -1, 0): BLUE, (1, 0): BLUE, (0, -1): TEAL, (0, 1): BLUE},
    "R": {( -1, 0): BLUE, (1, 0): BLUE, (0, -1): BLUE, (0, 1): TEAL},
}


def _neighbors4(r: int, c: int) -> list[tuple[int, int]]:
    return [(r - 1, c), (r + 1, c), (r, c - 1), (r, c + 1)]


def _components(mask: np.ndarray) -> list[list[tuple[int, int]]]:
    h, w = mask.shape
    seen = np.zeros_like(mask, dtype=bool)
    out: list[list[tuple[int, int]]] = []
    for r in range(h):
        for c in range(w):
            if not mask[r, c] or seen[r, c]:
                continue
            seen[r, c] = True
            q = deque([(r, c)])
            comp: list[tuple[int, int]] = []
            while q:
                cr, cc = q.popleft()
                comp.append((cr, cc))
                for nr, nc in _neighbors4(cr, cc):
                    if 0 <= nr < h and 0 <= nc < w and mask[nr, nc] and not seen[nr, nc]:
                        seen[nr, nc] = True
                        q.append((nr, nc))
            out.append(comp)
    return out


def _block_cells(center: tuple[int, int]) -> list[tuple[int, int]]:
    r, c = center
    return [(rr, cc) for rr in range(r - 1, r + 2) for cc in range(c - 1, c + 2)]


def _obstacle_cells(top_left: tuple[int, int]) -> list[tuple[int, int]]:
    r, c = top_left
    return [(r + dr, c + dc) for dr in range(2) for dc in range(2)]


def _path_cells(start: tuple[int, int], end: tuple[int, int], direction: str) -> set[tuple[int, int]]:
    dr, dc = DIRS[direction]
    cur = start
    out: set[tuple[int, int]] = set()
    while True:
        out.update(_block_cells(cur))
        if cur == end:
            return out
        cur = (cur[0] + dr, cur[1] + dc)


def _min_cell_manhattan(cells_a: list[tuple[int, int]], cells_b: list[tuple[int, int]]) -> int:
    return min(abs(ra - rb) + abs(ca - cb) for ra, ca in cells_a for rb, cb in cells_b)


def _parse_input(grid: np.ndarray) -> tuple[list[dict], list[dict]]:
    g = np.asarray(grid, dtype=int)
    blocks: list[dict] = []
    obstacles: list[dict] = []

    for comp in _components(g != BLACK):
        rs = [r for r, _ in comp]
        cs = [c for _, c in comp]
        r0, r1 = min(rs), max(rs)
        c0, c1 = min(cs), max(cs)
        patch = g[r0 : r1 + 1, c0 : c1 + 1]
        colors = set(int(v) for v in np.unique(patch))
        area = len(comp)

        if area == 9 and (r1 - r0 + 1, c1 - c0 + 1) == (3, 3) and colors <= {GREY, RED}:
            red_cells = [(r, c) for r, c in comp if int(g[r, c]) == RED]
            grey_count = sum(int(g[r, c]) == GREY for r, c in comp)
            center = (r0 + 1, c0 + 1)
            assert len(red_cells) == 1 and grey_count == 8, "each block must be a full 3x3 square with one red marker"
            rr, rc = red_cells[0]
            rel = (rr - center[0], rc - center[1])
            direction_by_rel = {(-1, 0): "U", (1, 0): "D", (0, -1): "L", (0, 1): "R"}
            assert rel in direction_by_rel, "red marker must be orthogonally adjacent to the center"
            blocks.append({"center": center, "direction": direction_by_rel[rel], "bbox": (r0, c0, r1, c1)})
        elif area == 4 and (r1 - r0 + 1, c1 - c0 + 1) == (2, 2) and colors == {ORANGE}:
            obstacles.append({"top_left": (r0, c0)})
        else:
            raise AssertionError("unexpected foreground object in input")

    assert 4 <= len(blocks) <= 6, "expected 4-6 encoded blocks"
    assert 2 <= len(obstacles) <= 4, "expected 2-4 orange obstacles"
    return blocks, obstacles


def _collision_reason(
    shape: tuple[int, int],
    center: tuple[int, int],
    blocked_mask: np.ndarray,
    fixed_mask: np.ndarray,
) -> str | None:
    h, w = shape
    for r, c in _block_cells(center):
        if not (0 <= r < h and 0 <= c < w):
            return "boundary"
    for r, c in _block_cells(center):
        if fixed_mask[r, c]:
            return "obstacle"
    for r, c in _block_cells(center):
        if blocked_mask[r, c]:
            return "block"
    return None


def _slide_distance(
    shape: tuple[int, int],
    start: tuple[int, int],
    direction: str,
    blocked_mask: np.ndarray,
    fixed_mask: np.ndarray,
) -> tuple[int, str]:
    dr, dc = DIRS[direction]
    cur = start
    steps = 0
    while True:
        nxt = (cur[0] + dr, cur[1] + dc)
        reason = _collision_reason(shape, nxt, blocked_mask, fixed_mask)
        if reason is not None:
            return steps, reason
        steps += 1
        cur = nxt


def _simulate(grid: np.ndarray) -> tuple[np.ndarray, list[dict]]:
    g = np.asarray(grid, dtype=int)
    h, w = g.shape
    blocks, obstacles = _parse_input(g)

    fixed_mask = np.zeros((h, w), dtype=bool)
    for obstacle in obstacles:
        for r, c in _obstacle_cells(obstacle["top_left"]):
            fixed_mask[r, c] = True

    ordered = []
    for block in blocks:
        dist, fixed_reason = _slide_distance(g.shape, block["center"], block["direction"], fixed_mask, fixed_mask)
        ordered.append(
            {
                **block,
                "sort_dist": dist,
                "fixed_reason": fixed_reason,
            }
        )
    ordered.sort(key=lambda item: (item["sort_dist"], item["center"][0], item["center"][1]))

    blocked_mask = fixed_mask.copy()
    placed: list[dict] = []
    for block in ordered:
        steps, reason = _slide_distance(g.shape, block["center"], block["direction"], blocked_mask, fixed_mask)
        dr, dc = DIRS[block["direction"]]
        final_center = (block["center"][0] + dr * steps, block["center"][1] + dc * steps)
        for r, c in _block_cells(final_center):
            blocked_mask[r, c] = True
        placed.append(
            {
                **block,
                "final_center": final_center,
                "runtime_steps": steps,
                "stop_reason": reason,
            }
        )

    out = np.zeros_like(g)
    out[fixed_mask] = ORANGE
    for block in placed:
        r, c = block["final_center"]
        out[r - 1 : r + 2, c - 1 : c + 2] = GREY
        out[r, c] = GREY
        for (dr, dc), color in ARM_COLOR[block["direction"]].items():
            out[r + dr, c + dc] = color
    return out, placed


def main(grid: np.ndarray) -> np.ndarray:
    """
    Slide each direction-encoded gray block until collision, then stamp a direction-colored cross at its landing center.

    Input structure:
    - A black background contains 4-6 full 3x3 gray blocks and 2-4 fixed orange 2x2 obstacles.
    - Each gray block has exactly one red pixel in the cell directly above, below, left, or right of its gray center, encoding a compass direction.
    - Blocks are separated in the input, and obstacles do not overlap them.

    Rule:
    - Read each block's motion direction from the red marker.
    - Sort blocks by their free sliding distance to the nearest fixed stopper along that direction, using only orange obstacles and the grid boundary; break ties by original center position from top to bottom, then left to right.
    - Process blocks in that order: slide each one cell at a time until the next step would hit an obstacle, a previously stopped block, or the boundary.
    - At each final center, keep the 3x3 gray square but remove the red marker and recolor the four orthogonal neighbors into a cross: the arm matching the slide direction becomes teal and the other three arms become blue.

    Key insight:
    - The single red marker is consumed twice: once as the motion vector that determines the landing position, and again as the compass label that selects which arm of the final cross is highlighted.
    """
    out, _ = _simulate(np.asarray(grid, dtype=int))
    return out


def _stamp_input_block(grid: np.ndarray, center: tuple[int, int], direction: str) -> None:
    r, c = center
    grid[r - 1 : r + 2, c - 1 : c + 2] = GREY
    dr, dc = DIRS[direction]
    grid[r + dr, c + dc] = RED


def _stamp_output_block(grid: np.ndarray, center: tuple[int, int], direction: str) -> None:
    r, c = center
    grid[r - 1 : r + 2, c - 1 : c + 2] = GREY
    for (dr, dc), color in ARM_COLOR[direction].items():
        grid[r + dr, c + dc] = color
    grid[r, c] = GREY


def _build_input_grid(h: int, w: int, blocks: list[dict], obstacles: list[tuple[int, int]]) -> np.ndarray:
    grid = np.zeros((h, w), dtype=int)
    for top_left in obstacles:
        for r, c in _obstacle_cells(top_left):
            grid[r, c] = ORANGE
    for block in blocks:
        _stamp_input_block(grid, block["start"], block["direction"])
    return grid


def _build_expected_output(h: int, w: int, blocks: list[dict], obstacles: list[tuple[int, int]]) -> np.ndarray:
    grid = np.zeros((h, w), dtype=int)
    for top_left in obstacles:
        for r, c in _obstacle_cells(top_left):
            grid[r, c] = ORANGE
    for block in blocks:
        _stamp_output_block(grid, block["final"], block["direction"])
    return grid


def _base_layout(h: int, w: int, include_e: bool) -> tuple[list[dict], list[tuple[int, int]]]:
    ra = random.randint(7, min(9, h // 2))
    ca = random.randint(max(5, w // 3), max(5, w // 2 - 2))
    oa = (ra - 1 if random.random() < 0.5 else ra, ca + 2)
    a_back = random.randint(2, min(5, ca - 1))
    block_a = {
        "name": "A",
        "direction": "R",
        "start": (ra, ca - a_back),
        "final": (ra, ca),
        "stop_reason": "obstacle",
    }

    cb = ca + random.choice([-1, 0, 1])
    rb = ra - 3
    b_back = random.randint(2, min(4, rb - 1))
    block_b = {
        "name": "B",
        "direction": "D",
        "start": (rb - b_back, cb),
        "final": (rb, cb),
        "stop_reason": "block",
    }

    rc = random.randint(h - 8, h - 6)
    cc = random.randint(4, min(6, w // 2 - 1))
    oc = (rc - 1 if random.random() < 0.5 else rc, cc - 3)
    c_back = random.randint(2, min(5, w - 2 - cc))
    block_c = {
        "name": "C",
        "direction": "L",
        "start": (rc, cc + c_back),
        "final": (rc, cc),
        "stop_reason": "obstacle",
    }

    cd = w - 7
    d_back = random.randint(4, min(7, h - 3))
    block_d = {
        "name": "D",
        "direction": "U",
        "start": (1 + d_back, cd),
        "final": (1, cd),
        "stop_reason": "boundary",
    }

    blocks = [block_a, block_b, block_c, block_d]
    if include_e:
        e_final = (1, cd + 3)
        e_back = random.randint(2, min(4, w - 2 - e_final[1]))
        blocks.append(
            {
                "name": "E",
                "direction": "L",
                "start": (1, e_final[1] + e_back),
                "final": e_final,
                "stop_reason": "block",
            }
        )

    return blocks, [oa, oc]


def _random_extra_block(
    h: int,
    w: int,
    existing_blocks: list[dict],
    existing_obstacles: list[tuple[int, int]],
) -> dict | None:
    reserved_paths = [_path_cells(block["start"], block["final"], block["direction"]) for block in existing_blocks]
    existing_starts = [block["start"] for block in existing_blocks]
    used_obstacle_cells = set()
    for top_left in existing_obstacles:
        used_obstacle_cells.update(_obstacle_cells(top_left))

    for _ in range(200):
        direction = random.choice(list(DIRS))
        stop_type = random.choice(["boundary", "obstacle"])

        obstacle_top_left = None
        if stop_type == "boundary":
            if direction == "U":
                final = (1, random.randint(2, w - 3))
            elif direction == "D":
                final = (h - 2, random.randint(2, w - 3))
            elif direction == "L":
                final = (random.randint(2, h - 3), 1)
            else:
                final = (random.randint(2, h - 3), w - 2)
        else:
            if direction == "U":
                final = (random.randint(4, h - 5), random.randint(2, w - 3))
                obstacle_top_left = (final[0] - 3, final[1] - 1 + random.choice([0, 1]))
            elif direction == "D":
                final = (random.randint(2, h - 5), random.randint(2, w - 3))
                obstacle_top_left = (final[0] + 2, final[1] - 1 + random.choice([0, 1]))
            elif direction == "L":
                final = (random.randint(2, h - 3), random.randint(4, w - 5))
                obstacle_top_left = (final[0] - 1 + random.choice([0, 1]), final[1] - 3)
            else:
                final = (random.randint(2, h - 3), random.randint(2, w - 5))
                obstacle_top_left = (final[0] - 1 + random.choice([0, 1]), final[1] + 2)

        dr, dc = DIRS[direction]
        max_back = 0
        while True:
            cand = (final[0] - dr * (max_back + 1), final[1] - dc * (max_back + 1))
            if any(not (0 <= r < h and 0 <= c < w) for r, c in _block_cells(cand)):
                break
            max_back += 1
        if max_back < 2:
            continue
        start = (final[0] - dr * random.randint(2, min(5, max_back)), final[1] - dc * random.randint(2, min(5, max_back)))
        if start == final:
            continue

        start_cells = _block_cells(start)
        if any(cell in used_obstacle_cells for cell in start_cells):
            continue
        if obstacle_top_left is not None:
            obstacle_cells = set(_obstacle_cells(obstacle_top_left))
            if any(r < 0 or c < 0 or r >= h or c >= w for r, c in obstacle_cells):
                continue
            if obstacle_cells & used_obstacle_cells:
                continue
        else:
            obstacle_cells = set()

        if any(_min_cell_manhattan(start_cells, _block_cells(center)) < 4 for center in existing_starts):
            continue

        path = _path_cells(start, final, direction)
        if any(path & other for other in reserved_paths):
            continue
        if path & used_obstacle_cells:
            continue
        if obstacle_cells and path & obstacle_cells:
            continue

        return {
            "name": "X",
            "direction": direction,
            "start": start,
            "final": final,
            "stop_reason": stop_type,
            "obstacle": obstacle_top_left,
        }
    return None


def _validate_layout(h: int, w: int, blocks: list[dict], obstacles: list[tuple[int, int]]) -> bool:
    directions = {block["direction"] for block in blocks}
    if directions != {"U", "D", "L", "R"}:
        return False

    start_cells = [_block_cells(block["start"]) for block in blocks]
    for i in range(len(start_cells)):
        for j in range(i + 1, len(start_cells)):
            if _min_cell_manhattan(start_cells[i], start_cells[j]) < 4:
                return False

    obstacle_sets = [set(_obstacle_cells(top_left)) for top_left in obstacles]
    for cells in start_cells:
        s = set(cells)
        if any(s & obs for obs in obstacle_sets):
            return False

    path_sets = [_path_cells(block["start"], block["final"], block["direction"]) for block in blocks]
    for i in range(len(path_sets)):
        for j in range(i + 1, len(path_sets)):
            if path_sets[i] & path_sets[j]:
                return False

    input_grid = _build_input_grid(h, w, blocks, obstacles)
    expected = _build_expected_output(h, w, blocks, obstacles)

    try:
        simulated, info = _simulate(input_grid)
    except Exception:
        return False
    if not np.array_equal(simulated, expected):
        return False

    reasons = {item["stop_reason"] for item in info}
    if not {"boundary", "obstacle", "block"} <= reasons:
        return False

    final_by_direction = {}
    for item in info:
        final_by_direction.setdefault(item["direction"], []).append(item["final_center"])
    if any(len(v) == 0 for v in final_by_direction.values()):
        return False

    return True


def generate_input() -> np.ndarray:
    for _ in range(400):
        h = random.randint(18, 24)
        w = random.randint(18, 24)
        count = random.choice([4, 5, 5, 6])

        include_e = count >= 5
        blocks, obstacles = _base_layout(h, w, include_e=include_e)

        if count == 6:
            extra = _random_extra_block(h, w, blocks, obstacles)
            if extra is None:
                continue
            blocks.append(
                {
                    "name": extra["name"],
                    "direction": extra["direction"],
                    "start": extra["start"],
                    "final": extra["final"],
                    "stop_reason": extra["stop_reason"],
                }
            )
            if extra.get("obstacle") is not None:
                obstacles.append(extra["obstacle"])

        if len(obstacles) > 4:
            continue
        if not (4 <= len(blocks) <= 6 and 2 <= len(obstacles) <= 4):
            continue
        if not _validate_layout(h, w, blocks, obstacles):
            continue
        return _build_input_grid(h, w, blocks, obstacles)

    raise RuntimeError("failed to generate a valid puzzle instance")
