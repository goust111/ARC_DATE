from __future__ import annotations

import random
from importlib.machinery import SourceFileLoader

import numpy as np

try:
    from arc_utils import Color, bbox_coords, components_mask, in_bounds
except ModuleNotFoundError:
    _ARC_UTILS = SourceFileLoader(
        "arc_utils_shared", "/home/lixinyu/snap/new/solvers/arc_utils.py"
    ).load_module()
    Color = _ARC_UTILS.Color
    bbox_coords = _ARC_UTILS.bbox_coords
    components_mask = _ARC_UTILS.components_mask
    in_bounds = _ARC_UTILS.in_bounds


BG = int(Color.BLACK)
RED = int(Color.RED)
GREEN = int(Color.GREEN)
YELLOW = int(Color.YELLOW)
PINK = int(Color.PINK)

DIRS = [(-1, 0), (0, 1), (1, 0), (0, -1)]
VALID_BLOCK = [(0, 0), (0, 1), (1, 0), (1, 1)]
GREEN_DISTRACTORS = [
    [(0, 0)],
    [(0, 0), (0, 1)],
    [(0, 0), (1, 0)],
    [(0, 0), (0, 1), (1, 0)],
    [(0, 0), (0, 1), (1, 1)],
    [(0, 0), (1, 0), (1, 1)],
    [(0, 1), (1, 0), (1, 1)],
    [(0, 0), (0, 1), (0, 2)],
    [(0, 0), (1, 0), (2, 0)],
]
RED_DISTRACTORS = [
    [(0, 0)],
    [(0, 0), (0, 1)],
    [(0, 0), (1, 0)],
]
GEN_MODES = ("clean", "blocker", "boundary")
GEN_ATTEMPTS = 400
_GEN_CALL_COUNT = 0


def _seed_coord(grid: np.ndarray) -> tuple[int, int]:
    seeds = np.argwhere(grid == YELLOW)
    assert seeds.shape[0] == 1, "expected exactly one yellow seed"
    return int(seeds[0, 0]), int(seeds[0, 1])


def _count_full_green_blocks(grid: np.ndarray) -> int:
    count = 0
    for coords in components_mask(grid == GREEN, connectivity=4):
        if coords.shape[0] != 4:
            continue
        r0, c0, r1, c1 = bbox_coords(coords)
        if (r1 - r0 + 1, c1 - c0 + 1) == (2, 2):
            count += 1
    return count


def _trace_spiral(
    shape: tuple[int, int], seed: tuple[int, int], base_step: int, max_legs: int = 12
) -> list[dict]:
    h, w = map(int, shape)
    r, c = map(int, seed)
    min_r = max_r = int(r)
    min_c = max_c = int(c)
    steps: list[dict] = []

    for leg in range(max_legs):
        length = (leg // 2 + 1) * int(base_step)
        dr, dc = DIRS[leg % 4]
        for step_in_leg in range(length):
            r += dr
            c += dc
            min_r = min(min_r, int(r))
            max_r = max(max_r, int(r))
            min_c = min(min_c, int(c))
            max_c = max(max_c, int(c))
            steps.append(
                {
                    "leg": int(leg),
                    "step_in_leg": int(step_in_leg + 1),
                    "pos": (int(r), int(c)),
                    "inside": bool(in_bounds((h, w), r, c)),
                }
            )
        if min_r < 0 and max_r >= h and min_c < 0 and max_c >= w:
            break
    return steps


def _simulate_spiral(
    grid: np.ndarray,
    base_step: int | None = None,
    seed: tuple[int, int] | None = None,
) -> tuple[np.ndarray, dict]:
    g = np.asarray(grid, dtype=int)
    h, w = g.shape
    n = _count_full_green_blocks(g) if base_step is None else int(base_step)
    assert n > 0, "expected at least one valid 2x2 green block"
    sr, sc = _seed_coord(g) if seed is None else map(int, seed)

    out = g.copy()
    r, c = int(sr), int(sc)
    min_r = max_r = int(r)
    min_c = max_c = int(c)
    had_oob = False
    painted_count = 0
    skipped_non_bg: list[tuple[int, int, int]] = []

    for leg in range(h * w + 20):
        length = (leg // 2 + 1) * n
        dr, dc = DIRS[leg % 4]
        for _ in range(length):
            r += dr
            c += dc
            min_r = min(min_r, int(r))
            max_r = max(max_r, int(r))
            min_c = min(min_c, int(c))
            max_c = max(max_c, int(c))
            if not in_bounds((h, w), r, c):
                had_oob = True
                continue

            value = int(out[int(r), int(c)])
            if value == PINK:
                return out, {
                    "stop_reason": "obstacle",
                    "stop_pos": (int(r), int(c)),
                    "completed_legs": int(leg),
                    "had_oob_before_stop": bool(had_oob),
                    "painted_count": int(painted_count),
                    "skipped_non_bg": skipped_non_bg,
                }
            if value == BG:
                out[int(r), int(c)] = YELLOW
                painted_count += 1
            else:
                skipped_non_bg.append((int(r), int(c), int(value)))

        if min_r < 0 and max_r >= h and min_c < 0 and max_c >= w:
            break

    return out, {
        "stop_reason": "exhausted",
        "stop_pos": None,
        "completed_legs": int(leg + 1),
        "had_oob_before_stop": bool(had_oob),
        "painted_count": int(painted_count),
        "skipped_non_bg": skipped_non_bg,
    }


def main(grid: np.ndarray) -> np.ndarray:
    """
    Count valid green 2x2 blocks and use that count as the base step of a yellow spiral.

    Input structure:
    - A black background contains green 4-connected objects, red distractors, one yellow seed, and magenta obstacle pixels.
    - Some green objects are solid 2x2 blocks; other green objects are incomplete singleton or small non-square distractors.
    - The red objects are also distractors and do not affect the count.

    Rule:
    - Identify green 4-connected components whose bbox is exactly 2x2 and whose area is exactly 4; let their count be N.
    - Starting from the yellow seed, trace a spiral with directions up, right, down, left and step lengths N, N, 2N, 2N, 3N, 3N, ...
    - Each unit step consumes length even if it leaves the grid; paint only background cells yellow, skip other non-background cells, and stop immediately before painting any magenta obstacle cell.

    Key insight:
    - The spiral geometry is not fixed: it is parameterized by the number of complete green 2x2 blocks, so correct filtering and counting are necessary before drawing.
    """
    out, _info = _simulate_spiral(np.asarray(grid, dtype=int))
    return out


def _chebyshev_from_point(point: tuple[int, int], cells: list[tuple[int, int]]) -> int:
    pr, pc = map(int, point)
    return min(max(abs(pr - r), abs(pc - c)) for r, c in cells)


def _reserve(reserved: np.ndarray, cells: list[tuple[int, int]], pad: int) -> None:
    h, w = reserved.shape
    for r, c in cells:
        for rr in range(max(0, r - pad), min(h, r + pad + 1)):
            for cc in range(max(0, c - pad), min(w, c + pad + 1)):
                reserved[rr, cc] = True


def _can_place(
    grid: np.ndarray,
    reserved: np.ndarray,
    cells: list[tuple[int, int]],
    forbidden: set[tuple[int, int]],
) -> bool:
    h, w = grid.shape
    for r, c in cells:
        if not (0 <= r < h and 0 <= c < w):
            return False
        if int(grid[r, c]) != BG or bool(reserved[r, c]) or (int(r), int(c)) in forbidden:
            return False
    return True


def _paint(grid: np.ndarray, reserved: np.ndarray, cells: list[tuple[int, int]], color: int) -> None:
    for r, c in cells:
        grid[r, c] = int(color)
    _reserve(reserved, cells, pad=1)


def _translate(pattern: list[tuple[int, int]], top: int, left: int) -> list[tuple[int, int]]:
    return [(int(top + dr), int(left + dc)) for dr, dc in pattern]


def _try_place_pattern(
    grid: np.ndarray,
    reserved: np.ndarray,
    forbidden: set[tuple[int, int]],
    pattern: list[tuple[int, int]],
    color: int,
    rng: random.Random,
    seed: tuple[int, int],
    min_seed_gap: int = 0,
    attempts: int = 160,
) -> bool:
    h, w = grid.shape
    pat_h = max(r for r, _ in pattern) + 1
    pat_w = max(c for _, c in pattern) + 1
    if pat_h > h or pat_w > w:
        return False

    for _ in range(attempts):
        top = rng.randint(0, h - pat_h)
        left = rng.randint(0, w - pat_w)
        cells = _translate(pattern, top, left)
        if min_seed_gap and _chebyshev_from_point(seed, cells) < min_seed_gap:
            continue
        if not _can_place(grid, reserved, cells, forbidden):
            continue
        _paint(grid, reserved, cells, color)
        return True
    return False


def _choose_seed_and_path(
    shape: tuple[int, int], n: int, mode: str, rng: random.Random
) -> tuple[tuple[int, int], int, tuple[int, int] | None]:
    h, w = map(int, shape)

    for _ in range(200):
        if mode == "boundary":
            if rng.random() < 0.5:
                row_lo = max(int(n), h - int(n))
                row = rng.randint(row_lo, h - 1)
                col_lo = int(n + 1)
                col_hi = int(w - n - 2)
                if col_lo > col_hi:
                    continue
                col = rng.randint(col_lo, col_hi)
            else:
                col_lo = max(int(n), w - int(n))
                col = rng.randint(col_lo, w - 1)
                row_lo = int(n + 1)
                row_hi = int(h - n - 2)
                if row_lo > row_hi:
                    continue
                row = rng.randint(row_lo, row_hi)
        else:
            row_lo = int(n + 1)
            row_hi = int(h - n - 2)
            col_lo = int(n + 1)
            col_hi = int(w - n - 2)
            if row_lo > row_hi or col_lo > col_hi:
                continue
            row = rng.randint(row_lo, row_hi)
            col = rng.randint(col_lo, col_hi)

        seed = (int(row), int(col))
        trace = _trace_spiral(shape, seed, n, max_legs=12)
        obstacle_options: list[tuple[int, tuple[int, int] | None]] = []

        for step_idx, step in enumerate(trace):
            if not step["inside"] or int(step["leg"]) < 2:
                continue
            pos = tuple(step["pos"])
            if _chebyshev_from_point(seed, [pos]) < 3:
                continue

            prefix = trace[:step_idx]
            had_oob = any(not prev["inside"] for prev in prefix)
            if mode == "boundary":
                if not had_oob or int(step["leg"]) < 4:
                    continue
            else:
                if had_oob or int(step["leg"]) > 5:
                    continue

            blocker_pos = None
            if mode == "blocker":
                blocker_candidates = [
                    tuple(prev["pos"])
                    for prev in prefix
                    if prev["inside"]
                    and _chebyshev_from_point(seed, [tuple(prev["pos"])]) >= 2
                    and _chebyshev_from_point(pos, [tuple(prev["pos"])]) >= 2
                ]
                if not blocker_candidates:
                    continue
                blocker_pos = rng.choice(blocker_candidates)

            obstacle_options.append((int(step_idx), blocker_pos))

        if obstacle_options:
            return seed, *rng.choice(obstacle_options)

    raise RuntimeError(f"failed to find a usable {mode} seed/path layout")


def _green_component_stats(grid: np.ndarray) -> tuple[int, int]:
    valid = 0
    distract = 0
    for coords in components_mask(grid == GREEN, connectivity=4):
        r0, c0, r1, c1 = bbox_coords(coords)
        area = int(coords.shape[0])
        if area == 4 and (r1 - r0 + 1, c1 - c0 + 1) == (2, 2):
            valid += 1
        else:
            distract += 1
    return int(valid), int(distract)


def _validate_generated(
    grid: np.ndarray,
    mode: str,
    expected_n: int,
    obstacle_pos: tuple[int, int],
    blocker_pos: tuple[int, int] | None,
) -> None:
    seed = _seed_coord(grid)
    valid_green, green_distractors = _green_component_stats(grid)
    assert valid_green == int(expected_n)
    assert green_distractors >= 1

    red_components = components_mask(grid == RED, connectivity=4)
    assert red_components
    assert np.count_nonzero(grid == PINK) >= 1

    for coords in components_mask(grid == GREEN, connectivity=4):
        cells = [(int(r), int(c)) for r, c in coords]
        r0, c0, r1, c1 = bbox_coords(coords)
        if int(coords.shape[0]) == 4 and (r1 - r0 + 1, c1 - c0 + 1) == (2, 2):
            assert _chebyshev_from_point(seed, cells) >= 3
    for r, c in np.argwhere(grid == PINK):
        assert _chebyshev_from_point(seed, [(int(r), int(c))]) >= 3

    out, info = _simulate_spiral(grid, base_step=expected_n, seed=seed)
    assert info["stop_reason"] == "obstacle"
    assert info["stop_pos"] == tuple(obstacle_pos)
    assert int(info["completed_legs"]) >= 2
    assert int(info["painted_count"]) >= 3
    assert not np.array_equal(out, grid)

    skipped_positions = {(r, c) for r, c, _v in info["skipped_non_bg"]}
    if mode == "boundary":
        assert bool(info["had_oob_before_stop"])
        assert not skipped_positions
    elif mode == "blocker":
        assert not bool(info["had_oob_before_stop"])
        assert blocker_pos is not None and tuple(blocker_pos) in skipped_positions
    else:
        assert not bool(info["had_oob_before_stop"])
        assert not skipped_positions


def _gen_attempt(mode: str, rng: random.Random) -> np.ndarray | None:
    h = rng.randint(17, 22)
    w = rng.randint(17, 22)
    n = rng.randint(1, 4)

    try:
        seed, obstacle_step_idx, blocker_pos = _choose_seed_and_path((h, w), n, mode, rng)
    except RuntimeError:
        return None

    trace = _trace_spiral((h, w), seed, n, max_legs=12)
    obstacle_pos = tuple(trace[obstacle_step_idx]["pos"])
    prefix_path = {
        tuple(step["pos"])
        for step in trace[:obstacle_step_idx]
        if bool(step["inside"])
    }
    if blocker_pos is not None:
        prefix_path.discard(tuple(blocker_pos))

    grid = np.zeros((h, w), dtype=int)
    reserved = np.zeros((h, w), dtype=bool)

    grid[seed] = YELLOW
    _reserve(reserved, [seed], pad=1)
    grid[obstacle_pos] = PINK
    _reserve(reserved, [obstacle_pos], pad=1)

    if blocker_pos is not None:
        grid[blocker_pos] = RED
        _reserve(reserved, [blocker_pos], pad=1)

    for _ in range(n):
        if not _try_place_pattern(
            grid,
            reserved,
            prefix_path,
            VALID_BLOCK,
            GREEN,
            rng,
            seed,
            min_seed_gap=3,
        ):
            return None

    first_green = rng.choice(GREEN_DISTRACTORS[3:7])
    if not _try_place_pattern(
        grid,
        reserved,
        prefix_path,
        first_green,
        GREEN,
        rng,
        seed,
    ):
        return None

    for _ in range(rng.randint(0, 2)):
        pattern = rng.choice(GREEN_DISTRACTORS)
        if not _try_place_pattern(grid, reserved, prefix_path, pattern, GREEN, rng, seed):
            return None

    red_to_place = rng.randint(1, 3) if mode == "blocker" else rng.randint(2, 4)
    for _ in range(red_to_place):
        pattern = rng.choice(RED_DISTRACTORS)
        if not _try_place_pattern(grid, reserved, prefix_path, pattern, RED, rng, seed):
            return None

    for _ in range(rng.randint(0, 2)):
        if not _try_place_pattern(
            grid,
            reserved,
            prefix_path,
            [(0, 0)],
            PINK,
            rng,
            seed,
            min_seed_gap=3,
        ):
            return None

    try:
        _validate_generated(grid, mode, n, obstacle_pos, blocker_pos)
    except AssertionError:
        return None
    return grid


def generate_input() -> np.ndarray:
    global _GEN_CALL_COUNT

    rng = random.Random()
    mode = GEN_MODES[_GEN_CALL_COUNT % len(GEN_MODES)]
    _GEN_CALL_COUNT += 1

    for _ in range(GEN_ATTEMPTS):
        grid = _gen_attempt(mode, rng)
        if grid is not None:
            return grid
    raise RuntimeError(f"failed to generate valid input for mode={mode}")
