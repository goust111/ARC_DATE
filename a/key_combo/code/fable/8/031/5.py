from __future__ import annotations

import random
import sys
from pathlib import Path

import numpy as np

_ARC_UTILS_DIR = Path("/home/lixinyu/snap/new/solvers")
if str(_ARC_UTILS_DIR) not in sys.path:
    sys.path.append(str(_ARC_UTILS_DIR))

from arc_utils import find_components, get_bg


Direction = tuple[int, int]
Coord = tuple[int, int]

DIRS: list[Direction] = [(-1, 0), (0, 1), (1, 0), (0, -1)]
NOTCH_REL_TO_DIR: dict[Coord, Direction] = {
    (0, 1): (-1, 0),
    (1, 2): (0, 1),
    (2, 1): (1, 0),
    (1, 0): (0, -1),
}
EAST_SQUARE_OFFSETS: tuple[Coord, ...] = (
    (-1, -2),
    (-1, -1),
    (-1, 0),
    (0, -2),
    (0, -1),
    (1, -2),
    (1, -1),
    (1, 0),
)
PALETTE = tuple(range(10))
MAX_GENERATION_ATTEMPTS = 400
_COLLISION_QUEUE: list[int] = []


def _in_bounds(shape: tuple[int, int], r: int, c: int) -> bool:
    return 0 <= int(r) < int(shape[0]) and 0 <= int(c) < int(shape[1])


def _rotate_cw(direction: Direction) -> Direction:
    idx = DIRS.index(direction)
    return DIRS[(idx + 1) % 4]


def _rotate_ccw(direction: Direction) -> Direction:
    idx = DIRS.index(direction)
    return DIRS[(idx - 1) % 4]


def _turn(direction: Direction, collision_count: int, policy: str) -> Direction:
    if policy == "alternate":
        return _rotate_cw(direction) if collision_count % 2 == 1 else _rotate_ccw(direction)
    if policy == "cw":
        return _rotate_cw(direction)
    if policy == "ccw":
        return _rotate_ccw(direction)
    raise ValueError(f"unknown turn policy: {policy}")


def _detect_notched_square(grid: np.ndarray) -> tuple[int, dict]:
    bg = int(get_bg(grid))
    components = find_components(grid, bg=bg, connectivity=4, monochromatic=True, return_format="dict")
    candidates: list[dict] = []
    for comp in components:
        if int(comp["area"]) != 8:
            continue
        r0, c0, r1, c1 = map(int, comp["bbox"])
        if (r1 - r0, c1 - c0) != (2, 2):
            continue
        color = int(comp["color"])
        patch = grid[r0 : r1 + 1, c0 : c1 + 1]
        mask = patch == color
        if int(mask.sum()) != 8:
            continue
        missing = np.argwhere(~mask)
        if missing.shape != (1, 2):
            continue
        mr, mc = map(int, missing[0])
        if (mr, mc) not in NOTCH_REL_TO_DIR:
            continue
        candidates.append(
            {
                "color": color,
                "bbox": (r0, c0, r1, c1),
                "notch": (r0 + mr, c0 + mc),
                "heading": NOTCH_REL_TO_DIR[(mr, mc)],
                "coords": {tuple(map(int, rc)) for rc in comp["coords"]},
            }
        )
    assert len(candidates) == 1, "expected exactly one 3x3 notched square"
    return bg, candidates[0]


def _obstacle_mask(grid: np.ndarray, square: dict, bg: int) -> np.ndarray:
    mask = grid != int(bg)
    for r, c in square["coords"]:
        mask[int(r), int(c)] = False
    for r, c in np.argwhere(mask):
        rr = int(r)
        cc = int(c)
        for dr, dc in DIRS:
            nr, nc = rr + dr, cc + dc
            if _in_bounds(mask.shape, nr, nc):
                assert not bool(mask[nr, nc]), "obstacles must be single pixels"
    return mask


def _trace_beam(grid: np.ndarray, turn_policy: str = "alternate") -> tuple[np.ndarray, int, list[Coord], dict, np.ndarray]:
    g = np.asarray(grid, dtype=int)
    bg, square = _detect_notched_square(g)
    blockers = _obstacle_mask(g, square, bg)
    out = g.copy()
    current = tuple(map(int, square["notch"]))
    direction = tuple(map(int, square["heading"]))
    collisions = 0
    visited: list[Coord] = []
    seen: set[tuple[int, int, int, int, int]] = set()

    while True:
        state = (current[0], current[1], direction[0], direction[1], collisions % 2)
        assert state not in seen, "beam entered a loop"
        seen.add(state)

        nr = current[0] + direction[0]
        nc = current[1] + direction[1]
        if not _in_bounds(g.shape, nr, nc):
            break
        if bool(blockers[nr, nc]):
            collisions += 1
            direction = _turn(direction, collisions, turn_policy)
            step_r = current[0] + direction[0]
            step_c = current[1] + direction[1]
            assert _in_bounds(g.shape, step_r, step_c), "every turn must leave at least one in-bounds step"
            assert not bool(blockers[step_r, step_c]), "turn target cannot be blocked"
            continue

        assert int(g[nr, nc]) == bg, "beam may only traverse background cells"
        current = (nr, nc)
        if current != tuple(map(int, square["notch"])):
            out[nr, nc] = int(square["color"])
        visited.append(current)

    return out, collisions, visited, square, blockers


def main(grid: np.ndarray) -> np.ndarray:
    """
    Extend a beam from the notched side of the unique 3x3 square and alternate right-angle turns at blockers.

    Input structure:
    - A dominant background contains exactly one 3x3 monochrome square with one missing edge-midpoint cell.
    - The missing midpoint is the unique notch and indicates the beam's initial heading away from the square.
    - Any other non-background cells are single-pixel blockers and are not part of the square.

    Rule:
    - Keep the square, its notch, and every blocker unchanged.
    - Start from the notch and color each traversed background cell with the square's color.
    - When the next cell in the current heading is a blocker, do not enter it: turn 90 degrees clockwise on the 1st, 3rd, 5th, ... collisions and 90 degrees counterclockwise on the 2nd, 4th, 6th, ... collisions.
    - Stop once the next step would leave the grid.

    Key insight:
    - The notch gives the initial ray direction, but the hidden state is collision parity: successive blocker hits alternate the turn direction instead of using one fixed bend rule.
    """
    out, _collisions, _visited, _square, _blockers = _trace_beam(grid, turn_policy="alternate")
    return out


def _rotate_from_east(coord: Coord, heading: Direction) -> Coord:
    r, c = map(int, coord)
    if heading == (0, 1):
        return r, c
    if heading == (1, 0):
        return c, -r
    if heading == (0, -1):
        return -r, -c
    if heading == (-1, 0):
        return -c, r
    raise ValueError(f"unsupported heading: {heading}")


def _normalized_scene(collision_count: int, run_lengths: list[int], final_length: int) -> tuple[list[Coord], list[Coord], Coord, Direction]:
    current = (0, 0)
    direction = (0, 1)
    path: list[Coord] = []
    blockers: list[Coord] = []

    for seg_len in run_lengths:
        for _ in range(int(seg_len)):
            current = (current[0] + direction[0], current[1] + direction[1])
            path.append(current)
        blockers.append((current[0] + direction[0], current[1] + direction[1]))
        direction = (1, 0) if direction == (0, 1) else (0, 1)

    for _ in range(int(final_length)):
        current = (current[0] + direction[0], current[1] + direction[1])
        path.append(current)

    assert len(blockers) == int(collision_count)
    return path, blockers, current, direction


def _choose_collision_count() -> int:
    global _COLLISION_QUEUE
    if not _COLLISION_QUEUE:
        _COLLISION_QUEUE = [1, 2, 3, 4]
        random.shuffle(_COLLISION_QUEUE)
    return int(_COLLISION_QUEUE.pop())


def _sample_lengths(collision_count: int) -> tuple[list[int], int]:
    while True:
        run_lengths = [random.randint(1, 4) for _ in range(collision_count)]
        final_length = random.randint(2, 6)
        horizontal_extent = sum(run_lengths[0::2]) + (collision_count + 1) // 2 + (final_length if collision_count % 2 == 0 else 0)
        vertical_extent = sum(run_lengths[1::2]) + collision_count // 2 + (final_length if collision_count % 2 == 1 else 0)
        if horizontal_extent <= 20 and vertical_extent <= 20:
            if collision_count == 1 or len(set(run_lengths + [final_length])) > 1:
                return run_lengths, final_length


def _embed_scene(
    heading: Direction,
    path_rel: list[Coord],
    blockers_rel: list[Coord],
    end_rel: Coord,
    final_dir_rel: Direction,
) -> tuple[dict[str, list[Coord] | Coord], tuple[int, int]]:
    square_rel = [_rotate_from_east(coord, heading) for coord in EAST_SQUARE_OFFSETS]
    path = [_rotate_from_east(coord, heading) for coord in path_rel]
    blockers = [_rotate_from_east(coord, heading) for coord in blockers_rel]
    gap = (0, 0)
    end = _rotate_from_east(end_rel, heading)
    final_dir = _rotate_from_east(final_dir_rel, heading)

    all_points = square_rel + path + blockers + [gap]
    min_r = min(r for r, _ in all_points)
    max_r = max(r for r, _ in all_points)
    min_c = min(c for _, c in all_points)
    max_c = max(c for _, c in all_points)

    side_pad = random.randint(1, 4)
    ortho_pad = random.randint(1, 4)
    extra_pad = random.randint(1, 4)

    if final_dir == (0, 1):
        tr = -min_r + ortho_pad
        tc = -min_c + side_pad
        H = max_r + tr + 1 + extra_pad
        W = max_c + tc + 1
        assert end[1] == max_c
    elif final_dir == (0, -1):
        tr = -min_r + ortho_pad
        tc = -min_c
        H = max_r + tr + 1 + extra_pad
        W = max_c + tc + 1 + side_pad
        assert end[1] == min_c
    elif final_dir == (1, 0):
        tr = -min_r + ortho_pad
        tc = -min_c + side_pad
        H = max_r + tr + 1
        W = max_c + tc + 1 + extra_pad
        assert end[0] == max_r
    else:
        tr = -min_r
        tc = -min_c + side_pad
        H = max_r + tr + 1 + extra_pad
        W = max_c + tc + 1 + ortho_pad
        assert end[0] == min_r

    assert 3 <= H <= 30 and 3 <= W <= 30

    def shift(coords: list[Coord]) -> list[Coord]:
        return [(r + tr, c + tc) for r, c in coords]

    scene = {
        "square": shift(square_rel),
        "path": shift(path),
        "blockers": shift(blockers),
        "gap": (gap[0] + tr, gap[1] + tc),
        "end": (end[0] + tr, end[1] + tc),
        "final_dir": final_dir,
    }
    return scene, (H, W)


def _render_input(scene: dict[str, list[Coord] | Coord], shape: tuple[int, int], bg: int, square_color: int, blocker_color: int) -> np.ndarray:
    grid = np.full(shape, int(bg), dtype=int)
    for r, c in scene["square"]:
        grid[r, c] = int(square_color)
    for r, c in scene["blockers"]:
        assert grid[r, c] == int(bg)
        grid[r, c] = int(blocker_color)
    return grid


def _validate_sample(grid: np.ndarray, expected_collisions: int) -> None:
    out, collisions, visited, square, blockers = _trace_beam(grid, turn_policy="alternate")
    assert collisions == int(expected_collisions)
    assert int(blockers.sum()) == int(expected_collisions)
    assert len(visited) >= int(expected_collisions) + 1
    assert int(grid[square["notch"]]) == int(get_bg(grid))
    assert int(out[square["notch"]]) == int(get_bg(grid))
    assert np.any(out != grid)

    ccw_out, _, _, _, _ = _trace_beam(grid, turn_policy="ccw")
    assert not np.array_equal(out, ccw_out), "sample must reject a fixed counterclockwise turn rule"
    if expected_collisions >= 2:
        cw_out, _, _, _, _ = _trace_beam(grid, turn_policy="cw")
        assert not np.array_equal(out, cw_out), "sample must reject a fixed clockwise turn rule"


def _gen_attempt() -> np.ndarray | None:
    collision_count = _choose_collision_count()
    run_lengths, final_length = _sample_lengths(collision_count)
    path_rel, blockers_rel, end_rel, final_dir_rel = _normalized_scene(collision_count, run_lengths, final_length)
    heading = random.choice(DIRS)
    scene, shape = _embed_scene(heading, path_rel, blockers_rel, end_rel, final_dir_rel)

    bg, square_color, blocker_color = random.sample(PALETTE, 3)
    grid = _render_input(scene, shape, bg, square_color, blocker_color)

    if int(get_bg(grid)) != int(bg):
        return None

    try:
        _validate_sample(grid, collision_count)
    except AssertionError:
        return None
    return grid


def generate_input() -> np.ndarray:
    for _ in range(MAX_GENERATION_ATTEMPTS):
        grid = _gen_attempt()
        if grid is not None:
            return grid
    raise RuntimeError("failed to generate a valid refracting-beacon input")
