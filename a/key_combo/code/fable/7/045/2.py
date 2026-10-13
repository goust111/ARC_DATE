from __future__ import annotations

import random

import numpy as np


BLACK = 0
GREY = 5
PALETTE = (1, 2, 3, 4, 6, 7, 8, 9)
GEN_RETRIES = 300


def _find_divider_and_commands(grid: np.ndarray) -> tuple[int, list[tuple[int, int]]]:
    h, w = grid.shape
    candidates: list[int] = []
    for r in range(1, h - 1):
        if not np.all(grid[r] == GREY):
            continue
        below = grid[r + 1 :]
        if below.size == 0:
            continue
        if all(np.all(below_row == below_row[0]) for below_row in below):
            candidates.append(r)
    assert len(candidates) == 1, "expected one full-width grey divider row"
    divider = candidates[0]

    commands: list[tuple[int, int]] = []
    r = h - 1
    while r > divider:
        color = int(grid[r, 0])
        assert color != GREY, "instruction stripes cannot be grey"
        run_top = r
        while run_top - 1 > divider and int(grid[run_top - 1, 0]) == color:
            run_top -= 1
        stripe = grid[run_top : r + 1]
        assert np.all(stripe == color), "instruction rows must be full-width stripes"
        commands.append((color, r - run_top + 1))
        r = run_top - 1

    stripe_colors = [color for color, _height in commands]
    assert len(stripe_colors) == len(set(stripe_colors)), "instruction colors must be distinct"
    return divider, commands


def _simulate_scene(
    scene: np.ndarray, commands: list[tuple[int, int]]
) -> tuple[np.ndarray, list[tuple[int, int, int]]]:
    work = np.array(scene, dtype=int, copy=True)
    scene_h, _scene_w = work.shape
    drops: list[tuple[int, int, int]] = []

    for color, distance in commands:
        mask = work == int(color)
        assert mask.any(), f"missing scene object for command color {color}"
        coords = np.argwhere(mask)
        work[mask] = BLACK

        drop = 0
        for step in range(1, int(distance) + 1):
            shifted = coords.copy()
            shifted[:, 0] += step
            if int(shifted[:, 0].max()) >= scene_h:
                break
            if np.any(work[shifted[:, 0], shifted[:, 1]] != BLACK):
                break
            drop = step

        work[coords[:, 0] + drop, coords[:, 1]] = int(color)
        drops.append((int(color), int(distance), int(drop)))

    return work, drops


def main(grid: np.ndarray) -> np.ndarray:
    """Treat the lower stripe stack as bottom-to-top drop commands for same-colored scene objects.

    Input structure:
    - A full-width grey divider row splits the grid into an upper scene and a lower instruction belt.
    - Above the divider, a black background contains several solid monochrome objects.
    - Each instruction-stripe color appears exactly once as a same-colored scene object, and extra scene colors may appear as static obstacles.
    - Below the divider, distinct full-width color stripes are stacked, each with a height that acts as a movement distance.

    Rule:
    - Read the stripes from bottom to top.
    - For each stripe, move the same-colored scene object straight downward by the stripe height.
    - If the next step would collide with another object or with the divider, stop early on the last free row.
    - Keep the divider row and the whole instruction belt unchanged.

    Key insight:
    - The stripe stack is a program rather than scenery: stripe height is the drop amount, and bottom-to-top execution order changes later collisions.
    """
    g = np.asarray(grid, dtype=int)
    assert g.ndim == 2

    divider, commands = _find_divider_and_commands(g)
    updated_scene, _drops = _simulate_scene(g[:divider], commands)

    out = g.copy()
    out[:divider] = updated_scene
    return out


def _rect_clear(
    occupied: np.ndarray,
    r0: int,
    c0: int,
    rect_h: int,
    rect_w: int,
    *,
    pad: int,
) -> bool:
    h, w = occupied.shape
    if r0 < 0 or c0 < 0 or r0 + rect_h > h or c0 + rect_w > w:
        return False
    rr0 = max(0, r0 - pad)
    cc0 = max(0, c0 - pad)
    rr1 = min(h, r0 + rect_h + pad)
    cc1 = min(w, c0 + rect_w + pad)
    return not occupied[rr0:rr1, cc0:cc1].any()


def _paint_rect(scene: np.ndarray, occupied: np.ndarray, color: int, r0: int, c0: int, rect_h: int, rect_w: int) -> None:
    scene[r0 : r0 + rect_h, c0 : c0 + rect_w] = int(color)
    occupied[r0 : r0 + rect_h, c0 : c0 + rect_w] = True


def _place_random_rect(
    scene: np.ndarray,
    occupied: np.ndarray,
    color: int,
    *,
    row_range: tuple[int, int] | None = None,
    col_range: tuple[int, int] | None = None,
    height_choices: tuple[int, ...] = (1, 2, 3),
    width_choices: tuple[int, ...] = (2, 3, 4),
    pad: int = 1,
    tries: int = 80,
) -> bool:
    h, w = scene.shape
    row_lo, row_hi = (0, h) if row_range is None else row_range
    col_lo, col_hi = (0, w) if col_range is None else col_range

    for _ in range(tries):
        rect_h = random.choice(height_choices)
        rect_w = random.choice(width_choices)
        if row_hi - row_lo < rect_h or col_hi - col_lo < rect_w:
            continue
        r0 = random.randint(row_lo, row_hi - rect_h)
        c0 = random.randint(col_lo, col_hi - rect_w)
        if not _rect_clear(occupied, r0, c0, rect_h, rect_w, pad=pad):
            continue
        _paint_rect(scene, occupied, color, r0, c0, rect_h, rect_w)
        return True
    return False


def _build_chain_branch(
    scene_h: int,
    scene_w: int,
    commands: list[tuple[int, int]],
    static_colors: list[int],
) -> np.ndarray | None:
    scene = np.full((scene_h, scene_w), BLACK, dtype=int)
    occupied = np.zeros((scene_h, scene_w), dtype=bool)
    moving_colors = [color for color, _height in commands]
    chain_len = random.randint(2, min(3, len(commands)))

    chain_width = random.randint(2, min(4, scene_w - 4))
    chain_heights = [random.randint(1, 3) for _ in range(chain_len)]
    gaps = [random.randint(1, 3) for _ in range(chain_len - 1)]
    bottom_gap = random.randint(2, 4)
    total_height = sum(chain_heights) + sum(gaps) + bottom_gap
    if total_height > scene_h:
        return None

    c0 = random.randint(1, scene_w - chain_width - 2)
    lower_edge = scene_h - bottom_gap
    rows_bottom_to_top: list[tuple[int, int]] = []
    for idx, rect_h in enumerate(chain_heights):
        lower_edge -= rect_h
        rows_bottom_to_top.append((lower_edge, rect_h))
        if idx < len(gaps):
            lower_edge -= gaps[idx]
    if rows_bottom_to_top[-1][0] < 0:
        return None

    for idx, (r0, rect_h) in enumerate(rows_bottom_to_top):
        _paint_rect(scene, occupied, moving_colors[idx], r0, c0, rect_h, chain_width)

    for color in moving_colors[chain_len:]:
        if not _place_random_rect(scene, occupied, color, pad=1):
            return None

    for color in static_colors:
        if not _place_random_rect(scene, occupied, color, pad=1):
            return None

    return scene


def _build_blocked_branch(
    scene_h: int,
    scene_w: int,
    commands: list[tuple[int, int]],
    static_colors: list[int],
) -> np.ndarray | None:
    scene = np.full((scene_h, scene_w), BLACK, dtype=int)
    occupied = np.zeros((scene_h, scene_w), dtype=bool)
    moving_colors = [color for color, _height in commands]

    blocker_color = static_colors[0]
    blocker_w = random.randint(2, min(4, scene_w - 4))
    blocker_h = random.randint(1, 2)
    blocker_c = random.randint(1, scene_w - blocker_w - 2)
    blocker_r = random.randint(max(4, scene_h // 2), scene_h - blocker_h - 2)
    if not _rect_clear(occupied, blocker_r, blocker_c, blocker_h, blocker_w, pad=1):
        return None
    _paint_rect(scene, occupied, blocker_color, blocker_r, blocker_c, blocker_h, blocker_w)

    mover_color, mover_dist = commands[0]
    mover_w = blocker_w
    mover_h = random.randint(1, 3)
    gap = random.randint(1, min(3, max(1, mover_dist)))
    mover_r = blocker_r - gap - mover_h
    if mover_r < 0:
        return None
    if not _rect_clear(occupied, mover_r, blocker_c, mover_h, mover_w, pad=0):
        return None
    _paint_rect(scene, occupied, mover_color, mover_r, blocker_c, mover_h, mover_w)

    if len(moving_colors) > 1:
        low_color = moving_colors[1]
        low_h = random.randint(1, 2)
        low_w = random.randint(2, min(4, scene_w - 4))
        low_r = scene_h - low_h - random.randint(1, 2)
        low_c = random.randint(1, scene_w - low_w - 2)
        if not _rect_clear(occupied, low_r, low_c, low_h, low_w, pad=1):
            return None
        _paint_rect(scene, occupied, low_color, low_r, low_c, low_h, low_w)

    for color in moving_colors[2:]:
        if not _place_random_rect(scene, occupied, color, pad=1):
            return None

    for color in static_colors[1:]:
        if not _place_random_rect(scene, occupied, color, pad=1):
            return None

    return scene


def _assemble_grid(scene: np.ndarray, commands: list[tuple[int, int]]) -> np.ndarray:
    stripe_rows = [
        np.full((height, scene.shape[1]), color, dtype=int)
        for color, height in reversed(commands)
    ]
    divider = np.full((1, scene.shape[1]), GREY, dtype=int)
    return np.vstack([scene, divider, *stripe_rows])


def _attempt_generate() -> np.ndarray | None:
    n_commands = random.randint(2, 4)
    stripe_heights = [random.randint(1, 3) for _ in range(n_commands)]
    belt_h = sum(stripe_heights)
    max_scene_h = min(18, 29 - belt_h)
    if max_scene_h < 9:
        return None
    scene_h = random.randint(9, max_scene_h)
    scene_w = random.randint(11, 18)

    n_static = random.randint(1, min(3, len(PALETTE) - n_commands))
    colors = random.sample(PALETTE, n_commands + n_static)
    command_colors = colors[:n_commands]
    static_colors = colors[n_commands:]
    commands = list(zip(command_colors, stripe_heights))

    use_chain = n_commands >= 2 and random.random() < 0.6
    if use_chain:
        scene = _build_chain_branch(scene_h, scene_w, commands, static_colors)
    else:
        scene = _build_blocked_branch(scene_h, scene_w, commands, static_colors)
    if scene is None:
        return None

    grid = _assemble_grid(scene, commands)
    out = main(grid)
    divider, parsed_commands = _find_divider_and_commands(grid)
    out_scene, drops = _simulate_scene(grid[:divider], parsed_commands)
    assert np.array_equal(out[:divider], out_scene)

    moved_count = sum(drop > 0 for _color, _dist, drop in drops)
    if moved_count < min(2, n_commands):
        return None
    if np.array_equal(grid, out):
        return None

    if use_chain:
        reversed_scene, _ = _simulate_scene(grid[:divider], list(reversed(parsed_commands)))
        if np.array_equal(out_scene, reversed_scene):
            return None

    return grid


def generate_input() -> np.ndarray:
    for _ in range(GEN_RETRIES):
        grid = _attempt_generate()
        if grid is not None:
            return grid
    raise RuntimeError("failed to generate a valid input grid")
