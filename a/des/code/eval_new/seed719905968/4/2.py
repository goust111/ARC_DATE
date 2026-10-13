from __future__ import annotations

import random

import numpy as np


BG = 0
GREY = 5
PALETTE = [1, 2, 3, 4, 6, 7, 8, 9]
GEN_ATTEMPTS = 600


def _extract_signal_objects(grid: np.ndarray) -> list[dict[str, int]]:
    """Parse each colored tower/receiver pair from the input grid."""
    g = np.asarray(grid, dtype=int)
    H, _W = g.shape
    objects: list[dict[str, int]] = []

    colors = sorted({int(v) for v in g.flat if int(v) not in (BG, GREY)})
    for color in colors:
        pts = np.argwhere(g == int(color))
        assert pts.size > 0

        cols = {int(c) for c in pts[:, 1]}
        assert len(cols) == 1, "tower and receiver must share one column"
        col = int(next(iter(cols)))

        rows = sorted(int(r) for r in pts[:, 0])
        tower_rows = [rows[-1]]
        idx = len(rows) - 2
        while idx >= 0 and rows[idx] == tower_rows[-1] - 1:
            tower_rows.append(rows[idx])
            idx -= 1
        tower_rows.reverse()

        receiver_rows = rows[: idx + 1]
        assert len(receiver_rows) == 1, "each color should have one receiver"
        receiver_row = int(receiver_rows[0])

        top = int(tower_rows[0])
        bottom = int(tower_rows[-1])
        height = int(len(tower_rows))

        assert 2 <= height <= 5
        assert receiver_row < top - 1, "receiver must be separated from the tower"
        if bottom < H - 1:
            assert int(g[bottom + 1, col]) == GREY, "floating towers must sit on the grey baseline"

        for r in range(receiver_row, top):
            assert int(g[r, col]) in (BG, GREY, color)

        objects.append(
            {
                "color": int(color),
                "col": int(col),
                "receiver_row": int(receiver_row),
                "top": int(top),
                "bottom": int(bottom),
                "height": int(height),
            }
        )

    objects.sort(key=lambda obj: (int(obj["col"]), int(obj["color"])))
    assert objects, "expected at least one signal tower"
    return objects


def main(grid: np.ndarray) -> np.ndarray:
    """Cast each tower's upward signal for exactly its height and activate receivers that are reached.

    Input structure:
    - Background is black. Each non-grey foreground color appears as one vertical tower and one same-colored receiver pixel above it in the same column.
    - Towers are width-1 vertical columns of height 2-5, resting either on the grid bottom or directly above a grey horizontal baseline.
    - Grey cells may form obstacle walls above towers.

    Rule:
    - For every tower, emit a vertical ray upward from the tower top using the tower's color.
    - The ray can travel at most as many cells as the tower height; it stops early if the next cell is grey.
    - If the ray reaches the same-colored receiver before its range ends, keep the vertical path and replace the receiver with a 3-cell horizontal bar centered on it.
    - If the receiver is farther away than the tower height or hidden behind grey, leave the receiver unchanged.

    Key insight:
    - The tower's visible height is the hidden range parameter for the ray, so solving the task requires measuring tower height before tracing each upward path.
    """
    g = np.asarray(grid, dtype=int)
    H, W = g.shape
    out = g.copy()

    for obj in _extract_signal_objects(g):
        color = int(obj["color"])
        col = int(obj["col"])
        top = int(obj["top"])
        receiver_row = int(obj["receiver_row"])
        height = int(obj["height"])

        hit_receiver = False
        for step in range(1, height + 1):
            row = int(top - step)
            if row < 0:
                break
            if int(g[row, col]) == GREY:
                break
            out[row, col] = int(color)
            if row == receiver_row:
                hit_receiver = True
                break

        if hit_receiver:
            for dc in (-1, 0, 1):
                cc = int(col + dc)
                if 0 <= cc < W:
                    out[receiver_row, cc] = int(color)

    return out


def _sample_spaced_columns(width: int, count: int, min_gap: int = 3) -> list[int] | None:
    cols = list(range(1, width - 1))
    random.shuffle(cols)
    chosen: list[int] = []
    for col in cols:
        if all(abs(int(col) - int(prev)) >= int(min_gap) for prev in chosen):
            chosen.append(int(col))
            if len(chosen) == count:
                return sorted(chosen)
    return None


def _branch_options(tower_bottom: int, branch: str) -> list[tuple[int, int, int | None]]:
    options: list[tuple[int, int, int | None]] = []
    for height in range(2, 6):
        if branch == "blocked" and height < 3:
            continue
        top = int(tower_bottom - height + 1)
        if top < 2:
            continue

        if branch == "hit":
            for dist in range(2, min(height, top) + 1):
                options.append((int(height), int(dist), None))
        elif branch == "short":
            max_dist = min(int(top), int(height + 4))
            for dist in range(height + 1, max_dist + 1):
                options.append((int(height), int(dist), None))
        elif branch == "blocked":
            for dist in range(3, min(height, top) + 1):
                for wall_gap in range(2, dist):
                    options.append((int(height), int(dist), int(wall_gap)))
        else:
            raise ValueError(f"unknown branch {branch}")
    return options


def _choose_branches(count: int) -> list[str]:
    if count == 2:
        branches = ["hit", random.choice(["blocked", "short"])]
    else:
        branches = ["hit", "blocked", "short"]
        while len(branches) < count:
            branches.append(random.choice(["hit", "blocked", "short"]))
    random.shuffle(branches)
    return branches


def _gen_attempt() -> np.ndarray | None:
    tower_count = random.randint(2, 5)
    width = random.randint(max(12, 3 * tower_count), 30)
    height = random.randint(12, 24)
    columns = _sample_spaced_columns(width, tower_count, min_gap=3)
    if columns is None:
        return None

    use_baseline = bool(random.getrandbits(1))
    baseline_row = None
    if use_baseline:
        baseline_row = int(height - 1 - random.randint(0, 1))
        tower_bottom = int(baseline_row - 1)
        if tower_bottom < 4:
            return None
    else:
        tower_bottom = int(height - 1)

    grid = np.full((height, width), BG, dtype=int)
    if baseline_row is not None:
        grid[baseline_row, :] = GREY

    colors = random.sample(PALETTE, tower_count)
    branches = _choose_branches(tower_count)

    placed_hit = 0
    placed_non_hit = 0
    for color, col, branch in zip(colors, columns, branches):
        options = _branch_options(tower_bottom, branch)
        if not options:
            return None
        tower_height, receiver_dist, wall_gap = random.choice(options)
        top = int(tower_bottom - tower_height + 1)
        receiver_row = int(top - receiver_dist)
        if receiver_row < 0:
            return None

        grid[receiver_row, col] = int(color)
        grid[top : tower_bottom + 1, col] = int(color)

        if branch == "blocked":
            wall_row = int(top - wall_gap)
            grid[wall_row, col - 1 : col + 2] = GREY
            placed_non_hit += 1
        elif branch == "short":
            placed_non_hit += 1
        else:
            placed_hit += 1

    if placed_hit == 0 or placed_non_hit == 0:
        return None

    try:
        objects = _extract_signal_objects(grid)
    except AssertionError:
        return None
    if len(objects) != tower_count:
        return None

    out = main(grid)
    if np.array_equal(out, grid):
        return None
    if out.shape != grid.shape or max(out.shape) > 30:
        return None
    return grid


def generate_input() -> np.ndarray:
    for _ in range(GEN_ATTEMPTS):
        grid = _gen_attempt()
        if grid is not None:
            return grid
    raise RuntimeError("failed to generate a valid signal-tower puzzle input")
