import random
from collections import Counter

import numpy as np


BG = 0
BLUE = 1
YELLOW = 4
TEAL = 8

GENERATE_ATTEMPTS = 400


def _paint_brick(grid: np.ndarray, row: int, c0: int, teal_left: bool) -> None:
    grid[row, c0] = TEAL if teal_left else BLUE
    grid[row, c0 + 1] = BLUE if teal_left else TEAL


def _flip_brick_in_grid(grid: np.ndarray, row: int, c0: int) -> None:
    grid[row, c0], grid[row, c0 + 1] = grid[row, c0 + 1], grid[row, c0]


def _parse_bricks(grid: np.ndarray) -> tuple[list[dict], dict[tuple[int, int], int]]:
    mask = (grid == BLUE) | (grid == TEAL)
    visited = np.zeros_like(mask, dtype=bool)
    bricks: list[dict] = []
    cell_to_brick: dict[tuple[int, int], int] = {}
    h, w = grid.shape

    for r in range(h):
        for c in range(w):
            if not mask[r, c] or visited[r, c]:
                continue
            stack = [(r, c)]
            comp = []
            visited[r, c] = True
            while stack:
                cr, cc = stack.pop()
                comp.append((cr, cc))
                for dr, dc in ((-1, 0), (1, 0), (0, -1), (0, 1)):
                    nr, nc = cr + dr, cc + dc
                    if 0 <= nr < h and 0 <= nc < w and mask[nr, nc] and not visited[nr, nc]:
                        visited[nr, nc] = True
                        stack.append((nr, nc))

            assert len(comp) == 2, "each brick must have exactly two cells"
            rows = {rr for rr, _ in comp}
            cols = sorted(cc for _, cc in comp)
            assert len(rows) == 1 and cols[1] == cols[0] + 1, "each brick must be horizontal 1x2"
            row = next(iter(rows))
            c0 = cols[0]
            vals = {int(grid[row, c0]), int(grid[row, c0 + 1])}
            assert vals == {BLUE, TEAL}, "each brick must contain one blue and one teal cell"
            idx = len(bricks)
            bricks.append({"row": row, "c0": c0, "teal_left": int(grid[row, c0]) == TEAL})
            for cell in comp:
                cell_to_brick[cell] = idx

    return bricks, cell_to_brick


def _simulate(
    grid: np.ndarray,
    *,
    order: str = "ltr",
    flip_after_use: bool = True,
    teal_guides: bool = True,
    strict: bool = True,
) -> tuple[np.ndarray | None, dict | None]:
    g = np.asarray(grid, dtype=int)
    h, w = g.shape
    assert g.ndim == 2 and h <= 30 and w <= 30

    ball_rows, ball_cols = np.where(g == YELLOW)
    assert np.all(ball_rows == 0), "all input balls must start on the top row"

    bricks, cell_to_brick = _parse_bricks(g)
    if order == "ltr":
        launch_cols = [int(c) for c in np.sort(ball_cols)]
    elif order == "rtl":
        launch_cols = [int(c) for c in np.sort(ball_cols)[::-1]]
    else:
        raise ValueError(f"unsupported order: {order}")

    settled = np.zeros((h, w), dtype=bool)
    usage = [0] * len(bricks)
    traces = []

    for start_c in launch_cols:
        r = 0
        c = start_c
        hits = []

        while True:
            while True:
                if r == h - 1:
                    break
                if settled[r + 1, c]:
                    break
                brick_idx = cell_to_brick.get((r + 1, c))
                if brick_idx is not None:
                    break
                r += 1

            if r == h - 1 or settled[r + 1, c]:
                if settled[r, c]:
                    if strict:
                        raise AssertionError("balls may not settle onto an occupied cell")
                    return None, None
                settled[r, c] = True
                traces.append(
                    {
                        "start_col": start_c,
                        "brick_hits": tuple(hits),
                        "final": (r, c),
                    }
                )
                break

            brick = bricks[brick_idx]
            hits.append(brick_idx)
            usage[brick_idx] += 1
            steer_left = brick["teal_left"] if teal_guides else (not brick["teal_left"])
            new_c = brick["c0"] - 1 if steer_left else brick["c0"] + 2
            if not (0 <= new_c < w):
                if strict:
                    raise AssertionError("horizontal slide left the grid")
                return None, None
            if settled[r, new_c]:
                if strict:
                    raise AssertionError("horizontal slide collided with a settled ball")
                return None, None
            c = new_c
            if flip_after_use:
                brick["teal_left"] = not brick["teal_left"]

    out = np.full_like(g, BG)
    for brick in bricks:
        _paint_brick(out, int(brick["row"]), int(brick["c0"]), bool(brick["teal_left"]))
    out[settled] = YELLOW
    return out, {"usage": usage, "traces": traces, "bricks": bricks}


def main(grid: np.ndarray) -> np.ndarray:
    """
    Drop the top-row yellow balls through flipping blue-teal bricks and draw their final stack.

    Input structure:
    - The background is black; yellow balls appear only on the top row.
    - The other foreground objects are isolated horizontal 1x2 bricks, each made of one blue cell and one teal cell.
    - The bottom grid edge acts as the floor, and previously settled balls can become later landing surfaces.

    Rule:
    - Process the yellow balls from left to right, with each ball fully settling before the next one starts.
    - A ball falls vertically; if the cell directly below belongs to a brick, the ball stays on its current row, slides to the column immediately outside the brick on the brick's current teal side, then that brick flips left-right.
    - Repeat this drop, deflect, and flip cycle until the ball lands on the bottom row or on top of an already settled ball in the same column.
    - Clear the input balls from the top row, redraw every brick in its final orientation after all uses, and place the yellow balls at their final settled cells.

    Key insight:
    - The teal half is a mutable steering arrow: each use both redirects the current ball and changes the future behavior for later balls.
    """
    out, _meta = _simulate(grid)
    assert out is not None
    return out


def _build_candidate(rng: random.Random) -> tuple[np.ndarray, tuple[int, int]]:
    w = rng.randint(11, 18)
    x = rng.randint(2, w - 7)

    r_b = rng.randint(2, 4)
    r_a = r_b + rng.randint(2, 4)
    r_lr = r_a + rng.randint(2, 4)
    h = r_lr + rng.randint(4, 8)
    if h > 18:
        raise ValueError("rows exceeded the visual-size cap")

    helper_side = rng.choice(["left", "right"])
    a_teal_left = rng.choice([True, False])

    grid = np.full((h, w), BG, dtype=int)

    if helper_side == "left":
        ball_cols = [x - 1, x + 1, x + 2]
        _paint_brick(grid, r_b, x - 1, False)
    else:
        ball_cols = [x + 1, x + 2, x + 4]
        _paint_brick(grid, r_b, x + 3, True)

    _paint_brick(grid, r_a, x + 1, a_teal_left)
    _paint_brick(grid, r_lr, x, False)
    _paint_brick(grid, r_lr, x + 3, True)

    forbidden = set(range(x - 1, x + 6))
    extra_choices = [c for c in range(w) if c not in forbidden]
    if extra_choices and rng.random() < 0.65:
        ball_cols.append(rng.choice(extra_choices))

    for c in sorted(ball_cols):
        grid[0, c] = YELLOW

    return grid, (r_a, x + 1)


def _validate_candidate(grid: np.ndarray, a_anchor: tuple[int, int]) -> bool:
    out, meta = _simulate(grid, strict=True)
    if out is None or meta is None:
        return False

    traces = meta["traces"]
    usage = meta["usage"]
    if len(traces) < 3:
        return False
    if max(len(trace["brick_hits"]) for trace in traces) < 3:
        return False
    if max(usage) < 2:
        return False

    final_cols = [trace["final"][1] for trace in traces]
    stack_heights = Counter(final_cols)
    if max(stack_heights.values()) < 2:
        return False
    if len(stack_heights) < 2:
        return False

    if not any(u % 2 == 1 for u in usage):
        return False
    if not any(u > 0 and u % 2 == 0 for u in usage):
        return False
    if np.array_equal(grid, out):
        return False

    static_out, _ = _simulate(grid, flip_after_use=False, strict=False)
    if static_out is not None and np.array_equal(out, static_out):
        return False

    blue_guided_out, _ = _simulate(grid, teal_guides=False, strict=False)
    if blue_guided_out is not None and np.array_equal(out, blue_guided_out):
        return False

    alt = grid.copy()
    _flip_brick_in_grid(alt, a_anchor[0], a_anchor[1])
    alt_out, _ = _simulate(alt, strict=True)
    if alt_out is None or np.array_equal(out, alt_out):
        return False

    return True


def generate_input() -> np.ndarray:
    rng = random.Random()
    for _ in range(GENERATE_ATTEMPTS):
        try:
            grid, a_anchor = _build_candidate(rng)
        except ValueError:
            continue
        if _validate_candidate(grid, a_anchor):
            return grid
    raise RuntimeError("failed to generate a valid bicolor-brick pinball input")
