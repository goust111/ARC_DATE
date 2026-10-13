import random
import numpy as np


DIRECTIONS = [
    (-1, 0),
    (-1, 1),
    (0, 1),
    (1, 1),
    (1, 0),
    (1, -1),
    (0, -1),
    (-1, -1),
]

GEN_MAX_RETRIES = 400
GEN_BRANCHES = ["direct", "wrap", "same"]


def _get_bg(grid: np.ndarray) -> int:
    values, counts = np.unique(grid, return_counts=True)
    return int(values[np.argmax(counts)])


def _ring_sequence(grid: np.ndarray, bg: int) -> list[int]:
    cols = np.where(grid[-1] != bg)[0]
    assert len(cols) >= 3, "color ring must contain at least three colors"
    assert int(cols[-1] - cols[0] + 1) == int(len(cols)), "color ring must be contiguous"
    ring = [int(grid[-1, c]) for c in cols]
    assert len(set(ring)) == len(ring), "color ring colors must be distinct"
    return ring


def _find_clocks(grid: np.ndarray, bg: int, ring_set: set[int]) -> list[dict[str, int]]:
    h, w = grid.shape
    clocks: list[dict[str, int]] = []

    for r in range(1, h - 1):
        for c in range(1, w - 1):
            target = int(grid[r, c])
            if target == bg or target not in ring_set:
                continue

            border_vals = [
                int(grid[r + dr, c + dc])
                for dr in (-1, 0, 1)
                for dc in (-1, 0, 1)
                if not (dr == 0 and dc == 0)
            ]
            dial = border_vals[0]
            if dial == bg or target == dial or any(v != dial for v in border_vals):
                continue

            pointers: list[tuple[int, int, int, int]] = []
            for dir_idx, (dr, dc) in enumerate(DIRECTIONS):
                pr = r + 2 * dr
                pc = c + 2 * dc
                if 0 <= pr < h and 0 <= pc < w:
                    color = int(grid[pr, pc])
                    if color != bg:
                        pointers.append((dir_idx, pr, pc, color))
            if len(pointers) != 1:
                continue

            dir_idx, pr, pc, pointer = pointers[0]
            if pointer == dial or pointer not in ring_set:
                continue

            clocks.append(
                {
                    "r": int(r),
                    "c": int(c),
                    "dial": int(dial),
                    "target": int(target),
                    "pointer": int(pointer),
                    "dir_idx": int(dir_idx),
                    "pr": int(pr),
                    "pc": int(pc),
                }
            )

    assert clocks, "expected at least one clock"
    return clocks


def main(grid: np.ndarray) -> np.ndarray:
    """Rotate each clock hand by the step count encoded by the bottom color ring, then recolor it to the target.

    Input structure:
    - The background is the most common color.
    - The grid contains several clock-like objects: each has a 3x3 dial whose eight outer cells share one dial color, a distinct target color at the dial center, and a single pointer pixel two cells away from the center in one of the eight compass directions.
    - The bottom row contains one contiguous sequence of distinct non-background colors that defines a cyclic color ring.

    Rule:
    - For each clock, read the pointer color and the center target color as positions on the bottom color ring.
    - Let K be the number of rightward ring steps needed to go from the pointer color to the target color, wrapping around and treating an equal-color match as one full lap.
    - Rotate the pointer clockwise by K forty-five-degree steps, move it to that new direction, recolor it to the target color, restore the dial center to the dial color, and clear the old pointer cell to background.

    Key insight:
    - The bottom row is an ordered cyclic legend: the visible ring distance between two colors directly encodes how many clockwise 45-degree turns each pointer must make.
    """
    g = np.asarray(grid, dtype=int)
    bg = _get_bg(g)
    ring = _ring_sequence(g, bg)
    ring_index = {color: idx for idx, color in enumerate(ring)}
    clocks = _find_clocks(g, bg, set(ring))

    out = g.copy()
    for clock in clocks:
        out[clock["r"], clock["c"]] = int(clock["dial"])
        out[clock["pr"], clock["pc"]] = int(bg)

    ring_len = len(ring)
    for clock in clocks:
        p1 = ring_index[int(clock["pointer"])]
        p2 = ring_index[int(clock["target"])]
        steps = (p2 - p1) % ring_len
        if steps == 0:
            steps = ring_len
        new_dir = (int(clock["dir_idx"]) + steps) % len(DIRECTIONS)
        dr, dc = DIRECTIONS[new_dir]
        nr = int(clock["r"] + 2 * dr)
        nc = int(clock["c"] + 2 * dc)
        out[nr, nc] = int(clock["target"])

    return out


def _choose_branch_pair(ring_len: int, branch: str) -> tuple[int, int]:
    if branch == "same":
        idx = random.randrange(ring_len)
        return idx, idx
    if branch == "direct":
        start = random.randrange(ring_len - 1)
        end = random.randrange(start + 1, ring_len)
        return start, end
    if branch == "wrap":
        start = random.randrange(1, ring_len)
        end = random.randrange(start)
        return start, end
    raise AssertionError(branch)


def _place_clock(
    grid: np.ndarray,
    center: tuple[int, int],
    dial: int,
    target: int,
    pointer: int,
    dir_idx: int,
) -> None:
    r, c = center
    grid[r - 1 : r + 2, c - 1 : c + 2] = int(dial)
    grid[r, c] = int(target)
    dr, dc = DIRECTIONS[dir_idx]
    grid[r + 2 * dr, c + 2 * dc] = int(pointer)


def _semantic_output(
    inp: np.ndarray,
    specs: list[dict[str, int]],
    bg: int,
    dial: int,
    ring: list[int],
) -> np.ndarray:
    out = inp.copy()
    ring_len = len(ring)

    for spec in specs:
        r = int(spec["r"])
        c = int(spec["c"])
        dr, dc = DIRECTIONS[int(spec["dir_idx"])]
        out[r, c] = int(dial)
        out[r + 2 * dr, c + 2 * dc] = int(bg)

    for spec in specs:
        p1 = int(spec["pointer_idx"])
        p2 = int(spec["target_idx"])
        steps = (p2 - p1) % ring_len
        if steps == 0:
            steps = ring_len
        new_dir = (int(spec["dir_idx"]) + steps) % len(DIRECTIONS)
        dr, dc = DIRECTIONS[new_dir]
        out[int(spec["r"] + 2 * dr), int(spec["c"] + 2 * dc)] = int(ring[p2])

    return out


def _clock_capacity(height: int, width: int) -> int:
    row_slots = len(range(2, height - 3, 5))
    col_slots = len(range(2, width - 2, 5))
    return row_slots * col_slots


def _choose_centers(height: int, width: int, count: int) -> list[tuple[int, int]] | None:
    candidates = [
        (r, c)
        for r in range(2, height - 3)
        for c in range(2, width - 2)
    ]
    random.shuffle(candidates)
    occupied = np.zeros((height, width), dtype=bool)
    centers: list[tuple[int, int]] = []

    for r, c in candidates:
        if occupied[r - 2 : r + 3, c - 2 : c + 3].any():
            continue
        occupied[r - 2 : r + 3, c - 2 : c + 3] = True
        centers.append((r, c))
        if len(centers) == count:
            return centers
    return None


def _generate_attempt() -> np.ndarray | None:
    bg = random.randrange(10)
    ring_len = random.randint(4, 8)
    min_width = max(10, ring_len + 1)
    width = random.randint(min_width, 30)
    height = random.randint(9, 24)

    max_clocks = min(5, _clock_capacity(height, width))
    if max_clocks <= 0:
        return None
    clock_count = random.randint(1, max_clocks)
    centers = _choose_centers(height, width, clock_count)
    if centers is None:
        return None

    ring_palette = [c for c in range(10) if c != bg]
    random.shuffle(ring_palette)
    ring = ring_palette[:ring_len]
    dial_choices = [c for c in range(10) if c not in {bg, *ring}]
    if not dial_choices:
        return None
    dial = random.choice(dial_choices)

    grid = np.full((height, width), bg, dtype=int)
    ring_start = random.randint(0, width - ring_len)
    grid[-1, ring_start : ring_start + ring_len] = np.array(ring, dtype=int)

    specs: list[dict[str, int]] = []
    for center in centers:
        branch = random.choice(GEN_BRANCHES)
        pointer_idx, target_idx = _choose_branch_pair(ring_len, branch)
        dir_idx = random.randrange(len(DIRECTIONS))
        _place_clock(
            grid,
            center,
            dial,
            ring[target_idx],
            ring[pointer_idx],
            dir_idx,
        )
        specs.append(
            {
                "r": int(center[0]),
                "c": int(center[1]),
                "dir_idx": int(dir_idx),
                "pointer_idx": int(pointer_idx),
                "target_idx": int(target_idx),
            }
        )

    expected = _semantic_output(grid, specs, bg, dial, ring)
    try:
        actual = main(grid)
    except Exception:
        return None

    if not np.array_equal(actual, expected):
        return None
    return grid


def generate_input() -> np.ndarray:
    for _ in range(GEN_MAX_RETRIES):
        grid = _generate_attempt()
        if grid is not None:
            return grid
    raise RuntimeError("failed to generate a valid input")
