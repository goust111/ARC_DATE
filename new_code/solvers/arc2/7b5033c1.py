from __future__ import annotations
import numpy as np
from arc_utils import get_bg
import random

def main(grid: np.ndarray) -> np.ndarray:
    """
    Straighten a colored snake path into a single-column strip preserving pixel order.

    Input structure:
    - bg = most common color.
    - Non-bg pixels form a single 4-connected path (snake) with exactly two endpoints.
    - Each path pixel may be a different non-bg color.

    Rule:
    - Find the two endpoints (cells with only one path neighbor) of the snake.
    - Begin at the top-left-most endpoint and trace the path cell by cell.
    - Output an (N×1) column listing each cell's color in traversal order.

    Key insight: The path is a simple chain with no branches; choosing the top-left-most endpoint fixes a unique reading direction.
    """
    g = np.array(grid, dtype=int)
    bg = int(get_bg(g))

    mask = g != bg
    pts = [(int(r), int(c)) for r, c in np.argwhere(mask)]
    if not pts:
        return np.zeros((0, 1), dtype=int)

    pset = set(pts)
    deg: dict[tuple[int, int], int] = {}
    for r, c in pts:
        d = 0
        for dr, dc in ((-1, 0), (1, 0), (0, -1), (0, 1)):
            if (r + dr, c + dc) in pset:
                d += 1
        deg[(r, c)] = int(d)

    ends = [p for p in pts if int(deg[p]) == 1]
    assert ends
    start = min(ends)

    order: list[tuple[int, int]] = []
    prev = None
    cur = start
    for _ in range(len(pts)):
        order.append(cur)

        r, c = cur
        nbrs = []
        for dr, dc in ((-1, 0), (1, 0), (0, -1), (0, 1)):
            q = (r + dr, c + dc)
            if q in pset and q != prev:
                nbrs.append(q)
        if not nbrs:
            break

        nxt = min(nbrs)
        prev, cur = cur, nxt

    cols = [int(g[r, c]) for r, c in order]
    return np.array(cols, dtype=int).reshape((-1, 1))

# --------------- generator (Claude#1) ---------------

def _try_extend(path_set: set, last: tuple, dr: int, dc: int, count: int, H: int, W: int) -> list:
    """Try to extend the path by `count` steps in direction (dr,dc).
    Returns the segment of new cells added, may be shorter than count."""
    segment = []
    r, c = last
    for i in range(1, count + 1):
        nr, nc = r + dr * i, c + dc * i
        if nr < 1 or nr >= H - 1 or nc < 1 or nc >= W - 1:
            break
        if (nr, nc) in path_set:
            break
        # The predecessor of this new cell is the previous cell in the segment,
        # or `last` if it's the first cell in the segment.
        pred = (r + dr * (i - 1), c + dc * (i - 1))
        # Check no 4-neighbor in path_set except pred
        bad = False
        for ddr, ddc in ((-1, 0), (1, 0), (0, -1), (0, 1)):
            nb = (nr + ddr, nc + ddc)
            if nb != pred and nb in path_set:
                bad = True
                break
        if bad:
            break
        segment.append((nr, nc))
        path_set.add((nr, nc))
    return segment


def _gen_attempt() -> np.ndarray | None:
    H, W = 16, 16
    bg = random.randint(0, 9)
    non_bg_count = random.randint(3, 5)
    palette = random.sample([c for c in range(10) if c != bg], non_bg_count)

    target_len = random.randint(20, 30)

    start_r = random.randint(1, 4)
    start_c = random.randint(1, W - 2)

    path = [(start_r, start_c)]
    path_set = {(start_r, start_c)}
    turns = 0
    is_vertical = True
    h_dir = random.choice([-1, 1])

    while len(path) < target_len:
        run_len = random.randint(1, 5)
        run_len = min(run_len, target_len - len(path))
        if run_len <= 0:
            break

        last = path[-1]
        segment = []

        if is_vertical:
            v_dir = 1 if random.random() < 0.85 else -1
            segment = _try_extend(path_set, last, v_dir, 0, run_len, H, W)
            if not segment:
                # Undo nothing (nothing was added), try other direction
                segment = _try_extend(path_set, last, -v_dir, 0, run_len, H, W)
        else:
            segment = _try_extend(path_set, last, 0, h_dir, run_len, H, W)
            if not segment:
                segment = _try_extend(path_set, last, 0, -h_dir, run_len, H, W)
                if segment:
                    h_dir = -h_dir
            # Flip h_dir for next horizontal segment
            h_dir = -h_dir

        if not segment:
            break

        path.extend(segment)
        turns += 1
        is_vertical = not is_vertical

    total = len(path)
    if total < 20 or total > 30:
        return None
    if turns < 7 or turns > 11:
        return None

    # Assign colors in contiguous segments, ensuring all palette colors appear
    num_segments = random.randint(max(3, non_bg_count), 6)
    seg_lengths = _random_partition(total, num_segments, min_len=2, max_len=9)
    if seg_lengths is None:
        return None

    # Build color list: first use each palette color once (shuffled), then random
    shuffled_palette = list(palette)
    random.shuffle(shuffled_palette)
    seg_colors = list(shuffled_palette[:num_segments])
    while len(seg_colors) < num_segments:
        choices = [c for c in palette if c != seg_colors[-1]]
        seg_colors.append(random.choice(choices))
    # Ensure no two adjacent segments have same color
    for i in range(1, len(seg_colors)):
        if seg_colors[i] == seg_colors[i - 1]:
            choices = [c for c in palette if c != seg_colors[i - 1]]
            seg_colors[i] = random.choice(choices)

    colors = []
    for seg_len, col in zip(seg_lengths, seg_colors):
        colors.extend([col] * seg_len)

    # Build grid
    grid = np.full((H, W), bg, dtype=int)
    for (r, c), col in zip(path, colors):
        grid[r, c] = col

    # Verify with solver
    expected_out = np.array(colors, dtype=int).reshape(-1, 1)
    actual_out = main(grid)
    if not np.array_equal(actual_out, expected_out):
        return None

    return grid


def _random_partition(total: int, num_parts: int, min_len: int, max_len: int) -> list[int] | None:
    """Partition total into num_parts values each in [min_len, max_len]."""
    if num_parts * min_len > total or num_parts * max_len < total:
        return None
    parts = [min_len] * num_parts
    remaining = total - sum(parts)
    for _ in range(remaining):
        eligible = [i for i in range(num_parts) if parts[i] < max_len]
        if not eligible:
            return None
        parts[random.choice(eligible)] += 1
    random.shuffle(parts)
    return parts


def generate_input() -> np.ndarray:
    for _ in range(300):
        result = _gen_attempt()
        if result is not None:
            return result
    raise RuntimeError("Failed to generate input after 300 attempts")
