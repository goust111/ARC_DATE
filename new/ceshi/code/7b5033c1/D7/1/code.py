# Variant of 7b5033c1
# Dimension: D7 — 背景与干扰 (src)
# Change: 纯净背景，无干扰元素，所有非背景像素均属于蛇形路径... → 添加噪声色块

from __future__ import annotations

import random
import sys
from pathlib import Path

import numpy as np

ARC_UTILS_DIR = Path("/home/lixinyu/snap/new/arc2_solver_select")
if str(ARC_UTILS_DIR) not in sys.path:
    sys.path.append(str(ARC_UTILS_DIR))

from arc_utils import get_bg


# Design analysis for the D7 variant
# a) Concrete behavior:
#    The input still contains one long multicolor 4-connected snake. The new part is
#    2-4 extra 4-connected noise blobs of size 2 or 3, spatially separated from the
#    snake and from each other. The output is unchanged: a single column containing
#    only the snake colors in traversal order from the top-left endpoint.
#    Small trace example:
#    - Original: every non-bg pixel belongs to the snake, so `mask = g != bg` is the
#      snake itself.
#    - Variant: `mask = g != bg` contains the snake plus several tiny detached blobs.
#      If the old solver uses all mask pixels, endpoint degrees are polluted and the
#      traversal length becomes wrong. The new solver first splits mask into connected
#      components and keeps the unique largest one as the snake.
# b) Complexity calibration:
#    Noise colors are drawn from the snake palette, so color alone does not separate
#    signal from noise. The snake has 20-30 pixels while every blob has only 2-3
#    pixels, making anomaly removal visible but not ambiguous.
# c) Old implicit assumptions that break:
#    - `pts = argwhere(g != bg)`: no longer means "snake pixels only".
#    - degree counting over all non-bg cells can create extra endpoints or branches.
#    - traversal loop length `len(pts)` would include noise and overrun the chain.
#    The fix is to isolate the main connected component before reusing the old logic.
# d) Placement legality:
#    Generator places blobs fully in-bounds, non-overlapping, and with no 4-neighbor
#    contact to the snake or prior blobs, so each blob forms its own component.
# e) Visual observability:
#    Every generated sample includes at least two detached blobs, so the variant is
#    visible in a few examples and cannot silently collapse back to the clean input.
# New solver assumptions introduced here:
#    - The snake is the unique largest non-background 4-connected component.
#    - That largest component is a simple chain with two endpoints.
#    `generate_input()` enforces both conditions directly.


NEI4 = ((-1, 0), (1, 0), (0, -1), (0, 1))
NOISE_SHAPES = (
    ((0, 0), (0, 1)),
    ((0, 0), (1, 0)),
    ((0, 0), (0, 1), (0, 2)),
    ((0, 0), (1, 0), (2, 0)),
    ((0, 0), (1, 0), (1, 1)),
    ((0, 0), (0, 1), (1, 0)),
    ((0, 0), (0, 1), (1, 1)),
    ((0, 1), (1, 0), (1, 1)),
)


def _connected_components(mask: np.ndarray) -> list[list[tuple[int, int]]]:
    """Return 4-connected components of a boolean mask as coordinate lists."""
    H, W = mask.shape
    seen: set[tuple[int, int]] = set()
    comps: list[list[tuple[int, int]]] = []

    for r, c in np.argwhere(mask):
        start = (int(r), int(c))
        if start in seen:
            continue
        stack = [start]
        seen.add(start)
        comp: list[tuple[int, int]] = []
        while stack:
            cr, cc = stack.pop()
            comp.append((cr, cc))
            for dr, dc in NEI4:
                nr, nc = cr + dr, cc + dc
                nxt = (nr, nc)
                if 0 <= nr < H and 0 <= nc < W and bool(mask[nr, nc]) and nxt not in seen:
                    seen.add(nxt)
                    stack.append(nxt)
        comps.append(comp)
    return comps


def _trace_chain(g: np.ndarray, comp: list[tuple[int, int]]) -> list[int]:
    """Trace a simple chain component from its top-left endpoint and return colors."""
    pset = set(comp)
    deg: dict[tuple[int, int], int] = {}
    for r, c in comp:
        d = 0
        for dr, dc in NEI4:
            if (r + dr, c + dc) in pset:
                d += 1
        deg[(r, c)] = d

    ends = [p for p in comp if deg[p] == 1]
    if len(comp) == 1:
        ends = [comp[0]]
    assert len(ends) == 2 or len(comp) == 1
    start = min(ends)

    order: list[tuple[int, int]] = []
    prev: tuple[int, int] | None = None
    cur = start
    for _ in range(len(comp)):
        order.append(cur)
        r, c = cur
        nbrs = []
        for dr, dc in NEI4:
            q = (r + dr, c + dc)
            if q in pset and q != prev:
                nbrs.append(q)
        if not nbrs:
            break
        assert len(nbrs) == 1
        prev, cur = cur, nbrs[0]

    assert len(order) == len(comp)
    return [int(g[r, c]) for r, c in order]


def main(grid: np.ndarray) -> np.ndarray:
    """
    Straighten the main colored snake into a single column, ignoring detached noise.

    Variant-specific rule:
    - Non-bg pixels may include several tiny disconnected noise blobs.
    - The true snake is the unique largest 4-connected non-bg component.
    - After selecting that component, the traversal/order rule is the original one.
    """
    g = np.array(grid, dtype=int)
    bg = int(get_bg(g))

    comps = _connected_components(g != bg)
    if not comps:
        return np.zeros((0, 1), dtype=int)

    snake = max(comps, key=len)
    colors = _trace_chain(g, snake)
    return np.array(colors, dtype=int).reshape((-1, 1))


def _try_extend(
    path_set: set[tuple[int, int]],
    last: tuple[int, int],
    dr: int,
    dc: int,
    count: int,
    H: int,
    W: int,
) -> list[tuple[int, int]]:
    """Try to extend the path by `count` steps in direction (dr, dc)."""
    segment: list[tuple[int, int]] = []
    r, c = last
    for i in range(1, count + 1):
        nr, nc = r + dr * i, c + dc * i
        if nr < 1 or nr >= H - 1 or nc < 1 or nc >= W - 1:
            break
        if (nr, nc) in path_set:
            break
        pred = (r + dr * (i - 1), c + dc * (i - 1))
        bad = False
        for ddr, ddc in NEI4:
            nb = (nr + ddr, nc + ddc)
            if nb != pred and nb in path_set:
                bad = True
                break
        if bad:
            break
        segment.append((nr, nc))
        path_set.add((nr, nc))
    return segment


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


def _can_place_blob(
    cells: list[tuple[int, int]],
    occupied: set[tuple[int, int]],
    H: int,
    W: int,
) -> bool:
    """Blob must stay in bounds, not overlap, and not touch prior components by edges."""
    cell_set = set(cells)
    for r, c in cells:
        if not (0 <= r < H and 0 <= c < W):
            return False
        if (r, c) in occupied:
            return False
    for r, c in cells:
        for dr, dc in NEI4:
            nb = (r + dr, c + dc)
            if nb in occupied and nb not in cell_set:
                return False
    return True


def _place_noise_blobs(
    grid: np.ndarray,
    occupied: set[tuple[int, int]],
    palette: list[int],
) -> bool:
    """Place 2-4 detached 2/3-pixel blobs whose colors overlap the snake palette."""
    H, W = grid.shape
    for _ in range(random.randint(2, 4)):
        placed = False
        for _ in range(120):
            shape = random.choice(NOISE_SHAPES)
            max_r = max(dr for dr, _ in shape)
            max_c = max(dc for _, dc in shape)
            base_r = random.randint(0, H - max_r - 1)
            base_c = random.randint(0, W - max_c - 1)
            cells = [(base_r + dr, base_c + dc) for dr, dc in shape]
            if not _can_place_blob(cells, occupied, H, W):
                continue
            color = random.choice(palette)
            for cell in cells:
                grid[cell] = color
                occupied.add(cell)
            placed = True
            break
        if not placed:
            return False
    return True


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
        if is_vertical:
            v_dir = 1 if random.random() < 0.85 else -1
            segment = _try_extend(path_set, last, v_dir, 0, run_len, H, W)
            if not segment:
                segment = _try_extend(path_set, last, -v_dir, 0, run_len, H, W)
        else:
            segment = _try_extend(path_set, last, 0, h_dir, run_len, H, W)
            if not segment:
                segment = _try_extend(path_set, last, 0, -h_dir, run_len, H, W)
                if segment:
                    h_dir = -h_dir
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

    num_segments = random.randint(max(3, non_bg_count), 6)
    seg_lengths = _random_partition(total, num_segments, min_len=2, max_len=9)
    if seg_lengths is None:
        return None

    shuffled_palette = list(palette)
    random.shuffle(shuffled_palette)
    seg_colors = list(shuffled_palette[:num_segments])
    while len(seg_colors) < num_segments:
        choices = [c for c in palette if c != seg_colors[-1]]
        seg_colors.append(random.choice(choices))
    for i in range(1, len(seg_colors)):
        if seg_colors[i] == seg_colors[i - 1]:
            choices = [c for c in palette if c != seg_colors[i - 1]]
            seg_colors[i] = random.choice(choices)

    colors: list[int] = []
    for seg_len, col in zip(seg_lengths, seg_colors):
        colors.extend([col] * seg_len)

    grid = np.full((H, W), bg, dtype=int)
    for (r, c), col in zip(path, colors):
        grid[r, c] = col

    occupied = set(path)
    if not _place_noise_blobs(grid, occupied, palette):
        return None

    comps = _connected_components(grid != bg)
    sizes = sorted((len(comp) for comp in comps), reverse=True)
    if len(sizes) < 3:
        return None
    if sizes[0] != total or sizes[1] > 3:
        return None

    expected_out = np.array(colors, dtype=int).reshape(-1, 1)
    actual_out = main(grid)
    if not np.array_equal(actual_out, expected_out):
        return None

    return grid


def generate_input() -> np.ndarray:
    for _ in range(500):
        result = _gen_attempt()
        if result is not None:
            return result
    raise RuntimeError("Failed to generate input after 500 attempts")
