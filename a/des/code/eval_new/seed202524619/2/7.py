import random
from collections import Counter
from typing import Dict, List, Optional, Tuple

import numpy as np

NEIGHBORS: Tuple[Tuple[int, int], ...] = ((-1, 0), (1, 0), (0, -1), (0, 1))


def _bg_color(grid: np.ndarray) -> int:
    vals, counts = np.unique(grid, return_counts=True)
    return int(vals[int(np.argmax(counts))])


def _scan_left_prompts(grid: np.ndarray, bg: int) -> list[dict[str, object]]:
    H, W = grid.shape
    prompts: list[dict[str, object]] = []
    for r in range(1, H):
        if int(grid[r, 0]) == bg:
            continue
        c = 0
        while c < W and int(grid[r, c]) != bg:
            c += 1
        if 3 <= c <= 5:
            colors = [int(grid[r, cc]) for cc in range(c)]
            if len(set(colors)) == c:
                prompts.append({"index": int(r), "colors": colors})
    return prompts


def _scan_top_prompts(grid: np.ndarray, bg: int) -> list[dict[str, object]]:
    H, W = grid.shape
    prompts: list[dict[str, object]] = []
    for c in range(1, W):
        if int(grid[0, c]) == bg:
            continue
        r = 0
        while r < H and int(grid[r, c]) != bg:
            r += 1
        if 3 <= r <= 5:
            colors = [int(grid[rr, c]) for rr in range(r)]
            if len(set(colors)) == r:
                prompts.append({"index": int(c), "colors": colors})
    return prompts


def _extract_locks(grid: np.ndarray, bg: int) -> list[dict[str, object]]:
    H, W = grid.shape
    mask = (grid != bg)
    mask[0, :] = False
    mask[:, 0] = False

    visited = np.zeros((H, W), dtype=bool)
    components: list[dict[str, object]] = []

    for r0 in range(H):
        for c0 in range(W):
            if not mask[r0, c0] or visited[r0, c0]:
                continue

            stack = [(int(r0), int(c0))]
            visited[r0, c0] = True
            pixels: list[tuple[int, int]] = []
            colors: Dict[int, int] = {}
            min_r, min_c = int(r0), int(c0)
            max_r, max_c = int(r0), int(c0)

            while stack:
                r, c = stack.pop()
                pixels.append((r, c))
                color = int(grid[r, c])
                colors[color] = colors.get(color, 0) + 1
                if r < min_r:
                    min_r = r
                if r > max_r:
                    max_r = r
                if c < min_c:
                    min_c = c
                if c > max_c:
                    max_c = c
                for dr, dc in NEIGHBORS:
                    rr, cc = r + dr, c + dc
                    if rr < 0 or rr >= H or cc < 0 or cc >= W:
                        continue
                    if not mask[rr, cc] or visited[rr, cc]:
                        continue
                    visited[rr, cc] = True
                    stack.append((rr, cc))

            if 3 <= len(colors) <= 5:
                components.append(
                    {
                        "pixels": pixels,
                        "colors": colors,
                        "bbox": (int(min_r), int(min_c), int(max_r), int(max_c)),
                    }
                )
    return components


def _select_prompt(
    prompts: list[dict[str, object]], low: int, high: int
) -> Optional[dict[str, object]]:
    if not prompts:
        return None
    inside = [p for p in prompts if low <= int(p["index"]) <= high]
    if len(inside) == 1:
        return inside[0]
    mid = (low + high) / 2
    if inside:
        return min(inside, key=lambda p: abs(int(p["index"]) - mid))
    return min(prompts, key=lambda p: abs(int(p["index"]) - mid))


def main(grid: np.ndarray) -> np.ndarray:
    """
    Recolor each lock object by using row-order importance and column-order replacement colors.

    Input structure:
    - Background is black. The grid contains several interior lock sprites, each made of 3 to 5 colors and connected orthogonally.
    - The left column stores short row hint bars (length 3 to 5, all distinct colors) aligned with each lock.
    - The top row stores short column hint bars (length 3 to 5, all distinct colors) aligned with each lock's column.

    Rule:
    - For each lock, find its matching row hint bar and matching column hint bar.
    - Count pixels per color inside the lock, then sort colors by descending area.
    - Resolve equal-area ties by the row hint bar order from left to right (earlier means more important).
    - Recolor all pixels of the rank k color to the kth color in the matched column hint bar.
    - Keep the lock shape coordinates and other pixels unchanged.

    Key insight:
    The row hint provides tie-breaking importance only, while the column hint defines the destination palette by area rank.
    """

    g = np.asarray(grid, dtype=int)
    out = g.copy()
    H, W = g.shape
    bg = _bg_color(g)

    row_hints = _scan_left_prompts(g, bg)
    col_hints = _scan_top_prompts(g, bg)
    if not row_hints or not col_hints:
        return out

    for comp in _extract_locks(g, bg):
        (r0, c0, r1, c1) = comp["bbox"]  # type: ignore[assignment]
        row_prompt = _select_prompt(row_hints, int(r0), int(r1))
        col_prompt = _select_prompt(col_hints, int(c0), int(c1))
        if row_prompt is None or col_prompt is None:
            continue

        row_order = {color: i for i, color in enumerate(row_prompt["colors"]) }
        col_colors = list(col_prompt["colors"])  # type: ignore[assignment]

        color_items = list(comp["colors"].items())  # type: ignore[assignment]
        fallback_order = {color: i for i, (color, _cnt) in enumerate(color_items)}
        rank = {color: idx for color, idx in fallback_order.items()}

        color_items.sort(
            key=lambda kv: (
                -int(kv[1]),
                row_order.get(int(kv[0]), len(col_colors) + 2),
                rank[int(kv[0])],
            )
        )

        mapping: Dict[int, int] = {}
        for k, (color, _cnt) in enumerate(color_items):
            if k < len(col_colors):
                mapping[int(color)] = int(col_colors[k])
            else:
                mapping[int(color)] = int(color)

        for rr, cc in comp["pixels"]:  # type: ignore[assignment]
            old = int(g[int(rr), int(cc)])
            out[int(rr), int(cc)] = mapping[old]

    return out


def _sample_color_areas(color_count: int) -> list[int]:
    areas = random.sample(range(2, 12), color_count)
    if random.random() < 0.5:
        i, j = random.sample(range(color_count), 2)
        shared = random.randint(2, 8)
        areas[i] = shared
        areas[j] = shared
    random.shuffle(areas)
    return [int(a) for a in areas]


def _grow_connected(total: int, h: int, w: int) -> Optional[list[tuple[int, int]]]:
    if total <= 0 or total > h * w:
        return None
    for _ in range(80):
        cells: list[tuple[int, int]] = []
        occupied = set[tuple[int, int]]()

        sr = random.randrange(h)
        sc = random.randrange(w)
        cells.append((sr, sc))
        occupied.add((sr, sc))

        while len(cells) < total:
            frontier: list[tuple[int, int, int, int]] = []
            for r, c in cells:
                for dr, dc in NEIGHBORS:
                    nr, nc = r + dr, c + dc
                    if 0 <= nr < h and 0 <= nc < w and (nr, nc) not in occupied:
                        frontier.append((r, c, nr, nc))
            if not frontier:
                break
            _, _, nr, nc = random.choice(frontier)
            cells.append((nr, nc))
            occupied.add((nr, nc))

        if len(cells) == total:
            return cells
    return None


def _color_cells(
    cells: list[tuple[int, int]], colors: list[int], areas: list[int]
) -> Optional[tuple[np.ndarray, list[tuple[int, int]]]]:
    assert len(colors) == len(areas)
    if sum(areas) != len(cells):
        return None

    cell_set = set(cells)
    remaining = set(cells)
    assigned: Dict[tuple[int, int], int] = {}
    all_assigned: set[tuple[int, int]] = set()

    for color, target in zip(colors, areas):
        if target <= 0:
            return None

        attach: list[tuple[int, int]] = []
        if not all_assigned:
            attach = [random.choice(list(remaining))]
        else:
            for r, c in remaining:
                for dr, dc in NEIGHBORS:
                    nr, nc = r + dr, c + dc
                    if (nr, nc) in all_assigned:
                        attach.append((r, c))
                        break
        if not attach:
            return None

        seed = random.choice(attach)
        region = [seed]
        assigned[seed] = int(color)
        remaining.remove(seed)
        all_assigned.add(seed)

        while len(region) < target:
            options: list[tuple[int, int, int, int]] = []
            for rr, cc in region:
                for dr, dc in NEIGHBORS:
                    nr, nc = rr + dr, cc + dc
                    if (nr, nc) in remaining:
                        options.append((rr, cc, nr, nc))

            if not options:
                return None
            _, _, nr, nc = random.choice(options)
            region.append((nr, nc))
            assigned[(nr, nc)] = int(color)
            remaining.remove((nr, nc))
            all_assigned.add((nr, nc))

    if remaining:
        # fill remainder to exact area contract in a deterministic fallback.
        # In practice this should not happen; return None to retry generation.
        return None

    max_r = max(r for r, _ in assigned)
    max_c = max(c for _, c in assigned)
    local = np.full((max_r + 1, max_c + 1), -1, dtype=int)
    for (r, c), color in assigned.items():
        local[r, c] = int(color)

    return local, list(assigned.keys())


def _place_lock(
    grid: np.ndarray,
    occupied: np.ndarray,
    row_hint: int,
    col_hint: int,
    colors: list[int],
    areas: list[int],
) -> bool:
    H, W = grid.shape
    total = int(sum(areas))
    if total <= 0:
        return False

    # Try many random local shapes and placements so we hit at least one valid layout.
    for _ in range(120):
        h_min = max(4, int(np.ceil(np.sqrt(total))))
        h_max = min(H - 2, 12)
        w_max = min(W - 2, 12)
        if h_min > h_max:
            return False
        h = random.randint(h_min, h_max)
        w = random.randint(h_min, w_max)
        if h * w < total:
            continue

        cells = _grow_connected(total, h, w)
        if cells is None:
            continue

        color_map = _color_cells(cells, colors, areas)
        if color_map is None:
            continue
        local, local_cells = color_map
        if local.size == 0:
            continue

        max_r = local.shape[0] - 1
        max_c = local.shape[1] - 1
        rows = sorted({r for r, _ in local_cells})
        cols = sorted({c for _, c in local_cells})

        candidates: list[tuple[int, int]] = []
        for ar in rows:
            r0 = row_hint - ar
            if r0 < 1 or r0 + max_r >= H - 1:
                continue
            for ac in cols:
                c0 = col_hint - ac
                if c0 < 1 or c0 + max_c >= W - 1:
                    continue

                valid = True
                for rr, cc in local_cells:
                    gr, gc = r0 + rr, c0 + cc
                    if occupied[gr, gc]:
                        valid = False
                        break
                if valid:
                    candidates.append((r0, c0))

        if not candidates:
            continue
        r0, c0 = random.choice(candidates)

        for rr in range(local.shape[0]):
            for cc in range(local.shape[1]):
                color = int(local[rr, cc])
                if color < 0:
                    continue
                grid[r0 + rr, c0 + cc] = int(color)
                occupied[r0 + rr, c0 + cc] = True
        return True

    return False


def generate_input() -> np.ndarray:
    rng = np.random.default_rng()
    h = int(rng.integers(5, 31))
    w = int(rng.integers(5, 31))
    grid = np.zeros((h, w), dtype=int)
    for color in range(1, int(rng.integers(2, 6))):
        for _ in range(int(rng.integers(1, 4))):
            grid[int(rng.integers(0, h)), int(rng.integers(0, w))] = color
    return grid
