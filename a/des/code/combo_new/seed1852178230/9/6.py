from __future__ import annotations

import numpy as np


BLACK = 0
BLUE = 1
RED = 2
GREEN = 3

GRID_LIMIT = 30
CONTAINER_WIDTH = 3
MIN_CONTAINER_HEIGHT = 4
MAX_GENERATION_ATTEMPTS = 400


def _color_components(grid: np.ndarray, color: int) -> list[dict]:
    mask = np.asarray(grid == int(color), dtype=bool)
    h, w = mask.shape
    seen = np.zeros_like(mask, dtype=bool)
    components: list[dict] = []

    for r in range(h):
        for c in range(w):
            if not mask[r, c] or seen[r, c]:
                continue

            stack = [(r, c)]
            seen[r, c] = True
            coords: list[tuple[int, int]] = []
            while stack:
                cr, cc = stack.pop()
                coords.append((int(cr), int(cc)))
                for nr, nc in ((cr - 1, cc), (cr + 1, cc), (cr, cc - 1), (cr, cc + 1)):
                    if 0 <= nr < h and 0 <= nc < w and mask[nr, nc] and not seen[nr, nc]:
                        seen[nr, nc] = True
                        stack.append((int(nr), int(nc)))

            arr = np.array(coords, dtype=int)
            r0, c0 = arr.min(axis=0)
            r1, c1 = arr.max(axis=0)
            components.append(
                {
                    "coords": arr,
                    "bbox": (int(r0), int(c0), int(r1), int(c1)),
                    "area": int(arr.shape[0]),
                }
            )
    return components


def _is_container(grid: np.ndarray, component: dict) -> bool:
    r0, c0, r1, c1 = component["bbox"]
    h = int(r1 - r0 + 1)
    w = int(c1 - c0 + 1)
    if w != CONTAINER_WIDTH or h < MIN_CONTAINER_HEIGHT:
        return False

    patch = np.asarray(grid[r0 : r1 + 1, c0 : c1 + 1], dtype=int)
    expected = np.zeros((h, w), dtype=bool)
    expected[0, :] = True
    expected[-1, :] = True
    expected[:, 0] = True
    expected[:, -1] = True
    if not np.array_equal(patch == BLUE, expected):
        return False

    interior = patch[1:-1, 1]
    return bool(np.all((interior == BLACK) | (interior == RED)))


def _find_containers(grid: np.ndarray) -> list[dict]:
    containers: list[dict] = []
    for component in _color_components(grid, BLUE):
        if _is_container(grid, component):
            r0, c0, r1, c1 = component["bbox"]
            containers.append(
                {
                    "bbox": (int(r0), int(c0), int(r1), int(c1)),
                    "height": int(r1 - r0 + 1),
                    "capacity": int(r1 - r0 - 1),
                }
            )
    containers.sort(key=lambda item: (item["bbox"][1], item["bbox"][0]))
    return containers


def _find_edges(grid: np.ndarray, containers: list[dict]) -> list[tuple[int, int]]:
    edges: set[tuple[int, int]] = set()
    for component in _color_components(grid, GREEN):
        r0, c0, r1, c1 = component["bbox"]
        if r0 != r1:
            continue
        row = int(r0)
        left_matches = [
            idx
            for idx, container in enumerate(containers)
            if int(container["bbox"][3]) + 1 == int(c0)
            and int(container["bbox"][0]) < row < int(container["bbox"][2])
        ]
        right_matches = [
            idx
            for idx, container in enumerate(containers)
            if int(container["bbox"][1]) - 1 == int(c1)
            and int(container["bbox"][0]) < row < int(container["bbox"][2])
        ]
        if len(left_matches) != 1 or len(right_matches) != 1:
            continue
        a, b = int(left_matches[0]), int(right_matches[0])
        if a != b:
            edges.add((min(a, b), max(a, b)))
    return sorted(edges)


def _count_balls(grid: np.ndarray, container: dict) -> int:
    r0, c0, r1, _c1 = container["bbox"]
    col = np.asarray(grid[r0 + 1 : r1, c0 + 1], dtype=int)
    if not np.all((col == BLACK) | (col == RED)):
        raise AssertionError("container interior must contain only black or red")

    count = int(np.sum(col == RED))
    expected = np.full(col.shape[0], BLACK, dtype=int)
    if count:
        expected[-count:] = RED
    if not np.array_equal(col, expected):
        raise AssertionError("balls must form one bottom-aligned stack")
    return count


def _pair_balance(left: int, right: int) -> tuple[int, int]:
    total = int(left) + int(right)
    low = total // 2
    high = total - low
    if left >= right:
        return int(high), int(low)
    return int(low), int(high)


def _paint_counts(grid: np.ndarray, containers: list[dict], counts: list[int]) -> np.ndarray:
    out = np.array(grid, dtype=int, copy=True)
    for container, count in zip(containers, counts):
        r0, c0, r1, _c1 = container["bbox"]
        capacity = int(container["capacity"])
        if not 0 <= int(count) <= capacity:
            raise AssertionError("ball count exceeds container capacity")
        out[r0 + 1 : r1, c0 + 1] = BLACK
        if count:
            out[r1 - count : r1, c0 + 1] = RED
    return out


def main(grid: np.ndarray) -> np.ndarray:
    """
    Balance the red ball stacks between directly pipe-connected containers.

    Input structure:
    - The background is black.
    - Several blue one-pixel-thick rectangular containers appear on the grid;
      every container has the same one-column-wide interior and the same height.
    - Each container interior contains a bottom-aligned vertical stack of red
      pixels with black empty space above it.
    - Some adjacent containers are linked by one-pixel-high horizontal green
      pipe segments that touch the two side walls at the same row.

    Rule:
    - Count the red pixels inside each container.
    - For every pair of containers directly connected by a green pipe, rebalance
      that pair so the two counts become equal when the total is even, or differ
      by exactly one when the total is odd, with the larger side keeping the
      extra red pixel.
    - Sum the pipe-wise transfers for each container, then redraw every stack as
      a contiguous bottom-aligned column of red pixels.
    - Leave the blue frames, green pipes, and overall grid size unchanged.

    Key insight:
    - The green pipes encode which containers share their red-ball totals; the
      output is determined by counting balls per container and equalizing each
      directly connected pair rather than by moving any visible sprite along the
      pipe path.
    """
    g = np.asarray(grid, dtype=int)
    if g.ndim != 2:
        raise AssertionError("expected a 2D grid")

    containers = _find_containers(g)
    if not containers:
        return g.copy()

    counts = [_count_balls(g, container) for container in containers]
    deltas = [0] * len(containers)
    for a, b in _find_edges(g, containers):
        new_a, new_b = _pair_balance(counts[a], counts[b])
        deltas[a] += int(new_a - counts[a])
        deltas[b] += int(new_b - counts[b])

    new_counts = [int(count + delta) for count, delta in zip(counts, deltas)]
    return _paint_counts(g, containers, new_counts)


def _draw_container(grid: np.ndarray, top: int, left: int, height: int, count: int) -> None:
    bottom = top + height - 1
    right = left + CONTAINER_WIDTH - 1
    grid[top : bottom + 1, left] = BLUE
    grid[top : bottom + 1, right] = BLUE
    grid[top, left : right + 1] = BLUE
    grid[bottom, left : right + 1] = BLUE
    if count:
        grid[bottom - count : bottom, left + 1] = RED


def _draw_pipe(grid: np.ndarray, row: int, left_container_left: int, gap: int) -> None:
    start = int(left_container_left + CONTAINER_WIDTH)
    end = int(left_container_left + CONTAINER_WIDTH + gap - 1)
    grid[row, start : end + 1] = GREEN


def _sample_edge_positions(rng: np.random.Generator, container_count: int) -> list[int]:
    target = int(rng.integers(1, container_count // 2 + 1))
    chosen: list[int] = []
    idx = 0
    while idx < container_count - 1:
        remaining_slots = int(container_count - 1 - idx)
        remaining_possible = (remaining_slots + 1) // 2
        need = int(target - len(chosen))
        take = False
        if need == remaining_possible:
            take = True
        elif need > 0 and rng.random() < 0.5:
            take = True
        if take:
            chosen.append(int(idx))
            idx += 2
        else:
            idx += 1
    if not chosen:
        chosen = [int(rng.integers(0, container_count - 1))]
    return chosen


def _sample_pair_counts(
    rng: np.random.Generator,
    capacity: int,
    want_odd: bool,
) -> tuple[int, int]:
    for _ in range(200):
        left = int(rng.integers(0, capacity + 1))
        right = int(rng.integers(0, capacity + 1))
        if left == right:
            continue
        if ((left + right) % 2 == 1) == bool(want_odd):
            return int(left), int(right)
    raise RuntimeError("failed to sample pair counts")


def _sample_counts(
    rng: np.random.Generator,
    container_count: int,
    capacity: int,
    edge_positions: list[int],
) -> list[int]:
    counts = [int(rng.integers(0, capacity + 1)) for _ in range(container_count)]
    edge_count = len(edge_positions)
    odd_count = edge_count // 2
    if edge_count % 2 == 1 and rng.random() < 0.5:
        odd_count += 1
    parity_flags = np.array([True] * odd_count + [False] * (edge_count - odd_count), dtype=bool)
    rng.shuffle(parity_flags)

    for slot, want_odd in zip(edge_positions, parity_flags):
        left, right = _sample_pair_counts(rng, capacity, bool(want_odd))
        counts[int(slot)] = int(left)
        counts[int(slot) + 1] = int(right)
    return counts


def _connected_pair_parities(counts: list[int], edge_positions: list[int]) -> set[int]:
    return {int((counts[pos] + counts[pos + 1]) % 2) for pos in edge_positions if counts[pos] != counts[pos + 1]}


def _build_candidate(rng: np.random.Generator) -> np.ndarray | None:
    for _ in range(300):
        container_count = int(rng.integers(2, 7))
        height = int(rng.integers(4, 9))
        capacity = int(height - 2)
        left_margin = int(rng.integers(1, 5))
        right_margin = int(rng.integers(1, 5))
        top_margin = int(rng.integers(1, 8))
        bottom_margin = int(rng.integers(1, 8))
        gaps = [int(rng.integers(2, 6)) for _ in range(container_count - 1)]

        width = int(left_margin + right_margin + container_count * CONTAINER_WIDTH + sum(gaps))
        grid_height = int(top_margin + height + bottom_margin)
        if width > GRID_LIMIT or grid_height > GRID_LIMIT:
            continue

        edge_positions = _sample_edge_positions(rng, container_count)
        counts = _sample_counts(rng, container_count, capacity, edge_positions)
        if len(edge_positions) > 1 and len(_connected_pair_parities(counts, edge_positions)) < 2:
            continue

        grid = np.full((grid_height, width), BLACK, dtype=int)
        lefts: list[int] = []
        cur_left = int(left_margin)
        for idx in range(container_count):
            lefts.append(int(cur_left))
            _draw_container(grid, top_margin, cur_left, height, counts[idx])
            if idx < container_count - 1:
                cur_left += int(CONTAINER_WIDTH + gaps[idx])

        for pos in edge_positions:
            row = int(top_margin + rng.integers(1, height - 1))
            _draw_pipe(grid, row, lefts[pos], gaps[pos])

        out = main(grid)
        if np.array_equal(out, grid):
            continue
        if not np.any(out == RED):
            continue
        return grid
    return None


def generate_input() -> np.ndarray:
    for _ in range(MAX_GENERATION_ATTEMPTS):
        rng = np.random.default_rng()
        grid = _build_candidate(rng)
        if grid is not None:
            return grid
    raise RuntimeError("failed to generate a valid container-balancing input")
