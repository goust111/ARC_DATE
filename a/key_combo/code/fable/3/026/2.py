from __future__ import annotations

from collections import deque
import random

import numpy as np


BG = 0
DRIVER_COLOR = 2
FRAME_COLOR = 5
FRAME_COLORS = {DRIVER_COLOR, FRAME_COLOR}
MOTIF_COLORS = (1, 3, 4, 6, 7, 8, 9)
DIRS4 = ((-1, 0), (1, 0), (0, -1), (0, 1))


def _valid_frame_color(g: np.ndarray, r0: int, c0: int, side: int) -> int | None:
    r1 = r0 + side - 1
    c1 = c0 + side - 1
    color = int(g[r0, c0])
    if color not in FRAME_COLORS:
        return None
    if not (
        np.all(g[r0, c0 : c1 + 1] == color)
        and np.all(g[r1, c0 : c1 + 1] == color)
        and np.all(g[r0 : r1 + 1, c0] == color)
        and np.all(g[r0 : r1 + 1, c1] == color)
    ):
        return None
    return color


def _overlap_bbox(a: tuple[int, int, int, int], b: tuple[int, int, int, int]) -> bool:
    ar0, ac0, ar1, ac1 = a
    br0, bc0, br1, bc1 = b
    return not (ar1 < br0 or br1 < ar0 or ac1 < bc0 or bc1 < ac0)


def _pairwise_nonoverlapping(frames: list[dict]) -> bool:
    for i in range(len(frames)):
        for j in range(i + 1, len(frames)):
            if _overlap_bbox(frames[i]["bbox"], frames[j]["bbox"]):
                return False
    return True


def _foreground_is_inside_frames(g: np.ndarray, frames: list[dict]) -> bool:
    covered = np.zeros(g.shape, dtype=bool)
    for frame in frames:
        r0, c0, r1, c1 = frame["bbox"]
        covered[r0 : r1 + 1, c0 : c1 + 1] = True
    return bool(np.all(covered[g != BG]))


def _touch(a: tuple[int, int, int, int], b: tuple[int, int, int, int]) -> bool:
    ar0, ac0, ar1, ac1 = a
    br0, bc0, br1, bc1 = b
    row_overlap = min(ar1, br1) >= max(ar0, br0)
    col_overlap = min(ac1, bc1) >= max(ac0, bc0)
    if (ac1 + 1 == bc0 or bc1 + 1 == ac0) and row_overlap:
        return True
    if (ar1 + 1 == br0 or br1 + 1 == ar0) and col_overlap:
        return True
    return False


def _contact_graph(frames: list[dict]) -> list[set[int]]:
    graph = [set() for _ in frames]
    for i in range(len(frames)):
        for j in range(i + 1, len(frames)):
            if _touch(frames[i]["bbox"], frames[j]["bbox"]):
                graph[i].add(j)
                graph[j].add(i)
    return graph


def _bfs_distances(graph: list[set[int]], start: int) -> list[int | None]:
    dist: list[int | None] = [None] * len(graph)
    dist[start] = 0
    q: deque[int] = deque([start])
    while q:
        i = q.popleft()
        assert dist[i] is not None
        for j in sorted(graph[i]):
            if dist[j] is None:
                dist[j] = int(dist[i]) + 1
                q.append(j)
    return dist


def _find_frames(grid: np.ndarray) -> tuple[list[dict], list[int]]:
    g = np.asarray(grid, dtype=int)
    H, W = g.shape
    valid_sets = []

    for side in range(4, min(H, W) + 1):
        frames: list[dict] = []
        for r0 in range(H - side + 1):
            for c0 in range(W - side + 1):
                color = _valid_frame_color(g, r0, c0, side)
                if color is None:
                    continue
                frames.append(
                    {
                        "bbox": (r0, c0, r0 + side - 1, c0 + side - 1),
                        "color": color,
                        "side": side,
                    }
                )

        if not frames:
            continue
        if sum(1 for frame in frames if frame["color"] == DRIVER_COLOR) != 1:
            continue
        if not _pairwise_nonoverlapping(frames):
            continue
        if not _foreground_is_inside_frames(g, frames):
            continue

        graph = _contact_graph(frames)
        edge_count = sum(len(nbrs) for nbrs in graph) // 2
        if edge_count != len(frames) - 1:
            continue

        start = next(i for i, frame in enumerate(frames) if frame["color"] == DRIVER_COLOR)
        distances = _bfs_distances(graph, start)
        if any(d is None for d in distances):
            continue
        valid_sets.append((len(frames), side, frames, [int(d) for d in distances]))

    assert valid_sets, "expected a connected tree of same-size red/gray square frames"
    valid_sets.sort(key=lambda item: (item[0], item[1]), reverse=True)
    _, _side, frames, distances = valid_sets[0]
    return frames, distances


def main(grid: np.ndarray) -> np.ndarray:
    """Rotate inner motifs of touching square frames with gear-like alternating directions.

    Input structure:
    - The grid has a black background and several equal-size hollow square frames.
    - The frames touch edge-to-edge, forming one contact tree; exactly one frame border is red and all other frame borders are grey.
    - Each frame interior contains a small asymmetric motif, possibly with its own colors and black gaps.

    Rule:
    - Build the contact graph of the square frames and measure each frame's distance from the red driver frame.
    - Frames at even distance, including the driver, have only their interiors rotated ninety degrees clockwise.
    - Frames at odd distance have only their interiors rotated ninety degrees counterclockwise; frame borders and positions stay fixed.

    Key insight:
    - The touching frames behave like meshed gears, so rotation direction is determined by contact-chain parity from the red driver.
    """
    g = np.asarray(grid, dtype=int)
    assert g.ndim == 2

    frames, distances = _find_frames(g)
    out = g.copy()
    for frame, distance in zip(frames, distances):
        r0, c0, r1, c1 = frame["bbox"]
        interior = g[r0 + 1 : r1, c0 + 1 : c1]
        k = -1 if distance % 2 == 0 else 1
        out[r0 + 1 : r1, c0 + 1 : c1] = np.rot90(interior, k)
    return out


# --------------- generator ---------------

SIDE_CHOICES = (5, 6, 7)
FRAME_COUNT_RANGE = (4, 8)
MAX_GENERATION_ATTEMPTS = 500
TREE_BUILD_ATTEMPTS = 120
MOTIF_ATTEMPTS = 200
MAX_MARGIN = 4


def _lattice_graph(coords: list[tuple[int, int]]) -> list[set[int]]:
    index = {coord: i for i, coord in enumerate(coords)}
    graph = [set() for _ in coords]
    for i, (r, c) in enumerate(coords):
        for dr, dc in DIRS4:
            j = index.get((r + dr, c + dc))
            if j is not None:
                graph[i].add(j)
    return graph


def _adjacent_existing_count(cell: tuple[int, int], occupied: set[tuple[int, int]]) -> int:
    r, c = cell
    return sum((r + dr, c + dc) in occupied for dr, dc in DIRS4)


def _sample_tree_coords(n: int, want_branch: bool) -> list[tuple[int, int]] | None:
    for _ in range(TREE_BUILD_ATTEMPTS):
        coords = [(0, 0)]
        occupied = {(0, 0)}
        while len(coords) < n:
            frontier: list[tuple[int, int]] = []
            for r, c in coords:
                for dr, dc in DIRS4:
                    cell = (r + dr, c + dc)
                    if cell in occupied:
                        continue
                    if _adjacent_existing_count(cell, occupied) == 1:
                        frontier.append(cell)
            if not frontier:
                break
            cell = random.choice(frontier)
            coords.append(cell)
            occupied.add(cell)

        if len(coords) != n:
            continue
        graph = _lattice_graph(coords)
        edge_count = sum(len(nbrs) for nbrs in graph) // 2
        if edge_count != n - 1:
            continue
        if want_branch and max(len(nbrs) for nbrs in graph) < 3:
            continue
        return coords
    return None


def _choose_driver(graph: list[set[int]]) -> tuple[int, list[int]] | None:
    options: list[tuple[int, list[int]]] = []
    for start in range(len(graph)):
        distances = _bfs_distances(graph, start)
        if any(d is None for d in distances):
            continue
        dist = [int(d) for d in distances]
        has_odd = any(d % 2 == 1 for d in dist)
        has_even_non_driver = any(d > 0 and d % 2 == 0 for d in dist)
        if has_odd and has_even_non_driver and max(dist) >= 2:
            options.append((start, dist))
    if not options:
        return None
    return random.choice(options)


def _place_lattice_frames(
    coords: list[tuple[int, int]], side: int
) -> tuple[tuple[int, int], list[tuple[int, int]]] | None:
    rows = [r for r, _c in coords]
    cols = [c for _r, c in coords]
    min_r, max_r = min(rows), max(rows)
    min_c, max_c = min(cols), max(cols)
    span_h = (max_r - min_r + 1) * side
    span_w = (max_c - min_c + 1) * side
    if span_h > 30 or span_w > 30:
        return None

    extra_h = random.randint(0, min(MAX_MARGIN, 30 - span_h))
    extra_w = random.randint(0, min(MAX_MARGIN, 30 - span_w))
    top_margin = random.randint(0, extra_h)
    left_margin = random.randint(0, extra_w)
    shape = (span_h + extra_h, span_w + extra_w)

    positions = [
        (top_margin + (r - min_r) * side, left_margin + (c - min_c) * side)
        for r, c in coords
    ]
    return shape, positions


def _draw_frame(grid: np.ndarray, r0: int, c0: int, side: int, color: int) -> None:
    r1 = r0 + side - 1
    c1 = c0 + side - 1
    grid[r0, c0 : c1 + 1] = color
    grid[r1, c0 : c1 + 1] = color
    grid[r0 : r1 + 1, c0] = color
    grid[r0 : r1 + 1, c1] = color


def _motif_is_nondegenerate(motif: np.ndarray) -> bool:
    cw = np.rot90(motif, -1)
    ccw = np.rot90(motif, 1)
    return (
        np.any(motif != BG)
        and not np.array_equal(motif, cw)
        and not np.array_equal(motif, ccw)
        and not np.array_equal(cw, ccw)
    )


def _random_motif(size: int) -> np.ndarray:
    cells = [(r, c) for r in range(size) for c in range(size)]
    min_count = min(len(cells), max(3, size))
    max_count = min(len(cells) - 1, max(min_count, int(round(size * size * 0.58))))
    for _ in range(MOTIF_ATTEMPTS):
        count = random.randint(min_count, max_count)
        coords = random.sample(cells, count)
        color_count = random.randint(1, min(3, count))
        colors = random.sample(MOTIF_COLORS, color_count)

        motif = np.full((size, size), BG, dtype=int)
        random.shuffle(coords)
        for i, (r, c) in enumerate(coords):
            motif[r, c] = colors[i] if i < color_count else random.choice(colors)

        if _motif_is_nondegenerate(motif):
            return motif
    raise RuntimeError("failed to build a nondegenerate motif")


def _expected_from_frames(grid: np.ndarray, frame_infos: list[dict]) -> np.ndarray:
    out = grid.copy()
    for info in frame_infos:
        r0, c0, r1, c1 = info["bbox"]
        interior = grid[r0 + 1 : r1, c0 + 1 : c1]
        k = -1 if info["distance"] % 2 == 0 else 1
        out[r0 + 1 : r1, c0 + 1 : c1] = np.rot90(interior, k)
    return out


def _rotate_all(grid: np.ndarray, frame_infos: list[dict], k: int) -> np.ndarray:
    out = grid.copy()
    for info in frame_infos:
        r0, c0, r1, c1 = info["bbox"]
        out[r0 + 1 : r1, c0 + 1 : c1] = np.rot90(grid[r0 + 1 : r1, c0 + 1 : c1], k)
    return out


def _passes_semantic_checks(grid: np.ndarray, expected: np.ndarray, frame_infos: list[dict]) -> bool:
    if max(grid.shape) > 30 or max(expected.shape) > 30:
        return False
    if np.array_equal(grid, expected):
        return False

    distances = [info["distance"] for info in frame_infos]
    if not any(d % 2 == 1 for d in distances):
        return False
    if not any(d > 0 and d % 2 == 0 for d in distances):
        return False
    if np.array_equal(expected, _rotate_all(grid, frame_infos, -1)):
        return False
    if np.array_equal(expected, _rotate_all(grid, frame_infos, 1)):
        return False

    try:
        frames, found_distances = _find_frames(grid)
        actual = main(grid)
    except Exception:
        return False
    if len(frames) != len(frame_infos):
        return False
    if sorted(found_distances) != sorted(distances):
        return False
    return bool(np.array_equal(actual, expected))


def _gen_attempt() -> np.ndarray | None:
    n_frames = random.randint(*FRAME_COUNT_RANGE)
    side = random.choice(SIDE_CHOICES)
    want_branch = n_frames >= 5 and random.random() < 0.7
    coords = _sample_tree_coords(n_frames, want_branch)
    if coords is None:
        return None

    graph = _lattice_graph(coords)
    chosen = _choose_driver(graph)
    if chosen is None:
        return None
    driver_idx, distances = chosen

    placed = _place_lattice_frames(coords, side)
    if placed is None:
        return None
    shape, positions = placed
    grid = np.full(shape, BG, dtype=int)

    frame_infos: list[dict] = []
    for i, (r0, c0) in enumerate(positions):
        color = DRIVER_COLOR if i == driver_idx else FRAME_COLOR
        _draw_frame(grid, r0, c0, side, color)
        motif = _random_motif(side - 2)
        grid[r0 + 1 : r0 + side - 1, c0 + 1 : c0 + side - 1] = motif
        frame_infos.append(
            {
                "bbox": (r0, c0, r0 + side - 1, c0 + side - 1),
                "distance": distances[i],
            }
        )

    expected = _expected_from_frames(grid, frame_infos)
    if _passes_semantic_checks(grid, expected, frame_infos):
        return grid
    return None


def generate_input() -> np.ndarray:
    for _ in range(MAX_GENERATION_ATTEMPTS):
        grid = _gen_attempt()
        if grid is not None:
            return grid
    raise RuntimeError("failed to generate a valid gear-transmission input")
