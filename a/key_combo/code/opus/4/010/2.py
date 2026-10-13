from __future__ import annotations

from collections import defaultdict, deque
from pathlib import Path
import random
import sys

import numpy as np

_SOLVER_DIR = Path("/home/lixinyu/snap/new/solvers")
if _SOLVER_DIR.exists():
    solver_dir_str = str(_SOLVER_DIR)
    if solver_dir_str not in sys.path:
        sys.path.append(solver_dir_str)

try:
    from arc_utils import Color, neighbors4
except Exception:
    class Color:
        BLACK = 0
        BLUE = 1
        RED = 2
        GREEN = 3
        YELLOW = 4
        GREY = 5
        PINK = 6
        ORANGE = 7
        TEAL = 8
        MAROON = 9

    def neighbors4(r: int, c: int):
        yield r - 1, c
        yield r + 1, c
        yield r, c - 1
        yield r, c + 1


BLACK = int(Color.BLACK)
GREY = int(Color.GREY)
MARKER_COLORS = [1, 2, 3, 4, 6, 7, 8, 9]
DIRS4 = ((-1, 0), (1, 0), (0, -1), (0, 1))

GEN_ATTEMPTS = 400
MIN_MARK_AREA = 2
MIN_COMPONENTS = 6
TARGET_SIZE_MIN = 18
TARGET_SIZE_MAX = 27

_PERIOD_CACHE: dict[tuple[tuple[int, int], bytes], tuple[int, int, np.ndarray]] = {}
_REPAIR_CACHE: dict[tuple[tuple[int, int], bytes], np.ndarray] = {}
_MAIN_CACHE: dict[tuple[tuple[int, int], bytes], np.ndarray] = {}


def _grid_key(grid: np.ndarray) -> tuple[tuple[int, int], bytes]:
    g = np.asarray(grid, dtype=int)
    return tuple(map(int, g.shape)), g.tobytes()


def _consistent_periods(grid: np.ndarray) -> list[tuple[int, int, int, np.ndarray]]:
    g = np.asarray(grid, dtype=int)
    wall_coords = np.argwhere(g == GREY)
    open_coords = np.argwhere((g != BLACK) & (g != GREY))

    candidates: list[tuple[int, int, int, np.ndarray]] = []
    height, width = map(int, g.shape)
    for tile_h in range(1, height + 1):
        for tile_w in range(1, width + 1):
            tile = np.full((tile_h, tile_w), -1, dtype=int)
            valid = True

            for r, c in wall_coords:
                tr, tc = int(r) % tile_h, int(c) % tile_w
                if tile[tr, tc] == 0:
                    valid = False
                    break
                tile[tr, tc] = 1
            if not valid:
                continue

            for r, c in open_coords:
                tr, tc = int(r) % tile_h, int(c) % tile_w
                if tile[tr, tc] == 1:
                    valid = False
                    break
                tile[tr, tc] = 0

            if valid:
                candidates.append((tile_h * tile_w, tile_h, tile_w, tile))
    return candidates


def _infer_wall_tile(grid: np.ndarray) -> tuple[int, int, np.ndarray]:
    key = _grid_key(grid)
    cached = _PERIOD_CACHE.get(key)
    if cached is not None:
        tile_h, tile_w, tile = cached
        return tile_h, tile_w, tile.copy()

    candidates = _consistent_periods(grid)
    assert candidates, "expected at least one consistent wall period"

    min_area = min(area for area, _th, _tw, _tile in candidates)
    best = [(th, tw, tile) for area, th, tw, tile in candidates if area == min_area]
    assert len(best) == 1, "expected a unique minimal wall period"
    tile_h, tile_w, tile = best[0]
    _PERIOD_CACHE[key] = (tile_h, tile_w, tile.copy())
    return tile_h, tile_w, tile.copy()


def _repair_walls(grid: np.ndarray) -> np.ndarray:
    g = np.asarray(grid, dtype=int)
    key = _grid_key(g)
    cached = _REPAIR_CACHE.get(key)
    if cached is not None:
        return cached.copy()

    tile_h, tile_w, tile = _infer_wall_tile(g)
    row_phase = np.arange(g.shape[0])[:, None] % tile_h
    col_phase = np.arange(g.shape[1])[None, :] % tile_w
    should_be_wall = tile[row_phase, col_phase] == 1

    out = g.copy()
    fill_mask = (out == BLACK) & should_be_wall
    out[fill_mask] = GREY
    _REPAIR_CACHE[key] = out.copy()
    return out


def _open_components(wall_mask: np.ndarray) -> tuple[np.ndarray, list[list[tuple[int, int]]]]:
    height, width = map(int, wall_mask.shape)
    labels = np.full((height, width), -1, dtype=int)
    components: list[list[tuple[int, int]]] = []

    for start_r in range(height):
        for start_c in range(width):
            if wall_mask[start_r, start_c] or labels[start_r, start_c] != -1:
                continue

            label = len(components)
            queue = deque([(start_r, start_c)])
            labels[start_r, start_c] = label
            coords: list[tuple[int, int]] = []

            while queue:
                r, c = queue.popleft()
                coords.append((r, c))
                for nr, nc in neighbors4(r, c):
                    if 0 <= nr < height and 0 <= nc < width:
                        if wall_mask[nr, nc] or labels[nr, nc] != -1:
                            continue
                        labels[nr, nc] = label
                        queue.append((nr, nc))

            components.append(coords)
    return labels, components


def _fill_marked_chambers(grid: np.ndarray) -> np.ndarray:
    g = np.asarray(grid, dtype=int)
    out = g.copy()
    wall_mask = g == GREY
    _labels, components = _open_components(wall_mask)

    for coords in components:
        markers = [(r, c) for r, c in coords if int(g[r, c]) not in (BLACK, GREY)]
        if not markers:
            continue
        assert len(markers) == 1, "each repaired chamber must contain at most one marker"
        marker_color = int(g[markers[0]])
        for r, c in coords:
            if out[r, c] == BLACK:
                out[r, c] = marker_color
    return out


def main(grid: np.ndarray) -> np.ndarray:
    """
    Reconstruct the periodic grey wall pattern, then flood-fill each repaired chamber from its unique marker.

    Input structure:
    - The grid has black background/open space, grey wall pixels, and a few singleton colored markers.
    - The grey walls come from a repeated tile, but some wall pixels were replaced by black gaps.
    - Each marker sits inside one chamber of the fully repaired wall layout.

    Rule:
    - Treat grey as observed wall and each colored marker as observed open space while inferring the unique
      smallest repeating wall period.
    - Restore every black cell whose tile phase is known to be a wall.
    - Using the repaired grey walls as 4-connected barriers, find each non-wall chamber.
    - If a chamber contains exactly one marker, repaint its remaining black cells with that marker color;
      leave unmarked chambers black.

    Key insight:
    - The missing grey pixels are not arbitrary holes: repairing the latent periodic wall topology changes
      which open cells belong to the same chamber, and that repaired topology determines the correct fill area.
    """
    g = np.asarray(grid, dtype=int)
    key = _grid_key(g)
    cached = _MAIN_CACHE.get(key)
    if cached is not None:
        return cached.copy()

    repaired = _repair_walls(g)
    out = _fill_marked_chambers(repaired)
    _MAIN_CACHE[key] = out.copy()
    return out


def _minimal_period(tile: np.ndarray) -> tuple[int, int]:
    tile = np.asarray(tile, dtype=int)
    height, width = map(int, tile.shape)
    row_idx = np.arange(height)[:, None]
    col_idx = np.arange(width)[None, :]
    for tile_h in range(1, height + 1):
        for tile_w in range(1, width + 1):
            if np.array_equal(tile, tile[row_idx % tile_h, col_idx % tile_w]):
                return tile_h, tile_w
    return height, width


def _repeat_wall_tile(tile: np.ndarray, reps_r: int, reps_c: int) -> np.ndarray:
    tile = np.asarray(tile, dtype=int)
    tile_h, tile_w = map(int, tile.shape)
    height = reps_r * tile_h + 1
    width = reps_c * tile_w + 1
    row_phase = np.arange(height)[:, None] % tile_h
    col_phase = np.arange(width)[None, :] % tile_w
    return tile[row_phase, col_phase].astype(bool)


def _target_repetitions(period: int) -> list[int]:
    return [
        rep
        for rep in range(3, 8)
        if TARGET_SIZE_MIN <= rep * period + 1 <= TARGET_SIZE_MAX
    ]


def _sample_tile() -> np.ndarray | None:
    tile_h = random.randint(4, 6)
    tile_w = random.randint(4, 6)
    tile = np.zeros((tile_h, tile_w), dtype=int)
    tile[0, :] = 1
    tile[:, 0] = 1

    if tile_h > 3 and random.random() < 0.8:
        row = random.randint(2, tile_h - 1)
        tile[row, 1:] = 1
    if tile_w > 3 and random.random() < 0.8:
        col = random.randint(2, tile_w - 1)
        tile[1:, col] = 1
    if int(tile[1:, 1:].sum()) == 0:
        if random.random() < 0.5:
            tile[random.randint(1, tile_h - 1), random.randint(1, tile_w - 1)] = 1
        else:
            if tile_h > tile_w:
                row = random.randint(1, tile_h - 1)
                tile[row, random.randint(1, tile_w - 1) :] = 1
            else:
                col = random.randint(1, tile_w - 1)
                tile[random.randint(1, tile_h - 1) :, col] = 1

    wall_cells = [tuple(map(int, rc)) for rc in np.argwhere(tile == 1)]
    for _ in range(random.randint(2, 5)):
        start_r, start_c = random.choice(wall_cells)
        dr, dc = random.choice(DIRS4)
        length = random.randint(1, max(tile_h, tile_w) - 1)
        r, c = start_r, start_c
        for _step in range(length):
            nr, nc = r + dr, c + dc
            if not (1 <= nr < tile_h and 1 <= nc < tile_w):
                break
            tile[nr, nc] = 1
            wall_cells.append((nr, nc))
            r, c = nr, nc

    if np.all(tile[1:, 1:] == 1):
        return None
    if int(tile[1:, 1:].sum()) < 2:
        return None
    if _minimal_period(tile) != (tile_h, tile_w):
        return None
    if not _target_repetitions(tile_h) or not _target_repetitions(tile_w):
        return None
    return tile


def _door_candidates(
    wall_mask: np.ndarray, labels: np.ndarray, tile_h: int, tile_w: int
) -> dict[tuple[int, int], list[dict[str, object]]]:
    height, width = map(int, wall_mask.shape)
    edge_to_doors: dict[tuple[int, int], list[dict[str, object]]] = defaultdict(list)

    for r in range(height):
        for c in range(width):
            if not wall_mask[r, c]:
                continue

            touching = sorted(
                {
                    int(labels[nr, nc])
                    for nr, nc in neighbors4(r, c)
                    if 0 <= nr < height and 0 <= nc < width and labels[nr, nc] != -1
                }
            )
            if len(touching) != 2:
                continue

            up = r > 0 and wall_mask[r - 1, c]
            down = r + 1 < height and wall_mask[r + 1, c]
            left = c > 0 and wall_mask[r, c - 1]
            right = c + 1 < width and wall_mask[r, c + 1]
            straight = (up and down and not left and not right) or (
                left and right and not up and not down
            )
            phase = (r % tile_h, c % tile_w)
            interior_phase = phase[0] != 0 and phase[1] != 0
            edge_to_doors[tuple(touching)].append(
                {
                    "cell": (r, c),
                    "phase": phase,
                    "straight": straight,
                    "interior_phase": interior_phase,
                }
            )
    return edge_to_doors


def _choose_door(candidates: list[dict[str, object]]) -> dict[str, object]:
    ranked = sorted(
        candidates,
        key=lambda item: (
            not bool(item["interior_phase"]),
            bool(item["straight"]),
            item["phase"],
            item["cell"],
        ),
    )
    best_rank = (
        bool(ranked[0]["interior_phase"]),
        bool(ranked[0]["straight"]),
    )
    best = [
        item
        for item in ranked
        if (
            bool(item["interior_phase"]),
            bool(item["straight"]),
        )
        == best_rank
    ]
    return random.choice(best)


def _choose_marked_cluster(
    edge_to_doors: dict[tuple[int, int], list[dict[str, object]]],
    component_areas: list[int],
) -> tuple[list[int], list[dict[str, object]]] | None:
    eligible = {idx for idx, area in enumerate(component_areas) if area >= MIN_MARK_AREA}
    graph: dict[int, set[int]] = {idx: set() for idx in eligible}
    valid_edges = [edge for edge in edge_to_doors if edge[0] in eligible and edge[1] in eligible]
    for a, b in valid_edges:
        graph[a].add(b)
        graph[b].add(a)

    triples: list[tuple[int, int, int]] = []
    for center in eligible:
        neighbors = sorted(graph[center])
        for i in range(len(neighbors)):
            for j in range(i + 1, len(neighbors)):
                triples.append((neighbors[i], center, neighbors[j]))

    if triples and random.random() < 0.4:
        a, b, c = random.choice(triples)
        edge_ab = tuple(sorted((a, b)))
        edge_bc = tuple(sorted((b, c)))
        doors = [_choose_door(edge_to_doors[edge_ab]), _choose_door(edge_to_doors[edge_bc])]
        return [a, b, c], doors

    if not valid_edges:
        return None

    edge = random.choice(valid_edges)
    return list(edge), [_choose_door(edge_to_doors[edge])]


def _choose_marker_cell(component: list[tuple[int, int]], wall_mask: np.ndarray) -> tuple[int, int]:
    roomy = []
    for r, c in component:
        open_neighbors = sum(
            1
            for nr, nc in neighbors4(r, c)
            if 0 <= nr < wall_mask.shape[0] and 0 <= nc < wall_mask.shape[1] and not wall_mask[nr, nc]
        )
        if open_neighbors >= 2:
            roomy.append((r, c))
    return random.choice(roomy if roomy else component)


def _build_expected_output(
    full_wall_mask: np.ndarray,
    components: list[list[tuple[int, int]]],
    marked_components: list[int],
    colors: list[int],
) -> np.ndarray:
    out = np.where(full_wall_mask, GREY, BLACK).astype(int)
    for comp_idx, color in zip(marked_components, colors):
        for r, c in components[comp_idx]:
            out[r, c] = int(color)
    return out


def _input_marker_collision(labels: np.ndarray, marker_cells: list[tuple[int, int]]) -> bool:
    counts: dict[int, int] = defaultdict(int)
    for r, c in marker_cells:
        label = int(labels[r, c])
        counts[label] += 1
    return any(count > 1 for count in counts.values())


def _validate_instance(
    inp: np.ndarray,
    expected: np.ndarray,
    full_wall_mask: np.ndarray,
    tile: np.ndarray,
    marked_components: list[int],
    marker_cells: list[tuple[int, int]],
) -> bool:
    tile_h, tile_w = map(int, tile.shape)
    wall_phase_set = {
        (int(r), int(c))
        for r, c in np.argwhere(tile == 1)
    }
    visible_phase_set = {
        (int(r) % tile_h, int(c) % tile_w)
        for r, c in np.argwhere(inp == GREY)
    }
    if wall_phase_set != visible_phase_set:
        return False

    try:
        inferred_h, inferred_w, _inferred_tile = _infer_wall_tile(inp)
    except AssertionError:
        return False
    if (inferred_h, inferred_w) != (tile_h, tile_w):
        return False

    repaired = _repair_walls(inp)
    if not np.array_equal(repaired == GREY, full_wall_mask):
        return False

    input_labels, _input_components = _open_components(inp == GREY)
    if not _input_marker_collision(input_labels, marker_cells):
        return False

    _repaired_labels, repaired_components = _open_components(full_wall_mask)
    if len(repaired_components) <= len(marked_components):
        return False
    if np.array_equal(inp, expected):
        return False

    try:
        actual = main(inp)
    except AssertionError:
        return False
    return bool(np.array_equal(actual, expected))


def _make_instance() -> tuple[np.ndarray, np.ndarray] | None:
    tile = _sample_tile()
    if tile is None:
        return None
    tile_h, tile_w = map(int, tile.shape)

    reps_r = random.choice(_target_repetitions(tile_h))
    reps_c = random.choice(_target_repetitions(tile_w))
    full_wall_mask = _repeat_wall_tile(tile, reps_r, reps_c)
    labels, components = _open_components(full_wall_mask)
    if len(components) < MIN_COMPONENTS:
        return None

    areas = [len(component) for component in components]
    edge_to_doors = _door_candidates(full_wall_mask, labels, tile_h, tile_w)
    cluster = _choose_marked_cluster(edge_to_doors, areas)
    if cluster is None:
        return None
    cluster_components, cluster_doors = cluster
    if not any(bool(door["interior_phase"]) for door in cluster_doors):
        return None

    remaining = [idx for idx, area in enumerate(areas) if area >= MIN_MARK_AREA and idx not in cluster_components]
    random.shuffle(remaining)
    extra_mark_count = random.randint(0, min(2, len(remaining), len(MARKER_COLORS) - len(cluster_components)))
    marked_components = cluster_components + remaining[:extra_mark_count]
    if len(marked_components) < 2:
        return None

    marker_colors = random.sample(MARKER_COLORS, len(marked_components))
    marker_cells = [_choose_marker_cell(components[comp_idx], full_wall_mask) for comp_idx in marked_components]

    removed_cells = {tuple(door["cell"]) for door in cluster_doors}
    extra_edges = [
        edge
        for edge in edge_to_doors
        if set(edge).isdisjoint(cluster_components)
    ]
    random.shuffle(extra_edges)
    extra_removals = random.randint(0, min(2, len(extra_edges)))
    for edge in extra_edges[:extra_removals]:
        removed_cells.add(tuple(_choose_door(edge_to_doors[edge])["cell"]))

    inp = np.where(full_wall_mask, GREY, BLACK).astype(int)
    for r, c in removed_cells:
        inp[r, c] = BLACK
    for (r, c), color in zip(marker_cells, marker_colors):
        inp[r, c] = int(color)

    expected = _build_expected_output(full_wall_mask, components, marked_components, marker_colors)
    if not _validate_instance(inp, expected, full_wall_mask, tile, marked_components, marker_cells):
        return None
    return inp, expected


def generate_input() -> np.ndarray:
    for _ in range(GEN_ATTEMPTS):
        sample = _make_instance()
        if sample is not None:
            return sample[0]
    raise RuntimeError("failed to generate a valid periodic-wall chamber puzzle")
