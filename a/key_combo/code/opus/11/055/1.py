import math
import random
from collections import deque

import numpy as np


BLACK = 0
BLUE = 1
RED = 2
GREEN = 3
YELLOW = 4
GREY = 5
ORANGE = 7

MARKER_COLORS = [BLUE, RED, GREEN, YELLOW, ORANGE]
NEIGHBORS4 = [(-1, 0), (1, 0), (0, -1), (0, 1)]
NEIGHBORS8 = [
    (-1, -1), (-1, 0), (-1, 1),
    (0, -1),           (0, 1),
    (1, -1),  (1, 0),  (1, 1),
]

DISTANCE_TEMPLATES = {
    3: [(2, 8, 9), (4, 8, 9), (2, 5, 9)],
    4: [(2, 4, 8, 9), (2, 5, 8, 9), (2, 4, 5, 9)],
    5: [(2, 4, 5, 8, 9), (2, 4, 8, 9, 13), (2, 5, 8, 9, 13)],
}

MAX_GENERATION_ATTEMPTS = 500


def _in_bounds(shape: tuple[int, int], r: int, c: int) -> bool:
    h, w = shape
    return 0 <= r < h and 0 <= c < w


def _components_from_mask(mask: np.ndarray) -> list[np.ndarray]:
    h, w = mask.shape
    seen = np.zeros((h, w), dtype=bool)
    comps: list[np.ndarray] = []
    for r in range(h):
        for c in range(w):
            if not mask[r, c] or seen[r, c]:
                continue
            q: deque[tuple[int, int]] = deque([(r, c)])
            seen[r, c] = True
            cells: list[tuple[int, int]] = []
            while q:
                rr, cc = q.popleft()
                cells.append((rr, cc))
                for dr, dc in NEIGHBORS4:
                    nr, nc = rr + dr, cc + dc
                    if _in_bounds(mask.shape, nr, nc) and mask[nr, nc] and not seen[nr, nc]:
                        seen[nr, nc] = True
                        q.append((nr, nc))
            comps.append(np.array(cells, dtype=int))
    return comps


def _bbox(coords: np.ndarray) -> tuple[int, int, int, int]:
    r0, c0 = coords.min(axis=0)
    r1, c1 = coords.max(axis=0)
    return int(r0), int(c0), int(r1), int(c1)


def _paint_square(out: np.ndarray, side: int, color: int, left_col: int) -> None:
    out[out.shape[0] - side : out.shape[0], left_col : left_col + side] = color


def _minimum_squared_distances(component: np.ndarray, markers: np.ndarray) -> np.ndarray:
    diffs = component[:, None, :] - markers[None, :, :]
    return (diffs * diffs).sum(axis=2).min(axis=0)


def _minimum_manhattan_distance(component: np.ndarray, marker: tuple[int, int]) -> int:
    mr, mc = marker
    deltas = np.abs(component - np.array([[mr, mc]], dtype=int))
    return int(deltas.sum(axis=1).min())


def _analyze_instance(grid: np.ndarray) -> dict:
    g = np.asarray(grid, dtype=int)
    assert g.ndim == 2, "expected a 2D grid"
    assert max(g.shape) <= 30, f"grid exceeds 30x30: {g.shape}"
    assert int(np.count_nonzero(g == BLACK)) > 0, "expected black background"

    grey_components = _components_from_mask(g == GREY)
    assert 3 <= len(grey_components) <= 5, "expected 3-5 grey components"

    marker_positions = np.argwhere((g != BLACK) & (g != GREY))
    assert 3 <= len(marker_positions) <= 5, "expected 3-5 markers"
    marker_colors = [int(g[r, c]) for r, c in marker_positions]
    assert len(set(marker_colors)) == len(marker_colors), "marker colors must be unique"

    marker_lookup = {tuple(map(int, pos)): int(g[tuple(pos)]) for pos in marker_positions}
    for (r, c), color in marker_lookup.items():
        assert color in MARKER_COLORS, "unexpected marker color"
        for dr, dc in NEIGHBORS4:
            nr, nc = r + dr, c + dc
            if _in_bounds(g.shape, nr, nc):
                assert g[nr, nc] != GREY, "markers cannot touch grey components orthogonally"
        for dr, dc in NEIGHBORS8:
            nr, nc = r + dr, c + dc
            if _in_bounds(g.shape, nr, nc) and (nr, nc) in marker_lookup:
                raise AssertionError("markers must stay visually isolated")

    blocks: list[dict] = []
    marker_array = marker_positions.astype(int)
    for comp in grey_components:
        area = int(comp.shape[0])
        side = math.isqrt(area)
        assert side * side == area, "grey component area must be a perfect square"

        r0, c0, r1, c1 = _bbox(comp)
        bbox_area = (r1 - r0 + 1) * (c1 - c0 + 1)
        assert bbox_area > area, "grey component must be irregular, not a filled rectangle"

        distances = _minimum_squared_distances(comp, marker_array)
        best = int(distances.min())
        winners = np.flatnonzero(distances == best)
        assert len(winners) == 1, "grey component must have a unique nearest marker"
        winner_idx = int(winners[0])
        marker_cell = tuple(map(int, marker_array[winner_idx]))
        blocks.append(
            {
                "coords": comp,
                "area": area,
                "side": side,
                "bbox": (r0, c0, r1, c1),
                "distance": best,
                "color": int(g[marker_cell]),
                "marker": marker_cell,
                "manhattan": _minimum_manhattan_distance(comp, marker_cell),
                "centroid": tuple(map(float, comp.mean(axis=0))),
            }
        )

    distances = [block["distance"] for block in blocks]
    sides = [block["side"] for block in blocks]
    assert len(set(distances)) == len(distances), "nearest squared distances must be globally unique"
    assert len(set(sides)) == len(sides), "square side lengths must be globally unique"

    ordered_by_distance = sorted(blocks, key=lambda block: block["distance"])
    packed_width = sum(block["side"] for block in ordered_by_distance) + len(ordered_by_distance) - 1
    assert packed_width <= g.shape[1], "packed output must fit inside the input width"
    assert max(block["side"] for block in ordered_by_distance) <= g.shape[0], "packed output must fit inside the input height"

    return {
        "grid": g,
        "blocks": blocks,
        "ordered_by_distance": ordered_by_distance,
    }


def main(grid: np.ndarray) -> np.ndarray:
    """
    Repack the scattered grey components into distance-ordered solid squares recolored by their nearest markers.

    Input structure:
    - The grid has a black background with 3-5 disjoint grey 4-connected components.
    - Each grey component has a unique perfect-square area and is an irregular non-rectangular shape.
    - The remaining foreground cells are isolated single-pixel markers in distinct colors.

    Rule:
    - For each grey component, compute its pixel count P and convert it to a square side length S = sqrt(P).
    - Measure the minimum squared Euclidean distance from that component to every marker pixel; the unique smallest one gives both the component's distance key D and its output color C.
    - Sort the components by D from smallest to largest, erase the markers, and place filled SxS squares from left to right on a black grid of the same size, bottom-aligned with one black column between neighbors.

    Key insight:
    - The same nearest-marker relation controls two independent output properties at once: it decides both the repacking order and the recolor, while the original grey pixel count alone fixes each square's size.
    """
    info = _analyze_instance(grid)
    g = info["grid"]
    ordered = info["ordered_by_distance"]

    out = np.full_like(g, BLACK)
    col = 0
    for index, block in enumerate(ordered):
        _paint_square(out, block["side"], block["color"], col)
        col += block["side"]
        if index != len(ordered) - 1:
            col += 1
    return out


def _connected_cells(cells: set[tuple[int, int]]) -> bool:
    if not cells:
        return False
    start = next(iter(cells))
    q = deque([start])
    seen = {start}
    while q:
        r, c = q.popleft()
        for dr, dc in NEIGHBORS4:
            nxt = (r + dr, c + dc)
            if nxt in cells and nxt not in seen:
                seen.add(nxt)
                q.append(nxt)
    return len(seen) == len(cells)


def _generate_irregular_shape(side: int) -> np.ndarray:
    area = side * side
    for _ in range(250):
        canvas_h = side + random.randint(2, 4)
        canvas_w = side + random.randint(2, 4)
        top = random.randint(1, canvas_h - side - 1)
        left = random.randint(1, canvas_w - side - 1)
        cells = {(top + r, left + c) for r in range(side) for c in range(side)}

        mutation_budget = random.randint(2, max(4, side + 2))
        for _ in range(mutation_budget):
            removable = []
            for cell in cells:
                if all((cell[0] + dr, cell[1] + dc) in cells for dr, dc in NEIGHBORS4):
                    continue
                remaining = cells - {cell}
                if _connected_cells(remaining):
                    removable.append(cell)
            if not removable:
                break

            remove_cell = random.choice(removable)
            remaining = cells - {remove_cell}
            add_candidates = []
            for rr in range(canvas_h):
                for cc in range(canvas_w):
                    if (rr, cc) in remaining:
                        continue
                    if not any((rr + dr, cc + dc) in remaining for dr, dc in NEIGHBORS4):
                        continue
                    moved = remaining | {(rr, cc)}
                    coords = np.array(sorted(moved), dtype=int)
                    r0, c0, r1, c1 = _bbox(coords)
                    if (r1 - r0 + 1) > side + 3 or (c1 - c0 + 1) > side + 3:
                        continue
                    if _connected_cells(moved):
                        add_candidates.append((rr, cc))
            if add_candidates:
                cells = remaining | {random.choice(add_candidates)}

        coords = np.array(sorted(cells), dtype=int)
        r0, c0, r1, c1 = _bbox(coords)
        bbox_h = r1 - r0 + 1
        bbox_w = c1 - c0 + 1
        if bbox_h < 2 or bbox_w < 2:
            continue
        if bbox_h * bbox_w == area:
            continue
        if bbox_h > side + 3 or bbox_w > side + 3:
            continue
        normalized = coords - np.array([[r0, c0]], dtype=int)
        if len({tuple(cell) for cell in normalized.tolist()}) != area:
            continue
        return normalized
    raise RuntimeError(f"failed to generate irregular component for side {side}")


def _padded_cells(cells: set[tuple[int, int]], shape: tuple[int, int]) -> set[tuple[int, int]]:
    out: set[tuple[int, int]] = set()
    for r, c in cells:
        for dr, dc in [(0, 0), *NEIGHBORS8]:
            nr, nc = r + dr, c + dc
            if _in_bounds(shape, nr, nc):
                out.add((nr, nc))
    return out


def _slot_layout(h: int, w: int, n_blocks: int) -> list[tuple[int, int, int, int]]:
    rows = 2
    cols = max(2, math.ceil(n_blocks / 2))
    slots: list[tuple[int, int, int, int]] = []
    row_edges = np.linspace(0, h, rows + 1, dtype=int)
    col_edges = np.linspace(0, w, cols + 1, dtype=int)
    for r_idx in range(rows):
        for c_idx in range(cols):
            r0 = int(row_edges[r_idx])
            r1 = int(row_edges[r_idx + 1])
            c0 = int(col_edges[c_idx])
            c1 = int(col_edges[c_idx + 1])
            slots.append((r0, r1, c0, c1))
    return slots


def _place_components(
    shapes: list[np.ndarray],
    h: int,
    w: int,
) -> list[np.ndarray] | None:
    slots = _slot_layout(h, w, len(shapes))
    slot_indices = random.sample(range(len(slots)), len(shapes))
    placed: list[np.ndarray] = []
    blocked: set[tuple[int, int]] = set()

    for shape, slot_idx in zip(shapes, slot_indices):
        r0, r1, c0, c1 = slots[slot_idx]
        bbox_h = int(shape[:, 0].max()) + 1
        bbox_w = int(shape[:, 1].max()) + 1
        max_top = r1 - bbox_h
        max_left = c1 - bbox_w
        if max_top < r0 or max_left < c0:
            return None

        candidates: list[tuple[int, int]] = []
        for top in range(r0, max_top + 1):
            for left in range(c0, max_left + 1):
                cells = {(top + int(rr), left + int(cc)) for rr, cc in shape}
                if cells & blocked:
                    continue
                candidates.append((top, left))
        if not candidates:
            return None

        random.shuffle(candidates)
        chosen = None
        for top, left in candidates:
            cells = {(top + int(rr), left + int(cc)) for rr, cc in shape}
            if cells & blocked:
                continue
            chosen = np.array(sorted(cells), dtype=int)
            placed.append(chosen)
            blocked |= _padded_cells(cells, (h, w))
            break
        if chosen is None:
            return None

    return placed


def _cell_touches_grey(coords: np.ndarray, cell: tuple[int, int]) -> bool:
    r, c = cell
    points = {tuple(map(int, xy)) for xy in coords.tolist()}
    for dr, dc in NEIGHBORS4:
        if (r + dr, c + dc) in points:
            return True
    return False


def _squared_distance_to_component(coords: np.ndarray, cell: tuple[int, int]) -> int:
    r, c = cell
    diff = coords - np.array([[r, c]], dtype=int)
    return int((diff * diff).sum(axis=1).min())


def _marker_candidates(
    components: list[np.ndarray],
    block_index: int,
    target_distance: int,
    grid_shape: tuple[int, int],
) -> list[tuple[int, int]]:
    h, w = grid_shape
    all_grey = set()
    for comp in components:
        all_grey |= {tuple(map(int, xy)) for xy in comp.tolist()}

    candidates: list[tuple[int, int, int]] = []
    own = components[block_index]
    for r in range(h):
        for c in range(w):
            cell = (r, c)
            if cell in all_grey:
                continue
            if any((r + dr, c + dc) in all_grey for dr, dc in NEIGHBORS4):
                continue
            dist = _squared_distance_to_component(own, cell)
            if dist != target_distance:
                continue
            other_best = min(
                _squared_distance_to_component(components[j], cell)
                for j in range(len(components))
                if j != block_index
            )
            candidates.append((other_best - dist, r, c))
    candidates.sort(key=lambda item: (-item[0], random.random()))
    return [(r, c) for _gap, r, c in candidates]


def _search_markers(
    components: list[np.ndarray],
    target_distances: list[int],
) -> list[tuple[int, int]] | None:
    h = max(int(comp[:, 0].max()) for comp in components) + 1
    w = max(int(comp[:, 1].max()) for comp in components) + 1
    candidate_lists = [
        _marker_candidates(components, index, target_distances[index], (h, w))
        for index in range(len(components))
    ]
    if any(not candidates for candidates in candidate_lists):
        return None

    order = sorted(range(len(components)), key=lambda idx: len(candidate_lists[idx]))
    chosen: list[tuple[int, int] | None] = [None] * len(components)
    used: set[tuple[int, int]] = set()

    def compatible(block_idx: int, cell: tuple[int, int]) -> bool:
        if cell in used:
            return False
        if any(abs(cell[0] - other[0]) <= 1 and abs(cell[1] - other[1]) <= 1 for other in used):
            return False

        own_target = target_distances[block_idx]
        for other_idx, other_cell in enumerate(chosen):
            if other_cell is None:
                continue
            if _squared_distance_to_component(components[block_idx], other_cell) <= own_target:
                return False
            other_target = target_distances[other_idx]
            if _squared_distance_to_component(components[other_idx], cell) <= other_target:
                return False
        return True

    def dfs(pos: int) -> bool:
        if pos == len(order):
            return True
        block_idx = order[pos]
        for cell in candidate_lists[block_idx]:
            if not compatible(block_idx, cell):
                continue
            chosen[block_idx] = cell
            used.add(cell)
            if dfs(pos + 1):
                return True
            used.remove(cell)
            chosen[block_idx] = None
        return False

    if not dfs(0):
        return None
    return [cell for cell in chosen if cell is not None]


def _build_grid(
    h: int,
    w: int,
    components: list[np.ndarray],
    markers: list[tuple[int, int]],
    colors_by_block: list[int],
) -> np.ndarray:
    grid = np.zeros((h, w), dtype=int)
    for comp in components:
        for r, c in comp:
            grid[int(r), int(c)] = GREY
    for color, (r, c) in zip(colors_by_block, markers):
        grid[r, c] = int(color)
    return grid


def _packed_output_from_order(blocks: list[dict]) -> np.ndarray:
    h = int(max(block["coords"][:, 0].max() for block in blocks)) + 1
    w = int(max(block["coords"][:, 1].max() for block in blocks)) + 1
    out = np.zeros((h, w), dtype=int)
    ordered = sorted(blocks, key=lambda block: block["distance"])
    col = 0
    for i, block in enumerate(ordered):
        _paint_square(out, block["side"], block["color"], col)
        col += block["side"]
        if i != len(ordered) - 1:
            col += 1
    return out


def _is_non_degenerate(grid: np.ndarray) -> bool:
    info = _analyze_instance(grid)
    blocks = info["blocks"]
    dist_order = [block["distance"] for block in sorted(blocks, key=lambda b: b["distance"])]

    by_distance = [id(block) for block in sorted(blocks, key=lambda b: b["distance"])]
    by_size = [id(block) for block in sorted(blocks, key=lambda b: b["side"])]
    if by_distance == by_size:
        return False

    by_left = [id(block) for block in sorted(blocks, key=lambda b: (b["bbox"][1], b["bbox"][0]))]
    by_top = [id(block) for block in sorted(blocks, key=lambda b: (b["bbox"][0], b["bbox"][1]))]
    if by_distance == by_left or by_distance == by_top:
        return False

    manhattan_order = [id(block) for block in sorted(blocks, key=lambda b: b["manhattan"])]
    if by_distance == manhattan_order:
        return False

    if 8 not in dist_order or 9 not in dist_order:
        return False

    colors = [block["color"] for block in blocks]
    if len(set(colors)) < 3:
        return False

    correct = main(grid)
    wrong_area = _packed_output_from_order(
        sorted(blocks, key=lambda b: b["side"])
    )
    if np.array_equal(correct, wrong_area):
        return False

    return True


def _attempt_generation() -> np.ndarray | None:
    n_blocks = random.randint(3, 5)
    side_choices = random.sample([2, 3, 4, 5, 6], n_blocks)
    output_width = sum(sorted(side_choices)) + n_blocks - 1

    min_h = max(14, max(side_choices) + 6)
    max_h = min(22, 30)
    min_w = max(20, output_width + 1)
    max_w = min(30, output_width + 6)
    if min_h > max_h or min_w > max_w:
        return None

    h = random.randint(min_h, max_h)
    w = random.randint(min_w, max_w)
    shapes = [_generate_irregular_shape(side) for side in side_choices]
    components = _place_components(shapes, h, w)
    if components is None:
        return None

    distance_values = list(random.choice(DISTANCE_TEMPLATES[n_blocks]))
    order = list(range(n_blocks))
    random.shuffle(order)
    if order == sorted(order, key=lambda idx: side_choices[idx]):
        return None

    target_distances = [0] * n_blocks
    for rank, block_idx in enumerate(order):
        target_distances[block_idx] = distance_values[rank]

    markers = _search_markers(components, target_distances)
    if markers is None:
        return None

    colors = random.sample(MARKER_COLORS, n_blocks)
    grid = _build_grid(h, w, components, markers, colors)

    try:
        info = _analyze_instance(grid)
        actual_distances = [block["distance"] for block in info["blocks"]]
        if sorted(actual_distances) != sorted(distance_values):
            return None
        if not _is_non_degenerate(grid):
            return None
        return grid
    except AssertionError:
        return None


def generate_input() -> np.ndarray:
    for _ in range(MAX_GENERATION_ATTEMPTS):
        grid = _attempt_generation()
        if grid is not None:
            return grid
    raise RuntimeError("failed to generate a non-degenerate distance-ordered repacking instance")
