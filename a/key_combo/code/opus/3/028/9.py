import itertools
import random
from collections import deque

import numpy as np


BLACK = 0
BLUE = 1
RED = 2
GREEN = 3
YELLOW = 4

Coord = tuple[int, int]
Shape = tuple[Coord, ...]

GEN_RETRY_LIMIT = 4000
SELF_VALIDATE_TRIALS = 3
SHAPE_AREAS = (3, 4, 5)
GRID_HEIGHT_RANGE = (14, 19)
GRID_WIDTH_RANGE = (19, 27)
MARKER_COL_RANGE = (7, 11)
SPRITE_COUNTS = (2, 3)
SPRITE_COUNT_WEIGHTS = (5, 4)
BLUE_AREA_RANGE = (30, 50)
OUTER_MARGIN_RANGE = (1, 2)
BLUE_GAP_RANGE = (1, 2)
TILING_MAX_SOLUTIONS = 2


def get_bg(grid: np.ndarray) -> int:
    values, counts = np.unique(grid, return_counts=True)
    return int(values[np.argmax(counts)])


def neighbors4(r: int, c: int):
    yield r - 1, c
    yield r + 1, c
    yield r, c - 1
    yield r, c + 1


def bbox_coords(coords: np.ndarray) -> tuple[int, int, int, int]:
    r0, c0 = coords.min(axis=0)
    r1, c1 = coords.max(axis=0)
    return int(r0), int(c0), int(r1), int(c1)


def bbox(mask: np.ndarray) -> tuple[int, int, int, int] | None:
    coords = np.argwhere(mask)
    if coords.size == 0:
        return None
    return bbox_coords(coords)


def crop_bbox(grid: np.ndarray, box: tuple[int, int, int, int]) -> np.ndarray:
    r0, c0, r1, c1 = box
    return grid[r0 : r1 + 1, c0 : c1 + 1].copy()


def components_mask(mask: np.ndarray, connectivity: int = 4) -> list[np.ndarray]:
    h, w = mask.shape
    seen = np.zeros_like(mask, dtype=bool)
    comps: list[np.ndarray] = []
    deltas = [(-1, 0), (1, 0), (0, -1), (0, 1)]
    if connectivity == 8:
        deltas += [(-1, -1), (-1, 1), (1, -1), (1, 1)]
    for r in range(h):
        for c in range(w):
            if not mask[r, c] or seen[r, c]:
                continue
            q = deque([(r, c)])
            seen[r, c] = True
            coords: list[Coord] = []
            while q:
                rr, cc = q.popleft()
                coords.append((rr, cc))
                for dr, dc in deltas:
                    nr, nc = rr + dr, cc + dc
                    if 0 <= nr < h and 0 <= nc < w and mask[nr, nc] and not seen[nr, nc]:
                        seen[nr, nc] = True
                        q.append((nr, nc))
            comps.append(np.array(coords, dtype=int))
    return comps


def normalize_shape(coords) -> Shape:
    cells = list(coords)
    min_r = min(r for r, _ in cells)
    min_c = min(c for _, c in cells)
    return tuple(sorted((int(r - min_r), int(c - min_c)) for r, c in cells))


def shape_dims(shape: Shape) -> tuple[int, int]:
    return max(r for r, _ in shape) + 1, max(c for _, c in shape) + 1


def mirror_shape_across_marker(shape_coords_abs: list[Coord], marker_col: int) -> Shape:
    mirrored = [(r, 2 * marker_col - c) for r, c in shape_coords_abs]
    return normalize_shape(mirrored)


def mirror_shape_local(shape: Shape) -> Shape:
    max_c = max(c for _, c in shape)
    return normalize_shape((r, max_c - c) for r, c in shape)


def shape_connected(shape: Shape) -> bool:
    cells = set(shape)
    start = next(iter(cells))
    q = deque([start])
    seen = {start}
    while q:
        r, c = q.popleft()
        for nr, nc in neighbors4(r, c):
            if (nr, nc) in cells and (nr, nc) not in seen:
                seen.add((nr, nc))
                q.append((nr, nc))
    return len(seen) == len(cells)


def all_shape_catalog() -> list[Shape]:
    universe = [(r, c) for r in range(4) for c in range(4)]
    seen: set[Shape] = set()
    shapes: list[Shape] = []
    for area in SHAPE_AREAS:
        for cells in itertools.combinations(universe, area):
            shape = normalize_shape(cells)
            if shape in seen:
                continue
            seen.add(shape)
            if not shape_connected(shape):
                continue
            h, w = shape_dims(shape)
            if h > 3 or w > 3:
                continue
            if shape == mirror_shape_local(shape):
                continue
            shapes.append(shape)
    shapes.sort(key=lambda s: (len(s), shape_dims(s), s))
    return shapes


SHAPE_CATALOG = all_shape_catalog()


def enumerate_placements(shape: Shape, hole_cells: set[Coord], hole_h: int, hole_w: int):
    sh, sw = shape_dims(shape)
    placements = []
    for r0 in range(hole_h - sh + 1):
        for c0 in range(hole_w - sw + 1):
            cells = frozenset((r0 + r, c0 + c) for r, c in shape)
            if cells.issubset(hole_cells):
                placements.append(((r0, c0), cells))
    return placements


def solve_exact_tiling(
    piece_shapes: list[Shape],
    hole_cells: set[Coord],
    hole_h: int,
    hole_w: int,
    max_solutions: int = TILING_MAX_SOLUTIONS,
):
    if sum(len(shape) for shape in piece_shapes) != len(hole_cells):
        return []

    placements_by_piece = [enumerate_placements(shape, hole_cells, hole_h, hole_w) for shape in piece_shapes]
    if any(len(opts) == 0 for opts in placements_by_piece):
        return []

    cell_to_options: dict[Coord, list[tuple[int, Coord, frozenset[Coord]]]] = {cell: [] for cell in hole_cells}
    for piece_idx, opts in enumerate(placements_by_piece):
        for anchor, cells in opts:
            for cell in cells:
                cell_to_options[cell].append((piece_idx, anchor, cells))

    used = [False] * len(piece_shapes)
    covered: set[Coord] = set()
    current: list[tuple[Coord, frozenset[Coord]] | None] = [None] * len(piece_shapes)
    solutions: list[list[tuple[Coord, frozenset[Coord]]]] = []

    def backtrack():
        if len(covered) == len(hole_cells):
            solutions.append([item for item in current if item is not None])
            return len(solutions) >= max_solutions

        remaining = hole_cells - covered
        best_options = None
        for cell in remaining:
            opts = []
            for piece_idx, anchor, cells in cell_to_options[cell]:
                if used[piece_idx]:
                    continue
                if not covered.isdisjoint(cells):
                    continue
                opts.append((piece_idx, anchor, cells))
            if not opts:
                return False
            if best_options is None or len(opts) < len(best_options):
                best_options = opts
                if len(best_options) == 1:
                    break

        assert best_options is not None
        best_options.sort(key=lambda item: (item[0], item[1][0], item[1][1]))
        for piece_idx, anchor, cells in best_options:
            used[piece_idx] = True
            current[piece_idx] = (anchor, cells)
            covered.update(cells)
            stop = backtrack()
            covered.difference_update(cells)
            current[piece_idx] = None
            used[piece_idx] = False
            if stop:
                return True
        return False

    backtrack()
    return solutions


def filter_oriented_solutions(solutions: list[list[tuple[Coord, frozenset[Coord]]]]):
    kept = []
    for solution in solutions:
        ok = True
        for idx in range(0, len(solution), 2):
            if solution[idx][0] > solution[idx + 1][0]:
                ok = False
                break
        if ok:
            kept.append(solution)
    return kept


def render_hole_pattern(
    piece_shapes: list[Shape],
    piece_colors: list[int],
    hole_h: int,
    hole_w: int,
    solution: list[tuple[Coord, frozenset[Coord]]],
) -> tuple[int, ...]:
    arr = np.zeros((hole_h, hole_w), dtype=int)
    for piece_idx, (anchor, _cells) in enumerate(solution):
        ar, ac = anchor
        for dr, dc in piece_shapes[piece_idx]:
            arr[ar + dr, ac + dc] = piece_colors[piece_idx]
    return tuple(int(v) for v in arr.ravel())


def find_marker_col(grid: np.ndarray) -> int:
    cols = [c for c in range(grid.shape[1]) if np.all(grid[:, c] == GREEN)]
    assert len(cols) == 1, "expected one full green marker column"
    return int(cols[0])


def extract_red_sprites(grid: np.ndarray, marker_col: int) -> list[np.ndarray]:
    red_mask = grid == RED
    comps = components_mask(red_mask, connectivity=4)
    sprites = []
    for coords in comps:
        assert np.all(coords[:, 1] < marker_col), "red sprites must lie left of the marker"
        sprites.append(coords)
    sprites.sort(key=lambda arr: (arr[:, 0].min(), arr[:, 1].min()))
    assert 2 <= len(sprites) <= 3, "expected 2 or 3 red sprites"
    return sprites


def find_rect_hole(grid: np.ndarray, blue_mask: np.ndarray) -> tuple[tuple[int, int, int, int], set[Coord]]:
    h, w = grid.shape
    bg_components = components_mask(grid == BLACK, connectivity=4)
    hole_box = None
    hole_cells_abs = None
    for coords in bg_components:
        if np.any(coords[:, 0] == 0) or np.any(coords[:, 0] == h - 1) or np.any(coords[:, 1] == 0) or np.any(coords[:, 1] == w - 1):
            continue
        r0, c0, r1, c1 = bbox_coords(coords)
        if coords.shape[0] != (r1 - r0 + 1) * (c1 - c0 + 1):
            continue
        window = grid[r0 : r1 + 1, c0 : c1 + 1]
        if not np.all(window == BLACK):
            continue
        if r0 == 0 or c0 == 0 or r1 == h - 1 or c1 == w - 1:
            continue
        ring = grid[r0 - 1 : r1 + 2, c0 - 1 : c1 + 2].copy()
        ring[1:-1, 1:-1] = BLUE
        if not np.all(ring == BLUE):
            continue
        if not np.any(blue_mask[r0 - 1 : r1 + 2, c0 - 1 : c1 + 2]):
            continue
        assert hole_box is None, "expected one enclosed rectangular hole"
        hole_box = (r0, c0, r1, c1)
        hole_cells_abs = {(int(r), int(c)) for r, c in coords}
    assert hole_box is not None and hole_cells_abs is not None, "expected one enclosed rectangular hole"
    return hole_box, hole_cells_abs


def placement_solution_for_grid(grid: np.ndarray):
    marker_col = find_marker_col(grid)
    sprites = extract_red_sprites(grid, marker_col)

    piece_shapes: list[Shape] = []
    piece_colors: list[int] = []
    source_cells: list[list[Coord]] = []
    for coords in sprites:
        cells = [(int(r), int(c)) for r, c in coords]
        shape = normalize_shape(cells)
        mirrored = mirror_shape_across_marker(cells, marker_col)
        piece_shapes.append(shape)
        piece_colors.append(RED)
        piece_shapes.append(mirrored)
        piece_colors.append(YELLOW)
        source_cells.append(cells)

    blue_mask = grid == BLUE
    blue_comps = components_mask(blue_mask, connectivity=8)
    assert len(blue_comps) == 1, "expected one blue object"
    hole_box, hole_cells_abs = find_rect_hole(grid, blue_mask)
    r0, c0, r1, c1 = hole_box
    hole_h = r1 - r0 + 1
    hole_w = c1 - c0 + 1
    hole_cells_norm = {(r - r0, c - c0) for r, c in hole_cells_abs}

    solutions = solve_exact_tiling(piece_shapes, hole_cells_norm, hole_h, hole_w, max_solutions=64)
    assert solutions, "expected at least one exact tiling"
    oriented = filter_oriented_solutions(solutions)
    assert len(oriented) == 1, "expected one oriented exact tiling"
    return {
        "marker_col": marker_col,
        "sprites": sprites,
        "piece_shapes": piece_shapes,
        "piece_colors": piece_colors,
        "hole_box": hole_box,
        "solution": oriented[0],
    }


def render_solution(grid: np.ndarray, info) -> np.ndarray:
    out = np.array(grid, dtype=int, copy=True)
    for coords in info["sprites"]:
        for r, c in coords:
            out[int(r), int(c)] = BLACK

    hole_r0, hole_c0, _, _ = info["hole_box"]
    for piece_idx, (anchor, _cells) in enumerate(info["solution"]):
        color = info["piece_colors"][piece_idx]
        ar, ac = anchor
        for dr, dc in info["piece_shapes"][piece_idx]:
            out[hole_r0 + ar + dr, hole_c0 + ac + dc] = color
    return out


def main(grid: np.ndarray) -> np.ndarray:
    """
    Reflect each red sprite across the green marker line, then use the red shapes and their yellow mirrors to tile the blue hole.

    Input structure:
    - Black is the background.
    - Exactly one full green column is a vertical marker line.
    - Left of that marker line are 2 or 3 red 4-connected sprites.
    - Right of the marker line is one 8-connected blue object containing one enclosed rectangular black hole.

    Rule:
    - Read the green full column as marker column m.
    - Extract every red sprite on the left and normalize its shape.
    - Mirror each sprite across column m using absolute coordinates (r, c) -> (r, 2m-c), then normalize the mirrored shape and treat it as a yellow candidate sprite.
    - Use every red sprite and every yellow mirrored sprite exactly once, translating but not rotating them, to obtain the unique exact tiling of the blue object's hole.
    - If the rectangle admits the unavoidable left-right companion cover, keep the orientation in which each red sprite's anchor is row-major no later than the anchor of its own yellow mirror.
    - Draw that tiling inside the hole, preserve the blue object and green marker line, and erase the original red sprites from the left side.

    Key insight:
    - The marker line is not just a divider: it creates a second sprite set by reflection, and only the combined original-plus-mirrored shape pool can exactly fill the hole.
    """
    g = np.asarray(grid, dtype=int)
    info = placement_solution_for_grid(g)
    return render_solution(g, info)


def choose_sprite_shapes(rng: random.Random, sprite_count: int) -> list[Shape]:
    for _ in range(200):
        chosen = rng.sample(SHAPE_CATALOG, sprite_count)
        all_shapes = set()
        ok = True
        for shape in chosen:
            mirrored = mirror_shape_local(shape)
            if shape == mirrored:
                ok = False
                break
            if shape in all_shapes or mirrored in all_shapes:
                ok = False
                break
            all_shapes.add(shape)
            all_shapes.add(mirrored)
        if ok:
            return chosen
    raise RuntimeError("failed to choose non-conflicting sprite shapes")


def choose_hole_layout(rng: random.Random, shapes: list[Shape]):
    pieces: list[Shape] = []
    for shape in shapes:
        pieces.append(shape)
        pieces.append(mirror_shape_local(shape))
    piece_colors = [RED if idx % 2 == 0 else YELLOW for idx in range(len(pieces))]
    total_area = sum(len(shape) for shape in pieces)
    max_h = max(shape_dims(shape)[0] for shape in pieces)
    max_w = max(shape_dims(shape)[1] for shape in pieces)

    factor_pairs = []
    for hole_h in range(2, total_area + 1):
        if total_area % hole_h != 0:
            continue
        hole_w = total_area // hole_h
        if hole_h < max_h or hole_w < max_w:
            continue
        if hole_h > 8 or hole_w > 8:
            continue
        factor_pairs.append((hole_h, hole_w))
    rng.shuffle(factor_pairs)

    for hole_h, hole_w in factor_pairs:
        hole_cells = {(r, c) for r in range(hole_h) for c in range(hole_w)}
        solutions = solve_exact_tiling(pieces, hole_cells, hole_h, hole_w, max_solutions=64)
        if not solutions:
            continue
        oriented = filter_oriented_solutions(solutions)
        if len(oriented) != 1:
            continue
        patterns = {render_hole_pattern(pieces, piece_colors, hole_h, hole_w, sol) for sol in oriented}
        if len(patterns) == 1:
            return pieces, (hole_h, hole_w), oriented[0]
    raise RuntimeError("failed to find a unique rectangular hole tiling")


def choose_blue_frame(rng: random.Random, hole_h: int, hole_w: int):
    options = []
    for top in (1, 2, 3):
        for bottom in (1, 2, 3):
            for left in (1, 2, 3):
                for right in (1, 2, 3):
                    blue_area = (hole_h + top + bottom) * (hole_w + left + right) - hole_h * hole_w
                    if BLUE_AREA_RANGE[0] <= blue_area <= BLUE_AREA_RANGE[1]:
                        options.append((top, bottom, left, right))
    if not options:
        raise RuntimeError("failed to choose blue frame thickness")
    return rng.choice(options)


def place_red_sprites(
    rng: random.Random,
    grid_shape: tuple[int, int],
    marker_col: int,
    shapes: list[Shape],
):
    h, _w = grid_shape
    occupied = np.zeros(grid_shape, dtype=bool)
    anchors: list[Coord] = []
    placed_cells: list[list[Coord]] = []
    for shape in rng.sample(shapes, len(shapes)):
        sh, sw = shape_dims(shape)
        candidates: list[Coord] = []
        max_col = marker_col - sw - 1
        if max_col < 1:
            raise RuntimeError("marker column leaves no room for sprites")
        for r0 in range(1, h - sh):
            for c0 in range(1, max_col + 1):
                cells = [(r0 + r, c0 + c) for r, c in shape]
                ok = True
                for rr, cc in cells:
                    if occupied[rr, cc]:
                        ok = False
                        break
                    for nr in range(rr - 1, rr + 2):
                        for nc in range(cc - 1, cc + 2):
                            if 0 <= nr < h and 0 <= nc < marker_col and occupied[nr, nc]:
                                ok = False
                                break
                        if not ok:
                            break
                if ok:
                    candidates.append((r0, c0))
        if not candidates:
            raise RuntimeError("failed to place red sprites")
        anchor = rng.choice(candidates)
        anchors.append(anchor)
        cells = []
        for dr, dc in shape:
            rr = anchor[0] + dr
            cc = anchor[1] + dc
            occupied[rr, cc] = True
            cells.append((rr, cc))
        placed_cells.append(cells)
    return anchors, placed_cells


def validate_generated_instance(grid: np.ndarray):
    info = placement_solution_for_grid(grid)
    out = render_solution(grid, info)

    marker_col = info["marker_col"]
    left = out[:, :marker_col]
    assert not np.any(left == RED), "left-side source sprites must be erased"

    hole_r0, hole_c0, hole_r1, hole_c1 = info["hole_box"]
    hole = out[hole_r0 : hole_r1 + 1, hole_c0 : hole_c1 + 1]
    assert not np.any(hole == BLACK), "hole must be fully tiled"
    assert np.any(hole == RED), "hole tiling must use original sprites"
    assert np.any(hole == YELLOW), "hole tiling must use mirrored sprites"

    red_sprites = extract_red_sprites(grid, marker_col)
    assert info["piece_colors"].count(RED) == len(red_sprites)
    assert info["piece_colors"].count(YELLOW) == len(red_sprites)

    for coords in red_sprites:
        cells = [(int(r), int(c)) for r, c in coords]
        shape = normalize_shape(cells)
        mirrored = mirror_shape_across_marker(cells, marker_col)
        assert shape != mirrored, "each sprite must change under reflection"

    return True


def generate_input() -> np.ndarray:
    for _ in range(GEN_RETRY_LIMIT):
        seed = random.randrange(1 << 30)
        rng = random.Random(seed)

        sprite_count = rng.choices(SPRITE_COUNTS, weights=SPRITE_COUNT_WEIGHTS, k=1)[0]
        try:
            shapes = choose_sprite_shapes(rng, sprite_count)
            _pieces, (hole_h, hole_w), _solution = choose_hole_layout(rng, shapes)
        except RuntimeError:
            continue

        top, bottom, left, right = choose_blue_frame(rng, hole_h, hole_w)
        outer_h = hole_h + top + bottom
        outer_w = hole_w + left + right

        marker_col = rng.randint(*MARKER_COL_RANGE)
        gap_to_blue = rng.randint(*BLUE_GAP_RANGE)
        blue_left = marker_col + gap_to_blue + 1
        min_w = max(GRID_WIDTH_RANGE[0], blue_left + outer_w + rng.randint(*OUTER_MARGIN_RANGE), 2 * marker_col + 2)
        max_w = GRID_WIDTH_RANGE[1]
        if min_w > max_w:
            continue
        grid_w = rng.randint(min_w, max_w)

        min_h = max(GRID_HEIGHT_RANGE[0], outer_h + 2)
        max_h = GRID_HEIGHT_RANGE[1]
        if min_h > max_h:
            continue
        grid_h = rng.randint(min_h, max_h)

        blue_top = rng.randint(1, grid_h - outer_h - 1)
        if blue_left + outer_w >= grid_w:
            continue

        grid = np.zeros((grid_h, grid_w), dtype=int)
        grid[:, marker_col] = GREEN

        outer_r0, outer_c0 = blue_top, blue_left
        outer_r1 = outer_r0 + outer_h - 1
        outer_c1 = outer_c0 + outer_w - 1
        grid[outer_r0 : outer_r1 + 1, outer_c0 : outer_c1 + 1] = BLUE

        hole_r0 = outer_r0 + top
        hole_c0 = outer_c0 + left
        hole_r1 = hole_r0 + hole_h - 1
        hole_c1 = hole_c0 + hole_w - 1
        grid[hole_r0 : hole_r1 + 1, hole_c0 : hole_c1 + 1] = BLACK

        try:
            _anchors, placed_cells = place_red_sprites(rng, grid.shape, marker_col, shapes)
        except RuntimeError:
            continue
        for cells in placed_cells:
            for r, c in cells:
                grid[r, c] = RED

        # Reject accidental foreground on the right beyond the intended blue object.
        right_foreground = np.argwhere((grid[:, marker_col + 1 :] != BLACK) & (grid[:, marker_col + 1 :] != BLUE))
        if right_foreground.size != 0:
            continue

        try:
            validate_generated_instance(grid)
        except AssertionError:
            continue

        # Spot-check stability on a few random relayouts of the same attempt seed.
        try:
            for _ in range(SELF_VALIDATE_TRIALS):
                _ = main(grid)
        except AssertionError:
            continue

        return grid

    raise RuntimeError("failed to generate a valid sparse-marker mirror tiling input")
