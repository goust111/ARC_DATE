import random

import numpy as np


BLACK = 0
BLUE = 1
RED = 2
GREEN = 3
YELLOW = 4
GREY = 5
ORANGE = 7
INTERFACE_COLORS = [BLUE, RED, GREEN, YELLOW, ORANGE]

DIRS = [(-1, 0), (0, 1), (1, 0), (0, -1)]
DIR_TO_INDEX = {step: idx for idx, step in enumerate(DIRS)}

MAX_GENERATION_ATTEMPTS = 400
MAX_PATH_ATTEMPTS = 200
MAX_PLACEMENT_ATTEMPTS = 200
MAX_PLACEMENT_TRIES_PER_PIECE = 200


def _components(mask: np.ndarray) -> list[np.ndarray]:
    h, w = mask.shape
    seen = np.zeros_like(mask, dtype=bool)
    components: list[np.ndarray] = []

    for r in range(h):
        for c in range(w):
            if not mask[r, c] or seen[r, c]:
                continue
            stack = [(r, c)]
            seen[r, c] = True
            coords: list[tuple[int, int]] = []
            while stack:
                rr, cc = stack.pop()
                coords.append((rr, cc))
                for dr, dc in DIRS:
                    nr = rr + dr
                    nc = cc + dc
                    if 0 <= nr < h and 0 <= nc < w and mask[nr, nc] and not seen[nr, nc]:
                        seen[nr, nc] = True
                        stack.append((nr, nc))
            components.append(np.array(coords, dtype=int))

    return components


def _edge_center(side: int, size: int) -> tuple[int, int]:
    mid = size // 2
    if side == 0:
        return (0, mid)
    if side == 1:
        return (mid, size - 1)
    if side == 2:
        return (size - 1, mid)
    return (mid, 0)


def _opposite(side: int) -> int:
    return (side + 2) % 4


def _rotate_side(side: int, rot_ccw: int) -> int:
    return (side - rot_ccw) % 4


def _rotation_to_side(original_side: int, desired_side: int) -> int:
    return (original_side - desired_side) % 4


def _extract_pieces(grid: np.ndarray) -> tuple[list[dict], int]:
    components = _components(grid != BLACK)
    assert len(components) >= 2

    pieces: list[dict] = []
    size = None
    for coords in components:
        r0, c0 = coords.min(axis=0)
        r1, c1 = coords.max(axis=0)
        side = int(r1 - r0 + 1)
        assert side == int(c1 - c0 + 1)
        assert side >= 3 and side % 2 == 1
        if size is None:
            size = side
        else:
            assert side == size

        patch = np.array(grid[int(r0) : int(r1) + 1, int(c0) : int(c1) + 1], dtype=int)
        assert np.all(patch != BLACK)

        centers = {_edge_center(side_idx, side): side_idx for side_idx in range(4)}
        interfaces: dict[int, int] = {}
        for rr in range(side):
            for cc in range(side):
                color = int(patch[rr, cc])
                if (rr, cc) in centers:
                    if color != GREY:
                        assert color in INTERFACE_COLORS
                        assert color not in interfaces
                        interfaces[color] = centers[(rr, cc)]
                else:
                    assert color == GREY
        assert 1 <= len(interfaces) <= 2
        pieces.append(
            {
                "bbox": (int(r0), int(c0), int(r1), int(c1)),
                "patch": patch,
                "interfaces": interfaces,
            }
        )

    assert size is not None
    return pieces, int(size)


def _build_chain(pieces: list[dict], size: int) -> tuple[list[int], dict[int, int], dict[int, tuple[int, int]]]:
    color_to_pieces: dict[int, list[int]] = {}
    for idx, piece in enumerate(pieces):
        for color in piece["interfaces"]:
            color_to_pieces.setdefault(int(color), []).append(idx)

    for piece_ids in color_to_pieces.values():
        assert len(piece_ids) == 2
        assert piece_ids[0] != piece_ids[1]

    degrees = [len(piece["interfaces"]) for piece in pieces]
    assert degrees.count(1) == 2
    assert all(degree in (1, 2) for degree in degrees)
    assert sum(degrees) == 2 * (len(pieces) - 1)

    endpoints = [idx for idx, degree in enumerate(degrees) if degree == 1]
    start = min(endpoints, key=lambda idx: pieces[idx]["bbox"][:2])

    order = [start]
    rotations = {start: 0}
    positions = {start: (0, 0)}
    visited = {start}
    current = start
    incoming_color = None

    while True:
        colors = list(pieces[current]["interfaces"].keys())
        next_colors = colors if incoming_color is None else [color for color in colors if color != incoming_color]
        if not next_colors:
            break
        assert len(next_colors) == 1
        link_color = int(next_colors[0])

        current_side = _rotate_side(pieces[current]["interfaces"][link_color], rotations[current])
        neighbors = color_to_pieces[link_color]
        nxt = neighbors[0] if neighbors[1] == current else neighbors[1]
        assert nxt not in visited

        desired_side = _opposite(current_side)
        rotations[nxt] = _rotation_to_side(pieces[nxt]["interfaces"][link_color], desired_side)
        dr, dc = DIRS[current_side]
        positions[nxt] = (positions[current][0] + dr * size, positions[current][1] + dc * size)

        visited.add(nxt)
        order.append(nxt)
        current = nxt
        incoming_color = link_color

    assert len(order) == len(pieces)
    assert len({positions[idx] for idx in order}) == len(order)
    return order, rotations, positions


def _render_output(pieces: list[dict], order: list[int], rotations: dict[int, int], positions: dict[int, tuple[int, int]], size: int) -> np.ndarray:
    min_r = min(positions[idx][0] for idx in order)
    min_c = min(positions[idx][1] for idx in order)
    max_r = max(positions[idx][0] for idx in order) + size
    max_c = max(positions[idx][1] for idx in order) + size

    out = np.full((max_r - min_r, max_c - min_c), BLACK, dtype=int)
    for idx in order:
        patch = np.rot90(pieces[idx]["patch"], rotations[idx]).copy()
        patch[patch != GREY] = GREY
        r = positions[idx][0] - min_r
        c = positions[idx][1] - min_c
        region = out[r : r + size, c : c + size]
        assert not np.any(region != BLACK)
        out[r : r + size, c : c + size] = patch

    return out


def main(grid: np.ndarray) -> np.ndarray:
    """Assemble scattered square blocks into one grey chain by matching equal-colored edge connectors.

    Input structure:
    - The background is black.
    - Several same-sized square blocks are scattered in the grid; each block is a solid grey square.
    - At the middle cell of up to two edges per block, a colored connector marker may replace the grey cell.
    - Every connector color appears exactly twice across different blocks, and exactly two blocks have only one connector.

    Rule:
    - Treat equal-colored connectors as adjacency instructions: the two blocks carrying that color must become neighbors in one chain.
    - Start from the one-connector endpoint whose input position is farther toward the top-left, and keep this first block in its original orientation.
    - For each following block, rotate it so the shared connector side faces the opposite side of the previous block's connector, then place the squares flush with no gap.
    - After assembly, all connector cells become grey and the output is tightly cropped around the whole chain.

    Key insight:
    - The global color pairings determine both the chain order and which side of each rotated square must face its predecessor, while opposite vs adjacent connector sides decide straight segments versus turns.
    """
    g = np.asarray(grid, dtype=int)
    assert g.ndim == 2

    pieces, size = _extract_pieces(g)
    order, rotations, positions = _build_chain(pieces, size)
    return _render_output(pieces, order, rotations, positions, size)


def _candidate_dirs(path: list[tuple[int, int]], prev_dir: int | None) -> list[int]:
    occupied = set(path)
    cur_r, cur_c = path[-1]
    candidates: list[int] = []

    for side, (dr, dc) in enumerate(DIRS):
        nr = cur_r + dr
        nc = cur_c + dc
        nxt = (nr, nc)
        if nxt in occupied:
            continue
        neighbors = 0
        bad = False
        for adr, adc in DIRS:
            adj = (nr + adr, nc + adc)
            if adj in occupied:
                neighbors += 1
                if adj != path[-1]:
                    bad = True
                    break
        if bad or neighbors != 1:
            continue
        candidates.append(side)

    if prev_dir is None or len(candidates) <= 1:
        random.shuffle(candidates)
        return candidates

    straight = [side for side in candidates if side == prev_dir]
    turns = [side for side in candidates if side != prev_dir]
    ordered: list[int] = []
    if straight and turns:
        groups = [straight[:], turns[:]]
        random.shuffle(groups)
        for group in groups:
            random.shuffle(group)
            ordered.extend(group)
        return ordered

    random.shuffle(candidates)
    return candidates


def _sample_path(piece_count: int) -> list[tuple[int, int]] | None:
    path = [(0, 0)]

    def dfs(prev_dir: int | None) -> bool:
        if len(path) == piece_count:
            return True
        for side in _candidate_dirs(path, prev_dir):
            dr, dc = DIRS[side]
            path.append((path[-1][0] + dr, path[-1][1] + dc))
            if dfs(side):
                return True
            path.pop()
        return False

    return path[:] if dfs(None) else None


def _path_dirs(path: list[tuple[int, int]]) -> list[int]:
    directions: list[int] = []
    for (r0, c0), (r1, c1) in zip(path, path[1:]):
        directions.append(DIR_TO_INDEX[(r1 - r0, c1 - c0)])
    return directions


def _piece_sides_from_path(path: list[tuple[int, int]]) -> list[list[int]]:
    directions = _path_dirs(path)
    sides: list[list[int]] = []
    for idx in range(len(path)):
        piece_sides: list[int] = []
        if idx > 0:
            piece_sides.append(_opposite(directions[idx - 1]))
        if idx < len(path) - 1:
            piece_sides.append(directions[idx])
        sides.append(piece_sides)
    return sides


def _make_patch(size: int, sides: list[int], colors: list[int]) -> np.ndarray:
    patch = np.full((size, size), GREY, dtype=int)
    for side, color in zip(sides, colors):
        rr, cc = _edge_center(side, size)
        patch[rr, cc] = int(color)
    return patch


def _render_expected(final_patches: list[np.ndarray], path: list[tuple[int, int]], size: int) -> np.ndarray:
    min_r = min(r for r, _ in path)
    min_c = min(c for _, c in path)
    max_r = max(r for r, _ in path) + 1
    max_c = max(c for _, c in path) + 1
    out = np.full(((max_r - min_r) * size, (max_c - min_c) * size), BLACK, dtype=int)

    for patch, (pr, pc) in zip(final_patches, path):
        rr = (pr - min_r) * size
        cc = (pc - min_c) * size
        out[rr : rr + size, cc : cc + size] = patch

    return out


def _slot_bounds(piece_count: int, size: int) -> tuple[int, int]:
    options: list[tuple[int, int]] = []
    for rows in range(1, piece_count + 1):
        cols = (piece_count + rows - 1) // rows
        min_h = rows * size + (rows - 1)
        min_w = cols * size + (cols - 1)
        if min_h <= 30 and min_w <= 30:
            options.append((min_h, min_w))
            options.append((min_w, min_h))
    assert options
    return random.choice(options)


def _place_input_patches(patches: list[np.ndarray], size: int) -> np.ndarray | None:
    count = len(patches)
    min_h, min_w = _slot_bounds(count, size)
    grid_h = random.randint(min_h, 30)
    grid_w = random.randint(min_w, 30)

    placements: list[tuple[int, int] | None] = [None] * count
    occupied = np.zeros((grid_h, grid_w), dtype=bool)
    order = list(range(count))
    random.shuffle(order)

    for idx in order:
        placed = False
        for _ in range(MAX_PLACEMENT_TRIES_PER_PIECE):
            r = random.randint(0, grid_h - size)
            c = random.randint(0, grid_w - size)
            r0 = max(0, r - 1)
            c0 = max(0, c - 1)
            r1 = min(grid_h, r + size + 1)
            c1 = min(grid_w, c + size + 1)
            if np.any(occupied[r0:r1, c0:c1]):
                continue
            occupied[r0:r1, c0:c1] = True
            placements[idx] = (r, c)
            placed = True
            break
        if not placed:
            return None

    assert placements[0] is not None and placements[-1] is not None
    if not (placements[0] < placements[-1]):
        return None

    grid = np.full((grid_h, grid_w), BLACK, dtype=int)
    for idx, patch in enumerate(patches):
        r, c = placements[idx]
        grid[r : r + size, c : c + size] = patch
    return grid


def _sample_instance() -> tuple[np.ndarray, np.ndarray] | None:
    piece_count = random.randint(2, len(INTERFACE_COLORS) + 1)
    size = random.choice([3, 5])

    path = None
    for _ in range(MAX_PATH_ATTEMPTS):
        candidate = _sample_path(piece_count)
        if candidate is not None:
            path = candidate
            break
    if path is None:
        return None

    piece_sides = _piece_sides_from_path(path)
    edge_colors = random.sample(INTERFACE_COLORS, piece_count - 1)

    final_patches: list[np.ndarray] = []
    input_patches: list[np.ndarray] = []
    for idx, sides in enumerate(piece_sides):
        colors: list[int] = []
        if idx > 0:
            colors.append(edge_colors[idx - 1])
        if idx < piece_count - 1:
            colors.append(edge_colors[idx])
        patch = _make_patch(size, sides, colors)
        final_patches.append(np.where(patch == GREY, GREY, GREY))
        rot = 0 if idx == 0 else random.randint(0, 3)
        input_patches.append(np.rot90(patch, rot))

    inp = None
    for _ in range(MAX_PLACEMENT_ATTEMPTS):
        candidate = _place_input_patches(input_patches, size)
        if candidate is not None:
            inp = candidate
            break
    if inp is None:
        return None

    expected = _render_expected(final_patches, path, size)
    return inp, expected


def generate_input() -> np.ndarray:
    for _ in range(MAX_GENERATION_ATTEMPTS):
        sample = _sample_instance()
        if sample is None:
            continue
        inp, expected = sample
        try:
            actual = main(inp)
        except AssertionError:
            continue
        if not np.array_equal(actual, expected):
            continue
        return inp
    raise RuntimeError("failed to generate valid input")
