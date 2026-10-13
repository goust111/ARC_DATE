from __future__ import annotations

import random
from collections import deque

import numpy as np


BACKGROUND = 0
BLUE = 1
GREY = 5

INPUT_SIZE = 5
BLOCK_SIZE = 3
STAMP_SIZE = INPUT_SIZE * BLOCK_SIZE

# The prompt's 15x15 canvas description is internally inconsistent with a 5x5
# mask plus outward 15x15 block stamps. Use the smallest self-consistent canvas:
# a 15x15 central lattice with one 3-cell stamp margin on each side.
CANVAS_SIZE = STAMP_SIZE + 2 * BLOCK_SIZE
INNER_OFFSET = BLOCK_SIZE

DIR_ORDER = (
    ("up", -1, 0),
    ("right", 0, 1),
    ("down", 1, 0),
    ("left", 0, -1),
)

STAMP_TOP_LEFT = {
    "up": (0, INNER_OFFSET),
    "right": (INNER_OFFSET, 2 * BLOCK_SIZE),
    "down": (2 * BLOCK_SIZE, INNER_OFFSET),
    "left": (INNER_OFFSET, 0),
}

NEIGHBORS8 = tuple(
    (dr, dc)
    for dr in (-1, 0, 1)
    for dc in (-1, 0, 1)
    if not (dr == 0 and dc == 0)
)

MAX_GENERATION_ATTEMPTS = 400
_GEN_COUNTER = 0


def _in_bounds(r: int, c: int, shape: tuple[int, int]) -> bool:
    return 0 <= r < shape[0] and 0 <= c < shape[1]


def _is_connected8(mask: np.ndarray) -> bool:
    coords = np.argwhere(mask)
    if len(coords) == 0:
        return False

    start = tuple(map(int, coords[0]))
    queue = deque([start])
    seen = {start}
    while queue:
        r, c = queue.popleft()
        for dr, dc in NEIGHBORS8:
            nr, nc = r + dr, c + dc
            if _in_bounds(nr, nc, mask.shape) and mask[nr, nc] and (nr, nc) not in seen:
                seen.add((nr, nc))
                queue.append((nr, nc))
    return len(seen) == len(coords)


def _blue_mask(grid: np.ndarray) -> np.ndarray:
    return np.asarray(grid == BLUE, dtype=bool)


def _edge_directions(mask: np.ndarray, r: int, c: int) -> list[str]:
    dirs = []
    for name, dr, dc in DIR_ORDER:
        nr, nc = r + dr, c + dc
        if not _in_bounds(nr, nc, mask.shape) or not mask[nr, nc]:
            dirs.append(name)
    return dirs


def _scaled_blue_stamp(mask: np.ndarray) -> np.ndarray:
    return np.kron(mask.astype(int), np.ones((BLOCK_SIZE, BLOCK_SIZE), dtype=int)).astype(int) * BLUE


def _render_base(grid: np.ndarray) -> np.ndarray:
    out = np.zeros((CANVAS_SIZE, CANVAS_SIZE), dtype=int)
    for r in range(INPUT_SIZE):
        for c in range(INPUT_SIZE):
            value = int(grid[r, c])
            if value == BACKGROUND:
                continue
            top = INNER_OFFSET + BLOCK_SIZE * r
            left = INNER_OFFSET + BLOCK_SIZE * c
            out[top : top + BLOCK_SIZE, left : left + BLOCK_SIZE] = value
    return out


def _solve(grid: np.ndarray, *, obey_blockers: bool) -> np.ndarray:
    g = np.asarray(grid, dtype=int)
    assert g.shape == (INPUT_SIZE, INPUT_SIZE), "expected a 5x5 input grid"
    assert set(map(int, np.unique(g))).issubset({BACKGROUND, BLUE, GREY}), "unexpected colors"

    blue = _blue_mask(g)
    blue_count = int(blue.sum())
    grey_count = int(np.count_nonzero(g == GREY))
    assert 3 <= blue_count <= 10, "expected 3-10 blue pixels"
    assert 1 <= grey_count <= 5, "expected 1-5 grey blockers"
    assert _is_connected8(blue), "blue pixels must be 8-connected"

    stamp = _scaled_blue_stamp(blue)
    stamp_mask = stamp == BLUE
    out = _render_base(g)

    for r, c in map(tuple, np.argwhere(blue)):
        r = int(r)
        c = int(c)
        for direction in _edge_directions(blue, r, c):
            dr = dc = 0
            for name, ddr, ddc in DIR_ORDER:
                if name == direction:
                    dr, dc = ddr, ddc
                    break
            nr, nc = r + dr, c + dc
            if obey_blockers and _in_bounds(nr, nc, g.shape) and int(g[nr, nc]) == GREY:
                continue

            top, left = STAMP_TOP_LEFT[direction]
            region = out[top : top + STAMP_SIZE, left : left + STAMP_SIZE]
            region[stamp_mask] = BLUE

    return out


def _input_features(grid: np.ndarray) -> dict[str, object]:
    g = np.asarray(grid, dtype=int)
    blue = _blue_mask(g)
    blockers = {(int(r), int(c)) for r, c in np.argwhere(g == GREY)}

    edge_info: list[dict[str, object]] = []
    blocked_directions = 0
    allowed_directions = 0
    mixed_edge_exists = False
    fully_free_exists = False

    for r, c in map(tuple, np.argwhere(blue)):
        r = int(r)
        c = int(c)
        dirs = _edge_directions(blue, r, c)
        if not dirs:
            continue

        blocked = []
        allowed = []
        for name, dr, dc in DIR_ORDER:
            if name not in dirs:
                continue
            nr, nc = r + dr, c + dc
            if _in_bounds(nr, nc, g.shape) and (nr, nc) in blockers:
                blocked.append(name)
                blocked_directions += 1
            else:
                allowed.append(name)
                allowed_directions += 1

        mixed_edge_exists |= bool(blocked and allowed)
        fully_free_exists |= not blocked
        edge_info.append(
            {
                "cell": (r, c),
                "dirs": dirs,
                "blocked": blocked,
                "allowed": allowed,
            }
        )

    return {
        "edge_info": edge_info,
        "blocked_directions": blocked_directions,
        "allowed_directions": allowed_directions,
        "mixed_edge_exists": mixed_edge_exists,
        "fully_free_exists": fully_free_exists,
    }


def main(grid: np.ndarray) -> np.ndarray:
    """Expand a 5x5 blue mask into solid blocks, then stamp it outward only where adjacent grey blockers do not suppress the edge direction.

    Input structure:
    - The input is a 5x5 black-background grid containing blue foreground pixels and grey blocker pixels.
    - The blue pixels form one 8-connected shape of 3-6 cells.
    - Grey pixels occupy nearby cells around the blue shape and can sit on cardinal or diagonal neighbors.

    Rule:
    - Enlarge every non-black input cell into a 3x3 solid block inside a centered block lattice.
    - For each blue cell, every missing blue neighbor among the four cardinal directions defines an outward edge direction.
    - If the adjacent input cell in that outward direction is grey, suppress the stamp on that side; otherwise place a blue-only block-scaled copy of the original 5x5 blue mask in the matching outer band.
    - Process blue edge cells in row-major order and directions in up, right, down, left order; later blue stamps overwrite earlier paint.

    Key insight:
    - Grey blockers do not change block size; they selectively disable only those outward edge stamps whose cardinal direction is immediately blocked, creating asymmetric expansions from the same underlying blue mask.
    """
    return _solve(grid, obey_blockers=True)


def _adjacent_candidates(shape: set[tuple[int, int]]) -> set[tuple[int, int]]:
    candidates = set()
    for r, c in shape:
        for dr, dc in NEIGHBORS8:
            nr, nc = r + dr, c + dc
            if 0 <= nr < INPUT_SIZE and 0 <= nc < INPUT_SIZE and (nr, nc) not in shape:
                candidates.add((nr, nc))
    return candidates


def _grow_shape(count: int | None = None) -> set[tuple[int, int]] | None:
    if count is None:
        count = random.randint(3, 10)
    start = (random.randrange(INPUT_SIZE), random.randrange(INPUT_SIZE))
    shape = {start}

    while len(shape) < count:
        frontier = set()
        for r, c in shape:
            for dr, dc in NEIGHBORS8:
                nr, nc = r + dr, c + dc
                if 0 <= nr < INPUT_SIZE and 0 <= nc < INPUT_SIZE and (nr, nc) not in shape:
                    frontier.add((nr, nc))
        if not frontier:
            return None
        shape.add(random.choice(tuple(frontier)))

    mask = np.zeros((INPUT_SIZE, INPUT_SIZE), dtype=bool)
    for r, c in shape:
        mask[r, c] = True
    if not _is_connected8(mask):
        return None
    return shape


def _apply_random_transform(grid: np.ndarray) -> np.ndarray:
    g = np.asarray(grid, dtype=int)
    k = random.randrange(4)
    g = np.rot90(g, k)
    if random.random() < 0.5:
        g = np.fliplr(g)
    if random.random() < 0.5:
        g = np.flipud(g)
    return np.asarray(g, dtype=int)


def _attempt_generate(target_blue_count: int | None = None, target_grey_count: int | None = None) -> np.ndarray | None:
    shape = _grow_shape(target_blue_count)
    if shape is None:
        return None

    blue_mask = np.zeros((INPUT_SIZE, INPUT_SIZE), dtype=bool)
    for r, c in shape:
        blue_mask[r, c] = True

    edge_cells = []
    mixed_edge_candidates = []
    for r, c in sorted(shape):
        dirs = _edge_directions(blue_mask, r, c)
        if not dirs:
            continue
        edge_cells.append((r, c, dirs))
        in_bounds_dirs = []
        for name, dr, dc in DIR_ORDER:
            if name in dirs:
                nr, nc = r + dr, c + dc
                if _in_bounds(nr, nc, blue_mask.shape) and not blue_mask[nr, nc]:
                    in_bounds_dirs.append((name, nr, nc))
        if len(dirs) >= 2 and in_bounds_dirs:
            mixed_edge_candidates.append((r, c, dirs, in_bounds_dirs))

    if not mixed_edge_candidates:
        return None

    _block_r, _block_c, _dirs, in_bounds_dirs = random.choice(mixed_edge_candidates)
    _direction_name, gr, gc = random.choice(in_bounds_dirs)
    grey_cells = {(gr, gc)}

    # Keep one edge pixel entirely free of grey on all of its outward sides.
    free_candidates = []
    for r, c, dirs in edge_cells:
        outward_neighbors = []
        blocked = False
        for name, dr, dc in DIR_ORDER:
            if name not in dirs:
                continue
            nr, nc = r + dr, c + dc
            if _in_bounds(nr, nc, blue_mask.shape):
                outward_neighbors.append((nr, nc))
            if (nr, nc) in grey_cells:
                blocked = True
                break
        if not blocked:
            free_candidates.append((r, c, outward_neighbors))
    if not free_candidates:
        return None

    adjacent = list(_adjacent_candidates(shape) - grey_cells)
    random.shuffle(adjacent)
    if target_grey_count is None:
        target_grey_count = random.randint(1, 5)
    for cell in adjacent:
        if len(grey_cells) >= target_grey_count:
            break
        if any(cell in neighbors for _r, _c, neighbors in free_candidates):
            continue
        grey_cells.add(cell)

    if len(grey_cells) < target_grey_count:
        extras = [
            (r, c)
            for r in range(INPUT_SIZE)
            for c in range(INPUT_SIZE)
            if (r, c) not in shape and (r, c) not in grey_cells
        ]
        random.shuffle(extras)
        for cell in extras:
            grey_cells.add(cell)
            if len(grey_cells) >= target_grey_count:
                break

    grid = np.zeros((INPUT_SIZE, INPUT_SIZE), dtype=int)
    for r, c in shape:
        grid[r, c] = BLUE
    for r, c in grey_cells:
        grid[r, c] = GREY

    grid = _apply_random_transform(grid)

    blue = _blue_mask(grid)
    if not _is_connected8(blue):
        return None

    grey_coords = np.argwhere(grid == GREY)
    if not (1 <= len(grey_coords) <= 5):
        return None
    if not any(
        _in_bounds(int(r) + dr, int(c) + dc, grid.shape) and int(grid[int(r) + dr, int(c) + dc]) == BLUE
        for r, c in map(tuple, grey_coords)
        for dr, dc in NEIGHBORS8
    ):
        return None

    features = _input_features(grid)
    if not features["mixed_edge_exists"]:
        return None
    if not features["fully_free_exists"]:
        return None
    if int(features["blocked_directions"]) < 1 or int(features["allowed_directions"]) < 1:
        return None

    out = main(grid)
    unblocked_out = _solve(grid, obey_blockers=False)
    central_only = _render_base(grid)

    if np.array_equal(out, unblocked_out):
        return None
    if np.array_equal(out, central_only):
        return None

    # Require the blocker logic to matter in more than one cell so shortcut
    # strategies like "always remove one side" fail on random samples.
    diff = np.count_nonzero(out != unblocked_out)
    if diff < BLOCK_SIZE * BLOCK_SIZE:
        return None

    return grid


def generate_input() -> np.ndarray:
    global _GEN_COUNTER
    count_schedule = [(blue, grey) for blue in range(3, 11) for grey in range(1, 6)]
    target_blue_count, target_grey_count = count_schedule[_GEN_COUNTER % len(count_schedule)]
    _GEN_COUNTER += 1

    for _ in range(MAX_GENERATION_ATTEMPTS):
        grid = _attempt_generate(target_blue_count, target_grey_count)
        if grid is not None:
            return grid

    for _ in range(MAX_GENERATION_ATTEMPTS):
        grid = _attempt_generate()
        if grid is not None:
            return grid
    raise RuntimeError("failed to generate a valid blocker-suppressed stamp input")
