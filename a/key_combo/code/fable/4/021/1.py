from __future__ import annotations

from collections import deque
import random

import numpy as np


BLACK = 0
YELLOW = 4
GREY = 5
ERASE = 6
TEAL = 8

PASSAGE = 0
WALL = 1
ENDPOINT = 2

GEN_ATTEMPTS = 400
SPRITE_ATTEMPTS = 200
MACRO_ATTEMPTS = 400


def _neighbors4(r: int, c: int) -> tuple[tuple[int, int], ...]:
    return ((r - 1, c), (r + 1, c), (r, c - 1), (r, c + 1))


def _distances(passable: np.ndarray, starts: list[tuple[int, int]]) -> np.ndarray:
    h, w = passable.shape
    dist = np.full((h, w), -1, dtype=int)
    q: deque[tuple[int, int]] = deque()
    for r, c in starts:
        rr = int(r)
        cc = int(c)
        assert passable[rr, cc]
        if dist[rr, cc] >= 0:
            continue
        dist[rr, cc] = 0
        q.append((rr, cc))

    while q:
        r, c = q.popleft()
        nd = int(dist[r, c]) + 1
        for nr, nc in _neighbors4(r, c):
            if 0 <= nr < h and 0 <= nc < w and passable[nr, nc] and dist[nr, nc] < 0:
                dist[nr, nc] = nd
                q.append((nr, nc))
    return dist


def _shortest_route_union(passable: np.ndarray, start: tuple[int, int], goal: tuple[int, int]) -> tuple[np.ndarray, int]:
    dist0 = _distances(passable, [start])
    dist1 = _distances(passable, [goal])
    shortest = int(dist0[goal])
    assert shortest >= 0, "endpoint blocks must be connected"
    route = (dist0 >= 0) & (dist1 >= 0) & (dist0 + dist1 == shortest)
    assert np.any(route)
    return route, shortest


def _block_origin(br: int, bc: int, block_size: int) -> tuple[int, int]:
    return br * (block_size + 1), bc * (block_size + 1)


def _separator_indices(count: int, block_size: int) -> list[int]:
    return [block_size + i * (block_size + 1) for i in range(count - 1)]


def _decode_for_block_size(grid: np.ndarray, block_size: int) -> np.ndarray:
    h, w = grid.shape
    if (h + 1) % (block_size + 1) != 0 or (w + 1) % (block_size + 1) != 0:
        raise AssertionError("shape does not fit a regular block lattice")

    rows = (h + 1) // (block_size + 1)
    cols = (w + 1) // (block_size + 1)
    if rows < 1 or cols < 1 or rows * cols < 2:
        raise AssertionError("macro grid must contain at least two blocks")

    allowed = {BLACK, GREY, ERASE, TEAL}
    if not set(map(int, np.unique(grid))).issubset(allowed):
        raise AssertionError("unexpected colors in the encoded grid")

    for r in _separator_indices(rows, block_size):
        if not set(map(int, grid[r, :])).issubset({BLACK, ERASE}):
            raise AssertionError("separator rows must be black aside from erasures")
    for c in _separator_indices(cols, block_size):
        if not set(map(int, grid[:, c])).issubset({BLACK, ERASE}):
            raise AssertionError("separator columns must be black aside from erasures")

    blocks: list[list[np.ndarray]] = []
    sprite_mask = np.zeros((block_size, block_size), dtype=bool)
    for br in range(rows):
        row_blocks = []
        for bc in range(cols):
            r0, c0 = _block_origin(br, bc, block_size)
            block = grid[r0 : r0 + block_size, c0 : c0 + block_size]
            row_blocks.append(block)
            sprite_mask |= (block == GREY) | (block == TEAL)
        blocks.append(row_blocks)

    if not sprite_mask.any() or np.all(sprite_mask):
        raise AssertionError("the repeated sprite must reveal both fg and bg cells")

    macro = np.full((rows, cols), -1, dtype=int)
    endpoints: list[tuple[int, int]] = []
    for br in range(rows):
        for bc in range(cols):
            block = blocks[br][bc]
            visible = block != ERASE

            if np.any(block[visible & ~sprite_mask] != BLACK):
                raise AssertionError("non-sprite cells may only contain black or erasures")

            seen = block[visible & sprite_mask]
            if seen.size == 0:
                raise AssertionError("every block must expose at least one sprite position")
            values = set(map(int, seen))

            if values == {BLACK}:
                macro[br, bc] = PASSAGE
            elif values == {GREY}:
                macro[br, bc] = WALL
            elif values == {TEAL}:
                macro[br, bc] = ENDPOINT
                endpoints.append((br, bc))
            else:
                raise AssertionError("each block must match blank, wall-sprite, or endpoint-sprite")

    if len(endpoints) != 2:
        raise AssertionError("expected exactly two endpoint blocks")

    route, _ = _shortest_route_union(macro != WALL, endpoints[0], endpoints[1])
    out = np.full((rows, cols), BLACK, dtype=int)
    out[macro == WALL] = GREY
    out[route & (macro == PASSAGE)] = YELLOW
    for r, c in endpoints:
        out[r, c] = TEAL
    return out


def main(grid: np.ndarray) -> np.ndarray:
    """
    Decode a sprite-coded block grid into a macro maze and mark the union of all shortest routes between the two endpoint blocks.

    Input structure:
    - The grid is partitioned by black separator rows and columns into equal-size square blocks.
    - Each block is either an all-black blank block, a repeated black-and-grey sprite, or the same sprite recolored from grey to teal for exactly two endpoint blocks.
    - A few magenta erasure cells may appear inside any block and must be ignored when reading the block state.

    Rule:
    - Infer the block size and the repeated sprite mask, treating grey and teal pixels as the sprite foreground and ignoring magenta erasures.
    - Decode each block into one macro cell: grey sprite blocks are walls, all-black blocks are passable corridor cells, and teal sprite blocks are the two passable endpoints.
    - On that macro maze, find every cell lying on some shortest 4-connected path between the two endpoints.
    - Output the downscaled macro grid with walls colored grey, endpoint cells colored teal, shortest-route corridor cells colored yellow, and all other corridor cells left black.

    Key insight:
    - The visible sprite is not the answer object itself; it is a block-level code for wall cells in an unseen maze, while the magenta pixels are wildcard erasures that must be ignored to decode each block correctly.
    """
    g = np.asarray(grid, dtype=int)
    assert g.ndim == 2, "expected a 2D grid"

    candidates: list[tuple[tuple[int, int, int], np.ndarray]] = []
    for block_size in range(2, min(g.shape) + 1):
        try:
            out = _decode_for_block_size(g, block_size)
        except AssertionError:
            continue
        score = (block_size, int(out.shape[0] * out.shape[1]), int(np.count_nonzero(out == GREY)))
        candidates.append((score, out))

    assert candidates, "failed to infer a valid block partition"
    best_score = max(score for score, _out in candidates)
    best_outputs = [out for score, out in candidates if score == best_score]
    first = best_outputs[0]
    for other in best_outputs[1:]:
        assert np.array_equal(first, other), "ambiguous block partition produced different outputs"
    return first


def _make_sprite(block_size: int) -> np.ndarray:
    for _ in range(SPRITE_ATTEMPTS):
        density = random.uniform(0.30, 0.58)
        mask = np.random.random((block_size, block_size)) < density
        fg = int(mask.sum())
        area = block_size * block_size
        if not (2 <= fg <= area - 2):
            continue
        if any(row.all() or (~row).all() for row in mask):
            continue
        if any(col.all() or (~col).all() for col in mask.T):
            continue
        if np.array_equal(mask, np.fliplr(mask)):
            continue
        if np.array_equal(mask, np.flipud(mask)):
            continue
        if np.array_equal(mask, np.rot90(mask, 2)):
            continue
        return mask
    raise RuntimeError("failed to sample a distinctive sprite")


def _choose_dimensions() -> tuple[int, int, int]:
    choices: list[tuple[int, int, int]] = []
    for rows in range(4, 8):
        for cols in range(4, 8):
            for block_size in range(3, 6):
                h = rows * block_size + rows - 1
                w = cols * block_size + cols - 1
                if h <= 30 and w <= 30:
                    choices.append((rows, cols, block_size))
    return random.choice(choices)


def _macro_with_endpoints(rows: int, cols: int) -> tuple[np.ndarray, np.ndarray]:
    cells = [(r, c) for r in range(rows) for c in range(cols)]
    want_multi = random.random() < 0.6

    for _ in range(MACRO_ATTEMPTS):
        wall_min = max(3, rows)
        wall_max = min(rows * cols - 4, (rows * cols) // 2 + 2)
        if wall_min > wall_max:
            continue
        wall_count = random.randint(wall_min, wall_max)
        wall_positions = set(random.sample(cells, wall_count))
        passable_cells = [cell for cell in cells if cell not in wall_positions]
        if len(passable_cells) < 4:
            continue

        for _endpoint_try in range(20):
            start, goal = random.sample(passable_cells, 2)
            passable = np.ones((rows, cols), dtype=bool)
            for r, c in wall_positions:
                passable[r, c] = False

            try:
                route, shortest = _shortest_route_union(passable, start, goal)
            except AssertionError:
                continue

            sprite_like = ~passable
            sprite_like[start] = True
            sprite_like[goal] = True
            if not np.all(sprite_like.any(axis=1)):
                continue
            if not np.all(sprite_like.any(axis=0)):
                continue

            route_cells = int(route.sum())
            route_passages = int(np.count_nonzero(route) - 2)
            off_route_passages = int(np.count_nonzero(passable & ~route))
            multiple_shortest = route_cells > shortest + 1

            if shortest < max(rows, cols) - 1:
                continue
            if route_passages < 2:
                continue
            if off_route_passages < 2:
                continue
            if route_cells > int(rows * cols * 0.65):
                continue
            if want_multi and not multiple_shortest:
                continue

            macro = np.full((rows, cols), PASSAGE, dtype=int)
            for r, c in wall_positions:
                macro[r, c] = WALL
            macro[start] = ENDPOINT
            macro[goal] = ENDPOINT
            return macro, route

    raise RuntimeError("failed to sample a valid macro maze")


def _render_expected(macro: np.ndarray, route: np.ndarray) -> np.ndarray:
    out = np.full(macro.shape, BLACK, dtype=int)
    out[macro == WALL] = GREY
    out[route & (macro == PASSAGE)] = YELLOW
    out[macro == ENDPOINT] = TEAL
    return out


def _render_input(macro: np.ndarray, sprite_mask: np.ndarray) -> np.ndarray:
    rows, cols = macro.shape
    block_size = int(sprite_mask.shape[0])
    h = rows * block_size + rows - 1
    w = cols * block_size + cols - 1
    grid = np.full((h, w), BLACK, dtype=int)

    wall_blocks = [(r, c) for r in range(rows) for c in range(cols) if macro[r, c] == WALL]
    passage_blocks = [(r, c) for r in range(rows) for c in range(cols) if macro[r, c] == PASSAGE]
    endpoint_blocks = [(r, c) for r in range(rows) for c in range(cols) if macro[r, c] == ENDPOINT]

    anchor_wall = random.choice(wall_blocks)
    noisy_wall = random.choice([cell for cell in wall_blocks if cell != anchor_wall]) if len(wall_blocks) > 1 else None
    noisy_passage = random.choice(passage_blocks) if passage_blocks else None
    fg_positions = [tuple(map(int, rc)) for rc in np.argwhere(sprite_mask)]

    for br in range(rows):
        for bc in range(cols):
            r0, c0 = _block_origin(br, bc, block_size)
            state = int(macro[br, bc])
            block = np.full((block_size, block_size), BLACK, dtype=int)
            if state == WALL:
                block[sprite_mask] = GREY
            elif state == ENDPOINT:
                block[sprite_mask] = TEAL
            grid[r0 : r0 + block_size, c0 : c0 + block_size] = block

    for br in range(rows):
        for bc in range(cols):
            if (br, bc) == anchor_wall:
                continue

            r0, c0 = _block_origin(br, bc, block_size)
            state = int(macro[br, bc])
            protected = {random.choice(fg_positions)}

            candidates = [
                (rr, cc)
                for rr in range(block_size)
                for cc in range(block_size)
                if (rr, cc) not in protected
            ]
            if not candidates:
                continue

            max_erases = min(len(candidates), max(1, block_size // 2 + 1))
            force_noise = (br, bc) == noisy_wall or (br, bc) == noisy_passage
            if force_noise:
                erase_count = random.randint(1, max_erases)
            elif random.random() < 0.55:
                erase_count = random.randint(0, max_erases)
            else:
                erase_count = 0

            if erase_count == 0 and state == ENDPOINT and random.random() < 0.5:
                erase_count = 1

            for rr, cc in random.sample(candidates, erase_count):
                grid[r0 + rr, c0 + cc] = ERASE

    return grid


def _gen_attempt() -> np.ndarray | None:
    rows, cols, block_size = _choose_dimensions()
    sprite_mask = _make_sprite(block_size)
    macro, route = _macro_with_endpoints(rows, cols)
    inp = _render_input(macro, sprite_mask)
    expected = _render_expected(macro, route)

    try:
        solved = main(inp)
    except AssertionError:
        return None
    if not np.array_equal(solved, expected):
        return None
    return inp


def generate_input() -> np.ndarray:
    for _ in range(GEN_ATTEMPTS):
        grid = _gen_attempt()
        if grid is not None:
            return grid
    raise RuntimeError("failed to generate a valid input for the block-coded maze task")
