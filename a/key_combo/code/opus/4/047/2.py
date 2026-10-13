from __future__ import annotations

import random

import numpy as np


BACKGROUND = 0
SPRITE_COLOR = 1
WALL_COLOR = 5

GRID_HEIGHT_RANGE = (13, 19)
OBJECT_COUNT_RANGE = (4, 6)
WIDTH_LIMIT = 30
GENERATION_ATTEMPTS = 500


def _components_mask(mask: np.ndarray, connectivity: int = 4) -> list[np.ndarray]:
    """Small local clone of arc_utils.components_mask using (r, c) coordinates."""
    h, w = mask.shape
    seen = np.zeros((h, w), dtype=bool)
    if connectivity == 4:
        steps = [(-1, 0), (1, 0), (0, -1), (0, 1)]
    else:
        steps = [(dr, dc) for dr in (-1, 0, 1) for dc in (-1, 0, 1) if (dr, dc) != (0, 0)]

    comps: list[np.ndarray] = []
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
                for dr, dc in steps:
                    nr = rr + dr
                    nc = cc + dc
                    if 0 <= nr < h and 0 <= nc < w and mask[nr, nc] and not seen[nr, nc]:
                        seen[nr, nc] = True
                        stack.append((nr, nc))
            comps.append(np.array(coords, dtype=int))
    return comps


def _touches_8(a: np.ndarray, b: np.ndarray) -> bool:
    for ra, ca in a:
        if np.any((np.abs(b[:, 0] - int(ra)) <= 1) & (np.abs(b[:, 1] - int(ca)) <= 1)):
            return True
    return False


def _sprite_info(coords: np.ndarray) -> dict:
    left_col = int(coords[:, 1].min())
    right_col = int(coords[:, 1].max())
    left_rows = frozenset(int(r) for r in coords[coords[:, 1] == left_col, 0])
    right_rows = frozenset(int(r) for r in coords[coords[:, 1] == right_col, 0])
    return {
        "coords": np.array(coords, dtype=int),
        "left_col": left_col,
        "right_col": right_col,
        "left_rows": left_rows,
        "right_rows": right_rows,
    }


def _extract_sprites(grid: np.ndarray) -> list[dict]:
    g = np.asarray(grid, dtype=int)
    assert g.ndim == 2, "expected a 2D grid"
    assert np.all(g[:, 0] == WALL_COLOR), "left wall must fill the first column"
    assert set(map(int, np.unique(g))).issubset({BACKGROUND, SPRITE_COLOR, WALL_COLOR}), "unexpected colors"
    comps = _components_mask(g == SPRITE_COLOR, connectivity=4)
    assert 4 <= len(comps) <= 6, "expected 4 to 6 blue sprites"
    infos = [_sprite_info(comp) for comp in comps]
    for info in infos:
        area = int(info["coords"].shape[0])
        assert 3 <= area <= 6, "sprite area must stay within the intended range"
        assert int(info["left_col"]) > 0, "sprites may not touch the wall"
    for i in range(len(infos)):
        for j in range(i + 1, len(infos)):
            assert not _touches_8(infos[i]["coords"], infos[j]["coords"]), "sprites must stay separated"
    return infos


def _build_chain(sprites: list[dict]) -> list[int]:
    n = len(sprites)
    preds = [[] for _ in range(n)]
    succs = [[] for _ in range(n)]
    for i in range(n):
        for j in range(n):
            if i == j:
                continue
            if sprites[i]["right_rows"] & sprites[j]["left_rows"]:
                succs[i].append(j)
                preds[j].append(i)

    heads = [i for i in range(n) if not preds[i]]
    assert len(heads) == 1, "chain head must be unique"
    assert all(len(preds[i]) <= 1 for i in range(n)), "each sprite must have at most one predecessor"

    head = heads[0]
    order = [head]
    used = {head}
    while len(order) < n:
        current = order[-1]
        choices = [j for j in succs[current] if j not in used]
        assert len(choices) == 1, "each step in the chain must have exactly one next sprite"
        nxt = choices[0]
        used.add(nxt)
        order.append(nxt)

    assert len(used) == n, "the chain must cover all sprites"
    return order


def _compress_in_order(sprites: list[dict], order: list[int], shape: tuple[int, int]) -> list[np.ndarray]:
    occupied = np.zeros(shape, dtype=bool)
    occupied[:, 0] = True
    placed: list[np.ndarray] = []

    for idx in order:
        coords = sprites[idx]["coords"]
        shift = 0
        while True:
            candidate = coords - np.array([0, shift + 1], dtype=int)
            if int(candidate[:, 1].min()) < 1:
                break
            if np.any(occupied[candidate[:, 0], candidate[:, 1]]):
                break
            shift += 1
        final_coords = coords - np.array([0, shift], dtype=int)
        assert not np.any(occupied[final_coords[:, 0], final_coords[:, 1]]), "settled sprites may not overlap"
        occupied[final_coords[:, 0], final_coords[:, 1]] = True
        placed.append(final_coords)
    return placed


def _render(shape: tuple[int, int], placed: list[np.ndarray]) -> np.ndarray:
    out = np.zeros(shape, dtype=int)
    out[:, 0] = WALL_COLOR
    for coords in placed:
        out[coords[:, 0], coords[:, 1]] = SPRITE_COLOR
    return out


def main(grid: np.ndarray) -> np.ndarray:
    """
    Determine a unique edge-overlap chain of blue sprites, then compact them leftward against the gray wall in that chain order.

    Input structure:
    - The background is black.
    - The full leftmost column is a fixed gray wall.
    - Away from the wall are 4 to 6 separated blue 4-connected sprites, each with a well-defined left edge column and right edge column.

    Rule:
    - For each blue sprite, collect the global row indices occupied in its leftmost column and in its rightmost column.
    - Find the unique chain head whose left-edge row set overlaps no other sprite's right-edge row set, then repeatedly follow the unique unmatched sprite whose left-edge row set overlaps the current sprite's right-edge row set.
    - Process sprites in that chain order, sliding each whole sprite horizontally left until one more step would hit the gray wall or an already settled sprite.
    - Erase the original sprite positions and draw the settled sprites in place, preserving each sprite's shape and rows.

    Key insight:
    - The edge-row overlaps are not just a sorting clue: they decide which sprite reaches the wall first and therefore which settled sprite becomes the obstacle for later ones.
    """
    sprites = _extract_sprites(np.asarray(grid, dtype=int))
    order = _build_chain(sprites)
    placed = _compress_in_order(sprites, order, grid.shape)
    out = _render(grid.shape, placed)
    assert int(np.sum(out == SPRITE_COLOR)) == int(np.sum(np.asarray(grid) == SPRITE_COLOR))
    return out


def _row_nodes(height: int, count: int) -> list[int] | None:
    low = 1
    high = height - 2
    for _ in range(200):
        start = random.randint(low, high)
        nodes = [start]
        used = {start}
        while len(nodes) < count + 1:
            cur = nodes[-1]
            candidates = [
                nxt
                for delta in (-3, -2, -1, 1, 2, 3)
                if low <= (nxt := cur + delta) <= high and nxt not in used
            ]
            if not candidates:
                break
            nxt = random.choice(candidates)
            nodes.append(nxt)
            used.add(nxt)
        if len(nodes) == count + 1:
            return nodes
    return None


def _make_sprite_coords(r_left: int, r_right: int, left_col: int) -> np.ndarray:
    cells = {(int(r_left), int(left_col)), (int(r_left), int(left_col + 1)), (int(r_right), int(left_col + 2))}
    step = 1 if r_right >= r_left else -1
    for r in range(int(r_left), int(r_right) + step, step):
        cells.add((int(r), int(left_col + 1)))
    return np.array(sorted(cells), dtype=int)


def _oracle_from_order(shape: tuple[int, int], sprites: list[dict], order: list[int]) -> np.ndarray:
    return _render(shape, _compress_in_order(sprites, order, shape))


def _order_changes_result(shape: tuple[int, int], sprites: list[dict], baseline: np.ndarray, order: list[int]) -> bool:
    try:
        alt = _oracle_from_order(shape, sprites, order)
    except AssertionError:
        return True
    return not np.array_equal(baseline, alt)


def _valid_instance(grid: np.ndarray) -> bool:
    try:
        sprites = _extract_sprites(grid)
        chain_order = _build_chain(sprites)
        expected = _oracle_from_order(grid.shape, sprites, chain_order)
        actual = main(grid)
    except AssertionError:
        return False
    if not np.array_equal(actual, expected):
        return False

    x_order = sorted(range(len(sprites)), key=lambda i: (sprites[i]["left_col"], int(sprites[i]["coords"][:, 0].min())))
    if x_order == chain_order:
        return False
    if not _order_changes_result(grid.shape, sprites, expected, x_order):
        return False

    reverse_order = list(reversed(chain_order))
    if not _order_changes_result(grid.shape, sprites, expected, reverse_order):
        return False

    return True


def _scatter_from_final(final_coords: list[np.ndarray], width: int) -> list[np.ndarray] | None:
    count = len(final_coords)
    raw_coords: list[np.ndarray | None] = [None] * count
    max_shifts = [int(width - 1 - coords[:, 1].max()) for coords in final_coords]
    place_order = list(range(count))
    random.shuffle(place_order)

    def backtrack(pos: int) -> bool:
        if pos == count:
            return True
        idx = place_order[pos]
        choices = list(range(2, max_shifts[idx] + 1))
        random.shuffle(choices)
        for shift in choices:
            candidate = final_coords[idx] + np.array([0, shift], dtype=int)
            if any(_touches_8(candidate, other) for other in raw_coords if other is not None):
                continue
            raw_coords[idx] = candidate
            if backtrack(pos + 1):
                return True
            raw_coords[idx] = None
        return False

    if not backtrack(0):
        return None
    return [coords for coords in raw_coords if coords is not None]


def _attempt_generate() -> np.ndarray | None:
    height = random.randint(*GRID_HEIGHT_RANGE)
    count = random.randint(*OBJECT_COUNT_RANGE)

    nodes = _row_nodes(height, count)
    if nodes is None:
        return None

    proto_left = WIDTH_LIMIT - 4
    proto_coords = [_make_sprite_coords(nodes[i], nodes[i + 1], proto_left) for i in range(count)]
    sprites = [_sprite_info(coords) for coords in proto_coords]
    final_coords = _compress_in_order(sprites, list(range(count)), (height, WIDTH_LIMIT))
    final_max_right = max(int(coords[:, 1].max()) for coords in final_coords)

    width_low = max(18, final_max_right + 3)
    if width_low > WIDTH_LIMIT:
        return None
    widths = list(range(width_low, WIDTH_LIMIT + 1))
    random.shuffle(widths)

    raw_coords = None
    width = None
    for candidate_width in widths:
        scattered = _scatter_from_final(final_coords, candidate_width)
        if scattered is None:
            continue
        raw_coords = scattered
        width = candidate_width
        break
    if raw_coords is None or width is None:
        return None

    grid = np.zeros((height, width), dtype=int)
    grid[:, 0] = WALL_COLOR
    for coords in raw_coords:
        if np.any(grid[coords[:, 0], coords[:, 1]] != BACKGROUND):
            return None
        grid[coords[:, 0], coords[:, 1]] = SPRITE_COLOR

    if not _valid_instance(grid):
        return None
    return grid


def generate_input() -> np.ndarray:
    for _ in range(GENERATION_ATTEMPTS):
        grid = _attempt_generate()
        if grid is not None:
            return grid
    raise RuntimeError("failed to generate a valid edge-chain gravity-compression input")
