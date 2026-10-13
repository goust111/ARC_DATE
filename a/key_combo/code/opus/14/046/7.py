from __future__ import annotations

import numpy as np


BLOCK = 3
SEP_COLOR = 6
SOURCE_COLOR = 7
TRAIL_COLOR = 4
PATTERN_COLORS = np.array([1, 2, 3, 8, 9], dtype=int)
RNG = np.random.default_rng()

GEN_BLOCK_ROWS = (3, 4)
GEN_BLOCK_COLS = (4, 5, 6)
GEN_MAX_ATTEMPTS = 400


def _neighbors4(r: int, c: int):
    yield r - 1, c
    yield r + 1, c
    yield r, c - 1
    yield r, c + 1


def _find_separator_row(grid: np.ndarray) -> int:
    rows = [r for r in range(grid.shape[0]) if np.all(grid[r] == SEP_COLOR)]
    assert len(rows) == 1, "expected exactly one solid magenta separator row"
    return int(rows[0])


def _follow_trail(bottom: np.ndarray) -> tuple[tuple[int, int], tuple[int, int], set[tuple[int, int]]]:
    sources = np.argwhere(bottom == SOURCE_COLOR)
    assert len(sources) == 1, "expected exactly one orange source pixel"
    source = tuple(map(int, sources[0]))

    trail = {tuple(map(int, rc)) for rc in np.argwhere(bottom == TRAIL_COLOR)}
    assert trail, "expected a yellow trail"

    adjacent = [rc for rc in _neighbors4(*source) if rc in trail]
    assert len(adjacent) == 1, "source must touch exactly one trail cell"

    prev = source
    cur = adjacent[0]
    used = {cur}
    while True:
        nxt = [rc for rc in _neighbors4(*cur) if rc != prev and rc in trail]
        assert len(nxt) <= 1, "trail must be an unbranched path"
        if not nxt:
            break
        prev, cur = cur, nxt[0]
        assert cur not in used, "trail cannot loop"
        used.add(cur)

    assert used == trail, "all yellow cells must belong to the same source-to-endpoint trail"
    return source, cur, used


def _block_slice(block_row: int, block_col: int) -> tuple[slice, slice]:
    r0 = int(block_row) * BLOCK
    c0 = int(block_col) * BLOCK
    return slice(r0, r0 + BLOCK), slice(c0, c0 + BLOCK)


def _recover_patch(top: np.ndarray, hole_br: int, hole_bc: int) -> np.ndarray:
    rs, cs = _block_slice(hole_br, hole_bc)
    c0 = cs.start
    c1 = cs.stop - 1
    mirror = top[rs, top.shape[1] - 1 - c1 : top.shape[1] - c0][:, ::-1].copy()
    assert mirror.shape == (BLOCK, BLOCK)
    assert np.any(mirror != 0), "reflected counterpart must not be fully black"
    return mirror


def _inspect_grid(grid: np.ndarray) -> dict:
    g = np.asarray(grid, dtype=int)
    assert g.ndim == 2, "expected a 2D grid"
    sep = _find_separator_row(g)

    top = g[:sep].copy()
    bottom = g[sep + 1 :].copy()
    height, width = top.shape
    assert height >= BLOCK * 2 and height % BLOCK == 0, "top area height must be a positive multiple of 3"
    assert width >= BLOCK * 3 and width % BLOCK == 0, "grid width must be a positive multiple of 3"

    block_rows = height // BLOCK
    block_cols = width // BLOCK
    assert bottom.shape[0] == block_rows, "bottom relative rows must match the number of top block rows"

    source, endpoint, trail = _follow_trail(bottom)
    target_br = endpoint[0]
    target_bc = endpoint[1] // BLOCK
    assert 0 <= target_br < block_rows
    assert 0 <= target_bc < block_cols

    holes: list[tuple[int, int]] = []
    for br in range(block_rows):
        for bc in range(block_cols):
            rs, cs = _block_slice(br, bc)
            if np.all(top[rs, cs] == 0):
                holes.append((br, bc))
    assert len(holes) == 1, "expected exactly one all-black 3x3 missing block"
    hole_br, hole_bc = holes[0]

    patch = _recover_patch(top, hole_br, hole_bc)
    target_rs, target_cs = _block_slice(target_br, target_bc)
    target_block = top[target_rs, target_cs].copy()

    return {
        "grid": g,
        "sep": sep,
        "top": top,
        "bottom": bottom,
        "block_rows": block_rows,
        "block_cols": block_cols,
        "source": source,
        "endpoint": endpoint,
        "trail": trail,
        "hole": (hole_br, hole_bc),
        "target": (target_br, target_bc),
        "patch": patch,
        "target_block": target_block,
    }


def main(grid: np.ndarray) -> np.ndarray:
    """
    Follow the lower trail to an addressed block, then either mirror-repair the missing block or swap it with that addressed block.

    Input structure:
    - The top area is partitioned into 3x3 blocks and is left-right mirror-symmetric except for exactly one pure-black 3x3 missing block.
    - A solid magenta separator row spans the whole width.
    - Below the separator, one orange source pixel touches a single unbranched yellow 4-connected trail; the bottom area has as many rows as the top area has block rows.

    Rule:
    - Starting from the orange source, follow the yellow trail to its far endpoint.
    - Let `rel_row` be the endpoint row inside the bottom area and let `rel_col` be its column; this addresses top block `(rel_row, rel_col // 3)`.
    - Recover the missing block by copying each missing cell from its left-right mirror column in the top area.
    - If the addressed block is the missing block, fill the hole with the recovered patch.
    - Otherwise, copy the addressed block into the hole and copy the recovered patch into the addressed block, then output only the modified top area.

    Key insight:
    - The trail endpoint is not the final drawing location itself; it is an address that selects which top 3x3 block participates in the conditional repair-or-swap operation.
    """
    info = _inspect_grid(grid)
    out = info["top"].copy()

    hole_br, hole_bc = info["hole"]
    target_br, target_bc = info["target"]
    patch = info["patch"]

    hole_rs, hole_cs = _block_slice(hole_br, hole_bc)
    target_rs, target_cs = _block_slice(target_br, target_bc)

    if (target_br, target_bc) == (hole_br, hole_bc):
        out[hole_rs, hole_cs] = patch
    else:
        out[hole_rs, hole_cs] = info["target_block"]
        out[target_rs, target_cs] = patch

    return out


def _block_key(block: np.ndarray) -> bytes:
    return np.asarray(block, dtype=int).tobytes()


def _random_block(symmetric: bool) -> np.ndarray:
    for _ in range(200):
        color_count = int(RNG.integers(2, min(5, len(PATTERN_COLORS)) + 1))
        palette = RNG.choice(PATTERN_COLORS, size=color_count, replace=False)
        if symmetric:
            side = RNG.choice(palette, size=(BLOCK, 1))
            center = RNG.choice(palette, size=(BLOCK, 1))
            block = np.concatenate([side, center, side], axis=1).astype(int)
        else:
            block = RNG.choice(palette, size=(BLOCK, BLOCK), replace=True).astype(int)
            if np.array_equal(block, np.fliplr(block)):
                continue
        if len(set(map(int, block.ravel()))) < 2:
            continue
        return block
    raise RuntimeError("failed to build a sufficiently distinctive block")


def _build_symmetric_top(block_rows: int, block_cols: int) -> np.ndarray | None:
    top = np.zeros((block_rows * BLOCK, block_cols * BLOCK), dtype=int)
    full_keys: set[bytes] = set()
    half_cols = (block_cols + 1) // 2

    for br in range(block_rows):
        for bc in range(half_cols):
            mirror_bc = block_cols - 1 - bc
            symmetric = bc == mirror_bc
            placed = False
            for _ in range(200):
                block = _random_block(symmetric=symmetric)
                mirror = np.fliplr(block)
                if _block_key(block) in full_keys or _block_key(mirror) in full_keys:
                    continue
                rs, cs = _block_slice(br, bc)
                top[rs, cs] = block
                mrs, mcs = _block_slice(br, mirror_bc)
                top[mrs, mcs] = block if symmetric else mirror
                full_keys.add(_block_key(block))
                full_keys.add(_block_key(mirror))
                placed = True
                break
            if not placed:
                return None
    return top


def _choose_hole(block_rows: int, block_cols: int) -> tuple[int, int]:
    candidates = [
        (br, bc)
        for br in range(block_rows)
        for bc in range(block_cols)
        if not (block_cols % 2 == 1 and bc == block_cols // 2)
    ]
    idx = int(RNG.integers(len(candidates)))
    return candidates[idx]


def _used_neighbor_count(cell: tuple[int, int], used: set[tuple[int, int]]) -> int:
    return sum((nr, nc) in used for nr, nc in _neighbors4(*cell))


def _search_path(height: int, width: int, endpoint: tuple[int, int], target_len: int, min_turns: int) -> list[tuple[int, int]] | None:
    path = [endpoint]
    used = {endpoint}

    def dfs(prev_dir: tuple[int, int] | None, turns: int) -> list[tuple[int, int]] | None:
        if len(path) == target_len:
            return path.copy() if turns >= min_turns else None

        cur = path[-1]
        remaining = target_len - len(path)
        options = []
        for nr, nc in _neighbors4(*cur):
            if not (0 <= nr < height and 0 <= nc < width):
                continue
            if (nr, nc) in used:
                continue
            if _used_neighbor_count((nr, nc), used) != 1:
                continue
            free_deg = sum(
                0 <= rr < height and 0 <= cc < width and (rr, cc) not in used
                for rr, cc in _neighbors4(nr, nc)
            )
            if remaining > 1 and free_deg == 0:
                continue
            step = (nr - cur[0], nc - cur[1])
            next_turns = turns + int(prev_dir is not None and step != prev_dir)
            if next_turns + (remaining - 1) < min_turns:
                continue
            options.append((free_deg, float(RNG.random()), step, (nr, nc), next_turns))

        options.sort(key=lambda item: (-item[0], item[1]))
        for _free_deg, _noise, step, nxt, next_turns in options:
            used.add(nxt)
            path.append(nxt)
            found = dfs(step, next_turns)
            if found is not None:
                return found
            path.pop()
            used.remove(nxt)
        return None

    return dfs(None, 0)


def _build_bottom(block_rows: int, width: int, target_br: int, target_bc: int) -> np.ndarray | None:
    endpoint_cols = list(range(target_bc * BLOCK, target_bc * BLOCK + BLOCK))
    max_len = min(width + block_rows, block_rows * width)
    min_len = min(4, max_len)
    if max_len < 3:
        return None

    for _ in range(150):
        endpoint_col = endpoint_cols[int(RNG.integers(len(endpoint_cols)))]
        endpoint = (target_br, endpoint_col)
        target_len = int(RNG.integers(min_len, max_len + 1))
        max_turns = min(3, target_len - 2)
        if max_turns <= 0:
            min_turns = 0
        else:
            turn_choices = [0, 1, 2, 3][: max_turns + 1]
            weights = np.array([1, 3, 3, 2][: len(turn_choices)], dtype=float)
            weights /= weights.sum()
            min_turns = int(RNG.choice(turn_choices, p=weights))

        path = _search_path(block_rows, width, endpoint, target_len, min_turns)
        if path is None:
            continue
        source = path[-1]
        if len(path) < 4:
            continue
        if source == endpoint:
            continue

        bottom = np.zeros((block_rows, width), dtype=int)
        for r, c in path[:-1]:
            bottom[r, c] = TRAIL_COLOR
        bottom[source] = SOURCE_COLOR
        return bottom
    return None


def _swap_free_direct_repair(top: np.ndarray, hole: tuple[int, int], patch: np.ndarray) -> np.ndarray:
    out = top.copy()
    rs, cs = _block_slice(*hole)
    out[rs, cs] = patch
    return out


def _validate_generated(grid: np.ndarray) -> None:
    info = _inspect_grid(grid)
    out = main(grid)
    assert out.shape == info["top"].shape

    hole = info["hole"]
    target = info["target"]
    patch = info["patch"]
    repair_only = _swap_free_direct_repair(info["top"], hole, patch)

    non_hole_keys = []
    for br in range(info["block_rows"]):
        for bc in range(info["block_cols"]):
            if (br, bc) == hole:
                continue
            rs, cs = _block_slice(br, bc)
            non_hole_keys.append(_block_key(info["top"][rs, cs]))
    assert len(non_hole_keys) == len(set(non_hole_keys)), "non-hole blocks should be exact-unique"

    if target == hole:
        assert np.array_equal(out, repair_only), "repair branch should only fill the missing block"
    else:
        assert not np.array_equal(info["target_block"], patch), "swap branch must use a genuinely different target block"
        assert not np.array_equal(out, repair_only), "swap branch must differ from direct mirror repair"
        hole_rs, hole_cs = _block_slice(*hole)
        target_rs, target_cs = _block_slice(*target)
        assert np.array_equal(out[hole_rs, hole_cs], info["target_block"])
        assert np.array_equal(out[target_rs, target_cs], patch)


def _gen_attempt() -> np.ndarray | None:
    block_rows = int(RNG.choice(GEN_BLOCK_ROWS))
    block_cols = int(RNG.choice(GEN_BLOCK_COLS))
    top = _build_symmetric_top(block_rows, block_cols)
    if top is None:
        return None

    hole = _choose_hole(block_rows, block_cols)
    hole_rs, hole_cs = _block_slice(*hole)
    patch = top[hole_rs, hole_cs].copy()

    repair_branch = bool(RNG.integers(0, 2))
    if repair_branch:
        target = hole
    else:
        mirror_target = (hole[0], block_cols - 1 - hole[1])
        candidates = [(br, bc) for br in range(block_rows) for bc in range(block_cols) if (br, bc) != hole]
        if np.array_equal(top[_block_slice(*mirror_target)], patch):
            candidates = [cand for cand in candidates if cand != mirror_target]
        elif RNG.random() < 0.4:
            target = mirror_target
            candidates = None
        if candidates is not None:
            if not candidates:
                return None
            target = candidates[int(RNG.integers(len(candidates)))]

    bottom = _build_bottom(block_rows, top.shape[1], target[0], target[1])
    if bottom is None:
        return None

    inp = top.copy()
    inp[hole_rs, hole_cs] = 0
    grid = np.vstack([inp, np.full((1, inp.shape[1]), SEP_COLOR, dtype=int), bottom])

    try:
        _validate_generated(grid)
    except AssertionError:
        return None
    return grid


def generate_input() -> np.ndarray:
    for _ in range(GEN_MAX_ATTEMPTS):
        grid = _gen_attempt()
        if grid is not None:
            return grid
    raise RuntimeError("failed to generate a valid conditional mirror-repair puzzle")
