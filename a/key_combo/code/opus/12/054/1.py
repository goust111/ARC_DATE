from __future__ import annotations

from collections import deque
import random

import numpy as np


BG = 0
OPEN = 5
RED = 2
GREEN = 3

TILE = 3
MASK_SIDE = 3
REGION_SIDE = TILE * MASK_SIDE
GRID_SIDE = 13
REGION_TOP = 2
REGION_LEFT = 2

INF = 10**9
MAX_RETRIES = 500


Coord = tuple[int, int]


def _in_bounds(shape: tuple[int, int], r: int, c: int) -> bool:
    return 0 <= r < shape[0] and 0 <= c < shape[1]


def _neighbors4(r: int, c: int):
    yield r - 1, c
    yield r + 1, c
    yield r, c - 1
    yield r, c + 1


def _bbox(mask: np.ndarray) -> tuple[int, int, int, int] | None:
    coords = np.argwhere(mask)
    if coords.size == 0:
        return None
    r0, c0 = coords.min(axis=0)
    r1, c1 = coords.max(axis=0)
    return int(r0), int(c0), int(r1), int(c1)


def _is_connected(mask: np.ndarray) -> bool:
    coords = np.argwhere(mask)
    if coords.size == 0:
        return False
    start = tuple(map(int, coords[0]))
    seen = {start}
    queue = deque([start])
    while queue:
        r, c = queue.popleft()
        for nr, nc in _neighbors4(r, c):
            if not _in_bounds(mask.shape, nr, nc) or not mask[nr, nc]:
                continue
            nxt = (nr, nc)
            if nxt in seen:
                continue
            seen.add(nxt)
            queue.append(nxt)
    return len(seen) == int(mask.sum())


def _graph_stats(mask: np.ndarray) -> tuple[int, int, int]:
    nodes = int(mask.sum())
    edges = 0
    max_deg = 0
    for r, c in np.argwhere(mask):
        deg = 0
        for nr, nc in _neighbors4(int(r), int(c)):
            if _in_bounds(mask.shape, nr, nc) and mask[nr, nc]:
                deg += 1
        max_deg = max(max_deg, deg)
        edges += deg
    return nodes, edges // 2, max_deg


def _bfs_distances(channel: np.ndarray, start: Coord) -> np.ndarray:
    dist = np.full(channel.shape, INF, dtype=int)
    assert channel[start], "seed must lie on a channel cell"
    dist[start] = 0
    queue = deque([start])
    while queue:
        r, c = queue.popleft()
        for nr, nc in _neighbors4(r, c):
            if not _in_bounds(channel.shape, nr, nc) or not channel[nr, nc]:
                continue
            if dist[nr, nc] != INF:
                continue
            dist[nr, nc] = dist[r, c] + 1
            queue.append((nr, nc))
    return dist


def _paint_channel(
    channel: np.ndarray, seeds: dict[int, Coord], *, parity: bool
) -> tuple[np.ndarray, dict[str, object]]:
    dist_red = _bfs_distances(channel, seeds[RED])
    dist_green = _bfs_distances(channel, seeds[GREEN])
    lex_winner = RED if seeds[RED] < seeds[GREEN] else GREEN

    out = np.zeros(channel.shape, dtype=int)
    owner_map = np.full(channel.shape, -1, dtype=int)
    tie_count = 0

    for r, c in np.argwhere(channel):
        r = int(r)
        c = int(c)
        dr = int(dist_red[r, c])
        dg = int(dist_green[r, c])
        if dr == INF and dg == INF:
            continue
        if dr < dg:
            owner = RED
            d = dr
        elif dg < dr:
            owner = GREEN
            d = dg
        else:
            owner = lex_winner
            d = dr
            tie_count += 1
        owner_map[r, c] = owner
        if not parity:
            out[r, c] = owner
        elif d % 2 == 0:
            out[r, c] = owner
        else:
            out[r, c] = GREEN if owner == RED else RED

    meta = {
        "dist_red": dist_red,
        "dist_green": dist_green,
        "owner_map": owner_map,
        "tie_count": tie_count,
        "owner_counts": {
            RED: int(np.sum(owner_map == RED)),
            GREEN: int(np.sum(owner_map == GREEN)),
        },
    }
    return out, meta


def _decode_input(grid: np.ndarray) -> tuple[np.ndarray, dict[int, Coord]]:
    g = np.asarray(grid, dtype=int)
    assert g.ndim == 2, "expected a 2D grid"
    assert set(map(int, np.unique(g))) <= {BG, OPEN, RED, GREEN}, "unexpected colors"

    box = _bbox(g != BG)
    assert box is not None, "expected a visible 9x9 region"
    r0, c0, r1, c1 = box
    crop = g[r0 : r1 + 1, c0 : c1 + 1]
    assert crop.shape == (REGION_SIDE, REGION_SIDE), "expected a 9x9 encoded region"

    mask = np.zeros((MASK_SIDE, MASK_SIDE), dtype=bool)
    seeds: dict[int, Coord] = {}
    seed_blocks: dict[int, tuple[int, int]] = {}

    for br in range(MASK_SIDE):
        for bc in range(MASK_SIDE):
            tile = crop[3 * br : 3 * br + 3, 3 * bc : 3 * bc + 3]
            vals = set(map(int, tile.flatten()))
            if vals <= {BG}:
                assert np.all(tile == BG), "blocked tiles must be uniformly black"
                continue

            assert BG not in vals, "open tiles must not mix black with non-black"
            assert vals <= {OPEN, RED, GREEN}, "open tiles may only contain gray plus one seed"
            special = np.argwhere((tile == RED) | (tile == GREEN))
            assert len(special) <= 1, "each tile may contain at most one seed"
            if len(special) == 1:
                sr, sc = map(int, special[0])
                color = int(tile[sr, sc])
                assert color not in seeds, "each seed color must appear exactly once"
                tmp = tile.copy()
                tmp[sr, sc] = OPEN
                assert np.all(tmp == OPEN), "seed tiles must otherwise be gray"
                seeds[color] = (3 * br + sr, 3 * bc + sc)
                seed_blocks[color] = (br, bc)
            else:
                assert np.all(tile == OPEN), "non-seed open tiles must be uniformly gray"
            mask[br, bc] = True

    assert seeds.keys() == {RED, GREEN}, "expected exactly one red seed and one green seed"
    assert seed_blocks[RED] != seed_blocks[GREEN], "seeds must lie in different open tiles"
    assert int(mask.sum()) >= 5, "expected at least five open coarse cells"
    return mask, seeds


def _solve_from_mask(mask: np.ndarray, seeds: dict[int, Coord]) -> tuple[np.ndarray, dict[str, object]]:
    channel = np.kron(mask.astype(int), mask.astype(int)).astype(bool)
    assert channel.shape == (REGION_SIDE, REGION_SIDE)
    assert channel[seeds[RED]] and channel[seeds[GREEN]], "mapped seeds must land on channels"
    return _paint_channel(channel, seeds, parity=True)


def main(grid: np.ndarray) -> np.ndarray:
    """
    Decode a block-encoded 3x3 mask, self-expand it into a 9x9 fractal maze, and color its channels by dual-source BFS parity.

    Input structure:
    - The background is black.
    - A single visible 9x9 region is partitioned into nine 3x3 tiles.
    - Each tile is either uniformly black or uniformly gray, except that exactly one red seed and one green seed replace one gray cell each inside two different gray tiles.

    Rule:
    - Compress the 9x9 region into a 3x3 binary mask whose open cells are the gray tiles.
    - Form a 9x9 channel map by taking the Kronecker self-product of that mask.
    - Keep each seed's offset inside the 9x9 region when mapping it into the channel map.
    - Run simultaneous 4-connected BFS on the channel cells from both mapped seeds; each channel cell is owned by the nearer seed, with equal-distance ties broken by the lexicographically smaller seed coordinate.
    - Paint an owned cell with its owner's color at even BFS distance and with the other seed's color at odd BFS distance; non-channel cells stay black.

    Key insight:
    - The input's coarse 3x3 mask controls both the large-scale block layout and the fine-scale intra-block openings, so only the self-similar channel graph combined with path-distance parity determines the final red/green weave.
    """
    mask, seeds = _decode_input(np.asarray(grid, dtype=int))
    out, _meta = _solve_from_mask(mask, seeds)
    return out.astype(int)


def _build_mask_candidates() -> list[np.ndarray]:
    candidates: list[np.ndarray] = []
    for bits in range(1 << (MASK_SIDE * MASK_SIDE)):
        mask = np.array([(bits >> i) & 1 for i in range(MASK_SIDE * MASK_SIDE)], dtype=bool)
        mask = mask.reshape(MASK_SIDE, MASK_SIDE)
        on_count = int(mask.sum())
        if not 5 <= on_count <= 8:
            continue
        if not np.all(mask.any(axis=0)) or not np.all(mask.any(axis=1)):
            continue
        if not _is_connected(mask):
            continue
        channel = np.kron(mask.astype(int), mask.astype(int)).astype(bool)
        if not _is_connected(channel):
            continue
        nodes, edges, max_deg = _graph_stats(channel)
        if edges < nodes or max_deg < 3:
            continue
        candidates.append(mask)
    assert candidates, "expected non-empty mask candidate set"
    return candidates


MASK_CANDIDATES = _build_mask_candidates()
USED_INPUT_SIGNATURES: set[bytes] = set()
USED_OUTPUT_SIGNATURES: set[bytes] = set()


def _render_input(mask: np.ndarray, seeds: dict[int, Coord]) -> np.ndarray:
    height = random.randint(REGION_SIDE, 30)
    width = random.randint(REGION_SIDE, 30)
    top = random.randint(0, height - REGION_SIDE)
    left = random.randint(0, width - REGION_SIDE)
    grid = np.zeros((height, width), dtype=int)
    region = np.kron(mask.astype(int), np.ones((TILE, TILE), dtype=int)) * OPEN
    grid[top : top + REGION_SIDE, left : left + REGION_SIDE] = region
    for color, (r, c) in seeds.items():
        grid[top + r, left + c] = color
    return grid


def _seed_record(mask: np.ndarray, seeds: dict[int, Coord]) -> dict[str, object] | None:
    channel = np.kron(mask.astype(int), mask.astype(int)).astype(bool)
    region_open = np.kron(mask.astype(int), np.ones((TILE, TILE), dtype=int)).astype(bool)
    if not (channel[seeds[RED]] and channel[seeds[GREEN]]):
        return None

    out, meta = _solve_from_mask(mask, seeds)
    owner_only, _ = _paint_channel(channel, seeds, parity=False)
    direct_region, _ = _paint_channel(region_open, seeds, parity=True)
    owner_counts = meta["owner_counts"]

    if not np.all(out[channel] != BG):
        return None
    if len(set(map(int, np.unique(out))) - {BG}) != 2:
        return None
    if owner_counts[RED] < 3 or owner_counts[GREEN] < 3:
        return None
    if np.array_equal(out, owner_only):
        return None
    if np.array_equal(out, direct_region):
        return None

    return {
        "seeds": seeds,
        "output": out,
        "tie_count": int(meta["tie_count"]),
        "same_row": seeds[RED][0] == seeds[GREEN][0],
    }


def _sample_seed_record(mask: np.ndarray, mode: str, attempts: int = 200) -> dict[str, object] | None:
    open_tiles = [tuple(map(int, pos)) for pos in np.argwhere(mask)]
    open_offsets = [tuple(map(int, pos)) for pos in np.argwhere(mask)]

    for _ in range(attempts):
        red_block, green_block = random.sample(open_tiles, 2)
        red_off = random.choice(open_offsets)
        green_off = random.choice(open_offsets)
        seeds = {
            RED: (3 * red_block[0] + red_off[0], 3 * red_block[1] + red_off[1]),
            GREEN: (3 * green_block[0] + green_off[0], 3 * green_block[1] + green_off[1]),
        }
        if seeds[RED] == seeds[GREEN]:
            continue

        rec = _seed_record(mask, seeds)
        if rec is None:
            continue
        if mode == "same_row_tie" and not (rec["same_row"] and rec["tie_count"] > 0):
            continue
        if mode == "tie" and not (rec["tie_count"] > 0):
            continue
        if mode == "no_tie" and rec["tie_count"] != 0:
            continue
        return rec
    return None


def _semantic_checks(mask: np.ndarray, seeds: dict[int, Coord], expected: np.ndarray, inp: np.ndarray) -> bool:
    if not np.array_equal(main(inp), expected):
        return False

    channel = np.kron(mask.astype(int), mask.astype(int)).astype(bool)
    region_open = np.kron(mask.astype(int), np.ones((TILE, TILE), dtype=int)).astype(bool)
    owner_only, _ = _paint_channel(channel, seeds, parity=False)
    direct_region, _ = _paint_channel(region_open, seeds, parity=True)

    if np.array_equal(expected, owner_only):
        return False
    if np.array_equal(expected, direct_region):
        return False

    adj_diff = False
    for r, c in np.argwhere(channel):
        r = int(r)
        c = int(c)
        for nr, nc in _neighbors4(r, c):
            if _in_bounds(channel.shape, nr, nc) and channel[nr, nc] and expected[r, c] != expected[nr, nc]:
                adj_diff = True
                break
        if adj_diff:
            break
    if not adj_diff:
        return False

    return True


def generate_input() -> np.ndarray:
    for _ in range(MAX_RETRIES):
        mask = random.choice(MASK_CANDIDATES).copy()
        mode = random.choices(
            ["same_row_tie", "tie", "no_tie"],
            weights=[3, 2, 2],
            k=1,
        )[0]
        rec = (
            _sample_seed_record(mask, mode)
            or _sample_seed_record(mask, "tie", attempts=120)
            or _sample_seed_record(mask, "no_tie", attempts=120)
            or _sample_seed_record(mask, "any", attempts=120)
        )
        if rec is None:
            continue
        seeds = rec["seeds"]
        expected = rec["output"]
        inp = _render_input(mask, seeds)
        in_sig = inp.tobytes()
        out_sig = expected.tobytes()
        if in_sig in USED_INPUT_SIGNATURES or out_sig in USED_OUTPUT_SIGNATURES:
            continue
        if _semantic_checks(mask, seeds, expected, inp):
            USED_INPUT_SIGNATURES.add(in_sig)
            USED_OUTPUT_SIGNATURES.add(out_sig)
            return inp
    raise RuntimeError("failed to generate a valid self-similar parity maze input")
