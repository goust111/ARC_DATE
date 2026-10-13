from __future__ import annotations

import random
from collections import Counter

import numpy as np


BLACK = 0
RED = 2
GREEN = 3
GREY = 5
ORANGE = 7

MAX_GEN_ATTEMPTS = 400


def _neighbors4(r: int, c: int) -> list[tuple[int, int]]:
    return [(r - 1, c), (r + 1, c), (r, c - 1), (r, c + 1)]


def _neighbors8(r: int, c: int) -> list[tuple[int, int]]:
    return [
        (r + dr, c + dc)
        for dr in (-1, 0, 1)
        for dc in (-1, 0, 1)
        if not (dr == 0 and dc == 0)
    ]


def _in_bounds(shape: tuple[int, int], r: int, c: int) -> bool:
    return 0 <= r < shape[0] and 0 <= c < shape[1]


def _separator_groups(grid: np.ndarray, axis: int) -> list[tuple[int, int]]:
    if axis == 0:
        full = [r for r in range(grid.shape[0]) if np.all(grid[r, :] == GREY)]
    else:
        full = [c for c in range(grid.shape[1]) if np.all(grid[:, c] == GREY)]

    groups: list[tuple[int, int]] = []
    if not full:
        return groups
    start = prev = full[0]
    for idx in full[1:]:
        if idx == prev + 1:
            prev = idx
            continue
        groups.append((start, prev))
        start = prev = idx
    groups.append((start, prev))
    return groups


def _spans_from_groups(length: int, groups: list[tuple[int, int]]) -> list[tuple[int, int]]:
    spans: list[tuple[int, int]] = []
    prev_end = -1
    for start, end in groups:
        spans.append((prev_end + 1, start))
        prev_end = end
    spans.append((prev_end + 1, length))
    return spans


def _parse_tile_grid(
    grid: np.ndarray,
) -> tuple[list[tuple[int, int]], list[tuple[int, int]], dict[tuple[int, int], np.ndarray]]:
    row_groups = _separator_groups(grid, axis=0)
    col_groups = _separator_groups(grid, axis=1)
    assert len(row_groups) == 2 and len(col_groups) == 2, "expected a 3x3 separator lattice"

    row_spans = _spans_from_groups(grid.shape[0], row_groups)
    col_spans = _spans_from_groups(grid.shape[1], col_groups)
    assert len(row_spans) == 3 and len(col_spans) == 3
    assert all(r1 > r0 for r0, r1 in row_spans)
    assert all(c1 > c0 for c0, c1 in col_spans)

    tile_hs = {r1 - r0 for r0, r1 in row_spans}
    tile_ws = {c1 - c0 for c0, c1 in col_spans}
    assert len(tile_hs) == 1 and len(tile_ws) == 1, "tiles must have a shared shape"

    tiles: dict[tuple[int, int], np.ndarray] = {}
    for tr, (r0, r1) in enumerate(row_spans):
        for tc, (c0, c1) in enumerate(col_spans):
            tiles[(tr, tc)] = np.array(grid[r0:r1, c0:c1], dtype=int)
    return row_spans, col_spans, tiles


def _majority_tile(tiles: dict[tuple[int, int], np.ndarray]) -> np.ndarray:
    counts = Counter(tile.tobytes() for tile in tiles.values())
    consensus_key, consensus_count = counts.most_common(1)[0]
    assert consensus_count >= 6, "consensus must be a strict tile majority"
    exemplar = next(tile for tile in tiles.values() if tile.tobytes() == consensus_key)
    return np.array(exemplar, dtype=int)


def _anomaly_extras(
    tiles: dict[tuple[int, int], np.ndarray], consensus: np.ndarray
) -> dict[tuple[int, int], set[tuple[int, int]]]:
    extras: dict[tuple[int, int], set[tuple[int, int]]] = {}
    for key, tile in tiles.items():
        if np.array_equal(tile, consensus):
            continue
        extra_mask = (tile == RED) & (consensus == BLACK)
        bad_mask = (tile != consensus) & ~extra_mask
        assert not bad_mask.any(), "anomalies may only add red cells"
        coords = {tuple(map(int, rc)) for rc in np.argwhere(extra_mask)}
        assert 1 <= len(coords) <= 2, "each anomaly must contribute 1-2 extra cells"
        extras[key] = coords
    assert 2 <= len(extras) <= 3, "expected 2-3 anomalous tiles"
    return extras


def _green_segments(
    row_spans: list[tuple[int, int]],
    col_spans: list[tuple[int, int]],
    extras_by_tile: dict[tuple[int, int], set[tuple[int, int]]],
) -> set[tuple[int, int]]:
    green: set[tuple[int, int]] = set()
    offsets = sorted({offset for offsets in extras_by_tile.values() for offset in offsets})

    for lr, lc in offsets:
        for tr in range(3):
            cols = sorted(tc for (r, tc), offsets2 in extras_by_tile.items() if r == tr and (lr, lc) in offsets2)
            if len(cols) >= 2:
                r = row_spans[tr][0] + lr
                c0 = col_spans[cols[0]][0] + lc
                c1 = col_spans[cols[-1]][0] + lc
                for c in range(c0, c1 + 1):
                    green.add((r, c))
        for tc in range(3):
            rows = sorted(tr for (tr, c), offsets2 in extras_by_tile.items() if c == tc and (lr, lc) in offsets2)
            if len(rows) >= 2:
                c = col_spans[tc][0] + lc
                r0 = row_spans[rows[0]][0] + lr
                r1 = row_spans[rows[-1]][0] + lr
                for r in range(r0, r1 + 1):
                    green.add((r, c))
    return green


def _render_from_description(
    grid: np.ndarray,
    row_spans: list[tuple[int, int]],
    col_spans: list[tuple[int, int]],
    extras_by_tile: dict[tuple[int, int], set[tuple[int, int]]],
) -> np.ndarray:
    out = np.array(grid, dtype=int, copy=True)

    for (tr, tc), offsets in extras_by_tile.items():
        r_base = row_spans[tr][0]
        c_base = col_spans[tc][0]
        for lr, lc in offsets:
            gr = r_base + lr
            gc = c_base + lc
            for nr, nc in _neighbors8(gr, gc):
                if _in_bounds(out.shape, nr, nc) and int(out[nr, nc]) == BLACK:
                    out[nr, nc] = ORANGE

    for r, c in _green_segments(row_spans, col_spans, extras_by_tile):
        if int(out[r, c]) != RED:
            out[r, c] = GREEN

    return out


def _solve(grid: np.ndarray) -> np.ndarray:
    row_spans, col_spans, tiles = _parse_tile_grid(grid)
    consensus = _majority_tile(tiles)
    extras_by_tile = _anomaly_extras(tiles, consensus)
    return _render_from_description(grid, row_spans, col_spans, extras_by_tile)


def main(grid: np.ndarray) -> np.ndarray:
    """Detect anomalous tiles, halo their extra pixels, and connect shared local-offset anomalies across the tile lattice.

    Input structure:
    - The grid is partitioned by full grey separator rows and columns into a 3x3 array of black-background tiles.
    - Most tiles contain the same red local pattern; 2-3 anomalous tiles add 1-2 extra red pixels on black cells.
    - At least one pair of anomalous tiles shares a tile row or tile column and contains an extra red pixel at the same local offset.

    Rule:
    - Infer the consensus tile by majority vote over the nine tile patterns.
    - In each anomalous tile, locate every local offset that is red in the anomaly and black in the consensus.
    - Paint orange on black 8-neighbors of those extra pixels, then draw green horizontal or vertical segments between anomalous tiles that share the same extra local offset in the same tile row or column; green overwrites orange and grey but never red.

    Key insight:
    - The same anomaly map drives two spatial operations: local halos around extra pixels and global line segments chosen in tile coordinates rather than by pairing arbitrary pixels in image space.
    """
    g = np.asarray(grid, dtype=int)
    assert g.ndim == 2
    return _solve(g)


def _random_consensus_tile(tile_h: int, tile_w: int) -> np.ndarray | None:
    if tile_h < 5 or tile_w < 5:
        return None

    size = random.randint(3, 5)
    interior = [(r, c) for r in range(1, tile_h - 1) for c in range(1, tile_w - 1)]
    start = random.choice(interior)
    cells = {start}

    while len(cells) < size:
        frontier = []
        for r, c in cells:
            for nr, nc in _neighbors4(r, c):
                if 1 <= nr < tile_h - 1 and 1 <= nc < tile_w - 1 and (nr, nc) not in cells:
                    frontier.append((nr, nc))
        if not frontier:
            return None
        cells.add(random.choice(frontier))

    tile = np.zeros((tile_h, tile_w), dtype=int)
    for r, c in cells:
        tile[r, c] = RED

    candidates = _extra_candidates(tile)
    if len(candidates["any"]) < 3:
        return None
    if not candidates["horiz"] and not candidates["vert"]:
        return None
    return tile


def _extra_candidates(tile: np.ndarray) -> dict[str, list[tuple[int, int]]]:
    tile_h, tile_w = tile.shape
    reds = {tuple(map(int, rc)) for rc in np.argwhere(tile == RED)}
    row_has_red = {r for r, _ in reds}
    col_has_red = {c for _, c in reds}

    candidates: set[tuple[int, int]] = set()
    for r, c in reds:
        for nr, nc in _neighbors4(r, c):
            if 1 <= nr < tile_h - 1 and 1 <= nc < tile_w - 1 and int(tile[nr, nc]) == BLACK:
                candidates.add((nr, nc))

    any_candidates = sorted(candidates)
    horiz_candidates = [pos for pos in any_candidates if pos[0] not in row_has_red]
    vert_candidates = [pos for pos in any_candidates if pos[1] not in col_has_red]
    return {"any": any_candidates, "horiz": horiz_candidates, "vert": vert_candidates}


def _pick_distinct(options: list[tuple[int, int]], forbidden: set[tuple[int, int]]) -> tuple[int, int] | None:
    choices = [pos for pos in options if pos not in forbidden]
    if not choices:
        return None
    return random.choice(choices)


def _build_case() -> tuple[np.ndarray, dict] | None:
    tile_h = random.choice([5, 6])
    tile_w = random.choice([5, 6])
    consensus = _random_consensus_tile(tile_h, tile_w)
    if consensus is None:
        return None

    candidates = _extra_candidates(consensus)
    feasible_modes = []
    if candidates["horiz"]:
        feasible_modes.append("pair_row")
    if candidates["vert"]:
        feasible_modes.append("pair_col")
    if candidates["horiz"] and len(candidates["any"]) >= 2:
        feasible_modes.append("row_triplet_mixed")
    horiz_cross = candidates["horiz"]
    vert_cross = [pos for pos in candidates["vert"] if pos not in set(horiz_cross)]
    if horiz_cross and vert_cross:
        feasible_modes.append("cross")
    if not feasible_modes:
        return None

    mode = random.choice(feasible_modes)
    extras_by_tile: dict[tuple[int, int], set[tuple[int, int]]] = {}

    if mode == "pair_row":
        row = random.randrange(3)
        cols = sorted(random.sample(range(3), 2))
        shared = random.choice(candidates["horiz"])
        extras_by_tile[(row, cols[0])] = {shared}
        extras_by_tile[(row, cols[1])] = {shared}
        if random.random() < 0.6:
            other_row = random.choice([r for r in range(3) if r != row])
            other_col = next(c for c in range(3) if c not in cols)
            extra = _pick_distinct(candidates["any"], set())
            if extra is None:
                return None
            extras_by_tile[(other_row, other_col)] = {extra}

    elif mode == "pair_col":
        col = random.randrange(3)
        rows = sorted(random.sample(range(3), 2))
        shared = random.choice(candidates["vert"])
        extras_by_tile[(rows[0], col)] = {shared}
        extras_by_tile[(rows[1], col)] = {shared}
        if random.random() < 0.6:
            other_col = random.choice([c for c in range(3) if c != col])
            other_row = next(r for r in range(3) if r not in rows)
            extra = _pick_distinct(candidates["any"], set())
            if extra is None:
                return None
            extras_by_tile[(other_row, other_col)] = {extra}

    elif mode == "row_triplet_mixed":
        row = random.randrange(3)
        shared = random.choice(candidates["horiz"])
        private = _pick_distinct(candidates["any"], {shared})
        if private is None:
            return None
        pair_cols = sorted(random.sample(range(3), 2))
        lone_col = next(c for c in range(3) if c not in pair_cols)
        extras_by_tile[(row, pair_cols[0])] = {shared}
        extras_by_tile[(row, pair_cols[1])] = {shared}
        extras_by_tile[(row, lone_col)] = {private}

    else:
        shared_h = random.choice(candidates["horiz"])
        shared_v = _pick_distinct(candidates["vert"], {shared_h})
        if shared_v is None:
            return None
        center = (random.randrange(3), random.randrange(3))
        row_choices = [c for c in range(3) if c != center[1]]
        col_choices = [r for r in range(3) if r != center[0]]
        row_partner = (center[0], random.choice(row_choices))
        col_partner = (random.choice(col_choices), center[1])
        extras_by_tile[center] = {shared_h, shared_v}
        extras_by_tile[row_partner] = {shared_h}
        extras_by_tile[col_partner] = {shared_v}

    assert 2 <= len(extras_by_tile) <= 3

    grid = np.zeros((3 * tile_h + 2, 3 * tile_w + 2), dtype=int)
    grid[tile_h, :] = GREY
    grid[2 * tile_h + 1, :] = GREY
    grid[:, tile_w] = GREY
    grid[:, 2 * tile_w + 1] = GREY

    row_spans = [(0, tile_h), (tile_h + 1, 2 * tile_h + 1), (2 * tile_h + 2, 3 * tile_h + 2)]
    col_spans = [(0, tile_w), (tile_w + 1, 2 * tile_w + 1), (2 * tile_w + 2, 3 * tile_w + 2)]

    for tr, (r0, r1) in enumerate(row_spans):
        for tc, (c0, c1) in enumerate(col_spans):
            tile = np.array(consensus, copy=True)
            for lr, lc in extras_by_tile.get((tr, tc), set()):
                tile[lr, lc] = RED
            grid[r0:r1, c0:c1] = tile

    meta = {
        "mode": mode,
        "tile_h": tile_h,
        "tile_w": tile_w,
        "consensus": np.array(consensus, copy=True),
        "row_spans": row_spans,
        "col_spans": col_spans,
        "extras_by_tile": {k: set(v) for k, v in extras_by_tile.items()},
    }
    return grid, meta


def _validate_generated(grid: np.ndarray, meta: dict) -> bool:
    row_spans, col_spans, tiles = _parse_tile_grid(grid)
    consensus = _majority_tile(tiles)
    if not np.array_equal(consensus, meta["consensus"]):
        return False

    extras_by_tile = _anomaly_extras(tiles, consensus)
    if extras_by_tile != meta["extras_by_tile"]:
        return False

    if row_spans != meta["row_spans"] or col_spans != meta["col_spans"]:
        return False

    for offsets in extras_by_tile.values():
        for lr, lc in offsets:
            if not (1 <= lr < meta["tile_h"] - 1 and 1 <= lc < meta["tile_w"] - 1):
                return False

    pair_exists = False
    for (tr0, tc0), offsets0 in extras_by_tile.items():
        for (tr1, tc1), offsets1 in extras_by_tile.items():
            if (tr0, tc0) >= (tr1, tc1):
                continue
            if (tr0 == tr1 or tc0 == tc1) and offsets0 & offsets1:
                pair_exists = True
    if not pair_exists:
        return False

    if meta["mode"] == "row_triplet_mixed":
        rows = [tr for tr, _ in extras_by_tile]
        if len(set(rows)) != 1:
            return False
        shared_offsets = Counter(offset for offsets in extras_by_tile.values() for offset in offsets)
        if shared_offsets.most_common(1)[0][1] != 2:
            return False

    if meta["mode"] == "cross":
        centers = [pos for pos, offsets in extras_by_tile.items() if len(offsets) == 2]
        if len(centers) != 1:
            return False
        center = centers[0]
        others = [pos for pos in extras_by_tile if pos != center]
        if len(others) != 2:
            return False
        if not any(pos[0] == center[0] for pos in others):
            return False
        if not any(pos[1] == center[1] for pos in others):
            return False

    expected = _render_from_description(grid, row_spans, col_spans, extras_by_tile)
    solved = main(grid)
    if not np.array_equal(solved, expected):
        return False
    if GREEN not in solved or ORANGE not in solved:
        return False
    return True


def generate_input() -> np.ndarray:
    for _ in range(MAX_GEN_ATTEMPTS):
        built = _build_case()
        if built is None:
            continue
        grid, meta = built
        if _validate_generated(grid, meta):
            return grid
    raise RuntimeError("failed to generate a valid anomalous tile lattice")
