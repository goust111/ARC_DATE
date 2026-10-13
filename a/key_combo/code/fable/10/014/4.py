from __future__ import annotations

from collections import Counter, deque
import random

import numpy as np


DIRS4 = [(-1, 0), (1, 0), (0, -1), (0, 1)]
COLOR_POOL = tuple(range(1, 10))
SIDE_OPTIONS = (3, 5)
TILE_COUNT_OPTIONS = (3, 4, 5)
LATTICE_SIDE = 4
MAX_GENERATION_ATTEMPTS = 600
RECENT_SIGNATURE_LIMIT = 256

_SEEN_INPUT_SIGNATURES: set[tuple[tuple[int, int], bytes]] = set()
_SEEN_OUTPUT_SIGNATURES: set[tuple[tuple[int, int], bytes]] = set()


def _grid_signature(grid: np.ndarray) -> tuple[tuple[int, int], bytes]:
    arr = np.asarray(grid, dtype=np.uint8)
    return tuple(map(int, arr.shape)), arr.tobytes()


def _remember_signature(store: set[tuple[tuple[int, int], bytes]], sig: tuple[tuple[int, int], bytes]) -> None:
    if len(store) >= RECENT_SIGNATURE_LIMIT:
        store.clear()
    store.add(sig)


def _get_bg(grid: np.ndarray) -> int:
    return int(Counter(map(int, np.asarray(grid).ravel())).most_common(1)[0][0])


def _bbox_coords(coords: np.ndarray) -> tuple[int, int, int, int]:
    r0, c0 = coords.min(axis=0)
    r1, c1 = coords.max(axis=0)
    return int(r0), int(c0), int(r1), int(c1)


def _crop_bbox(grid: np.ndarray, box: tuple[int, int, int, int]) -> np.ndarray:
    r0, c0, r1, c1 = box
    return np.asarray(grid)[r0 : r1 + 1, c0 : c1 + 1].copy()


def _components(mask: np.ndarray) -> list[np.ndarray]:
    mask = np.asarray(mask, dtype=bool)
    h, w = mask.shape
    seen = np.zeros_like(mask, dtype=bool)
    comps: list[np.ndarray] = []
    for r in range(h):
        for c in range(w):
            if not mask[r, c] or seen[r, c]:
                continue
            queue = deque([(r, c)])
            seen[r, c] = True
            coords: list[tuple[int, int]] = []
            while queue:
                rr, cc = queue.popleft()
                coords.append((rr, cc))
                for dr, dc in DIRS4:
                    nr, nc = rr + dr, cc + dc
                    if 0 <= nr < h and 0 <= nc < w and mask[nr, nc] and not seen[nr, nc]:
                        seen[nr, nc] = True
                        queue.append((nr, nc))
            comps.append(np.array(coords, dtype=int))
    return comps


def _edge_midpoints(side: int) -> dict[tuple[int, int], tuple[int, int]]:
    mid = side // 2
    return {
        (0, mid): (-1, 0),
        (side - 1, mid): (1, 0),
        (mid, 0): (0, -1),
        (mid, side - 1): (0, 1),
    }


def _direction_to_pointer_local(side: int, direction: tuple[int, int]) -> tuple[int, int]:
    mid = side // 2
    mapping = {
        (-1, 0): (0, mid),
        (1, 0): (side - 1, mid),
        (0, -1): (mid, 0),
        (0, 1): (mid, side - 1),
    }
    return mapping[direction]


def _dominant_non_pointer_color(patch: np.ndarray, pointer_color: int) -> int:
    values, counts = np.unique(patch[patch != pointer_color], return_counts=True)
    assert len(values) >= 1, "tile must have a visible base color"
    max_count = int(counts.max())
    winners = [int(v) for v, cnt in zip(values, counts) if int(cnt) == max_count]
    assert len(winners) == 1, "tile base color must be uniquely dominant"
    return winners[0]


def _analyze(grid: np.ndarray) -> dict:
    g = np.asarray(grid, dtype=int)
    assert g.ndim == 2, "expected a 2D grid"
    h, w = map(int, g.shape)
    bg = _get_bg(g)
    comps = _components(g != bg)
    assert len(comps) >= 2, "expected at least two tiles"

    tiles: list[dict] = []
    cell_to_tile: dict[tuple[int, int], int] = {}
    for coords in comps:
        box = _bbox_coords(coords)
        r0, c0, r1, c1 = box
        side_r = r1 - r0 + 1
        side_c = c1 - c0 + 1
        assert side_r == side_c, "each tile must be square"
        side = int(side_r)
        assert side % 2 == 1 and side >= 3, "tile side must be odd and at least 3"
        patch = _crop_bbox(g, box)
        assert np.all(patch != bg), "tiles must be fully filled"
        assert len(coords) == side * side, "tile component must fill its bounding square"
        center = (r0 + side // 2, c0 + side // 2)
        tile = {
            "bbox": box,
            "patch": patch,
            "side": side,
            "center": center,
        }
        tid = len(tiles)
        tiles.append(tile)
        for rr in range(r0, r1 + 1):
            for cc in range(c0, c1 + 1):
                cell_to_tile[(rr, cc)] = tid

    non_bg_colors = sorted(set(map(int, np.unique(g))) - {bg})
    pointer_candidates: list[int] = []
    for color in non_bg_colors:
        valid = True
        for tile in tiles:
            locs = np.argwhere(tile["patch"] == color)
            if len(locs) != 1:
                valid = False
                break
            local = tuple(map(int, locs[0]))
            if local not in _edge_midpoints(tile["side"]):
                valid = False
                break
        if valid:
            pointer_candidates.append(color)
    assert len(pointer_candidates) == 1, "expected a unique shared pointer color"
    pointer_color = pointer_candidates[0]

    successors: list[int | None] = []
    for tile in tiles:
        loc = tuple(map(int, np.argwhere(tile["patch"] == pointer_color)[0]))
        dr, dc = _edge_midpoints(tile["side"])[loc]
        r0, c0, _r1, _c1 = tile["bbox"]
        pr, pc = r0 + loc[0], c0 + loc[1]
        base_color = _dominant_non_pointer_color(tile["patch"], pointer_color)
        tile["pointer_local"] = loc
        tile["pointer_global"] = (pr, pc)
        tile["direction"] = (dr, dc)
        tile["base_color"] = base_color

        rr, cc = pr + dr, pc + dc
        succ: int | None = None
        while 0 <= rr < h and 0 <= cc < w:
            if g[rr, cc] != bg:
                succ = cell_to_tile[(rr, cc)]
                assert succ != len(successors), "pointer ray must leave its own tile before hitting another"
                break
            rr += dr
            cc += dc
        successors.append(succ)

    indegree = [0] * len(tiles)
    for succ in successors:
        if succ is not None:
            indegree[succ] += 1
    assert max(indegree) <= 1, "each tile may have at most one predecessor"
    heads = [idx for idx, deg in enumerate(indegree) if deg == 0]
    assert len(heads) == 1, "expected exactly one unpointed chain head"
    assert successors.count(None) == 1, "expected exactly one chain tail"

    order: list[int] = []
    cur = heads[0]
    seen: set[int] = set()
    while cur is not None:
        assert cur not in seen, "successor relation must not cycle"
        seen.add(cur)
        order.append(cur)
        cur = successors[cur]
    assert len(order) == len(tiles), "chain must cover every tile"

    return {
        "bg": bg,
        "pointer_color": pointer_color,
        "tiles": tiles,
        "successors": successors,
        "order": order,
    }


def main(grid: np.ndarray) -> np.ndarray:
    """Follow the unique pointer-hit chain across scattered tiles, restore each pointer cell, and concatenate the tiles horizontally.

    Input structure:
    - A black background contains several separated same-size square tiles placed at irregular positions.
    - Each tile is fully filled by its own base color plus an internal pattern, with one shared-color pointer pixel at the midpoint of one edge.
    - The pointer color does not appear anywhere except those edge-midpoint markers.

    Rule:
    - For each tile, cast a ray outward from its pointer cell along the normal of the edge that contains that pointer.
    - The first tile hit by that ray is the current tile's successor; the unique tile hit by no other ray is the chain head.
    - Starting from that head, walk the full successor chain, replace every pointer pixel by that tile's base color, and concatenate the restored tiles from left to right.

    Key insight:
    - Tile positions themselves do not determine reading order; the order is encoded only by which tile each edge pointer hits, and the chain must start at the uniquely unpointed tile.
    """
    info = _analyze(grid)
    pointer_color = info["pointer_color"]
    restored_tiles: list[np.ndarray] = []
    for idx in info["order"]:
        tile = info["tiles"][idx]
        patch = tile["patch"].copy()
        pr, pc = tile["pointer_local"]
        patch[pr, pc] = tile["base_color"]
        assert not np.any(patch == pointer_color), "restored tile should not keep the pointer color"
        restored_tiles.append(patch)
    return np.concatenate(restored_tiles, axis=1)


def _path_move_dirs(path: list[tuple[int, int]]) -> list[tuple[int, int]]:
    return [
        (path[i + 1][0] - path[i][0], path[i + 1][1] - path[i][1])
        for i in range(len(path) - 1)
    ]


def _free_tail_dirs(path: list[tuple[int, int]]) -> list[tuple[int, int]]:
    tail_r, tail_c = path[-1]
    occupied = set(path)
    options: list[tuple[int, int]] = []
    for dr, dc in DIRS4:
        rr, cc = tail_r + dr, tail_c + dc
        blocked = False
        while 0 <= rr < LATTICE_SIDE and 0 <= cc < LATTICE_SIDE:
            if (rr, cc) in occupied:
                blocked = True
                break
            rr += dr
            cc += dc
        if not blocked:
            options.append((dr, dc))
    return options


def _sample_path(tile_count: int) -> tuple[list[tuple[int, int]], tuple[int, int]] | None:
    for _ in range(300):
        start = (random.randrange(LATTICE_SIDE), random.randrange(LATTICE_SIDE))
        path = [start]
        used = {start}
        while len(path) < tile_count:
            r, c = path[-1]
            options = []
            for dr, dc in DIRS4:
                nr, nc = r + dr, c + dc
                if 0 <= nr < LATTICE_SIDE and 0 <= nc < LATTICE_SIDE and (nr, nc) not in used:
                    options.append((nr, nc))
            if not options:
                break
            if len(path) >= 2:
                prev_dr = path[-1][0] - path[-2][0]
                prev_dc = path[-1][1] - path[-2][1]
                turned = [pt for pt in options if (pt[0] - r, pt[1] - c) != (prev_dr, prev_dc)]
                if turned and random.random() < 0.75:
                    options = turned
            nxt = random.choice(options)
            path.append(nxt)
            used.add(nxt)
        if len(path) != tile_count:
            continue
        move_dirs = _path_move_dirs(path)
        has_h = any(dc != 0 for _dr, dc in move_dirs)
        has_v = any(dr != 0 for dr, _dc in move_dirs)
        if not (has_h and has_v):
            continue
        tail_dirs = _free_tail_dirs(path)
        if not tail_dirs:
            continue
        return path, random.choice(tail_dirs)
    return None


def _make_restored_tile(side: int, base_color: int, pointer_color: int, existing: list[np.ndarray]) -> np.ndarray:
    forbidden = set(_edge_midpoints(side))
    fill_cells = [(r, c) for r in range(side) for c in range(side) if (r, c) not in forbidden]
    accent_palette = [c for c in COLOR_POOL if c not in {base_color, pointer_color}]
    min_accents = 1 if side == 3 else 2
    max_accents = 2 if side == 3 else 5

    for _ in range(200):
        patch = np.full((side, side), base_color, dtype=int)
        accent_color = random.choice(accent_palette)
        accent_count = random.randint(min_accents, min(max_accents, len(fill_cells)))
        for r, c in random.sample(fill_cells, accent_count):
            patch[r, c] = accent_color
        if _dominant_non_pointer_color(patch, pointer_color) != base_color:
            continue
        if any(np.array_equal(patch, other) for other in existing):
            continue
        return patch
    raise RuntimeError("failed to generate a unique tile pattern")


def _row_major_order(tiles: list[dict]) -> list[int]:
    return sorted(range(len(tiles)), key=lambda idx: (tiles[idx]["center"][0], tiles[idx]["center"][1]))


def _col_major_order(tiles: list[dict]) -> list[int]:
    return sorted(range(len(tiles)), key=lambda idx: (tiles[idx]["center"][1], tiles[idx]["center"][0]))


def _semantic_ok(inp: np.ndarray, expected: np.ndarray, tile_count: int) -> bool:
    try:
        info = _analyze(inp)
    except Exception:
        return False

    if len(info["tiles"]) != tile_count:
        return False
    if not np.array_equal(main(inp), expected):
        return False

    order = info["order"]
    if order in (
        _row_major_order(info["tiles"]),
        list(reversed(_row_major_order(info["tiles"]))),
        _col_major_order(info["tiles"]),
        list(reversed(_col_major_order(info["tiles"]))),
    ):
        return False

    move_dirs = [info["tiles"][idx]["direction"] for idx in order[:-1]]
    has_h = any(dc != 0 for _dr, dc in move_dirs)
    has_v = any(dr != 0 for dr, _dc in move_dirs)
    if not (has_h and has_v):
        return False

    restored = []
    for idx in order:
        tile = info["tiles"][idx]
        patch = tile["patch"].copy()
        pr, pc = tile["pointer_local"]
        patch[pr, pc] = tile["base_color"]
        restored.append(patch)
    if len({patch.tobytes() for patch in restored}) != len(restored):
        return False
    if any(np.any(patch == info["pointer_color"]) for patch in restored):
        return False

    naive = np.concatenate([info["tiles"][idx]["patch"] for idx in order], axis=1)
    if np.array_equal(naive, expected):
        return False

    return True


def _gen_attempt() -> np.ndarray | None:
    tile_count = random.choice(TILE_COUNT_OPTIONS)
    side = random.choice(SIDE_OPTIONS)
    if tile_count * side > 30:
        return None

    sampled = _sample_path(tile_count)
    if sampled is None:
        return None
    path, tail_dir = sampled

    gap = random.choice((1, 2))
    pitch = side + gap
    min_r = min(r for r, _c in path)
    max_r = max(r for r, _c in path)
    min_c = min(c for _r, c in path)
    max_c = max(c for _r, c in path)
    box_h = (max_r - min_r) * pitch + side
    box_w = (max_c - min_c) * pitch + side
    if box_h > 30 or box_w > 30:
        return None

    spare_h = 30 - box_h
    spare_w = 30 - box_w
    top_margin = random.randint(1, min(4, spare_h))
    left_margin = random.randint(1, min(4, spare_w))
    bottom_margin = random.randint(1, min(4, spare_h - top_margin + 1))
    right_margin = random.randint(1, min(4, spare_w - left_margin + 1))
    h = box_h + top_margin + bottom_margin
    w = box_w + left_margin + right_margin
    if h > 30 or w > 30:
        return None

    pointer_color = random.choice(COLOR_POOL)
    base_colors = random.sample([c for c in COLOR_POOL if c != pointer_color], tile_count)
    restored_tiles: list[np.ndarray] = []
    tile_positions: list[tuple[int, int]] = []
    directions: list[tuple[int, int]] = []

    normalized = [(r - min_r, c - min_c) for r, c in path]
    for idx, (gr, gc) in enumerate(normalized):
        r0 = top_margin + gr * pitch
        c0 = left_margin + gc * pitch
        tile_positions.append((r0, c0))
        if idx + 1 < tile_count:
            nr, nc = normalized[idx + 1]
            directions.append((nr - gr, nc - gc))
        else:
            directions.append(tail_dir)

    for idx in range(tile_count):
        restored_tiles.append(_make_restored_tile(side, base_colors[idx], pointer_color, restored_tiles))

    inp = np.zeros((h, w), dtype=int)
    for idx, (r0, c0) in enumerate(tile_positions):
        patch = restored_tiles[idx].copy()
        pr, pc = _direction_to_pointer_local(side, directions[idx])
        patch[pr, pc] = pointer_color
        inp[r0 : r0 + side, c0 : c0 + side] = patch

    expected = np.concatenate(restored_tiles, axis=1)
    if not _semantic_ok(inp, expected, tile_count):
        return None

    in_sig = _grid_signature(inp)
    out_sig = _grid_signature(expected)
    if in_sig in _SEEN_INPUT_SIGNATURES or out_sig in _SEEN_OUTPUT_SIGNATURES:
        return None
    _remember_signature(_SEEN_INPUT_SIGNATURES, in_sig)
    _remember_signature(_SEEN_OUTPUT_SIGNATURES, out_sig)
    return inp


def generate_input() -> np.ndarray:
    for _ in range(MAX_GENERATION_ATTEMPTS):
        grid = _gen_attempt()
        if grid is not None:
            return grid
    raise RuntimeError("failed to generate a valid pointer-chain tile input")
