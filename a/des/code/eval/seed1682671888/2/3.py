import random
from collections import Counter, deque

import numpy as np


WALL = 8
WALL_THICKNESS = 2
MAX_RETRIES = 300

ROOM_ROWS_RANGE = (2, 4)
ROOM_COLS_RANGE = (2, 4)
ROOM_SIZE_RANGE = (8, 10)
KEY_COUNT_MAX = 5

BASE_SHAPES = [
    np.array([[1, 1]], dtype=bool),
    np.array([[1, 1, 1]], dtype=bool),
    np.array([[1, 0], [1, 1]], dtype=bool),
    np.array([[1, 1], [1, 1]], dtype=bool),
    np.array([[1, 0], [1, 0], [1, 1]], dtype=bool),
    np.array([[1, 1, 1], [0, 1, 0]], dtype=bool),
    np.array([[0, 1, 1], [1, 1, 0]], dtype=bool),
]

DIRS = {
    "top": (-1, 0),
    "bottom": (1, 0),
    "left": (0, -1),
    "right": (0, 1),
}


def _neighbors4(r: int, c: int):
    yield r - 1, c
    yield r + 1, c
    yield r, c - 1
    yield r, c + 1


def _components(mask: np.ndarray) -> list[np.ndarray]:
    h, w = mask.shape
    seen = np.zeros((h, w), dtype=bool)
    comps: list[np.ndarray] = []
    for r in range(h):
        for c in range(w):
            if not mask[r, c] or seen[r, c]:
                continue
            q = deque([(r, c)])
            seen[r, c] = True
            pts = []
            while q:
                cr, cc = q.popleft()
                pts.append((cr, cc))
                for nr, nc in _neighbors4(cr, cc):
                    if 0 <= nr < h and 0 <= nc < w and mask[nr, nc] and not seen[nr, nc]:
                        seen[nr, nc] = True
                        q.append((nr, nc))
            comps.append(np.array(pts, dtype=int))
    return comps


def _bbox(coords: np.ndarray) -> tuple[int, int, int, int]:
    return (
        int(coords[:, 0].min()),
        int(coords[:, 1].min()),
        int(coords[:, 0].max()),
        int(coords[:, 1].max()),
    )


def _trim_mask(mask: np.ndarray) -> np.ndarray:
    coords = np.argwhere(mask)
    r0, c0, r1, c1 = _bbox(coords)
    return mask[r0 : r1 + 1, c0 : c1 + 1]


def _dihedral_masks(mask: np.ndarray) -> list[np.ndarray]:
    base = []
    for k in range(4):
        rot = np.rot90(mask, k)
        base.append(rot)
        base.append(np.fliplr(rot))
    out = []
    seen = set()
    for item in base:
        trimmed = _trim_mask(item)
        key = (trimmed.shape, trimmed.astype(np.uint8).tobytes())
        if key in seen:
            continue
        seen.add(key)
        out.append(trimmed)
    return out


def _canonical_shape(mask: np.ndarray):
    return min((m.shape, m.astype(np.uint8).tobytes()) for m in _dihedral_masks(mask))


def _most_common_color(grid: np.ndarray) -> int:
    return int(Counter(map(int, grid.ravel())).most_common(1)[0][0])


def _group_runs(flags: np.ndarray) -> list[tuple[int, int]]:
    runs = []
    start = None
    for idx, flag in enumerate(map(bool, flags)):
        if flag and start is None:
            start = idx
        elif not flag and start is not None:
            runs.append((start, idx - 1))
            start = None
    if start is not None:
        runs.append((start, len(flags) - 1))
    return runs


def _find_wall_bands(grid: np.ndarray) -> tuple[list[tuple[int, int]], list[tuple[int, int]]]:
    wall_mask = grid == WALL
    row_flags = wall_mask.sum(axis=1) > (grid.shape[1] // 2)
    col_flags = wall_mask.sum(axis=0) > (grid.shape[0] // 2)
    h_bands = _group_runs(row_flags)
    v_bands = _group_runs(col_flags)
    assert len(h_bands) >= 2 and len(v_bands) >= 2, "expected a wall-framed room layout"
    return h_bands, v_bands


def _room_intervals(bands: list[tuple[int, int]]) -> list[tuple[int, int]]:
    rooms = []
    for idx in range(len(bands) - 1):
        start = bands[idx][1] + 1
        end = bands[idx + 1][0] - 1
        assert start <= end, "room interval collapsed"
        rooms.append((start, end))
    return rooms


def _gap_infos(segment: np.ndarray, abs_r0: int, abs_c0: int) -> list[dict]:
    infos = []
    for coords in _components(segment != WALL):
        r0, c0, r1, c1 = _bbox(coords)
        mask = np.zeros((r1 - r0 + 1, c1 - c0 + 1), dtype=bool)
        mask[coords[:, 0] - r0, coords[:, 1] - c0] = True
        abs_rows = sorted({abs_r0 + int(r) for r in coords[:, 0]})
        abs_cols = sorted({abs_c0 + int(c) for c in coords[:, 1]})
        infos.append(
            {
                "mask": mask,
                "abs_rows": abs_rows,
                "abs_cols": abs_cols,
            }
        )
    return infos


def _extract_key(room: np.ndarray, bg: int):
    fg = (room != bg) & (room != WALL)
    if not np.any(fg):
        return None
    colors = {int(v) for v in np.unique(room[fg])}
    assert len(colors) == 1, "key sprite must be monochrome"
    comps = _components(fg)
    assert len(comps) == 1, "room may contain at most one key sprite"
    coords = comps[0]
    r0, c0, r1, c1 = _bbox(coords)
    mask = np.zeros((r1 - r0 + 1, c1 - c0 + 1), dtype=bool)
    mask[coords[:, 0] - r0, coords[:, 1] - c0] = True
    return {
        "color": next(iter(colors)),
        "shape": _canonical_shape(mask),
    }


def main(grid: np.ndarray) -> np.ndarray:
    g = np.asarray(grid, dtype=int)
    assert g.ndim == 2, "expected a 2D grid"
    bg = _most_common_color(g)
    h_bands, v_bands = _find_wall_bands(g)
    row_rooms = _room_intervals(h_bands)
    col_rooms = _room_intervals(v_bands)

    out = g.copy()
    for ri, (rr0, rr1) in enumerate(row_rooms):
        for ci, (cc0, cc1) in enumerate(col_rooms):
            key = _extract_key(g[rr0 : rr1 + 1, cc0 : cc1 + 1], bg)
            if key is None:
                continue

            candidates = []
            if ri > 0:
                tr0, tr1 = h_bands[ri]
                seg = g[tr0 : tr1 + 1, cc0 : cc1 + 1]
                for info in _gap_infos(seg, tr0, cc0):
                    info["side"] = "top"
                    info["target"] = (ri - 1, ci)
                    candidates.append(info)
            if ri + 1 < len(row_rooms):
                br0, br1 = h_bands[ri + 1]
                seg = g[br0 : br1 + 1, cc0 : cc1 + 1]
                for info in _gap_infos(seg, br0, cc0):
                    info["side"] = "bottom"
                    info["target"] = (ri + 1, ci)
                    candidates.append(info)
            if ci > 0:
                lc0, lc1 = v_bands[ci]
                seg = g[rr0 : rr1 + 1, lc0 : lc1 + 1]
                for info in _gap_infos(seg, rr0, lc0):
                    info["side"] = "left"
                    info["target"] = (ri, ci - 1)
                    candidates.append(info)
            if ci + 1 < len(col_rooms):
                rc0, rc1 = v_bands[ci + 1]
                seg = g[rr0 : rr1 + 1, rc0 : rc1 + 1]
                for info in _gap_infos(seg, rr0, rc0):
                    info["side"] = "right"
                    info["target"] = (ri, ci + 1)
                    candidates.append(info)

            matches = [cand for cand in candidates if _canonical_shape(cand["mask"]) == key["shape"]]
            assert len(matches) == 1, "each key must match exactly one wall gap"
            match = matches[0]
            tri, tci = match["target"]
            tr0, tr1 = row_rooms[tri]
            tc0, tc1 = col_rooms[tci]

            if match["side"] in {"top", "bottom"}:
                for col in match["abs_cols"]:
                    out[tr0 : tr1 + 1, col] = key["color"]
            else:
                for row in match["abs_rows"]:
                    out[row, tc0 : tc1 + 1] = key["color"]

    return out


def _build_axis(room_sizes: list[int]) -> tuple[list[tuple[int, int]], list[tuple[int, int]], int]:
    bands = [(0, WALL_THICKNESS - 1)]
    rooms = []
    pos = WALL_THICKNESS
    for size in room_sizes:
        rooms.append((pos, pos + size - 1))
        pos += size
        bands.append((pos, pos + WALL_THICKNESS - 1))
        pos += WALL_THICKNESS
    return bands, rooms, pos


def _paint_walls(grid: np.ndarray, h_bands: list[tuple[int, int]], v_bands: list[tuple[int, int]]):
    for r0, r1 in h_bands:
        grid[r0 : r1 + 1, :] = WALL
    for c0, c1 in v_bands:
        grid[:, c0 : c1 + 1] = WALL


def _rand_variant(rng: random.Random, mask: np.ndarray, max_h: int, max_w: int) -> np.ndarray:
    options = [m for m in _dihedral_masks(mask) if m.shape[0] <= max_h and m.shape[1] <= max_w]
    assert options, "shape does not fit requested bounds"
    return rng.choice(options)


def _place_gap(
    grid: np.ndarray,
    rng: random.Random,
    shape: np.ndarray,
    side: str,
    room_rows: tuple[int, int],
    room_cols: tuple[int, int],
    bg: int,
):
    rr0, rr1 = room_rows
    cc0, cc1 = room_cols
    if side == "top":
        band_r0, band_r1 = rr0 - WALL_THICKNESS, rr0 - 1
        variant = _rand_variant(rng, shape, WALL_THICKNESS, cc1 - cc0 - 1)
        off_c = rng.randint(1, (cc1 - cc0 + 1) - variant.shape[1] - 1)
        patch = grid[band_r0 : band_r0 + variant.shape[0], cc0 + off_c : cc0 + off_c + variant.shape[1]]
        patch[variant] = bg
        coords = np.argwhere(variant)
        abs_rows = sorted({band_r0 + int(r) for r in coords[:, 0]})
        abs_cols = sorted({cc0 + off_c + int(c) for c in coords[:, 1]})
    elif side == "bottom":
        band_r0, band_r1 = rr1 + 1, rr1 + WALL_THICKNESS
        variant = _rand_variant(rng, shape, WALL_THICKNESS, cc1 - cc0 - 1)
        off_c = rng.randint(1, (cc1 - cc0 + 1) - variant.shape[1] - 1)
        patch = grid[band_r0 : band_r0 + variant.shape[0], cc0 + off_c : cc0 + off_c + variant.shape[1]]
        patch[variant] = bg
        coords = np.argwhere(variant)
        abs_rows = sorted({band_r0 + int(r) for r in coords[:, 0]})
        abs_cols = sorted({cc0 + off_c + int(c) for c in coords[:, 1]})
    elif side == "left":
        band_c0, band_c1 = cc0 - WALL_THICKNESS, cc0 - 1
        variant = _rand_variant(rng, shape, rr1 - rr0 - 1, WALL_THICKNESS)
        off_r = rng.randint(1, (rr1 - rr0 + 1) - variant.shape[0] - 1)
        patch = grid[rr0 + off_r : rr0 + off_r + variant.shape[0], band_c0 : band_c0 + variant.shape[1]]
        patch[variant] = bg
        coords = np.argwhere(variant)
        abs_rows = sorted({rr0 + off_r + int(r) for r in coords[:, 0]})
        abs_cols = sorted({band_c0 + int(c) for c in coords[:, 1]})
    else:
        assert side == "right"
        band_c0, band_c1 = cc1 + 1, cc1 + WALL_THICKNESS
        variant = _rand_variant(rng, shape, rr1 - rr0 - 1, WALL_THICKNESS)
        off_r = rng.randint(1, (rr1 - rr0 + 1) - variant.shape[0] - 1)
        patch = grid[rr0 + off_r : rr0 + off_r + variant.shape[0], band_c0 : band_c0 + variant.shape[1]]
        patch[variant] = bg
        coords = np.argwhere(variant)
        abs_rows = sorted({rr0 + off_r + int(r) for r in coords[:, 0]})
        abs_cols = sorted({band_c0 + int(c) for c in coords[:, 1]})
    return {"rows": abs_rows, "cols": abs_cols}


def _place_key(
    grid: np.ndarray,
    rng: random.Random,
    shape: np.ndarray,
    color: int,
    room_rows: tuple[int, int],
    room_cols: tuple[int, int],
):
    rr0, rr1 = room_rows
    cc0, cc1 = room_cols
    variant = _rand_variant(rng, shape, rr1 - rr0 - 1, cc1 - cc0 - 1)
    off_r = rng.randint(1, (rr1 - rr0 + 1) - variant.shape[0] - 1)
    off_c = rng.randint(1, (cc1 - cc0 + 1) - variant.shape[1] - 1)
    patch = grid[rr0 + off_r : rr0 + off_r + variant.shape[0], cc0 + off_c : cc0 + off_c + variant.shape[1]]
    patch[variant] = color


def _all_directed_edges(nr: int, nc: int) -> list[tuple[tuple[int, int], tuple[int, int], str]]:
    edges = []
    for r in range(nr):
        for c in range(nc):
            if r > 0:
                edges.append(((r, c), (r - 1, c), "top"))
            if r + 1 < nr:
                edges.append(((r, c), (r + 1, c), "bottom"))
            if c > 0:
                edges.append(((r, c), (r, c - 1), "left"))
            if c + 1 < nc:
                edges.append(((r, c), (r, c + 1), "right"))
    return edges


def _shared_wall_id(src: tuple[int, int], side: str):
    r, c = src
    if side == "top":
        return ("h", r, c)
    if side == "bottom":
        return ("h", r + 1, c)
    if side == "left":
        return ("v", r, c)
    return ("v", r, c + 1)


def _build_expected(
    grid: np.ndarray,
    row_rooms: list[tuple[int, int]],
    col_rooms: list[tuple[int, int]],
    active: list[dict],
) -> np.ndarray:
    out = grid.copy()
    for item in active:
        trg_r, trg_c = item["target"]
        rr0, rr1 = row_rooms[trg_r]
        cc0, cc1 = col_rooms[trg_c]
        color = item["color"]
        if item["side"] in {"top", "bottom"}:
            for col in item["gap"]["cols"]:
                out[rr0 : rr1 + 1, col] = color
        else:
            for row in item["gap"]["rows"]:
                out[row, cc0 : cc1 + 1] = color
    return out


def _gen_attempt(rng: random.Random) -> np.ndarray | None:
    bg = rng.choice([0, 1, 2, 3, 4, 5, 6, 9])
    key_colors = [c for c in range(10) if c not in {bg, WALL}]

    n_rows = rng.randint(*ROOM_ROWS_RANGE)
    n_cols = rng.randint(*ROOM_COLS_RANGE)
    room_h = [rng.randint(*ROOM_SIZE_RANGE) for _ in range(n_rows)]
    room_w = [rng.randint(*ROOM_SIZE_RANGE) for _ in range(n_cols)]

    h_bands, row_rooms, height = _build_axis(room_h)
    v_bands, col_rooms, width = _build_axis(room_w)
    grid = np.full((height, width), bg, dtype=int)
    _paint_walls(grid, h_bands, v_bands)

    all_edges = _all_directed_edges(n_rows, n_cols)
    rng.shuffle(all_edges)
    max_keys = min(KEY_COUNT_MAX, len(key_colors), (n_rows * n_cols) // 2, len(all_edges))
    if max_keys <= 0:
        return None
    desired = rng.randint(1, max_keys)

    active = []
    busy_rooms = set()
    used_walls = set()
    for src, dst, side in all_edges:
        wall_id = _shared_wall_id(src, side)
        if src in busy_rooms or dst in busy_rooms or wall_id in used_walls:
            continue
        active.append({"source": src, "target": dst, "side": side, "wall_id": wall_id})
        busy_rooms.add(src)
        busy_rooms.add(dst)
        used_walls.add(wall_id)
        if len(active) == desired:
            break
    if len(active) != desired:
        return None

    rng.shuffle(key_colors)
    room_key_shape = {}
    for idx, item in enumerate(active):
        shape = rng.choice(BASE_SHAPES)
        color = key_colors[idx]
        item["shape"] = shape
        item["color"] = color
        room_key_shape[item["source"]] = _canonical_shape(shape)

    for item in active:
        sr, sc = item["source"]
        rows = row_rooms[sr]
        cols = col_rooms[sc]
        _place_key(grid, rng, item["shape"], item["color"], rows, cols)
        item["gap"] = _place_gap(grid, rng, item["shape"], item["side"], rows, cols, bg)

    extra_candidates = []
    for src, dst, side in all_edges:
        wall_id = _shared_wall_id(src, side)
        if wall_id in used_walls:
            continue
        extra_candidates.append((src, side, wall_id))
    rng.shuffle(extra_candidates)
    extra_count = rng.randint(0, min(4, len(extra_candidates)))
    for src, side, wall_id in extra_candidates[:extra_count]:
        forbidden = set()
        if src in room_key_shape:
            forbidden.add(room_key_shape[src])
        dr, dc = DIRS[side]
        other = (src[0] + dr, src[1] + dc)
        if other in room_key_shape:
            forbidden.add(room_key_shape[other])
        options = [shape for shape in BASE_SHAPES if _canonical_shape(shape) not in forbidden]
        if not options:
            continue
        sr, sc = src
        _place_gap(grid, rng, rng.choice(options), side, row_rooms[sr], col_rooms[sc], bg)
        used_walls.add(wall_id)

    expected = _build_expected(grid, row_rooms, col_rooms, active)
    try:
        got = main(grid)
    except AssertionError:
        return None
    if not np.array_equal(got, expected):
        return None
    return grid


def generate_input() -> np.ndarray:
    rng = random.Random()
    for _ in range(MAX_RETRIES):
        grid = _gen_attempt(rng)
        if grid is not None:
            return grid
    raise RuntimeError("failed to generate a valid input")
