# Combo variant of 53fb4810 (index 15)
# combo_spec: D3/1+D5/0
# Name: 大瓦片颜色循环

from __future__ import annotations

import numpy as np

from arc_utils import bbox_coords, crop_bbox, get_bg, label_mask


# ---------------------------------------------------------------------------
# Design analysis for the combo variant
#
# a) Concrete behavior trace
#    Original 53fb4810 uses a tiny attached seed and simply repeats the same
#    colors forever. This combo changes both the period and the coloration:
#      - the seed is now a large filled rectangle such as 3x2, 3x3, 4x3
#      - every whole-tile step applies one global color permutation
#    Example with a 3x3 seed and cycle A->B->C->D->A:
#      seed0 = [[A,B,C],    chunk1 = map(seed0),    chunk2 = map(chunk1)
#               [D,A,B],              [[B,C,D],               [[C,D,A],
#               [C,D,A]]               [A,B,C],                [B,C,D],
#                                      [D,A,B]]                [A,B,C]]
#    The geometric rule "project away from the touched arm" is unchanged, but
#    the clue ray now teaches both the tile size and the recoloring cycle.
#
# b) Complexity calibration
#    If the clue tile were only 2x2, or if the cycle used too few colors, the
#    fusion would visually collapse back toward the original task or the single
#    D3 variant. To keep the change obvious while still inferable, generation
#    uses:
#      - only anchors whose nozzle width is 3
#      - large seeds with area >= 6 and 3-4 visible colors
#      - a single long clue ray built from a 3x3 seed using all 4 cycle colors
#      - a 4-color global permutation shared by every seed in the grid
#    This keeps the number of free degrees small enough for humans to compare
#    repeated chunks, while making the fusion visually distinct.
#
# c) Hidden assumptions checked against the original solver
#    - `crop_bbox(component)` no longer identifies the period for the long clue
#      ray, because its bbox spans many repeated large tiles. The combo solver
#      reconstructs the seed bbox from the plus nozzle and tests candidate
#      along-axis lengths.
#    - The old extension helper assumed color-stationary repetition. Here each
#      tile step must advance the whole rectangle through a learned color map.
#    - The original tiny-seed logic tolerated ambiguous period lengths because
#      the choices were only 1 or 2. With large clue rays, ambiguity would be a
#      real risk, so the generator forces the clue seed to have along-size 3 and
#      a total clue extent divisible only by that candidate among {2,3,4}.
#    - The solver now assumes exactly one elongated adjacent component reveals a
#      consistent non-identity global map. The generator constructs that clue.
#
# d) Placement legality
#    Output is still generated first. Each plus and its full ray are placed only
#    if the projected cells stay in bounds and do not overlap other pluses or
#    rays. The clue ray is placed first so its length and divisibility
#    constraints can be enforced; other rays are then added around it.
#
# e) Visual observability
#    A human can inspect the single long clue ray and see two independent facts:
#      - the repeated unit is a large 3-wide textured rectangle
#      - adjacent chunks keep the same geometry but all colors advance together
#    The non-clue seeds are compact but share the same palette, so the learned
#    cycle clearly transfers from the clue to the other anchors.
#
# f) Visual cleanliness
#    Rays are collision-free and pluses are kept away from existing rays. Only
#    one clue ray is long in the input, which limits clutter even though the
#    repeated tiles are larger and more colorful than in the base task.
#
# g) Color-change inferability
#    The generator uses exactly four cycle colors shared across the grid. The
#    clue tile contains all four, and at least five full clue chunks are shown,
#    so every edge of the cycle is directly visible somewhere in the input.
# ---------------------------------------------------------------------------


_DIRECTIONS = [(1, 0), (-1, 0), (0, 1), (0, -1)]
_SEED_ALONG_CHOICES = [2, 3, 4]
_MAX_SEED_ALONG = max(_SEED_ALONG_CHOICES)
_CLUE_MIN_CHUNKS = 5


def detect_dominant_big_component_color(
    grid: np.ndarray,
    *,
    bg: int | None = None,
    min_h: int = 3,
    min_w: int = 3,
    connectivity: int = 4,
) -> int | None:
    """Choose the non-background color with the most large components."""
    if bg is None:
        bg = get_bg(grid)
    bg = int(bg)

    best_key: tuple[int, int] | None = None
    best_col: int | None = None

    for col in sorted(set(map(int, grid.flatten())) - {int(bg)}):
        lab, n = label_mask(grid == int(col), connectivity=int(connectivity))
        big_areas: list[int] = []
        for k in range(1, int(n) + 1):
            coords = np.argwhere(lab == int(k))
            if coords.size == 0:
                continue
            r0, c0 = coords.min(axis=0)
            r1, c1 = coords.max(axis=0)
            h = int(r1 - r0 + 1)
            w = int(c1 - c0 + 1)
            if h >= int(min_h) and w >= int(min_w):
                big_areas.append(int(coords.shape[0]))
        if not big_areas:
            continue
        key = (int(len(big_areas)), int(sum(big_areas)))
        if best_key is None or key > best_key:
            best_key = key
            best_col = int(col)

    return int(best_col) if best_col is not None else None


def _apply_color_map(tile: np.ndarray, color_map: dict[int, int]) -> np.ndarray:
    """Apply one whole-tile permutation step."""
    if not color_map:
        return np.array(tile, dtype=int, copy=True)
    src = np.array(tile, dtype=int, copy=False)
    out = src.copy()
    for a, b in color_map.items():
        out[src == int(a)] = int(b)
    return out


def extend_color_cycle_tile(
    out: np.ndarray,
    seed_tile: np.ndarray,
    r0: int,
    c0: int,
    direction: tuple[int, int],
    *,
    bg: int,
    protect: set[int] | None = None,
    color_map: dict[int, int] | None = None,
) -> None:
    """Extend a large seed outward, recoloring each full-tile step."""
    protect = set() if protect is None else set(map(int, protect))
    color_map = {} if color_map is None else {int(k): int(v) for k, v in color_map.items()}
    H, W = out.shape
    h, w = seed_tile.shape
    dr, dc = map(int, direction)
    bg = int(bg)

    def put(rr: int, cc: int, v: int) -> None:
        if not (0 <= rr < H and 0 <= cc < W):
            return
        if int(v) == int(bg):
            return
        if int(out[rr, cc]) in protect:
            return
        out[rr, cc] = int(v)

    rr = int(r0 + dr * h) if dr != 0 else int(r0)
    cc = int(c0 + dc * w) if dc != 0 else int(c0)
    cur = _apply_color_map(seed_tile, color_map)

    while rr < H and rr + h > 0 and cc < W and cc + w > 0:
        for i in range(h):
            for j in range(w):
                put(int(rr + i), int(cc + j), int(cur[i, j]))
        rr += int(dr * h)
        cc += int(dc * w)
        cur = _apply_color_map(cur, color_map)


def _component_direction(
    pts: np.ndarray,
    plus_mask: np.ndarray,
    H: int,
    W: int,
) -> tuple[int, int] | None:
    dirs: set[tuple[int, int]] = set()
    for pr, pc in pts:
        pr = int(pr)
        pc = int(pc)
        for dr, dc in _DIRECTIONS:
            rr = pr + dr
            cc = pc + dc
            if 0 <= rr < H and 0 <= cc < W and bool(plus_mask[rr, cc]):
                dirs.add((-int(dr), -int(dc)))
    if len(dirs) != 1:
        return None
    return next(iter(dirs))


def _nozzle_axis_positions(
    plus_mask: np.ndarray,
    plus_coords: np.ndarray,
    direction: tuple[int, int],
) -> list[int]:
    pr0, pc0 = plus_coords.min(axis=0)
    pr1, pc1 = plus_coords.max(axis=0)
    dr, dc = map(int, direction)
    if dc == 0:
        nozzle_row = int(pr1) if dr == 1 else int(pr0)
        return [int(c) for c in range(int(pc0), int(pc1) + 1) if bool(plus_mask[nozzle_row, c])]
    nozzle_col = int(pc1) if dc == 1 else int(pc0)
    return [int(r) for r in range(int(pr0), int(pr1) + 1) if bool(plus_mask[r, nozzle_col])]


def _component_along_extent(pts: np.ndarray, direction: tuple[int, int]) -> int:
    r0, c0, r1, c1 = bbox_coords(pts)
    _dr, dc = map(int, direction)
    return int(r1 - r0 + 1) if dc == 0 else int(c1 - c0 + 1)


def _seed_bbox_from_plus(
    plus_mask: np.ndarray,
    plus_coords: np.ndarray,
    direction: tuple[int, int],
    along: int,
) -> tuple[int, int, int, int]:
    pr0, pc0 = plus_coords.min(axis=0)
    pr1, pc1 = plus_coords.max(axis=0)
    dr, dc = map(int, direction)
    along = int(along)

    if dc == 0:
        nozzle_cols = _nozzle_axis_positions(plus_mask, plus_coords, direction)
        c0 = int(min(nozzle_cols))
        c1 = int(max(nozzle_cols))
        r0 = int(pr1 + 1) if dr == 1 else int(pr0 - along)
        r1 = int(r0 + along - 1)
        return int(r0), int(c0), int(r1), int(c1)

    nozzle_rows = _nozzle_axis_positions(plus_mask, plus_coords, direction)
    r0 = int(min(nozzle_rows))
    r1 = int(max(nozzle_rows))
    c0 = int(pc1 + 1) if dc == 1 else int(pc0 - along)
    c1 = int(c0 + along - 1)
    return int(r0), int(c0), int(r1), int(c1)


def _seed_matches_nozzle(
    plus_mask: np.ndarray,
    plus_coords: np.ndarray,
    seed_bbox: tuple[int, int, int, int],
    direction: tuple[int, int],
) -> bool:
    r0, c0, r1, c1 = map(int, seed_bbox)
    axis = _nozzle_axis_positions(plus_mask, plus_coords, direction)
    if direction[1] == 0:
        return list(range(int(c0), int(c1) + 1)) == axis
    return list(range(int(r0), int(r1) + 1)) == axis


def _extract_full_chunks(
    grid: np.ndarray,
    r0: int,
    c0: int,
    shape: tuple[int, int],
    direction: tuple[int, int],
) -> list[np.ndarray]:
    H, W = grid.shape
    h, w = map(int, shape)
    dr, dc = map(int, direction)
    rr = int(r0)
    cc = int(c0)
    chunks: list[np.ndarray] = []

    while 0 <= rr and rr + h <= H and 0 <= cc and cc + w <= W:
        chunks.append(np.array(grid[rr : rr + h, cc : cc + w], dtype=int, copy=True))
        rr += int(dr * h)
        cc += int(dc * w)

    return chunks


def _infer_color_map_from_chunks(
    chunks: list[np.ndarray],
    *,
    bg: int,
    plus_col: int,
) -> tuple[dict[int, int] | None, int]:
    mapping: dict[int, int] = {}
    evidence = 0

    for a, b in zip(chunks, chunks[1:]):
        for va, vb in zip(a.flatten(), b.flatten()):
            va = int(va)
            vb = int(vb)
            if va in {int(bg), int(plus_col)} or vb in {int(bg), int(plus_col)}:
                return None, 0
            evidence += 1
            prev = mapping.get(int(va))
            if prev is not None and int(prev) != int(vb):
                return None, 0
            mapping[int(va)] = int(vb)

    if evidence == 0:
        return None, 0
    if len(mapping) < 3:
        return None, 0
    if any(int(k) == int(v) for k, v in mapping.items()):
        return None, 0
    image = set(map(int, mapping.values()))
    if image != set(map(int, mapping.keys())):
        return None, 0
    return mapping, int(evidence)


def _interpret_component(
    grid: np.ndarray,
    pts: np.ndarray,
    plus_mask: np.ndarray,
    plus_coords: np.ndarray,
    direction: tuple[int, int],
    *,
    bg: int,
    plus_col: int,
) -> dict | None:
    H, W = grid.shape
    comp_extent = int(_component_along_extent(pts, direction))
    candidates = [comp_extent] if comp_extent <= _MAX_SEED_ALONG else [
        int(a)
        for a in _SEED_ALONG_CHOICES
        if comp_extent % int(a) == 0 and comp_extent // int(a) >= _CLUE_MIN_CHUNKS
    ]
    if not candidates:
        return None

    best: dict | None = None
    for along in candidates:
        seed_bbox = _seed_bbox_from_plus(plus_mask, plus_coords, direction, int(along))
        sr0, sc0, sr1, sc1 = map(int, seed_bbox)
        if not (0 <= sr0 <= sr1 < H and 0 <= sc0 <= sc1 < W):
            continue
        if not _seed_matches_nozzle(plus_mask, plus_coords, seed_bbox, direction):
            continue

        tile = crop_bbox(grid, seed_bbox)
        if np.any(tile == int(bg)) or np.any(tile == int(plus_col)):
            continue

        info = {
            "direction": tuple(map(int, direction)),
            "seed_bbox": seed_bbox,
            "tile": np.array(tile, dtype=int, copy=True),
            "comp_extent": int(comp_extent),
            "along": int(along),
            "chunks": 1,
            "mapping": None,
            "score": (-1, -1, -1),
        }

        if comp_extent > _MAX_SEED_ALONG:
            chunks = _extract_full_chunks(
                grid,
                int(sr0),
                int(sc0),
                (int(sr1 - sr0 + 1), int(sc1 - sc0 + 1)),
                direction,
            )
            mapping, evidence = _infer_color_map_from_chunks(
                chunks,
                bg=int(bg),
                plus_col=int(plus_col),
            )
            if mapping is None:
                continue
            info["chunks"] = int(len(chunks))
            info["mapping"] = mapping
            info["score"] = (int(len(mapping)), int(evidence), -int(along))
        else:
            info["score"] = (0, 0, -int(along))

        if best is None or tuple(info["score"]) > tuple(best["score"]):
            best = info

    return best


def _infer_global_color_map(infos: list[dict]) -> dict[int, int]:
    best: dict | None = None
    for info in infos:
        if info.get("mapping") is None:
            continue
        if best is None or tuple(info["score"]) > tuple(best["score"]):
            best = info
    if best is None:
        return {}
    return {int(k): int(v) for k, v in dict(best["mapping"]).items()}


def main(grid: np.ndarray) -> np.ndarray:
    """
    Extend each large marker rectangle away from its '+' anchor while cycling its colors.

    Input structure:
    - The background is the most frequent color.
    - One dominant non-background color forms several identical plus-like anchors.
    - Every anchor has one adjacent large rectangular seed; exactly one of those seeds is
      already expanded into a long multi-chunk clue ray.

    Rule:
    - Determine which side of each '+' the seed touches and treat that side as the ray
      direction.
    - Use the long clue ray to recover the seed size and the global whole-tile color
      permutation seen between consecutive chunks.
    - Repeat every seed rectangle outward to the boundary; each new tile copy advances one
      step in the learned color cycle, and '+' cells are never overwritten.

    Key insight:
    The clue ray simultaneously encodes two hidden quantities: the period is a large
    rectangle rather than the whole elongated component, and the period is recolored by the
    same permutation at every step. Solving the task means separating geometry from color
    evolution and reusing both on the compact seeds.
    """
    g = np.array(grid, dtype=int)
    out = g.copy()
    H, W = g.shape

    bg = int(get_bg(g))
    plus_col = detect_dominant_big_component_color(g, bg=bg, min_h=3, min_w=3, connectivity=4)
    if plus_col is None:
        return out

    plus_lab, pn = label_mask(g == int(plus_col), connectivity=4)
    non_mask = (g != int(bg)) & (g != int(plus_col))
    mark_lab, _ = label_mask(non_mask, connectivity=4)

    infos: list[dict] = []
    for pk in range(1, int(pn) + 1):
        plus_mask = plus_lab == int(pk)
        plus_coords = np.argwhere(plus_mask)
        if plus_coords.size == 0:
            continue

        adj_ids: set[int] = set()
        for r, c in plus_coords:
            r = int(r)
            c = int(c)
            for dr, dc in _DIRECTIONS:
                rr = r + dr
                cc = c + dc
                if 0 <= rr < H and 0 <= cc < W and int(mark_lab[rr, cc]) > 0:
                    adj_ids.add(int(mark_lab[rr, cc]))

        for cid in sorted(adj_ids):
            pts = np.argwhere(mark_lab == int(cid))
            if pts.size == 0:
                continue
            direction = _component_direction(pts, plus_mask, H, W)
            if direction is None:
                continue
            info = _interpret_component(
                g,
                pts,
                plus_mask,
                plus_coords,
                direction,
                bg=int(bg),
                plus_col=int(plus_col),
            )
            if info is not None:
                infos.append(info)

    color_map = _infer_global_color_map(infos)

    for info in infos:
        sr0, sc0, _sr1, _sc1 = info["seed_bbox"]
        extend_color_cycle_tile(
            out,
            info["tile"],
            int(sr0),
            int(sc0),
            tuple(map(int, info["direction"])),
            bg=int(bg),
            protect={int(plus_col)},
            color_map=color_map,
        )

    return out


# --------------- generator ---------------

_H_RANGE = (24, 35)
_W_RANGE = (24, 35)
_N_PLUS_RANGE = (2, 5)
_SEED_MIN_COLORS = 3
_SEED_MAX_COLORS = 4
_MIN_RAY_UNITS = 2
_CLUE_ALONG = 3
_CLUE_UNITS_CHOICES = [5, 7]
_PLACEMENT_TRIES = 220
_GEN_RETRIES = 700
_CYCLE_LEN = 4

_PLUS_SHAPE = np.array(
    [
        [0, 1, 1, 1, 0],
        [1, 1, 1, 1, 1],
        [1, 1, 1, 1, 1],
        [1, 1, 1, 1, 1],
        [0, 1, 1, 1, 0],
    ],
    dtype=int,
)


def _nozzle_span(plus_mask: np.ndarray, direction: tuple[int, int]) -> tuple[int, int]:
    """Return the inclusive cross-axis span of the chosen nozzle."""
    dr, dc = map(int, direction)
    if dc == 0:
        row = plus_mask.shape[0] - 1 if dr == 1 else 0
        cols = np.where(plus_mask[row])[0]
        return int(cols[0]), int(cols[-1])
    col = plus_mask.shape[1] - 1 if dc == 1 else 0
    rows = np.where(plus_mask[:, col])[0]
    return int(rows[0]), int(rows[-1])


def _tile_has_rich_texture(tile: np.ndarray) -> bool:
    """Reject seeds that would visually collapse into a simple stripe."""
    colors = set(map(int, tile.flatten()))
    if len(colors) < _SEED_MIN_COLORS:
        return False
    if len({tuple(map(int, row)) for row in tile}) < 2:
        return False
    if len({tuple(map(int, tile[:, j])) for j in range(tile.shape[1])}) < 2:
        return False
    if max(np.bincount(tile.flatten(), minlength=10)) >= tile.size - 1:
        return False
    return True


def _clue_tile_is_good(tile: np.ndarray, cycle_colors: list[int], color_map: dict[int, int]) -> bool:
    """The clue must show the full 4-color cycle clearly."""
    if tile.shape != (3, 3):
        return False
    if set(map(int, tile.flatten())) != set(map(int, cycle_colors)):
        return False
    if len({tuple(map(int, row)) for row in tile}) < 3:
        return False
    if len({tuple(map(int, tile[:, j])) for j in range(tile.shape[1])}) < 3:
        return False

    nxt = _apply_color_map(tile, color_map)
    if np.array_equal(tile, nxt):
        return False
    if len({tuple(map(int, row)) for row in nxt}) < 3:
        return False
    return True


def _build_seed_tile(
    plus_mask: np.ndarray,
    direction: tuple[int, int],
    cycle_colors: list[int],
    color_map: dict[int, int],
    rng: np.random.Generator,
    *,
    force_clue: bool = False,
) -> np.ndarray | None:
    """Build a large rectangular seed whose cross-axis matches the nozzle."""
    s, e = _nozzle_span(plus_mask, direction)
    cross = int(e - s + 1)
    if cross != 3:
        return None

    if force_clue:
        for _ in range(60):
            vals = list(map(int, cycle_colors)) + list(map(int, rng.choice(cycle_colors, size=5, replace=True)))
            rng.shuffle(vals)
            tile = np.array(vals, dtype=int).reshape(3, 3)
            if _tile_has_rich_texture(tile) and _clue_tile_is_good(tile, cycle_colors, color_map):
                return tile
        return None

    for _ in range(40):
        along = int(rng.choice(_SEED_ALONG_CHOICES))
        dr, dc = map(int, direction)
        th, tw = (along, cross) if dc == 0 else (cross, along)
        max_k = min(_SEED_MAX_COLORS, len(cycle_colors), th * tw)
        if max_k < _SEED_MIN_COLORS:
            return None
        n_colors = int(rng.integers(_SEED_MIN_COLORS, max_k + 1))
        chosen = list(map(int, rng.choice(cycle_colors, size=n_colors, replace=False)))
        vals = chosen + list(map(int, rng.choice(chosen, size=th * tw - n_colors, replace=True)))
        rng.shuffle(vals)
        tile = np.array(vals, dtype=int).reshape(th, tw)
        if _tile_has_rich_texture(tile):
            return tile
    return None


def _ray_full_units(
    direction: tuple[int, int],
    tile: np.ndarray,
    r0: int,
    c0: int,
    ph: int,
    pw: int,
    H: int,
    W: int,
) -> tuple[int, int]:
    """Return (full_units, remainder_cells) from seed start to the boundary."""
    dr, dc = map(int, direction)
    th, tw = tile.shape
    if dr == 1:
        avail = int(H - (r0 + ph))
        return int(avail // th), int(avail % th)
    if dr == -1:
        avail = int(r0)
        return int(avail // th), int(avail % th)
    if dc == 1:
        avail = int(W - (c0 + pw))
        return int(avail // tw), int(avail % tw)
    avail = int(c0)
    return int(avail // tw), int(avail % tw)


def _project_cycle_coords(
    H: int,
    W: int,
    plus_bbox: tuple[int, int, int, int],
    plus_mask: np.ndarray,
    direction: tuple[int, int],
    tile: np.ndarray,
    color_map: dict[int, int],
) -> tuple[list[tuple[int, int, int]], int, int]:
    """Return all cells of the projected color-cycling ray and the seed origin."""
    pr0, pc0, pr1, pc1 = map(int, plus_bbox)
    dr, dc = map(int, direction)
    th, tw = tile.shape
    s, _ = _nozzle_span(plus_mask, direction)

    if dc == 0:
        c0 = int(pc0 + s)
        r0 = int(pr1 + 1) if dr == 1 else int(pr0 - th)
    else:
        r0 = int(pr0 + s)
        c0 = int(pc1 + 1) if dc == 1 else int(pc0 - tw)

    cells: list[tuple[int, int, int]] = []
    rr = int(r0)
    cc = int(c0)
    cur = np.array(tile, dtype=int, copy=True)

    while rr < H and rr + th > 0 and cc < W and cc + tw > 0:
        for i in range(th):
            for j in range(tw):
                rrr = int(rr + i)
                ccc = int(cc + j)
                if 0 <= rrr < H and 0 <= ccc < W:
                    cells.append((int(rrr), int(ccc), int(cur[i, j])))
        rr += int(dr * th)
        cc += int(dc * tw)
        cur = _apply_color_map(cur, color_map)

    return cells, int(r0), int(c0)


def _plus_touches_existing_ray(
    plus_cells: list[tuple[int, int]],
    occ: np.ndarray,
) -> bool:
    """Reject plus placements that would sit directly next to an existing ray."""
    H, W = occ.shape
    for r, c in plus_cells:
        for rr, cc in ((r, c), (r - 1, c), (r + 1, c), (r, c - 1), (r, c + 1)):
            if 0 <= rr < H and 0 <= cc < W and int(occ[rr, cc]) == 2:
                return True
    return False


def _place_plus_and_ray(
    rng: np.random.Generator,
    cycle_colors: list[int],
    color_map: dict[int, int],
    H: int,
    W: int,
    occ: np.ndarray,
    plus_occ: np.ndarray,
    out: np.ndarray,
    plus_color: int,
    *,
    force_clue: bool = False,
    forced_direction: tuple[int, int] | None = None,
) -> dict | None:
    """Place one plus with its full output ray."""
    ph, pw = _PLUS_SHAPE.shape

    for _ in range(_PLACEMENT_TRIES):
        r0 = int(rng.integers(1, H - ph))
        c0 = int(rng.integers(1, W - pw))
        plus_cells = [
            (int(r0 + dr), int(c0 + dc))
            for dr in range(ph)
            for dc in range(pw)
            if int(_PLUS_SHAPE[dr, dc]) == 1
        ]

        if any(int(occ[r, c]) != 0 for r, c in plus_cells):
            continue
        if any(int(plus_occ[r, c]) == 1 for r, c in plus_cells):
            continue
        if _plus_touches_existing_ray(plus_cells, occ):
            continue

        dirs = [tuple(map(int, forced_direction))] if forced_direction is not None else list(_DIRECTIONS)
        if forced_direction is None:
            rng.shuffle(dirs)

        for direction in dirs:
            tile = _build_seed_tile(
                _PLUS_SHAPE,
                direction,
                cycle_colors,
                color_map,
                rng,
                force_clue=bool(force_clue),
            )
            if tile is None:
                continue

            units, remainder = _ray_full_units(direction, tile, r0, c0, ph, pw, H, W)
            if force_clue:
                if int(tile.shape[0 if direction[1] == 0 else 1]) != _CLUE_ALONG:
                    continue
                if remainder != 0:
                    continue
                if units not in _CLUE_UNITS_CHOICES:
                    continue
            else:
                if units < _MIN_RAY_UNITS:
                    continue

            cells, sr0, sc0 = _project_cycle_coords(
                H,
                W,
                (r0, c0, r0 + ph - 1, c0 + pw - 1),
                _PLUS_SHAPE,
                direction,
                tile,
                color_map,
            )
            if any(int(occ[r, c]) != 0 for r, c, _v in cells if 0 <= r < H and 0 <= c < W):
                continue
            if any(int(plus_occ[r, c]) == 1 for r, c, _v in cells if 0 <= r < H and 0 <= c < W):
                continue

            for r, c in plus_cells:
                occ[r, c] = 1
                out[r, c] = int(plus_color)
                for dr2, dc2 in _DIRECTIONS:
                    rr = int(r + dr2)
                    cc = int(c + dc2)
                    if 0 <= rr < H and 0 <= cc < W:
                        plus_occ[rr, cc] = 1
            for r, c, v in cells:
                occ[r, c] = 2
                out[r, c] = int(v)

            return {
                "tile": np.array(tile, dtype=int, copy=True),
                "cells": cells,
                "seed_r0": int(sr0),
                "seed_c0": int(sc0),
                "units": int(units),
                "direction": tuple(map(int, direction)),
                "is_clue": bool(force_clue),
            }

    return None


def _build_input_from_output(
    out: np.ndarray,
    pluses: list[dict],
    bg: int,
    keep_idx: int,
) -> np.ndarray:
    """Keep one full clue ray and collapse the others back to their compact seeds."""
    inp = np.array(out, dtype=int, copy=True)
    for i, p in enumerate(pluses):
        if int(i) == int(keep_idx):
            continue
        th, tw = p["tile"].shape
        sr0 = int(p["seed_r0"])
        sc0 = int(p["seed_c0"])
        seed_set = {(int(sr0 + ii), int(sc0 + jj)) for ii in range(th) for jj in range(tw)}
        for r, c, _v in p["cells"]:
            if (int(r), int(c)) not in seed_set:
                inp[int(r), int(c)] = int(bg)
    return inp


def _gen_attempt(rng: np.random.Generator) -> np.ndarray | None:
    H = int(rng.integers(*_H_RANGE))
    W = int(rng.integers(*_W_RANGE))

    bg = int(rng.integers(0, 10))
    others = [int(c) for c in range(10) if int(c) != int(bg)]
    plus_color = int(rng.choice(others))
    cycle_colors = list(
        map(int, rng.choice([c for c in others if int(c) != int(plus_color)], size=_CYCLE_LEN, replace=False))
    )
    rng.shuffle(cycle_colors)
    color_map = {
        int(cycle_colors[i]): int(cycle_colors[(i + 1) % _CYCLE_LEN])
        for i in range(_CYCLE_LEN)
    }

    N = int(rng.integers(*_N_PLUS_RANGE))

    out = np.full((H, W), bg, dtype=int)
    occ = np.zeros((H, W), dtype=int)
    plus_occ = np.zeros((H, W), dtype=int)

    pluses: list[dict] = []

    clue = _place_plus_and_ray(
        rng=rng,
        cycle_colors=cycle_colors,
        color_map=color_map,
        H=H,
        W=W,
        occ=occ,
        plus_occ=plus_occ,
        out=out,
        plus_color=plus_color,
        force_clue=True,
        forced_direction=_DIRECTIONS[int(rng.integers(0, len(_DIRECTIONS)))],
    )
    if clue is None:
        return None
    pluses.append(clue)

    for _ in range(N - 1):
        info = _place_plus_and_ray(
            rng=rng,
            cycle_colors=cycle_colors,
            color_map=color_map,
            H=H,
            W=W,
            occ=occ,
            plus_occ=plus_occ,
            out=out,
            plus_color=plus_color,
            force_clue=False,
        )
        if info is None:
            return None
        pluses.append(info)

    inp = _build_input_from_output(out, pluses, bg, keep_idx=0)

    try:
        produced = main(inp)
    except Exception:
        return None
    if not np.array_equal(produced, out):
        return None
    return inp


def generate_input() -> np.ndarray:
    rng = np.random.default_rng()
    for _ in range(_GEN_RETRIES):
        g = _gen_attempt(rng)
        if g is not None:
            return g
    raise RuntimeError(f"generate_input: failed after {_GEN_RETRIES} attempts")
