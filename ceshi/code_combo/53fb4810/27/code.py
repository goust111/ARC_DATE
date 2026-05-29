# Combo variant of 53fb4810 (index 27)
# combo_spec: D3/1+D4/1+D5/0
# Name: 箭头精灵大瓦片颜色循环

from __future__ import annotations

import numpy as np

from arc_utils import bbox_coords, crop_bbox, get_bg, label_mask


# ---------------------------------------------------------------------------
# Design analysis for the combo variant
#
# a) Concrete behavior trace
#    The original task used plus anchors, tiny attached markers, and stationary
#    periodic copying. This combo changes all three at once:
#    - Anchor cue: the big monochrome objects are now rotated arrow sprites.
#      Their pointing direction, not the attachment side, decides the ray.
#    - Seed geometry: each marker is a filled large rectangle such as 3x2, 3x3,
#      or 4x3 instead of a tiny 1x2 / 2x2 patch.
#    - Repetition rule: the first chunk is copied from the visible seed, but
#      every later whole-tile chunk is recolored by one fixed 3-color cycle
#      learned from the single long clue ray.
#    Example: if a right-pointing arrow carries a 3x2 seed on its lower flank,
#    then the output continues to the right in 3x2 blocks; block 2 is seed
#    colors after one permutation step, block 3 after two steps, and so on.
#
# b) Complexity calibration
#    The variant would become ambiguous if arrow directions, seed sizes, and
#    color permutations all varied independently per object. To keep it human-
#    learnable, one grid shares:
#    - one arrow template family,
#    - one global large-tile base size (cross, along),
#    - one global 3-color cycle,
#    - exactly one pre-extended clue ray.
#    Seeds still differ by internal texture and arrow orientation, so the combo
#    remains visibly distinct from all three source variants.
#
# c) Hidden assumptions audit against the original solver
#    - Original direction-from-adjacency is invalid: the seed attaches on a
#      clockwise flank, orthogonal to the ray. We must recognize arrow shape.
#    - Original `crop_bbox(component)` for the clue marker is invalid because
#      the clue component spans many repeated chunks. We reconstruct the seed
#      origin from the matched arrow geometry plus a globally inferred tile size.
#    - Original stationary tiling is invalid because later chunks are recolored.
#      Repetition must advance in tile-sized steps and apply a global map.
#    - Original dominant-big-color heuristic is too weak here because large
#      colored rays can also create big components. We instead detect the anchor
#      color by exact arrow-shape matches.
#
# d) Placement legality
#    The generator is output-first. It places arrows with one-cell guard bands,
#    then places full rays while forbidding overlap and 4-neighbor touching with
#    any previous non-background payload. This guarantees that each arrow has
#    exactly one adjacent non-arrow component and label-based extraction stays
#    unambiguous.
#
# e) Visual observability
#    A human can see the fused rule because one long clue ray simultaneously
#    shows all three cues: the arrow points one way while the seed sits on a
#    side flank, the repeated unit is visibly large, and consecutive chunks keep
#    the same geometry but cycle their colors. Other arrows reuse the same size
#    and same color cycle, making the rule transferable from 2-4 examples.
#
# f) Visual cleanliness
#    Large tiles are limited to a small menu of sizes, rays do not touch each
#    other, and each grid uses only 2-4 arrows. This prevents component merges
#    and keeps the payload textures readable despite the richer structure.
#
# g) Color-change inferability
#    The cycle always has exactly three colors, and the long clue seed contains
#    all three. Consecutive clue chunks therefore expose the same whole-tile
#    permutation multiple times; the solver never has to guess an unseen mapping.
#
# h) Quantitative anti-degeneration target
#    The generator is designed so that:
#    - every seed area is at least 6 cells,
#    - every sample uses at least two arrow orientations,
#    - the fused output differs from plain non-cycling tiling on the clue ray.
#    This prevents collapse back to the original task or to a single-dimension
#    variant.
#
# i) New solver assumptions and generator support
#    - All arrows share one of the known templates, up to rotation.
#    - All seeds share one large base size per grid; only orientation swaps h/w.
#    - Exactly one adjacent marker component per arrow is present.
#    - Exactly one component is elongated enough to serve as the color-cycle
#      clue.
#    The generator explicitly enforces all four assumptions and rejects any
#    construction that fails the solver oracle.
# ---------------------------------------------------------------------------


_DIR_VECS = {
    "R": (0, 1),
    "U": (-1, 0),
    "L": (0, -1),
    "D": (1, 0),
}

_ARROW_SPECS = [
    {
        "mask": np.array(
            [
                [0, 0, 1, 0],
                [1, 1, 1, 1],
                [0, 0, 1, 0],
            ],
            dtype=int,
        ),
        "launch": (2, 2),
    },
    {
        "mask": np.array(
            [
                [0, 0, 1, 0],
                [0, 1, 1, 0],
                [1, 1, 1, 1],
                [0, 1, 1, 0],
                [0, 0, 1, 0],
            ],
            dtype=int,
        ),
        "launch": (4, 2),
    },
    {
        "mask": np.array(
            [
                [0, 0, 1, 0, 0],
                [0, 0, 1, 1, 0],
                [1, 1, 1, 1, 1],
                [0, 0, 1, 1, 0],
                [0, 0, 1, 0, 0],
            ],
            dtype=int,
        ),
        "launch": (4, 2),
    },
]

_CANONICAL_ATTACH_NORMAL = (1, 0)


def _apply_color_map(tile: np.ndarray, color_map: dict[int, int]) -> np.ndarray:
    src = np.array(tile, dtype=int, copy=False)
    out = src.copy()
    for a, b in color_map.items():
        out[src == int(a)] = int(b)
    return out


def _rotate_patch(patch: np.ndarray, direction: str) -> np.ndarray:
    if direction == "R":
        return np.array(patch, dtype=int, copy=True)
    if direction == "U":
        return np.rot90(patch, k=1).copy()
    if direction == "L":
        return np.rot90(patch, k=2).copy()
    if direction == "D":
        return np.rot90(patch, k=-1).copy()
    raise ValueError(direction)


def _rotate_point(
    point: tuple[int, int],
    shape: tuple[int, int],
    direction: str,
) -> tuple[int, int]:
    r, c = map(int, point)
    h, w = map(int, shape)
    if direction == "R":
        return int(r), int(c)
    if direction == "U":
        return int(w - 1 - c), int(r)
    if direction == "L":
        return int(h - 1 - r), int(w - 1 - c)
    if direction == "D":
        return int(c), int(h - 1 - r)
    raise ValueError(direction)


def _rotate_vector(vec: tuple[int, int], direction: str) -> tuple[int, int]:
    dr, dc = map(int, vec)
    if direction == "R":
        return int(dr), int(dc)
    if direction == "U":
        return int(-dc), int(dr)
    if direction == "L":
        return int(-dr), int(-dc)
    if direction == "D":
        return int(dc), int(-dr)
    raise ValueError(direction)


def _oriented_arrow_spec(
    spec: dict,
    direction: str,
) -> tuple[np.ndarray, tuple[int, int], tuple[int, int]]:
    mask = _rotate_patch(np.array(spec["mask"], dtype=int), direction)
    launch = _rotate_point(tuple(spec["launch"]), np.array(spec["mask"]).shape, direction)
    attach_normal = _rotate_vector(_CANONICAL_ATTACH_NORMAL, direction)
    return mask, launch, attach_normal


def _match_arrow_component(mask: np.ndarray) -> dict | None:
    local = np.array(mask, dtype=bool, copy=False)
    for spec in _ARROW_SPECS:
        for direction_name, direction in _DIR_VECS.items():
            rotated, launch_local, attach_normal = _oriented_arrow_spec(spec, direction_name)
            if rotated.shape == local.shape and np.array_equal(rotated != 0, local):
                return {
                    "direction": tuple(map(int, direction)),
                    "direction_name": direction_name,
                    "launch_local": tuple(map(int, launch_local)),
                    "attach_normal": tuple(map(int, attach_normal)),
                }
    return None


def _detect_arrow_color(grid: np.ndarray, *, bg: int) -> int | None:
    best_key: tuple[int, int] | None = None
    best_color: int | None = None

    for color in sorted(set(map(int, grid.flatten())) - {int(bg)}):
        lab, n = label_mask(grid == int(color), connectivity=4)
        count = 0
        total_area = 0
        for k in range(1, int(n) + 1):
            coords = np.argwhere(lab == int(k))
            if coords.size == 0:
                continue
            r0, c0, r1, c1 = bbox_coords(coords)
            local = grid[int(r0) : int(r1) + 1, int(c0) : int(c1) + 1] == int(color)
            if _match_arrow_component(local) is None:
                continue
            count += 1
            total_area += int(coords.shape[0])
        if count == 0:
            continue
        key = (int(count), int(total_area))
        if best_key is None or key > best_key:
            best_key = key
            best_color = int(color)

    return int(best_color) if best_color is not None else None


def _tile_shape_for_direction(
    direction: tuple[int, int],
    base_along: int,
    base_cross: int,
) -> tuple[int, int]:
    _dr, dc = map(int, direction)
    if dc == 0:
        return int(base_along), int(base_cross)
    return int(base_cross), int(base_along)


def _seed_origin_from_launch(
    launch_abs: tuple[int, int],
    tile_shape: tuple[int, int],
    attach_normal: tuple[int, int],
) -> tuple[int, int]:
    lr, lc = map(int, launch_abs)
    th, tw = map(int, tile_shape)
    nr, nc = map(int, attach_normal)
    if (nr, nc) == (1, 0):
        return int(lr + 1), int(lc)
    if (nr, nc) == (-1, 0):
        return int(lr - th), int(lc)
    if (nr, nc) == (0, 1):
        return int(lr), int(lc + 1)
    if (nr, nc) == (0, -1):
        return int(lr), int(lc - tw)
    raise ValueError(attach_normal)


def _along_extent(coords: np.ndarray, direction: tuple[int, int]) -> int:
    r0, c0, r1, c1 = bbox_coords(coords)
    _dr, dc = map(int, direction)
    if dc == 0:
        return int(r1 - r0 + 1)
    return int(c1 - c0 + 1)


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
) -> dict[int, int] | None:
    mapping: dict[int, int] = {}
    evidence = 0
    for a, b in zip(chunks, chunks[1:]):
        for va, vb in zip(a.flatten(), b.flatten()):
            va = int(va)
            vb = int(vb)
            if va == int(bg) or vb == int(bg):
                continue
            evidence += 1
            prev = mapping.get(int(va))
            if prev is not None and int(prev) != int(vb):
                return None
            mapping[int(va)] = int(vb)
    if evidence == 0 or len(mapping) < 2:
        return None
    return mapping


def _infer_global_color_map(
    grid: np.ndarray,
    infos: list[dict],
    *,
    bg: int,
) -> dict[int, int]:
    for info in infos:
        expected_along = int(info["expected_along"])
        if int(info["along_extent"]) <= expected_along:
            continue
        sr0, sc0, sr1, sc1 = map(int, info["seed_bbox"])
        shape = (int(sr1 - sr0 + 1), int(sc1 - sc0 + 1))
        chunks = _extract_full_chunks(
            grid,
            int(sr0),
            int(sc0),
            shape,
            tuple(map(int, info["direction"])),
        )
        mapping = _infer_color_map_from_chunks(chunks, bg=int(bg))
        if mapping:
            return mapping
    return {}


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


def main(grid: np.ndarray) -> np.ndarray:
    """
    Extend large marker tiles from arrow anchors, cycling colors on every repeated chunk.

    Input structure:
    - The most common color is background.
    - One non-background color forms several rotated arrow sprites from a small fixed template family.
    - Each arrow has exactly one adjacent large rectangular seed marker on a side flank, and exactly one arrow in the grid keeps its full long ray as the clue.

    Rule:
    - Read each arrow's pointing direction from its silhouette rather than from the marker attachment side.
    - Infer the shared seed-tile size from the compact markers, reconstruct the clue arrow's first tile, and learn a global color permutation by comparing consecutive full chunks on the long clue ray.
    - For every arrow, repeat its seed rectangle outward in the arrow direction until the boundary, applying one more step of the learned color cycle after each whole-tile advance; arrow pixels are never overwritten.

    Key insight:
    The task now fuses three cues: arrow shape gives direction, the payload is a large textured rectangle, and the single long example ray reveals that repetition preserves geometry but rotates colors through a fixed global cycle.
    """
    g = np.array(grid, dtype=int)
    out = g.copy()
    H, W = g.shape

    bg = int(get_bg(g))
    arrow_color = _detect_arrow_color(g, bg=int(bg))
    if arrow_color is None:
        return out

    arrow_lab, an = label_mask(g == int(arrow_color), connectivity=4)
    non_mask = (g != int(bg)) & (g != int(arrow_color))
    mark_lab, _ = label_mask(non_mask, connectivity=4)

    component_records: dict[int, dict] = {}
    arrow_records: list[dict] = []

    for ak in range(1, int(an) + 1):
        arrow_mask = arrow_lab == int(ak)
        arrow_coords = np.argwhere(arrow_mask)
        if arrow_coords.size == 0:
            continue

        ar0, ac0, ar1, ac1 = bbox_coords(arrow_coords)
        local_arrow = g[int(ar0) : int(ar1) + 1, int(ac0) : int(ac1) + 1] == int(arrow_color)
        matched = _match_arrow_component(local_arrow)
        if matched is None:
            continue

        adj_ids: set[int] = set()
        for r, c in arrow_coords:
            r = int(r)
            c = int(c)
            for dr, dc in ((1, 0), (-1, 0), (0, 1), (0, -1)):
                rr = int(r + dr)
                cc = int(c + dc)
                if 0 <= rr < H and 0 <= cc < W and int(mark_lab[rr, cc]) > 0:
                    adj_ids.add(int(mark_lab[rr, cc]))
        if len(adj_ids) != 1:
            continue

        cid = next(iter(adj_ids))
        if cid not in component_records:
            pts = np.argwhere(mark_lab == int(cid))
            if pts.size == 0:
                continue
            r0, c0, r1, c1 = bbox_coords(pts)
            component_records[int(cid)] = {
                "pts": pts,
                "bbox": (int(r0), int(c0), int(r1), int(c1)),
                "h": int(r1 - r0 + 1),
                "w": int(c1 - c0 + 1),
            }

        arrow_records.append(
            {
                "bbox": (int(ar0), int(ac0), int(ar1), int(ac1)),
                "matched": matched,
                "component_id": int(cid),
            }
        )

    if not arrow_records:
        return out

    base_cross = min(int(min(meta["h"], meta["w"])) for meta in component_records.values())
    base_along = min(int(max(meta["h"], meta["w"])) for meta in component_records.values())
    if base_along < base_cross:
        base_along, base_cross = base_cross, base_along

    infos: list[dict] = []
    for record in arrow_records:
        ar0, ac0, _ar1, _ac1 = map(int, record["bbox"])
        matched = record["matched"]
        direction = tuple(map(int, matched["direction"]))
        launch_local = tuple(map(int, matched["launch_local"]))
        attach_normal = tuple(map(int, matched["attach_normal"]))
        tile_shape = _tile_shape_for_direction(direction, int(base_along), int(base_cross))

        launch_abs = (int(ar0 + launch_local[0]), int(ac0 + launch_local[1]))
        seed_r0, seed_c0 = _seed_origin_from_launch(launch_abs, tile_shape, attach_normal)
        th, tw = map(int, tile_shape)
        seed_r1 = int(seed_r0 + th - 1)
        seed_c1 = int(seed_c0 + tw - 1)
        if not (0 <= seed_r0 <= seed_r1 < H and 0 <= seed_c0 <= seed_c1 < W):
            continue

        tile = crop_bbox(g, (int(seed_r0), int(seed_c0), int(seed_r1), int(seed_c1)))
        if np.any(tile == int(bg)) or np.any(tile == int(arrow_color)):
            continue

        comp = component_records[int(record["component_id"])]
        along_extent = _along_extent(comp["pts"], direction)
        infos.append(
            {
                "direction": direction,
                "seed_bbox": (int(seed_r0), int(seed_c0), int(seed_r1), int(seed_c1)),
                "tile": np.array(tile, dtype=int, copy=True),
                "along_extent": int(along_extent),
                "expected_along": int(th if direction[1] == 0 else tw),
            }
        )

    color_map = _infer_global_color_map(g, infos, bg=int(bg))

    for info in infos:
        sr0, sc0, _sr1, _sc1 = map(int, info["seed_bbox"])
        extend_color_cycle_tile(
            out,
            np.array(info["tile"], dtype=int, copy=True),
            int(sr0),
            int(sc0),
            tuple(map(int, info["direction"])),
            bg=int(bg),
            protect={int(arrow_color)},
            color_map=color_map,
        )

    return out


# --------------- generator ---------------

_H_RANGE = (22, 35)
_W_RANGE = (20, 31)
_N_ARROW_RANGE = (2, 5)
_BASE_CROSS_CHOICES = [2, 3]
_BASE_ALONG_CHOICES = [3, 4]
_MIN_FULL_UNITS = 3
_CLUE_MIN_FULL_UNITS = 4
_PLACEMENT_TRIES = 220
_GEN_RETRIES = 700
_CYCLE_LEN = 3


def _component_cells(mask: np.ndarray, origin: tuple[int, int]) -> list[tuple[int, int]]:
    r0, c0 = map(int, origin)
    coords = np.argwhere(mask != 0)
    return [(int(r0 + r), int(c0 + c)) for r, c in coords]


def _mark_guard(guard: np.ndarray, cells: list[tuple[int, int]]) -> None:
    H, W = guard.shape
    for r, c in cells:
        guard[int(r), int(c)] = True
        for dr, dc in ((1, 0), (-1, 0), (0, 1), (0, -1)):
            rr = int(r + dr)
            cc = int(c + dc)
            if 0 <= rr < H and 0 <= cc < W:
                guard[rr, cc] = True


def _cells_clear(
    cells: list[tuple[int, int]],
    occ: np.ndarray,
    *guards: np.ndarray,
) -> bool:
    H, W = occ.shape
    for r, c in cells:
        if not (0 <= int(r) < H and 0 <= int(c) < W):
            return False
        if int(occ[int(r), int(c)]) != 0:
            return False
        for guard in guards:
            if bool(guard[int(r), int(c)]):
                return False
    return True


def _in_bounds_bbox(H: int, W: int, r0: int, c0: int, h: int, w: int) -> bool:
    return 0 <= int(r0) and 0 <= int(c0) and int(r0) + int(h) <= H and int(c0) + int(w) <= W


def _count_full_units(
    H: int,
    W: int,
    seed_r0: int,
    seed_c0: int,
    tile_shape: tuple[int, int],
    direction: tuple[int, int],
) -> int:
    h, w = map(int, tile_shape)
    dr, dc = map(int, direction)
    rr = int(seed_r0)
    cc = int(seed_c0)
    count = 0

    while 0 <= rr and rr + h <= H and 0 <= cc and cc + w <= W:
        count += 1
        rr += int(dr * h)
        cc += int(dc * w)

    return int(count)


def _project_cycle_coords(
    H: int,
    W: int,
    seed_r0: int,
    seed_c0: int,
    direction: tuple[int, int],
    tile: np.ndarray,
    color_map: dict[int, int],
) -> list[tuple[int, int, int]]:
    h, w = tile.shape
    dr, dc = map(int, direction)
    rr = int(seed_r0)
    cc = int(seed_c0)
    cur = np.array(tile, dtype=int, copy=True)
    cells: list[tuple[int, int, int]] = []

    while rr < H and rr + h > 0 and cc < W and cc + w > 0:
        for i in range(h):
            for j in range(w):
                rrr = int(rr + i)
                ccc = int(cc + j)
                if 0 <= rrr < H and 0 <= ccc < W:
                    cells.append((int(rrr), int(ccc), int(cur[i, j])))
        rr += int(dr * h)
        cc += int(dc * w)
        cur = _apply_color_map(cur, color_map)

    return cells


def _tile_has_texture(
    tile: np.ndarray,
    *,
    min_colors: int,
) -> bool:
    if len(set(map(int, tile.flatten()))) < int(min_colors):
        return False
    if tile.shape[0] > 1 and len({tuple(map(int, row)) for row in tile}) < 2:
        return False
    if tile.shape[1] > 1 and len({tuple(map(int, tile[:, j])) for j in range(tile.shape[1])}) < 2:
        return False
    return True


def _build_seed_tile(
    tile_shape: tuple[int, int],
    cycle_colors: list[int],
    rng: np.random.Generator,
    *,
    force_clue: bool,
) -> np.ndarray | None:
    h, w = map(int, tile_shape)
    n_cells = int(h * w)
    min_colors = 3 if force_clue else 2

    for _ in range(40):
        if force_clue:
            vals = list(map(int, cycle_colors))
            vals.extend(list(map(int, rng.choice(cycle_colors, size=n_cells - len(vals), replace=True))))
        else:
            n_colors = int(rng.integers(min_colors, min(len(cycle_colors), n_cells) + 1))
            chosen = list(map(int, rng.choice(cycle_colors, size=n_colors, replace=False)))
            vals = list(chosen)
            vals.extend(list(map(int, rng.choice(chosen, size=n_cells - len(vals), replace=True))))
        rng.shuffle(vals)
        tile = np.array(vals, dtype=int).reshape(h, w)
        if _tile_has_texture(tile, min_colors=min_colors):
            return tile

    return None


def _place_arrow_and_ray(
    rng: np.random.Generator,
    spec: dict,
    direction_name: str,
    base_along: int,
    base_cross: int,
    cycle_colors: list[int],
    color_map: dict[int, int],
    H: int,
    W: int,
    occ: np.ndarray,
    arrow_guard: np.ndarray,
    payload_guard: np.ndarray,
    out: np.ndarray,
    arrow_color: int,
    *,
    force_clue: bool,
) -> dict | None:
    direction = _DIR_VECS[direction_name]
    patch, launch_local, attach_normal = _oriented_arrow_spec(spec, direction_name)
    tile_shape = _tile_shape_for_direction(direction, int(base_along), int(base_cross))
    tile = _build_seed_tile(tile_shape, cycle_colors, rng, force_clue=bool(force_clue))
    if tile is None:
        return None

    ph, pw = patch.shape
    th, tw = tile.shape

    for _ in range(_PLACEMENT_TRIES):
        r0 = int(rng.integers(0, H - ph + 1))
        c0 = int(rng.integers(0, W - pw + 1))
        arrow_cells = _component_cells(patch, (r0, c0))
        if not _cells_clear(arrow_cells, occ, arrow_guard, payload_guard):
            continue

        launch_abs = (int(r0 + launch_local[0]), int(c0 + launch_local[1]))
        seed_r0, seed_c0 = _seed_origin_from_launch(launch_abs, (th, tw), attach_normal)
        if not _in_bounds_bbox(H, W, seed_r0, seed_c0, th, tw):
            continue

        full_units = _count_full_units(H, W, seed_r0, seed_c0, (th, tw), direction)
        need = _CLUE_MIN_FULL_UNITS if force_clue else _MIN_FULL_UNITS
        if full_units < need:
            continue

        ray_cells = _project_cycle_coords(H, W, seed_r0, seed_c0, direction, tile, color_map)
        payload_cells = sorted({(int(r), int(c)) for r, c, _v in ray_cells})
        if not _cells_clear(payload_cells, occ, arrow_guard, payload_guard):
            continue

        for r, c in arrow_cells:
            occ[int(r), int(c)] = 1
            out[int(r), int(c)] = int(arrow_color)
        _mark_guard(arrow_guard, arrow_cells)

        for r, c, v in ray_cells:
            occ[int(r), int(c)] = 2
            out[int(r), int(c)] = int(v)
        _mark_guard(payload_guard, payload_cells)

        return {
            "direction": tuple(map(int, direction)),
            "direction_name": direction_name,
            "tile": np.array(tile, dtype=int, copy=True),
            "seed_r0": int(seed_r0),
            "seed_c0": int(seed_c0),
            "cells": ray_cells,
            "full_units": int(full_units),
            "is_clue": bool(force_clue),
            "base_along": int(base_along),
            "base_cross": int(base_cross),
        }

    return None


def _build_input_from_output(
    out: np.ndarray,
    arrows: list[dict],
    bg: int,
) -> np.ndarray:
    inp = np.array(out, dtype=int, copy=True)
    keep_idx = next(i for i, meta in enumerate(arrows) if bool(meta["is_clue"]))
    for i, meta in enumerate(arrows):
        if int(i) == int(keep_idx):
            continue
        th, tw = meta["tile"].shape
        sr0 = int(meta["seed_r0"])
        sc0 = int(meta["seed_c0"])
        seed_set = {(int(sr0 + rr), int(sc0 + cc)) for rr in range(th) for cc in range(tw)}
        for r, c, _v in meta["cells"]:
            if (int(r), int(c)) not in seed_set:
                inp[int(r), int(c)] = int(bg)
    return inp


def _sample_direction_names(rng: np.random.Generator, n: int) -> list[str]:
    dirs = ["U", "D", "L", "R"]
    while True:
        chosen = [str(rng.choice(dirs)) for _ in range(int(n))]
        if len(set(chosen)) >= 2:
            return chosen


def _plain_periodic_output(inp: np.ndarray) -> np.ndarray:
    g = np.array(inp, dtype=int)
    out = g.copy()
    H, W = g.shape
    bg = int(get_bg(g))
    arrow_color = _detect_arrow_color(g, bg=int(bg))
    if arrow_color is None:
        return out

    arrow_lab, an = label_mask(g == int(arrow_color), connectivity=4)
    non_mask = (g != int(bg)) & (g != int(arrow_color))
    mark_lab, _ = label_mask(non_mask, connectivity=4)

    component_records: dict[int, dict] = {}
    records: list[dict] = []
    for ak in range(1, int(an) + 1):
        arrow_mask = arrow_lab == int(ak)
        arrow_coords = np.argwhere(arrow_mask)
        if arrow_coords.size == 0:
            continue
        ar0, ac0, ar1, ac1 = bbox_coords(arrow_coords)
        local_arrow = g[int(ar0) : int(ar1) + 1, int(ac0) : int(ac1) + 1] == int(arrow_color)
        matched = _match_arrow_component(local_arrow)
        if matched is None:
            continue
        adj_ids: set[int] = set()
        for r, c in arrow_coords:
            for dr, dc in ((1, 0), (-1, 0), (0, 1), (0, -1)):
                rr = int(r + dr)
                cc = int(c + dc)
                if 0 <= rr < H and 0 <= cc < W and int(mark_lab[rr, cc]) > 0:
                    adj_ids.add(int(mark_lab[rr, cc]))
        if len(adj_ids) != 1:
            continue
        cid = next(iter(adj_ids))
        if cid not in component_records:
            pts = np.argwhere(mark_lab == int(cid))
            if pts.size == 0:
                continue
            r0, c0, r1, c1 = bbox_coords(pts)
            component_records[int(cid)] = {"h": int(r1 - r0 + 1), "w": int(c1 - c0 + 1)}
        records.append(
            {
                "bbox": (int(ar0), int(ac0), int(ar1), int(ac1)),
                "matched": matched,
                "cid": int(cid),
            }
        )

    if not records:
        return out

    base_cross = min(int(min(meta["h"], meta["w"])) for meta in component_records.values())
    base_along = min(int(max(meta["h"], meta["w"])) for meta in component_records.values())
    if base_along < base_cross:
        base_along, base_cross = base_cross, base_along

    for record in records:
        ar0, ac0, _ar1, _ac1 = map(int, record["bbox"])
        matched = record["matched"]
        direction = tuple(map(int, matched["direction"]))
        tile_shape = _tile_shape_for_direction(direction, int(base_along), int(base_cross))
        launch_abs = (int(ar0 + matched["launch_local"][0]), int(ac0 + matched["launch_local"][1]))
        seed_r0, seed_c0 = _seed_origin_from_launch(launch_abs, tile_shape, matched["attach_normal"])
        th, tw = map(int, tile_shape)
        seed_r1 = int(seed_r0 + th - 1)
        seed_c1 = int(seed_c0 + tw - 1)
        if not (0 <= seed_r0 <= seed_r1 < H and 0 <= seed_c0 <= seed_c1 < W):
            continue
        tile = crop_bbox(g, (int(seed_r0), int(seed_c0), int(seed_r1), int(seed_c1)))
        extend_color_cycle_tile(
            out,
            tile,
            int(seed_r0),
            int(seed_c0),
            direction,
            bg=int(bg),
            protect={int(arrow_color)},
            color_map={},
        )

    return out


def _gen_attempt(rng: np.random.Generator) -> np.ndarray | None:
    H = int(rng.integers(*_H_RANGE))
    W = int(rng.integers(*_W_RANGE))

    bg = int(rng.integers(0, 10))
    others = [int(c) for c in range(10) if int(c) != int(bg)]
    arrow_color = int(rng.choice(others))
    cycle_colors = list(map(int, rng.choice([c for c in others if int(c) != int(arrow_color)], size=_CYCLE_LEN, replace=False)))
    rng.shuffle(cycle_colors)
    color_map = {
        int(cycle_colors[i]): int(cycle_colors[(i + 1) % _CYCLE_LEN])
        for i in range(_CYCLE_LEN)
    }

    base_cross = int(rng.choice(_BASE_CROSS_CHOICES))
    base_along = int(rng.choice([a for a in _BASE_ALONG_CHOICES if int(a) >= int(base_cross)]))

    spec = _ARROW_SPECS[int(rng.integers(0, len(_ARROW_SPECS)))]
    n = int(rng.integers(*_N_ARROW_RANGE))
    direction_names = _sample_direction_names(rng, n)

    out = np.full((H, W), bg, dtype=int)
    occ = np.zeros((H, W), dtype=int)
    arrow_guard = np.zeros((H, W), dtype=bool)
    payload_guard = np.zeros((H, W), dtype=bool)

    arrows: list[dict] = []
    for idx, direction_name in enumerate(direction_names):
        meta = _place_arrow_and_ray(
            rng=rng,
            spec=spec,
            direction_name=direction_name,
            base_along=int(base_along),
            base_cross=int(base_cross),
            cycle_colors=cycle_colors,
            color_map=color_map,
            H=H,
            W=W,
            occ=occ,
            arrow_guard=arrow_guard,
            payload_guard=payload_guard,
            out=out,
            arrow_color=int(arrow_color),
            force_clue=bool(idx == 0),
        )
        if meta is None:
            return None
        arrows.append(meta)

    inp = _build_input_from_output(out, arrows, int(bg))

    try:
        produced = main(inp)
    except Exception:
        return None
    if not np.array_equal(produced, out):
        return None

    plain = _plain_periodic_output(inp)
    if np.array_equal(plain, out):
        return None

    return inp


def generate_input() -> np.ndarray:
    rng = np.random.default_rng()
    for _ in range(_GEN_RETRIES):
        grid = _gen_attempt(rng)
        if grid is not None:
            return grid
    raise RuntimeError(f"generate_input: failed after {_GEN_RETRIES} attempts")
