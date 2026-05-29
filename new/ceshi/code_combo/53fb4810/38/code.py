# Combo variant of 53fb4810 (index 38)
# combo_spec: D3/1+D18/0
# Name: 边缘编码颜色循环

from __future__ import annotations

import numpy as np

from arc_utils import bbox_coords, crop_bbox, get_bg, label_mask


# ---------------------------------------------------------------------------
# Design analysis for the combo variant
#
# a) Concrete behavior trace
#    - Original task:
#      1. A small rectangular seed touches one side of a '+'.
#      2. That touching side directly gives the ray direction.
#      3. The exact same tile is copied forever along that ray.
#    - This combo splits those responsibilities:
#      1. The seed still touches the '+' and still tells us where the compact
#         seed bbox is located.
#      2. Direction no longer comes from that touching side. A thin border
#         strip in a dedicated control color chooses up/down/left/right.
#      3. When the ray advances by one full tile, the tile is recolored by a
#         global permutation learned from the single long clue ray.
#    - Small trace:
#      seed near plus = [[A, B],
#                        [C, A]]
#      border strip says "right"
#      clue ray shows:
#         [[A, B]] [[B, C]] [[C, A]] [[A, B]]
#         [[C, A]] [[A, B]] [[B, C]] [[C, A]]
#      so the solver must read "project right" from the edge strip, then reuse
#      A->B, B->C, C->A on every other seed.
#
# b) Complexity calibration
#    - Direction has only four visible categories, and every grid repeats the
#      same strip vocabulary several times.
#    - The color cycle is fixed globally per grid and limited to three colors.
#      The long clue ray is forced to use a 2x2 seed containing all three cycle
#      colors so the permutation is visible from adjacent chunks.
#    - We keep only plus shapes whose nozzle span is 2. That makes the 2x2 clue
#      physically compatible without inflating the scene.
#
# c) Hidden assumptions that break from the original solver
#    - "Touching side = ray direction" is no longer valid. The new solver keeps
#      attachment only to reconstruct the compact seed bbox, then separately
#      decodes direction from edge controls.
#    - "Component bbox is the seed tile" fails for the long clue ray. We must
#      recover the seed bbox from plus geometry and the attachment side instead
#      of cropping the whole component.
#    - "Repeated chunks keep the same colors" also fails. Extension now advances
#      in tile-sized steps and applies a learned permutation after each step.
#    - New assumption introduced here: exactly one border-only control color
#      exists, and each seed span matches exactly one control strip. Generator
#      enforces that by reserving one color and keeping row/column spans unique.
#
# d) Placement legality
#    - Output is built first.
#    - Each object owns: one plus, one attached seed, one border strip, one full
#      color-cycling ray. All are collision-checked before painting.
#    - Pluses stay away from the border so control strips remain isolated and the
#      control color stays border-only.
#    - Rays never fire back through the plus. Sideways and same-side rays are
#      allowed, which is enough to demonstrate that edge strips, not adjacency,
#      drive direction.
#
# e) Visual observability
#    - The border strips are thin and spatially separated from the main scene,
#      so a human can notice they are symbolic instructions.
#    - At least one plus per generated grid has attachment direction different
#      from ray direction, so the new symbolic mechanism is necessary.
#    - The single long clue ray reveals the color cycle by repeating the same
#      geometry with shifted colors, not by hidden color-id ordering.
#
# f) Visual cleanliness
#    - Only one extra element is added compared with the base task: a thin edge
#      strip in one reserved color.
#    - Control strips are protected from overwrite, so the ray body and the
#      symbol layer do not visually merge in the middle of the grid.
#
# g) Color-change inferability
#    - Exactly three cycle colors are used grid-wide.
#    - The clue seed contains all three, so the solver sees every mapping step.
#    - Other seeds may use any subset of those three colors; after the global
#      permutation is learned once, applying it elsewhere is unambiguous.
#
# 9) New solver assumptions audited against the generator
#    - Assumption: one dominant non-bg color forms all pluses.
#      Guarantee: generator paints all pluses in one shared color.
#    - Assumption: one dedicated control color appears only on the border.
#      Guarantee: generator reserves one color for strips and keeps all strip
#      cells on the boundary.
#    - Assumption: each plus has exactly one adjacent non-control component.
#      Guarantee: generator places one seed per plus and blocks collisions.
#    - Assumption: each seed span matches exactly one control strip.
#      Guarantee: generator enforces unique absolute row spans and column spans.
#    - Assumption: at least one elongated clue ray exists and exposes the whole
#      3-cycle.
#      Guarantee: generator forces one 2x2 clue with >=4 full chunks.
# ---------------------------------------------------------------------------


def detect_dominant_big_component_color(
    grid: np.ndarray,
    *,
    bg: int | None = None,
    min_h: int = 3,
    min_w: int = 3,
    connectivity: int = 4,
) -> int | None:
    """Choose the non-bg color with the most large connected components."""
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


def detect_border_only_control_color(
    grid: np.ndarray,
    *,
    bg: int,
    exclude: set[int] | None = None,
) -> int | None:
    """Pick the unique non-bg color whose every occurrence lies on the border."""
    exclude = set() if exclude is None else set(map(int, exclude))
    H, W = grid.shape
    border = np.zeros((H, W), dtype=bool)
    border[0, :] = True
    border[-1, :] = True
    border[:, 0] = True
    border[:, -1] = True

    candidates: list[int] = []
    for col in sorted(set(map(int, grid.flatten())) - {int(bg)} - exclude):
        pts = np.argwhere(grid == int(col))
        if pts.size == 0:
            continue
        if all(bool(border[int(r), int(c)]) for r, c in pts):
            candidates.append(int(col))

    return int(candidates[0]) if len(candidates) == 1 else None


def _apply_color_map(tile: np.ndarray, color_map: dict[int, int]) -> np.ndarray:
    """Apply one whole-tile color permutation."""
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
    """Project a seed in tile-sized steps, recoloring after each full step."""
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


def _collect_control_specs(ctrl_mask: np.ndarray) -> list[dict]:
    """Describe every border strip as direction + aligned span."""
    H, W = ctrl_mask.shape
    ctrl_lab, cn = label_mask(ctrl_mask, connectivity=4)
    specs: list[dict] = []
    for k in range(1, int(cn) + 1):
        pts = np.argwhere(ctrl_lab == int(k))
        if pts.size == 0:
            continue
        r0, c0, r1, c1 = bbox_coords(pts)
        if int(r0) == 0 and int(r1) == 0:
            specs.append({"direction": (-1, 0), "span": tuple(range(int(c0), int(c1) + 1))})
        elif int(r0) == int(H - 1) and int(r1) == int(H - 1):
            specs.append({"direction": (1, 0), "span": tuple(range(int(c0), int(c1) + 1))})
        elif int(c0) == 0 and int(c1) == 0:
            specs.append({"direction": (0, -1), "span": tuple(range(int(r0), int(r1) + 1))})
        elif int(c0) == int(W - 1) and int(c1) == int(W - 1):
            specs.append({"direction": (0, 1), "span": tuple(range(int(r0), int(r1) + 1))})
    return specs


def _component_direction_from_plus(
    pts: np.ndarray,
    plus_mask: np.ndarray,
    H: int,
    W: int,
) -> tuple[int, int] | None:
    """Direction from plus to the touching seed side."""
    dirs: set[tuple[int, int]] = set()
    for pr, pc in pts:
        pr = int(pr)
        pc = int(pc)
        for dr, dc in ((1, 0), (-1, 0), (0, 1), (0, -1)):
            rr = pr + dr
            cc = pc + dc
            if 0 <= rr < H and 0 <= cc < W and bool(plus_mask[rr, cc]):
                dirs.add((-int(dr), -int(dc)))
    if len(dirs) != 1:
        return None
    return next(iter(dirs))


def _nozzle_cells(
    plus_mask: np.ndarray,
    plus_coords: np.ndarray,
    direction: tuple[int, int],
) -> list[int]:
    """Cross-axis cells along one chosen plus side."""
    pr0, pc0 = plus_coords.min(axis=0)
    pr1, pc1 = plus_coords.max(axis=0)
    dr, dc = map(int, direction)
    if dc == 0:
        nozzle_row = int(pr1) if dr == 1 else int(pr0)
        return [int(c) for c in range(int(pc0), int(pc1) + 1) if bool(plus_mask[nozzle_row, c])]
    nozzle_col = int(pc1) if dc == 1 else int(pc0)
    return [int(r) for r in range(int(pr0), int(pr1) + 1) if bool(plus_mask[r, nozzle_col])]


def _extent_along_direction(pts: np.ndarray, direction: tuple[int, int]) -> int:
    """BBox span measured along the chosen direction's axis."""
    r0, c0, r1, c1 = bbox_coords(pts)
    return int(r1 - r0 + 1) if int(direction[1]) == 0 else int(c1 - c0 + 1)


def _seed_bbox_from_plus(
    plus_mask: np.ndarray,
    plus_coords: np.ndarray,
    attach_direction: tuple[int, int],
    along: int,
) -> tuple[int, int, int, int]:
    """Reconstruct the compact seed tile adjacent to a plus."""
    pr0, pc0 = plus_coords.min(axis=0)
    pr1, pc1 = plus_coords.max(axis=0)
    dr, dc = map(int, attach_direction)
    along = int(along)

    if dc == 0:
        nozzle_cols = _nozzle_cells(plus_mask, plus_coords, attach_direction)
        c0 = int(min(nozzle_cols))
        c1 = int(max(nozzle_cols))
        r0 = int(pr1 + 1) if dr == 1 else int(pr0 - along)
        r1 = int(r0 + along - 1)
        return int(r0), int(c0), int(r1), int(c1)

    nozzle_rows = _nozzle_cells(plus_mask, plus_coords, attach_direction)
    r0 = int(min(nozzle_rows))
    r1 = int(max(nozzle_rows))
    c0 = int(pc1 + 1) if dc == 1 else int(pc0 - along)
    c1 = int(c0 + along - 1)
    return int(r0), int(c0), int(r1), int(c1)


def _extract_full_chunks(
    grid: np.ndarray,
    r0: int,
    c0: int,
    shape: tuple[int, int],
    direction: tuple[int, int],
) -> list[np.ndarray]:
    """Read full tile-sized chunks from the seed position to the boundary."""
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
    ctrl_color: int,
) -> dict[int, int] | None:
    """Infer one global permutation from consecutive clue chunks."""
    mapping: dict[int, int] = {}
    evidence = 0
    for a, b in zip(chunks, chunks[1:]):
        for va, vb in zip(a.flatten(), b.flatten()):
            va = int(va)
            vb = int(vb)
            if va in {int(bg), int(ctrl_color)} or vb in {int(bg), int(ctrl_color)}:
                continue
            evidence += 1
            prev = mapping.get(int(va))
            if prev is not None and int(prev) != int(vb):
                return None
            mapping[int(va)] = int(vb)

    if evidence == 0 or len(mapping) < 3:
        return None
    if len(set(mapping.values())) != len(mapping):
        return None
    if set(mapping.values()) != set(mapping.keys()):
        return None
    return mapping


def _infer_global_color_map(
    grid: np.ndarray,
    infos: list[dict],
    *,
    bg: int,
    ctrl_color: int,
) -> dict[int, int]:
    """Use the elongated clue ray to recover the grid-wide color cycle."""
    infos = sorted(infos, key=lambda d: int(d["ray_extent"]), reverse=True)
    for info in infos:
        seed_r0, seed_c0, seed_r1, seed_c1 = info["seed_bbox"]
        shape = (int(seed_r1 - seed_r0 + 1), int(seed_c1 - seed_c0 + 1))
        chunks = _extract_full_chunks(
            grid,
            int(seed_r0),
            int(seed_c0),
            shape,
            tuple(map(int, info["ray_direction"])),
        )
        mapping = _infer_color_map_from_chunks(chunks, bg=int(bg), ctrl_color=int(ctrl_color))
        if mapping:
            return mapping
    return {}


def main(grid: np.ndarray) -> np.ndarray:
    """
    Extend attached seed tiles using edge-decoded directions and a learned color cycle.

    Input structure:
    - The most common color is background.
    - One dominant non-background color forms several identical '+' anchors.
    - Each plus has one adjacent rectangular seed component, and one border-only control strip of a dedicated color exists for each seed.
    - Exactly one seed is already expanded into a long ray; that clue ray shows repeated tile chunks with changing colors.

    Rule:
    - Use the plus adjacency only to recover each compact seed bbox.
    - Match that seed to the unique border strip whose span aligns with the seed bbox; the border side gives the ray direction.
    - Read the long clue ray in full tile steps to infer one global color permutation, then extend every seed in its decoded direction, applying one more permutation step after each tile repetition.
    - Plus pixels and control-strip pixels are protected.

    Key insight:
    The attached seed still provides the payload, but not the trajectory. Edge strips choose direction, and the single long ray teaches how colors advance from one repeated chunk to the next.
    """
    g = np.array(grid, dtype=int)
    out = g.copy()
    H, W = g.shape

    bg = int(get_bg(g))
    plus_col = detect_dominant_big_component_color(g, bg=bg, min_h=3, min_w=3, connectivity=4)
    if plus_col is None:
        return out

    ctrl_color = detect_border_only_control_color(g, bg=bg, exclude={int(plus_col)})
    if ctrl_color is None:
        return out

    plus_lab, pn = label_mask(g == int(plus_col), connectivity=4)
    ctrl_specs = _collect_control_specs(g == int(ctrl_color))
    non_mask = (g != int(bg)) & (g != int(plus_col)) & (g != int(ctrl_color))
    mark_lab, _mn = label_mask(non_mask, connectivity=4)

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
            for dr, dc in ((1, 0), (-1, 0), (0, 1), (0, -1)):
                rr = r + dr
                cc = c + dc
                if 0 <= rr < H and 0 <= cc < W and int(mark_lab[rr, cc]) > 0:
                    adj_ids.add(int(mark_lab[rr, cc]))

        if len(adj_ids) != 1:
            continue

        cid = next(iter(adj_ids))
        pts = np.argwhere(mark_lab == int(cid))
        if pts.size == 0:
            continue

        attach_direction = _component_direction_from_plus(pts, plus_mask, H, W)
        if attach_direction is None:
            continue

        seed_along = min(2, _extent_along_direction(pts, attach_direction))
        seed_bbox = _seed_bbox_from_plus(plus_mask, plus_coords, attach_direction, seed_along)
        sr0, sc0, sr1, sc1 = seed_bbox
        if not (0 <= sr0 <= sr1 < H and 0 <= sc0 <= sc1 < W):
            continue

        tile = crop_bbox(g, (int(sr0), int(sc0), int(sr1), int(sc1)))
        if np.any(tile == int(bg)) or np.any(tile == int(plus_col)) or np.any(tile == int(ctrl_color)):
            continue

        span_rows = tuple(range(int(sr0), int(sr1) + 1))
        span_cols = tuple(range(int(sc0), int(sc1) + 1))
        dirs: list[tuple[int, int]] = []
        for spec in ctrl_specs:
            dr, dc = spec["direction"]
            if dc == 0 and tuple(spec["span"]) == span_cols:
                dirs.append((int(dr), int(dc)))
            elif dr == 0 and tuple(spec["span"]) == span_rows:
                dirs.append((int(dr), int(dc)))
        dirs = sorted(set(dirs))
        if len(dirs) != 1:
            continue

        ray_direction = dirs[0]
        infos.append(
            {
                "seed_bbox": (int(sr0), int(sc0), int(sr1), int(sc1)),
                "tile": np.array(tile, dtype=int, copy=True),
                "ray_direction": tuple(map(int, ray_direction)),
                "ray_extent": int(_extent_along_direction(pts, ray_direction)),
            }
        )

    color_map = _infer_global_color_map(g, infos, bg=int(bg), ctrl_color=int(ctrl_color))

    for info in infos:
        sr0, sc0, _sr1, _sc1 = info["seed_bbox"]
        extend_color_cycle_tile(
            out,
            info["tile"],
            int(sr0),
            int(sc0),
            tuple(map(int, info["ray_direction"])),
            bg=int(bg),
            protect={int(plus_col), int(ctrl_color)},
            color_map=color_map,
        )

    return out


# --------------- generator ---------------

_H_RANGE = (13, 29)
_W_RANGE = (13, 23)
_N_PLUS_RANGE = (2, 4)
_SEED_ALONG_CHOICES = [1, 2]
_MIN_RAY_UNITS = 2
_CLUE_MIN_FULL_UNITS = 4
_CLUE_IDEAL_UNITS = 4
_PLACEMENT_TRIES = 140
_GEN_RETRIES = 500
_CYCLE_LEN = 3

_PLUS_SHAPES = [
    np.array([[0, 1, 1, 0], [1, 1, 1, 1], [0, 1, 1, 0]], dtype=int),
    np.array(
        [
            [0, 1, 1, 0],
            [1, 1, 1, 1],
            [1, 1, 1, 1],
            [1, 1, 1, 1],
            [0, 1, 1, 0],
        ],
        dtype=int,
    ),
]

_DIRECTIONS = [(1, 0), (-1, 0), (0, 1), (0, -1)]


def _nozzle_span(plus_mask: np.ndarray, direction: tuple[int, int]) -> tuple[int, int]:
    """Return the inclusive cross-axis span for one plus side."""
    dr, dc = map(int, direction)
    if dc == 0:
        row = plus_mask.shape[0] - 1 if dr == 1 else 0
        cols = np.where(plus_mask[row])[0]
        return int(cols[0]), int(cols[-1])
    col = plus_mask.shape[1] - 1 if dc == 1 else 0
    rows = np.where(plus_mask[:, col])[0]
    return int(rows[0]), int(rows[-1])


def _build_seed_tile(
    plus_mask: np.ndarray,
    attach_direction: tuple[int, int],
    cycle_colors: list[int],
    rng: np.random.Generator,
    *,
    force_clue: bool = False,
) -> np.ndarray | None:
    """Build a rectangular seed whose cross-axis matches the chosen plus side."""
    s, e = _nozzle_span(plus_mask, attach_direction)
    cross = int(e - s + 1)
    dr, dc = map(int, attach_direction)

    if force_clue:
        if cross != 2:
            return None
        vals = list(map(int, cycle_colors)) + [int(rng.choice(cycle_colors))]
        rng.shuffle(vals)
        return np.array(vals, dtype=int).reshape(2, 2)

    along = int(rng.choice(_SEED_ALONG_CHOICES))
    th, tw = (along, cross) if dc == 0 else (cross, along)
    n_cells = int(th * tw)
    n_colors = int(rng.integers(1, min(len(cycle_colors), n_cells) + 1))
    chosen = list(map(int, rng.choice(cycle_colors, size=n_colors, replace=False)))
    vals = list(chosen) + list(map(int, rng.choice(chosen, size=n_cells - n_colors, replace=True)))
    rng.shuffle(vals)
    return np.array(vals, dtype=int).reshape(th, tw)


def _anchor_seed_tile(
    plus_bbox: tuple[int, int, int, int],
    plus_mask: np.ndarray,
    attach_direction: tuple[int, int],
    tile: np.ndarray,
) -> tuple[int, int]:
    """Place the compact seed flush against the chosen plus side."""
    pr0, pc0, pr1, pc1 = plus_bbox
    dr, dc = map(int, attach_direction)
    th, tw = tile.shape
    s, _ = _nozzle_span(plus_mask, attach_direction)

    if dc == 0:
        c0 = int(pc0 + s)
        r0 = int(pr1 + 1) if dr == 1 else int(pr0 - th)
    else:
        r0 = int(pr0 + s)
        c0 = int(pc1 + 1) if dc == 1 else int(pc0 - tw)
    return int(r0), int(c0)


def _compute_full_units_from_seed(
    direction: tuple[int, int],
    tile: np.ndarray,
    seed_r0: int,
    seed_c0: int,
    H: int,
    W: int,
) -> int:
    """Count full tile chunks from the seed position to the border, inclusive."""
    dr, dc = map(int, direction)
    th, tw = tile.shape
    if dr == 1:
        return int((H - seed_r0) // th)
    if dr == -1:
        return int((seed_r0 + th) // th)
    if dc == 1:
        return int((W - seed_c0) // tw)
    return int((seed_c0 + tw) // tw)


def _project_cycle_coords(
    H: int,
    W: int,
    tile: np.ndarray,
    seed_r0: int,
    seed_c0: int,
    direction: tuple[int, int],
    color_map: dict[int, int],
) -> list[tuple[int, int, int]]:
    """Return the full seed+ray cells under the color-cycle rule."""
    th, tw = tile.shape
    dr, dc = map(int, direction)
    rr = int(seed_r0)
    cc = int(seed_c0)
    cur = np.array(tile, dtype=int, copy=True)
    cells: list[tuple[int, int, int]] = []

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

    return cells


def _build_control_cells(
    H: int,
    W: int,
    tile: np.ndarray,
    seed_r0: int,
    seed_c0: int,
    direction: tuple[int, int],
) -> list[tuple[int, int]]:
    """Place a border strip whose span matches the seed cross-axis for the ray."""
    th, tw = tile.shape
    dr, dc = map(int, direction)
    if dc == 0:
        row = 0 if dr == -1 else int(H - 1)
        return [(int(row), int(seed_c0 + j)) for j in range(tw)]
    col = 0 if dc == -1 else int(W - 1)
    return [(int(seed_r0 + i), int(col)) for i in range(th)]


def _allowed_ray_directions(attach_direction: tuple[int, int]) -> list[tuple[int, int]]:
    """Allow same-side or sideways projection, but not back through the plus."""
    opposite = (-int(attach_direction[0]), -int(attach_direction[1]))
    return [d for d in _DIRECTIONS if d != opposite]


def _place_plus_and_ray(
    rng: np.random.Generator,
    plus_shape: np.ndarray,
    cycle_colors: list[int],
    color_map: dict[int, int],
    H: int,
    W: int,
    occ: np.ndarray,
    plus_occ: np.ndarray,
    out: np.ndarray,
    plus_color: int,
    ctrl_color: int,
    used_row_spans: set[tuple[int, ...]],
    used_col_spans: set[tuple[int, ...]],
    *,
    force_clue: bool = False,
    require_mismatch: bool = False,
) -> dict | None:
    """Try to place one plus, one seed, one control strip, and one full ray."""
    ph, pw = plus_shape.shape
    min_r = 2
    max_r = H - ph - 2
    min_c = 2
    max_c = W - pw - 2
    if max_r < min_r or max_c < min_c:
        return None

    for _ in range(_PLACEMENT_TRIES):
        r0 = int(rng.integers(min_r, max_r + 1))
        c0 = int(rng.integers(min_c, max_c + 1))
        plus_cells = [
            (int(r0 + dr), int(c0 + dc))
            for dr in range(ph)
            for dc in range(pw)
            if int(plus_shape[dr, dc]) == 1
        ]

        if any(int(occ[r, c]) != 0 for r, c in plus_cells):
            continue
        if any(int(plus_occ[r, c]) == 1 for r, c in plus_cells):
            continue

        attach_dirs = list(_DIRECTIONS)
        rng.shuffle(attach_dirs)
        for attach_direction in attach_dirs:
            tile = _build_seed_tile(
                plus_shape,
                attach_direction,
                cycle_colors,
                rng,
                force_clue=bool(force_clue),
            )
            if tile is None:
                continue

            seed_r0, seed_c0 = _anchor_seed_tile(
                (r0, c0, r0 + ph - 1, c0 + pw - 1),
                plus_shape,
                attach_direction,
                tile,
            )
            th, tw = tile.shape
            if seed_r0 < 1 or seed_c0 < 1 or seed_r0 + th > H - 1 or seed_c0 + tw > W - 1:
                continue

            seed_rows = tuple(range(int(seed_r0), int(seed_r0 + th)))
            seed_cols = tuple(range(int(seed_c0), int(seed_c0 + tw)))
            if seed_rows in used_row_spans or seed_cols in used_col_spans:
                continue

            ray_dirs = _allowed_ray_directions(attach_direction)
            rng.shuffle(ray_dirs)
            if require_mismatch:
                ray_dirs = [d for d in ray_dirs if d != tuple(map(int, attach_direction))]
                if not ray_dirs:
                    continue

            for ray_direction in ray_dirs:
                units = _compute_full_units_from_seed(
                    ray_direction, tile, int(seed_r0), int(seed_c0), H, W
                )
                need = _CLUE_MIN_FULL_UNITS if force_clue else _MIN_RAY_UNITS
                if units < need:
                    continue

                cells = _project_cycle_coords(
                    H, W, tile, int(seed_r0), int(seed_c0), ray_direction, color_map
                )
                if any(int(occ[r, c]) != 0 for r, c, _v in cells):
                    continue
                if any(int(plus_occ[r, c]) == 1 for r, c, _v in cells):
                    continue

                control_cells = _build_control_cells(
                    H, W, tile, int(seed_r0), int(seed_c0), ray_direction
                )
                control_set = {(int(r), int(c)) for r, c in control_cells}
                if any(int(occ[r, c]) != 0 for r, c in control_set):
                    continue

                for r, c in plus_cells:
                    occ[r, c] = 1
                    plus_occ[r, c] = 1
                    out[r, c] = int(plus_color)
                    for dr2, dc2 in _DIRECTIONS:
                        rr = int(r + dr2)
                        cc = int(c + dc2)
                        if 0 <= rr < H and 0 <= cc < W:
                            plus_occ[rr, cc] = 1

                for r, c, v in cells:
                    occ[r, c] = 2
                    out[r, c] = int(v)

                for r, c in control_set:
                    occ[r, c] = 3
                    out[r, c] = int(ctrl_color)

                used_row_spans.add(seed_rows)
                used_col_spans.add(seed_cols)
                return {
                    "tile": np.array(tile, dtype=int, copy=True),
                    "cells": cells,
                    "control_set": control_set,
                    "seed_r0": int(seed_r0),
                    "seed_c0": int(seed_c0),
                    "units": int(units),
                    "attach_direction": tuple(map(int, attach_direction)),
                    "ray_direction": tuple(map(int, ray_direction)),
                    "is_clue": bool(force_clue),
                }

    return None


def _build_input_from_output(
    out: np.ndarray,
    pluses: list[dict],
    bg: int,
    keep_idx: int,
) -> np.ndarray:
    """Keep one full clue ray and collapse all others back to compact seeds."""
    inp = np.array(out, dtype=int, copy=True)
    for i, p in enumerate(pluses):
        if int(i) == int(keep_idx):
            continue
        th, tw = p["tile"].shape
        sr0 = int(p["seed_r0"])
        sc0 = int(p["seed_c0"])
        seed_set = {(int(sr0 + ii), int(sc0 + jj)) for ii in range(th) for jj in range(tw)}
        for r, c, _v in p["cells"]:
            rc = (int(r), int(c))
            if rc in p["control_set"]:
                continue
            if rc not in seed_set:
                inp[rc] = int(bg)
    return inp


def _gen_attempt(rng: np.random.Generator) -> np.ndarray | None:
    H = int(rng.integers(*_H_RANGE))
    W = int(rng.integers(*_W_RANGE))

    bg = int(rng.integers(0, 10))
    others = [int(c) for c in range(10) if int(c) != int(bg)]
    plus_color = int(rng.choice(others))
    rem = [c for c in others if c != plus_color]
    ctrl_color = int(rng.choice(rem))
    cycle_colors = list(map(int, rng.choice([c for c in rem if c != ctrl_color], size=_CYCLE_LEN, replace=False)))
    rng.shuffle(cycle_colors)
    color_map = {
        int(cycle_colors[i]): int(cycle_colors[(i + 1) % _CYCLE_LEN])
        for i in range(_CYCLE_LEN)
    }

    plus_shape = _PLUS_SHAPES[int(rng.integers(0, len(_PLUS_SHAPES)))]
    N = int(rng.integers(_N_PLUS_RANGE[0], _N_PLUS_RANGE[1] + 1))

    out = np.full((H, W), bg, dtype=int)
    occ = np.zeros((H, W), dtype=int)
    plus_occ = np.zeros((H, W), dtype=int)
    used_row_spans: set[tuple[int, ...]] = set()
    used_col_spans: set[tuple[int, ...]] = set()

    pluses: list[dict] = []

    clue = _place_plus_and_ray(
        rng,
        plus_shape,
        cycle_colors,
        color_map,
        H,
        W,
        occ,
        plus_occ,
        out,
        plus_color,
        ctrl_color,
        used_row_spans,
        used_col_spans,
        force_clue=True,
    )
    if clue is None or int(clue["units"]) < _CLUE_IDEAL_UNITS:
        return None
    pluses.append(clue)

    for idx in range(N - 1):
        info = _place_plus_and_ray(
            rng,
            plus_shape,
            cycle_colors,
            color_map,
            H,
            W,
            occ,
            plus_occ,
            out,
            plus_color,
            ctrl_color,
            used_row_spans,
            used_col_spans,
            force_clue=False,
            require_mismatch=bool(idx == 0),
        )
        if info is None:
            return None
        pluses.append(info)

    mismatches = sum(int(p["attach_direction"] != p["ray_direction"]) for p in pluses)
    compact_mismatches = sum(
        int((not p["is_clue"]) and p["attach_direction"] != p["ray_direction"]) for p in pluses
    )
    distinct_dirs = {tuple(map(int, p["ray_direction"])) for p in pluses}
    has_vertical = any(int(d[0]) != 0 for d in distinct_dirs)
    has_horizontal = any(int(d[1]) != 0 for d in distinct_dirs)
    if mismatches < 1 or compact_mismatches < 1 or len(distinct_dirs) < 2:
        return None
    if not has_vertical or not has_horizontal:
        return None

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
