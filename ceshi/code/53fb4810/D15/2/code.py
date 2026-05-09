# Variant of 53fb4810
# Dimension: D15 — 背景色与精灵色约束 (meta)
# Change: 背景为单一最常见颜色（0-9任意），'+'精灵为单一非背景颜色，所有精灵共享同一颜色... → 精灵内部多色

from __future__ import annotations

import itertools

import numpy as np

from arc_utils import bbox_coords, crop_bbox, get_bg, label_mask


# Design analysis for the D15 variant
# a) Concrete behavior:
#    - Keep the original geometry: each object is still one plus-like sprite with one attached seed tile,
#      and the output still extends that seed periodically away from the sprite until the border.
#    - The changed part is the anchor sprite itself. It is no longer a single color. Every plus uses the
#      same two-color template: interior cells are one color and all arm-tip / outer-boundary cells are a
#      second color. The seed is attached only to the arm-tip color.
#    - Small trace example: in the original task a 3x3 plus could be all red and a 1x2 seed attached below.
#      The solver found red as the dominant large-component color, labeled red components, found the adjacent
#      seed, then extended it downward. In this variant the same 3x3 footprint might be
#        0 8 0
#        8 3 8
#        0 8 0
#      with the seed adjacent to one of the 8 cells. A single-color component search now fails because neither
#      3 nor 8 alone forms the whole plus; the solver must first recover the two-color union that repeats as
#      the shared plus footprint, then use the tip color (8 here) to locate valid attachments.
# b) Complexity calibration:
#    - Only one new freedom is added: the plus uses exactly two colors, shared by all pluses in a grid.
#      The geometry, counts, seed sizes, and ray mechanics stay original. This makes the visual change clear
#      without exploding the hypothesis space.
#    - The two-color pattern is repeated across all pluses in the same sample, so the new rule has visible
#      support in 2-4 examples rather than appearing as a one-off accident.
# c) Hidden assumptions in the original main():
#    - `detect_dominant_big_component_color(...)` assumed one color alone identifies every plus. That breaks.
#      Replace it with a search over color pairs whose union yields repeated symmetric big components.
#    - `plus_lab = label_mask(g == plus_col, ...)` assumed the plus mask is monochrome. Replace it with union
#      labeling over the detected pair.
#    - `protect={plus_col}` assumed only one sprite color must be preserved during extension. Replace it with
#      both plus colors.
#    - Adjacency collection from every plus cell was harmless before. In the variant the seed is defined to
#      touch the arm-tip color specifically, so the solver now gathers candidate markers only from tip-colored
#      plus cells.
#    - The original `assert` alignment checks could crash on malformed inputs. Here they are made explicit
#      guards so generator verification stays robust across retries.
# d) Placement legality:
#    - Generator still uses output-first placement and checks that full rays do not collide with existing
#      sprites or rays.
#    - Marker colors exclude both plus colors, so the two-color union isolates the pluses cleanly for the solver.
#    - The plus footprint remains identical across objects; seed cross-axis span still matches the nozzle span.
# e) Visual observability:
#    - The pluses visibly change from monochrome anchors to structured two-color anchors. Because the seed always
#      touches the outer boundary color, a human can infer that the same extension rule applies but the anchor
#      must now be read as a composite object instead of by a single color.
# f) Visual cleanliness:
#    - Only two plus colors are used, and rays never overwrite plus cells. The grid stays legible: plus body,
#      seed, and repeated ray remain visually separable.
# g) Color-change inferability:
#    - No hidden color-id ordering is used. The solver looks for repeated geometry formed by a color pair, and
#      the generator keeps that pair consistent within each sample. The change is structural, not numeric.


def _is_symmetric_plus_mask(mask: np.ndarray) -> bool:
    """Heuristic footprint check for the repeated plus-like anchor components."""
    if mask.shape[0] < 3 or mask.shape[1] < 3:
        return False
    if not np.array_equal(mask, mask[::-1, :]):
        return False
    if not np.array_equal(mask, mask[:, ::-1]):
        return False
    return bool(mask.any())


def detect_plus_color_pair(
    grid: np.ndarray,
    *,
    bg: int | None = None,
    min_h: int = 3,
    min_w: int = 3,
    connectivity: int = 4,
) -> tuple[int, int] | None:
    """
    Choose the pair of non-bg colors whose union forms the repeated plus components.

    Candidate pair requirements:
    - the union contains at least one big component,
    - every big component has bbox >= (min_h x min_w),
    - each big component is LR+UD symmetric,
    - all big components share the same cropped footprint.
    Tie-break: more matching components, then larger total area.
    """
    if bg is None:
        bg = get_bg(grid)
    bg = int(bg)

    colors = sorted(set(map(int, grid.flatten())) - {bg})
    best_key: tuple[int, int] | None = None
    best_pair: tuple[int, int] | None = None

    for a, b in itertools.combinations(colors, 2):
        union = (grid == int(a)) | (grid == int(b))
        lab, n = label_mask(union, connectivity=int(connectivity))
        big_masks: list[np.ndarray] = []
        total_area = 0

        for k in range(1, int(n) + 1):
            coords = np.argwhere(lab == int(k))
            if coords.size == 0:
                continue
            r0, c0, r1, c1 = bbox_coords(coords)
            h = int(r1 - r0 + 1)
            w = int(c1 - c0 + 1)
            if h < int(min_h) or w < int(min_w):
                continue
            comp = lab[int(r0) : int(r1) + 1, int(c0) : int(c1) + 1] == int(k)
            if not _is_symmetric_plus_mask(comp):
                continue
            big_masks.append(comp)
            total_area += int(coords.shape[0])

        if not big_masks:
            continue
        if any(not np.array_equal(m, big_masks[0]) for m in big_masks[1:]):
            continue

        key = (int(len(big_masks)), int(total_area))
        if best_key is None or key > best_key:
            best_key = key
            best_pair = (int(a), int(b))

    return best_pair


def detect_tip_color(
    grid: np.ndarray,
    plus_lab: np.ndarray,
    n_plus: int,
    colors: tuple[int, int],
) -> int | None:
    """
    Identify which plus color occupies every outer nozzle edge.

    In this variant the marker is attached to the arm-tip color, so we use the repeated
    extreme-row / extreme-col pattern to recover that color.
    """
    for col in colors:
        ok = True
        for pk in range(1, int(n_plus) + 1):
            plus_mask = plus_lab == int(pk)
            coords = np.argwhere(plus_mask)
            if coords.size == 0:
                ok = False
                break
            r0, c0, r1, c1 = bbox_coords(coords)
            sub_mask = plus_mask[int(r0) : int(r1) + 1, int(c0) : int(c1) + 1]
            sub_grid = grid[int(r0) : int(r1) + 1, int(c0) : int(c1) + 1]

            top_cols = np.where(sub_mask[0])[0]
            bottom_cols = np.where(sub_mask[-1])[0]
            left_rows = np.where(sub_mask[:, 0])[0]
            right_rows = np.where(sub_mask[:, -1])[0]
            if (
                top_cols.size == 0
                or bottom_cols.size == 0
                or left_rows.size == 0
                or right_rows.size == 0
            ):
                ok = False
                break
            if not np.all(sub_grid[0, top_cols] == int(col)):
                ok = False
                break
            if not np.all(sub_grid[-1, bottom_cols] == int(col)):
                ok = False
                break
            if not np.all(sub_grid[left_rows, 0] == int(col)):
                ok = False
                break
            if not np.all(sub_grid[right_rows, -1] == int(col)):
                ok = False
                break
        if ok:
            return int(col)
    return None


def extend_repeating_tile(
    out: np.ndarray,
    tile: np.ndarray,
    r0: int,
    c0: int,
    direction: tuple[int, int],
    *,
    bg: int,
    protect: set[int] | None = None,
) -> None:
    """Local helper: extend a tile pattern from (r0,c0) until boundary."""
    protect = set() if protect is None else set(map(int, protect))
    H, W = out.shape
    h, w = tile.shape
    dr, dc = map(int, direction)
    dr = 0 if dr == 0 else int(dr / abs(dr))
    dc = 0 if dc == 0 else int(dc / abs(dc))
    bg = int(bg)
    r0 = int(r0)
    c0 = int(c0)

    def put(rr: int, cc: int, v: int) -> None:
        if v == bg:
            return
        if int(out[rr, cc]) in protect:
            return
        out[rr, cc] = int(v)

    if (dr, dc) == (0, 1):
        start_c = c0 + w
        for k, tc in enumerate(range(start_c, W)):
            col_idx = k % w
            for i in range(h):
                put(r0 + i, tc, int(tile[i, col_idx]))
    elif (dr, dc) == (0, -1):
        for k, tc in enumerate(range(c0 - 1, -1, -1), start=1):
            col_idx = (-k) % w
            for i in range(h):
                put(r0 + i, tc, int(tile[i, col_idx]))
    elif (dr, dc) == (1, 0):
        start_r = r0 + h
        for k, tr in enumerate(range(start_r, H)):
            row_idx = k % h
            for j in range(w):
                put(tr, c0 + j, int(tile[row_idx, j]))
    elif (dr, dc) == (-1, 0):
        for k, tr in enumerate(range(r0 - 1, -1, -1), start=1):
            row_idx = (-k) % h
            for j in range(w):
                put(tr, c0 + j, int(tile[row_idx, j]))
    else:
        raise ValueError("extend_repeating_tile: direction must be one of (±1,0),(0,±1)")


def main(grid: np.ndarray) -> np.ndarray:
    """
    Extend marker tiles outward from two-color '+' sprites by repeating them to the grid edge.

    Input structure:
    - bg is still the most common color.
    - Each plus-shaped anchor is now a shared two-color sprite: one interior color plus one arm-tip color.
    - Small colored seed tiles use other colors and touch the arm-tip color on exactly one side; one seed's ray is already pre-extended.

    Rule:
    - Recover the repeated plus footprint by detecting the pair of colors whose union forms the large symmetric anchor components.
    - Use the arm-tip color to find which marker component is attached to which side of each plus, then repeat that seed away from the plus until the boundary.
    - Neither of the plus colors may be overwritten while extending the rays.

    Key insight:
    - The marker direction is still encoded by side-adjacency, but the anchor must now be recognized as a composite two-color object rather than by a single dominant color.
    """
    g = np.array(grid, dtype=int)
    out = g.copy()
    H, W = g.shape

    bg = int(get_bg(g))
    plus_pair = detect_plus_color_pair(g, bg=bg, min_h=3, min_w=3, connectivity=4)
    if plus_pair is None:
        return out

    plus_mask_all = np.isin(g, list(plus_pair))
    plus_lab, pn = label_mask(plus_mask_all, connectivity=4)
    if int(pn) == 0:
        return out

    tip_color = detect_tip_color(g, plus_lab, int(pn), plus_pair)
    if tip_color is None:
        return out

    plus_colors = {int(plus_pair[0]), int(plus_pair[1])}
    non_mask = (g != int(bg)) & (~np.isin(g, list(plus_colors)))
    mark_lab, _mn = label_mask(non_mask, connectivity=4)

    for pk in range(1, int(pn) + 1):
        plus_mask = plus_lab == int(pk)
        plus_coords = np.argwhere(plus_mask)
        if plus_coords.size == 0:
            continue

        tip_mask = plus_mask & (g == int(tip_color))
        adj_ids: set[int] = set()
        for r, c in np.argwhere(tip_mask):
            r = int(r)
            c = int(c)
            for dr, dc in ((1, 0), (-1, 0), (0, 1), (0, -1)):
                rr = r + dr
                cc = c + dc
                if 0 <= rr < H and 0 <= cc < W and int(mark_lab[rr, cc]) > 0:
                    adj_ids.add(int(mark_lab[rr, cc]))

        for cid in sorted(adj_ids):
            pts = np.argwhere(mark_lab == int(cid))
            if pts.size == 0:
                continue
            r0, c0, r1, c1 = bbox_coords(pts)
            tile = crop_bbox(g, (int(r0), int(c0), int(r1), int(c1)))

            dirs: set[tuple[int, int]] = set()
            for pr, pc in pts:
                pr = int(pr)
                pc = int(pc)
                for dr, dc in ((1, 0), (-1, 0), (0, 1), (0, -1)):
                    rr = pr + dr
                    cc = pc + dc
                    if 0 <= rr < H and 0 <= cc < W and bool(tip_mask[rr, cc]):
                        dirs.add((-int(dr), -int(dc)))
            if len(dirs) != 1:
                continue

            dr, dc = next(iter(dirs))

            pr0, pc0 = plus_coords.min(axis=0)
            pr1, pc1 = plus_coords.max(axis=0)
            if dc == 0:
                nozzle_row = int(pr1) if dr == 1 else int(pr0)
                nozzle_cols = sorted(
                    int(cc) for cc in range(int(pc0), int(pc1) + 1) if plus_mask[nozzle_row, cc]
                )
                seed_cols = list(range(int(c0), int(c1) + 1))
                if seed_cols != nozzle_cols:
                    continue
            else:
                nozzle_col = int(pc1) if dc == 1 else int(pc0)
                nozzle_rows = sorted(
                    int(rr) for rr in range(int(pr0), int(pr1) + 1) if plus_mask[rr, nozzle_col]
                )
                seed_rows = list(range(int(r0), int(r1) + 1))
                if seed_rows != nozzle_rows:
                    continue

            extend_repeating_tile(
                out,
                tile,
                int(r0),
                int(c0),
                (int(dr), int(dc)),
                bg=int(bg),
                protect=plus_colors,
            )

    return out


# --------------- generator (D15 variant) ---------------

_H_RANGE = (13, 29)
_W_RANGE = (13, 23)
_N_PLUS_RANGE = (2, 4)
_SEED_ALONG_CHOICES = [1, 2]
_SEED_MAX_COLORS = 4
_SEED_MAX_CROSS = 2
_MIN_RAY_UNITS = 2
_PRE_EXT_IDEAL_UNITS = 10
_PLACEMENT_TRIES = 100
_GEN_RETRIES = 300

_PLUS_SHAPES = [
    np.array([[0, 1, 0], [1, 1, 1], [0, 1, 0]], dtype=int),
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


def _plus_role_template(plus_shape: np.ndarray) -> np.ndarray:
    """
    Encode the multicolor sprite template.

    0 = empty, 1 = core color, 2 = arm-tip / boundary color.
    """
    mask = plus_shape.astype(bool)
    role = np.zeros_like(plus_shape, dtype=int)
    H, W = plus_shape.shape
    for r in range(H):
        for c in range(W):
            if not mask[r, c]:
                continue
            boundary = False
            for dr, dc in _DIRECTIONS:
                rr = r + dr
                cc = c + dc
                if not (0 <= rr < H and 0 <= cc < W) or not mask[rr, cc]:
                    boundary = True
                    break
            role[r, c] = 2 if boundary else 1
    return role


def _nozzle_span(plus_mask: np.ndarray, direction: tuple[int, int]) -> tuple[int, int]:
    """Return (start, end) inclusive of nozzle cells along the cross-axis."""
    dr, dc = direction
    if dc == 0:
        row = plus_mask.shape[0] - 1 if dr == 1 else 0
        cols = np.where(plus_mask[row])[0]
        return int(cols[0]), int(cols[-1])
    col = plus_mask.shape[1] - 1 if dc == 1 else 0
    rows = np.where(plus_mask[:, col])[0]
    return int(rows[0]), int(rows[-1])


def _build_seed_tile(
    plus_mask: np.ndarray,
    direction: tuple[int, int],
    palette: list[int],
    rng: np.random.Generator,
) -> np.ndarray | None:
    """Build a seed tile whose cross-axis matches the nozzle span."""
    s, e = _nozzle_span(plus_mask, direction)
    cross = e - s + 1
    if cross > _SEED_MAX_CROSS:
        return None

    along = int(rng.choice(_SEED_ALONG_CHOICES))
    if along == 1 and cross == 1:
        along = 2

    dr, dc = direction
    th, tw = (along, cross) if dc == 0 else (cross, along)

    n_cells = th * tw
    max_k = min(_SEED_MAX_COLORS, n_cells, len(palette))
    n_colors = int(rng.integers(1, max_k + 1))
    chosen = list(rng.choice(palette, size=n_colors, replace=False))
    vals = list(chosen) + list(rng.choice(chosen, size=n_cells - n_colors, replace=True))
    rng.shuffle(vals)
    return np.array(vals, dtype=int).reshape(th, tw)


def _compute_ray_units(
    direction: tuple[int, int],
    tile: np.ndarray,
    r0: int,
    c0: int,
    ph: int,
    pw: int,
    H: int,
    W: int,
) -> int:
    """Compute how many tile repetitions fit along the ray direction."""
    dr, dc = direction
    th, tw = tile.shape
    if dr == 1:
        return (H - (r0 + ph)) // th
    if dr == -1:
        return r0 // th
    if dc == 1:
        return (W - (c0 + pw)) // tw
    return c0 // tw


def _project_tile_coords(
    H: int,
    W: int,
    plus_bbox: tuple[int, int, int, int],
    plus_mask: np.ndarray,
    direction: tuple[int, int],
    tile: np.ndarray,
) -> tuple[list[tuple[int, int, int]], int, int]:
    """Compute (r, c, value) list for the full projected ray (seed + repetitions)."""
    pr0, pc0, pr1, pc1 = plus_bbox
    dr, dc = direction
    th, tw = tile.shape
    s, _ = _nozzle_span(plus_mask, direction)

    if dc == 0:
        c0 = pc0 + s
        r0 = (pr1 + 1) if dr == 1 else (pr0 - th)
    else:
        r0 = pr0 + s
        c0 = (pc1 + 1) if dc == 1 else (pc0 - tw)

    cells: list[tuple[int, int, int]] = []
    if dc == 0:
        r = r0
        step = th if dr == 1 else -th
        while (r < H) if dr == 1 else (r + th > 0):
            for i in range(th):
                rr = r + i
                if 0 <= rr < H:
                    for j in range(tw):
                        cells.append((rr, c0 + j, int(tile[i, j])))
            r += step
    else:
        c = c0
        step = tw if dc == 1 else -tw
        while (c < W) if dc == 1 else (c + tw > 0):
            for i in range(th):
                for j in range(tw):
                    cc = c + j
                    if 0 <= cc < W:
                        cells.append((r0 + i, cc, int(tile[i, j])))
            c += step

    return cells, r0, c0


def _place_plus_and_ray(
    rng: np.random.Generator,
    plus_roles: np.ndarray,
    palette: list[int],
    H: int,
    W: int,
    occ: np.ndarray,
    plus_occ: np.ndarray,
    out: np.ndarray,
    core_color: int,
    tip_color: int,
) -> dict | None:
    """Try to place one plus sprite with its ray. Returns info dict or None."""
    ph, pw = plus_roles.shape
    plus_mask = plus_roles > 0

    for _ in range(_PLACEMENT_TRIES):
        r0 = int(rng.integers(1, H - ph))
        c0 = int(rng.integers(1, W - pw))
        plus_cells = [
            (r0 + dr, c0 + dc, int(plus_roles[dr, dc]))
            for dr in range(ph)
            for dc in range(pw)
            if plus_roles[dr, dc] > 0
        ]

        if any(occ[r, c] != 0 for r, c, _role in plus_cells):
            continue
        if any(plus_occ[r, c] == 1 for r, c, _role in plus_cells):
            continue

        dirs = list(_DIRECTIONS)
        rng.shuffle(dirs)
        for d in dirs:
            tile = _build_seed_tile(plus_mask, d, palette, rng)
            if tile is None:
                continue
            units = _compute_ray_units(d, tile, r0, c0, ph, pw, H, W)
            if units < _MIN_RAY_UNITS:
                continue

            cells, sr0, sc0 = _project_tile_coords(
                H, W, (r0, c0, r0 + ph - 1, c0 + pw - 1), plus_mask, d, tile
            )
            if any(occ[r, c] != 0 for r, c, _ in cells if 0 <= r < H and 0 <= c < W):
                continue

            for r, c, role in plus_cells:
                occ[r, c] = 1
                out[r, c] = int(core_color if role == 1 else tip_color)
                for dr2, dc2 in _DIRECTIONS:
                    rr = r + dr2
                    cc = c + dc2
                    if 0 <= rr < H and 0 <= cc < W:
                        plus_occ[rr, cc] = 1
            for r, c, v in cells:
                occ[r, c] = 2
                out[r, c] = v

            return {
                "tile": tile,
                "cells": cells,
                "seed_r0": sr0,
                "seed_c0": sc0,
                "units": units,
            }

    return None


def _build_input_from_output(
    out: np.ndarray,
    pluses: list[dict],
    bg: int,
    rng: np.random.Generator,
) -> np.ndarray:
    """Erase non-pre-extended rays from the output to produce the input."""
    pre_idx = max(range(len(pluses)), key=lambda i: pluses[i]["units"])
    if pluses[pre_idx]["units"] < _PRE_EXT_IDEAL_UNITS:
        candidates = [i for i, p in enumerate(pluses) if p["units"] >= _PRE_EXT_IDEAL_UNITS]
        if candidates:
            pre_idx = int(rng.choice(candidates))

    inp = out.copy()
    for i, p in enumerate(pluses):
        if i == pre_idx:
            continue
        th, tw = p["tile"].shape
        sr0, sc0 = p["seed_r0"], p["seed_c0"]
        seed_set = {(sr0 + ii, sc0 + jj) for ii in range(th) for jj in range(tw)}
        for r, c, _ in p["cells"]:
            if (r, c) not in seed_set:
                inp[r, c] = int(bg)
    return inp


def _gen_attempt(rng: np.random.Generator) -> np.ndarray | None:
    H = int(rng.integers(*_H_RANGE))
    W = int(rng.integers(*_W_RANGE))

    bg = int(rng.integers(0, 10))
    others = [c for c in range(10) if c != bg]
    core_color, tip_color = map(int, rng.choice(others, size=2, replace=False))
    palette = [c for c in others if c not in {core_color, tip_color}]

    plus_shape = _PLUS_SHAPES[int(rng.integers(0, len(_PLUS_SHAPES)))]
    plus_roles = _plus_role_template(plus_shape)
    N = int(rng.integers(*_N_PLUS_RANGE))

    out = np.full((H, W), bg, dtype=int)
    occ = np.zeros((H, W), dtype=int)
    plus_occ = np.zeros((H, W), dtype=int)

    pluses: list[dict] = []
    for _ in range(N):
        info = _place_plus_and_ray(
            rng,
            plus_roles,
            palette,
            H,
            W,
            occ,
            plus_occ,
            out,
            core_color,
            tip_color,
        )
        if info is None:
            return None
        pluses.append(info)

    inp = _build_input_from_output(out, pluses, bg, rng)

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
    raise RuntimeError("generate_input: failed after %d attempts" % _GEN_RETRIES)
