# Combo variant of 53fb4810 (index 20)
# combo_spec: D7/0+D10/0+D15/0
# Name: 多色异形精灵消失
#
# Design analysis for the fused variant
# a) Concrete fused behavior and trace:
#    - Base task: 2-3 identical same-colored '+' sprites each own one adjacent seed tile;
#      one ray is already fully repeated in the input; output extends the remaining seeds
#      while keeping every '+' visible.
#    - Fused task: each source sprite may use a different silhouette and a different color,
#      but the local ray rule is unchanged: the touching seed determines a cardinal direction
#      and repeats periodically away from that source. After all rays are inferred, every
#      source sprite is erased to background, so the output shows only ray traces.
#    - Small trace example: a blue 3x3 cross with a 1x2 seed attached below produces a
#      downward striped ray; a red 3x4 fat cross with a 2x1 seed attached on the right
#      produces a rightward repeated column pair. In the original task both anchors would
#      share one color/shape and remain visible in the output. Here they differ in both
#      color and shape, and both anchor footprints disappear after projection.
# b) Complexity calibration:
#    - I keep the original object family small: 2-3 plus-like anchors, 3 template families,
#      seed cross-span <= 2, and 1-4 marker colors per seed.
#    - To make the fusion visually obvious, each generated grid uses unique colors per plus
#      and at least two distinct plus silhouettes. That gives repeated support for both
#      "multi-color anchors behave the same way" and "shape must be read locally".
#    - Marker colors are sampled from a palette disjoint from all plus colors, which prevents
#      ambiguity when plus footprints are erased in the output.
# c) Hidden-assumption audit of the original solver:
#    - `detect_dominant_big_component_color(...)` is invalid because pluses no longer share a
#      global color.
#    - `plus_lab = label_mask(g == plus_col)` is invalid for the same reason.
#    - `non_mask = (g != bg) & (g != plus_col)` is also invalid because other plus colors
#      would be mistaken for markers.
#    - `protect={plus_col}` is too weak because there are multiple plus colors.
#    - Keeping `out = g.copy()` without a final erase is invalid because D7 requires every
#      source sprite to disappear in the output.
#    - Any commentary implying one shared plus template is now wrong; nozzle-width checks must
#      remain component-local because each plus can expose a different arm-tip span.
# d) Placement legality:
#    - Generation remains output-first: place full rays plus full colored sources into a clean
#      scene, then derive the input by erasing the repeated suffix of every non-demo ray.
#    - Occupancy checks reject plus-plus overlap, ray-ray overlap, and ray-plus overlap.
#    - A one-cell moat prevents different pluses from touching. Seed cross-span is forced to
#      match the chosen arm tip, and shapes whose left/right arm would require cross-span 3
#      simply reject those directions.
# e) Visual observability:
#    - A human sees same-rule periodic rays emitted from differently colored, differently
#      shaped anchors; then all anchors vanish in the output. That triad is directly visible
#      from 2-3 examples without relying on color ids or hidden ordering.
# f) Visual cleanliness:
#    - Rays do not overlap each other or any source footprint. The grid remains sparse because
#      only one clue ray is pre-expanded in the input and every non-demo source keeps only its
#      compact seed.
# g) Color inferability:
#    - No color arithmetic or ordering is used. The only color rule is structural: plus colors
#      are unique per source and marker/ray colors avoid every plus color.
# h) New solver assumptions and generator guarantees:
#    - Solver assumption: every source sprite is a monochrome connected component with bbox at
#      least 3x3. Generator guarantee: all plus templates satisfy that.
#    - Solver assumption: each plus has exactly one touching marker component with a unique
#      away direction. Generator guarantee: one seed per plus and collision-free spacing.
#    - Solver assumption: seed bbox span equals the local nozzle span of its own plus. Generator
#      guarantee: tiles are built from the chosen shape+direction pair using that exact span.

from __future__ import annotations

import numpy as np

from arc_utils import bbox_coords, crop_bbox, get_bg, label_mask


def detect_big_plus_components(
    grid: np.ndarray,
    *,
    bg: int | None = None,
    min_h: int = 3,
    min_w: int = 3,
    connectivity: int = 4,
) -> list[dict]:
    """
    Local helper (only used by this solver): collect all non-bg monochrome components whose
    bbox is at least (min_h x min_w). In this task family those are the plus sprites.
    """
    if bg is None:
        bg = get_bg(grid)
    bg = int(bg)

    comps: list[dict] = []
    for col in sorted(set(map(int, grid.flatten())) - {int(bg)}):
        lab, n = label_mask(grid == int(col), connectivity=int(connectivity))
        for k in range(1, int(n) + 1):
            coords = np.argwhere(lab == int(k))
            if coords.size == 0:
                continue
            r0, c0, r1, c1 = bbox_coords(coords)
            h = int(r1 - r0 + 1)
            w = int(c1 - c0 + 1)
            if h >= int(min_h) and w >= int(min_w):
                comps.append(
                    {
                        "color": int(col),
                        "coords": coords,
                        "bbox": (int(r0), int(c0), int(r1), int(c1)),
                    }
                )

    comps.sort(key=lambda comp: (comp["bbox"][0], comp["bbox"][1], comp["color"]))
    return comps


def extend_repeating_tile(
    out: np.ndarray,
    tile: np.ndarray,
    r0: int,
    c0: int,
    direction: tuple[int, int],
    *,
    bg: int,
    protect_mask: np.ndarray | None = None,
) -> None:
    """Local helper (only used by this solver): extend a tile pattern from (r0,c0) until boundary."""
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
        if protect_mask is not None and bool(protect_mask[rr, cc]):
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


def _marker_direction(
    pts: np.ndarray,
    plus_mask: np.ndarray,
    H: int,
    W: int,
) -> tuple[int, int] | None:
    """Return the unique direction from one plus component to its touching marker component."""
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


def _edge_span_indices(plus_mask: np.ndarray, direction: tuple[int, int]) -> list[int]:
    """Return the occupied edge positions along the cross-axis for one plus component."""
    coords = np.argwhere(plus_mask)
    pr0, pc0 = coords.min(axis=0)
    pr1, pc1 = coords.max(axis=0)
    dr, dc = map(int, direction)

    if dc == 0:
        nozzle_row = int(pr1) if dr == 1 else int(pr0)
        return [
            int(cc)
            for cc in range(int(pc0), int(pc1) + 1)
            if bool(plus_mask[nozzle_row, cc])
        ]
    nozzle_col = int(pc1) if dc == 1 else int(pc0)
    return [
        int(rr)
        for rr in range(int(pr0), int(pr1) + 1)
        if bool(plus_mask[rr, nozzle_col])
    ]


def main(grid: np.ndarray) -> np.ndarray:
    """
    Extend every attached marker tile as a periodic ray from its own colored '+' source, then erase all sources.

    Input structure:
    - The background is the most common color.
    - There are 2-3 separate '+'-like monochrome source sprites, and those sprites may differ in both color and silhouette.
    - Each source has exactly one adjacent marker tile; one source may already show its full repeated ray as the clue.

    Rule:
    - Detect each large monochrome source component independently, find the touching marker component, and infer the away-from-source direction from the side of attachment.
    - Check the marker bbox against that source's own arm-tip span, then repeat the marker tile periodically in the same direction until the boundary.
    - After all rays are reconstructed, erase every original source pixel back to the background so only the ray traces remain.

    Key insight:
    - Color can no longer unify the sources, and shape can no longer be assumed global, so the solver must reason per connected source component.
    - The local geometry still tells everything: touching side gives direction, arm-tip span gives seed width, and the final disappearance rule removes the anchors after extension.
    """
    g = np.array(grid, dtype=int)
    out = g.copy()
    H, W = g.shape

    bg = int(get_bg(g))
    pluses = detect_big_plus_components(g, bg=bg, min_h=3, min_w=3, connectivity=4)
    if not pluses:
        return out

    plus_union_mask = np.zeros_like(g, dtype=bool)
    for plus in pluses:
        coords = plus["coords"]
        plus_union_mask[coords[:, 0], coords[:, 1]] = True

    mark_lab, _mn = label_mask((g != int(bg)) & (~plus_union_mask), connectivity=4)

    for plus in pluses:
        plus_coords = plus["coords"]
        if plus_coords.size == 0:
            continue

        plus_mask = np.zeros_like(g, dtype=bool)
        plus_mask[plus_coords[:, 0], plus_coords[:, 1]] = True

        adj_ids: set[int] = set()
        for r, c in plus_coords:
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

            direction = _marker_direction(pts, plus_mask, H, W)
            if direction is None:
                continue

            r0, c0, r1, c1 = bbox_coords(pts)
            tile = crop_bbox(g, (int(r0), int(c0), int(r1), int(c1)))
            dr, dc = direction
            nozzle_span = _edge_span_indices(plus_mask, direction)
            seed_span = (
                list(range(int(c0), int(c1) + 1))
                if dc == 0
                else list(range(int(r0), int(r1) + 1))
            )
            if seed_span != nozzle_span:
                continue

            extend_repeating_tile(
                out,
                tile,
                int(r0),
                int(c0),
                (int(dr), int(dc)),
                bg=int(bg),
                protect_mask=plus_union_mask,
            )

    out[plus_union_mask] = int(bg)
    return out


# --------------- generator (attempt_1) ---------------

# === CFG: Tunable generation parameters ===
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


def _supported_directions(plus_mask: np.ndarray) -> list[tuple[int, int]]:
    """Return directions whose nozzle span can support a legal seed tile."""
    supported: list[tuple[int, int]] = []
    for direction in _DIRECTIONS:
        s, e = _nozzle_span(plus_mask, direction)
        if e - s + 1 <= _SEED_MAX_CROSS:
            supported.append(direction)
    return supported


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

    _dr, dc = direction
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
    """Compute how many full tile repetitions fit from the seed onward along the ray."""
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
    """Compute (r, c, value) cells for the full projected ray (seed + repetitions)."""
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

    return cells, int(r0), int(c0)


def _place_plus_and_ray(
    rng: np.random.Generator,
    plus_shape: np.ndarray,
    plus_color: int,
    marker_palette: list[int],
    H: int,
    W: int,
    occ: np.ndarray,
    plus_occ: np.ndarray,
    scene: np.ndarray,
    direction_candidates: list[tuple[int, int]] | None = None,
) -> dict | None:
    """Try to place one colored plus sprite and its full ray. Returns info dict or None."""
    ph, pw = plus_shape.shape

    for _ in range(_PLACEMENT_TRIES):
        r0 = int(rng.integers(1, H - ph))
        c0 = int(rng.integers(1, W - pw))
        plus_cells = [
            (r0 + dr, c0 + dc)
            for dr in range(ph)
            for dc in range(pw)
            if plus_shape[dr, dc]
        ]

        if any(occ[r, c] != 0 for r, c in plus_cells):
            continue
        if any(plus_occ[r, c] == 1 for r, c in plus_cells):
            continue

        directions = list(_DIRECTIONS if direction_candidates is None else direction_candidates)
        rng.shuffle(directions)
        for direction in directions:
            tile = _build_seed_tile(plus_shape, direction, marker_palette, rng)
            if tile is None:
                continue
            units = _compute_ray_units(direction, tile, r0, c0, ph, pw, H, W)
            if units < _MIN_RAY_UNITS:
                continue

            cells, sr0, sc0 = _project_tile_coords(
                H,
                W,
                (r0, c0, r0 + ph - 1, c0 + pw - 1),
                plus_shape,
                direction,
                tile,
            )
            if any(occ[r, c] != 0 for r, c, _ in cells if 0 <= r < H and 0 <= c < W):
                continue

            for r, c in plus_cells:
                occ[r, c] = 1
                scene[r, c] = int(plus_color)
                for dr2, dc2 in _DIRECTIONS:
                    rr, cc = r + dr2, c + dc2
                    if 0 <= rr < H and 0 <= cc < W:
                        plus_occ[rr, cc] = 1
            for r, c, v in cells:
                occ[r, c] = 2
                scene[r, c] = int(v)

            return {
                "plus_color": int(plus_color),
                "shape_signature": tuple(map(int, plus_shape.flatten())),
                "tile": tile,
                "cells": cells,
                "seed_r0": int(sr0),
                "seed_c0": int(sc0),
                "direction": (int(direction[0]), int(direction[1])),
                "units": int(units),
                "plus_cells": plus_cells,
            }

    return None


def _erase_pluses(scene: np.ndarray, pluses: list[dict], bg: int) -> np.ndarray:
    """Return a copy of the full scene with every source-plus footprint erased."""
    out = scene.copy()
    for plus in pluses:
        for r, c in plus["plus_cells"]:
            out[r, c] = int(bg)
    return out


def _build_input_from_scene(
    scene: np.ndarray,
    pluses: list[dict],
    bg: int,
    rng: np.random.Generator,
) -> np.ndarray:
    """Keep all pluses, but erase the repeated suffix of every non-demo ray."""
    candidates = [i for i, p in enumerate(pluses) if p["units"] >= _PRE_EXT_IDEAL_UNITS]
    if not candidates:
        candidates = list(range(len(pluses)))
    pre_idx = int(rng.choice(candidates))

    inp = scene.copy()
    for i, plus in enumerate(pluses):
        if i == pre_idx:
            continue
        th, tw = plus["tile"].shape
        sr0, sc0 = plus["seed_r0"], plus["seed_c0"]
        seed_set = {(sr0 + ii, sc0 + jj) for ii in range(th) for jj in range(tw)}
        for r, c, _ in plus["cells"]:
            if (r, c) not in seed_set:
                inp[r, c] = int(bg)
    return inp


def _choose_plus_shapes(rng: np.random.Generator, n_plus: int) -> list[np.ndarray]:
    """Choose 2-3 plus shapes while forcing at least two distinct silhouettes per grid."""
    template_ids = list(range(len(_PLUS_SHAPES)))
    if n_plus == 2:
        chosen_ids = list(rng.choice(template_ids, size=2, replace=False))
    else:
        n_unique = int(rng.integers(2, min(len(_PLUS_SHAPES), n_plus) + 1))
        base_ids = list(rng.choice(template_ids, size=n_unique, replace=False))
        extras = list(rng.choice(base_ids, size=n_plus - n_unique, replace=True))
        chosen_ids = base_ids + extras
        rng.shuffle(chosen_ids)
    return [_PLUS_SHAPES[int(idx)] for idx in chosen_ids]


def _candidate_directions_for_role(
    plus_shape: np.ndarray,
    role: str,
) -> list[tuple[int, int]]:
    """Map an orientation role to the legal directions for one shape."""
    supported = _supported_directions(plus_shape)
    if role == "vertical":
        return [d for d in supported if d[0] != 0]
    if role == "horizontal":
        return [d for d in supported if d[1] != 0]
    return supported


def _assign_direction_roles(
    plus_shapes: list[np.ndarray],
    rng: np.random.Generator,
) -> list[str]:
    """
    Force every generated scene to include both a horizontal and a vertical branch.
    Remaining sources may use any legal direction.
    """
    roles = ["any"] * len(plus_shapes)
    horizontal_candidates = [
        i for i, shape in enumerate(plus_shapes) if _candidate_directions_for_role(shape, "horizontal")
    ]
    vertical_candidates = [
        i for i, shape in enumerate(plus_shapes) if _candidate_directions_for_role(shape, "vertical")
    ]
    if not horizontal_candidates or not vertical_candidates:
        return roles

    h_idx = int(rng.choice(horizontal_candidates))
    roles[h_idx] = "horizontal"

    remaining_vertical = [i for i in vertical_candidates if i != h_idx]
    if remaining_vertical:
        v_idx = int(rng.choice(remaining_vertical))
    else:
        v_idx = h_idx
    roles[v_idx] = "vertical"
    return roles


def _gen_attempt(rng: np.random.Generator) -> np.ndarray | None:
    H = int(rng.integers(*_H_RANGE))
    W = int(rng.integers(*_W_RANGE))

    bg = int(rng.integers(0, 10))
    others = [c for c in range(10) if c != bg]

    n_plus = int(rng.integers(*_N_PLUS_RANGE))
    plus_shapes = _choose_plus_shapes(rng, n_plus)
    direction_roles = _assign_direction_roles(plus_shapes, rng)
    plus_colors = list(map(int, rng.choice(others, size=n_plus, replace=False)))
    marker_palette = [c for c in others if c not in plus_colors]
    if not marker_palette:
        return None

    scene = np.full((H, W), bg, dtype=int)
    occ = np.zeros((H, W), dtype=int)
    plus_occ = np.zeros((H, W), dtype=int)

    pluses: list[dict] = []
    for plus_shape, plus_color, role in zip(plus_shapes, plus_colors, direction_roles):
        direction_candidates = _candidate_directions_for_role(plus_shape, role)
        if not direction_candidates:
            return None
        info = _place_plus_and_ray(
            rng,
            plus_shape,
            int(plus_color),
            marker_palette,
            H,
            W,
            occ,
            plus_occ,
            scene,
            direction_candidates=direction_candidates,
        )
        if info is None:
            return None
        pluses.append(info)

    if len({p["plus_color"] for p in pluses}) != len(pluses):
        return None
    if len({p["shape_signature"] for p in pluses}) < 2:
        return None

    inp = _build_input_from_scene(scene, pluses, bg, rng)
    out = _erase_pluses(scene, pluses, bg)

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
        grid = _gen_attempt(rng)
        if grid is not None:
            return grid
    raise RuntimeError("generate_input: failed after %d attempts" % _GEN_RETRIES)
