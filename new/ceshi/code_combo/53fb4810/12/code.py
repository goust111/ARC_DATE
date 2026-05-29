# Combo variant of 53fb4810 (index 12)
# combo_spec: D7/0+D9/0
# Name: 源消失无线索
#
# Design analysis for the combo variant
# a) Concrete behavior and trace:
#    - Base task flow: each plus has one adjacent seed tile; one chosen plus keeps a long
#      already-expanded ray in the input; the output extends every seed while keeping all plus
#      pixels visible.
#    - D9/0 removes the in-grid demonstration step. After building the full scene, the input is
#      collapsed so every plus keeps only its compact seed and no ray tail at all.
#    - D7/0 flips the output retention bit. After the periodic rays are inferred and drawn, the
#      plus footprints are erased back to background, so the final output contains only ray
#      trajectories.
#    - Small trace example: with a 3x3 plus and a 2x1 seed attached below, the base input might
#      show one long vertical example ray elsewhere; the base output would keep the 3x3 source
#      and repeat the 2x1 tile downward. In this combo input, every plus only shows the first
#      2x1 seed copy. The combo output still repeats downward, but the 3x3 source area is blanked
#      to background. Relative to the base task, one cue is removed from the input and one anchor
#      is removed from the output.
# b) Complexity calibration:
#    - The task stays in the same object family: 2-3 identical plus sprites, one compact seed per
#      plus, and one cardinal ray direction per object. That preserves repeated support across the
#      small ARC training set.
#    - Seed sizes remain limited to the original nozzle-driven family (1x2, 2x1, 2x2). This keeps
#      periodic repetition visually legible even though the clue ray is gone.
#    - The fused effect is still easy to notice in examples because every pair shows both changes:
#      inputs have zero long rays, and outputs have zero source-plus pixels.
# c) Implicit-assumption audit for main():
#    - `detect_dominant_big_component_color(...)` still assumes the plus color forms the dominant
#      large components in the input. This remains valid because the pluses are still present in
#      every input and compact seeds are much smaller.
#    - `non_mask = (g != bg) & (g != plus_col)` assumes marker payload colors exclude the plus
#      color. The generator enforces that, so marker labeling stays clean.
#    - Adjacency-based direction inference assumes each seed touches exactly one side of exactly
#      one plus. The generator still constructs exactly one seed component per plus.
#    - The nozzle-vs-seed alignment asserts assume the seed cross-span exactly matches the touched
#      arm-tip span. The generator keeps that invariant to avoid ambiguous phase shifts.
#    - Starting from `out = g.copy()` would preserve pluses by default, which is wrong here. The
#      fused solver therefore clears all plus pixels after ray growth is complete.
#    - No solver step should depend on a pre-expanded clue. The generator now guarantees none are
#      present, and the inference remains purely local to each plus+seed pair.
# d) Placement legality:
#    - Generation remains output-first: place pluses, attach seeds, project full rays, reject any
#      overlap with existing pluses or rays, then derive compact input and erased-source output.
#    - Rays never collide because occupancy is checked against the full projected cell list before
#      commit. Pluses do not touch each other because plus footprints and their 4-neighborhood
#      exclusion mask are both enforced.
#    - Seed placement stays in-bounds because plus positions are sampled with a one-cell border and
#      the first seed tile is always adjacent to a nozzle cell inside that margin.
# e) Visual observability:
#    - A human sees that the output contains longer versions of the compact colored seeds, but the
#      plus anchors visible in the input are absent in the output. Across 2-4 examples, that is a
#      direct and consistent visual rule rather than a hidden numeric convention.
#    - Removing the demo ray does not hide the mechanism because each seed is still attached to one
#      explicit side of a plus, and training pairs reveal that the extension direction is "away
#      from the touched side".
# f) Visual cleanliness:
#    - Inputs stay sparse because every non-plus object is only a compact seed. Outputs stay sparse
#      because rays are generated collision-free and plus erasure removes, rather than adds, visual
#      layering.
# g) Change inferability:
#    - No new color remapping is introduced. Colors remain literal tile payload values, so the
#      learner only has to infer geometric repetition plus source erasure, not any hidden palette
#      rule.
# h) New solver assumptions guaranteed by the generator:
#    - All pluses in one grid share the same color and shape family.
#    - Each plus has exactly one adjacent connected seed component.
#    - Marker tiles never use the plus color, so erasing source-color cells in the output cannot
#      accidentally delete ray content.

from __future__ import annotations

import numpy as np

from arc_utils import bbox_coords, crop_bbox, get_bg, label_mask


def detect_dominant_big_component_color(
    grid: np.ndarray,
    *,
    bg: int | None = None,
    min_h: int = 3,
    min_w: int = 3,
    connectivity: int = 4,
) -> int | None:
    """
    Local helper (only used by this solver): choose a color that has the most "big" components.

    Big component = connected component whose bbox is at least (min_h x min_w).
    Tie-break: total area of big components (larger wins).
    """
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
    """Local helper (only used by this solver): extend a tile pattern from (r0,c0) until boundary."""
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


def _solve(grid: np.ndarray, *, erase_source: bool) -> np.ndarray:
    g = np.array(grid, dtype=int)
    out = g.copy()
    H, W = g.shape

    bg = int(get_bg(g))
    plus_col = detect_dominant_big_component_color(g, bg=bg, min_h=3, min_w=3, connectivity=4)
    if plus_col is None:
        return out

    plus_lab, pn = label_mask(g == int(plus_col), connectivity=4)
    non_mask = (g != int(bg)) & (g != int(plus_col))
    mark_lab, _mn = label_mask(non_mask, connectivity=4)

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
                    if 0 <= rr < H and 0 <= cc < W and bool(plus_mask[rr, cc]):
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
                assert seed_cols == nozzle_cols
            else:
                nozzle_col = int(pc1) if dc == 1 else int(pc0)
                nozzle_rows = sorted(
                    int(rr) for rr in range(int(pr0), int(pr1) + 1) if plus_mask[rr, nozzle_col]
                )
                seed_rows = list(range(int(r0), int(r1) + 1))
                assert seed_rows == nozzle_rows

            extend_repeating_tile(
                out,
                tile,
                int(r0),
                int(c0),
                (int(dr), int(dc)),
                bg=int(bg),
                protect={int(plus_col)},
            )

    if erase_source:
        out[g == int(plus_col)] = int(bg)
    return out


def main(grid: np.ndarray) -> np.ndarray:
    """
    Extend every compact marker into a periodic ray, then delete the source-plus anchors.

    Input structure:
    - The background is the most frequent color.
    - One dominant non-background color forms several identical plus-like source sprites.
    - Each plus has exactly one adjacent compact marker tile, and no ray is pre-expanded
      anywhere in the input.

    Rule:
    - Use the side where a seed touches its plus to determine the outward cardinal direction.
    - Repeat the seed's bounding-box tile away from the plus until the grid boundary.
    - After all rays are drawn, erase every original plus pixel back to the background.

    Key insight:
    - The pluses are still the directional anchors for inference, but they are only
      temporary scaffolding. The final output keeps the repeated seed trajectories and
      discards the source sprites themselves.
    """
    return _solve(grid, erase_source=True)


# --------------- generator ---------------

# === CFG: Tunable generation parameters ===
_H_RANGE = (13, 29)
_W_RANGE = (13, 23)
_N_PLUS_RANGE = (2, 4)
_SEED_ALONG_CHOICES = [1, 2]
_SEED_MAX_COLORS = 4
_SEED_MAX_CROSS = 2
_MIN_RAY_UNITS = 2
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

    return cells, int(r0), int(c0)


def _place_plus_and_ray(
    rng: np.random.Generator,
    plus_shape: np.ndarray,
    palette: list[int],
    H: int,
    W: int,
    occ: np.ndarray,
    plus_occ: np.ndarray,
    scene: np.ndarray,
    plus_color: int,
    direction_order: list[tuple[int, int]] | None = None,
) -> dict | None:
    """Try to place one plus sprite with its ray. Returns info dict or None."""
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

        dirs = list(_DIRECTIONS) if direction_order is None else list(direction_order)
        if direction_order is None:
            rng.shuffle(dirs)
        for direction in dirs:
            tile = _build_seed_tile(plus_shape, direction, palette, rng)
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
                scene[r, c] = plus_color
                for dr2, dc2 in _DIRECTIONS:
                    rr, cc = r + dr2, c + dc2
                    if 0 <= rr < H and 0 <= cc < W:
                        plus_occ[rr, cc] = 1
            for r, c, v in cells:
                occ[r, c] = 2
                scene[r, c] = v

            return {
                "tile": tile,
                "cells": cells,
                "seed_r0": int(sr0),
                "seed_c0": int(sc0),
                "units": int(units),
                "direction": (int(direction[0]), int(direction[1])),
                "plus_cells": plus_cells,
            }

    return None


def _direction_try_order(
    rng: np.random.Generator,
    direction_counts: dict[tuple[int, int], int],
) -> list[tuple[int, int]]:
    """Prefer underused directions so one sample usually exhibits multiple branches."""
    scored = [
        (int(direction_counts.get(direction, 0)), float(rng.random()), direction)
        for direction in _DIRECTIONS
    ]
    scored.sort(key=lambda item: (item[0], item[1]))
    return [direction for _, _, direction in scored]


def _has_direction_variety(pluses: list[dict]) -> bool:
    """
    Require modest within-grid direction diversity.

    With no pre-expanded clue, multiple attachment directions in one example make the
    outward-repeat rule easier to infer and keep solver branches exercised.
    """
    if len(pluses) < 2:
        return True
    dirs = {tuple(map(int, p["direction"])) for p in pluses}
    if len(dirs) < 2:
        return False
    if len(pluses) < 3:
        return True
    axes = {0 if dr != 0 else 1 for dr, _dc in dirs}
    return len(axes) >= 2


def _erase_pluses(scene: np.ndarray, pluses: list[dict], bg: int) -> np.ndarray:
    """Return a copy of scene with every plus footprint erased to the background."""
    out = scene.copy()
    for p in pluses:
        for r, c in p["plus_cells"]:
            out[int(r), int(c)] = int(bg)
    return out


def _build_input_from_scene(scene: np.ndarray, pluses: list[dict], bg: int) -> np.ndarray:
    """Erase every ray tail so the input keeps only compact seeds and visible source pluses."""
    inp = scene.copy()
    for p in pluses:
        th, tw = p["tile"].shape
        sr0, sc0 = int(p["seed_r0"]), int(p["seed_c0"])
        seed_set = {(sr0 + ii, sc0 + jj) for ii in range(th) for jj in range(tw)}
        for r, c, _ in p["cells"]:
            if (int(r), int(c)) not in seed_set:
                inp[int(r), int(c)] = int(bg)
    return inp


def _generate_case(rng: np.random.Generator) -> tuple[np.ndarray, np.ndarray, dict] | None:
    H = int(rng.integers(*_H_RANGE))
    W = int(rng.integers(*_W_RANGE))

    bg = int(rng.integers(0, 10))
    others = [c for c in range(10) if c != bg]
    plus_color = int(rng.choice(others))
    palette = [c for c in others if c != plus_color]

    plus_shape = _PLUS_SHAPES[int(rng.integers(0, len(_PLUS_SHAPES)))]
    n_plus = int(rng.integers(*_N_PLUS_RANGE))

    scene = np.full((H, W), bg, dtype=int)
    occ = np.zeros((H, W), dtype=int)
    plus_occ = np.zeros((H, W), dtype=int)

    pluses: list[dict] = []
    direction_counts = {direction: 0 for direction in _DIRECTIONS}
    for _ in range(n_plus):
        direction_order = _direction_try_order(rng, direction_counts)
        info = _place_plus_and_ray(
            rng,
            plus_shape,
            palette,
            H,
            W,
            occ,
            plus_occ,
            scene,
            plus_color,
            direction_order=direction_order,
        )
        if info is None:
            return None
        pluses.append(info)
        direction_counts[tuple(info["direction"])] += 1

    if not _has_direction_variety(pluses):
        return None

    inp = _build_input_from_scene(scene, pluses, bg)
    out = _erase_pluses(scene, pluses, bg)

    try:
        produced = main(inp)
    except Exception:
        return None
    if not np.array_equal(produced, out):
        return None

    compact_seed_only = True
    for p in pluses:
        th, tw = p["tile"].shape
        sr0, sc0 = int(p["seed_r0"]), int(p["seed_c0"])
        seed_set = {(sr0 + ii, sc0 + jj) for ii in range(th) for jj in range(tw)}
        for r, c, _ in p["cells"]:
            is_seed = (int(r), int(c)) in seed_set
            if is_seed:
                compact_seed_only &= inp[int(r), int(c)] == scene[int(r), int(c)]
            else:
                compact_seed_only &= inp[int(r), int(c)] == int(bg)
    if not compact_seed_only:
        return None
    if np.any(out == int(plus_color)):
        return None

    trace = {
        "bg": int(bg),
        "plus_color": int(plus_color),
        "n_plus": int(n_plus),
        "plus_pixel_count": int(sum(len(p["plus_cells"]) for p in pluses)),
        "compact_seed_only": bool(compact_seed_only),
        "directions": [tuple(map(int, p["direction"])) for p in pluses],
        "ray_units": [int(p["units"]) for p in pluses],
    }
    return inp, out, trace


def generate_input() -> np.ndarray:
    rng = np.random.default_rng()
    for _ in range(_GEN_RETRIES):
        sample = _generate_case(rng)
        if sample is not None:
            return sample[0]
    raise RuntimeError(f"generate_input: failed after {_GEN_RETRIES} attempts")
