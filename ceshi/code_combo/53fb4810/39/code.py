# Combo variant of 53fb4810 (index 39)
# combo_spec: D1/0+D7/0+D11/0
# Name: 对角多射线源消失

from __future__ import annotations

import sys
from pathlib import Path

import numpy as np

_ARC_UTILS_DIR = Path("/home/lixinyu/snap/new/solvers")
if str(_ARC_UTILS_DIR) not in sys.path:
    sys.path.append(str(_ARC_UTILS_DIR))

from arc_utils import bbox_coords, crop_bbox, get_bg, label_mask


# Design analysis for the combo variant
# =====================================
# a) Concrete behavior and trace
# - Base task: one marker per '+'; marker is attached on one cardinal arm; the marker's bbox
#   repeats straight outward; '+' remains visible.
# - Combo task:
#   1. Each '+' has 2-4 markers, one per chosen corner quadrant.
#   2. A corner marker touches two orthogonal arms at once, so top+left means NW, top+right
#      means NE, bottom+left means SW, bottom+right means SE.
#   3. The marker's whole bbox is translated by one pixel per step along that diagonal,
#      creating a staircase-like diagonal strip.
#   4. After all strips are completed, every '+' footprint is erased to background.
# - Small trace example with a 3x3 plus:
#   * Original: right seed [2,3] extends horizontally right; plus survives.
#   * Here: the same plus may have NW seed [4,5] in the upper-left corner notch and SE seed
#     [[6],[7]] in the lower-right notch. The NW seed is shifted by (-1,-1), the SE seed by
#     (+1,+1), and the plus disappears in the output. One long diagonal strip may already be
#     present in the input as the exemplar, while the other strips remain compact seeds.
#
# b) Complexity calibration
# - Marker count is constrained to 2-4, matching the four diagonal quadrants. This keeps the
#   new degree of freedom visible but bounded.
# - Seed shapes are limited to 1x2 or 2x1. That is deliberate: with one-pixel diagonal shifts,
#   these shapes remain 4-connected along the strip, which keeps component parsing stable.
# - All pluses in one grid still share one color and one shape family, so object discovery is
#   not made harder than necessary.
#
# c) Implicit assumptions from the original main() that break here
# - Original adjacency logic assumed one cardinal contact side per marker. That fails because a
#   diagonal marker intentionally contacts two sides. The solver now infers a diagonal from the
#   pair of orthogonal contact directions.
# - Original periodic extension assumed axis-aligned stepping by tile height/width. That fails
#   for diagonal rays. The new extension translates the entire marker bbox by (±1, ±1) each
#   step instead.
# - Original "whole component bbox as tile" remains useful only if the ray component stays
#   connected. Restricting seeds to 1x2 / 2x1 preserves that property for both compact seeds
#   and the single pre-extended exemplar strip.
# - Original output kept plus cells protected. This combo intentionally erases every plus after
#   ray construction, so source retention must be inverted explicitly.
#
# d) Placement legality
# - Pluses are kept disjoint and 4-separated from all foreign occupied cells.
# - Every ray cell must stay in bounds, may not overlap any prior ray or plus cell, and may not
#   4-touch a foreign plus or foreign ray component. This guarantees that each non-plus
#   component in the input belongs to exactly one marker.
# - The entire plus-with-multiple-rays bundle is tested in a trial occupancy map before commit,
#   so acceptance means all chosen diagonal markers are simultaneously legal.
#
# e) Visual observability
# - A human can see several corner-attached seeds around the same '+', plus one long diagonal
#   exemplar somewhere in the grid. That makes the diagonal repetition rule and the
#   multi-marker ownership rule visible from geometry alone.
# - No rule depends on color id ordering. Colors are copied literally from the seed tiles.
#
# f) Visual cleanliness
# - Generator forbids 4-neighbor contacts between foreign structures, so the diagonal strips do
#   not fuse into ambiguous blobs.
# - Only one strip is pre-extended in the input, keeping the clue readable instead of flooding
#   the grid with complete outputs.
#
# g) Inferability of the change
# - The two-cell seed tile visibly determines the repeated strip motif; no hidden recoloring is
#   involved.
# - Source disappearance is global and obvious: every output removes the shared plus color while
#   preserving only the diagonal strips.
#
# h) New solver assumptions after the combo change
# - Each valid marker component touches its owning plus on exactly one vertical side and one
#   horizontal side, which together encode one diagonal quadrant.
# - Different marker/ray components never 4-touch each other or a foreign plus.
# - Marker palettes exclude the plus color, so erasing all plus-colored source cells in the
#   output cannot delete ray content by accident.


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
_DIAGONALS = [(-1, -1), (-1, 1), (1, -1), (1, 1)]
_CARD4 = [(-1, 0), (1, 0), (0, -1), (0, 1)]
_TILE_SHAPES = [(1, 2), (2, 1)]


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


def extend_diagonal_repeating_tile(
    out: np.ndarray,
    tile: np.ndarray,
    r0: int,
    c0: int,
    direction: tuple[int, int],
    *,
    bg: int,
    protect: set[int] | None = None,
) -> None:
    """Translate a tile diagonally by one cell per step until the bbox no longer intersects."""
    protect = set() if protect is None else set(map(int, protect))
    H, W = out.shape
    h, w = tile.shape
    dr, dc = map(int, direction)
    dr = 1 if dr > 0 else -1
    dc = 1 if dc > 0 else -1
    bg = int(bg)

    def put(rr: int, cc: int, v: int) -> None:
        if not (0 <= rr < H and 0 <= cc < W):
            return
        if v == bg:
            return
        if int(out[rr, cc]) in protect:
            return
        out[rr, cc] = int(v)

    step = 1
    while True:
        sr0 = int(r0 + step * dr)
        sc0 = int(c0 + step * dc)
        sr1 = int(sr0 + h - 1)
        sc1 = int(sc0 + w - 1)
        if sr1 < 0 or sr0 >= H or sc1 < 0 or sc0 >= W:
            break
        for i in range(h):
            for j in range(w):
                put(sr0 + i, sc0 + j, int(tile[i, j]))
        step += 1


def _infer_diagonal_direction(
    pts: np.ndarray,
    plus_mask: np.ndarray,
    H: int,
    W: int,
) -> tuple[int, int] | None:
    """Infer one diagonal from the pair of orthogonal plus->marker contacts."""
    contacts: set[tuple[int, int]] = set()
    for pr, pc in pts:
        pr = int(pr)
        pc = int(pc)
        for dr, dc in _CARD4:
            rr = pr + dr
            cc = pc + dc
            if 0 <= rr < H and 0 <= cc < W and bool(plus_mask[rr, cc]):
                contacts.add((int(pr - rr), int(pc - cc)))

    row_dirs = {d[0] for d in contacts if d[0] != 0}
    col_dirs = {d[1] for d in contacts if d[1] != 0}
    if len(row_dirs) != 1 or len(col_dirs) != 1:
        return None
    if len(contacts) != 2:
        return None
    dr = int(next(iter(row_dirs)))
    dc = int(next(iter(col_dirs)))
    if (dr, 0) not in contacts or (0, dc) not in contacts:
        return None
    return int(dr), int(dc)


def main(grid: np.ndarray) -> np.ndarray:
    """
    Extend every corner marker from each '+' as a diagonal periodic strip, then erase the source.

    Input structure:
    - The background is the most frequent color.
    - One dominant non-background color forms several identical '+'-like sprites.
    - Each '+' has 2-4 small marker components in different corner notches, and exactly one
      marker somewhere in the grid may already appear as a long diagonal strip.

    Rule:
    - For each non-plus component touching a '+' on one vertical side and one horizontal side,
      infer the corresponding diagonal quadrant.
    - Treat that component's bounding box as the repeating tile and translate it by one cell per
      step along the inferred diagonal until the grid boundary.
    - After all strips are inferred, erase every original '+' pixel back to the background.

    Key insight: a corner marker encodes its direction by which two arms it touches; the visible
    strip is just repeated one-cell diagonal shifts of that seed component, and the '+' itself is
    only a temporary anchor that disappears in the output.
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
            for dr, dc in _CARD4:
                rr = r + dr
                cc = c + dc
                if 0 <= rr < H and 0 <= cc < W and int(mark_lab[rr, cc]) > 0:
                    adj_ids.add(int(mark_lab[rr, cc]))

        for cid in sorted(adj_ids):
            pts = np.argwhere(mark_lab == int(cid))
            if pts.size == 0:
                continue

            direction = _infer_diagonal_direction(pts, plus_mask, H, W)
            if direction is None:
                continue

            r0, c0, r1, c1 = bbox_coords(pts)
            tile = crop_bbox(g, (int(r0), int(c0), int(r1), int(c1)))
            extend_diagonal_repeating_tile(
                out,
                tile,
                int(r0),
                int(c0),
                direction,
                bg=int(bg),
                protect={int(plus_col)},
            )

    out[g == int(plus_col)] = int(bg)
    return out


# --------------- generator (combo attempt_1) ---------------

_H_RANGE = (18, 31)
_W_RANGE = (18, 27)
_N_PLUS_RANGE = (2, 4)
_MIN_MARKERS_PER_PLUS = 2
_MAX_MARKERS_PER_PLUS = 4
_SEED_MAX_COLORS = 2
_MIN_RAY_STEPS = 4
_PRE_EXT_IDEAL_STEPS = 9
_PLACEMENT_TRIES = 220
_SUBSET_TRIES = 20
_TILE_TRIES = 20
_GEN_RETRIES = 600


def _tile_signature(tile: np.ndarray) -> tuple[tuple[int, int], tuple[int, ...]]:
    return tuple(map(int, tile.shape)), tuple(map(int, tile.flatten()))


def _build_seed_tile(palette: list[int], rng: np.random.Generator) -> np.ndarray:
    """Build a 1x2 or 2x1 tile with 1-2 non-plus colors."""
    th, tw = _TILE_SHAPES[int(rng.integers(0, len(_TILE_SHAPES)))]
    n_cells = th * tw
    max_k = min(_SEED_MAX_COLORS, n_cells, len(palette))
    n_colors = int(rng.integers(1, max_k + 1))
    chosen = list(rng.choice(palette, size=n_colors, replace=False))
    vals = list(chosen) + list(rng.choice(chosen, size=n_cells - n_colors, replace=True))
    rng.shuffle(vals)
    tile = np.array(vals, dtype=int).reshape(th, tw)
    if len(chosen) == 1 and len(palette) >= 2 and rng.random() < 0.5:
        alt = int(rng.choice([c for c in palette if c != int(tile.flat[0])]))
        tile.flat[-1] = alt
    return tile


def _seed_origin(
    plus_bbox: tuple[int, int, int, int],
    direction: tuple[int, int],
    tile: np.ndarray,
) -> tuple[int, int]:
    """Anchor a tile in one corner notch of the plus bbox."""
    pr0, pc0, pr1, pc1 = plus_bbox
    th, tw = tile.shape
    dr, dc = direction

    if (dr, dc) == (-1, -1):
        return (int(pr0), int(pc0 - 1)) if (th, tw) == (1, 2) else (int(pr0 - 1), int(pc0))
    if (dr, dc) == (-1, 1):
        return (int(pr0), int(pc1)) if (th, tw) == (1, 2) else (int(pr0 - 1), int(pc1))
    if (dr, dc) == (1, -1):
        return (int(pr1), int(pc0 - 1)) if (th, tw) == (1, 2) else (int(pr1), int(pc0))
    return (int(pr1), int(pc1)) if (th, tw) == (1, 2) else (int(pr1), int(pc1))


def _project_diagonal_component(
    H: int,
    W: int,
    plus_bbox: tuple[int, int, int, int],
    direction: tuple[int, int],
    tile: np.ndarray,
) -> tuple[list[tuple[int, int, int]], int, int, int] | None:
    """Compute full diagonal strip cells, seed origin, and number of shifted copies."""
    th, tw = tile.shape
    r0, c0 = _seed_origin(plus_bbox, direction, tile)
    if r0 < 0 or c0 < 0 or r0 + th > H or c0 + tw > W:
        return None

    dr, dc = direction
    cells: list[tuple[int, int, int]] = []
    steps = 0
    k = 0
    while True:
        sr0 = int(r0 + k * dr)
        sc0 = int(c0 + k * dc)
        sr1 = int(sr0 + th - 1)
        sc1 = int(sc0 + tw - 1)
        if sr1 < 0 or sr0 >= H or sc1 < 0 or sc0 >= W:
            break
        for i in range(th):
            rr = sr0 + i
            if not (0 <= rr < H):
                continue
            for j in range(tw):
                cc = sc0 + j
                if 0 <= cc < W:
                    cells.append((int(rr), int(cc), int(tile[i, j])))
        steps += 1
        k += 1
    return cells, int(r0), int(c0), int(steps)


def _plus_cells_fit(occ: np.ndarray, plus_cells: list[tuple[int, int]]) -> bool:
    """Candidate plus cells must not overlap or 4-touch any existing structure."""
    H, W = occ.shape
    for r, c in plus_cells:
        if not (0 <= r < H and 0 <= c < W):
            return False
        if int(occ[r, c]) != 0:
            return False
        for dr, dc in _CARD4:
            rr = r + dr
            cc = c + dc
            if 0 <= rr < H and 0 <= cc < W and int(occ[rr, cc]) != 0:
                return False
    return True


def _ray_cells_fit(
    trial_occ: np.ndarray,
    cells: list[tuple[int, int, int]],
    own_plus_cells: set[tuple[int, int]],
) -> bool:
    """
    Ray cells must remain isolated from foreign structures.

    This preserves the solver assumption that each marker/ray component is one 4-connected
    non-plus object touching exactly one owning plus.
    """
    H, W = trial_occ.shape
    cell_pos = {(int(r), int(c)) for r, c, _ in cells}
    if len(cell_pos) != len(cells):
        return False

    for r, c, _ in cells:
        if not (0 <= r < H and 0 <= c < W):
            return False
        if int(trial_occ[r, c]) != 0:
            return False
        for dr, dc in _CARD4:
            rr = r + dr
            cc = c + dc
            if not (0 <= rr < H and 0 <= cc < W):
                continue
            if (rr, cc) in cell_pos:
                continue
            val = int(trial_occ[rr, cc])
            if val == 2:
                return False
            if val == 1 and (rr, cc) not in own_plus_cells:
                return False
    return True


def _sample_direction_subset(rng: np.random.Generator) -> list[tuple[int, int]]:
    k = int(rng.integers(_MIN_MARKERS_PER_PLUS, _MAX_MARKERS_PER_PLUS + 1))
    dirs = _DIAGONALS.copy()
    rng.shuffle(dirs)
    return dirs[:k]


def _place_plus_and_rays(
    rng: np.random.Generator,
    plus_shape: np.ndarray,
    palette: list[int],
    H: int,
    W: int,
    occ: np.ndarray,
    scene: np.ndarray,
    plus_color: int,
) -> dict | None:
    """Try to place one plus sprite with 2-4 diagonal marker rays."""
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
        if not _plus_cells_fit(occ, plus_cells):
            continue

        plus_bbox = (int(r0), int(c0), int(r0 + ph - 1), int(c0 + pw - 1))
        own_plus_cells = {(int(r), int(c)) for r, c in plus_cells}

        for _ in range(_SUBSET_TRIES):
            chosen_dirs = _sample_direction_subset(rng)
            trial_occ = occ.copy()
            for r, c in plus_cells:
                trial_occ[r, c] = 1

            used_tiles: set[tuple[tuple[int, int], tuple[int, ...]]] = set()
            rays: list[dict] = []
            success = True

            for direction in chosen_dirs:
                ray_info = None
                for _ in range(_TILE_TRIES):
                    tile = _build_seed_tile(palette, rng)
                    sig = _tile_signature(tile)
                    if sig in used_tiles:
                        continue

                    projected = _project_diagonal_component(H, W, plus_bbox, direction, tile)
                    if projected is None:
                        continue
                    cells, sr0, sc0, steps = projected
                    if int(steps) < _MIN_RAY_STEPS:
                        continue
                    if not _ray_cells_fit(trial_occ, cells, own_plus_cells):
                        continue

                    for rr, cc, _vv in cells:
                        trial_occ[rr, cc] = 2
                    ray_info = {
                        "direction": tuple(map(int, direction)),
                        "tile": tile,
                        "cells": cells,
                        "seed_r0": int(sr0),
                        "seed_c0": int(sc0),
                        "steps": int(steps),
                    }
                    used_tiles.add(sig)
                    break

                if ray_info is None:
                    success = False
                    break
                rays.append(ray_info)

            if not success:
                continue

            for r, c in plus_cells:
                occ[r, c] = 1
                scene[r, c] = int(plus_color)
            for ray in rays:
                for rr, cc, vv in ray["cells"]:
                    occ[rr, cc] = 2
                    scene[rr, cc] = int(vv)

            return {
                "plus_cells": plus_cells,
                "plus_bbox": plus_bbox,
                "marker_count": len(rays),
                "rays": rays,
            }

    return None


def _erase_pluses(scene: np.ndarray, pluses: list[dict], bg: int) -> np.ndarray:
    """Return a copy of the full scene with every plus footprint erased."""
    out = scene.copy()
    for plus in pluses:
        for r, c in plus["plus_cells"]:
            out[int(r), int(c)] = int(bg)
    return out


def _build_input_from_scene(
    scene: np.ndarray,
    pluses: list[dict],
    bg: int,
    rng: np.random.Generator,
) -> np.ndarray:
    """Keep one full exemplar strip and collapse all others back to their compact seeds."""
    all_rays: list[tuple[int, int, dict]] = []
    for pi, plus in enumerate(pluses):
        for ri, ray in enumerate(plus["rays"]):
            all_rays.append((pi, ri, ray))

    pre_idx = max(
        range(len(all_rays)),
        key=lambda i: (int(pluses[all_rays[i][0]]["marker_count"]), int(all_rays[i][2]["steps"])),
    )
    strong = [
        i
        for i, (pi, _ri, ray) in enumerate(all_rays)
        if int(pluses[pi]["marker_count"]) >= 3 and int(ray["steps"]) >= _PRE_EXT_IDEAL_STEPS
    ]
    if strong:
        pre_idx = int(rng.choice(strong))

    keep_pi, keep_ri, _keep_ray = all_rays[pre_idx]
    inp = scene.copy()
    for pi, plus in enumerate(pluses):
        for ri, ray in enumerate(plus["rays"]):
            if pi == keep_pi and ri == keep_ri:
                continue
            th, tw = ray["tile"].shape
            sr0 = int(ray["seed_r0"])
            sc0 = int(ray["seed_c0"])
            seed_set = {(sr0 + ii, sc0 + jj) for ii in range(th) for jj in range(tw)}
            for r, c, _ in ray["cells"]:
                if (int(r), int(c)) not in seed_set:
                    inp[int(r), int(c)] = int(bg)
    return inp


def _scene_has_required_coverage(pluses: list[dict]) -> bool:
    """
    Keep the combo visibly multi-ray:
    - at least one plus must expose 3+ markers, so D11/0 is obvious;
    - the scene should use at least 3 diagonal directions overall.
    """
    if not pluses:
        return False

    max_markers = max(int(plus["marker_count"]) for plus in pluses)
    if max_markers < 3:
        return False

    dirs = {
        tuple(map(int, ray["direction"]))
        for plus in pluses
        for ray in plus["rays"]
    }
    return len(dirs) >= 3


def _gen_attempt(rng: np.random.Generator) -> np.ndarray | None:
    H = int(rng.integers(*_H_RANGE))
    W = int(rng.integers(*_W_RANGE))

    bg = int(rng.integers(0, 10))
    others = [c for c in range(10) if c != bg]
    plus_color = int(rng.choice(others))
    palette = [c for c in others if c != plus_color]

    plus_shape = _PLUS_SHAPES[int(rng.integers(0, len(_PLUS_SHAPES)))]
    N = int(rng.integers(*_N_PLUS_RANGE))

    scene = np.full((H, W), bg, dtype=int)
    occ = np.zeros((H, W), dtype=int)
    pluses: list[dict] = []

    for _ in range(N):
        info = _place_plus_and_rays(rng, plus_shape, palette, H, W, occ, scene, plus_color)
        if info is None:
            return None
        pluses.append(info)

    if not _scene_has_required_coverage(pluses):
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
        g = _gen_attempt(rng)
        if g is not None:
            return g
    raise RuntimeError(f"generate_input: failed after {_GEN_RETRIES} attempts")
