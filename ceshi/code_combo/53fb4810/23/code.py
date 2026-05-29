# Combo variant of 53fb4810 (index 23)
# combo_spec: D1/0+D7/0+D9/0
# Name: 对角线源消失无线索

from __future__ import annotations

import numpy as np

from arc_utils import bbox_coords, crop_bbox, get_bg, label_mask


# Design analysis for the combo variant
# a) Concrete behavior and trace:
#    - Base task: a small seed tile hangs off one cardinal arm of a plus, one example ray may
#      already be pre-expanded, and the output extends every seed straight outward while keeping
#      the plus visible.
#    - Combo behavior: each seed is moved into one of the four corner pockets of the plus bbox,
#      so it touches both a horizontal arm and a vertical arm. Those two touches jointly encode
#      a diagonal direction (NW/NE/SW/SE). The output repeats the seed by unit shifts along that
#      diagonal, then erases the plus source entirely. Inputs keep only compact corner seeds:
#      there is no long in-grid demonstration ray at all.
#    - Small trace: if a 2x2 seed occupies the top-left corner pocket of a 3x3 plus, then the
#      seed's corner cell touches the plus from the south and east, so the inferred direction is
#      up-left. The output paints the same 2x2 tile at offsets (-1,-1), (-2,-2), ... until the
#      border, and finally resets the plus pixels to background.
# b) Complexity calibration:
#    - The diagonal change is visually obvious because all trajectories are 45-degree bands
#      instead of axis-aligned stripes.
#    - To avoid oversimplifying into single-pixel dots, seeds still vary over a small family of
#      rectangles: 1x2, 2x1, or 2x2, with 1-4 colors. This preserves periodic texture while
#      keeping the direction cue stable.
#    - Every grid contains 2-3 pluses sharing one shape and one color, so the diagonal rule and
#      plus-erasure rule repeat enough times to be learnable from 2-4 examples.
# c) Implicit-assumption audit of main():
#    - `detect_dominant_big_component_color(...)` still works because pluses remain the only large
#      repeated components in the input; removing the clue ray makes other components smaller.
#    - The old solver assumed one cardinal touch direction. That breaks here because corner seeds
#      touch two arms. The replacement logic now combines the row-sign and col-sign of all plus ->
#      marker adjacencies; a valid combo input must yield exactly one nonzero row sign and one
#      nonzero col sign.
#    - The old nozzle-span asserts no longer apply. They are replaced by a corner-anchor check:
#      for a given diagonal direction and seed bbox shape, the seed must occupy the matching plus
#      corner pocket.
#    - Because D7 erases the source, `out[g == plus_col] = bg` is now a required final step.
# d) Placement legality:
#    - Generation remains output-first. It places a full scene with pluses and diagonal rays,
#      checks every painted cell for overlap, then derives the input by erasing every ray tail.
#    - A plus halo (expanded bbox) blocks later pluses and later rays from entering the local
#      neighborhood of an existing plus, ensuring each compact seed can only belong to one source.
#    - Seeds are always fully in bounds; only the last repeated copies may be clipped by the edge.
# e) Visual observability:
#    - Humans can see three coupled signals directly in the train pairs: seeds sit in corners
#      rather than on flat arms, outputs contain only diagonal traces, and pluses vanish.
#    - The absence of any pre-expanded clue is itself observable because every input contains only
#      compact corner seeds next to intact pluses.
# f) Visual cleanliness:
#    - Rays are generated disjointly in the full scene, and inputs erase all tails, so the visible
#      input remains sparse: isolated pluses with one nearby corner seed each.
#    - Output removes the bulky plus sources, leaving only the diagonal bands; that makes the new
#      path geometry easier to read rather than noisier.
# g) Color reasoning:
#    - No hidden color ordering is introduced. Colors remain payload values copied from the seed
#      tile, and the plus color is excluded from seed palettes so erasing the plus cannot delete
#      ray content by accident.
# h) Fusion-specific non-degeneracy target:
#    - The generator must enforce all three changes simultaneously: every seed must infer a true
#      diagonal direction, every output must erase plus cells, and every input must erase all ray
#      tails. If any of these control variables drift back to the base task, the combo degenerates.
# i) New solver assumptions and generator guarantees:
#    - Each plus has exactly one adjacent marker component.
#    - Each marker component occupies one of the four corner-pocket anchor positions.
#    - The marker's 4-neighbor contacts with the plus expose both a row sign and a col sign.
#    - `generate_input()` constructs only scenes satisfying those assumptions and rejects any
#      attempt that would create overlap, ambiguous attachment, or too-short diagonal tails.


_CARD4 = ((1, 0), (-1, 0), (0, 1), (0, -1))
_DIAGONALS = [(-1, -1), (-1, 1), (1, -1), (1, 1)]


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


def _corner_seed_anchor(
    plus_bbox: tuple[int, int, int, int],
    direction: tuple[int, int],
    tile_shape: tuple[int, int],
) -> tuple[int, int]:
    """Anchor a corner seed so one tile corner sits in the matching empty plus-bbox corner."""
    pr0, pc0, pr1, pc1 = map(int, plus_bbox)
    th, tw = map(int, tile_shape)
    dr, dc = map(int, direction)

    if (dr, dc) == (-1, -1):
        return int(pr0 - th + 1), int(pc0 - tw + 1)
    if (dr, dc) == (-1, 1):
        return int(pr0 - th + 1), int(pc1)
    if (dr, dc) == (1, -1):
        return int(pr1), int(pc0 - tw + 1)
    if (dr, dc) == (1, 1):
        return int(pr1), int(pc1)
    raise ValueError("_corner_seed_anchor expects a diagonal direction")


def _infer_diagonal_direction(
    plus_mask: np.ndarray,
    pts: np.ndarray,
) -> tuple[int, int] | None:
    """Infer NW/NE/SW/SE from the signs of all 4-neighbor plus->marker contacts."""
    H, W = plus_mask.shape
    row_signs: set[int] = set()
    col_signs: set[int] = set()

    for pr, pc in pts:
        pr = int(pr)
        pc = int(pc)
        for dr, dc in _CARD4:
            rr = pr + dr
            cc = pc + dc
            if 0 <= rr < H and 0 <= cc < W and bool(plus_mask[rr, cc]):
                delta_r = int(pr - rr)
                delta_c = int(pc - cc)
                if delta_r != 0:
                    row_signs.add(1 if delta_r > 0 else -1)
                if delta_c != 0:
                    col_signs.add(1 if delta_c > 0 else -1)

    if len(row_signs) != 1 or len(col_signs) != 1:
        return None
    return int(next(iter(row_signs))), int(next(iter(col_signs)))


def _diagonal_max_shift(
    H: int,
    W: int,
    r0: int,
    c0: int,
    tile_shape: tuple[int, int],
    direction: tuple[int, int],
) -> int:
    """Maximum shift k>=0 such that the shifted tile bbox still intersects the grid."""
    th, tw = map(int, tile_shape)
    dr, dc = map(int, direction)
    if dr == 1:
        row_limit = int(H - 1 - r0)
    else:
        row_limit = int(r0 + th - 1)
    if dc == 1:
        col_limit = int(W - 1 - c0)
    else:
        col_limit = int(c0 + tw - 1)
    return int(min(row_limit, col_limit))


def extend_repeating_tile_diagonal(
    out: np.ndarray,
    tile: np.ndarray,
    r0: int,
    c0: int,
    direction: tuple[int, int],
    *,
    bg: int,
) -> None:
    """Repeat a tile by unit diagonal shifts until the shifted bbox leaves the grid."""
    H, W = out.shape
    th, tw = tile.shape
    dr, dc = map(int, direction)
    bg = int(bg)

    max_shift = _diagonal_max_shift(H, W, int(r0), int(c0), (th, tw), (dr, dc))
    for k in range(1, int(max_shift) + 1):
        base_r = int(r0 + k * dr)
        base_c = int(c0 + k * dc)
        for i in range(th):
            rr = int(base_r + i)
            if not (0 <= rr < H):
                continue
            for j in range(tw):
                cc = int(base_c + j)
                if not (0 <= cc < W):
                    continue
                v = int(tile[i, j])
                if v != bg:
                    out[rr, cc] = int(v)


def main(grid: np.ndarray) -> np.ndarray:
    """
    Extend corner-attached marker tiles diagonally from '+' sources, then erase the sources.

    Input structure:
    - The background is the most common color.
    - One dominant non-background color forms several identical plus-like source sprites.
    - Each plus has exactly one compact marker component tucked into one corner pocket of the
      plus bbox, and there is no pre-expanded diagonal clue anywhere in the input.

    Rule:
    - For each marker touching a plus, combine its vertical and horizontal touch signs to infer
      which diagonal corner (NW/NE/SW/SE) it belongs to.
    - Repaint the marker's bounding-box tile by shifting it one cell at a time along that
      diagonal until the grid boundary.
    - After all diagonal rays are drawn, erase every original plus pixel back to background.

    Key insight: A corner seed simultaneously touches one vertical arm and one horizontal arm,
    so those two local contacts reveal the diagonal trajectory even though no example ray is
    pre-drawn in the input and the source itself disappears in the output.
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

    for pk in range(1, int(pn) + 1):
        plus_mask = plus_lab == int(pk)
        plus_coords = np.argwhere(plus_mask)
        if plus_coords.size == 0:
            continue
        plus_bbox = bbox_coords(plus_coords)

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
            direction = _infer_diagonal_direction(plus_mask, pts)
            if direction is None:
                continue

            r0, c0, r1, c1 = bbox_coords(pts)
            tile = crop_bbox(g, (int(r0), int(c0), int(r1), int(c1)))
            exp_r0, exp_c0 = _corner_seed_anchor(plus_bbox, direction, tile.shape)
            if (int(r0), int(c0)) != (int(exp_r0), int(exp_c0)):
                continue

            extend_repeating_tile_diagonal(
                out,
                tile,
                int(r0),
                int(c0),
                direction,
                bg=int(bg),
            )

    out[g == int(plus_col)] = int(bg)
    return out


# --------------- generator (combo attempt_1) ---------------

# === CFG: Tunable generation parameters ===
_H_RANGE = (13, 29)
_W_RANGE = (13, 23)
_N_PLUS_RANGE = (2, 4)
_SEED_SHAPES = [(1, 2), (2, 1), (2, 2)]
_SEED_MAX_COLORS = 4
_MIN_TAIL_SHIFTS = 4
_PLACEMENT_TRIES = 160
_GEN_RETRIES = 400

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


def _build_seed_tile(
    tile_shape: tuple[int, int],
    palette: list[int],
    rng: np.random.Generator,
) -> np.ndarray:
    """Build a dense rectangular seed tile with 1-4 payload colors."""
    th, tw = map(int, tile_shape)
    n_cells = int(th * tw)
    max_k = min(_SEED_MAX_COLORS, n_cells, len(palette))
    n_colors = int(rng.integers(1, max_k + 1))
    chosen = list(rng.choice(palette, size=n_colors, replace=False))
    vals = list(chosen) + list(rng.choice(chosen, size=n_cells - n_colors, replace=True))
    rng.shuffle(vals)
    return np.array(vals, dtype=int).reshape(th, tw)


def _stabilize_diagonal_overlap(tile: np.ndarray, direction: tuple[int, int]) -> np.ndarray:
    """
    Keep the self-overlap cell consistent for 2x2 seeds.

    A 2x2 tile shifted by one cell along a diagonal overlaps itself on exactly one corner.
    Matching that corner pair removes avoidable color flicker in the repeated ray while
    preserving the diagonal-repeat rule.
    """
    if tile.shape != (2, 2):
        return tile

    dr, dc = map(int, direction)
    out = tile.copy()
    if (dr, dc) in {(-1, -1), (1, 1)}:
        out[1, 1] = out[0, 0]
    elif (dr, dc) in {(-1, 1), (1, -1)}:
        out[1, 0] = out[0, 1]
    return out


def _project_diagonal_tile_coords(
    H: int,
    W: int,
    tile: np.ndarray,
    r0: int,
    c0: int,
    direction: tuple[int, int],
) -> tuple[list[tuple[int, int, int]], set[tuple[int, int]], int]:
    """Return final painted cells, seed cells, and max shift count for one diagonal ray."""
    th, tw = tile.shape
    dr, dc = map(int, direction)
    max_shift = _diagonal_max_shift(H, W, int(r0), int(c0), (th, tw), (dr, dc))

    painted: dict[tuple[int, int], int] = {}
    for k in range(int(max_shift) + 1):
        base_r = int(r0 + k * dr)
        base_c = int(c0 + k * dc)
        for i in range(th):
            rr = int(base_r + i)
            if not (0 <= rr < H):
                continue
            for j in range(tw):
                cc = int(base_c + j)
                if 0 <= cc < W:
                    painted[(rr, cc)] = int(tile[i, j])

    seed_cells = {
        (int(r0 + i), int(c0 + j))
        for i in range(th)
        for j in range(tw)
        if 0 <= int(r0 + i) < H and 0 <= int(c0 + j) < W
    }
    cells = [(int(r), int(c), int(v)) for (r, c), v in sorted(painted.items())]
    return cells, seed_cells, int(max_shift)


def _mark_plus_buffer(buffer: np.ndarray, plus_bbox: tuple[int, int, int, int]) -> None:
    """Block a 1-cell halo around a placed plus so later objects cannot become ambiguous."""
    H, W = buffer.shape
    r0, c0, r1, c1 = map(int, plus_bbox)
    rr0 = max(0, int(r0 - 1))
    cc0 = max(0, int(c0 - 1))
    rr1 = min(H, int(r1 + 2))
    cc1 = min(W, int(c1 + 2))
    buffer[rr0:rr1, cc0:cc1] = 1


def _place_plus_and_ray(
    rng: np.random.Generator,
    plus_shape: np.ndarray,
    palette: list[int],
    H: int,
    W: int,
    occ: np.ndarray,
    plus_buffer: np.ndarray,
    scene: np.ndarray,
    plus_color: int,
) -> dict | None:
    """Try to place one plus sprite with a compact corner seed and its full diagonal ray."""
    ph, pw = plus_shape.shape

    for _ in range(_PLACEMENT_TRIES):
        r0 = int(rng.integers(1, H - ph))
        c0 = int(rng.integers(1, W - pw))
        plus_bbox = (int(r0), int(c0), int(r0 + ph - 1), int(c0 + pw - 1))
        plus_cells = [
            (int(r0 + dr), int(c0 + dc))
            for dr in range(ph)
            for dc in range(pw)
            if int(plus_shape[dr, dc]) != 0
        ]

        if any(int(occ[r, c]) != 0 for r, c in plus_cells):
            continue
        if any(int(plus_buffer[r, c]) != 0 for r, c in plus_cells):
            continue

        dirs = list(_DIAGONALS)
        shapes = list(_SEED_SHAPES)
        rng.shuffle(dirs)
        rng.shuffle(shapes)

        for direction in dirs:
            for tile_shape in shapes:
                tile = _stabilize_diagonal_overlap(
                    _build_seed_tile(tile_shape, palette, rng),
                    direction,
                )
                sr0, sc0 = _corner_seed_anchor(plus_bbox, direction, tile.shape)
                th, tw = tile.shape
                if not (0 <= sr0 and 0 <= sc0 and sr0 + th <= H and sc0 + tw <= W):
                    continue

                cells, seed_cells, max_shift = _project_diagonal_tile_coords(
                    H, W, tile, int(sr0), int(sc0), direction
                )
                if int(max_shift) < _MIN_TAIL_SHIFTS:
                    continue
                if any(int(occ[r, c]) != 0 for r, c, _ in cells):
                    continue
                if any(int(plus_buffer[r, c]) != 0 for r, c, _ in cells):
                    continue

                for r, c in plus_cells:
                    occ[r, c] = 1
                    scene[r, c] = int(plus_color)
                _mark_plus_buffer(plus_buffer, plus_bbox)
                for r, c, v in cells:
                    occ[r, c] = 2
                    scene[r, c] = int(v)

                return {
                    "plus_cells": plus_cells,
                    "cells": cells,
                    "seed_cells": seed_cells,
                    "max_shift": int(max_shift),
                    "direction": tuple(map(int, direction)),
                }

    return None


def _erase_pluses(scene: np.ndarray, pluses: list[dict], bg: int) -> np.ndarray:
    """Return the output scene: full rays remain, every plus pixel becomes background."""
    out = scene.copy()
    for p in pluses:
        for r, c in p["plus_cells"]:
            out[int(r), int(c)] = int(bg)
    return out


def _build_input_from_scene(scene: np.ndarray, pluses: list[dict], bg: int) -> np.ndarray:
    """Erase every diagonal tail so only compact corner seeds remain in the input."""
    inp = scene.copy()
    for p in pluses:
        seed_cells = {(int(r), int(c)) for r, c in p["seed_cells"]}
        for r, c, _ in p["cells"]:
            if (int(r), int(c)) not in seed_cells:
                inp[int(r), int(c)] = int(bg)
    return inp


def _gen_attempt(rng: np.random.Generator) -> np.ndarray | None:
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
    plus_buffer = np.zeros((H, W), dtype=int)

    pluses: list[dict] = []
    for _ in range(n_plus):
        info = _place_plus_and_ray(
            rng,
            plus_shape,
            palette,
            H,
            W,
            occ,
            plus_buffer,
            scene,
            plus_color,
        )
        if info is None:
            return None
        pluses.append(info)

    inp = _build_input_from_scene(scene, pluses, bg)
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
    raise RuntimeError("generate_input: failed after %d attempts" % _GEN_RETRIES)
