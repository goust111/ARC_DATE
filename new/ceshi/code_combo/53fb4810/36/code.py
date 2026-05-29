# Combo variant of 53fb4810 (index 36)
# combo_spec: D2/1+D3/0
# Name: 颜色编码方向间歇跳跃

from __future__ import annotations

# Design analysis for combo D2/1 + D3/0
# =======================================
# a) Concrete behavior and trace
#    - Original task: a small marker tile touches one side of a plus, and that
#      touching side itself decides the ray direction. The tile is then copied
#      continuously with no gaps.
#    - Fused variant: the touching side no longer decides the ray. Instead, the
#      seed tile is always a 2-cell domino whose internal color order encodes
#      the travel direction:
#          [low, high]  -> right
#          [high, low]  -> left
#          [[low],[high]] -> down
#          [[high],[low]] -> up
#      The ray is then painted intermittently: visible copy, fixed number of
#      whole-tile blank slots, visible copy, and so on.
#    - Small trace:
#      Suppose the seed is horizontal `[2,7]`, attached above a plus, and the
#      global skip gap is 1. The original solver would project upward because it
#      only sees "seed above plus". The fused solver instead reads `[2,7]` as
#      left-to-right ascending, so the direction is right, and the final row is:
#      `[2,7][bg,bg][2,7][bg,bg]...`
#      This makes the two changed dimensions interact: the seed's color order
#      picks the direction, while the demonstration ray teaches the shared jump
#      length along that direction.
#
# b) Complexity calibration
#    - Direction coding is restricted to four easily repeated motifs built from
#      one global low/high color pair. This avoids an arbitrary per-seed mapping
#      that a human could not infer from 2-4 examples.
#    - Skip gap is restricted to {1, 2}. One long demonstration ray shows at
#      least three visible copies, so the jump pattern is recoverable rather
#      than guessed.
#    - Seeds stay tiny (1x2 or 2x1), which keeps the visual cue crisp: shape
#      gives the axis, color order gives the sign.
#
# c) Hidden assumptions from the original solver that break here
#    - The original code assumes adjacency side == ray direction. That fails on
#      purpose here: every generated seed is attached on a side perpendicular to
#      its true travel direction.
#    - The original nozzle-span assert no longer applies. A seed may touch only
#      part of a side ring around the plus, because attachment is now only a way
#      to associate the seed with a plus, not to define the ray geometry.
#    - The original extension painter assumes every tile slot is occupied. The
#      new painter must reason in whole-tile slots and leave repeated blank
#      slots between visible copies.
#
# d) Placement legality guarantees in the generator
#    - All pluses still share one symmetric shape and remain disjoint.
#    - Each seed is 4-neighbor adjacent to exactly one plus and is placed on a
#      side perpendicular to its encoded direction, ensuring the old rule would
#      be wrong.
#    - The generator reserves the entire ray lane, including skipped slots, so
#      no other plus or ray can intrude into blank intervals and corrupt
#      skip-gap inference.
#
# e) Visual observability
#    - One long dashed ray shows both ingredients: repeated whole-tile blanks
#      expose the intermittent spacing, while the same low/high domino order
#      appearing across objects teaches the directional code.
#    - Because the direction colors are global within a grid, the learner sees
#      repeated evidence of the same ordered pair instead of isolated arbitrary
#      color tuples.
#
# f) Visual cleanliness
#    - Lane reservation prevents overlaps between pluses, seeds, painted copies,
#      and blank skip slots.
#    - Using only domino seeds avoids noisy 2x2 textures. The marker reads as a
#      compact directional token rather than a decorative patch.
#
# g) Change inferability
#    - The color change is structural, not a raw color remap. The same two
#      colors appear in ordered pairs whose orientation is stable across the
#      sample. A human can infer "this ordering points this way" from examples.
#    - The solver uses ascending/descending color ids internally, but the
#      generator makes that relation visually learnable by reusing one global
#      low/high pair throughout each puzzle.
#
# h) Post-change solver assumptions and generator support
#    - main() assumes adjacent seed components have shape 1x2 or 2x1. The
#      generator only emits those two seed shapes.
#    - main() assumes the one demonstrated ray follows the same domino tile in
#      every occupied slot and that skipped slots are fully background. The
#      generator reserves full slots and paints exact copies only on allowed
#      steps.
#    - main() assumes one global skip gap is shared by all rays in a grid. The
#      generator samples one skip_gap per grid and uses it everywhere.

import sys
from pathlib import Path

import numpy as np

_ROOT = Path(__file__).resolve().parents[5]
_SOLVER_DIR = _ROOT / "solvers"
if str(_SOLVER_DIR) not in sys.path:
    sys.path.append(str(_SOLVER_DIR))

from arc_utils import bbox_coords, crop_bbox, get_bg, label_mask


def detect_dominant_big_component_color(
    grid: np.ndarray,
    *,
    bg: int | None = None,
    min_h: int = 3,
    min_w: int = 3,
    connectivity: int = 4,
) -> int | None:
    """Choose the non-background color with the strongest large-component evidence."""
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


_DIRECTIONS = [(1, 0), (-1, 0), (0, 1), (0, -1)]


def _slot_origin(
    r0: int,
    c0: int,
    tile: np.ndarray,
    direction: tuple[int, int],
    slot_idx: int,
) -> tuple[int, int]:
    """Top-left corner of the slot_idx-th tile slot measured from the seed slot."""
    h, w = tile.shape
    dr, dc = map(int, direction)
    if dc == 0:
        return int(r0 + slot_idx * dr * h), int(c0)
    return int(r0), int(c0 + slot_idx * dc * w)


def _slot_intersects_grid(H: int, W: int, r0: int, c0: int, h: int, w: int) -> bool:
    """Whether the tile bbox intersects the grid at all."""
    return not (int(r0) >= int(H) or int(c0) >= int(W) or int(r0 + h) <= 0 or int(c0 + w) <= 0)


def _slot_fully_in_bounds(H: int, W: int, r0: int, c0: int, h: int, w: int) -> bool:
    """Whether the whole tile bbox lies inside the grid."""
    return 0 <= int(r0) and 0 <= int(c0) and int(r0 + h) <= int(H) and int(c0 + w) <= int(W)


def _paint_tile(
    out: np.ndarray,
    tile: np.ndarray,
    r0: int,
    c0: int,
    *,
    bg: int,
    protect: set[int] | None = None,
) -> None:
    """Paint one tile copy, clipping at the grid boundary."""
    protect = set() if protect is None else set(map(int, protect))
    H, W = out.shape
    h, w = tile.shape
    bg = int(bg)

    for i in range(h):
        rr = int(r0 + i)
        if not (0 <= rr < H):
            continue
        for j in range(w):
            cc = int(c0 + j)
            if not (0 <= cc < W):
                continue
            v = int(tile[i, j])
            if v == bg:
                continue
            if int(out[rr, cc]) in protect:
                continue
            out[rr, cc] = int(v)


def decode_direction_from_tile(tile: np.ndarray) -> tuple[int, int] | None:
    """Decode ray direction from a 1x2 / 2x1 domino using ascending-vs-descending order."""
    h, w = tile.shape
    if (h, w) == (1, 2):
        return (0, 1) if int(tile[0, 0]) < int(tile[0, 1]) else (0, -1)
    if (h, w) == (2, 1):
        return (1, 0) if int(tile[0, 0]) < int(tile[1, 0]) else (-1, 0)
    return None


def extend_intermittent_tile(
    out: np.ndarray,
    tile: np.ndarray,
    r0: int,
    c0: int,
    direction: tuple[int, int],
    *,
    skip_gap: int,
    bg: int,
    protect: set[int] | None = None,
) -> None:
    """Extend a tile outward while leaving whole-tile blank slots between copies."""
    H, W = out.shape
    h, w = tile.shape
    step_slots = int(skip_gap) + 1
    slot_idx = int(step_slots)

    while True:
        sr, sc = _slot_origin(int(r0), int(c0), tile, direction, int(slot_idx))
        if not _slot_intersects_grid(H, W, sr, sc, h, w):
            break
        _paint_tile(out, tile, sr, sc, bg=int(bg), protect=protect)
        slot_idx += int(step_slots)


def _count_full_slots(
    H: int,
    W: int,
    tile: np.ndarray,
    r0: int,
    c0: int,
    direction: tuple[int, int],
) -> int:
    """Count full tile slots from the seed onward before the lane exits the grid."""
    h, w = tile.shape
    count = 0
    slot_idx = 0
    while True:
        sr, sc = _slot_origin(int(r0), int(c0), tile, direction, int(slot_idx))
        if not _slot_fully_in_bounds(H, W, sr, sc, h, w):
            break
        count += 1
        slot_idx += 1
    return int(count)


def infer_skip_gap_from_demo(
    grid: np.ndarray,
    tile: np.ndarray,
    r0: int,
    c0: int,
    direction: tuple[int, int],
    *,
    bg: int,
) -> tuple[int | None, int]:
    """
    Inspect full tile slots on one candidate ray.

    Returns:
    - inferred skip_gap if this marker is the demonstrated dashed ray, else None
    - number of visible tile copies seen on that candidate ray
    """
    H, W = grid.shape
    h, w = tile.shape
    max_full_slots = _count_full_slots(H, W, tile, int(r0), int(c0), direction)

    occupied_slots = [0]
    for slot_idx in range(1, int(max_full_slots)):
        sr, sc = _slot_origin(int(r0), int(c0), tile, direction, int(slot_idx))
        patch = grid[int(sr) : int(sr + h), int(sc) : int(sc + w)]
        if np.array_equal(patch, tile):
            occupied_slots.append(int(slot_idx))
            continue
        if np.all(patch == int(bg)):
            continue
        return None, int(len(occupied_slots))

    if len(occupied_slots) < 3:
        return None, int(len(occupied_slots))

    diffs = [b - a for a, b in zip(occupied_slots, occupied_slots[1:])]
    if len(set(map(int, diffs))) != 1:
        return None, int(len(occupied_slots))

    step_slots = int(diffs[0])
    if step_slots < 2:
        return None, int(len(occupied_slots))

    return int(step_slots - 1), int(len(occupied_slots))


def main(grid: np.ndarray) -> np.ndarray:
    """
    Extend attached domino markers by following their color-coded direction and a shared dashed rhythm.

    Input structure:
    - The background is the most common color, and one dominant non-background color forms several '+' sprites.
    - Each '+' has exactly one adjacent 2-cell domino seed, but that seed is side-hung: its touching side is only an attachment cue and is perpendicular to the true ray direction.
    - Exactly one seed is already shown as a longer dashed ray whose occupied slots repeat the same domino after a fixed number of blank tile slots.

    Rule:
    - Detect each seed component that touches a '+' and decode its direction from the domino's color order: ascending means right/down along the domino axis, descending means left/up.
    - Read the long demonstrated ray to infer the shared skip gap in whole-tile slots.
    - Repaint every seed outward along its decoded direction, placing exact tile copies at that fixed step and leaving intermediate slots as background.
    - '+' pixels are protected and never overwritten.

    Key insight: The side of attachment is a red herring in this variant. The only reliable direction cue is the seed's internal low-to-high color order, while the single long example ray reveals how many whole tile slots are skipped between visible copies.
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

    markers: list[dict] = []
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
            r0, c0, r1, c1 = bbox_coords(pts)
            tile = crop_bbox(g, (int(r0), int(c0), int(r1), int(c1)))
            direction = decode_direction_from_tile(tile)
            if direction is None:
                continue
            markers.append(
                {
                    "tile": tile,
                    "r0": int(r0),
                    "c0": int(c0),
                    "direction": tuple(map(int, direction)),
                }
            )

    skip_gap = 1
    best_visible = -1
    for m in markers:
        cand_gap, visible = infer_skip_gap_from_demo(
            g,
            m["tile"],
            int(m["r0"]),
            int(m["c0"]),
            m["direction"],
            bg=int(bg),
        )
        if cand_gap is None:
            continue
        if int(visible) > int(best_visible):
            skip_gap = int(cand_gap)
            best_visible = int(visible)

    for m in markers:
        extend_intermittent_tile(
            out,
            m["tile"],
            int(m["r0"]),
            int(m["c0"]),
            m["direction"],
            skip_gap=int(skip_gap),
            bg=int(bg),
            protect={int(plus_col)},
        )

    return out


# --------------- generator (combo D2/1 + D3/0) ---------------

_H_RANGE = (13, 29)
_W_RANGE = (13, 23)
_N_PLUS_RANGE = (2, 4)
_SKIP_GAP_CHOICES = [1, 2]
_MIN_VISIBLE_COPIES = 2
_PRE_EXT_MIN_VISIBLE = 3
_PLACEMENT_TRIES = 160
_GEN_RETRIES = 600

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


def _tile_for_direction(low: int, high: int, direction: tuple[int, int]) -> np.ndarray:
    """Build the canonical 2-cell seed tile for a given direction."""
    dr, dc = map(int, direction)
    if (dr, dc) == (0, 1):
        return np.array([[int(low), int(high)]], dtype=int)
    if (dr, dc) == (0, -1):
        return np.array([[int(high), int(low)]], dtype=int)
    if (dr, dc) == (1, 0):
        return np.array([[int(low)], [int(high)]], dtype=int)
    if (dr, dc) == (-1, 0):
        return np.array([[int(high)], [int(low)]], dtype=int)
    raise ValueError("unsupported direction")


def _touch_vectors(
    plus_cells: set[tuple[int, int]],
    tile_cells: list[tuple[int, int]],
) -> set[tuple[int, int]]:
    """Vectors from plus cell to adjacent tile cell for every 4-neighbor contact."""
    vecs: set[tuple[int, int]] = set()
    for tr, tc in tile_cells:
        for dr, dc in _DIRECTIONS:
            pr = int(tr - dr)
            pc = int(tc - dc)
            if (int(pr), int(pc)) in plus_cells:
                vecs.add((int(dr), int(dc)))
    return vecs


def _allowed_touch_vectors(direction: tuple[int, int]) -> set[tuple[int, int]]:
    """Seed must touch the plus on a side perpendicular to its movement axis."""
    dr, dc = map(int, direction)
    if dc != 0:
        return {(-1, 0), (1, 0)}
    return {(0, -1), (0, 1)}


def _project_tile_coords(
    H: int,
    W: int,
    tile: np.ndarray,
    seed_r0: int,
    seed_c0: int,
    direction: tuple[int, int],
    skip_gap: int,
) -> dict:
    """
    Build the intermittent ray layout from one seed origin.

    Returns:
    - cells: painted non-bg cells for visible copies, including the seed slot
    - lane_cells: every cell in every tile slot, including skipped slots
    - visible_count: number of fully in-bounds visible copies
    """
    th, tw = tile.shape
    step_slots = int(skip_gap) + 1

    cells: list[tuple[int, int, int]] = []
    lane_cells: set[tuple[int, int]] = set()
    visible_count = 0
    slot_idx = 0

    while True:
        sr, sc = _slot_origin(int(seed_r0), int(seed_c0), tile, direction, int(slot_idx))
        if not _slot_intersects_grid(H, W, sr, sc, th, tw):
            break

        full_slot = _slot_fully_in_bounds(H, W, sr, sc, th, tw)
        for i in range(th):
            rr = int(sr + i)
            if not (0 <= rr < H):
                continue
            for j in range(tw):
                cc = int(sc + j)
                if not (0 <= cc < W):
                    continue
                lane_cells.add((int(rr), int(cc)))
                if int(slot_idx) % int(step_slots) == 0:
                    cells.append((int(rr), int(cc), int(tile[i, j])))

        if int(slot_idx) % int(step_slots) == 0 and full_slot:
            visible_count += 1

        slot_idx += 1

    return {
        "cells": cells,
        "lane_cells": sorted(lane_cells),
        "visible_count": int(visible_count),
    }


def _find_seed_candidates(
    H: int,
    W: int,
    plus_bbox: tuple[int, int, int, int],
    plus_cells: set[tuple[int, int]],
    tile: np.ndarray,
    direction: tuple[int, int],
) -> list[tuple[int, int]]:
    """Enumerate legal adjacent seed positions whose touching side is perpendicular to the ray."""
    pr0, pc0, pr1, pc1 = plus_bbox
    th, tw = tile.shape
    allowed = _allowed_touch_vectors(direction)
    candidates: list[tuple[int, int]] = []

    for sr0 in range(int(pr0 - th), int(pr1 + 2)):
        for sc0 in range(int(pc0 - tw), int(pc1 + 2)):
            tile_cells = [(int(sr0 + i), int(sc0 + j)) for i in range(th) for j in range(tw)]
            if any(not (0 <= r < H and 0 <= c < W) for r, c in tile_cells):
                continue
            if any((int(r), int(c)) in plus_cells for r, c in tile_cells):
                continue
            vecs = _touch_vectors(plus_cells, tile_cells)
            if len(vecs) != 1:
                continue
            if next(iter(vecs)) not in allowed:
                continue
            candidates.append((int(sr0), int(sc0)))

    return candidates


def _place_plus_and_ray(
    rng: np.random.Generator,
    plus_shape: np.ndarray,
    H: int,
    W: int,
    occ: np.ndarray,
    plus_occ: np.ndarray,
    out: np.ndarray,
    plus_color: int,
    low_color: int,
    high_color: int,
    skip_gap: int,
) -> dict | None:
    """Try to place one plus and one perpendicular-attachment intermittent ray."""
    ph, pw = plus_shape.shape

    for _ in range(_PLACEMENT_TRIES):
        r0 = int(rng.integers(1, H - ph))
        c0 = int(rng.integers(1, W - pw))
        plus_cells = [
            (int(r0 + dr), int(c0 + dc))
            for dr in range(ph)
            for dc in range(pw)
            if int(plus_shape[dr, dc]) == 1
        ]
        plus_set = set(plus_cells)

        if any(int(occ[r, c]) != 0 for r, c in plus_cells):
            continue
        if any(int(plus_occ[r, c]) == 1 for r, c in plus_cells):
            continue
        if any(
            0 <= int(r + dr) < H
            and 0 <= int(c + dc) < W
            and (int(r + dr), int(c + dc)) not in plus_set
            and int(occ[int(r + dr), int(c + dc)]) != 0
            for r, c in plus_cells
            for dr, dc in _DIRECTIONS
        ):
            continue

        plus_bbox = (int(r0), int(c0), int(r0 + ph - 1), int(c0 + pw - 1))

        dirs = list(_DIRECTIONS)
        rng.shuffle(dirs)
        for direction in dirs:
            tile = _tile_for_direction(int(low_color), int(high_color), direction)
            candidates = _find_seed_candidates(H, W, plus_bbox, plus_set, tile, direction)
            rng.shuffle(candidates)

            for seed_r0, seed_c0 in candidates:
                proj = _project_tile_coords(
                    H,
                    W,
                    tile,
                    int(seed_r0),
                    int(seed_c0),
                    direction,
                    int(skip_gap),
                )
                if int(proj["visible_count"]) < _MIN_VISIBLE_COPIES:
                    continue
                if any(int(occ[r, c]) != 0 or int(plus_occ[r, c]) == 1 for r, c in proj["lane_cells"]):
                    continue

                for r, c in plus_cells:
                    occ[int(r), int(c)] = 1
                    out[int(r), int(c)] = int(plus_color)
                for r, c in plus_cells:
                    for dr2, dc2 in _DIRECTIONS:
                        rr = int(r + dr2)
                        cc = int(c + dc2)
                        if 0 <= rr < H and 0 <= cc < W:
                            plus_occ[int(rr), int(cc)] = 1

                for r, c in proj["lane_cells"]:
                    occ[int(r), int(c)] = 1
                for r, c, v in proj["cells"]:
                    out[int(r), int(c)] = int(v)

                touch_vec = next(iter(_touch_vectors(plus_set, [(int(seed_r0 + i), int(seed_c0 + j)) for i in range(tile.shape[0]) for j in range(tile.shape[1])])))
                return {
                    "tile": tile,
                    "cells": proj["cells"],
                    "seed_r0": int(seed_r0),
                    "seed_c0": int(seed_c0),
                    "visible_count": int(proj["visible_count"]),
                    "direction": tuple(map(int, direction)),
                    "touch_vec": tuple(map(int, touch_vec)),
                }

    return None


def _build_input_from_output(
    out: np.ndarray,
    pluses: list[dict],
    bg: int,
    rng: np.random.Generator,
) -> np.ndarray:
    """Keep one sufficiently long intermittent ray and erase all other non-seed copies."""
    candidates = [i for i, p in enumerate(pluses) if int(p["visible_count"]) >= _PRE_EXT_MIN_VISIBLE]
    if not candidates:
        raise ValueError("no sufficiently long demonstration ray")

    # Any candidate already shows at least three visible copies, which is enough
    # for a human (and infer_skip_gap_from_demo) to recover the dashed rhythm.
    # Sampling uniformly avoids systematically favoring directions that merely
    # have more free space, improving branch coverage across train examples.
    pre_idx = int(rng.choice(candidates))

    inp = out.copy()
    for i, p in enumerate(pluses):
        if i == pre_idx:
            continue
        th, tw = p["tile"].shape
        sr0 = int(p["seed_r0"])
        sc0 = int(p["seed_c0"])
        seed_set = {(int(sr0 + ii), int(sc0 + jj)) for ii in range(th) for jj in range(tw)}
        for r, c, _ in p["cells"]:
            if (int(r), int(c)) not in seed_set:
                inp[int(r), int(c)] = int(bg)
    return inp


def _gen_attempt(rng: np.random.Generator) -> np.ndarray | None:
    H = int(rng.integers(*_H_RANGE))
    W = int(rng.integers(*_W_RANGE))

    bg = int(rng.integers(0, 10))
    others = [c for c in range(10) if c != bg]
    plus_color = int(rng.choice(others))
    rest = [c for c in others if c != plus_color]
    pair = sorted(map(int, rng.choice(rest, size=2, replace=False)))
    low_color, high_color = int(pair[0]), int(pair[1])

    plus_shape = _PLUS_SHAPES[int(rng.integers(0, len(_PLUS_SHAPES)))]
    n_plus = int(rng.integers(*_N_PLUS_RANGE))
    skip_gap = int(rng.choice(_SKIP_GAP_CHOICES))

    out = np.full((H, W), bg, dtype=int)
    occ = np.zeros((H, W), dtype=int)
    plus_occ = np.zeros((H, W), dtype=int)

    pluses: list[dict] = []
    for _ in range(n_plus):
        info = _place_plus_and_ray(
            rng,
            plus_shape,
            H,
            W,
            occ,
            plus_occ,
            out,
            int(plus_color),
            int(low_color),
            int(high_color),
            int(skip_gap),
        )
        if info is None:
            return None
        pluses.append(info)

    if max(int(p["visible_count"]) for p in pluses) < _PRE_EXT_MIN_VISIBLE:
        return None
    if any(
        int(p["touch_vec"][0]) * int(p["direction"][0]) + int(p["touch_vec"][1]) * int(p["direction"][1]) != 0
        for p in pluses
    ):
        return None

    try:
        inp = _build_input_from_output(out, pluses, int(bg), rng)
    except ValueError:
        return None

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
