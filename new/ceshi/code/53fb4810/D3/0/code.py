# Variant of 53fb4810
# Dimension: D3 — 轨迹涂色模式 (trajectory)
# Change: periodic_tiling — 标记瓦片沿射线方向周期性完整重复，形成均匀的平铺图案... → 间歇性跳跃平铺

from __future__ import annotations

# Design analysis for the D3 variant
# ==================================
# a) Concrete behavior and trace
#    - Original task: from a seed tile attached to one arm of a plus, fill every
#      subsequent tile slot continuously to the boundary.
#    - Variant here: keep the same plus/seed geometry, but realize
#      "intermittent_skip" as: draw one full tile copy, then leave a fixed number
#      of whole-tile slots blank, then draw the next copy, and so on.
#      The blank run length is global within one grid and is inferred from the
#      single pre-extended demonstration ray.
#    - Example (horizontal 1x2 tile [2,3], skip_gap=1):
#      original slots: [23][23][23][23]...
#      variant slots : [23][bg][23][bg][23]...
#      The changed step is the trajectory painter, not the anchor detection.
#
# b) Complexity calibration
#    - To keep the new rule visible but not over-parameterized, skip_gap is
#      limited to {1, 2}. These two cases are easy to distinguish by eye:
#      alternating tile/gap, or tile separated by two blank tile slots.
#    - The one long example ray still provides repeated evidence, while the
#      remaining pluses only show compact seeds, so the puzzle remains an
#      extrapolation task rather than a memorization task.
#
# c) Hidden assumptions from the original solver that break here
#    - Original extend_repeating_tile assumes the pre-extended example is one
#      connected periodic strip. That is no longer true: the demonstration ray is
#      disconnected because the skipped slots stay background.
#    - Original logic never needed to infer spacing, only direction. The new
#      solver must inspect full tile-sized slots along the demonstrated ray and
#      recover a constant jump step.
#    - Original mark_lab usage is still valid because only the seed component
#      touches the plus. Detached copies farther away do not affect attachment
#      detection.
#
# d) Placement legality guarantees in the generator
#    - All pluses still share one symmetric shape and stay disjoint.
#    - Each seed is edge-adjacent to exactly one nozzle and cross-axis aligned
#      exactly as in the base task.
#    - The generator reserves the entire ray corridor, including skipped slots, so
#      no other plus or ray can intrude into a blank interval and break the
#      solver's slot-by-slot inference.
#
# e) Visual observability
#    - The only visible innovation is the ray itself: it becomes a dashed
#      repetition of complete tiles. This is directly observable from 2-4
#      examples without relying on color ids or hidden ordering.
#    - One example ray is kept long enough to show at least three visible copies,
#      so the jump size is recoverable instead of being guessed.
#
# f) Visual cleanliness
#    - Full-lane reservation avoids overlaps between occupied slots, skipped
#      slots, and other objects. Blank intervals remain genuinely empty, so the
#      dashed structure is easy to read.
#
# g) Color inference
#    - Colors do not change under the variant; only placement changes. The human
#      still infers tile content locally from each seed and infers spacing from
#      the single demonstrated ray.
#
# h) Post-change solver assumptions and generator support
#    - main() assumes exactly one seed component per plus is 4-neighbor adjacent
#      to that plus. The generator enforces one seed per plus and prevents other
#      non-bg cells from touching the plus.
#    - main() assumes the demonstrated ray occupies exact whole-tile slots whose
#      full bboxes are either identical to the seed tile or all background. The
#      generator enforces this by reserving the entire lane and drawing exact
#      copies only on allowed slots.
#    - main() infers one global skip gap shared by all pluses in a grid. The
#      generator samples a single skip_gap per grid and uses it for every ray.

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
    """Extend a tile outward while leaving fixed-size blank tile slots between copies."""
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
    """Count how many full tile slots fit from the seed slot onward."""
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
    - inferred skip_gap if this marker is the pre-extended demonstration ray,
      else None
    - number of visible full copies observed on this ray
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
    Extend marker tiles away from '+' anchors, but insert regular blank jumps between copies.

    Input structure:
    - The background is the most common color, and one dominant non-background color forms several '+' sprites.
    - Each '+' has exactly one small seed tile attached flush to one arm, with the seed span matching that arm's nozzle width.
    - Exactly one seed is already shown as a longer dashed ray: identical tile copies reappear after fixed blank tile-sized gaps.

    Rule:
    - For every seed touching a '+', use the touching side to determine the outward ray direction.
    - Read the long demonstrated ray to infer the shared jump spacing in whole-tile slots.
    - Repaint each seed tile along its ray by placing the same tile every fixed step, leaving the intermediate slots as background.
    - '+' pixels are protected and never overwritten.

    Key insight: The pre-extended example ray teaches not only the tile pattern but also the global spacing between visible copies; once that spacing is known, every other seed extends in the same dashed rhythm.
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
                nozzle_cols = sorted(int(cc) for cc in range(int(pc0), int(pc1) + 1) if plus_mask[nozzle_row, cc])
                seed_cols = list(range(int(c0), int(c1) + 1))
                assert seed_cols == nozzle_cols
            else:
                nozzle_col = int(pc1) if dc == 1 else int(pc0)
                nozzle_rows = sorted(int(rr) for rr in range(int(pr0), int(pr1) + 1) if plus_mask[rr, nozzle_col])
                seed_rows = list(range(int(r0), int(r1) + 1))
                assert seed_rows == nozzle_rows

            markers.append(
                {
                    "tile": tile,
                    "r0": int(r0),
                    "c0": int(c0),
                    "direction": (int(dr), int(dc)),
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


# --------------- generator (variant D3 / intermittent_skip) ---------------

# === CFG: Tunable generation parameters ===
_H_RANGE = (13, 29)
_W_RANGE = (13, 23)
_N_PLUS_RANGE = (2, 4)
_SEED_ALONG_CHOICES = [1, 2]
_SEED_MAX_COLORS = 4
_SEED_MAX_CROSS = 2
_SKIP_GAP_CHOICES = [1, 2]
_MIN_VISIBLE_COPIES = 2
_PRE_EXT_MIN_VISIBLE = 3
_PLACEMENT_TRIES = 120
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

_DIRECTIONS = [(1, 0), (-1, 0), (0, 1), (0, -1)]


def _nozzle_span(plus_mask: np.ndarray, direction: tuple[int, int]) -> tuple[int, int]:
    """Return the inclusive span of the nozzle along the seed cross-axis."""
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
    cross = int(e - s + 1)
    if cross > _SEED_MAX_CROSS:
        return None

    along = int(rng.choice(_SEED_ALONG_CHOICES))
    if along == 1 and cross == 1:
        along = 2

    _dr, dc = direction
    th, tw = (along, cross) if dc == 0 else (cross, along)

    n_cells = int(th * tw)
    max_k = min(_SEED_MAX_COLORS, n_cells, len(palette))
    n_colors = int(rng.integers(1, max_k + 1))
    chosen = list(rng.choice(palette, size=n_colors, replace=False))
    vals = list(chosen) + list(rng.choice(chosen, size=n_cells - n_colors, replace=True))
    rng.shuffle(vals)
    return np.array(vals, dtype=int).reshape(th, tw)


def _seed_origin(
    plus_bbox: tuple[int, int, int, int],
    plus_mask: np.ndarray,
    direction: tuple[int, int],
    tile: np.ndarray,
) -> tuple[int, int]:
    """Anchor the seed tile flush to the chosen nozzle."""
    pr0, pc0, pr1, pc1 = plus_bbox
    dr, dc = direction
    th, tw = tile.shape
    s, _ = _nozzle_span(plus_mask, direction)

    if dc == 0:
        return (int(pr1 + 1), int(pc0 + s)) if dr == 1 else (int(pr0 - th), int(pc0 + s))
    return (int(pr0 + s), int(pc1 + 1)) if dc == 1 else (int(pr0 + s), int(pc0 - tw))


def _project_tile_coords(
    H: int,
    W: int,
    plus_bbox: tuple[int, int, int, int],
    plus_mask: np.ndarray,
    direction: tuple[int, int],
    tile: np.ndarray,
    skip_gap: int,
) -> dict:
    """
    Build the intermittent ray layout.

    Returns:
    - cells: painted non-bg cells for all visible copies
    - lane_cells: every cell in every tile slot, including skipped slots
    - seed_r0 / seed_c0: top-left of the seed tile
    - visible_count: number of fully in-bounds visible copies
    """
    th, tw = tile.shape
    seed_r0, seed_c0 = _seed_origin(plus_bbox, plus_mask, direction, tile)
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
        "seed_r0": int(seed_r0),
        "seed_c0": int(seed_c0),
        "visible_count": int(visible_count),
    }


def _place_plus_and_ray(
    rng: np.random.Generator,
    plus_shape: np.ndarray,
    palette: list[int],
    H: int,
    W: int,
    occ: np.ndarray,
    plus_occ: np.ndarray,
    out: np.ndarray,
    plus_color: int,
    skip_gap: int,
) -> dict | None:
    """Try to place one plus sprite with one intermittent ray."""
    ph, pw = plus_shape.shape

    for _ in range(_PLACEMENT_TRIES):
        r0 = int(rng.integers(1, H - ph))
        c0 = int(rng.integers(1, W - pw))
        plus_cells = [
            (int(r0 + dr), int(c0 + dc))
            for dr in range(ph)
            for dc in range(pw)
            if plus_shape[dr, dc]
        ]

        if any(int(occ[r, c]) != 0 for r, c in plus_cells):
            continue
        if any(int(plus_occ[r, c]) == 1 for r, c in plus_cells):
            continue

        dirs = list(_DIRECTIONS)
        rng.shuffle(dirs)
        for d in dirs:
            tile = _build_seed_tile(plus_shape, d, palette, rng)
            if tile is None:
                continue

            proj = _project_tile_coords(
                H,
                W,
                (int(r0), int(c0), int(r0 + ph - 1), int(c0 + pw - 1)),
                plus_shape,
                d,
                tile,
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

            return {
                "tile": tile,
                "cells": proj["cells"],
                "seed_r0": int(proj["seed_r0"]),
                "seed_c0": int(proj["seed_c0"]),
                "visible_count": int(proj["visible_count"]),
                "direction": tuple(map(int, d)),
            }

    return None


def _build_input_from_output(
    out: np.ndarray,
    pluses: list[dict],
    bg: int,
    rng: np.random.Generator,
) -> np.ndarray:
    """Keep one full intermittent ray and erase all other non-seed copies."""
    candidates = [i for i, p in enumerate(pluses) if int(p["visible_count"]) >= _PRE_EXT_MIN_VISIBLE]
    if not candidates:
        raise ValueError("no sufficiently long demonstration ray")

    best_visible = max(int(pluses[i]["visible_count"]) for i in candidates)
    best = [i for i in candidates if int(pluses[i]["visible_count"]) == int(best_visible)]
    pre_idx = int(rng.choice(best))

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
    palette = [c for c in others if c != plus_color]

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
            palette,
            H,
            W,
            occ,
            plus_occ,
            out,
            plus_color,
            int(skip_gap),
        )
        if info is None:
            return None
        pluses.append(info)

    if max(int(p["visible_count"]) for p in pluses) < _PRE_EXT_MIN_VISIBLE:
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
