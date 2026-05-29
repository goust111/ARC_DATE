# Combo variant of e12f9a14 (index 40)
# combo_spec: D1/0+D8/0+D19/0
# Name: 正交射线+碰撞转90度+始终单像素

import random

import numpy as np

from arc_utils import get_bg, paste


# Design analysis for this combo variant
# --------------------------------------
# concretization:
#   The framed 2x2 color blocks are kept from the source task, but every meaningful
#   opening is now a side opening. Top/bottom openings launch vertical rays and
#   left/right openings launch horizontal rays. The output therefore contains only
#   grid-aligned 1-pixel paths, never diagonal strokes or 2x2 thick bands.
# behavior trace:
#   Source task local event:
#   - a corner opening could emit a diagonal beam;
#   - a 90-degree collision left the meeting cell empty and redirected both beams by
#     45 degrees, while a 45-degree collision created a thicker zigzag.
#   This combo local event:
#   - a side opening at local (1,3) emits a horizontal ray to the right;
#   - if that ray and a downward vertical ray would enter the same background cell,
#     neither paints the shared target cell;
#   - instead, each beam stays on its own last painted cell, swaps axis with the other
#     beam, and continues as a 1-pixel orthogonal turn into the same quadrant.
#   The visible difference is strong: the original bends diagonally or thickens, while
#   this combo produces clean rectilinear elbows around a single untouched hole.
# complexity calibration:
#   If the generator allowed arbitrary side-only rays with no collisions, the task would
#   collapse into a much simpler straight-line puzzle. To keep the fusion behavior
#   observable, each instance is forced to contain at least one synchronized horizontal/
#   vertical collision with visible approach and visible post-turn continuation.
# implicit assumption audit:
#   The original main() depended on several assumptions that no longer hold:
#   - gap -> 8-direction mapping: replaced by 4-direction mapping, so corner gaps are
#     ignored and only side gaps are emitted;
#   - collision classification by 45 vs 90 degrees: removed, because every supported
#     collision is orthogonal and always stays 1-pixel wide;
#   - ownership/truncation logic for previously painted diagonal cells: removed, because
#     this combo resolves collisions before the shared cell is painted, so no cell ever
#     needs to carry two colors or be widened afterward.
#   New assumptions introduced here:
#   - every meaningful collision is a same-tick horizontal/vertical meeting at one bg
#     cell; the generator enforces synchronized travel times to the meeting point;
#   - no ray later runs into an already painted perpendicular segment ("retro-collision");
#     the generator rejects layouts with any path overlap beyond the planned turns;
#   - same-axis overlaps or head-on conflicts are absent; extra beams are chosen so their
#     path sets are disjoint from all other beam paths.
# placement legality:
#   Frames keep the original 4x4 footprint and >=2-cell bbox clearance. The generator
#   first places full frames as obstacles, then chooses one side gap per block only if
#   the full emitted path is inside the grid, stays on background, and does not overlap
#   reserved paths from the collision pair or other filler beams.
# visual observability:
#   Every accepted puzzle has at least one turn collision with >=3 approach steps and
#   >=2 post-turn steps on both rays. A human can therefore see all three fused changes
#   at once: orthogonal launch, 90-degree bend, and uniform 1-pixel width.
# visual cleanliness:
#   The generator uses exactly one gap per block, forbids extra path intersections, and
#   keeps collision holes away from frame cells and seed cells. The result is sparse
#   enough that each elbow can be followed visually without ambiguous overpainting.
# color inference:
#   Colors still mean what they meant in the source task: each ray uses the color of its
#   own inner 2x2 block. No hidden ordering, color ranking, or palette-based remapping
#   is introduced.
# new solver assumptions checked against generator:
#   - every emitted gap is one of the eight side-border cells;
#   - at least one perpendicular beam pair reaches the same target cell on the same tick;
#   - after that turn, both outgoing corridors remain background for at least 2 cells;
#   - all non-collision beam paths are pairwise disjoint.


_SIDE_GAPS = (
    (0, 1), (0, 2),
    (1, 3), (2, 3),
    (3, 2), (3, 1),
    (2, 0), (1, 0),
)


def _find_blocks_2x2(g: np.ndarray, bg: int) -> list[tuple[int, int, int]]:
    h, w = g.shape
    blocks: list[tuple[int, int, int]] = []
    for r in range(h - 1):
        for c in range(w - 1):
            v = int(g[r, c])
            if v == bg:
                continue
            if int(g[r, c + 1]) == v and int(g[r + 1, c]) == v and int(g[r + 1, c + 1]) == v:
                blocks.append((v, r, c))
    return blocks


def _infer_frame_color(g: np.ndarray, bg: int, br: int, bc: int, block_col: int) -> int:
    h, w = g.shape
    fr0, fc0 = br - 1, bc - 1
    fr1, fc1 = br + 2, bc + 2
    vals: list[int] = []
    for r in range(fr0, fr1 + 1):
        for c in range(fc0, fc1 + 1):
            if not (0 <= r < h and 0 <= c < w):
                continue
            if not (r in (fr0, fr1) or c in (fc0, fc1)):
                continue
            v = int(g[r, c])
            if v != bg and v != block_col:
                vals.append(v)
    frame_colors = sorted(set(vals))
    assert len(frame_colors) == 1, "expected a unique frame color around each 2x2 block"
    return frame_colors[0]


def _frame_gaps_for_block(g: np.ndarray, bg: int, wall: int, br: int, bc: int) -> list[tuple[int, int]]:
    h, w = g.shape
    fr0, fc0 = br - 1, bc - 1
    fr1, fc1 = br + 2, bc + 2
    gaps: list[tuple[int, int]] = []
    for r in range(fr0, fr1 + 1):
        for c in range(fc0, fc1 + 1):
            if not (0 <= r < h and 0 <= c < w):
                continue
            if not (r in (fr0, fr1) or c in (fc0, fc1)):
                continue
            if int(g[r, c]) == bg:
                gaps.append((r, c))
    out: list[tuple[int, int]] = []
    for r, c in gaps:
        dr = -1 if r == fr0 else (1 if r == fr1 else 0)
        dc = -1 if c == fc0 else (1 if c == fc1 else 0)
        if dr == 0 and dc == 0:
            continue
        if dr != 0 and dc != 0:
            continue
        ok = True
        if dr == 0:
            ok = (
                0 <= r - 1 < h
                and 0 <= r + 1 < h
                and int(g[r - 1, c]) == wall
                and int(g[r + 1, c]) == wall
            )
        else:
            ok = (
                0 <= c - 1 < w
                and 0 <= c + 1 < w
                and int(g[r, c - 1]) == wall
                and int(g[r, c + 1]) == wall
            )
        if ok:
            out.append((r, c))
    return out


def _gap_dir_for_block(br: int, bc: int, gr: int, gc: int) -> tuple[int, int] | None:
    dr = -1 if gr == br - 1 else (1 if gr == br + 2 else 0)
    dc = -1 if gc == bc - 1 else (1 if gc == bc + 2 else 0)
    if dr != 0 and dc != 0:
        return None
    if dr == 0 and dc == 0:
        return None
    return dr, dc


def _axis(d: tuple[int, int]) -> str:
    return "h" if d[0] == 0 else "v"


def _path_from_seed(seed: tuple[int, int], direction: tuple[int, int], steps: int) -> list[tuple[int, int]]:
    r, c = seed
    dr, dc = direction
    return [(r + dr * i, c + dc * i) for i in range(steps + 1)]


def _tail_from(start: tuple[int, int], direction: tuple[int, int], h: int, w: int) -> list[tuple[int, int]]:
    dr, dc = direction
    r, c = start
    out: list[tuple[int, int]] = []
    while 0 <= r < h and 0 <= c < w:
        out.append((r, c))
        r += dr
        c += dc
    return out


def _ray_cells(grid: np.ndarray, bg: int, seed: tuple[int, int], direction: tuple[int, int]) -> list[tuple[int, int]] | None:
    h, w = grid.shape
    r, c = seed
    dr, dc = direction
    cells = [seed]
    while True:
        r += dr
        c += dc
        if not (0 <= r < h and 0 <= c < w):
            break
        if int(grid[r, c]) != bg:
            return None
        cells.append((r, c))
    return cells


class _Beam:
    __slots__ = ("bid", "color", "pos", "dir", "active", "turned")

    def __init__(self, bid: int, color: int, pos: tuple[int, int], direction: tuple[int, int]):
        self.bid = bid
        self.color = int(color)
        self.pos = pos
        self.dir = direction
        self.active = True
        self.turned = False


def _collect_beam_specs(g: np.ndarray) -> tuple[int, list[tuple[int, tuple[int, int], tuple[int, int]]]]:
    bg = get_bg(g)
    specs: list[tuple[int, tuple[int, int], tuple[int, int]]] = []
    for col, br, bc in _find_blocks_2x2(g, bg):
        frame_col = _infer_frame_color(g, bg, br, bc, col)
        for gr, gc in _frame_gaps_for_block(g, bg, frame_col, br, bc):
            direction = _gap_dir_for_block(br, bc, gr, gc)
            if direction is None:
                continue
            specs.append((col, (gr, gc), direction))
    return bg, specs


def _simulate(grid: np.ndarray) -> tuple[np.ndarray, dict[str, int]]:
    g = np.asarray(grid, dtype=int)
    h, w = g.shape
    bg, specs = _collect_beam_specs(g)
    assert specs, "expected at least one side-gap beam"

    out = g.copy()
    beams: dict[int, _Beam] = {}
    painted: set[tuple[int, int]] = set()
    blocked: set[tuple[int, int]] = set()

    turn_collisions = 0
    post_turn_steps = 0
    turned_beams: set[int] = set()

    def in_bounds(cell: tuple[int, int]) -> bool:
        r, c = cell
        return 0 <= r < h and 0 <= c < w

    def paint(cell: tuple[int, int], color: int) -> None:
        out[cell[0], cell[1]] = int(color)
        painted.add(cell)

    for bid, (color, pos, direction) in enumerate(specs):
        beams[bid] = _Beam(bid, color, pos, direction)
        paint(pos, color)

    max_iters = h * w * 20
    for _ in range(max_iters):
        prop: dict[int, tuple[int, int]] = {}
        for bid, beam in beams.items():
            if not beam.active:
                continue
            nxt = (beam.pos[0] + beam.dir[0], beam.pos[1] + beam.dir[1])
            if not in_bounds(nxt):
                beam.active = False
                continue
            if nxt in blocked:
                beam.active = False
                continue
            if int(g[nxt[0], nxt[1]]) != bg:
                beam.active = False
                continue
            if nxt in painted:
                beam.active = False
                continue
            prop[bid] = nxt

        if not prop:
            break

        cell_to_bids: dict[tuple[int, int], list[int]] = {}
        for bid, cell in prop.items():
            cell_to_bids.setdefault(cell, []).append(bid)

        collided: set[int] = set()
        progress = False
        for cell, bidders in sorted(cell_to_bids.items()):
            if len(bidders) != 2:
                continue
            a, b = sorted(bidders)
            da = beams[a].dir
            db = beams[b].dir
            if _axis(da) == _axis(db):
                continue
            blocked.add(cell)
            beams[a].dir = db
            beams[b].dir = da
            beams[a].turned = True
            beams[b].turned = True
            turned_beams.add(a)
            turned_beams.add(b)
            collided.add(a)
            collided.add(b)
            turn_collisions += 1
            progress = True

        for bid, cell in sorted(prop.items()):
            if bid in collided:
                continue
            beam = beams[bid]
            if cell in blocked or cell in painted:
                beam.active = False
                continue
            paint(cell, beam.color)
            beam.pos = cell
            if beam.turned:
                post_turn_steps += 1
            progress = True

        if not progress:
            break
    else:
        raise AssertionError("orthogonal turn simulation did not converge")

    stats = {
        "beam_count": len(specs),
        "turn_collisions": turn_collisions,
        "blocked_cells": len(blocked),
        "turned_beams": len(turned_beams),
        "post_turn_steps": post_turn_steps,
    }
    return out, stats


def main(grid: np.ndarray) -> np.ndarray:
    """
    Emit orthogonal single-pixel rays from framed 2x2 blocks, bending them by 90 degrees when perpendicular rays meet.

    Input structure:
    - The background is the most common color. Each foreground 2x2 monochrome block sits inside a shared 4x4 frame color.
    - Only side openings on the frame border are meaningful here; a top/bottom opening encodes a vertical launch and a left/right opening encodes a horizontal launch.

    Rule:
    - Every valid side gap emits one ray in the outward cardinal direction, using the color of that block.
    - Rays advance one cell at a time through background only, always staying 1 pixel wide.
    - If a horizontal ray and a vertical ray would enter the same cell on the same tick, that meeting cell stays background and both rays turn 90 degrees one step earlier by swapping axes, then keep extending along their new orthogonal directions.

    Key insight:
    - The fusion removes both diagonals and thickened collision trails. What remains is a pure row/column process where the distinctive event is a synchronized perpendicular meeting that produces two thin elbow-shaped continuations around a single untouched collision cell.
    """
    out, _stats = _simulate(np.asarray(grid, dtype=int))
    return out


# --------------- generator (o5x) ---------------

BLOCK_COUNT_RANGE = (2, 5)
GRID_SIZE_MAX = 31
GRID_SIZE_MIN_BY_BLOCKS = {2: 14, 3: 18, 4: 22}
BG_CHOICES = [0, 8]
BG_WEIGHTS = [0.25, 0.75]
FRAME_COLOR_FOR_BG = {8: 3, 0: 8}
PALETTE_FOR_BG = {
    8: [1, 2, 4, 6, 7, 9],
    0: [1, 2, 3, 4, 6, 7, 9],
}
FRAME_BBOX_GAP = 2
PLACEMENT_MARGIN = 2
MIN_STRAIGHT_STEPS = 3
MIN_POST_TURN_STEPS = 2
MIN_BEAM_PIXELS_PER_GAP = 4
GEN_RETRIES = 600


def _frames_separated(a: tuple[int, int], b: tuple[int, int]) -> bool:
    ar0, ac0 = a
    br0, bc0 = b
    ar1, ac1 = ar0 + 3, ac0 + 3
    br1, bc1 = br0 + 3, bc0 + 3
    gap = FRAME_BBOX_GAP
    return ar1 + gap < br0 or br1 + gap < ar0 or ac1 + gap < bc0 or bc1 + gap < ac0


def _choose_positions(n: int, count: int, rng: np.random.Generator) -> list[tuple[int, int]] | None:
    margin = PLACEMENT_MARGIN
    all_positions = [(r, c) for r in range(margin, n - 5) for c in range(margin, n - 5)]
    selected: list[tuple[int, int]] = []
    for _ in range(count):
        viable = [p for p in all_positions if all(_frames_separated(p, q) for q in selected)]
        if not viable:
            return None
        selected.append(viable[int(rng.integers(len(viable)))])
    return selected


def _draw_framed_block(
    grid: np.ndarray,
    r0: int,
    c0: int,
    bg: int,
    frame_color: int,
    block_color: int,
    gaps: tuple[tuple[int, int], ...],
) -> None:
    patch = np.full((4, 4), frame_color, dtype=int)
    patch[1:3, 1:3] = block_color
    for gr, gc in gaps:
        patch[gr, gc] = bg
    paste(grid, patch, r0, c0)


def _pick_colors(rng: np.random.Generator, block_count: int) -> tuple[int, int, list[int]]:
    bg = int(rng.choice(BG_CHOICES, p=BG_WEIGHTS))
    frame_color = FRAME_COLOR_FOR_BG[bg]
    palette = [c for c in PALETTE_FOR_BG[bg] if c != frame_color]
    block_colors = list(rng.choice(palette, size=block_count, replace=False))
    return bg, frame_color, block_colors


def _grid_size_min(block_count: int) -> int:
    result = 12
    for threshold, min_n in sorted(GRID_SIZE_MIN_BY_BLOCKS.items()):
        if block_count >= threshold:
            result = min_n
    return result


def _beam_candidates_for_block(
    obstacles: np.ndarray,
    bg: int,
    block_idx: int,
    r0: int,
    c0: int,
) -> list[dict[str, object]]:
    candidates: list[dict[str, object]] = []
    for gap in _SIDE_GAPS:
        seed = (r0 + gap[0], c0 + gap[1])
        direction = _gap_dir_for_block(r0 + 1, c0 + 1, seed[0], seed[1])
        assert direction is not None
        cells = _ray_cells(obstacles, bg, seed, direction)
        if cells is None or len(cells) - 1 < MIN_STRAIGHT_STEPS:
            continue
        candidates.append(
            {
                "block_idx": block_idx,
                "gap": gap,
                "seed": seed,
                "dir": direction,
                "cells": cells,
                "set": set(cells),
            }
        )
    return candidates


def _collision_plan(
    h_beam: dict[str, object],
    v_beam: dict[str, object],
    obstacles: np.ndarray,
    bg: int,
) -> dict[str, object] | None:
    h, w = obstacles.shape
    hr, hc = h_beam["seed"]
    vr, vc = v_beam["seed"]
    _, hdc = h_beam["dir"]
    vdr, _ = v_beam["dir"]
    collision = (hr, vc)

    steps_h = (vc - hc) * hdc
    steps_v = (hr - vr) * vdr
    if steps_h < MIN_STRAIGHT_STEPS or steps_v < MIN_STRAIGHT_STEPS:
        return None
    if steps_h != steps_v:
        return None
    if collision not in h_beam["set"] or collision not in v_beam["set"]:
        return None
    if int(obstacles[collision[0], collision[1]]) != bg:
        return None
    if collision == h_beam["seed"] or collision == v_beam["seed"]:
        return None

    h_in = _path_from_seed(h_beam["seed"], h_beam["dir"], steps_h - 1)
    v_in = _path_from_seed(v_beam["seed"], v_beam["dir"], steps_v - 1)

    prev_h = h_in[-1]
    prev_v = v_in[-1]
    h_out_start = (prev_h[0] + vdr, prev_h[1])
    v_out_start = (prev_v[0], prev_v[1] + hdc)
    h_out = _tail_from(h_out_start, v_beam["dir"], h, w)
    v_out = _tail_from(v_out_start, h_beam["dir"], h, w)

    if len(h_out) < MIN_POST_TURN_STEPS or len(v_out) < MIN_POST_TURN_STEPS:
        return None
    if not h_out or not v_out:
        return None
    if any(int(obstacles[r, c]) != bg for r, c in h_out + v_out):
        return None

    h_path = list(h_in) + h_out
    v_path = list(v_in) + v_out
    h_set = set(h_path)
    v_set = set(v_path)
    if h_set & v_set:
        return None
    if collision in h_set or collision in v_set:
        return None

    return {
        "collision": collision,
        "paths": {
            int(h_beam["block_idx"]): h_path,
            int(v_beam["block_idx"]): v_path,
        },
        "reserved": h_set | v_set | {collision},
        "pair_blocks": {int(h_beam["block_idx"]), int(v_beam["block_idx"])},
    }


def _find_collision_pairs(
    obstacles: np.ndarray,
    bg: int,
    candidates: list[dict[str, object]],
) -> list[tuple[dict[str, object], dict[str, object], dict[str, object]]]:
    horizontals = [c for c in candidates if _axis(c["dir"]) == "h"]
    verticals = [c for c in candidates if _axis(c["dir"]) == "v"]
    pairs: list[tuple[dict[str, object], dict[str, object], dict[str, object]]] = []
    for h_beam in horizontals:
        for v_beam in verticals:
            if h_beam["block_idx"] == v_beam["block_idx"]:
                continue
            plan = _collision_plan(h_beam, v_beam, obstacles, bg)
            if plan is None:
                continue
            pairs.append((h_beam, v_beam, plan))
    return pairs


def _choose_extra_beams(
    block_count: int,
    candidates_by_block: list[list[dict[str, object]]],
    pair_beams: tuple[dict[str, object], dict[str, object]],
    reserved: set[tuple[int, int]],
    rng: np.random.Generator,
) -> dict[int, dict[str, object]] | None:
    chosen = {
        int(pair_beams[0]["block_idx"]): pair_beams[0],
        int(pair_beams[1]["block_idx"]): pair_beams[1],
    }
    occupied = set(reserved)

    remaining_blocks = [i for i in range(block_count) if i not in chosen]
    rng.shuffle(remaining_blocks)
    for block_idx in remaining_blocks:
        options = candidates_by_block[block_idx][:]
        rng.shuffle(options)
        picked = None
        for beam in options:
            beam_cells = beam["set"]
            if beam_cells & occupied:
                continue
            picked = beam
            break
        if picked is None:
            return None
        chosen[block_idx] = picked
        occupied |= picked["set"]
    return chosen


def _gen_attempt() -> np.ndarray | None:
    rng = np.random.default_rng()
    py_rng = random.Random(int(rng.integers(1 << 30)))

    block_count = int(rng.integers(*BLOCK_COUNT_RANGE))
    n = int(rng.integers(_grid_size_min(block_count), GRID_SIZE_MAX))
    bg, frame_color, block_colors = _pick_colors(rng, block_count)

    positions = _choose_positions(n, block_count, rng)
    if positions is None:
        return None

    obstacles = np.full((n, n), bg, dtype=int)
    for (r0, c0), color in zip(positions, block_colors):
        _draw_framed_block(obstacles, r0, c0, bg, frame_color, int(color), ())

    candidates_by_block: list[list[dict[str, object]]] = []
    flat_candidates: list[dict[str, object]] = []
    for block_idx, (r0, c0) in enumerate(positions):
        candidates = _beam_candidates_for_block(obstacles, bg, block_idx, r0, c0)
        if not candidates:
            return None
        candidates_by_block.append(candidates)
        flat_candidates.extend(candidates)

    pairs = _find_collision_pairs(obstacles, bg, flat_candidates)
    if not pairs:
        return None
    py_rng.shuffle(pairs)

    chosen: dict[int, dict[str, object]] | None = None
    plan_used: dict[str, object] | None = None
    for h_beam, v_beam, plan in pairs:
        chosen = _choose_extra_beams(block_count, candidates_by_block, (h_beam, v_beam), set(plan["reserved"]), rng)
        if chosen is None:
            continue
        plan_used = plan
        break

    if chosen is None or plan_used is None:
        return None

    grid = np.full((n, n), bg, dtype=int)
    gap_total = 0
    for block_idx, ((r0, c0), color) in enumerate(zip(positions, block_colors)):
        gap = chosen[block_idx]["gap"]
        gap_total += 1
        _draw_framed_block(grid, r0, c0, bg, frame_color, int(color), (gap,))

    if int(np.bincount(grid.ravel(), minlength=10).argmax()) != bg:
        return None

    try:
        out, stats = _simulate(grid)
    except Exception:
        return None

    if stats["beam_count"] != gap_total:
        return None
    if stats["turn_collisions"] < 1:
        return None
    if stats["turned_beams"] < 2:
        return None
    if stats["post_turn_steps"] < 2 * MIN_POST_TURN_STEPS:
        return None
    if stats["blocked_cells"] != stats["turn_collisions"]:
        return None
    if np.count_nonzero(out != grid) < gap_total * MIN_BEAM_PIXELS_PER_GAP:
        return None
    return grid


def generate_input() -> np.ndarray:
    for _ in range(GEN_RETRIES):
        grid = _gen_attempt()
        if grid is not None:
            return grid
    raise RuntimeError("failed to generate a valid e12f9a14 combo input")
