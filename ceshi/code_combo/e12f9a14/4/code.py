# Combo variant of e12f9a14 (index 4)
# combo_spec: D2/2+D3/2
# Name: 穿过交换颜色+碰撞后颜色继承

from __future__ import annotations

import itertools

import numpy as np

from arc_utils import get_bg, paste


# Design analysis for the combo variant
# a) Concrete behavior and trace:
#    - We keep the original visible objects from the spec: a square background, several 2x2 monochrome
#      blocks, and one shared 4x4 frame color around each block. The combo changes only what happens
#      after emission.
#    - To make "pass through and exchange color" readable on a single-layer pixel raster, the generator
#      uses one corner gap per frame, so every beam is diagonal. A diagonal-vs-diagonal crossing is the
#      one place where two trajectories can visibly pass through each other without fighting for one cell.
#    - Small trace: a red NE beam and a blue NW beam approach the same 2x2 crossing square. In the base
#      task they would redirect or merge depending on the angle. In this combo they keep their diagonal
#      directions, step through the crossing square, and the red continuation is now painted blue while
#      the blue continuation is now painted red. If a later yellow beam crosses the already-drawn blue
#      continuation, that older suffix is recolored yellow-from-the-crossing-onward and the moving beam
#      inherits blue. This realizes the multi-collision color-transfer chain from the combo description.
# b) Complexity calibration:
#    - Pure pass-through is too weak if only one crossing occurs, so generation requires multiple swap
#      events and at least one beam that changes color twice. That keeps the D3-style inheritance chain
#      visible rather than collapsing into a single local exchange.
#    - We still cap each frame to one corner gap, which avoids clutter from many parallel trails and
#      keeps the number of distinct beam colors equal to the number of blocks, a structure humans can
#      track from a few examples.
# c) Implicit-assumption audit against the original main():
#    - The original solver assumed collisions are expressed by geometry changes: turning, zigzagging,
#      truncation, and blank gaps. None of those assumptions survive D2/2, so the combo removes every
#      branch that rewrites direction or blocks a cell on collision.
#    - The original paint logic assumed beam.color is immutable. D3/2 breaks that, so color becomes
#      persistent mutable state attached to each beam and must survive across later collisions.
#    - The original solver could safely forget old beam suffixes after a collision. The combo cannot,
#      because a later crossing with an already-painted segment must recolor the old suffix from the
#      crossing onward to preserve "post-collision uses the inherited color".
#    - Same-cell collisions are ambiguous under a one-layer pass-through drawing model. The combo solver
#      therefore treats them as invalid, and the generator guarantees they never appear by restricting
#      to corner gaps and filtering candidates through the simulator.
# d) Placement legality:
#    - Frames keep the original 4x4 geometry, remain fully in bounds, and preserve the original minimum
#      frame-box clearance so blocks do not overlap.
#    - Candidate corner gaps are accepted only when their diagonals have enough runway through background
#      cells; the simulator then rejects any layout that would create same-cell impacts.
# e) Visual observability:
#    - The geometry alone is now simple: straight diagonal lines. The interesting signal is the sharp
#      color boundary at crossings and the fact that a later crossing can recolor an earlier suffix.
#    - Distinct block colors plus repeated corner-gap emitters give humans a visible way to trace where
#      each continuation color came from without relying on numeric color ids.
# f) Visual cleanliness:
#    - One diagonal beam per frame prevents thick overdraw and keeps each crossing local and legible.
#    - Solver-side rejection of same-cell events removes the main source of unreadable overlap.
# g) Color-change inference:
#    - Every beam color is sourced from a visible 2x2 block, so each post-collision color segment can be
#      traced back to another emitter. Requiring multi-swap samples gives the training set repeated
#      evidence that colors propagate along collision chains instead of being fixed to one beam forever.
# New solver assumptions to review:
#    - Legal combo inputs use structural corner gaps only, so every beam direction is diagonal.
#    - All collisions are geometric diagonal crossings, either simultaneous beam-beam crossings or a
#      later beam crossing an already-painted diagonal segment. The generator enforces both assumptions.


_CORNER_GAPS = ((0, 0), (0, 3), (3, 3), (3, 0))


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
        assert dr != 0 or dc != 0, "expected candidate gap on frame border"
        ok = True
        if dr == 0:
            ok = (
                0 <= r - 1 < h
                and 0 <= r + 1 < h
                and int(g[r - 1, c]) == wall
                and int(g[r + 1, c]) == wall
            )
        elif dc == 0:
            ok = (
                0 <= c - 1 < w
                and 0 <= c + 1 < w
                and int(g[r, c - 1]) == wall
                and int(g[r, c + 1]) == wall
            )
        else:
            ok = (
                0 <= r - dr < h
                and 0 <= c < w
                and 0 <= r < h
                and 0 <= c - dc < w
                and int(g[r - dr, c]) == wall
                and int(g[r, c - dc]) == wall
            )
        if ok:
            out.append((r, c))
    out_set = set(out)
    if len(out_set) >= 3 and (fr1, fc1) in out_set and (fr0, bc + 1) in out_set:
        out = [p for p in out if p != (fr1, fc1)]
    return out


def _normalize_dir(dr: int, dc: int) -> tuple[int, int]:
    if dr == 0 and dc == 0:
        return (0, 0)
    if dr != 0 and dc != 0:
        return (int(np.sign(dr)), int(np.sign(dc)))
    if dr != 0:
        return (int(np.sign(dr)), 0)
    return (0, int(np.sign(dc)))


def _gap_dir(gap: tuple[int, int]) -> tuple[int, int]:
    gr, gc = gap
    dr = -1 if gr == 0 else (1 if gr == 3 else 0)
    dc = -1 if gc == 0 else (1 if gc == 3 else 0)
    return dr, dc


class _Beam:
    __slots__ = ("bid", "source_color", "color", "pos", "direction", "path", "enter_dirs", "active", "swap_count")

    def __init__(self, bid: int, color: int, pos: tuple[int, int], direction: tuple[int, int]):
        self.bid = bid
        self.source_color = int(color)
        self.color = int(color)
        self.pos = pos
        self.direction = direction
        self.path = [pos]
        self.enter_dirs = [direction]
        self.active = True
        self.swap_count = 0


def _simulate(
    grid: np.ndarray,
    *,
    strict_same_cell: bool,
    collect_stats: bool,
) -> tuple[np.ndarray | None, dict[str, int]]:
    g = np.asarray(grid, dtype=int)
    h, w = g.shape
    bg = get_bg(g)

    out = g.copy()
    blocks = _find_blocks_2x2(g, bg)
    assert blocks, "expected at least one 2x2 block in input"

    beams: dict[int, _Beam] = {}
    next_bid = 0

    cell_owner: dict[tuple[int, int], int] = {}
    cell_enter: dict[tuple[int, int], tuple[int, int]] = {}
    cell_index: dict[tuple[int, int], int] = {}
    ever_changed: set[int] = set()

    stats = {
        "beam_count": 0,
        "swap_events": 0,
        "visible_swap_events": 0,
        "cross_swaps": 0,
        "segment_swaps": 0,
        "same_cell_collisions": 0,
        "foreign_color_cells": 0,
        "changed_beam_count": 0,
        "max_beam_swaps": 0,
        "multi_swap_beams": 0,
    }

    def paint_cell(bid: int, cell: tuple[int, int], enter_dir: tuple[int, int]) -> None:
        r, c = cell
        out[r, c] = beams[bid].color
        cell_owner[cell] = bid
        cell_enter[cell] = enter_dir
        cell_index[cell] = len(beams[bid].path) - 1

    def move_beam(bid: int, nxt: tuple[int, int], enter_dir: tuple[int, int]) -> None:
        assert cell_owner.get(nxt) is None, "expected empty target cell before drawing"
        b = beams[bid]
        b.path.append(nxt)
        b.enter_dirs.append(enter_dir)
        b.pos = nxt
        paint_cell(bid, nxt, enter_dir)

    def assign_color(bid: int, color: int) -> bool:
        color = int(color)
        b = beams[bid]
        if b.color == color:
            return False
        b.color = color
        b.swap_count += 1
        ever_changed.add(bid)
        return True

    def recolor_suffix(bid: int, start_idx: int, new_color: int) -> bool:
        if not assign_color(bid, new_color):
            return False
        for cell in beams[bid].path[start_idx:]:
            if cell_owner.get(cell) == bid:
                out[cell[0], cell[1]] = int(new_color)
        return True

    def spawn_beam(color: int, pos: tuple[int, int], direction: tuple[int, int]) -> None:
        nonlocal next_bid
        assert int(out[pos[0], pos[1]]) == bg, "expected beam seed at a frame gap"
        assert pos not in cell_owner, "expected unique beam seed positions"
        assert abs(direction[0]) + abs(direction[1]) == 2, "combo variant expects corner-gap diagonal beams"
        bid = next_bid
        next_bid += 1
        beams[bid] = _Beam(bid, color, pos, direction)
        paint_cell(bid, pos, direction)

    for col, br, bc in blocks:
        frame_col = _infer_frame_color(g, bg, br, bc, col)
        gaps = _frame_gaps_for_block(g, bg, frame_col, br, bc)
        assert len(gaps) == 1, "combo generator is expected to provide one structural gap per frame"
        gr, gc = gaps[0]
        dr = -1 if gr == br - 1 else (1 if gr == br + 2 else 0)
        dc = -1 if gc == bc - 1 else (1 if gc == bc + 2 else 0)
        direction = _normalize_dir(dr, dc)
        spawn_beam(col, (gr, gc), direction)

    stats["beam_count"] = len(beams)
    assert beams, "expected at least one beam"

    def in_bounds(cell: tuple[int, int]) -> bool:
        r, c = cell
        return 0 <= r < h and 0 <= c < w

    def segment_owner_after_cross(p: tuple[int, int], q: tuple[int, int]) -> tuple[int, int] | None:
        dr, dc = q[0] - p[0], q[1] - p[1]
        if abs(dr) != 1 or abs(dc) != 1:
            return None
        owner_q = cell_owner.get(q)
        if owner_q is not None and cell_index.get(q, 0) > 0 and cell_enter.get(q) == (dr, dc):
            return owner_q, cell_index[q]
        owner_p = cell_owner.get(p)
        if owner_p is not None and cell_index.get(p, 0) > 0 and cell_enter.get(p) == (-dr, -dc):
            return owner_p, cell_index[p]
        return None

    max_iters = h * w * 50
    for _ in range(max_iters):
        active = [bid for bid, b in beams.items() if b.active]
        if not active:
            break

        prop: dict[int, tuple[tuple[int, int], tuple[int, int]]] = {}
        for bid in active:
            b = beams[bid]
            d = b.direction
            nxt = (b.pos[0] + d[0], b.pos[1] + d[1])
            if not in_bounds(nxt):
                b.active = False
                continue
            if int(g[nxt[0], nxt[1]]) != bg:
                b.active = False
                continue
            prop[bid] = (nxt, d)

        if not prop:
            break

        cell_events: list[tuple[int, int, int, int]] = []
        cell_to_bids: dict[tuple[int, int], list[int]] = {}
        for bid, (nxt, _d) in prop.items():
            cell_to_bids.setdefault(nxt, []).append(bid)
        for cell, bidders in cell_to_bids.items():
            if len(bidders) >= 2:
                ordered = sorted(bidders)
                cell_events.append((cell[0], cell[1], ordered[0], ordered[1]))
            elif cell in cell_owner:
                cell_events.append((cell[0], cell[1], bidders[0], cell_owner[cell]))

        if cell_events:
            stats["same_cell_collisions"] += len(cell_events)
            if strict_same_cell:
                raise AssertionError("combo variant expects only geometric diagonal crossings")
            return None, stats

        move_map = {(beams[bid].pos, nxt): bid for bid, (nxt, _d) in prop.items()}
        events: list[tuple[int, int, int, int, int, str]] = []

        cross_pairs: set[tuple[int, int]] = set()
        for (p, n), bid in list(move_map.items()):
            pr, pc = p
            nr, nc = n
            dr = nr - pr
            dc = nc - pc
            p2 = (pr, pc + dc)
            n2 = (pr + dr, pc)
            obid = move_map.get((p2, n2))
            if obid is None or obid == bid:
                continue
            cross_pairs.add(tuple(sorted((bid, obid))))
        for a, b in cross_pairs:
            ca = prop[a][0]
            cb = prop[b][0]
            events.append((0, min(ca[0], cb[0]), min(ca[1], cb[1]), a, b, "cross"))

        seg_hits: dict[tuple[int, int], tuple[int, int]] = {}
        for bid, (_nxt, d) in prop.items():
            pr, pc = beams[bid].pos
            dr, dc = d
            p2 = (pr, pc + dc)
            n2 = (pr + dr, pc)
            info = segment_owner_after_cross(p2, n2)
            if info is None:
                continue
            obid, split_idx = info
            if obid == bid:
                continue
            key = min(pr, pr + dr) * w + min(pc, pc + dc)
            seg_hits[(bid, obid)] = (split_idx, key)
        for (a, b), (split_idx, key) in seg_hits.items():
            events.append((1, key // w, key % w, a, b, f"seg:{split_idx}"))

        if events:
            events.sort()
            _prio, _r, _c, a, b, kind = events[0]
            if kind == "cross":
                old_a = beams[a].color
                old_b = beams[b].color
                changed_a = assign_color(a, old_b)
                changed_b = assign_color(b, old_a)
                stats["swap_events"] += 1
                stats["cross_swaps"] += 1
                if changed_a or changed_b:
                    stats["visible_swap_events"] += 1
                for bid in sorted((a, b)):
                    nxt, d = prop[bid]
                    move_beam(bid, nxt, d)
                continue

            split_idx = int(kind.split(":")[1])
            old_a = beams[a].color
            old_b = beams[b].color
            changed_b = recolor_suffix(b, split_idx, old_a)
            changed_a = assign_color(a, old_b)
            stats["swap_events"] += 1
            stats["segment_swaps"] += 1
            if changed_a or changed_b:
                stats["visible_swap_events"] += 1
            nxt, d = prop[a]
            move_beam(a, nxt, d)
            continue

        for bid in sorted(prop):
            nxt, d = prop[bid]
            assert cell_owner.get(nxt) is None, "expected collision-free target cell"
            move_beam(bid, nxt, d)
    else:
        raise AssertionError("beam simulation did not converge")

    if collect_stats:
        stats["foreign_color_cells"] = sum(
            1 for cell, bid in cell_owner.items() if int(out[cell[0], cell[1]]) != beams[bid].source_color
        )
        stats["changed_beam_count"] = len(ever_changed)
        stats["max_beam_swaps"] = max((beam.swap_count for beam in beams.values()), default=0)
        stats["multi_swap_beams"] = sum(1 for beam in beams.values() if beam.swap_count >= 2)

    return out, stats


def main(grid: np.ndarray) -> np.ndarray:
    """
    Emit diagonal beams from framed 2x2 blocks; crossings preserve geometry and only transfer color.

    Input structure:
    - The most common color is the background, and each monochrome 2x2 block sits inside the same 4x4 frame color.
    - Each legal frame has exactly one structural corner gap, so every emitted beam starts on a frame corner and moves diagonally outward.

    Rule:
    - Every corner gap launches one diagonal beam using its block's color, and the beam keeps that diagonal direction until it leaves the grid.
    - When two moving beams cross, they pass through the crossing square without turning and exchange colors; each continuation paints with the newly inherited color.
    - When a moving beam later crosses an already-painted diagonal segment, the moving beam inherits that segment's current color and the older segment is recolored from the crossing onward with the moving beam's previous color.

    Key insight: Collision geometry is completely inert in this combo; solving the task means following a chain of color hand-offs across successive diagonal crossings, including recoloring of older suffixes.
    """
    out, _stats = _simulate(np.asarray(grid, dtype=int), strict_same_cell=True, collect_stats=False)
    assert out is not None, "expected a concrete output grid"
    return out


# --------------- generator (o5x) ---------------

BLOCK_COUNT_RANGE = (3, 6)
GRID_SIZE_MAX = 31
GRID_SIZE_MIN_BY_BLOCKS = {3: 14, 4: 18, 5: 22}
BG_CHOICES = [0, 8]
BG_WEIGHTS = [0.25, 0.75]
FRAME_COLOR_FOR_BG = {8: 3, 0: 8}
PALETTE_FOR_BG = {
    8: [1, 2, 4, 6, 7, 9],
    0: [1, 2, 3, 4, 6, 7, 9],
}
FRAME_BBOX_GAP = 2
PLACEMENT_MARGIN = 2
MIN_RAY_STEPS = 3
MIN_BEAM_PIXELS_PER_GAP = 4
MIN_SWAP_EVENTS = 2
MIN_VISIBLE_SWAP_EVENTS = 2
MIN_FOREIGN_COLOR_CELLS = 8
GEN_RETRIES = 500


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


def _rays_clear(
    grid: np.ndarray,
    bg: int,
    seeds: list[tuple[tuple[int, int], tuple[int, int]]],
) -> bool:
    h, w = grid.shape
    for (r, c), (dr, dc) in seeds:
        steps = 0
        nr, nc = r + dr, c + dc
        while 0 <= nr < h and 0 <= nc < w:
            steps += 1
            if int(grid[nr, nc]) != bg:
                return False
            nr += dr
            nc += dc
        if steps < MIN_RAY_STEPS:
            return False
    return True


def _available_corner_gaps(obstacles: np.ndarray, bg: int, r0: int, c0: int) -> list[tuple[int, int]]:
    gaps: list[tuple[int, int]] = []
    for gap in _CORNER_GAPS:
        seed = (r0 + gap[0], c0 + gap[1])
        if _rays_clear(obstacles, bg, [(seed, _gap_dir(gap))]):
            gaps.append(gap)
    return gaps


def _pick_colors(rng: np.random.Generator, block_count: int) -> tuple[int, int, list[int]]:
    bg = int(rng.choice(BG_CHOICES, p=BG_WEIGHTS))
    frame_color = FRAME_COLOR_FOR_BG[bg]
    palette = [c for c in PALETTE_FOR_BG[bg] if c != frame_color]
    block_colors = list(rng.choice(palette, size=block_count, replace=False))
    return bg, frame_color, block_colors


def _grid_size_min(block_count: int) -> int:
    result = 8
    for threshold, min_n in sorted(GRID_SIZE_MIN_BY_BLOCKS.items()):
        if block_count >= threshold:
            result = min_n
    return result


def _gen_attempt() -> np.ndarray | None:
    rng = np.random.default_rng()

    block_count = int(rng.integers(*BLOCK_COUNT_RANGE))
    min_n = _grid_size_min(block_count)
    max_n = min(GRID_SIZE_MAX, min_n + 8)
    n = int(rng.integers(min_n, max_n))
    bg, frame_color, block_colors = _pick_colors(rng, block_count)

    positions = _choose_positions(n, block_count, rng)
    if positions is None:
        return None

    obstacles = np.full((n, n), bg, dtype=int)
    for (r0, c0), color in zip(positions, block_colors):
        _draw_framed_block(obstacles, r0, c0, bg, frame_color, int(color), ())

    options: list[list[tuple[int, int]]] = []
    for r0, c0 in positions:
        available = _available_corner_gaps(obstacles, bg, r0, c0)
        if not available:
            return None
        options.append(available)

    combos = list(itertools.product(*options))
    if not combos:
        return None

    for idx in rng.permutation(len(combos)):
        chosen = combos[int(idx)]
        grid = np.full((n, n), bg, dtype=int)
        for (r0, c0), color, gap in zip(positions, block_colors, chosen):
            _draw_framed_block(grid, r0, c0, bg, frame_color, int(color), (gap,))

        if int(np.bincount(grid.ravel(), minlength=10).argmax()) != bg:
            continue

        try:
            out, stats = _simulate(grid, strict_same_cell=False, collect_stats=True)
        except Exception:
            continue
        if out is None:
            continue
        if stats["same_cell_collisions"] != 0:
            continue
        if stats["swap_events"] < MIN_SWAP_EVENTS:
            continue
        if stats["visible_swap_events"] < MIN_VISIBLE_SWAP_EVENTS:
            continue
        if stats["multi_swap_beams"] < 1:
            continue
        if stats["foreign_color_cells"] < MIN_FOREIGN_COLOR_CELLS:
            continue
        if np.count_nonzero(out != grid) < block_count * MIN_BEAM_PIXELS_PER_GAP:
            continue
        return grid

    return None


def generate_input() -> np.ndarray:
    for _ in range(GEN_RETRIES):
        grid = _gen_attempt()
        if grid is not None:
            return grid
    raise RuntimeError("failed to generate a valid e12f9a14 combo input")
