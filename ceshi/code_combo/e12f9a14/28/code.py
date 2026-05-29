# Combo variant of e12f9a14 (index 28)
# combo_spec: D2/0+D14/2
# Name: 迭代碰撞+碰撞即停

from __future__ import annotations

import importlib.util
import sys
from pathlib import Path

import numpy as np

_SOLVER_DIR = Path("/home/lixinyu/snap/new/solvers")
if str(_SOLVER_DIR) not in sys.path:
    sys.path.append(str(_SOLVER_DIR))

from arc_utils import get_bg, paste


# Design analysis for the combo variant
# a) Concrete behavior and trace:
#    The visible objects stay the same as the base task: 4x4 frames surround monochrome 2x2 blocks, and
#    each background-colored frame gap is a beam seed. The fusion changes only the temporal logic of beam
#    growth. Take three gaps in reading order A, B, C:
#    - A is a south gap, so it draws a full downward vertical beam first.
#    - B is a west gap to the right of A, so it launches second and draws leftward until its next cell is
#      already occupied by A; B then stops immediately, with no bend, no gap, and no history rewrite.
#    - C is a north gap below the partial B line. Under this combo, C launches only after B has finished,
#      so C stops against B. In the original task, B would have redirected at A. In the plain D2 stop-only
#      variant, B and C would still advance in lockstep, so C could pass through before B reached that cell.
#      The combo therefore creates a layered obstacle chain: A blocks B, then B blocks C.
# b) Complexity calibration:
#    If we only removed turning but kept lockstep advancement, the D14 change would mostly disappear. If we
#    only changed launch order without making collisions terminal, the output would look closer to D14 alone.
#    To keep both dimensions visible, the generator always creates at least one 3-beam chain where a later
#    beam is blocked by an earlier beam that was itself truncated by an even earlier beam. This gives a small
#    number of repeated, human-visible cases instead of many low-probability collision geometries.
# c) Hidden assumptions audited from the original main():
#    - The base solver assumed collisions may rewrite history, so it kept truncate/unpaint/blocked logic.
#      That assumption is invalid here: once a beam paints a cell, that cell stays painted forever.
#    - The base solver also assumed collisions create new directions or zigzag patterns. Here every beam has
#      one immutable direction, so pattern/phase bookkeeping and 45°/90° redirection branches must be removed.
#    - The D14 single-variant assumed a collision-created group gets priority to move again. In this combo,
#      there is no post-collision moving group at all; the iterative effect comes from whole completed beams
#      becoming static obstacles for later launches.
#    - A new assumption appears: launch order must be human-visible. The solver therefore sorts seeds by their
#      gap coordinates in row-major order instead of by hidden color ids or RNG state.
# d) Placement legality:
#    The generator uses a structured 3-frame layout with the original 4x4-frame geometry and >=2-cell frame
#    bbox clearance. Parameters are chosen so all initial beam corridors are background-only, the first beam
#    can reach the second beam's row, the second beam can reach toward the first, and the third beam can aim
#    at a segment that exists only because the second beam finished earlier.
# e) Visual observability:
#    A human can read the order directly from geometry: the highest gap fires first, then the next lower gap,
#    then the lowest one. The outputs repeatedly show "straight line stopped by earlier straight line" rather
#    than ambiguous hidden priorities. Samples are rejected unless the combo output differs from both the base
#    solver and a lockstep stop-only baseline, so the fusion behavior is genuinely visible.
# f) Visual cleanliness:
#    Each core frame has only one gap. The beams meet in open space rather than inside dense frame clusters,
#    which keeps the layered blocking pattern legible. Randomness is limited to positions, distances, colors,
#    and which of two nearby columns the third beam targets, so the grids vary without becoming cluttered.
# g) Inference burden:
#    No new color mapping is introduced. Beam colors still match their inner 2x2 blocks, so the only concept
#    the solver must infer is time order: earlier gaps lay down permanent straight tracks that later gaps may
#    run into and stop against.


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


def _gap_dir(gap: tuple[int, int]) -> tuple[int, int]:
    gr, gc = gap
    dr = -1 if gr == 0 else (1 if gr == 3 else 0)
    dc = -1 if gc == 0 else (1 if gc == 3 else 0)
    return dr, dc


def _frames_separated(a: tuple[int, int], b: tuple[int, int], gap: int) -> bool:
    ar0, ac0 = a
    br0, bc0 = b
    ar1, ac1 = ar0 + 3, ac0 + 3
    br1, bc1 = br0 + 3, bc0 + 3
    return ar1 + gap < br0 or br1 + gap < ar0 or ac1 + gap < bc0 or bc1 + gap < ac0


def _diag_segment_owner(
    cell_owner: dict[tuple[int, int], int],
    cell_enter: dict[tuple[int, int], tuple[int, int]],
    cell_index: dict[tuple[int, int], int],
    p: tuple[int, int],
    q: tuple[int, int],
) -> int | None:
    dr, dc = q[0] - p[0], q[1] - p[1]
    if abs(dr) != 1 or abs(dc) != 1:
        return None
    oq = cell_owner.get(q)
    if oq is not None and cell_index.get(q, 0) > 0 and cell_enter.get(q) == (dr, dc):
        return oq
    op = cell_owner.get(p)
    if op is not None and cell_index.get(p, 0) > 0 and cell_enter.get(p) == (-dr, -dc):
        return op
    return None


class _Beam:
    __slots__ = ("bid", "color", "seed", "pos", "direction", "path", "active", "stop_depth", "stopped_by")

    def __init__(self, bid: int, color: int, pos: tuple[int, int], direction: tuple[int, int]):
        self.bid = bid
        self.color = int(color)
        self.seed = pos
        self.pos = pos
        self.direction = direction
        self.path = [pos]
        self.active = True
        self.stop_depth = 0
        self.stopped_by: int | None = None


def _extract_seeds(g: np.ndarray, bg: int) -> list[tuple[tuple[int, int], int, tuple[int, int]]]:
    seeds: list[tuple[tuple[int, int], int, tuple[int, int]]] = []
    for col, br, bc in _find_blocks_2x2(g, bg):
        frame_col = _infer_frame_color(g, bg, br, bc, col)
        for gr, gc in _frame_gaps_for_block(g, bg, frame_col, br, bc):
            dr = -1 if gr == br - 1 else (1 if gr == br + 2 else 0)
            dc = -1 if gc == bc - 1 else (1 if gc == bc + 2 else 0)
            direction = (dr, dc)
            assert direction != (0, 0), "expected each frame gap to face outward"
            seeds.append(((gr, gc), int(col), direction))
    seeds.sort(key=lambda item: item[0])
    assert seeds, "expected at least one beam spawned from frame gaps"
    return seeds


def _seed_beams(
    g: np.ndarray,
    out: np.ndarray,
    seeds: list[tuple[tuple[int, int], int, tuple[int, int]]],
) -> tuple[
    dict[int, _Beam],
    dict[tuple[int, int], int],
    dict[tuple[int, int], tuple[int, int]],
    dict[tuple[int, int], int],
]:
    beams: dict[int, _Beam] = {}
    cell_owner: dict[tuple[int, int], int] = {}
    cell_enter: dict[tuple[int, int], tuple[int, int]] = {}
    cell_index: dict[tuple[int, int], int] = {}

    for bid, (pos, color, direction) in enumerate(seeds):
        assert int(out[pos[0], pos[1]]) == int(g[pos[0], pos[1]]), "output should start as a copy of the input"
        assert pos not in cell_owner, "expected unique beam seed positions"
        beam = _Beam(bid, color, pos, direction)
        beams[bid] = beam
        out[pos[0], pos[1]] = color
        cell_owner[pos] = bid
        cell_enter[pos] = direction
        cell_index[pos] = 0
    return beams, cell_owner, cell_enter, cell_index


def _simulate_sequential(grid: np.ndarray, *, return_stats: bool = False):
    g = np.asarray(grid, dtype=int)
    h, w = g.shape
    bg = get_bg(g)
    out = g.copy()
    seeds = _extract_seeds(g, bg)
    beams, cell_owner, cell_enter, cell_index = _seed_beams(g, out, seeds)

    stats = {
        "beam_count": len(beams),
        "collision_stops": 0,
        "collision_cell_stops": 0,
        "collision_segment_stops": 0,
        "edge_stops": 0,
        "frame_stops": 0,
        "move_steps": 0,
        "max_stop_depth": 0,
        "layered_stops": 0,
    }

    def in_bounds(cell: tuple[int, int]) -> bool:
        r, c = cell
        return 0 <= r < h and 0 <= c < w

    def paint_cell(bid: int, cell: tuple[int, int], enter_dir: tuple[int, int]) -> None:
        beam = beams[bid]
        beam.path.append(cell)
        beam.pos = cell
        out[cell[0], cell[1]] = beam.color
        cell_owner[cell] = bid
        cell_enter[cell] = enter_dir
        cell_index[cell] = len(beam.path) - 1

    for bid in sorted(beams):
        beam = beams[bid]
        while True:
            dr, dc = beam.direction
            nxt = (beam.pos[0] + dr, beam.pos[1] + dc)

            if not in_bounds(nxt):
                beam.active = False
                stats["edge_stops"] += 1
                break
            if int(g[nxt[0], nxt[1]]) != bg:
                beam.active = False
                stats["frame_stops"] += 1
                break

            owner = cell_owner.get(nxt)
            if owner is not None and owner != bid:
                beam.active = False
                beam.stopped_by = owner
                beam.stop_depth = beams[owner].stop_depth + 1
                stats["collision_stops"] += 1
                stats["collision_cell_stops"] += 1
                stats["max_stop_depth"] = max(stats["max_stop_depth"], beam.stop_depth)
                if beams[owner].stop_depth > 0:
                    stats["layered_stops"] += 1
                break

            if abs(dr) + abs(dc) == 2:
                p2 = (beam.pos[0], beam.pos[1] + dc)
                n2 = (beam.pos[0] + dr, beam.pos[1])
                owner = _diag_segment_owner(cell_owner, cell_enter, cell_index, p2, n2)
                if owner is not None and owner != bid:
                    beam.active = False
                    beam.stopped_by = owner
                    beam.stop_depth = beams[owner].stop_depth + 1
                    stats["collision_stops"] += 1
                    stats["collision_segment_stops"] += 1
                    stats["max_stop_depth"] = max(stats["max_stop_depth"], beam.stop_depth)
                    if beams[owner].stop_depth > 0:
                        stats["layered_stops"] += 1
                    break

            paint_cell(bid, nxt, (dr, dc))
            stats["move_steps"] += 1

    stats["painted_cells"] = int(np.count_nonzero(out != g))
    if return_stats:
        return out, stats
    return out


def _simulate_lockstep_stop(grid: np.ndarray) -> np.ndarray:
    g = np.asarray(grid, dtype=int)
    h, w = g.shape
    bg = get_bg(g)
    out = g.copy()
    seeds = _extract_seeds(g, bg)
    beams, cell_owner, cell_enter, cell_index = _seed_beams(g, out, seeds)

    def in_bounds(cell: tuple[int, int]) -> bool:
        r, c = cell
        return 0 <= r < h and 0 <= c < w

    def paint_cell(bid: int, cell: tuple[int, int], enter_dir: tuple[int, int]) -> None:
        beam = beams[bid]
        beam.path.append(cell)
        beam.pos = cell
        out[cell[0], cell[1]] = beam.color
        cell_owner[cell] = bid
        cell_enter[cell] = enter_dir
        cell_index[cell] = len(beam.path) - 1

    max_iters = h * w + len(beams)
    for _ in range(max_iters):
        active_any = False
        moved_any = False
        for bid in sorted(beams):
            beam = beams[bid]
            if not beam.active:
                continue
            active_any = True
            dr, dc = beam.direction
            nxt = (beam.pos[0] + dr, beam.pos[1] + dc)
            if not in_bounds(nxt):
                beam.active = False
                continue
            if int(g[nxt[0], nxt[1]]) != bg:
                beam.active = False
                continue
            if nxt in cell_owner:
                beam.active = False
                continue
            if abs(dr) + abs(dc) == 2:
                p2 = (beam.pos[0], beam.pos[1] + dc)
                n2 = (beam.pos[0] + dr, beam.pos[1])
                owner = _diag_segment_owner(cell_owner, cell_enter, cell_index, p2, n2)
                if owner is not None and owner != bid:
                    beam.active = False
                    continue
            paint_cell(bid, nxt, (dr, dc))
            moved_any = True
        if not active_any or not moved_any:
            break
    else:
        raise AssertionError("lockstep stop simulation did not converge")
    return out


_ORIGINAL_MAIN = None


def _load_original_main():
    global _ORIGINAL_MAIN
    if _ORIGINAL_MAIN is not None:
        return _ORIGINAL_MAIN
    path = Path("/home/lixinyu/snap/new/arc2_solver_select/e12f9a14.py")
    if not path.exists():
        _ORIGINAL_MAIN = False
        return None
    spec = importlib.util.spec_from_file_location("_orig_e12f9a14", path)
    assert spec is not None and spec.loader is not None
    mod = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(mod)
    _ORIGINAL_MAIN = mod.main
    return _ORIGINAL_MAIN


def main(grid: np.ndarray) -> np.ndarray:
    """
    Launch straight rays from framed 2x2 blocks one by one, so earlier rays become permanent obstacles.

    Input structure:
    - The most common color is the background, and each colored 2x2 block is enclosed by a shared-color 4x4 frame.
    - Background-colored holes on a frame border are beam exits; corner exits launch diagonals and side exits launch cardinals.

    Rule:
    - Sort all exits by their grid coordinates in row-major order and launch them in that visible order.
    - Each launched beam paints its seed and then keeps moving in its one fixed outward direction until it reaches the grid edge,
      a non-background frame/block cell, an already painted beam cell, or an already painted diagonal segment.
    - On any collision, the moving beam stops immediately. There is no turning, no zigzag merge, and no erasing of earlier paint.

    Key insight:
    - The D14 effect survives here as temporal layering rather than repeated redirection: a beam that was drawn earlier can shorten a
      later beam, and that shortened later beam can itself become the obstacle that stops an even later beam.
    """
    return _simulate_sequential(grid, return_stats=False)


def collect_combo_metrics(grid: np.ndarray) -> dict[str, int]:
    out, stats = _simulate_sequential(grid, return_stats=True)
    lockstep = _simulate_lockstep_stop(grid)
    stats = dict(stats)
    stats["lockstep_diff_cells"] = int(np.count_nonzero(out != lockstep))
    orig_main = _load_original_main()
    if orig_main is not None:
        try:
            orig_out = orig_main(grid)
        except Exception:
            stats["original_diff_cells"] = -1
        else:
            stats["original_diff_cells"] = int(np.count_nonzero(out != orig_out))
    else:
        stats["original_diff_cells"] = -1
    return stats


# --------------- generator ---------------

GRID_SIZE_MIN = 20
GRID_SIZE_MAX = 31
GEN_RETRIES = 200
BG_CHOICES = [0, 8]
BG_WEIGHTS = [0.25, 0.75]
FRAME_COLOR_FOR_BG = {8: 3, 0: 8}
PALETTE_FOR_BG = {
    8: [1, 2, 4, 6, 7, 9],
    0: [1, 2, 3, 4, 6, 7, 9],
}
FRAME_BBOX_GAP = 2
MIN_BEAM_PIXELS = 16

_CORE_GAPS = {
    "A": ((3, 1),),  # south
    "B": ((2, 0),),  # west
    "C": ((0, 1),),  # north
}


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


def _pick_colors(rng: np.random.Generator) -> tuple[int, int, list[int]]:
    bg = int(rng.choice(BG_CHOICES, p=BG_WEIGHTS))
    frame_color = FRAME_COLOR_FOR_BG[bg]
    palette = [c for c in PALETTE_FOR_BG[bg] if c != frame_color]
    block_colors = list(map(int, rng.choice(palette, size=3, replace=False)))
    return bg, frame_color, block_colors


def _choose_core_layout(n: int, rng: np.random.Generator):
    for _ in range(80):
        v_span = int(rng.integers(8, 12))
        h_span = int(rng.integers(9, 14))
        hit_offset = int(rng.choice([3, 4]))
        c_gap = int(rng.integers(4, 7))

        max_ra = n - (v_span + c_gap + 4)
        max_ca = n - (h_span + 4)
        if max_ra < 2 or max_ca < 2:
            continue

        ra = int(rng.integers(2, max_ra + 1))
        ca = int(rng.integers(2, max_ca + 1))

        pos_a = (ra, ca)
        pos_b = (ra + v_span - 2, ca + h_span)
        pos_c = (ra + v_span + c_gap, ca + hit_offset - 1)

        if pos_b[0] + 3 >= n or pos_b[1] + 3 >= n:
            continue
        if pos_c[0] + 3 >= n or pos_c[1] + 3 >= n:
            continue
        if not _frames_separated(pos_a, pos_b, FRAME_BBOX_GAP):
            continue
        if not _frames_separated(pos_a, pos_c, FRAME_BBOX_GAP):
            continue
        if not _frames_separated(pos_b, pos_c, FRAME_BBOX_GAP):
            continue

        meet_row = pos_b[0] + 2
        a_col = pos_a[1] + 1
        c_col = pos_c[1] + 1
        b_col = pos_b[1]
        if not (a_col + 1 < c_col < b_col):
            continue
        if c_gap >= b_col - c_col:
            continue

        return [
            (pos_a, _CORE_GAPS["A"]),
            (pos_b, _CORE_GAPS["B"]),
            (pos_c, _CORE_GAPS["C"]),
        ]
    return None


def _gen_attempt() -> np.ndarray | None:
    rng = np.random.default_rng()
    n = int(rng.integers(GRID_SIZE_MIN, GRID_SIZE_MAX))
    bg, frame_color, block_colors = _pick_colors(rng)
    layout = _choose_core_layout(n, rng)
    if layout is None:
        return None

    grid = np.full((n, n), bg, dtype=int)
    for (r0, c0), gaps, color in zip((item[0] for item in layout), (item[1] for item in layout), block_colors):
        _draw_framed_block(grid, r0, c0, bg, frame_color, color, gaps)

    if int(np.bincount(grid.ravel(), minlength=10).argmax()) != bg:
        return None

    try:
        out, stats = _simulate_sequential(grid, return_stats=True)
    except Exception:
        return None

    if stats["beam_count"] != 3:
        return None
    if stats["collision_stops"] < 2:
        return None
    if stats["layered_stops"] < 1:
        return None
    if stats["max_stop_depth"] < 2:
        return None
    if stats["painted_cells"] < MIN_BEAM_PIXELS:
        return None

    lockstep = _simulate_lockstep_stop(grid)
    if np.array_equal(out, lockstep):
        return None

    return grid


def generate_input() -> np.ndarray:
    for _ in range(GEN_RETRIES):
        grid = _gen_attempt()
        if grid is not None:
            return grid
    raise RuntimeError("failed to generate a valid combo e12f9a14 input")
