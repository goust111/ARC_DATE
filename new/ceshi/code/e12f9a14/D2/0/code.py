# Variant of e12f9a14
# Dimension: D2 — 碰撞检测与轨迹变化规则 (trajectory)
# Change: collision_detection——90度交角碰撞时两条射线各转45度并在交叉点留1像素间隙、以单像素绘制；45度... → 碰撞即停止

from __future__ import annotations

import itertools
import sys
from pathlib import Path

import numpy as np

_SOLVER_DIR = Path("/home/lixinyu/snap/new/solvers")
if str(_SOLVER_DIR) not in sys.path:
    sys.path.append(str(_SOLVER_DIR))

from arc_utils import get_bg, paste


# Design analysis for the D2 variant
# a) Concrete behavior:
#    Input objects stay the same: monochrome 2x2 blocks sit inside 4x4 frames, and each bg-colored
#    frame gap emits one straight outward ray. The visual change is entirely in collisions. Originally,
#    the first collision could create a gap, a redirected pair of rays, or a zigzag band. In the variant,
#    no new direction is ever created: rays keep their initial direction forever, and the first ray that
#    reaches a contested place keeps painting while the later ray stops immediately. Example: if a top
#    gap at (4,10) shoots downward and a left gap at (8,6) shoots rightward, then the original solver
#    would redirect or merge them at the meeting point; the variant instead lets the gap with the smaller
#    (row,col) seed continue and truncates the other one exactly before the occupied cell/segment.
# b) Complexity calibration:
#    The rule is simpler than the original collision algebra, so the generator must compensate by making
#    collisions common and readable. To keep the variant observable, generated grids use at least two
#    framed blocks, prefer 2-4 gaps per frame, and reject samples with zero collision stops. The ordering
#    key is visible: gaps are processed in row-major order of their locations, not by color ids or hidden
#    randomness.
# c) Hidden assumptions audited in the original main():
#    - _dir_diff / 45-degree / 90-degree branching assumed every collision would redirect; this is no
#      longer valid, so the entire event-resolution pipeline is removed.
#    - truncate_beam/unpaint_cell/blocked assumed cells may need to be erased after a later collision.
#      That was harmless before because collisions could revise history; in the variant history is never
#      rewritten, so erasing would be wrong.
#    - find_partner and pattern/phase logic assumed two beams can become synchronized after merging.
#      The variant forbids turning/merging, so each beam now has one immutable direction.
#    - cross-pair resolution previously depended on global event sorting plus geometric redirection.
#      Here the only safe assumption is sequential claiming: earlier beams paint first and later beams
#      test against the already-painted state, including segments drawn earlier in the same tick.
# d) Placement legality:
#    Frames remain separated by the original bbox gap and placed away from the border, so 4x4 frames do
#    not overlap and every detected 2x2 block still has a unique surrounding frame color. Gap choices are
#    still validated against an obstacle grid with full frames to ensure rays never hit a frame/block cell.
#    The generator additionally validates every candidate by running the new simulator.
# e) Visual observability:
#    Humans can see the ordering because gap positions themselves define it: upper gaps and then leftmost
#    gaps win ties. Training examples therefore show straight rays that terminate early at another straight
#    ray instead of bending or thickening. Samples with no collision are rejected because they would look
#    too close to a trivial no-change straight-ray task.
# f) Visual cleanliness:
#    The output contains only the original frames plus thin straight rays. Since later beams stop instead
#    of piling multiple redirected bands into the same region, the variant is visually cleaner than the
#    source task. The generator still enforces minimum runway and frame spacing so rays remain readable.
# g) Color reasoning:
#    No color remapping is introduced. Each ray keeps the inner block's color, matching the original task,
#    so the only new inference burden is geometric ordering and stop points rather than any hidden palette
#    rule.


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


class _Beam:
    __slots__ = ("bid", "color", "pos", "direction", "path", "enter_dirs", "active", "seed")

    def __init__(self, bid: int, color: int, pos: tuple[int, int], direction: tuple[int, int]):
        self.bid = bid
        self.color = int(color)
        self.pos = pos
        self.direction = direction
        self.path = [pos]
        self.enter_dirs = [direction]
        self.active = True
        self.seed = pos


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


def _simulate(grid: np.ndarray, *, return_stats: bool = False):
    g = np.asarray(grid, dtype=int)
    h, w = g.shape
    bg = get_bg(g)

    out = g.copy()
    blocks = _find_blocks_2x2(g, bg)
    assert blocks, "expected at least one 2x2 block in input"

    seeds: list[tuple[tuple[int, int], int, tuple[int, int]]] = []
    for col, br, bc in blocks:
        frame_col = _infer_frame_color(g, bg, br, bc, col)
        for gr, gc in _frame_gaps_for_block(g, bg, frame_col, br, bc):
            dr = -1 if gr == br - 1 else (1 if gr == br + 2 else 0)
            dc = -1 if gc == bc - 1 else (1 if gc == bc + 2 else 0)
            direction = (dr, dc)
            assert direction != (0, 0), "expected each frame gap to face outward"
            seeds.append(((gr, gc), col, direction))

    assert seeds, "expected at least one beam spawned from frame gaps"
    seeds.sort(key=lambda item: item[0])

    beams: dict[int, _Beam] = {}
    cell_owner: dict[tuple[int, int], int] = {}
    cell_enter: dict[tuple[int, int], tuple[int, int]] = {}
    cell_index: dict[tuple[int, int], int] = {}

    def paint_cell(bid: int, cell: tuple[int, int], enter_dir: tuple[int, int]) -> None:
        r, c = cell
        out[r, c] = beams[bid].color
        cell_owner[cell] = bid
        cell_enter[cell] = enter_dir
        cell_index[cell] = len(beams[bid].path) - 1

    for bid, (pos, color, direction) in enumerate(seeds):
        assert int(out[pos[0], pos[1]]) == bg, "expected beam seed at a frame gap"
        assert pos not in cell_owner, "expected unique beam seed positions"
        beams[bid] = _Beam(bid, color, pos, direction)
        paint_cell(bid, pos, direction)

    stats = {
        "beam_count": len(beams),
        "collision_stops": 0,
        "collision_cell_stops": 0,
        "collision_segment_stops": 0,
        "edge_stops": 0,
        "frame_stops": 0,
        "move_steps": 0,
    }

    def in_bounds(cell: tuple[int, int]) -> bool:
        r, c = cell
        return 0 <= r < h and 0 <= c < w

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
                stats["edge_stops"] += 1
                continue
            if int(g[nxt[0], nxt[1]]) != bg:
                beam.active = False
                stats["frame_stops"] += 1
                continue
            if nxt in cell_owner:
                beam.active = False
                stats["collision_stops"] += 1
                stats["collision_cell_stops"] += 1
                continue
            if abs(dr) + abs(dc) == 2:
                p2 = (beam.pos[0], beam.pos[1] + dc)
                n2 = (beam.pos[0] + dr, beam.pos[1])
                owner = _diag_segment_owner(cell_owner, cell_enter, cell_index, p2, n2)
                if owner is not None and owner != bid:
                    beam.active = False
                    stats["collision_stops"] += 1
                    stats["collision_segment_stops"] += 1
                    continue

            beam.path.append(nxt)
            beam.enter_dirs.append((dr, dc))
            beam.pos = nxt
            paint_cell(bid, nxt, (dr, dc))
            stats["move_steps"] += 1
            moved_any = True

        if not active_any or not moved_any:
            break
    else:
        raise AssertionError("beam simulation did not converge")

    stats["painted_cells"] = int(np.count_nonzero(out != g))
    if return_stats:
        return out, stats
    return out


def main(grid: np.ndarray) -> np.ndarray:
    """
    Extend straight rays from framed 2x2 blocks, with collisions resolved by stopping.

    Input structure:
    - The most common color is the background, while one shared frame color outlines 4x4 borders around
      several monochrome 2x2 inner blocks.
    - Each frame has one or more border gaps; every gap is itself background-colored and marks a ray seed.

    Rule:
    - A gap emits one ray in the outward direction indicated by its side or corner position on the frame.
    - Rays keep that initial straight direction forever and paint background cells until they leave the
      grid, hit a frame/block cell, or meet another already-painted ray.
    - When two rays would contend for the same cell or a diagonal crossing segment, the ray whose gap was
      encountered earlier in row-major order has already been painted, so the later ray simply stops.

    Key insight:
    - There is no post-collision redirection anymore; the only nontrivial inference is the global drawing
      priority of the gaps, because that priority determines which straight ray survives at each collision.
    """
    return _simulate(grid, return_stats=False)


# --------------- generator (o5x) ---------------

# ---- CFG: tunable generation parameters ----
BLOCK_COUNT_RANGE = (2, 6)              # uniform draw [lo, hi); >=2 to keep collisions observable
GRID_SIZE_MAX = 31                      # exclusive upper bound for square grid side
GRID_SIZE_MIN_BY_BLOCKS = {2: 14, 5: 20}
BG_CHOICES = [0, 8]
BG_WEIGHTS = [0.25, 0.75]
FRAME_COLOR_FOR_BG = {8: 3, 0: 8}
PALETTE_FOR_BG = {
    8: [1, 2, 4, 6, 7, 9],
    0: [1, 2, 3, 4, 6, 7, 9],
}
FRAME_BBOX_GAP = 2
PLACEMENT_MARGIN = 2
MIN_RAY_STEPS = 2
MIN_BEAM_PIXELS_PER_GAP = 2
MIN_TOTAL_GAPS = 3
MIN_COLLISION_STOPS = 1
GEN_RETRIES = 400
GAP_ASSIGNMENT_TRIES = 80
GAP_COUNT_WEIGHTS = {1: 0.05, 2: 0.35, 3: 0.35, 4: 0.25}

_BORDER_ORDER = (
    (0, 0), (0, 1), (0, 2), (0, 3),
    (1, 3), (2, 3), (3, 3), (3, 2),
    (3, 1), (3, 0), (2, 0), (1, 0),
)
_BORDER_INDEX = {p: i for i, p in enumerate(_BORDER_ORDER)}


def _gap_combos(size: int) -> list[tuple[tuple[int, int], ...]]:
    combos = []
    for combo in itertools.combinations(_BORDER_ORDER, size):
        idxs = sorted(_BORDER_INDEX[p] for p in combo)
        if any((idxs[(i + 1) % size] - idxs[i]) % 12 == 1 for i in range(size)):
            continue
        if size >= 3 and (3, 3) in combo and (0, 2) in combo:
            continue
        combos.append(combo)
    return combos


_GAP_COMBOS = {n: _gap_combos(n) for n in range(1, 5)}


def _gap_dir(gap: tuple[int, int]) -> tuple[int, int]:
    gr, gc = gap
    dr = -1 if gr == 0 else (1 if gr == 3 else 0)
    dc = -1 if gc == 0 else (1 if gc == 3 else 0)
    return dr, dc


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


def _clear_gap_combos(
    obstacles: np.ndarray,
    bg: int,
    r0: int,
    c0: int,
) -> dict[int, list[tuple[tuple[int, int], ...]]]:
    clear: dict[int, list[tuple[tuple[int, int], ...]]] = {}
    for gap_count, combos in _GAP_COMBOS.items():
        for combo in combos:
            seeds = [((r0 + gr, c0 + gc), _gap_dir((gr, gc))) for gr, gc in combo]
            if _rays_clear(obstacles, bg, seeds):
                clear.setdefault(gap_count, []).append(combo)
    return clear


def _pick_colors(rng: np.random.Generator, block_count: int) -> tuple[int, int, list[int]]:
    bg = int(rng.choice(BG_CHOICES, p=BG_WEIGHTS))
    frame_color = FRAME_COLOR_FOR_BG[bg]
    palette = [c for c in PALETTE_FOR_BG[bg] if c != frame_color]
    block_colors = list(rng.choice(palette, size=block_count, replace=False))
    return bg, frame_color, block_colors


def _grid_size_min(block_count: int) -> int:
    result = 14
    for threshold, min_n in sorted(GRID_SIZE_MIN_BY_BLOCKS.items()):
        if block_count >= threshold:
            result = min_n
    return result


def _sample_gap_count(available: dict[int, list[tuple[tuple[int, int], ...]]], rng: np.random.Generator) -> int:
    counts = sorted(available)
    weights = np.array([GAP_COUNT_WEIGHTS[c] for c in counts], dtype=float)
    weights /= weights.sum()
    return int(rng.choice(counts, p=weights))


def _build_grid(
    n: int,
    bg: int,
    frame_color: int,
    positions: list[tuple[int, int]],
    block_colors: list[int],
    chosen_gaps: list[tuple[tuple[int, int], ...]],
) -> tuple[np.ndarray, int, list[tuple[tuple[int, int], tuple[int, int]]]]:
    grid = np.full((n, n), bg, dtype=int)
    gap_total = 0
    seeds: list[tuple[tuple[int, int], tuple[int, int]]] = []
    for (r0, c0), color, gaps in zip(positions, block_colors, chosen_gaps):
        gap_total += len(gaps)
        _draw_framed_block(grid, r0, c0, bg, frame_color, int(color), gaps)
        for gap in gaps:
            seeds.append(((r0 + gap[0], c0 + gap[1]), _gap_dir(gap)))
    return grid, gap_total, seeds


def _gen_attempt() -> np.ndarray | None:
    rng = np.random.default_rng()

    block_count = int(rng.integers(*BLOCK_COUNT_RANGE))
    n = int(rng.integers(_grid_size_min(block_count), GRID_SIZE_MAX))

    bg, frame_color, block_colors = _pick_colors(rng, block_count)
    positions = _choose_positions(n, block_count, rng)
    if positions is None:
        return None

    obstacles = np.full((n, n), bg, dtype=int)
    for (r0, c0), color in zip(positions, block_colors):
        _draw_framed_block(obstacles, r0, c0, bg, frame_color, int(color), ())

    all_available: list[dict[int, list[tuple[tuple[int, int], ...]]]] = []
    for r0, c0 in positions:
        available = _clear_gap_combos(obstacles, bg, r0, c0)
        if not available:
            return None
        all_available.append(available)

    for _ in range(GAP_ASSIGNMENT_TRIES):
        chosen_gaps: list[tuple[tuple[int, int], ...]] = []
        for available in all_available:
            gap_count = _sample_gap_count(available, rng)
            combos = available[gap_count]
            chosen_gaps.append(combos[int(rng.integers(len(combos)))])

        grid, gap_total, seeds = _build_grid(n, bg, frame_color, positions, block_colors, chosen_gaps)
        if gap_total < MIN_TOTAL_GAPS:
            continue
        if int(np.bincount(grid.ravel(), minlength=10).argmax()) != bg:
            continue
        if not _rays_clear(grid, bg, seeds):
            continue

        try:
            out, stats = _simulate(grid, return_stats=True)
        except Exception:
            continue

        if stats["collision_stops"] < MIN_COLLISION_STOPS:
            continue
        if stats["painted_cells"] < gap_total * MIN_BEAM_PIXELS_PER_GAP:
            continue
        if stats["beam_count"] < 3:
            continue
        if np.array_equal(out, grid):
            continue
        return grid
    return None


def generate_input() -> np.ndarray:
    for _ in range(GEN_RETRIES):
        grid = _gen_attempt()
        if grid is not None:
            return grid
    raise RuntimeError("failed to generate a valid e12f9a14 D2 variant input")
