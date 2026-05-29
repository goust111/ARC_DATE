# Variant of e12f9a14
# Dimension: D2 — 碰撞检测与轨迹变化规则 (trajectory)
# Change: collision_detection——90度交角碰撞时两条射线各转45度并在交叉点留1像素间隙、以单像素绘制；45度... → 碰撞时相互穿过并交换颜色

from __future__ import annotations

import itertools

import numpy as np

from arc_utils import get_bg, paste


# Design analysis for the D2 variant
# a) Concrete behavior and trace:
#    - We keep the same visible objects: monochrome 2x2 blocks inside a shared 4x4 frame color.
#      The crucial adaptation is that the generator only uses corner gaps, so every beam is diagonal.
#    - This makes the new rule visible on a single-layer raster: two diagonals can cross inside a 2x2
#      square without needing to share one cell. Before the crossing, each beam paints in its own color;
#      after the crossing, the two continuations keep their directions but exchange colors.
#    - Small trace: red NE beam and blue NW beam approach the same 2x2 square from below. In the old
#      task, they would redirect or merge depending on angle. In this variant, both still step into
#      their next diagonal cells, but the red continuation is painted blue and the blue continuation is
#      painted red. If a later beam crosses an already-drawn diagonal segment, the already-drawn suffix
#      of that older beam must be recolored from the crossing onward.
# b) Complexity calibration:
#    - Pure color exchange is easy to miss if many same-color beams interact, so the generator gives
#      each block a unique color and uses one gap per block.
#    - We still allow 2-4 framed blocks and both simultaneous diagonal crossings and later segment
#      crossings, which preserves nontrivial interaction without reintroducing the old turning logic.
# c) Implicit-assumption audit against the original main():
#    - Original code assumes collisions can be rendered by changing geometry (turning, zigzagging,
#      inserting blank gaps). That assumption no longer holds because the new state variable is color.
#    - Any code that truncates beams, blocks cells permanently, or rewrites direction on collision must
#      be removed. Instead we need persistent per-beam paths so a late crossing can recolor an already
#      drawn suffix.
#    - Single-cell pass-through is ambiguous on a one-layer pixel grid: two beams cannot both occupy the
#      same cell and still remain visually distinct. The new solver therefore treats same-cell collisions
#      as invalid for this variant, and the generator explicitly avoids them.
# d) Placement legality:
#    - Frames keep the original 4x4 geometry and non-overlap spacing.
#    - Candidate corner gaps are filtered so each beam has at least a short unobstructed diagonal runway.
#    - A candidate input is accepted only if the simulator reports zero same-cell collisions and at
#      least one color-swap event.
# e) Visual observability:
#    - The crossing rule is visible because the line geometry does not change; only the continuation
#      colors do. That creates a crisp “the colors trade places across the X” effect.
#    - Restricting to corner gaps avoids hidden bookkeeping with cardinal beams whose collisions would
#      otherwise need arbitrary tie-breaking inside a shared cell.
# f) Visual cleanliness:
#    - One beam per frame and diagonal-only motion keep the grids readable.
#    - The generator rejects layouts with ambiguous same-cell encounters, which are the main source of
#      unreadable overlap in this variant.
# g) Color-change inference:
#    - Every beam color comes from a unique source block, so when two continuations exchange colors the
#      source of each new color is visually traceable.
#    - There is no hidden dependence on color ids; only visible beam/block correspondences matter.
# New solver assumptions to review:
#    - All legal variant inputs use corner gaps only, so every beam is diagonal.
#    - The generator guarantees that collisions happen as geometric diagonal crossings, never as
#      same-cell meetings. The solver asserts that property in strict mode.


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
    """Find the unique non-bg color on the 4x4 border around a 2x2 block."""
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
    __slots__ = ("bid", "color", "pos", "direction", "path", "enter_dirs", "active")

    def __init__(self, bid: int, color: int, pos: tuple[int, int], direction: tuple[int, int]):
        self.bid = bid
        self.color = int(color)
        self.pos = pos
        self.direction = direction
        self.path = [pos]
        self.enter_dirs = [direction]
        self.active = True


def _simulate(grid: np.ndarray, *, strict_same_cell: bool) -> tuple[np.ndarray | None, dict[str, int]]:
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

    def paint_cell(bid: int, cell: tuple[int, int], enter_dir: tuple[int, int]) -> None:
        r, c = cell
        out[r, c] = beams[bid].color
        cell_owner[cell] = bid
        cell_enter[cell] = enter_dir
        cell_index[cell] = len(beams[bid].path) - 1

    def move_beam(bid: int, nxt: tuple[int, int], enter_dir: tuple[int, int]) -> None:
        assert cell_owner.get(nxt) is None, "expected an empty raster cell before painting"
        b = beams[bid]
        b.path.append(nxt)
        b.enter_dirs.append(enter_dir)
        b.pos = nxt
        paint_cell(bid, nxt, enter_dir)

    def recolor_suffix(bid: int, start_idx: int, new_color: int) -> None:
        b = beams[bid]
        for cell in b.path[start_idx:]:
            if cell_owner.get(cell) == bid:
                out[cell[0], cell[1]] = int(new_color)
        b.color = int(new_color)

    def spawn_beam(color: int, pos: tuple[int, int], direction: tuple[int, int]) -> None:
        nonlocal next_bid
        assert int(out[pos[0], pos[1]]) == bg, "expected beam seed at a frame gap"
        assert pos not in cell_owner, "expected unique beam seed positions"
        assert abs(direction[0]) + abs(direction[1]) == 2, "variant expects diagonal corner-gap beams"
        bid = next_bid
        next_bid += 1
        beams[bid] = _Beam(bid, color, pos, direction)
        paint_cell(bid, pos, direction)

    for col, br, bc in blocks:
        frame_col = _infer_frame_color(g, bg, br, bc, col)
        gaps = _frame_gaps_for_block(g, bg, frame_col, br, bc)
        assert gaps, "expected each frame to have at least one gap"
        for gr, gc in gaps:
            dr = -1 if gr == br - 1 else (1 if gr == br + 2 else 0)
            dc = -1 if gc == bc - 1 else (1 if gc == bc + 2 else 0)
            direction = _normalize_dir(dr, dc)
            spawn_beam(col, (gr, gc), direction)

    def in_bounds(cell: tuple[int, int]) -> bool:
        r, c = cell
        return 0 <= r < h and 0 <= c < w

    def segment_owner_after_cross(p: tuple[int, int], q: tuple[int, int]) -> tuple[int, int] | None:
        dr, dc = q[0] - p[0], q[1] - p[1]
        if abs(dr) != 1 or abs(dc) != 1:
            return None
        oq = cell_owner.get(q)
        if oq is not None and cell_index.get(q, 0) > 0 and cell_enter.get(q) == (dr, dc):
            return oq, cell_index[q]
        op = cell_owner.get(p)
        if op is not None and cell_index.get(p, 0) > 0 and cell_enter.get(p) == (-dr, -dc):
            return op, cell_index[p]
        return None

    stats = {
        "beam_count": len(beams),
        "swap_events": 0,
        "cross_swaps": 0,
        "segment_swaps": 0,
        "same_cell_collisions": 0,
    }

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
                raise AssertionError("color-swap variant expects only geometric diagonal crossings")
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
            seg_hits[(bid, obid)] = (split_idx, min(pr, pr + dr) * w + min(pc, pc + dc))
        for (a, b), (split_idx, key) in seg_hits.items():
            events.append((1, key // w, key % w, a, b, f"seg:{split_idx}"))

        if events:
            events.sort()
            _prio, _r, _c, a, b, kind = events[0]
            if kind == "cross":
                old_a = beams[a].color
                old_b = beams[b].color
                beams[a].color = old_b
                beams[b].color = old_a
                for bid in sorted((a, b)):
                    nxt, d = prop[bid]
                    move_beam(bid, nxt, d)
                stats["swap_events"] += 1
                stats["cross_swaps"] += 1
                continue

            split_idx = int(kind.split(":")[1])
            old_a = beams[a].color
            old_b = beams[b].color
            recolor_suffix(b, split_idx, old_a)
            beams[a].color = old_b
            nxt, d = prop[a]
            move_beam(a, nxt, d)
            stats["swap_events"] += 1
            stats["segment_swaps"] += 1
            continue

        for bid in sorted(prop):
            nxt, d = prop[bid]
            move_beam(bid, nxt, d)
    else:
        raise AssertionError("beam simulation did not converge")

    return out, stats


def main(grid: np.ndarray) -> np.ndarray:
    """
    Emit diagonal beams from framed 2x2 blocks; crossings swap beam colors instead of changing geometry.

    Input structure:
    - The dominant background surrounds several monochrome 2x2 blocks, each wrapped by the same 4x4 frame color.
    - Every usable opening is a corner gap on that frame border, so each emitted beam starts on a corner and moves diagonally outward.

    Rule:
    - Each corner gap launches one diagonal beam whose initial color matches the inner 2x2 block.
    - Beams keep moving in the same diagonal direction until they leave the grid.
    - When two beam trajectories cross, they pass straight through the crossing point and exchange colors; a later beam can also recolor the already-drawn suffix of an earlier diagonal when it crosses that segment.

    Key insight: Collision geometry is inert in this variant; the only thing transferred at an intersection is color, so solving the task means tracing where each color gets handed off.
    """
    out, _stats = _simulate(np.asarray(grid, dtype=int), strict_same_cell=True)
    assert out is not None, "expected a concrete output grid"
    return out


# --------------- generator (o5x) ---------------

# ---- CFG: tunable generation parameters ----
BLOCK_COUNT_RANGE = (2, 5)              # uniform draw [lo, hi)
GRID_SIZE_MAX = 31                      # exclusive upper bound for square grid side
GRID_SIZE_MIN_BY_BLOCKS = {2: 12, 3: 16, 4: 20}
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
GEN_RETRIES = 300


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
    n = int(rng.integers(_grid_size_min(block_count), GRID_SIZE_MAX))
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
            out, stats = _simulate(grid, strict_same_cell=False)
        except Exception:
            continue
        if out is None:
            continue
        if stats["same_cell_collisions"] != 0:
            continue
        if stats["swap_events"] < 1:
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
    raise RuntimeError("failed to generate a valid e12f9a14 color-swap input")
