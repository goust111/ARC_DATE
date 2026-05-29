# Combo variant of e12f9a14 (index 42)
# combo_spec: D2/2+D9/2+D19/2
# Name: 穿过交换颜色+沿对方方向转弯+交叉点特殊标记

from __future__ import annotations

import sys

import numpy as np

try:
    from arc_utils import get_bg, paste
except ImportError:
    sys.path.append("/home/lixinyu/snap/new/solvers")
    from arc_utils import get_bg, paste


# Design analysis for the combo variant
# a) Concrete behavior and trace:
#    - We keep the original visible objects: monochrome 2x2 blocks inside shared 4x4 frames. The
#      generator narrows legal exits to one corner gap per frame, so every beam is diagonal.
#    - A sample uses exactly two inward-facing diagonals. Example: left frame opens at local (3,3)
#      and right frame opens at local (3,0). The two beams step down toward each other and reach the
#      same background cell on the same tick.
#    - Original task at that moment would redirect both beams or create a thicker merged pattern,
#      depending on the angle class. In the combo rule the meeting cell is painted with a special
#      marker color, then each hidden beam state takes the other beam's color and direction.
#    - Because color and direction are swapped together, the visible colored diagonals continue
#      straight through the marker while the relay happens in the hidden beam identity. The marker is
#      therefore essential: it is the visible evidence that a collision occurred even though the arms
#      look like a clean X crossing.
# b) Complexity calibration:
#    - If we allowed arbitrary edge gaps, same-cell encounters between cardinal and diagonal beams
#      would create more bookkeeping than visual signal. Restricting to corner gaps preserves the
#      fused idea while keeping every collision a crisp diagonal X with a single marked center.
#    - Using exactly two frames prevents the combo from collapsing into clutter. One marked crossing
#      is enough to show the changed rule, and repeated examples can vary by orientation and spacing.
# c) Implicit-assumption audit against the original main():
#    - Original code assumes collision behavior is selected by 45-degree vs 90-degree geometry and
#      then implemented by truncation, gaps, bisector turns, or zigzag patterns. None of those
#      assumptions survive here.
#    - Original segment-cross and occupied-ray handlers are also tied to the old geometry-changing
#      semantics. The combo generator avoids those cases entirely; the new solver therefore treats
#      beam-into-old-ray events as invalid input rather than trying to reinterpret them.
#    - A new hidden-state assumption appears: after a collision, two beam identities may share the
#      same virtual collision cell before diverging. The solver stores that cell as the current head
#      for both beams but paints only the marker color there.
# d) Placement legality:
#    - Frames still use the original 4x4 geometry, remain fully inside the grid, and keep >=2 cells
#      of frame-bbox clearance.
#    - The generator chooses two frame positions and corner gaps so the diagonal seeds are aligned to
#      hit the same cell after at least two pre-collision steps and with at least two post-collision
#      steps remaining.
# e) Visual observability:
#    - Across 2-4 samples a human sees the same stable pattern: rays do not form the original gap or
#      zigzag outputs; instead they keep straight diagonal arms and every interaction site becomes a
#      grey marker. That is enough to infer that collisions are preserved but handled differently.
# f) Visual cleanliness:
#    - One beam per frame and one collision per grid keep the output sparse. The marker color is fixed
#      and excluded from backgrounds, frames, and source blocks, so the junction never blends into
#      other objects.
# g) Change inferability:
#    - There is no arbitrary color remapping keyed by numeric ids. Each beam starts with the color of
#      its own 2x2 block, and the only new visible color is the fixed collision marker. The relay of
#      beam identity is internal, but the changed collision geometry is still human-readable.
# New solver assumptions to review:
#    - Every legal combo instance uses corner gaps only and is intended to contain at least one
#      same-tick diagonal same-cell collision.
#    - The generator guarantees that beams never run into an older painted ray or a prior marker;
#      the solver asserts that constraint in strict mode.


MARK_COLOR = 5
BLOCK_COUNT = 2
FRAME_BBOX_GAP = 2
PLACEMENT_MARGIN = 2
GRID_SIZE_RANGE = (16, 31)
MIN_PRE_STEPS = 2
MIN_POST_STEPS = 2
MIN_BEAM_PIXELS = 8
GEN_RETRIES = 300

BG_CHOICES = [0, 8]
BG_WEIGHTS = [0.25, 0.75]
FRAME_COLOR_FOR_BG = {8: 3, 0: 8}
PALETTE_FOR_BG = {
    8: [1, 2, 4, 6, 7, 9],
    0: [1, 2, 3, 4, 6, 7, 9],
}

_PAIR_LAYOUTS = {
    "down": (((3, 3),), ((3, 0),)),
    "up": (((0, 3),), ((0, 0),)),
    "right": (((3, 3),), ((0, 3),)),
    "left": (((3, 0),), ((0, 0),)),
}


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


def _frames_separated(a: tuple[int, int], b: tuple[int, int]) -> bool:
    ar0, ac0 = a
    br0, bc0 = b
    ar1, ac1 = ar0 + 3, ac0 + 3
    br1, bc1 = br0 + 3, bc0 + 3
    gap = FRAME_BBOX_GAP
    return ar1 + gap < br0 or br1 + gap < ar0 or ac1 + gap < bc0 or bc1 + gap < ac0


def _rays_clear(grid: np.ndarray, bg: int, seeds: list[tuple[tuple[int, int], tuple[int, int]]]) -> bool:
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
        if steps < MIN_PRE_STEPS:
            return False
    return True


def _collision_cell_for_pair(
    positions: list[tuple[int, int]],
    gaps: list[tuple[tuple[int, int], ...]],
) -> tuple[int, int] | None:
    seeds = []
    for (r0, c0), (gap,) in zip(positions, gaps):
        sr, sc = r0 + gap[0], c0 + gap[1]
        dr, dc = _gap_dir(gap)
        seeds.append(((sr, sc), (dr, dc)))

    (r1, c1), (dr1, dc1) = seeds[0]
    (r2, c2), (dr2, dc2) = seeds[1]
    for t in range(1, 200):
        p1 = (r1 + dr1 * t, c1 + dc1 * t)
        p2 = (r2 + dr2 * t, c2 + dc2 * t)
        if p1 == p2:
            return p1
        if abs(p1[0] - p2[0]) > 200 or abs(p1[1] - p2[1]) > 200:
            break
    return None


def _pick_colors(rng: np.random.Generator) -> tuple[int, int, list[int]]:
    bg = int(rng.choice(BG_CHOICES, p=BG_WEIGHTS))
    frame_color = FRAME_COLOR_FOR_BG[bg]
    palette = [c for c in PALETTE_FOR_BG[bg] if c not in {frame_color, MARK_COLOR}]
    block_colors = list(rng.choice(palette, size=BLOCK_COUNT, replace=False))
    return bg, frame_color, block_colors


def _choose_collision_pair(
    n: int, rng: np.random.Generator
) -> tuple[list[tuple[int, int]], list[tuple[tuple[int, int], ...]]] | None:
    orient = str(rng.choice(sorted(_PAIR_LAYOUTS)))
    deltas = [d for d in range(2 * MIN_PRE_STEPS + 3, n - 6, 2)]
    if not deltas:
        return None
    delta = int(rng.choice(deltas))

    if orient in {"down", "up"}:
        c_max = n - 6 - delta
        if c_max < PLACEMENT_MARGIN:
            return None
        c_left = int(rng.integers(PLACEMENT_MARGIN, c_max + 1))
        positions = [(0, 0), (0, 0)]
        positions[0] = (0, c_left)
        positions[1] = (0, c_left + delta)
        t = (delta - 3) // 2
        if orient == "down":
            r_min = PLACEMENT_MARGIN
            r_max = n - 6 - (t + MIN_POST_STEPS)
        else:
            r_min = PLACEMENT_MARGIN + t + MIN_POST_STEPS
            r_max = n - 6
        if r_max < r_min:
            return None
        r0 = int(rng.integers(r_min, r_max + 1))
        positions[0] = (r0, positions[0][1])
        positions[1] = (r0, positions[1][1])
    else:
        r_max = n - 6 - delta
        if r_max < PLACEMENT_MARGIN:
            return None
        r_top = int(rng.integers(PLACEMENT_MARGIN, r_max + 1))
        positions = [(0, 0), (0, 0)]
        positions[0] = (r_top, 0)
        positions[1] = (r_top + delta, 0)
        t = (delta - 3) // 2
        if orient == "right":
            c_min = PLACEMENT_MARGIN
            c_max = n - 6 - (t + MIN_POST_STEPS)
        else:
            c_min = PLACEMENT_MARGIN + t + MIN_POST_STEPS
            c_max = n - 6
        if c_max < c_min:
            return None
        c0 = int(rng.integers(c_min, c_max + 1))
        positions[0] = (positions[0][0], c0)
        positions[1] = (positions[1][0], c0)

    assert _frames_separated(positions[0], positions[1]), "collision pair must preserve frame clearance"
    gaps = list(_PAIR_LAYOUTS[orient])
    return positions, gaps


class _Beam:
    __slots__ = ("bid", "color", "pos", "direction", "path", "active")

    def __init__(self, bid: int, color: int, pos: tuple[int, int], direction: tuple[int, int]):
        self.bid = bid
        self.color = int(color)
        self.pos = pos
        self.direction = direction
        self.path = [pos]
        self.active = True


def _simulate(grid: np.ndarray, *, strict: bool) -> tuple[np.ndarray | None, dict[str, int]]:
    g = np.asarray(grid, dtype=int)
    h, w = g.shape
    bg = get_bg(g)

    out = g.copy()
    blocks = _find_blocks_2x2(g, bg)
    assert blocks, "expected at least one 2x2 block in input"

    beams: dict[int, _Beam] = {}
    next_bid = 0
    painted: dict[tuple[int, int], int] = {}
    markers: set[tuple[int, int]] = set()

    stats = {
        "beam_count": 0,
        "collision_count": 0,
        "marker_count": 0,
        "unsupported_overlap_count": 0,
    }

    def spawn_beam(color: int, pos: tuple[int, int], direction: tuple[int, int]) -> None:
        nonlocal next_bid
        if abs(direction[0]) + abs(direction[1]) != 2:
            raise AssertionError("combo variant expects corner-gap diagonal beams only")
        if pos in painted:
            raise AssertionError("expected unique beam seed positions")
        bid = next_bid
        next_bid += 1
        beams[bid] = _Beam(bid, color, pos, direction)
        out[pos[0], pos[1]] = int(color)
        painted[pos] = bid

    for col, br, bc in blocks:
        frame_col = _infer_frame_color(g, bg, br, bc, col)
        gaps = _frame_gaps_for_block(g, bg, frame_col, br, bc)
        assert gaps, "expected each frame to have at least one gap"
        for gr, gc in gaps:
            dr = -1 if gr == br - 1 else (1 if gr == br + 2 else 0)
            dc = -1 if gc == bc - 1 else (1 if gc == bc + 2 else 0)
            direction = _normalize_dir(dr, dc)
            spawn_beam(col, (gr, gc), direction)

    stats["beam_count"] = len(beams)

    def in_bounds(cell: tuple[int, int]) -> bool:
        r, c = cell
        return 0 <= r < h and 0 <= c < w

    max_iters = h * w * 40
    for _ in range(max_iters):
        active = [bid for bid, beam in beams.items() if beam.active]
        if not active:
            break

        prop: dict[int, tuple[tuple[int, int], tuple[int, int]]] = {}
        for bid in active:
            beam = beams[bid]
            d = beam.direction
            nxt = (beam.pos[0] + d[0], beam.pos[1] + d[1])
            if not in_bounds(nxt) or int(g[nxt[0], nxt[1]]) != bg:
                beam.active = False
                continue
            if nxt in markers or nxt in painted:
                stats["unsupported_overlap_count"] += 1
                if strict:
                    raise AssertionError("combo generator must avoid beam-over-beam overlaps after launch")
                return None, stats
            prop[bid] = (nxt, d)

        if not prop:
            break

        cell_to_bids: dict[tuple[int, int], list[int]] = {}
        for bid, (nxt, _d) in prop.items():
            cell_to_bids.setdefault(nxt, []).append(bid)

        collision_events = [(cell, sorted(bidders)) for cell, bidders in cell_to_bids.items() if len(bidders) >= 2]
        if collision_events:
            collision_events.sort(key=lambda item: item[0])
            cell, bidders = collision_events[0]
            if len(bidders) != 2:
                raise AssertionError("expected only pairwise collisions in combo variant")
            a, b = bidders
            da = prop[a][1]
            db = prop[b][1]
            if abs(da[0]) + abs(da[1]) != 2 or abs(db[0]) + abs(db[1]) != 2:
                raise AssertionError("combo collision marker is defined only for diagonal beams")

            beams[a].path.append(cell)
            beams[b].path.append(cell)
            beams[a].pos = cell
            beams[b].pos = cell

            old_color_a, old_color_b = beams[a].color, beams[b].color
            old_dir_a, old_dir_b = beams[a].direction, beams[b].direction
            beams[a].color, beams[b].color = old_color_b, old_color_a
            beams[a].direction, beams[b].direction = old_dir_b, old_dir_a

            markers.add(cell)
            out[cell[0], cell[1]] = MARK_COLOR
            stats["collision_count"] += 1
            stats["marker_count"] = len(markers)
            continue

        for bid in sorted(prop):
            nxt, _d = prop[bid]
            beam = beams[bid]
            beam.path.append(nxt)
            beam.pos = nxt
            painted[nxt] = bid
            out[nxt[0], nxt[1]] = beam.color
    else:
        raise AssertionError("beam simulation did not converge")

    return out, stats


def main(grid: np.ndarray) -> np.ndarray:
    """
    Emit diagonal beams from framed 2x2 blocks; collisions become marked relay nodes instead of the original turns.

    Input structure:
    - The most common color is the background. Each monochrome 2x2 block is wrapped by the same 4x4 frame color.
    - Every legal opening is a single corner gap on that frame border, so each launched beam is diagonal.

    Rule:
    - Each corner gap emits one beam in the block's color and the beam keeps stepping diagonally outward.
    - When two beams reach the same background cell on the same tick, that cell is repainted with a fixed special marker color.
    - After that marked collision, the two hidden beam states swap both color and travel direction and continue from the marker cell; in the raster this yields straight crossing arms with an explicit marked junction.
    - Beams otherwise stop only at the grid boundary or when they would hit a frame; generator-legal inputs avoid later beam-over-beam overlaps.

    Key insight: The old task encoded collisions by changing path geometry, but this combo encodes them by preserving the crossing shape and making the junction explicit with a special marker.
    """
    out, _stats = _simulate(np.asarray(grid, dtype=int), strict=True)
    assert out is not None, "expected a concrete output grid"
    return out


def _gen_attempt() -> np.ndarray | None:
    rng = np.random.default_rng()
    n = int(rng.integers(*GRID_SIZE_RANGE))
    bg, frame_color, block_colors = _pick_colors(rng)

    chosen = _choose_collision_pair(n, rng)
    if chosen is None:
        return None
    positions, chosen_gaps = chosen
    collision = _collision_cell_for_pair(positions, chosen_gaps)
    if collision is None:
        return None

    grid = np.full((n, n), bg, dtype=int)
    seeds: list[tuple[tuple[int, int], tuple[int, int]]] = []
    for (r0, c0), color, gaps in zip(positions, block_colors, chosen_gaps):
        _draw_framed_block(grid, r0, c0, bg, frame_color, int(color), gaps)
        for gap in gaps:
            seeds.append(((r0 + gap[0], c0 + gap[1]), _gap_dir(gap)))

    if int(np.bincount(grid.ravel(), minlength=10).argmax()) != bg:
        return None
    if not _rays_clear(grid, bg, seeds):
        return None
    if collision[0] < MIN_POST_STEPS or collision[1] < MIN_POST_STEPS:
        return None
    if collision[0] >= n - MIN_POST_STEPS or collision[1] >= n - MIN_POST_STEPS:
        return None

    try:
        out, stats = _simulate(grid, strict=True)
    except Exception:
        return None

    if out is None:
        return None
    if stats["collision_count"] != 1 or stats["marker_count"] != 1:
        return None
    if stats["unsupported_overlap_count"] != 0:
        return None
    if np.count_nonzero(out != grid) < MIN_BEAM_PIXELS:
        return None
    if int(np.count_nonzero(out == MARK_COLOR) - np.count_nonzero(grid == MARK_COLOR)) != 1:
        return None
    return grid


def generate_input() -> np.ndarray:
    for _ in range(GEN_RETRIES):
        grid = _gen_attempt()
        if grid is not None:
            return grid
    raise RuntimeError("failed to generate a valid e12f9a14 combo input")
