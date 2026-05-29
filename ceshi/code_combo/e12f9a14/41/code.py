# Combo variant of e12f9a14 (index 41)
# combo_spec: D6/1+D7/1+D14/0
# Name: 框架作障碍物+按距离排序+碰前景停

from __future__ import annotations

import numpy as np

from arc_utils import get_bg, paste


# Design analysis for the combo variant
# -------------------------------------
# a) Concrete behavior and trace
#    - Visible objects stay close to the original task: several unique-color 2x2 blocks, each wrapped by
#      a shared-color 4x4 frame. Border background holes are beam exits.
#    - D6/1 removes the old beam-beam turning system. After this change, a beam is just a straight ray:
#      it paints the gap, keeps moving in that outward direction, and stops as soon as the next cell is
#      out of bounds or already foreground.
#    - D7/1 expands "foreground obstacle" to include the intact frame walls of other objects. So a beam can
#      now terminate directly against another frame instead of being considered an illegal placement.
#    - D14/0 changes when rays are allowed to see each other. Frames are ordered by distance to the grid
#      center; the nearest frame fires all of its gaps first, then its painted trail freezes. Later frames
#      treat that older trail as ordinary foreground and therefore stop before entering it.
#    - Small concrete example used by the generator:
#      * Frame A is closest to the center and has a single east-facing gap. Its ray travels rightward and
#        stops at frame B's left wall.
#      * Frame C sits farther away below-right with a single north-facing gap. Its upward ray would meet
#        A's ray at one background cell.
#      * Simultaneous baseline: A and C reach that meeting cell in the same step, so neither paints it.
#      * Combo rule: A is processed first, so it paints through that cell. When C is processed later, the
#        next cell is already foreground, so C stops one step earlier. The output difference is thus caused
#        by visible centrality order, while the final end of A is still controlled by a frame obstacle.
# b) Complexity calibration
#    - The rule is intentionally narrowed to straight rays and single-gap generated samples. This keeps the
#      visual cue legible: one central frame owns the contested crossing, one farther frame is a static wall,
#      and one farther frame gets cut off by the earlier trail.
#    - Solver support remains general enough for multiple gaps, but the generator fixes one gap per frame so
#      that the order effect is not buried inside many same-color trajectories.
# c) Hidden assumptions from the original solver that no longer hold
#    - Original proposal logic assumed a non-background next cell was either out-of-bounds or part of the
#      old collision engine. That assumption is invalid now: any non-background pixel is a terminal obstacle.
#    - Original cell-owner / segment-cross / 45-degree / 90-degree machinery relied on beams being dynamic
#      participants throughout one global synchronous simulation. Under this combo, older trails are static
#      and never redirect later beams, so that machinery must be removed.
#    - Original generator rejected layouts where beams would touch frames. That is the opposite of D7/1, so
#      generation must now deliberately create at least one frame hit.
# d) Placement legality
#    - Frames still use the original 4x4 geometry, stay inside the grid with a 2-cell outer margin, and keep
#      at least a 2-cell bounding-box gap from one another.
#    - The generator constructs a legal triple: A (near center), B (to the right, acting as a wall), and C
#      (below-right, later blocked by A's trail). Their positions are derived analytically, then rechecked.
# e) Visual observability
#    - The ordering key is purely geometric and human-visible: squared distance from each frame center to the
#      grid center, with reading-order tie-break only as a solver fallback.
#    - Samples are rejected unless sequential output differs from the simultaneous baseline, at least one beam
#      is stopped by an original frame wall, and at least one later beam is stopped by an earlier trail.
# f) Visual tidiness
#    - Using only three frames and one gap each avoids clutter. The key effects occupy different roles:
#      central long ray, right-side blocking frame, and lower ray truncated against the earlier trail.
# g) Color inference
#    - Colors are unchanged semantically: each beam keeps the color of its 2x2 block. No hidden color mapping
#      or id-dependent behavior is introduced; only visible geometry controls the output.
# New assumptions introduced by this solver and how the generator satisfies them
#    - Assumption: every frame has at least one structural gap. Generator stamps exactly one valid gap per frame.
#    - Assumption: generated layouts give the nearest frame a unique center-distance key. Generator rejects ties.
#    - Assumption: the showcased order effect uses ordinary cell occupancy, not geometric diagonal crossings.
#      Generator uses only cardinal rays in its forced layout, so the simplified straight-ray solver matches
#      the intended interaction exactly.


FRAME_BBOX_GAP = 2
PLACEMENT_MARGIN = 2

BG_CHOICES = [0, 8]
BG_WEIGHTS = [0.25, 0.75]
FRAME_COLOR_FOR_BG = {8: 3, 0: 8}
PALETTE_FOR_BG = {
    8: [1, 2, 4, 6, 7, 9],
    0: [1, 2, 3, 4, 6, 7, 9],
}

BLOCK_COUNT = 3
GRID_SIZE_MIN = 25
GRID_SIZE_MAX = 31
MIN_RAY_STEPS = 2
MIN_BEAM_PIXELS_PER_GAP = 2
GEN_RETRIES = 400
LAYOUT_RETRIES = 120
FORCED_INTERSECTION_STEPS = 5
FORCED_RUNWAY_CHOICES = [7, 8]


class _Beam:
    __slots__ = ("color", "pos", "direction", "active")

    def __init__(self, color: int, pos: tuple[int, int], direction: tuple[int, int]):
        self.color = int(color)
        self.pos = pos
        self.direction = direction
        self.active = True


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
    return _normalize_dir(dr, dc)


def _frame_center_dist2(br: int, bc: int, h: int, w: int) -> int:
    dr2 = 2 * br + 1 - (h - 1)
    dc2 = 2 * bc + 1 - (w - 1)
    return int(dr2 * dr2 + dc2 * dc2)


def _frame_wall_and_block_cells(
    g: np.ndarray,
    bg: int,
    blocks: list[tuple[int, int, int]],
) -> tuple[set[tuple[int, int]], set[tuple[int, int]]]:
    frame_cells: set[tuple[int, int]] = set()
    block_cells: set[tuple[int, int]] = set()
    for col, br, bc in blocks:
        wall = _infer_frame_color(g, bg, br, bc, col)
        gaps = set(_frame_gaps_for_block(g, bg, wall, br, bc))
        fr0, fc0 = br - 1, bc - 1
        for r in range(fr0, fr0 + 4):
            for c in range(fc0, fc0 + 4):
                if (r, c) in gaps:
                    continue
                if r in (fr0, fr0 + 3) or c in (fc0, fc0 + 3):
                    frame_cells.add((r, c))
                else:
                    block_cells.add((r, c))
    return frame_cells, block_cells


def _build_frames(g: np.ndarray, bg: int) -> list[dict[str, object]]:
    h, w = g.shape
    blocks = _find_blocks_2x2(g, bg)
    assert blocks, "expected at least one 2x2 block in input"
    frames: list[dict[str, object]] = []
    for color, br, bc in blocks:
        wall = _infer_frame_color(g, bg, br, bc, color)
        gaps = _frame_gaps_for_block(g, bg, wall, br, bc)
        assert gaps, "expected each frame to have at least one structural gap"
        frames.append(
            {
                "color": int(color),
                "br": int(br),
                "bc": int(bc),
                "gaps": list(gaps),
                "dist2": _frame_center_dist2(br, bc, h, w),
            }
        )
    return frames


def _frame_seeds(frame: dict[str, object]) -> list[tuple[int, tuple[int, int], tuple[int, int]]]:
    br = int(frame["br"])
    bc = int(frame["bc"])
    color = int(frame["color"])
    seeds: list[tuple[int, tuple[int, int], tuple[int, int]]] = []
    for gr, gc in frame["gaps"]:
        dr = -1 if gr == br - 1 else (1 if gr == br + 2 else 0)
        dc = -1 if gc == bc - 1 else (1 if gc == bc + 2 else 0)
        d = _normalize_dir(dr, dc)
        assert d != (0, 0), "expected a gap to face outward"
        seeds.append((color, (gr, gc), d))
    return seeds


def _simulate_batch(
    g: np.ndarray,
    out: np.ndarray,
    bg: int,
    seeds: list[tuple[int, tuple[int, int], tuple[int, int]]],
    trail_owner: dict[tuple[int, int], int],
    exec_idx: int,
    frame_cells: set[tuple[int, int]],
    block_cells: set[tuple[int, int]],
    collect_stats: bool,
) -> dict[str, int]:
    h, w = g.shape
    beams: dict[int, _Beam] = {}
    next_bid = 0
    stats = {
        "beam_count": 0,
        "painted_pixels": 0,
        "boundary_stops": 0,
        "frame_stops": 0,
        "block_stops": 0,
        "prior_trail_stops": 0,
        "same_batch_trail_stops": 0,
        "contested_stops": 0,
    }

    def paint(cell: tuple[int, int], color: int) -> None:
        out[cell[0], cell[1]] = int(color)
        trail_owner[cell] = exec_idx
        if collect_stats:
            stats["painted_pixels"] += 1

    def classify_stop(cell: tuple[int, int] | None) -> None:
        if not collect_stats:
            return
        if cell is None:
            stats["boundary_stops"] += 1
            return
        if int(g[cell]) != bg:
            if cell in frame_cells:
                stats["frame_stops"] += 1
            elif cell in block_cells:
                stats["block_stops"] += 1
            else:
                raise AssertionError("unexpected input foreground classification")
            return
        owner = trail_owner.get(cell)
        assert owner is not None, "expected beam trail owner for painted background obstacle"
        if owner < exec_idx:
            stats["prior_trail_stops"] += 1
        else:
            stats["same_batch_trail_stops"] += 1

    for color, pos, direction in seeds:
        assert int(out[pos]) == bg, "expected each beam seed to be an unpainted gap"
        paint(pos, color)
        beams[next_bid] = _Beam(color, pos, direction)
        next_bid += 1

    if collect_stats:
        stats["beam_count"] = len(beams)

    def in_bounds(cell: tuple[int, int]) -> bool:
        return 0 <= cell[0] < h and 0 <= cell[1] < w

    max_iters = h * w
    for _ in range(max_iters):
        proposals: dict[tuple[int, int], list[int]] = {}
        any_active = False

        for bid, beam in beams.items():
            if not beam.active:
                continue
            any_active = True
            nxt = (beam.pos[0] + beam.direction[0], beam.pos[1] + beam.direction[1])
            if not in_bounds(nxt):
                beam.active = False
                classify_stop(None)
                continue
            if int(out[nxt]) != bg:
                beam.active = False
                classify_stop(nxt)
                continue
            proposals.setdefault(nxt, []).append(bid)

        if not any_active or not proposals:
            break

        moved = False
        for cell, bidders in proposals.items():
            if len(bidders) > 1:
                for bid in bidders:
                    beams[bid].active = False
                if collect_stats:
                    stats["contested_stops"] += len(bidders)
                continue
            bid = bidders[0]
            beam = beams[bid]
            paint(cell, beam.color)
            beam.pos = cell
            moved = True

        if not moved:
            break
    else:
        raise AssertionError("straight-ray simulation did not converge")

    return stats


def _solve(grid: np.ndarray, sequential: bool, want_stats: bool = False):
    g = np.asarray(grid, dtype=int)
    bg = get_bg(g)
    frames = _build_frames(g, bg)
    blocks = _find_blocks_2x2(g, bg)
    frame_cells, block_cells = _frame_wall_and_block_cells(g, bg, blocks)
    out = g.copy()

    if not sequential:
        trail_owner: dict[tuple[int, int], int] = {}
        all_seeds: list[tuple[int, tuple[int, int], tuple[int, int]]] = []
        for frame in frames:
            all_seeds.extend(_frame_seeds(frame))
        stats = _simulate_batch(g, out, bg, all_seeds, trail_owner, 0, frame_cells, block_cells, want_stats)
        if want_stats:
            return out, stats
        return out

    order = sorted(range(len(frames)), key=lambda i: (int(frames[i]["dist2"]), int(frames[i]["br"]), int(frames[i]["bc"])))
    trail_owner: dict[tuple[int, int], int] = {}
    agg = {
        "frame_count": len(frames),
        "order": list(order),
        "dist2": [int(frames[i]["dist2"]) for i in order],
        "painted_pixels": 0,
        "boundary_stops": 0,
        "frame_stops": 0,
        "block_stops": 0,
        "prior_trail_stops": 0,
        "same_batch_trail_stops": 0,
        "contested_stops": 0,
        "frames_blocked_by_prior": 0,
    }

    for exec_idx, frame_idx in enumerate(order):
        batch = _simulate_batch(
            g,
            out,
            bg,
            _frame_seeds(frames[frame_idx]),
            trail_owner,
            exec_idx,
            frame_cells,
            block_cells,
            collect_stats=True,
        )
        for key in ("painted_pixels", "boundary_stops", "frame_stops", "block_stops", "prior_trail_stops", "same_batch_trail_stops", "contested_stops"):
            agg[key] += int(batch[key])
        if int(batch["prior_trail_stops"]) > 0:
            agg["frames_blocked_by_prior"] += 1

    if want_stats:
        return out, agg
    return out


def main(grid: np.ndarray) -> np.ndarray:
    """
    Emit straight rays from framed 2x2 blocks, but process the frames from center-near to center-far.

    Input structure:
    - The background is the most common color.
    - Each object is a monochrome 2x2 block surrounded by a shared-color 4x4 frame.
    - One or more background holes on the frame border are structural gaps; each gap is a ray source.

    Rule:
    - Rank frames by distance from the grid center, nearest first; ties break by top-to-bottom, then left-to-right.
    - For one frame at a time, recolor all of that frame's gaps with the inner block color and let those rays advance
      synchronously in their outward straight directions.
    - A ray stops immediately when its next step would leave the grid or enter any non-background pixel.
      Obstacles include other frames' walls, other blocks, and ray pixels painted by earlier frames or earlier steps.
    - If two rays from the currently firing frame try to enter the same empty cell on the same step, neither claims it.

    Key insight:
    - The same framed objects play two roles at once: they launch rays, and their intact walls also block rays.
      Because more central frames fire earlier, their finished trails become visible foreground barriers that can truncate
      farther frames before those later rays reach cells they would have contested under a simultaneous interpretation.
    """
    return _solve(grid, sequential=True)


def variant_metrics(grid: np.ndarray) -> dict[str, int]:
    seq, stats = _solve(grid, sequential=True, want_stats=True)
    sim = _solve(grid, sequential=False)
    return {
        "frame_count": int(stats["frame_count"]),
        "frame_stops": int(stats["frame_stops"]),
        "prior_trail_stops": int(stats["prior_trail_stops"]),
        "frames_blocked_by_prior": int(stats["frames_blocked_by_prior"]),
        "contested_stops": int(stats["contested_stops"]),
        "diff_vs_simultaneous": int(np.count_nonzero(seq != sim)),
        "painted_pixels": int(np.count_nonzero(seq != np.asarray(grid))),
    }


def _frames_separated(a: tuple[int, int], b: tuple[int, int]) -> bool:
    ar0, ac0 = a
    br0, bc0 = b
    ar1, ac1 = ar0 + 3, ac0 + 3
    br1, bc1 = br0 + 3, bc0 + 3
    gap = FRAME_BBOX_GAP
    return ar1 + gap < br0 or br1 + gap < ar0 or ac1 + gap < bc0 or bc1 + gap < ac0


def _frame_fits(n: int, pos: tuple[int, int]) -> bool:
    r0, c0 = pos
    return PLACEMENT_MARGIN <= r0 <= n - 4 - PLACEMENT_MARGIN and PLACEMENT_MARGIN <= c0 <= n - 4 - PLACEMENT_MARGIN


def _pick_colors(rng: np.random.Generator) -> tuple[int, int, list[int]]:
    bg = int(rng.choice(BG_CHOICES, p=BG_WEIGHTS))
    frame_color = FRAME_COLOR_FOR_BG[bg]
    palette = [c for c in PALETTE_FOR_BG[bg] if c != frame_color]
    block_colors = list(map(int, rng.choice(palette, size=BLOCK_COUNT, replace=False)))
    return bg, frame_color, block_colors


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


def _choose_layout(n: int, rng: np.random.Generator) -> dict[str, object] | None:
    t = FORCED_INTERSECTION_STEPS
    for _ in range(LAYOUT_RETRIES):
        a_rowoff = int(rng.choice([1, 2]))
        b_rowoff = int(rng.choice([1, 2]))
        c_coloff = int(rng.choice([1, 2]))
        runway = int(rng.choice(FORCED_RUNWAY_CHOICES))

        ra_lo = max(PLACEMENT_MARGIN, n // 2 - 4)
        ra_hi = min(n // 2 - 1, n - 4 - PLACEMENT_MARGIN - (a_rowoff + t))
        if ra_lo > ra_hi:
            continue
        ca_lo = max(PLACEMENT_MARGIN, n // 2 - 6)
        ca_hi = min(n // 2 - 4, n - 4 - PLACEMENT_MARGIN - (4 + runway))
        if ca_lo > ca_hi:
            continue

        ra = int(rng.integers(ra_lo, ra_hi + 1))
        ca = int(rng.integers(ca_lo, ca_hi + 1))

        a = (ra, ca)
        b = (ra, ca + 4 + runway)
        intersection_row = ra + a_rowoff
        intersection_col = ca + 3 + t
        c = (intersection_row + t, intersection_col - c_coloff)

        if not (_frame_fits(n, a) and _frame_fits(n, b) and _frame_fits(n, c)):
            continue
        if not (_frames_separated(a, b) and _frames_separated(a, c) and _frames_separated(b, c)):
            continue

        dists = [
            _frame_center_dist2(a[0] + 1, a[1] + 1, n, n),
            _frame_center_dist2(b[0] + 1, b[1] + 1, n, n),
            _frame_center_dist2(c[0] + 1, c[1] + 1, n, n),
        ]
        if len(set(dists)) != 3:
            continue
        if dists[0] != min(dists):
            continue

        return {
            "positions": [a, b, c],
            "gaps": [((a_rowoff, 3),), ((b_rowoff, 3),), ((0, c_coloff),)],
            "dists": dists,
            "intersection": (intersection_row, intersection_col),
        }
    return None


def _gen_attempt() -> np.ndarray | None:
    rng = np.random.default_rng()
    n = int(rng.integers(GRID_SIZE_MIN, GRID_SIZE_MAX))
    bg, frame_color, block_colors = _pick_colors(rng)

    layout = _choose_layout(n, rng)
    if layout is None:
        return None

    grid = np.full((n, n), bg, dtype=int)
    for pos, color, gaps in zip(layout["positions"], block_colors, layout["gaps"]):
        _draw_framed_block(grid, pos[0], pos[1], bg, frame_color, int(color), gaps)

    if int(np.bincount(grid.ravel(), minlength=10).argmax()) != bg:
        return None

    try:
        out = main(grid)
        metrics = variant_metrics(grid)
    except Exception:
        return None

    if np.count_nonzero(out != grid) < BLOCK_COUNT * MIN_BEAM_PIXELS_PER_GAP:
        return None
    if metrics["frame_stops"] < 1:
        return None
    if metrics["prior_trail_stops"] < 1:
        return None
    if metrics["frames_blocked_by_prior"] < 1:
        return None
    if metrics["diff_vs_simultaneous"] < 2:
        return None
    return grid


def generate_input() -> np.ndarray:
    for _ in range(GEN_RETRIES):
        grid = _gen_attempt()
        if grid is not None:
            return grid
    raise RuntimeError("failed to generate a valid combo e12f9a14 input")
