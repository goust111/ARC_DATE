# Combo variant of e12f9a14 (index 19)
# combo_spec: D1/0+D2/2+D3/0
# Name: 正交射线+颜色交换+周期涂色

import numpy as np

from arc_utils import get_bg, paste


# Design analysis for the combo variant
# -------------------------------------
# a) concretization and behavior trace:
#    - Visible objects stay close to the source task: monochrome 2x2 blocks, each wrapped by a
#      shared 4x4 frame color, with one missing border pixel acting as the exit.
#    - D1/0 changes the usable exits to side gaps only, so every emitted ray is orthogonal:
#      top/bottom gaps emit vertical rays and left/right gaps emit horizontal rays.
#    - D2/2 changes collisions from geometric redirection to color transfer. On a one-layer ARC
#      raster, two orthogonal rays cannot both own the same crossing cell with different colors,
#      so this combo renders the crossing cell as background while still treating it as a
#      traversed step for both beams. After that empty crossing step, both beams keep moving
#      straight and swap their carried source color.
#    - D3/0 changes solid trails into a fixed AABB cycle where A is the beam's current carried
#      source color and B is the shared frame color. Example trace:
#        red beam: R R G | collision-hole | B B R R G G ...
#        blue beam: B B G | collision-hole | R R G G B B ...
#      The hole is a visible pass-through marker, while the post-collision A-slots reveal that
#      the carried colors have exchanged.
# b) complexity calibration:
#    - The cycle length is fixed at 4 and reused everywhere; there is no hidden per-instance
#      variation.
#    - Each source block keeps a unique color, so the color handoff is easy to track visually.
#    - The generator forces at least one orthogonal collision and enough runway after the
#      collision to expose swapped A-slots, preventing degeneration into "just straight rays"
#      or "just periodic trails".
# c) implicit assumption audit versus the original main():
#    - Original code assumed 8-direction motion and collision-driven turning/zigzags. Those
#      assumptions are removed: directions are now one of four cardinals and never change.
#    - Original collision handling relied on truncation, blocked cells, and direction rewrites.
#      Here the only collision state change is `current_color` swapping between two beams.
#    - Original periodic-color variant tied color phase to painted cells while geometric rules
#      sometimes removed cells. This combo instead ties the AABB phase to traversed path index,
#      so the blank crossing cell still consumes one step. That keeps phase continuity simple and
#      makes the color-exchange effect unambiguous.
#    - A late beam crossing an already-painted orthogonal trail would require retroactive
#      recoloring. To keep the combo readable and deterministic, the generator guarantees that all
#      intended collisions are same-tick orthogonal meetings; the solver asserts that stricter
#      regime.
# d) placement legality:
#    - Frames remain 4x4 and keep >=2 cells of bbox clearance.
#    - Only side gaps are used, one per frame in generated examples.
#    - The generator constructs a horizontal beam and a vertical beam whose intersection lies on
#      background, occurs at equal travel distance from both seeds, and leaves enough cells after
#      the crossing to show the swapped cycle.
# e) visual observability:
#    - Human-solvable cues are all visible in the input/output pair: side-gap direction tells the
#      launch axis, the shared frame color supplies the B color of the cycle, and unique block
#      colors supply the A colors that later exchange.
#    - The collision leaves an empty center plus swapped-color continuations, which is visibly
#      different from both the original turning task and the single-dimension straight-ray task.
# f) visual cleanliness:
#    - Using only two emitted beams per puzzle keeps the output sparse and readable.
#    - There are no same-axis overlaps, no head-on clashes, and no delayed re-crossings, so the
#      output remains a clean orthogonal crossing with two alternating-color tails.
# g) color-change inference:
#    - The AABB cycle uses only colors already present in each framed source object.
#    - The exchanged color is not keyed off numeric color ids; it is keyed off the other beam at
#      the orthogonal collision. The empty center and long post-swap continuation provide enough
#      structure for a solver to infer that the A component, not the whole pattern template, is
#      what changes.
# new solver assumptions checked against the generator:
#    - every usable gap is a side gap;
#    - exactly one horizontal and one vertical beam are emitted in generated samples;
#    - their only interaction is one same-tick orthogonal crossing on a background cell;
#    - both beams have visible post-swap continuation and at least one swapped A-slot.


_SIDE_GAPS = (
    (0, 1), (0, 2),
    (1, 3), (2, 3),
    (3, 2), (3, 1),
    (2, 0), (1, 0),
)
_GAPS_BY_DIR = {
    (-1, 0): ((0, 1), (0, 2)),
    (1, 0): ((3, 1), (3, 2)),
    (0, -1): ((2, 0), (1, 0)),
    (0, 1): ((1, 3), (2, 3)),
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
        if dr != 0 and dc != 0:
            continue
        if dr == 0 and dc == 0:
            continue
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


def _axis(direction: tuple[int, int]) -> str:
    return "h" if direction[0] == 0 else "v"


class _Beam:
    __slots__ = (
        "bid",
        "original_color",
        "current_color",
        "frame_color",
        "pos",
        "direction",
        "path",
        "active",
        "swap_count",
    )

    def __init__(self, bid: int, source_color: int, frame_color: int, pos: tuple[int, int], direction: tuple[int, int]):
        self.bid = bid
        self.original_color = int(source_color)
        self.current_color = int(source_color)
        self.frame_color = int(frame_color)
        self.pos = pos
        self.direction = direction
        self.path = [pos]
        self.active = True
        self.swap_count = 0

    def trail_color_at(self, idx: int) -> int:
        return self.current_color if idx % 4 in (0, 1) else self.frame_color


def _collect_beam_specs(g: np.ndarray) -> tuple[int, int, list[tuple[int, int, tuple[int, int], tuple[int, int]]]]:
    bg = get_bg(g)
    blocks = _find_blocks_2x2(g, bg)
    assert blocks, "expected at least one 2x2 block"

    frame_color_global: int | None = None
    specs: list[tuple[int, int, tuple[int, int], tuple[int, int]]] = []
    for source_color, br, bc in blocks:
        frame_color = _infer_frame_color(g, bg, br, bc, source_color)
        if frame_color_global is None:
            frame_color_global = frame_color
        else:
            assert frame_color_global == frame_color, "expected one shared frame color"
        gaps = _frame_gaps_for_block(g, bg, frame_color, br, bc)
        for gr, gc in gaps:
            direction = _gap_dir_for_block(br, bc, gr, gc)
            if direction is None:
                continue
            specs.append((source_color, frame_color, (gr, gc), direction))

    assert frame_color_global is not None, "expected a frame color"
    return bg, frame_color_global, specs


def _simulate(grid: np.ndarray, collect_stats: bool = False) -> tuple[np.ndarray, dict[str, int]]:
    g = np.asarray(grid, dtype=int)
    h, w = g.shape
    bg, frame_color_global, specs = _collect_beam_specs(g)
    assert specs, "expected at least one side-gap beam"

    out = g.copy()
    visited: set[tuple[int, int]] = set()
    beams: dict[int, _Beam] = {}

    stats = {
        "beam_count": len(specs),
        "swap_events": 0,
        "cross_cells": 0,
        "beam_pixels": 0,
        "frame_trail_pixels": 0,
        "swapped_source_pixels": 0,
        "post_swap_pixels": 0,
    }

    def in_bounds(cell: tuple[int, int]) -> bool:
        r, c = cell
        return 0 <= r < h and 0 <= c < w

    def paint_current_head(beam: _Beam) -> None:
        idx = len(beam.path) - 1
        cell = beam.path[-1]
        color = beam.trail_color_at(idx)
        out[cell[0], cell[1]] = color
        if beam.swap_count > 0:
            stats["post_swap_pixels"] += 1
        if idx % 4 in (2, 3):
            stats["frame_trail_pixels"] += 1
        elif beam.current_color != beam.original_color:
            stats["swapped_source_pixels"] += 1

    for bid, (source_color, frame_color, pos, direction) in enumerate(specs):
        beam = _Beam(bid, source_color, frame_color, pos, direction)
        beams[bid] = beam
        visited.add(pos)
        paint_current_head(beam)

    max_iters = h * w * 10
    for _ in range(max_iters):
        proposals: dict[int, tuple[int, int]] = {}
        for bid, beam in beams.items():
            if not beam.active:
                continue
            nxt = (beam.pos[0] + beam.direction[0], beam.pos[1] + beam.direction[1])
            if not in_bounds(nxt):
                beam.active = False
                continue
            if int(g[nxt[0], nxt[1]]) != bg:
                beam.active = False
                continue
            assert nxt not in visited, "generator must avoid delayed crossings or reused cells"
            proposals[bid] = nxt

        if not proposals:
            break

        cell_to_bids: dict[tuple[int, int], list[int]] = {}
        for bid, cell in proposals.items():
            cell_to_bids.setdefault(cell, []).append(bid)

        cross_pairs: list[tuple[tuple[int, int], int, int]] = []
        singles: list[tuple[int, tuple[int, int]]] = []
        for cell, bidders in sorted(cell_to_bids.items()):
            if len(bidders) == 1:
                singles.append((bidders[0], cell))
                continue
            assert len(bidders) == 2, "expected at most one horizontal and one vertical beam per crossing"
            a, b = sorted(bidders)
            assert _axis(beams[a].direction) != _axis(beams[b].direction), "same-axis collisions are not part of this combo"
            cross_pairs.append((cell, a, b))

        for bid, cell in singles:
            beam = beams[bid]
            beam.path.append(cell)
            beam.pos = cell
            visited.add(cell)
            paint_current_head(beam)

        for cell, a, b in cross_pairs:
            beam_a = beams[a]
            beam_b = beams[b]
            beam_a.path.append(cell)
            beam_b.path.append(cell)
            beam_a.pos = cell
            beam_b.pos = cell
            visited.add(cell)
            out[cell[0], cell[1]] = bg
            color_a = beam_a.current_color
            color_b = beam_b.current_color
            beam_a.current_color = color_b
            beam_b.current_color = color_a
            beam_a.swap_count += 1
            beam_b.swap_count += 1
            stats["swap_events"] += 1
            stats["cross_cells"] += 1
    else:
        raise AssertionError("straight color-swap simulation did not converge")

    if collect_stats:
        painted_mask = (g == bg) & (out != bg)
        stats["beam_pixels"] = int(np.count_nonzero(painted_mask))
        stats["frame_trail_pixels"] = int(np.count_nonzero(painted_mask & (out == frame_color_global)))

    return out, stats


def main(grid: np.ndarray) -> np.ndarray:
    """
    Emit straight orthogonal rays from framed 2x2 blocks; when two rays meet, they pass through an empty crossing cell and exchange the color that drives their repeating trail pattern.

    Input structure:
    - The dominant background surrounds monochrome 2x2 blocks, each wrapped by the same 4x4 frame color.
    - Only side gaps on those frames are meaningful here, so every launch direction is horizontal or vertical.

    Rule:
    - Each side gap emits one straight ray in the outward cardinal direction.
    - A ray paints a repeating AABB cycle along its traversed path, where A is the ray's current carried source color and B is the shared frame color.
    - When a horizontal ray and a vertical ray reach the same cell on the same step, that crossing cell is left as background, both rays continue straight, and they swap carried source colors for all later A-slots.

    Key insight:
    - Geometry no longer changes after a collision; the real state change is that the A component of the periodic trail jumps from one source block to the other exactly after the empty orthogonal crossing.
    """
    return _simulate(np.asarray(grid, dtype=int), collect_stats=False)[0]


# --------------- generator ---------------

GRID_SIZE_RANGE = (16, 31)
BG_CHOICES = [0, 8]
BG_WEIGHTS = [0.25, 0.75]
FRAME_COLOR_FOR_BG = {8: 3, 0: 8}
PALETTE_FOR_BG = {
    8: [1, 2, 4, 6, 7, 9],
    0: [1, 2, 3, 4, 6, 7, 9],
}
FRAME_BBOX_GAP = 2
MIN_PRE_CROSS_STEPS = 3
MIN_POST_CROSS_STEPS = 4
MIN_TOTAL_BEAM_PIXELS = 12
GEN_RETRIES = 400


def _frames_separated(a: tuple[int, int], b: tuple[int, int]) -> bool:
    ar0, ac0 = a
    br0, bc0 = b
    ar1, ac1 = ar0 + 3, ac0 + 3
    br1, bc1 = br0 + 3, bc0 + 3
    gap = FRAME_BBOX_GAP
    return ar1 + gap < br0 or br1 + gap < ar0 or ac1 + gap < bc0 or bc1 + gap < ac0


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


def _ray_cells(
    grid: np.ndarray,
    bg: int,
    seed: tuple[int, int],
    direction: tuple[int, int],
) -> list[tuple[int, int]] | None:
    h, w = grid.shape
    cells = [seed]
    r, c = seed
    dr, dc = direction
    while True:
        r += dr
        c += dc
        if not (0 <= r < h and 0 <= c < w):
            break
        if int(grid[r, c]) != bg:
            return None
        cells.append((r, c))
    return cells


def _pick_colors(rng: np.random.Generator) -> tuple[int, int, list[int]]:
    bg = int(rng.choice(BG_CHOICES, p=BG_WEIGHTS))
    frame_color = FRAME_COLOR_FOR_BG[bg]
    palette = [c for c in PALETTE_FOR_BG[bg] if c != frame_color]
    block_colors = list(rng.choice(palette, size=2, replace=False))
    return bg, frame_color, block_colors


def _candidate_crossings(
    n: int,
    hdir: tuple[int, int],
    vdir: tuple[int, int],
) -> list[tuple[int, int, int]]:
    rows: list[int] = []
    cols: list[int] = []
    if hdir == (0, 1):
        cols = list(range(MIN_PRE_CROSS_STEPS + 3, n - MIN_POST_CROSS_STEPS))
    else:
        cols = list(range(MIN_POST_CROSS_STEPS, n - MIN_PRE_CROSS_STEPS - 3))
    if vdir == (1, 0):
        rows = list(range(MIN_PRE_CROSS_STEPS + 3, n - MIN_POST_CROSS_STEPS))
    else:
        rows = list(range(MIN_POST_CROSS_STEPS, n - MIN_PRE_CROSS_STEPS - 3))

    cands: list[tuple[int, int, int]] = []
    for ir in rows:
        for ic in cols:
            max_dist = min(
                ic if hdir == (0, 1) else (n - 1 - ic),
                (n - 1 - ic) if hdir == (0, 1) else ic,
                ir if vdir == (1, 0) else (n - 1 - ir),
                (n - 1 - ir) if vdir == (1, 0) else ir,
            )
            for t in range(MIN_PRE_CROSS_STEPS, max_dist + 1):
                cands.append((ir, ic, t))
    return cands


def _gen_attempt() -> np.ndarray | None:
    rng = np.random.default_rng()
    n = int(rng.integers(*GRID_SIZE_RANGE))
    bg, frame_color, block_colors = _pick_colors(rng)

    hdir = (0, 1) if int(rng.integers(2)) == 0 else (0, -1)
    vdir = (1, 0) if int(rng.integers(2)) == 0 else (-1, 0)
    candidates = _candidate_crossings(n, hdir, vdir)
    if not candidates:
        return None

    for idx in rng.permutation(len(candidates)):
        ir, ic, t = candidates[int(idx)]
        hseed = (ir - t * hdir[0], ic - t * hdir[1])
        vseed = (ir - t * vdir[0], ic - t * vdir[1])

        gap_choices_h = list(_GAPS_BY_DIR[hdir])
        gap_choices_v = list(_GAPS_BY_DIR[vdir])
        rng.shuffle(gap_choices_h)
        rng.shuffle(gap_choices_v)

        for hgap in gap_choices_h:
            hr0 = hseed[0] - hgap[0]
            hc0 = hseed[1] - hgap[1]
            if not (0 <= hr0 <= n - 4 and 0 <= hc0 <= n - 4):
                continue
            for vgap in gap_choices_v:
                vr0 = vseed[0] - vgap[0]
                vc0 = vseed[1] - vgap[1]
                if not (0 <= vr0 <= n - 4 and 0 <= vc0 <= n - 4):
                    continue
                if not _frames_separated((hr0, hc0), (vr0, vc0)):
                    continue

                obstacles = np.full((n, n), bg, dtype=int)
                _draw_framed_block(obstacles, hr0, hc0, bg, frame_color, int(block_colors[0]), ())
                _draw_framed_block(obstacles, vr0, vc0, bg, frame_color, int(block_colors[1]), ())

                hcells = _ray_cells(obstacles, bg, hseed, hdir)
                vcells = _ray_cells(obstacles, bg, vseed, vdir)
                if hcells is None or vcells is None:
                    continue
                if (ir, ic) not in set(hcells) or (ir, ic) not in set(vcells):
                    continue

                post_h = len(hcells) - hcells.index((ir, ic)) - 1
                post_v = len(vcells) - vcells.index((ir, ic)) - 1
                if post_h < MIN_POST_CROSS_STEPS or post_v < MIN_POST_CROSS_STEPS:
                    continue

                grid = np.full((n, n), bg, dtype=int)
                _draw_framed_block(grid, hr0, hc0, bg, frame_color, int(block_colors[0]), (hgap,))
                _draw_framed_block(grid, vr0, vc0, bg, frame_color, int(block_colors[1]), (vgap,))

                if int(np.bincount(grid.ravel(), minlength=10).argmax()) != bg:
                    continue

                try:
                    out, stats = _simulate(grid, collect_stats=True)
                except Exception:
                    continue

                if stats["beam_count"] != 2:
                    continue
                if stats["swap_events"] != 1 or stats["cross_cells"] != 1:
                    continue
                if stats["beam_pixels"] < MIN_TOTAL_BEAM_PIXELS:
                    continue
                if stats["frame_trail_pixels"] < 4:
                    continue
                if stats["swapped_source_pixels"] < 2:
                    continue
                if stats["post_swap_pixels"] < 6:
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
    raise RuntimeError("failed to generate a valid e12f9a14 combo input")
