# Variant of e12f9a14
# Dimension: D1 — 射线发射方向 (path)
# Change: 对角线射线（diagonal_ray）——彩色像素通过绿色框架缺口沿45度对角线方向向外发射，方向由缺口相对于内部像素的... → 直线射线（水平/垂直）

import itertools

import numpy as np

from arc_utils import get_bg, paste


# Design analysis for this D1 variant
# -----------------------------------
# concretization:
#   The framed 2x2 colored blocks remain unchanged, but every usable opening is now
#   a side opening, not a corner opening. A side opening has a unique outward normal,
#   so the emitted mark is an orthogonal ray: top gaps shoot upward, bottom downward,
#   left/right gaps horizontally. This keeps the object vocabulary from the source task
#   while making the path geometry visibly different.
# behavior trace:
#   Original local step: a NW-corner-like opening could launch a diagonal ray; after a
#   collision the solver could turn beams into diagonal/cardinal zigzags.
#   Variant local step: a top-side opening at local (0,1) launches straight upward.
#   If an upward ray later reaches a horizontal ray, the crossing cell becomes a 1-cell
#   hole and both rays keep going on their own axes. The difference is visible exactly
#   at the intersection: the old task bends, this variant leaves an orthogonal cross
#   with a blank center.
# complexity calibration:
#   Removing diagonals makes corner openings ambiguous in orthogonal space, so the
#   generator uses only side openings. To avoid the variant collapsing into the source
#   solver on side-only no-collision scenes, the generator requires at least one
#   horizontal/vertical crossing per instance.
# implicit assumption audit:
#   The original main() assumed:
#   - gaps may map to 8 directions; no longer true, so direction inference is reduced
#     to the four cardinals and corner gaps are ignored.
#   - collisions are classified by 45/90 degrees and may create new diagonal states;
#     no longer true, so the beam state now has a single fixed direction.
#   - cell ownership can be truncated and redirected; no longer needed because a
#     crossing only punches a hole, it does not redirect prior paths.
# placement legality:
#   The generator must ensure every chosen side gap has unobstructed straight runway to
#   the border, that no two horizontal rays share a row, and that no two vertical rays
#   share a column; otherwise same-axis overlaps or head-on clashes would create extra
#   cases unsupported by the intended rule set.
# visual observability:
#   Each puzzle has at least one orthogonal crossing. That guarantees a visible blank
#   hole inside a plus-shaped intersection, which differentiates the variant from both
#   simple one-ray scenes and the original diagonal-bending task.
# visual cleanliness:
#   Bounding boxes still keep >=2 cells of separation, crossings are forbidden on seed
#   cells, and same-axis overlaps are rejected. This keeps the output as clean rows and
#   columns rather than dense overpainted clutter.
# color inference:
#   Colors still mean "ray color equals inner 2x2 block color". No hidden ordering or
#   palette-dependent mapping is introduced.
# new solver assumptions checked against generator:
#   - every structural gap used by the puzzle is a side gap;
#   - at least one horizontal ray and one vertical ray exist;
#   - every perpendicular crossing happens on background, not on a frame pixel or gap
#     seed;
#   - no same-axis row/column duplication occurs among emitted rays.


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


class _Beam:
    __slots__ = ("bid", "color", "pos", "dir", "active")

    def __init__(self, bid: int, color: int, pos: tuple[int, int], direction: tuple[int, int]):
        self.bid = bid
        self.color = int(color)
        self.pos = pos
        self.dir = direction
        self.active = True


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
    painted_dir: dict[tuple[int, int], tuple[int, int]] = {}
    blocked: set[tuple[int, int]] = set()

    cross_same_tick = 0
    cross_with_painted = 0

    def in_bounds(cell: tuple[int, int]) -> bool:
        r, c = cell
        return 0 <= r < h and 0 <= c < w

    def erase(cell: tuple[int, int]) -> None:
        painted_dir.pop(cell, None)
        out[cell[0], cell[1]] = bg

    def paint(cell: tuple[int, int], color: int, direction: tuple[int, int]) -> None:
        out[cell[0], cell[1]] = int(color)
        painted_dir[cell] = direction

    for bid, (color, pos, direction) in enumerate(specs):
        beams[bid] = _Beam(bid, color, pos, direction)
        paint(pos, color, direction)

    max_iters = h * w * 10
    for _ in range(max_iters):
        prop: dict[int, tuple[int, int]] = {}
        for bid, beam in beams.items():
            if not beam.active:
                continue
            nxt = (beam.pos[0] + beam.dir[0], beam.pos[1] + beam.dir[1])
            if not in_bounds(nxt):
                beam.active = False
                continue
            if int(g[nxt[0], nxt[1]]) != bg:
                beam.active = False
                continue
            prop[bid] = nxt

        if not prop:
            break

        cell_to_bids: dict[tuple[int, int], list[int]] = {}
        for bid, cell in prop.items():
            cell_to_bids.setdefault(cell, []).append(bid)

        progress = False
        for cell, bidders in sorted(cell_to_bids.items()):
            axes = {_axis(beams[bid].dir) for bid in bidders}
            if len(bidders) >= 2 and len(axes) >= 2:
                if cell in painted_dir:
                    erase(cell)
                blocked.add(cell)
                cross_same_tick += 1
                for bid in bidders:
                    beams[bid].pos = cell
                    beams[bid].active = True
                    progress = True
                continue

            if len(bidders) >= 2:
                for bid in bidders:
                    beams[bid].active = False
                continue

            bid = bidders[0]
            beam = beams[bid]
            direction = beam.dir

            if cell in blocked:
                beam.pos = cell
                progress = True
                continue

            if cell not in painted_dir:
                paint(cell, beam.color, direction)
                beam.pos = cell
                progress = True
                continue

            prev_dir = painted_dir[cell]
            if _axis(prev_dir) != _axis(direction):
                erase(cell)
                blocked.add(cell)
                cross_with_painted += 1
                beam.pos = cell
                progress = True
                continue

            beam.active = False

        if not progress:
            break
    else:
        raise AssertionError("straight-ray simulation did not converge")

    stats = {
        "beam_count": len(specs),
        "blocked_crossings": len(blocked),
        "same_tick_crossings": cross_same_tick,
        "retro_crossings": cross_with_painted,
    }
    return out, stats


def main(grid: np.ndarray) -> np.ndarray:
    """
    Emit straight orthogonal rays from framed 2x2 blocks through side gaps, leaving holes where rays cross.

    Input structure:
    - bg is the dominant color. Each colored 2x2 block is enclosed by a shared frame color.
    - Only side gaps on the 4x4 frame border are meaningful in this variant; each such gap is a launch point.

    Rule:
    - A gap on the top, bottom, left, or right side emits one ray of the block's color in that outward cardinal direction.
    - Rays extend cell by cell through background until they leave the grid or meet a non-background frame/block pixel.
    - When a horizontal ray and a vertical ray meet, the intersection cell is cleared back to background and both rays keep moving straight through that 1-cell hole.

    Key insight:
    - The source task's distinctive bends are replaced by pure row/column propagation: openings encode orthogonal direction, and collisions are read as blank-centered crossings rather than turns.
    """
    out, _stats = _simulate(np.asarray(grid, dtype=int))
    return out


# --------------- generator (o5x) ---------------

BLOCK_COUNT_RANGE = (2, 6)
GRID_SIZE_MAX = 31
GRID_SIZE_MIN_BY_BLOCKS = {2: 12, 4: 18}
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
MIN_REQUIRED_CROSSINGS = 1
GEN_RETRIES = 500
GAP_PICK_RETRIES = 160


_BORDER_ORDER = (
    (0, 0), (0, 1), (0, 2), (0, 3),
    (1, 3), (2, 3), (3, 3), (3, 2),
    (3, 1), (3, 0), (2, 0), (1, 0),
)
_BORDER_INDEX = {p: i for i, p in enumerate(_BORDER_ORDER)}


def _gap_combos(size: int) -> list[tuple[tuple[int, int], ...]]:
    combos = []
    for combo in itertools.combinations(_SIDE_GAPS, size):
        idxs = sorted(_BORDER_INDEX[p] for p in combo)
        if any((idxs[(i + 1) % size] - idxs[i]) % 12 == 1 for i in range(size)):
            continue
        combos.append(combo)
    return combos


_GAP_COMBOS = {n: _gap_combos(n) for n in range(1, 5)}


def _gap_dir(gap: tuple[int, int]) -> tuple[int, int]:
    gr, gc = gap
    if gr == 0:
        return (-1, 0)
    if gr == 3:
        return (1, 0)
    if gc == 0:
        return (0, -1)
    if gc == 3:
        return (0, 1)
    raise AssertionError("expected side gap")


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


def _clear_gap_combos(
    obstacles: np.ndarray,
    bg: int,
    r0: int,
    c0: int,
) -> dict[int, list[tuple[tuple[int, int], ...]]]:
    clear: dict[int, list[tuple[tuple[int, int], ...]]] = {}
    for gap_count, combos in _GAP_COMBOS.items():
        for combo in combos:
            ok = True
            for gr, gc in combo:
                ray = _ray_cells(obstacles, bg, (r0 + gr, c0 + gc), _gap_dir((gr, gc)))
                if ray is None or len(ray) - 1 < MIN_RAY_STEPS:
                    ok = False
                    break
            if ok:
                clear.setdefault(gap_count, []).append(combo)
    return clear


def _pick_colors(rng: np.random.Generator, block_count: int) -> tuple[int, int, list[int]]:
    bg = int(rng.choice(BG_CHOICES, p=BG_WEIGHTS))
    frame_color = FRAME_COLOR_FOR_BG[bg]
    palette = [c for c in PALETTE_FOR_BG[bg] if c != frame_color]
    block_colors = list(rng.choice(palette, size=block_count, replace=False))
    return bg, frame_color, block_colors


def _grid_size_min(block_count: int) -> int:
    result = 10
    for threshold, min_n in sorted(GRID_SIZE_MIN_BY_BLOCKS.items()):
        if block_count >= threshold:
            result = min_n
    return result


def _evaluate_layout(
    obstacles: np.ndarray,
    bg: int,
    positions: list[tuple[int, int]],
    chosen_gaps: list[tuple[tuple[int, int], ...]],
) -> dict[str, int] | None:
    rays: list[dict[str, object]] = []
    seeds: set[tuple[int, int]] = set()
    rows_h: set[int] = set()
    cols_v: set[int] = set()

    for (r0, c0), gaps in zip(positions, chosen_gaps):
        for gap in gaps:
            seed = (r0 + gap[0], c0 + gap[1])
            direction = _gap_dir(gap)
            cells = _ray_cells(obstacles, bg, seed, direction)
            if cells is None or len(cells) - 1 < MIN_RAY_STEPS:
                return None
            if seed in seeds:
                return None
            seeds.add(seed)
            if _axis(direction) == "h":
                if seed[0] in rows_h:
                    return None
                rows_h.add(seed[0])
            else:
                if seed[1] in cols_v:
                    return None
                cols_v.add(seed[1])
            rays.append(
                {
                    "seed": seed,
                    "dir": direction,
                    "cells": cells,
                    "set": set(cells),
                }
            )

    if not rows_h or not cols_v:
        return None

    for ray in rays:
        path_tail = set(ray["cells"][1:])
        if path_tail & (seeds - {ray["seed"]}):
            return None

    crossings = 0
    for hray in rays:
        if _axis(hray["dir"]) != "h":
            continue
        for vray in rays:
            if _axis(vray["dir"]) != "v":
                continue
            cell = (hray["seed"][0], vray["seed"][1])
            if cell in seeds:
                continue
            if cell not in hray["set"] or cell not in vray["set"]:
                continue
            if int(obstacles[cell[0], cell[1]]) != bg:
                continue
            crossings += 1

    if crossings < MIN_REQUIRED_CROSSINGS:
        return None

    return {
        "ray_count": len(rays),
        "crossings": crossings,
    }


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

    available_per_block = []
    for r0, c0 in positions:
        available = _clear_gap_combos(obstacles, bg, r0, c0)
        if not available:
            return None
        available_per_block.append(available)

    chosen_gaps: list[tuple[tuple[int, int], ...]] | None = None
    layout_stats: dict[str, int] | None = None
    for _ in range(GAP_PICK_RETRIES):
        proposal: list[tuple[tuple[int, int], ...]] = []
        for available in available_per_block:
            gap_count = int(rng.choice(sorted(available)))
            combos = available[gap_count]
            proposal.append(combos[int(rng.integers(len(combos)))])
        layout = _evaluate_layout(obstacles, bg, positions, proposal)
        if layout is None:
            continue
        chosen_gaps = proposal
        layout_stats = layout
        break

    if chosen_gaps is None or layout_stats is None:
        return None

    grid = np.full((n, n), bg, dtype=int)
    gap_total = 0
    for (r0, c0), color, gaps in zip(positions, block_colors, chosen_gaps):
        gap_total += len(gaps)
        _draw_framed_block(grid, r0, c0, bg, frame_color, int(color), gaps)

    if int(np.bincount(grid.ravel(), minlength=10).argmax()) != bg:
        return None

    try:
        out, stats = _simulate(grid)
    except Exception:
        return None

    if stats["blocked_crossings"] < MIN_REQUIRED_CROSSINGS:
        return None
    if stats["beam_count"] != gap_total:
        return None
    if np.count_nonzero(out != grid) < gap_total * MIN_BEAM_PIXELS_PER_GAP:
        return None
    return grid


def generate_input() -> np.ndarray:
    for _ in range(GEN_RETRIES):
        grid = _gen_attempt()
        if grid is not None:
            return grid
    raise RuntimeError("failed to generate a valid e12f9a14 straight-ray input")
