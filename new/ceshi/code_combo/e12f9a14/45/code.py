# Combo variant of e12f9a14 (index 45)
# combo_spec: D1/0+D10/0
# Name: 方向正交映射+正交射线

import itertools

import numpy as np

from arc_utils import get_bg, paste


# Design analysis for this combo variant
# --------------------------------------
# concretization:
#   The source task still uses the same visual vocabulary: a shared-color 4x4 frame
#   around each monochrome 2x2 block, with background-colored openings on the frame
#   border. The fusion changes only the beam geometry. We keep only side openings,
#   because once D1 turns motion into orthogonal rays, corner openings would become
#   ambiguous under the D10 "orthogonal to the gap" remapping. A top or bottom gap
#   now emits a horizontal ray; a left or right gap emits a vertical ray. Because a
#   straight tangent ray cannot continue from the border cell without re-entering the
#   same frame, the painted ray starts one cell outside the gap and then runs along the
#   exterior strip adjacent to that side.
# behavior trace:
#   Original behavior on a top gap: the beam exits upward on a diagonal chosen by the
#   opening's relation to the block, and later collisions may bend that beam again.
#   Fused behavior on a top gap at local (0,1): the beam exits to the cell just above
#   the gap, and from there travels left along that outside row instead of upward. A
#   top gap at local (0,2) travels right. Likewise, a left-side gap at local (1,0)
#   exits to the cell just left of the frame and then travels up, while (2,0) travels
#   down. When one horizontal and one vertical ray meet, the crossing cell is cleared
#   to background and both continue straight, producing a blank-centered plus.
#   The visible difference from the source task is therefore twofold: no diagonals, and
#   no "gap side equals travel side" mapping.
# complexity calibration:
#   If every puzzle used only one orthogonal sign per axis, a solver could mistake the
#   rule for a simpler "all top gaps go left" or "all side gaps go outward" heuristic.
#   The generator therefore requires all four travel directions (left/right/up/down) to
#   appear in each instance, plus at least one horizontal/vertical crossing. That keeps
#   the orthogonal remapping observable without reintroducing the source task's heavy
#   collision-state complexity.
# implicit assumption audit:
#   The original solver depended on three assumptions that no longer hold:
#   - beam direction lived in 8-way space and could begin diagonally;
#   - 45-degree and 90-degree collisions created new direction patterns;
#   - truncation/ownership logic was needed because collisions redirected beams.
#   In this combo, direction is always one of four cardinals, determined by the gap's
#   slot along its side rather than by the side's outward normal. Collisions no longer
#   turn beams; they only punch a 1-cell hole at perpendicular crossings. This lets the
#   solver collapse to fixed-direction propagation with blocked crossing cells.
# placement legality:
#   The generator must guarantee that every chosen side gap has a clear orthogonal path
#   to the grid edge, that no two horizontal rays share a row, and that no two vertical
#   rays share a column. Otherwise same-axis overlaps or head-on clashes would create
#   unsupported cases. It must also forbid any ray from running through another seed.
# visual observability:
#   Each instance contains both horizontal and vertical rays, all four signed directions,
#   and at least one crossing. A human comparing input/output can therefore see that top
#   and bottom openings do not fire outward, because the colored marks appear on the
#   outside strip and extend sideways; similarly left/right openings create vertical
#   marks on the adjacent exterior column. The repeated left/right and up/down examples
#   provide enough support to infer the slot-based sign choice.
# visual cleanliness:
#   Frames keep >=2 cells of clearance, crossings happen only on background cells, and
#   same-axis sharing is rejected. Outputs stay as sparse rows/columns with occasional
#   blank centers, rather than dense overlapping traces that would obscure the mapping.
# color inference:
#   Colors remain purely object-local: each ray uses the color of its own 2x2 block.
#   There is no palette-dependent remapping; only geometry changes.
# new solver assumptions checked against generator:
#   - every emitted gap is a structural side gap in one of the eight side slots;
#   - all four ray directions appear at least once per generated puzzle;
#   - every perpendicular crossing lands on background and never on a seed cell;
#   - no same-axis row/column duplication occurs among emitted rays;
#   - every emitted ray adds at least a few visible cells beyond its seed.


_SIDE_GAPS = (
    (0, 1), (0, 2),
    (1, 3), (2, 3),
    (3, 2), (3, 1),
    (2, 0), (1, 0),
)

_BORDER_ORDER = (
    (0, 0), (0, 1), (0, 2), (0, 3),
    (1, 3), (2, 3), (3, 3), (3, 2),
    (3, 1), (3, 0), (2, 0), (1, 0),
)
_BORDER_INDEX = {p: i for i, p in enumerate(_BORDER_ORDER)}


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


def _orth_gap_dir_local(gap: tuple[int, int]) -> tuple[int, int]:
    gr, gc = gap
    if gr in (0, 3):
        assert gc in (1, 2), "expected top/bottom side gap"
        return (0, -1 if gc == 1 else 1)
    assert gc in (0, 3) and gr in (1, 2), "expected left/right side gap"
    return (-1 if gr == 1 else 1, 0)


def _orth_gap_dir_for_block(br: int, bc: int, gr: int, gc: int) -> tuple[int, int] | None:
    local = (gr - (br - 1), gc - (bc - 1))
    if local not in _SIDE_GAPS:
        return None
    return _orth_gap_dir_local(local)


def _launch_from_gap(br: int, bc: int, gr: int, gc: int) -> tuple[tuple[int, int], tuple[int, int]] | None:
    local = (gr - (br - 1), gc - (bc - 1))
    if local not in _SIDE_GAPS:
        return None
    direction = _orth_gap_dir_local(local)
    if local[0] == 0:
        start = (gr - 1, gc)
    elif local[0] == 3:
        start = (gr + 1, gc)
    elif local[1] == 0:
        start = (gr, gc - 1)
    else:
        start = (gr, gc + 1)
    return start, direction


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
            launch = _launch_from_gap(br, bc, gr, gc)
            if launch is None:
                continue
            start, direction = launch
            specs.append((col, start, direction))
    return bg, specs


def _simulate(grid: np.ndarray) -> tuple[np.ndarray, dict[str, int]]:
    g = np.asarray(grid, dtype=int)
    h, w = g.shape
    bg, specs = _collect_beam_specs(g)
    assert specs, "expected at least one orthogonally-mapped side-gap beam"

    out = g.copy()
    beams: dict[int, _Beam] = {}
    painted_dir: dict[tuple[int, int], tuple[int, int]] = {}
    blocked: set[tuple[int, int]] = set()

    cross_same_tick = 0
    cross_with_painted = 0
    dir_counts = {"left": 0, "right": 0, "up": 0, "down": 0}

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
        if direction == (0, -1):
            dir_counts["left"] += 1
        elif direction == (0, 1):
            dir_counts["right"] += 1
        elif direction == (-1, 0):
            dir_counts["up"] += 1
        else:
            dir_counts["down"] += 1

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
        raise AssertionError("orthogonal-ray simulation did not converge")

    stats = {
        "beam_count": len(specs),
        "blocked_crossings": len(blocked),
        "same_tick_crossings": cross_same_tick,
        "retro_crossings": cross_with_painted,
        "left_count": dir_counts["left"],
        "right_count": dir_counts["right"],
        "up_count": dir_counts["up"],
        "down_count": dir_counts["down"],
    }
    return out, stats


def main(grid: np.ndarray) -> np.ndarray:
    """
    Emit straight orthogonal rays from framed 2x2 blocks, using the gap's along-the-side slot instead of its outward face.

    Input structure:
    - The background is the dominant color. Each foreground object is a monochrome 2x2 block enclosed by one shared 4x4 frame color.
    - Meaningful openings are side gaps on that frame border; their slot along the side matters.

    Rule:
    - A top or bottom gap launches a horizontal ray of the block's color on the exterior row just outside that side: the left slot goes left, the right slot goes right.
    - A left or right gap launches a vertical ray on the exterior column just outside that side: the upper slot goes up, the lower slot goes down.
    - Rays extend cell by cell through background until they leave the grid or meet a frame/block pixel. When a horizontal ray and a vertical ray meet, the crossing cell is reset to background and both rays continue straight through that 1-cell hole.

    Key insight:
    - This fusion changes both the path family and the direction cue: the beams are orthogonal, and the frame side tells you the axis but not the travel sign. The sign comes from the gap's offset along that side, while the painted path lives just outside the frame rather than on the gap cell itself.
    """
    out, _stats = _simulate(np.asarray(grid, dtype=int))
    return out


# --------------- generator (o5x) ---------------

BLOCK_COUNT_RANGE = (3, 6)
GRID_SIZE_MAX = 31
GRID_SIZE_MIN_BY_BLOCKS = {3: 14, 5: 20}
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
GEN_RETRIES = 600
GAP_PICK_RETRIES = 320


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
    return _orth_gap_dir_local(gap)


def _launch_from_local_gap(r0: int, c0: int, gap: tuple[int, int]) -> tuple[tuple[int, int], tuple[int, int]]:
    gr, gc = gap
    direction = _gap_dir(gap)
    if gr == 0:
        start = (r0 - 1, c0 + gc)
    elif gr == 3:
        start = (r0 + 4, c0 + gc)
    elif gc == 0:
        start = (r0 + gr, c0 - 1)
    else:
        start = (r0 + gr, c0 + 4)
    return start, direction


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
                start, direction = _launch_from_local_gap(r0, c0, (gr, gc))
                if int(obstacles[start[0], start[1]]) != bg:
                    ok = False
                    break
                ray = _ray_cells(obstacles, bg, start, direction)
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
    result = 12
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
    dir_counts = {"left": 0, "right": 0, "up": 0, "down": 0}

    for (r0, c0), gaps in zip(positions, chosen_gaps):
        for gap in gaps:
            seed, direction = _launch_from_local_gap(r0, c0, gap)
            if int(obstacles[seed[0], seed[1]]) != bg:
                return None
            cells = _ray_cells(obstacles, bg, seed, direction)
            if cells is None or len(cells) - 1 < MIN_RAY_STEPS:
                return None
            if seed in seeds:
                return None
            seeds.add(seed)
            if direction == (0, -1):
                dir_counts["left"] += 1
                if seed[0] in rows_h:
                    return None
                rows_h.add(seed[0])
            elif direction == (0, 1):
                dir_counts["right"] += 1
                if seed[0] in rows_h:
                    return None
                rows_h.add(seed[0])
            elif direction == (-1, 0):
                dir_counts["up"] += 1
                if seed[1] in cols_v:
                    return None
                cols_v.add(seed[1])
            else:
                dir_counts["down"] += 1
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

    if min(dir_counts.values()) == 0:
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
        "left_count": dir_counts["left"],
        "right_count": dir_counts["right"],
        "up_count": dir_counts["up"],
        "down_count": dir_counts["down"],
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
    if min(stats["left_count"], stats["right_count"], stats["up_count"], stats["down_count"]) == 0:
        return None
    if np.count_nonzero(out != grid) < gap_total * MIN_BEAM_PIXELS_PER_GAP:
        return None
    return grid


def generate_input() -> np.ndarray:
    for _ in range(GEN_RETRIES):
        grid = _gen_attempt()
        if grid is not None:
            return grid
    raise RuntimeError("failed to generate a valid e12f9a14 orthogonal-mapping input")
