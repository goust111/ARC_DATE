# Combo variant of e12f9a14 (index 2)
# combo_spec: D1/0+D3/0
# Name: 正交射线+周期涂色

import itertools

import numpy as np

from arc_utils import get_bg, paste


# Design analysis for the combo variant
# -------------------------------------
# a) concretization_and_behavior_trace:
#    - Keep the original object vocabulary: monochrome 2x2 blocks inside shared-color
#      4x4 frames.
#    - Apply D1/0 first: only side gaps are meaningful, so every emitted path is a
#      straight orthogonal ray. Top/bottom gaps launch vertical rays; left/right gaps
#      launch horizontal rays. Corner gaps are ignored by the solver and never generated.
#    - Apply D3/0 on top of that path geometry: each ray paints an AABB cycle where
#      A = block color and B = frame color.
#    - Small trace:
#      * source task: a corner opening could emit a diagonal solid-color beam and later
#        bend after a collision.
#      * combo task: a top-side gap emits upward cells with colors A, A, B, B, A, A...
#      * if a horizontal and vertical ray cross, the crossing cell is left blank and
#        both rays continue straight. That blank cell does not advance either beam's
#        color phase, so the visible stripe sequence resumes as if the hole were skipped.
#    - The main visible difference from the source task is therefore twofold at once:
#      orthogonal geometry replaces diagonal motion, and solid trails become striped.
# b) complexity_calibration:
#    - The color cycle is fixed globally to AABB. This is small enough to infer from
#      a few examples, but visually different enough from the source task's solid trails.
#    - The generator requires at least one horizontal/vertical crossing and enough long
#      rays to expose frame-colored trail pixels, so the variant cannot collapse into a
#      trivial no-collision or single-color scene.
# c) implicit_assumption_audit:
#    - Original main() depended on 8-direction motion, 45/90-degree collision classes,
#      truncation, and redirection. None of those assumptions survive D1/0.
#    - Once paths are purely orthogonal and never turn, the robust way to preserve D3/0
#      phase semantics is to compute each full ray first, then blank all crossings, then
#      color the remaining cells in path order. This avoids retroactive phase bugs where
#      a later crossing would otherwise need to recolor an already-painted prefix.
#    - New solver assumptions:
#      * every used opening is a structural side gap;
#      * no two horizontal rays share a row and no two vertical rays share a column;
#      * ray tails do not pass through another seed gap;
#      * every crossing is exactly one horizontal ray with one vertical ray on bg.
# d) placement_legality:
#    - The generator places non-overlapping 4x4 frames with >=2-cell clearance, keeps
#      them away from borders, and validates every candidate gap against an obstacle grid
#      containing the full closed frames.
#    - This guarantees rays never tunnel through another frame and every launch has
#      enough straight runway.
# e) visual_observability:
#    - A solver can read direction from gap orientation and read the second stripe color
#      directly from the frame color already present around the block.
#    - Requiring crossings plus visible frame-colored trail pixels makes both modified
#      dimensions observable in 2-3 examples.
# f) visual_cleanliness:
#    - Unique horizontal rows and unique vertical columns prevent same-axis overpainting.
#    - Blank-centered orthogonal crossings keep intersections readable; there is no
#      multi-rule bend clutter from the source task.
# g) change_inferability:
#    - No hidden palette lookup is introduced: the periodic colors are exactly the two
#      colors already adjacent in each source object.
#    - The generator checks that multiple beams actually show frame-colored trail cells,
#      so the AABB rule is inferable rather than merely theoretically present.


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


def _gap_dir_for_block(br: int, bc: int, gr: int, gc: int) -> tuple[int, int] | None:
    dr = -1 if gr == br - 1 else (1 if gr == br + 2 else 0)
    dc = -1 if gc == bc - 1 else (1 if gc == bc + 2 else 0)
    if dr == 0 and dc == 0:
        return None
    if dr != 0 and dc != 0:
        return None
    return dr, dc


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


def _axis(direction: tuple[int, int]) -> str:
    return "h" if direction[0] == 0 else "v"


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


def _trail_color(source_color: int, frame_color: int, visible_idx: int) -> int:
    return int(source_color if visible_idx % 4 < 2 else frame_color)


def _simulate(grid: np.ndarray, collect_stats: bool = False) -> tuple[np.ndarray, dict[str, int]]:
    g = np.asarray(grid, dtype=int)
    bg = int(get_bg(g))
    out = g.copy()

    blocks = _find_blocks_2x2(g, bg)
    assert blocks, "expected at least one 2x2 block in input"

    frame_color_global: int | None = None
    beams: list[dict[str, object]] = []
    source_colors: set[int] = set()

    for block_color, br, bc in blocks:
        source_colors.add(int(block_color))
        frame_color = _infer_frame_color(g, bg, br, bc, block_color)
        if frame_color_global is None:
            frame_color_global = frame_color
        else:
            assert frame_color_global == frame_color, "expected one shared frame color"
        for gr, gc in _frame_gaps_for_block(g, bg, frame_color, br, bc):
            direction = _gap_dir_for_block(br, bc, gr, gc)
            if direction is None:
                continue
            seed = (gr, gc)
            path = _ray_cells(g, bg, seed, direction)
            assert path is not None, "expected orthogonal ray runway to stay on background"
            beams.append(
                {
                    "source_color": int(block_color),
                    "frame_color": int(frame_color),
                    "seed": seed,
                    "dir": direction,
                    "axis": _axis(direction),
                    "path": path,
                }
            )

    assert beams, "expected at least one side-gap beam"

    cell_to_bids: dict[tuple[int, int], list[int]] = {}
    seed_cells = {beam["seed"] for beam in beams}
    for bid, beam in enumerate(beams):
        for cell in beam["path"]:
            cell_to_bids.setdefault(cell, []).append(bid)

    crossings: set[tuple[int, int]] = set()
    cross_count_per_beam = [0] * len(beams)
    for cell, bidders in cell_to_bids.items():
        axes = {str(beams[bid]["axis"]) for bid in bidders}
        if len(axes) == 1:
            assert len(bidders) == 1, "expected no same-axis ray overlaps"
            continue
        assert len(axes) == 2 and len(bidders) == 2, "expected each crossing to involve one horizontal and one vertical ray"
        assert cell not in seed_cells, "expected crossings away from seed gaps"
        crossings.add(cell)
        for bid in bidders:
            cross_count_per_beam[bid] += 1

    stats = {
        "beam_count": len(beams),
        "blocked_crossings": len(crossings),
        "beam_pixels": 0,
        "frame_trail_pixels": 0,
        "source_trail_pixels": 0,
        "striped_beams": 0,
        "full_pattern_beams": 0,
    }

    for bid, beam in enumerate(beams):
        visible_idx = 0
        used_frame = False
        path = beam["path"]
        assert isinstance(path, list)
        for cell in path:
            r, c = cell
            if cell in crossings:
                out[r, c] = bg
                continue
            color = _trail_color(int(beam["source_color"]), int(beam["frame_color"]), visible_idx)
            out[r, c] = color
            if color == int(beam["frame_color"]):
                stats["frame_trail_pixels"] += 1
                used_frame = True
            else:
                stats["source_trail_pixels"] += 1
            visible_idx += 1
        if used_frame:
            stats["striped_beams"] += 1
        if visible_idx >= 4:
            stats["full_pattern_beams"] += 1
        assert visible_idx == len(path) - cross_count_per_beam[bid], "crossing holes must be the only skipped phase slots"

    if collect_stats:
        painted_mask = (g == bg) & (out != bg)
        stats["beam_pixels"] = int(np.count_nonzero(painted_mask))
        if frame_color_global is not None:
            stats["frame_trail_pixels"] = int(np.count_nonzero(painted_mask & (out == int(frame_color_global))))
        source_mask = np.isin(out, list(source_colors))
        stats["source_trail_pixels"] = int(np.count_nonzero(painted_mask & source_mask))
        striped = 0
        full_pattern = 0
        for bid, beam in enumerate(beams):
            visible_len = len(beam["path"]) - cross_count_per_beam[bid]
            if visible_len >= 3:
                striped += 1
            if visible_len >= 4:
                full_pattern += 1
        stats["striped_beams"] = striped
        stats["full_pattern_beams"] = full_pattern
    else:
        stats["beam_pixels"] = int(np.count_nonzero((g == bg) & (out != bg)))

    return out, stats


def main(grid: np.ndarray) -> np.ndarray:
    """
    Emit straight orthogonal rays from framed 2x2 blocks through side gaps, and color each ray with a repeating two-color stripe pattern.

    Input structure:
    - The dominant color is background. Each source is a monochrome 2x2 block enclosed by one shared 4x4 frame color.
    - Only structural side gaps on the frame border matter in this combo; they define both launch points and outward cardinal directions.

    Rule:
    - A top, bottom, left, or right gap emits one straight ray of background-overwriting cells along that outward row or column.
    - The ray is not solid: along each beam, visible painted cells follow an AABB cycle where A is the inner block color and B is the frame color.
    - When a horizontal ray and a vertical ray cross, the meeting cell stays background and both rays continue straight; that blank crossing cell does not consume the stripe phase for either ray.

    Key insight:
    - The puzzle now asks for two linked inferences at once: openings encode orthogonal motion instead of diagonal motion, and the output stripes alternate between the two colors already present in each framed source object.
    """
    return _simulate(np.asarray(grid, dtype=int), collect_stats=False)[0]


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
MIN_RAY_STEPS = 4
MIN_VISIBLE_LEN_PER_BEAM = 3
MIN_BEAM_PIXELS_PER_GAP = 4
MIN_REQUIRED_CROSSINGS = 1
MIN_FRAME_TRAIL_PIXELS = 6
MIN_STRIPED_BEAMS = 2
MIN_FULL_PATTERN_BEAMS = 2
GEN_RETRIES = 600
GAP_PICK_RETRIES = 240


def _gap_combos(size: int) -> list[tuple[tuple[int, int], ...]]:
    combos = []
    for combo in itertools.combinations(_SIDE_GAPS, size):
        idxs = sorted(_BORDER_INDEX[p] for p in combo)
        if any((idxs[(i + 1) % size] - idxs[i]) % 12 == 1 for i in range(size)):
            continue
        combos.append(combo)
    return combos


_GAP_COMBOS = {n: _gap_combos(n) for n in range(1, 5)}


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
            for gap in combo:
                ray = _ray_cells(obstacles, bg, (r0 + gap[0], c0 + gap[1]), _gap_dir(gap))
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


def _pick_gap_count(available_counts: list[int], rng: np.random.Generator) -> int:
    weighted: list[int] = []
    for gap_count in available_counts:
        weighted.extend([gap_count] * gap_count)
    return int(weighted[int(rng.integers(len(weighted)))])


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
                    "axis": _axis(direction),
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

    crossings: set[tuple[int, int]] = set()
    cross_count_per_beam = [0] * len(rays)
    for hidx, hray in enumerate(rays):
        if hray["axis"] != "h":
            continue
        for vidx, vray in enumerate(rays):
            if vray["axis"] != "v":
                continue
            cell = (int(hray["seed"][0]), int(vray["seed"][1]))
            if cell in seeds:
                continue
            if cell not in hray["set"] or cell not in vray["set"]:
                continue
            if int(obstacles[cell[0], cell[1]]) != bg:
                continue
            crossings.add(cell)
            cross_count_per_beam[hidx] += 1
            cross_count_per_beam[vidx] += 1

    if len(crossings) < MIN_REQUIRED_CROSSINGS:
        return None

    striped_beams = 0
    full_pattern_beams = 0
    frame_pixels = 0
    for bid, ray in enumerate(rays):
        visible_len = len(ray["cells"]) - cross_count_per_beam[bid]
        if visible_len < MIN_VISIBLE_LEN_PER_BEAM:
            return None
        local_frame_pixels = sum(1 for idx in range(visible_len) if idx % 4 >= 2)
        if local_frame_pixels > 0:
            striped_beams += 1
        if visible_len >= 4:
            full_pattern_beams += 1
        frame_pixels += local_frame_pixels

    if striped_beams < MIN_STRIPED_BEAMS:
        return None
    if full_pattern_beams < MIN_FULL_PATTERN_BEAMS:
        return None
    if frame_pixels < MIN_FRAME_TRAIL_PIXELS:
        return None

    return {
        "ray_count": len(rays),
        "crossings": len(crossings),
        "striped_beams": striped_beams,
        "full_pattern_beams": full_pattern_beams,
        "frame_pixels": frame_pixels,
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
            gap_count = _pick_gap_count(sorted(available), rng)
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
        out, stats = _simulate(grid, collect_stats=True)
    except Exception:
        return None

    if stats["blocked_crossings"] < MIN_REQUIRED_CROSSINGS:
        return None
    if stats["beam_count"] != gap_total:
        return None
    if stats["beam_pixels"] < gap_total * MIN_BEAM_PIXELS_PER_GAP:
        return None
    if stats["frame_trail_pixels"] < MIN_FRAME_TRAIL_PIXELS:
        return None
    if stats["striped_beams"] < MIN_STRIPED_BEAMS:
        return None
    if stats["full_pattern_beams"] < MIN_FULL_PATTERN_BEAMS:
        return None
    if np.array_equal(out, grid):
        return None
    return grid


def generate_input() -> np.ndarray:
    for _ in range(GEN_RETRIES):
        grid = _gen_attempt()
        if grid is not None:
            return grid
    raise RuntimeError("failed to generate a valid e12f9a14 combo input")
