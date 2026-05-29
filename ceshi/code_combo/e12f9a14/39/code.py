# Combo variant of e12f9a14 (index 39)
# combo_spec: D6/2+D11/0
# Name: 目标标记终止+源目标配对

import numpy as np

from arc_utils import get_bg, paste


# Design analysis for the combo variant
# -------------------------------------
# concreteization:
# - Each logical object now appears as a same-color pair of framed 2x2 blocks.
# - Both frames keep exactly one structural gap. The plain frame is the source.
#   The marked frame is the target: it has the same-color 2x2 interior, one gap
#   on the side facing the incoming ray, and a 1x1 marker pixel placed one step
#   outside that gap.
# - Small trace example:
#   original behavior on two 1-gap frames would emit from both gaps.
#   combo behavior on a red pair:
#   source frame has an east gap; target frame has a west gap plus a yellow marker
#   outside that west gap. The red ray grows east from the source, paints until the
#   next step would hit the yellow marker, and then stops. The target frame does not
#   emit even though it also has a gap.
#
# complexity calibration:
# - Pair count is kept at 2-4 so every sample contains repeated evidence for the
#   "match source and target by inner block color" rule.
# - Every pair color is unique, but each color appears twice as a framed 2x2 block,
#   which avoids ambiguous many-to-one matching while still forcing pairing.
# - A single shared marker color is used for all targets. This prevents the task
#   from collapsing into the D6 single-variant "follow the same-colored singleton".
# - Gaps are restricted to one canonical position per direction, so the visible
#   degrees of freedom are direction, distance, and pairing rather than noisy border variants.
#
# implicit assumption review vs. the original main():
# - Original code assumes every structural gap launches a beam. That is false now:
#   target frames also have a gap, but a marker adjacent to that gap reclassifies the
#   frame as a receiver, not a second emitter.
# - Original code treats a non-background cell in front of a beam as a generic stop
#   reason. Here the solver must verify that the stop cell is the marker attached to
#   the matched target frame, not merely "the first obstacle in the world".
# - Original collision bookkeeping assumes multiple independently emitting frames.
#   The generator now forbids beam-beam interactions, so the solver can require
#   disjoint straight corridors and remove collision machinery entirely.
#
# placement legality:
# - The generator constructs each pair as one plan containing source frame, target
#   frame, target marker, and the entire painted corridor.
# - Plans are rejected if any visible cells overlap, if any 4x4 frame boxes violate
#   the minimum separation, or if two diagonal unit segments would geometrically cross
#   inside the same 2x2 square.
# - The target frame always places its gap opposite the source direction, with the
#   marker immediately outside that gap, so the endpoint geometry is physically valid.
#
# visual observability:
# - Human solvers can see two frame roles: source = lone gap, target = gap with a
#   shared marker color attached just outside it.
# - Because the marker color is reused across all targets, the only stable pairing cue
#   is the repeated interior 2x2 color shared by one source frame and one target frame.
# - The output visibly differs from the original task because marked target gaps do
#   not emit; only the unmarked partner does.
#
# visual cleanliness:
# - Straight corridors are kept disjoint and collision-free.
# - Markers sit adjacent to target gaps rather than floating elsewhere, so endpoint
#   semantics remain local and easy to read.
#
# color inference:
# - No recoloring occurs in the output. Beam color is still the source block color.
# - The new color relationship is structural: block color identifies the source/target
#   pair, while one distinct shared marker color identifies all receiver endpoints.


_DIR_TO_GAP = {
    (-1, -1): (0, 0),
    (-1, 0): (0, 1),
    (-1, 1): (0, 3),
    (0, 1): (1, 3),
    (1, 1): (3, 3),
    (1, 0): (3, 1),
    (1, -1): (3, 0),
    (0, -1): (1, 0),
}
_DIRS = tuple(_DIR_TO_GAP)


def _opp(d: tuple[int, int]) -> tuple[int, int]:
    return (-d[0], -d[1])


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
    assert len(frame_colors) == 1, "expected one frame color around each 2x2 block"
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
        assert dr != 0 or dc != 0, "expected gap on a frame border"
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
                and 0 <= c - dc < w
                and int(g[r - dr, c]) == wall
                and int(g[r, c - dc]) == wall
            )
        if ok:
            out.append((r, c))
    return out


def _gap_dir_from_abs(br: int, bc: int, gr: int, gc: int) -> tuple[int, int]:
    dr = -1 if gr == br - 1 else (1 if gr == br + 2 else 0)
    dc = -1 if gc == bc - 1 else (1 if gc == bc + 2 else 0)
    d = (dr, dc)
    assert d in _DIR_TO_GAP, "expected a canonical single-gap direction"
    return d


def _frame_bbox_cells(r0: int, c0: int) -> set[tuple[int, int]]:
    return {(r, c) for r in range(r0, r0 + 4) for c in range(c0, c0 + 4)}


def _is_diag_segment(seg: tuple[tuple[int, int], tuple[int, int]]) -> bool:
    (r0, c0), (r1, c1) = seg
    return abs(r1 - r0) == 1 and abs(c1 - c0) == 1


def _diag_signature(seg: tuple[tuple[int, int], tuple[int, int]]) -> tuple[tuple[int, int], int] | None:
    if not _is_diag_segment(seg):
        return None
    (r0, c0), (r1, c1) = seg
    slope = (r1 - r0) * (c1 - c0)
    mid = (r0 + r1, c0 + c1)
    return mid, slope


def _marker_from_target_frame(r0: int, c0: int, source_dir: tuple[int, int]) -> tuple[int, int]:
    td = _opp(source_dir)
    gap = _DIR_TO_GAP[td]
    return (r0 + gap[0] + td[0], c0 + gap[1] + td[1])


def main(grid: np.ndarray) -> np.ndarray:
    """
    Draw one straight ray from each source frame to the marker on its paired target frame.

    Input structure:
    - The background is the most frequent color. A shared frame color surrounds every
      monochrome 2x2 block with a 4x4 border.
    - Each block color appears in exactly two 1-gap frames. One is an unmarked source.
      The other is a marked target: its gap has a single shared marker-color pixel
      immediately outside it.

    Rule:
    - Pair frames by matching the color of their inner 2x2 blocks.
    - Only the unmarked frame emits. Its gap fixes a straight cardinal or diagonal
      direction, and the paired target must have the opposite-facing gap.
    - Paint the source gap and every background cell along that corridor, stopping
      just before the target's adjacent marker pixel. The target frame itself does not emit.

    Key insight: gap count alone no longer tells you which frames fire. The solver
    must first classify same-colored 1-gap frames into source vs. marked target, then
    use that pairing to decide which gaps launch rays and where those rays terminate.
    """
    g = np.asarray(grid, dtype=int)
    h, w = g.shape
    bg = get_bg(g)
    out = g.copy()

    blocks = _find_blocks_2x2(g, bg)
    assert blocks, "expected at least one framed 2x2 block"

    frames_by_color: dict[int, list[dict[str, object]]] = {}
    frame_color: int | None = None
    frame_bbox_cells_all: set[tuple[int, int]] = set()

    for color, br, bc in blocks:
        wall = _infer_frame_color(g, bg, br, bc, color)
        if frame_color is None:
            frame_color = wall
        else:
            assert wall == frame_color, "expected one shared frame color"
        gaps = _frame_gaps_for_block(g, bg, wall, br, bc)
        assert len(gaps) == 1, "combo expects exactly one structural gap per frame"
        gap = gaps[0]
        d = _gap_dir_from_abs(br, bc, gap[0], gap[1])
        fr0, fc0 = br - 1, bc - 1
        entry = {"br": br, "bc": bc, "fr0": fr0, "fc0": fc0, "gap": gap, "dir": d}
        frames_by_color.setdefault(color, []).append(entry)
        frame_bbox_cells_all |= _frame_bbox_cells(fr0, fc0)

    nonbg_cells = [tuple(map(int, rc)) for rc in np.argwhere(g != bg)]
    marker_cells = {cell for cell in nonbg_cells if cell not in frame_bbox_cells_all}
    assert marker_cells, "expected receiver markers outside the frames"
    marker_colors = {int(g[r, c]) for r, c in marker_cells}
    assert len(marker_colors) == 1, "expected one shared marker color"
    marker_color = next(iter(marker_colors))

    sources: dict[int, dict[str, object]] = {}
    targets: dict[int, dict[str, object]] = {}
    used_markers: set[tuple[int, int]] = set()

    for color, entries in frames_by_color.items():
        assert len(entries) == 2, "expected one source frame and one target frame per block color"
        marked = []
        plain = []
        for entry in entries:
            gap = entry["gap"]
            d = entry["dir"]
            mark = (gap[0] + d[0], gap[1] + d[1])
            if 0 <= mark[0] < h and 0 <= mark[1] < w and int(g[mark[0], mark[1]]) == marker_color:
                marked.append(entry)
            else:
                plain.append(entry)
        assert len(marked) == 1 and len(plain) == 1, "expected exactly one marked target and one plain source"
        source = plain[0]
        target = marked[0]
        sd = source["dir"]
        td = target["dir"]
        assert td == _opp(sd), "paired target gap must face back toward its source"
        expected_marker = (target["gap"][0] + td[0], target["gap"][1] + td[1])
        assert expected_marker in marker_cells, "expected target marker adjacent to the target gap"
        sources[color] = source
        targets[color] = {"entry": target, "marker": expected_marker}
        used_markers.add(expected_marker)

    assert used_markers == marker_cells, "expected every outside-frame singleton to be a target marker"

    def in_bounds(cell: tuple[int, int]) -> bool:
        r, c = cell
        return 0 <= r < h and 0 <= c < w

    for color in sorted(sources):
        source = sources[color]
        target = targets[color]
        gap = source["gap"]
        d = source["dir"]
        marker = target["marker"]
        out[gap[0], gap[1]] = color
        pos = gap
        for _ in range(h * w):
            nxt = (pos[0] + d[0], pos[1] + d[1])
            assert in_bounds(nxt), "ray left the grid before reaching its paired target"
            if nxt == marker:
                break
            assert int(g[nxt[0], nxt[1]]) == bg, "expected a clear corridor from source to paired target"
            assert int(out[nxt[0], nxt[1]]) == bg, "generator must keep corridors disjoint"
            out[nxt[0], nxt[1]] = color
            pos = nxt
        else:
            raise AssertionError("ray tracing did not terminate at a paired target marker")

    return out


# --------------- generator (combo variant) ---------------

PAIR_COUNT_RANGE = (2, 5)              # uniform draw [lo, hi)
GRID_SIZE_MAX = 31                     # exclusive upper bound
GRID_SIZE_MIN_BY_PAIRS = {2: 16, 3: 20, 4: 24}
BG_CHOICES = [0, 8]
BG_WEIGHTS = [0.25, 0.75]
FRAME_COLOR_FOR_BG = {8: 3, 0: 8}
PALETTE_FOR_BG = {
    8: [1, 2, 4, 6, 7, 9],
    0: [1, 2, 3, 4, 6, 7, 9],
}
FRAME_BBOX_GAP = 2
PLACEMENT_MARGIN = 2
MIN_TARGET_STEPS = 4
MIN_BEAM_PIXELS_PER_PAIR = 4
PAIR_SAMPLE_TRIES = 350
LAYOUT_RESTARTS = 90
GEN_RETRIES = 400


def _frames_separated(a: tuple[int, int], b: tuple[int, int]) -> bool:
    ar0, ac0 = a
    br0, bc0 = b
    ar1, ac1 = ar0 + 3, ac0 + 3
    br1, bc1 = br0 + 3, bc0 + 3
    gap = FRAME_BBOX_GAP
    return ar1 + gap < br0 or br1 + gap < ar0 or ac1 + gap < bc0 or bc1 + gap < ac0


def _frame_fits(n: int, pos: tuple[int, int]) -> bool:
    r0, c0 = pos
    margin = PLACEMENT_MARGIN
    return margin <= r0 <= n - 4 - margin and margin <= c0 <= n - 4 - margin


def _draw_framed_block(
    grid: np.ndarray,
    r0: int,
    c0: int,
    bg: int,
    frame_color: int,
    block_color: int,
    gap_dir: tuple[int, int],
) -> None:
    patch = np.full((4, 4), frame_color, dtype=int)
    patch[1:3, 1:3] = block_color
    gr, gc = _DIR_TO_GAP[gap_dir]
    patch[gr, gc] = bg
    paste(grid, patch, r0, c0)


def _pick_colors(rng: np.random.Generator, pair_count: int) -> tuple[int, int, int, list[int]]:
    bg = int(rng.choice(BG_CHOICES, p=BG_WEIGHTS))
    frame_color = FRAME_COLOR_FOR_BG[bg]
    palette = [c for c in PALETTE_FOR_BG[bg] if c != frame_color]
    marker_color = int(rng.choice(palette))
    block_palette = [c for c in palette if c != marker_color]
    block_colors = list(rng.choice(block_palette, size=pair_count, replace=False))
    return bg, frame_color, marker_color, block_colors


def _grid_size_min(pair_count: int) -> int:
    result = 14
    for threshold, min_n in sorted(GRID_SIZE_MIN_BY_PAIRS.items()):
        if pair_count >= threshold:
            result = min_n
    return result


def _all_frame_positions(n: int) -> list[tuple[int, int]]:
    margin = PLACEMENT_MARGIN
    hi = n - 4 - margin
    return [(r, c) for r in range(margin, hi + 1) for c in range(margin, hi + 1)]


def _marker_offset_for_source_dir(d: tuple[int, int]) -> tuple[int, int]:
    td = _opp(d)
    gr, gc = _DIR_TO_GAP[td]
    return (gr + td[0], gc + td[1])


def _make_plan(
    n: int,
    source_pos: tuple[int, int],
    d: tuple[int, int],
    steps: int,
) -> dict[str, object] | None:
    sgap_local = _DIR_TO_GAP[d]
    sgap = (source_pos[0] + sgap_local[0], source_pos[1] + sgap_local[1])
    marker_offset = _marker_offset_for_source_dir(d)
    marker = (sgap[0] + steps * d[0], sgap[1] + steps * d[1])
    target_pos = (marker[0] - marker_offset[0], marker[1] - marker_offset[1])
    if not _frame_fits(n, target_pos):
        return None
    if not _frames_separated(source_pos, target_pos):
        return None

    td = _opp(d)
    tgap_local = _DIR_TO_GAP[td]
    target_gap = (target_pos[0] + tgap_local[0], target_pos[1] + tgap_local[1])
    if marker != (target_gap[0] + td[0], target_gap[1] + td[1]):
        return None

    points = [(sgap[0] + i * d[0], sgap[1] + i * d[1]) for i in range(steps)]
    if len(points) < MIN_BEAM_PIXELS_PER_PAIR:
        return None
    for r, c in points:
        if not (0 <= r < n and 0 <= c < n):
            return None
    if not (0 <= marker[0] < n and 0 <= marker[1] < n):
        return None

    source_cells = _frame_bbox_cells(*source_pos)
    target_cells = _frame_bbox_cells(*target_pos)
    if marker in source_cells or marker in target_cells:
        return None
    for cell in points[1:]:
        if cell in source_cells or cell in target_cells:
            return None

    segments = list(zip(points, points[1:]))
    diag_sigs = tuple(sig for sig in (_diag_signature(seg) for seg in segments) if sig is not None)

    return {
        "source_pos": source_pos,
        "target_pos": target_pos,
        "dir": d,
        "target_dir": td,
        "marker": marker,
        "points": tuple(points),
        "diag_sigs": diag_sigs,
        "cells": frozenset(source_cells | target_cells | {marker} | set(points)),
        "frames": (source_pos, target_pos),
    }


def _compatible(
    plan: dict[str, object],
    used_cells: frozenset[tuple[int, int]],
    used_frames: tuple[tuple[int, int], ...],
    diag_map: dict[tuple[int, int], int],
) -> bool:
    if used_cells & plan["cells"]:
        return False
    for pos in plan["frames"]:
        if any(not _frames_separated(pos, prev) for prev in used_frames):
            return False
    for mid, slope in plan["diag_sigs"]:
        prev = diag_map.get(mid)
        if prev is not None and prev != slope:
            return False
    return True


def _choose_plans(n: int, pair_count: int, rng: np.random.Generator) -> list[dict[str, object]] | None:
    positions = _all_frame_positions(n)
    dirs = list(_DIRS)

    for _ in range(LAYOUT_RESTARTS):
        plans: list[dict[str, object]] = []
        used_cells: frozenset[tuple[int, int]] = frozenset()
        used_frames: tuple[tuple[int, int], ...] = ()
        diag_map: dict[tuple[int, int], int] = {}

        for _pair_idx in range(pair_count):
            placed = False
            for _ in range(PAIR_SAMPLE_TRIES):
                source_pos = positions[int(rng.integers(len(positions)))]
                d = dirs[int(rng.integers(len(dirs)))]
                steps = int(rng.integers(MIN_TARGET_STEPS, n + n))
                plan = _make_plan(n, source_pos, d, steps)
                if plan is None:
                    continue
                if not _compatible(plan, used_cells, used_frames, diag_map):
                    continue
                plans.append(plan)
                used_cells = used_cells | plan["cells"]
                used_frames = used_frames + tuple(plan["frames"])
                diag_map = dict(diag_map)
                for mid, slope in plan["diag_sigs"]:
                    diag_map[mid] = slope
                placed = True
                break
            if not placed:
                break
        if len(plans) == pair_count:
            return plans
    return None


def _gen_attempt() -> np.ndarray | None:
    rng = np.random.default_rng()

    pair_count = int(rng.integers(*PAIR_COUNT_RANGE))
    n = int(rng.integers(_grid_size_min(pair_count), GRID_SIZE_MAX))
    bg, frame_color, marker_color, block_colors = _pick_colors(rng, pair_count)

    plans = _choose_plans(n, pair_count, rng)
    if plans is None:
        return None

    grid = np.full((n, n), bg, dtype=int)
    for color, plan in zip(block_colors, plans):
        sr0, sc0 = plan["source_pos"]
        tr0, tc0 = plan["target_pos"]
        _draw_framed_block(grid, sr0, sc0, bg, frame_color, int(color), plan["dir"])
        _draw_framed_block(grid, tr0, tc0, bg, frame_color, int(color), plan["target_dir"])
        mr, mc = plan["marker"]
        assert int(grid[mr, mc]) == bg, "target marker must not overlap a frame"
        grid[mr, mc] = marker_color

    if int(np.bincount(grid.ravel(), minlength=10).argmax()) != bg:
        return None

    try:
        out = main(grid)
    except Exception:
        return None

    if np.count_nonzero(out != grid) < pair_count * MIN_BEAM_PIXELS_PER_PAIR:
        return None
    return grid


def generate_input() -> np.ndarray:
    for _ in range(GEN_RETRIES):
        grid = _gen_attempt()
        if grid is not None:
            return grid
    raise RuntimeError("failed to generate a valid e12f9a14 combo input")
