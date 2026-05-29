# Combo variant of e87109e9 (index 30)
# combo_spec: D2/0+D16/1
# Name: 镜面反射加边缘反弹
#
# Design analysis for the fused variant
# a) Concrete behavior:
#    Original e87109e9 used a teal square that slid in the four cardinal
#    directions; whenever the next footprint hit a colored rectangle, the beam
#    looked up that color in the top legend and turned left or right. Here that
#    lookup is deleted. Instead, only the newly entered 1-cell frontier strip of
#    the moving square matters: if the frontier strip meets obstacle cells on its
#    left half, the beam deflects away from that side; if the whole strip is
#    blocked, it reverses. Example: a 2x2 source moving north enters a new top
#    strip of width 2. If only the left frontier cell is blocked, the beam turns
#    east; if both frontier cells are blocked, it turns back south. At the board
#    edge the same beam reflects instead of terminating, so a path can bounce off
#    a wall, revisit a previous corridor, and only stop when it meets trail laid
#    down by an earlier beam or repeats states.
# b) Complexity calibration:
#    The fused rule must be visibly different from the original task, so the
#    generator always creates a layout with four edge-touching rectangles and at
#    least one beam that undergoes both obstacle reflection and a border bounce.
#    The geometry is kept low-entropy: one source size, four rectangles, and a
#    small family of mirrored layouts. That keeps the rule learnable while still
#    making reflection visually obvious.
# c) Broken implicit assumptions from the original solver:
#    - Legend bars no longer encode control flow, so parsing turn directions is
#      invalid and the legend becomes decorative/cropped only.
#    - The original whole-footprint obstacle test is too coarse once beams can
#      bounce repeatedly; with a sliding 2x2/3x3 square, consecutive positions
#      overlap. Using full-footprint trail blocking would freeze every beam at
#      step 0. The new solver therefore checks only the frontier strip that is
#      newly entered on the next step.
#    - Edge termination is no longer valid; the state machine must support
#      reflection and guard against cycles with an energy bound plus seen states.
# d) Placement legality:
#    The generator uses four non-overlapping edge-touching rectangles with fixed
#    thickness 4 and length 5, keeps the teal square well inside the board, and
#    uses a background color distinct from obstacle/legend colors. Optional board
#    flips preserve legality because they are rigid transforms of a valid layout.
# e) Visual observability:
#    The useful cue is spatial, not chromatic: offset obstacle placement creates
#    different teal corridors after collision. The legend is still cropped away
#    in the output, but its bar positions are randomized precisely so that they
#    cannot be mistaken for the active control variable.
# f) Visual cleanliness:
#    Obstacles stay on the borders, so the central region remains readable. The
#    solver paints only background cells on each frontier strip, which avoids
#    mixing teal with obstacle colors and keeps the reflected corridors legible.
# g) Change inferability:
#    No hidden color ordering or numeric comparison is used. The only behavior
#    changes are: obstacle collisions depend on contact geometry, and boundaries
#    bounce instead of absorbing the beam. The generator also verifies that the
#    edge-bounce ablation produces a different output, so the border interaction
#    is not merely latent.

from __future__ import annotations

import random

import numpy as np

from arc_utils import bbox_coords, components_mask

_N, _E, _S, _W = range(4)
_DIRS = [(-1, 0), (0, 1), (1, 0), (0, -1)]

LEGEND_BG = 5
TEAL = 8
GRID_W = 24
LEGEND_H = 6
LEGEND_CELL_STARTS = [1, 7, 13, 19]
LEGEND_COLOR_POOL = [1, 2, 3, 4, 6]
BG_COLOR_POOL = [1, 2, 3, 4, 6, 7, 9]
MAX_RETRIES = 200


def _legend_height(grid: np.ndarray) -> int:
    rows = np.where(np.any(grid == LEGEND_BG, axis=1))[0]
    assert rows.size, "legend rows must be present"
    legend_h = int(rows.max() + 1)
    assert np.array_equal(rows, np.arange(legend_h)), "legend band must be contiguous from the top"
    return legend_h


def _find_teal_square(g: np.ndarray, teal: int = TEAL) -> tuple[int, int, int]:
    candidates: list[tuple[int, int, int]] = []
    for cells in components_mask(g == teal, connectivity=4):
        r0, c0, r1, c1 = bbox_coords(cells)
        hh = r1 - r0 + 1
        ww = c1 - c0 + 1
        if hh != ww:
            continue
        size = hh
        if int(np.sum(g[r0 : r1 + 1, c0 : c1 + 1] == teal)) != size * size:
            continue
        candidates.append((int(r0), int(c0), int(size)))
    assert len(candidates) == 1, "main grid must contain exactly one solid teal square"
    return candidates[0]


def _frontier_slice(r: int, c: int, sz: int, d: int) -> tuple[slice, slice]:
    if d == _N:
        return slice(r, r + 1), slice(c, c + sz)
    if d == _S:
        return slice(r + sz - 1, r + sz), slice(c, c + sz)
    if d == _W:
        return slice(r, r + sz), slice(c, c + 1)
    return slice(r, r + sz), slice(c + sz - 1, c + sz)


def _reflect_from_frontier(mask: np.ndarray, d: int, sz: int) -> int:
    """
    Discrete mirror rule on the newly entered frontier strip.

    If the whole strip is blocked, reflect straight back. If only one side of the
    strip is blocked, deflect away from the contacted side, which is the discrete
    analogue of a glancing mirror hit for the thick square beam.
    """
    if d in (_N, _S):
        cols = np.where(mask[0])[0]
        assert cols.size
        if int(cols[0]) == 0 and int(cols[-1]) == sz - 1:
            return d ^ 2
        return _E if float(cols.mean()) < (sz - 1) / 2 else _W

    rows = np.where(mask[:, 0])[0]
    assert rows.size
    if int(rows[0]) == 0 and int(rows[-1]) == sz - 1:
        return d ^ 2
    return _S if float(rows.mean()) < (sz - 1) / 2 else _N


def _simulate_reflections(
    base: np.ndarray,
    *,
    border_bounce: bool = True,
    max_steps: int | None = None,
) -> tuple[np.ndarray, dict[str, int]]:
    vals, cnts = np.unique(base, return_counts=True)
    bg = int(vals[cnts.argmax()])
    sr, sc, sz = _find_teal_square(base, teal=TEAL)
    H, W = base.shape

    out = base.copy()
    global_trail = np.zeros_like(base, dtype=bool)
    global_trail[sr : sr + sz, sc : sc + sz] = True
    energy = int(max_steps if max_steps is not None else 8 * H * W)

    stats = {
        "edge_bounces": 0,
        "obstacle_bounces": 0,
        "turn90": 0,
        "trail_stops": 0,
        "painted_steps": 0,
    }

    for d0 in (_N, _E, _S, _W):
        r, c, d = int(sr), int(sc), int(d0)
        seen: set[tuple[int, int, int]] = set()
        preexisting = global_trail.copy()

        for _ in range(energy):
            state = (r, c, d)
            if state in seen:
                break
            seen.add(state)

            dr, dc = _DIRS[d]
            nr, nc = r + dr, c + dc
            if not (0 <= nr and 0 <= nc and nr + sz <= H and nc + sz <= W):
                if not border_bounce:
                    break
                d = d ^ 2
                stats["edge_bounces"] += 1
                continue

            fr = _frontier_slice(nr, nc, sz, d)
            frontier = base[fr]
            obstacle_mask = (frontier != bg) & (frontier != TEAL)
            if bool(obstacle_mask.any()):
                nd = _reflect_from_frontier(obstacle_mask, d, sz)
                stats["obstacle_bounces"] += 1
                if nd != (d ^ 2):
                    stats["turn90"] += 1
                d = nd
                continue

            if bool(preexisting[fr].any()):
                stats["trail_stops"] += 1
                break

            r, c = nr, nc
            band = out[fr].copy()
            band[band == bg] = TEAL
            out[fr] = band
            stats["painted_steps"] += 1

        global_trail |= out == TEAL

    return out, stats


def _make_legend(legend_colors: list[int]) -> np.ndarray:
    legend = np.full((LEGEND_H, GRID_W), LEGEND_BG, dtype=int)
    for start, color in zip(LEGEND_CELL_STARTS, legend_colors):
        legend[1:5, start : start + 4] = 0
        bar_col = start if random.random() < 0.5 else start + 3
        legend[1:5, bar_col] = color
    return legend


def _maybe_flip(base: np.ndarray) -> np.ndarray:
    out = base
    if random.random() < 0.5:
        out = np.fliplr(out)
    if random.random() < 0.5:
        out = np.flipud(out)
    return np.array(out, copy=True)


def _build_template_board() -> np.ndarray:
    H = random.randint(19, 24)
    W = GRID_W
    sz = 2 if random.random() < 0.75 else 3
    thickness = 4
    span_len = 5

    legend_colors = random.sample(LEGEND_COLOR_POOL, 4)
    bg_choices = [c for c in BG_COLOR_POOL if c not in legend_colors]
    bg = random.choice(bg_choices)

    sr = random.randint(9, H - sz - 6)
    sc = random.randint(8, 13)
    top_color, left_color, bottom_color, right_color = legend_colors

    base = np.full((H, W), bg, dtype=int)
    base[sr : sr + sz, sc : sc + sz] = TEAL

    # Canonical "pinball lane" layout. The active beam hits:
    # top obstacle -> right wall -> left obstacle -> bottom obstacle -> right obstacle.
    base[:thickness, sc - span_len + 1 : sc + 1] = top_color
    base[thickness - span_len + 1 : thickness + 1, :thickness] = left_color
    base[H - thickness :, thickness - span_len + 1 : thickness + 1] = bottom_color
    right_start = H - thickness - 1
    base[right_start : right_start + span_len, W - thickness :] = right_color

    base = _maybe_flip(base)
    legend = _make_legend(legend_colors)
    return np.vstack([legend, base])


def _legend_perturbation(inp: np.ndarray) -> np.ndarray:
    legend_h = _legend_height(inp)
    out = np.array(inp, copy=True)
    legend = out[:legend_h].copy()
    colors = [int(c) for c in np.unique(legend) if c not in {0, LEGEND_BG}]
    legend[:, :] = LEGEND_BG
    for start, color in zip(LEGEND_CELL_STARTS, colors):
        legend[1:5, start : start + 4] = 0
        bar_col = start + 3
        legend[1:5, bar_col] = color
    out[:legend_h] = legend
    return out


def _is_good_sample(inp: np.ndarray) -> bool:
    legend_h = _legend_height(inp)
    base = inp[legend_h:, :]
    out, stats = _simulate_reflections(base, border_bounce=True)
    out_no_edge, _ = _simulate_reflections(base, border_bounce=False)
    perturbed = _legend_perturbation(inp)
    perturbed_out = main(perturbed)

    diff_no_edge = int(np.count_nonzero(out != out_no_edge))
    return (
        stats["edge_bounces"] >= 2
        and stats["obstacle_bounces"] >= 2
        and stats["turn90"] >= 1
        and diff_no_edge >= 10
        and np.array_equal(perturbed_out, out)
    )


def main(grid: np.ndarray) -> np.ndarray:
    """
    Project four thick teal beams through a cropped pinball board.

    Input structure:
    - The input still has two regions: a 6-row grey legend band on top and the
      actual board below it.
    - The legend bars may reuse obstacle colors, but their left/right placement
      is now irrelevant; they survive only as a removable header.
    - The main board contains one solid teal 2x2 or 3x3 source square, one
      background color, and several edge-touching rectangular obstacles.

    Rule:
    - Remove the top legend band and emit one square-width beam in each cardinal
      direction from the teal source.
    - Each step paints only the newly entered frontier strip of the moving
      square. If that frontier strip hits obstacle cells, reflect according to
      the contact geometry instead of consulting a color-turn legend.
    - If the next step would leave the board, bounce off the boundary and keep
      going. A beam stops when it reaches a strip already occupied by earlier
      beams or when repeated states exhaust the simulation.

    Key insight: The active control variable is purely geometric. Offset obstacle
    placement determines whether a collision is a full rebound or a sideways
    deflection, and border rebounds keep the path alive long enough for those
    reflections to matter visibly.
    """
    g = np.asarray(grid, dtype=int)
    legend_h = _legend_height(g)
    base = g[legend_h:, :].copy()
    assert base.size, "main grid must remain after legend crop"
    out, _stats = _simulate_reflections(base, border_bounce=True)
    return out


def generate_input() -> np.ndarray:
    for _ in range(MAX_RETRIES):
        inp = _build_template_board()
        try:
            out = main(inp)
        except AssertionError:
            continue
        if out.size == 0:
            continue
        if _is_good_sample(inp):
            return inp
    raise RuntimeError("failed to generate e87109e9 combo input")
