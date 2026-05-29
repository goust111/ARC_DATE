# Combo variant of e87109e9 (index 33)
# combo_spec: D2/3+D4/0+D14/0
# Name: 交替转弯顺序发射颜色继承

import random

import numpy as np

from arc_utils import bbox_coords, components_mask

# Design analysis
# ---------------
# a) Concrete behavior / trace
#    - The input still looks like the base task: a 6x24 color-5 legend strip on
#      top, then one main grid with a solid teal source square and edge-touching
#      colored obstacle bars.
#    - The legend is now only structural/decorative. The real process is one
#      single timeline:
#        1. fire the up beam,
#        2. then the right beam,
#        3. then the down beam,
#        4. then the left beam.
#    - The turn rule is global, not per color and not per beam. Collision #1 in
#      the whole timeline turns clockwise, collision #2 counterclockwise,
#      collision #3 clockwise, and so on.
#    - The paint rule is stateful. The first segment of a beam is teal; after a
#      collision, the beam's future segments inherit the color of the blocking
#      obstacle that caused that turn.
#    - Earlier finished trails become later obstacles and keep their painted
#      colors. In the template used by the generator, the up beam usually hits
#      north then east, the right beam later hits east, and the down beam hits
#      south and then the right beam's inherited-color trail. This makes the
#      three dimensions genuinely interact rather than appearing independently.
#
# b) Complexity calibration
#    - If the alternating count reset on every beam, sequential firing would be a
#      weak add-on. The fused version therefore carries the turn parity across
#      the whole ordered simulation.
#    - If trails became obstacles but stayed always teal, the color-inheritance
#      dimension would be visually underused. The generator instead forces at
#      least one non-teal trail hit so a later beam visibly inherits a previous
#      obstacle color through an earlier trail.
#    - To keep the puzzle readable, the generator fixes a compact 3-obstacle
#      family (north/east/south) with a stable hit pattern instead of allowing
#      arbitrary 3-4 side combinations that often produce messy or degenerate
#      outputs.
#
# c) Hidden-assumption audit against the original main()
#    - Original assumption: legend colors encode turning. Broken here; the
#      solver should only validate legend structure, then ignore bar sides.
#    - Original assumption: beams can inspect the evolving painted output to find
#      blockers because every trail cell is teal and intentionally ignored.
#      Broken here twice: trails are colored, and earlier trails now matter.
#      Blocker detection therefore combines immutable base obstacles with a
#      separate earlier-trail color grid.
#    - Original assumption: each beam is independent, so state (r, c, d) is
#      enough. Broken here because the future also depends on the current
#      global-turn parity and the beam's current paint color.
#    - Original assumption: only colored edge obstacles can block a beam. Broken
#      here because later beams can hit earlier inherited-color trails.
#
# d) Placement legality
#    - Obstacles stay as non-overlapping edge-touching rectangles with thickness
#      2-4 and spans chosen to avoid corner overlap.
#    - The source square is placed with comfortable margins and with room above,
#      right, and below for the intended north/east/south interactions.
#    - The generator validates the full simulation and rejects any mixed-color
#      blocker footprint, cycle, overlap, or solver mismatch.
#
# e) Visual observability
#    - The output visibly differs from the base task in three ways at once:
#      turning is not color-fixed, later beams bend around earlier trails, and
#      later segments change away from teal.
#    - The generator requires at least one later beam to collide with a non-teal
#      earlier trail, so humans can see that order and inherited color both
#      matter.
#    - Decorative legend bars are randomized independently of the real behavior,
#      preventing a misleading but still-valid color->turn explanation.
#
# f) Visual cleanliness
#    - Using only the north/east/south obstacle family keeps the interior mostly
#      open and produces one readable upper hook plus one readable lower bend.
#    - Oracle checks reject ambiguous mixed-color blocker footprints and outputs
#      that collapse to simpler ablations.
#
# g) Change inferability
#    - Non-teal trail colors are limited to obstacle colors already present in
#      the input, so the recoloring rule has visible support.
#    - The generator rejects samples identical to any of three simpler models:
#      no earlier-trail interaction, per-beam turn reset, or fixed-teal trails.
#      This keeps the dataset from silently degenerating.

_DIRS = [(-1, 0), (0, 1), (1, 0), (0, -1)]
_N, _E, _S, _W = range(4)

# Palettes
LEGEND_COLOR_POOL = [1, 2, 3, 4, 6]
BG_COLOR_POOL = [1, 2, 3, 4, 6, 7, 9]
LEGEND_BG = 5
TEAL = 8

# Geometry
GRID_H_RANGE = (19, 24)
GRID_W = 24
LEGEND_H = 6
LEGEND_CELL_STARTS = [1, 7, 13, 19]

TEAL_SZ2_PROB = 0.75
SQUARE_EDGE_GAP = 5

OBS_THICKNESS_RANGE = (2, 4)
MAX_RETRIES = 400


def _legend_colors(grid: np.ndarray, legend_h: int) -> list[int]:
    """Read the four legend bar colors for structure validation only."""
    legend = grid[:legend_h, :]
    assert legend.size, "legend must be non-empty"
    vals, cnts = np.unique(legend, return_counts=True)
    bg = int(vals[cnts.argmax()])
    sep_cols = [c for c in range(legend.shape[1]) if np.all(legend[:, c] == bg)]
    assert len(sep_cols) >= 2, "legend separator columns must exist"

    colors: list[int] = []
    for a, b in zip(sep_cols, sep_cols[1:], strict=False):
        if b - a <= 2:
            continue
        interior = legend[1 : legend_h - 1, a + 1 : b]
        cell_colors = set(map(int, np.unique(interior)))
        cell_colors.discard(bg)
        cell_colors.discard(0)
        assert len(cell_colors) == 1, "each legend cell must contain exactly one colored bar"
        colors.append(int(next(iter(cell_colors))))
    assert len(colors) == 4, "legend must encode four colors"
    return colors


def _find_teal_square(g: np.ndarray, teal: int = TEAL) -> tuple[int, int, int]:
    """Return (r0, c0, size) for the unique solid teal square."""
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


def _rot_ccw(d: int) -> int:
    return (d - 1) % 4


def _rot_cw(d: int) -> int:
    return (d + 1) % 4


def _make_legend(legend_colors: list[int], legend_dirs: dict[int, str]) -> np.ndarray:
    """Build a decorative legend strip matching the original visual language."""
    legend = np.full((LEGEND_H, GRID_W), LEGEND_BG, dtype=int)
    for start, color in zip(LEGEND_CELL_STARTS, legend_colors):
        legend[1:5, start : start + 4] = 0
        bar_col = start if legend_dirs[color] == "L" else start + 3
        legend[1:5, bar_col] = color
    return legend


def _random_legend_dirs(colors: list[int]) -> dict[int, str]:
    """Decorative legend bars: keep both sides represented for visual variety."""
    while True:
        dirs = {color: random.choice(["L", "R"]) for color in colors}
        if len(set(dirs.values())) == 2:
            return dirs


def _obstacle_rect(
    side: int, thickness: int, span: tuple[int, int], h: int, w: int
) -> tuple[int, int, int, int]:
    s0, s1 = span
    if side == _N:
        return 0, thickness - 1, s0, s1
    if side == _S:
        return h - thickness, h - 1, s0, s1
    if side == _W:
        return s0, s1, 0, thickness - 1
    return s0, s1, w - thickness, w - 1


def _simulate_variant(
    base: np.ndarray,
    bg: int,
    *,
    use_trails: bool,
    global_turns: bool,
    inherit_color: bool,
) -> tuple[np.ndarray, list[dict[str, object]], list[dict[tuple[int, int], int]]]:
    """
    Simulate the fused rule family with toggles for ablation tests.

    - `use_trails`: completed earlier beams become blockers for later beams.
    - `global_turns`: alternate turns across the whole timeline; otherwise reset
      to clockwise at the start of each beam.
    - `inherit_color`: after a hit, later segments use the blocker color;
      otherwise every painted cell stays teal.
    """
    sr, sc, sz = _find_teal_square(base, teal=TEAL)
    out = base.copy()
    h, w = out.shape
    earlier_trails = np.zeros_like(base, dtype=int)
    next_turn_cw = True
    beam_stats: list[dict[str, object]] = []
    beam_paints: list[dict[tuple[int, int], int]] = []

    def in_bounds(r: int, c: int) -> bool:
        return 0 <= r and 0 <= c and r + sz - 1 < h and c + sz - 1 < w

    def blocker_at_next(nr: int, nc: int) -> tuple[int | None, str | None]:
        base_blk = base[nr : nr + sz, nc : nc + sz]
        trail_blk = earlier_trails[nr : nr + sz, nc : nc + sz]

        base_colors = set(map(int, np.unique(base_blk)))
        base_colors.discard(bg)
        base_colors.discard(TEAL)

        trail_colors = set(map(int, np.unique(trail_blk)))
        trail_colors.discard(0)

        colors = set(base_colors)
        if use_trails:
            colors |= trail_colors
        if not colors:
            return None, None
        assert len(colors) == 1, "beam footprint must not hit mixed blocker colors"
        kind = "trail" if use_trails and trail_colors else "base"
        return int(next(iter(colors))), kind

    for d0 in range(4):
        r, c, d = sr, sc, d0
        beam_next_turn_cw = next_turn_cw if global_turns else True
        paint_color = TEAL
        seen: set[tuple[int, int, int, int, int]] = set()
        hits: list[dict[str, object]] = []
        beam_cells: dict[tuple[int, int], int] = {}
        beam_trail = np.zeros_like(base, dtype=int)

        def paint_block(rr: int, cc: int) -> None:
            blk_base = base[rr : rr + sz, cc : cc + sz]
            rows, cols = np.where(blk_base == bg)
            color = int(paint_color if inherit_color else TEAL)
            for roff, coff in zip(rows, cols):
                cell = (int(rr + roff), int(cc + coff))
                beam_cells[cell] = color
                beam_trail[cell] = color
                out[cell] = color

        while True:
            state = (r, c, d, int(paint_color if inherit_color else TEAL), int(beam_next_turn_cw))
            assert state not in seen, "beam path must not cycle"
            seen.add(state)

            dr, dc = _DIRS[d]
            nr, nc = r + dr, c + dc
            if not in_bounds(nr, nc):
                break

            obs_color, obs_kind = blocker_at_next(nr, nc)
            if obs_color is None:
                r, c = nr, nc
                paint_block(r, c)
                continue

            hits.append(
                {
                    "kind": str(obs_kind),
                    "color": int(obs_color),
                    "dir": int(d),
                    "at": (int(nr), int(nc)),
                    "turn": "R" if beam_next_turn_cw else "L",
                }
            )
            d = _rot_cw(d) if beam_next_turn_cw else _rot_ccw(d)
            beam_next_turn_cw = not beam_next_turn_cw
            if global_turns:
                next_turn_cw = beam_next_turn_cw
            if inherit_color:
                paint_color = int(obs_color)

        earlier_trails = np.where(beam_trail != 0, beam_trail, earlier_trails)
        beam_stats.append(
            {
                "dir": int(d0),
                "hits": hits,
                "trail_cells": int(np.count_nonzero(beam_trail)),
            }
        )
        beam_paints.append(beam_cells)

    return out, beam_stats, beam_paints


def main(grid: np.ndarray) -> np.ndarray:
    """
    Trace four ordered square beams whose turns alternate globally and whose colors propagate through time.

    Input structure:
    - The top band is still a 6-row legend strip on color-5 background with four colored bars, but its left/right placement is now decorative rather than operational.
    - Below the legend is one main grid with a uniform background, one solid teal source square, and several edge-touching colored obstacle rectangles.

    Rule:
    - Remove the entire legend band in the output.
    - Emit beams in the fixed order up, then right, then down, then left.
    - The first clear segment of a beam paints teal on background cells only.
    - Every collision in the whole timeline alternates turn direction: collision 1 turns clockwise, collision 2 counterclockwise, collision 3 clockwise, and so on, regardless of color.
    - After a collision, the beam's future segments inherit the color of the blocker that caused that turn; completed earlier trails then become later blockers carrying those same colors.
    - A beam stops when its next sz-wide step would leave the cropped main grid.

    Key insight: The puzzle no longer separates into geometry plus a static legend lookup. You must simulate one shared chronology where turn parity, inherited trail color, and earlier-trail blocking all feed into each later step.
    """
    g = np.asarray(grid, dtype=int)

    rows_with_legend = np.where(np.any(g == LEGEND_BG, axis=1))[0]
    assert rows_with_legend.size, "legend rows must be present"
    legend_h = int(rows_with_legend.max() + 1)
    assert np.array_equal(rows_with_legend, np.arange(legend_h)), "legend color must be confined to the top band"
    _legend_colors(g, legend_h)

    base = g[legend_h:, :].copy()
    assert base.size, "main grid must remain after legend crop"

    vals, cnts = np.unique(base, return_counts=True)
    bg = int(vals[cnts.argmax()])
    _find_teal_square(base, teal=TEAL)

    out, _beam_stats, _beam_paints = _simulate_variant(
        base,
        bg,
        use_trails=True,
        global_turns=True,
        inherit_color=True,
    )
    return out


def _gen_attempt() -> np.ndarray | None:
    """Generate a clean north/east/south template that exhibits the fused rule."""
    h, w = random.randint(*GRID_H_RANGE), GRID_W
    sz = 2 if random.random() < TEAL_SZ2_PROB else 3
    t_n = random.randint(*OBS_THICKNESS_RANGE)
    t_e = random.randint(*OBS_THICKNESS_RANGE)
    t_s = random.randint(*OBS_THICKNESS_RANGE)

    lane_n = t_n + 1
    lane_e = w - t_e - sz
    lane_s = h - t_s - sz

    min_r = max(SQUARE_EDGE_GAP, t_n + 2)
    max_r = min(h - sz - SQUARE_EDGE_GAP, lane_s - 2)
    min_c = SQUARE_EDGE_GAP
    max_c = min(w - sz - SQUARE_EDGE_GAP, lane_e - 2)
    if min_r > max_r or min_c > max_c:
        return None

    sr = random.randint(min_r, max_r)
    sc = random.randint(min_c, max_c)

    # Spans are chosen so:
    # - north covers the source and reaches toward the east side,
    # - east covers both the up-beam lane and the right-beam lane,
    # - south covers the source and reaches toward the east so the down beam can
    #   turn into the right beam's inherited-color trail.
    n_end_hi = w - t_e - 2
    n_end_lo = max(sc + sz - 1, n_end_hi - 8)
    if n_end_lo > n_end_hi:
        return None
    n0 = random.randint(max(0, sc - 4), sc)
    n1 = random.randint(n_end_lo, n_end_hi)

    e1_lo = max(sr + sz - 1, lane_n + sz - 1)
    e1_hi = min(lane_s - 1, h - t_s - 2)
    if e1_lo > e1_hi:
        return None
    e0 = random.randint(t_n, lane_n)
    e1 = random.randint(e1_lo, e1_hi)

    s1_hi = w - t_e - 2
    s1_lo = max(sc + sz - 1, s1_hi - 8)
    if s1_lo > s1_hi:
        return None
    s0 = random.randint(max(0, sc - 4), sc)
    s1 = random.randint(s1_lo, s1_hi)

    legend_colors = random.sample(LEGEND_COLOR_POOL, 4)
    obstacle_colors = random.sample(legend_colors, 3)
    side_color = {_N: obstacle_colors[0], _E: obstacle_colors[1], _S: obstacle_colors[2]}
    bg = random.choice([c for c in BG_COLOR_POOL if c not in legend_colors])

    base = np.full((h, w), bg, dtype=int)
    base[sr : sr + sz, sc : sc + sz] = TEAL

    occupied = np.zeros((h, w), dtype=bool)
    occupied[sr : sr + sz, sc : sc + sz] = True
    spans = {_N: (n0, n1), _E: (e0, e1), _S: (s0, s1)}
    thickness = {_N: t_n, _E: t_e, _S: t_s}
    for side in (_N, _E, _S):
        r0, r1, c0, c1 = _obstacle_rect(side, thickness[side], spans[side], h, w)
        if occupied[r0 : r1 + 1, c0 : c1 + 1].any():
            return None
        if not (r1 < sr - 1 or r0 > sr + sz or c1 < sc - 1 or c0 > sc + sz):
            return None
        occupied[r0 : r1 + 1, c0 : c1 + 1] = True
        base[r0 : r1 + 1, c0 : c1 + 1] = side_color[side]

    vals, cnts = np.unique(base, return_counts=True)
    if int(vals[cnts.argmax()]) != bg:
        return None

    try:
        expected, stats, beam_paints = _simulate_variant(
            base,
            bg,
            use_trails=True,
            global_turns=True,
            inherit_color=True,
        )
        no_trails, _stats_no_trails, _ = _simulate_variant(
            base,
            bg,
            use_trails=False,
            global_turns=True,
            inherit_color=True,
        )
        reset_parity, _stats_reset, _ = _simulate_variant(
            base,
            bg,
            use_trails=True,
            global_turns=False,
            inherit_color=True,
        )
        fixed_teal, _stats_fixed, _ = _simulate_variant(
            base,
            bg,
            use_trails=True,
            global_turns=True,
            inherit_color=False,
        )
    except AssertionError:
        return None

    hit_counts = [len(info["hits"]) for info in stats]
    total_hits = sum(hit_counts)
    trail_hits = [
        hit
        for info in stats
        for hit in info["hits"]
        if hit["kind"] == "trail"
    ]
    nonteal_trail_hits = [hit for hit in trail_hits if int(hit["color"]) != TEAL]
    visible_non_teal = {
        color
        for paints in beam_paints
        for color in paints.values()
        if color != TEAL
    }

    if hit_counts != [2, 1, 2, 0]:
        return None
    if total_hits != 5:
        return None
    if len(trail_hits) != 1 or len(nonteal_trail_hits) != 1:
        return None
    if len(visible_non_teal) < 2:
        return None
    if np.array_equal(expected, no_trails):
        return None
    if np.array_equal(expected, reset_parity):
        return None
    if np.array_equal(expected, fixed_teal):
        return None

    legend_dirs = _random_legend_dirs(legend_colors)
    legend = _make_legend(legend_colors, legend_dirs)
    inp = np.vstack([legend, base])

    try:
        actual = main(inp)
    except AssertionError:
        return None
    if not np.array_equal(actual, expected):
        return None

    return inp


def generate_input() -> np.ndarray:
    for _ in range(MAX_RETRIES):
        grid = _gen_attempt()
        if grid is not None:
            return grid
    raise RuntimeError("failed to generate e87109e9 combo variant input")
