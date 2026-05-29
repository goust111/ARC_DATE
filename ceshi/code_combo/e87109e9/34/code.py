# Combo variant of e87109e9 (index 34)
# combo_spec: D5/0+D7/2+D9/0
# Name: 全像素级精细系统

from __future__ import annotations

import random

import numpy as np

# Design analysis for the fused variant:
# a) Concrete behavior shift:
#    Original: a 2x2 teal square emits 2-cell-wide beams, and wide obstacle rectangles
#    can be contacted by multiple "lanes" because they occupy several cells.
#    Fused: the source is one teal pixel, each obstacle is one colored pixel, and each
#    beam is a 1-pixel trace. Example: if the source is at (10,10) and a north obstacle
#    sits at (6,10), the north beam paints (9,10),(8,10),(7,10); when its next step
#    would enter (6,10), it turns in place according to the legend and continues from
#    the last painted pixel. The rule is unchanged, but the geometry is exact point-hit
#    rather than block-overlap.
# b) Complexity calibration:
#    Pure single-pixel obstacles can become too sparse if every color is used only once.
#    To keep the rule inferable, the generator still uses exactly four legend colors and
#    reuses obstacle colors across several singleton obstacles, so the left/right turn
#    categories have repeated support. This avoids a "one color, one accident" puzzle.
# c) Broken implicit assumptions from the original task distribution:
#    - The original dataset/generator assumed a teal source block with visible area.
#      In the fused variant the input must contain exactly one teal source pixel instead.
#    - Beam collision formerly inspected an sz x sz footprint and relied on the fact
#      that a wide obstacle could be hit by several shifted lanes. With 1x1 obstacles,
#      collision reduces to checking one next cell.
#    - The original generator could let one rectangle serve as both a first hit and a
#      later side hit. That is no longer possible geometrically, so the fused generator
#      uses multiple singleton obstacles that may share a color/turn rule instead.
# d) Placement legality:
#    The generator places the source away from edges, chooses obstacle cells in-bounds,
#    keeps all obstacle cells distinct, and rejects any sample whose simulated beams hit
#    obstacles in an unintended order or count.
# e) Visual observability:
#    Humans can still see the same legend-to-turn mapping because the top legend is kept
#    intact and obstacle colors repeat. What changes visibly is that the bottom system
#    becomes a sparse pixel circuit rather than chunky rectangles and thick bars.
# f) Visual cleanliness:
#    The source is centered with generous margins, first-hit obstacles are at least three
#    cells away, and only 0-2 beams get a second turn. This keeps the output readable
#    instead of producing dense self-overlapping pixel mazes.
# g) Inferability of color behavior:
#    Colors do not perform arbitrary remapping; each obstacle color only means "turn L"
#    or "turn R" exactly as encoded in the legend. The generator ensures both turn types
#    appear and usually uses 3-4 obstacle colors so the mapping has repeated evidence.

_DIRS = [(-1, 0), (0, 1), (1, 0), (0, -1)]
_N, _E, _S, _W = range(4)

LEGEND_COLOR_POOL = [1, 2, 3, 4, 6]
BG_COLOR_POOL = [1, 2, 3, 4, 6, 7, 9]
LEGEND_BG = 5
TEAL = 8

GRID_H_RANGE = (19, 24)
GRID_W = 24
LEGEND_H = 6
LEGEND_CELL_STARTS = [1, 7, 13, 19]

SOURCE_MARGIN = 6
FIRST_HIT_MIN = 3
SECOND_HIT_MIN = 3
MAX_RETRIES = 500


def _parse_legend(grid: np.ndarray, legend_h: int) -> dict[int, str]:
    """Read the top legend and map each legend color to 'L' or 'R'."""
    legend = grid[:legend_h, :]
    assert legend.size, "legend must be non-empty"
    vals, cnts = np.unique(legend, return_counts=True)
    bg = int(vals[cnts.argmax()])

    sep_cols = [c for c in range(legend.shape[1]) if np.all(legend[:, c] == bg)]
    assert len(sep_cols) >= 2, "legend separator columns must exist"

    turn: dict[int, str] = {}
    for a, b in zip(sep_cols, sep_cols[1:], strict=False):
        if b - a <= 2:
            continue
        interior = legend[1 : legend_h - 1, a + 1 : b]
        colors = set(map(int, np.unique(interior)))
        colors.discard(bg)
        colors.discard(0)
        assert len(colors) == 1, "each legend cell must encode exactly one color"
        color = int(next(iter(colors)))
        xs = np.where(interior == color)[1]
        x = int(xs[0])
        turn[color] = "L" if x <= (interior.shape[1] - 1) // 2 else "R"
    assert len(turn) == 4, "legend must encode four turn colors"
    return turn


def _find_teal_source(base: np.ndarray, teal: int = TEAL) -> tuple[int, int]:
    """Return the unique teal source pixel in the cropped main grid."""
    pts = np.argwhere(base == teal)
    assert len(pts) == 1, "main grid must contain exactly one teal source pixel"
    r, c = pts[0]
    return int(r), int(c)


def _rot_ccw(d: int) -> int:
    return (d - 1) % 4


def _rot_cw(d: int) -> int:
    return (d + 1) % 4


def _rot(d: int, turn: str) -> int:
    return _rot_ccw(d) if turn == "L" else _rot_cw(d)


def _step(pos: tuple[int, int], d: int, k: int = 1) -> tuple[int, int]:
    dr, dc = _DIRS[d]
    r, c = pos
    return int(r + dr * k), int(c + dc * k)


def _distance_to_edge(pos: tuple[int, int], d: int, h: int, w: int) -> int:
    r, c = pos
    if d == _N:
        return int(r)
    if d == _S:
        return int(h - 1 - r)
    if d == _W:
        return int(c)
    return int(w - 1 - c)


def main(grid: np.ndarray) -> np.ndarray:
    """
    Emit four 1-pixel teal beams from a single teal source, turning at singleton obstacles.

    Input structure:
    - Top band: a grey (5) legend with four color cells; each cell contains one colored
      vertical bar placed on the left or right side to encode the turn rule for that color.
    - Below: a main grid with one teal source pixel, several colored obstacle pixels, and
      a dominant non-5 background color.

    Rule:
    - Remove the legend from the output.
    - From the teal source, send one 1-pixel beam in each cardinal direction.
    - A beam paints teal only onto background cells.
    - When the beam's next cell is a colored obstacle pixel, do not enter that obstacle;
      instead turn counterclockwise for a left-bar color or clockwise for a right-bar color,
      then keep tracing from the current beam tip.
    - Beams stop at the grid boundary.

    Key insight: The puzzle is the same turning system as the original task, but every
    geometric object has been collapsed to single pixels, so solving it depends only on
    exact next-cell collisions plus the legend's left/right bar encoding.
    """
    g = np.asarray(grid, dtype=int)

    rows_with_legend = np.where(np.any(g == LEGEND_BG, axis=1))[0]
    assert rows_with_legend.size, "legend rows must be present"
    legend_h = int(rows_with_legend.max() + 1)
    assert np.array_equal(rows_with_legend, np.arange(legend_h)), "legend must be a top band"

    legend_turn = _parse_legend(g, legend_h)

    base = g[legend_h:, :].copy()
    assert base.size, "main grid must remain after legend crop"

    vals, cnts = np.unique(base, return_counts=True)
    bg = int(vals[cnts.argmax()])
    sr, sc = _find_teal_source(base)

    out = base.copy()
    h, w = out.shape

    def in_bounds(r: int, c: int) -> bool:
        return 0 <= r < h and 0 <= c < w

    for d0 in range(4):
        r, c, d = sr, sc, d0
        seen: set[tuple[int, int, int]] = set()
        while True:
            state = (r, c, d)
            assert state not in seen, "beam path must not cycle"
            seen.add(state)

            nr, nc = _step((r, c), d)
            if not in_bounds(nr, nc):
                break

            cell = int(out[nr, nc])
            if cell not in (bg, TEAL):
                assert cell in legend_turn, "obstacle color must be encoded by the legend"
                d = _rot(d, legend_turn[cell])
                continue

            r, c = nr, nc
            if out[r, c] == bg:
                out[r, c] = TEAL

    return out


def _simulate_paths(
    base: np.ndarray, turns: dict[int, str], bg: int
) -> tuple[np.ndarray, list[list[tuple[int, int, int, int]]]]:
    """Oracle simulator used by the generator."""
    out = base.copy()
    h, w = out.shape
    sr, sc = _find_teal_source(base)
    hits: list[list[tuple[int, int, int, int]]] = []

    def in_bounds(r: int, c: int) -> bool:
        return 0 <= r < h and 0 <= c < w

    for d0 in range(4):
        r, c, d = sr, sc, d0
        beam_hits: list[tuple[int, int, int, int]] = []
        seen: set[tuple[int, int, int]] = set()
        while True:
            state = (r, c, d)
            assert state not in seen
            seen.add(state)

            nr, nc = _step((r, c), d)
            if not in_bounds(nr, nc):
                break

            cell = int(out[nr, nc])
            if cell not in (bg, TEAL):
                assert cell in turns
                beam_hits.append((d, cell, nr, nc))
                d = _rot(d, turns[cell])
                continue

            r, c = nr, nc
            if out[r, c] == bg:
                out[r, c] = TEAL
        hits.append(beam_hits)

    return out, hits


def _make_legend(legend_colors: list[int], turns: dict[int, str]) -> np.ndarray:
    legend = np.full((LEGEND_H, GRID_W), LEGEND_BG, dtype=int)
    for start, color in zip(LEGEND_CELL_STARTS, legend_colors):
        legend[1:5, start : start + 4] = 0
        bar_col = start if turns[color] == "L" else start + 3
        legend[1:5, bar_col] = color
    return legend


def _sample_turns(legend_colors: list[int]) -> dict[int, str]:
    assignments = ["L", "L", "R", "R"]
    random.shuffle(assignments)
    return {color: turn for color, turn in zip(legend_colors, assignments)}


def _pick_color(turn: str, turns: dict[int, str]) -> int:
    pool = [color for color, val in turns.items() if val == turn]
    assert pool
    return int(random.choice(pool))


def _gen_attempt() -> np.ndarray | None:
    h, w = random.randint(*GRID_H_RANGE), GRID_W
    legend_colors = random.sample(LEGEND_COLOR_POOL, 4)
    turns = _sample_turns(legend_colors)
    bg = random.choice([c for c in BG_COLOR_POOL if c not in legend_colors])

    min_r, max_r = SOURCE_MARGIN, h - 1 - SOURCE_MARGIN
    min_c, max_c = SOURCE_MARGIN, w - 1 - SOURCE_MARGIN
    if min_r > max_r or min_c > max_c:
        return None
    sr = random.randint(min_r, max_r)
    sc = random.randint(min_c, max_c)

    base = np.full((h, w), bg, dtype=int)
    base[sr, sc] = TEAL

    obstacle_cells: dict[tuple[int, int], int] = {}
    second_hit_dirs = set(random.sample(range(4), random.choice([0, 1, 1, 2])))

    for d0 in range(4):
        room0 = _distance_to_edge((sr, sc), d0, h, w)
        if room0 < FIRST_HIT_MIN + 2:
            return None

        turn1 = random.choice(["L", "R"])
        d1 = _rot(d0, turn1)
        first_dist = random.randint(FIRST_HIT_MIN, room0 - 2)
        pos1 = _step((sr, sc), d0, first_dist)
        if pos1 in obstacle_cells or pos1 == (sr, sc):
            return None
        obstacle_cells[pos1] = _pick_color(turn1, turns)

        if d0 not in second_hit_dirs:
            continue

        pivot1 = _step(pos1, (d0 + 2) % 4, 1)
        room1 = _distance_to_edge(pivot1, d1, h, w)
        if room1 < SECOND_HIT_MIN + 2:
            return None

        second_dist = random.randint(SECOND_HIT_MIN, room1 - 2)
        pos2 = _step(pivot1, d1, second_dist)
        turn2 = "R" if turn1 == "L" else "L"
        if pos2 in obstacle_cells or pos2 == (sr, sc):
            return None
        obstacle_cells[pos2] = _pick_color(turn2, turns)

    used_colors = set(obstacle_cells.values())
    if len(used_colors) < 3:
        return None
    if {turns[c] for c in used_colors} != {"L", "R"}:
        return None

    for (r, c), color in obstacle_cells.items():
        base[r, c] = color

    try:
        expected, hits = _simulate_paths(base, turns, bg)
    except AssertionError:
        return None

    hit_counts = [len(beam) for beam in hits]
    hit_positions = {(r, c) for beam in hits for _, _, r, c in beam}

    if any(count < 1 or count > 2 for count in hit_counts):
        return None
    if sum(hit_counts) != len(obstacle_cells):
        return None
    if hit_positions != set(obstacle_cells):
        return None

    legend = _make_legend(legend_colors, turns)
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
    raise RuntimeError("failed to generate fused e87109e9 input")
