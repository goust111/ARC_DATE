# Combo variant of e87109e9 (index 6)
# combo_spec: D7/0+D8/1
# Name: 统一方块加高密度障碍

import random

import numpy as np

from arc_utils import bbox_coords, components_mask


_DIRS = [(-1, 0), (0, 1), (1, 0), (0, -1)]
_N, _E, _S, _W = range(4)

LEGEND_BG = 5
TEAL = 8
LEGEND_H = 6
GRID_H = 24
GRID_W = 24
SOURCE_SIZE = 2
OBS_SIZE = 3
MAX_RETRIES = 400

LEGEND_COLOR_POOL = [1, 2, 3, 4, 6]
BG_COLOR_POOL = [1, 2, 3, 4, 6, 7, 9]
LEGEND_CELL_STARTS = [1, 7, 13, 19]


# Design analysis for D7/0 + D8/1
# ---------------------------------
# Concrete behavior:
# - In the original task, 3-4 large edge bars usually gave each beam only 1-2 turns.
# - Here the modified dimension is obstacle geometry and density: every obstacle is the same 3x3 square,
#   and there are 6-8 of them.
# - Example branch: with a 2x2 teal source at (11,11), the north beam can travel to state (6,11), hit a 3x3
#   square at (3,10), turn east, travel to (6,16), hit another 3x3 square at (5,18), then turn north and exit.
#   The rule is unchanged, but the path now depends on repeated beam simulation rather than a single side obstacle.
#
# Complexity calibration:
# - All obstacles look alike geometrically, so color+legend is the only turn signal.
# - The generator uses 4 mandatory first-hit squares plus 2-4 second-hit squares, yielding 6-8 obstacles total.
# - Source size is fixed at 2x2 on purpose: the fusion target is obstacle shape/count, and fixing the beam size keeps
#   the dense layouts legible while leaving enough room for repeated bends.
#
# Assumption audit against the original code:
# - Obstacles are no longer long edge-touching rectangles, so the old side/span construction is discarded.
# - Multiple obstacles can share a color, so the solver must map color -> turn, not obstacle identity -> turn.
# - Beams can turn well beyond the old 0-2-hit cap, so hit-count limits must be removed.
# - Dense layouts increase the risk of mixed-color collisions; the generator therefore spaces branch lanes so that any
#   single next-step footprint intersects at most one obstacle square.
#
# Placement legality and observability:
# - Every obstacle is a disjoint 3x3 square fully inside the 24x24 main grid and at least one cell away from the source.
# - Every placed obstacle is intentionally hit by some beam; no decorative dead obstacles are added.
# - Repeated colors support human inference: each of the four legend colors appears on multiple same-sized squares or
#   at least once with a matching turn class, so the visual difference from the original task is obvious in few-shot ARC.
#
# New solver assumptions guaranteed by the generator:
# - Legend background color 5 appears only in the top legend band.
# - There is exactly one solid teal source square.
# - All obstacle colors appear in the legend.
# - Beam paths are acyclic, and each collision footprint contains at most one obstacle color.


def _rot_ccw(d: int) -> int:
    return (d - 1) % 4


def _rot_cw(d: int) -> int:
    return (d + 1) % 4


def _rot(d: int, turn: str) -> int:
    return _rot_ccw(d) if turn == "L" else _rot_cw(d)


def _parse_legend(grid: np.ndarray, legend_h: int) -> dict[int, str]:
    legend = grid[:legend_h, :]
    assert legend.size, "legend must be non-empty"
    vals, counts = np.unique(legend, return_counts=True)
    bg = int(vals[counts.argmax()])

    sep_cols = [c for c in range(legend.shape[1]) if np.all(legend[:, c] == bg)]
    assert len(sep_cols) >= 2, "legend separator columns must exist"

    turns: dict[int, str] = {}
    for a, b in zip(sep_cols, sep_cols[1:], strict=False):
        if b - a <= 2:
            continue
        cell = legend[1 : legend_h - 1, a + 1 : b]
        colors = set(map(int, np.unique(cell)))
        colors.discard(bg)
        colors.discard(0)
        assert len(colors) == 1, "each legend cell must encode exactly one color"
        color = int(next(iter(colors)))
        xs = np.where(cell == color)[1]
        bar_x = int(xs[0])
        turns[color] = "L" if bar_x <= (cell.shape[1] - 1) // 2 else "R"
    assert len(turns) == 4, "legend must encode four turn colors"
    return turns


def _find_teal_square(g: np.ndarray, teal: int = TEAL) -> tuple[int, int, int]:
    candidates: list[tuple[int, int, int]] = []
    for cells in components_mask(g == teal, connectivity=4):
        r0, c0, r1, c1 = bbox_coords(cells)
        h = r1 - r0 + 1
        w = c1 - c0 + 1
        if h != w:
            continue
        if int(np.sum(g[r0 : r1 + 1, c0 : c1 + 1] == teal)) != h * w:
            continue
        candidates.append((int(r0), int(c0), int(h)))
    assert len(candidates) == 1, "main grid must contain exactly one solid teal square"
    return candidates[0]


def _simulate_beams(
    base: np.ndarray,
    turns: dict[int, str],
    bg: int,
    obstacle_ids: np.ndarray | None = None,
) -> tuple[np.ndarray, list[list[tuple[int, int, int, int, int | None]]]]:
    out = base.copy()
    sr, sc, sz = _find_teal_square(base)
    h, w = out.shape
    hit_log: list[list[tuple[int, int, int, int, int | None]]] = []

    def in_bounds(r: int, c: int) -> bool:
        return 0 <= r and 0 <= c and r + sz - 1 < h and c + sz - 1 < w

    def paint(r: int, c: int) -> None:
        blk = out[r : r + sz, c : c + sz]
        blk[blk == bg] = TEAL
        out[r : r + sz, c : c + sz] = blk

    def obstacle_info(nr: int, nc: int) -> tuple[int | None, int | None]:
        blk = out[nr : nr + sz, nc : nc + sz]
        colors = set(map(int, np.unique(blk)))
        colors.discard(bg)
        colors.discard(TEAL)
        if not colors:
            return None, None
        assert len(colors) == 1, "beam footprint must not hit mixed obstacle colors"
        oid = None
        if obstacle_ids is not None:
            ids = set(map(int, np.unique(obstacle_ids[nr : nr + sz, nc : nc + sz])))
            ids.discard(-1)
            if ids:
                assert len(ids) == 1, "beam footprint must not hit multiple obstacle squares"
                oid = int(next(iter(ids)))
        return int(next(iter(colors))), oid

    for d0 in range(4):
        r, c, d = sr, sc, d0
        seen: set[tuple[int, int, int]] = set()
        beam_hits: list[tuple[int, int, int, int, int | None]] = []
        while True:
            state = (r, c, d)
            assert state not in seen, "beam path must not cycle"
            seen.add(state)

            dr, dc = _DIRS[d]
            nr, nc = r + dr, c + dc
            if not in_bounds(nr, nc):
                break

            color, oid = obstacle_info(nr, nc)
            if color is None:
                r, c = nr, nc
                paint(r, c)
                continue

            assert color in turns, "obstacle color must be encoded by the legend"
            beam_hits.append((d, color, nr, nc, oid))
            d = _rot(d, turns[color])
        hit_log.append(beam_hits)
    return out, hit_log


def main(grid: np.ndarray) -> np.ndarray:
    """
    Trace four teal square beams through a dense field of equal-sized square obstacles.

    Input structure:
    - The top band is a 6-row legend whose grey background (5) surrounds four colored vertical bars.
    - Each bar's horizontal side inside its legend cell encodes the turn for that color: left means counterclockwise,
      right means clockwise.
    - Below the legend is the main grid with one solid teal source square and 6-8 obstacle squares of the same size,
      with obstacle colors drawn from the legend colors.

    Rule:
    - Remove the legend in the output.
    - Launch a sz x sz teal beam from the source in each cardinal direction.
    - A beam paints teal only onto background cells. Whenever its next footprint overlaps an obstacle color, it does
      not enter the obstacle; instead it rotates according to that color's legend entry and keeps going.
    - Beams may turn several times because the obstacle field is dense. Each beam stops only when its next move would
      leave the cropped main grid.

    Key insight:
    - After the fusion, obstacle geometry no longer distinguishes anything: every blocker is the same square. The only
      actionable information is the color-to-turn mapping in the legend, so the task is to repeatedly simulate the beam
      footprints through multiple bends until they exit.
    """
    g = np.asarray(grid, dtype=int)

    rows_with_legend = np.where(np.any(g == LEGEND_BG, axis=1))[0]
    assert rows_with_legend.size, "legend rows must be present"
    legend_h = int(rows_with_legend.max() + 1)
    assert np.array_equal(rows_with_legend, np.arange(legend_h)), "legend background must stay in the top band"

    turns = _parse_legend(g, legend_h)
    base = g[legend_h:, :].copy()
    assert base.size, "main grid must remain after cropping the legend"

    vals, counts = np.unique(base, return_counts=True)
    bg = int(vals[counts.argmax()])
    out, _ = _simulate_beams(base, turns, bg)
    return out


def _make_legend(legend_colors: list[int], turns: dict[int, str]) -> np.ndarray:
    legend = np.full((LEGEND_H, GRID_W), LEGEND_BG, dtype=int)
    for start, color in zip(LEGEND_CELL_STARTS, legend_colors):
        legend[1:5, start : start + 4] = 0
        bar_col = start if turns[color] == "L" else start + 3
        legend[1:5, bar_col] = color
    return legend


def _hit_obstacle_top_left(r: int, c: int, d: int) -> tuple[int, int]:
    if d == _N:
        return r - 3, c - 1
    if d == _E:
        return r - 1, c + 2
    if d == _S:
        return r + 2, c
    return r, c - 3


def _branch_first_state(sr: int, sc: int, d: int, steps: int) -> tuple[int, int]:
    dr, dc = _DIRS[d]
    return sr + dr * steps, sc + dc * steps


def _place_square(
    base: np.ndarray,
    obstacle_ids: np.ndarray,
    oid: int,
    r0: int,
    c0: int,
    color: int,
) -> bool:
    if r0 < 0 or c0 < 0 or r0 + OBS_SIZE > base.shape[0] or c0 + OBS_SIZE > base.shape[1]:
        return False
    if np.any(obstacle_ids[r0 : r0 + OBS_SIZE, c0 : c0 + OBS_SIZE] != -1):
        return False
    base[r0 : r0 + OBS_SIZE, c0 : c0 + OBS_SIZE] = color
    obstacle_ids[r0 : r0 + OBS_SIZE, c0 : c0 + OBS_SIZE] = oid
    return True


def _color_sequence(pool: list[int], count: int) -> list[int]:
    if count <= 0:
        return []
    seq = pool[:]
    while len(seq) < count:
        seq.append(random.choice(pool))
    random.shuffle(seq)
    return seq[:count]


def _gen_attempt() -> tuple[np.ndarray, dict[str, int]] | None:
    h, w = GRID_H, GRID_W
    sr = random.randint(9, 11)
    sc = random.randint(9, 11)
    bg_candidates = BG_COLOR_POOL[:]

    legend_colors = random.sample(LEGEND_COLOR_POOL, 4)
    l_colors = random.sample(legend_colors, 2)
    r_colors = [c for c in legend_colors if c not in l_colors]
    turns = {c: "L" for c in l_colors} | {c: "R" for c in r_colors}
    bg = random.choice([c for c in bg_candidates if c not in legend_colors])

    outer_dirs = set(random.sample([_N, _E, _S, _W], random.randint(2, 4)))
    first_steps = {d: random.randint(4, 6) for d in (_N, _E, _S, _W)}
    second_steps = {d: random.randint(4, 6) for d in outer_dirs}

    base = np.full((h, w), bg, dtype=int)
    base[sr : sr + SOURCE_SIZE, sc : sc + SOURCE_SIZE] = TEAL
    obstacle_ids = np.full((h, w), -1, dtype=int)

    first_colors = _color_sequence(r_colors, 4)
    second_colors = _color_sequence(l_colors, len(outer_dirs))

    obstacles: list[dict[str, int]] = []
    oid = 0

    for idx, d in enumerate((_N, _E, _S, _W)):
        r1, c1 = _branch_first_state(sr, sc, d, first_steps[d])
        or0, oc0 = _hit_obstacle_top_left(r1, c1, d)
        color = first_colors[idx]
        if not _place_square(base, obstacle_ids, oid, or0, oc0, color):
            return None
        obstacles.append({"id": oid, "dir": d, "turn": 1, "color": color})
        oid += 1

    for color, d in zip(second_colors, sorted(outer_dirs)):
        d1 = _rot(d, "R")
        r1, c1 = _branch_first_state(sr, sc, d, first_steps[d])
        r2, c2 = _branch_first_state(r1, c1, d1, second_steps[d])
        or0, oc0 = _hit_obstacle_top_left(r2, c2, d1)
        if not _place_square(base, obstacle_ids, oid, or0, oc0, color):
            return None
        obstacles.append({"id": oid, "dir": d, "turn": 2, "color": color})
        oid += 1

    try:
        expected, hits = _simulate_beams(base, turns, bg, obstacle_ids)
    except AssertionError:
        return None

    hit_ids = [oid for beam in hits for _, _, _, _, oid in beam if oid is not None]
    if len(hit_ids) != len(obstacles):
        return None
    if set(hit_ids) != {obs["id"] for obs in obstacles}:
        return None

    turns_per_beam = [len(beam) for beam in hits]
    if sorted(turns_per_beam, reverse=True)[0] < 2:
        return None
    if sum(turns_per_beam) != len(obstacles):
        return None
    if sum(t >= 2 for t in turns_per_beam) < len(outer_dirs):
        return None

    legend = _make_legend(legend_colors, turns)
    inp = np.vstack([legend, base])
    try:
        actual = main(inp)
    except AssertionError:
        return None
    if not np.array_equal(actual, expected):
        return None

    metrics = {
        "obstacle_count": len(obstacles),
        "total_turns": sum(turns_per_beam),
        "max_turns_per_beam": max(turns_per_beam),
        "multi_turn_beams": sum(t >= 2 for t in turns_per_beam),
    }
    return inp, metrics


def generate_input() -> np.ndarray:
    for _ in range(MAX_RETRIES):
        result = _gen_attempt()
        if result is not None:
            return result[0]
    raise RuntimeError("failed to generate fused e87109e9 input")
