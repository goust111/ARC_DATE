# Combo variant of da515329 (index 30)
# combo_spec: D2/1+D5/0+D14/1
# Name: 2像素间距 + 双源 + 四色

# Design analysis
# ---------------
# a) Concretization and behavior trace
#    - Input stays minimal: one hollow teal plus-sign cross on black, with four equal arms.
#    - The fused output no longer launches from all four tips. Only one opposing axis is active:
#      the pair whose tips visibly have the larger minimum border clearance. This makes the choice
#      inferable from the input placement instead of arbitrary hidden metadata.
#    - The original solver's visible spacing came from 1/2 local stride lengths. For the fused
#      "2-pixel corridor" variant, the same right-hand turtle logic is kept but all nonzero moves
#      are scaled by 3: turn-step = 3 cells and long straight = 6 cells. With the same adjacency
#      blocker, successive spiral lanes land three rows/cols apart, leaving two blank cells
#      between parallel arms.
#    - Small trace: if the upper tip is active, the source task would paint a 1-cell turn and then
#      2-cell straights in teal. Here it paints a 3-cell turn and 6-cell straights in red, while
#      the opposite lower tip does the same in green. The left/right tips remain just part of the
#      teal cross and never emit.
# b) Complexity calibration
#    - There are only two active spirals per sample, which strongly changes density and makes the
#      sparse weave obvious.
#    - The four-color idea is preserved as a fixed direction-to-color map, but each sample shows
#      only the two colors belonging to the active opposing pair. Across samples both vertical
#      and horizontal pairings can appear.
#    - The active-axis rule uses one visually legible cue: which axis has more room to the border.
#      Generator enforces a clear margin so humans can see it from 2-4 examples.
# c) Hidden-assumption review
#    - Original code assumed four agents and nearly full tiling; its parity cleanup and border-cap
#      repairs depended on that. Those post-process writes are invalid for sparse two-source output
#      and are removed entirely.
#    - Occupancy checks must depend on nonzero, not literal 8, because spiral trails are colored.
#    - The original "short arm" bootstrap remains conceptually valid, but its launch step must be
#      scaled from 1 to 3 so the path joins the new lane lattice.
# d) Placement legality
#    - Generator uses size = 4*arm + 12 and keeps each tip at least arm+3 from every border.
#    - It validates every candidate by running the solver and only keeps placements with two visible
#      spiral colors, sufficient spiral length on both sources, and moderate sparse fill.
# e) Visual observability
#    - Three cues are repeated across samples: only one opposing pair grows, the chosen pair is the
#      roomier axis, and the resulting corridors are much wider than in the source task.
#    - Because the color map is direction-locked, red/green imply vertical activity and blue/yellow
#      imply horizontal activity. The input placement explains which pair should activate.
# f) Visual cleanliness
#    - Only black background, teal cross, and exactly two spiral colors appear in each output.
#    - Removing the dense end-game patch stage avoids isolated caps that would look unrelated to
#      the sparse double-spiral rule.
# g) Change inferability
#    - Color changes are structural rather than arbitrary: color is determined by endpoint direction.
#    - Source selection is structural rather than arbitrary: it follows the visibly larger axis
#      clearance. Generator rejects near-tie placements so the cue does not collapse.
# new_main_assumptions_checked_against_generator:
#    - The input contains exactly one teal plus-sign cross with four non-empty equal arms.
#    - One axis has at least a 3-cell clearance advantage, so active-pair selection is unambiguous.
#    - Both active spirals can each paint at least 15 colored cells, preventing degenerate outputs.

import random

import numpy as np

from arc_utils import neighbors4

_DIRS_URDL = [(-1, 0), (0, 1), (1, 0), (0, -1)]
_DIR_TO_COLOR = {
    (-1, 0): 2,  # up tip -> red
    (0, 1): 1,   # right tip -> blue
    (1, 0): 3,   # down tip -> green
    (0, -1): 4,  # left tip -> yellow
}

BG = 0
CROSS_COLOR = 8
TURN_LEN = 3
STRAIGHT_LEN = 6

# --------------- generator cfg ---------------
ARM_LEN_MIN, ARM_LEN_MAX = 1, 4
GRID_SIZE_BASE = 12
TIP_GAP_MIN = 3
FILL_LO, FILL_HI = 0.16, 0.32
MIN_AXIS_MARGIN = 3
MIN_SPIRAL_PIXELS = 15
MAX_OUTER_RETRIES = 300
# ------------- end generator cfg -------------

_POOL_BY_ARM: dict[int, list[tuple[int, int, int]]] | None = None


def _right(d: tuple[int, int]) -> tuple[int, int]:
    return _DIRS_URDL[(_DIRS_URDL.index(d) + 1) % 4]


def _left(d: tuple[int, int]) -> tuple[int, int]:
    return _DIRS_URDL[(_DIRS_URDL.index(d) + 3) % 4]


def _extract_arms(grid: np.ndarray) -> tuple[int, int, dict[tuple[int, int], list[tuple[int, int]]]]:
    h, w = grid.shape
    row_counts = np.sum(grid == CROSS_COLOR, axis=1)
    col_counts = np.sum(grid == CROSS_COLOR, axis=0)
    cr = int(np.argmax(row_counts))
    cc = int(np.argmax(col_counts))

    def inb(r: int, c: int) -> bool:
        return 0 <= r < h and 0 <= c < w

    arms: dict[tuple[int, int], list[tuple[int, int]]] = {}
    for d in _DIRS_URDL:
        dr, dc = d
        r, c = cr, cc
        cells: list[tuple[int, int]] = []
        while inb(r + dr, c + dc) and int(grid[r + dr, c + dc]) == CROSS_COLOR:
            r += dr
            c += dc
            cells.append((r, c))
        if cells:
            arms[d] = cells
    return cr, cc, arms


def _choose_active_pair(
    arms: dict[tuple[int, int], list[tuple[int, int]]],
    h: int,
    w: int,
) -> tuple[tuple[tuple[int, int], tuple[int, int]], dict[str, int | str]]:
    top_tip = arms[(-1, 0)][-1]
    right_tip = arms[(0, 1)][-1]
    bottom_tip = arms[(1, 0)][-1]
    left_tip = arms[(0, -1)][-1]

    top_gap = top_tip[0]
    bottom_gap = (h - 1) - bottom_tip[0]
    left_gap = left_tip[1]
    right_gap = (w - 1) - right_tip[1]

    vertical_gap = min(top_gap, bottom_gap)
    horizontal_gap = min(left_gap, right_gap)

    if vertical_gap > horizontal_gap or (
        vertical_gap == horizontal_gap and top_gap + bottom_gap >= left_gap + right_gap
    ):
        active_pair = ((-1, 0), (1, 0))
        active_axis = "vertical"
    else:
        active_pair = ((0, 1), (0, -1))
        active_axis = "horizontal"

    meta: dict[str, int | str] = {
        "active_axis": active_axis,
        "vertical_gap": vertical_gap,
        "horizontal_gap": horizontal_gap,
        "axis_margin": abs(vertical_gap - horizontal_gap),
        "top_gap": top_gap,
        "bottom_gap": bottom_gap,
        "left_gap": left_gap,
        "right_gap": right_gap,
    }
    return active_pair, meta


def _solve(grid: np.ndarray, return_stats: bool = False):
    grid = np.asarray(grid, dtype=int)
    h, w = grid.shape
    out = grid.copy()

    _cr, _cc, arms = _extract_arms(grid)
    active_pair, axis_meta = _choose_active_pair(arms, h, w)

    def inb(r: int, c: int) -> bool:
        return 0 <= r < h and 0 <= c < w

    def is_border(r: int, c: int) -> bool:
        return r == 0 or r == h - 1 or c == 0 or c == w - 1

    def add(pos: tuple[int, int], d: tuple[int, int], k: int = 1) -> tuple[int, int]:
        return (pos[0] + d[0] * k, pos[1] + d[1] * k)

    def spacing_ok(
        cell: tuple[int, int],
        prev: tuple[int, int],
        start: tuple[int, int],
    ) -> bool:
        r, c = cell
        for nr, nc in neighbors4(r, c):
            if inb(nr, nc) and out[nr, nc] != BG and (nr, nc) != prev and (nr, nc) != start:
                return False
        return True

    def can_move(cur: tuple[int, int], d: tuple[int, int], length: int) -> bool:
        prev = cur
        for t in range(1, length + 1):
            rr, cc2 = add(cur, d, t)
            if not inb(rr, cc2) or out[rr, cc2] != BG:
                return False
            if not spacing_ok((rr, cc2), prev, cur):
                return False
            prev = (rr, cc2)
        return True

    agents: list[dict[str, object]] = []
    for outward in active_pair:
        cells = arms[outward]
        agents.append(
            {
                "id": len(agents),
                "out_dir": outward,
                "cur": cells[-1],
                "dir": _right(outward),
                "color": _DIR_TO_COLOR[outward],
                "phase": "out_first" if len(cells) == 1 else None,
                "done": False,
                "p": 0,
                "pixels": 0,
                "turn_moves": 0,
                "long_moves": 0,
                "short_straights": 0,
            }
        )

    def do_move(agent: dict[str, object], d: tuple[int, int], length: int, kind: str) -> None:
        r0, c0 = agent["cur"]
        for t in range(1, length + 1):
            rr, cc2 = r0 + d[0] * t, c0 + d[1] * t
            out[rr, cc2] = int(agent["color"])
            agent["pixels"] = int(agent["pixels"]) + 1
        agent["cur"] = (r0 + d[0] * length, c0 + d[1] * length)
        if kind == "turn":
            agent["turn_moves"] = int(agent["turn_moves"]) + 1
        elif kind == "long":
            agent["long_moves"] = int(agent["long_moves"]) + 1
        else:
            agent["short_straights"] = int(agent["short_straights"]) + 1

    def step_agent(agent: dict[str, object]) -> bool:
        if bool(agent["done"]):
            return False

        if agent["phase"] == "out_first":
            if can_move(agent["cur"], agent["out_dir"], TURN_LEN):
                do_move(agent, agent["out_dir"], TURN_LEN, "turn")
            agent["phase"] = None
            agent["dir"] = _right(agent["out_dir"])
            agent["p"] = 0
            return True

        cur = agent["cur"]
        dirs = [_right(agent["dir"]), agent["dir"]]
        if not is_border(*cur):
            dirs.append(_left(agent["dir"]))

        for nd in dirs:
            turned = nd != agent["dir"]
            length = TURN_LEN if turned else (STRAIGHT_LEN if int(agent["p"]) == 0 else TURN_LEN)
            if can_move(agent["cur"], nd, length):
                do_move(agent, nd, length, "turn" if turned else ("long" if length == STRAIGHT_LEN else "short"))
                agent["p"] = 1 if (int(agent["p"]) == 0 and turned) else 0
                agent["dir"] = nd
                return True
            if not turned and length == STRAIGHT_LEN and can_move(agent["cur"], nd, TURN_LEN):
                do_move(agent, nd, TURN_LEN, "short")
                agent["p"] = 0
                return True

        agent["done"] = True
        return False

    guard = 0
    while guard < 200000:
        guard += 1
        changed = False
        for agent in agents:
            if step_agent(agent):
                changed = True
        if not changed or all(bool(agent["done"]) for agent in agents):
            break

    if not return_stats:
        return out

    spiral_colors = sorted(set(int(x) for x in out.flatten()) - {BG, CROSS_COLOR})
    per_color_pixels = {
        int(agent["color"]): int(agent["pixels"])
        for agent in agents
    }
    stats = {
        **axis_meta,
        "fill": float(np.mean(out != BG)),
        "spiral_color_count": len(spiral_colors),
        "spiral_colors": spiral_colors,
        "per_color_pixels": per_color_pixels,
        "min_spiral_pixels": min(per_color_pixels.values()) if per_color_pixels else 0,
        "max_spiral_pixels": max(per_color_pixels.values()) if per_color_pixels else 0,
        "turn_moves": {int(agent["color"]): int(agent["turn_moves"]) for agent in agents},
        "long_moves": {int(agent["color"]): int(agent["long_moves"]) for agent in agents},
        "short_straights": {int(agent["color"]): int(agent["short_straights"]) for agent in agents},
    }
    return out, stats


def main(grid: np.ndarray) -> np.ndarray:
    """
    Grow a sparse pair of clockwise rectangular spirals from the roomier opposing tips of a teal cross.

    Input structure:
    - bg = 0 (black). The only foreground object is one teal (8) plus-sign cross with a 1-cell center hole.

    Rule:
    - Compare the cross's two axes by visible tip-to-border clearance; only the axis with more room is active,
      so either the up/down pair or the left/right pair emits spirals.
    - Each active tip still follows the same clockwise turtle policy as the source task: prefer turn-right,
      then straight, then left when not on a border.
    - The spacing is widened by scaling the local stride pattern to 3/6 cells instead of 1/2, which leaves
      two blank cells between neighboring parallel spiral lanes. The two active directions paint with their
      own fixed colors, while the original cross stays teal.

    Key insight: This is not a dense four-source tiling anymore. Solving requires identifying the roomier axis,
    then applying the original right-hand spiral logic on a coarser 3-cell lane lattice, with direction-based
    colors for the two active endpoints.
    """

    return _solve(grid, return_stats=False)


def _build_input(arm: int, size: int, cr: int, cc: int) -> np.ndarray:
    grid = np.zeros((size, size), dtype=int)
    offsets = np.arange(1, arm + 1)
    grid[cr - offsets, cc] = CROSS_COLOR
    grid[cr + offsets, cc] = CROSS_COLOR
    grid[cr, cc - offsets] = CROSS_COLOR
    grid[cr, cc + offsets] = CROSS_COLOR
    return grid


def _candidate_ok(stats: dict[str, object]) -> bool:
    if int(stats["axis_margin"]) < MIN_AXIS_MARGIN:
        return False
    if int(stats["spiral_color_count"]) != 2:
        return False
    if int(stats["min_spiral_pixels"]) < MIN_SPIRAL_PIXELS:
        return False
    fill = float(stats["fill"])
    if not (FILL_LO <= fill <= FILL_HI):
        return False
    return True


def _build_pool() -> dict[int, list[tuple[int, int, int]]]:
    pool_by_arm: dict[int, list[tuple[int, int, int]]] = {}
    for arm in range(ARM_LEN_MIN, ARM_LEN_MAX + 1):
        size = 4 * arm + GRID_SIZE_BASE
        lo = arm + TIP_GAP_MIN
        hi = size - arm - TIP_GAP_MIN - 1
        candidates: list[tuple[int, int, int]] = []
        for cr in range(lo, hi + 1):
            for cc in range(lo, hi + 1):
                inp = _build_input(arm, size, cr, cc)
                try:
                    out, stats = _solve(inp, return_stats=True)
                except Exception:
                    continue
                if not _candidate_ok(stats):
                    continue
                if not np.array_equal(_solve(inp, return_stats=False), out):
                    continue
                candidates.append((size, cr, cc))
        if not candidates:
            raise RuntimeError(f"no valid placements found for arm={arm}")
        pool_by_arm[arm] = candidates

    active_axes = set()
    for arm, entries in pool_by_arm.items():
        for size, cr, cc in entries:
            _out, stats = _solve(_build_input(arm, size, cr, cc), return_stats=True)
            active_axes.add(str(stats["active_axis"]))
            if active_axes == {"vertical", "horizontal"}:
                break
        if active_axes == {"vertical", "horizontal"}:
            break
    if active_axes != {"vertical", "horizontal"}:
        raise RuntimeError("generator pool failed to preserve both active-axis cases")

    return pool_by_arm


def generate_input() -> np.ndarray:
    global _POOL_BY_ARM
    if _POOL_BY_ARM is None:
        _POOL_BY_ARM = _build_pool()

    for _ in range(MAX_OUTER_RETRIES):
        arm = random.choice(sorted(_POOL_BY_ARM))
        size, cr, cc = random.choice(_POOL_BY_ARM[arm])
        grid = _build_input(arm, size, cr, cc)
        _out, stats = _solve(grid, return_stats=True)
        if _candidate_ok(stats):
            return grid
    raise RuntimeError("generate_input: failed after retries")
