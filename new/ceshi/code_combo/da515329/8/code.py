# Combo variant of da515329 (index 8)
# combo_spec: D4/3+D5/0
# Name: T/L形源 + 仅2端点发射
#
# Design analysis
# fused_spec:
#   source_object:
#     base_task:
#       - one black hub cell with 4 equal teal rays
#       - every tip launches a clockwise rectangular spiral
#     combo_task:
#       - the hub is still a single black cell, but the visible teal source is now either:
#         1) a T shape with 3 equal rays, or
#         2) an L shape with 2 equal perpendicular rays
#       - only 2 tips may launch:
#         * for T, the active pair is the unique opposite pair on the crossbar
#         * for L, there is no opposite pair, so the two visible tips are the only candidates
#           and both launch
#       - this resolves the D4/3 + D5/0 interaction without hidden metadata
#   traced_example:
#     original_local_case:
#       - a source with [U, R, D, L] launches 4 agents from all arm tips
#       - all four quadrants receive interlocking spiral walls
#     fused_local_case:
#       - a T with [U, R, L] still has 3 visible endpoints
#       - only [R, L] are active because they are the only opposite pair
#       - the upper tip remains part of the source shape but seeds no spiral
#       - compared with the D4/3-only variant, one visible endpoint is intentionally silent
#     second_case:
#       - an L with [U, R] has exactly 2 visible endpoints and no opposite pair
#       - both endpoints launch, so the D4/3 source-shape cue remains valid while the
#         D5/0 restriction becomes a degenerate 2-candidate case
#   complexity_calibration:
#     - only 8 source orientations are allowed: 4 T orientations and 4 L orientations
#     - arm length stays in a small range [2, 4] so T/L shapes are legible
#     - generator strongly prefers T cases because they are the clearest evidence that
#       "present endpoint" and "active endpoint" are not the same thing
#   main_assumption_audit:
#     old_assumption_1:
#       code: row/col argmax recovers the hub of a 4-arm cross
#       status: broken
#       fix: scan black cells and validate the complete teal footprint around the hub
#     old_assumption_2:
#       code: every detected arm tip launches
#       status: broken for T cases
#       fix: separate present_dirs from active_dirs before spiral growth
#     old_assumption_3:
#       code: 2-arm inputs never occur
#       status: broken
#       fix: accept exactly 2 perpendicular rays as an L source
#     old_assumption_4:
#       code: parity/border cleanup depends only on hub parity and arm length
#       status: kept
#       reason: these corrections patch the global lattice after growth and do not depend
#         on how many launchers were active
#   placement_legality:
#     - generator draws exactly one source around one black hub
#     - all present rays share one arm length and stay inside bounds
#     - center search preserves conservative tip-to-border margins from the base task
#     - every candidate is validated by running main() and checking density + determinism
#   visual_observability:
#     - T samples expose the core fusion behavior directly: 3 visible tips but only 2 spiral
#       sectors emerge
#     - L samples repeat the irregular-source family while staying visually clean
#     - no hidden ordering, color encoding, or metadata is used; the only cues are shape and
#       which regions of the maze actually grow
#   cleanliness_and_inferability:
#     - one color is retained, so the source silhouette stays easy to read
#     - no extra markers are added near the hub, avoiding local clutter
#     - generator biases toward T examples so the "silent visible endpoint" pattern appears
#       often enough to be learnable from a few examples
#   new_main_assumptions_and_generator_guarantees:
#     assumption:
#       - exactly one teal T/L source exists, built from equal-length contiguous rays
#     guarantee:
#       - _build_input constructs that exact footprint and _find_source verifies it
#     assumption:
#       - if 3 rays are present, exactly one opposite pair exists
#     guarantee:
#       - T_SHAPES are sampled from the 4 standard missing-direction patterns
#     assumption:
#       - enough generated samples should differ from the D4/3-only behavior
#     guarantee:
#       - generator samples T cases with high probability, and post-run validation can
#         measure how often present_dirs != active_dirs

import math
import random

import numpy as np

from arc_utils import neighbors4

_DIRS_URDL = [(-1, 0), (0, 1), (1, 0), (0, -1)]
SOURCE_COLOR = 8


def _right(d):
    return _DIRS_URDL[(_DIRS_URDL.index(d) + 1) % 4]


def _left(d):
    return _DIRS_URDL[(_DIRS_URDL.index(d) + 3) % 4]


def _is_opposite(a, b) -> bool:
    return a[0] == -b[0] and a[1] == -b[1]


def _find_source(grid: np.ndarray) -> tuple[int, int, int, list[tuple[int, int]]]:
    """Return (hub_row, hub_col, arm_len, present_dirs) for one teal T/L source."""
    h, w = grid.shape
    teal_cells = {tuple(map(int, rc)) for rc in np.argwhere(grid == SOURCE_COLOR)}

    def inb(r: int, c: int) -> bool:
        return 0 <= r < h and 0 <= c < w

    for cr in range(h):
        for cc in range(w):
            if grid[cr, cc] != 0:
                continue

            present: list[tuple[tuple[int, int], int]] = []
            for d in _DIRS_URDL:
                r, c = cr, cc
                length = 0
                while inb(r + d[0], c + d[1]) and grid[r + d[0], c + d[1]] == SOURCE_COLOR:
                    r += d[0]
                    c += d[1]
                    length += 1
                if length > 0:
                    present.append((d, length))

            if len(present) not in (2, 3):
                continue

            arm_lengths = {length for _, length in present}
            if len(arm_lengths) != 1:
                continue

            dirs = [d for d, _ in present]
            if len(dirs) == 2 and _is_opposite(dirs[0], dirs[1]):
                continue

            arm_len = present[0][1]
            expected = {
                (cr + d[0] * k, cc + d[1] * k)
                for d in dirs
                for k in range(1, arm_len + 1)
            }
            if expected == teal_cells:
                return cr, cc, arm_len, dirs

    raise ValueError("main: expected one teal T/L source around a single black hub")


def _select_active_dirs(present_dirs: list[tuple[int, int]]) -> list[tuple[int, int]]:
    """
    Keep only the two launching tips.

    - T: choose the unique opposite pair.
    - L: both visible tips launch because there is no opposite pair to distinguish.
    """
    if len(present_dirs) == 2:
        return list(present_dirs)

    active = [
        d for d in present_dirs if any(_is_opposite(d, other) for other in present_dirs if other != d)
    ]
    if len(active) != 2:
        raise ValueError("main: expected a T source to contain exactly one opposite tip pair")
    return active


def _grow_spirals(
    grid: np.ndarray,
    cr: int,
    cc: int,
    arm_len: int,
    launch_dirs: list[tuple[int, int]],
) -> np.ndarray:
    """Run the original spiral engine from the chosen launch directions."""
    grid = np.asarray(grid, dtype=int)
    h, w = grid.shape
    out = grid.copy()

    def inb(r: int, c: int) -> bool:
        return 0 <= r < h and 0 <= c < w

    def is_border(r: int, c: int) -> bool:
        return r == 0 or r == h - 1 or c == 0 or c == w - 1

    def add(pos, d, k=1):
        return (pos[0] + d[0] * k, pos[1] + d[1] * k)

    def spacing_ok(cell, prev, start) -> bool:
        r, c = cell
        for nr, nc in neighbors4(r, c):
            if inb(nr, nc) and out[nr, nc] != 0 and (nr, nc) != prev and (nr, nc) != start:
                return False
        return True

    def can_move(cur, d, length: int) -> bool:
        prev = cur
        for t in range(1, length + 1):
            rr, cc2 = add(cur, d, t)
            if not inb(rr, cc2) or out[rr, cc2] != 0:
                return False
            if not spacing_ok((rr, cc2), prev, cur):
                return False
            prev = (rr, cc2)
        return True

    def do_move(agent, d, length: int) -> None:
        r0, c0 = agent["cur"]
        for t in range(1, length + 1):
            rr, cc2 = r0 + d[0] * t, c0 + d[1] * t
            out[rr, cc2] = SOURCE_COLOR
        agent["cur"] = (r0 + d[0] * length, c0 + d[1] * length)

    agents = []
    for outward in launch_dirs:
        agents.append(
            {
                "cur": (cr + outward[0] * arm_len, cc + outward[1] * arm_len),
                "dir": _right(outward),
                "out_dir": outward,
                "phase": "out_first" if arm_len == 1 else None,
                "done": False,
                "p": 0,
            }
        )

    def step_agent(agent) -> bool:
        if agent["done"]:
            return False

        if agent["phase"] == "out_first":
            if can_move(agent["cur"], agent["out_dir"], 1):
                do_move(agent, agent["out_dir"], 1)
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
            length = 1 if turned else (2 if agent["p"] == 0 else 1)
            if can_move(agent["cur"], nd, length):
                do_move(agent, nd, length)
                agent["p"] = 1 if (agent["p"] == 0 and turned) else 0
                agent["dir"] = nd
                return True
            if not turned and length == 2 and can_move(agent["cur"], nd, 1):
                do_move(agent, nd, 1)
                agent["p"] = 0
                return True

        agent["done"] = True
        return False

    changed = True
    guard = 0
    while changed and guard < 200000:
        guard += 1
        changed = False
        for ag in agents:
            if step_agent(ag):
                changed = True
        if all(ag["done"] for ag in agents):
            break

    even_center = (cr + cc) % 2 == 0
    if not even_center:
        rp = 1 - (cr % 2)
        cp = 1 - (cc % 2)
        rr, ccg = np.indices((h, w))
        out[(rr % 2 == rp) & (ccg % 2 == cp)] = 0
    elif arm_len % 2 == 1:
        to_clear = []
        for r in range(h):
            for c in range(w):
                if out[r, c] != SOURCE_COLOR or not is_border(r, c):
                    continue
                deg = 0
                nbr = None
                for nr, nc in neighbors4(r, c):
                    if inb(nr, nc) and out[nr, nc] == SOURCE_COLOR:
                        deg += 1
                        nbr = (nr, nc)
                if deg == 1 and nbr is not None and is_border(*nbr):
                    to_clear.append((r, c))
        for r, c in to_clear:
            out[r, c] = 0

        br = (h - 1, w - 1)
        if out[br] == 0:
            if all(
                not (inb(nr, nc) and out[nr, nc] == SOURCE_COLOR)
                for nr, nc in neighbors4(br[0], br[1])
            ):
                out[br] = SOURCE_COLOR
    else:
        if arm_len >= 2:
            out[0 : min(h, arm_len - 1), 0] = SOURCE_COLOR
        if arm_len >= 1:
            out[0, max(0, w - arm_len) : w] = SOURCE_COLOR
        if h >= 2:
            for c in range(w):
                if out[h - 1, c] == SOURCE_COLOR and out[h - 2, c] == 0:
                    out[h - 1, c] = 0

    return out


def main(grid: np.ndarray) -> np.ndarray:
    """
    Grow the spiral maze from only two selected tips of a teal T/L source.

    Input structure:
    - bg = 0 (black). One teal (8) source surrounds a single black hub cell.
    - The source is either a T with 3 equal arms or an L with 2 equal perpendicular arms.

    Rule:
    - Detect the hub, list the visible endpoint directions, then choose the 2 active launchers.
    - For a T, the active pair is the opposite pair on the crossbar; the third visible tip is silent.
    - For an L, the 2 visible tips are also the only launchers. Each active tip starts a clockwise
      spiral with priority turn-right > straight > turn-left, using the same 2/1 step rhythm and
      1-cell orthogonal spacing rule as the base task.

    Key insight:
    - The hard part is no longer just "which source arms exist"; it is separating visible endpoints
      from active endpoints. A T exposes 3 tips but only 2 actually seed the maze.
    """

    grid = np.asarray(grid, dtype=int)
    cr, cc, arm_len, present_dirs = _find_source(grid)
    active_dirs = _select_active_dirs(present_dirs)
    return _grow_spirals(grid, cr, cc, arm_len, active_dirs)


# --------------- generator (cx) ---------------

# --- CFG ---------------------------------------------------------------
ARM_LEN_MIN, ARM_LEN_MAX = 2, 4
GRID_SIZE_BASE = 6
GRID_SIZE_DELTAS = (-2, 0, 2)
TIP_GAP_MIN = 2
FILL_LO, FILL_HI = 0.30, 0.48
MAX_OUTER_RETRIES = 400
T_SAMPLE_WEIGHT = 0.8

T_SHAPES = [
    (_DIRS_URDL[1], _DIRS_URDL[2], _DIRS_URDL[3]),  # missing up
    (_DIRS_URDL[0], _DIRS_URDL[2], _DIRS_URDL[3]),  # missing right
    (_DIRS_URDL[0], _DIRS_URDL[1], _DIRS_URDL[3]),  # missing down
    (_DIRS_URDL[0], _DIRS_URDL[1], _DIRS_URDL[2]),  # missing left
]
L_SHAPES = [
    (_DIRS_URDL[0], _DIRS_URDL[1]),  # upper-right corner
    (_DIRS_URDL[1], _DIRS_URDL[2]),  # lower-right corner
    (_DIRS_URDL[2], _DIRS_URDL[3]),  # lower-left corner
    (_DIRS_URDL[3], _DIRS_URDL[0]),  # upper-left corner
]
# --- end CFG -----------------------------------------------------------


def _build_input(
    arm: int,
    size: int,
    cr: int,
    cc: int,
    dirs: tuple[tuple[int, int], ...],
) -> np.ndarray:
    """Build one teal T/L source around a black hub at (cr, cc)."""
    grid = np.zeros((size, size), dtype=int)
    offsets = np.arange(1, arm + 1)
    for dr, dc in dirs:
        grid[cr + dr * offsets, cc + dc * offsets] = SOURCE_COLOR
    return grid


def _choose_shape() -> tuple[tuple[int, int], ...]:
    if random.random() < T_SAMPLE_WEIGHT:
        return random.choice(T_SHAPES)
    return random.choice(L_SHAPES)


def _gen_attempt() -> np.ndarray | None:
    arm = random.randint(ARM_LEN_MIN, ARM_LEN_MAX)
    base_size = 4 * arm + 2 * math.ceil(arm / 2) + GRID_SIZE_BASE
    dirs = _choose_shape()

    size_options = [base_size + delta for delta in GRID_SIZE_DELTAS if base_size + delta >= 10]
    random.shuffle(size_options)

    for size in size_options:
        lo = 2 * arm + TIP_GAP_MIN
        hi = size - 2 * arm - TIP_GAP_MIN - 1
        if lo > hi:
            continue

        candidates = [(r, c) for r in range(lo, hi + 1) for c in range(lo, hi + 1)]
        random.shuffle(candidates)

        for cr, cc in candidates:
            inp = _build_input(arm, size, cr, cc, dirs)
            try:
                out = main(inp)
            except Exception:
                continue

            fill = float(np.mean(out == SOURCE_COLOR))
            active_dirs = _select_active_dirs(list(dirs))
            changed_tip_count = len(dirs) - len(active_dirs)
            if not (FILL_LO <= fill <= FILL_HI):
                continue
            if changed_tip_count == 0 and random.random() < 0.65:
                continue
            if np.array_equal(main(inp), out):
                return inp

    return None


def generate_input() -> np.ndarray:
    for _ in range(MAX_OUTER_RETRIES):
        g = _gen_attempt()
        if g is not None:
            return g
    raise RuntimeError("generate_input: failed after retries")
