# Variant of da515329
# Dimension: D4 — 源对象形状 (src)
# Change: 青色十字形（中心点 + 四个方向各延伸等长臂），四个端点作为螺旋发射源... → T形或L形源（部分端点缺失）

# Design analysis
# spec_variant:
#   source_shape:
#     base_task: one implicit black hub cell with 4 equal-length teal rays
#     variant: one implicit black hub cell with either:
#       - T source: 3 equal-length teal rays
#       - L source: 2 equal-length teal rays, required to be perpendicular
#     unchanged:
#       - color stays teal (8)
#       - every ray still starts one cell away from the black hub
#       - all present rays share the same arm length
#   concrete_pixel_behavior:
#     input:
#       - all teal cells still lie on one row and/or one column around a single black hub
#       - missing directions are visible directly as missing rays, not as hidden metadata
#     output:
#       - only existing ray tips launch spiral agents
#       - the spiral motion, spacing rule, and post-processing remain the same as the base task
#       - because some launchers are absent, the maze becomes visibly asymmetric
#   traced_example:
#     original_local_case:
#       - a full 4-arm cross with directions [U, R, D, L] launches 4 spiral agents
#       - every quadrant gets filled by one of those interacting clockwise walls
#     variant_local_case:
#       - an L source with directions [U, R] launches only the top and right tips
#       - the lower and left launchers do not exist, so no walls are seeded from those sides
#       - the same movement engine now produces a one-sided rectangular labyrinth
#     key_difference:
#       - the behavioral change happens before spiral growth: source recognition decides
#         which cardinal tips exist, and that subset controls the entire downstream fill
#   complexity_calibration:
#     keep_small_choice_set:
#       - only two source families: T or L
#       - T has 4 orientations; L has 4 corner orientations
#     observability:
#       - generator avoids arm length 1 so the input still reads clearly as T/L instead of
#         a few isolated pixels around a black hub
#       - both 2-tip and 3-tip cases are generated so the missing-launcher rule is repeated
#         across samples rather than hidden behind one fixed branch
#   main_assumption_audit:
#     old_assumption_1:
#       code: row/col argmax identifies the hub of a 4-arm cross
#       status: invalid for T/L, especially when arm length is short
#       fix: scan black cells and validate equal contiguous rays exactly matching all teal cells
#     old_assumption_2:
#       code: every cardinal direction may contribute an arm
#       status: invalid in the variant
#       fix: build agents only for the detected present directions
#     old_assumption_3:
#       code: arm length can be taken as max ray length from the hub
#       status: still valid only after verifying all present rays are equal
#       fix: source detector enforces equal ray lengths before returning the hub
#     old_assumption_4:
#       code: parity/border cleanup depends on hub parity and arm length, not on symmetry
#       status: still acceptable
#       reason: these corrections patch the global lattice geometry after growth and do not
#         require the absent launchers to be present
#   placement_legality:
#     generator:
#       - keeps the hub far enough from every border using the original conservative margin
#       - draws exactly one source object with no overlaps or extra teal cells
#       - validates every candidate by running main() and checking fill + determinism
#   visual_cleanliness:
#     - single color is retained, so the only visible cue is source geometry and the resulting
#       asymmetric coverage
#     - no extra markers are added; the source remains a compact, legible orthogonal object
#   new_main_assumptions_and_generator_guarantees:
#     assumption:
#       - there is exactly one teal source whose rays are equal-length and meet at one missing hub
#     guarantee:
#       - _build_input constructs exactly that pattern
#     assumption:
#       - L sources use perpendicular rays rather than opposite rays
#     guarantee:
#       - generator samples only corner pairs
#     assumption:
#       - fill should stay visually rich enough for humans to infer the rule
#     guarantee:
#       - generator searches two nearby grid sizes and accepts only outputs in a medium-density band

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
    """Return (hub_row, hub_col, arm_len, present_dirs) for a single teal T/L source."""
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


def main(grid: np.ndarray) -> np.ndarray:
    """
    Launch clockwise spiral walls only from the tips that exist in a teal T/L source.

    Input structure:
    - bg = 0 (black). A single teal (8) source surrounds one black hub cell.
    - The source is either a T (3 equal arms) or an L (2 perpendicular equal arms).

    Rule:
    - Detect the black hub and keep only the cardinal tips that are actually present.
    - Each present tip launches a turtle heading 90° clockwise from its outward ray.
    - Movement priority is turn right > go straight > turn left; a turned move has length 1,
      and a straight move alternates between 2 and 1 while preserving a 1-cell orthogonal gap.

    Key insight:
    - The spiral engine is unchanged from the base task; the variant only changes which launch
      directions exist, so the missing source arms directly become missing spiral sectors.
    """

    grid = np.asarray(grid, dtype=int)
    h, w = grid.shape
    out = grid.copy()
    cr, cc, arm_len, present_dirs = _find_source(grid)

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
    for outward in present_dirs:
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


# --------------- generator (cx) ---------------

# --- CFG ---------------------------------------------------------------
ARM_LEN_MIN, ARM_LEN_MAX = 2, 4        # avoid arm=1 so the T/L source remains visually obvious
GRID_SIZE_BASE = 6                      # inherited from the base task
GRID_SIZE_DELTAS = (-2, 0)             # smaller grids compensate for having only 2 or 3 launchers
TIP_GAP_MIN = 2                        # conservative hub margin preserved from the base task
FILL_LO, FILL_HI = 0.40, 0.52          # keep the output dense enough to read as a maze
MAX_OUTER_RETRIES = 300

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


def _build_input(arm: int, size: int, cr: int, cc: int, dirs: tuple[tuple[int, int], ...]) -> np.ndarray:
    """Build a square grid with a single teal T/L source around a black hub at (cr, cc)."""
    grid = np.zeros((size, size), dtype=int)
    for dr, dc in dirs:
        offsets = np.arange(1, arm + 1)
        grid[cr + dr * offsets, cc + dc * offsets] = SOURCE_COLOR
    return grid


def _gen_attempt() -> np.ndarray | None:
    arm = random.randint(ARM_LEN_MIN, ARM_LEN_MAX)
    base_size = 4 * arm + 2 * math.ceil(arm / 2) + GRID_SIZE_BASE

    shape_pool = T_SHAPES if random.random() < 0.5 else L_SHAPES
    dirs = random.choice(shape_pool)

    size_options = [base_size + delta for delta in GRID_SIZE_DELTAS if base_size + delta >= 8]
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
            if FILL_LO <= fill <= FILL_HI and np.array_equal(main(inp), out):
                return inp

    return None


def generate_input() -> np.ndarray:
    for _ in range(MAX_OUTER_RETRIES):
        g = _gen_attempt()
        if g is not None:
            return g
    raise RuntimeError("generate_input: failed after retries")
