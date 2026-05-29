# Combo variant of da515329 (index 31)
# combo_spec: D4/3+D5/0+D9/1
# Name: T/L形源 + 双源发射 + 臂重着色

# Design analysis
# - concrete_behavior_trace:
#   - Base task: a 4-arm teal cross around one black hub launches 4 clockwise spiral agents,
#     so every cardinal tip both exists and emits.
#   - Fused task: the source geometry is first read as an irregular T/L object.
#     If it is a T, only the unique opposite pair is active; the side branch stays in place
#     and is recolored in the output. If it is an L, the two visible tips are the only pair
#     available, so they emit and there is no passive branch to recolor.
#   - Small local example: a T with directions [up, right, down] used to launch 3 tips in the
#     D4-only variant. Here only [up, down] launch; the right arm remains as a red clue.
#     The output therefore differs in two coupled ways: one spiral sector is missing, and the
#     leftover arm survives as a second color rather than blending into teal walls.
# - complexity_calibration:
#   - The detector supports both T and L so the solver matches the stated source family.
#   - The generator intentionally emits only T samples. Reason: D5/0 requires an opposite pair,
#     while D9/1 needs a visible passive arm; an L cannot show both cues in the same example.
#   - Freedom is kept small: 4 T orientations, arm lengths 2-4, and a short list of grid sizes.
# - implicit_assumption_audit:
#   - The base argmax-based hub finder assumed a symmetric 4-arm cross; this breaks on T/L, so
#     the new solver scans black cells and validates the exact teal ray set.
#   - The base agent builder assumed every present arm emits; this now fails on T because one
#     present arm is passive. Active directions are therefore derived from source geometry first.
#   - The base output was monochrome. Recoloring is delayed until after spiral growth and border
#     cleanup so passive-arm cells still behave as occupied walls during simulation.
# - placement_legality:
#   - The generator builds exactly one equal-arm T around a black hub, keeps all tips away from
#     borders using the original conservative margin, and validates each candidate through main().
# - visual_observability_and_cleanliness:
#   - Teal still marks the spiral system, while red marks only the passive branch; there are only
#     two visible foreground colors and they occupy disjoint semantic roles.
#   - The generator rejects degenerate samples where the combo output differs from the
#     all-present T counterfactual only by recoloring. This forces the double-source change to
#     affect maze geometry, not just source-arm color.
# - new_main_assumptions_and_generator_guarantees:
#   - Assumption: the input contains exactly one teal T/L source with equal contiguous arms around
#     one black hub. Guarantee: _build_input constructs exactly that and no extra teal cells.
#   - Assumption: passive-arm recolor is meaningful. Guarantee: generate_input uses only T shapes,
#     so each sample contains one non-active arm whose red length equals the arm length.

import math
import random

import numpy as np

from arc_utils import neighbors4

_DIRS_URDL = [(-1, 0), (0, 1), (1, 0), (0, -1)]
SOURCE_COLOR = 8
PASSIVE_COLOR = 2


def _right(d: tuple[int, int]) -> tuple[int, int]:
    return _DIRS_URDL[(_DIRS_URDL.index(d) + 1) % 4]


def _left(d: tuple[int, int]) -> tuple[int, int]:
    return _DIRS_URDL[(_DIRS_URDL.index(d) + 3) % 4]


def _is_opposite(a: tuple[int, int], b: tuple[int, int]) -> bool:
    return a[0] == -b[0] and a[1] == -b[1]


def _find_source(
    grid: np.ndarray,
) -> tuple[int, int, int, list[tuple[int, int]], str]:
    """Return (hub_row, hub_col, arm_len, present_dirs, shape_kind)."""
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
            shape_kind = "T" if len(dirs) == 3 else "L"
            if shape_kind == "L" and _is_opposite(dirs[0], dirs[1]):
                continue

            arm_len = present[0][1]
            expected = {
                (cr + d[0] * k, cc + d[1] * k)
                for d in dirs
                for k in range(1, arm_len + 1)
            }
            if expected == teal_cells:
                return cr, cc, arm_len, dirs, shape_kind

    raise ValueError("main: expected one teal T/L source around a single black hub")


def _classify_tips(
    present_dirs: list[tuple[int, int]],
) -> tuple[list[tuple[int, int]], list[tuple[int, int]]]:
    """Split source directions into active emitters and passive branches."""
    if len(present_dirs) == 2:
        return list(present_dirs), []

    active = []
    passive = []
    for d in present_dirs:
        if any(_is_opposite(d, other) for other in present_dirs if other != d):
            active.append(d)
        else:
            passive.append(d)

    if len(active) != 2 or len(passive) != 1:
        raise ValueError("main: T source must contain exactly one opposite pair and one side arm")
    return active, passive


def _arm_cells(
    cr: int,
    cc: int,
    arm_len: int,
    dirs: list[tuple[int, int]],
) -> list[tuple[int, int]]:
    return [
        (cr + d[0] * k, cc + d[1] * k)
        for d in dirs
        for k in range(1, arm_len + 1)
    ]


def _solve_with_launch_dirs(
    grid: np.ndarray,
    cr: int,
    cc: int,
    arm_len: int,
    launch_dirs: list[tuple[int, int]],
) -> np.ndarray:
    """Run the inherited spiral engine from the chosen launch directions."""
    grid = np.asarray(grid, dtype=int)
    h, w = grid.shape
    out = grid.copy()

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
            if inb(nr, nc) and out[nr, nc] != 0 and (nr, nc) != prev and (nr, nc) != start:
                return False
        return True

    def can_move(cur: tuple[int, int], d: tuple[int, int], length: int) -> bool:
        prev = cur
        for t in range(1, length + 1):
            rr, cc2 = add(cur, d, t)
            if not inb(rr, cc2) or out[rr, cc2] != 0:
                return False
            if not spacing_ok((rr, cc2), prev, cur):
                return False
            prev = (rr, cc2)
        return True

    def do_move(agent: dict, d: tuple[int, int], length: int) -> None:
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

    def step_agent(agent: dict) -> bool:
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
        for agent in agents:
            if step_agent(agent):
                changed = True
        if all(agent["done"] for agent in agents):
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
        if out[br] == 0 and all(
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
    Grow clockwise spirals from the active pair of an irregular teal source and preserve the spare arm in red.

    Input structure:
    - bg = 0 (black). A single teal (8) source surrounds one black hub cell.
    - The source is either a T (3 equal arms) or an L (2 perpendicular equal arms).

    Rule:
    - Detect which cardinal arms are present. In a T, only the opposite pair is active and the side arm is passive;
      in an L, the two visible tips are the only active pair.
    - Each active tip launches the inherited clockwise spiral turtle: start 90° clockwise from the arm, prefer
      right > straight > left, use 1-step after turns, and keep a 1-cell orthogonal gap from existing walls.
    - After spiral growth and the inherited parity/border cleanup, recolor the passive arm cells red (2) while
      keeping the active arms and spiral walls teal (8).

    Key insight:
    - The solver must read source geometry before simulating: the same teal object encodes both the launch pair and,
      when present, a non-emitting branch whose only job is to remain visible as a differently colored clue.
    """

    grid = np.asarray(grid, dtype=int)
    cr, cc, arm_len, present_dirs, _shape_kind = _find_source(grid)
    active_dirs, passive_dirs = _classify_tips(present_dirs)
    out = _solve_with_launch_dirs(grid, cr, cc, arm_len, active_dirs)

    for r, c in _arm_cells(cr, cc, arm_len, passive_dirs):
        out[r, c] = PASSIVE_COLOR
    return out


# --------------- generator (cx) ---------------

# --- CFG ---------------------------------------------------------------
ARM_LEN_MIN, ARM_LEN_MAX = 2, 4
GRID_SIZE_BASE = 6
GRID_SIZE_DELTAS = (-4, -2, 0)
TIP_GAP_MIN = 2
FILL_LO, FILL_HI = 0.34, 0.50
COUNTERFACTUAL_EXTRA_DIFF_MIN = 2
MAX_OUTER_RETRIES = 400

T_SHAPES = [
    (_DIRS_URDL[1], _DIRS_URDL[2], _DIRS_URDL[3]),  # missing up
    (_DIRS_URDL[0], _DIRS_URDL[2], _DIRS_URDL[3]),  # missing right
    (_DIRS_URDL[0], _DIRS_URDL[1], _DIRS_URDL[3]),  # missing down
    (_DIRS_URDL[0], _DIRS_URDL[1], _DIRS_URDL[2]),  # missing left
]
# --- end CFG -----------------------------------------------------------


def _build_input(
    arm: int,
    size: int,
    cr: int,
    cc: int,
    dirs: tuple[tuple[int, int], ...],
) -> np.ndarray:
    """Build a square grid with one teal T source around a black hub."""
    grid = np.zeros((size, size), dtype=int)
    offsets = np.arange(1, arm + 1)
    for dr, dc in dirs:
        grid[cr + dr * offsets, cc + dc * offsets] = SOURCE_COLOR
    return grid


def _gen_attempt() -> np.ndarray | None:
    arm = random.randint(ARM_LEN_MIN, ARM_LEN_MAX)
    base_size = 4 * arm + 2 * math.ceil(arm / 2) + GRID_SIZE_BASE
    dirs = random.choice(T_SHAPES)
    active_dirs, passive_dirs = _classify_tips(list(dirs))

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

            full_out = _solve_with_launch_dirs(inp, cr, cc, arm, list(dirs))
            fill = float(np.mean(out != 0))
            red_count = int(np.sum(out == PASSIVE_COLOR))
            diff_vs_full = int(np.count_nonzero(out != full_out))
            geom_extra = diff_vs_full - red_count

            if not (FILL_LO <= fill <= FILL_HI):
                continue
            if red_count != arm * len(passive_dirs):
                continue
            if geom_extra < COUNTERFACTUAL_EXTRA_DIFF_MIN:
                continue
            if not np.array_equal(main(inp), out):
                continue
            if _find_source(inp)[:4] != (cr, cc, arm, list(dirs)):
                continue
            if _classify_tips(_find_source(inp)[3])[0] != active_dirs:
                continue
            return inp

    return None


def generate_input() -> np.ndarray:
    for _ in range(MAX_OUTER_RETRIES):
        grid = _gen_attempt()
        if grid is not None:
            return grid
    raise RuntimeError("generate_input: failed after retries")
