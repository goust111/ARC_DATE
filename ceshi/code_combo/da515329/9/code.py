# Combo variant of da515329 (index 9)
# combo_spec: D4/1+D12/1
# Name: 非对称十字 + 完全非对称四臂

from __future__ import annotations

import random
import sys
from pathlib import Path

import numpy as np

_ROOT = Path(__file__).resolve().parents[4]
_SOLVERS = _ROOT / "solvers"
if str(_SOLVERS) not in sys.path:
    sys.path.append(str(_SOLVERS))

from arc_utils import neighbors4

_DIRS_URDL = [(-1, 0), (0, 1), (1, 0), (0, -1)]
_DIR_TO_IDX = {d: i for i, d in enumerate(_DIRS_URDL)}


# Design analysis for D4/1 + D12/1:
# a) Concrete behavior: replace the equal-arm plus with a fully asymmetric plus whose
#    four arm lengths are all different. Example lengths (U,R,D,L) = (1,4,2,3):
#    in the original task all four tips start immediately and share one global spiral
#    rhythm; here the top tip first slides 0 cells, the right tip 3 cells, the bottom
#    tip 1 cell, and the left tip 2 cells along its clockwise tangent before joining
#    the same wall-building process. After that launch offset, the parity of the arm
#    length chooses the entry phase of that arm's 2/1 straight-step rhythm, so two
#    tips do not begin the rectangular spiral at the same phase.
# b) Complexity calibration: the only free visual variable is arm length, and the
#    generator always uses a permutation of {1,2,3,4}. That gives four distinct,
#    repeated, human-visible categories without exploding the hypothesis space.
# c) Hidden-assumption audit: the original code assumed equal arm lengths when it
#    derived one shared arm_len and one shared post-process branch. This variant
#    removes that assumption entirely: each arm is collected separately, every agent
#    stores its own arm length, launch offset, and entry phase, and no later logic
#    relies on a single global arm length.
# d) Placement legality: generator placement keeps the cross away from borders by a
#    fixed inner margin, then validates the produced output. This prevents overlap,
#    clipping, and degenerate cases where a long tangential launch is immediately
#    blocked.
# e) Visual observability: the input already exposes the four lengths. In the output,
#    those lengths visibly control how far each tip shifts clockwise before the first
#    bend, and they also alter where that arm falls into the spiral rhythm.
# f) Visual cleanliness: only one color is used, and all four agents still obey the
#    original 4-neighbor spacing constraint. That keeps the result legible as four
#    interleaved wall paths rather than a dense overwrite.
# g) Fusion testability: generator acceptance requires the asymmetric variant output
#    to differ substantially from a uniform-parameter reference solver on the same
#    input, so samples cannot silently collapse back to the original behavior.


def _right(d: tuple[int, int]) -> tuple[int, int]:
    return _DIRS_URDL[(_DIR_TO_IDX[d] + 1) % 4]


def _left(d: tuple[int, int]) -> tuple[int, int]:
    return _DIRS_URDL[(_DIR_TO_IDX[d] + 3) % 4]


def _detect_center(grid: np.ndarray, color: int) -> tuple[int, int]:
    row_counts = np.sum(grid == color, axis=1)
    col_counts = np.sum(grid == color, axis=0)
    return int(np.argmax(row_counts)), int(np.argmax(col_counts))


def _collect_arms(
    grid: np.ndarray, cr: int, cc: int, color: int
) -> list[tuple[tuple[int, int], list[tuple[int, int]]]]:
    h, w = grid.shape

    def inb(r: int, c: int) -> bool:
        return 0 <= r < h and 0 <= c < w

    arms = []
    for d in _DIRS_URDL:
        dr, dc = d
        r, c = cr, cc
        cells = []
        while inb(r + dr, c + dc) and grid[r + dr, c + dc] == color:
            r += dr
            c += dc
            cells.append((r, c))
        arms.append((d, cells))
    return arms


def _trace_spirals(grid: np.ndarray, variant_mode: bool) -> np.ndarray:
    color = 8
    grid = np.asarray(grid, dtype=int)
    h, w = grid.shape
    out = grid.copy()
    cr, cc = _detect_center(grid, color)
    arms = _collect_arms(grid, cr, cc, color)

    if len(arms) != 4 or any(len(cells) == 0 for _, cells in arms):
        raise ValueError("expected one four-armed teal cross with a center hole")

    def inb(r: int, c: int) -> bool:
        return 0 <= r < h and 0 <= c < w

    def is_border(r: int, c: int) -> bool:
        return r == 0 or r == h - 1 or c == 0 or c == w - 1

    def add(pos: tuple[int, int], d: tuple[int, int], k: int = 1) -> tuple[int, int]:
        return pos[0] + d[0] * k, pos[1] + d[1] * k

    def spacing_ok(
        cell: tuple[int, int], prev: tuple[int, int], start: tuple[int, int]
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

    def do_move(agent: dict[str, object], d: tuple[int, int], length: int) -> None:
        r0, c0 = agent["cur"]
        for t in range(1, length + 1):
            rr, cc2 = r0 + d[0] * t, c0 + d[1] * t
            out[rr, cc2] = color
        agent["cur"] = (r0 + d[0] * length, c0 + d[1] * length)

    agents = []
    for outward, cells in arms:
        arm_len = len(cells)
        agents.append(
            {
                "cur": cells[-1],
                "dir": _right(outward),
                "done": False,
                "p": (arm_len + 1) % 2 if variant_mode else 0,
                "launch": arm_len - 1 if variant_mode else 0,
            }
        )

    def step_agent(agent: dict[str, object]) -> bool:
        if agent["done"]:
            return False

        if agent["launch"] > 0:
            d = agent["dir"]
            if can_move(agent["cur"], d, 1):
                do_move(agent, d, 1)
                agent["launch"] -= 1
                return True
            agent["launch"] = 0

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
                agent["dir"] = nd
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

    return out


def _uniform_reference_output(grid: np.ndarray) -> np.ndarray:
    return _trace_spirals(grid, variant_mode=False)


def _variant_output(grid: np.ndarray) -> np.ndarray:
    return _trace_spirals(grid, variant_mode=True)


def _build_input(
    arm_lengths: tuple[int, int, int, int], size: int, cr: int, cc: int
) -> np.ndarray:
    grid = np.zeros((size, size), dtype=int)
    for (dr, dc), arm_len in zip(_DIRS_URDL, arm_lengths):
        offsets = np.arange(1, arm_len + 1)
        grid[cr + dr * offsets, cc + dc * offsets] = 8
    return grid


def _validate_cross_lengths(grid: np.ndarray) -> tuple[int, int, int, int]:
    cr, cc = _detect_center(grid, 8)
    arms = _collect_arms(grid, cr, cc, 8)
    lens = tuple(len(cells) for _, cells in arms)
    if sorted(lens) != [1, 2, 3, 4]:
        raise ValueError("expected four distinct arm lengths 1..4")
    return lens


def _variant_delta(inp: np.ndarray, out: np.ndarray) -> float:
    ref = _uniform_reference_output(inp)
    return float(np.mean(out != ref))


def main(grid: np.ndarray) -> np.ndarray:
    """
    Grow four clockwise spiral wall paths from the tips of a fully asymmetric teal cross.

    Input structure:
    - bg = 0 (black). A single teal (8) plus sign with a center hole.
    - The four arms point in the cardinal directions but their lengths are all different.

    Rule:
    - Each arm tip launches one spiral agent, still rotating clockwise around the scene.
    - Before joining the shared spiral rule, each agent first slides along its clockwise
      tangent by (arm_length - 1) cells, so longer arms start farther from the tip.
    - After launch, movement priority remains turn right > go straight > turn left
      (left only away from borders), with the original 1-cell spacing constraint.
    - Arm length also chooses the agent's entry phase in the 2/1 straight-step rhythm,
      so different tips do not enter the spiral at the same moment.

    Key insight:
    The input no longer asks for one symmetric global spiral. Instead, each visible arm
    length is a local code that sets both the tip's clockwise offset and that arm's first
    straight-step phase, producing four related but non-identical spiral starts.
    """

    grid = np.asarray(grid, dtype=int)
    _validate_cross_lengths(grid)
    return _variant_output(grid)


# --------------- generator (cx) ---------------

# --- CFG ---------------------------------------------------------------
ARM_LENGTH_SET = (1, 2, 3, 4)
GRID_SIZES = (22, 24, 26)
CENTER_MARGIN = 6
FILL_LO, FILL_HI = 0.45, 0.58
DELTA_LO = 0.06
MAX_OUTER_RETRIES = 200
CROSS_COLOR = 8
# --- end CFG -----------------------------------------------------------


def _candidate_centers(size: int) -> list[tuple[int, int]]:
    return [
        (r, c)
        for r in range(CENTER_MARGIN, size - CENTER_MARGIN)
        for c in range(CENTER_MARGIN, size - CENTER_MARGIN)
    ]


def _gen_attempt() -> np.ndarray | None:
    arm_lengths = tuple(random.sample(ARM_LENGTH_SET, 4))
    size = random.choice(GRID_SIZES)
    candidates = _candidate_centers(size)
    random.shuffle(candidates)

    for cr, cc in candidates:
        inp = _build_input(arm_lengths, size, cr, cc)
        try:
            out = main(inp)
        except Exception:
            continue

        fill = float(np.mean(out == CROSS_COLOR))
        delta = _variant_delta(inp, out)
        if FILL_LO <= fill <= FILL_HI and delta >= DELTA_LO:
            if np.array_equal(main(inp), out):
                return inp
    return None


def generate_input() -> np.ndarray:
    for _ in range(MAX_OUTER_RETRIES):
        grid = _gen_attempt()
        if grid is not None:
            return grid
    raise RuntimeError("generate_input: failed after retries")
