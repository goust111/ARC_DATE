# Combo variant of da515329 (index 3)
# combo_spec: D1/0+D14/0
# Name: 逆时针螺旋 + 前景背景反转

# Design analysis (YAML-style notes kept with the code as requested)
# design_analysis:
#   source_task: da515329
#   fusion_summary:
#     - D14/0 flips visible polarity: the canvas starts teal and the structure is carved in black.
#     - D1/0 flips spiral chirality: the four tip-launched paths must now be the mirror-image
#       counterclockwise version of the original clockwise labyrinth.
#     - The clean way to fuse them is not to rewrite the geometry in visible space. Instead:
#       visible teal/black -> internal black/teal occupancy -> horizontal mirror -> original
#       clockwise solver -> unmirror -> visible teal/black again.
#   concretization:
#     input:
#       - square even grid, side length 12..26 from `4*arm + 2*ceil(arm/2) + 6`
#       - visible background is teal 8
#       - one black 0 plus-sign cross with equal arm length 1..4 and a teal center hole
#       - the cross is off-center but every tip still keeps gap >= arm+2 from the border
#     output:
#       - black counterclockwise spiral trenches spread from the four black arm tips
#       - untouched cells remain teal, so the result looks like a carved maze on a solid field
#       - fill stays near 50%, preserving the dense kaleidoscopic rectangular-ring look
#   trace_example:
#     - Example with arm=2 on a teal field:
#       original visible D14-only logic would carve the cell immediately to the right of the top tip,
#       because the internal solver still starts that tip facing clockwise.
#       fused logic first mirrors the occupancy grid, so that same top-tip situation becomes the
#       mirrored one seen by the original solver; after solving and unmirroring, the carved neighbor
#       lands on the left instead.
#     - Difference:
#       original base task = paint teal clockwise on black
#       D14-only variant = carve black clockwise on teal
#       fused variant = carve black counterclockwise on teal
#   complexity_calibration:
#     - The task still has only one object family, two colors, and the same arm-length range 1..4.
#     - The direction change is visually strong because clockwise vs counterclockwise spirals disagree
#       along many border caps and inner elbows, not just at one local turn.
#     - No per-instance hidden mode is introduced; every sample uses the same fused rule.
#   old_assumptions_to_flip:
#     - `(grid == 8)` no longer detects the visible cross; in visible space teal is background.
#     - `out[cell] != 0` used to mean "already occupied by wall"; in visible space that would
#       incorrectly treat all teal background as blocked, so occupancy checks must run after polarity conversion.
#     - The original post-processing is handedness-sensitive. Reusing it directly with naively swapped
#       turn priorities risks wrong border caps; mirroring the full draw-space problem preserves those branches exactly.
#   placement_legality:
#     - Generator keeps the original size formula and center-range constraints, so tips remain in-bounds
#       and the mirrored internal solve has exactly the same amount of room as the base task.
#     - Input construction is a full teal canvas plus four black arms, so there is no overlap ambiguity.
#     - Accepted samples are filtered by solver success, determinism, fill range, and a direction-gap metric.
#   visual_observability:
#     - Across 2-4 samples a human can see two stable cues: the grid begins mostly teal, and the carved
#       maze curls the opposite way from the original task.
#     - The rule is inferable from visible geometry alone; it does not rely on numeric color ids or tie-breaking by ids.
#   visual_cleanliness:
#     - Only colors 0 and 8 are present, and the black cross remains readable because its center stays teal.
#     - The one-cell spacing rule survives unchanged, so the carved output remains crisp rather than muddy.
#   color_inference:
#     - There is one global polarity rule: teal is erasable field, black is carved structure.
#     - No conditional recoloring by arm, rank, or parity is added.
#   anti_degeneracy_metric:
#     - For each candidate input, compute the visible clockwise-carving output on the same input and measure
#       `direction_gap = mean(counterclockwise_output != clockwise_output)`.
#     - Reject samples with too-small gap so D1/0 never collapses into an unobservable latent variable.
#   new_main_assumptions:
#     - Valid inputs use only colors 0 and 8.
#     - The four black arms are axis-aligned, equal-length, and separated by a single teal center hole.
#     - Generator guarantees those assumptions by direct construction and by rejecting samples that fail
#       the fill-range and direction-gap checks.

import math
import random
import sys

import numpy as np

_SOLVER_UTILS = "/home/lixinyu/snap/new/solvers"
if _SOLVER_UTILS not in sys.path:
    sys.path.append(_SOLVER_UTILS)

from arc_utils import neighbors4

_DIRS_URDL = [(-1, 0), (0, 1), (1, 0), (0, -1)]

VISIBLE_BG = 8
VISIBLE_PATH = 0
DRAW_BG = 0
DRAW_PATH = 8


def _mirror_lr(grid: np.ndarray) -> np.ndarray:
    """Return a left-right mirror copy."""
    return np.fliplr(np.asarray(grid, dtype=int)).copy()


def _to_draw_space(grid: np.ndarray) -> np.ndarray:
    """Map the visible carve-on-teal task into the original draw-on-black occupancy space."""
    return np.where(grid == VISIBLE_BG, DRAW_BG, DRAW_PATH).astype(int, copy=False)


def _from_draw_space(grid: np.ndarray) -> np.ndarray:
    """Map the internal occupancy result back to the visible carve-on-teal task."""
    return np.where(grid == DRAW_PATH, VISIBLE_PATH, VISIBLE_BG).astype(int, copy=False)


def _solve_clockwise_draw_space(grid: np.ndarray) -> np.ndarray:
    """Original clockwise spiral construction in black-background / teal-path occupancy space."""
    grid = np.asarray(grid, dtype=int)
    h, w = grid.shape
    out = grid.copy()

    row_counts = np.sum(grid == DRAW_PATH, axis=1)
    col_counts = np.sum(grid == DRAW_PATH, axis=0)
    cr = int(np.argmax(row_counts))
    cc = int(np.argmax(col_counts))

    def inb(r: int, c: int) -> bool:
        return 0 <= r < h and 0 <= c < w

    def is_border(r: int, c: int) -> bool:
        return r == 0 or r == h - 1 or c == 0 or c == w - 1

    def add(pos, d, k=1):
        return (pos[0] + d[0] * k, pos[1] + d[1] * k)

    def right(d):
        return _DIRS_URDL[(_DIRS_URDL.index(d) + 1) % 4]

    def left(d):
        return _DIRS_URDL[(_DIRS_URDL.index(d) + 3) % 4]

    def spacing_ok(cell, prev, start) -> bool:
        r, c = cell
        for nr, nc in neighbors4(r, c):
            if inb(nr, nc) and out[nr, nc] != DRAW_BG and (nr, nc) != prev and (nr, nc) != start:
                return False
        return True

    def can_move(cur, d, length: int) -> bool:
        prev = cur
        for t in range(1, length + 1):
            rr, cc2 = add(cur, d, t)
            if not inb(rr, cc2) or out[rr, cc2] != DRAW_BG:
                return False
            if not spacing_ok((rr, cc2), prev, cur):
                return False
            prev = (rr, cc2)
        return True

    def do_move(agent, d, length: int) -> None:
        r0, c0 = agent["cur"]
        for t in range(1, length + 1):
            rr, cc2 = r0 + d[0] * t, c0 + d[1] * t
            out[rr, cc2] = DRAW_PATH
        agent["cur"] = (r0 + d[0] * length, c0 + d[1] * length)

    arms = []
    for d in _DIRS_URDL:
        dr, dc = d
        r, c = cr, cc
        cells = []
        while inb(r + dr, c + dc) and grid[r + dr, c + dc] == DRAW_PATH:
            r += dr
            c += dc
            cells.append((r, c))
        if cells:
            arms.append((d, cells))

    even_center = (cr + cc) % 2 == 0
    arm_len = max((len(cells) for _, cells in arms), default=0)

    agents = []
    for outward, cells in arms:
        agents.append(
            {
                "cur": cells[-1],
                "dir": right(outward),
                "out_dir": outward,
                "phase": "out_first" if len(cells) == 1 else None,
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
            agent["dir"] = right(agent["out_dir"])
            agent["p"] = 0
            return True

        cur = agent["cur"]
        dirs = [right(agent["dir"]), agent["dir"]]
        if not is_border(*cur):
            dirs.append(left(agent["dir"]))

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

    if not even_center:
        rp = 1 - (cr % 2)
        cp = 1 - (cc % 2)
        rr, ccg = np.indices((h, w))
        out[(rr % 2 == rp) & (ccg % 2 == cp)] = DRAW_BG
    elif arm_len % 2 == 1:
        to_clear = []
        for r in range(h):
            for c in range(w):
                if out[r, c] != DRAW_PATH:
                    continue
                if not is_border(r, c):
                    continue
                deg = 0
                nbr = None
                for nr, nc in neighbors4(r, c):
                    if inb(nr, nc) and out[nr, nc] == DRAW_PATH:
                        deg += 1
                        nbr = (nr, nc)
                if deg == 1 and nbr is not None and is_border(*nbr):
                    to_clear.append((r, c))
        for r, c in to_clear:
            out[r, c] = DRAW_BG

        br = (h - 1, w - 1)
        if out[br] == DRAW_BG:
            if all(
                not (inb(nr, nc) and out[nr, nc] == DRAW_PATH)
                for nr, nc in neighbors4(br[0], br[1])
            ):
                out[br] = DRAW_PATH
    else:
        if arm_len >= 2:
            out[0 : min(h, arm_len - 1), 0] = DRAW_PATH
        if arm_len >= 1:
            out[0, max(0, w - arm_len) : w] = DRAW_PATH
        if h >= 2:
            for c in range(w):
                if out[h - 1, c] == DRAW_PATH and out[h - 2, c] == DRAW_BG:
                    out[h - 1, c] = DRAW_BG

    return out


def _solve_draw_space(grid: np.ndarray, *, counterclockwise: bool) -> np.ndarray:
    """
    Solve in draw space.

    Counterclockwise mode is obtained by mirroring the whole occupancy problem, using the
    original clockwise solver, then mirroring the finished path back. A mirror flips handedness
    while preserving all spacing and border-cleanup behavior.
    """
    draw = np.asarray(grid, dtype=int)
    if not counterclockwise:
        return _solve_clockwise_draw_space(draw)
    mirrored = _mirror_lr(draw)
    solved = _solve_clockwise_draw_space(mirrored)
    return _mirror_lr(solved)


def _solve_visible(grid: np.ndarray, *, counterclockwise: bool) -> np.ndarray:
    """Apply the draw-space solver and map the result back into the visible teal-background task."""
    grid = np.asarray(grid, dtype=int)
    draw_space = _to_draw_space(grid)
    solved = _solve_draw_space(draw_space, counterclockwise=counterclockwise)
    return _from_draw_space(solved)


def _direction_gap(grid: np.ndarray, ccw: np.ndarray | None = None) -> float:
    """Fraction of cells where visible clockwise and counterclockwise carve outputs disagree."""
    if ccw is None:
        ccw = _solve_visible(grid, counterclockwise=True)
    cw = _solve_visible(grid, counterclockwise=False)
    return float(np.mean(ccw != cw))


def main(grid: np.ndarray) -> np.ndarray:
    """
    Carve four counterclockwise spiral trenches from the tips of a black cross.

    Input structure:
    - bg = 8 (teal). A single black (0) cross with a teal center hole marks the four launch tips.

    Rule:
    - Read the black arms as the already-carved seed path and extend one spiral trench from each tip.
    - The visible carving is counterclockwise: relative to the original task, the rectangular spiral layout is mirrored.
    - Step lengths, one-cell spacing, and the parity/border cleanup rules stay the same; only the handedness and visible polarity change.

    Key insight: This fusion is easiest to solve in two hidden stages. First complement the teal/black image into the original occupancy space. Then mirror that occupancy grid, run the original clockwise solver, and unmirror the result; the mirror converts clockwise growth into the desired counterclockwise carving without re-deriving the cleanup cases.
    """

    return _solve_visible(grid, counterclockwise=True)


# --------------- generator (cx) ---------------

# --- CFG ---------------------------------------------------------------
ARM_LEN_MIN, ARM_LEN_MAX = 1, 4
GRID_SIZE_BASE = 6
TIP_GAP_MIN = 2
FILL_LO, FILL_HI = 0.48, 0.52
MIN_DIRECTION_GAP = 0.08
MAX_OUTER_RETRIES = 400
CROSS_COLOR = VISIBLE_PATH
BACKGROUND_COLOR = VISIBLE_BG
# --- end CFG -----------------------------------------------------------


def _build_input(arm: int, size: int, cr: int, cc: int) -> np.ndarray:
    """Build a square teal grid with a black plus-sign cross (center hole left teal)."""
    grid = np.full((size, size), BACKGROUND_COLOR, dtype=int)
    offsets = np.arange(1, arm + 1)
    grid[cr - offsets, cc] = CROSS_COLOR
    grid[cr + offsets, cc] = CROSS_COLOR
    grid[cr, cc - offsets] = CROSS_COLOR
    grid[cr, cc + offsets] = CROSS_COLOR
    return grid


def _gen_attempt() -> np.ndarray | None:
    """Try one arm-length, enumerate valid centers, and return the first non-degenerate sample."""
    arm = random.randint(ARM_LEN_MIN, ARM_LEN_MAX)
    size = 4 * arm + 2 * math.ceil(arm / 2) + GRID_SIZE_BASE

    lo = 2 * arm + TIP_GAP_MIN
    hi = size - 2 * arm - TIP_GAP_MIN - 1
    if lo > hi:
        return None

    candidates = [(r, c) for r in range(lo, hi + 1) for c in range(lo, hi + 1)]
    random.shuffle(candidates)

    for cr, cc in candidates:
        inp = _build_input(arm, size, cr, cc)
        try:
            out = main(inp)
        except Exception:
            continue

        fill = float(np.mean(out == CROSS_COLOR))
        if not (FILL_LO <= fill <= FILL_HI):
            continue

        if not np.array_equal(main(inp), out):
            continue

        if _direction_gap(inp, ccw=out) < MIN_DIRECTION_GAP:
            continue

        return inp
    return None


def generate_input() -> np.ndarray:
    for _ in range(MAX_OUTER_RETRIES):
        g = _gen_attempt()
        if g is not None:
            return g
    raise RuntimeError("generate_input: failed after retries")
