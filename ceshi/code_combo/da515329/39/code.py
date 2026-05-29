# Combo variant of da515329 (index 39)
# combo_spec: D5/1+D15/0
# Name: 八方向发射 + 45度转弯
#
# Design analysis
# ---------------
# a) Concrete behavior / trace
#    - Input stays visually simple: one teal(8) hollow plus-sign on a black background.
#    - The fused output adds four diagonal launch seeds in addition to the four real arm tips.
#      If the arm length is `a`, the real tips are at radius `a` in the four cardinal directions,
#      and the virtual tips are at radius `a+1` in the four diagonal directions. The output therefore
#      begins from an 8-point ring around the cross rather than from only four cardinal tips.
#    - Each source starts heading one 45-degree clockwise step away from its outward direction.
#      Example for `a = 1`:
#        * Original top tip path: source `(cr-1, cc)`, initial heading east, then only 90-degree turns.
#        * Fused top tip path: source `(cr-1, cc)`, initial heading north-east.
#        * New north-west virtual source: `(cr-2, cc-2)`, initial heading north.
#      The local effect is immediate: outputs show long diagonal ladders and octagonal turns that do
#      not exist in the base task's purely orthogonal spiral walls.
# b) Complexity calibration
#    - I keep a single fixed virtual-source rule: always 4 extra diagonal sources, always at radius
#      `arm+1`, and always 45-degree turning. There are no sample-specific modes to guess.
#    - Arm length is restricted to 1..3. With 8 simultaneous seeds, the original `arm=4` footprint
#      becomes over-constrained and collapses into sparse dead starts, which weakens both readability
#      and generator stability.
#    - The visual change is strong rather than subtle: every accepted sample must show many diagonal
#      adjacencies and all four virtual sources must actually move.
# c) Hidden assumptions broken from the original solver
#    - Original turning helpers assumed a 4-direction cycle. The fused solver must promote heading
#      state, right-turn, and left-turn to an 8-direction clock.
#    - Original source enumeration used only the real arm tips. The fused solver must synthesize the
#      4 diagonal virtual tips from the detected center and arm length.
#    - The original spacing rule was checked only against existing orthogonal neighbors. Keeping that
#      orthogonal moat is intentional here: if spacing were upgraded to full 8-neighborhood blocking,
#      the board becomes too sparse and the "extremely dense" fused look disappears.
#    - Because the 8 launch cells are packed more tightly than later corridors, movement checks must
#      treat those initial source cells as exempt from the normal moat rule. Generator-side placement
#      guarantees that only those seeds need the exemption.
# d) Placement legality
#    - Generator still constructs exactly one equal-arm cross, so center detection via max teal row /
#      column remains valid.
#    - With virtual sources at radius `arm+1`, the center must satisfy `center_margin >= 2*arm+3`.
#      This keeps every source in bounds and leaves at least `arm+2` clearance from any launch seed
#      to the border, which is enough for visible outward growth before boundary pressure dominates.
#    - Every candidate input is validated by running both the fused solver and the copied base solver.
# e) Visual observability
#    - From 2-4 examples, a human can see three stable clues:
#        1. the output contains diagonal teal chains everywhere, not just orthogonal corridors;
#        2. there are visible diagonal seed pixels around the cross that were absent in the input;
#        3. the overall filling looks octagonal / kaleidoscopic rather than rectangular.
#    - No invisible metadata such as color ids or ordering-by-value is used.
# f) Visual cleanliness
#    - Only two colors remain in play, and the input object stays a single clean cross.
#    - The fused rule is dense but not muddy because spacing is still enforced in the 4-neighborhood,
#      so parallel corridors keep a 1-cell orthogonal gap even while diagonal runs appear.
# g) Change inferability / anti-degeneration metric
#    - I use two quantitative checks in generation:
#        * all 4 virtual sources must paint at least one extra cell;
#        * the output's diagonal-pair count must exceed the base solver's count by at least 40.
#      This prevents silent collapse back to the original 4-source / 90-degree behavior on the same
#      input shape.
# h) New solver assumptions and how generation enforces them
#    - Assumption: input uses only colors 0 and 8 and contains exactly one hollow equal-arm cross.
#      Enforced by direct construction.
#    - Assumption: arm length is 1..3 and the cross center obeys the widened launch margin.
#      Enforced by generator search space.
#    - Assumption: diagonal source cells at radius `arm+1` are empty in the input.
#      Enforced automatically by the hollow-plus construction and the chosen center margin.

import math
import random
import sys

import numpy as np

_SOLVER_UTILS = "/home/lixinyu/snap/new/solvers"
if _SOLVER_UTILS not in sys.path:
    sys.path.append(_SOLVER_UTILS)

from arc_utils import neighbors4

_DIRS_URDL = [(-1, 0), (0, 1), (1, 0), (0, -1)]
_DIRS_8 = [
    (-1, 0),
    (-1, 1),
    (0, 1),
    (1, 1),
    (1, 0),
    (1, -1),
    (0, -1),
    (-1, -1),
]
_DIRS_DIAG = [(-1, 1), (1, 1), (1, -1), (-1, -1)]

CROSS_COLOR = 8
ARM_LEN_MIN, ARM_LEN_MAX = 1, 3
GRID_SIZE_BASE = 6
LAUNCH_CLEARANCE = 2
VIRTUAL_RADIUS_OFFSET = 1
FILL_LO, FILL_HI = 0.33, 0.48
MIN_DIAG_PAIR_GAP = 40
MAX_OUTER_RETRIES = 200


def _right90(d):
    return _DIRS_URDL[(_DIRS_URDL.index(d) + 1) % 4]


def _left90(d):
    return _DIRS_URDL[(_DIRS_URDL.index(d) + 3) % 4]


def _right45(d):
    return _DIRS_8[(_DIRS_8.index(d) + 1) % 8]


def _left45(d):
    return _DIRS_8[(_DIRS_8.index(d) - 1) % 8]


def _add(pos, d, k=1):
    return (pos[0] + d[0] * k, pos[1] + d[1] * k)


def _grid_size_for_arm(arm: int) -> int:
    return 4 * arm + 2 * math.ceil(arm / 2) + GRID_SIZE_BASE


def _build_input(arm: int, size: int, cr: int, cc: int) -> np.ndarray:
    grid = np.zeros((size, size), dtype=int)
    offsets = np.arange(1, arm + 1)
    grid[cr - offsets, cc] = CROSS_COLOR
    grid[cr + offsets, cc] = CROSS_COLOR
    grid[cr, cc - offsets] = CROSS_COLOR
    grid[cr, cc + offsets] = CROSS_COLOR
    return grid


def _extract_cross(grid: np.ndarray):
    grid = np.asarray(grid, dtype=int)
    h, w = grid.shape
    row_counts = np.sum(grid == CROSS_COLOR, axis=1)
    col_counts = np.sum(grid == CROSS_COLOR, axis=0)
    cr = int(np.argmax(row_counts))
    cc = int(np.argmax(col_counts))

    def inb(r: int, c: int) -> bool:
        return 0 <= r < h and 0 <= c < w

    arms = []
    for d in _DIRS_URDL:
        dr, dc = d
        r, c = cr, cc
        cells = []
        while inb(r + dr, c + dc) and grid[r + dr, c + dc] == CROSS_COLOR:
            r += dr
            c += dc
            cells.append((r, c))
        if cells:
            arms.append((d, cells))
    return cr, cc, arms


def _diagonal_pair_count(grid: np.ndarray) -> int:
    h, w = grid.shape
    total = 0
    for r in range(h - 1):
        for c in range(w - 1):
            if grid[r, c] == CROSS_COLOR and grid[r + 1, c + 1] == CROSS_COLOR:
                total += 1
            if grid[r + 1, c] == CROSS_COLOR and grid[r, c + 1] == CROSS_COLOR:
                total += 1
    return total


def _solve_base_reference(grid: np.ndarray) -> np.ndarray:
    grid = np.asarray(grid, dtype=int)
    h, w = grid.shape
    out = grid.copy()
    cr, cc, arms = _extract_cross(grid)

    def inb(r: int, c: int) -> bool:
        return 0 <= r < h and 0 <= c < w

    def is_border(r: int, c: int) -> bool:
        return r == 0 or r == h - 1 or c == 0 or c == w - 1

    arm_len = max((len(cells) for _, cells in arms), default=0)
    even_center = (cr + cc) % 2 == 0

    def spacing_ok(cell, prev, start) -> bool:
        r, c = cell
        for nr, nc in neighbors4(r, c):
            if inb(nr, nc) and out[nr, nc] != 0 and (nr, nc) != prev and (nr, nc) != start:
                return False
        return True

    def can_move(cur, d, length: int) -> bool:
        prev = cur
        for t in range(1, length + 1):
            rr, cc2 = _add(cur, d, t)
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
            out[rr, cc2] = CROSS_COLOR
        agent["cur"] = (r0 + d[0] * length, c0 + d[1] * length)

    agents = []
    for outward, cells in arms:
        agents.append(
            {
                "cur": cells[-1],
                "dir": _right90(outward),
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
            agent["dir"] = _right90(agent["out_dir"])
            agent["p"] = 0
            return True

        dirs = [_right90(agent["dir"]), agent["dir"]]
        if not is_border(*agent["cur"]):
            dirs.append(_left90(agent["dir"]))

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
        out[(rr % 2 == rp) & (ccg % 2 == cp)] = 0
    elif arm_len % 2 == 1:
        to_clear = []
        for r in range(h):
            for c in range(w):
                if out[r, c] != CROSS_COLOR or not is_border(r, c):
                    continue
                deg = 0
                nbr = None
                for nr, nc in neighbors4(r, c):
                    if inb(nr, nc) and out[nr, nc] == CROSS_COLOR:
                        deg += 1
                        nbr = (nr, nc)
                if deg == 1 and nbr is not None and is_border(*nbr):
                    to_clear.append((r, c))
        for r, c in to_clear:
            out[r, c] = 0

        br = (h - 1, w - 1)
        if out[br] == 0 and all(
            not (inb(nr, nc) and out[nr, nc] == CROSS_COLOR)
            for nr, nc in neighbors4(br[0], br[1])
        ):
            out[br] = CROSS_COLOR
    else:
        if arm_len >= 2:
            out[0 : min(h, arm_len - 1), 0] = CROSS_COLOR
        if arm_len >= 1:
            out[0, max(0, w - arm_len) : w] = CROSS_COLOR
        if h >= 2:
            for c in range(w):
                if out[h - 1, c] == CROSS_COLOR and out[h - 2, c] == 0:
                    out[h - 1, c] = 0

    return out


def _solve_variant(grid: np.ndarray, want_stats: bool = False):
    grid = np.asarray(grid, dtype=int)
    h, w = grid.shape
    out = grid.copy()
    cr, cc, arms = _extract_cross(grid)
    arm_len = max((len(cells) for _, cells in arms), default=0)
    diag_radius = arm_len + VIRTUAL_RADIUS_OFFSET

    def inb(r: int, c: int) -> bool:
        return 0 <= r < h and 0 <= c < w

    def is_border(r: int, c: int) -> bool:
        return r == 0 or r == h - 1 or c == 0 or c == w - 1

    virtual_sources = []
    for d in _DIRS_DIAG:
        src = (cr + d[0] * diag_radius, cc + d[1] * diag_radius)
        if not inb(*src):
            raise ValueError("virtual source out of bounds")
        virtual_sources.append((src, d))
        out[src] = CROSS_COLOR

    source_cells = {cells[-1] for _, cells in arms} | {src for src, _ in virtual_sources}
    stats = {
        "virtual_movers": 0,
        "diag_steps": 0,
        "card_steps": 0,
    }

    def spacing_ok(cell, prev, source_allow) -> bool:
        r, c = cell
        for nr, nc in neighbors4(r, c):
            if (
                inb(nr, nc)
                and out[nr, nc] != 0
                and (nr, nc) != prev
                and (nr, nc) not in source_allow
            ):
                return False
        return True

    def can_move(cur, d, length: int, source_allow) -> bool:
        prev = cur
        for t in range(1, length + 1):
            rr, cc2 = _add(cur, d, t)
            if not inb(rr, cc2) or out[rr, cc2] != 0:
                return False
            if not spacing_ok((rr, cc2), prev, source_allow):
                return False
            prev = (rr, cc2)
        return True

    def do_move(agent, d, length: int) -> None:
        r0, c0 = agent["cur"]
        if not agent["moved"]:
            agent["moved"] = True
            if agent["is_virtual"]:
                stats["virtual_movers"] += 1
        if d[0] != 0 and d[1] != 0:
            stats["diag_steps"] += length
        else:
            stats["card_steps"] += length
        for t in range(1, length + 1):
            rr, cc2 = r0 + d[0] * t, c0 + d[1] * t
            out[rr, cc2] = CROSS_COLOR
        agent["cur"] = (r0 + d[0] * length, c0 + d[1] * length)

    agents = []
    for outward, cells in arms:
        agents.append(
            {
                "cur": cells[-1],
                "dir": _right45(outward),
                "done": False,
                "p": 0,
                "moved": False,
                "is_virtual": False,
                "source_allow": set(source_cells - {cells[-1]}),
            }
        )
    for src, outward in virtual_sources:
        agents.append(
            {
                "cur": src,
                "dir": _right45(outward),
                "done": False,
                "p": 0,
                "moved": False,
                "is_virtual": True,
                "source_allow": set(source_cells - {src}),
            }
        )

    def step_agent(agent) -> bool:
        if agent["done"]:
            return False

        idx = _DIRS_8.index(agent["dir"])
        dirs = [_DIRS_8[(idx + 1) % 8], agent["dir"]]
        if not is_border(*agent["cur"]):
            dirs.append(_DIRS_8[(idx - 1) % 8])

        for nd in dirs:
            turned = nd != agent["dir"]
            length = 1 if turned else (2 if agent["p"] == 0 else 1)
            if can_move(agent["cur"], nd, length, agent["source_allow"]):
                do_move(agent, nd, length)
                agent["p"] = 1 if (agent["p"] == 0 and turned) else 0
                agent["dir"] = nd
                return True
            if not turned and length == 2 and can_move(agent["cur"], nd, 1, agent["source_allow"]):
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

    return (out, stats) if want_stats else out


def main(grid: np.ndarray) -> np.ndarray:
    """
    Grow eight clockwise octilinear spirals from a teal cross and four diagonal virtual tips.

    Input structure:
    - bg = 0 (black). A single teal (8) plus-sign cross with a center hole marks the launch hub.
    - Only the four cardinal arms are drawn in the input; the extra diagonal sources are implicit.

    Rule:
    - Launch 8 agents: 4 from the real arm tips and 4 from diagonal virtual tips one cell farther
      out than the arm length.
    - Each agent starts one 45-degree clockwise step from its outward direction, then prefers
      turn-right-by-45 > go-straight > turn-left-by-45, with left still disallowed on the border.
    - Straight motion keeps the base task's 2/1 rhythm: length 2 normally, but only length 1
      immediately after a turn. New teal cells must still keep a 1-cell orthogonal moat from older
      corridors, except for the other 7 launch seeds.

    Key insight: The cross still tells you the center and arm length, but the real pattern is no
    longer a 4-way rectangular spiral. You must synthesize the 4 missing diagonal tips and follow an
    8-direction turning clock, which creates a visibly denser octagonal labyrinth.
    """

    return _solve_variant(grid, want_stats=False)


def _is_good_instance(inp: np.ndarray, out: np.ndarray, stats: dict) -> bool:
    fill = float(np.mean(out == CROSS_COLOR))
    if not (FILL_LO <= fill <= FILL_HI):
        return False
    if stats["virtual_movers"] != 4:
        return False
    base_out = _solve_base_reference(inp)
    diag_gap = _diagonal_pair_count(out) - _diagonal_pair_count(base_out)
    if diag_gap < MIN_DIAG_PAIR_GAP:
        return False
    if np.array_equal(out, base_out):
        return False
    return True


def _gen_attempt() -> np.ndarray | None:
    arm = random.randint(ARM_LEN_MIN, ARM_LEN_MAX)
    size = _grid_size_for_arm(arm)
    center_margin = 2 * arm + VIRTUAL_RADIUS_OFFSET + LAUNCH_CLEARANCE
    lo = center_margin
    hi = size - center_margin - 1
    if lo > hi:
        return None

    candidates = [(r, c) for r in range(lo, hi + 1) for c in range(lo, hi + 1)]
    random.shuffle(candidates)
    for cr, cc in candidates:
        inp = _build_input(arm, size, cr, cc)
        try:
            out, stats = _solve_variant(inp, want_stats=True)
        except Exception:
            continue
        if _is_good_instance(inp, out, stats):
            if np.array_equal(main(inp), out):
                return inp
    return None


def generate_input() -> np.ndarray:
    for _ in range(MAX_OUTER_RETRIES):
        grid = _gen_attempt()
        if grid is not None:
            return grid
    raise RuntimeError("generate_input: failed after retries")
