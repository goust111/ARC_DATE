# Combo variant of da515329 (index 18)
# combo_spec: D10/2+D13/0
# Name: 正方形居中裁剪 + 十字强制居中

import math
import random

import numpy as np

_DIRS_URDL = [(-1, 0), (0, 1), (1, 0), (0, -1)]
CROSS_COLOR = 8

# Design analysis:
# spec_variant:
#   input:
#     grid: odd square, side length 13-31 in the generator below
#     background: black (0)
#     object:
#       - one teal plus sign with a 1-cell black center hole
#       - the hole is forced to the exact grid center, so all four arm tips are
#         equally far from the corresponding borders
#   output:
#     transformation:
#       - first run the seed task's four clockwise spiral walkers on the full
#         canvas, using the same spacing and turn-priority logic
#       - then discard the outer termination ring and keep only the largest
#         equal-radius square around the center hole
#   concretization:
#     - Small example: arm=2 on a 17x17 centered input. The seed solver grows the
#       same four turtles it would normally grow, but its last border-fixing cells
#       are no longer needed once the cross is perfectly centered.
#     - Cropping the centered 15x15 square removes those asymmetric edge caps and
#       leaves a 90-degree rotationally symmetric square spiral.
#     - Compared with the original task, position reasoning disappears: the hole is
#       always at the visible center and the output is always a smaller centered
#       square rather than the full input canvas.
#   complexity_calibration:
#     - Arm length still varies across four values, so the solver must infer how
#       the spiral expands from the tip positions.
#     - Extra even padding varies the number of outer rings without reintroducing
#       off-center placement, so samples remain visibly distinct.
#   old_assumptions_that_break:
#     - The seed task relied on arbitrary center placement and then learned several
#       asymmetric border cleanups. Those edge hacks are no longer the final answer
#       once the cross is forced to the center.
#     - Keeping the full canvas would visually preserve those termination artifacts,
#       which would contradict the "perfect centered square" intent of the combo.
#     - A literal even-side crop around a single center cell is not exactly centered
#       on a discrete grid, so this implementation uses the largest odd centered
#       square; the generator matches that interpretation.
#   placement_legality:
#     - The generator only builds odd square inputs with the hole at `(n//2, n//2)`.
#     - Arm length is bounded so the cross stays well inside the canvas even for the
#       smallest size choice.
#     - Validation checks the output is square, strictly smaller than the input, and
#       rotationally symmetric, so degenerate uncropped samples are rejected.
#   observability_and_cleanliness:
#     - The fusion is visible from geometry alone: centered input hole, output size
#       reduced by a uniform 1-cell margin, and a clean square spiral with no lopsided
#       border stubs.
#     - No hidden ids, ordering tricks, or color remapping are used.
#   new_main_assumptions_audited:
#     - `argmax` on row/column counts assumes a single cross; the generator creates
#       exactly one cross.
#     - The arm scan assumes four equal nonzero arms around the center hole; the
#       generator enforces that by construction.
#     - The centered crop assumes there is at least one full ring outside the kept
#       square (`radius >= 1`); the smallest generated input is 13x13, so this holds.


def neighbors4(r: int, c: int):
    yield r - 1, c
    yield r + 1, c
    yield r, c - 1
    yield r, c + 1


def _turn_right(d):
    return _DIRS_URDL[(_DIRS_URDL.index(d) + 1) % 4]


def _turn_left(d):
    return _DIRS_URDL[(_DIRS_URDL.index(d) + 3) % 4]


def _find_center_and_arm_len(grid: np.ndarray) -> tuple[int, int, int]:
    row_counts = np.sum(grid == CROSS_COLOR, axis=1)
    col_counts = np.sum(grid == CROSS_COLOR, axis=0)
    cr = int(np.argmax(row_counts))
    cc = int(np.argmax(col_counts))

    arm_len = 0
    while cr - arm_len - 1 >= 0 and grid[cr - arm_len - 1, cc] == CROSS_COLOR:
        arm_len += 1
    return cr, cc, arm_len


def _solve_full_canvas(grid: np.ndarray) -> np.ndarray:
    grid = np.asarray(grid, dtype=int)
    h, w = grid.shape
    out = grid.copy()
    cr, cc, arm_len = _find_center_and_arm_len(grid)

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
            out[rr, cc2] = CROSS_COLOR
        agent["cur"] = (r0 + d[0] * length, c0 + d[1] * length)

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

    even_center = (cr + cc) % 2 == 0

    agents = []
    for outward, cells in arms:
        agents.append(
            {
                "cur": cells[-1],
                "dir": _turn_right(outward),
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
            agent["dir"] = _turn_right(agent["out_dir"])
            agent["p"] = 0
            return True

        dirs = [_turn_right(agent["dir"]), agent["dir"]]
        if not is_border(*agent["cur"]):
            dirs.append(_turn_left(agent["dir"]))

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


def _center_square_crop(full: np.ndarray, center: tuple[int, int]) -> np.ndarray:
    cr, cc = center
    radius = min(cr, full.shape[0] - 1 - cr, cc, full.shape[1] - 1 - cc) - 1
    if radius < 1:
        raise ValueError("centered crop is empty")
    return full[cr - radius : cr + radius + 1, cc - radius : cc + radius + 1]


def main(grid: np.ndarray) -> np.ndarray:
    """
    Grow the original clockwise spiral pattern from a centered teal cross, then keep only the centered square core.

    Input structure:
    - bg = 0 (black), with one teal (8) plus sign whose 1-cell center hole sits at the exact middle of an odd square grid.
    - The four arms have equal length and stop before the border, so every tip has the same surrounding room.

    Rule:
    - Launch one spiral walker from each arm tip, with the same local motion policy as the seed task:
      right turn is preferred, then straight, then left when not on a border; straight segments use length 2 except
      for the step immediately after a turn, which uses length 1; each new wall cell must keep a 1-cell orthogonal gap.
    - After drawing on the full canvas, crop away the outer termination ring and return the largest odd square that
      remains perfectly centered on the cross hole.

    Key insight:
    - Forcing the source to the exact center removes positional ambiguity, and the centered crop strips off the
      asymmetric border-fixing cells, leaving a clean square spiral whose structure is determined only by the spiral growth.
    """

    grid = np.asarray(grid, dtype=int)
    cr, cc, _arm_len = _find_center_and_arm_len(grid)
    full = _solve_full_canvas(grid)
    return _center_square_crop(full, (cr, cc))


ARM_LEN_MIN, ARM_LEN_MAX = 1, 4
EXTRA_PADDING_CHOICES = (0, 2, 4)
FILL_LO, FILL_HI = 0.42, 0.58
MAX_OUTER_RETRIES = 200


def _build_centered_input(arm: int, size: int) -> np.ndarray:
    grid = np.zeros((size, size), dtype=int)
    cr = cc = size // 2
    offsets = np.arange(1, arm + 1)
    grid[cr - offsets, cc] = CROSS_COLOR
    grid[cr + offsets, cc] = CROSS_COLOR
    grid[cr, cc - offsets] = CROSS_COLOR
    grid[cr, cc + offsets] = CROSS_COLOR
    return grid


def _gen_attempt() -> np.ndarray | None:
    arm = random.randint(ARM_LEN_MIN, ARM_LEN_MAX)
    base_size = 4 * arm + 2 * math.ceil(arm / 2) + 7
    size = base_size + random.choice(EXTRA_PADDING_CHOICES)
    inp = _build_centered_input(arm, size)

    out = main(inp)
    fill = float(np.mean(out == CROSS_COLOR))
    if not (FILL_LO <= fill <= FILL_HI):
        return None
    if out.shape[0] != out.shape[1]:
        return None
    if out.shape[0] >= inp.shape[0]:
        return None
    if not np.array_equal(out, np.rot90(out)):
        return None
    if not np.array_equal(main(inp), out):
        return None
    return inp


def generate_input() -> np.ndarray:
    for _ in range(MAX_OUTER_RETRIES):
        grid = _gen_attempt()
        if grid is not None:
            return grid
    raise RuntimeError("generate_input: failed after retries")
