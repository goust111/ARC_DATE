# Combo variant of da515329 (index 44)
# combo_spec: D4/2+D15/0
# Name: X形对角源 + 45度转弯

from __future__ import annotations

import random

import numpy as np

from arc_utils import neighbors8


# Design analysis for the fused variant:
# - Concrete behavior:
#   The original task fires four spirals from the tips of an orthogonal plus. Here the source
#   becomes an X with an empty center, so the four tips sit on NW/NE/SE/SW diagonals. Each tip
#   first nudges outward along its own diagonal, then starts a clockwise spiral whose heading
#   changes by 45 degrees at each turn. A small arm-2 example is:
#   input tip NW -> outward nudge to the farther NW cell -> heading rotates to N -> preferred
#   right turn rotates to NE -> later to E -> later to SE, etc. Compared with the original,
#   turns now alternate between cardinal and diagonal runs, so the output becomes a diamond /
#   octagonal spiral instead of a rectangular one.
# - Complexity calibration:
#   Using full 8-neighbor spacing keeps the pattern readable: diagonal segments are visually
#   obvious, but the paths still leave single-pixel gaps and do not collapse into a solid fill.
#   The rule still has only a few repeated moving parts: four sources, one clockwise preference,
#   and the same 2/1 step rhythm everywhere.
# - Hidden assumptions that had to change:
#   The original code assumes row/column maxima reveal the center, only four launch directions
#   exist, orthogonal spacing is sufficient, and parity-specific border patches repair the maze.
#   None of those survive a diagonal X. The fused solver instead detects the source from the
#   diagonal-cross bounding box, uses an 8-direction turn system, checks spacing in the Moore
#   neighborhood, and avoids any row/column parity cleanup.
# - Placement legality:
#   The generator keeps every tip at least arm+2 cells from the border, which is enough for the
#   parity-dependent outward launch (1 step for odd arms, 2 for even arms) and prevents overlap
#   between the input X and the first spiral ring.
# - Visual observability:
#   Across 2-4 examples a human can see two stable cues: the input object is unmistakably an X,
#   and the output contains long diagonal chains plus 45-degree bends that the original task does
#   not have. No color remapping or hidden ordering is involved.
# - Cleanliness / non-degeneration:
#   If even-length X arms launch immediately into the spiral, the four agents crash back into the
#   center and the output degenerates into a tiny knot. The parity-aware outward spacer is the
#   minimum extra rule that keeps all arm lengths visually consistent.


_COLOR = 8
_DIRS_8 = [
    (-1, 0),   # N
    (-1, 1),   # NE
    (0, 1),    # E
    (1, 1),    # SE
    (1, 0),    # S
    (1, -1),   # SW
    (0, -1),   # W
    (-1, -1),  # NW
]
_DIAG_DIRS = [(-1, -1), (-1, 1), (1, 1), (1, -1)]


def _detect_x_source(grid: np.ndarray) -> tuple[int, int, int, set[tuple[int, int]]]:
    pts = np.argwhere(grid == _COLOR)
    if pts.size == 0:
        return 0, 0, 0, set()

    r0, c0 = np.min(pts, axis=0)
    r1, c1 = np.max(pts, axis=0)
    cr = int((int(r0) + int(r1)) // 2)
    cc = int((int(c0) + int(c1)) // 2)
    cells = {(int(r), int(c)) for r, c in pts}
    arm = max((abs(r - cr) for r, _c in cells), default=0)
    return cr, cc, arm, cells


def _turn_right(d: tuple[int, int]) -> tuple[int, int]:
    return _DIRS_8[(_DIRS_8.index(d) + 1) % 8]


def _turn_left(d: tuple[int, int]) -> tuple[int, int]:
    return _DIRS_8[(_DIRS_8.index(d) - 1) % 8]


def _path_mix_metrics(grid: np.ndarray) -> tuple[int, int]:
    diag_pairs = 0
    ortho_pairs = 0
    h, w = grid.shape

    for r in range(h):
        for c in range(w):
            if int(grid[r, c]) != _COLOR:
                continue
            if r + 1 < h and int(grid[r + 1, c]) == _COLOR:
                ortho_pairs += 1
            if c + 1 < w and int(grid[r, c + 1]) == _COLOR:
                ortho_pairs += 1
            if r + 1 < h and c + 1 < w and int(grid[r + 1, c + 1]) == _COLOR:
                diag_pairs += 1
            if r + 1 < h and c - 1 >= 0 and int(grid[r + 1, c - 1]) == _COLOR:
                diag_pairs += 1

    return diag_pairs, ortho_pairs


def main(grid: np.ndarray) -> np.ndarray:
    """
    Grow four clockwise diamond spirals from the tips of a teal diagonal X.

    Input structure:
    - bg = 0 (black). A single teal (8) diagonal cross with an empty center; the four equal arms point to the corners.

    Rule:
    - Each diagonal tip first takes a short outward launch, then heads 45 degrees clockwise from its outward ray.
    - Movement priority is turn right by 45 degrees > go straight > turn left by 45 degrees (left only away from the border).
    - A turn always advances 1 cell; a straight advance uses length 2 except for the straight step immediately after a turn, which also uses length 1.
    - Every new cell must keep a 1-cell 8-neighbor gap from previously grown spiral cells, but the original input X itself may be approached during launch.

    Key insight:
    - Once the plus-cross assumptions are replaced by a diagonal X and an 8-direction heading system, the same clockwise wall-following rhythm yields a clean octagonal / diamond spiral maze.
    """

    grid = np.asarray(grid, dtype=int)
    h, w = grid.shape
    out = grid.copy()
    cr, cc, arm, source_cells = _detect_x_source(grid)
    if arm == 0:
        return out

    def inb(r: int, c: int) -> bool:
        return 0 <= r < h and 0 <= c < w

    def is_border(r: int, c: int) -> bool:
        return r == 0 or r == h - 1 or c == 0 or c == w - 1

    def add(pos: tuple[int, int], d: tuple[int, int], k: int = 1) -> tuple[int, int]:
        return pos[0] + d[0] * k, pos[1] + d[1] * k

    def spacing_ok(cell: tuple[int, int], prev: tuple[int, int], start: tuple[int, int]) -> bool:
        for nr, nc in neighbors8(cell[0], cell[1]):
            if not inb(nr, nc):
                continue
            if int(out[nr, nc]) != _COLOR:
                continue
            if (nr, nc) == prev or (nr, nc) == start or (nr, nc) in source_cells:
                continue
            return False
        return True

    def can_move(cur: tuple[int, int], d: tuple[int, int], length: int) -> bool:
        prev = cur
        for t in range(1, length + 1):
            rr, cc2 = add(cur, d, t)
            if not inb(rr, cc2) or int(out[rr, cc2]) != 0:
                return False
            if not spacing_ok((rr, cc2), prev, cur):
                return False
            prev = (rr, cc2)
        return True

    def do_move(agent: dict[str, object], d: tuple[int, int], length: int) -> None:
        r0, c0 = agent["cur"]  # type: ignore[index]
        for t in range(1, length + 1):
            rr, cc2 = r0 + d[0] * t, c0 + d[1] * t
            out[rr, cc2] = _COLOR
        agent["cur"] = (r0 + d[0] * length, c0 + d[1] * length)

    launch_steps = 1 if arm % 2 == 1 else 2
    agents: list[dict[str, object]] = []
    for outward in _DIAG_DIRS:
        agents.append(
            {
                "cur": (cr + outward[0] * arm, cc + outward[1] * arm),
                "dir": _turn_right(outward),
                "out_dir": outward,
                "phase": launch_steps,
                "done": False,
                "p": 0,
            }
        )

    def step_agent(agent: dict[str, object]) -> bool:
        if agent["done"]:
            return False

        if int(agent["phase"]) > 0:
            out_dir = agent["out_dir"]  # type: ignore[assignment]
            if can_move(agent["cur"], out_dir, 1):  # type: ignore[arg-type]
                do_move(agent, out_dir, 1)
            agent["phase"] = int(agent["phase"]) - 1
            agent["dir"] = _turn_right(out_dir)
            agent["p"] = 0
            return True

        cur = agent["cur"]  # type: ignore[assignment]
        dirs = [_turn_right(agent["dir"]), agent["dir"]]  # type: ignore[arg-type]
        if not is_border(*cur):
            dirs.append(_turn_left(agent["dir"]))  # type: ignore[arg-type]

        for nd in dirs:
            turned = nd != agent["dir"]
            length = 1 if turned else (2 if int(agent["p"]) == 0 else 1)
            if can_move(cur, nd, length):
                do_move(agent, nd, length)
                agent["p"] = 1 if (int(agent["p"]) == 0 and turned) else 0
                agent["dir"] = nd
                return True
            if not turned and length == 2 and can_move(cur, nd, 1):
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
        if all(bool(ag["done"]) for ag in agents):
            break

    return out


# --------------- generator ---------------

ARM_LEN_MIN, ARM_LEN_MAX = 1, 4
GRID_SIZE_PAD = 8
TIP_GAP_MIN = 2
FILL_LO, FILL_HI = 0.37, 0.45
DIAG_PAIR_MIN = 6
MAX_OUTER_RETRIES = 300


def _build_input(arm: int, size: int, cr: int, cc: int) -> np.ndarray:
    grid = np.zeros((size, size), dtype=int)
    for k in range(1, arm + 1):
        for dr, dc in _DIAG_DIRS:
            grid[cr + dr * k, cc + dc * k] = _COLOR
    return grid


def _gen_attempt() -> np.ndarray | None:
    arm = random.randint(ARM_LEN_MIN, ARM_LEN_MAX)
    size = 4 * arm + GRID_SIZE_PAD

    lo = 2 * arm + TIP_GAP_MIN
    hi = size - 2 * arm - TIP_GAP_MIN - 1
    if lo > hi:
        return None

    candidates = [(r, c) for r in range(lo, hi + 1) for c in range(lo, hi + 1)]
    random.shuffle(candidates)

    for cr, cc in candidates:
        inp = _build_input(arm, size, cr, cc)
        out = main(inp)
        fill = float(np.mean(out == _COLOR))
        diag_pairs, ortho_pairs = _path_mix_metrics(out)
        if not (FILL_LO <= fill <= FILL_HI):
            continue
        if diag_pairs < DIAG_PAIR_MIN or ortho_pairs <= diag_pairs:
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
