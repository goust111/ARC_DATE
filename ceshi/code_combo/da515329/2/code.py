# Combo variant of da515329 (index 2)
# combo_spec: D1/2+D3/3
# Name: 交替方向螺旋 + 转弯变色

import math
import random

import numpy as np

from arc_utils import neighbors4

_DIRS_URDL = [(-1, 0), (0, 1), (1, 0), (0, -1)]

# Design analysis for the combo variant
# ------------------------------------
# a) Concrete behavior and trace
#    - Input stays the same: one hollow teal cross on black, with four equal arms.
#    - Output still occupies the same family of 1-cell-gap rectangular spiral corridors, but two
#      visible rules are fused:
#      1. The visible spiral layers alternate handedness. Layer 0 (closest to the cross) uses the
#         original clockwise geometry, layer 1 uses the mirrored counter-clockwise geometry, layer 2
#         returns to clockwise, and so on.
#      2. Within whichever handedness a layer uses, the path flips between teal (8) and red (2)
#         exactly when a successful 90-degree turn is taken.
#    - Concrete trace on a small arm=1 case:
#      * In the clockwise solve, the top tip first extends one cell upward, then starts an eastward
#        corridor in teal; when that corridor turns south, the turning step and following southward
#        run become red.
#      * In the mirrored counter-clockwise solve, the same visible ring bends at the mirrored corner,
#        so the red segment appears at a different location.
#      * The combo output keeps the inner ring from the clockwise solve, the next ring from the
#        counter-clockwise solve, etc. The red/teal pattern therefore moves together with the chosen
#        ring geometry instead of being painted independently afterward.
# b) Complexity calibration
#    - Only one structural binary choice is introduced per ring: clockwise vs counter-clockwise.
#    - Only one color binary choice is introduced per segment: teal vs red, toggled by turns.
#    - This is rich enough to make the output visibly different from the seed task, but small enough
#      that a human can infer the rule from a few examples because the same two-color alternation is
#      repeated at many corners across several nested rings.
# c) Implicit-assumption audit
#    - Center detection may still rely on the input cross being all teal; generate_input() preserves
#      that and never recolors the input.
#    - Occupancy checks can no longer use "== 8" once path cells may be red, so every collision,
#      spacing, degree, and border-cleanup test uses "!= 0" instead.
#    - The original parity/border cleanup is delicate. Instead of rewriting a second local turtle for
#      counter-clockwise motion, we mirror the input, reuse the colored clockwise solver exactly, and
#      mirror the result back. That keeps the old cleanup assumptions valid in both handednesses.
#    - Layer selection assumes the seed task's ring spacing remains 2 cells per visible layer. The
#      same Chebyshev-based layer index used in the D1 reference still matches this geometry.
# d) Placement legality
#    - The generator keeps the original square-size formula and center range, so every cross tip
#      remains at least arm+2 away from borders before spiral growth begins.
#    - Both handedness solves are produced from the same legal input family, so mixed layers stay
#      in-bounds and non-overlapping. No new free placement decisions are introduced.
# e) Visual observability
#    - A solver can see direction alternation directly because successive square-like rings bend in
#      mirrored ways.
#    - A solver can see turn-driven recoloring directly because long straight runs stay one color and
#      color changes cluster at corners rather than at fixed rows, columns, or parity classes.
#    - The generator rejects samples whose combo output is too similar to the pure clockwise colored
#      solve, so the alternating-handed part is visible in the final training set rather than hidden.
# f) Visual cleanliness
#    - No extra markers or helper pixels are added; the combo only changes which ring geometry is
#      selected and which of two colors each segment uses.
#    - The palette is capped at two path colors, keeping the dense spiral readable instead of noisy.
# g) Change inferability
#    - The two visible controls reinforce each other: choosing a different handedness changes where
#      turns occur on a layer, and turn locations are exactly where colors flip. This ties the color
#      evidence to the geometric evidence instead of asking the human to infer two unrelated rules.
# 9) New assumptions introduced here
#    - main() assumes a single hollow teal cross with equal arm lengths; generate_input() constructs
#      only that family.
#    - The combo mixer assumes at least one odd-index ring differs visibly from the clockwise solve;
#      generate_input() enforces this with an odd-layer visible-difference threshold.
#    - The recoloring rule is only inferable if both colors appear and multiple color transitions
#      exist, so generate_input() filters for a minimum red fraction and a minimum number of visible
#      heterochromatic adjacencies in the final mixed output.

PRIMARY_COLOR = 8
ALT_COLOR = 2
CROSS_COLOR = PRIMARY_COLOR

# --------------- generator (cx) ---------------

# --- CFG ---------------------------------------------------------------
ARM_LEN_MIN, ARM_LEN_MAX = 1, 4
GRID_SIZE_BASE = 6
TIP_GAP_MIN = 2
FILL_LO, FILL_HI = 0.48, 0.52
MAX_OUTER_RETRIES = 300
MIN_ALT_FRAC, MAX_ALT_FRAC = 0.24, 0.50
MIN_COLOR_TRANSITIONS = 8
MIN_ODD_LAYER_DIFF = 18
# --- end CFG -----------------------------------------------------------


def _right(d: tuple[int, int]) -> tuple[int, int]:
    return _DIRS_URDL[(_DIRS_URDL.index(d) + 1) % 4]


def _left(d: tuple[int, int]) -> tuple[int, int]:
    return _DIRS_URDL[(_DIRS_URDL.index(d) + 3) % 4]


def _other_color(color: int) -> int:
    return ALT_COLOR if color == PRIMARY_COLOR else PRIMARY_COLOR


def _build_input(arm: int, size: int, cr: int, cc: int) -> np.ndarray:
    grid = np.zeros((size, size), dtype=int)
    offsets = np.arange(1, arm + 1)
    grid[cr - offsets, cc] = CROSS_COLOR
    grid[cr + offsets, cc] = CROSS_COLOR
    grid[cr, cc - offsets] = CROSS_COLOR
    grid[cr, cc + offsets] = CROSS_COLOR
    return grid


def _find_cross(grid: np.ndarray) -> tuple[int, int, int]:
    row_counts = np.sum(grid == CROSS_COLOR, axis=1)
    col_counts = np.sum(grid == CROSS_COLOR, axis=0)
    cr = int(np.argmax(row_counts))
    cc = int(np.argmax(col_counts))
    arm = int(max(np.max(row_counts), np.max(col_counts)) // 2)
    return cr, cc, arm


def _layer_index(shape: tuple[int, int], cr: int, cc: int, arm: int) -> np.ndarray:
    rr, ccg = np.indices(shape)
    cheb = np.maximum(np.abs(rr - cr), np.abs(ccg - cc))
    return np.maximum((cheb - arm + 1) // 2, 0)


def _solve_colored_clockwise(grid: np.ndarray) -> np.ndarray:
    grid = np.asarray(grid, dtype=int)
    h, w = grid.shape
    out = grid.copy()

    cr, cc, _arm = _find_cross(grid)

    def inb(r: int, c: int) -> bool:
        return 0 <= r < h and 0 <= c < w

    def is_border(r: int, c: int) -> bool:
        return r == 0 or r == h - 1 or c == 0 or c == w - 1

    def add(pos: tuple[int, int], d: tuple[int, int], k: int = 1) -> tuple[int, int]:
        return (pos[0] + d[0] * k, pos[1] + d[1] * k)

    def spacing_ok(cell: tuple[int, int], prev: tuple[int, int], start: tuple[int, int]) -> bool:
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

    def do_move(agent: dict[str, object], d: tuple[int, int], length: int, color: int) -> None:
        r0, c0 = agent["cur"]
        for t in range(1, length + 1):
            rr, cc2 = r0 + d[0] * t, c0 + d[1] * t
            out[rr, cc2] = color
        agent["cur"] = (r0 + d[0] * length, c0 + d[1] * length)
        agent["color"] = color

    def neighbor_colors(r: int, c: int) -> list[int]:
        return [
            int(out[nr, nc])
            for nr, nc in neighbors4(r, c)
            if inb(nr, nc) and out[nr, nc] != 0
        ]

    def paint_added_cells(cells: list[tuple[int, int]], fallback: int = PRIMARY_COLOR) -> None:
        pending = [
            (int(r), int(c))
            for r, c in cells
            if inb(int(r), int(c)) and out[int(r), int(c)] == 0
        ]
        while pending:
            progressed = False
            still_pending = []
            for r, c in pending:
                nbrs = neighbor_colors(r, c)
                if nbrs:
                    out[r, c] = nbrs[0]
                    progressed = True
                else:
                    still_pending.append((r, c))
            if not progressed:
                for r, c in still_pending:
                    out[r, c] = fallback
                break
            pending = still_pending

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
    arm_len = max((len(cells) for _, cells in arms), default=0)

    agents = []
    for outward, cells in arms:
        agents.append(
            {
                "cur": cells[-1],
                "dir": _right(outward),
                "out_dir": outward,
                "phase": "out_first" if len(cells) == 1 else None,
                "done": False,
                "p": 0,
                "color": PRIMARY_COLOR,
            }
        )

    def step_agent(agent: dict[str, object]) -> bool:
        if agent["done"]:
            return False

        if agent["phase"] == "out_first":
            if can_move(agent["cur"], agent["out_dir"], 1):
                do_move(agent, agent["out_dir"], 1, agent["color"])
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
                color = _other_color(agent["color"]) if turned else agent["color"]
                do_move(agent, nd, length, color)
                agent["p"] = 1 if (agent["p"] == 0 and turned) else 0
                agent["dir"] = nd
                return True
            if not turned and length == 2 and can_move(agent["cur"], nd, 1):
                do_move(agent, nd, 1, agent["color"])
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
                if out[r, c] == 0 or not is_border(r, c):
                    continue
                deg = 0
                nbr = None
                for nr, nc in neighbors4(r, c):
                    if inb(nr, nc) and out[nr, nc] != 0:
                        deg += 1
                        nbr = (nr, nc)
                if deg == 1 and nbr is not None and is_border(*nbr):
                    to_clear.append((r, c))
        for r, c in to_clear:
            out[r, c] = 0

        br = (h - 1, w - 1)
        if out[br] == 0 and all(out[nr, nc] == 0 for nr, nc in neighbors4(br[0], br[1]) if inb(nr, nc)):
            out[br] = PRIMARY_COLOR
    else:
        if arm_len >= 2:
            paint_added_cells([(r, 0) for r in range(min(h, arm_len - 1))])
        if arm_len >= 1:
            paint_added_cells([(0, c) for c in range(max(0, w - arm_len), w)])
        if h >= 2:
            for c in range(w):
                if out[h - 1, c] != 0 and out[h - 2, c] == 0:
                    out[h - 1, c] = 0

    return out


def _solve_colored_counterclockwise(grid: np.ndarray) -> np.ndarray:
    mirrored_in = np.fliplr(np.asarray(grid, dtype=int))
    mirrored_out = _solve_colored_clockwise(mirrored_in)
    return np.fliplr(mirrored_out)


def _mix_layers(grid: np.ndarray, cw: np.ndarray, ccw: np.ndarray) -> tuple[np.ndarray, np.ndarray]:
    cr, cc, arm = _find_cross(grid)
    layer = _layer_index(grid.shape, cr, cc, arm)
    out = np.where(layer % 2 == 0, cw, ccw)
    out[grid == CROSS_COLOR] = CROSS_COLOR
    return out.astype(int), layer


def _analyze_output(out: np.ndarray) -> dict[str, int]:
    out = np.asarray(out, dtype=int)
    h, w = out.shape
    mask = out != 0

    def inb(r: int, c: int) -> bool:
        return 0 <= r < h and 0 <= c < w

    turns = 0
    for r in range(h):
        for c in range(w):
            if not mask[r, c]:
                continue
            nbr_dirs = []
            for dr, dc in _DIRS_URDL:
                nr, nc = r + dr, c + dc
                if inb(nr, nc) and mask[nr, nc]:
                    nbr_dirs.append((dr, dc))
            if len(nbr_dirs) != 2:
                continue
            has_vertical = any(dr != 0 for dr, _dc in nbr_dirs)
            has_horizontal = any(dc != 0 for _dr, dc in nbr_dirs)
            if has_vertical and has_horizontal:
                turns += 1

    color_transitions = 0
    for r in range(h):
        for c in range(w):
            if out[r, c] == 0:
                continue
            if c + 1 < w and out[r, c + 1] != 0 and out[r, c + 1] != out[r, c]:
                color_transitions += 1
            if r + 1 < h and out[r + 1, c] != 0 and out[r + 1, c] != out[r, c]:
                color_transitions += 1

    return {
        "painted_cells": int(np.sum(mask)),
        "primary_cells": int(np.sum(out == PRIMARY_COLOR)),
        "alt_cells": int(np.sum(out == ALT_COLOR)),
        "turn_cells": int(turns),
        "color_transitions": int(color_transitions),
    }


def _solve_combo_with_stats(grid: np.ndarray) -> tuple[np.ndarray, dict[str, int]]:
    grid = np.asarray(grid, dtype=int)
    cw = _solve_colored_clockwise(grid)
    ccw = _solve_colored_counterclockwise(grid)
    out, layer = _mix_layers(grid, cw, ccw)
    stats = _analyze_output(out)
    stats["odd_layer_diff"] = int(np.sum((layer % 2 == 1) & (out != cw)))
    stats["even_layer_diff"] = int(np.sum((layer % 2 == 0) & (out != ccw)))
    stats["max_layer"] = int(np.max(layer))
    return out, stats


def main(grid: np.ndarray) -> np.ndarray:
    """
    Build a two-color rectangular spiral whose visible layers alternate handedness.

    Input structure:
    - bg = 0 (black). One hollow teal (8) cross with four equal cardinal arms seeds the pattern.

    Rule:
    - Compute the usual self-avoiding spiral from the cross tips, but treat the visible square-like
      layers as alternating sources: layer 0 uses the clockwise growth, layer 1 uses the mirrored
      counter-clockwise growth, layer 2 returns to clockwise, and so on.
    - Inside each chosen handedness, the path keeps the original 1-cell-gap movement constraints and
      flips between teal (8) and red (2) whenever a successful move changes heading by 90 degrees.
    - Straight continuation keeps the current color, and the original teal input cross remains teal.

    Key insight: The color rule is attached to turning events, so once alternating handedness shifts
    where the corners of each layer occur, it also shifts where the red/teal segments appear. The
    final output is therefore best built by mixing complete colored clockwise and counter-clockwise
    solves layer by layer.
    """

    out, _stats = _solve_combo_with_stats(grid)
    return out


def _gen_attempt() -> np.ndarray | None:
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
            out, stats = _solve_combo_with_stats(inp)
        except Exception:
            continue
        fill = float(np.mean(out != 0))
        alt_frac = stats["alt_cells"] / max(1, stats["painted_cells"])
        if not (FILL_LO <= fill <= FILL_HI):
            continue
        if not (MIN_ALT_FRAC <= alt_frac <= MAX_ALT_FRAC):
            continue
        if stats["alt_cells"] == 0 or stats["primary_cells"] == 0:
            continue
        if stats["color_transitions"] < MIN_COLOR_TRANSITIONS:
            continue
        if stats["odd_layer_diff"] < MIN_ODD_LAYER_DIFF:
            continue
        if np.array_equal(main(inp), out):
            return inp
    return None


def generate_input() -> np.ndarray:
    for _ in range(MAX_OUTER_RETRIES):
        grid = _gen_attempt()
        if grid is not None:
            return grid
    raise RuntimeError("generate_input: failed after retries")
