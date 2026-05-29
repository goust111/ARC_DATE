# Combo variant of da515329 (index 28)
# combo_spec: D1/3+D12/2+D15/1
# Name: 图例编码方向 + 臂上标记 + 障碍物决定转弯

import math
import random

import numpy as np

from arc_utils import neighbors4

_DIRS_URDL = [(-1, 0), (0, 1), (1, 0), (0, -1)]

CROSS_COLOR = 8
LEGEND_COLOR = 6
ARM_MARKER_COLOR = 1
LEFT_OBSTACLE = 2
RIGHT_OBSTACLE = 3

ARM_LEN_MIN, ARM_LEN_MAX = 2, 4
GRID_SIZE_BASE = 6
TIP_GAP_MIN = 3
FILL_LO, FILL_HI = 0.32, 0.56
MAX_OUTER_RETRIES = 400
MIN_OUTPUT_DIFF = 8
MIN_ARM_DIFF = 6


# Design analysis
# ---------------
# a) Concrete fused behavior:
#    Start from the seed task's hollow teal cross. Add one border legend: a 3-cell L in a 2x2
#    corner block. Pattern {(0,0),(1,0),(1,1)} means clockwise; {(0,1),(1,0),(1,1)} means
#    counterclockwise. Also attach one white marker to each arm on the arm's clockwise side.
#    If that marker sits beside arm cell #1 from the center, that arm launches directly into the
#    spiral; if it sits beside arm cell #2, that arm first takes one extra outward step. Example:
#    with a clockwise legend, the north tip would normally begin eastward. If its arm marker is at
#    depth 2, it first steps one cell farther north, then starts the eastward spiral. If it later
#    lands on a red obstacle cell, the next heading becomes "left of current move"; on a green
#    obstacle, it becomes "right of current move". Relative to the seed task, the global handedness,
#    the per-arm starting offset, and the local bend direction are all now explicit in the input.
# b) Complexity calibration:
#    Each code system has only two states: legend CW/CCW, arm marker depth 1/2, obstacle left/right.
#    That keeps inference finite and repeated. One sample already contains four arm markers and four
#    obstacle events, so the local rules are reinforced inside a single grid rather than depending on
#    many unrelated examples.
# c) Hidden-assumption audit against the original main():
#    The seed solver assumed every nonzero input cell was already a wall candidate, and that all four
#    agents shared one global clockwise bias. Here those assumptions break: non-teal instruction cells
#    must be traversable, center detection must only look at teal cross cells, and the initial launch
#    phase can differ per arm. The fused solver therefore copies only teal cells into `out`, parses the
#    legend separately, parses one arm marker per arm, and lets obstacle colors override the next
#    heading after a move.
# d) Placement legality:
#    The legend always sits in the top-left 2x2 corner, so it never overlaps the cross. Arm markers
#    are placed in empty side pockets computed from the arm geometry. Obstacle markers are sampled only
#    from cells that the no-obstacle spiral actually traverses and only where the input is otherwise
#    background, so they never overlap the cross, legend, or arm markers.
# e) Visual observability:
#    The three instruction layers are visually distinct: a border L-shape for global handedness,
#    symmetric side-attached white pixels for per-arm parameters, and sparse red/green route markers
#    for local turns. Because outputs are still large teal spirals, a solver can compare examples and
#    see that flipping any one instruction family changes the path.
# f) Visual cleanliness:
#    The instruction colors are few (1,2,3,6) and sparse. They do not stack on the same cells, and
#    only the teal output remains after transformation, so the output keeps the seed task's clean
#    maze-like appearance.
# g) Generator-side non-degeneration checks:
#    The generator rejects any candidate unless both obstacle colors are used, and unless the final
#    output differs materially from three ablations on the same input scaffold:
#      - removing all obstacles,
#      - flipping the legend CW↔CCW,
#      - normalizing all arm markers to depth 1.
#    Those checks make the three control dimensions causally active instead of decorative.


def _right(d: tuple[int, int]) -> tuple[int, int]:
    return _DIRS_URDL[(_DIRS_URDL.index(d) + 1) % 4]


def _left(d: tuple[int, int]) -> tuple[int, int]:
    return _DIRS_URDL[(_DIRS_URDL.index(d) + 3) % 4]


def _turn_from_spin(d: tuple[int, int], spin: str) -> tuple[int, int]:
    return _right(d) if spin == "CW" else _left(d)


def _other_turn(d: tuple[int, int], spin: str) -> tuple[int, int]:
    return _left(d) if spin == "CW" else _right(d)


def _build_input(arm: int, size: int, cr: int, cc: int) -> np.ndarray:
    grid = np.zeros((size, size), dtype=int)
    offsets = np.arange(1, arm + 1)
    grid[cr - offsets, cc] = CROSS_COLOR
    grid[cr + offsets, cc] = CROSS_COLOR
    grid[cr, cc - offsets] = CROSS_COLOR
    grid[cr, cc + offsets] = CROSS_COLOR
    return grid


def _place_legend(grid: np.ndarray, spin: str) -> None:
    cells = [(0, 0), (1, 0), (1, 1)] if spin == "CW" else [(0, 1), (1, 0), (1, 1)]
    for r, c in cells:
        grid[r, c] = LEGEND_COLOR


def _extract_cross(grid: np.ndarray) -> tuple[int, int, list[tuple[tuple[int, int], list[tuple[int, int]]]]]:
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


def _parse_legend(grid: np.ndarray) -> str:
    pts = [tuple(p) for p in np.argwhere(grid == LEGEND_COLOR)]
    assert pts, "legend must be present"
    rows = [r for r, _ in pts]
    cols = [c for _, c in pts]
    r0, c0 = min(rows), min(cols)
    rel = {(r - r0, c - c0) for r, c in pts}
    if rel == {(0, 0), (1, 0), (1, 1)}:
        return "CW"
    if rel == {(0, 1), (1, 0), (1, 1)}:
        return "CCW"
    raise AssertionError("legend shape must encode CW or CCW")


def _place_arm_markers(
    grid: np.ndarray,
    cr: int,
    cc: int,
    pads: list[int],
) -> bool:
    for outward, pad in zip(_DIRS_URDL, pads, strict=False):
        depth = 1 + pad
        side = _right(outward)
        r = cr + outward[0] * depth + side[0]
        c = cc + outward[1] * depth + side[1]
        if not (0 <= r < grid.shape[0] and 0 <= c < grid.shape[1]):
            return False
        if grid[r, c] != 0:
            return False
        grid[r, c] = ARM_MARKER_COLOR
    return True


def _parse_arm_pads(
    grid: np.ndarray,
    cr: int,
    cc: int,
    arms: list[tuple[tuple[int, int], list[tuple[int, int]]]],
) -> list[int]:
    pads = []
    for outward, _cells in arms:
        side = _right(outward)
        found = []
        for depth in (1, 2):
            r = cr + outward[0] * depth + side[0]
            c = cc + outward[1] * depth + side[1]
            if 0 <= r < grid.shape[0] and 0 <= c < grid.shape[1] and grid[r, c] == ARM_MARKER_COLOR:
                found.append(depth)
        assert len(found) == 1, "each arm must carry exactly one side marker at depth 1 or 2"
        pads.append(found[0] - 1)
    return pads


def _solve(grid: np.ndarray, want_stats: bool = False):
    grid = np.asarray(grid, dtype=int)
    h, w = grid.shape
    out = np.where(grid == CROSS_COLOR, CROSS_COLOR, 0)

    spin = _parse_legend(grid)
    cr, cc, arms = _extract_cross(grid)
    pads = _parse_arm_pads(grid, cr, cc, arms)

    def inb(r: int, c: int) -> bool:
        return 0 <= r < h and 0 <= c < w

    def is_border(r: int, c: int) -> bool:
        return r == 0 or r == h - 1 or c == 0 or c == w - 1

    def add(pos: tuple[int, int], d: tuple[int, int], k: int = 1) -> tuple[int, int]:
        return (pos[0] + d[0] * k, pos[1] + d[1] * k)

    def spacing_ok(cell: tuple[int, int], prev: tuple[int, int], start: tuple[int, int]) -> bool:
        r, c = cell
        for nr, nc in neighbors4(r, c):
            if inb(nr, nc) and out[nr, nc] == CROSS_COLOR and (nr, nc) != prev and (nr, nc) != start:
                return False
        return True

    def can_move(cur: tuple[int, int], d: tuple[int, int], length: int) -> bool:
        prev = cur
        for t in range(1, length + 1):
            rr, cc2 = add(cur, d, t)
            if not inb(rr, cc2) or out[rr, cc2] == CROSS_COLOR:
                return False
            if not spacing_ok((rr, cc2), prev, cur):
                return False
            prev = (rr, cc2)
        return True

    stats = {
        "spin": spin,
        "pads": pads,
        "used_left": 0,
        "used_right": 0,
        "visited": [[] for _ in arms],
    }

    def do_move(
        agent_idx: int,
        agent: dict[str, object],
        d: tuple[int, int],
        length: int,
    ) -> tuple[int, int] | None:
        r0, c0 = agent["cur"]
        next_dir = None
        for t in range(1, length + 1):
            rr, cc2 = r0 + d[0] * t, c0 + d[1] * t
            val = int(grid[rr, cc2])
            if val == LEFT_OBSTACLE:
                next_dir = _left(d)
                stats["used_left"] += 1
            elif val == RIGHT_OBSTACLE:
                next_dir = _right(d)
                stats["used_right"] += 1
            out[rr, cc2] = CROSS_COLOR
            stats["visited"][agent_idx].append((rr, cc2))
        agent["cur"] = (r0 + d[0] * length, c0 + d[1] * length)
        return next_dir

    agents = []
    for idx, ((outward, cells), pad) in enumerate(zip(arms, pads, strict=False)):
        agents.append(
            {
                "idx": idx,
                "cur": cells[-1],
                "dir": _turn_from_spin(outward, spin),
                "out_dir": outward,
                "launch": pad,
                "done": False,
                "p": 0,
            }
        )

    changed = True
    guard = 0
    while changed and guard < 200000:
        guard += 1
        changed = False
        for ag in agents:
            if ag["done"]:
                continue

            if ag["launch"] > 0:
                if can_move(ag["cur"], ag["out_dir"], 1):
                    do_move(ag["idx"], ag, ag["out_dir"], 1)
                ag["launch"] = 0
                ag["dir"] = _turn_from_spin(ag["out_dir"], spin)
                ag["p"] = 0
                changed = True
                continue

            dirs = [_turn_from_spin(ag["dir"], spin), ag["dir"]]
            if not is_border(*ag["cur"]):
                dirs.append(_other_turn(ag["dir"], spin))

            moved = False
            for nd in dirs:
                turned = nd != ag["dir"]
                length = 1 if turned else (2 if ag["p"] == 0 else 1)
                if can_move(ag["cur"], nd, length):
                    forced = do_move(ag["idx"], ag, nd, length)
                    ag["p"] = 1 if (ag["p"] == 0 and turned) else 0
                    ag["dir"] = forced if forced is not None else nd
                    changed = True
                    moved = True
                    break
                if not turned and length == 2 and can_move(ag["cur"], nd, 1):
                    forced = do_move(ag["idx"], ag, nd, 1)
                    ag["p"] = 0
                    ag["dir"] = forced if forced is not None else nd
                    changed = True
                    moved = True
                    break

            if not moved:
                ag["done"] = True
        if all(ag["done"] for ag in agents):
            break

    return (out, stats) if want_stats else out


def _with_flipped_legend(grid: np.ndarray) -> np.ndarray:
    alt = grid.copy()
    alt[alt == LEGEND_COLOR] = 0
    _place_legend(alt, "CCW" if _parse_legend(grid) == "CW" else "CW")
    return alt


def _with_normalized_pads(grid: np.ndarray) -> np.ndarray:
    norm = grid.copy()
    norm[norm == ARM_MARKER_COLOR] = 0
    cr, cc, _arms = _extract_cross(grid)
    ok = _place_arm_markers(norm, cr, cc, [0, 0, 0, 0])
    assert ok, "normalized arm markers must fit"
    return norm


def _behavior_metrics(grid: np.ndarray) -> dict[str, int]:
    out, stats = _solve(grid, want_stats=True)
    no_obs = grid.copy()
    no_obs[(no_obs == LEFT_OBSTACLE) | (no_obs == RIGHT_OBSTACLE)] = 0
    out_no_obs = _solve(no_obs, want_stats=False)
    out_flip = _solve(_with_flipped_legend(grid), want_stats=False)
    out_norm = _solve(_with_normalized_pads(grid), want_stats=False)
    return {
        "used_left": int(stats["used_left"]),
        "used_right": int(stats["used_right"]),
        "diff_no_obstacles": int(np.sum(out != out_no_obs)),
        "diff_flip_legend": int(np.sum(out != out_flip)),
        "diff_normalized_pads": int(np.sum(out != out_norm)),
    }


def main(grid: np.ndarray) -> np.ndarray:
    """
    Grow four spiral paths from a teal cross, but decode three instruction layers first.

    Input structure:
    - bg = 0 (black). A hollow teal cross is the only permanent seed object.
    - A border L-shaped legend in color 6 encodes whether the shared spiral bias is clockwise or counterclockwise.
    - Each arm has one attached white marker; its depth along the arm encodes whether that arm takes an extra outward launch step.
    - Red (2) and green (3) cells lie on future routes as turn-trigger obstacles.

    Rule:
    - Every tip launches a turtle. The legend decides the initial perpendicular heading and the preferred bend order: clockwise samples use turn-right > straight > turn-left, while counterclockwise samples mirror that order.
    - A white arm marker at depth 1 means normal launch; at depth 2 that arm first steps one more cell outward before joining the spiral.
    - When a turtle lands on red, its next heading becomes the left turn of the current move; when it lands on green, its next heading becomes the right turn.
    - The spacing rule and 2/1 step rhythm from the seed task still prevent adjacent teal walls from collapsing together.

    Key insight: The final maze is not controlled by one cue anymore. You must read the border legend for global handedness, read each arm marker for its local start offset, and then follow the colored obstacle cells that inject left/right turns along the route.
    """

    return _solve(grid, want_stats=False)


def _candidate_obstacle_cells(base_grid: np.ndarray) -> tuple[np.ndarray, list[list[tuple[int, int]]]] | None:
    out, stats = _solve(base_grid, want_stats=True)
    per_arm = []
    for seq in stats["visited"]:
        if len(seq) < 5:
            return None
        lo = 2
        hi = min(len(seq) - 2, 7)
        if lo > hi:
            return None
        per_arm.append(seq[lo : hi + 1])
    return out, per_arm


def _gen_attempt() -> np.ndarray | None:
    arm = random.randint(ARM_LEN_MIN, ARM_LEN_MAX)
    size = 4 * arm + 2 * math.ceil(arm / 2) + GRID_SIZE_BASE
    lo = 2 * arm + TIP_GAP_MIN
    hi = size - 2 * arm - TIP_GAP_MIN - 1
    if lo > hi:
        return None

    centers = [(r, c) for r in range(lo, hi + 1) for c in range(lo, hi + 1)]
    random.shuffle(centers)

    for cr, cc in centers:
        spin = random.choice(["CW", "CCW"])
        pads = [0, 0, 1, 1]
        random.shuffle(pads)

        base = _build_input(arm, size, cr, cc)
        _place_legend(base, spin)
        if not _place_arm_markers(base, cr, cc, pads):
            continue

        base_out_and_cells = _candidate_obstacle_cells(base)
        if base_out_and_cells is None:
            continue
        out_no_obs, per_arm_candidates = base_out_and_cells

        marked = base.copy()
        colors = [LEFT_OBSTACLE, RIGHT_OBSTACLE, LEFT_OBSTACLE, RIGHT_OBSTACLE]
        random.shuffle(colors)
        placed = []
        ok = True
        for cells, col in zip(per_arm_candidates, colors, strict=False):
            cell = random.choice(cells)
            if marked[cell] != 0:
                ok = False
                break
            marked[cell] = col
            placed.append(cell)
        if not ok:
            continue

        try:
            out = main(marked)
        except Exception:
            continue

        fill = float(np.mean(out == CROSS_COLOR))
        if not (FILL_LO <= fill <= FILL_HI):
            continue

        metrics = _behavior_metrics(marked)
        if metrics["used_left"] < 1 or metrics["used_right"] < 1:
            continue
        if metrics["diff_no_obstacles"] < MIN_OUTPUT_DIFF:
            continue
        if metrics["diff_flip_legend"] < MIN_OUTPUT_DIFF:
            continue
        if metrics["diff_normalized_pads"] < MIN_ARM_DIFF:
            continue
        if np.sum(out != out_no_obs) < MIN_OUTPUT_DIFF:
            continue

        if np.array_equal(main(marked), out):
            return marked
    return None


def generate_input() -> np.ndarray:
    for _ in range(MAX_OUTER_RETRIES):
        grid = _gen_attempt()
        if grid is not None:
            return grid
    raise RuntimeError("generate_input: failed after retries")
