# Combo variant of da515329 (index 16)
# combo_spec: D9/0+D16/0
# Name: 十字臂擦除 + 直接前进无回退

# Design analysis for combo D9/0 + D16/0
#
# a) Concrete behavior trace on a small arm=2 case.
#    - Original code copies the whole input cross into `out`, so the inward cell behind each tip is
#      occupied from the start. For the top tip, the spiral head begins at the tip with heading
#      "up". The first legal action is therefore not an inward retreat; it is a right turn onto the
#      row beside the still-visible cross, and the cross remains embedded in the final pattern.
#    - Fused variant: erase the whole cross from the visible canvas, but keep each tip as a
#      no-return planning anchor. Every tip first paints exactly one cell farther outward, then the
#      normal clockwise priority takes over. After the spiral finishes, every original cross cell is
#      cleared again, so the output contains no intact cross arm at all.
#    - Difference: the four cells just beyond the tips are now guaranteed to be painted, while the
#      original arm cells themselves are guaranteed to be background in the final output.
#
# b) Complexity calibration.
#    - If arm_len=1, the combo becomes visually weak: erasing a one-pixel arm changes only a few
#      cells and can look too close to the source task.
#    - If arm_len=3, the valid placements are brittle under the fused geometry.
#    - The generator therefore uses arm lengths 2 and 4 only, with several grid sizes each. That
#      keeps repeated support for the rule while preserving clear visual differences.
#
# c) Implicit-assumption audit of the original solver.
#    - `out = grid.copy()` assumed the cross was part of the wall system. This is broken by D9/0,
#      so the combo starts from a blank canvas.
#    - The original first-move geometry relied on the copied arm to block inward motion. After D9/0
#      that blocker disappears, so D16/0 must add an explicit outward-first phase.
#    - The parity and border cleanup still make sense, but they must run before the final arm erase:
#      they tidy the spiral, not the input clue.
#
# d) Placement legality.
#    - The generator enumerates only centers where every arm fits in bounds and each tip has at
#      least one outward cell available.
#    - It keeps only placements whose solved output passes fill, forward-step, and baseline-diff
#      checks, so no accepted sample can be degenerate or physically invalid.
#
# e) Visual observability.
#    - Humans can see the combo because two strong cues happen together:
#      1. the input cross vanishes completely in the output;
#      2. each tip immediately sprouts one outward pixel before the spiral turns.
#    - The generator rejects placements where the combo would be too close to the original output.
#
# f) Visual cleanliness.
#    - There is still only one object family: one teal cross in the input and one teal spiral field
#      in the output. No extra colors or auxiliary markers are introduced.
#    - The accepted size/center families produce medium-density spiral bands rather than tiny local
#      tangles or nearly empty canvases.
#
# g) Change inferability.
#    - Color semantics stay unchanged: teal remains the only foreground color.
#    - All variation is geometric and repeated four times symmetrically, which is easy to infer from
#      2-4 ARC examples without hidden metadata.

import random

import numpy as np

from arc_utils import neighbors4

_DIRS_URDL = [(-1, 0), (0, 1), (1, 0), (0, -1)]
_CROSS_COLOR = 8


def _right(d: tuple[int, int]) -> tuple[int, int]:
    return _DIRS_URDL[(_DIRS_URDL.index(d) + 1) % 4]


def _left(d: tuple[int, int]) -> tuple[int, int]:
    return _DIRS_URDL[(_DIRS_URDL.index(d) + 3) % 4]


def _find_cross(grid: np.ndarray) -> tuple[int, int, list[tuple[tuple[int, int], list[tuple[int, int]]]]]:
    row_counts = np.sum(grid == _CROSS_COLOR, axis=1)
    col_counts = np.sum(grid == _CROSS_COLOR, axis=0)
    cr = int(np.argmax(row_counts))
    cc = int(np.argmax(col_counts))
    h, w = grid.shape

    def inb(r: int, c: int) -> bool:
        return 0 <= r < h and 0 <= c < w

    arms: list[tuple[tuple[int, int], list[tuple[int, int]]]] = []
    for d in _DIRS_URDL:
        dr, dc = d
        r, c = cr, cc
        cells = []
        while inb(r + dr, c + dc) and grid[r + dr, c + dc] == _CROSS_COLOR:
            r += dr
            c += dc
            cells.append((r, c))
        if cells:
            arms.append((d, cells))
    return cr, cc, arms


def _apply_postprocess(
    out: np.ndarray,
    cr: int,
    cc: int,
    arm_len: int,
) -> None:
    h, w = out.shape

    def inb(r: int, c: int) -> bool:
        return 0 <= r < h and 0 <= c < w

    def is_border(r: int, c: int) -> bool:
        return r == 0 or r == h - 1 or c == 0 or c == w - 1

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
                if out[r, c] != _CROSS_COLOR or not is_border(r, c):
                    continue
                deg = 0
                nbr = None
                for nr, nc in neighbors4(r, c):
                    if inb(nr, nc) and out[nr, nc] == _CROSS_COLOR:
                        deg += 1
                        nbr = (nr, nc)
                if deg == 1 and nbr is not None and is_border(*nbr):
                    to_clear.append((r, c))
        for r, c in to_clear:
            out[r, c] = 0

        br = (h - 1, w - 1)
        if out[br] == 0 and all(
            not (inb(nr, nc) and out[nr, nc] == _CROSS_COLOR)
            for nr, nc in neighbors4(br[0], br[1])
        ):
            out[br] = _CROSS_COLOR
    else:
        if arm_len >= 2:
            out[0 : min(h, arm_len - 1), 0] = _CROSS_COLOR
        if arm_len >= 1:
            out[0, max(0, w - arm_len) : w] = _CROSS_COLOR
        if h >= 2:
            for c in range(w):
                if out[h - 1, c] == _CROSS_COLOR and out[h - 2, c] == 0:
                    out[h - 1, c] = 0


def _solve_original_reference(grid: np.ndarray) -> np.ndarray:
    grid = np.asarray(grid, dtype=int)
    h, w = grid.shape
    out = grid.copy()
    cr, cc, arms = _find_cross(grid)
    arm_len = max((len(cells) for _, cells in arms), default=0)

    def inb(r: int, c: int) -> bool:
        return 0 <= r < h and 0 <= c < w

    def is_border(r: int, c: int) -> bool:
        return r == 0 or r == h - 1 or c == 0 or c == w - 1

    def spacing_ok(cell: tuple[int, int], prev: tuple[int, int], start: tuple[int, int]) -> bool:
        r, c = cell
        for nr, nc in neighbors4(r, c):
            if inb(nr, nc) and out[nr, nc] != 0 and (nr, nc) != prev and (nr, nc) != start:
                return False
        return True

    def can_move(cur: tuple[int, int], d: tuple[int, int], length: int) -> bool:
        prev = cur
        for t in range(1, length + 1):
            rr = cur[0] + d[0] * t
            cc2 = cur[1] + d[1] * t
            if not inb(rr, cc2) or out[rr, cc2] != 0:
                return False
            if not spacing_ok((rr, cc2), prev, cur):
                return False
            prev = (rr, cc2)
        return True

    def do_move(agent: dict, d: tuple[int, int], length: int) -> None:
        r0, c0 = agent["cur"]
        for t in range(1, length + 1):
            out[r0 + d[0] * t, c0 + d[1] * t] = _CROSS_COLOR
        agent["cur"] = (r0 + d[0] * length, c0 + d[1] * length)

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
        for ag in agents:
            if step_agent(ag):
                changed = True
        if all(ag["done"] for ag in agents):
            break

    _apply_postprocess(out, cr, cc, arm_len)
    return out


def main(grid: np.ndarray) -> np.ndarray:
    """
    Erase a teal cross and regrow it as four outward-first clockwise spirals.

    Input structure:
    - bg = 0 (black). A single teal (8) plus sign with a center hole is present.
    - The four arms are equal length and define four cardinal tip anchors.

    Rule:
    - Clear the visible cross, but keep each former tip as a no-return anchor for planning.
    - Every anchor first advances one cell farther outward, then follows the same clockwise wall
      growth policy: prefer right turn > straight > left, use 2-cell straight moves except for the
      1-cell turn / post-turn rhythm, and maintain a one-cell orthogonal gap from other walls.
    - After the spiral field is completed and parity cleanup is applied, erase every original cross
      cell so the input cross leaves no visible residue in the output.

    Key insight:
    - D9/0 removes the cross walls that originally shaped the launch geometry, so D16/0 must supply
      an explicit outward-first step; otherwise the erased tips would retreat inward and the spiral
      would collapse into a sparse local knot.
    """

    grid = np.asarray(grid, dtype=int)
    h, w = grid.shape
    out = np.zeros_like(grid)
    cr, cc, arms = _find_cross(grid)
    arm_len = max((len(cells) for _, cells in arms), default=0)

    def inb(r: int, c: int) -> bool:
        return 0 <= r < h and 0 <= c < w

    def is_border(r: int, c: int) -> bool:
        return r == 0 or r == h - 1 or c == 0 or c == w - 1

    def spacing_ok(cell: tuple[int, int], prev: tuple[int, int], start: tuple[int, int]) -> bool:
        r, c = cell
        for nr, nc in neighbors4(r, c):
            if inb(nr, nc) and out[nr, nc] != 0 and (nr, nc) != prev and (nr, nc) != start:
                return False
        return True

    def can_move(cur: tuple[int, int], d: tuple[int, int], length: int) -> bool:
        prev = cur
        for t in range(1, length + 1):
            rr = cur[0] + d[0] * t
            cc2 = cur[1] + d[1] * t
            if not inb(rr, cc2) or out[rr, cc2] != 0:
                return False
            if not spacing_ok((rr, cc2), prev, cur):
                return False
            prev = (rr, cc2)
        return True

    def do_move(agent: dict, d: tuple[int, int], length: int) -> None:
        r0, c0 = agent["cur"]
        for t in range(1, length + 1):
            out[r0 + d[0] * t, c0 + d[1] * t] = _CROSS_COLOR
        agent["cur"] = (r0 + d[0] * length, c0 + d[1] * length)

    agents = []
    for outward, cells in arms:
        tip = cells[-1]
        # The tip is a planning anchor: paths may start from it but may not step back through it.
        out[tip] = _CROSS_COLOR
        agents.append(
            {
                "cur": tip,
                "dir": outward,
                "out_dir": outward,
                "phase": "forward_first",
                "done": False,
                "p": 0,
            }
        )

    def step_agent(agent: dict) -> bool:
        if agent["done"]:
            return False

        if agent["phase"] == "forward_first":
            if can_move(agent["cur"], agent["out_dir"], 1):
                do_move(agent, agent["out_dir"], 1)
            agent["phase"] = None
            agent["dir"] = agent["out_dir"]
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

    _apply_postprocess(out, cr, cc, arm_len)
    out[grid == _CROSS_COLOR] = 0
    return out


# --------------- generator (combo D9/0 + D16/0) ---------------

_ARM_SIZE_CHOICES = (
    (2, 10),
    (2, 11),
    (2, 12),
    (2, 13),
    (2, 14),
    (4, 13),
    (4, 14),
    (4, 15),
    (4, 16),
)
_FILL_LO = 0.33
_FILL_HI = 0.50
_DIFF_MIN = 45

_VALID_BUCKETS: list[tuple[int, int, list[tuple[int, int]]]] | None = None


def _build_input(arm: int, size: int, cr: int, cc: int) -> np.ndarray:
    grid = np.zeros((size, size), dtype=int)
    offsets = np.arange(1, arm + 1)
    grid[cr - offsets, cc] = _CROSS_COLOR
    grid[cr + offsets, cc] = _CROSS_COLOR
    grid[cr, cc - offsets] = _CROSS_COLOR
    grid[cr, cc + offsets] = _CROSS_COLOR
    return grid


def _combo_metrics(inp: np.ndarray, out: np.ndarray) -> dict[str, int | float]:
    cr, cc, arms = _find_cross(inp)
    del cr, cc

    forward_hits = 0
    for outward, cells in arms:
        tip = cells[-1]
        rr = tip[0] + outward[0]
        cc2 = tip[1] + outward[1]
        if 0 <= rr < out.shape[0] and 0 <= cc2 < out.shape[1] and out[rr, cc2] == _CROSS_COLOR:
            forward_hits += 1

    return {
        "fill": float(np.mean(out == _CROSS_COLOR)),
        "forward_hits": forward_hits,
        "cleared_cross": int(np.sum((inp == _CROSS_COLOR) & (out == 0))),
    }


def _build_valid_buckets() -> list[tuple[int, int, list[tuple[int, int]]]]:
    buckets: list[tuple[int, int, list[tuple[int, int]]]] = []
    for arm, size in _ARM_SIZE_CHOICES:
        centers: list[tuple[int, int]] = []
        lo = arm + 1
        hi = size - arm - 2
        for cr in range(lo, hi + 1):
            for cc in range(lo, hi + 1):
                inp = _build_input(arm, size, cr, cc)
                out = main(inp)
                ref = _solve_original_reference(inp)
                metrics = _combo_metrics(inp, out)
                diff = int(np.sum(out != ref))
                if not (_FILL_LO <= float(metrics["fill"]) <= _FILL_HI):
                    continue
                if int(metrics["forward_hits"]) != 4:
                    continue
                if int(metrics["cleared_cross"]) != 4 * arm:
                    continue
                if diff < _DIFF_MIN:
                    continue
                centers.append((cr, cc))
        if centers:
            buckets.append((arm, size, centers))
    if not buckets:
        raise RuntimeError("no valid combo buckets found for da515329 D9/0+D16/0")
    return buckets


def generate_input() -> np.ndarray:
    global _VALID_BUCKETS
    if _VALID_BUCKETS is None:
        _VALID_BUCKETS = _build_valid_buckets()

    arm, size, centers = random.choice(_VALID_BUCKETS)
    cr, cc = random.choice(centers)
    inp = _build_input(arm, size, cr, cc)

    out = main(inp)
    ref = _solve_original_reference(inp)
    metrics = _combo_metrics(inp, out)
    diff = int(np.sum(out != ref))
    if not (_FILL_LO <= float(metrics["fill"]) <= _FILL_HI):
        raise AssertionError("generated fill fell outside the combo acceptance band")
    if int(metrics["forward_hits"]) != 4:
        raise AssertionError("generated sample lost the direct-forward tip signature")
    if int(metrics["cleared_cross"]) != 4 * arm:
        raise AssertionError("generated sample still preserves visible cross pixels")
    if diff < _DIFF_MIN:
        raise AssertionError("generated sample is too close to the original task behavior")

    return inp
