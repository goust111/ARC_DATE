# Combo variant of da515329 (index 38)
# combo_spec: D4/1+D9/2
# Name: 十字臂分裂 + 非对称十字

import math
import random

import numpy as np

from arc_utils import neighbors4

_DIRS_URDL = [(-1, 0), (0, 1), (1, 0), (0, -1)]
CROSS_COLOR = 8
ARM_LENGTH_CHOICES = (1, 2, 3, 4)
GRID_EXTRA_CHOICES = (8, 10)
TIP_GAP_MIN = 2
FILL_LO, FILL_HI = 0.40, 0.62
MAX_OUTER_RETRIES = 400


# Design analysis for the fused variant
# a) Concrete behavior trace:
#    - Example input: the cross has lengths U=2, R=4, D=3, L=1 around one empty center.
#    - Original task: the four tips launch clockwise spirals; because all arms were equal, one
#      scalar `arm_len` governed both generation and post-cleanup.
#    - Fused task:
#      1. The input itself is now asymmetric, so each spiral starts from a different tip distance.
#      2. After the spiral field is drawn, the source cross is kept in place as the "hollow" part:
#         it remains a one-cell-wide emitter around the empty center instead of collapsing into a
#         solid shape.
#      3. Each arm additionally spawns one solid square whose side length equals that arm length.
#         That square is moved to the corresponding spiral endpoint and attached beside that end.
#      4. Thus arm lengths are visible twice: near the center as unequal arm extents, and near the
#         periphery as four different terminal square sizes.
# b) Complexity calibration:
#    - The generator always uses the same multiset of arm lengths {1,2,3,4}, only permuted across
#      directions. This keeps the hypothesis space small: the solver must infer one consistent
#      size-transfer rule instead of arbitrary continuous parameters.
#    - The split effect is visually strong because the output gains four terminal solids with
#      repeated size evidence, not a subtle parity tweak.
# c) Implicit-assumption audit against the original solver:
#    - "All arms share one arm_len" is false now. The solver therefore tracks `length` per agent.
#    - "Only the max arm length matters for input parsing" is false for the transform layer.
#      Endpoint blocks must use per-arm lengths, not the global maximum.
#    - "A moved piece can be stamped on the endpoint cell itself" fails for length-1 arms because
#      that would be invisible. The fused version therefore places each block in an adjacent square
#      attached to the endpoint, maximizing newly added foreground pixels.
#    - The original parity/border cleanup is still useful for the spiral field, but it is keyed by
#      `max_arm` only after spiral growth; the later split layer is independent of that cleanup.
# d) Placement legality:
#    - Inputs contain exactly one cross, all four cardinal arms present, no overlaps by
#      construction, and all tips remain well inside the border search band.
#    - Output blocks are chosen from in-bounds adjacent LxL candidates. Generation only accepts
#      examples where every block adds visible new cells and block boxes do not overlap.
# e) Visual observability / cleanliness / inferability:
#    - A human sees one asymmetric cross in the input and, in the output, the same cross plus four
#      differently sized terminal squares. The repeated size transfer from center to periphery is
#      observable across 2-4 examples without hidden ordering tricks.
#    - Only one color is used, so the transform is conveyed by shape, size, and destination, not by
#      arbitrary recoloring.
# f) New assumptions guaranteed by the generator:
#    - The center row/column remain the unique maxima of teal counts, so center detection by row/col
#      argmax is stable.
#    - All arm lengths are distinct and all four directions are populated.
#    - The chosen center makes the spiral solver deterministic and leaves enough local room that
#      every endpoint block has at least one visible adjacent placement.


def _right(d):
    return _DIRS_URDL[(_DIRS_URDL.index(d) + 1) % 4]


def _left(d):
    return _DIRS_URDL[(_DIRS_URDL.index(d) + 3) % 4]


def _inb(h: int, w: int, r: int, c: int) -> bool:
    return 0 <= r < h and 0 <= c < w


def _is_border(h: int, w: int, r: int, c: int) -> bool:
    return r == 0 or r == h - 1 or c == 0 or c == w - 1


def _add(pos, d, k=1):
    return (pos[0] + d[0] * k, pos[1] + d[1] * k)


def _detect_cross(grid: np.ndarray):
    row_counts = np.sum(grid == CROSS_COLOR, axis=1)
    col_counts = np.sum(grid == CROSS_COLOR, axis=0)
    cr = int(np.argmax(row_counts))
    cc = int(np.argmax(col_counts))

    h, w = grid.shape
    arms = []
    for outward in _DIRS_URDL:
        dr, dc = outward
        r, c = cr, cc
        cells = []
        while _inb(h, w, r + dr, c + dc) and grid[r + dr, c + dc] == CROSS_COLOR:
            r += dr
            c += dc
            cells.append((r, c))
        if not cells:
            raise ValueError("expected a full four-armed cross")
        arms.append(
            {
                "outward": outward,
                "cells": cells,
                "length": len(cells),
                "tip": cells[-1],
            }
        )
    return cr, cc, arms


def _solve_spirals_only(grid: np.ndarray):
    grid = np.asarray(grid, dtype=int)
    h, w = grid.shape
    out = grid.copy()
    cr, cc, arms = _detect_cross(grid)

    def spacing_ok(cell, prev, start) -> bool:
        r, c = cell
        for nr, nc in neighbors4(r, c):
            if _inb(h, w, nr, nc) and out[nr, nc] != 0 and (nr, nc) != prev and (nr, nc) != start:
                return False
        return True

    def can_move(cur, d, length: int) -> bool:
        prev = cur
        for t in range(1, length + 1):
            rr, cc2 = _add(cur, d, t)
            if not _inb(h, w, rr, cc2) or out[rr, cc2] != 0:
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
        agent["last_dir"] = d

    agents = []
    for arm in arms:
        outward = arm["outward"]
        cells = arm["cells"]
        agents.append(
            {
                "cur": cells[-1],
                "dir": _right(outward),
                "out_dir": outward,
                "phase": "out_first" if len(cells) == 1 else None,
                "done": False,
                "p": 0,
                "length": len(cells),
                "tip": cells[-1],
                "last_dir": None,
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
        if not _is_border(h, w, *cur):
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
    max_arm = max(agent["length"] for agent in agents)

    if not even_center:
        rp = 1 - (cr % 2)
        cp = 1 - (cc % 2)
        rr, ccg = np.indices((h, w))
        out[(rr % 2 == rp) & (ccg % 2 == cp)] = 0
    elif max_arm % 2 == 1:
        to_clear = []
        for r in range(h):
            for c in range(w):
                if out[r, c] != CROSS_COLOR or not _is_border(h, w, r, c):
                    continue
                deg = 0
                nbr = None
                for nr, nc in neighbors4(r, c):
                    if _inb(h, w, nr, nc) and out[nr, nc] == CROSS_COLOR:
                        deg += 1
                        nbr = (nr, nc)
                if deg == 1 and nbr is not None and _is_border(h, w, *nbr):
                    to_clear.append((r, c))
        for r, c in to_clear:
            out[r, c] = 0

        br = (h - 1, w - 1)
        if out[br] == 0 and all(
            not (_inb(h, w, nr, nc) and out[nr, nc] == CROSS_COLOR)
            for nr, nc in neighbors4(br[0], br[1])
        ):
            out[br] = CROSS_COLOR
    else:
        if max_arm >= 2:
            out[0 : min(h, max_arm - 1), 0] = CROSS_COLOR
        if max_arm >= 1:
            out[0, max(0, w - max_arm) : w] = CROSS_COLOR
        if h >= 2:
            for c in range(w):
                if out[h - 1, c] == CROSS_COLOR and out[h - 2, c] == 0:
                    out[h - 1, c] = 0

    meta = {
        "center": (cr, cc),
        "arms": arms,
        "agents": agents,
        "max_arm": max_arm,
        "even_center": even_center,
    }
    return out, meta


def _adjacent_square_candidates(anchor: tuple[int, int], side: int, h: int, w: int):
    r, c = anchor
    boxes = set()

    for k in range(side):
        r0 = r - k
        r1 = r0 + side - 1
        c0 = c - side
        c1 = c - 1
        if _inb(h, w, r0, c0) and _inb(h, w, r1, c1):
            boxes.add((r0, c0, r1, c1))

        r0 = r - k
        r1 = r0 + side - 1
        c0 = c + 1
        c1 = c + side
        if _inb(h, w, r0, c0) and _inb(h, w, r1, c1):
            boxes.add((r0, c0, r1, c1))

        r0 = r - side
        r1 = r - 1
        c0 = c - k
        c1 = c0 + side - 1
        if _inb(h, w, r0, c0) and _inb(h, w, r1, c1):
            boxes.add((r0, c0, r1, c1))

        r0 = r + 1
        r1 = r + side
        c0 = c - k
        c1 = c0 + side - 1
        if _inb(h, w, r0, c0) and _inb(h, w, r1, c1):
            boxes.add((r0, c0, r1, c1))

    return sorted(boxes)


def _place_endpoint_blocks(base: np.ndarray, agents: list[dict]):
    h, w = base.shape
    out = base.copy()
    used = np.zeros((h, w), dtype=bool)
    infos = []

    order = sorted(agents, key=lambda ag: (-ag["length"], _DIRS_URDL.index(ag["out_dir"])))
    for ag in order:
        side = ag["length"]
        candidates = _adjacent_square_candidates(ag["cur"], side, h, w)
        if not candidates:
            raise ValueError("no legal endpoint block placement")

        best_box = None
        best_key = None
        best_visible = None
        for box in candidates:
            r0, c0, r1, c1 = box
            overlap = int(np.sum(used[r0 : r1 + 1, c0 : c1 + 1]))
            sub = base[r0 : r1 + 1, c0 : c1 + 1]
            visible = int(np.sum(sub == 0))
            occupied = int(np.sum(sub != 0))
            key = (overlap == 0, visible, -overlap, -occupied, -r0, -c0)
            if best_key is None or key > best_key:
                best_key = key
                best_box = box
                best_visible = visible

        r0, c0, r1, c1 = best_box
        out[r0 : r1 + 1, c0 : c1 + 1] = CROSS_COLOR
        used[r0 : r1 + 1, c0 : c1 + 1] = True
        infos.append(
            {
                "out_dir": ag["out_dir"],
                "side": side,
                "anchor": ag["cur"],
                "box": best_box,
                "visible_gain": best_visible,
            }
        )

    return out, infos


def _render_variant(grid: np.ndarray, *, return_meta: bool = False):
    base, meta = _solve_spirals_only(grid)
    out, blocks = _place_endpoint_blocks(base, meta["agents"])
    if return_meta:
        fused_meta = dict(meta)
        fused_meta["base_output"] = base
        fused_meta["blocks"] = blocks
        return out, fused_meta
    return out


def main(grid: np.ndarray) -> np.ndarray:
    """
    Grow four clockwise spiral walls from an asymmetric teal cross, then split each arm into a preserved emitter and a terminal square.

    Input structure:
    - bg = 0 (black). A single teal (8) cross with an empty center cell.
    - The four cardinal arms are all present but have pairwise different lengths.

    Rule:
    - Each arm tip launches the same clockwise spiral walker as in the base task: turn-right priority,
      then straight, then left when not on the border; straight steps prefer length 2, but the first
      step after a turn uses length 1.
    - The original thin cross stays in place as the hollow source object.
    - After the spiral field is completed, each arm contributes one solid LxL square, where L is that
      arm's own length; the square is attached beside the corresponding spiral endpoint.

    Key insight: The asymmetry is not decorative; each arm length reappears at a distant endpoint, so
    the output is the original spiral tiling plus a four-way size transfer from the central cross.
    """

    return _render_variant(grid)


def _build_input(lengths_by_dir: dict[tuple[int, int], int], size: int, cr: int, cc: int) -> np.ndarray:
    grid = np.zeros((size, size), dtype=int)
    for (dr, dc), arm_len in lengths_by_dir.items():
        for k in range(1, arm_len + 1):
            grid[cr + dr * k, cc + dc * k] = CROSS_COLOR
    return grid


def _visible_requirement(side: int) -> int:
    if side == 1:
        return 1
    return max(2, (side * side + 3) // 4)


def _gen_attempt() -> np.ndarray | None:
    lengths = list(ARM_LENGTH_CHOICES)
    random.shuffle(lengths)
    lengths_by_dir = {d: lengths[i] for i, d in enumerate(_DIRS_URDL)}

    max_arm = max(lengths)
    size = 4 * max_arm + 2 * math.ceil(max_arm / 2) + random.choice(GRID_EXTRA_CHOICES)
    lo = 2 * max_arm + TIP_GAP_MIN
    hi = size - 2 * max_arm - TIP_GAP_MIN - 1
    if lo > hi:
        return None

    candidates = [(r, c) for r in range(lo, hi + 1) for c in range(lo, hi + 1)]
    random.shuffle(candidates)

    for cr, cc in candidates:
        inp = _build_input(lengths_by_dir, size, cr, cc)
        try:
            out, meta = _render_variant(inp, return_meta=True)
        except Exception:
            continue

        base = meta["base_output"]
        if np.array_equal(out, base):
            continue
        if not np.array_equal(main(inp), out):
            continue

        fill = float(np.mean(out == CROSS_COLOR))
        if not (FILL_LO <= fill <= FILL_HI):
            continue

        block_sides = sorted(info["side"] for info in meta["blocks"])
        if block_sides != sorted(lengths):
            continue

        if any(info["visible_gain"] < _visible_requirement(info["side"]) for info in meta["blocks"]):
            continue

        changed_pixels = int(np.sum(out != base))
        if changed_pixels < sum(lengths):
            continue

        boxes = [info["box"] for info in meta["blocks"]]
        overlap = False
        for i, (r0, c0, r1, c1) in enumerate(boxes):
            for j in range(i + 1, len(boxes)):
                rr0, cc0, rr1, cc1 = boxes[j]
                if not (r1 < rr0 or rr1 < r0 or c1 < cc0 or cc1 < c0):
                    overlap = True
                    break
            if overlap:
                break
        if overlap:
            continue

        return inp
    return None


def generate_input() -> np.ndarray:
    for _ in range(MAX_OUTER_RETRIES):
        g = _gen_attempt()
        if g is not None:
            return g
    raise RuntimeError("generate_input: failed after retries")
