# Combo variant of da515329 (index 34)
# combo_spec: D3/3+D11/1+D14/1
# Name: 逐圈交替展开 + 转弯变色 + 四色象限

# Design analysis for the combo variant
# a) Concrete behavior and trace
#    - Input stays in the original family: one black square grid with a single teal plus-sign cross,
#      equal arm lengths, and an empty center cell.
#    - Output geometry still comes from four clockwise self-avoiding spirals launched from the four arm tips,
#      but the execution order changes: after all four tips get the same launch opportunity, the upper, right,
#      lower, and left spirals take turns by segment. On its turn, one spiral keeps advancing until it
#      completes the next corner event, then control passes to the next spiral. In this geometry family the
#      path skeleton stays the same, so the fixed round-robin order is made visible by also choosing which
#      member of each two-color pair appears on that spiral's launch segment.
#    - Each spiral owns a fixed two-color palette tied to its launch direction:
#      up=(2,6), right=(1,7), down=(3,9), left=(4,5). Straight continuation keeps the current color;
#      the exact move that bends 90 degrees switches to the other color in that pair.
#    - Small trace: the top tip first gets the same launch privilege as the other three tips. Because it is
#      first in the round-robin order, its launch segment uses the first shade of its pair, red(2). Later,
#      during the segment-round-robin phase, it may continue east for several straight advances. When it can
#      finally turn south, that turning move is painted in orange(6), and the scheduler immediately yields to
#      the right-tip spiral. The next time the top spiral is scheduled, it continues south in orange until its
#      next bend flips it back to red.
#    - Compared with the base task, there are now two coupled differences at every corner: a local color flip
#      and a global handoff to the next spiral. Compared with the single-dimension color variants, the output
#      also reveals which quadrant owns which pair of colors.
# b) Complexity calibration
#    - The palette uses exactly four direction-owned pairs, so the color hypothesis space is structured rather
#      than arbitrary. Every accepted sample shows the same pair mapping multiple times.
#    - The generator filters for multiple turns per spiral and for a nontrivial visible delta against the same
#      segment solver with round-robin launch phasing disabled. This keeps the fused behavior visible instead
#      of collapsing back toward a plain four-pair recoloring.
# c) Implicit assumption review
#    - Original occupancy tests that looked for literal 8 would fail once path cells use eight different
#      colors; all geometric legality checks must use nonzero occupancy instead.
#    - Original post-processing wrote hard-coded teal border cells. That would break both the quadrant-owner
#      rule and the turn-driven recoloring rule, so added cells now inherit a nearby spiral owner and a locally
#      compatible color.
#    - Original simultaneous stepping assumed that scheduler choice did not matter. In this geometry family the
#      segment scheduler alone leaves the occupied skeleton unchanged, so the combo makes D11 observable by
#      tying each spiral's initial color phase to its fixed round-robin slot. The solver explicitly models that
#      coupling, and the generator compares it against the same solver with slot-phase disabled to ensure the
#      D11 control variable is active in visible pixels.
#    - Center detection still assumes the input foreground is only the teal cross. The generator preserves that.
# d) Placement legality
#    - The generator keeps the original size formula and center-gap constraints from the spec, so all four
#      launched spirals fit inside the square and retain the one-cell orthogonal spacing rule.
#    - Accepted inputs are solved before use; failed placements, low-fill placements, or placements where some
#      spiral pair is missing one of its two colors are discarded.
# e) Visual observability
#    - A human can read the rule from 2-4 examples because three independent cues line up:
#      endpoint direction -> color pair, corner location -> within-pair flip, and round-robin slot -> which
#      shade of that pair appears first on the launch segment. None of these cues depends on hidden numeric
#      ordering of colors.
# f) Visual cleanliness
#    - There is no overlap layer or marker clutter: each cell still belongs to at most one spiral path, and the
#      original teal cross remains a distinct seed object. The eight path colors are partitioned by quadrant,
#      which keeps the denser output readable.
# g) Change inferability and new assumptions
#    - Although the combo uses many colors, they are not free-floating: each output consistently reuses four
#      visible two-color pairs attached to the same four launch directions, and the flips always occur exactly
#      at bends. That repeated structure makes the recoloring inferable.
#    - New solver assumptions checked by generate_input():
#      the input contains exactly one equal-arm teal cross, all four arms are present, each spiral gets enough
#      room to make multiple turns, and the slot-phased round-robin coloring differs measurably from the same
#      segment solver with slot-phase disabled.

import math
import random

import numpy as np

from arc_utils import neighbors4

_DIRS_URDL = [(-1, 0), (0, 1), (1, 0), (0, -1)]
_DIR_TO_PAIR = {
    (-1, 0): (2, 6),  # up tip
    (0, 1): (1, 7),   # right tip
    (1, 0): (3, 9),   # down tip
    (0, -1): (4, 5),  # left tip
}

BG = 0
CROSS_COLOR = 8
BG_OWNER = -2
CROSS_OWNER = -1
SPIRAL_PAIRS = tuple(_DIR_TO_PAIR[d] for d in _DIRS_URDL)
SPIRAL_COLORS = tuple(color for pair in SPIRAL_PAIRS for color in pair)

# --------------- generator (cx) ---------------

# --- CFG ---------------------------------------------------------------
ARM_LEN_MIN, ARM_LEN_MAX = 1, 4
GRID_SIZE_BASE = 6
TIP_GAP_MIN = 2
FILL_LO, FILL_HI = 0.48, 0.52
MAX_OUTER_RETRIES = 400
MIN_TURNS_PER_AGENT = 2
MIN_ALT_PIXELS_PER_AGENT = 3
MIN_PRIMARY_PIXELS_PER_AGENT = 3
MIN_TOTAL_TURNS = 12
MIN_PHASE_DELTA = 10
# --- end CFG -----------------------------------------------------------


def _other_index(idx: int) -> int:
    return 1 - idx


def _solve_with_stats(
    grid: np.ndarray,
    schedule_mode: str = "segment",
    slot_phase: bool = True,
) -> tuple[np.ndarray, dict[str, object]]:
    grid = np.asarray(grid, dtype=int)
    h, w = grid.shape
    out = grid.copy()
    owner = np.full((h, w), BG_OWNER, dtype=int)
    owner[grid == CROSS_COLOR] = CROSS_OWNER

    row_counts = np.sum(grid == CROSS_COLOR, axis=1)
    col_counts = np.sum(grid == CROSS_COLOR, axis=0)
    cr = int(np.argmax(row_counts))
    cc = int(np.argmax(col_counts))

    def inb(r: int, c: int) -> bool:
        return 0 <= r < h and 0 <= c < w

    def is_border(r: int, c: int) -> bool:
        return r == 0 or r == h - 1 or c == 0 or c == w - 1

    def add(pos: tuple[int, int], d: tuple[int, int], k: int = 1) -> tuple[int, int]:
        return (pos[0] + d[0] * k, pos[1] + d[1] * k)

    def right(d: tuple[int, int]) -> tuple[int, int]:
        return _DIRS_URDL[(_DIRS_URDL.index(d) + 1) % 4]

    def left(d: tuple[int, int]) -> tuple[int, int]:
        return _DIRS_URDL[(_DIRS_URDL.index(d) + 3) % 4]

    def erase_cell(r: int, c: int) -> None:
        out[r, c] = BG
        owner[r, c] = BG_OWNER

    def spacing_ok(
        cell: tuple[int, int],
        prev: tuple[int, int],
        start: tuple[int, int],
    ) -> bool:
        r, c = cell
        for nr, nc in neighbors4(r, c):
            if inb(nr, nc) and out[nr, nc] != BG and (nr, nc) != prev and (nr, nc) != start:
                return False
        return True

    def can_move(cur: tuple[int, int], d: tuple[int, int], length: int) -> bool:
        prev = cur
        for t in range(1, length + 1):
            rr, cc2 = add(cur, d, t)
            if not inb(rr, cc2) or out[rr, cc2] != BG:
                return False
            if not spacing_ok((rr, cc2), prev, cur):
                return False
            prev = (rr, cc2)
        return True

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
                "id": len(agents),
                "cur": cells[-1],
                "dir": right(outward),
                "out_dir": outward,
                "colors": _DIR_TO_PAIR[outward],
                "color_idx": len(agents) % 2 if slot_phase else 0,
                "phase": "out_first" if len(cells) == 1 else None,
                "done": False,
                "p": 0,
                "turns": 0,
            }
        )

    def paint_cell(r: int, c: int, agent_id: int, color: int) -> None:
        out[r, c] = color
        owner[r, c] = agent_id

    def do_move(agent: dict[str, object], d: tuple[int, int], length: int, color_idx: int) -> None:
        r0, c0 = agent["cur"]
        color = agent["colors"][color_idx]
        for t in range(1, length + 1):
            rr, cc2 = r0 + d[0] * t, c0 + d[1] * t
            paint_cell(rr, cc2, agent["id"], color)
        agent["cur"] = (r0 + d[0] * length, c0 + d[1] * length)
        agent["color_idx"] = color_idx

    def choose_owner_for_new_cell(cell: tuple[int, int]) -> int:
        r, c = cell
        adjacent_ids = []
        for nr, nc in neighbors4(r, c):
            if inb(nr, nc) and owner[nr, nc] >= 0:
                adjacent_ids.append(int(owner[nr, nc]))
        if adjacent_ids:
            counts: dict[int, int] = {}
            for agent_id in adjacent_ids:
                counts[agent_id] = counts.get(agent_id, 0) + 1
            return min(
                counts,
                key=lambda agent_id: (
                    -counts[agent_id],
                    abs(agents[agent_id]["cur"][0] - r) + abs(agents[agent_id]["cur"][1] - c),
                    agent_id,
                ),
            )
        return min(
            range(len(agents)),
            key=lambda agent_id: (
                abs(agents[agent_id]["cur"][0] - r) + abs(agents[agent_id]["cur"][1] - c),
                agent_id,
            ),
        )

    def choose_color_for_new_cell(agent_id: int, cell: tuple[int, int]) -> int:
        r, c = cell
        own_colors = []
        any_colors = []
        for nr, nc in neighbors4(r, c):
            if not inb(nr, nc) or out[nr, nc] == BG:
                continue
            any_colors.append(int(out[nr, nc]))
            if owner[nr, nc] == agent_id:
                own_colors.append(int(out[nr, nc]))
        if own_colors:
            return own_colors[0]
        if any_colors:
            return any_colors[0]
        agent = agents[agent_id]
        return agent["colors"][agent["color_idx"]]

    def add_post_cell(r: int, c: int) -> None:
        if out[r, c] != BG:
            return
        agent_id = choose_owner_for_new_cell((r, c))
        color = choose_color_for_new_cell(agent_id, (r, c))
        paint_cell(r, c, agent_id, color)

    def step_agent_once(agent: dict[str, object]) -> bool:
        if agent["done"]:
            return False

        if agent["phase"] == "out_first":
            if can_move(agent["cur"], agent["out_dir"], 1):
                do_move(agent, agent["out_dir"], 1, agent["color_idx"])
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
                next_idx = _other_index(agent["color_idx"]) if turned else agent["color_idx"]
                do_move(agent, nd, length, next_idx)
                if turned:
                    agent["turns"] += 1
                agent["p"] = 1 if (agent["p"] == 0 and turned) else 0
                agent["dir"] = nd
                return True
            if not turned and length == 2 and can_move(agent["cur"], nd, 1):
                do_move(agent, nd, 1, agent["color_idx"])
                agent["p"] = 0
                return True

        agent["done"] = True
        return False

    def advance_agent_segment(agent: dict[str, object]) -> bool:
        if agent["done"]:
            return False

        moved_any = False
        if agent["phase"] == "out_first":
            if can_move(agent["cur"], agent["out_dir"], 1):
                do_move(agent, agent["out_dir"], 1, agent["color_idx"])
                moved_any = True
            agent["phase"] = None
            agent["dir"] = right(agent["out_dir"])
            agent["p"] = 0

        while True:
            cur = agent["cur"]
            dirs = [right(agent["dir"]), agent["dir"]]
            if not is_border(*cur):
                dirs.append(left(agent["dir"]))

            found = False
            for nd in dirs:
                turned = nd != agent["dir"]
                length = 1 if turned else (2 if agent["p"] == 0 else 1)
                if can_move(agent["cur"], nd, length):
                    next_idx = _other_index(agent["color_idx"]) if turned else agent["color_idx"]
                    do_move(agent, nd, length, next_idx)
                    if turned:
                        agent["turns"] += 1
                    agent["p"] = 1 if (agent["p"] == 0 and turned) else 0
                    agent["dir"] = nd
                    moved_any = True
                    found = True
                    if turned:
                        return True
                    break
                if not turned and length == 2 and can_move(agent["cur"], nd, 1):
                    do_move(agent, nd, 1, agent["color_idx"])
                    agent["p"] = 0
                    moved_any = True
                    found = True
                    break

            if not found:
                agent["done"] = True
                return moved_any

    changed = True
    guard = 0
    step_fn = advance_agent_segment if schedule_mode == "segment" else step_agent_once
    if schedule_mode == "segment":
        for ag in agents:
            if ag["phase"] == "out_first":
                if can_move(ag["cur"], ag["out_dir"], 1):
                    do_move(ag, ag["out_dir"], 1, ag["color_idx"])
                ag["phase"] = None
                ag["dir"] = right(ag["out_dir"])
                ag["p"] = 0
        changed = True
    while changed and guard < 200000:
        guard += 1
        changed = False
        for ag in agents:
            if step_fn(ag):
                changed = True
        if all(ag["done"] for ag in agents):
            break

    if not even_center:
        rp = 1 - (cr % 2)
        cp = 1 - (cc % 2)
        rr, ccg = np.indices((h, w))
        mask = (rr % 2 == rp) & (ccg % 2 == cp)
        out[mask] = BG
        owner[mask] = BG_OWNER
    elif arm_len % 2 == 1:
        to_clear = []
        for r in range(h):
            for c in range(w):
                if out[r, c] == BG or not is_border(r, c):
                    continue
                deg = 0
                nbr = None
                for nr, nc in neighbors4(r, c):
                    if inb(nr, nc) and out[nr, nc] != BG:
                        deg += 1
                        nbr = (nr, nc)
                if deg == 1 and nbr is not None and is_border(*nbr):
                    to_clear.append((r, c))
        for r, c in to_clear:
            erase_cell(r, c)

        br = (h - 1, w - 1)
        if out[br] == BG and all(out[nr, nc] == BG for nr, nc in neighbors4(br[0], br[1]) if inb(nr, nc)):
            add_post_cell(*br)
    else:
        if arm_len >= 2:
            for r in range(min(h, arm_len - 1) - 1, -1, -1):
                add_post_cell(r, 0)
        if arm_len >= 1:
            for c in range(max(0, w - arm_len), w):
                add_post_cell(0, c)
        if h >= 2:
            for c in range(w):
                if out[h - 1, c] != BG and out[h - 2, c] == BG:
                    erase_cell(h - 1, c)

    color_counts = {color: int(np.sum(out == color)) for color in SPIRAL_COLORS}
    agent_turns = [int(agent["turns"]) for agent in agents]
    pair_counts = []
    for agent in agents:
        mask = owner == agent["id"]
        c0, c1 = agent["colors"]
        pair_counts.append((int(np.sum(mask & (out == c0))), int(np.sum(mask & (out == c1)))))

    stats: dict[str, object] = {
        "schedule_mode": schedule_mode,
        "slot_phase": slot_phase,
        "shape": (h, w),
        "arm_len": arm_len,
        "center": (cr, cc),
        "fill_fraction": float(np.mean(out != BG)),
        "painted_cells": int(np.sum(out != BG)),
        "agent_turns": agent_turns,
        "total_turns": int(sum(agent_turns)),
        "color_counts": color_counts,
        "pair_counts": pair_counts,
    }
    return out, stats


def main(grid: np.ndarray) -> np.ndarray:
    """
    Grow four quadrant-owned clockwise spirals from a teal cross, alternating by corner-to-corner rounds and flipping each spiral between its own two colors at every bend.

    Input structure:
    - bg = 0 (black). The only seed object is one teal (8) plus-sign cross with equal arm lengths and a 1-cell empty center.

    Rule:
    - Each arm tip launches one clockwise spiral whose initial heading is 90 degrees clockwise from the arm's outward direction.
    - The four spirals do not advance in the old move-by-move lockstep. Instead, all four tips get the same launch opportunity first, and then the spirals rotate in the fixed order up, right, down, left. When one spiral is active, it keeps moving until it finishes its next turn, then the next spiral takes over.
    - Geometry legality is otherwise preserved: turn-right > straight > turn-left priority, straight steps of length 2 except immediately after a turn, when the next straight is length 1, and every new cell must keep a 1-cell orthogonal gap from earlier occupied cells except the predecessor and launch attachment.
    - Paint is direction-owned and turn-sensitive: up uses 2/6, right uses 1/7, down uses 3/9, left uses 4/5. Straight continuation keeps the current color, and the move that turns 90 degrees switches to the other color in that spiral's pair. The fixed round-robin slot also chooses the launch shade within each pair: the first and third spirals start with the pair's first color, while the second and fourth start with the pair's second color.

    Key insight: The output is no longer explained by geometry alone. The quadrant of origin tells you which two-color palette is active, corner locations tell you when that palette flips, and the fixed round-robin slot tells you which shade of the pair appears first even though the underlying spiral skeleton still comes from the same self-avoiding geometry.
    """

    out, _stats = _solve_with_stats(grid, schedule_mode="segment", slot_phase=True)
    return out


def _build_input(arm: int, size: int, cr: int, cc: int) -> np.ndarray:
    grid = np.zeros((size, size), dtype=int)
    offsets = np.arange(1, arm + 1)
    grid[cr - offsets, cc] = CROSS_COLOR
    grid[cr + offsets, cc] = CROSS_COLOR
    grid[cr, cc - offsets] = CROSS_COLOR
    grid[cr, cc + offsets] = CROSS_COLOR
    return grid


def _phase_delta(inp: np.ndarray, segment_out: np.ndarray | None = None) -> tuple[int, dict[str, object]]:
    if segment_out is None:
        segment_out, segment_stats = _solve_with_stats(inp, schedule_mode="segment", slot_phase=True)
    else:
        segment_stats = {"total_turns": None}
    unphased_out, _baseline_stats = _solve_with_stats(inp, schedule_mode="segment", slot_phase=False)
    delta = int(np.sum(segment_out != unphased_out))
    metrics = {
        "phase_delta": delta,
        "segment_turns": segment_stats["total_turns"],
    }
    return delta, metrics


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
            out, stats = _solve_with_stats(inp, schedule_mode="segment", slot_phase=True)
        except Exception:
            continue

        fill = float(stats["fill_fraction"])
        if not (FILL_LO <= fill <= FILL_HI):
            continue

        if int(stats["total_turns"]) < MIN_TOTAL_TURNS:
            continue

        pair_counts = stats["pair_counts"]
        if any(c0 < MIN_PRIMARY_PIXELS_PER_AGENT or c1 < MIN_ALT_PIXELS_PER_AGENT for c0, c1 in pair_counts):
            continue

        if any(turns < MIN_TURNS_PER_AGENT for turns in stats["agent_turns"]):
            continue

        delta, _metrics = _phase_delta(inp, segment_out=out)
        if delta < MIN_PHASE_DELTA:
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
