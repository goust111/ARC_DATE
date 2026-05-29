# Combo variant of 7ed72f31 (index 21)
# combo_spec: D3/2+D8/1+D9/0
# Name: 变色步进链

from __future__ import annotations

# Design analysis
# ===============
# This combo fuses three changes into one coupled rule:
#   1) D8 chain order: red markers are not independent; they are consumed in visible spatial order.
#   2) D9 deletion: each step moves the current payload to its reflected location and deletes the old one.
#   3) D3 recolor: each step also advances the payload color through a fixed cycle.
#
# a) Concretization and behavior trace
# ------------------------------------
# Small concrete trace:
#   - Input: a blue payload P0 touches the first red marker; later red markers are empty.
#   - Step 1: reflect P0 across a red row -> P1 appears on the opposite side in green; P0 is deleted.
#   - Step 2: reflect P1 across a red column -> P2 appears in yellow; P1 is deleted.
#   - Step 3: point-reflect P2 around a red dot -> P3 appears in teal; P2 is deleted.
#   - Output: all red markers remain, but only P3 is visible.
# Original task difference:
#   - Original 7ed72f31 reads each marker independently from static local payload and keeps originals.
#   - This combo instead carries one moving payload through the whole marker chain, and only the
#     last state survives. Color therefore depends on step count as well as geometry.
#
# b) Complexity calibration
# -------------------------
# We keep only one chain per grid so the execution order is legible. To make the recolor rule
# observable despite D9 deletion, the generator fixes the seed color and uses a short 5-cycle:
#   1 -> 3 -> 4 -> 8 -> 9 -> 1
# Chain length is restricted to 3 or 4 steps, so the final color is always visibly different
# from both the input color and the one-step recolor. This avoids degeneracy into a plain
# single-step recolor or plain chain-without-color.
#
# c) Implicit assumption audit
# ----------------------------
# Original solver assumptions that break:
#   - Global bbox/color scan for payload pixels:
#     no longer valid because only the first marker has input payload; later steps must consume
#     the dynamically produced state, not static same-color pixels in the grid.
#   - "out = g.copy(); just add mirrors":
#     no longer valid because D9 requires deleting the previous payload state.
#   - Isolated red only acts on adjacent static input components:
#     no longer valid for later chain steps, because a point marker may act on the current
#     transient payload even when its local input neighborhood is empty.
# New solver strategy:
#   - Parse all red markers first and sort them by visible row-major position.
#   - Extract exactly one seed payload from the first marker.
#   - Maintain `current_payload`; each loop computes the next reflected-and-recolored payload.
#   - Erase the seed payload from the copied grid and paint only the final payload.
#
# d) Placement legality
# ---------------------
# The generator constructs the whole chain in local coordinates first, tracking every historical
# payload state and every marker in an `occupied` set. This guarantees:
#   - no overlap between any step state and any marker,
#   - no out-of-bounds placements after the local chain is embedded into the final grid,
#   - first marker is row/col and 4-connected to the seed payload,
#   - later markers may be empty in the input but still geometrically valid for the transient state.
#
# e) Visual observability
# -----------------------
# A human sees: one colored sprite near the first red marker, a visible sequence of later red
# markers, and the same shape ending far away in a different color. Because later markers are
# empty in the input but still affect the output, the chain rule is visible. Because the final
# color differs from both the input color and the one-step mapped color, the repeated color
# advancement is also visible.
#
# f) Visual cleanliness
# ---------------------
# The chain always progresses down/right, with a one-cell moat between all historical states and
# markers. That keeps the scene sparse: one seed, one final sprite, and a clean marker trail.
#
# g) Inferability of the color change
# -----------------------------------
# The color rule is intentionally small and fixed-task-level, not data-dependent:
#   1 -> 3 -> 4 -> 8 -> 9 -> 1
# Using one fixed seed color across samples gives repeated support from only a few examples.
# Restricting chain length to 3 or 4 ensures the observed final color is a genuine multi-step
# accumulation, not an accidental no-op or a disguised one-step recolor.
#
# New solver assumptions and generator guarantees
# -----------------------------------------------
# Assumption: the first marker is the only one with an input payload.
# Guarantee: generator attaches the seed only to the first row/col marker.
# Assumption: marker order is row-major visible order.
# Guarantee: generator builds the chain monotonically down/right, so row-major order matches time.
# Assumption: the seed payload is a single monochrome 4-connected sprite.
# Guarantee: generator builds exactly such a seed.
# Assumption: reflected states never overlap markers or earlier states.
# Guarantee: generator rejects any local chain that violates this.

import random
import sys
from pathlib import Path

import numpy as np

SOLVER_DIR = Path("/home/lixinyu/snap/new/solvers")
if str(SOLVER_DIR) not in sys.path:
    sys.path.append(str(SOLVER_DIR))

from arc_utils import get_bg, label_mask


Coord = tuple[int, int]
Payload = list[tuple[int, int, int]]

MARK = 2
COLOR_CYCLE = (1, 3, 4, 8, 9)
COLOR_MAP = {int(src): int(dst) for src, dst in zip(COLOR_CYCLE, COLOR_CYCLE[1:] + COLOR_CYCLE[:1])}
SEED_COLOR = int(COLOR_CYCLE[0])


def advance_color(v: int) -> int:
    return int(COLOR_MAP.get(int(v), int(v)))


def _inb(h: int, w: int, r: int, c: int) -> bool:
    return 0 <= int(r) < int(h) and 0 <= int(c) < int(w)


def _marker_info(red_pts: list[Coord]) -> dict[str, object] | None:
    pts = sorted((int(r), int(c)) for r, c in red_pts)
    if not pts:
        return None
    rs = {int(r) for r, _c in pts}
    cs = {int(c) for _r, c in pts}
    if len(pts) == 1:
        r, c = pts[0]
        return {
            "kind": "point",
            "pts": pts,
            "anchor": (int(r), int(c)),
            "sort_key": (int(r), int(c)),
        }
    if len(rs) == 1:
        r = int(pts[0][0])
        c0 = min(int(c) for _r, c in pts)
        c1 = max(int(c) for _r, c in pts)
        return {
            "kind": "row",
            "pts": pts,
            "row": int(r),
            "c0": int(c0),
            "c1": int(c1),
            "anchor": (int(r), int(c0)),
            "sort_key": (int(r), int(c0)),
        }
    if len(cs) == 1:
        c = int(pts[0][1])
        r0 = min(int(r) for r, _c in pts)
        r1 = max(int(r) for r, _c in pts)
        return {
            "kind": "col",
            "pts": pts,
            "col": int(c),
            "r0": int(r0),
            "r1": int(r1),
            "anchor": (int(r0), int(c)),
            "sort_key": (int(r0), int(c)),
        }
    return None


def _extract_seed_payload(
    g: np.ndarray,
    bg: int,
    marker: dict[str, object],
    nonbg_lab: np.ndarray,
) -> Payload:
    labels = {int(nonbg_lab[r, c]) for r, c in marker["pts"] if int(nonbg_lab[r, c]) != 0}
    if len(labels) == 1:
        lbl = next(iter(labels))
        seed = [
            (int(r), int(c), int(g[r, c]))
            for r, c in np.argwhere(nonbg_lab == int(lbl))
            if int(g[r, c]) != int(MARK)
        ]
        if seed:
            return sorted(seed)

    if marker["kind"] != "point":
        return []

    h, w = g.shape
    pr, pc = marker["pts"][0]
    seed: Payload = []
    seen: set[Coord] = set()
    for dr in (-1, 0, 1):
        for dc in (-1, 0, 1):
            if dr == 0 and dc == 0:
                continue
            rr, cc = int(pr + dr), int(pc + dc)
            if not _inb(h, w, rr, cc):
                continue
            v = int(g[rr, cc])
            if v == int(bg) or v == int(MARK):
                continue
            same_lab, _ = label_mask(g == int(v), connectivity=8)
            lbl = int(same_lab[rr, cc])
            for r, c in np.argwhere(same_lab == int(lbl)):
                pt = (int(r), int(c))
                if pt in seen:
                    continue
                seen.add(pt)
                seed.append((int(r), int(c), int(v)))
    return sorted(seed)


def _reflect_payload(
    payload: Payload,
    marker: dict[str, object],
    h: int,
    w: int,
) -> Payload:
    if not payload:
        return []

    out: Payload = []
    if marker["kind"] == "row":
        row = int(marker["row"])
        c0 = int(marker["c0"])
        c1 = int(marker["c1"])
        cols = [int(c) for _r, c, _v in payload]
        rows = [int(r) for r, _c, _v in payload]
        if min(cols) < c0 or max(cols) > c1:
            return []
        if not (max(rows) < row or min(rows) > row):
            return []
        for r, c, v in payload:
            rr, cc = int(2 * row - r), int(c)
            if not _inb(h, w, rr, cc):
                return []
            out.append((int(rr), int(cc), advance_color(int(v))))
    elif marker["kind"] == "col":
        col = int(marker["col"])
        r0 = int(marker["r0"])
        r1 = int(marker["r1"])
        cols = [int(c) for _r, c, _v in payload]
        rows = [int(r) for r, _c, _v in payload]
        if min(rows) < r0 or max(rows) > r1:
            return []
        if not (max(cols) < col or min(cols) > col):
            return []
        for r, c, v in payload:
            rr, cc = int(r), int(2 * col - c)
            if not _inb(h, w, rr, cc):
                return []
            out.append((int(rr), int(cc), advance_color(int(v))))
    else:
        pr, pc = marker["pts"][0]
        for r, c, v in payload:
            rr, cc = int(2 * pr - r), int(2 * pc - c)
            if not _inb(h, w, rr, cc):
                return []
            out.append((int(rr), int(cc), advance_color(int(v))))
    return sorted(out)


def main(grid: np.ndarray) -> np.ndarray:
    """
    Process red markers as a chained move:
    - extract one seed payload from the first marker,
    - reflect it through every marker in row-major order,
    - recolor after each step,
    - delete the old state each time,
    - keep only the final payload plus all red markers.
    """
    g = np.array(grid, dtype=int)
    h, w = g.shape
    bg = int(get_bg(g))
    out = g.copy()

    red_lab, red_n = label_mask(g == int(MARK), connectivity=4)
    markers: list[dict[str, object]] = []
    for k in range(1, int(red_n) + 1):
        pts = [(int(r), int(c)) for r, c in np.argwhere(red_lab == int(k))]
        info = _marker_info(pts)
        if info is not None:
            markers.append(info)
    if not markers:
        return out

    markers.sort(key=lambda m: tuple(map(int, m["sort_key"])))

    nonbg_lab, _ = label_mask(g != int(bg), connectivity=4)
    seed_payload = _extract_seed_payload(g, int(bg), markers[0], nonbg_lab)
    if not seed_payload:
        return out

    current_payload = sorted(seed_payload)
    for marker in markers:
        current_payload = _reflect_payload(current_payload, marker, int(h), int(w))
        if not current_payload:
            return out

    for r, c, _v in seed_payload:
        out[int(r), int(c)] = int(bg)
    for r, c, v in current_payload:
        out[int(r), int(c)] = int(v)
    return out


def _norm_shape(cells: set[Coord]) -> set[Coord]:
    min_r = min(int(r) for r, _c in cells)
    min_c = min(int(c) for _r, c in cells)
    return {(int(r - min_r), int(c - min_c)) for r, c in cells}


def _random_seed_shape(rng: random.Random) -> set[Coord]:
    for _ in range(300):
        hh = rng.randint(2, 4)
        ww = rng.randint(2, 4)
        target = rng.randint(4, min(7, int(hh * ww)))
        start = (rng.randint(0, hh - 1), rng.randint(0, ww - 1))
        cells = {start}
        while len(cells) < int(target):
            r, c = rng.choice(tuple(cells))
            nbrs = []
            for dr, dc in ((1, 0), (-1, 0), (0, 1), (0, -1)):
                rr, cc = int(r + dr), int(c + dc)
                if 0 <= rr < hh and 0 <= cc < ww:
                    nbrs.append((int(rr), int(cc)))
            rng.shuffle(nbrs)
            for nb in nbrs:
                if nb not in cells:
                    cells.add(nb)
                    break
            else:
                break
        if len(cells) != int(target):
            continue
        cells = _norm_shape(cells)
        max_r = max(int(r) for r, _c in cells)
        max_c = max(int(c) for _r, c in cells)
        row_m = _norm_shape({(int(max_r - r), int(c)) for r, c in cells})
        col_m = _norm_shape({(int(r), int(max_c - c)) for r, c in cells})
        pt_m = _norm_shape({(int(max_r - r), int(max_c - c)) for r, c in cells})
        if cells in (row_m, col_m, pt_m):
            continue
        return cells
    raise RuntimeError("failed to generate asymmetric seed shape")


def _bbox_payload(payload: Payload) -> tuple[int, int, int, int]:
    rs = [int(r) for r, _c, _v in payload]
    cs = [int(c) for _r, c, _v in payload]
    return int(min(rs)), int(min(cs)), int(max(rs)), int(max(cs))


def _coords_only(payload: Payload) -> set[Coord]:
    return {(int(r), int(c)) for r, c, _v in payload}


def _color_trace(states: list[Payload]) -> list[int]:
    trace = []
    for state in states:
        cols = sorted({int(v) for _r, _c, v in state})
        if len(cols) != 1:
            raise RuntimeError("payload lost monochrome property")
        trace.append(int(cols[0]))
    return trace


def _try_place_step(
    current: Payload,
    step_kind: str,
    occupied: set[Coord],
    rng: random.Random,
) -> tuple[dict[str, object], Payload] | None:
    r0, c0, r1, c1 = _bbox_payload(current)
    width = int(c1 - c0 + 1)
    height = int(r1 - r0 + 1)

    if step_kind == "row":
        if width > 6:
            return None
        extra_cap = max(0, min(2, 6 - width))
        extra = rng.randint(0, extra_cap)
        left_pad = rng.randint(0, extra)
        right_pad = extra - left_pad
        row = int(r1 + 1)
        mk_c0 = int(c0 - left_pad)
        mk_c1 = int(c1 + right_pad)
        marker = {
            "kind": "row",
            "pts": [(int(row), int(c)) for c in range(int(mk_c0), int(mk_c1) + 1)],
            "row": int(row),
            "c0": int(mk_c0),
            "c1": int(mk_c1),
            "anchor": (int(row), int(mk_c0)),
            "sort_key": (int(row), int(mk_c0)),
        }
    elif step_kind == "col":
        if height > 6:
            return None
        extra_cap = max(0, min(2, 6 - height))
        extra = rng.randint(0, extra_cap)
        top_pad = rng.randint(0, extra)
        bot_pad = extra - top_pad
        col = int(c1 + 1)
        mk_r0 = int(r0 - top_pad)
        mk_r1 = int(r1 + bot_pad)
        marker = {
            "kind": "col",
            "pts": [(int(r), int(col)) for r in range(int(mk_r0), int(mk_r1) + 1)],
            "col": int(col),
            "r0": int(mk_r0),
            "r1": int(mk_r1),
            "anchor": (int(mk_r0), int(col)),
            "sort_key": (int(mk_r0), int(col)),
        }
    elif step_kind == "point":
        pr = int(r1 + 1)
        pc = int(c1 + 1)
        marker = {
            "kind": "point",
            "pts": [(int(pr), int(pc))],
            "anchor": (int(pr), int(pc)),
            "sort_key": (int(pr), int(pc)),
        }
    else:
        return None

    marker_cells = {(int(r), int(c)) for r, c in marker["pts"]}
    if marker_cells & occupied:
        return None

    nxt = _reflect_payload(current, marker, 10_000, 10_000)
    if not nxt:
        return None
    nxt_cells = _coords_only(nxt)
    if len(nxt_cells) != len(nxt):
        return None
    if nxt_cells & occupied:
        return None
    if nxt_cells & marker_cells:
        return None
    return marker, nxt


def _build_chain(rng: random.Random) -> tuple[list[dict[str, object]], list[Payload], dict[str, object]]:
    shape = _random_seed_shape(rng)
    states: list[Payload] = [sorted((int(r), int(c), int(SEED_COLOR)) for r, c in shape)]
    markers: list[dict[str, object]] = []
    occupied = _coords_only(states[0])
    chain_len = rng.choice([3, 4])
    kinds_used: list[str] = []

    for step_idx in range(chain_len):
        if step_idx == 0:
            kinds = ["row", "col"]
        else:
            kinds = ["point", "row", "col"]
        rng.shuffle(kinds)

        placed = None
        for kind in kinds:
            placed = _try_place_step(states[-1], kind, occupied, rng)
            if placed is not None:
                kinds_used.append(kind)
                break
        if placed is None:
            raise RuntimeError("failed to place chain step")

        marker, nxt = placed
        markers.append(marker)
        states.append(nxt)
        occupied |= {(int(r), int(c)) for r, c in marker["pts"]}
        occupied |= _coords_only(nxt)

    trace = _color_trace(states)
    if trace[-1] in {trace[0], advance_color(trace[0])}:
        raise RuntimeError("degenerate color trace")

    meta = {
        "chain_len": int(chain_len),
        "step_kinds": kinds_used,
        "color_trace": trace,
        "seed_area": int(len(states[0])),
    }
    return markers, states, meta


def _local_to_grid(
    markers: list[dict[str, object]],
    states: list[Payload],
    bg: int,
    rng: random.Random,
) -> tuple[np.ndarray, np.ndarray]:
    all_cells: list[Coord] = []
    for state in states:
        all_cells.extend((int(r), int(c)) for r, c, _v in state)
    for marker in markers:
        all_cells.extend((int(r), int(c)) for r, c in marker["pts"])

    min_r = min(int(r) for r, _c in all_cells)
    max_r = max(int(r) for r, _c in all_cells)
    min_c = min(int(c) for _r, c in all_cells)
    max_c = max(int(c) for _r, c in all_cells)
    need_h = int(max_r - min_r + 1)
    need_w = int(max_c - min_c + 1)

    h_lo = max(18, need_h + 2)
    h_hi = min(28, need_h + 8)
    w_lo = max(18, need_w + 2)
    w_hi = min(28, need_w + 8)
    if h_lo > h_hi or w_lo > w_hi:
        raise RuntimeError("chain too large for target grid")

    h = rng.randint(h_lo, h_hi)
    w = rng.randint(w_lo, w_hi)
    off_r = rng.randint(1 - min_r, h - 2 - max_r)
    off_c = rng.randint(1 - min_c, w - 2 - max_c)

    inp = np.full((h, w), int(bg), dtype=int)
    out = np.full((h, w), int(bg), dtype=int)

    for r, c, v in states[0]:
        rr, cc = int(r + off_r), int(c + off_c)
        inp[rr, cc] = int(v)
    for marker in markers:
        for r, c in marker["pts"]:
            rr, cc = int(r + off_r), int(c + off_c)
            inp[rr, cc] = int(MARK)
            out[rr, cc] = int(MARK)
    for r, c, v in states[-1]:
        rr, cc = int(r + off_r), int(c + off_c)
        out[rr, cc] = int(v)

    return inp, out


def generate_case(rng: random.Random | None = None) -> tuple[np.ndarray, np.ndarray, dict[str, object]]:
    rng = rng or random.Random()
    bg_choices = [int(v) for v in range(10) if int(v) not in set(COLOR_CYCLE) | {int(MARK)}]

    for _ in range(400):
        bg = int(rng.choice(bg_choices))
        try:
            markers, states, meta = _build_chain(rng)
            inp, out = _local_to_grid(markers, states, int(bg), rng)
        except RuntimeError:
            continue

        solved = main(inp)
        if not np.array_equal(solved, out):
            continue

        meta = dict(meta)
        meta["bg"] = int(bg)
        meta["grid_shape"] = (int(inp.shape[0]), int(inp.shape[1]))
        meta["final_color"] = int(meta["color_trace"][-1])
        meta["one_step_color"] = int(advance_color(SEED_COLOR))
        meta["visible_recolor"] = bool(meta["final_color"] != SEED_COLOR)
        meta["multi_step_recolor"] = bool(meta["final_color"] != meta["one_step_color"])
        return inp, out, meta

    raise RuntimeError("failed to generate valid combo case")


def generate_input() -> np.ndarray:
    inp, _out, _meta = generate_case()
    return inp
