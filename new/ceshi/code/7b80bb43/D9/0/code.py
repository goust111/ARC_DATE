# Variant of 7b80bb43
# Dimension: D9 — 线段的边缘锚定约束 (meta)
# Change: 每条自由端点的线段从网格边缘延伸，内部连接器（两端都在交叉节点）不受此约束... → 线段不锚定到边缘（浮动线段）

from __future__ import annotations

from pathlib import Path
import random
import sys

import numpy as np

_SOLVER_DIR = Path("/home/lixinyu/snap/new/solvers")
if str(_SOLVER_DIR) not in sys.path:
    sys.path.append(str(_SOLVER_DIR))

from arc_utils import components_mask, find_components, get_bg

# Design analysis for the D9 variant
# a) Concrete behavior:
#    The visible corruption rule stays the same: diagonal-only fg stair pixels are hints,
#    deleting them reveals where a 3-cell axis-aligned gap should be refilled. The D9
#    change is purely geometric: in the clean output, free-ended horizontal/vertical
#    segments now float strictly inside the grid instead of reaching a border. Example:
#    an original anchored T might be "left edge -> junction" plus "top edge -> junction";
#    in the variant the same LT-shaped T is translated inward so both free ends stop in
#    background, while the input still deletes a 3-cell gap and adds a 2-pixel diagonal
#    clue next to that gap. Solver-wise, the local row/column scan still sees two collinear
#    pieces separated by <=3 cells with a nearby diagonal clue, so the restoration logic
#    is unchanged.
# b) Complexity calibration:
#    To keep the floating change obvious, every generated output has zero fg pixels on the
#    border. To avoid making the task trivial, the number of segments/junctions and the
#    diagonal corruption process are preserved. Components still repeat the same few motifs
#    (T-junctions, simple H, simple V), so the search space grows visually without adding
#    extra symbolic rules.
# c) Implicit-assumption audit of main():
#    - Color extraction assumes exactly one fg color; generator still guarantees that.
#    - diag detection only checks 4-neighbor degree and diagonal adjacency; it does not
#      assume edge anchoring, so floating segments remain valid.
#    - MAXG inference assumes every diagonal connected component is tiny (2-pixel gap clue
#      or isolated distractor of length 2-3); generator preserves that.
#    - Row/column gap filling assumes any true repair is a short collinear gap with a
#      nearby diagonal clue. Floating placement must therefore keep unrelated segments far
#      enough apart that they are not mistaken for one repairable line.
#    - Local pruning assumes any leaf removed is inside the Chebyshev-1 neighborhood of a
#      diagonal clue. Distractors remain isolated so legitimate floating endpoints are not
#      pruned away.
# d) Placement legality:
#    The generator builds each component locally, then places whole components inside an
#    inner frame with at least one-cell Chebyshev clearance from other components and at
#    least a two-cell margin from the border. This prevents overlaps, accidental diagonal
#    contacts, and out-of-bounds gap-chain placements.
# e) Visual observability:
#    Across a few examples, a human can now directly see the meta-change because the clean
#    network no longer "hangs" from any border. The diagonal clean-up rule is still the
#    same, so examples remain learnable from local evidence plus the new global geometry.
# f) Visual cleanliness:
#    Components are spaced apart and kept monochrome. No stacking or multi-layer markers
#    are introduced beyond the original diagonal hints, so the grid remains readable.
# g) Color inference:
#    No new color rule is introduced. The task stays binary (bg + fg), so D9 is expressed
#    only through geometry, not hidden color semantics.


def main(grid: np.ndarray) -> np.ndarray:
    """
    Remove diagonal staircase noise and complete broken straight lines.

    D9 note:
    This solver is intentionally unchanged in spirit from the source task because the
    restoration rule is local: diagonal-only fg pixels identify small breaks in an
    axis-aligned network regardless of whether the cleaned lines touch the border.
    """
    g = np.array(grid, dtype=int)
    bg = int(get_bg(g))

    colors = {int(x) for x in np.unique(g)}
    colors.discard(int(bg))
    assert colors
    fg = int(next(iter(colors)))

    mask = g == int(fg)
    h, w = g.shape

    def deg4(m: np.ndarray, r: int, c: int) -> int:
        d = 0
        if r > 0 and bool(m[int(r - 1), int(c)]):
            d += 1
        if r + 1 < h and bool(m[int(r + 1), int(c)]):
            d += 1
        if c > 0 and bool(m[int(r), int(c - 1)]):
            d += 1
        if c + 1 < w and bool(m[int(r), int(c + 1)]):
            d += 1
        return int(d)

    def has_diag(m: np.ndarray, r: int, c: int) -> bool:
        if r > 0 and c > 0 and bool(m[int(r - 1), int(c - 1)]):
            return True
        if r > 0 and c + 1 < w and bool(m[int(r - 1), int(c + 1)]):
            return True
        if r + 1 < h and c > 0 and bool(m[int(r + 1), int(c - 1)]):
            return True
        if r + 1 < h and c + 1 < w and bool(m[int(r + 1), int(c + 1)]):
            return True
        return False

    diag = np.zeros((h, w), dtype=bool)
    for r in range(h):
        for c in range(w):
            if bool(mask[int(r), int(c)]) and deg4(mask, int(r), int(c)) == 0 and has_diag(mask, int(r), int(c)):
                diag[int(r), int(c)] = True

    near = np.zeros((h, w), dtype=bool)
    for r in range(h):
        for c in range(w):
            if not bool(diag[int(r), int(c)]):
                continue
            for dr in (-1, 0, 1):
                for dc in (-1, 0, 1):
                    rr = int(r + dr)
                    cc = int(c + dc)
                    if 0 <= rr < h and 0 <= cc < w:
                        near[rr, cc] = True

    out = g.copy()
    out[diag] = int(bg)

    max_gap = 3
    if bool(diag.any()):
        for coords in components_mask(diag, connectivity=8):
            if coords.size == 0:
                continue
            roots = set()
            for r, c in coords:
                for dr, dc in ((-1, -1), (-1, 1), (1, -1), (1, 1)):
                    rr = int(r + dr)
                    cc = int(c + dc)
                    if 0 <= rr < h and 0 <= cc < w and bool(mask[rr, cc]) and not bool(diag[rr, cc]):
                        roots.add((rr, cc))
            assert len(coords) + len(roots) <= 3, f"diagonal chain too large ({len(coords)} diag + {len(roots)} root)"
            r0, c0 = coords.min(axis=0)
            r1, c1 = coords.max(axis=0)
            max_gap = max(int(max_gap), int(r1 - r0 + 1), int(c1 - c0 + 1))

    for r in range(h):
        cols = np.where(mask[int(r), :] & (~diag[int(r), :]))[0]
        if cols.size == 0:
            continue
        segs: list[tuple[int, int]] = []
        start = int(cols[0])
        prev = int(cols[0])
        for c in cols[1:]:
            c = int(c)
            if c == prev + 1:
                prev = c
            else:
                segs.append((int(start), int(prev)))
                start = prev = c
        segs.append((int(start), int(prev)))

        for (_a1, b1), (a2, _b2) in zip(segs, segs[1:]):
            gap0 = int(b1 + 1)
            gap1 = int(a2 - 1)
            if gap1 < gap0:
                continue
            if int(gap1 - gap0 + 1) > int(max_gap):
                continue
            rr0 = max(0, int(r - 1))
            rr1 = min(int(h - 1), int(r + 1))
            cc0 = max(0, int(gap0 - 1))
            cc1 = min(int(w - 1), int(gap1 + 1))
            if bool(diag[rr0 : rr1 + 1, cc0 : cc1 + 1].any()):
                out[int(r), int(gap0) : int(gap1) + 1] = int(fg)

    for c in range(w):
        rows = np.where(mask[:, int(c)] & (~diag[:, int(c)]))[0]
        if rows.size == 0:
            continue
        segs: list[tuple[int, int]] = []
        start = int(rows[0])
        prev = int(rows[0])
        for r in rows[1:]:
            r = int(r)
            if r == prev + 1:
                prev = r
            else:
                segs.append((int(start), int(prev)))
                start = prev = r
        segs.append((int(start), int(prev)))

        for (_a1, b1), (a2, _b2) in zip(segs, segs[1:]):
            gap0 = int(b1 + 1)
            gap1 = int(a2 - 1)
            if gap1 < gap0:
                continue
            if int(gap1 - gap0 + 1) > int(max_gap):
                continue
            rr0 = max(0, int(gap0 - 1))
            rr1 = min(int(h - 1), int(gap1 + 1))
            cc0 = max(0, int(c - 1))
            cc1 = min(int(w - 1), int(c + 1))
            if bool(diag[rr0 : rr1 + 1, cc0 : cc1 + 1].any()):
                out[int(gap0) : int(gap1) + 1, int(c)] = int(fg)

    m = out == int(fg)
    changed = True
    while changed:
        changed = False
        to_clear: list[tuple[int, int]] = []
        for r in range(h):
            for c in range(w):
                if not bool(m[int(r), int(c)]) or not bool(near[int(r), int(c)]):
                    continue
                if deg4(m, int(r), int(c)) != 1:
                    continue
                if not has_diag(m, int(r), int(c)):
                    continue
                to_clear.append((int(r), int(c)))
        if to_clear:
            changed = True
            for r, c in to_clear:
                m[int(r), int(c)] = False

    out2 = np.full((h, w), int(bg), dtype=int)
    out2[m] = int(fg)
    return out2


# --- Configuration constants ---

GRID_H_RANGE = (16, 29)
GRID_W_RANGE = (17, 30)
N_COLORS = 10

VALID_CONFIGS = [
    {"segments": 6, "h": 2, "v": 4, "junctions": 2},
    {"segments": 6, "h": 3, "v": 3, "junctions": 2},
    {"segments": 6, "h": 3, "v": 3, "junctions": 3},
    {"segments": 7, "h": 2, "v": 5, "junctions": 2},
    {"segments": 7, "h": 3, "v": 4, "junctions": 2},
    {"segments": 7, "h": 4, "v": 3, "junctions": 2},
    {"segments": 7, "h": 3, "v": 4, "junctions": 3},
    {"segments": 7, "h": 4, "v": 3, "junctions": 3},
    {"segments": 8, "h": 3, "v": 5, "junctions": 3},
    {"segments": 8, "h": 4, "v": 4, "junctions": 3},
    {"segments": 8, "h": 5, "v": 3, "junctions": 3},
]
T_SLOTS = ("LT", "RT", "LB", "RB")

FLOAT_MARGIN = 2
COMPONENT_CLEARANCE = 1
T_SEG_LEN_MIN = 8
T_SEG_LEN_MAX = 12
SIMPLE_H_LEN_MIN = 7
SIMPLE_H_LEN_MAX = 11
SIMPLE_V_LEN_MIN = 6
SIMPLE_V_LEN_MAX = 10

GAP_SIZE = 3
CHAIN_LENGTH = 2
GAP_BLOCKED_LO = 2
GAP_BLOCKED_HI = 4
GAP_SEG_INSET = 2

DISTRACTOR_LENGTHS = (3, 2)
DISTRACTOR_MARGIN = 1

GEN_RETRIES = 300


def _chain_coords(row: int, gap_start: int, anchor: str) -> list[tuple[int, int]]:
    if anchor == "left":
        return [(row + i, gap_start - 1 + i) for i in range(1, CHAIN_LENGTH + 1)]
    return [(row + i, gap_start + GAP_SIZE - i) for i in range(1, CHAIN_LENGTH + 1)]


def _v_chain_coords(col: int, gap_start: int, anchor: str, dc: int) -> list[tuple[int, int]]:
    if anchor == "top":
        return [(gap_start + i, col + dc * (i + 1)) for i in range(CHAIN_LENGTH)]
    return [(gap_start + GAP_SIZE - 1 - i, col + dc * (i + 1)) for i in range(CHAIN_LENGTH)]


def _cells_are_color(grid: np.ndarray, coords: list[tuple[int, int]], color: int) -> bool:
    h, w = grid.shape
    for r, c in coords:
        if not (0 <= r < h and 0 <= c < w) or grid[int(r), int(c)] != int(color):
            return False
    return True


def _validate_chain(
    grid: np.ndarray,
    coords: list[tuple[int, int]],
    fg: int,
    occupied: set[tuple[int, int]],
) -> bool:
    h, w = grid.shape
    coord_set = set(coords)
    if any(cell in occupied for cell in coords):
        return False
    for r, c in coords:
        for rr, cc in ((r - 1, c), (r + 1, c), (r, c - 1), (r, c + 1)):
            if (rr, cc) in coord_set:
                return False
            if 0 <= rr < h and 0 <= cc < w and grid[int(rr), int(cc)] == int(fg):
                return False
        has_any_diag = False
        for rr, cc in ((r - 1, c - 1), (r - 1, c + 1), (r + 1, c - 1), (r + 1, c + 1)):
            if (rr, cc) in coord_set or (0 <= rr < h and 0 <= cc < w and grid[int(rr), int(cc)] == int(fg)):
                has_any_diag = True
                break
        if not has_any_diag:
            return False
    return True


def _h_gap_options(
    seg: dict,
    out: np.ndarray,
    fg: int,
    bg: int,
    occupied: set[tuple[int, int]],
) -> list[tuple[int, str]]:
    row, c0, c1 = seg["row"], seg["c0"], seg["c1"]
    blocked = seg["blocked"]
    options: list[tuple[int, str]] = []
    for g in range(c0 + GAP_SEG_INSET, c1 - GAP_SIZE - 1):
        if any(g - GAP_BLOCKED_LO <= b <= g + GAP_BLOCKED_HI for b in blocked):
            continue
        temp = out.copy()
        temp[int(row), int(g) : int(g) + GAP_SIZE] = int(bg)
        for anchor in ("left", "right"):
            coords = _chain_coords(int(row), int(g), anchor)
            if not _cells_are_color(temp, coords, int(bg)):
                continue
            if not _validate_chain(temp, coords, int(fg), occupied):
                continue
            options.append((int(g), anchor))
    return options


def _v_gap_options(
    seg: dict,
    out: np.ndarray,
    fg: int,
    bg: int,
    occupied: set[tuple[int, int]],
) -> list[tuple[int, str, int]]:
    col, r0, r1 = seg["col"], seg["r0"], seg["r1"]
    blocked = seg["blocked"]
    options: list[tuple[int, str, int]] = []
    for g in range(r0 + GAP_SEG_INSET, r1 - GAP_SIZE - 1):
        if any(g - GAP_BLOCKED_LO <= b <= g + GAP_BLOCKED_HI for b in blocked):
            continue
        temp = out.copy()
        temp[int(g) : int(g) + GAP_SIZE, int(col)] = int(bg)
        for anchor in ("top", "bottom"):
            for dc in (-1, 1):
                coords = _v_chain_coords(int(col), int(g), anchor, int(dc))
                if not _cells_are_color(temp, coords, int(bg)):
                    continue
                if not _validate_chain(temp, coords, int(fg), occupied):
                    continue
                options.append((int(g), anchor, int(dc)))
    return options


def _reserve_neighborhood(cells: set[tuple[int, int]], coords: list[tuple[int, int]]) -> None:
    for r, c in coords:
        for dr in (-1, 0, 1):
            for dc in (-1, 0, 1):
                cells.add((int(r + dr), int(c + dc)))


def _distractor_options(
    inp: np.ndarray,
    bg: int,
    forbidden: set[tuple[int, int]],
) -> list[list[tuple[int, int]]]:
    h, w = inp.shape
    options: list[list[tuple[int, int]]] = []
    for length in DISTRACTOR_LENGTHS:
        for r in range(DISTRACTOR_MARGIN, h - length):
            for c in range(DISTRACTOR_MARGIN, w - DISTRACTOR_MARGIN):
                for direction in (-1, 1):
                    coords = [(int(r + i), int(c + direction * i)) for i in range(length)]
                    if not _cells_are_color(inp, coords, int(bg)):
                        continue
                    if any(cell in forbidden for cell in coords):
                        continue
                    options.append(coords)
    return options


def _make_t_component(slot: str, h_len: int, v_len: int) -> dict:
    mask = np.zeros((v_len, h_len), dtype=bool)
    if slot == "LT":
        jr, jc = v_len - 1, h_len - 1
        mask[int(jr), : int(jc) + 1] = True
        mask[: int(jr) + 1, int(jc)] = True
        h_segments = [{"row": int(jr), "c0": 0, "c1": int(jc), "blocked": {int(jc)}}]
        v_segments = [{"col": int(jc), "r0": 0, "r1": int(jr), "blocked": {int(jr)}}]
    elif slot == "RT":
        jr, jc = v_len - 1, 0
        mask[int(jr), int(jc) : h_len] = True
        mask[: int(jr) + 1, int(jc)] = True
        h_segments = [{"row": int(jr), "c0": int(jc), "c1": h_len - 1, "blocked": {int(jc)}}]
        v_segments = [{"col": int(jc), "r0": 0, "r1": int(jr), "blocked": {int(jr)}}]
    elif slot == "LB":
        jr, jc = 0, h_len - 1
        mask[int(jr), : int(jc) + 1] = True
        mask[int(jr) : v_len, int(jc)] = True
        h_segments = [{"row": int(jr), "c0": 0, "c1": int(jc), "blocked": {int(jc)}}]
        v_segments = [{"col": int(jc), "r0": int(jr), "r1": v_len - 1, "blocked": {int(jr)}}]
    else:
        jr, jc = 0, 0
        mask[int(jr), int(jc) : h_len] = True
        mask[int(jr) : v_len, int(jc)] = True
        h_segments = [{"row": int(jr), "c0": int(jc), "c1": h_len - 1, "blocked": {int(jc)}}]
        v_segments = [{"col": int(jc), "r0": int(jr), "r1": v_len - 1, "blocked": {int(jr)}}]
    return {
        "shape": mask.shape,
        "cells": [tuple(map(int, rc)) for rc in np.argwhere(mask)],
        "h_segments": h_segments,
        "v_segments": v_segments,
    }


def _make_simple_h_component(length: int) -> dict:
    return {
        "shape": (1, int(length)),
        "cells": [(0, int(c)) for c in range(length)],
        "h_segments": [{"row": 0, "c0": 0, "c1": int(length - 1), "blocked": set()}],
        "v_segments": [],
    }


def _make_simple_v_component(length: int) -> dict:
    return {
        "shape": (int(length), 1),
        "cells": [(int(r), 0) for r in range(length)],
        "h_segments": [],
        "v_segments": [{"col": 0, "r0": 0, "r1": int(length - 1), "blocked": set()}],
    }


def _shift_h_segment(seg: dict, dr: int, dc: int) -> dict:
    return {
        "row": int(seg["row"] + dr),
        "c0": int(seg["c0"] + dc),
        "c1": int(seg["c1"] + dc),
        "blocked": set(seg["blocked"]),
    }


def _shift_v_segment(seg: dict, dr: int, dc: int) -> dict:
    return {
        "col": int(seg["col"] + dc),
        "r0": int(seg["r0"] + dr),
        "r1": int(seg["r1"] + dr),
        "blocked": set(seg["blocked"]),
    }


def _expanded_cells(cells: list[tuple[int, int]], radius: int) -> set[tuple[int, int]]:
    out: set[tuple[int, int]] = set()
    for r, c in cells:
        for dr in range(-radius, radius + 1):
            for dc in range(-radius, radius + 1):
                out.add((int(r + dr), int(c + dc)))
    return out


def _build_output(
    rng: random.Random,
    h: int,
    w: int,
    bg: int,
    fg: int,
    cfg: dict,
) -> tuple[np.ndarray, list[dict], list[dict]] | None:
    out = np.full((h, w), int(bg), dtype=int)
    h_segments: list[dict] = []
    v_segments: list[dict] = []

    components: list[dict] = []
    for _ in range(cfg["junctions"]):
        components.append(
            _make_t_component(
                rng.choice(T_SLOTS),
                rng.randint(T_SEG_LEN_MIN, T_SEG_LEN_MAX),
                rng.randint(T_SEG_LEN_MIN, T_SEG_LEN_MAX),
            )
        )
    for _ in range(cfg["h"] - cfg["junctions"]):
        components.append(_make_simple_h_component(rng.randint(SIMPLE_H_LEN_MIN, SIMPLE_H_LEN_MAX)))
    for _ in range(cfg["v"] - cfg["junctions"]):
        components.append(_make_simple_v_component(rng.randint(SIMPLE_V_LEN_MIN, SIMPLE_V_LEN_MAX)))

    components.sort(key=lambda comp: comp["shape"][0] * comp["shape"][1], reverse=True)
    reserved: set[tuple[int, int]] = set()

    for comp in components:
        ch, cw = comp["shape"]
        row_choices = list(range(FLOAT_MARGIN, h - FLOAT_MARGIN - ch + 1))
        col_choices = list(range(FLOAT_MARGIN, w - FLOAT_MARGIN - cw + 1))
        if not row_choices or not col_choices:
            return None
        placements = [(int(r0), int(c0)) for r0 in row_choices for c0 in col_choices]
        rng.shuffle(placements)
        placed = False
        for r0, c0 in placements[:200]:
            cells = [(int(r0 + r), int(c0 + c)) for r, c in comp["cells"]]
            if any(cell in reserved for cell in cells):
                continue
            for r, c in cells:
                out[int(r), int(c)] = int(fg)
            h_segments.extend(_shift_h_segment(seg, int(r0), int(c0)) for seg in comp["h_segments"])
            v_segments.extend(_shift_v_segment(seg, int(r0), int(c0)) for seg in comp["v_segments"])
            reserved.update(_expanded_cells(cells, COMPONENT_CLEARANCE))
            placed = True
            break
        if not placed:
            return None

    comps = find_components(out, bg=int(bg), connectivity=4, monochromatic=True, return_format="dict")
    if len(comps) != cfg["segments"] - cfg["junctions"]:
        return None
    if bool((out[0, :] == int(fg)).any() or (out[-1, :] == int(fg)).any() or (out[:, 0] == int(fg)).any() or (out[:, -1] == int(fg)).any()):
        return None
    return out, h_segments, v_segments


def _gen_attempt() -> np.ndarray | None:
    rng = random.Random()
    h = rng.randint(*GRID_H_RANGE)
    w = rng.randint(*GRID_W_RANGE)
    bg = rng.randrange(N_COLORS)
    fg = rng.choice([c for c in range(N_COLORS) if c != bg])
    cfg = rng.choice(VALID_CONFIGS)
    gap_count = 2 if cfg["h"] == 2 else rng.choice([2, 3])
    distractor_count = rng.randint(0, 2)

    built = _build_output(rng, h, w, bg, fg, cfg)
    if built is None:
        return None
    out, h_segments, v_segments = built

    inp = out.copy()
    occupied: set[tuple[int, int]] = set()

    h_ready = [(seg, opts) for seg in h_segments for opts in [_h_gap_options(seg, out, fg, bg, occupied)] if opts]
    v_ready = [(seg, opts) for seg in v_segments for opts in [_v_gap_options(seg, out, fg, bg, occupied)] if opts]
    if not h_ready or not v_ready:
        return None

    v_gap_count = gap_count - 1
    if len(v_ready) < v_gap_count:
        return None

    rng.shuffle(h_ready)
    rng.shuffle(v_ready)

    h_seg, h_initial = h_ready[0]
    h_opts = _h_gap_options(h_seg, out, fg, bg, occupied) or h_initial
    gap_start, anchor = rng.choice(h_opts)
    row = h_seg["row"]
    chain = _chain_coords(int(row), int(gap_start), anchor)
    inp[int(row), int(gap_start) : int(gap_start) + GAP_SIZE] = int(bg)
    for r, c in chain:
        inp[int(r), int(c)] = int(fg)
    occupied.update(chain)

    for v_seg, v_initial in v_ready[:v_gap_count]:
        v_opts = _v_gap_options(v_seg, out, fg, bg, occupied) or v_initial
        if not v_opts:
            return None
        gap_start, anchor, dc = rng.choice(v_opts)
        col = v_seg["col"]
        chain = _v_chain_coords(int(col), int(gap_start), anchor, int(dc))
        inp[int(gap_start) : int(gap_start) + GAP_SIZE, int(col)] = int(bg)
        for r, c in chain:
            inp[int(r), int(c)] = int(fg)
        occupied.update(chain)

    forbidden: set[tuple[int, int]] = set()
    _reserve_neighborhood(forbidden, [tuple(map(int, rc)) for rc in np.argwhere(inp == int(fg))])
    _reserve_neighborhood(forbidden, list(occupied))
    distractors = _distractor_options(inp, bg, forbidden)
    rng.shuffle(distractors)
    placed = 0
    for coords in distractors:
        if placed >= distractor_count:
            break
        if any(cell in forbidden for cell in coords):
            continue
        for r, c in coords:
            inp[int(r), int(c)] = int(fg)
        _reserve_neighborhood(forbidden, coords)
        placed += 1

    try:
        if not np.array_equal(main(inp), out):
            return None
    except Exception:
        return None
    return inp


def generate_input() -> np.ndarray:
    for _ in range(GEN_RETRIES):
        grid = _gen_attempt()
        if grid is not None:
            return grid
    raise RuntimeError("failed to generate valid input for 7b80bb43 D9 floating variant")
