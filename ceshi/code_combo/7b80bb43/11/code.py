from __future__ import annotations

# Combo variant of 7b80bb43 (index 11)
# combo_spec: D6/1+D7/3
# Name: 大量干扰递归清理

import random
import sys
from pathlib import Path

import numpy as np

ROOT = Path(__file__).resolve().parents[4]
SOLVER_DIR = ROOT / "solvers"
if str(SOLVER_DIR) not in sys.path:
    sys.path.append(str(SOLVER_DIR))

from arc_utils import components_mask, draw_line, find_components, get_bg


# -----------------------------------------------------------------------------
# Design analysis for the fused variant
#
# a) Concrete behavior trace:
#    - Original task: remove every purely diagonal fg chain, use chains near an
#      exact 3-cell H/V break to refill the straight segment, then do a small
#      local cleanup near those hints.
#    - Fused task: keep the true gap hints unchanged, but also inject 3-5 extra
#      diagonal chains in open background. Each distractor chain owns a short
#      orthogonal twig. After diagonal pixels are deleted, those twigs remain as
#      deg-1 paths and must be peeled recursively until stable.
#    - Small example: a distractor object shaped like twig {(8,10),(9,10),(10,10)}
#      plus diagonal chain {(7,9),(6,8)} becomes just the 3-pixel vertical twig
#      after diagonal deletion. The original solver would leave that twig in
#      place because no diagonal neighbors remain. The fused solver seeds pruning
#      from the unused diagonal component, deletes the twig tip, then the new
#      tip, then the last pixel.
#
# b) Complexity calibration:
#    - Purely increasing the distractor count is not enough; if the distractors
#      were only diagonal pixels, the new pruning rule would be visually inert.
#    - To make D7/3 observable without collapsing the whole line network, every
#      distractor includes a short orthogonal residue, and real gap hints remain
#      structurally different because they sit beside an exact 3-cell straight
#      break.
#
# c) Implicit-assumption audit of the original solver:
#    - The original MAXG / local-prune logic assumes deleted diagonal pixels are
#      almost always tied to a nearby repair site. That fails once many unused
#      diagonal components are added.
#    - A naive global recursive deg<=1 pruning would also fail: the original
#      clean outputs are mostly trees, so a full graph peel would erase the real
#      network. The fused solver therefore fills true gaps first, then only
#      launches recursive pruning from foreground pixels touched by *unused*
#      diagonal components.
#
# d) Placement legality:
#    - Real gap hints still satisfy the original geometry: exact 3-cell gap,
#      2-pixel diagonal chain, no orthogonal fg contact, in-bounds placement.
#    - Distractor assemblies are placed only on background cells with a
#      Chebyshev-1 safety margin from the main network, true hints, and each
#      other; their diagonal chains are validated to remain diagonal-only.
#
# e) Visual observability:
#    - Across 2-4 ARC examples, a human can still spot the same two signal
#      classes: diagonal chains attached to a broken straight segment refill it,
#      while many free-floating diagonal-plus-twig objects disappear entirely.
#    - The pruning effect is now visible because some noise survives the first
#      diagonal deletion pass and only vanishes after recursive branch peeling.
#
# f) Visual cleanliness:
#    - Distractors are isolated from the true network, so the noisy field looks
#      denser but not tangled. Their twigs are short and uniform, which keeps the
#      extra clutter legible.
#
# g) Rule inferability:
#    - No hidden color ordering is used. All reasoning is structural: exact
#      3-cell straight gaps versus isolated diagonal noise with attached leaf
#      branches. The generator keeps these two categories repeated enough that
#      the distinction is learnable from few examples.
#
# h) New solver assumptions introduced here:
#    - Unused diagonal components may seed recursive pruning, but the generator
#      guarantees they are isolated from the main network by at least one blank
#      cell. Even if a rare sample violates that, pruning only propagates from
#      exposed leaves outward, so it stops when a path reaches a degree-2+
#      backbone.
#    - Every true repair gap is exactly 3 cells wide, matching the spec and the
#      generator; the solver uses that stricter condition to avoid false fills
#      from the denser distractor field.
# -----------------------------------------------------------------------------


GAP_SIZE = 3
TRUE_CHAIN_LENGTH = 2
DISTRACTOR_COUNT_RANGE = (3, 5)
DISTRACTOR_TWIG_RANGE = (2, 4)
DISTRACTOR_CHAIN_RANGE = (2, 3)


def _deg4(mask: np.ndarray, r: int, c: int) -> int:
    h, w = mask.shape
    deg = 0
    if r > 0 and bool(mask[r - 1, c]):
        deg += 1
    if r + 1 < h and bool(mask[r + 1, c]):
        deg += 1
    if c > 0 and bool(mask[r, c - 1]):
        deg += 1
    if c + 1 < w and bool(mask[r, c + 1]):
        deg += 1
    return deg


def _has_diag(mask: np.ndarray, r: int, c: int) -> bool:
    h, w = mask.shape
    for dr in (-1, 1):
        for dc in (-1, 1):
            rr = r + dr
            cc = c + dc
            if 0 <= rr < h and 0 <= cc < w and bool(mask[rr, cc]):
                return True
    return False


def _build_diag_mask(mask: np.ndarray) -> np.ndarray:
    h, w = mask.shape
    diag = np.zeros((h, w), dtype=bool)
    for r in range(h):
        for c in range(w):
            if bool(mask[r, c]) and _deg4(mask, r, c) == 0 and _has_diag(mask, r, c):
                diag[r, c] = True
    return diag


def _diag_components(diag: np.ndarray) -> tuple[np.ndarray, list[np.ndarray]]:
    ids = np.zeros(diag.shape, dtype=int)
    comps = components_mask(diag, connectivity=8)
    for idx, coords in enumerate(comps, start=1):
        for r, c in coords:
            ids[int(r), int(c)] = idx
    return ids, comps


def _segments_1d(indices: np.ndarray) -> list[tuple[int, int]]:
    if indices.size == 0:
        return []
    segs: list[tuple[int, int]] = []
    start = prev = int(indices[0])
    for value in indices[1:]:
        value = int(value)
        if value == prev + 1:
            prev = value
            continue
        segs.append((start, prev))
        start = prev = value
    segs.append((start, prev))
    return segs


def _solve_core(grid: np.ndarray, *, return_metrics: bool = False):
    g = np.array(grid, dtype=int)
    bg = int(get_bg(g))
    colors = {int(x) for x in np.unique(g)}
    colors.discard(bg)
    assert colors, "expected one foreground color"
    fg = int(next(iter(colors)))

    mask = g == fg
    h, w = g.shape

    diag = _build_diag_mask(mask)
    diag_ids, diag_comps = _diag_components(diag)
    base = mask & (~diag)
    repaired = base.copy()
    used_diag_ids: set[int] = set()

    for r in range(h):
        cols = np.where(base[r])[0]
        segs = _segments_1d(cols)
        for (_a1, b1), (a2, _b2) in zip(segs, segs[1:]):
            gap0 = int(b1 + 1)
            gap1 = int(a2 - 1)
            if gap1 - gap0 + 1 != GAP_SIZE:
                continue
            rr0 = max(0, r - 1)
            rr1 = min(h - 1, r + 1)
            cc0 = max(0, gap0 - 1)
            cc1 = min(w - 1, gap1 + 1)
            nearby = set(int(x) for x in np.unique(diag_ids[rr0 : rr1 + 1, cc0 : cc1 + 1]))
            nearby.discard(0)
            if not nearby:
                continue
            repaired[r, gap0 : gap1 + 1] = True
            used_diag_ids.update(nearby)

    for c in range(w):
        rows = np.where(base[:, c])[0]
        segs = _segments_1d(rows)
        for (_a1, b1), (a2, _b2) in zip(segs, segs[1:]):
            gap0 = int(b1 + 1)
            gap1 = int(a2 - 1)
            if gap1 - gap0 + 1 != GAP_SIZE:
                continue
            rr0 = max(0, gap0 - 1)
            rr1 = min(h - 1, gap1 + 1)
            cc0 = max(0, c - 1)
            cc1 = min(w - 1, c + 1)
            nearby = set(int(x) for x in np.unique(diag_ids[rr0 : rr1 + 1, cc0 : cc1 + 1]))
            nearby.discard(0)
            if not nearby:
                continue
            repaired[gap0 : gap1 + 1, c] = True
            used_diag_ids.update(nearby)

    distractor_seed = np.zeros((h, w), dtype=bool)
    for idx, coords in enumerate(diag_comps, start=1):
        if idx in used_diag_ids:
            continue
        for r, c in coords:
            for dr in (-1, 0, 1):
                for dc in (-1, 0, 1):
                    rr = int(r + dr)
                    cc = int(c + dc)
                    if 0 <= rr < h and 0 <= cc < w:
                        distractor_seed[rr, cc] = True

    frontier = {tuple(map(int, rc)) for rc in np.argwhere(repaired & distractor_seed)}
    pruned_pixels = 0
    while frontier:
        to_clear: list[tuple[int, int]] = []
        for r, c in frontier:
            if bool(repaired[r, c]) and _deg4(repaired, r, c) <= 1:
                to_clear.append((r, c))
        if not to_clear:
            break
        next_frontier: set[tuple[int, int]] = set()
        for r, c in to_clear:
            if not bool(repaired[r, c]):
                continue
            repaired[r, c] = False
            pruned_pixels += 1
            for rr, cc in ((r, c), (r - 1, c), (r + 1, c), (r, c - 1), (r, c + 1)):
                if 0 <= rr < h and 0 <= cc < w and bool(repaired[rr, cc]):
                    next_frontier.add((rr, cc))
        frontier = next_frontier

    out = np.full((h, w), bg, dtype=int)
    out[repaired] = fg
    if not return_metrics:
        return out
    metrics = {
        "fg": fg,
        "bg": bg,
        "diag_components": len(diag_comps),
        "used_diag_components": len(used_diag_ids),
        "unused_diag_components": len(diag_comps) - len(used_diag_ids),
        "pruned_pixels": pruned_pixels,
    }
    return out, metrics


def main(grid: np.ndarray) -> np.ndarray:
    """
    Remove all diagonal staircases, restore the real line breaks, and recursively peel distractor residue.

    Input structure:
    - One foreground color forms an axis-aligned H/V segment network on a background color.
    - Some short diagonal fg chains sit beside exact 3-cell breaks in the network, and 3-5 extra diagonal chains sit in empty space with small orthogonal twigs.

    Rule:
    - Delete every fg pixel that is connected only diagonally.
    - Refill any exact 3-cell horizontal or vertical gap whose neighborhood contained one of those deleted diagonal components.
    - Treat the remaining unused diagonal components as distractor seeds and recursively delete nearby degree<=1 foreground residue until no seeded branch can shrink further.

    Key insight:
    - Useful diagonal chains are the ones that explain a straight 3-cell gap; all other diagonal chains are noise markers whose nearby leaf branches must be peeled away after the real gaps are repaired.
    """
    return _solve_core(grid, return_metrics=False)


def _analyze_case(grid: np.ndarray) -> dict[str, int]:
    _, metrics = _solve_core(grid, return_metrics=True)
    return metrics


# --------------- generator ---------------

# Grid dimensions
GRID_H_RANGE = (16, 29)
GRID_W_RANGE = (17, 30)
N_COLORS = 10

# Segment network topology
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

# Layout spacing
EDGE_MARGIN = 3
ROW_COL_GAP = 2
T_BRANCH_OFFSET_MIN = 3
T_BRANCH_OFFSET_MAX = 9
H_SEG_END_MARGIN = 4
H_SEG_MAX_END = 25

# Simple H-segment sizing
SIMPLE_H_LEN_MIN = 7
SIMPLE_H_LEN_MAX = 11
SIMPLE_V_MARGIN_NEAR = 7
SIMPLE_V_MARGIN_FAR = 3

# Gap / true hint placement
GAP_BLOCKED_LO = 2
GAP_BLOCKED_HI = 4
GAP_SEG_INSET = 2

# Distractor placement
DISTRACTOR_MARGIN = 1
DISTRACTOR_ORIENTATIONS = [
    ((1, 0), (-1, -1)),
    ((1, 0), (-1, 1)),
    ((-1, 0), (1, -1)),
    ((-1, 0), (1, 1)),
    ((0, 1), (-1, -1)),
    ((0, 1), (1, -1)),
    ((0, -1), (-1, 1)),
    ((0, -1), (1, 1)),
]

# Retry budget
GEN_RETRIES = 600


def _pick_distinct(rng: random.Random, values: list[int], k: int, gap: int = ROW_COL_GAP) -> list[int]:
    vals = values[:]
    rng.shuffle(vals)
    picked: list[int] = []
    for value in vals:
        if all(abs(value - prev) >= gap for prev in picked):
            picked.append(value)
            if len(picked) == k:
                return sorted(picked)
    return []


def _chain_coords(row: int, gap_start: int, anchor: str) -> list[tuple[int, int]]:
    if anchor == "left":
        return [(row + i, gap_start - 1 + i) for i in range(1, TRUE_CHAIN_LENGTH + 1)]
    return [(row + i, gap_start + GAP_SIZE - i) for i in range(1, TRUE_CHAIN_LENGTH + 1)]


def _v_chain_coords(col: int, gap_start: int, anchor: str, dc: int) -> list[tuple[int, int]]:
    if anchor == "top":
        return [(gap_start + i, col + dc * (i + 1)) for i in range(TRUE_CHAIN_LENGTH)]
    return [(gap_start + GAP_SIZE - 1 - i, col + dc * (i + 1)) for i in range(TRUE_CHAIN_LENGTH)]


def _cells_are_color(grid: np.ndarray, coords: list[tuple[int, int]], color: int) -> bool:
    h, w = grid.shape
    for r, c in coords:
        if not (0 <= r < h and 0 <= c < w) or int(grid[r, c]) != int(color):
            return False
    return True


def _validate_chain(grid: np.ndarray, coords: list[tuple[int, int]], fg: int,
                    occupied: set[tuple[int, int]]) -> bool:
    h, w = grid.shape
    coord_set = set(coords)
    if any(cell in occupied for cell in coords):
        return False
    for r, c in coords:
        for rr, cc in ((r - 1, c), (r + 1, c), (r, c - 1), (r, c + 1)):
            if (rr, cc) in coord_set:
                return False
            if 0 <= rr < h and 0 <= cc < w and int(grid[rr, cc]) == int(fg):
                return False
        has_diag = False
        for rr, cc in ((r - 1, c - 1), (r - 1, c + 1), (r + 1, c - 1), (r + 1, c + 1)):
            if (rr, cc) in coord_set or (0 <= rr < h and 0 <= cc < w and int(grid[rr, cc]) == int(fg)):
                has_diag = True
                break
        if not has_diag:
            return False
    return True


def _place_t_component(out: np.ndarray, fg: int, slot: str, row: int,
                       branch_col: int, end_col: int) -> tuple[dict, dict]:
    h, w = out.shape
    h_c0 = 0 if slot[0] == "L" else end_col
    h_c1 = end_col if slot[0] == "L" else w - 1
    v_r0 = 0 if slot[1] == "T" else row
    v_r1 = row if slot[1] == "T" else h - 1
    draw_line(out, row, h_c0, end_r=row, end_c=h_c1, color=fg)
    draw_line(out, v_r0, branch_col, end_r=v_r1, end_c=branch_col, color=fg)
    h_seg = {"row": row, "c0": h_c0, "c1": h_c1, "blocked": {branch_col}}
    v_seg = {"col": branch_col, "r0": v_r0, "r1": v_r1, "blocked": {row}}
    return h_seg, v_seg


def _place_simple_h(out: np.ndarray, fg: int, side: str, row: int, length: int) -> dict:
    w = out.shape[1]
    c0 = 0 if side == "left" else w - length
    c1 = length - 1 if side == "left" else w - 1
    draw_line(out, row, c0, end_r=row, end_c=c1, color=fg)
    return {"row": row, "c0": c0, "c1": c1, "blocked": set()}


def _place_simple_v(out: np.ndarray, fg: int, side: str, col: int, length: int) -> dict:
    h = out.shape[0]
    r0 = 0 if side == "top" else h - length
    r1 = length - 1 if side == "top" else h - 1
    draw_line(out, r0, col, end_r=r1, end_c=col, color=fg)
    return {"col": col, "r0": r0, "r1": r1, "blocked": set()}


def _h_gap_options(seg: dict, out: np.ndarray, fg: int, bg: int,
                   occupied: set[tuple[int, int]]) -> list[tuple[int, str]]:
    row, c0, c1 = seg["row"], seg["c0"], seg["c1"]
    blocked = seg["blocked"]
    options: list[tuple[int, str]] = []
    for gap_start in range(c0 + GAP_SEG_INSET, c1 - GAP_SIZE - 1):
        if any(gap_start - GAP_BLOCKED_LO <= b <= gap_start + GAP_BLOCKED_HI for b in blocked):
            continue
        temp = out.copy()
        temp[row, gap_start : gap_start + GAP_SIZE] = bg
        for anchor in ("left", "right"):
            coords = _chain_coords(row, gap_start, anchor)
            if not _cells_are_color(temp, coords, bg):
                continue
            if not _validate_chain(temp, coords, fg, occupied):
                continue
            options.append((gap_start, anchor))
    return options


def _v_gap_options(seg: dict, out: np.ndarray, fg: int, bg: int,
                   occupied: set[tuple[int, int]]) -> list[tuple[int, str, int]]:
    col, r0, r1 = seg["col"], seg["r0"], seg["r1"]
    blocked = seg["blocked"]
    options: list[tuple[int, str, int]] = []
    for gap_start in range(r0 + GAP_SEG_INSET, r1 - GAP_SIZE - 1):
        if any(gap_start - GAP_BLOCKED_LO <= b <= gap_start + GAP_BLOCKED_HI for b in blocked):
            continue
        temp = out.copy()
        temp[gap_start : gap_start + GAP_SIZE, col] = bg
        for anchor in ("top", "bottom"):
            for dc in (-1, 1):
                coords = _v_chain_coords(col, gap_start, anchor, dc)
                if not _cells_are_color(temp, coords, bg):
                    continue
                if not _validate_chain(temp, coords, fg, occupied):
                    continue
                options.append((gap_start, anchor, dc))
    return options


def _reserve_neighborhood(cells: set[tuple[int, int]], coords: list[tuple[int, int]]) -> None:
    for r, c in coords:
        for dr in (-1, 0, 1):
            for dc in (-1, 0, 1):
                cells.add((r + dr, c + dc))


def _twig_and_chain(root: tuple[int, int], twig_dir: tuple[int, int], twig_len: int,
                    diag_dir: tuple[int, int], chain_len: int) -> tuple[list[tuple[int, int]], list[tuple[int, int]]]:
    r0, c0 = root
    twig = [(r0 + i * twig_dir[0], c0 + i * twig_dir[1]) for i in range(twig_len)]
    chain = [(r0 + (i + 1) * diag_dir[0], c0 + (i + 1) * diag_dir[1]) for i in range(chain_len)]
    return twig, chain


def _distractor_options(inp: np.ndarray, fg: int, bg: int,
                        forbidden: set[tuple[int, int]]) -> list[tuple[list[tuple[int, int]], list[tuple[int, int]]]]:
    h, w = inp.shape
    options: list[tuple[list[tuple[int, int]], list[tuple[int, int]]]] = []
    for r in range(DISTRACTOR_MARGIN, h - DISTRACTOR_MARGIN):
        for c in range(DISTRACTOR_MARGIN, w - DISTRACTOR_MARGIN):
            root = (r, c)
            for twig_dir, diag_dir in DISTRACTOR_ORIENTATIONS:
                for twig_len in range(DISTRACTOR_TWIG_RANGE[0], DISTRACTOR_TWIG_RANGE[1] + 1):
                    for chain_len in range(DISTRACTOR_CHAIN_RANGE[0], DISTRACTOR_CHAIN_RANGE[1] + 1):
                        twig, chain = _twig_and_chain(root, twig_dir, twig_len, diag_dir, chain_len)
                        coords = twig + chain
                        if any(cell in forbidden for cell in coords):
                            continue
                        if not _cells_are_color(inp, coords, bg):
                            continue
                        temp = inp.copy()
                        for rr, cc in twig:
                            temp[rr, cc] = fg
                        if not _validate_chain(temp, chain, fg, set()):
                            continue
                        options.append((twig, chain))
    return options


def _build_output(rng: random.Random, h: int, w: int, bg: int, fg: int,
                  cfg: dict) -> tuple[np.ndarray, list[dict], list[dict]] | None:
    out = np.full((h, w), bg, dtype=int)
    h_segments: list[dict] = []
    v_segments: list[dict] = []

    h_simple = cfg["h"] - cfg["junctions"]
    v_simple = cfg["v"] - cfg["junctions"]
    t_slots = rng.sample(list(T_SLOTS), cfg["junctions"])

    top_slots = [slot for slot in t_slots if slot[1] == "T"]
    bottom_slots = [slot for slot in t_slots if slot[1] == "B"]
    top_pool = list(range(EDGE_MARGIN, max(EDGE_MARGIN + 1, h // 2 - 2)))
    bottom_pool = list(range(max(h // 2 + 1, EDGE_MARGIN + 2), h - EDGE_MARGIN))
    top_rows = _pick_distinct(rng, top_pool, len(top_slots))
    bottom_rows = _pick_distinct(rng, bottom_pool, len(bottom_slots))
    if len(top_rows) != len(top_slots) or len(bottom_rows) != len(bottom_slots):
        return None
    row_by_slot = dict(zip(top_slots, top_rows)) | dict(zip(bottom_slots, bottom_rows))

    branch_cols = _pick_distinct(rng, list(range(EDGE_MARGIN, w - EDGE_MARGIN)), len(t_slots))
    if len(branch_cols) != len(t_slots):
        return None
    rng.shuffle(branch_cols)
    branch_by_slot = dict(zip(t_slots, branch_cols))

    for slot in t_slots:
        branch_col = branch_by_slot[slot]
        row = row_by_slot[slot]
        if slot[0] == "L":
            min_end = max(SIMPLE_H_LEN_MIN + 1, branch_col + T_BRANCH_OFFSET_MIN)
            max_end = min(w - H_SEG_END_MARGIN, branch_col + T_BRANCH_OFFSET_MAX, H_SEG_MAX_END)
        else:
            min_end = max(2, branch_col - T_BRANCH_OFFSET_MAX, w - H_SEG_MAX_END - 1)
            max_end = min(w - SIMPLE_H_LEN_MIN - 1, branch_col - T_BRANCH_OFFSET_MIN)
        if min_end > max_end:
            return None
        end_col = rng.randint(min_end, max_end)
        h_seg, v_seg = _place_t_component(out, fg, slot, row, branch_col, end_col)
        h_segments.append(h_seg)
        v_segments.append(v_seg)

    center_start = max(top_rows + [1]) + 2
    center_end = min(bottom_rows + [h - 2]) - 2
    if h_simple > 0 and center_start > center_end:
        return None

    simple_h_rows = _pick_distinct(rng, list(range(center_start, center_end + 1)), h_simple) if h_simple else []
    if len(simple_h_rows) != h_simple:
        return None
    if h_simple == 0:
        h_sides: list[str] = []
    elif h_simple == 1:
        h_sides = [rng.choice(["left", "right"])]
    else:
        h_sides = ["left", "right"]
        rng.shuffle(h_sides)

    simple_v_cols: list[int] = []
    used_branch_cols = set(branch_by_slot.values())
    if v_simple:
        has_left = "left" in h_sides
        has_right = "right" in h_sides
        if has_left and has_right:
            col_pool = list(range(SIMPLE_V_MARGIN_NEAR, w - SIMPLE_V_MARGIN_NEAR))
        elif has_left:
            col_pool = list(range(SIMPLE_V_MARGIN_NEAR, w - SIMPLE_V_MARGIN_FAR))
        elif has_right:
            col_pool = list(range(2, w - SIMPLE_V_MARGIN_NEAR))
        else:
            col_pool = list(range(2, w - 2))
        col_pool = [col for col in col_pool if col not in used_branch_cols]
        simple_v_cols = _pick_distinct(rng, col_pool, v_simple)
        if len(simple_v_cols) != v_simple:
            return None

    for row, side in zip(simple_h_rows, h_sides):
        if side == "left":
            max_len = (min(simple_v_cols) if simple_v_cols else w - EDGE_MARGIN) - 1
        else:
            max_len = w - (max(simple_v_cols) if simple_v_cols else 2) - 2
        max_len = min(SIMPLE_H_LEN_MAX, max_len)
        if max_len < SIMPLE_H_LEN_MIN:
            return None
        h_segments.append(_place_simple_h(out, fg, side, row, rng.randint(SIMPLE_H_LEN_MIN, max_len)))

    top_v = min(2, v_simple)
    top_v_cols = simple_v_cols[:top_v]
    bottom_v_cols = simple_v_cols[top_v:]
    top_limit = min(min(top_rows) - 2 if top_rows else center_start - 2, center_start - 2)
    bottom_start = max(max(bottom_rows) + 2 if bottom_rows else center_end + 2, center_end + 2)
    for col in top_v_cols:
        if top_limit < 2:
            return None
        v_segments.append(_place_simple_v(out, fg, "top", col, rng.randint(2, top_limit)))
    for col in bottom_v_cols:
        max_len = h - bottom_start
        if max_len < 2:
            return None
        v_segments.append(_place_simple_v(out, fg, "bottom", col, rng.randint(2, max_len)))

    comps = find_components(out, bg=bg, connectivity=4, monochromatic=True, return_format="dict")
    if len(comps) != cfg["segments"] - cfg["junctions"]:
        return None
    return out, h_segments, v_segments


def _gen_attempt() -> np.ndarray | None:
    rng = random.Random()
    h = rng.randint(*GRID_H_RANGE)
    w = rng.randint(*GRID_W_RANGE)
    bg = rng.randrange(N_COLORS)
    fg = rng.choice([color for color in range(N_COLORS) if color != bg])
    cfg = rng.choice(VALID_CONFIGS)
    gap_count = 2 if cfg["h"] == 2 else rng.choice([2, 3])
    distractor_count = rng.randint(*DISTRACTOR_COUNT_RANGE)

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
    chain = _chain_coords(row, gap_start, anchor)
    inp[row, gap_start : gap_start + GAP_SIZE] = bg
    for r, c in chain:
        inp[r, c] = fg
    occupied.update(chain)

    for v_seg, v_initial in v_ready[:v_gap_count]:
        v_opts = _v_gap_options(v_seg, out, fg, bg, occupied) or v_initial
        gap_start, anchor, dc = rng.choice(v_opts)
        col = v_seg["col"]
        chain = _v_chain_coords(col, gap_start, anchor, dc)
        inp[gap_start : gap_start + GAP_SIZE, col] = bg
        for r, c in chain:
            inp[r, c] = fg
        occupied.update(chain)

    forbidden: set[tuple[int, int]] = set()
    _reserve_neighborhood(forbidden, list(map(tuple, np.argwhere(inp == fg))))
    placed = 0
    distractor_options = _distractor_options(inp, fg, bg, forbidden)
    rng.shuffle(distractor_options)
    for twig, chain in distractor_options:
        if placed >= distractor_count:
            break
        coords = twig + chain
        if any(cell in forbidden for cell in coords):
            continue
        if not _cells_are_color(inp, coords, bg):
            continue
        for r, c in coords:
            inp[r, c] = fg
        _reserve_neighborhood(forbidden, coords)
        placed += 1
    if placed != distractor_count:
        return None

    try:
        solved, metrics = _solve_core(inp, return_metrics=True)
    except Exception:
        return None
    if not np.array_equal(solved, out):
        return None
    if metrics["unused_diag_components"] < distractor_count:
        return None
    if metrics["pruned_pixels"] < distractor_count * DISTRACTOR_TWIG_RANGE[0]:
        return None
    return inp


def generate_input() -> np.ndarray:
    for _ in range(GEN_RETRIES):
        grid = _gen_attempt()
        if grid is not None:
            return grid
    raise RuntimeError("failed to generate valid input for fused 7b80bb43")
