from __future__ import annotations
import numpy as np
from arc_utils import components_mask, get_bg
import random
from arc_utils import draw_line, find_components

def main(grid: np.ndarray) -> np.ndarray:
    """
    Remove diagonal staircase noise and complete broken straight lines.

    Input structure:
    - bg = background; one foreground color fg forming axis-aligned lines.
    - Some fg pixels form 45° diagonal chains (8-connected but 4-disconnected from lines).

    Rule:
    - Delete all diagonal-only fg pixels (those with no orthogonal fg neighbor, only diagonal ones).
    - Fill short gaps in horizontal/vertical lines where a deleted diagonal was nearby.
    - Prune remaining diagonal-adjacent leaf pixels near former diagonal sites.

    Key insight: Diagonal pixels are hints marking where a straight line is broken; their span sets the maximum gap size to fill.
    """
    g = np.array(grid, dtype=int)
    bg = int(get_bg(g))

    colors = {int(x) for x in np.unique(g)}
    colors.discard(int(bg))
    assert not (not colors)
    fg = int(next(iter(colors)))

    mask = g == int(fg)
    H, W = g.shape

    def deg4(m: np.ndarray, r: int, c: int) -> int:
        d = 0
        if r > 0 and bool(m[int(r - 1), int(c)]):
            d += 1
        if r + 1 < H and bool(m[int(r + 1), int(c)]):
            d += 1
        if c > 0 and bool(m[int(r), int(c - 1)]):
            d += 1
        if c + 1 < W and bool(m[int(r), int(c + 1)]):
            d += 1
        return int(d)

    def has_diag(m: np.ndarray, r: int, c: int) -> bool:
        if r > 0 and c > 0 and bool(m[int(r - 1), int(c - 1)]):
            return True
        if r > 0 and c + 1 < W and bool(m[int(r - 1), int(c + 1)]):
            return True
        if r + 1 < H and c > 0 and bool(m[int(r + 1), int(c - 1)]):
            return True
        if r + 1 < H and c + 1 < W and bool(m[int(r + 1), int(c + 1)]):
            return True
        return False

    # --- diagonal-only pixels (to remove) ---
    diag = np.zeros((H, W), dtype=bool)
    for r in range(H):
        for c in range(W):
            if bool(mask[int(r), int(c)]) and deg4(mask, int(r), int(c)) == 0 and has_diag(mask, int(r), int(c)):
                diag[int(r), int(c)] = True

    # Neighborhood of diagonal hints (Chebyshev radius 1).
    near = np.zeros((H, W), dtype=bool)
    for r in range(H):
        for c in range(W):
            if not bool(diag[int(r), int(c)]):
                continue
            for dr in (-1, 0, 1):
                for dc in (-1, 0, 1):
                    rr = int(r + dr)
                    cc = int(c + dc)
                    if 0 <= rr < H and 0 <= cc < W:
                        near[rr, cc] = True

    out = g.copy()
    out[diag] = int(bg)

    # Gap threshold: small default, optionally increased by diagonal-component span.
    MAXG = 3
    if bool(diag.any()):
        for coords in components_mask(diag, connectivity=8):
            if coords.size == 0:
                continue
            roots = set()
            for r, c in coords:
                for dr, dc in ((-1, -1), (-1, 1), (1, -1), (1, 1)):
                    rr, cc = int(r + dr), int(c + dc)
                    if 0 <= rr < H and 0 <= cc < W and bool(mask[rr, cc]) and not bool(diag[rr, cc]):
                        roots.add((rr, cc))
            assert len(coords) + len(roots) <= 3, f"diagonal chain too large ({len(coords)} diag + {len(roots)} root)"
            r0, c0 = coords.min(axis=0)
            r1, c1 = coords.max(axis=0)
            MAXG = max(int(MAXG), int(r1 - r0 + 1), int(c1 - c0 + 1))

    # Fill short horizontal gaps between segments when a diagonal hint is nearby.
    for r in range(H):
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

        for (a1, b1), (a2, _b2) in zip(segs, segs[1:]):
            gap0 = int(b1 + 1)
            gap1 = int(a2 - 1)
            if gap1 < gap0:
                continue
            if int(gap1 - gap0 + 1) > int(MAXG):
                continue
            rr0 = max(0, int(r - 1))
            rr1 = min(int(H - 1), int(r + 1))
            cc0 = max(0, int(gap0 - 1))
            cc1 = min(int(W - 1), int(gap1 + 1))
            if bool(diag[rr0 : rr1 + 1, cc0 : cc1 + 1].any()):
                out[int(r), int(gap0) : int(gap1) + 1] = int(fg)

    # Fill short vertical gaps between segments when a diagonal hint is nearby.
    for c in range(W):
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

        for (a1, b1), (a2, _b2) in zip(segs, segs[1:]):
            gap0 = int(b1 + 1)
            gap1 = int(a2 - 1)
            if gap1 < gap0:
                continue
            if int(gap1 - gap0 + 1) > int(MAXG):
                continue
            rr0 = max(0, int(gap0 - 1))
            rr1 = min(int(H - 1), int(gap1 + 1))
            cc0 = max(0, int(c - 1))
            cc1 = min(int(W - 1), int(c + 1))
            if bool(diag[rr0 : rr1 + 1, cc0 : cc1 + 1].any()):
                out[int(gap0) : int(gap1) + 1, int(c)] = int(fg)

    # Local pruning: remove diagonal-adjacent leaf pixels only near diagonal hints.
    m = out == int(fg)
    changed = True
    while changed:
        changed = False
        to_clear: list[tuple[int, int]] = []
        for r in range(H):
            for c in range(W):
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

    out2 = np.full((H, W), int(bg), dtype=int)
    out2[m] = int(fg)
    return out2

# --------------- generator (attempt_2) ---------------

# --- Configuration constants ---

# Grid dimensions
GRID_H_RANGE = (16, 29)           # row count bounds
GRID_W_RANGE = (17, 30)           # column count bounds
N_COLORS = 10                     # ARC color palette size

# Segment network topology (h + v = segments; junctions = T-junction count)
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
T_SLOTS = ("LT", "RT", "LB", "RB")  # T-junction placement quadrants

# Layout spacing
EDGE_MARGIN = 3                   # min distance from grid edge for branch columns / row pools
ROW_COL_GAP = 2                   # minimum separation between distinct row/column picks
T_BRANCH_OFFSET_MIN = 3           # min col distance from branch to H-segment free end
T_BRANCH_OFFSET_MAX = 9           # max col distance from branch to H-segment free end
H_SEG_END_MARGIN = 4              # margin from opposite grid edge for H-segment free end
H_SEG_MAX_END = 25                # max absolute col for left-anchored H-segment free end

# Simple H-segment sizing
SIMPLE_H_LEN_MIN = 7              # min length of simple (non-junction) horizontal segment
SIMPLE_H_LEN_MAX = 11             # max length of simple horizontal segment
SIMPLE_V_MARGIN_NEAR = 7          # col pool margin on the side with a simple H-segment
SIMPLE_V_MARGIN_FAR = 3           # col pool margin on the side without a simple H-segment (or edge)

# Gap / chain placement
GAP_SIZE = 3                      # pixels removed from a segment to create a gap
CHAIN_LENGTH = 2                  # diagonal chain pixels per gap
GAP_BLOCKED_LO = 2               # gap exclusion zone below blocked position
GAP_BLOCKED_HI = 4               # gap exclusion zone above blocked position
GAP_SEG_INSET = 2                # min distance from segment start to first gap candidate

# Distractor chains
DISTRACTOR_LENGTHS = (3, 2)      # candidate distractor diagonal lengths (tried in order)
DISTRACTOR_MARGIN = 1            # min distance from grid edge for distractor start

# Retry budget
GEN_RETRIES = 300                 # max attempts before raising


# --- Helpers ---

def _pick_distinct(rng: random.Random, values: list[int], k: int, gap: int = ROW_COL_GAP) -> list[int]:
    """Select k values from *values* such that all pairs differ by >= gap."""
    vals = values[:]
    rng.shuffle(vals)
    picked: list[int] = []
    for v in vals:
        if all(abs(v - p) >= gap for p in picked):
            picked.append(v)
            if len(picked) == k:
                return sorted(picked)
    return []


def _chain_coords(row: int, gap_start: int, anchor: str) -> list[tuple[int, int]]:
    """Diagonal chain coordinates for a horizontal gap."""
    if anchor == "left":
        return [(row + i, gap_start - 1 + i) for i in range(1, CHAIN_LENGTH + 1)]
    return [(row + i, gap_start + GAP_SIZE - i) for i in range(1, CHAIN_LENGTH + 1)]


def _v_chain_coords(col: int, gap_start: int, anchor: str, dc: int) -> list[tuple[int, int]]:
    """Diagonal chain coordinates for a vertical gap."""
    if anchor == "top":
        return [(gap_start + i, col + dc * (i + 1)) for i in range(CHAIN_LENGTH)]
    return [(gap_start + GAP_SIZE - 1 - i, col + dc * (i + 1)) for i in range(CHAIN_LENGTH)]


def _cells_are_color(grid: np.ndarray, coords: list[tuple[int, int]], color: int) -> bool:
    h, w = grid.shape
    for r, c in coords:
        if not (0 <= r < h and 0 <= c < w) or grid[r, c] != color:
            return False
    return True


def _validate_chain(grid: np.ndarray, coords: list[tuple[int, int]], fg: int,
                    occupied: set[tuple[int, int]]) -> bool:
    """Check that chain coords sit on bg, have no ortho fg neighbor, and have diagonal fg."""
    h, w = grid.shape
    coord_set = set(coords)
    if any(cell in occupied for cell in coords):
        return False
    for r, c in coords:
        # Orthogonal neighbors must not be fg (or another chain pixel)
        for rr, cc in ((r - 1, c), (r + 1, c), (r, c - 1), (r, c + 1)):
            if (rr, cc) in coord_set:
                return False
            if 0 <= rr < h and 0 <= cc < w and grid[rr, cc] == fg:
                return False
        # At least one diagonal neighbor must be fg or another chain pixel
        has_diag = False
        for rr, cc in ((r - 1, c - 1), (r - 1, c + 1), (r + 1, c - 1), (r + 1, c + 1)):
            if (rr, cc) in coord_set or (0 <= rr < h and 0 <= cc < w and grid[rr, cc] == fg):
                has_diag = True
                break
        if not has_diag:
            return False
    return True


def _place_t_component(out: np.ndarray, fg: int, slot: str, row: int,
                       branch_col: int, end_col: int) -> tuple[dict, dict]:
    """Draw a T-junction (one H + one V segment sharing one pixel) and return segment dicts."""
    H, W = out.shape
    # H-segment spans: left-side slots start at col 0, right-side at end_col
    h_c0 = 0 if slot[0] == "L" else end_col
    h_c1 = end_col if slot[0] == "L" else W - 1
    # V-segment spans: top slots start at row 0, bottom at the junction row
    v_r0 = 0 if slot[1] == "T" else row
    v_r1 = row if slot[1] == "T" else H - 1
    draw_line(out, row, h_c0, end_r=row, end_c=h_c1, color=fg)
    draw_line(out, v_r0, branch_col, end_r=v_r1, end_c=branch_col, color=fg)
    h_seg = {"row": row, "c0": h_c0, "c1": h_c1, "blocked": {branch_col}}
    v_seg = {"col": branch_col, "r0": v_r0, "r1": v_r1, "blocked": {row}}
    return h_seg, v_seg


def _place_simple_h(out: np.ndarray, fg: int, side: str, row: int, length: int) -> dict:
    """Draw a simple horizontal segment anchored to one grid edge."""
    W = out.shape[1]
    c0 = 0 if side == "left" else W - length
    c1 = length - 1 if side == "left" else W - 1
    draw_line(out, row, c0, end_r=row, end_c=c1, color=fg)
    return {"row": row, "c0": c0, "c1": c1, "blocked": set()}


def _place_simple_v(out: np.ndarray, fg: int, side: str, col: int, length: int) -> dict:
    """Draw a simple vertical segment anchored to one grid edge."""
    H = out.shape[0]
    r0 = 0 if side == "top" else H - length
    r1 = length - 1 if side == "top" else H - 1
    draw_line(out, r0, col, end_r=r1, end_c=col, color=fg)
    return {"col": col, "r0": r0, "r1": r1, "blocked": set()}


def _h_gap_options(seg: dict, out: np.ndarray, fg: int, bg: int,
                   occupied: set[tuple[int, int]]) -> list[tuple[int, str]]:
    """Find valid (gap_start, anchor) pairs for cutting a horizontal segment."""
    row, c0, c1 = seg["row"], seg["c0"], seg["c1"]
    blocked = seg["blocked"]
    options: list[tuple[int, str]] = []
    for g in range(c0 + GAP_SEG_INSET, c1 - GAP_SIZE - 1):
        if any(g - GAP_BLOCKED_LO <= b <= g + GAP_BLOCKED_HI for b in blocked):
            continue
        temp = out.copy()
        temp[row, g:g + GAP_SIZE] = bg
        for anchor in ("left", "right"):
            coords = _chain_coords(row, g, anchor)
            if not _cells_are_color(temp, coords, bg):
                continue
            if not _validate_chain(temp, coords, fg, occupied):
                continue
            options.append((g, anchor))
    return options


def _v_gap_options(seg: dict, out: np.ndarray, fg: int, bg: int,
                   occupied: set[tuple[int, int]]) -> list[tuple[int, str, int]]:
    """Find valid (gap_start, anchor, dc) triples for cutting a vertical segment."""
    col, r0, r1 = seg["col"], seg["r0"], seg["r1"]
    blocked = seg["blocked"]
    options: list[tuple[int, str, int]] = []
    for g in range(r0 + GAP_SEG_INSET, r1 - GAP_SIZE - 1):
        if any(g - GAP_BLOCKED_LO <= b <= g + GAP_BLOCKED_HI for b in blocked):
            continue
        temp = out.copy()
        temp[g:g + GAP_SIZE, col] = bg
        for anchor in ("top", "bottom"):
            for dc in (-1, 1):
                coords = _v_chain_coords(col, g, anchor, dc)
                if not _cells_are_color(temp, coords, bg):
                    continue
                if not _validate_chain(temp, coords, fg, occupied):
                    continue
                options.append((g, anchor, dc))
    return options


def _reserve_neighborhood(cells: set[tuple[int, int]], coords) -> None:
    """Add Chebyshev-1 neighborhood of all coords to the set."""
    for r, c in coords:
        for dr in (-1, 0, 1):
            for dc in (-1, 0, 1):
                cells.add((r + dr, c + dc))


def _distractor_options(inp: np.ndarray, bg: int,
                        forbidden: set[tuple[int, int]]) -> list[list[tuple[int, int]]]:
    """Enumerate diagonal chain placements in open background space."""
    h, w = inp.shape
    options: list[list[tuple[int, int]]] = []
    for length in DISTRACTOR_LENGTHS:
        for r in range(DISTRACTOR_MARGIN, h - length):
            for c in range(DISTRACTOR_MARGIN, w - DISTRACTOR_MARGIN):
                for direction in (-1, 1):
                    coords = [(r + i, c + direction * i) for i in range(length)]
                    if not _cells_are_color(inp, coords, bg):
                        continue
                    if any(cell in forbidden for cell in coords):
                        continue
                    options.append(coords)
    return options


# --- Output network builder ---

def _build_output(rng: random.Random, h: int, w: int, bg: int, fg: int,
                  cfg: dict) -> tuple[np.ndarray, list[dict], list[dict]] | None:
    """Build the clean output grid with H/V segment network. Returns None on layout failure."""
    out = np.full((h, w), bg, dtype=int)
    h_segments: list[dict] = []
    v_segments: list[dict] = []

    h_simple = cfg["h"] - cfg["junctions"]
    v_simple = cfg["v"] - cfg["junctions"]
    t_slots = rng.sample(list(T_SLOTS), cfg["junctions"])

    # Assign rows to T-junction slots (top half vs bottom half)
    top_slots = [s for s in t_slots if s[1] == "T"]
    bottom_slots = [s for s in t_slots if s[1] == "B"]
    top_pool = list(range(EDGE_MARGIN, max(EDGE_MARGIN + 1, h // 2 - 2)))
    bottom_pool = list(range(max(h // 2 + 1, EDGE_MARGIN + 2), h - EDGE_MARGIN))
    top_rows = _pick_distinct(rng, top_pool, len(top_slots))
    bottom_rows = _pick_distinct(rng, bottom_pool, len(bottom_slots))
    if len(top_rows) != len(top_slots) or len(bottom_rows) != len(bottom_slots):
        return None
    row_by_slot = dict(zip(top_slots, top_rows)) | dict(zip(bottom_slots, bottom_rows))

    # Assign columns to T-junction branches
    branch_cols = _pick_distinct(rng, list(range(EDGE_MARGIN, w - EDGE_MARGIN)), len(t_slots))
    if len(branch_cols) != len(t_slots):
        return None
    rng.shuffle(branch_cols)
    branch_by_slot = dict(zip(t_slots, branch_cols))

    # Place T-junction components
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

    # Determine center band for simple H-segments
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

    # Pick columns for simple V-segments, avoiding branch columns
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
        col_pool = [c for c in col_pool if c not in used_branch_cols]
        simple_v_cols = _pick_distinct(rng, col_pool, v_simple)
        if len(simple_v_cols) != v_simple:
            return None

    # Place simple H-segments
    for row, side in zip(simple_h_rows, h_sides):
        if side == "left":
            max_len = (min(simple_v_cols) if simple_v_cols else w - EDGE_MARGIN) - 1
        else:
            max_len = w - (max(simple_v_cols) if simple_v_cols else 2) - 2
        max_len = min(SIMPLE_H_LEN_MAX, max_len)
        if max_len < SIMPLE_H_LEN_MIN:
            return None
        h_segments.append(_place_simple_h(out, fg, side, row, rng.randint(SIMPLE_H_LEN_MIN, max_len)))

    # Place simple V-segments (first batch from top, rest from bottom)
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

    # Validate component count
    comps = find_components(out, bg=bg, connectivity=4, monochromatic=True, return_format="dict")
    if len(comps) != cfg["segments"] - cfg["junctions"]:
        return None
    return out, h_segments, v_segments


# --- Main generation attempt ---

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

    # Collect gap candidates per segment
    h_ready = [(seg, opts) for seg in h_segments
               for opts in [_h_gap_options(seg, out, fg, bg, occupied)] if opts]
    v_ready = [(seg, opts) for seg in v_segments
               for opts in [_v_gap_options(seg, out, fg, bg, occupied)] if opts]
    if not h_ready or not v_ready:
        return None

    # 1 H gap mandatory, rest from V (eval pattern: 1H + 1-2V)
    v_gap_count = gap_count - 1
    if len(v_ready) < v_gap_count:
        return None
    rng.shuffle(h_ready)
    rng.shuffle(v_ready)

    # Apply H gap
    h_seg, h_initial = h_ready[0]
    h_opts = _h_gap_options(h_seg, out, fg, bg, occupied) or h_initial
    gap_start, anchor = rng.choice(h_opts)
    row = h_seg["row"]
    chain = _chain_coords(row, gap_start, anchor)
    inp[row, gap_start:gap_start + GAP_SIZE] = bg
    for r, c in chain:
        inp[r, c] = fg
    occupied.update(chain)

    # Apply V gaps
    for v_seg, v_initial in v_ready[:v_gap_count]:
        v_opts = _v_gap_options(v_seg, out, fg, bg, occupied) or v_initial
        gap_start, anchor, dc = rng.choice(v_opts)
        col = v_seg["col"]
        chain = _v_chain_coords(col, gap_start, anchor, dc)
        inp[gap_start:gap_start + GAP_SIZE, col] = bg
        for r, c in chain:
            inp[r, c] = fg
        occupied.update(chain)

    # Place distractor chains in open background
    forbidden: set[tuple[int, int]] = set()
    _reserve_neighborhood(forbidden, list(map(tuple, np.argwhere(inp == fg))))
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
            inp[r, c] = fg
        _reserve_neighborhood(forbidden, coords)
        placed += 1

    # Oracle check
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
    raise RuntimeError("failed to generate valid input for 7b80bb43")
