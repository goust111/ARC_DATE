# Combo variant of 7b80bb43 (index 14)
# combo_spec: D9/2+D15/0+D18/1
# Name: 小型边框锚定长短交替网络

from __future__ import annotations

import random

import numpy as np

from arc_utils import get_bg


# Design analysis for the fused variant
# a) Concrete behavior:
#    The clean output is a compact 10-14 square grid with three horizontal
#    segments and three vertical segments. Rows alternate long-short-long; cols
#    alternate short-long-short. Long segments are dual-anchored by a control
#    color on opposite border cells, while short segments are singly anchored.
#    Only the long segments receive 3-cell gaps plus 2-pixel diagonal hints.
#    Example: a top long row spans almost the full width between two control
#    markers, a middle short row reaches from one border marker to the long
#    central column, and a bottom long row mirrors the top row. Removing the
#    diagonal hints should restore only the dual-anchored rows/cols.
# b) Complexity calibration:
#    The template keeps only one control color, one line color, three long
#    candidates, and three short candidates. That is enough to make the border
#    coding and long/short alternation clearly visible without introducing many
#    unrelated layouts that would be hard to infer from 2-4 examples.
# c) Broken assumptions from the original task:
#    - The original solver assumed exactly one non-bg color; this variant adds a
#      border control color, so fg must be separated from ctrl explicitly.
#    - The original solver filled any small row/col gap near a diagonal chain;
#      here that would incorrectly modify short segments. We now rebuild only
#      the rows/cols with paired border anchors.
#    - The original generator relied on larger canvases and generic T-networks;
#      the fused small-grid layout needs a tighter, more structured template.
# d) Placement legality:
#    Every segment lies strictly inside the border except for its control marker.
#    Gap placement excludes junction columns/rows and keeps both remaining line
#    pieces non-empty. Diagonal hints and distractors are validated so they do
#    not overlap, leave the grid, or touch orthogonally.
# e) Visual observability:
#    Human-solvable cues repeat: paired anchors always mean a long segment,
#    single anchors always mean a short segment, and the row/col ordering shows
#    long-short-long vs short-long-short. The repaired output visibly restores a
#    compact anchor-guided lattice, not a generic free-floating line set.
# f) Visual cleanliness:
#    The network is intentionally sparse: one central long column, two long
#    rows, one short row, and two short side columns. Distractors are optional
#    and kept in open background only, so border markers, hints, and the true
#    network remain distinguishable.
# g) New solver assumptions audited against the generator:
#    - Border non-bg pixels all use the single control color.
#    - Long horizontal segments are exactly the rows with ctrl on both left and
#      right borders; long vertical segments are exactly the cols with ctrl on
#      both top and bottom borders.
#    - Gaps appear only on those long rows/cols, never on singly anchored short
#      segments. The generator enforces these constraints before oracle-checking
#      `main(inp) == out`.


GRID_SIZE_RANGE = (10, 14)
N_COLORS = 10
GAP_SIZE = 3
CHAIN_LENGTH = 2
DISTRACTOR_LENGTHS = (3, 2)
GEN_RETRIES = 400


def _deg4(mask: np.ndarray, r: int, c: int) -> int:
    h, w = mask.shape
    d = 0
    if r > 0 and bool(mask[r - 1, c]):
        d += 1
    if r + 1 < h and bool(mask[r + 1, c]):
        d += 1
    if c > 0 and bool(mask[r, c - 1]):
        d += 1
    if c + 1 < w and bool(mask[r, c + 1]):
        d += 1
    return d


def _has_diag(mask: np.ndarray, r: int, c: int) -> bool:
    h, w = mask.shape
    for dr, dc in ((-1, -1), (-1, 1), (1, -1), (1, 1)):
        rr = r + dr
        cc = c + dc
        if 0 <= rr < h and 0 <= cc < w and bool(mask[rr, cc]):
            return True
    return False


def _chain_coords_h(row: int, gap_start: int, anchor: str, dr: int) -> list[tuple[int, int]]:
    if anchor == "left":
        return [(row + dr * i, gap_start - 1 + i) for i in range(1, CHAIN_LENGTH + 1)]
    return [(row + dr * i, gap_start + GAP_SIZE - i) for i in range(1, CHAIN_LENGTH + 1)]


def _chain_coords_v(col: int, gap_start: int, anchor: str, dc: int) -> list[tuple[int, int]]:
    if anchor == "top":
        return [(gap_start + i, col + dc * (i + 1)) for i in range(CHAIN_LENGTH)]
    return [(gap_start + GAP_SIZE - 1 - i, col + dc * (i + 1)) for i in range(CHAIN_LENGTH)]


def _find_fg_and_ctrl(grid: np.ndarray, bg: int) -> tuple[int, int]:
    border_vals = set(map(int, grid[0, :])) | set(map(int, grid[-1, :])) | set(map(int, grid[:, 0])) | set(map(int, grid[:, -1]))
    border_vals.discard(int(bg))
    assert len(border_vals) == 1, "expected exactly one non-bg border control color"
    ctrl = int(next(iter(border_vals)))

    colors = set(map(int, np.unique(grid)))
    colors.discard(int(bg))
    colors.discard(int(ctrl))
    assert len(colors) == 1, "expected exactly one foreground line color"
    fg = int(next(iter(colors)))
    return fg, ctrl


def main(grid: np.ndarray) -> np.ndarray:
    """
    Remove diagonal hint staircases and rebuild the dual-anchored long segments.

    Input structure:
    - Small 10-14 square grid with bg, one line color fg, and one border control color ctrl.
    - The clean network is compact: horizontal rows alternate long-short-long and vertical columns alternate short-long-short.
    - Long segments are marked by ctrl on both opposite borders; short segments have only one border marker.
    - Only long segments contain 3-cell gaps, each accompanied by a 2-pixel diagonal fg staircase.

    Rule:
    - Treat ctrl as fixed border metadata, not as part of the line network.
    - Delete fg pixels that are diagonal-only hints, then restore every row/column whose border markers indicate a dual-anchored long segment.
    - Keep singly anchored short segments from the surviving fg mask and locally prune diagonal-adjacent leaf remnants near former hint sites.

    Key insight:
    Paired border markers encode which rows/cols are supposed to be long continuous lines, so the diagonal hints only tell you where to clean and reconnect, while the border controls tell you what must be fully restored.
    """
    g = np.array(grid, dtype=int)
    h, w = g.shape
    bg = int(get_bg(g))
    fg, ctrl = _find_fg_and_ctrl(g, bg)

    mask = g == fg

    diag = np.zeros((h, w), dtype=bool)
    for r in range(h):
        for c in range(w):
            if bool(mask[r, c]) and _deg4(mask, r, c) == 0 and _has_diag(mask, r, c):
                diag[r, c] = True

    near = np.zeros((h, w), dtype=bool)
    for r, c in np.argwhere(diag):
        r = int(r)
        c = int(c)
        for dr in (-1, 0, 1):
            for dc in (-1, 0, 1):
                rr = r + dr
                cc = c + dc
                if 0 <= rr < h and 0 <= cc < w:
                    near[rr, cc] = True

    long_rows = [r for r in range(h) if int(g[r, 0]) == ctrl and int(g[r, w - 1]) == ctrl]
    long_cols = [c for c in range(w) if int(g[0, c]) == ctrl and int(g[h - 1, c]) == ctrl]

    m = mask & (~diag)
    for r in long_rows:
        m[r, 1 : w - 1] = True
    for c in long_cols:
        m[1 : h - 1, c] = True

    changed = True
    while changed:
        changed = False
        to_clear: list[tuple[int, int]] = []
        for r in range(h):
            for c in range(w):
                if not bool(m[r, c]) or not bool(near[r, c]):
                    continue
                if _deg4(m, r, c) != 1:
                    continue
                if not _has_diag(m, r, c):
                    continue
                to_clear.append((r, c))
        if to_clear:
            changed = True
            for r, c in to_clear:
                m[r, c] = False

    out = np.full((h, w), bg, dtype=int)
    out[m] = fg
    out[g == ctrl] = ctrl
    return out


def _cells_are_bg(grid: np.ndarray, coords: list[tuple[int, int]], bg: int) -> bool:
    h, w = grid.shape
    for r, c in coords:
        if not (0 <= r < h and 0 <= c < w):
            return False
        if int(grid[r, c]) != bg:
            return False
    return True


def _validate_chain(grid: np.ndarray, coords: list[tuple[int, int]], fg: int, bg: int, occupied: set[tuple[int, int]]) -> bool:
    h, w = grid.shape
    coord_set = set(coords)
    if any(cell in occupied for cell in coords):
        return False
    for r, c in coords:
        if not (0 <= r < h and 0 <= c < w):
            return False
        if int(grid[r, c]) != bg:
            return False
        for rr, cc in ((r - 1, c), (r + 1, c), (r, c - 1), (r, c + 1)):
            if (rr, cc) in coord_set:
                return False
            if 0 <= rr < h and 0 <= cc < w and int(grid[rr, cc]) != bg:
                return False
        found_diag = False
        for rr, cc in ((r - 1, c - 1), (r - 1, c + 1), (r + 1, c - 1), (r + 1, c + 1)):
            if (rr, cc) in coord_set:
                found_diag = True
                break
            if 0 <= rr < h and 0 <= cc < w and int(grid[rr, cc]) == fg:
                found_diag = True
                break
        if not found_diag:
            return False
    return True


def _reserve_neighborhood(cells: set[tuple[int, int]], coords: list[tuple[int, int]]) -> None:
    for r, c in coords:
        for dr in (-1, 0, 1):
            for dc in (-1, 0, 1):
                cells.add((r + dr, c + dc))


def _h_gap_options(seg: dict, out: np.ndarray, fg: int, bg: int, occupied: set[tuple[int, int]]) -> list[tuple[int, str, int]]:
    row = int(seg["row"])
    c0 = int(seg["c0"])
    c1 = int(seg["c1"])
    blocked = set(map(int, seg["blocked"]))
    options: list[tuple[int, str, int]] = []
    for g0 in range(c0 + 2, c1 - GAP_SIZE):
        if g0 - c0 < 2 or c1 - (g0 + GAP_SIZE) + 1 < 2:
            continue
        if any(g0 <= b <= g0 + GAP_SIZE - 1 for b in blocked):
            continue
        temp = out.copy()
        temp[row, g0 : g0 + GAP_SIZE] = bg
        for anchor in ("left", "right"):
            for dr in (-1, 1):
                coords = _chain_coords_h(row, g0, anchor, dr)
                if _validate_chain(temp, coords, fg, bg, occupied):
                    options.append((g0, anchor, dr))
    return options


def _v_gap_options(seg: dict, out: np.ndarray, fg: int, bg: int, occupied: set[tuple[int, int]]) -> list[tuple[int, str, int]]:
    col = int(seg["col"])
    r0 = int(seg["r0"])
    r1 = int(seg["r1"])
    blocked = set(map(int, seg["blocked"]))
    options: list[tuple[int, str, int]] = []
    for g0 in range(r0 + 2, r1 - GAP_SIZE):
        if g0 - r0 < 2 or r1 - (g0 + GAP_SIZE) + 1 < 2:
            continue
        if any(g0 <= b <= g0 + GAP_SIZE - 1 for b in blocked):
            continue
        temp = out.copy()
        temp[g0 : g0 + GAP_SIZE, col] = bg
        for anchor in ("top", "bottom"):
            for dc in (-1, 1):
                coords = _chain_coords_v(col, g0, anchor, dc)
                if _validate_chain(temp, coords, fg, bg, occupied):
                    options.append((g0, anchor, dc))
    return options


def _distractor_options(inp: np.ndarray, bg: int, forbidden: set[tuple[int, int]]) -> list[list[tuple[int, int]]]:
    h, w = inp.shape
    options: list[list[tuple[int, int]]] = []
    for length in DISTRACTOR_LENGTHS:
        for r in range(1, h - length):
            for c in range(1, w - 1):
                for dc in (-1, 1):
                    coords = [(r + i, c + dc * i) for i in range(length)]
                    if any(cell in forbidden for cell in coords):
                        continue
                    if _cells_are_bg(inp, coords, bg) and _validate_chain(inp, coords, fg=9, bg=bg, occupied=set()):
                        options.append(coords)
    return options


def _choose_layout(rng: random.Random, size: int) -> dict | None:
    side = rng.choice(["left", "right"])
    for _ in range(80):
        top_row = rng.randint(3, 5)
        bottom_row = rng.randint(size - 6, size - 4)
        if bottom_row - top_row < 2:
            continue
        mid_lo = top_row + 1
        mid_hi = bottom_row - 1
        if mid_lo > mid_hi:
            continue
        mid_row = rng.randint(mid_lo, mid_hi)

        if side == "left":
            mid_col = rng.randint(3, 5)
            left_candidates = [c for c in range(1, mid_col - 1)]
            right_candidates = [c for c in range(max(mid_col + 2, size - 4), size - 1)]
        else:
            mid_col = rng.randint(size - 6, size - 4)
            left_candidates = [c for c in range(1, min(4, mid_col - 1))]
            right_candidates = [c for c in range(mid_col + 2, size - 1)]

        if not left_candidates or not right_candidates:
            continue

        left_col = rng.choice(left_candidates)
        right_col = rng.choice(right_candidates)
        if not (1 <= left_col < mid_col < right_col <= size - 2):
            continue

        short_h_len = mid_col if side == "left" else size - 1 - mid_col
        top_short_len = top_row
        bottom_short_len = size - 1 - bottom_row
        if not (3 <= short_h_len <= 5):
            continue
        if not (3 <= top_short_len <= 5 and 3 <= bottom_short_len <= 5):
            continue

        return {
            "side": side,
            "top_row": top_row,
            "mid_row": mid_row,
            "bottom_row": bottom_row,
            "left_col": left_col,
            "mid_col": mid_col,
            "right_col": right_col,
            "short_h_len": short_h_len,
            "top_short_len": top_short_len,
            "bottom_short_len": bottom_short_len,
        }
    return None


def _build_output(rng: random.Random, size: int, bg: int, fg: int, ctrl: int) -> tuple[np.ndarray, list[dict], list[dict], dict] | None:
    layout = _choose_layout(rng, size)
    if layout is None:
        return None

    out = np.full((size, size), bg, dtype=int)
    top_row = int(layout["top_row"])
    mid_row = int(layout["mid_row"])
    bottom_row = int(layout["bottom_row"])
    left_col = int(layout["left_col"])
    mid_col = int(layout["mid_col"])
    right_col = int(layout["right_col"])
    side = str(layout["side"])

    for row in (top_row, bottom_row):
        out[row, 0] = ctrl
        out[row, size - 1] = ctrl
        out[row, 1 : size - 1] = fg

    out[0, mid_col] = ctrl
    out[size - 1, mid_col] = ctrl
    out[1 : size - 1, mid_col] = fg

    if side == "left":
        out[mid_row, 0] = ctrl
        out[mid_row, 1 : mid_col + 1] = fg
    else:
        out[mid_row, size - 1] = ctrl
        out[mid_row, mid_col : size - 1] = fg

    out[0, left_col] = ctrl
    out[1 : top_row + 1, left_col] = fg

    out[size - 1, right_col] = ctrl
    out[bottom_row : size - 1, right_col] = fg

    h_segments = [
        {"row": top_row, "c0": 1, "c1": size - 2, "blocked": {left_col, mid_col}},
        {"row": bottom_row, "c0": 1, "c1": size - 2, "blocked": {mid_col, right_col}},
    ]
    v_segments = [
        {"col": mid_col, "r0": 1, "r1": size - 2, "blocked": {top_row, mid_row, bottom_row}},
    ]
    return out, h_segments, v_segments, layout


def _gen_attempt() -> np.ndarray | None:
    rng = random.Random()
    size = rng.randint(*GRID_SIZE_RANGE)
    bg = rng.randrange(N_COLORS)
    fg = rng.choice([c for c in range(N_COLORS) if c != bg])
    ctrl = rng.choice([c for c in range(N_COLORS) if c not in (bg, fg)])

    built = _build_output(rng, size, bg, fg, ctrl)
    if built is None:
        return None
    out, h_segments, v_segments, _layout = built

    inp = out.copy()
    occupied: set[tuple[int, int]] = set()
    segments: list[tuple[str, dict, list[tuple]]] = []

    for seg in h_segments:
        opts = _h_gap_options(seg, out, fg, bg, occupied)
        if opts:
            segments.append(("h", seg, opts))
    for seg in v_segments:
        opts = _v_gap_options(seg, out, fg, bg, occupied)
        if opts:
            segments.append(("v", seg, opts))

    if len(segments) < 2:
        return None

    rng.shuffle(segments)
    gap_target = rng.choice([2, 3]) if len(segments) >= 3 else 2
    selected = segments[:gap_target]

    for kind, seg, initial_opts in selected:
        if kind == "h":
            opts = _h_gap_options(seg, out, fg, bg, occupied) or initial_opts
            if not opts:
                return None
            gap_start, anchor, dr = rng.choice(opts)
            row = int(seg["row"])
            chain = _chain_coords_h(row, gap_start, anchor, dr)
            inp[row, gap_start : gap_start + GAP_SIZE] = bg
        else:
            opts = _v_gap_options(seg, out, fg, bg, occupied) or initial_opts
            if not opts:
                return None
            gap_start, anchor, dc = rng.choice(opts)
            col = int(seg["col"])
            chain = _chain_coords_v(col, gap_start, anchor, dc)
            inp[gap_start : gap_start + GAP_SIZE, col] = bg
        for r, c in chain:
            inp[r, c] = fg
        occupied.update(chain)

    forbidden: set[tuple[int, int]] = set()
    _reserve_neighborhood(forbidden, list(map(tuple, np.argwhere(inp != bg))))
    distractor_options = _distractor_options(inp, bg, forbidden)
    if distractor_options and rng.random() < 0.5:
        coords = rng.choice(distractor_options)
        for r, c in coords:
            inp[r, c] = fg

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
    raise RuntimeError("failed to generate valid input for fused 7b80bb43 combo")
