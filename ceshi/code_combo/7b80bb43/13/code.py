# Combo variant of 7b80bb43 (index 13)
# combo_spec: D10/3+D15/1+D18/0
# Name: 大型等长密集间隙网络

from __future__ import annotations

import random

import numpy as np

from arc_utils import draw_line, find_components, get_bg


# Design analysis for the fused variant
# -------------------------------------
# a) Concrete behavior and trace
#    - Original task: a small-to-medium single-color line network contains 2-3 broken
#      places. Each true break is accompanied by a tiny 45-degree staircase of the same
#      foreground color. The solver deletes diagonal-only pixels, fills the 3-cell gap
#      beside them, then trims any tiny leftover diagonal leaves.
#    - Fused task: the local rule is unchanged, but the board is much larger and the
#      clean output is deliberately regular. The generator builds four symmetric T-shaped
#      components whose horizontal and vertical segments all have the same length L, with
#      L chosen around 10-15. Then it cuts 4-6 different 3-cell gaps across both axes
#      and adds a dense collection of 2-pixel diagonal hint chains.
#    - Pixel-level trace example:
#      original: a row segment `xxxx___xxxx` plus a diagonal chain just below one gap
#      endpoint becomes `xxxxxxxxxxx` after the chain is removed and the 3 blank cells
#      are restored.
#      fused: the exact same local repair happens, but now the same sample also contains
#      several more row/column breaks elsewhere, and every intact output segment shares
#      the same length. The eye sees a large, regular, heavily fractured network rather
#      than a sparse small one.
#    - Difference: the transformation logic remains topological, while the generator
#      pushes scale, gap count, and output regularity so the visual regime is clearly
#      different from the seed.
#
# b) Complexity calibration
#    - The variant must look visibly denser without becoming unreadable. Four symmetric
#      corner T-components give a repeated visual grammar: same length, same thickness,
#      same corner anchoring, repeated across the board.
#    - Gap count is restricted to 4-6 and split across horizontal and vertical segments.
#      That is enough to make the input look intentionally broken, but still small enough
#      that each repaired place can be visually tracked by a human.
#    - All segments share one length per sample. That makes the output noticeably more
#      uniform and lets a human use "every branch should end up equally long" as a
#      confirmation cue without introducing a new symbolic rule.
#
# c) Hidden-assumption audit against the original solver
#    - The old `assert len(coords) + len(roots) <= 3` assumed every diagonal component
#      was tiny and isolated. Dense hint placement can still satisfy that, but the solver
#      no longer needs the assertion, so the new `main()` drops it.
#    - The old gap threshold was derived from diagonal component span. That coupling is
#      unnecessary here because the generator still uses the original 3-cell gap size.
#      The new solver therefore repairs only exact 3-cell gaps, which removes a brittle
#      dependency on how close two diagonal hints happen to be.
#    - Row/column repair loops implicitly assume that unrelated components on the same
#      line are separated by more than the legal gap size. The new generator guarantees
#      wide central empty bands between left/right and top/bottom components, so a 3-cell
#      scan cannot accidentally bridge independent structures.
#
# d) Placement legality
#    - Output is built first. Each component is an edge-anchored T with one horizontal
#      and one vertical segment, both length L, sharing exactly one junction pixel.
#    - Gap sites are selected only from validated interior positions: the 3-cell hole
#      stays away from endpoints and from the junction's blocked coordinate, and the
#      diagonal chain must lie fully on background with no orthogonal contact to any line.
#    - Hint neighborhoods are reserved after each placement, so true hints do not collide
#      with other hints or distractors. Distractors are added only in open background.
#
# e) Visual observability
#    - In 2-4 examples a human can see the same repair rule repeating many times:
#      remove the slanted pixels, then complete the straight run.
#    - The fused dimensions are also directly visible: the boards are around 30-40 in
#      each direction, there are clearly more repair sites than in the source task, and
#      the cleaned output has equal branch lengths everywhere.
#    - No invisible property such as numeric color identity is used. All inference comes
#      from visible geometry: axis-aligned lines versus diagonal staircase hints.
#
# f) Cleanliness
#    - The generator uses one foreground color and large empty central corridors, so the
#      image stays legible even with more breaks.
#    - Symmetric corner placement prevents the big board from turning into random clutter.
#      Dense hints are present, but each sits next to a single local break or in clearly
#      isolated background when acting as a distractor.
#
# g) New solver assumptions and generator guarantees
#    - Assumption: true breaks are always exactly 3 cells long.
#      Guarantee: every generated gap uses `GAP_SIZE = 3`.
#    - Assumption: diagonal hint pixels are foreground pixels with zero orthogonal degree
#      and at least one diagonal neighbor.
#      Guarantee: both repair hints and distractors are validated against that topology.
#    - Assumption: a 3-cell blank run should be filled only if a diagonal hint exists in
#      its immediate Chebyshev-1 neighborhood.
#      Guarantee: every real gap gets such a hint, while unrelated components are spaced
#      much farther apart than 3 cells.


GRID_H_RANGE = (30, 40)
GRID_W_RANGE = (30, 40)
N_COLORS = 10

SEG_LEN_MIN = 10
SEG_LEN_MAX = 15
GAP_SIZE = 3
CHAIN_LENGTH = 2
GAP_COUNT_RANGE = (4, 6)
DISTRACTOR_COUNT_RANGE = (1, 2)

GAP_SEG_INSET = 2
GAP_BLOCKED_LO = 2
GAP_BLOCKED_HI = 4

DISTRACTOR_LENGTHS = (3, 2)
DISTRACTOR_MARGIN = 1
GEN_RETRIES = 1200


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
    return int(deg)


def _has_diag(mask: np.ndarray, r: int, c: int) -> bool:
    h, w = mask.shape
    for dr in (-1, 1):
        for dc in (-1, 1):
            rr = int(r + dr)
            cc = int(c + dc)
            if 0 <= rr < h and 0 <= cc < w and bool(mask[rr, cc]):
                return True
    return False


def _runs(indices: np.ndarray) -> list[tuple[int, int]]:
    if indices.size == 0:
        return []
    vals = [int(x) for x in indices]
    out: list[tuple[int, int]] = []
    start = vals[0]
    prev = vals[0]
    for cur in vals[1:]:
        if cur == prev + 1:
            prev = cur
        else:
            out.append((int(start), int(prev)))
            start = prev = cur
    out.append((int(start), int(prev)))
    return out


def main(grid: np.ndarray) -> np.ndarray:
    """
    Remove dense diagonal staircase hints and restore the regular line network.

    Input structure:
    - The board is a large 30-40 sized grid with one background color and one foreground color.
    - Foreground pixels form long axis-aligned segments of roughly equal length, broken by many 3-cell gaps.
    - Extra foreground pixels appear only as 45-degree staircase hints or isolated distractor staircases.

    Rule:
    - Delete every diagonal-only foreground pixel: no orthogonal foreground neighbor, but at least one diagonal foreground neighbor.
    - Refill a horizontal or vertical gap only when it is exactly 3 cells wide/tall and a removed diagonal hint touched that gap's local neighborhood.
    - After repairing the straight runs, prune diagonal-adjacent leaf pixels near former hint sites to recover the clean rectilinear output.

    Key insight:
    - The fusion changes scale and density, not the local reasoning. Diagonal pixels are still disposable clues that point to nearby 3-cell breaks, while the equal segment lengths simply make the final repaired structure look more uniform.
    """
    g = np.array(grid, dtype=int)
    bg = int(get_bg(g))

    colors = {int(x) for x in np.unique(g)}
    colors.discard(int(bg))
    assert colors, "grid must contain foreground pixels"
    assert len(colors) == 1, "this variant expects a single foreground color"
    fg = int(next(iter(colors)))

    mask = g == int(fg)
    h, w = g.shape

    diag = np.zeros((h, w), dtype=bool)
    for r in range(h):
        for c in range(w):
            if bool(mask[r, c]) and _deg4(mask, int(r), int(c)) == 0 and _has_diag(mask, int(r), int(c)):
                diag[r, c] = True

    near = np.zeros((h, w), dtype=bool)
    for r, c in np.argwhere(diag):
        r = int(r)
        c = int(c)
        for dr in (-1, 0, 1):
            for dc in (-1, 0, 1):
                rr = int(r + dr)
                cc = int(c + dc)
                if 0 <= rr < h and 0 <= cc < w:
                    near[rr, cc] = True

    repaired = mask & (~diag)

    for r in range(h):
        cols = np.where(repaired[r, :])[0]
        segs = _runs(cols)
        for (_a1, b1), (a2, _b2) in zip(segs, segs[1:]):
            gap0 = int(b1 + 1)
            gap1 = int(a2 - 1)
            if gap1 < gap0 or int(gap1 - gap0 + 1) != GAP_SIZE:
                continue
            rr0 = max(0, int(r - 1))
            rr1 = min(int(h - 1), int(r + 1))
            cc0 = max(0, int(gap0 - 1))
            cc1 = min(int(w - 1), int(gap1 + 1))
            if bool(diag[rr0 : rr1 + 1, cc0 : cc1 + 1].any()):
                repaired[r, gap0 : gap1 + 1] = True

    for c in range(w):
        rows = np.where(repaired[:, c])[0]
        segs = _runs(rows)
        for (_a1, b1), (a2, _b2) in zip(segs, segs[1:]):
            gap0 = int(b1 + 1)
            gap1 = int(a2 - 1)
            if gap1 < gap0 or int(gap1 - gap0 + 1) != GAP_SIZE:
                continue
            rr0 = max(0, int(gap0 - 1))
            rr1 = min(int(h - 1), int(gap1 + 1))
            cc0 = max(0, int(c - 1))
            cc1 = min(int(w - 1), int(c + 1))
            if bool(diag[rr0 : rr1 + 1, cc0 : cc1 + 1].any()):
                repaired[gap0 : gap1 + 1, c] = True

    changed = True
    while changed:
        changed = False
        to_clear: list[tuple[int, int]] = []
        for r in range(h):
            for c in range(w):
                if not bool(repaired[r, c]) or not bool(near[r, c]):
                    continue
                if _deg4(repaired, int(r), int(c)) != 1:
                    continue
                if not _has_diag(repaired, int(r), int(c)):
                    continue
                to_clear.append((int(r), int(c)))
        if to_clear:
            changed = True
            for r, c in to_clear:
                repaired[r, c] = False

    out = np.full((h, w), int(bg), dtype=int)
    out[repaired] = int(fg)
    return out


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
        if not (0 <= r < h and 0 <= c < w):
            return False
        if int(grid[r, c]) != int(color):
            return False
    return True


def _validate_chain(
    grid: np.ndarray,
    coords: list[tuple[int, int]],
    fg: int,
    reserved: set[tuple[int, int]],
) -> bool:
    h, w = grid.shape
    coord_set = set(coords)
    if any(cell in reserved for cell in coords):
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


def _reserve_neighborhood(cells: set[tuple[int, int]], coords: list[tuple[int, int]]) -> None:
    for r, c in coords:
        for dr in (-1, 0, 1):
            for dc in (-1, 0, 1):
                cells.add((int(r + dr), int(c + dc)))


def _distractor_options(inp: np.ndarray, bg: int, forbidden: set[tuple[int, int]]) -> list[list[tuple[int, int]]]:
    h, w = inp.shape
    options: list[list[tuple[int, int]]] = []
    for length in DISTRACTOR_LENGTHS:
        for r in range(DISTRACTOR_MARGIN, h - length):
            for c in range(DISTRACTOR_MARGIN, w - DISTRACTOR_MARGIN):
                for dc in (-1, 1):
                    coords = [(int(r + i), int(c + dc * i)) for i in range(length)]
                    if not _cells_are_color(inp, coords, bg):
                        continue
                    if any(cell in forbidden for cell in coords):
                        continue
                    temp = inp.copy()
                    for rr, cc in coords:
                        temp[rr, cc] = 1
                    mask = temp == 1
                    if not all(_deg4(mask, rr, cc) == 0 and _has_diag(mask, rr, cc) for rr, cc in coords):
                        continue
                    options.append(coords)
    return options


def _place_t_component(
    out: np.ndarray,
    fg: int,
    row: int,
    col: int,
    h_side: str,
    v_side: str,
    seg_len: int,
) -> tuple[dict, dict]:
    h, w = out.shape
    if h_side == "left":
        c0, c1 = 0, seg_len - 1
    else:
        c0, c1 = w - seg_len, w - 1
    if v_side == "top":
        r0, r1 = 0, seg_len - 1
    else:
        r0, r1 = h - seg_len, h - 1
    draw_line(out, row, c0, end_r=row, end_c=c1, color=fg)
    draw_line(out, r0, col, end_r=r1, end_c=col, color=fg)
    h_seg = {"row": int(row), "c0": int(c0), "c1": int(c1), "blocked": {int(col)}}
    v_seg = {"col": int(col), "r0": int(r0), "r1": int(r1), "blocked": {int(row)}}
    return h_seg, v_seg


def _build_output(
    rng: random.Random,
    h: int,
    w: int,
    bg: int,
    fg: int,
) -> tuple[np.ndarray, list[dict], list[dict], int] | None:
    max_len = min(SEG_LEN_MAX, (h - 4) // 2, (w - 4) // 2)
    if max_len < SEG_LEN_MIN:
        return None
    seg_len = rng.randint(SEG_LEN_MIN, max_len)
    if seg_len < 7:
        return None

    offset_min = 3
    offset_max = seg_len - 4
    if offset_min > offset_max:
        return None
    row_offset = rng.randint(offset_min, offset_max)
    col_offset = rng.randint(offset_min, offset_max)

    top_row = int(row_offset)
    bottom_row = int(h - seg_len + row_offset)
    left_col = int(col_offset)
    right_col = int(w - seg_len + col_offset)

    out = np.full((h, w), int(bg), dtype=int)
    h_segments: list[dict] = []
    v_segments: list[dict] = []

    placements = [
        (top_row, left_col, "left", "top"),
        (top_row, right_col, "right", "top"),
        (bottom_row, left_col, "left", "bottom"),
        (bottom_row, right_col, "right", "bottom"),
    ]
    for row, col, h_side, v_side in placements:
        h_seg, v_seg = _place_t_component(out, fg, row, col, h_side, v_side, seg_len)
        h_segments.append(h_seg)
        v_segments.append(v_seg)

    comps = find_components(out, bg=bg, connectivity=4, monochromatic=True, return_format="dict")
    if len(comps) != 4:
        return None
    return out, h_segments, v_segments, int(seg_len)


def _h_gap_options(
    seg: dict,
    out: np.ndarray,
    fg: int,
    bg: int,
    reserved: set[tuple[int, int]],
) -> list[tuple[int, str]]:
    row = int(seg["row"])
    c0 = int(seg["c0"])
    c1 = int(seg["c1"])
    blocked = {int(x) for x in seg["blocked"]}
    options: list[tuple[int, str]] = []
    start = int(c0 + GAP_SEG_INSET)
    stop = int(c1 - GAP_SIZE - 1)
    for g in range(start, stop + 1):
        if any(int(g - GAP_BLOCKED_LO) <= b <= int(g + GAP_BLOCKED_HI) for b in blocked):
            continue
        temp = out.copy()
        temp[row, g : g + GAP_SIZE] = int(bg)
        for anchor in ("left", "right"):
            coords = _chain_coords(row, g, anchor)
            if not _cells_are_color(temp, coords, bg):
                continue
            if not _validate_chain(temp, coords, fg, reserved):
                continue
            options.append((int(g), anchor))
    return options


def _v_gap_options(
    seg: dict,
    out: np.ndarray,
    fg: int,
    bg: int,
    reserved: set[tuple[int, int]],
) -> list[tuple[int, str, int]]:
    col = int(seg["col"])
    r0 = int(seg["r0"])
    r1 = int(seg["r1"])
    blocked = {int(x) for x in seg["blocked"]}
    options: list[tuple[int, str, int]] = []
    start = int(r0 + GAP_SEG_INSET)
    stop = int(r1 - GAP_SIZE - 1)
    for g in range(start, stop + 1):
        if any(int(g - GAP_BLOCKED_LO) <= b <= int(g + GAP_BLOCKED_HI) for b in blocked):
            continue
        temp = out.copy()
        temp[g : g + GAP_SIZE, col] = int(bg)
        for anchor in ("top", "bottom"):
            for dc in (-1, 1):
                coords = _v_chain_coords(col, g, anchor, dc)
                if not _cells_are_color(temp, coords, bg):
                    continue
                if not _validate_chain(temp, coords, fg, reserved):
                    continue
                options.append((int(g), anchor, int(dc)))
    return options


def _apply_h_gap(
    rng: random.Random,
    inp: np.ndarray,
    seg: dict,
    fg: int,
    bg: int,
    reserved: set[tuple[int, int]],
) -> bool:
    opts = _h_gap_options(seg, inp, fg, bg, reserved)
    if not opts:
        return False
    gap_start, anchor = rng.choice(opts)
    row = int(seg["row"])
    chain = _chain_coords(row, gap_start, anchor)
    inp[row, gap_start : gap_start + GAP_SIZE] = int(bg)
    for r, c in chain:
        inp[r, c] = int(fg)
    _reserve_neighborhood(reserved, chain)
    return True


def _apply_v_gap(
    rng: random.Random,
    inp: np.ndarray,
    seg: dict,
    fg: int,
    bg: int,
    reserved: set[tuple[int, int]],
) -> bool:
    opts = _v_gap_options(seg, inp, fg, bg, reserved)
    if not opts:
        return False
    gap_start, anchor, dc = rng.choice(opts)
    col = int(seg["col"])
    chain = _v_chain_coords(col, gap_start, anchor, dc)
    inp[gap_start : gap_start + GAP_SIZE, col] = int(bg)
    for r, c in chain:
        inp[r, c] = int(fg)
    _reserve_neighborhood(reserved, chain)
    return True


def _gen_attempt() -> np.ndarray | None:
    rng = random.Random()
    h = rng.randint(*GRID_H_RANGE)
    w = rng.randint(*GRID_W_RANGE)
    bg = rng.randrange(N_COLORS)
    fg = rng.choice([c for c in range(N_COLORS) if c != bg])

    built = _build_output(rng, h, w, bg, fg)
    if built is None:
        return None
    out, h_segments, v_segments, _seg_len = built

    gap_count = rng.randint(*GAP_COUNT_RANGE)
    h_gap_count = rng.randint(2, min(3, gap_count - 2))
    v_gap_count = int(gap_count - h_gap_count)
    if v_gap_count < 2 or h_gap_count > len(h_segments) or v_gap_count > len(v_segments):
        return None

    inp = out.copy()
    reserved: set[tuple[int, int]] = set()

    h_order = h_segments[:]
    v_order = v_segments[:]
    rng.shuffle(h_order)
    rng.shuffle(v_order)

    for seg in h_order[:h_gap_count]:
        if not _apply_h_gap(rng, inp, seg, fg, bg, reserved):
            return None
    for seg in v_order[:v_gap_count]:
        if not _apply_v_gap(rng, inp, seg, fg, bg, reserved):
            return None

    forbidden: set[tuple[int, int]] = set()
    _reserve_neighborhood(forbidden, list(map(tuple, np.argwhere(inp == fg))))
    distractor_count = rng.randint(*DISTRACTOR_COUNT_RANGE)
    distractors = _distractor_options(inp, bg, forbidden)
    rng.shuffle(distractors)
    placed = 0
    for coords in distractors:
        if placed >= distractor_count:
            break
        if any(cell in forbidden for cell in coords):
            continue
        for r, c in coords:
            inp[r, c] = int(fg)
        _reserve_neighborhood(forbidden, coords)
        placed += 1
    if placed < distractor_count:
        return None

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
    raise RuntimeError("failed to generate valid input for 7b80bb43 combo variant 13")
