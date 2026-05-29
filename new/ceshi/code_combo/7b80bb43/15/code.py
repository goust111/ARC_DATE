# Combo variant of 7b80bb43 (index 15)
# combo_spec: D13/0+D17/1
# Name: 角落示例引导先填后删

from __future__ import annotations

import random
import numpy as np

from arc_utils import draw_line, get_bg


# Design analysis for the fusion:
# - Concrete behavior: the top-left corner contains three mini-panels A/B/C.
#   A = broken line + diagonal hint, B = gap already filled while the same hint
#   still remains, C = clean line with the hint removed. The body of the grid is
#   a larger corrupted line network. The intended reasoning is "use the hint to
#   fill first, then delete that original hint".
# - Why order matters here: after the fill step, one hint pixel becomes
#   orthogonally adjacent to the newly restored line, so a naive "fill then
#   recompute diagonal-only pixels and delete them" solver would miss it. The
#   solver therefore records hint pixels from the original body before filling.
# - Complexity calibration: the instruction example uses only one fg color and a
#   single, repeated 3-pixel gap pattern. The body still has 3 T-junction
#   components, 2-3 repaired gaps, and optional distractor diagonals, so the
#   task remains visually close to the seed while the added corner logic is easy
#   to see.
# - Hidden assumptions checked against the new rule:
#   * Any instruction pixels must be outside the body processing area.
#   * Top-anchored body segments cannot enter the instruction footprint.
#   * Gap filling must use the original hint mask, not a post-fill re-detection.
#   * Pruning must stay local to hint neighborhoods so the corner example does
#     not affect the body.
# - Placement legality: the generator reserves a moat under the example, keeps
#   right/top edge anchors away from that corner, validates gap-chain geometry,
#   and oracle-checks every sample with the new solver.
# - Visual observability: humans can see A->B adds exactly three collinear pixels
#   while keeping the staircase, and B->C removes only the staircase. This makes
#   the multi-step sequence inferable from 2-3 examples.


EXAMPLE_H = 3
PANEL_W = 5
PANEL_GAP = 1
EXAMPLE_W = PANEL_W * 3 + PANEL_GAP * 2
RESERVED_ROWS = EXAMPLE_H + 2  # example plus blank moat
RESERVED_COLS = EXAMPLE_W + 2

GRID_H_RANGE = (18, 29)
GRID_W_RANGE = (25, 30)
N_COLORS = 10

GAP_SIZE = 3
CHAIN_LENGTH = 2
GAP_SEG_INSET = 2
DISTRACTOR_LENGTHS = (3, 2)
GEN_RETRIES = 400


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
    for dr, dc in ((-1, -1), (-1, 1), (1, -1), (1, 1)):
        rr, cc = r + dr, c + dc
        if 0 <= rr < h and 0 <= cc < w and bool(mask[rr, cc]):
            return True
    return False


def _extract_instruction(g: np.ndarray, fg: int) -> tuple[bool, int, bool]:
    panels = []
    for start in (0, PANEL_W + PANEL_GAP, 2 * (PANEL_W + PANEL_GAP)):
        panels.append(g[0:EXAMPLE_H, start : start + PANEL_W] == fg)
    best_order = None
    best_gap = GAP_SIZE
    fill_then_delete = False
    for a in range(3):
        for b in range(3):
            if b == a:
                continue
            for c in range(3):
                if c == a or c == b:
                    continue
                pa, pb, pc = panels[a], panels[b], panels[c]
                added_ab = pb & (~pa)
                removed_ab = pa & (~pb)
                added_bc = pc & (~pb)
                removed_bc = pb & (~pc)
                if int(added_ab.sum()) != GAP_SIZE or int(removed_bc.sum()) != CHAIN_LENGTH:
                    continue
                if int(removed_ab.sum()) != 0 or int(added_bc.sum()) != 0:
                    continue
                if not np.all(pa <= pb):
                    continue
                if not np.all(pc <= pb):
                    continue
                if np.any(added_ab & removed_bc):
                    continue
                # B must be the "fill first" panel: it preserves A's hint while
                # adding the missing straight run, and C only removes that hint.
                best_order = (a, b, c)
                best_gap = int(added_ab.sum())
                fill_then_delete = True
    return best_order is not None, best_gap, fill_then_delete


def _instruction_mask(shape: tuple[int, int]) -> np.ndarray:
    m = np.zeros(shape, dtype=bool)
    m[0:EXAMPLE_H, 0:EXAMPLE_W] = True
    m[0:RESERVED_ROWS, 0:RESERVED_COLS] = True
    return m


def _find_hint_pixels(mask: np.ndarray) -> np.ndarray:
    h, w = mask.shape
    hint = np.zeros_like(mask, dtype=bool)
    for r in range(h):
        for c in range(w):
            if bool(mask[r, c]) and _deg4(mask, r, c) == 0 and _has_diag(mask, r, c):
                hint[r, c] = True
    return hint


def _hint_neighborhood(hint: np.ndarray) -> np.ndarray:
    h, w = hint.shape
    near = np.zeros_like(hint, dtype=bool)
    for r, c in np.argwhere(hint):
        for dr in (-1, 0, 1):
            for dc in (-1, 0, 1):
                rr, cc = int(r + dr), int(c + dc)
                if 0 <= rr < h and 0 <= cc < w:
                    near[rr, cc] = True
    return near


def _fill_gaps_first(body: np.ndarray, fg: int, bg: int, gap_size: int) -> tuple[np.ndarray, np.ndarray, np.ndarray]:
    mask = body == fg
    hint = _find_hint_pixels(mask)
    near = _hint_neighborhood(hint)
    out = body.copy()
    h, w = body.shape

    for r in range(h):
        cols = np.where(mask[r, :] & (~hint[r, :]))[0]
        if cols.size < 2:
            continue
        segs: list[tuple[int, int]] = []
        start = prev = int(cols[0])
        for c in cols[1:]:
            c = int(c)
            if c == prev + 1:
                prev = c
            else:
                segs.append((start, prev))
                start = prev = c
        segs.append((start, prev))
        for (_a1, b1), (a2, _b2) in zip(segs, segs[1:]):
            gap0 = int(b1 + 1)
            gap1 = int(a2 - 1)
            if gap1 - gap0 + 1 != gap_size:
                continue
            rr0 = max(0, r - 1)
            rr1 = min(h - 1, r + 1)
            cc0 = max(0, gap0 - 1)
            cc1 = min(w - 1, gap1 + 1)
            if bool(hint[rr0 : rr1 + 1, cc0 : cc1 + 1].any()):
                out[r, gap0 : gap1 + 1] = fg

    for c in range(w):
        rows = np.where(mask[:, c] & (~hint[:, c]))[0]
        if rows.size < 2:
            continue
        segs: list[tuple[int, int]] = []
        start = prev = int(rows[0])
        for r in rows[1:]:
            r = int(r)
            if r == prev + 1:
                prev = r
            else:
                segs.append((start, prev))
                start = prev = r
        segs.append((start, prev))
        for (_a1, b1), (a2, _b2) in zip(segs, segs[1:]):
            gap0 = int(b1 + 1)
            gap1 = int(a2 - 1)
            if gap1 - gap0 + 1 != gap_size:
                continue
            rr0 = max(0, gap0 - 1)
            rr1 = min(h - 1, gap1 + 1)
            cc0 = max(0, c - 1)
            cc1 = min(w - 1, c + 1)
            if bool(hint[rr0 : rr1 + 1, cc0 : cc1 + 1].any()):
                out[gap0 : gap1 + 1, c] = fg

    return out, hint, near


def _prune_local_leaves(mask: np.ndarray, near: np.ndarray) -> np.ndarray:
    m = mask.copy()
    h, w = m.shape
    changed = True
    while changed:
        changed = False
        clear: list[tuple[int, int]] = []
        for r in range(h):
            for c in range(w):
                if not bool(m[r, c]) or not bool(near[r, c]):
                    continue
                if _deg4(m, r, c) != 1:
                    continue
                if not _has_diag(m, r, c):
                    continue
                clear.append((r, c))
        if clear:
            changed = True
            for r, c in clear:
                m[r, c] = False
    return m


def _solve_body(grid: np.ndarray, fg: int, bg: int, gap_size: int) -> np.ndarray:
    body = grid.copy()
    body[_instruction_mask(grid.shape)] = bg
    filled, hint, near = _fill_gaps_first(body, fg, bg, gap_size)
    filled[hint] = bg
    pruned = _prune_local_leaves(filled == fg, near)
    out = np.full(grid.shape, bg, dtype=int)
    out[pruned] = fg
    out[_instruction_mask(grid.shape)] = bg
    return out


def _count_order_sensitive_hints(grid: np.ndarray, fg: int, bg: int, gap_size: int) -> int:
    body = grid.copy()
    body[_instruction_mask(grid.shape)] = bg
    filled, hint, _near = _fill_gaps_first(body, fg, bg, gap_size)
    current = filled == fg
    count = 0
    for r, c in np.argwhere(hint):
        if not (_deg4(current, int(r), int(c)) == 0 and _has_diag(current, int(r), int(c))):
            count += 1
    return count


def main(grid: np.ndarray) -> np.ndarray:
    """
    Read the corner teaching example, then repair the main line network by filling before deleting.

    Input structure:
    - One foreground color forms both the main axis-aligned network and a tiny three-panel lesson in a corner.
    - The lesson shows a broken mini-line, then the same mini-line after gap filling while the diagonal hint still remains, then the final cleaned line.
    - The larger body contains the real target network with 3-pixel gaps and diagonal staircase hint pixels.

    Rule:
    - Detect the instructional corner trio and infer that the body must be processed in two stages: restore the straight gap first, then erase the original diagonal hint pixels.
    - Use the original hint locations to fill nearby 3-cell horizontal or vertical gaps in the body.
    - After filling, remove those recorded hint pixels and prune any residual local leaf pixels near former hint sites.

    Key insight: The middle corner panel proves that some diagonal hint pixels survive the fill step, so the solver must remember which pixels were hints in the initial body instead of re-detecting them after filling.
    """
    g = np.asarray(grid, dtype=int)
    bg = int(get_bg(g))
    colors = [int(x) for x in np.unique(g) if int(x) != bg]
    assert colors
    fg = colors[0]
    learned, gap_size, fill_then_delete = _extract_instruction(g, fg)
    assert learned, "instruction example missing or malformed"
    assert fill_then_delete, "instruction example does not encode fill-then-delete"
    return _solve_body(g, fg, bg, gap_size)


def fusion_metrics(grid: np.ndarray) -> dict[str, int]:
    g = np.asarray(grid, dtype=int)
    bg = int(get_bg(g))
    fg = next(int(x) for x in np.unique(g) if int(x) != bg)
    learned, gap_size, fill_then_delete = _extract_instruction(g, fg)
    return {
        "instruction_learned": int(learned),
        "fill_then_delete_instruction": int(fill_then_delete),
        "gap_size": int(gap_size),
        "order_sensitive_hints": int(_count_order_sensitive_hints(g, fg, bg, gap_size)),
    }


def _pick_distinct(rng: random.Random, values: list[int], k: int, gap: int = 2) -> list[int]:
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
        if not (0 <= r < h and 0 <= c < w) or int(grid[r, c]) != color:
            return False
    return True


def _validate_chain(grid: np.ndarray, coords: list[tuple[int, int]], fg: int,
                    occupied: set[tuple[int, int]]) -> bool:
    h, w = grid.shape
    s = set(coords)
    if any(cell in occupied for cell in coords):
        return False
    for r, c in coords:
        for rr, cc in ((r - 1, c), (r + 1, c), (r, c - 1), (r, c + 1)):
            if (rr, cc) in s:
                return False
            if 0 <= rr < h and 0 <= cc < w and int(grid[rr, cc]) == fg:
                return False
        ok = False
        for rr, cc in ((r - 1, c - 1), (r - 1, c + 1), (r + 1, c - 1), (r + 1, c + 1)):
            if (rr, cc) in s or (0 <= rr < h and 0 <= cc < w and int(grid[rr, cc]) == fg):
                ok = True
                break
        if not ok:
            return False
    return True


def _reserve_neighborhood(cells: set[tuple[int, int]], coords) -> None:
    for r, c in coords:
        for dr in (-1, 0, 1):
            for dc in (-1, 0, 1):
                cells.add((int(r + dr), int(c + dc)))


def _distractor_options(inp: np.ndarray, bg: int,
                        forbidden: set[tuple[int, int]]) -> list[list[tuple[int, int]]]:
    h, w = inp.shape
    opts: list[list[tuple[int, int]]] = []
    for length in DISTRACTOR_LENGTHS:
        for r in range(1, h - length):
            for c in range(1, w - 1):
                for direction in (-1, 1):
                    coords = [(r + i, c + direction * i) for i in range(length)]
                    if not _cells_are_color(inp, coords, bg):
                        continue
                    if any(cell in forbidden for cell in coords):
                        continue
                    opts.append(coords)
    return opts


def _place_t_component(out: np.ndarray, fg: int, slot: str, row: int,
                       branch_col: int, end_col: int) -> tuple[dict, dict]:
    h, w = out.shape
    h_c0 = 0 if slot[0] == "L" else end_col
    h_c1 = end_col if slot[0] == "L" else w - 1
    v_r0 = 0 if slot[1] == "T" else row
    v_r1 = row if slot[1] == "T" else h - 1
    draw_line(out, row, h_c0, end_r=row, end_c=h_c1, color=fg)
    draw_line(out, v_r0, branch_col, end_r=v_r1, end_c=branch_col, color=fg)
    return (
        {"row": row, "c0": h_c0, "c1": h_c1, "blocked": {branch_col}},
        {"col": branch_col, "r0": v_r0, "r1": v_r1, "blocked": {row}},
    )


def _build_output(rng: random.Random, h: int, w: int, bg: int, fg: int) -> tuple[np.ndarray, list[dict], list[dict]] | None:
    out = np.full((h, w), bg, dtype=int)
    h_segments: list[dict] = []
    v_segments: list[dict] = []

    top_pool = list(range(RESERVED_ROWS, max(RESERVED_ROWS + 1, h // 2 - 1)))
    bottom_pool = list(range(max(h // 2 + 2, RESERVED_ROWS + 4), h - 3))
    if not top_pool or len(bottom_pool) < 2:
        return None
    top_rows = _pick_distinct(rng, top_pool, 1)
    bottom_rows = _pick_distinct(rng, bottom_pool, 2, gap=3)
    if len(top_rows) != 1 or len(bottom_rows) != 2:
        return None
    rt_row = top_rows[0]
    lb_row, rb_row = bottom_rows

    left_cols = _pick_distinct(rng, list(range(3, max(4, w // 2 - 3))), 1)
    right_cols = _pick_distinct(rng, list(range(max(RESERVED_COLS, w // 2 + 2), w - 3)), 2, gap=2)
    if len(left_cols) != 1 or len(right_cols) != 2:
        return None
    lb_col = left_cols[0]
    rt_col, rb_col = right_cols

    rt_lo = max(1, rt_col - 8)
    rt_hi = rt_col - 3
    lb_lo = lb_col + 3
    lb_hi = min(w // 2, rb_col - 4)
    rb_lo = max(w // 2 + 1, rb_col - 8)
    rb_hi = rb_col - 3
    if rt_lo > rt_hi or lb_lo > lb_hi or rb_lo > rb_hi:
        return None

    rt_h, rt_v = _place_t_component(out, fg, "RT", rt_row, rt_col, rng.randint(rt_lo, rt_hi))
    lb_h, lb_v = _place_t_component(out, fg, "LB", lb_row, lb_col, rng.randint(lb_lo, lb_hi))
    rb_h, rb_v = _place_t_component(out, fg, "RB", rb_row, rb_col, rng.randint(rb_lo, rb_hi))
    h_segments.extend([rt_h, lb_h, rb_h])
    v_segments.extend([rt_v, lb_v, rb_v])

    if np.any(out[0:RESERVED_ROWS, 0:RESERVED_COLS] != bg):
        return None
    return out, h_segments, v_segments


def _h_gap_options(seg: dict, out: np.ndarray, fg: int, bg: int,
                   occupied: set[tuple[int, int]]) -> list[tuple[int, str]]:
    row, c0, c1 = seg["row"], seg["c0"], seg["c1"]
    blocked = seg["blocked"]
    opts: list[tuple[int, str]] = []
    for g in range(c0 + GAP_SEG_INSET, c1 - GAP_SIZE):
        if any(abs(g - b) <= 3 for b in blocked):
            continue
        temp = out.copy()
        temp[row, g : g + GAP_SIZE] = bg
        for anchor in ("left", "right"):
            coords = _chain_coords(row, g, anchor)
            if not _cells_are_color(temp, coords, bg):
                continue
            if not _validate_chain(temp, coords, fg, occupied):
                continue
            opts.append((g, anchor))
    return opts


def _v_gap_options(seg: dict, out: np.ndarray, fg: int, bg: int,
                   occupied: set[tuple[int, int]]) -> list[tuple[int, str, int]]:
    col, r0, r1 = seg["col"], seg["r0"], seg["r1"]
    blocked = seg["blocked"]
    opts: list[tuple[int, str, int]] = []
    for g in range(r0 + GAP_SEG_INSET, r1 - GAP_SIZE):
        if any(abs(g - b) <= 3 for b in blocked):
            continue
        temp = out.copy()
        temp[g : g + GAP_SIZE, col] = bg
        for anchor in ("top", "bottom"):
            for dc in (-1, 1):
                coords = _v_chain_coords(col, g, anchor, dc)
                if not _cells_are_color(temp, coords, bg):
                    continue
                if not _validate_chain(temp, coords, fg, occupied):
                    continue
                opts.append((g, anchor, dc))
    return opts


def _place_example(inp: np.ndarray, fg: int) -> None:
    starts = (0, PANEL_W + PANEL_GAP, 2 * (PANEL_W + PANEL_GAP))
    # A: broken line + diagonal hint.
    a = starts[0]
    inp[0, a + 0] = fg
    inp[0, a + 4] = fg
    inp[1, a + 1] = fg
    inp[2, a + 2] = fg
    # B: filled line while keeping the same hint.
    b = starts[1]
    inp[0, b : b + PANEL_W] = fg
    inp[1, b + 1] = fg
    inp[2, b + 2] = fg
    # C: final clean line.
    c = starts[2]
    inp[0, c : c + PANEL_W] = fg


def _gen_attempt() -> np.ndarray | None:
    rng = random.Random()
    h = rng.randint(*GRID_H_RANGE)
    w = rng.randint(*GRID_W_RANGE)
    bg = rng.randrange(N_COLORS)
    fg = rng.choice([c for c in range(N_COLORS) if c != bg])

    built = _build_output(rng, h, w, bg, fg)
    if built is None:
        return None
    out, h_segments, v_segments = built

    inp = out.copy()
    occupied: set[tuple[int, int]] = set()

    h_ready = [(seg, _h_gap_options(seg, out, fg, bg, occupied)) for seg in h_segments]
    v_ready = [(seg, _v_gap_options(seg, out, fg, bg, occupied)) for seg in v_segments]
    h_ready = [(seg, opts) for seg, opts in h_ready if opts]
    v_ready = [(seg, opts) for seg, opts in v_ready if opts]
    if len(h_ready) < 1 or len(v_ready) < 1:
        return None

    gap_pattern = rng.choice([(1, 1), (1, 2), (2, 1)])
    h_gap_count, v_gap_count = gap_pattern
    if len(h_ready) < h_gap_count or len(v_ready) < v_gap_count:
        return None

    rng.shuffle(h_ready)
    rng.shuffle(v_ready)

    for seg, initial in h_ready[:h_gap_count]:
        opts = _h_gap_options(seg, out, fg, bg, occupied) or initial
        if not opts:
            return None
        gap_start, anchor = rng.choice(opts)
        row = seg["row"]
        chain = _chain_coords(row, gap_start, anchor)
        inp[row, gap_start : gap_start + GAP_SIZE] = bg
        for r, c in chain:
            inp[r, c] = fg
        occupied.update(chain)

    for seg, initial in v_ready[:v_gap_count]:
        opts = _v_gap_options(seg, out, fg, bg, occupied) or initial
        if not opts:
            return None
        gap_start, anchor, dc = rng.choice(opts)
        col = seg["col"]
        chain = _v_chain_coords(col, gap_start, anchor, dc)
        inp[gap_start : gap_start + GAP_SIZE, col] = bg
        for r, c in chain:
            inp[r, c] = fg
        occupied.update(chain)

    forbidden: set[tuple[int, int]] = set()
    _reserve_neighborhood(forbidden, list(map(tuple, np.argwhere(inp == fg))))
    distractors = _distractor_options(inp, bg, forbidden)
    rng.shuffle(distractors)
    for coords in distractors[: rng.randint(0, 2)]:
        if any(cell in forbidden for cell in coords):
            continue
        for r, c in coords:
            inp[r, c] = fg
        _reserve_neighborhood(forbidden, coords)

    if np.any(inp[0:RESERVED_ROWS, 0:RESERVED_COLS] != bg):
        return None
    _place_example(inp, fg)

    try:
        solved = main(inp)
    except Exception:
        return None
    if not np.array_equal(solved, out):
        return None
    if _count_order_sensitive_hints(inp, fg, bg, GAP_SIZE) == 0:
        return None
    return inp


def generate_input() -> np.ndarray:
    for _ in range(GEN_RETRIES):
        grid = _gen_attempt()
        if grid is not None:
            return grid
    raise RuntimeError("failed to generate valid combo input for 7b80bb43")
