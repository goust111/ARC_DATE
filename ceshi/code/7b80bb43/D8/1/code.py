# Variant of 7b80bb43
# Dimension: D8 — 线段网络拓扑结构 (meta)
# Change: 6-8条轴对齐线段（2-5水平+3-6垂直），形成3-5个4连通分量，含2-3个T型交叉节点... → 十字交叉节点代替T型交叉

from __future__ import annotations

import random

import numpy as np

from arc_utils import components_mask, draw_line, find_components, get_bg


"""
design_analysis:
  a_specific_behavior:
    original:
      output is a sparse axis-aligned segment network whose only H/V junctions are T-nodes.
      Each corruption is still a 3-cell gap plus a 2-pixel diagonal hint.
    variant:
      some connected components are now isolated plus-shapes: one horizontal segment and one
      vertical segment fully pass through the same center cell, creating a 4-way cross node.
      Gaps may be cut on an arm immediately next to that center, but the center cell itself is
      never deleted. Example: a cross arm "...###|###..." can become "...###|..."+diag-hint on
      the right arm; solver must fill the missing arm while preserving the center, instead of
      treating the local T-like intermediate shape as a final topology.
    affected_pipeline_stage:
      generator changes the most: T-specific placement and blocked-gap rules do not apply.
      solver only needs a hardening change: pruning near former diagonal hints must not delete
      protected junction pixels that temporarily look leaf-like around restored crosses.
  b_complexity_calibration:
    - Keep one foreground color and the same diagonal-hint corruption rule, so the new idea is
      still readable from 2-4 examples.
    - Make 2-3 true cross nodes per grid and force at least one repaired gap on a cross arm near
      the node, so the D8 change is visually present instead of hiding in the output only.
    - Avoid overfitting to many topologies: output is built from isolated cross components plus a
      few simple edge-anchored segments, keeping the family tight and learnable.
  c_hidden_assumptions_audit:
    - Original generator assumes every junction is a T, encoded by slot names and asymmetric
      blocked ranges around one branch column/row. That fails for crosses because both directions
      continue through the node.
    - Original component-count check `segments - junctions` is only valid for isolated T merges.
      With crosses, one long H crossing several V's could collapse many segments into one giant
      component. The variant avoids that by generating isolated one-to-one cross components.
    - Original pruning can safely delete near-hint degree-1 pixels for T-only layouts. In the
      variant, a cross center with one damaged arm can transiently resemble a T; pruning must
      explicitly protect junction-like pixels.
  d_legality:
    - Crosses are placed as bounded plus-shapes with a one-cell moat from other fg structure.
    - Simple segments remain edge-anchored, also with moat spacing.
    - Gaps never cover a cross center; candidate cuts require both remaining sides to stay length
      >= 2 and chain pixels to remain diagonal-only.
    - Output-first oracle check `main(inp) == out` is retained.
  e_visual_observability:
    - Humans can see plus-shaped components in the output and T-like damaged local neighborhoods
      in the input that become full crosses after repair.
    - At least one gap is placed on a cross arm with distance 1-2 from the center, so the new
      node-protection rule appears directly in examples.
  f_visual_cleanliness:
    - Components are isolated by padding; distractor chains are only placed in open background.
    - The puzzle still uses one fg color, so topological shape is the only variable the viewer
      needs to track.
  g_color_inference:
    - No new color logic is introduced; colors remain irrelevant except fg vs bg.
"""


# --- Configuration constants ---

GRID_H_RANGE = (16, 29)
GRID_W_RANGE = (17, 30)
N_COLORS = 10

VALID_CONFIGS = [
    {"crosses": 2, "simple_h": 0, "simple_v": 2, "segments": 6, "h": 2, "v": 4, "components": 4},
    {"crosses": 2, "simple_h": 1, "simple_v": 1, "segments": 6, "h": 3, "v": 3, "components": 4},
    {"crosses": 2, "simple_h": 1, "simple_v": 2, "segments": 7, "h": 3, "v": 4, "components": 5},
    {"crosses": 2, "simple_h": 2, "simple_v": 1, "segments": 7, "h": 4, "v": 3, "components": 5},
    {"crosses": 3, "simple_h": 0, "simple_v": 0, "segments": 6, "h": 3, "v": 3, "components": 3},
    {"crosses": 3, "simple_h": 0, "simple_v": 1, "segments": 7, "h": 3, "v": 4, "components": 4},
    {"crosses": 3, "simple_h": 1, "simple_v": 0, "segments": 7, "h": 4, "v": 3, "components": 4},
    {"crosses": 3, "simple_h": 1, "simple_v": 1, "segments": 8, "h": 4, "v": 4, "components": 5},
    {"crosses": 3, "simple_h": 0, "simple_v": 2, "segments": 8, "h": 3, "v": 5, "components": 5},
    {"crosses": 3, "simple_h": 2, "simple_v": 0, "segments": 8, "h": 5, "v": 3, "components": 5},
]

CROSS_ARM_MIN = 2
CROSS_ARM_MAX = 5
SIMPLE_H_LEN_MIN = 6
SIMPLE_H_LEN_MAX = 11
SIMPLE_V_LEN_MIN = 6
SIMPLE_V_LEN_MAX = 11
PLACEMENT_PAD = 1

GAP_SIZE = 3
CHAIN_LENGTH = 2
GAP_SEG_INSET = 2
GAP_COUNT_CHOICES = (2, 3)

DISTRACTOR_LENGTHS = (3, 2)
DISTRACTOR_MARGIN = 1

GEN_RETRIES = 500


def main(grid: np.ndarray) -> np.ndarray:
    """
    Remove diagonal staircase noise and complete broken straight lines.

    Variant-specific note:
    - Junctions may now be 4-way crosses. Gaps can appear next to the crossing center, but the
      center itself remains present and must be preserved during cleanup.
    """
    g = np.array(grid, dtype=int)
    bg = int(get_bg(g))

    colors = {int(x) for x in np.unique(g)}
    colors.discard(bg)
    assert colors
    fg = int(next(iter(colors)))

    mask = g == fg
    H, W = g.shape

    def deg4(m: np.ndarray, r: int, c: int) -> int:
        d = 0
        if r > 0 and bool(m[r - 1, c]):
            d += 1
        if r + 1 < H and bool(m[r + 1, c]):
            d += 1
        if c > 0 and bool(m[r, c - 1]):
            d += 1
        if c + 1 < W and bool(m[r, c + 1]):
            d += 1
        return d

    def has_diag(m: np.ndarray, r: int, c: int) -> bool:
        if r > 0 and c > 0 and bool(m[r - 1, c - 1]):
            return True
        if r > 0 and c + 1 < W and bool(m[r - 1, c + 1]):
            return True
        if r + 1 < H and c > 0 and bool(m[r + 1, c - 1]):
            return True
        if r + 1 < H and c + 1 < W and bool(m[r + 1, c + 1]):
            return True
        return False

    def is_protected_node(m: np.ndarray, r: int, c: int) -> bool:
        if not bool(m[r, c]):
            return False
        up = r > 0 and bool(m[r - 1, c])
        down = r + 1 < H and bool(m[r + 1, c])
        left = c > 0 and bool(m[r, c - 1])
        right = c + 1 < W and bool(m[r, c + 1])
        has_h = left or right
        has_v = up or down
        return bool(has_h and has_v and ((left and right) or (up and down)))

    diag = np.zeros((H, W), dtype=bool)
    for r in range(H):
        for c in range(W):
            if bool(mask[r, c]) and deg4(mask, r, c) == 0 and has_diag(mask, r, c):
                diag[r, c] = True

    near = np.zeros((H, W), dtype=bool)
    for r in range(H):
        for c in range(W):
            if not bool(diag[r, c]):
                continue
            for dr in (-1, 0, 1):
                for dc in (-1, 0, 1):
                    rr = r + dr
                    cc = c + dc
                    if 0 <= rr < H and 0 <= cc < W:
                        near[rr, cc] = True

    out = g.copy()
    out[diag] = bg

    max_gap = GAP_SIZE
    if bool(diag.any()):
        for coords in components_mask(diag, connectivity=8):
            if coords.size == 0:
                continue
            r0, c0 = coords.min(axis=0)
            r1, c1 = coords.max(axis=0)
            max_gap = max(max_gap, int(r1 - r0 + 1), int(c1 - c0 + 1))

    for r in range(H):
        cols = np.where(mask[r, :] & (~diag[r, :]))[0]
        if cols.size == 0:
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
            if gap1 < gap0 or gap1 - gap0 + 1 > max_gap:
                continue
            rr0 = max(0, r - 1)
            rr1 = min(H - 1, r + 1)
            cc0 = max(0, gap0 - 1)
            cc1 = min(W - 1, gap1 + 1)
            if bool(diag[rr0 : rr1 + 1, cc0 : cc1 + 1].any()):
                out[r, gap0 : gap1 + 1] = fg

    for c in range(W):
        rows = np.where(mask[:, c] & (~diag[:, c]))[0]
        if rows.size == 0:
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
            if gap1 < gap0 or gap1 - gap0 + 1 > max_gap:
                continue
            rr0 = max(0, gap0 - 1)
            rr1 = min(H - 1, gap1 + 1)
            cc0 = max(0, c - 1)
            cc1 = min(W - 1, c + 1)
            if bool(diag[rr0 : rr1 + 1, cc0 : cc1 + 1].any()):
                out[gap0 : gap1 + 1, c] = fg

    m = out == fg
    changed = True
    while changed:
        changed = False
        to_clear: list[tuple[int, int]] = []
        for r in range(H):
            for c in range(W):
                if not bool(m[r, c]) or not bool(near[r, c]):
                    continue
                if is_protected_node(m, r, c):
                    continue
                if deg4(m, r, c) != 1:
                    continue
                if not has_diag(m, r, c):
                    continue
                to_clear.append((r, c))
        if to_clear:
            changed = True
            for r, c in to_clear:
                m[r, c] = False

    out2 = np.full((H, W), bg, dtype=int)
    out2[m] = fg
    return out2


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
        if not (0 <= r < h and 0 <= c < w) or int(grid[r, c]) != int(color):
            return False
    return True


def _validate_chain(
    grid: np.ndarray, coords: list[tuple[int, int]], fg: int, occupied: set[tuple[int, int]]
) -> bool:
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


def _patch_clear(grid: np.ndarray, patch: np.ndarray, top: int, left: int, bg: int, pad: int = PLACEMENT_PAD) -> bool:
    h, w = grid.shape
    ph, pw = patch.shape
    if top < 0 or left < 0 or top + ph > h or left + pw > w:
        return False
    coords = np.argwhere(patch != bg)
    for rr, cc in coords:
        gr = int(top + rr)
        gc = int(left + cc)
        r0 = max(0, gr - pad)
        r1 = min(h - 1, gr + pad)
        c0 = max(0, gc - pad)
        c1 = min(w - 1, gc + pad)
        if bool((grid[r0 : r1 + 1, c0 : c1 + 1] != bg).any()):
            return False
    return True


def _stamp_patch(grid: np.ndarray, patch: np.ndarray, top: int, left: int, bg: int) -> None:
    coords = np.argwhere(patch != bg)
    for rr, cc in coords:
        grid[int(top + rr), int(left + cc)] = int(patch[int(rr), int(cc)])


def _cross_patch(left_len: int, right_len: int, up_len: int, down_len: int, fg: int, bg: int) -> tuple[np.ndarray, int, int]:
    ph = up_len + down_len + 1
    pw = left_len + right_len + 1
    patch = np.full((ph, pw), bg, dtype=int)
    cr = up_len
    cc = left_len
    draw_line(patch, cr, 0, end_r=cr, end_c=pw - 1, color=fg)
    draw_line(patch, 0, cc, end_r=ph - 1, end_c=cc, color=fg)
    return patch, cr, cc


def _place_cross(out: np.ndarray, fg: int, bg: int, rng: random.Random) -> tuple[dict, dict] | None:
    h, w = out.shape
    for _ in range(160):
        left_len = rng.randint(CROSS_ARM_MIN, CROSS_ARM_MAX)
        right_len = rng.randint(CROSS_ARM_MIN, CROSS_ARM_MAX)
        up_len = rng.randint(CROSS_ARM_MIN, CROSS_ARM_MAX)
        down_len = rng.randint(CROSS_ARM_MIN, CROSS_ARM_MAX)
        patch, cr, cc = _cross_patch(left_len, right_len, up_len, down_len, fg, bg)
        ph, pw = patch.shape
        if ph >= h - 1 or pw >= w - 1:
            continue
        top = rng.randint(1, h - ph - 1)
        left = rng.randint(1, w - pw - 1)
        if not _patch_clear(out, patch, top, left, bg):
            continue
        _stamp_patch(out, patch, top, left, bg)
        row = top + cr
        col = left + cc
        h_seg = {"row": row, "c0": left, "c1": left + pw - 1, "cross": col}
        v_seg = {"col": col, "r0": top, "r1": top + ph - 1, "cross": row}
        return h_seg, v_seg
    return None


def _place_simple_h(out: np.ndarray, fg: int, bg: int, rng: random.Random) -> dict | None:
    h, w = out.shape
    for _ in range(180):
        length = rng.randint(SIMPLE_H_LEN_MIN, min(SIMPLE_H_LEN_MAX, w - 4))
        side = rng.choice(["left", "right"])
        row = rng.randint(0, h - 1)
        c0 = 0 if side == "left" else w - length
        patch = np.full((1, length), fg, dtype=int)
        if not _patch_clear(out, patch, row, c0, bg):
            continue
        _stamp_patch(out, patch, row, c0, bg)
        return {"row": row, "c0": c0, "c1": c0 + length - 1, "cross": None}
    return None


def _place_simple_v(out: np.ndarray, fg: int, bg: int, rng: random.Random) -> dict | None:
    h, w = out.shape
    for _ in range(180):
        length = rng.randint(SIMPLE_V_LEN_MIN, min(SIMPLE_V_LEN_MAX, h - 4))
        side = rng.choice(["top", "bottom"])
        col = rng.randint(0, w - 1)
        r0 = 0 if side == "top" else h - length
        patch = np.full((length, 1), fg, dtype=int)
        if not _patch_clear(out, patch, r0, col, bg):
            continue
        _stamp_patch(out, patch, r0, col, bg)
        return {"col": col, "r0": r0, "r1": r0 + length - 1, "cross": None}
    return None


def _gap_distance_to_cross(start: int, center: int) -> int:
    return min(abs(center - x) for x in range(start, start + GAP_SIZE))


def _h_gap_options(
    seg: dict, out: np.ndarray, fg: int, bg: int, occupied: set[tuple[int, int]]
) -> tuple[list[tuple[int, str]], list[tuple[int, str]]]:
    row, c0, c1 = int(seg["row"]), int(seg["c0"]), int(seg["c1"])
    center = seg.get("cross")
    preferred: list[tuple[int, str]] = []
    options: list[tuple[int, str]] = []
    for g in range(c0 + GAP_SEG_INSET, c1 - GAP_SIZE - GAP_SEG_INSET + 2):
        left_len = g - c0
        right_len = c1 - (g + GAP_SIZE - 1)
        if left_len < 2 or right_len < 2:
            continue
        if center is not None and g <= int(center) <= g + GAP_SIZE - 1:
            continue
        temp = out.copy()
        temp[row, g : g + GAP_SIZE] = bg
        for anchor in ("left", "right"):
            coords = _chain_coords(row, g, anchor)
            if not _cells_are_color(temp, coords, bg):
                continue
            if not _validate_chain(temp, coords, fg, occupied):
                continue
            item = (g, anchor)
            options.append(item)
            if center is not None and _gap_distance_to_cross(g, int(center)) <= 2:
                preferred.append(item)
    return preferred, options


def _v_gap_options(
    seg: dict, out: np.ndarray, fg: int, bg: int, occupied: set[tuple[int, int]]
) -> tuple[list[tuple[int, str, int]], list[tuple[int, str, int]]]:
    col, r0, r1 = int(seg["col"]), int(seg["r0"]), int(seg["r1"])
    center = seg.get("cross")
    preferred: list[tuple[int, str, int]] = []
    options: list[tuple[int, str, int]] = []
    for g in range(r0 + GAP_SEG_INSET, r1 - GAP_SIZE - GAP_SEG_INSET + 2):
        top_len = g - r0
        bottom_len = r1 - (g + GAP_SIZE - 1)
        if top_len < 2 or bottom_len < 2:
            continue
        if center is not None and g <= int(center) <= g + GAP_SIZE - 1:
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
                item = (g, anchor, dc)
                options.append(item)
                if center is not None and _gap_distance_to_cross(g, int(center)) <= 2:
                    preferred.append(item)
    return preferred, options


def _reserve_neighborhood(cells: set[tuple[int, int]], coords: list[tuple[int, int]]) -> None:
    for r, c in coords:
        for dr in (-1, 0, 1):
            for dc in (-1, 0, 1):
                cells.add((r + dr, c + dc))


def _distractor_options(inp: np.ndarray, bg: int, forbidden: set[tuple[int, int]]) -> list[list[tuple[int, int]]]:
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


def _build_output(
    rng: random.Random, h: int, w: int, bg: int, fg: int, cfg: dict
) -> tuple[np.ndarray, list[dict], list[dict]] | None:
    out = np.full((h, w), bg, dtype=int)
    h_segments: list[dict] = []
    v_segments: list[dict] = []

    for _ in range(int(cfg["crosses"])):
        placed = _place_cross(out, fg, bg, rng)
        if placed is None:
            return None
        h_seg, v_seg = placed
        h_segments.append(h_seg)
        v_segments.append(v_seg)

    for _ in range(int(cfg["simple_h"])):
        seg = _place_simple_h(out, fg, bg, rng)
        if seg is None:
            return None
        h_segments.append(seg)

    for _ in range(int(cfg["simple_v"])):
        seg = _place_simple_v(out, fg, bg, rng)
        if seg is None:
            return None
        v_segments.append(seg)

    comps = find_components(out, bg=bg, connectivity=4, monochromatic=True, return_format="dict")
    if len(comps) != int(cfg["components"]):
        return None
    return out, h_segments, v_segments


def _apply_h_gap(inp: np.ndarray, seg: dict, choice: tuple[int, str], fg: int, bg: int) -> list[tuple[int, int]]:
    gap_start, anchor = choice
    row = int(seg["row"])
    chain = _chain_coords(row, int(gap_start), str(anchor))
    inp[row, int(gap_start) : int(gap_start) + GAP_SIZE] = bg
    for r, c in chain:
        inp[r, c] = fg
    return chain


def _apply_v_gap(inp: np.ndarray, seg: dict, choice: tuple[int, str, int], fg: int, bg: int) -> list[tuple[int, int]]:
    gap_start, anchor, dc = choice
    col = int(seg["col"])
    chain = _v_chain_coords(col, int(gap_start), str(anchor), int(dc))
    inp[int(gap_start) : int(gap_start) + GAP_SIZE, col] = bg
    for r, c in chain:
        inp[r, c] = fg
    return chain


def _gen_attempt() -> np.ndarray | None:
    rng = random.Random()
    h = rng.randint(*GRID_H_RANGE)
    w = rng.randint(*GRID_W_RANGE)
    bg = rng.randrange(N_COLORS)
    fg = rng.choice([c for c in range(N_COLORS) if c != bg])
    cfg = rng.choice(VALID_CONFIGS)
    gap_count = rng.choice(GAP_COUNT_CHOICES)
    distractor_count = rng.randint(0, 2)

    built = _build_output(rng, h, w, bg, fg, cfg)
    if built is None:
        return None
    out, h_segments, v_segments = built
    inp = out.copy()
    occupied: set[tuple[int, int]] = set()

    cross_h_ready = []
    for seg in h_segments:
        preferred, options = _h_gap_options(seg, out, fg, bg, occupied)
        if options:
            cross_h_ready.append((seg, preferred, options))
    cross_h_ready = [item for item in cross_h_ready if item[0].get("cross") is not None and item[1]]
    if not cross_h_ready:
        return None

    h_seg, h_preferred, _h_options = rng.choice(cross_h_ready)
    h_chain = _apply_h_gap(inp, h_seg, rng.choice(h_preferred), fg, bg)
    occupied.update(h_chain)

    v_gap_count = gap_count - 1
    remaining_v = v_segments[:]
    chosen_cross_v = False
    for i in range(v_gap_count):
        ready = []
        for seg in remaining_v:
            preferred, options = _v_gap_options(seg, out, fg, bg, occupied)
            if options:
                ready.append((seg, preferred, options))
        if len(ready) < v_gap_count - i:
            return None
        if not chosen_cross_v:
            cross_pref = [item for item in ready if item[0].get("cross") is not None and item[1]]
            pool = cross_pref if cross_pref else ready
        else:
            pool = ready
        seg, preferred, options = rng.choice(pool)
        choice = rng.choice(preferred if preferred else options)
        chain = _apply_v_gap(inp, seg, choice, fg, bg)
        occupied.update(chain)
        chosen_cross_v = chosen_cross_v or bool(seg.get("cross") is not None and preferred)
        remaining_v.remove(seg)

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
    raise RuntimeError("failed to generate valid input for 7b80bb43 D8 variant")
