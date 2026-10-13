from __future__ import annotations
from collections import defaultdict
import numpy as np
from arc_utils import get_bg, label_mask


def largest_enclosed_component_bbox(mask: np.ndarray, *, connectivity: int = 4) -> tuple[int, int, int, int] | None:
    """
    Local helper (only used by this solver): bbox of largest connected component in `mask`
    that does NOT touch the grid border.
    """
    H, W = mask.shape
    best_area = -1
    best_bbox: tuple[int, int, int, int] | None = None

    lab, n = label_mask(mask.astype(bool), connectivity=int(connectivity))
    for k in range(1, int(n) + 1):
        coords = np.argwhere(lab == int(k))
        if coords.size == 0:
            continue
        rs = coords[:, 0]
        cs = coords[:, 1]
        if bool((rs == 0).any() or (rs == int(H - 1)).any() or (cs == 0).any() or (cs == int(W - 1)).any()):
            continue
        area = int(coords.shape[0])
        if area > best_area:
            best_area = area
            r0, c0 = coords.min(axis=0)
            r1, c1 = coords.max(axis=0)
            best_bbox = (int(r0), int(c0), int(r1), int(c1))

    return best_bbox

def main(grid: np.ndarray) -> np.ndarray:
    """
    Move colored sprites from the left strip into a framed area at marker-defined targets.

    Input structure:
    - bg background; rectangular frame on right with colored markers on its edges; sprites in a narrow left strip separated by a constant-color divider column.

    Rule:
    - For each color, target rows/cols are where both opposite frame edges share that color. Targets are the Cartesian product.
    - Erase each sprite from the left and stamp it centered at every target. Higher sprites cover lower ones.

    Key insight: One sprite replicates to a grid of row × column marker intersections.
    """
    g = np.array(grid, dtype=int)
    H, W = g.shape
    bg = int(get_bg(g))

    # --- frame interior: largest bg component not touching boundary ---
    inner_bbox = largest_enclosed_component_bbox(g == int(bg), connectivity=4)
    assert not (inner_bbox is None)
    ir0, ic0, ir1, ic1 = map(int, inner_bbox)
    fr0, fc0, fr1, fc1 = int(ir0 - 1), int(ic0 - 1), int(ir1 + 1), int(ic1 + 1)

    # --- divider columns (fixed) on the left side ---
    divider_mask = np.zeros((H, W), dtype=bool)
    for c in range(0, max(0, int(fc0))):
        colv = g[:, int(c)]
        non = colv[colv != int(bg)]
        if int(len(non)) >= int(H - 1) and len(set(map(int, non))) == 1:
            divider_mask[:, int(c)] = True

    left_cols = (np.arange(W)[None, :] < int(fc0))

    # --- sprites: per-color 8-connected components in left area (excluding divider) ---
    sprites = []
    for col in sorted(set(map(int, g.flatten())) - {int(bg)}):
        mask = (g == int(col)) & left_cols & (~divider_mask)
        if not bool(mask.any()):
            continue
        slab, sn = label_mask(mask, connectivity=8)
        for sid in range(1, int(sn) + 1):
            pts = np.argwhere(slab == int(sid))
            r0, c0 = pts.min(axis=0)
            r1, c1 = pts.max(axis=0)
            r0, c0, r1, c1 = map(int, (r0, c0, r1, c1))
            m = slab[int(r0) : int(r1) + 1, int(c0) : int(c1) + 1] == int(sid)
            sprites.append({"color": int(col), "r0": int(r0), "c0": int(c0), "r1": int(r1), "c1": int(c1), "mask": m})

    move_colors = {int(s["color"]) for s in sprites}

    # --- marker-derived targets (restricted to colors that actually appear as sprites) ---
    row_marks: dict[int, list[int]] = defaultdict(list)
    col_marks: dict[int, list[int]] = defaultdict(list)
    for r in range(int(ir0), int(ir1) + 1):
        cl = int(g[int(r), int(fc0)])
        cr = int(g[int(r), int(fc1)])
        if cl == cr and cl != int(bg) and cl in move_colors:
            row_marks[int(cl)].append(int(r))
    for c in range(int(ic0), int(ic1) + 1):
        ct = int(g[int(fr0), int(c)])
        cb = int(g[int(fr1), int(c)])
        if ct == cb and ct != int(bg) and ct in move_colors:
            col_marks[int(ct)].append(int(c))

    targets: dict[int, list[tuple[int, int]]] = {}
    for col in set(row_marks.keys()) & set(col_marks.keys()):
        pts = {(int(r), int(c)) for r in row_marks[int(col)] for c in col_marks[int(col)]}
        targets[int(col)] = sorted(pts)

    out = g.copy()
    movers = [s for s in sprites if int(s["color"]) in targets]

    # Erase originals of moved sprites.
    for s in movers:
        r0, c0 = int(s["r0"]), int(s["c0"])
        for rr, cc in np.argwhere(s["mask"]):
            out[int(r0) + int(rr), int(c0) + int(cc)] = int(bg)

    # Bottom-to-top ordering: lower sprites first, upper sprites last (covering).
    movers.sort(key=lambda s: (-((int(s["r0"]) + int(s["r1"])) / 2.0), int(s["c0"])))

    for s in movers:
        col = int(s["color"])
        cr = (int(s["r0"]) + int(s["r1"])) // 2
        cc = (int(s["c0"]) + int(s["c1"])) // 2
        pix = [(int(s["r0"]) + int(rr), int(s["c0"]) + int(cc2)) for rr, cc2 in np.argwhere(s["mask"])]

        for tr, tc in targets[int(col)]:
            dr = int(tr - cr)
            dc = int(tc - cc)
            for r, c in pix:
                rr = int(r + dr)
                cc3 = int(c + dc)
                if int(ir0) <= rr <= int(ir1) and int(ic0) <= cc3 <= int(ic1):
                    out[rr, cc3] = int(col)

    return out


# ---------------------------------------------------------------------------
# generate_input
# ---------------------------------------------------------------------------
import random


def _pick_spaced_marks(candidates: list[int], n: int, min_gap: int) -> list[int] | None:
    if not candidates:
        return None
    random.shuffle(candidates)
    if n == 1:
        return [candidates[0]]
    picked: list[int] = []
    for v in candidates:
        if all(abs(v - p) >= min_gap for p in picked):
            picked.append(v)
            if len(picked) == n:
                return picked
    return picked if picked else None


def _random_walk_sprite(color: int, max_h: int, max_w: int) -> np.ndarray | None:
    sh = min(random.choice([2, 3, 3, 4, 5]), max_h)
    sw = min(random.choice([2, 3, 3, 4, 5]), max_w)
    if sh < 1 or sw < 1:
        return None
    n_target = random.randint(3, min(9, sh * sw))
    cells = {(sh // 2, sw // 2)}
    r, c = sh // 2, sw // 2
    dirs = [(-1, -1), (-1, 0), (-1, 1), (0, -1), (0, 1), (1, -1), (1, 0), (1, 1)]
    for _ in range(n_target * 40):
        if len(cells) >= n_target:
            break
        dr, dc = random.choice(dirs)
        nr, nc = r + dr, c + dc
        if 0 <= nr < sh and 0 <= nc < sw:
            cells.add((nr, nc))
            r, c = nr, nc
    if len(cells) < 3:
        return None
    rs = [p[0] for p in cells]
    cs = [p[1] for p in cells]
    r0, r1 = min(rs), max(rs)
    c0, c1 = min(cs), max(cs)
    patch = np.zeros((r1 - r0 + 1, c1 - c0 + 1), dtype=int)
    for pr, pc in cells:
        patch[pr - r0, pc - c0] = color
    return patch


def _gen_attempt() -> np.ndarray | None:
    bg = 0

    H = random.randint(10, 15)
    interior_w = random.randint(8, 13)
    sprite_area_w = random.randint(3, 5)

    frame_w = interior_w + 2
    W = sprite_area_w + 1 + frame_w
    if not (14 <= W <= 20):
        return None

    div_col = sprite_area_w
    fc0 = div_col + 1
    fc1 = W - 1
    ir0, ir1 = 1, H - 2
    ic0, ic1 = fc0 + 1, fc1 - 1

    palette = list(range(1, 10))
    random.shuffle(palette)
    ci = 0
    divider_color = palette[ci]; ci += 1
    frame_base = palette[ci]; ci += 1
    n_movers = random.randint(2, 3)
    n_dist = random.choice([0, 0, 1])
    if ci + n_movers + n_dist > len(palette):
        return None
    mover_colors = palette[ci:ci + n_movers]; ci += n_movers
    dist_colors = palette[ci:ci + n_dist]; ci += n_dist

    mover_sprites: list[np.ndarray] = []
    for mc in mover_colors:
        sp = _random_walk_sprite(mc, 5, sprite_area_w)
        if sp is None:
            return None
        mover_sprites.append(sp)

    mover_marks: dict[int, tuple[list[int], list[int]]] = {}
    used_row_ranges: list[tuple[int, int]] = []
    used_col_ranges: list[tuple[int, int]] = []
    for mc, sp in zip(mover_colors, mover_sprites):
        sh, sw = sp.shape
        top_off = (sh - 1) // 2
        bot_off = sh - 1 - top_off
        left_off = (sw - 1) // 2
        right_off = sw - 1 - left_off

        row_lo = ir0 + top_off
        row_hi = ir1 - bot_off
        col_lo = ic0 + left_off
        col_hi = ic1 - right_off
        if row_lo > row_hi or col_lo > col_hi:
            return None

        avail_r = [r for r in range(row_lo, row_hi + 1)
                   if all(r + bot_off < elo or r - top_off > ehi
                          for elo, ehi in used_row_ranges)]
        rm = _pick_spaced_marks(avail_r, random.randint(1, 2), sh)
        if not rm:
            return None

        avail_c = [c for c in range(col_lo, col_hi + 1)
                   if all(c + right_off < elo or c - left_off > ehi
                          for elo, ehi in used_col_ranges)]
        cm = _pick_spaced_marks(avail_c, random.randint(1, 2), sw)
        if not cm:
            return None

        mover_marks[mc] = (sorted(rm), sorted(cm))
        for r in rm:
            used_row_ranges.append((r - top_off, r + bot_off))
        for c in cm:
            used_col_ranges.append((c - left_off, c + right_off))

    grid = np.full((H, W), bg, dtype=int)
    grid[:, div_col] = divider_color

    for c in range(fc0, fc1 + 1):
        grid[0, c] = frame_base
        grid[H - 1, c] = frame_base
    for mc, (_, cmarks) in mover_marks.items():
        for cc in cmarks:
            grid[0, cc] = mc
            grid[H - 1, cc] = mc

    for r in range(ir0, ir1 + 1):
        val = frame_base
        for mc, (rmarks, _) in mover_marks.items():
            if r in rmarks:
                val = mc
                break
        grid[r, fc0] = val
        grid[r, fc1] = val

    items: list[np.ndarray] = list(mover_sprites)
    for dc in dist_colors:
        ds = _random_walk_sprite(dc, 5, sprite_area_w)
        if ds is None:
            return None
        items.append(ds)

    total_sprite_h = sum(s.shape[0] for s in items)
    slack = H - total_sprite_h
    if slack < 0:
        return None

    n_gaps = len(items) + 1
    gaps = [0] * n_gaps
    for _ in range(slack):
        gaps[random.randint(0, n_gaps - 1)] += 1

    cur_row = gaps[0]
    for i, sp in enumerate(items):
        sh, sw = sp.shape
        if cur_row + sh > H or sw > sprite_area_w:
            return None
        c_off = random.randint(0, sprite_area_w - sw)
        for dr in range(sh):
            for dc in range(sw):
                if sp[dr, dc] != bg:
                    grid[cur_row + dr, c_off + dc] = sp[dr, dc]
        cur_row += sh + gaps[i + 1]

    vals, counts = np.unique(grid, return_counts=True)
    if int(vals[np.argmax(counts)]) != bg:
        return None

    try:
        out = main(grid)
    except Exception:
        return None
    if np.array_equal(grid, out):
        return None
    return grid


def generate_input() -> np.ndarray:
    for _ in range(300):
        result = _gen_attempt()
        if result is not None:
            return result
    raise RuntimeError("generate_input: failed after 300 attempts")
