import random
from collections import deque

import numpy as np


BG = 0
PALETTE = tuple(range(1, 10))
MAX_ATTEMPTS = 500
_RNG = random.Random()


def _neighbors4(r: int, c: int, h: int, w: int):
    if r > 0:
        yield r - 1, c
    if r + 1 < h:
        yield r + 1, c
    if c > 0:
        yield r, c - 1
    if c + 1 < w:
        yield r, c + 1


def _components(mask: np.ndarray) -> list[np.ndarray]:
    h, w = mask.shape
    seen = np.zeros((h, w), dtype=bool)
    comps: list[np.ndarray] = []
    for r in range(h):
        for c in range(w):
            if not mask[r, c] or seen[r, c]:
                continue
            seen[r, c] = True
            q = deque([(r, c)])
            pts = []
            while q:
                rr, cc = q.popleft()
                pts.append((rr, cc))
                for nr, nc in _neighbors4(rr, cc, h, w):
                    if mask[nr, nc] and not seen[nr, nc]:
                        seen[nr, nc] = True
                        q.append((nr, nc))
            comps.append(np.array(pts, dtype=int))
    return comps


def _bbox(coords: np.ndarray) -> tuple[int, int, int, int]:
    rows = coords[:, 0]
    cols = coords[:, 1]
    return int(rows.min()), int(cols.min()), int(rows.max()), int(cols.max())


def _contiguous_spans(row_mask: np.ndarray) -> list[tuple[int, int]]:
    spans: list[tuple[int, int]] = []
    start = None
    for idx, val in enumerate(row_mask.tolist()):
        if val and start is None:
            start = idx
        elif (not val) and start is not None:
            spans.append((start, idx - 1))
            start = None
    if start is not None:
        spans.append((start, len(row_mask) - 1))
    return spans


def _find_legend(grid: np.ndarray) -> tuple[str, int, int, int, int]:
    g = np.asarray(grid, dtype=int)
    h, w = g.shape
    assert w >= 4, "grid must be wide enough for a 2-column legend and a gap"

    all_colors = {int(c) for c in np.unique(g) if int(c) != BG}
    assert len(all_colors) >= 4 and len(all_colors) % 2 == 0

    candidates: list[tuple[str, int, int, int, int]] = []
    side_specs = [("left", 0, 1, 2), ("right", w - 2, w - 1, w - 3)]

    for side, c0, c1, sep_c in side_specs:
        row_mask = (g[:, c0] != BG) & (g[:, c1] != BG) & (g[:, sep_c] == BG)
        for r0, r1 in _contiguous_spans(row_mask):
            colors: list[int] = []
            ok = True
            for r in range(r0, r1 + 1):
                left = int(g[r, c0])
                right = int(g[r, c1])
                if left == right:
                    ok = False
                    break
                colors.extend([left, right])
            if not ok or len(set(colors)) != len(colors):
                continue

            legend_colors = set(colors)
            if legend_colors != all_colors:
                continue

            remaining = g.copy()
            remaining[r0 : r1 + 1, c0 : c1 + 1] = BG
            rem_colors = {int(c) for c in np.unique(remaining) if int(c) != BG}
            if rem_colors != legend_colors:
                continue

            if any(len(_components(remaining == color)) != 1 for color in legend_colors):
                continue

            candidates.append((side, r0, r1, c0, c1))

    assert len(candidates) == 1, "legend must be uniquely identifiable"
    return candidates[0]


def main(grid: np.ndarray) -> np.ndarray:
    g = np.asarray(grid, dtype=int)
    legend_side, r0, r1, c0, c1 = _find_legend(g)
    _ = legend_side

    pairs = [(int(g[r, c0]), int(g[r, c1])) for r in range(r0, r1 + 1)]

    work = g.copy()
    work[r0 : r1 + 1, c0 : c1 + 1] = BG
    out = work.copy()

    colors = {color for pair in pairs for color in pair}
    objects = {}
    for color in colors:
        comps = _components(work == color)
        assert len(comps) == 1, "each object color must appear in exactly one component"
        objects[color] = comps[0]

    for color_a, color_b in pairs:
        coords_a = objects[color_a]
        coords_b = objects[color_b]
        area_a = int(len(coords_a))
        area_b = int(len(coords_b))
        assert area_a != area_b, "paired objects must differ in area"

        if area_a < area_b:
            small_color, small_coords = color_a, coords_a
            big_color, big_coords = color_b, coords_b
        else:
            small_color, small_coords = color_b, coords_b
            big_color, big_coords = color_a, coords_a
        _ = big_color

        for rr, cc in small_coords:
            out[int(rr), int(cc)] = BG

        br0, bc0, br1, bc1 = _bbox(big_coords)
        sr0, sc0, sr1, sc1 = _bbox(small_coords)
        big_h = br1 - br0 + 1
        big_w = bc1 - bc0 + 1
        small_h = sr1 - sr0 + 1
        small_w = sc1 - sc0 + 1

        place_r = br0 + (big_h - small_h) // 2
        place_c = bc0 + (big_w - small_w) // 2
        big_set = {(int(rr), int(cc)) for rr, cc in big_coords}

        for rr, cc in small_coords:
            tr = place_r + int(rr) - sr0
            tc = place_c + int(cc) - sc0
            if (tr, tc) in big_set:
                out[tr, tc] = int(small_color)

    return out


def _rot(mask: np.ndarray, k: int) -> np.ndarray:
    return np.rot90(mask, int(k)).copy()


def _mask_rect(h: int, w: int) -> np.ndarray:
    return np.ones((h, w), dtype=bool)


def _mask_frame(h: int, w: int, thickness: int = 1) -> np.ndarray:
    m = np.zeros((h, w), dtype=bool)
    t = int(thickness)
    m[:t, :] = True
    m[-t:, :] = True
    m[:, :t] = True
    m[:, -t:] = True
    return m


def _mask_plus(h: int, w: int, row_thick: int = 1, col_thick: int = 1) -> np.ndarray:
    m = np.zeros((h, w), dtype=bool)
    rt = int(row_thick)
    ct = int(col_thick)
    r0 = (h - rt) // 2
    c0 = (w - ct) // 2
    m[r0 : r0 + rt, :] = True
    m[:, c0 : c0 + ct] = True
    return m


def _mask_l(h: int, w: int, rot: int) -> np.ndarray:
    m = np.zeros((h, w), dtype=bool)
    m[0, :] = True
    m[:, 0] = True
    return _rot(m, rot)


def _mask_t(h: int, w: int, rot: int) -> np.ndarray:
    m = np.zeros((h, w), dtype=bool)
    m[0, :] = True
    m[:, w // 2] = True
    return _rot(m, rot)


def _mask_stairs(h: int, w: int, rot: int) -> np.ndarray:
    m = np.zeros((h, w), dtype=bool)
    col = 0
    m[0, 0] = True
    for r in range(1, h):
        new_col = int(round(r * (w - 1) / max(1, h - 1)))
        lo = min(col, new_col)
        hi = max(col, new_col)
        m[r, lo : hi + 1] = True
        m[r, col] = True
        col = new_col
    return _rot(m, rot)


def _mask_h(h: int, w: int, bar_w: int, mid_h: int) -> np.ndarray:
    m = np.zeros((h, w), dtype=bool)
    bw = int(bar_w)
    mh = int(mid_h)
    m[:, :bw] = True
    m[:, -bw:] = True
    r0 = (h - mh) // 2
    m[r0 : r0 + mh, :] = True
    return m


def _mask_u(h: int, w: int, bar_w: int, base_h: int, rot: int) -> np.ndarray:
    m = np.zeros((h, w), dtype=bool)
    bw = int(bar_w)
    bh = int(base_h)
    m[:, :bw] = True
    m[:, -bw:] = True
    m[-bh:, :] = True
    return _rot(m, rot)


def _mask_c(h: int, w: int, bar_w: int, rail_h: int, rot: int) -> np.ndarray:
    m = np.zeros((h, w), dtype=bool)
    bw = int(bar_w)
    rh = int(rail_h)
    m[:, :bw] = True
    m[:rh, :] = True
    m[-rh:, :] = True
    return _rot(m, rot)


def _mask_dented_rect(h: int, w: int, corner: str, cut_h: int, cut_w: int) -> np.ndarray:
    m = np.ones((h, w), dtype=bool)
    ch = int(cut_h)
    cw = int(cut_w)
    if corner == "tl":
        m[:ch, :cw] = False
    elif corner == "tr":
        m[:ch, -cw:] = False
    elif corner == "bl":
        m[-ch:, :cw] = False
    else:
        m[-ch:, -cw:] = False
    return m


def _dedup_masks(masks: list[np.ndarray]) -> list[np.ndarray]:
    uniq = {}
    for mask in masks:
        uniq[(mask.shape, mask.tobytes())] = mask
    return list(uniq.values())


def _small_mask_candidates(h: int, w: int) -> list[np.ndarray]:
    masks = [_mask_rect(h, w)]
    for rot in range(4):
        masks.append(_mask_l(h, w, rot))
        masks.append(_mask_t(h, w, rot))
        masks.append(_mask_stairs(h, w, rot))
    if h >= 3 and w >= 3:
        masks.append(_mask_frame(h, w, 1))
        masks.append(_mask_plus(h, w, 1, 1))
    return _dedup_masks(masks)


def _big_mask_candidates(h: int, w: int) -> list[np.ndarray]:
    masks = [_mask_rect(h, w)]

    if h >= 4 and w >= 4:
        max_cut_h = min(2, h - 2)
        max_cut_w = min(2, w - 2)
        for corner in ("tl", "tr", "bl", "br"):
            for cut_h in range(1, max_cut_h + 1):
                for cut_w in range(1, max_cut_w + 1):
                    masks.append(_mask_dented_rect(h, w, corner, cut_h, cut_w))

    if h >= 3 and w >= 3:
        max_thick = min(2, (min(h, w) - 1) // 2)
        for thickness in range(1, max_thick + 1):
            masks.append(_mask_frame(h, w, thickness))

    for row_thick in range(1, min(2, h) + 1):
        for col_thick in range(1, min(2, w) + 1):
            masks.append(_mask_plus(h, w, row_thick, col_thick))

    max_bar_w = min(2, max(1, w // 2))
    max_mid_h = min(2, max(1, h // 2))
    for bar_w in range(1, max_bar_w + 1):
        for mid_h in range(1, max_mid_h + 1):
            masks.append(_mask_h(h, w, bar_w, mid_h))
            for rot in range(4):
                masks.append(_mask_u(h, w, bar_w, mid_h, rot))
                masks.append(_mask_c(h, w, bar_w, mid_h, rot))

    return _dedup_masks(masks)


def _centered_overlap(big_mask: np.ndarray, small_mask: np.ndarray) -> int:
    bh, bw = big_mask.shape
    sh, sw = small_mask.shape
    assert sh <= bh and sw <= bw
    r0 = (bh - sh) // 2
    c0 = (bw - sw) // 2
    centered = np.zeros((bh, bw), dtype=bool)
    centered[r0 : r0 + sh, c0 : c0 + sw] = small_mask
    return int(np.count_nonzero(big_mask & centered))


def _make_small_mask() -> np.ndarray:
    for _ in range(200):
        h = _RNG.randint(2, 5)
        w = _RNG.randint(2, 5)
        masks = _small_mask_candidates(h, w)
        mask = _RNG.choice(masks)
        area = int(mask.sum())
        if 3 <= area <= 16:
            return mask
    raise RuntimeError("failed to create small mask")


def _make_big_mask(small_mask: np.ndarray, mode: str) -> np.ndarray:
    sh, sw = small_mask.shape
    small_area = int(small_mask.sum())

    if mode == "partial":
        h_lo = max(3, sh)
        h_hi = max(h_lo, min(8, sh + 2))
        w_lo = max(3, sw)
        w_hi = max(w_lo, min(8, sw + 2))
    else:
        h_lo = max(3, sh)
        h_hi = max(h_lo, min(9, sh + 4))
        w_lo = max(3, sw)
        w_hi = max(w_lo, min(9, sw + 4))

    dims = [(h, w) for h in range(h_lo, h_hi + 1) for w in range(w_lo, w_hi + 1)]
    _RNG.shuffle(dims)

    for h, w in dims:
        masks = _big_mask_candidates(h, w)
        _RNG.shuffle(masks)
        for mask in masks:
            mh, mw = mask.shape
            if mh < sh or mw < sw:
                continue
            area = int(mask.sum())
            if area <= small_area:
                continue
            overlap = _centered_overlap(mask, small_mask)
            if mode == "full" and overlap == small_area:
                return mask
            if mode == "partial" and 0 < overlap < small_area:
                return mask

    raise RuntimeError(f"failed to create {mode} big mask")


def _find_position(reserved: np.ndarray, h: int, w: int) -> tuple[int, int] | None:
    grid_h, grid_w = reserved.shape
    candidates = [(r, c) for r in range(grid_h - h + 1) for c in range(grid_w - w + 1)]
    _RNG.shuffle(candidates)
    for r, c in candidates:
        r0 = max(0, r - 1)
        c0 = max(0, c - 1)
        r1 = min(grid_h, r + h + 1)
        c1 = min(grid_w, c + w + 1)
        if not reserved[r0:r1, c0:c1].any():
            return r, c
    return None


def _expected_output(inp: np.ndarray, pair_specs: list[dict], legend_side: str, legend_top: int) -> np.ndarray:
    out = inp.copy()
    pair_count = len(pair_specs)
    if legend_side == "left":
        out[legend_top : legend_top + pair_count, 0:2] = BG
    else:
        out[legend_top : legend_top + pair_count, -2:] = BG

    for spec in pair_specs:
        sr, sc = spec["small_pos"]
        br, bc = spec["big_pos"]
        small_mask = spec["small_mask"]
        big_mask = spec["big_mask"]
        small_color = int(spec["small_color"])

        out[sr : sr + small_mask.shape[0], sc : sc + small_mask.shape[1]][small_mask] = BG

        sh, sw = small_mask.shape
        bh, bw = big_mask.shape
        off_r = br + (bh - sh) // 2
        off_c = bc + (bw - sw) // 2
        for rr, cc in np.argwhere(small_mask):
            tr = off_r + int(rr)
            tc = off_c + int(cc)
            if big_mask[tr - br, tc - bc]:
                out[tr, tc] = small_color

    return out


def _make_pair_specs(colors: list[int]) -> list[dict]:
    pair_specs: list[dict] = []
    color_pool = list(colors)
    _RNG.shuffle(color_pool)

    for idx in range(0, len(color_pool), 2):
        color_a = int(color_pool[idx])
        color_b = int(color_pool[idx + 1])
        small_mask = _make_small_mask()
        mode = _RNG.choice(["full", "partial"])
        big_mask = _make_big_mask(small_mask, mode)

        if _RNG.random() < 0.5:
            small_color, big_color = color_a, color_b
        else:
            small_color, big_color = color_b, color_a

        if _RNG.random() < 0.5:
            legend_left, legend_right = color_a, color_b
        else:
            legend_left, legend_right = color_b, color_a

        pair_specs.append(
            {
                "legend_left": legend_left,
                "legend_right": legend_right,
                "small_color": small_color,
                "big_color": big_color,
                "small_mask": small_mask,
                "big_mask": big_mask,
                "mode": mode,
            }
        )

    _RNG.shuffle(pair_specs)
    return pair_specs


def _gen_attempt() -> np.ndarray | None:
    pair_count = _RNG.choice([2, 3, 4])
    colors = _RNG.sample(PALETTE, 2 * pair_count)
    pair_specs = _make_pair_specs(colors)

    if pair_count == 2:
        h = _RNG.randint(16, 23)
        w = _RNG.randint(18, 25)
    elif pair_count == 3:
        h = _RNG.randint(18, 26)
        w = _RNG.randint(21, 28)
    else:
        h = _RNG.randint(22, 30)
        w = _RNG.randint(24, 30)

    if w < 6 or h < pair_count + 2:
        return None

    legend_side = _RNG.choice(["left", "right"])
    legend_top = _RNG.randint(0, h - pair_count)

    grid = np.zeros((h, w), dtype=int)
    reserved = np.zeros((h, w), dtype=bool)

    if legend_side == "left":
        reserved[:, :3] = True
        for i, spec in enumerate(pair_specs):
            grid[legend_top + i, 0] = int(spec["legend_left"])
            grid[legend_top + i, 1] = int(spec["legend_right"])
    else:
        reserved[:, -3:] = True
        for i, spec in enumerate(pair_specs):
            grid[legend_top + i, -2] = int(spec["legend_left"])
            grid[legend_top + i, -1] = int(spec["legend_right"])

    objects = []
    for pair_idx, spec in enumerate(pair_specs):
        objects.append(("big", pair_idx, int(spec["big_color"]), spec["big_mask"]))
        objects.append(("small", pair_idx, int(spec["small_color"]), spec["small_mask"]))
    objects.sort(key=lambda item: item[3].shape[0] * item[3].shape[1], reverse=True)

    for role, pair_idx, color, mask in objects:
        mh, mw = mask.shape
        pos = _find_position(reserved, mh, mw)
        if pos is None:
            return None
        r, c = pos
        grid[r : r + mh, c : c + mw][mask] = color
        reserved[max(0, r - 1) : min(h, r + mh + 1), max(0, c - 1) : min(w, c + mw + 1)] = True
        pair_specs[pair_idx][f"{role}_pos"] = (r, c)

    expected = _expected_output(grid, pair_specs, legend_side, legend_top)
    try:
        actual = main(grid)
    except Exception:
        return None
    if not np.array_equal(actual, expected):
        return None
    return grid


def generate_input() -> np.ndarray:
    for _ in range(MAX_ATTEMPTS):
        grid = _gen_attempt()
        if grid is not None:
            return grid
    raise RuntimeError("failed to generate valid input")
