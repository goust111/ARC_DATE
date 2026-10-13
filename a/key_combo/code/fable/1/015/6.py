from __future__ import annotations

import random
from collections import Counter

import numpy as np


# Minimal canonical helpers mirrored from arc_utils so this file stays loadable
# via SourceFileLoader without depending on import paths.
def _bbox(mask: np.ndarray) -> tuple[int, int, int, int] | None:
    coords = np.argwhere(mask)
    if coords.size == 0:
        return None
    r0, c0 = coords.min(axis=0)
    r1, c1 = coords.max(axis=0)
    return int(r0), int(c0), int(r1), int(c1)


def _crop_bbox(grid: np.ndarray, bbox: tuple[int, int, int, int]) -> np.ndarray:
    r0, c0, r1, c1 = bbox
    return grid[r0 : r1 + 1, c0 : c1 + 1].copy()


def _paste(
    dst: np.ndarray,
    patch: np.ndarray,
    r0: int,
    c0: int,
    *,
    transparent: int | None = None,
    clip: bool = False,
) -> np.ndarray:
    h, w = patch.shape
    r0 = int(r0)
    c0 = int(c0)
    if not clip:
        if r0 < 0 or c0 < 0 or r0 + h > dst.shape[0] or c0 + w > dst.shape[1]:
            raise ValueError("paste out of bounds")
        rr0, cc0 = r0, c0
        rr1, cc1 = r0 + h, c0 + w
        pr0, pc0 = 0, 0
    else:
        rr0 = max(0, r0)
        cc0 = max(0, c0)
        rr1 = min(dst.shape[0], r0 + h)
        cc1 = min(dst.shape[1], c0 + w)
        if rr0 >= rr1 or cc0 >= cc1:
            return dst
        pr0 = rr0 - r0
        pc0 = cc0 - c0
    pr1 = pr0 + (rr1 - rr0)
    pc1 = pc0 + (cc1 - cc0)
    view = patch[pr0:pr1, pc0:pc1]
    target = dst[rr0:rr1, cc0:cc1]
    if transparent is None:
        target[:, :] = view
    else:
        mask = view != int(transparent)
        target[mask] = view[mask]
    return dst


def _mode_color(arrays: list[np.ndarray]) -> int:
    vals = np.concatenate([a.ravel() for a in arrays])
    counts = Counter(map(int, vals))
    return int(counts.most_common(1)[0][0])


def _parse_windows(grid: np.ndarray) -> tuple[int, list[np.ndarray]]:
    g = np.asarray(grid, dtype=int)
    assert g.ndim == 2
    h, w = g.shape
    candidates: list[tuple[int, list[np.ndarray]]] = []

    for color in map(int, np.unique(g)):
        rows = np.where(np.all(g == color, axis=1))[0]
        cols = np.where(np.all(g == color, axis=0))[0]
        if len(rows) != 2 or rows[0] != 0 or rows[-1] != h - 1:
            continue
        if len(cols) < 3 or cols[0] != 0 or cols[-1] != w - 1:
            continue
        gaps = np.diff(cols)
        if np.any(gaps <= 1) or len(set(map(int, gaps))) != 1:
            continue
        window_w = int(gaps[0] - 1)
        if window_w <= 0 or h <= 2:
            continue

        windows: list[np.ndarray] = []
        ok = True
        for c0, c1 in zip(cols[:-1], cols[1:]):
            patch = g[1:-1, int(c0) + 1 : int(c1)].copy()
            if patch.shape[1] != window_w or np.any(patch == color):
                ok = False
                break
            windows.append(patch)
        if ok:
            candidates.append((color, windows))

    assert len(candidates) == 1, "expected one frame color defining equal windows"
    return candidates[0]


def _overlap_slices(shape: tuple[int, int], dr: int, dc: int) -> tuple[slice, slice, slice, slice] | None:
    h, w = map(int, shape)
    ar0 = max(0, -int(dr))
    ac0 = max(0, -int(dc))
    ar1 = min(h, h - int(dr))
    ac1 = min(w, w - int(dc))
    if ar0 >= ar1 or ac0 >= ac1:
        return None
    br0 = ar0 + int(dr)
    bc0 = ac0 + int(dc)
    br1 = ar1 + int(dr)
    bc1 = ac1 + int(dc)
    return slice(ar0, ar1), slice(ac0, ac1), slice(br0, br1), slice(bc0, bc1)


def _valid_shift_scores(windows: list[np.ndarray], bg: int) -> list[tuple[int, int, int]]:
    h, w = windows[0].shape
    scores: list[tuple[int, int, int]] = []
    for dr in range(-(h - 1), h):
        for dc in range(-(w - 1), w):
            if dr == 0 and dc == 0:
                continue
            total_support = 0
            ok = True
            for a, b in zip(windows[:-1], windows[1:]):
                slices = _overlap_slices(a.shape, dr, dc)
                if slices is None:
                    ok = False
                    break
                ars, acs, brs, bcs = slices
                av = a[ars, acs]
                bv = b[brs, bcs]
                if not np.array_equal(av, bv):
                    ok = False
                    break
                support = int(np.count_nonzero(av != int(bg)))
                if support == 0:
                    ok = False
                    break
                total_support += support
            if ok:
                scores.append((total_support, int(dr), int(dc)))
    return scores


def _infer_shift(windows: list[np.ndarray], bg: int) -> tuple[int, int]:
    scores = _valid_shift_scores(windows, bg)
    assert scores, "expected at least one consistent inter-frame shift"
    best_score = max(score for score, _dr, _dc in scores)
    best = [(dr, dc) for score, dr, dc in scores if score == best_score]
    assert len(best) == 1, "inter-frame shift must be uniquely determined"
    return best[0]


def _reconstruct_sprite(windows: list[np.ndarray], bg: int, dr: int, dc: int) -> np.ndarray:
    h, w = windows[0].shape
    offsets = [(-k * int(dr), -k * int(dc)) for k in range(len(windows))]
    min_r = min(r for r, _c in offsets)
    min_c = min(c for _r, c in offsets)
    max_r = max(r for r, _c in offsets)
    max_c = max(c for _r, c in offsets)

    canvas = np.full((h + max_r - min_r, w + max_c - min_c), int(bg), dtype=int)
    for (shift_r, shift_c), frame in zip(offsets, windows):
        r0 = shift_r - min_r
        c0 = shift_c - min_c
        region = canvas[r0 : r0 + h, c0 : c0 + w]
        mask = frame != int(bg)
        overlap = mask & (region != int(bg))
        assert np.all(region[overlap] == frame[overlap]), "aligned fragments must agree on colors"
        region[mask] = frame[mask]

    box = _bbox(canvas != int(bg))
    assert box is not None, "reconstruction must contain foreground"
    return _crop_bbox(canvas, box)


def main(grid: np.ndarray) -> np.ndarray:
    """
    Reverse-translate sequential viewing windows, union their visible fragments, and crop the complete moving sprite.

    Input structure:
    - One line color forms a single horizontal strip of equal-sized viewing windows with a top/bottom border and vertical separators.
    - Inside the windows, one multicolor sprite moves by a constant per-frame displacement against a uniform background.
    - Each window shows only the clipped intersection of that sprite with the fixed viewing area, so no single frame contains the whole sprite.

    Rule:
    - Identify the frame color and extract the equal-sized window interiors in left-to-right time order.
    - Infer the constant shift vector by finding the unique displacement that makes every adjacent pair agree exactly on their overlapping visible region.
    - Move window k back by k times that shift, take the union of all non-background pixels, and output the tight bbox crop of the reconstructed sprite.

    Key insight:
    - The windows are not separate objects: they are consecutive clips of one rigid sprite, so only inverting the motion before unioning the fragments reveals the full shape and preserves internal holes.
    """
    g = np.asarray(grid, dtype=int)
    _frame_color, windows = _parse_windows(g)
    assert len(windows) >= 2, "expected multiple frames"
    bg = _mode_color(windows)
    dr, dc = _infer_shift(windows, bg)
    return _reconstruct_sprite(windows, bg, dr, dc)


PALETTE = tuple(range(10))
GEN_ATTEMPTS = 400


def _connected(mask: np.ndarray) -> bool:
    coords = np.argwhere(mask)
    if coords.size == 0:
        return False
    seen = {tuple(map(int, coords[0]))}
    stack = [tuple(map(int, coords[0]))]
    h, w = mask.shape
    while stack:
        r, c = stack.pop()
        for nr, nc in ((r - 1, c), (r + 1, c), (r, c - 1), (r, c + 1)):
            if 0 <= nr < h and 0 <= nc < w and bool(mask[nr, nc]) and (nr, nc) not in seen:
                seen.add((nr, nc))
                stack.append((nr, nc))
    return len(seen) == int(mask.sum())


def _make_sprite_mask(rng: random.Random, h: int, w: int) -> np.ndarray:
    for _ in range(200):
        top = 1 if h <= 6 else rng.choice([1, 2])
        bottom = 1 if h <= 6 else rng.choice([1, 2])
        left = 1 if w <= 6 else rng.choice([1, 2])
        right = 1 if w <= 6 else rng.choice([1, 2])
        hole_h = h - top - bottom
        hole_w = w - left - right
        if hole_h < 2 or hole_w < 2:
            continue

        mask = np.ones((h, w), dtype=bool)
        mask[top : h - bottom, left : w - right] = False

        tab_count = rng.randint(0, 3)
        for _tab in range(tab_count):
            side = rng.choice(["top", "bottom", "left", "right"])
            if side in {"top", "bottom"} and hole_w >= 2:
                span = rng.randint(1, max(1, hole_w - 1))
                c0 = rng.randint(left, w - right - span)
                depth = rng.randint(1, max(1, hole_h - 1))
                if side == "top":
                    mask[top : top + depth, c0 : c0 + span] = True
                else:
                    mask[h - bottom - depth : h - bottom, c0 : c0 + span] = True
            elif side in {"left", "right"} and hole_h >= 2:
                span = rng.randint(1, max(1, hole_h - 1))
                r0 = rng.randint(top, h - bottom - span)
                depth = rng.randint(1, max(1, hole_w - 1))
                if side == "left":
                    mask[r0 : r0 + span, left : left + depth] = True
                else:
                    mask[r0 : r0 + span, w - right - depth : w - right] = True

        if not _connected(mask):
            continue
        if not (mask[0, :].any() and mask[-1, :].any() and mask[:, 0].any() and mask[:, -1].any()):
            continue
        inner_bg = (~mask)[1:-1, 1:-1]
        if not inner_bg.any():
            continue
        area = int(mask.sum())
        if not (max(h + w, (h * w) // 3) <= area <= (h * w * 4) // 5):
            continue
        return mask
    raise RuntimeError("failed to build sprite mask")


def _pick_spread_seeds(rng: random.Random, coords: np.ndarray, k: int) -> list[tuple[int, int]]:
    seeds = [tuple(map(int, coords[rng.randrange(len(coords))]))]
    while len(seeds) < k:
        best = None
        best_score = -1
        for _ in range(40):
            cand = tuple(map(int, coords[rng.randrange(len(coords))]))
            score = min(abs(cand[0] - sr) + abs(cand[1] - sc) for sr, sc in seeds)
            if score > best_score and cand not in seeds:
                best_score = score
                best = cand
        assert best is not None
        seeds.append(best)
    return seeds


def _color_sprite(
    rng: random.Random,
    mask: np.ndarray,
    bg: int,
    colors: list[int],
) -> np.ndarray:
    coords = np.argwhere(mask)
    max_k = min(4, len(colors), max(2, len(coords) // 6))
    min_k = min(2, max_k)
    for _ in range(120):
        k = rng.randint(min_k, max_k)
        palette = colors[:k]
        seeds = _pick_spread_seeds(rng, coords, k)
        sprite = np.full(mask.shape, int(bg), dtype=int)
        for r, c in coords:
            rr = int(r)
            cc = int(c)
            best_idx = min(
                range(k),
                key=lambda idx: (abs(rr - seeds[idx][0]) + abs(cc - seeds[idx][1]), idx),
            )
            sprite[rr, cc] = int(palette[best_idx])

        if k >= 3 and rng.random() < 0.7:
            accent = palette[-1]
            orientation = rng.choice(["row", "col"])
            if orientation == "row":
                rr = rng.randrange(mask.shape[0])
                row_mask = mask[rr]
                sprite[rr, row_mask] = accent
            else:
                cc = rng.randrange(mask.shape[1])
                col_mask = mask[:, cc]
                sprite[col_mask, cc] = accent

        used = [int(v) for v in np.unique(sprite) if int(v) != int(bg)]
        if len(used) < 2:
            continue
        counts = {color: int(np.count_nonzero(sprite == color)) for color in used}
        if min(counts.values()) < 2:
            continue
        return sprite
    raise RuntimeError("failed to color sprite")


def _build_sprite(
    rng: random.Random,
    h: int,
    w: int,
    bg: int,
    colors: list[int],
) -> np.ndarray:
    mask = _make_sprite_mask(rng, h, w)
    sprite = _color_sprite(rng, mask, bg, colors)
    box = _bbox(sprite != int(bg))
    assert box == (0, 0, h - 1, w - 1)
    return sprite


def _choose_layout(rng: random.Random) -> tuple[int, int, int, int, int]:
    n_frames = rng.choice([3, 4])
    if n_frames == 3:
        window_h = rng.randint(5, 8)
        window_w = rng.randint(6, 8)
    else:
        window_h = rng.randint(5, 8)
        window_w = rng.randint(4, 6)
    mode = rng.choice(["horizontal", "vertical", "diagonal"])
    if mode == "horizontal":
        dr = 0
        dc = rng.choice([-2, -1, 1, 2])
    elif mode == "vertical":
        dr = rng.choice([-2, -1, 1, 2])
        dc = 0
    else:
        dr = rng.choice([-2, -1, 1, 2])
        dc = rng.choice([-2, -1, 1, 2])
    return n_frames, window_h, window_w, dr, dc


def _render_windows(
    sprite: np.ndarray,
    bg: int,
    n_frames: int,
    window_h: int,
    window_w: int,
    dr: int,
    dc: int,
) -> list[np.ndarray]:
    total_dr = (n_frames - 1) * int(dr)
    total_dc = (n_frames - 1) * int(dc)
    base_r = -max(total_dr, 0)
    base_c = -max(total_dc, 0)
    windows: list[np.ndarray] = []
    for k in range(n_frames):
        frame = np.full((window_h, window_w), int(bg), dtype=int)
        _paste(
            frame,
            sprite,
            base_r + k * int(dr),
            base_c + k * int(dc),
            transparent=bg,
            clip=True,
        )
        windows.append(frame)
    return windows


def _assemble_input(frame_color: int, windows: list[np.ndarray]) -> np.ndarray:
    window_h, window_w = windows[0].shape
    n_frames = len(windows)
    grid = np.full((window_h + 2, n_frames * window_w + n_frames + 1), int(frame_color), dtype=int)
    for idx, window in enumerate(windows):
        c0 = 1 + idx * (window_w + 1)
        grid[1:-1, c0 : c0 + window_w] = window
    return grid


def _semantic_checks(
    grid: np.ndarray,
    sprite: np.ndarray,
    windows: list[np.ndarray],
    bg: int,
    dr: int,
    dc: int,
) -> bool:
    interior_counts = Counter(map(int, np.concatenate([w.ravel() for w in windows])))
    bg_count = interior_counts[int(bg)]
    if any(color != int(bg) and count >= bg_count for color, count in interior_counts.items()):
        return False

    scores = _valid_shift_scores(windows, bg)
    if not scores:
        return False
    best_score = max(score for score, _dr, _dc in scores)
    best = [(cand_dr, cand_dc) for score, cand_dr, cand_dc in scores if score == best_score]
    if best != [(int(dr), int(dc))]:
        return False

    for a, b in zip(windows[:-1], windows[1:]):
        slices = _overlap_slices(a.shape, dr, dc)
        if slices is None:
            return False
        ars, acs, brs, bcs = slices
        av = a[ars, acs]
        bv = b[brs, bcs]
        if not np.array_equal(av, bv):
            return False
        if int(np.count_nonzero(av != int(bg))) == 0:
            return False

    if not np.array_equal(_reconstruct_sprite(windows, bg, dr, dc), sprite):
        return False

    if all(np.array_equal(window, sprite) for window in windows):
        return False

    try:
        solved = main(grid)
    except Exception:
        return False
    return np.array_equal(solved, sprite)


def _gen_attempt() -> np.ndarray | None:
    rng = random.Random()
    n_frames, window_h, window_w, dr, dc = _choose_layout(rng)
    total_dr = abs((n_frames - 1) * dr)
    total_dc = abs((n_frames - 1) * dc)
    sprite_h = window_h + total_dr
    sprite_w = window_w + total_dc

    colors = list(PALETTE)
    rng.shuffle(colors)
    bg = colors.pop()
    frame_color = colors.pop()
    sprite = _build_sprite(rng, sprite_h, sprite_w, bg, colors)
    windows = _render_windows(sprite, bg, n_frames, window_h, window_w, dr, dc)

    if any(int(np.count_nonzero(window != int(bg))) == 0 for window in windows):
        return None

    grid = _assemble_input(frame_color, windows)
    if max(grid.shape) > 30 or max(sprite.shape) > 30:
        return None
    if not _semantic_checks(grid, sprite, windows, bg, dr, dc):
        return None
    return grid


def generate_input() -> np.ndarray:
    for _ in range(GEN_ATTEMPTS):
        grid = _gen_attempt()
        if grid is not None:
            return grid
    raise RuntimeError("failed to generate a valid viewing-window sprite puzzle")
