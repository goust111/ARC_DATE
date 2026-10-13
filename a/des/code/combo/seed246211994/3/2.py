from __future__ import annotations

import random
from collections import deque

import numpy as np


def _get_bg(grid: np.ndarray) -> int:
    values = np.asarray(grid, dtype=int).ravel()
    counts = np.bincount(values, minlength=10)
    return int(np.argmax(counts))


def _neighbors4(r: int, c: int) -> tuple[tuple[int, int], ...]:
    return ((r - 1, c), (r + 1, c), (r, c - 1), (r, c + 1))


def _components(mask: np.ndarray) -> list[list[tuple[int, int]]]:
    h, w = mask.shape
    seen = np.zeros((h, w), dtype=bool)
    out: list[list[tuple[int, int]]] = []
    for r in range(h):
        for c in range(w):
            if not mask[r, c] or seen[r, c]:
                continue
            seen[r, c] = True
            q = deque([(r, c)])
            pts: list[tuple[int, int]] = []
            while q:
                cr, cc = q.popleft()
                pts.append((cr, cc))
                for nr, nc in _neighbors4(cr, cc):
                    if 0 <= nr < h and 0 <= nc < w and mask[nr, nc] and not seen[nr, nc]:
                        seen[nr, nc] = True
                        q.append((nr, nc))
            out.append(pts)
    return out


def _bbox(coords: list[tuple[int, int]]) -> tuple[int, int, int, int]:
    rs = [r for r, _ in coords]
    cs = [c for _, c in coords]
    return min(rs), min(cs), max(rs), max(cs)


def _border_mask(shape: tuple[int, int]) -> np.ndarray:
    h, w = shape
    mask = np.zeros((h, w), dtype=bool)
    mask[0, :] = True
    mask[-1, :] = True
    mask[:, 0] = True
    mask[:, -1] = True
    return mask


def _find_frame(grid: np.ndarray, bg: int) -> tuple[int, tuple[int, int, int, int]]:
    candidates: list[tuple[int, tuple[int, int, int, int]]] = []
    for color in map(int, np.unique(grid)):
        if color == bg:
            continue
        for pts in _components(grid == color):
            r0, c0, r1, c1 = _bbox(pts)
            if r1 - r0 < 2 or c1 - c0 < 2:
                continue
            sub = grid[r0 : r1 + 1, c0 : c1 + 1]
            border = _border_mask(sub.shape)
            if not np.all(sub[border] == color):
                continue
            if np.all(sub[~border] == color):
                continue
            inner = sub[1:-1, 1:-1]
            if not np.any((inner != bg) & (inner != color)):
                continue
            candidates.append((color, (r0, c0, r1, c1)))
    assert len(candidates) == 1, f"expected one frame, found {len(candidates)}"
    return candidates[0]


def _extract_placeholders(crop: np.ndarray, bg: int, frame_color: int) -> dict[int, list[tuple[int, int]]]:
    h, w = crop.shape
    inner = np.zeros((h, w), dtype=bool)
    inner[1:-1, 1:-1] = True
    mask = inner & (crop != bg) & (crop != frame_color)
    assert np.any(mask), "expected placeholders inside frame"

    placeholders: dict[int, list[tuple[int, int]]] = {}
    for pts in _components(mask):
        assert len(pts) == 1, "placeholders must be single pixels"
        r, c = pts[0]
        color = int(crop[r, c])
        placeholders.setdefault(color, []).append((r, c))
    return placeholders


def _find_legend(
    grid: np.ndarray,
    outside_mask: np.ndarray,
    bg: int,
    frame_color: int,
    placeholder_colors: set[int],
) -> tuple[dict[int, int], np.ndarray]:
    candidates: list[tuple[dict[int, int], np.ndarray]] = []
    fg_outside = outside_mask & (grid != bg)
    for pts in _components(fg_outside):
        r0, c0, r1, c1 = _bbox(pts)
        sub = grid[r0 : r1 + 1, c0 : c1 + 1]
        if sub.shape[1] != 2 or sub.shape[0] != len(placeholder_colors):
            continue
        if len(pts) != sub.size:
            continue
        if np.any(sub == bg) or np.any(sub == frame_color):
            continue
        left = [int(x) for x in sub[:, 0]]
        right = [int(x) for x in sub[:, 1]]
        if len(set(left)) != len(left):
            continue
        if set(left) != placeholder_colors:
            continue
        if any(l == r for l, r in zip(left, right)):
            continue
        mapping = {src: dst for src, dst in zip(left, right)}
        legend_mask = np.zeros_like(grid, dtype=bool)
        legend_mask[r0 : r1 + 1, c0 : c1 + 1] = True
        candidates.append((mapping, legend_mask))
    assert len(candidates) == 1, f"expected one legend, found {len(candidates)}"
    return candidates[0]


def _extract_sprites(
    grid: np.ndarray,
    search_mask: np.ndarray,
    placeholder_colors: set[int],
) -> dict[int, tuple[list[tuple[int, int]], int, int]]:
    sprites: dict[int, tuple[list[tuple[int, int]], int, int]] = {}
    for color in sorted(placeholder_colors):
        comps = _components(search_mask & (grid == color))
        assert len(comps) == 1, f"expected one outside sprite for color {color}, found {len(comps)}"
        pts = comps[0]
        r0, c0, r1, c1 = _bbox(pts)
        h = r1 - r0 + 1
        w = c1 - c0 + 1
        rel = sorted((r - r0, c - c0) for r, c in pts)
        sprites[color] = (rel, h, w)
    return sprites


def _stamp_centered(
    canvas: np.ndarray,
    center: tuple[int, int],
    sprite: list[tuple[int, int]],
    sprite_h: int,
    sprite_w: int,
    color: int,
) -> None:
    cr, cc = center
    ar = sprite_h // 2
    ac = sprite_w // 2
    for dr, dc in sprite:
        rr = cr + dr - ar
        cc2 = cc + dc - ac
        canvas[rr, cc2] = color


def main(grid: np.ndarray) -> np.ndarray:
    """
    Pair outside sprites with inside-frame placeholders by color, recolor from a legend, and stamp them.

    The frame crop is the output canvas. Each placeholder is a single-pixel anchor for a same-color outside sprite.
    """
    g = np.asarray(grid, dtype=int)
    bg = _get_bg(g)

    frame_color, (fr0, fc0, fr1, fc1) = _find_frame(g, bg)
    crop = g[fr0 : fr1 + 1, fc0 : fc1 + 1]
    placeholders = _extract_placeholders(crop, bg, frame_color)
    placeholder_colors = set(placeholders)

    outside_mask = np.ones(g.shape, dtype=bool)
    outside_mask[fr0 : fr1 + 1, fc0 : fc1 + 1] = False
    color_map, legend_mask = _find_legend(g, outside_mask, bg, frame_color, placeholder_colors)
    sprites = _extract_sprites(g, outside_mask & (~legend_mask), placeholder_colors)

    out = crop.copy()
    out[1:-1, 1:-1] = bg
    inner_h = out.shape[0] - 2
    inner_w = out.shape[1] - 2

    for src, anchors in placeholders.items():
        sprite, sh, sw = sprites[src]
        for anchor in anchors:
            ar, ac = anchor
            top = ar - sh // 2
            left = ac - sw // 2
            bottom = top + sh - 1
            right = left + sw - 1
            assert 1 <= top <= bottom <= inner_h, "stamped sprite must fit inside frame interior"
            assert 1 <= left <= right <= inner_w, "stamped sprite must fit inside frame interior"
            _stamp_centered(out, anchor, sprite, sh, sw, color_map[src])

    return out


def _render_patch(mask: np.ndarray, color: int, bg: int) -> np.ndarray:
    patch = np.full(mask.shape, bg, dtype=int)
    patch[mask] = color
    return patch


def _reserve_box(reserve: np.ndarray, r0: int, c0: int, h: int, w: int, pad: int = 1) -> bool:
    rr0 = max(0, r0 - pad)
    cc0 = max(0, c0 - pad)
    rr1 = min(reserve.shape[0], r0 + h + pad)
    cc1 = min(reserve.shape[1], c0 + w + pad)
    if np.any(reserve[rr0:rr1, cc0:cc1]):
        return False
    reserve[rr0:rr1, cc0:cc1] = True
    return True


def _paste(grid: np.ndarray, patch: np.ndarray, r0: int, c0: int, bg: int) -> None:
    h, w = patch.shape
    sub = grid[r0 : r0 + h, c0 : c0 + w]
    mask = patch != bg
    sub[mask] = patch[mask]


def _choose_bbox(
    reserve: np.ndarray,
    patch_h: int,
    patch_w: int,
    forbidden: np.ndarray | None = None,
    rng: random.Random | None = None,
) -> tuple[int, int] | None:
    if rng is None:
        rng = random
    h, w = reserve.shape
    positions = [
        (r, c)
        for r in range(h - patch_h + 1)
        for c in range(w - patch_w + 1)
        if (forbidden is None or not np.any(forbidden[r : r + patch_h, c : c + patch_w]))
    ]
    rng.shuffle(positions)
    for r, c in positions:
        trial = reserve.copy()
        if _reserve_box(trial, r, c, patch_h, patch_w, pad=1):
            reserve[:] = trial
            return r, c
    return None


def _random_sprite_mask(size: int, rng: random.Random) -> np.ndarray:
    center = size // 2
    mask = np.zeros((size, size), dtype=bool)
    mask[center, center] = True

    anchors = {
        "top": (0, rng.randrange(size)),
        "bottom": (size - 1, rng.randrange(size)),
        "left": (rng.randrange(size), 0),
        "right": (rng.randrange(size), size - 1),
    }
    for tr, tc in anchors.values():
        r, c = center, center
        mask[r, c] = True
        if rng.random() < 0.5:
            while c != tc:
                c += 1 if tc > c else -1
                mask[r, c] = True
            while r != tr:
                r += 1 if tr > r else -1
                mask[r, c] = True
        else:
            while r != tr:
                r += 1 if tr > r else -1
                mask[r, c] = True
            while c != tc:
                c += 1 if tc > c else -1
                mask[r, c] = True

    low = max(int(mask.sum()), size + 2)
    high = size * size - 1
    target = low if low >= high else rng.randint(low, high)
    while int(mask.sum()) < target:
        frontier = []
        for r in range(size):
            for c in range(size):
                if mask[r, c]:
                    continue
                if any(0 <= nr < size and 0 <= nc < size and mask[nr, nc] for nr, nc in _neighbors4(r, c)):
                    frontier.append((r, c))
        if not frontier:
            break
        r, c = rng.choice(frontier)
        mask[r, c] = True
    return mask


def _place_placeholders(
    frame_h: int,
    frame_w: int,
    jobs: list[tuple[int, int, int]],
    rng: random.Random,
) -> dict[int, list[tuple[int, int]]]:
    reserve = np.zeros((frame_h, frame_w), dtype=bool)
    reserve[0, :] = True
    reserve[-1, :] = True
    reserve[:, 0] = True
    reserve[:, -1] = True

    placed: dict[int, list[tuple[int, int]]] = {}
    jobs_sorted = sorted(jobs, key=lambda item: (item[1] * item[2]), reverse=True)
    for color, sh, sw in jobs_sorted:
        ar = sh // 2
        ac = sw // 2
        candidates = [
            (r, c)
            for r in range(1 + ar, frame_h - 1 - (sh - ar - 1))
            for c in range(1 + ac, frame_w - 1 - (sw - ac - 1))
        ]
        rng.shuffle(candidates)
        chosen = None
        for r, c in candidates:
            trial = reserve.copy()
            if _reserve_box(trial, r - ar, c - ac, sh, sw, pad=1):
                reserve[:] = trial
                chosen = (r, c)
                break
        if chosen is None:
            raise ValueError("failed to place placeholder")
        placed.setdefault(color, []).append(chosen)
    return placed


def _build_expected_output(
    frame_h: int,
    frame_w: int,
    bg: int,
    frame_color: int,
    placeholders: dict[int, list[tuple[int, int]]],
    sprites: dict[int, np.ndarray],
    color_map: dict[int, int],
) -> np.ndarray:
    out = np.full((frame_h, frame_w), bg, dtype=int)
    out[0, :] = out[-1, :] = frame_color
    out[:, 0] = out[:, -1] = frame_color
    for src, anchors in placeholders.items():
        mask = sprites[src]
        pts = [(int(r), int(c)) for r, c in np.argwhere(mask)]
        sh, sw = mask.shape
        for anchor in anchors:
            _stamp_centered(out, anchor, pts, sh, sw, color_map[src])
    return out


def _gen_attempt(rng: random.Random) -> np.ndarray | None:
    bg = rng.randrange(10)
    frame_color = rng.choice([c for c in range(10) if c != bg])
    usable = [c for c in range(10) if c not in (bg, frame_color)]

    n_colors = rng.randint(2, 4)
    src_colors = rng.sample(usable, n_colors)
    tgt_pool = [c for c in usable if c not in src_colors]
    tgt_colors = rng.sample(tgt_pool, n_colors)
    row_order = src_colors[:]
    rng.shuffle(row_order)
    color_map = {src: tgt for src, tgt in zip(src_colors, tgt_colors)}

    sprite_masks = {src: _random_sprite_mask(rng.choice([3, 5]), rng) for src in src_colors}

    total_placeholders = n_colors + rng.randint(0, 4)
    counts = {src: 1 for src in src_colors}
    extras = total_placeholders - n_colors
    for _ in range(extras):
        counts[rng.choice(src_colors)] += 1

    jobs = []
    for src in src_colors:
        sh, sw = sprite_masks[src].shape
        jobs.extend([(src, sh, sw)] * counts[src])

    frame_h = rng.randint(max(11, max(mask.shape[0] for mask in sprite_masks.values()) + 6), 20)
    frame_w = rng.randint(max(11, max(mask.shape[1] for mask in sprite_masks.values()) + 6), 22)
    try:
        placeholders = _place_placeholders(frame_h, frame_w, jobs, rng)
    except ValueError:
        return None

    input_frame = np.full((frame_h, frame_w), bg, dtype=int)
    input_frame[0, :] = input_frame[-1, :] = frame_color
    input_frame[:, 0] = input_frame[:, -1] = frame_color
    for src, anchors in placeholders.items():
        for r, c in anchors:
            input_frame[r, c] = src

    expected = _build_expected_output(frame_h, frame_w, bg, frame_color, placeholders, sprite_masks, color_map)

    legend = np.array([[src, color_map[src]] for src in row_order], dtype=int)
    sprite_patches = {src: _render_patch(sprite_masks[src], src, bg) for src in src_colors}

    ext_need = legend.size + sum(patch.size for patch in sprite_patches.values()) + 30
    dims = [
        (h, w)
        for h in range(frame_h + 4, 29)
        for w in range(frame_w + 4, 31)
        if h * w - frame_h * frame_w >= ext_need
    ]
    if not dims:
        return None
    grid_h, grid_w = rng.choice(dims)
    grid = np.full((grid_h, grid_w), bg, dtype=int)
    reserve = np.zeros((grid_h, grid_w), dtype=bool)

    frame_positions = [
        (r, c)
        for r in range(grid_h - frame_h + 1)
        for c in range(grid_w - frame_w + 1)
    ]
    rng.shuffle(frame_positions)
    fr0, fc0 = frame_positions[0]
    if not _reserve_box(reserve, fr0, fc0, frame_h, frame_w, pad=1):
        return None
    _paste(grid, input_frame, fr0, fc0, bg)

    forbidden = np.zeros((grid_h, grid_w), dtype=bool)
    forbidden[fr0 : fr0 + frame_h, fc0 : fc0 + frame_w] = True

    pos = _choose_bbox(reserve, legend.shape[0], legend.shape[1], forbidden, rng)
    if pos is None:
        return None
    lr0, lc0 = pos
    grid[lr0 : lr0 + legend.shape[0], lc0 : lc0 + legend.shape[1]] = legend

    sprite_items = sorted(sprite_patches.items(), key=lambda item: item[1].size, reverse=True)
    for _src, patch in sprite_items:
        pos = _choose_bbox(reserve, patch.shape[0], patch.shape[1], forbidden, rng)
        if pos is None:
            return None
        pr0, pc0 = pos
        _paste(grid, patch, pr0, pc0, bg)

    counts_arr = np.bincount(grid.ravel(), minlength=10)
    if counts_arr[bg] <= max(counts_arr[c] for c in range(10) if c != bg):
        return None

    try:
        actual = main(grid)
    except Exception:
        return None
    if not np.array_equal(actual, expected):
        return None
    return grid


def generate_input() -> np.ndarray:
    rng = random.Random()
    for _ in range(400):
        grid = _gen_attempt(rng)
        if grid is not None:
            return grid
    raise RuntimeError("failed to generate a valid input")
