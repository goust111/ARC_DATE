from __future__ import annotations

from collections import deque

import numpy as np


BG = 0
BORDER = 5
PALETTE_COLORS = [1, 2, 3, 4, 6, 7, 8, 9]
GEN_ATTEMPTS = 600


def _neighbors4(r: int, c: int) -> list[tuple[int, int]]:
    return [(r + 1, c), (r - 1, c), (r, c + 1), (r, c - 1)]


def _components(mask: np.ndarray) -> list[list[tuple[int, int]]]:
    h, w = mask.shape
    seen = np.zeros((h, w), dtype=bool)
    comps: list[list[tuple[int, int]]] = []
    for r in range(h):
        for c in range(w):
            if not mask[r, c] or seen[r, c]:
                continue
            seen[r, c] = True
            q = deque([(r, c)])
            comp: list[tuple[int, int]] = []
            while q:
                rr, cc = q.popleft()
                comp.append((int(rr), int(cc)))
                for nr, nc in _neighbors4(int(rr), int(cc)):
                    if 0 <= nr < h and 0 <= nc < w and mask[nr, nc] and not seen[nr, nc]:
                        seen[nr, nc] = True
                        q.append((int(nr), int(nc)))
            comps.append(comp)
    return comps


def _bbox(coords: list[tuple[int, int]]) -> tuple[int, int, int, int]:
    rows = [r for r, _c in coords]
    cols = [c for _r, c in coords]
    return min(rows), min(cols), max(rows), max(cols)


def _enclosed_cells(border_mask: np.ndarray) -> np.ndarray:
    h, w = border_mask.shape
    outside = np.zeros((h, w), dtype=bool)
    q: deque[tuple[int, int]] = deque()

    for r in range(h):
        for c in (0, w - 1):
            if not border_mask[r, c] and not outside[r, c]:
                outside[r, c] = True
                q.append((int(r), int(c)))
    for c in range(w):
        for r in (0, h - 1):
            if not border_mask[r, c] and not outside[r, c]:
                outside[r, c] = True
                q.append((int(r), int(c)))

    while q:
        r, c = q.popleft()
        for nr, nc in _neighbors4(r, c):
            if 0 <= nr < h and 0 <= nc < w and not border_mask[nr, nc] and not outside[nr, nc]:
                outside[nr, nc] = True
                q.append((int(nr), int(nc)))

    return (~border_mask) & (~outside)


def _is_palette_patch(patch: np.ndarray) -> bool:
    if patch.shape != (3, 3):
        return False
    color = int(patch[0, 0])
    if color in (BG, BORDER):
        return False
    if not np.all((patch == BG) | (patch == color)):
        return False
    colored = int(np.sum(patch == color))
    return 3 <= colored <= 8


def _collect_right_palettes(grid: np.ndarray) -> dict[int, np.ndarray]:
    h, w = grid.shape
    if w < 3:
        return {}
    palettes: dict[int, np.ndarray] = {}
    r = 0
    while r <= h - 3:
        patch = grid[r : r + 3, w - 3 : w]
        if _is_palette_patch(patch):
            color = int(patch[0, 0])
            if color in palettes:
                return {}
            palettes[color] = patch.copy()
            r += 3
            while r < h and np.all(grid[r, w - 3 : w] == BG):
                r += 1
        else:
            r += 1
    return palettes


def _collect_bottom_palettes(grid: np.ndarray) -> dict[int, np.ndarray]:
    h, w = grid.shape
    if h < 3:
        return {}
    palettes: dict[int, np.ndarray] = {}
    c = 0
    while c <= w - 3:
        patch = grid[h - 3 : h, c : c + 3]
        if _is_palette_patch(patch):
            color = int(patch[0, 0])
            if color in palettes:
                return {}
            palettes[color] = patch.copy()
            c += 3
            while c < w and np.all(grid[h - 3 : h, c] == BG):
                c += 1
        else:
            c += 1
    return palettes


def _find_palettes(grid: np.ndarray) -> dict[int, np.ndarray]:
    right = _collect_right_palettes(grid)
    bottom = _collect_bottom_palettes(grid)
    if len(right) > len(bottom):
        palettes = right
    elif len(bottom) > len(right):
        palettes = bottom
    else:
        palettes = right if right else bottom
    assert palettes, "expected palette blocks on the right or bottom edge"
    return palettes


def _frame_regions(grid: np.ndarray) -> list[tuple[np.ndarray, tuple[int, int, int, int], int]]:
    frames: list[tuple[np.ndarray, tuple[int, int, int, int], int]] = []
    for comp in _components(grid == BORDER):
        r0, c0, r1, c1 = _bbox(comp)
        if r1 - r0 + 1 < 3 or c1 - c0 + 1 < 3:
            continue
        local = grid[r0 : r1 + 1, c0 : c1 + 1]
        border_mask = local == BORDER
        enclosed = _enclosed_cells(border_mask)
        if not enclosed.any():
            continue
        enclosed_comps = _components(enclosed)
        if len(enclosed_comps) != 1:
            continue
        seed_colors = set(map(int, np.unique(local[enclosed]))) - {BG}
        assert len(seed_colors) == 1, "each frame interior must contain exactly one seed color"
        seed_color = seed_colors.pop()
        frames.append((enclosed, (r0, c0, r1, c1), int(seed_color)))
    assert frames, "expected at least one grey closed frame"
    return frames


def main(grid: np.ndarray) -> np.ndarray:
    """
    Fill each grey-enclosed region by tiling the 3x3 palette pattern keyed by its seed color.
    """
    g = np.asarray(grid, dtype=int)
    palettes = _find_palettes(g)
    out = g.copy()

    for enclosed, (r0, c0, _r1, _c1), seed_color in _frame_regions(g):
        assert seed_color in palettes, "seed color must exist in the palette"
        pattern = palettes[seed_color]
        coords = np.argwhere(enclosed)
        rmin = int(coords[:, 0].min())
        cmin = int(coords[:, 1].min())
        for rr, cc in coords:
            out[r0 + rr, c0 + cc] = int(pattern[(int(rr) - rmin) % 3, (int(cc) - cmin) % 3])
    return out


def _erode8(mask: np.ndarray) -> np.ndarray:
    h, w = mask.shape
    inner = np.zeros((h, w), dtype=bool)
    for r in range(1, h - 1):
        for c in range(1, w - 1):
            if mask[r - 1 : r + 2, c - 1 : c + 2].all():
                inner[r, c] = True
    return inner


def _pattern_library() -> list[np.ndarray]:
    patterns: list[np.ndarray] = []
    rest_cells = [(r, c) for r in range(3) for c in range(3) if (r, c) != (0, 0)]
    for bits in range(1 << len(rest_cells)):
        patch = np.zeros((3, 3), dtype=int)
        patch[0, 0] = 1
        colored = 1
        for idx, (r, c) in enumerate(rest_cells):
            if (bits >> idx) & 1:
                patch[r, c] = 1
                colored += 1
        if 3 <= colored <= 8 and np.any(patch == 0):
            patterns.append(patch)
    return patterns


PATTERN_LIBRARY = _pattern_library()


def _random_outer_mask(rng: np.random.Generator) -> np.ndarray | None:
    if rng.random() < 0.35:
        h = int(rng.integers(5, 10))
        w = int(rng.integers(5, 10))
        return np.ones((h, w), dtype=bool)

    h = int(rng.integers(6, 11))
    base_w = int(rng.integers(5, 9))
    intervals: list[tuple[int, int]] = [(0, base_w - 1)]

    for _ in range(1, h):
        prev_l, prev_r = intervals[-1]
        left = prev_l + int(rng.integers(-1, 2))
        right = prev_r + int(rng.integers(-1, 2))
        if right - left + 1 < 5:
            right = left + 4
        if right - left + 1 > 10:
            right = left + 9
        if right < prev_l:
            shift = prev_l - right
            left += shift
            right += shift
        if left > prev_r:
            shift = left - prev_r
            left -= shift
            right -= shift
        intervals.append((int(left), int(right)))

    min_left = min(left for left, _right in intervals)
    intervals = [(left - min_left, right - min_left) for left, right in intervals]
    width = max(right for _left, right in intervals) + 1
    mask = np.zeros((h, width), dtype=bool)
    for r, (left, right) in enumerate(intervals):
        mask[r, left : right + 1] = True
    return mask


def _random_frame_patch(rng: np.random.Generator, seed_color: int) -> tuple[np.ndarray, np.ndarray] | None:
    for _ in range(80):
        outer = _random_outer_mask(rng)
        if outer is None:
            continue
        inner = _erode8(outer)
        if int(inner.sum()) < 4:
            continue
        if len(_components(inner)) != 1:
            continue
        border = outer & (~inner)
        if len(_components(border)) != 1:
            continue
        if not np.array_equal(_enclosed_cells(border), inner):
            continue

        patch = np.full(border.shape, BG, dtype=int)
        patch[border] = BORDER
        inner_cells = np.argwhere(inner)
        seed_r, seed_c = inner_cells[int(rng.integers(len(inner_cells)))]
        patch[int(seed_r), int(seed_c)] = int(seed_color)
        return patch, inner
    return None


def _choose_seed_colors(rng: np.random.Generator, palette_colors: list[int], region_count: int) -> list[int]:
    while True:
        colors = [int(palette_colors[int(i)]) for i in rng.integers(0, len(palette_colors), size=region_count)]
        if len(set(colors)) >= min(2, region_count):
            return colors


def _place_patch(
    grid: np.ndarray,
    occupied: np.ndarray,
    patch: np.ndarray,
    r0: int,
    c0: int,
) -> bool:
    h, w = grid.shape
    ph, pw = patch.shape
    r1 = r0 + ph
    c1 = c0 + pw
    if r0 < 0 or c0 < 0 or r1 > h or c1 > w:
        return False

    rr0 = max(0, r0 - 1)
    cc0 = max(0, c0 - 1)
    rr1 = min(h, r1 + 1)
    cc1 = min(w, c1 + 1)
    if occupied[rr0:rr1, cc0:cc1].any():
        return False

    mask = patch != BG
    target = grid[r0:r1, c0:c1]
    if np.any(target[mask] != BG):
        return False

    target[mask] = patch[mask]
    occupied[r0:r1, c0:c1] = True
    return True


def _build_palette_patch(mask: np.ndarray, color: int) -> np.ndarray:
    patch = np.full((3, 3), BG, dtype=int)
    patch[mask.astype(bool)] = int(color)
    patch[0, 0] = int(color)
    return patch


def _paste_palette_blocks(
    grid: np.ndarray,
    palettes: dict[int, np.ndarray],
    orient_right: bool,
    rng: np.random.Generator,
) -> bool:
    count = len(palettes)
    span = 3 * count + (count - 1)
    if orient_right:
        h, w = grid.shape
        if span > h - 3:
            return False
        start_r = int(rng.integers(0, h - 3 - span + 1))
        c0 = w - 3
        for idx, color in enumerate(palettes):
            r0 = start_r + idx * 4
            grid[r0 : r0 + 3, c0 : c0 + 3] = palettes[color]
    else:
        h, w = grid.shape
        if span > w - 3:
            return False
        start_c = int(rng.integers(0, w - 3 - span + 1))
        r0 = h - 3
        for idx, color in enumerate(palettes):
            c0 = start_c + idx * 4
            grid[r0 : r0 + 3, c0 : c0 + 3] = palettes[color]
    return True


def _gen_attempt(rng: np.random.Generator) -> np.ndarray | None:
    orient_right = bool(rng.integers(0, 2))
    palette_count = int(rng.integers(2, 6))
    region_count = int(rng.integers(2, 5))
    if palette_count > len(PALETTE_COLORS):
        return None

    chosen_colors = [int(c) for c in rng.choice(PALETTE_COLORS, size=palette_count, replace=False)]
    seed_colors = _choose_seed_colors(rng, chosen_colors, region_count)

    pattern_indices = rng.choice(len(PATTERN_LIBRARY), size=palette_count, replace=False)
    palette_patterns = {
        int(color): _build_palette_patch(PATTERN_LIBRARY[int(idx)], int(color))
        for color, idx in zip(chosen_colors, pattern_indices, strict=False)
    }

    frame_patches: list[np.ndarray] = []
    for seed_color in seed_colors:
        item = _random_frame_patch(rng, int(seed_color))
        if item is None:
            return None
        patch, _inner = item
        frame_patches.append(patch)

    palette_span = 3 * palette_count + (palette_count - 1)
    body_h = int(rng.integers(max(16, palette_span - 1), 27))
    body_w = int(rng.integers(max(16, palette_span - 1), 27))

    if orient_right:
        height = max(body_h, palette_span + 3)
        width = body_w + 4
        place_h, place_w = height - 3, width - 4
    else:
        height = body_h + 4
        width = max(body_w, palette_span + 3)
        place_h, place_w = height - 4, width - 3

    grid = np.full((height, width), BG, dtype=int)
    occupied = np.zeros((height, width), dtype=bool)

    order = list(range(region_count))
    rng.shuffle(order)
    for idx in order:
        patch = frame_patches[idx]
        ph, pw = patch.shape
        if ph > place_h or pw > place_w:
            return None
        placed = False
        for _ in range(120):
            r0 = int(rng.integers(0, place_h - ph + 1))
            c0 = int(rng.integers(0, place_w - pw + 1))
            if _place_patch(grid, occupied, patch, r0, c0):
                placed = True
                break
        if not placed:
            return None

    if not _paste_palette_blocks(grid, palette_patterns, orient_right, rng):
        return None

    try:
        out = main(grid)
    except Exception:
        return None

    if np.array_equal(out, grid):
        return None

    # Every seed color must be resolvable and every frame must be changed.
    for enclosed, (r0, c0, _r1, _c1), seed_color in _frame_regions(grid):
        if seed_color not in palette_patterns:
            return None
        local_in = grid[r0 : r0 + enclosed.shape[0], c0 : c0 + enclosed.shape[1]]
        local_out = out[r0 : r0 + enclosed.shape[0], c0 : c0 + enclosed.shape[1]]
        if np.array_equal(local_in[enclosed], local_out[enclosed]):
            return None

    return grid


def generate_input() -> np.ndarray:
    rng = np.random.default_rng()
    for _ in range(GEN_ATTEMPTS):
        grid = _gen_attempt(rng)
        if grid is not None:
            return grid
    raise RuntimeError("failed to generate a valid input for puzzle 1")
