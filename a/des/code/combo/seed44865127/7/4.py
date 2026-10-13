from __future__ import annotations

import random
from collections import deque

import numpy as np


def _dominant_color(arr: np.ndarray) -> int:
    vals, counts = np.unique(arr, return_counts=True)
    max_count = int(counts.max())
    candidates = vals[counts == max_count]
    assert len(candidates) == 1, "background must be structurally unique"
    return int(candidates[0])


def _separator_band(grid: np.ndarray) -> tuple[int, int, int]:
    h, w = grid.shape
    candidates: list[tuple[int, int, int]] = []
    c = 0
    while c < w:
        if not np.all(grid[:, c] == grid[0, c]):
            c += 1
            continue
        color = int(grid[0, c])
        start = c
        c += 1
        while c < w and np.all(grid[:, c] == color):
            c += 1
        end = c - 1
        left_w = start
        right_w = w - end - 1
        if 0 < left_w == right_w:
            candidates.append((start, end, color))
    assert len(candidates) == 1, f"expected one equal-width separator band, got {candidates}"
    return candidates[0]


def _components_by_color(panel: np.ndarray, bg: int) -> list[dict[str, np.ndarray | int]]:
    h, w = panel.shape
    seen = np.zeros((h, w), dtype=bool)
    comps: list[dict[str, np.ndarray | int]] = []

    for r in range(h):
        for c in range(w):
            if seen[r, c] or int(panel[r, c]) == int(bg):
                continue
            color = int(panel[r, c])
            q = deque([(r, c)])
            seen[r, c] = True
            cells: list[tuple[int, int]] = []
            while q:
                rr, cc = q.popleft()
                cells.append((rr, cc))
                for dr, dc in ((1, 0), (-1, 0), (0, 1), (0, -1)):
                    nr, nc = rr + dr, cc + dc
                    if 0 <= nr < h and 0 <= nc < w and not seen[nr, nc] and int(panel[nr, nc]) == color:
                        seen[nr, nc] = True
                        q.append((nr, nc))
            coords = np.asarray(cells, dtype=int)
            comps.append({"color": color, "coords": coords, "area": int(coords.shape[0])})

    assert comps, "left panel must contain foreground objects"
    return comps


def _largest_component(panel: np.ndarray, bg: int) -> dict[str, np.ndarray | int]:
    comps = _components_by_color(panel, bg)
    largest_area = max(int(comp["area"]) for comp in comps)
    candidates = [comp for comp in comps if int(comp["area"]) == largest_area]
    assert len(candidates) == 1, "largest left-panel object must be unique"
    return candidates[0]


def _component_mask(coords: np.ndarray) -> np.ndarray:
    r0, c0 = coords.min(axis=0)
    r1, c1 = coords.max(axis=0)
    mask = np.zeros((int(r1 - r0 + 1), int(c1 - c0 + 1)), dtype=bool)
    rel = coords - np.array([r0, c0], dtype=int)
    mask[rel[:, 0], rel[:, 1]] = True
    return mask


def _expand_mask(mask: np.ndarray, shape: tuple[int, int]) -> np.ndarray:
    mh, mw = mask.shape
    h, w = map(int, shape)
    assert h % mh == 0 and w % mw == 0, "mask bbox must tile the right panel"
    return np.tile(mask, (h // mh, w // mw))


def main(grid: np.ndarray) -> np.ndarray:
    """
    Use the largest left-panel object as a crop mask for the right panel.

    The grid is split by a vertical separator into equal-width left/right panels.
    The unique largest foreground object in the left panel defines a boolean mask.
    Its bbox is aligned to the right panel's top-left corner; if that bbox is
    smaller than the right panel, the mask tiles the panel exactly.
    """
    g = np.asarray(grid, dtype=int)
    assert g.ndim == 2, "grid must be 2D"

    sep0, sep1, _sep_color = _separator_band(g)
    left = g[:, :sep0]
    right = g[:, sep1 + 1 :]
    assert left.shape == right.shape and left.size > 0

    bg = _dominant_color(left)
    largest = _largest_component(left, bg)
    mask = _component_mask(np.asarray(largest["coords"], dtype=int))
    full_mask = _expand_mask(mask, right.shape)

    out = np.full(right.shape, bg, dtype=int)
    out[full_mask] = right[full_mask]
    return out


# --------------- generator (o5x) ---------------

COLORS = tuple(range(10))
PANEL_SIZE_OPTIONS = (8, 10, 12, 14, 15)
SEPARATOR_WIDTH_OPTIONS = (1, 2)
OBJECT_COUNT_RANGE = (2, 4)
PATTERN_COLOR_COUNT_RANGE = (3, 4)
GENERATION_ATTEMPTS = 400
PLACEMENT_ATTEMPTS = 80


def _draw_frame(mask: np.ndarray, r0: int, c0: int, r1: int, c1: int) -> None:
    mask[r0, c0 : c1 + 1] = True
    mask[r1, c0 : c1 + 1] = True
    mask[r0 : r1 + 1, c0] = True
    mask[r0 : r1 + 1, c1] = True


def _closed_mask(height: int, width: int) -> np.ndarray:
    height = int(height)
    width = int(width)
    mask = np.zeros((height, width), dtype=bool)
    _draw_frame(mask, 0, 0, height - 1, width - 1)
    base = mask.copy()

    styles = ["plain", "plain", "bar"]
    if height >= 7 and width >= 7:
        styles.extend(["inner", "inner"])
    style = random.choice(styles)

    if style == "bar" and height >= 5 and width >= 5:
        if random.random() < 0.5:
            c = random.randint(1, width - 2)
            mask[1 : height - 1, c] = True
        else:
            r = random.randint(1, height - 2)
            mask[r, 1 : width - 1] = True
    elif style == "inner":
        margins = [
            m
            for m in range(2, min(height, width) // 2 + 1)
            if height - 2 * m >= 3 and width - 2 * m >= 3
        ]
        if margins:
            margin = random.choice(margins)
            _draw_frame(mask, margin, margin, height - margin - 1, width - margin - 1)
            side = random.choice(("top", "bottom", "left", "right"))
            if side == "top":
                c = random.choice([margin, width - margin - 1])
                mask[: margin + 1, c] = True
            elif side == "bottom":
                c = random.choice([margin, width - margin - 1])
                mask[height - margin - 1 :, c] = True
            elif side == "left":
                r = random.choice([margin, height - margin - 1])
                mask[r, : margin + 1] = True
            else:
                r = random.choice([margin, height - margin - 1])
                mask[r, width - margin - 1 :] = True

    if mask.mean() > 0.45:
        mask = base
    assert mask[0].any() and mask[-1].any() and mask[:, 0].any() and mask[:, -1].any()
    assert mask.any() and (~mask).any()
    return mask


def _divisors(n: int, *, proper_only: bool = False, min_size: int = 4) -> list[int]:
    vals = [d for d in range(min_size, int(n) + 1) if int(n) % d == 0]
    if proper_only:
        vals = [d for d in vals if d < int(n)]
    return vals


def _candidate_bbox_pairs(panel_h: int, panel_w: int, *, allow_full: bool) -> list[tuple[int, int]]:
    hs = _divisors(panel_h, proper_only=not allow_full)
    ws = _divisors(panel_w, proper_only=not allow_full)
    pairs: list[tuple[int, int]] = []
    for h in hs:
        for w in ws:
            if allow_full:
                if h == panel_h and w == panel_w:
                    continue
                if h * w > 0.75 * panel_h * panel_w:
                    continue
            pairs.append((int(h), int(w)))
    return pairs


def _can_place(panel: np.ndarray, mask: np.ndarray, top: int, left: int, bg: int) -> bool:
    ph, pw = panel.shape
    h, w = mask.shape
    if top < 0 or left < 0 or top + h > ph or left + w > pw:
        return False
    coords = np.argwhere(mask)
    for rr, cc in coords:
        r = int(top + rr)
        c = int(left + cc)
        r0, r1 = max(0, r - 1), min(ph, r + 2)
        c0, c1 = max(0, c - 1), min(pw, c + 2)
        if np.any(panel[r0:r1, c0:c1] != int(bg)):
            return False
    return True


def _place_object(
    panel: np.ndarray,
    mask: np.ndarray,
    color: int,
    bg: int,
    *,
    fixed: tuple[int, int] | None = None,
) -> tuple[int, int] | None:
    h, w = mask.shape
    ph, pw = panel.shape
    if fixed is not None:
        positions = [fixed]
    else:
        positions = [(r, c) for r in range(ph - h + 1) for c in range(pw - w + 1)]
        random.shuffle(positions)
    for top, left in positions[:PLACEMENT_ATTEMPTS] if fixed is None else positions:
        if _can_place(panel, mask, int(top), int(left), bg):
            view = panel[int(top) : int(top) + h, int(left) : int(left) + w]
            view[mask] = int(color)
            return int(top), int(left)
    return None


def _pattern_panel(height: int, width: int, bg: int, palette: list[int]) -> np.ndarray:
    rr, cc = np.indices((height, width))
    colors = np.asarray(palette, dtype=int)
    mod = len(palette)
    a = random.randint(1, 3)
    b = random.randint(2, 4)
    block_h = random.randint(2, max(2, height // 3))
    block_w = random.randint(2, max(2, width // 3))
    idx = (a * rr + b * cc + rr // block_h + 2 * (cc // block_w) + random.randrange(mod)) % mod
    panel = colors[idx]

    if height >= 4 and width >= 4:
        patch_h = random.randint(2, max(2, height // 2))
        patch_w = random.randint(2, max(2, width // 2))
        r0 = random.randint(0, height - patch_h)
        c0 = random.randint(0, width - patch_w)
        panel[r0 : r0 + patch_h, c0 : c0 + patch_w] = random.choice(palette)

    hole_mod = random.choice((5, 6, 7))
    holes = (rr + random.randint(1, 3) * cc + random.randrange(hole_mod)) % hole_mod == 0
    panel[holes] = int(bg)
    return panel.astype(int)


def _full_size_branch(panel_h: int, panel_w: int, bg: int, colors: list[int]) -> tuple[np.ndarray, np.ndarray] | None:
    left = np.full((panel_h, panel_w), bg, dtype=int)
    largest_mask = _closed_mask(panel_h, panel_w)
    _place_object(left, largest_mask, colors[0], bg, fixed=(0, 0))
    largest_area = int(largest_mask.sum())

    pairs = [
        (h, w)
        for h in _divisors(panel_h, proper_only=True)
        for w in _divisors(panel_w, proper_only=True)
        if h <= panel_h - 4 and w <= panel_w - 4
    ]
    if not pairs:
        return None

    distractor_target = random.randint(1, min(3, len(colors) - 1))
    placed = 0
    for color in colors[1:]:
        pair_order = pairs[:]
        random.shuffle(pair_order)
        success = False
        for h, w in pair_order:
            mask = _closed_mask(h, w)
            if int(mask.sum()) >= largest_area:
                continue
            if _place_object(left, mask, color, bg) is not None:
                placed += 1
                success = True
                break
        if not success and placed < distractor_target:
            return None
        if placed >= distractor_target:
            break

    if placed < distractor_target:
        return None
    return left, largest_mask


def _tiled_branch(panel_h: int, panel_w: int, bg: int, colors: list[int]) -> tuple[np.ndarray, np.ndarray] | None:
    left = np.full((panel_h, panel_w), bg, dtype=int)
    pairs = _candidate_bbox_pairs(panel_h, panel_w, allow_full=True)
    random.shuffle(pairs)

    largest_mask = None
    largest_area = -1
    for h, w in pairs:
        mask = _closed_mask(h, w)
        area = int(mask.sum())
        if area < 10:
            continue
        if _place_object(left, mask, colors[0], bg) is not None:
            largest_mask = mask
            largest_area = area
            break
    if largest_mask is None:
        return None

    smaller_pairs = [
        (h, w)
        for h in _divisors(panel_h, proper_only=False)
        for w in _divisors(panel_w, proper_only=False)
        if h * w < largest_mask.shape[0] * largest_mask.shape[1]
    ]
    if not smaller_pairs:
        return None

    distractor_target = random.randint(1, min(3, len(colors) - 1))
    placed = 0
    for color in colors[1:]:
        pair_order = smaller_pairs[:]
        random.shuffle(pair_order)
        success = False
        for h, w in pair_order:
            mask = _closed_mask(h, w)
            if int(mask.sum()) >= largest_area:
                continue
            if _place_object(left, mask, color, bg) is not None:
                placed += 1
                success = True
                break
        if not success and placed < distractor_target:
            return None
        if placed >= distractor_target:
            break

    if placed < distractor_target:
        return None
    return left, largest_mask


def _gen_attempt() -> np.ndarray | None:
    panel_h = random.choice(PANEL_SIZE_OPTIONS)
    panel_w = random.choice(PANEL_SIZE_OPTIONS)
    sep_w = random.choice(SEPARATOR_WIDTH_OPTIONS)
    n_objects = random.randint(*OBJECT_COUNT_RANGE)

    bg, sep = random.sample(COLORS, 2)
    remaining = [c for c in COLORS if c not in (bg, sep)]
    object_colors = random.sample(remaining, n_objects)
    pattern_pool = [c for c in remaining if c not in object_colors]
    n_pattern = random.randint(*PATTERN_COLOR_COUNT_RANGE)
    pattern_colors = random.sample(pattern_pool, n_pattern)

    branch = random.choice(("full", "tile"))
    if branch == "full":
        left_pack = _full_size_branch(panel_h, panel_w, bg, object_colors)
    else:
        left_pack = _tiled_branch(panel_h, panel_w, bg, object_colors)
    if left_pack is None:
        return None

    left, largest_mask = left_pack
    right = _pattern_panel(panel_h, panel_w, bg, pattern_colors)
    sep_block = np.full((panel_h, sep_w), sep, dtype=int)
    grid = np.hstack([left, sep_block, right])

    try:
        tiled = _expand_mask(largest_mask, right.shape)
        expected = np.full(right.shape, bg, dtype=int)
        expected[tiled] = right[tiled]
        solved = main(grid)
    except AssertionError:
        return None

    if not np.array_equal(solved, expected):
        return None
    if len(np.unique(right)) < 3:
        return None
    return grid


def generate_input() -> np.ndarray:
    for _ in range(GENERATION_ATTEMPTS):
        grid = _gen_attempt()
        if grid is not None:
            return grid
    raise RuntimeError("failed to generate a valid input")
