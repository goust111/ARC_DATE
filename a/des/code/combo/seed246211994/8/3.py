from __future__ import annotations

import random
from collections import deque

import numpy as np


def _most_common_color(grid: np.ndarray) -> int:
    values, counts = np.unique(grid, return_counts=True)
    return int(values[int(np.argmax(counts))])


def _components(mask: np.ndarray, connectivity: int = 4) -> list[list[tuple[int, int]]]:
    h, w = mask.shape
    seen = np.zeros((h, w), dtype=bool)
    if connectivity == 4:
        neighbors = [(-1, 0), (1, 0), (0, -1), (0, 1)]
    else:
        neighbors = [
            (-1, 0),
            (1, 0),
            (0, -1),
            (0, 1),
            (-1, -1),
            (-1, 1),
            (1, -1),
            (1, 1),
        ]

    comps: list[list[tuple[int, int]]] = []
    for r in range(h):
        for c in range(w):
            if not mask[r, c] or seen[r, c]:
                continue
            seen[r, c] = True
            q = deque([(r, c)])
            comp: list[tuple[int, int]] = []
            while q:
                cr, cc = q.popleft()
                comp.append((cr, cc))
                for dr, dc in neighbors:
                    nr, nc = cr + dr, cc + dc
                    if 0 <= nr < h and 0 <= nc < w and mask[nr, nc] and not seen[nr, nc]:
                        seen[nr, nc] = True
                        q.append((nr, nc))
            comps.append(comp)
    return comps


def _legend_bbox(grid: np.ndarray, bg: int) -> tuple[int, int, int, int]:
    candidates: list[tuple[tuple[int, int, int], tuple[int, int, int, int]]] = []
    for comp in _components(grid != bg, connectivity=8):
        rows = [r for r, _c in comp]
        cols = [c for _r, c in comp]
        r0, r1 = min(rows), max(rows)
        c0, c1 = min(cols), max(cols)
        height = r1 - r0 + 1
        width = c1 - c0 + 1
        if width != 2 or height < 2:
            continue
        if len(comp) != height * width:
            continue
        sub = grid[r0 : r1 + 1, c0 : c1 + 1]
        if np.any(sub == bg):
            continue
        colors = {int(v) for v in np.unique(sub)}
        if len(colors) < 3:
            continue
        score = (height, len(colors), -c0)
        candidates.append((score, (r0, c0, r1, c1)))

    assert candidates, "expected at least one filled 2-column legend"
    candidates.sort(reverse=True)
    return candidates[0][1]


def _legend_pairs(grid: np.ndarray, bg: int) -> tuple[list[tuple[int, int]], np.ndarray]:
    r0, c0, r1, c1 = _legend_bbox(grid, bg)
    pairs = [(int(grid[r, c0]), int(grid[r, c1])) for r in range(r0, r1 + 1)]
    legend_mask = np.zeros(grid.shape, dtype=bool)
    legend_mask[r0 : r1 + 1, c0 : c1 + 1] = True
    return pairs, legend_mask


def _apply_chain(color: int, pairs: list[tuple[int, int]]) -> int:
    current = int(color)
    for src, dst in pairs:
        if current == src:
            current = dst
    return current


def _chain_steps(color: int, pairs: list[tuple[int, int]]) -> int:
    current = int(color)
    steps = 0
    for src, dst in pairs:
        if current == src:
            current = dst
            steps += 1
    return steps


def main(grid: np.ndarray) -> np.ndarray:
    """
    Read the vertical two-column legend as an ordered substitution chain.

    Every non-background pixel outside the legend is scanned against the legend
    rows from top to bottom. When its current color matches a row's left color,
    it is replaced by that row's right color and continues through later rows.
    The legend and the background stay unchanged.
    """
    g = np.asarray(grid, dtype=int)
    assert g.ndim == 2, "expected a 2D grid"

    bg = _most_common_color(g)
    pairs, legend_mask = _legend_pairs(g, bg)

    out = g.copy()
    targets = np.argwhere((g != bg) & (~legend_mask))
    for r, c in targets:
        out[r, c] = _apply_chain(int(g[r, c]), pairs)
    return out


# --------------- generator ---------------

GRID_HEIGHT_BOUNDS = (14, 24)
GRID_WIDTH_BOUNDS = (14, 24)
LEGEND_ROWS_CHOICES = (3, 4, 5, 6, 7)
LEGEND_ROWS_WEIGHTS = (1, 2, 3, 2, 1)
AVAILABLE_COLOR_COUNT_BOUNDS = (5, 8)
PAIR_BUILD_RETRIES = 120

TARGET_COUNT_BOUNDS = (5, 13)
TARGET_REQUIRED_TYPES = 3
PLACEMENT_PADDING = 1
PLACEMENT_ATTEMPTS = 120
GENERATION_ATTEMPTS = 400

SPRITE_SHAPES = ("pixel", "segment", "rectangle", "blob")
SPRITE_HEIGHT_BOUNDS = (1, 4)
SPRITE_WIDTH_BOUNDS = (1, 4)
SPRITE_FILL_RANGE = (0.45, 0.85)


def _build_pairs(available: list[int], n_rows: int) -> tuple[list[tuple[int, int]], int, int, int] | None:
    for _ in range(PAIR_BUILD_RETRIES):
        chain_len = 3 if n_rows >= 4 and random.random() < 0.5 else 2
        chain_colors = random.sample(available, chain_len + 1)
        chain_pairs = [(chain_colors[i], chain_colors[i + 1]) for i in range(chain_len)]

        rows: list[tuple[int, int] | None] = [None] * n_rows
        chain_slots = sorted(random.sample(range(n_rows), chain_len))
        for slot, pair in zip(chain_slots, chain_pairs, strict=False):
            rows[slot] = pair

        used_pairs = set(chain_pairs)
        ok = True
        for i in range(n_rows):
            if rows[i] is not None:
                continue
            for _retry in range(50):
                src = random.choice(available)
                dst = random.choice([c for c in available if c != src])
                if (src, dst) in used_pairs:
                    continue
                rows[i] = (src, dst)
                used_pairs.add((src, dst))
                break
            if rows[i] is None:
                ok = False
                break
        if not ok:
            continue

        pairs = [pair for pair in rows if pair is not None]
        src_colors = {src for src, _dst in pairs}
        if len({color for pair in pairs for color in pair}) < 3:
            continue
        if any(_apply_chain(src, pairs) == src for src in src_colors):
            continue

        multi_sources = [color for color in available if _chain_steps(color, pairs) >= 2]
        one_step_sources = [color for color in available if _chain_steps(color, pairs) == 1]
        unchanged_colors = [color for color in available if _chain_steps(color, pairs) == 0]
        if not multi_sources or not one_step_sources or not unchanged_colors:
            continue

        return pairs, random.choice(multi_sources), random.choice(one_step_sources), random.choice(unchanged_colors)
    return None


def _grow_blob_mask(height: int, width: int, area: int) -> np.ndarray:
    mask = np.zeros((height, width), dtype=bool)
    seed = (random.randrange(height), random.randrange(width))
    mask[seed] = True
    frontier = [seed]

    while int(mask.sum()) < area:
        base_r, base_c = random.choice(frontier)
        candidates = []
        for dr, dc in [(-1, 0), (1, 0), (0, -1), (0, 1)]:
            nr, nc = base_r + dr, base_c + dc
            if 0 <= nr < height and 0 <= nc < width and not mask[nr, nc]:
                candidates.append((nr, nc))
        if not candidates:
            frontier = [(r, c) for r in range(height) for c in range(width) if mask[r, c]]
            continue
        chosen = random.choice(candidates)
        mask[chosen] = True
        frontier.append(chosen)

    return mask


def _make_sprite(color: int, bg: int) -> np.ndarray:
    shape_kind = random.choice(SPRITE_SHAPES)

    if shape_kind == "pixel":
        return np.array([[color]], dtype=int)

    if shape_kind == "segment":
        if random.random() < 0.5:
            length = random.randint(2, 6)
            sprite = np.full((1, length), bg, dtype=int)
            sprite[0, :] = color
        else:
            length = random.randint(2, 6)
            sprite = np.full((length, 1), bg, dtype=int)
            sprite[:, 0] = color
        return sprite

    if shape_kind == "rectangle":
        height = random.randint(*SPRITE_HEIGHT_BOUNDS)
        width = random.randint(*SPRITE_WIDTH_BOUNDS)
        sprite = np.full((height, width), bg, dtype=int)
        sprite[:, :] = color
        return sprite

    height = random.randint(*SPRITE_HEIGHT_BOUNDS)
    width = random.randint(*SPRITE_WIDTH_BOUNDS)
    max_area = height * width
    min_area = max(1, int(np.ceil(SPRITE_FILL_RANGE[0] * max_area)))
    max_area_used = max(min_area, int(np.floor(SPRITE_FILL_RANGE[1] * max_area)))
    area = random.randint(min_area, max_area_used)
    mask = _grow_blob_mask(height, width, area)
    sprite = np.full((height, width), bg, dtype=int)
    sprite[mask] = color
    return sprite


def _can_place(grid: np.ndarray, sprite: np.ndarray, r0: int, c0: int, bg: int) -> bool:
    h, w = grid.shape
    sh, sw = sprite.shape
    if r0 < 0 or c0 < 0 or r0 + sh > h or c0 + sw > w:
        return False

    rr0 = max(0, r0 - PLACEMENT_PADDING)
    rr1 = min(h, r0 + sh + PLACEMENT_PADDING)
    cc0 = max(0, c0 - PLACEMENT_PADDING)
    cc1 = min(w, c0 + sw + PLACEMENT_PADDING)
    if np.any(grid[rr0:rr1, cc0:cc1] != bg):
        return False
    return True


def _paste_sprite(grid: np.ndarray, sprite: np.ndarray, r0: int, c0: int, bg: int) -> None:
    rows, cols = np.where(sprite != bg)
    grid[r0 + rows, c0 + cols] = sprite[rows, cols]


def _place_sprite(grid: np.ndarray, sprite: np.ndarray, bg: int) -> bool:
    h, w = grid.shape
    sh, sw = sprite.shape
    options = [(r0, c0) for r0 in range(h - sh + 1) for c0 in range(w - sw + 1)]
    random.shuffle(options)
    for r0, c0 in options[:PLACEMENT_ATTEMPTS]:
        if _can_place(grid, sprite, r0, c0, bg):
            _paste_sprite(grid, sprite, r0, c0, bg)
            return True
    return False


def _expected_output(grid: np.ndarray, bg: int, pairs: list[tuple[int, int]], legend_bbox: tuple[int, int, int, int]) -> np.ndarray:
    r0, c0, r1, c1 = legend_bbox
    out = grid.copy()
    for r in range(grid.shape[0]):
        for c in range(grid.shape[1]):
            if r0 <= r <= r1 and c0 <= c <= c1:
                continue
            if int(grid[r, c]) == bg:
                continue
            out[r, c] = _apply_chain(int(grid[r, c]), pairs)
    return out


def _gen_attempt() -> np.ndarray | None:
    height = random.randint(*GRID_HEIGHT_BOUNDS)
    width = random.randint(*GRID_WIDTH_BOUNDS)
    legend_rows = random.choices(LEGEND_ROWS_CHOICES, weights=LEGEND_ROWS_WEIGHTS, k=1)[0]
    if legend_rows + 2 > height or 4 > width:
        return None

    bg = random.randrange(10)
    palette = [color for color in range(10) if color != bg]
    available_count = random.randint(*AVAILABLE_COLOR_COUNT_BOUNDS)
    available = random.sample(palette, k=available_count)

    built = _build_pairs(available, legend_rows)
    if built is None:
        return None
    pairs, multi_color, one_step_color, unchanged_color = built

    grid = np.full((height, width), bg, dtype=int)
    legend_r0 = random.randint(1, height - legend_rows - 1)
    legend_c0 = random.randint(1, width - 3)
    legend_bbox = (legend_r0, legend_c0, legend_r0 + legend_rows - 1, legend_c0 + 1)
    for offset, (src, dst) in enumerate(pairs):
        grid[legend_r0 + offset, legend_c0] = src
        grid[legend_r0 + offset, legend_c0 + 1] = dst

    target_count = random.randint(*TARGET_COUNT_BOUNDS)
    object_colors = [multi_color, one_step_color, unchanged_color]
    extra_pool = available + [color for color in range(10) if color != bg]
    while len(object_colors) < target_count:
        object_colors.append(random.choice(extra_pool))
    random.shuffle(object_colors)

    changed_by_steps = {0: 0, 1: 0, 2: 0}
    for color in object_colors:
        sprite = _make_sprite(color, bg)
        if not _place_sprite(grid, sprite, bg):
            return None
        steps = _chain_steps(color, pairs)
        if steps == 0:
            changed_by_steps[0] += 1
        elif steps == 1:
            changed_by_steps[1] += 1
        else:
            changed_by_steps[2] += 1

    if any(count == 0 for count in changed_by_steps.values()):
        return None
    if np.count_nonzero(grid != bg) * 2 >= grid.size:
        return None

    expected = _expected_output(grid, bg, pairs, legend_bbox)
    if np.array_equal(grid, expected):
        return None

    try:
        actual = main(grid)
    except Exception:
        return None
    if not np.array_equal(actual, expected):
        return None

    detected_bbox = _legend_bbox(grid, bg)
    if detected_bbox != legend_bbox:
        return None

    return grid


def generate_input() -> np.ndarray:
    for _ in range(GENERATION_ATTEMPTS):
        grid = _gen_attempt()
        if grid is not None:
            return grid
    raise RuntimeError("failed to generate valid input")
