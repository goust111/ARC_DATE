from __future__ import annotations

import random
from collections import Counter

import numpy as np


BG = 0
FRAME = 5
PALETTE = (1, 2, 3, 4, 6, 7, 8, 9)
GEN_RETRIES = 400


def _neighbors4(r: int, c: int) -> list[tuple[int, int]]:
    return [(r - 1, c), (r + 1, c), (r, c - 1), (r, c + 1)]


def _components_mask(mask: np.ndarray) -> list[list[tuple[int, int]]]:
    h, w = mask.shape
    seen = np.zeros_like(mask, dtype=bool)
    comps: list[list[tuple[int, int]]] = []
    for r in range(h):
        for c in range(w):
            if not mask[r, c] or seen[r, c]:
                continue
            stack = [(r, c)]
            seen[r, c] = True
            comp: list[tuple[int, int]] = []
            while stack:
                cr, cc = stack.pop()
                comp.append((cr, cc))
                for nr, nc in _neighbors4(cr, cc):
                    if 0 <= nr < h and 0 <= nc < w and mask[nr, nc] and not seen[nr, nc]:
                        seen[nr, nc] = True
                        stack.append((nr, nc))
            comps.append(comp)
    return comps


def _bbox(coords: list[tuple[int, int]]) -> tuple[int, int, int, int]:
    rows = [r for r, _ in coords]
    cols = [c for _, c in coords]
    return min(rows), min(cols), max(rows), max(cols)


def _is_rect_frame(grid: np.ndarray, coords: list[tuple[int, int]], color: int) -> bool:
    r0, c0, r1, c1 = _bbox(coords)
    h = r1 - r0 + 1
    w = c1 - c0 + 1
    if h < 3 or w < 3:
        return False
    expected = 2 * (h + w) - 4
    if len(coords) != expected:
        return False
    for r, c in coords:
        if r not in (r0, r1) and c not in (c0, c1):
            return False
    for c in range(c0, c1 + 1):
        if int(grid[r0, c]) != color or int(grid[r1, c]) != color:
            return False
    for r in range(r0, r1 + 1):
        if int(grid[r, c0]) != color or int(grid[r, c1]) != color:
            return False
    return True


def _find_frames(grid: np.ndarray, bg: int) -> list[dict[str, object]]:
    frames: list[dict[str, object]] = []
    for color in sorted(set(map(int, grid.flatten())) - {bg}):
        for comp in _components_mask(grid == color):
            if not _is_rect_frame(grid, comp, color):
                continue
            r0, c0, r1, c1 = _bbox(comp)
            inner = grid[r0 + 1 : r1, c0 + 1 : c1]
            frames.append(
                {
                    "color": color,
                    "bbox": (r0, c0, r1, c1),
                    "inner": inner,
                    "non_bg": int(np.count_nonzero(inner != bg)),
                    "inner_area": int(inner.size),
                }
            )
    return frames


def _pick_seed_and_canvas(grid: np.ndarray, bg: int) -> tuple[dict[str, object], dict[str, object]]:
    frames = _find_frames(grid, bg)
    if len(frames) < 2:
        raise ValueError("expected two framed regions")

    seed_candidates = [f for f in frames if int(f["non_bg"]) > 0]
    canvas_candidates = [f for f in frames if int(f["non_bg"]) == 0]
    if not seed_candidates or not canvas_candidates:
        raise ValueError("could not distinguish seed and canvas frames")

    seed = min(seed_candidates, key=lambda f: (int(f["bbox"][1]), int(f["inner_area"])))
    canvas = max(canvas_candidates, key=lambda f: (int(f["bbox"][1]), int(f["inner_area"])))
    return seed, canvas


def _ordered_colors_by_area(seed_inner: np.ndarray, bg: int) -> list[tuple[int, int]]:
    counts: Counter[int] = Counter()
    first_pos: dict[int, tuple[int, int]] = {}
    h, w = seed_inner.shape
    for r in range(h):
        for c in range(w):
            color = int(seed_inner[r, c])
            if color == bg:
                continue
            counts[color] += 1
            if color not in first_pos:
                first_pos[color] = (r, c)
    if not counts:
        raise ValueError("seed must contain at least one non-background color")
    return sorted(counts.items(), key=lambda item: (-item[1], first_pos[item[0]][0], first_pos[item[0]][1]))


def _stripe_heights(ordered_counts: list[tuple[int, int]], total_height: int) -> list[int]:
    total = sum(count for _, count in ordered_counts)
    heights: list[int] = []
    used = 0
    for idx, (_color, count) in enumerate(ordered_counts):
        if idx == len(ordered_counts) - 1:
            h = total_height - used
        else:
            h = (total_height * count) // total
            used += h
        heights.append(h)
    return heights


def main(grid: np.ndarray) -> np.ndarray:
    """
    Convert the seed-region color areas into proportional horizontal stripes on the canvas.
    """
    g = np.asarray(grid, dtype=int)
    bg = int(Counter(map(int, g.flatten())).most_common(1)[0][0])
    seed, canvas = _pick_seed_and_canvas(g, bg)

    ordered = _ordered_colors_by_area(np.asarray(seed["inner"]), bg)
    canvas_r0, canvas_c0, canvas_r1, canvas_c1 = map(int, canvas["bbox"])
    canvas_inner_h = canvas_r1 - canvas_r0 - 1
    canvas_inner_w = canvas_c1 - canvas_c0 - 1
    heights = _stripe_heights(ordered, canvas_inner_h)

    out = g.copy()
    start = canvas_r0 + 1
    for (color, _count), stripe_h in zip(ordered, heights):
        if stripe_h > 0:
            out[start : start + stripe_h, canvas_c0 + 1 : canvas_c1] = color
        start += stripe_h

    return out


def _grow_partition(h: int, w: int, n_regions: int, rng: random.Random) -> np.ndarray | None:
    cells = [(r, c) for r in range(h) for c in range(w)]
    seeds = rng.sample(cells, n_regions)
    labels = np.full((h, w), -1, dtype=int)
    frontiers: list[set[tuple[int, int]]] = [set() for _ in range(n_regions)]
    sizes = [0] * n_regions

    for idx, (r, c) in enumerate(seeds):
        labels[r, c] = idx
        sizes[idx] = 1

    for idx, (r, c) in enumerate(seeds):
        for nr, nc in _neighbors4(r, c):
            if 0 <= nr < h and 0 <= nc < w and labels[nr, nc] == -1:
                frontiers[idx].add((nr, nc))

    remaining = h * w - n_regions
    while remaining > 0:
        expandable = [idx for idx in range(n_regions) if frontiers[idx]]
        if not expandable:
            return None

        min_size = min(sizes[idx] for idx in expandable)
        weighted_regions = []
        for idx in expandable:
            weight = 3 if sizes[idx] == min_size else 1
            weighted_regions.extend([idx] * weight)
        region = rng.choice(weighted_regions)
        cell = rng.choice(tuple(frontiers[region]))
        rr, cc = cell
        labels[rr, cc] = region
        sizes[region] += 1
        remaining -= 1

        for frontier in frontiers:
            frontier.discard(cell)
        for nr, nc in _neighbors4(rr, cc):
            if 0 <= nr < h and 0 <= nc < w and labels[nr, nc] == -1:
                frontiers[region].add((nr, nc))

    return labels


def _partition_ok(labels: np.ndarray, n_regions: int) -> bool:
    counts = [int(np.count_nonzero(labels == idx)) for idx in range(n_regions)]
    if min(counts) < 2 or len(set(counts)) != n_regions:
        return False

    boxy_regions = 0
    for idx in range(n_regions):
        pts = np.argwhere(labels == idx)
        r0, c0 = pts.min(axis=0)
        r1, c1 = pts.max(axis=0)
        if (r1 - r0 + 1) * (c1 - c0 + 1) == len(pts):
            boxy_regions += 1
    if boxy_regions == n_regions:
        return False
    return True


def _paint_seed(labels: np.ndarray, colors: list[int]) -> np.ndarray:
    h, w = labels.shape
    seed = np.full((h, w), BG, dtype=int)
    for idx, color in enumerate(colors):
        seed[labels == idx] = color
    return seed


def _expected_canvas(seed_inner: np.ndarray, canvas_inner_h: int, canvas_inner_w: int) -> np.ndarray:
    ordered = _ordered_colors_by_area(seed_inner, BG)
    heights = _stripe_heights(ordered, canvas_inner_h)
    out = np.full((canvas_inner_h, canvas_inner_w), BG, dtype=int)
    start = 0
    for (color, _count), stripe_h in zip(ordered, heights):
        if stripe_h > 0:
            out[start : start + stripe_h, :] = color
        start += stripe_h
    return out


def _gen_attempt(rng: random.Random) -> np.ndarray | None:
    seed_inner_h = rng.randint(4, 6)
    seed_inner_w = rng.randint(4, 6)
    n_colors = rng.randint(2, 4)
    if seed_inner_h * seed_inner_w < n_colors * (n_colors + 3) // 2:
        return None

    labels = _grow_partition(seed_inner_h, seed_inner_w, n_colors, rng)
    if labels is None or not _partition_ok(labels, n_colors):
        return None

    colors = rng.sample(PALETTE, n_colors)
    rng.shuffle(colors)
    seed_inner = _paint_seed(labels, colors)

    seed_outer_h = seed_inner_h + 2
    seed_outer_w = seed_inner_w + 2

    scales = list(range(2, 6))
    rng.shuffle(scales)
    canvas_outer_h = None
    canvas_inner_h = None
    for scale in scales:
        candidate_outer_h = seed_outer_h * scale
        candidate_inner_h = candidate_outer_h - 2
        heights = _stripe_heights(_ordered_colors_by_area(seed_inner, BG), candidate_inner_h)
        if min(heights) >= 1:
            canvas_outer_h = candidate_outer_h
            canvas_inner_h = candidate_inner_h
            break
    if canvas_outer_h is None or canvas_inner_h is None:
        return None

    canvas_outer_w = rng.randint(max(6, seed_outer_w), 14)
    canvas_inner_w = canvas_outer_w - 2

    left_margin = rng.randint(1, 4)
    gap = rng.randint(2, 6)
    right_margin = rng.randint(1, 4)
    seed_top = rng.randint(1, 4)
    canvas_top = rng.randint(1, 4)
    bottom_margin = rng.randint(1, 4)

    seed_left = left_margin
    canvas_left = seed_left + seed_outer_w + gap
    grid_h = max(seed_top + seed_outer_h, canvas_top + canvas_outer_h) + bottom_margin
    grid_w = canvas_left + canvas_outer_w + right_margin

    grid = np.full((grid_h, grid_w), BG, dtype=int)

    sr0, sc0 = seed_top, seed_left
    sr1, sc1 = sr0 + seed_outer_h - 1, sc0 + seed_outer_w - 1
    grid[sr0, sc0 : sc1 + 1] = FRAME
    grid[sr1, sc0 : sc1 + 1] = FRAME
    grid[sr0 : sr1 + 1, sc0] = FRAME
    grid[sr0 : sr1 + 1, sc1] = FRAME
    grid[sr0 + 1 : sr1, sc0 + 1 : sc1] = seed_inner

    cr0, cc0 = canvas_top, canvas_left
    cr1, cc1 = cr0 + canvas_outer_h - 1, cc0 + canvas_outer_w - 1
    grid[cr0, cc0 : cc1 + 1] = FRAME
    grid[cr1, cc0 : cc1 + 1] = FRAME
    grid[cr0 : cr1 + 1, cc0] = FRAME
    grid[cr0 : cr1 + 1, cc1] = FRAME

    expected_canvas = _expected_canvas(seed_inner, canvas_inner_h, canvas_inner_w)
    produced = main(grid)
    if np.array_equal(produced, grid):
        return None
    if not np.array_equal(produced[cr0 + 1 : cr1, cc0 + 1 : cc1], expected_canvas):
        return None
    if not np.array_equal(produced[sr0 : sr1 + 1, sc0 : sc1 + 1], grid[sr0 : sr1 + 1, sc0 : sc1 + 1]):
        return None

    return grid


def generate_input() -> np.ndarray:
    rng = random.Random()
    for _ in range(GEN_RETRIES):
        grid = _gen_attempt(rng)
        if grid is not None:
            return grid
    raise RuntimeError("generate_input: exhausted attempts")
