import math
import random
from collections import Counter, deque

import numpy as np


GEN_RETRIES = 400
COLOR_POOL = tuple(range(1, 10))
LEGEND_COUNT_RANGE = (3, 6)
FENCE_COUNT_RANGE = (2, 4)


def _neighbors4(r: int, c: int):
    yield r - 1, c
    yield r + 1, c
    yield r, c - 1
    yield r, c + 1


def _components(mask: np.ndarray) -> list[list[tuple[int, int]]]:
    h, w = mask.shape
    seen = np.zeros((h, w), dtype=bool)
    comps: list[list[tuple[int, int]]] = []
    for r in range(h):
        for c in range(w):
            if not bool(mask[r, c]) or bool(seen[r, c]):
                continue
            q = deque([(r, c)])
            seen[r, c] = True
            comp: list[tuple[int, int]] = []
            while q:
                rr, cc = q.popleft()
                comp.append((rr, cc))
                for nr, nc in _neighbors4(rr, cc):
                    if 0 <= nr < h and 0 <= nc < w and bool(mask[nr, nc]) and not bool(seen[nr, nc]):
                        seen[nr, nc] = True
                        q.append((nr, nc))
            comps.append(comp)
    return comps


def _bbox(cells: list[tuple[int, int]]) -> tuple[int, int, int, int]:
    rows = [r for r, _ in cells]
    cols = [c for _, c in cells]
    return min(rows), min(cols), max(rows), max(cols)


def _hole_regions(void_mask: np.ndarray) -> list[list[tuple[int, int]]]:
    h, w = void_mask.shape
    seen = np.zeros((h, w), dtype=bool)
    holes: list[list[tuple[int, int]]] = []
    for r in range(h):
        for c in range(w):
            if not bool(void_mask[r, c]) or bool(seen[r, c]):
                continue
            q = deque([(r, c)])
            seen[r, c] = True
            region: list[tuple[int, int]] = []
            touches_border = False
            while q:
                rr, cc = q.popleft()
                region.append((rr, cc))
                if rr in (0, h - 1) or cc in (0, w - 1):
                    touches_border = True
                for nr, nc in _neighbors4(rr, cc):
                    if 0 <= nr < h and 0 <= nc < w and bool(void_mask[nr, nc]) and not bool(seen[nr, nc]):
                        seen[nr, nc] = True
                        q.append((nr, nc))
            if not touches_border:
                holes.append(region)
    return holes


def _legend_colors(grid: np.ndarray) -> list[int]:
    legend = grid[-2:, :]
    colors: list[int] = []
    for c in range(legend.shape[1]):
        vals = legend[:, c][legend[:, c] != 0]
        if vals.size:
            colors.append(int(vals[0]))
    return colors


def main(grid: np.ndarray) -> np.ndarray:
    """
    Fill each fenced interior using the legend color indexed by its hole count.

    Input structure:
    - Background is black.
    - The bottom two rows form a legend strip of separated 1x1 colored pixels read from left to right.
    - Above the legend are several 1-pixel-thick closed fences, all drawn in one shared foreground color, and internal walls of that same color may split a fence into multiple black compartments.

    Rule:
    - Read the bottom legend left to right so the first colored pixel means 1, the second means 2, and so on.
    - In the area above the legend, find each 4-connected fence component and count the black 4-connected regions inside its bounding box that do not touch that box boundary.
    - Fill every black pixel in those enclosed regions with the legend color selected by that component's hole count, leaving the fence pixels and legend unchanged.

    Key insight:
    The lookup key is the number of distinct black compartments created inside one closed fence, not the fence's size, position, or exact geometry.
    """
    g = np.asarray(grid, dtype=int)
    out = g.copy()
    if g.ndim != 2 or g.shape[0] < 3:
        return out

    legend_colors = _legend_colors(g)
    if not legend_colors:
        return out

    upper = g[:-2, :]
    foreground = [int(v) for v in upper.flat if int(v) != 0]
    if not foreground:
        return out
    fence_color = Counter(foreground).most_common(1)[0][0]

    for comp in _components(upper == int(fence_color)):
        r0, c0, r1, c1 = _bbox(comp)
        sub = upper[r0 : r1 + 1, c0 : c1 + 1]
        holes = _hole_regions(sub == 0)
        hole_count = len(holes)
        if not (1 <= hole_count <= len(legend_colors)):
            continue
        fill_color = int(legend_colors[hole_count - 1])
        for region in holes:
            for rr, cc in region:
                out[r0 + rr, c0 + cc] = fill_color

    return out


def _factor_pairs(n: int) -> list[tuple[int, int]]:
    pairs: list[tuple[int, int]] = []
    for r in range(1, math.isqrt(n) + 1):
        if n % r != 0:
            continue
        c = n // r
        pairs.append((r, c))
        if r != c:
            pairs.append((c, r))
    return pairs


def _room_spans(parts: int, rng: random.Random) -> list[int]:
    if parts >= 4:
        lo, hi = 1, 2
    elif parts == 3:
        lo, hi = 1, 3
    else:
        lo, hi = 2, 4
    return [rng.randint(lo, hi) for _ in range(parts)]


def _build_patch(hole_count: int, rng: random.Random) -> np.ndarray | None:
    pairs = _factor_pairs(hole_count)
    rng.shuffle(pairs)
    for rows, cols in pairs:
        for _ in range(30):
            heights = _room_spans(rows, rng)
            widths = _room_spans(cols, rng)
            h = 2 + sum(heights) + rows - 1
            w = 2 + sum(widths) + cols - 1
            if h > 15 or w > 18:
                continue

            patch = np.zeros((h, w), dtype=int)
            patch[0, :] = 1
            patch[-1, :] = 1
            patch[:, 0] = 1
            patch[:, -1] = 1

            cut_r = 1
            for span in heights[:-1]:
                cut_r += span
                patch[cut_r, 1:-1] = 1
                cut_r += 1

            cut_c = 1
            for span in widths[:-1]:
                cut_c += span
                patch[1:-1, cut_c] = 1
                cut_c += 1
            return patch
    return None


def _sample_hole_counts(legend_count: int, fence_count: int, rng: random.Random) -> list[int]:
    if fence_count >= 3 and rng.random() < 0.5:
        counts = [rng.randint(1, legend_count) for _ in range(fence_count)]
        if len(set(counts)) == 1:
            counts[-1] = legend_count if counts[0] != legend_count else 1
        return counts
    return rng.sample(list(range(1, legend_count + 1)), k=fence_count)


def _place_items(items: list[dict], object_h: int, width: int, rng: random.Random) -> list[dict] | None:
    blocked = np.zeros((object_h, width), dtype=bool)
    placed: list[dict] = []
    for item in sorted(items, key=lambda x: x["patch"].shape[0] * x["patch"].shape[1], reverse=True):
        patch = item["patch"]
        ph, pw = patch.shape
        if ph + 2 > object_h or pw + 2 > width:
            return None

        candidates: list[tuple[int, int]] = []
        for top in range(1, object_h - ph):
            for left in range(1, width - pw):
                if not bool(blocked[top - 1 : top + ph + 1, left - 1 : left + pw + 1].any()):
                    candidates.append((top, left))
        if not candidates:
            return None

        top, left = rng.choice(candidates)
        blocked[top - 1 : top + ph + 1, left - 1 : left + pw + 1] = True
        placed.append({**item, "top": top, "left": left})
    return placed


def _legend_rows(count: int, rng: random.Random) -> list[int]:
    mode = rng.randrange(4)
    if mode == 0:
        return [0] * count
    if mode == 1:
        return [1] * count
    if mode == 2:
        start = rng.randint(0, 1)
        return [(start + i) % 2 for i in range(count)]
    rows = [rng.randint(0, 1) for _ in range(count)]
    if count > 1 and len(set(rows)) == 1:
        rows[rng.randrange(count)] = 1 - rows[0]
    return rows


def _gen_attempt(rng: random.Random) -> np.ndarray | None:
    legend_count = rng.randint(*LEGEND_COUNT_RANGE)
    fence_count = rng.randint(2, min(FENCE_COUNT_RANGE[1], legend_count))

    fence_color = rng.choice(COLOR_POOL)
    palette = [c for c in COLOR_POOL if c != fence_color]
    legend_colors = rng.sample(palette, legend_count)
    hole_counts = _sample_hole_counts(legend_count, fence_count, rng)

    items: list[dict] = []
    for hole_count in hole_counts:
        patch = _build_patch(hole_count, rng)
        if patch is None:
            return None
        items.append({"patch": patch, "holes": hole_count})

    legend_span = 2 * legend_count - 1
    max_ph = max(item["patch"].shape[0] for item in items)
    max_pw = max(item["patch"].shape[1] for item in items)
    halo_area = sum((item["patch"].shape[0] + 2) * (item["patch"].shape[1] + 2) for item in items)

    width_low = max(legend_span + 2, max_pw + 2, 12)
    if width_low > 30:
        return None

    for _ in range(50):
        width = rng.randint(width_low, 30)
        gap = rng.randint(1, 3)
        height_low = max(max_ph + gap + 4, math.ceil(halo_area / max(1, width - 2)) + gap + 3, 12)
        if height_low > 30:
            continue
        height = rng.randint(height_low, 30)
        object_h = height - gap - 2
        placed = _place_items(items, object_h, width, rng)
        if placed is None:
            continue

        grid = np.zeros((height, width), dtype=int)
        for item in placed:
            patch = item["patch"]
            ph, pw = patch.shape
            top, left = item["top"], item["left"]
            grid[top : top + ph, left : left + pw][patch == 1] = fence_color

        start_col = rng.randint(0, width - legend_span)
        for idx, row_sel in enumerate(_legend_rows(legend_count, rng)):
            grid[height - 2 + row_sel, start_col + 2 * idx] = legend_colors[idx]

        expected = grid.copy()
        for item in placed:
            patch = item["patch"]
            ph, pw = patch.shape
            top, left = item["top"], item["left"]
            fill_color = legend_colors[item["holes"] - 1]
            expected[top : top + ph, left : left + pw][patch == 0] = fill_color

        if np.array_equal(main(grid), expected):
            return grid
    return None


def generate_input() -> np.ndarray:
    rng = random.Random()
    for _ in range(GEN_RETRIES):
        grid = _gen_attempt(rng)
        if grid is not None:
            return grid
    raise RuntimeError("failed to generate a valid input")
