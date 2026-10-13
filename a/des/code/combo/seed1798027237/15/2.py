from __future__ import annotations

import numpy as np


def _most_common_color(grid: np.ndarray) -> int:
    vals, counts = np.unique(grid, return_counts=True)
    return int(vals[np.argmax(counts)])


def _row_runs(rows: np.ndarray) -> list[tuple[int, int]]:
    if rows.size == 0:
        return []
    runs: list[tuple[int, int]] = []
    start = int(rows[0])
    prev = int(rows[0])
    for raw in rows[1:]:
        cur = int(raw)
        if cur == prev + 1:
            prev = cur
            continue
        runs.append((start, prev))
        start = cur
        prev = cur
    runs.append((start, prev))
    return runs


def _find_palette_column(grid: np.ndarray, bg: int) -> tuple[int, list[tuple[int, int, int]]]:
    h, w = grid.shape
    best: tuple[int, int, list[tuple[int, int, int]]] | None = None

    for c in range(w // 2, w):
        rows = np.flatnonzero(grid[:, c] != bg)
        if rows.size < 4:
            continue

        runs = _row_runs(rows)
        if not runs or any(r1 - r0 + 1 != 2 for r0, r1 in runs):
            continue

        pairs: list[tuple[int, int, int]] = []
        ok = True
        for r0, r1 in runs:
            top = int(grid[r0, c])
            bottom = int(grid[r1, c])
            if top == bg or bottom == bg or top == bottom:
                ok = False
                break
            for rr in (r0, r1):
                if c - 1 >= 0 and int(grid[rr, c - 1]) != bg:
                    ok = False
                    break
                if c + 1 < w and int(grid[rr, c + 1]) != bg:
                    ok = False
                    break
            if not ok:
                break
            pairs.append((int(r0), top, bottom))

        if not ok:
            continue

        darks = [dark for _r, dark, _light in pairs]
        if len(darks) != len(set(darks)):
            continue

        score = (len(pairs), c)
        if best is None or score > (best[0], best[1]):
            best = (score[0], score[1], pairs)

    assert best is not None, "failed to find the palette column"
    pair_count, c, pairs = best
    assert pair_count >= 2, "palette should provide at least two color pairs"
    return int(c), pairs


def main(grid: np.ndarray) -> np.ndarray:
    """
    Group same-color fragments by color, wrap each group in its minimal bounding
    rectangle, fill the interior from a right-side color mapping, and layer
    smaller rectangles on top when overlaps occur.
    """
    g = np.asarray(grid, dtype=int)
    assert g.ndim == 2, "expected a 2D grid"
    bg = _most_common_color(g)

    pal_c, pairs = _find_palette_column(g, bg)
    mapping = {dark: light for _r, dark, light in pairs}

    palette_mask = np.zeros_like(g, dtype=bool)
    for r0, _dark, _light in pairs:
        palette_mask[r0 : r0 + 2, pal_c] = True

    fragment_mask = (g != bg) & (~palette_mask)
    colors_in_scan = [int(v) for v in g[fragment_mask].ravel().tolist()]
    colors = list(dict.fromkeys(colors_in_scan))
    assert colors, "expected left-side fragments"

    groups: list[dict[str, object]] = []
    for color in colors:
        assert color in mapping, "every fragment color must appear in the palette"
        coords = np.argwhere((g == color) & (~palette_mask))
        r0, c0 = map(int, coords.min(axis=0))
        r1, c1 = map(int, coords.max(axis=0))
        groups.append(
            {
                "color": int(color),
                "light": int(mapping[color]),
                "bbox": (r0, c0, r1, c1),
                "area": int((r1 - r0 + 1) * (c1 - c0 + 1)),
            }
        )

    groups.sort(
        key=lambda item: (
            -int(item["area"]),
            int(item["bbox"][0]),
            int(item["bbox"][1]),
            int(item["bbox"][2]),
            int(item["bbox"][3]),
        )
    )

    out = np.full_like(g, bg)
    for item in groups:
        color = int(item["color"])
        light = int(item["light"])
        r0, c0, r1, c1 = item["bbox"]
        patch = np.full((r1 - r0 + 1, c1 - c0 + 1), light, dtype=int)
        patch[0, :] = color
        patch[-1, :] = color
        patch[:, 0] = color
        patch[:, -1] = color

        local = g[r0 : r1 + 1, c0 : c1 + 1]
        local_palette = palette_mask[r0 : r1 + 1, c0 : c1 + 1]
        patch[(local == color) & (~local_palette)] = color
        out[r0 : r1 + 1, c0 : c1 + 1] = patch

    return out


# --------------- generator (codex) ---------------

PAIR_OPTIONS: tuple[tuple[int, int], ...] = ((1, 2), (3, 4), (5, 6), (7, 8))
GEN_ATTEMPTS = 400


def _randint(rng: np.random.Generator, lo: int, hi: int) -> int:
    return int(rng.integers(int(lo), int(hi) + 1))


def _boxes_intersect(
    a: tuple[int, int, int, int],
    b: tuple[int, int, int, int],
    margin: int = 0,
) -> bool:
    ar0, ac0, ar1, ac1 = a
    br0, bc0, br1, bc1 = b
    return not (
        ar1 + margin < br0
        or br1 + margin < ar0
        or ac1 + margin < bc0
        or bc1 + margin < ac0
    )


def _intersection_area(a: tuple[int, int, int, int], b: tuple[int, int, int, int]) -> int:
    r = max(0, min(a[2], b[2]) - max(a[0], b[0]) + 1)
    c = max(0, min(a[3], b[3]) - max(a[1], b[1]) + 1)
    return int(r * c)


def _sample_palette_rows(
    height: int,
    pair_count: int,
    rng: np.random.Generator,
) -> list[int] | None:
    for _ in range(60):
        gaps = [_randint(rng, 1, 3) for _ in range(pair_count - 1)]
        total = 2 * pair_count + sum(gaps)
        slack = height - total
        if slack < 2:
            continue
        top_margin = _randint(rng, 1, slack - 1)
        rows: list[int] = []
        cur = top_margin
        for i in range(pair_count):
            rows.append(cur)
            if i + 1 < pair_count:
                cur += 2 + gaps[i]
        if rows[-1] + 2 <= height - 1:
            return rows
    return None


def _random_bbox(
    rng: np.random.Generator,
    row_min: int,
    row_max: int,
    col_min: int,
    col_max: int,
    h_lo: int,
    h_hi: int,
    w_lo: int,
    w_hi: int,
) -> tuple[int, int, int, int] | None:
    if h_lo > h_hi or w_lo > w_hi:
        return None
    bh = _randint(rng, h_lo, h_hi)
    bw = _randint(rng, w_lo, w_hi)
    if row_min > row_max - bh + 1 or col_min > col_max - bw + 1:
        return None
    r0 = _randint(rng, row_min, row_max - bh + 1)
    c0 = _randint(rng, col_min, col_max - bw + 1)
    return (r0, c0, r0 + bh - 1, c0 + bw - 1)


def _sample_fragment_cells(
    bbox: tuple[int, int, int, int],
    occupied: np.ndarray,
    rng: np.random.Generator,
) -> set[tuple[int, int]] | None:
    r0, c0, r1, c1 = bbox
    area = (r1 - r0 + 1) * (c1 - c0 + 1)
    target_hi = min(max(5, area // 3), 10)

    for _ in range(120):
        top = [(r0, c) for c in range(c0, c1 + 1) if not occupied[r0, c]]
        bottom = [(r1, c) for c in range(c0, c1 + 1) if not occupied[r1, c]]
        left = [(r, c0) for r in range(r0, r1 + 1) if not occupied[r, c0]]
        right = [(r, c1) for r in range(r0, r1 + 1) if not occupied[r, c1]]
        if not top or not bottom or not left or not right:
            continue

        cells = {
            top[_randint(rng, 0, len(top) - 1)],
            bottom[_randint(rng, 0, len(bottom) - 1)],
            left[_randint(rng, 0, len(left) - 1)],
            right[_randint(rng, 0, len(right) - 1)],
        }
        target = _randint(rng, len(cells), target_hi)

        while len(cells) < target:
            frontier: set[tuple[int, int]] = set()
            for r, c in cells:
                for dr, dc in ((-1, 0), (1, 0), (0, -1), (0, 1)):
                    nr, nc = r + dr, c + dc
                    if r0 <= nr <= r1 and c0 <= nc <= c1 and not occupied[nr, nc]:
                        if (nr, nc) not in cells:
                            frontier.add((nr, nc))
            if frontier and rng.random() < 0.65:
                frontier_list = list(frontier)
                cells.add(frontier_list[_randint(rng, 0, len(frontier_list) - 1)])
                continue

            free = [
                (r, c)
                for r in range(r0, r1 + 1)
                for c in range(c0, c1 + 1)
                if not occupied[r, c] and (r, c) not in cells
            ]
            if not free:
                break
            cells.add(free[_randint(rng, 0, len(free) - 1)])

        rows = [r for r, _c in cells]
        cols = [c for _r, c in cells]
        if min(rows) == r0 and max(rows) == r1 and min(cols) == c0 and max(cols) == c1:
            return cells

    return None


def _sample_bboxes(
    height: int,
    width: int,
    group_count: int,
    overlap_mode: bool,
    rng: np.random.Generator,
) -> list[tuple[int, int, int, int]] | None:
    row_min, row_max = 1, height - 2
    col_min, col_max = 1, width // 2 - 2
    max_bh = min(8, row_max - row_min + 1)
    max_bw = min(8, col_max - col_min + 1)
    if max_bh < 4 or max_bw < 4:
        return None

    bboxes: list[tuple[int, int, int, int]] = []

    if overlap_mode:
        large = _random_bbox(
            rng,
            row_min,
            row_max,
            col_min,
            col_max,
            min(5, max_bh),
            max_bh,
            min(5, max_bw),
            max_bw,
        )
        if large is None:
            return None

        area_large = (large[2] - large[0] + 1) * (large[3] - large[1] + 1)
        small = None
        for _ in range(120):
            bh = _randint(rng, 4, min(6, max_bh))
            bw = _randint(rng, 4, min(6, max_bw))
            area_small = bh * bw
            if area_small == area_large:
                continue

            r0_lo = max(row_min, large[0] - bh + 2)
            r0_hi = min(row_max - bh + 1, large[2] - 1)
            c0_lo = max(col_min, large[1] - bw + 2)
            c0_hi = min(col_max - bw + 1, large[3] - 1)
            if r0_lo > r0_hi or c0_lo > c0_hi:
                continue

            r0 = _randint(rng, r0_lo, r0_hi)
            c0 = _randint(rng, c0_lo, c0_hi)
            cand = (r0, c0, r0 + bh - 1, c0 + bw - 1)
            if _intersection_area(large, cand) == 0:
                continue
            small = cand
            break

        if small is None:
            return None
        bboxes.extend([large, small])

    h_hi = min(max_bh, 6 if group_count >= 3 else 7)
    w_hi = min(max_bw, 6 if group_count >= 3 else 7)
    for _ in range(group_count - len(bboxes)):
        placed = None
        for _ in range(120):
            cand = _random_bbox(rng, row_min, row_max, col_min, col_max, 4, h_hi, 4, w_hi)
            if cand is None:
                return None
            if all(not _boxes_intersect(cand, other, margin=1) for other in bboxes):
                placed = cand
                break
        if placed is None:
            return None
        bboxes.append(placed)

    for i, box_a in enumerate(bboxes):
        area_a = (box_a[2] - box_a[0] + 1) * (box_a[3] - box_a[1] + 1)
        for box_b in bboxes[i + 1 :]:
            area_b = (box_b[2] - box_b[0] + 1) * (box_b[3] - box_b[1] + 1)
            if _intersection_area(box_a, box_b) > 0 and area_a == area_b:
                return None

    return bboxes


def _generate_attempt(rng: np.random.Generator) -> np.ndarray | None:
    group_count = _randint(rng, 2, len(PAIR_OPTIONS))
    min_height = max(12, 3 * group_count + 2)
    height = _randint(rng, min_height, 20)
    width = _randint(rng, 18, 28)

    palette_rows = _sample_palette_rows(height, group_count, rng)
    if palette_rows is None:
        return None

    pal_c_lo = max(width // 2 + 1, width - 4)
    pal_c_hi = width - 2
    if pal_c_lo > pal_c_hi:
        return None
    pal_c = _randint(rng, pal_c_lo, pal_c_hi)

    overlap_mode = bool(group_count >= 2 and rng.random() < 0.5)
    bboxes = _sample_bboxes(height, width, group_count, overlap_mode, rng)
    if bboxes is None:
        return None

    chosen_idx = rng.choice(len(PAIR_OPTIONS), size=group_count, replace=False)
    pairs = [PAIR_OPTIONS[int(i)] for i in chosen_idx.tolist()]
    rng.shuffle(pairs)

    grid = np.zeros((height, width), dtype=int)
    for r0, pair in zip(palette_rows, pairs, strict=False):
        dark, light = pair
        grid[r0, pal_c] = dark
        grid[r0 + 1, pal_c] = light

    occupied = np.zeros_like(grid, dtype=bool)
    occupied[grid != 0] = True

    for bbox, pair in zip(bboxes, pairs, strict=False):
        dark, _light = pair
        cells = _sample_fragment_cells(bbox, occupied, rng)
        if cells is None:
            return None
        for r, c in cells:
            grid[r, c] = dark
            occupied[r, c] = True

    try:
        out = main(grid)
    except Exception:
        return None

    if np.array_equal(out, grid):
        return None
    return grid


def generate_input() -> np.ndarray:
    for _ in range(GEN_ATTEMPTS):
        rng = np.random.default_rng()
        grid = _generate_attempt(rng)
        if grid is not None:
            return grid
    raise RuntimeError(f"failed to generate a valid input after {GEN_ATTEMPTS} attempts")
