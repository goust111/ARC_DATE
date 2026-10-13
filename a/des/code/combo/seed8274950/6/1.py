from __future__ import annotations

from collections import Counter
from itertools import combinations, product
import random

import numpy as np


ORIENTATION_ORDER = ("up", "down", "left", "right")


def _get_bg(grid: np.ndarray) -> int:
    values, counts = np.unique(grid, return_counts=True)
    return int(values[np.argmax(counts)])


def _components4(mask: np.ndarray) -> list[list[tuple[int, int]]]:
    h, w = mask.shape
    seen = np.zeros((h, w), dtype=bool)
    comps: list[list[tuple[int, int]]] = []
    for r0 in range(h):
        for c0 in range(w):
            if not mask[r0, c0] or seen[r0, c0]:
                continue
            stack = [(r0, c0)]
            seen[r0, c0] = True
            comp: list[tuple[int, int]] = []
            while stack:
                r, c = stack.pop()
                comp.append((int(r), int(c)))
                for nr, nc in ((r - 1, c), (r + 1, c), (r, c - 1), (r, c + 1)):
                    if 0 <= nr < h and 0 <= nc < w and mask[nr, nc] and not seen[nr, nc]:
                        seen[nr, nc] = True
                        stack.append((nr, nc))
            comps.append(comp)
    return comps


def _find_legend(grid: np.ndarray, bg: int) -> tuple[str, int, int, list[int]]:
    h, _ = grid.shape
    candidates: list[tuple[tuple[int, int], str, int, int, list[int]]] = []
    for side, row, inner_row in (("top", 0, 1), ("bottom", h - 1, h - 2)):
        arr = grid[row]
        start = 0
        while start < arr.size:
            if int(arr[start]) == bg:
                start += 1
                continue
            end = start
            while end < arr.size and int(arr[end]) != bg:
                end += 1
            colors = [int(x) for x in arr[start:end].tolist()]
            if len(colors) >= 2 and np.all(grid[inner_row, start:end] == bg):
                score = (len(colors), len(set(colors)))
                candidates.append((score, side, int(start), int(end), colors))
            start = end
    assert candidates, "expected a top or bottom legend row"
    _, side, start, end, colors = max(candidates, key=lambda item: item[0])
    return side, start, end, colors


def _find_wall_color(grid: np.ndarray, bg: int, legend_mask: np.ndarray) -> int:
    h, w = grid.shape
    counts = Counter(int(v) for v in grid[~legend_mask].ravel().tolist() if int(v) != bg)
    assert counts, "expected a non-background wall color"
    best_color = None
    best_score = None
    for color, count in counts.items():
        pts = np.argwhere((grid == int(color)) & (~legend_mask))
        inner = int(np.sum((pts[:, 0] > 0) & (pts[:, 0] < h - 1) & (pts[:, 1] > 0) & (pts[:, 1] < w - 1)))
        score = (count, inner)
        if best_score is None or score > best_score:
            best_score = score
            best_color = int(color)
    assert best_color is not None
    return best_color


def _gap_orientation(r: int, c: int, r0: int, c0: int, r1: int, c1: int) -> str:
    if r == r0 and c0 < c < c1:
        return "up"
    if r == r1 and c0 < c < c1:
        return "down"
    if c == c0 and r0 < r < r1:
        return "left"
    if c == c1 and r0 < r < r1:
        return "right"
    raise AssertionError("gap must sit on a non-corner border cell")


def _extract_regions(grid: np.ndarray, wall: int, bg: int, legend_mask: np.ndarray) -> list[dict[str, object]]:
    wall_cells = {(int(r), int(c)) for r, c in np.argwhere((grid == int(wall)) & (~legend_mask))}
    rows_by_col: dict[int, list[int]] = {}
    cols_by_row: dict[int, list[int]] = {}
    for r, c in wall_cells:
        rows_by_col.setdefault(c, []).append(r)
        cols_by_row.setdefault(r, []).append(c)
    for seq in rows_by_col.values():
        seq.sort()
    for seq in cols_by_row.values():
        seq.sort()

    candidates: list[dict[str, object]] = []
    for r0, c0 in sorted(wall_cells):
        right_cols = [c1 for c1 in cols_by_row[r0] if c1 >= c0 + 2]
        down_rows = [r1 for r1 in rows_by_col[c0] if r1 >= r0 + 2]
        for c1 in right_cols:
            for r1 in down_rows:
                if (r1, c1) not in wall_cells:
                    continue
                if np.any(grid[r0 + 1 : r1, c0 + 1 : c1] == int(wall)):
                    continue

                gaps: list[tuple[int, int]] = []
                edge_wall_counts = {side: 0 for side in ORIENTATION_ORDER}
                valid = True

                for c in range(c0, c1 + 1):
                    for r, side in ((r0, "up"), (r1, "down")):
                        value = int(grid[r, c])
                        if c in (c0, c1):
                            if value != int(wall):
                                valid = False
                                break
                            edge_wall_counts[side] += 1
                        elif value == int(wall):
                            edge_wall_counts[side] += 1
                        elif value == int(bg):
                            gaps.append((r, c))
                        else:
                            valid = False
                            break
                    if not valid:
                        break
                if not valid:
                    continue

                for r in range(r0 + 1, r1):
                    for c, side in ((c0, "left"), (c1, "right")):
                        value = int(grid[r, c])
                        if value == int(wall):
                            edge_wall_counts[side] += 1
                        elif value == int(bg):
                            gaps.append((r, c))
                        else:
                            valid = False
                            break
                    if not valid:
                        break
                if not valid or not gaps:
                    continue
                if any(edge_wall_counts[side] < 3 for side in ORIENTATION_ORDER):
                    continue
                candidates.append(
                    {
                        "bbox": (r0, c0, r1, c1),
                        "gaps": gaps,
                        "area": (r1 - r0 + 1) * (c1 - c0 + 1),
                    }
                )

    regions: list[dict[str, object]] = []
    used: list[tuple[int, int, int, int]] = []
    for candidate in sorted(candidates, key=lambda item: item["area"], reverse=True):
        bbox = candidate["bbox"]
        if any(_boxes_overlap(bbox, other, pad=0) for other in used):
            continue
        used.append(bbox)
        regions.append({"bbox": bbox, "gaps": candidate["gaps"]})

    assert regions, "expected at least one rectangular wall region"
    return regions


def _paint_centered_span(out: np.ndarray, fixed: int, start: int, span: int, length: int, color: int, horizontal: bool) -> None:
    if length <= 0:
        return
    offset = (span - length) // 2
    lo = start + offset
    hi = lo + length
    if horizontal:
        out[fixed, lo:hi] = int(color)
    else:
        out[lo:hi, fixed] = int(color)


def _apply_gradient(out: np.ndarray, bbox: tuple[int, int, int, int], orientation: str, color: int) -> None:
    r0, c0, r1, c1 = bbox
    inner_h = r1 - r0 - 1
    inner_w = c1 - c0 - 1
    if inner_h <= 0 or inner_w <= 0:
        return

    if orientation == "up":
        for d in range(inner_h):
            _paint_centered_span(out, r1 - 1 - d, c0 + 1, inner_w, inner_w - d, color, True)
        return
    if orientation == "down":
        for d in range(inner_h):
            _paint_centered_span(out, r0 + 1 + d, c0 + 1, inner_w, inner_w - d, color, True)
        return
    if orientation == "left":
        for d in range(inner_w):
            _paint_centered_span(out, c1 - 1 - d, r0 + 1, inner_h, inner_h - d, color, False)
        return
    if orientation == "right":
        for d in range(inner_w):
            _paint_centered_span(out, c0 + 1 + d, r0 + 1, inner_h, inner_h - d, color, False)
        return
    raise AssertionError(f"bad orientation: {orientation}")


def main(grid: np.ndarray) -> np.ndarray:
    """
    Fill each gapped rectangular wall region with a centered taper whose color is
    looked up from the edge legend by gap count.
    """
    g = np.asarray(grid, dtype=int)
    bg = _get_bg(g)
    legend_side, legend_start, legend_end, legend_colors = _find_legend(g, bg)
    legend_mask = np.zeros_like(g, dtype=bool)
    legend_row = 0 if legend_side == "top" else g.shape[0] - 1
    legend_mask[legend_row, legend_start:legend_end] = True

    wall = _find_wall_color(g, bg, legend_mask)
    regions = _extract_regions(g, wall, bg, legend_mask)

    out = g.copy()
    for region in regions:
        bbox = region["bbox"]
        gaps = region["gaps"]
        counts = Counter(_gap_orientation(r, c, *bbox) for r, c in gaps)
        orientation = max(ORIENTATION_ORDER, key=lambda side: counts.get(side, 0))
        gap_count = len(gaps)
        assert 1 <= gap_count <= len(legend_colors), "gap count must be covered by the legend"
        color = int(legend_colors[gap_count - 1])
        _apply_gradient(out, bbox, orientation, color)
    return out


def _non_adjacent_choices(side_len: int, k: int) -> list[tuple[int, ...]]:
    if k == 0:
        return [()]
    positions = range(1, side_len + 1)
    return [combo for combo in combinations(positions, k) if all(b - a >= 2 for a, b in zip(combo, combo[1:], strict=False))]


def _sample_gap_layout(outer_h: int, outer_w: int, gap_count: int, major_side: str) -> dict[str, tuple[int, ...]] | None:
    side_spans = {
        "up": outer_w - 2,
        "down": outer_w - 2,
        "left": outer_h - 2,
        "right": outer_h - 2,
    }
    side_caps = {side: (side_spans[side] + 1) // 2 for side in ORIENTATION_ORDER}
    min_major = gap_count // 2 + 1
    major_choices = list(range(min_major, min(side_caps[major_side], gap_count) + 1))
    random.shuffle(major_choices)

    other_sides = [side for side in ORIENTATION_ORDER if side != major_side]
    for major_count in major_choices:
        remain = gap_count - major_count
        count_options = []
        for counts in product(*(range(min(side_caps[side], major_count - 1) + 1) for side in other_sides)):
            if sum(counts) == remain:
                count_options.append(dict(zip(other_sides, counts, strict=False)))
        random.shuffle(count_options)
        for other_counts in count_options:
            layout_counts = {side: 0 for side in ORIENTATION_ORDER}
            layout_counts[major_side] = major_count
            layout_counts.update(other_counts)
            chosen: dict[str, tuple[int, ...]] = {}
            ok = True
            for side, count in layout_counts.items():
                options = _non_adjacent_choices(side_spans[side], count)
                if not options:
                    ok = False
                    break
                chosen[side] = tuple(random.choice(options))
            if ok:
                return chosen
    return None


def _boxes_overlap(a: tuple[int, int, int, int], b: tuple[int, int, int, int], pad: int) -> bool:
    ar0, ac0, ar1, ac1 = a
    br0, bc0, br1, bc1 = b
    return not (ar1 + pad < br0 or br1 + pad < ar0 or ac1 + pad < bc0 or bc1 + pad < ac0)


def _draw_region(grid: np.ndarray, bbox: tuple[int, int, int, int], wall: int, bg: int, gap_layout: dict[str, tuple[int, ...]]) -> None:
    r0, c0, r1, c1 = bbox
    grid[r0, c0 : c1 + 1] = int(wall)
    grid[r1, c0 : c1 + 1] = int(wall)
    grid[r0 : r1 + 1, c0] = int(wall)
    grid[r0 : r1 + 1, c1] = int(wall)

    for pos in gap_layout["up"]:
        grid[r0, c0 + pos] = int(bg)
    for pos in gap_layout["down"]:
        grid[r1, c0 + pos] = int(bg)
    for pos in gap_layout["left"]:
        grid[r0 + pos, c0] = int(bg)
    for pos in gap_layout["right"]:
        grid[r0 + pos, c1] = int(bg)


def _gen_attempt() -> np.ndarray | None:
    legend_len = random.choice([4, 5])
    region_count = random.randint(2, min(4, legend_len))
    min_outer = 2 * legend_len + 1

    base = 16 + 3 * region_count + 2 * (legend_len - 4)
    grid_h = random.randint(base, base + 6)
    grid_w = random.randint(base, base + 8)
    legend_side = random.choice(["top", "bottom"])

    colors = list(range(10))
    bg, wall = random.sample(colors, 2)
    legend_colors = random.sample([c for c in colors if c not in (bg, wall)], legend_len)

    grid = np.full((grid_h, grid_w), int(bg), dtype=int)
    legend_start = random.randint(1, grid_w - legend_len - 1)
    legend_row = 0 if legend_side == "top" else grid_h - 1
    grid[legend_row, legend_start : legend_start + legend_len] = np.array(legend_colors, dtype=int)

    counts = random.sample(range(1, legend_len + 1), region_count)
    orientations = random.sample(list(ORIENTATION_ORDER), region_count)
    random.shuffle(counts)
    random.shuffle(orientations)

    boxes: list[tuple[int, int, int, int]] = []
    specs: list[tuple[tuple[int, int, int, int], dict[str, tuple[int, ...]]]] = []
    for gap_count, orientation in zip(counts, orientations, strict=False):
        placed = False
        for _ in range(300):
            outer_h = random.randint(min_outer, min_outer + 4)
            outer_w = random.randint(min_outer, min_outer + 4)
            gap_layout = _sample_gap_layout(outer_h, outer_w, gap_count, orientation)
            if gap_layout is None:
                continue

            min_r0 = 2 if legend_side == "top" else 1
            max_r1 = grid_h - 3 if legend_side == "bottom" else grid_h - 2
            if max_r1 - min_r0 + 1 < outer_h:
                continue
            r0 = random.randint(min_r0, max_r1 - outer_h + 1)
            c0 = random.randint(1, grid_w - outer_w - 1)
            bbox = (r0, c0, r0 + outer_h - 1, c0 + outer_w - 1)
            if any(_boxes_overlap(bbox, other, pad=2) for other in boxes):
                continue
            boxes.append(bbox)
            specs.append((bbox, gap_layout))
            placed = True
            break
        if not placed:
            return None

    for bbox, gap_layout in specs:
        _draw_region(grid, bbox, wall, bg, gap_layout)

    try:
        out = main(grid)
    except AssertionError:
        return None
    if np.array_equal(out, grid):
        return None
    return grid


def generate_input() -> np.ndarray:
    for _ in range(500):
        candidate = _gen_attempt()
        if candidate is not None:
            return candidate
    raise RuntimeError("failed to generate a valid sample")
