from __future__ import annotations

import random
import sys
from collections import Counter
from pathlib import Path

import numpy as np

UTILS_DIR = Path("/home/lixinyu/snap/new/solvers")
if str(UTILS_DIR) not in sys.path:
    sys.path.append(str(UTILS_DIR))

from arc_utils import Color, bbox


BG = Color.BLACK
BLUE = Color.BLUE
RED = Color.RED
GREEN = Color.GREEN
YELLOW = Color.YELLOW
GREY = Color.GREY
ORANGE = Color.ORANGE
PATTERN_COLORS = (BLUE, RED, GREEN, YELLOW)
GEN_RETRIES = 500


def _orange_lines(grid: np.ndarray) -> tuple[list[int], list[int]]:
    rows = [r for r in range(grid.shape[0]) if np.all(grid[r, :] == ORANGE)]
    cols = [c for c in range(grid.shape[1]) if np.all(grid[:, c] == ORANGE)]
    assert len(rows) >= 2 and len(cols) >= 2, "expected at least a 3x3 region layout"
    return rows, cols


def _intervals(lines: list[int], length: int) -> list[tuple[int, int]]:
    pts = [-1, *sorted(map(int, lines)), int(length)]
    spans = []
    for left, right in zip(pts, pts[1:]):
        if left + 1 <= right - 1:
            spans.append((left + 1, right - 1))
    return spans


def _region_id(r: int, c: int, orange_rows: list[int], orange_cols: list[int]) -> tuple[int, int] | None:
    if r in orange_rows or c in orange_cols:
        return None
    rr = sum(line < r for line in orange_rows)
    cc = sum(line < c for line in orange_cols)
    return rr, cc


def _gray_bbox(grid: np.ndarray) -> tuple[int, int, int, int]:
    grey_box = bbox(grid == GREY)
    assert grey_box is not None, "expected one grey occlusion rectangle"
    return tuple(map(int, grey_box))


def _restore_grid(grid: np.ndarray) -> tuple[np.ndarray, tuple[int, int, int, int], list[int], list[int]]:
    g = np.asarray(grid, dtype=int)
    H, W = g.shape
    orange_rows, orange_cols = _orange_lines(g)
    grey_box = _gray_bbox(g)
    r0, c0, r1, c1 = grey_box

    for rr in range(r0, r1 + 1):
        for cc in range(c0, c1 + 1):
            if int(g[rr, cc]) == GREY:
                continue
            assert rr in orange_rows or cc in orange_cols, "grey bbox may only contain orange-line holes"

    out = g.copy()
    for rr, cc in np.argwhere(g == GREY):
        rr = int(rr)
        cc = int(cc)
        if rr in orange_rows or cc in orange_cols:
            continue
        partners = ((rr, W - 1 - cc), (H - 1 - rr, cc), (H - 1 - rr, W - 1 - cc))
        for pr, pc in partners:
            color = int(g[pr, pc])
            if color not in (GREY, ORANGE):
                out[rr, cc] = color
                break
        else:
            raise AssertionError("every occluded pixel must have a visible non-orange mirror partner")

    assert not np.any(out == GREY), "restoration must eliminate grey occlusion pixels"
    return out, grey_box, orange_rows, orange_cols


def _region_counts(
    grid: np.ndarray,
    orange_rows: list[int],
    orange_cols: list[int],
    *,
    ignore_grey: bool,
) -> tuple[dict[tuple[int, int], Counter], list[tuple[int, int]], list[tuple[int, int]]]:
    row_spans = _intervals(orange_rows, grid.shape[0])
    col_spans = _intervals(orange_cols, grid.shape[1])
    counts: dict[tuple[int, int], Counter] = {}
    for ri, (r0, r1) in enumerate(row_spans):
        for ci, (c0, c1) in enumerate(col_spans):
            patch = grid[r0 : r1 + 1, c0 : c1 + 1]
            valid = (patch != BG) & (patch != ORANGE)
            if ignore_grey:
                valid &= patch != GREY
            colors = [int(v) for v in patch[valid].flat if int(v) in PATTERN_COLORS]
            counts[(ri, ci)] = Counter(colors)
    return counts, row_spans, col_spans


def _dominant(counter: Counter) -> tuple[int | None, bool]:
    if not counter:
        return None, False
    best = counter.most_common()
    top = best[0][1]
    winners = [color for color, count in best if count == top]
    if len(winners) != 1:
        return None, False
    return int(winners[0]), True


def _compress_regions(restored: np.ndarray, orange_rows: list[int], orange_cols: list[int]) -> np.ndarray:
    counts, row_spans, col_spans = _region_counts(restored, orange_rows, orange_cols, ignore_grey=False)
    out = np.zeros((len(row_spans), len(col_spans)), dtype=int)
    for region, counter in counts.items():
        color, unique = _dominant(counter)
        assert unique and color is not None, "each region must have one unique dominant non-orange color"
        out[region] = color
    return out


def main(grid: np.ndarray) -> np.ndarray:
    """
    Restore a grey-occluded doubly mirrored pattern, then compress the orange-separated regions to their dominant colors.

    Input structure:
    - The background is black on an odd-sized grid.
    - Blue, red, green, and yellow single pixels form a sparse pattern whose non-orange content is mirrored across both the horizontal and vertical center axes.
    - A solid grey rectangle hides part of that pattern, while complete orange rows and columns stay visible and partition the canvas into rectangular regions.

    Rule:
    - Detect the grey rectangle by taking the bounding box of the grey pixels, treating orange cells inside that box as line holes rather than hidden cells.
    - For every non-orange grey cell at (r, c), test the horizontal mirror (r, W-1-c), then the vertical mirror (H-1-r, c), then the 180-degree partner (H-1-r, W-1-c), and copy the first partner color that is visible and not orange.
    - Detect all complete orange rows and columns, use adjacent lines and grid borders to define the rectangular regions, and count the non-black non-orange colors inside each region after restoration.
    - Output one pixel per region, colored by that region's unique dominant color.

    Key insight:
    - The orange partitioning step is only meaningful after pixel-level mirror repair, because the occluder removes enough evidence to change some regions' dominant colors.
    """
    restored, _grey_box, orange_rows, orange_cols = _restore_grid(grid)
    return _compress_regions(restored, orange_rows, orange_cols)


def _sample_lines(length: int, rng: random.Random) -> list[int] | None:
    region_count = rng.choice((3, 4))
    min_span = 2
    slack = length - (region_count - 1) - region_count * min_span
    if slack < 0:
        return None
    spans = [min_span] * region_count
    for _ in range(slack):
        spans[rng.randrange(region_count)] += 1
    rng.shuffle(spans)

    lines: list[int] = []
    pos = spans[0]
    for span in spans[1:]:
        lines.append(pos)
        pos += 1 + span
    return lines


def _sample_rect(
    H: int,
    W: int,
    orange_rows: list[int],
    orange_cols: list[int],
    rng: random.Random,
) -> tuple[int, int, int, int] | None:
    max_h = min(8, H - 2)
    max_w = min(8, W - 2)
    for _ in range(80):
        rect_h = rng.randint(4, max_h)
        rect_w = rng.randint(4, max_w)
        r0 = rng.randint(1, H - rect_h - 1)
        c0 = rng.randint(1, W - rect_w - 1)
        r1 = r0 + rect_h - 1
        c1 = c0 + rect_w - 1
        crosses_line = any(r0 <= rr <= r1 for rr in orange_rows) or any(c0 <= cc <= c1 for cc in orange_cols)
        if crosses_line and _rect_is_recoverable(H, W, orange_rows, orange_cols, (r0, c0, r1, c1)):
            return r0, c0, r1, c1
    return None


def _orbit_cells(H: int, W: int, r: int, c: int) -> tuple[tuple[int, int], ...]:
    return ((r, c), (r, W - 1 - c), (H - 1 - r, c), (H - 1 - r, W - 1 - c))


def _in_rect(rect: tuple[int, int, int, int], r: int, c: int) -> bool:
    r0, c0, r1, c1 = rect
    return r0 <= r <= r1 and c0 <= c <= c1


def _rect_is_recoverable(
    H: int,
    W: int,
    orange_rows: list[int],
    orange_cols: list[int],
    rect: tuple[int, int, int, int],
) -> bool:
    r0, c0, r1, c1 = rect
    for rr in range(r0, r1 + 1):
        for cc in range(c0, c1 + 1):
            if rr in orange_rows or cc in orange_cols:
                continue
            partners = ((rr, W - 1 - cc), (H - 1 - rr, cc), (H - 1 - rr, W - 1 - cc))
            if not any(
                (not _in_rect(rect, pr, pc)) and pr not in orange_rows and pc not in orange_cols
                for pr, pc in partners
            ):
                return False
    return True


def _build_candidates(
    H: int,
    W: int,
    orange_rows: list[int],
    orange_cols: list[int],
    rect: tuple[int, int, int, int],
) -> list[dict]:
    candidates = []
    for r in range(H // 2):
        for c in range(W // 2):
            cells = _orbit_cells(H, W, r, c)
            if any(rr in orange_rows or cc in orange_cols for rr, cc in cells):
                continue

            hidden = tuple(_in_rect(rect, rr, cc) for rr, cc in cells)
            if sum(hidden) == 4:
                continue
            recoverable = True
            for (rr, cc), is_hidden in zip(cells, hidden):
                if not is_hidden:
                    continue
                partners = ((rr, W - 1 - cc), (H - 1 - rr, cc), (H - 1 - rr, W - 1 - cc))
                if not any(
                    (not _in_rect(rect, pr, pc)) and pr not in orange_rows and pc not in orange_cols
                    for pr, pc in partners
                ):
                    recoverable = False
                    break
            if not recoverable:
                continue

            region_counts: Counter = Counter()
            hidden_counts: Counter = Counter()
            for (rr, cc), is_hidden in zip(cells, hidden):
                region = _region_id(rr, cc, orange_rows, orange_cols)
                assert region is not None
                region_counts[region] += 1
                if is_hidden:
                    hidden_counts[region] += 1

            candidates.append(
                {
                    "cells": cells,
                    "hidden": hidden,
                    "hidden_total": int(sum(hidden)),
                    "region_counts": region_counts,
                    "hidden_counts": hidden_counts,
                }
            )
    return candidates


def _paint_orbit(grid: np.ndarray, orbit: dict, color: int) -> None:
    for rr, cc in orbit["cells"]:
        grid[rr, cc] = int(color)


def _candidate_regions(candidates: list[dict]) -> list[tuple[int, int]]:
    regions = set()
    for orbit in candidates:
        regions.update(orbit["region_counts"].keys())
    return sorted(regions)


def _pick_best_candidate(
    remaining: list[dict],
    predicate,
    score_fn,
) -> dict | None:
    scored = []
    for orbit in remaining:
        if predicate(orbit):
            scored.append((score_fn(orbit), random.random(), orbit))
    if not scored:
        return None
    scored.sort(key=lambda item: (item[0], item[1]), reverse=True)
    return scored[0][2]


def _render_pattern(
    shape: tuple[int, int],
    orange_rows: list[int],
    orange_cols: list[int],
    assignments: list[tuple[dict, int]],
) -> np.ndarray:
    grid = np.zeros(shape, dtype=int)
    grid[orange_rows, :] = ORANGE
    grid[:, orange_cols] = ORANGE
    for orbit, color in assignments:
        _paint_orbit(grid, orbit, color)
    return grid


def _occlude(grid: np.ndarray, rect: tuple[int, int, int, int]) -> np.ndarray:
    out = grid.copy()
    r0, c0, r1, c1 = rect
    for rr in range(r0, r1 + 1):
        for cc in range(c0, c1 + 1):
            if int(out[rr, cc]) != ORANGE:
                out[rr, cc] = GREY
    return out


def _validate_instance(inp: np.ndarray) -> bool:
    if inp.ndim != 2 or max(inp.shape) > 30:
        return False
    if not set(map(int, np.unique(inp))).issubset({BG, BLUE, RED, GREEN, YELLOW, GREY, ORANGE}):
        return False

    try:
        restored, rect, orange_rows, orange_cols = _restore_grid(inp)
    except AssertionError:
        return False

    r0, c0, r1, c1 = rect
    if not (any(r0 <= rr <= r1 for rr in orange_rows) or any(c0 <= cc <= c1 for cc in orange_cols)):
        return False

    try:
        out = main(inp)
    except AssertionError:
        return False

    expected = _compress_regions(restored, orange_rows, orange_cols)
    if not np.array_equal(out, expected):
        return False

    restored_counts, row_spans, col_spans = _region_counts(restored, orange_rows, orange_cols, ignore_grey=False)
    visible_counts, _row_spans2, _col_spans2 = _region_counts(inp, orange_rows, orange_cols, ignore_grey=True)
    if out.shape != (len(row_spans), len(col_spans)):
        return False

    hidden_colored = (inp == GREY) & (restored != BG)
    if int(hidden_colored.sum()) < 4:
        return False
    hidden_regions = {
        _region_id(int(r), int(c), orange_rows, orange_cols)
        for r, c in np.argwhere(hidden_colored)
    }
    if len(hidden_regions) < 2:
        return False

    changed_or_ambiguous = 0
    changed_unique = 0
    has_competition = False
    reps = []
    for region, counter in restored_counts.items():
        rep, unique = _dominant(counter)
        if not unique or rep is None:
            return False
        reps.append(rep)
        if sum(count > 0 for count in counter.values()) >= 2:
            has_competition = True

        visible_total = sum(visible_counts[region].values())
        if visible_total == 0:
            return False
        visible_rep, visible_unique = _dominant(visible_counts[region])
        if (not visible_unique) or visible_rep != rep:
            changed_or_ambiguous += 1
        if visible_unique and visible_rep != rep:
            changed_unique += 1

    if len(set(reps)) < 2:
        return False
    if not has_competition:
        return False
    if changed_or_ambiguous < 2 or changed_unique < 1:
        return False
    return True


def _instance_penalty(inp: np.ndarray) -> int:
    try:
        restored, rect, orange_rows, orange_cols = _restore_grid(inp)
    except AssertionError:
        return 1000

    r0, c0, r1, c1 = rect
    penalty = 0
    if not (any(r0 <= rr <= r1 for rr in orange_rows) or any(c0 <= cc <= c1 for cc in orange_cols)):
        penalty += 20

    restored_counts, _row_spans, _col_spans = _region_counts(restored, orange_rows, orange_cols, ignore_grey=False)
    visible_counts, _row_spans2, _col_spans2 = _region_counts(inp, orange_rows, orange_cols, ignore_grey=True)

    hidden_colored = (inp == GREY) & (restored != BG)
    hidden_regions = {
        _region_id(int(r), int(c), orange_rows, orange_cols)
        for r, c in np.argwhere(hidden_colored)
    }
    penalty += max(0, 4 - int(hidden_colored.sum()))
    penalty += 3 * max(0, 2 - len(hidden_regions))

    changed_or_ambiguous = 0
    changed_unique = 0
    has_competition = False
    reps = []
    for region, counter in restored_counts.items():
        rep, unique = _dominant(counter)
        if not unique or rep is None:
            penalty += 8
            continue
        reps.append(rep)
        if sum(count > 0 for count in counter.values()) >= 2:
            has_competition = True

        visible_total = sum(visible_counts[region].values())
        if visible_total == 0:
            penalty += 1
            continue

        visible_rep, visible_unique = _dominant(visible_counts[region])
        if (not visible_unique) or visible_rep != rep:
            changed_or_ambiguous += 1
        if visible_unique and visible_rep != rep:
            changed_unique += 1

    if len(set(reps)) < 2:
        penalty += 4
    if not has_competition:
        penalty += 2
    penalty += 3 * max(0, 2 - changed_or_ambiguous)
    if changed_unique < 1:
        penalty += 4
    return penalty


def _search_input(
    H: int,
    W: int,
    orange_rows: list[int],
    orange_cols: list[int],
    rect: tuple[int, int, int, int],
    pool: list[dict],
    rng: random.Random,
) -> np.ndarray | None:
    palette = (0, *PATTERN_COLORS)
    regions = _candidate_regions(pool)

    def render(colors: list[int]) -> np.ndarray:
        assignments = [(orbit, color) for orbit, color in zip(pool, colors) if color != 0]
        return _occlude(_render_pattern((H, W), orange_rows, orange_cols, assignments), rect)

    def initial_colors() -> list[int]:
        targets = {region: rng.choice(PATTERN_COLORS) for region in regions}
        colors: list[int] = []
        for orbit in pool:
            scored = []
            for color in PATTERN_COLORS:
                score = 0
                for region, amount in orbit["region_counts"].items():
                    if targets[region] == color:
                        score += 2 * int(amount) + int(orbit["hidden_counts"].get(region, 0))
                scored.append((score, rng.random(), color))

            best_score, _jitter, best_color = max(scored)
            if best_score <= 0 and rng.random() < 0.45:
                color = 0
            elif rng.random() < 0.75:
                color = best_color
            else:
                color = rng.choice(PATTERN_COLORS)
            colors.append(color)

        hidden_indices = [idx for idx, orbit in enumerate(pool) if orbit["hidden_total"] > 0]
        rng.shuffle(hidden_indices)
        for idx in hidden_indices[:4]:
            if colors[idx] == 0:
                colors[idx] = rng.choice(PATTERN_COLORS)
        return colors

    best_penalty = 1000
    best_input: np.ndarray | None = None
    for _restart in range(6):
        colors = initial_colors()
        current_input = render(colors)
        current_penalty = _instance_penalty(current_input)
        if current_penalty < best_penalty:
            best_penalty = current_penalty
            best_input = current_input
        if current_penalty == 0:
            return current_input

        for _step in range(12):
            improved = False
            order = list(range(len(pool)))
            rng.shuffle(order)
            for idx in order:
                base_color = colors[idx]
                options = list(palette)
                rng.shuffle(options)
                for color in options:
                    if color == base_color:
                        continue
                    colors[idx] = color
                    candidate_input = render(colors)
                    candidate_penalty = _instance_penalty(candidate_input)
                    if candidate_penalty < current_penalty:
                        current_penalty = candidate_penalty
                        current_input = candidate_input
                        improved = True
                        if candidate_penalty < best_penalty:
                            best_penalty = candidate_penalty
                            best_input = candidate_input
                        break
                else:
                    colors[idx] = base_color
                    continue

                if current_penalty == 0:
                    return current_input

            if improved:
                continue

            for _ in range(2):
                idx = rng.randrange(len(colors))
                colors[idx] = rng.choice(palette)
            current_input = render(colors)
            current_penalty = _instance_penalty(current_input)
            if current_penalty < best_penalty:
                best_penalty = current_penalty
                best_input = current_input
            if current_penalty == 0:
                return current_input

    if best_input is not None and best_penalty == 0:
        return best_input
    return None


def _cells_for_region(
    row_spans: list[tuple[int, int]],
    col_spans: list[tuple[int, int]],
    region: tuple[int, int],
    rect: tuple[int, int, int, int],
    *,
    hidden: bool,
) -> list[tuple[int, int]]:
    ri, ci = region
    r0, r1 = row_spans[ri]
    c0, c1 = col_spans[ci]
    cells = []
    for rr in range(r0, r1 + 1):
        for cc in range(c0, c1 + 1):
            in_rect = _in_rect(rect, rr, cc)
            if bool(in_rect) == bool(hidden):
                cells.append((rr, cc))
    return cells


def _paint_orbit(
    grid: np.ndarray,
    H: int,
    W: int,
    orange_rows: list[int],
    orange_cols: list[int],
    r: int,
    c: int,
    color: int,
) -> None:
    for rr, cc in _orbit_cells(H, W, int(r), int(c)):
        if rr not in orange_rows and cc not in orange_cols:
            grid[rr, cc] = int(color)


def _paint_many(
    grid: np.ndarray,
    H: int,
    W: int,
    orange_rows: list[int],
    orange_cols: list[int],
    cells: list[tuple[int, int]],
    color: int,
    count: int,
    rng: random.Random,
) -> None:
    if not cells:
        return
    sample = rng.sample(cells, min(int(count), len(cells)))
    for rr, cc in sample:
        _paint_orbit(grid, H, W, orange_rows, orange_cols, rr, cc, color)


def _gen_attempt() -> np.ndarray | None:
    rng = random.Random()
    H = rng.choice((17, 19, 21))
    W = rng.choice((17, 19, 21))

    top_line = rng.randint(4, H // 2 - 3)
    left_line = rng.randint(4, W // 2 - 3)
    orange_rows = [top_line, H - 1 - top_line]
    orange_cols = [left_line, W - 1 - left_line]
    row_spans = _intervals(orange_rows, H)
    col_spans = _intervals(orange_cols, W)
    if len(row_spans) != 3 or len(col_spans) != 3:
        return None

    rect = (
        max(1, top_line - 3),
        max(1, left_line - 3),
        min(H - 2, top_line + 3),
        min(W - 2, left_line + 3),
    )
    if not _rect_is_recoverable(H, W, orange_rows, orange_cols, rect):
        return None

    full = np.zeros((H, W), dtype=int)
    full[orange_rows, :] = ORANGE
    full[:, orange_cols] = ORANGE

    affected = [(0, 0), (0, 1), (1, 0), (1, 1)]
    color_cycle = list(PATTERN_COLORS)
    rng.shuffle(color_cycle)
    affected_competitors: dict[tuple[int, int], int] = {}
    for idx, region in enumerate(affected):
        hidden_cells = _cells_for_region(row_spans, col_spans, region, rect, hidden=True)
        visible_cells = _cells_for_region(row_spans, col_spans, region, rect, hidden=False)
        if len(hidden_cells) < 3 or len(visible_cells) < 2:
            return None
        dominant = color_cycle[idx % len(color_cycle)]
        competitor = color_cycle[(idx + 1) % len(color_cycle)]
        affected_competitors[region] = competitor
        _paint_many(full, H, W, orange_rows, orange_cols, hidden_cells, dominant, 5, rng)
        _paint_many(full, H, W, orange_rows, orange_cols, visible_cells, competitor, 4, rng)

    for ri in range(len(row_spans)):
        for ci in range(len(col_spans)):
            region = (ri, ci)
            visible_cells = _cells_for_region(row_spans, col_spans, region, rect, hidden=False)
            if not visible_cells:
                continue
            counts, _rs, _cs = _region_counts(full, orange_rows, orange_cols, ignore_grey=False)
            current, unique = _dominant(counts[region])
            fill_color = int(current) if unique and current is not None else rng.choice(PATTERN_COLORS)
            _paint_many(full, H, W, orange_rows, orange_cols, visible_cells, fill_color, 4, rng)

    for region, competitor in affected_competitors.items():
        visible_cells = _cells_for_region(row_spans, col_spans, region, rect, hidden=False)
        _paint_many(full, H, W, orange_rows, orange_cols, visible_cells, competitor, 7, rng)

    inp = _occlude(full, rect)
    if not _validate_instance(inp):
        return None
    return inp


def generate_input() -> np.ndarray:
    for _ in range(GEN_RETRIES):
        grid = _gen_attempt()
        if grid is not None:
            return grid
    raise RuntimeError("failed to generate a valid mirrored-occlusion partition instance")
