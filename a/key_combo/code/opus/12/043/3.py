from __future__ import annotations

from collections import Counter, deque
import random

import numpy as np


BLACK = 0
BLUE = 1
RED = 2
GREEN = 3
YELLOW = 4
GREY = 5
KEY_COLORS = (BLUE, RED, YELLOW)
PLUS_OFFSETS = ((0, 0), (-1, 0), (1, 0), (0, -1), (0, 1))

GEN_ATTEMPTS = 700
LAYOUT_ATTEMPTS = 240
LABEL_ATTEMPTS = 360
ROW_RANGE = (18, 25)
COL_RANGE = (21, 29)
LEGEND_H = 7
LEGEND_W_OPTIONS = (11, 13)
DIAGONAL_STEPS = (3, 4)

BASE_TEMPLATE = ((0, 0), (1, -1), (1, 1), (2, 0), (3, -1), (3, 1), (4, 0))
DROP_OPTIONS_BY_COUNT = {
    7: ((),),
    6: ((0,), (6,), (1,), (2,), (4,), (5,)),
    5: ((0, 6), (1, 2), (4, 5), (0, 4), (0, 5), (2, 6)),
}


def in_bounds(shape: tuple[int, int], r: int, c: int) -> bool:
    h, w = shape
    return 0 <= r < h and 0 <= c < w


def neighbors4(r: int, c: int):
    yield r - 1, c
    yield r + 1, c
    yield r, c - 1
    yield r, c + 1


def neighbors8(r: int, c: int):
    for dr in (-1, 0, 1):
        for dc in (-1, 0, 1):
            if dr == 0 and dc == 0:
                continue
            yield r + dr, c + dc


def bbox_coords(coords: np.ndarray) -> tuple[int, int, int, int]:
    r0, c0 = coords.min(axis=0)
    r1, c1 = coords.max(axis=0)
    return int(r0), int(c0), int(r1), int(c1)


def components(mask: np.ndarray, *, conn8: bool) -> list[np.ndarray]:
    h, w = mask.shape
    seen = np.zeros((h, w), dtype=bool)
    out: list[np.ndarray] = []
    neigh = neighbors8 if conn8 else neighbors4
    for r in range(h):
        for c in range(w):
            if not mask[r, c] or seen[r, c]:
                continue
            q = deque([(r, c)])
            seen[r, c] = True
            pts: list[tuple[int, int]] = []
            while q:
                cr, cc = q.popleft()
                pts.append((cr, cc))
                for nr, nc in neigh(cr, cc):
                    if in_bounds(mask.shape, nr, nc) and mask[nr, nc] and not seen[nr, nc]:
                        seen[nr, nc] = True
                        q.append((nr, nc))
            out.append(np.array(pts, dtype=int))
    return out


def plus_cells(center: tuple[int, int]) -> set[tuple[int, int]]:
    r, c = center
    return {(r + dr, c + dc) for dr, dc in PLUS_OFFSETS}


def find_legend(grid: np.ndarray) -> dict:
    candidates = []
    for comp in components(grid == GREY, conn8=False):
        if comp.shape[0] < 16:
            continue
        r0, c0, r1, c1 = bbox_coords(comp)
        h = r1 - r0 + 1
        w = c1 - c0 + 1
        if h < 5 or w < 5:
            continue
        if int(comp.shape[0]) != 2 * (h + w) - 4:
            continue
        sub = grid[r0 : r1 + 1, c0 : c1 + 1]
        if not (sub[0, :] == GREY).all():
            continue
        if not (sub[-1, :] == GREY).all():
            continue
        if not (sub[:, 0] == GREY).all():
            continue
        if not (sub[:, -1] == GREY).all():
            continue
        if (sub[1:-1, 1:-1] == GREY).any():
            continue

        key_col = None
        for rel_c in range(1, w - 1):
            if (sub[1:-1, rel_c] != BLACK).any():
                key_col = c0 + rel_c
                break
        if key_col is None:
            continue

        rows_by_color: dict[int, int] = {}
        for rr in range(r0 + 1, r1):
            color = int(grid[rr, key_col])
            if color != BLACK:
                rows_by_color[color] = rr
        if set(rows_by_color) != set(KEY_COLORS):
            continue
        candidates.append((h * w, r1, c1, r0, c0, r1, c1, key_col, rows_by_color))

    assert candidates, "legend box not found"
    _area, _r1, _c1, r0, c0, r1, c1, key_col, rows_by_color = max(candidates)
    return {
        "bbox": (r0, c0, r1, c1),
        "key_col": key_col,
        "rows_by_color": rows_by_color,
        "slots": len(range(key_col, c1, 2)),
    }


def find_pluses(grid: np.ndarray) -> list[dict]:
    h, w = grid.shape
    pluses = []
    seen_centers: set[tuple[int, int]] = set()
    for r in range(1, h - 1):
        for c in range(1, w - 1):
            center_color = int(grid[r, c])
            if center_color not in (GREEN, BLUE, RED, YELLOW):
                continue
            if int(grid[r - 1, c]) != GREEN or int(grid[r + 1, c]) != GREEN:
                continue
            if int(grid[r, c - 1]) != GREEN or int(grid[r, c + 1]) != GREEN:
                continue
            if any(int(grid[r + dr, c + dc]) != BLACK for dr in (-1, 1) for dc in (-1, 1)):
                continue
            center = (r, c)
            if center in seen_centers:
                continue
            seen_centers.add(center)
            pluses.append(
                {
                    "center": center,
                    "label_color": center_color if center_color in KEY_COLORS else None,
                    "pixels": plus_cells(center),
                }
            )

    pluses.sort(key=lambda item: item["center"])
    assert 5 <= len(pluses) <= 7, "expected 5-7 pluses"
    return pluses


def diagonal_connectors(grid: np.ndarray, pluses: list[dict]) -> tuple[list[tuple[int, int]], set[tuple[int, int]]]:
    pairs: list[tuple[int, int]] = []
    connector_cells: set[tuple[int, int]] = set()
    for i, plus_a in enumerate(pluses):
        r1, c1 = plus_a["center"]
        for j in range(i + 1, len(pluses)):
            r2, c2 = pluses[j]["center"]
            dr = r2 - r1
            dc = c2 - c1
            if abs(dr) != abs(dc):
                continue
            sr = 1 if dr > 0 else -1
            sc = 1 if dc > 0 else -1
            gap = [(r1 + k * sr, c1 + k * sc) for k in range(1, abs(dr))]
            if all(int(grid[r, c]) == BLACK for r, c in gap):
                pairs.append((i, j))
                connector_cells.update(gap)
    return pairs, connector_cells


def analyze_instance(grid: np.ndarray, *, use_connectors: bool = True) -> dict:
    g = np.asarray(grid, dtype=int)
    legend = find_legend(g)
    pluses = find_pluses(g)
    connector_pairs, connector_cells = diagonal_connectors(g, pluses) if use_connectors else ([], set())

    plus_pixel_owner: dict[tuple[int, int], int] = {}
    barrier = np.zeros_like(g, dtype=bool)
    for idx, plus in enumerate(pluses):
        for r, c in plus["pixels"]:
            plus_pixel_owner[(r, c)] = idx
            barrier[r, c] = True
    for r, c in connector_cells:
        barrier[r, c] = True
    r0, c0, r1, c1 = legend["bbox"]
    barrier[r0 : r1 + 1, c0 : c1 + 1] = True

    passable = (g == BLACK) & ~barrier
    seen = np.zeros_like(passable, dtype=bool)
    regions = []
    for r in range(g.shape[0]):
        for c in range(g.shape[1]):
            if not passable[r, c] or seen[r, c]:
                continue
            q = deque([(r, c)])
            seen[r, c] = True
            cells: list[tuple[int, int]] = []
            adjacent_plus_ids: set[int] = set()
            while q:
                cr, cc = q.popleft()
                cells.append((cr, cc))
                for nr, nc in neighbors4(cr, cc):
                    if in_bounds(g.shape, nr, nc) and passable[nr, nc] and not seen[nr, nc]:
                        seen[nr, nc] = True
                        q.append((nr, nc))
                for nr, nc in neighbors8(cr, cc):
                    owner = plus_pixel_owner.get((nr, nc))
                    if owner is not None and pluses[owner]["label_color"] is not None:
                        adjacent_plus_ids.add(owner)

            votes = Counter(pluses[idx]["label_color"] for idx in adjacent_plus_ids)
            tie = False
            winner = None
            if votes:
                max_count = max(votes.values())
                winners = [color for color, count in votes.items() if count == max_count]
                tie = len(winners) > 1
                winner = min(
                    winners,
                    key=lambda color: min(
                        pluses[idx]["center"] for idx in adjacent_plus_ids if pluses[idx]["label_color"] == color
                    ),
                )

            regions.append(
                {
                    "cells": cells,
                    "adjacent_plus_ids": adjacent_plus_ids,
                    "votes": votes,
                    "winner": winner,
                    "tie": tie,
                }
            )

    counts = Counter(region["winner"] for region in regions if region["winner"] is not None)
    return {
        "legend": legend,
        "pluses": pluses,
        "connector_pairs": connector_pairs,
        "connector_cells": connector_cells,
        "regions": regions,
        "counts": counts,
    }


def render_output(grid: np.ndarray, analysis: dict) -> np.ndarray:
    out = np.asarray(grid, dtype=int).copy()
    for r, c in analysis["connector_cells"]:
        out[r, c] = GREY

    r0, c0, r1, c1 = analysis["legend"]["bbox"]
    out[r0 + 1 : r1, c0 + 1 : c1] = BLACK
    key_col = analysis["legend"]["key_col"]
    for color in KEY_COLORS:
        row = analysis["legend"]["rows_by_color"][color]
        count = int(analysis["counts"].get(color, 0))
        for step in range(count):
            out[row, key_col + 2 * step] = color
    return out


def main(grid: np.ndarray) -> np.ndarray:
    """Draw diagonal plus-to-plus barriers, classify open regions by adjacent labeled pluses, and tally them in the legend.

    Input structure:
    - Black is the background.
    - The main field contains 5-7 standard 5-pixel green plus objects whose centers are either green or one of the legend colors.
    - A grey rectangular legend box sits in the lower-right area, with one blue, red, and yellow key pixel inside its leftmost keyed column.

    Rule:
    - For every pair of plus centers on the same 45-degree diagonal, fill the interior diagonal cells grey if and only if every interior cell is black.
    - Treat all plus pixels, labeled centers, newly drawn grey diagonal cells, and the entire legend box as barriers; flood-fill the remaining black pixels into 4-connected open regions.
    - For each region, collect the labeled pluses touched by the region's 8-neighborhood, vote by label color, and break color ties by the smallest adjacent plus-center row, then column.
    - Keep the field unchanged except for the grey connectors, and rewrite the legend so each key row shows one dot per region assigned to that key color.

    Key insight:
    - The diagonal connectors are not decoration: together with the plus bodies they create the topology of the open regions, and the labeled plus centers then determine which legend row each region contributes to.
    """
    analysis = analyze_instance(np.asarray(grid, dtype=int), use_connectors=True)
    return render_output(grid, analysis)


def draw_legend(grid: np.ndarray, bbox: tuple[int, int, int, int], row_order: tuple[int, int, int]) -> None:
    r0, c0, r1, c1 = bbox
    grid[r0 : r1 + 1, c0 : c1 + 1] = BLACK
    grid[r0, c0 : c1 + 1] = GREY
    grid[r1, c0 : c1 + 1] = GREY
    grid[r0 : r1 + 1, c0] = GREY
    grid[r0 : r1 + 1, c1] = GREY
    key_col = c0 + 1
    for rel_row, color in zip((1, 3, 5), row_order):
        grid[r0 + rel_row, key_col] = color


def draw_plus(grid: np.ndarray, center: tuple[int, int], label_color: int | None) -> None:
    r, c = center
    for dr, dc in PLUS_OFFSETS[1:]:
        grid[r + dr, c + dc] = GREEN
    grid[r, c] = GREEN if label_color is None else label_color


def center_ok(
    center: tuple[int, int],
    shape: tuple[int, int],
    legend_bbox: tuple[int, int, int, int],
    occupied: set[tuple[int, int]],
) -> bool:
    r0, c0, r1, c1 = legend_bbox
    for cell in plus_cells(center):
        if cell in occupied:
            return False
        r, c = cell
        if not in_bounds(shape, r, c):
            return False
        if r0 - 1 <= r <= r1 + 1 and c0 - 1 <= c <= c1 + 1:
            return False
    return True


def transform_template(
    points: tuple[tuple[int, int], ...],
    *,
    swap_axes: bool,
    flip_rows: bool,
    flip_cols: bool,
) -> list[tuple[int, int]]:
    out = []
    for r, c in points:
        rr, cc = (c, r) if swap_axes else (r, c)
        if flip_rows:
            rr = -rr
        if flip_cols:
            cc = -cc
        out.append((rr, cc))
    min_r = min(r for r, _ in out)
    min_c = min(c for _, c in out)
    return [(r - min_r, c - min_c) for r, c in out]


def sample_template_centers(
    shape: tuple[int, int],
    legend_bbox: tuple[int, int, int, int],
    rng: random.Random,
) -> list[tuple[int, int]] | None:
    max_center_col = legend_bbox[1] - 3
    max_center_row = shape[0] - 2
    if max_center_col < 2:
        return None

    for _ in range(LAYOUT_ATTEMPTS):
        target_count = rng.choice((5, 6, 7))
        drop_idxs = set(rng.choice(DROP_OPTIONS_BY_COUNT[target_count]))
        kept_points = tuple(point for idx, point in enumerate(BASE_TEMPLATE) if idx not in drop_idxs)

        transformed = transform_template(
            kept_points,
            swap_axes=bool(rng.randrange(2)),
            flip_rows=bool(rng.randrange(2)),
            flip_cols=bool(rng.randrange(2)),
        )
        step = rng.choice(DIAGONAL_STEPS)
        rel_points = [(r * step, c * step) for r, c in transformed]
        height = max(r for r, _ in rel_points)
        width = max(c for _, c in rel_points)
        if height > max_center_row - 1 or width > max_center_col - 1:
            continue

        top_r = rng.randint(1, max_center_row - height)
        left_c = rng.randint(1, max_center_col - width)
        centers = sorted((top_r + r, left_c + c) for r, c in rel_points)

        occupied: set[tuple[int, int]] = set()
        ok = True
        for center in centers:
            if not center_ok(center, shape, legend_bbox, occupied):
                ok = False
                break
            occupied.update(plus_cells(center))
        if ok:
            return centers
    return None


def build_grid(
    shape: tuple[int, int],
    legend_bbox: tuple[int, int, int, int],
    row_order: tuple[int, int, int],
    centers: list[tuple[int, int]],
    labels: list[int | None],
) -> np.ndarray:
    grid = np.zeros(shape, dtype=int)
    draw_legend(grid, legend_bbox, row_order)
    for center, label in zip(centers, labels):
        draw_plus(grid, center, label)
    return grid


def analyze_layout(
    shape: tuple[int, int],
    legend_bbox: tuple[int, int, int, int],
    row_order: tuple[int, int, int],
    centers: list[tuple[int, int]],
) -> dict:
    probe_grid = build_grid(shape, legend_bbox, row_order, centers, [BLUE] * len(centers))
    with_lines = analyze_instance(probe_grid, use_connectors=True)
    no_lines = analyze_instance(probe_grid, use_connectors=False)
    return {
        "analysis": with_lines,
        "no_line_analysis": no_lines,
        "region_adj": [set(region["adjacent_plus_ids"]) for region in with_lines["regions"]],
        "no_line_region_adj": [set(region["adjacent_plus_ids"]) for region in no_lines["regions"]],
    }


def choose_labels(centers: list[tuple[int, int]], rng: random.Random, *, want_unlabeled: bool) -> list[int | None]:
    n = len(centers)
    order = list(range(n))
    rng.shuffle(order)
    labels: list[int | None] = [None] * n

    # Seed all three colors first so the legend palette is always relevant.
    for idx, color in zip(order[:3], KEY_COLORS):
        labels[idx] = color

    extra = order[3:]
    unlabeled_indices: set[int] = set()
    if want_unlabeled and extra:
        unlabeled_indices = set(rng.sample(extra, rng.randint(1, min(2, len(extra)))))

    mirrored = sorted(range(n), key=lambda idx: (centers[idx][0], centers[idx][1]))
    if len(extra) >= 2:
        a = mirrored[0]
        b = mirrored[-1]
        if a != b and labels[a] is None and labels[b] is None and rng.random() < 0.55:
            color = rng.choice(KEY_COLORS)
            labels[a] = color
            labels[b] = color

    for idx in extra:
        if idx in unlabeled_indices:
            labels[idx] = None
        elif labels[idx] is None:
            labels[idx] = rng.choice(KEY_COLORS)

    if set(color for color in labels if color is not None) != set(KEY_COLORS):
        missing = [color for color in KEY_COLORS if color not in labels]
        candidates = [idx for idx in range(n) if idx not in unlabeled_indices]
        rng.shuffle(candidates)
        for color, idx in zip(missing, candidates):
            labels[idx] = color
    return labels


def assignment_stats(
    region_adj: list[set[int]],
    labels: list[int | None],
    centers: list[tuple[int, int]],
) -> tuple[list[dict], Counter[int]]:
    regions = []
    counts: Counter[int] = Counter()
    for adjacent in region_adj:
        votes = Counter(labels[idx] for idx in adjacent if labels[idx] is not None)
        if not votes:
            return [], Counter()
        max_count = max(votes.values())
        winners = [color for color, count in votes.items() if count == max_count]
        winner = min(
            winners,
            key=lambda color: min(centers[idx] for idx in adjacent if labels[idx] == color),
        )
        tie = len(winners) > 1
        counts[winner] += 1
        regions.append({"votes": votes, "winner": winner, "tie": tie, "adjacent_plus_ids": adjacent})
    return regions, counts


def labels_fit_layout(
    layout: dict,
    centers: list[tuple[int, int]],
    labels: list[int | None],
    *,
    want_tie: bool,
    want_unlabeled: bool,
) -> bool:
    if set(color for color in labels if color is not None) != set(KEY_COLORS):
        return False
    if want_unlabeled:
        if all(color is not None for color in labels):
            return False
    elif any(color is None for color in labels):
        return False

    regions, counts = assignment_stats(layout["region_adj"], labels, centers)
    if not regions:
        return False
    if any(int(counts.get(color, 0)) < 1 for color in KEY_COLORS):
        return False
    if not any(int(counts.get(color, 0)) > 1 for color in KEY_COLORS):
        return False
    if max(int(counts.get(color, 0)) for color in KEY_COLORS) > layout["analysis"]["legend"]["slots"]:
        return False
    if not any(len(region["votes"]) >= 2 for region in regions):
        return False
    if want_tie:
        if not any(region["tie"] for region in regions):
            return False
    elif not any((not region["tie"]) and len(region["votes"]) >= 2 for region in regions):
        return False

    _, no_line_counts = assignment_stats(layout["no_line_region_adj"], labels, centers)
    if len(layout["region_adj"]) == len(layout["no_line_region_adj"]) and no_line_counts == counts:
        return False
    return True


def has_blocked_aligned_pair(pluses: list[dict], connector_pairs: list[tuple[int, int]]) -> bool:
    connected = {tuple(sorted(pair)) for pair in connector_pairs}
    for i, plus_a in enumerate(pluses):
        r1, c1 = plus_a["center"]
        for j in range(i + 1, len(pluses)):
            r2, c2 = pluses[j]["center"]
            if abs(r2 - r1) == abs(c2 - c1) and (i, j) not in connected:
                return True
    return False


def validate_semantics(grid: np.ndarray, *, want_tie: bool, want_unlabeled: bool) -> bool:
    try:
        analysis = analyze_instance(grid, use_connectors=True)
        no_line = analyze_instance(grid, use_connectors=False)
        out = main(grid)
    except AssertionError:
        return False

    plus_labels = [plus["label_color"] for plus in analysis["pluses"]]
    if set(color for color in plus_labels if color is not None) != set(KEY_COLORS):
        return False
    if want_unlabeled and all(color is not None for color in plus_labels):
        return False
    if len(analysis["connector_pairs"]) < 2:
        return False
    if not has_blocked_aligned_pair(analysis["pluses"], analysis["connector_pairs"]):
        return False
    if len(analysis["regions"]) < 4:
        return False
    if any(not region["votes"] for region in analysis["regions"]):
        return False
    if not any(len(region["votes"]) >= 2 for region in analysis["regions"]):
        return False
    if want_tie:
        if not any(region["tie"] for region in analysis["regions"]):
            return False
    elif not any((not region["tie"]) and len(region["votes"]) >= 2 for region in analysis["regions"]):
        return False

    counts = analysis["counts"]
    if any(int(counts.get(color, 0)) < 1 for color in KEY_COLORS):
        return False
    if max(int(counts[color]) for color in KEY_COLORS) > analysis["legend"]["slots"]:
        return False
    if not any(int(counts.get(color, 0)) > 1 for color in KEY_COLORS):
        return False
    if len(no_line["regions"]) == len(analysis["regions"]) and no_line["counts"] == counts:
        return False

    legend = analysis["legend"]
    r0, c0, r1, c1 = legend["bbox"]
    if (out[r0, c0 : c1 + 1] != GREY).any():
        return False
    if (out[r1, c0 : c1 + 1] != GREY).any():
        return False
    if (out[r0 + 1 : r1, c0] != GREY).any():
        return False
    if (out[r0 + 1 : r1, c1] != GREY).any():
        return False
    if (out[r0 + 1 : r1, c0 + 1 : c1] == GREY).any():
        return False

    for color in KEY_COLORS:
        row = legend["rows_by_color"][color]
        count = int(counts[color])
        for offset in range(legend["slots"]):
            col = legend["key_col"] + 2 * offset
            if col >= c1:
                break
            expected = color if offset < count else BLACK
            if int(out[row, col]) != expected:
                return False
    return True


def generate_input() -> np.ndarray:
    rng = random.Random()
    for attempt in range(GEN_ATTEMPTS):
        h = rng.randint(*ROW_RANGE)
        w = rng.randint(*COL_RANGE)
        legend_w = rng.choice(LEGEND_W_OPTIONS)
        if legend_w > w - 3:
            continue
        legend_bbox = (h - LEGEND_H - 1, w - legend_w - 1, h - 2, w - 2)
        row_order = tuple(rng.sample(KEY_COLORS, len(KEY_COLORS)))

        centers = sample_template_centers((h, w), legend_bbox, rng)
        if centers is None:
            continue
        layout = analyze_layout((h, w), legend_bbox, row_order, centers)
        if len(layout["analysis"]["connector_pairs"]) < 2:
            continue
        if len(layout["analysis"]["regions"]) < 4:
            continue
        if not has_blocked_aligned_pair(layout["analysis"]["pluses"], layout["analysis"]["connector_pairs"]):
            continue

        want_tie = attempt % 2 == 0
        want_unlabeled = attempt % 3 == 0
        for _ in range(LABEL_ATTEMPTS):
            labels = choose_labels(centers, rng, want_unlabeled=want_unlabeled)
            if not labels_fit_layout(layout, centers, labels, want_tie=want_tie, want_unlabeled=want_unlabeled):
                continue
            grid = build_grid((h, w), legend_bbox, row_order, centers, labels)
            if validate_semantics(grid, want_tie=want_tie, want_unlabeled=want_unlabeled):
                return grid
    raise RuntimeError("failed to generate valid input")
