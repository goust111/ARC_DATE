from __future__ import annotations

from collections import deque
from fractions import Fraction
import random

import numpy as np


BG = 0
BLUE = 1
RED = 2
YELLOW = 4
GREY = 5
DIRS4 = [(-1, 0), (1, 0), (0, -1), (0, 1)]

# Generator configuration.
OBJECT_COUNT_CHOICES = [5, 6, 7]
OBJECT_COUNT_WEIGHTS = [0.35, 0.5, 0.15]
GRID_WIDTH_RANGE = (15, 18)
ROW_SLACK_RANGE = (0, 3)
DISTRACTOR_COUNT_CHOICES = [0, 1, 2]
DISTRACTOR_COUNT_WEIGHTS = [0.15, 0.5, 0.35]
MAX_GENERATION_ATTEMPTS = 600

SIDE_PATTERNS = {
    5: [
        ("L", "R", "R", "L", "L"),
        ("R", "L", "L", "R", "R"),
        ("L", "L", "R", "R", "L"),
        ("R", "R", "L", "L", "R"),
    ],
    6: [
        ("L", "R", "R", "L", "L", "R"),
        ("R", "L", "L", "R", "R", "L"),
        ("L", "L", "R", "R", "L", "R"),
        ("R", "R", "L", "L", "R", "L"),
    ],
    7: [
        ("L", "R", "R", "L", "L", "R", "R"),
        ("R", "L", "L", "R", "R", "L", "L"),
        ("L", "L", "R", "R", "L", "R", "L"),
        ("R", "R", "L", "L", "R", "L", "R"),
    ],
}

BASE_SHAPES = [
    ((0, 0), (1, 0), (1, 1)),
    ((0, 1), (1, 0), (1, 1), (1, 2)),
    ((0, 0), (0, 1), (1, 1), (1, 2)),
    ((0, 0), (0, 2), (1, 0), (1, 1), (1, 2)),
    ((0, 0), (0, 1), (1, 1), (1, 2), (1, 3)),
    ((0, 0), (0, 1), (0, 3), (1, 1), (1, 2), (1, 3)),
    ((0, 0), (0, 2), (0, 3), (1, 0), (1, 1), (1, 3)),
]


def _normalize(coords: np.ndarray | list[tuple[int, int]] | tuple[tuple[int, int], ...]) -> tuple[tuple[int, int], ...]:
    arr = np.array(coords, dtype=int)
    arr -= arr.min(axis=0)
    order = np.lexsort((arr[:, 1], arr[:, 0]))
    arr = arr[order]
    return tuple(tuple(map(int, rc)) for rc in arr.tolist())


def _shape_variants() -> list[dict]:
    variants: list[dict] = []
    for family_id, base in enumerate(BASE_SHAPES):
        seen = set()
        for flip_rows in (False, True):
            for flip_cols in (False, True):
                arr = np.array(base, dtype=int)
                if flip_rows:
                    arr[:, 0] *= -1
                if flip_cols:
                    arr[:, 1] *= -1
                norm = _normalize(arr)
                if norm in seen:
                    continue
                seen.add(norm)
                coords = np.array(norm, dtype=int)
                height = int(coords[:, 0].max()) + 1
                width = int(coords[:, 1].max()) + 1
                assert height == 2, "generator assumes two-row shapes"
                variants.append(
                    {
                        "family": family_id,
                        "coords": coords,
                        "area": int(coords.shape[0]),
                        "height": height,
                        "width": width,
                        "offset_sum": int(coords[:, 1].sum()),
                    }
                )
    return variants


SHAPE_VARIANTS = _shape_variants()


def _components(mask: np.ndarray) -> list[np.ndarray]:
    h, w = mask.shape
    seen = np.zeros_like(mask, dtype=bool)
    parts: list[np.ndarray] = []
    for r in range(h):
        for c in range(w):
            if not mask[r, c] or seen[r, c]:
                continue
            q: deque[tuple[int, int]] = deque([(r, c)])
            seen[r, c] = True
            cells = []
            while q:
                cr, cc = q.popleft()
                cells.append((cr, cc))
                for dr, dc in DIRS4:
                    nr, nc = cr + dr, cc + dc
                    if 0 <= nr < h and 0 <= nc < w and mask[nr, nc] and not seen[nr, nc]:
                        seen[nr, nc] = True
                        q.append((nr, nc))
            parts.append(np.array(cells, dtype=int))
    return parts


def _opposite_edge_color(color: int) -> int:
    assert color in (RED, BLUE)
    return BLUE if color == RED else RED


def _sprite_record(coords: np.ndarray, width: int) -> dict:
    area = int(coords.shape[0])
    col_sum = int(coords[:, 1].sum())
    left_num = col_sum
    right_num = area * (width - 1) - col_sum
    assert left_num != right_num, "centroid cannot be equidistant from both edge lines"

    min_col = int(coords[:, 1].min())
    max_col = int(coords[:, 1].max())
    assert 3 <= area <= 6

    bbox_center = Fraction(min_col + max_col, 2)
    cell_left = min_col
    cell_right = width - 1 - max_col
    bbox_left_num = min_col + max_col
    bbox_right_num = 2 * (width - 1) - bbox_left_num

    return {
        "coords": coords,
        "area": area,
        "nearest_color": RED if left_num < right_num else BLUE,
        "nearest_dist": Fraction(min(left_num, right_num), area),
        "centroid_col": Fraction(col_sum, area),
        "cell_nearest_color": RED if cell_left < cell_right else BLUE,
        "cell_nearest_dist": Fraction(min(cell_left, cell_right), 1),
        "bbox_nearest_color": RED if bbox_left_num < bbox_right_num else BLUE,
        "bbox_nearest_dist": min(bbox_center, Fraction(width - 1, 1) - bbox_center),
        "min_col": min_col,
        "max_col": max_col,
    }


def _parse_sprite_records(grid: np.ndarray) -> list[dict]:
    grey_components = _components(grid == GREY)
    assert 5 <= len(grey_components) <= 7, "expected 5-7 grey sprites"

    yellow_components = _components(grid == YELLOW)
    assert len(yellow_components) <= 2
    assert all(int(comp.shape[0]) == 1 for comp in yellow_components), "yellow distractors must be singletons"

    width = grid.shape[1]
    records = [_sprite_record(comp, width) for comp in grey_components]
    nearest_dists = [record["nearest_dist"] for record in records]
    assert len(set(nearest_dists)) == len(nearest_dists), "nearest centroid distances must be unique"
    assert all(record["cell_nearest_dist"] > 0 for record in records)
    return records


def _apply_rank_rule(grid: np.ndarray, records: list[dict], order_key: str, color_key: str, odd_uses_farthest: bool) -> np.ndarray:
    out = grid.copy()
    ordered = sorted(records, key=lambda record: (record[order_key], record["centroid_col"]))
    for rank, record in enumerate(ordered):
        near_color = int(record[color_key])
        color = near_color if (rank % 2 == 0 or not odd_uses_farthest) else _opposite_edge_color(near_color)
        coords = record["coords"]
        out[coords[:, 0], coords[:, 1]] = color
    return out


def main(grid: np.ndarray) -> np.ndarray:
    """Recolor grey sprites by edge-distance rank, alternating between nearest-edge and farthest-edge colors.

    Input structure:
    - A black background grid has a full red line in the first column and a full blue line in the last column.
    - Inside those edge lines are 5-7 separated grey 4-connected irregular sprites of size 3-6.
    - Optional yellow singleton distractors may appear in the interior.

    Rule:
    - Ignore the yellow singletons.
    - For each grey sprite, compute the centroid column and use it as the Manhattan distance to the red edge line; the distance to the blue edge line is the mirrored distance from the last column.
    - Sort the grey sprites by their smaller centroid-to-edge distance from near to far.
    - Recolor even ranks with the color of their nearer edge and odd ranks with the color of their farther edge, leaving the edge lines and distractors unchanged.

    Key insight:
    - The ordering key is not the sprite's left edge or any pixel-to-edge gap: it is the centroid's nearest edge distance, and the rank parity flips whether that nearest edge color is kept or inverted.
    """
    g = np.asarray(grid, dtype=int)
    assert g.ndim == 2, "expected a 2D grid"
    h, w = g.shape
    assert 5 <= h <= 30 and 5 <= w <= 30
    assert np.all(g[:, 0] == RED), "expected a full red left edge line"
    assert np.all(g[:, -1] == BLUE), "expected a full blue right edge line"
    assert np.all(~np.isin(g[:, 1:-1], [RED, BLUE])), "interior should not contain red or blue input pixels"

    allowed = {BG, BLUE, RED, YELLOW, GREY}
    assert set(map(int, np.unique(g))) <= allowed

    records = _parse_sprite_records(g)
    return _apply_rank_rule(g, records, "nearest_dist", "nearest_color", odd_uses_farthest=True)


def _choose_top_rows(n: int) -> list[int]:
    min_height = 2 * n + (n - 1)
    extra = random.randint(*ROW_SLACK_RANGE)
    height = min_height + extra
    slots = [0] * (n + 1)
    for _ in range(extra):
        slots[random.randrange(len(slots))] += 1

    tops = []
    row = slots[0]
    for i in range(n):
        tops.append(row)
        row += 2
        if i < n - 1:
            row += 1 + slots[i + 1]
    assert height <= 30
    return tops


def _candidate_maps(width: int) -> dict[str, dict[Fraction, list[dict]]]:
    maps: dict[str, dict[Fraction, list[dict]]] = {"L": {}, "R": {}}
    for variant in SHAPE_VARIANTS:
        max_col = width - 1 - variant["width"]
        for left_col in range(1, max_col + 1):
            left_num = variant["area"] * left_col + variant["offset_sum"]
            right_num = variant["area"] * (width - 1) - left_num
            if left_num == right_num:
                continue
            side = "L" if left_num < right_num else "R"
            dist = Fraction(min(left_num, right_num), variant["area"])
            maps[side].setdefault(dist, []).append(
                {
                    "family": variant["family"],
                    "coords": variant["coords"],
                    "area": variant["area"],
                    "left_col": left_col,
                    "nearest_dist": dist,
                    "nearest_color": RED if side == "L" else BLUE,
                    "centroid_col": Fraction(left_num, variant["area"]),
                }
            )
    return maps


def _distance_sequence(side_pattern: tuple[str, ...], maps: dict[str, dict[Fraction, list[dict]]]) -> list[Fraction] | None:
    ordered_dists = {side: sorted(maps[side]) for side in maps}

    def backtrack(rank: int, prev: Fraction) -> list[Fraction] | None:
        if rank == len(side_pattern):
            return []
        side = side_pattern[rank]
        options = [dist for dist in ordered_dists[side] if dist > prev]
        random.shuffle(options)
        options.sort()
        for dist in options:
            tail = backtrack(rank + 1, dist)
            if tail is not None:
                return [dist, *tail]
        return None

    return backtrack(0, Fraction(-1, 1))


def _choose_candidates(side_pattern: tuple[str, ...], dists: list[Fraction], maps: dict[str, dict[Fraction, list[dict]]]) -> list[dict]:
    chosen: list[dict] = []
    family_counts: dict[int, int] = {}
    for side, dist in zip(side_pattern, dists):
        options = maps[side][dist][:]
        random.shuffle(options)
        options.sort(key=lambda item: (family_counts.get(int(item["family"]), 0), int(item["left_col"])))
        pick = dict(options[0])
        family = int(pick["family"])
        family_counts[family] = family_counts.get(family, 0) + 1
        chosen.append(pick)
    return chosen


def _abs_sprite_records(candidates: list[dict], top_rows: list[int], width: int) -> list[dict]:
    records = []
    for top_row, candidate in zip(top_rows, candidates):
        rel = candidate["coords"]
        coords = rel + np.array([top_row, int(candidate["left_col"])], dtype=int)
        record = _sprite_record(coords, width)
        record["family"] = int(candidate["family"])
        records.append(record)
    return records


def _place_distractors(grid: np.ndarray, count: int) -> bool:
    if count == 0:
        return True
    h, w = grid.shape
    candidates = []
    for r in range(h):
        for c in range(1, w - 1):
            if grid[r, c] != BG:
                continue
            r0 = max(0, r - 1)
            r1 = min(h, r + 2)
            c0 = max(0, c - 1)
            c1 = min(w, c + 2)
            hood = grid[r0:r1, c0:c1].copy()
            hood[r - r0, c - c0] = BG
            if np.all(hood == BG):
                candidates.append((r, c))
    if len(candidates) < count:
        return False
    for r, c in random.sample(candidates, count):
        grid[r, c] = YELLOW
    return True


def _expected_output(grid: np.ndarray, records: list[dict]) -> np.ndarray:
    return _apply_rank_rule(grid, records, "nearest_dist", "nearest_color", odd_uses_farthest=True)


def _semantic_counterexamples(grid: np.ndarray, records: list[dict], expected: np.ndarray) -> bool:
    ordered = sorted(records, key=lambda record: record["nearest_dist"])
    sides_even = {ordered[i]["nearest_color"] for i in range(0, len(ordered), 2)}
    sides_odd = {ordered[i]["nearest_color"] for i in range(1, len(ordered), 2)}
    if sides_even != {RED, BLUE} or sides_odd != {RED, BLUE}:
        return False

    centroid_order = [record["nearest_dist"] for record in sorted(records, key=lambda record: record["centroid_col"])]
    if centroid_order == [record["nearest_dist"] for record in ordered]:
        return False

    if len({int(record["family"]) for record in records}) < 3:
        return False

    if any(record["min_col"] == grid.shape[1] - 1 - record["max_col"] for record in records):
        return False
    if any(record["min_col"] + record["max_col"] == grid.shape[1] - 1 for record in records):
        return False

    wrong_always_near = _apply_rank_rule(grid, records, "nearest_dist", "nearest_color", odd_uses_farthest=False)
    wrong_left_to_right = _apply_rank_rule(grid, records, "centroid_col", "nearest_color", odd_uses_farthest=True)
    wrong_cell_frame = _apply_rank_rule(grid, records, "cell_nearest_dist", "cell_nearest_color", odd_uses_farthest=True)
    wrong_bbox_frame = _apply_rank_rule(grid, records, "bbox_nearest_dist", "bbox_nearest_color", odd_uses_farthest=True)
    if np.array_equal(expected, wrong_always_near):
        return False
    if np.array_equal(expected, wrong_left_to_right):
        return False
    if np.array_equal(expected, wrong_cell_frame):
        return False
    if np.array_equal(expected, wrong_bbox_frame):
        return False
    return True


def _gen_attempt() -> np.ndarray | None:
    n = random.choices(OBJECT_COUNT_CHOICES, weights=OBJECT_COUNT_WEIGHTS, k=1)[0]
    width = random.randint(*GRID_WIDTH_RANGE)
    top_rows = _choose_top_rows(n)
    height = top_rows[-1] + 2
    maps = _candidate_maps(width)

    patterns = SIDE_PATTERNS[n][:]
    random.shuffle(patterns)
    for side_pattern in patterns:
        dists = _distance_sequence(side_pattern, maps)
        if dists is None or len(set(dists)) != n:
            continue
        candidates = _choose_candidates(side_pattern, dists, maps)
        random.shuffle(top_rows)
        records = _abs_sprite_records(candidates, top_rows, width)

        grid = np.full((height, width), BG, dtype=int)
        grid[:, 0] = RED
        grid[:, -1] = BLUE
        for record in records:
            coords = record["coords"]
            grid[coords[:, 0], coords[:, 1]] = GREY

        distractor_count = random.choices(DISTRACTOR_COUNT_CHOICES, weights=DISTRACTOR_COUNT_WEIGHTS, k=1)[0]
        if not _place_distractors(grid, distractor_count):
            continue

        expected = _expected_output(grid, records)
        if not _semantic_counterexamples(grid, records, expected):
            continue

        try:
            actual = main(grid)
        except Exception:
            continue
        if np.array_equal(actual, expected):
            return grid
    return None


def generate_input() -> np.ndarray:
    for _ in range(MAX_GENERATION_ATTEMPTS):
        grid = _gen_attempt()
        if grid is not None:
            return grid
    raise RuntimeError("failed to generate a valid input for puzzle 6")
