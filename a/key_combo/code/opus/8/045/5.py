from __future__ import annotations

import random
from typing import Iterable

import numpy as np


BLACK = 0
BLUE = 1
RED = 2
GREEN = 3
YELLOW = 4

SEED_COLORS = (BLUE, RED, GREEN)
PLUS_OFFSETS = ((0, 0), (-1, 0), (1, 0), (0, -1), (0, 1))
BOX_RING_OFFSETS = (
    (-1, -1),
    (-1, 0),
    (-1, 1),
    (0, -1),
    (0, 1),
    (1, -1),
    (1, 0),
    (1, 1),
)
BOX_BAND_OFFSETS = (
    (-2, -1),
    (-2, 0),
    (-2, 1),
    (-1, -2),
    (-1, 2),
    (0, -2),
    (0, 2),
    (1, -2),
    (1, 2),
    (2, -1),
    (2, 0),
    (2, 1),
)
NEIGHBORS8 = tuple(
    (dr, dc)
    for dr in (-1, 0, 1)
    for dc in (-1, 0, 1)
    if not (dr == 0 and dc == 0)
)

GRID_RANGE = (14, 17)
BOX_COUNT_CHOICES = (2, 3)
MIN_SEED_GAP = 5
MAX_GENERATION_ATTEMPTS = 400
MAX_COMBO_ATTEMPTS = 24


def _in_bounds(shape: tuple[int, int], r: int, c: int) -> bool:
    return 0 <= r < shape[0] and 0 <= c < shape[1]


def _chebyshev(a: tuple[int, int], b: tuple[int, int]) -> int:
    return max(abs(a[0] - b[0]), abs(a[1] - b[1]))


def _scan_seeds(grid: np.ndarray) -> list[dict]:
    seeds: list[dict] = []
    for color in SEED_COLORS:
        coords = np.argwhere(grid == color)
        assert len(coords) == 1, "expected exactly one seed of each color"
        r, c = map(int, coords[0])
        seeds.append({"color": int(color), "pos": (r, c)})
    return seeds


def _is_box_at(grid: np.ndarray, r: int, c: int) -> bool:
    if not (1 <= r < grid.shape[0] - 1 and 1 <= c < grid.shape[1] - 1):
        return False
    if int(grid[r, c]) != BLACK:
        return False
    return all(int(grid[r + dr, c + dc]) == YELLOW for dr, dc in BOX_RING_OFFSETS)


def _scan_boxes(grid: np.ndarray) -> list[tuple[int, int]]:
    centers = [(r, c) for r in range(1, grid.shape[0] - 1) for c in range(1, grid.shape[1] - 1) if _is_box_at(grid, r, c)]
    assert 2 <= len(centers) <= 3, "expected two or three hollow yellow boxes"
    return sorted(centers)


def _band_cells(shape: tuple[int, int], center: tuple[int, int]) -> list[tuple[int, int]]:
    r, c = center
    return [
        (r + dr, c + dc)
        for dr, dc in BOX_BAND_OFFSETS
        if _in_bounds(shape, r + dr, c + dc)
    ]


def _simulate_growth(
    shape: tuple[int, int],
    seeds: list[dict],
    obstacle_mask: np.ndarray,
    *,
    tie_mode: str,
) -> tuple[np.ndarray, list[int], dict]:
    h, w = shape
    owners = np.full((h, w), -1, dtype=int)
    frontiers: list[set[tuple[int, int]]] = []
    counts = [1] * len(seeds)
    seed_positions = [seed["pos"] for seed in seeds]

    for idx, seed in enumerate(seeds):
        r, c = seed["pos"]
        owners[r, c] = idx
        frontiers.append({(r, c)})

    stats = {"contested_total": 0, "contested_decisive": 0}

    while True:
        proposals: dict[tuple[int, int], list[int]] = {}
        for idx, frontier in enumerate(frontiers):
            for r, c in frontier:
                for dr, dc in NEIGHBORS8:
                    rr, cc = r + dr, c + dc
                    if not _in_bounds(shape, rr, cc):
                        continue
                    if obstacle_mask[rr, cc] or owners[rr, cc] != -1:
                        continue
                    proposals.setdefault((rr, cc), []).append(idx)

        if not proposals:
            break

        round_counts = counts.copy()
        next_frontiers = [set() for _ in seeds]
        for cell in sorted(proposals):
            contenders = proposals[cell]
            if len(contenders) == 1:
                winner = contenders[0]
            elif tie_mode == "pos_only":
                winner = min(contenders, key=lambda idx: seed_positions[idx])
            elif tie_mode == "count_pos":
                stats["contested_total"] += 1
                contender_counts = {round_counts[idx] for idx in contenders}
                if len(contender_counts) > 1:
                    stats["contested_decisive"] += 1
                winner = min(
                    contenders,
                    key=lambda idx: (round_counts[idx], seed_positions[idx]),
                )
            else:
                raise ValueError(f"unknown tie mode: {tie_mode}")

            rr, cc = cell
            owners[rr, cc] = winner
            next_frontiers[winner].add(cell)

        any_new = False
        for idx, frontier in enumerate(next_frontiers):
            counts[idx] += len(frontier)
            if frontier:
                any_new = True
        if not any_new:
            break
        frontiers = next_frontiers

    return owners, counts, stats


def _box_owner_sets(
    owners: np.ndarray, boxes: Iterable[tuple[int, int]]
) -> list[set[int]]:
    shape = owners.shape
    out: list[set[int]] = []
    for center in boxes:
        band = _band_cells(shape, center)
        owner_set = {
            int(owners[r, c]) for r, c in band if int(owners[r, c]) >= 0
        }
        out.append(owner_set)
    return out


def _render_output(
    grid: np.ndarray,
    seeds: list[dict],
    boxes: list[tuple[int, int]],
    owners: np.ndarray,
) -> np.ndarray:
    out = np.asarray(grid, dtype=int).copy()
    owner_sets = _box_owner_sets(owners, boxes)
    for (r, c), owner_set in zip(boxes, owner_sets):
        out[r - 1 : r + 2, c - 1 : c + 2] = BLACK
        if len(owner_set) == 1:
            winner = next(iter(owner_set))
            color = int(seeds[winner]["color"])
            for dr, dc in PLUS_OFFSETS:
                out[r + dr, c + dc] = color
    return out


def main(grid: np.ndarray) -> np.ndarray:
    """
    Grow three competing territories around yellow hollow boxes, then convert each box by the territory that surrounds it.

    Input structure:
    - A black grid contains exactly three single-cell seeds in blue, red, and green.
    - Two or three yellow 3x3 hollow boxes are present; each box has eight yellow border cells and a black center.
    - The yellow box borders are permanent obstacles during growth.

    Rule:
    - Starting from the three seeds simultaneously, expand one 8-neighbor step per round through non-yellow cells only.
    - Each newly reached cell can be claimed by only one seed; if multiple seeds reach it in the same round, give it to the seed with fewer already-claimed cells, breaking any remaining tie by the seed position (upper rows first, then left columns).
    - After growth stops, inspect the 4-neighborhood expansion band around each 3x3 box: if all claimed cells in that band belong to one seed, replace the box with that seed's plus sign; otherwise erase the whole 3x3 box to black.

    Key insight:
    - The yellow boxes are both barriers that reshape the competitive growth and targets whose fate is determined by the reshaped territories around them.
    """
    g = np.asarray(grid, dtype=int)
    assert g.ndim == 2, "expected a 2D grid"
    seeds = _scan_seeds(g)
    boxes = _scan_boxes(g)
    owners, _counts, _stats = _simulate_growth(g.shape, seeds, g == YELLOW, tie_mode="count_pos")
    return _render_output(g, seeds, boxes, owners)


def _seed_slots(h: int, w: int) -> list[list[tuple[int, int]]]:
    top_rows = range(1, max(2, h // 3 + 1))
    bottom_rows = range(max(h // 2 + 1, h - h // 3), h - 1)
    left_cols = range(1, max(2, w // 3 + 1))
    right_cols = range(max(w // 2 + 1, w - w // 3), w - 1)
    center_cols = range(max(1, w // 4), min(w - 1, 3 * w // 4))
    return [
        [(r, c) for r in top_rows for c in left_cols],
        [(r, c) for r in top_rows for c in right_cols],
        [(r, c) for r in bottom_rows for c in center_cols],
    ]


def _sample_seeds(h: int, w: int) -> list[dict] | None:
    slots = _seed_slots(h, w)
    colors = list(SEED_COLORS)
    random.shuffle(colors)
    for _ in range(80):
        positions = [random.choice(slot) for slot in slots]
        if min(_chebyshev(a, b) for i, a in enumerate(positions) for b in positions[i + 1 :]) < MIN_SEED_GAP:
            continue
        if len(set(positions)) != 3:
            continue
        return [
            {"color": int(color), "pos": pos}
            for color, pos in zip(colors, positions)
        ]
    return None


def _candidate_box_centers(h: int, w: int, seeds: list[dict]) -> list[tuple[int, int]]:
    seed_positions = [seed["pos"] for seed in seeds]
    centers = []
    for r in range(1, h - 1):
        for c in range(1, w - 1):
            if min(_chebyshev((r, c), seed_pos) for seed_pos in seed_positions) < 3:
                continue
            if any(abs(seed_pos[0] - r) <= 1 and abs(seed_pos[1] - c) <= 1 for seed_pos in seed_positions):
                continue
            centers.append((r, c))
    random.shuffle(centers)
    return centers


def _boxes_are_separate(boxes: list[tuple[int, int]]) -> bool:
    for i, a in enumerate(boxes):
        for b in boxes[i + 1 :]:
            if abs(a[0] - b[0]) < 4 and abs(a[1] - b[1]) < 4:
                return False
    return True


def _draw_input(h: int, w: int, seeds: list[dict], boxes: list[tuple[int, int]]) -> np.ndarray:
    grid = np.zeros((h, w), dtype=int)
    for seed in seeds:
        r, c = seed["pos"]
        grid[r, c] = int(seed["color"])
    for r, c in boxes:
        for dr, dc in BOX_RING_OFFSETS:
            grid[r + dr, c + dc] = YELLOW
    return grid


def _output_with_rule_variant(
    grid: np.ndarray,
    seeds: list[dict],
    boxes: list[tuple[int, int]],
    *,
    obstacle_mask: np.ndarray,
    tie_mode: str,
) -> tuple[np.ndarray, dict]:
    owners, counts, stats = _simulate_growth(grid.shape, seeds, obstacle_mask, tie_mode=tie_mode)
    analysis = {
        "owners": owners,
        "counts": counts,
        "stats": stats,
        "owner_sets": _box_owner_sets(owners, boxes),
    }
    return _render_output(grid, seeds, boxes, owners), analysis


def _valid_example(grid: np.ndarray, seeds: list[dict], boxes: list[tuple[int, int]]) -> bool:
    normal_out, normal = _output_with_rule_variant(
        grid,
        seeds,
        boxes,
        obstacle_mask=(grid == YELLOW),
        tie_mode="count_pos",
    )
    owner_sets = normal["owner_sets"]
    if any(len(owner_set) == 0 for owner_set in owner_sets):
        return False
    if not any(len(owner_set) == 1 for owner_set in owner_sets):
        return False
    if not any(len(owner_set) >= 2 for owner_set in owner_sets):
        return False
    if min(normal["counts"]) < 12:
        return False
    if normal["stats"]["contested_total"] == 0:
        return False
    if normal["stats"]["contested_decisive"] == 0:
        return False

    no_obstacle_out, _ = _output_with_rule_variant(
        grid,
        seeds,
        boxes,
        obstacle_mask=np.zeros_like(grid, dtype=bool),
        tie_mode="count_pos",
    )
    if np.array_equal(normal_out, no_obstacle_out):
        return False

    try:
        solved = main(grid)
    except Exception:
        return False
    return bool(np.array_equal(solved, normal_out))


def _pick_boxes(h: int, w: int, seeds: list[dict]) -> list[tuple[int, int]] | None:
    candidates = _candidate_box_centers(h, w, seeds)
    if len(candidates) < 6:
        return None

    base_grid = _draw_input(h, w, seeds, [])
    base_owners, _counts, _stats = _simulate_growth(
        base_grid.shape,
        seeds,
        np.zeros((h, w), dtype=bool),
        tie_mode="count_pos",
    )
    mixed_like = []
    single_like = []
    edge_like = []
    for center in candidates:
        owner_set = _box_owner_sets(base_owners, [center])[0]
        if len(owner_set) >= 2:
            mixed_like.append(center)
        if len(owner_set) == 1:
            single_like.append(center)
        if center[0] in (1, h - 2) or center[1] in (1, w - 2):
            edge_like.append(center)
    if not mixed_like or not single_like:
        return None

    random.shuffle(mixed_like)
    random.shuffle(single_like)
    random.shuffle(edge_like)
    mixed_like = mixed_like[:12]
    single_like = single_like[:12]

    for mixed_center in mixed_like:
        for single_center in single_like:
            if mixed_center == single_center:
                continue
            base_pair = [mixed_center, single_center]
            if not _boxes_are_separate(base_pair):
                continue

            grid = _draw_input(h, w, seeds, base_pair)
            if _valid_example(grid, seeds, base_pair):
                return sorted(base_pair)

            third_candidates = edge_like + candidates
            random.shuffle(third_candidates)
            tried = 0
            for third_center in third_candidates:
                if tried >= MAX_COMBO_ATTEMPTS:
                    break
                tried += 1
                if third_center in base_pair:
                    continue
                chosen = base_pair + [third_center]
                if not _boxes_are_separate(chosen):
                    continue
                grid = _draw_input(h, w, seeds, chosen)
                if _valid_example(grid, seeds, chosen):
                    return sorted(chosen)
    return None


def generate_input() -> np.ndarray:
    for _ in range(MAX_GENERATION_ATTEMPTS):
        h = random.randint(*GRID_RANGE)
        w = random.randint(*GRID_RANGE)
        seeds = _sample_seeds(h, w)
        if seeds is None:
            continue
        boxes = _pick_boxes(h, w, seeds)
        if boxes is None:
            continue
        return _draw_input(h, w, seeds, boxes)
    raise RuntimeError("failed to generate a valid competitive-growth ARC input")
