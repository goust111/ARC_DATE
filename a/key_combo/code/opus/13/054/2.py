from __future__ import annotations

from collections import deque
import random

import numpy as np


BACKGROUND = 0
INPUT_COLOR = 5
OUTPUT_PALETTE = (1, 2, 3, 4)
NEIGHBORS4 = ((1, 0), (-1, 0), (0, 1), (0, -1))
GEN_ATTEMPTS = 400
MASK_ATTEMPTS = 250
PLACEMENT_ATTEMPTS = 160


def _components(mask: np.ndarray) -> list[np.ndarray]:
    h, w = mask.shape
    seen = np.zeros((h, w), dtype=bool)
    comps: list[np.ndarray] = []
    for r in range(h):
        for c in range(w):
            if not mask[r, c] or seen[r, c]:
                continue
            queue = deque([(r, c)])
            seen[r, c] = True
            coords: list[tuple[int, int]] = []
            while queue:
                cr, cc = queue.popleft()
                coords.append((cr, cc))
                for dr, dc in NEIGHBORS4:
                    nr, nc = cr + dr, cc + dc
                    if 0 <= nr < h and 0 <= nc < w and mask[nr, nc] and not seen[nr, nc]:
                        seen[nr, nc] = True
                        queue.append((nr, nc))
            comps.append(np.array(coords, dtype=int))
    return comps


def _bbox(coords: np.ndarray) -> tuple[int, int, int, int]:
    mins = coords.min(axis=0)
    maxs = coords.max(axis=0)
    return int(mins[0]), int(mins[1]), int(maxs[0]), int(maxs[1])


def _crop_mask_from_coords(coords: np.ndarray) -> tuple[np.ndarray, tuple[int, int, int, int]]:
    r0, c0, r1, c1 = _bbox(coords)
    crop = np.zeros((r1 - r0 + 1, c1 - c0 + 1), dtype=bool)
    rel = coords - np.array([r0, c0], dtype=int)
    crop[rel[:, 0], rel[:, 1]] = True
    return crop, (r0, c0, r1, c1)


def _is_lr_symmetric(mask: np.ndarray) -> bool:
    return bool(np.array_equal(mask, np.fliplr(mask)))


def main(grid: np.ndarray) -> np.ndarray:
    """
    Keep only the left-right symmetric grey objects, then recolor them by the left-to-right rank of their bbox columns.

    Input structure:
    - The grid is a black background containing several grey 4-connected objects.
    - Each object's tight bounding box is between 3x3 and 5x5.
    - Some objects are mirror-symmetric left-to-right inside their own tight crops, and some are asymmetric distractors.

    Rule:
    - Split the grey pixels into 4-connected objects and crop each one to its tight bounding box.
    - Keep an object iff its cropped foreground mask matches its left-right reflection; erase every asymmetric object.
    - Collect the surviving objects' bbox left columns, sort the distinct columns left-to-right, and map ranks 1..4 to colors blue, red, green, yellow.
    - Recolor every surviving object's pixels by the rank of its bbox left column and leave the background black.

    Key insight:
    - Symmetry decides which objects participate, and only the survivors' columns define the recoloring ranks, so a wrongly kept distractor changes the global column ordering.
    """
    g = np.asarray(grid, dtype=int)
    assert g.ndim == 2, "expected a 2D grid"

    mask = g == INPUT_COLOR
    components = _components(mask)
    assert components, "expected at least one grey object"

    survivors: list[tuple[np.ndarray, tuple[int, int, int, int]]] = []
    for coords in components:
        crop, box = _crop_mask_from_coords(coords)
        if _is_lr_symmetric(crop):
            survivors.append((coords, box))

    left_cols = sorted({box[1] for _coords, box in survivors})
    assert len(left_cols) == 4, "expected exactly four distinct survivor columns"
    color_by_col = {col: OUTPUT_PALETTE[i] for i, col in enumerate(left_cols)}

    out = np.zeros_like(g)
    for coords, box in survivors:
        out[coords[:, 0], coords[:, 1]] = color_by_col[box[1]]
    return out


def _in_bounds(h: int, w: int, r: int, c: int) -> bool:
    return 0 <= r < h and 0 <= c < w


def _random_path(start: tuple[int, int], end: tuple[int, int], rng: random.Random) -> list[tuple[int, int]]:
    r, c = start
    er, ec = end
    steps: list[tuple[int, int]] = []
    steps.extend([(1 if er > r else -1, 0)] * abs(er - r))
    steps.extend([(0, 1 if ec > c else -1)] * abs(ec - c))
    rng.shuffle(steps)

    path = [(r, c)]
    for dr, dc in steps:
        r += dr
        c += dc
        path.append((r, c))
    return path


def _frontier(cells: set[tuple[int, int]], h: int, w: int) -> list[tuple[int, int]]:
    frontier: set[tuple[int, int]] = set()
    for r, c in cells:
        for dr, dc in NEIGHBORS4:
            nr, nc = r + dr, c + dc
            if _in_bounds(h, w, nr, nc) and (nr, nc) not in cells:
                frontier.add((nr, nc))
    return list(frontier)


def _choose_weighted(candidates: list[tuple[int, int]], h: int, w: int, rng: random.Random) -> tuple[int, int]:
    weights = [3 if r in (0, h - 1) or c in (0, w - 1) else 1 for r, c in candidates]
    return rng.choices(candidates, weights=weights, k=1)[0]


def _cells_to_mask(cells: set[tuple[int, int]], h: int, w: int) -> np.ndarray:
    mask = np.zeros((h, w), dtype=bool)
    for r, c in cells:
        mask[r, c] = True
    return mask


def _mirrored_area(half_cells: set[tuple[int, int]], width: int) -> int:
    half_w = (width + 1) // 2
    if width % 2 == 0:
        return 2 * len(half_cells)
    return sum(1 if c == half_w - 1 else 2 for _r, c in half_cells)


def _mirror_half_mask(half_mask: np.ndarray, width: int) -> np.ndarray:
    if width % 2 == 0:
        return np.concatenate([half_mask, np.fliplr(half_mask)], axis=1)
    return np.concatenate([half_mask[:, :-1], np.fliplr(half_mask)], axis=1)


def _is_single_component(mask: np.ndarray) -> bool:
    comps = _components(mask)
    return len(comps) == 1


def _touches_all_bbox_edges(mask: np.ndarray) -> bool:
    return bool(mask[0, :].any() and mask[-1, :].any() and mask[:, 0].any() and mask[:, -1].any())


def _valid_object_mask(mask: np.ndarray, *, symmetric: bool) -> bool:
    area = int(mask.sum())
    h, w = mask.shape
    if area < max(h, w) or area >= h * w:
        return False
    if not _touches_all_bbox_edges(mask):
        return False
    if not _is_single_component(mask):
        return False
    return _is_lr_symmetric(mask) if symmetric else not _is_lr_symmetric(mask)


def _target_area_range(h: int, w: int) -> tuple[int, int]:
    low = max(max(h, w), int(np.ceil(0.38 * h * w)))
    high = min(h * w - 1, int(np.floor(0.74 * h * w)))
    return low, max(low, high)


def _grow_connected(points: list[tuple[int, int]], h: int, w: int, target_area: int, rng: random.Random) -> set[tuple[int, int]]:
    ordered = points[:]
    rng.shuffle(ordered)
    cells = {ordered[0]}
    for point in ordered[1:]:
        start = min(cells, key=lambda cell: abs(cell[0] - point[0]) + abs(cell[1] - point[1]))
        cells.update(_random_path(start, point, rng))

    while len(cells) < target_area:
        border = _frontier(cells, h, w)
        if not border:
            break
        cells.add(_choose_weighted(border, h, w, rng))
    return cells


def _generate_symmetric_mask(h: int, w: int, rng: random.Random) -> np.ndarray:
    half_w = (w + 1) // 2
    area_lo, area_hi = _target_area_range(h, w)
    for _ in range(MASK_ATTEMPTS):
        points = [
            (0, rng.randrange(half_w)),
            (h - 1, rng.randrange(half_w)),
            (rng.randrange(h), 0),
            (rng.randrange(h), half_w - 1),
        ]
        target_full_area = rng.randint(area_lo, area_hi)
        seed_area = max(1, int(np.ceil(target_full_area / 2)))
        half_cells = _grow_connected(points, h, half_w, seed_area, rng)
        while _mirrored_area(half_cells, w) < target_full_area:
            border = _frontier(half_cells, h, half_w)
            if not border:
                break
            half_cells.add(_choose_weighted(border, h, half_w, rng))
        full_mask = _mirror_half_mask(_cells_to_mask(half_cells, h, half_w), w)
        if _valid_object_mask(full_mask, symmetric=True):
            return full_mask
    raise RuntimeError("failed to generate symmetric mask")


def _generate_generic_asymmetric_mask(h: int, w: int, rng: random.Random) -> np.ndarray:
    area_lo, area_hi = _target_area_range(h, w)
    for _ in range(MASK_ATTEMPTS):
        points = [
            (0, rng.randrange(w)),
            (h - 1, rng.randrange(w)),
            (rng.randrange(h), 0),
            (rng.randrange(h), w - 1),
        ]
        target_area = rng.randint(area_lo, area_hi)
        mask = _cells_to_mask(_grow_connected(points, h, w, target_area, rng), h, w)
        if _valid_object_mask(mask, symmetric=False):
            return mask
    raise RuntimeError("failed to generate asymmetric mask")


def _generate_near_symmetric_asymmetric_mask(h: int, w: int, rng: random.Random) -> np.ndarray:
    base = _generate_symmetric_mask(h, w, rng)
    candidates: list[tuple[str, int, int]] = []
    for r in range(h):
        for c in range((w + 1) // 2):
            mirrored_c = w - 1 - c
            if c == mirrored_c:
                continue
            if base[r, c]:
                candidates.append(("remove", r, c))
            else:
                candidates.append(("add", r, c))
    rng.shuffle(candidates)

    for action, r, c in candidates:
        mask = base.copy()
        if action == "add":
            if not any(_in_bounds(h, w, r + dr, c + dc) and base[r + dr, c + dc] for dr, dc in NEIGHBORS4):
                continue
            mask[r, c] = True
        else:
            mask[r, c] = False
        if _valid_object_mask(mask, symmetric=False):
            return mask
    return _generate_generic_asymmetric_mask(h, w, rng)


def _boxes_are_separated(a: tuple[int, int, int, int], b: tuple[int, int, int, int], gap: int = 1) -> bool:
    return (
        a[2] + gap < b[0]
        or b[2] + gap < a[0]
        or a[3] + gap < b[1]
        or b[3] + gap < a[1]
    )


def _place_objects(
    h: int,
    w: int,
    specs: list[dict[str, object]],
    rng: random.Random,
) -> list[dict[str, object]] | None:
    placed: list[dict[str, object]] = []
    order = list(specs)
    order.sort(key=lambda rec: int(np.prod(np.array(rec["mask"]).shape)), reverse=True)
    for spec in order:
        mask = np.array(spec["mask"], dtype=bool)
        mh, mw = mask.shape
        left = int(spec["left"])
        assert left + mw <= w
        for _ in range(PLACEMENT_ATTEMPTS):
            top = rng.randrange(h - mh + 1)
            box = (top, left, top + mh - 1, left + mw - 1)
            if all(_boxes_are_separated(box, tuple(other["bbox"])) for other in placed):
                coords = np.argwhere(mask) + np.array([top, left], dtype=int)
                placed.append(
                    {
                        "kind": spec["kind"],
                        "mask": mask,
                        "top": top,
                        "left": left,
                        "bbox": box,
                        "coords": coords,
                    }
                )
                break
        else:
            return None
    return placed


def _render_input(h: int, w: int, placed: list[dict[str, object]]) -> np.ndarray:
    grid = np.zeros((h, w), dtype=int)
    for rec in placed:
        coords = np.array(rec["coords"], dtype=int)
        grid[coords[:, 0], coords[:, 1]] = INPUT_COLOR
    return grid


def _expected_output(shape: tuple[int, int], placed: list[dict[str, object]]) -> np.ndarray:
    out = np.zeros(shape, dtype=int)
    sym_cols = sorted({int(rec["bbox"][1]) for rec in placed if rec["kind"] == "sym"})
    color_by_col = {col: OUTPUT_PALETTE[i] for i, col in enumerate(sym_cols)}
    for rec in placed:
        if rec["kind"] != "sym":
            continue
        coords = np.array(rec["coords"], dtype=int)
        out[coords[:, 0], coords[:, 1]] = color_by_col[int(rec["bbox"][1])]
    return out


def _actual_object_summary(grid: np.ndarray) -> list[tuple[tuple[int, int, int, int], bool]]:
    summary: list[tuple[tuple[int, int, int, int], bool]] = []
    for coords in _components(grid == INPUT_COLOR):
        crop, box = _crop_mask_from_coords(coords)
        summary.append((box, _is_lr_symmetric(crop)))
    return sorted(summary)


def _validate_example(grid: np.ndarray, placed: list[dict[str, object]]) -> None:
    h, w = grid.shape
    assert 0 < h <= 30 and 0 < w <= 30
    assert np.isin(grid, [BACKGROUND, INPUT_COLOR]).all()

    expected_summary = sorted((tuple(rec["bbox"]), rec["kind"] == "sym") for rec in placed)
    actual_summary = _actual_object_summary(grid)
    assert actual_summary == expected_summary

    sym_records = [rec for rec in placed if rec["kind"] == "sym"]
    asym_records = [rec for rec in placed if rec["kind"] == "asym"]
    assert 5 <= len(placed) <= 7
    assert len(asym_records) >= 2

    sym_cols = sorted({int(rec["bbox"][1]) for rec in sym_records})
    asym_cols = [int(rec["bbox"][1]) for rec in asym_records]
    assert len(sym_cols) == 4
    assert all(col not in sym_cols and col < sym_cols[-1] for col in asym_cols)

    correct_ranks = {col: i for i, col in enumerate(sym_cols)}
    for asym_col in asym_cols:
        wrong_cols = sorted(set(sym_cols) | {asym_col})
        wrong_ranks = {col: i for i, col in enumerate(wrong_cols)}
        assert any(wrong_ranks[col] != correct_ranks[col] for col in sym_cols)

    expected = _expected_output(grid.shape, placed)
    actual = main(grid)
    assert np.array_equal(actual, expected)

    used_colors = set(np.unique(actual)) - {BACKGROUND}
    assert used_colors == set(OUTPUT_PALETTE)

    if len(sym_records) > 4:
        assert len([rec for rec in sym_records if int(rec["bbox"][1]) == int(sym_records[-1]["bbox"][1])]) >= 1


def _sample_columns(
    width: int,
    asym_count: int,
    rng: random.Random,
) -> tuple[list[int], list[int]]:
    possible = list(range(0, width - 4))
    for _ in range(100):
        sym_cols = sorted(rng.sample(possible, 4))
        pool = [col for col in possible if col not in sym_cols and col < sym_cols[-1]]
        if len(pool) >= asym_count:
            asym_cols = rng.sample(pool, asym_count)
            return sym_cols, asym_cols
    raise RuntimeError("failed to sample valid columns")


def _build_specs(width: int, rng: random.Random) -> list[dict[str, object]]:
    sym_count = 5 if rng.random() < 0.6 else 4
    asym_count = 2 if sym_count == 5 else rng.choice([2, 3])
    sym_cols, asym_cols = _sample_columns(width, asym_count, rng)

    sym_object_cols = sym_cols[:]
    if sym_count == 5:
        sym_object_cols.append(rng.choice(sym_cols))
    rng.shuffle(sym_object_cols)
    rng.shuffle(asym_cols)

    specs: list[dict[str, object]] = []
    for left in sym_object_cols:
        h = rng.randint(3, 5)
        w = rng.randint(3, 5)
        specs.append({"kind": "sym", "left": left, "mask": _generate_symmetric_mask(h, w, rng)})

    for idx, left in enumerate(asym_cols):
        h = rng.randint(3, 5)
        w = rng.randint(3, 5)
        if idx == 0 or rng.random() < 0.5:
            mask = _generate_near_symmetric_asymmetric_mask(h, w, rng)
        else:
            mask = _generate_generic_asymmetric_mask(h, w, rng)
        specs.append({"kind": "asym", "left": left, "mask": mask})
    return specs


def _gen_attempt(rng: random.Random) -> np.ndarray | None:
    h = rng.randint(16, 20)
    w = rng.randint(16, 20)
    specs = _build_specs(w, rng)
    placed = _place_objects(h, w, specs, rng)
    if placed is None:
        return None

    grid = _render_input(h, w, placed)
    if np.count_nonzero(grid == INPUT_COLOR) >= grid.size // 2:
        return None

    try:
        _validate_example(grid, placed)
    except AssertionError:
        return None
    return grid


def generate_input() -> np.ndarray:
    rng = random.Random()
    for _ in range(GEN_ATTEMPTS):
        grid = _gen_attempt(rng)
        if grid is not None:
            return grid
    raise RuntimeError("failed to generate a valid input")
