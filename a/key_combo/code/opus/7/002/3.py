from __future__ import annotations

import random
from collections import Counter, deque

import numpy as np


BACKGROUND = 0
SPRITE_COLOR = 2
BASELINE_COLOR = 5
SLOT_COLORS = (1, 3, 4, 6, 7, 8, 9)
ORIENTATION_MASKS = {
    "ul": np.array([[0, SPRITE_COLOR], [SPRITE_COLOR, SPRITE_COLOR]], dtype=int),
    "ur": np.array([[SPRITE_COLOR, 0], [SPRITE_COLOR, SPRITE_COLOR]], dtype=int),
    "ll": np.array([[SPRITE_COLOR, SPRITE_COLOR], [0, SPRITE_COLOR]], dtype=int),
    "lr": np.array([[SPRITE_COLOR, SPRITE_COLOR], [SPRITE_COLOR, 0]], dtype=int),
}
MAX_GENERATION_ATTEMPTS = 500


def _get_bg(grid: np.ndarray) -> int:
    return int(Counter(map(int, np.asarray(grid).ravel())).most_common(1)[0][0])


def _bbox(coords: np.ndarray) -> tuple[int, int, int, int]:
    r0, c0 = coords.min(axis=0)
    r1, c1 = coords.max(axis=0)
    return int(r0), int(c0), int(r1), int(c1)


def _find_components_of_color(grid: np.ndarray, color: int) -> list[np.ndarray]:
    mask = np.asarray(grid == int(color), dtype=bool)
    h, w = mask.shape
    seen = np.zeros((h, w), dtype=bool)
    comps: list[np.ndarray] = []
    for r in range(h):
        for c in range(w):
            if not mask[r, c] or seen[r, c]:
                continue
            q: deque[tuple[int, int]] = deque([(r, c)])
            seen[r, c] = True
            coords: list[tuple[int, int]] = []
            while q:
                rr, cc = q.popleft()
                coords.append((rr, cc))
                for nr, nc in ((rr - 1, cc), (rr + 1, cc), (rr, cc - 1), (rr, cc + 1)):
                    if 0 <= nr < h and 0 <= nc < w and mask[nr, nc] and not seen[nr, nc]:
                        seen[nr, nc] = True
                        q.append((nr, nc))
            comps.append(np.asarray(coords, dtype=int))
    return comps


def _slot_info(grid: np.ndarray) -> tuple[int, int, list[int]]:
    g = np.asarray(grid, dtype=int)
    baseline_rows = np.flatnonzero(np.all(g == BASELINE_COLOR, axis=1))
    assert baseline_rows.size == 1, "expected exactly one full-width grey baseline"
    baseline_r = int(baseline_rows[0])
    assert baseline_r >= 2, "baseline must leave room above it"
    slot_r = baseline_r - 1
    row = g[slot_r]
    slot_cols = np.flatnonzero((row != BACKGROUND) & (row != BASELINE_COLOR))
    assert slot_cols.size >= 6 and slot_cols.size % 2 == 0, "slot row must contain 3-5 adjacent 2-cell slots"
    start = int(slot_cols[0])
    assert int(slot_cols[-1] - start + 1) == int(slot_cols.size), "slot cells must form one contiguous run"

    colors: list[int] = []
    for c0 in range(start, start + int(slot_cols.size), 2):
        pair = row[c0 : c0 + 2]
        assert pair.shape == (2,) and int(pair[0]) == int(pair[1]), "each slot must be a 2-cell color block"
        color = int(pair[0])
        assert color not in {BACKGROUND, BASELINE_COLOR, SPRITE_COLOR}, "slot color must differ from bg/baseline/sprite"
        colors.append(color)
    assert len(colors) == len(set(colors)), "slot colors must be distinct"
    return baseline_r, start, colors


def _sprite_records(grid: np.ndarray, baseline_r: int) -> list[dict[str, object]]:
    g = np.asarray(grid, dtype=int)
    comps = _find_components_of_color(g[: baseline_r - 1], SPRITE_COLOR)
    assert 3 <= len(comps) <= 5, "expected 3-5 red sprites above the slots"

    records: list[dict[str, object]] = []
    for coords in comps:
        assert coords.shape == (3, 2), "each sprite must contain exactly 3 pixels"
        r0, c0, r1, c1 = _bbox(coords)
        assert (r1 - r0 + 1, c1 - c0 + 1) == (2, 2), "each sprite must fit a 2x2 bbox"
        crop = g[r0 : r1 + 1, c0 : c1 + 1]
        assert np.array_equal((crop == SPRITE_COLOR).astype(int), (crop != BACKGROUND).astype(int))
        zeroes = np.argwhere(crop == BACKGROUND)
        assert zeroes.shape == (1, 2), "each 2x2 sprite crop must have exactly one missing corner"
        missing_r, missing_c = map(int, zeroes[0])
        transformed = np.fliplr(crop) if missing_c == 0 else np.flipud(crop)
        records.append(
            {
                "coords": coords,
                "bbox": (r0, c0, r1, c1),
                "crop": crop.copy(),
                "missing": (missing_r, missing_c),
                "transformed": transformed.copy(),
            }
        )

    top_rows = [int(rec["bbox"][0]) for rec in records]
    assert len(top_rows) == len(set(top_rows)), "sprite bbox top rows must be unique"
    return sorted(records, key=lambda rec: (int(rec["bbox"][0]), int(rec["bbox"][1])))


def _solve(grid: np.ndarray, *, dock_overlap_slot_row: bool) -> np.ndarray:
    g = np.asarray(grid, dtype=int)
    assert g.ndim == 2, "expected a 2D grid"
    assert _get_bg(g) == BACKGROUND, "expected black background"

    baseline_r, slot_start, slot_colors = _slot_info(g)
    sprites = _sprite_records(g, baseline_r)
    assert len(sprites) == len(slot_colors), "slot count must equal sprite count"

    out = g.copy()
    out[g == SPRITE_COLOR] = BACKGROUND

    top_r = baseline_r - 2 if dock_overlap_slot_row else baseline_r - 3
    assert top_r >= 0, "not enough room to place sprites above the baseline"

    for idx, sprite in enumerate(sprites):
        color = int(slot_colors[idx])
        c0 = slot_start + 2 * idx
        patch = np.where(np.asarray(sprite["transformed"]) == SPRITE_COLOR, color, BACKGROUND)
        for pr in range(2):
            for pc in range(2):
                if patch[pr, pc] != BACKGROUND:
                    out[top_r + pr, c0 + pc] = int(patch[pr, pc])

    return out


def main(grid: np.ndarray) -> np.ndarray:
    """
    Reorder tiny red L-sprites by row, mirror each by its missing-corner side, and dock them above colored slots.

    Input structure:
    - Black background, with 3-5 red 3-cell L-sprites, each occupying a 2x2 bbox with one missing corner.
    - The sprites lie in the upper area and have distinct bbox top-row positions.
    - Near the bottom, a full-width grey baseline is present; directly above it sits one contiguous run of distinct colored 2x1 slot markers, one per sprite.

    Rule:
    - Crop each red sprite to its 2x2 bbox and read the missing-corner column.
    - If the missing corner is on the left, mirror the sprite horizontally; if it is on the right, mirror it vertically.
    - Sort sprites by bbox top row from top to bottom.
    - Clear the original red sprites, then place the mirrored sprites from left to right above the colored slots in that sorted order.
    - Recolor each placed sprite to the color of its destination slot, while keeping the slot markers and grey baseline unchanged.

    Key insight:
    - The missing-corner side controls which mirror operation to apply, while the input row order independently controls which slot receives each mirrored shape.
    """
    return _solve(grid, dock_overlap_slot_row=False)


def _expected_from_scene(
    grid: np.ndarray,
    rows: list[int],
    cols: list[int],
    orientations: list[str],
    slot_start: int,
    slot_colors: list[int],
) -> np.ndarray:
    out = np.asarray(grid, dtype=int).copy()
    out[out == SPRITE_COLOR] = BACKGROUND
    top_r = out.shape[0] - 4
    order = sorted(range(len(rows)), key=lambda i: rows[i])
    for slot_idx, obj_idx in enumerate(order):
        patch = ORIENTATION_MASKS[orientations[obj_idx]]
        missing_c = int(np.argwhere(patch == BACKGROUND)[0, 1])
        transformed = np.fliplr(patch) if missing_c == 0 else np.flipud(patch)
        color = int(slot_colors[slot_idx])
        c0 = slot_start + 2 * slot_idx
        for pr in range(2):
            for pc in range(2):
                if int(transformed[pr, pc]) == SPRITE_COLOR:
                    out[top_r + pr, c0 + pc] = color
    return out


def _wrong_sorted_by_col(
    grid: np.ndarray,
    rows: list[int],
    cols: list[int],
    orientations: list[str],
    slot_start: int,
    slot_colors: list[int],
) -> np.ndarray:
    out = np.asarray(grid, dtype=int).copy()
    out[out == SPRITE_COLOR] = BACKGROUND
    top_r = out.shape[0] - 4
    order = sorted(range(len(cols)), key=lambda i: (cols[i], rows[i]))
    for slot_idx, obj_idx in enumerate(order):
        patch = ORIENTATION_MASKS[orientations[obj_idx]]
        missing_c = int(np.argwhere(patch == BACKGROUND)[0, 1])
        transformed = np.fliplr(patch) if missing_c == 0 else np.flipud(patch)
        color = int(slot_colors[slot_idx])
        c0 = slot_start + 2 * slot_idx
        for pr in range(2):
            for pc in range(2):
                if int(transformed[pr, pc]) == SPRITE_COLOR:
                    out[top_r + pr, c0 + pc] = color
    return out


def _wrong_mirror_all_horizontal(
    grid: np.ndarray,
    rows: list[int],
    orientations: list[str],
    slot_start: int,
    slot_colors: list[int],
) -> np.ndarray:
    out = np.asarray(grid, dtype=int).copy()
    out[out == SPRITE_COLOR] = BACKGROUND
    top_r = out.shape[0] - 4
    order = sorted(range(len(rows)), key=lambda i: rows[i])
    for slot_idx, obj_idx in enumerate(order):
        transformed = np.fliplr(ORIENTATION_MASKS[orientations[obj_idx]])
        color = int(slot_colors[slot_idx])
        c0 = slot_start + 2 * slot_idx
        for pr in range(2):
            for pc in range(2):
                if int(transformed[pr, pc]) == SPRITE_COLOR:
                    out[top_r + pr, c0 + pc] = color
    return out


def _orientation_sample(n: int) -> list[str]:
    if n >= 4:
        pool = ["ul", "ur", "ll", "lr"]
        random.shuffle(pool)
        while len(pool) < n:
            pool.append(random.choice(("ul", "ur", "ll", "lr")))
        return pool

    while True:
        sample = random.sample(("ul", "ur", "ll", "lr"), n)
        left = any(name in {"ul", "ll"} for name in sample)
        right = any(name in {"ur", "lr"} for name in sample)
        ur_like = any(name in {"ul", "lr"} for name in sample)
        lr_like = any(name in {"ll", "ur"} for name in sample)
        if left and right and ur_like and lr_like:
            return sample


def _separated(existing: list[tuple[int, int, int, int]], bbox: tuple[int, int, int, int]) -> bool:
    r0, c0, r1, c1 = bbox
    for er0, ec0, er1, ec1 in existing:
        if not (r1 < er0 - 1 or er1 < r0 - 1 or c1 < ec0 - 1 or ec1 < c0 - 1):
            return False
    return True


def _place_sprites(h: int, w: int, n: int) -> tuple[list[int], list[int]] | None:
    row_choices = list(range(0, h - 5))
    if len(row_choices) < n:
        return None

    for _ in range(120):
        rows = sorted(random.sample(row_choices, n))
        boxes: list[tuple[int, int, int, int]] = []
        cols: list[int] = []
        ok = True
        order = list(range(n))
        random.shuffle(order)
        for idx in order:
            candidates = []
            for c0 in range(0, w - 1):
                box = (rows[idx], c0, rows[idx] + 1, c0 + 1)
                if _separated(boxes, box):
                    candidates.append(c0)
            if not candidates:
                ok = False
                break
            c0 = random.choice(candidates)
            cols.append((idx, c0))
            boxes.append((rows[idx], c0, rows[idx] + 1, c0 + 1))
        if not ok:
            continue

        cols_sorted = [c0 for idx, c0 in sorted(cols)]
        if all(cols_sorted[i] < cols_sorted[i + 1] for i in range(n - 1)):
            continue
        return rows, cols_sorted
    return None


def _build_input(
    h: int,
    w: int,
    rows: list[int],
    cols: list[int],
    orientations: list[str],
    slot_start: int,
    slot_colors: list[int],
) -> np.ndarray:
    grid = np.full((h, w), BACKGROUND, dtype=int)
    grid[h - 1, :] = BASELINE_COLOR
    for idx, color in enumerate(slot_colors):
        c0 = slot_start + 2 * idx
        grid[h - 2, c0 : c0 + 2] = int(color)
    for r0, c0, name in zip(rows, cols, orientations):
        patch = ORIENTATION_MASKS[name]
        grid[r0 : r0 + 2, c0 : c0 + 2] = np.where(patch == SPRITE_COLOR, SPRITE_COLOR, grid[r0 : r0 + 2, c0 : c0 + 2])
    return grid


def _gen_attempt() -> np.ndarray | None:
    n = random.choices((3, 4, 5), weights=(1, 4, 3), k=1)[0]
    h = random.randint(max(11, n + 6), 18)
    w = random.randint(max(12, 2 * n + 4), 20)
    slot_start = random.randint(0, w - 2 * n)
    slot_colors = random.sample(SLOT_COLORS, n)
    orientations = _orientation_sample(n)

    placed = _place_sprites(h, w, n)
    if placed is None:
        return None
    rows, cols = placed

    grid = _build_input(h, w, rows, cols, orientations, slot_start, slot_colors)
    expected = _expected_from_scene(grid, rows, cols, orientations, slot_start, slot_colors)

    try:
        solved = main(grid)
    except AssertionError:
        return None

    if not np.array_equal(solved, expected):
        return None
    if np.array_equal(expected, _wrong_sorted_by_col(grid, rows, cols, orientations, slot_start, slot_colors)):
        return None
    if np.array_equal(expected, _wrong_mirror_all_horizontal(grid, rows, orientations, slot_start, slot_colors)):
        return None
    if np.array_equal(expected, _solve(grid, dock_overlap_slot_row=True)):
        return None
    shape_signatures = {
        tuple(map(tuple, (solved[h - 4 : h - 2, slot_start + 2 * i : slot_start + 2 * i + 2] != BACKGROUND).astype(int)))
        for i in range(n)
    }
    if len(shape_signatures) < 2:
        return None
    return grid


def generate_input() -> np.ndarray:
    for _ in range(MAX_GENERATION_ATTEMPTS):
        grid = _gen_attempt()
        if grid is not None:
            return grid
    raise RuntimeError("failed to generate a valid input for puzzle 3")
