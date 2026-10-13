from __future__ import annotations

import random

import numpy as np


BLACK = 0
GREY = 5
SEED_COLORS = (1, 2, 3, 4, 6, 7, 8, 9)


GRID_H_RANGE = (12, 20)
GRID_W_RANGE = (12, 20)
RECT_COUNT_CHOICES = (2, 3, 4, 5)
RECT_COUNT_WEIGHTS = (2, 4, 3, 1)
RECT_H_RANGE = (2, 6)
RECT_W_RANGE = (2, 6)
RECT_PLACEMENT_RETRIES = 120
SEED_GAP_MAX = 4
GENERATION_RETRIES = 400


def _components(mask: np.ndarray) -> list[np.ndarray]:
    h, w = mask.shape
    seen = np.zeros_like(mask, dtype=bool)
    comps: list[np.ndarray] = []
    for r in range(h):
        for c in range(w):
            if not mask[r, c] or seen[r, c]:
                continue
            stack = [(r, c)]
            seen[r, c] = True
            cells: list[tuple[int, int]] = []
            while stack:
                cr, cc = stack.pop()
                cells.append((cr, cc))
                for nr, nc in ((cr - 1, cc), (cr + 1, cc), (cr, cc - 1), (cr, cc + 1)):
                    if 0 <= nr < h and 0 <= nc < w and mask[nr, nc] and not seen[nr, nc]:
                        seen[nr, nc] = True
                        stack.append((nr, nc))
            comps.append(np.asarray(cells, dtype=int))
    return comps


def _bbox(coords: np.ndarray) -> tuple[int, int, int, int]:
    rows = coords[:, 0]
    cols = coords[:, 1]
    return int(rows.min()), int(cols.min()), int(rows.max()), int(cols.max())


def _find_soil_rectangles(grid: np.ndarray) -> list[tuple[int, int, int, int]]:
    rectangles: list[tuple[int, int, int, int]] = []
    for coords in _components(grid == GREY):
        r0, c0, r1, c1 = _bbox(coords)
        area = (r1 - r0 + 1) * (c1 - c0 + 1)
        assert len(coords) == area, "grey components must be solid rectangles"
        assert np.all(grid[r0 : r1 + 1, c0 : c1 + 1] == GREY), "grey rectangles must be filled"
        rectangles.append((r0, c0, r1, c1))
    rectangles.sort()
    return rectangles


def _seed_matches(
    grid: np.ndarray,
    rectangles: list[tuple[int, int, int, int]],
    seed: tuple[int, int],
) -> list[tuple[int, str, str]]:
    sr, sc = seed
    matches: list[tuple[int, str, str]] = []
    for rect_index, (r0, c0, r1, c1) in enumerate(rectangles):
        if r0 <= sr <= r1:
            if sc < c0 and np.all(grid[sr, sc + 1 : c0] == BLACK):
                matches.append((rect_index, "horizontal", "left"))
            if sc > c1 and np.all(grid[sr, c1 + 1 : sc] == BLACK):
                matches.append((rect_index, "horizontal", "right"))
        if c0 <= sc <= c1:
            if sr < r0 and np.all(grid[sr + 1 : r0, sc] == BLACK):
                matches.append((rect_index, "vertical", "up"))
            if sr > r1 and np.all(grid[r1 + 1 : sr, sc] == BLACK):
                matches.append((rect_index, "vertical", "down"))
    return matches


def _effect_cells(
    rect: tuple[int, int, int, int],
    seed: tuple[int, int],
    orientation: str,
) -> list[tuple[int, int]]:
    sr, sc = seed
    r0, c0, r1, c1 = rect
    cells: list[tuple[int, int]] = []
    if orientation == "horizontal":
        if sc < c0:
            cells.extend((sr, c) for c in range(sc + 1, c0))
        else:
            cells.extend((sr, c) for c in range(c1 + 1, sc))
        cells.extend((sr, c) for c in range(c0, c1 + 1))
    else:
        if sr < r0:
            cells.extend((r, sc) for r in range(sr + 1, r0))
        else:
            cells.extend((r, sc) for r in range(r1 + 1, sr))
        cells.extend((r, sc) for r in range(r0, r1 + 1))
    return cells


def main(grid: np.ndarray) -> np.ndarray:
    """
    Draw a seed-colored connector from each seed to its aligned grey rectangle,
    then repaint the aligned row or column inside that rectangle.
    """
    g = np.asarray(grid, dtype=int)
    assert g.ndim == 2, "expected a 2D grid"

    rectangles = _find_soil_rectangles(g)
    out = g.copy()

    for sr, sc in np.argwhere((g != BLACK) & (g != GREY)):
        color = int(g[sr, sc])
        matches = _seed_matches(g, rectangles, (int(sr), int(sc)))
        assert len(matches) == 1, "each seed must align with exactly one rectangle"
        rect_index, orientation, _side = matches[0]
        r0, c0, r1, c1 = rectangles[rect_index]

        if orientation == "horizontal":
            if sc < c0:
                out[sr, sc + 1 : c0] = color
            else:
                out[sr, c1 + 1 : sc] = color
            out[sr, c0 : c1 + 1] = color
        else:
            if sr < r0:
                out[sr + 1 : r0, sc] = color
            else:
                out[r1 + 1 : sr, sc] = color
            out[r0 : r1 + 1, sc] = color

    return out


def _clear_with_pad(mask: np.ndarray, r0: int, c0: int, r1: int, c1: int, pad: int) -> bool:
    rr0 = max(0, r0 - pad)
    cc0 = max(0, c0 - pad)
    rr1 = min(mask.shape[0] - 1, r1 + pad)
    cc1 = min(mask.shape[1] - 1, c1 + pad)
    return not bool(mask[rr0 : rr1 + 1, cc0 : cc1 + 1].any())


def _place_rectangles(height: int, width: int) -> list[tuple[int, int, int, int]] | None:
    rect_count = random.choices(RECT_COUNT_CHOICES, weights=RECT_COUNT_WEIGHTS, k=1)[0]
    sizes = [
        (random.randint(*RECT_H_RANGE), random.randint(*RECT_W_RANGE))
        for _ in range(rect_count)
    ]
    sizes.sort(key=lambda hw: hw[0] * hw[1], reverse=True)

    occupied = np.zeros((height, width), dtype=bool)
    rects: list[tuple[int, int, int, int]] = []
    for rect_h, rect_w in sizes:
        placed = False
        for _ in range(RECT_PLACEMENT_RETRIES):
            if rect_h > height - 2 or rect_w > width - 2:
                break
            r0 = random.randint(1, height - rect_h - 1)
            c0 = random.randint(1, width - rect_w - 1)
            r1 = r0 + rect_h - 1
            c1 = c0 + rect_w - 1
            if not _clear_with_pad(occupied, r0, c0, r1, c1, pad=1):
                continue
            occupied[r0 : r1 + 1, c0 : c1 + 1] = True
            rects.append((r0, c0, r1, c1))
            placed = True
            break
        if not placed:
            return None
    rects.sort()
    return rects


def _seed_buffer_mark(seed_buffer: np.ndarray, r: int, c: int) -> None:
    r0 = max(0, r - 1)
    c0 = max(0, c - 1)
    r1 = min(seed_buffer.shape[0] - 1, r + 1)
    c1 = min(seed_buffer.shape[1] - 1, c + 1)
    seed_buffer[r0 : r1 + 1, c0 : c1 + 1] = True


def _seed_candidates(
    grid: np.ndarray,
    rectangles: list[tuple[int, int, int, int]],
    rect_index: int,
    orientation: str,
    seed_mask: np.ndarray,
    seed_buffer: np.ndarray,
    effect_used: np.ndarray,
) -> list[dict]:
    h, w = grid.shape
    r0, c0, r1, c1 = rectangles[rect_index]
    candidates: list[dict] = []

    if orientation == "horizontal":
        lines = list(range(r0, r1 + 1))
        random.shuffle(lines)
        for row in lines:
            sides = ["left", "right"]
            random.shuffle(sides)
            for side in sides:
                max_gap = min(SEED_GAP_MAX, c0 - 1) if side == "left" else min(SEED_GAP_MAX, w - c1 - 2)
                if max_gap < 1:
                    continue
                gaps = list(range(1, max_gap + 1))
                random.shuffle(gaps)
                for gap in gaps:
                    seed = (row, c0 - gap - 1) if side == "left" else (row, c1 + gap + 1)
                    sr, sc = seed
                    if grid[sr, sc] != BLACK or seed_buffer[sr, sc]:
                        continue
                    effect = _effect_cells(rectangles[rect_index], seed, orientation)
                    if any(seed_mask[er, ec] or effect_used[er, ec] for er, ec in effect):
                        continue
                    temp = grid.copy()
                    temp[sr, sc] = SEED_COLORS[0]
                    matches = _seed_matches(temp, rectangles, seed)
                    if len(matches) != 1 or matches[0][:2] != (rect_index, orientation):
                        continue
                    candidates.append({"seed": seed, "effect": effect, "side": side})
    else:
        lines = list(range(c0, c1 + 1))
        random.shuffle(lines)
        for col in lines:
            sides = ["up", "down"]
            random.shuffle(sides)
            for side in sides:
                max_gap = min(SEED_GAP_MAX, r0 - 1) if side == "up" else min(SEED_GAP_MAX, h - r1 - 2)
                if max_gap < 1:
                    continue
                gaps = list(range(1, max_gap + 1))
                random.shuffle(gaps)
                for gap in gaps:
                    seed = (r0 - gap - 1, col) if side == "up" else (r1 + gap + 1, col)
                    sr, sc = seed
                    if grid[sr, sc] != BLACK or seed_buffer[sr, sc]:
                        continue
                    effect = _effect_cells(rectangles[rect_index], seed, orientation)
                    if any(seed_mask[er, ec] or effect_used[er, ec] for er, ec in effect):
                        continue
                    temp = grid.copy()
                    temp[sr, sc] = SEED_COLORS[0]
                    matches = _seed_matches(temp, rectangles, seed)
                    if len(matches) != 1 or matches[0][:2] != (rect_index, orientation):
                        continue
                    candidates.append({"seed": seed, "effect": effect, "side": side})

    return candidates


def _sample_seed_plan(rectangles: list[tuple[int, int, int, int]]) -> tuple[list[int], dict[int, str]]:
    rect_count = len(rectangles)
    if rect_count == 1:
        seeded_count = 1
    elif random.random() < 0.45:
        seeded_count = random.randint(1, rect_count - 1)
    else:
        seeded_count = random.randint(2, rect_count)

    seeded_rects = random.sample(range(rect_count), seeded_count)
    orientations: dict[int, str] = {}
    if seeded_count == 1:
        orientations[seeded_rects[0]] = random.choice(("horizontal", "vertical"))
        return seeded_rects, orientations

    orient_list = ["horizontal", "vertical"]
    orient_list.extend(random.choice(("horizontal", "vertical")) for _ in range(seeded_count - 2))
    random.shuffle(orient_list)
    for rect_index, orientation in zip(seeded_rects, orient_list):
        orientations[rect_index] = orientation
    return seeded_rects, orientations


def _apply_placements(
    grid: np.ndarray,
    rectangles: list[tuple[int, int, int, int]],
    placements: list[dict],
) -> np.ndarray:
    out = grid.copy()
    for placement in placements:
        color = int(placement["color"])
        for r, c in _effect_cells(
            rectangles[int(placement["rect_index"])],
            tuple(placement["seed"]),
            str(placement["orientation"]),
        ):
            out[r, c] = color
    return out


def generate_input() -> np.ndarray:
    for _ in range(GENERATION_RETRIES):
        height = random.randint(*GRID_H_RANGE)
        width = random.randint(*GRID_W_RANGE)
        rectangles = _place_rectangles(height, width)
        if rectangles is None:
            continue

        grid = np.full((height, width), BLACK, dtype=int)
        for r0, c0, r1, c1 in rectangles:
            grid[r0 : r1 + 1, c0 : c1 + 1] = GREY

        seeded_rects, orientations = _sample_seed_plan(rectangles)
        if len(seeded_rects) > len(SEED_COLORS):
            continue
        color_pool = list(SEED_COLORS)
        random.shuffle(color_pool)

        seed_mask = np.zeros_like(grid, dtype=bool)
        seed_buffer = np.zeros_like(grid, dtype=bool)
        effect_used = np.zeros_like(grid, dtype=bool)
        placements: list[dict] = []
        success = True

        rect_order = seeded_rects[:]
        random.shuffle(rect_order)
        for rect_index, color in zip(rect_order, color_pool):
            orientation = orientations[rect_index]
            candidates = _seed_candidates(
                grid,
                rectangles,
                rect_index,
                orientation,
                seed_mask,
                seed_buffer,
                effect_used,
            )
            if not candidates:
                success = False
                break
            chosen = random.choice(candidates)
            sr, sc = chosen["seed"]
            grid[sr, sc] = color
            seed_mask[sr, sc] = True
            _seed_buffer_mark(seed_buffer, sr, sc)
            for er, ec in chosen["effect"]:
                effect_used[er, ec] = True
            placements.append(
                {
                    "seed": (sr, sc),
                    "rect_index": rect_index,
                    "orientation": orientation,
                    "side": chosen["side"],
                    "color": color,
                }
            )

        if not success or not placements:
            continue

        expected = _apply_placements(grid, rectangles, placements)
        try:
            actual = main(grid)
        except AssertionError:
            continue
        if not np.array_equal(actual, expected):
            continue
        return grid

    raise RuntimeError("failed to generate a valid ARC instance")
