from __future__ import annotations

import random
from collections import Counter

import numpy as np


BACKGROUND = 5
RED = 2
TEAL = 8
BAR_COLORS = (1, 3, 4)
DIR4 = ((-1, 0), (1, 0), (0, -1), (0, 1))


def _crop(grid: np.ndarray, box: tuple[int, int, int, int]) -> np.ndarray:
    r0, c0, r1, c1 = box
    return grid[r0 : r1 + 1, c0 : c1 + 1]


def _most_common_color(grid: np.ndarray) -> int:
    return int(Counter(map(int, grid.ravel())).most_common(1)[0][0])


def _solid_panel_boxes(grid: np.ndarray, bg: int) -> list[tuple[int, int, int, int]]:
    h, w = grid.shape
    seen = np.zeros((h, w), dtype=bool)
    boxes: list[tuple[int, int, int, int]] = []

    for r in range(h):
        for c in range(w):
            if seen[r, c] or int(grid[r, c]) == bg:
                continue

            stack = [(r, c)]
            seen[r, c] = True
            coords = []
            while stack:
                rr, cc = stack.pop()
                coords.append((rr, cc))
                for dr, dc in DIR4:
                    nr, nc = rr + dr, cc + dc
                    if 0 <= nr < h and 0 <= nc < w and not seen[nr, nc] and int(grid[nr, nc]) != bg:
                        seen[nr, nc] = True
                        stack.append((nr, nc))

            rows = [rr for rr, _cc in coords]
            cols = [cc for _rr, cc in coords]
            box = (min(rows), min(cols), max(rows), max(cols))
            panel = _crop(grid, box)
            assert np.all(panel != bg), "each panel must be a solid rectangle with no background holes"
            assert panel.size == len(coords), "foreground component must exactly fill its bounding box"
            boxes.append(box)

    boxes.sort()
    return boxes


def _validate_panel(panel: np.ndarray) -> None:
    assert panel.ndim == 2 and panel.shape[0] >= 2 and panel.shape[1] >= 2
    bar_color = int(panel[0, 0])
    assert bar_color in BAR_COLORS, "left edge bar must use one of the allowed visible colors"
    assert np.all(panel[:, 0] == bar_color), "leftmost column must be a full edge bar"
    assert not np.any(panel[:, 0] == RED), "the edge bar cannot contain red markers"
    assert set(map(int, np.unique(panel))).issubset(set(BAR_COLORS) | {RED, TEAL}), "unexpected panel colors"


def _stretch_panel(panel: np.ndarray, bg: int = BACKGROUND) -> np.ndarray:
    _validate_panel(panel)
    clean = np.where(panel == RED, TEAL, panel)
    row_factors = [int(np.count_nonzero(row == RED)) for row in panel]
    assert all(1 <= k <= 3 for k in row_factors), "selected panel rows must encode factors 1..3"

    h, w = panel.shape
    max_factor = max(row_factors)
    out = np.full((h, w * max_factor), bg, dtype=int)
    for r, factor in enumerate(row_factors):
        out[r, : w * factor] = np.tile(clean[r], factor)
    return out


def main(grid: np.ndarray) -> np.ndarray:
    """Select the red-rich panel, then stretch each of its rows horizontally by that row's red-marker count.

    Input structure:
    - The grid has a grey background separating exactly three non-touching solid rectangular panels.
    - Every panel is filled by teal cells plus embedded red marker pixels, and its leftmost column is a full
      vertical edge bar of a non-teal, non-red color.
    - The panels are isolated from each other by at least one background cell.

    Rule:
    - Count the red marker pixels in each panel and select the unique panel with the largest total.
    - In the selected panel, count red pixels row by row; a row with k red pixels is repeated horizontally k times.
    - Replace red markers with teal before repeating, keep the edge-bar column inside every repeated copy, and pad
      shorter stretched rows on the right with grey background so all output rows share the maximum width.

    Key insight:
    - The same red markers encode two levels of information: globally they pick the target panel, locally they set
      the per-row stretch factor, then they disappear because they were metadata rather than visible content.
    """
    g = np.asarray(grid, dtype=int)
    assert g.ndim == 2 and g.size > 0, "expected a non-empty 2D grid"

    bg = _most_common_color(g)
    assert bg == BACKGROUND, "this puzzle uses grey as the background"
    boxes = _solid_panel_boxes(g, bg)
    assert len(boxes) == 3, "expected exactly three panels"

    scored: list[tuple[int, tuple[int, int, int, int]]] = []
    for box in boxes:
        panel = _crop(g, box)
        _validate_panel(panel)
        red_count = int(np.count_nonzero(panel == RED))
        assert red_count > 0, "every panel must contain at least one red marker"
        scored.append((red_count, box))

    best = max(score for score, _box in scored)
    winners = [box for score, box in scored if score == best]
    assert len(winners) == 1, "the most-red panel must be unique"
    return _stretch_panel(_crop(g, winners[0]), bg=bg)


def _sample_counts(
    rng: random.Random,
    length: int,
    total: int,
    lo: int,
    hi: int,
    *,
    require_variation: bool,
) -> list[int] | None:
    for _ in range(200):
        remaining = total
        counts = []
        for i in range(length):
            min_rest = (length - i - 1) * lo
            max_rest = (length - i - 1) * hi
            low = max(lo, remaining - max_rest)
            high = min(hi, remaining - min_rest)
            if low > high:
                break
            value = rng.randint(low, high)
            counts.append(value)
            remaining -= value
        if len(counts) != length or remaining != 0:
            continue
        if require_variation and len(set(counts)) < 2:
            continue
        return counts
    return None


def _choose_dims(rng: random.Random, winner: int) -> list[tuple[int, int]] | None:
    mode = rng.choice(("free", "winner_smallest", "winner_largest"))
    for _ in range(200):
        dims = [(rng.randint(5, 8), rng.randint(4, 6)) for _ in range(3)]
        areas = [h * w for h, w in dims]
        if mode == "winner_smallest":
            if areas[winner] != min(areas) or areas.count(min(areas)) != 1:
                continue
        elif mode == "winner_largest":
            if areas[winner] != max(areas) or areas.count(max(areas)) != 1:
                continue
        return dims
    return None


def _choose_row_counts(
    rng: random.Random,
    dims: list[tuple[int, int]],
    winner: int,
) -> list[list[int]] | None:
    other = [i for i in range(3) if i != winner]
    runner = rng.choice(other)
    third = other[0] if other[1] == runner else other[1]

    winner_h = dims[winner][0]
    runner_h = dims[runner][0]
    third_h = dims[third][0]

    for _ in range(300):
        winner_counts = [rng.randint(1, 3) for _ in range(winner_h)]
        if len(set(winner_counts)) < 2:
            continue
        winner_total = sum(winner_counts)
        if winner_total <= runner_h:
            continue

        runner_low = runner_h
        runner_high = min(3 * runner_h, winner_total - 1)
        if runner_low > runner_high:
            continue
        near_low = max(runner_low, winner_total - 3)
        if near_low > runner_high:
            continue
        runner_total = rng.randint(near_low, runner_high)
        runner_counts = _sample_counts(rng, runner_h, runner_total, 1, 3, require_variation=rng.random() < 0.6)
        if runner_counts is None:
            continue

        third_lo = 1 if rng.random() < 0.45 else 0
        third_low_total = max(1, third_lo * third_h)
        third_high_total = min(3 * third_h, runner_total - 1)
        if third_low_total > third_high_total:
            continue
        third_total = rng.randint(third_low_total, third_high_total)
        third_counts = _sample_counts(rng, third_h, third_total, third_lo, 3, require_variation=False)
        if third_counts is None:
            continue

        counts: list[list[int] | None] = [None, None, None]
        counts[winner] = winner_counts
        counts[runner] = runner_counts
        counts[third] = third_counts
        assert all(c is not None for c in counts)
        return [list(c) for c in counts]  # type: ignore[arg-type]
    return None


def _build_panel(rng: random.Random, h: int, w: int, bar_color: int, row_counts: list[int]) -> np.ndarray:
    panel = np.full((h, w), TEAL, dtype=int)
    panel[:, 0] = bar_color
    for r, red_count in enumerate(row_counts):
        if red_count == 0:
            continue
        cols = rng.sample(list(range(1, w)), red_count)
        panel[r, cols] = RED
    return panel


def _place_boxes(
    rng: random.Random,
    grid_shape: tuple[int, int],
    dims: list[tuple[int, int]],
) -> list[tuple[int, int, int, int]] | None:
    h, w = grid_shape
    order = sorted(range(len(dims)), key=lambda i: dims[i][0] * dims[i][1], reverse=True)
    placed: list[tuple[int, int, int, int] | None] = [None] * len(dims)

    for idx in order:
        ph, pw = dims[idx]
        found = None
        for _ in range(400):
            r0 = rng.randint(0, h - ph)
            c0 = rng.randint(0, w - pw)
            cand = (r0, c0, r0 + ph - 1, c0 + pw - 1)
            ok = True
            for box in placed:
                if box is None:
                    continue
                br0, bc0, br1, bc1 = box
                if not (cand[2] + 1 < br0 or br1 + 1 < cand[0] or cand[3] + 1 < bc0 or bc1 + 1 < cand[1]):
                    ok = False
                    break
            if ok:
                found = cand
                break
        if found is None:
            return None
        placed[idx] = found

    return [box for box in placed if box is not None]


def _manual_expected(panel: np.ndarray) -> np.ndarray:
    row_factors = [int(np.count_nonzero(row == RED)) for row in panel]
    h, w = panel.shape
    out = np.full((h, w * max(row_factors)), BACKGROUND, dtype=int)
    clean = np.where(panel == RED, TEAL, panel)
    for r, k in enumerate(row_factors):
        out[r, : w * k] = np.tile(clean[r], k)
    return out


def _semantic_ok(
    grid: np.ndarray,
    panels: list[np.ndarray],
    boxes: list[tuple[int, int, int, int]],
    winner: int,
    row_counts: list[list[int]],
) -> bool:
    if grid.ndim != 2 or max(grid.shape) > 30:
        return False
    if _most_common_color(grid) != BACKGROUND:
        return False

    found_boxes = _solid_panel_boxes(grid, BACKGROUND)
    if len(found_boxes) != 3 or sorted(found_boxes) != sorted(boxes):
        return False

    red_totals = [int(np.count_nonzero(panel == RED)) for panel in panels]
    if red_totals[winner] != max(red_totals) or red_totals.count(max(red_totals)) != 1:
        return False
    if len(set(row_counts[winner])) < 2:
        return False
    if not all(1 <= k <= 3 for k in row_counts[winner]):
        return False

    expected = _manual_expected(panels[winner])
    try:
        solved = main(grid)
    except Exception:
        return False
    return np.array_equal(solved, expected)


def _gen_attempt(rng: random.Random) -> np.ndarray | None:
    winner = rng.randrange(3)
    dims = _choose_dims(rng, winner)
    if dims is None:
        return None

    row_counts = _choose_row_counts(rng, dims, winner)
    if row_counts is None:
        return None

    bars = [rng.choice(BAR_COLORS) for _ in range(3)]
    panels = [
        _build_panel(rng, dims[i][0], dims[i][1], bars[i], row_counts[i])
        for i in range(3)
    ]

    min_h = max(18, max(h for h, _w in dims) + 6)
    min_w = max(18, max(w for _h, w in dims) + 9)
    grid_shape = (rng.randint(min_h, 24), rng.randint(min_w, 24))
    boxes = _place_boxes(rng, grid_shape, dims)
    if boxes is None:
        return None

    grid = np.full(grid_shape, BACKGROUND, dtype=int)
    for panel, box in zip(panels, boxes):
        r0, c0, r1, c1 = box
        grid[r0 : r1 + 1, c0 : c1 + 1] = panel

    if not _semantic_ok(grid, panels, boxes, winner, row_counts):
        return None
    return grid


def generate_input() -> np.ndarray:
    rng = random.Random()
    for _ in range(400):
        grid = _gen_attempt(rng)
        if grid is not None:
            return grid
    raise RuntimeError("failed to generate a valid input for puzzle 6")
