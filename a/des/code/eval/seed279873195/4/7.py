from __future__ import annotations

import random
from collections import Counter

import numpy as np


BLACK = 0
GREY = 5
COLOR_POOL = [1, 2, 3, 4, 6, 7, 8, 9]


def _neighbors4(r: int, c: int, h: int, w: int) -> list[tuple[int, int]]:
    out: list[tuple[int, int]] = []
    if r > 0:
        out.append((r - 1, c))
    if r + 1 < h:
        out.append((r + 1, c))
    if c > 0:
        out.append((r, c - 1))
    if c + 1 < w:
        out.append((r, c + 1))
    return out


def _components(mask: np.ndarray) -> list[list[tuple[int, int]]]:
    h, w = mask.shape
    seen = np.zeros((h, w), dtype=bool)
    comps: list[list[tuple[int, int]]] = []
    for r in range(h):
        for c in range(w):
            if not mask[r, c] or seen[r, c]:
                continue
            stack = [(r, c)]
            seen[r, c] = True
            pts: list[tuple[int, int]] = []
            while stack:
                rr, cc = stack.pop()
                pts.append((rr, cc))
                for nr, nc in _neighbors4(rr, cc, h, w):
                    if mask[nr, nc] and not seen[nr, nc]:
                        seen[nr, nc] = True
                        stack.append((nr, nc))
            comps.append(pts)
    return comps


def _bbox(points: list[tuple[int, int]]) -> tuple[int, int, int, int]:
    rows = [r for r, _ in points]
    cols = [c for _, c in points]
    return min(rows), min(cols), max(rows), max(cols)


def _ordered_border(r0: int, c0: int, r1: int, c1: int) -> list[tuple[int, int]]:
    coords: list[tuple[int, int]] = []
    for c in range(c0, c1 + 1):
        coords.append((r0, c))
    for r in range(r0 + 1, r1):
        coords.append((r, c1))
    for c in range(c1, c0 - 1, -1):
        coords.append((r1, c))
    for r in range(r1 - 1, r0, -1):
        coords.append((r, c0))
    return coords


def _draw_solid_border(grid: np.ndarray, box: tuple[int, int, int, int], color: int) -> None:
    h, w = grid.shape
    for r, c in _ordered_border(*box):
        if 0 <= r < h and 0 <= c < w:
            grid[r, c] = int(color)


def _draw_dashed_border(grid: np.ndarray, box: tuple[int, int, int, int], color: int) -> None:
    h, w = grid.shape
    for idx, (r, c) in enumerate(_ordered_border(*box)):
        if idx % 2 == 0 and 0 <= r < h and 0 <= c < w:
            grid[r, c] = int(color)


def _frame_bbox(points: list[tuple[int, int]]) -> tuple[int, int, int, int] | None:
    r0, c0, r1, c1 = _bbox(points)
    if r1 - r0 < 2 or c1 - c0 < 2:
        return None
    height = r1 - r0 + 1
    width = c1 - c0 + 1
    expected = 2 * (height + width) - 4
    if len(points) != expected:
        return None
    pts = set(points)
    for c in range(c0, c1 + 1):
        if (r0, c) not in pts or (r1, c) not in pts:
            return None
    for r in range(r0, r1 + 1):
        if (r, c0) not in pts or (r, c1) not in pts:
            return None
    for r in range(r0 + 1, r1):
        for c in range(c0 + 1, c1):
            if (r, c) in pts:
                return None
    return r0, c0, r1, c1


def _find_frames(grid: np.ndarray) -> list[tuple[int, int, int, int]]:
    frames: list[tuple[int, int, int, int]] = []
    for pts in _components(grid == GREY):
        box = _frame_bbox(pts)
        if box is not None:
            frames.append(box)
    assert frames, "expected at least one grey rectangular frame"
    return sorted(frames)


def _find_mapping_table(grid: np.ndarray) -> tuple[list[tuple[int, int]], tuple[int, int, int, int]]:
    h, w = grid.shape
    candidates: list[tuple[int, int, int, int]] = []
    mask = (grid != BLACK) & (grid != GREY)
    for pts in _components(mask):
        r0, c0, r1, c1 = _bbox(pts)
        height = r1 - r0 + 1
        width = c1 - c0 + 1
        if width != 2 or height < 4:
            continue
        if r0 > 2 or c1 < w - 2:
            continue
        ptset = set(pts)
        if any((r, c) not in ptset for r in range(r0, r1 + 1) for c in range(c0, c1 + 1)):
            continue
        candidates.append((r0, c0, r1, c1))
    assert len(candidates) == 1, f"expected one mapping table, found {len(candidates)}"
    r0, c0, r1, c1 = candidates[0]
    rows = [(int(grid[r, c0]), int(grid[r, c0 + 1])) for r in range(r0, r1 + 1)]
    lookups = [src for src, _ in rows]
    assert len(set(lookups)) == len(lookups), "lookup colors must be unique"
    return rows, (r0, c0, r1, c1)


def main(grid: np.ndarray) -> np.ndarray:
    """
    Recolor each grey rectangular border via a top-right color table.

    The dominant interior marker color is looked up in the table to obtain the
    border color. The number of marker pixels selects solid, dashed, or double
    border rendering. Markers and the table are removed in the output.
    """
    g = np.asarray(grid, dtype=int)
    assert g.ndim == 2, "expected a 2D grid"

    mapping_rows, _ = _find_mapping_table(g)
    priority = {src: idx for idx, (src, _) in enumerate(mapping_rows)}
    target_by_source = {src: tgt for src, tgt in mapping_rows}

    out = np.zeros_like(g)
    for r0, c0, r1, c1 in _find_frames(g):
        interior = g[r0 + 1 : r1, c0 + 1 : c1]
        markers = [int(v) for v in interior.ravel() if int(v) not in (BLACK, GREY)]
        assert 1 <= len(markers) <= 3, "each frame must contain 1 to 3 markers"
        counts = Counter(markers)
        top_count = max(counts.values())
        winners = [color for color, count in counts.items() if count == top_count]
        chosen = min(winners, key=lambda color: priority[color])
        draw_color = int(target_by_source[chosen])

        if len(markers) == 1:
            _draw_solid_border(out, (r0, c0, r1, c1), draw_color)
        elif len(markers) == 2:
            _draw_dashed_border(out, (r0, c0, r1, c1), draw_color)
        else:
            _draw_solid_border(out, (r0, c0, r1, c1), draw_color)
            _draw_solid_border(out, (r0 - 1, c0 - 1, r1 + 1, c1 + 1), draw_color)
    return out


GRID_H_RANGE = (18, 26)
GRID_W_RANGE = (20, 30)
MAP_ROWS_RANGE = (4, 6)
REGION_COUNT_RANGE = (3, 5)
INNER_H_RANGE = (2, 6)
INNER_W_RANGE = (2, 7)
PLACEMENT_TRIES = 200
GEN_RETRIES = 300


def _random_derangement(values: list[int]) -> list[int]:
    assert len(values) >= 2
    for _ in range(100):
        out = values[:]
        random.shuffle(out)
        if all(a != b for a, b in zip(values, out)):
            return out
    raise RuntimeError("failed to build derangement")


def _sample_styles(region_count: int) -> list[int]:
    styles = [1, 2, 3] * ((region_count + 2) // 3)
    random.shuffle(styles)
    return styles[:region_count]


def _sample_marker_colors(style: int, lookups: list[int]) -> tuple[list[int], int]:
    if style == 1:
        chosen = random.choice(lookups)
        return [chosen], chosen

    if style == 2:
        if random.random() < 0.5 or len(lookups) < 2:
            chosen = random.choice(lookups)
            return [chosen, chosen], chosen
        winner_idx = random.randrange(len(lookups) - 1)
        chosen = lookups[winner_idx]
        other = random.choice(lookups[winner_idx + 1 :])
        colors = [chosen, other]
        random.shuffle(colors)
        return colors, chosen

    modes = ["all_same", "two_plus_one", "tie"]
    random.shuffle(modes)
    for mode in modes:
        if mode == "all_same":
            chosen = random.choice(lookups)
            return [chosen, chosen, chosen], chosen
        if mode == "two_plus_one" and len(lookups) >= 2:
            chosen = random.choice(lookups)
            other = random.choice([c for c in lookups if c != chosen])
            colors = [chosen, chosen, other]
            random.shuffle(colors)
            return colors, chosen
        if mode == "tie" and len(lookups) >= 3:
            winner_idx = random.randrange(len(lookups) - 2)
            chosen = lookups[winner_idx]
            others = random.sample(lookups[winner_idx + 1 :], 2)
            colors = [chosen, others[0], others[1]]
            random.shuffle(colors)
            return colors, chosen
    chosen = random.choice(lookups)
    return [chosen, chosen, chosen], chosen


def _reserve_box(
    reserve: np.ndarray, r0: int, c0: int, r1: int, c1: int, margin: int
) -> tuple[int, int, int, int]:
    h, w = reserve.shape
    rr0 = max(0, r0 - margin)
    cc0 = max(0, c0 - margin)
    rr1 = min(h - 1, r1 + margin)
    cc1 = min(w - 1, c1 + margin)
    return rr0, cc0, rr1, cc1


def _pick_cells(
    inner_h: int, inner_w: int, count: int
) -> list[tuple[int, int]]:
    cells = [(r, c) for r in range(inner_h) for c in range(inner_w)]
    return random.sample(cells, count)


def _render_expected(
    shape: tuple[int, int], specs: list[dict[str, object]]
) -> np.ndarray:
    out = np.zeros(shape, dtype=int)
    for spec in specs:
        box = spec["box"]
        color = int(spec["target"])
        style = int(spec["style"])
        if style == 1:
            _draw_solid_border(out, box, color)
        elif style == 2:
            _draw_dashed_border(out, box, color)
        else:
            r0, c0, r1, c1 = box
            _draw_solid_border(out, box, color)
            _draw_solid_border(out, (r0 - 1, c0 - 1, r1 + 1, c1 + 1), color)
    return out


def _gen_attempt() -> np.ndarray | None:
    h = random.randint(*GRID_H_RANGE)
    w = random.randint(*GRID_W_RANGE)
    grid = np.zeros((h, w), dtype=int)
    reserve = np.zeros((h, w), dtype=bool)

    map_rows = random.randint(*MAP_ROWS_RANGE)
    lookups = random.sample(COLOR_POOL, map_rows)
    targets = _random_derangement(lookups)
    mapping = list(zip(lookups, targets))
    target_by_source = {src: tgt for src, tgt in mapping}

    map_r0 = random.randint(0, 1)
    map_c0 = w - 2 - random.randint(0, 1)
    map_r1 = map_r0 + map_rows - 1
    map_c1 = map_c0 + 1
    if map_r1 >= h:
        return None
    for idx, (src, tgt) in enumerate(mapping):
        grid[map_r0 + idx, map_c0] = int(src)
        grid[map_r0 + idx, map_c0 + 1] = int(tgt)
    rr0, cc0, rr1, cc1 = _reserve_box(reserve, map_r0, map_c0, map_r1, map_c1, margin=1)
    cc0 = max(0, cc0 - 1)
    reserve[rr0 : rr1 + 1, cc0 : cc1 + 1] = True

    region_count = random.randint(*REGION_COUNT_RANGE)
    styles = _sample_styles(region_count)
    specs: list[dict[str, object]] = []

    for style in styles:
        placed = False
        for _ in range(PLACEMENT_TRIES):
            inner_h = random.randint(*INNER_H_RANGE)
            inner_w = random.randint(*INNER_W_RANGE)
            if inner_h * inner_w < style:
                continue
            outer_h = inner_h + 2
            outer_w = inner_w + 2
            max_r0 = h - outer_h - 2
            max_c0 = w - outer_w - 2
            if max_r0 < 2 or max_c0 < 2:
                return None
            r0 = random.randint(2, max_r0)
            c0 = random.randint(2, max_c0)
            r1 = r0 + outer_h - 1
            c1 = c0 + outer_w - 1
            rr0, cc0, rr1, cc1 = _reserve_box(reserve, r0, c0, r1, c1, margin=2)
            if reserve[rr0 : rr1 + 1, cc0 : cc1 + 1].any():
                continue

            marker_colors, winner = _sample_marker_colors(style, lookups)
            marker_cells = _pick_cells(inner_h, inner_w, style)

            grid[r0, c0 : c1 + 1] = GREY
            grid[r1, c0 : c1 + 1] = GREY
            grid[r0 : r1 + 1, c0] = GREY
            grid[r0 : r1 + 1, c1] = GREY
            for (dr, dc), color in zip(marker_cells, marker_colors):
                grid[r0 + 1 + dr, c0 + 1 + dc] = int(color)

            reserve[rr0 : rr1 + 1, cc0 : cc1 + 1] = True
            specs.append(
                {
                    "box": (r0, c0, r1, c1),
                    "style": style,
                    "winner": winner,
                    "target": int(target_by_source[winner]),
                }
            )
            placed = True
            break
        if not placed:
            return None

    expected = _render_expected((h, w), specs)
    try:
        actual = main(grid)
    except Exception:
        return None
    if np.array_equal(actual, expected) and not np.array_equal(actual, grid):
        return grid
    return None


def generate_input() -> np.ndarray:
    for _ in range(GEN_RETRIES):
        grid = _gen_attempt()
        if grid is not None:
            return grid
    raise RuntimeError("failed to generate valid input")
