from __future__ import annotations

from collections import Counter
from dataclasses import dataclass
import random

import numpy as np


@dataclass(frozen=True)
class Rect:
    color: int
    r0: int
    c0: int
    r1: int
    c1: int

    @property
    def area(self) -> int:
        return (self.r1 - self.r0 + 1) * (self.c1 - self.c0 + 1)


def _periodic_canvas(tile: np.ndarray, shape: tuple[int, int]) -> np.ndarray:
    rows = np.arange(int(shape[0])) % int(tile.shape[0])
    cols = np.arange(int(shape[1])) % int(tile.shape[1])
    return tile[rows[:, None], cols[None, :]]


def _divisors(n: int) -> list[int]:
    return [d for d in range(1, int(n) + 1) if int(n) % d == 0]


def _minimal_tile(tile: np.ndarray) -> np.ndarray:
    h, w = map(int, tile.shape)
    for th in _divisors(h):
        for tw in _divisors(w):
            candidate = tile[:th, :tw]
            if np.array_equal(tile, _periodic_canvas(candidate, (h, w))):
                return candidate.copy()
    return tile.copy()


def _mode_tile(g: np.ndarray, period_h: int, period_w: int) -> np.ndarray | None:
    tile = np.empty((int(period_h), int(period_w)), dtype=int)
    for rr in range(int(period_h)):
        for cc in range(int(period_w)):
            values = [int(v) for v in g[rr::period_h, cc::period_w].ravel()]
            counts = Counter(values)
            top = max(counts.values())
            winners = [color for color, count in counts.items() if count == top]
            if len(winners) != 1:
                return None
            tile[rr, cc] = int(winners[0])
    return tile


def _same_color_components(g: np.ndarray, active: np.ndarray) -> list[Rect]:
    h, w = map(int, g.shape)
    seen = np.zeros((h, w), dtype=bool)
    components: list[Rect] = []

    for sr, sc in np.argwhere(active):
        sr, sc = int(sr), int(sc)
        if seen[sr, sc]:
            continue
        color = int(g[sr, sc])
        stack = [(sr, sc)]
        seen[sr, sc] = True
        coords: list[tuple[int, int]] = []

        while stack:
            r, c = stack.pop()
            coords.append((r, c))
            for dr, dc in ((-1, 0), (1, 0), (0, -1), (0, 1)):
                nr, nc = r + dr, c + dc
                if not (0 <= nr < h and 0 <= nc < w):
                    continue
                if seen[nr, nc] or not active[nr, nc] or int(g[nr, nc]) != color:
                    continue
                seen[nr, nc] = True
                stack.append((nr, nc))

        rows = [r for r, _c in coords]
        cols = [c for _r, c in coords]
        components.append(Rect(color, min(rows), min(cols), max(rows), max(cols)))

    return components


def _is_solid_rectangle(g: np.ndarray, rect: Rect) -> bool:
    patch = g[rect.r0 : rect.r1 + 1, rect.c0 : rect.c1 + 1]
    return int(patch.size) == int(rect.area) and bool(np.all(patch == int(rect.color)))


def _single_solid_rectangle_colors(g: np.ndarray) -> set[int]:
    by_color: dict[int, list[Rect]] = {}
    active = np.ones(g.shape, dtype=bool)
    for comp in _same_color_components(g, active):
        by_color.setdefault(int(comp.color), []).append(comp)

    out: set[int] = set()
    for color, comps in by_color.items():
        if len(comps) == 1 and _is_solid_rectangle(g, comps[0]):
            out.add(int(color))
    return out


def _mismatch_rectangles(
    g: np.ndarray,
    expected: np.ndarray,
    tile_colors: set[int],
) -> list[Rect] | None:
    mismatch = g != expected
    if not bool(mismatch.any()):
        return None

    rects: list[Rect] = []
    seen_colors: set[int] = set()
    for comp in _same_color_components(g, mismatch):
        color = int(comp.color)
        if color in tile_colors or color in seen_colors:
            return None
        if not _is_solid_rectangle(g, comp):
            return None
        if not bool(np.all(mismatch[comp.r0 : comp.r1 + 1, comp.c0 : comp.c1 + 1])):
            return None
        rects.append(comp)
        seen_colors.add(color)

    largest = max(rect.area for rect in rects)
    if sum(1 for rect in rects if rect.area == largest) != 1:
        return None
    return rects


def _infer_background(g: np.ndarray) -> tuple[np.ndarray, list[Rect]]:
    h, w = map(int, g.shape)
    single_rect_colors = _single_solid_rectangle_colors(g)
    max_period_h = min(8, h)
    max_period_w = min(8, w)

    for strict in (True, False):
        candidates: list[tuple[int, int, int, int, np.ndarray, list[Rect]]] = []
        for ph in range(1, max_period_h + 1):
            for pw in range(1, max_period_w + 1):
                raw_tile = _mode_tile(g, ph, pw)
                if raw_tile is None:
                    continue
                tile = _minimal_tile(raw_tile)
                tile_colors = {int(v) for v in tile.ravel()}
                if strict and (tile_colors & single_rect_colors):
                    continue

                expected = _periodic_canvas(tile, (h, w))
                rects = _mismatch_rectangles(g, expected, tile_colors)
                if rects is None:
                    continue

                rect_colors = {int(rect.color) for rect in rects}
                if strict and not single_rect_colors.issubset(rect_colors):
                    continue

                mismatch_area = sum(rect.area for rect in rects)
                candidates.append(
                    (mismatch_area, int(tile.size), int(tile.shape[0]), int(tile.shape[1]), tile, rects)
                )

        if candidates:
            candidates.sort(key=lambda item: (item[0], item[1], item[2], item[3]))
            _area, _size, _th, _tw, tile, rects = candidates[0]
            return tile, rects

    raise AssertionError("could not infer a periodic background and rectangular masks")


def main(grid: np.ndarray) -> np.ndarray:
    """Restore the largest solid rectangle with the inferred periodic background.

    Input structure:
    - The grid is covered by a small row/column-periodic colored tile.
    - Several non-background colors form disjoint solid axis-aligned rectangles
      drawn over the periodic pattern.
    - Rectangle colors do not appear in the underlying periodic tile.

    Rule:
    - Infer the smallest repeated background tile from the visible cells.
    - Treat all non-tile-colored solid rectangles as occluding masks.
    - Find the unique largest rectangle by area.
    - Replace only that rectangle with the corresponding periodic background
      cells, leaving every smaller rectangle unchanged.

    Key insight:
    The smaller rectangles are not removed; they are evidence for separating
    occluder colors from the periodic background before restoring only the
    largest masked region.
    """
    g = np.asarray(grid, dtype=int)
    assert g.ndim == 2, "expected a 2D grid"

    tile, rects = _infer_background(g)
    background = _periodic_canvas(tile, g.shape)
    largest = max(rects, key=lambda rect: rect.area)

    out = g.copy()
    out[largest.r0 : largest.r1 + 1, largest.c0 : largest.c1 + 1] = background[
        largest.r0 : largest.r1 + 1, largest.c0 : largest.c1 + 1
    ]
    return out


# --------------- generator ---------------

ARC_COLORS = tuple(range(10))
PERIOD_CHOICES = (
    (1, 1),
    (1, 2),
    (2, 1),
    (2, 2),
    (2, 3),
    (3, 2),
    (3, 3),
    (2, 4),
    (4, 2),
    (3, 4),
    (4, 3),
)
RECT_COUNT_RANGE = (2, 5)
MAX_TILE_COLORS = 5
GENERATION_ATTEMPTS = 400
PLACEMENT_ATTEMPTS = 500


def _make_tile(period_h: int, period_w: int, colors: list[int]) -> np.ndarray | None:
    h, w = int(period_h), int(period_w)
    tile = np.asarray(np.random.choice(colors, size=(h, w)), dtype=int)
    cells = [(r, c) for r in range(h) for c in range(w)]
    random.shuffle(cells)
    for color, (r, c) in zip(colors, cells):
        tile[r, c] = int(color)

    if _minimal_tile(tile).shape != (h, w):
        return None
    return tile


def _sample_rect_size(height: int, width: int) -> tuple[int, int]:
    mode = random.choice(("wide", "tall", "block"))
    max_h = max(2, min(8, int(height) // 2))
    max_w = max(2, min(8, int(width) // 2))

    if mode == "wide":
        rh = random.randint(2, min(4, max_h))
        rw = random.randint(4, max_w) if max_w >= 4 else random.randint(2, max_w)
    elif mode == "tall":
        rh = random.randint(4, max_h) if max_h >= 4 else random.randint(2, max_h)
        rw = random.randint(2, min(4, max_w))
    else:
        rh = random.randint(2, max_h)
        rw = random.randint(2, max_w)
    return int(rh), int(rw)


def _place_rectangles(height: int, width: int, count: int) -> list[Rect] | None:
    occupied = np.zeros((int(height), int(width)), dtype=bool)
    rects: list[Rect] = []
    used_areas: set[int] = set()

    for _ in range(int(count)):
        placed = False
        for _attempt in range(PLACEMENT_ATTEMPTS):
            rh, rw = _sample_rect_size(height, width)
            area = int(rh * rw)
            if area in used_areas:
                continue

            r0 = random.randint(0, int(height) - rh)
            c0 = random.randint(0, int(width) - rw)
            r1, c1 = r0 + rh - 1, c0 + rw - 1

            pr0 = max(0, r0 - 1)
            pc0 = max(0, c0 - 1)
            pr1 = min(int(height) - 1, r1 + 1)
            pc1 = min(int(width) - 1, c1 + 1)
            if bool(occupied[pr0 : pr1 + 1, pc0 : pc1 + 1].any()):
                continue

            occupied[r0 : r1 + 1, c0 : c1 + 1] = True
            rects.append(Rect(-1, r0, c0, r1, c1))
            used_areas.add(area)
            placed = True
            break

        if not placed:
            return None
    return rects


def _background_wins_modes(rects: list[Rect], shape: tuple[int, int], tile_shape: tuple[int, int]) -> bool:
    height, width = map(int, shape)
    ph, pw = map(int, tile_shape)
    covered: dict[tuple[int, int], list[int]] = {(r, c): [] for r in range(ph) for c in range(pw)}
    total = {(r, c): 0 for r in range(ph) for c in range(pw)}

    for r in range(height):
        for c in range(width):
            total[(r % ph, c % pw)] += 1

    for rect in rects:
        per_residue = {(r, c): 0 for r in range(ph) for c in range(pw)}
        for r in range(rect.r0, rect.r1 + 1):
            for c in range(rect.c0, rect.c1 + 1):
                per_residue[(r % ph, c % pw)] += 1
        for key, amount in per_residue.items():
            covered[key].append(int(amount))

    for key in total:
        object_total = sum(covered[key])
        visible_background = total[key] - object_total
        if visible_background < 2:
            return False
        if covered[key] and visible_background <= max(covered[key]):
            return False
    return True


def _gen_attempt() -> np.ndarray | None:
    period_h, period_w = random.choice(PERIOD_CHOICES)
    min_h = max(10, int(period_h) * 4)
    min_w = max(10, int(period_w) * 4)
    if min_h > 30 or min_w > 30:
        return None
    height = random.randint(min_h, 30)
    width = random.randint(min_w, 30)

    rect_count = random.randint(*RECT_COUNT_RANGE)
    tile_cells = int(period_h * period_w)
    if tile_cells == 1:
        bg_count = 1
    else:
        bg_count = random.randint(2, min(MAX_TILE_COLORS, tile_cells, 10 - rect_count))

    palette = random.sample(list(ARC_COLORS), bg_count + rect_count)
    bg_colors = [int(c) for c in palette[:bg_count]]
    rect_colors = [int(c) for c in palette[bg_count:]]

    tile = _make_tile(period_h, period_w, bg_colors)
    if tile is None:
        return None

    plain_background = _periodic_canvas(tile, (height, width))
    rects_without_color = _place_rectangles(height, width, rect_count)
    if rects_without_color is None:
        return None

    random.shuffle(rect_colors)
    rects = [
        Rect(color, rect.r0, rect.c0, rect.r1, rect.c1)
        for color, rect in zip(rect_colors, rects_without_color)
    ]
    if not _background_wins_modes(rects, (height, width), tile.shape):
        return None

    grid = plain_background.copy()
    for rect in rects:
        grid[rect.r0 : rect.r1 + 1, rect.c0 : rect.c1 + 1] = int(rect.color)

    largest = max(rects, key=lambda rect: rect.area)
    expected = grid.copy()
    expected[largest.r0 : largest.r1 + 1, largest.c0 : largest.c1 + 1] = plain_background[
        largest.r0 : largest.r1 + 1, largest.c0 : largest.c1 + 1
    ]

    try:
        actual = main(grid)
    except Exception:
        return None
    if np.array_equal(actual, expected):
        return grid
    return None


def generate_input() -> np.ndarray:
    for _ in range(GENERATION_ATTEMPTS):
        grid = _gen_attempt()
        if grid is not None:
            return grid
    raise RuntimeError("failed to generate valid periodic-mask input")
