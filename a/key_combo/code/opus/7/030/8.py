from __future__ import annotations

from collections import Counter, deque
import random

import numpy as np


BG = 0
FG = 5
HOLE_TO_COLOR = {1: 1, 2: 2, 3: 3}

GEN_HEIGHT_RANGE = (18, 30)
GEN_WIDTH_RANGE = (13, 18)
GEN_ATTEMPTS = 500
GEN_COUNTS = (1, 2, 3)
GEN_HOLE_MARGIN = 2
GEN_OBJECT_HEIGHT_RANGE = (5, 8)
GEN_OBJECT_WIDTH_RANGE = (6, 8)
GEN_SIDE_MARGIN = 1
GEN_TOP_MARGIN_RANGE = (0, 2)
GEN_BOTTOM_MARGIN_RANGE = (0, 2)
GEN_GAP_RANGE = (1, 4)
GEN_HOLE_SHAPES = ((1, 1), (1, 2), (2, 1))


def _components(mask: np.ndarray) -> list[np.ndarray]:
    h, w = mask.shape
    seen = np.zeros_like(mask, dtype=bool)
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
            comps.append(np.array(coords, dtype=int))
    return comps


def _bbox(coords: np.ndarray) -> tuple[int, int, int, int]:
    r0, c0 = coords.min(axis=0)
    r1, c1 = coords.max(axis=0)
    return int(r0), int(c0), int(r1), int(c1)


def _hole_count(mask: np.ndarray) -> int:
    padded_bg = np.pad(~mask.astype(bool), 1, constant_values=True)
    comps = _components(padded_bg)
    count = 0
    for comp in comps:
        if any(int(r) in (0, padded_bg.shape[0] - 1) or int(c) in (0, padded_bg.shape[1] - 1) for r, c in comp):
            continue
        count += 1
    return count


def _band_bounds(anchors: list[int], height: int) -> list[tuple[int, int]]:
    bounds = []
    for i, row in enumerate(anchors):
        start = 0 if i == 0 else (int(anchors[i - 1]) + int(row)) // 2 + 1
        end = height - 1 if i == len(anchors) - 1 else (int(row) + int(anchors[i + 1])) // 2
        bounds.append((int(start), int(end)))
    return bounds


def main(grid: np.ndarray) -> np.ndarray:
    """Count each gray object's enclosed holes and use that count to color its horizontal band frame.

    Input structure
    - A black background with several gray 4-connected foreground objects.
    - Each object's bounding-box row span is disjoint from the others.
    - Every gray object contains one to three enclosed black holes.

    Rule
    - For each gray object, count its enclosed 4-connected black-hole components.
    - Map hole counts to frame colors: 1 -> blue, 2 -> red, 3 -> green.
    - Compute each object's anchor row as the upper middle row of its bbox, sort anchors top-to-bottom,
      split the grid into horizontal bands at anchor-row midpoints, then draw that band's frame:
      full-width anchor row, left and right side columns across the band, plus the top or bottom edge
      when the band touches the grid border.
    - Erase the original gray objects; keep only the band frames on black.

    Key insight
    - The objects contribute only two signals: hole count chooses the color, while vertical position chooses
      the band's extent. The output discards the original shapes completely.
    """
    g = np.asarray(grid, dtype=int)
    assert g.ndim == 2
    h, w = map(int, g.shape)
    assert h > 0 and w > 1

    comps = _components(g == FG)
    assert comps

    objects: list[tuple[int, int]] = []
    for comp in comps:
        r0, c0, r1, c1 = _bbox(comp)
        local_mask = (g[r0 : r1 + 1, c0 : c1 + 1] == FG)
        holes = _hole_count(local_mask)
        assert holes in HOLE_TO_COLOR
        anchor = int(r0 + (local_mask.shape[0] - 1) // 2)
        objects.append((anchor, int(HOLE_TO_COLOR[holes])))

    objects.sort(key=lambda item: item[0])
    anchors = [anchor for anchor, _color in objects]
    bounds = _band_bounds(anchors, h)

    out = np.full_like(g, BG)
    for (anchor, color), (start, end) in zip(objects, bounds):
        out[start : end + 1, 0] = int(color)
        out[start : end + 1, w - 1] = int(color)
        out[anchor, :] = int(color)
        if start == 0:
            out[0, :] = int(color)
        if end == h - 1:
            out[h - 1, :] = int(color)

    return out


def _hole_options(height: int, width: int) -> list[tuple[tuple[int, int], ...]]:
    options: list[tuple[tuple[int, int], ...]] = []
    for hh, ww in GEN_HOLE_SHAPES:
        for r0 in range(GEN_HOLE_MARGIN, height - hh - GEN_HOLE_MARGIN + 1):
            for c0 in range(GEN_HOLE_MARGIN, width - ww - GEN_HOLE_MARGIN + 1):
                cells = tuple((r, c) for r in range(r0, r0 + hh) for c in range(c0, c0 + ww))
                options.append(cells)
    return options


def _holes_separated(cells: tuple[tuple[int, int], ...], used: tuple[tuple[int, int], ...]) -> bool:
    for r0, c0 in cells:
        for r1, c1 in used:
            if abs(int(r0) - int(r1)) + abs(int(c0) - int(c1)) < 2:
                return False
    return True


def _choose_holes(
    options: list[tuple[tuple[int, int], ...]],
    remaining: int,
    used: tuple[tuple[int, int], ...],
) -> list[tuple[tuple[int, int], ...]] | None:
    if remaining == 0:
        return []
    pool = options[:]
    random.shuffle(pool)
    for cells in pool:
        if not _holes_separated(cells, used):
            continue
        tail = _choose_holes(options, remaining - 1, used + cells)
        if tail is not None:
            return [cells] + tail
    return None


def _connected(mask: np.ndarray) -> bool:
    comps = _components(mask)
    return len(comps) == 1


def _try_remove_cells(mask: np.ndarray, cells: list[tuple[int, int]], holes: int) -> bool:
    trial = mask.copy()
    for r, c in cells:
        if not trial[r, c]:
            return False
        trial[r, c] = False
    if not _connected(trial):
        return False
    if _hole_count(trial) != holes:
        return False
    mask[:] = trial
    return True


def _random_notches(mask: np.ndarray, holes: int) -> bool:
    h, w = mask.shape
    removed = 0
    patterns: list[list[tuple[int, int]]] = []
    for c in range(1, w - 1):
        patterns.append([(0, c)])
        patterns.append([(h - 1, c)])
    for r in range(1, h - 1):
        patterns.append([(r, 0)])
        patterns.append([(r, w - 1)])
    for c in range(1, w - 2):
        patterns.append([(0, c), (0, c + 1)])
        patterns.append([(h - 1, c), (h - 1, c + 1)])
    for r in range(1, h - 2):
        patterns.append([(r, 0), (r + 1, 0)])
        patterns.append([(r, w - 1), (r + 1, w - 1)])
    patterns.extend(
        [
            [(0, 0)],
            [(0, w - 1)],
            [(h - 1, 0)],
            [(h - 1, w - 1)],
        ]
    )
    random.shuffle(patterns)
    target = random.randint(1, 3)
    for cells in patterns:
        if removed >= target:
            break
        if _try_remove_cells(mask, cells, holes):
            removed += 1
    return removed > 0


def _make_object(n_holes: int) -> np.ndarray | None:
    for _ in range(200):
        height = random.randint(*GEN_OBJECT_HEIGHT_RANGE)
        width = random.randint(*GEN_OBJECT_WIDTH_RANGE)
        options = _hole_options(height, width)
        holes = _choose_holes(options, n_holes, ())
        if holes is None:
            continue

        mask = np.ones((height, width), dtype=bool)
        for cells in holes:
            for r, c in cells:
                mask[r, c] = False

        if _hole_count(mask) != n_holes:
            continue
        if not _random_notches(mask, n_holes):
            continue
        if _hole_count(mask) != n_holes or not _connected(mask):
            continue

        obj = np.full((height, width), BG, dtype=int)
        obj[mask] = FG
        return obj
    return None


def _object_anchor(top: int, obj: np.ndarray) -> int:
    return int(top + (obj.shape[0] - 1) // 2)


def _expected_output(height: int, width: int, anchors: list[int], counts: list[int]) -> np.ndarray:
    out = np.full((height, width), BG, dtype=int)
    bounds = _band_bounds(anchors, height)
    for anchor, holes, (start, end) in zip(anchors, counts, bounds):
        color = HOLE_TO_COLOR[int(holes)]
        out[start : end + 1, 0] = int(color)
        out[start : end + 1, width - 1] = int(color)
        out[anchor, :] = int(color)
        if start == 0:
            out[0, :] = int(color)
        if end == height - 1:
            out[height - 1, :] = int(color)
    return out


def _background_is_majority(grid: np.ndarray) -> bool:
    counts = Counter(map(int, grid.ravel()))
    return counts[BG] > max((count for color, count in counts.items() if color != BG), default=0)


def _place_scene(order: list[int], objects: list[np.ndarray]) -> np.ndarray | None:
    widths = [obj.shape[1] for obj in objects]
    height_needed = 0
    top_margin = random.randint(*GEN_TOP_MARGIN_RANGE)
    bottom_margin = random.randint(*GEN_BOTTOM_MARGIN_RANGE)
    gaps = [random.randint(*GEN_GAP_RANGE) for _ in range(len(objects) - 1)]

    tops = [top_margin]
    for i in range(1, len(objects)):
        prev_bottom = tops[i - 1] + objects[i - 1].shape[0] - 1
        tops.append(prev_bottom + 1 + gaps[i - 1])
    height_needed = tops[-1] + objects[-1].shape[0] + bottom_margin

    if not GEN_HEIGHT_RANGE[0] <= height_needed <= GEN_HEIGHT_RANGE[1]:
        return None

    min_width = max(widths) + 2 * GEN_SIDE_MARGIN
    width = random.randint(max(GEN_WIDTH_RANGE[0], min_width), GEN_WIDTH_RANGE[1])

    anchors = [_object_anchor(top, obj) for top, obj in zip(tops, objects)]
    bounds = _band_bounds(anchors, height_needed)
    for top, obj, (start, end) in zip(tops, objects, bounds):
        bottom = top + obj.shape[0] - 1
        if top < start or bottom > end:
            return None

    grid = np.full((height_needed, width), BG, dtype=int)
    for top, obj in zip(tops, objects):
        left = random.randint(GEN_SIDE_MARGIN, width - obj.shape[1] - GEN_SIDE_MARGIN)
        patch_mask = obj == FG
        grid[top : top + obj.shape[0], left : left + obj.shape[1]][patch_mask] = FG

    if not _background_is_majority(grid):
        return None

    expected = _expected_output(height_needed, width, anchors, order)
    if not np.array_equal(main(grid), expected):
        return None
    return grid


def _gen_attempt() -> np.ndarray | None:
    order = list(GEN_COUNTS)
    random.shuffle(order)
    objects = []
    for holes in order:
        obj = _make_object(int(holes))
        if obj is None:
            return None
        objects.append(obj)
    return _place_scene(order, objects)


def generate_input() -> np.ndarray:
    for _ in range(GEN_ATTEMPTS):
        grid = _gen_attempt()
        if grid is not None:
            return grid
    raise RuntimeError("failed to generate valid input for puzzle 8")
