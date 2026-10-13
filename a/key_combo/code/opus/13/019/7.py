from __future__ import annotations

import itertools
import random
from collections import deque

import numpy as np


BLACK = 0
BLUE = 1
RED = 2
GREEN = 3
YELLOW = 4
GREY = 5
MAGENTA = 6
ORANGE = 7

MARKER_TO_RANK_INDEX = {
    RED: 0,    # smallest area
    BLUE: 1,   # middle area
    GREEN: 2,  # largest area
}

WINDOW_SIZE = 3
STRIDE = 4
GRID_WIDTH = 11
BOTTOM_HEIGHT = 3
TEMPLATE_COLORS = (YELLOW, MAGENTA, ORANGE)
RETRY_LIMIT = 400


def _neighbors4(r: int, c: int):
    yield r - 1, c
    yield r + 1, c
    yield r, c - 1
    yield r, c + 1


def _intervals_from_separators(limit: int, separators: list[int]) -> list[tuple[int, int]]:
    intervals: list[tuple[int, int]] = []
    start = 0
    for sep in sorted(separators):
        if start < sep:
            intervals.append((start, sep))
        start = sep + 1
    if start < limit:
        intervals.append((start, limit))
    return intervals


def _mask_components(mask: np.ndarray) -> list[list[tuple[int, int]]]:
    h, w = mask.shape
    seen = np.zeros((h, w), dtype=bool)
    components: list[list[tuple[int, int]]] = []
    for r in range(h):
        for c in range(w):
            if not mask[r, c] or seen[r, c]:
                continue
            queue = deque([(r, c)])
            seen[r, c] = True
            coords: list[tuple[int, int]] = []
            while queue:
                rr, cc = queue.popleft()
                coords.append((rr, cc))
                for nr, nc in _neighbors4(rr, cc):
                    if 0 <= nr < h and 0 <= nc < w and mask[nr, nc] and not seen[nr, nc]:
                        seen[nr, nc] = True
                        queue.append((nr, nc))
            components.append(coords)
    return components


def _extract_template(bottom: np.ndarray, coords: list[tuple[int, int]]) -> dict:
    rows = [r for r, _ in coords]
    cols = [c for _, c in coords]
    r0, r1 = min(rows), max(rows)
    c0, c1 = min(cols), max(cols)
    patch = bottom[r0 : r1 + 1, c0 : c1 + 1].copy()
    return {
        "area": int(len(coords)),
        "patch": patch,
    }


def _stamp_centered(out: np.ndarray, r0: int, c0: int, patch: np.ndarray) -> None:
    h, w = patch.shape
    rr = r0 + (WINDOW_SIZE - h) // 2
    cc = c0 + (WINDOW_SIZE - w) // 2
    mask = patch != BLACK
    out[rr : rr + h, cc : cc + w][mask] = patch[mask]


def _solve(grid: np.ndarray) -> np.ndarray:
    g = np.asarray(grid, dtype=int)
    assert g.ndim == 2 and max(g.shape) <= 30

    full_grey_rows = [r for r in range(g.shape[0]) if bool(np.all(g[r, :] == GREY))]
    assert full_grey_rows, "expected at least one full-width grey separator row"
    divider = max(full_grey_rows)
    assert 0 < divider < g.shape[0] - 1

    top = g[:divider, :]
    sep_rows = [r for r in full_grey_rows if r < divider]
    sep_cols = [c for c in range(g.shape[1]) if bool(np.all(top[:, c] == GREY))]

    row_blocks = _intervals_from_separators(divider, sep_rows)
    col_blocks = _intervals_from_separators(g.shape[1], sep_cols)
    assert row_blocks and col_blocks
    assert all((r1 - r0) == WINDOW_SIZE for r0, r1 in row_blocks)
    assert all((c1 - c0) == WINDOW_SIZE for c0, c1 in col_blocks)

    bottom = g[divider + 1 :, :]
    template_mask = (bottom != BLACK) & (bottom != GREY)
    components = _mask_components(template_mask)
    assert len(components) == 3, "expected exactly three bottom templates"

    templates = [_extract_template(bottom, coords) for coords in components]
    areas = sorted(t["area"] for t in templates)
    assert len(set(areas)) == 3, "template areas must be strictly distinct"
    ranked = sorted(templates, key=lambda item: item["area"])

    out = g.copy()
    out_bottom = out[divider + 1 :, :]
    out_bottom[template_mask] = BLACK
    for r0, r1 in row_blocks:
        for c0, c1 in col_blocks:
            window = g[r0:r1, c0:c1]
            assert window.shape == (WINDOW_SIZE, WINDOW_SIZE)
            marker = int(window[1, 1])
            assert marker in MARKER_TO_RANK_INDEX
            out[r0:r1, c0:c1] = BLACK
            patch = ranked[MARKER_TO_RANK_INDEX[marker]]["patch"]
            _stamp_centered(out, r0, c0, patch)

    return out


def main(grid: np.ndarray) -> np.ndarray:
    """
    Stamp the area-ranked bottom template into each grey-partitioned window selected by its center marker.

    Input structure
    - A black background contains an upper grid of 3x3 windows separated by grey lines, plus one full grey row that separates this window area from a lower reserve area.
    - Each 3x3 window contains exactly one center marker pixel, colored red, blue, or green.
    - The lower reserve area contains exactly three disconnected monochrome templates, each with a distinct area and a bounding box no larger than 3x3.

    Rule
    - Identify the three bottom templates and sort them by area from small to large.
    - Interpret center marker colors as rank selectors: red chooses the smallest template, blue chooses the middle template, and green chooses the largest template.
    - For every 3x3 window, erase the marker and stamp the selected template into that window using its original color, centered by bounding box; keep grey separator lines unchanged.
    - Erase the bottom templates to black.

    Key insight
    - The marker colors do not name template colors directly; they point to dynamic area ranks computed from the three bottom templates in the current input.
    """
    return _solve(grid)


def _normalize_cells(cells: set[tuple[int, int]]) -> frozenset[tuple[int, int]]:
    min_r = min(r for r, _ in cells)
    min_c = min(c for _, c in cells)
    return frozenset((r - min_r, c - min_c) for r, c in cells)


def _is_connected(cells: set[tuple[int, int]]) -> bool:
    start = next(iter(cells))
    queue = deque([start])
    seen = {start}
    while queue:
        r, c = queue.popleft()
        for nr, nc in _neighbors4(r, c):
            if (nr, nc) in cells and (nr, nc) not in seen:
                seen.add((nr, nc))
                queue.append((nr, nc))
    return len(seen) == len(cells)


def _shape_library() -> dict[int, list[np.ndarray]]:
    library: dict[int, list[np.ndarray]] = {area: [] for area in range(2, 6)}
    for area in range(2, 6):
        unique: set[frozenset[tuple[int, int]]] = set()
        for combo in itertools.combinations([(r, c) for r in range(3) for c in range(3)], area):
            cells = set(combo)
            if not _is_connected(cells):
                continue
            norm = _normalize_cells(cells)
            max_r = max(r for r, _ in norm)
            max_c = max(c for _, c in norm)
            h, w = max_r + 1, max_c + 1
            if (h // 2, w // 2) not in norm:
                continue
            unique.add(norm)
        for norm in sorted(unique):
            max_r = max(r for r, _ in norm)
            max_c = max(c for _, c in norm)
            patch = np.zeros((max_r + 1, max_c + 1), dtype=int)
            for r, c in norm:
                patch[r, c] = 1
            library[area].append(patch)
    return library


SHAPE_LIBRARY = _shape_library()


def _colorize_patch(mask_patch: np.ndarray, color: int) -> np.ndarray:
    return np.where(mask_patch != 0, int(color), BLACK).astype(int)


def _distribute_extra(total: int, buckets: int) -> list[int]:
    parts = [0] * buckets
    for _ in range(total):
        parts[random.randrange(buckets)] += 1
    return parts


def _build_templates() -> list[tuple[np.ndarray, int]]:
    areas = random.choice(
        [
            (2, 3, 4),
            (2, 3, 5),
            (2, 4, 5),
            (3, 4, 5),
        ]
    )
    colors = list(TEMPLATE_COLORS)
    random.shuffle(colors)

    templates: list[tuple[np.ndarray, int]] = []
    for area, color in zip(areas, colors):
        mask_patch = random.choice(SHAPE_LIBRARY[area])
        templates.append((_colorize_patch(mask_patch, color), area))
    random.shuffle(templates)
    return templates


def _place_templates(grid: np.ndarray, divider: int, templates: list[tuple[np.ndarray, int]]) -> bool:
    widths = [patch.shape[1] for patch, _area in templates]
    minimum = sum(widths) + 4
    if minimum > GRID_WIDTH:
        return False

    extras = _distribute_extra(GRID_WIDTH - minimum, 4)
    left = extras[0]
    gap1 = 2 + extras[1]
    gap2 = 2 + extras[2]

    starts = [
        left,
        left + widths[0] + gap1,
        left + widths[0] + gap1 + widths[1] + gap2,
    ]
    for (patch, _area), c0 in zip(templates, starts):
        h, w = patch.shape
        r0 = divider + 1 + random.randint(0, BOTTOM_HEIGHT - h)
        view = grid[r0 : r0 + h, c0 : c0 + w]
        if np.any(view != BLACK):
            return False
        mask = patch != BLACK
        view[mask] = patch[mask]
    return True


def _fill_markers(grid: np.ndarray, row_cells: int) -> None:
    total = row_cells * 3
    markers = [RED, BLUE, GREEN]
    markers.extend(random.choice((RED, BLUE, GREEN)) for _ in range(total - 3))
    random.shuffle(markers)
    idx = 0
    for rr in range(row_cells):
        for cc in range(3):
            grid[rr * STRIDE + 1, cc * STRIDE + 1] = markers[idx]
            idx += 1


def _build_input() -> np.ndarray | None:
    row_cells = random.choice((3, 4))
    divider = row_cells * STRIDE - 1
    height = divider + 1 + BOTTOM_HEIGHT

    grid = np.zeros((height, GRID_WIDTH), dtype=int)
    for r in range(3, divider + 1, STRIDE):
        grid[r, :] = GREY
    for c in range(3, GRID_WIDTH, STRIDE):
        grid[:divider, c] = GREY

    _fill_markers(grid, row_cells)
    templates = _build_templates()
    if not _place_templates(grid, divider, templates):
        return None

    bottom = grid[divider + 1 :, :]
    template_mask = (bottom != BLACK) & (bottom != GREY)
    comps = _mask_components(template_mask)
    if len(comps) != 3:
        return None
    areas = sorted(len(comp) for comp in comps)
    if len(set(areas)) != 3:
        return None

    try:
        out = main(grid)
    except AssertionError:
        return None

    if np.array_equal(out, grid):
        return None
    marker_set = {
        int(grid[rr * STRIDE + 1, cc * STRIDE + 1])
        for rr in range(row_cells)
        for cc in range(3)
    }
    if marker_set != {RED, BLUE, GREEN}:
        return None
    return grid


def generate_input() -> np.ndarray:
    for _ in range(RETRY_LIMIT):
        grid = _build_input()
        if grid is not None:
            return grid
    raise RuntimeError("failed to generate a valid input")
