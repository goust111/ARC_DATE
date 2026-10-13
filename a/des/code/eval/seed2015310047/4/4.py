from __future__ import annotations

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
AZURE = 8
BROWN = 9

PATTERN_COLORS = (BLUE, RED, GREEN, YELLOW, MAGENTA, BROWN)
MAX_GENERATION_ATTEMPTS = 300

CORNER_ORDER = ("tl", "tr", "bl", "br")
CORNER_TO_INDEX = {name: idx for idx, name in enumerate(CORNER_ORDER)}


def _components(mask: np.ndarray) -> list[list[tuple[int, int]]]:
    h, w = mask.shape
    seen = np.zeros((h, w), dtype=bool)
    comps: list[list[tuple[int, int]]] = []
    for r in range(h):
        for c in range(w):
            if not mask[r, c] or seen[r, c]:
                continue
            q = deque([(r, c)])
            seen[r, c] = True
            cells: list[tuple[int, int]] = []
            while q:
                rr, cc = q.popleft()
                cells.append((rr, cc))
                for dr, dc in ((1, 0), (-1, 0), (0, 1), (0, -1)):
                    nr, nc = rr + dr, cc + dc
                    if 0 <= nr < h and 0 <= nc < w and mask[nr, nc] and not seen[nr, nc]:
                        seen[nr, nc] = True
                        q.append((nr, nc))
            comps.append(cells)
    return comps


def _bbox(cells: list[tuple[int, int]]) -> tuple[int, int, int, int]:
    rows = [r for r, _ in cells]
    cols = [c for _, c in cells]
    return min(rows), min(cols), max(rows), max(cols)


def _perimeter_cells(r0: int, c0: int, r1: int, c1: int) -> set[tuple[int, int]]:
    cells = set()
    for c in range(c0, c1 + 1):
        cells.add((r0, c))
        cells.add((r1, c))
    for r in range(r0, r1 + 1):
        cells.add((r, c0))
        cells.add((r, c1))
    return cells


def _frame_info(grid: np.ndarray, cells: list[tuple[int, int]]) -> dict | None:
    r0, c0, r1, c1 = _bbox(cells)
    if r1 - r0 < 2 or c1 - c0 < 2:
        return None
    cell_set = set(cells)
    perimeter = _perimeter_cells(r0, c0, r1, c1)
    if cell_set != perimeter:
        return None

    corners = ((r0, c0), (r0, c1), (r1, c0), (r1, c1))
    azure_corner: str | None = None
    for idx, (r, c) in enumerate(corners):
        value = int(grid[r, c])
        if value == AZURE:
            if azure_corner is not None:
                return None
            azure_corner = CORNER_ORDER[idx]
        elif value != GREY:
            return None

    for r, c in perimeter - set(corners):
        if int(grid[r, c]) != GREY:
            return None

    return {
        "bbox": (r0, c0, r1, c1),
        "inner_shape": (r1 - r0 - 1, c1 - c0 - 1),
        "azure_corner": azure_corner,
    }


def _find_divider_row(grid: np.ndarray) -> int:
    rows = [r for r in range(grid.shape[0]) if np.all(grid[r] == ORANGE)]
    assert len(rows) == 1, "expected exactly one full orange divider row"
    return rows[0]


def _find_frames(grid: np.ndarray) -> list[dict]:
    mask = (grid == GREY) | (grid == AZURE)
    frames: list[dict] = []
    for cells in _components(mask):
        info = _frame_info(grid, cells)
        if info is not None:
            frames.append(info)
    return frames


def _transform_pattern(pattern: np.ndarray, corner: str) -> np.ndarray:
    if corner == "tl":
        return pattern.copy()
    if corner == "tr":
        return np.fliplr(pattern)
    if corner == "bl":
        return np.flipud(pattern)
    if corner == "br":
        return np.rot90(pattern, 2)
    raise AssertionError(f"unknown corner {corner}")


def main(grid: np.ndarray) -> np.ndarray:
    """
    Copy the pattern from the upper template frame into each lower target frame.

    The target frame's unique azure corner encodes which symmetry to apply:
    top-left keeps the pattern unchanged, top-right flips left-right,
    bottom-left flips top-bottom, and bottom-right rotates 180 degrees.
    """
    g = np.asarray(grid, dtype=int)
    assert g.ndim == 2

    divider = _find_divider_row(g)
    frames = _find_frames(g)

    template_candidates: list[dict] = []
    target_candidates: list[dict] = []
    for frame in frames:
        r0, c0, r1, c1 = frame["bbox"]
        if r1 < divider:
            inner = g[r0 + 1 : r1, c0 + 1 : c1]
            if frame["azure_corner"] is None and np.any(inner != BLACK):
                template_candidates.append(frame)
        elif r0 > divider:
            inner = g[r0 + 1 : r1, c0 + 1 : c1]
            if frame["azure_corner"] is not None and np.all(inner == BLACK):
                target_candidates.append(frame)

    assert len(template_candidates) == 1, "expected exactly one template frame"
    template = template_candidates[0]
    tr0, tc0, tr1, tc1 = template["bbox"]
    pattern = g[tr0 + 1 : tr1, tc0 + 1 : tc1].copy()
    inner_shape = pattern.shape

    assert target_candidates, "expected at least one target frame"
    out = g.copy()
    for frame in target_candidates:
        assert frame["inner_shape"] == inner_shape, "target size must match template size"
        r0, c0, r1, c1 = frame["bbox"]
        transformed = _transform_pattern(pattern, frame["azure_corner"])
        out[r0 + 1 : r1, c0 + 1 : c1] = transformed
        out[r0, c0] = GREY
        out[r0, c1] = GREY
        out[r1, c0] = GREY
        out[r1, c1] = GREY

    return out


def _draw_frame(grid: np.ndarray, r0: int, c0: int, inner: np.ndarray, marker_corner: str | None) -> None:
    inner_h, inner_w = inner.shape
    r1 = r0 + inner_h + 1
    c1 = c0 + inner_w + 1
    grid[r0 : r1 + 1, c0] = GREY
    grid[r0 : r1 + 1, c1] = GREY
    grid[r0, c0 : c1 + 1] = GREY
    grid[r1, c0 : c1 + 1] = GREY
    if marker_corner is not None:
        cr, cc = ((r0, c0), (r0, c1), (r1, c0), (r1, c1))[CORNER_TO_INDEX[marker_corner]]
        grid[cr, cc] = AZURE
    grid[r0 + 1 : r1, c0 + 1 : c1] = inner


def _pattern_variants(pattern: np.ndarray) -> list[np.ndarray]:
    return [
        pattern,
        np.fliplr(pattern),
        np.flipud(pattern),
        np.rot90(pattern, 2),
    ]


def _random_pattern(rng: random.Random, h: int, w: int) -> np.ndarray | None:
    area = h * w
    palette_size = rng.randint(2, min(5, len(PATTERN_COLORS)))
    palette = rng.sample(PATTERN_COLORS, palette_size)
    pattern = np.full((h, w), BLACK, dtype=int)

    min_cells = max(3, area // 3)
    max_cells = max(min_cells, area - rng.randint(0, max(0, area // 4)))
    filled = rng.randint(min_cells, max_cells)
    positions = rng.sample(range(area), filled)

    # Seed one instance of at least two colors to avoid monochrome templates.
    pattern.flat[positions[0]] = palette[0]
    pattern.flat[positions[1]] = palette[1]
    for pos in positions[2:]:
        pattern.flat[pos] = rng.choice(palette)

    non_black_colors = {int(v) for v in pattern.ravel() if int(v) != BLACK}
    if len(non_black_colors) < 2:
        return None

    sigs = {variant.tobytes() for variant in _pattern_variants(pattern)}
    if len(sigs) != 4:
        return None
    return pattern


def _random_partition(rng: random.Random, total: int, parts: int) -> list[int]:
    counts = [1] * parts
    for _ in range(total - parts):
        counts[rng.randrange(parts)] += 1
    rng.shuffle(counts)
    return counts


def _capped_partition(rng: random.Random, total: int, parts: int, cap: int) -> list[int] | None:
    if total < parts or total > parts * cap:
        return None
    counts = [1] * parts
    for _ in range(total - parts):
        choices = [idx for idx, value in enumerate(counts) if value < cap]
        if not choices:
            return None
        counts[rng.choice(choices)] += 1
    rng.shuffle(counts)
    return counts


def _bounded_gap_list(rng: random.Random, count: int, max_total: int, cap: int) -> list[int] | None:
    if count == 0:
        return []
    min_total = count
    max_allowed = min(max_total, count * cap)
    if min_total > max_allowed:
        return None
    total = rng.randint(min_total, max_allowed)
    values = [1] * count
    for _ in range(total - count):
        choices = [idx for idx, value in enumerate(values) if value < cap]
        if not choices:
            return None
        values[rng.choice(choices)] += 1
    rng.shuffle(values)
    return values


def _positive_parts(rng: random.Random, total: int, parts: int) -> list[int]:
    assert total >= parts >= 1
    values = [1] * parts
    for _ in range(total - parts):
        values[rng.randrange(parts)] += 1
    rng.shuffle(values)
    return values


def _balanced_corners(rng: random.Random, n_targets: int) -> list[str]:
    corners = list(CORNER_ORDER) * (n_targets // 4)
    extras = list(CORNER_ORDER)
    rng.shuffle(extras)
    corners.extend(extras[: n_targets % 4])
    rng.shuffle(corners)
    return corners


def _gen_attempt(rng: random.Random) -> np.ndarray | None:
    inner_h = rng.randint(3, 6)
    inner_w = rng.randint(3, 6)
    pattern = _random_pattern(rng, inner_h, inner_w)
    if pattern is None:
        return None

    outer_h = inner_h + 2
    outer_w = inner_w + 2

    max_per_row = 29 // (outer_w + 1)
    if max_per_row < 1:
        return None

    max_rows = 0
    for candidate in range(1, 4):
        min_bottom_h = candidate * outer_h + (candidate - 1)
        min_grid_h = outer_h + 1 + min_bottom_h + 4
        if min_grid_h <= 30:
            max_rows = candidate
    if max_rows < 1:
        return None

    max_targets = min(8, max_per_row * max_rows)
    if max_targets < 4:
        return None

    n_targets = rng.randint(4, max_targets)
    min_rows = (n_targets + max_per_row - 1) // max_per_row
    if min_rows > max_rows:
        return None
    n_rows = rng.randint(min_rows, max_rows)
    row_counts = _capped_partition(rng, n_targets, n_rows, max_per_row)
    if row_counts is None:
        return None

    h_gaps: list[list[int]] = []
    row_widths: list[int] = []
    for count in row_counts:
        max_gap_total = 28 - count * outer_w
        gaps = _bounded_gap_list(rng, max(0, count - 1), max_gap_total, cap=3)
        if gaps is None:
            return None
        h_gaps.append(gaps)
        row_widths.append(count * outer_w + sum(gaps))

    max_vgap_total = (25 - outer_h) - n_rows * outer_h
    v_gaps = _bounded_gap_list(rng, max(0, n_rows - 1), max_vgap_total, cap=3)
    if v_gaps is None:
        return None
    bottom_block_h = n_rows * outer_h + sum(v_gaps)

    base_w = max(outer_w, max(row_widths))
    max_extra_w = 30 - base_w
    if max_extra_w < 2:
        return None
    total_extra_w = rng.randint(2, max_extra_w)
    left_margin = rng.randint(1, total_extra_w - 1)
    right_margin = rng.randint(1, total_extra_w - left_margin)
    width_slack = total_extra_w - left_margin - right_margin
    grid_w = base_w + total_extra_w

    base_h = outer_h + 1 + bottom_block_h
    max_extra_h = 30 - base_h
    if max_extra_h < 4:
        return None
    total_extra_h = rng.randint(4, max_extra_h)
    top_margin, gap_before_divider, gap_after_divider, bottom_margin = _positive_parts(rng, total_extra_h, 4)
    divider_r = top_margin + outer_h + gap_before_divider
    grid_h = base_h + total_extra_h

    grid = np.full((grid_h, grid_w), BLACK, dtype=int)
    grid[divider_r, :] = ORANGE

    template_c = rng.randint(left_margin, grid_w - right_margin - outer_w)
    _draw_frame(grid, top_margin, template_c, pattern, None)

    corners = _balanced_corners(rng, n_targets)
    cursor = 0
    row_r = divider_r + 1 + gap_after_divider
    for row_idx, count in enumerate(row_counts):
        row_width = row_widths[row_idx]
        row_c = rng.randint(left_margin, grid_w - right_margin - row_width)
        for item_idx in range(count):
            marker_corner = corners[cursor]
            cursor += 1
            c0 = row_c
            empty = np.full((inner_h, inner_w), BLACK, dtype=int)
            _draw_frame(grid, row_r, c0, empty, marker_corner)
            row_c += outer_w
            if item_idx < count - 1:
                row_c += h_gaps[row_idx][item_idx]
        if row_idx < n_rows - 1:
            row_r += outer_h + v_gaps[row_idx]

    # Validate the generated sample against the solver and the intended structure.
    out = main(grid)
    if out.shape != grid.shape or np.array_equal(out, grid):
        return None

    sigs = {variant.tobytes() for variant in _pattern_variants(pattern)}
    if len(sigs) != 4:
        return None

    return grid


def generate_input() -> np.ndarray:
    rng = random.Random()
    for _ in range(MAX_GENERATION_ATTEMPTS):
        try:
            grid = _gen_attempt(rng)
        except AssertionError:
            continue
        if grid is not None:
            return grid
    raise RuntimeError("failed to generate valid input")
