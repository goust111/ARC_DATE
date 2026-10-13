from __future__ import annotations

import random
import sys
from collections import Counter, deque
from pathlib import Path
from typing import Iterable

import numpy as np


_ARC_UTILS_DIR = Path("/home/lixinyu/snap/new/solvers")
if _ARC_UTILS_DIR.exists():
    arc_utils_dir = str(_ARC_UTILS_DIR)
    if arc_utils_dir not in sys.path:
        sys.path.append(arc_utils_dir)

try:
    from arc_utils import Color, colors_present, crop_bbox, find_components, get_bg
except Exception:
    class Color:
        BLACK = 0
        BLUE = 1
        RED = 2
        GREY = 5
        GRAY = 5
        MAROON = 9

    def get_bg(grid: np.ndarray) -> int:
        return int(Counter(map(int, grid.flatten())).most_common(1)[0][0])

    def colors_present(grid: np.ndarray, *, exclude: Iterable[int] | None = None) -> list[int]:
        cols = set(map(int, grid.flatten()))
        if exclude is not None:
            cols -= set(map(int, exclude))
        return sorted(cols)

    def crop_bbox(grid: np.ndarray, box: tuple[int, int, int, int]) -> np.ndarray:
        r0, c0, r1, c1 = map(int, box)
        return grid[r0 : r1 + 1, c0 : c1 + 1].copy()

    def _mask_components(mask: np.ndarray, connectivity: int) -> list[np.ndarray]:
        h, w = mask.shape
        seen = np.zeros_like(mask, dtype=bool)
        if int(connectivity) == 4:
            steps = [(-1, 0), (1, 0), (0, -1), (0, 1)]
        else:
            steps = [
                (-1, -1),
                (-1, 0),
                (-1, 1),
                (0, -1),
                (0, 1),
                (1, -1),
                (1, 0),
                (1, 1),
            ]

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
                    for dr, dc in steps:
                        nr, nc = rr + dr, cc + dc
                        if 0 <= nr < h and 0 <= nc < w and mask[nr, nc] and not seen[nr, nc]:
                            seen[nr, nc] = True
                            q.append((nr, nc))
                comps.append(np.array(coords, dtype=int))
        return comps

    def find_components(
        grid: np.ndarray,
        *,
        bg: int | None = None,
        connectivity: int = 4,
        monochromatic: bool = True,
        colors: Iterable[int] | None = None,
        min_area: int = 1,
        return_format: str = "dict",
    ) -> list[dict]:
        assert monochromatic and return_format == "dict"
        if bg is None:
            bg = get_bg(grid)
        if colors is None:
            colors_iter = [c for c in colors_present(grid, exclude={bg})]
        else:
            colors_iter = [int(c) for c in colors if int(c) != int(bg)]

        out: list[dict] = []
        for color in colors_iter:
            for coords in _mask_components(grid == int(color), connectivity):
                if int(coords.shape[0]) < int(min_area):
                    continue
                r0, c0 = coords.min(axis=0)
                r1, c1 = coords.max(axis=0)
                out.append(
                    {
                        "color": int(color),
                        "coords": coords,
                        "bbox": (int(r0), int(c0), int(r1), int(c1)),
                        "area": int(coords.shape[0]),
                    }
                )
        return out


GEN_ATTEMPTS = 400
GRID_HEIGHT_RANGE = (14, 24)
GRID_WIDTH_RANGE = (16, 28)
AXIS_ROW_RANGE = (4, 8)
RECT_HEIGHT_RANGE = (2, 5)
RECT_WIDTH_RANGE = (4, 8)
GAP_TO_AXIS_RANGE = (1, 4)
DISTRACTOR_COUNT_RANGE = (0, 4)


def _find_separator_row(g: np.ndarray, bg: int) -> tuple[int, int]:
    rows: list[tuple[int, int]] = []
    for r in range(g.shape[0]):
        vals = np.unique(g[r, :])
        if vals.size == 1 and int(vals[0]) != int(bg):
            rows.append((int(r), int(vals[0])))
    assert len(rows) == 1, "expected exactly one full-width separator row"
    return rows[0]


def _is_solid_rect(g: np.ndarray, component: dict) -> bool:
    r0, c0, r1, c1 = map(int, component["bbox"])
    area = (r1 - r0 + 1) * (c1 - c0 + 1)
    if int(component["area"]) != area:
        return False
    patch = crop_bbox(g, (r0, c0, r1, c1))
    return bool(np.all(patch == int(component["color"])))


def _find_source_rectangle(g: np.ndarray, bg: int, axis_row: int, axis_color: int) -> tuple[int, dict]:
    candidates: list[tuple[int, dict]] = []
    for color in colors_present(g, exclude={bg, axis_color}):
        components = find_components(
            g,
            bg=bg,
            connectivity=4,
            monochromatic=True,
            colors=[int(color)],
            return_format="dict",
        )
        if len(components) != 1:
            continue
        component = components[0]
        if int(component["area"]) <= 1:
            continue
        if int(component["bbox"][2]) >= int(axis_row):
            continue
        if _is_solid_rect(g, component):
            candidates.append((int(color), component))

    assert len(candidates) == 1, "expected one solid rectangle above the separator"
    return candidates[0]


def _find_control_color(
    g: np.ndarray,
    bg: int,
    axis_color: int,
    source_color: int,
    rect_bbox: tuple[int, int, int, int],
    axis_row: int,
) -> int | None:
    r0, c0, r1, c1 = map(int, rect_bbox)
    candidates: list[int] = []
    for color in colors_present(g, exclude={bg, axis_color, source_color}):
        coords = np.argwhere(g == int(color))
        if coords.size == 0:
            continue
        rows_ok = np.all((coords[:, 0] > r1) & (coords[:, 0] < axis_row))
        cols_ok = np.all((coords[:, 1] >= c0) & (coords[:, 1] <= c1))
        if rows_ok and cols_ok:
            candidates.append(int(color))

    if not candidates:
        return None
    assert len(candidates) == 1, "expected at most one corridor control color"
    return candidates[0]


def _active_columns(
    g: np.ndarray,
    rect_bbox: tuple[int, int, int, int],
    axis_row: int,
    control_color: int | None,
) -> list[int]:
    _r0, c0, r1, c1 = map(int, rect_bbox)
    blocked: set[int] = set()
    if control_color is not None and r1 + 1 < axis_row:
        corridor = g[r1 + 1 : axis_row, c0 : c1 + 1]
        blocked = {int(c0 + local_c) for local_c in np.argwhere(corridor == int(control_color))[:, 1]}
    active = [c for c in range(c0, c1 + 1) if c not in blocked]
    assert active, "at least one rectangle column must remain active"
    return active


def _build_source_mask(
    shape: tuple[int, int],
    rect_bbox: tuple[int, int, int, int],
    axis_row: int,
    active_cols: list[int],
) -> np.ndarray:
    mask = np.zeros(shape, dtype=bool)
    r0, c0, r1, c1 = map(int, rect_bbox)
    mask[r0 : r1 + 1, c0 : c1 + 1] = True
    for c in active_cols:
        if r1 + 1 < axis_row:
            mask[r1 + 1 : axis_row, int(c)] = True
    return mask


def main(grid: np.ndarray) -> np.ndarray:
    """
    Extend the unblocked rectangle columns to a separator row, then mirror the rectangle-plus-stripes across that row.

    Input structure:
    - A dominant background contains exactly one full-width horizontal separator row in a non-background color.
    - Above the separator sits one solid rectangle in a different color.
    - Between the rectangle and the separator, isolated control pixels may mark rectangle columns that must not extend; other isolated pixels below the separator are distractors.

    Rule:
    - Identify the solid rectangle above the separator and inspect only its column span.
    - For each rectangle column with no control pixel between the rectangle bottom and the separator, fill the black cells in that corridor with the rectangle color.
    - Mirror the rectangle together with those added stripes across the separator row, painting the mirrored pixels on the lower side and letting them overwrite distractors there.
    - Keep the separator row in the separator color and leave blocked columns unextended.

    Key insight:
    - The control pixels do not move directly; they only disable specific stripe columns, and the reflected lower pattern is the mirror of the filtered upper comb.
    """
    g = np.asarray(grid, dtype=int)
    assert g.ndim == 2

    bg = int(get_bg(g))
    axis_row, axis_color = _find_separator_row(g, bg)
    source_color, source_rect = _find_source_rectangle(g, bg, axis_row, axis_color)
    rect_bbox = tuple(map(int, source_rect["bbox"]))
    control_color = _find_control_color(g, bg, axis_color, source_color, rect_bbox, axis_row)
    active_cols = _active_columns(g, rect_bbox, axis_row, control_color)

    source_mask = _build_source_mask(g.shape, rect_bbox, axis_row, active_cols)
    source_coords = np.argwhere(source_mask)
    mirrored = source_coords.copy()
    mirrored[:, 0] = 2 * int(axis_row) - mirrored[:, 0]
    assert np.all((0 <= mirrored[:, 0]) & (mirrored[:, 0] < g.shape[0])), "reflection must stay in bounds"

    out = g.copy()
    out[source_mask] = int(source_color)
    out[int(axis_row), np.array(active_cols, dtype=int)] = int(axis_color)
    out[mirrored[:, 0], mirrored[:, 1]] = int(source_color)
    return out


def _future_blue_positions(
    shape: tuple[int, int],
    rect_bbox: tuple[int, int, int, int],
    axis_row: int,
    blocked_cols: set[int],
) -> set[tuple[int, int]]:
    active_cols = [c for c in range(int(rect_bbox[1]), int(rect_bbox[3]) + 1) if c not in blocked_cols]
    source_mask = _build_source_mask(shape, rect_bbox, axis_row, active_cols)
    coords = np.argwhere(source_mask)
    mirrored = coords.copy()
    mirrored[:, 0] = 2 * int(axis_row) - mirrored[:, 0]
    return {tuple(map(int, rc)) for rc in mirrored}


def _place_distractors(
    grid: np.ndarray,
    future_blue: set[tuple[int, int]],
    axis_row: int,
    count: int,
) -> None:
    if count <= 0:
        return

    h, w = grid.shape
    below = [(r, c) for r in range(axis_row + 1, h) for c in range(w)]
    inside_pool = [pos for pos in below if pos in future_blue]
    outside_pool = [pos for pos in below if pos not in future_blue]

    chosen: list[tuple[int, int]] = []
    used: set[tuple[int, int]] = set()

    if inside_pool and random.random() < 0.65:
        pos = random.choice(inside_pool)
        chosen.append(pos)
        used.add(pos)

    if len(chosen) < count and outside_pool and random.random() < 0.8:
        pos = random.choice(outside_pool)
        if pos not in used:
            chosen.append(pos)
            used.add(pos)

    combined = [pos for pos in below if pos not in used]
    while len(chosen) < count and combined:
        idx = random.randrange(len(combined))
        pos = combined.pop(idx)
        chosen.append(pos)

    for r, c in chosen:
        grid[r, c] = int(Color.MAROON)


def _gen_attempt() -> np.ndarray | None:
    h = random.randint(*GRID_HEIGHT_RANGE)
    w = random.randint(*GRID_WIDTH_RANGE)
    axis_hi = min(AXIS_ROW_RANGE[1], h // 2)
    if axis_hi < AXIS_ROW_RANGE[0]:
        return None
    axis_row = random.randint(AXIS_ROW_RANGE[0], axis_hi)

    feasible_sizes = []
    for rect_h in range(RECT_HEIGHT_RANGE[0], RECT_HEIGHT_RANGE[1] + 1):
        for gap in range(GAP_TO_AXIS_RANGE[0], GAP_TO_AXIS_RANGE[1] + 1):
            if rect_h + gap <= axis_row - 1:
                feasible_sizes.append((rect_h, gap))
    if not feasible_sizes:
        return None
    rect_h, gap = random.choice(feasible_sizes)

    rect_w_max = min(RECT_WIDTH_RANGE[1], w - 2)
    if rect_w_max < RECT_WIDTH_RANGE[0]:
        return None
    rect_w = random.randint(RECT_WIDTH_RANGE[0], rect_w_max)

    rect_top = axis_row - gap - rect_h
    rect_left = random.randint(1, w - rect_w - 1)
    rect_bbox = (rect_top, rect_left, rect_top + rect_h - 1, rect_left + rect_w - 1)

    grid = np.full((h, w), int(Color.BLACK), dtype=int)
    grid[axis_row, :] = int(Color.GREY)
    r0, c0, r1, c1 = map(int, rect_bbox)
    grid[r0 : r1 + 1, c0 : c1 + 1] = int(Color.BLUE)

    blocked_count = random.randint(1, rect_w - 1)
    blocked_offsets = set(random.sample(range(rect_w), blocked_count))
    blocked_cols = {c0 + off for off in blocked_offsets}
    for c in blocked_cols:
        rr = random.randint(r1 + 1, axis_row - 1)
        grid[rr, c] = int(Color.RED)

    future_blue = _future_blue_positions(grid.shape, rect_bbox, axis_row, blocked_cols)
    distractor_count = random.randint(*DISTRACTOR_COUNT_RANGE)
    _place_distractors(grid, future_blue, axis_row, distractor_count)

    try:
        out = main(grid)
    except Exception:
        return None

    if np.array_equal(out, grid):
        return None
    if np.count_nonzero(out == int(Color.BLUE)) <= np.count_nonzero(grid == int(Color.BLUE)):
        return None
    return grid


def generate_input() -> np.ndarray:
    for _ in range(GEN_ATTEMPTS):
        grid = _gen_attempt()
        if grid is not None:
            return grid
    raise RuntimeError(f"failed to generate a valid input after {GEN_ATTEMPTS} attempts")
