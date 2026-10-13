from __future__ import annotations

from collections import deque
import random

import numpy as np


BG = 0
LEGEND_SIZE = 5
MAX_GENERATION_ATTEMPTS = 400

STYLE_LIBRARY = (
    {
        "corners": (1, 1, 1, 1),
        "top": (1, 1, 1),
        "right": (1, 1, 1),
        "bottom": (1, 1, 1),
        "left": (1, 1, 1),
    },
    {
        "corners": (1, 1, 1, 1),
        "top": (0, 1, 0),
        "right": (0, 1, 0),
        "bottom": (0, 1, 0),
        "left": (0, 1, 0),
    },
    {
        "corners": (1, 1, 1, 1),
        "top": (0, 0, 0),
        "right": (0, 0, 0),
        "bottom": (0, 0, 0),
        "left": (0, 0, 0),
    },
    {
        "corners": (1, 1, 1, 1),
        "top": (1, 1, 1),
        "right": (0, 1, 0),
        "bottom": (1, 1, 1),
        "left": (0, 1, 0),
    },
    {
        "corners": (1, 1, 1, 1),
        "top": (0, 1, 0),
        "right": (1, 1, 1),
        "bottom": (0, 1, 0),
        "left": (1, 1, 1),
    },
)


def _outer_border_bg(grid: np.ndarray) -> int:
    border = np.concatenate(
        [grid[0, :], grid[-1, :], grid[1:-1, 0], grid[1:-1, -1]]
    )
    values, counts = np.unique(border, return_counts=True)
    return int(values[counts.argmax()])


def _components(mask: np.ndarray) -> list[list[tuple[int, int]]]:
    height, width = mask.shape
    seen = np.zeros_like(mask, dtype=bool)
    comps: list[list[tuple[int, int]]] = []

    for r in range(height):
        for c in range(width):
            if not mask[r, c] or seen[r, c]:
                continue
            seen[r, c] = True
            q: deque[tuple[int, int]] = deque([(r, c)])
            comp: list[tuple[int, int]] = []
            while q:
                rr, cc = q.popleft()
                comp.append((int(rr), int(cc)))
                for nr, nc in ((rr - 1, cc), (rr + 1, cc), (rr, cc - 1), (rr, cc + 1)):
                    if 0 <= nr < height and 0 <= nc < width and mask[nr, nc] and not seen[nr, nc]:
                        seen[nr, nc] = True
                        q.append((int(nr), int(nc)))
            comps.append(comp)
    return comps


def _bbox(points: list[tuple[int, int]]) -> tuple[int, int, int, int]:
    rows = [r for r, _ in points]
    cols = [c for _, c in points]
    return min(rows), min(cols), max(rows), max(cols)


def _spans(flags: np.ndarray) -> list[tuple[int, int]]:
    out: list[tuple[int, int]] = []
    start = None
    for idx, flag in enumerate(flags.tolist()):
        if flag and start is None:
            start = idx
        elif not flag and start is not None:
            out.append((start, idx - 1))
            start = None
    if start is not None:
        out.append((start, len(flags) - 1))
    return out


def _spec_from_patch(patch: np.ndarray, bg: int) -> tuple[int, int, dict[str, tuple[int, ...]]]:
    assert patch.shape[0] >= 3 and patch.shape[1] >= 3
    interior = patch[1:-1, 1:-1]
    fill_values = np.unique(interior[interior != bg])
    assert fill_values.size == 1, "legend core must carry one fill color"
    fill_color = int(fill_values[0])

    border_values = [int(v) for v in np.unique(patch) if int(v) not in {bg, fill_color}]
    assert len(border_values) == 1, "legend must carry exactly one border color"
    border_color = border_values[0]

    spec = {
        "corners": (
            int(patch[0, 0] == border_color),
            int(patch[0, -1] == border_color),
            int(patch[-1, -1] == border_color),
            int(patch[-1, 0] == border_color),
        ),
        "top": tuple(int(v == border_color) for v in patch[0, 1:-1]),
        "right": tuple(int(v == border_color) for v in patch[1:-1, -1]),
        "bottom": tuple(int(v == border_color) for v in patch[-1, 1:-1]),
        "left": tuple(int(v == border_color) for v in patch[1:-1, 0]),
    }
    return fill_color, border_color, spec


def _paint_border(
    out: np.ndarray,
    r0: int,
    c0: int,
    r1: int,
    c1: int,
    fill_color: int,
    border_color: int,
    spec: dict[str, tuple[int, ...]],
    bg: int,
) -> None:
    out[r0 : r1 + 1, c0 : c1 + 1] = int(bg)
    out[r0 + 1 : r1, c0 + 1 : c1] = int(fill_color)

    corners = spec["corners"]
    if corners[0]:
        out[r0, c0] = int(border_color)
    if corners[1]:
        out[r0, c1] = int(border_color)
    if corners[2]:
        out[r1, c1] = int(border_color)
    if corners[3]:
        out[r1, c0] = int(border_color)

    for idx in range(c1 - c0 - 1):
        if spec["top"][idx % len(spec["top"])]:
            out[r0, c0 + 1 + idx] = int(border_color)
        if spec["bottom"][idx % len(spec["bottom"])]:
            out[r1, c0 + 1 + idx] = int(border_color)

    for idx in range(r1 - r0 - 1):
        if spec["left"][idx % len(spec["left"])]:
            out[r0 + 1 + idx, c0] = int(border_color)
        if spec["right"][idx % len(spec["right"])]:
            out[r0 + 1 + idx, c1] = int(border_color)


def main(grid: np.ndarray) -> np.ndarray:
    """Redraw each filled rectangle's border from the legend box whose core color matches its interior.

    Input structure:
    - The background is black, and the main area contains several non-overlapping filled rectangles.
    - Every main rectangle is solid in the input: a shared placeholder border color surrounds a single interior fill color.
    - Along the bottom sits a horizontal legend row of small boxes. Each legend box has one core fill color and one border color, with some border pixels omitted to encode a border style template.

    Rule:
    - Find the bottom legend row and parse each legend box into a mapping from core fill color to border color plus pixel-level border template.
    - For each main rectangle, read its interior fill color and look up the legend box with the same core color.
    - Replace that rectangle's placeholder border with the legend's border color and border pattern, keep the interior unchanged, and erase the whole legend row to background.

    Key insight:
    - The legend key is the inner fill color, while the legend border simultaneously specifies both the output border color and which edge pixels should remain present.
    """
    g = np.asarray(grid, dtype=int)
    bg = _outer_border_bg(g)
    height, width = g.shape

    legend_top = height - 1
    while legend_top >= 0 and np.any(g[legend_top] != bg):
        legend_top -= 1
    legend_top += 1
    assert 0 < legend_top < height, "expected a separated legend row at the bottom"

    legend_band = g[legend_top:, :]
    col_spans = _spans(np.any(legend_band != bg, axis=0))
    assert col_spans, "expected at least one legend box"

    mapping: dict[int, tuple[int, dict[str, tuple[int, ...]]]] = {}
    for c0, c1 in col_spans:
        patch = legend_band[:, c0 : c1 + 1]
        fill_color, border_color, spec = _spec_from_patch(patch, bg)
        mapping[fill_color] = (border_color, spec)

    out = g.copy()
    out[legend_top:, :] = int(bg)

    for points in _components(g[:legend_top, :] != bg):
        r0, c0, r1, c1 = _bbox(points)
        patch = g[r0 : r1 + 1, c0 : c1 + 1]
        if patch.shape[0] < 3 or patch.shape[1] < 3:
            continue
        if np.any(patch == bg):
            continue

        border = np.concatenate([patch[0, :], patch[-1, :], patch[1:-1, 0], patch[1:-1, -1]])
        border_values = np.unique(border)
        if border_values.size != 1:
            continue

        interior = patch[1:-1, 1:-1]
        interior_values = np.unique(interior)
        if interior_values.size != 1:
            continue

        fill_color = int(interior_values[0])
        if fill_color == int(border_values[0]):
            continue
        assert fill_color in mapping, "every region fill color must appear in the legend"

        border_color, spec = mapping[fill_color]
        _paint_border(out, r0, c0, r1, c1, fill_color, border_color, spec, bg)

    return out


def _render_legend(fill_color: int, border_color: int, spec: dict[str, tuple[int, ...]]) -> np.ndarray:
    patch = np.full((LEGEND_SIZE, LEGEND_SIZE), BG, dtype=int)
    patch[1:-1, 1:-1] = int(fill_color)
    _paint_border(
        patch,
        0,
        0,
        LEGEND_SIZE - 1,
        LEGEND_SIZE - 1,
        fill_color,
        border_color,
        spec,
        BG,
    )
    return patch


def _draw_input_region(
    grid: np.ndarray,
    r0: int,
    c0: int,
    height: int,
    width: int,
    border_color: int,
    fill_color: int,
) -> None:
    r1 = r0 + height - 1
    c1 = c0 + width - 1
    grid[r0 : r1 + 1, c0 : c1 + 1] = int(fill_color)
    grid[r0 : r1 + 1, c0] = int(border_color)
    grid[r0 : r1 + 1, c1] = int(border_color)
    grid[r0, c0 : c1 + 1] = int(border_color)
    grid[r1, c0 : c1 + 1] = int(border_color)


def _sample_region_sizes(main_height: int, width: int, n_regions: int) -> tuple[int, int]:
    max_h = min(main_height - 2, 8 if n_regions <= 3 else 6)
    max_w = min(width - 2, 9 if n_regions <= 3 else 7)
    height = random.randint(4, max_h)
    box_width = random.randint(4, max_w)
    if random.random() < 0.35 and max_h >= 5:
        height = random.randint(5, max_h)
    if random.random() < 0.35 and max_w >= 5:
        box_width = random.randint(5, max_w)
    return height, box_width


def _build_expected(
    shape: tuple[int, int],
    regions: list[tuple[int, int, int, int, int]],
    legend_specs: dict[int, tuple[int, dict[str, tuple[int, ...]]]],
) -> np.ndarray:
    out = np.full(shape, BG, dtype=int)
    for r0, c0, height, width, fill_color in regions:
        r1 = r0 + height - 1
        c1 = c0 + width - 1
        border_color, spec = legend_specs[fill_color]
        _paint_border(out, r0, c0, r1, c1, fill_color, border_color, spec, BG)
    return out


def generate_input() -> np.ndarray:
    for _ in range(MAX_GENERATION_ATTEMPTS):
        n_legends = random.randint(2, 4)
        chosen_styles = random.sample(STYLE_LIBRARY, n_legends)
        color_pool = random.sample(list(range(1, 10)), 2 * n_legends + 1)
        placeholder_border = color_pool[0]
        fill_colors = color_pool[1 : 1 + n_legends]
        border_colors = color_pool[1 + n_legends :]

        gaps = [random.randint(1, 2) for _ in range(n_legends - 1)]
        legend_width = n_legends * LEGEND_SIZE + sum(gaps)
        cols = random.randint(max(18, legend_width + 2), 30)
        rows = random.randint(16, 30)
        main_height = rows - LEGEND_SIZE - 1
        if main_height < 8:
            continue

        grid = np.full((rows, cols), BG, dtype=int)
        legend_specs: dict[int, tuple[int, dict[str, tuple[int, ...]]]] = {}

        start_col = random.randint(1, cols - legend_width - 1)
        cursor = start_col
        for idx in range(n_legends):
            fill_color = fill_colors[idx]
            border_color = border_colors[idx]
            spec = chosen_styles[idx]
            legend_specs[fill_color] = (border_color, spec)
            patch = _render_legend(fill_color, border_color, spec)
            grid[main_height + 1 :, cursor : cursor + LEGEND_SIZE] = patch
            cursor += LEGEND_SIZE
            if idx < n_legends - 1:
                cursor += gaps[idx]

        used_fill_count = random.randint(2, n_legends)
        used_fills = random.sample(fill_colors, used_fill_count)
        n_regions = min(5, used_fill_count + random.randint(0, 2))

        assignments = used_fills[:]
        while len(assignments) < n_regions:
            assignments.append(random.choice(used_fills))
        random.shuffle(assignments)

        occupied = np.zeros((main_height, cols), dtype=bool)
        regions: list[tuple[int, int, int, int, int]] = []
        failed = False

        for fill_color in assignments:
            placed = False
            for _try in range(160):
                region_h, region_w = _sample_region_sizes(main_height, cols, n_regions)
                if region_h > main_height - 2 or region_w > cols - 2:
                    continue
                r0 = random.randint(1, main_height - region_h - 1)
                c0 = random.randint(1, cols - region_w - 1)
                r1 = r0 + region_h - 1
                c1 = c0 + region_w - 1
                if occupied[r0 - 1 : r1 + 2, c0 - 1 : c1 + 2].any():
                    continue
                occupied[r0 : r1 + 1, c0 : c1 + 1] = True
                regions.append((r0, c0, region_h, region_w, fill_color))
                placed = True
                break
            if not placed:
                failed = True
                break

        if failed:
            continue

        for r0, c0, region_h, region_w, fill_color in regions:
            _draw_input_region(
                grid,
                r0,
                c0,
                region_h,
                region_w,
                placeholder_border,
                fill_color,
            )

        expected = _build_expected(grid.shape, regions, legend_specs)
        try:
            actual = main(grid)
        except Exception:
            continue
        if np.array_equal(actual, expected):
            return grid

    raise RuntimeError("failed to generate a valid puzzle instance")
