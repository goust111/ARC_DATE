from __future__ import annotations

from collections import deque
import random

import numpy as np


GEN_RETRIES = 400
PLACE_RETRIES = 160
GRID_H_RANGE = (18, 28)
GRID_W_RANGE = (18, 30)
FENCE_COUNT_RANGE = (2, 4)
BOX_H_RANGE = (5, 8)
BOX_W_RANGE = (5, 9)
COLOR_POOL = tuple(range(10))
SIDES = ("top", "bottom", "left", "right")

_GEN_CALLS = 0


def _get_bg(grid: np.ndarray) -> int:
    vals, counts = np.unique(grid, return_counts=True)
    return int(vals[np.argmax(counts)])


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
            pts: list[tuple[int, int]] = []
            while q:
                rr, cc = q.popleft()
                pts.append((rr, cc))
                for dr, dc in ((1, 0), (-1, 0), (0, 1), (0, -1)):
                    nr, nc = rr + dr, cc + dc
                    if 0 <= nr < h and 0 <= nc < w and mask[nr, nc] and not seen[nr, nc]:
                        seen[nr, nc] = True
                        q.append((nr, nc))
            comps.append(pts)
    return comps


def _color_runs(row: np.ndarray) -> list[tuple[int, int, int]]:
    runs: list[tuple[int, int, int]] = []
    start = 0
    cur = int(row[0])
    for idx in range(1, len(row)):
        val = int(row[idx])
        if val != cur:
            runs.append((start, idx - 1, cur))
            start = idx
            cur = val
    runs.append((start, len(row) - 1, cur))
    return runs


def _find_palette_row(grid: np.ndarray, bg: int) -> tuple[int, list[int]]:
    h = grid.shape[0]
    for r in range(h - 1, -1, -1):
        runs = _color_runs(grid[r])
        colors = [color for _c0, _c1, color in runs]
        if len(runs) == 4 and all(color != bg for color in colors) and len(set(colors)) == 4:
            return r, colors
    raise AssertionError("expected a bottom palette row with four color runs")


def _analyze_fence(
    grid: np.ndarray, pts: list[tuple[int, int]], bg: int
) -> dict[str, object]:
    colors = {int(grid[r, c]) for r, c in pts}
    assert len(colors) == 1, "each fence must be monochrome"
    color = int(next(iter(colors)))
    rows = [r for r, _c in pts]
    cols = [c for _r, c in pts]
    r0, r1 = min(rows), max(rows)
    c0, c1 = min(cols), max(cols)
    assert r1 - r0 >= 2 and c1 - c0 >= 2, "fence bbox must enclose an interior"

    comp = set(pts)
    assert (r0, c0) in comp and (r0, c1) in comp and (r1, c0) in comp and (r1, c1) in comp
    for r, c in comp:
        assert r in (r0, r1) or c in (c0, c1), "fence pixels must lie on the bbox border"

    gaps: list[tuple[int, int, int, int]] = []
    for c in range(c0 + 1, c1):
        if (r0, c) not in comp:
            assert int(grid[r0, c]) == bg
            gaps.append((r0, c, -1, 0))
        if (r1, c) not in comp:
            assert int(grid[r1, c]) == bg
            gaps.append((r1, c, 1, 0))
    for r in range(r0 + 1, r1):
        if (r, c0) not in comp:
            assert int(grid[r, c0]) == bg
            gaps.append((r, c0, 0, -1))
        if (r, c1) not in comp:
            assert int(grid[r, c1]) == bg
            gaps.append((r, c1, 0, 1))

    assert 1 <= len(gaps) <= 4, "gap count must be between 1 and 4"
    return {
        "color": color,
        "bbox": (r0, c0, r1, c1),
        "gaps": gaps,
        "gap_count": len(gaps),
    }


def _ray_cells(
    start: tuple[int, int], direction: tuple[int, int], h: int, w: int, palette_row: int
) -> list[tuple[int, int]]:
    rr, cc = start
    dr, dc = direction
    cells: list[tuple[int, int]] = []
    while 0 <= rr < h and 0 <= cc < w and rr != palette_row:
        cells.append((rr, cc))
        rr += dr
        cc += dc
    return cells


def main(grid: np.ndarray) -> np.ndarray:
    """
    Fill each gapped rectangular fence using the bottom color legend indexed by gap count, then emit rays through the gaps.

    Input structure:
    - The most common color is the background.
    - Several monochrome non-background fences form axis-aligned rectangular borders with 1 to 4 single-cell gaps on their sides.
    - Gaps occur only on side middles, never on corners.
    - The bottom row is a four-block color legend whose left-to-right order corresponds to gap counts 1, 2, 3, and 4.

    Rule:
    - Read the bottom legend to obtain the fill color for each possible gap count.
    - For every fence, recover its bounding box, count the missing border cells on that perimeter, and select the corresponding legend color.
    - Fill every background cell strictly inside that fence's bounding box with the selected color.
    - For every gap, color the gap cell itself and continue outward perpendicular to that side until reaching the grid edge or a non-background input pixel.

    Key insight:
    - The same visible feature, the border gap pattern, serves two roles at once: its count indexes the bottom legend while each gap location also determines one outward ray direction.
    """
    g = np.asarray(grid, dtype=int)
    assert g.ndim == 2, "expected a 2D grid"
    h, w = g.shape
    bg = _get_bg(g)
    palette_row, palette = _find_palette_row(g, bg)

    fence_mask = g != bg
    fence_mask[palette_row, :] = False
    fence_colors = sorted({int(color) for color in g[fence_mask]})
    fences = []
    for color in fence_colors:
        pts = [tuple(map(int, rc)) for rc in np.argwhere((g == color) & fence_mask)]
        fences.append(_analyze_fence(g, pts, bg))
    assert fences, "expected at least one fence"

    out = g.copy()
    for fence in sorted(fences, key=lambda item: tuple(item["bbox"])):  # deterministic only
        r0, c0, r1, c1 = fence["bbox"]
        fill_color = int(palette[int(fence["gap_count"]) - 1])

        inner_in = g[r0 + 1 : r1, c0 + 1 : c1]
        inner_out = out[r0 + 1 : r1, c0 + 1 : c1]
        inner_out[inner_in == bg] = fill_color

        for gr, gc, dr, dc in fence["gaps"]:
            rr, cc = gr, gc
            while 0 <= rr < h and 0 <= cc < w and int(g[rr, cc]) == bg:
                out[rr, cc] = fill_color
                rr += dr
                cc += dc

    return out


def _split_length(total: int, parts: int, min_size: int, rng: random.Random) -> list[int]:
    assert total >= parts * min_size
    remaining = total - parts * min_size
    cuts = sorted(rng.sample(range(remaining + parts - 1), parts - 1))
    lengths: list[int] = []
    prev = -1
    for cut in cuts + [remaining + parts - 1]:
        lengths.append(cut - prev)
        prev = cut
    return [length + min_size - 1 for length in lengths]


def _choose_gap_counts(n_fences: int, rng: random.Random) -> list[int]:
    global _GEN_CALLS
    focus = (_GEN_CALLS % 4) + 1
    _GEN_CALLS += 1

    counts = [focus]
    others = [count for count in (1, 2, 3, 4) if count != focus]
    rng.shuffle(others)
    while len(counts) < n_fences and others:
        counts.append(others.pop())
    while len(counts) < n_fences:
        counts.append(rng.randint(1, 4))
    rng.shuffle(counts)
    return counts


def _gap_specs(
    r0: int, c0: int, box_h: int, box_w: int, gap_count: int, rng: random.Random
) -> list[tuple[int, int, int, int]]:
    r1 = r0 + box_h - 1
    c1 = c0 + box_w - 1
    gaps: list[tuple[int, int, int, int]] = []
    for side in rng.sample(SIDES, gap_count):
        if side == "top":
            gaps.append((r0, rng.randint(c0 + 1, c1 - 1), -1, 0))
        elif side == "bottom":
            gaps.append((r1, rng.randint(c0 + 1, c1 - 1), 1, 0))
        elif side == "left":
            gaps.append((rng.randint(r0 + 1, r1 - 1), c0, 0, -1))
        else:
            gaps.append((rng.randint(r0 + 1, r1 - 1), c1, 0, 1))
    return gaps


def _perimeter_cells(
    r0: int, c0: int, box_h: int, box_w: int, gaps: list[tuple[int, int, int, int]]
) -> set[tuple[int, int]]:
    r1 = r0 + box_h - 1
    c1 = c0 + box_w - 1
    cells = set()
    for c in range(c0, c1 + 1):
        cells.add((r0, c))
        cells.add((r1, c))
    for r in range(r0, r1 + 1):
        cells.add((r, c0))
        cells.add((r, c1))
    for gr, gc, _dr, _dc in gaps:
        cells.remove((gr, gc))
    return cells


def _validate_output(
    grid: np.ndarray, specs: list[dict[str, object]], palette: list[int]
) -> None:
    expected = grid.copy()
    h, w = grid.shape
    palette_row = h - 1
    bg = _get_bg(grid)
    for spec in specs:
        r0, c0, r1, c1 = spec["bbox"]
        fill_color = int(palette[int(spec["gap_count"]) - 1])
        window = grid[r0 + 1 : r1, c0 + 1 : c1]
        out_window = expected[r0 + 1 : r1, c0 + 1 : c1]
        out_window[window == bg] = fill_color
        for gr, gc, dr, dc in spec["gaps"]:
            for rr, cc in _ray_cells((gr, gc), (dr, dc), h, w, palette_row):
                assert int(grid[rr, cc]) == bg
                expected[rr, cc] = fill_color

    solved = main(grid)
    assert np.array_equal(solved, expected), "main() does not match generator expectations"
    assert int(np.count_nonzero(solved != grid)) >= 12, "output change is too small"


def _gen_attempt(rng: random.Random) -> np.ndarray | None:
    h = rng.randint(*GRID_H_RANGE)
    w = rng.randint(*GRID_W_RANGE)
    n_fences = rng.randint(*FENCE_COUNT_RANGE)

    bg = rng.choice(COLOR_POOL)
    remaining = [color for color in COLOR_POOL if color != bg]
    rng.shuffle(remaining)
    palette = remaining[:4]
    fence_colors = remaining[4 : 4 + n_fences]
    if len(fence_colors) < n_fences:
        return None

    grid = np.full((h, w), bg, dtype=int)
    palette_widths = _split_length(w, 4, 2, rng)
    col = 0
    for width, color in zip(palette_widths, palette):
        grid[h - 1, col : col + width] = color
        col += width

    occupied = np.zeros((h, w), dtype=bool)
    occupied[h - 1, :] = True
    reserved = np.zeros((h, w), dtype=bool)
    reserved[h - 1, :] = True

    specs: list[dict[str, object]] = []
    counts = _choose_gap_counts(n_fences, rng)
    for color, gap_count in zip(fence_colors, counts):
        placed = False
        for _ in range(PLACE_RETRIES):
            box_h = rng.randint(*BOX_H_RANGE)
            box_w = rng.randint(*BOX_W_RANGE)
            if box_h >= h - 4 or box_w >= w - 2:
                continue

            r0 = rng.randint(1, h - box_h - 3)
            c0 = rng.randint(1, w - box_w - 1)
            r1 = r0 + box_h - 1
            c1 = c0 + box_w - 1

            gaps = _gap_specs(r0, c0, box_h, box_w, gap_count, rng)
            border = _perimeter_cells(r0, c0, box_h, box_w, gaps)

            pad_r0 = max(0, r0 - 1)
            pad_c0 = max(0, c0 - 1)
            pad_r1 = min(h - 1, r1 + 1)
            pad_c1 = min(w - 1, c1 + 1)
            if reserved[pad_r0 : pad_r1 + 1, pad_c0 : pad_c1 + 1].any():
                continue

            ray_cells: list[tuple[int, int]] = []
            blocked = False
            for gr, gc, dr, dc in gaps:
                cells = _ray_cells((gr, gc), (dr, dc), h, w, h - 1)
                if len(cells) < 2:
                    blocked = True
                    break
                for rr, cc in cells[1:]:
                    if reserved[rr, cc]:
                        blocked = True
                        break
                if blocked:
                    break
                ray_cells.extend(cells)
            if blocked:
                continue

            for r, c in border:
                grid[r, c] = color
                occupied[r, c] = True
                reserved[r, c] = True

            reserved[pad_r0 : pad_r1 + 1, pad_c0 : pad_c1 + 1] = True
            for rr, cc in ray_cells:
                reserved[rr, cc] = True

            specs.append(
                {
                    "bbox": (r0, c0, r1, c1),
                    "gaps": gaps,
                    "gap_count": gap_count,
                }
            )
            placed = True
            break

        if not placed:
            return None

    try:
        _validate_output(grid, specs, palette)
    except AssertionError:
        return None
    return grid


def generate_input() -> np.ndarray:
    rng = random.Random()
    for _ in range(GEN_RETRIES):
        grid = _gen_attempt(rng)
        if grid is not None:
            return grid
    raise RuntimeError("generate_input: exhausted attempts")
