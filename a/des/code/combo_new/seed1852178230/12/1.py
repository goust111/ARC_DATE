from __future__ import annotations

from collections import deque

import numpy as np


BG_COLOR = 0
FRAME_COLOR = 5
PALETTE = (1, 2, 3, 4, 6, 7, 8, 9)
LEGEND_LEN_RANGE = (3, 7)
FRAME_COUNT_RANGE = (2, 5)
MAX_GEN_ATTEMPTS = 400


def _most_common_color(grid: np.ndarray) -> int:
    values, counts = np.unique(grid, return_counts=True)
    return int(values[np.argmax(counts)])


def _border_mask(height: int, width: int) -> np.ndarray:
    mask = np.zeros((height, width), dtype=bool)
    mask[0, :] = True
    mask[-1, :] = True
    mask[:, 0] = True
    mask[:, -1] = True
    return mask


def _components(mask: np.ndarray) -> list[np.ndarray]:
    height, width = mask.shape
    seen = np.zeros_like(mask, dtype=bool)
    comps: list[np.ndarray] = []
    for r in range(height):
        for c in range(width):
            if not bool(mask[r, c]) or bool(seen[r, c]):
                continue
            queue = deque([(r, c)])
            seen[r, c] = True
            coords: list[tuple[int, int]] = []
            while queue:
                rr, cc = queue.popleft()
                coords.append((rr, cc))
                for dr, dc in ((1, 0), (-1, 0), (0, 1), (0, -1)):
                    nr = rr + dr
                    nc = cc + dc
                    if 0 <= nr < height and 0 <= nc < width and bool(mask[nr, nc]) and not bool(seen[nr, nc]):
                        seen[nr, nc] = True
                        queue.append((nr, nc))
            comps.append(np.array(coords, dtype=int))
    return comps


def _find_legend_row(grid: np.ndarray, bg: int) -> int:
    height, _width = grid.shape
    for r in range(height - 1, -1, -1):
        cols = np.flatnonzero(grid[r] != bg)
        if cols.size < 3:
            continue
        colors = [int(grid[r, c]) for c in cols]
        if len(set(colors)) != len(colors):
            continue
        if r > 0 and any(int(grid[r - 1, c]) != bg for c in cols):
            continue
        return int(r)
    raise AssertionError("expected a unique bottom legend row")


def _find_frames(grid: np.ndarray, frame_color: int, stop_row: int) -> list[tuple[int, int, int, int]]:
    mask = grid[:stop_row] == frame_color
    frames: list[tuple[int, int, int, int]] = []
    for comp in _components(mask):
        rows = comp[:, 0]
        cols = comp[:, 1]
        r0 = int(rows.min())
        r1 = int(rows.max())
        c0 = int(cols.min())
        c1 = int(cols.max())
        height = r1 - r0 + 1
        width = c1 - c0 + 1
        if height < 3 or width < 3:
            continue
        border = _border_mask(height, width)
        patch = grid[r0 : r1 + 1, c0 : c1 + 1]
        if int(comp.shape[0]) != int(border.sum()):
            continue
        if np.array_equal(patch == frame_color, border):
            frames.append((r0, c0, r1, c1))
    assert frames, "expected at least one hollow rectangle frame"
    frames.sort()
    return frames


def main(grid: np.ndarray) -> np.ndarray:
    """
    Recolor each grey rectangular frame using the legend color indexed by its interior pixel count.

    Input structure:
    - The background is the dominant color.
    - The bottom row contains a legend made of distinct single-pixel color swatches arranged from left to right.
    - Above the legend are several grey 1-pixel-thick hollow rectangular frames, each containing some non-background single pixels.

    Rule:
    - For each grey frame, count how many non-background, non-grey pixels lie strictly inside it.
    - Use that count N as a 1-based index into the bottom legend row and read the N-th swatch color.
    - Repaint the entire grey border of that frame with the looked-up legend color.
    - Leave the interior pixels and the legend row unchanged.

    Key insight:
    - The border color is determined indirectly: first count interior pixels, then use that count to look up a legend color.
    """
    g = np.asarray(grid, dtype=int)
    bg = _most_common_color(g)
    legend_row = _find_legend_row(g, bg)
    legend_cols = np.flatnonzero(g[legend_row] != bg)
    legend_colors = [int(g[legend_row, c]) for c in legend_cols]

    upper = g[:legend_row]
    upper_non_bg = upper[upper != bg]
    assert upper_non_bg.size > 0, "expected framed region above the legend"
    values, counts = np.unique(upper_non_bg, return_counts=True)
    frame_color = int(values[np.argmax(counts)])

    out = g.copy()
    for r0, c0, r1, c1 in _find_frames(g, frame_color, legend_row):
        interior = g[r0 + 1 : r1, c0 + 1 : c1]
        count = int(np.count_nonzero((interior != bg) & (interior != frame_color)))
        assert 1 <= count <= len(legend_colors), "interior count must map into the legend"
        new_color = int(legend_colors[count - 1])
        border = _border_mask(r1 - r0 + 1, c1 - c0 + 1)
        patch = out[r0 : r1 + 1, c0 : c1 + 1]
        patch[border] = new_color

    return out


def _legend_positions(rng: np.random.Generator, width: int, legend_len: int) -> list[int]:
    min_total = 2 * legend_len + 1
    assert width >= min_total
    extra = int(width - min_total)
    gaps = np.ones(legend_len + 1, dtype=int)
    if extra > 0:
        gaps += rng.multinomial(extra, [1.0 / (legend_len + 1)] * (legend_len + 1))
    cols: list[int] = []
    c = int(gaps[0])
    for i in range(legend_len):
        cols.append(c)
        if i + 1 < legend_len:
            c += 1 + int(gaps[i + 1])
    return cols


def _checker_capacity(height: int, width: int) -> int:
    return int(((height + 1) // 2) * ((width + 1) // 2))


def _frame_size_candidates(count: int) -> list[tuple[int, int]]:
    candidates: list[tuple[int, int]] = []
    for inner_h in range(2, 8):
        for inner_w in range(2, 9):
            if _checker_capacity(inner_h, inner_w) >= count:
                candidates.append((inner_h + 2, inner_w + 2))
    return candidates


def _can_place(occupied: np.ndarray, r0: int, c0: int, height: int, width: int, pad: int = 1) -> bool:
    rr0 = max(0, r0 - pad)
    cc0 = max(0, c0 - pad)
    rr1 = min(occupied.shape[0], r0 + height + pad)
    cc1 = min(occupied.shape[1], c0 + width + pad)
    return not bool(occupied[rr0:rr1, cc0:cc1].any())


def _mark_occupied(occupied: np.ndarray, r0: int, c0: int, height: int, width: int) -> None:
    occupied[r0 : r0 + height, c0 : c0 + width] = True


def _sample_inner_cells(
    rng: np.random.Generator,
    inner_h: int,
    inner_w: int,
    count: int,
) -> list[tuple[int, int]] | None:
    cells = [(r, c) for r in range(inner_h) for c in range(inner_w)]
    for _ in range(20):
        rng.shuffle(cells)
        chosen: list[tuple[int, int]] = []
        for cell in cells:
            if all(max(abs(cell[0] - rr), abs(cell[1] - cc)) > 1 for rr, cc in chosen):
                chosen.append(cell)
                if len(chosen) == count:
                    return chosen
    return None


def _build_attempt() -> np.ndarray | None:
    rng = np.random.default_rng()

    legend_len = int(rng.integers(*LEGEND_LEN_RANGE))
    frame_count = int(rng.integers(*FRAME_COUNT_RANGE))
    counts = [int(rng.integers(1, legend_len + 1)) for _ in range(frame_count)]

    min_width = max(14, 2 * legend_len + 2)
    width = int(rng.integers(min_width, 31))
    height = int(rng.integers(12, 31))
    main_height = height - 2
    if main_height < 6:
        return None

    legend_colors = list(rng.choice(np.array(PALETTE, dtype=int), size=legend_len, replace=False))
    legend_cols = _legend_positions(rng, width, legend_len)

    frame_specs: list[dict[str, int]] = []
    for count in counts:
        candidates = _frame_size_candidates(count)
        if not candidates:
            return None
        frame_h, frame_w = candidates[int(rng.integers(0, len(candidates)))]
        frame_specs.append({"count": count, "h": int(frame_h), "w": int(frame_w)})

    occupied = np.zeros((main_height, width), dtype=bool)
    placements: list[dict[str, int]] = []
    for spec in sorted(frame_specs, key=lambda item: item["h"] * item["w"], reverse=True):
        placed = False
        for _ in range(80):
            max_r0 = main_height - spec["h"]
            max_c0 = width - spec["w"]
            if max_r0 < 0 or max_c0 < 0:
                return None
            r0 = int(rng.integers(0, max_r0 + 1))
            c0 = int(rng.integers(0, max_c0 + 1))
            if not _can_place(occupied, r0, c0, spec["h"], spec["w"], pad=1):
                continue
            _mark_occupied(occupied, r0, c0, spec["h"], spec["w"])
            placements.append({"r0": r0, "c0": c0, **spec})
            placed = True
            break
        if not placed:
            return None

    grid = np.full((height, width), BG_COLOR, dtype=int)
    out = grid.copy()
    grid[-1, legend_cols] = legend_colors
    out[-1, legend_cols] = legend_colors

    for placement in placements:
        r0 = int(placement["r0"])
        c0 = int(placement["c0"])
        frame_h = int(placement["h"])
        frame_w = int(placement["w"])
        count = int(placement["count"])
        inner_h = frame_h - 2
        inner_w = frame_w - 2

        border = _border_mask(frame_h, frame_w)
        patch = grid[r0 : r0 + frame_h, c0 : c0 + frame_w]
        patch[border] = FRAME_COLOR

        inner_cells = _sample_inner_cells(rng, inner_h, inner_w, count)
        if inner_cells is None:
            return None
        inner_colors = list(rng.choice(np.array(PALETTE, dtype=int), size=count, replace=False))
        for color, (rr, cc) in zip(inner_colors, inner_cells):
            grid[r0 + 1 + rr, c0 + 1 + cc] = int(color)

        border_color = int(legend_colors[count - 1])
        out_patch = out[r0 : r0 + frame_h, c0 : c0 + frame_w]
        out_patch[border] = border_color
        for color, (rr, cc) in zip(inner_colors, inner_cells):
            out[r0 + 1 + rr, c0 + 1 + cc] = int(color)

    if np.array_equal(main(grid), out):
        return grid
    return None


def generate_input() -> np.ndarray:
    for _ in range(MAX_GEN_ATTEMPTS):
        grid = _build_attempt()
        if grid is not None:
            return grid
    raise RuntimeError(f"failed to generate a valid input after {MAX_GEN_ATTEMPTS} attempts")
