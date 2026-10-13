from __future__ import annotations

import random
from collections import deque

import numpy as np


BG = 0
GEN_FRAME_COLOR = 3
PATTERN_POOL = (1, 2, 4, 7, 8, 9)
MARKER_POOL = (5, 6)
SIDE_TO_STEP = {
    "top": (-1, 0),
    "bottom": (1, 0),
    "left": (0, -1),
    "right": (0, 1),
}
DIR4 = ((1, 0), (-1, 0), (0, 1), (0, -1))
GEN_ATTEMPTS = 400


def _components_of_color(grid: np.ndarray, color: int) -> list[np.ndarray]:
    h, w = grid.shape
    seen = np.zeros((h, w), dtype=bool)
    comps: list[np.ndarray] = []
    for r in range(h):
        for c in range(w):
            if seen[r, c] or int(grid[r, c]) != color:
                continue
            seen[r, c] = True
            q: deque[tuple[int, int]] = deque([(r, c)])
            coords: list[tuple[int, int]] = []
            while q:
                rr, cc = q.popleft()
                coords.append((rr, cc))
                for dr, dc in DIR4:
                    nr, nc = rr + dr, cc + dc
                    if 0 <= nr < h and 0 <= nc < w and not seen[nr, nc] and int(grid[nr, nc]) == color:
                        seen[nr, nc] = True
                        q.append((nr, nc))
            comps.append(np.array(coords, dtype=int))
    return comps


def _bbox(coords: np.ndarray) -> tuple[int, int, int, int]:
    r0, c0 = coords.min(axis=0)
    r1, c1 = coords.max(axis=0)
    return int(r0), int(c0), int(r1), int(c1)


def _parse_frame(grid: np.ndarray, color: int, coords: np.ndarray) -> dict[str, object] | None:
    r0, c0, r1, c1 = _bbox(coords)
    height = r1 - r0 + 1
    width = c1 - c0 + 1
    if height < 3 or width < 3:
        return None

    perimeter = 2 * (height + width) - 4
    if int(coords.shape[0]) != perimeter - 1:
        return None

    coord_set = {tuple(map(int, rc)) for rc in coords}
    for rr, cc in coord_set:
        if rr not in (r0, r1) and cc not in (c0, c1):
            return None

    missing: list[tuple[int, int]] = []
    for rr in range(r0, r1 + 1):
        for cc in range(c0, c1 + 1):
            on_perimeter = rr in (r0, r1) or cc in (c0, c1)
            if not on_perimeter:
                continue
            if int(grid[rr, cc]) != color:
                missing.append((rr, cc))
    if len(missing) != 1:
        return None

    gap = missing[0]
    if int(grid[gap]) != BG:
        return None
    if gap in ((r0, c0), (r0, c1), (r1, c0), (r1, c1)):
        return None

    if gap[0] == r0:
        side = "top"
    elif gap[0] == r1:
        side = "bottom"
    elif gap[1] == c0:
        side = "left"
    else:
        side = "right"

    interior = grid[r0 + 1 : r1, c0 + 1 : c1]
    inner_colors = set(map(int, np.unique(interior))) - {BG}
    if color in inner_colors or len(inner_colors) != 1:
        return None

    return {
        "bbox": (r0, c0, r1, c1),
        "gap": gap,
        "side": side,
        "pattern_color": next(iter(inner_colors)),
    }


def _detect_frames(grid: np.ndarray) -> tuple[int, list[dict[str, object]]]:
    colors = [int(c) for c in np.unique(grid) if int(c) != BG]
    candidates: list[tuple[int, list[dict[str, object]]]] = []
    for color in colors:
        comps = _components_of_color(grid, color)
        if not comps:
            continue
        parsed = [_parse_frame(grid, color, comp) for comp in comps]
        if all(meta is not None for meta in parsed):
            frames = [meta for meta in parsed if meta is not None]
            if frames:
                frames.sort(key=lambda item: tuple(item["bbox"]))  # type: ignore[index]
                candidates.append((color, frames))
    assert len(candidates) == 1, "expected exactly one frame color"
    return candidates[0]


def _ray_cells_from_gap(gap: tuple[int, int], step: tuple[int, int], shape: tuple[int, int]) -> list[tuple[int, int]]:
    h, w = shape
    r = gap[0] + step[0]
    c = gap[1] + step[1]
    cells: list[tuple[int, int]] = []
    while 0 <= r < h and 0 <= c < w:
        cells.append((r, c))
        r += step[0]
        c += step[1]
    return cells


def main(grid: np.ndarray) -> np.ndarray:
    """
    Project each framed inner motif outward through its unique border gap, preserving marker pixels on the ray.

    Input structure:
    - The background is black.
    - One shared frame color forms several rectangular one-pixel-thick frames; each frame has exactly one one-pixel gap on one side.
    - Inside each frame there is a monochrome motif color, and outside along the gap direction there may be singleton marker pixels of other colors.

    Rule:
    - Detect the missing border pixel of each rectangular frame to determine whether its projection goes up, down, left, or right.
    - Cast a one-pixel-wide ray from the cell immediately outside that gap to the grid edge.
    - Recolor every background cell on that ray with the frame's inner motif color, but keep any pre-existing non-background marker pixels unchanged so the painted ray passes through them.

    Key insight:
    - The border gap chooses the ray direction, while exterior singleton color blocks are preserved separators rather than obstacles.
    """
    g = np.asarray(grid, dtype=int)
    assert g.ndim == 2, "expected a 2D grid"

    _frame_color, frames = _detect_frames(g)
    out = g.copy()
    for frame in frames:
        step = SIDE_TO_STEP[frame["side"]]  # type: ignore[index]
        for r, c in _ray_cells_from_gap(frame["gap"], step, g.shape):  # type: ignore[arg-type]
            if int(g[r, c]) == BG:
                out[r, c] = int(frame["pattern_color"])
    return out


def _balanced_binary_choices(n: int, a: object, b: object, rng: random.Random) -> list[object]:
    if n % 2 == 0:
        items = [a] * (n // 2) + [b] * (n // 2)
    else:
        majority = a if rng.random() < 0.5 else b
        minority = b if majority == a else a
        items = [majority] * ((n + 1) // 2) + [minority] * (n // 2)
    rng.shuffle(items)
    return items


def _sample_bands(
    total: int,
    n_bands: int,
    min_size: int,
    max_size: int,
    gap: int,
    rng: random.Random,
) -> list[tuple[int, int]] | None:
    for _ in range(200):
        sizes = [rng.randint(min_size, max_size) for _ in range(n_bands)]
        needed = sum(sizes) + gap * (n_bands - 1)
        if needed > total:
            continue
        extra = total - needed
        pads = [0] * (n_bands + 1)
        for _ in range(extra):
            pads[rng.randrange(n_bands + 1)] += 1
        bands: list[tuple[int, int]] = []
        pos = pads[0]
        for i, size in enumerate(sizes):
            bands.append((pos, pos + size - 1))
            pos += size + gap + pads[i + 1]
        return bands
    return None


def _grow_connected_shape(height: int, width: int, area: int, rng: random.Random) -> list[tuple[int, int]]:
    start = (rng.randrange(height), rng.randrange(width))
    cells = {start}
    while len(cells) < area:
        frontier = set()
        for r, c in cells:
            for dr, dc in DIR4:
                rr, cc = r + dr, c + dc
                if 0 <= rr < height and 0 <= cc < width and (rr, cc) not in cells:
                    frontier.add((rr, cc))
        if not frontier:
            break
        cells.add(rng.choice(tuple(frontier)))
    return sorted(cells)


def _draw_frame(grid: np.ndarray, r0: int, c0: int, r1: int, c1: int, gap: tuple[int, int], color: int) -> None:
    grid[r0, c0 : c1 + 1] = color
    grid[r1, c0 : c1 + 1] = color
    grid[r0 : r1 + 1, c0] = color
    grid[r0 : r1 + 1, c1] = color
    grid[gap] = BG


def _place_pattern(grid: np.ndarray, r0: int, c0: int, r1: int, c1: int, color: int, rng: random.Random) -> None:
    inner_h = r1 - r0 - 1
    inner_w = c1 - c0 - 1
    patch_h = max(1, inner_h - 2)
    patch_w = max(1, inner_w - 2)
    area = rng.randint(1, min(4, patch_h * patch_w))
    coords = _grow_connected_shape(patch_h, patch_w, area, rng)
    base_r = r0 + 2
    base_c = c0 + 2
    for rr, cc in coords:
        grid[base_r + rr, base_c + cc] = color


def _sample_marker_indices(ray_len: int, count: int, rng: random.Random) -> list[int]:
    if count == 0:
        return []
    pools = [list(range(1, ray_len - 1, 2)), list(range(2, ray_len - 1, 2))]
    pools = [pool for pool in pools if len(pool) >= count]
    assert pools, "insufficient ray length for requested markers"
    return sorted(rng.sample(rng.choice(pools), count))


def _expected_output(grid: np.ndarray, specs: list[dict[str, object]]) -> np.ndarray:
    out = grid.copy()
    for spec in specs:
        marker_set = set(spec["markers"])  # type: ignore[arg-type]
        for r, c in spec["ray_cells"]:  # type: ignore[index]
            if (r, c) not in marker_set and int(grid[r, c]) == BG:
                out[r, c] = int(spec["pattern_color"])
    return out


def _gen_attempt(rng: random.Random) -> np.ndarray | None:
    n_frames = rng.randint(2, 4)
    orientation = rng.choice(("vertical", "horizontal"))
    band_min = 5
    band_max = 7
    band_gap = 2
    axis_need = band_min * n_frames + band_gap * (n_frames - 1)

    if orientation == "vertical":
        h = rng.randint(15, 30)
        w_min = max(18, axis_need)
        if w_min > 30:
            return None
        w = rng.randint(w_min, 30)
        bands = _sample_bands(w, n_frames, band_min, band_max, band_gap, rng)
        sides = _balanced_binary_choices(n_frames, "top", "bottom", rng)
    else:
        w = rng.randint(15, 30)
        h_min = max(18, axis_need)
        if h_min > 30:
            return None
        h = rng.randint(h_min, 30)
        bands = _sample_bands(h, n_frames, band_min, band_max, band_gap, rng)
        sides = _balanced_binary_choices(n_frames, "left", "right", rng)

    if bands is None:
        return None

    grid = np.full((h, w), BG, dtype=int)
    pattern_colors = rng.sample(PATTERN_POOL, n_frames)
    marker_flags = _balanced_binary_choices(n_frames, False, True, rng)
    specs: list[dict[str, object]] = []

    for i in range(n_frames):
        side = str(sides[i])
        wants_markers = bool(marker_flags[i])
        pattern_color = pattern_colors[i]

        if orientation == "vertical":
            band_c0, band_c1 = bands[i]
            frame_w = rng.randint(5, min(7, band_c1 - band_c0 + 1))
            frame_h = rng.randint(5, 7)
            c0 = rng.randint(band_c0, band_c1 - frame_w + 1)
            min_ray = 4 if wants_markers else 2
            if side == "top":
                r0_min = min_ray
                r0_max = h - frame_h - 1
            else:
                r0_min = 1
                r0_max = h - frame_h - min_ray
            if r0_min > r0_max:
                return None
            r0 = rng.randint(r0_min, r0_max)
            r1 = r0 + frame_h - 1
            c1 = c0 + frame_w - 1
            gap = (r0, rng.randint(c0 + 1, c1 - 1)) if side == "top" else (r1, rng.randint(c0 + 1, c1 - 1))
        else:
            band_r0, band_r1 = bands[i]
            frame_h = rng.randint(5, min(7, band_r1 - band_r0 + 1))
            frame_w = rng.randint(5, 7)
            r0 = rng.randint(band_r0, band_r1 - frame_h + 1)
            min_ray = 4 if wants_markers else 2
            if side == "left":
                c0_min = min_ray
                c0_max = w - frame_w - 1
            else:
                c0_min = 1
                c0_max = w - frame_w - min_ray
            if c0_min > c0_max:
                return None
            c0 = rng.randint(c0_min, c0_max)
            r1 = r0 + frame_h - 1
            c1 = c0 + frame_w - 1
            gap = (rng.randint(r0 + 1, r1 - 1), c0) if side == "left" else (rng.randint(r0 + 1, r1 - 1), c1)

        _draw_frame(grid, r0, c0, r1, c1, gap, GEN_FRAME_COLOR)
        _place_pattern(grid, r0, c0, r1, c1, pattern_color, rng)

        ray_cells = _ray_cells_from_gap(gap, SIDE_TO_STEP[side], grid.shape)
        if len(ray_cells) < min_ray:
            return None

        marker_capacity = max(len(range(1, len(ray_cells) - 1, 2)), len(range(2, len(ray_cells) - 1, 2)))
        marker_count = 0 if not wants_markers else rng.randint(1, min(3, marker_capacity))
        marker_indices = _sample_marker_indices(len(ray_cells), marker_count, rng)
        markers: list[tuple[int, int]] = []
        for idx in marker_indices:
            mr, mc = ray_cells[idx]
            grid[mr, mc] = rng.choice(MARKER_POOL)
            markers.append((mr, mc))

        specs.append(
            {
                "bbox": (r0, c0, r1, c1),
                "gap": gap,
                "side": side,
                "pattern_color": pattern_color,
                "ray_cells": ray_cells,
                "markers": markers,
            }
        )

    if not np.any(grid == GEN_FRAME_COLOR):
        return None

    expected = _expected_output(grid, specs)
    parsed_color, frames = _detect_frames(grid)
    if parsed_color != GEN_FRAME_COLOR or len(frames) != n_frames:
        return None
    if np.array_equal(expected, grid):
        return None
    if not np.array_equal(main(grid), expected):
        return None
    return grid


def generate_input() -> np.ndarray:
    rng = random.Random()
    for _ in range(GEN_ATTEMPTS):
        grid = _gen_attempt(rng)
        if grid is not None:
            return grid
    raise RuntimeError("failed to generate a valid input")
