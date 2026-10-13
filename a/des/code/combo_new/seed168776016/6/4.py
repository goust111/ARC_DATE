from __future__ import annotations

import random

import numpy as np


BG = 0
PALETTE = tuple(range(1, 10))
DIRS = {
    "U": (-1, 0),
    "D": (1, 0),
    "L": (0, -1),
    "R": (0, 1),
}
SCENARIOS = ("edge", "block", "gap_block")
SOURCE_COUNT = 3
MAX_ATTEMPTS = 300
FRAME_RETRIES = 80
BAND_GAP = 1


def _get_bg(grid: np.ndarray) -> int:
    values, counts = np.unique(np.asarray(grid, dtype=int), return_counts=True)
    return int(values[int(np.argmax(counts))])


def _components(mask: np.ndarray) -> list[list[tuple[int, int]]]:
    mask = np.asarray(mask, dtype=bool)
    h, w = mask.shape
    seen = np.zeros_like(mask, dtype=bool)
    comps: list[list[tuple[int, int]]] = []
    for r in range(h):
        for c in range(w):
            if not mask[r, c] or seen[r, c]:
                continue
            stack = [(r, c)]
            seen[r, c] = True
            coords: list[tuple[int, int]] = []
            while stack:
                rr, cc = stack.pop()
                coords.append((int(rr), int(cc)))
                for dr, dc in DIRS.values():
                    nr = rr + dr
                    nc = cc + dc
                    if 0 <= nr < h and 0 <= nc < w and mask[nr, nc] and not seen[nr, nc]:
                        seen[nr, nc] = True
                        stack.append((nr, nc))
            comps.append(coords)
    return comps


def _bbox(coords: list[tuple[int, int]]) -> tuple[int, int, int, int]:
    rows = [r for r, _c in coords]
    cols = [c for _r, c in coords]
    return min(rows), min(cols), max(rows), max(cols)


def _border_mask(height: int, width: int) -> np.ndarray:
    mask = np.zeros((height, width), dtype=bool)
    mask[0, :] = True
    mask[-1, :] = True
    mask[:, 0] = True
    mask[:, -1] = True
    return mask


def _source_mask(direction: str, shift: int) -> np.ndarray:
    mask = np.zeros((3, 3), dtype=bool)
    if direction == "R":
        mask[:, 2] = True
        mask[shift : shift + 2, 0:2] = True
    elif direction == "L":
        mask[:, 0] = True
        mask[shift : shift + 2, 1:3] = True
    elif direction == "U":
        mask[0, :] = True
        mask[1:3, shift : shift + 2] = True
    else:
        assert direction == "D"
        mask[2, :] = True
        mask[0:2, shift : shift + 2] = True
    return mask


_SOURCE_MASKS = {
    (direction, shift): _source_mask(direction, shift)
    for direction in ("U", "D", "L", "R")
    for shift in (0, 1)
}
_SOURCE_PATTERNS = {
    tuple(int(v) for v in mask.ravel()): {"direction": direction, "shift": shift}
    for (direction, shift), mask in _SOURCE_MASKS.items()
}


def _source_start(top: int, left: int, direction: str) -> tuple[int, int]:
    if direction == "R":
        return top + 1, left + 3
    if direction == "L":
        return top + 1, left - 1
    if direction == "U":
        return top - 1, left + 1
    assert direction == "D"
    return top + 3, left + 1


def _classify_source(coords: list[tuple[int, int]], color: int) -> dict[str, object] | None:
    if len(coords) != 7:
        return None
    r0, c0, r1, c1 = _bbox(coords)
    if (r1 - r0 + 1, c1 - c0 + 1) != (3, 3):
        return None
    local = np.zeros((3, 3), dtype=int)
    for r, c in coords:
        local[r - r0, c - c0] = 1
    pattern = _SOURCE_PATTERNS.get(tuple(int(v) for v in local.ravel()))
    if pattern is None:
        return None
    return {
        "color": int(color),
        "direction": str(pattern["direction"]),
        "shift": int(pattern["shift"]),
        "bbox": (int(r0), int(c0), int(r1), int(c1)),
        "start": _source_start(int(r0), int(c0), str(pattern["direction"])),
        "cells": tuple(sorted((int(r), int(c)) for r, c in coords)),
    }


def _classify_frame(coords: list[tuple[int, int]], color: int) -> dict[str, object] | None:
    r0, c0, r1, c1 = _bbox(coords)
    height = r1 - r0 + 1
    width = c1 - c0 + 1
    if height < 3 or width < 3:
        return None
    local = np.zeros((height, width), dtype=bool)
    for r, c in coords:
        local[r - r0, c - c0] = True
    border = _border_mask(height, width)
    if np.any(local & ~border):
        return None
    missing = np.argwhere(border & ~local)
    if int(missing.shape[0]) > 1:
        return None
    gap = None
    if missing.shape[0] == 1:
        gap = (int(r0 + int(missing[0, 0])), int(c0 + int(missing[0, 1])))
    return {
        "color": int(color),
        "bbox": (int(r0), int(c0), int(r1), int(c1)),
        "gap": gap,
        "cells": tuple(sorted((int(r), int(c)) for r, c in coords)),
    }


def _parse_scene(grid: np.ndarray) -> dict[str, object]:
    g = np.asarray(grid, dtype=int)
    bg = _get_bg(g)
    sources: list[dict[str, object]] = []
    frames: list[dict[str, object]] = []
    frame_cells: dict[tuple[int, int], int] = {}
    gap_cells: set[tuple[int, int]] = set()
    source_cells: set[tuple[int, int]] = set()

    for color in sorted(int(v) for v in np.unique(g) if int(v) != bg):
        for comp in _components(g == color):
            source = _classify_source(comp, color)
            if source is not None:
                sources.append(source)
                source_cells |= set(source["cells"])
                continue

            frame = _classify_frame(comp, color)
            assert frame is not None, "unexpected foreground component"
            frames.append(frame)
            for cell in frame["cells"]:
                assert cell not in frame_cells, "frame borders must not overlap"
                frame_cells[cell] = int(color)
            if frame["gap"] is not None:
                gap = tuple(frame["gap"])
                assert int(g[gap]) == bg, "frame gap must stay background"
                gap_cells.add(gap)

    assert sources, "expected at least one source"
    return {
        "bg": int(bg),
        "sources": sources,
        "frames": frames,
        "frame_cells": frame_cells,
        "gap_cells": gap_cells,
        "source_cells": source_cells,
    }


def _trace_ray(
    grid: np.ndarray,
    bg: int,
    source: dict[str, object],
    frame_cells: dict[tuple[int, int], int],
    gap_cells: set[tuple[int, int]],
) -> tuple[list[tuple[int, int]], int]:
    g = np.asarray(grid, dtype=int)
    h, w = g.shape
    dr, dc = DIRS[str(source["direction"])]
    r, c = map(int, source["start"])
    path: list[tuple[int, int]] = []

    while 0 <= r < h and 0 <= c < w:
        cell = (int(r), int(c))
        if cell in gap_cells:
            assert int(g[cell]) == bg, "gap cell must remain background"
            r += dr
            c += dc
            continue

        value = int(g[cell])
        if value == bg:
            path.append(cell)
            r += dr
            c += dc
            continue

        blocker = frame_cells.get(cell)
        assert blocker is not None, "rays may only be blocked by frame borders"
        return path, int(blocker)

    return path, int(source["color"])


def main(grid: np.ndarray) -> np.ndarray:
    """
    Cast a straight ray from each arrow-marked source block, letting gap-aligned rays pass through frames and recoloring blocked rays to the blocking frame color.

    Input structure:
    - The background is black.
    - Several monochrome source components each contain a solid 2x2 block plus a same-color three-pixel side arrow that points left, right, up, or down.
    - Several other monochrome components are hollow rectangular frames with line width 1; each frame is either complete or has exactly one missing border pixel.

    Rule:
    - Keep every source block, arrow marker, and frame exactly as they are.
    - From each source, emit a one-pixel-wide straight ray starting one cell beyond the arrow tip and continuing along the arrow direction.
    - A ray paints background cells in its current color, skips any frame-gap cell without painting it, and stops immediately before the first non-gap frame border cell it hits.
    - If a frame border blocks the ray, recolor every already-painted pixel of that ray to the blocking frame's color; if the ray instead reaches the grid edge, it keeps the source color.

    Key insight:
    - The task requires combining three cues: the side-arrow geometry sets the ray direction, a one-pixel border gap allows a ray to pass that exact border cell only, and any eventual blocker retroactively changes the whole ray to the blocking frame color.
    """
    g = np.asarray(grid, dtype=int)
    assert g.ndim == 2, "expected a 2D grid"
    scene = _parse_scene(g)
    out = g.copy()
    painted: set[tuple[int, int]] = set()

    sources = sorted(
        scene["sources"],
        key=lambda item: (int(item["start"][0]), int(item["start"][1]), int(item["color"])),
    )
    for source in sources:
        path, ray_color = _trace_ray(
            g,
            int(scene["bg"]),
            source,
            scene["frame_cells"],
            scene["gap_cells"],
        )
        for cell in path:
            assert cell not in painted, "rays must not overlap in valid inputs"
            painted.add(cell)
        if path:
            rows = [r for r, _c in path]
            cols = [c for _r, c in path]
            out[rows, cols] = int(ray_color)
    return out


def _stamp_source(grid: np.ndarray, top: int, left: int, direction: str, shift: int, color: int) -> None:
    mask = _SOURCE_MASKS[(direction, shift)]
    patch = grid[top : top + 3, left : left + 3]
    assert patch.shape == (3, 3)
    assert np.all(patch == BG), "source area must be empty"
    patch[mask] = int(color)


def _stamp_frame(
    grid: np.ndarray,
    top: int,
    left: int,
    height: int,
    width: int,
    color: int,
    gap: tuple[int, int] | None = None,
) -> None:
    assert height >= 4 and width >= 4
    border = _border_mask(height, width)
    region = grid[top : top + height, left : left + width]
    assert region.shape == (height, width)
    assert np.all(region[border] == BG), "frame border must be empty before placement"
    region[border] = int(color)
    if gap is not None:
        gr, gc = gap
        assert top <= gr < top + height and left <= gc < left + width
        assert gr in (top, top + height - 1) or gc in (left, left + width - 1)
        assert (gr, gc) not in (
            (top, left),
            (top, left + width - 1),
            (top + height - 1, left),
            (top + height - 1, left + width - 1),
        )
        grid[gr, gc] = BG


def _place_frame_horizontal(
    rng: random.Random,
    width: int,
    band_top: int,
    band_height: int,
    ray_row: int,
    direction: str,
    source_left: int,
    scenario: str,
) -> tuple[int, int, int, int, tuple[int, int] | None] | None:
    assert scenario in ("block", "gap_block")
    for _ in range(FRAME_RETRIES):
        frame_h = rng.randint(4, min(6, band_height))
        frame_w = rng.randint(4, min(8, width - 2))
        top_options = [
            top
            for top in range(band_top, band_top + band_height - frame_h + 1)
            if top + 1 <= ray_row <= top + frame_h - 2
        ]
        if not top_options:
            continue
        top = rng.choice(top_options)

        if direction == "R":
            start_c = source_left + 3
            min_left = start_c + 2
            max_left = width - frame_w - 1
            if min_left > max_left:
                continue
            left = rng.randint(min_left, max_left)
            gap = (ray_row, left) if scenario == "gap_block" else None
        else:
            assert direction == "L"
            start_c = source_left - 1
            max_right = start_c - 2
            max_left = max_right - frame_w + 1
            if 1 > max_left:
                continue
            left = rng.randint(1, max_left)
            gap = (ray_row, left + frame_w - 1) if scenario == "gap_block" else None

        return int(top), int(left), int(frame_h), int(frame_w), gap
    return None


def _place_frame_vertical(
    rng: random.Random,
    height: int,
    band_left: int,
    band_width: int,
    ray_col: int,
    direction: str,
    source_top: int,
    scenario: str,
) -> tuple[int, int, int, int, tuple[int, int] | None] | None:
    assert scenario in ("block", "gap_block")
    for _ in range(FRAME_RETRIES):
        frame_w = rng.randint(4, min(6, band_width))
        frame_h = rng.randint(4, min(8, height - 2))
        left_options = [
            left
            for left in range(band_left, band_left + band_width - frame_w + 1)
            if left + 1 <= ray_col <= left + frame_w - 2
        ]
        if not left_options:
            continue
        left = rng.choice(left_options)

        if direction == "D":
            start_r = source_top + 3
            min_top = start_r + 2
            max_top = height - frame_h - 1
            if min_top > max_top:
                continue
            top = rng.randint(min_top, max_top)
            gap = (top, ray_col) if scenario == "gap_block" else None
        else:
            assert direction == "U"
            start_r = source_top - 1
            max_bottom = start_r - 2
            max_top = max_bottom - frame_h + 1
            if 1 > max_top:
                continue
            top = rng.randint(1, max_top)
            gap = (top + frame_h - 1, ray_col) if scenario == "gap_block" else None

        return int(top), int(left), int(frame_h), int(frame_w), gap
    return None


def _generate_horizontal(rng: random.Random) -> np.ndarray | None:
    band_height = rng.randint(7, 8)
    top_pad = rng.randint(0, 2)
    bottom_pad = rng.randint(0, 2)
    height = top_pad + SOURCE_COUNT * band_height + (SOURCE_COUNT - 1) * BAND_GAP + bottom_pad
    width = rng.randint(20, 30)
    if max(height, width) > 30:
        return None

    grid = np.full((height, width), BG, dtype=int)
    colors = rng.sample(list(PALETTE), 5)
    source_colors = colors[:SOURCE_COUNT]
    frame_colors = colors[SOURCE_COUNT:]
    frame_idx = 0
    scenarios = list(SCENARIOS)
    rng.shuffle(scenarios)

    for idx, scenario in enumerate(scenarios):
        band_top = top_pad + idx * (band_height + BAND_GAP)
        ray_row = band_top + rng.randint(2, band_height - 3)
        direction = rng.choice(["L", "R"])
        shift = rng.randint(0, 1)
        margin = rng.randint(1, 3)
        source_left = margin if direction == "R" else width - 3 - margin
        _stamp_source(grid, ray_row - 1, source_left, direction, shift, source_colors[idx])

        if scenario == "edge":
            continue

        frame = _place_frame_horizontal(
            rng,
            width,
            band_top,
            band_height,
            ray_row,
            direction,
            source_left,
            scenario,
        )
        if frame is None:
            return None
        top, left, frame_h, frame_w, gap = frame
        _stamp_frame(grid, top, left, frame_h, frame_w, frame_colors[frame_idx], gap)
        frame_idx += 1

    return grid


def _generate_vertical(rng: random.Random) -> np.ndarray | None:
    band_width = rng.randint(7, 8)
    left_pad = rng.randint(0, 2)
    right_pad = rng.randint(0, 2)
    width = left_pad + SOURCE_COUNT * band_width + (SOURCE_COUNT - 1) * BAND_GAP + right_pad
    height = rng.randint(20, 30)
    if max(height, width) > 30:
        return None

    grid = np.full((height, width), BG, dtype=int)
    colors = rng.sample(list(PALETTE), 5)
    source_colors = colors[:SOURCE_COUNT]
    frame_colors = colors[SOURCE_COUNT:]
    frame_idx = 0
    scenarios = list(SCENARIOS)
    rng.shuffle(scenarios)

    for idx, scenario in enumerate(scenarios):
        band_left = left_pad + idx * (band_width + BAND_GAP)
        ray_col = band_left + rng.randint(2, band_width - 3)
        direction = rng.choice(["U", "D"])
        shift = rng.randint(0, 1)
        margin = rng.randint(1, 3)
        source_top = margin if direction == "D" else height - 3 - margin
        _stamp_source(grid, source_top, ray_col - 1, direction, shift, source_colors[idx])

        if scenario == "edge":
            continue

        frame = _place_frame_vertical(
            rng,
            height,
            band_left,
            band_width,
            ray_col,
            direction,
            source_top,
            scenario,
        )
        if frame is None:
            return None
        top, left, frame_h, frame_w, gap = frame
        _stamp_frame(grid, top, left, frame_h, frame_w, frame_colors[frame_idx], gap)
        frame_idx += 1

    return grid


def _generate_attempt(rng: random.Random) -> np.ndarray | None:
    grid = _generate_horizontal(rng) if rng.random() < 0.5 else _generate_vertical(rng)
    if grid is None:
        return None

    try:
        parsed = _parse_scene(grid)
        out = main(grid)
    except AssertionError:
        return None

    if int(parsed["bg"]) != BG:
        return None
    if len(parsed["sources"]) != SOURCE_COUNT:
        return None
    if len(parsed["frames"]) != 2:
        return None
    if np.array_equal(out, grid):
        return None
    return grid


def generate_input() -> np.ndarray:
    rng = random.Random()
    for _ in range(MAX_ATTEMPTS):
        grid = _generate_attempt(rng)
        if grid is not None:
            return grid
    raise RuntimeError(f"failed to generate a valid input after {MAX_ATTEMPTS} attempts")
