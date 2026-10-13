from __future__ import annotations

import numpy as np

from arc_utils import find_components, get_bg, paste
import random


def _component_mask(shape: tuple[int, int], coords: np.ndarray) -> np.ndarray:
    mask = np.zeros(shape, dtype=bool)
    mask[coords[:, 0], coords[:, 1]] = True
    return mask


def _rectangular_frame_bbox(grid_shape: tuple[int, int], coords: np.ndarray) -> tuple[int, int, int, int] | None:
    mask = _component_mask(grid_shape, coords)
    r0, c0 = coords.min(axis=0)
    r1, c1 = coords.max(axis=0)
    r0, c0, r1, c1 = map(int, (r0, c0, r1, c1))
    height = r1 - r0 + 1
    width = c1 - c0 + 1
    if height < 3 or width < 3:
        return None

    ring = np.zeros((height, width), dtype=bool)
    ring[0, :] = True
    ring[-1, :] = True
    ring[:, 0] = True
    ring[:, -1] = True
    if not np.array_equal(mask[r0 : r1 + 1, c0 : c1 + 1], ring):
        return None
    return r0, c0, r1, c1


def _checkerboard_capacity(bbox: tuple[int, int, int, int]) -> int:
    r0, c0, r1, c1 = bbox
    interior_height = r1 - r0 - 1
    interior_width = c1 - c0 - 1
    return (interior_height * interior_width + 1) // 2


def main(grid: np.ndarray) -> np.ndarray:
    """
    Move each scattered color into the rectangular frame whose interior checkerboard has the same cell count.

    Input structure:
    - The background is the most common color.
    - Foreground contains one or more one-cell-thick rectangular frames.
    - All non-frame foreground pixels are isolated markers grouped by color.

    Rule:
    - For each frame, count the cells of the checkerboard pattern inside its border,
      starting at the upper-left interior cell.
    - Match that frame to the unique outside marker color with the same number of pixels.
    - Erase all outside markers and paint the matched color into the frame interior's
      upper-left-starting checkerboard cells.

    Key insight:
    - The scattered pixels are not placed by their coordinates; their color populations
      are counts that select which frame interior they should fill.
    """
    g = np.asarray(grid, dtype=int)
    assert g.ndim == 2, "expected a 2D grid"

    bg = get_bg(g)
    frame_mask = np.zeros(g.shape, dtype=bool)
    frames: list[tuple[int, tuple[int, int, int, int]]] = []
    for comp in find_components(g, bg=bg, connectivity=4, monochromatic=True):
        bbox = _rectangular_frame_bbox(g.shape, comp["coords"])
        if bbox is None:
            assert comp["area"] == 1, "non-frame components must be isolated marker pixels"
            continue
        capacity = _checkerboard_capacity(bbox)
        frames.append((capacity, bbox))
        frame_mask[comp["coords"][:, 0], comp["coords"][:, 1]] = True

    assert frames, "expected at least one rectangular frame"
    capacities = [capacity for capacity, _bbox in frames]
    assert len(set(capacities)) == len(capacities), "frame checkerboard counts must be unique"

    outside_mask = (g != bg) & ~frame_mask
    source_by_count: dict[int, int] = {}
    for color in sorted(set(map(int, g.ravel())) - {bg}):
        count = int(np.sum((g == color) & outside_mask))
        if count == 0:
            continue
        assert count not in source_by_count, "outside color counts must be unique"
        source_by_count[count] = color

    assert set(source_by_count) == set(capacities), "outside counts must match frame capacities"

    out = g.copy()
    out[outside_mask] = bg
    for capacity, (r0, c0, r1, c1) in frames:
        color = source_by_count[capacity]
        for r in range(r0 + 1, r1):
            for c in range(c0 + 1, c1):
                if (r - r0 - 1 + c - c0 - 1) % 2 == 0:
                    assert out[r, c] == bg, "frame interior must be empty before filling"
                    out[r, c] = color
                else:
                    assert out[r, c] == bg, "unused checkerboard cells must be background"
    return out

# --------------- generator (o5x) ---------------

# Canvas, palette, and retry bounds.
GEN_GRID_HEIGHT_RANGE = (12, 22)
GEN_GRID_WIDTH_RANGE = (20, 26)
GEN_BACKGROUND_COLOR = 0
GEN_COLOR_VALUES = tuple(range(1, 10))
GEN_MAX_ATTEMPTS = 300

# Frame counts and geometry.
GEN_FRAME_COUNT_RANGE = (2, 3)
GEN_INTERIOR_HEIGHTS = tuple(range(1, 7))
GEN_INTERIOR_WIDTHS = tuple(range(1, 10))
GEN_FRAME_MARGIN = 1
GEN_FRAME_BORDER = 1

# Marker scattering.
GEN_SPARSE_CELL_RATIO = 3
GEN_BLOCKED_MARKER_OFFSETS = ((0, 0), (-1, 0), (1, 0), (0, -1), (0, 1))


def _gen_capacity_map() -> dict[int, list[tuple[int, int]]]:
    sizes: dict[int, list[tuple[int, int]]] = {}
    for ih in GEN_INTERIOR_HEIGHTS:
        for iw in GEN_INTERIOR_WIDTHS:
            cap = (ih * iw + 1) // 2
            sizes.setdefault(cap, []).append((ih, iw))
    return sizes


def _gen_weighted_capacity_sample(sizes_by_capacity: dict[int, list[tuple[int, int]]], n: int) -> list[int]:
    pool = list(sizes_by_capacity)
    picked: list[int] = []
    for _ in range(n):
        weights = [len(sizes_by_capacity[cap]) for cap in pool]
        cap = random.choices(pool, weights=weights, k=1)[0]
        picked.append(cap)
        pool.remove(cap)
    return picked


def _gen_frame_patch(ih: int, iw: int, frame_color: int, marker_color: int) -> np.ndarray:
    border = GEN_FRAME_BORDER
    patch = np.zeros((ih + 2 * border, iw + 2 * border), dtype=int)
    patch[:border, :] = frame_color
    patch[-border:, :] = frame_color
    patch[:, :border] = frame_color
    patch[:, -border:] = frame_color
    interior = patch[border:-border, border:-border]
    interior[0::2, 0::2] = marker_color
    interior[1::2, 1::2] = marker_color
    return patch


def _gen_valid_frame_locations(occupied: np.ndarray, oh: int, ow: int) -> list[tuple[int, int]]:
    h, w = occupied.shape
    locs: list[tuple[int, int]] = []
    max_r = h - oh - GEN_FRAME_MARGIN + 1
    max_c = w - ow - GEN_FRAME_MARGIN + 1
    for r in range(GEN_FRAME_MARGIN, max_r):
        for c in range(GEN_FRAME_MARGIN, max_c):
            if not occupied[r : r + oh, c : c + ow].any():
                locs.append((r, c))
    return locs


def _gen_build_frames(n_frames: int, sizes_by_capacity: dict[int, list[tuple[int, int]]]) -> list[dict[str, object]]:
    capacities = _gen_weighted_capacity_sample(sizes_by_capacity, n_frames)
    colors = random.sample(GEN_COLOR_VALUES, 2 * n_frames)

    frames: list[dict[str, object]] = []
    for cap, frame_color, marker_color in zip(capacities, colors[:n_frames], colors[n_frames:]):
        ih, iw = random.choice(sizes_by_capacity[cap])
        frames.append(
            {
                "ih": ih,
                "iw": iw,
                "capacity": cap,
                "marker_color": marker_color,
                "patch": _gen_frame_patch(ih, iw, frame_color, marker_color),
            }
        )
    return frames


def _gen_place_frames(output: np.ndarray, frames: list[dict[str, object]]) -> bool:
    occupied = np.zeros(output.shape, dtype=bool)
    for frame in sorted(frames, key=lambda f: int(f["ih"]) * int(f["iw"]), reverse=True):
        patch = frame["patch"]
        oh, ow = patch.shape
        locs = _gen_valid_frame_locations(occupied, oh, ow)
        if not locs:
            return False
        r0, c0 = random.choice(locs)
        paste(output, patch, r0, c0)
        occupied[r0 : r0 + oh, c0 : c0 + ow] = True
        frame["bbox"] = (r0, c0, r0 + oh - 1, c0 + ow - 1)
    return True


def _gen_sparse_marker_cells(cells: list[tuple[int, int]], count: int) -> list[tuple[int, int]] | None:
    candidates = set((int(r), int(c)) for r, c in cells)
    chosen: list[tuple[int, int]] = []
    while len(chosen) < count:
        if not candidates:
            return None
        r, c = random.choice(tuple(candidates))
        chosen.append((r, c))
        for dr, dc in GEN_BLOCKED_MARKER_OFFSETS:
            candidates.discard((r + dr, c + dc))
    return chosen


def _gen_parity_marker_cells(cells: list[tuple[int, int]], count: int) -> list[tuple[int, int]] | None:
    parities = [0, 1]
    random.shuffle(parities)
    for parity in parities:
        parity_cells = [(int(r), int(c)) for r, c in cells if (int(r) + int(c)) % 2 == parity]
        if len(parity_cells) >= count:
            random.shuffle(parity_cells)
            return parity_cells[:count]
    return None


def _gen_scatter_markers(grid: np.ndarray, frames: list[dict[str, object]]) -> bool:
    allowed = grid == GEN_BACKGROUND_COLOR
    for frame in frames:
        r0, c0, r1, c1 = frame["bbox"]
        allowed[r0 : r1 + 1, c0 : c1 + 1] = False

    jobs: list[int] = []
    for frame in frames:
        jobs.extend([int(frame["marker_color"])] * int(frame["capacity"]))
    random.shuffle(jobs)

    cells = list(zip(*np.where(allowed)))
    if len(cells) < len(jobs):
        return False

    if len(jobs) * GEN_SPARSE_CELL_RATIO <= len(cells):
        chosen = _gen_sparse_marker_cells(cells, len(jobs))
    else:
        chosen = _gen_parity_marker_cells(cells, len(jobs))
    if chosen is None:
        return False

    for (r, c), color in zip(chosen, jobs):
        grid[r, c] = color
    return True


def _gen_attempt() -> np.ndarray | None:
    height = random.randint(*GEN_GRID_HEIGHT_RANGE)
    width = random.randint(*GEN_GRID_WIDTH_RANGE)
    n_frames = random.randint(*GEN_FRAME_COUNT_RANGE)

    sizes_by_capacity = _gen_capacity_map()
    frames = _gen_build_frames(n_frames, sizes_by_capacity)

    output = np.zeros((height, width), dtype=int)
    if not _gen_place_frames(output, frames):
        return None
    inp = output.copy()

    for frame in frames:
        r0, c0, r1, c1 = frame["bbox"]
        inp[r0 + GEN_FRAME_BORDER : r1, c0 + GEN_FRAME_BORDER : c1] = GEN_BACKGROUND_COLOR
    if not _gen_scatter_markers(inp, frames):
        return None
    try:
        return inp if np.array_equal(main(inp), output) else None
    except Exception:
        return None


def generate_input() -> np.ndarray:
    for _ in range(GEN_MAX_ATTEMPTS):
        grid = _gen_attempt()
        if grid is not None:
            return grid
    raise RuntimeError("failed to generate df8cc377 input")
