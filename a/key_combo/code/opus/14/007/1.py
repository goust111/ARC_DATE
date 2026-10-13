from __future__ import annotations

import random

import numpy as np


BLACK = 0
BLUE = 1
MARKER_COLORS = (2, 3, 4, 6, 7, 8, 9)
GAP = 1
MAX_SIDE = 30
GEN_RETRIES = 300


FrameSpec = tuple[int, int, int, int, int, int, int]


def _neighbors4(r: int, c: int):
    yield r - 1, c
    yield r + 1, c
    yield r, c - 1
    yield r, c + 1


def _bbox(coords: np.ndarray) -> tuple[int, int, int, int]:
    r0, c0 = coords.min(axis=0)
    r1, c1 = coords.max(axis=0)
    return int(r0), int(c0), int(r1), int(c1)


def _components(mask: np.ndarray) -> list[np.ndarray]:
    seen = np.zeros(mask.shape, dtype=bool)
    comps: list[np.ndarray] = []
    h, w = mask.shape
    for r in range(h):
        for c in range(w):
            if not mask[r, c] or seen[r, c]:
                continue
            stack = [(r, c)]
            seen[r, c] = True
            cells = []
            while stack:
                cr, cc = stack.pop()
                cells.append((cr, cc))
                for nr, nc in _neighbors4(cr, cc):
                    if 0 <= nr < h and 0 <= nc < w and mask[nr, nc] and not seen[nr, nc]:
                        seen[nr, nc] = True
                        stack.append((nr, nc))
            comps.append(np.array(cells, dtype=int))
    return comps


def _extract_frames(grid: np.ndarray) -> list[FrameSpec]:
    blue_components = _components(grid == BLUE)
    assert blue_components, "expected at least one blue frame"

    frame_specs: list[FrameSpec] = []
    marker_cells = set()
    for coords in blue_components:
        r0, c0, r1, c1 = _bbox(coords)
        h = r1 - r0 + 1
        w = c1 - c0 + 1
        assert h >= 5 and w >= 5, "frame interior must be at least 3x3"
        assert len(coords) == 2 * h + 2 * w - 4, "blue component must be a hollow rectangle border"

        local = np.zeros((h, w), dtype=bool)
        local[coords[:, 0] - r0, coords[:, 1] - c0] = True
        border = np.zeros((h, w), dtype=bool)
        border[0, :] = True
        border[-1, :] = True
        border[:, 0] = True
        border[:, -1] = True
        assert np.array_equal(local, border), "blue component must match its bbox perimeter"

        interior = grid[r0 + 1 : r1, c0 + 1 : c1]
        marker_locs = np.argwhere((interior != BLACK) & (interior != BLUE))
        assert marker_locs.shape[0] == 1, "each frame must contain exactly one marker pixel"
        mr, mc = map(int, marker_locs[0])
        color = int(interior[mr, mc])

        cleared = interior.copy()
        cleared[mr, mc] = BLACK
        assert np.all(cleared == BLACK), "frame interior must otherwise be black"

        abs_marker = (r0 + 1 + mr, c0 + 1 + mc)
        frame_specs.append((r0, c0, r1, c1, abs_marker[0], abs_marker[1], color))
        marker_cells.add(abs_marker)

    colored = {
        (int(r), int(c))
        for r, c in np.argwhere((grid != BLACK) & (grid != BLUE))
    }
    assert colored == marker_cells, "all non-black non-blue pixels must be frame markers"
    return sorted(frame_specs)


def _apply_rule(grid: np.ndarray, frames: list[FrameSpec]) -> np.ndarray:
    out = grid.copy()
    for r0, c0, r1, c1, mr, mc, color in frames:
        rows = np.arange(r0 + 1, r1)[:, None]
        cols = np.arange(c0 + 1, c1)[None, :]
        view = out[r0 + 1 : r1, c0 + 1 : c1]
        fill_mask = (
            (view == BLACK)
            & (rows % 2 == mr % 2)
            & (cols % 2 == mc % 2)
        )
        view[fill_mask] = color
    return out


def main(grid: np.ndarray) -> np.ndarray:
    """Fill each hollow blue rectangle interior on the marker's absolute parity lattice while preserving borders and markers.

    Input structure:
    - A black background contains several disjoint one-cell-thick blue hollow rectangles.
    - Each rectangle has an interior of at least 3x3 cells.
    - Inside each rectangle there is exactly one non-black, non-blue marker pixel, and the rest of the interior is black.

    Rule:
    - Copy the input and process each blue frame independently.
    - Let the unique interior marker have color C and absolute coordinates (r, c).
    - Recolor each black interior cell of that frame to C iff its own absolute row parity and column parity match (r mod 2, c mod 2).
    - Leave the marker pixel, the blue border, and everything outside the frame unchanged.

    Key insight:
    - The marker encodes both the fill color and the checkerboard phase through its absolute row/column parity, so moving the marker by one cell changes which lattice points inside the same frame are painted.
    """
    g = np.asarray(grid, dtype=int)
    assert g.ndim == 2
    return _apply_rule(g, _extract_frames(g))


def _split_slack(extra: int, slots: int) -> list[int]:
    if slots == 1:
        return [extra]
    cuts = sorted(random.randint(0, extra) for _ in range(slots - 1))
    parts = []
    prev = 0
    for cut in cuts + [extra]:
        parts.append(cut - prev)
        prev = cut
    random.shuffle(parts)
    return parts


def _sample_outer_sizes(count: int) -> list[tuple[int, int]]:
    sizes = []
    for _ in range(count):
        inner_h = random.randint(3, 7)
        inner_w = random.randint(3, 7)
        sizes.append((inner_h + 2, inner_w + 2))

    if count >= 2 and all(h == w for h, w in sizes):
        idx = random.randrange(count)
        h, w = sizes[idx]
        candidates = [dim for dim in range(5, 10) if dim != h]
        sizes[idx] = (h, random.choice(candidates))
    return sizes


def _compositions(total: int, parts: int) -> list[tuple[int, ...]]:
    if parts == 1:
        return [(total,)]
    out = []
    for first in range(1, total - parts + 2):
        for rest in _compositions(total - first, parts - 1):
            out.append((first, *rest))
    return out


def _choose_layout(sizes: list[tuple[int, int]]) -> tuple[int, int, list[list[tuple[int, int]]]] | None:
    items = sizes[:]
    random.shuffle(items)
    feasible = []
    for row_count in range(1, min(3, len(items)) + 1):
        for counts in _compositions(len(items), row_count):
            rows = []
            cursor = 0
            for count in counts:
                row_items = items[cursor : cursor + count]
                random.shuffle(row_items)
                rows.append(row_items)
                cursor += count

            row_heights = [max(h for h, _ in row) for row in rows]
            row_widths = [sum(w for _, w in row) + GAP * (len(row) - 1) for row in rows]
            min_h = sum(row_heights) + GAP * (len(rows) - 1)
            min_w = max(row_widths)
            if min_h <= MAX_SIDE and min_w <= MAX_SIDE:
                feasible.append((min_h, min_w, rows))

    if not feasible:
        return None
    return random.choice(feasible)


def _choose_phase_plan(count: int) -> list[tuple[int, int]]:
    all_pairs = [(0, 0), (0, 1), (1, 0), (1, 1)]
    target_unique = min(4, max(2, count - 1))
    chosen = random.sample(all_pairs, target_unique)
    plan = chosen[:]
    while len(plan) < count:
        plan.append(random.choice(chosen))
    random.shuffle(plan)
    return plan


def _place_frames(rows: list[list[tuple[int, int]]], min_h: int, min_w: int) -> tuple[np.ndarray, list[FrameSpec]]:
    extra_h = random.randint(0, min(6, MAX_SIDE - min_h))
    extra_w = random.randint(0, min(6, MAX_SIDE - min_w))
    grid_h = min_h + extra_h
    grid_w = min_w + extra_w

    grid = np.zeros((grid_h, grid_w), dtype=int)
    specs: list[FrameSpec] = []
    phases = _choose_phase_plan(sum(len(row) for row in rows))
    colors = random.sample(MARKER_COLORS, len(phases))
    phase_idx = 0

    row_heights = [max(h for h, _ in row) for row in rows]
    used_h = sum(row_heights) + GAP * (len(rows) - 1)
    v_slack = _split_slack(grid_h - used_h, len(rows) + 1)
    r_cursor = v_slack[0]

    for row_idx, row in enumerate(rows):
        row_height = row_heights[row_idx]
        row_width = sum(w for _, w in row) + GAP * (len(row) - 1)
        h_slack = _split_slack(grid_w - row_width, len(row) + 1)
        c_cursor = h_slack[0]

        for item_idx, (outer_h, outer_w) in enumerate(row):
            r0 = r_cursor + random.randint(0, row_height - outer_h)
            c0 = c_cursor
            r1 = r0 + outer_h - 1
            c1 = c0 + outer_w - 1

            grid[r0, c0 : c1 + 1] = BLUE
            grid[r1, c0 : c1 + 1] = BLUE
            grid[r0 : r1 + 1, c0] = BLUE
            grid[r0 : r1 + 1, c1] = BLUE

            target_r_parity, target_c_parity = phases[phase_idx]
            color = colors[phase_idx]
            phase_idx += 1

            marker_options = [
                (r, c)
                for r in range(r0 + 1, r1)
                for c in range(c0 + 1, c1)
                if r % 2 == target_r_parity and c % 2 == target_c_parity
            ]
            assert marker_options
            mr, mc = random.choice(marker_options)
            grid[mr, mc] = color
            specs.append((r0, c0, r1, c1, mr, mc, color))

            c_cursor += outer_w + GAP + h_slack[item_idx + 1]
        r_cursor += row_height + GAP + v_slack[row_idx + 1]

    return grid, specs


def _nondegenerate(specs: list[FrameSpec]) -> bool:
    phases = {(mr % 2, mc % 2) for _, _, _, _, mr, mc, _ in specs}
    if len(phases) < min(2, len(specs)):
        return False

    shapes = {(r1 - r0 + 1, c1 - c0 + 1) for r0, c0, r1, c1, _, _, _ in specs}
    if len(specs) >= 2 and len(shapes) == 1:
        return False

    if len(specs) >= 2 and not any((r1 - r0) != (c1 - c0) for r0, c0, r1, c1, _, _, _ in specs):
        return False
    return True


def _attempt() -> np.ndarray | None:
    count = random.randint(3, 5)
    sizes = _sample_outer_sizes(count)
    layout = _choose_layout(sizes)
    if layout is None:
        return None
    min_h, min_w, rows = layout
    grid, specs = _place_frames(rows, min_h, min_w)
    if not _nondegenerate(specs):
        return None

    expected = _apply_rule(grid, specs)
    try:
        observed = main(grid)
    except AssertionError:
        return None
    if not np.array_equal(observed, expected):
        return None
    return grid


def generate_input() -> np.ndarray:
    for _ in range(GEN_RETRIES):
        grid = _attempt()
        if grid is not None:
            return grid
    raise RuntimeError("failed to generate a valid input")
