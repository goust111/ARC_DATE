from __future__ import annotations

import random

import numpy as np


BG = 0
YELLOW = 4
AZURE = 8
PATTERN_COLORS = (1, 2, 3, 5, 6, 7, 9)
DIR_VECTORS = {
    "up": (-1, 0),
    "down": (1, 0),
    "left": (0, -1),
    "right": (0, 1),
}
GEN_ATTEMPTS = 400


def _components(mask: np.ndarray) -> list[np.ndarray]:
    h, w = mask.shape
    seen = np.zeros((h, w), dtype=bool)
    comps: list[np.ndarray] = []
    for r in range(h):
        for c in range(w):
            if not mask[r, c] or seen[r, c]:
                continue
            stack = [(r, c)]
            seen[r, c] = True
            cells: list[tuple[int, int]] = []
            while stack:
                rr, cc = stack.pop()
                cells.append((rr, cc))
                for dr, dc in ((-1, 0), (1, 0), (0, -1), (0, 1)):
                    nr = rr + dr
                    nc = cc + dc
                    if 0 <= nr < h and 0 <= nc < w and mask[nr, nc] and not seen[nr, nc]:
                        seen[nr, nc] = True
                        stack.append((nr, nc))
            comps.append(np.array(cells, dtype=int))
    return comps


def _bbox(coords: np.ndarray) -> tuple[int, int, int, int]:
    r0, c0 = coords.min(axis=0)
    r1, c1 = coords.max(axis=0)
    return int(r0), int(c0), int(r1), int(c1)


def _frame_direction(frame_mask: np.ndarray) -> str | None:
    h, w = frame_mask.shape
    if h < 3 or w < 3:
        return None

    border = np.zeros((h, w), dtype=bool)
    border[0, :] = True
    border[-1, :] = True
    border[:, 0] = True
    border[:, -1] = True
    if np.any(frame_mask & ~border):
        return None

    candidates: list[tuple[str, np.ndarray]] = []

    up = border.copy()
    up[0, 1:-1] = False
    candidates.append(("up", up))

    down = border.copy()
    down[-1, 1:-1] = False
    candidates.append(("down", down))

    left = border.copy()
    left[1:-1, 0] = False
    candidates.append(("left", left))

    right = border.copy()
    right[1:-1, -1] = False
    candidates.append(("right", right))

    matches = [name for name, candidate in candidates if np.array_equal(frame_mask, candidate)]
    if len(matches) != 1:
        return None
    return matches[0]


def main(grid: np.ndarray) -> np.ndarray:
    """
    Read each azure open frame as a directional launcher and replicate its inner pattern outward by the count of yellow markers.

    Input structure:
    - The background is black.
    - One or more azure single-pixel rectangular frames appear as 4-connected border components.
    - Each frame keeps all four corners but is missing exactly one full interior side segment, creating an opening on the top, bottom, left, or right.
    - Inside each frame there is a small colored pattern plus 1 to 4 yellow singleton markers on otherwise black interior cells.

    Rule:
    - For each azure frame, identify which side is open; that opening direction is the launch direction.
    - Count the yellow markers inside the frame to get N.
    - Copy only the non-black, non-yellow, non-azure interior pattern cells N times along the launch direction.
    - The kth copy is translated by k times the frame's outer height for vertical launches or outer width for horizontal launches, while the original frame and its contents stay unchanged.

    Key insight:
    - The missing side is not damage but an encoded exit direction, and the number of yellow interior markers controls how many frame-sized translations of the inner pattern are emitted.
    """
    g = np.asarray(grid, dtype=int)
    assert g.ndim == 2
    h, w = g.shape

    out = g.copy()
    frame_components = _components(g == AZURE)
    assert frame_components, "expected at least one azure frame"

    for coords in frame_components:
        r0, c0, r1, c1 = _bbox(coords)
        frame_mask = g[r0 : r1 + 1, c0 : c1 + 1] == AZURE
        direction = _frame_direction(frame_mask)
        assert direction is not None, "azure component must be an open rectangular frame"

        interior = g[r0 + 1 : r1, c0 + 1 : c1]
        marker_count = int(np.count_nonzero(interior == YELLOW))
        assert 1 <= marker_count <= 4

        pattern_mask = (interior != BG) & (interior != YELLOW) & (interior != AZURE)
        pattern_cells = np.argwhere(pattern_mask)
        if pattern_cells.size == 0:
            continue

        dr, dc = DIR_VECTORS[direction]
        step = (r1 - r0 + 1) if dr != 0 else (c1 - c0 + 1)
        for k in range(1, marker_count + 1):
            shift_r = dr * step * k
            shift_c = dc * step * k
            for ir, ic in pattern_cells:
                rr = int(r0 + 1 + ir + shift_r)
                cc = int(c0 + 1 + ic + shift_c)
                if 0 <= rr < h and 0 <= cc < w:
                    out[rr, cc] = int(interior[int(ir), int(ic)])

    return out


def _random_pattern(ih: int, iw: int, max_cells: int) -> list[tuple[int, int]]:
    target = random.randint(2, min(max_cells, ih * iw))
    start = (random.randrange(ih), random.randrange(iw))
    cells = {start}
    frontier = {start}
    while len(cells) < target:
        base_r, base_c = random.choice(tuple(frontier))
        neighbors = []
        for dr, dc in ((-1, 0), (1, 0), (0, -1), (0, 1)):
            nr = base_r + dr
            nc = base_c + dc
            if 0 <= nr < ih and 0 <= nc < iw and (nr, nc) not in cells:
                neighbors.append((nr, nc))
        if not neighbors:
            frontier.remove((base_r, base_c))
            if not frontier:
                frontier = set(cells)
            continue
        chosen = random.choice(neighbors)
        cells.add(chosen)
        frontier.add(chosen)
    return sorted(cells)


def _sample_window_spec(color: int) -> dict | None:
    for _ in range(100):
        outer_h = random.randint(5, 8)
        outer_w = random.randint(5, 8)
        inner_h = outer_h - 2
        inner_w = outer_w - 2
        marker_count = random.randint(1, 4)
        if inner_h * inner_w - marker_count < 2:
            continue
        pattern_cells = _random_pattern(inner_h, inner_w, min(6, inner_h * inner_w - marker_count))
        pattern_set = set(pattern_cells)
        free_cells = [(r, c) for r in range(inner_h) for c in range(inner_w) if (r, c) not in pattern_set]
        if len(free_cells) < marker_count:
            continue
        marker_cells = random.sample(free_cells, marker_count)
        return {
            "outer_h": outer_h,
            "outer_w": outer_w,
            "direction": random.choice(tuple(DIR_VECTORS.keys())),
            "marker_count": marker_count,
            "pattern_color": int(color),
            "pattern_cells": pattern_cells,
            "marker_cells": marker_cells,
            "prefer_clip": bool(random.getrandbits(1)),
        }
    return None


def _window_footprint(shape: tuple[int, int], r0: int, c0: int, spec: dict) -> tuple[np.ndarray, int, bool]:
    h, w = shape
    outer_h = int(spec["outer_h"])
    outer_w = int(spec["outer_w"])
    direction = str(spec["direction"])
    marker_count = int(spec["marker_count"])
    dr, dc = DIR_VECTORS[direction]
    step = outer_h if dr != 0 else outer_w

    mask = np.zeros((h, w), dtype=bool)
    visible_copies = 0
    clipped = False
    for k in range(marker_count + 1):
        rr = r0 + dr * step * k
        cc = c0 + dc * step * k
        rs = max(0, rr)
        re = min(h, rr + outer_h)
        cs = max(0, cc)
        ce = min(w, cc + outer_w)
        if rs < re and cs < ce:
            mask[rs:re, cs:ce] = True
            if k >= 1:
                visible_copies += 1
                if rr < 0 or cc < 0 or rr + outer_h > h or cc + outer_w > w:
                    clipped = True
    return mask, visible_copies, clipped


def _expand_mask(mask: np.ndarray) -> np.ndarray:
    h, w = mask.shape
    out = mask.copy()
    for r, c in np.argwhere(mask):
        r0 = max(0, int(r) - 1)
        r1 = min(h, int(r) + 2)
        c0 = max(0, int(c) - 1)
        c1 = min(w, int(c) + 2)
        out[r0:r1, c0:c1] = True
    return out


def _render_window_patch(spec: dict) -> np.ndarray:
    outer_h = int(spec["outer_h"])
    outer_w = int(spec["outer_w"])
    patch = np.full((outer_h, outer_w), BG, dtype=int)
    patch[0, :] = AZURE
    patch[-1, :] = AZURE
    patch[:, 0] = AZURE
    patch[:, -1] = AZURE

    direction = str(spec["direction"])
    if direction == "up":
        patch[0, 1:-1] = BG
    elif direction == "down":
        patch[-1, 1:-1] = BG
    elif direction == "left":
        patch[1:-1, 0] = BG
    else:
        patch[1:-1, -1] = BG

    for r, c in spec["pattern_cells"]:
        patch[1 + int(r), 1 + int(c)] = int(spec["pattern_color"])
    for r, c in spec["marker_cells"]:
        patch[1 + int(r), 1 + int(c)] = YELLOW
    return patch


def _apply_expected(output: np.ndarray, spec: dict, r0: int, c0: int) -> None:
    outer_h = int(spec["outer_h"])
    outer_w = int(spec["outer_w"])
    direction = str(spec["direction"])
    marker_count = int(spec["marker_count"])
    color = int(spec["pattern_color"])
    dr, dc = DIR_VECTORS[direction]
    step = outer_h if dr != 0 else outer_w
    for k in range(1, marker_count + 1):
        shift_r = dr * step * k
        shift_c = dc * step * k
        for pr, pc in spec["pattern_cells"]:
            rr = int(r0 + 1 + pr + shift_r)
            cc = int(c0 + 1 + pc + shift_c)
            if 0 <= rr < output.shape[0] and 0 <= cc < output.shape[1]:
                output[rr, cc] = color


def _grid_shape(window_count: int) -> tuple[int, int]:
    min_dim = 14 + 4 * window_count
    return random.randint(min_dim, 30), random.randint(min_dim, 30)


def _generate_attempt() -> np.ndarray | None:
    window_count = random.randint(1, 3)
    grid_h, grid_w = _grid_shape(window_count)
    colors = random.sample(PATTERN_COLORS, window_count)

    specs: list[dict] = []
    for color in colors:
        spec = _sample_window_spec(color)
        if spec is None:
            return None
        specs.append(spec)
    specs.sort(key=lambda spec: int(spec["outer_h"]) * int(spec["outer_w"]) * int(spec["marker_count"]), reverse=True)

    blocked = np.zeros((grid_h, grid_w), dtype=bool)
    placements: list[tuple[dict, int, int]] = []
    for spec in specs:
        outer_h = int(spec["outer_h"])
        outer_w = int(spec["outer_w"])
        candidates: list[tuple[int, int, np.ndarray, int, bool]] = []
        for r0 in range(grid_h - outer_h + 1):
            for c0 in range(grid_w - outer_w + 1):
                footprint, visible_copies, clipped = _window_footprint((grid_h, grid_w), r0, c0, spec)
                if visible_copies == 0:
                    continue
                if np.any(footprint & blocked):
                    continue
                candidates.append((r0, c0, footprint, visible_copies, clipped))
        if not candidates:
            return None

        if bool(spec["prefer_clip"]):
            preferred = [cand for cand in candidates if cand[4]]
        else:
            preferred = [cand for cand in candidates if cand[3] == int(spec["marker_count"])]
        if preferred:
            candidates = preferred

        r0, c0, footprint, _visible_copies, _clipped = random.choice(candidates)
        placements.append((spec, r0, c0))
        blocked |= _expand_mask(footprint)

    grid = np.full((grid_h, grid_w), BG, dtype=int)
    for spec, r0, c0 in placements:
        patch = _render_window_patch(spec)
        target = grid[r0 : r0 + patch.shape[0], c0 : c0 + patch.shape[1]]
        if np.any(target != BG):
            return None
        target[:, :] = patch

    expected = grid.copy()
    for spec, r0, c0 in placements:
        _apply_expected(expected, spec, r0, c0)

    if not np.array_equal(main(grid), expected):
        return None
    return grid


def generate_input() -> np.ndarray:
    for _ in range(GEN_ATTEMPTS):
        grid = _generate_attempt()
        if grid is not None:
            return grid
    raise RuntimeError("failed to generate a valid launcher-frame ARC input")
