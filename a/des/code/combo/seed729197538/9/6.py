from __future__ import annotations

import random

import numpy as np


BG = 7  # orange
PIPE = 5  # grey
MARKER_COLORS = (1, 2, 3, 6)  # blue, red, green, magenta
ORIENTATIONS = ("lb", "rb", "lt", "rt")  # grey sides: left/right + bottom/top


def _components4(mask: np.ndarray) -> list[list[tuple[int, int]]]:
    h, w = map(int, mask.shape)
    seen = np.zeros((h, w), dtype=bool)
    comps: list[list[tuple[int, int]]] = []
    for r in range(h):
        for c in range(w):
            if not mask[r, c] or seen[r, c]:
                continue
            stack = [(r, c)]
            seen[r, c] = True
            comp: list[tuple[int, int]] = []
            while stack:
                rr, cc = stack.pop()
                comp.append((rr, cc))
                for nr, nc in ((rr - 1, cc), (rr + 1, cc), (rr, cc - 1), (rr, cc + 1)):
                    if 0 <= nr < h and 0 <= nc < w and mask[nr, nc] and not seen[nr, nc]:
                        seen[nr, nc] = True
                        stack.append((nr, nc))
            comps.append(comp)
    return comps


def _pipe_mask(height: int, width: int, orient: str) -> np.ndarray:
    assert height >= 4 and width >= 4
    mask = np.zeros((height, width), dtype=bool)
    if orient[0] == "l":
        mask[:, :2] = True
    else:
        mask[:, width - 2 :] = True
    if orient[1] == "b":
        mask[height - 2 :, :] = True
    else:
        mask[:2, :] = True
    return mask


def _match_orientation(local_mask: np.ndarray) -> str:
    h, w = map(int, local_mask.shape)
    matches = [orient for orient in ORIENTATIONS if np.array_equal(local_mask, _pipe_mask(h, w, orient))]
    assert len(matches) == 1, "grey component must be a width-2 L pipe"
    return matches[0]


def _fill_paths(height: int, width: int, orient: str) -> tuple[list[tuple[int, int]], list[tuple[int, int]]]:
    if orient == "lb":
        vertical = [(r, 2) for r in range(0, height - 2)]
        horizontal = [(height - 3, c) for c in range(width - 1, 1, -1)]
    elif orient == "rb":
        vertical = [(r, width - 3) for r in range(0, height - 2)]
        horizontal = [(height - 3, c) for c in range(0, width - 2)]
    elif orient == "lt":
        vertical = [(r, 2) for r in range(height - 1, 1, -1)]
        horizontal = [(2, c) for c in range(width - 1, 1, -1)]
    elif orient == "rt":
        vertical = [(r, width - 3) for r in range(height - 1, 1, -1)]
        horizontal = [(2, c) for c in range(0, width - 2)]
    else:
        raise AssertionError("unknown pipe orientation")
    return vertical, horizontal


def _solve_component(g: np.ndarray, out: np.ndarray, comp: list[tuple[int, int]]) -> None:
    coords = np.array(comp, dtype=int)
    r0, c0 = map(int, coords.min(axis=0))
    r1, c1 = map(int, coords.max(axis=0))
    height = r1 - r0 + 1
    width = c1 - c0 + 1
    box = g[r0 : r1 + 1, c0 : c1 + 1]

    local_mask = np.zeros((height, width), dtype=bool)
    local_mask[coords[:, 0] - r0, coords[:, 1] - c0] = True
    orient = _match_orientation(local_mask)
    cavity = ~local_mask

    marker_values = box[cavity & (box != BG)]
    colors = {int(v) for v in marker_values if int(v) != PIPE}
    assert len(colors) == 1, "each pipe cavity must contain one marker color"
    fill_color = colors.pop()
    count = int(np.count_nonzero(marker_values == fill_color))
    assert 1 <= count <= 4, "marker count must be between 1 and 4"

    out_box = out[r0 : r1 + 1, c0 : c1 + 1]
    out_box[cavity & (box != BG) & (box != PIPE)] = BG
    out_box[local_mask] = fill_color

    path_a, path_b = _fill_paths(height, width, orient)
    assert count <= len(path_a) and count <= len(path_b), "pipe arms must be long enough for the fill depth"
    for rr, cc in path_a[:count]:
        out_box[rr, cc] = fill_color
    for rr, cc in path_b[:count]:
        out_box[rr, cc] = fill_color


def main(grid: np.ndarray) -> np.ndarray:
    """
    Recolor each L-shaped pipe with its marker color, then fill inward from both openings by the marker count.

    Input structure:
    - Background is orange.
    - Grey pixels form one or more width-2 solid L-shapes.
    - Each L encloses one rectangular cavity that touches the grid exterior on two sides.
    - The cavity contains 1 to 4 marker pixels of a single color.

    Rule:
    - Count the marker pixels inside each pipe cavity.
    - Recolor the whole grey pipe to the marker color.
    - From each opening, fill that many pixels along the cavity path toward the elbow.
    - Remove the original marker pixels unless a fill path overwrites them.
    """
    g = np.asarray(grid, dtype=int)
    assert g.ndim == 2, "expected a 2D grid"

    out = g.copy()
    for comp in _components4(g == PIPE):
        _solve_component(g, out, comp)
    return out


# --------------- generator ---------------

GRID_MIN = 18
GRID_MAX = 30
PIPE_SIDE_MIN = 6
PIPE_SIDE_MAX = 11
PIPE_COUNT_CHOICES = (1, 2, 2, 3, 3, 4)
COUNT_MIN = 1
COUNT_MAX = 4
PLACEMENT_TRIES = 80
GENERATION_TRIES = 400
PADDING = 1
EDGE_TOUCH_PROB = 0.4
OFF_PATH_MARKER_PROB = 0.8


def _open_sides(orient: str) -> tuple[str, str]:
    if orient == "lb":
        return "top", "right"
    if orient == "rb":
        return "top", "left"
    if orient == "lt":
        return "bottom", "right"
    if orient == "rt":
        return "bottom", "left"
    raise AssertionError("unknown pipe orientation")


def _free_with_padding(occupied: np.ndarray, r0: int, c0: int, height: int, width: int) -> bool:
    h, w = map(int, occupied.shape)
    rs = max(0, r0 - PADDING)
    re = min(h, r0 + height + PADDING)
    cs = max(0, c0 - PADDING)
    ce = min(w, c0 + width + PADDING)
    return not bool(np.any(occupied[rs:re, cs:ce]))


def _origin_candidates(occupied: np.ndarray, height: int, width: int, orient: str) -> list[tuple[int, int]]:
    h, w = map(int, occupied.shape)
    all_candidates = [
        (r0, c0)
        for r0 in range(h - height + 1)
        for c0 in range(w - width + 1)
        if _free_with_padding(occupied, r0, c0, height, width)
    ]
    if not all_candidates:
        return []

    if random.random() >= EDGE_TOUCH_PROB:
        return all_candidates

    side = random.choice(_open_sides(orient))
    edge_candidates: list[tuple[int, int]] = []
    for r0, c0 in all_candidates:
        if side == "top" and r0 == 0:
            edge_candidates.append((r0, c0))
        elif side == "bottom" and r0 + height == h:
            edge_candidates.append((r0, c0))
        elif side == "left" and c0 == 0:
            edge_candidates.append((r0, c0))
        elif side == "right" and c0 + width == w:
            edge_candidates.append((r0, c0))
    return edge_candidates or all_candidates


def _marker_positions(height: int, width: int, orient: str, count: int) -> list[tuple[int, int]]:
    mask = _pipe_mask(height, width, orient)
    cavity = [(r, c) for r in range(height) for c in range(width) if not mask[r, c]]
    path_a, path_b = _fill_paths(height, width, orient)
    path_cells = set(path_a + path_b)
    off_path = [cell for cell in cavity if cell not in path_cells]
    pool = off_path if len(off_path) >= count and random.random() < OFF_PATH_MARKER_PROB else cavity
    return random.sample(pool, count)


def _paint_pipe(grid: np.ndarray, pipe: dict[str, object]) -> None:
    r0 = int(pipe["r0"])
    c0 = int(pipe["c0"])
    height = int(pipe["height"])
    width = int(pipe["width"])
    orient = str(pipe["orient"])
    color = int(pipe["color"])

    mask = _pipe_mask(height, width, orient)
    box = grid[r0 : r0 + height, c0 : c0 + width]
    box[mask] = PIPE
    for rr, cc in pipe["markers"]:
        box[int(rr), int(cc)] = color


def _expected_output(grid: np.ndarray, pipes: list[dict[str, object]]) -> np.ndarray:
    out = grid.copy()
    for pipe in pipes:
        r0 = int(pipe["r0"])
        c0 = int(pipe["c0"])
        height = int(pipe["height"])
        width = int(pipe["width"])
        orient = str(pipe["orient"])
        color = int(pipe["color"])
        markers = [(int(r), int(c)) for r, c in pipe["markers"]]

        box = out[r0 : r0 + height, c0 : c0 + width]
        for rr, cc in markers:
            box[rr, cc] = BG

        mask = _pipe_mask(height, width, orient)
        box[mask] = color

        path_a, path_b = _fill_paths(height, width, orient)
        count = len(markers)
        for rr, cc in path_a[:count]:
            box[rr, cc] = color
        for rr, cc in path_b[:count]:
            box[rr, cc] = color
    return out


def _sample_grid_size(pipe_count: int) -> tuple[int, int]:
    low = max(GRID_MIN, 10 + pipe_count * 4)
    return random.randint(low, GRID_MAX), random.randint(low, GRID_MAX)


def _place_pipe(occupied: np.ndarray, color: int) -> dict[str, object] | None:
    for _ in range(PLACEMENT_TRIES):
        height = random.randint(PIPE_SIDE_MIN, PIPE_SIDE_MAX)
        width = random.randint(PIPE_SIDE_MIN, PIPE_SIDE_MAX)
        orient = random.choice(ORIENTATIONS)
        count = random.randint(COUNT_MIN, COUNT_MAX)
        candidates = _origin_candidates(occupied, height, width, orient)
        if not candidates:
            continue
        r0, c0 = random.choice(candidates)
        markers = _marker_positions(height, width, orient, count)
        occupied[r0 : r0 + height, c0 : c0 + width] = True
        return {
            "r0": r0,
            "c0": c0,
            "height": height,
            "width": width,
            "orient": orient,
            "color": color,
            "markers": markers,
        }
    return None


def _gen_attempt() -> np.ndarray | None:
    pipe_count = random.choice(PIPE_COUNT_CHOICES)
    colors = random.sample(MARKER_COLORS, pipe_count)
    height, width = _sample_grid_size(pipe_count)
    occupied = np.zeros((height, width), dtype=bool)
    pipes: list[dict[str, object]] = []

    for color in colors:
        pipe = _place_pipe(occupied, color)
        if pipe is None:
            return None
        pipes.append(pipe)

    grid = np.full((height, width), BG, dtype=int)
    for pipe in pipes:
        _paint_pipe(grid, pipe)

    expected = _expected_output(grid, pipes)
    try:
        actual = main(grid)
    except Exception:
        return None
    if np.array_equal(actual, expected):
        return grid
    return None


def generate_input() -> np.ndarray:
    for _ in range(GENERATION_TRIES):
        grid = _gen_attempt()
        if grid is not None:
            return grid
    raise RuntimeError("failed to generate a valid pipe scene")
