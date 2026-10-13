from __future__ import annotations

import random

import numpy as np


BG = 0
COLORS = tuple(range(1, 10))
ORTHOGONALS = ((1, 0), (-1, 0), (0, 1), (0, -1))
GENERATION_ATTEMPTS = 400
_RNG = random.Random()


def _components(mask: np.ndarray) -> list[list[tuple[int, int]]]:
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
                coords.append((rr, cc))
                for dr, dc in ORTHOGONALS:
                    nr, nc = rr + dr, cc + dc
                    if 0 <= nr < h and 0 <= nc < w and mask[nr, nc] and not seen[nr, nc]:
                        seen[nr, nc] = True
                        stack.append((nr, nc))
            comps.append(coords)
    return comps


def _bbox(coords: list[tuple[int, int]]) -> tuple[int, int, int, int]:
    rows = [r for r, _ in coords]
    cols = [c for _, c in coords]
    return min(rows), min(cols), max(rows), max(cols)


def _is_rectangular_frame(mask: np.ndarray, box: tuple[int, int, int, int]) -> bool:
    r0, c0, r1, c1 = box
    local = mask[r0 : r1 + 1, c0 : c1 + 1]
    h, w = local.shape
    if h < 3 or w < 3:
        return False
    expected = np.zeros((h, w), dtype=bool)
    expected[0, :] = True
    expected[-1, :] = True
    expected[:, 0] = True
    expected[:, -1] = True
    return np.array_equal(local, expected)


def _extract_regions(grid: np.ndarray) -> list[tuple[tuple[int, int, int, int], int, int, tuple[int, int]]]:
    g = np.asarray(grid, dtype=int)
    assert g.ndim == 2, "expected a 2D grid"

    handled = np.zeros_like(g, dtype=bool)
    regions: list[tuple[tuple[int, int, int, int], int, int, tuple[int, int]]] = []

    for color in np.unique(g):
        color = int(color)
        if color == BG:
            continue
        mask = g == color
        for coords in _components(mask):
            box = _bbox(coords)
            if not _is_rectangular_frame(mask, box):
                continue

            r0, c0, r1, c1 = box
            interior = g[r0 + 1 : r1, c0 + 1 : c1]
            non_bg = np.argwhere(interior != BG)
            assert len(non_bg) == 1, "each frame interior must contain exactly one non-black marker"
            mr, mc = map(int, non_bg[0])
            marker_color = int(interior[mr, mc])
            assert marker_color != color, "marker color must differ from the frame color"

            marker_pos = (r0 + 1 + mr, c0 + 1 + mc)
            handled[r0, c0 : c1 + 1] = True
            handled[r1, c0 : c1 + 1] = True
            handled[r0 : r1 + 1, c0] = True
            handled[r0 : r1 + 1, c1] = True
            handled[marker_pos] = True
            regions.append((box, color, marker_color, marker_pos))

    assert regions, "expected at least one framed region"
    assert np.array_equal(handled, g != BG), "every non-black pixel must belong to a frame or its marker"
    return regions


def main(grid: np.ndarray) -> np.ndarray:
    """
    Swap each hollow rectangle's border color with its unique interior marker color.

    Input structure:
    - The background is black.
    - The grid contains several one-pixel-thick closed rectangular frames on the background.
    - Each frame encloses exactly one non-black marker pixel whose color differs from the frame color.
    - Separate frames do not overlap or touch.

    Rule:
    - Detect every rectangular frame and the unique marker inside its interior.
    - Recolor the entire border of that frame to the marker's color.
    - Replace the marker pixel itself with the frame's original border color.
    - Leave every black background pixel unchanged.

    Key insight:
    - The single interior marker identifies the new border color, while the old border color moves back into the marker cell.
    """
    g = np.asarray(grid, dtype=int)
    regions = _extract_regions(g)
    out = g.copy()

    for (r0, c0, r1, c1), frame_color, marker_color, (mr, mc) in regions:
        out[r0, c0 : c1 + 1] = marker_color
        out[r1, c0 : c1 + 1] = marker_color
        out[r0 : r1 + 1, c0] = marker_color
        out[r0 : r1 + 1, c1] = marker_color
        out[mr, mc] = frame_color

    return out


# --------------- generator ---------------

GRID_MIN = 12
GRID_MAX = 30
RECT_COUNT_CAP = 5
PLACEMENT_TRIES = 200


def _sample_inner_size(rng: random.Random, limit: int, crowded: bool) -> int:
    upper = min(limit, 6 if crowded else 10)
    assert upper >= 1
    branch = rng.randrange(4)
    if branch == 0:
        return 1
    if branch == 1:
        return rng.randint(1, min(3, upper))
    return rng.randint(1, upper)


def _draw_frame(grid: np.ndarray, box: tuple[int, int, int, int], color: int) -> None:
    r0, c0, r1, c1 = box
    grid[r0, c0 : c1 + 1] = color
    grid[r1, c0 : c1 + 1] = color
    grid[r0 : r1 + 1, c0] = color
    grid[r0 : r1 + 1, c1] = color


def _gen_attempt(rng: random.Random) -> np.ndarray | None:
    h = rng.randint(GRID_MIN, GRID_MAX)
    w = rng.randint(GRID_MIN, GRID_MAX)
    grid = np.full((h, w), BG, dtype=int)

    frame_color = rng.choice(COLORS)
    marker_pool = [color for color in COLORS if color != frame_color]
    max_rects = min(RECT_COUNT_CAP, len(marker_pool), max(1, (h * w) // 55))
    rect_count = rng.randint(1, max_rects)
    marker_colors = rng.sample(marker_pool, rect_count)

    occupied = np.zeros((h, w), dtype=bool)

    for marker_color in marker_colors:
        placed = False
        crowded = rect_count >= 4
        for _ in range(PLACEMENT_TRIES):
            inner_h = _sample_inner_size(rng, h - 2, crowded)
            inner_w = _sample_inner_size(rng, w - 2, crowded)
            rect_h = inner_h + 2
            rect_w = inner_w + 2
            if rect_h > h or rect_w > w:
                continue

            r0 = rng.randint(0, h - rect_h)
            c0 = rng.randint(0, w - rect_w)
            r1 = r0 + rect_h - 1
            c1 = c0 + rect_w - 1

            er0 = max(0, r0 - 1)
            ec0 = max(0, c0 - 1)
            er1 = min(h - 1, r1 + 1)
            ec1 = min(w - 1, c1 + 1)
            if occupied[er0 : er1 + 1, ec0 : ec1 + 1].any():
                continue

            _draw_frame(grid, (r0, c0, r1, c1), frame_color)
            mr = rng.randint(r0 + 1, r1 - 1)
            mc = rng.randint(c0 + 1, c1 - 1)
            grid[mr, mc] = marker_color
            occupied[er0 : er1 + 1, ec0 : ec1 + 1] = True
            placed = True
            break

        if not placed:
            return None

    regions = _extract_regions(grid)
    if len(regions) != rect_count:
        return None
    if np.array_equal(main(grid), grid):
        return None
    return grid


def generate_input() -> np.ndarray:
    for _ in range(GENERATION_ATTEMPTS):
        grid = _gen_attempt(_RNG)
        if grid is not None:
            return grid
    raise RuntimeError("failed to generate a valid input")
