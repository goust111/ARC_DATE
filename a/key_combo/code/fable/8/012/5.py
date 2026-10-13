import random

import numpy as np


BLACK = 0
PALETTE = tuple(range(1, 10))
MAX_GEN_ATTEMPTS = 400


def _find_markers(grid: np.ndarray) -> tuple[int, tuple[int, int], tuple[int, int]]:
    candidates = []
    for color in np.unique(grid):
        color = int(color)
        if color == BLACK:
            continue
        coords = np.argwhere(grid == color)
        if len(coords) != 2:
            continue
        (r0, c0), (r1, c1) = coords
        if int(r0) == int(r1) or int(c0) == int(c1):
            continue
        top = min(int(r0), int(r1))
        bottom = max(int(r0), int(r1))
        left = min(int(c0), int(c1))
        right = max(int(c0), int(c1))
        if bottom - top < 2 or right - left < 2:
            continue
        inner = grid[top + 1 : bottom, left + 1 : right]
        if np.any(inner == BLACK):
            candidates.append((color, (int(r0), int(c0)), (int(r1), int(c1))))
    assert len(candidates) == 1, "expected one marker-color pair"
    return candidates[0]


def _period_consistent(grid: np.ndarray, visible_mask: np.ndarray, pr: int, pc: int) -> bool:
    seen: dict[tuple[int, int], int] = {}
    for r, c in np.argwhere(visible_mask):
        key = (int(r) % pr, int(c) % pc)
        color = int(grid[int(r), int(c)])
        prev = seen.get(key)
        if prev is None:
            seen[key] = color
        elif prev != color:
            return False
    return True


def _infer_period(grid: np.ndarray, marker_color: int) -> tuple[int, int]:
    h, w = grid.shape
    visible_mask = (grid != BLACK) & (grid != marker_color)
    candidates = sorted(((pr, pc) for pr in range(1, h + 1) for pc in range(1, w + 1)), key=lambda x: (x[0] * x[1], x[0], x[1]))
    for pr, pc in candidates:
        if _period_consistent(grid, visible_mask, pr, pc):
            return pr, pc
    raise AssertionError("no consistent period found")


def _period_tile(grid: np.ndarray, marker_color: int, pr: int, pc: int) -> np.ndarray:
    tile = np.full((pr, pc), -1, dtype=int)
    visible_mask = (grid != BLACK) & (grid != marker_color)
    for r, c in np.argwhere(visible_mask):
        rr = int(r) % pr
        cc = int(c) % pc
        color = int(grid[int(r), int(c)])
        if tile[rr, cc] == -1:
            tile[rr, cc] = color
        else:
            assert tile[rr, cc] == color
    assert np.all(tile != -1), "every period class must be visible somewhere"
    return tile


def _minimal_period_of_array(arr: np.ndarray) -> tuple[int, int]:
    h, w = arr.shape
    mask = np.ones((h, w), dtype=bool)
    for pr in range(1, h + 1):
        for pc in range(1, w + 1):
            if _period_consistent(arr, mask, pr, pc):
                return pr, pc
    raise AssertionError("array must have a period")


def _build_pattern(tile: np.ndarray, h: int, w: int) -> np.ndarray:
    pr, pc = tile.shape
    rows = np.arange(h)[:, None] % pr
    cols = np.arange(w)[None, :] % pc
    return tile[rows, cols]


def _sample_minimal_tile() -> np.ndarray:
    for _ in range(MAX_GEN_ATTEMPTS):
        pr = random.randint(2, 5)
        pc = random.randint(2, 5)
        color_count = random.randint(3, min(5, pr * pc))
        palette = random.sample(PALETTE, color_count)
        tile = np.array([[random.choice(palette) for _ in range(pc)] for _ in range(pr)], dtype=int)
        if len(set(map(int, tile.flat))) < 3:
            continue
        if _minimal_period_of_array(tile) == (pr, pc):
            return tile
    raise RuntimeError("failed to sample a minimal tile")


def _visibility_classes(grid: np.ndarray, marker_color: int, pr: int, pc: int) -> set[tuple[int, int]]:
    visible = (grid != BLACK) & (grid != marker_color)
    return {(int(r) % pr, int(c) % pc) for r, c in np.argwhere(visible)}


def main(grid: np.ndarray) -> np.ndarray:
    """
    Crop the rectangle interior defined by two markers, repairing black gaps from the global periodic pattern first.

    Input structure:
    - The whole grid is covered by a colored 2D periodic pattern with some black missing pixels.
    - Exactly two equal-colored marker pixels lie on different rows and columns, and that marker color does not belong
      to the periodic pattern.
    - The two markers are opposite corners of a rectangle whose strict interior contains black gaps.

    Rule:
    - Use all non-black, non-marker pixels in the full grid to infer the smallest repeating period rectangle.
    - Take the strict interior of the marker-defined rectangle as the output canvas.
    - Inside that window, keep visible pixels unchanged and fill each black cell from the inferred period by its
      absolute row and column residues.

    Key insight:
    - The crop alone does not contain enough evidence; the pattern must be inferred from the whole image before the
      marker window is extracted and repaired.
    """
    g = np.asarray(grid, dtype=int)
    assert g.ndim == 2, "expected a 2D grid"

    marker_color, a, b = _find_markers(g)
    pr, pc = _infer_period(g, marker_color)
    tile = _period_tile(g, marker_color, pr, pc)

    top = min(a[0], b[0])
    bottom = max(a[0], b[0])
    left = min(a[1], b[1])
    right = max(a[1], b[1])
    out = g[top + 1 : bottom, left + 1 : right].copy()

    for rr in range(out.shape[0]):
        for cc in range(out.shape[1]):
            if out[rr, cc] == BLACK:
                abs_r = top + 1 + rr
                abs_c = left + 1 + cc
                out[rr, cc] = tile[abs_r % pr, abs_c % pc]
    return out


def generate_input() -> np.ndarray:
    for _ in range(MAX_GEN_ATTEMPTS):
        tile = _sample_minimal_tile()
        pr, pc = tile.shape

        rep_r = random.randint(max(3, (14 + pr - 1) // pr), 30 // pr)
        rep_c = random.randint(max(3, (14 + pc - 1) // pc), 30 // pc)
        h = pr * rep_r
        w = pc * rep_c
        if h < 8 or w < 8:
            continue

        inner_h = random.randint(2, min(h - 4, max(2, pr * 3)))
        inner_w = random.randint(2, min(w - 4, max(2, pc * 3 + 1)))
        top = random.randint(1, h - inner_h - 3)
        left = random.randint(1, w - inner_w - 3)
        bottom = top + inner_h + 1
        right = left + inner_w + 1

        grid = _build_pattern(tile, h, w)
        inner_area = inner_h * inner_w
        inside_cells = [(r, c) for r in range(top + 1, bottom) for c in range(left + 1, right)]
        outside_cells = [
            (r, c)
            for r in range(h)
            for c in range(w)
            if not (top <= r <= bottom and left <= c <= right)
        ]

        inside_missing = random.randint(1, max(1, min(inner_area - 1, inner_area // 3 + 1)))
        outside_cap = min(len(outside_cells), max(0, (h * w) // 10))
        outside_missing = random.randint(0, outside_cap)
        missing_cells = set(random.sample(inside_cells, inside_missing))
        if outside_missing:
            missing_cells.update(random.sample(outside_cells, outside_missing))

        for r, c in missing_cells:
            grid[r, c] = BLACK

        marker_choices = [color for color in PALETTE if color not in set(map(int, tile.flat))]
        if not marker_choices:
            continue
        marker_color = random.choice(marker_choices)
        grid[top, left] = marker_color
        grid[bottom, right] = marker_color

        if _visibility_classes(grid, marker_color, pr, pc) != {(r, c) for r in range(pr) for c in range(pc)}:
            continue
        if _infer_period(grid, marker_color) != (pr, pc):
            continue

        return grid

    raise RuntimeError("failed to generate a valid input")
