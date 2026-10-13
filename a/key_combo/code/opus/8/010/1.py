from __future__ import annotations

import random

import numpy as np


BG = 0
GRAY = 5
MARKER_COLORS = [1, 2, 3, 4, 6, 7, 8, 9]
GEN_RETRIES = 400


def _pair_orientation(coords: np.ndarray) -> str:
    pts = [tuple(map(int, rc)) for rc in coords]
    assert len(pts) == 2, "each marker color must appear exactly twice"
    pts.sort()
    (r0, c0), (r1, c1) = pts
    assert r0 != r1 and c0 != c1, "marker pair must define a non-degenerate diagonal rectangle"
    return "nwse" if c0 < c1 else "nesw"


def _rect_from_pair(coords: np.ndarray) -> tuple[int, int, int, int]:
    rows = coords[:, 0].astype(int)
    cols = coords[:, 1].astype(int)
    return int(rows.min()), int(cols.min()), int(rows.max()), int(cols.max())


def _expected_corners(bbox: tuple[int, int, int, int], orientation: str) -> set[tuple[int, int]]:
    r0, c0, r1, c1 = bbox
    if orientation == "nwse":
        return {(r0, c0), (r1, c1)}
    return {(r0, c1), (r1, c0)}


def _mirror_coords(
    gray_coords: np.ndarray,
    height: int,
    width: int,
    orientation: str,
) -> list[tuple[int, int]]:
    mirrored: list[tuple[int, int]] = []
    for rr, cc in gray_coords:
        r = int(rr)
        c = int(cc)
        if orientation == "nwse":
            mirrored.append((r, width - 1 - c))
        else:
            mirrored.append((height - 1 - r, c))
    return mirrored


def main(grid: np.ndarray) -> np.ndarray:
    """Reflect each rectangle's gray motif across the axis encoded by its marker diagonal and paint the reflection in the marker color.

    Input structure:
    - Black background contains 2-3 non-gray marker colors, each appearing as exactly two isolated pixels.
    - Each same-color pair occupies opposite corners of an axis-aligned rectangle, and the rectangles do not overlap.
    - Inside each rectangle, a sparse gray motif lies away from the rectangle boundary.

    Rule:
    - For each non-black, non-gray color, take its two pixels as the diagonal corners of one rectangle.
    - If the upper marker is also the left marker (NW-SE diagonal), mirror all gray cells in that rectangle left-right; otherwise (NE-SW diagonal), mirror them top-bottom.
    - Draw the mirrored motif back into the same rectangle using the marker color, while keeping the original gray cells and the marker pixels.

    Key insight:
    - The same pair of colored corner markers simultaneously defines the working rectangle and chooses which reflection axis to apply to the gray motif inside it.
    """
    g = np.asarray(grid, dtype=int)
    assert g.ndim == 2

    out = g.copy()
    colors = sorted(int(color) for color in np.unique(g) if int(color) not in (BG, GRAY))
    assert 2 <= len(colors) <= 3, "expected 2-3 marker colors"

    for color in colors:
        coords = np.argwhere(g == color)
        assert coords.shape == (2, 2), "each marker color must occur exactly twice"

        bbox = _rect_from_pair(coords)
        orientation = _pair_orientation(coords)
        assert {tuple(map(int, rc)) for rc in coords} == _expected_corners(bbox, orientation), (
            "markers must occupy the rectangle's diagonal corners"
        )

        r0, c0, r1, c1 = bbox
        patch = g[r0 : r1 + 1, c0 : c1 + 1]
        gray_coords = np.argwhere(patch == GRAY)
        assert gray_coords.size > 0, "each rectangle must contain a gray motif"

        height, width = patch.shape
        marker_cells = {(rr - r0, cc - c0) for rr, cc in map(tuple, coords.tolist())}
        for mr, mc in _mirror_coords(gray_coords, height, width, orientation):
            if (mr, mc) in marker_cells:
                continue
            if int(g[r0 + mr, c0 + mc]) not in (BG, GRAY):
                continue
            out[r0 + mr, c0 + mc] = color

    return out


def _bbox_overlaps(a: tuple[int, int, int, int], b: tuple[int, int, int, int], pad: int = 1) -> bool:
    ar0, ac0, ar1, ac1 = a
    br0, bc0, br1, bc1 = b
    return not (
        ar1 + pad < br0
        or br1 + pad < ar0
        or ac1 + pad < bc0
        or bc1 + pad < ac0
    )


def _candidate_cells(height: int, width: int) -> list[tuple[int, int]]:
    cells = []
    blocked = {
        (1, 1),
        (1, width - 2),
        (height - 2, 1),
        (height - 2, width - 2),
    }
    for r in range(1, height - 1):
        for c in range(1, width - 1):
            if (r, c) not in blocked:
                cells.append((r, c))
    return cells


def _neighbors4(cell: tuple[int, int], allowed: set[tuple[int, int]]) -> list[tuple[int, int]]:
    r, c = cell
    out = []
    for dr, dc in ((-1, 0), (1, 0), (0, -1), (0, 1)):
        nxt = (r + dr, c + dc)
        if nxt in allowed:
            out.append(nxt)
    return out


def _is_asymmetric(coords: set[tuple[int, int]], height: int, width: int) -> bool:
    vertical = {(r, width - 1 - c) for r, c in coords}
    horizontal = {(height - 1 - r, c) for r, c in coords}
    return vertical != coords and horizontal != coords


def _random_pattern(height: int, width: int) -> set[tuple[int, int]] | None:
    candidates = _candidate_cells(height, width)
    allowed = set(candidates)
    if len(candidates) < 6:
        return None

    for _ in range(120):
        target = random.randint(3, 6)
        if random.random() < 0.55:
            start = random.choice(candidates)
            cells = {start}
            frontier = set(_neighbors4(start, allowed) or [])
            while len(cells) < target and frontier:
                cell = random.choice(tuple(frontier))
                frontier.remove(cell)
                cells.add(cell)
                for nxt in _neighbors4(cell, allowed):
                    if nxt not in cells:
                        frontier.add(nxt)
            if len(cells) != target:
                continue
        else:
            cells = set(random.sample(candidates, target))

        if _is_asymmetric(cells, height, width):
            return cells
    return None


def _paint_rectangle(
    grid: np.ndarray,
    bbox: tuple[int, int, int, int],
    color: int,
    orientation: str,
    pattern: set[tuple[int, int]],
) -> None:
    r0, c0, r1, c1 = bbox
    if orientation == "nwse":
        markers = [(r0, c0), (r1, c1)]
    else:
        markers = [(r0, c1), (r1, c0)]

    for r, c in markers:
        grid[r, c] = color
    for dr, dc in pattern:
        grid[r0 + dr, c0 + dc] = GRAY


def _gen_attempt() -> np.ndarray | None:
    height = random.randint(16, 24)
    width = random.randint(16, 24)
    grid = np.zeros((height, width), dtype=int)

    pair_count = random.randint(2, 3)
    colors = random.sample(MARKER_COLORS, pair_count)

    orientations = ["nwse", "nesw"]
    if pair_count == 3:
        orientations.append(random.choice(["nwse", "nesw"]))
    random.shuffle(orientations)

    bboxes: list[tuple[int, int, int, int]] = []
    for color, orientation in zip(colors, orientations):
        placed = False
        for _ in range(120):
            rect_h = random.randint(6, 9)
            rect_w = random.randint(6, 9)
            if rect_h >= height or rect_w >= width:
                continue
            r0 = random.randint(0, height - rect_h)
            c0 = random.randint(0, width - rect_w)
            bbox = (r0, c0, r0 + rect_h - 1, c0 + rect_w - 1)
            if any(_bbox_overlaps(bbox, other, pad=1) for other in bboxes):
                continue

            pattern = _random_pattern(rect_h, rect_w)
            if pattern is None:
                continue

            _paint_rectangle(grid, bbox, color, orientation, pattern)
            bboxes.append(bbox)
            placed = True
            break
        if not placed:
            return None

    try:
        out = main(grid)
    except AssertionError:
        return None

    if np.array_equal(out, grid):
        return None
    return grid


def generate_input() -> np.ndarray:
    for _ in range(GEN_RETRIES):
        grid = _gen_attempt()
        if grid is not None:
            return grid
    raise RuntimeError("failed to generate valid input")
