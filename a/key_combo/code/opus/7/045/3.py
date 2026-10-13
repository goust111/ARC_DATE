from __future__ import annotations

import random

import numpy as np


BG = 0
BLUE = 1
RED = 2
GREEN = 3
YELLOW = 4

SOURCE_COLORS = (GREEN, YELLOW)
MARKER_COLORS = (BLUE, RED)
DIR4 = ((1, 0), (-1, 0), (0, 1), (0, -1))

GRID_MIN = 16
GRID_MAX = 22
RECT_COUNT_CHOICES = (2, 3)
SPRITE_SIZE = 3
SPRITE_AREA_CHOICES = (4, 5)
MAX_GENERATION_ATTEMPTS = 400


def _components(mask: np.ndarray) -> list[list[tuple[int, int]]]:
    h, w = mask.shape
    seen = np.zeros((h, w), dtype=bool)
    components: list[list[tuple[int, int]]] = []
    for r in range(h):
        for c in range(w):
            if not mask[r, c] or seen[r, c]:
                continue
            stack = [(r, c)]
            seen[r, c] = True
            coords: list[tuple[int, int]] = []
            while stack:
                cr, cc = stack.pop()
                coords.append((cr, cc))
                for dr, dc in DIR4:
                    nr = cr + dr
                    nc = cc + dc
                    if 0 <= nr < h and 0 <= nc < w and mask[nr, nc] and not seen[nr, nc]:
                        seen[nr, nc] = True
                        stack.append((nr, nc))
            components.append(coords)
    return components


def _bbox(coords: list[tuple[int, int]]) -> tuple[int, int, int, int]:
    rows = [r for r, _ in coords]
    cols = [c for _, c in coords]
    return min(rows), min(cols), max(rows), max(cols)


def _paste(dst: np.ndarray, patch: np.ndarray, top: int, left: int) -> None:
    mask = patch != BG
    view = dst[top : top + patch.shape[0], left : left + patch.shape[1]]
    view[mask] = patch[mask]


def _extract_template(grid: np.ndarray) -> tuple[np.ndarray, tuple[int, int, int, int]]:
    source_mask = np.isin(grid, SOURCE_COLORS)
    components = _components(source_mask)
    assert len(components) == 1, "expected one multicolor source pattern"
    source_bbox = _bbox(components[0])
    r0, c0, r1, c1 = source_bbox
    template = grid[r0 : r1 + 1, c0 : c1 + 1].copy()
    assert set(np.unique(template)) <= {BG, GREEN, YELLOW}
    assert GREEN in template and YELLOW in template, "source must use both colors"
    assert not np.array_equal(template, template[:, ::-1]), "source must change under horizontal mirror"
    return template, source_bbox


def _find_rectangles(grid: np.ndarray, template_shape: tuple[int, int]) -> list[tuple[int, int, int, int, int]]:
    h, w = template_shape
    dr = h + 1
    dc = w + 1
    marker_coords = {tuple(map(int, rc)) for rc in np.argwhere(np.isin(grid, MARKER_COLORS))}
    rectangles: list[tuple[int, int, int, int, int]] = []

    for r0, c0 in sorted(marker_coords):
        r1 = r0 + dr
        c1 = c0 + dc
        if r1 >= grid.shape[0] or c1 >= grid.shape[1]:
            continue
        corners = [(r0, c0), (r0, c1), (r1, c0), (r1, c1)]
        if not all((r, c) in marker_coords for r, c in corners):
            continue

        tl = int(grid[r0, c0])
        tr = int(grid[r0, c1])
        bl = int(grid[r1, c0])
        br = int(grid[r1, c1])
        if tl != br or tr != bl or tl == tr:
            continue
        if np.any(grid[r0, c0 + 1 : c1] != BG):
            continue
        if np.any(grid[r1, c0 + 1 : c1] != BG):
            continue
        if np.any(grid[r0 + 1 : r1, c0] != BG):
            continue
        if np.any(grid[r0 + 1 : r1, c1] != BG):
            continue
        if np.any(grid[r0 + 1 : r1, c0 + 1 : c1] != BG):
            continue
        rectangles.append((r0, c0, r1, c1, tl))

    used_markers = set()
    for r0, c0, r1, c1, _ in rectangles:
        used_markers.update({(r0, c0), (r0, c1), (r1, c0), (r1, c1)})
    assert len(rectangles) >= 2 and used_markers == marker_coords, "all marker pixels must belong to corner rectangles"
    return rectangles


def main(grid: np.ndarray) -> np.ndarray:
    """Stamp the corner source pattern into every marked rectangle, mirroring it when the top-left marker is red.

    Input structure:
    - A black background contains one small 4-connected source pattern near the upper left, using green and yellow.
    - Elsewhere, each target is an empty rectangle indicated only by four corner pixels in blue and red.
    - Within each rectangle, opposite corners share a color, and all interiors have the same size as the source pattern's bbox.

    Rule:
    - Crop the source pattern to its tight bbox.
    - Erase the source from its original location.
    - For every corner-marked rectangle, stamp the source into the interior unchanged when the top-left corner is blue, or stamped after a horizontal mirror when the top-left corner is red.
    - Keep all corner markers in place.

    Key insight:
    - The corner markers encode both the destination window and a one-bit transform instruction: the top-left corner color decides whether the source must be mirrored.
    """
    g = np.asarray(grid, dtype=int)
    template, source_bbox = _extract_template(g)
    rectangles = _find_rectangles(g, template.shape)

    out = g.copy()
    out[np.isin(out, SOURCE_COLORS)] = BG

    for r0, c0, r1, c1, top_left_color in rectangles:
        patch = template if top_left_color == BLUE else template[:, ::-1]
        _paste(out, patch, r0 + 1, c0 + 1)

    sr0, sc0, sr1, sc1 = source_bbox
    assert np.all(out[sr0 : sr1 + 1, sc0 : sc1 + 1][template == BG] == BG)
    return out


def _generate_mask() -> np.ndarray:
    for _ in range(200):
        area = random.choice(SPRITE_AREA_CHOICES)
        cells = {(random.randrange(SPRITE_SIZE), random.randrange(SPRITE_SIZE))}
        while len(cells) < area:
            r, c = random.choice(tuple(cells))
            neighbors = []
            for dr, dc in DIR4:
                nr = r + dr
                nc = c + dc
                if 0 <= nr < SPRITE_SIZE and 0 <= nc < SPRITE_SIZE and (nr, nc) not in cells:
                    neighbors.append((nr, nc))
            if not neighbors:
                continue
            cells.add(random.choice(neighbors))

        rows = {r for r, _ in cells}
        cols = {c for _, c in cells}
        if rows != {0, 1, 2} or cols != {0, 1, 2}:
            continue

        mask = np.zeros((SPRITE_SIZE, SPRITE_SIZE), dtype=bool)
        for r, c in cells:
            mask[r, c] = True
        if np.array_equal(mask, mask[:, ::-1]):
            continue
        return mask
    raise RuntimeError("failed to generate an asymmetric 3x3 source mask")


def _generate_sprite() -> np.ndarray:
    for _ in range(200):
        mask = _generate_mask()
        sprite = np.zeros((SPRITE_SIZE, SPRITE_SIZE), dtype=int)
        coords = list(zip(*np.where(mask)))
        yellow_count = random.randint(1, len(coords) - 1)
        yellow_cells = set(random.sample(coords, yellow_count))
        for r, c in coords:
            sprite[r, c] = YELLOW if (r, c) in yellow_cells else GREEN
        if GREEN in sprite and YELLOW in sprite and not np.array_equal(sprite, sprite[:, ::-1]):
            return sprite
    raise RuntimeError("failed to color the source sprite")


def _boxes_too_close(a: tuple[int, int, int, int], b: tuple[int, int, int, int], pad: int = 1) -> bool:
    ar0, ac0, ar1, ac1 = a
    br0, bc0, br1, bc1 = b
    return not (ar1 + pad < br0 or br1 + pad < ar0 or ac1 + pad < bc0 or bc1 + pad < ac0)


def _sample_rectangles(
    grid_shape: tuple[int, int], source_bbox: tuple[int, int, int, int], count: int
) -> list[tuple[int, int, int, int]]:
    h, w = grid_shape
    outer_h = SPRITE_SIZE + 2
    outer_w = SPRITE_SIZE + 2
    candidates = [
        (r0, c0, r0 + outer_h - 1, c0 + outer_w - 1)
        for r0 in range(h - outer_h + 1)
        for c0 in range(w - outer_w + 1)
        if not _boxes_too_close((r0, c0, r0 + outer_h - 1, c0 + outer_w - 1), source_bbox)
    ]
    random.shuffle(candidates)

    rectangles: list[tuple[int, int, int, int]] = []
    for candidate in candidates:
        if any(_boxes_too_close(candidate, placed) for placed in rectangles):
            continue
        rectangles.append(candidate)
        if len(rectangles) == count:
            return rectangles
    raise RuntimeError("failed to place all target rectangles")


def _draw_marker_rectangle(grid: np.ndarray, rect: tuple[int, int, int, int], top_left_color: int) -> None:
    r0, c0, r1, c1 = rect
    other = RED if top_left_color == BLUE else BLUE
    grid[r0, c0] = top_left_color
    grid[r0, c1] = other
    grid[r1, c0] = other
    grid[r1, c1] = top_left_color


def _expected_output(
    grid: np.ndarray, sprite: np.ndarray, rectangles: list[tuple[int, int, int, int]], tl_colors: list[int]
) -> np.ndarray:
    out = grid.copy()
    out[np.isin(out, SOURCE_COLORS)] = BG
    for rect, top_left_color in zip(rectangles, tl_colors):
        patch = sprite if top_left_color == BLUE else sprite[:, ::-1]
        _paste(out, patch, rect[0] + 1, rect[1] + 1)
    return out


def generate_input() -> np.ndarray:
    for _ in range(MAX_GENERATION_ATTEMPTS):
        h = random.randint(GRID_MIN, GRID_MAX)
        w = random.randint(GRID_MIN, GRID_MAX)
        sprite = _generate_sprite()

        grid = np.zeros((h, w), dtype=int)
        src_r = random.randint(1, min(4, h - SPRITE_SIZE - 1))
        src_c = random.randint(1, min(4, w - SPRITE_SIZE - 1))
        _paste(grid, sprite, src_r, src_c)
        source_bbox = (src_r, src_c, src_r + SPRITE_SIZE - 1, src_c + SPRITE_SIZE - 1)

        rect_count = random.choice(RECT_COUNT_CHOICES)
        try:
            rectangles = _sample_rectangles((h, w), source_bbox, rect_count)
        except RuntimeError:
            continue

        tl_colors = [BLUE, RED]
        while len(tl_colors) < rect_count:
            tl_colors.append(random.choice(MARKER_COLORS))
        random.shuffle(tl_colors)

        for rect, top_left_color in zip(rectangles, tl_colors):
            _draw_marker_rectangle(grid, rect, top_left_color)

        expected = _expected_output(grid, sprite, rectangles, tl_colors)
        try:
            actual = main(grid)
        except AssertionError:
            continue
        if np.array_equal(actual, expected):
            return grid

    raise RuntimeError("failed to generate a valid input for puzzle 3")
