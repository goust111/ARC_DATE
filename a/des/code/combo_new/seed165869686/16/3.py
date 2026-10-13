import random
from typing import Iterable

import numpy as np


BLACK = 0
BLUE = 1
RED = 2
GREEN = 3

HORIZONTAL = "horizontal"
VERTICAL = "vertical"
DIAGONAL = "diagonal"

HSTRIPE = "horizontal_stripes"
VSTRIPE = "vertical_stripes"
CHECKER = "checkerboard"

REL_TO_TEXTURE = {
    HORIZONTAL: HSTRIPE,
    VERTICAL: VSTRIPE,
    DIAGONAL: CHECKER,
}

LEGEND_BLOCK_H = 5
LEGEND_BLOCK_W = 3
GEN_ATTEMPTS = 400


def _neighbors4(r: int, c: int) -> Iterable[tuple[int, int]]:
    yield r - 1, c
    yield r + 1, c
    yield r, c - 1
    yield r, c + 1


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
            comp: list[tuple[int, int]] = []
            while stack:
                cr, cc = stack.pop()
                comp.append((cr, cc))
                for nr, nc in _neighbors4(cr, cc):
                    if 0 <= nr < h and 0 <= nc < w and mask[nr, nc] and not seen[nr, nc]:
                        seen[nr, nc] = True
                        stack.append((nr, nc))
            comps.append(comp)
    return comps


def _classify_pair(points: list[tuple[int, int]]) -> str | None:
    if len(points) != 2:
        return None
    (r0, c0), (r1, c1) = points
    if r0 == r1 and c0 != c1:
        return HORIZONTAL
    if c0 == c1 and r0 != r1:
        return VERTICAL
    if abs(r0 - r1) == abs(c0 - c1) and r0 != r1:
        return DIAGONAL
    return None


def _texture_grid(texture: str, h: int, w: int) -> np.ndarray:
    rr, cc = np.indices((h, w))
    if texture == HSTRIPE:
        green_mask = (rr % 2) == 0
    elif texture == VSTRIPE:
        green_mask = (cc % 2) == 0
    elif texture == CHECKER:
        green_mask = ((rr + cc) % 2) == 0
    else:
        raise ValueError(f"unknown texture: {texture}")
    return np.where(green_mask, GREEN, BLACK)


def _texture_kind_from_demo(block: np.ndarray) -> str | None:
    if block.shape != (3, 3):
        return None
    if not np.all((block == BLACK) | (block == GREEN)):
        return None
    for texture in (HSTRIPE, VSTRIPE, CHECKER):
        if np.array_equal(block, _texture_grid(texture, 3, 3)):
            return texture
    return None


def _detect_legend_mapping(grid: np.ndarray) -> dict[str, str]:
    g = np.asarray(grid, dtype=int)
    h, w = g.shape
    found: dict[str, set[str]] = {}
    for r in range(h - LEGEND_BLOCK_H + 1):
        for c in range(w - LEGEND_BLOCK_W + 1):
            marker_zone = g[r : r + 2, c : c + 3]
            if np.count_nonzero(marker_zone == RED) != 2:
                continue
            if np.any((marker_zone != BLACK) & (marker_zone != RED)):
                continue

            red_points = [tuple(map(int, p)) for p in np.argwhere(marker_zone == RED)]
            relation = _classify_pair(red_points)
            if relation is None:
                continue

            texture = _texture_kind_from_demo(g[r + 2 : r + 5, c : c + 3])
            if texture is None:
                continue

            found.setdefault(relation, set()).add(texture)

    mapping = {relation: next(iter(textures)) for relation, textures in found.items() if len(textures) == 1}
    assert set(mapping) == {HORIZONTAL, VERTICAL, DIAGONAL}, "legend must define all three mappings"
    return mapping


def _find_rectangular_frames(grid: np.ndarray) -> list[tuple[int, int, int, int]]:
    frames: list[tuple[int, int, int, int]] = []
    for comp in _components(grid == BLUE):
        rows = [r for r, _ in comp]
        cols = [c for _, c in comp]
        r0, r1 = min(rows), max(rows)
        c0, c1 = min(cols), max(cols)
        if r1 - r0 < 2 or c1 - c0 < 2:
            continue
        border = set()
        for c in range(c0, c1 + 1):
            border.add((r0, c))
            border.add((r1, c))
        for r in range(r0, r1 + 1):
            border.add((r, c0))
            border.add((r, c1))
        if set(comp) == border:
            frames.append((r0, c0, r1, c1))
    frames.sort()
    assert frames, "expected at least one blue rectangular frame"
    return frames


def main(grid: np.ndarray) -> np.ndarray:
    """
    Fill each blue rectangular ring with the texture indicated by its two red markers.

    Input structure:
    - The background is black.
    - Blue pixels form several disjoint hollow rectangular frames whose interiors are otherwise black.
    - Each frame interior contains exactly two red marker pixels arranged horizontally, vertically, or on a 45-degree diagonal.
    - A corner legend contains three 3x3 green/black demo textures, each with two red pixels above it encoding one of the three marker arrangements.

    Rule:
    - Read the corner legend to infer a mapping from red-pair arrangement to texture type.
    - For each blue frame, classify the arrangement of its two interior red markers as horizontal, vertical, or diagonal.
    - Fill the whole frame interior with the corresponding texture, anchored at the interior's top-left corner.
    - Preserve the blue frame border, the legend, and the two red marker pixels inside each frame.

    Key insight:
    - The legend is a symbolic lookup table: the red-pair geometry does not directly say what to draw until it is matched to the adjacent demo texture, and that inferred mapping must then be reused inside every frame.
    """
    g = np.asarray(grid, dtype=int)
    assert g.ndim == 2, "expected a 2D grid"

    legend_mapping = _detect_legend_mapping(g)
    out = g.copy()

    for r0, c0, r1, c1 in _find_rectangular_frames(g):
        interior = g[r0 + 1 : r1, c0 + 1 : c1]
        red_points = [tuple(map(int, p)) for p in np.argwhere(interior == RED)]
        assert len(red_points) == 2, "each frame interior must contain exactly two red markers"
        relation = _classify_pair(red_points)
        assert relation in legend_mapping, "frame marker relation must be defined by the legend"

        texture = _texture_grid(legend_mapping[relation], interior.shape[0], interior.shape[1])
        out[r0 + 1 : r1, c0 + 1 : c1] = texture
        for rr, cc in red_points:
            out[r0 + 1 + rr, c0 + 1 + cc] = RED

    return out


def _reserve(mask: np.ndarray, r0: int, c0: int, r1: int, c1: int, pad: int = 1) -> None:
    h, w = mask.shape
    rr0 = max(0, r0 - pad)
    cc0 = max(0, c0 - pad)
    rr1 = min(h - 1, r1 + pad)
    cc1 = min(w - 1, c1 + pad)
    mask[rr0 : rr1 + 1, cc0 : cc1 + 1] = True


def _legend_layout(rng: random.Random) -> tuple[list[tuple[int, int]], int, int]:
    layout = rng.choice(("row", "column"))
    gap = rng.choice((1, 2))
    if layout == "row":
        positions = [(0, i * (LEGEND_BLOCK_W + gap)) for i in range(3)]
        total_h = LEGEND_BLOCK_H
        total_w = 3 * LEGEND_BLOCK_W + 2 * gap
    else:
        positions = [(i * (LEGEND_BLOCK_H + gap), 0) for i in range(3)]
        total_h = 3 * LEGEND_BLOCK_H + 2 * gap
        total_w = LEGEND_BLOCK_W
    return positions, total_h, total_w


def _legend_marker_positions(relation: str, rng: random.Random) -> list[tuple[int, int]]:
    choices = {
        HORIZONTAL: [
            [(0, 0), (0, 1)],
            [(0, 1), (0, 2)],
            [(1, 0), (1, 1)],
            [(1, 1), (1, 2)],
            [(0, 0), (0, 2)],
            [(1, 0), (1, 2)],
        ],
        VERTICAL: [[(0, c), (1, c)] for c in range(3)],
        DIAGONAL: [
            [(0, 0), (1, 1)],
            [(0, 1), (1, 0)],
            [(0, 1), (1, 2)],
            [(0, 2), (1, 1)],
        ],
    }
    return rng.choice(choices[relation])


def _paint_legend_demo(grid: np.ndarray, top: int, left: int, relation: str, rng: random.Random) -> None:
    for rr, cc in _legend_marker_positions(relation, rng):
        grid[top + rr, left + cc] = RED
    grid[top + 2 : top + 5, left : left + 3] = _texture_grid(REL_TO_TEXTURE[relation], 3, 3)


def _place_legend(grid: np.ndarray, reserved: np.ndarray, rng: random.Random) -> tuple[int, int, int, int] | None:
    h, w = grid.shape
    positions, total_h, total_w = _legend_layout(rng)
    margin_r = rng.randint(0, 1)
    margin_c = rng.randint(0, 1)
    corner = rng.choice(("tl", "tr", "bl", "br"))

    if total_h + margin_r > h or total_w + margin_c > w:
        return None

    if corner == "tl":
        top, left = margin_r, margin_c
    elif corner == "tr":
        top, left = margin_r, w - total_w - margin_c
    elif corner == "bl":
        top, left = h - total_h - margin_r, margin_c
    else:
        top, left = h - total_h - margin_r, w - total_w - margin_c

    relations = [HORIZONTAL, VERTICAL, DIAGONAL]
    rng.shuffle(relations)
    for relation, (dr, dc) in zip(relations, positions):
        _paint_legend_demo(grid, top + dr, left + dc, relation, rng)

    _reserve(reserved, top, left, top + total_h - 1, left + total_w - 1, pad=1)
    return top, left, top + total_h - 1, left + total_w - 1


def _frame_marker_positions(inner_h: int, inner_w: int, relation: str, rng: random.Random) -> list[tuple[int, int]]:
    if relation == HORIZONTAL:
        row = rng.randrange(inner_h)
        c0, c1 = sorted(rng.sample(range(inner_w), 2))
        return [(row, c0), (row, c1)]
    if relation == VERTICAL:
        col = rng.randrange(inner_w)
        r0, r1 = sorted(rng.sample(range(inner_h), 2))
        return [(r0, col), (r1, col)]

    delta = rng.randint(1, min(inner_h, inner_w) - 1)
    if rng.random() < 0.5:
        r0 = rng.randint(0, inner_h - delta - 1)
        c0 = rng.randint(0, inner_w - delta - 1)
        return [(r0, c0), (r0 + delta, c0 + delta)]

    r0 = rng.randint(0, inner_h - delta - 1)
    c0 = rng.randint(delta, inner_w - 1)
    return [(r0, c0), (r0 + delta, c0 - delta)]


def _draw_frame(grid: np.ndarray, top: int, left: int, inner_h: int, inner_w: int, relation: str, rng: random.Random) -> None:
    bottom = top + inner_h + 1
    right = left + inner_w + 1
    grid[top, left : right + 1] = BLUE
    grid[bottom, left : right + 1] = BLUE
    grid[top : bottom + 1, left] = BLUE
    grid[top : bottom + 1, right] = BLUE
    for rr, cc in _frame_marker_positions(inner_h, inner_w, relation, rng):
        grid[top + 1 + rr, left + 1 + cc] = RED


def _validate_generated(grid: np.ndarray, expected_frames: int) -> bool:
    try:
        mapping = _detect_legend_mapping(grid)
        if mapping != REL_TO_TEXTURE:
            return False

        frames = _find_rectangular_frames(grid)
        if len(frames) != expected_frames:
            return False

        out = main(grid)
        for r0, c0, r1, c1 in frames:
            interior = grid[r0 + 1 : r1, c0 + 1 : c1]
            red_points = [tuple(map(int, p)) for p in np.argwhere(interior == RED)]
            if len(red_points) != 2:
                return False
            relation = _classify_pair(red_points)
            if relation is None:
                return False

            expected = _texture_grid(REL_TO_TEXTURE[relation], interior.shape[0], interior.shape[1])
            for rr, cc in red_points:
                expected[rr, cc] = RED
            if not np.array_equal(out[r0 + 1 : r1, c0 + 1 : c1], expected):
                return False
    except AssertionError:
        return False

    return True


def _gen_attempt(rng: random.Random) -> np.ndarray | None:
    h = rng.randint(22, 30)
    w = rng.randint(22, 30)
    grid = np.zeros((h, w), dtype=int)
    reserved = np.zeros((h, w), dtype=bool)

    if _place_legend(grid, reserved, rng) is None:
        return None

    n_frames = 3
    if h >= 26 and w >= 26 and rng.random() < 0.6:
        n_frames = 4

    relations = [HORIZONTAL, VERTICAL, DIAGONAL]
    while len(relations) < n_frames:
        relations.append(rng.choice((HORIZONTAL, VERTICAL, DIAGONAL)))
    rng.shuffle(relations)

    max_inner = 6 if n_frames == 4 else 7
    specs = []
    for relation in relations:
        inner_h = rng.randint(3, max_inner)
        inner_w = rng.randint(3, max_inner)
        specs.append((relation, inner_h, inner_w))
    specs.sort(key=lambda item: (item[1] + 2) * (item[2] + 2), reverse=True)

    for relation, inner_h, inner_w in specs:
        box_h = inner_h + 2
        box_w = inner_w + 2
        max_top = h - box_h - 1
        max_left = w - box_w - 1
        if max_top < 1 or max_left < 1:
            return None

        placed = False
        for _ in range(250):
            top = rng.randint(1, max_top)
            left = rng.randint(1, max_left)
            if reserved[top - 1 : top + box_h + 1, left - 1 : left + box_w + 1].any():
                continue
            _draw_frame(grid, top, left, inner_h, inner_w, relation, rng)
            _reserve(reserved, top, left, top + box_h - 1, left + box_w - 1, pad=1)
            placed = True
            break
        if not placed:
            return None

    if not _validate_generated(grid, n_frames):
        return None
    return grid


def generate_input() -> np.ndarray:
    rng = random.Random()
    for _ in range(GEN_ATTEMPTS):
        try:
            grid = _gen_attempt(rng)
        except Exception:
            continue
        if grid is not None:
            return grid
    raise RuntimeError("failed to generate a valid ARC input within retry budget")
