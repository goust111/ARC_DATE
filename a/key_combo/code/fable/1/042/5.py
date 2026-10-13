from __future__ import annotations

import random
from collections import deque

import numpy as np


class Color:
    BLACK = 0
    BLUE = 1
    RED = 2
    GREEN = 3
    YELLOW = 4
    GREY = 5
    PINK = 6
    ORANGE = 7
    TEAL = 8
    MAROON = 9


DIR4 = [(-1, 0), (1, 0), (0, -1), (0, 1)]
PALETTE = [
    Color.BLUE,
    Color.RED,
    Color.GREEN,
    Color.YELLOW,
    Color.PINK,
    Color.ORANGE,
    Color.TEAL,
    Color.MAROON,
]

GEN_RETRY_LIMIT = 400
CAVITY_H_RANGE = (4, 8)
CAVITY_W_RANGE = (5, 9)
BLOCK_SIZE_CHOICES = (2, 3)
OBJECT_COUNT_RANGE = (2, 5)
OBJECT_SIZE_WEIGHTS = [(1, 3), (2, 4), (3, 3), (4, 2)]


def bbox(mask: np.ndarray) -> tuple[int, int, int, int] | None:
    coords = np.argwhere(mask)
    if coords.size == 0:
        return None
    r0, c0 = coords.min(axis=0)
    r1, c1 = coords.max(axis=0)
    return int(r0), int(c0), int(r1), int(c1)


def _downsample_bool(mask: np.ndarray, mini_h: int, mini_w: int) -> np.ndarray:
    out = np.zeros((mini_h, mini_w), dtype=bool)
    H, W = mask.shape
    block_h = H // mini_h
    block_w = W // mini_w
    for r in range(mini_h):
        rs = slice(r * block_h, (r + 1) * block_h)
        for c in range(mini_w):
            cs = slice(c * block_w, (c + 1) * block_w)
            out[r, c] = bool(np.any(mask[rs, cs]))
    return out


def _downsample_blocks(grid: np.ndarray, mini_h: int, mini_w: int) -> np.ndarray:
    H, W = grid.shape
    assert H % mini_h == 0 and W % mini_w == 0, "grid must split evenly into coarse blocks"
    block_h = H // mini_h
    block_w = W // mini_w
    out = np.full((mini_h, mini_w), Color.BLACK, dtype=int)
    for r in range(mini_h):
        rs = slice(r * block_h, (r + 1) * block_h)
        for c in range(mini_w):
            cs = slice(c * block_w, (c + 1) * block_w)
            block = grid[rs, cs]
            colors = [int(x) for x in np.unique(block) if int(x) not in {Color.BLACK, Color.GREY}]
            assert len(colors) <= 1, "each block may contain at most one object color"
            if colors:
                out[r, c] = colors[0]
            elif np.any(block == Color.GREY):
                out[r, c] = Color.GREY
    return out


def _frame_border_mask(shape: tuple[int, int], box: tuple[int, int, int, int]) -> np.ndarray:
    r0, c0, r1, c1 = box
    mask = np.zeros(shape, dtype=bool)
    mask[r0, c0 : c1 + 1] = True
    mask[r1, c0 : c1 + 1] = True
    mask[r0 : r1 + 1, c0] = True
    mask[r0 : r1 + 1, c1] = True
    return mask


def _forms_hollow_rect(mask: np.ndarray) -> bool:
    box = bbox(mask)
    if box is None:
        return False
    r0, c0, r1, c1 = box
    if r1 - r0 < 2 or c1 - c0 < 2:
        return False
    expected = _frame_border_mask(mask.shape, box)
    return bool(np.array_equal(mask, expected))


def _parse_frame(grid: np.ndarray) -> tuple[int, int, int, int]:
    grey_mask = grid == Color.GREY
    frame_box = bbox(grey_mask)
    assert frame_box is not None, "expected one grey frame"
    r0, c0, r1, c1 = frame_box
    assert r1 - r0 >= 2 and c1 - c0 >= 2, "frame must enclose a non-empty cavity"
    expected_frame = _frame_border_mask(grid.shape, frame_box)
    assert np.array_equal(grey_mask, expected_frame), "grey pixels must form exactly one hollow rectangle"
    cavity = grid[r0 + 1 : r1, c0 + 1 : c1]
    assert cavity.size > 0 and np.all(cavity == Color.BLACK), "frame cavity must start empty"
    return frame_box


def main(grid: np.ndarray) -> np.ndarray:
    """
    Compress the whole scene into a coarse self-miniature and write it into the grey frame's cavity.

    Input structure:
    - The background is black.
    - Exactly one grey hollow rectangular frame is present, and its cavity is entirely black.
    - Separate colored solid objects are scattered outside the cavity.
    - The full grid height and width are integer multiples of the cavity height and width.

    Rule:
    - Let the cavity size be h x w and split the entire grid into h x w equal blocks.
    - If a block contains colored object pixels, color the corresponding cavity cell with that color.
    - Otherwise, if the block contains grey frame pixels, color the cavity cell grey; otherwise leave it black.
    - Keep everything outside the cavity unchanged and replace the cavity with this h x w miniature.

    Key insight:
    - The frame is a viewfinder: its cavity size defines the downsampling resolution, and the miniature must
      include the frame itself, so grey cells appear inside the inserted map as the frame's own reflection.
    """
    g = np.asarray(grid, dtype=int)
    assert g.ndim == 2, "expected a 2D grid"

    r0, c0, r1, c1 = _parse_frame(g)
    cavity_h = r1 - r0 - 1
    cavity_w = c1 - c0 - 1
    miniature = _downsample_blocks(g, cavity_h, cavity_w)

    out = g.copy()
    out[r0 + 1 : r1, c0 + 1 : c1] = miniature
    return out


def _dilate4(mask: np.ndarray) -> np.ndarray:
    out = mask.copy()
    H, W = mask.shape
    for r, c in np.argwhere(mask):
        rr = int(r)
        cc = int(c)
        for dr, dc in DIR4:
            nr = rr + dr
            nc = cc + dc
            if 0 <= nr < H and 0 <= nc < W:
                out[nr, nc] = True
    return out


def _normalize(cells: set[tuple[int, int]]) -> tuple[tuple[int, int], ...]:
    min_r = min(r for r, _ in cells)
    min_c = min(c for _, c in cells)
    return tuple(sorted((r - min_r, c - min_c) for r, c in cells))


def _random_polyomino(size: int) -> tuple[tuple[int, int], ...] | None:
    cells = {(0, 0)}
    while len(cells) < size:
        frontier: list[tuple[tuple[int, int], int]] = []
        seen = set()
        for r, c in cells:
            for dr, dc in DIR4:
                nr = r + dr
                nc = c + dc
                if (nr, nc) in cells or (nr, nc) in seen:
                    continue
                trial = cells | {(nr, nc)}
                rows = [x for x, _ in trial]
                cols = [x for _, x in trial]
                if max(rows) - min(rows) + 1 > 3 or max(cols) - min(cols) + 1 > 3:
                    continue
                contact = sum((nr + rr, nc + cc) in cells for rr, cc in DIR4)
                frontier.append(((nr, nc), 1 + 3 * contact))
                seen.add((nr, nc))
        if not frontier:
            return None
        cells.add(random.choices([cell for cell, _ in frontier], weights=[w for _, w in frontier], k=1)[0])
    return _normalize(cells)


def _translated_mask(shape: tuple[int, int], cells: tuple[tuple[int, int], ...], top: int, left: int) -> np.ndarray:
    mask = np.zeros(shape, dtype=bool)
    for dr, dc in cells:
        mask[top + dr, left + dc] = True
    return mask


def _place_shape(
    coarse_shape: tuple[int, int],
    pattern: tuple[tuple[int, int], ...],
    unavailable: np.ndarray,
) -> list[tuple[int, int]] | None:
    pat_h = max(r for r, _ in pattern) + 1
    pat_w = max(c for _, c in pattern) + 1
    H, W = coarse_shape
    placements = []
    for top in range(H - pat_h + 1):
        for left in range(W - pat_w + 1):
            coords = [(top + dr, left + dc) for dr, dc in pattern]
            if all(not unavailable[r, c] for r, c in coords):
                placements.append(coords)
    if not placements:
        return None
    return random.choice(placements)


def _find_frame_candidates(
    cavity_h: int,
    cavity_w: int,
    block_h: int,
    block_w: int,
) -> list[dict[str, object]]:
    H = cavity_h * block_h
    W = cavity_w * block_w
    candidates: list[dict[str, object]] = []
    outer_h = cavity_h + 2
    outer_w = cavity_w + 2
    for r0 in range(H - outer_h + 1):
        for c0 in range(W - outer_w + 1):
            box = (r0, c0, r0 + outer_h - 1, c0 + outer_w - 1)
            border = _frame_border_mask((H, W), box)
            outer = np.zeros((H, W), dtype=bool)
            outer[r0 : r0 + outer_h, c0 : c0 + outer_w] = True
            coarse_border = _downsample_bool(border, cavity_h, cavity_w)
            coarse_outer = _downsample_bool(outer, cavity_h, cavity_w)
            coarse_box = bbox(coarse_border)
            if coarse_box is None or not _forms_hollow_rect(coarse_border):
                continue
            gr0, gc0, gr1, gc1 = coarse_box
            if gr0 == 0 or gc0 == 0 or gr1 == cavity_h - 1 or gc1 == cavity_w - 1:
                continue
            if gr1 - gr0 < 2 or gc1 - gc0 < 2:
                continue
            guard = coarse_outer | _dilate4(coarse_border)
            if int(np.count_nonzero(~guard)) < OBJECT_COUNT_RANGE[0] + 2:
                continue
            candidates.append(
                {
                    "box": box,
                    "coarse_border": coarse_border,
                    "coarse_outer": coarse_outer,
                    "guard": guard,
                }
            )
    return candidates


def _pick_weighted_size() -> int:
    return random.choices(
        [size for size, _ in OBJECT_SIZE_WEIGHTS],
        weights=[weight for _, weight in OBJECT_SIZE_WEIGHTS],
        k=1,
    )[0]


def _render_input(
    cavity_h: int,
    cavity_w: int,
    block_h: int,
    block_w: int,
    frame_box: tuple[int, int, int, int],
    objects: list[tuple[int, list[tuple[int, int]]]],
) -> np.ndarray:
    H = cavity_h * block_h
    W = cavity_w * block_w
    grid = np.full((H, W), Color.BLACK, dtype=int)
    for color, cells in objects:
        for r, c in cells:
            rs = slice(r * block_h, (r + 1) * block_h)
            cs = slice(c * block_w, (c + 1) * block_w)
            grid[rs, cs] = color
    frame = _frame_border_mask((H, W), frame_box)
    grid[frame] = Color.GREY
    return grid


def _find_colored_components(grid: np.ndarray) -> list[tuple[int, int]]:
    H, W = grid.shape
    visited = np.zeros((H, W), dtype=bool)
    components: list[tuple[int, int]] = []
    for r in range(H):
        for c in range(W):
            color = int(grid[r, c])
            if color in {Color.BLACK, Color.GREY} or visited[r, c]:
                continue
            queue = deque([(r, c)])
            visited[r, c] = True
            area = 0
            while queue:
                rr, cc = queue.popleft()
                area += 1
                assert int(grid[rr, cc]) == color, "components must stay monochrome"
                for dr, dc in DIR4:
                    nr = rr + dr
                    nc = cc + dc
                    if 0 <= nr < H and 0 <= nc < W and not visited[nr, nc] and int(grid[nr, nc]) == color:
                        visited[nr, nc] = True
                        queue.append((nr, nc))
            components.append((color, area))
    return components


def _semantic_checks(
    grid: np.ndarray,
    expected_inner: np.ndarray,
    object_count: int,
) -> bool:
    try:
        frame_box = _parse_frame(grid)
        r0, c0, r1, c1 = frame_box
        cavity_h = r1 - r0 - 1
        cavity_w = c1 - c0 - 1
        actual_inner = _downsample_blocks(grid, cavity_h, cavity_w)
        if not np.array_equal(actual_inner, expected_inner):
            return False
        if not _forms_hollow_rect(expected_inner == Color.GREY):
            return False
        if not np.any(expected_inner == Color.BLACK):
            return False
        object_colors = sorted(set(int(x) for x in expected_inner.ravel()) - {Color.BLACK, Color.GREY})
        if len(object_colors) < 2:
            return False
        components = _find_colored_components(grid)
        if len(components) != object_count:
            return False
        if max(area for _color, area in components) <= 1:
            return False
        out = main(grid)
        if not np.array_equal(out[r0 + 1 : r1, c0 + 1 : c1], expected_inner):
            return False
    except Exception:
        return False
    return True


def _gen_attempt() -> np.ndarray | None:
    cavity_h = random.randint(*CAVITY_H_RANGE)
    cavity_w = random.randint(*CAVITY_W_RANGE)
    block_h = random.choice(BLOCK_SIZE_CHOICES)
    block_w = random.choice(BLOCK_SIZE_CHOICES)

    candidates = _find_frame_candidates(cavity_h, cavity_w, block_h, block_w)
    if not candidates:
        return None
    frame_info = random.choice(candidates)
    guard = np.array(frame_info["guard"], copy=True)
    frame_box = tuple(frame_info["box"])

    free_cells = int(np.count_nonzero(~guard))
    max_objects = min(OBJECT_COUNT_RANGE[1], len(PALETTE), free_cells)
    if max_objects < OBJECT_COUNT_RANGE[0]:
        return None
    object_count = random.randint(OBJECT_COUNT_RANGE[0], max_objects)
    colors = random.sample(PALETTE, object_count)

    objects: list[tuple[int, list[tuple[int, int]]]] = []
    saw_large_object = False
    used_patterns: set[tuple[tuple[int, int], ...]] = set()
    for color in colors:
        placed = None
        for _ in range(120):
            size = _pick_weighted_size()
            pattern = _random_polyomino(size)
            if pattern is None:
                continue
            if len(used_patterns) < object_count and pattern in used_patterns and random.random() < 0.7:
                continue
            coords = _place_shape((cavity_h, cavity_w), pattern, guard)
            if coords is None:
                continue
            placed = coords
            used_patterns.add(pattern)
            saw_large_object |= len(coords) > 1
            cell_mask = np.zeros((cavity_h, cavity_w), dtype=bool)
            for r, c in coords:
                cell_mask[r, c] = True
            guard |= cell_mask | _dilate4(cell_mask)
            objects.append((color, coords))
            break
        if placed is None:
            return None

    if not saw_large_object:
        return None

    grid = _render_input(cavity_h, cavity_w, block_h, block_w, frame_box, objects)
    expected_inner = _downsample_blocks(grid, cavity_h, cavity_w)
    if not _semantic_checks(grid, expected_inner, object_count):
        return None
    return grid


def generate_input() -> np.ndarray:
    for _ in range(GEN_RETRY_LIMIT):
        grid = _gen_attempt()
        if grid is not None:
            return grid
    raise RuntimeError("failed to generate a valid viewfinder input")
