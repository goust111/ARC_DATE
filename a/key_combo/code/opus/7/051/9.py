from __future__ import annotations

import random
from collections import Counter

import numpy as np


def _get_bg(grid: np.ndarray) -> int:
    return int(Counter(map(int, grid.ravel())).most_common(1)[0][0])


def _neighbors4(r: int, c: int) -> list[tuple[int, int]]:
    return [(r - 1, c), (r + 1, c), (r, c - 1), (r, c + 1)]


def _components(mask: np.ndarray) -> list[list[tuple[int, int]]]:
    h, w = mask.shape
    seen = np.zeros_like(mask, dtype=bool)
    parts: list[list[tuple[int, int]]] = []
    for r in range(h):
        for c in range(w):
            if not mask[r, c] or seen[r, c]:
                continue
            stack = [(r, c)]
            seen[r, c] = True
            part: list[tuple[int, int]] = []
            while stack:
                rr, cc = stack.pop()
                part.append((rr, cc))
                for nr, nc in _neighbors4(rr, cc):
                    if 0 <= nr < h and 0 <= nc < w and mask[nr, nc] and not seen[nr, nc]:
                        seen[nr, nc] = True
                        stack.append((nr, nc))
            parts.append(part)
    return parts


def _bbox(coords: list[tuple[int, int]]) -> tuple[int, int, int, int]:
    rows = [r for r, _ in coords]
    cols = [c for _, c in coords]
    return min(rows), min(cols), max(rows), max(cols)


def _touches_any_edge(coords: list[tuple[int, int]], shape: tuple[int, int]) -> bool:
    h, w = shape
    return any(r == 0 or r == h - 1 or c == 0 or c == w - 1 for r, c in coords)


def _touches_left_edge(coords: list[tuple[int, int]]) -> bool:
    return any(c == 0 for _, c in coords)


def _completion_candidates(
    shape: tuple[int, int],
    source: list[tuple[int, int]],
    anchor: list[tuple[int, int]],
) -> list[tuple[int, int, int, int, int, int]]:
    h, w = shape
    source_set = set(source)
    anchor_set = set(anchor)
    frag_area = len(source)
    candidates: list[tuple[int, int, int, int, int, int]] = []

    for dr in range(1 - h, h):
        for dc in range(1 - w, w):
            moved = {(r + dr, c + dc) for r, c in source_set}
            if any(r < 0 or r >= h or c < 0 or c >= w for r, c in moved):
                continue
            if moved & anchor_set:
                continue
            union = moved | anchor_set
            rows = [r for r, _ in union]
            cols = [c for _, c in union]
            r0, r1 = min(rows), max(rows)
            c0, c1 = min(cols), max(cols)
            rect_area = (r1 - r0 + 1) * (c1 - c0 + 1)
            if len(union) != rect_area:
                continue
            if rect_area % frag_area != 0:
                continue
            repeat = rect_area // frag_area
            if repeat not in (2, 3):
                continue
            candidates.append((dr, dc, r0, c0, r1, c1))
    return candidates


def _alternating_reflection_tile(tile: np.ndarray, repeat: int) -> np.ndarray:
    row_band = []
    for c in range(repeat):
        row_band.append(np.fliplr(tile) if c % 2 else tile)
    band = np.concatenate(row_band, axis=1)

    rows = []
    flipped_band = np.flipud(band)
    for r in range(repeat):
        rows.append(flipped_band if r % 2 else band)
    return np.concatenate(rows, axis=0)


def main(grid: np.ndarray) -> np.ndarray:
    """
    Move the left-edge fragment onto its anchor to complete a rectangle, then use that rectangle's area ratio to choose a mirrored tiling size.

    Input structure:
    - The grid has a black background and exactly one non-black color.
    - That foreground color forms exactly two 4-connected fragments: one touches the left edge, and one anchor fragment lies away from every grid edge.
    - Translating the left-edge fragment by one unique offset makes the two fragments complement each other into a filled axis-aligned rectangle.

    Rule:
    - Identify the left-edge fragment and the interior anchor fragment of the shared foreground color.
    - Find the unique translation that makes their union a solid rectangle, and compute the rectangle area A and moved fragment area F.
    - Let R = A / F, then discard the original canvas and use the completed rectangle as a source tile.
    - Tile that rectangle into an R-by-R array, alternating horizontal flips across columns and vertical flips across rows.

    Key insight:
    - The fragment must be completed before anything else: the same completion simultaneously reveals the source tile and encodes the output tiling multiplier through the rectangle-to-fragment area ratio.
    """
    g = np.asarray(grid, dtype=int)
    assert g.ndim == 2

    bg = _get_bg(g)
    colors = [int(v) for v in np.unique(g) if int(v) != bg]
    assert len(colors) == 1, "expected exactly one foreground color"
    color = colors[0]

    parts = _components(g == color)
    assert len(parts) == 2, "expected exactly two foreground fragments"

    source_parts = [part for part in parts if _touches_left_edge(part)]
    anchor_parts = [part for part in parts if not _touches_any_edge(part, g.shape)]
    assert len(source_parts) == 1, "expected one left-edge fragment"
    assert len(anchor_parts) == 1, "expected one interior anchor fragment"

    source = source_parts[0]
    anchor = anchor_parts[0]
    assert source is not anchor

    candidates = _completion_candidates(g.shape, source, anchor)
    assert len(candidates) == 1, "expected a unique rectangle-completing translation"
    _dr, _dc, r0, c0, r1, c1 = candidates[0]

    rect_h = r1 - r0 + 1
    rect_w = c1 - c0 + 1
    repeat = (rect_h * rect_w) // len(source)

    tile = np.full((rect_h, rect_w), color, dtype=int)
    return _alternating_reflection_tile(tile, repeat)


_PALETTE = (1, 2, 3, 7)
_MAX_ATTEMPTS = 400


def _rect_options() -> list[tuple[int, int, int, int]]:
    options = []
    for repeat in (2, 3):
        for frag_area in range(2, 6):
            rect_area = repeat * frag_area
            for rect_h in range(2, 6):
                if rect_area % rect_h != 0:
                    continue
                rect_w = rect_area // rect_h
                if 2 <= rect_w <= 5:
                    options.append((repeat, frag_area, rect_h, rect_w))
    return options


_RECT_OPTIONS = _rect_options()


def _connected_mask(mask: np.ndarray) -> bool:
    return len(_components(mask)) == 1


def _random_connected_subset(rect_h: int, rect_w: int, area: int) -> np.ndarray:
    mask = np.zeros((rect_h, rect_w), dtype=bool)
    start = (random.randrange(rect_h), random.randrange(rect_w))
    mask[start] = True
    while int(mask.sum()) < area:
        frontier = set()
        for r, c in np.argwhere(mask):
            for nr, nc in _neighbors4(int(r), int(c)):
                if 0 <= nr < rect_h and 0 <= nc < rect_w and not mask[nr, nc]:
                    frontier.add((nr, nc))
        pick = random.choice(list(frontier))
        mask[pick] = True
    return mask


def _sample_split_masks(rect_h: int, rect_w: int, frag_area: int) -> tuple[np.ndarray, np.ndarray] | None:
    for _ in range(150):
        source_mask = _random_connected_subset(rect_h, rect_w, frag_area)
        anchor_mask = ~source_mask
        if _connected_mask(anchor_mask):
            return source_mask, anchor_mask
    return None


def _place_component(
    grid: np.ndarray,
    coords: list[tuple[int, int]],
    top: int,
    left: int,
    color: int,
) -> None:
    for r, c in coords:
        grid[top + r, left + c] = color


def _normalize_coords(coords: list[tuple[int, int]]) -> list[tuple[int, int]]:
    r0, c0, _r1, _c1 = _bbox(coords)
    return sorted((r - r0, c - c0) for r, c in coords)


def _build_attempt() -> tuple[np.ndarray, np.ndarray] | None:
    repeat, frag_area, rect_h, rect_w = random.choice(_RECT_OPTIONS)
    split = _sample_split_masks(rect_h, rect_w, frag_area)
    if split is None:
        return None
    source_mask, anchor_mask = split

    source_local = [tuple(map(int, p)) for p in np.argwhere(source_mask)]
    anchor_local = [tuple(map(int, p)) for p in np.argwhere(anchor_mask)]

    source_bbox = _normalize_coords(source_local)
    src_rows = [r for r, _ in source_bbox]
    src_cols = [c for _, c in source_bbox]
    src_h = max(src_rows) + 1
    src_w = max(src_cols) + 1

    min_h = max(8, rect_h + 2, src_h + 2)
    min_w = max(8, rect_w + 2, src_w + rect_w + 3)
    if min_h > 12 or min_w > 12:
        return None
    grid_h = random.randint(min_h, 12)
    grid_w = random.randint(min_w, 12)

    rect_top = random.randint(1, grid_h - rect_h - 1)
    rect_left_min = src_w + 2
    rect_left_max = grid_w - rect_w - 1
    if rect_left_min > rect_left_max:
        return None
    rect_left = random.randint(rect_left_min, rect_left_max)

    src_top = random.randint(1, grid_h - src_h - 1)
    color = random.choice(_PALETTE)

    grid = np.zeros((grid_h, grid_w), dtype=int)
    _place_component(grid, anchor_local, rect_top, rect_left, color)
    _place_component(grid, source_bbox, src_top, 0, color)

    parts = _components(grid == color)
    if len(parts) != 2:
        return None

    left_parts = [part for part in parts if _touches_left_edge(part)]
    anchor_parts = [part for part in parts if not _touches_any_edge(part, grid.shape)]
    if len(left_parts) != 1 or len(anchor_parts) != 1:
        return None
    if left_parts[0] is anchor_parts[0]:
        return None
    if len(_completion_candidates(grid.shape, left_parts[0], anchor_parts[0])) != 1:
        return None

    expected = _alternating_reflection_tile(np.full((rect_h, rect_w), color, dtype=int), repeat)
    if not np.array_equal(main(grid), expected):
        return None
    return grid, expected


def generate_input() -> np.ndarray:
    for _ in range(_MAX_ATTEMPTS):
        built = _build_attempt()
        if built is not None:
            return built[0]
    raise RuntimeError("failed to generate valid puzzle input")
