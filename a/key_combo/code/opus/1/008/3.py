import random
from collections import deque

import numpy as np


BG = 0
NEUTRAL = 5
MARKER_COLORS = (1, 2, 3, 4)
GEN_RETRIES = 200


def _neighbors4(r: int, c: int):
    yield r - 1, c
    yield r + 1, c
    yield r, c - 1
    yield r, c + 1


def _most_common_color(grid: np.ndarray) -> int:
    values, counts = np.unique(grid, return_counts=True)
    return int(values[np.argmax(counts)])


def _components(mask: np.ndarray) -> list[list[tuple[int, int]]]:
    h, w = mask.shape
    seen = np.zeros((h, w), dtype=bool)
    comps: list[list[tuple[int, int]]] = []
    for r in range(h):
        for c in range(w):
            if not mask[r, c] or seen[r, c]:
                continue
            q = deque([(r, c)])
            seen[r, c] = True
            comp: list[tuple[int, int]] = []
            while q:
                rr, cc = q.popleft()
                comp.append((rr, cc))
                for nr, nc in _neighbors4(rr, cc):
                    if 0 <= nr < h and 0 <= nc < w and mask[nr, nc] and not seen[nr, nc]:
                        seen[nr, nc] = True
                        q.append((nr, nc))
            comps.append(comp)
    return comps


def _bbox(coords: list[tuple[int, int]]) -> tuple[int, int, int, int]:
    rows = [r for r, _ in coords]
    cols = [c for _, c in coords]
    return min(rows), min(cols), max(rows), max(cols)


def _transform_component(
    coords: list[tuple[int, int]], width: int, mode: str
) -> set[tuple[int, int]]:
    if mode == "bbox":
        r0, c0, _r1, c1 = _bbox(coords)
        sprite_w = c1 - c0 + 1
        dst_c0 = width - sprite_w - c0
        return {(r, dst_c0 + (c - c0)) for r, c in coords}
    if mode == "pixel":
        return {(r, width - 1 - c) for r, c in coords}
    raise ValueError(f"unknown mirror mode: {mode}")


def _extract_markers(
    grid: np.ndarray, bg: int, neutral: int
) -> dict[tuple[int, int], int]:
    h, w = grid.shape
    markers: dict[tuple[int, int], int] = {}
    other = np.argwhere((grid != bg) & (grid != neutral))
    for r, c in other:
        rr = int(r)
        cc = int(c)
        assert cc == w - 1, "all non-neutral labels must sit on the right border"
        markers[(rr, cc)] = int(grid[rr, cc])
    assert markers, "expected border markers"
    assert len(set(markers.values())) == len(markers), "marker colors must be unique"
    return markers


def _solve_with_mode(grid: np.ndarray, mode: str) -> np.ndarray:
    g = np.asarray(grid, dtype=int)
    h, w = g.shape
    bg = _most_common_color(g)
    markers = _extract_markers(g, bg, NEUTRAL)
    sprites = _components(g == NEUTRAL)
    assert sprites, "expected neutral sprites"

    out = np.full_like(g, bg)
    occupied: set[tuple[int, int]] = set()
    used_markers: set[tuple[int, int]] = set()

    for comp in sprites:
        moved = _transform_component(comp, w, mode)
        assert moved, "transformed sprite cannot be empty"
        assert occupied.isdisjoint(moved), "mirrored sprites must not overlap"
        touched = set()
        for r, c in moved:
            assert 0 <= r < h and 0 <= c < w, "mirrored sprite must stay in bounds"
            assert (r, c) not in markers, "marker cannot occupy a mirrored sprite pixel"
            for nr, nc in _neighbors4(r, c):
                if (nr, nc) in markers:
                    touched.add((nr, nc))
        assert len(touched) == 1, "each mirrored sprite must touch exactly one marker"
        marker = next(iter(touched))
        assert marker not in used_markers, "each marker must label exactly one sprite"
        used_markers.add(marker)
        color = markers[marker]
        for r, c in moved:
            occupied.add((r, c))
            out[r, c] = color

    assert used_markers == set(markers), "every marker must be consumed by one mirrored sprite"
    return out


def main(grid: np.ndarray) -> np.ndarray:
    """
    Mirror each gray sprite to the opposite side, then recolor it from the border marker it only touches after the move.

    Input structure:
    - The background is black.
    - The left side contains several small gray 4-connected sprites.
    - The rightmost column contains isolated colored marker pixels, one per sprite.

    Rule:
    - Move every gray sprite to its horizontally mirrored counterpart on the other side of the grid.
    - For each moved sprite, find the unique right-border marker that is 4-neighbor adjacent to the moved sprite and repaint the whole sprite with that marker color.
    - Erase the original gray sprites and erase all marker pixels, leaving only the moved recolored sprites on black.

    Key insight:
    - The marker does not label the sprite in its input position; the label becomes adjacent only after the mirror move, so the mirror must happen before color transfer.
    """
    g = np.asarray(grid, dtype=int)
    assert g.ndim == 2 and g.size > 0, "expected a non-empty 2D grid"

    candidates: list[np.ndarray] = []
    for mode in ("bbox", "pixel"):
        try:
            candidates.append(_solve_with_mode(g, mode))
        except AssertionError:
            continue

    assert candidates, "no consistent mirrored placement found"
    if len(candidates) == 2:
        assert np.array_equal(candidates[0], candidates[1]), "input would be ambiguous under two mirror interpretations"
    return candidates[0]


SHAPES = (
    ((0, 0), (0, 1), (1, 1)),
    ((0, 1), (1, 0), (1, 1)),
    ((0, 0), (0, 1), (1, 1), (1, 2)),
    ((0, 1), (0, 2), (1, 0), (1, 1)),
    ((0, 1), (0, 2), (1, 0), (1, 1), (1, 2)),
    ((0, 0), (0, 1), (0, 2), (1, 1), (1, 2)),
    ((0, 0), (0, 1), (0, 2), (1, 2)),
    ((0, 2), (1, 0), (1, 1), (1, 2)),
)


def _shape_info(shape: tuple[tuple[int, int], ...]) -> dict[str, object]:
    rows = [r for r, _ in shape]
    cols = [c for _, c in shape]
    h = max(rows) + 1
    w = max(cols) + 1
    left_rows = {r for r, c in shape if c == 0}
    right_rows = {r for r, c in shape if c == w - 1}
    right_only = tuple(sorted(right_rows - left_rows))
    return {"coords": shape, "height": h, "width": w, "right_only_rows": right_only}


SHAPE_INFOS = tuple(_shape_info(shape) for shape in SHAPES)


def _validate_generated_instance(grid: np.ndarray) -> None:
    g = np.asarray(grid, dtype=int)
    assert g.ndim == 2
    h, w = g.shape
    assert h <= 30 and w <= 30
    assert _most_common_color(g) == BG

    sprites = _components(g == NEUTRAL)
    assert 2 <= len(sprites) <= 4

    markers = _extract_markers(g, BG, NEUTRAL)
    assert len(markers) == len(sprites)
    assert set(markers.values()).issubset(set(MARKER_COLORS))

    marker_positions = set(markers)
    for comp in sprites:
        r0, c0, r1, c1 = _bbox(comp)
        assert r1 - r0 + 1 == 2
        assert c1 - c0 + 1 in (2, 3)
        assert 3 <= len(comp) <= 5
        assert len(comp) < (r1 - r0 + 1) * (c1 - c0 + 1)

        rel = {(r - r0, c - c0) for r, c in comp}
        left_rows = {rr for rr, cc in rel if cc == 0}
        right_rows = {rr for rr, cc in rel if cc == c1 - c0}
        assert right_rows - left_rows, "need a row that distinguishes mirrored position from pixel flip"

        for r, c in comp:
            for nr, nc in _neighbors4(r, c):
                assert (nr, nc) not in marker_positions, "markers must be far from input sprites"

    bbox_out = _solve_with_mode(g, "bbox")
    try:
        pixel_out = _solve_with_mode(g, "pixel")
    except AssertionError:
        pixel_out = None
    assert pixel_out is None or not np.array_equal(bbox_out, pixel_out), "generator must exclude flip-based shortcuts"

    assert NEUTRAL not in set(map(int, np.unique(bbox_out)))
    assert np.count_nonzero(bbox_out) == np.count_nonzero(g == NEUTRAL)


def _gen_attempt() -> np.ndarray | None:
    num_sprites = random.randint(2, 4)
    colors = random.sample(MARKER_COLORS, k=num_sprites)
    infos = [dict(random.choice(SHAPE_INFOS)) for _ in range(num_sprites)]

    base_rows = [0]
    for _ in range(1, num_sprites):
        base_rows.append(base_rows[-1] + 2 + random.randint(1, 2))
    used_height = base_rows[-1] + 2
    min_height = max(10, used_height)
    max_height = min(18, used_height + 6)
    if min_height > 18:
        return None
    max_height = max(min_height, max_height)
    height = random.randint(min_height, max_height)
    top_offset = random.randint(0, height - used_height)
    rows = [r + top_offset for r in base_rows]

    width = random.randint(10, 16)
    grid = np.full((height, width), BG, dtype=int)
    sprite_col = 1

    sprite_specs = list(zip(rows, colors, infos))
    random.shuffle(sprite_specs)

    for r0, color, info in sprite_specs:
        coords = info["coords"]
        sprite_w = int(info["width"])
        if sprite_col + sprite_w >= width - 2:
            return None

        for dr, dc in coords:
            grid[r0 + dr, sprite_col + dc] = NEUTRAL

        marker_row = r0 + random.choice(info["right_only_rows"])
        if grid[marker_row, width - 1] != BG:
            return None
        grid[marker_row, width - 1] = color

    return grid


def generate_input() -> np.ndarray:
    for _ in range(GEN_RETRIES):
        grid = _gen_attempt()
        if grid is None:
            continue
        try:
            _validate_generated_instance(grid)
        except AssertionError:
            continue
        return grid
    raise RuntimeError("failed to generate a valid mirrored-marker recoloring task")
