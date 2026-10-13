from __future__ import annotations

import random
from collections import deque

import numpy as np


BG = 0
STRIPE = 1
SPRITE = 3
ANCHOR = 4

PROTO_A = frozenset({(0, 0), (0, 1), (0, 2), (1, 0)})
PROTO_B = frozenset({(0, 0), (0, 1), (0, 2), (1, 0), (1, 1), (1, 2)})
PROTOS = (PROTO_A, PROTO_B)

GRID_ROWS = (18, 24)
GRID_COLS = (18, 24)
SPRITE_COUNTS = (3, 4)
MIN_COL_GAP = 2
MAX_GEN_ATTEMPTS = 400


def _components(mask: np.ndarray) -> list[np.ndarray]:
    h, w = mask.shape
    seen = np.zeros_like(mask, dtype=bool)
    comps: list[np.ndarray] = []
    for r in range(h):
        for c in range(w):
            if not mask[r, c] or seen[r, c]:
                continue
            q = deque([(r, c)])
            seen[r, c] = True
            coords: list[tuple[int, int]] = []
            while q:
                rr, cc = q.popleft()
                coords.append((rr, cc))
                for nr, nc in ((rr - 1, cc), (rr + 1, cc), (rr, cc - 1), (rr, cc + 1)):
                    if 0 <= nr < h and 0 <= nc < w and mask[nr, nc] and not seen[nr, nc]:
                        seen[nr, nc] = True
                        q.append((nr, nc))
            comps.append(np.array(coords, dtype=int))
    return comps


def _extract_sprites(grid: np.ndarray) -> list[dict]:
    stripe_rows = [r for r in range(grid.shape[0]) if np.all(grid[r] == STRIPE)]
    assert stripe_rows == [grid.shape[0] - 1], "expected exactly one full-width bottom stripe"
    stripe_row = stripe_rows[0]

    mask = np.zeros_like(grid, dtype=bool)
    mask[:stripe_row] = np.isin(grid[:stripe_row], [SPRITE, ANCHOR])
    comps = _components(mask)
    assert 3 <= len(comps) <= 4, "expected 3-4 sprites"

    sprites: list[dict] = []
    for coords in comps:
        values = grid[coords[:, 0], coords[:, 1]]
        anchor_cells = coords[values == ANCHOR]
        assert len(anchor_cells) == 1, "each sprite must contain exactly one anchor"
        anchor_r, anchor_c = map(int, anchor_cells[0])
        shape = frozenset((int(r - anchor_r), int(c - anchor_c)) for r, c in coords)
        rows = coords[:, 0]
        cols = coords[:, 1]
        sprites.append(
            {
                "coords": coords,
                "anchor": (anchor_r, anchor_c),
                "shape": shape,
                "area": int(len(shape)),
                "bbox": (
                    int(rows.min()),
                    int(cols.min()),
                    int(rows.max()),
                    int(cols.max()),
                ),
            }
        )
    sprites.sort(key=lambda item: item["anchor"][1])
    return sprites


def _other_proto(shape: frozenset[tuple[int, int]]) -> frozenset[tuple[int, int]]:
    a, b = sorted(PROTOS, key=len)
    if shape == a:
        return b
    if shape == b:
        return a
    raise AssertionError("unexpected prototype shape")


def _paint_shape(grid: np.ndarray, shape: frozenset[tuple[int, int]], anchor_r: int, anchor_c: int) -> None:
    for dr, dc in shape:
        rr = anchor_r + dr
        cc = anchor_c + dc
        assert 0 <= rr < grid.shape[0] - 1 and 0 <= cc < grid.shape[1]
        grid[rr, cc] = ANCHOR if (dr, dc) == (0, 0) else SPRITE


def _safe_paint_shape(
    grid: np.ndarray,
    shape: frozenset[tuple[int, int]],
    anchor_r: int,
    anchor_c: int,
) -> bool:
    for dr, dc in shape:
        rr = anchor_r + dr
        cc = anchor_c + dc
        if not (0 <= rr < grid.shape[0] - 1 and 0 <= cc < grid.shape[1]):
            return False
        if grid[rr, cc] != BG:
            return False
    _paint_shape(grid, shape, anchor_r, anchor_c)
    return True


def main(grid: np.ndarray) -> np.ndarray:
    """
    Swap each anchored sprite to the other prototype, then move it upward by the new prototype's area.

    Input structure:
    - The background is black, and the last row is a solid blue stripe that stays fixed.
    - Above the stripe are 3-4 disconnected green sprites, each containing exactly one yellow anchor pixel.
    - Every sprite is one of two prototype shapes, both aligned by the anchor cell.

    Rule:
    - Identify the two prototype shapes from the sprites themselves.
    - For each sprite, replace its current prototype with the other prototype while keeping the anchor-aligned local layout.
    - After swapping, translate the whole new sprite straight upward by the number of cells in the new prototype; leave the blue stripe unchanged.

    Key insight:
    - The displacement depends on the swapped-in shape, so the solver must toggle the prototype first and only then count the new area to know how far to move.
    """
    g = np.asarray(grid, dtype=int)
    sprites = _extract_sprites(g)
    proto_shapes = sorted({sprite["shape"] for sprite in sprites}, key=len)
    assert len(proto_shapes) == 2, "expected exactly two prototype shapes"

    out = g.copy()
    for sprite in sprites:
        coords = sprite["coords"]
        out[coords[:, 0], coords[:, 1]] = BG

    occupied = np.zeros_like(g, dtype=bool)
    occupied[-1, :] = True
    for sprite in sprites:
        target_shape = _other_proto(sprite["shape"])
        shift = len(target_shape)
        anchor_r, anchor_c = sprite["anchor"]
        new_anchor_r = anchor_r - shift
        for dr, dc in target_shape:
            rr = new_anchor_r + dr
            cc = anchor_c + dc
            assert 0 <= rr < g.shape[0] - 1 and 0 <= cc < g.shape[1]
            assert not occupied[rr, cc], "translated sprites must not overlap"
            occupied[rr, cc] = True
            out[rr, cc] = ANCHOR if (dr, dc) == (0, 0) else SPRITE
    return out


def _column_starts(width: int, count: int) -> list[int]:
    min_used = 3 * count + MIN_COL_GAP * (count - 1)
    assert width >= min_used
    slack = width - min_used
    gaps = [0] * (count + 1)
    for _ in range(slack):
        gaps[random.randrange(count + 1)] += 1

    starts: list[int] = []
    c = gaps[0]
    for i in range(count):
        starts.append(c)
        c += 3
        if i < count - 1:
            c += MIN_COL_GAP + gaps[i + 1]
    return starts


def _sample_kinds(count: int) -> list[frozenset[tuple[int, int]]]:
    num_a = random.randint(1, count - 1)
    kinds = [PROTO_A] * num_a + [PROTO_B] * (count - num_a)
    random.shuffle(kinds)
    return kinds


def _render_scene(height: int, width: int, sprites: list[dict]) -> np.ndarray:
    grid = np.full((height, width), BG, dtype=int)
    grid[-1, :] = STRIPE
    for sprite in sprites:
        _paint_shape(grid, sprite["shape"], sprite["anchor"][0], sprite["anchor"][1])
    return grid


def _solve_with_old_area(grid: np.ndarray) -> np.ndarray:
    g = np.asarray(grid, dtype=int)
    sprites = _extract_sprites(g)
    out = g.copy()
    for sprite in sprites:
        coords = sprite["coords"]
        out[coords[:, 0], coords[:, 1]] = BG
    for sprite in sprites:
        target_shape = _other_proto(sprite["shape"])
        shift = len(sprite["shape"])
        anchor_r, anchor_c = sprite["anchor"]
        if not _safe_paint_shape(out, target_shape, anchor_r - shift, anchor_c):
            return np.full_like(g, 9)
    return out


def _solve_without_move(grid: np.ndarray) -> np.ndarray:
    g = np.asarray(grid, dtype=int)
    sprites = _extract_sprites(g)
    out = g.copy()
    for sprite in sprites:
        coords = sprite["coords"]
        out[coords[:, 0], coords[:, 1]] = BG
    for sprite in sprites:
        target_shape = _other_proto(sprite["shape"])
        anchor_r, anchor_c = sprite["anchor"]
        if not _safe_paint_shape(out, target_shape, anchor_r, anchor_c):
            return np.full_like(g, 9)
    return out


def _validate_scene(inp: np.ndarray, expected: np.ndarray) -> bool:
    if not np.array_equal(main(inp), expected):
        return False
    if np.array_equal(_solve_with_old_area(inp), expected):
        return False
    if np.array_equal(_solve_without_move(inp), expected):
        return False

    in_sprites = _extract_sprites(inp)
    out_sprites = _extract_sprites(expected)
    if len(in_sprites) != len(out_sprites):
        return False

    in_shapes = {sprite["shape"] for sprite in in_sprites}
    out_shapes = {sprite["shape"] for sprite in out_sprites}
    if in_shapes != {PROTO_A, PROTO_B}:
        return False
    if out_shapes != {PROTO_A, PROTO_B}:
        return False

    by_col_in = {sprite["anchor"][1]: sprite for sprite in in_sprites}
    by_col_out = {sprite["anchor"][1]: sprite for sprite in out_sprites}
    if set(by_col_in) != set(by_col_out):
        return False

    for col, sprite_in in by_col_in.items():
        sprite_out = by_col_out[col]
        target_shape = _other_proto(sprite_in["shape"])
        if sprite_out["shape"] != target_shape:
            return False
        if sprite_out["anchor"][0] != sprite_in["anchor"][0] - len(target_shape):
            return False

    return True


def _attempt_generate() -> np.ndarray | None:
    height = random.randint(*GRID_ROWS)
    width = random.randint(*GRID_COLS)
    count = random.choice(SPRITE_COUNTS)
    if width < 3 * count + MIN_COL_GAP * (count - 1):
        return None

    starts = _column_starts(width, count)
    current_shapes = _sample_kinds(count)

    input_sprites: list[dict] = []
    output_sprites: list[dict] = []
    for col, shape in zip(starts, current_shapes):
        target_shape = _other_proto(shape)
        shift = len(target_shape)
        max_out_top = height - shift - 3
        if max_out_top < 0:
            return None
        out_top = random.randint(0, max_out_top)
        in_top = out_top + shift
        input_sprites.append({"shape": shape, "anchor": (in_top, col)})
        output_sprites.append({"shape": target_shape, "anchor": (out_top, col)})

    inp = _render_scene(height, width, input_sprites)
    out = _render_scene(height, width, output_sprites)
    if not _validate_scene(inp, out):
        return None
    return inp


def generate_input() -> np.ndarray:
    for _ in range(MAX_GEN_ATTEMPTS):
        grid = _attempt_generate()
        if grid is not None:
            return grid
    raise RuntimeError("failed to generate a valid input grid")
