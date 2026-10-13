from __future__ import annotations

from collections import deque
import random

import numpy as np


BLACK = 0
BLUE = 1
RED = 2
YELLOW = 4
GREY = 5
PINK = 6
ORANGE = 7
TEAL = 8

SYMBOL_TO_COLOR: dict[tuple[tuple[int, ...], ...], int] = {
    (
        (0, 1, 0),
        (1, 1, 1),
        (0, 1, 0),
    ): BLUE,
    (
        (1, 0, 1),
        (0, 1, 0),
        (1, 0, 1),
    ): YELLOW,
    (
        (1, 1, 1),
        (0, 1, 0),
        (0, 1, 0),
    ): ORANGE,
    (
        (1, 0, 0),
        (1, 0, 0),
        (1, 1, 1),
    ): TEAL,
    (
        (1, 1, 0),
        (0, 1, 0),
        (0, 1, 1),
    ): PINK,
}
SYMBOL_PATTERNS = list(SYMBOL_TO_COLOR)
OUTPUT_COLORS = set(SYMBOL_TO_COLOR.values())

GRID_HEIGHT_RANGE = (18, 27)
GRID_WIDTH_RANGE = (18, 27)
REGION_MARGIN = 1
REGION_GAP = 2
MAX_GENERATION_ATTEMPTS = 400

BASE_INTERIOR_SIZE = 5
SHAPE_SPECS = [
    ("top", 1),
    ("top", 2),
    ("top", 3),
    ("bottom", 1),
    ("bottom", 2),
    ("bottom", 3),
    ("left", 1),
    ("left", 2),
    ("left", 3),
    ("right", 1),
    ("right", 2),
    ("right", 3),
]
SCHEDULED_SYMBOL_GROUPS = (
    (0, 1, 2, 3),
    (1, 2, 3, 4),
    (0, 2, 4),
)
_GENERATION_CALL_COUNT = 0


def _neighbors4(r: int, c: int) -> tuple[tuple[int, int], ...]:
    return ((r - 1, c), (r + 1, c), (r, c - 1), (r, c + 1))


def _bbox_coords(coords: np.ndarray) -> tuple[int, int, int, int]:
    r0, c0 = coords.min(axis=0)
    r1, c1 = coords.max(axis=0)
    return int(r0), int(c0), int(r1), int(c1)


def _components(mask: np.ndarray) -> list[np.ndarray]:
    mask = np.asarray(mask, dtype=bool)
    h, w = mask.shape
    seen = np.zeros((h, w), dtype=bool)
    out: list[np.ndarray] = []
    for r in range(h):
        for c in range(w):
            if not mask[r, c] or seen[r, c]:
                continue
            seen[r, c] = True
            queue = deque([(r, c)])
            coords = []
            while queue:
                rr, cc = queue.popleft()
                coords.append((rr, cc))
                for nr, nc in _neighbors4(rr, cc):
                    if 0 <= nr < h and 0 <= nc < w and mask[nr, nc] and not seen[nr, nc]:
                        seen[nr, nc] = True
                        queue.append((nr, nc))
            out.append(np.asarray(coords, dtype=int))
    return out


def _dilate4(mask: np.ndarray) -> np.ndarray:
    mask = np.asarray(mask, dtype=bool)
    out = mask.copy()
    out[1:, :] |= mask[:-1, :]
    out[:-1, :] |= mask[1:, :]
    out[:, 1:] |= mask[:, :-1]
    out[:, :-1] |= mask[:, 1:]
    return out


def _dilate8(mask: np.ndarray) -> np.ndarray:
    mask = np.asarray(mask, dtype=bool)
    out = mask.copy()
    for dr in (-1, 0, 1):
        for dc in (-1, 0, 1):
            if dr == 0 and dc == 0:
                continue
            src_r0 = max(0, -dr)
            src_r1 = mask.shape[0] - max(0, dr)
            src_c0 = max(0, -dc)
            src_c1 = mask.shape[1] - max(0, dc)
            dst_r0 = max(0, dr)
            dst_r1 = dst_r0 + (src_r1 - src_r0)
            dst_c0 = max(0, dc)
            dst_c1 = dst_c0 + (src_c1 - src_c0)
            out[dst_r0:dst_r1, dst_c0:dst_c1] |= mask[src_r0:src_r1, src_c0:src_c1]
    return out


def _reachable_from_outside(passable: np.ndarray) -> np.ndarray:
    passable = np.asarray(passable, dtype=bool)
    padded = np.ones((passable.shape[0] + 2, passable.shape[1] + 2), dtype=bool)
    padded[1:-1, 1:-1] = passable

    seen = np.zeros_like(padded, dtype=bool)
    seen[0, 0] = True
    queue = deque([(0, 0)])
    while queue:
        r, c = queue.popleft()
        for nr, nc in _neighbors4(r, c):
            if 0 <= nr < padded.shape[0] and 0 <= nc < padded.shape[1] and padded[nr, nc] and not seen[nr, nc]:
                seen[nr, nc] = True
                queue.append((nr, nc))
    return seen[1:-1, 1:-1]


def _enclosed_region_components(grid: np.ndarray) -> list[np.ndarray]:
    passable = np.asarray(grid != GREY, dtype=bool)
    reachable = _reachable_from_outside(passable)
    enclosed = passable & ~reachable
    return _components(enclosed)


def _region_symbol_signature(grid: np.ndarray, coords: np.ndarray) -> tuple[tuple[int, ...], ...]:
    rr = coords[:, 0]
    cc = coords[:, 1]
    red_coords = coords[grid[rr, cc] == RED]
    assert red_coords.size > 0, "each enclosed region must contain a red symbol"
    r0, c0, r1, c1 = _bbox_coords(red_coords)
    assert (r1 - r0 + 1, c1 - c0 + 1) == (3, 3), "symbol bbox must be 3x3"

    region_mask = np.zeros_like(grid, dtype=bool)
    region_mask[rr, cc] = True
    assert region_mask[r0 : r1 + 1, c0 : c1 + 1].all(), "symbol bbox must lie fully inside one region"

    patch = grid[r0 : r1 + 1, c0 : c1 + 1]
    assert set(map(int, np.unique(patch))) <= {BLACK, RED}, "symbol patch must contain only black and red"
    signature = tuple(tuple(int(v == RED) for v in row) for row in patch)
    assert signature in SYMBOL_TO_COLOR, "unknown symbol pattern"
    return signature


def main(grid: np.ndarray) -> np.ndarray:
    """Decode each enclosed 3x3 red symbol and flood-fill its irregular grey-bounded interior with the decoded color.

    Input structure:
    - The background is black.
    - Several irregular one-pixel-thick grey loops each enclose one black interior region.
    - Every enclosed interior contains exactly one 3x3 red-on-black symbol mask separated from the grey boundary by black cells.

    Rule:
    - Treat grey pixels as walls and identify each strictly enclosed region inside a closed loop.
    - Extract the red symbol's 3x3 bounding box inside that region and decode the binary mask with the task's fixed symbol-to-color codebook.
    - Recolor every pixel strictly inside that loop, including the red symbol pixels themselves, to the decoded color while keeping the grey boundary and exterior background unchanged.

    Key insight:
    - The boundary shape tells you where filling may occur, but only the symbol inside that boundary tells you which color to use, so both channels must be read together.
    """
    g = np.asarray(grid, dtype=int)
    assert g.ndim == 2, "expected a 2D grid"

    out = g.copy()
    regions = _enclosed_region_components(g)
    assert regions, "expected at least one enclosed region"
    for coords in regions:
        signature = _region_symbol_signature(g, coords)
        out[coords[:, 0], coords[:, 1]] = SYMBOL_TO_COLOR[signature]
    return out


def _symbol_indices_for_call(call_index: int) -> tuple[int, ...]:
    return SCHEDULED_SYMBOL_GROUPS[call_index % len(SCHEDULED_SYMBOL_GROUPS)]


def _make_region_patch(spec: tuple[str, int], symbol_index: int) -> np.ndarray:
    side, length = spec
    interior = np.zeros((9, 9), dtype=bool)
    interior[2 : 2 + BASE_INTERIOR_SIZE, 2 : 2 + BASE_INTERIOR_SIZE] = True
    start = 4 - (length - 1) // 2
    if side == "top":
        interior[1, start : start + length] = True
    elif side == "bottom":
        interior[7, start : start + length] = True
    elif side == "left":
        interior[start : start + length, 1] = True
    else:
        interior[start : start + length, 7] = True

    boundary = _dilate8(interior) & ~interior
    filled = boundary | interior
    r0, c0, r1, c1 = _bbox_coords(np.argwhere(filled))
    interior = interior[r0 : r1 + 1, c0 : c1 + 1]
    boundary = boundary[r0 : r1 + 1, c0 : c1 + 1]

    patch = np.zeros_like(interior, dtype=int)
    patch[boundary] = GREY
    assert len(_components(boundary)) == 1, "grey boundary must be a single 4-connected loop"

    candidates = []
    for r0 in range(1, patch.shape[0] - 3):
        for c0 in range(1, patch.shape[1] - 3):
            if interior[r0 - 1 : r0 + 4, c0 - 1 : c0 + 4].all():
                candidates.append((r0, c0))
    assert candidates, "interior must allow a 3x3 symbol with one-cell margin"

    sr, sc = candidates[0]
    symbol = SYMBOL_PATTERNS[symbol_index]
    for dr in range(3):
        for dc in range(3):
            if symbol[dr][dc]:
                patch[sr + dr, sc + dc] = RED

    filled_area = int(np.count_nonzero(boundary) + interior.sum())
    assert 25 <= filled_area <= 60, "region area must match the puzzle bounds"
    return patch


def _boxes_separated(
    top_left: tuple[int, int],
    shape: tuple[int, int],
    placed_boxes: list[tuple[int, int, int, int]],
) -> bool:
    r0, c0 = top_left
    h, w = shape
    r1 = r0 + h - 1
    c1 = c0 + w - 1
    for pr0, pc0, pr1, pc1 in placed_boxes:
        if not (r1 < pr0 - REGION_GAP or pr1 < r0 - REGION_GAP or c1 < pc0 - REGION_GAP or pc1 < c0 - REGION_GAP):
            return False
    return True


def _place_patches(
    grid: np.ndarray,
    patches: list[np.ndarray],
) -> bool:
    h, w = grid.shape
    placed_boxes: list[tuple[int, int, int, int]] = []
    for patch in patches:
        ph, pw = patch.shape
        candidates = []
        for r0 in range(REGION_MARGIN, h - ph - REGION_MARGIN + 1):
            for c0 in range(REGION_MARGIN, w - pw - REGION_MARGIN + 1):
                if _boxes_separated((r0, c0), (ph, pw), placed_boxes):
                    candidates.append((r0, c0))
        if not candidates:
            return False
        r0, c0 = random.choice(candidates)
        mask = patch != BLACK
        view = grid[r0 : r0 + ph, c0 : c0 + pw]
        assert not np.any(view[mask] != BLACK)
        view[mask] = patch[mask]
        placed_boxes.append((r0, c0, r0 + ph - 1, c0 + pw - 1))
    return True


def _normalized_shape(coords: np.ndarray) -> tuple[tuple[int, ...], ...]:
    r0, c0, r1, c1 = _bbox_coords(coords)
    mask = np.zeros((r1 - r0 + 1, c1 - c0 + 1), dtype=int)
    mask[coords[:, 0] - r0, coords[:, 1] - c0] = 1
    return tuple(tuple(int(v) for v in row) for row in mask)


def _repeated_shape_with_distinct_symbols_ok(grid: np.ndarray) -> bool:
    regions = _enclosed_region_components(grid)
    shapes: dict[tuple[tuple[int, ...], ...], list[tuple[tuple[int, ...], ...]]] = {}
    for coords in regions:
        shapes.setdefault(_normalized_shape(coords), []).append(_region_symbol_signature(grid, coords))
    for signatures in shapes.values():
        if len(signatures) >= 2 and len(set(signatures)) >= 2:
            return True
    return False


def _semantic_checks(inp: np.ndarray, expected_region_count: int) -> bool:
    try:
        regions = _enclosed_region_components(inp)
        if len(regions) != expected_region_count:
            return False
        grey_components = _components(inp == GREY)
        if len(grey_components) != expected_region_count:
            return False
        reachable = _reachable_from_outside(inp != GREY)

        signatures = []
        for coords in regions:
            signature = _region_symbol_signature(inp, coords)
            signatures.append(signature)

        if not _repeated_shape_with_distinct_symbols_ok(inp):
            return False

        out = main(inp)
        if out.shape != inp.shape:
            return False
        if set(map(int, np.unique(out))) - {BLACK, GREY} - OUTPUT_COLORS:
            return False

        for coords, signature in zip(regions, signatures):
            color = SYMBOL_TO_COLOR[signature]
            rr = coords[:, 0]
            cc = coords[:, 1]
            if not np.all(out[rr, cc] == color):
                return False

        if np.any((inp == GREY) & (out != GREY)):
            return False
        if not np.array_equal(out[reachable], inp[reachable]):
            return False
        return not np.array_equal(inp, out)
    except Exception:
        return False


def _random_shape_specs(region_count: int) -> list[tuple[str, int]]:
    repeated = random.choice(SHAPE_SPECS)
    specs = [repeated, repeated]
    while len(specs) < region_count:
        specs.append(random.choice(SHAPE_SPECS))
    random.shuffle(specs)
    return specs


def _attempt_generate(symbol_group: tuple[int, ...]) -> np.ndarray | None:
    symbol_indices = list(symbol_group)
    region_count = len(symbol_indices)

    grid_h = random.randint(*GRID_HEIGHT_RANGE)
    grid_w = random.randint(*GRID_WIDTH_RANGE)
    grid = np.zeros((grid_h, grid_w), dtype=int)

    shape_specs = _random_shape_specs(region_count)
    random.shuffle(symbol_indices)
    patches = [_make_region_patch(spec, symbol_index) for spec, symbol_index in zip(shape_specs, symbol_indices)]
    random.shuffle(patches)

    if not _place_patches(grid, patches):
        return None
    if not _semantic_checks(grid, region_count):
        return None
    return grid


def generate_input() -> np.ndarray:
    global _GENERATION_CALL_COUNT
    symbol_group = _symbol_indices_for_call(_GENERATION_CALL_COUNT)
    for _ in range(MAX_GENERATION_ATTEMPTS):
        grid = _attempt_generate(symbol_group)
        if grid is not None:
            _GENERATION_CALL_COUNT += 1
            return grid
    raise RuntimeError("failed to generate a valid symbol-decoding flood-fill input")


_symbol_pixel_counts = {sum(sum(row) for row in signature) for signature in SYMBOL_PATTERNS}
assert len(SYMBOL_TO_COLOR) == 5
assert len(OUTPUT_COLORS) == 5
assert len(_symbol_pixel_counts) == 1, "all symbols should have the same red-pixel count to block count-based shortcuts"
