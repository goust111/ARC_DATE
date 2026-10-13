from __future__ import annotations

import random
from collections import Counter, deque

import numpy as np


BACKGROUND = 0
GREY = 5
OPS = ("mirror_lr", "mirror_ud", "rotate_180")
MAX_ATTEMPTS = 500

_OP_CYCLE: list[str] = []


def _mode(grid: np.ndarray) -> int:
    return int(Counter(map(int, grid.ravel())).most_common(1)[0][0])


def _bbox(mask: np.ndarray) -> tuple[int, int, int, int] | None:
    coords = np.argwhere(mask)
    if coords.size == 0:
        return None
    r0, c0 = coords.min(axis=0)
    r1, c1 = coords.max(axis=0)
    return int(r0), int(c0), int(r1), int(c1)


def _crop_bbox(grid: np.ndarray, box: tuple[int, int, int, int]) -> np.ndarray:
    r0, c0, r1, c1 = box
    return grid[r0 : r1 + 1, c0 : c1 + 1].copy()


def _components(mask: np.ndarray, connectivity: int = 4) -> list[np.ndarray]:
    h, w = mask.shape
    seen = np.zeros_like(mask, dtype=bool)
    if connectivity == 8:
        steps = [
            (-1, -1), (-1, 0), (-1, 1),
            (0, -1), (0, 1),
            (1, -1), (1, 0), (1, 1),
        ]
    else:
        steps = [(-1, 0), (1, 0), (0, -1), (0, 1)]

    out: list[np.ndarray] = []
    for sr, sc in np.argwhere(mask):
        sr = int(sr)
        sc = int(sc)
        if seen[sr, sc]:
            continue
        seen[sr, sc] = True
        q: deque[tuple[int, int]] = deque([(sr, sc)])
        cells: list[tuple[int, int]] = []
        while q:
            r, c = q.popleft()
            cells.append((r, c))
            for dr, dc in steps:
                nr, nc = r + dr, c + dc
                if 0 <= nr < h and 0 <= nc < w and mask[nr, nc] and not seen[nr, nc]:
                    seen[nr, nc] = True
                    q.append((nr, nc))
        out.append(np.asarray(cells, dtype=int))
    return out


def _is_connected(mask: np.ndarray, connectivity: int = 8) -> bool:
    return bool(mask.any()) and len(_components(mask, connectivity=connectivity)) == 1


def _symmetry_ops(mask: np.ndarray) -> list[str]:
    ops = []
    if np.array_equal(mask, mask[:, ::-1]):
        ops.append("mirror_lr")
    if np.array_equal(mask, mask[::-1, :]):
        ops.append("mirror_ud")
    if np.array_equal(mask, mask[::-1, ::-1]):
        ops.append("rotate_180")
    return ops


def _apply_transform(sprite: np.ndarray, op: str) -> np.ndarray:
    if op == "mirror_lr":
        return sprite[:, ::-1].copy()
    if op == "mirror_ud":
        return sprite[::-1, :].copy()
    if op == "rotate_180":
        return sprite[::-1, ::-1].copy()
    raise ValueError(f"unknown operation: {op}")


def _all_transform_results_distinct(sprite: np.ndarray) -> bool:
    transformed = [_apply_transform(sprite, op) for op in OPS]
    for i in range(len(transformed)):
        if np.array_equal(sprite, transformed[i]):
            return False
        for j in range(i + 1, len(transformed)):
            if np.array_equal(transformed[i], transformed[j]):
                return False
    return True


def _extract_glyph_mask(g: np.ndarray) -> np.ndarray:
    grey = g == GREY
    box = _bbox(grey)
    assert box is not None, "expected one grey instruction glyph"
    glyph = _crop_bbox(grey, box)
    assert glyph.any(), "instruction glyph cannot be empty"
    return glyph


def _extract_sprite(g: np.ndarray, bg: int) -> np.ndarray:
    sprite_mask = (g != int(bg)) & (g != GREY)
    box = _bbox(sprite_mask)
    assert box is not None, "expected one colored sprite"
    sprite = _crop_bbox(g, box)
    assert not np.any(sprite == GREY), "sprite crop must not contain the grey glyph"
    colors = set(map(int, sprite[sprite != int(bg)]))
    assert len(colors) >= 2, "sprite must use at least two non-background colors"
    return sprite


def main(grid: np.ndarray) -> np.ndarray:
    """Decode a grey glyph by its unique self-symmetry and transform the colored sprite.

    Input structure:
    - The black background contains one grey binary instruction glyph near a corner.
    - The remaining non-black, non-grey cells form one tight multicolor sprite.
    - The grey glyph has exactly one nontrivial self-symmetry among left-right mirror, up-down mirror, and 180 degree rotation.

    Rule:
    - Crop the grey glyph mask and test which of the three candidate transforms keeps that mask unchanged.
    - Crop the colored sprite to the bounding box of its non-background cells.
    - Apply the glyph-selected transform to that sprite and return only the transformed sprite crop.

    Key insight:
    - The glyph is not a memorized code; its own visible symmetry is the instruction for how the sprite must change.
    """
    g = np.asarray(grid, dtype=int)
    assert g.ndim == 2, "expected a 2D ARC grid"
    assert max(g.shape) <= 30, "ARC grids must be at most 30 by 30"

    bg = _mode(g)
    assert bg == BACKGROUND, "expected a dominant black background"

    glyph = _extract_glyph_mask(g)
    ops = _symmetry_ops(glyph)
    assert len(ops) == 1, "instruction glyph must have exactly one candidate self-symmetry"

    sprite = _extract_sprite(g, bg)
    assert _all_transform_results_distinct(sprite), "sprite must distinguish the three candidate transforms"
    return _apply_transform(sprite, ops[0])


# --------------- generator ---------------


def _next_operation() -> str:
    global _OP_CYCLE
    if not _OP_CYCLE:
        _OP_CYCLE = list(OPS)
        random.shuffle(_OP_CYCLE)
    return _OP_CYCLE.pop()


def _symmetry_orbits(h: int, w: int, op: str) -> list[list[tuple[int, int]]]:
    seen: set[tuple[int, int]] = set()
    orbits: list[list[tuple[int, int]]] = []
    for r in range(h):
        for c in range(w):
            if (r, c) in seen:
                continue
            if op == "mirror_lr":
                cells = {(r, c), (r, w - 1 - c)}
            elif op == "mirror_ud":
                cells = {(r, c), (h - 1 - r, c)}
            else:
                cells = {(r, c), (h - 1 - r, w - 1 - c)}
            seen.update(cells)
            orbits.append(sorted(cells))
    return orbits


def _touches_all_edges(mask: np.ndarray) -> bool:
    return bool(mask[0, :].any() and mask[-1, :].any() and mask[:, 0].any() and mask[:, -1].any())


def _valid_glyph_mask(mask: np.ndarray, op: str) -> bool:
    area = int(mask.size)
    count = int(mask.sum())
    if count < 4 or count > area - 2:
        return False
    if not _touches_all_edges(mask):
        return False
    if not _is_connected(mask, connectivity=8):
        return False
    return _symmetry_ops(mask) == [op]


def _gen_glyph_mask(op: str) -> np.ndarray | None:
    for _ in range(MAX_ATTEMPTS):
        h = random.randint(4, 7)
        w = random.randint(4, 7)
        density = random.uniform(0.36, 0.62)
        mask = np.zeros((h, w), dtype=bool)
        for orbit in _symmetry_orbits(h, w, op):
            if random.random() < density:
                for r, c in orbit:
                    mask[r, c] = True
        if _valid_glyph_mask(mask, op):
            return mask
    return None


def _grow_mask(h: int, w: int, target: int) -> np.ndarray | None:
    mask = np.zeros((h, w), dtype=bool)
    start = (random.randrange(h), random.randrange(w))
    mask[start] = True
    cells = [start]
    while len(cells) < target:
        frontier = []
        for r, c in cells:
            for dr, dc in ((-1, 0), (1, 0), (0, -1), (0, 1)):
                nr, nc = r + dr, c + dc
                if 0 <= nr < h and 0 <= nc < w and not mask[nr, nc]:
                    frontier.append((nr, nc))
        if not frontier:
            return None
        cell = random.choice(frontier)
        mask[cell] = True
        cells.append(cell)
    return mask


def _gen_sprite_patch() -> np.ndarray | None:
    palette = [1, 2, 3, 4, 6, 7, 8, 9]
    for _ in range(MAX_ATTEMPTS):
        h = random.randint(4, 8)
        w = random.randint(4, 8)
        target = random.randint(max(7, (h * w) // 3), max(8, (h * w * 3) // 4))
        mask = _grow_mask(h, w, target)
        if mask is None or not _touches_all_edges(mask) or not _is_connected(mask, connectivity=4):
            continue

        n_colors = random.randint(2, min(4, int(mask.sum()), len(palette)))
        colors = random.sample(palette, n_colors)
        patch = np.full((h, w), BACKGROUND, dtype=int)
        coords = np.argwhere(mask)
        random.shuffle(coords)
        for i, (r, c) in enumerate(coords):
            if i < n_colors:
                patch[int(r), int(c)] = colors[i]
            else:
                patch[int(r), int(c)] = random.choice(colors)

        if len(set(map(int, patch[patch != BACKGROUND]))) < 2:
            continue
        if _all_transform_results_distinct(patch):
            return patch
    return None


def _rect_clear(
    rect: tuple[int, int, int, int],
    other: tuple[int, int, int, int],
    gap: int,
) -> bool:
    r0, c0, r1, c1 = rect
    or0, oc0, or1, oc1 = other
    return r1 + gap < or0 or or1 + gap < r0 or c1 + gap < oc0 or oc1 + gap < c0


def _paste_patch(grid: np.ndarray, patch: np.ndarray, r0: int, c0: int) -> None:
    h, w = patch.shape
    grid[r0 : r0 + h, c0 : c0 + w] = patch


def _place_glyph(grid: np.ndarray, glyph: np.ndarray, corner: str) -> tuple[int, int, int, int]:
    h, w = glyph.shape
    gh, gw = grid.shape
    margin = 1
    r0 = margin if corner[0] == "T" else gh - h - margin
    c0 = margin if corner[1] == "L" else gw - w - margin
    patch = np.full((h, w), BACKGROUND, dtype=int)
    patch[glyph] = GREY
    _paste_patch(grid, patch, r0, c0)
    return r0, c0, r0 + h - 1, c0 + w - 1


def _place_sprite(grid: np.ndarray, sprite: np.ndarray, glyph_box: tuple[int, int, int, int]) -> bool:
    h, w = sprite.shape
    gh, gw = grid.shape
    candidates = []
    for r0 in range(1, gh - h):
        for c0 in range(1, gw - w):
            rect = (r0, c0, r0 + h - 1, c0 + w - 1)
            if _rect_clear(rect, glyph_box, gap=1):
                candidates.append((r0, c0))
    if not candidates:
        return False
    r0, c0 = random.choice(candidates)
    _paste_patch(grid, sprite, r0, c0)
    return True


def _render_input(glyph: np.ndarray, sprite: np.ndarray) -> np.ndarray | None:
    h_min = max(12, glyph.shape[0] + sprite.shape[0] + 4)
    w_min = max(12, glyph.shape[1] + sprite.shape[1] + 4)
    h = random.randint(h_min, min(30, h_min + 8))
    w = random.randint(w_min, min(30, w_min + 8))
    grid = np.full((h, w), BACKGROUND, dtype=int)

    glyph_box = _place_glyph(grid, glyph, random.choice(("TL", "TR", "BL", "BR")))
    if not _place_sprite(grid, sprite, glyph_box):
        return None
    if int(np.count_nonzero(grid == BACKGROUND)) <= grid.size // 2:
        return None
    return grid


def _validate_generated(grid: np.ndarray, expected: np.ndarray, op: str) -> bool:
    if grid.ndim != 2 or expected.ndim != 2:
        return False
    if max(grid.shape) > 30 or max(expected.shape) > 30:
        return False
    glyph = _extract_glyph_mask(grid)
    if _symmetry_ops(glyph) != [op]:
        return False
    sprite = _extract_sprite(grid, BACKGROUND)
    if not _all_transform_results_distinct(sprite):
        return False
    if not np.array_equal(_apply_transform(sprite, op), expected):
        return False
    try:
        actual = main(grid)
    except AssertionError:
        return False
    return np.array_equal(actual, expected)


def _gen_attempt(op: str) -> np.ndarray | None:
    glyph = _gen_glyph_mask(op)
    sprite = _gen_sprite_patch()
    if glyph is None or sprite is None:
        return None
    grid = _render_input(glyph, sprite)
    if grid is None:
        return None
    expected = _apply_transform(sprite, op)
    if _validate_generated(grid, expected, op):
        return grid
    return None


def generate_input() -> np.ndarray:
    op = _next_operation()
    for _ in range(MAX_ATTEMPTS):
        grid = _gen_attempt(op)
        if grid is not None:
            return grid
    raise RuntimeError("failed to generate a valid symmetry-instruction input")
