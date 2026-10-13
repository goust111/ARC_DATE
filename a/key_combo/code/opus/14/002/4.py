from __future__ import annotations

import random
from collections import deque

import numpy as np


BLUE = 1
MARKER_COLORS = (2, 3, 4, 6, 7)
MAX_ATTEMPTS = 400


def _neighbors4(r: int, c: int):
    yield r - 1, c
    yield r + 1, c
    yield r, c - 1
    yield r, c + 1


def _bbox(mask: np.ndarray) -> tuple[int, int, int, int] | None:
    coords = np.argwhere(mask)
    if coords.size == 0:
        return None
    r0, c0 = coords.min(axis=0)
    r1, c1 = coords.max(axis=0)
    return int(r0), int(c0), int(r1), int(c1)


def _crop(grid: np.ndarray, box: tuple[int, int, int, int]) -> np.ndarray:
    r0, c0, r1, c1 = box
    return grid[r0 : r1 + 1, c0 : c1 + 1].copy()


def _paste(dst: np.ndarray, patch: np.ndarray, r0: int, c0: int) -> None:
    h, w = patch.shape
    dst[r0 : r0 + h, c0 : c0 + w] = patch


def _components(mask: np.ndarray) -> list[list[tuple[int, int]]]:
    h, w = mask.shape
    seen = np.zeros((h, w), dtype=bool)
    comps: list[list[tuple[int, int]]] = []
    for r in range(h):
        for c in range(w):
            if not mask[r, c] or seen[r, c]:
                continue
            comp: list[tuple[int, int]] = []
            q = deque([(r, c)])
            seen[r, c] = True
            while q:
                rr, cc = q.popleft()
                comp.append((rr, cc))
                for nr, nc in _neighbors4(rr, cc):
                    if 0 <= nr < h and 0 <= nc < w and mask[nr, nc] and not seen[nr, nc]:
                        seen[nr, nc] = True
                        q.append((nr, nc))
            comps.append(comp)
    return comps


def _enclosed_hole_mask(sprite: np.ndarray) -> np.ndarray:
    hole_mask = np.zeros_like(sprite, dtype=bool)
    for comp in _components(sprite == 0):
        if any(r == 0 or c == 0 or r == sprite.shape[0] - 1 or c == sprite.shape[1] - 1 for r, c in comp):
            continue
        for r, c in comp:
            hole_mask[r, c] = True
    return hole_mask


def _transform(sprite: np.ndarray, quadrant: str) -> np.ndarray:
    if quadrant == "tr":
        return np.fliplr(sprite)
    if quadrant == "bl":
        return np.flipud(sprite)
    if quadrant == "br":
        return np.flipud(np.fliplr(sprite))
    raise ValueError(f"unknown quadrant {quadrant}")


def _quadrant_origin(quadrant: str, qh: int, qw: int) -> tuple[int, int]:
    if quadrant == "tr":
        return 0, qw
    if quadrant == "bl":
        return qh, 0
    if quadrant == "br":
        return qh, qw
    raise ValueError(f"unknown quadrant {quadrant}")


def _target_local_top_left(src_r0: int, src_c0: int, h: int, w: int, qh: int, qw: int, quadrant: str) -> tuple[int, int]:
    if quadrant == "tr":
        return src_r0, qw - w - src_c0
    if quadrant == "bl":
        return qh - h - src_r0, src_c0
    if quadrant == "br":
        return qh - h - src_r0, qw - w - src_c0
    raise ValueError(f"unknown quadrant {quadrant}")


def main(grid: np.ndarray) -> np.ndarray:
    """
    Mirror the top-left holey sprite into the other quadrants and color matched holes from local marker positions.

    Input structure:
    - The grid has a black background and even height/width, so it splits into four equal quadrants.
    - The top-left quadrant contains one blue sprite; black cells enclosed inside its tight bounding box are holes.
    - The other three quadrants contain only scattered single-pixel color markers.

    Rule:
    - Tight-crop the blue top-left sprite while preserving its internal holes.
    - Place mirrored copies into the other quadrants: horizontal mirror in the top-right, vertical mirror in the bottom-left, and 180-degree rotation in the bottom-right.
    - Each copy's bounding box is positioned as the quadrant-symmetric counterpart of the source bounding box.
    - For every marker, use its quadrant-local coordinate to find the hole at that same local coordinate inside that quadrant's mirrored copy; fill that hole with the marker's color, then erase the marker itself.
    - Any hole without a matching marker stays black, and the original top-left sprite is unchanged.

    Key insight:
    - The marker coordinates only make sense after inferring the quadrant-specific mirror transform, because the holes move to different local positions in each copied sprite.
    """
    g = np.asarray(grid, dtype=int)
    h, w = g.shape
    assert g.ndim == 2 and h % 2 == 0 and w % 2 == 0
    qh, qw = h // 2, w // 2

    tl = g[:qh, :qw]
    source_box = _bbox(tl == BLUE)
    assert source_box is not None, "top-left quadrant must contain a blue sprite"
    src_r0, src_c0, src_r1, src_c1 = source_box
    sprite = _crop(tl, source_box)

    out = np.zeros_like(g)
    out[:qh, :qw] = tl

    for quadrant in ("tr", "bl", "br"):
        sprite_t = _transform(sprite, quadrant)
        local_r0, local_c0 = _target_local_top_left(src_r0, src_c0, sprite_t.shape[0], sprite_t.shape[1], qh, qw, quadrant)
        q_r0, q_c0 = _quadrant_origin(quadrant, qh, qw)
        _paste(out, sprite_t, q_r0 + local_r0, q_c0 + local_c0)

        hole_mask = _enclosed_hole_mask(sprite_t)
        quadrant_view = g[q_r0 : q_r0 + qh, q_c0 : q_c0 + qw]
        for mr, mc in np.argwhere((quadrant_view != 0) & (quadrant_view != BLUE)):
            rr = int(mr) - local_r0
            cc = int(mc) - local_c0
            assert 0 <= rr < sprite_t.shape[0] and 0 <= cc < sprite_t.shape[1], "marker must point inside the mirrored sprite"
            assert hole_mask[rr, cc], "marker must point to a mirrored hole"
            out[q_r0 + int(mr), q_c0 + int(mc)] = int(quadrant_view[mr, mc])

    return out


def _sprite_is_valid(sprite: np.ndarray, hole_count: int) -> bool:
    blue_mask = sprite == BLUE
    box = _bbox(blue_mask)
    if box != (0, 0, sprite.shape[0] - 1, sprite.shape[1] - 1):
        return False
    comps = _components(blue_mask)
    if len(comps) != 1:
        return False
    hole_mask = _enclosed_hole_mask(sprite)
    hole_comps = _components(hole_mask)
    if len(hole_comps) != hole_count:
        return False
    if any(len(comp) != 1 for comp in hole_comps):
        return False
    blue_pixels = int(blue_mask.sum())
    return 8 <= blue_pixels <= 20


def _choose_holes(height: int, width: int, hole_count: int) -> list[tuple[int, int]] | None:
    interior = [(r, c) for r in range(1, height - 1) for c in range(1, width - 1)]
    random.shuffle(interior)
    holes: list[tuple[int, int]] = []
    for r, c in interior:
        if all(abs(r - rr) + abs(c - cc) > 1 for rr, cc in holes):
            holes.append((r, c))
            if len(holes) == hole_count:
                return holes
    return None


def _generate_sprite() -> np.ndarray | None:
    for _ in range(120):
        height, width = random.choice([(4, 4), (4, 5), (5, 4), (5, 5)])
        hole_count = random.choice([2, 3])
        holes = _choose_holes(height, width, hole_count)
        if holes is None:
            continue

        sprite = np.full((height, width), BLUE, dtype=int)
        for r, c in holes:
            sprite[r, c] = 0

        for _ in range(random.randint(0, 4)):
            blue_cells = [(r, c) for r, c in np.argwhere(sprite == BLUE)]
            random.shuffle(blue_cells)
            changed = False
            for r, c in blue_cells:
                candidate = sprite.copy()
                candidate[r, c] = 0
                if _sprite_is_valid(candidate, hole_count):
                    sprite = candidate
                    changed = True
                    break
            if not changed:
                break

        if _sprite_is_valid(sprite, hole_count):
            return sprite
    return None


def _build_expected(qh: int, qw: int, src_r0: int, src_c0: int, sprite: np.ndarray, fills: dict[str, list[tuple[tuple[int, int], int]]]) -> np.ndarray:
    out = np.zeros((2 * qh, 2 * qw), dtype=int)
    _paste(out, sprite, src_r0, src_c0)
    for quadrant in ("tr", "bl", "br"):
        sprite_t = _transform(sprite, quadrant)
        local_r0, local_c0 = _target_local_top_left(src_r0, src_c0, sprite_t.shape[0], sprite_t.shape[1], qh, qw, quadrant)
        q_r0, q_c0 = _quadrant_origin(quadrant, qh, qw)
        _paste(out, sprite_t, q_r0 + local_r0, q_c0 + local_c0)
        for (mr, mc), color in fills[quadrant]:
            out[q_r0 + mr, q_c0 + mc] = color
    return out


def _gen_attempt() -> np.ndarray | None:
    sprite = _generate_sprite()
    if sprite is None:
        return None

    sh, sw = sprite.shape
    qh = random.randint(max(6, sh), 10)
    qw = random.randint(max(6, sw), 10)
    src_r0 = random.randint(0, qh - sh)
    src_c0 = random.randint(0, qw - sw)

    transformed_holes: dict[str, list[tuple[int, int]]] = {}
    for quadrant in ("tr", "bl", "br"):
        sprite_t = _transform(sprite, quadrant)
        local_r0, local_c0 = _target_local_top_left(src_r0, src_c0, sprite_t.shape[0], sprite_t.shape[1], qh, qw, quadrant)
        hole_mask = _enclosed_hole_mask(sprite_t)
        transformed_holes[quadrant] = [(local_r0 + r, local_c0 + c) for r, c in np.argwhere(hole_mask)]
        if len(transformed_holes[quadrant]) < 2:
            return None

    hole_count = len(transformed_holes["tr"])
    marker_counts = {quadrant: random.randint(1, min(2, hole_count)) for quadrant in ("tr", "bl", "br")}
    if all(count == hole_count for count in marker_counts.values()):
        marker_counts[random.choice(["tr", "bl", "br"])] = 1

    fills: dict[str, list[tuple[tuple[int, int], int]]] = {}
    inp = np.zeros((2 * qh, 2 * qw), dtype=int)
    _paste(inp, sprite, src_r0, src_c0)

    for quadrant in ("tr", "bl", "br"):
        markers = random.sample(transformed_holes[quadrant], marker_counts[quadrant])
        colors = random.sample(MARKER_COLORS, marker_counts[quadrant])
        fills[quadrant] = list(zip(markers, colors))
        q_r0, q_c0 = _quadrant_origin(quadrant, qh, qw)
        for (mr, mc), color in fills[quadrant]:
            inp[q_r0 + mr, q_c0 + mc] = color

    expected = _build_expected(qh, qw, src_r0, src_c0, sprite, fills)
    if not np.array_equal(main(inp), expected):
        return None
    return inp


def generate_input() -> np.ndarray:
    for _ in range(MAX_ATTEMPTS):
        grid = _gen_attempt()
        if grid is not None:
            return grid
    raise RuntimeError("failed to generate valid input")
