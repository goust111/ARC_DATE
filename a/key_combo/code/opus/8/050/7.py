from __future__ import annotations

import random
from collections import deque

import numpy as np


BG = 0
FRAME = 1
MARKER = 2
PLACEHOLDER_COLORS = (3, 4)
TARGET_POOL = (5, 6, 7, 8, 9)


def neighbors4(r: int, c: int):
    yield r - 1, c
    yield r + 1, c
    yield r, c - 1
    yield r, c + 1


def bbox_coords(coords: np.ndarray) -> tuple[int, int, int, int]:
    r0, c0 = coords.min(axis=0)
    r1, c1 = coords.max(axis=0)
    return int(r0), int(c0), int(r1), int(c1)


def components_mask(mask: np.ndarray) -> list[np.ndarray]:
    h, w = mask.shape
    seen = np.zeros((h, w), dtype=bool)
    comps: list[np.ndarray] = []
    for r in range(h):
        for c in range(w):
            if not mask[r, c] or seen[r, c]:
                continue
            q = deque([(r, c)])
            seen[r, c] = True
            pts: list[tuple[int, int]] = []
            while q:
                rr, cc = q.popleft()
                pts.append((rr, cc))
                for nr, nc in neighbors4(rr, cc):
                    if 0 <= nr < h and 0 <= nc < w and mask[nr, nc] and not seen[nr, nc]:
                        seen[nr, nc] = True
                        q.append((nr, nc))
            comps.append(np.asarray(pts, dtype=int))
    return comps


def _find_frame(grid: np.ndarray) -> tuple[int, int, int, int]:
    candidates = []
    for comp in components_mask(grid == FRAME):
        r0, c0, r1, c1 = bbox_coords(comp)
        if r1 - r0 < 2 or c1 - c0 < 2:
            continue
        sub = grid[r0 : r1 + 1, c0 : c1 + 1]
        border = np.zeros_like(sub, dtype=bool)
        border[0, :] = True
        border[-1, :] = True
        border[:, 0] = True
        border[:, -1] = True
        if np.all(sub[border] == FRAME) and np.any(sub[~border] != FRAME):
            candidates.append((r0, c0, r1, c1))
    assert len(candidates) == 1, f"expected one frame, found {len(candidates)}"
    return candidates[0]


def _find_legend(
    grid: np.ndarray, outside_mask: np.ndarray, placeholder_colors: set[int]
) -> tuple[dict[int, int], set[tuple[int, int]]]:
    h, w = grid.shape
    hits = []
    for r in range(h - 1):
        for c in range(w - 1):
            cells = {(r, c), (r, c + 1), (r + 1, c), (r + 1, c + 1)}
            if not all(outside_mask[rr, cc] for rr, cc in cells):
                continue
            sub = grid[r : r + 2, c : c + 2]
            if np.any(sub == BG):
                continue
            top = [int(sub[0, 0]), int(sub[0, 1])]
            if set(top) != placeholder_colors:
                continue
            if top[0] == top[1]:
                continue
            bottom = [int(sub[1, 0]), int(sub[1, 1])]
            if bottom[0] in placeholder_colors or bottom[1] in placeholder_colors:
                continue
            if len(set(bottom)) != 2:
                continue
            mapping = {int(sub[0, 0]): int(sub[1, 0]), int(sub[0, 1]): int(sub[1, 1])}
            hits.append((mapping, cells))
    assert len(hits) == 1, f"expected one legend, found {len(hits)}"
    return hits[0]


def _component_coords(grid: np.ndarray, color: int, mask: np.ndarray) -> list[np.ndarray]:
    return components_mask((grid == color) & mask)


def _extract_sprite(
    grid: np.ndarray,
    color: int,
    outside_mask: np.ndarray,
    legend_cells: set[tuple[int, int]],
) -> tuple[np.ndarray, int, int]:
    usable_mask = outside_mask.copy()
    for r, c in legend_cells:
        usable_mask[r, c] = False
    comps = _component_coords(grid, color, usable_mask)
    large = [comp for comp in comps if len(comp) >= 3]
    assert len(large) == 1, f"expected one sprite for color {color}, found {len(large)}"
    comp = large[0]
    r0, c0, r1, c1 = bbox_coords(comp)
    shape = np.zeros((r1 - r0 + 1, c1 - c0 + 1), dtype=bool)
    for r, c in comp:
        shape[int(r) - r0, int(c) - c0] = True
    assert shape[0, 0], "sprite must expose its bbox top-left cell"
    return shape, r1 - r0 + 1, c1 - c0 + 1


def _marker_side(grid: np.ndarray, frame_bbox: tuple[int, int, int, int]) -> str:
    fr0, fc0, fr1, fc1 = frame_bbox
    red = set(map(tuple, np.argwhere(grid == MARKER)))
    assert len(red) in (2, 3), "marker must contain 2 or 3 cells"
    sides = []
    if any(r == fr0 - 1 and fc0 <= c <= fc1 for r, c in red):
        sides.append("top")
    if any(r == fr1 + 1 and fc0 <= c <= fc1 for r, c in red):
        sides.append("bottom")
    if any(c == fc0 - 1 and fr0 <= r <= fr1 for r, c in red):
        sides.append("left")
    if any(c == fc1 + 1 and fr0 <= r <= fr1 for r, c in red):
        sides.append("right")
    assert len(sides) == 1, f"expected one marker side, found {sides}"
    return sides[0]


def _stamp(shape: np.ndarray, color: int) -> np.ndarray:
    out = np.zeros(shape.shape, dtype=int)
    out[shape] = color
    return out


def _paint_sprite(canvas: np.ndarray, anchor: tuple[int, int], sprite: np.ndarray, color: int) -> None:
    ar, ac = anchor
    h, w = sprite.shape
    patch = _stamp(sprite, color)
    canvas[ar : ar + h, ac : ac + w][patch != BG] = color


def _mirror_region_bounds(
    frame_bbox: tuple[int, int, int, int], inner_shape: tuple[int, int], side: str
) -> tuple[slice, slice]:
    fr0, fc0, fr1, fc1 = frame_bbox
    ih, iw = inner_shape
    if side == "left":
        return slice(fr0 + 1, fr1), slice(fc0 - iw, fc0)
    if side == "right":
        return slice(fr0 + 1, fr1), slice(fc1 + 1, fc1 + 1 + iw)
    if side == "top":
        return slice(fr0 - ih, fr0), slice(fc0 + 1, fc1)
    assert side == "bottom"
    return slice(fr1 + 1, fr1 + 1 + ih), slice(fc0 + 1, fc1)


def _solve_from_parts(
    grid: np.ndarray,
    frame_bbox: tuple[int, int, int, int],
    placeholder_positions: dict[int, tuple[int, int]],
    mapping: dict[int, int],
    sprites: dict[int, np.ndarray],
    side: str,
) -> np.ndarray:
    out = grid.copy()
    out[out == MARKER] = BG

    fr0, fc0, fr1, fc1 = frame_bbox
    inner = np.zeros((fr1 - fr0 - 1, fc1 - fc0 - 1), dtype=int)
    for src_color, (r, c) in placeholder_positions.items():
        inner_r = r - (fr0 + 1)
        inner_c = c - (fc0 + 1)
        _paint_sprite(inner, (inner_r, inner_c), sprites[src_color], mapping[src_color])

    out[fr0 + 1 : fr1, fc0 + 1 : fc1] = inner

    mirror = np.fliplr(inner) if side in {"left", "right"} else np.flipud(inner)
    inv_map = {tgt: src for src, tgt in mapping.items()}
    mirrored = mirror.copy()
    for tgt, src in inv_map.items():
        mirrored[mirror == tgt] = src

    rs, cs = _mirror_region_bounds(frame_bbox, inner.shape, side)
    out[rs, cs] = mirrored
    return out


def main(grid: np.ndarray) -> np.ndarray:
    """
    Stamp placeholder-matched sprites inside a frame, then mirror that interior toward the red marker and inverse-recolor the mirrored copy.

    Input structure:
    - A black grid contains one blue rectangular frame with a non-blue interior and a small red marker object touching exactly one outer side.
    - Inside the frame are two single-pixel placeholders of different colors, and outside the frame are two same-colored sprite templates plus a compact two-column color legend.
    - Each legend column is a vertical pair whose top pixel is a placeholder color and whose bottom pixel is that placeholder's stamp color.

    Rule:
    - Find the frame, the two interior placeholder pixels, the outside sprite of each placeholder color, and the legend's forward color mapping.
    - Stamp each matched sprite into the frame so the sprite bbox top-left aligns to the placeholder position, then recolor every stamped pixel with the legend target for that placeholder.
    - Use the red marker side as the mirror direction: reflect the entire stamped frame interior across the frame's outer boundary on that side, place the mirrored copy outside the frame, and apply the inverse legend mapping to every mirrored non-background pixel.
    - Remove the red marker, keep the blue frame, and leave the outside legend and sprite templates unchanged.

    Key insight:
    - The legend is used twice in opposite directions: placeholder colors choose which sprite to stamp and how to recolor it inside the frame, then the mirrored copy converts those stamp colors back through the inverse of the same visible legend.
    """
    g = np.asarray(grid, dtype=int)
    assert g.ndim == 2

    frame_bbox = _find_frame(g)
    fr0, fc0, fr1, fc1 = frame_bbox

    inside = g[fr0 + 1 : fr1, fc0 + 1 : fc1]
    placeholder_positions: dict[int, tuple[int, int]] = {}
    for color in sorted(set(map(int, np.unique(inside))) - {BG}):
        coords = np.argwhere(inside == color)
        assert len(coords) == 1, f"expected a single placeholder for color {color}"
        rr, cc = coords[0]
        placeholder_positions[color] = (fr0 + 1 + int(rr), fc0 + 1 + int(cc))

    placeholder_colors = set(placeholder_positions)
    assert placeholder_colors == set(PLACEHOLDER_COLORS), f"unexpected placeholder colors {placeholder_colors}"

    outside_mask = np.ones(g.shape, dtype=bool)
    outside_mask[fr0 : fr1 + 1, fc0 : fc1 + 1] = False

    mapping, legend_cells = _find_legend(g, outside_mask, placeholder_colors)
    sprites = {color: _extract_sprite(g, color, outside_mask, legend_cells)[0] for color in sorted(placeholder_colors)}
    side = _marker_side(g, frame_bbox)
    return _solve_from_parts(g, frame_bbox, placeholder_positions, mapping, sprites, side)


SPRITE_PATTERNS = (
    np.array([[1, 1], [1, 0]], dtype=bool),
    np.array([[1, 1], [0, 1]], dtype=bool),
    np.array([[1, 0], [1, 1]], dtype=bool),
    np.array([[1, 1, 0], [0, 1, 1]], dtype=bool),
    np.array([[1, 1, 1], [0, 1, 0]], dtype=bool),
    np.array([[1, 1], [1, 0], [1, 0]], dtype=bool),
    np.array([[1, 0], [1, 1], [0, 1]], dtype=bool),
)


def _sprite_cells(anchor: tuple[int, int], sprite: np.ndarray) -> set[tuple[int, int]]:
    ar, ac = anchor
    cells = set()
    for r, c in np.argwhere(sprite):
        cells.add((ar + int(r), ac + int(c)))
    return cells


def _choose_placeholders(ih: int, iw: int, sprites: dict[int, np.ndarray], side: str) -> dict[int, tuple[int, int]] | None:
    options: dict[int, list[tuple[int, int]]] = {}
    for color, sprite in sprites.items():
        sh, sw = sprite.shape
        locs = [(r, c) for r in range(ih - sh + 1) for c in range(iw - sw + 1)]
        random.shuffle(locs)
        options[color] = locs

    for a in options[PLACEHOLDER_COLORS[0]]:
        occ_a = _sprite_cells(a, sprites[PLACEHOLDER_COLORS[0]])
        for b in options[PLACEHOLDER_COLORS[1]]:
            occ_b = _sprite_cells(b, sprites[PLACEHOLDER_COLORS[1]])
            if occ_a & occ_b:
                continue
            inner = np.zeros((ih, iw), dtype=int)
            _paint_sprite(inner, a, sprites[PLACEHOLDER_COLORS[0]], 7)
            _paint_sprite(inner, b, sprites[PLACEHOLDER_COLORS[1]], 8)
            flipped = np.fliplr(inner) if side in {"left", "right"} else np.flipud(inner)
            if np.array_equal(inner, flipped):
                continue
            return {PLACEHOLDER_COLORS[0]: a, PLACEHOLDER_COLORS[1]: b}
    return None


def _can_place_bbox(occupied: np.ndarray, r0: int, c0: int, h: int, w: int, padding: int = 1) -> bool:
    H, W = occupied.shape
    if r0 < 0 or c0 < 0 or r0 + h > H or c0 + w > W:
        return False
    rr0 = max(0, r0 - padding)
    cc0 = max(0, c0 - padding)
    rr1 = min(H, r0 + h + padding)
    cc1 = min(W, c0 + w + padding)
    return not occupied[rr0:rr1, cc0:cc1].any()


def _mark_bbox(occupied: np.ndarray, r0: int, c0: int, h: int, w: int) -> None:
    occupied[r0 : r0 + h, c0 : c0 + w] = True


def _place_patch(grid: np.ndarray, top_left: tuple[int, int], patch: np.ndarray) -> None:
    r0, c0 = top_left
    h, w = patch.shape
    sub = grid[r0 : r0 + h, c0 : c0 + w]
    sub[patch != BG] = patch[patch != BG]


def _random_patch_positions(
    occupied: np.ndarray, patch_shape: tuple[int, int], tries: int = 200
) -> tuple[int, int] | None:
    H, W = occupied.shape
    h, w = patch_shape
    coords = [(r, c) for r in range(H - h + 1) for c in range(W - w + 1)]
    random.shuffle(coords)
    for r, c in coords[:tries]:
        if _can_place_bbox(occupied, r, c, h, w, padding=1):
            return r, c
    return None


def _marker_cells(frame_bbox: tuple[int, int, int, int], side: str, length: int) -> list[tuple[int, int]]:
    fr0, fc0, fr1, fc1 = frame_bbox
    if side in {"left", "right"}:
        start = random.randint(fr0 + 1, fr1 - length)
        col = fc0 - 1 if side == "left" else fc1 + 1
        return [(start + i, col) for i in range(length)]
    start = random.randint(fc0 + 1, fc1 - length)
    row = fr0 - 1 if side == "top" else fr1 + 1
    return [(row, start + i) for i in range(length)]


def _build_example() -> tuple[np.ndarray, np.ndarray]:
    ih = random.randint(5, 6)
    iw = random.randint(5, 6)
    fh, fw = ih + 2, iw + 2
    side = random.choice(("left", "right", "top", "bottom"))

    sprites = {
        3: SPRITE_PATTERNS[random.randrange(len(SPRITE_PATTERNS))].copy(),
        4: SPRITE_PATTERNS[random.randrange(len(SPRITE_PATTERNS))].copy(),
    }
    anchors = _choose_placeholders(ih, iw, sprites, side)
    if anchors is None:
        raise ValueError("failed to place sprites inside frame")

    tgt3, tgt4 = random.sample(TARGET_POOL, 2)
    mapping = {3: tgt3, 4: tgt4}

    H = random.randint(max(14, fh + ih + 4), min(22, 30))
    W = random.randint(max(14, fw + iw + 4), min(22, 30))

    if side == "left":
        fr0 = random.randint(2, H - fh - 2)
        fc0 = random.randint(iw + 1, W - fw - 2)
    elif side == "right":
        fr0 = random.randint(2, H - fh - 2)
        fc0 = random.randint(2, W - fw - iw - 1)
    elif side == "top":
        fr0 = random.randint(ih + 1, H - fh - 2)
        fc0 = random.randint(2, W - fw - 2)
    else:
        fr0 = random.randint(2, H - fh - ih - 1)
        fc0 = random.randint(2, W - fw - 2)
    frame_bbox = (fr0, fc0, fr0 + fh - 1, fc0 + fw - 1)

    grid = np.zeros((H, W), dtype=int)
    occupied = np.zeros((H, W), dtype=bool)

    fr0, fc0, fr1, fc1 = frame_bbox
    grid[fr0, fc0 : fc1 + 1] = FRAME
    grid[fr1, fc0 : fc1 + 1] = FRAME
    grid[fr0 : fr1 + 1, fc0] = FRAME
    grid[fr0 : fr1 + 1, fc1] = FRAME
    _mark_bbox(occupied, fr0, fc0, fh, fw)

    rs, cs = _mirror_region_bounds(frame_bbox, (ih, iw), side)
    _mark_bbox(occupied, rs.start, cs.start, rs.stop - rs.start, cs.stop - cs.start)

    for color, (r, c) in anchors.items():
        grid[fr0 + 1 + r, fc0 + 1 + c] = color

    legend_cols = [3, 4]
    if random.random() < 0.5:
        legend_cols.reverse()
    legend = np.array([[legend_cols[0], legend_cols[1]], [mapping[legend_cols[0]], mapping[legend_cols[1]]]], dtype=int)
    legend_pos = _random_patch_positions(occupied, legend.shape)
    if legend_pos is None:
        raise ValueError("failed to place legend")
    _place_patch(grid, legend_pos, legend)
    _mark_bbox(occupied, legend_pos[0], legend_pos[1], 2, 2)

    for color in PLACEHOLDER_COLORS:
        sprite_patch = np.zeros(sprites[color].shape, dtype=int)
        sprite_patch[sprites[color]] = color
        pos = _random_patch_positions(occupied, sprite_patch.shape)
        if pos is None:
            raise ValueError("failed to place sprite")
        _place_patch(grid, pos, sprite_patch)
        _mark_bbox(occupied, pos[0], pos[1], sprite_patch.shape[0], sprite_patch.shape[1])

    for r, c in _marker_cells(frame_bbox, side, random.choice((2, 3))):
        grid[r, c] = MARKER

    placeholder_positions = {color: (fr0 + 1 + rc[0], fc0 + 1 + rc[1]) for color, rc in anchors.items()}
    out = _solve_from_parts(grid, frame_bbox, placeholder_positions, mapping, sprites, side)
    return grid, out


def _semantic_checks(inp: np.ndarray, out: np.ndarray) -> None:
    calc = main(inp)
    assert np.array_equal(calc, out), "solver/output mismatch"

    frame_bbox = _find_frame(inp)
    fr0, fc0, fr1, fc1 = frame_bbox
    side = _marker_side(inp, frame_bbox)
    inside_out = out[fr0 + 1 : fr1, fc0 + 1 : fc1]
    mirror_out = out[_mirror_region_bounds(frame_bbox, inside_out.shape, side)]

    placeholder_colors = set(map(int, np.unique(inp[fr0 + 1 : fr1, fc0 + 1 : fc1]))) - {BG}
    outside_mask = np.ones(inp.shape, dtype=bool)
    outside_mask[fr0 : fr1 + 1, fc0 : fc1 + 1] = False
    mapping, _legend_cells = _find_legend(inp, outside_mask, placeholder_colors)
    assert len(mapping) == 2
    assert all(np.any(inside_out == tgt) for tgt in mapping.values()), "both forward legend colors must appear in-frame"
    assert all(np.any(mirror_out == src) for src in mapping), "both placeholder colors must appear in mirrored copy"

    if side in {"left", "right"}:
        plain = np.fliplr(inside_out)
    else:
        plain = np.flipud(inside_out)
    assert not np.array_equal(plain, mirror_out), "inverse mapping step became unnecessary"

    markerless = inp.copy()
    markerless[markerless == MARKER] = BG
    assert not np.array_equal(markerless, out), "mirror/stamp steps must change the grid"

    naive = out.copy()
    naive[_mirror_region_bounds(frame_bbox, inside_out.shape, side)] = plain
    assert not np.array_equal(naive, out), "mirror recoloring must matter"


def generate_input() -> np.ndarray:
    for _ in range(400):
        try:
            inp, out = _build_example()
            _semantic_checks(inp, out)
            return inp
        except Exception:
            continue
    raise RuntimeError("failed to generate a valid puzzle instance")
