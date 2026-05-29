# Combo variant of dbff022c (index 29)
# combo_spec: D8/1+D12/2
# Name: 凸包填充 + T/L形调色板

import random

import numpy as np

from arc_utils import components_mask


# Design analysis for the fused variant
# ------------------------------------
# a) Behavior trace:
#    The original task finds a rectangular 2xN / Nx2 palette and fills only truly enclosed
#    background holes. In this fused version the palette is a connected T or L shape on the
#    border, with the edge-aligned branch used for small sprites and the inward branch used for
#    large sprites. Filling no longer follows flood-filled holes; it uses the convex hull of each
#    sprite's border pixels. A concave or open "C/U" border therefore becomes a larger filled
#    convex region in the output.
# b) Complexity calibration:
#    The palette keeps only two semantic classes, small and large, so the rule is visible but not
#    combinatorially noisy. The same border color is intentionally reused in both branches with
#    different fill colors; this prevents the task from degenerating into a single flat color map.
# c) Hidden assumptions replaced:
#    Old code assumed a rectangular palette and closed holes. New code assumes instead:
#    - exactly one connected non-background border-touching component is the palette;
#    - top/bottom palettes are T-shaped and left/right palettes are L-shaped;
#    - sprite size is inferred from bbox area, with the generator keeping small <= 16 and
#      large >= 25 so there is no ambiguous middle band;
#    - sprites stay away from the frame and from one another, so convex-hull filling cannot
#      accidentally absorb unrelated objects.
# d) Placement legality:
#    The generator places the palette first, then sprites with a 1-cell moat. Sprites are kept
#    strictly inside the frame, and convex hulls stay inside each sprite bbox, so no overlap or
#    out-of-bounds fill can occur.
# e) Visual observability:
#    Every sample contains a small sprite and a large sprite sharing the same border color but
#    receiving different fills, so branch semantics are visible. At least one keyed sprite is a
#    concave/open shape whose convex hull adds cells beyond any ordinary enclosed hole fill.
# f) Visual cleanliness:
#    Only three sprites are used, all monochrome and separated. The palette is the lone
#    boundary-touching component, which keeps object roles legible.
# g) Inferability of change:
#    The color change is always justified by two visible cues: palette branch membership and
#    sprite size class. The generator fixes those cues across all samples instead of relying on
#    numeric color ids or hidden ordering.


_GRID_H_MIN, _GRID_H_MAX = 10, 13
_GRID_W_MIN, _GRID_W_MAX = 11, 15
_MAX_RETRIES = 1200
_SMALL_MAX_AREA = 16
_LARGE_MIN_AREA = 25


def _mask(*rows: str) -> np.ndarray:
    return np.array([[ch == "1" for ch in row] for row in rows], dtype=bool)


_SMALL_SHAPES = [
    _mask(
        "1111",
        "1000",
        "1111",
    ),
    _mask(
        "1001",
        "1001",
        "1001",
        "1111",
    ),
    _mask(
        "1111",
        "1000",
        "1001",
        "1111",
    ),
]

_LARGE_SHAPES = [
    _mask(
        "111111",
        "100000",
        "101111",
        "100000",
        "111111",
    ),
    _mask(
        "100001",
        "100001",
        "100001",
        "100001",
        "100001",
        "111111",
    ),
    _mask(
        "11111",
        "10000",
        "10111",
        "10001",
        "11111",
    ),
]


def _bg_boundary(g: np.ndarray) -> int:
    border = np.concatenate([g[0, :], g[-1, :], g[1:-1, 0], g[1:-1, -1]])
    vals, cnts = np.unique(border, return_counts=True)
    return int(vals[cnts.argmax()])


def _bbox_area(coords: np.ndarray) -> int:
    r0, c0 = coords.min(axis=0)
    r1, c1 = coords.max(axis=0)
    return int((r1 - r0 + 1) * (c1 - c0 + 1))


def _cross(o: tuple[float, float], a: tuple[float, float], b: tuple[float, float]) -> float:
    return (a[0] - o[0]) * (b[1] - o[1]) - (a[1] - o[1]) * (b[0] - o[0])


def _convex_hull(points: list[tuple[float, float]]) -> list[tuple[float, float]]:
    pts = sorted(set(points))
    if len(pts) <= 1:
        return pts

    lower: list[tuple[float, float]] = []
    for p in pts:
        while len(lower) >= 2 and _cross(lower[-2], lower[-1], p) <= 0:
            lower.pop()
        lower.append(p)

    upper: list[tuple[float, float]] = []
    for p in reversed(pts):
        while len(upper) >= 2 and _cross(upper[-2], upper[-1], p) <= 0:
            upper.pop()
        upper.append(p)

    return lower[:-1] + upper[:-1]


def _inside_convex_polygon(p: tuple[float, float], poly: list[tuple[float, float]]) -> bool:
    if len(poly) < 3:
        return False
    sign = 0
    for i, a in enumerate(poly):
        b = poly[(i + 1) % len(poly)]
        cp = _cross(a, b, p)
        if abs(cp) < 1e-9:
            continue
        cur = 1 if cp > 0 else -1
        if sign == 0:
            sign = cur
        elif cur != sign:
            return False
    return True


def _convex_fill_mask_from_coords(coords: np.ndarray, shape: tuple[int, int]) -> np.ndarray:
    pts = [(float(c), float(r)) for r, c in coords.tolist()]
    hull = _convex_hull(pts)
    out = np.zeros(shape, dtype=bool)
    if len(hull) < 3:
        return out

    r0, c0 = coords.min(axis=0)
    r1, c1 = coords.max(axis=0)
    sprite_mask = np.zeros(shape, dtype=bool)
    sprite_mask[coords[:, 0], coords[:, 1]] = True

    for r in range(int(r0), int(r1) + 1):
        for c in range(int(c0), int(c1) + 1):
            if sprite_mask[r, c]:
                continue
            if _inside_convex_polygon((float(c), float(r)), hull):
                out[r, c] = True
    return out


def _convex_fill_mask_local(mask: np.ndarray) -> np.ndarray:
    coords = np.argwhere(mask)
    return _convex_fill_mask_from_coords(coords, mask.shape)


def _palette_component(g: np.ndarray, bg: int) -> np.ndarray:
    mask = g != bg
    comps = components_mask(mask, connectivity=4)
    boundary = []
    h, w = g.shape
    for pts in comps:
        if any(r == 0 or c == 0 or r == h - 1 or c == w - 1 for r, c in pts):
            boundary.append(pts)
    assert len(boundary) == 1, "_palette_component expects exactly one boundary-touching component"
    pal_mask = np.zeros_like(g, dtype=bool)
    pal_mask[boundary[0][:, 0], boundary[0][:, 1]] = True
    return pal_mask


def _parse_top_t(g: np.ndarray, pal_mask: np.ndarray) -> tuple[dict[int, int], dict[int, int]]:
    cols = np.where(pal_mask[0])[0]
    assert cols.size >= 3
    c0, c1 = int(cols.min()), int(cols.max()) + 1
    assert np.all(pal_mask[0:2, c0:c1])
    assert np.sum(pal_mask[0:2]) == 2 * (c1 - c0)

    tail_pts = np.argwhere(pal_mask[2:, :])
    assert tail_pts.size > 0
    tail_pts[:, 0] += 2
    tail_rows = sorted(set(int(r) for r, _ in tail_pts))
    tail_cols = sorted(set(int(c) for _, c in tail_pts))
    assert tail_rows == list(range(2, tail_rows[-1] + 1))
    assert len(tail_cols) == 2 and tail_cols[1] == tail_cols[0] + 1
    for r in tail_rows:
        assert pal_mask[r, tail_cols[0]] and pal_mask[r, tail_cols[1]]
    expected = np.zeros_like(pal_mask)
    expected[0:2, c0:c1] = True
    expected[2 : tail_rows[-1] + 1, tail_cols[0] : tail_cols[1] + 1] = True
    assert np.array_equal(expected, pal_mask)

    small = {int(g[0, c]): int(g[1, c]) for c in range(c0, c1)}
    large = {int(g[r, tail_cols[0]]): int(g[r, tail_cols[1]]) for r in tail_rows}
    return small, large


def _parse_bottom_t(g: np.ndarray, pal_mask: np.ndarray) -> tuple[dict[int, int], dict[int, int]]:
    h = g.shape[0]
    cols = np.where(pal_mask[h - 1])[0]
    assert cols.size >= 3
    c0, c1 = int(cols.min()), int(cols.max()) + 1
    assert np.all(pal_mask[h - 2 : h, c0:c1])
    assert np.sum(pal_mask[h - 2 : h]) == 2 * (c1 - c0)

    tail_pts = np.argwhere(pal_mask[: h - 2, :])
    assert tail_pts.size > 0
    tail_rows = sorted(set(int(r) for r, _ in tail_pts))
    tail_cols = sorted(set(int(c) for _, c in tail_pts))
    assert tail_rows == list(range(tail_rows[0], h - 2))
    assert len(tail_cols) == 2 and tail_cols[1] == tail_cols[0] + 1
    for r in tail_rows:
        assert pal_mask[r, tail_cols[0]] and pal_mask[r, tail_cols[1]]
    expected = np.zeros_like(pal_mask)
    expected[h - 2 : h, c0:c1] = True
    expected[tail_rows[0] : h - 2, tail_cols[0] : tail_cols[1] + 1] = True
    assert np.array_equal(expected, pal_mask)

    small = {int(g[h - 1, c]): int(g[h - 2, c]) for c in range(c0, c1)}
    large = {int(g[r, tail_cols[0]]): int(g[r, tail_cols[1]]) for r in tail_rows}
    return small, large


def _parse_left_l(g: np.ndarray, pal_mask: np.ndarray) -> tuple[dict[int, int], dict[int, int]]:
    rows = np.where(pal_mask[:, 0])[0]
    assert rows.size >= 3
    r0, r1 = int(rows.min()), int(rows.max()) + 1
    assert np.all(pal_mask[r0:r1, 0:2])
    assert np.sum(pal_mask[:, 0:2]) == 2 * (r1 - r0)

    foot_pts = np.argwhere(pal_mask[:, 2:])
    assert foot_pts.size > 0
    foot_pts[:, 1] += 2
    foot_rows = sorted(set(int(r) for r, _ in foot_pts))
    foot_cols = sorted(set(int(c) for _, c in foot_pts))
    assert len(foot_rows) == 2 and foot_rows[1] == foot_rows[0] + 1
    assert foot_rows == [r1 - 2, r1 - 1]
    assert foot_cols == list(range(2, foot_cols[-1] + 1))
    for c in foot_cols:
        assert pal_mask[foot_rows[0], c] and pal_mask[foot_rows[1], c]
    expected = np.zeros_like(pal_mask)
    expected[r0:r1, 0:2] = True
    expected[foot_rows[0] : foot_rows[1] + 1, 2 : foot_cols[-1] + 1] = True
    assert np.array_equal(expected, pal_mask)

    small = {int(g[r, 0]): int(g[r, 1]) for r in range(r0, r1)}
    large = {int(g[foot_rows[0], c]): int(g[foot_rows[1], c]) for c in foot_cols}
    return small, large


def _parse_right_l(g: np.ndarray, pal_mask: np.ndarray) -> tuple[dict[int, int], dict[int, int]]:
    h, w = g.shape
    rows = np.where(pal_mask[:, w - 1])[0]
    assert rows.size >= 3
    r0, r1 = int(rows.min()), int(rows.max()) + 1
    assert np.all(pal_mask[r0:r1, w - 2 : w])
    assert np.sum(pal_mask[:, w - 2 : w]) == 2 * (r1 - r0)

    foot_pts = np.argwhere(pal_mask[:, : w - 2])
    assert foot_pts.size > 0
    foot_rows = sorted(set(int(r) for r, _ in foot_pts))
    foot_cols = sorted(set(int(c) for _, c in foot_pts))
    assert len(foot_rows) == 2 and foot_rows[1] == foot_rows[0] + 1
    assert foot_rows == [r1 - 2, r1 - 1]
    assert foot_cols == list(range(foot_cols[0], w - 2))
    for c in foot_cols:
        assert pal_mask[foot_rows[0], c] and pal_mask[foot_rows[1], c]
    expected = np.zeros_like(pal_mask)
    expected[r0:r1, w - 2 : w] = True
    expected[foot_rows[0] : foot_rows[1] + 1, foot_cols[0] : w - 2] = True
    assert np.array_equal(expected, pal_mask)

    small = {int(g[r, w - 1]): int(g[r, w - 2]) for r in range(r0, r1)}
    large = {int(g[foot_rows[0], c]): int(g[foot_rows[1], c]) for c in foot_cols}
    return small, large


def _palette_mappings(g: np.ndarray, bg: int) -> tuple[dict[str, dict[int, int]], np.ndarray]:
    pal_mask = _palette_component(g, bg)
    h, w = g.shape
    touches = {
        "top": bool(pal_mask[0].any()),
        "bottom": bool(pal_mask[h - 1].any()),
        "left": bool(pal_mask[:, 0].any()),
        "right": bool(pal_mask[:, w - 1].any()),
    }
    side_count = sum(touches.values())
    assert side_count == 1, "_palette_mappings expects a palette attached to exactly one side"

    if touches["top"]:
        small, large = _parse_top_t(g, pal_mask)
    elif touches["bottom"]:
        small, large = _parse_bottom_t(g, pal_mask)
    elif touches["left"]:
        small, large = _parse_left_l(g, pal_mask)
    else:
        small, large = _parse_right_l(g, pal_mask)

    return {"small": small, "large": large}, pal_mask


def _size_class(coords: np.ndarray) -> str:
    area = _bbox_area(coords)
    return "small" if area <= _SMALL_MAX_AREA else "large"


def _fill_sprite_hulls(out: np.ndarray, bg: int, mappings: dict[str, dict[int, int]], pal_mask: np.ndarray) -> None:
    sprite_mask = (out != bg) & (~pal_mask)
    comps = components_mask(sprite_mask, connectivity=4)
    for pts in comps:
        color = int(out[tuple(pts[0])])
        branch = _size_class(pts)
        mapping = mappings[branch]
        if color not in mapping:
            continue
        fill_color = int(mapping[color])
        fill_mask = _convex_fill_mask_from_coords(pts, out.shape)
        out[fill_mask & (out == bg) & (~pal_mask)] = fill_color


def main(grid: np.ndarray) -> np.ndarray:
    """
    Fill qualifying hollow or concave sprites using a border legend whose branches encode size classes.

    Input structure:
    - Background is 0. One connected palette touches exactly one grid edge and forms a T or L shape instead of a plain strip.
    - The branch that lies flush with the outer edge lists mappings for small sprites; the inward branch lists mappings for large sprites.
    - Sprite borders are isolated monochrome components away from the frame, and the same border color may appear in both palette branches with different fill colors.

    Rule:
    - Parse the T/L palette into two border->fill mappings, one for small sprites and one for large sprites.
    - For each non-palette sprite, choose the branch by its bbox size class rather than by the border color alone.
    - If the sprite color exists in that branch, fill every background cell inside the convex hull of the sprite's border pixels with the mapped fill color.

    Key insight:
    - The palette is not a symmetric rectangle anymore: branch location carries meaning, and the same key color can map differently in the two branches.
    - The filled region is not the true hole of the sprite; it is the convex hull interior of the border pixels, so concave or open borders get "completed" into a larger convex fill.
    """
    g = np.asarray(grid, dtype=int)
    bg = _bg_boundary(g)
    mappings, pal_mask = _palette_mappings(g, bg)
    out = g.copy()
    _fill_sprite_hulls(out, bg, mappings, pal_mask)
    return out


def _build_t_palette(side: str, small_keys: list[int], small_fills: list[int], large_keys: list[int], large_fills: list[int], stem_col: int) -> np.ndarray:
    assert side in {"top", "bottom"}
    assert len(small_keys) == 3 and len(small_fills) == 3
    assert len(large_keys) == 2 and len(large_fills) == 2
    pal = np.zeros((4, 3), dtype=int)
    if side == "top":
        pal[0, :] = small_keys
        pal[1, :] = small_fills
    else:
        pal[2, :] = small_fills
        pal[3, :] = small_keys
    for i in range(2):
        pal[2 + i if side == "top" else i, stem_col] = large_keys[i]
        pal[2 + i if side == "top" else i, stem_col + 1] = large_fills[i]
    return pal


def _build_l_palette(side: str, small_keys: list[int], small_fills: list[int], large_keys: list[int], large_fills: list[int]) -> np.ndarray:
    assert side in {"left", "right"}
    assert len(small_keys) == 3 and len(small_fills) == 3
    assert len(large_keys) == 2 and len(large_fills) == 2
    pal = np.zeros((3, 4), dtype=int)
    if side == "left":
        pal[:, 0] = small_keys
        pal[:, 1] = small_fills
        pal[1, 2:4] = large_keys
        pal[2, 2:4] = large_fills
    else:
        pal[:, 2] = small_fills
        pal[:, 3] = small_keys
        pal[1, 0] = large_keys[0]
        pal[1, 1] = large_keys[1]
        pal[2, 0] = large_fills[0]
        pal[2, 1] = large_fills[1]
    return pal


def _choose_palette_colors(rng: random.Random) -> tuple[list[int], list[int], list[int], list[int], int, int]:
    colors = list(range(1, 10))
    rng.shuffle(colors)
    shared_key = colors[0]
    small_only = colors[1:3]
    large_only = [colors[3]]
    fills = colors[4:9]
    small_keys = [shared_key, small_only[0], small_only[1]]
    large_keys = [shared_key, large_only[0]]
    small_fills = fills[:3]
    large_fills = fills[3:]
    rng.shuffle(small_keys)
    rng.shuffle(large_keys)
    rng.shuffle(small_fills)
    rng.shuffle(large_fills)
    small_map = dict(zip(small_keys, small_fills))
    large_map = dict(zip(large_keys, large_fills))
    assert small_map[shared_key] != large_map[shared_key]
    distractor = fills[rng.randrange(len(fills))]
    return small_keys, small_fills, large_keys, large_fills, shared_key, distractor


def _place_palette(grid: np.ndarray, occ: np.ndarray, pal: np.ndarray, side: str, rng: random.Random) -> None:
    h, w = grid.shape
    ph, pw = pal.shape
    if side == "top":
        r0, c0 = 0, rng.randint(0, w - pw)
    elif side == "bottom":
        r0, c0 = h - ph, rng.randint(0, w - pw)
    elif side == "left":
        r0, c0 = rng.randint(0, h - ph), 0
    else:
        r0, c0 = rng.randint(0, h - ph), w - pw
    grid[r0 : r0 + ph, c0 : c0 + pw] = pal
    occ[max(0, r0 - 1) : min(h, r0 + ph + 1), max(0, c0 - 1) : min(w, c0 + pw + 1)] = True


def _find_positions(h: int, w: int, occ: np.ndarray, mask: np.ndarray) -> list[tuple[int, int]]:
    sh, sw = mask.shape
    positions: list[tuple[int, int]] = []
    for r0 in range(1, h - sh):
        for c0 in range(1, w - sw):
            if occ[r0 - 1 : r0 + sh + 1, c0 - 1 : c0 + sw + 1].any():
                continue
            positions.append((r0, c0))
    return positions


def _paint_mask(grid: np.ndarray, r0: int, c0: int, mask: np.ndarray, color: int) -> None:
    view = grid[r0 : r0 + mask.shape[0], c0 : c0 + mask.shape[1]]
    view[mask] = color


def _gen_attempt() -> np.ndarray | None:
    rng = random.Random()
    bg = 0
    h = rng.randint(_GRID_H_MIN, _GRID_H_MAX)
    w = rng.randint(_GRID_W_MIN, _GRID_W_MAX)

    palette_side = rng.choice(["top", "bottom", "left", "right"])
    stem_col = rng.choice([0, 1]) if palette_side in {"top", "bottom"} else 0

    small_keys, small_fills, large_keys, large_fills, shared_key, distractor = _choose_palette_colors(rng)
    small_map = dict(zip(small_keys, small_fills))
    large_map = dict(zip(large_keys, large_fills))

    if palette_side in {"top", "bottom"}:
        palette = _build_t_palette(palette_side, small_keys, small_fills, large_keys, large_fills, stem_col)
    else:
        palette = _build_l_palette(palette_side, small_keys, small_fills, large_keys, large_fills)

    inp = np.full((h, w), bg, dtype=int)
    out = np.full((h, w), bg, dtype=int)
    occ = np.zeros((h, w), dtype=bool)
    _place_palette(inp, occ, palette, palette_side, rng)
    out[:, :] = inp

    small_mask = rng.choice(_SMALL_SHAPES)
    large_mask = rng.choice(_LARGE_SHAPES)
    distractor_mask = rng.choice(_SMALL_SHAPES + _LARGE_SHAPES)

    sprites = [
        (large_mask, shared_key, large_map[shared_key]),
        (distractor_mask, distractor, None),
        (small_mask, shared_key, small_map[shared_key]),
    ]
    sprites.sort(key=lambda item: item[0].shape[0] * item[0].shape[1], reverse=True)

    placed: list[tuple[int, int, np.ndarray, int | None]] = []
    for mask, border_color, fill_color in sprites:
        positions = _find_positions(h, w, occ, mask)
        if not positions:
            return None
        r0, c0 = rng.choice(positions)
        _paint_mask(inp, r0, c0, mask, border_color)
        _paint_mask(out, r0, c0, mask, border_color)
        if fill_color is not None:
            local_fill = _convex_fill_mask_local(mask)
            out_region = out[r0 : r0 + mask.shape[0], c0 : c0 + mask.shape[1]]
            out_region[local_fill & (~mask)] = fill_color
        sh, sw = mask.shape
        occ[r0 - 1 : r0 + sh + 1, c0 - 1 : c0 + sw + 1] = True
        placed.append((r0, c0, mask, fill_color))

    border = np.concatenate([inp[0, :], inp[-1, :], inp[1:-1, 0], inp[1:-1, -1]])
    vals, cnts = np.unique(border, return_counts=True)
    if int(vals[cnts.argmax()]) != bg:
        return None

    if np.array_equal(inp, out):
        return None

    try:
        if not np.array_equal(main(inp), out):
            return None
    except Exception:
        return None

    # Enforce visible convex-hull behavior beyond ordinary enclosed-hole filling.
    convex_gain = 0
    for r0, c0, mask, fill_color in placed:
        if fill_color is None:
            continue
        local_fill = _convex_fill_mask_local(mask)
        sprite = mask.copy()
        hole_fill = np.zeros_like(mask, dtype=bool)
        for pts in components_mask(~sprite, connectivity=4):
            if any(r == 0 or c == 0 or r == mask.shape[0] - 1 or c == mask.shape[1] - 1 for r, c in pts):
                continue
            hole_fill[pts[:, 0], pts[:, 1]] = True
        convex_gain += int(np.sum(local_fill & (~hole_fill)))
    if convex_gain <= 0:
        return None

    return inp


def generate_input() -> np.ndarray:
    for _ in range(_MAX_RETRIES):
        grid = _gen_attempt()
        if grid is not None:
            return grid
    raise RuntimeError(f"Failed to generate input after {_MAX_RETRIES} attempts")
