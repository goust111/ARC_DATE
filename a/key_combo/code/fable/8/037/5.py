from __future__ import annotations

import itertools
from collections import Counter

import numpy as np


Coord = tuple[int, int]
BBox = tuple[int, int, int, int]


def _get_bg(grid: np.ndarray) -> int:
    return int(Counter(map(int, grid.ravel())).most_common(1)[0][0])


def _in_bounds(shape: tuple[int, int], r: int, c: int) -> bool:
    h, w = map(int, shape)
    return 0 <= int(r) < h and 0 <= int(c) < w


def _components(mask: np.ndarray, connectivity: int = 4) -> list[np.ndarray]:
    h, w = map(int, mask.shape)
    seen = np.zeros((h, w), dtype=bool)
    if connectivity == 4:
        dirs = [(-1, 0), (1, 0), (0, -1), (0, 1)]
    else:
        dirs = [(dr, dc) for dr in (-1, 0, 1) for dc in (-1, 0, 1) if (dr, dc) != (0, 0)]

    comps: list[np.ndarray] = []
    for r in range(h):
        for c in range(w):
            if not mask[r, c] or seen[r, c]:
                continue
            stack = [(r, c)]
            seen[r, c] = True
            pts: list[Coord] = []
            while stack:
                cr, cc = stack.pop()
                pts.append((cr, cc))
                for dr, dc in dirs:
                    nr = cr + dr
                    nc = cc + dc
                    if _in_bounds(mask.shape, nr, nc) and mask[nr, nc] and not seen[nr, nc]:
                        seen[nr, nc] = True
                        stack.append((nr, nc))
            comps.append(np.array(pts, dtype=int))
    return comps


def _bbox_coords(coords: np.ndarray) -> BBox:
    r0, c0 = coords.min(axis=0)
    r1, c1 = coords.max(axis=0)
    return int(r0), int(c0), int(r1), int(c1)


def _crop_bbox(grid: np.ndarray, bbox: BBox) -> np.ndarray:
    r0, c0, r1, c1 = bbox
    return grid[r0 : r1 + 1, c0 : c1 + 1].copy()


def _paste_transparent(dst: np.ndarray, patch: np.ndarray, r0: int, c0: int, bg: int) -> None:
    h, w = map(int, patch.shape)
    r0 = int(r0)
    c0 = int(c0)
    view = dst[r0 : r0 + h, c0 : c0 + w]
    mask = patch != bg
    view[mask] = patch[mask]


def _sign(x: int) -> int:
    return (x > 0) - (x < 0)


def _decompose_offset(dr: int, dc: int) -> tuple[Coord, Coord]:
    abs_dr = abs(int(dr))
    abs_dc = abs(int(dc))
    assert abs_dr > 0 and abs_dc > 0 and abs_dr != abs_dc

    diag_len = min(abs_dr, abs_dc)
    sr = _sign(dr)
    sc = _sign(dc)
    diag = (sr * diag_len, sc * diag_len)
    if abs_dr > abs_dc:
        axis = (sr * (abs_dr - abs_dc), 0)
    else:
        axis = (0, sc * (abs_dc - abs_dr))
    return diag, axis


def _flip_patch(patch: np.ndarray, axis_is_horizontal: bool) -> np.ndarray:
    return np.flipud(patch) if axis_is_horizontal else np.fliplr(patch)


def _find_marker_color(grid: np.ndarray, bg: int) -> int:
    colors = [int(c) for c in np.unique(grid) if int(c) != bg]
    candidates: list[int] = []
    for color in colors:
        coords = np.argwhere(grid == color)
        if coords.shape != (2, 2):
            continue
        rest_mask = (grid != bg) & (grid != color)
        if len(_components(rest_mask, connectivity=8)) != 1:
            continue
        dr = int(coords[1, 0] - coords[0, 0])
        dc = int(coords[1, 1] - coords[0, 1])
        if dr == 0 or dc == 0 or abs(dr) == abs(dc):
            continue
        candidates.append(color)
    assert len(candidates) == 1, "expected one color to encode the two marker pixels"
    return candidates[0]


def _enumerate_lattice(
    shape: tuple[int, int],
    origin: Coord,
    sprite_shape: tuple[int, int],
    diag: Coord,
    axis: Coord,
) -> list[tuple[int, int, int, int]]:
    h, w = map(int, sprite_shape)
    limit = int(sum(shape)) + 2
    placements: list[tuple[int, int, int, int]] = []
    seen = set()
    for m in range(-limit, limit + 1):
        for n in range(-limit, limit + 1):
            r = int(origin[0] + m * diag[0] + n * axis[0])
            c = int(origin[1] + m * diag[1] + n * axis[1])
            if 0 <= r <= shape[0] - h and 0 <= c <= shape[1] - w:
                key = (r, c)
                assert key not in seen, "basis vectors must define unique lattice anchors"
                seen.add(key)
                placements.append((m, n, r, c))
    placements.sort()
    return placements


def _render_from_placements(
    shape: tuple[int, int],
    bg: int,
    sprite: np.ndarray,
    placements: list[tuple[int, int, int, int]],
    axis_is_horizontal: bool,
    parity_source: str = "m",
    use_flip: bool = True,
    flip_axis_override: bool | None = None,
) -> np.ndarray:
    out = np.full(shape, bg, dtype=int)
    h, w = map(int, sprite.shape)
    for m, n, r, c in placements:
        parity_value = m if parity_source == "m" else n
        patch = sprite
        do_flip = use_flip and (parity_value % 2 != 0)
        if do_flip:
            axis_flag = axis_is_horizontal if flip_axis_override is None else flip_axis_override
            patch = _flip_patch(sprite, axis_flag)
        window = out[r : r + h, c : c + w]
        assert np.all(window == bg), "generated copies must have disjoint bounding boxes"
        _paste_transparent(out, patch, r, c, bg)
    return out


def _delta_line_placements(
    shape: tuple[int, int],
    origin: Coord,
    sprite_shape: tuple[int, int],
    delta: Coord,
) -> list[tuple[int, int, int, int]]:
    h, w = map(int, sprite_shape)
    limit = int(sum(shape)) + 2
    placements: list[tuple[int, int, int, int]] = []
    for k in range(-limit, limit + 1):
        r = int(origin[0] + k * delta[0])
        c = int(origin[1] + k * delta[1])
        if 0 <= r <= shape[0] - h and 0 <= c <= shape[1] - w:
            placements.append((k, 0, r, c))
    placements.sort()
    return placements


def _solve_details(grid: np.ndarray) -> tuple[np.ndarray, dict]:
    g = np.asarray(grid, dtype=int)
    assert g.ndim == 2

    bg = _get_bg(g)
    marker_color = _find_marker_color(g, bg)
    marker_coords = np.argwhere(g == marker_color)
    p = tuple(map(int, marker_coords[0]))
    q = tuple(map(int, marker_coords[1]))

    sprite_mask = (g != bg) & (g != marker_color)
    sprite_components = _components(sprite_mask, connectivity=8)
    assert len(sprite_components) == 1, "expected one foreground sprite"
    sprite_bbox = _bbox_coords(sprite_components[0])
    sprite = _crop_bbox(g, sprite_bbox)
    origin = (sprite_bbox[0], sprite_bbox[1])

    diag, axis = _decompose_offset(q[0] - p[0], q[1] - p[1])
    axis_is_horizontal = axis[0] == 0
    placements = _enumerate_lattice(g.shape, origin, sprite.shape, diag, axis)
    out = _render_from_placements(g.shape, bg, sprite, placements, axis_is_horizontal)
    return out, {
        "bg": bg,
        "marker_color": marker_color,
        "markers": (p, q),
        "sprite": sprite,
        "origin": origin,
        "diag": diag,
        "axis": axis,
        "axis_is_horizontal": axis_is_horizontal,
        "placements": placements,
    }


def main(grid: np.ndarray) -> np.ndarray:
    """
    Build an oblique sprite lattice from the marker offset, alternating mirror state along the diagonal generator.

    Input structure:
    - A dominant background contains one multi-pixel sprite and exactly two same-color marker pixels.
    - The marker color does not appear in the sprite, and the marker offset has nonzero row and column parts with unequal magnitudes.
    - The non-marker foreground pixels form one 8-connected sprite that may use multiple colors and preserve its bounding-box holes.

    Rule:
    - Decompose the marker-to-marker offset into a 45-degree diagonal vector `d` plus the remaining horizontal or vertical axis vector `a`.
    - Use the input sprite bbox as the placement origin and place copies at every in-bounds anchor `origin + m*d + n*a` for integer `m, n`.
    - When `m` is odd, mirror the sprite before pasting: flip up/down if `a` is horizontal, or left/right if `a` is vertical.
    - Paste only copies whose full bbox stays inside the grid, and erase the two marker pixels back to background.

    Key insight:
    - The marker pair does not mark a destination; it compresses two independent lattice generators, and the diagonal generator also carries an alternating glide-reflection state.
    """
    out, _ = _solve_details(grid)
    return out


_RNG = np.random.default_rng()
_COLOR_IDS = np.arange(10, dtype=int)
_MAX_ATTEMPTS = 800
_REQUIRED_TEMPLATES = [
    [(0, 0), (1, 0), (2, 0), (0, 1), (1, 1)],
    [(0, 0), (1, 0), (2, 0), (0, -1), (1, -1)],
    [(0, 0), (-1, 0), (-2, 0), (0, 1), (-1, 1)],
    [(0, 0), (-1, 0), (-2, 0), (0, -1), (-1, -1)],
]


def _random_mask(h: int, w: int) -> np.ndarray | None:
    for _ in range(200):
        mask = _RNG.random((h, w)) < float(_RNG.uniform(0.45, 0.72))
        area = int(mask.sum())
        if area < 6 or area > h * w - 1:
            continue
        if not mask[0].any() or not mask[-1].any() or not mask[:, 0].any() or not mask[:, -1].any():
            continue
        if len(_components(mask, connectivity=8)) != 1:
            continue
        if np.array_equal(mask, np.flipud(mask)) or np.array_equal(mask, np.fliplr(mask)):
            continue
        return mask
    return None


def _random_sprite(bg: int, colors: tuple[int, int]) -> np.ndarray | None:
    for _ in range(250):
        h = int(_RNG.integers(3, 6))
        w = int(_RNG.integers(3, 6))
        mask = _random_mask(h, w)
        if mask is None:
            continue
        coords = np.argwhere(mask)
        area = int(coords.shape[0])
        if area < 6:
            continue
        accent_count = int(_RNG.integers(3, area - 2))
        accent_ids = _RNG.choice(area, size=accent_count, replace=False)
        sprite = np.full((h, w), bg, dtype=int)
        sprite[mask] = colors[0]
        accent_pts = coords[accent_ids]
        sprite[accent_pts[:, 0], accent_pts[:, 1]] = colors[1]
        if np.array_equal(sprite, np.flipud(sprite)) or np.array_equal(sprite, np.fliplr(sprite)):
            continue
        counts = Counter(map(int, sprite[sprite != bg]))
        if min(counts.values()) < 3:
            continue
        return sprite
    return None


def _choose_vectors(sprite_shape: tuple[int, int]) -> tuple[Coord, Coord]:
    h, w = map(int, sprite_shape)
    axis_is_horizontal = bool(_RNG.integers(0, 2))
    diag_min = max(h, w)
    diag_len = int(_RNG.integers(diag_min, min(8, diag_min + 3)))
    axis_min = w if axis_is_horizontal else h
    axis_len = int(_RNG.integers(axis_min, max(axis_min + 1, 8)))
    sr = int(_RNG.choice([-1, 1]))
    sc = int(_RNG.choice([-1, 1]))
    diag = (sr * diag_len, sc * diag_len)
    axis = (0, sc * axis_len) if axis_is_horizontal else (sr * axis_len, 0)
    return diag, axis


def _anchor_interval(
    grid_shape: tuple[int, int],
    sprite_shape: tuple[int, int],
    shifts: list[Coord],
) -> tuple[tuple[int, int], tuple[int, int]] | None:
    h, w = map(int, sprite_shape)
    h_lim = grid_shape[0] - h
    w_lim = grid_shape[1] - w
    r_lo = max(-sr for sr, _ in shifts)
    r_hi = min(h_lim - sr for sr, _ in shifts)
    c_lo = max(-sc for _, sc in shifts)
    c_hi = min(w_lim - sc for _, sc in shifts)
    if r_lo > r_hi or c_lo > c_hi:
        return None
    return (int(r_lo), int(r_hi)), (int(c_lo), int(c_hi))


def _render_shortcuts(
    shape: tuple[int, int],
    bg: int,
    sprite: np.ndarray,
    origin: Coord,
    diag: Coord,
    axis: Coord,
    axis_is_horizontal: bool,
    placements: list[tuple[int, int, int, int]],
) -> list[np.ndarray]:
    delta = (diag[0] + axis[0], diag[1] + axis[1])
    outputs = []
    for shortcut_placements, kwargs in [
        (placements, {"use_flip": False}),
        (placements, {"parity_source": "n"}),
        (placements, {"flip_axis_override": not axis_is_horizontal}),
        (
            _delta_line_placements(shape, origin, sprite.shape, delta),
            {"use_flip": False},
        ),
    ]:
        try:
            outputs.append(
                _render_from_placements(
                    shape,
                    bg,
                    sprite,
                    shortcut_placements,
                    axis_is_horizontal,
                    **kwargs,
                )
            )
        except AssertionError:
            continue
    return outputs


def _all_bg_neighborhood(grid: np.ndarray, r: int, c: int) -> bool:
    r0 = max(0, r - 1)
    r1 = min(grid.shape[0], r + 2)
    c0 = max(0, c - 1)
    c1 = min(grid.shape[1], c + 2)
    return bool(np.all(grid[r0:r1, c0:c1] == grid[r, c]))


def _place_markers_on_bg(out: np.ndarray, delta: Coord, marker_color: int) -> tuple[Coord, Coord] | None:
    bg = int(out[0, 0]) if np.all(out == out[0, 0]) else _get_bg(out)
    candidates: list[tuple[Coord, Coord]] = []
    for r in range(out.shape[0]):
        for c in range(out.shape[1]):
            rr = r + delta[0]
            cc = c + delta[1]
            if not _in_bounds(out.shape, rr, cc):
                continue
            if out[r, c] != bg or out[rr, cc] != bg:
                continue
            if not _all_bg_neighborhood(out, r, c) or not _all_bg_neighborhood(out, rr, cc):
                continue
            candidates.append(((r, c), (rr, cc)))
    if not candidates:
        return None
    p, q = candidates[int(_RNG.integers(len(candidates)))]
    assert p != q
    return p, q


def _build_case() -> tuple[np.ndarray, np.ndarray] | None:
    bg, marker_color, sprite_c1, sprite_c2 = map(
        int, _RNG.choice(_COLOR_IDS, size=4, replace=False)
    )
    sprite = _random_sprite(bg, (sprite_c1, sprite_c2))
    if sprite is None:
        return None

    diag, axis = _choose_vectors(sprite.shape)
    axis_is_horizontal = axis[0] == 0
    template = _REQUIRED_TEMPLATES[int(_RNG.integers(len(_REQUIRED_TEMPLATES)))]
    shifts = [(m * diag[0] + n * axis[0], m * diag[1] + n * axis[1]) for m, n in template]

    row_span = max(sr for sr, _ in shifts) - min(sr for sr, _ in shifts)
    col_span = max(sc for _, sc in shifts) - min(sc for _, sc in shifts)
    min_h = sprite.shape[0] + row_span + 4
    min_w = sprite.shape[1] + col_span + 4
    if min_h > 30 or min_w > 30:
        return None

    grid_h = int(_RNG.integers(max(16, min_h), min(31, min_h + 7)))
    grid_w = int(_RNG.integers(max(16, min_w), min(31, min_w + 7)))
    anchor_range = _anchor_interval((grid_h, grid_w), sprite.shape, shifts)
    if anchor_range is None:
        return None
    (r_lo, r_hi), (c_lo, c_hi) = anchor_range
    origin = (
        int(_RNG.integers(r_lo, r_hi + 1)),
        int(_RNG.integers(c_lo, c_hi + 1)),
    )

    placements = _enumerate_lattice((grid_h, grid_w), origin, sprite.shape, diag, axis)
    placement_pairs = {(m, n) for m, n, _, _ in placements}
    if not all(pair in placement_pairs for pair in template):
        return None
    if len(placements) < 5:
        return None
    if len({m for m, _, _, _ in placements}) < 3:
        return None
    if len({n for _, n, _, _ in placements}) < 2:
        return None
    if not any(m % 2 != 0 for m, _, _, _ in placements):
        return None
    if not any(n % 2 != 0 for _, n, _, _ in placements):
        return None

    try:
        out = _render_from_placements((grid_h, grid_w), bg, sprite, placements, axis_is_horizontal)
    except AssertionError:
        return None

    shortcuts = _render_shortcuts(
        (grid_h, grid_w), bg, sprite, origin, diag, axis, axis_is_horizontal, placements
    )
    if any(np.array_equal(out, wrong) for wrong in shortcuts):
        return None

    marker_pair = _place_markers_on_bg(out, (diag[0] + axis[0], diag[1] + axis[1]), marker_color)
    if marker_pair is None:
        return None
    p, q = marker_pair

    inp = np.full((grid_h, grid_w), bg, dtype=int)
    _paste_transparent(inp, sprite, origin[0], origin[1], bg)
    inp[p] = marker_color
    inp[q] = marker_color

    if np.array_equal(inp, out):
        return None
    try:
        actual, details = _solve_details(inp)
    except AssertionError:
        return None
    if not np.array_equal(actual, out):
        return None

    if details["diag"] != diag or details["axis"] != axis:
        return None
    if details["origin"] != origin:
        return None
    if not any(abs(m) == 1 for m, _, _, _ in details["placements"]):
        return None
    if not any(abs(n) == 1 for _, n, _, _ in details["placements"]):
        return None
    return inp, out


def generate_input() -> np.ndarray:
    for _ in range(_MAX_ATTEMPTS):
        case = _build_case()
        if case is not None:
            return case[0]
    raise RuntimeError("failed to generate a valid oblique-lattice ARC input")
