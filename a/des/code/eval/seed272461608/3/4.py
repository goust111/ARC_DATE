from __future__ import annotations

import numpy as np


def _most_common_color(grid: np.ndarray) -> int:
    vals, counts = np.unique(np.asarray(grid, dtype=int), return_counts=True)
    return int(vals[counts.argmax()])


def _components(mask: np.ndarray) -> list[np.ndarray]:
    mask = np.asarray(mask, dtype=bool)
    h, w = mask.shape
    seen = np.zeros((h, w), dtype=bool)
    comps: list[np.ndarray] = []
    for r in range(h):
        for c in range(w):
            if not mask[r, c] or seen[r, c]:
                continue
            stack = [(r, c)]
            seen[r, c] = True
            pts: list[tuple[int, int]] = []
            while stack:
                rr, cc = stack.pop()
                pts.append((int(rr), int(cc)))
                for nr, nc in ((rr - 1, cc), (rr + 1, cc), (rr, cc - 1), (rr, cc + 1)):
                    if 0 <= nr < h and 0 <= nc < w and mask[nr, nc] and not seen[nr, nc]:
                        seen[nr, nc] = True
                        stack.append((nr, nc))
            comps.append(np.asarray(pts, dtype=int))
    return comps


def _bbox(coords: np.ndarray) -> tuple[int, int, int, int]:
    rows = coords[:, 0]
    cols = coords[:, 1]
    return int(rows.min()), int(cols.min()), int(rows.max()), int(cols.max())


def _find_frame_component(grid: np.ndarray, bg: int) -> tuple[int, tuple[int, int, int, int]]:
    g = np.asarray(grid, dtype=int)
    best: tuple[tuple[int, int, int, int, int], int, tuple[int, int, int, int]] | None = None

    for color in sorted(int(v) for v in np.unique(g) if int(v) != bg):
        for coords in _components(g == color):
            area = int(len(coords))
            if area <= 2:
                continue

            r0, c0, r1, c1 = _bbox(coords)
            height = int(r1 - r0 + 1)
            width = int(c1 - c0 + 1)
            if height < 6 or width < 6:
                continue

            pts = {tuple(map(int, p)) for p in coords}
            corners = ((r0, c0), (r0, c1), (r1, c0), (r1, c1))
            corner_hits = sum((r, c) in pts for r, c in corners)
            if corner_hits != 3:
                continue

            top = sum((r0, c) in pts for c in range(c0, c1 + 1))
            bottom = sum((r1, c) in pts for c in range(c0, c1 + 1))
            left = sum((r, c0) in pts for r in range(r0, r1 + 1))
            right = sum((r, c1) in pts for r in range(r0, r1 + 1))
            if min(top, bottom, left, right) == 0:
                continue

            boundary = sum(r in (r0, r1) or c in (c0, c1) for r, c in pts)
            interior = area - boundary
            perimeter = 2 * height + 2 * width - 4
            if 100 * boundary < 55 * perimeter:
                continue
            if 8 * interior > area:
                continue

            score = (corner_hits, min(top, bottom, left, right), boundary, -interior, area)
            candidate = (score, int(color), (r0, c0, r1, c1))
            if best is None or candidate[0] > best[0]:
                best = candidate

    assert best is not None, "expected a unique frame-like component"
    return best[1], best[2]


def _infer_tile(inner: np.ndarray, bg: int) -> np.ndarray:
    patch = np.asarray(inner, dtype=int)
    h, w = patch.shape
    observed = [(r, c, int(patch[r, c])) for r in range(h) for c in range(w) if int(patch[r, c]) != bg]
    assert observed, "expected visible pattern cells inside the frame"

    best: tuple[tuple[int, int, int], np.ndarray] | None = None
    for pr in range(1, h + 1):
        for pc in range(1, w + 1):
            tile = np.full((pr, pc), -1, dtype=int)
            valid = True
            for r, c, value in observed:
                rr = r % pr
                cc = c % pc
                existing = int(tile[rr, cc])
                if existing == -1:
                    tile[rr, cc] = value
                elif existing != value:
                    valid = False
                    break
            if not valid or np.any(tile == -1):
                continue
            score = (pr * pc, pr + pc, max(pr, pc))
            if best is None or score < best[0]:
                best = (score, tile.copy())

    assert best is not None, "pattern period must be fully observable"
    return best[1]


def _render_periodic(tile: np.ndarray, shape: tuple[int, int]) -> np.ndarray:
    pr, pc = tile.shape
    h, w = shape
    out = np.empty((h, w), dtype=int)
    for r in range(h):
        for c in range(w):
            out[r, c] = int(tile[r % pr, c % pc])
    return out


def main(grid: np.ndarray) -> np.ndarray:
    """
    Recover a partially erased rectangular frame and extrapolate its periodic interior pattern.

    Input structure:
    - bg = most common color.
    - One hollow rectangular frame is present, but one corner block is erased to bg.
    - The frame interior contains a repeating 2-3 color pattern.
    - Outside the frame, tiny noise components may reuse the frame color or pattern colors.

    Rule:
    - Ignore tiny frame-external noise components.
    - Detect the frame component from the three surviving corners.
    - Infer the smallest row/column period consistent with the visible interior cells.
    - Reconstruct the missing corner of the border and the interior pattern.
    """
    g = np.asarray(grid, dtype=int)
    bg = _most_common_color(g)
    frame_color, (r0, c0, r1, c1) = _find_frame_component(g, bg)

    inner = g[r0 + 1 : r1, c0 + 1 : c1]
    tile = _infer_tile(inner, bg)
    filled_inner = _render_periodic(tile, inner.shape)

    out = g.copy()
    out[r0 + 1 : r1, c0 + 1 : c1] = filled_inner
    out[r0, c0 : c1 + 1] = frame_color
    out[r1, c0 : c1 + 1] = frame_color
    out[r0 : r1 + 1, c0] = frame_color
    out[r0 : r1 + 1, c1] = frame_color
    return out


# --------------- generator ---------------

_COLOR_IDS = tuple(range(10))
_PATTERN_KINDS = ("checker", "row_stripes", "col_stripes", "block")
_CORNERS = ("tl", "tr", "bl", "br")
_MAX_GEN_ATTEMPTS = 300


def _make_tile(kind: str, palette: list[int], rng: np.random.Generator) -> np.ndarray:
    if kind == "checker":
        a, b = palette[:2]
        return np.asarray([[a, b], [b, a]], dtype=int)
    if kind == "row_stripes":
        period = 2 if len(palette) == 2 else int(rng.integers(2, 4))
        colors = palette[:period]
        return np.asarray(colors, dtype=int).reshape(period, 1)
    if kind == "col_stripes":
        period = 2 if len(palette) == 2 else int(rng.integers(2, 4))
        colors = palette[:period]
        return np.asarray(colors, dtype=int).reshape(1, period)
    assert kind == "block"
    a, b, c = palette[:3]
    return np.asarray([[a, b], [c, a]], dtype=int)


def _draw_frame(grid: np.ndarray, bbox: tuple[int, int, int, int], color: int) -> None:
    r0, c0, r1, c1 = bbox
    grid[r0, c0 : c1 + 1] = color
    grid[r1, c0 : c1 + 1] = color
    grid[r0 : r1 + 1, c0] = color
    grid[r0 : r1 + 1, c1] = color


def _corner_slice(
    bbox: tuple[int, int, int, int], corner: str, miss_h: int, miss_w: int
) -> tuple[slice, slice]:
    r0, c0, r1, c1 = bbox
    if corner == "tl":
        return slice(r0, r0 + miss_h), slice(c0, c0 + miss_w)
    if corner == "tr":
        return slice(r0, r0 + miss_h), slice(c1 - miss_w + 1, c1 + 1)
    if corner == "bl":
        return slice(r1 - miss_h + 1, r1 + 1), slice(c0, c0 + miss_w)
    return slice(r1 - miss_h + 1, r1 + 1), slice(c1 - miss_w + 1, c1 + 1)


def _can_place_noise(
    grid: np.ndarray, cells: list[tuple[int, int]], bg: int, bbox: tuple[int, int, int, int]
) -> bool:
    h, w = grid.shape
    r0, c0, r1, c1 = bbox
    cell_set = set(cells)
    for r, c in cells:
        if not (0 <= r < h and 0 <= c < w):
            return False
        if grid[r, c] != bg:
            return False
        if r0 - 1 <= r <= r1 + 1 and c0 - 1 <= c <= c1 + 1:
            return False
    for r, c in cells:
        for rr in range(r - 1, r + 2):
            for cc in range(c - 1, c + 2):
                if not (0 <= rr < h and 0 <= cc < w):
                    continue
                if (rr, cc) in cell_set:
                    continue
                if grid[rr, cc] != bg:
                    return False
    return True


def _place_noise_components(
    grid: np.ndarray,
    bg: int,
    bbox: tuple[int, int, int, int],
    color: int,
    count: int,
    rng: np.random.Generator,
) -> bool:
    shapes = (
        ((0, 0),),
        ((0, 0), (1, 0)),
        ((0, 0), (0, 1)),
    )
    h, w = grid.shape
    for _ in range(count):
        placed = False
        for _ in range(80):
            shape = shapes[int(rng.integers(0, len(shapes)))]
            base_r = int(rng.integers(0, h))
            base_c = int(rng.integers(0, w))
            cells = [(base_r + dr, base_c + dc) for dr, dc in shape]
            if not _can_place_noise(grid, cells, bg, bbox):
                continue
            for r, c in cells:
                grid[r, c] = color
            placed = True
            break
        if not placed:
            return False
    return True


def _sample_missing_size(full_len: int, rng: np.random.Generator) -> int:
    center = full_len // 2
    lo = max(3, center - 1)
    hi = min(full_len - 2, center + 1)
    choices = list(range(lo, hi + 1))
    return int(choices[int(rng.integers(0, len(choices)))])


def _gen_attempt(rng: np.random.Generator) -> tuple[np.ndarray, np.ndarray] | None:
    kind = _PATTERN_KINDS[int(rng.integers(0, len(_PATTERN_KINDS)))]
    corner = _CORNERS[int(rng.integers(0, len(_CORNERS)))]

    if kind in {"checker", "row_stripes", "col_stripes"}:
        pattern_colors_needed = 3 if kind != "checker" and int(rng.integers(0, 2)) == 1 else 2
    else:
        pattern_colors_needed = 3

    colors = [int(v) for v in rng.permutation(_COLOR_IDS)]
    bg = colors[0]
    frame_color = colors[1]
    pattern_palette = colors[2 : 2 + pattern_colors_needed]
    tile = _make_tile(kind, pattern_palette, rng)
    tile_h, tile_w = tile.shape

    inner_h = int(rng.integers(max(6, tile_h * 2 + 1), max(7, tile_h * 4 + 3)))
    inner_w = int(rng.integers(max(6, tile_w * 2 + 1), max(7, tile_w * 4 + 3)))
    inner_h = min(inner_h, 11)
    inner_w = min(inner_w, 11)
    frame_h = inner_h + 2
    frame_w = inner_w + 2

    grid_h = frame_h + int(rng.integers(7, 13))
    grid_w = frame_w + int(rng.integers(7, 13))
    if grid_h < frame_h + 4 or grid_w < frame_w + 4:
        return None

    r0 = int(rng.integers(2, grid_h - frame_h))
    c0 = int(rng.integers(2, grid_w - frame_w))
    bbox = (r0, c0, r0 + frame_h - 1, c0 + frame_w - 1)

    out = np.full((grid_h, grid_w), bg, dtype=int)
    _draw_frame(out, bbox, frame_color)
    out[r0 + 1 : r0 + 1 + inner_h, c0 + 1 : c0 + 1 + inner_w] = _render_periodic(tile, (inner_h, inner_w))

    inp = out.copy()
    miss_h = _sample_missing_size(frame_h, rng)
    miss_w = _sample_missing_size(frame_w, rng)
    rs, cs = _corner_slice(bbox, corner, miss_h, miss_w)
    inp[rs, cs] = bg

    noisy_in = inp.copy()
    noisy_out = out.copy()

    border_noise = int(rng.integers(3, 7))
    pattern_noise = int(rng.integers(2, 6))
    if not _place_noise_components(noisy_in, bg, bbox, frame_color, border_noise, rng):
        return None
    noisy_out[noisy_in == frame_color] = frame_color

    pattern_colors = list(dict.fromkeys(int(v) for v in tile.ravel()))
    for _ in range(pattern_noise):
        color = pattern_colors[int(rng.integers(0, len(pattern_colors)))]
        if not _place_noise_components(noisy_in, bg, bbox, color, 1, rng):
            return None
    for r in range(grid_h):
        for c in range(grid_w):
            if noisy_in[r, c] != inp[r, c] and not (r0 <= r <= r0 + frame_h - 1 and c0 <= c <= c0 + frame_w - 1):
                noisy_out[r, c] = noisy_in[r, c]

    try:
        recovered_color, recovered_bbox = _find_frame_component(noisy_in, bg)
        if recovered_color != frame_color or recovered_bbox != bbox:
            return None
        if not np.array_equal(main(noisy_in), noisy_out):
            return None
    except Exception:
        return None

    return noisy_in, noisy_out


def generate_input() -> np.ndarray:
    rng = np.random.default_rng()
    for _ in range(_MAX_GEN_ATTEMPTS):
        candidate = _gen_attempt(rng)
        if candidate is not None:
            return candidate[0]
    raise RuntimeError(f"failed to generate a valid sample after {_MAX_GEN_ATTEMPTS} attempts")
