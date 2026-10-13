from __future__ import annotations

import numpy as np


BG = 0
MAX_GRID = 30
SCHEMA_GAP = 1
MAX_GENERATION_ATTEMPTS = 500
_RNG = np.random.default_rng()


def _bbox(coords: np.ndarray) -> tuple[int, int, int, int]:
    r0 = int(coords[:, 0].min())
    c0 = int(coords[:, 1].min())
    r1 = int(coords[:, 0].max())
    c1 = int(coords[:, 1].max())
    return r0, c0, r1, c1


def _find_components_of_color(grid: np.ndarray, color: int) -> list[np.ndarray]:
    mask = grid == int(color)
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
                cr, cc = stack.pop()
                pts.append((cr, cc))
                for dr, dc in ((1, 0), (-1, 0), (0, 1), (0, -1)):
                    nr = cr + dr
                    nc = cc + dc
                    if 0 <= nr < h and 0 <= nc < w and mask[nr, nc] and not seen[nr, nc]:
                        seen[nr, nc] = True
                        stack.append((nr, nc))
            comps.append(np.array(pts, dtype=int))
    return comps


def _component_sprite(coords: np.ndarray, color: int) -> np.ndarray:
    r0, c0, r1, c1 = _bbox(coords)
    out = np.full((r1 - r0 + 1, c1 - c0 + 1), BG, dtype=int)
    for r, c in coords:
        out[int(r) - r0, int(c) - c0] = int(color)
    return out


def _solid_rectangle_component(coords: np.ndarray) -> bool:
    r0, c0, r1, c1 = _bbox(coords)
    return int(coords.shape[0]) == (r1 - r0 + 1) * (c1 - c0 + 1)


def _detect_schema_color(grid: np.ndarray, by_color: dict[int, list[np.ndarray]]) -> int | None:
    h, w = grid.shape
    candidates: list[tuple[int, int, int]] = []
    for color, comps in by_color.items():
        if len(comps) < 2:
            continue
        if not all(_solid_rectangle_component(comp) for comp in comps):
            continue
        merged = np.concatenate(comps, axis=0)
        r0, c0, r1, c1 = _bbox(merged)
        if r1 != h - 1 or c1 != w - 1:
            continue
        area = int(merged.shape[0])
        candidates.append((area, -len(comps), int(color)))
    if not candidates:
        return None
    candidates.sort()
    return int(candidates[0][2])


def _coords_from_mask(mask: np.ndarray) -> set[tuple[int, int]]:
    return {(int(r), int(c)) for r, c in np.argwhere(mask)}


def _contiguous(values: list[int]) -> bool:
    if not values:
        return False
    lo = min(values)
    hi = max(values)
    return hi - lo + 1 == len(values)


def _vertical_boundary(
    coords_a: set[tuple[int, int]], coords_b: set[tuple[int, int]], height: int, width: int
) -> list[int] | None:
    boundary: list[int] = []
    for r in range(height):
        cols_a = [c for rr, c in coords_a if rr == r]
        cols_b = [c for rr, c in coords_b if rr == r]
        if not cols_a or not cols_b or not _contiguous(cols_a) or not _contiguous(cols_b):
            return None
        a0, a1 = min(cols_a), max(cols_a)
        b0, b1 = min(cols_b), max(cols_b)
        if a0 == 0 and b1 == width - 1 and a1 + 1 == b0:
            boundary.append(a1 + 1)
            continue
        if b0 == 0 and a1 == width - 1 and b1 + 1 == a0:
            boundary.append(b1 + 1)
            continue
        return None
    return boundary


def _horizontal_boundary(
    coords_a: set[tuple[int, int]], coords_b: set[tuple[int, int]], height: int, width: int
) -> list[int] | None:
    boundary: list[int] = []
    for c in range(width):
        rows_a = [r for r, cc in coords_a if cc == c]
        rows_b = [r for r, cc in coords_b if cc == c]
        if not rows_a or not rows_b or not _contiguous(rows_a) or not _contiguous(rows_b):
            return None
        a0, a1 = min(rows_a), max(rows_a)
        b0, b1 = min(rows_b), max(rows_b)
        if a0 == 0 and b1 == height - 1 and a1 + 1 == b0:
            boundary.append(a1 + 1)
            continue
        if b0 == 0 and a1 == height - 1 and b1 + 1 == a0:
            boundary.append(b1 + 1)
            continue
        return None
    return boundary


def _staircase_like(boundary: list[int]) -> bool:
    if len(boundary) < 2 or len(set(boundary)) < 2:
        return False
    diffs = np.diff(np.array(boundary, dtype=int))
    return bool(np.all(np.abs(diffs) <= 1))


def _reconstruct_pair(piece_a: np.ndarray, piece_b: np.ndarray) -> tuple[int, int]:
    mask_a = piece_a != BG
    mask_b = piece_b != BG
    coords_a = _coords_from_mask(mask_a)
    coords_b_local = list(_coords_from_mask(mask_b))
    candidates: list[tuple[int, int, int, int]] = []

    for dr in range(-piece_b.shape[0], piece_a.shape[0] + 1):
        for dc in range(-piece_b.shape[1], piece_a.shape[1] + 1):
            shifted_b = {(r + dr, c + dc) for r, c in coords_b_local}
            if coords_a & shifted_b:
                continue
            union = coords_a | shifted_b
            rows = [r for r, _ in union]
            cols = [c for _, c in union]
            r0 = min(rows)
            r1 = max(rows)
            c0 = min(cols)
            c1 = max(cols)
            height = r1 - r0 + 1
            width = c1 - c0 + 1
            if height * width != len(union):
                continue

            norm_a = {(r - r0, c - c0) for r, c in coords_a}
            norm_b = {(r - r0, c - c0) for r, c in shifted_b}
            v_boundary = _vertical_boundary(norm_a, norm_b, height, width)
            h_boundary = _horizontal_boundary(norm_a, norm_b, height, width)

            score = 0
            if v_boundary is not None:
                score = max(score, 1 + int(_staircase_like(v_boundary)))
            if h_boundary is not None:
                score = max(score, 1 + int(_staircase_like(h_boundary)))
            if score:
                candidates.append((score, height * width, height, width))

    if not candidates:
        raise ValueError("pair does not reconstruct to a rectangle")

    candidates.sort(key=lambda item: (-item[0], -item[1], -item[2], -item[3]))
    _score, _area, height, width = candidates[0]
    return int(height), int(width)


def main(grid: np.ndarray) -> np.ndarray:
    """
    Reassemble same-colored jigsaw halves into full rectangles and stack them by area.

    Input structure:
    - bg = Black. Each non-bg fragment is a monochrome rectangle half cut by a horizontal or vertical staircase seam.
    - Every true object color appears in exactly two disconnected pieces; those two pieces come from one solid rectangle.
    - A small cluster of Grey solid mini-rectangles sits in the lower-right corner as a sorting hint and is not part of any pair.

    Rule:
    - Group fragments by color and ignore the Grey hint cluster.
    - For each remaining color, translate its two pieces until they fit without overlap into one filled rectangle; the valid fit is the one whose union is a rectangle and whose seam is row-wise or column-wise contiguous.
    - Rebuild that full solid rectangle in the source color.
    - Sort the rebuilt rectangles by descending area and stack them top-to-bottom, left-aligned, padding the right side with bg.

    Key insight:
    - Color reveals which pieces belong together, but the real inference is finding the unique translation where the complementary staircase edges interlock to form a complete rectangle.
    """
    g = np.asarray(grid, dtype=int)
    colors = [int(c) for c in np.unique(g) if int(c) != BG]

    by_color: dict[int, list[np.ndarray]] = {}
    for color in colors:
        by_color[int(color)] = _find_components_of_color(g, int(color))

    schema_color = _detect_schema_color(g, by_color)

    rects: list[tuple[int, int, int, int]] = []
    for color in colors:
        if color == schema_color:
            continue
        comps = by_color[color]
        if len(comps) != 2:
            continue
        sprite_a = _component_sprite(comps[0], color)
        sprite_b = _component_sprite(comps[1], color)
        height, width = _reconstruct_pair(sprite_a, sprite_b)
        rects.append((height * width, height, width, color))

    if not rects:
        raise ValueError("no reconstructable pairs found")

    rects.sort(key=lambda item: (-item[0], -item[1], -item[2], item[3]))
    out_h = int(sum(height for _area, height, _width, _color in rects))
    out_w = int(max(width for _area, _height, width, _color in rects))
    out = np.full((out_h, out_w), BG, dtype=int)

    r0 = 0
    for _area, height, width, color in rects:
        out[r0 : r0 + height, :width] = int(color)
        r0 += int(height)
    return out


def _make_staircase(length: int, low: int, high: int) -> list[int]:
    cur = int(_RNG.integers(low, high + 1))
    sign = int(_RNG.choice([-1, 1]))
    boundary: list[int] = []
    while len(boundary) < length:
        run = int(_RNG.integers(1, 3))
        boundary.extend([cur] * min(run, length - len(boundary)))
        if len(boundary) >= length:
            break
        nxt = cur + sign
        if nxt < low or nxt > high:
            sign *= -1
            nxt = cur + sign
        cur = int(nxt)
        sign *= -1
    if len(set(boundary)) < 2:
        idx = int(_RNG.integers(length))
        boundary[idx] = low if boundary[idx] != low else high
    return boundary


def _crop_mask(mask: np.ndarray) -> np.ndarray:
    coords = np.argwhere(mask)
    r0, c0, r1, c1 = _bbox(coords)
    return mask[r0 : r1 + 1, c0 : c1 + 1]


def _split_rectangle(height: int, width: int, orientation: str, color: int) -> tuple[np.ndarray, np.ndarray]:
    if orientation == "v":
        boundary = _make_staircase(height, 2, width - 2)
        left = np.zeros((height, width), dtype=bool)
        right = np.zeros((height, width), dtype=bool)
        for r, cut in enumerate(boundary):
            left[r, :cut] = True
            right[r, cut:] = True
        mask_a = _crop_mask(left)
        mask_b = _crop_mask(right)
    else:
        boundary = _make_staircase(width, 2, height - 2)
        top = np.zeros((height, width), dtype=bool)
        bottom = np.zeros((height, width), dtype=bool)
        for c, cut in enumerate(boundary):
            top[:cut, c] = True
            bottom[cut:, c] = True
        mask_a = _crop_mask(top)
        mask_b = _crop_mask(bottom)

    piece_a = np.full(mask_a.shape, BG, dtype=int)
    piece_b = np.full(mask_b.shape, BG, dtype=int)
    piece_a[mask_a] = int(color)
    piece_b[mask_b] = int(color)
    return piece_a, piece_b


def _schema_sprite(rects: list[dict]) -> np.ndarray:
    mini_dims: list[tuple[int, int]] = []
    for rect in rects:
        h = 2 if rect["height"] >= 6 else 1
        w = max(2, min(6, rect["width"] - 1))
        mini_dims.append((h, w))
    schema_h = sum(h for h, _w in mini_dims) + SCHEMA_GAP * (len(mini_dims) - 1)
    schema_w = max(w for _h, w in mini_dims)
    sprite = np.full((schema_h, schema_w), BG, dtype=int)
    r0 = 0
    for h, w in mini_dims:
        sprite[r0 : r0 + h, :w] = 5
        r0 += h + SCHEMA_GAP
    return sprite


def _dilate(mask: np.ndarray) -> np.ndarray:
    out = mask.copy()
    out[1:, :] |= mask[:-1, :]
    out[:-1, :] |= mask[1:, :]
    out[:, 1:] |= mask[:, :-1]
    out[:, :-1] |= mask[:, 1:]
    out[1:, 1:] |= mask[:-1, :-1]
    out[1:, :-1] |= mask[:-1, 1:]
    out[:-1, 1:] |= mask[1:, :-1]
    out[:-1, :-1] |= mask[1:, 1:]
    return out


def _place_mask(occupied: np.ndarray, mask: np.ndarray) -> tuple[int, int] | None:
    h, w = occupied.shape
    ph, pw = mask.shape
    dilated = _dilate(occupied)
    valid: list[tuple[int, int]] = []
    for r0 in range(h - ph + 1):
        for c0 in range(w - pw + 1):
            if bool((dilated[r0 : r0 + ph, c0 : c0 + pw] & mask).any()):
                continue
            valid.append((r0, c0))
    if not valid:
        return None
    idx = int(_RNG.integers(len(valid)))
    return valid[idx]


def _paint(grid: np.ndarray, sprite: np.ndarray, r0: int, c0: int) -> None:
    mask = sprite != BG
    sub = grid[r0 : r0 + sprite.shape[0], c0 : c0 + sprite.shape[1]]
    sub[mask] = sprite[mask]


def _sample_rectangles() -> list[dict] | None:
    n_rects = int(_RNG.integers(2, 5))
    colors = list(_RNG.choice([1, 2, 3, 4, 6, 7, 8, 9], size=n_rects, replace=False))
    rects: list[dict] = []
    used_areas: set[int] = set()

    for color in colors:
        for _ in range(100):
            height = int(_RNG.integers(4, 8))
            width = int(_RNG.integers(4, 9))
            area = height * width
            if area in used_areas:
                continue
            orientation = str(_RNG.choice(["v", "h"]))
            if orientation == "v" and width < 4:
                continue
            if orientation == "h" and height < 4:
                continue
            piece_a, piece_b = _split_rectangle(height, width, orientation, int(color))
            rects.append(
                {
                    "color": int(color),
                    "height": height,
                    "width": width,
                    "area": area,
                    "pieces": [piece_a, piece_b],
                }
            )
            used_areas.add(area)
            break
        else:
            return None

    rects.sort(key=lambda item: (-item["area"], -item["height"], -item["width"], item["color"]))
    if sum(rect["height"] for rect in rects) > MAX_GRID:
        return None
    return rects


def _build_output(rects: list[dict]) -> np.ndarray:
    out_h = sum(rect["height"] for rect in rects)
    out_w = max(rect["width"] for rect in rects)
    out = np.full((out_h, out_w), BG, dtype=int)
    r0 = 0
    for rect in rects:
        out[r0 : r0 + rect["height"], : rect["width"]] = int(rect["color"])
        r0 += rect["height"]
    return out


def _build_input(rects: list[dict]) -> np.ndarray | None:
    schema = _schema_sprite(rects)

    for _ in range(80):
        height = int(_RNG.integers(22, 31))
        width = int(_RNG.integers(22, 31))
        if schema.shape[0] >= height - 2 or schema.shape[1] >= width - 2:
            continue

        grid = np.full((height, width), BG, dtype=int)
        occupied = np.zeros((height, width), dtype=bool)

        sr = height - schema.shape[0]
        sc = width - schema.shape[1]
        _paint(grid, schema, sr, sc)
        occupied[sr : sr + schema.shape[0], sc : sc + schema.shape[1]] |= schema != BG

        pieces: list[np.ndarray] = []
        for rect in rects:
            order = list(_RNG.permutation(2))
            for idx in order:
                pieces.append(rect["pieces"][int(idx)])
        pieces.sort(key=lambda sprite: -(sprite != BG).sum())

        ok = True
        for sprite in pieces:
            mask = sprite != BG
            loc = _place_mask(occupied, mask)
            if loc is None:
                ok = False
                break
            r0, c0 = loc
            _paint(grid, sprite, int(r0), int(c0))
            occupied[r0 : r0 + sprite.shape[0], c0 : c0 + sprite.shape[1]] |= mask
        if ok:
            return grid
    return None


def _gen_attempt() -> np.ndarray | None:
    rects = _sample_rectangles()
    if rects is None:
        return None

    expected = _build_output(rects)
    if expected.shape[0] > MAX_GRID or expected.shape[1] > MAX_GRID:
        return None

    inp = _build_input(rects)
    if inp is None:
        return None
    if not np.array_equal(main(inp), expected):
        return None
    return inp


def generate_input() -> np.ndarray:
    for _ in range(MAX_GENERATION_ATTEMPTS):
        grid = _gen_attempt()
        if grid is not None:
            return grid
    raise RuntimeError("failed to generate a valid input for puzzle 2")
