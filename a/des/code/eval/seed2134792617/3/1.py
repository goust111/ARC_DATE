from __future__ import annotations

from collections import Counter
import random

import numpy as np


BG = 0
GREY = 5
PALETTE = [1, 2, 3, 4, 6, 7, 8, 9]
MAX_GENERATION_ATTEMPTS = 400


def _most_common_color(grid: np.ndarray) -> int:
    counts = Counter(int(v) for v in grid.ravel())
    return int(counts.most_common(1)[0][0])


def _neighbors4(r: int, c: int) -> tuple[tuple[int, int], ...]:
    return ((r - 1, c), (r + 1, c), (r, c - 1), (r, c + 1))


def _find_components(grid: np.ndarray, bg: int) -> list[dict]:
    g = np.asarray(grid, dtype=int)
    h, w = g.shape
    seen = np.zeros((h, w), dtype=bool)
    comps: list[dict] = []

    for r in range(h):
        for c in range(w):
            color = int(g[r, c])
            if color == bg or seen[r, c]:
                continue

            stack = [(r, c)]
            seen[r, c] = True
            coords: list[tuple[int, int]] = []
            while stack:
                rr, cc = stack.pop()
                coords.append((rr, cc))
                for nr, nc in _neighbors4(rr, cc):
                    if 0 <= nr < h and 0 <= nc < w and not seen[nr, nc] and int(g[nr, nc]) == color:
                        seen[nr, nc] = True
                        stack.append((nr, nc))

            arr = np.array(coords, dtype=int)
            r0, c0 = arr.min(axis=0)
            r1, c1 = arr.max(axis=0)
            comps.append(
                {
                    "color": color,
                    "coords": arr,
                    "bbox": (int(r0), int(c0), int(r1), int(c1)),
                    "area": int(arr.shape[0]),
                }
            )

    return comps


def _component_patch(comp: dict) -> np.ndarray:
    r0, c0, r1, c1 = comp["bbox"]
    patch = np.zeros((r1 - r0 + 1, c1 - c0 + 1), dtype=bool)
    coords = np.asarray(comp["coords"], dtype=int)
    patch[coords[:, 0] - r0, coords[:, 1] - c0] = True
    return patch


def _is_square_ring(comp: dict) -> bool:
    r0, c0, r1, c1 = comp["bbox"]
    side_h = r1 - r0 + 1
    side_w = c1 - c0 + 1
    if side_h != side_w or side_h < 5:
        return False

    coords = set(map(tuple, np.asarray(comp["coords"], dtype=int)))
    expected = set()
    for rr in range(r0, r1 + 1):
        for cc in range(c0, c1 + 1):
            if rr in (r0, r1) or cc in (c0, c1):
                expected.add((rr, cc))
    return coords == expected


def _find_frame(grid: np.ndarray, bg: int) -> dict:
    candidates = [comp for comp in _find_components(grid, bg) if _is_square_ring(comp)]
    assert candidates, "expected one hollow square frame"
    return max(candidates, key=lambda comp: comp["bbox"][2] - comp["bbox"][0])


def _extract_band(
    grid: np.ndarray, frame_bbox: tuple[int, int, int, int], bg: int
) -> tuple[int, list[tuple[int, int]]]:
    _fr0, fc0, _fr1, _fc1 = frame_bbox
    best_col: int | None = None
    best_segments: list[tuple[int, int]] | None = None
    best_key: tuple[int, int] | None = None

    for c in range(fc0):
        rows = np.flatnonzero(grid[:, c] != bg)
        if rows.size == 0:
            continue
        if not np.array_equal(rows, np.arange(int(rows[0]), int(rows[-1]) + 1)):
            continue

        segments: list[tuple[int, int]] = []
        used: set[int] = set()
        cur_color = int(grid[int(rows[0]), c])
        cur_len = 1
        used.add(cur_color)
        ok = True

        for r in rows[1:]:
            color = int(grid[int(r), c])
            if color == cur_color:
                cur_len += 1
                continue
            segments.append((cur_color, cur_len))
            if color in used:
                ok = False
                break
            used.add(color)
            cur_color = color
            cur_len = 1

        if not ok:
            continue
        segments.append((cur_color, cur_len))
        if len(segments) < 2:
            continue

        key = (int(rows.size), -int(c))
        if best_key is None or key > best_key:
            best_key = key
            best_col = int(c)
            best_segments = segments

    assert best_col is not None
    assert best_segments is not None, "failed to locate the color band"
    return best_col, best_segments


def _draw_ring(grid: np.ndarray, r0: int, c0: int, side: int, thickness: int, color: int) -> None:
    if side <= 0 or thickness <= 0:
        return
    grid[r0 : r0 + thickness, c0 : c0 + side] = color
    grid[r0 + side - thickness : r0 + side, c0 : c0 + side] = color
    grid[r0 : r0 + side, c0 : c0 + thickness] = color
    grid[r0 : r0 + side, c0 + side - thickness : c0 + side] = color


def _place_patch_on_layer_top(
    out: np.ndarray,
    layer_r0: int,
    layer_c0: int,
    layer_side: int,
    layer_thickness: int,
    patch: np.ndarray,
    color: int,
) -> None:
    ph, pw = map(int, patch.shape)
    assert ph <= layer_thickness
    assert pw <= layer_side
    rr = layer_r0 + (layer_thickness - ph) // 2
    cc = layer_c0 + (layer_side - pw) // 2
    region = out[rr : rr + ph, cc : cc + pw]
    region[patch] = color


def _compose_output(
    shape: tuple[int, int],
    frame: dict,
    band: list[tuple[int, int]],
    fragments: dict[int, np.ndarray],
    bg: int,
) -> np.ndarray:
    out = np.full(shape, bg, dtype=int)
    fr0, fc0, fr1, _fc1 = frame["bbox"]
    frame_color = int(frame["color"])
    frame_side = int(fr1 - fr0 + 1)

    _draw_ring(out, fr0, fc0, frame_side, 1, frame_color)

    inset = 1
    layers: list[tuple[int, int, int, int, int]] = []
    for color, thickness in band:
        layer_side = frame_side - 2 * inset
        assert layer_side >= max(3, 2 * thickness + 1)
        _draw_ring(out, fr0 + inset, fc0 + inset, layer_side, thickness, color)
        layers.append((fr0 + inset, fc0 + inset, layer_side, thickness, color))
        inset += thickness

    for idx, (layer_r0, layer_c0, layer_side, layer_thickness, color) in enumerate(layers):
        patch = fragments.get(int(color))
        if patch is None:
            continue
        next_color = int(band[idx + 1][0]) if idx + 1 < len(band) else frame_color
        _place_patch_on_layer_top(out, layer_r0, layer_c0, layer_side, layer_thickness, patch, next_color)

    return out


def main(grid: np.ndarray) -> np.ndarray:
    """
    Build nested concentric square frames from the left color band, then recolor and
    embed the matching scattered fragments into their corresponding layers.
    """
    g = np.asarray(grid, dtype=int)
    assert g.ndim == 2
    bg = _most_common_color(g)

    frame = _find_frame(g, bg)
    band_col, band = _extract_band(g, frame["bbox"], bg)

    work = g.copy()
    for rr, cc in frame["coords"]:
        work[int(rr), int(cc)] = bg

    rows = np.flatnonzero(g[:, band_col] != bg)
    work[rows, band_col] = bg

    fragments: dict[int, np.ndarray] = {}
    for comp in _find_components(work, bg):
        fragments[int(comp["color"])] = _component_patch(comp)

    return _compose_output(g.shape, frame, band, fragments, bg)


_BASE_PATTERNS = [
    np.array([[1, 1, 1]], dtype=bool),
    np.array([[1, 1, 1, 1]], dtype=bool),
    np.array([[1, 1, 1, 1, 1]], dtype=bool),
    np.array([[1, 0], [1, 1]], dtype=bool),
    np.array([[1, 1], [1, 0]], dtype=bool),
    np.array([[1, 1], [1, 1]], dtype=bool),
    np.array([[1, 1, 1], [0, 1, 0]], dtype=bool),
    np.array([[1, 1, 0], [0, 1, 1]], dtype=bool),
    np.array([[1, 0, 0], [1, 1, 1]], dtype=bool),
    np.array([[0, 0, 1], [1, 1, 1]], dtype=bool),
    np.array([[1, 0, 1], [1, 1, 1]], dtype=bool),
    np.array([[1, 1, 1], [1, 0, 1]], dtype=bool),
    np.array([[0, 1, 0], [1, 1, 1], [0, 1, 0]], dtype=bool),
    np.array([[1, 1, 1], [0, 1, 0], [0, 1, 0]], dtype=bool),
    np.array([[0, 1, 0], [0, 1, 0], [1, 1, 1]], dtype=bool),
    np.array([[1, 0, 0], [1, 1, 1], [1, 0, 0]], dtype=bool),
]


def _pattern_variants() -> list[np.ndarray]:
    variants: list[np.ndarray] = []
    seen: set[tuple[tuple[int, ...], ...]] = set()
    for base in _BASE_PATTERNS:
        for flip in (False, True):
            arr = np.fliplr(base) if flip else base
            for k in range(4):
                rot = np.rot90(arr, k)
                key = tuple(tuple(int(v) for v in row) for row in rot.astype(int))
                if key in seen:
                    continue
                if int(rot.sum()) < 3 or int(rot.sum()) > 6:
                    continue
                seen.add(key)
                variants.append(rot.astype(bool))
    return variants


_PATTERN_VARIANTS = _pattern_variants()


def _pick_fragment_pattern(max_height: int, max_width: int) -> np.ndarray | None:
    choices = [
        pattern
        for pattern in _PATTERN_VARIANTS
        if int(pattern.shape[0]) <= max_height and int(pattern.shape[1]) <= max_width
    ]
    if not choices:
        return None
    return random.choice(choices).copy()


def _can_place_patch(grid: np.ndarray, patch: np.ndarray, r0: int, c0: int, bg: int, padding: int = 1) -> bool:
    h, w = grid.shape
    ph, pw = patch.shape
    if r0 < 0 or c0 < 0 or r0 + ph > h or c0 + pw > w:
        return False

    coords = np.argwhere(patch)
    for rr, cc in coords:
        ar = int(r0 + rr)
        ac = int(c0 + cc)
        if int(grid[ar, ac]) != bg:
            return False
        for dr in range(-padding, padding + 1):
            for dc in range(-padding, padding + 1):
                nr = ar + dr
                nc = ac + dc
                if 0 <= nr < h and 0 <= nc < w and int(grid[nr, nc]) != bg and not (dr == 0 and dc == 0):
                    return False
    return True


def _paste_patch(grid: np.ndarray, patch: np.ndarray, r0: int, c0: int, color: int) -> None:
    ph, pw = patch.shape
    region = grid[r0 : r0 + ph, c0 : c0 + pw]
    region[patch] = color


def _random_fragment_layout(
    grid: np.ndarray,
    colors: list[int],
    patterns: dict[int, np.ndarray],
    min_col: int,
    bg: int,
) -> bool:
    h, w = grid.shape
    order = colors.copy()
    random.shuffle(order)
    for color in order:
        patch = patterns[int(color)]
        ph, pw = patch.shape
        candidates = [
            (r0, c0)
            for r0 in range(1, h - ph)
            for c0 in range(min_col, w - pw)
        ]
        random.shuffle(candidates)
        placed = False
        for r0, c0 in candidates:
            if _can_place_patch(grid, patch, r0, c0, bg, padding=1):
                _paste_patch(grid, patch, r0, c0, int(color))
                placed = True
                break
        if not placed:
            return False
    return True


def _gen_attempt() -> np.ndarray | None:
    bg = BG
    frame_color = GREY

    layer_count = random.randint(2, 4)
    layer_colors = random.sample(PALETTE, layer_count)
    thicknesses = [random.randint(1, 3) for _ in range(layer_count)]
    band = list(zip(layer_colors, thicknesses))

    hole_side = random.randint(3, 7)
    outer_side = int(2 * (1 + sum(thicknesses)) + hole_side)

    right_room = random.randint(9, 13)
    left_gap = random.randint(3, 5)
    top_bottom_margin = random.randint(2, 5)
    height = outer_side + 2 * top_bottom_margin + random.randint(0, 6)
    width = left_gap + outer_side + right_room + random.randint(1, 5)

    if width < outer_side + 14:
        width = outer_side + 14

    grid = np.full((height, width), bg, dtype=int)

    frame_r0 = random.randint(1, height - outer_side - 1)
    frame_c0 = random.randint(4, left_gap + 2)
    frame_side = outer_side
    _draw_ring(grid, frame_r0, frame_c0, frame_side, 1, frame_color)
    frame = {
        "color": frame_color,
        "bbox": (frame_r0, frame_c0, frame_r0 + frame_side - 1, frame_c0 + frame_side - 1),
        "coords": np.argwhere(grid == frame_color),
    }

    band_height = sum(thicknesses)
    band_col_hi = frame_c0 - 3
    if band_col_hi < 1:
        return None
    band_col = random.randint(1, band_col_hi)
    band_r0 = random.randint(1, height - band_height - 1)

    rr = band_r0
    for color, thickness in band:
        grid[rr : rr + thickness, band_col] = int(color)
        rr += thickness

    patterns: dict[int, np.ndarray] = {}
    inset = 1
    for color, thickness in band:
        layer_side = frame_side - 2 * inset
        max_width = min(6, layer_side - 1)
        patch = _pick_fragment_pattern(thickness, max_width)
        if patch is None:
            return None
        patterns[int(color)] = patch
        inset += thickness

    min_fragment_col = frame_c0 + frame_side + 2
    if min_fragment_col >= width - 2:
        return None
    if not _random_fragment_layout(grid, layer_colors, patterns, min_fragment_col, bg):
        return None

    expected = _compose_output(grid.shape, frame, band, patterns, bg)
    try:
        actual = main(grid)
    except AssertionError:
        return None
    if not np.array_equal(actual, expected):
        return None
    return grid


def generate_input() -> np.ndarray:
    for _ in range(MAX_GENERATION_ATTEMPTS):
        grid = _gen_attempt()
        if grid is not None:
            return grid
    raise RuntimeError("failed to generate a valid input")
