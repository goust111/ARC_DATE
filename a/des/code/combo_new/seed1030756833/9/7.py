from __future__ import annotations

from collections import deque
import random

import numpy as np


BLACK = 0
BLUE = 1
RED = 2
GREY = 5
ORANGE = 7
AZURE = 8

DIR4 = ((1, 0), (-1, 0), (0, 1), (0, -1))
DIR8 = (
    (-1, -1),
    (-1, 0),
    (-1, 1),
    (0, -1),
    (0, 1),
    (1, -1),
    (1, 0),
    (1, 1),
)


def _background(g: np.ndarray) -> int:
    vals, counts = np.unique(g, return_counts=True)
    return int(vals[int(np.argmax(counts))])


def _components(mask: np.ndarray) -> list[np.ndarray]:
    h, w = mask.shape
    seen = np.zeros((h, w), dtype=bool)
    comps: list[np.ndarray] = []

    for sr in range(h):
        for sc in range(w):
            if seen[sr, sc] or not bool(mask[sr, sc]):
                continue
            q = deque([(int(sr), int(sc))])
            seen[sr, sc] = True
            pts: list[tuple[int, int]] = []
            while q:
                r, c = q.popleft()
                pts.append((r, c))
                for dr, dc in DIR4:
                    rr, cc = r + dr, c + dc
                    if 0 <= rr < h and 0 <= cc < w and not seen[rr, cc] and bool(mask[rr, cc]):
                        seen[rr, cc] = True
                        q.append((rr, cc))
            comps.append(np.array(pts, dtype=int))
    return comps


def _bbox(coords: np.ndarray | set[tuple[int, int]]) -> tuple[int, int, int, int]:
    if isinstance(coords, set):
        rows = [r for r, _c in coords]
        cols = [c for _r, c in coords]
        return min(rows), min(cols), max(rows), max(cols)
    return (
        int(coords[:, 0].min()),
        int(coords[:, 1].min()),
        int(coords[:, 0].max()),
        int(coords[:, 1].max()),
    )


def _is_full_rect(cells: set[tuple[int, int]]) -> bool:
    if not cells:
        return False
    r0, c0, r1, c1 = _bbox(cells)
    return len(cells) == (r1 - r0 + 1) * (c1 - c0 + 1)


def _find_library_bbox(g: np.ndarray, bg: int) -> tuple[int, int, int, int]:
    h, w = g.shape
    best: tuple[tuple[int, int, int], tuple[int, int, int, int]] | None = None

    for comp in _components(g == BLACK):
        r0, c0, r1, c1 = _bbox(comp)
        bh = r1 - r0 + 1
        bw = c1 - c0 + 1
        if bh < 6 or bw < 4:
            continue
        if not (
            np.all(g[r0, c0 : c1 + 1] == BLACK)
            and np.all(g[r1, c0 : c1 + 1] == BLACK)
            and np.all(g[r0 : r1 + 1, c0] == BLACK)
            and np.all(g[r0 : r1 + 1, c1] == BLACK)
        ):
            continue

        interior = g[r0 + 1 : r1, c0 + 1 : c1]
        colors = {int(v) for v in interior.ravel() if int(v) not in (BLACK, bg)}
        if len(colors) < 3:
            continue

        # Prefer the framed legend nearest the lower-right corner.
        score = (int(r1 == h - 1) + int(c1 == w - 1), r1 + c1, len(colors))
        if best is None or score > best[0]:
            best = (score, (int(r0), int(c0), int(r1), int(c1)))

    assert best is not None, "expected a lower-right black-framed pattern library"
    return best[1]


def _parse_library_entry(
    g: np.ndarray,
    bg: int,
    bbox: tuple[int, int, int, int],
    row0: int,
    row1: int,
) -> tuple[int, np.ndarray] | None:
    _r0, c0, _r1, c1 = bbox
    sub = g[row0 : row1 + 1, c0 + 1 : c1]
    marks = (sub != BLACK) & (sub != bg)
    coords = np.argwhere(marks)
    if coords.size == 0:
        return None

    label_col = int(coords[:, 1].min())
    label_rows = [int(r) for r, c in coords if int(c) == label_col]
    if not label_rows:
        return None
    label = int(sub[label_rows[0], label_col])

    pattern_coords = coords[coords[:, 1] > label_col]
    if pattern_coords.size == 0:
        return None
    pr0 = int(pattern_coords[:, 0].min())
    pc0 = int(pattern_coords[:, 1].min())
    if pr0 + 2 > sub.shape[0] or pc0 + 2 > sub.shape[1]:
        return None
    pattern = np.array(sub[pr0 : pr0 + 2, pc0 : pc0 + 2], dtype=int)
    if pattern.shape != (2, 2):
        return None
    if bool(np.any((pattern == BLACK) | (pattern == bg))):
        return None
    if len(_pattern_colors(pattern)) != 2:
        return None
    return label, pattern


def _patterns_from_library(g: np.ndarray, bg: int, bbox: tuple[int, int, int, int]) -> dict[int, np.ndarray]:
    r0, c0, r1, c1 = bbox
    sep_rows = [
        r
        for r in range(r0 + 1, r1)
        if bool(np.all(g[r, c0 : c1 + 1] == BLACK))
    ]
    entries: list[tuple[int, int]]
    if sep_rows:
        sep = sep_rows[len(sep_rows) // 2]
        entries = [(r0 + 1, sep - 1), (sep + 1, r1 - 1)]
    else:
        rows_with_marks = []
        for r in range(r0 + 1, r1):
            row = g[r, c0 + 1 : c1]
            if bool(np.any((row != BLACK) & (row != bg))):
                rows_with_marks.append(r)
        assert rows_with_marks, "pattern library contains no visible entries"
        gaps = [i for i in range(len(rows_with_marks) - 1) if rows_with_marks[i + 1] > rows_with_marks[i] + 1]
        if gaps:
            i = gaps[0]
            entries = [(rows_with_marks[0], rows_with_marks[i]), (rows_with_marks[i + 1], rows_with_marks[-1])]
        else:
            mid = (rows_with_marks[0] + rows_with_marks[-1]) // 2
            entries = [(rows_with_marks[0], mid), (mid + 1, rows_with_marks[-1])]

    patterns: dict[int, np.ndarray] = {}
    for er0, er1 in entries:
        if er0 > er1:
            continue
        parsed = _parse_library_entry(g, bg, bbox, er0, er1)
        if parsed is not None:
            label, pattern = parsed
            patterns[int(label)] = pattern

    assert patterns, "expected at least one seed-to-pattern mapping"
    return patterns


def _pattern_colors(pattern: np.ndarray) -> list[int]:
    colors: list[int] = []
    for v in pattern.ravel().tolist():
        iv = int(v)
        if iv not in colors:
            colors.append(iv)
    return colors


def _invert_pattern(pattern: np.ndarray) -> np.ndarray:
    colors = _pattern_colors(pattern)
    assert len(colors) == 2, "fill pattern must have exactly two colors"
    a, b = colors
    out = np.array(pattern, dtype=int)
    a_mask = pattern == int(a)
    b_mask = pattern == int(b)
    out[a_mask] = int(b)
    out[b_mask] = int(a)
    return out


def _paint_pattern(
    out: np.ndarray,
    cells: set[tuple[int, int]],
    seed: tuple[int, int],
    pattern: np.ndarray,
) -> None:
    sr, sc = seed
    for r, c in cells:
        out[r, c] = int(pattern[(r - sr) % 2, (c - sc) % 2])


def _split_l_shape(cells: set[tuple[int, int]]) -> tuple[set[tuple[int, int]], set[tuple[int, int]]]:
    r0, c0, r1, c1 = _bbox(cells)
    candidates: list[tuple[set[tuple[int, int]], set[tuple[int, int]], str, int]] = []

    for cut in range(r0, r1):
        a = {p for p in cells if p[0] <= cut}
        b = cells - a
        if _is_full_rect(a) and _is_full_rect(b):
            candidates.append((a, b, "h", int(cut)))

    for cut in range(c0, c1):
        a = {p for p in cells if p[1] <= cut}
        b = cells - a
        if _is_full_rect(a) and _is_full_rect(b):
            candidates.append((a, b, "v", int(cut)))

    assert candidates, "L-shaped region must split into two rectangles"

    def key(item: tuple[set[tuple[int, int]], set[tuple[int, int]], str, int]) -> tuple[int, int, int, int]:
        a, b, orient, cut = item
        area_a = len(a)
        area_b = len(b)
        return (max(area_a, area_b), abs(area_a - area_b), min(area_a, area_b), -cut if orient == "h" else cut)

    a, b, _orient, _cut = max(candidates, key=key)
    return set(a), set(b)


def main(grid: np.ndarray) -> np.ndarray:
    """
    Fill enclosed rectangle and L-shaped regions with seed-anchored patterns read from the lower-right library.

    Input structure:
    - The background is grey.
    - Black one-pixel-thick closed outlines enclose disjoint regions; each interior is either a rectangle or an orthogonal L shape.
    - Every outlined region contains exactly one red or blue seed pixel.
    - A black-framed pattern library in the lower-right shows the red-seed 2x2 checker pattern and the blue-seed 2x2 stripe pattern.

    Rule:
    - Read the library as a mapping from each seed color to its visible 2x2 fill pattern.
    - A rectangular region is tiled completely with the seed's pattern, with the pattern's top-left cell anchored on the seed pixel.
    - An L-shaped region is split at its concave corner into two rectangles; the rectangle containing the seed gets the seed pattern and the other rectangle gets the color-swapped pattern.
    - Black outlines and the pattern library are preserved.

    Key insight:
    The legend supplies the pattern, the outline shape selects rectangle versus L logic, and the seed position fixes both the tiling phase and which L subrectangle keeps the original color order.
    """
    g = np.asarray(grid, dtype=int)
    out = g.copy()
    h, w = g.shape
    bg = _background(g)

    lib_bbox = _find_library_bbox(g, bg)
    patterns = _patterns_from_library(g, bg, lib_bbox)
    seed_colors = set(patterns.keys())

    lr0, lc0, lr1, lc1 = lib_bbox
    library_mask = np.zeros((h, w), dtype=bool)
    library_mask[lr0 : lr1 + 1, lc0 : lc1 + 1] = True

    passable = (g != BLACK) & (~library_mask)
    for comp in _components(passable):
        if bool(
            np.any((comp[:, 0] == 0) | (comp[:, 0] == h - 1) | (comp[:, 1] == 0) | (comp[:, 1] == w - 1))
        ):
            continue

        cells = {(int(r), int(c)) for r, c in comp}
        seed_cells = [(r, c) for r, c in cells if int(g[r, c]) in seed_colors]
        if len(seed_cells) != 1:
            continue
        seed = seed_cells[0]
        pattern = patterns[int(g[seed])]

        if _is_full_rect(cells):
            _paint_pattern(out, cells, seed, pattern)
            continue

        rect_a, rect_b = _split_l_shape(cells)
        area_a = len(rect_a)
        area_b = len(rect_b)
        large, small = (rect_a, rect_b) if area_a >= area_b else (rect_b, rect_a)

        base_cells = large
        inverse_cells = small
        if seed in small:
            base_cells = small
            inverse_cells = large

        _paint_pattern(out, base_cells, seed, pattern)
        _paint_pattern(out, inverse_cells, seed, _invert_pattern(pattern))

    return out


# ----------------------------- generator -----------------------------

LIBRARY_H = 7
LIBRARY_W = 5
GENERATION_ATTEMPTS = 500
PLACEMENT_ATTEMPTS = 200


def _library_patch() -> np.ndarray:
    patch = np.full((LIBRARY_H, LIBRARY_W), BLACK, dtype=int)
    patch[1, 1] = RED
    patch[1:3, 2:4] = np.array([[RED, ORANGE], [ORANGE, RED]], dtype=int)
    patch[2, 1] = GREY
    patch[4, 1] = BLUE
    patch[4:6, 2:4] = np.array([[BLUE, AZURE], [BLUE, AZURE]], dtype=int)
    patch[5, 1] = GREY
    return patch


def _rect_cells(height: int, width: int) -> set[tuple[int, int]]:
    return {(r, c) for r in range(height) for c in range(width)}


def _l_cells(
    height: int,
    width: int,
    missing_h: int,
    missing_w: int,
    corner: str,
) -> set[tuple[int, int]]:
    cells = _rect_cells(height, width)
    missing: set[tuple[int, int]] = set()
    if corner == "tl":
        missing = {(r, c) for r in range(missing_h) for c in range(missing_w)}
    elif corner == "tr":
        missing = {(r, c) for r in range(missing_h) for c in range(width - missing_w, width)}
    elif corner == "bl":
        missing = {(r, c) for r in range(height - missing_h, height) for c in range(missing_w)}
    else:
        missing = {(r, c) for r in range(height - missing_h, height) for c in range(width - missing_w, width)}
    return cells - missing


def _normalize_region(
    interior: set[tuple[int, int]],
) -> tuple[set[tuple[int, int]], set[tuple[int, int]], int, int]:
    wall: set[tuple[int, int]] = set()
    for r, c in interior:
        for dr, dc in DIR8:
            p = (r + dr, c + dc)
            if p not in interior:
                wall.add(p)

    all_cells = interior | wall
    min_r = min(r for r, _c in all_cells)
    min_c = min(c for _r, c in all_cells)
    norm_interior = {(r - min_r, c - min_c) for r, c in interior}
    norm_wall = {(r - min_r, c - min_c) for r, c in wall}
    max_r = max(r for r, _c in norm_interior | norm_wall)
    max_c = max(c for _r, c in norm_interior | norm_wall)
    return norm_interior, norm_wall, max_r + 1, max_c + 1


def _make_region(kind: str, seed_pref: str | None, seed_color: int, rng: random.Random) -> dict | None:
    for _ in range(100):
        if kind == "rect":
            ih = rng.randint(2, 5)
            iw = rng.randint(3, 7)
            interior = _rect_cells(ih, iw)
            norm_interior, wall, ph, pw = _normalize_region(interior)
            seed = rng.choice(tuple(norm_interior))
            return {
                "kind": kind,
                "interior": norm_interior,
                "wall": wall,
                "height": ph,
                "width": pw,
                "seed": seed,
                "seed_color": int(seed_color),
            }

        ih = rng.randint(4, 7)
        iw = rng.randint(4, 7)
        mh = rng.randint(1, ih - 2)
        mw = rng.randint(1, iw - 2)
        corner = rng.choice(["tl", "tr", "bl", "br"])
        interior = _l_cells(ih, iw, mh, mw, corner)
        norm_interior, wall, ph, pw = _normalize_region(interior)
        if _is_full_rect(norm_interior):
            continue
        try:
            a, b = _split_l_shape(norm_interior)
        except AssertionError:
            continue
        if len(a) == len(b):
            continue
        large, small = (a, b) if len(a) > len(b) else (b, a)
        seed_pool = large if seed_pref == "large" else small
        if not seed_pool:
            continue
        seed = rng.choice(tuple(seed_pool))
        return {
            "kind": kind,
            "interior": norm_interior,
            "wall": wall,
            "height": ph,
            "width": pw,
            "seed": seed,
            "seed_color": int(seed_color),
        }
    return None


def _mark_reserved(occ: np.ndarray, r0: int, c0: int, height: int, width: int, pad: int = 1) -> None:
    h, w = occ.shape
    occ[max(0, r0 - pad) : min(h, r0 + height + pad), max(0, c0 - pad) : min(w, c0 + width + pad)] = True


def _can_place(occ: np.ndarray, region: dict, r0: int, c0: int) -> bool:
    h, w = occ.shape
    ph = int(region["height"])
    pw = int(region["width"])
    if r0 <= 0 or c0 <= 0 or r0 + ph >= h or c0 + pw >= w:
        return False
    if bool(np.any(occ[r0 - 1 : r0 + ph + 1, c0 - 1 : c0 + pw + 1])):
        return False
    return True


def _paint_region(grid: np.ndarray, region: dict, r0: int, c0: int) -> None:
    for rr, cc in region["wall"]:
        grid[r0 + rr, c0 + cc] = BLACK
    for rr, cc in region["interior"]:
        grid[r0 + rr, c0 + cc] = GREY
    sr, sc = region["seed"]
    grid[r0 + sr, c0 + sc] = int(region["seed_color"])


def _place_region(grid: np.ndarray, occ: np.ndarray, region: dict, rng: random.Random) -> bool:
    h, w = grid.shape
    ph = int(region["height"])
    pw = int(region["width"])
    if ph + 2 >= h or pw + 2 >= w:
        return False

    for _ in range(PLACEMENT_ATTEMPTS):
        r0 = rng.randint(1, h - ph - 1)
        c0 = rng.randint(1, w - pw - 1)
        if not _can_place(occ, region, r0, c0):
            continue
        _paint_region(grid, region, r0, c0)
        _mark_reserved(occ, r0, c0, ph, pw, pad=1)
        return True
    return False


def _generated_grid_valid(inp: np.ndarray) -> bool:
    if inp.ndim != 2 or max(inp.shape) > 30:
        return False
    if _background(inp) != GREY:
        return False
    try:
        out = main(inp)
    except Exception:
        return False
    if out.shape != inp.shape or max(out.shape) > 30:
        return False
    if np.array_equal(inp, out):
        return False
    lib = _find_library_bbox(inp, GREY)
    r0, c0, r1, c1 = lib
    return bool(np.array_equal(inp[r0 : r1 + 1, c0 : c1 + 1], out[r0 : r1 + 1, c0 : c1 + 1]))


def _gen_attempt(rng: random.Random) -> np.ndarray | None:
    height = rng.randint(20, 30)
    width = rng.randint(20, 30)
    grid = np.full((height, width), GREY, dtype=int)
    occ = np.zeros((height, width), dtype=bool)

    lib = _library_patch()
    lr0 = height - LIBRARY_H
    lc0 = width - LIBRARY_W
    grid[lr0:height, lc0:width] = lib
    _mark_reserved(occ, lr0, lc0, LIBRARY_H, LIBRARY_W, pad=1)

    specs: list[tuple[str, str | None]] = [("rect", None), ("l", "large"), ("l", "small")]
    if rng.random() < 0.75:
        specs.append((rng.choice(["rect", "l"]), rng.choice(["large", "small"])))
    if rng.random() < 0.35:
        specs.append((rng.choice(["rect", "l"]), rng.choice(["large", "small"])))
    rng.shuffle(specs)

    colors = [RED, BLUE]
    while len(colors) < len(specs):
        colors.append(rng.choice([RED, BLUE]))
    rng.shuffle(colors)

    regions = []
    for (kind, pref), seed_color in zip(specs, colors, strict=False):
        region = _make_region(kind, pref, seed_color, rng)
        if region is None:
            return None
        regions.append(region)

    rng.shuffle(regions)
    for region in regions:
        if not _place_region(grid, occ, region, rng):
            return None

    if not _generated_grid_valid(grid):
        return None
    return grid


def generate_input() -> np.ndarray:
    rng = random.Random()
    for _ in range(GENERATION_ATTEMPTS):
        grid = _gen_attempt(rng)
        if grid is not None:
            return grid
    raise RuntimeError("failed to generate a valid seed1030756833/9/7 input")
