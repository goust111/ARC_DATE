from __future__ import annotations

import random
from collections import defaultdict

import numpy as np


BG = 0
LEGEND_HEIGHT = 3
SHAPE_LIBRARY = (
    ((0, 0), (0, 1), (0, 2)),
    ((0, 0), (1, 0), (1, 1)),
    ((0, 0), (0, 1), (1, 0), (1, 1)),
    ((0, 0), (0, 1), (0, 2), (1, 1)),
    ((0, 1), (0, 2), (1, 0), (1, 1)),
    ((0, 0), (1, 0), (2, 0), (2, 1)),
    ((0, 0), (0, 1), (0, 2), (0, 3)),
    ((0, 1), (1, 0), (1, 1), (1, 2), (2, 1)),
    ((0, 0), (0, 1), (1, 0), (1, 1), (2, 0)),
    ((0, 0), (0, 1), (0, 2), (1, 1), (2, 1)),
    ((0, 0), (1, 0), (2, 0), (2, 1), (2, 2)),
)
PAIR_PATTERNS = ("down_right", "down_left", "same_row", "same_col")
GEN_RETRIES = 400


def _neighbors4(r: int, c: int) -> tuple[tuple[int, int], ...]:
    return ((r - 1, c), (r + 1, c), (r, c - 1), (r, c + 1))


def _tighten_mask(mask: np.ndarray) -> np.ndarray:
    coords = np.argwhere(mask)
    r0, c0 = coords.min(axis=0)
    r1, c1 = coords.max(axis=0)
    return mask[int(r0) : int(r1) + 1, int(c0) : int(c1) + 1].astype(bool)


def _coords_to_mask(coords: np.ndarray) -> np.ndarray:
    pts = np.asarray(coords, dtype=int)
    pts = pts - pts.min(axis=0)
    h = int(pts[:, 0].max()) + 1
    w = int(pts[:, 1].max()) + 1
    mask = np.zeros((h, w), dtype=bool)
    mask[pts[:, 0], pts[:, 1]] = True
    return mask


def _shape_signature(coords: np.ndarray) -> tuple[tuple[int, ...], ...]:
    base = _tighten_mask(_coords_to_mask(coords))
    variants: list[tuple[tuple[int, ...], ...]] = []
    for k in range(4):
        rotated = np.rot90(base, k)
        for candidate in (rotated, np.fliplr(rotated)):
            tight = _tighten_mask(candidate)
            variants.append(tuple(tuple(int(v) for v in row) for row in tight.astype(int)))
    return min(variants)


def _center_pixel(coords: np.ndarray) -> tuple[int, int]:
    pts = np.asarray(coords, dtype=int)
    best_score = None
    best_point = None
    for i in range(int(pts.shape[0])):
        pr, pc = map(int, pts[i])
        score = int(np.abs(pts - pts[i]).sum())
        key = (score, pr, pc)
        if best_score is None or key < best_score:
            best_score = key
            best_point = (pr, pc)
    assert best_point is not None
    return best_point


def _find_components(grid: np.ndarray, bg: int = BG) -> list[dict[str, object]]:
    g = np.asarray(grid, dtype=int)
    h, w = g.shape
    seen = np.zeros((h, w), dtype=bool)
    comps: list[dict[str, object]] = []
    for r in range(h):
        for c in range(w):
            color = int(g[r, c])
            if color == bg or seen[r, c]:
                continue
            stack = [(r, c)]
            seen[r, c] = True
            cells: list[tuple[int, int]] = []
            while stack:
                cr, cc = stack.pop()
                cells.append((cr, cc))
                for nr, nc in _neighbors4(cr, cc):
                    if 0 <= nr < h and 0 <= nc < w and not seen[nr, nc] and int(g[nr, nc]) == color:
                        seen[nr, nc] = True
                        stack.append((nr, nc))
            coords = np.array(cells, dtype=int)
            comps.append(
                {
                    "color": color,
                    "coords": coords,
                    "coord_set": set(cells),
                    "center": _center_pixel(coords),
                }
            )
    return comps


def _legend_width(grid: np.ndarray, bg: int = BG) -> int:
    g = np.asarray(grid, dtype=int)
    h, w = g.shape
    assert h >= LEGEND_HEIGHT
    width = 0
    for c in range(w - 1, -1, -1):
        if not np.all(g[h - LEGEND_HEIGHT : h, c] != bg):
            break
        if h > LEGEND_HEIGHT and int(g[h - LEGEND_HEIGHT - 1, c]) != bg:
            break
        width += 1
    assert width >= 1
    return width


def _legend_mapping(grid: np.ndarray, bg: int = BG) -> tuple[dict[frozenset[int], int], int]:
    g = np.asarray(grid, dtype=int)
    h, w = g.shape
    width = _legend_width(g, bg)
    mapping: dict[frozenset[int], int] = {}
    for c in range(w - width, w):
        top = int(g[h - 3, c])
        mix = int(g[h - 2, c])
        bottom = int(g[h - 1, c])
        assert top != bg and mix != bg and bottom != bg
        key = frozenset((top, bottom))
        assert key not in mapping
        mapping[key] = mix
    return mapping, width


def _axis_cells(
    obj_mask: np.ndarray,
    source_cells: set[tuple[int, int]],
    start: tuple[int, int],
    end: tuple[int, int],
    axis: int,
) -> list[tuple[int, int]]:
    if axis == 1:
        sr, sc = start
        er, ec = end
        assert sr == er
        if sc == ec:
            return []
        step = 1 if ec > sc else -1
        left_source = False
        out = []
        c = sc
        while c != ec:
            c += step
            cell = (sr, c)
            if cell in source_cells and not left_source:
                continue
            if obj_mask[cell]:
                break
            left_source = True
            out.append(cell)
        return out

    sr, sc = start
    er, ec = end
    assert sc == ec
    if sr == er:
        return []
    step = 1 if er > sr else -1
    left_source = False
    out = []
    r = sr
    while r != er:
        r += step
        cell = (r, sc)
        if cell in source_cells and not left_source:
            continue
        if obj_mask[cell]:
            break
        left_source = True
        out.append(cell)
    return out


def _pair_path_cells(
    obj_mask: np.ndarray,
    source: dict[str, object],
    target: dict[str, object],
) -> tuple[list[tuple[int, int]], list[tuple[int, int]]]:
    src_center = tuple(source["center"])
    dst_center = tuple(target["center"])
    src_cells = source["coord_set"]
    horizontal = _axis_cells(obj_mask, src_cells, src_center, (src_center[0], dst_center[1]), axis=1)
    vertical = _axis_cells(obj_mask, src_cells, (src_center[0], dst_center[1]), dst_center, axis=0)
    return horizontal, vertical


def main(grid: np.ndarray) -> np.ndarray:
    """
    Connect each same-shape object pair using the mixed color defined by the bottom-right palette.

    Input structure:
    - The background is black.
    - The bottom-right corner contains a 3xN palette legend whose columns encode top-color + bottom-color -> middle-color.
    - The main area contains several 4-connected monochrome objects of sizes 3-6, and each shape appears exactly twice up to rotation or reflection.

    Rule:
    - Remove the legend from consideration, find every foreground object, and group objects by tight-mask shape up to rotation and reflection.
    - For each two-object shape class, look up the unordered pair of object colors in the legend to obtain the mixed line color.
    - Order the pair by row-major center pixel, then draw an orthogonal path from the earlier center: first horizontally toward the later center's column, then vertically toward the later center's row.
    - Only paint background cells; each segment skips source-object interior cells and stops immediately before entering any object.

    Key insight:
    - The pairing key is shape equivalence rather than color or proximity, and the line color is not inferred numerically: it must be read from the corner legend's visible color-combination table.
    """
    g = np.asarray(grid, dtype=int)
    h, w = g.shape
    mapping, legend_w = _legend_mapping(g, BG)

    work = g.copy()
    work[h - LEGEND_HEIGHT : h, w - legend_w : w] = BG
    components = _find_components(work, BG)
    assert components

    groups: dict[tuple[tuple[int, ...], ...], list[dict[str, object]]] = defaultdict(list)
    for comp in components:
        groups[_shape_signature(np.asarray(comp["coords"], dtype=int))].append(comp)
    assert all(len(group) == 2 for group in groups.values())

    obj_mask = work != BG
    out = g.copy()
    for group in groups.values():
        first, second = sorted(group, key=lambda comp: tuple(comp["center"]))
        color_key = frozenset((int(first["color"]), int(second["color"])))
        assert color_key in mapping
        line_color = mapping[color_key]
        horizontal, vertical = _pair_path_cells(obj_mask, first, second)
        for r, c in horizontal + vertical:
            out[r, c] = int(line_color)
    return out


def _oriented_shape(base_shape: tuple[tuple[int, int], ...]) -> np.ndarray:
    mask = _coords_to_mask(np.array(base_shape, dtype=int))
    mask = np.rot90(mask, random.randrange(4))
    if random.random() < 0.5:
        mask = np.fliplr(mask)
    return np.argwhere(mask).astype(int)


def _relative_shape(coords: np.ndarray) -> tuple[list[tuple[int, int]], tuple[int, int]]:
    center = _center_pixel(coords)
    rel = [(int(r - center[0]), int(c - center[1])) for r, c in coords]
    return rel, center


def _cells_from_center(rel: list[tuple[int, int]], center: tuple[int, int]) -> list[tuple[int, int]]:
    cr, cc = center
    return [(cr + dr, cc + dc) for dr, dc in rel]


def _expanded_cells(cells: list[tuple[int, int]]) -> set[tuple[int, int]]:
    expanded: set[tuple[int, int]] = set()
    for r, c in cells:
        for dr in (-1, 0, 1):
            for dc in (-1, 0, 1):
                expanded.add((r + dr, c + dc))
    return expanded


def _feasible_placements(
    rel: list[tuple[int, int]],
    band_r0: int,
    band_r1: int,
    usable_w: int,
) -> list[tuple[tuple[int, int], list[tuple[int, int]], set[tuple[int, int]]]]:
    rows = [dr for dr, _dc in rel]
    cols = [dc for _dr, dc in rel]
    min_dr, max_dr = min(rows), max(rows)
    min_dc, max_dc = min(cols), max(cols)
    placements = []
    for r in range(band_r0 + 1 - min_dr, band_r1 - 1 - max_dr + 1):
        for c in range(1 - min_dc, usable_w - 2 - max_dc + 1):
            cells = _cells_from_center(rel, (r, c))
            placements.append(((r, c), cells, _expanded_cells(cells)))
    random.shuffle(placements)
    return placements


def _matches_pattern(pattern: str, src: tuple[int, int], dst: tuple[int, int]) -> bool:
    sr, sc = src
    tr, tc = dst
    if pattern == "down_right":
        return sr < tr and sc + 3 <= tc
    if pattern == "down_left":
        return sr < tr and tc + 3 <= sc
    if pattern == "same_row":
        return sr == tr and sc + 3 <= tc
    if pattern == "same_col":
        return sc == tc and sr + 3 <= tr
    return False


def _pattern_clear(
    pattern: str,
    horizontal: list[tuple[int, int]],
    vertical: list[tuple[int, int]],
    src: tuple[int, int],
    dst: tuple[int, int],
) -> bool:
    if pattern in ("down_right", "down_left"):
        return bool(horizontal) and bool(vertical) and horizontal[-1] == (src[0], dst[1])
    if pattern == "same_row":
        return bool(horizontal) and not vertical
    if pattern == "same_col":
        return not horizontal and bool(vertical)
    return False


def _place_pair(
    pattern: str,
    shape_a: np.ndarray,
    shape_b: np.ndarray,
    band_r0: int,
    band_r1: int,
    usable_w: int,
) -> tuple[dict[str, object], dict[str, object]] | None:
    rel_a, _ = _relative_shape(shape_a)
    rel_b, _ = _relative_shape(shape_b)
    placements_a = _feasible_placements(rel_a, band_r0, band_r1, usable_w)
    placements_b = _feasible_placements(rel_b, band_r0, band_r1, usable_w)

    for center_a, cells_a, pad_a in placements_a[:80]:
        for center_b, cells_b, pad_b in placements_b[:120]:
            if not _matches_pattern(pattern, center_a, center_b):
                continue
            set_a = set(cells_a)
            set_b = set(cells_b)
            if set_a & set_b:
                continue
            if pad_a & set_b or pad_b & set_a:
                continue

            obj_mask = np.zeros((band_r1 + 1, usable_w), dtype=bool)
            for r, c in cells_a:
                obj_mask[r, c] = True
            for r, c in cells_b:
                obj_mask[r, c] = True

            first = {"coords": np.array(cells_a, dtype=int), "coord_set": set_a, "center": center_a}
            second = {"coords": np.array(cells_b, dtype=int), "coord_set": set_b, "center": center_b}
            horizontal, vertical = _pair_path_cells(obj_mask, first, second)
            if not _pattern_clear(pattern, horizontal, vertical, center_a, center_b):
                continue
            return first, second
    return None


def _sample_layout(pair_count: int) -> tuple[list[tuple[int, int]], int, int, int] | None:
    band_heights = [random.randint(5, 7) for _ in range(pair_count)]
    gaps = [random.randint(1, 2) for _ in range(pair_count - 1)]
    top_pad = random.randint(1, 2)
    bottom_pad = random.randint(1, 2)
    main_h = top_pad + sum(band_heights) + sum(gaps) + bottom_pad
    full_h = main_h + 4
    if full_h > 30:
        return None

    usable_w = random.randint(12, 20 if pair_count == 3 else 22)
    right_gap = random.randint(2, min(5, 30 - usable_w - pair_count))
    full_w = usable_w + right_gap + pair_count
    if full_w > 30:
        return None

    bands = []
    cursor = top_pad
    for i, height in enumerate(band_heights):
        bands.append((cursor, cursor + height - 1))
        cursor += height
        if i < len(gaps):
            cursor += gaps[i]
    return bands, full_h, full_w, usable_w


def _build_input() -> np.ndarray | None:
    pair_count = random.choice((2, 3))
    layout = _sample_layout(pair_count)
    if layout is None:
        return None
    bands, full_h, full_w, usable_w = layout

    base_shapes = random.sample(SHAPE_LIBRARY, pair_count)
    colors = random.sample(range(1, 10), 3 * pair_count)
    pair_specs = []
    for i in range(pair_count):
        pair_specs.append(
            {
                "shape": base_shapes[i],
                "obj_colors": (colors[3 * i], colors[3 * i + 1]),
                "mix": colors[3 * i + 2],
                "pattern": random.choice(PAIR_PATTERNS),
            }
        )

    grid = np.zeros((full_h, full_w), dtype=int)
    objects = []
    for band, spec in zip(bands, pair_specs):
        placed = None
        for _ in range(80):
            placed = _place_pair(
                spec["pattern"],
                _oriented_shape(spec["shape"]),
                _oriented_shape(spec["shape"]),
                band[0],
                band[1],
                usable_w,
            )
            if placed is not None:
                break
        if placed is None:
            return None

        first, second = placed
        first["color"] = int(spec["obj_colors"][0])
        second["color"] = int(spec["obj_colors"][1])
        objects.extend((first, second))

    for obj in objects:
        for r, c in obj["coord_set"]:
            grid[r, c] = int(obj["color"])

    legend_order = pair_specs[:]
    random.shuffle(legend_order)
    start_c = full_w - pair_count
    for idx, spec in enumerate(legend_order):
        c = start_c + idx
        a, b = spec["obj_colors"]
        if random.random() < 0.5:
            top, bottom = a, b
        else:
            top, bottom = b, a
        grid[full_h - 3, c] = int(top)
        grid[full_h - 2, c] = int(spec["mix"])
        grid[full_h - 1, c] = int(bottom)

    try:
        out = main(grid)
    except Exception:
        return None
    if np.array_equal(out, grid):
        return None
    return grid


def generate_input() -> np.ndarray:
    library_sigs = {_shape_signature(np.array(shape, dtype=int)) for shape in SHAPE_LIBRARY}
    assert len(library_sigs) == len(SHAPE_LIBRARY)
    for _ in range(GEN_RETRIES):
        grid = _build_input()
        if grid is not None:
            return grid
    raise RuntimeError("failed to generate a valid input")
