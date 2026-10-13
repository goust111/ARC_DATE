from collections import deque
import random

import numpy as np


BLACK = 0
BLUE = 1
RED = 2
GREEN = 3
YELLOW = 4
GREY = 5

_GLYPH_CROSS = np.array(
    [[0, 1, 0], [1, 1, 1], [0, 1, 0]],
    dtype=bool,
)
_GLYPH_X = np.array(
    [[1, 0, 1], [0, 1, 0], [1, 0, 1]],
    dtype=bool,
)
_BLOCK_OFFSETS = ((0, 0), (0, 1), (1, 0), (1, 1))

_GEN_REGION_COUNT_WEIGHTS = ((2, 2), (3, 5), (4, 3))
_GEN_SHAPE_ATTEMPTS = 200
_GEN_LAYOUT_ATTEMPTS = 80
_GEN_ATTEMPTS = 300


def _neighbors4(r: int, c: int):
    yield r - 1, c
    yield r + 1, c
    yield r, c - 1
    yield r, c + 1


def _in_bounds(shape: tuple[int, int], r: int, c: int) -> bool:
    h, w = map(int, shape)
    return 0 <= int(r) < h and 0 <= int(c) < w


def _bbox_coords(coords: list[tuple[int, int]]) -> tuple[int, int, int, int]:
    rows = [r for r, _ in coords]
    cols = [c for _, c in coords]
    return min(rows), min(cols), max(rows), max(cols)


def _components(mask: np.ndarray) -> list[list[tuple[int, int]]]:
    h, w = map(int, mask.shape)
    seen = np.zeros((h, w), dtype=bool)
    comps: list[list[tuple[int, int]]] = []
    for r in range(h):
        for c in range(w):
            if not bool(mask[r, c]) or bool(seen[r, c]):
                continue
            seen[r, c] = True
            queue = deque([(r, c)])
            coords: list[tuple[int, int]] = []
            while queue:
                rr, cc = queue.popleft()
                coords.append((rr, cc))
                for nr, nc in _neighbors4(rr, cc):
                    if _in_bounds(mask.shape, nr, nc) and bool(mask[nr, nc]) and not bool(seen[nr, nc]):
                        seen[nr, nc] = True
                        queue.append((nr, nc))
            comps.append(coords)
    return comps


def _boundary_from_filled(mask: np.ndarray) -> np.ndarray:
    boundary = np.zeros_like(mask, dtype=bool)
    h, w = map(int, mask.shape)
    for r in range(h):
        for c in range(w):
            if not bool(mask[r, c]):
                continue
            for nr, nc in _neighbors4(r, c):
                if not _in_bounds(mask.shape, nr, nc) or not bool(mask[nr, nc]):
                    boundary[r, c] = True
                    break
    return boundary


def _outside_black_mask(grid: np.ndarray) -> np.ndarray:
    g = np.asarray(grid, dtype=int)
    h, w = map(int, g.shape)
    outside = np.zeros((h, w), dtype=bool)
    queue: deque[tuple[int, int]] = deque()
    for r, c in ((0, 0), (0, w - 1), (h - 1, 0), (h - 1, w - 1)):
        if int(g[r, c]) == BLACK and not bool(outside[r, c]):
            outside[r, c] = True
            queue.append((r, c))
    assert queue, "expected at least one black corner for the exterior flood"
    while queue:
        r, c = queue.popleft()
        for nr, nc in _neighbors4(r, c):
            if _in_bounds(g.shape, nr, nc) and int(g[nr, nc]) == BLACK and not bool(outside[nr, nc]):
                outside[nr, nc] = True
                queue.append((nr, nc))
    return outside


def _decode_glyph(grey_coords: list[tuple[int, int]]) -> int:
    assert len(grey_coords) in (4, 5), "expected one 4- or 5-pixel glyph per region"
    r0, c0, r1, c1 = _bbox_coords(grey_coords)
    mask = np.zeros((r1 - r0 + 1, c1 - c0 + 1), dtype=bool)
    for r, c in grey_coords:
        mask[r - r0, c - c0] = True
    if len(grey_coords) == 5 and mask.shape == (3, 3):
        if np.array_equal(mask, _GLYPH_CROSS):
            return BLUE
        if np.array_equal(mask, _GLYPH_X):
            return RED
    if len(grey_coords) == 4 and mask.shape == (2, 2) and bool(mask.all()):
        return YELLOW
    raise AssertionError("unrecognized glyph pattern")


def _analyze_regions(grid: np.ndarray) -> tuple[list[dict], np.ndarray]:
    g = np.asarray(grid, dtype=int)
    outside = _outside_black_mask(g)
    interior_mask = (g != GREEN) & ~outside
    regions: list[dict] = []
    for coords in _components(interior_mask):
        grey_coords = [(r, c) for r, c in coords if int(g[r, c]) == GREY]
        assert grey_coords, "every enclosed region must contain a grey glyph"
        fill = _decode_glyph(grey_coords)
        colors = {int(g[r, c]) for r, c in coords}
        assert colors <= {BLACK, GREY}, "interior regions may only contain black and grey"
        regions.append(
            {
                "coords": coords,
                "grey_coords": grey_coords,
                "fill": fill,
                "bbox": _bbox_coords(coords),
                "area": len(coords),
            }
        )
    assert regions, "expected at least one enclosed region"
    return regions, outside


def main(grid: np.ndarray) -> np.ndarray:
    """
    Fill each green-enclosed region with the color encoded by its embedded grey glyph.

    Input structure:
    - The grid has black background, several one-cell-thick green closed boundaries, and one grey 3x3 glyph inside each enclosed chamber.
    - The outer black area is connected to the grid corners, while enclosed chambers are separated from it by green boundary pixels.
    - Each glyph is either a five-pixel cross, a five-pixel X, or a four-pixel 2x2 corner block.

    Rule:
    - Flood the black exterior from the grid corners using 4-connectivity.
    - Treat every remaining non-green connected component as one enclosed region.
    - Decode that region's grey glyph: cross -> blue, X -> red, corner block -> yellow.
    - Recolor every non-green cell in the region, including the glyph itself, with the decoded color.
    - Keep the green boundaries and the exterior black background unchanged.

    Key insight:
    - The green topology identifies which cells belong to each sealed chamber, and the chamber-local grey glyph provides the fill color for that entire chamber.
    """
    g = np.asarray(grid, dtype=int)
    out = g.copy()
    regions, _outside = _analyze_regions(g)
    for region in regions:
        color = int(region["fill"])
        for r, c in region["coords"]:
            out[r, c] = color
    return out


def _random_cut(mask: np.ndarray) -> None:
    h, w = map(int, mask.shape)
    mode = random.choice(("corner", "side"))
    if mode == "corner":
        cut_h = random.randint(1, max(1, min(3, h - 5)))
        cut_w = random.randint(1, max(1, min(3, w - 5)))
        corner = random.choice(("tl", "tr", "bl", "br"))
        if corner == "tl":
            mask[:cut_h, :cut_w] = False
        elif corner == "tr":
            mask[:cut_h, w - cut_w :] = False
        elif corner == "bl":
            mask[h - cut_h :, :cut_w] = False
        else:
            mask[h - cut_h :, w - cut_w :] = False
        return

    side = random.choice(("top", "bottom", "left", "right"))
    if side in ("top", "bottom"):
        depth = random.randint(1, max(1, min(3, h - 5)))
        width = random.randint(1, max(1, w - 4))
        start = random.randint(1, w - width - 1)
        rows = slice(0, depth) if side == "top" else slice(h - depth, h)
        mask[rows, start : start + width] = False
    else:
        depth = random.randint(1, max(1, min(3, w - 5)))
        width = random.randint(1, max(1, h - 4))
        start = random.randint(1, h - width - 1)
        cols = slice(0, depth) if side == "left" else slice(w - depth, w)
        mask[start : start + width, cols] = False


def _glyph_windows(chamber: np.ndarray, glyph: np.ndarray | None = None) -> list[tuple[int, int]]:
    h, w = map(int, chamber.shape)
    windows: list[tuple[int, int]] = []
    for r in range(h - 2):
        for c in range(w - 2):
            if not bool(chamber[r : r + 3, c : c + 3].all()):
                continue
            if glyph is not None:
                ok = True
                for gr, gc in np.argwhere(glyph):
                    rr = r + int(gr)
                    cc = c + int(gc)
                    for nr, nc in _neighbors4(rr, cc):
                        if not _in_bounds(chamber.shape, nr, nc) or not bool(chamber[nr, nc]):
                            ok = False
                            break
                    if not ok:
                        break
                if not ok:
                    continue
            windows.append((r, c))
    return windows


def _make_filled_shape() -> tuple[np.ndarray, np.ndarray, np.ndarray]:
    for _ in range(_GEN_SHAPE_ATTEMPTS):
        h = random.randint(6, 9)
        w = random.randint(6, 9)
        mask = np.ones((h, w), dtype=bool)
        cut_count = random.choices((0, 1, 2), weights=(1, 4, 3), k=1)[0]
        for _cut_idx in range(cut_count):
            _random_cut(mask)
        if not bool(mask.any()):
            continue
        boundary = _boundary_from_filled(mask)
        chamber = mask & ~boundary
        if not 12 <= int(boundary.sum()) <= 30:
            continue
        chamber_comps = _components(chamber)
        if len(chamber_comps) != 1:
            continue
        if len(chamber_comps[0]) < 16:
            continue
        if not _glyph_windows(chamber):
            continue
        if len(_components(boundary)) != 1:
            continue
        return mask, boundary, chamber
    raise RuntimeError("failed to build a valid enclosed shape")


def _glyph_patch(kind: str) -> np.ndarray:
    if kind == "cross":
        return _GLYPH_CROSS.copy()
    if kind == "x":
        return _GLYPH_X.copy()
    assert kind == "block"
    patch = np.zeros((3, 3), dtype=bool)
    rr, cc = random.choice(_BLOCK_OFFSETS)
    patch[rr : rr + 2, cc : cc + 2] = True
    return patch


def _choose_corner_window(chamber: np.ndarray, glyph: np.ndarray) -> tuple[int, int]:
    windows = _glyph_windows(chamber, glyph)
    assert windows, "expected at least one valid glyph window"
    chamber_coords = [(int(r), int(c)) for r, c in np.argwhere(chamber)]
    r0, c0, r1, c1 = _bbox_coords(chamber_coords)
    anchors = ((r0, c0), (r0, c1 - 2), (r1 - 2, c0), (r1 - 2, c1 - 2))
    anchor_r, anchor_c = random.choice(anchors)
    best_dist = min(abs(r - anchor_r) + abs(c - anchor_c) for r, c in windows)
    best = [(r, c) for r, c in windows if abs(r - anchor_r) + abs(c - anchor_c) == best_dist]
    return random.choice(best)


def _build_region_patch(kind: str) -> np.ndarray:
    for _ in range(_GEN_SHAPE_ATTEMPTS):
        _filled, boundary, chamber = _make_filled_shape()
        glyph = _glyph_patch(kind)
        windows = _glyph_windows(chamber, glyph)
        if not windows:
            continue
        patch = np.zeros(boundary.shape, dtype=int)
        patch[boundary] = GREEN
        r0, c0 = _choose_corner_window(chamber, glyph)
        region = patch[r0 : r0 + 3, c0 : c0 + 3]
        region[glyph] = GREY
        return patch
    raise RuntimeError("failed to place glyph inside a valid enclosed region")


def _partition_lengths(n_regions: int) -> list[tuple[int, ...]]:
    if n_regions == 2:
        return [(2,), (1, 1)]
    if n_regions == 3:
        return [(3,), (2, 1), (1, 2)]
    return [(2, 2), (3, 1), (1, 3)]


def _sample_region_kinds(n_regions: int) -> list[str]:
    if n_regions == 2:
        return random.sample(["cross", "x", "block"], 2)
    kinds = ["cross", "x", "block"]
    if n_regions == 4:
        kinds.append(random.choice(kinds))
    random.shuffle(kinds)
    return kinds


def _place_patches(patches: list[np.ndarray]) -> np.ndarray | None:
    for _ in range(_GEN_LAYOUT_ATTEMPTS):
        ordered = patches[:]
        random.shuffle(ordered)
        gap_x = random.randint(2, 4)
        gap_y = random.randint(2, 4)
        margins = [random.randint(1, 3) for _ in range(4)]
        top, left, bottom, right = margins
        valid_layouts = []
        for part in _partition_lengths(len(ordered)):
            rows = []
            idx = 0
            for count in part:
                rows.append(ordered[idx : idx + count])
                idx += count
            row_widths = [sum(int(p.shape[1]) for p in row) + gap_x * (len(row) - 1) for row in rows]
            row_heights = [max(int(p.shape[0]) for p in row) for row in rows]
            min_h = top + sum(row_heights) + gap_y * (len(rows) - 1) + bottom
            min_w = left + max(row_widths) + right
            if min_h <= 30 and min_w <= 30:
                valid_layouts.append((rows, row_widths, row_heights, min_h, min_w))
        if not valid_layouts:
            continue

        rows, row_widths, row_heights, min_h, min_w = random.choice(valid_layouts)
        max_h = min(30, max(min_h, 14) + 5)
        max_w = min(30, max(min_w, 16) + 6)
        height = random.randint(max(min_h, 14), max_h)
        width = random.randint(max(min_w, 16), max_w)
        grid = np.zeros((height, width), dtype=int)

        row_top = top
        for row, row_w, row_h in zip(rows, row_widths, row_heights):
            start_c = random.randint(left, width - right - row_w)
            col = start_c
            for patch in row:
                patch_h, patch_w = map(int, patch.shape)
                rr = row_top + random.randint(0, row_h - patch_h)
                view = grid[rr : rr + patch_h, col : col + patch_w]
                fg = patch != BLACK
                if np.any(view[fg] != BLACK):
                    return None
                view[fg] = patch[fg]
                col += patch_w + gap_x
            row_top += row_h + gap_y
        return grid
    return None


def _validate_example(grid: np.ndarray) -> None:
    g = np.asarray(grid, dtype=int)
    assert g.ndim == 2
    assert max(g.shape) <= 30
    assert set(map(int, np.unique(g))) <= {BLACK, GREEN, GREY}
    assert np.all(g[0, :] == BLACK) and np.all(g[-1, :] == BLACK)
    assert np.all(g[:, 0] == BLACK) and np.all(g[:, -1] == BLACK)

    green_components = _components(g == GREEN)
    assert 2 <= len(green_components) <= 4
    for comp in green_components:
        assert 12 <= len(comp) <= 30

    regions, outside = _analyze_regions(g)
    assert len(regions) == len(green_components)
    assert not np.any(outside & (g != BLACK))
    fills = [int(region["fill"]) for region in regions]
    assert len(set(fills)) >= 2
    for region in regions:
        assert int(region["area"]) >= 16
        grey = region["grey_coords"]
        bbox = _bbox_coords(grey)
        glyph_window = np.zeros((bbox[2] - bbox[0] + 1, bbox[3] - bbox[1] + 1), dtype=bool)
        for r, c in grey:
            glyph_window[r - bbox[0], c - bbox[1]] = True
        if len(grey) == 4:
            assert glyph_window.shape == (2, 2)
        else:
            assert glyph_window.shape == (3, 3)
        for r, c in grey:
            for nr, nc in _neighbors4(r, c):
                if _in_bounds(g.shape, nr, nc):
                    assert int(g[nr, nc]) != GREEN, "glyph may not touch the boundary"

    out = main(g)
    assert np.array_equal(out[g == GREEN], g[g == GREEN])
    assert np.array_equal(out[outside], g[outside])
    for region in regions:
        rr = [r for r, _ in region["coords"]]
        cc = [c for _, c in region["coords"]]
        values = {int(out[r, c]) for r, c in region["coords"]}
        assert values == {int(region["fill"])}
        assert int(max(rr) - min(rr) + 1) >= 4
        assert int(max(cc) - min(cc) + 1) >= 4


def _gen_attempt() -> np.ndarray | None:
    n_regions = random.choices(
        [count for count, _weight in _GEN_REGION_COUNT_WEIGHTS],
        weights=[weight for _count, weight in _GEN_REGION_COUNT_WEIGHTS],
        k=1,
    )[0]
    patches = [_build_region_patch(kind) for kind in _sample_region_kinds(n_regions)]
    grid = _place_patches(patches)
    if grid is None:
        return None
    _validate_example(grid)
    return grid


def generate_input() -> np.ndarray:
    for _ in range(_GEN_ATTEMPTS):
        grid = _gen_attempt()
        if grid is not None:
            return grid
    raise RuntimeError("failed to generate a valid enclosed-glyph fill input")
