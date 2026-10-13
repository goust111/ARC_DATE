from __future__ import annotations

import random
from itertools import combinations

import numpy as np


BLACK = 0
BLUE = 1
RED = 2
GREEN = 3
YELLOW = 4
GREY = 5
PINK = 6
ORANGE = 7

MARKER_COLORS = (BLUE, YELLOW, PINK, ORANGE)
SHAPE_HEIGHTS = (2, 4)
MAX_AREA = 8
MIN_AREA = 3
GEN_ATTEMPTS = 400

_SHAPE_LIBRARY: dict[bool, list[np.ndarray]] | None = None


def _bbox(mask: np.ndarray) -> tuple[int, int, int, int] | None:
    coords = np.argwhere(mask)
    if coords.size == 0:
        return None
    r0, c0 = coords.min(axis=0)
    r1, c1 = coords.max(axis=0)
    return int(r0), int(c0), int(r1), int(c1)


def _neighbors4(r: int, c: int, h: int, w: int):
    for dr, dc in ((-1, 0), (1, 0), (0, -1), (0, 1)):
        nr = r + dr
        nc = c + dc
        if 0 <= nr < h and 0 <= nc < w:
            yield nr, nc


def _is_connected4(mask: np.ndarray) -> bool:
    coords = np.argwhere(mask)
    if coords.size == 0:
        return False
    h, w = mask.shape
    start = tuple(map(int, coords[0]))
    stack = [start]
    seen = {start}
    while stack:
        r, c = stack.pop()
        for nr, nc in _neighbors4(r, c, h, w):
            if mask[nr, nc] and (nr, nc) not in seen:
                seen.add((nr, nc))
                stack.append((nr, nc))
    return len(seen) == int(mask.sum())


def _mask_ud_symmetric(mask: np.ndarray) -> bool:
    h = mask.shape[0]
    if h % 2 != 0:
        return False
    mid = h // 2
    return bool(np.array_equal(mask[:mid][::-1], mask[mid:]))


def _recolor_crop(crop: np.ndarray) -> tuple[np.ndarray, bool]:
    grey_mask = crop == GREY
    b = _bbox(grey_mask)
    assert b is not None, "expected a grey object inside the cropped region"
    r0, c0, r1, c1 = b
    shape_mask = grey_mask[r0 : r1 + 1, c0 : c1 + 1]
    height = shape_mask.shape[0]
    assert height in SHAPE_HEIGHTS and height % 2 == 0
    symmetric = _mask_ud_symmetric(shape_mask)

    out = crop.copy()
    split_row = r0 + height // 2
    lower_rows = np.arange(crop.shape[0])[:, None] >= split_row
    out[grey_mask & lower_rows] = RED if symmetric else GREEN
    return out, symmetric


def _extract_clusters(grid: np.ndarray) -> list[dict]:
    g = np.asarray(grid, dtype=int)
    assert g.ndim == 2, "expected a 2D grid"

    cluster_colors = sorted(set(map(int, g.flatten())) - {BLACK, GREY})
    assert 3 <= len(cluster_colors) <= 4, "expected 3 or 4 corner-marker colors"

    clusters: list[dict] = []
    grey_covered = np.zeros_like(g, dtype=bool)
    for color in cluster_colors:
        coords = np.argwhere(g == color)
        assert len(coords) == 4, "each marker color must appear exactly four times"

        rows = sorted({int(r) for r in coords[:, 0]})
        cols = sorted({int(c) for c in coords[:, 1]})
        assert len(rows) == 2 and len(cols) == 2, "markers must define a rectangle"

        expected = {
            (rows[0], cols[0]),
            (rows[0], cols[1]),
            (rows[1], cols[0]),
            (rows[1], cols[1]),
        }
        actual = {tuple(map(int, rc)) for rc in coords}
        assert actual == expected, "marker pixels must sit exactly at the four corners"

        r0, r1 = rows
        c0, c1 = cols
        assert r1 - r0 >= 2 and c1 - c0 >= 2, "marker rectangle must enclose a non-empty interior"

        crop = g[r0 + 1 : r1, c0 + 1 : c1].copy()
        assert crop.size > 0
        assert np.all(np.isin(crop, (BLACK, GREY))), "cropped interior must contain only background and the grey object"

        grey_mask = crop == GREY
        assert grey_mask.any(), "each cluster must contain a central grey object"
        assert _is_connected4(grey_mask), "grey object must be 4-connected"

        b = _bbox(grey_mask)
        assert b is not None
        sr0, sc0, sr1, sc1 = b
        assert sr0 > 0 and sc0 > 0 and sr1 < crop.shape[0] - 1 and sc1 < crop.shape[1] - 1, "grey object must not touch the cropped border"
        shape_h = sr1 - sr0 + 1
        assert shape_h in SHAPE_HEIGHTS, "grey object bbox height must be 2 or 4"

        processed, symmetric = _recolor_crop(crop)
        grey_covered[r0 + 1 : r1, c0 + 1 : c1] |= grey_mask
        clusters.append(
            {
                "marker_color": color,
                "outer_bbox": (r0, c0, r1, c1),
                "crop": crop,
                "processed": processed,
                "symmetric": symmetric,
            }
        )

    stray_grey = (g == GREY) & ~grey_covered
    assert not stray_grey.any(), "all grey pixels must belong to one of the marked clusters"

    raw_keys = {(tuple(rec["crop"].shape), rec["crop"].tobytes()) for rec in clusters}
    assert len(raw_keys) == len(clusters), "clusters must remain distinguishable after cropping"
    return sorted(clusters, key=lambda rec: (rec["outer_bbox"][1], rec["outer_bbox"][0]))


def _assemble_panels(panels: list[np.ndarray]) -> np.ndarray:
    max_h = max(panel.shape[0] for panel in panels)
    total_w = sum(panel.shape[1] for panel in panels)
    out = np.full((max_h, total_w), BLACK, dtype=int)
    c = 0
    for panel in panels:
        h, w = panel.shape
        out[max_h - h :, c : c + w] = panel
        c += w
    return out


def main(grid: np.ndarray) -> np.ndarray:
    """
    Crop the interior of each corner-marked cluster, test the grey shape for up-down symmetry, and recolor its lower half accordingly.

    Input structure:
    - The grid has a black background with 3 or 4 disjoint clusters.
    - Each cluster is defined by four same-color single-pixel corner markers and one 4-connected grey object inside their rectangle.
    - The grey object's bbox has even height, and the object does not touch the marker-bounded interior border.

    Rule:
    - For each marker color, use its four pixels as the corners of a rectangle and crop the interior region between them.
    - Within that crop, find the grey object's own bbox and test whether its upper half flipped vertically matches the lower half.
    - If symmetric, recolor only the grey pixels in the lower half of that bbox red; otherwise recolor those lower-half grey pixels green.
    - Sort the processed crops by the original corner rectangle's left column and concatenate them horizontally, top-padding shorter crops so their bottoms align.

    Key insight:
    - The corner markers only define which local window to crop; the decisive property is the grey object's symmetry inside that crop, and that boolean drives a spatial recoloring rather than a single indicator pixel.
    """
    clusters = _extract_clusters(np.asarray(grid, dtype=int))
    return _assemble_panels([rec["processed"] for rec in clusters])


def _build_shape_library() -> dict[bool, list[np.ndarray]]:
    lib: dict[bool, list[np.ndarray]] = {True: [], False: []}
    for h in SHAPE_HEIGHTS:
        for w in range(2, 6):
            cell_count = h * w
            for area in range(MIN_AREA, min(MAX_AREA, cell_count) + 1):
                for combo in combinations(range(cell_count), area):
                    mask = np.zeros((h, w), dtype=bool)
                    flat = mask.reshape(-1)
                    flat[list(combo)] = True
                    if _bbox(mask) != (0, 0, h - 1, w - 1):
                        continue
                    if not _is_connected4(mask):
                        continue
                    lib[_mask_ud_symmetric(mask)].append(mask.copy())
    assert lib[True] and lib[False]
    return lib


def _shape_library() -> dict[bool, list[np.ndarray]]:
    global _SHAPE_LIBRARY
    if _SHAPE_LIBRARY is None:
        _SHAPE_LIBRARY = _build_shape_library()
    return _SHAPE_LIBRARY


def _sample_shape(symmetric: bool) -> np.ndarray:
    return random.choice(_shape_library()[symmetric]).copy()


def _centered_offset(extra: int) -> int:
    if extra <= 0:
        return 0
    base = extra // 2
    return random.choice(sorted({base, extra - base}))


def _make_cluster_patch(marker_color: int, symmetric: bool) -> dict:
    shape = _sample_shape(symmetric)
    shape_h, shape_w = shape.shape

    inner_h = random.randint(shape_h + 2, min(8, shape_h + 4))
    inner_w = random.randint(shape_w + 2, min(7, shape_w + 4))

    extra_h = inner_h - shape_h - 2
    extra_w = inner_w - shape_w - 2
    top_margin = 1 + _centered_offset(extra_h)
    left_margin = 1 + _centered_offset(extra_w)

    crop = np.full((inner_h, inner_w), BLACK, dtype=int)
    crop[top_margin : top_margin + shape_h, left_margin : left_margin + shape_w][shape] = GREY

    patch = np.full((inner_h + 2, inner_w + 2), BLACK, dtype=int)
    patch[0, 0] = marker_color
    patch[0, -1] = marker_color
    patch[-1, 0] = marker_color
    patch[-1, -1] = marker_color
    patch[1:-1, 1:-1] = crop

    processed, detected = _recolor_crop(crop)
    assert detected == symmetric
    return {
        "patch": patch,
        "crop": crop,
        "processed": processed,
        "symmetric": symmetric,
    }


def _rects_clear(rects: list[tuple[int, int, int, int]], cand: tuple[int, int, int, int], padding: int = 1) -> bool:
    r0, c0, r1, c1 = cand
    for ar0, ac0, ar1, ac1 in rects:
        if not (
            r1 + padding < ar0
            or ar1 + padding < r0
            or c1 + padding < ac0
            or ac1 + padding < c0
        ):
            return False
    return True


def _place_patches(grid_h: int, grid_w: int, patches: list[np.ndarray]) -> np.ndarray | None:
    grid = np.full((grid_h, grid_w), BLACK, dtype=int)
    rects: list[tuple[int, int, int, int]] = []
    order = list(range(len(patches)))
    random.shuffle(order)
    for idx in order:
        patch = patches[idx]
        h, w = patch.shape
        candidates = []
        for r0 in range(0, grid_h - h + 1):
            for c0 in range(0, grid_w - w + 1):
                rect = (r0, c0, r0 + h - 1, c0 + w - 1)
                if _rects_clear(rects, rect, padding=1):
                    candidates.append((r0, c0))
        if not candidates:
            return None
        r0, c0 = random.choice(candidates)
        grid[r0 : r0 + h, c0 : c0 + w] = patch
        rects.append((r0, c0, r0 + h - 1, c0 + w - 1))
    return grid


def _semantic_ok(grid: np.ndarray, expected: np.ndarray, want_flags: list[bool]) -> bool:
    try:
        clusters = _extract_clusters(grid)
        solved = main(grid)
    except AssertionError:
        return False

    if not np.array_equal(solved, expected):
        return False
    if [rec["symmetric"] for rec in clusters] != want_flags:
        return False
    if not any(want_flags) or all(want_flags):
        return False

    panel_keys = {
        (tuple(rec["processed"].shape), rec["processed"].tobytes()) for rec in clusters
    }
    if len(panel_keys) != len(clusters):
        return False

    heights = {rec["crop"].shape[0] for rec in clusters}
    if len(heights) < 2:
        return False

    if max(grid.shape) > 30 or max(expected.shape) > 30:
        return False
    return True


def _attempt_generate() -> np.ndarray | None:
    cluster_count = random.randint(3, 4)
    marker_colors = random.sample(MARKER_COLORS, cluster_count)

    sym_count = random.randint(1, cluster_count - 1)
    sym_flags = [True] * sym_count + [False] * (cluster_count - sym_count)
    random.shuffle(sym_flags)

    clusters = [_make_cluster_patch(color, flag) for color, flag in zip(marker_colors, sym_flags)]

    raw_keys = {(tuple(rec["crop"].shape), rec["crop"].tobytes()) for rec in clusters}
    out_keys = {(tuple(rec["processed"].shape), rec["processed"].tobytes()) for rec in clusters}
    if len(raw_keys) != cluster_count or len(out_keys) != cluster_count:
        return None

    patches = [rec["patch"] for rec in clusters]
    max_patch_h = max(p.shape[0] for p in patches)
    total_patch_w = sum(p.shape[1] for p in patches)
    grid_h = random.randint(max(18, max_patch_h + 4), 26)
    grid_w = random.randint(max(18, max(total_patch_w // 2 + 3, max(p.shape[1] for p in patches) + 4)), 26)

    grid = _place_patches(grid_h, grid_w, patches)
    if grid is None:
        return None

    extracted = _extract_clusters(grid)
    expected = _assemble_panels([rec["processed"] for rec in extracted])
    want_flags = [rec["symmetric"] for rec in extracted]
    if not _semantic_ok(grid, expected, want_flags):
        return None
    return grid


def generate_input() -> np.ndarray:
    for _ in range(GEN_ATTEMPTS):
        grid = _attempt_generate()
        if grid is not None:
            return grid
    raise RuntimeError("failed to generate a valid input for puzzle 3")
