import random

import numpy as np


BG = 0
SEP = 5
COLOR_POOL = tuple(color for color in range(1, 10) if color != SEP)
NEIGHBORS4 = ((1, 0), (-1, 0), (0, 1), (0, -1))

GRID_ODD_SIZES = tuple(range(15, 30, 2))
MIN_QUADRANT_SIZE = 6
GEN_RETRIES = 500
SHAPE_RETRIES = 80


def _get_bg(grid: np.ndarray) -> int:
    values, counts = np.unique(grid, return_counts=True)
    return int(values[np.argmax(counts)])


def _find_separators(grid: np.ndarray, bg: int) -> tuple[int, int, int]:
    row_hits = [
        (r, int(grid[r, 0]))
        for r in range(grid.shape[0])
        if np.all(grid[r, :] == grid[r, 0]) and int(grid[r, 0]) != int(bg)
    ]
    col_hits = [
        (c, int(grid[0, c]))
        for c in range(grid.shape[1])
        if np.all(grid[:, c] == grid[0, c]) and int(grid[0, c]) != int(bg)
    ]
    assert len(row_hits) == 1 and len(col_hits) == 1, "expected one full separator row and one full separator column"
    sep_row, row_color = row_hits[0]
    sep_col, col_color = col_hits[0]
    assert row_color == col_color, "separator row and column must share one color"
    return int(sep_row), int(sep_col), int(row_color)


def _quadrant_metas(height: int, width: int, sep_row: int, sep_col: int) -> list[dict]:
    top_h = int(sep_row)
    bottom_h = int(height - sep_row - 1)
    left_w = int(sep_col)
    right_w = int(width - sep_col - 1)
    assert min(top_h, bottom_h, left_w, right_w) >= 1
    return [
        {
            "id": 0,
            "row": 0,
            "col": 0,
            "r0": 0,
            "c0": 0,
            "h": top_h,
            "w": left_w,
            "marker": (top_h - 1, left_w - 1),
        },
        {
            "id": 1,
            "row": 0,
            "col": 1,
            "r0": 0,
            "c0": sep_col + 1,
            "h": top_h,
            "w": right_w,
            "marker": (top_h - 1, 0),
        },
        {
            "id": 2,
            "row": 1,
            "col": 0,
            "r0": sep_row + 1,
            "c0": 0,
            "h": bottom_h,
            "w": left_w,
            "marker": (0, left_w - 1),
        },
        {
            "id": 3,
            "row": 1,
            "col": 1,
            "r0": sep_row + 1,
            "c0": sep_col + 1,
            "h": bottom_h,
            "w": right_w,
            "marker": (0, 0),
        },
    ]


def _coords_to_mask(coords: np.ndarray, shape: tuple[int, int]) -> np.ndarray:
    mask = np.zeros(shape, dtype=bool)
    if len(coords):
        mask[coords[:, 0], coords[:, 1]] = True
    return mask


def _is_connected(mask: np.ndarray) -> bool:
    coords = np.argwhere(mask)
    if len(coords) == 0:
        return False
    start = tuple(map(int, coords[0]))
    stack = [start]
    seen = {start}
    while stack:
        r, c = stack.pop()
        for dr, dc in NEIGHBORS4:
            nr = int(r + dr)
            nc = int(c + dc)
            if 0 <= nr < mask.shape[0] and 0 <= nc < mask.shape[1] and mask[nr, nc] and (nr, nc) not in seen:
                seen.add((nr, nc))
                stack.append((nr, nc))
    return len(seen) == int(mask.sum())


def _extract_quadrant(grid: np.ndarray, meta: dict, bg: int, sep_color: int) -> dict:
    region = grid[meta["r0"] : meta["r0"] + meta["h"], meta["c0"] : meta["c0"] + meta["w"]]
    mr, mc = meta["marker"]
    marker_color = int(region[mr, mc])
    assert marker_color not in (int(bg), int(sep_color)), "every quadrant needs a center-corner marker"

    fg_coords = np.argwhere((region != int(bg)) & (region != int(sep_color)))
    assert len(fg_coords) >= 1
    marker_mask = (fg_coords[:, 0] == int(mr)) & (fg_coords[:, 1] == int(mc))
    assert int(marker_mask.sum()) == 1, "marker must be a single pixel at the center-adjacent corner"
    pattern_coords = fg_coords[~marker_mask]

    info = dict(meta)
    info["marker_color"] = marker_color
    if len(pattern_coords) == 0:
        info["pattern_color"] = None
        info["pattern_coords"] = np.zeros((0, 2), dtype=int)
        return info

    pattern_colors = {int(region[r, c]) for r, c in pattern_coords}
    assert len(pattern_colors) == 1, "pattern pixels inside one quadrant must share one color"
    pattern_color = int(next(iter(pattern_colors)))
    pattern_mask = _coords_to_mask(pattern_coords, (meta["h"], meta["w"]))
    assert 3 <= len(pattern_coords) <= 8, "expected a small source pattern"
    assert _is_connected(pattern_mask), "pattern must be 4-connected"

    info["pattern_color"] = pattern_color
    info["pattern_coords"] = pattern_coords.astype(int)
    return info


def _transform_axis(coords: np.ndarray, src_len: int, dst_len: int, sign: int) -> np.ndarray:
    numerators = (int(dst_len) - 1) + int(sign) * (2 * coords - (int(src_len) - 1))
    assert np.all(numerators % 2 == 0), "quadrant parities must allow center-symmetric placement"
    transformed = numerators // 2
    assert np.all((0 <= transformed) & (transformed < int(dst_len))), "mirrored pattern left the target quadrant"
    return transformed.astype(int)


def _transform_coords(coords: np.ndarray, src_meta: dict, dst_meta: dict) -> np.ndarray:
    row_sign = 1 if int(src_meta["row"]) == int(dst_meta["row"]) else -1
    col_sign = 1 if int(src_meta["col"]) == int(dst_meta["col"]) else -1
    out = np.empty_like(coords)
    out[:, 0] = _transform_axis(coords[:, 0], src_meta["h"], dst_meta["h"], row_sign)
    out[:, 1] = _transform_axis(coords[:, 1], src_meta["w"], dst_meta["w"], col_sign)
    return out


def _paint_local(out: np.ndarray, meta: dict, coords: np.ndarray, color: int) -> None:
    region = out[meta["r0"] : meta["r0"] + meta["h"], meta["c0"] : meta["c0"] + meta["w"]]
    region[coords[:, 0], coords[:, 1]] = int(color)


def main(grid: np.ndarray) -> np.ndarray:
    """Mirror-copy each source sprite into the marked target quadrant selected by cross-matched colors.

    Input structure:
    - The background is black.
    - One full grey row and one full grey column split the grid into four rectangular quadrants.
    - Each quadrant has one single-pixel marker at the corner closest to the grid center.
    - Exactly two quadrants also contain a small 4-connected monochrome pattern; the marker color names a source pattern color from another quadrant.

    Rule:
    - Keep the separator lines, markers, and original source patterns unchanged.
    - For each quadrant that only has its center-corner marker, use that marker color to find the source quadrant whose pattern has the same color.
    - Copy that source pattern into the empty quadrant.
    - Mirror the copy left-right when source and target are horizontal neighbors, up-down when they are vertical neighbors, and on both axes when they are diagonal; place it center-symmetrically inside the target quadrant.

    Key insight:
    - The marker does not indicate a local color to extend: it points across quadrants to a source pattern, and the spatial relation between source and target independently determines which mirror axes to use.
    """
    g = np.asarray(grid, dtype=int)
    assert g.ndim == 2, "expected a 2D grid"
    bg = _get_bg(g)
    sep_row, sep_col, sep_color = _find_separators(g, bg)
    metas = _quadrant_metas(g.shape[0], g.shape[1], sep_row, sep_col)
    quadrants = [_extract_quadrant(g, meta, bg, sep_color) for meta in metas]

    sources = [quad for quad in quadrants if quad["pattern_color"] is not None]
    targets = [quad for quad in quadrants if quad["pattern_color"] is None]
    assert len(sources) == 2 and len(targets) == 2, "expected exactly two source quadrants and two empty target quadrants"

    source_by_color = {int(quad["pattern_color"]): quad for quad in sources}
    assert len(source_by_color) == 2, "source pattern colors must be distinct"

    out = g.copy()
    for target in targets:
        marker_color = int(target["marker_color"])
        assert marker_color in source_by_color, "target marker must reference one source pattern color"
        source = source_by_color[marker_color]
        mirrored = _transform_coords(source["pattern_coords"], source, target)
        mr, mc = target["marker"]
        assert not np.any((mirrored[:, 0] == int(mr)) & (mirrored[:, 1] == int(mc))), "copied pattern may not overwrite the target marker"
        _paint_local(out, target, mirrored, int(source["pattern_color"]))
    return out


def _normalize_mask(mask: np.ndarray) -> np.ndarray:
    coords = np.argwhere(mask)
    r0, c0 = coords.min(axis=0)
    r1, c1 = coords.max(axis=0)
    return mask[r0 : r1 + 1, c0 : c1 + 1].copy()


def _random_shape_mask() -> np.ndarray:
    for _ in range(SHAPE_RETRIES):
        area = random.randint(3, 8)
        h = random.randint(2, min(5, area))
        w = random.randint(2, min(5, area))
        if h * w < area:
            continue

        mask = np.zeros((h, w), dtype=bool)
        start = (random.randrange(h), random.randrange(w))
        mask[start] = True
        cells = {start}
        while len(cells) < area:
            frontier: list[tuple[int, int]] = []
            seen = set()
            for r, c in cells:
                for dr, dc in NEIGHBORS4:
                    nr = int(r + dr)
                    nc = int(c + dc)
                    if 0 <= nr < h and 0 <= nc < w and (nr, nc) not in cells and (nr, nc) not in seen:
                        frontier.append((nr, nc))
                        seen.add((nr, nc))
            if not frontier:
                break
            chosen = random.choice(frontier)
            mask[chosen] = True
            cells.add(chosen)

        if int(mask.sum()) != area:
            continue

        mask = _normalize_mask(mask)
        if mask.shape[0] < 2 or mask.shape[1] < 2:
            continue
        if np.array_equal(mask, np.flipud(mask)) or np.array_equal(mask, np.fliplr(mask)):
            continue
        return mask
    raise RuntimeError("failed to sample an asymmetric connected shape")


def _far_from_marker(coords: np.ndarray, marker: tuple[int, int]) -> bool:
    mr, mc = marker
    return all(abs(int(r) - int(mr)) + abs(int(c) - int(mc)) > 1 for r, c in coords)


def _placement_candidates(mask: np.ndarray, src_meta: dict, target_metas: list[dict]) -> list[np.ndarray]:
    base = np.argwhere(mask).astype(int)
    candidates: list[np.ndarray] = []
    for r0 in range(src_meta["h"] - mask.shape[0] + 1):
        for c0 in range(src_meta["w"] - mask.shape[1] + 1):
            placed = base + np.array([r0, c0], dtype=int)
            if not _far_from_marker(placed, src_meta["marker"]):
                continue

            ok = True
            for target_meta in target_metas:
                try:
                    mirrored = _transform_coords(placed, src_meta, target_meta)
                except AssertionError:
                    ok = False
                    break
                if not _far_from_marker(mirrored, target_meta["marker"]):
                    ok = False
                    break
            if ok:
                candidates.append(placed)
    return candidates


def _render_expected(
    inp: np.ndarray,
    metas: list[dict],
    sources: dict[int, dict],
    empty_markers: dict[int, int],
) -> np.ndarray:
    out = inp.copy()
    source_by_color = {int(spec["color"]): qid for qid, spec in sources.items()}
    for qid, marker_color in empty_markers.items():
        src_qid = source_by_color[int(marker_color)]
        mirrored = _transform_coords(sources[src_qid]["coords"], metas[src_qid], metas[qid])
        _paint_local(out, metas[qid], mirrored, int(marker_color))
    return out


def _gen_attempt() -> np.ndarray | None:
    height = random.choice(GRID_ODD_SIZES)
    width = random.choice(GRID_ODD_SIZES)
    sep_row = random.randint(MIN_QUADRANT_SIZE, height - MIN_QUADRANT_SIZE - 1)
    sep_col = random.randint(MIN_QUADRANT_SIZE, width - MIN_QUADRANT_SIZE - 1)
    metas = _quadrant_metas(height, width, sep_row, sep_col)
    if min(meta["h"] for meta in metas) < MIN_QUADRANT_SIZE or min(meta["w"] for meta in metas) < MIN_QUADRANT_SIZE:
        return None

    if random.random() < 0.5:
        source_ids = (0, 3)
        empty_ids = (1, 2)
    else:
        source_ids = (1, 2)
        empty_ids = (0, 3)

    source_colors = random.sample(COLOR_POOL, 2)
    sources: dict[int, dict] = {}
    for qid, color in zip(source_ids, source_colors):
        mask = _random_shape_mask()
        candidates = _placement_candidates(mask, metas[qid], [metas[tid] for tid in empty_ids])
        if not candidates:
            return None
        coords = random.choice(candidates)
        sources[qid] = {"color": int(color), "coords": coords.astype(int)}

    inp = np.full((height, width), BG, dtype=int)
    inp[sep_row, :] = SEP
    inp[:, sep_col] = SEP

    color_by_source = {qid: int(spec["color"]) for qid, spec in sources.items()}
    marker_by_source = {
        source_ids[0]: color_by_source[source_ids[1]],
        source_ids[1]: color_by_source[source_ids[0]],
    }

    for qid in source_ids:
        _paint_local(inp, metas[qid], sources[qid]["coords"], color_by_source[qid])
        mr, mc = metas[qid]["marker"]
        inp[metas[qid]["r0"] + mr, metas[qid]["c0"] + mc] = marker_by_source[qid]

    empty_markers = {
        qid: random.choice(source_colors)
        for qid in empty_ids
    }
    for qid, marker_color in empty_markers.items():
        mr, mc = metas[qid]["marker"]
        inp[metas[qid]["r0"] + mr, metas[qid]["c0"] + mc] = int(marker_color)

    expected = _render_expected(inp, metas, sources, empty_markers)
    try:
        actual = main(inp)
    except AssertionError:
        return None
    if np.array_equal(actual, inp):
        return None
    if not np.array_equal(actual, expected):
        return None
    return inp


def generate_input() -> np.ndarray:
    for _ in range(GEN_RETRIES):
        grid = _gen_attempt()
        if grid is not None:
            return grid
    raise RuntimeError("failed to generate a valid mirrored-quadrant ARC input")
