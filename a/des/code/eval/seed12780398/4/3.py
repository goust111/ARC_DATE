import random
from typing import Dict, Iterable, List, Optional, Sequence, Tuple

import numpy as np


BG = 0
RED = 2
GREY = 5
COLOR_POOL = [1, 3, 4, 6, 7, 8, 9]
MAX_GENERATION_ATTEMPTS = 400


def _separator_rows(grid: np.ndarray) -> List[int]:
    return [r for r in range(grid.shape[0]) if np.all(np.isin(grid[r], (GREY, RED)))]


def _separator_cols(grid: np.ndarray) -> List[int]:
    return [c for c in range(grid.shape[1]) if np.all(np.isin(grid[:, c], (GREY, RED)))]


def _spans(length: int, separators: Sequence[int]) -> List[Tuple[int, int]]:
    spans: List[Tuple[int, int]] = []
    start = 0
    for sep in separators:
        if start <= sep - 1:
            spans.append((start, sep - 1))
        start = sep + 1
    if start <= length - 1:
        spans.append((start, length - 1))
    return spans


def _find_span_index(spans: Sequence[Tuple[int, int]], value: int) -> Optional[int]:
    for index, (lo, hi) in enumerate(spans):
        if lo <= value <= hi:
            return index
    return None


def _region_bbox(
    row_spans: Sequence[Tuple[int, int]],
    col_spans: Sequence[Tuple[int, int]],
    region: Tuple[int, int],
) -> Tuple[int, int, int, int]:
    rr, cc = region
    r0, r1 = row_spans[rr]
    c0, c1 = col_spans[cc]
    return r0, c0, r1, c1


def _region_pixels(grid: np.ndarray, bbox: Tuple[int, int, int, int]) -> List[Tuple[int, int, int]]:
    r0, c0, r1, c1 = bbox
    pts: List[Tuple[int, int, int]] = []
    for r in range(r0, r1 + 1):
        for c in range(c0, c1 + 1):
            color = int(grid[r, c])
            if color not in {BG, RED, GREY}:
                pts.append((r, c, color))
    return pts


def _stamp_mirrored(
    out: np.ndarray,
    src_bbox: Tuple[int, int, int, int],
    tgt_bbox: Tuple[int, int, int, int],
    src_pixels: Iterable[Tuple[int, int, int]],
    kind: str,
    source_on_first_side: bool,
) -> None:
    sr0, sc0, sr1, sc1 = src_bbox
    tr0, tc0, tr1, tc1 = tgt_bbox

    for r, c, color in src_pixels:
        if kind == "vertical":
            nr = r
            if source_on_first_side:
                nc = tc0 + (sc1 - c)
            else:
                nc = tc1 - (c - sc0)
        else:
            nc = c
            if source_on_first_side:
                nr = tr0 + (sr1 - r)
            else:
                nr = tr1 - (r - sr0)
        if tr0 <= nr <= tr1 and tc0 <= nc <= tc1:
            out[nr, nc] = color


def main(grid: np.ndarray) -> np.ndarray:
    """
    Mirror-stamp each patterned region into its adjacent blank region across red-marked separators.
    """
    g = np.asarray(grid, dtype=int)
    assert g.ndim == 2 and g.size > 0

    sep_rows = _separator_rows(g)
    sep_cols = _separator_cols(g)
    row_spans = _spans(g.shape[0], sep_rows)
    col_spans = _spans(g.shape[1], sep_cols)
    assert row_spans and col_spans

    sep_row_to_index = {value: idx for idx, value in enumerate(sep_rows)}
    sep_col_to_index = {value: idx for idx, value in enumerate(sep_cols)}
    region_cache: Dict[Tuple[int, int], List[Tuple[int, int, int]]] = {}

    def region_pixels(region: Tuple[int, int]) -> List[Tuple[int, int, int]]:
        if region not in region_cache:
            region_cache[region] = _region_pixels(g, _region_bbox(row_spans, col_spans, region))
        return region_cache[region]

    markers = [tuple(map(int, rc)) for rc in np.argwhere(g == RED)]
    markers.sort()
    out = g.copy()

    for r, c in markers:
        on_sep_row = r in sep_row_to_index
        on_sep_col = c in sep_col_to_index
        if on_sep_row == on_sep_col:
            continue

        if on_sep_col:
            row_index = _find_span_index(row_spans, r)
            sep_index = sep_col_to_index[c]
            if row_index is None or not (0 <= sep_index < len(col_spans)):
                continue
            if sep_index >= len(col_spans) - 1:
                continue
            left_region = (row_index, sep_index)
            right_region = (row_index, sep_index + 1)
            left_pixels = region_pixels(left_region)
            right_pixels = region_pixels(right_region)
            if bool(left_pixels) == bool(right_pixels):
                continue
            if left_pixels:
                src_region, tgt_region = left_region, right_region
                src_pixels = left_pixels
                source_on_first_side = True
            else:
                src_region, tgt_region = right_region, left_region
                src_pixels = right_pixels
                source_on_first_side = False
            _stamp_mirrored(
                out,
                _region_bbox(row_spans, col_spans, src_region),
                _region_bbox(row_spans, col_spans, tgt_region),
                src_pixels,
                kind="vertical",
                source_on_first_side=source_on_first_side,
            )
        else:
            col_index = _find_span_index(col_spans, c)
            sep_index = sep_row_to_index[r]
            if col_index is None or not (0 <= sep_index < len(row_spans)):
                continue
            if sep_index >= len(row_spans) - 1:
                continue
            top_region = (sep_index, col_index)
            bottom_region = (sep_index + 1, col_index)
            top_pixels = region_pixels(top_region)
            bottom_pixels = region_pixels(bottom_region)
            if bool(top_pixels) == bool(bottom_pixels):
                continue
            if top_pixels:
                src_region, tgt_region = top_region, bottom_region
                src_pixels = top_pixels
                source_on_first_side = True
            else:
                src_region, tgt_region = bottom_region, top_region
                src_pixels = bottom_pixels
                source_on_first_side = False
            _stamp_mirrored(
                out,
                _region_bbox(row_spans, col_spans, src_region),
                _region_bbox(row_spans, col_spans, tgt_region),
                src_pixels,
                kind="horizontal",
                source_on_first_side=source_on_first_side,
            )

    return out


def _make_partition_grid(rows: int, cols: int, cell_h: int, cell_w: int) -> Tuple[np.ndarray, List[Tuple[int, int]], List[Tuple[int, int]]]:
    height = rows * cell_h + (rows - 1)
    width = cols * cell_w + (cols - 1)
    grid = np.full((height, width), BG, dtype=int)

    row_spans: List[Tuple[int, int]] = []
    col_spans: List[Tuple[int, int]] = []
    for rr in range(rows):
        r0 = rr * (cell_h + 1)
        row_spans.append((r0, r0 + cell_h - 1))
    for cc in range(cols):
        c0 = cc * (cell_w + 1)
        col_spans.append((c0, c0 + cell_w - 1))

    for rr in range(rows - 1):
        grid[row_spans[rr][1] + 1, :] = GREY
    for cc in range(cols - 1):
        grid[:, col_spans[cc][1] + 1] = GREY
    return grid, row_spans, col_spans


def _neighbors(region: Tuple[int, int], rows: int, cols: int) -> Dict[str, Tuple[int, int]]:
    r, c = region
    result: Dict[str, Tuple[int, int]] = {}
    if r > 0:
        result["up"] = (r - 1, c)
    if r + 1 < rows:
        result["down"] = (r + 1, c)
    if c > 0:
        result["left"] = (r, c - 1)
    if c + 1 < cols:
        result["right"] = (r, c + 1)
    return result


def _random_connected_shape(
    rng: random.Random,
    box_h: int,
    box_w: int,
    min_cells: int = 2,
) -> np.ndarray:
    total = box_h * box_w
    target = rng.randint(min_cells, min(total, max(min_cells, total - 1)))
    coords = {(rng.randrange(box_h), rng.randrange(box_w))}
    while len(coords) < target:
        r, c = rng.choice(tuple(coords))
        candidates = []
        for nr, nc in ((r - 1, c), (r + 1, c), (r, c - 1), (r, c + 1)):
            if 0 <= nr < box_h and 0 <= nc < box_w and (nr, nc) not in coords:
                candidates.append((nr, nc))
        if candidates:
            coords.add(rng.choice(candidates))
        else:
            coords.add((rng.randrange(box_h), rng.randrange(box_w)))
    mask = np.zeros((box_h, box_w), dtype=bool)
    for r, c in coords:
        mask[r, c] = True
    return mask


def _place_random_pattern(
    rng: random.Random,
    cell_h: int,
    cell_w: int,
    row_range: Tuple[int, int],
    col_range: Tuple[int, int],
) -> List[Tuple[int, int]]:
    r_lo, r_hi = row_range
    c_lo, c_hi = col_range
    max_h = min(3, r_hi - r_lo + 1)
    max_w = min(3, c_hi - c_lo + 1)
    box_h = rng.randint(2, max_h)
    box_w = rng.randint(2, max_w)
    row_start = rng.randint(r_lo, r_hi - box_h + 1)
    col_start = rng.randint(c_lo, c_hi - box_w + 1)
    mask = _random_connected_shape(rng, box_h, box_w)
    pts: List[Tuple[int, int]] = []
    for rr in range(box_h):
        for cc in range(box_w):
            if mask[rr, cc]:
                pts.append((row_start + rr, col_start + cc))
    return pts


def _full_region_pattern(rng: random.Random, cell_h: int, cell_w: int) -> List[Tuple[int, int]]:
    row_margin = rng.randint(0, max(1, cell_h - 4))
    col_margin = rng.randint(0, max(1, cell_w - 4))
    row_lo = row_margin
    row_hi = cell_h - 1 - max(0, row_margin - 1)
    col_lo = col_margin
    col_hi = cell_w - 1 - max(0, col_margin - 1)
    row_hi = max(row_lo + 1, row_hi)
    col_hi = max(col_lo + 1, col_hi)
    return _place_random_pattern(rng, cell_h, cell_w, (row_lo, row_hi), (col_lo, col_hi))


def _sector_ranges(cell_h: int, cell_w: int, sector: str) -> Tuple[Tuple[int, int], Tuple[int, int]]:
    top_rows = (0, max(1, cell_h // 2) - 1)
    bottom_rows = (cell_h - max(2, cell_h // 2), cell_h - 1)
    left_cols = (0, max(1, cell_w // 2) - 1)
    right_cols = (cell_w - max(2, cell_w // 2), cell_w - 1)

    if sector == "tl":
        return top_rows, left_cols
    if sector == "tr":
        return top_rows, right_cols
    if sector == "bl":
        return bottom_rows, left_cols
    return bottom_rows, right_cols


def _invert_target_cells(
    target_cells: Sequence[Tuple[int, int]],
    source_region: Tuple[int, int],
    target_region: Tuple[int, int],
    cell_h: int,
    cell_w: int,
) -> Tuple[List[Tuple[int, int]], str]:
    src_r, src_c = source_region
    tgt_r, tgt_c = target_region
    if src_r == tgt_r:
        kind = "vertical"
        result = []
        for tr, tc in target_cells:
            sr = tr
            if src_c < tgt_c:
                sc = cell_w - 1 - tc
            else:
                sc = cell_w - 1 - tc
            result.append((sr, sc))
        return result, kind

    kind = "horizontal"
    result = []
    for tr, tc in target_cells:
        sc = tc
        sr = cell_h - 1 - tr
        result.append((sr, sc))
    return result, kind


def _absolute_cells(
    relative_cells: Sequence[Tuple[int, int]],
    row_spans: Sequence[Tuple[int, int]],
    col_spans: Sequence[Tuple[int, int]],
    region: Tuple[int, int],
) -> List[Tuple[int, int]]:
    r0, c0, _r1, _c1 = _region_bbox(row_spans, col_spans, region)
    return [(r0 + rr, c0 + cc) for rr, cc in relative_cells]


def _paint_cells(grid: np.ndarray, coords: Sequence[Tuple[int, int]], color: int) -> None:
    for r, c in coords:
        grid[r, c] = color


def _marker_position(
    row_spans: Sequence[Tuple[int, int]],
    col_spans: Sequence[Tuple[int, int]],
    source: Tuple[int, int],
    target: Tuple[int, int],
    target_cells: Sequence[Tuple[int, int]],
) -> Tuple[int, int]:
    sr, sc = source
    tr, tc = target
    tgt_bbox = _region_bbox(row_spans, col_spans, target)
    if sr == tr:
        r0, _c0, _r1, _c1 = tgt_bbox
        sep_c = max(_region_bbox(row_spans, col_spans, source)[1], tgt_bbox[1]) - 1
        if sc < tc:
            sep_c = _region_bbox(row_spans, col_spans, source)[3] + 1
        else:
            sep_c = tgt_bbox[3] + 1
        avg_row = int(round(sum(r0 + rr for rr, _cc in target_cells) / len(target_cells)))
        avg_row = min(max(avg_row, tgt_bbox[0] + 1), tgt_bbox[2] - 1)
        return avg_row, sep_c

    sep_r = _region_bbox(row_spans, col_spans, source)[2] + 1 if sr < tr else tgt_bbox[2] + 1
    avg_col = int(round(sum(tgt_bbox[1] + cc for _rr, cc in target_cells) / len(target_cells)))
    avg_col = min(max(avg_col, tgt_bbox[1] + 1), tgt_bbox[3] - 1)
    return sep_r, avg_col


def _edge_kind(source: Tuple[int, int], target: Tuple[int, int]) -> str:
    return "vertical" if source[0] == target[0] else "horizontal"


def _mirrored_target_rel(
    source_rel: Sequence[Tuple[int, int]],
    source: Tuple[int, int],
    target: Tuple[int, int],
    cell_h: int,
    cell_w: int,
) -> List[Tuple[int, int]]:
    if source[0] == target[0]:
        return [(rr, cell_w - 1 - cc) for rr, cc in source_rel]
    return [(cell_h - 1 - rr, cc) for rr, cc in source_rel]


def _build_multi_target_case(
    rng: random.Random,
    rows: int,
    cols: int,
) -> Tuple[Tuple[int, int], List[Tuple[int, int]]]:
    candidates = [
        (r, c)
        for r in range(rows)
        for c in range(cols)
        if (_neighbors((r, c), rows, cols).keys() & {"up", "down"})
        and (_neighbors((r, c), rows, cols).keys() & {"left", "right"})
    ]
    target = rng.choice(candidates)
    neigh = _neighbors(target, rows, cols)
    vertical_source = rng.choice([neigh[d] for d in ("up", "down") if d in neigh])
    horizontal_source = rng.choice([neigh[d] for d in ("left", "right") if d in neigh])
    return target, [vertical_source, horizontal_source]


def _build_single_source_case(
    rng: random.Random,
    rows: int,
    cols: int,
) -> Tuple[Tuple[int, int], List[Tuple[int, int]]]:
    candidates = [
        (r, c)
        for r in range(rows)
        for c in range(cols)
        if (_neighbors((r, c), rows, cols).keys() & {"up", "down"})
        and (_neighbors((r, c), rows, cols).keys() & {"left", "right"})
    ]
    source = rng.choice(candidates)
    neigh = _neighbors(source, rows, cols)
    vertical_target = rng.choice([neigh[d] for d in ("up", "down") if d in neigh])
    horizontal_target = rng.choice([neigh[d] for d in ("left", "right") if d in neigh])
    return source, [vertical_target, horizontal_target]


def _gen_attempt(rng: random.Random) -> Optional[np.ndarray]:
    rows = rng.randint(2, 4)
    cols = rng.randint(2, 4)
    cell_h = rng.randint(5, 7)
    cell_w = rng.randint(5, 7)
    grid, row_spans, col_spans = _make_partition_grid(rows, cols, cell_h, cell_w)

    branch_multi = rng.random() < 0.5
    source_colors: Dict[Tuple[int, int], int] = {}
    source_patterns: Dict[Tuple[int, int], List[Tuple[int, int]]] = {}
    used_targets: set[Tuple[int, int]] = set()
    edges: List[Dict[str, object]] = []

    if branch_multi:
        target, sources = _build_multi_target_case(rng, rows, cols)
        used_targets.add(target)
        sectors = rng.sample(["tl", "tr", "bl", "br"], k=2)
        colors = rng.sample(COLOR_POOL, k=2)
        occupied_target_rel: set[Tuple[int, int]] = set()
        for source, sector, color in zip(sources, sectors, colors, strict=False):
            row_range, col_range = _sector_ranges(cell_h, cell_w, sector)
            target_rel = _place_random_pattern(rng, cell_h, cell_w, row_range, col_range)
            if occupied_target_rel & set(target_rel):
                return None
            source_rel, kind = _invert_target_cells(target_rel, source, target, cell_h, cell_w)
            source_abs = _absolute_cells(source_rel, row_spans, col_spans, source)
            if any(grid[r, c] != BG for r, c in source_abs):
                return None
            _paint_cells(grid, source_abs, color)
            occupied_target_rel.update(target_rel)
            source_colors[source] = color
            source_patterns[source] = source_rel
            edges.append(
                {
                    "source": source,
                    "target": target,
                    "kind": kind,
                    "target_rel": target_rel,
                    "marker": _marker_position(row_spans, col_spans, source, target, target_rel),
                }
            )
    else:
        source, targets = _build_single_source_case(rng, rows, cols)
        color = rng.choice(COLOR_POOL)
        source_rel = _full_region_pattern(rng, cell_h, cell_w)
        source_abs = _absolute_cells(source_rel, row_spans, col_spans, source)
        _paint_cells(grid, source_abs, color)
        source_colors[source] = color
        source_patterns[source] = source_rel
        for target in targets:
            used_targets.add(target)
            target_rel = _mirrored_target_rel(source_rel, source, target, cell_h, cell_w)
            edges.append(
                {
                    "source": source,
                    "target": target,
                    "kind": _edge_kind(source, target),
                    "target_rel": target_rel,
                    "marker": _marker_position(
                        row_spans,
                        col_spans,
                        source,
                        target,
                        target_rel,
                    ),
                }
            )

    marker_cells = set()
    for edge in edges:
        mr, mc = edge["marker"]  # type: ignore[index]
        if (mr, mc) in marker_cells:
            return None
        marker_cells.add((mr, mc))
        grid[mr, mc] = RED

    if not np.any(grid == RED):
        return None
    expected = grid.copy()
    for edge in sorted(edges, key=lambda item: item["marker"]):  # type: ignore[index]
        source = edge["source"]  # type: ignore[index]
        target = edge["target"]  # type: ignore[index]
        target_rel = edge["target_rel"]  # type: ignore[index]
        color = source_colors[source]
        target_abs = _absolute_cells(target_rel, row_spans, col_spans, target)
        _paint_cells(expected, target_abs, color)

    solved = main(grid)
    if np.array_equal(solved, grid):
        return None
    if not np.array_equal(solved, expected):
        return None
    return grid


def generate_input() -> np.ndarray:
    rng = random.Random()
    for _ in range(MAX_GENERATION_ATTEMPTS):
        grid = _gen_attempt(rng)
        if grid is not None:
            return grid
    raise RuntimeError("failed to generate a valid mirror-stamp puzzle")
