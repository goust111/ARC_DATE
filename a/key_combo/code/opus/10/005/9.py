from __future__ import annotations

import hashlib
import random
from typing import Iterable

import numpy as np


BLACK = 0
GREY = 5
SHAPE_PALETTE = (1, 2, 3, 4, 6, 7)
MAX_GENERATION_ATTEMPTS = 500
_SEEN_OUTPUT_SIGNATURES: set[str] = set()

_BASE_SHAPES = (
    ((0, 0), (1, 0), (2, 0)),
    ((0, 0), (0, 1), (1, 0)),
    ((0, 0), (0, 1), (1, 0), (1, 1)),
    ((0, 0), (1, 0), (2, 0), (2, 1)),
    ((0, 1), (1, 0), (1, 1), (1, 2)),
    ((0, 0), (0, 1), (0, 2), (1, 1)),
    ((0, 0), (1, 0), (1, 1), (2, 1)),
    ((0, 1), (1, 0), (1, 1), (1, 2), (2, 1)),
    ((0, 0), (1, 0), (1, 1), (1, 2), (2, 2)),
    ((0, 0), (0, 1), (1, 1), (2, 0), (2, 1)),
    ((0, 0), (1, 0), (2, 0), (2, 1), (2, 2)),
    ((0, 0), (0, 1), (0, 2), (1, 0), (2, 0)),
    ((0, 0), (0, 1), (0, 2), (1, 1), (2, 1)),
    ((0, 0), (0, 1), (0, 2), (1, 0), (1, 1), (1, 2)),
    ((0, 0), (0, 1), (1, 1), (1, 2), (2, 1), (2, 2)),
)


def _normalize_cells(cells: Iterable[tuple[int, int]]) -> tuple[tuple[int, int], ...]:
    pts = sorted((int(r), int(c)) for r, c in cells)
    r0 = min(r for r, _ in pts)
    c0 = min(c for _, c in pts)
    return tuple(sorted((r - r0, c - c0) for r, c in pts))


def _transform_shape(shape: tuple[tuple[int, int], ...]) -> tuple[tuple[int, int], ...]:
    h = max(r for r, _ in shape) + 1
    w = max(c for _, c in shape) + 1
    arr = np.zeros((h, w), dtype=int)
    for r, c in shape:
        arr[r, c] = 1
    if random.random() < 0.5:
        arr = np.fliplr(arr)
    arr = np.rot90(arr, k=random.randrange(4))
    return _normalize_cells(np.argwhere(arr == 1))


def _neighbors4(r: int, c: int) -> tuple[tuple[int, int], ...]:
    return ((r - 1, c), (r + 1, c), (r, c - 1), (r, c + 1))


def _components(mask: np.ndarray, connectivity: int = 4) -> list[np.ndarray]:
    h, w = mask.shape
    seen = np.zeros_like(mask, dtype=bool)
    out: list[np.ndarray] = []
    if connectivity == 4:
        steps = ((-1, 0), (1, 0), (0, -1), (0, 1))
    else:
        steps = tuple((dr, dc) for dr in (-1, 0, 1) for dc in (-1, 0, 1) if dr or dc)
    for r in range(h):
        for c in range(w):
            if not mask[r, c] or seen[r, c]:
                continue
            stack = [(r, c)]
            seen[r, c] = True
            coords: list[tuple[int, int]] = []
            while stack:
                rr, cc = stack.pop()
                coords.append((rr, cc))
                for dr, dc in steps:
                    nr, nc = rr + dr, cc + dc
                    if 0 <= nr < h and 0 <= nc < w and mask[nr, nc] and not seen[nr, nc]:
                        seen[nr, nc] = True
                        stack.append((nr, nc))
            out.append(np.array(coords, dtype=int))
    return out


def _separator_indices(grid: np.ndarray) -> tuple[list[int], list[int]]:
    sep_rows = [r for r in range(grid.shape[0]) if np.all(grid[r, :] == GREY)]
    sep_cols = [c for c in range(grid.shape[1]) if np.all(grid[:, c] == GREY)]
    return sep_rows, sep_cols


def _segments(size: int, separators: list[int]) -> list[tuple[int, int]]:
    pts = [-1] + sorted(int(x) for x in separators) + [size]
    return [(pts[i] + 1, pts[i + 1] - 1) for i in range(len(pts) - 1) if pts[i] + 1 <= pts[i + 1] - 1]


def _region_infos_from_grid(grid: np.ndarray) -> list[dict]:
    sep_rows, sep_cols = _separator_indices(grid)
    row_segments = _segments(grid.shape[0], sep_rows)
    col_segments = _segments(grid.shape[1], sep_cols)
    regions = []
    for i, (r0, r1) in enumerate(row_segments):
        for j, (c0, c1) in enumerate(col_segments):
            regions.append(
                {
                    "index": len(regions),
                    "r0": int(r0),
                    "r1": int(r1),
                    "c0": int(c0),
                    "c1": int(c1),
                    "h": int(r1 - r0 + 1),
                    "w": int(c1 - c0 + 1),
                    "top_sep": i > 0,
                    "bottom_sep": i < len(row_segments) - 1,
                    "left_sep": j > 0,
                    "right_sep": j < len(col_segments) - 1,
                }
            )
    return regions


def _extract_bar(region: np.ndarray) -> dict:
    comps = _components(region == GREY, connectivity=4)
    assert len(comps) == 1, "each region must contain exactly one gray bar"
    coords = comps[0]
    rows = {int(r) for r, _ in coords}
    cols = {int(c) for _, c in coords}
    if len(rows) == 1 and len(cols) > 1:
        orient = "H"
    elif len(cols) == 1 and len(rows) > 1:
        orient = "V"
    else:
        raise AssertionError("gray bar must be a non-degenerate straight segment")
    return {"coords": coords, "length": int(coords.shape[0]), "orient": orient}


def _extract_shape(region: np.ndarray) -> dict:
    mask = (region != BLACK) & (region != GREY)
    comps = _components(mask, connectivity=4)
    assert len(comps) == 1, "each region must contain exactly one shape"
    coords = comps[0]
    colors = {int(region[r, c]) for r, c in coords}
    assert len(colors) == 1, "shape must be monochromatic"
    return {"coords": coords, "color": int(next(iter(colors)))}


def _shift_local(coords: np.ndarray, h: int, w: int, orient: str, step: int, direction_mode: str) -> np.ndarray:
    moved = np.array(coords, dtype=int, copy=True)
    if direction_mode == "perpendicular":
        dr, dc = (1, 0) if orient == "H" else (0, 1)
    elif direction_mode == "parallel":
        dr, dc = (0, 1) if orient == "H" else (1, 0)
    elif direction_mode == "reverse":
        dr, dc = (-1, 0) if orient == "H" else (0, -1)
    else:
        raise ValueError(direction_mode)
    moved[:, 0] = (moved[:, 0] + dr * int(step)) % int(h)
    moved[:, 1] = (moved[:, 1] + dc * int(step)) % int(w)
    return moved


def _analyze_regions(grid: np.ndarray) -> tuple[list[dict], dict[int, int]]:
    regions = _region_infos_from_grid(grid)
    assert 4 <= len(regions) <= 6, "expected 4 to 6 regions"
    lengths = []
    for info in regions:
        region = grid[info["r0"] : info["r1"] + 1, info["c0"] : info["c1"] + 1]
        bar = _extract_bar(region)
        shape = _extract_shape(region)
        info["bar"] = bar
        info["shape"] = shape
        lengths.append(int(bar["length"]))
    assert len(set(lengths)) == len(lengths), "gray bar lengths must be globally distinct"
    ranked = sorted(regions, key=lambda item: int(item["bar"]["length"]), reverse=True)
    rank_by_region = {int(info["index"]): rank for rank, info in enumerate(ranked, start=1)}
    return regions, rank_by_region


def _solve_variant(grid: np.ndarray, *, distance_mode: str, direction_mode: str) -> np.ndarray:
    g = np.asarray(grid, dtype=int)
    regions, rank_by_region = _analyze_regions(g)
    out = g.copy()
    for info in regions:
        r0, r1, c0, c1 = info["r0"], info["r1"], info["c0"], info["c1"]
        out[r0 : r1 + 1, c0 : c1 + 1] = BLACK
        rank = int(rank_by_region[int(info["index"])])
        length = int(info["bar"]["length"])
        if distance_mode == "rank":
            step = rank
        elif distance_mode == "length":
            step = length
        elif distance_mode == "reverse_rank":
            step = len(regions) + 1 - rank
        elif distance_mode == "one":
            step = 1
        else:
            raise ValueError(distance_mode)
        moved = _shift_local(info["shape"]["coords"], info["h"], info["w"], info["bar"]["orient"], step, direction_mode)
        out[r0 + moved[:, 0], c0 + moved[:, 1]] = int(info["shape"]["color"])
    return out


def main(grid: np.ndarray) -> np.ndarray:
    """
    Slide each region's colored shape by a distance encoded by the global gray-bar length ranking.

    Input structure:
    - Black background with full gray separator rows and columns splitting the grid into 4 to 6 rectangular regions.
    - Each region contains one small monochromatic 4-connected shape and one gray 1-cell-thick straight bar.
    - Gray bar lengths are globally distinct, while each bar's orientation is horizontal or vertical.

    Rule:
    - Rank all regional gray bars by length from longest to shortest; the longest gets rank 1, the next rank 2, and so on.
    - In each region, move the colored shape cyclically inside that region by its bar's rank: horizontal bar means slide downward, vertical bar means slide rightward.
    - Remove every non-separator gray bar pixel in the output, but keep the full gray separator lines unchanged.

    Key insight:
    - The bar length does not directly equal the movement; only its global rank matters, and the bar orientation encodes the orthogonal slide direction.
    """
    return _solve_variant(grid, distance_mode="rank", direction_mode="perpendicular")


def _bar_capacity(region: dict, orient: str) -> int:
    if orient == "H":
        return int(region["w"] - int(region["left_sep"]) - int(region["right_sep"]))
    return int(region["h"] - int(region["top_sep"]) - int(region["bottom_sep"]))


def _slide_axis_len(region: dict, orient: str) -> int:
    return int(region["h"] if orient == "H" else region["w"])


def _build_partition() -> list[dict] | None:
    row_count, col_count = random.choice(((2, 2), (2, 3), (3, 2)))
    row_sizes = [random.randint(7, 9) for _ in range(row_count)]
    col_sizes = [random.randint(7, 9) for _ in range(col_count)]
    h = sum(row_sizes) + row_count - 1
    w = sum(col_sizes) + col_count - 1
    if max(h, w) > 30:
        return None
    row_starts = []
    cur = 0
    for size in row_sizes:
        row_starts.append(cur)
        cur += size + 1
    col_starts = []
    cur = 0
    for size in col_sizes:
        col_starts.append(cur)
        cur += size + 1
    regions = []
    for i, (r0, rh) in enumerate(zip(row_starts, row_sizes)):
        for j, (c0, cw) in enumerate(zip(col_starts, col_sizes)):
            regions.append(
                {
                    "index": len(regions),
                    "r0": int(r0),
                    "r1": int(r0 + rh - 1),
                    "c0": int(c0),
                    "c1": int(c0 + cw - 1),
                    "h": int(rh),
                    "w": int(cw),
                    "top_sep": i > 0,
                    "bottom_sep": i < row_count - 1,
                    "left_sep": j > 0,
                    "right_sep": j < col_count - 1,
                    "row_count": row_count,
                    "col_count": col_count,
                    "grid_h": h,
                    "grid_w": w,
                }
            )
    return regions


def _base_grid(regions: list[dict]) -> np.ndarray:
    h = int(regions[0]["grid_h"])
    w = int(regions[0]["grid_w"])
    grid = np.zeros((h, w), dtype=int)
    row_breaks = sorted({int(info["r1"]) + 1 for info in regions if info["bottom_sep"]})
    col_breaks = sorted({int(info["c1"]) + 1 for info in regions if info["right_sep"]})
    for r in row_breaks:
        grid[r, :] = GREY
    for c in col_breaks:
        grid[:, c] = GREY
    return grid


def _choose_bar_specs(regions: list[dict]) -> list[dict] | None:
    n = len(regions)
    global_max = max(max(_bar_capacity(region, "H"), _bar_capacity(region, "V")) for region in regions)
    pool = list(range(2, min(8, global_max) + 1))
    if len(pool) < n:
        return None
    for _ in range(200):
        lengths_by_rank = sorted(random.sample(pool, n), reverse=True)
        ranks = list(range(1, n + 1))
        random.shuffle(ranks)
        orientations = [random.choice(("H", "V")) for _ in regions]
        if len(set(orientations)) == 1:
            orientations[random.randrange(n)] = "V" if orientations[0] == "H" else "H"
        specs = []
        ok = True
        for region, orient, rank in zip(regions, orientations, ranks):
            length = int(lengths_by_rank[rank - 1])
            if _bar_capacity(region, orient) < length:
                ok = False
                break
            if _slide_axis_len(region, orient) <= rank:
                ok = False
                break
            specs.append({"orient": orient, "rank": rank, "length": length})
        if ok:
            return specs
    return None


def _shape_dims(cells: tuple[tuple[int, int], ...]) -> tuple[int, int]:
    return max(r for r, _ in cells) + 1, max(c for _, c in cells) + 1


def _shape_candidates(region: dict, orient: str, rank: int, want_wrap: bool | None) -> list[dict]:
    candidates = []
    for base in _BASE_SHAPES:
        transformed = _transform_shape(base)
        sh, sw = _shape_dims(transformed)
        if sh > region["h"] or sw > region["w"]:
            continue
        if orient == "H":
            nonwrap_max = region["h"] - rank - sh
            wrap_min = region["h"] - rank - sh + 1
            top_positions = []
            if want_wrap is not True:
                top_positions.extend(range(0, max(-1, nonwrap_max) + 1))
            if want_wrap is not False:
                top_positions.extend(range(max(0, wrap_min), region["h"] - sh + 1))
            top_positions = sorted(set(int(x) for x in top_positions if 0 <= x <= region["h"] - sh))
            for top in top_positions:
                wraps = bool(top >= wrap_min)
                if want_wrap is not None and wraps != want_wrap:
                    continue
                for left in range(0, region["w"] - sw + 1):
                    candidates.append({"cells": transformed, "top": int(top), "left": int(left), "wraps": wraps})
        else:
            nonwrap_max = region["w"] - rank - sw
            wrap_min = region["w"] - rank - sw + 1
            left_positions = []
            if want_wrap is not True:
                left_positions.extend(range(0, max(-1, nonwrap_max) + 1))
            if want_wrap is not False:
                left_positions.extend(range(max(0, wrap_min), region["w"] - sw + 1))
            left_positions = sorted(set(int(x) for x in left_positions if 0 <= x <= region["w"] - sw))
            for left in left_positions:
                wraps = bool(left >= wrap_min)
                if want_wrap is not None and wraps != want_wrap:
                    continue
                for top in range(0, region["h"] - sh + 1):
                    candidates.append({"cells": transformed, "top": int(top), "left": int(left), "wraps": wraps})
    random.shuffle(candidates)
    return candidates


def _forbidden_from_shape(shape_cells: set[tuple[int, int]], h: int, w: int) -> set[tuple[int, int]]:
    forbidden = set(shape_cells)
    for r, c in list(shape_cells):
        for nr, nc in _neighbors4(r, c):
            if 0 <= nr < h and 0 <= nc < w:
                forbidden.add((nr, nc))
    return forbidden


def _bar_positions(region: dict, orient: str, length: int, forbidden: set[tuple[int, int]]) -> list[list[tuple[int, int]]]:
    top_margin = 1 if region["top_sep"] else 0
    bottom_margin = 1 if region["bottom_sep"] else 0
    left_margin = 1 if region["left_sep"] else 0
    right_margin = 1 if region["right_sep"] else 0
    positions: list[list[tuple[int, int]]] = []
    if orient == "H":
        for row in range(top_margin, region["h"] - bottom_margin):
            for start in range(left_margin, region["w"] - right_margin - length + 1):
                cells = [(row, start + dc) for dc in range(length)]
                if not any(cell in forbidden for cell in cells):
                    positions.append(cells)
    else:
        for col in range(left_margin, region["w"] - right_margin):
            for start in range(top_margin, region["h"] - bottom_margin - length + 1):
                cells = [(start + dr, col) for dr in range(length)]
                if not any(cell in forbidden for cell in cells):
                    positions.append(cells)
    random.shuffle(positions)
    return positions


def _place_region_content(region: dict, spec: dict, color: int, want_wrap: bool | None) -> dict | None:
    candidates = _shape_candidates(region, spec["orient"], spec["rank"], want_wrap)
    for cand in candidates:
        shape_cells = {(cand["top"] + r, cand["left"] + c) for r, c in cand["cells"]}
        forbidden = _forbidden_from_shape(shape_cells, region["h"], region["w"])
        bar_options = _bar_positions(region, spec["orient"], spec["length"], forbidden)
        if not bar_options:
            continue
        return {
            "shape_cells": sorted(shape_cells),
            "bar_cells": bar_options[0],
            "color": int(color),
            "rank": int(spec["rank"]),
            "length": int(spec["length"]),
            "orient": spec["orient"],
            "wraps": bool(cand["wraps"]),
        }
    return None


def _render_example(regions: list[dict], placements: list[dict]) -> tuple[np.ndarray, np.ndarray]:
    inp = _base_grid(regions)
    out = _base_grid(regions)
    for region, placed in zip(regions, placements):
        r0, c0 = region["r0"], region["c0"]
        for rr, cc in placed["bar_cells"]:
            inp[r0 + rr, c0 + cc] = GREY
        for rr, cc in placed["shape_cells"]:
            inp[r0 + rr, c0 + cc] = int(placed["color"])
        local_shape = np.array(placed["shape_cells"], dtype=int)
        moved = _shift_local(local_shape, region["h"], region["w"], placed["orient"], placed["rank"], "perpendicular")
        for rr, cc in moved:
            out[r0 + rr, c0 + cc] = int(placed["color"])
    return inp, out


def _signature(grid: np.ndarray) -> str:
    return hashlib.md5(grid.tobytes() + str(grid.shape).encode()).hexdigest()


def _semantic_checks(inp: np.ndarray, expected: np.ndarray, placements: list[dict]) -> bool:
    if not np.array_equal(main(inp), expected):
        return False
    if len({placement["color"] for placement in placements}) != len(placements):
        return False
    if len({placement["length"] for placement in placements}) != len(placements):
        return False
    if len({placement["rank"] for placement in placements}) != len(placements):
        return False
    if len({placement["orient"] for placement in placements}) < 2:
        return False
    wrap_count = sum(int(placement["wraps"]) for placement in placements)
    if wrap_count == 0 or wrap_count == len(placements):
        return False
    wrong_outputs = (
        _solve_variant(inp, distance_mode="length", direction_mode="perpendicular"),
        _solve_variant(inp, distance_mode="reverse_rank", direction_mode="perpendicular"),
        _solve_variant(inp, distance_mode="one", direction_mode="perpendicular"),
        _solve_variant(inp, distance_mode="rank", direction_mode="parallel"),
        _solve_variant(inp, distance_mode="rank", direction_mode="reverse"),
    )
    if any(np.array_equal(expected, wrong) for wrong in wrong_outputs):
        return False
    return True


def _build_example() -> tuple[np.ndarray, np.ndarray, list[dict]] | None:
    regions = _build_partition()
    if regions is None:
        return None
    specs = _choose_bar_specs(regions)
    if specs is None:
        return None
    colors = random.sample(SHAPE_PALETTE, len(regions))
    placements: list[dict] = []
    need_wrap = True
    need_nonwrap = True
    order = list(range(len(regions)))
    random.shuffle(order)
    by_index = [None] * len(regions)
    for pos, idx in enumerate(order):
        region = regions[idx]
        spec = specs[idx]
        remaining = len(order) - pos - 1
        wrap_choice: bool | None = random.choice((True, False, None))
        if need_wrap and remaining == 0:
            wrap_choice = True
        elif need_nonwrap and remaining == 0:
            wrap_choice = False
        elif need_wrap and not need_nonwrap:
            wrap_choice = True if random.random() < 0.7 else None
        elif need_nonwrap and not need_wrap:
            wrap_choice = False if random.random() < 0.7 else None
        placed = _place_region_content(region, spec, colors[idx], wrap_choice)
        if placed is None:
            return None
        by_index[idx] = placed
        need_wrap = need_wrap and not placed["wraps"]
        need_nonwrap = need_nonwrap and placed["wraps"]
    placements = [item for item in by_index if item is not None]
    inp, expected = _render_example(regions, placements)
    if not _semantic_checks(inp, expected, placements):
        return None
    if _signature(expected) in _SEEN_OUTPUT_SIGNATURES:
        return None
    return inp, expected, placements


def generate_input() -> np.ndarray:
    for _ in range(MAX_GENERATION_ATTEMPTS):
        built = _build_example()
        if built is None:
            continue
        inp, expected, _placements = built
        _SEEN_OUTPUT_SIGNATURES.add(_signature(expected))
        return inp
    raise RuntimeError("failed to generate a valid puzzle instance")
