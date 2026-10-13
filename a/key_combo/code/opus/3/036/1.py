from __future__ import annotations

import importlib.util
import random
from collections import Counter
from pathlib import Path

import numpy as np


_ARC_UTILS_PATH = Path("/home/lixinyu/snap/new/solvers/arc_utils.py")


def _load_arc_utils():
    if not _ARC_UTILS_PATH.exists():
        return None
    spec = importlib.util.spec_from_file_location("_shared_arc_utils", _ARC_UTILS_PATH)
    if spec is None or spec.loader is None:
        return None
    module = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(module)
    return module


_ARC_UTILS = _load_arc_utils()

if _ARC_UTILS is not None:
    Color = _ARC_UTILS.Color
    bbox_coords = _ARC_UTILS.bbox_coords
    components_mask = _ARC_UTILS.components_mask
    in_bounds = _ARC_UTILS.in_bounds
else:
    class Color:
        BLACK = 0
        RED = 2

    def bbox_coords(coords: np.ndarray) -> tuple[int, int, int, int]:
        r0, c0 = coords.min(axis=0)
        r1, c1 = coords.max(axis=0)
        return int(r0), int(c0), int(r1), int(c1)

    def in_bounds(shape: tuple[int, int], r: int, c: int) -> bool:
        h, w = map(int, shape)
        return 0 <= int(r) < h and 0 <= int(c) < w

    def components_mask(mask: np.ndarray, connectivity: int = 4) -> list[np.ndarray]:
        mask = np.asarray(mask, dtype=bool)
        h, w = mask.shape
        visited = np.zeros_like(mask, dtype=bool)
        if int(connectivity) == 4:
            steps = [(-1, 0), (1, 0), (0, -1), (0, 1)]
        else:
            steps = [(dr, dc) for dr in (-1, 0, 1) for dc in (-1, 0, 1) if dr or dc]
        out: list[np.ndarray] = []
        for r in range(h):
            for c in range(w):
                if not mask[r, c] or visited[r, c]:
                    continue
                stack = [(r, c)]
                visited[r, c] = True
                comp: list[tuple[int, int]] = []
                while stack:
                    rr, cc = stack.pop()
                    comp.append((rr, cc))
                    for dr, dc in steps:
                        nr, nc = rr + dr, cc + dc
                        if in_bounds(mask.shape, nr, nc) and mask[nr, nc] and not visited[nr, nc]:
                            visited[nr, nc] = True
                            stack.append((nr, nc))
                out.append(np.array(comp, dtype=int))
        return out


DIAGONALS = [(-1, -1), (-1, 1), (1, -1), (1, 1)]
GEN_RETRY_LIMIT = 300
GRID_MIN = 18
GRID_MAX = 24
OBJECT_COUNT_RANGE = (3, 5)
OUTLINE_COLORS = [1, 3, 4, 5, 6, 7, 8, 9]
OUTLINE_GAP = 2
PLACEMENT_RETRY_LIMIT = 200

INTERIOR_OPTIONS_BY_AREA = {
    3: [(1, 3), (3, 1)],
    4: [(1, 4), (2, 2), (4, 1)],
    5: [(1, 5), (5, 1)],
    6: [(2, 3), (3, 2)],
    8: [(2, 4), (4, 2)],
    9: [(3, 3)],
    10: [(2, 5), (5, 2)],
    12: [(3, 4), (4, 3)],
}


def _row_span_interior(grid: np.ndarray, coords: np.ndarray, bg: int) -> tuple[tuple[int, int], ...]:
    occupied = {(int(r), int(c)) for r, c in coords.tolist()}
    rows: dict[int, list[int]] = {}
    for r, c in occupied:
        rows.setdefault(r, []).append(c)

    interior: list[tuple[int, int]] = []
    for r, cols in rows.items():
        left = min(cols)
        right = max(cols)
        if right - left <= 1:
            continue
        for c in range(left + 1, right):
            if (r, c) in occupied:
                continue
            assert int(grid[r, c]) == int(bg), "outline interiors must remain background"
            interior.append((r, c))
    return tuple(sorted(interior))


def _extract_outlines(grid: np.ndarray) -> list[dict]:
    g = np.asarray(grid, dtype=int)
    colors = sorted(set(map(int, g.ravel())) - {Color.BLACK, Color.RED})
    outlines: list[dict] = []
    for color in colors:
        for coords in components_mask(g == color, connectivity=8):
            interior = _row_span_interior(g, coords, Color.BLACK)
            if not interior:
                continue
            outlines.append(
                {
                    "color": int(color),
                    "coords": tuple(map(tuple, coords.tolist())),
                    "bbox": bbox_coords(coords),
                    "interior": interior,
                    "area": len(interior),
                }
            )
    assert outlines, "expected at least one outline object"
    return outlines


def _hit_counts(grid: np.ndarray, source_cells: list[tuple[int, int]], outlines: list[dict]) -> np.ndarray:
    cell_to_outline: dict[tuple[int, int], int] = {}
    for idx, outline in enumerate(outlines):
        for cell in outline["interior"]:
            assert cell not in cell_to_outline, "outline interiors must be disjoint"
            cell_to_outline[cell] = idx

    counts = np.zeros(len(outlines), dtype=int)
    for r, c in source_cells:
        for dr, dc in DIAGONALS:
            rr, cc = r + dr, c + dc
            while in_bounds(grid.shape, rr, cc):
                idx = cell_to_outline.get((rr, cc))
                if idx is not None:
                    counts[idx] += 1
                rr += dr
                cc += dc
    return counts


def main(grid: np.ndarray) -> np.ndarray:
    """
    Erase the red source cluster and fill the unique outline cavity whose diagonal-ray hit count matches the source size.

    Input structure:
    - Black (0) background contains several non-red monochrome 8-connected outline objects.
    - Each outline encloses a row-span cavity: background cells strictly between the leftmost and rightmost outline pixels on occupied rows.
    - A small red (2) 4-connected source cluster sits outside every outline.

    Rule:
    - Extract every outline object's row-span cavity and count the total red source pixels N.
    - From each red pixel, trace four diagonal rays (NE, NW, SE, SW) to the grid boundary without painting the grid.
    - For each outline, accumulate how many cavity cells are traversed by those rays, counting repeated traversals separately.
    - Select the unique outline whose cavity hit total equals N, erase the source pixels to black, and fill that outline's cavity red.

    Key insight:
    - The chosen cavity is encoded indirectly by spatial diagonal alignments from the red source cluster, so hit count replaces direct area matching.
    """
    g = np.asarray(grid, dtype=int)
    assert g.ndim == 2

    source = [tuple(map(int, cell)) for cell in np.argwhere(g == Color.RED)]
    assert source, "expected red source pixels"

    outlines = _extract_outlines(g)
    counts = _hit_counts(g, source, outlines)
    matches = np.flatnonzero(counts == len(source))
    assert len(matches) == 1, "expected exactly one outline with hit count equal to source size"
    target = outlines[int(matches[0])]

    out = g.copy()
    out[g == Color.RED] = Color.BLACK
    for r, c in target["interior"]:
        out[r, c] = Color.RED
    return out


def _normalize_cells(cells: set[tuple[int, int]]) -> tuple[tuple[int, int], ...]:
    min_r = min(r for r, _ in cells)
    min_c = min(c for _, c in cells)
    return tuple(sorted((r - min_r, c - min_c) for r, c in cells))


def _cells_to_mask(cells: tuple[tuple[int, int], ...]) -> np.ndarray:
    h = max(r for r, _ in cells) + 1
    w = max(c for _, c in cells) + 1
    mask = np.zeros((h, w), dtype=bool)
    for r, c in cells:
        mask[r, c] = True
    return mask


def _mask_key(mask: np.ndarray) -> tuple[tuple[int, ...], ...]:
    return tuple(tuple(int(v) for v in row) for row in mask.astype(int))


def _all_mask_variants(base_cells: tuple[tuple[int, int], ...]) -> list[np.ndarray]:
    base = _cells_to_mask(base_cells)
    variants: list[np.ndarray] = []
    seen: set[tuple[tuple[int, ...], ...]] = set()
    for k in range(4):
        rotated = np.rot90(base, k=k)
        for mask in (rotated, np.fliplr(rotated)):
            coords = np.argwhere(mask)
            key = _normalize_cells({(int(r), int(c)) for r, c in coords.tolist()})
            cropped = _cells_to_mask(key)
            sig = _mask_key(cropped)
            if sig not in seen:
                seen.add(sig)
                variants.append(cropped)
    return variants


BASE_SOURCE_SHAPES = [
    ((0, 0), (0, 1), (0, 2)),
    ((0, 0), (1, 0), (1, 1)),
    ((0, 0), (0, 1), (0, 2), (0, 3)),
    ((0, 0), (0, 1), (1, 0), (1, 1)),
    ((0, 0), (1, 0), (2, 0), (2, 1)),
    ((0, 0), (0, 1), (0, 2), (1, 1)),
    ((0, 0), (0, 1), (1, 1), (1, 2)),
    ((0, 0), (0, 1), (0, 2), (0, 3), (0, 4)),
    ((0, 0), (1, 0), (2, 0), (3, 0), (3, 1)),
    ((0, 0), (0, 1), (0, 2), (1, 1), (2, 1)),
    ((0, 1), (1, 0), (1, 1), (1, 2), (2, 1)),
    ((0, 0), (0, 1), (1, 0), (1, 1), (2, 0)),
    ((0, 0), (1, 0), (1, 1), (2, 1), (2, 2)),
    ((0, 0), (0, 1), (0, 2), (0, 3), (0, 4), (0, 5)),
    ((0, 0), (0, 1), (0, 2), (1, 0), (1, 1), (1, 2)),
    ((0, 0), (1, 0), (2, 0), (3, 0), (3, 1), (3, 2)),
    ((0, 0), (0, 1), (0, 2), (1, 1), (2, 1), (2, 2)),
    ((0, 0), (0, 1), (1, 1), (1, 2), (2, 2), (2, 3)),
    ((0, 0), (0, 1), (0, 2), (0, 3), (0, 4), (0, 5), (0, 6)),
    ((0, 0), (0, 1), (0, 2), (1, 1), (2, 1), (3, 1), (3, 2)),
    ((0, 0), (0, 1), (1, 1), (2, 1), (2, 2), (3, 2), (3, 3)),
    ((0, 0), (0, 1), (0, 2), (0, 3), (1, 0), (1, 1), (1, 2)),
    ((0, 0), (0, 1), (0, 2), (0, 3), (0, 4), (0, 5), (0, 6), (0, 7)),
    ((0, 0), (0, 1), (0, 2), (0, 3), (1, 0), (1, 1), (1, 2), (1, 3)),
    ((0, 0), (1, 0), (2, 0), (3, 0), (4, 0), (4, 1), (4, 2), (4, 3)),
    ((0, 0), (0, 1), (0, 2), (0, 3), (1, 1), (2, 1), (3, 1), (3, 2)),
]


def _build_source_masks() -> list[np.ndarray]:
    masks: list[np.ndarray] = []
    seen: set[tuple[tuple[int, ...], ...]] = set()
    for cells in BASE_SOURCE_SHAPES:
        for mask in _all_mask_variants(tuple(cells)):
            n = int(mask.sum())
            if 3 <= n <= 8:
                key = _mask_key(mask)
                if key not in seen:
                    seen.add(key)
                    masks.append(mask)
    return masks


SOURCE_MASKS = _build_source_masks()


def _sample_outline_specs() -> list[dict]:
    areas = random.sample(sorted(INTERIOR_OPTIONS_BY_AREA), random.randint(*OBJECT_COUNT_RANGE))
    colors = random.sample(OUTLINE_COLORS, len(areas))
    specs = []
    for area, color in zip(areas, colors, strict=False):
        inner_h, inner_w = random.choice(INTERIOR_OPTIONS_BY_AREA[area])
        specs.append(
            {
                "color": color,
                "inner_h": inner_h,
                "inner_w": inner_w,
                "outer_h": inner_h + 2,
                "outer_w": inner_w + 2,
                "area": area,
            }
        )
    specs.sort(key=lambda spec: spec["outer_h"] * spec["outer_w"], reverse=True)
    return specs


def _boxes_too_close(a: tuple[int, int, int, int], b: tuple[int, int, int, int], gap: int) -> bool:
    return not (
        a[2] + gap < b[0]
        or b[2] + gap < a[0]
        or a[3] + gap < b[1]
        or b[3] + gap < a[1]
    )


def _place_outlines(height: int, width: int, specs: list[dict]) -> list[dict] | None:
    placed: list[dict] = []
    for spec in specs:
        outer_h = int(spec["outer_h"])
        outer_w = int(spec["outer_w"])
        choices = [
            (r0, c0)
            for r0 in range(1, height - outer_h)
            for c0 in range(1, width - outer_w)
        ]
        random.shuffle(choices)
        for r0, c0 in choices[:PLACEMENT_RETRY_LIMIT]:
            box = (r0, c0, r0 + outer_h - 1, c0 + outer_w - 1)
            if any(_boxes_too_close(box, other["bbox"], OUTLINE_GAP) for other in placed):
                continue
            placed.append({**spec, "bbox": box})
            break
        else:
            return None
    return placed


def _draw_frame(grid: np.ndarray, box: tuple[int, int, int, int], color: int) -> tuple[tuple[int, int], ...]:
    r0, c0, r1, c1 = box
    grid[r0, c0 : c1 + 1] = color
    grid[r1, c0 : c1 + 1] = color
    grid[r0 : r1 + 1, c0] = color
    grid[r0 : r1 + 1, c1] = color
    return tuple((r, c) for r in range(r0 + 1, r1) for c in range(c0 + 1, c1))


def _outline_clearance_mask(grid: np.ndarray, outlines: list[dict]) -> np.ndarray:
    allowed = grid == Color.BLACK
    h, w = grid.shape
    for outline in outlines:
        for r, c in outline["interior"]:
            allowed[r, c] = False
        for r, c in outline["coords"]:
            for dr in (-1, 0, 1):
                for dc in (-1, 0, 1):
                    rr, cc = r + dr, c + dc
                    if 0 <= rr < h and 0 <= cc < w:
                        allowed[rr, cc] = False
    return allowed


def _outline_diagonal_tables(outlines: list[dict]) -> list[tuple[Counter, Counter]]:
    tables = []
    for outline in outlines:
        diff_counts: Counter = Counter()
        sum_counts: Counter = Counter()
        for r, c in outline["interior"]:
            diff_counts[r - c] += 1
            sum_counts[r + c] += 1
        tables.append((diff_counts, sum_counts))
    return tables


def _candidate_sources(grid: np.ndarray, outlines: list[dict]) -> list[dict]:
    h, w = grid.shape
    allowed = _outline_clearance_mask(grid, outlines)
    tables = _outline_diagonal_tables(outlines)
    contrib = np.zeros((h, w, len(outlines)), dtype=int)

    for r in range(h):
        for c in range(w):
            if not allowed[r, c]:
                continue
            for idx, (diff_counts, sum_counts) in enumerate(tables):
                contrib[r, c, idx] = diff_counts.get(r - c, 0) + sum_counts.get(r + c, 0)

    masks = SOURCE_MASKS.copy()
    random.shuffle(masks)
    candidates: list[dict] = []
    for mask in masks:
        n = int(mask.sum())
        rr, cc = np.where(mask)
        for r0 in range(h - mask.shape[0] + 1):
            for c0 in range(w - mask.shape[1] + 1):
                rows = r0 + rr
                cols = c0 + cc
                if not np.all(allowed[rows, cols]):
                    continue
                counts = contrib[rows, cols].sum(axis=0)
                matches = np.flatnonzero(counts == n)
                if len(matches) != 1:
                    continue
                target_idx = int(matches[0])
                non_target_area_match = any(
                    outline["area"] == n
                    for idx, outline in enumerate(outlines)
                    if idx != target_idx
                )
                score = 0
                if outlines[target_idx]["area"] != n:
                    score += 2
                if non_target_area_match:
                    score += 1
                candidates.append(
                    {
                        "cells": tuple(sorted(zip(rows.tolist(), cols.tolist(), strict=False))),
                        "target_idx": target_idx,
                        "size": n,
                        "score": score,
                    }
                )
    return candidates


def _construct_grid(height: int, width: int, placed_specs: list[dict]) -> tuple[np.ndarray, list[dict]]:
    grid = np.zeros((height, width), dtype=int)
    outlines = []
    for spec in placed_specs:
        interior = _draw_frame(grid, spec["bbox"], int(spec["color"]))
        r0, c0, r1, c1 = spec["bbox"]
        coords = np.argwhere(grid[r0 : r1 + 1, c0 : c1 + 1] == spec["color"])
        coords[:, 0] += r0
        coords[:, 1] += c0
        outlines.append(
            {
                "color": int(spec["color"]),
                "coords": tuple(map(tuple, coords.tolist())),
                "bbox": spec["bbox"],
                "interior": interior,
                "area": len(interior),
            }
        )
    return grid, outlines


def _gen_attempt() -> np.ndarray | None:
    height = random.randint(GRID_MIN, GRID_MAX)
    width = random.randint(GRID_MIN, GRID_MAX)
    placed_specs = _place_outlines(height, width, _sample_outline_specs())
    if placed_specs is None:
        return None

    grid, outlines = _construct_grid(height, width, placed_specs)
    candidates = _candidate_sources(grid, outlines)
    if not candidates:
        return None

    best_score = max(candidate["score"] for candidate in candidates)
    pool = [candidate for candidate in candidates if candidate["score"] == best_score]
    choice = random.choice(pool)

    inp = grid.copy()
    for r, c in choice["cells"]:
        inp[r, c] = Color.RED

    try:
        out = main(inp)
    except Exception:
        return None

    expected = grid.copy()
    for r, c in choice["cells"]:
        expected[r, c] = Color.BLACK
    for r, c in outlines[choice["target_idx"]]["interior"]:
        expected[r, c] = Color.RED

    if not np.array_equal(out, expected):
        return None
    return inp


def generate_input() -> np.ndarray:
    for _ in range(GEN_RETRY_LIMIT):
        grid = _gen_attempt()
        if grid is not None:
            return grid
    raise RuntimeError("failed to generate a valid diagonal-ray cavity input")
