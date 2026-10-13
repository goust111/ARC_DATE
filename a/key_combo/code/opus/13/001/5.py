from __future__ import annotations

import random
from collections import deque

import numpy as np


BG = 0
ORANGE = 7

SOURCE_TO_STAMP = {
    2: 1,
    3: 9,
    4: 5,
    6: 8,
}

SOURCE_COLORS = tuple(SOURCE_TO_STAMP.keys())

PATTERN_CHOICES = (
    np.array(
        [
            [0, 1, 0],
            [1, 1, 1],
            [0, 1, 0],
        ],
        dtype=bool,
    ),
    np.array(
        [
            [1, 1, 0],
            [0, 1, 0],
            [0, 1, 1],
        ],
        dtype=bool,
    ),
    np.array(
        [
            [1, 1, 1],
            [0, 1, 0],
            [0, 1, 0],
        ],
        dtype=bool,
    ),
)

L_CORNERS = {(0, 0), (0, 1), (1, 0), (1, 1)}
MAX_GEN_ATTEMPTS = 400


def _in_bounds(shape: tuple[int, int], r: int, c: int) -> bool:
    h, w = shape
    return 0 <= r < h and 0 <= c < w


def _neighbors4(r: int, c: int):
    yield r - 1, c
    yield r + 1, c
    yield r, c - 1
    yield r, c + 1


def _component_coords(mask: np.ndarray) -> list[np.ndarray]:
    h, w = mask.shape
    seen = np.zeros_like(mask, dtype=bool)
    comps: list[np.ndarray] = []
    for r in range(h):
        for c in range(w):
            if not mask[r, c] or seen[r, c]:
                continue
            q = deque([(r, c)])
            seen[r, c] = True
            pts: list[tuple[int, int]] = []
            while q:
                cr, cc = q.popleft()
                pts.append((cr, cc))
                for nr, nc in _neighbors4(cr, cc):
                    if _in_bounds(mask.shape, nr, nc) and mask[nr, nc] and not seen[nr, nc]:
                        seen[nr, nc] = True
                        q.append((nr, nc))
            comps.append(np.array(pts, dtype=int))
    return comps


def _bbox(coords: np.ndarray) -> tuple[int, int, int, int]:
    r0, c0 = coords.min(axis=0)
    r1, c1 = coords.max(axis=0)
    return int(r0), int(c0), int(r1), int(c1)


def _missing_corner_center_and_anchor(
    coords: np.ndarray,
) -> tuple[tuple[int, int], tuple[int, int], tuple[int, int]]:
    r0, c0, r1, c1 = _bbox(coords)
    assert r1 - r0 == 1 and c1 - c0 == 1
    rel = {(int(r - r0), int(c - c0)) for r, c in coords}
    missing = L_CORNERS - rel
    assert len(missing) == 1
    mr, mc = next(iter(missing))
    center = (r0 + (1 - mr), c0 + (1 - mc))
    diag = (-1 if mr == 0 else 1, -1 if mc == 0 else 1)
    anchor = (center[0] + 2 * diag[0], center[1] + 2 * diag[1])
    return (mr, mc), center, anchor


def _pattern_cells(pattern: np.ndarray, anchor: tuple[int, int]) -> list[tuple[int, int]]:
    pts: list[tuple[int, int]] = []
    for pr in range(3):
        for pc in range(3):
            if pattern[pr, pc]:
                pts.append((anchor[0] + pr - 1, anchor[1] + pc - 1))
    return pts


def _visible_pattern_cells(
    shape: tuple[int, int], pattern: np.ndarray, anchor: tuple[int, int]
) -> list[tuple[int, int]]:
    return [(r, c) for r, c in _pattern_cells(pattern, anchor) if _in_bounds(shape, r, c)]


def _find_reference_pattern(grid: np.ndarray) -> np.ndarray:
    orange_comps = _component_coords(grid == ORANGE)
    assert len(orange_comps) == 1, "expected one orange reference component"
    coords = orange_comps[0]
    r0, c0, r1, c1 = _bbox(coords)
    assert r1 - r0 == 2 and c1 - c0 == 2, "reference pattern must occupy a 3x3 bbox"
    pattern = np.zeros((3, 3), dtype=bool)
    for r, c in coords:
        pattern[int(r - r0), int(c - c0)] = True
    assert pattern[1, 1], "reference pattern must reveal its center"
    return pattern


def _triomino_records(grid: np.ndarray) -> list[dict[str, object]]:
    records: list[dict[str, object]] = []
    mask = (grid != BG) & (grid != ORANGE)
    for coords in _component_coords(mask):
        color = int(grid[tuple(coords[0])])
        assert color in SOURCE_TO_STAMP
        assert all(int(grid[r, c]) == color for r, c in coords)
        assert coords.shape[0] == 3
        missing, center, anchor = _missing_corner_center_and_anchor(coords)
        records.append(
            {
                "color": color,
                "coords": coords,
                "missing": missing,
                "center": center,
                "anchor": anchor,
            }
        )
    assert 3 <= len(records) <= 5
    return records


def _distance_key(center: tuple[int, int], shape: tuple[int, int]) -> tuple[float, int, int]:
    grid_center = ((shape[0] - 1) / 2.0, (shape[1] - 1) / 2.0)
    dr = center[0] - grid_center[0]
    dc = center[1] - grid_center[1]
    return (dr * dr + dc * dc, center[0], center[1])


def _render_output(
    base: np.ndarray, pattern: np.ndarray, records: list[dict[str, object]]
) -> np.ndarray:
    out = np.asarray(base, dtype=int).copy()
    immutable = base != BG
    ordered = sorted(
        records,
        key=lambda rec: _distance_key(rec["center"], base.shape),
        reverse=True,
    )
    for rec in ordered:
        stamp_color = SOURCE_TO_STAMP[int(rec["color"])]
        for r, c in _pattern_cells(pattern, rec["anchor"]):
            if _in_bounds(base.shape, r, c) and not immutable[r, c]:
                out[r, c] = stamp_color
    return out


def main(grid: np.ndarray) -> np.ndarray:
    """
    Stamp a recolored copy of the orange reference pattern at anchors inferred from missing-corner L triominoes.

    Input structure:
    - The grid is mostly black background with one orange 3x3 reference pattern in a corner.
    - Three to five non-orange foreground objects are monochrome L triominoes, each filling three cells of a 2x2 box.
    - Every triomino color belongs to a fixed source palette whose colors encode the stamp recoloring.

    Rule:
    - Find the orange 3x3 reference pattern and record its occupied cells relative to the pattern center.
    - For each L triomino, identify the missing 2x2 corner; the anchor is two diagonal steps from the triomino's opposite corner cell toward that missing corner.
    - Recolor the orange reference pattern using the triomino's source color mapping, then paste the recolored pattern centered at the anchor with boundary clipping; original triominoes and the reference pattern stay unchanged.
    - If multiple pasted patterns target the same background cell, draw the one whose triomino center is closer to the grid center later so it overwrites earlier ones.

    Key insight:
    - Each triomino contributes two coupled signals: its missing corner fixes the stamp location, while its color selects the stamp color.
    """
    g = np.asarray(grid, dtype=int)
    assert g.ndim == 2
    assert max(g.shape) <= 30
    pattern = _find_reference_pattern(g)
    records = _triomino_records(g)
    return _render_output(g, pattern, records)


def _source_cells(r0: int, c0: int, missing: tuple[int, int]) -> tuple[tuple[int, int], ...]:
    return tuple(
        (r0 + rr, c0 + cc)
        for rr in (0, 1)
        for cc in (0, 1)
        if (rr, cc) != missing
    )


def _bbox_gap(a: tuple[int, int, int, int], b: tuple[int, int, int, int]) -> int:
    row_gap = max(0, b[0] - a[2] - 1, a[0] - b[2] - 1)
    col_gap = max(0, b[1] - a[3] - 1, a[1] - b[3] - 1)
    return max(row_gap, col_gap)


def _reference_cells(
    shape: tuple[int, int], pattern: np.ndarray, corner: str
) -> tuple[set[tuple[int, int]], tuple[int, int]]:
    h, w = shape
    if corner == "tl":
        r0, c0 = 0, 0
    elif corner == "tr":
        r0, c0 = 0, w - 3
    elif corner == "bl":
        r0, c0 = h - 3, 0
    else:
        r0, c0 = h - 3, w - 3
    cells = {
        (r0 + pr, c0 + pc)
        for pr in range(3)
        for pc in range(3)
        if pattern[pr, pc]
    }
    return cells, (r0, c0)


def _candidate_records(
    shape: tuple[int, int],
    pattern: np.ndarray,
    missing: tuple[int, int],
    ref_cells: set[tuple[int, int]],
    source_boxes: list[tuple[int, int, int, int]],
    blocked_output: set[tuple[int, int]],
    allow_clip: bool,
) -> list[dict[str, object]]:
    h, w = shape
    candidates: list[dict[str, object]] = []
    for r0 in range(h - 1):
        for c0 in range(w - 1):
            src = _source_cells(r0, c0, missing)
            src_set = set(src)
            if src_set & blocked_output:
                continue
            bbox = (r0, c0, r0 + 1, c0 + 1)
            if any(_bbox_gap(bbox, prev) < 5 for prev in source_boxes):
                continue
            coords = np.array(src, dtype=int)
            _missing, center, anchor = _missing_corner_center_and_anchor(coords)
            assert _missing == missing
            if not _in_bounds(shape, anchor[0], anchor[1]):
                continue
            full_cells = _pattern_cells(pattern, anchor)
            visible_cells = [(r, c) for r, c in full_cells if _in_bounds(shape, r, c)]
            if not visible_cells:
                continue
            clipped = len(visible_cells) != len(full_cells)
            if allow_clip:
                if not clipped or len(visible_cells) < 4:
                    continue
            else:
                if clipped:
                    continue
            visible_set = set(visible_cells)
            if visible_set & blocked_output:
                continue
            if visible_set & ref_cells:
                continue
            if visible_set & src_set:
                continue
            candidates.append(
                {
                    "coords": coords,
                    "center": center,
                    "anchor": anchor,
                    "bbox": bbox,
                    "stamp_cells": visible_set,
                }
            )
    return candidates


def _plan_instance() -> tuple[np.ndarray, np.ndarray] | None:
    h = random.randint(15, 20)
    w = random.randint(15, 20)
    pattern = random.choice(PATTERN_CHOICES)
    ref_cells, _ = _reference_cells((h, w), pattern, random.choice(("tl", "tr", "bl", "br")))

    count = 4 if random.random() < 0.65 else 3
    colors = list(SOURCE_COLORS) if count == 4 else random.sample(SOURCE_COLORS, count)
    random.shuffle(colors)

    orientations = random.sample(list(L_CORNERS), count)
    random.shuffle(orientations)

    clip_index = random.randrange(count) if random.random() < 0.28 else None

    blocked_output = set(ref_cells)
    source_boxes: list[tuple[int, int, int, int]] = []
    records: list[dict[str, object]] = []

    order = list(range(count))
    random.shuffle(order)
    for idx in order:
        candidates = _candidate_records(
            (h, w),
            pattern,
            orientations[idx],
            ref_cells,
            source_boxes,
            blocked_output,
            allow_clip=(clip_index == idx),
        )
        if not candidates and clip_index == idx:
            candidates = _candidate_records(
                (h, w),
                pattern,
                orientations[idx],
                ref_cells,
                source_boxes,
                blocked_output,
                allow_clip=False,
            )
        if not candidates:
            return None
        chosen = random.choice(candidates)
        chosen["color"] = colors[idx]
        records.append(chosen)
        blocked_output.update(set(map(tuple, chosen["coords"])))
        blocked_output.update(chosen["stamp_cells"])
        source_boxes.append(chosen["bbox"])

    inp = np.zeros((h, w), dtype=int)
    for r, c in ref_cells:
        inp[r, c] = ORANGE
    for rec in records:
        for r, c in rec["coords"]:
            inp[r, c] = int(rec["color"])

    out = _render_output(inp, pattern, records)

    if len(np.unique(inp)) < 5:
        return None
    if not np.array_equal(main(inp), out):
        return None
    return inp, out


def generate_input() -> np.ndarray:
    for _ in range(MAX_GEN_ATTEMPTS):
        planned = _plan_instance()
        if planned is not None:
            return planned[0]
    raise RuntimeError("failed to generate a valid puzzle instance")
