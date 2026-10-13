import random

import numpy as np


BG = 0
RECT_COLOR = 1
GREEN = 3
YELLOW = 4
MAGENTA = 6
MARKER_COLOR = 7
TEAL = 8

MISSING_TO_FILL = {
    (0, 0): GREEN,
    (0, 1): TEAL,
    (1, 0): YELLOW,
    (1, 1): MAGENTA,
}
MISSING_CORNERS = tuple(MISSING_TO_FILL)
MAX_GEN_ATTEMPTS = 500


def _get_bg(grid: np.ndarray) -> int:
    values, counts = np.unique(grid, return_counts=True)
    return int(values[np.argmax(counts)])


def _bbox(coords: np.ndarray) -> tuple[int, int, int, int]:
    r0, c0 = coords.min(axis=0)
    r1, c1 = coords.max(axis=0)
    return int(r0), int(c0), int(r1), int(c1)


def _components(mask: np.ndarray) -> list[np.ndarray]:
    h, w = mask.shape
    seen = np.zeros_like(mask, dtype=bool)
    out: list[np.ndarray] = []
    for r in range(h):
        for c in range(w):
            if not mask[r, c] or seen[r, c]:
                continue
            stack = [(r, c)]
            seen[r, c] = True
            coords = []
            while stack:
                rr, cc = stack.pop()
                coords.append((rr, cc))
                for nr, nc in ((rr - 1, cc), (rr + 1, cc), (rr, cc - 1), (rr, cc + 1)):
                    if 0 <= nr < h and 0 <= nc < w and mask[nr, nc] and not seen[nr, nc]:
                        seen[nr, nc] = True
                        stack.append((nr, nc))
            out.append(np.array(coords, dtype=int))
    return out


def _ranges_overlap(a0: int, a1: int, b0: int, b1: int) -> bool:
    return max(a0, b0) <= min(a1, b1)


def _decode_marker(grid: np.ndarray, marker_bbox: tuple[int, int, int, int], bg: int) -> int:
    r0, c0, r1, c1 = marker_bbox
    if (r1 - r0 + 1, c1 - c0 + 1) != (2, 2):
        raise ValueError("marker must occupy a 2x2 box")
    missing = []
    for dr in range(2):
        for dc in range(2):
            if int(grid[r0 + dr, c0 + dc]) == bg:
                missing.append((dr, dc))
    if len(missing) != 1 or missing[0] not in MISSING_TO_FILL:
        raise ValueError("marker must be a 2x2 L triomino")
    return MISSING_TO_FILL[missing[0]]


def _extract_rectangles(grid: np.ndarray, bg: int) -> list[dict]:
    rectangles = []
    for coords in _components(grid == RECT_COLOR):
        box = _bbox(coords)
        r0, c0, r1, c1 = box
        if coords.shape[0] != (r1 - r0 + 1) * (c1 - c0 + 1):
            raise ValueError("blue objects must be solid rectangles")
        rectangles.append({"bbox": box, "coords": coords})
    if len(rectangles) < 3:
        raise ValueError("expected multiple blue rectangles")
    return rectangles


def _extract_markers(grid: np.ndarray, bg: int) -> list[tuple[int, int, int, int]]:
    markers = []
    for coords in _components(grid == MARKER_COLOR):
        box = _bbox(coords)
        if coords.shape[0] != 3:
            raise ValueError("each marker must have area 3")
        r0, c0, r1, c1 = box
        if (r1 - r0 + 1, c1 - c0 + 1) != (2, 2):
            raise ValueError("each marker must be a 2x2 L triomino")
        markers.append(box)
    return markers


def _scene(grid: np.ndarray) -> tuple[int, list[dict]]:
    g = np.asarray(grid, dtype=int)
    if g.ndim != 2:
        raise ValueError("grid must be 2D")
    bg = _get_bg(g)
    rectangles = _extract_rectangles(g, bg)
    markers = _extract_markers(g, bg)
    if len(markers) != len(rectangles):
        raise ValueError("expected one marker per rectangle")

    used_markers: set[tuple[int, int, int, int]] = set()
    enriched = []
    for rect in rectangles:
        r0, c0, _r1, c1 = rect["bbox"]
        candidates = []
        for marker in markers:
            mr0, mc0, mr1, mc1 = marker
            if mr1 + 1 == r0 and max(c0, mc0) <= min(c1, mc1):
                candidates.append(marker)
        if len(candidates) != 1:
            raise ValueError("each rectangle must have a unique directly-above marker")
        marker = candidates[0]
        used_markers.add(marker)
        enriched.append(
            {
                **rect,
                "marker_bbox": marker,
                "fill_color": _decode_marker(g, marker, bg),
            }
        )

    if len(used_markers) != len(markers):
        raise ValueError("all markers must be assigned")
    enriched.sort(key=lambda item: (item["bbox"][1], item["bbox"][0]))
    return bg, enriched


def _pair_relations(rectangles: list[dict]) -> list[dict]:
    relations = []
    for i in range(len(rectangles)):
        ar0, ac0, ar1, ac1 = rectangles[i]["bbox"]
        for j in range(i + 1, len(rectangles)):
            br0, bc0, br1, bc1 = rectangles[j]["bbox"]
            if ac1 < bc0:
                left_idx, right_idx = i, j
            elif bc1 < ac0:
                left_idx, right_idx = j, i
            else:
                relations.append({"kind": "column_overlap", "left": i, "right": j})
                continue

            left = rectangles[left_idx]
            right = rectangles[right_idx]
            lr0, lc0, lr1, lc1 = left["bbox"]
            rr0, rc0, rr1, rc1 = right["bbox"]
            row0 = max(lr0, rr0)
            row1 = min(lr1, rr1)
            if row0 > row1:
                relations.append(
                    {
                        "kind": "no_row_overlap",
                        "left": left_idx,
                        "right": right_idx,
                    }
                )
                continue

            gap0 = lc1 + 1
            gap1 = rc0 - 1
            if gap0 > gap1:
                relations.append(
                    {
                        "kind": "touching_columns",
                        "left": left_idx,
                        "right": right_idx,
                    }
                )
                continue

            blockers = []
            for k, rect in enumerate(rectangles):
                if k in (left_idx, right_idx):
                    continue
                kr0, kc0, kr1, kc1 = rect["bbox"]
                if _ranges_overlap(gap0, gap1, kc0, kc1) and _ranges_overlap(row0, row1, kr0, kr1):
                    blockers.append(k)

            relations.append(
                {
                    "kind": "blocked" if blockers else "fill",
                    "left": left_idx,
                    "right": right_idx,
                    "row0": row0,
                    "row1": row1,
                    "gap0": gap0,
                    "gap1": gap1,
                    "fill_color": left["fill_color"],
                    "blockers": blockers,
                }
            )
    return relations


def main(grid: np.ndarray) -> np.ndarray:
    """
    Erase the orange L markers and fill each unobstructed horizontal gap using the left rectangle's decoded gap color.

    Input structure:
    - A black background contains 3-5 solid blue rectangles placed at separated column spans.
    - Each rectangle has a directly-above orange 2x2 L-triomino marker whose missing corner encodes a color.
    - Some rectangle pairs overlap in row span, some do not, and some potential gaps are crossed by a third rectangle's bbox.

    Rule:
    - For every blue rectangle, read the missing corner of its orange L marker: top-left -> green, top-right -> teal, bottom-left -> yellow, bottom-right -> magenta.
    - Erase all orange marker pixels to black.
    - For each left-right rectangle pair with disjoint column spans and overlapping row spans, consider the rectangular gap between their bboxes over the shared rows.
    - Fill that gap iff no third blue rectangle bbox intersects both the gap columns and the shared row span; use the left rectangle's decoded color. Keep all blue rectangle pixels unchanged.

    Key insight:
    - The L marker does not recolor its own rectangle; it stores the color that the rectangle projects into valid horizontal gaps, so decoding the marker and checking bbox-level visibility are both required.
    """
    g = np.asarray(grid, dtype=int)
    bg, rectangles = _scene(g)
    out = g.copy()

    for rect in rectangles:
        r0, c0, r1, c1 = rect["marker_bbox"]
        out[r0 : r1 + 1, c0 : c1 + 1] = bg

    for relation in _pair_relations(rectangles):
        if relation["kind"] != "fill":
            continue
        row0, row1 = relation["row0"], relation["row1"]
        gap0, gap1 = relation["gap0"], relation["gap1"]
        out[row0 : row1 + 1, gap0 : gap1 + 1] = relation["fill_color"]

    return out


def _make_marker(missing: tuple[int, int]) -> np.ndarray:
    patch = np.full((2, 2), MARKER_COLOR, dtype=int)
    patch[missing] = BG
    return patch


def _paint_rect(grid: np.ndarray, top: int, left: int, height: int, width: int, missing: tuple[int, int]) -> None:
    grid[top : top + height, left : left + width] = RECT_COLOR
    marker_left = left + random.randint(0, width - 2)
    marker = _make_marker(missing)
    grid[top - 2 : top, marker_left : marker_left + 2] = marker


def _sample_layout() -> tuple[np.ndarray, list[dict]]:
    count = 4 + int(random.random() < 0.45)
    heights = [random.randint(2, 3) for _ in range(count)]
    widths = [random.randint(3, 5) for _ in range(count)]
    overlap01 = random.randint(1, min(heights[0] - 1, heights[1]))

    shared_row = random.randint(max(heights[2], heights[3]) + 1, max(heights[2], heights[3]) + 4)
    tops = [
        shared_row,
        shared_row + heights[0] - overlap01,
        shared_row - heights[2] + 1,
        shared_row - heights[3] + 1,
    ]
    if tops[2] < 2 or tops[3] < 2:
        raise ValueError("upper rectangles need room for markers")

    if count == 5:
        tops.append(tops[1] + heights[1] + random.randint(1, 2))

    bottoms = [top + height - 1 for top, height in zip(tops, heights)]
    h = max(16, max(bottoms) + random.randint(3, 6))
    if h > 30:
        raise ValueError("grid too tall")

    gap_hi = 2 if count == 5 else 3
    gaps = [random.randint(1, gap_hi) for _ in range(count - 1)]
    min_width = sum(widths) + sum(gaps) + 2
    if min_width > 30:
        raise ValueError("grid too wide")
    w = min(30, max(18, min_width + random.randint(0, min(4, 30 - min_width))))
    free = w - (sum(widths) + sum(gaps))
    if free < 2:
        raise ValueError("not enough side padding")
    left_margin = random.randint(1, free - 1)

    lefts = []
    cur = left_margin
    for width, gap in zip(widths, gaps + [0]):
        lefts.append(cur)
        cur += width + gap

    order_missing = random.sample(list(MISSING_CORNERS), 4)
    missings = order_missing[:4]
    if count == 5:
        missings.append(random.choice(MISSING_CORNERS))

    specs = []
    for idx in range(count):
        specs.append(
            {
                "top": tops[idx],
                "left": lefts[idx],
                "height": heights[idx],
                "width": widths[idx],
                "missing": missings[idx],
            }
        )

    grid = np.zeros((h, w), dtype=int)
    for spec in specs:
        _paint_rect(grid, spec["top"], spec["left"], spec["height"], spec["width"], spec["missing"])
    return grid, specs


def _semantic_checks(grid: np.ndarray) -> bool:
    bg, rectangles = _scene(grid)
    relations = _pair_relations(rectangles)
    erased = grid.copy()
    erased[erased == MARKER_COLOR] = bg

    fills = [rel for rel in relations if rel["kind"] == "fill"]
    blocked = [rel for rel in relations if rel["kind"] == "blocked"]
    no_row_overlap = [rel for rel in relations if rel["kind"] == "no_row_overlap"]
    if not fills or not blocked or not no_row_overlap:
        return False

    for rel in fills:
        patch = erased[rel["row0"] : rel["row1"] + 1, rel["gap0"] : rel["gap1"] + 1]
        if np.any(patch != bg):
            return False

    seen = set()
    for rel in fills:
        for r in range(rel["row0"], rel["row1"] + 1):
            for c in range(rel["gap0"], rel["gap1"] + 1):
                if (r, c) in seen:
                    return False
                seen.add((r, c))

    fill_colors = {rect["fill_color"] for rect in rectangles}
    if len(fill_colors) < 4:
        return False

    out = main(grid)
    if out.shape != grid.shape:
        return False
    if np.any(out == MARKER_COLOR):
        return False
    if np.array_equal(out, grid):
        return False

    for rect in rectangles:
        coords = rect["coords"]
        if not np.all(out[coords[:, 0], coords[:, 1]] == RECT_COLOR):
            return False
        mr0, mc0, mr1, mc1 = rect["marker_bbox"]
        if np.any(out[mr0 : mr1 + 1, mc0 : mc1 + 1] == MARKER_COLOR):
            return False

    return True


def generate_input() -> np.ndarray:
    for _ in range(MAX_GEN_ATTEMPTS):
        try:
            grid, _specs = _sample_layout()
        except ValueError:
            continue

        if random.random() < 0.5:
            grid = grid[:, ::-1].copy()

        if _semantic_checks(grid):
            return grid
    raise RuntimeError("failed to generate a valid input")
