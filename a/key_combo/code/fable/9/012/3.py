from __future__ import annotations

from collections import Counter, defaultdict
import random

import numpy as np


Coord = tuple[int, int]
BBox = tuple[int, int, int, int]

BG = 0
COLOR_POOL = tuple(range(1, 10))
SIZE_RANGE = (18, 26)
RECT_COUNT_CHOICES = (2, 3)
DISTRACTOR_COUNT_CHOICES = (1, 2)
RECT_EXTENT_RANGE = (2, 8)
MAX_ATTEMPTS = 500

QUADRANT_DELTAS: dict[str, tuple[Coord, Coord]] = {
    "NW": ((-1, 0), (0, -1)),
    "NE": ((-1, 0), (0, 1)),
    "SW": ((1, 0), (0, -1)),
    "SE": ((1, 0), (0, 1)),
}

CORNER_TO_OPENING = {
    "NW": {"solid": "SE", "hollow": "NW"},
    "NE": {"solid": "SW", "hollow": "NE"},
    "SW": {"solid": "NE", "hollow": "SW"},
    "SE": {"solid": "NW", "hollow": "SE"},
}


def _get_bg(grid: np.ndarray) -> int:
    return int(Counter(map(int, grid.flat)).most_common(1)[0][0])


def _components_of_color(grid: np.ndarray, color: int) -> list[np.ndarray]:
    mask = grid == int(color)
    seen = np.zeros_like(mask, dtype=bool)
    comps: list[np.ndarray] = []
    h, w = grid.shape
    for r, c in np.argwhere(mask):
        r = int(r)
        c = int(c)
        if seen[r, c]:
            continue
        stack = [(r, c)]
        seen[r, c] = True
        coords: list[Coord] = []
        while stack:
            rr, cc = stack.pop()
            coords.append((rr, cc))
            for nr, nc in ((rr - 1, cc), (rr + 1, cc), (rr, cc - 1), (rr, cc + 1)):
                if 0 <= nr < h and 0 <= nc < w and mask[nr, nc] and not seen[nr, nc]:
                    seen[nr, nc] = True
                    stack.append((nr, nc))
        comps.append(np.array(coords, dtype=int))
    return comps


def _corner_label(pivot: Coord, bbox: BBox) -> str | None:
    r0, c0, r1, c1 = bbox
    mapping = {
        (r0, c0): "NW",
        (r0, c1): "NE",
        (r1, c0): "SW",
        (r1, c1): "SE",
    }
    return mapping.get(tuple(map(int, pivot)))


def _bracket_cells(pivot: Coord, opening: str) -> list[Coord]:
    (dr1, dc1), (dr2, dc2) = QUADRANT_DELTAS[opening]
    r, c = pivot
    return [(r, c), (r + dr1, c + dc1), (r + dr2, c + dc2)]


def _analyze_bracket(coords: np.ndarray) -> dict | None:
    if coords.shape != (3, 2):
        return None
    pts = [tuple(map(int, p)) for p in coords]
    pt_set = set(pts)
    pivot: Coord | None = None
    for r, c in pts:
        deg = sum((nr, nc) in pt_set for nr, nc in ((r - 1, c), (r + 1, c), (r, c - 1), (r, c + 1)))
        if deg == 2:
            pivot = (r, c)
            break
    if pivot is None:
        return None
    arms = [(r - pivot[0], c - pivot[1]) for r, c in pts if (r, c) != pivot]
    vertical = [dr for dr, dc in arms if dc == 0 and abs(dr) == 1]
    horizontal = [dc for dr, dc in arms if dr == 0 and abs(dc) == 1]
    if len(vertical) != 1 or len(horizontal) != 1:
        return None
    opening = {
        (-1, -1): "NW",
        (-1, 1): "NE",
        (1, -1): "SW",
        (1, 1): "SE",
    }[(vertical[0], horizontal[0])]
    return {
        "coords": coords,
        "pivot": pivot,
        "opening": opening,
    }


def _draw_rectangle(out: np.ndarray, rect: dict) -> None:
    r0, c0, r1, c1 = rect["bbox"]
    color = int(rect["color"])
    if rect["mode"] == "solid":
        out[r0 : r1 + 1, c0 : c1 + 1] = color
        return
    out[r0, c0 : c1 + 1] = color
    out[r1, c0 : c1 + 1] = color
    out[r0 : r1 + 1, c0] = color
    out[r0 : r1 + 1, c1] = color


def _extract_scene(grid: np.ndarray) -> tuple[list[dict], list[dict], int]:
    bg = _get_bg(grid)
    brackets_by_color: dict[int, list[dict]] = defaultdict(list)
    leftovers: list[dict] = []
    for color in sorted(set(map(int, grid.flat)) - {bg}):
        for coords in _components_of_color(grid, color):
            bracket = _analyze_bracket(coords)
            if bracket is None:
                leftovers.append({"color": color, "coords": coords})
            else:
                bracket["color"] = color
                brackets_by_color[color].append(bracket)

    matched: list[dict] = []
    unmatched: list[dict] = list(leftovers)
    for color, brackets in brackets_by_color.items():
        if len(brackets) != 2:
            unmatched.extend({"color": color, "coords": b["coords"]} for b in brackets)
            continue
        a, b = brackets
        pa, pb = a["pivot"], b["pivot"]
        if pa[0] == pb[0] or pa[1] == pb[1]:
            unmatched.extend({"color": color, "coords": x["coords"]} for x in brackets)
            continue
        bbox = (min(pa[0], pb[0]), min(pa[1], pb[1]), max(pa[0], pb[0]), max(pa[1], pb[1]))
        corner_a = _corner_label(pa, bbox)
        corner_b = _corner_label(pb, bbox)
        diagonal_ok = {corner_a, corner_b} in ({"NW", "SE"}, {"NE", "SW"})
        if not diagonal_ok:
            unmatched.extend({"color": color, "coords": x["coords"]} for x in brackets)
            continue
        if (
            a["opening"] == CORNER_TO_OPENING[corner_a]["solid"]
            and b["opening"] == CORNER_TO_OPENING[corner_b]["solid"]
        ):
            mode = "solid"
        elif (
            a["opening"] == CORNER_TO_OPENING[corner_a]["hollow"]
            and b["opening"] == CORNER_TO_OPENING[corner_b]["hollow"]
        ):
            mode = "hollow"
        else:
            unmatched.extend({"color": color, "coords": x["coords"]} for x in brackets)
            continue
        r0, c0, r1, c1 = bbox
        matched.append(
            {
                "color": color,
                "bbox": bbox,
                "mode": mode,
                "area": (r1 - r0 + 1) * (c1 - c0 + 1),
            }
        )
    return matched, unmatched, bg


def _render_scene(shape: tuple[int, int], rects: list[dict], unmatched: list[dict], bg: int) -> np.ndarray:
    out = np.full(shape, int(bg), dtype=int)
    for rect in sorted(rects, key=lambda item: item["area"], reverse=True):
        _draw_rectangle(out, rect)
    for bracket in unmatched:
        coords = bracket["coords"]
        out[coords[:, 0], coords[:, 1]] = int(bracket["color"])
    return out


def main(grid: np.ndarray) -> np.ndarray:
    """
    Complete each same-color diagonal corner-bracket pair into a solid rectangle or a hollow frame, then stack the implied rectangles by area.

    Input structure:
    - A dominant background contains several monochrome 3-cell L-shaped corner brackets.
    - For every valid color, exactly two brackets share that color, and their bend cells are opposite corners of one axis-aligned rectangle.
    - Each pair is either inward-facing toward that rectangle's interior or outward-facing away from it; one or two singleton brackets may appear as distractors.

    Rule:
    - Pair brackets by color and treat each pair's bend cells as diagonal rectangle corners.
    - If both brackets open toward the rectangle interior, fill the whole rectangle with that color; if both open away, draw only the rectangle's 1-cell border.
    - Paint all implied rectangles from largest area to smallest area so smaller ones overwrite larger ones, and finally keep every unpaired bracket unchanged on top.

    Key insight:
    - The rectangles are never drawn in the input: the bracket geometry simultaneously identifies each hidden rectangle and encodes whether it should become solid or hollow, so the final overlap pattern only becomes determined after reconstructing every rectangle and ordering them by area.
    """
    g = np.asarray(grid, dtype=int)
    assert g.ndim == 2
    rects, unmatched, bg = _extract_scene(g)
    return _render_scene(g.shape, rects, unmatched, bg)


def _bbox_contains(a: BBox, b: BBox) -> bool:
    return a[0] <= b[0] and a[1] <= b[1] and a[2] >= b[2] and a[3] >= b[3]


def _bboxes_overlap(a: BBox, b: BBox) -> bool:
    return not (a[2] < b[0] or b[2] < a[0] or a[3] < b[1] or b[3] < a[1])


def _rect_to_brackets(rect: dict) -> list[dict]:
    r0, c0, r1, c1 = rect["bbox"]
    corners = {
        "NW": (r0, c0),
        "NE": (r0, c1),
        "SW": (r1, c0),
        "SE": (r1, c1),
    }
    diagonal = ("NW", "SE") if rect["diag"] == 0 else ("NE", "SW")
    brackets = []
    for corner in diagonal:
        opening = CORNER_TO_OPENING[corner][rect["mode"]]
        coords = np.array(_bracket_cells(corners[corner], opening), dtype=int)
        brackets.append({"color": rect["color"], "coords": coords})
    return brackets


def _pairwise_overlap(rects: list[dict]) -> bool:
    for i, rect_a in enumerate(rects):
        for rect_b in rects[i + 1 :]:
            if not _bboxes_overlap(rect_a["bbox"], rect_b["bbox"]):
                return False
    return True


def _visible_without(rects: list[dict], unmatched: list[dict], shape: tuple[int, int], idx: int) -> bool:
    expected = _render_scene(shape, rects, unmatched, BG)
    reduced = _render_scene(shape, rects[:idx] + rects[idx + 1 :], unmatched, BG)
    return not np.array_equal(expected, reduced)


def _variant_rects(rects: list[dict], *, mode: str | None = None, reverse: bool = False) -> list[dict]:
    changed = []
    for rect in rects:
        clone = dict(rect)
        if mode is not None:
            clone["mode"] = mode
        changed.append(clone)
    changed.sort(key=lambda item: item["area"], reverse=not reverse)
    return changed


def _render_in_order(shape: tuple[int, int], rects: list[dict], unmatched: list[dict], bg: int) -> np.ndarray:
    out = np.full(shape, int(bg), dtype=int)
    for rect in rects:
        _draw_rectangle(out, rect)
    for bracket in unmatched:
        coords = bracket["coords"]
        out[coords[:, 0], coords[:, 1]] = int(bracket["color"])
    return out


def _sample_rectangles(h: int, w: int) -> list[dict] | None:
    n_rects = random.choice(RECT_COUNT_CHOICES)
    center_r = random.randint(6, h - 7)
    center_c = random.randint(6, w - 7)
    modes = ["solid", "hollow"]
    if n_rects == 3:
        modes.append(random.choice(modes))
    random.shuffle(modes)
    areas: set[int] = set()
    rects: list[dict] = []
    for idx in range(n_rects):
        placed = False
        for _ in range(100):
            top = random.randint(*RECT_EXTENT_RANGE)
            bottom = random.randint(*RECT_EXTENT_RANGE)
            left = random.randint(*RECT_EXTENT_RANGE)
            right = random.randint(*RECT_EXTENT_RANGE)
            r0, c0 = center_r - top, center_c - left
            r1, c1 = center_r + bottom, center_c + right
            if r0 < 1 or c0 < 1 or r1 > h - 2 or c1 > w - 2:
                continue
            area = (r1 - r0 + 1) * (c1 - c0 + 1)
            bbox = (r0, c0, r1, c1)
            if area in areas:
                continue
            if any(_bbox_contains(bbox, other["bbox"]) or _bbox_contains(other["bbox"], bbox) for other in rects):
                continue
            rects.append(
                {
                    "bbox": bbox,
                    "area": area,
                    "mode": modes[idx],
                    "diag": random.randint(0, 1),
                }
            )
            areas.add(area)
            placed = True
            break
        if not placed:
            return None
    if not _pairwise_overlap(rects):
        return None
    return rects


def _brackets_are_clear(brackets: list[dict], shape: tuple[int, int]) -> bool:
    h, w = shape
    occupied: dict[Coord, int] = {}
    cell_sets: list[set[Coord]] = []
    for bracket in brackets:
        coords = [tuple(map(int, p)) for p in bracket["coords"]]
        for r, c in coords:
            if not (0 <= r < h and 0 <= c < w):
                return False
            if (r, c) in occupied:
                return False
        cell_sets.append(set(coords))
        for cell in coords:
            occupied[cell] = bracket["color"]
    for i, cells_a in enumerate(cell_sets):
        for cells_b in cell_sets[i + 1 :]:
            for r1, c1 in cells_a:
                for r2, c2 in cells_b:
                    if max(abs(r1 - r2), abs(c1 - c2)) <= 1:
                        return False
    return True


def _sample_distractors(
    h: int,
    w: int,
    rects: list[dict],
    used_cells: set[Coord],
    count: int,
) -> list[dict] | None:
    blocked = np.zeros((h, w), dtype=bool)
    for rect in rects:
        r0, c0, r1, c1 = rect["bbox"]
        rr0, cc0 = max(0, r0 - 1), max(0, c0 - 1)
        rr1, cc1 = min(h - 1, r1 + 1), min(w - 1, c1 + 1)
        blocked[rr0 : rr1 + 1, cc0 : cc1 + 1] = True
    distractors: list[dict] = []
    attempts = 0
    while len(distractors) < count and attempts < 300:
        attempts += 1
        pivot = (random.randint(1, h - 2), random.randint(1, w - 2))
        opening = random.choice(tuple(QUADRANT_DELTAS))
        coords = _bracket_cells(pivot, opening)
        if any(blocked[r, c] for r, c in coords):
            continue
        cells = set(coords)
        if any(cell in used_cells for cell in cells):
            continue
        if any(max(abs(r1 - r2), abs(c1 - c2)) <= 1 for r1, c1 in cells for r2, c2 in used_cells):
            continue
        distractors.append({"coords": np.array(coords, dtype=int)})
        used_cells |= cells
    return distractors if len(distractors) == count else None


def _semantic_checks(inp: np.ndarray, rects: list[dict], unmatched: list[dict], expected: np.ndarray) -> bool:
    if not np.array_equal(main(inp), expected):
        return False
    if np.array_equal(_render_in_order(inp.shape, _variant_rects(rects, reverse=True), unmatched, BG), expected):
        return False
    if np.array_equal(_render_in_order(inp.shape, _variant_rects(rects, mode="solid"), unmatched, BG), expected):
        return False
    if np.array_equal(_render_in_order(inp.shape, _variant_rects(rects, mode="hollow"), unmatched, BG), expected):
        return False
    if not all(_visible_without(rects, unmatched, inp.shape, i) for i in range(len(rects))):
        return False
    if not any(rect["mode"] == "solid" for rect in rects):
        return False
    if not any(rect["mode"] == "hollow" for rect in rects):
        return False
    return True


def _gen_attempt() -> np.ndarray | None:
    h = random.randint(*SIZE_RANGE)
    w = random.randint(*SIZE_RANGE)
    rects = _sample_rectangles(h, w)
    if rects is None:
        return None

    colors = random.sample(COLOR_POOL, len(rects) + random.choice(DISTRACTOR_COUNT_CHOICES))
    for rect, color in zip(rects, colors):
        rect["color"] = color

    brackets: list[dict] = []
    for rect in rects:
        brackets.extend(_rect_to_brackets(rect))
    if not _brackets_are_clear(brackets, (h, w)):
        return None

    used_cells = {tuple(map(int, p)) for bracket in brackets for p in bracket["coords"]}
    distractor_count = len(colors) - len(rects)
    distractors = _sample_distractors(h, w, rects, used_cells, distractor_count)
    if distractors is None:
        return None
    for dist, color in zip(distractors, colors[len(rects) :]):
        dist["color"] = color

    inp = np.full((h, w), BG, dtype=int)
    for bracket in brackets + distractors:
        coords = bracket["coords"]
        inp[coords[:, 0], coords[:, 1]] = int(bracket["color"])

    unmatched = [{"color": d["color"], "coords": d["coords"]} for d in distractors]
    expected = _render_scene(inp.shape, rects, unmatched, BG)
    if not _semantic_checks(inp, rects, unmatched, expected):
        return None
    return inp


def generate_input() -> np.ndarray:
    for _ in range(MAX_ATTEMPTS):
        grid = _gen_attempt()
        if grid is not None:
            return grid
    raise RuntimeError("failed to generate a valid input")
