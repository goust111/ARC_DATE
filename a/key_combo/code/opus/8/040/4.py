from __future__ import annotations

import itertools
import random
import sys
from pathlib import Path

import numpy as np

_ARC_UTILS_DIR = Path("/home/lixinyu/snap/new/solvers")
if str(_ARC_UTILS_DIR) not in sys.path:
    sys.path.append(str(_ARC_UTILS_DIR))

from arc_utils import Color, find_components, scale_sprite


MARKER_COUNT = 4
MARKER_COLORS = [Color.BLUE, Color.GREEN, Color.YELLOW, Color.GREY, Color.PINK, Color.ORANGE, Color.TEAL, Color.MAROON]
SPRITE_COLORS = [Color.BLUE, Color.GREEN, Color.YELLOW, Color.GREY, Color.PINK, Color.ORANGE, Color.TEAL, Color.MAROON]
GRID_HEIGHT_RANGE = (18, 24)
GRID_WIDTH_RANGE = (18, 24)
SPRITE_COUNT_RANGE = (4, 6)
SPRITE_AREA_RANGE = (3, 7)
MAX_GENERATION_ATTEMPTS = 600


def _sorted_markers(markers: list[dict]) -> list[dict]:
    return sorted(markers, key=lambda m: (int(m["r"]), int(m["c"])))


def _extract_objects(grid: np.ndarray) -> tuple[list[dict], list[dict]]:
    comps = find_components(grid, bg=Color.BLACK, connectivity=4, monochromatic=True, return_format="dict")
    markers: list[dict] = []
    sprites: list[dict] = []
    for comp in comps:
        color = int(comp["color"])
        coords = comp["coords"]
        area = int(comp["area"])
        if area == 1:
            r, c = map(int, coords[0])
            markers.append({"r": r, "c": c, "color": color, "coords": coords})
        else:
            sprites.append({"color": color, "coords": coords, "area": area})
    return _sorted_markers(markers), sprites


def _validate_input(markers: list[dict], sprites: list[dict]) -> None:
    assert len(markers) >= MARKER_COUNT, "need at least four marker pixels to form red-corner rectangles"
    rows = [int(m["r"]) for m in markers]
    cols = [int(m["c"]) for m in markers]
    colors = [int(m["color"]) for m in markers]
    assert len(set(rows)) == len(rows), "marker rows must be unique"
    assert len(set(cols)) == len(cols), "marker cols must be unique"
    assert len(set(colors)) == len(colors), "marker colors must be unique"
    assert Color.RED not in colors and Color.BLACK not in colors

    marker_rows = set(rows)
    marker_cols = set(cols)
    assert sprites, "expected sprite components"
    for sprite in sprites:
        assert int(sprite["area"]) >= 3, "sprites must be larger than marker singletons"
        for r, c in sprite["coords"]:
            assert int(r) not in marker_rows and int(c) not in marker_cols, "sprites may not touch marker rows/cols"


def _red_points(markers: list[dict]) -> set[tuple[int, int]]:
    return {
        (int(a["r"]), int(b["c"]))
        for i, a in enumerate(markers)
        for j, b in enumerate(markers)
        if i != j
    }


def _candidate_rectangles(markers: list[dict]) -> list[dict]:
    red = _red_points(markers)
    rows = sorted({int(m["r"]) for m in markers})
    cols = sorted({int(m["c"]) for m in markers})
    rects: list[dict] = []
    for r0, r1 in itertools.combinations(rows, 2):
        if r1 - r0 <= 1:
            continue
        for c0, c1 in itertools.combinations(cols, 2):
            if c1 - c0 <= 1:
                continue
            corners = ((r0, c0), (r0, c1), (r1, c0), (r1, c1))
            if all(corner in red for corner in corners):
                rects.append(
                    {
                        "bbox": (r0, c0, r1, c1),
                        "area": int((r1 - r0 - 1) * (c1 - c0 - 1)),
                    }
                )
    rects.sort(key=lambda rect: (rect["bbox"][0], rect["bbox"][1], rect["bbox"][2], rect["bbox"][3]))
    return rects


def _sprites_in_rect(rect: dict, sprites: list[dict]) -> list[dict]:
    r0, c0, r1, c1 = map(int, rect["bbox"])
    inside: list[dict] = []
    for sprite in sprites:
        coords = sprite["coords"]
        hit = (coords[:, 0] > r0) & (coords[:, 0] < r1) & (coords[:, 1] > c0) & (coords[:, 1] < c1)
        if bool(np.any(hit)):
            inside.append(sprite)
    return inside


def _score_rectangles(rects: list[dict], sprites: list[dict]) -> list[dict]:
    scored: list[dict] = []
    for rect in rects:
        inside = _sprites_in_rect(rect, sprites)
        colors = {int(sprite["color"]) for sprite in inside}
        scored.append(
            {
                **rect,
                "sprite_count": len(inside),
                "sprite_colors": colors,
                "sprites": inside,
            }
        )
    return scored


def _choose_rectangle(scored: list[dict]) -> dict:
    assert scored, "expected at least one candidate rectangle"
    best = sorted(
        scored,
        key=lambda rect: (-int(rect["sprite_count"]), int(rect["area"]), rect["bbox"][0], rect["bbox"][1], rect["bbox"][2], rect["bbox"][3]),
    )[0]
    assert int(best["sprite_count"]) >= 1, "selected rectangle must contain at least one sprite"
    return best


def _render_selected_interior(rect: dict) -> np.ndarray:
    r0, c0, r1, c1 = map(int, rect["bbox"])
    crop = np.zeros((r1 - r0 - 1, c1 - c0 - 1), dtype=int)
    for sprite in rect["sprites"]:
        local = sprite["coords"] - np.array([[r0 + 1, c0 + 1]])
        crop[local[:, 0], local[:, 1]] = int(sprite["color"])
    return crop


def main(grid: np.ndarray) -> np.ndarray:
    """
    Use guide-line intersections to choose a sprite-rich crop, then scale that crop by its sprite-color diversity.

    Input structure:
    - The grid has a black background, several isolated colored marker pixels, and several separated monochrome sprite components.
    - Marker pixels occupy distinct rows and distinct columns, while sprite pixels avoid every marker row and column.
    - Red is reserved for guide-line crossings rather than for the input objects themselves.

    Rule:
    - Extend every marker through its full row and column, treating off-marker intersections between different markers as red guide-line crossings.
    - Enumerate every axis-aligned rectangle whose four corners are such red crossings.
    - For each rectangle, count how many original sprite components have at least one pixel strictly inside it, pick the rectangle with the largest count, and break ties by smaller interior area.
    - Keep only the original sprite pixels and background from that rectangle's interior, then scale the cropped interior by the number of distinct sprite colors contained inside it.

    Key insight:
    - The guide lines are not part of the answer; they only define candidate crop windows, and the chosen window's internal color diversity determines the magnification factor.
    """
    g = np.asarray(grid, dtype=int)
    assert g.ndim == 2, "expected a 2D grid"

    markers, sprites = _extract_objects(g)
    _validate_input(markers, sprites)
    rects = _candidate_rectangles(markers)
    scored = _score_rectangles(rects, sprites)
    chosen = _choose_rectangle(scored)

    scale = len(chosen["sprite_colors"])
    assert scale >= 1
    crop = _render_selected_interior(chosen)
    return scale_sprite(crop, scale)


def _random_spaced_positions(length: int, count: int, gap_lo: int, gap_hi: int) -> list[int] | None:
    choices = list(range(2, length - 2))
    if len(choices) < count:
        return None
    for _ in range(250):
        vals = sorted(random.sample(choices, count))
        gaps = [b - a for a, b in zip(vals, vals[1:])]
        if all(gap_lo <= gap <= gap_hi for gap in gaps):
            return vals
    return None


def _random_sprite_shape(area: int) -> np.ndarray:
    for _ in range(250):
        pts = {(0, 0)}
        while len(pts) < area:
            frontier = []
            for r, c in pts:
                for dr, dc in ((-1, 0), (1, 0), (0, -1), (0, 1)):
                    nxt = (r + dr, c + dc)
                    if nxt not in pts:
                        frontier.append(nxt)
            pts.add(random.choice(frontier))
        rows = [r for r, _c in pts]
        cols = [c for _r, c in pts]
        min_r, min_c = min(rows), min(cols)
        norm = sorted((r - min_r, c - min_c) for r, c in pts)
        h = max(r for r, _c in norm) + 1
        w = max(c for _r, c in norm) + 1
        if h <= 4 and w <= 4:
            return np.array(norm, dtype=int)
    raise RuntimeError("failed to sample sprite shape")


def _sprite_cells(shape: np.ndarray, top: int, left: int) -> np.ndarray:
    return shape + np.array([[top, left]])


def _can_place(cells: np.ndarray, occupied: set[tuple[int, int]], h: int, w: int, marker_rows: set[int], marker_cols: set[int]) -> bool:
    for r, c in cells:
        rr = int(r)
        cc = int(c)
        if not (0 <= rr < h and 0 <= cc < w):
            return False
        if rr in marker_rows or cc in marker_cols:
            return False
        if (rr, cc) in occupied:
            return False
    for r, c in cells:
        rr = int(r)
        cc = int(c)
        for nr in range(rr - 1, rr + 2):
            for nc in range(cc - 1, cc + 2):
                if (nr, nc) in occupied and (nr, nc) not in {(int(x), int(y)) for x, y in cells}:
                    return False
    return True


def _paint_cells(grid: np.ndarray, cells: np.ndarray, color: int) -> None:
    for r, c in cells:
        grid[int(r), int(c)] = int(color)


def _interior_mask(shape: tuple[int, int], rects: list[dict]) -> np.ndarray:
    mask = np.zeros(shape, dtype=bool)
    for rect in rects:
        r0, c0, r1, c1 = map(int, rect["bbox"])
        mask[r0 + 1 : r1, c0 + 1 : c1] = True
    return mask


def _place_sprites(grid: np.ndarray, markers: list[dict], rects: list[dict]) -> bool:
    h, w = grid.shape
    occupied = {(int(m["r"]), int(m["c"])) for m in markers}
    marker_rows = {int(m["r"]) for m in markers}
    marker_cols = {int(m["c"]) for m in markers}
    union_mask = _interior_mask(grid.shape, rects)
    n_sprites = random.randint(*SPRITE_COUNT_RANGE)
    palette = random.sample(SPRITE_COLORS, random.randint(2, min(4, len(SPRITE_COLORS))))

    for _ in range(n_sprites):
        shape = _random_sprite_shape(random.randint(*SPRITE_AREA_RANGE))
        box_h = int(shape[:, 0].max()) + 1
        box_w = int(shape[:, 1].max()) + 1
        color = random.choice(palette)
        placed = False
        for _ in range(160):
            prefer_rect = random.random() < 0.8
            if prefer_rect:
                rect = random.choice(rects)
                r0, c0, r1, c1 = map(int, rect["bbox"])
                top_min, top_max = r0 + 1, r1 - box_h
                left_min, left_max = c0 + 1, c1 - box_w
                if top_min > top_max or left_min > left_max:
                    continue
                top = random.randint(top_min, top_max)
                left = random.randint(left_min, left_max)
            else:
                top = random.randint(0, h - box_h)
                left = random.randint(0, w - box_w)

            cells = _sprite_cells(shape, top, left)
            if not _can_place(cells, occupied, h, w, marker_rows, marker_cols):
                continue
            if prefer_rect:
                if not all(union_mask[int(r), int(c)] for r, c in cells):
                    continue
            _paint_cells(grid, cells, color)
            occupied.update((int(r), int(c)) for r, c in cells)
            placed = True
            break
        if not placed:
            return False
    return True


def _branch_ok(scored: list[dict], chosen: dict, want_tie: bool) -> bool:
    positive = [rect for rect in scored if int(rect["sprite_count"]) > 0]
    if len(positive) < 2:
        return False

    max_count = max(int(rect["sprite_count"]) for rect in scored)
    top = [rect for rect in scored if int(rect["sprite_count"]) == max_count]
    if want_tie:
        if len(top) < 2:
            return False
        min_area = min(int(rect["area"]) for rect in top)
        area_winners = [rect for rect in top if int(rect["area"]) == min_area]
        return len(area_winners) == 1 and area_winners[0]["bbox"] == chosen["bbox"]
    return len(top) == 1 and top[0]["bbox"] == chosen["bbox"]


def _gen_attempt() -> np.ndarray | None:
    h = random.randint(*GRID_HEIGHT_RANGE)
    w = random.randint(*GRID_WIDTH_RANGE)
    rows = _random_spaced_positions(h, MARKER_COUNT, 3, 7)
    cols = _random_spaced_positions(w, MARKER_COUNT, 3, 7)
    if rows is None or cols is None:
        return None

    permuted_cols = cols[:]
    random.shuffle(permuted_cols)
    marker_colors = random.sample(MARKER_COLORS, MARKER_COUNT)
    grid = np.zeros((h, w), dtype=int)
    markers: list[dict] = []
    for r, c, color in zip(rows, permuted_cols, marker_colors):
        grid[int(r), int(c)] = int(color)
        markers.append({"r": int(r), "c": int(c), "color": int(color), "coords": np.array([[int(r), int(c)]], dtype=int)})

    rects = _candidate_rectangles(_sorted_markers(markers))
    if len(rects) < 2:
        return None
    if max(int(rect["area"]) for rect in rects) < 6:
        return None

    if not _place_sprites(grid, markers, rects):
        return None

    parsed_markers, sprites = _extract_objects(grid)
    try:
        _validate_input(parsed_markers, sprites)
    except AssertionError:
        return None

    scored = _score_rectangles(_candidate_rectangles(parsed_markers), sprites)
    if not scored:
        return None
    if max(int(rect["sprite_count"]) for rect in scored) < 1:
        return None
    chosen = _choose_rectangle(scored)
    if len(chosen["sprite_colors"]) > 3:
        return None

    crop = _render_selected_interior(chosen)
    if max(crop.shape) * max(1, len(chosen["sprite_colors"])) > 30:
        return None

    want_tie = random.random() < 0.25
    if not _branch_ok(scored, chosen, want_tie):
        return None

    try:
        out = main(grid)
    except Exception:
        return None
    if out.ndim != 2 or max(out.shape) > 30:
        return None
    return grid


def generate_input() -> np.ndarray:
    for _ in range(MAX_GENERATION_ATTEMPTS):
        grid = _gen_attempt()
        if grid is not None:
            return grid
    raise RuntimeError("failed to generate a valid guide-line crop puzzle")
