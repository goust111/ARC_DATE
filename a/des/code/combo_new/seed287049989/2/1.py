import random
from collections import deque

import numpy as np


BG = 0
WALL = 5
MARKER = 9
ANCHOR_COLORS = (1, 2, 3, 4, 6, 7, 8)

GEN_RETRIES = 300
REGION_COUNT_CHOICES = (2, 3, 4)


def _neighbors4(r: int, c: int) -> tuple[tuple[int, int], ...]:
    return ((r - 1, c), (r + 1, c), (r, c - 1), (r, c + 1))


def _components(mask: np.ndarray) -> list[list[tuple[int, int]]]:
    h, w = mask.shape
    seen = np.zeros((h, w), dtype=bool)
    comps: list[list[tuple[int, int]]] = []

    for r in range(h):
        for c in range(w):
            if not mask[r, c] or seen[r, c]:
                continue
            seen[r, c] = True
            q = deque([(r, c)])
            comp: list[tuple[int, int]] = []
            while q:
                cr, cc = q.popleft()
                comp.append((cr, cc))
                for nr, nc in _neighbors4(cr, cc):
                    if 0 <= nr < h and 0 <= nc < w and mask[nr, nc] and not seen[nr, nc]:
                        seen[nr, nc] = True
                        q.append((nr, nc))
            comps.append(comp)
    return comps


def _bbox(coords: list[tuple[int, int]]) -> tuple[int, int, int, int]:
    rows = [r for r, _ in coords]
    cols = [c for _, c in coords]
    return min(rows), min(cols), max(rows), max(cols)


def _rect_ring_bbox(coords: list[tuple[int, int]]) -> tuple[int, int, int, int]:
    r0, c0, r1, c1 = _bbox(coords)
    ring = {
        (r, c)
        for r in range(r0, r1 + 1)
        for c in range(c0, c1 + 1)
        if r in (r0, r1) or c in (c0, c1)
    }
    assert set(coords) == ring, "grey component must be a single rectangular ring"
    assert r1 - r0 >= 2 and c1 - c0 >= 2, "region must enclose at least one interior cell"
    return r0, c0, r1, c1


def _find_anchor_color(grid: np.ndarray, r0: int, c0: int, r1: int, c1: int) -> int:
    h, w = grid.shape
    anchors: list[int] = []

    if r0 > 0:
        for c in range(c0, c1 + 1):
            val = int(grid[r0 - 1, c])
            if val not in (BG, WALL):
                anchors.append(val)
    if r1 + 1 < h:
        for c in range(c0, c1 + 1):
            val = int(grid[r1 + 1, c])
            if val not in (BG, WALL):
                anchors.append(val)
    if c0 > 0:
        for r in range(r0, r1 + 1):
            val = int(grid[r, c0 - 1])
            if val not in (BG, WALL):
                anchors.append(val)
    if c1 + 1 < w:
        for r in range(r0, r1 + 1):
            val = int(grid[r, c1 + 1])
            if val not in (BG, WALL):
                anchors.append(val)

    assert len(anchors) == 1, "each region must have exactly one exterior color anchor"
    return anchors[0]


def _apply_region_rule(out: np.ndarray, src: np.ndarray, r0: int, c0: int, r1: int, c1: int, anchor: int) -> None:
    inner = out[r0 + 1 : r1, c0 + 1 : c1]
    inner[inner == BG] = anchor

    markers = np.argwhere(src[r0 + 1 : r1, c0 + 1 : c1] == MARKER)
    inner_h = r1 - r0 - 1
    inner_w = c1 - c0 - 1
    erase_rows: set[int] = set()
    erase_cols: set[int] = set()

    for mr, mc in markers:
        gr = r0 + 1 + int(mr)
        gc = c0 + 1 + int(mc)
        if inner_w > inner_h:
            erase_cols.add(gc)
        else:
            erase_rows.add(gr)

    for rr in erase_rows:
        row = out[rr, c0 + 1 : c1]
        row[(row == anchor) | (row == MARKER)] = BG
    for cc in erase_cols:
        col = out[r0 + 1 : r1, cc]
        col[(col == anchor) | (col == MARKER)] = BG

    for mr, mc in markers:
        out[r0 + 1 + int(mr), c0 + 1 + int(mc)] = BG


def main(grid: np.ndarray) -> np.ndarray:
    """
    Fill each grey-walled rectangle from its exterior anchor color, then erase marker-indicated row or column stripes.

    Input structure:
    - The background is black.
    - One or more non-overlapping grey rectangular wall rings appear in the grid, each with wall thickness 1.
    - Each rectangle has exactly one non-grey, non-background anchor pixel touching one outside side of its wall.
    - Inside a rectangle, cells are background except for zero or more maroon single-pixel markers.

    Rule:
    - For each grey rectangle, find its unique adjacent outside anchor and use that color to fill every background cell in the rectangle interior.
    - Then inspect the maroon markers inside that same rectangle. If the marker's horizontal span inside the rectangle is greater than its vertical span, treat it as a column marker and erase that interior column; otherwise treat it as a row marker and erase that interior row.
    - Marker pixels themselves are also erased to background, while grey walls and exterior anchor pixels remain unchanged.

    Key insight:
    - The anchor outside each wall determines the fill color, while the rectangle's aspect ratio resolves whether an interior maroon marker removes a row or a column.
    """
    g = np.asarray(grid, dtype=int)
    assert g.ndim == 2, "expected a 2D grid"

    out = g.copy()
    wall_components = _components(g == WALL)
    assert wall_components, "expected at least one grey rectangular region"

    for comp in wall_components:
        r0, c0, r1, c1 = _rect_ring_bbox(comp)
        anchor = _find_anchor_color(g, r0, c0, r1, c1)
        _apply_region_rule(out, g, r0, c0, r1, c1, anchor)

    return out


def _sample_inner_dims(kind: str, n_regions: int, rng: random.Random) -> tuple[int, int]:
    if n_regions >= 4:
        short_max, extra_max = 4, 3
    elif n_regions == 3:
        short_max, extra_max = 5, 4
    else:
        short_max, extra_max = 6, 5

    if kind == "wide":
        inner_h = rng.randint(2, short_max)
        inner_w = rng.randint(inner_h + 1, min(9, inner_h + extra_max))
    elif kind == "tall":
        inner_w = rng.randint(2, short_max)
        inner_h = rng.randint(inner_w + 1, min(9, inner_w + extra_max))
    else:
        side = rng.randint(2, min(7, short_max + 1))
        inner_h = side
        inner_w = side
    return inner_h, inner_w


def _choose_anchor_cell(r0: int, c0: int, r1: int, c1: int, rng: random.Random) -> tuple[int, int]:
    side = rng.choice(("top", "bottom", "left", "right"))
    if side == "top":
        return r0 - 1, rng.randint(c0 + 1, c1 - 1)
    if side == "bottom":
        return r1 + 1, rng.randint(c0 + 1, c1 - 1)
    if side == "left":
        return rng.randint(r0 + 1, r1 - 1), c0 - 1
    return rng.randint(r0 + 1, r1 - 1), c1 + 1


def _choose_marker_positions(
    kind: str,
    r0: int,
    c0: int,
    inner_h: int,
    inner_w: int,
    count: int,
    rng: random.Random,
) -> list[tuple[int, int]]:
    rows = list(range(r0 + 1, r0 + 1 + inner_h))
    cols = list(range(c0 + 1, c0 + 1 + inner_w))
    markers: list[tuple[int, int]] = []

    if kind == "wide":
        for col in rng.sample(cols, count):
            markers.append((rng.choice(rows), col))
    else:
        for row in rng.sample(rows, count):
            markers.append((row, rng.choice(cols)))
    return markers


def _kind_order(n_regions: int, rng: random.Random) -> list[str]:
    kinds = ["wide", rng.choice(("tall", "square"))]
    while len(kinds) < n_regions:
        kinds.append(rng.choice(("wide", "tall", "square")))
    rng.shuffle(kinds)
    return kinds


def _sample_specs(rng: random.Random) -> list[dict]:
    n_regions = rng.choice(REGION_COUNT_CHOICES)
    kinds = _kind_order(n_regions, rng)
    colors = rng.sample(list(ANCHOR_COLORS), n_regions)

    specs: list[dict] = []
    for i, (kind, color) in enumerate(zip(kinds, colors)):
        inner_h, inner_w = _sample_inner_dims(kind, n_regions, rng)
        stripe_capacity = inner_w if kind == "wide" else inner_h
        if i < 2:
            marker_count = rng.randint(1, min(3, stripe_capacity))
        else:
            marker_count = rng.randint(0, min(3, stripe_capacity))
        specs.append(
            {
                "kind": kind,
                "anchor_color": color,
                "inner_h": inner_h,
                "inner_w": inner_w,
                "outer_h": inner_h + 2,
                "outer_w": inner_w + 2,
                "marker_count": marker_count,
            }
        )

    specs.sort(key=lambda spec: spec["outer_h"] * spec["outer_w"], reverse=True)
    return specs


def _place_specs(specs: list[dict], rng: random.Random) -> tuple[int, int, list[dict]] | None:
    n_regions = len(specs)
    base = 12 + 3 * n_regions
    h = rng.randint(base, 30)
    w = rng.randint(base, 30)
    occupied = np.zeros((h, w), dtype=bool)
    placed: list[dict] = []

    for spec in specs:
        outer_h = int(spec["outer_h"])
        outer_w = int(spec["outer_w"])
        positions: list[tuple[int, int]] = []

        for r0 in range(1, h - outer_h):
            for c0 in range(1, w - outer_w):
                if occupied[r0 - 1 : r0 + outer_h + 1, c0 - 1 : c0 + outer_w + 1].any():
                    continue
                positions.append((r0, c0))

        if not positions:
            return None

        r0, c0 = rng.choice(positions)
        r1 = r0 + outer_h - 1
        c1 = c0 + outer_w - 1
        occupied[r0 - 1 : r0 + outer_h + 1, c0 - 1 : c0 + outer_w + 1] = True

        placed_spec = dict(spec)
        placed_spec["r0"] = r0
        placed_spec["c0"] = c0
        placed_spec["r1"] = r1
        placed_spec["c1"] = c1
        placed_spec["anchor_pos"] = _choose_anchor_cell(r0, c0, r1, c1, rng)
        placed_spec["markers"] = _choose_marker_positions(
            placed_spec["kind"],
            r0,
            c0,
            placed_spec["inner_h"],
            placed_spec["inner_w"],
            placed_spec["marker_count"],
            rng,
        )
        placed.append(placed_spec)

    return h, w, placed


def _render_input_and_expected(h: int, w: int, specs: list[dict]) -> tuple[np.ndarray, np.ndarray]:
    inp = np.full((h, w), BG, dtype=int)

    for spec in specs:
        r0 = int(spec["r0"])
        c0 = int(spec["c0"])
        r1 = int(spec["r1"])
        c1 = int(spec["c1"])
        color = int(spec["anchor_color"])
        ar, ac = spec["anchor_pos"]

        inp[r0, c0 : c1 + 1] = WALL
        inp[r1, c0 : c1 + 1] = WALL
        inp[r0 : r1 + 1, c0] = WALL
        inp[r0 : r1 + 1, c1] = WALL
        inp[int(ar), int(ac)] = color

        for mr, mc in spec["markers"]:
            inp[int(mr), int(mc)] = MARKER

    expected = inp.copy()
    for spec in specs:
        r0 = int(spec["r0"])
        c0 = int(spec["c0"])
        r1 = int(spec["r1"])
        c1 = int(spec["c1"])
        color = int(spec["anchor_color"])

        inner = expected[r0 + 1 : r1, c0 + 1 : c1]
        inner[inner == BG] = color

        if spec["inner_w"] > spec["inner_h"]:
            erase_cols = {int(mc) for _mr, mc in spec["markers"]}
            for cc in erase_cols:
                col = expected[r0 + 1 : r1, cc]
                col[(col == color) | (col == MARKER)] = BG
        else:
            erase_rows = {int(mr) for mr, _mc in spec["markers"]}
            for rr in erase_rows:
                row = expected[rr, c0 + 1 : c1]
                row[(row == color) | (row == MARKER)] = BG

        for mr, mc in spec["markers"]:
            expected[int(mr), int(mc)] = BG

    return inp, expected


def _gen_attempt(rng: random.Random) -> np.ndarray | None:
    specs = _sample_specs(rng)
    placed = _place_specs(specs, rng)
    if placed is None:
        return None

    h, w, placed_specs = placed
    inp, expected = _render_input_and_expected(h, w, placed_specs)

    try:
        actual = main(inp)
    except Exception:
        return None
    if not np.array_equal(actual, expected):
        return None
    return inp


def generate_input() -> np.ndarray:
    rng = random.Random()
    for _ in range(GEN_RETRIES):
        grid = _gen_attempt(rng)
        if grid is not None:
            return grid
    raise RuntimeError(f"failed to generate a valid sample after {GEN_RETRIES} attempts")
