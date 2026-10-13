from __future__ import annotations

import random
import sys
from pathlib import Path

import numpy as np

_UTILS_DIR = Path("/home/lixinyu/snap/new/solvers")
if str(_UTILS_DIR) not in sys.path:
    sys.path.append(str(_UTILS_DIR))

from arc_utils import Color, bbox_h, bbox_w, find_components


BACKGROUND = Color.BLACK
DIVIDER = Color.GREY
ANCHOR = Color.GREY
SCATTER_COLORS = (Color.BLUE, Color.RED, Color.GREEN, Color.YELLOW)
SLOT_OFFSETS = ((0, 0), (0, 1), (1, 0), (1, 1))  # UL, UR, LL, LR


def _intervals(separators: list[int], length: int) -> list[tuple[int, int]]:
    bounds = [-1, *separators, int(length)]
    spans = [(bounds[i] + 1, bounds[i + 1]) for i in range(len(bounds) - 1)]
    assert spans and all(end - start >= 5 for start, end in spans)
    return spans


def _separator_lines(grid: np.ndarray) -> tuple[list[int], list[int]]:
    rows = [r for r in range(grid.shape[0]) if np.all(grid[r, :] == DIVIDER)]
    cols = [c for c in range(grid.shape[1]) if np.all(grid[:, c] == DIVIDER)]
    assert 1 <= len(rows) <= 2 and 1 <= len(cols) <= 2
    return rows, cols


def _anchor_components(grid: np.ndarray) -> list[dict]:
    grey_components = find_components(
        grid,
        bg=BACKGROUND,
        connectivity=4,
        monochromatic=True,
        colors=[DIVIDER],
        return_format="dict",
    )
    anchors = []
    for comp in grey_components:
        box = comp["bbox"]
        if int(comp["area"]) == 4 and bbox_h(box) == 2 and bbox_w(box) == 2:
            anchors.append(comp)
    return anchors


def _quadrant_index(r: int, c: int, anchor_r: int, anchor_c: int) -> int:
    if r < anchor_r and c < anchor_c:
        return 0
    if r < anchor_r and c > anchor_c + 1:
        return 1
    if r > anchor_r + 1 and c < anchor_c:
        return 2
    if r > anchor_r + 1 and c > anchor_c + 1:
        return 3
    raise AssertionError("scatter pixel must lie in a diagonal quadrant of its anchor")


def _parse_grid(grid: np.ndarray) -> dict:
    g = np.asarray(grid, dtype=int)
    assert g.ndim == 2
    assert max(g.shape) <= 30
    assert set(map(int, np.unique(g))).issubset({BACKGROUND, DIVIDER, *SCATTER_COLORS})

    sep_rows, sep_cols = _separator_lines(g)
    row_spans = _intervals(sep_rows, g.shape[0])
    col_spans = _intervals(sep_cols, g.shape[1])
    anchors = _anchor_components(g)
    assert len(anchors) == len(row_spans) * len(col_spans)

    anchor_by_region: dict[tuple[int, int], tuple[int, int]] = {}
    for comp in anchors:
        r0, c0, r1, c1 = map(int, comp["bbox"])
        hits = []
        for ri, (rs, re) in enumerate(row_spans):
            for ci, (cs, ce) in enumerate(col_spans):
                if rs <= r0 and r1 < re and cs <= c0 and c1 < ce:
                    hits.append((ri, ci))
        assert len(hits) == 1
        anchor_by_region[hits[0]] = (r0, c0)
    assert len(anchor_by_region) == len(anchors)

    regions = []
    for ri, (rs, re) in enumerate(row_spans):
        for ci, (cs, ce) in enumerate(col_spans):
            assert (ri, ci) in anchor_by_region
            ar, ac = anchor_by_region[(ri, ci)]
            slot_colors = [ANCHOR] * 4
            singletons: dict[int, tuple[int, int, int]] = {}
            region = g[rs:re, cs:ce]
            coords = np.argwhere(np.isin(region, SCATTER_COLORS))
            assert 2 <= len(coords) <= 4
            for rr, cc in coords:
                r = int(rs + rr)
                c = int(cs + cc)
                color = int(g[r, c])
                slot = _quadrant_index(r, c, ar, ac)
                assert slot not in singletons
                singletons[slot] = (r, c, color)
                slot_colors[slot] = color

            regions.append(
                {
                    "ri": ri,
                    "ci": ci,
                    "rows": (rs, re),
                    "cols": (cs, ce),
                    "anchor": (ar, ac),
                    "slot_colors": tuple(slot_colors),
                    "singletons": singletons,
                }
            )

    return {
        "grid": g,
        "row_spans": row_spans,
        "col_spans": col_spans,
        "regions": regions,
    }


def _paint_aggregated(parsed: dict) -> np.ndarray:
    out = parsed["grid"].copy()
    for region in parsed["regions"]:
        for r, c, _color in region["singletons"].values():
            out[r, c] = BACKGROUND

        ar, ac = region["anchor"]
        for slot, color in enumerate(region["slot_colors"]):
            dr, dc = SLOT_OFFSETS[slot]
            out[ar + dr, ac + dc] = int(color)
    return out


def _plan_connections(parsed: dict, aggregated: np.ndarray) -> dict[tuple[int, int], int]:
    planned: dict[tuple[int, int], int] = {}
    regions = parsed["regions"]
    n_rows = len(parsed["row_spans"])
    n_cols = len(parsed["col_spans"])

    def paint_segment(a: tuple[int, int], b: tuple[int, int], color: int) -> None:
        ar, ac = a
        br, bc = b
        if ar == br:
            for c in range(min(ac, bc) + 1, max(ac, bc)):
                if int(aggregated[ar, c]) != BACKGROUND:
                    continue
                prev = planned.get((ar, c))
                assert prev in (None, color), "conflicting connection colors"
                planned[(ar, c)] = color
        else:
            assert ac == bc
            for r in range(min(ar, br) + 1, max(ar, br)):
                if int(aggregated[r, ac]) != BACKGROUND:
                    continue
                prev = planned.get((r, ac))
                assert prev in (None, color), "conflicting connection colors"
                planned[(r, ac)] = color

    for ri in range(n_rows):
        row_regions = [region for region in regions if region["ri"] == ri]
        for slot in range(4):
            groups: dict[int, list[tuple[int, tuple[int, int]]]] = {}
            for region in row_regions:
                color = int(region["slot_colors"][slot])
                if color == ANCHOR:
                    continue
                ar, ac = region["anchor"]
                dr, dc = SLOT_OFFSETS[slot]
                groups.setdefault(color, []).append((region["ci"], (ar + dr, ac + dc)))
            for color, items in groups.items():
                items.sort(key=lambda item: item[0])
                for (_, start), (_, end) in zip(items, items[1:]):
                    assert start[0] == end[0]
                    paint_segment(start, end, color)

    for ci in range(n_cols):
        col_regions = [region for region in regions if region["ci"] == ci]
        for slot in range(4):
            groups = {}
            for region in col_regions:
                color = int(region["slot_colors"][slot])
                if color == ANCHOR:
                    continue
                ar, ac = region["anchor"]
                dr, dc = SLOT_OFFSETS[slot]
                groups.setdefault(color, []).append((region["ri"], (ar + dr, ac + dc)))
            for color, items in groups.items():
                items.sort(key=lambda item: item[0])
                for (_, start), (_, end) in zip(items, items[1:]):
                    assert start[1] == end[1]
                    paint_segment(start, end, color)

    return planned


def _expected_output(parsed: dict) -> np.ndarray:
    aggregated = _paint_aggregated(parsed)
    planned = _plan_connections(parsed, aggregated)
    out = aggregated.copy()
    for (r, c), color in planned.items():
        out[r, c] = color
    return out


def main(grid: np.ndarray) -> np.ndarray:
    """
    Aggregate each region's scattered quadrant pixels into its 2x2 anchor, then connect aligned matching anchor cells across the lattice.

    Input structure:
    - A black grid is partitioned by full grey rows and columns into a 2x2 or 3x3 array of rectangular regions.
    - Each region contains one isolated solid grey 2x2 anchor block near its center.
    - Around each anchor, diagonal quadrants may contain single blue, red, green, or yellow pixels; each region has at least two such scatter pixels.

    Rule:
    - For every region, map each scatter pixel into the anchor cell with the same quadrant relation: upper-left, upper-right, lower-left, or lower-right.
    - Remove the original scatter pixels; anchor cells with no matching scatter remain grey.
    - After aggregation, whenever two anchor cells have the same color in the same anchor slot and their regions share a partition row or partition column, fill the black cells between consecutive such matches along that row or column with the shared color, without overwriting grey lattice lines.

    Key insight:
    - The scatter pixels encode both destination slot and future connection identity: quadrant chooses where the color lands inside the anchor, and the resulting aligned same-slot same-color anchor cells define the lattice segments to draw.
    """
    parsed = _parse_grid(np.asarray(grid, dtype=int))
    return _expected_output(parsed)


def _quadrant_choices(region_h: int, region_w: int, anchor_r: int, anchor_c: int) -> list[list[tuple[int, int]]]:
    return [
        [(r, c) for r in range(0, anchor_r) for c in range(0, anchor_c)],
        [(r, c) for r in range(0, anchor_r) for c in range(anchor_c + 2, region_w)],
        [(r, c) for r in range(anchor_r + 2, region_h) for c in range(0, anchor_c)],
        [(r, c) for r in range(anchor_r + 2, region_h) for c in range(anchor_c + 2, region_w)],
    ]


def _connected_slots(slot_grid: np.ndarray) -> set[tuple[int, int, int]]:
    nr, nc, _ = slot_grid.shape
    connected: set[tuple[int, int, int]] = set()
    for ri in range(nr):
        for slot in range(4):
            groups: dict[int, list[tuple[int, int]]] = {}
            for ci in range(nc):
                color = int(slot_grid[ri, ci, slot])
                if color != ANCHOR:
                    groups.setdefault(color, []).append((ri, ci))
            for items in groups.values():
                if len(items) >= 2:
                    connected.update((r, c, slot) for r, c in items)
    for ci in range(nc):
        for slot in range(4):
            groups = {}
            for ri in range(nr):
                color = int(slot_grid[ri, ci, slot])
                if color != ANCHOR:
                    groups.setdefault(color, []).append((ri, ci))
            for items in groups.values():
                if len(items) >= 2:
                    connected.update((r, c, slot) for r, c in items)
    return connected


def _has_cross_slot_repeat(slot_grid: np.ndarray) -> bool:
    seen: dict[int, set[int]] = {}
    for slot in range(4):
        for color in slot_grid[:, :, slot].ravel():
            color = int(color)
            if color == ANCHOR:
                continue
            seen.setdefault(color, set()).add(slot)
    return any(len(slots) >= 2 for slots in seen.values())


def _has_diagonal_duplicate(slot_grid: np.ndarray) -> bool:
    nr, nc, _ = slot_grid.shape
    for slot in range(4):
        groups: dict[int, list[tuple[int, int]]] = {}
        for ri in range(nr):
            for ci in range(nc):
                color = int(slot_grid[ri, ci, slot])
                if color != ANCHOR:
                    groups.setdefault(color, []).append((ri, ci))
        for items in groups.values():
            for i, (r1, c1) in enumerate(items):
                for r2, c2 in items[i + 1 :]:
                    if r1 != r2 and c1 != c2:
                        return True
    return False


def _make_slot_grid(rng: random.Random, nr: int, nc: int) -> np.ndarray:
    slot_grid = np.full((nr, nc, 4), ANCHOR, dtype=int)

    mode = rng.choices(("row", "col", "both"), weights=(4, 4, 2), k=1)[0]
    row_spec = None
    if mode in {"row", "both"}:
        ri = rng.randrange(nr)
        slot = rng.randrange(4)
        color = rng.choice(SCATTER_COLORS)
        count = 2 if nc == 2 else rng.choice((2, 3))
        cols = rng.sample(range(nc), count)
        for ci in cols:
            slot_grid[ri, ci, slot] = color
        row_spec = (ri, slot, color)

    if mode in {"col", "both"}:
        ci = rng.randrange(nc)
        slot = rng.randrange(4)
        if row_spec is not None and slot == row_spec[1] and slot_grid[row_spec[0], ci, slot] != ANCHOR:
            color = int(slot_grid[row_spec[0], ci, slot])
        else:
            color = rng.choice(SCATTER_COLORS)
        count = 2 if nr == 2 else rng.choice((2, 3))
        rows = rng.sample(range(nr), count)
        for ri in rows:
            if slot_grid[ri, ci, slot] not in (ANCHOR, color):
                return slot_grid
            slot_grid[ri, ci, slot] = color

    for ri in range(nr):
        for ci in range(nc):
            target = rng.randint(2, 4)
            empty_slots = [slot for slot in range(4) if slot_grid[ri, ci, slot] == ANCHOR]
            rng.shuffle(empty_slots)
            while int(np.count_nonzero(slot_grid[ri, ci] != ANCHOR)) < target and empty_slots:
                slot = empty_slots.pop()
                slot_grid[ri, ci, slot] = rng.choice(SCATTER_COLORS)

    if not np.any(slot_grid == ANCHOR):
        ri = rng.randrange(nr)
        ci = rng.randrange(nc)
        slot = rng.randrange(4)
        slot_grid[ri, ci, slot] = ANCHOR

    connected = _connected_slots(slot_grid)
    all_colored = {
        (ri, ci, slot)
        for ri in range(nr)
        for ci in range(nc)
        for slot in range(4)
        if int(slot_grid[ri, ci, slot]) != ANCHOR
    }
    if connected == all_colored:
        for ri, ci, slot in all_colored:
            if int(np.count_nonzero(slot_grid[ri, ci] != ANCHOR)) > 2:
                slot_grid[ri, ci, slot] = rng.choice(SCATTER_COLORS)
                break

    return slot_grid


def _render_input(
    rng: random.Random,
    row_heights: list[int],
    col_widths: list[int],
    slot_grid: np.ndarray,
) -> np.ndarray:
    nr = len(row_heights)
    nc = len(col_widths)
    height = sum(row_heights) + (nr - 1)
    width = sum(col_widths) + (nc - 1)
    grid = np.full((height, width), BACKGROUND, dtype=int)

    row_starts = []
    col_starts = []
    cursor = 0
    for i, size in enumerate(row_heights):
        row_starts.append(cursor)
        cursor += size
        if i < nr - 1:
            grid[cursor, :] = DIVIDER
            cursor += 1

    cursor = 0
    for i, size in enumerate(col_widths):
        col_starts.append(cursor)
        cursor += size
        if i < nc - 1:
            grid[:, cursor] = DIVIDER
            cursor += 1

    for ri, h in enumerate(row_heights):
        anchor_r = (h - 2) // 2
        for ci, w in enumerate(col_widths):
            anchor_c = (w - 2) // 2
            rs = row_starts[ri]
            cs = col_starts[ci]
            ar = rs + anchor_r
            ac = cs + anchor_c
            grid[ar : ar + 2, ac : ac + 2] = ANCHOR

            choices = _quadrant_choices(h, w, anchor_r, anchor_c)
            for slot in range(4):
                color = int(slot_grid[ri, ci, slot])
                if color == ANCHOR:
                    continue
                rr, cc = rng.choice(choices[slot])
                grid[rs + rr, cs + cc] = color

    return grid


def _validate_example(inp: np.ndarray, out: np.ndarray) -> bool:
    parsed = _parse_grid(inp)
    aggregated = _paint_aggregated(parsed)
    expected = _expected_output(parsed)
    if not np.array_equal(out, expected):
        return False

    line_cells = np.argwhere((expected != aggregated) & (aggregated == BACKGROUND))
    if len(line_cells) == 0:
        return False

    slot_grid = np.array([region["slot_colors"] for region in parsed["regions"]], dtype=int)
    slot_grid = slot_grid.reshape(len(parsed["row_spans"]), len(parsed["col_spans"]), 4)
    connected = _connected_slots(slot_grid)
    colored = {
        (ri, ci, slot)
        for ri in range(slot_grid.shape[0])
        for ci in range(slot_grid.shape[1])
        for slot in range(4)
        if int(slot_grid[ri, ci, slot]) != ANCHOR
    }
    if not connected or connected == colored:
        return False
    if not np.any(slot_grid == ANCHOR):
        return False
    if not (_has_cross_slot_repeat(slot_grid) or _has_diagonal_duplicate(slot_grid)):
        return False
    return True


def generate_input() -> np.ndarray:
    rng = random.Random()
    for _ in range(500):
        nr = rng.choice((2, 3))
        nc = rng.choice((2, 3))
        row_heights = [rng.randint(5, 7) for _ in range(nr)]
        col_widths = [rng.randint(5, 7) for _ in range(nc)]
        slot_grid = _make_slot_grid(rng, nr, nc)
        inp = _render_input(rng, row_heights, col_widths, slot_grid)
        try:
            out = main(inp)
        except AssertionError:
            continue
        if _validate_example(inp, out):
            return inp
    raise RuntimeError("failed to generate a valid puzzle instance")
