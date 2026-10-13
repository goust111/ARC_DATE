from __future__ import annotations

from collections import Counter, deque
from pathlib import Path
import random
import sys

import numpy as np


_SOLVER_DIR = Path("/home/lixinyu/snap/new/solvers")
if _SOLVER_DIR.exists() and str(_SOLVER_DIR) not in sys.path:
    sys.path.append(str(_SOLVER_DIR))

try:
    from arc_utils import bbox_coords as _arc_bbox_coords
except Exception:  # pragma: no cover - keep direct SourceFileLoader use robust.
    _arc_bbox_coords = None


BG = 0
PALETTE = list(range(1, 10))
Coord = tuple[int, int]
BBox = tuple[int, int, int, int]


def _bbox_coords(coords: np.ndarray) -> BBox:
    if _arc_bbox_coords is not None:
        return _arc_bbox_coords(coords)
    r0, c0 = coords.min(axis=0)
    r1, c1 = coords.max(axis=0)
    return int(r0), int(c0), int(r1), int(c1)


def _most_common(grid: np.ndarray) -> int:
    vals, counts = np.unique(grid, return_counts=True)
    return int(vals[int(np.argmax(counts))])


def _neighbors4(r: int, c: int, shape: tuple[int, int]) -> list[Coord]:
    h, w = shape
    out: list[Coord] = []
    for dr, dc in ((-1, 0), (1, 0), (0, -1), (0, 1)):
        rr, cc = int(r + dr), int(c + dc)
        if 0 <= rr < h and 0 <= cc < w:
            out.append((rr, cc))
    return out


def _components(mask: np.ndarray) -> list[np.ndarray]:
    seen = np.zeros(mask.shape, dtype=bool)
    comps: list[np.ndarray] = []
    for sr0, sc0 in np.argwhere(mask):
        sr, sc = int(sr0), int(sc0)
        if seen[sr, sc]:
            continue
        seen[sr, sc] = True
        q: deque[Coord] = deque([(sr, sc)])
        pts: list[Coord] = []
        while q:
            r, c = q.popleft()
            pts.append((r, c))
            for nr, nc in _neighbors4(r, c, mask.shape):
                if bool(mask[nr, nc]) and not seen[nr, nc]:
                    seen[nr, nc] = True
                    q.append((nr, nc))
        comps.append(np.array(pts, dtype=int))
    return comps


def _bottom_left_legend(g: np.ndarray, bg: int) -> list[int]:
    h, w = g.shape
    legend: list[int] = []
    c = 0
    while c < w and int(g[h - 1, c]) != bg:
        legend.append(int(g[h - 1, c]))
        c += 1
    assert len(legend) >= 2, "expected a bottom-left horizontal legend"
    assert len(set(legend)) == len(legend), "legend colors must be unique"
    assert np.all(g[h - 1, c:] == bg), "only the left end of the bottom row may hold the legend"
    return legend


def _wall_cells_for_box(box: BBox) -> set[Coord]:
    r0, c0, r1, c1 = box
    cells: set[Coord] = set()
    for r in range(r0, r1 + 1):
        cells.add((r, c0))
        cells.add((r, c1))
    for c in range(c0, c1 + 1):
        cells.add((r1, c))
    return cells


def _tube_from_component(coords: np.ndarray) -> dict | None:
    r0, c0, r1, c1 = _bbox_coords(coords)
    h = r1 - r0 + 1
    w = c1 - c0 + 1
    if h < 4 or w < 3:
        return None

    box = (int(r0), int(c0), int(r1), int(c1))
    actual = {(int(r), int(c)) for r, c in coords}
    expected = _wall_cells_for_box(box)
    if actual != expected:
        return None

    return {
        "bbox": box,
        "inner": (int(r0), int(c0 + 1), int(r1 - 1), int(c1 - 1)),
        "wall_cells": expected,
    }


def _find_tubes(g: np.ndarray, bg: int, legend: list[int]) -> tuple[int, list[dict]]:
    foreground = sorted(set(map(int, g.ravel())) - {int(bg)} - set(map(int, legend)))
    candidates: dict[int, list[dict]] = {}
    for color in foreground:
        comps = _components(g == int(color))
        tubes: list[dict] = []
        ok = bool(comps)
        for comp in comps:
            tube = _tube_from_component(comp)
            if tube is None:
                ok = False
                break
            tubes.append(tube)
        if ok:
            candidates[int(color)] = sorted(tubes, key=lambda item: item["bbox"])

    assert len(candidates) == 1, "expected one wall color whose components are U-shaped tubes"
    wall_color, tubes = next(iter(candidates.items()))
    assert tubes, "expected at least one tube"
    return int(wall_color), tubes


def _interior_cells(tube: dict) -> list[Coord]:
    r0, c0, r1, c1 = tube["inner"]
    return [(r, c) for r in range(r0, r1 + 1) for c in range(c0, c1 + 1)]


def _fill_order(tube: dict) -> list[Coord]:
    r0, c0, r1, c1 = tube["inner"]
    return [(r, c) for r in range(r1, r0 - 1, -1) for c in range(c0, c1 + 1)]


def _parse_scene(g: np.ndarray) -> tuple[int, list[int], int, int, list[dict]]:
    bg = _most_common(g)
    legend = _bottom_left_legend(g, bg)
    legend_set = set(legend)
    wall_color, tubes = _find_tubes(g, bg, legend)

    legend_mask = np.zeros(g.shape, dtype=bool)
    legend_mask[g.shape[0] - 1, : len(legend)] = True
    wall_mask = np.zeros(g.shape, dtype=bool)
    interior_mask = np.zeros(g.shape, dtype=bool)

    for tube in tubes:
        for r, c in tube["wall_cells"]:
            assert not wall_mask[r, c], "tube walls must not overlap"
            wall_mask[r, c] = True
        r0, c0, r1, c1 = tube["inner"]
        assert r0 <= r1 and c0 <= c1
        assert not np.any(interior_mask[r0 : r1 + 1, c0 : c1 + 1]), "tube interiors must not overlap"
        interior_mask[r0 : r1 + 1, c0 : c1 + 1] = True

    outside = ~(legend_mask | wall_mask | interior_mask)
    assert np.all(g[outside] == bg), "non-background cells must be legend, tube walls, or tube contents"
    assert np.all(g[wall_mask] == wall_color)

    contents = set(map(int, g[interior_mask])) - {int(bg)}
    assert contents, "expected colored droplets inside the tubes"
    extra = contents - legend_set
    assert len(extra) == 1, "tube contents must contain exactly one color absent from the legend"
    bubble = int(next(iter(extra)))
    assert bubble != wall_color
    assert wall_color not in contents

    allowed = legend_set | {bubble}
    for tube in tubes:
        vals = [int(g[r, c]) for r, c in _interior_cells(tube) if int(g[r, c]) != bg]
        assert vals, "each tube must contain droplets"
        assert set(vals).issubset(allowed), "tube contents must use legend colors plus the bubble color"
        assert bubble in vals, "the absent legend color must appear as a bubble in every tube"
        inner_w = tube["inner"][3] - tube["inner"][1] + 1
        counts = Counter(vals)
        assert all(count % inner_w == 0 for count in counts.values()), "layers must fill whole tube rows"

    return int(bg), legend, int(wall_color), int(bubble), tubes


def main(grid: np.ndarray) -> np.ndarray:
    """
    Layer each U-tube's droplets by the bottom-row density legend and float the unlisted bubble color.

    Input structure:
    - A background grid contains several separated one-cell-thick U-shaped vertical tubes in one wall color.
    - Each tube's inner shaft contains scattered colored droplets and background gaps.
    - The bottom row starts with a contiguous color legend; left-to-right gives heaviest to lightest liquid.
    - Exactly one color appearing inside tubes is absent from the legend, and that color is the bubble color.

    Rule:
    - For each tube, count all non-background cells in its inner shaft and preserve those counts.
    - Clear the shaft, then stack complete rows from the tube bottom upward in legend order from heavy to light.
    - Stack the unlisted bubble color directly above all legend liquids; cells above the bubble layer remain background.
    - Preserve the legend, tube walls, and every background cell outside tube interiors.

    Key insight:
    - The legend is a density table, not a palette: tube contents ignore their input order, while the only unlisted color floats above all listed liquids.
    """
    g = np.asarray(grid, dtype=int)
    assert g.ndim == 2
    assert max(g.shape) <= 30

    bg, legend, _wall_color, bubble, tubes = _parse_scene(g)
    out = g.copy()

    for tube in tubes:
        r0, c0, r1, c1 = tube["inner"]
        vals = [int(g[r, c]) for r, c in _interior_cells(tube) if int(g[r, c]) != bg]
        counts = Counter(vals)
        capacity = (r1 - r0 + 1) * (c1 - c0 + 1)
        assert sum(counts.values()) <= capacity

        out[r0 : r1 + 1, c0 : c1 + 1] = bg
        order = _fill_order(tube)
        k = 0
        for color in legend:
            for _ in range(counts.get(int(color), 0)):
                rr, cc = order[k]
                out[rr, cc] = int(color)
                k += 1
        for _ in range(counts.get(int(bubble), 0)):
            rr, cc = order[k]
            out[rr, cc] = int(bubble)
            k += 1

    return out


# ----------------------------- generator -----------------------------

_MAX_ATTEMPTS = 800


def _boxes_overlap_with_gap(a: BBox, b: BBox, gap: int = 1) -> bool:
    ar0, ac0, ar1, ac1 = a
    br0, bc0, br1, bc1 = b
    return not (
        ar1 + gap < br0
        or br1 + gap < ar0
        or ac1 + gap < bc0
        or bc1 + gap < ac0
    )


def _choose_colors(legend_len: int) -> tuple[int, int, list[int]] | None:
    for _ in range(100):
        colors = PALETTE[:]
        random.shuffle(colors)
        wall = colors.pop()
        contents = random.sample(colors, legend_len + 1)
        bubble = random.choice(contents)
        legend = [color for color in contents if color != bubble]
        random.shuffle(legend)

        all_contents = set(legend) | {bubble}
        if bubble in (min(all_contents), max(all_contents)):
            continue
        if legend == sorted(legend) or legend == sorted(legend, reverse=True):
            continue
        return int(wall), int(bubble), [int(color) for color in legend]
    return None


def _place_tubes(shape: tuple[int, int], count: int) -> list[dict] | None:
    h, w = shape
    tubes: list[dict] = []
    for _ in range(count):
        placed = False
        for _attempt in range(250):
            inner_w = random.choice([1, 1, 2, 2, 3])
            max_inner_h = min(10, h - 4)
            if max_inner_h < 5:
                return None
            inner_h = random.randint(5, max_inner_h)
            outer_h = inner_h + 1
            outer_w = inner_w + 2
            if outer_w > w:
                continue

            max_r0 = h - outer_h - 2
            max_c0 = w - outer_w
            if max_r0 < 0 or max_c0 < 0:
                continue
            r0 = random.randint(0, max_r0)
            c0 = random.randint(0, max_c0)
            box = (r0, c0, r0 + outer_h - 1, c0 + outer_w - 1)
            if any(_boxes_overlap_with_gap(box, other["bbox"], gap=1) for other in tubes):
                continue

            tubes.append(
                {
                    "bbox": box,
                    "inner": (r0, c0 + 1, r0 + inner_h - 1, c0 + inner_w),
                    "wall_cells": _wall_cells_for_box(box),
                }
            )
            placed = True
            break
        if not placed:
            return None
    return sorted(tubes, key=lambda item: item["bbox"])


def _paint_walls(grid: np.ndarray, tubes: list[dict], wall: int) -> None:
    for tube in tubes:
        for r, c in tube["wall_cells"]:
            grid[r, c] = int(wall)


def _choose_counts(
    tube: dict,
    legend: list[int],
    bubble: int,
    required: set[int],
) -> dict[int, int] | None:
    r0, c0, r1, c1 = tube["inner"]
    inner_h = r1 - r0 + 1
    inner_w = c1 - c0 + 1
    max_selected = min(len(legend), inner_h - 2)
    if max_selected < 2 or len(required) > max_selected:
        return None

    selected_count = random.randint(max(2, len(required)), max_selected)
    selected = set(required)
    options = [color for color in legend if color not in selected]
    random.shuffle(options)
    while len(selected) < selected_count and options:
        selected.add(options.pop())

    row_counts = {int(color): 1 for color in selected}
    row_counts[int(bubble)] = 1
    extra_room = inner_h - 1 - sum(row_counts.values())
    if extra_room < 0:
        return None
    extra = random.randint(0, extra_room)
    choices = list(row_counts.keys())
    for _ in range(extra):
        row_counts[random.choice(choices)] += 1

    return {int(color): int(rows * inner_w) for color, rows in row_counts.items()}


def _scatter_contents(grid: np.ndarray, tube: dict, counts: dict[int, int]) -> None:
    cells = _interior_cells(tube)
    colors: list[int] = []
    for color, count in counts.items():
        colors.extend([int(color)] * int(count))
    random.shuffle(colors)
    chosen = random.sample(cells, len(colors))
    for (r, c), color in zip(chosen, colors):
        grid[r, c] = int(color)


def _known_expected(inp: np.ndarray, tubes: list[dict], legend: list[int], bubble: int, bg: int = BG) -> np.ndarray:
    out = inp.copy()
    for tube in tubes:
        r0, c0, r1, c1 = tube["inner"]
        vals = [int(inp[r, c]) for r, c in _interior_cells(tube) if int(inp[r, c]) != bg]
        counts = Counter(vals)
        out[r0 : r1 + 1, c0 : c1 + 1] = bg
        order = _fill_order(tube)
        k = 0
        for color in legend:
            for _ in range(counts.get(int(color), 0)):
                rr, cc = order[k]
                out[rr, cc] = int(color)
                k += 1
        for _ in range(counts.get(int(bubble), 0)):
            rr, cc = order[k]
            out[rr, cc] = int(bubble)
            k += 1
    return out


def _semantic_checks(
    inp: np.ndarray,
    expected: np.ndarray,
    tubes: list[dict],
    legend: list[int],
    bubble: int,
    wall: int,
) -> bool:
    if inp.ndim != 2 or expected.ndim != 2 or max(inp.shape) > 30 or max(expected.shape) > 30:
        return False
    if _most_common(inp) != BG:
        return False
    if wall in legend or wall == bubble or bubble in legend:
        return False
    if legend == sorted(legend) or legend == sorted(legend, reverse=True):
        return False
    all_contents = set(legend) | {bubble}
    if bubble in (min(all_contents), max(all_contents)):
        return False
    if np.array_equal(inp, expected):
        return False
    if not any(tube["inner"][3] - tube["inner"][1] + 1 > 1 for tube in tubes):
        return False

    global_counts: Counter[int] = Counter()
    at_least_one_missing_legend_color = False
    for tube in tubes:
        r0, c0, r1, c1 = tube["inner"]
        input_part = inp[r0 : r1 + 1, c0 : c1 + 1]
        output_part = expected[r0 : r1 + 1, c0 : c1 + 1]
        if np.array_equal(input_part, output_part):
            return False

        vals = [int(v) for v in input_part.ravel() if int(v) != BG]
        counts = Counter(vals)
        if not vals or counts.get(int(bubble), 0) == 0:
            return False
        if not set(counts).issubset(all_contents):
            return False

        inner_w = c1 - c0 + 1
        if any(count % inner_w != 0 for count in counts.values()):
            return False
        if any(counts.get(color, 0) == 0 for color in legend):
            at_least_one_missing_legend_color = True
        global_counts.update(counts)

        seq = [int(expected[r, c]) for r, c in _fill_order(tube)]
        target_seq: list[int] = []
        for color in legend:
            target_seq.extend([int(color)] * counts.get(int(color), 0))
        target_seq.extend([int(bubble)] * counts.get(int(bubble), 0))
        target_seq.extend([BG] * (len(seq) - len(target_seq)))
        if seq != target_seq:
            return False

    if any(global_counts.get(int(color), 0) == 0 for color in legend):
        return False
    if len(legend) >= 3 and not at_least_one_missing_legend_color:
        return False
    return True


def _gen_attempt() -> np.ndarray | None:
    rows = random.randint(15, 27)
    cols = random.randint(16, 30)
    legend_len = random.choice([3, 4])
    color_spec = _choose_colors(legend_len)
    if color_spec is None:
        return None
    wall, bubble, legend = color_spec

    max_count = max(2, min(5, (cols - 1) // 4))
    tube_count = random.randint(2, max_count)
    tubes = _place_tubes((rows, cols), tube_count)
    if tubes is None:
        return None

    grid = np.full((rows, cols), BG, dtype=int)
    grid[rows - 1, : len(legend)] = np.array(legend, dtype=int)
    _paint_walls(grid, tubes, wall)

    forced_by_tube = [set() for _ in tubes]
    shuffled_legend = legend[:]
    random.shuffle(shuffled_legend)
    for idx, color in enumerate(shuffled_legend):
        forced_by_tube[idx % len(tubes)].add(int(color))

    for tube, required in zip(tubes, forced_by_tube):
        counts = _choose_counts(tube, legend, bubble, required)
        if counts is None:
            return None
        _scatter_contents(grid, tube, counts)

    expected = _known_expected(grid, tubes, legend, bubble)
    if not _semantic_checks(grid, expected, tubes, legend, bubble, wall):
        return None
    try:
        actual = main(grid)
    except Exception:
        return None
    if not np.array_equal(actual, expected):
        return None
    return grid


def generate_input() -> np.ndarray:
    for _ in range(_MAX_ATTEMPTS):
        grid = _gen_attempt()
        if grid is not None:
            return grid
    raise RuntimeError("generate_input: failed to build a valid tube legend puzzle")
