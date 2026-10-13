from __future__ import annotations

import itertools
import random
from importlib.machinery import SourceFileLoader

import numpy as np

try:
    from arc_utils import Color, bbox_coords
except Exception:
    try:
        _arc_utils = SourceFileLoader(
            "_arc_utils_022_3", "/home/lixinyu/snap/new/solvers/arc_utils.py"
        ).load_module()
        Color = _arc_utils.Color
        bbox_coords = _arc_utils.bbox_coords
    except Exception:
        class Color:
            BLACK = 0
            RED = 2
            GREY = 5

        def bbox_coords(coords: np.ndarray) -> tuple[int, int, int, int]:
            r0, c0 = coords.min(axis=0)
            r1, c1 = coords.max(axis=0)
            return int(r0), int(c0), int(r1), int(c1)


BLACK = int(Color.BLACK)
RED = int(Color.RED)
GREY = int(Color.GREY)

BAR_LENGTHS = (2, 3, 4)
DIAMOND_COLORS = [c for c in range(1, 10) if c not in (RED, GREY)]

GEN_WIDTH_RANGE = (25, 30)
GEN_HEIGHT_RANGE = (27, 30)
GEN_LEGEND_HEIGHT_RANGE = (7, 8)
GEN_RADIUS_VALUES = tuple(range(3, 9))
GEN_MAX_FOOTPRINT = 9
GEN_ATTEMPTS = 500
GEN_BAR_TRIES = 120
GEN_PLACE_TRIES = 250

GEN_PATTERNS = [
    (RED, GREY, RED),
    (RED, GREY, GREY),
    (GREY, RED, RED),
    (GREY, RED, GREY),
    (GREY, GREY, RED),
]
GEN_PATTERN_WEIGHTS = [4, 1, 4, 1, 4]

_RADIUS_CACHE: dict[tuple[int, ...], list[tuple[int, ...]]] = {}


def _components(mask: np.ndarray, connectivity: int) -> list[np.ndarray]:
    mask = np.asarray(mask, dtype=bool)
    h, w = mask.shape
    seen = np.zeros_like(mask, dtype=bool)
    if int(connectivity) == 4:
        steps = [(-1, 0), (1, 0), (0, -1), (0, 1)]
    else:
        steps = [
            (-1, -1),
            (-1, 0),
            (-1, 1),
            (0, -1),
            (0, 1),
            (1, -1),
            (1, 0),
            (1, 1),
        ]

    comps: list[np.ndarray] = []
    for r, c in np.argwhere(mask):
        r = int(r)
        c = int(c)
        if seen[r, c]:
            continue
        seen[r, c] = True
        stack = [(r, c)]
        coords: list[tuple[int, int]] = []
        while stack:
            rr, cc = stack.pop()
            coords.append((rr, cc))
            for dr, dc in steps:
                nr = rr + dr
                nc = cc + dc
                if 0 <= nr < h and 0 <= nc < w and mask[nr, nc] and not seen[nr, nc]:
                    seen[nr, nc] = True
                    stack.append((nr, nc))
        comps.append(np.array(coords, dtype=int))
    return comps


def _diamond_outline_coords(cr: int, cc: int, radius: int) -> set[tuple[int, int]]:
    cr = int(cr)
    cc = int(cc)
    radius = int(radius)
    if radius == 0:
        return {(cr, cc)}
    coords: set[tuple[int, int]] = set()
    for dr in range(-radius, radius + 1):
        dc = radius - abs(dr)
        if dc == 0:
            coords.add((cr + dr, cc))
        else:
            coords.add((cr + dr, cc - dc))
            coords.add((cr + dr, cc + dc))
    return coords


def _diamond_fill_coords(cr: int, cc: int, radius: int) -> set[tuple[int, int]]:
    cr = int(cr)
    cc = int(cc)
    radius = int(radius)
    coords: set[tuple[int, int]] = set()
    for dr in range(-radius, radius + 1):
        span = radius - abs(dr)
        rr = cr + dr
        for dc in range(-span, span + 1):
            coords.add((rr, cc + dc))
    return coords


def _halo8(coords: set[tuple[int, int]], shape: tuple[int, int]) -> set[tuple[int, int]]:
    h, w = shape
    halo: set[tuple[int, int]] = set()
    for r, c in coords:
        for dr in (-1, 0, 1):
            for dc in (-1, 0, 1):
                nr = r + dr
                nc = c + dc
                if 0 <= nr < h and 0 <= nc < w:
                    halo.add((nr, nc))
    return halo


def _paint_coords(grid: np.ndarray, coords: set[tuple[int, int]], color: int) -> None:
    for r, c in coords:
        grid[int(r), int(c)] = int(color)


def _extract_bars(grid: np.ndarray) -> list[dict]:
    bars: list[dict] = []
    next_id = 0
    for color in (RED, GREY):
        for comp in _components(grid == color, connectivity=4):
            r0, c0, r1, c1 = bbox_coords(comp)
            assert r0 == r1, "legend bar must be horizontal"
            expected = {(r0, c) for c in range(c0, c1 + 1)}
            actual = {tuple(map(int, cell)) for cell in comp.tolist()}
            assert actual == expected, "legend bar must be a contiguous segment"
            bars.append(
                {
                    "id": next_id,
                    "color": int(color),
                    "row": int(r0),
                    "c0": int(c0),
                    "c1": int(c1),
                    "length": int(c1 - c0 + 1),
                }
            )
            next_id += 1
    lengths = [bar["length"] for bar in bars]
    assert len(lengths) == len(set(lengths)), "bar lengths must be unique"
    return bars


def _extract_diamonds(grid: np.ndarray) -> list[dict]:
    diamonds: list[dict] = []
    next_id = 0
    colors = sorted(int(c) for c in np.unique(grid) if int(c) not in (BLACK, RED, GREY))
    for color in colors:
        comps = _components(grid == color, connectivity=8)
        assert len(comps) == 1, "each diamond color must form one component"
        comp = comps[0]
        r0, c0, r1, c1 = bbox_coords(comp)
        assert r1 - r0 == c1 - c0, "diamond bbox must be square"
        assert (r1 - r0) % 2 == 0, "diamond bbox must have odd side length"
        radius = (r1 - r0) // 2
        cr = (r0 + r1) // 2
        cc = (c0 + c1) // 2
        expected = _diamond_outline_coords(cr, cc, radius)
        actual = {tuple(map(int, cell)) for cell in comp.tolist()}
        assert actual == expected, "component must be a full diamond outline"
        diamonds.append(
            {
                "id": next_id,
                "color": int(color),
                "center": (int(cr), int(cc)),
                "radius": int(radius),
            }
        )
        next_id += 1
    radii = [diamond["radius"] for diamond in diamonds]
    assert len(radii) == len(set(radii)), "diamond radii must be unique"
    return diamonds


def _assignment_from_orders(bars_order: list[dict], diamonds_order: list[dict]) -> dict[int, dict]:
    return {diamonds_order[i]["id"]: bars_order[i] for i in range(len(diamonds_order))}


def _assignment_from_perm(bars: list[dict], diamonds: list[dict], perm: tuple[int, ...]) -> dict[int, dict]:
    return {diamonds[i]["id"]: bars[perm[i]] for i in range(len(diamonds))}


def _render_from_assignment(
    shape: tuple[int, int], diamonds: list[dict], assignment: dict[int, dict], *, strict: bool = True
) -> np.ndarray | None:
    out = np.zeros(shape, dtype=int)
    h, w = shape
    for diamond in diamonds:
        bar = assignment[diamond["id"]]
        if int(bar["color"]) == GREY:
            radius = int(diamond["radius"]) + int(bar["length"])
        else:
            radius = int(diamond["radius"]) - int(bar["length"])
        if radius < 1:
            if strict:
                raise AssertionError("diamond radius must stay positive")
            return None
        coords = _diamond_outline_coords(*diamond["center"], radius)
        if any(not (0 <= r < h and 0 <= c < w) for r, c in coords):
            if strict:
                raise AssertionError("diamond must stay inside the canvas")
            return None
        _paint_coords(out, coords, diamond["color"])
    return out


def _bar_midpoint(bar: dict) -> tuple[float, float]:
    return float(bar["row"]), 0.5 * float(bar["c0"] + bar["c1"])


def _distance(bar: dict, diamond: dict) -> float:
    br, bc = _bar_midpoint(bar)
    dr, dc = diamond["center"]
    return abs(br - float(dr)) + abs(bc - float(dc))


def _nearest_permutations(bars: list[dict], diamonds: list[dict]) -> list[tuple[int, ...]]:
    best = None
    winners: list[tuple[int, ...]] = []
    for perm in itertools.permutations(range(len(bars))):
        total = sum(_distance(bars[perm[i]], diamonds[i]) for i in range(len(diamonds)))
        if best is None or total < best - 1e-9:
            best = total
            winners = [perm]
        elif abs(total - best) <= 1e-9:
            winners.append(perm)
    return winners


def _valid_radius_tuples(pattern: tuple[int, ...]) -> list[tuple[int, ...]]:
    cached = _RADIUS_CACHE.get(pattern)
    if cached is not None:
        return cached

    tuples: list[tuple[int, ...]] = []
    for radii in itertools.combinations(GEN_RADIUS_VALUES, len(BAR_LENGTHS)):
        finals = []
        ok = True
        for radius, length, color in zip(radii, BAR_LENGTHS, pattern):
            final_radius = radius + length if color == GREY else radius - length
            if final_radius < 1 or max(radius, final_radius) > GEN_MAX_FOOTPRINT:
                ok = False
                break
            finals.append(final_radius)
        if ok and len(set(finals)) == len(finals):
            tuples.append(tuple(int(v) for v in radii))

    _RADIUS_CACHE[pattern] = tuples
    return tuples


def _build_true_assignment(bars: list[dict], diamonds: list[dict]) -> dict[int, dict]:
    bars_by_len = sorted(bars, key=lambda bar: (bar["length"], bar["row"], bar["c0"]))
    diamonds_by_radius = sorted(diamonds, key=lambda diamond: (diamond["radius"], diamond["center"]))
    return _assignment_from_orders(bars_by_len, diamonds_by_radius)


def _reject_shortcuts(shape: tuple[int, int], bars: list[dict], diamonds: list[dict], truth: np.ndarray) -> bool:
    bars_by_len = sorted(bars, key=lambda bar: (bar["length"], bar["row"], bar["c0"]))
    bars_left = sorted(bars, key=lambda bar: (_bar_midpoint(bar)[1], bar["row"]))
    bars_top = sorted(bars, key=lambda bar: (bar["row"], bar["c0"]))
    diamonds_by_radius = sorted(diamonds, key=lambda diamond: (diamond["radius"], diamond["center"]))
    diamonds_left = sorted(diamonds, key=lambda diamond: (diamond["center"][1], diamond["center"][0]))
    diamonds_top = sorted(diamonds, key=lambda diamond: (diamond["center"][0], diamond["center"][1]))

    if [d["id"] for d in diamonds_by_radius] == [d["id"] for d in diamonds_left]:
        return False
    if [d["id"] for d in diamonds_by_radius] == [d["id"] for d in diamonds_top]:
        return False
    if [b["id"] for b in bars_by_len] == [b["id"] for b in bars_left]:
        return False
    if [b["id"] for b in bars_by_len] == [b["id"] for b in bars_top]:
        return False

    wrong_assignments = [
        _assignment_from_orders(bars_by_len, diamonds_left),
        _assignment_from_orders(bars_by_len, diamonds_top),
        _assignment_from_orders(bars_left, diamonds_left),
        _assignment_from_orders(bars_top, diamonds_top),
    ]
    for assignment in wrong_assignments:
        wrong = _render_from_assignment(shape, diamonds, assignment, strict=False)
        if wrong is not None and np.array_equal(wrong, truth):
            return False

    for perm in _nearest_permutations(bars, diamonds):
        wrong = _render_from_assignment(
            shape, diamonds, _assignment_from_perm(bars, diamonds, perm), strict=False
        )
        if wrong is not None and np.array_equal(wrong, truth):
            return False
    return True


def main(grid: np.ndarray) -> np.ndarray:
    """
    Pair legend bars with canvas diamonds by rank and redraw each diamond at its new radius.

    Input structure:
    - The grid has a black background, several colored one-cell-thick Manhattan diamond outlines, and a bottom legend zone with horizontal bars.
    - Diamond colors are all distinct and never red or grey; legend bars use only red and grey, and their lengths are all distinct.
    - Diamonds do not touch each other, and the bars need not align with any diamond by row, column, or proximity.

    Rule:
    - Rank the legend bars from shortest to longest, and rank the diamonds from smallest radius to largest.
    - Pair equal ranks: a grey bar expands its matched diamond by the bar length, while a red bar shrinks it by the bar length.
    - Erase the entire legend and redraw all diamonds on the same black canvas with unchanged centers and colors.

    Key insight:
    - The only valid matching signal is internal size rank on each side: bar length first chooses which diamond it controls, then the same numeric length gives the expansion or contraction amount.
    """
    g = np.asarray(grid, dtype=int)
    assert g.ndim == 2

    bars = _extract_bars(g)
    diamonds = _extract_diamonds(g)
    assert len(bars) == len(diamonds) and len(bars) >= 2

    assignment = _build_true_assignment(bars, diamonds)
    return _render_from_assignment(g.shape, diamonds, assignment)


def _place_diamonds(
    canvas_h: int, width: int, specs: list[dict], rng: random.Random
) -> list[dict] | None:
    occupied_halo: set[tuple[int, int]] = set()
    shape = (canvas_h, width)
    placed = [dict(spec) for spec in specs]
    order = sorted(range(len(specs)), key=lambda i: (-specs[i]["footprint"], specs[i]["radius"]))

    for idx in order:
        footprint = int(specs[idx]["footprint"])
        row_low = footprint
        row_high = canvas_h - 1 - footprint
        col_low = footprint
        col_high = width - 1 - footprint
        if row_low > row_high or col_low > col_high:
            return None

        found = False
        for _ in range(GEN_PLACE_TRIES):
            cr = rng.randint(row_low, row_high)
            cc = rng.randint(col_low, col_high)
            fill = _diamond_fill_coords(cr, cc, footprint)
            if fill & occupied_halo:
                continue
            placed[idx]["center"] = (cr, cc)
            placed[idx]["input_coords"] = _diamond_outline_coords(cr, cc, specs[idx]["radius"])
            occupied_halo |= _halo8(fill, shape)
            found = True
            break
        if not found:
            return None
    return placed


def _bar_row_options(legend_top: int, height: int) -> list[tuple[int, int, int]]:
    rows = list(range(legend_top + 1, height))
    options = []
    for combo in itertools.combinations(rows, 3):
        if combo[1] - combo[0] >= 2 and combo[2] - combo[1] >= 2:
            options.append(tuple(int(v) for v in combo))
    return options


def _render_input(shape: tuple[int, int], diamonds: list[dict], bars: list[dict]) -> np.ndarray:
    grid = np.zeros(shape, dtype=int)
    for diamond in diamonds:
        _paint_coords(grid, diamond["input_coords"], diamond["color"])
    for bar in bars:
        grid[int(bar["row"]), int(bar["c0"]) : int(bar["c1"]) + 1] = int(bar["color"])
    return grid


def _try_place_bars(
    shape: tuple[int, int], legend_top: int, diamonds: list[dict], rng: random.Random
) -> np.ndarray | None:
    height, width = shape
    row_options = _bar_row_options(legend_top, height)
    if not row_options:
        return None

    for _ in range(GEN_BAR_TRIES):
        rows = list(rng.choice(row_options))
        rng.shuffle(rows)
        rank_order = list(range(len(BAR_LENGTHS)))
        rng.shuffle(rank_order)
        bars = []
        for row, rank in zip(rows, rank_order):
            length = BAR_LENGTHS[rank]
            c0 = rng.randint(0, width - length)
            bars.append(
                {
                    "id": len(bars),
                    "color": int(diamonds[rank]["bar_color"]),
                    "row": int(row),
                    "c0": int(c0),
                    "c1": int(c0 + length - 1),
                    "length": int(length),
                }
            )

        grid = _render_input(shape, diamonds, bars)
        try:
            parsed_bars = _extract_bars(grid)
            parsed_diamonds = _extract_diamonds(grid)
            truth = _render_from_assignment(shape, parsed_diamonds, _build_true_assignment(parsed_bars, parsed_diamonds))
            if not np.array_equal(main(grid), truth):
                continue
            if not _reject_shortcuts(shape, parsed_bars, parsed_diamonds, truth):
                continue
        except AssertionError:
            continue
        return grid
    return None


def _generate_case(rng: random.Random) -> np.ndarray | None:
    pattern = rng.choices(GEN_PATTERNS, weights=GEN_PATTERN_WEIGHTS, k=1)[0]
    radius_choices = _valid_radius_tuples(pattern)
    if not radius_choices:
        return None
    radii = list(rng.choice(radius_choices))
    colors = rng.sample(DIAMOND_COLORS, len(BAR_LENGTHS))

    specs = []
    for color, radius, length, bar_color in zip(colors, radii, BAR_LENGTHS, pattern):
        final_radius = radius + length if bar_color == GREY else radius - length
        specs.append(
            {
                "color": int(color),
                "radius": int(radius),
                "bar_color": int(bar_color),
                "final_radius": int(final_radius),
                "footprint": int(max(radius, final_radius)),
            }
        )

    legend_h = rng.randint(*GEN_LEGEND_HEIGHT_RANGE)
    min_height = max(GEN_HEIGHT_RANGE[0], 2 * max(spec["footprint"] for spec in specs) + 1 + legend_h)
    if min_height > GEN_HEIGHT_RANGE[1]:
        return None
    height = rng.randint(min_height, GEN_HEIGHT_RANGE[1])
    width = rng.randint(*GEN_WIDTH_RANGE)
    canvas_h = height - legend_h
    legend_top = canvas_h

    diamonds = _place_diamonds(canvas_h, width, specs, rng)
    if diamonds is None:
        return None
    return _try_place_bars((height, width), legend_top, diamonds, rng)


def generate_input() -> np.ndarray:
    rng = random.Random()
    for _ in range(GEN_ATTEMPTS):
        grid = _generate_case(rng)
        if grid is not None:
            return grid
    raise RuntimeError(f"failed to generate valid input after {GEN_ATTEMPTS} attempts")
