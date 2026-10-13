from __future__ import annotations

import itertools
import random
from collections import Counter, deque

import numpy as np


BG = 0
RED = 2
GREY = 5
BODY_COLORS = [1, 3, 4, 6, 7, 8, 9]
MAX_GENERATION_ATTEMPTS = 500

Coord = tuple[int, int]
Rect = tuple[int, int, int, int]


def _bbox(coords: np.ndarray) -> Rect:
    r0, c0 = coords.min(axis=0)
    r1, c1 = coords.max(axis=0)
    return int(r0), int(c0), int(r1), int(c1)


def _neighbors(r: int, c: int, connectivity: int):
    if connectivity == 4:
        steps = ((-1, 0), (1, 0), (0, -1), (0, 1))
    else:
        steps = tuple(
            (dr, dc)
            for dr in (-1, 0, 1)
            for dc in (-1, 0, 1)
            if dr != 0 or dc != 0
        )
    for dr, dc in steps:
        yield r + dr, c + dc


def _components(mask: np.ndarray, connectivity: int = 4) -> list[np.ndarray]:
    h, w = mask.shape
    seen = np.zeros_like(mask, dtype=bool)
    comps: list[np.ndarray] = []
    for sr, sc in np.argwhere(mask):
        sr, sc = int(sr), int(sc)
        if seen[sr, sc]:
            continue
        q: deque[Coord] = deque([(sr, sc)])
        seen[sr, sc] = True
        cells: list[Coord] = []
        while q:
            r, c = q.popleft()
            cells.append((r, c))
            for rr, cc in _neighbors(r, c, connectivity):
                if 0 <= rr < h and 0 <= cc < w and mask[rr, cc] and not seen[rr, cc]:
                    seen[rr, cc] = True
                    q.append((rr, cc))
        comps.append(np.array(cells, dtype=int))
    return comps


def _in_rect(pt: Coord, rect: Rect) -> bool:
    r, c = pt
    r0, c0, r1, c1 = rect
    return r0 <= r <= r1 and c0 <= c <= c1


def _rect_has_area(rect: Rect) -> bool:
    r0, c0, r1, c1 = rect
    return r0 <= r1 and c0 <= c1


def _split_by_separator(g: np.ndarray) -> tuple[Rect, Rect]:
    h, w = g.shape
    full_rows = [r for r in range(h) if np.all(g[r, :] == GREY)]
    full_cols = [c for c in range(w) if np.all(g[:, c] == GREY)]
    assert len(full_rows) + len(full_cols) == 1, "expected exactly one full grey separator line"

    if full_rows:
        r = full_rows[0]
        a = (0, 0, r - 1, w - 1)
        b = (r + 1, 0, h - 1, w - 1)
    else:
        c = full_cols[0]
        a = (0, 0, h - 1, c - 1)
        b = (0, c + 1, h - 1, w - 1)
    assert _rect_has_area(a) and _rect_has_area(b), "separator must leave two non-empty regions"
    return a, b


def _subgrid(g: np.ndarray, rect: Rect) -> np.ndarray:
    r0, c0, r1, c1 = rect
    return g[r0 : r1 + 1, c0 : c1 + 1]


def _parse_legend(g: np.ndarray, rect: Rect) -> list[dict]:
    sub = _subgrid(g, rect)
    assert not np.any(sub == GREY), "grey must appear only on the separator"
    comps = _components(sub != BG, connectivity=4)
    assert len(comps) >= 2, "expected several legend sprites"

    sprites: list[dict] = []
    r_base, c_base, _r1, _c1 = rect
    for comp in comps:
        vals = [int(sub[r, c]) for r, c in comp]
        reds = [(int(r), int(c)) for r, c in comp if int(sub[r, c]) == RED]
        colors = sorted(set(vals) - {RED})
        assert len(reds) == 1 and len(colors) == 1, "each legend sprite has one red anchor and one body color"
        assert len(comp) >= 5, "legend sprites must be larger than work traces"

        ar, ac = reds[0]
        body_color = int(colors[0])
        rel = frozenset((int(r - ar), int(c - ac)) for r, c in comp)
        body_rel = frozenset(pt for pt in rel if pt != (0, 0))
        assert (0, 0) in rel and body_rel, "anchor must be a covered sprite cell"
        assert any(
            (ar + dr, ac + dc) in {(int(r), int(c)) for r, c in comp}
            for dr, dc in ((-1, 0), (1, 0), (0, -1), (0, 1))
        ), "anchor must be embedded in the sprite"

        sprites.append(
            {
                "color": body_color,
                "rel": rel,
                "body_rel": body_rel,
                "anchor": (r_base + ar, c_base + ac),
                "bbox": _bbox(comp),
            }
        )

    color_counts = Counter(sprite["color"] for sprite in sprites)
    assert max(color_counts.values()) >= 2, "at least one legend color must have multiple shapes"
    return sprites


def _parse_work(g: np.ndarray, rect: Rect) -> list[dict]:
    sub = _subgrid(g, rect)
    assert not np.any(sub == GREY), "grey must appear only on the separator"
    comps = _components(sub != BG, connectivity=8)
    assert len(comps) >= 2, "expected several anchored traces in the work area"

    traces: list[dict] = []
    r_base, c_base, _r1, _c1 = rect
    for comp in comps:
        vals = [int(sub[r, c]) for r, c in comp]
        reds = [(int(r), int(c)) for r, c in comp if int(sub[r, c]) == RED]
        colors = sorted(set(vals) - {RED})
        assert len(reds) == 1 and len(colors) == 1, "each work clue has one red point and one trace color"
        assert 1 <= len(comp) - 1 <= 2, "each work clue must have one or two trace cells"

        ar, ac = reds[0]
        color = int(colors[0])
        coords = frozenset((r_base + int(r), c_base + int(c)) for r, c in comp)
        trace_cells = [
            (int(r), int(c))
            for r, c in comp
            if int(sub[r, c]) != RED
        ]
        offsets = frozenset((int(r - ar), int(c - ac)) for r, c in trace_cells)
        assert (0, 0) not in offsets
        traces.append(
            {
                "color": color,
                "anchor": (r_base + ar, c_base + ac),
                "offsets": offsets,
                "coords": coords,
            }
        )
    return traces


def _candidate_cells(anchor: Coord, rel: frozenset[Coord]) -> frozenset[Coord]:
    ar, ac = anchor
    return frozenset((ar + dr, ac + dc) for dr, dc in rel)


def _solve_as(g: np.ndarray, legend_rect: Rect, work_rect: Rect) -> np.ndarray:
    sprites = _parse_legend(g, legend_rect)
    traces = _parse_work(g, work_rect)
    used_colors = Counter(trace["color"] for trace in traces)
    legend_counts = Counter(sprite["color"] for sprite in sprites)
    assert any(legend_counts[color] >= 2 for color in used_colors), "some work clues must need trace disambiguation"

    placements: list[tuple[int, frozenset[Coord]]] = []
    for trace in traces:
        matches: list[tuple[int, frozenset[Coord]]] = []
        for sprite in sprites:
            if int(sprite["color"]) != int(trace["color"]):
                continue
            if not trace["offsets"].issubset(sprite["body_rel"]):
                continue

            cells = _candidate_cells(trace["anchor"], sprite["rel"])
            if not all(_in_rect(cell, work_rect) for cell in cells):
                continue

            ok = True
            for cell in cells:
                v = int(g[cell])
                if cell in trace["coords"]:
                    continue
                if v != BG:
                    ok = False
                    break
            if ok:
                matches.append((int(sprite["color"]), cells))

        assert len(matches) == 1, "each red point must have one trace-consistent legend sprite"
        placements.append(matches[0])

    painted: dict[Coord, int] = {}
    for color, cells in placements:
        for cell in cells:
            previous = painted.get(cell)
            assert previous is None, "completed work sprites must not overlap"
            painted[cell] = int(color)

    out = g.copy()
    for (r, c), color in painted.items():
        out[r, c] = int(color)
    return out


def main(grid: np.ndarray) -> np.ndarray:
    """
    Complete each partial trace by aligning the matching legend sprite's red anchor to the work red point.

    Input structure:
    - A black grid is split by one full grey row or column into a legend area and a work area.
    - The legend contains several monochrome sprites; each sprite has exactly one of its cells covered by a red anchor, and at least two same-colored sprites have different shapes.
    - The work area contains several isolated red anchor points, each joined to one or two nearby cells of a sprite's body color.

    Rule:
    - For each work red point, translate every legend sprite of the trace color so that the sprite's red anchor coincides with that point.
    - Keep the unique translated sprite whose non-anchor body includes all visible trace offsets and fits in the work area without covering another clue.
    - Paint the entire selected sprite in its body color, including the anchor cell, while leaving the legend and grey separator unchanged.

    Key insight:
    - The red cell is an alignment marker rather than paint; the partial body-color trace disambiguates same-colored legend shapes after the anchors are made coincident.
    """
    g = np.asarray(grid, dtype=int)
    assert g.ndim == 2 and g.size > 0
    assert max(g.shape) <= 30

    side_a, side_b = _split_by_separator(g)
    solved: list[np.ndarray] = []
    for legend_rect, work_rect in ((side_a, side_b), (side_b, side_a)):
        try:
            solved.append(_solve_as(g, legend_rect, work_rect))
        except AssertionError:
            continue
    assert len(solved) == 1, "legend/work assignment must be unique"
    return solved[0]


# --------------- generator ---------------

BASE_SHAPES: tuple[frozenset[Coord], ...] = (
    frozenset({(0, 0), (-1, 0), (1, 0), (0, -1), (0, 1), (-1, -1)}),
    frozenset({(0, 0), (-1, 0), (1, 0), (0, -1), (0, 1), (1, 1)}),
    frozenset({(0, 0), (-1, 0), (-2, 0), (0, 1), (1, 1), (1, 2)}),
    frozenset({(0, 0), (1, 0), (2, 0), (0, -1), (-1, -1), (-1, -2)}),
    frozenset({(0, 0), (-1, 0), (1, 0), (-1, 1), (0, 1), (1, 1), (1, 2)}),
    frozenset({(0, 0), (-1, 0), (1, 0), (-1, -1), (0, -1), (1, -1), (-1, -2)}),
    frozenset({(0, 0), (0, -1), (0, 1), (-1, 1), (-2, 1), (1, -1)}),
    frozenset({(0, 0), (-1, 0), (1, 0), (1, -1), (1, -2), (-1, 1)}),
)


def _rot(pt: Coord, k: int) -> Coord:
    r, c = pt
    for _ in range(k % 4):
        r, c = c, -r
    return int(r), int(c)


def _transform_shape(shape: frozenset[Coord]) -> frozenset[Coord]:
    k = random.randrange(4)
    flip = random.random() < 0.5
    out: set[Coord] = set()
    for r, c in shape:
        if flip:
            c = -c
        out.add(_rot((r, c), k))
    return frozenset(out)


def _shape_connected(shape: frozenset[Coord], connectivity: int = 4) -> bool:
    mask_coords = set(shape)
    start = next(iter(mask_coords))
    q: deque[Coord] = deque([start])
    seen = {start}
    while q:
        r, c = q.popleft()
        for rr, cc in _neighbors(r, c, connectivity):
            if (rr, cc) in mask_coords and (rr, cc) not in seen:
                seen.add((rr, cc))
                q.append((rr, cc))
    return len(seen) == len(mask_coords)


def _all_shape_variants() -> list[frozenset[Coord]]:
    variants: list[frozenset[Coord]] = []
    seen: set[frozenset[Coord]] = set()
    for shape in BASE_SHAPES:
        for k in range(4):
            for flip in (False, True):
                out: set[Coord] = set()
                for r, c in shape:
                    if flip:
                        c = -c
                    out.add(_rot((r, c), k))
                frozen = frozenset(out)
                if frozen not in seen and _shape_connected(frozen, 4):
                    seen.add(frozen)
                    variants.append(frozen)
    return variants


SHAPE_VARIANTS = _all_shape_variants()


def _trace_connected(offsets: frozenset[Coord]) -> bool:
    cells = set(offsets) | {(0, 0)}
    q: deque[Coord] = deque([(0, 0)])
    seen = {(0, 0)}
    while q:
        r, c = q.popleft()
        for rr, cc in _neighbors(r, c, 8):
            if (rr, cc) in cells and (rr, cc) not in seen:
                seen.add((rr, cc))
                q.append((rr, cc))
    return len(seen) == len(cells)


def _trace_options(sprite: dict, sprites: list[dict], count: int) -> list[frozenset[Coord]]:
    body = sorted(sprite["rel"] - {(0, 0)})
    same_color_others = [
        other
        for other in sprites
        if other is not sprite and int(other["color"]) == int(sprite["color"])
    ]
    options: list[frozenset[Coord]] = []
    for combo in itertools.combinations(body, int(count)):
        offsets = frozenset(combo)
        if not _trace_connected(offsets):
            continue
        if max(max(abs(r), abs(c)) for r, c in offsets) > 2:
            continue
        if any(offsets.issubset(other["rel"] - {(0, 0)}) for other in same_color_others):
            continue
        options.append(offsets)
    random.shuffle(options)
    return options


def _make_sprites() -> list[dict] | None:
    n_sprites = random.randint(3, 5)
    dup_color = random.choice(BODY_COLORS)
    remaining_colors = [c for c in BODY_COLORS if c != dup_color]
    colors = [dup_color, dup_color]
    colors.extend(random.sample(remaining_colors, n_sprites - 2))
    random.shuffle(colors)

    variants = SHAPE_VARIANTS[:]
    random.shuffle(variants)
    sprites: list[dict] = []
    for color in colors:
        choices = [
            shape
            for shape in variants
            if all(int(s["color"]) != int(color) or s["rel"] != shape for s in sprites)
        ]
        if not choices:
            return None
        shape = choices.pop()
        variants.remove(shape)
        sprites.append({"color": int(color), "rel": shape})

    color_counts = Counter(s["color"] for s in sprites)
    if max(color_counts.values()) < 2:
        return None
    if not all(_trace_options(sprite, sprites, 1) or _trace_options(sprite, sprites, 2) for sprite in sprites):
        return None
    return sprites


def _shape_global_cells(anchor: Coord, rel: frozenset[Coord]) -> set[Coord]:
    ar, ac = anchor
    return {(ar + dr, ac + dc) for dr, dc in rel}


def _cells_clear_with_padding(occupied: np.ndarray, cells: set[Coord], pad: int) -> bool:
    h, w = occupied.shape
    for r, c in cells:
        for rr in range(r - pad, r + pad + 1):
            for cc in range(c - pad, c + pad + 1):
                if 0 <= rr < h and 0 <= cc < w and occupied[rr, cc]:
                    return False
    return True


def _reserve_cells(occupied: np.ndarray, cells: set[Coord]) -> None:
    for r, c in cells:
        occupied[r, c] = True


def _place_shape(
    rect: Rect,
    rel: frozenset[Coord],
    occupied: np.ndarray,
    *,
    pad: int,
    attempts: int = 100,
) -> Coord | None:
    r0, c0, r1, c1 = rect
    drs = [dr for dr, _dc in rel]
    dcs = [dc for _dr, dc in rel]
    ar_min = r0 - min(drs)
    ar_max = r1 - max(drs)
    ac_min = c0 - min(dcs)
    ac_max = c1 - max(dcs)
    if ar_min > ar_max or ac_min > ac_max:
        return None

    for _ in range(attempts):
        anchor = (random.randint(ar_min, ar_max), random.randint(ac_min, ac_max))
        cells = _shape_global_cells(anchor, rel)
        if all(_in_rect(cell, rect) for cell in cells) and _cells_clear_with_padding(occupied, cells, pad):
            return anchor
    return None


def _paint_legend_sprite(g: np.ndarray, sprite: dict, anchor: Coord) -> None:
    color = int(sprite["color"])
    for r, c in _shape_global_cells(anchor, sprite["rel"]):
        g[r, c] = color
    g[anchor] = RED


def _paint_full_sprite(g: np.ndarray, sprite: dict, anchor: Coord) -> None:
    color = int(sprite["color"])
    for r, c in _shape_global_cells(anchor, sprite["rel"]):
        g[r, c] = color


def _paint_trace(g: np.ndarray, sprite: dict, anchor: Coord, offsets: frozenset[Coord]) -> None:
    ar, ac = anchor
    g[ar, ac] = RED
    color = int(sprite["color"])
    for dr, dc in offsets:
        g[ar + dr, ac + dc] = color


def _choose_trace_offsets(sprite: dict, sprites: list[dict], desired_count: int | None = None) -> frozenset[Coord] | None:
    counts = [desired_count] if desired_count is not None else [1, 2]
    if desired_count is None:
        random.shuffle(counts)
    for count in counts:
        options = _trace_options(sprite, sprites, int(count))
        if options:
            return random.choice(options)
    return None


def _choose_regions(shape: tuple[int, int]) -> tuple[Rect, Rect, str] | None:
    h, w = shape
    horizontal = random.random() < 0.5
    legend_first = random.random() < 0.5
    if horizontal:
        if h < 20:
            return None
        legend_h = random.randint(8, min(12, h - 9))
        work_h = h - legend_h - 1
        if work_h < 8:
            return None
        sep = legend_h if legend_first else work_h
        if legend_first:
            legend = (0, 0, sep - 1, w - 1)
            work = (sep + 1, 0, h - 1, w - 1)
        else:
            work = (0, 0, sep - 1, w - 1)
            legend = (sep + 1, 0, h - 1, w - 1)
        return legend, work, "h"

    if w < 20:
        return None
    legend_w = random.randint(8, min(12, w - 9))
    work_w = w - legend_w - 1
    if work_w < 8:
        return None
    sep = legend_w if legend_first else work_w
    if legend_first:
        legend = (0, 0, h - 1, sep - 1)
        work = (0, sep + 1, h - 1, w - 1)
    else:
        work = (0, 0, h - 1, sep - 1)
        legend = (0, sep + 1, h - 1, w - 1)
    return legend, work, "v"


def _draw_separator(g: np.ndarray, legend: Rect, work: Rect) -> None:
    h, w = g.shape
    if legend[1] == work[1] and legend[3] == work[3]:
        sep = legend[2] + 1 if legend[2] < work[0] else work[2] + 1
        g[sep, :] = GREY
    else:
        sep = legend[3] + 1 if legend[3] < work[1] else work[3] + 1
        g[:, sep] = GREY
    assert np.count_nonzero(g == GREY) in (h, w)


def _place_legend(g: np.ndarray, legend: Rect, sprites: list[dict]) -> bool:
    occupied = np.zeros_like(g, dtype=bool)
    order = sprites[:]
    random.shuffle(order)
    for sprite in order:
        anchor = _place_shape(legend, sprite["rel"], occupied, pad=1, attempts=200)
        if anchor is None:
            return False
        cells = _shape_global_cells(anchor, sprite["rel"])
        _reserve_cells(occupied, cells)
        _paint_legend_sprite(g, sprite, anchor)
    return True


def _target_sprites(sprites: list[dict]) -> list[dict]:
    by_color: dict[int, list[dict]] = {}
    for sprite in sprites:
        by_color.setdefault(int(sprite["color"]), []).append(sprite)
    duplicate_groups = [group for group in by_color.values() if len(group) >= 2]
    group = random.choice(duplicate_groups)
    targets = random.sample(group, 2)
    remaining = sprites[:]
    n_targets = random.randint(2, min(5, len(sprites) + 1))
    while len(targets) < n_targets:
        targets.append(random.choice(remaining))
    random.shuffle(targets)
    return targets


def _place_work(
    inp: np.ndarray,
    expected: np.ndarray,
    work: Rect,
    sprites: list[dict],
) -> bool:
    occupied_full = np.zeros_like(inp, dtype=bool)
    occupied_trace = np.zeros_like(inp, dtype=bool)
    targets = _target_sprites(sprites)
    desired_counts = [1, 2] + [None] * max(0, len(targets) - 2)
    random.shuffle(desired_counts)

    for sprite, desired in zip(targets, desired_counts):
        offsets = _choose_trace_offsets(sprite, sprites, desired)
        if offsets is None:
            return False

        for _ in range(200):
            anchor = _place_shape(work, sprite["rel"], occupied_full, pad=1, attempts=1)
            if anchor is None:
                continue
            full_cells = _shape_global_cells(anchor, sprite["rel"])
            trace_cells = {(anchor[0] + dr, anchor[1] + dc) for dr, dc in offsets} | {anchor}
            if not _cells_clear_with_padding(occupied_trace, trace_cells, 1):
                continue

            _reserve_cells(occupied_full, full_cells)
            _reserve_cells(occupied_trace, trace_cells)
            _paint_trace(inp, sprite, anchor, offsets)
            _paint_full_sprite(expected, sprite, anchor)
            break
        else:
            return False
    return True


def _expected_from_input(inp: np.ndarray) -> np.ndarray | None:
    try:
        return main(inp)
    except AssertionError:
        return None


def _valid_generated(inp: np.ndarray, expected: np.ndarray) -> bool:
    if max(inp.shape) > 30 or max(expected.shape) > 30:
        return False
    solved = _expected_from_input(inp)
    if solved is None or not np.array_equal(solved, expected):
        return False

    side_a, side_b = _split_by_separator(inp)
    ok_assignments = []
    for legend, work in ((side_a, side_b), (side_b, side_a)):
        try:
            sprites = _parse_legend(inp, legend)
            traces = _parse_work(inp, work)
        except AssertionError:
            continue
        legend_counts = Counter(sprite["color"] for sprite in sprites)
        trace_counts = Counter(trace["color"] for trace in traces)
        if any(legend_counts[color] >= 2 for color in trace_counts):
            ok_assignments.append((sprites, traces))

    if len(ok_assignments) != 1:
        return False
    sprites, traces = ok_assignments[0]
    if not any(len(trace["offsets"]) == 1 for trace in traces):
        return False
    if not any(len(trace["offsets"]) == 2 for trace in traces):
        return False

    diff = (inp != expected)
    if int(diff.sum()) < len(traces) * 2:
        return False
    return True


def _gen_attempt() -> np.ndarray | None:
    rows = random.randint(21, 30)
    cols = random.randint(21, 30)
    regions = _choose_regions((rows, cols))
    if regions is None:
        return None
    legend, work, _orientation = regions

    sprites = _make_sprites()
    if sprites is None:
        return None

    inp = np.full((rows, cols), BG, dtype=int)
    expected = inp.copy()
    _draw_separator(inp, legend, work)
    _draw_separator(expected, legend, work)

    if not _place_legend(inp, legend, sprites):
        return None
    expected[:, :] = inp
    if not _place_work(inp, expected, work, sprites):
        return None

    if _valid_generated(inp, expected):
        return inp
    return None


def generate_input() -> np.ndarray:
    for _ in range(MAX_GENERATION_ATTEMPTS):
        grid = _gen_attempt()
        if grid is not None:
            return grid
    raise RuntimeError("failed to generate a valid anchored trace completion input")
