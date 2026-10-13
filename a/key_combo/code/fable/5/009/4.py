from __future__ import annotations

import random
from itertools import combinations

import numpy as np


Coord = tuple[int, int]
ColoredCell = tuple[int, int, int]


def _background(grid: np.ndarray) -> int:
    """Return the most frequent color, following arc_utils.get_bg."""
    values, counts = np.unique(grid, return_counts=True)
    return int(values[int(np.argmax(counts))])


def _components8(grid: np.ndarray, bg: int) -> list[list[ColoredCell]]:
    """Find 8-connected multicolor foreground components in (row, col) coordinates."""
    height, width = grid.shape
    unseen = {(int(r), int(c)) for r, c in np.argwhere(grid != int(bg))}
    components: list[list[ColoredCell]] = []
    while unseen:
        start = unseen.pop()
        stack = [start]
        coords = [start]
        while stack:
            r, c = stack.pop()
            for dr in (-1, 0, 1):
                for dc in (-1, 0, 1):
                    if dr == 0 and dc == 0:
                        continue
                    nxt = (r + dr, c + dc)
                    if 0 <= nxt[0] < height and 0 <= nxt[1] < width and nxt in unseen:
                        unseen.remove(nxt)
                        stack.append(nxt)
                        coords.append(nxt)
        components.append([(r, c, int(grid[r, c])) for r, c in coords])
    return components


def _component_bbox(component: list[ColoredCell]) -> tuple[int, int, int, int]:
    rows = [r for r, _c, _color in component]
    cols = [c for _r, c, _color in component]
    return min(rows), min(cols), max(rows), max(cols)


def _align_partial(
    component: list[ColoredCell],
    sprite: dict[Coord, int],
    sprite_h: int,
    sprite_w: int,
    grid_shape: tuple[int, int],
) -> Coord:
    """Return the unique full-sprite anchor giving maximum exact color overlap."""
    height, width = grid_shape
    shifts: set[Coord] = set()
    for tr, tc, target_color in component:
        for (sr, sc), source_color in sprite.items():
            if int(target_color) == int(source_color):
                shifts.add((tr - sr, tc - sc))

    scored: list[tuple[int, Coord]] = []
    for anchor_r, anchor_c in shifts:
        if not (0 <= anchor_r <= height - sprite_h and 0 <= anchor_c <= width - sprite_w):
            continue
        score = sum(
            sprite.get((r - anchor_r, c - anchor_c)) == color
            for r, c, color in component
        )
        scored.append((int(score), (int(anchor_r), int(anchor_c))))

    assert scored, "every partial copy must have an in-bounds same-color alignment"
    best_score = max(score for score, _anchor in scored)
    best = sorted({anchor for score, anchor in scored if score == best_score})
    assert best_score == len(component), "a partial copy must be an exact colored subset of the sprite"
    assert len(best) == 1, "same-color maximum overlap must determine one anchor"
    return best[0]


def _recover(grid: np.ndarray) -> tuple[int, dict[Coord, int], int, int, set[Coord], int, int]:
    """Recover background, master sprite, visible anchors, and the two lattice periods."""
    bg = _background(grid)
    components = _components8(grid, bg)
    assert len(components) >= 3, "expected one master sprite and several partial copies"

    largest_area = max(len(component) for component in components)
    masters = [component for component in components if len(component) == largest_area]
    assert len(masters) == 1, "the complete sprite must be the unique largest component"
    master = masters[0]
    assert len({color for _r, _c, color in master}) >= 2, "the master sprite must be multicolored"

    r0, c0, r1, c1 = _component_bbox(master)
    sprite_h = r1 - r0 + 1
    sprite_w = c1 - c0 + 1
    sprite = {(r - r0, c - c0): color for r, c, color in master}
    master_anchor = (r0, c0)

    anchors = {master_anchor}
    for component in components:
        if component is master:
            continue
        assert 2 <= len(component) < len(master), "copies must be proper subsets containing at least two pixels"
        anchors.add(_align_partial(component, sprite, sprite_h, sprite_w, grid.shape))
    assert len(anchors) == len(components), "distinct components must describe distinct lattice slots"

    horizontal_diffs = [
        abs(c1_ - c0_)
        for r0_, c0_ in anchors
        for r1_, c1_ in anchors
        if r0_ == r1_ and c0_ < c1_
    ]
    vertical_diffs = [
        abs(r1_ - r0_)
        for r0_, c0_ in anchors
        for r1_, c1_ in anchors
        if c0_ == c1_ and r0_ < r1_
    ]
    assert horizontal_diffs and vertical_diffs, "visible anchors must expose both pure periods"
    period_c = min(horizontal_diffs)
    period_r = min(vertical_diffs)
    assert period_r > 0 and period_c > 0

    phase_r = next(iter(anchors))[0] % period_r
    phase_c = next(iter(anchors))[1] % period_c
    assert all(r % period_r == phase_r and c % period_c == phase_c for r, c in anchors), (
        "all anchors must share one rectangular-lattice phase"
    )
    assert all((r1_ - r0_) % period_r == 0 for r0_, _c0 in anchors for r1_, _c1 in anchors)
    assert all((c1_ - c0_) % period_c == 0 for _r0, c0_ in anchors for _r1, c1_ in anchors)
    return bg, sprite, sprite_h, sprite_w, anchors, period_r, period_c


def main(grid: np.ndarray) -> np.ndarray:
    """
    Complete a periodic lattice of multicolor sprites from one master and its partial afterimages.

    Input structure:
    - The dominant color is background; one unique largest 8-connected component is a complete
      multicolor sprite, and every other component is a proper colored subset of that sprite.
    - The top-left anchors of all visible copies share a rectangular lattice phase, with at least
      one visible adjacent horizontal pair and one visible adjacent vertical pair.
    - Some lattice slots whose complete sprite bounding box fits in the grid contain no pixels.

    Rule:
    - Align every partial copy to the master by its unique maximum same-color overlap, using the
      master's bounding-box top-left as the anchor and preserving the grid's row/column frame.
    - Infer the horizontal and vertical periods from same-row and same-column visible anchors.
    - Stamp the complete master at every in-bounds position of that lattice phase, completing
      partial copies and also filling slots that were entirely background.

    Key insight:
    The partial pixels first reveal their own anchors through color-sensitive overlap; those
    anchors then reveal a two-dimensional rhythm whose invisible in-bounds slots must also be stamped.
    """
    g = np.asarray(grid, dtype=int)
    assert g.ndim == 2 and g.size > 0, "expected a non-empty 2D grid"
    bg, sprite, sprite_h, sprite_w, anchors, period_r, period_c = _recover(g)

    phase_r = next(iter(anchors))[0] % period_r
    phase_c = next(iter(anchors))[1] % period_c
    lattice_rows = range(phase_r, g.shape[0] - sprite_h + 1, period_r)
    lattice_cols = range(phase_c, g.shape[1] - sprite_w + 1, period_c)

    out = g.copy()
    for anchor_r in lattice_rows:
        for anchor_c in lattice_cols:
            for (dr, dc), color in sprite.items():
                r, c = anchor_r + dr, anchor_c + dc
                assert int(out[r, c]) in {int(bg), int(color)}, "lattice stamps must not create color conflicts"
                out[r, c] = int(color)
    return out


# ------------------------------ generator ------------------------------

MAX_GENERATION_ATTEMPTS = 300


def _connected_offsets(offsets: set[Coord]) -> bool:
    if not offsets:
        return False
    seen = {next(iter(offsets))}
    stack = list(seen)
    while stack:
        r, c = stack.pop()
        for dr in (-1, 0, 1):
            for dc in (-1, 0, 1):
                nxt = (r + dr, c + dc)
                if nxt in offsets and nxt not in seen:
                    seen.add(nxt)
                    stack.append(nxt)
    return seen == offsets


def _grow_shape(height: int, width: int, area: int) -> set[Coord] | None:
    """Grow a 4-connected irregular shape whose bbox is exactly height x width."""
    start = (random.randrange(height), random.randrange(width))
    shape = {start}
    while len(shape) < area:
        frontier: set[Coord] = set()
        for r, c in shape:
            for dr, dc in ((-1, 0), (1, 0), (0, -1), (0, 1)):
                nxt = (r + dr, c + dc)
                if 0 <= nxt[0] < height and 0 <= nxt[1] < width and nxt not in shape:
                    frontier.add(nxt)
        if not frontier:
            return None
        shape.add(random.choice(tuple(frontier)))
    rows = {r for r, _c in shape}
    cols = {c for _r, c in shape}
    if not ({0, height - 1} <= rows and {0, width - 1} <= cols):
        return None
    return shape


def _local_match_shifts(
    sprite: dict[Coord, int], partial: dict[Coord, int], *, use_color: bool
) -> set[Coord]:
    """All master-anchor shifts that reproduce a local partial, with optional color matching."""
    candidates: set[Coord] = set()
    for (pr, pc), pcolor in partial.items():
        for (sr, sc), scolor in sprite.items():
            if not use_color or int(pcolor) == int(scolor):
                candidates.add((pr - sr, pc - sc))
    valid: set[Coord] = set()
    for dr, dc in candidates:
        good = True
        for (pr, pc), pcolor in partial.items():
            source = (pr - dr, pc - dc)
            if source not in sprite or (use_color and int(sprite[source]) != int(pcolor)):
                good = False
                break
        if good:
            valid.add((dr, dc))
    return valid


def _partial_catalog(sprite: dict[Coord, int]) -> tuple[list[dict[Coord, int]], list[dict[Coord, int]]]:
    """Return all uniquely colored partials and the subset that needs color to disambiguate."""
    offsets = sorted(sprite)
    unique: list[dict[Coord, int]] = []
    color_critical: list[dict[Coord, int]] = []
    max_size = min(5, len(offsets) - 1)
    for size in range(2, max_size + 1):
        for combo in combinations(offsets, size):
            coord_set = set(combo)
            if not _connected_offsets(coord_set):
                continue
            partial = {coord: int(sprite[coord]) for coord in combo}
            colored = _local_match_shifts(sprite, partial, use_color=True)
            if colored != {(0, 0)}:
                continue
            unique.append(partial)
            if len(_local_match_shifts(sprite, partial, use_color=False)) > 1:
                color_critical.append(partial)
    return unique, color_critical


def _make_sprite(bg: int) -> tuple[dict[Coord, int], int, int, list[dict[Coord, int]], list[dict[Coord, int]]] | None:
    height = random.randint(3, 5)
    width = random.randint(3, 5)
    max_area = min(11, height * width - 1)
    if max_area < 7:
        return None
    shape = _grow_shape(height, width, random.randint(7, max_area))
    if shape is None:
        return None

    palette = [color for color in range(10) if color != bg]
    color_count = random.randint(3, min(4, len(shape)))
    colors = random.sample(palette, color_count)
    shuffled = list(shape)
    random.shuffle(shuffled)
    sprite: dict[Coord, int] = {}
    for index, coord in enumerate(shuffled):
        sprite[coord] = colors[index] if index < color_count else random.choice(colors)

    unique, critical = _partial_catalog(sprite)
    trimmed = [
        partial
        for partial in unique
        if min(r for r, _c in partial) > 0 or min(c for _r, c in partial) > 0
    ]
    if not critical or not trimmed:
        return None
    return sprite, height, width, unique, critical


def _axis_geometry(sprite_dim: int) -> tuple[int, int, int, list[int]]:
    """Choose canvas length, period, phase, and all fully in-bounds anchor coordinates."""
    options: list[tuple[int, int, int, list[int]]] = []
    for period in range(sprite_dim + 2, sprite_dim + 5):
        for count in (3, 4):
            for phase in range(period):
                for trailing in range(period):
                    length = phase + (count - 1) * period + sprite_dim + trailing
                    if 16 <= length <= 30:
                        anchors = [phase + index * period for index in range(count)]
                        options.append((length, period, phase, anchors))
    assert options
    return random.choice(options)


def _paint_sprite(grid: np.ndarray, anchor: Coord, sprite: dict[Coord, int]) -> None:
    ar, ac = anchor
    for (dr, dc), color in sprite.items():
        grid[ar + dr, ac + dc] = int(color)


def _oracle(
    shape: tuple[int, int], bg: int, anchors: list[Coord], sprite: dict[Coord, int]
) -> np.ndarray:
    out = np.full(shape, int(bg), dtype=int)
    for anchor in anchors:
        _paint_sprite(out, anchor, sprite)
    return out


def _build_case() -> tuple[np.ndarray, np.ndarray, dict[str, object]] | None:
    bg = random.randrange(10)
    made = _make_sprite(bg)
    if made is None:
        return None
    sprite, sprite_h, sprite_w, partials, criticals = made
    trimmed = [
        partial
        for partial in partials
        if min(r for r, _c in partial) > 0 or min(c for _r, c in partial) > 0
    ]

    height, period_r, _phase_r, rows = _axis_geometry(sprite_h)
    width, period_c, _phase_c, cols = _axis_geometry(sprite_w)
    indexed = [(ri, ci) for ri in range(len(rows)) for ci in range(len(cols))]

    # An entirely invisible extreme row/column forces genuine periodic extension beyond
    # the bounding box of visible anchors. The side and orientation vary independently.
    empty_axis = random.choice(("row", "col"))
    empty_side = random.choice((0, -1))
    if empty_axis == "row":
        empty_index = empty_side % len(rows)
        forced_empty = {(empty_index, ci) for ci in range(len(cols))}
    else:
        empty_index = empty_side % len(cols)
        forced_empty = {(ri, empty_index) for ri in range(len(rows))}

    occupied_indices = [slot for slot in indexed if slot not in forced_empty]
    core_options = [
        {(ri, ci), (ri + 1, ci), (ri, ci + 1), (ri + 1, ci + 1)}
        for ri in range(len(rows) - 1)
        for ci in range(len(cols) - 1)
        if not ({(ri, ci), (ri + 1, ci), (ri, ci + 1), (ri + 1, ci + 1)} & forced_empty)
    ]
    if not core_options:
        return None
    core = random.choice(core_options)

    optional = [slot for slot in occupied_indices if slot not in core]
    random.shuffle(optional)
    extra_empty_count = random.randint(0, min(2, max(0, len(optional) - 2)))
    empty_indices = set(forced_empty) | set(optional[:extra_empty_count])
    occupied_indices = [slot for slot in indexed if slot not in empty_indices]
    if len(occupied_indices) < 5:
        return None

    all_anchors = [(rows[ri], cols[ci]) for ri, ci in indexed]
    empty_anchors = {(rows[ri], cols[ci]) for ri, ci in empty_indices}
    occupied_anchors = [(rows[ri], cols[ci]) for ri, ci in occupied_indices]
    source_anchor = random.choice(occupied_anchors)
    copy_anchors = [anchor for anchor in occupied_anchors if anchor != source_anchor]
    if len(copy_anchors) < 4:
        return None

    random.shuffle(copy_anchors)
    critical_anchor = copy_anchors[0]
    trimmed_anchor = copy_anchors[1]
    critical_partial = random.choice(criticals)
    trimmed_choices = [p for p in trimmed if p != critical_partial] or trimmed
    trimmed_partial = random.choice(trimmed_choices)

    chosen: dict[Coord, dict[Coord, int]] = {
        critical_anchor: critical_partial,
        trimmed_anchor: trimmed_partial,
    }
    for anchor in copy_anchors[2:]:
        chosen[anchor] = random.choice(partials)

    inp = np.full((height, width), int(bg), dtype=int)
    _paint_sprite(inp, source_anchor, sprite)
    for anchor, partial in chosen.items():
        _paint_sprite(inp, anchor, partial)
    expected = _oracle(inp.shape, bg, all_anchors, sprite)
    meta: dict[str, object] = {
        "bg": bg,
        "sprite": sprite,
        "sprite_h": sprite_h,
        "sprite_w": sprite_w,
        "all_anchors": set(all_anchors),
        "empty_anchors": empty_anchors,
        "occupied_anchors": set(occupied_anchors),
        "source_anchor": source_anchor,
        "partials": chosen,
        "critical_anchor": critical_anchor,
        "trimmed_anchor": trimmed_anchor,
        "period_r": period_r,
        "period_c": period_c,
    }
    return inp, expected, meta


def _semantic_check(inp: np.ndarray, expected: np.ndarray, meta: dict[str, object]) -> bool:
    """Check task semantics and anti-shortcut properties, not only solver closure."""
    bg = int(meta["bg"])
    sprite = meta["sprite"]
    assert isinstance(sprite, dict)
    sprite_h = int(meta["sprite_h"])
    sprite_w = int(meta["sprite_w"])
    empty_anchors = set(meta["empty_anchors"])
    occupied_anchors = set(meta["occupied_anchors"])
    all_anchors = set(meta["all_anchors"])
    partials = meta["partials"]
    assert isinstance(partials, dict)

    if inp.ndim != 2 or max(inp.shape) > 30 or _background(inp) != bg or _background(expected) != bg:
        return False
    if np.array_equal(inp, expected) or not empty_anchors or occupied_anchors | empty_anchors != all_anchors:
        return False

    # Empty slots really contain no trace, and the oracle adds a whole sprite there.
    for ar, ac in empty_anchors:
        if np.any(inp[ar : ar + sprite_h, ac : ac + sprite_w] != bg):
            return False
        if any(int(expected[ar + dr, ac + dc]) != int(color) for (dr, dc), color in sprite.items()):
            return False

    # One invisible extreme lattice line lies beyond the bbox of all visible anchors.
    visible_rows = [r for r, _c in occupied_anchors]
    visible_cols = [c for _r, c in occupied_anchors]
    if not any(
        r < min(visible_rows) or r > max(visible_rows) or c < min(visible_cols) or c > max(visible_cols)
        for r, c in empty_anchors
    ):
        return False

    critical = partials[meta["critical_anchor"]]
    trimmed = partials[meta["trimmed_anchor"]]
    if _local_match_shifts(sprite, critical, use_color=True) != {(0, 0)}:
        return False
    if len(_local_match_shifts(sprite, critical, use_color=False)) <= 1:
        return False
    if not (min(r for r, _c in trimmed) > 0 or min(c for _r, c in trimmed) > 0):
        return False

    try:
        bg2, sprite2, sh2, sw2, anchors2, pr2, pc2 = _recover(inp)
        solved = main(inp)
    except (AssertionError, IndexError, ValueError):
        return False
    if bg2 != bg or sprite2 != sprite or (sh2, sw2) != (sprite_h, sprite_w):
        return False
    if anchors2 != occupied_anchors or pr2 != int(meta["period_r"]) or pc2 != int(meta["period_c"]):
        return False
    if not np.array_equal(solved, expected):
        return False

    # Both period directions are exposed by adjacent visible pairs.
    pr, pc = int(meta["period_r"]), int(meta["period_c"])
    if not any((r, c + pc) in occupied_anchors for r, c in occupied_anchors):
        return False
    if not any((r + pr, c) in occupied_anchors for r, c in occupied_anchors):
        return False

    # There are additions caused independently by partial completion and empty-slot stamping.
    partial_addition = any(
        int(inp[ar + dr, ac + dc]) == bg
        for (ar, ac), partial in partials.items()
        for dr, dc in sprite
        if (dr, dc) not in partial
    )
    empty_addition = any(int(inp[ar + dr, ac + dc]) == bg for ar, ac in empty_anchors for dr, dc in sprite)
    return bool(partial_addition and empty_addition)


def generate_input() -> np.ndarray:
    """Generate varied, semantically validated lattice-completion inputs up to 30x30."""
    for _ in range(MAX_GENERATION_ATTEMPTS):
        case = _build_case()
        if case is None:
            continue
        inp, expected, meta = case
        if _semantic_check(inp, expected, meta):
            return inp
    raise RuntimeError("failed to generate a valid residual-lattice input")
