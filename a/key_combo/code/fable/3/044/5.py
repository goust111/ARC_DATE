from __future__ import annotations

import random

import numpy as np


BLACK = 0
PALETTE = tuple(range(1, 10))
Coord = tuple[int, int]
Step = tuple[Coord, int, set[Coord], tuple[int, int]]


def _neighbors8(r: int, c: int) -> list[Coord]:
    return [
        (r + dr, c + dc)
        for dr in (-1, 0, 1)
        for dc in (-1, 0, 1)
        if not (dr == 0 and dc == 0)
    ]


def _bbox(coords: set[Coord]) -> tuple[int, int, int, int]:
    rows = [r for r, _c in coords]
    cols = [c for _r, c in coords]
    return min(rows), min(cols), max(rows), max(cols)


def _components_by_color(g: np.ndarray) -> list[dict]:
    h, w = g.shape
    seen = np.zeros(g.shape, dtype=bool)
    comps: list[dict] = []

    for sr in range(h):
        for sc in range(w):
            color = int(g[sr, sc])
            if color == BLACK or seen[sr, sc]:
                continue

            stack = [(sr, sc)]
            seen[sr, sc] = True
            coords: set[Coord] = set()
            while stack:
                r, c = stack.pop()
                coords.add((int(r), int(c)))
                for nr, nc in _neighbors8(r, c):
                    if 0 <= nr < h and 0 <= nc < w and not seen[nr, nc] and int(g[nr, nc]) == color:
                        seen[nr, nc] = True
                        stack.append((nr, nc))

            comps.append({"color": color, "coords": coords, "area": len(coords), "bbox": _bbox(coords)})

    return comps


def _touching(coords: set[Coord], p: Coord) -> bool:
    pr, pc = p
    return any((r, c) != p and max(abs(r - pr), abs(c - pc)) == 1 for r, c in coords)


def _indicator_relation(coords: set[Coord], p: Coord) -> tuple[int, int]:
    r0, c0, r1, c1 = _bbox(coords)
    pr, pc = p

    if pr == r0 - 1:
        rr = -1
    elif pr == r1 + 1:
        rr = 1
    elif r0 <= pr <= r1:
        rr = 0
    else:
        raise AssertionError("indicator must sit in the current object's one-cell bbox ring")

    if pc == c0 - 1:
        cc = -1
    elif pc == c1 + 1:
        cc = 1
    elif c0 <= pc <= c1:
        cc = 0
    else:
        raise AssertionError("indicator must sit in the current object's one-cell bbox ring")

    assert (rr, cc) != (0, 0), "indicator cannot overlap the current object's bbox"
    assert _touching(coords, p), "indicator must touch the current object by edge or corner"
    return rr, cc


def _reflect_coords(coords: set[Coord], p: Coord) -> tuple[set[Coord], tuple[int, int]]:
    r0, c0, r1, c1 = _bbox(coords)
    rr, cc = _indicator_relation(coords, p)
    out: set[Coord] = set()

    for r, c in coords:
        nr = r
        nc = c
        if rr == -1:
            nr = 2 * r0 - 1 - r
        elif rr == 1:
            nr = 2 * r1 + 1 - r
        if cc == -1:
            nc = 2 * c0 - 1 - c
        elif cc == 1:
            nc = 2 * c1 + 1 - c
        out.add((int(nr), int(nc)))

    assert len(out) == len(coords), "reflection must preserve sprite cell count"
    return out, (rr, cc)


def _parse_input(g: np.ndarray) -> tuple[int, set[Coord], dict[Coord, int]]:
    assert g.ndim == 2, "expected a 2D grid"
    assert max(g.shape) <= 30, "ARC grids must fit within 30x30"

    comps = _components_by_color(g)
    sprites = [comp for comp in comps if int(comp["area"]) > 1]
    indicators = [comp for comp in comps if int(comp["area"]) == 1]
    assert len(sprites) == 1, "expected exactly one multi-cell sprite"
    assert indicators, "expected at least one single-pixel indicator"

    sprite = sprites[0]
    sprite_color = int(sprite["color"])
    indicator_map: dict[Coord, int] = {}
    indicator_colors: list[int] = []
    for comp in indicators:
        color = int(comp["color"])
        assert color != sprite_color, "indicator colors must differ from the sprite color"
        coord = next(iter(comp["coords"]))
        indicator_map[coord] = color
        indicator_colors.append(color)

    assert len(indicator_colors) == len(set(indicator_colors)), "indicator colors must be mutually distinct"
    return sprite_color, set(sprite["coords"]), indicator_map


def _trace_relay(g: np.ndarray) -> tuple[int, set[Coord], dict[Coord, int], list[Step]]:
    h, w = g.shape
    sprite_color, sprite_coords, indicators = _parse_input(g)
    unused = dict(indicators)
    current = set(sprite_coords)
    steps: list[Step] = []

    for _ in range(len(indicators)):
        adjacent = [p for p in unused if _touching(current, p)]
        if not adjacent:
            break
        assert len(adjacent) == 1, "each relay state must touch exactly one unused indicator"

        p = adjacent[0]
        color = unused.pop(p)
        new_coords, relation = _reflect_coords(current, p)
        for r, c in new_coords:
            assert 0 <= r < h and 0 <= c < w, "reflected copy must stay in bounds"
        steps.append((p, color, new_coords, relation))
        current = new_coords

    assert steps, "the original sprite must touch one relay indicator"
    return sprite_color, sprite_coords, indicators, steps


def _render_from_trace(
    shape: tuple[int, int],
    sprite_color: int,
    sprite_coords: set[Coord],
    indicators: dict[Coord, int],
    steps: list[Step],
) -> np.ndarray:
    out = np.zeros(shape, dtype=int)
    for r, c in sprite_coords:
        out[r, c] = int(sprite_color)

    used = {p for p, _color, _coords, _rel in steps}
    for p, color in indicators.items():
        if p not in used:
            out[p] = int(color)

    for _p, color, coords, _rel in steps:
        for r, c in coords:
            existing = int(out[r, c])
            assert existing in (BLACK, int(color)), "reflected copies and remaining indicators must not conflict"
            out[r, c] = int(color)
    return out


def main(grid: np.ndarray) -> np.ndarray:
    """
    Relay-mirror a sprite through adjacent indicator pixels, recoloring each reflected copy.

    Input structure:
    - A black background contains one monochrome multi-cell sprite and several mutually distinct single-pixel indicators.
    - Exactly one unused indicator touches the current sprite or latest reflected copy at each relay step.
    - Some indicators may be unused distractors, but the final copy touches none of the remaining indicators.

    Rule:
    - Starting from the original sprite, find the sole unused indicator adjacent to the current object.
    - Reflect the current object across the bbox side or corner marked by that indicator, then recolor the copy to the indicator color.
    - Repeat from the newest copy until no unused adjacent indicator remains; remove used indicators and keep unused ones.

    Key insight:
    - The next mirror axis is determined recursively by adjacency to the latest produced copy, not by color order or by repeatedly using the original sprite.
    """
    g = np.asarray(grid, dtype=int)
    sprite_color, sprite_coords, indicators, steps = _trace_relay(g)
    return _render_from_trace(g.shape, sprite_color, sprite_coords, indicators, steps)


# --------------- generator ---------------


_MAX_ATTEMPTS = 800
_EDGE_DIRS = ((-1, 0), (1, 0), (0, -1), (0, 1))
_CORNER_DIRS = ((-1, -1), (-1, 1), (1, -1), (1, 1))
_ALL_DIRS = _EDGE_DIRS + _CORNER_DIRS
_SPRITE_SIZES = ((3, 3), (3, 4), (4, 3), (4, 4), (4, 5), (5, 4))


def _mask_connected8(mask: np.ndarray) -> bool:
    pts = [tuple(map(int, rc)) for rc in np.argwhere(mask)]
    if not pts:
        return False
    todo = [pts[0]]
    seen = {pts[0]}
    pset = set(pts)
    while todo:
        r, c = todo.pop()
        for nb in _neighbors8(r, c):
            if nb in pset and nb not in seen:
                seen.add(nb)
                todo.append(nb)
    return len(seen) == len(pset)


def _random_sprite_mask() -> np.ndarray | None:
    for _ in range(100):
        h, w = random.choice(_SPRITE_SIZES)
        mask = np.ones((h, w), dtype=bool)
        corners = {(0, 0), (0, w - 1), (h - 1, 0), (h - 1, w - 1)}
        removable = [(r, c) for r in range(h) for c in range(w) if (r, c) not in corners]
        random.shuffle(removable)

        target_remove = random.randint(2, min(5, max(2, len(removable) // 2)))
        removed = 0
        for r, c in removable:
            if removed >= target_remove:
                break
            trial = mask.copy()
            trial[r, c] = False
            if int(trial.sum()) < 5:
                continue
            if _mask_connected8(trial):
                mask = trial
                removed += 1

        if removed < 2:
            continue
        if np.array_equal(mask, np.flipud(mask)):
            continue
        if np.array_equal(mask, np.fliplr(mask)):
            continue
        if np.array_equal(mask, np.flipud(np.fliplr(mask))):
            continue
        return mask
    return None


def _candidate_indicators(coords: set[Coord], direction: tuple[int, int]) -> list[Coord]:
    r0, c0, r1, c1 = _bbox(coords)
    dr, dc = direction
    out: set[Coord] = set()

    if direction == (-1, 0):
        out = {(r0 - 1, c) for r, c in coords if r == r0}
    elif direction == (1, 0):
        out = {(r1 + 1, c) for r, c in coords if r == r1}
    elif direction == (0, -1):
        out = {(r, c0 - 1) for r, c in coords if c == c0}
    elif direction == (0, 1):
        out = {(r, c1 + 1) for r, c in coords if c == c1}
    else:
        corner_cell = (r0 if dr == -1 else r1, c0 if dc == -1 else c1)
        if corner_cell in coords:
            out = {(r0 - 1 if dr == -1 else r1 + 1, c0 - 1 if dc == -1 else c1 + 1)}

    return sorted(out)


def _direction_sequence(length: int) -> list[tuple[int, int]]:
    required = [random.choice(_EDGE_DIRS), random.choice(_CORNER_DIRS)]
    random.shuffle(required)
    dirs: list[tuple[int, int]] = []

    while len(dirs) < length:
        if required:
            choices = [required.pop()]
        else:
            choices = list(_ALL_DIRS)
            random.shuffle(choices)

        chosen = None
        for cand in choices:
            if not dirs or cand != (-dirs[-1][0], -dirs[-1][1]):
                chosen = cand
                break
        if chosen is None:
            chosen = random.choice(_ALL_DIRS)
        dirs.append(chosen)

    return dirs


def _build_chain(mask: np.ndarray, length: int) -> tuple[set[Coord], list[Coord], list[set[Coord]], list[tuple[int, int]]] | None:
    original = {tuple(map(int, rc)) for rc in np.argwhere(mask)}
    current = set(original)
    indicators: list[Coord] = []
    copies: list[set[Coord]] = []
    relations: list[tuple[int, int]] = []

    for direction in _direction_sequence(length):
        choices = _candidate_indicators(current, direction)
        if not choices:
            return None
        p = random.choice(choices)
        if p in indicators or p in original:
            return None
        new_coords, relation = _reflect_coords(current, p)
        indicators.append(p)
        copies.append(new_coords)
        relations.append(relation)
        current = new_coords

    all_objects = [original] + copies
    occupied: set[Coord] = set()
    for obj in all_objects:
        if occupied & obj:
            return None
        occupied |= obj

    if len(set(indicators)) != len(indicators):
        return None
    indicator_set = set(indicators)
    for i, p in enumerate(indicators):
        for j, obj in enumerate(all_objects):
            if p in obj and j != i + 1:
                return None

    for a in indicators:
        for b in indicators:
            if a < b and max(abs(a[0] - b[0]), abs(a[1] - b[1])) <= 1:
                return None

    if not any(rel[0] == 0 or rel[1] == 0 for rel in relations):
        return None
    if not any(rel[0] != 0 and rel[1] != 0 for rel in relations):
        return None
    if indicator_set & original:
        return None
    return original, indicators, copies, relations


def _shift(coords: set[Coord], dr: int, dc: int) -> set[Coord]:
    return {(r + dr, c + dc) for r, c in coords}


def _place_distractors(
    grid: np.ndarray,
    colors: list[int],
    chain_objects: list[set[Coord]],
    used_indicators: list[Coord],
) -> dict[Coord, int] | None:
    if not colors:
        return {}

    h, w = grid.shape
    blocked: set[Coord] = set()
    for coords in chain_objects:
        for r, c in coords:
            for dr in (-1, 0, 1):
                for dc in (-1, 0, 1):
                    nr, nc = r + dr, c + dc
                    if 0 <= nr < h and 0 <= nc < w:
                        blocked.add((nr, nc))
    for r, c in used_indicators:
        for dr in (-1, 0, 1):
            for dc in (-1, 0, 1):
                nr, nc = r + dr, c + dc
                if 0 <= nr < h and 0 <= nc < w:
                    blocked.add((nr, nc))

    free = [(r, c) for r in range(h) for c in range(w) if int(grid[r, c]) == BLACK and (r, c) not in blocked]
    if len(free) < len(colors):
        return None
    random.shuffle(free)

    out: dict[Coord, int] = {}
    for p, color in zip(free, colors):
        out[p] = int(color)
        grid[p] = int(color)
    return out


def _expected_from_chain(
    shape: tuple[int, int],
    primary_color: int,
    original: set[Coord],
    copies: list[set[Coord]],
    indicator_colors: list[int],
    distractors: dict[Coord, int],
) -> np.ndarray | None:
    out = np.zeros(shape, dtype=int)
    for r, c in original:
        if int(out[r, c]) != BLACK:
            return None
        out[r, c] = int(primary_color)
    for p, color in distractors.items():
        if int(out[p]) != BLACK:
            return None
        out[p] = int(color)
    for coords, color in zip(copies, indicator_colors):
        for r, c in coords:
            if int(out[r, c]) not in (BLACK, int(color)):
                return None
            out[r, c] = int(color)
    return out


def _semantic_checks(inp: np.ndarray, expected: np.ndarray) -> bool:
    try:
        solved = main(inp)
        if not np.array_equal(solved, expected):
            return False

        sprite_color, sprite_coords, indicators, steps = _trace_relay(inp)
        if len(steps) < 2:
            return False
        relations = [rel for _p, _color, _coords, rel in steps]
        if not any(rel[0] == 0 or rel[1] == 0 for rel in relations):
            return False
        if not any(rel[0] != 0 and rel[1] != 0 for rel in relations):
            return False

        sprite_area = len(sprite_coords)
        used_colors = [color for _p, color, _coords, _rel in steps]
        for color in used_colors:
            if int(np.sum(expected == int(color))) != sprite_area:
                return False
        if int(np.sum(expected == int(sprite_color))) != sprite_area:
            return False

        first_step_only = _render_from_trace(inp.shape, sprite_color, sprite_coords, indicators, steps[:1])
        if np.array_equal(first_step_only, expected):
            return False
        if len(indicators) == len(steps):
            return False
    except Exception:
        return False
    return True


def _gen_attempt() -> np.ndarray | None:
    mask = _random_sprite_mask()
    if mask is None:
        return None

    chain_len = random.randint(2, 4)
    built = _build_chain(mask, chain_len)
    if built is None:
        return None
    original, indicators, copies, _relations = built

    all_coords = set(original)
    for coords in copies:
        all_coords |= coords
    all_coords |= set(indicators)
    r0, c0, r1, c1 = _bbox(all_coords)
    span_h = r1 - r0 + 1
    span_w = c1 - c0 + 1
    if span_h + 2 > 30 or span_w + 2 > 30:
        return None

    h = random.randint(max(14, span_h + 2), 30)
    w = random.randint(max(14, span_w + 2), 30)
    dr = random.randint(1 - r0, h - 2 - r1)
    dc = random.randint(1 - c0, w - 2 - c1)

    original = _shift(original, dr, dc)
    copies = [_shift(coords, dr, dc) for coords in copies]
    indicators = [(r + dr, c + dc) for r, c in indicators]

    distractor_count = random.choice((1, 1, 2))
    if 1 + chain_len + distractor_count > len(PALETTE):
        distractor_count = len(PALETTE) - 1 - chain_len
    colors = random.sample(PALETTE, 1 + chain_len + distractor_count)
    primary_color = colors[0]
    indicator_colors = colors[1 : 1 + chain_len]
    distractor_colors = colors[1 + chain_len :]

    inp = np.zeros((h, w), dtype=int)
    for r, c in original:
        inp[r, c] = int(primary_color)
    for p, color in zip(indicators, indicator_colors):
        if int(inp[p]) != BLACK:
            return None
        inp[p] = int(color)

    distractors = _place_distractors(inp, distractor_colors, [original] + copies, indicators)
    if distractors is None:
        return None

    expected = _expected_from_chain((h, w), primary_color, original, copies, indicator_colors, distractors)
    if expected is None:
        return None
    if not _semantic_checks(inp, expected):
        return None
    return inp


def generate_input() -> np.ndarray:
    for _ in range(_MAX_ATTEMPTS):
        grid = _gen_attempt()
        if grid is not None:
            return grid
    raise RuntimeError("failed to generate a valid mirror relay input")
