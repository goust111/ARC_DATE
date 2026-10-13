from __future__ import annotations

from collections import Counter, deque
import itertools
import random

import numpy as np


BG = 0
GREY = 5
ARC_COLORS = tuple(range(10))
NEIGHBORS4 = ((-1, 0), (1, 0), (0, -1), (0, 1))
GROUP_COUNT_CHOICES = (3, 4, 5)
HEIGHT_RANGE = (7, 12)
WIDTH_CHOICES = (3, 4, 5, 6)
SEPARATOR_CHOICES = (1, 2)
SHIFT_RANGE = range(-3, 4)
MAX_GENERATION_ATTEMPTS = 800

Coord = tuple[int, int]


def _background(grid: np.ndarray) -> int:
    values, counts = np.unique(grid, return_counts=True)
    return int(values[int(np.argmax(counts))])


def _neighbors4(r: int, c: int):
    for dr, dc in NEIGHBORS4:
        yield int(r) + dr, int(c) + dc


def _components(mask: np.ndarray) -> list[np.ndarray]:
    h, w = mask.shape
    seen = np.zeros((h, w), dtype=bool)
    out: list[np.ndarray] = []
    for sr_raw, sc_raw in np.argwhere(mask):
        sr, sc = int(sr_raw), int(sc_raw)
        if seen[sr, sc]:
            continue
        q: deque[Coord] = deque([(sr, sc)])
        seen[sr, sc] = True
        coords: list[Coord] = []
        while q:
            r, c = q.popleft()
            coords.append((r, c))
            for rr, cc in _neighbors4(r, c):
                if 0 <= rr < h and 0 <= cc < w and mask[rr, cc] and not seen[rr, cc]:
                    seen[rr, cc] = True
                    q.append((rr, cc))
        out.append(np.array(coords, dtype=int))
    return out


def _degree(cells: set[Coord], r: int, c: int) -> int:
    return sum((rr, cc) in cells for rr, cc in _neighbors4(r, c))


def _column_groups(g: np.ndarray, bg: int) -> list[np.ndarray]:
    nonempty = np.any(g != int(bg), axis=0)
    groups: list[np.ndarray] = []
    c = 0
    while c < g.shape[1]:
        if not nonempty[c]:
            c += 1
            continue
        c0 = c
        while c < g.shape[1] and nonempty[c]:
            c += 1
        groups.append(g[:, c0:c].copy())
    assert groups, "expected at least one foreground column group"
    return groups


def _group_info(group: np.ndarray, bg: int, index: int) -> dict:
    h, w = group.shape
    assert h > 0 and w >= 2, "each group must have positive height and width"
    mask = group != int(bg)
    assert mask.any(), "empty group"

    nonempty_cols = np.flatnonzero(mask.any(axis=0))
    assert len(nonempty_cols) == w, "a group must occupy one continuous run of non-background columns"
    assert int(nonempty_cols[0]) == 0 and int(nonempty_cols[-1]) == w - 1

    left_rows = np.flatnonzero(mask[:, 0])
    right_rows = np.flatnonzero(mask[:, w - 1])
    assert len(left_rows) == 1 and len(right_rows) == 1, "each group needs one left and one right edge endpoint"
    left_row = int(left_rows[0])
    right_row = int(right_rows[0])
    left = (left_row, 0)
    right = (right_row, w - 1)

    comps = _components(mask)
    assert len(comps) == 1, "the path cells in a group must be 4-connected"
    cells = {tuple(map(int, p)) for p in comps[0].tolist()}
    degree_one: set[Coord] = set()
    for r, c in cells:
        deg = _degree(cells, r, c)
        assert deg in (1, 2), "path groups may not branch or thicken"
        if deg == 1:
            degree_one.add((r, c))
    assert degree_one == {left, right}, "the visible path endpoints must be the two edge connectors"

    endpoint_cells = {left, right}
    body_colors = {
        int(group[r, c])
        for r, c in cells
        if (r, c) not in endpoint_cells
    }
    assert len(body_colors) == 1, "all non-connector path cells must share one body color"
    body = next(iter(body_colors))

    left_color = int(group[left])
    right_color = int(group[right])
    assert left_color != int(bg) and right_color != int(bg), "connectors must be foreground colors"
    assert left_color != body and right_color != body, "connector colors must differ from the path body color"

    return {
        "index": int(index),
        "patch": group,
        "width": int(w),
        "left_row": int(left_row),
        "right_row": int(right_row),
        "left_color": int(left_color),
        "right_color": int(right_color),
        "body_color": int(body),
    }


def _connector_counts(infos: list[dict]) -> Counter:
    counts: Counter = Counter()
    for info in infos:
        counts[int(info["left_color"])] += 1
        counts[int(info["right_color"])] += 1
    return counts


def _validate_connector_counts(g: np.ndarray, infos: list[dict]) -> Counter:
    counts = _connector_counts(infos)
    singles = {color for color, count in counts.items() if int(count) == 1}
    assert len(singles) == 2 and GREY in singles, "exactly two connector colors are singletons, one of them gray"
    assert all(int(count) in (1, 2) for count in counts.values()), "all non-singleton connector colors must appear as pairs"
    for color, count in counts.items():
        assert int(np.count_nonzero(g == int(color))) == int(count), "connector colors may appear only on connector pixels"
    return counts


def _chain_order(infos: list[dict], counts: Counter) -> list[int]:
    used: set[int] = set()
    order: list[int] = []
    current = GREY
    for _ in range(len(infos)):
        candidates = [
            i
            for i, info in enumerate(infos)
            if i not in used and int(info["left_color"]) == int(current)
        ]
        assert len(candidates) == 1, "gray-start connector chain must be unique"
        idx = int(candidates[0])
        order.append(idx)
        used.add(idx)
        current = int(infos[idx]["right_color"])
        if len(order) < len(infos):
            assert int(counts[current]) == 2, "only the final right connector may be a non-gray singleton"
    assert int(counts[current]) == 1 and int(current) != GREY, "the chain must end at the other singleton connector"
    return order


def _shift_group(group: np.ndarray, bg: int, dr: int) -> np.ndarray:
    shifted = np.full_like(group, int(bg))
    coords = np.argwhere(group != int(bg))
    rows = coords[:, 0] + int(dr)
    assert bool(np.all((0 <= rows) & (rows < group.shape[0]))), "vertical shift would leave the strip"
    shifted[rows, coords[:, 1]] = group[coords[:, 0], coords[:, 1]]
    return shifted


def _parse(grid: np.ndarray) -> tuple[int, list[np.ndarray], list[dict], Counter, list[int]]:
    g = np.asarray(grid, dtype=int)
    assert g.ndim == 2 and g.size > 0, "expected a non-empty 2D grid"
    assert max(g.shape) <= 30, "ARC grid dimensions must not exceed 30"
    bg = _background(g)
    groups = _column_groups(g, bg)
    infos = [_group_info(group, bg, i) for i, group in enumerate(groups)]
    counts = _validate_connector_counts(g, infos)
    order = _chain_order(infos, counts)
    return bg, groups, infos, counts, order


def main(grid: np.ndarray) -> np.ndarray:
    """
    Reorder connector-colored path groups and slide them into one gray-start domino chain.

    Input structure:
    - A horizontal strip has path groups occupying continuous non-background column runs, with all-background separator columns between groups.
    - Each group is one non-branching path: its left-edge and right-edge degree-1 endpoint cells are recolored as connector pixels, and all other path cells share one body color.
    - Exactly two connector colors occur once in the whole grid, one of them gray; every other connector color occurs exactly twice.

    Rule:
    - Delete separator columns and arrange the groups as directed dominoes from left connector color to right connector color.
    - The group whose left connector is gray is placed first and keeps its input row position.
    - Each later group is translated vertically, without rotation, mirroring, or recoloring, so its left connector has the same row as the previous placed group's right connector, then it is pasted immediately to the right.

    Key insight:
    - The input left-to-right order is a distractor; the unique chain is determined by paired connector colors, while connector row equality propagates the required vertical shifts.
    """
    bg, groups, infos, _counts, order = _parse(np.asarray(grid, dtype=int))
    h = int(np.asarray(grid).shape[0])
    total_w = sum(int(groups[i].shape[1]) for i in order)
    assert total_w <= 30, "packed output width must fit ARC bounds"

    out = np.full((h, total_w), int(bg), dtype=int)
    c = 0

    first_i = int(order[0])
    first = groups[first_i]
    out[:, : first.shape[1]] = first
    prev_row = int(infos[first_i]["right_row"])
    prev_color = int(infos[first_i]["right_color"])
    c += int(first.shape[1])

    for idx in order[1:]:
        idx = int(idx)
        info = infos[idx]
        assert int(info["left_color"]) == prev_color, "adjacent connector colors must match"
        dr = int(prev_row) - int(info["left_row"])
        shifted = _shift_group(groups[idx], bg, dr)
        assert int(shifted[prev_row, 0]) == int(prev_color)
        out[:, c : c + shifted.shape[1]] = shifted
        prev_row = int(info["right_row"]) + int(dr)
        prev_color = int(info["right_color"])
        c += int(shifted.shape[1])

    return out


# --------------- generator ---------------


def _path_is_valid(cells: list[Coord], h: int, w: int, left_row: int, right_row: int) -> bool:
    cell_set = {tuple(map(int, p)) for p in cells}
    if len(cell_set) != len(cells):
        return False
    if (int(left_row), 0) not in cell_set or (int(right_row), w - 1) not in cell_set:
        return False
    if any(r < 0 or r >= h or c < 0 or c >= w for r, c in cell_set):
        return False
    if {c for _r, c in cell_set} != set(range(w)):
        return False

    mask = np.zeros((h, w), dtype=bool)
    for r, c in cell_set:
        mask[r, c] = True
    if len(_components(mask)) != 1:
        return False

    degree_one = {p for p in cell_set if _degree(cell_set, p[0], p[1]) == 1}
    if degree_one != {(int(left_row), 0), (int(right_row), w - 1)}:
        return False
    return all(_degree(cell_set, r, c) in (1, 2) for r, c in cell_set)


def _turn_column_options(w: int, k: int) -> list[tuple[int, ...]]:
    return [
        combo
        for combo in itertools.combinations(range(1, int(w) - 1), int(k))
        if all(combo[i + 1] - combo[i] >= 2 for i in range(len(combo) - 1))
    ]


def _make_path_cells(h: int, w: int, left_row: int, right_row: int) -> list[Coord] | None:
    for _ in range(120):
        max_turns = min(3, max(0, (int(w) - 1) // 2))
        turn_counts = list(range(max_turns + 1))
        if int(left_row) != int(right_row):
            turn_counts = [k for k in turn_counts if k >= 1]
        else:
            turn_counts = [k for k in turn_counts if k == 0 or k >= 2]
        random.shuffle(turn_counts)

        for k in turn_counts:
            options = _turn_column_options(w, k)
            if not options:
                continue
            turn_cols = random.choice(options)
            current = int(left_row)
            targets: list[int] = []
            ok = True
            for i in range(k):
                if i == k - 1:
                    target = int(right_row)
                else:
                    choices = [r for r in range(h) if r != current]
                    target = random.choice(choices)
                if target == current:
                    ok = False
                    break
                targets.append(int(target))
                current = int(target)
            if not ok:
                continue

            move_to = dict(zip(turn_cols, targets))
            cells: list[Coord] = [(int(left_row), 0)]
            current = int(left_row)
            for c in range(1, int(w)):
                cells.append((current, c))
                if c in move_to:
                    target = int(move_to[c])
                    step = 1 if target > current else -1
                    for r in range(current + step, target + step, step):
                        cells.append((int(r), int(c)))
                    current = target
            if current != int(right_row):
                continue
            if _path_is_valid(cells, h, w, left_row, right_row):
                return cells
    return None


def _make_group(
    h: int,
    w: int,
    left_row: int,
    right_row: int,
    body: int,
    left_connector: int,
    right_connector: int,
) -> np.ndarray | None:
    cells = _make_path_cells(h, w, left_row, right_row)
    if cells is None:
        return None
    group = np.full((h, w), BG, dtype=int)
    for r, c in cells:
        group[int(r), int(c)] = int(body)
    group[int(left_row), 0] = int(left_connector)
    group[int(right_row), int(w) - 1] = int(right_connector)
    try:
        _group_info(group, BG, 0)
    except AssertionError:
        return None
    return group


def _concat_groups(groups: list[np.ndarray]) -> np.ndarray:
    h = int(groups[0].shape[0])
    width = sum(int(group.shape[1]) for group in groups)
    out = np.full((h, width), BG, dtype=int)
    c = 0
    for group in groups:
        out[:, c : c + group.shape[1]] = group
        c += int(group.shape[1])
    return out


def _valid_shifts(group: np.ndarray) -> list[int]:
    rows = np.argwhere(group != BG)[:, 0]
    return [
        int(dr)
        for dr in SHIFT_RANGE
        if int(rows.min()) + int(dr) >= 0 and int(rows.max()) + int(dr) < group.shape[0]
    ]


def _choose_input_shifts(groups: list[np.ndarray]) -> list[int] | None:
    shifts = [0] * len(groups)
    forceable: list[tuple[int, list[int]]] = []
    for i, group in enumerate(groups[1:], start=1):
        valid = _valid_shifts(group)
        if not valid:
            return None
        nonzero = [dr for dr in valid if dr != 0]
        if nonzero:
            forceable.append((i, nonzero))
        shifts[i] = random.choice(nonzero if nonzero and random.random() < 0.8 else valid)

    if not any(dr != 0 for dr in shifts[1:]):
        if not forceable:
            return None
        i, choices = random.choice(forceable)
        shifts[i] = random.choice(choices)
    return shifts


def _input_order(n: int) -> list[int] | None:
    order = list(range(int(n)))
    for _ in range(40):
        random.shuffle(order)
        if order[0] != 0:
            return list(order)
    return None


def _separator_widths(widths: list[int]) -> list[int] | None:
    if len(widths) == 1:
        return []
    for _ in range(50):
        seps = [random.choice(SEPARATOR_CHOICES) for _ in range(len(widths) - 1)]
        if sum(widths) + sum(seps) <= 30:
            return seps
    return None


def _render_input(groups: list[np.ndarray], order: list[int], seps: list[int]) -> np.ndarray:
    h = int(groups[0].shape[0])
    width = sum(int(groups[i].shape[1]) for i in order) + sum(int(s) for s in seps)
    grid = np.full((h, width), BG, dtype=int)
    c = 0
    for pos, idx in enumerate(order):
        group = groups[int(idx)]
        grid[:, c : c + group.shape[1]] = group
        c += int(group.shape[1])
        if pos < len(seps):
            c += int(seps[pos])
    return grid


def _pack_in_input_order(inp: np.ndarray) -> np.ndarray | None:
    try:
        bg = _background(inp)
        groups = _column_groups(inp, bg)
        infos = [_group_info(group, bg, i) for i, group in enumerate(groups)]
        out = np.full((inp.shape[0], sum(group.shape[1] for group in groups)), bg, dtype=int)
        c = 0
        first = groups[0]
        out[:, : first.shape[1]] = first
        prev_row = int(infos[0]["right_row"])
        c += int(first.shape[1])
        for group, info in zip(groups[1:], infos[1:]):
            shifted = _shift_group(group, bg, prev_row - int(info["left_row"]))
            out[:, c : c + group.shape[1]] = shifted
            prev_row = int(info["right_row"]) + prev_row - int(info["left_row"])
            c += int(group.shape[1])
        return out
    except AssertionError:
        return None


def _semantic_valid(inp: np.ndarray, expected: np.ndarray) -> bool:
    try:
        bg, _groups, infos, counts, order = _parse(inp)
        actual = main(inp)
    except Exception:
        return False
    if bg != BG or not np.array_equal(actual, expected):
        return False
    if order == list(range(len(order))) or int(order[0]) == 0:
        return False
    if len({int(info["left_row"]) for info in infos} | {int(info["right_row"]) for info in infos}) < 2:
        return False
    if len([color for color, count in counts.items() if int(count) == 2]) != len(infos) - 1:
        return False

    prev_row = int(infos[order[0]]["right_row"])
    needs_slide = False
    for idx in order[1:]:
        info = infos[int(idx)]
        dr = prev_row - int(info["left_row"])
        needs_slide = needs_slide or dr != 0
        prev_row = int(info["right_row"]) + int(dr)
    if not needs_slide:
        return False

    packed = _pack_in_input_order(inp)
    if packed is not None and np.array_equal(packed, expected):
        return False
    return True


def _gen_attempt() -> np.ndarray | None:
    h = random.randint(*HEIGHT_RANGE)
    n = random.choice(GROUP_COUNT_CHOICES)
    connector_tail = random.sample([c for c in ARC_COLORS if c not in (BG, GREY)], n)
    connectors = [GREY] + connector_tail
    body_palette = [c for c in ARC_COLORS if c not in connectors and c != BG]
    if not body_palette:
        return None

    widths = [random.choice(WIDTH_CHOICES) for _ in range(n)]
    if sum(widths) > 30:
        return None

    body_colors = [random.choice(body_palette) for _ in range(n)]
    rows = [random.randrange(h)]
    for _ in range(n):
        rows.append(random.randrange(h))
    if all(rows[i] == rows[i + 1] for i in range(n)):
        return None

    chain_groups: list[np.ndarray] = []
    for i in range(n):
        group = _make_group(
            h,
            widths[i],
            rows[i],
            rows[i + 1],
            body_colors[i],
            connectors[i],
            connectors[i + 1],
        )
        if group is None:
            return None
        chain_groups.append(group)

    expected = _concat_groups(chain_groups)
    if max(expected.shape) > 30:
        return None

    shifts = _choose_input_shifts(chain_groups)
    if shifts is None:
        return None
    input_groups = [_shift_group(group, BG, dr) for group, dr in zip(chain_groups, shifts)]

    order = _input_order(n)
    if order is None:
        return None
    seps = _separator_widths([int(input_groups[i].shape[1]) for i in order])
    if seps is None:
        return None
    inp = _render_input(input_groups, order, seps)
    if max(inp.shape) > 30:
        return None

    return inp if _semantic_valid(inp, expected) else None


def generate_input() -> np.ndarray:
    for _ in range(MAX_GENERATION_ATTEMPTS):
        grid = _gen_attempt()
        if grid is not None:
            return grid
    raise RuntimeError("failed to generate a valid connector-color domino chain input")
