import random
from typing import Iterable

import numpy as np


BG = 5
PALETTE = [0, 1, 2, 3, 4, 6, 7, 8, 9]
GEN_ATTEMPTS = 400
PAIR_ATTEMPTS = 200
CORE_ATTEMPTS = 200
SHIFT_OPTIONS = [
    (-2, 0),
    (-1, 0),
    (0, -2),
    (0, -1),
    (0, 1),
    (0, 2),
    (1, 0),
    (2, 0),
    (-1, -1),
    (-1, 1),
    (1, -1),
    (1, 1),
]


def _get_bg(grid: np.ndarray) -> int:
    values, counts = np.unique(grid, return_counts=True)
    return int(values[np.argmax(counts)])


def _neighbors4(r: int, c: int, h: int, w: int) -> Iterable[tuple[int, int]]:
    if r > 0:
        yield r - 1, c
    if r + 1 < h:
        yield r + 1, c
    if c > 0:
        yield r, c - 1
    if c + 1 < w:
        yield r, c + 1


def _components(mask: np.ndarray) -> list[np.ndarray]:
    h, w = mask.shape
    seen = np.zeros((h, w), dtype=bool)
    parts: list[np.ndarray] = []
    for r in range(h):
        for c in range(w):
            if not bool(mask[r, c]) or bool(seen[r, c]):
                continue
            stack = [(r, c)]
            seen[r, c] = True
            cells: list[tuple[int, int]] = []
            while stack:
                rr, cc = stack.pop()
                cells.append((rr, cc))
                for nr, nc in _neighbors4(rr, cc, h, w):
                    if bool(mask[nr, nc]) and not bool(seen[nr, nc]):
                        seen[nr, nc] = True
                        stack.append((nr, nc))
            parts.append(np.array(cells, dtype=int))
    return parts


def _bbox(coords: np.ndarray) -> tuple[int, int, int, int]:
    r0 = int(np.min(coords[:, 0]))
    c0 = int(np.min(coords[:, 1]))
    r1 = int(np.max(coords[:, 0]))
    c1 = int(np.max(coords[:, 1]))
    return r0, c0, r1, c1


def _panel_layout(grid: np.ndarray, bg: int) -> tuple[int, list[tuple[int, int]]]:
    mask = grid != bg
    comps = _components(mask)
    panel_count = len(comps)
    assert panel_count >= 2 and panel_count % 2 == 0, "expected an even number of panels"

    h, w = grid.shape
    assert (w - (panel_count - 1)) % panel_count == 0, "expected one separator column between equal-width panels"
    panel_w = (w - (panel_count - 1)) // panel_count
    assert panel_w > 0

    slices: list[tuple[int, int]] = []
    for idx in range(panel_count):
        c0 = idx * (panel_w + 1)
        c1 = c0 + panel_w
        assert c1 <= w
        slices.append((c0, c1))
        if idx + 1 < panel_count:
            assert bool(np.all(grid[:, c1] == bg)), "separator columns must use the background color"
    assert slices[-1][1] == w
    return panel_w, slices


def _extract_panel_object(panel: np.ndarray, bg: int) -> dict[str, object]:
    colors = [int(v) for v in np.unique(panel) if int(v) != bg]
    assert len(colors) == 1, "each panel must contain exactly one foreground color"

    mask = panel != bg
    comps = _components(mask)
    assert len(comps) == 1, "each panel must contain exactly one 4-connected object"
    coords = comps[0]
    color = colors[0]
    assert bool(np.all(panel[coords[:, 0], coords[:, 1]] == color))

    r0, c0, _r1, _c1 = _bbox(coords)
    norm = frozenset((int(r - r0), int(c - c0)) for r, c in coords)
    return {
        "color": color,
        "coords": coords,
        "origin": (r0, c0),
        "norm": norm,
    }


def _render_pair(
    left_obj: dict[str, object], right_obj: dict[str, object], panel_h: int, panel_w: int, bg: int
) -> np.ndarray:
    common = set(left_obj["norm"]) & set(right_obj["norm"])
    assert common, "normalized masks in each pair must overlap"

    out = np.full((panel_h, panel_w), bg, dtype=int)
    left_r0, left_c0 = left_obj["origin"]
    right_r0, right_c0 = right_obj["origin"]
    left_color = int(left_obj["color"])
    right_color = int(right_obj["color"])

    for dr, dc in common:
        rr = int(left_r0 + dr)
        cc = int(left_c0 + dc)
        assert 0 <= rr < panel_h and 0 <= cc < panel_w
        out[rr, cc] = left_color

    for dr, dc in common:
        rr = int(right_r0 + dr)
        cc = int(right_c0 + dc)
        assert 0 <= rr < panel_h and 0 <= cc < panel_w
        if int(out[rr, cc]) == bg:
            out[rr, cc] = right_color

    return out


def _transform(grid: np.ndarray) -> np.ndarray:
    g = np.asarray(grid, dtype=int)
    assert g.ndim == 2, "expected a 2D grid"
    panel_h, _w = g.shape
    bg = _get_bg(g)

    panel_w, slices = _panel_layout(g, bg)
    objs = [_extract_panel_object(g[:, c0:c1], bg) for c0, c1 in slices]

    pair_panels = [
        _render_pair(objs[idx], objs[idx + 1], panel_h, panel_w, bg)
        for idx in range(0, len(objs), 2)
    ]

    out_w = len(pair_panels) * panel_w + max(0, len(pair_panels) - 1)
    out = np.full((panel_h, out_w), bg, dtype=int)
    for idx, pair_panel in enumerate(pair_panels):
        c0 = idx * (panel_w + 1)
        out[:, c0 : c0 + panel_w] = pair_panel
    return out


def main(grid: np.ndarray) -> np.ndarray:
    """
    For each horizontal panel pair, intersect the two normalized foreground masks
    and project that shared shape back into both original panel positions.

    Left projected pixels use the left color, right-only projected pixels use the
    right color, and left color wins where the two projected copies overlap.
    """
    return _transform(grid)


def _norm_cells(cells: set[tuple[int, int]]) -> set[tuple[int, int]]:
    r0 = min(r for r, _c in cells)
    c0 = min(c for _r, c in cells)
    return {(r - r0, c - c0) for r, c in cells}


def _shape_dims(cells: set[tuple[int, int]]) -> tuple[int, int]:
    return max(r for r, _c in cells) + 1, max(c for _r, c in cells) + 1


def _grow_shape(
    base: set[tuple[int, int]],
    target_area: int,
    max_h: int,
    max_w: int,
    forbidden: set[tuple[int, int]] | None = None,
) -> set[tuple[int, int]] | None:
    cells = set(base)
    blocked = set() if forbidden is None else (set(forbidden) - cells)

    while len(cells) < target_area:
        frontier: list[tuple[int, int]] = []
        seen = set()
        for r, c in cells:
            for nr, nc in ((r - 1, c), (r + 1, c), (r, c - 1), (r, c + 1)):
                if not (0 <= nr < max_h and 0 <= nc < max_w):
                    continue
                if (nr, nc) in cells or (nr, nc) in blocked or (nr, nc) in seen:
                    continue
                seen.add((nr, nc))
                frontier.append((nr, nc))
        if not frontier:
            return None
        weights = [
            1
            + sum((nr, nc) in cells for nr, nc in ((r - 1, c), (r + 1, c), (r, c - 1), (r, c + 1)))
            for r, c in frontier
        ]
        cells.add(random.choices(frontier, weights=weights, k=1)[0])

    return _norm_cells(cells)


def _candidate_shifts(cells: set[tuple[int, int]]) -> list[tuple[int, int]]:
    candidates: list[tuple[int, int]] = []
    for dr, dc in SHIFT_OPTIONS:
        shifted = {(r + dr, c + dc) for r, c in cells}
        if cells & shifted:
            candidates.append((dr, dc))
    return candidates


def _random_core(max_h: int, max_w: int) -> tuple[set[tuple[int, int]], list[tuple[int, int]]]:
    for _ in range(CORE_ATTEMPTS):
        h = random.randint(2, max_h)
        w = random.randint(2, max_w)
        seed = {(0, 0), (0, min(1, w - 1)), (min(1, h - 1), 0)}
        max_area = min(h * w, 8)
        min_area = min(max_area, max(4, (h * w + 1) // 2))
        area = random.randint(max(len(seed), min_area), max_area)
        core = _grow_shape(seed, area, h, w)
        if core is None:
            continue
        shifts = _candidate_shifts(core)
        if shifts:
            return core, shifts
    raise RuntimeError("failed to build a shared core")


def _place_shape(panel: np.ndarray, cells: set[tuple[int, int]], origin: tuple[int, int], color: int) -> None:
    r0, c0 = origin
    for r, c in cells:
        panel[r0 + r, c0 + c] = color


def _generate_pair(panel_h: int, panel_w: int, left_color: int, right_color: int) -> tuple[np.ndarray, np.ndarray]:
    for _ in range(PAIR_ATTEMPTS):
        core_max_h = min(panel_h - 1, 5)
        core_max_w = min(panel_w - 1, 5)
        if core_max_h < 2 or core_max_w < 2:
            break

        shared, shifts = _random_core(core_max_h, core_max_w)
        shift = random.choice(shifts)
        shared_h, shared_w = _shape_dims(shared)

        left_limit_h = min(panel_h - 1, shared_h + random.randint(0, 2))
        left_limit_w = min(panel_w - 1, shared_w + random.randint(0, 2))
        right_limit_h = min(panel_h - 1, shared_h + random.randint(0, 2))
        right_limit_w = min(panel_w - 1, shared_w + random.randint(0, 2))

        left_room = left_limit_h * left_limit_w - len(shared)
        right_room = right_limit_h * right_limit_w - len(shared)
        if left_room < 0 or right_room < 0:
            continue

        left_extra = random.randint(0, min(4, left_room))
        right_extra = random.randint(0, min(4, right_room))
        if left_extra == 0 and right_extra == 0 and random.random() < 0.7:
            continue

        left_shape = _grow_shape(shared, len(shared) + left_extra, left_limit_h, left_limit_w)
        if left_shape is None:
            continue
        right_shape = _grow_shape(
            shared,
            len(shared) + right_extra,
            right_limit_h,
            right_limit_w,
            forbidden=left_shape - shared,
        )
        if right_shape is None or (left_shape & right_shape) != shared:
            continue

        left_h, left_w = _shape_dims(left_shape)
        right_h, right_w = _shape_dims(right_shape)

        feasible: list[tuple[int, int]] = []
        dr, dc = shift
        for left_r0 in range(panel_h - left_h + 1):
            right_r0 = left_r0 + dr
            if not (0 <= right_r0 <= panel_h - right_h):
                continue
            for left_c0 in range(panel_w - left_w + 1):
                right_c0 = left_c0 + dc
                if 0 <= right_c0 <= panel_w - right_w:
                    feasible.append((left_r0, left_c0))
        if not feasible:
            continue

        left_origin = random.choice(feasible)
        right_origin = (left_origin[0] + dr, left_origin[1] + dc)

        left_copy = {(left_origin[0] + r, left_origin[1] + c) for r, c in shared}
        right_copy = {(right_origin[0] + r, right_origin[1] + c) for r, c in shared}
        if not (left_copy & right_copy):
            continue
        if not (left_copy - right_copy) or not (right_copy - left_copy):
            continue

        left_panel = np.full((panel_h, panel_w), BG, dtype=int)
        right_panel = np.full((panel_h, panel_w), BG, dtype=int)
        _place_shape(left_panel, left_shape, left_origin, left_color)
        _place_shape(right_panel, right_shape, right_origin, right_color)

        left_obj = {"color": left_color, "origin": left_origin, "norm": frozenset(shared)}
        right_obj = {"color": right_color, "origin": right_origin, "norm": frozenset(shared)}
        pair_out = _render_pair(left_obj, right_obj, panel_h, panel_w, BG)
        visible = {int(v) for v in np.unique(pair_out) if int(v) != BG}
        if visible != {left_color, right_color}:
            continue
        return left_panel, right_panel

    raise RuntimeError("failed to generate a valid panel pair")


def _expected_from_panels(panels: list[np.ndarray]) -> np.ndarray:
    pair_count = len(panels) // 2
    panel_h, panel_w = panels[0].shape
    out = np.full((panel_h, pair_count * panel_w + max(0, pair_count - 1)), BG, dtype=int)
    for idx in range(pair_count):
        left_obj = _extract_panel_object(panels[2 * idx], BG)
        right_obj = _extract_panel_object(panels[2 * idx + 1], BG)
        pair_out = _render_pair(left_obj, right_obj, panel_h, panel_w, BG)
        c0 = idx * (panel_w + 1)
        out[:, c0 : c0 + panel_w] = pair_out
    return out


def _gen_attempt() -> np.ndarray | None:
    pair_count = random.choice([1, 2, 3])
    if pair_count == 1:
        panel_w = random.randint(6, 10)
    elif pair_count == 2:
        panel_w = random.randint(5, 6)
    else:
        panel_w = 4
    panel_h = random.randint(6, 10)

    colors = random.sample(PALETTE, 2 * pair_count)
    panels: list[np.ndarray] = []
    for idx in range(pair_count):
        try:
            left_panel, right_panel = _generate_pair(
                panel_h, panel_w, colors[2 * idx], colors[2 * idx + 1]
            )
        except RuntimeError:
            return None
        panels.extend([left_panel, right_panel])

    grid_w = (2 * pair_count) * panel_w + (2 * pair_count - 1)
    grid = np.full((panel_h, grid_w), BG, dtype=int)
    for idx, panel in enumerate(panels):
        c0 = idx * (panel_w + 1)
        grid[:, c0 : c0 + panel_w] = panel

    if _get_bg(grid) != BG:
        return None

    expected = _expected_from_panels(panels)
    actual = _transform(grid)
    if not np.array_equal(actual, expected):
        return None
    return grid


def generate_input() -> np.ndarray:
    for _ in range(GEN_ATTEMPTS):
        grid = _gen_attempt()
        if grid is not None:
            return grid
    raise RuntimeError(f"failed to generate a valid input after {GEN_ATTEMPTS} attempts")
