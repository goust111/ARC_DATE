from __future__ import annotations

import random
from dataclasses import dataclass

import numpy as np


DIRS = {
    "up": (-1, 0),
    "down": (1, 0),
    "left": (0, -1),
    "right": (0, 1),
}
OPPOSITE = {"up": "down", "down": "up", "left": "right", "right": "left"}
PALETTE = list(range(10))
GEN_ATTEMPTS = 400


@dataclass(frozen=True)
class TubeSpec:
    r0: int
    c0: int
    inner_len: int
    open_dir: str
    wall_color: int
    liquid_color: int
    liquid_len: int
    spill_space: int


def _get_bg(grid: np.ndarray) -> int:
    values = np.asarray(grid, dtype=int).ravel()
    counts = np.bincount(values, minlength=max(10, int(values.max(initial=0)) + 1))
    return int(np.argmax(counts))


def _components(mask: np.ndarray) -> list[list[tuple[int, int]]]:
    h, w = mask.shape
    seen = np.zeros_like(mask, dtype=bool)
    comps: list[list[tuple[int, int]]] = []
    for r in range(h):
        for c in range(w):
            if not mask[r, c] or seen[r, c]:
                continue
            stack = [(r, c)]
            seen[r, c] = True
            coords: list[tuple[int, int]] = []
            while stack:
                rr, cc = stack.pop()
                coords.append((rr, cc))
                for dr, dc in DIRS.values():
                    nr, nc = rr + dr, cc + dc
                    if 0 <= nr < h and 0 <= nc < w and mask[nr, nc] and not seen[nr, nc]:
                        seen[nr, nc] = True
                        stack.append((nr, nc))
            comps.append(coords)
    return comps


def _bbox(coords: list[tuple[int, int]]) -> tuple[int, int, int, int]:
    rows = [r for r, _ in coords]
    cols = [c for _, c in coords]
    return min(rows), min(cols), max(rows), max(cols)


def _spec_outer_shape(spec: TubeSpec) -> tuple[int, int]:
    if spec.open_dir in ("up", "down"):
        return spec.inner_len + 2, 3
    return 3, spec.inner_len + 2


def _tube_bbox(spec: TubeSpec) -> tuple[int, int, int, int]:
    outer_h, outer_w = _spec_outer_shape(spec)
    return spec.r0, spec.c0, spec.r0 + outer_h - 1, spec.c0 + outer_w - 1


def _end_center(spec: TubeSpec, side: str | None = None) -> tuple[int, int]:
    if side is None:
        side = spec.open_dir
    r0, c0, r1, c1 = _tube_bbox(spec)
    if side == "up":
        return r0, c0 + 1
    if side == "down":
        return r1, c0 + 1
    if side == "left":
        return r0 + 1, c0
    return r0 + 1, c1


def _control_coord(spec: TubeSpec) -> tuple[int, int]:
    er, ec = _end_center(spec, spec.open_dir)
    dr, dc = DIRS[spec.open_dir]
    return er + dr * (spec.spill_space + 1), ec + dc * (spec.spill_space + 1)


def _outside_cells(spec: TubeSpec) -> list[tuple[int, int]]:
    er, ec = _end_center(spec, spec.open_dir)
    dr, dc = DIRS[spec.open_dir]
    return [(er + dr * step, ec + dc * step) for step in range(1, spec.spill_space + 1)]


def _cavity_cells(spec: TubeSpec, from_open: bool = True) -> list[tuple[int, int]]:
    r0, c0, r1, c1 = _tube_bbox(spec)
    if spec.open_dir == "up":
        cells = [(r, c0 + 1) for r in range(r0 + 1, r1)]
    elif spec.open_dir == "down":
        cells = [(r, c0 + 1) for r in range(r1 - 1, r0, -1)]
    elif spec.open_dir == "left":
        cells = [(r0 + 1, c) for c in range(c0 + 1, c1)]
    else:
        cells = [(r0 + 1, c) for c in range(c1 - 1, c0, -1)]
    if from_open:
        return cells
    return list(reversed(cells))


def _render_input(shape: tuple[int, int], bg: int, control_color: int, tubes: list[TubeSpec]) -> np.ndarray:
    grid = np.full(shape, bg, dtype=int)
    for spec in tubes:
        r0, c0, r1, c1 = _tube_bbox(spec)
        grid[r0 : r1 + 1, c0 : c1 + 1] = spec.wall_color
        for rr, cc in _cavity_cells(spec, from_open=True):
            grid[rr, cc] = bg
        cavity = _cavity_cells(spec, from_open=True)
        for rr, cc in cavity[-spec.liquid_len :]:
            grid[rr, cc] = spec.liquid_color
        cr, cc = _control_coord(spec)
        grid[cr, cc] = control_color
    return grid


def _render_output(inp: np.ndarray, bg: int, tubes: list[TubeSpec]) -> np.ndarray:
    out = inp.copy()
    for spec in tubes:
        end_r, end_c = _end_center(spec, spec.open_dir)
        out[end_r, end_c] = bg

        cavity = _cavity_cells(spec, from_open=True)
        poured = min(spec.liquid_len, spec.spill_space)
        remain = spec.liquid_len - poured
        for rr, cc in cavity:
            out[rr, cc] = bg
        for rr, cc in cavity[-remain:]:
            out[rr, cc] = spec.liquid_color
        for rr, cc in _outside_cells(spec)[:poured]:
            out[rr, cc] = spec.liquid_color
    return out


def _scan_axis(grid: np.ndarray, start: tuple[int, int], direction: str, bg: int) -> tuple[int, tuple[int, int] | None]:
    h, w = grid.shape
    dr, dc = DIRS[direction]
    rr, cc = start
    space = 0
    while True:
        rr += dr
        cc += dc
        if not (0 <= rr < h and 0 <= cc < w):
            return space, None
        if int(grid[rr, cc]) == bg:
            space += 1
            continue
        return space, (rr, cc)


def _parse_tube_component(
    grid: np.ndarray, color: int, coords: list[tuple[int, int]]
) -> dict[str, object] | None:
    r0, c0, r1, c1 = _bbox(coords)
    h = r1 - r0 + 1
    w = c1 - c0 + 1
    if min(h, w) != 3 or max(h, w) < 5:
        return None

    window = grid[r0 : r1 + 1, c0 : c1 + 1]
    if h == 3:
        if not np.all(window[0, :] == color):
            return None
        if not np.all(window[2, :] == color):
            return None
        if not np.all(window[:, 0] == color):
            return None
        if not np.all(window[:, -1] == color):
            return None
        if np.any(window[1, 1:-1] == color):
            return None
        axis = "horizontal"
        cavity = [(r0 + 1, c) for c in range(c0 + 1, c1)]
        ends = {"left": (r0 + 1, c0), "right": (r0 + 1, c1)}
    else:
        if not np.all(window[:, 0] == color):
            return None
        if not np.all(window[:, 2] == color):
            return None
        if not np.all(window[0, :] == color):
            return None
        if not np.all(window[-1, :] == color):
            return None
        if np.any(window[1:-1, 1] == color):
            return None
        axis = "vertical"
        cavity = [(r, c0 + 1) for r in range(r0 + 1, r1)]
        ends = {"up": (r0, c0 + 1), "down": (r1, c0 + 1)}

    wall_cells = {(r, c) for r, c in coords}
    return {
        "color": color,
        "axis": axis,
        "bbox": (r0, c0, r1, c1),
        "cavity": cavity,
        "ends": ends,
        "wall_cells": wall_cells,
    }


def main(grid: np.ndarray) -> np.ndarray:
    """
    Open the tube end pointed to by each control pixel and pour out exactly the conserved liquid volume.

    Input structure:
    - The grid has a dominant background plus several straight thin tubes: each tube is a one-cell-thick monochrome wall around a one-cell-wide cavity.
    - Inside each cavity, one contiguous liquid segment of a single color fills length L from the closed bottom end, while the rest of the cavity is background.
    - Exactly one external single-pixel control lies on the axis extension of one tube end, separated from that end by only background cells.

    Rule:
    - Remove the wall cell at the tube end nearest the control, turning that end into an opening.
    - Let the liquid pour straight outward along the tube axis across background cells until just before the control pixel.
    - If m cells are poured outside, remove exactly m cells from the in-tube liquid segment, keeping the remaining L-m cells attached to the opposite closed end.
    - Keep every control pixel and every unrelated structure unchanged.

    Key insight:
    - The control pixel does not enter the tube; it chooses which end opens and caps the outflow distance, so the outside spill length and the inside shortening must match exactly.
    """
    g = np.asarray(grid, dtype=int)
    h, w = g.shape
    bg = _get_bg(g)

    comp_area: dict[tuple[int, int], int] = {}
    tubes: list[dict[str, object]] = []
    for color in sorted(set(map(int, np.unique(g))) - {bg}):
        for coords in _components(g == color):
            area = len(coords)
            for rc in coords:
                comp_area[rc] = area
            tube = _parse_tube_component(g, color, coords)
            if tube is not None:
                tubes.append(tube)

    assert tubes, "expected at least one tube"

    out = g.copy()
    for tube in tubes:
        ends = tube["ends"]
        candidates: list[tuple[str, int, tuple[int, int]]] = []
        for side, end_center in ends.items():
            space, hit = _scan_axis(g, end_center, side, bg)
            if hit is None or space < 1:
                continue
            if hit in tube["wall_cells"]:
                continue
            if comp_area.get(hit, 0) != 1:
                continue
            candidates.append((side, space, hit))
        assert len(candidates) == 1, "each tube must have exactly one aligned control pixel"
        open_dir, spill_space, _control = candidates[0]

        cavity = list(tube["cavity"])
        if open_dir in ("down", "right"):
            cavity = list(reversed(cavity))

        liquid_cells = [(r, c) for r, c in cavity if int(g[r, c]) != bg]
        assert liquid_cells, "tube cavity must contain liquid"
        liquid_colors = {int(g[r, c]) for r, c in liquid_cells}
        assert len(liquid_colors) == 1, "tube liquid must be monochrome"
        liquid_color = liquid_colors.pop()

        first_liquid = next(i for i, (r, c) in enumerate(cavity) if int(g[r, c]) != bg)
        assert all(int(g[r, c]) == bg for r, c in cavity[:first_liquid])
        assert all(int(g[r, c]) == liquid_color for r, c in cavity[first_liquid:])
        liquid_len = len(cavity) - first_liquid

        poured = min(liquid_len, spill_space)
        remain = liquid_len - poured

        end_r, end_c = ends[open_dir]
        out[end_r, end_c] = bg
        for rr, cc in cavity:
            out[rr, cc] = bg
        for rr, cc in cavity[-remain:]:
            out[rr, cc] = liquid_color

        dr, dc = DIRS[open_dir]
        for step in range(1, poured + 1):
            rr = end_r + dr * step
            cc = end_c + dc * step
            assert 0 <= rr < h and 0 <= cc < w
            assert int(g[rr, cc]) == bg
            out[rr, cc] = liquid_color

    return out


def _choose_lengths(branch: str, spill_space: int) -> tuple[int, int]:
    if branch == "partial":
        inner_len = random.randint(max(4, spill_space + 1), 8)
        liquid_len = random.randint(spill_space + 1, inner_len)
        return inner_len, liquid_len
    inner_len = random.randint(4, 8)
    liquid_len = random.randint(1, min(inner_len, spill_space))
    return inner_len, liquid_len


def _expanded_bbox(spec: TubeSpec, shape: tuple[int, int]) -> tuple[int, int, int, int]:
    h, w = shape
    r0, c0, r1, c1 = _tube_bbox(spec)
    cr, cc = _control_coord(spec)
    rows = [r0, r1, cr]
    cols = [c0, c1, cc]
    er0 = max(0, min(rows) - 1)
    ec0 = max(0, min(cols) - 1)
    er1 = min(h - 1, max(rows) + 1)
    ec1 = min(w - 1, max(cols) + 1)
    return er0, ec0, er1, ec1


def _boxes_overlap(a: tuple[int, int, int, int], b: tuple[int, int, int, int]) -> bool:
    return not (a[2] < b[0] or b[2] < a[0] or a[3] < b[1] or b[3] < a[1])


def _random_spec(
    shape: tuple[int, int],
    open_dir: str,
    branch: str,
    wall_color: int,
    liquid_color: int,
) -> TubeSpec | None:
    h, w = shape
    spill_space = random.randint(1, 5)
    inner_len, liquid_len = _choose_lengths(branch, spill_space)
    if open_dir in ("up", "down"):
        outer_h, outer_w = inner_len + 2, 3
        c0 = random.randint(0, w - outer_w)
        if open_dir == "up":
            r0_min = spill_space + 1
            r0_max = h - outer_h
        else:
            r0_min = 0
            r0_max = h - outer_h - (spill_space + 1)
    else:
        outer_h, outer_w = 3, inner_len + 2
        r0 = random.randint(0, h - outer_h)
        if open_dir == "left":
            c0_min = spill_space + 1
            c0_max = w - outer_w
        else:
            c0_min = 0
            c0_max = w - outer_w - (spill_space + 1)

    if open_dir in ("up", "down"):
        if r0_min > r0_max:
            return None
        r0 = random.randint(r0_min, r0_max)
    else:
        if c0_min > c0_max:
            return None
        c0 = random.randint(c0_min, c0_max)

    return TubeSpec(
        r0=r0,
        c0=c0,
        inner_len=inner_len,
        open_dir=open_dir,
        wall_color=wall_color,
        liquid_color=liquid_color,
        liquid_len=liquid_len,
        spill_space=spill_space,
    )


def _gen_attempt() -> np.ndarray | None:
    h = random.randint(18, 26)
    w = random.randint(18, 26)
    shape = (h, w)
    tube_count = random.randint(2, 4)

    bg, control_color = random.sample(PALETTE, 2)
    remaining = [color for color in PALETTE if color not in (bg, control_color)]
    random.shuffle(remaining)

    dir_choices = [random.choice(["up", "down"]), random.choice(["left", "right"])]
    while len(dir_choices) < tube_count:
        dir_choices.append(random.choice(list(DIRS)))
    random.shuffle(dir_choices)

    branches = ["partial", "full"]
    while len(branches) < tube_count:
        branches.append(random.choice(["partial", "full"]))
    random.shuffle(branches)

    tubes: list[TubeSpec] = []
    boxes: list[tuple[int, int, int, int]] = []
    for idx in range(tube_count):
        wall_color = remaining[2 * idx]
        liquid_color = remaining[2 * idx + 1]
        placed = None
        for _ in range(80):
            candidate = _random_spec(shape, dir_choices[idx], branches[idx], wall_color, liquid_color)
            if candidate is None:
                continue
            box = _expanded_bbox(candidate, shape)
            if any(_boxes_overlap(box, old) for old in boxes):
                continue
            placed = candidate
            boxes.append(box)
            tubes.append(candidate)
            break
        if placed is None:
            return None

    inp = _render_input(shape, bg, control_color, tubes)
    expected = _render_output(inp, bg, tubes)

    if _get_bg(inp) != bg:
        return None
    if not any(spec.liquid_len > spec.spill_space for spec in tubes):
        return None
    if not any(spec.liquid_len <= spec.spill_space for spec in tubes):
        return None

    try:
        actual = main(inp)
    except Exception:
        return None
    if not np.array_equal(actual, expected):
        return None
    if np.array_equal(inp, expected):
        return None
    return inp


def generate_input() -> np.ndarray:
    for _ in range(GEN_ATTEMPTS):
        grid = _gen_attempt()
        if grid is not None:
            return grid
    raise RuntimeError("failed to generate a valid tube-pouring input")
