from __future__ import annotations

from collections import Counter, defaultdict
from dataclasses import dataclass
import random

import numpy as np


@dataclass(frozen=True)
class Belt:
    belt_id: int
    orientation: str
    fixed: int
    lo: int
    hi: int
    color: int
    cargo_side: str
    cargo_color: int

    def cargo_cell(self) -> tuple[int, int]:
        if self.orientation == "h":
            c = self.lo - 1 if self.cargo_side == "lo" else self.hi + 1
            return self.fixed, c
        r = self.lo - 1 if self.cargo_side == "lo" else self.hi + 1
        return r, self.fixed

    def start_cell(self) -> tuple[int, int]:
        if self.orientation == "h":
            c = self.lo if self.cargo_side == "lo" else self.hi
            return self.fixed, c
        r = self.lo if self.cargo_side == "lo" else self.hi
        return r, self.fixed

    def source_dir(self) -> tuple[int, int]:
        if self.orientation == "h":
            return (0, 1) if self.cargo_side == "lo" else (0, -1)
        return (1, 0) if self.cargo_side == "lo" else (-1, 0)

    def body_cells(self) -> list[tuple[int, int]]:
        if self.orientation == "h":
            return [(self.fixed, c) for c in range(self.lo, self.hi + 1)]
        return [(r, self.fixed) for r in range(self.lo, self.hi + 1)]

    def end_outside(self, step: tuple[int, int]) -> tuple[int, int]:
        dr, dc = step
        if self.orientation == "h":
            end_c = self.hi if dc > 0 else self.lo
            return self.fixed + dr, end_c + dc
        end_r = self.hi if dr > 0 else self.lo
        return end_r + dr, self.fixed + dc

    def contains(self, cell: tuple[int, int]) -> bool:
        r, c = cell
        if self.orientation == "h":
            return r == self.fixed and self.lo <= c <= self.hi
        return c == self.fixed and self.lo <= r <= self.hi

    def farther_dir(self, cell: tuple[int, int]) -> tuple[int, int]:
        r, c = cell
        if self.orientation == "h":
            left = c - self.lo
            right = self.hi - c
            assert left != right, "turned belt arm lengths must be unequal"
            return (0, -1) if left > right else (0, 1)
        up = r - self.lo
        down = self.hi - r
        assert up != down, "turned belt arm lengths must be unequal"
        return (-1, 0) if up > down else (1, 0)


def _background_color(g: np.ndarray) -> int:
    counts = Counter(map(int, g.ravel()))
    return int(max(counts.items(), key=lambda kv: (kv[1], -kv[0]))[0])


def _unique_mode(values: list[int]) -> int | None:
    if not values:
        return None
    counts = Counter(values)
    best = max(counts.values())
    winners = [color for color, count in counts.items() if count == best]
    if len(winners) != 1:
        return None
    return int(winners[0])


def _occupied_runs(mask_1d: np.ndarray) -> list[tuple[int, int]]:
    runs: list[tuple[int, int]] = []
    i = 0
    n = int(mask_1d.size)
    while i < n:
        if not bool(mask_1d[i]):
            i += 1
            continue
        j = i
        while j + 1 < n and bool(mask_1d[j + 1]):
            j += 1
        runs.append((i, j))
        i = j + 1
    return runs


def _detect_belts(g: np.ndarray, bg: int) -> list[Belt]:
    occ = g != bg
    h, w = g.shape
    belts: list[Belt] = []
    belt_id = 0

    for r in range(h):
        for c0, c1 in _occupied_runs(occ[r]):
            if c1 - c0 + 1 < 4:
                continue
            cross_cols = {
                c
                for c in range(c0, c1 + 1)
                if 0 < r < h - 1 and bool(occ[r - 1, c]) and bool(occ[r + 1, c])
            }
            sample_colors = [int(g[r, c]) for c in range(c0, c1 + 1) if c not in cross_cols]
            base = _unique_mode(sample_colors)
            if base is None:
                continue

            left_cargo = c0 not in cross_cols and int(g[r, c0]) != base
            right_cargo = c1 not in cross_cols and int(g[r, c1]) != base
            if left_cargo == right_cargo:
                continue

            lo = c0 + 1 if left_cargo else c0
            hi = c1 - 1 if right_cargo else c1
            if hi - lo + 1 < 3:
                continue
            if lo <= 0 or hi >= w - 1:
                continue
            if any(int(g[r, c]) != base for c in range(lo, hi + 1) if c not in cross_cols):
                continue

            cargo_side = "lo" if left_cargo else "hi"
            cargo_c = c0 if left_cargo else c1
            belts.append(
                Belt(
                    belt_id=belt_id,
                    orientation="h",
                    fixed=r,
                    lo=lo,
                    hi=hi,
                    color=base,
                    cargo_side=cargo_side,
                    cargo_color=int(g[r, cargo_c]),
                )
            )
            belt_id += 1

    for c in range(w):
        for r0, r1 in _occupied_runs(occ[:, c]):
            if r1 - r0 + 1 < 4:
                continue
            cross_rows = {
                r
                for r in range(r0, r1 + 1)
                if 0 < c < w - 1 and bool(occ[r, c - 1]) and bool(occ[r, c + 1])
            }
            sample_colors = [int(g[r, c]) for r in range(r0, r1 + 1) if r not in cross_rows]
            base = _unique_mode(sample_colors)
            if base is None:
                continue

            top_cargo = r0 not in cross_rows and int(g[r0, c]) != base
            bottom_cargo = r1 not in cross_rows and int(g[r1, c]) != base
            if top_cargo == bottom_cargo:
                continue

            lo = r0 + 1 if top_cargo else r0
            hi = r1 - 1 if bottom_cargo else r1
            if hi - lo + 1 < 3:
                continue
            if lo <= 0 or hi >= h - 1:
                continue
            if any(int(g[r, c]) != base for r in range(lo, hi + 1) if r not in cross_rows):
                continue

            cargo_side = "lo" if top_cargo else "hi"
            cargo_r = r0 if top_cargo else r1
            belts.append(
                Belt(
                    belt_id=belt_id,
                    orientation="v",
                    fixed=c,
                    lo=lo,
                    hi=hi,
                    color=base,
                    cargo_side=cargo_side,
                    cargo_color=int(g[cargo_r, c]),
                )
            )
            belt_id += 1

    assert belts, "expected at least one conveyor belt"
    return belts


def _build_cell_index(belts: list[Belt]) -> dict[tuple[int, int], list[int]]:
    cell_to_belts: dict[tuple[int, int], list[int]] = defaultdict(list)
    for belt in belts:
        for cell in belt.body_cells():
            cell_to_belts[cell].append(belt.belt_id)
    for cell, owners in cell_to_belts.items():
        assert len(owners) <= 2, f"unexpected overlap multiplicity at {cell}"
        if len(owners) == 2:
            assert owners[0] != owners[1]
    return cell_to_belts


def _all_non_bg_cells(g: np.ndarray, bg: int) -> set[tuple[int, int]]:
    return {tuple(map(int, cell)) for cell in np.argwhere(g != bg)}


def _trace_destination(
    g: np.ndarray,
    belt: Belt,
    belt_by_id: dict[int, Belt],
    cell_to_belts: dict[tuple[int, int], list[int]],
) -> tuple[int, int]:
    current = belt
    step = current.source_dir()
    cell = current.start_cell()
    seen: set[tuple[int, tuple[int, int], tuple[int, int]]] = set()

    while True:
        state = (current.belt_id, cell, step)
        assert state not in seen, "cargo path should not loop"
        seen.add(state)

        owners = cell_to_belts[cell]
        if len(owners) == 2:
            other_id = owners[0] if owners[1] == current.belt_id else owners[1]
            other = belt_by_id[other_id]
            pixel_color = int(g[cell])
            if pixel_color != current.color:
                assert pixel_color == other.color, "intersection color must match one belt"
                current = other
                step = current.farther_dir(cell)

        next_cell = (cell[0] + step[0], cell[1] + step[1])
        if current.contains(next_cell):
            cell = next_cell
            continue
        return next_cell


def main(grid: np.ndarray) -> np.ndarray:
    """Move cargo pixels along colored conveyor segments and route them through switch-colored crossings.

    Input structure:
    - The background is the dominant black color.
    - Foreground line segments are horizontal or vertical one-cell-thick conveyors, each mostly one color.
    - Conveyors may cross at shared pixels; the crossing pixel keeps one of the two belt colors.
    - Each conveyor has a single cargo pixel just outside one endpoint, using a non-belt color.

    Rule:
    - Identify each conveyor and the endpoint that has an adjacent cargo.
    - Move that cargo onto the belt and follow the belt from its cargo-bearing end toward the opposite end.
    - At a crossing, continue straight if the crossing pixel has the current belt color; otherwise switch to the other belt and continue toward that belt arm farther from the crossing.
    - Remove each cargo from its original source cell and place it one cell beyond the endpoint where its routed path exits the final belt. All conveyor pixels stay unchanged.

    Key insight:
    - Crossing colors act as switches: they decide whether a cargo stays on its current straight segment or turns into the perpendicular segment, so the full routed path must be traced instead of treating each line independently.
    """
    g = np.asarray(grid, dtype=int)
    bg = _background_color(g)
    belts = _detect_belts(g, bg)
    belt_by_id = {belt.belt_id: belt for belt in belts}
    cell_to_belts = _build_cell_index(belts)

    cargo_cells = {belt.cargo_cell() for belt in belts}
    belt_cells = set(cell_to_belts)
    non_bg = _all_non_bg_cells(g, bg)
    assert cargo_cells.isdisjoint(belt_cells)
    assert non_bg == belt_cells | cargo_cells, "unexpected extra foreground pixels"

    out = g.copy()
    for belt in belts:
        r, c = belt.cargo_cell()
        out[r, c] = bg

    placed: dict[tuple[int, int], int] = {}
    for belt in belts:
        dest = _trace_destination(g, belt, belt_by_id, cell_to_belts)
        r, c = dest
        assert 0 <= r < g.shape[0] and 0 <= c < g.shape[1], "cargo destination must remain in bounds"
        if dest in placed:
            assert placed[dest] == belt.cargo_color, "colliding cargos must agree in color"
        else:
            assert int(out[r, c]) == bg, "final cargo cell must be empty after clearing sources"
            placed[dest] = belt.cargo_color
            out[r, c] = belt.cargo_color

    return out


PALETTE = list(range(10))
BG = 0
GEN_RETRIES = 400


def _belt_body(spec: dict[str, int | str]) -> list[tuple[int, int]]:
    if spec["orientation"] == "h":
        return [(int(spec["fixed"]), c) for c in range(int(spec["lo"]), int(spec["hi"]) + 1)]
    return [(r, int(spec["fixed"])) for r in range(int(spec["lo"]), int(spec["hi"]) + 1)]


def _cargo_cell(spec: dict[str, int | str]) -> tuple[int, int]:
    if spec["orientation"] == "h":
        c = int(spec["lo"]) - 1 if spec["cargo_side"] == "lo" else int(spec["hi"]) + 1
        return int(spec["fixed"]), c
    r = int(spec["lo"]) - 1 if spec["cargo_side"] == "lo" else int(spec["hi"]) + 1
    return r, int(spec["fixed"])


def _empty_exit(spec: dict[str, int | str]) -> tuple[int, int]:
    if spec["orientation"] == "h":
        c = int(spec["hi"]) + 1 if spec["cargo_side"] == "lo" else int(spec["lo"]) - 1
        return int(spec["fixed"]), c
    r = int(spec["hi"]) + 1 if spec["cargo_side"] == "lo" else int(spec["lo"]) - 1
    return r, int(spec["fixed"])


def _expanded(cells: set[tuple[int, int]], h: int, w: int, radius: int = 1) -> set[tuple[int, int]]:
    out: set[tuple[int, int]] = set()
    for r, c in cells:
        for dr in range(-radius, radius + 1):
            for dc in range(-radius, radius + 1):
                rr, cc = r + dr, c + dc
                if 0 <= rr < h and 0 <= cc < w:
                    out.add((rr, cc))
    return out


def _can_place_independent(
    spec: dict[str, int | str],
    occupied: set[tuple[int, int]],
    guarded: set[tuple[int, int]],
    h: int,
    w: int,
) -> bool:
    cells = set(_belt_body(spec))
    cells.add(_cargo_cell(spec))
    cells.add(_empty_exit(spec))
    if any(not (0 <= r < h and 0 <= c < w) for r, c in cells):
        return False
    if occupied & cells:
        return False
    return not (guarded & cells)


def _random_independent_spec(
    h: int,
    w: int,
    belt_color: int,
    cargo_color: int,
) -> dict[str, int | str]:
    orientation = random.choice(["h", "v"])
    cargo_side = random.choice(["lo", "hi"])
    if orientation == "h":
        max_len = min(8, w - 3)
        length = random.randint(3, max_len)
        fixed = random.randint(1, h - 2)
        lo = random.randint(1, w - length - 2)
        hi = lo + length - 1
    else:
        max_len = min(8, h - 3)
        length = random.randint(3, max_len)
        fixed = random.randint(1, w - 2)
        lo = random.randint(1, h - length - 2)
        hi = lo + length - 1
    return {
        "orientation": orientation,
        "fixed": fixed,
        "lo": lo,
        "hi": hi,
        "color": belt_color,
        "cargo_side": cargo_side,
        "cargo_color": cargo_color,
    }


def _render_scene(
    h: int,
    w: int,
    belts: list[dict[str, int | str]],
    crossings: list[tuple[int, int, int]],
) -> np.ndarray:
    grid = np.full((h, w), BG, dtype=int)
    for spec in belts:
        for r, c in _belt_body(spec):
            grid[r, c] = int(spec["color"])
    for r, c, color in crossings:
        grid[r, c] = color
    for spec in belts:
        r, c = _cargo_cell(spec)
        grid[r, c] = int(spec["cargo_color"])
    return grid


def _sample_independent_scene() -> np.ndarray | None:
    h = random.randint(10, 24)
    w = random.randint(10, 24)
    belt_count = random.randint(1, 4)
    colors = random.sample(PALETTE[1:], 2 * belt_count)
    belt_colors = colors[:belt_count]
    cargo_colors = colors[belt_count:]

    belts: list[dict[str, int | str]] = []
    occupied: set[tuple[int, int]] = set()
    guarded: set[tuple[int, int]] = set()

    for belt_color, cargo_color in zip(belt_colors, cargo_colors):
        placed = False
        for _ in range(80):
            spec = _random_independent_spec(h, w, belt_color, cargo_color)
            if not _can_place_independent(spec, occupied, guarded, h, w):
                continue
            cells = set(_belt_body(spec))
            cells.add(_cargo_cell(spec))
            cells.add(_empty_exit(spec))
            belts.append(spec)
            occupied |= cells
            guarded |= _expanded(cells, h, w)
            placed = True
            break
        if not placed:
            return None

    return _render_scene(h, w, belts, [])


def _sample_cross_scene() -> np.ndarray | None:
    h = random.randint(11, 24)
    w = random.randint(11, 24)
    extra_count = random.randint(0, 2)
    colors = random.sample(PALETTE[1:], 4 + 2 * extra_count)
    h_color, v_color, h_cargo, v_cargo = colors[:4]
    extra_pairs = colors[4:]

    receiver = random.choice(["h", "v"])

    # Horizontal belt.
    row = random.randint(2, h - 3)
    left_arm = random.randint(1, 4)
    right_arm = random.randint(1, 4)
    if receiver == "h":
        while left_arm == right_arm:
            right_arm = random.randint(1, 4)
    cross_c = random.randint(3, w - 4)
    h_lo = cross_c - left_arm
    h_hi = cross_c + right_arm
    if h_lo < 1 or h_hi > w - 2:
        return None

    # Vertical belt.
    col = cross_c
    up_arm = random.randint(1, 4)
    down_arm = random.randint(1, 4)
    if receiver == "v":
        while up_arm == down_arm:
            down_arm = random.randint(1, 4)
    cross_r = row
    v_lo = cross_r - up_arm
    v_hi = cross_r + down_arm
    if v_lo < 1 or v_hi > h - 2:
        return None

    if receiver == "h":
        h_side = "lo" if left_arm > right_arm else "hi"
        v_side = random.choice(["lo", "hi"])
        cross_color = h_color
    else:
        v_side = "lo" if up_arm > down_arm else "hi"
        h_side = random.choice(["lo", "hi"])
        cross_color = v_color

    h_spec: dict[str, int | str] = {
        "orientation": "h",
        "fixed": row,
        "lo": h_lo,
        "hi": h_hi,
        "color": h_color,
        "cargo_side": h_side,
        "cargo_color": h_cargo,
    }
    v_spec: dict[str, int | str] = {
        "orientation": "v",
        "fixed": col,
        "lo": v_lo,
        "hi": v_hi,
        "color": v_color,
        "cargo_side": v_side,
        "cargo_color": v_cargo,
    }

    belts = [h_spec, v_spec]
    occupied = set(_belt_body(h_spec)) | set(_belt_body(v_spec))
    occupied.add(_cargo_cell(h_spec))
    occupied.add(_cargo_cell(v_spec))
    occupied.add(_empty_exit(h_spec))
    occupied.add(_empty_exit(v_spec))
    guarded = _expanded(occupied, h, w)

    for i in range(extra_count):
        belt_color = extra_pairs[2 * i]
        cargo_color = extra_pairs[2 * i + 1]
        placed = False
        for _ in range(100):
            spec = _random_independent_spec(h, w, belt_color, cargo_color)
            if not _can_place_independent(spec, occupied, guarded, h, w):
                continue
            cells = set(_belt_body(spec))
            cells.add(_cargo_cell(spec))
            cells.add(_empty_exit(spec))
            belts.append(spec)
            occupied |= cells
            guarded |= _expanded(cells, h, w)
            placed = True
            break
        if not placed:
            return None

    return _render_scene(h, w, belts, [(row, col, cross_color)])


def _gen_attempt() -> np.ndarray | None:
    grid = _sample_cross_scene() if random.random() < 0.7 else _sample_independent_scene()
    if grid is None:
        return None
    try:
        _ = main(grid)
    except AssertionError:
        return None
    return grid


def generate_input() -> np.ndarray:
    for _ in range(GEN_RETRIES):
        grid = _gen_attempt()
        if grid is not None:
            return grid
    raise RuntimeError("failed to generate a valid conveyor-switch input")
