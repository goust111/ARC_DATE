from __future__ import annotations

from dataclasses import dataclass
import random

import numpy as np


BLACK = 0
GREY = 5
MOVER_COLORS = (1, 2, 3, 4, 6, 7, 8, 9)
DIR4 = ((-1, 0), (1, 0), (0, -1), (0, 1))
BRANCHES = ("meet", "face", "gap")
GEN_RETRIES = 300


@dataclass(frozen=True)
class Mover:
    pos: tuple[int, int]
    color: int
    tail_dir: tuple[int, int]
    move_dir: tuple[int, int]
    fuel: int


@dataclass(frozen=True)
class PairSpec:
    orientation: str
    fuel_a: int
    fuel_b: int
    gap: int
    color_a: int
    color_b: int
    branch: str

    @property
    def span(self) -> int:
        return int(self.fuel_a + self.gap + self.fuel_b + 2)


@dataclass(frozen=True)
class Placement:
    spec: PairSpec
    start: tuple[int, int]


def _in_bounds(shape: tuple[int, int], r: int, c: int) -> bool:
    h, w = map(int, shape)
    return 0 <= int(r) < h and 0 <= int(c) < w


def _gray_run(grid: np.ndarray, r: int, c: int, dr: int, dc: int) -> int:
    rr = int(r) + int(dr)
    cc = int(c) + int(dc)
    run = 0
    while _in_bounds(grid.shape, rr, cc) and int(grid[rr, cc]) == GREY:
        run += 1
        rr += int(dr)
        cc += int(dc)
    return int(run)


def _tail_cells(pos: tuple[int, int], tail_dir: tuple[int, int], fuel: int) -> list[tuple[int, int]]:
    r, c = pos
    dr, dc = tail_dir
    return [(r + dr * step, c + dc * step) for step in range(1, fuel + 1)]


def _detect_movers(grid: np.ndarray) -> dict[tuple[int, int], Mover]:
    movers: dict[tuple[int, int], Mover] = {}
    coords = np.argwhere((grid != BLACK) & (grid != GREY))
    assert coords.size > 0, "expected at least one mover"

    for r0, c0 in coords:
        r = int(r0)
        c = int(c0)
        color = int(grid[r, c])
        tail_options: list[tuple[int, int, int]] = []
        for dr, dc in DIR4:
            fuel = _gray_run(grid, r, c, dr, dc)
            if fuel > 0:
                tail_options.append((dr, dc, fuel))

        assert len(tail_options) == 1, "expected each mover to have one gray tail"
        tail_dr, tail_dc, fuel = tail_options[0]
        move_dir = (-tail_dr, -tail_dc)
        movers[(r, c)] = Mover(
            pos=(r, c),
            color=color,
            tail_dir=(tail_dr, tail_dc),
            move_dir=move_dir,
            fuel=int(fuel),
        )
    return movers


def _scan_forward_pair(grid: np.ndarray, movers: dict[tuple[int, int], Mover], mover: Mover) -> Mover:
    r, c = mover.pos
    dr, dc = mover.move_dir
    rr = r + dr
    cc = c + dc
    while _in_bounds(grid.shape, rr, cc) and int(grid[rr, cc]) == BLACK:
        rr += dr
        cc += dc

    assert _in_bounds(grid.shape, rr, cc), "expected an opposite mover ahead"
    assert int(grid[rr, cc]) not in (BLACK, GREY), "expected the first non-black cell ahead to be a mover"
    other = movers[(rr, cc)]
    assert other.move_dir == (-dr, -dc), "expected paired movers to face each other"
    return other


def _pair_movers(grid: np.ndarray, movers: dict[tuple[int, int], Mover]) -> list[tuple[Mover, Mover]]:
    pairs: list[tuple[Mover, Mover]] = []
    used: set[tuple[int, int]] = set()

    for pos in sorted(movers):
        if pos in used:
            continue
        mover = movers[pos]
        other = _scan_forward_pair(grid, movers, mover)
        assert other.pos not in used, "expected disjoint mover pairs"
        used.add(mover.pos)
        used.add(other.pos)
        if mover.move_dir in ((0, 1), (1, 0)):
            pairs.append((mover, other))
        else:
            pairs.append((other, mover))

    assert len(used) == len(movers), "expected every mover to belong to exactly one pair"
    return pairs


def _consume_fuel(out: np.ndarray, tail_cells: list[tuple[int, int]], fuel: int) -> int:
    assert fuel > 0, "cannot consume empty fuel"
    tr, tc = tail_cells[fuel - 1]
    out[tr, tc] = BLACK
    return fuel - 1


def _simulate_pair(out: np.ndarray, first: Mover, second: Mover) -> None:
    pos_a = first.pos
    pos_b = second.pos
    fuel_a = int(first.fuel)
    fuel_b = int(second.fuel)
    tail_a = _tail_cells(first.pos, first.tail_dir, first.fuel)
    tail_b = _tail_cells(second.pos, second.tail_dir, second.fuel)
    color_a = int(first.color)
    color_b = int(second.color)
    dir_a = first.move_dir
    dir_b = second.move_dir

    while True:
        next_a = (pos_a[0] + dir_a[0], pos_a[1] + dir_a[1]) if fuel_a > 0 else None
        next_b = (pos_b[0] + dir_b[0], pos_b[1] + dir_b[1]) if fuel_b > 0 else None

        if next_a is not None and next_b is not None and next_a == next_b:
            fuel_a = _consume_fuel(out, tail_a, fuel_a)
            fuel_b = _consume_fuel(out, tail_b, fuel_b)
            out[next_a[0], next_a[1]] = GREY
            return

        move_a = next_a is not None and next_a != pos_b
        move_b = next_b is not None and next_b != pos_a
        if not move_a and not move_b:
            return

        if move_a:
            fuel_a = _consume_fuel(out, tail_a, fuel_a)
            out[pos_a[0], pos_a[1]] = color_a
            pos_a = next_a
            out[pos_a[0], pos_a[1]] = color_a

        if move_b:
            fuel_b = _consume_fuel(out, tail_b, fuel_b)
            out[pos_b[0], pos_b[1]] = color_b
            pos_b = next_b
            out[pos_b[0], pos_b[1]] = color_b


def main(grid: np.ndarray) -> np.ndarray:
    """
    Move each opposite-facing mover pair inward, letting gray fuel limit how far their colored trails extend.

    Input structure:
    - The background is black, and every active object is a single colored mover pixel with a contiguous gray tail behind it.
    - Movers appear in horizontal or vertical pairs on the same lane, with their tails pointing away from the center.
    - The space between the two movers of a pair is initially black.

    Rule:
    - For each pair, infer the travel direction from the attached gray tail and march both movers inward one step at a time.
    - Every successful step leaves the departed cell in the mover's own color and deletes one gray fuel cell from the far end of that mover's tail.
    - A mover stops when it has no fuel left, and it also stops if its next cell is occupied by the other mover; if both movers would enter the same cell simultaneously, paint that meeting cell gray and stop both.

    Key insight:
    - The gray tail is a visible step budget: it determines both the maximum trail length and whether a lane ends with a gap, a face-to-face stop, or a gray meeting mark.
    """
    g = np.asarray(grid, dtype=int)
    assert g.ndim == 2, "expected a 2D grid"
    movers = _detect_movers(g)
    pairs = _pair_movers(g, movers)

    out = g.copy()
    for first, second in pairs:
        _simulate_pair(out, first, second)
    return out


def _simulate_scalar(gap: int, fuel_a: int, fuel_b: int) -> str:
    pos_a = 0
    pos_b = int(gap) + 1
    fuel_left = int(fuel_a)
    fuel_right = int(fuel_b)

    while True:
        next_a = pos_a + 1 if fuel_left > 0 else None
        next_b = pos_b - 1 if fuel_right > 0 else None

        if next_a is not None and next_b is not None and next_a == next_b:
            return "meet"

        move_a = next_a is not None and next_a != pos_b
        move_b = next_b is not None and next_b != pos_a
        if not move_a and not move_b:
            break

        if move_a:
            fuel_left -= 1
            pos_a = next_a
        if move_b:
            fuel_right -= 1
            pos_b = next_b

    return "face" if pos_b - pos_a == 1 else "gap"


def _sample_pair_params(branch: str, rng: random.Random) -> tuple[int, int, int]:
    for _ in range(500):
        gap = rng.randint(1, 11)
        fuel_a = rng.randint(1, 7)
        fuel_b = rng.randint(1, 7)
        span = fuel_a + gap + fuel_b + 2
        if span > 26:
            continue
        if _simulate_scalar(gap, fuel_a, fuel_b) == branch:
            return gap, fuel_a, fuel_b
    raise RuntimeError(f"failed to sample branch {branch}")


def _pick_pair_count(rng: random.Random) -> int:
    return rng.choice([1, 2, 2, 3, 3, 4])


def _pick_branches(pair_count: int, rng: random.Random) -> list[str]:
    if pair_count >= 3:
        branches = ["meet", "face", "gap"]
        while len(branches) < pair_count:
            branches.append(rng.choice(BRANCHES))
    elif pair_count == 2:
        branches = rng.sample(list(BRANCHES), 2)
    else:
        branches = [rng.choice(BRANCHES)]
    rng.shuffle(branches)
    return branches


def _pick_orientations(pair_count: int, rng: random.Random) -> list[str]:
    if pair_count >= 2 and rng.random() < 0.65:
        orientations = ["h", "v"]
        while len(orientations) < pair_count:
            orientations.append(rng.choice(["h", "v"]))
    else:
        orientations = [rng.choice(["h", "v"]) for _ in range(pair_count)]
    rng.shuffle(orientations)
    return orientations


def _make_specs(rng: random.Random) -> list[PairSpec]:
    pair_count = _pick_pair_count(rng)
    branches = _pick_branches(pair_count, rng)
    orientations = _pick_orientations(pair_count, rng)
    colors = rng.sample(list(MOVER_COLORS), 2 * pair_count)

    specs: list[PairSpec] = []
    for idx in range(pair_count):
        gap, fuel_a, fuel_b = _sample_pair_params(branches[idx], rng)
        specs.append(
            PairSpec(
                orientation=orientations[idx],
                fuel_a=fuel_a,
                fuel_b=fuel_b,
                gap=gap,
                color_a=colors[2 * idx],
                color_b=colors[2 * idx + 1],
                branch=branches[idx],
            )
        )
    specs.sort(key=lambda spec: spec.span, reverse=True)
    return specs


def _canvas_bounds(specs: list[PairSpec]) -> tuple[int, int] | None:
    h_spans = [spec.span for spec in specs if spec.orientation == "h"]
    v_spans = [spec.span for spec in specs if spec.orientation == "v"]
    h_count = sum(spec.orientation == "h" for spec in specs)
    v_count = sum(spec.orientation == "v" for spec in specs)

    min_h = max(12, max(v_spans, default=0), 3 * h_count + 1)
    min_w = max(12, max(h_spans, default=0), 3 * v_count + 1)
    if min_h > 30 or min_w > 30:
        return None
    return int(min_h), int(min_w)


def _margin_bounds(orientation: str, start: tuple[int, int], span: int, shape: tuple[int, int]) -> tuple[int, int, int, int]:
    h, w = shape
    r0, c0 = start
    if orientation == "h":
        r1, c1 = r0, c0 + span - 1
    else:
        r1, c1 = r0 + span - 1, c0
    return max(0, r0 - 1), max(0, c0 - 1), min(h - 1, r1 + 1), min(w - 1, c1 + 1)


def _place_specs(specs: list[PairSpec], rng: random.Random) -> list[Placement] | None:
    bounds = _canvas_bounds(specs)
    if bounds is None:
        return None
    min_h, min_w = bounds
    h = rng.randint(min_h, 30)
    w = rng.randint(min_w, 30)
    blocked = np.zeros((h, w), dtype=bool)
    placed: list[Placement] = []

    for spec in specs:
        candidates: list[tuple[int, int]] = []
        if spec.orientation == "h":
            for r in range(h):
                for c in range(w - spec.span + 1):
                    mr0, mc0, mr1, mc1 = _margin_bounds("h", (r, c), spec.span, (h, w))
                    if not blocked[mr0 : mr1 + 1, mc0 : mc1 + 1].any():
                        candidates.append((r, c))
        else:
            for r in range(h - spec.span + 1):
                for c in range(w):
                    mr0, mc0, mr1, mc1 = _margin_bounds("v", (r, c), spec.span, (h, w))
                    if not blocked[mr0 : mr1 + 1, mc0 : mc1 + 1].any():
                        candidates.append((r, c))

        if not candidates:
            return None

        start = rng.choice(candidates)
        mr0, mc0, mr1, mc1 = _margin_bounds(spec.orientation, start, spec.span, (h, w))
        blocked[mr0 : mr1 + 1, mc0 : mc1 + 1] = True
        placed.append(Placement(spec=spec, start=start))

    return placed


def _render_pair(grid: np.ndarray, placement: Placement) -> None:
    spec = placement.spec
    r0, c0 = placement.start

    if spec.orientation == "h":
        row = r0
        left_head = c0 + spec.fuel_a
        right_head = left_head + spec.gap + 1
        if spec.fuel_a > 0:
            grid[row, c0:left_head] = GREY
        grid[row, left_head] = spec.color_a
        grid[row, right_head] = spec.color_b
        if spec.fuel_b > 0:
            grid[row, right_head + 1 : right_head + 1 + spec.fuel_b] = GREY
    else:
        col = c0
        top_head = r0 + spec.fuel_a
        bottom_head = top_head + spec.gap + 1
        if spec.fuel_a > 0:
            grid[r0:top_head, col] = GREY
        grid[top_head, col] = spec.color_a
        grid[bottom_head, col] = spec.color_b
        if spec.fuel_b > 0:
            grid[bottom_head + 1 : bottom_head + 1 + spec.fuel_b, col] = GREY


def _gen_attempt(rng: random.Random) -> np.ndarray | None:
    specs = _make_specs(rng)
    placements = _place_specs(specs, rng)
    if placements is None:
        return None

    h = max(
        start[0] + (placement.spec.span if placement.spec.orientation == "v" else 1)
        for placement in placements
        for start in [placement.start]
    )
    w = max(
        start[1] + (placement.spec.span if placement.spec.orientation == "h" else 1)
        for placement in placements
        for start in [placement.start]
    )
    grid = np.zeros((h, w), dtype=int)
    for placement in placements:
        _render_pair(grid, placement)

    out = main(grid)
    assert out.shape == grid.shape
    assert not np.array_equal(out, grid), "expected at least one mover to advance"
    return grid


def generate_input() -> np.ndarray:
    rng = random.Random()
    for _ in range(GEN_RETRIES):
        grid = _gen_attempt(rng)
        if grid is not None:
            return grid
    raise RuntimeError("failed to generate a valid input")
