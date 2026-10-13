from __future__ import annotations

import random
from collections import defaultdict, deque

import numpy as np


DIRS = {
    "up": (-1, 0),
    "down": (1, 0),
    "left": (0, -1),
    "right": (0, 1),
}

PALETTE = [1, 2, 3, 4, 5, 6, 7, 8, 9]
STEP_CHOICES = [3, 4]
MAX_GEN_ATTEMPTS = 400


def _neighbors4(r: int, c: int) -> list[tuple[int, int]]:
    return [(r - 1, c), (r + 1, c), (r, c - 1), (r, c + 1)]


def _components_by_color(grid: np.ndarray, bg: int) -> dict[int, list[list[tuple[int, int]]]]:
    h, w = grid.shape
    seen = np.zeros((h, w), dtype=bool)
    comps: dict[int, list[list[tuple[int, int]]]] = defaultdict(list)

    for r in range(h):
        for c in range(w):
            color = int(grid[r, c])
            if seen[r, c] or color == bg:
                continue
            seen[r, c] = True
            q = deque([(r, c)])
            comp: list[tuple[int, int]] = []
            while q:
                rr, cc = q.popleft()
                comp.append((int(rr), int(cc)))
                for nr, nc in _neighbors4(rr, cc):
                    if 0 <= nr < h and 0 <= nc < w and not seen[nr, nc] and int(grid[nr, nc]) == color:
                        seen[nr, nc] = True
                        q.append((nr, nc))
            comps[color].append(comp)
    return comps


def _is_solid_2x2_block(coords: list[tuple[int, int]]) -> bool:
    if len(coords) != 4:
        return False
    rows = sorted({r for r, _ in coords})
    cols = sorted({c for _, c in coords})
    if len(rows) != 2 or len(cols) != 2:
        return False
    return {(r, c) for r in rows for c in cols} == set(coords)


def _decode_l_instruction(coords: list[tuple[int, int]]) -> tuple[str, int]:
    pts = set(coords)
    candidates: list[tuple[int, int, str]] = []

    by_row: dict[int, list[int]] = defaultdict(list)
    by_col: dict[int, list[int]] = defaultdict(list)
    for r, c in pts:
        by_row[r].append(c)
        by_col[c].append(r)

    for r, cols in by_row.items():
        cols = sorted(cols)
        if len(cols) < 2 or len(cols) > 4:
            continue
        if cols != list(range(cols[0], cols[-1] + 1)):
            continue
        axis = {(r, c) for c in cols}
        extra = pts - axis
        if len(extra) != 1:
            continue
        er, ec = next(iter(extra))
        if ec not in (cols[0], cols[-1]) or abs(er - r) != 1:
            continue
        direction = "up" if er < r else "down"
        candidates.append((len(cols), 1, direction))

    for c, rows in by_col.items():
        rows = sorted(rows)
        if len(rows) < 2 or len(rows) > 4:
            continue
        if rows != list(range(rows[0], rows[-1] + 1)):
            continue
        axis = {(r, c) for r in rows}
        extra = pts - axis
        if len(extra) != 1:
            continue
        er, ec = next(iter(extra))
        if er not in (rows[0], rows[-1]) or abs(ec - c) != 1:
            continue
        direction = "left" if ec < c else "right"
        candidates.append((len(rows), 0, direction))

    if not candidates:
        raise AssertionError("component is not a valid L instruction")

    steps, _orientation_priority, direction = max(candidates)
    return direction, steps


def _component_sort_key(coords: list[tuple[int, int]], h: int, w: int) -> tuple[int, int, int]:
    min_r = min(r for r, _ in coords)
    min_c = min(c for _, c in coords)
    edge_dist = min(min(r, c, h - 1 - r, w - 1 - c) for r, c in coords)
    return (int(edge_dist), int(min_r), int(min_c))


def _rects_overlap(a: tuple[int, int], b: tuple[int, int]) -> bool:
    return abs(a[0] - b[0]) < 2 and abs(a[1] - b[1]) < 2


def _simulate_instruction(
    positions: dict[int, tuple[int, int]],
    color: int,
    direction: str,
    steps: int,
    h: int,
    w: int,
) -> tuple[dict[int, tuple[int, int]], int, str]:
    dr, dc = DIRS[direction]
    r, c = positions[color]
    moved = 0
    reason = "full"

    while moved < steps:
        nr, nc = r + dr, c + dc
        if nr < 0 or nc < 0 or nr + 1 >= h or nc + 1 >= w:
            reason = "boundary"
            break
        blocked = any(_rects_overlap((nr, nc), pos) for other, pos in positions.items() if other != color)
        if blocked:
            reason = "block"
            break
        r, c = nr, nc
        moved += 1

    new_positions = dict(positions)
    new_positions[color] = (int(r), int(c))
    return new_positions, int(moved), reason


def _paint_blocks(shape: tuple[int, int], bg: int, positions: dict[int, tuple[int, int]]) -> np.ndarray:
    out = np.full(shape, bg, dtype=int)
    for color, (r, c) in positions.items():
        out[r : r + 2, c : c + 2] = int(color)
    return out


def main(grid: np.ndarray) -> np.ndarray:
    """
    Move 2x2 colored blocks by decoding same-colored L symbols.

    Each L encodes:
    - move direction = the side on which the bend pixel protrudes
    - move distance = the length of the straight arm
    Ls execute from outermost to innermost, and blocks stop before boundaries
    or other blocks.
    """
    g = np.asarray(grid, dtype=int)
    h, w = g.shape
    bg = int(np.bincount(g.ravel()).argmax())

    blocks: dict[int, tuple[int, int]] = {}
    instructions: list[tuple[tuple[int, int, int], int, str, int]] = []

    for color, comps in _components_by_color(g, bg).items():
        for comp in comps:
            rows = [r for r, _ in comp]
            cols = [c for _, c in comp]
            if _is_solid_2x2_block(comp):
                if color in blocks:
                    raise AssertionError("each color can have at most one block")
                blocks[color] = (int(min(rows)), int(min(cols)))
            else:
                direction, steps = _decode_l_instruction(comp)
                instructions.append((_component_sort_key(comp, h, w), int(color), direction, int(steps)))

    instructions.sort(key=lambda item: item[0])

    positions = dict(blocks)
    for _key, color, direction, steps in instructions:
        if color not in positions:
            raise AssertionError("instruction color must match an existing block")
        positions, _moved, _reason = _simulate_instruction(positions, color, direction, steps, h, w)

    return _paint_blocks((h, w), bg, positions)


def _block_cells(pos: tuple[int, int]) -> set[tuple[int, int]]:
    r, c = pos
    return {(r + dr, c + dc) for dr in range(2) for dc in range(2)}


def _halo(cells: set[tuple[int, int]], h: int, w: int) -> set[tuple[int, int]]:
    out: set[tuple[int, int]] = set()
    for r, c in cells:
        for rr in range(r - 1, r + 2):
            for cc in range(c - 1, c + 2):
                if 0 <= rr < h and 0 <= cc < w:
                    out.add((int(rr), int(cc)))
    return out


def _normalized_l_variants(direction: str, steps: int) -> list[list[tuple[int, int]]]:
    if direction == "up":
        return [
            [(1, c) for c in range(steps)] + [(0, 0)],
            [(1, c) for c in range(steps)] + [(0, steps - 1)],
        ]
    if direction == "down":
        return [
            [(0, c) for c in range(steps)] + [(1, 0)],
            [(0, c) for c in range(steps)] + [(1, steps - 1)],
        ]
    if direction == "left":
        return [
            [(r, 1) for r in range(steps)] + [(0, 0)],
            [(r, 1) for r in range(steps)] + [(steps - 1, 0)],
        ]
    return [
        [(r, 0) for r in range(steps)] + [(0, 1)],
        [(r, 0) for r in range(steps)] + [(steps - 1, 1)],
    ]


def _all_l_placements(h: int, w: int, direction: str, steps: int) -> list[tuple[tuple[int, int, int], tuple[tuple[int, int], ...]]]:
    placements: list[tuple[tuple[int, int, int], tuple[tuple[int, int], ...]]] = []
    seen: set[tuple[tuple[int, int], ...]] = set()

    for variant in _normalized_l_variants(direction, steps):
        vh = max(r for r, _ in variant) + 1
        vw = max(c for _, c in variant) + 1
        for r0 in range(h - vh + 1):
            for c0 in range(w - vw + 1):
                cells = tuple(sorted((r0 + r, c0 + c) for r, c in variant))
                if cells in seen:
                    continue
                seen.add(cells)
                min_r = cells[0][0]
                min_c = min(c for _, c in cells)
                max_r = max(r for r, _ in cells)
                max_c = max(c for _, c in cells)
                dist = min(min(r, c, h - 1 - r, w - 1 - c) for r, c in cells)
                key = (int(dist), int(min_r), int(min_c))
                placements.append((key, cells))
    return placements


def _run_sequence(
    positions: dict[int, tuple[int, int]],
    instructions: list[tuple[int, str, int]],
    h: int,
    w: int,
) -> tuple[dict[int, tuple[int, int]], list[tuple[int, str, int, int, str]]]:
    cur = dict(positions)
    outcomes: list[tuple[int, str, int, int, str]] = []
    for color, direction, steps in instructions:
        cur, moved, reason = _simulate_instruction(cur, color, direction, steps, h, w)
        outcomes.append((int(color), direction, int(steps), int(moved), reason))
    return cur, outcomes


def _base_pair_candidates(
    h: int,
    w: int,
    axis: str,
    sign: int,
    step1: int,
    safe: int,
) -> list[dict[str, object]]:
    gap = step1 + safe + 2
    out: list[dict[str, object]] = []
    if axis == "h":
        if w < gap + 4:
            return out
        for r in range(1, h - 2):
            for c0 in range(1, w - gap - 2):
                if sign > 0:
                    apos = (r, c0)
                    bpos = (r, c0 + gap)
                    adir, bdir = "right", "left"
                else:
                    apos = (r, c0 + gap)
                    bpos = (r, c0)
                    adir, bdir = "left", "right"
                out.append({"A": apos, "B": bpos, "adir": adir, "bdir": bdir})
    else:
        if h < gap + 4:
            return out
        for c in range(1, w - 2):
            for r0 in range(1, h - gap - 2):
                if sign > 0:
                    apos = (r0, c)
                    bpos = (r0 + gap, c)
                    adir, bdir = "down", "up"
                else:
                    apos = (r0 + gap, c)
                    bpos = (r0, c)
                    adir, bdir = "up", "down"
                out.append({"A": apos, "B": bpos, "adir": adir, "bdir": bdir})
    return out


def _non_overlapping_block(pos: tuple[int, int], positions: dict[int, tuple[int, int]]) -> bool:
    return all(not _rects_overlap(pos, other) for other in positions.values())


def _boundary_positions(h: int, w: int, direction: str, moved: int) -> list[tuple[int, int]]:
    if direction == "up":
        return [(moved, c) for c in range(w - 1)]
    if direction == "down":
        return [(h - 2 - moved, c) for c in range(w - 1)]
    if direction == "left":
        return [(r, moved) for r in range(h - 1)]
    return [(r, w - 2 - moved) for r in range(h - 1)]


def _find_instruction_of_type(
    positions: dict[int, tuple[int, int]],
    h: int,
    w: int,
    desired: str,
    rng: random.Random,
    preferred_colors: list[int] | None = None,
) -> tuple[int, str, int] | None:
    colors = list(positions.keys())
    ordered_colors = []
    if preferred_colors is not None:
        for color in preferred_colors:
            if color in positions and color not in ordered_colors:
                ordered_colors.append(color)
    for color in colors:
        if color not in ordered_colors:
            ordered_colors.append(color)

    candidates: list[tuple[int, str, int]] = []
    for color in ordered_colors:
        directions = list(DIRS.keys())
        rng.shuffle(directions)
        for direction in directions:
            steps_list = STEP_CHOICES[:]
            rng.shuffle(steps_list)
            for steps in steps_list:
                _new_positions, moved, reason = _simulate_instruction(positions, color, direction, steps, h, w)
                if reason == desired and 0 < moved < steps:
                    candidates.append((int(color), direction, int(steps)))
                if desired == "full" and reason == "full":
                    candidates.append((int(color), direction, int(steps)))
    if not candidates:
        return None
    return rng.choice(candidates)


def _place_l_shapes(
    h: int,
    w: int,
    positions: dict[int, tuple[int, int]],
    instructions: list[tuple[int, str, int]],
    rng: random.Random,
) -> list[tuple[tuple[int, int], ...]] | None:
    blocked = set()
    for pos in positions.values():
        blocked |= _halo(_block_cells(pos), h, w)

    distances = list(range(len(instructions)))
    if len(instructions) >= 2 and rng.random() < 0.4:
        tie_idx = rng.randrange(len(instructions) - 1)
        distances[tie_idx + 1] = distances[tie_idx]

    all_placements = {
        (direction, steps): _all_l_placements(h, w, direction, steps)
        for _color, direction, steps in instructions
    }

    def backtrack(
        idx: int,
        cur_blocked: set[tuple[int, int]],
        prev_key: tuple[int, int, int],
    ) -> list[tuple[tuple[int, int], ...]] | None:
        if idx == len(instructions):
            return []
        color, direction, steps = instructions[idx]
        del color
        candidates: list[tuple[tuple[int, int, int], tuple[tuple[int, int], ...]]] = []
        for key, cells in all_placements[(direction, steps)]:
            if key[0] != distances[idx] or key <= prev_key:
                continue
            if any(cell in cur_blocked for cell in cells):
                continue
            candidates.append((key, cells))
        rng.shuffle(candidates)
        for key, cells in candidates:
            next_blocked = cur_blocked | _halo(set(cells), h, w)
            rest = backtrack(idx + 1, next_blocked, key)
            if rest is not None:
                return [cells] + rest
        return None

    return backtrack(0, blocked, (-1, -1, -1))


def _generate_attempt(rng: random.Random) -> np.ndarray | None:
    h = rng.randint(14, 20)
    w = rng.randint(14, 20)
    colors = rng.sample(PALETTE, rng.choice([3, 4]))
    color_a, color_b, color_c = colors[:3]
    color_d = colors[3] if len(colors) == 4 else None

    pair_axis = rng.choice(["h", "v"])
    pair_sign = rng.choice([-1, 1])
    step1 = rng.choice(STEP_CHOICES)
    safe = rng.choice([1, 2])
    step2 = rng.randint(safe + 1, 4)

    pair_candidates = _base_pair_candidates(h, w, pair_axis, pair_sign, step1, safe)
    rng.shuffle(pair_candidates)

    for pair in pair_candidates:
        positions: dict[int, tuple[int, int]] = {
            color_a: pair["A"],  # type: ignore[index]
            color_b: pair["B"],  # type: ignore[index]
        }
        base_instructions: list[tuple[int, str, int]] = [
            (color_a, pair["adir"], step1),  # type: ignore[index]
            (color_b, pair["bdir"], step2),  # type: ignore[index]
        ]

        after_ab, ab_outcomes = _run_sequence(positions, base_instructions, h, w)
        reasons = [item[-1] for item in ab_outcomes]
        if reasons != ["full", "block"]:
            continue

        boundary_dir = rng.choice(list(DIRS.keys()))
        boundary_move = rng.choice([1, 2])
        boundary_steps = rng.choice(STEP_CHOICES)
        if boundary_steps <= boundary_move:
            continue

        c_candidates = _boundary_positions(h, w, boundary_dir, boundary_move)
        rng.shuffle(c_candidates)
        c_pos = None
        for cand in c_candidates:
            if not _non_overlapping_block(cand, positions):
                continue
            trial_positions = dict(positions)
            trial_positions[color_c] = cand
            _final_trial, trial_outcomes = _run_sequence(
                trial_positions,
                base_instructions + [(color_c, boundary_dir, boundary_steps)],
                h,
                w,
            )
            if [item[-1] for item in trial_outcomes] == ["full", "block", "boundary"]:
                c_pos = cand
                positions = trial_positions
                break
        if c_pos is None:
            continue

        instructions = base_instructions + [(color_c, boundary_dir, boundary_steps)]

        if color_d is not None and rng.random() < 0.65:
            d_slots = [(r, c) for r in range(h - 1) for c in range(w - 1)]
            rng.shuffle(d_slots)
            for cand in d_slots:
                if not _non_overlapping_block(cand, positions):
                    continue
                block_cells = _block_cells(cand)
                halos = set()
                for pos in positions.values():
                    halos |= _halo(_block_cells(pos), h, w)
                if any(cell in halos for cell in block_cells):
                    continue
                trial_positions = dict(positions)
                trial_positions[color_d] = cand
                _final_trial, trial_outcomes = _run_sequence(trial_positions, instructions, h, w)
                if [item[-1] for item in trial_outcomes] == ["full", "block", "boundary"]:
                    positions = trial_positions
                    break

        final_positions, _base_outcomes = _run_sequence(positions, instructions, h, w)

        extra_target_count = rng.choice([0, 1, 1, 2])
        cur_positions = dict(final_positions)
        for _ in range(extra_target_count):
            desired = rng.choice(["full", "boundary", "block"])
            preferred = [color_a, color_b] if rng.random() < 0.6 else None
            extra = _find_instruction_of_type(cur_positions, h, w, desired, rng, preferred)
            if extra is None:
                continue
            instructions.append(extra)
            cur_positions, moved, reason = _simulate_instruction(cur_positions, extra[0], extra[1], extra[2], h, w)
            if reason != desired or moved == 0:
                return None

        if len(instructions) > 5:
            continue

        placements = _place_l_shapes(h, w, positions, instructions, rng)
        if placements is None:
            continue

        grid = np.zeros((h, w), dtype=int)
        for color, pos in positions.items():
            r, c = pos
            grid[r : r + 2, c : c + 2] = int(color)
        for (color, _direction, _steps), cells in zip(instructions, placements):
            for r, c in cells:
                if grid[r, c] != 0:
                    return None
                grid[r, c] = int(color)

        expected_positions, expected_outcomes = _run_sequence(positions, instructions, h, w)
        outcome_types = {item[-1] for item in expected_outcomes}
        if not {"full", "boundary", "block"} <= outcome_types:
            continue

        expected = _paint_blocks((h, w), 0, expected_positions)
        actual = main(grid)
        if not np.array_equal(actual, expected):
            continue
        return grid

    return None


def generate_input() -> np.ndarray:
    rng = random.Random()
    for _ in range(MAX_GEN_ATTEMPTS):
        grid = _generate_attempt(rng)
        if grid is not None:
            return grid
    raise RuntimeError(f"generate_input failed after {MAX_GEN_ATTEMPTS} attempts")
