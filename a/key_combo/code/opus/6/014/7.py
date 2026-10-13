import random
from typing import Iterable

import numpy as np


BG = 3
SEP = 5
SEED = 7
PINK = 6
BAR_COLORS = (1, 2, 4)

MAX_GENERATION_ATTEMPTS = 400


def _get_bg(grid: np.ndarray) -> int:
    values, counts = np.unique(np.asarray(grid, dtype=int), return_counts=True)
    return int(values[np.argmax(counts)])


def _children_between_gapped_pairs(mask: np.ndarray) -> np.ndarray:
    mask = np.asarray(mask, dtype=bool)
    children = np.zeros_like(mask, dtype=bool)
    if mask.size >= 3:
        children[1:-1] = mask[:-2] & (~mask[1:-1]) & mask[2:]
    return children


def _find_separator_col(grid: np.ndarray) -> int:
    cols = [c for c in range(grid.shape[1]) if np.all(grid[:, c] == SEP)]
    if len(cols) != 1:
        raise AssertionError("expected exactly one full-height grey separator column")
    return int(cols[0])


def _find_seed_row(grid: np.ndarray, sep_col: int) -> tuple[int, np.ndarray]:
    left = grid[:, :sep_col]
    rows = [r for r in range(grid.shape[0]) if np.any(left[r] == SEED)]
    if len(rows) != 1:
        raise AssertionError("expected exactly one orange seed row on the left side")
    row = int(rows[0])
    mask = left[row] == SEED
    return row, mask


def _grow_rows(seed_mask: np.ndarray, height: int, seed_row: int) -> list[tuple[int, int, np.ndarray]]:
    rows: list[tuple[int, int, np.ndarray]] = []
    active = np.asarray(seed_mask, dtype=bool).copy()
    row = int(seed_row) + 2
    color = PINK
    while row < height:
        children = _children_between_gapped_pairs(active)
        if not np.any(children):
            break
        rows.append((int(row), int(color), children.copy()))
        active = children
        color = SEED if color == PINK else PINK
        row += 2
    return rows


def _extract_bars(grid: np.ndarray, sep_col: int, bg: int) -> list[dict]:
    bars: list[dict] = []
    h, w = grid.shape
    for r in range(h):
        c = sep_col + 1
        while c < w:
            value = int(grid[r, c])
            if value in (bg, SEP):
                c += 1
                continue
            c0 = c
            while c < w and int(grid[r, c]) == value:
                c += 1
            c1 = c - 1
            bars.append(
                {
                    "row": int(r),
                    "c0": int(c0),
                    "c1": int(c1),
                    "length": int(c1 - c0 + 1),
                    "color": int(value),
                    "distance": int(c0 - (sep_col + 1)),
                }
            )
    bars.sort(key=lambda bar: (bar["distance"], bar["row"], bar["c0"]))
    return bars


def _extension_layer(
    grid: np.ndarray,
    sep_col: int,
    seed_row: int,
    seed_mask: np.ndarray,
    bars: list[dict],
) -> tuple[np.ndarray, set[tuple[int, int]]]:
    ext = np.full(grid.shape, -1, dtype=int)
    owner_distance = np.full(grid.shape, grid.shape[1] + 1, dtype=int)
    attempted: set[tuple[int, int]] = set()
    for bar in bars:
        r = int(bar["row"])
        c = int(bar["c0"]) - int(bar["length"])
        step = int(bar["length"])
        while c >= 0:
            attempted.add((r, c))
            if c != sep_col:
                protected_seed = r == seed_row and c < sep_col and bool(seed_mask[c])
                if not protected_seed and bar["distance"] < owner_distance[r, c]:
                    owner_distance[r, c] = int(bar["distance"])
                    ext[r, c] = int(bar["color"])
            c -= step
    return ext, attempted


def _render_growth_only(grid: np.ndarray, sep_col: int, seed_row: int, seed_mask: np.ndarray) -> np.ndarray:
    out = np.asarray(grid, dtype=int).copy()
    for row, color, mask in _grow_rows(seed_mask, out.shape[0], seed_row):
        cols = np.flatnonzero(mask)
        out[row, cols] = int(color)
    return out


def _simulate(grid: np.ndarray) -> tuple[np.ndarray, dict]:
    g = np.asarray(grid, dtype=int)
    if g.ndim != 2:
        raise AssertionError("expected 2D grid")
    bg = _get_bg(g)
    sep_col = _find_separator_col(g)
    seed_row, seed_mask = _find_seed_row(g, sep_col)
    growth_only = _render_growth_only(g, sep_col, seed_row, seed_mask)
    bars = _extract_bars(g, sep_col, bg)
    ext, attempted = _extension_layer(g, sep_col, seed_row, seed_mask, bars)
    out = growth_only.copy()
    ext_mask = ext != -1
    out[ext_mask] = ext[ext_mask]
    details = {
        "bg": bg,
        "sep_col": sep_col,
        "seed_row": seed_row,
        "seed_mask": seed_mask,
        "growth_only": growth_only,
        "bars": bars,
        "ext": ext,
        "attempted": attempted,
        "growth_rows": _grow_rows(seed_mask, g.shape[0], seed_row),
    }
    return out, details


def main(grid: np.ndarray) -> np.ndarray:
    """
    Grow a left-side seed row downward while right-side bars periodically extend left and cut through the growth.

    Input structure:
    - The grid has green background, a full-height grey separator column, and an orange seed row on the left side.
    - The seed row is the only orange content in the input and may contain isolated pixels or short adjacent runs.
    - The right side contains 2 or 3 horizontal color bars in blue, red, or yellow, each with its own length.

    Rule:
    - First grow the left-side seed downward every other row: each active-background-active triplet creates a child
      in the middle two rows below, and generated row colors alternate pink then orange.
    - Then, for every right-side bar, use its length as a period and place that bar's color repeatedly to the left
      from the bar's left edge, continuing across the separator toward the left border.
    - Extension pixels overwrite generated growth pixels, but never overwrite the original orange seed cells, and the
      grey separator column itself stays grey.

    Key insight:
    - The final pattern is the superposition of two independent mechanisms: triangular midpoint growth on the left and
      periodic leftward bar echoes from the right, with the periodic echoes cutting visible holes through the growth.
    """
    out, _details = _simulate(grid)
    return out


def _choose_seed_mask(rng: random.Random, left_w: int, desired_levels: int, height: int, seed_row: int) -> tuple[np.ndarray, list[tuple[int, int, np.ndarray]]] | None:
    count = rng.randint(3, min(5, left_w))
    for _ in range(200):
        mode = rng.choice(["alternating", "mixed", "free"])
        positions: set[int]
        if mode == "alternating":
            candidates = list(range(rng.randint(0, 1), left_w, 2))
            if len(candidates) < count:
                continue
            positions = set(rng.sample(candidates, count))
            if rng.random() < 0.45:
                pivot = rng.choice(sorted(positions))
                shift = rng.choice([-1, 1])
                if 0 <= pivot + shift < left_w:
                    positions.add(pivot + shift)
        elif mode == "mixed":
            base = set(rng.sample(range(left_w), count))
            positions = set(base)
            if rng.random() < 0.6:
                pivot = rng.choice(sorted(base))
                for shift in (-1, 1):
                    if 0 <= pivot + shift < left_w and rng.random() < 0.5:
                        positions.add(pivot + shift)
        else:
            positions = set(rng.sample(range(left_w), count))
        if not (3 <= len(positions) <= min(5, left_w)):
            continue
        mask = np.zeros(left_w, dtype=bool)
        mask[list(sorted(positions))] = True
        rows = _grow_rows(mask, height, seed_row)
        if len(rows) < desired_levels:
            continue
        if not any(int(np.sum(row_mask)) > 0 for _, _, row_mask in rows):
            continue
        if not any(np.any(mask[:-1] & mask[1:]) for _ in [0]) and rng.random() < 0.35:
            continue
        return mask, rows
    return None


def _bar_candidates(
    target_cols: Iterable[int],
    sep_col: int,
    width: int,
    require_visible_non_target: bool,
) -> list[dict]:
    targets = set(int(c) for c in target_cols)
    candidates: list[dict] = []
    for length in range(2, min(4, width - (sep_col + 1)) + 1):
        left_start_max = min(width - length, sep_col + 5)
        for c0 in range(sep_col + 1, left_start_max + 1):
            positions: list[int] = []
            c = c0 - length
            while c >= 0:
                if c != sep_col:
                    positions.append(int(c))
                c -= length
            hit_targets = sorted(targets.intersection(positions))
            if not hit_targets:
                continue
            if require_visible_non_target and not any(pos not in targets for pos in positions):
                continue
            candidates.append(
                {
                    "c0": int(c0),
                    "length": int(length),
                    "distance": int(c0 - (sep_col + 1)),
                    "positions": positions,
                    "hits": hit_targets,
                }
            )
    candidates.sort(key=lambda cand: (-len(cand["hits"]), -len(cand["positions"]), cand["distance"], cand["c0"]))
    return candidates


def _pick_collision_bar(
    rng: random.Random,
    row: int,
    target_cols: Iterable[int],
    sep_col: int,
    width: int,
    used_rows: set[int],
    require_visible_non_target: bool,
) -> dict | None:
    if row in used_rows:
        return None
    candidates = _bar_candidates(target_cols, sep_col, width, require_visible_non_target)
    if not candidates:
        return None
    top = candidates[: min(10, len(candidates))]
    chosen = dict(rng.choice(top))
    chosen["row"] = int(row)
    return chosen


def _pick_extra_bar(
    rng: random.Random,
    row: int,
    sep_col: int,
    width: int,
    used_rows: set[int],
) -> dict | None:
    if row in used_rows:
        return None
    choices: list[dict] = []
    for length in range(2, min(4, width - (sep_col + 1)) + 1):
        left_start_max = min(width - length, sep_col + 5)
        for c0 in range(sep_col + 1, left_start_max + 1):
            positions: list[int] = []
            c = c0 - length
            while c >= 0:
                if c != sep_col:
                    positions.append(int(c))
                c -= length
            if positions:
                choices.append(
                    {
                        "row": int(row),
                        "c0": int(c0),
                        "length": int(length),
                        "distance": int(c0 - (sep_col + 1)),
                        "positions": positions,
                    }
                )
    if not choices:
        return None
    return dict(rng.choice(choices))


def _validate_generated_case(grid: np.ndarray) -> None:
    out, details = _simulate(grid)
    growth_only = details["growth_only"]
    ext = details["ext"]
    ext_mask = ext != -1
    bars = details["bars"]
    seed_row = int(details["seed_row"])
    seed_mask = np.asarray(details["seed_mask"], dtype=bool)
    attempted = set(details["attempted"])
    growth_rows = details["growth_rows"]

    assert len(bars) in (2, 3), "expected 2 or 3 bars"
    assert len({bar["row"] for bar in bars}) == len(bars), "bars must occupy different rows"
    assert len({bar["color"] for bar in bars}) == len(bars), "bar colors must be distinct"
    assert len({bar["distance"] for bar in bars}) >= 2, "need varied bar distances"
    assert growth_rows, "growth must produce at least one generated row"

    generated_positions = {
        (int(row), int(c))
        for row, _color, mask in growth_rows
        for c in np.flatnonzero(mask)
    }
    seed_positions = {(seed_row, int(c)) for c in np.flatnonzero(seed_mask)}

    overwritten_generated = {
        pos
        for pos in generated_positions
        if int(out[pos]) in BAR_COLORS and int(growth_only[pos]) in (PINK, SEED)
    }
    surviving_generated = {
        pos
        for pos in generated_positions
        if int(out[pos]) in (PINK, SEED)
    }
    background_extensions = {
        (int(r), int(c))
        for r, c in np.argwhere(ext_mask)
        if int(growth_only[r, c]) == BG
    }
    seed_hits = seed_positions.intersection(attempted)
    preserved_seed_hits = {pos for pos in seed_hits if int(out[pos]) == SEED}

    assert overwritten_generated, "need at least one real growth-overwrite collision"
    assert surviving_generated, "need some generated growth pixels to remain visible"
    assert background_extensions, "need periodic pixels outside the growth"
    assert preserved_seed_hits, "need an attempted seed overwrite that remains orange"
    assert np.any(out != growth_only), "extension layer must change the growth-only result"
    assert np.any(out != grid), "output must differ from input"


def _gen_attempt(rng: random.Random) -> np.ndarray | None:
    height = rng.randint(13, 18)
    left_w = rng.randint(6, 10)
    right_w = rng.randint(6, 10)
    width = left_w + 1 + right_w
    if width > 30:
        return None

    seed_row = 1
    max_levels = (height - 1 - seed_row) // 2
    if max_levels < 1:
        return None
    desired_levels = rng.choices(
        population=list(range(1, min(4, max_levels) + 1)),
        weights=[2, 4, 3, 1][: min(4, max_levels)],
        k=1,
    )[0]

    seed_result = _choose_seed_mask(rng, left_w, desired_levels, height, seed_row)
    if seed_result is None:
        return None
    seed_mask, growth_rows = seed_result
    generated_rows = [(row, mask) for row, _color, mask in growth_rows if np.any(mask)]
    if not generated_rows:
        return None

    grid = np.full((height, width), BG, dtype=int)
    sep_col = left_w
    grid[:, sep_col] = SEP
    grid[seed_row, np.flatnonzero(seed_mask)] = SEED

    bars: list[dict] = []
    used_rows: set[int] = set()

    collision_row, collision_mask = rng.choice(generated_rows)
    collision_bar = _pick_collision_bar(
        rng,
        collision_row,
        np.flatnonzero(collision_mask),
        sep_col,
        width,
        used_rows,
        require_visible_non_target=False,
    )
    if collision_bar is None:
        return None
    bars.append(collision_bar)
    used_rows.add(int(collision_row))

    seed_bar = _pick_collision_bar(
        rng,
        seed_row,
        np.flatnonzero(seed_mask),
        sep_col,
        width,
        used_rows,
        require_visible_non_target=True,
    )
    if seed_bar is None:
        return None
    bars.append(seed_bar)
    used_rows.add(seed_row)

    want_third = rng.random() < 0.6
    if want_third:
        non_active_rows = [r for r in range(height) if r not in used_rows and r != seed_row and all(r != grow_row for grow_row, _ in generated_rows)]
        other_growth_rows = [r for r, _mask in generated_rows if r not in used_rows]
        choice_mode = rng.choice(["non_active", "other_growth"]) if non_active_rows and other_growth_rows else ("non_active" if non_active_rows else "other_growth")
        extra_bar = None
        if choice_mode == "other_growth":
            extra_row = rng.choice(other_growth_rows)
            extra_mask = next(mask for row, mask in generated_rows if row == extra_row)
            extra_bar = _pick_collision_bar(
                rng,
                extra_row,
                np.flatnonzero(extra_mask),
                sep_col,
                width,
                used_rows,
                require_visible_non_target=False,
            )
        elif choice_mode == "non_active":
            extra_row = rng.choice(non_active_rows)
            extra_bar = _pick_extra_bar(rng, extra_row, sep_col, width, used_rows)
        if extra_bar is None:
            return None
        bars.append(extra_bar)
        used_rows.add(int(extra_bar["row"]))

    if len(bars) not in (2, 3):
        return None

    colors = list(BAR_COLORS)
    rng.shuffle(colors)
    for bar, color in zip(bars, colors):
        bar["color"] = int(color)
        c0 = int(bar["c0"])
        c1 = c0 + int(bar["length"])
        if c1 > width:
            return None
        grid[int(bar["row"]), c0:c1] = int(color)

    try:
        _validate_generated_case(grid)
    except AssertionError:
        return None
    return grid


def generate_input() -> np.ndarray:
    rng = random.Random()
    for _ in range(MAX_GENERATION_ATTEMPTS):
        grid = _gen_attempt(rng)
        if grid is not None:
            return grid
    raise RuntimeError("failed to generate a valid input for puzzle 7")
