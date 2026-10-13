import math
import random
from collections import Counter
from itertools import product

import numpy as np


MASK = 5
PALETTE = (1, 2, 3, 4, 7)
GEN_ATTEMPTS = 500


def _strict_desc(counts: dict[int, int], ranked_colors: list[int]) -> bool:
    values = [int(counts[color]) for color in ranked_colors]
    return all(values[i] > values[i + 1] for i in range(len(values) - 1))


def _mirror_orbits(grid: np.ndarray) -> tuple[dict[int, list[tuple[tuple[int, int], tuple[int, int]]]], dict[int, list[tuple[int, int]]]]:
    h, w = map(int, grid.shape)
    pair_orbits: dict[int, list[tuple[tuple[int, int], tuple[int, int]]]] = {}
    center_cells: dict[int, list[tuple[int, int]]] = {}
    for r in range(h):
        for c in range(w // 2):
            mc = w - 1 - c
            color = int(grid[r, c])
            assert color == int(grid[r, mc]), "expected an already symmetric output grid"
            pair_orbits.setdefault(color, []).append(((r, c), (r, mc)))
        if w % 2 == 1:
            color = int(grid[r, w // 2])
            center_cells.setdefault(color, []).append((r, w // 2))
    return pair_orbits, center_cells


def _analyze_input(grid: np.ndarray):
    g = np.asarray(grid, dtype=int)
    h, w = map(int, g.shape)

    visible_counts = Counter(int(value) for value in g.ravel() if int(value) != MASK)
    assert 3 <= len(visible_counts) <= 4, "expected 3-4 visible non-grey colors"
    ranked_colors = sorted(visible_counts, key=lambda color: -visible_counts[color])
    assert len({visible_counts[color] for color in ranked_colors}) == len(ranked_colors), "visible color counts must have a unique rank"

    base_counts = {color: int(visible_counts[color]) for color in ranked_colors}
    forced: list[tuple[int, int, int]] = []
    groups: list[dict[str, object]] = []

    for r in range(h):
        for c in range(w // 2):
            mc = w - 1 - c
            left = int(g[r, c])
            right = int(g[r, mc])
            if left != MASK and right != MASK:
                assert left == right, "visible mirror cells must already agree"
            elif left == MASK and right == MASK:
                groups.append({"positions": ((r, c), (r, mc)), "weight": 2})
            else:
                color = right if left == MASK else left
                pos = (r, c) if left == MASK else (r, mc)
                assert color in base_counts, "forced mirror fill must use an existing visible color"
                base_counts[color] += 1
                forced.append((pos[0], pos[1], color))

        if w % 2 == 1 and int(g[r, w // 2]) == MASK:
            groups.append({"positions": ((r, w // 2),), "weight": 1})

    return g, ranked_colors, base_counts, forced, groups


def _solve_ranked_groups(
    ranked_colors: list[int],
    base_counts: dict[int, int],
    groups: list[dict[str, object]],
    *,
    limit: int = 2,
) -> list[tuple[int, ...]]:
    if not groups:
        return [tuple()]

    color_to_index = {color: i for i, color in enumerate(ranked_colors)}
    weights = [int(group["weight"]) for group in groups]
    remaining = [0] * (len(weights) + 1)
    for i in range(len(weights) - 1, -1, -1):
        remaining[i] = remaining[i + 1] + weights[i]

    increments = [0] * len(ranked_colors)
    solutions: list[tuple[int, ...]] = []
    assignment: list[int] = []

    def dfs(i: int) -> None:
        if len(solutions) >= limit:
            return

        current = [base_counts[color] + increments[j] for j, color in enumerate(ranked_colors)]
        rem = remaining[i]
        for j in range(len(current) - 1):
            if current[j] + rem <= current[j + 1]:
                return

        if i == len(groups):
            if all(current[j] > current[j + 1] for j in range(len(current) - 1)):
                solutions.append(tuple(assignment))
            return

        weight = weights[i]
        for color in ranked_colors:
            idx = color_to_index[color]
            increments[idx] += weight
            assignment.append(color)
            dfs(i + 1)
            assignment.pop()
            increments[idx] -= weight
            if len(solutions) >= limit:
                return

    dfs(0)
    return solutions


def main(grid: np.ndarray) -> np.ndarray:
    """Fill grey cells so the completed grid is left-right symmetric and keeps the visible color-frequency ranking unchanged.

    Input structure:
    - The grid contains only colored cells and grey damage cells (color 5); there is no black background.
    - Visible non-grey cells use 3-4 colors and some mirror positions may be hidden by grey.
    - Grey cells can appear singly against a visible mirror or as fully grey mirror pairs, and odd-width grids may also have grey center-column cells.

    Rule:
    - Rank the visible non-grey colors by their input frequencies from most frequent to least frequent.
    - Fill every grey cell while preserving all visible cells and enforcing exact left-right mirror symmetry.
    - When symmetry alone leaves a grey mirror pair or grey center cell ambiguous, choose the unique color assignment that keeps the completed grid's frequency ranking identical to the visible-input ranking.

    Key insight:
    - One-sided grey damage is fixed locally by mirror copying, while fully hidden mirror orbits require a global frequency-order constraint to break the remaining symmetry ambiguity.
    """
    g, ranked_colors, base_counts, forced, groups = _analyze_input(grid)

    solutions = _solve_ranked_groups(ranked_colors, base_counts, groups, limit=2)
    assert len(solutions) == 1, "expected a unique completion under symmetry and rank preservation"

    out = g.copy()
    for r, c, color in forced:
        out[r, c] = int(color)

    for group, color in zip(groups, solutions[0]):
        for r, c in group["positions"]:
            out[r, c] = int(color)

    assert not np.any(out == MASK), "all grey cells must be filled"
    h, w = map(int, out.shape)
    for r in range(h):
        for c in range(w):
            assert int(out[r, c]) == int(out[r, w - 1 - c]), "output must be mirror-symmetric"

    final_counts = Counter(map(int, out.ravel()))
    final_rank_counts = {color: int(final_counts[color]) for color in ranked_colors}
    assert _strict_desc(final_rank_counts, ranked_colors), "completed counts must keep the visible ranking"
    assert np.all((g == MASK) | (g == out)), "visible cells must stay unchanged"
    return out


def _make_half_pattern(height: int, rep_width: int, colors: list[int], rng: np.random.Generator) -> np.ndarray:
    weights = rng.integers(1, 6, size=len(colors))
    probs = weights / weights.sum()
    half = rng.choice(colors, size=(height, rep_width), p=probs).astype(int)

    for _ in range(3):
        updated = half.copy()
        for r in range(height):
            for c in range(rep_width):
                r0 = max(0, r - 1)
                r1 = min(height, r + 2)
                c0 = max(0, c - 1)
                c1 = min(rep_width, c + 2)
                neighborhood = half[r0:r1, c0:c1].ravel()
                counts = Counter(map(int, neighborhood))
                best = max(counts.values())
                winners = [color for color, count in counts.items() if count == best]
                if int(half[r, c]) in winners and rng.random() < 0.65:
                    updated[r, c] = int(half[r, c])
                else:
                    updated[r, c] = int(rng.choice(winners))
        half = updated
    return half


def _mirror_half(half: np.ndarray, full_width: int) -> np.ndarray:
    if full_width % 2 == 0:
        return np.concatenate([half, np.fliplr(half)], axis=1)
    left = half[:, : full_width // 2]
    center = half[:, full_width // 2 : full_width // 2 + 1]
    return np.concatenate([left, center, np.fliplr(left)], axis=1)


def _build_output(rng: np.random.Generator) -> np.ndarray | None:
    height = int(rng.integers(8, 10))
    width = int(rng.choice([9, 10]))
    n_colors = 3
    colors = rng.choice(PALETTE, size=n_colors, replace=False).tolist()
    rep_width = width // 2 if width % 2 == 0 else width // 2 + 1

    for _ in range(80):
        half = _make_half_pattern(height, rep_width, colors, rng)
        out = _mirror_half(half, width)
        counts = Counter(map(int, out.ravel()))
        if set(counts) != set(colors):
            continue
        if min(counts.values()) < max(4, height):
            continue
        if len(set(counts.values())) != len(colors):
            continue
        return out
    return None


def _search_hide_plan(output: np.ndarray, rng: np.random.Generator):
    counts = Counter(map(int, output.ravel()))
    ranked_colors = sorted(counts, key=lambda color: -counts[color])
    pair_orbits, center_cells = _mirror_orbits(output)
    top = ranked_colors[0]
    area = int(output.size)
    min_hidden = math.ceil(area * 0.10)
    max_hidden = math.floor(area * 0.25)

    candidates: list[tuple[str, int]] = []
    if pair_orbits.get(top):
        candidates.append(("pair", 2))
    if output.shape[1] % 2 == 1 and center_cells.get(top):
        candidates.append(("center", 1))
    rng.shuffle(candidates)

    for group_type, weight in candidates:
        max_direct = {
            color: len(pair_orbits.get(color, [])) - (1 if group_type == "pair" and color == top else 0)
            for color in ranked_colors
        }
        gap_choices = list(product(range(1, weight + 1), repeat=len(ranked_colors) - 1))
        rng.shuffle(gap_choices)

        for gaps in gap_choices:
            prefix = [0]
            for gap in gaps:
                prefix.append(prefix[-1] + int(gap))

            top_visible_max = min(
                int(counts[color]) - (weight if i == 0 else 0) + prefix[i]
                for i, color in enumerate(ranked_colors)
            )
            top_visible_min = prefix[-1] + 1
            if top_visible_min > top_visible_max:
                continue

            top_values = list(range(top_visible_min, top_visible_max + 1))
            rng.shuffle(top_values)
            for top_visible in top_values:
                visible = [top_visible - prefix[i] for i in range(len(ranked_colors))]
                if visible[-1] <= 0:
                    continue

                direct_counts: dict[int, int] = {}
                ok = True
                for i, color in enumerate(ranked_colors):
                    extra = weight if i == 0 else 0
                    direct = int(counts[color]) - int(visible[i]) - extra
                    if direct < 0 or direct > max_direct[color]:
                        ok = False
                        break
                    direct_counts[color] = direct
                if not ok:
                    continue

                direct_total = sum(direct_counts.values())
                if group_type == "center" and direct_total < 2:
                    continue
                if direct_total == 0:
                    continue
                hidden_total = direct_total + weight
                if not (min_hidden <= hidden_total <= max_hidden):
                    continue

                return {
                    "ranked_colors": ranked_colors,
                    "group_type": group_type,
                    "weight": weight,
                    "direct_counts": direct_counts,
                }
    return None


def _gray_spread_ok(grid: np.ndarray) -> bool:
    coords = np.argwhere(grid == MASK)
    if coords.size == 0:
        return False
    h, w = map(int, grid.shape)
    n = int(coords.shape[0])
    if not (math.ceil(grid.size * 0.10) <= n <= math.floor(grid.size * 0.25)):
        return False

    rows = {int(r) for r, _c in coords}
    cols = {int(c) for _r, c in coords}
    if n >= 3 and len(rows) < 3:
        return False
    if n >= 2 and len(cols) < 2:
        return False

    mid = (w - 1) / 2.0
    has_left = any(int(c) < mid for _r, c in coords)
    has_right = any(int(c) > mid for _r, c in coords)
    return has_left and has_right


def _apply_hide_plan(output: np.ndarray, plan, rng: np.random.Generator) -> tuple[np.ndarray, tuple[int, ...]] | None:
    inp = output.copy()
    pair_orbits, center_cells = _mirror_orbits(output)
    ranked_colors = plan["ranked_colors"]
    top = ranked_colors[0]

    intended: list[int] = []
    if plan["group_type"] == "pair":
        ambiguous_pair = random.choice(pair_orbits[top])
        for r, c in ambiguous_pair:
            inp[r, c] = MASK
        blocked = {ambiguous_pair}
        intended.append(top)
    else:
        ambiguous_cell = random.choice(center_cells[top])
        inp[ambiguous_cell[0], ambiguous_cell[1]] = MASK
        blocked = set()
        intended.append(top)

    direct_orbits: list[tuple[int, tuple[tuple[int, int], tuple[int, int]]]] = []
    for color in ranked_colors:
        need = int(plan["direct_counts"][color])
        if need == 0:
            continue
        pool = [orbit for orbit in pair_orbits.get(color, []) if orbit not in blocked]
        if len(pool) < need:
            return None
        chosen_indices = rng.choice(len(pool), size=need, replace=False)
        direct_orbits.extend((color, pool[int(i)]) for i in chosen_indices)

    rng.shuffle(direct_orbits)
    if len(direct_orbits) == 1:
        sides = [random.choice([0, 1])]
    else:
        left_count = len(direct_orbits) // 2
        sides = [0] * left_count + [1] * (len(direct_orbits) - left_count)
        rng.shuffle(sides)

    for (_color, orbit), side in zip(direct_orbits, sides):
        r, c = orbit[int(side)]
        inp[r, c] = MASK

    if not _gray_spread_ok(inp):
        return None
    return inp, tuple(intended)


def _validate_instance(inp: np.ndarray, out: np.ndarray, intended: tuple[int, ...]) -> bool:
    analyzed = _analyze_input(inp)
    _g, ranked_colors, base_counts, _forced, groups = analyzed
    if not groups:
        return False
    if len(groups) != len(intended):
        return False
    if len(ranked_colors) ** len(groups) <= 1:
        return False

    solutions = _solve_ranked_groups(ranked_colors, base_counts, groups, limit=2)
    if len(solutions) != 1 or solutions[0] != tuple(int(color) for color in intended):
        return False
    if not np.array_equal(main(inp), out):
        return False

    visible_counts = Counter(int(value) for value in inp.ravel() if int(value) != MASK)
    final_counts = Counter(map(int, out.ravel()))
    visible_rank = sorted(visible_counts, key=lambda color: -visible_counts[color])
    if len({visible_counts[color] for color in visible_rank}) != len(visible_rank):
        return False
    if not _strict_desc({color: final_counts[color] for color in visible_rank}, visible_rank):
        return False
    return True


def _gen_attempt() -> np.ndarray | None:
    rng = np.random.default_rng()
    output = _build_output(rng)
    if output is None:
        return None

    plan = _search_hide_plan(output, rng)
    if plan is None:
        return None

    applied = _apply_hide_plan(output, plan, rng)
    if applied is None:
        return None
    inp, intended = applied

    if not _validate_instance(inp, output, intended):
        return None
    return inp


def generate_input() -> np.ndarray:
    for _ in range(GEN_ATTEMPTS):
        grid = _gen_attempt()
        if grid is not None:
            return grid
    raise RuntimeError("failed to generate a valid symmetry-and-frequency ARC input")
