from __future__ import annotations

import random
from collections import Counter, defaultdict

import numpy as np


BG = 0
SEPARATOR = 5
DECORATION = 9
POLYLINE_COLORS = (1, 2, 3, 4, 6, 7, 8)
MAX_ATTEMPTS = 400


def _most_common_color(grid: np.ndarray) -> int:
    values, counts = np.unique(np.asarray(grid, dtype=int), return_counts=True)
    return int(values[int(np.argmax(counts))])


def _mirror(strip: np.ndarray) -> np.ndarray:
    return np.fliplr(strip).copy()


def _edge_profile(strip: np.ndarray, side: str, bg: int) -> tuple[tuple[int, int], ...]:
    col = 0 if side == "left" else strip.shape[1] - 1
    profile = [(r, int(strip[r, col])) for r in range(strip.shape[0]) if int(strip[r, col]) != bg]
    return tuple(sorted(profile))


def _non_bg_neighbors4(strip: np.ndarray, r: int, c: int, bg: int) -> int:
    total = 0
    for dr, dc in ((-1, 0), (1, 0), (0, -1), (0, 1)):
        rr, cc = r + dr, c + dc
        if 0 <= rr < strip.shape[0] and 0 <= cc < strip.shape[1] and int(strip[rr, cc]) != bg:
            total += 1
    return total


def _isolated_rows(strip: np.ndarray, bg: int) -> tuple[int, ...]:
    rows = []
    for r in range(1, strip.shape[0] - 1):
        for c in range(1, strip.shape[1] - 1):
            if int(strip[r, c]) == bg:
                continue
            if _non_bg_neighbors4(strip, r, c, bg) == 0:
                rows.append(r)
    return tuple(sorted(rows))


def _split_strips(grid: np.ndarray) -> tuple[list[np.ndarray], int, int]:
    g = np.asarray(grid, dtype=int)
    h, w = g.shape

    uniform_cols: dict[int, list[int]] = defaultdict(list)
    for c in range(w):
        val = int(g[0, c])
        if val != BG and np.all(g[:, c] == val):
            uniform_cols[val].append(c)

    assert uniform_cols, "expected full-height separator columns"
    sep_color, sep_cols = max(uniform_cols.items(), key=lambda item: len(item[1]))
    sep_cols = sorted(sep_cols)
    assert len(sep_cols) >= 2, "expected at least three strips"
    assert all(b - a > 1 for a, b in zip(sep_cols, sep_cols[1:])), "separator columns must be width 1"

    spans = []
    start = 0
    for c in sep_cols:
        spans.append((start, c))
        start = c + 1
    spans.append((start, w))

    widths = [c1 - c0 for c0, c1 in spans]
    assert min(widths) > 0 and len(set(widths)) == 1, "expected equal-width strips"
    strips = [g[:, c0:c1].copy() for c0, c1 in spans]
    return strips, int(sep_color), widths[0]


def _oriented_variants(strips: list[np.ndarray], bg: int) -> dict[tuple[int, int], dict]:
    variants: dict[tuple[int, int], dict] = {}
    for idx, strip in enumerate(strips):
        for flipped in (0, 1):
            view = _mirror(strip) if flipped else strip.copy()
            variants[(idx, flipped)] = {
                "strip": view,
                "left": _edge_profile(view, "left", bg),
                "right": _edge_profile(view, "right", bg),
                "isolated_rows": _isolated_rows(view, bg),
            }
    return variants


def _enumerate_solutions(strips: list[np.ndarray], bg: int) -> tuple[list[tuple[tuple[int, int], ...]], dict]:
    variants = _oriented_variants(strips, bg)
    left_index: dict[tuple[tuple[int, int], ...], list[tuple[int, int]]] = defaultdict(list)
    for key, info in variants.items():
        left_index[info["left"]].append(key)

    n = len(strips)
    solutions: list[tuple[tuple[int, int], ...]] = []

    def dfs(path: list[tuple[int, int]], used_mask: int) -> None:
        if len(path) == n:
            solutions.append(tuple(path))
            return

        last = path[-1]
        for nxt in left_index[variants[last]["right"]]:
            idx = nxt[0]
            if used_mask & (1 << idx):
                continue
            path.append(nxt)
            dfs(path, used_mask | (1 << idx))
            path.pop()

    for start in variants:
        dfs([start], 1 << start[0])

    deduped = list(dict.fromkeys(solutions))
    return deduped, variants


def _guide_rows(solution: tuple[tuple[int, int], ...], variants: dict[tuple[int, int], dict]) -> tuple[int, ...] | None:
    rows = []
    for key in solution:
        isolated = variants[key]["isolated_rows"]
        if len(isolated) != 1:
            return None
        rows.append(int(isolated[0]))
    return tuple(rows)


def _is_strictly_increasing(values: tuple[int, ...] | None) -> bool:
    return values is not None and all(a < b for a, b in zip(values, values[1:]))


def _choose_solution(
    solutions: list[tuple[tuple[int, int], ...]],
    variants: dict[tuple[int, int], dict],
) -> tuple[tuple[int, int], ...]:
    assert solutions, "no continuity-preserving assembly found"
    if len(solutions) == 1:
        return solutions[0]

    increasing = [sol for sol in solutions if _is_strictly_increasing(_guide_rows(sol, variants))]
    assert len(increasing) == 1, "assembly must be uniquely determined"
    return increasing[0]


def _assemble(solution: tuple[tuple[int, int], ...], variants: dict[tuple[int, int], dict], sep_color: int, bg: int) -> np.ndarray:
    h, strip_w = variants[solution[0]]["strip"].shape
    n = len(solution)
    out = np.full((h, n * strip_w + (n - 1)), int(bg), dtype=int)
    for i, key in enumerate(solution):
        c0 = i * (strip_w + 1)
        out[:, c0 : c0 + strip_w] = variants[key]["strip"]
        if i + 1 < n:
            out[:, c0 + strip_w] = int(sep_color)
    return out


def main(grid: np.ndarray) -> np.ndarray:
    """
    Restore the scrambled vertical strips so the multicolor cut profiles reconnect across every separator.

    Input structure:
    - The grid is partitioned by full-height gray separator columns into equal-width vertical strips.
    - Each strip contains part of a multicolor line pattern whose only edge-touching cells lie on the strip's left and right borders.
    - Each strip also carries isolated non-edge decoration pixels that stay attached to that strip.

    Rule:
    - Treat every strip as having two possible poses: as given or horizontally mirrored.
    - Strip B can stand immediately to the right of strip A exactly when A's right-edge (row, color) profile equals B's left-edge profile.
    - Reorder the strips and choose their poses so all separator crossings reconnect; if the continuity chain has the unavoidable global mirror twin, the isolated decorations fix the left-to-right reading.

    Key insight:
    - The broken line leaves a complete signature on every cut edge, so solving the task is a constrained path search over strip profiles plus per-strip mirroring.
    """
    g = np.asarray(grid, dtype=int)
    assert g.ndim == 2

    bg = _most_common_color(g)
    strips, sep_color, _strip_w = _split_strips(g)
    solutions, variants = _enumerate_solutions(strips, bg)
    chosen = _choose_solution(solutions, variants)
    return _assemble(chosen, variants, sep_color, bg)


def _valid_layouts() -> list[tuple[int, int, int]]:
    layouts = []
    for strip_count in range(4, 7):
        for strip_w in range(4, 7):
            total_w = strip_count * strip_w + (strip_count - 1)
            if total_w > 30:
                continue
            for height in range(max(11, strip_count + 7), 23):
                layouts.append((height, strip_count, strip_w))
    return layouts


LAYOUTS = _valid_layouts()


def _random_profile(height: int, rng: random.Random) -> tuple[tuple[int, int], ...]:
    size = rng.choice((1, 1, 2, 2, 3))
    rows = sorted(rng.sample(range(1, height - 1), size))
    return tuple(sorted((r, rng.choice(POLYLINE_COLORS)) for r in rows))


def _sample_profile_chain(height: int, strip_count: int, rng: random.Random) -> list[tuple[tuple[int, int], ...]] | None:
    for _ in range(200):
        chain: list[tuple[tuple[int, int], ...]] = []
        used: set[tuple[tuple[int, int], ...]] = set()
        ok = True
        for i in range(strip_count + 1):
            chosen = None
            for _ in range(200):
                candidate = _random_profile(height, rng)
                if candidate in used:
                    continue
                if i > 0:
                    prev_rows = {r for r, _ in chain[-1]}
                    curr_rows = {r for r, _ in candidate}
                    if prev_rows & curr_rows:
                        continue
                chosen = candidate
                break
            if chosen is None:
                ok = False
                break
            chain.append(chosen)
            used.add(chosen)
        if ok and any(len(profile) > 1 for profile in chain):
            return chain
    return None


def _render_strip(
    height: int,
    strip_w: int,
    left_profile: tuple[tuple[int, int], ...],
    right_profile: tuple[tuple[int, int], ...],
    rng: random.Random,
) -> np.ndarray:
    strip = np.zeros((height, strip_w), dtype=int)
    spine_col = rng.randint(1, strip_w - 2)
    all_rows = sorted({r for r, _ in left_profile + right_profile})
    lo, hi = min(all_rows), max(all_rows)

    for r, color in left_profile:
        strip[r, : spine_col + 1] = int(color)
    for r, color in right_profile:
        strip[r, spine_col:] = int(color)
    for r in range(lo, hi + 1):
        if strip[r, spine_col] == 0:
            strip[r, spine_col] = rng.choice(POLYLINE_COLORS)

    if strip_w >= 5 and len(all_rows) >= 2 and rng.random() < 0.6:
        extra_col = rng.choice([c for c in range(1, strip_w - 1) if c != spine_col])
        a, b = rng.sample(all_rows, 2)
        r0, r1 = sorted((a, b))
        if extra_col < spine_col:
            strip[r0 : r1 + 1, extra_col] = rng.choice(POLYLINE_COLORS)
            strip[a, extra_col : spine_col + 1] = rng.choice(POLYLINE_COLORS)
        else:
            strip[r0 : r1 + 1, extra_col] = rng.choice(POLYLINE_COLORS)
            strip[b, spine_col : extra_col + 1] = rng.choice(POLYLINE_COLORS)

    return strip


def _isolated_slots_by_row(strip: np.ndarray) -> dict[int, list[int]]:
    slots: dict[int, list[int]] = defaultdict(list)
    h, w = strip.shape
    for r in range(1, h - 1):
        for c in range(1, w - 1):
            if int(strip[r, c]) != BG:
                continue
            clear = True
            for dr in (-1, 0, 1):
                for dc in (-1, 0, 1):
                    if dr == 0 and dc == 0:
                        continue
                    rr, cc = r + dr, c + dc
                    if int(strip[rr, cc]) != BG:
                        clear = False
                        break
                if not clear:
                    break
            if clear:
                slots[r].append(c)
    return slots


def _choose_increasing_rows(candidates: list[dict[int, list[int]]], rng: random.Random) -> list[tuple[int, int]] | None:
    chosen: list[tuple[int, int]] = []

    def dfs(i: int, prev_row: int) -> bool:
        if i == len(candidates):
            return True
        rows = [row for row in sorted(candidates[i]) if row > prev_row]
        rng.shuffle(rows)
        for row in rows:
            cols = candidates[i][row][:]
            rng.shuffle(cols)
            chosen.append((row, cols[0]))
            if dfs(i + 1, row):
                return True
            chosen.pop()
        return False

    if dfs(0, -1):
        return chosen
    return None


def _assemble_plain(strips: list[np.ndarray], sep_color: int = SEPARATOR, bg: int = BG) -> np.ndarray:
    h, strip_w = strips[0].shape
    out = np.full((h, len(strips) * strip_w + (len(strips) - 1)), int(bg), dtype=int)
    for i, strip in enumerate(strips):
        c0 = i * (strip_w + 1)
        out[:, c0 : c0 + strip_w] = strip
        if i + 1 < len(strips):
            out[:, c0 + strip_w] = int(sep_color)
    return out


def _solution_to_grid(solution: tuple[tuple[int, int], ...], variants: dict[tuple[int, int], dict]) -> np.ndarray:
    return _assemble(solution, variants, SEPARATOR, BG)


def _validate_semantics(inp: np.ndarray, expected: np.ndarray) -> None:
    solved = main(inp)
    assert np.array_equal(solved, expected), "main() failed on generated sample"

    strips, sep_color, _strip_w = _split_strips(inp)
    assert sep_color == SEPARATOR
    solutions, variants = _enumerate_solutions(strips, BG)
    assert len(solutions) == 2, "generator should keep exactly the global mirror twin"

    increasing = [sol for sol in solutions if _is_strictly_increasing(_guide_rows(sol, variants))]
    assert len(increasing) == 1, "decorations must remove the mirror ambiguity"

    candidate_grids = [_solution_to_grid(sol, variants) for sol in solutions]
    assert len({grid.tobytes() for grid in candidate_grids}) == 2, "continuity solutions must differ"
    assert np.array_equal(_solution_to_grid(increasing[0], variants), expected)

    first_solution = increasing[0]
    edge_sizes = []
    for key in first_solution:
        edge_sizes.append(len(variants[key]["left"]))
        edge_sizes.append(len(variants[key]["right"]))
    assert max(edge_sizes) >= 2, "need multi-pixel edge profiles in generated tasks"

    assert isinstance(solved, np.ndarray) and solved.ndim == 2
    assert max(inp.shape) <= 30 and max(expected.shape) <= 30


def _attempt_generate(rng: random.Random) -> np.ndarray | None:
    height, strip_count, strip_w = rng.choice(LAYOUTS)
    profiles = _sample_profile_chain(height, strip_count, rng)
    if profiles is None:
        return None

    original = [
        _render_strip(height, strip_w, profiles[i], profiles[i + 1], rng)
        for i in range(strip_count)
    ]

    slots = [_isolated_slots_by_row(strip) for strip in original]
    chosen = _choose_increasing_rows(slots, rng)
    if chosen is None:
        return None

    for strip, (row, col) in zip(original, chosen):
        strip[row, col] = DECORATION

    expected = _assemble_plain(original)

    perm = list(range(strip_count))
    rng.shuffle(perm)
    flips = [rng.random() < 0.6 for _ in range(strip_count)]
    if perm == list(range(strip_count)) and not any(flips):
        flips[rng.randrange(strip_count)] = True

    scrambled = []
    for idx in perm:
        piece = original[idx].copy()
        if flips[idx]:
            piece = _mirror(piece)
        scrambled.append(piece)
    inp = _assemble_plain(scrambled)

    if np.array_equal(inp, expected):
        return None

    try:
        _validate_semantics(inp, expected)
    except AssertionError:
        return None
    return inp


def generate_input() -> np.ndarray:
    rng = random.Random()
    for _ in range(MAX_ATTEMPTS):
        grid = _attempt_generate(rng)
        if grid is not None:
            return grid
    raise RuntimeError("failed to generate a valid strip-reassembly sample")
