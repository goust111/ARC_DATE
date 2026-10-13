import random
from collections import Counter, defaultdict

import numpy as np


BG = 0
PALETTE = tuple(range(1, 10))
GEN_RETRIES = 300


def _max_true_run(mask: np.ndarray) -> int:
    best = 0
    cur = 0
    for value in mask:
        if bool(value):
            cur += 1
            if cur > best:
                best = cur
        else:
            cur = 0
    return best


def _detect_horizontal_lines(grid: np.ndarray) -> list[dict[str, int]]:
    h, _ = grid.shape
    lines: list[dict[str, int]] = []
    for r in range(h):
        row = grid[r]
        counts = Counter(int(v) for v in row if int(v) != BG)
        best = None
        for color, count in counts.items():
            if count < 3:
                continue
            mask = row == int(color)
            run = _max_true_run(mask)
            if run < 2:
                continue
            cols = np.flatnonzero(mask)
            c0 = int(cols[0])
            c1 = int(cols[-1])
            if c1 - c0 + 1 < 3:
                continue
            score = (int(count), int(run), int(c1 - c0))
            candidate = {"row": int(r), "color": int(color), "c0": c0, "c1": c1}
            if best is None or score > best[0]:
                best = (score, candidate)
        if best is not None:
            lines.append(best[1])
    return lines


def _detect_vertical_lines(grid: np.ndarray) -> list[dict[str, int]]:
    _, w = grid.shape
    lines: list[dict[str, int]] = []
    for c in range(w):
        col = grid[:, c]
        counts = Counter(int(v) for v in col if int(v) != BG)
        best = None
        for color, count in counts.items():
            if count < 3:
                continue
            mask = col == int(color)
            run = _max_true_run(mask)
            if run < 2:
                continue
            rows = np.flatnonzero(mask)
            r0 = int(rows[0])
            r1 = int(rows[-1])
            if r1 - r0 + 1 < 3:
                continue
            score = (int(count), int(run), int(r1 - r0))
            candidate = {"col": int(c), "color": int(color), "r0": r0, "r1": r1}
            if best is None or score > best[0]:
                best = (score, candidate)
        if best is not None:
            lines.append(best[1])
    return lines


def main(grid: np.ndarray) -> np.ndarray:
    """
    Fill missing orthogonal-path intersections by inferring the per-color-pair crossing table.
    """
    g = np.asarray(grid, dtype=int)
    out = g.copy()

    horizontal = _detect_horizontal_lines(g)
    vertical = _detect_vertical_lines(g)

    observed: dict[tuple[int, int], Counter[int]] = defaultdict(Counter)
    crossings: list[tuple[int, int, tuple[int, int]]] = []

    for hline in horizontal:
        r = int(hline["row"])
        hcol = int(hline["color"])
        c0 = int(hline["c0"])
        c1 = int(hline["c1"])
        for vline in vertical:
            c = int(vline["col"])
            if not (c0 <= c <= c1):
                continue
            r0 = int(vline["r0"])
            r1 = int(vline["r1"])
            if not (r0 <= r <= r1):
                continue
            key = (hcol, int(vline["color"]))
            crossings.append((r, c, key))
            value = int(g[r, c])
            if value != BG:
                observed[key][value] += 1

    lookup = {key: counts.most_common(1)[0][0] for key, counts in observed.items()}

    for r, c, key in crossings:
        if int(out[r, c]) == BG and key in lookup:
            out[r, c] = int(lookup[key])

    return out


def _sample_spaced_positions(rng: random.Random, lo: int, hi: int, count: int, min_gap: int) -> list[int]:
    max_base_hi = int(hi) - (int(min_gap) - 1) * (int(count) - 1)
    if max_base_hi < lo:
        raise ValueError("range too small for spaced positions")
    bases = sorted(rng.sample(range(int(lo), int(max_base_hi) + 1), int(count)))
    return [int(base + i * (int(min_gap) - 1)) for i, base in enumerate(bases)]


def _pair_outputs(
    rng: random.Random,
    horizontal_colors: list[int],
    vertical_colors: list[int],
    extra_colors: list[int],
) -> dict[tuple[int, int], int]:
    pairs = [(h, v) for h in horizontal_colors for v in vertical_colors]
    rng.shuffle(pairs)

    outputs: dict[tuple[int, int], int] = {}
    if pairs:
        first = pairs.pop()
        outputs[first] = int(first[0])
    if pairs:
        second = pairs.pop()
        outputs[second] = int(second[1])
    if pairs:
        third = pairs.pop()
        outputs[third] = int(rng.choice(extra_colors))

    for pair in pairs:
        mode = rng.choice(["h", "v", "third"])
        if mode == "h":
            outputs[pair] = int(pair[0])
        elif mode == "v":
            outputs[pair] = int(pair[1])
        else:
            outputs[pair] = int(rng.choice(extra_colors))
    return outputs


def _build_instance(rng: random.Random) -> tuple[np.ndarray, np.ndarray]:
    n_h_colors = rng.choice([2, 3])
    n_v_colors = rng.choice([2, 3])
    n_h_lines = 2 * n_h_colors
    n_v_lines = 2 * n_v_colors

    h_min = max(12, 2 * n_h_lines + 5)
    w_min = max(12, 2 * n_v_lines + 5)
    h = rng.randint(h_min, min(24, h_min + 6))
    w = rng.randint(w_min, min(24, w_min + 6))

    colors = list(PALETTE)
    rng.shuffle(colors)
    h_colors = [int(c) for c in colors[:n_h_colors]]
    v_colors = [int(c) for c in colors[n_h_colors : n_h_colors + n_v_colors]]
    remaining = [int(c) for c in colors[n_h_colors + n_v_colors :]]
    extra_count = rng.randint(1, min(3, len(remaining)))
    extra_colors = remaining[:extra_count]
    noise_colors = remaining[extra_count:] or extra_colors

    row_positions = _sample_spaced_positions(rng, 3, h - 4, n_h_lines, 2)
    col_positions = _sample_spaced_positions(rng, 3, w - 4, n_v_lines, 2)

    horizontal_assignments = [color for color in h_colors for _ in range(2)]
    vertical_assignments = [color for color in v_colors for _ in range(2)]
    rng.shuffle(horizontal_assignments)
    rng.shuffle(vertical_assignments)

    pair_to_color = _pair_outputs(rng, h_colors, v_colors, extra_colors)

    grid = np.zeros((h, w), dtype=int)
    left, right = 1, w - 2
    top, bottom = 1, h - 2

    for r, color in zip(row_positions, horizontal_assignments):
        grid[int(r), left : right + 1] = int(color)

    for c, color in zip(col_positions, vertical_assignments):
        grid[top : bottom + 1, int(c)] = int(color)

    pair_occurrences: dict[tuple[int, int], list[tuple[int, int]]] = defaultdict(list)
    for r, h_color in zip(row_positions, horizontal_assignments):
        for c, v_color in zip(col_positions, vertical_assignments):
            pair_occurrences[(int(h_color), int(v_color))].append((int(r), int(c)))

    hidden: set[tuple[int, int]] = set()
    for pts in pair_occurrences.values():
        roll = rng.random()
        hide_n = 1 if roll < 0.7 else 2 if roll < 0.85 else 0
        if hide_n == 0:
            continue
        for pos in rng.sample(pts, hide_n):
            hidden.add((int(pos[0]), int(pos[1])))
    if not hidden:
        some_pts = rng.choice(list(pair_occurrences.values()))
        pos = rng.choice(some_pts)
        hidden.add((int(pos[0]), int(pos[1])))

    expected = grid.copy()
    for pair, pts in pair_occurrences.items():
        out_color = int(pair_to_color[pair])
        for r, c in pts:
            expected[r, c] = out_color
            grid[r, c] = BG if (r, c) in hidden else out_color

    border_positions = []
    for c in range(w):
        border_positions.append((0, int(c)))
        border_positions.append((h - 1, int(c)))
    for r in range(1, h - 1):
        border_positions.append((int(r), 0))
        border_positions.append((int(r), w - 1))
    rng.shuffle(border_positions)

    for r, c in border_positions[: rng.choice([0, 1, 2])]:
        if grid[r, c] != BG:
            continue
        grid[r, c] = int(rng.choice(noise_colors))
        expected[r, c] = int(grid[r, c])

    return grid, expected


def generate_input() -> np.ndarray:
    rng = random.Random()
    for _ in range(GEN_RETRIES):
        try:
            grid, expected = _build_instance(rng)
        except Exception:
            continue
        if np.array_equal(main(grid), expected):
            return grid
    raise RuntimeError("generate_input: exhausted retries")
