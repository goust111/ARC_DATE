from __future__ import annotations

import random
import sys
from pathlib import Path

import numpy as np

_UTILS_DIR = Path("/home/lixinyu/snap/new/solvers")
if str(_UTILS_DIR) not in sys.path:
    sys.path.append(str(_UTILS_DIR))

from arc_utils import bbox_coords, components_mask, get_bg


def _extract_crossings(grid: np.ndarray):
    g = np.asarray(grid, dtype=int)
    bg = int(get_bg(g))

    comps = components_mask(g != bg, connectivity=4)
    assert comps, "expected visible crossing traces"

    blocks: dict[tuple[tuple[int, int], tuple[int, int]], int] = {}
    row_spans: set[tuple[int, int]] = set()
    col_spans: set[tuple[int, int]] = set()

    for coords in comps:
        r0, c0, r1, c1 = bbox_coords(coords)
        patch = g[r0 : r1 + 1, c0 : c1 + 1]
        assert int(coords.shape[0]) == patch.size, "each trace must be a solid rectangle"
        color = int(patch[0, 0])
        assert color != bg and np.all(patch == color), "each trace must be monochrome"

        row_span = (int(r0), int(r1))
        col_span = (int(c0), int(c1))
        key = (row_span, col_span)
        assert key not in blocks, "duplicate crossing block"
        blocks[key] = color
        row_spans.add(row_span)
        col_spans.add(col_span)

    row_list = sorted(row_spans)
    col_list = sorted(col_spans)
    for (a0, a1), (b0, _b1) in zip(row_list, row_list[1:]):
        assert a1 < b0, "row bands must be disjoint"
    for (a0, a1), (b0, _b1) in zip(col_list, col_list[1:]):
        assert a1 < b0, "column bands must be disjoint"

    assert len(blocks) == len(row_list) * len(col_list), (
        "every recovered row band must cross every recovered column band"
    )

    obs = np.zeros((len(row_list), len(col_list)), dtype=int)
    for i, row_span in enumerate(row_list):
        for j, col_span in enumerate(col_list):
            key = (row_span, col_span)
            assert key in blocks, "missing crossing for some row/column pair"
            obs[i, j] = int(blocks[key])

    return g, bg, row_list, col_list, obs


def _propagate_assignments(obs: np.ndarray, state: dict[tuple[str, int], int]):
    R, C = obs.shape
    state = dict(state)
    changed = True
    while changed:
        changed = False
        for i in range(R):
            for j in range(C):
                target = int(obs[i, j])
                row_val = state.get(("r", i))
                col_val = state.get(("c", j))

                if row_val is not None and col_val is not None:
                    if row_val != target and col_val != target:
                        return None
                    continue

                if row_val is not None and row_val != target:
                    key = ("c", j)
                    cur = state.get(key)
                    if cur is None:
                        state[key] = target
                        changed = True
                    elif cur != target:
                        return None

                if col_val is not None and col_val != target:
                    key = ("r", i)
                    cur = state.get(key)
                    if cur is None:
                        state[key] = target
                        changed = True
                    elif cur != target:
                        return None

    return state


def _candidate_colors(
    obs: np.ndarray,
    state: dict[tuple[str, int], int],
    key: tuple[str, int],
    palette: list[int],
    row_seen: list[set[int]],
    col_seen: list[set[int]],
) -> list[int]:
    kind, idx = key
    required = None

    if kind == "r":
        for j in range(obs.shape[1]):
            col_val = state.get(("c", j))
            target = int(obs[idx, j])
            if col_val is not None and col_val != target:
                if required is None:
                    required = target
                elif required != target:
                    return []
        preferred = row_seen[idx]
    else:
        for i in range(obs.shape[0]):
            row_val = state.get(("r", i))
            target = int(obs[i, idx])
            if row_val is not None and row_val != target:
                if required is None:
                    required = target
                elif required != target:
                    return []
        preferred = col_seen[idx]

    if required is not None:
        return [int(required)]

    return [c for c in palette if c in preferred] + [c for c in palette if c not in preferred]


def _enumerate_colorings(obs: np.ndarray, *, max_solutions: int = 2):
    obs = np.asarray(obs, dtype=int)
    R, C = obs.shape
    palette = [int(x) for x in np.unique(obs)]
    row_seen = [set(map(int, obs[i, :])) for i in range(R)]
    col_seen = [set(map(int, obs[:, j])) for j in range(C)]
    keys = [("r", i) for i in range(R)] + [("c", j) for j in range(C)]
    solutions: list[tuple[list[int], list[int]]] = []

    def backtrack(state: dict[tuple[str, int], int]) -> None:
        if len(solutions) >= max_solutions:
            return

        state2 = _propagate_assignments(obs, state)
        if state2 is None:
            return

        if len(state2) == R + C:
            row_colors = [int(state2[("r", i)]) for i in range(R)]
            col_colors = [int(state2[("c", j)]) for j in range(C)]
            solutions.append((row_colors, col_colors))
            return

        best_key = None
        best_domain = None
        best_score = None
        for key in keys:
            if key in state2:
                continue
            domain = _candidate_colors(obs, state2, key, palette, row_seen, col_seen)
            if not domain:
                return

            kind, idx = key
            seen_count = len(row_seen[idx]) if kind == "r" else len(col_seen[idx])
            score = (len(domain), seen_count)
            if best_score is None or score < best_score:
                best_key = key
                best_domain = domain
                best_score = score

        assert best_key is not None and best_domain is not None
        for color in best_domain:
            next_state = dict(state2)
            next_state[best_key] = int(color)
            backtrack(next_state)
            if len(solutions) >= max_solutions:
                return

    backtrack({})
    return solutions


def _render_output(
    shape: tuple[int, int],
    bg: int,
    row_spans: list[tuple[int, int]],
    col_spans: list[tuple[int, int]],
    obs: np.ndarray,
    row_colors: list[int],
    col_colors: list[int],
) -> np.ndarray:
    out = np.full(shape, int(bg), dtype=int)
    for (r0, r1), color in zip(row_spans, row_colors):
        out[r0 : r1 + 1, :] = int(color)
    for (c0, c1), color in zip(col_spans, col_colors):
        out[:, c0 : c1 + 1] = int(color)
    for i, (r0, r1) in enumerate(row_spans):
        for j, (c0, c1) in enumerate(col_spans):
            out[r0 : r1 + 1, c0 : c1 + 1] = int(obs[i, j])
    return out


def main(grid: np.ndarray) -> np.ndarray:
    """
    Reconstruct the hidden full-span horizontal and vertical bars from their crossing traces alone.

    Input structure:
    - The background is the most common color.
    - Every non-background component is a solid monochrome rectangle.
    - These rectangles lie on a row-band / column-band lattice and represent only the visible crossing patches of unseen full-width horizontal bars and full-height vertical bars.

    Rule:
    - Recover the horizontal row bands and vertical column bands from the rectangles' shared row and column intervals.
    - Assign one color to every row band and every column band so that each crossing rectangle color equals either its row-band color or its column-band color.
    - Keep only the uniquely implied coloring, then draw all row bands and column bands across the whole grid.
    - At every crossing rectangle, preserve the original input color; elsewhere on a band, use that band's own color.

    Key insight:
    - The rectangles do not depict the bars themselves, only their overlaps, so the bar colors must be inferred globally from the compatibility constraints "crossing color belongs to one of the two participating bars."
    """
    g, bg, row_spans, col_spans, obs = _extract_crossings(grid)
    solutions = _enumerate_colorings(obs, max_solutions=2)
    assert len(solutions) == 1, "crossing traces must determine a unique bar coloring"
    row_colors, col_colors = solutions[0]
    return _render_output(g.shape, bg, row_spans, col_spans, obs, row_colors, col_colors)


# --------------- generator (o5x) ---------------

COLOR_POOL = tuple(range(10))
MAX_GENERATION_ATTEMPTS = 400
MIN_CANVAS_SIDE = 12
MAX_CANVAS_SIDE = 30
THICKNESS_CHOICES = (1, 1, 2, 2, 3)
BASE_GAP_RANGE = (1, 4)
ROW_MODE_PAIRS = ((3, 2), (3, 3), (3, 4), (4, 2), (4, 3))
COL_MODE_PAIRS = ((2, 3), (2, 4), (3, 3), (3, 4), (4, 3))
MAX_PADDING_BONUS = 6
MAX_CROSSING_FRACTION = 0.28
MAX_OUTPUT_FILL_FRACTION = 0.72


def _sample_spans(count: int):
    for _ in range(120):
        thicknesses = [random.choice(THICKNESS_CHOICES) for _ in range(count)]
        gaps = [random.randint(*BASE_GAP_RANGE) for _ in range(count + 1)]
        total = sum(thicknesses) + sum(gaps)
        if total > MAX_CANVAS_SIDE:
            continue

        lo = max(MIN_CANVAS_SIDE, total)
        hi = max(lo, min(MAX_CANVAS_SIDE, total + MAX_PADDING_BONUS))
        target = random.randint(lo, hi)
        for _ in range(target - total):
            gaps[random.randrange(len(gaps))] += 1

        spans: list[tuple[int, int]] = []
        pos = gaps[0]
        for i, thick in enumerate(thicknesses):
            spans.append((int(pos), int(pos + thick - 1)))
            pos += int(thick + gaps[i + 1])
        return spans, target, thicknesses
    return None


def _build_owner_matrix(n_rows: int, n_cols: int, mode: str):
    owner = np.full((n_rows, n_cols), -1, dtype=int)

    if mode == "row":
        if n_rows < 3:
            return None
        anchor = random.randrange(n_rows)
        owner[anchor, :] = 1
        other_rows = [i for i in range(n_rows) if i != anchor]

        for i in other_rows:
            row_j, col_j = random.sample(range(n_cols), 2)
            owner[i, row_j] = 1
            owner[i, col_j] = 0
            for j in range(n_cols):
                if owner[i, j] == -1:
                    owner[i, j] = random.randint(0, 1)

        for j in range(n_cols):
            if np.all(owner[:, j] == 1):
                candidates = [i for i in other_rows if int(np.sum(owner[i, :] == 1)) >= 2]
                if not candidates:
                    return None
                i = random.choice(candidates)
                owner[i, j] = 0

    else:
        if n_cols < 3:
            return None
        anchor = random.randrange(n_cols)
        owner[:, anchor] = 0
        other_cols = [j for j in range(n_cols) if j != anchor]

        for j in other_cols:
            row_i, col_i = random.sample(range(n_rows), 2)
            owner[row_i, j] = 1
            owner[col_i, j] = 0
            for i in range(n_rows):
                if owner[i, j] == -1:
                    owner[i, j] = random.randint(0, 1)

        for i in range(n_rows):
            if not np.any(owner[i, :] == 1):
                candidates = [j for j in other_cols if int(np.sum(owner[:, j] == 0)) >= 2]
                if not candidates:
                    return None
                j = random.choice(candidates)
                owner[i, j] = 1

    assert not np.any(owner == -1)
    return owner.astype(bool)


def _render_input(shape, bg, row_spans, col_spans, obs):
    grid = np.full(shape, int(bg), dtype=int)
    for i, (r0, r1) in enumerate(row_spans):
        for j, (c0, c1) in enumerate(col_spans):
            grid[r0 : r1 + 1, c0 : c1 + 1] = int(obs[i, j])
    return grid


def _validate_instance(
    inp: np.ndarray,
    expected: np.ndarray,
    row_spans: list[tuple[int, int]],
    col_spans: list[tuple[int, int]],
    obs: np.ndarray,
    row_colors: list[int],
    col_colors: list[int],
    owner: np.ndarray,
) -> bool:
    H, W = inp.shape
    row_total = sum(r1 - r0 + 1 for r0, r1 in row_spans)
    col_total = sum(c1 - c0 + 1 for c0, c1 in col_spans)
    crossing_area = row_total * col_total
    output_fill = row_total * W + col_total * H - crossing_area

    if crossing_area > H * W * MAX_CROSSING_FRACTION:
        return False
    if output_fill > H * W * MAX_OUTPUT_FILL_FRACTION:
        return False

    if not owner.any() or np.all(owner):
        return False

    if not all(int(row_colors[i]) in set(map(int, obs[i, :])) for i in range(obs.shape[0])):
        return False
    if not all(int(col_colors[j]) in set(map(int, obs[:, j])) for j in range(obs.shape[1])):
        return False

    row_uniform = any(len(set(map(int, obs[i, :]))) == 1 for i in range(obs.shape[0]))
    col_uniform = any(len(set(map(int, obs[:, j]))) == 1 for j in range(obs.shape[1]))
    if not (row_uniform or col_uniform):
        return False

    if not any(len(set(map(int, obs[i, :]))) > 1 for i in range(obs.shape[0])):
        return False
    if not any(len(set(map(int, obs[:, j]))) > 1 for j in range(obs.shape[1])):
        return False

    parsed = _extract_crossings(inp)
    if parsed[2] != row_spans or parsed[3] != col_spans or not np.array_equal(parsed[4], obs):
        return False

    solutions = _enumerate_colorings(obs, max_solutions=2)
    if len(solutions) != 1:
        return False
    if solutions[0] != (list(map(int, row_colors)), list(map(int, col_colors))):
        return False

    actual = main(inp)
    if not np.array_equal(actual, expected):
        return False
    if np.array_equal(inp, expected):
        return False
    return True


def _gen_attempt() -> np.ndarray | None:
    mode = random.choice(("row", "col"))
    n_rows, n_cols = random.choice(ROW_MODE_PAIRS if mode == "row" else COL_MODE_PAIRS)

    row_info = _sample_spans(n_rows)
    col_info = _sample_spans(n_cols)
    if row_info is None or col_info is None:
        return None
    row_spans, H, _row_thicknesses = row_info
    col_spans, W, _col_thicknesses = col_info

    bg = random.choice(COLOR_POOL)
    palette = random.sample([c for c in COLOR_POOL if c != bg], n_rows + n_cols)
    row_colors = [int(c) for c in palette[:n_rows]]
    col_colors = [int(c) for c in palette[n_rows:]]

    owner = _build_owner_matrix(n_rows, n_cols, mode)
    if owner is None:
        return None

    obs = np.where(owner, np.array(row_colors)[:, None], np.array(col_colors)[None, :]).astype(int)
    if len(np.unique(obs)) != n_rows + n_cols:
        return None

    inp = _render_input((H, W), bg, row_spans, col_spans, obs)
    expected = _render_output((H, W), bg, row_spans, col_spans, obs, row_colors, col_colors)

    if not _validate_instance(inp, expected, row_spans, col_spans, obs, row_colors, col_colors, owner):
        return None
    return inp


def generate_input() -> np.ndarray:
    for _ in range(MAX_GENERATION_ATTEMPTS):
        grid = _gen_attempt()
        if grid is not None:
            return grid
    raise RuntimeError("failed to generate a valid hidden-bar crossing task")
