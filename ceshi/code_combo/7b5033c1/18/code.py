# Combo variant of 7b5033c1 (index 18)
# combo_spec: D3/3+D5/2
# Name: 多路径+分行输出

from __future__ import annotations

import random
import sys
from pathlib import Path

import numpy as np

ARC_UTILS_DIR = Path("/home/lixinyu/snap/new/solvers")
if str(ARC_UTILS_DIR) not in sys.path:
    sys.path.append(str(ARC_UTILS_DIR))

from arc_utils import get_bg


# Design analysis for the combo variant
# a) Concretization and trace:
#    The original task has one long snake and outputs a single N x 1 column. For a
#    small example, a traced color order [2, 2, 5, 3, 3, 8] becomes
#        [[2],
#         [2],
#         [5],
#         [3],
#         [3],
#         [8]]
#    In this combo, the input contains 3-5 disconnected short snakes. Suppose the
#    traced path colors are:
#      path A (top):    [2, 2, 5, 3, 3]
#      path B (middle): [8, 4, 4, 1, 1, 6]
#      path C (bottom): [7, 7, 2, 9]
#    The original solver would fail because it assumes one connected path. A naive
#    D3-only adaptation would concatenate to one column. The fused rule instead:
#    1. finds each connected snake separately;
#    2. traces each from its own top-left endpoint;
#    3. sorts snakes by that chosen endpoint from top to bottom;
#    4. writes one whole traced snake per output row, left-aligned with bg padding.
#    Output becomes
#        [[2, 2, 5, 3, 3, bg],
#         [8, 4, 4, 1, 1, 6 ],
#         [7, 7, 2, 9, bg, bg]]
#    The D5 idea is therefore reused at the row-packing stage, but the row units
#    are whole paths rather than monochrome color runs.
# b) Complexity calibration:
#    The generator forces 3-5 paths of length 5-8, with at least two distinct path
#    lengths and with every path using at least two color runs. This avoids:
#    - collapse back to the original one-path task;
#    - collapse to a trivial rectangle with no visible padding;
#    - confusion with D5-only "one color run per row", because each row is usually
#      multicolor and longer than one run.
# c) Implicit-assumption audit against the original main():
#    - `pts` used to represent exactly one component. No longer true; we must split
#      into connected components first.
#    - `ends = [p for p in pts if deg[p] == 1]` used to find the two global
#      endpoints. Now every component has its own endpoint pair, so degree logic is
#      valid only inside one component at a time.
#    - The original traversal loop stopped after tracing one chain. In the combo we
#      must trace every component and keep their results separate instead of merging
#      everything into one list immediately.
#    - The original final reshape((-1, 1)) assumed a single-column output. That is
#      replaced by row packing with bg padding to the longest path length.
#    New solver assumptions introduced here:
#    - Every connected component is a simple 4-connected chain with exactly two
#      endpoints (or length 1 in the degenerate empty helper case).
#    - Sorting key is the top-left endpoint selected by tracing.
#    - bg padding is safe because generator never paints any path cell with bg.
# d) Placement legality:
#    Components are sampled from short staircase templates, mirrored but not rotated
#    into taller shapes. They are placed in separated row bands, one path per band,
#    so they cannot overlap, go out of bounds, or form accidental 4-neighbor merges.
# e) Visual observability:
#    Across a few examples a human sees several separate snakes in different vertical
#    bands, while the output is a compact table with exactly one row per snake.
#    Because rows are multicolor and row order follows top-to-bottom input order, the
#    fused rule is visible without relying on numeric color identities.
# f) Visual cleanliness:
#    Only one object family exists: short snakes. There are no extra markers or
#    overlapping overlays. Wide row-band spacing keeps each object visually isolated,
#    and output padding only appears at row tails.
# g) Color-change inference:
#    There is no recoloring rule to infer. Colors are preserved exactly in traversal
#    order; only the layout changes. Palette size is limited to 3-5 non-bg colors so
#    samples stay readable rather than noisy.


NEIGHBORS4 = ((-1, 0), (1, 0), (0, -1), (0, 1))
PATH_TEMPLATES: dict[int, list[list[tuple[int, int]]]] = {
    5: [
        [(0, 0), (1, 0), (1, 1), (2, 1), (2, 2)],
        [(0, 0), (0, 1), (1, 1), (2, 1), (2, 2)],
    ],
    6: [
        [(0, 0), (1, 0), (1, 1), (2, 1), (2, 2), (3, 2)],
        [(0, 0), (0, 1), (1, 1), (1, 2), (2, 2), (2, 3)],
    ],
    7: [
        [(0, 0), (1, 0), (1, 1), (2, 1), (2, 2), (3, 2), (3, 3)],
        [(0, 0), (0, 1), (1, 1), (2, 1), (2, 2), (3, 2), (3, 3)],
    ],
    8: [
        [(0, 0), (0, 1), (1, 1), (1, 2), (2, 2), (2, 3), (3, 3), (3, 4)],
        [(0, 0), (1, 0), (1, 1), (2, 1), (2, 2), (3, 2), (3, 3), (3, 4)],
    ],
}
ROW_BANDS = [1, 6, 11, 16, 21]


def _connected_components(points: list[tuple[int, int]]) -> list[list[tuple[int, int]]]:
    pset = set(points)
    seen: set[tuple[int, int]] = set()
    components: list[list[tuple[int, int]]] = []
    for start in sorted(points):
        if start in seen:
            continue
        stack = [start]
        seen.add(start)
        comp: list[tuple[int, int]] = []
        while stack:
            cur = stack.pop()
            comp.append(cur)
            r, c = cur
            for dr, dc in NEIGHBORS4:
                nxt = (r + dr, c + dc)
                if nxt in pset and nxt not in seen:
                    seen.add(nxt)
                    stack.append(nxt)
        components.append(sorted(comp))
    return components


def _trace_component(component: list[tuple[int, int]]) -> list[tuple[int, int]]:
    pts = sorted(component)
    if not pts:
        return []
    if len(pts) == 1:
        return pts

    pset = set(pts)
    deg: dict[tuple[int, int], int] = {}
    for r, c in pts:
        deg[(r, c)] = sum((r + dr, c + dc) in pset for dr, dc in NEIGHBORS4)

    ends = [p for p in pts if deg[p] == 1]
    mids = [p for p in pts if deg[p] == 2]
    assert len(ends) == 2
    assert len(mids) == len(pts) - 2

    start = min(ends)
    order: list[tuple[int, int]] = []
    prev: tuple[int, int] | None = None
    cur = start
    visited: set[tuple[int, int]] = set()
    while True:
        assert cur not in visited
        visited.add(cur)
        order.append(cur)
        r, c = cur
        nbrs = sorted(
            (r + dr, c + dc)
            for dr, dc in NEIGHBORS4
            if (r + dr, c + dc) in pset and (r + dr, c + dc) != prev
        )
        if not nbrs:
            break
        assert len(nbrs) == 1
        prev, cur = cur, nbrs[0]

    assert len(order) == len(pts)
    return order


def _pack_rows(rows: list[list[int]], bg: int) -> np.ndarray:
    if not rows:
        return np.zeros((0, 0), dtype=int)

    out_h = int(len(rows))
    out_w = int(max(len(row) for row in rows))
    out = np.full((out_h, out_w), int(bg), dtype=int)
    for r, row in enumerate(rows):
        out[int(r), : len(row)] = np.array(row, dtype=int)
    return out


def _split_color_runs(colors: list[int]) -> list[list[int]]:
    if not colors:
        return []

    runs: list[list[int]] = [[int(colors[0])]]
    for color in colors[1:]:
        color = int(color)
        if color == runs[-1][-1]:
            runs[-1].append(color)
        else:
            runs.append([color])
    return runs


def _extract_path_rows(g: np.ndarray) -> tuple[int, list[list[int]]]:
    bg = int(get_bg(g))
    points = [(int(r), int(c)) for r, c in np.argwhere(g != bg)]
    if not points:
        return bg, []

    traced_with_start: list[tuple[tuple[int, int], list[tuple[int, int]]]] = []
    for comp in _connected_components(points):
        order = _trace_component(comp)
        assert order
        traced_with_start.append((order[0], order))

    traced_with_start.sort(key=lambda item: item[0])
    rows: list[list[int]] = []
    for _start, order in traced_with_start:
        rows.append([int(g[r, c]) for r, c in order])
    return bg, rows


def _build_expected_output(rows: list[list[int]], bg: int) -> np.ndarray:
    return _pack_rows(rows, bg)


def _analyze_grid(grid: np.ndarray) -> dict[str, int]:
    """
    Inspect a generated example for anti-degeneracy checks.

    Metrics:
    - path_count: should be 3-5.
    - multicolor_rows: should equal path_count, proving rows are whole paths rather
      than single monochrome runs.
    - total_runs: should exceed path_count, showing D5-style row units are not just
      individual color runs.
    - padded_rows: should be > 0 often, making the variable row-length packing visible.
    """
    g = np.array(grid, dtype=int)
    bg, rows = _extract_path_rows(g)
    out = _pack_rows(rows, bg)
    total_runs = sum(len(_split_color_runs(row)) for row in rows)
    max_len = max((len(row) for row in rows), default=0)
    padded_rows = sum(len(row) < max_len for row in rows)
    return {
        "path_count": int(len(rows)),
        "max_len": int(max_len),
        "output_h": int(out.shape[0]),
        "output_w": int(out.shape[1]) if out.size else 0,
        "multicolor_rows": int(sum(len(set(row)) >= 2 for row in rows)),
        "total_runs": int(total_runs),
        "padded_rows": int(padded_rows),
    }


def main(grid: np.ndarray) -> np.ndarray:
    """
    Combo rule:
    - Non-bg cells form 3-5 disconnected simple snakes.
    - Trace each snake from its own top-left endpoint.
    - Sort snakes by that chosen endpoint from top to bottom (lexicographic tie-break).
    - Output one row per traced snake, left-aligned and bg-padded to the longest row.

    Loop-by-loop assumption audit:
    - `_connected_components` assumes different snakes never 4-touch.
    - `_trace_component` assumes each component is a simple chain with two endpoints.
    - `_pack_rows` assumes bg is absent from path colors, so padding stays readable.
    `generate_input()` constructs inputs to satisfy all three assumptions.
    """
    g = np.array(grid, dtype=int)
    bg, rows = _extract_path_rows(g)
    return _pack_rows(rows, bg)


def _random_partition(
    total: int,
    num_parts: int,
    min_len: int,
    max_len: int,
    *,
    require_variety: bool = False,
) -> list[int] | None:
    if num_parts * min_len > total or num_parts * max_len < total:
        return None

    for _ in range(100):
        parts = [int(min_len)] * int(num_parts)
        remaining = int(total - sum(parts))
        for _ in range(remaining):
            eligible = [i for i, part in enumerate(parts) if part < max_len]
            if not eligible:
                break
            parts[random.choice(eligible)] += 1
        if sum(parts) != total:
            continue
        if require_variety and len(set(parts)) < 2:
            continue
        random.shuffle(parts)
        return [int(part) for part in parts]
    return None


def _mirror_path(path: list[tuple[int, int]]) -> list[tuple[int, int]]:
    arr = np.array(path, dtype=int)
    if random.random() < 0.5:
        arr[:, 0] *= -1
    if random.random() < 0.5:
        arr[:, 1] *= -1
    arr[:, 0] -= int(arr[:, 0].min())
    arr[:, 1] -= int(arr[:, 1].min())
    return [(int(r), int(c)) for r, c in arr.tolist()]


def _sample_relative_path(length: int) -> list[tuple[int, int]]:
    shape = random.choice(PATH_TEMPLATES[length])
    return _mirror_path(shape)


def _make_row_colors(length: int, palette: list[int]) -> list[int]:
    """
    Build one path's colors.

    New generator guarantee:
    - Every path has at least two color runs and at least two distinct colors, so
      output rows are visibly whole paths rather than monochrome atoms.
    """
    num_runs = random.randint(2, min(4, length))
    run_lengths = _random_partition(length, num_runs, min_len=1, max_len=4)
    assert run_lengths is not None

    run_colors = [random.choice(palette)]
    for _ in range(1, num_runs):
        choices = [color for color in palette if color != run_colors[-1]]
        run_colors.append(random.choice(choices))

    colors: list[int] = []
    for run_len, color in zip(run_lengths, run_colors):
        colors.extend([int(color)] * int(run_len))
    assert len(set(colors)) >= 2
    return colors


def _place_components(
    band_rows: list[int],
    lengths: list[int],
    H: int,
    W: int,
) -> list[list[tuple[int, int]]]:
    components: list[list[tuple[int, int]]] = []
    for band_row, length in zip(band_rows, lengths):
        shape = _sample_relative_path(length)
        height = max(r for r, _ in shape) + 1
        width = max(c for _, c in shape) + 1
        if band_row + height > H - 1:
            raise AssertionError("band placement exceeds grid height")
        start_col = random.randint(1, W - width - 1)
        components.append([(band_row + r, start_col + c) for r, c in shape])
    return components


def _gen_attempt() -> np.ndarray | None:
    H, W = 26, 20
    bg = random.randint(0, 9)
    num_paths = random.randint(3, 5)
    palette_size = random.randint(3, 5)
    palette = random.sample([c for c in range(10) if c != bg], palette_size)

    lengths = _random_partition(
        total=random.randint(num_paths * 5, num_paths * 8),
        num_parts=num_paths,
        min_len=5,
        max_len=8,
        require_variety=True,
    )
    if lengths is None:
        return None

    band_rows = sorted(random.sample(ROW_BANDS, num_paths))
    components = _place_components(band_rows, lengths, H, W)

    all_points = [pt for comp in components for pt in comp]
    if len(all_points) != len(set(all_points)):
        return None

    traced_with_start: list[tuple[tuple[int, int], list[tuple[int, int]]]] = []
    for comp in components:
        order = _trace_component(comp)
        traced_with_start.append((order[0], order))
    traced_with_start.sort(key=lambda item: item[0])

    grid = np.full((H, W), int(bg), dtype=int)
    expected_rows: list[list[int]] = []
    for _start, order in traced_with_start:
        colors = _make_row_colors(len(order), palette)
        expected_rows.append(colors)
        for (r, c), color in zip(order, colors):
            grid[int(r), int(c)] = int(color)

    analysis = _analyze_grid(grid)
    if analysis["path_count"] != num_paths:
        return None
    if analysis["multicolor_rows"] != num_paths:
        return None
    if analysis["total_runs"] <= num_paths:
        return None
    if analysis["padded_rows"] == 0:
        return None

    expected_out = _build_expected_output(expected_rows, bg)
    actual_out = main(grid)
    if not np.array_equal(actual_out, expected_out):
        return None
    return grid


def generate_input() -> np.ndarray:
    for _ in range(400):
        grid = _gen_attempt()
        if grid is not None:
            return grid
    raise RuntimeError("Failed to generate input after 400 attempts")
