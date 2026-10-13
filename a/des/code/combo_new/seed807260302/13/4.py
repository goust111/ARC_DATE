import random
from collections import Counter

import numpy as np


MARKER_MAX = 4
GEN_RETRIES = 500


def _majority_color(grid: np.ndarray) -> int:
    values, counts = np.unique(grid, return_counts=True)
    return int(values[np.argmax(counts)])


def _build_specs(
    top_map: dict[int, list[int]], left_map: dict[int, list[int]]
) -> list[dict[str, object]]:
    assert top_map, "expected at least one marker color"
    assert set(top_map) == set(left_map), "top-row and left-column colors must match"

    specs: list[dict[str, object]] = []
    for color in top_map:
        cols = sorted(int(c) for c in top_map[color])
        rows = sorted(int(r) for r in left_map[color])
        assert 1 <= len(cols) <= MARKER_MAX, "each marker color must appear 1..4 times per ruler"
        assert len(cols) == len(rows), "top-row and left-column counts must match"
        nodes = [(int(r), int(c)) for r, c in zip(rows, cols)]
        nodes.sort(key=lambda rc: (int(rc[1]), int(rc[0])))
        specs.append(
            {
                "color": int(color),
                "first_col": int(cols[0]),
                "cols": cols,
                "rows": rows,
                "nodes": nodes,
            }
        )

    specs.sort(key=lambda item: (int(item["first_col"]), int(item["color"])))
    return specs


def _extract_specs(grid: np.ndarray) -> tuple[int, int, list[dict[str, object]]]:
    g = np.asarray(grid, dtype=int)
    h, w = g.shape
    assert h >= 2 and w >= 2, "grid must include a ruler row and ruler column"

    bg = _majority_color(g)
    ruler = int(g[0, 0])

    top_map: dict[int, list[int]] = {}
    left_map: dict[int, list[int]] = {}

    for c in range(1, w):
        color = int(g[0, c])
        if color in (bg, ruler):
            continue
        top_map.setdefault(color, []).append(int(c))

    for r in range(1, h):
        color = int(g[r, 0])
        if color in (bg, ruler):
            continue
        left_map.setdefault(color, []).append(int(r))

    return bg, ruler, _build_specs(top_map, left_map)


def _path_cells(nodes: list[tuple[int, int]]) -> set[tuple[int, int]]:
    cells = set((int(r), int(c)) for r, c in nodes)
    for (r1, c1), (r2, c2) in zip(nodes, nodes[1:]):
        c_lo, c_hi = sorted((int(c1), int(c2)))
        r_lo, r_hi = sorted((int(r1), int(r2)))
        for c in range(c_lo, c_hi + 1):
            cells.add((int(r1), int(c)))
        for r in range(r_lo, r_hi + 1):
            cells.add((int(r), int(c2)))
    return cells


def _paint_path(out: np.ndarray, color: int, nodes: list[tuple[int, int]]) -> None:
    for r, c in nodes:
        out[int(r), int(c)] = int(color)

    for (r1, c1), (r2, c2) in zip(nodes, nodes[1:]):
        c_lo, c_hi = sorted((int(c1), int(c2)))
        r_lo, r_hi = sorted((int(r1), int(r2)))
        out[int(r1), c_lo : c_hi + 1] = int(color)
        out[r_lo : r_hi + 1, int(c2)] = int(color)


def main(grid: np.ndarray) -> np.ndarray:
    """
    Read colored ruler ticks to place same-color nodes and connect them with ordered L-shaped paths.

    Input structure:
    - The background is the dominant color, and the interior area from row 1/col 1 onward starts empty.
    - The top row and left column act as rulers whose default color is Grey, with some positions replaced by colored marker pixels.
    - Each marker color appears the same number of times on the top ruler and the left ruler.

    Rule:
    - For each marker color, collect its top-row columns from left to right and its left-column rows from top to bottom.
    - Pair the k-th column with the k-th row to create a node at that interior intersection, then place the color there.
    - If a color has multiple nodes, connect consecutive nodes in left-to-right column order using an L path that goes horizontal first, then vertical; later colors overwrite earlier overlaps based on first appearance on the top ruler.

    Key insight:
    - The border pixels are an ordered coordinate code: top ticks supply x positions, left ticks supply y positions, and matching colors decode which coordinates belong together.
    """
    g = np.asarray(grid, dtype=int)
    out = g.copy()
    _bg, _ruler, specs = _extract_specs(g)

    for spec in specs:
        _paint_path(out, int(spec["color"]), list(spec["nodes"]))

    out[0, :] = g[0, :]
    out[:, 0] = g[:, 0]
    return out


def _choose_counts(rng: random.Random) -> list[int]:
    n_colors = rng.choices([1, 2, 3, 4], weights=[1, 4, 4, 2], k=1)[0]

    if n_colors == 1:
        if rng.random() < 0.5:
            return [1]
        return [rng.randint(2, MARKER_MAX)]

    mode = rng.choice(["all_single", "one_multi", "mixed", "all_multi"])

    if mode == "all_single":
        return [1] * n_colors

    if mode == "one_multi":
        counts = [1] * n_colors
        counts[rng.randrange(n_colors)] = rng.randint(2, MARKER_MAX)
        return counts

    if mode == "all_multi":
        for _ in range(100):
            counts = [rng.randint(2, MARKER_MAX) for _ in range(n_colors)]
            if sum(counts) <= 8:
                return counts
        return [2] * n_colors

    for _ in range(100):
        counts = [rng.randint(1, MARKER_MAX) for _ in range(n_colors)]
        if any(count == 1 for count in counts) and any(count > 1 for count in counts) and sum(counts) <= 8:
            return counts
    counts = [1] * n_colors
    counts[0] = 2
    return counts


def _sample_positions(rng: random.Random, upper: int, count: int) -> list[int]:
    assert 1 <= count <= upper
    need_gap = upper >= 2 * count - 1 and rng.random() < 0.7

    for _ in range(200):
        positions = sorted(rng.sample(range(1, upper + 1), count))
        if not need_gap or all(b - a >= 2 for a, b in zip(positions, positions[1:])):
            return positions

    return sorted(rng.sample(range(1, upper + 1), count))


def _assignment_map(
    rng: random.Random, positions: list[int], colors: list[int], counts: list[int]
) -> dict[int, list[int]]:
    labels: list[int] = []
    for color, count in zip(colors, counts):
        labels.extend([int(color)] * int(count))
    rng.shuffle(labels)

    mapping: dict[int, list[int]] = {}
    for pos, color in zip(positions, labels):
        mapping.setdefault(int(color), []).append(int(pos))

    for color in mapping:
        mapping[color].sort()
    return mapping


def _overlap_count(specs: list[dict[str, object]]) -> int:
    seen: Counter[tuple[int, int]] = Counter()
    for spec in specs:
        for cell in _path_cells(list(spec["nodes"])):
            seen[cell] += 1
    return sum(1 for count in seen.values() if count > 1)


def _build_input(
    h: int,
    w: int,
    bg: int,
    ruler: int,
    top_map: dict[int, list[int]],
    left_map: dict[int, list[int]],
) -> np.ndarray:
    grid = np.full((int(h), int(w)), int(bg), dtype=int)
    grid[0, :] = int(ruler)
    grid[:, 0] = int(ruler)

    for color, cols in top_map.items():
        for c in cols:
            grid[0, int(c)] = int(color)

    for color, rows in left_map.items():
        for r in rows:
            grid[int(r), 0] = int(color)

    return grid


def _gen_attempt(rng: random.Random) -> np.ndarray | None:
    counts = _choose_counts(rng)
    total_marks = int(sum(counts))

    side_min = max(8, total_marks + 5)
    side_max = min(30, side_min + 10)
    h = rng.randint(side_min, side_max)
    w = rng.randint(side_min, side_max)

    palette = list(range(10))
    bg = rng.choice(palette)
    palette.remove(bg)
    ruler = rng.choice(palette)
    palette.remove(ruler)

    if len(counts) > len(palette):
        return None
    colors = rng.sample(palette, len(counts))

    want_overlap = len(colors) >= 2 and any(count > 1 for count in counts) and rng.random() < 0.5

    for attempt in range(200):
        top_positions = _sample_positions(rng, w - 1, total_marks)
        left_positions = _sample_positions(rng, h - 1, total_marks)
        top_map = _assignment_map(rng, top_positions, colors, counts)
        left_map = _assignment_map(rng, left_positions, colors, counts)
        specs = _build_specs(top_map, left_map)
        overlap_count = _overlap_count(specs)

        if want_overlap and overlap_count == 0:
            continue
        if not want_overlap and overlap_count > 0 and attempt < 120:
            continue

        grid = _build_input(h, w, bg, ruler, top_map, left_map)

        parsed_bg, parsed_ruler, parsed_specs = _extract_specs(grid)
        if parsed_bg != bg or parsed_ruler != ruler:
            continue
        if len(parsed_specs) != len(specs):
            continue

        out = main(grid)
        if out.shape != grid.shape or max(out.shape) > 30:
            continue
        if np.array_equal(out[1:, 1:], grid[1:, 1:]):
            continue
        return grid

    return None


def generate_input() -> np.ndarray:
    rng = random.Random()
    for _ in range(GEN_RETRIES):
        grid = _gen_attempt(rng)
        if grid is not None:
            return grid
    raise RuntimeError("failed to generate a valid ruler-connection input")
