import random
from typing import Dict, List, Sequence, Set, Tuple

import numpy as np


BACKGROUND = 0
MARKER_COLOR = 5
TREE_COLORS = [1, 2, 3, 4, 6, 7, 8, 9]

MIN_TREES = 2
MAX_TREES = 5
MIN_TRUNK_LEN = 3
MAX_TRUNK_LEN = 7
MAX_BRANCH_LEN = 4
MAX_BRANCHES_PER_TREE = 4
GENERATION_RETRIES = 600


Coord = Tuple[int, int]


def _dominant_color(grid: np.ndarray) -> int:
    values, counts = np.unique(grid, return_counts=True)
    return int(values[np.argmax(counts)])


def _infer_marker_color(grid: np.ndarray, bg: int) -> int:
    h, w = grid.shape
    candidates: List[int] = []
    for color in map(int, np.unique(grid)):
        if color == bg:
            continue
        coords = np.argwhere(grid == color)
        if coords.shape[0] < 2:
            continue
        ok = True
        for r, c in coords:
            rr, cc = int(r), int(c)
            if rr + 1 >= h:
                ok = False
                break
            below = int(grid[rr + 1, cc])
            if below in (bg, color):
                ok = False
                break
            for dr, dc in [(-1, 0), (1, 0), (0, -1), (0, 1)]:
                nr, nc = rr + dr, cc + dc
                if 0 <= nr < h and 0 <= nc < w and int(grid[nr, nc]) == color:
                    ok = False
                    break
            if not ok:
                break
        if ok:
            candidates.append(color)
    assert len(candidates) == 1, "expected a unique marker color"
    return candidates[0]


def _component_from_root(grid: np.ndarray, start: Coord, color: int) -> Set[Coord]:
    h, w = grid.shape
    sr, sc = start
    assert 0 <= sr < h and 0 <= sc < w and int(grid[sr, sc]) == color
    stack = [start]
    seen = {start}
    while stack:
        r, c = stack.pop()
        for dr, dc in [(-1, 0), (1, 0), (0, -1), (0, 1)]:
            nr, nc = r + dr, c + dc
            nxt = (nr, nc)
            if 0 <= nr < h and 0 <= nc < w and nxt not in seen and int(grid[nr, nc]) == color:
                seen.add(nxt)
                stack.append(nxt)
    return seen


def _nearest_neighbor_color(index: int, trees: Sequence[Dict[str, object]]) -> int:
    root_col = int(trees[index]["root_col"])
    best_key = None
    best_color = None
    for j, tree in enumerate(trees):
        if j == index:
            continue
        other_col = int(tree["root_col"])
        key = (abs(other_col - root_col), 0 if other_col < root_col else 1, other_col)
        if best_key is None or key < best_key:
            best_key = key
            best_color = int(tree["color"])
    assert best_color is not None, "each tree needs another tree as a color source"
    return best_color


def _extract_trees(grid: np.ndarray) -> Tuple[int, int, List[Dict[str, object]]]:
    bg = _dominant_color(grid)
    marker_color = _infer_marker_color(grid, bg)

    trees: List[Dict[str, object]] = []
    for r, c in sorted(map(tuple, np.argwhere(grid == marker_color)), key=lambda rc: (int(rc[1]), int(rc[0]))):
        mr, mc = int(r), int(c)
        assert mr + 1 < grid.shape[0], "marker must have a root directly below it"
        color = int(grid[mr + 1, mc])
        assert color not in (bg, marker_color), "root must be a tree color"
        coords = _component_from_root(grid, (mr + 1, mc), color)
        cols = [cc for _rr, cc in coords]
        trees.append(
            {
                "axis_row": mr,
                "root_col": mc,
                "color": color,
                "coords": coords,
                "col_min": min(cols),
                "col_max": max(cols),
            }
        )

    trees.sort(key=lambda tree: int(tree["root_col"]))
    assert len(trees) >= 2, "expected at least two trees"
    prev_max = None
    for tree in trees:
        if prev_max is not None:
            assert int(tree["col_min"]) > prev_max, "tree bounding boxes must not overlap horizontally"
        prev_max = int(tree["col_max"])
    return bg, marker_color, trees


def main(grid: np.ndarray) -> np.ndarray:
    """Mirror each tree above its grey marker using the nearest other tree's color.

    Input structure:
    - A black background contains several monochrome tree objects and one grey pixel above each tree root.
    - Each tree is a 4-connected one-pixel-wide structure: a vertical trunk growing downward from the root with horizontal branches attached on trunk rows.
    - Different trees occupy disjoint column ranges, and the grey pixel above each root marks that tree's reflection axis.

    Rule:
    - For every tree, use the grey marker row as a horizontal mirror axis and reflect the whole tree upward across that axis.
    - Keep the original tree and the grey marker unchanged; only add the reflected copy above the marker.
    - Recolor the reflected tree with the color of the other tree whose root column is closest; if two trees are equally close, use the left tree's color.
    - If reflected pixels would fall above the grid, clip them instead of expanding the canvas.

    Key insight:
    - The grey singleton is not part of the tree body: it encodes a mirror line, and the reflected copy borrows its color from the nearest neighboring tree by root-column distance rather than from the source tree itself.
    """
    g = np.asarray(grid, dtype=int)
    assert g.ndim == 2, "expected a 2D grid"

    _bg, _marker_color, trees = _extract_trees(g)
    out = g.copy()
    for i, tree in enumerate(trees):
        axis_row = int(tree["axis_row"])
        reflection_color = _nearest_neighbor_color(i, trees)
        for r, c in tree["coords"]:
            mirrored_r = 2 * axis_row - int(r)
            if mirrored_r >= 0:
                out[mirrored_r, int(c)] = reflection_color
    return out


def _sample_root_columns(tree_count: int, force_tie: bool) -> List[int]:
    root_cols = [0]
    if force_tie:
        tie_index = random.randint(1, tree_count - 2)
        tie_gap = random.randint(5, 7)
        gaps = []
        for i in range(tree_count - 1):
            if i == tie_index - 1 or i == tie_index:
                gaps.append(tie_gap)
            else:
                gaps.append(random.randint(4, 7))
    else:
        gaps = [random.randint(4, 7) for _ in range(tree_count - 1)]
    for gap in gaps:
        root_cols.append(root_cols[-1] + gap)
    return root_cols


def _branch_caps(root_cols: Sequence[int], index: int) -> Tuple[int, int]:
    left_cap = MAX_BRANCH_LEN
    right_cap = MAX_BRANCH_LEN
    if index > 0:
        left_cap = min(left_cap, max(0, (root_cols[index] - root_cols[index - 1] - 2) // 2))
    if index + 1 < len(root_cols):
        right_cap = min(right_cap, max(0, (root_cols[index + 1] - root_cols[index] - 2) // 2))
    return left_cap, right_cap


def _sample_branches(trunk_len: int, left_cap: int, right_cap: int) -> List[Tuple[int, int, int]]:
    options = []
    if left_cap > 0:
        options.append(-1)
    if right_cap > 0:
        options.append(1)
    if not options:
        raise ValueError("tree needs at least one available branching side")

    branch_count = random.randint(1, min(MAX_BRANCHES_PER_TREE, trunk_len))
    rows = random.sample(range(trunk_len), branch_count)
    branches: List[Tuple[int, int, int]] = []
    for depth in sorted(rows):
        direction = random.choice(options)
        cap = left_cap if direction == -1 else right_cap
        length = random.randint(1, cap)
        branches.append((depth, direction, length))
    return branches


def _tree_pixels(marker_row: int, root_col: int, trunk_len: int, branches: Sequence[Tuple[int, int, int]]) -> Set[Coord]:
    coords: Set[Coord] = set()
    root_row = marker_row + 1
    for depth in range(trunk_len):
        coords.add((root_row + depth, root_col))
    for depth, direction, length in branches:
        rr = root_row + depth
        for offset in range(1, length + 1):
            coords.add((rr, root_col + direction * offset))
    return coords


def _render_scene(height: int, width: int, specs: Sequence[Dict[str, object]]) -> np.ndarray:
    grid = np.full((height, width), BACKGROUND, dtype=int)
    for spec in specs:
        color = int(spec["color"])
        for r, c in spec["coords"]:
            grid[int(r), int(c)] = color
        grid[int(spec["marker_row"]), int(spec["root_col"])] = MARKER_COLOR
    return grid


def _expected_from_specs(height: int, width: int, specs: Sequence[Dict[str, object]]) -> np.ndarray:
    trees = sorted(specs, key=lambda spec: int(spec["root_col"]))
    out = _render_scene(height, width, trees)
    for i, spec in enumerate(trees):
        axis_row = int(spec["marker_row"])
        reflection_color = _nearest_neighbor_color(i, trees)
        for r, c in spec["coords"]:
            mirrored_r = 2 * axis_row - int(r)
            if 0 <= mirrored_r < height:
                out[mirrored_r, int(c)] = reflection_color
    return out


def _make_scene() -> Tuple[np.ndarray, np.ndarray] | None:
    force_tie = random.random() < 0.5
    min_trees = 3 if force_tie else MIN_TREES
    tree_count = random.randint(min_trees, MAX_TREES)
    root_cols = _sample_root_columns(tree_count, force_tie)

    specs: List[Dict[str, object]] = []
    for i, rel_root_col in enumerate(root_cols):
        trunk_len = random.randint(MIN_TRUNK_LEN, MAX_TRUNK_LEN)
        left_cap, right_cap = _branch_caps(root_cols, i)
        if left_cap == 0 and right_cap == 0:
            return None
        branches = _sample_branches(trunk_len, left_cap, right_cap)
        max_left = max((length for _depth, direction, length in branches if direction == -1), default=0)
        max_right = max((length for _depth, direction, length in branches if direction == 1), default=0)
        specs.append(
            {
                "rel_root_col": rel_root_col,
                "trunk_len": trunk_len,
                "branches": branches,
                "max_left": max_left,
                "max_right": max_right,
            }
        )

    min_col = min(int(spec["rel_root_col"]) - int(spec["max_left"]) for spec in specs)
    max_col = max(int(spec["rel_root_col"]) + int(spec["max_right"]) for spec in specs)
    left_margin = random.randint(0, 2)
    shift = left_margin - min_col
    for spec in specs:
        spec["root_col"] = int(spec["rel_root_col"]) + shift
    max_shifted_col = max(int(spec["root_col"]) + int(spec["max_right"]) for spec in specs)
    width = max_shifted_col + 1 + random.randint(0, 3)
    if width > 30:
        return None

    force_clipping = random.random() < 0.5
    clipped_index = random.randrange(tree_count) if force_clipping else None
    deepest_row = 0
    for i, spec in enumerate(specs):
        trunk_len = int(spec["trunk_len"])
        clipped = i == clipped_index or (force_clipping and random.random() < 0.25)
        if clipped:
            marker_row = random.randint(1, trunk_len - 1)
        else:
            low = trunk_len
            high = min(12, 28 - trunk_len)
            if low > high:
                return None
            marker_row = random.randint(low, high)
        spec["marker_row"] = marker_row
        deepest_row = max(deepest_row, marker_row + trunk_len)

    height = deepest_row + 1 + random.randint(0, 3)
    if height > 30:
        return None

    colors = random.sample(TREE_COLORS, tree_count)
    for spec, color in zip(specs, colors):
        spec["color"] = color
        spec["coords"] = _tree_pixels(
            int(spec["marker_row"]),
            int(spec["root_col"]),
            int(spec["trunk_len"]),
            spec["branches"],
        )

    grid = _render_scene(height, width, specs)
    expected = _expected_from_specs(height, width, specs)
    return grid, expected


def generate_input() -> np.ndarray:
    for _ in range(GENERATION_RETRIES):
        scene = _make_scene()
        if scene is None:
            continue
        grid, expected = scene
        try:
            actual = main(grid)
        except Exception:
            continue
        if np.array_equal(actual, expected):
            return grid
    raise RuntimeError("failed to generate a valid mirrored-tree input")
