import random
from typing import Dict, List, Sequence, Tuple

import numpy as np


BG = 0
GREY = 5
PALETTE = (1, 2, 3, 4, 6, 7, 8, 9)

GEN_RETRIES = 600
TREE_COUNT_RANGE = (2, 4)
BRANCH_COUNT_CHOICES = (1, 2, 3, 4, 5)
FIRST_BRANCH_ROW = 2
BRANCH_ROW_STEP = 3
TRUNK_TAIL_RANGE = (2, 4)
BRANCH_LEN_RANGE = (1, 4)
TREE_GAP_RANGE = (2, 4)
LEGEND_GAP_RANGE = (2, 4)
GRID_EXTRA_WIDTH = 6
GRID_EXTRA_HEIGHT = 6


def _background_color(grid: np.ndarray) -> int:
    values, counts = np.unique(grid, return_counts=True)
    return int(values[np.argmax(counts)])


def _parse_legend(row: np.ndarray, bg: int) -> Dict[int, int]:
    mapping: Dict[int, int] = {}
    c = 0
    width = len(row)
    while c < width:
        if int(row[c]) == bg:
            c += 1
            continue
        if int(row[c]) != GREY:
            raise ValueError("legend groups must start with grey count pixels")
        start = c
        while c < width and int(row[c]) == GREY:
            c += 1
        count = c - start
        if c >= width or int(row[c]) in (bg, GREY):
            raise ValueError("legend count must be followed by one fruit color")
        mapping[count] = int(row[c])
        c += 1
    if not mapping:
        raise ValueError("legend row is empty")
    return mapping


def _find_trunk(mask: np.ndarray) -> Tuple[int, int, int]:
    rows = np.where(mask)[0]
    if rows.size == 0:
        raise ValueError("empty tree mask")
    bottom = int(rows.max())
    best: Tuple[int, int, int] | None = None
    width = mask.shape[1]
    for left in range(width - 1):
        pair_rows = np.where(mask[:, left] & mask[:, left + 1])[0]
        if pair_rows.size == 0 or int(pair_rows[-1]) != bottom:
            continue
        row_set = set(int(r) for r in pair_rows)
        top = bottom
        while top - 1 in row_set:
            top -= 1
        height = bottom - top + 1
        candidate = (height, -left, top)
        if best is None or candidate > best:
            best = candidate
    if best is None:
        raise ValueError("failed to locate 2-pixel trunk")
    height, neg_left, top = best
    left = -neg_left
    if height < 2:
        raise ValueError("trunk too short")
    return int(left), int(top), int(bottom)


def _is_leaf(mask: np.ndarray, r: int, c: int) -> bool:
    if not (0 <= r < mask.shape[0] and 0 <= c < mask.shape[1] and mask[r, c]):
        return False
    neighbors = set()
    for dr, dc in ((-1, 0), (1, 0), (0, -1), (0, 1)):
        nr, nc = r + dr, c + dc
        if 0 <= nr < mask.shape[0] and 0 <= nc < mask.shape[1] and mask[nr, nc]:
            neighbors.add((dr, dc))
    return neighbors == {(1, 0)}


def _analyze_tree(mask: np.ndarray) -> Tuple[int, List[Tuple[int, int]]]:
    trunk_left, trunk_top, trunk_bottom = _find_trunk(mask)
    trunk_right = trunk_left + 1
    branch_count = 0
    leaves: List[Tuple[int, int]] = []
    for r in range(trunk_top, trunk_bottom + 1):
        left_len = 0
        c = trunk_left - 1
        while c >= 0 and mask[r, c]:
            left_len += 1
            c -= 1
        if left_len > 0:
            branch_count += 1
            endpoint = trunk_left - left_len
            if _is_leaf(mask, r - 1, endpoint):
                leaves.append((r - 1, endpoint))

        right_len = 0
        c = trunk_right + 1
        while c < mask.shape[1] and mask[r, c]:
            right_len += 1
            c += 1
        if right_len > 0:
            branch_count += 1
            endpoint = trunk_right + right_len
            if _is_leaf(mask, r - 1, endpoint):
                leaves.append((r - 1, endpoint))
    return branch_count, leaves


def main(grid: np.ndarray) -> np.ndarray:
    """
    Add fruits above tree leaf tips using a bottom legend keyed by branch count.

    Input structure:
    - A black background grid contains several single-color trees above a one-row legend.
    - Each tree has a 2-pixel-wide vertical trunk, horizontal branches extending left and/or right, and optional one-pixel leaf nodes above branch endpoints.
    - The bottom row is a legend made of groups of N grey pixels immediately followed by one colored fruit pixel.

    Rule:
    - Read the bottom legend as a mapping from grey-run length N to fruit color.
    - For each tree color above the legend, locate the 2-pixel trunk, count all horizontal main branches leaving the trunk, and find which branch endpoints carry a one-pixel leaf node.
    - Paint one fruit pixel directly above every leaf node using the legend color matched to that tree's branch count.
    - Clear the entire legend row back to the background color.

    Key insight:
    - The grey run length is a numeric code, and it must be matched to a structural count taken from each tree rather than to any specific tree color.
    """
    g = np.asarray(grid, dtype=int)
    if g.ndim != 2:
        raise ValueError("expected a 2D grid")

    bg = _background_color(g)
    legend = _parse_legend(g[-1], bg)
    out = g.copy()
    out[-1, :] = bg

    upper = g[:-1]
    tree_colors = [int(c) for c in np.unique(upper) if int(c) not in (bg, GREY)]
    for color in tree_colors:
        mask = upper == color
        branch_count, leaves = _analyze_tree(mask)
        fruit_color = legend[branch_count]
        for leaf_r, leaf_c in leaves:
            fruit_r = leaf_r - 1
            if 0 <= fruit_r < out.shape[0]:
                out[fruit_r, leaf_c] = fruit_color
    return out


def _random_composition(total: int, parts: int) -> List[int]:
    if parts == 1:
        return [total]
    cuts = sorted(random.sample(range(total + parts - 1), parts - 1))
    result: List[int] = []
    prev = -1
    for cut in cuts + [total + parts - 1]:
        result.append(cut - prev - 1)
        prev = cut
    return result


def _branch_row_patterns(branch_count: int) -> List[Tuple[str, ...]]:
    min_rows = (branch_count + 1) // 2
    row_count = random.randint(min_rows, branch_count)
    remaining = branch_count
    patterns: List[Tuple[str, ...]] = []
    for i in range(row_count):
        rows_left = row_count - i - 1
        low = max(1, remaining - 2 * rows_left)
        high = min(2, remaining - rows_left)
        branches_here = random.randint(low, high)
        if branches_here == 2:
            patterns.append(("L", "R"))
        else:
            patterns.append((random.choice(("L", "R")),))
        remaining -= branches_here
    return patterns


def _make_tree_spec(color: int, branch_count: int) -> Dict[str, object]:
    patterns = _branch_row_patterns(branch_count)
    row_specs: List[Dict[str, object]] = []
    branch_leaf_flags: List[bool] = []
    branch_sides: List[str] = []
    max_left = 0
    max_right = 0

    for i, sides in enumerate(patterns):
        row = FIRST_BRANCH_ROW + BRANCH_ROW_STEP * i
        row_spec: Dict[str, object] = {"row": row}
        if "L" in sides:
            length = random.randint(*BRANCH_LEN_RANGE)
            leaf = bool(random.getrandbits(1))
            row_spec["L"] = (length, leaf)
            branch_leaf_flags.append(leaf)
            branch_sides.append("L")
            max_left = max(max_left, length)
        if "R" in sides:
            length = random.randint(*BRANCH_LEN_RANGE)
            leaf = bool(random.getrandbits(1))
            row_spec["R"] = (length, leaf)
            branch_leaf_flags.append(leaf)
            branch_sides.append("R")
            max_right = max(max_right, length)
        row_specs.append(row_spec)

    trunk_left = max_left
    trunk_bottom = row_specs[-1]["row"] + random.randint(*TRUNK_TAIL_RANGE)
    height = trunk_bottom + 1
    width = max_left + 2 + max_right

    coords: set[Tuple[int, int]] = set()
    fruit_slots: List[Tuple[int, int]] = []

    for r in range(FIRST_BRANCH_ROW, height):
        coords.add((r, trunk_left))
        coords.add((r, trunk_left + 1))

    for row_spec in row_specs:
        row = int(row_spec["row"])
        if "L" in row_spec:
            length, leaf = row_spec["L"]
            for c in range(trunk_left - length, trunk_left):
                coords.add((row, c))
            if leaf:
                coords.add((row - 1, trunk_left - length))
                fruit_slots.append((row - 2, trunk_left - length))
        if "R" in row_spec:
            length, leaf = row_spec["R"]
            for c in range(trunk_left + 2, trunk_left + 2 + length):
                coords.add((row, c))
            if leaf:
                coords.add((row - 1, trunk_left + 1 + length))
                fruit_slots.append((row - 2, trunk_left + 1 + length))

    return {
        "color": color,
        "branch_count": branch_count,
        "coords": sorted(coords),
        "fruit_slots": fruit_slots,
        "height": height,
        "width": width,
        "branch_leaf_flags": branch_leaf_flags,
        "branch_sides": branch_sides,
    }


def _render_tree(grid: np.ndarray, spec: Dict[str, object], top: int, left: int) -> None:
    color = int(spec["color"])
    for r, c in spec["coords"]:
        grid[top + r, left + c] = color


def _build_expected(
    grid: np.ndarray,
    placed_trees: Sequence[Tuple[Dict[str, object], int, int]],
    legend_map: Dict[int, int],
) -> np.ndarray:
    out = grid.copy()
    out[-1, :] = BG
    for spec, top, left in placed_trees:
        fruit_color = legend_map[int(spec["branch_count"])]
        for r, c in spec["fruit_slots"]:
            out[top + r, left + c] = fruit_color
    return out


def _gen_attempt() -> np.ndarray | None:
    tree_count = random.randint(*TREE_COUNT_RANGE)
    distinct_counts = random.randint(2, min(4, tree_count))
    legend_counts = random.sample(BRANCH_COUNT_CHOICES, distinct_counts)
    tree_branch_counts = legend_counts[:]
    while len(tree_branch_counts) < tree_count:
        tree_branch_counts.append(random.choice(legend_counts))
    random.shuffle(tree_branch_counts)

    color_pool = list(PALETTE)
    random.shuffle(color_pool)
    tree_colors = color_pool[:tree_count]
    fruit_colors = color_pool[tree_count : tree_count + distinct_counts]
    legend_map = {count: color for count, color in zip(legend_counts, fruit_colors)}

    trees = [_make_tree_spec(color, count) for color, count in zip(tree_colors, tree_branch_counts)]

    all_leaf_flags = [flag for tree in trees for flag in tree["branch_leaf_flags"]]
    if not all_leaf_flags or not any(all_leaf_flags) or all(all_leaf_flags):
        return None
    all_sides = {side for tree in trees for side in tree["branch_sides"]}
    if all_sides != {"L", "R"}:
        return None

    random.shuffle(trees)
    tree_widths = [int(tree["width"]) for tree in trees]
    tree_heights = [int(tree["height"]) for tree in trees]
    tree_gaps = [random.randint(*TREE_GAP_RANGE) for _ in range(tree_count - 1)]
    tree_base_width = 2 + sum(tree_widths) + sum(tree_gaps)

    legend_items = list(legend_map.items())
    random.shuffle(legend_items)
    legend_gaps = [random.randint(*LEGEND_GAP_RANGE) for _ in range(len(legend_items) - 1)]
    legend_total = sum(count + 1 for count, _ in legend_items) + sum(legend_gaps)
    legend_base_width = legend_total + 2

    min_width = max(tree_base_width, legend_base_width)
    if min_width > 30:
        return None
    width = random.randint(min_width, min(30, min_width + GRID_EXTRA_WIDTH))

    max_tree_height = max(tree_heights)
    min_height = max_tree_height + 2
    if min_height > 30:
        return None
    height = random.randint(min_height, min(30, min_height + GRID_EXTRA_HEIGHT))

    grid = np.full((height, width), BG, dtype=int)

    tree_extra = width - tree_base_width
    tree_spaces = _random_composition(tree_extra, tree_count + 1)
    current_left = 1 + tree_spaces[0]
    placed_trees: List[Tuple[Dict[str, object], int, int]] = []
    for i, tree in enumerate(trees):
        top_max = height - int(tree["height"]) - 1
        if top_max < 0:
            return None
        top = random.randint(0, top_max)
        _render_tree(grid, tree, top, current_left)
        placed_trees.append((tree, top, current_left))
        if i < tree_count - 1:
            current_left += int(tree["width"]) + tree_gaps[i] + tree_spaces[i + 1]

    legend_start_max = width - legend_total - 1
    if legend_start_max < 1:
        return None
    c = random.randint(1, legend_start_max)
    for i, (count, color) in enumerate(legend_items):
        grid[-1, c : c + count] = GREY
        grid[-1, c + count] = color
        c += count + 1
        if i < len(legend_gaps):
            c += legend_gaps[i]

    expected = _build_expected(grid, placed_trees, legend_map)
    try:
        produced = main(grid)
    except Exception:
        return None
    if not np.array_equal(produced, expected):
        return None
    return grid


def generate_input() -> np.ndarray:
    for _ in range(GEN_RETRIES):
        result = _gen_attempt()
        if result is not None:
            return result
    raise RuntimeError(f"failed to generate a valid input after {GEN_RETRIES} attempts")
