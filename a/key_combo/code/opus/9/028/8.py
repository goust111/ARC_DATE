from __future__ import annotations

import random

import numpy as np


BG = 0
WALL = 5
MARKER_COLORS = (1, 2, 3)
LEAF_TO_COLOR = {1: 4, 2: 7}
THREE_PLUS_COLOR = 9
MAX_GENERATION_ATTEMPTS = 300


def _segment_bounds(grid: np.ndarray, row: int, col: int) -> tuple[int, int]:
    """Return the maximal contiguous wall segment on `row` containing `col`."""
    width = int(grid.shape[1])
    lo = hi = int(col)
    while lo - 1 >= 0 and int(grid[row, lo - 1]) == WALL:
        lo -= 1
    while hi + 1 < width and int(grid[row, hi + 1]) == WALL:
        hi += 1
    return lo, hi


def _trace_tree(grid: np.ndarray, root: tuple[int, int]) -> dict:
    """Trace one downward branching tree from a top-edge marker."""
    height, width = map(int, grid.shape)
    root = (int(root[0]), int(root[1]))
    tree_cells: set[tuple[int, int]] = {root}
    terminals: list[tuple[int, int]] = []
    stack: list[tuple[int, int]] = [root]
    seen_starts: set[tuple[int, int]] = set()

    while stack:
        start_r, start_c = stack.pop()
        if (start_r, start_c) in seen_starts:
            continue
        seen_starts.add((start_r, start_c))
        assert 0 <= start_r < height and 0 <= start_c < width
        assert int(grid[start_r, start_c]) != WALL

        cur_r, cur_c = start_r, start_c
        tree_cells.add((cur_r, cur_c))

        while True:
            if cur_r == height - 1:
                terminals.append((cur_r, cur_c))
                break

            next_r = cur_r + 1
            next_val = int(grid[next_r, cur_c])
            if next_val != WALL:
                cur_r = next_r
                tree_cells.add((cur_r, cur_c))
                continue

            left, right = _segment_bounds(grid, next_r, cur_c)
            child_starts: list[tuple[int, int]] = []
            for opening_c in (left - 1, right + 1):
                if not (0 <= opening_c < width):
                    continue
                if int(grid[next_r, opening_c]) != BG:
                    continue
                child_r = next_r + 1
                if child_r >= height or int(grid[child_r, opening_c]) != BG:
                    continue
                child_starts.append((child_r, opening_c))

            if not child_starts:
                terminals.append((cur_r, cur_c))
                break

            for child in child_starts:
                stack.append(child)
            break

    return {
        "root": root,
        "cells": tree_cells,
        "terminals": terminals,
        "leaf_count": len(terminals),
    }


def _analyze_grid(grid: np.ndarray) -> list[dict]:
    """Extract and trace every top-edge marker tree."""
    g = np.asarray(grid, dtype=int)
    assert g.ndim == 2, "expected a 2D grid"
    assert np.all(np.isin(g, np.arange(10))), "expected ARC palette ids 0-9"

    height, width = map(int, g.shape)
    assert height >= 4 and width >= 5
    assert not np.any(g[0, :] == WALL), "top edge may only contain markers or background"

    markers = [(0, int(c)) for c in np.flatnonzero((g[0, :] != BG) & (g[0, :] != WALL))]
    assert 2 <= len(markers) <= 3, "expected 2-3 top-edge markers"
    marker_values = [int(g[r, c]) for r, c in markers]
    assert len(marker_values) == len(set(marker_values)), "top-edge markers must have distinct colors"

    for r in range(height):
        for c in range(width):
            value = int(g[r, c])
            if r == 0 and (r, c) in markers:
                continue
            assert value in {BG, WALL}, "only top-edge markers and walls are allowed in the input"

    trees = [_trace_tree(g, root) for root in markers]
    return trees


def main(grid: np.ndarray) -> np.ndarray:
    """
    Trace each top-edge marker tree through broken horizontal walls, count its terminal nodes, and recolor the whole tree by that count.

    Input structure:
    - The background is black, with 2-3 distinct single-cell markers on the top edge.
    - Several grey horizontal wall layers span across the grid as broken segments with gaps.
    - Apart from the top markers and grey walls, all other cells are background.

    Rule:
    - From each top marker, extend a ray downward through background cells.
    - Whenever the next cell is part of a grey wall segment, find that whole contiguous segment and branch from the background openings immediately outside its left and right ends, starting one row below each valid opening.
    - A branch stops when it reaches the bottom row or when a hit wall offers no valid opening below it.
    - After a whole marker-rooted tree is traced, count its terminal branches and recolor every traced non-wall cell in that tree except the top marker itself: 1 leaf -> yellow, 2 leaves -> orange, 3 or more leaves -> maroon.

    Key insight:
    - The marker color only identifies the root; the final color is determined later by the completed tree topology, so the solver must trace forward first and only then recolor backward from the terminal count.
    """
    g = np.asarray(grid, dtype=int)
    trees = _analyze_grid(g)
    out = g.copy()

    for tree in trees:
        leaf_count = int(tree["leaf_count"])
        color = LEAF_TO_COLOR.get(leaf_count, THREE_PLUS_COLOR)
        root = tree["root"]
        for r, c in tree["cells"]:
            if (r, c) == root:
                continue
            out[r, c] = color

    return out


def _base_columns(width: int) -> dict[str, int]:
    """Choose non-overlapping columns for the three planned trees."""
    left_child = random.choice((4, 5))
    left_root = random.randint(1, left_child - 1)

    middle_root = width // 2 - 1 + random.randint(0, 1)
    middle_left = middle_root - 1
    middle_right = middle_root + 2

    right_center = width - 4
    right_pass = right_center - 2
    right_left_leaf = right_center - 1
    right_right_leaf = right_center + 1
    right_unary_parent = right_center + 2
    right_root = right_center + 1

    assert left_child < middle_left
    assert middle_right < right_pass
    assert right_unary_parent < width

    return {
        "left_root": left_root,
        "left_child": left_child,
        "middle_root": middle_root,
        "middle_left": middle_left,
        "middle_right": middle_right,
        "right_root": right_root,
        "right_pass": right_pass,
        "right_center": right_center,
        "right_left_leaf": right_left_leaf,
        "right_right_leaf": right_right_leaf,
        "right_unary_parent": right_unary_parent,
    }


def _reflect_columns(cols: dict[str, int], width: int) -> dict[str, int]:
    return {name: width - 1 - int(value) for name, value in cols.items()}


def _plan_rows(cols: dict[str, int]) -> dict[str, list[int]]:
    """Gap columns for the three semantic wall rows and passive states between them."""
    row1 = sorted(
        {
            cols["left_child"],
            cols["middle_root"],
            cols["right_pass"],
            cols["right_unary_parent"],
        }
    )
    row2 = sorted(
        {
            cols["left_child"],
            cols["middle_left"],
            cols["middle_right"],
            cols["right_pass"],
            cols["right_center"],
        }
    )
    row3 = sorted(
        {
            cols["left_child"],
            cols["middle_left"],
            cols["middle_right"],
            cols["right_pass"],
            cols["right_left_leaf"],
            cols["right_right_leaf"],
        }
    )
    return {
        "row1": row1,
        "row2": row2,
        "row3": row3,
        "state0": sorted({cols["left_root"], cols["middle_root"], cols["right_root"]}),
        "state1": sorted({cols["left_child"], cols["middle_root"], cols["right_pass"], cols["right_unary_parent"]}),
        "state2": sorted({cols["left_child"], cols["middle_left"], cols["middle_right"], cols["right_pass"], cols["right_center"]}),
        "state3": sorted(
            {
                cols["left_child"],
                cols["middle_left"],
                cols["middle_right"],
                cols["right_pass"],
                cols["right_left_leaf"],
                cols["right_right_leaf"],
            }
        ),
    }


def _add_safe_decoy_gaps(gaps: set[int], width: int) -> set[int]:
    """Add a few extra gaps far from active columns to diversify passive rows."""
    candidates = [c for c in range(1, width - 1) if all(abs(c - g) >= 2 for g in gaps)]
    random.shuffle(candidates)
    for c in candidates[: random.randint(0, 2)]:
        gaps.add(c)
    return gaps


def _render_wall_row(width: int, gaps: list[int], *, allow_decoys: bool) -> np.ndarray:
    row = np.full(width, WALL, dtype=int)
    gap_set = set(map(int, gaps))
    if allow_decoys:
        gap_set = _add_safe_decoy_gaps(gap_set, width)
    for c in gap_set:
        row[int(c)] = BG
    return row


def _choose_wall_layout() -> tuple[list[str], list[int], int]:
    """Choose wall row types and absolute row positions."""
    extra_slot = random.choice((None, 0, 1, 2, 3))
    row_types = ["row1", "row2", "row3"]
    if extra_slot is not None:
        row_types.insert(extra_slot, f"passive{extra_slot}")

    positions = [random.randint(2, 3)]
    while len(positions) < len(row_types):
        positions.append(positions[-1] + random.randint(2, 4))
    height = positions[-1] + random.randint(3, 5) + 1
    return row_types, positions, height


def _make_grid() -> np.ndarray:
    width = random.randint(17, 21)
    columns = _base_columns(width)
    if random.choice((False, True)):
        columns = _reflect_columns(columns, width)
    plan = _plan_rows(columns)

    row_types, wall_rows, height = _choose_wall_layout()
    grid = np.zeros((height, width), dtype=int)

    passive_lookup = {
        "passive0": plan["state0"],
        "passive1": plan["state1"],
        "passive2": plan["state2"],
        "passive3": plan["state3"],
    }
    for row_type, row_index in zip(row_types, wall_rows):
        gaps = plan[row_type] if row_type in plan else passive_lookup[row_type]
        allow_decoys = row_type.startswith("passive")
        grid[int(row_index), :] = _render_wall_row(width, gaps, allow_decoys=allow_decoys)

    marker_cols = sorted([columns["left_root"], columns["middle_root"], columns["right_root"]])
    for color, col in zip(MARKER_COLORS, marker_cols):
        grid[0, int(col)] = int(color)

    return grid


def _validate_instance(grid: np.ndarray) -> bool:
    try:
        trees = _analyze_grid(grid)
        out = main(grid)
    except Exception:
        return False

    if len(trees) != 3:
        return False

    leaf_counts = [int(tree["leaf_count"]) for tree in trees]
    classes = sorted(1 if t == 1 else 2 if t == 2 else 3 for t in leaf_counts)
    if classes != [1, 2, 3]:
        return False

    all_cells: list[tuple[int, int]] = []
    for tree in trees:
        all_cells.extend(tree["cells"])
    if len(set(all_cells)) != len(all_cells):
        return False

    if not np.all(out[grid == WALL] == WALL):
        return False

    marker_cols = np.flatnonzero(grid[0, :] != BG)
    for c in marker_cols:
        if int(out[0, c]) != int(grid[0, c]):
            return False

    present = set(map(int, np.unique(out)))
    if not {4, 7, 9}.issubset(present):
        return False

    if np.array_equal(out, grid):
        return False

    return True


def generate_input() -> np.ndarray:
    for _ in range(MAX_GENERATION_ATTEMPTS):
        grid = _make_grid()
        if _validate_instance(grid):
            return grid
    raise RuntimeError("failed to generate a valid tracing-tree input")
