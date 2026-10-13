from __future__ import annotations

from importlib.machinery import SourceFileLoader
from typing import Iterable

import numpy as np
import random


_ARC_UTILS = SourceFileLoader(
    "_arc_utils_key_combo_3_013_4",
    "/home/lixinyu/snap/new/solvers/arc_utils.py",
).load_module()

Color = _ARC_UTILS.Color
bbox = _ARC_UTILS.bbox
crop_bbox = _ARC_UTILS.crop_bbox
draw_line = _ARC_UTILS.draw_line
find_components = _ARC_UTILS.find_components
get_bg = _ARC_UTILS.get_bg

BG = int(Color.BLACK)
GRAY = int(Color.GRAY)

ColorPool = tuple[int, ...]


def _is_bg_path_row(grid: np.ndarray, row: int, c0: int, c1: int, bg: int) -> bool:
    a, b = sorted((int(c0), int(c1)))
    return bool(np.all(grid[int(row), a : b + 1] == bg))


def _is_bg_path_col(grid: np.ndarray, col: int, r0: int, r1: int, bg: int) -> bool:
    a, b = sorted((int(r0), int(r1)))
    return bool(np.all(grid[a : b + 1, int(col)] == bg))


def _collect_notches(grid: np.ndarray, bg: int) -> list[dict]:
    g = np.asarray(grid, dtype=int)
    notches: list[dict] = []

    for comp in find_components(g, bg=bg, connectivity=4, monochromatic=True):
        color = int(comp["color"])
        r0, c0, r1, c1 = comp["bbox"]
        h = int(r1 - r0 + 1)
        w = int(c1 - c0 + 1)

        patch = crop_bbox(g, comp["bbox"])
        miss = np.argwhere(patch != color)
        miss_count = int(miss.shape[0])

        assert miss_count in {0, 1}, (
            "rectangles are expected to be full rectangles or missing exactly one boundary pixel"
        )
        if miss_count == 0:
            continue

        assert miss_count == 1
        (mr, mc) = tuple(miss[0].tolist())
        if not (mr == 0 or mr == h - 1 or mc == 0 or mc == w - 1):
            raise AssertionError("the missing cell is not on the rectangle boundary")

        global_r = int(r0 + mr)
        global_c = int(c0 + mc)

        if mr == 0:
            if mr + 1 < h and int(patch[mr + 1, mc]) == color:
                side = "top"
            else:
                raise AssertionError("cannot determine notch opening side")
        elif mr == h - 1:
            if mr - 1 >= 0 and int(patch[mr - 1, mc]) == color:
                side = "bottom"
            else:
                raise AssertionError("cannot determine notch opening side")
        elif mc == 0:
            if mc + 1 < w and int(patch[mr, mc + 1]) == color:
                side = "left"
            else:
                raise AssertionError("cannot determine notch opening side")
        else:
            if mc - 1 >= 0 and int(patch[mr, mc - 1]) == color:
                side = "right"
            else:
                raise AssertionError("cannot determine notch opening side")

        notches.append(
            {
                "r": global_r,
                "c": global_c,
                "color": color,
                "side": side,
            }
        )

    return notches


def _can_pair(a: dict, b: dict, g: np.ndarray, bg: int) -> bool:
    if a["r"] == b["r"]:
        ar, ac = int(a["r"]), int(a["c"])
        br, bc = int(b["r"]), int(b["c"])
        if ac == bc:
            return False

        if a["side"] == "right" and b["side"] == "left" and ac < bc:
            return _is_bg_path_row(g, ar, ac, bc, bg)
        if b["side"] == "right" and a["side"] == "left" and bc < ac:
            return _is_bg_path_row(g, br, bc, ac, bg)
        return False

    if a["c"] == b["c"]:
        ar, ac = int(a["r"]), int(a["c"])
        br, bc = int(b["r"]), int(b["c"])
        if ar == br:
            return False

        if a["side"] == "bottom" and b["side"] == "top" and ar < br:
            return _is_bg_path_col(g, ac, ar, br, bg)
        if b["side"] == "bottom" and a["side"] == "top" and br < ar:
            return _is_bg_path_col(g, bc, br, ar, bg)

    return False


def _match_pairs(notches: list[dict], g: np.ndarray, bg: int) -> list[tuple[int, int]]:
    n = len(notches)
    assert n > 0 and n % 2 == 0, "expected an even number of notches"

    adj: list[list[int]] = [[] for _ in range(n)]
    for i in range(n):
        for j in range(i + 1, n):
            if _can_pair(notches[i], notches[j], g, bg):
                adj[i].append(j)
                adj[j].append(i)

    used = [False] * n
    solutions: list[list[tuple[int, int]]] = []

    def dfs(curr: list[tuple[int, int]]) -> None:
        if len(solutions) > 1:
            return
        if all(used):
            solutions.append(curr.copy())
            return

        i = used.index(False)
        used[i] = True
        for j in adj[i]:
            if used[j]:
                continue
            used[j] = True
            curr.append((i, j))
            dfs(curr)
            curr.pop()
            used[j] = False
        used[i] = False

    dfs([])
    assert len(solutions) == 1, "expected exactly one notch pairing"
    return solutions[0]


def main(grid: np.ndarray) -> np.ndarray:
    """
    Draw a grey line through each facing visible notch pair and fill each notch with the opposite rectangle color.

    Input structure:
    - Black background.
    - Several solid rectangles in non-background colors.
    - Some rectangles miss exactly one boundary pixel, forming 1-cell notches (ports).

    Rule:
    - A pair is any two notches that are on the same row or column, face each other, and have an unobstructed background straight segment between them.
    - For every pair, draw a grey line along that full straight segment.
    - Recolor each notch pixel with the color of the opposite rectangle.

    Key insight:
    - Notches are portals, not repair targets: the line only connects paired opposite-facing portals and the fill comes from the counterpart rectangle's color.
    """

    g = np.asarray(grid, dtype=int)
    assert g.ndim == 2, "expected a 2D grid"
    bg = get_bg(g)

    notches = _collect_notches(g, bg)
    assert len(notches) > 0, "expected at least one notch"

    pairs = _match_pairs(notches, g, bg)
    out = g.copy()

    for i, j in pairs:
        a = notches[i]
        b = notches[j]
        draw_line(
            out,
            int(a["r"]),
            int(a["c"]),
            end_r=int(b["r"]),
            end_c=int(b["c"]),
            color=GRAY,
        )

    for i, j in pairs:
        a = notches[i]
        b = notches[j]
        out[int(a["r"]), int(a["c"])] = int(b["color"])
        out[int(b["r"]), int(b["c"])] = int(a["color"])

    return out


def _first_unused_color(palette: ColorPool, used: Iterable[int]) -> int:
    used_set = set(int(c) for c in used)
    for c in palette:
        v = int(c)
        if v not in used_set:
            return v
    return int(palette[0])


def _row_segment_clear(grid: np.ndarray, row: int, c0: int, c1: int, bg: int) -> bool:
    return _is_bg_path_row(grid, row, c0, c1, bg)


def _col_segment_clear(grid: np.ndarray, col: int, r0: int, r1: int, bg: int) -> bool:
    return _is_bg_path_col(grid, col, r0, r1, bg)


def _place_horizontal_pair(
    grid: np.ndarray,
    blocked: np.ndarray,
    color_left: int,
    color_right: int,
    fixed_row: int | None = None,
    min_left_col: int = 0,
) -> dict | None:
    H, W = grid.shape
    for _ in range(700):
        h_left = int(np.random.randint(3, 7))
        h_right = int(np.random.randint(3, 7))
        w_left = int(np.random.randint(2, 7))
        w_right = int(np.random.randint(2, 7))

        top_left = int(np.random.randint(0, H - h_left + 1))
        top_right = int(np.random.randint(0, H - h_right + 1))

        row_min = max(top_left + 1, top_right + 1)
        row_max = min(top_left + h_left - 2, top_right + h_right - 2)
        if row_min > row_max:
            continue
        row = fixed_row if fixed_row is not None else int(np.random.randint(row_min, row_max + 1))
        if not (row_min <= row <= row_max):
            continue

        gap = int(np.random.randint(1, 4))
        max_left = W - (w_left + gap + w_right)
        left = max(0, int(min_left_col))
        if max_left < left:
            continue

        col_left = int(np.random.randint(left, max_left + 1))
        col_right = col_left + w_left + gap
        notch_left_col = col_left + w_left - 1
        notch_right_col = col_right

        if not _row_segment_clear(grid, row, notch_left_col, notch_right_col, BG):
            continue

        if blocked[top_left : top_left + h_left, col_left : col_left + w_left].any():
            continue
        if blocked[top_right : top_right + h_right, col_right : col_right + w_right].any():
            continue

        grid[top_left : top_left + h_left, col_left : col_left + w_left] = color_left
        grid[top_right : top_right + h_right, col_right : col_right + w_right] = color_right
        notch_left = (row, notch_left_col)
        notch_right = (row, notch_right_col)
        grid[row, notch_left_col] = BG
        grid[row, notch_right_col] = BG

        blocked[top_left : top_left + h_left, col_left : col_left + w_left] = True
        blocked[top_right : top_right + h_right, col_right : col_right + w_right] = True
        blocked[row, notch_left_col:notch_right_col + 1] = True

        return {
            "axis": "row",
            "notch_a": notch_left,
            "notch_b": notch_right,
            "color_a": color_left,
            "color_b": color_right,
        }

    return None


def _place_vertical_pair(
    grid: np.ndarray,
    blocked: np.ndarray,
    color_top: int,
    color_bottom: int,
) -> dict | None:
    H, W = grid.shape
    for _ in range(700):
        w_top = int(np.random.randint(3, 7))
        w_bottom = int(np.random.randint(3, 7))
        h_top = int(np.random.randint(2, 7))
        h_bottom = int(np.random.randint(2, 7))

        col_top = int(np.random.randint(0, W - w_top + 1))
        col_bottom = int(np.random.randint(0, W - w_bottom + 1))
        col_min = max(col_top + 1, col_bottom + 1)
        col_max = min(col_top + w_top - 2, col_bottom + w_bottom - 2)
        if col_min > col_max:
            continue
        col = int(np.random.randint(col_min, col_max + 1))

        gap = int(np.random.randint(1, 4))
        top_max = H - (h_top + gap + h_bottom)
        if top_max < 0:
            continue

        top_top = int(np.random.randint(0, top_max + 1))
        top_bottom = top_top + h_top + gap
        notch_top_row = top_top + h_top - 1
        notch_bottom_row = top_bottom

        if not _col_segment_clear(grid, col, notch_top_row, notch_bottom_row, BG):
            continue

        if blocked[top_top : top_top + h_top, col_top : col_top + w_top].any():
            continue
        if blocked[top_bottom : top_bottom + h_bottom, col_bottom : col_bottom + w_bottom].any():
            continue

        grid[top_top : top_top + h_top, col_top : col_top + w_top] = color_top
        grid[top_bottom : top_bottom + h_bottom, col_bottom : col_bottom + w_bottom] = color_bottom
        notch_top = (notch_top_row, col)
        notch_bottom = (notch_bottom_row, col)
        grid[notch_top_row, col] = BG
        grid[notch_bottom_row, col] = BG

        blocked[top_top : top_top + h_top, col_top : col_top + w_top] = True
        blocked[top_bottom : top_bottom + h_bottom, col_bottom : col_bottom + w_bottom] = True
        blocked[notch_top_row : notch_bottom_row + 1, col] = True

        return {
            "axis": "col",
            "notch_a": notch_top,
            "notch_b": notch_bottom,
            "color_a": color_top,
            "color_b": color_bottom,
        }

    return None


def _place_solid(
    grid: np.ndarray,
    blocked: np.ndarray,
    color: int,
    *,
    min_size: int = 2,
    max_size: int = 6,
    margin: int = 1,
) -> bool:
    H, W = grid.shape
    for _ in range(500):
        h = int(np.random.randint(min_size, max_size + 1))
        w = int(np.random.randint(min_size, max_size + 1))

        cand_r = []
        cand_c = []
        for r in range(H - h + 1):
            for c in range(W - w + 1):
                r0 = max(0, r - margin)
                c0 = max(0, c - margin)
                r1 = min(H, r + h + margin)
                c1 = min(W, c + w + margin)
                if np.any(blocked[r0:r1, c0:c1]):
                    continue
                cand_r.append(r)
                cand_c.append(c)

        if not cand_r:
            continue

        idx = int(np.random.randint(0, len(cand_r)))
        r = cand_r[idx]
        c = cand_c[idx]

        grid[r : r + h, c : c + w] = color
        blocked[r : r + h, c : c + w] = True
        return True

    return False


def _place_blocker(
    grid: np.ndarray,
    blocked: np.ndarray,
    left_pair: dict,
    right_pair: dict,
    color: int,
) -> bool:
    left_row, left_col = map(int, left_pair["notch_b"])
    right_row, right_col = map(int, right_pair["notch_a"])
    assert left_row == right_row and left_col < right_col, "blocked branch requires same row and ordering"

    H, W = grid.shape
    row = left_row
    start = left_col + 1
    end = right_col - 1
    if start >= end:
        return False

    for _ in range(500):
        width = int(np.random.randint(1, min(3, end - start + 1) + 1))
        height = int(np.random.randint(2, 5))
        if width <= 0 or height <= 0 or width > end - start + 1:
            continue

        c0 = int(np.random.randint(start, end - width + 2))
        r0_min = max(0, row - height + 1)
        r0_max = min(row, H - height)
        if r0_min > r0_max:
            continue
        r0 = int(np.random.randint(r0_min, r0_max + 1))

        if blocked[r0 : r0 + height, c0 : c0 + width].any():
            continue
        if not np.all(grid[r0 : r0 + height, c0 : c0 + width] == BG):
            continue

        grid[r0 : r0 + height, c0 : c0 + width] = color
        blocked[r0 : r0 + height, c0 : c0 + width] = True
        blocked[r0 : r0 + height, c0 : c0 + width] = True
        return True

    return False


def generate_input() -> np.ndarray:
    palette: ColorPool = (1, 2, 3, 4, 6, 7, 8, 9)
    for _ in range(500):
        H = int(np.random.randint(14, 29))
        W = int(np.random.randint(14, 29))
        grid = np.full((H, W), BG, dtype=int)
        blocked = np.zeros((H, W), dtype=bool)

        used_colors: set[int] = set()
        pairs: list[dict] = []

        include_block_branch = bool(np.random.random() < 0.35)
        pair_count = int(np.random.randint(1, 4))
        if include_block_branch:
            pair_count = max(2, pair_count)

        try:
            if include_block_branch and pair_count >= 2:
                color_a = int(_first_unused_color(palette, used_colors))
                used_colors.add(color_a)
                color_b = int(_first_unused_color(palette, used_colors))
                used_colors.add(color_b)

                first = _place_horizontal_pair(grid, blocked, color_a, color_b)
                if first is None:
                    raise RuntimeError("failed first blocked branch pair")

                second_left = first["notch_b"][1] + 3
                color_c = int(_first_unused_color(palette, used_colors))
                used_colors.add(color_c)
                color_d = int(_first_unused_color(palette, used_colors))
                used_colors.add(color_d)

                second = _place_horizontal_pair(
                    grid,
                    blocked,
                    color_c,
                    color_d,
                    fixed_row=int(first["notch_a"][0]),
                    min_left_col=second_left,
                )
                if second is None:
                    raise RuntimeError("failed second blocked branch pair")

                if second["notch_a"][1] - first["notch_b"][1] < 3:
                    raise RuntimeError("blocked branch gap too narrow")

                blocker_color = int(_first_unused_color(palette, used_colors))
                used_colors.add(blocker_color)
                if not _place_blocker(grid, blocked, first, second, blocker_color):
                    raise RuntimeError("failed to place blocker")

                pairs.extend([first, second])

            while len(pairs) < pair_count:
                color_a = int(_first_unused_color(palette, used_colors))
                used_colors.add(color_a)
                color_b = int(_first_unused_color(palette, used_colors))
                used_colors.add(color_b)

                if bool(np.random.random() < 0.55):
                    pair = _place_horizontal_pair(grid, blocked, color_a, color_b)
                else:
                    pair = _place_vertical_pair(grid, blocked, color_a, color_b)
                if pair is None:
                    raise RuntimeError("failed regular pair placement")
                pairs.append(pair)

            remaining_colors = [c for c in palette if c not in used_colors]
            if remaining_colors:
                max_extra = 2 if not include_block_branch else 1
                extra = int(
                    np.random.randint(0, min(len(remaining_colors), max_extra) + 1)
                )
                for i in range(extra):
                    color = remaining_colors[i % len(remaining_colors)]
                    if not _place_solid(grid, blocked, color, min_size=2, max_size=6, margin=1):
                        # If one distractor fails, continue with whatever has been placed.
                        break

            out = main(grid)
            assert out.shape == grid.shape

            # Ensure every generated input has at least one actual line segment.
            if np.any(out != grid):
                return grid
        except Exception:
            # Retry a fresh geometry until constraints are satisfied.
            continue

    raise RuntimeError("failed to generate a valid input for task 4")
