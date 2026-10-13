from __future__ import annotations

from collections import Counter
import random

import numpy as np


def _get_bg(grid: np.ndarray) -> int:
    return int(Counter(map(int, grid.ravel())).most_common(1)[0][0])


def _bbox_of_positions(positions: list[tuple[int, int]]) -> tuple[int, int, int, int]:
    rows = [r for r, _ in positions]
    cols = [c for _, c in positions]
    return min(rows), min(cols), max(rows), max(cols)


def _render_positions(pos_to_color: dict[tuple[int, int], int]) -> np.ndarray:
    assert pos_to_color
    positions = list(pos_to_color)
    r0, c0, r1, c1 = _bbox_of_positions(positions)
    out = np.zeros((r1 - r0 + 1, c1 - c0 + 1), dtype=int)
    for (r, c), color in pos_to_color.items():
        out[r - r0, c - c0] = int(color)
    return out


def _axis_unmatched(mask: np.ndarray, axis_k: int) -> tuple[int, list[tuple[int, int]], list[tuple[int, int]]]:
    """Return unmatched count, visible unmatched pixels, and mirrored missing cells."""
    h, w = mask.shape
    pts = np.argwhere(mask)
    unmatched_visible: list[tuple[int, int]] = []
    missing_cells: list[tuple[int, int]] = []
    for r, c in pts:
        rr = int(r)
        cc = int(c)
        mc = int(axis_k) - cc
        if 0 <= mc < w and bool(mask[rr, mc]):
            continue
        unmatched_visible.append((rr, cc))
        if 0 <= mc < w:
            missing_cells.append((rr, mc))
    return len(unmatched_visible), unmatched_visible, missing_cells


def _best_vertical_axis(mask: np.ndarray) -> tuple[int, int, int]:
    """Return (best_axis_k, best_unmatched, gap_to_second_best)."""
    _h, w = mask.shape
    scores: list[tuple[int, int]] = []
    for axis_k in range(0, 2 * (w - 1) + 1):
        unmatched, _unpaired, _missing = _axis_unmatched(mask, axis_k)
        scores.append((unmatched, axis_k))
    scores.sort()
    best_unmatched, best_axis_k = scores[0]
    second_unmatched = scores[1][0] if len(scores) > 1 else best_unmatched
    return int(best_axis_k), int(best_unmatched), int(second_unmatched - best_unmatched)


def _gap_output_from_grid(grid: np.ndarray) -> np.ndarray:
    g = np.asarray(grid, dtype=int)
    bg = _get_bg(g)
    colors = [int(color) for color in np.unique(g) if int(color) != int(bg)]
    gaps: dict[tuple[int, int], int] = {}

    for color in colors:
        mask = g == int(color)
        best_axis_k, _best_unmatched, _gap = _best_vertical_axis(mask)
        _unmatched, _visible, missing = _axis_unmatched(mask, best_axis_k)
        for r, c in missing:
            if g[r, c] == bg:
                gaps[(int(r), int(c))] = int(color)

    assert gaps, "expected at least one mirrored gap"
    return _render_positions(gaps)


def main(grid: np.ndarray) -> np.ndarray:
    """
    Collect the per-color mirror gaps and crop just those missing cells.

    Input structure:
    - Black background with several non-black color layers.
    - Each color layer is almost vertically mirror-symmetric about its own axis.
    - A few cells that should have mirrored partners are missing, so those partner
      locations remain black in the input.

    Rule:
    - Process each non-black color independently and search all vertical mirror
      axes across the grid.
    - Choose the axis that leaves the fewest unmatched cells for that color.
    - For every unmatched visible cell, reflect it across that axis and record the
      mirrored black location as a missing cell of the same color.
    - Combine all colors' missing cells, then crop to their joint bounding box.

    Key insight:
    - The answer is not the nearly symmetric objects themselves but the cells that
      their symmetry implies should exist and are absent.
    """
    return _gap_output_from_grid(np.asarray(grid, dtype=int))


def _neighbors4(r: int, c: int) -> list[tuple[int, int]]:
    return [(r - 1, c), (r + 1, c), (r, c - 1), (r, c + 1)]


def _neighbors8(r: int, c: int) -> list[tuple[int, int]]:
    out = []
    for dr in (-1, 0, 1):
        for dc in (-1, 0, 1):
            if dr == 0 and dc == 0:
                continue
            out.append((r + dr, c + dc))
    return out


def _build_path(start: tuple[int, int], goal: tuple[int, int]) -> list[tuple[int, int]]:
    r, c = start
    gr, gc = goal
    path = [(r, c)]
    while (r, c) != (gr, gc):
        moves = []
        if r < gr:
            moves.append((r + 1, c))
        elif r > gr:
            moves.append((r - 1, c))
        if c < gc:
            moves.append((r, c + 1))
        elif c > gc:
            moves.append((r, c - 1))
        r, c = random.choice(moves)
        path.append((r, c))
    return path


def _connected_half_mask(h: int, side_w: int, target_count: int) -> set[tuple[int, int]]:
    seed = (random.randrange(h), side_w - 1)
    anchor_cols = [0]
    if side_w >= 3:
        anchor_cols.append(1)
    anchor = (random.randrange(h), random.choice(anchor_cols))

    cells = set(_build_path(seed, anchor))
    while len(cells) < target_count:
        frontier = []
        for r, c in sorted(cells):
            for nr, nc in _neighbors4(r, c):
                if 0 <= nr < h and 0 <= nc < side_w and (nr, nc) not in cells:
                    frontier.append((nr, nc))
        if not frontier:
            break
        weights = []
        for nr, nc in frontier:
            score = 1
            if nc <= 1:
                score += 3
            if nc == side_w - 1:
                score += 2
            weights.append(score)
        cells.add(random.choices(frontier, weights=weights, k=1)[0])
    return cells


def _full_mask_from_half(
    h: int,
    w: int,
    left_cells: set[tuple[int, int]],
    center_rows: set[int],
) -> np.ndarray:
    side_w = w // 2
    mask = np.zeros((h, w), dtype=bool)
    for r, c in left_cells:
        mask[r, c] = True
        mask[r, w - 1 - c] = True
    if w % 2 == 1:
        axis_c = side_w
        for r in center_rows:
            mask[r, axis_c] = True
    return mask


def _pick_gap_cells(mask: np.ndarray, side: str) -> set[tuple[int, int]] | None:
    h, w = mask.shape
    side_limit = w // 2
    bands = [1, 2, 3]
    candidates: list[tuple[int, int]] = []
    for band in bands:
        if side == "left":
            candidates = [
                (r, c)
                for r in range(h)
                for c in range(min(side_limit, band))
                if mask[r, c]
            ]
        else:
            candidates = [
                (r, c)
                for r in range(h)
                for c in range(max(side_limit, w - band), w)
                if mask[r, c]
            ]
        if len(candidates) >= 2:
            break
    if len(candidates) < 2:
        return None

    candidate_set = set(candidates)
    min_count = 2
    max_count = min(5, len(candidates))
    for _ in range(30):
        need = random.randint(min_count, max_count)
        seed = random.choice(candidates)
        chosen = {seed}
        frontier = {nbr for nbr in _neighbors8(*seed) if nbr in candidate_set}
        while len(chosen) < need and frontier:
            nxt = random.choice(sorted(frontier))
            frontier.remove(nxt)
            chosen.add(nxt)
            for nbr in _neighbors8(*nxt):
                if nbr in candidate_set and nbr not in chosen:
                    frontier.add(nbr)
        if len(chosen) < need:
            leftovers = [cell for cell in candidates if cell not in chosen]
            random.shuffle(leftovers)
            for cell in leftovers:
                chosen.add(cell)
                if len(chosen) == need:
                    break
        rows = {r for r, _ in chosen}
        cols = {c for _, c in chosen}
        if len(chosen) >= 3 and (len(rows) == 1 or len(cols) == 1):
            continue
        local = _render_positions({cell: 1 for cell in chosen})
        if np.array_equal(local, np.fliplr(local)):
            continue
        return chosen
    return None


def _make_local_object(parity: str, gap_side: str) -> dict | None:
    for _ in range(120):
        h = random.randint(4, 7)
        if parity == "odd":
            side_w = random.randint(2, 3)
            w = 2 * side_w + 1
            center_options = [1, 2]
        else:
            side_w = random.randint(2, 3)
            w = 2 * side_w
            center_options = [0]

        area = h * w
        full_min = max(10, int(np.ceil(0.38 * area)))
        full_max = min(26, int(np.floor(0.72 * area)))
        combos = []
        seed_to_anchor_floor = side_w
        for center_count in center_options:
            for left_count in range(seed_to_anchor_floor, h * side_w + 1):
                total = 2 * left_count + center_count
                if full_min <= total <= full_max:
                    combos.append((center_count, left_count))
        if not combos:
            continue

        center_count, left_target = random.choice(combos)
        left_cells = _connected_half_mask(h, side_w, left_target)
        if len(left_cells) != left_target:
            continue

        center_rows: set[int] = set()
        if center_count:
            near_axis_rows = sorted({r for r, c in left_cells if c == side_w - 1})
            random.shuffle(near_axis_rows)
            picked = near_axis_rows[:center_count]
            if len(picked) < center_count:
                others = [r for r in range(h) if r not in picked]
                random.shuffle(others)
                picked.extend(others[: center_count - len(picked)])
            center_rows = set(picked)

        full_mask = _full_mask_from_half(h, w, left_cells, center_rows)
        if not full_mask.any():
            continue

        gap_cells = _pick_gap_cells(full_mask, gap_side)
        if gap_cells is None:
            continue

        visible_mask = full_mask.copy()
        for r, c in gap_cells:
            visible_mask[r, c] = False

        axis_k, best_unmatched, gap_to_second = _best_vertical_axis(visible_mask)
        if axis_k != w - 1 or best_unmatched != len(gap_cells) or gap_to_second < 2:
            continue

        missing_local = sorted(gap_cells)
        unmatched_local = _axis_unmatched(visible_mask, axis_k)[1]
        wrong_crop = _render_positions({cell: 1 for cell in unmatched_local})
        expected_crop = _render_positions({cell: 1 for cell in missing_local})
        if np.array_equal(wrong_crop, expected_crop):
            continue

        return {
            "full_mask": full_mask,
            "visible_mask": visible_mask,
            "missing_local": missing_local,
            "axis_local_k": axis_k,
            "h": h,
            "w": w,
            "gap_side": gap_side,
        }
    return None


def _object_color_pool(count: int) -> list[int]:
    return random.sample([1, 2, 3, 4, 5, 6, 7, 8, 9], count)


def _side_pattern(n_objects: int) -> list[str]:
    if n_objects == 2:
        return ["right", "left"]
    middle = random.choice(["left", "right"])
    return ["right", middle, "left"]


def _parity_pattern(n_objects: int) -> list[str]:
    if n_objects == 2:
        return random.sample(["odd", "even"], 2)
    pattern = ["odd", "even", random.choice(["odd", "even"])]
    random.shuffle(pattern)
    return pattern


def _place_objects(objs: list[dict]) -> tuple[np.ndarray, dict[tuple[int, int], int]] | None:
    widths = [obj["w"] for obj in objs]
    heights = [obj["h"] for obj in objs]
    gaps = [random.randint(2, 3) for _ in range(len(objs) - 1)]
    left_margin = random.randint(1, 3)
    right_margin = random.randint(1, 3)
    grid_w = left_margin + sum(widths) + sum(gaps) + right_margin
    if grid_w > 30:
        return None

    band_top = random.randint(1, 5)
    band_bottom = band_top + max(heights) + 4
    grid_h = min(30, band_bottom + random.randint(1, 3))
    if grid_h < max(heights) + 4:
        return None

    grid = np.zeros((grid_h, grid_w), dtype=int)
    missing_abs: dict[tuple[int, int], int] = {}

    col = left_margin
    axes = []
    for idx, obj in enumerate(objs):
        h = obj["h"]
        w = obj["w"]
        max_top = grid_h - h - 1
        if max_top < 1:
            return None
        top = max(1, min(max_top, band_top + random.randint(0, 2)))
        color = int(obj["color"])
        mask = obj["visible_mask"]
        obj["top"] = top
        obj["left"] = col
        obj["axis_global_k"] = 2 * col + int(obj["axis_local_k"])
        for r in range(h):
            for c in range(w):
                if mask[r, c]:
                    if grid[top + r, col + c] != 0:
                        return None
                    grid[top + r, col + c] = color
        for r, c in obj["missing_local"]:
            missing_abs[(top + r, col + c)] = color
        axes.append(2 * col + obj["axis_local_k"])
        col += w
        if idx < len(gaps):
            col += gaps[idx]

    if len(set(axes)) != len(axes):
        return None
    return grid, missing_abs


def _global_wrong_output(grid: np.ndarray) -> np.ndarray:
    g = np.asarray(grid, dtype=int)
    fg = g != 0
    axis_k, _best, _gap = _best_vertical_axis(fg)
    pos_to_color: dict[tuple[int, int], int] = {}
    for color in [int(c) for c in np.unique(g) if int(c) != 0]:
        mask = g == int(color)
        _count, _unmatched, missing = _axis_unmatched(mask, axis_k)
        for r, c in missing:
            if g[r, c] == 0:
                pos_to_color[(r, c)] = int(color)
    if not pos_to_color:
        return np.zeros((1, 1), dtype=int)
    return _render_positions(pos_to_color)


def _is_semantically_valid(grid: np.ndarray, expected: np.ndarray, objs: list[dict]) -> bool:
    if len({int(obj["color"]) for obj in objs}) < 2:
        return False
    if int(np.count_nonzero(expected)) < 5:
        return False
    if len([int(c) for c in np.unique(expected) if int(c) != 0]) < 2:
        return False
    if np.array_equal(main(grid), expected) is False:
        return False

    wrong_visible: dict[tuple[int, int], int] = {}
    for obj in objs:
        color = int(obj["color"])
        coords = np.argwhere(grid == color)
        mask = np.zeros_like(grid, dtype=bool)
        mask[coords[:, 0], coords[:, 1]] = True
        axis_k, best_unmatched, gap_to_second = _best_vertical_axis(mask)
        if axis_k != int(obj["axis_global_k"]):
            return False
        if best_unmatched != len(obj["missing_abs"]) or gap_to_second < 2:
            return False
        _count, visible_unmatched, _missing = _axis_unmatched(mask, axis_k)
        for r, c in visible_unmatched:
            wrong_visible[(int(r), int(c))] = color
    if wrong_visible and np.array_equal(_render_positions(wrong_visible), expected):
        return False
    if np.array_equal(_global_wrong_output(grid), expected):
        return False
    return True


def _attempt_generate() -> np.ndarray | None:
    n_objects = random.choice([2, 3])
    colors = _object_color_pool(n_objects)
    gap_sides = _side_pattern(n_objects)
    parities = _parity_pattern(n_objects)

    objs: list[dict] = []
    for color, side, parity in zip(colors, gap_sides, parities):
        obj = _make_local_object(parity, side)
        if obj is None:
            return None
        obj["color"] = int(color)
        objs.append(obj)

    placed = _place_objects(objs)
    if placed is None:
        return None
    grid, missing_abs = placed

    for obj in objs:
        color = int(obj["color"])
        obj["missing_abs"] = [pos for pos, clr in missing_abs.items() if clr == color]

    expected = _render_positions(missing_abs)
    if not _is_semantically_valid(grid, expected, objs):
        return None
    return grid


def generate_input() -> np.ndarray:
    for _ in range(300):
        grid = _attempt_generate()
        if grid is not None:
            return grid
    raise RuntimeError("failed to generate a valid symmetry-gap ARC input")
