from __future__ import annotations

import random

import numpy as np


BG = 8
ARROW = 9
NON_SPECIAL_COLORS = [0, 1, 2, 3, 4, 5, 6, 7]
DIR4 = [(-1, 0), (1, 0), (0, -1), (0, 1)]
MAX_GENERATION_ATTEMPTS = 400


def _components_from_mask(mask: np.ndarray) -> list[np.ndarray]:
    h, w = mask.shape
    seen = np.zeros((h, w), dtype=bool)
    components: list[np.ndarray] = []

    for r in range(h):
        for c in range(w):
            if not bool(mask[r, c]) or bool(seen[r, c]):
                continue
            stack = [(r, c)]
            seen[r, c] = True
            coords: list[tuple[int, int]] = []
            while stack:
                rr, cc = stack.pop()
                coords.append((rr, cc))
                for dr, dc in DIR4:
                    nr, nc = rr + dr, cc + dc
                    if 0 <= nr < h and 0 <= nc < w and bool(mask[nr, nc]) and not bool(seen[nr, nc]):
                        seen[nr, nc] = True
                        stack.append((nr, nc))
            components.append(np.array(coords, dtype=int))
    return components


def _extract_objects(grid: np.ndarray, arrow_mask: np.ndarray) -> list[dict]:
    h, w = grid.shape
    seen = np.zeros((h, w), dtype=bool)
    objects: list[dict] = []

    for r in range(h):
        for c in range(w):
            if bool(seen[r, c]) or bool(arrow_mask[r, c]) or int(grid[r, c]) == BG:
                continue
            color = int(grid[r, c])
            stack = [(r, c)]
            seen[r, c] = True
            coords: list[tuple[int, int]] = []
            while stack:
                rr, cc = stack.pop()
                coords.append((rr, cc))
                for dr, dc in DIR4:
                    nr, nc = rr + dr, cc + dc
                    if not (0 <= nr < h and 0 <= nc < w):
                        continue
                    if bool(seen[nr, nc]) or bool(arrow_mask[nr, nc]) or int(grid[nr, nc]) != color:
                        continue
                    seen[nr, nc] = True
                    stack.append((nr, nc))

            pts = np.array(coords, dtype=int)
            r0, c0 = pts.min(axis=0)
            r1, c1 = pts.max(axis=0)
            rel = pts - np.array([r0, c0], dtype=int)
            mask = np.zeros((int(r1 - r0 + 1), int(c1 - c0 + 1)), dtype=bool)
            mask[rel[:, 0], rel[:, 1]] = True
            objects.append(
                {
                    "color": color,
                    "r0": int(r0),
                    "c0": int(c0),
                    "h": int(mask.shape[0]),
                    "w": int(mask.shape[1]),
                    "cells": np.argwhere(mask),
                }
            )

    return objects


def _parse_arrow(coords: np.ndarray, shape: tuple[int, int]) -> dict | None:
    h, w = shape
    if int(coords.shape[0]) != 3:
        return None

    rows = coords[:, 0]
    cols = coords[:, 1]
    matches: list[dict] = []

    if (
        int(cols.min()) == 0
        and int(cols.max()) == 1
        and int(rows.max() - rows.min()) == 1
        and int(np.count_nonzero(cols == 0)) == 2
        and int(np.count_nonzero(cols == 1)) == 1
    ):
        tip = coords[cols == 1][0]
        matches.append(
            {
                "axis": "h",
                "line": int(tip[0]),
                "direction": "right",
                "side": "left",
                "tip": (int(tip[0]), int(tip[1])),
                "coords": coords.copy(),
            }
        )

    if (
        int(cols.min()) == w - 2
        and int(cols.max()) == w - 1
        and int(rows.max() - rows.min()) == 1
        and int(np.count_nonzero(cols == w - 1)) == 2
        and int(np.count_nonzero(cols == w - 2)) == 1
    ):
        tip = coords[cols == w - 2][0]
        matches.append(
            {
                "axis": "h",
                "line": int(tip[0]),
                "direction": "left",
                "side": "right",
                "tip": (int(tip[0]), int(tip[1])),
                "coords": coords.copy(),
            }
        )

    if (
        int(rows.min()) == 0
        and int(rows.max()) == 1
        and int(cols.max() - cols.min()) == 1
        and int(np.count_nonzero(rows == 0)) == 2
        and int(np.count_nonzero(rows == 1)) == 1
    ):
        tip = coords[rows == 1][0]
        matches.append(
            {
                "axis": "v",
                "line": int(tip[1]),
                "direction": "down",
                "side": "top",
                "tip": (int(tip[0]), int(tip[1])),
                "coords": coords.copy(),
            }
        )

    if (
        int(rows.min()) == h - 2
        and int(rows.max()) == h - 1
        and int(cols.max() - cols.min()) == 1
        and int(np.count_nonzero(rows == h - 1)) == 2
        and int(np.count_nonzero(rows == h - 2)) == 1
    ):
        tip = coords[rows == h - 2][0]
        matches.append(
            {
                "axis": "v",
                "line": int(tip[1]),
                "direction": "up",
                "side": "bottom",
                "tip": (int(tip[0]), int(tip[1])),
                "coords": coords.copy(),
            }
        )

    if len(matches) != 1:
        return None
    return matches[0]


def _detect_arrows(grid: np.ndarray) -> tuple[list[dict], np.ndarray]:
    comps = _components_from_mask(grid == ARROW)
    arrows: list[dict] = []
    arrow_mask = np.zeros_like(grid, dtype=bool)

    for comp in comps:
        arrow = _parse_arrow(comp, grid.shape)
        if arrow is None:
            continue
        arrows.append(arrow)
        arrow_mask[comp[:, 0], comp[:, 1]] = True

    arrows.sort(key=lambda item: (item["tip"][0], item["tip"][1], item["side"]))
    return arrows, arrow_mask


def _intersects_line(obj: dict, arrow: dict) -> bool:
    if arrow["axis"] == "h":
        return bool(np.any(obj["cells"][:, 0] + int(obj["r0"]) == int(arrow["line"])))
    return bool(np.any(obj["cells"][:, 1] + int(obj["c0"]) == int(arrow["line"])))


def _apply_arrow(objects: list[dict], arrow: dict, shape: tuple[int, int]) -> None:
    h, w = shape
    affected = [obj for obj in objects if _intersects_line(obj, arrow)]
    if not affected:
        return

    if arrow["direction"] == "right":
        ordered = sorted(
            affected,
            key=lambda obj: (int(obj["c0"]) + int(obj["w"]) - 1, int(obj["c0"]), int(obj["r0"])),
            reverse=True,
        )
        cursor = w - 1
        for obj in ordered:
            obj["c0"] = int(cursor - int(obj["w"]) + 1)
            cursor = int(obj["c0"]) - 1
    elif arrow["direction"] == "left":
        ordered = sorted(
            affected,
            key=lambda obj: (int(obj["c0"]), int(obj["r0"]), int(obj["c0"]) + int(obj["w"]) - 1),
        )
        cursor = 0
        for obj in ordered:
            obj["c0"] = int(cursor)
            cursor = int(obj["c0"]) + int(obj["w"])
    elif arrow["direction"] == "down":
        ordered = sorted(
            affected,
            key=lambda obj: (int(obj["r0"]) + int(obj["h"]) - 1, int(obj["r0"]), int(obj["c0"])),
            reverse=True,
        )
        cursor = h - 1
        for obj in ordered:
            obj["r0"] = int(cursor - int(obj["h"]) + 1)
            cursor = int(obj["r0"]) - 1
    else:
        ordered = sorted(
            affected,
            key=lambda obj: (int(obj["r0"]), int(obj["c0"]), int(obj["r0"]) + int(obj["h"]) - 1),
        )
        cursor = 0
        for obj in ordered:
            obj["r0"] = int(cursor)
            cursor = int(obj["r0"]) + int(obj["h"])


def _render(objects: list[dict], shape: tuple[int, int]) -> np.ndarray:
    out = np.full(shape, BG, dtype=int)
    for obj in objects:
        rows = int(obj["r0"]) + obj["cells"][:, 0]
        cols = int(obj["c0"]) + obj["cells"][:, 1]
        out[rows, cols] = int(obj["color"])
    return out


def main(grid: np.ndarray) -> np.ndarray:
    """
    Compress every object intersecting an arrow-selected row or column in the arrow direction.

    Arrows are 3-pixel maroon edge markers. A left-edge arrow points right, a right-edge
    arrow points left, a top-edge arrow points down, and a bottom-edge arrow points up.
    """
    g = np.asarray(grid, dtype=int)
    assert g.ndim == 2

    arrows, arrow_mask = _detect_arrows(g)
    objects = _extract_objects(g, arrow_mask)

    for arrow in arrows:
        _apply_arrow(objects, arrow, g.shape)

    return _render(objects, g.shape)


def _split_nonnegative(total: int, parts: int) -> list[int]:
    values = [0] * parts
    for _ in range(total):
        values[random.randrange(parts)] += 1
    return values


def _random_connected_mask(max_h: int, max_w: int) -> np.ndarray:
    box_h = random.randint(1, max_h)
    box_w = random.randint(1, max_w)
    total = box_h * box_w
    min_cells = 1 if random.random() < 0.2 else min(2, total)
    target = random.randint(min_cells, total)

    seed = (random.randrange(box_h), random.randrange(box_w))
    cells = {seed}
    frontier: set[tuple[int, int]] = set()

    def add_neighbors(r: int, c: int) -> None:
        for dr, dc in DIR4:
            nr, nc = r + dr, c + dc
            if 0 <= nr < box_h and 0 <= nc < box_w and (nr, nc) not in cells:
                frontier.add((nr, nc))

    add_neighbors(*seed)
    while len(cells) < target and frontier:
        cell = random.choice(tuple(frontier))
        frontier.remove(cell)
        cells.add(cell)
        add_neighbors(*cell)

    rows = [r for r, _ in cells]
    cols = [c for _, c in cells]
    r0, r1 = min(rows), max(rows)
    c0, c1 = min(cols), max(cols)
    mask = np.zeros((r1 - r0 + 1, c1 - c0 + 1), dtype=bool)
    for r, c in cells:
        mask[r - r0, c - c0] = True
    return mask


def _scatter_horizontal(widths: list[int], gravity_right: bool, width: int, margin: int = 2) -> list[int] | None:
    span = sum(widths)
    budget = int(width - margin - span)
    if budget < 1:
        return None

    total_extra = random.randint(1, budget)
    positions = [0] * len(widths)

    if gravity_right:
        extras = _split_nonnegative(total_extra, len(widths))
        between = extras[:-1]
        right_gap = extras[-1]
        cursor = width - 1 - right_gap
        for i in range(len(widths) - 1, -1, -1):
            positions[i] = int(cursor - widths[i] + 1)
            cursor = positions[i] - 1
            if i > 0:
                cursor -= between[i - 1]
        if positions[0] < margin:
            return None
    else:
        extras = _split_nonnegative(total_extra, len(widths))
        left_gap = extras[0]
        between = extras[1:]
        cursor = left_gap
        for i, obj_w in enumerate(widths):
            positions[i] = int(cursor)
            cursor += obj_w
            if i < len(widths) - 1:
                cursor += between[i]
        if width - cursor < margin:
            return None

    return positions


def _scatter_vertical(heights: list[int], gravity_down: bool, height: int, margin: int = 2) -> list[int] | None:
    span = sum(heights)
    budget = int(height - margin - span)
    if budget < 1:
        return None

    total_extra = random.randint(1, budget)
    positions = [0] * len(heights)

    if gravity_down:
        extras = _split_nonnegative(total_extra, len(heights))
        between = extras[:-1]
        bottom_gap = extras[-1]
        cursor = height - 1 - bottom_gap
        for i in range(len(heights) - 1, -1, -1):
            positions[i] = int(cursor - heights[i] + 1)
            cursor = positions[i] - 1
            if i > 0:
                cursor -= between[i - 1]
        if positions[0] < margin:
            return None
    else:
        extras = _split_nonnegative(total_extra, len(heights))
        top_gap = extras[0]
        between = extras[1:]
        cursor = top_gap
        for i, obj_h in enumerate(heights):
            positions[i] = int(cursor)
            cursor += obj_h
            if i < len(heights) - 1:
                cursor += between[i]
        if height - cursor < margin:
            return None

    return positions


def _arrow_cells(side: str, line: int, h: int, w: int) -> list[tuple[int, int]]:
    if side == "left":
        if random.random() < 0.5:
            return [(line, 1), (line, 0), (line - 1, 0)]
        return [(line, 1), (line, 0), (line + 1, 0)]
    if side == "right":
        if random.random() < 0.5:
            return [(line, w - 2), (line, w - 1), (line - 1, w - 1)]
        return [(line, w - 2), (line, w - 1), (line + 1, w - 1)]
    if side == "top":
        if random.random() < 0.5:
            return [(1, line), (0, line), (0, line - 1)]
        return [(1, line), (0, line), (0, line + 1)]
    if random.random() < 0.5:
        return [(h - 2, line), (h - 1, line), (h - 1, line - 1)]
    return [(h - 2, line), (h - 1, line), (h - 1, line + 1)]


def _paint_mask(grid: np.ndarray, mask: np.ndarray, r0: int, c0: int, color: int) -> None:
    rows, cols = np.nonzero(mask)
    grid[r0 + rows, c0 + cols] = int(color)


def _can_place_with_moat(occupied: np.ndarray, mask: np.ndarray, r0: int, c0: int) -> bool:
    h, w = occupied.shape
    rows, cols = np.nonzero(mask)
    cells = {(int(r0 + rr), int(c0 + cc)) for rr, cc in zip(rows, cols)}

    for rr, cc in cells:
        if not (0 <= rr < h and 0 <= cc < w):
            return False
        if bool(occupied[rr, cc]):
            return False
        for dr in (-1, 0, 1):
            for dc in (-1, 0, 1):
                nr, nc = rr + dr, cc + dc
                if 0 <= nr < h and 0 <= nc < w and (nr, nc) not in cells and bool(occupied[nr, nc]):
                    return False
    return True


def _choose_lines(candidates: list[int], count: int, min_gap: int) -> list[int] | None:
    if count == 1:
        return [random.choice(candidates)] if candidates else None
    options = [
        [a, b]
        for i, a in enumerate(candidates)
        for b in candidates[i + 1 :]
        if abs(a - b) >= min_gap
    ]
    if not options:
        return None
    return random.choice(options)


def _make_horizontal_group(line: int, side: str, h: int, w: int, colors: list[int]) -> list[dict] | None:
    gravity_right = side == "left"

    for _ in range(120):
        n_objs = random.randint(2, 3)
        if len(colors) < n_objs:
            return None

        masks: list[np.ndarray] = []
        tops: list[int] = []
        widths: list[int] = []
        group_ok = True

        for _idx in range(n_objs):
            placed = False
            for _ in range(80):
                mask = _random_connected_mask(3, 4)
                occ_rows = np.flatnonzero(mask.any(axis=1))
                anchor_row = int(random.choice(list(occ_rows)))
                r0 = int(line - anchor_row)
                if 1 <= r0 and r0 + mask.shape[0] <= h - 1:
                    masks.append(mask)
                    tops.append(r0)
                    widths.append(int(mask.shape[1]))
                    placed = True
                    break
            if not placed:
                group_ok = False
                break

        if not group_ok:
            continue

        lefts = _scatter_horizontal(widths, gravity_right, w, margin=2)
        if lefts is None:
            continue

        objects: list[dict] = []
        for mask, r0, c0 in zip(masks, tops, lefts):
            color = colors.pop()
            objects.append({"color": color, "mask": mask, "r0": int(r0), "c0": int(c0)})
        return objects

    return None


def _make_vertical_group(line: int, side: str, h: int, w: int, colors: list[int]) -> list[dict] | None:
    gravity_down = side == "top"

    for _ in range(120):
        n_objs = random.randint(2, 3)
        if len(colors) < n_objs:
            return None

        masks: list[np.ndarray] = []
        lefts: list[int] = []
        heights: list[int] = []
        group_ok = True

        for _idx in range(n_objs):
            placed = False
            for _ in range(80):
                mask = _random_connected_mask(4, 3)
                occ_cols = np.flatnonzero(mask.any(axis=0))
                anchor_col = int(random.choice(list(occ_cols)))
                c0 = int(line - anchor_col)
                if 1 <= c0 and c0 + mask.shape[1] <= w - 1:
                    masks.append(mask)
                    lefts.append(c0)
                    heights.append(int(mask.shape[0]))
                    placed = True
                    break
            if not placed:
                group_ok = False
                break

        if not group_ok:
            continue

        tops = _scatter_vertical(heights, gravity_down, h, margin=2)
        if tops is None:
            continue

        objects: list[dict] = []
        for mask, r0, c0 in zip(masks, tops, lefts):
            color = colors.pop()
            objects.append({"color": color, "mask": mask, "r0": int(r0), "c0": int(c0)})
        return objects

    return None


def _add_distractors(
    grid: np.ndarray,
    axis: str,
    forbidden: set[int],
    colors: list[int],
) -> None:
    occupied = grid != BG
    n_extra = random.choice([0, 0, 1, 2])

    for _ in range(n_extra):
        if not colors:
            return
        placed = False
        for _ in range(120):
            mask = _random_connected_mask(3, 3)
            mh, mw = mask.shape
            if axis == "h":
                row_choices = [
                    r0
                    for r0 in range(1, grid.shape[0] - mh)
                    if all((r0 + dr) not in forbidden for dr in range(mh))
                ]
                if not row_choices:
                    break
                col_choices = list(range(1, grid.shape[1] - mw))
            else:
                col_choices = [
                    c0
                    for c0 in range(1, grid.shape[1] - mw)
                    if all((c0 + dc) not in forbidden for dc in range(mw))
                ]
                if not col_choices:
                    break
                row_choices = list(range(1, grid.shape[0] - mh))

            if not row_choices or not col_choices:
                break

            r0 = random.choice(row_choices)
            c0 = random.choice(col_choices)
            if not _can_place_with_moat(occupied, mask, r0, c0):
                continue

            _paint_mask(grid, mask, r0, c0, colors.pop())
            occupied = grid != BG
            placed = True
            break
        if not placed:
            return


def _gen_attempt() -> np.ndarray | None:
    h = random.randint(10, 16)
    w = random.randint(10, 16)
    n_arrows = random.choice([1, 2])
    axis = random.choice(["h", "v"])

    line_candidates = list(range(2, h - 2)) if axis == "h" else list(range(2, w - 2))
    lines = _choose_lines(line_candidates, n_arrows, min_gap=5)
    if lines is None:
        return None

    color_pool = random.sample(NON_SPECIAL_COLORS, len(NON_SPECIAL_COLORS))
    arrows: list[tuple[str, int]] = []
    objects: list[dict] = []

    for line in lines:
        if axis == "h":
            side = random.choice(["left", "right"])
            group = _make_horizontal_group(line, side, h, w, color_pool)
        else:
            side = random.choice(["top", "bottom"])
            group = _make_vertical_group(line, side, h, w, color_pool)
        if group is None:
            return None
        arrows.append((side, line))
        objects.extend(group)

    grid = np.full((h, w), BG, dtype=int)
    for obj in objects:
        _paint_mask(grid, obj["mask"], int(obj["r0"]), int(obj["c0"]), int(obj["color"]))

    for side, line in arrows:
        cells = _arrow_cells(side, line, h, w)
        for rr, cc in cells:
            if int(grid[rr, cc]) != BG:
                return None
            grid[rr, cc] = ARROW

    if axis == "h":
        forbidden = {
            r
            for obj in objects
            for r in range(int(obj["r0"]), int(obj["r0"]) + int(obj["mask"].shape[0]))
        }
    else:
        forbidden = {
            c
            for obj in objects
            for c in range(int(obj["c0"]), int(obj["c0"]) + int(obj["mask"].shape[1]))
        }
    _add_distractors(grid, axis, forbidden, color_pool)

    try:
        out = main(grid)
    except Exception:
        return None

    if np.array_equal(out, grid):
        return None
    if bool(np.any(out == ARROW)):
        return None
    return grid


def generate_input() -> np.ndarray:
    for _ in range(MAX_GENERATION_ATTEMPTS):
        grid = _gen_attempt()
        if grid is not None:
            return grid
    raise RuntimeError("failed to generate a valid puzzle input")
