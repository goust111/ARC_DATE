from __future__ import annotations

import random

import numpy as np


BG = 0
GREY = 5
COLOR_POOL = [1, 2, 3, 4, 6, 7, 8, 9]
GEN_RETRIES = 400
NEIGHBORS4 = [(-1, 0), (1, 0), (0, -1), (0, 1)]
NEIGHBORS8 = [
    (-1, -1), (-1, 0), (-1, 1),
    (0, -1),           (0, 1),
    (1, -1),  (1, 0),  (1, 1),
]


def _components(mask: np.ndarray) -> list[list[tuple[int, int]]]:
    h, w = mask.shape
    seen = np.zeros((h, w), dtype=bool)
    comps: list[list[tuple[int, int]]] = []
    for r0 in range(h):
        for c0 in range(w):
            if not mask[r0, c0] or seen[r0, c0]:
                continue
            seen[r0, c0] = True
            stack = [(r0, c0)]
            comp: list[tuple[int, int]] = []
            while stack:
                r, c = stack.pop()
                comp.append((r, c))
                for dr, dc in NEIGHBORS4:
                    nr, nc = r + dr, c + dc
                    if 0 <= nr < h and 0 <= nc < w and mask[nr, nc] and not seen[nr, nc]:
                        seen[nr, nc] = True
                        stack.append((nr, nc))
            comps.append(comp)
    return comps


def _bbox(coords: list[tuple[int, int]]) -> tuple[int, int, int, int]:
    rows = [r for r, _ in coords]
    cols = [c for _, c in coords]
    return min(rows), min(cols), max(rows), max(cols)


def _legend_from_components(g: np.ndarray) -> tuple[set[tuple[int, int]], list[int]]:
    h, w = g.shape
    comps = _components((g != BG) & (g != GREY))
    candidates: list[tuple[set[tuple[int, int]], list[int]]] = []
    for comp in comps:
        r0, c0, r1, c1 = _bbox(comp)
        height = r1 - r0 + 1
        width = c1 - c0 + 1
        touches_corner = (r0 == 0 or r1 == h - 1) and (c0 == 0 or c1 == w - 1)
        if not touches_corner:
            continue
        if not ((height == 1 and width >= 2) or (width == 1 and height >= 2)):
            continue
        values = [int(g[r, c]) for r, c in comp]
        if len(set(values)) != len(values):
            continue
        if width == 1:
            ordered = sorted(comp, key=lambda p: p[0])
        else:
            ordered = sorted(comp, key=lambda p: p[1])
        colors = [int(g[r, c]) for r, c in ordered]
        candidates.append((set(comp), colors))

    assert len(candidates) == 1, "expected exactly one corner legend strip"
    return candidates[0]


def _is_frame_component(g: np.ndarray, comp: list[tuple[int, int]]) -> bool:
    r0, c0, r1, c1 = _bbox(comp)
    height = r1 - r0 + 1
    width = c1 - c0 + 1
    if height < 3 or width < 3:
        return False
    values = {int(g[r, c]) for r, c in comp}
    if len(values) != 1:
        return False
    sub = np.zeros((height, width), dtype=bool)
    for r, c in comp:
        sub[r - r0, c - c0] = True
    expected = np.zeros((height, width), dtype=bool)
    expected[0, :] = True
    expected[-1, :] = True
    expected[:, 0] = True
    expected[:, -1] = True
    return np.array_equal(sub, expected)


def _frame_boxes(g: np.ndarray, legend_cells: set[tuple[int, int]]) -> list[tuple[int, int, int, int]]:
    boxes: list[tuple[int, int, int, int]] = []
    for comp in _components((g != BG) & (g != GREY)):
        if comp[0] in legend_cells:
            continue
        if _is_frame_component(g, comp):
            boxes.append(_bbox(comp))
    assert boxes, "expected at least one rectangular frame"
    return boxes


def _depth_at(r: int, c: int, frame_boxes: list[tuple[int, int, int, int]]) -> int:
    return sum(1 for r0, c0, r1, c1 in frame_boxes if r0 < r < r1 and c0 < c < c1)


def main(grid: np.ndarray) -> np.ndarray:
    """
    Recolor grey pixels by the number of rectangular frame borders that contain them.

    The color legend is a thin corner strip whose top-to-bottom (or left-to-right) order
    maps depth 1, 2, 3, ... to output colors.
    """
    g = np.asarray(grid, dtype=int)
    assert g.ndim == 2, "expected a 2D grid"

    legend_cells, legend_colors = _legend_from_components(g)
    frames = _frame_boxes(g, legend_cells)

    out = g.copy()
    grey_points = np.argwhere(g == GREY)
    for r, c in grey_points:
        depth = _depth_at(int(r), int(c), frames)
        if 1 <= depth <= len(legend_colors):
            out[int(r), int(c)] = int(legend_colors[depth - 1])
    return out


def _draw_frame(g: np.ndarray, box: tuple[int, int, int, int], color: int) -> None:
    r0, c0, r1, c1 = box
    g[r0, c0 : c1 + 1] = color
    g[r1, c0 : c1 + 1] = color
    g[r0 : r1 + 1, c0] = color
    g[r0 : r1 + 1, c1] = color


def _nested_boxes(
    outer_box: tuple[int, int, int, int],
    gaps: list[int],
) -> list[tuple[int, int, int, int]]:
    r0, c0, r1, c1 = outer_box
    inset = 0
    boxes = []
    for i in range(len(gaps) + 1):
        boxes.append((r0 + inset, c0 + inset, r1 - inset, c1 - inset))
        if i < len(gaps):
            inset += gaps[i]
    return boxes


def _place_legend(g: np.ndarray, colors: list[int], corner: str) -> None:
    h, w = g.shape
    rows = list(range(len(colors))) if corner.startswith("t") else list(range(h - len(colors), h))
    col = 0 if corner.endswith("l") else w - 1
    for r, color in zip(rows, colors, strict=False):
        g[r, col] = int(color)


def _frame_mask(g: np.ndarray, frame_boxes: list[tuple[int, int, int, int]]) -> np.ndarray:
    mask = np.zeros_like(g, dtype=bool)
    for box in frame_boxes:
        r0, c0, r1, c1 = box
        mask[r0, c0 : c1 + 1] = True
        mask[r1, c0 : c1 + 1] = True
        mask[r0 : r1 + 1, c0] = True
        mask[r0 : r1 + 1, c1] = True
    return mask


def _safe_grey_candidates(
    shape: tuple[int, int],
    frame_boxes: list[tuple[int, int, int, int]],
    frame_mask: np.ndarray,
) -> dict[int, list[tuple[int, int]]]:
    h, w = shape
    candidates: dict[int, list[tuple[int, int]]] = {}
    for r in range(1, h - 1):
        for c in range(1, w - 1):
            if frame_mask[r, c]:
                continue
            touches_frame = False
            for dr, dc in NEIGHBORS4:
                if frame_mask[r + dr, c + dc]:
                    touches_frame = True
                    break
            if touches_frame:
                continue
            depth = _depth_at(r, c, frame_boxes)
            candidates.setdefault(depth, []).append((r, c))
    return candidates


def _pick_scattered(
    candidates: list[tuple[int, int]],
    wanted: int,
    used: set[tuple[int, int]],
) -> list[tuple[int, int]]:
    pool = candidates[:]
    random.shuffle(pool)
    blocked = set(used)
    for r, c in list(used):
        for dr, dc in NEIGHBORS8:
            blocked.add((r + dr, c + dc))

    picked: list[tuple[int, int]] = []
    for cell in pool:
        if cell in blocked:
            continue
        picked.append(cell)
        rr, cc = cell
        blocked.add(cell)
        for dr, dc in NEIGHBORS8:
            blocked.add((rr + dr, cc + dc))
        if len(picked) == wanted:
            break
    return picked


def _expected_output(
    inp: np.ndarray,
    frame_boxes: list[tuple[int, int, int, int]],
    legend_colors: list[int],
) -> np.ndarray:
    out = inp.copy()
    for r, c in np.argwhere(inp == GREY):
        depth = _depth_at(int(r), int(c), frame_boxes)
        if 1 <= depth <= len(legend_colors):
            out[int(r), int(c)] = int(legend_colors[depth - 1])
    return out


def _generate_attempt() -> np.ndarray | None:
    depth = random.choice([3, 4])
    colors = random.sample(COLOR_POOL, 2 * depth)
    frame_colors = colors[:depth]
    legend_colors = colors[depth:]

    if depth == 3:
        inner_h = random.randint(5, 10)
        inner_w = random.randint(5, 10)
    else:
        inner_h = random.randint(5, 7)
        inner_w = random.randint(5, 7)

    gaps = [random.choice([3, 4]) for _ in range(depth - 1)]
    outer_h = inner_h + 2 * sum(gaps)
    outer_w = inner_w + 2 * sum(gaps)

    min_h = outer_h + 4
    min_w = outer_w + 4
    max_h = min(30, outer_h + 8)
    max_w = min(30, outer_w + 8)
    if min_h > max_h or min_w > max_w:
        return None

    h = random.randint(min_h, max_h)
    w = random.randint(min_w, max_w)
    r0 = random.randint(2, h - outer_h - 2)
    c0 = random.randint(2, w - outer_w - 2)
    outer_box = (r0, c0, r0 + outer_h - 1, c0 + outer_w - 1)
    frame_boxes = _nested_boxes(outer_box, gaps)

    grid = np.zeros((h, w), dtype=int)
    corner = random.choice(["tl", "tr", "bl", "br"])
    _place_legend(grid, legend_colors, corner)
    for box, color in zip(frame_boxes, frame_colors, strict=False):
        _draw_frame(grid, box, color)

    frame_mask = _frame_mask(grid, frame_boxes)
    candidates = _safe_grey_candidates(grid.shape, frame_boxes, frame_mask)
    if any(not candidates.get(d) for d in range(1, depth + 1)):
        return None

    chosen: set[tuple[int, int]] = set()
    for d in range(1, depth + 1):
        depth_candidates = candidates[d]
        want = 1 + random.randint(0, min(2, len(depth_candidates) - 1))
        picked = _pick_scattered(depth_candidates, want, chosen)
        if not picked:
            return None
        chosen.update(picked)

    if random.random() < 0.6 and candidates.get(0):
        outside_candidates = candidates[0]
        extra_want = 1 + random.randint(0, min(1, len(outside_candidates) - 1))
        chosen.update(_pick_scattered(outside_candidates, extra_want, chosen))

    for r, c in chosen:
        grid[r, c] = GREY

    expected = _expected_output(grid, frame_boxes, legend_colors)
    if np.array_equal(grid, expected):
        return None
    if not np.array_equal(main(grid), expected):
        return None
    return grid


def generate_input() -> np.ndarray:
    for _ in range(GEN_RETRIES):
        grid = _generate_attempt()
        if grid is not None:
            return grid
    raise RuntimeError("failed to generate a valid nesting-depth input")
