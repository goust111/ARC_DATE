from __future__ import annotations

import random
from collections import deque
from typing import Iterable

import numpy as np


BLACK = 0
GREY = 5
PALETTE = [1, 2, 3, 4, 6, 7, 8, 9]
GEN_RETRIES = 400


def _components(mask: np.ndarray) -> list[list[tuple[int, int]]]:
    h, w = mask.shape
    seen = np.zeros((h, w), dtype=bool)
    comps: list[list[tuple[int, int]]] = []
    for r in range(h):
        for c in range(w):
            if not mask[r, c] or seen[r, c]:
                continue
            seen[r, c] = True
            q = deque([(r, c)])
            comp: list[tuple[int, int]] = []
            while q:
                rr, cc = q.popleft()
                comp.append((rr, cc))
                for dr, dc in ((1, 0), (-1, 0), (0, 1), (0, -1)):
                    nr, nc = rr + dr, cc + dc
                    if 0 <= nr < h and 0 <= nc < w and mask[nr, nc] and not seen[nr, nc]:
                        seen[nr, nc] = True
                        q.append((nr, nc))
            comps.append(comp)
    return comps


def _bbox(points: Iterable[tuple[int, int]]) -> tuple[int, int, int, int]:
    pts = list(points)
    rows = [r for r, _ in pts]
    cols = [c for _, c in pts]
    return min(rows), min(cols), max(rows), max(cols)


def _find_cycle_strip(grid: np.ndarray) -> list[int]:
    if grid.shape[0] == 0 or grid.shape[1] == 0 or int(grid[0, 0]) == BLACK:
        raise ValueError("missing cycle strip")
    colors: list[int] = []
    c = 0
    while c < grid.shape[1] and int(grid[0, c]) != BLACK:
        colors.append(int(grid[0, c]))
        c += 1
    if not (2 <= len(colors) <= 5):
        raise ValueError("cycle strip must have length 2..5")
    return colors


def _is_frame_component(points: list[tuple[int, int]]) -> tuple[bool, tuple[int, int, int, int] | None]:
    r0, c0, r1, c1 = _bbox(points)
    h = r1 - r0 + 1
    w = c1 - c0 + 1
    if h < 3 or w < 3:
        return False, None
    border = set()
    for r in range(r0, r1 + 1):
        border.add((r, c0))
        border.add((r, c1))
    for c in range(c0, c1 + 1):
        border.add((r0, c))
        border.add((r1, c))
    point_set = set(points)
    if point_set != border:
        return False, None
    return True, (r0, c0, r1, c1)


def _count_to_new_color(cycle: list[int], holes: int) -> int:
    return int(cycle[holes % len(cycle)])


def _merge_centered(masks: list[np.ndarray]) -> np.ndarray:
    out_h = max(mask.shape[0] for mask in masks)
    out_w = max(mask.shape[1] for mask in masks)
    merged = np.zeros((out_h, out_w), dtype=bool)
    for mask in masks:
        h, w = mask.shape
        r0 = (out_h - h) // 2
        c0 = (out_w - w) // 2
        merged[r0 : r0 + h, c0 : c0 + w] |= mask
    return merged


def _render_grouped_output(cycle: list[int], groups: dict[int, list[np.ndarray]]) -> np.ndarray:
    blocks: list[np.ndarray] = []
    seen: set[int] = set()
    for color in cycle:
        if color in groups and color not in seen:
            seen.add(color)
            mask = _merge_centered(groups[color])
            block = np.zeros(mask.shape, dtype=int)
            block[mask] = int(color)
            blocks.append(block)
    if not blocks:
        raise ValueError("no output blocks")
    out_h = max(block.shape[0] for block in blocks)
    out_w = sum(block.shape[1] for block in blocks) + (len(blocks) - 1)
    out = np.zeros((out_h, out_w), dtype=int)
    c0 = 0
    for block in blocks:
        h, w = block.shape
        out[:h, c0 : c0 + w] = block
        c0 += w + 1
    return out


def main(grid: np.ndarray) -> np.ndarray:
    g = np.asarray(grid, dtype=int)
    cycle = _find_cycle_strip(g)

    groups: dict[int, list[np.ndarray]] = {}
    for comp in _components(g == GREY):
        ok, box = _is_frame_component(comp)
        if not ok or box is None:
            continue
        r0, c0, r1, c1 = box
        interior = g[r0 + 1 : r1, c0 + 1 : c1]
        if interior.size == 0:
            continue
        holes = int(np.sum(interior == BLACK))
        if not (1 <= holes <= len(cycle)):
            continue
        shape = interior != BLACK
        if not np.any(shape):
            continue
        new_color = _count_to_new_color(cycle, holes)
        groups.setdefault(new_color, []).append(shape)

    return _render_grouped_output(cycle, groups)


def _candidate_lattices(h: int, w: int) -> list[list[tuple[int, int]]]:
    lattices: list[list[tuple[int, int]]] = []
    for r_off in (0, 1):
        for c_off in (0, 1):
            pts = [(r, c) for r in range(r_off, h, 2) for c in range(c_off, w, 2)]
            if pts:
                lattices.append(pts)
    return lattices


def _sample_dims(
    rng: random.Random,
    holes: int,
    parity: tuple[int, int] | None = None,
    require_multiple_hole_sets: bool = False,
) -> tuple[int, int]:
    candidates: list[tuple[int, int]] = []
    for h in range(3, 9):
        for w in range(3, 9):
            if parity is not None and (h % 2, w % 2) != parity:
                continue
            if h * w <= holes:
                continue
            max_capacity = max(len(pts) for pts in _candidate_lattices(h, w))
            if max_capacity < holes:
                continue
            if require_multiple_hole_sets and max_capacity <= holes:
                continue
            candidates.append((h, w))
    if not candidates:
        raise ValueError("no interior size fits")
    return rng.choice(candidates)


def _sample_hole_positions(
    rng: random.Random,
    h: int,
    w: int,
    holes: int,
    forbid: set[tuple[int, int]] | None = None,
) -> list[tuple[int, int]]:
    options = [pts for pts in _candidate_lattices(h, w) if len(pts) >= holes]
    rng.shuffle(options)
    for pts in options:
        if forbid is None:
            return list(rng.sample(pts, holes))
        for _ in range(20):
            pick = set(rng.sample(pts, holes))
            if pick != forbid:
                return list(pick)
    raise ValueError("failed to sample holes")


def _make_shape_mask(h: int, w: int, holes: list[tuple[int, int]]) -> np.ndarray:
    mask = np.ones((h, w), dtype=bool)
    for r, c in holes:
        mask[r, c] = False
    return mask


def _place_rectangles(
    rng: random.Random,
    cycle: list[int],
    rects: list[dict],
) -> np.ndarray | None:
    total_area = sum((int(rect["shape"].shape[0]) + 2) * (int(rect["shape"].shape[1]) + 2) for rect in rects)
    max_oh = max(int(rect["shape"].shape[0]) + 2 for rect in rects)
    max_ow = max(int(rect["shape"].shape[1]) + 2 for rect in rects)

    min_h = max(12, max_oh + 4, int(total_area ** 0.5) + 6)
    min_w = max(len(cycle) + 4, max_ow + 4, int(total_area ** 0.5) + 8)
    max_h = 28
    max_w = 30
    if min_h > max_h or min_w > max_w:
        return None

    grid_h = rng.randint(min_h, max_h)
    grid_w = rng.randint(min_w, max_w)
    grid = np.zeros((grid_h, grid_w), dtype=int)
    busy = np.zeros((grid_h, grid_w), dtype=bool)

    for c, color in enumerate(cycle):
        grid[0, c] = int(color)
    busy[0 : min(grid_h, 2), 0 : min(grid_w, len(cycle) + 1)] = True

    ordered = sorted(
        rects,
        key=lambda rect: (int(rect["shape"].shape[0]) + 2) * (int(rect["shape"].shape[1]) + 2),
        reverse=True,
    )

    for rect in ordered:
        shape = rect["shape"]
        fill = int(rect["fill"])
        oh = int(shape.shape[0]) + 2
        ow = int(shape.shape[1]) + 2
        placed = False
        for _ in range(250):
            if grid_h - oh - 1 < 2 or grid_w - ow - 1 < 1:
                return None
            r0 = rng.randint(2, grid_h - oh - 1)
            c0 = rng.randint(1, grid_w - ow - 1)
            rr0 = max(0, r0 - 1)
            rr1 = min(grid_h, r0 + oh + 1)
            cc0 = max(0, c0 - 1)
            cc1 = min(grid_w, c0 + ow + 1)
            if busy[rr0:rr1, cc0:cc1].any():
                continue

            grid[r0, c0 : c0 + ow] = GREY
            grid[r0 + oh - 1, c0 : c0 + ow] = GREY
            grid[r0 : r0 + oh, c0] = GREY
            grid[r0 : r0 + oh, c0 + ow - 1] = GREY

            inner = np.full(shape.shape, fill, dtype=int)
            inner[~shape] = BLACK
            grid[r0 + 1 : r0 + oh - 1, c0 + 1 : c0 + ow - 1] = inner

            busy[r0 : r0 + oh, c0 : c0 + ow] = True
            placed = True
            break
        if not placed:
            return None

    return grid


def _build_expected(cycle: list[int], rects: list[dict]) -> np.ndarray:
    groups: dict[int, list[np.ndarray]] = {}
    for rect in rects:
        holes = int(rect["holes"])
        new_color = _count_to_new_color(cycle, holes)
        groups.setdefault(new_color, []).append(np.array(rect["shape"], dtype=bool))
    return _render_grouped_output(cycle, groups)


def _make_rect_specs(rng: random.Random, cycle: list[int]) -> tuple[list[dict], int]:
    m = len(cycle)
    n_rects = rng.randint(3, 6)
    distinct = rng.randint(2, min(m, n_rects - 1))
    present = rng.sample(list(range(1, m + 1)), distinct)
    if m not in present and rng.random() < 0.45:
        present[rng.randrange(len(present))] = m
    present = sorted(set(present))
    while len(present) < distinct:
        add = rng.randint(1, m)
        if add not in present:
            present.append(add)
    present.sort()

    merge_holes = rng.choice(present)
    group_sizes = {holes: 1 for holes in present}
    extras = n_rects - len(present)
    group_sizes[merge_holes] += 1
    extras -= 1
    for _ in range(extras):
        group_sizes[rng.choice(present)] += 1

    fill_palette = [color for color in PALETTE if color not in cycle]
    rects: list[dict] = []

    for holes in present:
        qty = group_sizes[holes]
        parity = (rng.randint(0, 1), rng.randint(0, 1))
        if holes == merge_holes:
            h, w = _sample_dims(rng, holes, parity, require_multiple_hole_sets=True)
            first_holes = _sample_hole_positions(rng, h, w, holes)
            second_holes = _sample_hole_positions(rng, h, w, holes, forbid=set(first_holes))
            base_fill = rng.choice(fill_palette)
            rects.append(
                {
                    "holes": holes,
                    "shape": _make_shape_mask(h, w, first_holes),
                    "fill": base_fill,
                }
            )
            rects.append(
                {
                    "holes": holes,
                    "shape": _make_shape_mask(h, w, second_holes),
                    "fill": rng.choice(fill_palette),
                }
            )
            qty -= 2

        for _ in range(qty):
            h, w = _sample_dims(rng, holes, parity)
            hole_positions = _sample_hole_positions(rng, h, w, holes)
            rects.append(
                {
                    "holes": holes,
                    "shape": _make_shape_mask(h, w, hole_positions),
                    "fill": rng.choice(fill_palette),
                }
            )

    groups: dict[int, list[np.ndarray]] = {}
    for rect in rects:
        groups.setdefault(int(rect["holes"]), []).append(rect["shape"])
    merged = _merge_centered(groups[merge_holes])
    member_views = []
    for mask in groups[merge_holes]:
        canvas = np.zeros_like(merged)
        mh, mw = mask.shape
        r0 = (merged.shape[0] - mh) // 2
        c0 = (merged.shape[1] - mw) // 2
        canvas[r0 : r0 + mh, c0 : c0 + mw] = mask
        member_views.append(canvas)
    if any(not np.any(merged != view) for view in member_views):
        raise ValueError("merge group degenerated to a single member")

    return rects, merge_holes


def _gen_attempt() -> np.ndarray | None:
    rng = random.Random()
    cycle = rng.sample(PALETTE, rng.randint(2, 5))

    try:
        rects, _merge_holes = _make_rect_specs(rng, cycle)
    except ValueError:
        return None

    expected = _build_expected(cycle, rects)
    grid = _place_rectangles(rng, cycle, rects)
    if grid is None:
        return None

    try:
        if not np.array_equal(main(grid), expected):
            return None
    except Exception:
        return None
    return grid


def generate_input() -> np.ndarray:
    for _ in range(GEN_RETRIES):
        grid = _gen_attempt()
        if grid is not None:
            return grid
    raise RuntimeError("failed to generate valid ARC input")
