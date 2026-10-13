from __future__ import annotations

import random

import numpy as np


BG = 0
START_COLOR = 7
PALETTE = [1, 2, 3, 4, 5, 6, 8, 9]

UP = "up"
DOWN = "down"
LEFT = "left"
RIGHT = "right"

DIR_VEC = {
    UP: (-1, 0),
    DOWN: (1, 0),
    LEFT: (0, -1),
    RIGHT: (0, 1),
}

BODY_OFFSETS = {
    UP: ((-2, 0), (-1, -1), (-1, 0), (-1, 1), (0, 0), (1, 0)),
    DOWN: ((-1, 0), (0, 0), (1, 0), (1, -1), (1, 1), (2, 0)),
    LEFT: ((0, -2), (0, -1), (0, 0), (-1, -1), (1, -1), (0, 1)),
    RIGHT: ((0, -1), (0, 0), (0, 1), (-1, 1), (1, 1), (0, 2)),
}

ROD_OFFSETS = {
    UP: ((-1, 0), (0, 0), (1, 0)),
    DOWN: ((-1, 0), (0, 0), (1, 0)),
    LEFT: ((0, -1), (0, 0), (0, 1)),
    RIGHT: ((0, -1), (0, 0), (0, 1)),
}

TIP_OFFSET = {
    UP: (-2, 0),
    DOWN: (2, 0),
    LEFT: (0, -2),
    RIGHT: (0, 2),
}

TAIL_OFFSET = {
    UP: (2, 0),
    DOWN: (-2, 0),
    LEFT: (0, 2),
    RIGHT: (0, -2),
}

TEMPLATE_BY_BBOX = {
    ((0, 1), (1, 0), (1, 1), (1, 2), (2, 1), (3, 1)): {
        "dir": UP,
        "center_in_bbox": (2, 1),
        "tip_in_bbox": (0, 1),
        "tail_in_bbox": (4, 1),
        "rod_in_bbox": ((1, 1), (2, 1), (3, 1)),
    },
    ((0, 1), (1, 1), (2, 0), (2, 1), (2, 2), (3, 1)): {
        "dir": DOWN,
        "center_in_bbox": (1, 1),
        "tip_in_bbox": (3, 1),
        "tail_in_bbox": (-1, 1),
        "rod_in_bbox": ((0, 1), (1, 1), (2, 1)),
    },
    ((0, 1), (1, 0), (1, 1), (1, 2), (1, 3), (2, 1)): {
        "dir": LEFT,
        "center_in_bbox": (1, 2),
        "tip_in_bbox": (1, 0),
        "tail_in_bbox": (1, 4),
        "rod_in_bbox": ((1, 1), (1, 2), (1, 3)),
    },
    ((0, 2), (1, 0), (1, 1), (1, 2), (1, 3), (2, 2)): {
        "dir": RIGHT,
        "center_in_bbox": (1, 1),
        "tip_in_bbox": (1, 3),
        "tail_in_bbox": (1, -1),
        "rod_in_bbox": ((1, 0), (1, 1), (1, 2)),
    },
}

GEN_ATTEMPTS = 400
GEN_H_RANGE = (18, 29)
GEN_W_RANGE = (18, 29)
GEN_ARROW_COUNT = (3, 5)
GEN_STEP_RANGE = (5, 9)
GEN_FIRST_CENTER_RANGE = (4, 7)
GEN_MIN_FINAL_PATH = 2


def _components_same_color(grid: np.ndarray, ignore: np.ndarray | None = None) -> list[dict[str, object]]:
    h, w = grid.shape
    seen = np.zeros((h, w), dtype=bool)
    if ignore is None:
        ignore = np.zeros((h, w), dtype=bool)
    out: list[dict[str, object]] = []
    for r in range(h):
        for c in range(w):
            if seen[r, c] or ignore[r, c]:
                continue
            color = int(grid[r, c])
            if color == BG:
                continue
            stack = [(r, c)]
            seen[r, c] = True
            coords: list[tuple[int, int]] = []
            while stack:
                rr, cc = stack.pop()
                coords.append((rr, cc))
                for dr, dc in ((-1, 0), (1, 0), (0, -1), (0, 1)):
                    nr = rr + dr
                    nc = cc + dc
                    if 0 <= nr < h and 0 <= nc < w and not seen[nr, nc] and not ignore[nr, nc]:
                        if int(grid[nr, nc]) == color:
                            seen[nr, nc] = True
                            stack.append((nr, nc))
            out.append({"color": color, "coords": coords})
    return out


def _find_start_block(grid: np.ndarray) -> tuple[tuple[int, int], np.ndarray]:
    h, w = grid.shape
    candidates: list[tuple[int, int]] = []
    for r0, c0 in ((0, 0), (0, w - 2), (h - 2, 0), (h - 2, w - 2)):
        if r0 < 0 or c0 < 0:
            continue
        patch = grid[r0 : r0 + 2, c0 : c0 + 2]
        if patch.shape != (2, 2):
            continue
        if int(patch[0, 0]) == BG:
            continue
        if np.all(patch == patch[0, 0]):
            candidates.append((r0, c0))
    assert len(candidates) == 1, "expected exactly one corner 2x2 start block"
    r0, c0 = candidates[0]
    mask = np.zeros_like(grid, dtype=bool)
    mask[r0 : r0 + 2, c0 : c0 + 2] = True
    return (r0, c0), mask


def _match_arrow_component(coords: list[tuple[int, int]]) -> dict[str, object] | None:
    if len(coords) != 6:
        return None
    rs = [r for r, _ in coords]
    cs = [c for _, c in coords]
    r0 = min(rs)
    c0 = min(cs)
    normalized = tuple(sorted((r - r0, c - c0) for r, c in coords))
    info = TEMPLATE_BY_BBOX.get(normalized)
    if info is None:
        return None
    center_dr, center_dc = info["center_in_bbox"]
    tip_dr, tip_dc = info["tip_in_bbox"]
    tail_dr, tail_dc = info["tail_in_bbox"]
    center = (r0 + center_dr, c0 + center_dc)
    tip = (r0 + tip_dr, c0 + tip_dc)
    tail = (r0 + tail_dr, c0 + tail_dc)
    rod = tuple((r0 + dr, c0 + dc) for dr, dc in info["rod_in_bbox"])
    return {
        "dir": info["dir"],
        "center": center,
        "tip": tip,
        "tail": tail,
        "rod": rod,
        "bbox": (r0, c0, max(rs), max(cs)),
    }


def _extract_arrows(grid: np.ndarray, start_mask: np.ndarray) -> list[dict[str, object]]:
    arrows: list[dict[str, object]] = []
    body_cells: set[tuple[int, int]] = set()
    tail_cells: set[tuple[int, int]] = set()
    for comp in _components_same_color(grid, ignore=start_mask):
        match = _match_arrow_component(comp["coords"])
        if match is None:
            continue
        tr, tc = match["tail"]
        if not (0 <= tr < grid.shape[0] and 0 <= tc < grid.shape[1]):
            continue
        tail_color = int(grid[tr, tc])
        body_color = int(comp["color"])
        if tail_color in (BG, body_color):
            continue
        arrow = {
            "body_color": body_color,
            "tail_color": tail_color,
            "dir": match["dir"],
            "center": match["center"],
            "tip": match["tip"],
            "tail": match["tail"],
            "rod": tuple(match["rod"]),
            "bbox": match["bbox"],
        }
        arrows.append(arrow)
        body_cells.update(comp["coords"])
        tail_cells.add(match["tail"])

    occupied = set(map(tuple, np.argwhere((grid != BG) & (~start_mask))))
    known = body_cells | tail_cells
    assert occupied == known, "all non-start foreground cells must belong to arrows or tails"
    assert len(tail_cells) == len(arrows), "each arrow must have one distinct tail cell"
    return arrows


def _visual_center(arrow: dict[str, object]) -> tuple[float, float]:
    r0, c0, r1, c1 = arrow["bbox"]
    return ((r0 + r1) / 2.0, (c0 + c1) / 2.0)


def main(grid: np.ndarray) -> np.ndarray:
    """
    Chain scattered arrows together by extending each tip toward the next arrow's rod,
    using the source arrow's tail-marker color for that segment.
    """
    g = np.asarray(grid, dtype=int)
    assert g.ndim == 2
    h, w = g.shape

    (sr, sc), start_mask = _find_start_block(g)
    arrows = _extract_arrows(g, start_mask)
    assert arrows, "expected at least one arrow"

    start_center = (sr + 0.5, sc + 0.5)
    first = min(
        range(len(arrows)),
        key=lambda idx: (
            (_visual_center(arrows[idx])[0] - start_center[0]) ** 2
            + (_visual_center(arrows[idx])[1] - start_center[1]) ** 2,
            idx,
        ),
    )

    rod_lookup: dict[tuple[int, int], int] = {}
    for idx, arrow in enumerate(arrows):
        for cell in arrow["rod"]:
            rod_lookup[cell] = idx

    out = g.copy()
    seen: set[int] = set()
    cur = first
    while cur not in seen:
        seen.add(cur)
        arrow = arrows[cur]
        dr, dc = DIR_VEC[arrow["dir"]]
        color = int(arrow["tail_color"])
        r = arrow["tip"][0] + dr
        c = arrow["tip"][1] + dc
        next_idx = None

        while 0 <= r < h and 0 <= c < w:
            hit = rod_lookup.get((r, c))
            if hit is not None and hit != cur:
                next_idx = hit
                break
            if int(g[r, c]) != BG:
                break
            out[r, c] = color
            r += dr
            c += dc

        if next_idx is None:
            break
        cur = next_idx

    return out


def _arrow_layout(center: tuple[int, int], direction: str) -> dict[str, object]:
    cr, cc = center
    body = {(cr + dr, cc + dc) for dr, dc in BODY_OFFSETS[direction]}
    rod = {(cr + dr, cc + dc) for dr, dc in ROD_OFFSETS[direction]}
    tip = (cr + TIP_OFFSET[direction][0], cc + TIP_OFFSET[direction][1])
    tail = (cr + TAIL_OFFSET[direction][0], cc + TAIL_OFFSET[direction][1])
    return {
        "center": center,
        "dir": direction,
        "body": body,
        "rod": rod,
        "tip": tip,
        "tail": tail,
        "cells": body | {tail},
    }


def _in_bounds(shape: tuple[int, int], cells: set[tuple[int, int]]) -> bool:
    h, w = shape
    return all(0 <= r < h and 0 <= c < w for r, c in cells)


def _perpendicular_dirs(direction: str) -> tuple[str, str]:
    if direction in (LEFT, RIGHT):
        return (UP, DOWN)
    return (LEFT, RIGHT)


def _lane_to_center(tip: tuple[int, int], direction: str, target_center: tuple[int, int]) -> list[tuple[int, int]] | None:
    dr, dc = DIR_VEC[direction]
    r = tip[0] + dr
    c = tip[1] + dc
    cells: list[tuple[int, int]] = []
    if dr != 0:
        if c != target_center[1]:
            return None
    else:
        if r != target_center[0]:
            return None
    while (r, c) != target_center:
        cells.append((r, c))
        r += dr
        c += dc
        if len(cells) > 200:
            return None
    return cells


def _lane_to_edge(shape: tuple[int, int], tip: tuple[int, int], direction: str) -> list[tuple[int, int]]:
    h, w = shape
    dr, dc = DIR_VEC[direction]
    r = tip[0] + dr
    c = tip[1] + dc
    cells: list[tuple[int, int]] = []
    while 0 <= r < h and 0 <= c < w:
        cells.append((r, c))
        r += dr
        c += dc
    return cells


def _first_is_unique_nearest(arrows: list[dict[str, object]]) -> bool:
    first_center = (0.5, 0.5)
    dists = []
    for arrow in arrows:
        r0 = min(r for r, _ in arrow["body"])
        c0 = min(c for _, c in arrow["body"])
        r1 = max(r for r, _ in arrow["body"])
        c1 = max(c for _, c in arrow["body"])
        center = ((r0 + r1) / 2.0, (c0 + c1) / 2.0)
        dists.append((center[0] - first_center[0]) ** 2 + (center[1] - first_center[1]) ** 2)
    best = min(dists)
    return dists[0] == best and sum(abs(d - best) < 1e-9 for d in dists) == 1


def _build_canonical_chain(h: int, w: int, count: int) -> list[dict[str, object]] | None:
    start_block = {(0, 0), (0, 1), (1, 0), (1, 1)}
    first_hi_r = min(GEN_FIRST_CENTER_RANGE[1], h - 5)
    first_hi_c = min(GEN_FIRST_CENTER_RANGE[1], w - 5)
    if first_hi_r < GEN_FIRST_CENTER_RANGE[0] or first_hi_c < GEN_FIRST_CENTER_RANGE[0]:
        return None

    for _ in range(200):
        c0 = (
            random.randint(GEN_FIRST_CENTER_RANGE[0], first_hi_r),
            random.randint(GEN_FIRST_CENTER_RANGE[0], first_hi_c),
        )
        d0 = random.choice((RIGHT, DOWN))
        first = _arrow_layout(c0, d0)
        if not _in_bounds((h, w), first["cells"]):
            continue

        arrows = [first]
        occupied = set(start_block) | set(first["cells"])
        reserved: set[tuple[int, int]] = set()
        success = True

        for _idx in range(count - 1):
            cur = arrows[-1]
            step_choices = list(range(GEN_STEP_RANGE[0], GEN_STEP_RANGE[1] + 1))
            random.shuffle(step_choices)
            dir_choices = list(_perpendicular_dirs(cur["dir"]))
            random.shuffle(dir_choices)
            candidates = []
            for next_dir in dir_choices:
                for step in step_choices:
                    dr, dc = DIR_VEC[cur["dir"]]
                    next_center = (cur["center"][0] + dr * step, cur["center"][1] + dc * step)
                    nxt = _arrow_layout(next_center, next_dir)
                    if not _in_bounds((h, w), nxt["cells"]):
                        continue
                    if occupied & nxt["cells"]:
                        continue
                    if reserved & nxt["cells"]:
                        continue
                    lane = _lane_to_center(cur["tip"], cur["dir"], next_center)
                    if lane is None or len(lane) < 2:
                        continue
                    lane_set = set(lane)
                    if lane_set & occupied or lane_set & reserved:
                        continue
                    candidates.append((nxt, lane_set))
            if not candidates:
                success = False
                break
            nxt, lane_set = random.choice(candidates)
            reserved |= lane_set
            occupied |= set(nxt["cells"])
            arrows.append(nxt)

        if not success:
            continue

        final_lane = _lane_to_edge((h, w), arrows[-1]["tip"], arrows[-1]["dir"])
        if len(final_lane) < GEN_MIN_FINAL_PATH:
            continue
        final_lane_set = set(final_lane)
        if final_lane_set & occupied or final_lane_set & reserved:
            continue
        if not _first_is_unique_nearest(arrows):
            continue
        return arrows

    return None


def _paint_canonical(arrows: list[dict[str, object]], shape: tuple[int, int]) -> np.ndarray:
    h, w = shape
    grid = np.full((h, w), BG, dtype=int)
    grid[0:2, 0:2] = START_COLOR

    body_colors = random.sample(PALETTE, len(arrows))
    tail_colors = [random.choice([color for color in PALETTE if color != body_colors[i]]) for i in range(len(arrows))]
    for arrow, body_color, tail_color in zip(arrows, body_colors, tail_colors, strict=False):
        for r, c in arrow["body"]:
            grid[r, c] = body_color
        tr, tc = arrow["tail"]
        grid[tr, tc] = tail_color
    return grid


def generate_input() -> np.ndarray:
    for _ in range(GEN_ATTEMPTS):
        h = random.randint(*GEN_H_RANGE)
        w = random.randint(*GEN_W_RANGE)
        count = random.randint(*GEN_ARROW_COUNT)
        arrows = _build_canonical_chain(h, w, count)
        if arrows is None:
            continue

        inp = _paint_canonical(arrows, (h, w))
        inp = np.rot90(inp, random.randint(0, 3)).copy()
        try:
            out = main(inp)
        except Exception:
            continue
        if out.shape != inp.shape:
            continue
        if np.array_equal(out, inp):
            continue
        return inp
    raise RuntimeError(f"generate_input failed after {GEN_ATTEMPTS} attempts")
