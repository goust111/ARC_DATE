from __future__ import annotations

from collections import deque
import random
from typing import Iterable

import numpy as np


BG = 0
ARROW = 1
BLOCK = 3

ARROW_MAX_AREA = 5
GRID_RANGE = (18, 24)
BLOCK_COUNT_OPTIONS = (2, 3)
ARROW_COUNT_OPTIONS = (3, 4)
BLOCK_SIZES = tuple(
    (h, w)
    for h in range(3, 6)
    for w in range(3, 6)
    if 9 <= h * w <= 20
)
MAX_ATTEMPTS = 600


def _find_components(grid: np.ndarray, bg: int = BG) -> list[dict]:
    h, w = grid.shape
    seen = np.zeros((h, w), dtype=bool)
    out: list[dict] = []
    for r in range(h):
        for c in range(w):
            if seen[r, c] or int(grid[r, c]) == bg:
                continue
            color = int(grid[r, c])
            q = deque([(r, c)])
            seen[r, c] = True
            coords: list[tuple[int, int]] = []
            while q:
                rr, cc = q.popleft()
                coords.append((rr, cc))
                for dr, dc in ((-1, 0), (1, 0), (0, -1), (0, 1)):
                    nr = rr + dr
                    nc = cc + dc
                    if 0 <= nr < h and 0 <= nc < w and not seen[nr, nc] and int(grid[nr, nc]) == color:
                        seen[nr, nc] = True
                        q.append((nr, nc))
            arr = np.asarray(coords, dtype=int)
            r0, c0 = arr.min(axis=0)
            r1, c1 = arr.max(axis=0)
            out.append(
                {
                    "color": color,
                    "coords": arr,
                    "bbox": (int(r0), int(c0), int(r1), int(c1)),
                    "area": int(arr.shape[0]),
                }
            )
    return out


def _rect_mask(shape: tuple[int, int], rects: Iterable[tuple[int, int, int, int]]) -> np.ndarray:
    mask = np.zeros(shape, dtype=bool)
    for r0, c0, h, w in rects:
        mask[r0 : r0 + h, c0 : c0 + w] = True
    return mask


def _paint_rects(grid: np.ndarray, rects: Iterable[tuple[int, int, int, int]], color: int) -> None:
    for r0, c0, h, w in rects:
        grid[r0 : r0 + h, c0 : c0 + w] = int(color)


def _sign(x: float) -> int:
    if x > 1e-9:
        return 1
    if x < -1e-9:
        return -1
    return 0


def _arrow_direction(component: dict) -> tuple[tuple[int, int], tuple[int, int], np.ndarray]:
    coords = component["coords"].astype(float)
    r0, c0, r1, c1 = component["bbox"]
    center = np.array([(r0 + r1) / 2.0, (c0 + c1) / 2.0], dtype=float)
    centroid = coords.mean(axis=0)
    offset = centroid - center

    sprite = np.zeros((r1 - r0 + 1, c1 - c0 + 1), dtype=bool)
    rel = component["coords"] - np.array([r0, c0], dtype=int)
    sprite[rel[:, 0], rel[:, 1]] = True
    holes = np.argwhere(~sprite)
    if holes.size > 0:
        hole_center = holes.mean(axis=0) + np.array([r0, c0], dtype=float)
        asym = hole_center - centroid
    else:
        asym = -offset

    if abs(offset[0]) >= abs(offset[1]):
        primary = (_sign(offset[0]) or _sign(asym[0]), 0)
        secondary = (0, _sign(offset[1]) or _sign(asym[1]) or 1)
    else:
        primary = (0, _sign(offset[1]) or _sign(asym[1]))
        secondary = (_sign(offset[0]) or _sign(asym[0]) or 1, 0)

    assert primary in {(-1, 0), (1, 0), (0, -1), (0, 1)}
    return primary, secondary, offset


def _ray_start(component: dict, primary: tuple[int, int], secondary: tuple[int, int]) -> tuple[int, int]:
    r0, c0, r1, c1 = component["bbox"]
    dr, dc = primary
    sr, sc = secondary
    if dr == -1:
        return r0 - 1, c0 if sc < 0 else c1
    if dr == 1:
        return r1 + 1, c0 if sc < 0 else c1
    if dc == -1:
        return r0 if sr < 0 else r1, c0 - 1
    return r0 if sr < 0 else r1, c1 + 1


def _trace_ray(
    shape: tuple[int, int],
    block_mask: np.ndarray,
    component: dict,
) -> tuple[list[tuple[int, int]], str, tuple[int, int]]:
    primary, secondary, _offset = _arrow_direction(component)
    start = _ray_start(component, primary, secondary)
    h, w = shape
    r, c = start
    path: list[tuple[int, int]] = []
    while 0 <= r < h and 0 <= c < w and not bool(block_mask[r, c]):
        path.append((r, c))
        nr = r + primary[0]
        nc = c + primary[1]
        if not (0 <= nr < h and 0 <= nc < w):
            return path, "boundary", start
        if bool(block_mask[nr, nc]):
            return path, "block", start
        r, c = nr, nc
    return path, "boundary", start


def _drop_blocks(blocks: list[dict], shape: tuple[int, int]) -> list[dict]:
    h, _w = shape
    occupied = np.zeros(shape, dtype=bool)
    order = sorted(blocks, key=lambda comp: (-int(comp["bbox"][2]), int(comp["bbox"][1])))
    settled: list[dict] = []
    for comp in order:
        coords = comp["coords"]
        shift = 0
        while True:
            moved = coords + np.array([shift + 1, 0], dtype=int)
            if int(moved[:, 0].max()) >= h:
                break
            if bool(occupied[moved[:, 0], moved[:, 1]].any()):
                break
            shift += 1
        final_coords = coords + np.array([shift, 0], dtype=int)
        occupied[final_coords[:, 0], final_coords[:, 1]] = True
        r0, c0 = final_coords.min(axis=0)
        r1, c1 = final_coords.max(axis=0)
        settled.append(
            {
                "color": int(comp["color"]),
                "coords": final_coords,
                "bbox": (int(r0), int(c0), int(r1), int(c1)),
                "area": int(final_coords.shape[0]),
            }
        )
    return settled


def main(grid: np.ndarray) -> np.ndarray:
    """
    Drop the large green blocks first, then let each compact blue arrow fire until the fallen blocks or the edge stop it.

    Input structure:
    - The grid is black background with several small blue arrowhead sprites and several larger green solid rectangles.
    - Each arrow sprite is a tiny 4-connected object with one-corner asymmetry, so its centroid is biased toward one cardinal direction.
    - Each large object is a solid rectangle that acts as a falling obstacle rather than as a final painted effect.

    Rule:
    - Split foreground objects by area: components of area at most five are arrows, while larger components are falling blocks.
    - Clear every falling block and drop it straight downward, processing blocks by original bottom row descending and then left column ascending, until the next step would hit the grid bottom or a previously settled block.
    - Keep the arrows fixed, infer each arrow's direction from the centroid offset inside its bounding box, and start a blue ray from the matching outer corner on that side of the box.
    - Extend the ray one cell at a time in that direction, stopping before the boundary or before the next blocked cell of the fallen rectangles.

    Key insight:
    - The arrows must read the obstacle layout after gravity, because the fallen rectangles redefine where every ray terminates.
    """
    g = np.asarray(grid, dtype=int)
    assert g.ndim == 2
    components = _find_components(g, BG)
    arrows = [comp for comp in components if int(comp["area"]) <= ARROW_MAX_AREA]
    blocks = [comp for comp in components if int(comp["area"]) > ARROW_MAX_AREA]
    assert arrows and blocks

    out = g.copy()
    for comp in blocks:
        out[comp["coords"][:, 0], comp["coords"][:, 1]] = BG

    settled = _drop_blocks(blocks, g.shape)
    block_mask = np.zeros_like(g, dtype=bool)
    for comp in settled:
        block_mask[comp["coords"][:, 0], comp["coords"][:, 1]] = True
        out[comp["coords"][:, 0], comp["coords"][:, 1]] = int(comp["color"])

    for comp in sorted(arrows, key=lambda item: tuple(item["bbox"])):
        path, _reason, _start = _trace_ray(g.shape, block_mask, comp)
        for r, c in path:
            out[r, c] = int(comp["color"])
    return out


def _make_arrow_component(mask: np.ndarray, top: int, left: int) -> dict:
    coords = np.argwhere(mask) + np.array([top, left], dtype=int)
    r0, c0 = coords.min(axis=0)
    r1, c1 = coords.max(axis=0)
    return {
        "color": ARROW,
        "coords": coords.astype(int),
        "bbox": (int(r0), int(c0), int(r1), int(c1)),
        "area": int(coords.shape[0]),
    }


def _arrow_templates() -> list[dict]:
    raw = [
        np.array([[1, 1, 1], [0, 1, 1]], dtype=bool),
        np.array([[0, 1, 1], [1, 1, 1]], dtype=bool),
        np.array([[1, 1, 1], [1, 1, 0]], dtype=bool),
        np.array([[1, 1, 0], [1, 1, 1]], dtype=bool),
        np.array([[1, 1], [1, 1], [1, 0]], dtype=bool),
        np.array([[1, 1], [1, 1], [0, 1]], dtype=bool),
        np.array([[1, 0], [1, 1], [1, 1]], dtype=bool),
        np.array([[0, 1], [1, 1], [1, 1]], dtype=bool),
    ]
    out = []
    for mask in raw:
        comp = _make_arrow_component(mask, 0, 0)
        primary, secondary, offset = _arrow_direction(comp)
        out.append(
            {
                "mask": mask,
                "primary": primary,
                "secondary": secondary,
                "offset": offset,
                "dir_name": {
                    (-1, 0): "up",
                    (1, 0): "down",
                    (0, -1): "left",
                    (0, 1): "right",
                }[primary],
            }
        )
    return out


ARROW_TEMPLATES = _arrow_templates()


def _neighbors8(cell: tuple[int, int]) -> Iterable[tuple[int, int]]:
    r, c = cell
    for dr in (-1, 0, 1):
        for dc in (-1, 0, 1):
            if dr == 0 and dc == 0:
                continue
            yield r + dr, c + dc


def _pad_cells(cells: Iterable[tuple[int, int]], shape: tuple[int, int]) -> set[tuple[int, int]]:
    h, w = shape
    out: set[tuple[int, int]] = set()
    for r, c in cells:
        if 0 <= r < h and 0 <= c < w:
            out.add((r, c))
        for nr, nc in _neighbors8((r, c)):
            if 0 <= nr < h and 0 <= nc < w:
                out.add((nr, nc))
    return out


def _sample_final_rects(h: int, w: int) -> list[tuple[int, int, int, int]] | None:
    scenario = random.choice(("separate", "separate", "stacked"))
    count = random.choice(BLOCK_COUNT_OPTIONS)

    if scenario == "separate":
        dims = [random.choice(BLOCK_SIZES) for _ in range(count)]
        total_width = sum(dw for _dh, dw in dims)
        gaps_needed = count + 1
        if total_width + gaps_needed > w:
            return None
        gap_budget = w - total_width
        gaps = [1] * (count + 1)
        extra = gap_budget - len(gaps)
        for _ in range(max(0, extra)):
            gaps[random.randrange(len(gaps))] += 1
        random.shuffle(dims)
        rects = []
        c = gaps[0]
        for i, (rh, rw) in enumerate(dims):
            rects.append((h - rh, c, rh, rw))
            c += rw + gaps[i + 1]
        return rects

    base_h, base_w = random.choice(BLOCK_SIZES)
    support_c = random.randint(1, w - base_w - 1)
    rects = [(h - base_h, support_c, base_h, base_w)]

    upper_h, upper_w = random.choice(BLOCK_SIZES)
    overlap_lo = max(1, support_c - upper_w + 1)
    overlap_hi = min(w - upper_w - 1, support_c + base_w - 1)
    if overlap_lo > overlap_hi:
        return None
    upper_c = random.randint(overlap_lo, overlap_hi)
    upper_r0 = h - base_h - upper_h
    if upper_r0 < 1:
        return None
    rects.append((upper_r0, upper_c, upper_h, upper_w))

    if count == 3:
        for _ in range(100):
            rh, rw = random.choice(BLOCK_SIZES)
            c0 = random.randint(1, w - rw - 1)
            r0 = h - rh
            cand = (r0, c0, rh, rw)
            if _rects_overlap(rects + [cand]):
                continue
            rects.append(cand)
            break
        if len(rects) != 3:
            return None
    return rects


def _rects_overlap(rects: list[tuple[int, int, int, int]]) -> bool:
    for i in range(len(rects)):
        r0a, c0a, ha, wa = rects[i]
        for j in range(i + 1, len(rects)):
            r0b, c0b, hb, wb = rects[j]
            if not (r0a + ha <= r0b or r0b + hb <= r0a or c0a + wa <= c0b or c0b + wb <= c0a):
                return True
    return False


def _lift_rects(
    final_rects: list[tuple[int, int, int, int]],
    h: int,
    want_equal_bottoms: bool,
) -> list[tuple[int, int, int, int]] | None:
    order = list(range(len(final_rects)))
    random.shuffle(order)
    for _ in range(300):
        rects = [None] * len(final_rects)
        equal_pair = None
        if want_equal_bottoms and len(final_rects) >= 2:
            candidates = [pair for pair in ((i, j) for i in range(len(final_rects)) for j in range(i + 1, len(final_rects))) if _horiz_disjoint(final_rects[pair[0]], final_rects[pair[1]])]
            if candidates:
                equal_pair = random.choice(candidates)
        target_bottom = None
        if equal_pair is not None:
            max_bottom = min(h // 2 + 1, h - 5)
            min_bottom = max(final_rects[equal_pair[0]][2] + 1, final_rects[equal_pair[1]][2] + 1)
            if min_bottom <= max_bottom:
                target_bottom = random.randint(min_bottom, max_bottom)
        ok = True
        for idx in order:
            fr0, fc0, fh, fw = final_rects[idx]
            final_bottom = fr0 + fh - 1
            min_bottom = fh + 1
            max_bottom = min(h // 2 + 2, final_bottom - 2)
            if equal_pair is not None and idx in equal_pair and target_bottom is not None:
                bottom = target_bottom
            else:
                if min_bottom > max_bottom:
                    ok = False
                    break
                bottom = random.randint(min_bottom, max_bottom)
            r0 = bottom - fh + 1
            rects[idx] = (r0, fc0, fh, fw)
        if not ok or _rects_overlap(rects):  # type: ignore[arg-type]
            continue
        assert all(rect is not None for rect in rects)
        return [rect for rect in rects if rect is not None]
    return None


def _horiz_disjoint(a: tuple[int, int, int, int], b: tuple[int, int, int, int]) -> bool:
    _r0a, c0a, _ha, wa = a
    _r0b, c0b, _hb, wb = b
    return c0a + wa <= c0b or c0b + wb <= c0a


def _component_from_rect(rect: tuple[int, int, int, int]) -> dict:
    r0, c0, h, w = rect
    rr, cc = np.indices((h, w))
    coords = np.column_stack([rr.ravel() + r0, cc.ravel() + c0]).astype(int)
    return {
        "color": BLOCK,
        "coords": coords,
        "bbox": (int(r0), int(c0), int(r0 + h - 1), int(c0 + w - 1)),
        "area": int(h * w),
    }


def _candidate_arrows(
    shape: tuple[int, int],
    input_block_mask: np.ndarray,
    final_block_mask: np.ndarray,
) -> dict[str, list[dict]]:
    h, w = shape
    input_block_cells = set(map(tuple, np.argwhere(input_block_mask)))
    input_block_pad = _pad_cells(input_block_cells, shape)
    by_dir: dict[str, list[dict]] = {"up": [], "down": [], "left": [], "right": []}

    for spec in ARROW_TEMPLATES:
        mask = spec["mask"]
        mh, mw = mask.shape
        for top in range(1, h - mh - 1):
            for left in range(1, w - mw - 1):
                comp = _make_arrow_component(mask, top, left)
                cells = {tuple(x) for x in comp["coords"].tolist()}
                if cells & input_block_pad:
                    continue
                final_path, final_reason, start = _trace_ray(shape, final_block_mask, comp)
                init_path, init_reason, _ = _trace_ray(shape, input_block_mask, comp)
                if len(final_path) < 2:
                    continue
                if start[0] < 0 or start[1] < 0 or start[0] >= h or start[1] >= w:
                    continue
                if set(final_path) & cells:
                    continue
                by_dir[spec["dir_name"]].append(
                    {
                        "component": comp,
                        "cells": cells,
                        "pad": _pad_cells(cells, shape),
                        "path": tuple(final_path),
                        "reason": final_reason,
                        "depends": tuple(final_path) != tuple(init_path) or final_reason != init_reason,
                        "dir_name": spec["dir_name"],
                        "output_cells": cells | set(final_path),
                        "vertical": spec["dir_name"] in {"up", "down"},
                    }
                )

    for dir_name, items in by_dir.items():
        random.shuffle(items)
        items.sort(
            key=lambda item: (
                item["reason"] != "block",
                not item["depends"],
                -len(item["path"]),
                tuple(item["component"]["bbox"]),
            )
        )
        by_dir[dir_name] = items[:80]
    return by_dir


def _choose_arrows(
    shape: tuple[int, int],
    input_block_mask: np.ndarray,
    final_block_mask: np.ndarray,
    arrow_count: int,
) -> list[dict] | None:
    candidates = _candidate_arrows(shape, input_block_mask, final_block_mask)
    dir_sets = [("up", "down", "left", "right")]
    if arrow_count == 3:
        dir_sets = [
            dirs
            for dirs in (
                ("up", "down", "left"),
                ("up", "down", "right"),
                ("up", "left", "right"),
                ("down", "left", "right"),
            )
        ]
        random.shuffle(dir_sets)

    input_pad = _pad_cells(set(map(tuple, np.argwhere(input_block_mask))), shape)
    final_blocks = set(map(tuple, np.argwhere(final_block_mask)))

    def search(
        dirs: tuple[str, ...],
        idx: int,
        chosen: list[dict],
        used_pad: set[tuple[int, int]],
        used_output: set[tuple[int, int]],
        seen_block: bool,
        seen_boundary: bool,
        seen_dep: bool,
        seen_vertical: bool,
        seen_horizontal: bool,
    ) -> list[dict] | None:
        if idx == len(dirs):
            if seen_block and seen_boundary and seen_dep and seen_vertical and seen_horizontal:
                return chosen.copy()
            return None

        dir_name = dirs[idx]
        items = candidates.get(dir_name, [])
        for cand in items:
            if cand["cells"] & used_pad:
                continue
            if cand["output_cells"] & used_output:
                continue
            if cand["output_cells"] & final_blocks:
                continue
            chosen.append(cand)
            found = search(
                dirs,
                idx + 1,
                chosen,
                used_pad | cand["pad"],
                used_output | cand["output_cells"],
                seen_block or cand["reason"] == "block",
                seen_boundary or cand["reason"] == "boundary",
                seen_dep or cand["depends"],
                seen_vertical or cand["vertical"],
                seen_horizontal or (not cand["vertical"]),
            )
            if found is not None:
                return found
            chosen.pop()
        return None

    for dirs in dir_sets:
        if any(not candidates[d] for d in dirs):
            continue
        result = search(
            dirs,
            0,
            [],
            input_pad.copy(),
            set(),
            False,
            False,
            False,
            False,
            False,
        )
        if result is not None:
            return result
    return None


def _build_input_grid(
    shape: tuple[int, int],
    input_rects: list[tuple[int, int, int, int]],
    arrows: list[dict],
) -> np.ndarray:
    grid = np.full(shape, BG, dtype=int)
    _paint_rects(grid, input_rects, BLOCK)
    for arrow in arrows:
        coords = arrow["component"]["coords"]
        grid[coords[:, 0], coords[:, 1]] = ARROW
    return grid


def _analyze_generated_case(grid: np.ndarray) -> dict:
    comps = _find_components(grid, BG)
    arrows = [comp for comp in comps if comp["area"] <= ARROW_MAX_AREA]
    blocks = [comp for comp in comps if comp["area"] > ARROW_MAX_AREA]
    assert len(arrows) in ARROW_COUNT_OPTIONS
    assert len(blocks) in BLOCK_COUNT_OPTIONS
    assert all(int(comp["color"]) == ARROW for comp in arrows)
    assert all(int(comp["color"]) == BLOCK for comp in blocks)

    dropped = _drop_blocks(blocks, grid.shape)
    final_block_mask = np.zeros_like(grid, dtype=bool)
    for comp in dropped:
        final_block_mask[comp["coords"][:, 0], comp["coords"][:, 1]] = True

    input_block_mask = np.zeros_like(grid, dtype=bool)
    for comp in blocks:
        input_block_mask[comp["coords"][:, 0], comp["coords"][:, 1]] = True

    reasons = []
    depends = 0
    total_ray = 0
    for comp in arrows:
        path, reason, start = _trace_ray(grid.shape, final_block_mask, comp)
        init_path, init_reason, _ = _trace_ray(grid.shape, input_block_mask, comp)
        assert len(path) >= 2
        assert 0 <= start[0] < grid.shape[0] and 0 <= start[1] < grid.shape[1]
        reasons.append(reason)
        total_ray += len(path)
        if tuple(path) != tuple(init_path) or reason != init_reason:
            depends += 1
    out = main(grid)

    dropped_only = grid.copy()
    for comp in blocks:
        dropped_only[comp["coords"][:, 0], comp["coords"][:, 1]] = BG
    for comp in dropped:
        dropped_only[comp["coords"][:, 0], comp["coords"][:, 1]] = BLOCK

    assert "block" in reasons
    assert "boundary" in reasons
    assert depends >= 1
    assert total_ray >= 8
    assert not np.array_equal(out, dropped_only)
    assert np.count_nonzero(out == ARROW) > np.count_nonzero(grid == ARROW)
    return {
        "arrow_count": len(arrows),
        "block_count": len(blocks),
        "depends": depends,
        "reasons": tuple(sorted(reasons)),
    }


def generate_input() -> np.ndarray:
    for _ in range(MAX_ATTEMPTS):
        h = random.randint(*GRID_RANGE)
        w = random.randint(*GRID_RANGE)
        final_rects = _sample_final_rects(h, w)
        if final_rects is None:
            continue
        want_equal_bottoms = random.random() < 0.35
        input_rects = _lift_rects(final_rects, h, want_equal_bottoms)
        if input_rects is None:
            continue
        input_block_mask = _rect_mask((h, w), input_rects)
        final_block_mask = _rect_mask((h, w), final_rects)

        arrow_count = 4 if random.random() < 0.65 else 3
        arrows = _choose_arrows((h, w), input_block_mask, final_block_mask, arrow_count)
        if arrows is None:
            continue

        grid = _build_input_grid((h, w), input_rects, arrows)
        try:
            _analyze_generated_case(grid)
        except AssertionError:
            continue
        return grid
    raise RuntimeError("failed to generate a valid gravity-ray puzzle")
