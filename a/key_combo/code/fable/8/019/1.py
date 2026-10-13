from __future__ import annotations

from collections import Counter
from typing import Any

import numpy as np


BLACK = 0
GREY = 5
BLOCK_COLORS = (1, 2, 3, 4, 6, 7, 8, 9)
DIR4 = ((1, 0), (-1, 0), (0, 1), (0, -1))


def _get_bg(grid: np.ndarray) -> int:
    return int(Counter(map(int, grid.ravel())).most_common(1)[0][0])


def _bbox(cells: list[tuple[int, int]]) -> tuple[int, int, int, int]:
    rows = [r for r, _ in cells]
    cols = [c for _, c in cells]
    return min(rows), min(cols), max(rows), max(cols)


def _find_color_components(grid: np.ndarray, color: int) -> list[dict[str, Any]]:
    h, w = grid.shape
    mask = grid == color
    seen = np.zeros_like(mask, dtype=bool)
    components: list[dict[str, Any]] = []
    for r in range(h):
        for c in range(w):
            if not mask[r, c] or seen[r, c]:
                continue
            stack = [(r, c)]
            seen[r, c] = True
            cells: list[tuple[int, int]] = []
            while stack:
                cr, cc = stack.pop()
                cells.append((cr, cc))
                for dr, dc in DIR4:
                    nr, nc = cr + dr, cc + dc
                    if 0 <= nr < h and 0 <= nc < w and mask[nr, nc] and not seen[nr, nc]:
                        seen[nr, nc] = True
                        stack.append((nr, nc))
            components.append(
                {
                    "color": color,
                    "cells": cells,
                    "bbox": _bbox(cells),
                    "area": len(cells),
                }
            )
    return components


def _build_slot_mask(direction: str, cavity_h: int, cavity_w: int) -> np.ndarray:
    if direction == "up":
        slot = np.zeros((cavity_h + 1, cavity_w + 2), dtype=bool)
        slot[:, 0] = True
        slot[:, -1] = True
        slot[-1, :] = True
        return slot
    if direction == "left":
        slot = np.zeros((cavity_h + 2, cavity_w + 1), dtype=bool)
        slot[0, :] = True
        slot[-1, :] = True
        slot[:, -1] = True
        return slot
    if direction == "right":
        slot = np.zeros((cavity_h + 2, cavity_w + 1), dtype=bool)
        slot[0, :] = True
        slot[-1, :] = True
        slot[:, 0] = True
        return slot
    raise ValueError(f"unknown direction: {direction}")


def _parse_slots(grid: np.ndarray) -> list[dict[str, Any]]:
    slots: list[dict[str, Any]] = []
    for comp in _find_color_components(grid, GREY):
        r0, c0, r1, c1 = comp["bbox"]
        mask = np.zeros((r1 - r0 + 1, c1 - c0 + 1), dtype=bool)
        for r, c in comp["cells"]:
            mask[r - r0, c - c0] = True
        parsed = None
        h, w = mask.shape
        if h >= 2 and w >= 3:
            expected = _build_slot_mask("up", h - 1, w - 2)
            if np.array_equal(mask, expected):
                parsed = {
                    "direction": "up",
                    "bbox": (r0, c0, r1, c1),
                    "cavity_r0": r0,
                    "cavity_c0": c0 + 1,
                    "cavity_h": h - 1,
                    "cavity_w": w - 2,
                }
        if parsed is None and h >= 3 and w >= 2:
            expected = _build_slot_mask("left", h - 2, w - 1)
            if np.array_equal(mask, expected):
                parsed = {
                    "direction": "left",
                    "bbox": (r0, c0, r1, c1),
                    "cavity_r0": r0 + 1,
                    "cavity_c0": c0,
                    "cavity_h": h - 2,
                    "cavity_w": w - 1,
                }
        if parsed is None and h >= 3 and w >= 2:
            expected = _build_slot_mask("right", h - 2, w - 1)
            if np.array_equal(mask, expected):
                parsed = {
                    "direction": "right",
                    "bbox": (r0, c0, r1, c1),
                    "cavity_r0": r0 + 1,
                    "cavity_c0": c0 + 1,
                    "cavity_h": h - 2,
                    "cavity_w": w - 1,
                }
        if parsed is not None:
            slots.append(parsed)
    return slots


def _parse_blocks(grid: np.ndarray, bg: int) -> list[dict[str, Any]]:
    blocks: list[dict[str, Any]] = []
    for color in sorted(int(v) for v in np.unique(grid) if int(v) not in {bg, GREY}):
        for comp in _find_color_components(grid, color):
            r0, c0, r1, c1 = comp["bbox"]
            bh = r1 - r0 + 1
            bw = c1 - c0 + 1
            if comp["area"] != bh * bw:
                continue
            if not np.all(grid[r0 : r1 + 1, c0 : c1 + 1] == color):
                continue
            blocks.append(
                {
                    "color": color,
                    "bbox": (r0, c0, r1, c1),
                    "h": bh,
                    "w": bw,
                }
            )
    return blocks


def _candidate_for_slot(block: dict[str, Any], slot: dict[str, Any]) -> tuple[int, int] | None:
    options = [(block["h"], block["w"])]
    if block["h"] != block["w"]:
        options.append((block["w"], block["h"]))
    for oh, ow in options:
        if slot["direction"] == "up":
            if ow == slot["cavity_w"] and oh <= slot["cavity_h"]:
                return oh, ow
        else:
            if oh == slot["cavity_h"] and ow <= slot["cavity_w"]:
                return oh, ow
    return None


def _unique_maximum_matching(
    candidates: dict[int, list[tuple[int, tuple[int, int]]]]
) -> dict[int, tuple[int, tuple[int, int]]]:
    order = sorted(candidates, key=lambda idx: (len(candidates[idx]), idx))
    best_size = -1
    best: list[dict[int, tuple[int, tuple[int, int]]]] = []

    def dfs(i: int, used_slots: set[int], current: dict[int, tuple[int, tuple[int, int]]]) -> None:
        nonlocal best_size, best
        remaining = len(order) - i
        if len(current) + remaining < best_size:
            return
        if i == len(order):
            size = len(current)
            snapshot = dict(current)
            if size > best_size:
                best_size = size
                best = [snapshot]
            elif size == best_size:
                best.append(snapshot)
            return

        block_idx = order[i]
        dfs(i + 1, used_slots, current)
        for slot_idx, oriented in candidates[block_idx]:
            if slot_idx in used_slots:
                continue
            used_slots.add(slot_idx)
            current[block_idx] = (slot_idx, oriented)
            dfs(i + 1, used_slots, current)
            del current[block_idx]
            used_slots.remove(slot_idx)

    dfs(0, set(), {})
    if not best:
        return {}
    return best[0]


def main(grid: np.ndarray) -> np.ndarray:
    """
    Rotate each solid rectangle if needed, insert it through the matching U-slot opening, and let it slide to the slot bottom.

    Input structure:
    - A black background contains several differently colored solid rectangular blocks.
    - Grey 1-pixel-thick U-shaped slots are also present; each slot opens upward, leftward, or rightward.
    - Some blocks have a unique size match with one slot opening after a possible 90-degree rotation, while other blocks may match no slot.

    Rule:
    - Detect every solid colored rectangle and every grey U-slot.
    - A block matches a slot when one orientation fits through the opening exactly in the dimension perpendicular to motion and also fits within the slot depth.
    - Move each uniquely matched block into that slot, clearing its old position; unmatched blocks stay in place and slots remain unchanged.

    Key insight:
    - Color and proximity are irrelevant; the only pairing signal is whether a rectangle can pass through a slot opening after an optional 90-degree rotation, and the slot orientation fixes the sliding direction.
    """
    g = np.asarray(grid, dtype=int)
    bg = _get_bg(g)
    slots = _parse_slots(g)
    blocks = _parse_blocks(g, bg)

    candidates: dict[int, list[tuple[int, tuple[int, int]]]] = {}
    for bi, block in enumerate(blocks):
        opts: list[tuple[int, tuple[int, int]]] = []
        for si, slot in enumerate(slots):
            oriented = _candidate_for_slot(block, slot)
            if oriented is not None:
                opts.append((si, oriented))
        candidates[bi] = opts

    matching = _unique_maximum_matching(candidates)
    out = g.copy()

    for bi, (si, (oh, ow)) in matching.items():
        block = blocks[bi]
        slot = slots[si]
        r0, c0, r1, c1 = block["bbox"]
        out[r0 : r1 + 1, c0 : c1 + 1] = bg

        cavity_r0 = slot["cavity_r0"]
        cavity_c0 = slot["cavity_c0"]
        cavity_h = slot["cavity_h"]
        cavity_w = slot["cavity_w"]

        if slot["direction"] == "up":
            dest_r = cavity_r0 + cavity_h - oh
            dest_c = cavity_c0
        elif slot["direction"] == "left":
            dest_r = cavity_r0
            dest_c = cavity_c0 + cavity_w - ow
        else:
            dest_r = cavity_r0
            dest_c = cavity_c0

        out[dest_r : dest_r + oh, dest_c : dest_c + ow] = block["color"]

    return out


def _place_shape(
    grid: np.ndarray,
    occupied: np.ndarray,
    shape: np.ndarray,
    rng: np.random.Generator,
    pad: int = 1,
) -> tuple[int, int] | None:
    gh, gw = grid.shape
    sh, sw = shape.shape
    if sh > gh or sw > gw:
        return None

    rows = np.arange(gh - sh + 1)
    cols = np.arange(gw - sw + 1)
    positions = [(int(r), int(c)) for r in rows for c in cols]
    rng.shuffle(positions)
    for r, c in positions:
        rr0 = max(0, r - pad)
        cc0 = max(0, c - pad)
        rr1 = min(gh, r + sh + pad)
        cc1 = min(gw, c + sw + pad)
        if occupied[rr0:rr1, cc0:cc1].any():
            continue
        grid[r : r + sh, c : c + sw] = np.where(shape != BLACK, shape, grid[r : r + sh, c : c + sw])
        occupied[rr0:rr1, cc0:cc1] = True
        return r, c
    return None


def _make_slot(direction: str, cavity_h: int, cavity_w: int) -> np.ndarray:
    slot = np.full(_build_slot_mask(direction, cavity_h, cavity_w).shape, BLACK, dtype=int)
    slot[_build_slot_mask(direction, cavity_h, cavity_w)] = GREY
    return slot


def _generate_case(rng: np.random.Generator) -> np.ndarray | None:
    gh = int(rng.integers(20, 31))
    gw = int(rng.integers(20, 31))
    grid = np.full((gh, gw), BLACK, dtype=int)
    occupied = np.zeros((gh, gw), dtype=bool)

    matched_count = int(rng.integers(2, 5))
    unmatched_count = int(rng.choice([0, 1, 2], p=[0.35, 0.45, 0.20]))
    extra_slot_count = int(rng.choice([0, 1], p=[0.7, 0.3]))
    total_blocks = matched_count + unmatched_count
    total_slots = matched_count + extra_slot_count
    if total_blocks > len(BLOCK_COLORS):
        return None

    slot_measures_pool = np.arange(2, 8)
    if total_slots > len(slot_measures_pool):
        return None
    slot_measures = list(map(int, rng.choice(slot_measures_pool, size=total_slots, replace=False)))

    non_measures = [d for d in range(1, 9) if d not in slot_measures]
    if not non_measures:
        return None

    colors = list(map(int, rng.choice(BLOCK_COLORS, size=total_blocks, replace=False)))

    directions = ["up", "left", "right"]
    slot_dirs: list[str] = []
    distinct = min(total_slots, len(directions))
    perm = list(rng.permutation(directions))
    slot_dirs.extend(perm[:distinct])
    while len(slot_dirs) < total_slots:
        slot_dirs.append(str(rng.choice(directions)))
    rng.shuffle(slot_dirs)

    slots_to_place: list[np.ndarray] = []
    blocks_to_place: list[np.ndarray] = []

    for i in range(matched_count):
        measure = slot_measures[i]
        direction = slot_dirs[i]
        other = int(rng.choice(non_measures))
        extra = int(rng.integers(0, 3))

        if direction == "up":
            cavity_h = other + extra
            cavity_w = measure
            placed_h, placed_w = other, measure
        else:
            cavity_h = measure
            cavity_w = other + extra
            placed_h, placed_w = measure, other

        slot = _make_slot(direction, cavity_h, cavity_w)
        block_h, block_w = placed_h, placed_w
        if block_h != block_w and rng.random() < 0.5:
            block_h, block_w = block_w, block_h
        block = np.full((block_h, block_w), colors[i], dtype=int)

        slots_to_place.append(slot)
        blocks_to_place.append(block)

    for i in range(extra_slot_count):
        measure = slot_measures[matched_count + i]
        direction = slot_dirs[matched_count + i]
        other = int(rng.choice(non_measures))
        extra = int(rng.integers(0, 3))
        if direction == "up":
            cavity_h = other + extra
            cavity_w = measure
        else:
            cavity_h = measure
            cavity_w = other + extra
        slots_to_place.append(_make_slot(direction, cavity_h, cavity_w))

    for i in range(unmatched_count):
        h = int(rng.choice(non_measures))
        w = int(rng.choice(non_measures))
        block = np.full((h, w), colors[matched_count + i], dtype=int)
        blocks_to_place.append(block)

    slots_to_place.sort(key=lambda arr: arr.shape[0] * arr.shape[1], reverse=True)
    blocks_to_place.sort(key=lambda arr: arr.shape[0] * arr.shape[1], reverse=True)

    for slot in slots_to_place:
        if _place_shape(grid, occupied, slot, rng, pad=1) is None:
            return None
    for block in blocks_to_place:
        if _place_shape(grid, occupied, block, rng, pad=1) is None:
            return None

    parsed_slots = _parse_slots(grid)
    parsed_blocks = _parse_blocks(grid, BLACK)
    if len(parsed_slots) != total_slots or len(parsed_blocks) != total_blocks:
        return None

    candidates = {
        bi: [(si, orient) for si, slot in enumerate(parsed_slots) if (orient := _candidate_for_slot(block, slot)) is not None]
        for bi, block in enumerate(parsed_blocks)
    }
    matching = _unique_maximum_matching(candidates)
    if len(matching) != matched_count:
        return None
    if not any(len(opts) == 0 for opts in candidates.values()) and unmatched_count > 0:
        return None

    try:
        out = main(grid)
    except Exception:
        return None
    if out.shape != grid.shape or np.array_equal(out, grid):
        return None
    return grid


def generate_input() -> np.ndarray:
    rng = np.random.default_rng()
    for _ in range(400):
        grid = _generate_case(rng)
        if grid is not None:
            return grid
    raise RuntimeError("failed to generate a valid rotating-into-slot ARC input")
