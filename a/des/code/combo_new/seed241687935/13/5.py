import random
from collections import defaultdict

import numpy as np


BG = 0
ARROW_COLOR = 5
COLOR_POOL = [1, 2, 3, 4, 6, 7, 8, 9]

DIR_ORDER = ("up", "down", "left", "right")
DIR_VECS = {
    "up": (-1, 0),
    "down": (1, 0),
    "left": (0, -1),
    "right": (0, 1),
}

ARROW_PATTERNS = {
    frozenset({(0, 0), (1, 1), (2, 0)}): "right",
    frozenset({(0, 1), (1, 0), (2, 1)}): "left",
    frozenset({(0, 0), (0, 2), (1, 1)}): "down",
    frozenset({(1, 0), (0, 1), (1, 2)}): "up",
}


def _components(mask: np.ndarray, connectivity: int) -> list[list[tuple[int, int]]]:
    h, w = mask.shape
    seen = np.zeros_like(mask, dtype=bool)
    if connectivity == 4:
        deltas = [(-1, 0), (1, 0), (0, -1), (0, 1)]
    else:
        deltas = [
            (-1, -1),
            (-1, 0),
            (-1, 1),
            (0, -1),
            (0, 1),
            (1, -1),
            (1, 0),
            (1, 1),
        ]

    out = []
    for r in range(h):
        for c in range(w):
            if not mask[r, c] or seen[r, c]:
                continue
            stack = [(r, c)]
            seen[r, c] = True
            coords = []
            while stack:
                cr, cc = stack.pop()
                coords.append((int(cr), int(cc)))
                for dr, dc in deltas:
                    nr, nc = cr + dr, cc + dc
                    if 0 <= nr < h and 0 <= nc < w and mask[nr, nc] and not seen[nr, nc]:
                        seen[nr, nc] = True
                        stack.append((nr, nc))
            out.append(coords)
    return out


def _normalize(coords: list[tuple[int, int]]) -> tuple[int, int, tuple[tuple[int, int], ...]]:
    rows = [r for r, _ in coords]
    cols = [c for _, c in coords]
    r0 = min(rows)
    c0 = min(cols)
    rel = tuple(sorted((r - r0, c - c0) for r, c in coords))
    return int(r0), int(c0), rel


def _make_block(color: int, coords: list[tuple[int, int]], direction: str | None = None) -> dict:
    r0, c0, rel = _normalize(coords)
    return {
        "color": int(color),
        "r0": int(r0),
        "c0": int(c0),
        "rel": rel,
        "direction": direction,
    }


def _block_cells(block: dict, delta: tuple[int, int] = (0, 0)) -> set[tuple[int, int]]:
    dr, dc = delta
    return {
        (int(block["r0"]) + rr + dr, int(block["c0"]) + cc + dc)
        for rr, cc in block["rel"]
    }


def _bbox(cells: set[tuple[int, int]]) -> tuple[int, int, int, int]:
    rows = [r for r, _ in cells]
    cols = [c for _, c in cells]
    return min(rows), min(cols), max(rows), max(cols)


def _touching4(a: set[tuple[int, int]], b: set[tuple[int, int]]) -> bool:
    for r, c in a:
        if (r - 1, c) in b or (r + 1, c) in b or (r, c - 1) in b or (r, c + 1) in b:
            return True
    return False


def _parse_scene(grid: np.ndarray) -> list[dict]:
    g = np.asarray(grid, dtype=int)
    h, w = g.shape

    blocks = []
    for color in sorted(int(v) for v in np.unique(g) if int(v) not in (BG, ARROW_COLOR)):
        for comp in _components(g == color, connectivity=4):
            blocks.append(_make_block(int(color), comp))
    blocks.sort(key=lambda b: (int(b["color"]), int(b["r0"]), int(b["c0"])))

    arrows = []
    for comp in _components(g == ARROW_COLOR, connectivity=8):
        if len(comp) != 3:
            raise ValueError("every gray component must be a 3-pixel arrow")
        r0, c0, rel = _normalize(comp)
        direction = ARROW_PATTERNS.get(frozenset(rel))
        if direction is None:
            raise ValueError("unrecognized arrow shape")
        arrows.append({"cells": tuple(comp), "direction": direction, "r0": r0, "c0": c0})

    if len(arrows) != len(blocks):
        raise ValueError("blocks and arrows must match one-to-one")

    used_blocks = set()
    for arrow in arrows:
        candidates = []
        for idx, block in enumerate(blocks):
            cells = _block_cells(block)
            if any(abs(ar - br) + abs(ac - bc) == 1 for ar, ac in arrow["cells"] for br, bc in cells):
                candidates.append(idx)
        if len(candidates) != 1:
            dists = []
            for idx, block in enumerate(blocks):
                cells = _block_cells(block)
                dist = min(abs(ar - br) + abs(ac - bc) for ar, ac in arrow["cells"] for br, bc in cells)
                dists.append((dist, idx))
            best = min(dist for dist, _idx in dists)
            candidates = [idx for dist, idx in dists if dist == best]
        if len(candidates) != 1 or candidates[0] in used_blocks:
            raise ValueError("arrow pairing is ambiguous")
        block = blocks[candidates[0]]
        block["direction"] = str(arrow["direction"])
        used_blocks.add(candidates[0])

    if any(block["direction"] is None for block in blocks):
        raise ValueError("every block needs one arrow")

    for block in blocks:
        for r, c in _block_cells(block):
            if not (0 <= r < h and 0 <= c < w):
                raise ValueError("block out of bounds")

    return blocks


def _simulate_scene(blocks: list[dict], shape: tuple[int, int]) -> list[dict]:
    h, w = shape
    state = [{k: (v if k != "rel" else tuple(v)) for k, v in block.items()} for block in blocks]

    limit = h * w * 2 + 10
    for _ in range(limit):
        current = [_block_cells(block) for block in state]
        proposals: dict[int, set[tuple[int, int]]] = {}
        movable = set()
        for idx, block in enumerate(state):
            dr, dc = DIR_VECS[str(block["direction"])]
            shifted = {(r + dr, c + dc) for r, c in current[idx]}
            if all(0 <= r < h and 0 <= c < w for r, c in shifted):
                proposals[idx] = shifted
                movable.add(idx)

        while True:
            blocked = set()
            stationary = set(range(len(state))) - movable
            for idx in movable:
                if any(proposals[idx] & current[j] for j in stationary):
                    blocked.add(idx)
            movable_list = sorted(movable)
            for i_pos, i in enumerate(movable_list):
                for j in movable_list[i_pos + 1 :]:
                    if proposals[i] & proposals[j]:
                        blocked.add(i)
                        blocked.add(j)
                    elif (proposals[i] & current[j]) and (proposals[j] & current[i]):
                        blocked.add(i)
                        blocked.add(j)
            new_movable = movable - blocked
            if new_movable == movable:
                break
            movable = new_movable

        if not movable:
            return state

        for idx in movable:
            dr, dc = DIR_VECS[str(state[idx]["direction"])]
            state[idx]["r0"] = int(state[idx]["r0"]) + dr
            state[idx]["c0"] = int(state[idx]["c0"]) + dc

    raise RuntimeError("simulation did not converge")


def _same_color_groups(blocks: list[dict]) -> dict[int, list[list[int]]]:
    by_color: dict[int, list[int]] = defaultdict(list)
    cell_sets = [_block_cells(block) for block in blocks]
    for idx, block in enumerate(blocks):
        by_color[int(block["color"])].append(idx)

    groups: dict[int, list[list[int]]] = defaultdict(list)
    for color, indices in by_color.items():
        seen = set()
        for start in indices:
            if start in seen:
                continue
            stack = [start]
            seen.add(start)
            comp = []
            while stack:
                cur = stack.pop()
                comp.append(cur)
                for nxt in indices:
                    if nxt in seen:
                        continue
                    if _touching4(cell_sets[cur], cell_sets[nxt]):
                        seen.add(nxt)
                        stack.append(nxt)
            groups[int(color)].append(comp)
    return groups


def _render_output(blocks: list[dict], shape: tuple[int, int]) -> np.ndarray:
    out = np.zeros(shape, dtype=int)
    cell_sets = [_block_cells(block) for block in blocks]
    for idx, block in enumerate(blocks):
        for r, c in cell_sets[idx]:
            out[r, c] = int(block["color"])

    groups = _same_color_groups(blocks)
    for color, comps in groups.items():
        for comp in comps:
            if len(comp) <= 1:
                continue
            union = set()
            for idx in comp:
                union |= cell_sets[idx]
            r0, c0, r1, c1 = _bbox(union)
            out[r0 : r1 + 1, c0 : c1 + 1] = int(color)
    return out


def _stop_reason(blocks: list[dict], idx: int, shape: tuple[int, int]) -> str:
    h, w = shape
    cells = _block_cells(blocks[idx])
    dr, dc = DIR_VECS[str(blocks[idx]["direction"])]
    shifted = {(r + dr, c + dc) for r, c in cells}
    if any(r < 0 or r >= h or c < 0 or c >= w for r, c in shifted):
        return "boundary"
    others = set()
    for j, other in enumerate(blocks):
        if j != idx:
            others |= _block_cells(other)
    if shifted & others:
        return "block"
    return "free"


def main(grid: np.ndarray) -> np.ndarray:
    """
    Slide each colored block along its gray-arrow direction, then merge touching same-color pairs into solid rectangles.

    Input structure:
    - Black background with several small monochrome rectangular blocks.
    - Each block has one nearby three-pixel gray arrow marker encoded as a cardinal arrowhead shape.
    - Each foreground color appears in exactly two separate blocks.

    Rule:
    - Read the direction of each arrow and pair it with the unique adjacent block.
    - Erase the arrows, then repeatedly move every block one cell at a time in its own direction until another step would leave the grid or collide with another block.
    - After motion stops, any same-color blocks that end up 4-neighbor touching merge by filling their shared bounding box with that color.

    Key insight:
    - The gray three-pixel motifs are directional controls rather than output content, and the second layer is that post-slide same-color contact triggers a rectangular fill.
    """
    g = np.asarray(grid, dtype=int)
    blocks = _parse_scene(g)
    final_blocks = _simulate_scene(blocks, g.shape)
    return _render_output(final_blocks, g.shape)


def _arrow_cells(r0: int, c0: int, h: int, w: int, direction: str) -> list[tuple[int, int]]:
    if direction == "right":
        row = random.randint(r0, r0 + h - 1)
        top = row - 1
        left = c0 - 2
        rel = [(0, 0), (1, 1), (2, 0)]
    elif direction == "left":
        row = random.randint(r0, r0 + h - 1)
        top = row - 1
        left = c0 + w
        rel = [(0, 1), (1, 0), (2, 1)]
    elif direction == "down":
        col = random.randint(c0, c0 + w - 1)
        top = r0 - 2
        left = col - 1
        rel = [(0, 0), (0, 2), (1, 1)]
    else:
        col = random.randint(c0, c0 + w - 1)
        top = r0 + h
        left = col - 1
        rel = [(1, 0), (0, 1), (1, 2)]
    return [(top + dr, left + dc) for dr, dc in rel]


def _rect_cells(r0: int, c0: int, h: int, w: int) -> list[tuple[int, int]]:
    return [(r, c) for r in range(r0, r0 + h) for c in range(c0, c0 + w)]


def _random_free_block(grid_h: int, grid_w: int, color: int, direction: str) -> dict | None:
    for _ in range(60):
        block_h = random.randint(1, min(4, grid_h - 3))
        block_w = random.randint(1, min(4, grid_w - 3))
        if direction in ("left", "right"):
            if grid_h - block_h - 2 < 1:
                continue
            r0 = random.randint(1, grid_h - block_h - 2)
            final_c = 0 if direction == "left" else grid_w - block_w
            max_shift = (grid_w - block_w - 2) if direction == "left" else (final_c - 2)
            if max_shift < 1:
                continue
            shift = random.randint(1, max_shift)
            c0 = shift if direction == "left" else final_c - shift
        else:
            if grid_w - block_w - 2 < 1:
                continue
            c0 = random.randint(1, grid_w - block_w - 2)
            final_r = 0 if direction == "up" else grid_h - block_h
            max_shift = (grid_h - block_h - 2) if direction == "up" else (final_r - 2)
            if max_shift < 1:
                continue
            shift = random.randint(1, max_shift)
            r0 = shift if direction == "up" else final_r - shift

        arrow = _arrow_cells(int(r0), int(c0), int(block_h), int(block_w), direction)
        cells = _rect_cells(int(r0), int(c0), int(block_h), int(block_w))
        all_cells = cells + arrow
        if any(not (0 <= r < grid_h and 0 <= c < grid_w) for r, c in all_cells):
            continue
        return {
            "color": int(color),
            "direction": direction,
            "r0": int(r0),
            "c0": int(c0),
            "h": int(block_h),
            "w": int(block_w),
            "cells": tuple(cells),
            "arrow": tuple(arrow),
        }
    return None


def _merge_pair(grid_h: int, grid_w: int, color: int, direction: str) -> list[dict] | None:
    for _ in range(80):
        if direction in ("right", "left"):
            block_h = random.randint(1, min(4, grid_h - 3))
            if grid_h - block_h - 2 < 1:
                continue
            r0 = random.randint(1, grid_h - block_h - 2)
            trail_w = random.randint(1, min(4, grid_w - 5))
            lead_w = random.randint(1, min(4, grid_w - trail_w - 1))
            if direction == "right":
                lead_final_c = grid_w - lead_w
                trail_final_c = lead_final_c - trail_w
                max_lead = lead_final_c - 2
                max_trail = trail_final_c - 2
                if trail_final_c < 0 or max_lead < 1 or max_trail < 3:
                    continue
                lead_shift = random.randint(1, max_lead)
                min_trail = lead_shift + 2
                if min_trail > max_trail:
                    continue
                trail_shift = random.randint(min_trail, max_trail)
                lead_c = lead_final_c - lead_shift
                trail_c = trail_final_c - trail_shift
            else:
                lead_final_c = 0
                trail_final_c = lead_w
                max_lead = grid_w - lead_w - 2
                max_trail = grid_w - trail_w - 2 - trail_final_c
                if max_lead < 1 or max_trail < 3:
                    continue
                lead_shift = random.randint(1, max_lead)
                min_trail = lead_shift + 2
                if min_trail > max_trail:
                    continue
                trail_shift = random.randint(min_trail, max_trail)
                lead_c = lead_final_c + lead_shift
                trail_c = trail_final_c + trail_shift

            trail = {
                "color": int(color),
                "direction": direction,
                "r0": int(r0),
                "c0": int(trail_c),
                "h": int(block_h),
                "w": int(trail_w),
            }
            lead = {
                "color": int(color),
                "direction": direction,
                "r0": int(r0),
                "c0": int(lead_c),
                "h": int(block_h),
                "w": int(lead_w),
            }
        else:
            block_w = random.randint(1, min(4, grid_w - 3))
            if grid_w - block_w - 2 < 1:
                continue
            c0 = random.randint(1, grid_w - block_w - 2)
            trail_h = random.randint(1, min(4, grid_h - 5))
            lead_h = random.randint(1, min(4, grid_h - trail_h - 1))
            if direction == "down":
                lead_final_r = grid_h - lead_h
                trail_final_r = lead_final_r - trail_h
                max_lead = lead_final_r - 2
                max_trail = trail_final_r - 2
                if trail_final_r < 0 or max_lead < 1 or max_trail < 3:
                    continue
                lead_shift = random.randint(1, max_lead)
                min_trail = lead_shift + 2
                if min_trail > max_trail:
                    continue
                trail_shift = random.randint(min_trail, max_trail)
                lead_r = lead_final_r - lead_shift
                trail_r = trail_final_r - trail_shift
            else:
                lead_final_r = 0
                trail_final_r = lead_h
                max_lead = grid_h - lead_h - 2
                max_trail = grid_h - trail_h - 2 - trail_final_r
                if max_lead < 1 or max_trail < 3:
                    continue
                lead_shift = random.randint(1, max_lead)
                min_trail = lead_shift + 2
                if min_trail > max_trail:
                    continue
                trail_shift = random.randint(min_trail, max_trail)
                lead_r = lead_final_r + lead_shift
                trail_r = trail_final_r + trail_shift

            trail = {
                "color": int(color),
                "direction": direction,
                "r0": int(trail_r),
                "c0": int(c0),
                "h": int(trail_h),
                "w": int(block_w),
            }
            lead = {
                "color": int(color),
                "direction": direction,
                "r0": int(lead_r),
                "c0": int(c0),
                "h": int(lead_h),
                "w": int(block_w),
            }

        for block in (trail, lead):
            block["cells"] = tuple(_rect_cells(int(block["r0"]), int(block["c0"]), int(block["h"]), int(block["w"])))
            block["arrow"] = tuple(_arrow_cells(int(block["r0"]), int(block["c0"]), int(block["h"]), int(block["w"]), direction))

        all_cells = []
        for block in (trail, lead):
            all_cells.extend(block["cells"])
            all_cells.extend(block["arrow"])
        if any(not (0 <= r < grid_h and 0 <= c < grid_w) for r, c in all_cells):
            continue
        if len(set(all_cells)) != len(all_cells):
            continue
        return [trail, lead]
    return None


def _render_input(specs: list[dict], shape: tuple[int, int]) -> np.ndarray | None:
    grid = np.zeros(shape, dtype=int)
    for spec in specs:
        for r, c in spec["cells"]:
            if grid[r, c] != BG:
                return None
            grid[r, c] = int(spec["color"])
        for r, c in spec["arrow"]:
            if grid[r, c] != BG:
                return None
            grid[r, c] = ARROW_COLOR
    return grid


def _merged_colors(final_blocks: list[dict]) -> set[int]:
    merged = set()
    groups = _same_color_groups(final_blocks)
    for color, comps in groups.items():
        if any(len(comp) > 1 for comp in comps):
            merged.add(int(color))
    return merged


def _merge_bboxes_clear(final_blocks: list[dict]) -> bool:
    cell_sets = [_block_cells(block) for block in final_blocks]
    groups = _same_color_groups(final_blocks)
    for color, comps in groups.items():
        for comp in comps:
            if len(comp) <= 1:
                continue
            union = set()
            others = set()
            for idx in range(len(final_blocks)):
                if idx in comp:
                    union |= cell_sets[idx]
                else:
                    others |= cell_sets[idx]
            r0, c0, r1, c1 = _bbox(union)
            bbox_cells = {(r, c) for r in range(r0, r1 + 1) for c in range(c0, c1 + 1)}
            if bbox_cells & others:
                return False
    return True


def _gen_attempt() -> np.ndarray | None:
    pair_count = random.randint(2, 4)
    size_floor = 15 + 3 * (pair_count - 2)
    grid_h = random.randint(size_floor, 30)
    grid_w = random.randint(size_floor, 30)

    colors = random.sample(COLOR_POOL, pair_count)
    directions = list(DIR_ORDER)
    random.shuffle(directions)

    specs: list[dict] = []
    merge_color = colors[0]
    merge_pair = _merge_pair(grid_h, grid_w, merge_color, directions[0])
    if merge_pair is None:
        return None
    specs.extend(merge_pair)

    nonmerge_color = colors[1]
    free_dirs = [directions[1], directions[2]]
    for direction in free_dirs:
        block = _random_free_block(grid_h, grid_w, nonmerge_color, direction)
        if block is None:
            return None
        specs.append(block)

    extra_dirs = directions[3:] + directions[:]
    for color in colors[2:]:
        for _ in range(2):
            direction = random.choice(extra_dirs)
            block = _random_free_block(grid_h, grid_w, color, direction)
            if block is None:
                return None
            specs.append(block)

    grid = _render_input(specs, (grid_h, grid_w))
    if grid is None:
        return None
    if np.count_nonzero(grid) > grid_h * grid_w * 0.28:
        return None

    try:
        parsed = _parse_scene(grid)
    except Exception:
        return None

    if len(parsed) != len(specs):
        return None

    parsed_signature = sorted((int(block["color"]), str(block["direction"])) for block in parsed)
    spec_signature = sorted((int(spec["color"]), str(spec["direction"])) for spec in specs)
    if parsed_signature != spec_signature:
        return None

    final_blocks = _simulate_scene(parsed, grid.shape)
    if any(_stop_reason(final_blocks, idx, grid.shape) == "free" for idx in range(len(final_blocks))):
        return None
    stop_reasons = {_stop_reason(final_blocks, idx, grid.shape) for idx in range(len(final_blocks))}
    if not {"boundary", "block"} <= stop_reasons:
        return None

    merged_colors = _merged_colors(final_blocks)
    if merge_color not in merged_colors:
        return None
    if nonmerge_color in merged_colors:
        return None
    if len(merged_colors) >= pair_count:
        return None
    if not _merge_bboxes_clear(final_blocks):
        return None

    if len({block["direction"] for block in parsed}) < 3:
        return None

    out = _render_output(final_blocks, grid.shape)
    if np.array_equal(out, grid):
        return None
    return grid


def generate_input() -> np.ndarray:
    for _ in range(400):
        grid = _gen_attempt()
        if grid is not None:
            return grid
    raise RuntimeError("failed to generate a valid arrow-slide puzzle")
