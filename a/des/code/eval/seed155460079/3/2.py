from __future__ import annotations

import itertools
from functools import lru_cache

import numpy as np


BG = 0
FRAME_COLOR = 5
FRAGMENT_COLORS = (1, 2, 3, 4, 6, 7, 8, 9)
MAX_GENERATION_ATTEMPTS = 400
PADDING = 1


def _most_common_color(grid: np.ndarray) -> int:
    values = np.asarray(grid, dtype=int).ravel()
    counts = np.bincount(values, minlength=10)
    return int(np.argmax(counts))


def _find_components(grid: np.ndarray, bg: int) -> list[dict]:
    g = np.asarray(grid, dtype=int)
    h, w = g.shape
    seen = np.zeros((h, w), dtype=bool)
    comps: list[dict] = []

    for r in range(h):
        for c in range(w):
            color = int(g[r, c])
            if color == bg or seen[r, c]:
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
                    if 0 <= nr < h and 0 <= nc < w and not seen[nr, nc] and int(g[nr, nc]) == color:
                        seen[nr, nc] = True
                        stack.append((nr, nc))

            rows = [rr for rr, _ in coords]
            cols = [cc for _, cc in coords]
            comps.append(
                {
                    "color": color,
                    "coords": tuple(coords),
                    "bbox": (min(rows), min(cols), max(rows), max(cols)),
                    "area": len(coords),
                }
            )
    return comps


def _component_mask(comp: dict) -> np.ndarray:
    r0, c0, r1, c1 = comp["bbox"]
    mask = np.zeros((r1 - r0 + 1, c1 - c0 + 1), dtype=bool)
    for r, c in comp["coords"]:
        mask[int(r) - r0, int(c) - c0] = True
    return mask


def _is_frame(grid: np.ndarray, comp: dict, bg: int) -> bool:
    r0, c0, r1, c1 = comp["bbox"]
    h = r1 - r0 + 1
    w = c1 - c0 + 1
    if h < 4 or w < 4:
        return False
    if int(comp["area"]) != 2 * h + 2 * w - 4:
        return False

    color = int(comp["color"])
    sub = np.asarray(grid[r0 : r1 + 1, c0 : c1 + 1], dtype=int)
    if not np.all(sub[0, :] == color):
        return False
    if not np.all(sub[-1, :] == color):
        return False
    if not np.all(sub[:, 0] == color):
        return False
    if not np.all(sub[:, -1] == color):
        return False
    return bool(np.all(sub[1:-1, 1:-1] == bg))


def _extract_objects(grid: np.ndarray) -> tuple[int, list[dict], list[dict]]:
    g = np.asarray(grid, dtype=int)
    bg = _most_common_color(g)
    comps = _find_components(g, bg)

    slots: list[dict] = []
    fragments: list[dict] = []
    for comp in comps:
        r0, c0, r1, c1 = comp["bbox"]
        if _is_frame(g, comp, bg):
            slots.append(
                {
                    "color": int(comp["color"]),
                    "bbox": (r0, c0, r1, c1),
                    "inner_h": r1 - r0 - 1,
                    "inner_w": c1 - c0 - 1,
                }
            )
            continue

        fragments.append(
            {
                "color": int(comp["color"]),
                "coords": tuple((int(r), int(c)) for r, c in comp["coords"]),
                "bbox": (r0, c0, r1, c1),
                "mask": _component_mask(comp),
                "area": int(comp["area"]),
            }
        )

    slots.sort(key=lambda slot: (slot["bbox"][0], slot["bbox"][1], slot["inner_w"]))
    fragments.sort(
        key=lambda frag: (
            frag["bbox"][1] + frag["bbox"][3],
            frag["bbox"][1],
            frag["bbox"][0],
            frag["color"],
        )
    )
    return bg, slots, fragments


def _try_pair_in_slot(mask_left: np.ndarray, mask_right: np.ndarray, inner_h: int, inner_w: int) -> tuple[np.ndarray, np.ndarray] | None:
    if int(mask_left.sum()) + int(mask_right.sum()) != inner_h * inner_w:
        return None

    h_left, w_left = mask_left.shape
    h_right, w_right = mask_right.shape
    if h_left > inner_h or w_left > inner_w or h_right > inner_h or w_right > inner_w:
        return None

    left_coords = np.argwhere(mask_left)
    right_coords = np.argwhere(mask_right)
    for top_left in range(inner_h - h_left + 1):
        for col_left in range(inner_w - w_left + 1):
            if left_coords.size == 0:
                continue
            placed_left = np.zeros((inner_h, inner_w), dtype=bool)
            placed_left[left_coords[:, 0] + top_left, left_coords[:, 1] + col_left] = True

            for top_right in range(inner_h - h_right + 1):
                for col_right in range(col_left, inner_w - w_right + 1):
                    placed_right = np.zeros((inner_h, inner_w), dtype=bool)
                    placed_right[right_coords[:, 0] + top_right, right_coords[:, 1] + col_right] = True
                    if np.any(placed_left & placed_right):
                        continue

                    union = placed_left | placed_right
                    if not np.all(union):
                        continue

                    valid = True
                    for row in range(inner_h):
                        left_cols = np.flatnonzero(placed_left[row])
                        right_cols = np.flatnonzero(placed_right[row])
                        if left_cols.size and right_cols.size and int(left_cols[-1]) >= int(right_cols[0]):
                            valid = False
                            break
                    if valid:
                        return placed_left, placed_right
    return None


def _find_assignments(slots: list[dict], fragments: list[dict], limit: int = 1) -> list[list[tuple[int, int, np.ndarray, np.ndarray]]]:
    if len(fragments) != 2 * len(slots):
        return []

    candidates: list[list[tuple[int, int, np.ndarray, np.ndarray]]] = [[] for _ in slots]
    for slot_idx, slot in enumerate(slots):
        target_area = int(slot["inner_h"] * slot["inner_w"])
        for left_idx in range(len(fragments)):
            for right_idx in range(left_idx + 1, len(fragments)):
                if int(fragments[left_idx]["area"]) + int(fragments[right_idx]["area"]) != target_area:
                    continue
                placed = _try_pair_in_slot(
                    fragments[left_idx]["mask"],
                    fragments[right_idx]["mask"],
                    int(slot["inner_h"]),
                    int(slot["inner_w"]),
                )
                if placed is None:
                    continue
                candidates[slot_idx].append((left_idx, right_idx, placed[0], placed[1]))

    if any(not cand_list for cand_list in candidates):
        return []

    order = sorted(
        range(len(slots)),
        key=lambda idx: (len(candidates[idx]), slots[idx]["inner_w"], slots[idx]["bbox"][0]),
    )
    used = np.zeros(len(fragments), dtype=bool)
    assignment: list[tuple[int, int, np.ndarray, np.ndarray] | None] = [None] * len(slots)
    solutions: list[list[tuple[int, int, np.ndarray, np.ndarray]]] = []

    def dfs(depth: int) -> None:
        if len(solutions) >= limit:
            return
        if depth == len(order):
            if int(used.sum()) == len(fragments):
                solutions.append([item for item in assignment if item is not None])
            return

        slot_idx = order[depth]
        for cand in candidates[slot_idx]:
            left_idx, right_idx, placed_left, placed_right = cand
            if used[left_idx] or used[right_idx]:
                continue
            used[left_idx] = True
            used[right_idx] = True
            assignment[slot_idx] = (left_idx, right_idx, placed_left, placed_right)
            dfs(depth + 1)
            assignment[slot_idx] = None
            used[left_idx] = False
            used[right_idx] = False
            if len(solutions) >= limit:
                return

    dfs(0)
    return solutions


def _render_solution(grid: np.ndarray, bg: int, slots: list[dict], fragments: list[dict], assignment: list[tuple[int, int, np.ndarray, np.ndarray]]) -> np.ndarray:
    out = np.asarray(grid, dtype=int).copy()

    for frag in fragments:
        for r, c in frag["coords"]:
            out[int(r), int(c)] = bg

    for slot, match in zip(slots, assignment):
        left_idx, right_idx, placed_left, placed_right = match
        patch = np.full((int(slot["inner_h"]), int(slot["inner_w"])), bg, dtype=int)
        patch[placed_left] = int(fragments[left_idx]["color"])
        patch[placed_right] = int(fragments[right_idx]["color"])
        r0, c0, r1, c1 = slot["bbox"]
        out[r0 + 1 : r1, c0 + 1 : c1] = patch

    return out


def main(grid: np.ndarray) -> np.ndarray:
    g = np.asarray(grid, dtype=int)
    bg, slots, fragments = _extract_objects(g)
    assert slots, "expected at least one slot"
    assert fragments, "expected fragments"
    solutions = _find_assignments(slots, fragments, limit=1)
    assert solutions, "expected at least one valid pairing"
    return _render_solution(g, bg, slots, fragments, solutions[0])


@lru_cache(maxsize=None)
def _profile_options(inner_h: int, inner_w: int) -> tuple[tuple[int, ...], ...]:
    options: list[tuple[int, ...]] = []
    for split in itertools.product(range(1, inner_w), repeat=inner_h):
        left_area = sum(split)
        right_area = inner_h * inner_w - left_area
        if 3 <= left_area <= 6 and 3 <= right_area <= 6:
            options.append(tuple(int(v) for v in split))
    return tuple(options)


def _pair_patches(inner_h: int, inner_w: int, split: tuple[int, ...], left_color: int, right_color: int, bg: int) -> tuple[np.ndarray, np.ndarray, np.ndarray]:
    left_width = max(split)
    right_lengths = [inner_w - int(v) for v in split]
    right_width = max(right_lengths)

    left_patch = np.full((inner_h, left_width), bg, dtype=int)
    right_patch = np.full((inner_h, right_width), bg, dtype=int)
    fill_patch = np.full((inner_h, inner_w), bg, dtype=int)

    for row, left_len in enumerate(split):
        right_len = inner_w - int(left_len)
        left_patch[row, : int(left_len)] = left_color
        right_patch[row, right_width - right_len :] = right_color
        fill_patch[row, : int(left_len)] = left_color
        fill_patch[row, int(left_len) :] = right_color

    return left_patch, right_patch, fill_patch


def _draw_frame(grid: np.ndarray, top: int, left: int, inner_h: int, inner_w: int, color: int) -> None:
    grid[top, left : left + inner_w + 2] = color
    grid[top + inner_h + 1, left : left + inner_w + 2] = color
    grid[top : top + inner_h + 2, left] = color
    grid[top : top + inner_h + 2, left + inner_w + 1] = color


def _can_place(occupied: np.ndarray, mask: np.ndarray, top: int, left: int, padding: int = PADDING) -> bool:
    h, w = occupied.shape
    patch_h, patch_w = mask.shape
    if top < 0 or left < 0 or top + patch_h > h or left + patch_w > w:
        return False

    rows, cols = np.where(mask)
    for rr, cc in zip(rows, cols):
        tr = int(top + rr)
        tc = int(left + cc)
        if occupied[tr, tc]:
            return False
        r0 = max(0, tr - padding)
        r1 = min(h, tr + padding + 1)
        c0 = max(0, tc - padding)
        c1 = min(w, tc + padding + 1)
        if np.any(occupied[r0:r1, c0:c1]):
            return False
    return True


def _occupy(occupied: np.ndarray, mask: np.ndarray, top: int, left: int) -> None:
    patch_h, patch_w = mask.shape
    occupied[top : top + patch_h, left : left + patch_w] |= mask


def _stamp(grid: np.ndarray, occupied: np.ndarray, patch: np.ndarray, top: int, left: int, bg: int) -> None:
    mask = patch != bg
    grid[top : top + patch.shape[0], left : left + patch.shape[1]][mask] = patch[mask]
    _occupy(occupied, mask, top, left)


def _place_pair(
    grid: np.ndarray,
    occupied: np.ndarray,
    left_patch: np.ndarray,
    right_patch: np.ndarray,
    bg: int,
    rng: np.random.Generator,
    row_lo: int,
    row_hi: int,
    col_lo: int,
    col_hi: int,
) -> tuple[tuple[int, int], tuple[int, int]] | None:
    left_mask = left_patch != bg
    right_mask = right_patch != bg
    left_h, left_w = left_patch.shape
    right_h, right_w = right_patch.shape

    max_left_col = col_hi - (left_w + 1 + right_w)
    if row_hi - row_lo < max(left_h, right_h) or max_left_col < col_lo:
        return None

    for _ in range(250):
        top_left = int(rng.integers(row_lo, row_hi - left_h + 1))
        col_left = int(rng.integers(col_lo, max_left_col + 1))
        if not _can_place(occupied, left_mask, top_left, col_left):
            continue

        right_col_lo = col_left + left_w + 1
        if right_col_lo > col_hi - right_w:
            continue

        top_right = int(rng.integers(row_lo, row_hi - right_h + 1))
        col_right = int(rng.integers(right_col_lo, col_hi - right_w + 1))

        temp_occupied = occupied.copy()
        _occupy(temp_occupied, left_mask, top_left, col_left)
        if not _can_place(temp_occupied, right_mask, top_right, col_right):
            continue

        _stamp(grid, occupied, left_patch, top_left, col_left, bg)
        _stamp(grid, occupied, right_patch, top_right, col_right, bg)
        return (top_left, col_left), (top_right, col_right)

    return None


def _expected_output(
    grid: np.ndarray,
    bg: int,
    slot_records: list[dict],
    pair_records: list[dict],
    placements: list[dict],
) -> np.ndarray:
    out = np.asarray(grid, dtype=int).copy()

    for placed in placements:
        patch = placed["patch"]
        top, left = placed["pos"]
        mask = patch != bg
        window = out[top : top + patch.shape[0], left : left + patch.shape[1]]
        window[mask] = bg

    for pair_idx, pair in enumerate(pair_records):
        slot = slot_records[pair_idx]
        top, left = slot["top"], slot["left"]
        out[top + 1 : top + 1 + pair["fill"].shape[0], left + 1 : left + 1 + pair["fill"].shape[1]] = pair["fill"]

    return out


def _gen_attempt(rng: np.random.Generator) -> np.ndarray | None:
    inner_h = 2 if bool(rng.integers(0, 2)) else 3
    width_pool = [3, 4, 5, 6] if inner_h == 2 else [3, 4]
    pair_count = int(rng.integers(2, len(width_pool) + 1))
    widths = [int(v) for v in rng.choice(width_pool, size=pair_count, replace=False)]

    colors = [int(v) for v in rng.choice(FRAGMENT_COLORS, size=2 * pair_count, replace=False)]
    rng.shuffle(widths)

    pair_records: list[dict] = []
    for pair_idx, inner_w in enumerate(widths):
        options = list(_profile_options(inner_h, inner_w))
        irregular = [split for split in options if len(set(split)) > 1]
        use_irregular = irregular and bool(rng.integers(0, 4))
        split = irregular[int(rng.integers(0, len(irregular)))] if use_irregular else options[int(rng.integers(0, len(options)))]
        left_patch, right_patch, fill_patch = _pair_patches(
            inner_h,
            inner_w,
            split,
            colors[2 * pair_idx],
            colors[2 * pair_idx + 1],
            BG,
        )
        pair_records.append(
            {
                "width": inner_w,
                "left_patch": left_patch,
                "right_patch": right_patch,
                "fill": fill_patch,
            }
        )

    vertical_gap = int(rng.integers(1, 3))
    top_margin = int(rng.integers(1, 4))
    bottom_margin = int(rng.integers(1, 4))
    slot_stack_h = pair_count * (inner_h + 2) + (pair_count - 1) * vertical_gap
    grid_h = top_margin + slot_stack_h + bottom_margin + int(rng.integers(0, 4))

    max_slot_w = max(widths) + 2
    frag_region_w = int(rng.integers(12 + 2 * pair_count, 18 + 2 * pair_count))
    sep_gap = int(rng.integers(2, 5))
    right_margin = int(rng.integers(1, 3))
    slot_left = frag_region_w + sep_gap
    grid_w = slot_left + max_slot_w + right_margin

    grid = np.full((grid_h, grid_w), BG, dtype=int)
    occupied = np.zeros((grid_h, grid_w), dtype=bool)

    slot_order = list(range(pair_count))
    rng.shuffle(slot_order)
    slot_records = [None] * pair_count
    for visual_idx, pair_idx in enumerate(slot_order):
        pair = pair_records[pair_idx]
        top = top_margin + visual_idx * (inner_h + 2 + vertical_gap)
        slot_records[pair_idx] = {
            "top": top,
            "left": slot_left,
            "inner_h": inner_h,
            "inner_w": int(pair["width"]),
        }
        _draw_frame(grid, top, slot_left, inner_h, int(pair["width"]), FRAME_COLOR)
        frame_mask = np.zeros_like(grid, dtype=bool)
        frame_mask[top, slot_left : slot_left + pair["width"] + 2] = True
        frame_mask[top + inner_h + 1, slot_left : slot_left + pair["width"] + 2] = True
        frame_mask[top : top + inner_h + 2, slot_left] = True
        frame_mask[top : top + inner_h + 2, slot_left + pair["width"] + 1] = True
        occupied |= frame_mask

    pair_place_order = list(range(pair_count))
    rng.shuffle(pair_place_order)
    placements: list[dict] = []
    for pair_idx in pair_place_order:
        pair = pair_records[pair_idx]
        placed = _place_pair(
            grid,
            occupied,
            pair["left_patch"],
            pair["right_patch"],
            BG,
            rng,
            0,
            grid_h,
            0,
            slot_left - 2,
        )
        if placed is None:
            return None
        placements.append({"patch": pair["left_patch"], "pos": placed[0]})
        placements.append({"patch": pair["right_patch"], "pos": placed[1]})

    expected = _expected_output(grid, BG, slot_records, pair_records, placements)

    solved_bg, solved_slots, solved_fragments = _extract_objects(grid)
    if solved_bg != BG or len(solved_slots) != pair_count:
        return None
    solutions = _find_assignments(solved_slots, solved_fragments, limit=2)
    if len(solutions) != 1:
        return None
    if not np.array_equal(_render_solution(grid, BG, solved_slots, solved_fragments, solutions[0]), expected):
        return None
    return grid


def generate_input() -> np.ndarray:
    rng = np.random.default_rng()
    for _ in range(MAX_GENERATION_ATTEMPTS):
        grid = _gen_attempt(rng)
        if grid is not None:
            return grid
    raise RuntimeError("failed to generate a valid input")
