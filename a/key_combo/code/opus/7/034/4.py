from __future__ import annotations

import random
from collections import deque

import numpy as np


BG = 0
LEGEND_BORDER = 4
FRAME_BORDER = 5
EXEMPT_COLOR = 9
KEY_COLOR_POOL = (1, 2, 3, 6, 7, 8)
GEN_ATTEMPTS = 400


def _neighbors4(r: int, c: int) -> tuple[tuple[int, int], ...]:
    return ((r - 1, c), (r + 1, c), (r, c - 1), (r, c + 1))


def _components(mask: np.ndarray) -> list[np.ndarray]:
    h, w = mask.shape
    seen = np.zeros_like(mask, dtype=bool)
    comps: list[np.ndarray] = []
    for r in range(h):
        for c in range(w):
            if not mask[r, c] or seen[r, c]:
                continue
            q: deque[tuple[int, int]] = deque([(r, c)])
            seen[r, c] = True
            coords: list[tuple[int, int]] = []
            while q:
                rr, cc = q.popleft()
                coords.append((rr, cc))
                for nr, nc in _neighbors4(rr, cc):
                    if 0 <= nr < h and 0 <= nc < w and mask[nr, nc] and not seen[nr, nc]:
                        seen[nr, nc] = True
                        q.append((nr, nc))
            comps.append(np.array(coords, dtype=int))
    return comps


def _bbox(coords: np.ndarray) -> tuple[int, int, int, int]:
    r0, c0 = coords.min(axis=0)
    r1, c1 = coords.max(axis=0)
    return int(r0), int(c0), int(r1), int(c1)


def _is_rect_frame(mask: np.ndarray, box: tuple[int, int, int, int]) -> bool:
    r0, c0, r1, c1 = box
    if r1 - r0 < 2 or c1 - c0 < 2:
        return False
    sub = mask[r0 : r1 + 1, c0 : c1 + 1]
    border = np.zeros_like(sub, dtype=bool)
    border[0, :] = True
    border[-1, :] = True
    border[:, 0] = True
    border[:, -1] = True
    return bool(np.array_equal(sub, border))


def _find_rect_frames(grid: np.ndarray, color: int) -> list[tuple[int, int, int, int]]:
    frames: list[tuple[int, int, int, int]] = []
    for coords in _components(grid == int(color)):
        box = _bbox(coords)
        if _is_rect_frame(grid == int(color), box):
            frames.append(box)
    return frames


def _find_legend(grid: np.ndarray) -> tuple[tuple[int, int, int, int], list[tuple[int, int]], int, int]:
    candidates = []
    for box in _find_rect_frames(grid, LEGEND_BORDER):
        r0, c0, r1, c1 = box
        interior = grid[r0 + 1 : r1, c0 + 1 : c1]
        if interior.shape[0] < 2 or interior.shape[1] < 3:
            continue
        key_col = None
        for cc in range(interior.shape[1]):
            if np.any(interior[:-1, cc] != BG):
                key_col = cc
                break
        if key_col is None:
            continue
        keys = [(rr, int(interior[rr, key_col])) for rr in range(interior.shape[0] - 1) if int(interior[rr, key_col]) != BG]
        bottom_vals = [int(v) for v in interior[-1] if int(v) != BG]
        if not keys or len(bottom_vals) != 1:
            continue
        candidates.append((box, keys, bottom_vals[0], key_col))
    assert len(candidates) == 1, "expected a unique yellow legend frame"
    return candidates[0]


def _count_colors_in_frames(
    grid: np.ndarray,
    frame_boxes: list[tuple[int, int, int, int]],
    key_colors: list[int],
    exempt_color: int,
) -> dict[int, int]:
    counts = {int(color): 0 for color in key_colors}
    for r0, c0, r1, c1 in frame_boxes:
        interior = grid[r0 + 1 : r1, c0 + 1 : c1]
        adjacent_exempt: dict[int, set[tuple[int, int]]] = {int(color): set() for color in key_colors}
        for rr, cc in np.argwhere(interior == int(exempt_color)):
            rr = int(rr)
            cc = int(cc)
            for nr, nc in _neighbors4(rr, cc):
                if 0 <= nr < interior.shape[0] and 0 <= nc < interior.shape[1]:
                    color = int(interior[nr, nc])
                    if color in counts:
                        adjacent_exempt[color].add((int(nr), int(nc)))
        for color in key_colors:
            raw = int(np.count_nonzero(interior == int(color)))
            counts[int(color)] += raw - len(adjacent_exempt[int(color)])
    return counts


def _paint_legend_counts(
    out: np.ndarray,
    legend_box: tuple[int, int, int, int],
    key_rows: list[tuple[int, int]],
    key_col: int,
    counts: dict[int, int],
) -> np.ndarray:
    r0, c0, r1, c1 = legend_box
    interior = out[r0 + 1 : r1, c0 + 1 : c1]
    for rr, color in key_rows:
        rr = int(rr)
        color = int(color)
        interior[rr, :] = BG
        interior[rr, key_col] = color
        start = key_col + 2
        for i in range(int(counts[color])):
            cc = start + 2 * i
            if 0 <= cc < interior.shape[1]:
                interior[rr, cc] = color
    return out


def main(grid: np.ndarray) -> np.ndarray:
    """
    Fill the legend with per-color tallies counted inside grey frames after subtracting exemption-adjacent pixels.

    Input structure:
    - A black background contains one yellow one-pixel rectangular legend frame and several grey one-pixel hollow rectangular frames.
    - The legend interior has key-color pixels in its leftmost used column, plus a single colored marker on the bottom interior row.
    - Each grey frame encloses isolated key-color pixels and optional exemption-color pixels, while same-colored distractor pixels may also appear outside every grey frame.

    Rule:
    - Read the legend keys from top to bottom and read the bottom-row marker color as the global exemption color.
    - For each key color, count all pixels of that color strictly inside the grey frames.
    - Subtract each counted key-color pixel that is 4-neighbor adjacent to at least one exemption-color pixel inside the same grey frame, then place that many dots in the matching legend row starting two cells to the right of the key and stepping every other cell.

    Key insight:
    - The legend is self-referential: its bottom marker defines which interior frame-adjacent pixels are ignored, and the remaining totals are written back into the legend rows.
    """
    g = np.asarray(grid, dtype=int)
    legend_box, key_rows, exempt_color, key_col = _find_legend(g)
    frame_boxes = _find_rect_frames(g, FRAME_BORDER)
    key_colors = [int(color) for _row, color in key_rows]
    counts = _count_colors_in_frames(g, frame_boxes, key_colors, int(exempt_color))
    out = g.copy()
    return _paint_legend_counts(out, legend_box, key_rows, key_col, counts)


def _choose_counts(include_exempt: bool, key_colors: list[int]) -> tuple[dict[int, int], dict[int, int]]:
    final_counts = {int(color): random.randint(1, 4) for color in key_colors}
    if random.random() < 0.35:
        boosted = random.choice(key_colors)
        final_counts[int(boosted)] = min(5, final_counts[int(boosted)] + random.randint(1, 2))

    subtract_counts = {int(color): 0 for color in key_colors}
    if include_exempt:
        colors = key_colors[:]
        random.shuffle(colors)
        positives = random.randint(1, min(len(colors), 3))
        for color in colors[:positives]:
            subtract_counts[int(color)] = random.randint(1, 2)
    return final_counts, subtract_counts


def _pick_exempt_positions(inner_h: int, inner_w: int, count: int) -> list[tuple[int, int]] | None:
    candidates = [
        (r, c)
        for r in range(inner_h)
        for c in range(inner_w)
        if (r, c) not in ((0, 0), (0, inner_w - 1), (inner_h - 1, 0), (inner_h - 1, inner_w - 1))
    ]
    if count == 0:
        return []
    if len(candidates) < count:
        return None
    for _ in range(50):
        picked = random.sample(candidates, count)
        if len(set(picked)) == count:
            return picked
    return None


def _frame_cell_sets(
    inner_h: int,
    inner_w: int,
    exempt_positions: list[tuple[int, int]],
) -> tuple[set[tuple[int, int]], set[tuple[int, int]]]:
    exempt_set = set(exempt_positions)
    adjacent: set[tuple[int, int]] = set()
    for rr, cc in exempt_positions:
        for nr, nc in _neighbors4(rr, cc):
            if 0 <= nr < inner_h and 0 <= nc < inner_w and (nr, nc) not in exempt_set:
                adjacent.add((int(nr), int(nc)))
    safe = {
        (r, c)
        for r in range(inner_h)
        for c in range(inner_w)
        if (r, c) not in exempt_set and (r, c) not in adjacent
    }
    return adjacent, safe


def _build_frame_specs(
    key_colors: list[int],
    final_counts: dict[int, int],
    subtract_counts: dict[int, int],
    include_exempt: bool,
) -> list[dict] | None:
    total_final = sum(final_counts.values())
    total_sub = sum(subtract_counts.values())
    for _ in range(200):
        n_frames = random.randint(2, 4)
        frame_specs = []
        exempt_counts = [0] * n_frames
        if include_exempt:
            n_exempt_frames = random.randint(1, max(1, n_frames - 1))
            for idx in random.sample(range(n_frames), n_exempt_frames):
                exempt_counts[idx] = random.randint(1, 2)

        for i in range(n_frames):
            if exempt_counts[i] > 0:
                outer_h = random.randint(5, 7)
                outer_w = random.randint(5, 7)
            else:
                outer_h = random.randint(4, 7)
                outer_w = random.randint(4, 7)
            inner_h = outer_h - 2
            inner_w = outer_w - 2
            exempt_positions = _pick_exempt_positions(inner_h, inner_w, exempt_counts[i])
            if exempt_positions is None:
                break
            adjacent, safe = _frame_cell_sets(inner_h, inner_w, exempt_positions)
            frame_specs.append(
                {
                    "outer_h": outer_h,
                    "outer_w": outer_w,
                    "inner_h": inner_h,
                    "inner_w": inner_w,
                    "exempt_positions": exempt_positions,
                    "adjacent_free": set(adjacent),
                    "safe_free": set(safe),
                    "pixels": {},
                }
            )
        if len(frame_specs) != n_frames:
            continue
        if sum(len(spec["safe_free"]) for spec in frame_specs) < total_final:
            continue
        if sum(len(spec["adjacent_free"]) for spec in frame_specs) < total_sub:
            continue
        populated = _populate_frame_specs(frame_specs, key_colors, final_counts, subtract_counts)
        if populated is not None:
            return populated
    return None


def _consume_color(remaining: dict[int, int]) -> int | None:
    choices = [color for color, count in remaining.items() if count > 0]
    if not choices:
        return None
    max_count = max(remaining[color] for color in choices)
    top = [color for color in choices if remaining[color] == max_count]
    return int(random.choice(top))


def _populate_frame_specs(
    frame_specs: list[dict],
    key_colors: list[int],
    final_counts: dict[int, int],
    subtract_counts: dict[int, int],
) -> list[dict] | None:
    remaining_final = {int(color): int(final_counts[int(color)]) for color in key_colors}
    remaining_sub = {int(color): int(subtract_counts[int(color)]) for color in key_colors}

    for spec in frame_specs:
        can_final = bool(spec["safe_free"]) and sum(remaining_final.values()) > 0
        can_sub = bool(spec["adjacent_free"]) and sum(remaining_sub.values()) > 0
        if not can_final and not can_sub:
            return None
        if can_final and (not can_sub or sum(remaining_final.values()) >= sum(remaining_sub.values()) or random.random() < 0.7):
            color = _consume_color(remaining_final)
            pos = random.choice(tuple(spec["safe_free"]))
            spec["safe_free"].remove(pos)
            spec["pixels"][pos] = int(color)
            remaining_final[int(color)] -= 1
        else:
            color = _consume_color(remaining_sub)
            if color is None:
                return None
            pos = random.choice(tuple(spec["adjacent_free"]))
            spec["adjacent_free"].remove(pos)
            spec["pixels"][pos] = int(color)
            remaining_sub[int(color)] -= 1

    sub_items = [int(color) for color, count in remaining_sub.items() for _ in range(count)]
    random.shuffle(sub_items)
    for color in sub_items:
        candidates = [spec for spec in frame_specs if spec["adjacent_free"]]
        if not candidates:
            return None
        spec = random.choice(candidates)
        pos = random.choice(tuple(spec["adjacent_free"]))
        spec["adjacent_free"].remove(pos)
        spec["pixels"][pos] = int(color)

    final_items = [int(color) for color, count in remaining_final.items() for _ in range(count)]
    random.shuffle(final_items)
    for color in final_items:
        candidates = [spec for spec in frame_specs if spec["safe_free"]]
        if not candidates:
            return None
        spec = random.choice(candidates)
        pos = random.choice(tuple(spec["safe_free"]))
        spec["safe_free"].remove(pos)
        spec["pixels"][pos] = int(color)

    return frame_specs


def _render_frame_patch(spec: dict) -> np.ndarray:
    patch = np.full((spec["outer_h"], spec["outer_w"]), BG, dtype=int)
    patch[0, :] = FRAME_BORDER
    patch[-1, :] = FRAME_BORDER
    patch[:, 0] = FRAME_BORDER
    patch[:, -1] = FRAME_BORDER
    for rr, cc in spec["exempt_positions"]:
        patch[1 + rr, 1 + cc] = EXEMPT_COLOR
    for (rr, cc), color in spec["pixels"].items():
        patch[1 + rr, 1 + cc] = int(color)
    return patch


def _legend_patch(key_colors: list[int], counts: dict[int, int]) -> tuple[np.ndarray, list[tuple[int, int]], int]:
    max_count = max(counts.values())
    slack = random.randint(0, 2)
    inner_h = len(key_colors) + 1
    inner_w = 2 * max_count + 1 + slack
    patch = np.full((inner_h + 2, inner_w + 2), BG, dtype=int)
    patch[0, :] = LEGEND_BORDER
    patch[-1, :] = LEGEND_BORDER
    patch[:, 0] = LEGEND_BORDER
    patch[:, -1] = LEGEND_BORDER

    key_rows: list[tuple[int, int]] = []
    for i, color in enumerate(key_colors):
        rr = 1 + i
        patch[rr, 1] = int(color)
        key_rows.append((rr, int(color)))

    marker_col = random.randint(2, inner_w) if inner_w >= 2 else 1
    patch[inner_h, marker_col] = EXEMPT_COLOR
    return patch, key_rows, 1


def _can_place(occupied: np.ndarray, top: int, left: int, h: int, w: int) -> bool:
    r0 = max(0, top - 1)
    c0 = max(0, left - 1)
    r1 = min(occupied.shape[0], top + h + 1)
    c1 = min(occupied.shape[1], left + w + 1)
    return not bool(occupied[r0:r1, c0:c1].any())


def _mark_occupied(occupied: np.ndarray, top: int, left: int, h: int, w: int) -> None:
    r0 = max(0, top - 1)
    c0 = max(0, left - 1)
    r1 = min(occupied.shape[0], top + h + 1)
    c1 = min(occupied.shape[1], left + w + 1)
    occupied[r0:r1, c0:c1] = True


def _paste(dst: np.ndarray, patch: np.ndarray, top: int, left: int) -> None:
    dst[top : top + patch.shape[0], left : left + patch.shape[1]] = patch


def _place_scene(
    legend_patch: np.ndarray,
    frame_patches: list[np.ndarray],
    layout_mode: str,
) -> tuple[np.ndarray, tuple[int, int, int, int]] | None:
    legend_h, legend_w = legend_patch.shape
    frame_hs = [patch.shape[0] for patch in frame_patches]
    frame_ws = [patch.shape[1] for patch in frame_patches]
    total_area = sum((h + 2) * (w + 2) for h, w in zip(frame_hs, frame_ws))

    left_w = max(max(frame_ws) + 3, min(15, int(np.ceil(np.sqrt(total_area * 1.4)))))
    grid_w = left_w + legend_w + random.randint(3, 5)
    if grid_w > 30:
        return None
    min_h = max(legend_h + 2, int(np.ceil(total_area / max(6, left_w - 1))) + 4)
    grid_h = min(30, max(18, min_h + random.randint(0, 4)))
    if grid_h < max(max(frame_hs) + 2, legend_h + 2):
        return None

    grid = np.zeros((grid_h, grid_w), dtype=int)
    occupied = np.zeros((grid_h, grid_w), dtype=bool)

    legend_top = random.randint(1, grid_h - legend_h - 1)
    legend_left = grid_w - legend_w - random.randint(1, 2)
    _paste(grid, legend_patch, legend_top, legend_left)
    _mark_occupied(occupied, legend_top, legend_left, legend_h, legend_w)

    left_limit = legend_left - 2
    if left_limit <= max(frame_ws):
        return None

    positions: list[tuple[int, int]] = []
    order = sorted(range(len(frame_patches)), key=lambda i: frame_patches[i].shape[0] * frame_patches[i].shape[1], reverse=True)
    stack_cursor = 1
    for idx in order:
        patch = frame_patches[idx]
        ph, pw = patch.shape
        placed = None
        if layout_mode == "stacked":
            for _ in range(40):
                top_lo = stack_cursor
                top_hi = min(grid_h - ph - 1, stack_cursor + 2)
                if top_lo > top_hi:
                    break
                top = random.randint(top_lo, top_hi)
                left = random.randint(1, max(1, min(left_limit - pw - 1, 3)))
                if _can_place(occupied, top, left, ph, pw):
                    placed = (top, left)
                    stack_cursor = top + ph + random.randint(1, 2)
                    break
        if placed is None:
            for _ in range(250):
                top = random.randint(1, grid_h - ph - 1)
                left = random.randint(1, max(1, left_limit - pw - 1))
                if _can_place(occupied, top, left, ph, pw):
                    placed = (top, left)
                    break
        if placed is None:
            return None
        top, left = placed
        _paste(grid, patch, top, left)
        _mark_occupied(occupied, top, left, ph, pw)
        positions.append((idx, top, left))

    frame_boxes = [None] * len(frame_patches)
    for idx, top, left in positions:
        ph, pw = frame_patches[idx].shape
        frame_boxes[idx] = (top, left, top + ph - 1, left + pw - 1)

    return grid, tuple(frame_boxes)  # type: ignore[arg-type]


def _place_distractors(
    grid: np.ndarray,
    forbidden_boxes: list[tuple[int, int, int, int]],
    key_colors: list[int],
    include_exempt: bool,
) -> None:
    h, w = grid.shape
    buffer = np.zeros((h, w), dtype=bool)
    for r0, c0, r1, c1 in forbidden_boxes:
        rr0 = max(0, r0 - 1)
        cc0 = max(0, c0 - 1)
        rr1 = min(h, r1 + 2)
        cc1 = min(w, c1 + 2)
        buffer[rr0:rr1, cc0:cc1] = True

    colors = list(key_colors)
    if include_exempt and random.random() < 0.7:
        colors.append(EXEMPT_COLOR)

    n_points = random.randint(4, 9)
    free = [(r, c) for r in range(h) for c in range(w) if grid[r, c] == BG and not buffer[r, c]]
    random.shuffle(free)
    for r, c in free[:n_points]:
        color = random.choice(colors)
        grid[r, c] = int(color)


def _expected_output(
    grid: np.ndarray,
    legend_box: tuple[int, int, int, int],
    key_colors: list[int],
    counts: dict[int, int],
) -> np.ndarray:
    out = grid.copy()
    r0, c0, r1, c1 = legend_box
    interior = out[r0 + 1 : r1, c0 + 1 : c1]
    for i, color in enumerate(key_colors):
        rr = i
        interior[rr, :] = BG
        interior[rr, 0] = int(color)
        for j in range(int(counts[int(color)])):
            cc = 2 + 2 * j
            if cc < interior.shape[1]:
                interior[rr, cc] = int(color)
    return out


def _generate_attempt() -> np.ndarray | None:
    key_count = random.randint(2, 4)
    key_colors = random.sample(list(KEY_COLOR_POOL), key_count)
    include_exempt = random.random() < 0.7
    final_counts, subtract_counts = _choose_counts(include_exempt, key_colors)

    frame_specs = _build_frame_specs(key_colors, final_counts, subtract_counts, include_exempt)
    if frame_specs is None:
        return None

    legend_patch, _key_rows_abs, _key_col_abs = _legend_patch(key_colors, final_counts)
    frame_patches = [_render_frame_patch(spec) for spec in frame_specs]
    layout_mode = random.choice(("stacked", "scatter"))
    placed = _place_scene(legend_patch, frame_patches, layout_mode)
    if placed is None:
        return None
    grid, frame_boxes = placed

    legend_h, legend_w = legend_patch.shape
    legend_box = None
    for box in _find_rect_frames(grid, LEGEND_BORDER):
        r0, c0, r1, c1 = box
        if r1 - r0 + 1 == legend_h and c1 - c0 + 1 == legend_w:
            legend_box = box
            break
    if legend_box is None:
        return None

    _place_distractors(grid, [legend_box, *list(frame_boxes)], key_colors, include_exempt)
    expected = _expected_output(grid, legend_box, key_colors, final_counts)

    try:
        predicted = main(grid)
    except Exception:
        return None
    if not np.array_equal(predicted, expected):
        return None
    return grid


def generate_input() -> np.ndarray:
    for _ in range(GEN_ATTEMPTS):
        grid = _generate_attempt()
        if grid is not None:
            return grid
    raise RuntimeError("failed to generate a valid input")
