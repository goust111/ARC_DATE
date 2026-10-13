from __future__ import annotations

from collections import deque
import random

import numpy as np


BG = 0
FRAME_ASC = 7
FRAME_DESC = 9
FRAME_COLORS = (FRAME_ASC, FRAME_DESC)
PIXEL_COLORS = (1, 2, 3, 4)
MAX_GENERATE_ATTEMPTS = 400


def _neighbors8(r: int, c: int):
    for dr in (-1, 0, 1):
        for dc in (-1, 0, 1):
            if dr == 0 and dc == 0:
                continue
            yield r + dr, c + dc


def _components_mask(mask: np.ndarray, connectivity: int = 8) -> list[np.ndarray]:
    h, w = mask.shape
    seen = np.zeros((h, w), dtype=bool)
    comps: list[np.ndarray] = []
    if connectivity == 4:
        nbrs = [(-1, 0), (1, 0), (0, -1), (0, 1)]
    else:
        nbrs = [(dr, dc) for dr in (-1, 0, 1) for dc in (-1, 0, 1) if (dr, dc) != (0, 0)]

    for r in range(h):
        for c in range(w):
            if not mask[r, c] or seen[r, c]:
                continue
            q = deque([(r, c)])
            seen[r, c] = True
            pts: list[tuple[int, int]] = []
            while q:
                rr, cc = q.popleft()
                pts.append((rr, cc))
                for dr, dc in nbrs:
                    nr = rr + dr
                    nc = cc + dc
                    if 0 <= nr < h and 0 <= nc < w and mask[nr, nc] and not seen[nr, nc]:
                        seen[nr, nc] = True
                        q.append((nr, nc))
            comps.append(np.array(pts, dtype=int))
    return comps


def _bbox_coords(coords: np.ndarray) -> tuple[int, int, int, int]:
    r0, c0 = coords.min(axis=0)
    r1, c1 = coords.max(axis=0)
    return int(r0), int(c0), int(r1), int(c1)


def _diamond_record(frame_color: int, cr: int, cc: int, radius: int) -> dict:
    border = []
    interior_rows_by_col: dict[int, list[int]] = {}
    for r in range(cr - radius, cr + radius + 1):
        for c in range(cc - radius, cc + radius + 1):
            dist = abs(r - cr) + abs(c - cc)
            if dist == radius:
                border.append((r, c))
            elif dist < radius:
                interior_rows_by_col.setdefault(c, []).append(r)
    for rows in interior_rows_by_col.values():
        rows.sort()
    stack_rows = list(range(cr + radius - 1, cr - radius, -1))
    return {
        "color": int(frame_color),
        "center": (int(cr), int(cc)),
        "radius": int(radius),
        "bbox": (int(cr - radius), int(cc - radius), int(cr + radius), int(cc + radius)),
        "border": set(border),
        "interior_rows_by_col": interior_rows_by_col,
        "stack_slots": [(int(r), int(cc)) for r in stack_rows],
    }


def _find_diamonds(grid: np.ndarray) -> list[dict]:
    diamonds: list[dict] = []
    for color in FRAME_COLORS:
        for coords in _components_mask(grid == color, connectivity=8):
            r0, c0, r1, c1 = _bbox_coords(coords)
            h = r1 - r0 + 1
            w = c1 - c0 + 1
            assert h == w and h % 2 == 1 and h >= 5
            cr = (r0 + r1) // 2
            cc = (c0 + c1) // 2
            radius = h // 2
            rec = _diamond_record(color, cr, cc, radius)
            assert rec["bbox"] == (r0, c0, r1, c1)
            assert {tuple(map(int, p)) for p in coords} == rec["border"]
            diamonds.append(rec)
    diamonds.sort(key=lambda d: (d["center"][1], d["center"][0]))
    assert 2 <= len(diamonds) <= 3
    return diamonds


def _find_pixels(grid: np.ndarray) -> list[tuple[int, int, int]]:
    mask = np.isin(grid, PIXEL_COLORS)
    for comp in _components_mask(mask, connectivity=8):
        assert comp.shape[0] == 1
    pixels = [(int(r), int(c), int(grid[r, c])) for r, c in np.argwhere(mask)]
    assert pixels
    return pixels


def _assign_pixels_to_diamonds(
    pixels: list[tuple[int, int, int]],
    diamonds: list[dict],
) -> dict[int, list[tuple[int, int, int]]]:
    groups: dict[int, list[tuple[int, int, int]]] = {i: [] for i in range(len(diamonds))}
    for r, c, color in pixels:
        candidates = []
        for idx, diamond in enumerate(diamonds):
            rows = diamond["interior_rows_by_col"].get(c)
            if rows is not None and r < rows[0]:
                candidates.append(idx)
        assert len(candidates) == 1
        groups[candidates[0]].append((r, c, color))
    return groups


def _render_from_groups(
    shape: tuple[int, int],
    diamonds: list[dict],
    groups: dict[int, list[tuple[int, int, int]]],
    mode: str = "frame",
) -> np.ndarray:
    out = np.zeros(shape, dtype=int)
    for idx, diamond in enumerate(diamonds):
        items = list(groups[idx])
        if mode == "frame":
            reverse = diamond["color"] == FRAME_DESC
            items.sort(key=lambda x: (x[1], x[0]), reverse=reverse)
        elif mode == "asc":
            items.sort(key=lambda x: (x[1], x[0]))
        elif mode == "desc":
            items.sort(key=lambda x: (x[1], x[0]), reverse=True)
        elif mode == "row":
            items.sort(key=lambda x: (x[0], x[1]))
        else:
            raise ValueError(f"unknown mode: {mode}")
        assert len(items) <= len(diamond["stack_slots"])
        for (slot_r, slot_c), (_, _, color) in zip(diamond["stack_slots"], items, strict=False):
            out[slot_r, slot_c] = color
    return out


def main(grid: np.ndarray) -> np.ndarray:
    """
    Drop scattered colored pixels into the diamond bins, sort them by original column, and stack them on the bin axis.

    Input structure:
    - The background is black.
    - The lower half contains 2 or 3 hollow one-pixel-wide diamond frames colored orange or maroon.
    - The upper half contains isolated single pixels colored blue, red, green, or yellow; each such pixel has exactly one diamond interior below it in the same column.

    Rule:
    - Clear the original frame lines and the scattered source pixels.
    - Group each source pixel by the unique diamond whose interior the pixel would first enter by falling straight down in its original column.
    - For an orange frame, sort that group's pixels by original column from left to right; for a maroon frame, sort from right to left.
    - Place the sorted colors one per row along the diamond's center column, starting from the lowest interior row and moving upward.

    Key insight:
    - The diamond is a collecting container, not a reflector: gravity only decides the destination bin, while the frame color encodes the stacking order inside that bin.
    """
    g = np.asarray(grid, dtype=int)
    assert g.ndim == 2

    diamonds = _find_diamonds(g)
    pixels = _find_pixels(g)
    groups = _assign_pixels_to_diamonds(pixels, diamonds)
    return _render_from_groups(g.shape, diamonds, groups, mode="frame")


def _choose_counts(capacities: list[int]) -> list[int] | None:
    choices = [list(range(2, cap + 1)) for cap in capacities]
    all_counts: list[tuple[int, ...]] = []

    def backtrack(i: int, picked: list[int]) -> None:
        if i == len(choices):
            if len(set(picked)) == len(picked) and max(picked) >= 3:
                all_counts.append(tuple(picked))
            return
        for value in choices[i]:
            if value in picked:
                continue
            picked.append(value)
            backtrack(i + 1, picked)
            picked.pop()

    backtrack(0, [])
    if not all_counts:
        return None
    return list(random.choice(all_counts))


def _pick_palette_sequence(length: int) -> list[int]:
    for _ in range(200):
        seq = [random.choice(PIXEL_COLORS) for _ in range(length)]
        if len(set(seq)) < 2:
            continue
        if seq == list(reversed(seq)):
            continue
        return seq
    raise RuntimeError("failed to sample an informative color order")


def _place_singletons(
    grid: np.ndarray,
    requests: list[tuple[int, int]],
) -> list[tuple[int, int]]:
    placed: list[tuple[int, int]] = []
    occupied: set[tuple[int, int]] = set()
    for top_limit, col in requests:
        candidates = list(range(top_limit + 1))
        random.shuffle(candidates)
        found = None
        for row in candidates:
            if grid[row, col] != BG:
                continue
            ok = True
            for rr, cc in occupied:
                if abs(rr - row) <= 1 and abs(cc - col) <= 1:
                    ok = False
                    break
            if ok:
                found = row
                break
        if found is None:
            return []
        occupied.add((found, col))
        placed.append((found, col))
    return placed


def _frame_colors(count: int) -> list[int]:
    if count == 2:
        colors = [FRAME_ASC, FRAME_DESC]
    else:
        colors = [FRAME_ASC, FRAME_DESC, random.choice(FRAME_COLORS)]
    random.shuffle(colors)
    return colors


def _build_instance() -> tuple[np.ndarray, np.ndarray]:
    num_diamonds = random.choice([2, 3])
    radii = [random.choice([2, 3, 4]) for _ in range(num_diamonds)]
    widths = [2 * r + 1 for r in radii]
    gaps = [random.randint(2, 4) for _ in range(num_diamonds - 1)]
    min_w = sum(widths) + sum(gaps) + 2
    if min_w > 30:
        raise RuntimeError("layout too wide")
    w = random.randint(max(15, min_w), min(30, min_w + 6))

    max_radius = max(radii)
    min_h = max(18, 2 * max_radius + 8)
    h = random.randint(min_h, 26)

    left_margin = random.randint(1, max(1, w - (sum(widths) + sum(gaps)) - 1))
    centers_c = []
    cursor = left_margin
    for i, radius in enumerate(radii):
        centers_c.append(cursor + radius)
        cursor += 2 * radius + 1
        if i + 1 < num_diamonds:
            cursor += gaps[i]

    colors = _frame_colors(num_diamonds)
    diamonds = []
    bboxes = []
    for radius, cc, color in zip(radii, centers_c, colors, strict=False):
        bottom_margin = random.randint(1, 3)
        cr = random.randint(h - radius - bottom_margin - 1, h - radius - bottom_margin)
        rec = _diamond_record(color, cr, cc, radius)
        r0, c0, r1, c1 = rec["bbox"]
        if r0 < 4 or c0 < 0 or r1 >= h or c1 >= w:
            raise RuntimeError("diamond out of bounds")
        diamonds.append(rec)
        bboxes.append(rec["bbox"])

    for i in range(len(bboxes)):
        r0, c0, r1, c1 = bboxes[i]
        for j in range(i + 1, len(bboxes)):
            rr0, cc0, rr1, cc1 = bboxes[j]
            if not (r1 < rr0 or rr1 < r0 or c1 < cc0 or cc1 < c0):
                raise RuntimeError("diamonds overlap")

    capacities = [len(d["stack_slots"]) for d in diamonds]
    counts = _choose_counts(capacities)
    if counts is None:
        raise RuntimeError("no distinct count assignment")

    grid = np.zeros((h, w), dtype=int)
    assignments: dict[int, list[tuple[int, int, int]]] = {}
    placement_requests: list[tuple[int, int]] = []
    color_requests: list[tuple[int, int, int]] = []

    for idx, diamond in enumerate(diamonds):
        for r, c in diamond["border"]:
            grid[r, c] = diamond["color"]

        count = counts[idx]
        interior_cols = sorted(diamond["interior_rows_by_col"])
        if count > len(interior_cols):
            raise RuntimeError("not enough distinct columns")
        chosen_cols = sorted(random.sample(interior_cols, count))
        seq = _pick_palette_sequence(count)
        items: list[tuple[int, int, int]] = []
        for col, color in zip(chosen_cols, seq, strict=False):
            top_limit = diamond["interior_rows_by_col"][col][0] - 2
            if top_limit < 0:
                raise RuntimeError("not enough space above diamond")
            placement_requests.append((top_limit, col))
            color_requests.append((idx, col, color))
            items.append((-1, col, color))
        assignments[idx] = items

    placed_rows = _place_singletons(grid, placement_requests)
    if not placed_rows:
        raise RuntimeError("failed to place isolated source pixels")

    for idx, ((_, col, color), (row, final_col)) in enumerate(zip(color_requests, placed_rows, strict=False)):
        assert col == final_col
        grid[row, col] = color
        diamond_idx = color_requests[idx][0]
        bucket = assignments[diamond_idx]
        bucket_col_positions = [k for k, item in enumerate(bucket) if item[1] == col and item[2] == color and item[0] == -1]
        assert len(bucket_col_positions) == 1
        k = bucket_col_positions[0]
        bucket[k] = (row, col, color)

    diamonds = _find_diamonds(grid)
    groups = _assign_pixels_to_diamonds(_find_pixels(grid), diamonds)
    expected = _render_from_groups(grid.shape, diamonds, groups, mode="frame")

    counts_seen = sorted(len(groups[i]) for i in range(len(diamonds)))
    assert len(set(counts_seen)) == len(counts_seen)
    assert all(v >= 2 for v in counts_seen)

    out_asc = _render_from_groups(grid.shape, diamonds, groups, mode="asc")
    out_desc = _render_from_groups(grid.shape, diamonds, groups, mode="desc")
    out_row = _render_from_groups(grid.shape, diamonds, groups, mode="row")
    assert not np.array_equal(expected, out_asc)
    assert not np.array_equal(expected, out_desc)
    assert not np.array_equal(expected, out_row)
    assert np.array_equal(main(grid), expected)
    return grid, expected


def generate_input() -> np.ndarray:
    for _ in range(MAX_GENERATE_ATTEMPTS):
        try:
            grid, _ = _build_instance()
        except (AssertionError, RuntimeError, ValueError):
            continue
        return grid
    raise RuntimeError("failed to generate a valid non-degenerate input")
