from __future__ import annotations

from collections import Counter, deque
import random

import numpy as np


BG = 0
BODY = 5
ENDPOINT_COLORS = (1, 2, 3, 4, 6, 7, 8, 9)

GEN_ATTEMPTS = 400
PIECE_ATTEMPTS = 120
PLACE_ATTEMPTS = 200


def _neighbors4(r: int, c: int):
    yield r - 1, c
    yield r + 1, c
    yield r, c - 1
    yield r, c + 1


def _component_mask(grid: np.ndarray, start_r: int, start_c: int, seen: np.ndarray) -> list[tuple[int, int]]:
    q: deque[tuple[int, int]] = deque([(int(start_r), int(start_c))])
    seen[start_r, start_c] = True
    coords: list[tuple[int, int]] = []
    h, w = grid.shape

    while q:
        r, c = q.popleft()
        coords.append((r, c))
        for nr, nc in _neighbors4(r, c):
            if 0 <= nr < h and 0 <= nc < w and not seen[nr, nc] and grid[nr, nc] != BG:
                seen[nr, nc] = True
                q.append((nr, nc))
    return coords


def _find_components(grid: np.ndarray) -> list[dict]:
    seen = np.zeros(grid.shape, dtype=bool)
    pieces: list[dict] = []
    h, w = grid.shape

    for r in range(h):
        for c in range(w):
            if grid[r, c] == BG or seen[r, c]:
                continue
            coords = _component_mask(grid, r, c, seen)
            rows = [x for x, _ in coords]
            cols = [y for _, y in coords]
            r0, r1 = min(rows), max(rows)
            c0, c1 = min(cols), max(cols)
            patch = grid[r0 : r1 + 1, c0 : c1 + 1].copy()
            pieces.append({"bbox": (r0, c0, r1, c1), "patch": patch})

    assert pieces, "expected at least one path segment"
    return pieces


def _degree(mask: np.ndarray, r: int, c: int) -> int:
    deg = 0
    h, w = mask.shape
    for nr, nc in _neighbors4(r, c):
        if 0 <= nr < h and 0 <= nc < w and mask[nr, nc]:
            deg += 1
    return deg


def _piece_summary(patch: np.ndarray) -> dict:
    mask = patch != BG
    assert mask.any(), "empty piece"
    h, w = patch.shape

    left = []
    right = []
    endpoints = []
    for r, c in np.argwhere(mask):
        deg = _degree(mask, int(r), int(c))
        if deg == 1:
            endpoints.append((int(r), int(c)))
            if int(c) == 0:
                left.append((int(r), int(c)))
            if int(c) == w - 1:
                right.append((int(r), int(c)))

    assert len(endpoints) == 2, "piece must be a simple path"
    assert len(left) == 1 and len(right) == 1, "piece must expose one endpoint on each outer side"
    left_r, left_c = left[0]
    right_r, right_c = right[0]
    return {
        "patch": patch,
        "left_coord": (left_r, left_c),
        "right_coord": (right_r, right_c),
        "left_color": int(patch[left_r, left_c]),
        "right_color": int(patch[right_r, right_c]),
        "width": w,
        "height": h,
    }


def _orient_piece(patch: np.ndarray, want_left_color: int) -> dict:
    info = _piece_summary(patch)
    if info["left_color"] == int(want_left_color):
        return info
    if info["right_color"] == int(want_left_color):
        return _piece_summary(np.fliplr(patch))
    raise AssertionError("requested endpoint color is not present on the piece")


def _render_placed(placements: list[tuple[np.ndarray, int, int]]) -> np.ndarray:
    painted: list[tuple[int, int, int]] = []
    for patch, top, left in placements:
        for r, c in np.argwhere(patch != BG):
            painted.append((int(top + r), int(left + c), int(patch[r, c])))

    min_r = min(r for r, _, _ in painted)
    max_r = max(r for r, _, _ in painted)
    min_c = min(c for _, c, _ in painted)
    max_c = max(c for _, c, _ in painted)

    out = np.full((max_r - min_r + 1, max_c - min_c + 1), BG, dtype=int)
    for r, c, color in painted:
        out[r - min_r, c - min_c] = color
    return out


def main(grid: np.ndarray) -> np.ndarray:
    """
    Reassemble scattered endpoint-colored path segments into one slid-and-aligned chain.

    Input structure:
    - A black background contains several disconnected path-like segments whose interior pixels are gray.
    - Each segment is a simple 4-connected path with exactly one degree-1 endpoint on its left outer column and one on its right outer column.
    - Endpoint pixels are colored, and across all endpoints exactly two colors appear once while every other endpoint color appears exactly twice.

    Rule:
    - Use equal endpoint colors to determine the unique chain order; a segment may be horizontally flipped, but never rotated, so the matched endpoint sits on the required side.
    - Choose as the first segment the unique-end segment whose input bbox is highest, breaking ties by smaller column, then orient it so its unique endpoint is on the left and keep its original top row.
    - Append each next segment immediately to the right, shifting it vertically so the new left endpoint is on the same row as the current chain's right endpoint, and crop the final chain to its tight bbox.

    Key insight:
    - Screen position is a decoy: the chain is recovered from endpoint-color pairings, then each segment is only flipped horizontally and slid vertically to make matching endpoints touch.
    """
    g = np.asarray(grid, dtype=int)
    assert g.ndim == 2

    raw_pieces = _find_components(g)
    pieces: list[dict] = []
    counts: Counter[int] = Counter()
    for raw in raw_pieces:
        info = _piece_summary(raw["patch"])
        info["bbox"] = raw["bbox"]
        pieces.append(info)
        counts[info["left_color"]] += 1
        counts[info["right_color"]] += 1

    unique_colors = {color for color, count in counts.items() if count == 1}
    assert len(unique_colors) == 2, "expected exactly two outer endpoint colors"

    start_candidates = [
        i
        for i, piece in enumerate(pieces)
        if piece["left_color"] in unique_colors or piece["right_color"] in unique_colors
    ]
    assert start_candidates, "expected at least one segment carrying a unique endpoint color"
    start_idx = min(start_candidates, key=lambda i: (pieces[i]["bbox"][0], pieces[i]["bbox"][1]))

    start_piece = pieces[start_idx]
    if start_piece["left_color"] in unique_colors:
        current = _piece_summary(start_piece["patch"])
    else:
        current = _piece_summary(np.fliplr(start_piece["patch"]))

    placements: list[tuple[np.ndarray, int, int]] = []
    current_top = int(start_piece["bbox"][0])
    current_left = 0
    placements.append((current["patch"], current_top, current_left))

    used = {start_idx}
    current_right_row = current_top + int(current["right_coord"][0])
    current_right_col = current_left + int(current["right_coord"][1])
    current_color = int(current["right_color"])

    while len(used) < len(pieces):
        matches = [
            i
            for i, piece in enumerate(pieces)
            if i not in used and (piece["left_color"] == current_color or piece["right_color"] == current_color)
        ]
        assert len(matches) == 1, "endpoint pairing must define a unique next segment"
        next_idx = matches[0]
        oriented = _orient_piece(pieces[next_idx]["patch"], current_color)
        top = current_right_row - int(oriented["left_coord"][0])
        left = current_right_col + 1
        placements.append((oriented["patch"], top, left))

        used.add(next_idx)
        current_right_row = top + int(oriented["right_coord"][0])
        current_right_col = left + int(oriented["right_coord"][1])
        current_color = int(oriented["right_color"])

    return _render_placed(placements)


def _make_segment(left_color: int, right_color: int) -> np.ndarray:
    for _ in range(PIECE_ATTEMPTS):
        width = random.randint(3, 6)
        height = random.randint(3, 7)
        start_row = random.randrange(height)
        end_choices = [r for r in range(height) if r != start_row]
        end_row = random.choice(end_choices)
        direction = 1 if end_row > start_row else -1
        total_shift = abs(end_row - start_row)

        column_shifts = [0] * (width - 1)
        for _step in range(total_shift):
            column_shifts[random.randrange(width - 1)] += 1

        rows = [start_row]
        current_row = start_row
        for shift in column_shifts:
            current_row += direction * shift
            rows.append(current_row)

        patch = np.full((height, width), BG, dtype=int)
        cells: list[tuple[int, int]] = [(rows[0], 0)]
        for c in range(width - 1):
            cur_r = rows[c]
            next_r = rows[c + 1]
            cells.append((cur_r, c + 1))
            if next_r != cur_r:
                direction = 1 if next_r > cur_r else -1
                for r in range(cur_r + direction, next_r + direction, direction):
                    cells.append((r, c + 1))

        if len(cells) != len(set(cells)):
            continue

        for r, c in cells:
            patch[r, c] = BODY
        patch[cells[0]] = int(left_color)
        patch[cells[-1]] = int(right_color)

        occupied = np.argwhere(patch != BG)
        r0, c0 = occupied.min(axis=0)
        r1, c1 = occupied.max(axis=0)
        patch = patch[int(r0) : int(r1) + 1, int(c0) : int(c1) + 1]
        info = _piece_summary(patch)
        if info["left_color"] == int(left_color) and info["right_color"] == int(right_color):
            return patch

    raise AssertionError("failed to generate a valid path segment")


def _can_place(blocked: np.ndarray, patch: np.ndarray, top: int, left: int) -> bool:
    h, w = patch.shape
    if top < 0 or left < 0 or top + h > blocked.shape[0] or left + w > blocked.shape[1]:
        return False
    coords = np.argwhere(patch != BG)
    for r, c in coords:
        if blocked[top + int(r), left + int(c)]:
            return False
    return True


def _mark_blocked(blocked: np.ndarray, patch: np.ndarray, top: int, left: int) -> None:
    for r, c in np.argwhere(patch != BG):
        rr = top + int(r)
        cc = left + int(c)
        for nr in range(rr - 1, rr + 2):
            for nc in range(cc - 1, cc + 2):
                if 0 <= nr < blocked.shape[0] and 0 <= nc < blocked.shape[1]:
                    blocked[nr, nc] = True


def _rect_clear(rects: list[tuple[int, int, int, int]], top: int, left: int, h: int, w: int) -> bool:
    bottom = top + h - 1
    right = left + w - 1
    for r0, c0, r1, c1 in rects:
        if not (bottom < r0 - 1 or top > r1 + 1 or right < c0 - 1 or left > c1 + 1):
            return False
    return True


def _place_scattered(patches: list[np.ndarray], canvas_h: int, canvas_w: int):
    blocked = np.zeros((canvas_h, canvas_w), dtype=bool)
    grid = np.full((canvas_h, canvas_w), BG, dtype=int)
    positions: list[tuple[int, int]] = [(-1, -1)] * len(patches)
    rects: list[tuple[int, int, int, int]] = []

    order = list(range(len(patches)))
    random.shuffle(order)
    for idx in order:
        patch = patches[idx]
        ph, pw = patch.shape
        placed = False
        for _ in range(PLACE_ATTEMPTS):
            top = random.randint(0, canvas_h - ph)
            left = random.randint(0, canvas_w - pw)
            if not _rect_clear(rects, top, left, ph, pw):
                continue
            if not _can_place(blocked, patch, top, left):
                continue
            coords = np.argwhere(patch != BG)
            for r, c in coords:
                grid[top + int(r), left + int(c)] = int(patch[r, c])
            _mark_blocked(blocked, patch, top, left)
            positions[idx] = (top, left)
            rects.append((top, left, top + ph - 1, left + pw - 1))
            placed = True
            break
        if not placed:
            return None
    return grid, positions


def _x_order(positions: list[tuple[int, int]]) -> list[int]:
    return sorted(range(len(positions)), key=lambda i: (positions[i][1], positions[i][0]))


def generate_input() -> np.ndarray:
    for _ in range(GEN_ATTEMPTS):
        n_pieces = random.randint(3, 5)
        colors = random.sample(ENDPOINT_COLORS, n_pieces + 1)

        chain = [_make_segment(colors[i], colors[i + 1]) for i in range(n_pieces)]
        input_patches = [np.fliplr(p) if random.random() < 0.5 else p.copy() for p in chain]

        total_width = sum(int(p.shape[1]) for p in input_patches)
        max_height = max(int(p.shape[0]) for p in input_patches)
        min_h = max(12, max_height + 4)
        min_w = max(16, max(int(total_width * 0.7), max(p.shape[1] for p in input_patches) + 4))
        canvas_h = random.randint(min_h, min(26, max(min_h, min_h + 8)))
        canvas_w = random.randint(min_w, 30)

        placed = _place_scattered(input_patches, canvas_h, canvas_w)
        if placed is None:
            continue
        grid, positions = placed

        left_to_right = _x_order(positions)
        if left_to_right == list(range(n_pieces)) or left_to_right == list(reversed(range(n_pieces))):
            continue

        output = main(grid)
        if output.shape[0] > 30 or output.shape[1] > 30:
            continue
        if output.shape == grid.shape and np.array_equal(output, grid):
            continue
        return grid

    raise AssertionError("failed to generate a valid ARC instance")
