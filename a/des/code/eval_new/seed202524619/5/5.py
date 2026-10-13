"""Map nested-frame annulus sprites to colors from the border sequence.
Input structure:
- bg is the most frequent color.
- 1-cell-thick rectangular grey frames form a strict nesting hierarchy (largest to smallest).
- Small monochrome sprites occupy the annular regions between adjacent frames or outside the outer frame.
- A color sequence bar on the bottom row or rightmost column gives the target color for each nested layer from outside to inside.
Rule:
- Detect frame rectangles from connected-frame-color border rings and compute each pixel's nesting depth by counting how many frame interiors contain it.
- Determine the layer of each connected sprite component from its depth, where layer 1 is outside all frames.
- Repaint every sprite component entirely with the corresponding color from the sequence (k-th layer → k-th sequence pixel).
- Keep frame pixels unchanged and erase the border color sequence in the output.
Key insight:
- Layering is implicit in spatial containment, so solving requires reconstructing the nesting depth first instead of using explicit color cues on the frames.
"""

from __future__ import annotations

import collections
from collections import deque
from typing import List, Tuple

import numpy as np


def _most_frequent_color(grid: np.ndarray) -> int:
    values, counts = np.unique(grid, return_counts=True)
    return int(values[np.argmax(counts)])


def _connected_components(mask: np.ndarray) -> List[np.ndarray]:
    h, w = mask.shape
    visited = np.zeros_like(mask, dtype=bool)
    comps: List[np.ndarray] = []
    dirs = ((1, 0), (-1, 0), (0, 1), (0, -1))

    for r0 in range(h):
        row = mask[r0]
        for c0 in range(w):
            if not row[c0] or visited[r0, c0]:
                continue
            q = deque([(r0, c0)])
            visited[r0, c0] = True
            cells: List[Tuple[int, int]] = []
            while q:
                r, c = q.popleft()
                cells.append((r, c))
                for dr, dc in dirs:
                    nr, nc = r + dr, c + dc
                    if (
                        0 <= nr < h
                        and 0 <= nc < w
                        and mask[nr, nc]
                        and not visited[nr, nc]
                    ):
                        visited[nr, nc] = True
                        q.append((nr, nc))
            comps.append(np.array(cells, dtype=int))
    return comps


def _is_border_ring(comp: np.ndarray) -> Tuple[bool, Tuple[int, int, int, int]]:
    if comp.size == 0:
        return False, (0, 0, 0, 0)
    rs = comp[:, 0]
    cs = comp[:, 1]
    r0, r1 = int(rs.min()), int(rs.max())
    c0, c1 = int(cs.min()), int(cs.max())

    h = r1 - r0 + 1
    w = c1 - c0 + 1
    if h < 3 or w < 3:
        return False, (0, 0, 0, 0)

    cell_set = {tuple(v) for v in comp}
    if len(cell_set) != comp.shape[0]:
        return False, (0, 0, 0, 0)

    expected = 2 * (h + w) - 4
    if comp.shape[0] != expected:
        return False, (0, 0, 0, 0)

    # every cell must lie on the border of the bbox
    for r, c in cell_set:
        if r not in (r0, r1) and c not in (c0, c1):
            return False, (0, 0, 0, 0)

    # border must be complete
    for c in range(c0, c1 + 1):
        if (r0, c) not in cell_set or (r1, c) not in cell_set:
            return False, (0, 0, 0, 0)
    for r in range(r0, r1 + 1):
        if (r, c0) not in cell_set or (r, c1) not in cell_set:
            return False, (0, 0, 0, 0)

    return True, (r0, c0, r1, c1)


def _detect_frames(grid: np.ndarray, bg: int):
    colors, counts = np.unique(grid, return_counts=True)
    candidates = []
    for color in colors:
        if int(color) == bg:
            continue
        mask = grid == color
        if not mask.any():
            continue
        comps = _connected_components(mask)
        rects = []
        for comp in comps:
            ok, rect = _is_border_ring(comp)
            if ok:
                rects.append(rect)
        if rects:
            candidates.append((color, rects))

    if not candidates:
        return None, []

    # Prefer color with most ring components; tie-break by total ring area.
    best = max(
        candidates,
        key=lambda item: (len(item[1]), sum((r1 - r0 + 1) * (c1 - c0 + 1) for r0, c0, r1, c1 in item[1])),
    )
    return int(best[0]), list(best[1])


def _longest_non_bg_segment(values: np.ndarray, bg: int):
    n = values.shape[0]
    best = None
    i = 0
    while i < n:
        if values[i] == bg:
            i += 1
            continue
        j = i
        while j < n and values[j] != bg:
            j += 1
        if j - i >= 1 and (best is None or (j - i) > best[0]):
            best = (j - i, i, j)
        i = j
    return best


def _extract_palette_bar(grid: np.ndarray, bg: int):
    h, w = grid.shape
    bottom = _longest_non_bg_segment(grid[h - 1, :], bg)
    right = _longest_non_bg_segment(grid[:, w - 1], bg)

    candidates = []
    if bottom is not None:
        length, s, e = bottom
        if length > 1:
            colors = grid[h - 1, s:e].copy()
            mask = np.zeros_like(grid, dtype=bool)
            mask[h - 1, s:e] = True
            candidates.append((length, "bottom", colors, mask))

    if right is not None:
        length, s, e = right
        if length > 1:
            colors = grid[s:e, w - 1].copy()
            mask = np.zeros_like(grid, dtype=bool)
            mask[s:e, w - 1] = True
            candidates.append((length, "right", colors, mask))

    if not candidates:
        return np.zeros_like(grid, dtype=bool), np.array([], dtype=int)

    # Prefer longer sequence segment; if tie, use bottom before right.
    length, side, colors, mask = max(candidates, key=lambda x: (x[0], 1 if x[1] == "bottom" else 0))
    return mask, colors.astype(int)


def main(grid: np.ndarray) -> np.ndarray:
    g = np.array(grid, dtype=int)
    h, w = g.shape
    bg = _most_frequent_color(g)

    output = g.copy()

    frame_color, frame_rects = _detect_frames(g, bg)
    if not frame_rects:
        # No valid frame structure found.
        bar_mask, colors = _extract_palette_bar(g, bg)
        output[bar_mask] = bg
        return output

    # Sort outermost to innermost by area.
    frame_rects = sorted(frame_rects, key=lambda rc: (rc[2] - rc[0] + 1) * (rc[3] - rc[1] + 1), reverse=True)

    bar_mask, palette = _extract_palette_bar(g, bg)
    if palette.size == 0:
        # Unknown sequence; keep all except erased bar area.
        output[bar_mask] = bg
        return output

    # Pixel depth = number of containing frame interiors.
    depth = np.zeros((h, w), dtype=int)
    frame_interior_mask = np.zeros((h, w), dtype=bool)
    for r0, c0, r1, c1 in frame_rects:
        if r1 - r0 > 1 and c1 - c0 > 1:
            depth[r0 + 1 : r1, c0 + 1 : c1] += 1
            frame_interior_mask[r0 + 1 : r1, c0 + 1 : c1] = True

    # remove bar first
    output[bar_mask] = bg

    sprite_mask = (g != bg) & (g != frame_color) & (~bar_mask) & (~frame_interior_mask)
    # In case sprite has interior touching frame, allow by only excluding frame lines and bar pixels.
    # Connected sprites are fully recolored according to layer depth.
    if frame_color is None:
        return output

    sprite_mask = (g != bg) & (g != frame_color) & (~bar_mask)
    comps = _connected_components(sprite_mask)
    if not comps:
        return output

    max_layer = len(palette)

    for comp in comps:
        if comp.size == 0:
            continue
        r, c = comp[0]
        layer = int(depth[r, c]) + 1
        if layer < 1:
            layer = 1
        if layer > max_layer:
            # malformed but valid generator guarantees this does not happen.
            layer = max_layer
        target = int(palette[layer - 1])
        output[comp[:, 0], comp[:, 1]] = target

    return output


def _random_blob(area_mask: np.ndarray, occupied: np.ndarray, rng: np.random.Generator, target_size: int):
    choices = np.argwhere(area_mask & (~occupied))
    if choices.size == 0:
        return np.empty((0, 2), dtype=int)

    idx = int(rng.integers(len(choices)))
    r, c = map(int, choices[idx])

    selected = [(r, c)]
    in_sel = np.zeros_like(area_mask, dtype=bool)
    in_sel[r, c] = True
    dirs = np.array([[1, 0], [-1, 0], [0, 1], [0, -1]], dtype=int)

    while len(selected) < target_size:
        seed = int(rng.integers(len(selected)))
        sr, sc = selected[seed]
        neighbors = []
        for dr, dc in dirs:
            nr, nc = sr + dr, sc + dc
            if 0 <= nr < area_mask.shape[0] and 0 <= nc < area_mask.shape[1]:
                if area_mask[nr, nc] and (not occupied[nr, nc]) and (not in_sel[nr, nc]):
                    neighbors.append((nr, nc))
        if not neighbors:
            # pick another grown cell next loop
            if not any(area_mask[r_, c_] and (not occupied[r_, c_]) and (not in_sel[r_, c_]) for r_, c_ in selected):
                break
            # randomly jump to an unfilled candidate in area if all chosen-frontier cells stuck
            all_free = np.argwhere(area_mask & (~occupied) & (~in_sel))
            if all_free.size == 0:
                break
            jump = int(rng.integers(len(all_free)))
            nr, nc = map(int, all_free[jump])
            selected.append((nr, nc))
            in_sel[nr, nc] = True
            continue
        nr, nc = neighbors[int(rng.integers(len(neighbors)))]
        selected.append((nr, nc))
        in_sel[nr, nc] = True

    if not selected:
        return np.empty((0, 2), dtype=int)
    return np.array(selected, dtype=int)


def generate_input() -> np.ndarray:
    rng = np.random.default_rng()

    bg = 0
    frame_color = 5
    sprite_palette = np.array([1, 2, 3, 4, 6, 7, 8, 9], dtype=int)

    for _ in range(200):
        orientation = int(rng.integers(0, 2))  # 0: bottom bar, 1: right bar
        frame_count = int(rng.integers(2, 5))  # number of nested frames
        seq_len = frame_count + 1

        if orientation == 0:
            h_min = 12
            w_min = seq_len + 6
            h = int(rng.integers(h_min, 30))
            w = int(rng.integers(w_min, 30))
            inner_bottom_max = h - 2  # keep one row for the bar
            inner_right_max = w - 1
        else:
            h_min = seq_len + 6
            w_min = 12
            h = int(rng.integers(h_min, 30))
            w = int(rng.integers(12, 30))
            inner_right_max = w - 2  # keep one col for the bar
            inner_bottom_max = h - 1

        if h < 8 or w < 8:
            continue
        if inner_bottom_max < 4 or inner_right_max < 4:
            continue

        # Outer frame size constraints for full nesting.
        top0 = int(rng.integers(1, min(4, h - 3)))
        left0 = int(rng.integers(1, min(4, w - 3)))

        min_bottom = top0 + 2 * frame_count + 1
        min_right = left0 + 2 * frame_count + 1

        if min_bottom > inner_bottom_max or min_right > inner_right_max:
            continue

        bottom0 = int(rng.integers(min_bottom, inner_bottom_max + 1))
        right0 = int(rng.integers(min_right, inner_right_max + 1))

        if (bottom0 - top0 + 1) < 2 * frame_count + 1 or (right0 - left0 + 1) < 2 * frame_count + 1:
            continue

        # Build nested frames by shrinking by 1..2 each side, leaving enough room.
        rects = []
        top, left, bottom, right = top0, left0, bottom0, right0
        valid = True

        for idx in range(frame_count):
            if top >= bottom or left >= right:
                valid = False
                break
            rects.append((top, left, bottom, right))

            if idx == frame_count - 1:
                break

            remaining = frame_count - 1 - idx
            h_room = bottom - top + 1
            w_room = right - left + 1
            min_h_needed = 2 * remaining + 1
            min_w_needed = 2 * remaining + 1

            chosen = False
            for _ in range(40):
                dt = int(rng.integers(1, 3))
                db = int(rng.integers(1, 3))
                dl = int(rng.integers(1, 3))
                dr = int(rng.integers(1, 3))

                nt = top + dt
                nb = bottom - db
                nl = left + dl
                nr = right - dr

                if nt < nb and nl < nr:
                    if nb - nt + 1 >= min_h_needed and nr - nl + 1 >= min_w_needed:
                        top, left, bottom, right = nt, nl, nb, nr
                        chosen = True
                        break

            if not chosen:
                valid = False
                break

        if not valid:
            continue

        sprite_color = int(rng.choice(sprite_palette[sprite_palette != frame_color]))
        available_seq = [c for c in sprite_palette if c not in (frame_color, sprite_color, bg)]
        if len(available_seq) < seq_len:
            continue
        palette = rng.choice(available_seq, size=seq_len, replace=False)

        grid = np.full((h, w), bg, dtype=int)

        # Draw nested frames.
        frame_mask = np.zeros((h, w), dtype=bool)
        for r0, c0, r1, c1 in rects:
            grid[r0, c0 : c1 + 1] = frame_color
            grid[r1, c0 : c1 + 1] = frame_color
            grid[r0 : r1 + 1, c0] = frame_color
            grid[r0 : r1 + 1, c1] = frame_color
            frame_mask[r0, c0 : c1 + 1] = True
            frame_mask[r1, c0 : c1 + 1] = True
            frame_mask[r0 : r1 + 1, c0] = True
            frame_mask[r0 : r1 + 1, c1] = True

        # Depth map: number of frame interiors containing each cell.
        depth = np.zeros((h, w), dtype=int)
        for r0, c0, r1, c1 in rects:
            depth[r0 + 1 : r1, c0 + 1 : c1] += 1

        occupied = frame_mask.copy()

        # Place one or more blobs in each layer.
        for layer in range(len(rects) + 1):
            layer_mask = depth == layer
            layer_mask = layer_mask & (~frame_mask)

            # keep layer non-empty enough
            available = np.count_nonzero(layer_mask)
            if available < 2:
                continue

            n_sprites = int(rng.integers(1, 3))
            for _ in range(n_sprites):
                attempts = 0
                placed = False
                layer_available = np.count_nonzero(layer_mask & (~occupied))
                if layer_available < 4:
                    break

                target = int(rng.integers(3, 12))
                target = min(target, layer_available)

                while attempts < 30 and not placed:
                    attempts += 1
                    blob = _random_blob(layer_mask, occupied, rng, target)
                    if blob.size == 0:
                        break
                    if blob.shape[0] < 3:
                        continue
                    grid[blob[:, 0], blob[:, 1]] = sprite_color
                    for r, c in blob:
                        occupied[r, c] = True
                    placed = True

        # Place the color sequence on border.
        if orientation == 0:
            y = h - 1
            start = int(rng.integers(1, max(2, w - seq_len - 1)))
            # ensure room and avoid touching frame.
            if start + seq_len > w - 1:
                continue
            grid[y, start : start + seq_len] = palette
            # Keep the bar non-interfering with the outer frame.
            if w > 2 and start <= 1:
                continue
        else:
            x = w - 1
            start = int(rng.integers(1, max(2, h - seq_len - 1)))
            if start + seq_len > h - 1:
                continue
            grid[start : start + seq_len, x] = palette
            if h > 2 and start <= 1:
                continue

        # Simple visibility guarantee: at least one recolorable pixel.
        if np.count_nonzero((grid != bg) & (grid != frame_color)) == 0:
            continue

        if grid.shape[0] > 30 or grid.shape[1] > 30:
            continue

        return grid

    # Very last-resort fallback for extreme rng failures.
    # Keeps constraints valid and deterministic enough for smoke tests.
    h = w = 20
    grid = np.zeros((h, w), dtype=int)
    grid[1:9, 1:17] = 5
    grid[1, 1:17] = 5
    grid[8, 1:17] = 5
    grid[1:9, 1] = 5
    grid[1:9, 16] = 5
    grid[1:3, 4:6] = 1
    return grid


if __name__ == "__main__":
    sample = generate_input()
    result = main(sample)
    print("input:")
    print(sample)
    print("output:")
    print(result)
