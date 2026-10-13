import numpy as np


def _most_frequent_color(grid: np.ndarray) -> int:
    vals, counts = np.unique(grid, return_counts=True)
    return int(vals[int(np.argmax(counts))])


def _connected_components(mask: np.ndarray):
    h, w = mask.shape
    seen = np.zeros((h, w), dtype=bool)
    comps = []

    for r0 in range(h):
        for c0 in range(w):
            if not mask[r0, c0] or seen[r0, c0]:
                continue
            stack = [(r0, c0)]
            seen[r0, c0] = True
            pts = []
            while stack:
                r, c = stack.pop()
                pts.append((r, c))
                if r > 0 and mask[r - 1, c] and not seen[r - 1, c]:
                    seen[r - 1, c] = True
                    stack.append((r - 1, c))
                if r + 1 < h and mask[r + 1, c] and not seen[r + 1, c]:
                    seen[r + 1, c] = True
                    stack.append((r + 1, c))
                if c > 0 and mask[r, c - 1] and not seen[r, c - 1]:
                    seen[r, c - 1] = True
                    stack.append((r, c - 1))
                if c + 1 < w and mask[r, c + 1] and not seen[r, c + 1]:
                    seen[r, c + 1] = True
                    stack.append((r, c + 1))
            comps.append(np.array(pts, dtype=int))
    return comps


def _rect_like(comp: np.ndarray) -> bool:
    rmin, cmin = comp.min(axis=0)
    rmax, cmax = comp.max(axis=0)
    area = (rmax - rmin + 1) * (cmax - cmin + 1)
    return area == comp.shape[0]


def _is_terminal_block(comp: np.ndarray) -> bool:
    if comp.shape[0] != 2:
        return False
    rmin, cmin = comp.min(axis=0)
    rmax, cmax = comp.max(axis=0)
    if not _rect_like(comp):
        return False
    return (rmax - rmin == 1 and cmax - cmin == 0) or (cmax - cmin == 1 and rmax - rmin == 0)


def _axis_orientation(mask: np.ndarray):
    h_pairs = int(np.sum(mask[:, :-1] & mask[:, 1:]))
    v_pairs = int(np.sum(mask[:-1, :] & mask[1:, :]))
    return "h" if h_pairs >= v_pairs else "v"


def _axis_position(mask: np.ndarray, orient: str):
    h, w = mask.shape
    if orient == "h":
        row_scores = np.zeros(h, dtype=int)
        for r in range(h):
            for c in range(w):
                if not mask[r, c]:
                    continue
                if (c > 0 and mask[r, c - 1]) or (c + 1 < w and mask[r, c + 1]):
                    row_scores[r] += 1
        if row_scores.max() == 0:
            counts = mask.sum(axis=1)
            if counts.max() == 0:
                return None
            return int(np.argmax(counts))
        return int(np.argmax(row_scores))

    col_scores = np.zeros(w, dtype=int)
    for c in range(w):
        for r in range(h):
            if not mask[r, c]:
                continue
            if (r > 0 and mask[r - 1, c]) or (r + 1 < h and mask[r + 1, c]):
                col_scores[c] += 1
    if col_scores.max() == 0:
        counts = mask.sum(axis=0)
        if counts.max() == 0:
            return None
        return int(np.argmax(counts))
    return int(np.argmax(col_scores))


def _find_terminal_pair(mask: np.ndarray, orient: str, axis_pos: int):
    h, w = mask.shape
    coords = set(zip(*np.where(mask)))

    best = []
    if orient == "h":
        for r, c in zip(*np.where(mask)):
            if c + 1 >= w or (r, c + 1) not in coords:
                continue
            if r != axis_pos:
                continue
            left = (r, c - 1) in coords if c - 1 >= 0 else False
            right = (r, c + 2) in coords if c + 2 < w else False
            if left ^ right:
                direction = 1 if left else -1
                score = c if direction == -1 else (w - (c + 2))
                best.append((score, r, c, direction))
    else:
        for r, c in zip(*np.where(mask)):
            if r + 1 >= h or (r + 1, c) not in coords:
                continue
            if c != axis_pos:
                continue
            up = (r - 1, c) in coords if r - 1 >= 0 else False
            down = (r + 2, c) in coords if r + 2 < h else False
            if up ^ down:
                direction = 1 if up else -1
                score = r if direction == -1 else (h - (r + 2))
                best.append((score, r, c, direction))

    if not best:
        # fallback: ignore axis filter
        if orient == "h":
            for r, c in zip(*np.where(mask)):
                if c + 1 >= w or (r, c + 1) not in coords:
                    continue
                left = (r, c - 1) in coords if c - 1 >= 0 else False
                right = (r, c + 2) in coords if c + 2 < w else False
                if left ^ right:
                    direction = 1 if left else -1
                    score = c if direction == -1 else (w - (c + 2))
                    best.append((score, r, c, direction))
        else:
            for r, c in zip(*np.where(mask)):
                if r + 1 >= h or (r + 1, c) not in coords:
                    continue
                up = (r - 1, c) in coords if r - 1 >= 0 else False
                down = (r + 2, c) in coords if r + 2 < h else False
                if up ^ down:
                    direction = 1 if up else -1
                    score = r if direction == -1 else (h - (r + 2))
                    best.append((score, r, c, direction))

    if not best:
        return None

    _, r, c, direction = max(best, key=lambda x: x[0])
    if orient == "h":
        cells = [(r, c), (r, c + 1)]
    else:
        cells = [(r, c), (r + 1, c)]
    return {
        "cells": cells,
        "direction": direction,
        "axis_line": axis_pos,
    }


def _pick_line_component(comps, orient, axis_pos):
    best_i = None
    best_score = -10**9
    for i, comp in enumerate(comps):
        if comp.shape[0] < 2:
            continue
        rmin, cmin = comp.min(axis=0)
        rmax, cmax = comp.max(axis=0)
        span_axis = (cmax - cmin) if orient == "h" else (rmax - rmin)
        span_orth = (rmax - rmin) if orient == "h" else (cmax - cmin)
        if span_axis < 2:
            continue
        if orient == "h":
            axis_hit = int(np.sum(comp[:, 0] == axis_pos))
        else:
            axis_hit = int(np.sum(comp[:, 1] == axis_pos))
        score = span_axis * 3 - span_orth * 2 + axis_hit * 2
        if score > best_score:
            best_score = score
            best_i = i

    return best_i


def _interval_non_overlap(intervals, a0, a1, gap=1):
    for b0, b1 in intervals:
        if not (a1 + gap < b0 or b1 + gap < a0):
            return False
    return True


def main(grid: np.ndarray) -> np.ndarray:
    """Slide same-color fragments along each color rail to the same-color terminal block and keep only the stacked result and terminal blocks.

Input structure:
- Grid background is dominant and may contain several single-pixel-wide orthogonal rails, each in a unique foreground color.
- Each rail has a same-color 2x1 terminal block marking its motion end and several same-color rectangular fragments placed on/near that rail.
- Fragments are solid rectangles with small width/height and do not overlap the same-color rail in the final stacking result.

Rule:
- For each foreground color, identify its rail orientation, terminal block, and motion direction toward the terminal.
- Extract all fragment components, sort fragments by distance to the terminal so the nearest fragment moves first.
- Move each fragment along the rail axis to pack them against the terminal block (or already stacked fragments), preserving each fragment's original shape.
- Erase all rail pixels that are not occupied by final fragments or the terminal block; only terminal blocks and final stacked fragments remain.

Key insight:
- Rail geometry is the only path guide: detect the rail axis and terminal 2-cell marker, then apply a strict nearest-first projection along that axis while keeping shape rigidity.
"""
    g = np.array(grid, copy=False)
    h, w = g.shape
    bg = _most_frequent_color(g)
    out = np.full_like(g, bg)
    colors = [int(c) for c in np.unique(g) if int(c) != bg]

    for color in colors:
        mask = g == color
        if not np.any(mask):
            continue

        comps = _connected_components(mask)
        if not comps:
            continue

        orient = _axis_orientation(mask)
        axis_pos = _axis_position(mask, orient)
        if axis_pos is None:
            continue

        term = _find_terminal_pair(mask, orient, axis_pos)
        terminal_cells = []

        if term is not None:
            terminal_cells = term["cells"]
            direction = term["direction"]
            term_set = set(terminal_cells)
        else:
            # Fallback: keep the most plausible 2x1 block so output still visually stable.
            fallback = None
            for comp in comps:
                if _is_terminal_block(comp):
                    fallback = comp
                    break
            if fallback is not None:
                terminal_cells = [tuple(p) for p in fallback]
                term_set = set(terminal_cells)
                if orient == "h":
                    cmin, cmax = fallback[:, 1].min(), fallback[:, 1].max()
                    direction = 1 if fallback.shape[0] > 0 and fallback[0, 0] % 2 == 0 else -1
                else:
                    rmin, rmax = fallback[:, 0].min(), fallback[:, 0].max()
                    direction = 1 if rmin % 2 == 0 else -1
            else:
                continue

        line_idx = _pick_line_component(comps, orient, axis_pos)

        frags = []
        for i, comp in enumerate(comps):
            if term is not None and set(map(tuple, comp)) == set(map(tuple, term["cells"])):
                continue
            if i == line_idx:
                continue
            frags.append(comp)

        if not frags:
            # if no fragments detected, preserve only terminal block
            for r, c in terminal_cells:
                if 0 <= r < h and 0 <= c < w:
                    out[r, c] = color
            continue

        if orient == "h":
            if direction == 1:
                term_row = terminal_cells[0][0]
                term_col = min(c for _, c in terminal_cells)
                place_pos = term_col - 1
                frags_sorted = sorted(frags, key=lambda comp: int(comp[:, 1].max()), reverse=True)
                for comp in frags_sorted:
                    cols = comp[:, 1]
                    shift = place_pos - int(cols.max())
                    moved = comp.copy()
                    moved[:, 1] += shift
                    if moved[:, 1].min() < 0 or moved[:, 1].max() >= w:
                        continue
                    if np.any((moved[:, 1] >= 0) & (moved[:, 1] < w) == False):
                        continue
                    for r, c in moved:
                        out[r, c] = color
                    place_pos = int(moved[:, 1].min()) - 1
            else:
                term_col = max(c for _, c in terminal_cells)
                place_pos = term_col + 1
                frags_sorted = sorted(frags, key=lambda comp: int(comp[:, 1].min()), reverse=False)
                for comp in frags_sorted:
                    cols = comp[:, 1]
                    shift = place_pos - int(cols.min())
                    moved = comp.copy()
                    moved[:, 1] += shift
                    if moved[:, 1].min() < 0 or moved[:, 1].max() >= w:
                        continue
                    if np.any((moved[:, 1] >= 0) & (moved[:, 1] < w) == False):
                        continue
                    for r, c in moved:
                        out[r, c] = color
                    place_pos = int(moved[:, 1].max()) + 1
        else:
            if direction == 1:
                term_row = min(r for r, _ in terminal_cells)
                place_pos = term_row - 1
                frags_sorted = sorted(frags, key=lambda comp: int(comp[:, 0].max()), reverse=True)
                for comp in frags_sorted:
                    rows = comp[:, 0]
                    shift = place_pos - int(rows.max())
                    moved = comp.copy()
                    moved[:, 0] += shift
                    if moved[:, 0].min() < 0 or moved[:, 0].max() >= h:
                        continue
                    if np.any((moved[:, 0] >= 0) & (moved[:, 0] < h) == False):
                        continue
                    for r, c in moved:
                        out[r, c] = color
                    place_pos = int(moved[:, 0].min()) - 1
            else:
                term_row = max(r for r, _ in terminal_cells)
                place_pos = term_row + 1
                frags_sorted = sorted(frags, key=lambda comp: int(comp[:, 0].min()), reverse=False)
                for comp in frags_sorted:
                    rows = comp[:, 0]
                    shift = place_pos - int(rows.min())
                    moved = comp.copy()
                    moved[:, 0] += shift
                    if moved[:, 0].min() < 0 or moved[:, 0].max() >= h:
                        continue
                    if np.any((moved[:, 0] >= 0) & (moved[:, 0] < h) == False):
                        continue
                    for r, c in moved:
                        out[r, c] = color
                    place_pos = int(moved[:, 0].max()) + 1

        for r, c in terminal_cells:
            if 0 <= r < h and 0 <= c < w:
                out[r, c] = color

    return out


def generate_input() -> np.ndarray:
    rng = np.random.default_rng()
    h = int(rng.integers(12, 28))
    w = int(rng.integers(12, 28))

    palette = [1, 2, 3, 4, 5, 6, 7, 8, 9, 10]
    bg = int(rng.choice(palette))
    colors = [c for c in palette if c != bg]
    rng.shuffle(colors)

    grid = np.full((h, w), bg, dtype=int)
    occupied = np.zeros((h, w), dtype=bool)

    num_tracks = int(rng.integers(1, min(4, len(colors) + 1)))

    used_rows = []
    used_cols = []

    for tidx in range(num_tracks):
        if not colors:
            break
        color = colors.pop()

        orient = "h" if rng.random() < 0.5 else "v"

        if orient == "h":
            row_candidates = [r for r in range(1, h - 1) if all(abs(r - rr) > 1 for rr in used_rows)]
            if row_candidates:
                row = int(rng.choice(row_candidates))
            else:
                row = int(rng.integers(1, h - 1))
            used_rows.append(row)

            direction = 1 if rng.random() < 0.5 else -1
            if direction == 1:
                term_left = int(rng.integers(4, max(5, w - 1)))
                term_left = min(term_left, w - 3)
                max_start = max(0, term_left - 4)
                start = int(rng.integers(0, max_start + 1))
                end = term_left - 1
            else:
                if w < 9:
                    continue
                term_left = int(rng.integers(1, w - 6))
                end = int(rng.integers(term_left + 5, w))
                start = int(rng.integers(term_left + 2, end - 1))

            if start > end:
                start, end = end, start
                direction *= -1

            for c in range(start, end + 1):
                grid[row, c] = color
                occupied[row, c] = True
            if direction == 1:
                grid[row, term_left : term_left + 2] = color
                occupied[row, term_left : term_left + 2] = True
            else:
                # terminal is on the left of the line
                grid[row, term_left : term_left + 2] = color
                occupied[row, term_left : term_left + 2] = True

            path_cols = list(range(start, end + 1))
            intervals = []
            frag_count = int(rng.integers(1, 1 + min(4, len(path_cols) // 2)))
            for _ in range(frag_count):
                attempts = 0
                while attempts < 160:
                    attempts += 1
                    fw = int(rng.integers(1, 4))
                    fh = int(rng.integers(1, 4))
                    if fw * fh == 2:
                        fw = 3 if fw == 1 else 2
                    if fw > len(path_cols):
                        fw = max(1, len(path_cols) // 2)

                    axis_c = int(rng.integers(start, end - fw + 2))
                    if not _interval_non_overlap(intervals, axis_c, axis_c + fw - 1, gap=1):
                        continue

                    off_opts = []
                    if row + 2 + fh - 1 < h:
                        off_opts.append(2)
                    if row - fh - 1 >= 0:
                        off_opts.append(-2)
                    if not off_opts:
                        if row + 1 + fh - 1 < h:
                            off_opts.append(1)
                        if row - fh >= 0:
                            off_opts.append(-1)
                    if not off_opts:
                        break

                    off = int(rng.choice(off_opts))
                    if off >= 0:
                        sr = row + off
                    else:
                        sr = row + off
                    sc = axis_c
                    if sr < 0 or sr + fh > h or sc < 0 or sc + fw > w:
                        continue
                    if np.any(occupied[sr : sr + fh, sc : sc + fw]):
                        continue

                    grid[sr : sr + fh, sc : sc + fw] = color
                    occupied[sr : sr + fh, sc : sc + fw] = True
                    intervals.append((axis_c, axis_c + fw - 1))
                    break
        else:
            col_candidates = [c for c in range(1, w - 1) if all(abs(c - cc) > 1 for cc in used_cols)]
            if col_candidates:
                col = int(rng.choice(col_candidates))
            else:
                col = int(rng.integers(1, w - 1))
            used_cols.append(col)

            direction = 1 if rng.random() < 0.5 else -1
            if direction == 1:
                term_top = int(rng.integers(4, max(5, h - 1)))
                term_top = min(term_top, h - 3)
                max_start = max(0, term_top - 4)
                start = int(rng.integers(0, max_start + 1))
                end = term_top - 1
            else:
                if h < 9:
                    continue
                term_top = int(rng.integers(1, h - 6))
                end = int(rng.integers(term_top + 5, h))
                start = int(rng.integers(term_top + 2, end - 1))

            if start > end:
                start, end = end, start
                direction *= -1

            for r in range(start, end + 1):
                grid[r, col] = color
                occupied[r, col] = True
            grid[term_top : term_top + 2, col] = color
            occupied[term_top : term_top + 2, col] = True

            path_rows = list(range(start, end + 1))
            intervals = []
            frag_count = int(rng.integers(1, 1 + min(4, len(path_rows) // 2)))
            for _ in range(frag_count):
                attempts = 0
                while attempts < 160:
                    attempts += 1
                    fh = int(rng.integers(1, 4))
                    fw = int(rng.integers(1, 4))
                    if fw * fh == 2:
                        fw = 3 if fw == 1 else 2
                    if fh > len(path_rows):
                        fh = max(1, len(path_rows) // 2)

                    axis_r = int(rng.integers(start, end - fh + 2))
                    if not _interval_non_overlap(intervals, axis_r, axis_r + fh - 1, gap=1):
                        continue

                    off_opts = []
                    if col + 2 + fw - 1 < w:
                        off_opts.append(2)
                    if col - fw - 1 >= 0:
                        off_opts.append(-2)
                    if not off_opts:
                        if col + 1 + fw - 1 < w:
                            off_opts.append(1)
                        if col - fw >= 0:
                            off_opts.append(-1)
                    if not off_opts:
                        break

                    off = int(rng.choice(off_opts))
                    if off >= 0:
                        sc = col + off
                    else:
                        sc = col + off
                    sr = axis_r
                    if sr < 0 or sr + fh > h or sc < 0 or sc + fw > w:
                        continue
                    if np.any(occupied[sr : sr + fh, sc : sc + fw]):
                        continue

                    grid[sr : sr + fh, sc : sc + fw] = color
                    occupied[sr : sr + fh, sc : sc + fw] = True
                    intervals.append((axis_r, axis_r + fh - 1))
                    break

    return grid
