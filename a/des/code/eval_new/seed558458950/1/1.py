import random
from collections import defaultdict, deque
from typing import Dict, List, Optional, Sequence, Set, Tuple

import numpy as np


def _most_common_color(grid: np.ndarray) -> int:
    values, counts = np.unique(grid, return_counts=True)
    return int(values[int(np.argmax(counts))])


def _run_blocks(values: Sequence[int], bg: int) -> List[Tuple[int, List[int]]]:
    """Return consecutive non-bg color runs as (start_index, colors)."""
    runs: List[Tuple[int, List[int]]] = []
    n = len(values)
    i = 0
    while i < n:
        if int(values[i]) == bg:
            i += 1
            continue
        start = i
        colors: List[int] = []
        while i < n and int(values[i]) != bg:
            colors.append(int(values[i]))
            i += 1
        runs.append((start, colors))
    return runs


def _best_legend_run(runs: List[Tuple[int, List[int]]]) -> Optional[Tuple[int, List[int]]]:
    if not runs:
        return None
    unique_runs = [run for run in runs if len(run[1]) == len(set(run[1]))]
    candidates = unique_runs if unique_runs else runs
    return max(candidates, key=lambda r: len(r[1]))


def _extract_legend(grid: np.ndarray, bg: int) -> Tuple[List[int], Set[Tuple[int, int]]]:
    h, w = grid.shape
    right_values = [int(grid[r, w - 1]) for r in range(h)]
    bottom_values = [int(grid[h - 1, c]) for c in range(w)]

    right_runs = _best_legend_run(_run_blocks(right_values, bg))
    bottom_runs = _best_legend_run(_run_blocks(bottom_values, bg))

    right_len = len(right_runs[1]) if right_runs else 0
    bottom_len = len(bottom_runs[1]) if bottom_runs else 0

    if right_len == 0 and bottom_len == 0:
        return [], set()

    use_right = right_len > bottom_len
    if right_len == bottom_len:
        use_right = right_runs is not None

    if use_right:
        start, colors = right_runs  # type: ignore[assignment]
        return colors, {(start + i, w - 1) for i in range(len(colors))}
    else:
        start, colors = bottom_runs  # type: ignore[assignment]
        return colors, {(h - 1, start + i) for i in range(len(colors))}


def _is_frame(cells: List[Tuple[int, int]], r0: int, r2: int, c0: int, c2: int) -> bool:
    h = r2 - r0 + 1
    w = c2 - c0 + 1
    if h < 2 or w < 2:
        return False
    if len(cells) != 2 * (h + w) - 4:
        return False

    cell_set = set(cells)
    for r in range(r0, r2 + 1):
        if (r, c0) not in cell_set or (r, c2) not in cell_set:
            return False
    for c in range(c0, c2 + 1):
        if (r0, c) not in cell_set or (r2, c) not in cell_set:
            return False

    for r, c in cells:
        if r not in (r0, r2) and c not in (c0, c2):
            return False
    return True


def _collect_frames(
    grid: np.ndarray, bg: int, ignore: Set[Tuple[int, int]]
) -> Dict[int, List[Dict[str, int]]]:
    h, w = grid.shape
    visited = np.zeros_like(grid, dtype=bool)
    frames = defaultdict(list)

    for r in range(h):
        for c in range(w):
            if (r, c) in ignore or visited[r, c] or int(grid[r, c]) == bg:
                continue

            color = int(grid[r, c])
            q = deque([(r, c)])
            visited[r, c] = True
            cells: List[Tuple[int, int]] = [(r, c)]

            while q:
                cr, cc = q.popleft()
                for nr, nc in ((cr - 1, cc), (cr + 1, cc), (cr, cc - 1), (cr, cc + 1)):
                    if not (0 <= nr < h and 0 <= nc < w):
                        continue
                    if visited[nr, nc] or (nr, nc) in ignore:
                        continue
                    if int(grid[nr, nc]) != color:
                        continue
                    visited[nr, nc] = True
                    q.append((nr, nc))
                    cells.append((nr, nc))

            rs = [p[0] for p in cells]
            cs = [p[1] for p in cells]
            r0, r1 = min(rs), max(rs)
            c0, c1 = min(cs), max(cs)
            if not _is_frame(cells, r0, r1, c0, c1):
                continue

            frames[color].append(
                {
                    "color": color,
                    "r0": r0,
                    "r1": r1,
                    "c0": c0,
                    "c1": c1,
                    "area": len(cells),
                }
            )

    return frames


def _inside(inner: Dict[str, int], outer: Dict[str, int]) -> bool:
    return (
        inner["r0"] > outer["r0"]
        and inner["r1"] < outer["r1"]
        and inner["c0"] > outer["c0"]
        and inner["c1"] < outer["c1"]
    )


def _group_nested_frames(frames_by_color: Dict[int, List[Dict[str, int]]]) -> List[List[Dict[str, int]]]:
    groups: List[List[Dict[str, int]]] = []

    for _, frame_list in frames_by_color.items():
        ordered = sorted(frame_list, key=lambda f: f["area"], reverse=True)
        used: Set[int] = set()

        for i, outer in enumerate(ordered):
            if i in used:
                continue
            used.add(i)
            chain = [outer]
            current = outer

            while True:
                candidates = [
                    (j, f)
                    for j, f in enumerate(ordered)
                    if j not in used and _inside(f, current)
                ]
                if not candidates:
                    break
                j, nxt = max(candidates, key=lambda x: x[1]["area"])
                used.add(j)
                chain.append(nxt)
                current = nxt

            groups.append(chain)

    return groups


def _fill_gap(
    grid: np.ndarray,
    outer: Dict[str, int],
    inner: Dict[str, int],
    bg: int,
    color: int,
) -> None:
    for rr in range(outer["r0"] + 1, outer["r1"]):
        for cc in range(outer["c0"] + 1, outer["c1"]):
            if inner["r0"] <= rr <= inner["r1"] and inner["c0"] <= cc <= inner["c1"]:
                continue
            if int(grid[rr, cc]) == bg:
                grid[rr, cc] = color


def _draw_frame(grid: np.ndarray, r0: int, r1: int, c0: int, c1: int, color: int) -> None:
    for r in range(r0, r1 + 1):
        grid[r, c0] = color
        grid[r, c1] = color
    for c in range(c0 + 1, c1):
        grid[r0, c] = color
        grid[r1, c] = color


def _bbox_free(
    occupied: np.ndarray,
    r0: int,
    c0: int,
    r1: int,
    c1: int,
    margin: int = 1,
) -> bool:
    h, w = occupied.shape
    rr0 = max(0, r0 - margin)
    rr1 = min(h, r1 + margin + 1)
    cc0 = max(0, c0 - margin)
    cc1 = min(w, c1 + margin + 1)
    return not np.any(occupied[rr0:rr1, cc0:cc1])


def generate_input() -> np.ndarray:
    """Generate one valid ARC input for nested-frame fill-by-legend instances."""
    palette_pool = [1, 2, 3, 4, 5, 6, 7, 8, 9]
    bg = 0

    for _ in range(200):
        h = int(np.random.randint(16, 31))
        w = int(np.random.randint(16, 31))

        side = int(np.random.randint(2))  # 0=right, 1=bottom
        max_len = h if side == 0 else w
        max_legend = min(6, max_len - 1)
        if max_legend < 2:
            continue

        legend_len = int(np.random.randint(2, max_legend + 1))
        grid = np.full((h, w), bg, dtype=np.int64)
        occupied = np.zeros_like(grid, dtype=bool)

        palette_colors = random.sample(palette_pool, k=legend_len)

        if side == 0:
            start = int(np.random.randint(0, h - legend_len + 1))
            for i, color in enumerate(palette_colors):
                rr = start + i
                grid[rr, w - 1] = color
                occupied[rr, w - 1] = True
            frame_rows = h
            frame_cols = w - 1
        else:
            start = int(np.random.randint(0, w - legend_len + 1))
            for i, color in enumerate(palette_colors):
                cc = start + i
                grid[h - 1, cc] = color
                occupied[h - 1, cc] = True
            frame_rows = h - 1
            frame_cols = w

        avail_colors = [c for c in palette_pool if c not in palette_colors]
        if len(avail_colors) < 3:
            avail_colors = palette_pool.copy()

        num_groups = int(np.random.randint(1, min(4, legend_len, 4) + 1))
        placed_groups = 0
        placed_layers: List[int] = []
        all_ok = True

        for _g in range(num_groups):
            if not avail_colors:
                break

            color = random.choice(avail_colors)
            max_n_h = max(1, (frame_rows - 3) // 2)
            max_n_w = max(1, (frame_cols - 3) // 2)
            max_layers = max(1, min(max_n_h, max_n_w, legend_len))
            if max_layers < 1:
                all_ok = False
                break

            layers = int(np.random.randint(1, max_layers + 1))
            step = int(np.random.randint(1, 3))

            max_layers = max(1, min(max_layers, (frame_rows - 3) // (2 * step) + 1, (frame_cols - 3) // (2 * step) + 1, legend_len))
            if max_layers < layers:
                layers = int(np.random.randint(1, max_layers + 1))

            min_side = 3 + 2 * step * (layers - 1)
            if min_side > frame_rows or min_side > frame_cols:
                all_ok = False
                break

            outer_h = int(np.random.randint(min_side, frame_rows + 1))
            outer_w = int(np.random.randint(min_side, frame_cols + 1))

            placed = False
            for _ in range(400):
                r0 = int(np.random.randint(0, frame_rows - outer_h + 1))
                c0 = int(np.random.randint(0, frame_cols - outer_w + 1))
                r1 = r0 + outer_h - 1
                c1 = c0 + outer_w - 1
                if not _bbox_free(occupied, r0, c0, r1, c1, margin=1):
                    continue

                cr0, cr1, cc0, cc1 = r0, r1, c0, c1
                ok = True
                for _layer in range(layers):
                    if cr0 + 1 > cr1 - 1 or cc0 + 1 > cc1 - 1:
                        ok = False
                        break
                    _draw_frame(grid, cr0, cr1, cc0, cc1, color)
                    if _layer < layers - 1:
                        cr0 += step
                        cr1 -= step
                        cc0 += step
                        cc1 -= step
                if not ok:
                    continue

                cr0, cr1, cc0, cc1 = r0, r1, c0, c1
                for _layer in range(layers):
                    for rr in range(cr0, cr1 + 1):
                        occupied[rr, cc0] = True
                        occupied[rr, cc1] = True
                    for cc in range(cc0 + 1, cc1):
                        occupied[cr0, cc] = True
                        occupied[cr1, cc] = True
                    if _layer < layers - 1:
                        cr0 += step
                        cr1 -= step
                        cc0 += step
                        cc1 -= step

                placed = True
                placed_groups += 1
                placed_layers.append(layers)
                break

            if not placed:
                all_ok = False
                break

        if not all_ok or placed_groups == 0:
            continue

        if all(l == 1 for l in placed_layers) and legend_len >= 2:
            # keep at least one group with multiple layers for visible behavior
            continue

        return grid

    raise RuntimeError("generate_input failed to place valid nested-frame sample")


def main(grid: np.ndarray) -> np.ndarray:
    """Repaint only the gaps between nested rectangular frame layers according to legend index.
Input structure:
- The most frequent color is background.
- On the right or bottom edge sits a 1-cell-wide color legend list, one entry per nesting depth (top->down or left->right).
- The main area contains independent groups of concentric rectangular frame outlines, each group drawn in a single color.
- Frames are separated by background gaps; only nested-frame gaps are to be recolored.
Rule:
- Find background and parse the edge legend to get the depth->color mapping (depth 1 at legend first entry).
- Detect all frame components in the grid (full 1-cell-thick rectangular borders) excluding legend cells.
- Group each color into nested sequences by strict containment and count each group's layer number N.
- Fill every background annulus between layer N and N-1 for that group with legend[N-1], keeping frame pixels and legend unchanged.
- Frames that are single-layer (N=1) are not changed.
Key insight:
- The mapping key is not a geometric position of a frame but the number of nested layers, so counting depth first is required before coloring gaps."""

    out = np.array(grid, copy=True)
    bg = _most_common_color(grid)
    legend_colors, legend_cells = _extract_legend(grid, bg)
    if not legend_colors:
        return out

    frames_by_color = _collect_frames(grid, bg, legend_cells)
    if not frames_by_color:
        return out

    groups = _group_nested_frames(frames_by_color)
    if not groups:
        return out

    for group in groups:
        layers = len(group)
        if layers <= 1 or layers > len(legend_colors):
            continue

        fill_color = legend_colors[layers - 1]
        for i in range(layers - 1):
            outer = group[i]
            inner = group[i + 1]
            _fill_gap(out, outer, inner, bg, fill_color)

    return out
