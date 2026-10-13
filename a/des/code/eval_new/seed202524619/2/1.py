import random
from collections import Counter, deque
from typing import List, Tuple

import numpy as np


def _most_common_color(grid: np.ndarray) -> int:
    vals = [int(v) for v in grid.ravel()]
    cnt = Counter(vals)
    max_count = max(cnt.values())
    for v in vals:
        if cnt[v] == max_count:
            return v
    return int(vals[0])


def _components_by_color(grid: np.ndarray, bg: int) -> List[Tuple[int, List[Tuple[int, int]], Tuple[int, int, int, int]]]:
    h, w = grid.shape
    visited = np.zeros_like(grid, dtype=bool)
    comps: List[Tuple[int, List[Tuple[int, int]], Tuple[int, int, int, int]]] = []
    dirs = [(1, 0), (-1, 0), (0, 1), (0, -1)]

    for r0 in range(h):
        for c0 in range(w):
            if visited[r0, c0] or int(grid[r0, c0]) == bg:
                continue
            color = int(grid[r0, c0])
            q = deque([(r0, c0)])
            visited[r0, c0] = True
            cells: List[Tuple[int, int]] = []
            rmin, rmax = r0, r0
            cmin, cmax = c0, c0

            while q:
                r, c = q.popleft()
                cells.append((r, c))
                if r < rmin:
                    rmin = r
                if r > rmax:
                    rmax = r
                if c < cmin:
                    cmin = c
                if c > cmax:
                    cmax = c

                for dr, dc in dirs:
                    nr, nc = r + dr, c + dc
                    if nr < 0 or nc < 0 or nr >= h or nc >= w:
                        continue
                    if visited[nr, nc] or int(grid[nr, nc]) != color:
                        continue
                    visited[nr, nc] = True
                    q.append((nr, nc))

            comps.append((color, cells, (rmin, rmax, cmin, cmax)))
    return comps


def _extract_frames(components: List[Tuple[int, List[Tuple[int, int]], Tuple[int, int, int, int]]]) -> List[dict]:
    frames: List[dict] = []

    for color, cells, (rmin, rmax, cmin, cmax) in components:
        h = rmax - rmin + 1
        w = cmax - cmin + 1

        if h < 3 or w < 3 or h != w:
            continue

        side = h
        expected_border = 2 * (side + side) - 4
        if len(cells) != expected_border:
            continue

        cell_set = set(cells)
        valid = True

        for r in range(rmin, rmax + 1):
            if (r, cmin) not in cell_set or (r, cmax) not in cell_set:
                valid = False
                break
        if not valid:
            continue

        for c in range(cmin + 1, cmax):
            if (rmin, c) not in cell_set or (rmax, c) not in cell_set:
                valid = False
                break
        if not valid:
            continue

        for r in range(rmin + 1, rmax):
            for c in range(cmin + 1, cmax):
                if (r, c) in cell_set:
                    valid = False
                    break
            if not valid:
                break

        if not valid:
            continue

        frames.append(
            dict(
                border_color=color,
                r0=rmin,
                r1=rmax,
                c0=cmin,
                c1=cmax,
                size=side - 2,
                color=color,
            )
        )

    return frames


def _find_library_blocks(grid: np.ndarray, bg: int, expected_size: int) -> dict:
    h, w = grid.shape
    comps = _components_by_color(grid, bg)
    blocks = {}

    for color, cells, (rmin, rmax, cmin, cmax) in comps:
        if rmin != 0 and cmin != 0 and rmax != h - 1 and cmax != w - 1:
            continue

        hh = rmax - rmin + 1
        ww = cmax - cmin + 1
        if hh != expected_size or ww != expected_size:
            continue
        if len(cells) != expected_size * expected_size:
            continue

        blocks[color] = (rmin, rmax, cmin, cmax)

    return blocks


def main(grid: np.ndarray) -> np.ndarray:
    """Fill missing color layers inside square frames using color chips from the border library.

Input structure:
- Dominant background color (bg).
- Multiple same-sized square frames with uniform border color, each enclosing a square interior.
- In each frame interior, one whole color layer is replaced by bg.
- On the grid edge there is a library of same-size solid-color square chips, one chip per color layer.

Rule:
- Detect square frame borders and identify each frame interior.
- For each frame, determine which color is missing by cross-frame comparison: the color absent from this frame but present in the other frames.
- Find the matching solid chip in the edge library by color.
- Replace only bg pixels inside the frame interior with that chip color.

Key insight:
- The missing colors are inferred only through cross-frame consistency, and the border library supplies the exact single-color patch for each missing color.
"""

    g = np.asarray(grid)
    out = g.copy()
    h, w = out.shape

    bg = _most_common_color(out)
    comps = _components_by_color(out, bg)
    frames = _extract_frames(comps)

    if not frames:
        return out

    # Keep only the common frame size in case of noisy components.
    size_counter = Counter(f["size"] for f in frames)
    major_size = size_counter.most_common(1)[0][0]
    frames = [f for f in frames if f["size"] == major_size]
    if not frames:
        return out

    interior_colors = []
    for f in frames:
        inner = out[f["r0"] + 1 : f["r1"], f["c0"] + 1 : f["c1"]]
        colors = {int(v) for v in inner.ravel() if int(v) != bg}
        interior_colors.append(colors)

    all_colors = set().union(*interior_colors) if interior_colors else set()
    if not all_colors:
        return out

    presence = Counter()
    for colors in interior_colors:
        for c in colors:
            presence[c] += 1

    library_blocks = _find_library_blocks(out, bg, major_size)
    if not library_blocks:
        return out

    n_frames = len(frames)

    for idx, f in enumerate(frames):
        colors = interior_colors[idx]
        missing_candidates = [
            c for c in all_colors
            if c not in colors and presence[c] == n_frames - 1
        ]

        if len(missing_candidates) != 1:
            fallback = [c for c in all_colors if c not in colors and c in library_blocks]
            if not fallback:
                continue
            missing_color = fallback[0]
        else:
            missing_color = missing_candidates[0]

        if missing_color not in library_blocks:
            continue

        patch_r0, patch_r1, patch_c0, patch_c1 = library_blocks[missing_color]
        chip_color = int(out[patch_r0, patch_c0])

        inner = out[f["r0"] + 1 : f["r1"], f["c0"] + 1 : f["c1"]]
        inner[inner == bg] = chip_color

    return out


def _random_pattern(inner: int, colors: List[int], rng: random.Random) -> np.ndarray:
    cells = [(r, c) for r in range(inner) for c in range(inner)]
    rng.shuffle(cells)
    pattern = np.empty((inner, inner), dtype=int)

    for idx, col in enumerate(colors):
        r, c = cells[idx]
        pattern[r, c] = col

    for r, c in cells[len(colors) :]:
        pattern[r, c] = rng.choice(colors)

    return pattern


def _draw_frame(out: np.ndarray, r0: int, c0: int, size: int, border: int, border_inner: np.ndarray) -> None:
    s = size + 2
    r1 = r0 + s
    c1 = c0 + s

    out[r0, c0:c1] = border
    out[r1 - 1, c0:c1] = border
    out[r0:r1, c0] = border
    out[r0:r1, c1 - 1] = border

    inner = out[r0 + 1 : r1 - 1, c0 + 1 : c1 - 1]
    inner[:, :] = border_inner


def generate_input() -> np.ndarray:
    rng = random.Random()

    color_pool = [1, 2, 3, 4, 5, 6, 7, 8, 9]
    bg_color = 0

    for _ in range(400):
        n_frames = rng.randint(2, 4)
        inner = rng.randint(3, 6)
        outer = inner + 2

        if len(color_pool) - 1 < n_frames:
            continue

        border_color = rng.choice([c for c in color_pool if c != bg_color])
        available = [c for c in color_pool if c not in (bg_color, border_color)]
        if len(available) < n_frames:
            continue

        layer_colors = rng.sample(available, n_frames)
        layout_opts = {
            2: [(1, 2), (2, 1)],
            3: [(1, 3), (2, 2)],
            4: [(1, 4), (2, 2)],
        }
        frame_rows, frame_cols = rng.choice(layout_opts[n_frames])
        if frame_rows * frame_cols < n_frames:
            continue

        gap_frames = rng.randint(1, 2)
        frame_grid_w = frame_cols * outer + (frame_cols - 1) * gap_frames
        frame_grid_h = frame_rows * outer + (frame_rows - 1) * gap_frames

        gap_lib = rng.randint(1, 2)
        side = rng.choice(["top", "bottom", "left", "right"])

        if side in ("top", "bottom"):
            lib_w = n_frames * inner + (n_frames - 1) * gap_lib
            lib_h = inner
            req_h = inner + 1 + frame_grid_h + 1
            req_w = max(lib_w, frame_grid_w)
        else:
            lib_w = inner
            lib_h = n_frames * inner + (n_frames - 1) * gap_lib
            req_h = max(lib_h, frame_grid_h)
            req_w = inner + 1 + frame_grid_w + 1

        if req_h > 30 or req_w > 30:
            continue

        h = req_h + rng.randint(0, min(4, 30 - req_h))
        w = req_w + rng.randint(0, min(4, 30 - req_w))

        if h > 30 or w > 30:
            continue

        if side == "top":
            frame_row_min = inner + 1
            frame_row_max = h - frame_grid_h - 1
            if frame_row_min > frame_row_max:
                continue
            frame_r0 = rng.randint(frame_row_min, frame_row_max)

            lib_r0 = 0
            if w < n_frames * inner:
                continue
            lib_c0 = rng.randint(0, w - lib_w)
        elif side == "bottom":
            frame_row_max = h - inner - 1 - frame_grid_h
            if frame_row_max < 0:
                continue
            frame_r0 = rng.randint(0, frame_row_max)

            lib_r0 = h - inner
            lib_c0 = rng.randint(0, w - lib_w)
        elif side == "left":
            frame_col_max = w - frame_grid_w - 1
            if frame_cols < 0 or frame_col_max < inner + 1:
                continue
            frame_c0 = rng.randint(inner + 1, frame_col_max)
            frame_r0 = rng.randint(0, max(0, h - frame_grid_h))

            lib_c0 = 0
            if h < n_frames * inner + (n_frames - 1) * gap_lib:
                continue
            lib_r0 = rng.randint(0, h - lib_h)
        else:
            frame_col_max = w - frame_grid_w - inner - 1
            if frame_col_max < 0:
                continue
            frame_c0 = rng.randint(0, frame_col_max)
            frame_r0 = rng.randint(0, max(0, h - frame_grid_h))

            lib_c0 = w - inner
            if h < n_frames * inner + (n_frames - 1) * gap_lib:
                continue
            lib_r0 = rng.randint(0, h - lib_h)

        # Initialize with background.
        out = np.full((h, w), bg_color, dtype=int)

        if side in ("top", "bottom"):
            frame_c0 = rng.randint(0, w - frame_grid_w)
        # For left/right, frame_c0 set above.

        # Library generation.
        lib_blocks = layer_colors[:]
        rng.shuffle(lib_blocks)
        for i, color in enumerate(lib_blocks):
            if side in ("top", "bottom"):
                c0 = lib_c0 + i * (inner + gap_lib)
                out[lib_r0 : lib_r0 + inner, c0 : c0 + inner] = color
            else:
                r0 = lib_r0 + i * (inner + gap_lib)
                out[r0 : r0 + inner, lib_c0 : lib_c0 + inner] = color

        pattern = _random_pattern(inner, layer_colors, rng)
        frame_slot_order = list(range(n_frames))
        rng.shuffle(frame_slot_order)

        missing_colors = layer_colors[:]
        rng.shuffle(missing_colors)

        for t in range(n_frames):
            sr = frame_slot_order[t]
            r_idx, c_idx = divmod(sr, frame_cols)
            r1 = frame_r0 + r_idx * (outer + gap_frames)
            c1 = frame_c0 + c_idx * (outer + gap_frames)

            if r1 + outer > h or c1 + outer > w:
                continue

            interior = np.where(pattern == missing_colors[t], bg_color, pattern)
            _draw_frame(out, r1, c1, inner, border_color, interior)

        if np.sum(out == bg_color) <= (h * w) // 2:
            continue

        # Ensure all missing colors are unique and correspond to one frame each.
        for idx in range(n_frames):
            _ = missing_colors[idx]

        return out

    # Fallback: simple valid instance.
    out = np.full((25, 25), bg_color, dtype=int)
    inner = 3
    outer = inner + 2
    pattern = np.array(
        [[1, 2, 3],
         [2, 3, 1],
         [3, 1, 2]],
        dtype=int,
    )
    _draw_frame(out, 2, 2, inner, 9, np.array([[1, 2, 0], [2, 3, 1], [3, 1, 2]]))
    _draw_frame(out, 2, 12, inner, 9, np.array([[2, 0, 1], [1, 3, 2], [3, 1, 0]]))
    _draw_frame(out, 10, 2, inner, 9, np.array([[0, 2, 3], [2, 1, 1], [3, 1, 1]]))
    out[:3, :3] = 1
    out[:3, 4:7] = 2
    out[:3, 8:11] = 3
    return out


if __name__ == "__main__":
    for _ in range(3):
        sample = generate_input()
        print(sample.shape)
        print(main(sample))
