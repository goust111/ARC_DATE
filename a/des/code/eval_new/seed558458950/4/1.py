import random
from collections import Counter, deque
from typing import Dict, List

import numpy as np


PATTERN_TEMPLATES = [
    np.array(
        [
            [1, 1, 1],
            [1, 0, 0],
            [1, 0, 0],
        ],
        dtype=bool,
    ),
    np.array(
        [
            [1, 1, 1],
            [0, 1, 0],
            [0, 1, 0],
        ],
        dtype=bool,
    ),
    np.array(
        [
            [0, 1, 0],
            [1, 1, 1],
            [0, 1, 0],
        ],
        dtype=bool,
    ),
    np.array(
        [
            [1, 1, 1, 1, 1],
            [0, 1, 0, 1, 0],
            [0, 1, 0, 1, 0],
            [0, 1, 0, 1, 0],
            [1, 1, 1, 1, 1],
        ],
        dtype=bool,
    ),
    np.array(
        [
            [1, 1, 1, 1],
            [1, 0, 0, 1],
            [1, 1, 1, 0],
            [0, 1, 1, 1],
        ],
        dtype=bool,
    ),
    np.array(
        [
            [1, 0, 1, 0],
            [1, 1, 1, 1],
            [0, 0, 1, 0],
            [1, 1, 1, 1],
        ],
        dtype=bool,
    ),
    np.array(
        [
            [1, 0, 0, 1],
            [1, 1, 1, 1],
            [0, 1, 1, 0],
            [1, 0, 0, 1],
        ],
        dtype=bool,
    ),
]


def _connected_components(grid: np.ndarray, bg: int) -> Dict[int, List[dict]]:
    h, w = grid.shape
    visited = np.zeros((h, w), dtype=bool)
    by_color: Dict[int, List[dict]] = {}

    for r0 in range(h):
        for c0 in range(w):
            if visited[r0, c0] or int(grid[r0, c0]) == bg:
                continue

            color = int(grid[r0, c0])
            queue = deque([(r0, c0)])
            visited[r0, c0] = True
            cells: List[tuple] = []

            while queue:
                r, c = queue.popleft()
                cells.append((r, c))
                nr, nc = r - 1, c
                if nr >= 0 and not visited[nr, nc] and int(grid[nr, nc]) == color:
                    visited[nr, nc] = True
                    queue.append((nr, nc))
                nr, nc = r + 1, c
                if nr < h and not visited[nr, nc] and int(grid[nr, nc]) == color:
                    visited[nr, nc] = True
                    queue.append((nr, nc))
                nr, nc = r, c - 1
                if nc >= 0 and not visited[nr, nc] and int(grid[nr, nc]) == color:
                    visited[nr, nc] = True
                    queue.append((nr, nc))
                nr, nc = r, c + 1
                if nc < w and not visited[nr, nc] and int(grid[nr, nc]) == color:
                    visited[nr, nc] = True
                    queue.append((nr, nc))

            rows = [rr for rr, _ in cells]
            cols = [cc for _, cc in cells]
            comp = {
                "color": color,
                "coords": cells,
                "coord_set": set(cells),
                "rmin": min(rows),
                "rmax": max(rows),
                "cmin": min(cols),
                "cmax": max(cols),
                "h": max(rows) - min(rows) + 1,
                "w": max(cols) - min(cols) + 1,
                "size": len(cells),
            }
            by_color.setdefault(color, []).append(comp)

    return by_color


def _extract_frames(
    grid: np.ndarray,
    bg: int,
    components_by_color: Dict[int, List[dict]],
) -> List[dict]:
    frames: List[dict] = []

    for color, comps in components_by_color.items():
        for comp in comps:
            h = comp["h"]
            w = comp["w"]
            if h < 3 or w < 3:
                continue

            rmin, rmax = comp["rmin"], comp["rmax"]
            cmin, cmax = comp["cmin"], comp["cmax"]
            perimeter_size = 2 * (h + w) - 4
            if comp["size"] != perimeter_size - 1:
                continue

            coords = comp["coord_set"]
            perimeter = []
            for cc in range(cmin, cmax + 1):
                perimeter.append((rmin, cc))
                perimeter.append((rmax, cc))
            for rr in range(rmin + 1, rmax):
                perimeter.append((rr, cmin))
                perimeter.append((rr, cmax))

            missing = [p for p in perimeter if p not in coords]
            if len(missing) != 1:
                continue

            seed_r, seed_c = missing[0]
            seed_color = int(grid[seed_r, seed_c])
            if seed_color == bg or seed_color == color:
                continue

            inside_clear = True
            for rr in range(rmin + 1, rmax):
                for cc in range(cmin + 1, cmax):
                    if int(grid[rr, cc]) != bg:
                        inside_clear = False
                        break
                if not inside_clear:
                    break
            if not inside_clear:
                continue

            if seed_r == rmin:
                side = "top"
            elif seed_r == rmax:
                side = "bottom"
            elif seed_c == cmin:
                side = "left"
            elif seed_c == cmax:
                side = "right"
            else:
                continue

            frames.append(
                {
                    "seed_color": seed_color,
                    "frame_color": color,
                    "seed": (seed_r, seed_c),
                    "side": side,
                    "rmin": rmin,
                    "rmax": rmax,
                    "cmin": cmin,
                    "cmax": cmax,
                }
            )

    return frames


def _component_to_mask(comp: dict) -> np.ndarray:
    mask = np.zeros((comp["h"], comp["w"]), dtype=bool)
    rmin, cmin = comp["rmin"], comp["cmin"]
    for rr, cc in comp["coords"]:
        mask[rr - rmin, cc - cmin] = True
    return mask


def _legend_mask_for_seed(
    seed_color: int,
    components_by_color: Dict[int, List[dict]],
) -> np.ndarray | None:
    candidates = []
    for comp in components_by_color.get(seed_color, []):
        if comp["size"] < 3:
            continue
        if comp["size"] > 25:
            continue
        if comp["h"] < 3 or comp["h"] > 5:
            continue
        if comp["w"] < 3 or comp["w"] > 5:
            continue
        candidates.append(comp)

    if not candidates:
        return None

    best = max(candidates, key=lambda item: item["size"])
    return _component_to_mask(best)


def main(grid: np.ndarray) -> np.ndarray:
    """Grow colored legend motifs from frame-edge seeds into frame interiors.

Input structure:
- background is the most frequent color.
- each task has several hollow rectangular frame borders in solid colors.
- each frame has exactly one border pixel replaced by a different seed color.
- a corner legend contains small solid-color motif components, one per seed color.

Rule:
- find each hollow frame by checking its color ring and single missing border pixel.
- use the seed color at the missing border point to choose the motif in the legend.
- rotate the motif by growth side: top/bottom no rotation, left is clockwise, right is counter-clockwise.
- place the motif in the frame interior anchored at the seed-adjacent interior cell,
  with top or bottom side aligned to the seed row and left or right side aligned to the seed column.
- keep frame borders and legend pixels unchanged.

Key insight:
- seed color links a frame to one legend motif, while seed side links that motif to a deterministic
  placement side/rotation rule.
"""

    grid = np.array(grid, dtype=int)
    flat = [int(v) for v in grid.reshape(-1)]
    bg = int(Counter(flat).most_common(1)[0][0])

    components_by_color = _connected_components(grid, bg)
    frames = _extract_frames(grid, bg, components_by_color)
    out = np.array(grid, copy=True)

    for frame in frames:
        mask = _legend_mask_for_seed(frame["seed_color"], components_by_color)
        if mask is None:
            continue

        shape = np.array(mask, dtype=bool)
        side = frame["side"]
        if side == "left":
            shape = np.rot90(shape, -1)
        elif side == "right":
            shape = np.rot90(shape, 1)

        ph, pw = shape.shape
        rmin, rmax = frame["rmin"], frame["rmax"]
        cmin, cmax = frame["cmin"], frame["cmax"]
        sr, sc = frame["seed"]

        if side == "top":
            base_r = rmin + 1
            base_c = sc
        elif side == "bottom":
            base_r = rmax - ph
            base_c = sc
        elif side == "left":
            base_r = sr
            base_c = cmin + 1
        else:
            base_r = sr
            base_c = cmax - pw

        if base_r < rmin + 1 or base_c < cmin + 1:
            continue
        if base_r + ph > rmax or base_c + pw > cmax:
            continue

        for dr, dc in np.argwhere(shape):
            rr = int(base_r + int(dr))
            cc = int(base_c + int(dc))
            if rmin < rr < rmax and cmin < cc < cmax:
                out[rr, cc] = frame["seed_color"]

    return out


def _area_free(
    occupied: np.ndarray,
    rmin: int,
    rmax: int,
    cmin: int,
    cmax: int,
    margin: int,
) -> bool:
    h, w = occupied.shape
    rr0 = max(0, rmin - margin)
    rr1 = min(h, rmax + margin)
    cc0 = max(0, cmin - margin)
    cc1 = min(w, cmax + margin)
    return not bool(np.any(occupied[rr0:rr1, cc0:cc1]))


def generate_input() -> np.ndarray:
    while True:
        h = random.randint(18, 30)
        w = random.randint(18, 30)
        bg = 0
        grid = np.full((h, w), bg, dtype=int)

        frame_count = random.randint(1, 3)
        palette = list(range(1, 12))
        colors = random.sample(palette, frame_count * 2)
        frame_colors = colors[:frame_count]
        seed_colors = colors[frame_count:]

        selected_templates = random.sample(PATTERN_TEMPLATES, len(PATTERN_TEMPLATES))[:frame_count]

        occupied = np.zeros((h, w), dtype=bool)
        legend_area = 12
        if h < 16 or w < 16:
            continue
        occupied[:legend_area, :legend_area] = True

        slots = [(1, 1), (1, 7), (7, 1), (7, 7)]
        random.shuffle(slots)

        for i in range(frame_count):
            shape = selected_templates[i]
            color = seed_colors[i]
            sr, sc = slots[i]
            ph, pw = shape.shape
            if sr + ph > legend_area or sc + pw > legend_area:
                break
            for dr, dc in np.argwhere(shape):
                rr = int(sr + int(dr))
                cc = int(sc + int(dc))
                grid[rr, cc] = color

        else:
            side_choices = ["top", "bottom", "left", "right"]
            all_placed = True

            for i in range(frame_count):
                frame_color = frame_colors[i]
                seed_color = seed_colors[i]
                template = selected_templates[i]
                side = random.choice(side_choices)

                if side == "left":
                    expected = np.rot90(template, -1)
                elif side == "right":
                    expected = np.rot90(template, 1)
                else:
                    expected = template

                ph, pw = expected.shape
                min_h = max(ph, 3)
                min_w = max(pw, 3)
                max_h = min(14, h - 2)
                max_w = min(14, w - 2)

                success = False
                for _ in range(400):
                    inner_h = random.randint(min_h, min(min_h + 6, max_h))
                    inner_w = random.randint(min_w, min(min_w + 6, max_w))
                    frame_h = inner_h + 2
                    frame_w = inner_w + 2

                    top = random.randint(0, h - frame_h)
                    left = random.randint(0, w - frame_w)
                    if not _area_free(occupied, top, top + frame_h, left, left + frame_w, 1):
                        continue

                    if side in ("top", "bottom"):
                        offset_max = inner_w - pw
                        if offset_max < 0:
                            continue
                        offset = random.randint(0, offset_max)
                        seed_col = left + 1 + offset
                        seed_row = top if side == "top" else (top + frame_h - 1)
                    else:
                        offset_max = inner_h - ph
                        if offset_max < 0:
                            continue
                        offset = random.randint(0, offset_max)
                        seed_row = top + 1 + offset
                        seed_col = left if side == "left" else (left + frame_w - 1)

                    # draw frame boundary
                    for cc in range(left, left + frame_w):
                        grid[top, cc] = frame_color
                        grid[top + frame_h - 1, cc] = frame_color
                    for rr in range(top + 1, top + frame_h - 1):
                        grid[rr, left] = frame_color
                        grid[rr, left + frame_w - 1] = frame_color

                    grid[seed_row, seed_col] = seed_color

                    occupied[top : top + frame_h, left : left + frame_w] = True
                    success = True
                    break

                if not success:
                    all_placed = False
                    break

            if all_placed:
                return grid

        # retry on any placement failure
        continue
