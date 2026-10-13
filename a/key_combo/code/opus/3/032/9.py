from __future__ import annotations

import random
from collections import deque

import numpy as np


BLACK = 0
BLUE = 1
RED = 2
GREEN = 3
YELLOW = 4
GREY = 5
ORANGE = 7

DIRS4 = [(-1, 0), (1, 0), (0, -1), (0, 1)]


def _neighbors4(r: int, c: int):
    for dr, dc in DIRS4:
        yield r + dr, c + dc


def _bbox_coords(coords: np.ndarray) -> tuple[int, int, int, int]:
    r0, c0 = coords.min(axis=0)
    r1, c1 = coords.max(axis=0)
    return int(r0), int(c0), int(r1), int(c1)


def _components(mask: np.ndarray) -> list[np.ndarray]:
    h, w = mask.shape
    seen = np.zeros((h, w), dtype=bool)
    out: list[np.ndarray] = []
    for r in range(h):
        for c in range(w):
            if seen[r, c] or not bool(mask[r, c]):
                continue
            q = deque([(r, c)])
            seen[r, c] = True
            coords: list[tuple[int, int]] = []
            while q:
                rr, cc = q.popleft()
                coords.append((rr, cc))
                for nr, nc in _neighbors4(rr, cc):
                    if 0 <= nr < h and 0 <= nc < w and not seen[nr, nc] and bool(mask[nr, nc]):
                        seen[nr, nc] = True
                        q.append((nr, nc))
            out.append(np.array(coords, dtype=int))
    return out


def _append_segment(path: list[tuple[int, int]], r0: int, c0: int, r1: int, c1: int) -> None:
    if r0 == r1:
        step = 1 if c1 >= c0 else -1
        cells = [(r0, c) for c in range(c0, c1 + step, step)]
    else:
        step = 1 if r1 >= r0 else -1
        cells = [(r, c0) for r in range(r0, r1 + step, step)]
    if path and cells and path[-1] == cells[0]:
        cells = cells[1:]
    path.extend(cells)


def _path_from_profiles(top: list[int], bottom: list[int]) -> list[tuple[int, int]]:
    width = len(top)
    path: list[tuple[int, int]] = []
    _append_segment(path, top[0], 0, top[0], 0)
    for c in range(width - 1):
        _append_segment(path, top[c], c, top[c], c + 1)
        _append_segment(path, top[c], c + 1, top[c + 1], c + 1)
    _append_segment(path, top[-1], width - 1, bottom[-1], width - 1)
    for c in range(width - 1, 0, -1):
        _append_segment(path, bottom[c], c, bottom[c], c - 1)
        _append_segment(path, bottom[c], c - 1, bottom[c - 1], c - 1)
    _append_segment(path, bottom[0], 0, top[0], 0)
    if path and path[-1] == path[0]:
        path.pop()
    return path


def _enclosed_component(patch: np.ndarray) -> np.ndarray | None:
    inside = []
    for coords in _components(patch != RED):
        if np.any(coords[:, 0] == 0) or np.any(coords[:, 0] == patch.shape[0] - 1):
            continue
        if np.any(coords[:, 1] == 0) or np.any(coords[:, 1] == patch.shape[1] - 1):
            continue
        inside.append(coords)
    if len(inside) != 1:
        return None
    return inside[0]


def _make_loop_patch(height: int, width: int, force_irregular: bool) -> tuple[np.ndarray, np.ndarray, bool] | None:
    top = [0] * width
    bottom = [height - 1] * width
    irregular = False

    order = ["top", "bottom"]
    random.shuffle(order)
    for side in order:
        if width < 5 or height < 5:
            continue
        want_notch = force_irregular if not irregular else (random.random() < 0.35)
        if not want_notch:
            continue
        start = random.randint(1, width - 3)
        end = random.randint(start, width - 2)
        if side == "top":
            max_depth = min(bottom[c] - 2 for c in range(start, end + 1))
            if max_depth < 1:
                continue
            depth = random.randint(1, max_depth)
            for c in range(start, end + 1):
                top[c] = depth
        else:
            max_depth = min(height - 3 - top[c] for c in range(start, end + 1))
            if max_depth < 1:
                continue
            depth = random.randint(1, max_depth)
            for c in range(start, end + 1):
                bottom[c] = height - 1 - depth
        irregular = True

    if force_irregular and not irregular:
        return None

    patch = np.zeros((height, width), dtype=int)
    for r, c in _path_from_profiles(top, bottom):
        patch[r, c] = RED
    interior = _enclosed_component(patch)
    if interior is None or interior.shape[0] == 0:
        return None
    return patch, interior, irregular


def _choose_rich_flags(count: int) -> list[bool]:
    rich = [bool(random.getrandbits(1)) for _ in range(count)]
    for parity in (0, 1):
        idxs = [i for i in range(count) if i % 2 == parity]
        if not any(rich[i] for i in idxs):
            rich[random.choice(idxs)] = True
        if all(rich[i] for i in idxs):
            rich[random.choice(idxs)] = False
    return rich


def main(grid: np.ndarray) -> np.ndarray:
    """Fill each red-enclosed region with a color determined by left-to-right parity and marker count.

    Input structure:
    - The grid uses black background, red boundary lines, and blue marker pixels inside enclosed regions.
    - Red 4-connected line loops partition the non-red space into one exterior area and several enclosed regions.
    - Each enclosed region has a unique horizontal bbox center, so left-to-right ordering is unambiguous.

    Rule:
    - Find every 4-connected component of non-red-barrier space using red as the blocking color, and ignore the component that touches the grid border.
    - Sort the enclosed regions by bbox center x from left to right and label them 0, 1, 2, ....
    - Count blue markers in each region; counts of at least three are rich, otherwise poor.
    - Recolor every cell of each enclosed region, including its blue markers, with this matrix: even+rich -> yellow, even+poor -> green, odd+rich -> orange, odd+poor -> grey. Keep red boundaries unchanged.

    Key insight:
    - The output color is a 2x2 decision matrix formed by two independent signals: horizontal order parity and the number of blue markers inside the same enclosed region.
    """
    g = np.asarray(grid, dtype=int)
    out = g.copy()

    regions = []
    for coords in _components(g != RED):
        if np.any(coords[:, 0] == 0) or np.any(coords[:, 0] == g.shape[0] - 1):
            continue
        if np.any(coords[:, 1] == 0) or np.any(coords[:, 1] == g.shape[1] - 1):
            continue
        r0, c0, r1, c1 = _bbox_coords(coords)
        regions.append(
            {
                "coords": coords,
                "center_key": int(c0 + c1),
                "blue_count": int(np.sum(g[coords[:, 0], coords[:, 1]] == BLUE)),
            }
        )

    regions.sort(key=lambda region: region["center_key"])

    for idx, region in enumerate(regions):
        rich = region["blue_count"] >= 3
        if idx % 2 == 0:
            fill = YELLOW if rich else GREEN
        else:
            fill = ORANGE if rich else GREY
        coords = region["coords"]
        out[coords[:, 0], coords[:, 1]] = fill

    return out


def _build_scene() -> np.ndarray | None:
    count = random.randint(6, 8)
    rich_flags = _choose_rich_flags(count)
    row_phase = random.randint(0, 1)

    widths = [5 if random.random() < 0.55 else 3 for _ in range(count)]
    irregular_targets = [i for i, w in enumerate(widths) if w == 5]
    if len(irregular_targets) < 2:
        narrow = [i for i, w in enumerate(widths) if w == 3]
        random.shuffle(narrow)
        for idx in narrow[: 2 - len(irregular_targets)]:
            widths[idx] = 5
        irregular_targets = [i for i, w in enumerate(widths) if w == 5]
    random.shuffle(irregular_targets)
    irregular_targets = set(irregular_targets[: random.randint(2, min(4, len(irregular_targets)))])

    specs = []
    for i in range(count):
        is_rich = rich_flags[i]
        width = widths[i]
        h_lo = 6 if is_rich else 5
        h_hi = 9 if is_rich else 8
        force_irregular = i in irregular_targets

        patch_data = None
        for _ in range(80):
            height = random.randint(h_lo, h_hi)
            patch_data = _make_loop_patch(height, width, force_irregular)
            if patch_data is None:
                continue
            patch, interior, irregular = patch_data
            area = int(interior.shape[0])
            if is_rich and area < 3:
                continue
            patch_data = (patch, interior, irregular)
            break
        if patch_data is None:
            return None
        patch, interior, irregular = patch_data
        specs.append(
            {
                "patch": patch,
                "interior": interior,
                "width": width,
                "height": int(patch.shape[0]),
                "rich": is_rich,
                "row_group": (i + row_phase) % 2,
                "irregular": irregular,
            }
        )

    top_specs = [spec for spec in specs if spec["row_group"] == 0]
    bottom_specs = [spec for spec in specs if spec["row_group"] == 1]
    max_top_h = max(spec["height"] for spec in top_specs)
    max_bottom_h = max(spec["height"] for spec in bottom_specs)

    top_margin = random.randint(1, 2)
    bottom_margin = random.randint(1, 2)
    row_gap = random.randint(2, 4)
    height = top_margin + max_top_h + row_gap + max_bottom_h + bottom_margin
    if height > 30:
        return None

    start_center = random.randint(3, 5)
    step = 3
    centers = [start_center + step * i for i in range(count)]
    width = max(centers) + 4
    if width > 30:
        return None

    grid = np.zeros((height, width), dtype=int)
    top_slack = max_top_h - min(spec["height"] for spec in top_specs)
    bottom_slack = max_bottom_h - min(spec["height"] for spec in bottom_specs)
    top_shift_cap = min(2, top_slack)
    bottom_shift_cap = min(2, bottom_slack)

    for i, spec in enumerate(specs):
        center = centers[i]
        patch = spec["patch"]
        h, w = patch.shape
        c0 = center - w // 2
        if spec["row_group"] == 0:
            r0 = top_margin + random.randint(0, top_shift_cap)
        else:
            r0 = height - bottom_margin - h - random.randint(0, bottom_shift_cap)
        if r0 < 1 or c0 < 1 or r0 + h >= height or c0 + w >= width:
            return None
        sub = grid[r0 : r0 + h, c0 : c0 + w]
        if np.any((sub != BLACK) & (patch != BLACK)):
            return None
        sub[patch != BLACK] = patch[patch != BLACK]

        area = int(spec["interior"].shape[0])
        if spec["rich"]:
            marker_count = random.randint(3, min(5, area))
        else:
            marker_count = random.randint(0, min(2, area))
        marker_cells = random.sample(spec["interior"].tolist(), marker_count)
        for ir, ic in marker_cells:
            grid[r0 + ir, c0 + ic] = BLUE

    if sum(spec["irregular"] for spec in specs) < 2:
        return None

    out = main(grid)
    fills = {GREEN, YELLOW, GREY, ORANGE}
    if not fills.issubset(set(map(int, np.unique(out)))):
        return None
    if np.array_equal(out, grid):
        return None
    return grid


def generate_input() -> np.ndarray:
    for _ in range(400):
        grid = _build_scene()
        if grid is not None:
            return grid
    raise RuntimeError("failed to generate a valid enclosed-region coloring input")
