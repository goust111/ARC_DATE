from __future__ import annotations

import random

import numpy as np


SIDE_TO_DELTA = {
    "top": (-1, 0),
    "bottom": (1, 0),
    "left": (0, -1),
    "right": (0, 1),
}

GEN_MAX_ATTEMPTS = 400
GEN_FRAME_ATTEMPTS = 250
GEN_GRID_RANGE = (18, 29)
GEN_FRAME_COUNT_CHOICES = (1, 2, 3)
GEN_FRAME_SIZE_RANGE = (5, 10)
GEN_BG_CHOICES = (0, 8)
GEN_FRAME_CHOICES = (5, 6, 7)


def _most_common_color(grid: np.ndarray) -> int:
    values, counts = np.unique(grid, return_counts=True)
    return int(values[np.argmax(counts)])


def _components(mask: np.ndarray) -> list[list[tuple[int, int]]]:
    h, w = mask.shape
    seen = np.zeros((h, w), dtype=bool)
    comps: list[list[tuple[int, int]]] = []
    for r in range(h):
        for c in range(w):
            if not mask[r, c] or seen[r, c]:
                continue
            stack = [(r, c)]
            seen[r, c] = True
            comp: list[tuple[int, int]] = []
            while stack:
                cr, cc = stack.pop()
                comp.append((cr, cc))
                for dr, dc in ((1, 0), (-1, 0), (0, 1), (0, -1)):
                    nr = cr + dr
                    nc = cc + dc
                    if 0 <= nr < h and 0 <= nc < w and mask[nr, nc] and not seen[nr, nc]:
                        seen[nr, nc] = True
                        stack.append((nr, nc))
            comps.append(comp)
    return comps


def _border_mask(height: int, width: int) -> np.ndarray:
    mask = np.zeros((height, width), dtype=bool)
    mask[0, :] = True
    mask[-1, :] = True
    mask[:, 0] = True
    mask[:, -1] = True
    return mask


def _scan_ray(
    h: int,
    w: int,
    start_r: int,
    start_c: int,
    dr: int,
    dc: int,
) -> list[tuple[int, int]]:
    ray: list[tuple[int, int]] = []
    r = start_r
    c = start_c
    while 0 <= r < h and 0 <= c < w:
        ray.append((r, c))
        r += dr
        c += dc
    return ray


def _frame_candidate(
    grid: np.ndarray,
    color: int,
    comp: list[tuple[int, int]],
    bg: int,
) -> dict | None:
    coords = np.array(comp, dtype=int)
    r0, c0 = coords.min(axis=0)
    r1, c1 = coords.max(axis=0)
    h = int(r1 - r0 + 1)
    w = int(c1 - c0 + 1)
    if h < 3 or w < 3:
        return None

    if not np.all((coords[:, 0] == r0) | (coords[:, 0] == r1) | (coords[:, 1] == c0) | (coords[:, 1] == c1)):
        return None

    sub = grid[r0 : r1 + 1, c0 : c1 + 1]
    border = _border_mask(h, w)
    if not np.all(sub[0, 0] == color) or not np.all(sub[0, -1] == color):
        return None
    if not np.all(sub[-1, 0] == color) or not np.all(sub[-1, -1] == color):
        return None
    if not np.all((sub[border] == color) | (sub[border] == bg)):
        return None
    if np.any(sub[1:-1, 1:-1] == color):
        return None

    side_gaps = {
        "top": [int(c0 + c) for c in np.where(sub[0, 1:-1] == bg)[0] + 1],
        "bottom": [int(c0 + c) for c in np.where(sub[-1, 1:-1] == bg)[0] + 1],
        "left": [int(r0 + r) for r in np.where(sub[1:-1, 0] == bg)[0] + 1],
        "right": [int(r0 + r) for r in np.where(sub[1:-1, -1] == bg)[0] + 1],
    }
    if any(len(gaps) > 1 for gaps in side_gaps.values()):
        return None

    gaps: list[dict] = []
    for side, items in side_gaps.items():
        if not items:
            continue
        if side in ("top", "bottom"):
            gaps.append({"side": side, "coord": (r0 if side == "top" else r1, items[0])})
        else:
            gaps.append({"side": side, "coord": (items[0], c0 if side == "left" else c1)})
    if not gaps:
        return None

    interior = sub[1:-1, 1:-1]
    seed_positions = np.argwhere((interior != bg) & (interior != color))
    if seed_positions.shape != (1, 2):
        return None
    sr = int(r0 + 1 + seed_positions[0, 0])
    sc = int(c0 + 1 + seed_positions[0, 1])
    seed_color = int(grid[sr, sc])

    for gap in gaps:
        side = gap["side"]
        gr, gc = gap["coord"]
        if side in ("top", "bottom") and gc != sc:
            return None
        if side in ("left", "right") and gr != sr:
            return None

    return {
        "frame_color": int(color),
        "bbox": (int(r0), int(c0), int(r1), int(c1)),
        "seed": (sr, sc),
        "seed_color": seed_color,
        "gaps": gaps,
        "area": int(len(comp)),
    }


def _find_frames(grid: np.ndarray, bg: int) -> list[dict]:
    frame_by_color: dict[int, list[dict]] = {}
    frame_area: dict[int, int] = {}
    for color in map(int, np.unique(grid)):
        if color == bg:
            continue
        candidates: list[dict] = []
        total_area = 0
        for comp in _components(grid == color):
            candidate = _frame_candidate(grid, color, comp, bg)
            if candidate is None:
                continue
            candidates.append(candidate)
            total_area += candidate["area"]
        if candidates:
            frame_by_color[color] = candidates
            frame_area[color] = total_area

    assert frame_by_color, "expected at least one frame color"
    frame_color = max(frame_area, key=frame_area.get)
    frames = sorted(frame_by_color[frame_color], key=lambda item: item["bbox"])
    assert frames, "expected frame candidates"
    return frames


def main(grid: np.ndarray) -> np.ndarray:
    """
    Grow colored branches from seeds inside hollow frames.

    Input structure
    - The dominant color is the background.
    - One foreground color forms one or more hollow rectangular frames.
    - Each frame contains exactly one colored seed pixel and has one-pixel gaps on some sides.
    - Outside a gap's ray sit same-colored nutrient pixels for that frame's seed.

    Rule
    - Each gap defines a cardinal growth direction.
    - The number of nutrient pixels on the outward ray gives the branch length.
    - Starting from the seed, draw a straight branch of that total length.
    - The branch may pass through a gap, but if it meets a non-gap frame border it stops before that border.
    - Nutrients disappear in the output while frames and seeds stay unchanged.
    """
    g = np.asarray(grid, dtype=int)
    assert g.ndim == 2
    h, w = g.shape
    bg = _most_common_color(g)
    frames = _find_frames(g, bg)

    out = g.copy()
    nutrients_to_clear: set[tuple[int, int]] = set()
    growth_specs: list[tuple[tuple[int, int], int, int, int, int]] = []

    for frame in frames:
        seed_r, seed_c = frame["seed"]
        seed_color = int(frame["seed_color"])
        frame_color = int(frame["frame_color"])
        for gap in frame["gaps"]:
            side = gap["side"]
            gr, gc = gap["coord"]
            dr, dc = SIDE_TO_DELTA[side]
            ray = _scan_ray(h, w, gr + dr, gc + dc, dr, dc)
            nutrient_positions = [(r, c) for r, c in ray if int(g[r, c]) == seed_color]
            nutrients_to_clear.update(nutrient_positions)
            growth_specs.append(((seed_r, seed_c), dr, dc, len(nutrient_positions), frame_color))

    for r, c in nutrients_to_clear:
        out[r, c] = bg

    for (seed_r, seed_c), dr, dc, length, frame_color in growth_specs:
        for step in range(1, length):
            nr = seed_r + dr * step
            nc = seed_c + dc * step
            if not (0 <= nr < h and 0 <= nc < w):
                break
            if int(g[nr, nc]) == frame_color:
                break
            out[nr, nc] = int(g[seed_r, seed_c])

    return out


def _expand_cells(cells: set[tuple[int, int]], h: int, w: int, pad: int = 1) -> set[tuple[int, int]]:
    expanded: set[tuple[int, int]] = set()
    for r, c in cells:
        for dr in range(-pad, pad + 1):
            for dc in range(-pad, pad + 1):
                nr = r + dr
                nc = c + dc
                if 0 <= nr < h and 0 <= nc < w:
                    expanded.add((nr, nc))
    return expanded


def _bbox_cells(r0: int, c0: int, r1: int, c1: int, h: int, w: int, pad: int = 1) -> set[tuple[int, int]]:
    rr0 = max(0, r0 - pad)
    cc0 = max(0, c0 - pad)
    rr1 = min(h - 1, r1 + pad)
    cc1 = min(w - 1, c1 + pad)
    return {(r, c) for r in range(rr0, rr1 + 1) for c in range(cc0, cc1 + 1)}


def _sorted_sample(indices: list[int], count: int) -> list[int]:
    chosen = sorted(random.sample(indices, count))
    return [int(x) for x in chosen]


def _place_one_frame(
    grid: np.ndarray,
    forbidden: np.ndarray,
    bg: int,
    frame_color: int,
    seed_color: int,
) -> bool:
    h, w = grid.shape
    for _ in range(GEN_FRAME_ATTEMPTS):
        sides_pool = ["top", "bottom", "left", "right"]
        side_count = random.randint(1, 4)
        sides = set(random.sample(sides_pool, side_count))

        outer_h = random.randint(*GEN_FRAME_SIZE_RANGE)
        outer_w = random.randint(*GEN_FRAME_SIZE_RANGE)
        if outer_h >= h - 2 or outer_w >= w - 2:
            continue

        r0 = random.randint(1, h - outer_h - 1)
        c0 = random.randint(1, w - outer_w - 1)
        r1 = r0 + outer_h - 1
        c1 = c0 + outer_w - 1

        local_r = random.randint(1, outer_h - 2)
        local_c = random.randint(1, outer_w - 2)
        seed_r = r0 + local_r
        seed_c = c0 + local_c

        gap_coords = {
            "top": (r0, seed_c),
            "bottom": (r1, seed_c),
            "left": (seed_r, c0),
            "right": (seed_r, c1),
        }

        branch_path_outside: set[tuple[int, int]] = set()
        nutrient_cells: set[tuple[int, int]] = set()
        reserved_ray_cells: set[tuple[int, int]] = set()

        valid = True
        for side in sides:
            gr, gc = gap_coords[side]
            dr, dc = SIDE_TO_DELTA[side]
            if side == "top":
                distance_to_gap = local_r
            elif side == "bottom":
                distance_to_gap = outer_h - 1 - local_r
            elif side == "left":
                distance_to_gap = local_c
            else:
                distance_to_gap = outer_w - 1 - local_c

            ray = _scan_ray(h, w, gr + dr, gc + dc, dr, dc)
            if not ray:
                valid = False
                break

            min_n = distance_to_gap + 2
            max_n = len(ray)
            if min_n > max_n:
                valid = False
                break

            nutrient_count = random.randint(min_n, max_n)
            chosen_idx = _sorted_sample(list(range(len(ray))), nutrient_count)
            nutrient_cells.update(ray[i] for i in chosen_idx)

            visible_outside = max(0, nutrient_count - distance_to_gap - 1)
            branch_path_outside.update(ray[:visible_outside])
            reserved_ray_cells.update(ray[: chosen_idx[-1] + 1])

        if not valid:
            continue

        frame_border: set[tuple[int, int]] = set()
        for rr in range(r0, r1 + 1):
            for cc in range(c0, c1 + 1):
                on_border = rr in (r0, r1) or cc in (c0, c1)
                if not on_border:
                    continue
                if (rr, cc) in gap_coords.values() and any((rr, cc) == gap_coords[side] for side in sides):
                    continue
                frame_border.add((rr, cc))

        actual_cells = set(frame_border)
        actual_cells.add((seed_r, seed_c))
        actual_cells.update(nutrient_cells)

        if any(forbidden[r, c] for r, c in actual_cells):
            continue
        if any(int(grid[r, c]) != bg for r, c in actual_cells):
            continue

        reserve_cells = _bbox_cells(r0, c0, r1, c1, h, w, pad=1)
        reserve_cells |= _expand_cells(reserved_ray_cells, h, w, pad=1)
        reserve_cells |= _expand_cells(branch_path_outside, h, w, pad=1)
        if any(forbidden[r, c] for r, c in reserve_cells):
            continue

        for r, c in frame_border:
            grid[r, c] = frame_color
        grid[seed_r, seed_c] = seed_color
        for r, c in nutrient_cells:
            grid[r, c] = seed_color
        for r, c in reserve_cells:
            forbidden[r, c] = True
        return True
    return False


def _generate_attempt() -> np.ndarray | None:
    h = random.randint(*GEN_GRID_RANGE)
    w = random.randint(*GEN_GRID_RANGE)
    bg = random.choice(GEN_BG_CHOICES)
    frame_options = [color for color in GEN_FRAME_CHOICES if color != bg]
    frame_color = random.choice(frame_options)

    seed_palette = [color for color in range(10) if color not in {bg, frame_color}]
    frame_count = random.choice(GEN_FRAME_COUNT_CHOICES)
    if frame_count > len(seed_palette):
        return None
    seed_colors = random.sample(seed_palette, frame_count)

    grid = np.full((h, w), bg, dtype=int)
    forbidden = np.zeros((h, w), dtype=bool)

    for seed_color in seed_colors:
        if not _place_one_frame(grid, forbidden, bg, frame_color, seed_color):
            return None

    try:
        out = main(grid)
    except Exception:
        return None

    if np.array_equal(out, grid):
        return None
    if _most_common_color(grid) != bg:
        return None
    return grid


def generate_input() -> np.ndarray:
    for _ in range(GEN_MAX_ATTEMPTS):
        grid = _generate_attempt()
        if grid is not None:
            return grid
    raise RuntimeError("failed to generate a valid frame-growth puzzle")
