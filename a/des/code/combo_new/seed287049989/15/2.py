from __future__ import annotations

import random
from collections import deque

import numpy as np


BG = 0
DIR4 = [(-1, 0), (1, 0), (0, -1), (0, 1)]
DIR8 = DIR4 + [(-1, -1), (-1, 1), (1, -1), (1, 1)]

ARROW_PATTERNS: dict[frozenset[tuple[int, int]], tuple[tuple[int, int], tuple[int, int]]] = {
    frozenset({(0, 0), (1, 1), (2, 0)}): ((1, 1), (0, 1)),   # >
    frozenset({(0, 1), (1, 0), (2, 1)}): ((1, 0), (0, -1)),  # <
    frozenset({(0, 1), (1, 0), (1, 2)}): ((0, 1), (-1, 0)),  # ^
    frozenset({(0, 0), (0, 2), (1, 1)}): ((1, 1), (1, 0)),   # v
}

GEN_ATTEMPTS = 500
GRID_MIN = 18
GRID_MAX = 30
OBJECT_COUNT_RANGE = (2, 4)
OBJECT_SIZE_RANGE = (8, 12)
OBJECT_BUILD_RETRIES = 250


def _in_bounds(shape: tuple[int, int], r: int, c: int) -> bool:
    return 0 <= r < shape[0] and 0 <= c < shape[1]


def _find_components(
    grid: np.ndarray,
    *,
    connectivity: int,
    bg: int = BG,
    ignore: np.ndarray | None = None,
) -> list[dict]:
    g = np.asarray(grid, dtype=int)
    h, w = g.shape
    seen = np.zeros((h, w), dtype=bool)
    if ignore is None:
        ignore = np.zeros((h, w), dtype=bool)
    dirs = DIR4 if connectivity == 4 else DIR8
    comps: list[dict] = []

    for r in range(h):
        for c in range(w):
            if seen[r, c] or ignore[r, c] or int(g[r, c]) == bg:
                continue
            color = int(g[r, c])
            queue = deque([(r, c)])
            seen[r, c] = True
            coords: list[tuple[int, int]] = []
            while queue:
                cr, cc = queue.popleft()
                coords.append((cr, cc))
                for dr, dc in dirs:
                    nr, nc = cr + dr, cc + dc
                    if not _in_bounds((h, w), nr, nc):
                        continue
                    if seen[nr, nc] or ignore[nr, nc]:
                        continue
                    if int(g[nr, nc]) != color:
                        continue
                    seen[nr, nc] = True
                    queue.append((nr, nc))
            rows = [pt[0] for pt in coords]
            cols = [pt[1] for pt in coords]
            comps.append(
                {
                    "color": color,
                    "coords": coords,
                    "area": len(coords),
                    "bbox": (min(rows), min(cols), max(rows), max(cols)),
                }
            )
    return comps


def _classify_arrow(coords: list[tuple[int, int]]) -> tuple[tuple[int, int], tuple[int, int]] | None:
    if len(coords) != 3:
        return None
    rows = [r for r, _ in coords]
    cols = [c for _, c in coords]
    r0, c0 = min(rows), min(cols)
    rel = frozenset((r - r0, c - c0) for r, c in coords)
    pattern = ARROW_PATTERNS.get(rel)
    if pattern is None:
        return None
    tip_rel, forward = pattern
    return (r0 + tip_rel[0], c0 + tip_rel[1]), forward


def _count_single_pixel_holes(grid: np.ndarray, cells: set[tuple[int, int]], bbox: tuple[int, int, int, int]) -> int:
    r0, c0, r1, c1 = bbox
    holes = 0
    for r in range(r0 + 1, r1):
        for c in range(c0 + 1, c1):
            if int(grid[r, c]) != BG:
                continue
            if all((r + dr, c + dc) in cells for dr, dc in DIR4):
                holes += 1
    return holes


def _extract_scene(grid: np.ndarray) -> tuple[list[dict], list[dict], dict[tuple[int, int], int], list[int]]:
    comps8 = _find_components(grid, connectivity=8)
    arrow_mask = np.zeros_like(grid, dtype=bool)
    arrows: list[dict] = []
    for comp in comps8:
        classified = _classify_arrow(comp["coords"])
        if classified is None:
            continue
        tip, forward = classified
        for r, c in comp["coords"]:
            arrow_mask[r, c] = True
        arrows.append(
            {
                "color": int(comp["color"]),
                "tip": tip,
                "forward": forward,
                "coords": list(comp["coords"]),
            }
        )

    objects = _find_components(grid, connectivity=4, ignore=arrow_mask)
    cell_to_object: dict[tuple[int, int], int] = {}
    hole_counts: list[int] = []
    for idx, obj in enumerate(objects):
        cell_set = set(obj["coords"])
        obj["cell_set"] = cell_set
        for cell in cell_set:
            cell_to_object[cell] = idx
        hole_counts.append(_count_single_pixel_holes(grid, cell_set, obj["bbox"]))

    return arrows, objects, cell_to_object, hole_counts


def _find_target_object(
    grid: np.ndarray,
    tip: tuple[int, int],
    forward: tuple[int, int],
    cell_to_object: dict[tuple[int, int], int],
) -> int:
    h, w = grid.shape
    r, c = tip
    dr, dc = forward
    while True:
        r += dr
        c += dc
        if not (0 <= r < h and 0 <= c < w):
            raise AssertionError("arrow ray left the grid without hitting an object")
        if int(grid[r, c]) == BG:
            continue
        target = cell_to_object.get((r, c))
        if target is not None:
            return target


def main(grid: np.ndarray) -> np.ndarray:
    """
    Extend each arrow backward by the number of isolated single-cell holes inside the object it points at.

    Input structure:
    - The background is black.
    - Foreground includes several monochrome closed objects; each object has an outer boundary and may contain zero or more isolated black single-pixel holes fully 4-surrounded by that object's color.
    - Each arrow is a 3-pixel monochrome marker: one tip pixel plus two diagonal tail-wing pixels, pointing left, right, up, or down.
    - Looking straight ahead from each arrow tip along its pointing direction, the first non-black hit is the edge of exactly one target object.

    Rule:
    - For each arrow, follow its pointing ray to the first object it hits.
    - Count how many black single pixels inside that object's bounding area are 4-neighbor surrounded by the object's color.
    - Starting one cell behind the tip, draw a same-colored straight segment in the opposite direction whose length equals that count.
    - Keep all original objects and the original 3 arrow pixels unchanged; if the count is zero, draw nothing.

    Key insight:
    - The arrow is a query pointer: it reads a structural property of a distant object, then converts that count into a backward geometric extension at the arrow's own tail.
    """
    g = np.asarray(grid, dtype=int)
    assert g.ndim == 2

    arrows, _objects, cell_to_object, hole_counts = _extract_scene(g)
    out = g.copy()

    for arrow in arrows:
        target = _find_target_object(g, arrow["tip"], arrow["forward"], cell_to_object)
        n = hole_counts[target]
        dr, dc = arrow["forward"]
        tip_r, tip_c = arrow["tip"]
        for step in range(1, n + 1):
            rr = tip_r - dr * step
            cc = tip_c - dc * step
            assert _in_bounds(g.shape, rr, cc)
            out[rr, cc] = int(arrow["color"])

    return out


def _color_component_count(patch: np.ndarray, color: int) -> int:
    comps = _find_components(patch, connectivity=4, bg=BG)
    return sum(1 for comp in comps if int(comp["color"]) == color)


def _build_object_patch(height: int, width: int, color: int, hole_count: int, rng: random.Random) -> np.ndarray | None:
    assert height >= 8 and width >= 8
    for _ in range(OBJECT_BUILD_RETRIES):
        patch = np.zeros((height, width), dtype=int)
        patch[0, :] = color
        patch[-1, :] = color
        patch[:, 0] = color
        patch[:, -1] = color

        sides = rng.sample(["N", "S", "W", "E"], hole_count)
        ok = True
        for side in sides:
            placed = False
            for _local_try in range(40):
                if side == "N":
                    c = rng.randint(2, width - 3)
                    r = rng.randint(3, max(3, height // 2 - 1))
                    cells = {
                        (rr, cc)
                        for rr in range(r - 1, r + 2)
                        for cc in range(c - 1, c + 2)
                        if not (rr == r and cc == c)
                    } | {(rr, c) for rr in range(1, r - 1)}
                elif side == "S":
                    c = rng.randint(2, width - 3)
                    r = rng.randint(min(height - 4, (height + 1) // 2), height - 4)
                    cells = {
                        (rr, cc)
                        for rr in range(r - 1, r + 2)
                        for cc in range(c - 1, c + 2)
                        if not (rr == r and cc == c)
                    } | {(rr, c) for rr in range(r + 2, height - 1)}
                elif side == "W":
                    r = rng.randint(2, height - 3)
                    c = rng.randint(3, max(3, width // 2 - 1))
                    cells = {
                        (rr, cc)
                        for rr in range(r - 1, r + 2)
                        for cc in range(c - 1, c + 2)
                        if not (rr == r and cc == c)
                    } | {(r, cc) for cc in range(1, c - 1)}
                else:
                    r = rng.randint(2, height - 3)
                    c = rng.randint(min(width - 4, (width + 1) // 2), width - 4)
                    cells = {
                        (rr, cc)
                        for rr in range(r - 1, r + 2)
                        for cc in range(c - 1, c + 2)
                        if not (rr == r and cc == c)
                    } | {(r, cc) for cc in range(c + 2, width - 1)}
                if any(int(patch[rr, cc]) != BG for rr, cc in cells):
                    continue
                for rr, cc in cells:
                    patch[rr, cc] = color
                placed = True
                break
            if not placed:
                ok = False
                break

        if not ok:
            continue
        if _color_component_count(patch, color) != 1:
            continue
        if _count_single_pixel_holes(patch, {(r, c) for r, c in zip(*np.where(patch == color))}, (0, 0, height - 1, width - 1)) != hole_count:
            continue
        return patch
    return None


def _padded_free(mask: np.ndarray, r0: int, c0: int, h: int, w: int) -> bool:
    rr0 = max(0, r0 - 1)
    cc0 = max(0, c0 - 1)
    rr1 = min(mask.shape[0], r0 + h + 1)
    cc1 = min(mask.shape[1], c0 + w + 1)
    return not bool(mask[rr0:rr1, cc0:cc1].any())


def _place_objects(
    grid_h: int,
    grid_w: int,
    specs: list[dict],
    rng: random.Random,
) -> tuple[np.ndarray, list[dict]] | None:
    grid = np.zeros((grid_h, grid_w), dtype=int)
    blocked = np.zeros((grid_h, grid_w), dtype=bool)
    placed: list[dict] = []

    for spec in sorted(specs, key=lambda item: item["patch"].shape[0] * item["patch"].shape[1], reverse=True):
        patch = spec["patch"]
        ph, pw = patch.shape
        options: list[tuple[int, int]] = []
        for r0 in range(1, grid_h - ph):
            for c0 in range(1, grid_w - pw):
                if _padded_free(blocked, r0, c0, ph, pw):
                    options.append((r0, c0))
        if not options:
            return None
        r0, c0 = rng.choice(options)
        patch_mask = patch != BG
        grid[r0 : r0 + ph, c0 : c0 + pw][patch_mask] = patch[patch_mask]
        blocked[max(0, r0 - 1) : min(grid_h, r0 + ph + 1), max(0, c0 - 1) : min(grid_w, c0 + pw + 1)] = True
        placed.append(
            {
                "color": spec["color"],
                "holes": spec["holes"],
                "patch": patch,
                "arrow_color": spec["arrow_color"],
                "bbox": (r0, c0, r0 + ph - 1, c0 + pw - 1),
            }
        )
    return grid, placed


def _check_clear_row(grid: np.ndarray, r: int, c_start: int, c_stop: int) -> bool:
    if c_start > c_stop:
        return True
    return bool(np.all(grid[r, c_start : c_stop + 1] == BG))


def _check_clear_col(grid: np.ndarray, c: int, r_start: int, r_stop: int) -> bool:
    if r_start > r_stop:
        return True
    return bool(np.all(grid[r_start : r_stop + 1, c] == BG))


def _arrow_candidates(
    grid: np.ndarray,
    bbox: tuple[int, int, int, int],
    line_len: int,
    reserved: np.ndarray,
) -> list[dict]:
    h, w = grid.shape
    r0, c0, r1, c1 = bbox
    options: list[dict] = []

    for r in range(max(1, r0), min(h - 2, r1) + 1):
        for tip_c in range(max(1, line_len), c0):
            tip = (r, tip_c)
            pixels = {(r, tip_c), (r - 1, tip_c - 1), (r + 1, tip_c - 1)}
            tail = {(r, tip_c - step) for step in range(1, line_len + 1)}
            reserve = pixels | tail | {(r, cc) for cc in range(tip_c + 1, c0)}
            if not _check_clear_row(grid, r, 0, c0 - 1):
                continue
            if any(not _in_bounds((h, w), rr, cc) for rr, cc in pixels | tail):
                continue
            if any(int(grid[rr, cc]) != BG for rr, cc in pixels):
                continue
            if any(reserved[rr, cc] for rr, cc in reserve):
                continue
            options.append({"tip": tip, "forward": (0, 1), "pixels": pixels, "reserve": reserve})

    for r in range(max(1, r0), min(h - 2, r1) + 1):
        upper = min(w - 2, w - 1 - line_len)
        for tip_c in range(c1 + 1, upper + 1):
            tip = (r, tip_c)
            pixels = {(r, tip_c), (r - 1, tip_c + 1), (r + 1, tip_c + 1)}
            tail = {(r, tip_c + step) for step in range(1, line_len + 1)}
            reserve = pixels | tail | {(r, cc) for cc in range(c1 + 1, tip_c)}
            if not _check_clear_row(grid, r, c1 + 1, w - 1):
                continue
            if any(not _in_bounds((h, w), rr, cc) for rr, cc in pixels | tail):
                continue
            if any(int(grid[rr, cc]) != BG for rr, cc in pixels):
                continue
            if any(reserved[rr, cc] for rr, cc in reserve):
                continue
            options.append({"tip": tip, "forward": (0, -1), "pixels": pixels, "reserve": reserve})

    for c in range(max(1, c0), min(w - 2, c1) + 1):
        for tip_r in range(max(1, line_len), r0):
            tip = (tip_r, c)
            pixels = {(tip_r, c), (tip_r - 1, c - 1), (tip_r - 1, c + 1)}
            tail = {(tip_r - step, c) for step in range(1, line_len + 1)}
            reserve = pixels | tail | {(rr, c) for rr in range(tip_r + 1, r0)}
            if not _check_clear_col(grid, c, 0, r0 - 1):
                continue
            if any(not _in_bounds((h, w), rr, cc) for rr, cc in pixels | tail):
                continue
            if any(int(grid[rr, cc]) != BG for rr, cc in pixels):
                continue
            if any(reserved[rr, cc] for rr, cc in reserve):
                continue
            options.append({"tip": tip, "forward": (1, 0), "pixels": pixels, "reserve": reserve})

    for c in range(max(1, c0), min(w - 2, c1) + 1):
        upper = min(h - 2, h - 1 - line_len)
        for tip_r in range(r1 + 1, upper + 1):
            tip = (tip_r, c)
            pixels = {(tip_r, c), (tip_r + 1, c - 1), (tip_r + 1, c + 1)}
            tail = {(tip_r + step, c) for step in range(1, line_len + 1)}
            reserve = pixels | tail | {(rr, c) for rr in range(r1 + 1, tip_r)}
            if not _check_clear_col(grid, c, r1 + 1, h - 1):
                continue
            if any(not _in_bounds((h, w), rr, cc) for rr, cc in pixels | tail):
                continue
            if any(int(grid[rr, cc]) != BG for rr, cc in pixels):
                continue
            if any(reserved[rr, cc] for rr, cc in reserve):
                continue
            options.append({"tip": tip, "forward": (-1, 0), "pixels": pixels, "reserve": reserve})

    return options


def _paint_arrow(grid: np.ndarray, arrow: dict, color: int) -> None:
    tip_r, tip_c = arrow["tip"]
    dr, dc = arrow["forward"]
    grid[tip_r, tip_c] = color
    if (dr, dc) == (0, 1):
        wings = [(tip_r - 1, tip_c - 1), (tip_r + 1, tip_c - 1)]
    elif (dr, dc) == (0, -1):
        wings = [(tip_r - 1, tip_c + 1), (tip_r + 1, tip_c + 1)]
    elif (dr, dc) == (1, 0):
        wings = [(tip_r - 1, tip_c - 1), (tip_r - 1, tip_c + 1)]
    else:
        wings = [(tip_r + 1, tip_c - 1), (tip_r + 1, tip_c + 1)]
    for wr, wc in wings:
        grid[wr, wc] = color


def _choose_object_colors(num_objects: int, rng: random.Random) -> list[int]:
    palette = list(range(1, 10))
    rng.shuffle(palette)
    if num_objects >= 3 and rng.random() < 0.4:
        base = palette[: num_objects - 1]
        colors = [rng.choice(base) for _ in range(num_objects)]
        if len(set(colors)) == num_objects:
            colors[-1] = colors[0]
        return colors
    return palette[:num_objects]


def _generate_attempt(rng: random.Random) -> np.ndarray | None:
    num_objects = rng.randint(*OBJECT_COUNT_RANGE)
    object_colors = _choose_object_colors(num_objects, rng)
    used_object_colors = set(object_colors)
    free_colors = [c for c in range(1, 10) if c not in used_object_colors]
    if len(free_colors) < num_objects:
        return None
    rng.shuffle(free_colors)
    arrow_colors = free_colors[:num_objects]

    specs: list[dict] = []
    for idx in range(num_objects):
        holes = rng.randint(0, 4)
        min_size = max(OBJECT_SIZE_RANGE[0], 8 if holes == 0 else 9 if holes <= 2 else 10)
        height = rng.randint(min_size, OBJECT_SIZE_RANGE[1])
        width = rng.randint(min_size, OBJECT_SIZE_RANGE[1])
        patch = _build_object_patch(height, width, object_colors[idx], holes, rng)
        if patch is None:
            return None
        specs.append({"color": object_colors[idx], "holes": holes, "patch": patch, "arrow_color": arrow_colors[idx]})

    grid_h = rng.randint(max(GRID_MIN, 6 + 5 * num_objects), GRID_MAX)
    grid_w = rng.randint(max(GRID_MIN, 6 + 5 * num_objects), GRID_MAX)
    placed_scene = _place_objects(grid_h, grid_w, specs, rng)
    if placed_scene is None:
        return None
    grid, placed = placed_scene

    reserved = np.zeros_like(grid, dtype=bool)
    for idx, obj in enumerate(placed):
        candidates = _arrow_candidates(grid, obj["bbox"], obj["holes"], reserved)
        if not candidates:
            return None
        arrow = rng.choice(candidates)
        _paint_arrow(grid, arrow, obj["arrow_color"])
        for rr, cc in arrow["reserve"]:
            reserved[rr, cc] = True

    return grid


def generate_input() -> np.ndarray:
    for attempt in range(GEN_ATTEMPTS):
        rng = random.Random(random.randrange(1 << 30) ^ attempt)
        grid = _generate_attempt(rng)
        if grid is None:
            continue
        try:
            out = main(grid)
        except Exception:
            continue
        if out.shape != grid.shape:
            continue
        arrows, objects, _mapping, hole_counts = _extract_scene(grid)
        if len(arrows) != len(objects) or not arrows:
            continue
        if any(not (0 <= count <= 4) for count in hole_counts):
            continue
        return grid
    raise RuntimeError(f"generate_input failed after {GEN_ATTEMPTS} attempts")
