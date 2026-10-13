import random
from collections import deque

import numpy as np


BACKGROUND = 5
PALETTE = [1, 2, 3, 4, 6, 7, 8, 9]
GEN_ATTEMPTS = 500
DIR_TO_DELTA = {
    "right": (0, 1),
    "left": (0, -1),
    "down": (1, 0),
    "up": (-1, 0),
}
ROTATE_FROM_RIGHT = {"right": 0, "up": 1, "left": 2, "down": 3}


def _dominant_color(grid: np.ndarray) -> int:
    values, counts = np.unique(grid, return_counts=True)
    return int(values[np.argmax(counts)])


def _neighbors4(r: int, c: int) -> tuple[tuple[int, int], ...]:
    return ((r - 1, c), (r + 1, c), (r, c - 1), (r, c + 1))


def _find_components(grid: np.ndarray, bg: int) -> list[dict[str, object]]:
    h, w = grid.shape
    seen = np.zeros((h, w), dtype=bool)
    comps: list[dict[str, object]] = []
    for r in range(h):
        for c in range(w):
            if seen[r, c] or int(grid[r, c]) == bg:
                continue
            color = int(grid[r, c])
            q = deque([(r, c)])
            seen[r, c] = True
            coords: list[tuple[int, int]] = []
            while q:
                cr, cc = q.popleft()
                coords.append((cr, cc))
                for nr, nc in _neighbors4(cr, cc):
                    if 0 <= nr < h and 0 <= nc < w and not seen[nr, nc] and int(grid[nr, nc]) == color:
                        seen[nr, nc] = True
                        q.append((nr, nc))
            arr = np.array(coords, dtype=int)
            comps.append({"color": color, "coords": arr, "area": int(len(coords))})
    return comps


def _arrow_template(direction: str, shaft_len: int) -> tuple[set[tuple[int, int]], tuple[int, int]]:
    if direction == "right":
        coords = {(1, c) for c in range(shaft_len)}
        coords |= {(0, shaft_len - 1), (2, shaft_len - 1), (1, shaft_len)}
        return coords, (1, shaft_len)
    if direction == "left":
        coords = {(1, c) for c in range(1, shaft_len + 1)}
        coords |= {(0, 1), (2, 1), (1, 0)}
        return coords, (1, 0)
    if direction == "down":
        coords = {(r, 1) for r in range(shaft_len)}
        coords |= {(shaft_len - 1, 0), (shaft_len - 1, 2), (shaft_len, 1)}
        return coords, (shaft_len, 1)
    if direction == "up":
        coords = {(r, 1) for r in range(1, shaft_len + 1)}
        coords |= {(1, 0), (1, 2), (0, 1)}
        return coords, (0, 1)
    raise ValueError(direction)


def _match_arrow(coords: np.ndarray) -> dict[str, object] | None:
    area = int(coords.shape[0])
    shaft_len = area - 3
    if shaft_len < 1:
        return None
    r0 = int(coords[:, 0].min())
    c0 = int(coords[:, 1].min())
    norm = {(int(r - r0), int(c - c0)) for r, c in coords}
    matches = []
    for direction in DIR_TO_DELTA:
        templ, tip = _arrow_template(direction, shaft_len)
        if norm == templ:
            matches.append((direction, tip))
    if len(matches) != 1:
        return None
    direction, tip = matches[0]
    return {
        "direction": direction,
        "tip": (r0 + int(tip[0]), c0 + int(tip[1])),
        "size": area,
        "coords": coords,
    }


def _find_arrows(grid: np.ndarray, bg: int) -> list[dict[str, object]]:
    arrows: list[dict[str, object]] = []
    for comp in _find_components(grid, bg):
        match = _match_arrow(comp["coords"])
        if match is None:
            continue
        arrows.append(
            {
                "color": int(comp["color"]),
                "coords": comp["coords"],
                "tip": match["tip"],
                "direction": match["direction"],
                "size": int(match["size"]),
            }
        )
    arrows.sort(key=lambda item: (int(item["tip"][0]), int(item["tip"][1])))
    return arrows


def main(grid: np.ndarray) -> np.ndarray:
    g = np.asarray(grid, dtype=int)
    bg = _dominant_color(g)
    out = g.copy()
    arrows = _find_arrows(g, bg)

    for arrow in arrows:
        tip_r, tip_c = map(int, arrow["tip"])
        dr, dc = DIR_TO_DELTA[str(arrow["direction"])]
        for step in range(1, int(arrow["size"]) + 1):
            nr = tip_r + dr * step
            nc = tip_c + dc * step
            if not (0 <= nr < g.shape[0] and 0 <= nc < g.shape[1]):
                break
            if int(out[nr, nc]) != bg:
                out[nr, nc] = int(out[nr, nc])
                break
            out[nr, nc] = int(arrow["color"])
    return out


def _paint_arrow(grid: np.ndarray, top: int, left: int, direction: str, shaft_len: int, color: int) -> tuple[int, int]:
    coords, tip = _arrow_template(direction, shaft_len)
    for r, c in coords:
        grid[top + r, left + c] = color
    return top + int(tip[0]), left + int(tip[1])


def _make_free_patch(direction: str, arrow_color: int) -> np.ndarray:
    shaft_len = random.randint(2, 6)
    ray_len = shaft_len + 3
    extra = random.randint(0, 2)
    patch = np.full((5, shaft_len + ray_len + 3 + extra), BACKGROUND, dtype=int)
    _paint_arrow(patch, 1, 1, "right", shaft_len, arrow_color)
    return np.rot90(patch, ROTATE_FROM_RIGHT[direction]).copy()


def _make_block_patch(direction: str, arrow_color: int, obstacle_color: int) -> np.ndarray:
    shaft_len = random.randint(2, 6)
    ray_len = shaft_len + 3
    if random.random() < 0.2:
        dist = 1
    else:
        dist = random.randint(2, ray_len)
    rect_h = random.randint(1, 3)
    rect_w = random.randint(1, 3)
    if rect_h == 1 and rect_w == 1:
        rect_w = 2
    row_offset = random.randint(0, rect_h - 1)
    extra = random.randint(0, 2)

    patch_h = rect_h + 5
    patch_w = shaft_len + dist + rect_w + 3 + extra
    patch = np.full((patch_h, patch_w), BACKGROUND, dtype=int)

    tip_row = 2 + row_offset
    arrow_top = tip_row - 1
    _paint_arrow(patch, arrow_top, 1, "right", shaft_len, arrow_color)

    obstacle_left = 1 + shaft_len + dist
    patch[2 : 2 + rect_h, obstacle_left : obstacle_left + rect_w] = obstacle_color
    return np.rot90(patch, ROTATE_FROM_RIGHT[direction]).copy()


def _make_ray_hit_patch(first_color: int, second_color: int) -> np.ndarray:
    shaft_a = random.randint(2, 5)
    shaft_b = random.randint(2, 5)
    ray_a = shaft_a + 3
    ray_b = shaft_b + 3
    dist_a = random.randint(2, ray_a)
    dist_b = random.randint(2, ray_b)
    extra_h = random.randint(0, 2)
    extra_w = random.randint(0, 2)

    tip_a_row = 1 + shaft_a
    cross_row = tip_a_row + dist_a
    tip_b_col = 1 + shaft_b
    cross_col = tip_b_col + dist_b

    height = tip_a_row + ray_a + 2 + extra_h
    width = max(tip_b_col + ray_b + 2, cross_col + 3) + extra_w
    patch = np.full((height, width), BACKGROUND, dtype=int)

    _paint_arrow(patch, 1, cross_col - 1, "down", shaft_a, first_color)
    _paint_arrow(patch, cross_row - 1, 1, "right", shaft_b, second_color)

    if random.random() < 0.5:
        patch = np.fliplr(patch).copy()
    return patch


def _place_patch(grid: np.ndarray, used: np.ndarray, patch: np.ndarray) -> bool:
    h, w = grid.shape
    ph, pw = patch.shape
    if ph >= h - 1 or pw >= w - 1:
        return False
    for _ in range(60):
        top = random.randint(1, h - ph - 1)
        left = random.randint(1, w - pw - 1)
        r0 = top - 1
        r1 = top + ph + 1
        c0 = left - 1
        c1 = left + pw + 1
        if used[r0:r1, c0:c1].any():
            continue
        mask = patch != BACKGROUND
        if np.any(grid[top : top + ph, left : left + pw][mask] != BACKGROUND):
            continue
        grid[top : top + ph, left : left + pw][mask] = patch[mask]
        used[r0:r1, c0:c1] = True
        return True
    return False


def _sample_patch_plan() -> list[tuple[str, str | None, int]]:
    include_ray = random.random() < 0.55
    singles = random.randint(1, 3) if include_ray else random.randint(2, 4)

    plan: list[tuple[str, str | None, int]] = []
    if include_ray:
        plan.append(("ray", None, 2))

    kinds = [random.choice(["free", "block"]) for _ in range(singles)]
    if "block" not in kinds:
        kinds[random.randrange(singles)] = "block"
    if not include_ray and "free" not in kinds:
        kinds[random.randrange(singles)] = "free"

    directions = list(DIR_TO_DELTA)
    for kind in kinds:
        plan.append((kind, random.choice(directions), 1 if kind == "free" else 2))
    return plan


def _gen_attempt() -> np.ndarray | None:
    plan = _sample_patch_plan()
    colors_needed = sum(cost for _kind, _direction, cost in plan)
    if colors_needed > len(PALETTE):
        return None
    color_pool = random.sample(PALETTE, colors_needed)
    patches: list[np.ndarray] = []
    arrow_count = 0
    obstacle_count = 0
    idx = 0
    for kind, direction, cost in plan:
        if kind == "ray":
            patches.append(_make_ray_hit_patch(color_pool[idx], color_pool[idx + 1]))
            idx += 2
            arrow_count += 2
        elif kind == "free":
            patches.append(_make_free_patch(str(direction), color_pool[idx]))
            idx += 1
            arrow_count += 1
        else:
            patches.append(_make_block_patch(str(direction), color_pool[idx], color_pool[idx + 1]))
            idx += 2
            arrow_count += 1
            obstacle_count += 1

    max_h = max(p.shape[0] for p in patches)
    max_w = max(p.shape[1] for p in patches)
    total_h = sum(p.shape[0] for p in patches) + len(patches) + 4
    total_w = sum(p.shape[1] for p in patches) + len(patches) + 4
    h_low = max(18, max_h + 3, min(total_h, 30) - random.randint(0, 2))
    w_low = max(18, max_w + 3, min(total_w, 30) - random.randint(0, 2))
    h = random.randint(min(h_low, 30), 30)
    w = random.randint(min(w_low, 30), 30)
    grid = np.full((h, w), BACKGROUND, dtype=int)
    used = np.zeros((h, w), dtype=bool)

    random.shuffle(patches)
    for patch in patches:
        if not _place_patch(grid, used, patch):
            return None

    if int(np.count_nonzero(grid != BACKGROUND)) >= (h * w) // 2:
        return None

    arrows = _find_arrows(grid, BACKGROUND)
    if len(arrows) != arrow_count:
        return None

    comps = _find_components(grid, BACKGROUND)
    if len(comps) < arrow_count + obstacle_count:
        return None

    try:
        out = main(grid)
    except Exception:
        return None

    if out.shape != grid.shape:
        return None
    if np.array_equal(out, grid):
        return None
    return grid


def generate_input() -> np.ndarray:
    for _ in range(GEN_ATTEMPTS):
        grid = _gen_attempt()
        if grid is not None:
            return grid
    raise RuntimeError(f"failed to generate a valid input after {GEN_ATTEMPTS} attempts")
