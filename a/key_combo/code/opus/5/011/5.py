from __future__ import annotations

from collections import defaultdict, deque
from pathlib import Path
import random
import sys

import numpy as np

_ARC_UTILS_DIR = Path("/home/lixinyu/snap/new/solvers")
if str(_ARC_UTILS_DIR) not in sys.path:
    sys.path.append(str(_ARC_UTILS_DIR))

from arc_utils import find_components, get_bg, neighbors4, paste, random_free_location


BG = 0
FRAME_COLOR_POOL = [1, 2, 3, 4, 6, 7, 8, 9]
GEN_ATTEMPTS = 400

_SHAPE_MASKS = {
    "block2": np.array([[1, 1], [1, 1]], dtype=bool),
    "rect23": np.array([[1, 1, 1], [1, 1, 1]], dtype=bool),
    "rect32": np.array([[1, 1], [1, 1], [1, 1]], dtype=bool),
    "ring33": np.array([[1, 1, 1], [1, 0, 1], [1, 1, 1]], dtype=bool),
}


def _shape_patch(kind: str, color: int) -> np.ndarray:
    mask = _SHAPE_MASKS[kind]
    patch = np.full(mask.shape, BG, dtype=int)
    patch[mask] = int(color)
    return patch


def _frame_patch(height: int, width: int, color: int) -> np.ndarray:
    patch = np.full((int(height), int(width)), BG, dtype=int)
    patch[0, :] = int(color)
    patch[-1, :] = int(color)
    patch[:, 0] = int(color)
    patch[:, -1] = int(color)
    return patch


def _bbox_size(bbox: tuple[int, int, int, int]) -> tuple[int, int]:
    r0, c0, r1, c1 = bbox
    return int(r1 - r0 + 1), int(c1 - c0 + 1)


def _is_hollow_frame(grid: np.ndarray, comp: dict, bg: int) -> bool:
    color = int(comp["color"])
    r0, c0, r1, c1 = map(int, comp["bbox"])
    height, width = _bbox_size(comp["bbox"])
    if height < 5 or width < 5:
        return False
    border = np.zeros((height, width), dtype=bool)
    border[0, :] = True
    border[-1, :] = True
    border[:, 0] = True
    border[:, -1] = True
    sub = grid[r0 : r1 + 1, c0 : c1 + 1]
    if not np.array_equal(sub == color, border):
        return False
    return bool(np.all(sub[1:-1, 1:-1] == int(bg)))


def _frame_capacity(frame: dict) -> int:
    _h, w = _bbox_size(frame["bbox"])
    interior_width = max(0, w - 2)
    return (interior_width + 1) // 2


def _exterior_background(grid: np.ndarray, bg: int) -> np.ndarray:
    h, w = grid.shape
    seen = np.zeros((h, w), dtype=bool)
    q: deque[tuple[int, int]] = deque()

    for r in range(h):
        for c in (0, w - 1):
            if grid[r, c] == int(bg) and not seen[r, c]:
                seen[r, c] = True
                q.append((int(r), int(c)))
    for c in range(w):
        for r in (0, h - 1):
            if grid[r, c] == int(bg) and not seen[r, c]:
                seen[r, c] = True
                q.append((int(r), int(c)))

    while q:
        r, c = q.popleft()
        for nr, nc in neighbors4(r, c):
            if 0 <= nr < h and 0 <= nc < w and not seen[nr, nc] and grid[nr, nc] == int(bg):
                seen[nr, nc] = True
                q.append((int(nr), int(nc)))
    return seen


def _shape_outer_count(
    grid: np.ndarray,
    coords: np.ndarray,
    bg: int,
    exterior_bg: np.ndarray,
) -> int:
    h, w = grid.shape
    outline: set[tuple[int, int]] = set()
    for r, c in coords:
        for nr, nc in neighbors4(int(r), int(c)):
            if 0 <= nr < h and 0 <= nc < w and grid[nr, nc] == int(bg) and exterior_bg[nr, nc]:
                outline.add((int(nr), int(nc)))
    return int(len(outline))


def _naive_adjacent_bg_count(grid: np.ndarray, coords: np.ndarray, bg: int) -> int:
    h, w = grid.shape
    outline: set[tuple[int, int]] = set()
    for r, c in coords:
        for nr, nc in neighbors4(int(r), int(c)):
            if 0 <= nr < h and 0 <= nc < w and grid[nr, nc] == int(bg):
                outline.add((int(nr), int(nc)))
    return int(len(outline))


def _scene_parts(grid: np.ndarray) -> tuple[int, list[dict], list[dict]]:
    g = np.asarray(grid, dtype=int)
    bg = int(get_bg(g))
    comps = find_components(g, bg=bg, connectivity=4, monochromatic=True)
    tagged = [(comp, _is_hollow_frame(g, comp, bg)) for comp in comps]
    frames = [comp for comp, is_frame in tagged if is_frame]
    shapes = [comp for comp, is_frame in tagged if not is_frame]
    assert 2 <= len(frames) <= 3, "expected 2 or 3 hollow frames"
    assert shapes, "expected contour shapes outside the frames"
    return bg, frames, shapes


def _color_totals(grid: np.ndarray, shapes: list[dict], bg: int) -> dict[int, int]:
    exterior_bg = _exterior_background(grid, bg)
    totals: dict[int, int] = defaultdict(int)
    for comp in shapes:
        totals[int(comp["color"])] += _shape_outer_count(grid, comp["coords"], bg, exterior_bg)
    return totals


def main(grid: np.ndarray) -> np.ndarray:
    """
    Count each small contour's outside-adjacent background pixels and use those totals to draw same-color dots inside matching frames.

    Input structure:
    - Black is the dominant background.
    - There are 2 or 3 large single-color hollow rectangular frames, each with a black interior.
    - Outside the frames are several smaller same-color contour components; each small component belongs to exactly one frame color.

    Rule:
    - For every small contour component, collect the black cells that are 4-neighbors of the contour and belong to the exterior background rather than an internal hole.
    - Sum that outside-contour count per color, then erase all small contour components to black.
    - For each frame, draw that color's total as same-color dots on the frame's interior center row, starting at the rightmost interior column and stepping left by 2; if needed, clip to the frame's row capacity.

    Key insight:
    The traced contour is not the final picture; its outside pixel count becomes a numeric signal that is transferred across same-color matching into dot placement inside the large frame.
    """
    g = np.asarray(grid, dtype=int)
    bg, frames, shapes = _scene_parts(g)
    totals = _color_totals(g, shapes, bg)

    out = g.copy()
    for comp in shapes:
        for r, c in comp["coords"]:
            out[int(r), int(c)] = int(bg)

    for frame in frames:
        color = int(frame["color"])
        r0, c0, r1, c1 = map(int, frame["bbox"])
        center_r = (r0 + r1) // 2
        cols = list(range(c1 - 1, c0, -2))
        for c in cols[: min(len(cols), int(totals.get(color, 0)))]:
            out[center_r, int(c)] = color

    return out


def _count_mask_outer(mask: np.ndarray) -> int:
    grid = np.zeros((mask.shape[0] + 4, mask.shape[1] + 4), dtype=int)
    grid[2 : 2 + mask.shape[0], 2 : 2 + mask.shape[1]][mask] = 1
    comp = {"coords": np.argwhere(grid == 1)}
    exterior = _exterior_background(grid, BG)
    return _shape_outer_count(grid, comp["coords"], BG, exterior)


SHAPE_OUTER_COUNTS = {name: _count_mask_outer(mask) for name, mask in _SHAPE_MASKS.items()}


def _reserve_bbox(mask: np.ndarray, bbox: tuple[int, int, int, int], pad: int = 1) -> None:
    r0, c0, r1, c1 = bbox
    rr0 = max(0, int(r0) - pad)
    cc0 = max(0, int(c0) - pad)
    rr1 = min(mask.shape[0] - 1, int(r1) + pad)
    cc1 = min(mask.shape[1] - 1, int(c1) + pad)
    mask[rr0 : rr1 + 1, cc0 : cc1 + 1] = True


def _colored_from_mask(mask: np.ndarray, color: int) -> np.ndarray:
    patch = np.full(mask.shape, BG, dtype=int)
    patch[mask] = int(color)
    return patch


def _choose_color_plans(n_frames: int) -> list[dict] | None:
    colors = random.sample(FRAME_COLOR_POOL, n_frames)
    ring_idx = random.randrange(n_frames)
    clip_idx = None
    candidates = [i for i in range(n_frames) if i != ring_idx]
    if candidates and random.random() < 0.7:
        clip_idx = random.choice(candidates)

    plans: list[dict] = []
    for i, color in enumerate(colors):
        if i == ring_idx:
            kinds = ["ring33"]
            capacity = 13
        elif i == clip_idx:
            n_shapes = random.choice([2, 2, 3])
            pool = ["block2", "rect23", "rect32", "ring33"]
            kinds = [random.choice(pool) for _ in range(n_shapes)]
            capacity = random.randint(4, 8)
        else:
            kind = random.choice(["block2", "rect23", "rect32", "ring33"])
            kinds = [kind]
            total = SHAPE_OUTER_COUNTS[kind]
            capacity = total + random.choice([0, 0, 1, 2])
        total = sum(SHAPE_OUTER_COUNTS[k] for k in kinds)
        plans.append(
            {
                "color": int(color),
                "kinds": kinds,
                "total": int(total),
                "capacity": int(capacity),
                "drawn": int(min(total, capacity)),
                "has_ring": "ring33" in kinds,
            }
        )

    drawn_counts = {plan["drawn"] for plan in plans}
    if len(drawn_counts) == 1:
        adjustable = [plan for plan in plans if plan["capacity"] > 4]
        if not adjustable:
            return None
        plan = random.choice(adjustable)
        plan["capacity"] = max(4, plan["capacity"] - 1)
        plan["drawn"] = min(plan["total"], plan["capacity"])
    if len({plan["drawn"] for plan in plans}) == 1:
        return None
    if not any(plan["has_ring"] and plan["total"] == 12 and plan["capacity"] >= 13 for plan in plans):
        return None
    if clip_idx is not None and not any(plan["total"] > plan["capacity"] for plan in plans):
        return None
    return plans


def _layout_frames(plans: list[dict]) -> tuple[int, int, list[dict]] | None:
    widths = [2 * int(plan["capacity"]) + 1 for plan in plans]
    heights = [random.choice([5, 7]) for _ in plans]
    gaps = [random.randint(2, 3) for _ in range(max(0, len(plans) - 1))]
    top_margin = random.randint(1, 3)
    bottom_margin = random.randint(1, 3)
    min_h = top_margin + bottom_margin + sum(heights) + sum(gaps)
    min_w = max(max(widths) + 3, 24)
    if min_h > 30 or min_w > 30:
        return None

    h = random.randint(min_h, 30)
    w = random.randint(min_w, 30)
    frames: list[dict] = []
    row = top_margin
    for i, plan in enumerate(plans):
        height = heights[i]
        width = widths[i]
        c0_min = 1
        c0_max = w - width - 1
        if c0_max < c0_min:
            return None
        c0 = random.randint(c0_min, c0_max)
        r0 = row
        bbox = (r0, c0, r0 + height - 1, c0 + width - 1)
        frames.append(
            {
                "color": int(plan["color"]),
                "bbox": bbox,
                "height": int(height),
                "width": int(width),
                "capacity": int(plan["capacity"]),
            }
        )
        row += height
        if i < len(gaps):
            row += gaps[i]
    return h, w, frames


def _place_shapes(
    canvas: np.ndarray,
    reserve: np.ndarray,
    plans: list[dict],
) -> tuple[np.ndarray, list[dict]] | None:
    placements: list[dict] = []
    order = []
    for plan in plans:
        for kind in plan["kinds"]:
            order.append({"color": int(plan["color"]), "kind": kind})
    random.shuffle(order)
    order.sort(key=lambda item: -int(_SHAPE_MASKS[item["kind"]].sum()))

    for item in order:
        patch = _shape_patch(item["kind"], item["color"])
        try:
            r0, c0 = random_free_location(
                reserve.astype(int),
                patch,
                transparent=BG,
                border=1,
                padding=1,
                padding_connectivity=8,
            )
        except ValueError:
            return None
        paste(canvas, patch, int(r0), int(c0), transparent=BG)
        paste(reserve, _colored_from_mask(np.ones(patch.shape, dtype=bool), 9), int(r0), int(c0), transparent=BG)
        coords = np.argwhere(patch != BG) + np.array([r0, c0])
        placements.append(
            {
                "color": int(item["color"]),
                "kind": item["kind"],
                "bbox": (int(r0), int(c0), int(r0 + patch.shape[0] - 1), int(c0 + patch.shape[1] - 1)),
                "coords": coords,
            }
        )
    return canvas, placements


def _generate_candidate() -> tuple[np.ndarray, list[dict], list[dict]] | None:
    n_frames = random.choice([2, 2, 3])
    plans = _choose_color_plans(n_frames)
    if plans is None:
        return None
    layout = _layout_frames(plans)
    if layout is None:
        return None
    h, w, frames = layout

    canvas = np.zeros((h, w), dtype=int)
    reserve = np.zeros((h, w), dtype=bool)
    for frame in frames:
        r0, c0, r1, c1 = frame["bbox"]
        patch = _frame_patch(frame["height"], frame["width"], frame["color"])
        paste(canvas, patch, r0, c0, transparent=BG)
        _reserve_bbox(reserve, (r0, c0, r1, c1), pad=1)

    placed = _place_shapes(canvas, reserve, plans)
    if placed is None:
        return None
    canvas, placements = placed
    return canvas, plans, placements


def _validate_candidate(grid: np.ndarray, plans: list[dict], placements: list[dict]) -> bool:
    bg, frames, shapes = _scene_parts(grid)
    if len(frames) != len(plans):
        return False
    if len(shapes) != len(placements):
        return False

    totals = _color_totals(grid, shapes, bg)
    plan_by_color = {int(plan["color"]): plan for plan in plans}
    if set(totals) != set(plan_by_color):
        return False
    for color, total in totals.items():
        if int(total) != int(plan_by_color[int(color)]["total"]):
            return False

    output = main(grid)
    if output.shape != grid.shape:
        return False

    for comp in shapes:
        for r, c in comp["coords"]:
            if output[int(r), int(c)] != int(bg):
                return False

    frame_drawn: list[int] = []
    for frame in frames:
        color = int(frame["color"])
        capacity = _frame_capacity(frame)
        target = min(capacity, int(totals[color]))
        r0, c0, r1, c1 = map(int, frame["bbox"])
        center_r = (r0 + r1) // 2
        cols = list(range(c1 - 1, c0, -2))
        row_slice = output[center_r, c0 + 1 : c1]
        if int(np.count_nonzero(row_slice == color)) != target:
            return False
        expected_cols = set(cols[:target])
        for c in range(c0 + 1, c1):
            if c in expected_cols:
                if output[center_r, c] != color:
                    return False
            elif output[center_r, c] != bg:
                return False
        frame_drawn.append(int(target))

    if len(set(frame_drawn)) == 1:
        return False

    hole_sensitive = False
    exterior_bg = _exterior_background(grid, bg)
    for comp in shapes:
        if _bbox_size(comp["bbox"]) != (3, 3):
            continue
        sub = grid[comp["bbox"][0] : comp["bbox"][2] + 1, comp["bbox"][1] : comp["bbox"][3] + 1]
        if not np.array_equal(sub != bg, _SHAPE_MASKS["ring33"]):
            continue
        color = int(comp["color"])
        plan = plan_by_color[color]
        if int(plan["capacity"]) < 13 or int(plan["total"]) != 12:
            continue
        exact = _shape_outer_count(grid, comp["coords"], bg, exterior_bg)
        naive = _naive_adjacent_bg_count(grid, comp["coords"], bg)
        if exact == 12 and naive == 13 and min(naive, int(plan["capacity"])) != min(exact, int(plan["capacity"])):
            hole_sensitive = True
            break
    if not hole_sensitive:
        return False

    return True


def generate_input() -> np.ndarray:
    for _ in range(GEN_ATTEMPTS):
        candidate = _generate_candidate()
        if candidate is None:
            continue
        grid, plans, placements = candidate
        if _validate_candidate(grid, plans, placements):
            return grid
    raise RuntimeError("failed to generate a valid contour-count frame task")
