from __future__ import annotations

import random
from collections import Counter, deque

import numpy as np


BLACK = 0
PALETTE = (1, 2, 3, 4, 5, 6, 7, 8, 9)
RAY_SPECS = (
    ("nw", 0, 0, -1, -1),
    ("ne", 0, -1, -1, 1),
    ("sw", -1, 0, 1, -1),
    ("se", -1, -1, 1, 1),
)

GEN_ATTEMPTS = 600
GRID_RANGE = (18, 24)
RECT_COUNT_WEIGHTS = ((2, 3), (3, 4), (4, 3))
MODE_WEIGHTS = (("collision", 5), ("open", 3), ("mixed", 2))


def _get_bg(grid: np.ndarray) -> int:
    return int(Counter(map(int, grid.flat)).most_common(1)[0][0])


def _bbox_from_coords(coords: list[tuple[int, int]]) -> tuple[int, int, int, int]:
    rows = [r for r, _c in coords]
    cols = [c for _r, c in coords]
    return min(rows), min(cols), max(rows), max(cols)


def _box_intersects(
    a: tuple[int, int, int, int],
    b: tuple[int, int, int, int],
    pad: int = 0,
) -> bool:
    ar0, ac0, ar1, ac1 = a
    br0, bc0, br1, bc1 = b
    return not (
        ar1 + pad < br0 - pad
        or br1 + pad < ar0 - pad
        or ac1 + pad < bc0 - pad
        or bc1 + pad < ac0 - pad
    )


def _rect_box(rect: dict) -> tuple[int, int, int, int]:
    return rect["r0"], rect["c0"], rect["r0"] + rect["h"] - 1, rect["c0"] + rect["w"] - 1


def _original_rect(rect: dict) -> dict:
    return {
        **rect,
        "r0": rect["r0"] + rect["h"],
    }


def _row_bounds(height: int, rect_h: int) -> tuple[int, int]:
    return max(0, height // 2 - rect_h), height - 2 * rect_h


def _paint_rect(grid: np.ndarray, rect: dict) -> None:
    r0, c0 = rect["r0"], rect["c0"]
    grid[r0 : r0 + rect["h"], c0 : c0 + rect["w"]] = rect["color"]


def _find_components(grid: np.ndarray, bg: int) -> list[dict]:
    h, w = grid.shape
    seen = np.zeros((h, w), dtype=bool)
    comps: list[dict] = []
    for r in range(h):
        for c in range(w):
            if seen[r, c] or int(grid[r, c]) == bg:
                continue
            color = int(grid[r, c])
            q: deque[tuple[int, int]] = deque([(r, c)])
            seen[r, c] = True
            coords: list[tuple[int, int]] = []
            while q:
                rr, cc = q.popleft()
                coords.append((rr, cc))
                for dr, dc in ((-1, 0), (1, 0), (0, -1), (0, 1)):
                    nr = rr + dr
                    nc = cc + dc
                    if 0 <= nr < h and 0 <= nc < w and not seen[nr, nc] and int(grid[nr, nc]) == color:
                        seen[nr, nc] = True
                        q.append((nr, nc))
            comps.append(
                {
                    "color": color,
                    "coords": coords,
                    "bbox": _bbox_from_coords(coords),
                    "area": len(coords),
                }
            )
    return comps


def _extract_rectangles(grid: np.ndarray) -> list[dict]:
    g = np.asarray(grid, dtype=int)
    assert g.ndim == 2, "expected a 2D grid"
    bg = _get_bg(g)
    comps = _find_components(g, bg)
    assert 2 <= len(comps) <= 4, "expected 2-4 foreground rectangles"

    rects: list[dict] = []
    colors = set()
    heights = set()
    for idx, comp in enumerate(comps):
        color = int(comp["color"])
        r0, c0, r1, c1 = comp["bbox"]
        h = r1 - r0 + 1
        w = c1 - c0 + 1
        patch = g[r0 : r1 + 1, c0 : c1 + 1]
        assert int(comp["area"]) == h * w, "foreground objects must be solid rectangles"
        assert np.all(patch == color), "rectangle must be monochromatic"
        rects.append(
            {
                "id": idx,
                "color": color,
                "r0": r0,
                "c0": c0,
                "h": h,
                "w": w,
                "area": h * w,
            }
        )
        colors.add(color)
        heights.add(h)

    assert len(colors) == len(rects), "rectangles must use distinct colors"
    assert len(heights) == len(rects), "rectangles must have distinct heights"
    return rects


def _move_rectangles(rects: list[dict]) -> list[dict]:
    moved: list[dict] = []
    for rect in rects:
        nr0 = rect["r0"] - rect["h"]
        assert nr0 >= 0, "rectangle must fit after moving upward by its own height"
        moved.append({**rect, "r0": nr0})
    return moved


def _render_rectangles(shape: tuple[int, int], rects: list[dict]) -> tuple[np.ndarray, np.ndarray]:
    h, w = shape
    grid = np.full((h, w), BLACK, dtype=int)
    owner = np.full((h, w), -1, dtype=int)
    for rect in rects:
        r0, c0, r1, c1 = _rect_box(rect)
        assert 0 <= r0 <= r1 < h and 0 <= c0 <= c1 < w
        assert np.all(owner[r0 : r1 + 1, c0 : c1 + 1] < 0), "rectangles must not overlap"
        grid[r0 : r1 + 1, c0 : c1 + 1] = rect["color"]
        owner[r0 : r1 + 1, c0 : c1 + 1] = rect["id"]
    return grid, owner


def _ray_order(rects: list[dict]) -> list[dict]:
    return sorted(rects, key=lambda rect: (-rect["area"], rect["r0"], rect["c0"]))


def _trace_rays_for_rect(
    shape: tuple[int, int],
    rect: dict,
    owner: np.ndarray,
    prior_rays: set[tuple[int, int]],
    block_on_rays: bool,
) -> tuple[list[dict], list[tuple[int, int, int]]]:
    h, w = shape
    infos: list[dict] = []
    new_cells: list[tuple[int, int, int]] = []
    for name, r_off, c_off, dr, dc in RAY_SPECS:
        start_r = rect["r0"] if r_off == 0 else rect["r0"] + rect["h"] - 1
        start_c = rect["c0"] if c_off == 0 else rect["c0"] + rect["w"] - 1
        rr = start_r
        cc = start_c
        ray_cells: list[tuple[int, int, int]] = []
        while True:
            nr = rr + dr
            nc = cc + dc
            if not (0 <= nr < h and 0 <= nc < w):
                stop = "edge"
                break
            blocker = int(owner[nr, nc])
            if blocker >= 0 and blocker != rect["id"]:
                stop = "rectangle"
                break
            if block_on_rays and (nr, nc) in prior_rays:
                stop = "ray"
                break
            ray_cells.append((nr, nc, rect["color"]))
            rr, cc = nr, nc
        infos.append(
            {
                "rect_id": rect["id"],
                "ray": name,
                "length": len(ray_cells),
                "stop": stop,
            }
        )
        new_cells.extend(ray_cells)
    return infos, new_cells


def _render_output_from_moved(
    shape: tuple[int, int],
    moved_rects: list[dict],
    *,
    block_on_rays: bool = True,
    order: list[dict] | None = None,
) -> tuple[np.ndarray, list[dict]]:
    out, owner = _render_rectangles(shape, moved_rects)
    ordered = _ray_order(moved_rects) if order is None else order
    prior_rays: set[tuple[int, int]] = set()
    infos: list[dict] = []

    for rect in ordered:
        rect_infos, cells = _trace_rays_for_rect(shape, rect, owner, prior_rays, block_on_rays)
        infos.extend(rect_infos)
        for r, c, color in cells:
            if owner[r, c] < 0:
                out[r, c] = color
        if block_on_rays:
            prior_rays.update((r, c) for r, c, _color in cells)
    return out, infos


def main(grid: np.ndarray) -> np.ndarray:
    """Move each solid rectangle upward by its own height, then emit clipped diagonal corner rays from the moved rectangles.

    Input structure:
    - The grid has a black background and 2-4 disconnected solid monochromatic rectangles.
    - Rectangle colors are distinct, rectangle heights are distinct, and each rectangle has enough empty space directly above it to move upward by its own height.
    - No other foreground pixels are present.

    Rule:
    - Erase every original rectangle and place it exactly one rectangle-height higher.
    - After all rectangles have moved, process the moved rectangles in descending area order; ties use the moved top-left corner in row-major order.
    - For each moved rectangle, emit four same-color diagonal rays from its four corners outward (NW, NE, SW, SE). The rectangle's four rays are simultaneous, but moved rectangles and previously painted rays are blockers; each ray stops before leaving the grid, entering another moved rectangle, or entering an earlier ray cell.

    Key insight:
    - The upward self-height translation must happen before any ray logic, because the moved layout both defines the ray origins and creates the obstacle pattern that clips later rays.
    """
    rects = _extract_rectangles(np.asarray(grid, dtype=int))
    moved_rects = _move_rectangles(rects)
    out, _infos = _render_output_from_moved(np.asarray(grid).shape, moved_rects)
    return out


def _weighted_choice(weighted_items: tuple[tuple[object, int], ...]) -> object:
    items = [item for item, _weight in weighted_items]
    weights = [weight for _item, weight in weighted_items]
    return random.choices(items, weights=weights, k=1)[0]


def _sample_specs(height: int, count: int) -> list[dict]:
    max_h = min(5, height // 2 - 1)
    heights = random.sample(list(range(2, max_h + 1)), count)
    colors = random.sample(PALETTE, count)
    specs: list[dict] = []
    for color, rect_h in zip(colors, heights):
        rect_w = random.randint(2, 5)
        if random.random() < 0.15:
            rect_w = 1
        specs.append({"color": color, "h": rect_h, "w": rect_w, "area": rect_h * rect_w})
    random.shuffle(specs)
    return specs


def _candidate_ok(candidate: dict, placed: list[dict], height: int) -> bool:
    moved_box = _rect_box(candidate)
    orig_box = _rect_box(_original_rect(candidate))
    if _original_rect(candidate)["r0"] < height // 2:
        return False
    for rect in placed:
        other_moved = _rect_box(rect)
        other_orig = _rect_box(_original_rect(rect))
        if _box_intersects(moved_box, other_moved, pad=0):
            return False
        if _box_intersects(orig_box, other_orig, pad=1):
            return False
        if _box_intersects(moved_box, other_orig, pad=0):
            return False
        if _box_intersects(orig_box, other_moved, pad=0):
            return False
    return True


def _try_place_spec(
    shape: tuple[int, int],
    spec: dict,
    placed: list[dict],
    *,
    bias_edge: bool,
) -> dict | None:
    height, width = shape
    row_lo, row_hi = _row_bounds(height, spec["h"])
    if row_lo > row_hi or spec["w"] > width:
        return None

    for _ in range(240):
        if bias_edge and random.random() < 0.7:
            r0 = random.choice((row_lo, random.randint(row_lo, row_hi)))
            c0 = random.choice((0, max(0, width - spec["w"]), random.randint(0, width - spec["w"])))
        else:
            r0 = random.randint(row_lo, row_hi)
            c0 = random.randint(0, width - spec["w"])
        candidate = {
            "id": len(placed),
            "color": spec["color"],
            "r0": r0,
            "c0": c0,
            "h": spec["h"],
            "w": spec["w"],
            "area": spec["area"],
        }
        if _candidate_ok(candidate, placed, height):
            return candidate
    return None


def _build_collision_scene(shape: tuple[int, int], specs: list[dict]) -> list[dict] | None:
    height, width = shape
    specs = sorted(specs, key=lambda spec: (-spec["area"], -spec["h"], -spec["w"]))
    rect_a = specs[0]
    rect_b = specs[-1]
    extras = specs[1:-1]

    for _ in range(240):
        gap = random.randint(2, 4)
        max_c0 = width - rect_a["w"] - gap - rect_b["w"]
        if max_c0 < 0:
            continue
        a_row_lo, a_row_hi = _row_bounds(height, rect_a["h"])
        b_row_lo, b_row_hi = _row_bounds(height, rect_b["h"])
        a_row_hi = min(a_row_hi, b_row_hi - rect_a["h"] - gap)
        if a_row_lo > a_row_hi:
            continue

        if random.random() < 0.65:
            c0_a = random.choice([0, min(1, max_c0), max_c0])
        else:
            c0_a = random.randint(0, max_c0)
        r0_a = a_row_lo if random.random() < 0.55 else random.randint(a_row_lo, a_row_hi)
        r0_b = r0_a + rect_a["h"] + gap
        c0_b = c0_a + rect_a["w"] + gap

        placed = [
            {
                "id": 0,
                "color": rect_a["color"],
                "r0": r0_a,
                "c0": c0_a,
                "h": rect_a["h"],
                "w": rect_a["w"],
                "area": rect_a["area"],
            }
        ]
        if not _candidate_ok(placed[0], [], height):
            continue

        candidate_b = {
            "id": 1,
            "color": rect_b["color"],
            "r0": r0_b,
            "c0": c0_b,
            "h": rect_b["h"],
            "w": rect_b["w"],
            "area": rect_b["area"],
        }
        if not _candidate_ok(candidate_b, placed, height):
            continue
        placed.append(candidate_b)

        ok = True
        for idx, spec in enumerate(sorted(extras, key=lambda item: (-item["area"], -item["h"]))):
            candidate = _try_place_spec(shape, spec, placed, bias_edge=idx == 0)
            if candidate is None:
                ok = False
                break
            placed.append(candidate)
        if ok:
            return placed
    return None


def _build_open_scene(shape: tuple[int, int], specs: list[dict]) -> list[dict] | None:
    placed: list[dict] = []
    specs = sorted(specs, key=lambda spec: (-spec["area"], -spec["h"], -spec["w"]))
    edge_owner = random.randrange(len(specs))
    for idx, spec in enumerate(specs):
        candidate = _try_place_spec(shape, spec, placed, bias_edge=idx == edge_owner)
        if candidate is None:
            return None
        placed.append(candidate)
    return placed


def _build_mixed_scene(shape: tuple[int, int], specs: list[dict]) -> list[dict] | None:
    if len(specs) < 3:
        return _build_collision_scene(shape, specs)
    placed = _build_collision_scene(shape, specs[:2] + specs[2:])
    if placed is None:
        return None
    return placed


def _build_input_from_moved(shape: tuple[int, int], moved_rects: list[dict]) -> np.ndarray | None:
    grid = np.full(shape, BLACK, dtype=int)
    originals = [_original_rect(rect) for rect in moved_rects]

    for rect in originals:
        r0, c0, r1, c1 = _rect_box(rect)
        if not (0 <= r0 <= r1 < shape[0] and 0 <= c0 <= c1 < shape[1]):
            return None
        if np.any(grid[r0 : r1 + 1, c0 : c1 + 1] != BLACK):
            return None
        _paint_rect(grid, rect)

    for rect in moved_rects:
        r0, c0, r1, c1 = _rect_box(rect)
        if np.any(grid[r0 : r1 + 1, c0 : c1 + 1] != BLACK):
            return None

    try:
        parsed = _extract_rectangles(grid)
    except AssertionError:
        return None
    if len(parsed) != len(moved_rects):
        return None
    return grid


def _ray_metrics(infos: list[dict]) -> dict:
    stop_counts = Counter(info["stop"] for info in infos)
    lengths = [info["length"] for info in infos]
    active_rects = len({info["rect_id"] for info in infos if info["length"] > 0})
    return {
        "stop_counts": stop_counts,
        "total_ray_cells": sum(lengths),
        "long_rays": sum(length >= 2 for length in lengths),
        "active_rects": active_rects,
    }


def _semantic_validation(inp: np.ndarray, moved_rects: list[dict], mode: str) -> bool:
    expected, infos = _render_output_from_moved(inp.shape, moved_rects)
    actual = main(inp)
    if not np.array_equal(actual, expected):
        return False

    moved_only, _infos = _render_rectangles(inp.shape, moved_rects)
    if np.array_equal(actual, moved_only):
        return False

    original_rays, _infos = _render_output_from_moved(inp.shape, [_original_rect(rect) for rect in moved_rects])
    if np.array_equal(actual, original_rays):
        return False

    metrics = _ray_metrics(infos)
    filled_ratio = int(np.count_nonzero(actual)) / actual.size
    if metrics["total_ray_cells"] < 12 or metrics["active_rects"] < 2 or filled_ratio > 0.35:
        return False
    if metrics["stop_counts"]["edge"] == 0:
        return False

    if mode == "collision":
        if metrics["stop_counts"]["rectangle"] == 0 or metrics["stop_counts"]["ray"] == 0:
            return False
        no_ray_block, _infos = _render_output_from_moved(inp.shape, moved_rects, block_on_rays=False)
        if np.array_equal(actual, no_ray_block):
            return False
    return True


def _gen_attempt() -> np.ndarray | None:
    height = random.randint(*GRID_RANGE)
    width = random.randint(*GRID_RANGE)
    count = _weighted_choice(RECT_COUNT_WEIGHTS)
    mode = _weighted_choice(MODE_WEIGHTS)
    specs = _sample_specs(height, count)

    if mode == "collision":
        moved_rects = _build_collision_scene((height, width), specs)
    elif mode == "mixed":
        moved_rects = _build_collision_scene((height, width), specs)
        if moved_rects is None:
            moved_rects = _build_open_scene((height, width), specs)
    else:
        moved_rects = _build_open_scene((height, width), specs)

    if moved_rects is None:
        return None
    inp = _build_input_from_moved((height, width), moved_rects)
    if inp is None:
        return None
    if not _semantic_validation(inp, moved_rects, "collision" if mode == "mixed" else mode):
        return None
    return inp


def generate_input() -> np.ndarray:
    for _ in range(GEN_ATTEMPTS):
        grid = _gen_attempt()
        if grid is not None:
            return grid
    raise RuntimeError("failed to generate a valid input for puzzle 6")
