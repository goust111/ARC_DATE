from __future__ import annotations

import random
from collections import defaultdict

import numpy as np


BACKGROUND = 0
LEFT_WALL = 1
RIGHT_WALL = 2
GEN_ATTEMPTS = 500


def _bbox_from_coords(coords: np.ndarray) -> tuple[int, int, int, int]:
    r0, c0 = coords.min(axis=0)
    r1, c1 = coords.max(axis=0)
    return int(r0), int(c0), int(r1), int(c1)


def _component_mask(coords: np.ndarray) -> np.ndarray:
    r0, c0, r1, c1 = _bbox_from_coords(coords)
    mask = np.zeros((r1 - r0 + 1, c1 - c0 + 1), dtype=bool)
    rel = coords - np.array([r0, c0], dtype=int)
    mask[rel[:, 0], rel[:, 1]] = True
    return mask


def _is_vertically_symmetric(mask: np.ndarray) -> bool:
    return bool(np.array_equal(mask, np.fliplr(mask)))


def _find_inner_components(grid: np.ndarray) -> list[dict]:
    h, w = grid.shape
    visited = np.zeros_like(grid, dtype=bool)
    comps: list[dict] = []

    for r in range(h):
        for c in range(1, w - 1):
            color = int(grid[r, c])
            if color not in (LEFT_WALL, RIGHT_WALL) or visited[r, c]:
                continue

            stack = [(r, c)]
            visited[r, c] = True
            cells: list[tuple[int, int]] = []
            while stack:
                rr, cc = stack.pop()
                cells.append((rr, cc))
                for dr, dc in ((-1, 0), (1, 0), (0, -1), (0, 1)):
                    nr, nc = rr + dr, cc + dc
                    if not (0 <= nr < h and 1 <= nc < w - 1):
                        continue
                    if visited[nr, nc] or int(grid[nr, nc]) != color:
                        continue
                    visited[nr, nc] = True
                    stack.append((nr, nc))

            coords = np.array(cells, dtype=int)
            comps.append(
                {
                    "color": color,
                    "coords": coords,
                    "bbox": _bbox_from_coords(coords),
                    "mask": _component_mask(coords),
                    "area": int(coords.shape[0]),
                }
            )
    return comps


def _settle_horizontally(coords: np.ndarray, direction: int, occupied: np.ndarray) -> np.ndarray:
    moved = coords.copy()
    assert not np.any(occupied[moved[:, 0], moved[:, 1]])
    while True:
        trial = moved.copy()
        trial[:, 1] += int(direction)
        if trial[:, 1].min() < 0 or trial[:, 1].max() >= occupied.shape[1]:
            break
        if np.any(occupied[trial[:, 0], trial[:, 1]]):
            break
        moved = trial
    return moved


def _solve(grid: np.ndarray, *, return_trace: bool = False):
    g = np.asarray(grid, dtype=int)
    assert g.ndim == 2 and min(g.shape) >= 3
    h, w = g.shape
    assert np.all(g[:, 0] == LEFT_WALL), "left wall must be a full blue column"
    assert np.all(g[:, w - 1] == RIGHT_WALL), "right wall must be a full red column"
    assert set(map(int, np.unique(g))).issubset({BACKGROUND, LEFT_WALL, RIGHT_WALL})

    comps = _find_inner_components(g)
    out = np.full_like(g, BACKGROUND)
    out[:, 0] = LEFT_WALL
    out[:, w - 1] = RIGHT_WALL
    occupied = out != BACKGROUND

    items = []
    for idx, comp in enumerate(comps):
        symmetric = _is_vertically_symmetric(comp["mask"])
        final_color = int(comp["color"] if symmetric else LEFT_WALL + RIGHT_WALL - comp["color"])
        bbox = comp["bbox"]
        distance = int(bbox[1] - 1) if final_color == LEFT_WALL else int((w - 2) - bbox[3])
        assert distance >= 1, "inner sprites must not start already touching a wall"
        items.append(
            {
                "id": idx,
                "coords": comp["coords"],
                "bbox": bbox,
                "area": comp["area"],
                "input_color": int(comp["color"]),
                "symmetric": symmetric,
                "final_color": final_color,
                "distance": distance,
            }
        )

    items.sort(key=lambda item: (item["distance"], item["bbox"][0], item["bbox"][1], item["area"], item["id"]))

    trace = []
    for item in items:
        direction = -1 if item["final_color"] == LEFT_WALL else 1
        final_coords = _settle_horizontally(item["coords"], direction, occupied)
        assert not np.any(occupied[final_coords[:, 0], final_coords[:, 1]])
        out[final_coords[:, 0], final_coords[:, 1]] = int(item["final_color"])
        occupied[final_coords[:, 0], final_coords[:, 1]] = True

        final_bbox = _bbox_from_coords(final_coords)
        final_gap = int(final_bbox[1] - 1) if item["final_color"] == LEFT_WALL else int((w - 2) - final_bbox[3])
        trace.append(
            {
                **item,
                "final_coords": final_coords,
                "final_bbox": final_bbox,
                "final_gap": final_gap,
            }
        )

    if return_trace:
        return out, trace
    return out


def main(grid: np.ndarray) -> np.ndarray:
    """
    Use each sprite's vertical symmetry to decide recoloring, then slide it toward the matching wall.

    Input structure:
    - The background is black.
    - The left column is a full blue wall and the right column is a full red wall.
    - Inside the walls are several small monochrome 4-connected sprites, each colored blue or red.

    Rule:
    - Crop every sprite to its bounding box and test whether that crop is symmetric under left-right reflection.
    - Symmetric sprites keep their color, while asymmetric sprites swap blue and red.
    - After that recoloring, each sprite targets the wall of its current color.
    - Process sprites from nearest to farthest from their target wall, sliding each horizontally until it would hit the wall column or an already settled sprite, without changing its rows.

    Key insight:
    - Vertical symmetry is not the final selection criterion; it silently changes a sprite's color, and that new color is what determines the later sliding direction.
    """
    return _solve(grid, return_trace=False)


def _is_connected(mask: np.ndarray) -> bool:
    pts = np.argwhere(mask)
    if pts.size == 0:
        return False
    start = tuple(int(x) for x in pts[0])
    seen = {start}
    stack = [start]
    while stack:
        r, c = stack.pop()
        for dr, dc in ((-1, 0), (1, 0), (0, -1), (0, 1)):
            nr, nc = r + dr, c + dc
            if 0 <= nr < mask.shape[0] and 0 <= nc < mask.shape[1] and mask[nr, nc] and (nr, nc) not in seen:
                seen.add((nr, nc))
                stack.append((nr, nc))
    return len(seen) == int(mask.sum())


def _shape_pool() -> tuple[list[np.ndarray], list[np.ndarray]]:
    symmetric: list[np.ndarray] = []
    asymmetric: list[np.ndarray] = []
    for h in (2, 3):
        for w in (2, 3):
            for bits in range(1 << (h * w)):
                mask = np.zeros((h, w), dtype=bool)
                for k in range(h * w):
                    if bits & (1 << k):
                        mask[k // w, k % w] = True
                area = int(mask.sum())
                if not (3 <= area <= 6):
                    continue
                if not (mask[0].any() and mask[-1].any() and mask[:, 0].any() and mask[:, -1].any()):
                    continue
                if mask.all():
                    continue
                if not _is_connected(mask):
                    continue
                if _is_vertically_symmetric(mask):
                    symmetric.append(mask.copy())
                else:
                    asymmetric.append(mask.copy())
    return symmetric, asymmetric


SYMMETRIC_MASKS, ASYMMETRIC_MASKS = _shape_pool()


def _mask_to_coords(mask: np.ndarray, top: int, left: int) -> np.ndarray:
    rel = np.argwhere(mask)
    return rel + np.array([int(top), int(left)], dtype=int)


def _overlaps_rows(a: tuple[int, int], b: tuple[int, int]) -> bool:
    return not (a[1] < b[0] or b[1] < a[0])


def _random_group_tops(objects: list[dict], top_min: int, top_max: int) -> list[int] | None:
    tops: list[int] = []
    for i, obj in enumerate(objects):
        h = obj["mask"].shape[0]
        if i == 0:
            lo = int(top_min)
            hi = int(top_max) - h + 1
        else:
            prev_top = tops[-1]
            prev_h = objects[i - 1]["mask"].shape[0]
            lo = max(int(top_min), prev_top - h + 1)
            hi = min(int(top_max) - h + 1, prev_top + prev_h - 1)
        if lo > hi:
            return None
        tops.append(random.randint(lo, hi))
    return tops


def _assign_increasing_columns(widths: list[int], lows: list[int], highs: list[int]) -> list[int] | None:
    n = len(widths)

    def rec(i: int, prev_end: int) -> list[int] | None:
        if i == n:
            return []
        lo = max(int(lows[i]), int(prev_end + 2))
        hi = int(highs[i])
        candidates = list(range(lo, hi + 1))
        random.shuffle(candidates)
        for c0 in candidates:
            rest = rec(i + 1, c0 + widths[i] - 1)
            if rest is not None:
                return [c0] + rest
        return None

    return rec(0, -2)


def _spec_for(input_color: int, symmetric: bool) -> dict:
    final_color = int(input_color if symmetric else LEFT_WALL + RIGHT_WALL - input_color)
    return {
        "input_color": int(input_color),
        "symmetric": bool(symmetric),
        "final_color": final_color,
        "side": "L" if final_color == LEFT_WALL else "R",
        "mask": random.choice(SYMMETRIC_MASKS if symmetric else ASYMMETRIC_MASKS).copy(),
    }


def _base_specs(count: int) -> list[dict]:
    specs = [
        _spec_for(LEFT_WALL, True),
        _spec_for(LEFT_WALL, False),
        _spec_for(RIGHT_WALL, True),
        _spec_for(RIGHT_WALL, False),
    ]
    if count == 5:
        if random.random() < 0.5:
            specs.append(_spec_for(*random.choice([(LEFT_WALL, True), (RIGHT_WALL, False)])))
        else:
            specs.append(_spec_for(*random.choice([(LEFT_WALL, False), (RIGHT_WALL, True)])))
    elif count == 6:
        specs.append(_spec_for(*random.choice([(LEFT_WALL, True), (RIGHT_WALL, False)])))
        specs.append(_spec_for(*random.choice([(LEFT_WALL, False), (RIGHT_WALL, True)])))
    random.shuffle(specs)
    return specs


def _semantic_check(inp: np.ndarray, expected: np.ndarray) -> bool:
    if inp.ndim != 2 or expected.ndim != 2 or max(inp.shape) > 30 or max(expected.shape) > 30:
        return False
    if not (np.all(inp[:, 0] == LEFT_WALL) and np.all(inp[:, -1] == RIGHT_WALL)):
        return False

    comps = _find_inner_components(inp)
    if not (4 <= len(comps) <= 6):
        return False

    by_input_color: dict[int, list[bool]] = defaultdict(list)
    for comp in comps:
        by_input_color[int(comp["color"])].append(_is_vertically_symmetric(comp["mask"]))
        if not (3 <= comp["area"] <= 6):
            return False
        h = comp["mask"].shape[0]
        w = comp["mask"].shape[1]
        if h > 3 or w > 3:
            return False

    for color in (LEFT_WALL, RIGHT_WALL):
        flags = by_input_color[color]
        if len(flags) < 2 or not any(flags) or all(flags):
            return False

    solved, trace = _solve(inp, return_trace=True)
    if not np.array_equal(solved, expected):
        return False
    if np.array_equal(inp, solved):
        return False

    distances = [int(item["distance"]) for item in trace]
    if len(distances) != len(set(distances)):
        return False

    by_final_color: dict[int, list[dict]] = defaultdict(list)
    for item in trace:
        by_final_color[int(item["final_color"])].append(item)
    for color in (LEFT_WALL, RIGHT_WALL):
        if len(by_final_color[color]) < 2:
            return False
        if not any(int(item["final_gap"]) > 0 for item in by_final_color[color]):
            return False
    return True


def _generate_attempt() -> np.ndarray | None:
    h = random.randint(18, 22)
    w = random.randint(18, 22)
    count = random.randint(4, 6)
    specs = _base_specs(count)

    left_group = [dict(spec) for spec in specs if spec["side"] == "L"]
    right_group = [dict(spec) for spec in specs if spec["side"] == "R"]
    random.shuffle(left_group)
    random.shuffle(right_group)

    left_heights = [obj["mask"].shape[0] for obj in left_group]
    right_heights = [obj["mask"].shape[0] for obj in right_group]
    if not left_heights or not right_heights:
        return None

    left_top_max = max(2, h // 2 - 3)
    right_top_min = min(h - 4, h // 2 + 1)
    left_tops = _random_group_tops(left_group, 1, left_top_max)
    right_tops = _random_group_tops(right_group, right_top_min, h - 2)
    if left_tops is None or right_tops is None:
        return None

    occupied = np.zeros((h, w), dtype=bool)
    occupied[:, 0] = True
    occupied[:, -1] = True
    expected = np.full((h, w), BACKGROUND, dtype=int)
    expected[:, 0] = LEFT_WALL
    expected[:, -1] = RIGHT_WALL

    for group, tops, direction, start_col in (
        (left_group, left_tops, -1, max(4, w // 2 - 3)),
        (right_group, right_tops, 1, min(w - 5, w // 2)),
    ):
        for obj, top in zip(group, tops):
            width = obj["mask"].shape[1]
            c0 = int(start_col if direction < 0 else min(start_col, w - width - 1))
            coords = _mask_to_coords(obj["mask"], top, c0)
            if np.any(occupied[coords[:, 0], coords[:, 1]]):
                return None
            final_coords = _settle_horizontally(coords, direction, occupied)
            final_bbox = _bbox_from_coords(final_coords)
            final_gap = int(final_bbox[1] - 1) if direction < 0 else int((w - 2) - final_bbox[3])
            obj["top"] = int(top)
            obj["final_coords"] = final_coords
            obj["final_bbox"] = final_bbox
            obj["final_gap"] = final_gap
            occupied[final_coords[:, 0], final_coords[:, 1]] = True
            expected[final_coords[:, 0], final_coords[:, 1]] = int(obj["final_color"])

    if not any(int(obj["final_gap"]) > 0 for obj in left_group):
        return None
    if not any(int(obj["final_gap"]) > 0 for obj in right_group):
        return None

    left_widths = [obj["mask"].shape[1] for obj in left_group]
    left_lows = [int(obj["final_bbox"][1] + 1) for obj in left_group]
    left_highs = [int(w - 1 - width) for width in left_widths]
    left_cols = _assign_increasing_columns(left_widths, left_lows, left_highs)
    if left_cols is None:
        return None

    right_spatial = list(reversed(right_group))
    right_widths = [obj["mask"].shape[1] for obj in right_spatial]
    right_lows = [1 for _ in right_spatial]
    right_highs = [int(obj["final_bbox"][1] - 1) for obj in right_spatial]
    right_cols_spatial = _assign_increasing_columns(right_widths, right_lows, right_highs)
    if right_cols_spatial is None:
        return None

    inp = np.full((h, w), BACKGROUND, dtype=int)
    inp[:, 0] = LEFT_WALL
    inp[:, -1] = RIGHT_WALL

    for obj, c0 in zip(left_group, left_cols):
        coords = _mask_to_coords(obj["mask"], obj["top"], c0)
        if np.any(inp[coords[:, 0], coords[:, 1]] != BACKGROUND):
            return None
        inp[coords[:, 0], coords[:, 1]] = int(obj["input_color"])

    right_col_map = {id(obj): c0 for obj, c0 in zip(right_spatial, right_cols_spatial)}
    for obj in right_group:
        c0 = right_col_map[id(obj)]
        coords = _mask_to_coords(obj["mask"], obj["top"], c0)
        if np.any(inp[coords[:, 0], coords[:, 1]] != BACKGROUND):
            return None
        inp[coords[:, 0], coords[:, 1]] = int(obj["input_color"])

    if not _semantic_check(inp, expected):
        return None
    return inp


def generate_input() -> np.ndarray:
    for _ in range(GEN_ATTEMPTS):
        grid = _generate_attempt()
        if grid is not None:
            return grid
    raise RuntimeError("failed to generate a valid symmetry-recolor-slide input")
