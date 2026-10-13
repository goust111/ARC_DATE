import random
from collections import deque

import numpy as np


DIR4 = ((1, 0), (-1, 0), (0, 1), (0, -1))
DIVIDER_COLOR = 5
GEN_ATTEMPTS = 300
SIDE_BORDER = 1
SIDE_GAP_CHOICES = (1, 2)


def _mode_color(grid: np.ndarray) -> int:
    values, counts = np.unique(grid, return_counts=True)
    return int(values[np.argmax(counts)])


def _bbox(coords: np.ndarray) -> tuple[int, int, int, int]:
    return (
        int(coords[:, 0].min()),
        int(coords[:, 1].min()),
        int(coords[:, 0].max()),
        int(coords[:, 1].max()),
    )


def _mask_connected(mask: np.ndarray) -> bool:
    cells = np.argwhere(mask)
    if len(cells) == 0:
        return False
    start = tuple(map(int, cells[0]))
    seen = {start}
    queue = deque([start])
    h, w = mask.shape
    while queue:
        r, c = queue.popleft()
        for dr, dc in DIR4:
            nr, nc = r + dr, c + dc
            if 0 <= nr < h and 0 <= nc < w and mask[nr, nc] and (nr, nc) not in seen:
                seen.add((nr, nc))
                queue.append((nr, nc))
    return len(seen) == int(mask.sum())


def _components_of_color(grid: np.ndarray, color: int) -> list[np.ndarray]:
    mask = grid == color
    h, w = grid.shape
    seen = np.zeros((h, w), dtype=bool)
    components: list[np.ndarray] = []
    for r in range(h):
        for c in range(w):
            if not mask[r, c] or seen[r, c]:
                continue
            queue = deque([(r, c)])
            seen[r, c] = True
            coords = []
            while queue:
                rr, cc = queue.popleft()
                coords.append((rr, cc))
                for dr, dc in DIR4:
                    nr, nc = rr + dr, cc + dc
                    if 0 <= nr < h and 0 <= nc < w and mask[nr, nc] and not seen[nr, nc]:
                        seen[nr, nc] = True
                        queue.append((nr, nc))
            components.append(np.array(coords, dtype=int))
    return components


def _crop_mask(mask: np.ndarray) -> np.ndarray:
    coords = np.argwhere(mask)
    r0, c0, r1, c1 = _bbox(coords)
    return mask[r0 : r1 + 1, c0 : c1 + 1].copy()


def _mask_signature(mask: np.ndarray) -> tuple[tuple[int, ...], ...]:
    return tuple(tuple(int(v) for v in row) for row in mask.astype(int))


def _find_divider(grid: np.ndarray, bg: int) -> tuple[int, int, int]:
    h, w = grid.shape
    row_candidates = [r for r in range(h) if int(grid[r, 0]) != bg and np.all(grid[r, :] == grid[r, 0])]
    col_candidates = [c for c in range(w) if int(grid[0, c]) != bg and np.all(grid[:, c] == grid[0, c])]
    assert (len(row_candidates) == 1) ^ (len(col_candidates) == 1), "expected a unique full divider line"
    if row_candidates:
        r = row_candidates[0]
        return 0, r, int(grid[r, 0])
    c = col_candidates[0]
    return 1, c, int(grid[0, c])


def _reflect_fragment(coords: np.ndarray, flip_axis: int) -> np.ndarray:
    r0, c0, r1, c1 = _bbox(coords)
    if flip_axis == 0:
        reflected = [(r1 - int(r), int(c) - c0) for r, c in coords]
    else:
        reflected = [(int(r) - r0, c1 - int(c)) for r, c in coords]
    return np.array(reflected, dtype=int)


def _touches(anchor_set: set[tuple[int, int]], fragment_set: set[tuple[int, int]]) -> bool:
    for r, c in fragment_set:
        for dr, dc in DIR4:
            if (r + dr, c + dc) in anchor_set:
                return True
    return False


def _complete_rectangle(
    anchor_coords: np.ndarray, fragment_coords: np.ndarray, side_shape: tuple[int, int], flip_axis: int
) -> np.ndarray | None:
    h, w = side_shape
    reflected = _reflect_fragment(fragment_coords, flip_axis)
    frag_h = int(reflected[:, 0].max()) + 1
    frag_w = int(reflected[:, 1].max()) + 1
    anchor_set = {tuple(map(int, rc)) for rc in anchor_coords}
    candidates: list[np.ndarray] = []
    for top in range(h - frag_h + 1):
        for left in range(w - frag_w + 1):
            shifted = reflected + np.array([top, left], dtype=int)
            shifted_set = {tuple(map(int, rc)) for rc in shifted}
            if anchor_set & shifted_set:
                continue
            if not _touches(anchor_set, shifted_set):
                continue
            union = anchor_set | shifted_set
            union_arr = np.array(list(union), dtype=int)
            r0, c0, r1, c1 = _bbox(union_arr)
            if (r1 - r0 + 1) * (c1 - c0 + 1) != len(union):
                continue
            candidates.append(shifted)
    if len(candidates) != 1:
        return None
    return candidates[0]


def _to_global(coords: np.ndarray, divider_axis: int, divider_index: int, side_index: int) -> np.ndarray:
    out = coords.copy()
    if divider_axis == 0 and side_index == 1:
        out[:, 0] += divider_index + 1
    if divider_axis == 1 and side_index == 1:
        out[:, 1] += divider_index + 1
    return out


def main(grid: np.ndarray) -> np.ndarray:
    """
    将散落的彩色碎片按照它们与分隔线的相对位置，拼接到分隔线对面的同色锚点区域上，补全锚点的矩形形状。
    """
    g = np.asarray(grid, dtype=int)
    assert g.ndim == 2 and g.size > 0

    bg = _mode_color(g)
    divider_axis, divider_index, divider_color = _find_divider(g, bg)
    if divider_axis == 0:
        sides = [g[:divider_index, :], g[divider_index + 1 :, :]]
        flip_axis = 0
    else:
        sides = [g[:, :divider_index], g[:, divider_index + 1 :]]
        flip_axis = 1

    colors = [int(c) for c in np.unique(g) if int(c) not in {bg, divider_color}]
    out = g.copy()

    for color in colors:
        comps0 = _components_of_color(sides[0], color)
        comps1 = _components_of_color(sides[1], color)
        assert len(comps0) == 1 and len(comps1) == 1, "each color must have one object on each side"

        options = []
        placed0 = _complete_rectangle(comps0[0], comps1[0], sides[0].shape, flip_axis)
        if placed0 is not None:
            options.append((0, 1, comps1[0], placed0))
        placed1 = _complete_rectangle(comps1[0], comps0[0], sides[1].shape, flip_axis)
        if placed1 is not None:
            options.append((1, 0, comps0[0], placed1))

        assert len(options) == 1, "anchor/fragment roles must be uniquely determined"
        anchor_side, fragment_side, fragment_source, placed = options[0]

        source_global = _to_global(fragment_source, divider_axis, divider_index, fragment_side)
        out[source_global[:, 0], source_global[:, 1]] = bg

        placed_global = _to_global(placed, divider_axis, divider_index, anchor_side)
        for r, c in placed_global:
            current = int(out[int(r), int(c)])
            assert current in {bg, color}, "placement cannot overwrite another object"
            out[int(r), int(c)] = color

    return out


def _edge_cells(h: int, w: int, side: str) -> list[tuple[int, int]]:
    if side == "top":
        return [(0, c) for c in range(w)]
    if side == "bottom":
        return [(h - 1, c) for c in range(w)]
    if side == "left":
        return [(r, 0) for r in range(h)]
    return [(r, w - 1) for r in range(h)]


def _generate_missing_mask(rect_h: int, rect_w: int, flip_axis: int) -> np.ndarray | None:
    max_missing = max(2, min(rect_h * rect_w - 3, rect_h * rect_w // 3 + 1))
    for _ in range(100):
        target_size = random.randint(2, max_missing)
        seed = random.choice(_edge_cells(rect_h, rect_w, random.choice(["top", "bottom", "left", "right"])))
        cells = {seed}
        while len(cells) < target_size:
            frontier = []
            for r, c in cells:
                for dr, dc in DIR4:
                    nr, nc = r + dr, c + dc
                    if 0 <= nr < rect_h and 0 <= nc < rect_w and (nr, nc) not in cells:
                        frontier.append((nr, nc))
            if not frontier:
                break
            cells.add(random.choice(frontier))
        if len(cells) != target_size:
            continue

        missing = np.zeros((rect_h, rect_w), dtype=bool)
        for r, c in cells:
            missing[r, c] = True
        anchor = ~missing
        if not _mask_connected(anchor):
            continue
        if int(anchor.sum()) < 3:
            continue
        if anchor[0, :].sum() == 0 or anchor[-1, :].sum() == 0:
            continue
        if anchor[:, 0].sum() == 0 or anchor[:, -1].sum() == 0:
            continue

        original = _mask_signature(_crop_mask(missing))
        flipped = _mask_signature(_crop_mask(np.flip(missing, axis=flip_axis)))
        if original == flipped:
            continue
        return missing
    return None


def _make_item(color: int, role: str, mask: np.ndarray) -> dict:
    return {"color": color, "role": role, "mask": mask}


def _plan_side(items: list[dict]) -> dict:
    ordered = items[:]
    random.shuffle(ordered)
    gaps = [random.choice(SIDE_GAP_CHOICES) for _ in range(max(0, len(ordered) - 1))]
    min_h = 2 * SIDE_BORDER + sum(int(item["mask"].shape[0]) for item in ordered) + sum(gaps)
    min_w = 2 * SIDE_BORDER + max(int(item["mask"].shape[1]) for item in ordered)
    return {"items": ordered, "gaps": gaps, "min_h": min_h, "min_w": min_w}


def _render_side(plan: dict, height: int, width: int, bg: int) -> tuple[np.ndarray, list[dict]]:
    side = np.full((height, width), bg, dtype=int)
    placements = []
    slack = height - plan["min_h"]
    y = SIDE_BORDER + random.randint(0, slack)
    for idx, item in enumerate(plan["items"]):
        mask = item["mask"]
        mh, mw = mask.shape
        x = random.randint(SIDE_BORDER, width - mw - SIDE_BORDER)
        side[y : y + mh, x : x + mw][mask] = item["color"]
        placements.append(
            {
                "color": item["color"],
                "role": item["role"],
                "top": y,
                "left": x,
                "height": mh,
                "width": mw,
            }
        )
        if idx < len(plan["gaps"]):
            y += mh + plan["gaps"][idx]
    return side, placements


def _gen_attempt() -> tuple[np.ndarray, np.ndarray] | None:
    bg = random.choice([c for c in range(10) if c != DIVIDER_COLOR])
    colors = random.sample([c for c in range(10) if c not in {bg, DIVIDER_COLOR}], random.randint(2, 4))
    divider_axis = random.randint(0, 1)
    flip_axis = divider_axis
    anchor_side = random.randint(0, 1)

    side_items = [[], []]
    anchor_shapes: dict[int, tuple[int, int]] = {}
    for color in colors:
        rect_h = random.randint(3, 5)
        rect_w = random.randint(3, 5)
        missing = _generate_missing_mask(rect_h, rect_w, flip_axis)
        if missing is None:
            return None
        anchor_mask = ~missing
        fragment_mask = _crop_mask(np.flip(missing, axis=flip_axis))
        side_items[anchor_side].append(_make_item(color, "anchor", anchor_mask))
        side_items[1 - anchor_side].append(_make_item(color, "fragment", fragment_mask))
        anchor_shapes[color] = anchor_mask.shape

    plans = [_plan_side(side_items[0]), _plan_side(side_items[1])]

    if divider_axis == 1:
        common_h = max(plans[0]["min_h"], plans[1]["min_h"]) + random.randint(0, 4)
        side_dims = [
            (common_h, plans[0]["min_w"] + random.randint(0, 4)),
            (common_h, plans[1]["min_w"] + random.randint(0, 4)),
        ]
        side0, place0 = _render_side(plans[0], side_dims[0][0], side_dims[0][1], bg)
        side1, place1 = _render_side(plans[1], side_dims[1][0], side_dims[1][1], bg)
        grid = np.full((common_h, side_dims[0][1] + 1 + side_dims[1][1]), bg, dtype=int)
        grid[:, side_dims[0][1]] = DIVIDER_COLOR
        grid[:, : side_dims[0][1]] = side0
        grid[:, side_dims[0][1] + 1 :] = side1
        offsets = [(0, 0), (0, side_dims[0][1] + 1)]
    else:
        common_w = max(plans[0]["min_w"], plans[1]["min_w"]) + random.randint(0, 4)
        side_dims = [
            (plans[0]["min_h"] + random.randint(0, 4), common_w),
            (plans[1]["min_h"] + random.randint(0, 4), common_w),
        ]
        side0, place0 = _render_side(plans[0], side_dims[0][0], side_dims[0][1], bg)
        side1, place1 = _render_side(plans[1], side_dims[1][0], side_dims[1][1], bg)
        grid = np.full((side_dims[0][0] + 1 + side_dims[1][0], common_w), bg, dtype=int)
        grid[side_dims[0][0], :] = DIVIDER_COLOR
        grid[: side_dims[0][0], :] = side0
        grid[side_dims[0][0] + 1 :, :] = side1
        offsets = [(0, 0), (side_dims[0][0] + 1, 0)]

    expected = grid.copy()
    for side_index, placements in enumerate([place0, place1]):
        row_off, col_off = offsets[side_index]
        for info in placements:
            color = info["color"]
            top = row_off + info["top"]
            left = col_off + info["left"]
            if info["role"] == "fragment":
                fragment = grid[top : top + info["height"], left : left + info["width"]] == color
                expected[top : top + info["height"], left : left + info["width"]][fragment] = bg
            else:
                rect_h, rect_w = anchor_shapes[color]
                expected[top : top + rect_h, left : left + rect_w] = color

    try:
        actual = main(grid)
    except AssertionError:
        return None
    if not np.array_equal(actual, expected):
        return None
    return grid, expected


def generate_input() -> np.ndarray:
    for _ in range(GEN_ATTEMPTS):
        result = _gen_attempt()
        if result is not None:
            return result[0]
    raise RuntimeError("failed to generate a valid input")
