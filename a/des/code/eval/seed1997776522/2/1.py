from __future__ import annotations

import random

import numpy as np


BG = 0
DIR4 = ((1, 0), (-1, 0), (0, 1), (0, -1))


def _neighbors4(r: int, c: int, h: int, w: int):
    for dr, dc in DIR4:
        rr = r + dr
        cc = c + dc
        if 0 <= rr < h and 0 <= cc < w:
            yield rr, cc


def _find_components(grid: np.ndarray) -> list[dict]:
    h, w = map(int, grid.shape)
    seen = np.zeros((h, w), dtype=bool)
    components: list[dict] = []

    for r in range(h):
        for c in range(w):
            color = int(grid[r, c])
            if color == BG or seen[r, c]:
                continue

            stack = [(r, c)]
            seen[r, c] = True
            cells: list[tuple[int, int]] = []
            while stack:
                rr, cc = stack.pop()
                cells.append((rr, cc))
                for nr, nc in _neighbors4(rr, cc, h, w):
                    if not seen[nr, nc] and int(grid[nr, nc]) == color:
                        seen[nr, nc] = True
                        stack.append((nr, nc))

            coords = np.array(cells, dtype=int)
            r0, c0 = coords.min(axis=0)
            r1, c1 = coords.max(axis=0)
            components.append(
                {
                    "color": color,
                    "coords": coords,
                    "bbox": (int(r0), int(c0), int(r1), int(c1)),
                    "area": int(len(cells)),
                }
            )

    return components


def _component_mask(component: dict) -> np.ndarray:
    r0, c0, r1, c1 = component["bbox"]
    mask = np.zeros((r1 - r0 + 1, c1 - c0 + 1), dtype=bool)
    coords = component["coords"]
    mask[coords[:, 0] - r0, coords[:, 1] - c0] = True
    return mask


def _is_frame(component: dict) -> bool:
    r0, c0, r1, c1 = component["bbox"]
    h = r1 - r0 + 1
    w = c1 - c0 + 1
    if h < 3 or w < 3:
        return False
    if int(component["area"]) != 2 * (h + w) - 4:
        return False

    mask = _component_mask(component)
    border = np.zeros((h, w), dtype=bool)
    border[0, :] = True
    border[-1, :] = True
    border[:, 0] = True
    border[:, -1] = True
    return bool(np.array_equal(mask, border))


def _is_legend_block(component: dict) -> bool:
    r0, c0, r1, c1 = component["bbox"]
    return int(component["area"]) == 4 and (r1 - r0 + 1) == 2 and (c1 - c0 + 1) == 2


def _contains(outer: dict, inner: dict) -> bool:
    r0, c0, r1, c1 = outer["bbox"]
    rr0, cc0, rr1, cc1 = inner["bbox"]
    return r0 < rr0 and c0 < cc0 and r1 > rr1 and c1 > cc1


def main(grid: np.ndarray) -> np.ndarray:
    """
    Recolor each nested rectangular-frame group according to its nesting depth,
    using the right-side vertical legend, then erase the legend.
    """
    g = np.asarray(grid, dtype=int)
    assert g.ndim == 2

    components = _find_components(g)
    legend_blocks = [component for component in components if _is_legend_block(component)]
    assert 1 <= len(legend_blocks) <= 4, "expected 1 to 4 legend blocks"

    legend_blocks.sort(key=lambda component: (component["bbox"][0], component["bbox"][1]))
    legend_col = legend_blocks[0]["bbox"][1]
    for i, component in enumerate(legend_blocks):
        r0, c0, _r1, _c1 = component["bbox"]
        assert c0 == legend_col, "legend blocks must form one vertical column"
        if i:
            prev_r0 = legend_blocks[i - 1]["bbox"][0]
            assert r0 - prev_r0 == 3, "legend blocks must be separated by one background row"

    legend_colors = [int(component["color"]) for component in legend_blocks]
    frames = [component for component in components if _is_frame(component)]
    assert frames, "expected at least one hollow rectangular frame"

    parents = [-1] * len(frames)
    frame_areas = []
    for component in frames:
        r0, c0, r1, c1 = component["bbox"]
        frame_areas.append((r1 - r0 + 1) * (c1 - c0 + 1))

    for i, frame in enumerate(frames):
        candidates = [j for j, other in enumerate(frames) if i != j and _contains(other, frame)]
        if candidates:
            parents[i] = min(candidates, key=lambda j: frame_areas[j])

    children = [[] for _ in frames]
    for child, parent in enumerate(parents):
        if parent != -1:
            children[parent].append(child)

    depth_cache: dict[int, int] = {}

    def depth_to_leaf(idx: int) -> int:
        if idx not in depth_cache:
            depth_cache[idx] = 1 + max((depth_to_leaf(child) for child in children[idx]), default=0)
        return depth_cache[idx]

    root_cache: dict[int, int] = {}

    def root_of(idx: int) -> int:
        if idx not in root_cache:
            cur = idx
            while parents[cur] != -1:
                cur = parents[cur]
            root_cache[idx] = cur
        return root_cache[idx]

    out = g.copy()
    for idx, frame in enumerate(frames):
        depth = depth_to_leaf(root_of(idx))
        assert depth <= len(legend_colors), "legend must cover every nesting depth"
        color = legend_colors[depth - 1]
        coords = frame["coords"]
        out[coords[:, 0], coords[:, 1]] = color

    for component in legend_blocks:
        coords = component["coords"]
        out[coords[:, 0], coords[:, 1]] = BG

    return out


# --------------- generator ---------------

GRID_H_RANGE = (18, 30)
GRID_W_RANGE = (18, 30)
LEGEND_GAP_RANGE = (2, 4)
GROUP_COUNT_BY_DEPTH = {1: (1, 3), 2: (1, 3), 3: (1, 2), 4: (1, 2)}
GROUP_SIDE_LIMIT = {1: 10, 2: 15, 3: 17, 4: 19}
INNER_SIDE_MAX = {2: 7, 3: 6, 4: 5}
PLACEMENT_RETRIES = 200
GENERATION_RETRIES = 500


def _draw_frame(grid: np.ndarray, box: tuple[int, int, int, int], color: int) -> None:
    r0, c0, r1, c1 = map(int, box)
    grid[r0, c0 : c1 + 1] = color
    grid[r1, c0 : c1 + 1] = color
    grid[r0 : r1 + 1, c0] = color
    grid[r0 : r1 + 1, c1] = color


def _shift_box(box: tuple[int, int, int, int], dr: int, dc: int) -> tuple[int, int, int, int]:
    r0, c0, r1, c1 = box
    return (r0 + dr, c0 + dc, r1 + dr, c1 + dc)


def _inflate_box(box: tuple[int, int, int, int], pad: int) -> tuple[int, int, int, int]:
    r0, c0, r1, c1 = box
    return (r0 - pad, c0 - pad, r1 + pad, c1 + pad)


def _boxes_overlap(a: tuple[int, int, int, int], b: tuple[int, int, int, int]) -> bool:
    return not (a[2] < b[0] or b[2] < a[0] or a[3] < b[1] or b[3] < a[1])


def _sample_group(depth: int) -> dict | None:
    if depth == 1:
        h = random.randint(3, GROUP_SIDE_LIMIT[1])
        w = random.randint(3, GROUP_SIDE_LIMIT[1])
        return {"depth": 1, "shape": (h, w), "boxes": [(0, 0, h - 1, w - 1)]}

    for _ in range(120):
        inner_h = random.randint(3, INNER_SIDE_MAX[depth])
        inner_w = random.randint(3, INNER_SIDE_MAX[depth])
        row_offsets = [random.randint(2, 3) for _ in range(depth - 1)]
        col_offsets = [random.randint(2, 3) for _ in range(depth - 1)]
        outer_h = inner_h + 2 * sum(row_offsets)
        outer_w = inner_w + 2 * sum(col_offsets)
        if outer_h > GROUP_SIDE_LIMIT[depth] or outer_w > GROUP_SIDE_LIMIT[depth]:
            continue

        boxes: list[tuple[int, int, int, int]] = [(0, 0, outer_h - 1, outer_w - 1)]
        cur = boxes[0]
        for dr, dc in zip(row_offsets, col_offsets):
            cur = (cur[0] + dr, cur[1] + dc, cur[2] - dr, cur[3] - dc)
            boxes.append(cur)
        return {"depth": depth, "shape": (outer_h, outer_w), "boxes": boxes}

    return None


def _build_expected(
    inp: np.ndarray,
    placed_groups: list[dict],
    legend_colors: list[int],
    legend_boxes: list[tuple[int, int, int, int]],
) -> np.ndarray:
    out = inp.copy()
    for group in placed_groups:
        color = int(legend_colors[group["depth"] - 1])
        for box in group["abs_boxes"]:
            _draw_frame(out, box, color)
    for box in legend_boxes:
        r0, c0, r1, c1 = box
        out[r0 : r1 + 1, c0 : c1 + 1] = BG
    return out


def _place_groups(
    grid_h: int,
    grid_w: int,
    specs: list[dict],
    legend_col: int,
    legend_gap: int,
) -> list[dict] | None:
    occupied: list[tuple[int, int, int, int]] = []
    placed: list[dict] = []
    main_right = legend_col - legend_gap - 1

    for spec in specs:
        outer_h, outer_w = spec["shape"]
        if outer_h > grid_h - 2 or outer_w > main_right - 1:
            return None

        max_r0 = grid_h - outer_h - 1
        max_c0 = main_right - outer_w
        if max_r0 < 1 or max_c0 < 1:
            return None

        for _ in range(PLACEMENT_RETRIES):
            r0 = random.randint(1, max_r0)
            c0 = random.randint(1, max_c0)
            outer_box = (r0, c0, r0 + outer_h - 1, c0 + outer_w - 1)
            padded = _inflate_box(outer_box, 1)
            if any(_boxes_overlap(padded, other) for other in occupied):
                continue

            placed_spec = dict(spec)
            placed_spec["abs_boxes"] = [_shift_box(box, r0, c0) for box in spec["boxes"]]
            placed.append(placed_spec)
            occupied.append(padded)
            break
        else:
            return None

    return placed


def _gen_attempt() -> np.ndarray | None:
    legend_len = random.randint(1, 4)
    min_groups, max_groups = GROUP_COUNT_BY_DEPTH[legend_len]
    group_count = random.randint(min_groups, max_groups)

    depths = [legend_len] + [random.randint(1, legend_len) for _ in range(group_count - 1)]
    random.shuffle(depths)

    specs: list[dict] = []
    for depth in depths:
        spec = _sample_group(depth)
        if spec is None:
            return None
        specs.append(spec)

    legend_gap = random.randint(*LEGEND_GAP_RANGE)
    legend_h = 3 * legend_len - 1
    max_group_h = max(spec["shape"][0] for spec in specs)
    max_group_w = max(spec["shape"][1] for spec in specs)

    min_grid_h = max(GRID_H_RANGE[0], legend_h + 1, max_group_h + 2)
    min_grid_w = max(GRID_W_RANGE[0], max_group_w + legend_gap + 4)
    if min_grid_h > GRID_H_RANGE[1] or min_grid_w > GRID_W_RANGE[1]:
        return None

    grid_h = random.randint(min_grid_h, GRID_H_RANGE[1])
    grid_w = random.randint(min_grid_w, GRID_W_RANGE[1])
    legend_col = grid_w - 2
    legend_top = random.randint(0, grid_h - legend_h)

    colors = list(range(1, 10))
    legend_colors = random.sample(colors, legend_len)
    source_pool = [color for color in colors if color not in legend_colors]
    source_count = random.randint(2, min(4, len(source_pool)))
    source_colors = random.sample(source_pool, source_count)

    placed_groups = _place_groups(grid_h, grid_w, specs, legend_col, legend_gap)
    if placed_groups is None:
        return None

    inp = np.zeros((grid_h, grid_w), dtype=int)
    legend_boxes: list[tuple[int, int, int, int]] = []
    for i, color in enumerate(legend_colors):
        r0 = legend_top + 3 * i
        box = (r0, legend_col, r0 + 1, legend_col + 1)
        legend_boxes.append(box)
        inp[r0 : r0 + 2, legend_col : legend_col + 2] = color

    for group in placed_groups:
        if random.random() < 0.5:
            color_seq = [random.choice(source_colors)] * group["depth"]
        else:
            color_seq = [random.choice(source_colors) for _ in range(group["depth"])]
        for box, color in zip(group["abs_boxes"], color_seq):
            _draw_frame(inp, box, int(color))

    expected = _build_expected(inp, placed_groups, legend_colors, legend_boxes)
    try:
        if not np.array_equal(main(inp), expected):
            return None
    except Exception:
        return None
    return inp


def generate_input() -> np.ndarray:
    for _ in range(GENERATION_RETRIES):
        grid = _gen_attempt()
        if grid is not None:
            return grid
    raise RuntimeError("failed to generate valid nested-frame legend input")
