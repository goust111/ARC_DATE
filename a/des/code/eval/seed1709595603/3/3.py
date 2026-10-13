from __future__ import annotations

import random
from collections import Counter, defaultdict, deque

import numpy as np


CONNECTOR_COLOR = 5
ALL_COLORS = tuple(range(10))
BLOCK_HEIGHT_RANGE = (1, 3)
BLOCK_WIDTH_RANGE = (1, 4)
GAP_RANGE = (1, 4)
MAX_GENERATION_ATTEMPTS = 400


def get_bg(grid: np.ndarray) -> int:
    values, counts = np.unique(grid, return_counts=True)
    return int(values[np.argmax(counts)])


def components_mask(mask: np.ndarray) -> list[np.ndarray]:
    h, w = mask.shape
    seen = np.zeros((h, w), dtype=bool)
    comps: list[np.ndarray] = []
    for r in range(h):
        for c in range(w):
            if not mask[r, c] or seen[r, c]:
                continue
            q = deque([(r, c)])
            seen[r, c] = True
            pts: list[tuple[int, int]] = []
            while q:
                cr, cc = q.popleft()
                pts.append((cr, cc))
                for nr, nc in ((cr - 1, cc), (cr + 1, cc), (cr, cc - 1), (cr, cc + 1)):
                    if 0 <= nr < h and 0 <= nc < w and mask[nr, nc] and not seen[nr, nc]:
                        seen[nr, nc] = True
                        q.append((nr, nc))
            comps.append(np.array(pts, dtype=int))
    return comps


def label_mask(mask: np.ndarray) -> tuple[np.ndarray, int]:
    h, w = mask.shape
    labels = np.zeros((h, w), dtype=int)
    comp_id = 0
    for r in range(h):
        for c in range(w):
            if not mask[r, c] or labels[r, c] != 0:
                continue
            comp_id += 1
            q = deque([(r, c)])
            labels[r, c] = comp_id
            while q:
                cr, cc = q.popleft()
                for nr, nc in ((cr - 1, cc), (cr + 1, cc), (cr, cc - 1), (cr, cc + 1)):
                    if 0 <= nr < h and 0 <= nc < w and mask[nr, nc] and labels[nr, nc] == 0:
                        labels[nr, nc] = comp_id
                        q.append((nr, nc))
    return labels, comp_id


def extract_blocks(grid: np.ndarray, bg: int) -> list[dict[str, int]]:
    blocks: list[dict[str, int]] = []
    for color in np.unique(grid):
        color = int(color)
        if color in (bg, CONNECTOR_COLOR):
            continue
        for pts in components_mask(grid == color):
            r0, c0 = pts.min(axis=0)
            r1, c1 = pts.max(axis=0)
            r0, c0, r1, c1 = map(int, (r0, c0, r1, c1))
            area = int(pts.shape[0])
            if area != (r1 - r0 + 1) * (c1 - c0 + 1):
                raise AssertionError("non-rectangular colored component")
            if not np.all(grid[r0 : r1 + 1, c0 : c1 + 1] == color):
                raise AssertionError("component bbox must be solid")
            blocks.append(
                {
                    "color": color,
                    "r0": r0,
                    "c0": c0,
                    "r1": r1,
                    "c1": c1,
                    "area": area,
                }
            )
    return blocks


def analyze_groups(grid: np.ndarray) -> tuple[int, list[dict[str, object]]]:
    bg = get_bg(grid)
    occ_labels, n_occ = label_mask(grid != bg)
    by_occ: dict[int, dict[str, object]] = {}
    for block in extract_blocks(grid, bg):
        occ_id = int(occ_labels[block["r0"], block["c0"]])
        if occ_id == 0:
            raise AssertionError("block must belong to a non-background component")
        if occ_id not in by_occ:
            by_occ[occ_id] = {"blocks": [], "has_connector": False}
        by_occ[occ_id]["blocks"].append(block)

    for occ_id in range(1, n_occ + 1):
        mask = occ_labels == occ_id
        if occ_id not in by_occ:
            continue
        by_occ[occ_id]["has_connector"] = bool(np.any((grid == CONNECTOR_COLOR) & mask))

    groups = [by_occ[occ_id] for occ_id in sorted(by_occ)]
    return bg, groups


def main(grid: np.ndarray) -> np.ndarray:
    """
    Group colored rectangles by grey-connector topology, then replace each connected
    group by its block-only bounding box filled with the largest block's color.
    """
    g = np.asarray(grid, dtype=int)
    bg, groups = analyze_groups(g)

    out = g.copy()
    out[g == CONNECTOR_COLOR] = bg

    for group in groups:
        blocks = list(group["blocks"])
        if len(blocks) < 2 or not bool(group["has_connector"]):
            continue
        winner = min(blocks, key=lambda b: (-int(b["area"]), int(b["r0"]), int(b["c0"])))
        r0 = min(int(block["r0"]) for block in blocks)
        c0 = min(int(block["c0"]) for block in blocks)
        r1 = max(int(block["r1"]) for block in blocks)
        c1 = max(int(block["c1"]) for block in blocks)
        out[r0 : r1 + 1, c0 : c1 + 1] = int(winner["color"])

    return out


def sample_colors(bg: int, count: int) -> list[int]:
    palette = [color for color in ALL_COLORS if color not in (bg, CONNECTOR_COLOR)]
    if count <= len(palette):
        return random.sample(palette, count)
    return [random.choice(palette) for _ in range(count)]


def sample_rect() -> tuple[int, int]:
    return (
        random.randint(*BLOCK_HEIGHT_RANGE),
        random.randint(*BLOCK_WIDTH_RANGE),
    )


def sample_unique_largest_specs(bg: int, count: int) -> list[dict[str, int]]:
    colors = sample_colors(bg, count)
    for _ in range(200):
        dims = [sample_rect() for _ in range(count)]
        areas = [h * w for h, w in dims]
        if areas.count(max(areas)) != 1:
            continue
        return [
            {"h": int(h), "w": int(w), "color": int(color)}
            for (h, w), color in zip(dims, colors)
        ]
    raise RuntimeError("failed to sample unique-largest rectangles")


def sample_tie_dims(require_tall: bool = False) -> tuple[int, int]:
    choices = [
        (h, w)
        for h in range(BLOCK_HEIGHT_RANGE[0], BLOCK_HEIGHT_RANGE[1] + 1)
        for w in range(BLOCK_WIDTH_RANGE[0], BLOCK_WIDTH_RANGE[1] + 1)
        if not require_tall or h >= 2
    ]
    return random.choice(choices)


def horizontal_connector(block_a: dict[str, int], block_b: dict[str, int]) -> list[tuple[int, int]]:
    left, right = sorted((block_a, block_b), key=lambda b: int(b["c"]))
    overlap_top = max(int(left["r"]), int(right["r"]))
    overlap_bottom = min(int(left["r"] + left["h"] - 1), int(right["r"] + right["h"] - 1))
    if overlap_top > overlap_bottom:
        raise AssertionError("horizontal connection needs row overlap")
    line_row = random.randint(overlap_top, overlap_bottom)
    start = int(left["c"] + left["w"])
    stop = int(right["c"])
    if start >= stop:
        raise AssertionError("horizontal connection needs a positive gap")
    return [(line_row, c) for c in range(start, stop)]


def vertical_connector(block_a: dict[str, int], block_b: dict[str, int]) -> list[tuple[int, int]]:
    top, bottom = sorted((block_a, block_b), key=lambda b: int(b["r"]))
    overlap_left = max(int(top["c"]), int(bottom["c"]))
    overlap_right = min(int(top["c"] + top["w"] - 1), int(bottom["c"] + bottom["w"] - 1))
    if overlap_left > overlap_right:
        raise AssertionError("vertical connection needs column overlap")
    line_col = random.randint(overlap_left, overlap_right)
    start = int(top["r"] + top["h"])
    stop = int(bottom["r"])
    if start >= stop:
        raise AssertionError("vertical connection needs a positive gap")
    return [(r, line_col) for r in range(start, stop)]


def rect_overlap(a: dict[str, int], b: dict[str, int]) -> bool:
    return not (
        int(a["r"]) + int(a["h"]) - 1 < int(b["r"])
        or int(b["r"]) + int(b["h"]) - 1 < int(a["r"])
        or int(a["c"]) + int(a["w"]) - 1 < int(b["c"])
        or int(b["c"]) + int(b["w"]) - 1 < int(a["c"])
    )


def render_item(
    blocks: list[dict[str, int]],
    connectors: list[list[tuple[int, int]]],
    bg: int,
    connected: bool,
) -> dict[str, np.ndarray]:
    if connected and len(blocks) < 2:
        raise AssertionError("connected item needs at least two blocks")
    if not connected and connectors:
        raise AssertionError("isolated item cannot contain connectors")

    for i in range(len(blocks)):
        for j in range(i + 1, len(blocks)):
            if rect_overlap(blocks[i], blocks[j]):
                raise AssertionError("blocks must not overlap")

    filled: dict[tuple[int, int], int] = {}
    for block in blocks:
        for r in range(int(block["r"]), int(block["r"]) + int(block["h"])):
            for c in range(int(block["c"]), int(block["c"]) + int(block["w"])):
                key = (r, c)
                if key in filled:
                    raise AssertionError("block overlap at rendered cell")
                filled[key] = int(block["color"])

    line_cells: list[tuple[int, int]] = []
    for connector in connectors:
        for cell in connector:
            if cell in filled:
                raise AssertionError("connector overlaps a block")
            line_cells.append(cell)
            filled[cell] = CONNECTOR_COLOR

    if not filled:
        raise AssertionError("item must paint something")

    rows = [r for r, _ in filled]
    cols = [c for _, c in filled]
    min_r, max_r = min(rows), max(rows)
    min_c, max_c = min(cols), max(cols)
    patch = np.full((max_r - min_r + 1, max_c - min_c + 1), bg, dtype=int)

    norm_blocks: list[dict[str, int]] = []
    for block in blocks:
        nr = int(block["r"]) - min_r
        nc = int(block["c"]) - min_c
        h = int(block["h"])
        w = int(block["w"])
        patch[nr : nr + h, nc : nc + w] = int(block["color"])
        norm_blocks.append({"r": nr, "c": nc, "h": h, "w": w, "color": int(block["color"])})

    for r, c in line_cells:
        patch[r - min_r, c - min_c] = CONNECTOR_COLOR

    block_extent_h = max(block["r"] + block["h"] for block in norm_blocks)
    block_extent_w = max(block["c"] + block["w"] for block in norm_blocks)
    if connected and patch.shape != (block_extent_h, block_extent_w):
        raise AssertionError("connectors must stay inside the block-only bounding box")

    if connected:
        winner = min(norm_blocks, key=lambda b: (-int(b["h"] * b["w"]), int(b["r"]), int(b["c"])))
        out_patch = np.full(patch.shape, int(winner["color"]), dtype=int)
    else:
        out_patch = patch.copy()

    return {"patch": patch, "out_patch": out_patch}


def make_isolated_item(bg: int) -> dict[str, np.ndarray]:
    h, w = sample_rect()
    color = sample_colors(bg, 1)[0]
    blocks = [{"r": 0, "c": 0, "h": h, "w": w, "color": color}]
    return render_item(blocks, [], bg, connected=False)


def make_tie_row_item(bg: int) -> dict[str, np.ndarray]:
    colors = sample_colors(bg, 2)
    gap = random.randint(*GAP_RANGE)
    if random.random() < 0.5:
        h, w = sample_tie_dims(require_tall=False)
        shift = random.randint(-(w - 1), w - 1)
        blocks = [
            {"r": 0, "c": 0, "h": h, "w": w, "color": colors[0]},
            {"r": h + gap, "c": shift, "h": h, "w": w, "color": colors[1]},
        ]
        connectors = [vertical_connector(blocks[0], blocks[1])]
    else:
        h, w = sample_tie_dims(require_tall=True)
        top_offset = random.randint(1, h - 1)
        blocks = [
            {"r": 0, "c": 0, "h": h, "w": w, "color": colors[0]},
            {"r": top_offset, "c": w + gap, "h": h, "w": w, "color": colors[1]},
        ]
        connectors = [horizontal_connector(blocks[0], blocks[1])]
    return render_item(blocks, connectors, bg, connected=True)


def make_tie_col_item(bg: int) -> dict[str, np.ndarray]:
    colors = sample_colors(bg, 2)
    h, w = sample_tie_dims(require_tall=False)
    gap = random.randint(*GAP_RANGE)
    blocks = [
        {"r": 0, "c": 0, "h": h, "w": w, "color": colors[0]},
        {"r": 0, "c": w + gap, "h": h, "w": w, "color": colors[1]},
    ]
    return render_item(blocks, [horizontal_connector(blocks[0], blocks[1])], bg, connected=True)


def make_pair_h_unique(bg: int) -> dict[str, np.ndarray]:
    specs = sample_unique_largest_specs(bg, 2)
    gap = random.randint(*GAP_RANGE)
    offset = random.randint(-specs[1]["h"] + 1, specs[0]["h"] - 1)
    blocks = [
        {"r": 0, "c": 0, **specs[0]},
        {"r": offset, "c": specs[0]["w"] + gap, **specs[1]},
    ]
    return render_item(blocks, [horizontal_connector(blocks[0], blocks[1])], bg, connected=True)


def make_pair_v_unique(bg: int) -> dict[str, np.ndarray]:
    specs = sample_unique_largest_specs(bg, 2)
    gap = random.randint(*GAP_RANGE)
    offset = random.randint(-specs[1]["w"] + 1, specs[0]["w"] - 1)
    blocks = [
        {"r": 0, "c": 0, **specs[0]},
        {"r": specs[0]["h"] + gap, "c": offset, **specs[1]},
    ]
    return render_item(blocks, [vertical_connector(blocks[0], blocks[1])], bg, connected=True)


def make_elbow3_unique(bg: int) -> dict[str, np.ndarray]:
    for _ in range(200):
        center, horiz, vert = sample_unique_largest_specs(bg, 3)
        gap_h = random.randint(*GAP_RANGE)
        gap_v = random.randint(*GAP_RANGE)
        side_h = random.choice([-1, 1])
        side_v = random.choice([-1, 1])

        block_center = {"r": 0, "c": 0, **center}
        block_h = {
            "r": random.randint(-horiz["h"] + 1, center["h"] - 1),
            "c": center["w"] + gap_h if side_h > 0 else -gap_h - horiz["w"],
            **horiz,
        }
        block_v = {
            "r": center["h"] + gap_v if side_v > 0 else -gap_v - vert["h"],
            "c": random.randint(-vert["w"] + 1, center["w"] - 1),
            **vert,
        }
        blocks = [block_center, block_h, block_v]
        if rect_overlap(block_h, block_v):
            continue
        connectors = [
            horizontal_connector(block_center, block_h),
            vertical_connector(block_center, block_v),
        ]
        return render_item(blocks, connectors, bg, connected=True)
    raise RuntimeError("failed to build elbow group")


def make_tee4_unique(bg: int) -> dict[str, np.ndarray]:
    for _ in range(200):
        center, left, right, vertical = sample_unique_largest_specs(bg, 4)
        gap_left = random.randint(*GAP_RANGE)
        gap_right = random.randint(*GAP_RANGE)
        gap_vert = random.randint(*GAP_RANGE)
        side_v = random.choice([-1, 1])

        block_center = {"r": 0, "c": 0, **center}
        block_left = {
            "r": random.randint(-left["h"] + 1, center["h"] - 1),
            "c": -gap_left - left["w"],
            **left,
        }
        block_right = {
            "r": random.randint(-right["h"] + 1, center["h"] - 1),
            "c": center["w"] + gap_right,
            **right,
        }
        block_vert = {
            "r": center["h"] + gap_vert if side_v > 0 else -gap_vert - vertical["h"],
            "c": random.randint(-vertical["w"] + 1, center["w"] - 1),
            **vertical,
        }
        blocks = [block_center, block_left, block_right, block_vert]
        if rect_overlap(block_left, block_vert) or rect_overlap(block_right, block_vert):
            continue
        connectors = [
            horizontal_connector(block_center, block_left),
            horizontal_connector(block_center, block_right),
            vertical_connector(block_center, block_vert),
        ]
        return render_item(blocks, connectors, bg, connected=True)
    raise RuntimeError("failed to build tee group")


def make_random_group(bg: int) -> dict[str, np.ndarray]:
    return random.choice(
        [
            make_pair_h_unique,
            make_pair_v_unique,
            make_elbow3_unique,
            make_tee4_unique,
        ]
    )(bg)


def pack_items(items: list[dict[str, np.ndarray]], bg: int) -> tuple[np.ndarray, np.ndarray] | None:
    shuffled = items[:]
    random.shuffle(shuffled)
    shelf_count = random.randint(1, min(3, len(shuffled)))
    shelves: list[list[dict[str, np.ndarray]]] = [[] for _ in range(shelf_count)]
    widths = [0] * shelf_count

    for item in shuffled:
        idxs = sorted(range(shelf_count), key=lambda i: widths[i])
        choice = random.choice(idxs[: min(2, shelf_count)])
        shelves[choice].append(item)
        widths[choice] += int(item["patch"].shape[1])

    shelf_specs: list[dict[str, object]] = []
    for shelf in shelves:
        if not shelf:
            continue
        random.shuffle(shelf)
        item_gaps = [random.randint(2, 4) for _ in range(max(0, len(shelf) - 1))]
        shelf_width = sum(int(item["patch"].shape[1]) for item in shelf) + sum(item_gaps)
        shelf_height = max(int(item["patch"].shape[0]) for item in shelf)
        shelf_specs.append(
            {
                "items": shelf,
                "gaps": item_gaps,
                "width": shelf_width,
                "height": shelf_height,
            }
        )

    if not shelf_specs:
        return None

    top = random.randint(1, 3)
    bottom = random.randint(1, 3)
    left = random.randint(1, 3)
    right = random.randint(1, 3)
    row_gaps = [random.randint(2, 4) for _ in range(max(0, len(shelf_specs) - 1))]

    width = left + max(int(spec["width"]) for spec in shelf_specs) + right
    height = top + sum(int(spec["height"]) for spec in shelf_specs) + sum(row_gaps) + bottom

    if width > 30 or height > 30:
        return None
    if width < 12 or height < 12:
        return None

    total_fg = sum(int(np.count_nonzero(item["patch"] != bg)) for item in shuffled)
    bg_fraction = 1.0 - total_fg / float(height * width)
    if bg_fraction < 0.45:
        return None

    grid = np.full((height, width), bg, dtype=int)
    expected = np.full((height, width), bg, dtype=int)

    y = top
    for shelf_idx, spec in enumerate(shelf_specs):
        shelf_width = int(spec["width"])
        start_x = random.randint(left, width - right - shelf_width)
        x = start_x
        items_in_shelf = list(spec["items"])
        gaps = list(spec["gaps"])
        for item_idx, item in enumerate(items_in_shelf):
            patch = np.asarray(item["patch"], dtype=int)
            out_patch = np.asarray(item["out_patch"], dtype=int)
            ph, pw = patch.shape
            grid[y : y + ph, x : x + pw] = patch
            expected[y : y + ph, x : x + pw] = out_patch
            if item_idx < len(gaps):
                x += pw + int(gaps[item_idx])
        y += int(spec["height"])
        if shelf_idx < len(row_gaps):
            y += int(row_gaps[shelf_idx])

    return grid, expected


def _gen_attempt() -> np.ndarray | None:
    bg = random.choice([color for color in ALL_COLORS if color != CONNECTOR_COLOR])
    tie_mode = random.choice(["none", "row", "col"])
    group_count = random.randint(1, 3)
    include_isolated = random.random() < 0.5
    isolated_count = random.randint(1, 2) if include_isolated else 0

    items: list[dict[str, np.ndarray]] = []
    if tie_mode == "row":
        items.append(make_tie_row_item(bg))
        remaining_groups = group_count - 1
    elif tie_mode == "col":
        items.append(make_tie_col_item(bg))
        remaining_groups = group_count - 1
    else:
        remaining_groups = group_count

    for _ in range(remaining_groups):
        items.append(make_random_group(bg))
    for _ in range(isolated_count):
        items.append(make_isolated_item(bg))

    packed = pack_items(items, bg)
    if packed is None:
        return None
    grid, expected = packed

    if np.array_equal(grid, expected):
        return None
    if get_bg(grid) != bg:
        return None
    if not np.array_equal(main(grid), expected):
        return None

    _, groups = analyze_groups(grid)
    connected_groups = [
        group for group in groups if bool(group["has_connector"]) and len(group["blocks"]) >= 2
    ]
    isolated_groups = [
        group for group in groups if not bool(group["has_connector"]) and len(group["blocks"]) == 1
    ]
    if not connected_groups:
        return None
    if include_isolated and not isolated_groups:
        return None

    return grid


def generate_input() -> np.ndarray:
    for _ in range(MAX_GENERATION_ATTEMPTS):
        grid = _gen_attempt()
        if grid is not None:
            return grid
    raise RuntimeError("failed to generate valid input")
