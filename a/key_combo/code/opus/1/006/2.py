from __future__ import annotations

import random
from collections import deque

import numpy as np


DIR4 = ((1, 0), (-1, 0), (0, 1), (0, -1))
NON_BLACK = tuple(range(1, 10))
MAX_GEN_ATTEMPTS = 400


def _neighbors(r: int, c: int, h: int, w: int):
    for dr, dc in DIR4:
        nr, nc = r + dr, c + dc
        if 0 <= nr < h and 0 <= nc < w:
            yield int(nr), int(nc)


def _random_split(total: int, parts: int) -> list[int]:
    values = [0] * parts
    for _ in range(total):
        values[random.randrange(parts)] += 1
    return values


def _same_color_components(grid: np.ndarray) -> list[tuple[int, np.ndarray]]:
    h, w = grid.shape
    seen = np.zeros((h, w), dtype=bool)
    components: list[tuple[int, np.ndarray]] = []

    for r in range(h):
        for c in range(w):
            color = int(grid[r, c])
            if color == 0 or seen[r, c]:
                continue
            seen[r, c] = True
            queue = deque([(r, c)])
            cells = []
            while queue:
                cr, cc = queue.popleft()
                cells.append((cr, cc))
                for nr, nc in _neighbors(cr, cc, h, w):
                    if not seen[nr, nc] and int(grid[nr, nc]) == color:
                        seen[nr, nc] = True
                        queue.append((nr, nc))
            components.append((color, np.array(cells, dtype=int)))
    return components


def _frame_from_component(grid: np.ndarray, color: int, coords: np.ndarray) -> dict | None:
    rows = coords[:, 0]
    cols = coords[:, 1]
    r0, r1 = int(rows.min()), int(rows.max())
    c0, c1 = int(cols.min()), int(cols.max())
    h = r1 - r0 + 1
    w = c1 - c0 + 1
    if h < 3 or w < 3:
        return None

    border = set()
    for c in range(c0, c1 + 1):
        border.add((r0, c))
        border.add((r1, c))
    for r in range(r0, r1 + 1):
        border.add((r, c0))
        border.add((r, c1))

    pixels = {tuple(map(int, cell)) for cell in coords}
    if not pixels.issubset(border):
        return None
    missing = list(border - pixels)
    if len(missing) != 1:
        return None
    gap = missing[0]
    if int(grid[gap]) != 0 or len(pixels) != len(border) - 1:
        return None

    return {
        "color": int(color),
        "bbox": (r0, c0, r1, c1),
        "gap": gap,
        "area": h * w,
    }


def _parse_scene(grid: np.ndarray) -> tuple[list[dict], list[dict]]:
    frames: list[dict] = []
    seeds: list[dict] = []
    for color, coords in _same_color_components(grid):
        if int(coords.shape[0]) == 1:
            seeds.append({"color": int(color), "pos": tuple(map(int, coords[0]))})
            continue
        frame = _frame_from_component(grid, color, coords)
        assert frame is not None, "non-seed objects must be single-gap rectangular frames"
        frames.append(frame)
    assert frames, "expected at least one frame"
    assert seeds, "expected at least one seed"
    return frames, seeds


def _inside_mask(frames: list[dict], shape: tuple[int, int]) -> np.ndarray:
    mask = np.zeros(shape, dtype=bool)
    for frame in frames:
        r0, c0, r1, c1 = frame["bbox"]
        if r1 - r0 >= 2 and c1 - c0 >= 2:
            mask[r0 + 1 : r1, c0 + 1 : c1] = True
    return mask


def _zero_components_with_seed_sets(grid: np.ndarray, seeds: list[dict]) -> list[tuple[list[tuple[int, int]], set[int]]]:
    h, w = grid.shape
    seed_lookup = {seed["pos"]: int(seed["color"]) for seed in seeds}
    seen = np.zeros((h, w), dtype=bool)
    pieces: list[tuple[list[tuple[int, int]], set[int]]] = []

    for r in range(h):
        for c in range(w):
            if int(grid[r, c]) != 0 or seen[r, c]:
                continue
            seen[r, c] = True
            queue = deque([(r, c)])
            cells: list[tuple[int, int]] = []
            seed_colors: set[int] = set()
            while queue:
                cr, cc = queue.popleft()
                cells.append((cr, cc))
                for nr, nc in _neighbors(cr, cc, h, w):
                    value = int(grid[nr, nc])
                    if value == 0 and not seen[nr, nc]:
                        seen[nr, nc] = True
                        queue.append((nr, nc))
                    elif (nr, nc) in seed_lookup:
                        seed_colors.add(seed_lookup[(nr, nc)])
            pieces.append((cells, seed_colors))
    return pieces


def main(grid: np.ndarray) -> np.ndarray:
    """
    Recolor the black regions inside nested single-gap frames according to which outside seed can actually flood into them.

    Input structure:
    - A black background contains 2-4 monochrome rectangular frames, each with a single black gap on its border.
    - Some frames are strictly nested inside others.
    - 2-3 single-pixel seeds of distinct colors lie outside all frames.

    Rule:
    - Identify every rectangular frame and treat only black cells as floodable space.
    - Start a 4-connected flood from each seed through adjacent black cells; non-black frame pixels remain solid barriers, and a frame can only be crossed through its black gap.
    - For every black connected region that is adjacent to exactly one seed, recolor only the cells of that region that lie inside at least one frame with that seed's color.
    - Leave unreached interior black cells black, and keep all frame pixels and seed pixels unchanged.

    Key insight:
    - Reachability is topological: nested interiors are colored by the seed whose flood can pass through the whole gap chain, not by frame color or containment depth.
    """
    g = np.asarray(grid, dtype=int)
    assert g.ndim == 2, "expected a 2D grid"

    frames, seeds = _parse_scene(g)
    inside = _inside_mask(frames, g.shape)
    out = g.copy()

    for cells, seed_colors in _zero_components_with_seed_sets(g, seeds):
        if len(seed_colors) != 1:
            continue
        fill = next(iter(seed_colors))
        for r, c in cells:
            if inside[r, c]:
                out[r, c] = fill

    return out


def _oracle_output(grid: np.ndarray) -> np.ndarray:
    frames, seeds = _parse_scene(grid)
    inside = _inside_mask(frames, grid.shape)
    h, w = grid.shape

    reached_by: dict[tuple[int, int], set[int]] = {}
    for seed in seeds:
        color = int(seed["color"])
        sr, sc = seed["pos"]
        seen = np.zeros((h, w), dtype=bool)
        queue = deque()
        for nr, nc in _neighbors(sr, sc, h, w):
            if int(grid[nr, nc]) == 0 and not seen[nr, nc]:
                seen[nr, nc] = True
                queue.append((nr, nc))
        while queue:
            r, c = queue.popleft()
            reached_by.setdefault((r, c), set()).add(color)
            for nr, nc in _neighbors(r, c, h, w):
                if int(grid[nr, nc]) == 0 and not seen[nr, nc]:
                    seen[nr, nc] = True
                    queue.append((nr, nc))

    out = np.asarray(grid, dtype=int).copy()
    for (r, c), colors in reached_by.items():
        if inside[r, c] and len(colors) == 1:
            out[r, c] = next(iter(colors))
    return out


def _side_names(orientation: str) -> tuple[str, str]:
    return ("left", "right") if orientation == "vertical" else ("top", "bottom")


def _opposite_side(orientation: str, side: str) -> str:
    first, second = _side_names(orientation)
    return second if side == first else first


def _side_strip_index(root_index: int, orientation: str, side: str) -> int:
    first, second = _side_names(orientation)
    assert side in {first, second}
    return root_index if side == first else root_index + 1


def _min_span(length: int) -> int:
    return 6 + 2 * (length - 1)


def _min_thickness(length: int) -> int:
    return 6 + 4 * (length - 1)


def _choose_lengths() -> list[int]:
    two_root_options = ([2, 1], [1, 2], [2, 2], [3, 1], [1, 3])
    one_root_options = ([2], [3], [4])
    options = two_root_options if random.random() < 0.65 else one_root_options
    return list(random.choice(options))


def _build_vertical_layout(lengths: list[int]) -> dict | None:
    root_min_widths = [_min_thickness(length) for length in lengths]
    min_width = sum(root_min_widths) + 2 * (len(lengths) + 1)
    if min_width > 30:
        return None

    max_length = max(lengths)
    min_height = max(12, _min_span(max_length))
    max_height = min(24, min_height + 8)
    height = random.randint(min_height, max_height)

    extra = random.randint(0, min(30 - min_width, 8))
    split = _random_split(extra, len(root_min_widths) + len(lengths) + 1)
    root_widths = [base + split[i] for i, base in enumerate(root_min_widths)]
    strip_widths = [2 + split[len(root_min_widths) + i] for i in range(len(lengths) + 1)]

    roots = []
    strips = []
    col = 0
    for idx, width in enumerate(root_widths):
        strips.append((col, col + strip_widths[idx] - 1))
        col += strip_widths[idx]
        roots.append((0, col, height - 1, col + width - 1))
        col += width
    strips.append((col, col + strip_widths[-1] - 1))
    width = col + strip_widths[-1]

    return {"orientation": "vertical", "shape": (height, width), "roots": roots, "strips": strips}


def _build_horizontal_layout(lengths: list[int]) -> dict | None:
    root_min_heights = [_min_thickness(length) for length in lengths]
    min_height = sum(root_min_heights) + 2 * (len(lengths) + 1)
    if min_height > 30:
        return None

    max_length = max(lengths)
    min_width = max(12, _min_span(max_length))
    max_width = min(24, min_width + 8)
    width = random.randint(min_width, max_width)

    extra = random.randint(0, min(30 - min_height, 8))
    split = _random_split(extra, len(root_min_heights) + len(lengths) + 1)
    root_heights = [base + split[i] for i, base in enumerate(root_min_heights)]
    strip_heights = [2 + split[len(root_min_heights) + i] for i in range(len(lengths) + 1)]

    roots = []
    strips = []
    row = 0
    for idx, height in enumerate(root_heights):
        strips.append((row, row + strip_heights[idx] - 1))
        row += strip_heights[idx]
        roots.append((row, 0, row + height - 1, width - 1))
        row += height
    strips.append((row, row + strip_heights[-1] - 1))
    height = row + strip_heights[-1]

    return {"orientation": "horizontal", "shape": (height, width), "roots": roots, "strips": strips}


def _build_layout(orientation: str, lengths: list[int]) -> dict | None:
    if orientation == "vertical":
        return _build_vertical_layout(lengths)
    return _build_horizontal_layout(lengths)


def _child_box(parent: tuple[int, int, int, int], orientation: str, child_length: int) -> tuple[int, int, int, int]:
    r0, c0, r1, c1 = parent
    height = r1 - r0 + 1
    width = c1 - c0 + 1

    if orientation == "vertical":
        child_width_min = _min_thickness(child_length)
        child_width_max = width - 4
        assert child_width_min <= child_width_max
        child_width = random.randint(child_width_min, child_width_max)
        side_space = width - 2 - child_width
        left_gap = random.randint(1, side_space - 1)
        child_c0 = c0 + 1 + left_gap
        child_c1 = child_c0 + child_width - 1
        return (r0 + 1, child_c0, r1 - 1, child_c1)

    child_height_min = _min_thickness(child_length)
    child_height_max = height - 4
    assert child_height_min <= child_height_max
    child_height = random.randint(child_height_min, child_height_max)
    side_space = height - 2 - child_height
    top_gap = random.randint(1, side_space - 1)
    child_r0 = r0 + 1 + top_gap
    child_r1 = child_r0 + child_height - 1
    return (child_r0, c0 + 1, child_r1, c1 - 1)


def _build_chain(root_box: tuple[int, int, int, int], orientation: str, length: int) -> list[tuple[int, int, int, int]]:
    boxes = [root_box]
    parent = root_box
    for depth in range(1, length):
        child_length = length - depth
        parent = _child_box(parent, orientation, child_length)
        boxes.append(parent)
    return boxes


def _choose_seed_strips(n_strips: int) -> list[int]:
    if n_strips == 2:
        return [0, 1]
    if random.random() < 0.45:
        return sorted(random.sample(range(n_strips), 2))
    return list(range(n_strips))


def _place_seeds(layout: dict, seed_strips: list[int], colors: list[int]) -> dict[int, dict]:
    h, w = layout["shape"]
    orientation = layout["orientation"]
    seeds: dict[int, dict] = {}
    for strip_idx, color in zip(seed_strips, colors):
        start, end = layout["strips"][strip_idx]
        if orientation == "vertical":
            pos = (random.randint(0, h - 1), random.randint(start, end))
        else:
            pos = (random.randint(start, end), random.randint(0, w - 1))
        seeds[int(strip_idx)] = {"color": int(color), "pos": pos}
    return seeds


def _plan_chains(layout: dict, lengths: list[int], seed_by_strip: dict[int, dict]) -> list[dict]:
    orientation = layout["orientation"]
    roots = layout["roots"]
    side_a, side_b = _side_names(orientation)

    blocked_idx = random.choice([idx for idx, length in enumerate(lengths) if length > 1])
    open_candidates = [
        idx
        for idx in range(len(lengths))
        if any(strip in seed_by_strip for strip in (idx, idx + 1))
    ]
    open_idx = random.choice(open_candidates)

    plans = []
    for idx, (root, length) in enumerate(zip(roots, lengths)):
        seeded_sides = [
            side
            for side in (side_a, side_b)
            if _side_strip_index(idx, orientation, side) in seed_by_strip
        ]
        unseeded_sides = [side for side in (side_a, side_b) if side not in seeded_sides]
        assert seeded_sides or unseeded_sides

        if idx == blocked_idx == open_idx:
            root_side = random.choice(seeded_sides)
            reachable_depth = random.randint(1, length - 1)
        elif idx == blocked_idx:
            if seeded_sides and (not unseeded_sides or random.random() < 0.7):
                root_side = random.choice(seeded_sides)
                reachable_depth = random.randint(1, length - 1)
            else:
                root_side = random.choice(unseeded_sides)
                reachable_depth = 0
        elif idx == open_idx:
            root_side = random.choice(seeded_sides)
            reachable_depth = random.randint(1, length)
        else:
            if seeded_sides and (not unseeded_sides or random.random() < 0.8):
                root_side = random.choice(seeded_sides)
                if length > 1 and random.random() < 0.35:
                    reachable_depth = random.randint(1, length - 1)
                else:
                    reachable_depth = random.randint(1, length)
            else:
                root_side = random.choice(unseeded_sides)
                reachable_depth = 0

        boxes = _build_chain(root, orientation, length)
        gap_sides = []
        for depth in range(1, length + 1):
            if reachable_depth > 0 and depth <= reachable_depth:
                gap_sides.append(root_side)
            elif depth == 1:
                gap_sides.append(root_side)
            elif depth == reachable_depth + 1 and reachable_depth < length:
                gap_sides.append(_opposite_side(orientation, root_side))
            else:
                gap_sides.append(random.choice((side_a, side_b)))

        seed_strip = _side_strip_index(idx, orientation, root_side)
        seed_color = seed_by_strip[seed_strip]["color"] if seed_strip in seed_by_strip else None
        plans.append(
            {
                "root_index": idx,
                "length": length,
                "reachable_depth": reachable_depth,
                "root_side": root_side,
                "boxes": boxes,
                "gap_sides": gap_sides,
                "seed_color": seed_color,
            }
        )
    return plans


def _gap_cell(box: tuple[int, int, int, int], orientation: str, side: str) -> tuple[int, int]:
    r0, c0, r1, c1 = box
    if orientation == "vertical":
        row = random.randint(r0 + 1, r1 - 1)
        return (row, c0 if side == "left" else c1)
    col = random.randint(c0 + 1, c1 - 1)
    return (r0 if side == "top" else r1, col)


def _draw_frame(grid: np.ndarray, box: tuple[int, int, int, int], color: int, gap: tuple[int, int]) -> None:
    r0, c0, r1, c1 = box
    grid[r0, c0 : c1 + 1] = color
    grid[r1, c0 : c1 + 1] = color
    grid[r0 : r1 + 1, c0] = color
    grid[r0 : r1 + 1, c1] = color
    grid[gap] = 0


def _interior_has_multiple_seeds(grid: np.ndarray) -> bool:
    frames, seeds = _parse_scene(grid)
    inside = _inside_mask(frames, grid.shape)
    for cells, seed_colors in _zero_components_with_seed_sets(grid, seeds):
        if len(seed_colors) > 1 and any(inside[r, c] for r, c in cells):
            return True
    return False


def _validate_generated(grid: np.ndarray, expected: np.ndarray, chains: list[dict], seed_by_strip: dict[int, dict]) -> bool:
    if not np.array_equal(main(grid), expected):
        return False
    if _interior_has_multiple_seeds(grid):
        return False

    try:
        frames, seeds = _parse_scene(grid)
    except AssertionError:
        return False

    if len(frames) != sum(chain["length"] for chain in chains):
        return False
    if len(seeds) != len(seed_by_strip) or not (2 <= len(seeds) <= 3):
        return False

    inside = _inside_mask(frames, grid.shape)
    for seed in seeds:
        r, c = seed["pos"]
        if inside[r, c]:
            return False

    recolored = inside & (grid == 0) & (expected != 0)
    unreached = inside & (grid == 0) & (expected == 0)
    if not np.any(recolored) or not np.any(unreached):
        return False

    seed_colors = np.array(sorted(seed["color"] for seed in seeds), dtype=int)
    for chain in chains:
        for depth, frame in enumerate(chain["frames"], start=1):
            r0, c0, r1, c1 = frame["bbox"]
            interior = expected[r0 + 1 : r1, c0 + 1 : c1]
            if chain["reachable_depth"] >= depth:
                if chain["seed_color"] is None or not np.any(interior == chain["seed_color"]):
                    return False
            else:
                if seed_colors.size and np.any(np.isin(interior, seed_colors)):
                    return False
    return True


def _generate_attempt() -> np.ndarray | None:
    lengths = _choose_lengths()
    orientation = random.choice(("vertical", "horizontal"))
    layout = _build_layout(orientation, lengths)
    if layout is None:
        return None

    seed_strips = _choose_seed_strips(len(lengths) + 1)
    n_frames = sum(lengths)
    n_seeds = len(seed_strips)
    if not (2 <= n_seeds <= 3):
        return None

    palette = random.sample(NON_BLACK, n_frames + n_seeds)
    frame_colors = iter(palette[:n_frames])
    seed_by_strip = _place_seeds(layout, seed_strips, palette[n_frames:])
    chain_plans = _plan_chains(layout, lengths, seed_by_strip)

    grid = np.zeros(layout["shape"], dtype=int)
    for chain in chain_plans:
        frames = []
        for box, side in zip(chain["boxes"], chain["gap_sides"]):
            color = next(frame_colors)
            gap = _gap_cell(box, layout["orientation"], side)
            _draw_frame(grid, box, color, gap)
            frames.append({"bbox": box, "color": color, "gap": gap, "side": side})
        chain["frames"] = frames

    for seed in seed_by_strip.values():
        grid[seed["pos"]] = int(seed["color"])

    expected = _oracle_output(grid)
    if not _validate_generated(grid, expected, chain_plans, seed_by_strip):
        return None
    return grid


def generate_input() -> np.ndarray:
    for _ in range(MAX_GEN_ATTEMPTS):
        grid = _generate_attempt()
        if grid is not None:
            return grid
    raise RuntimeError("failed to generate a valid nested-gap flood puzzle")
