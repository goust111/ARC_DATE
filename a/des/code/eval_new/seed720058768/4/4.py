import random
from typing import Iterable

import numpy as np


BG = 0
WALL = 5
CELL = 2
DIR4 = [(-1, 0), (1, 0), (0, -1), (0, 1)]
GEN_RETRIES = 600
COLORS = [1, 2, 3, 4, 6, 7, 8, 9]


def most_common_color(grid: np.ndarray) -> int:
    vals, counts = np.unique(np.asarray(grid, dtype=int), return_counts=True)
    return int(vals[np.argmax(counts)])


def components_mask(mask: np.ndarray, connectivity: int = 4) -> list[np.ndarray]:
    m = np.asarray(mask, dtype=bool)
    h, w = m.shape
    seen = np.zeros((h, w), dtype=bool)
    dirs = DIR4 if connectivity == 4 else [
        (-1, 0),
        (1, 0),
        (0, -1),
        (0, 1),
        (-1, -1),
        (-1, 1),
        (1, -1),
        (1, 1),
    ]
    comps: list[np.ndarray] = []
    for r in range(h):
        for c in range(w):
            if not m[r, c] or seen[r, c]:
                continue
            stack = [(r, c)]
            seen[r, c] = True
            coords: list[tuple[int, int]] = []
            while stack:
                cr, cc = stack.pop()
                coords.append((cr, cc))
                for dr, dc in dirs:
                    nr, nc = cr + dr, cc + dc
                    if 0 <= nr < h and 0 <= nc < w and m[nr, nc] and not seen[nr, nc]:
                        seen[nr, nc] = True
                        stack.append((nr, nc))
            comps.append(np.asarray(coords, dtype=int))
    return comps


def bbox_coords(coords: np.ndarray) -> tuple[int, int, int, int]:
    pts = np.asarray(coords, dtype=int)
    return (
        int(pts[:, 0].min()),
        int(pts[:, 1].min()),
        int(pts[:, 0].max()),
        int(pts[:, 1].max()),
    )


def crop_bbox(arr: np.ndarray, bbox: tuple[int, int, int, int]) -> np.ndarray:
    r0, c0, r1, c1 = map(int, bbox)
    return np.asarray(arr)[r0 : r1 + 1, c0 : c1 + 1]


def mask_from_coords(coords: Iterable[tuple[int, int]] | np.ndarray, shape: tuple[int, int]) -> np.ndarray:
    out = np.zeros(shape, dtype=bool)
    pts = np.asarray(list(coords) if not isinstance(coords, np.ndarray) else coords, dtype=int)
    if pts.size:
        out[pts[:, 0], pts[:, 1]] = True
    return out


def shape_key(mask: np.ndarray) -> tuple[tuple[int, int], bytes]:
    m = np.asarray(mask, dtype=bool)
    return (tuple(map(int, m.shape)), m.tobytes())


def has_hole(mask: np.ndarray) -> bool:
    m = np.asarray(mask, dtype=bool)
    h, w = m.shape
    outside = np.zeros((h, w), dtype=bool)
    stack: list[tuple[int, int]] = []
    for r in range(h):
        for c in (0, w - 1):
            if not m[r, c] and not outside[r, c]:
                outside[r, c] = True
                stack.append((r, c))
    for c in range(w):
        for r in (0, h - 1):
            if not m[r, c] and not outside[r, c]:
                outside[r, c] = True
                stack.append((r, c))
    while stack:
        r, c = stack.pop()
        for dr, dc in DIR4:
            nr, nc = r + dr, c + dc
            if 0 <= nr < h and 0 <= nc < w and not m[nr, nc] and not outside[nr, nc]:
                outside[nr, nc] = True
                stack.append((nr, nc))
    return bool(np.any((~m) & (~outside)))


def extract_regions(crop: np.ndarray, bg: int) -> list[dict]:
    infos: list[dict] = []
    for pts in components_mask(crop == int(bg), connectivity=4):
        b = bbox_coords(pts)
        rel = pts - np.array([b[0], b[1]])
        mask = mask_from_coords(rel, (b[2] - b[0] + 1, b[3] - b[1] + 1))
        infos.append({"coords": pts, "bbox": b, "mask": mask, "key": shape_key(mask)})
    return infos


def main(grid: np.ndarray) -> np.ndarray:
    """
    Recolor matching template cells by mapping scattered sample-shape colors back into the left mosaic.

    Input structure:
    - The background is black and fills empty space.
    - The left side contains a wall-colored template made of thin connected divider lines and an outer frame.
    - Inside that template, each enclosed background region is a distinct shape that may remain black or receive a color.
    - The right side contains several disconnected monochrome sample sprites whose silhouettes exactly match some template regions, without rotation or reflection.

    Rule:
    - Identify the template by the most common non-background wall color and crop to its bounding box.
    - Extract every enclosed background region inside the crop and record its exact shape.
    - For each colored sample on the right, match its cropped binary silhouette to the unique template region with the same shape.
    - Fill the matched template region with the sample's color, leave unmatched template regions black, and keep the wall lines unchanged.

    Key insight:
    The scattered colored sprites are not output pieces themselves; each one is a shape-coded instruction telling which enclosed template region should be painted with its color.
    """
    g = np.asarray(grid, dtype=int)
    bg = most_common_color(g)
    non_bg = g[g != bg]
    if non_bg.size == 0:
        return g.copy()

    vals, counts = np.unique(non_bg, return_counts=True)
    wall_color = int(vals[np.argmax(counts)])

    wall_coords = np.argwhere(g == wall_color)
    if wall_coords.size == 0:
        return g.copy()
    template_bbox = bbox_coords(wall_coords)
    out = crop_bbox(g, template_bbox).copy()

    regions = extract_regions(out, bg)
    by_shape: dict[tuple[tuple[int, int], bytes], list[np.ndarray]] = {}
    for info in regions:
        by_shape.setdefault(info["key"], []).append(info["coords"])

    sample_mask = (g != bg) & (g != wall_color)
    for pts in components_mask(sample_mask, connectivity=4):
        sample_colors = np.unique(g[pts[:, 0], pts[:, 1]])
        if len(sample_colors) != 1:
            continue
        color = int(sample_colors[0])
        b = bbox_coords(pts)
        rel = pts - np.array([b[0], b[1]])
        mask = mask_from_coords(rel, (b[2] - b[0] + 1, b[3] - b[1] + 1))
        matches = by_shape.get(shape_key(mask), [])
        if len(matches) != 1:
            continue
        target = matches[0]
        out[target[:, 0], target[:, 1]] = color

    return out


def random_region_sizes(total_cells: int, n_regions: int) -> list[int]:
    sizes = [2] * n_regions
    remaining = total_cells - 2 * n_regions
    while remaining > 0:
        idx = random.randrange(n_regions)
        sizes[idx] += 1
        remaining -= 1
    random.shuffle(sizes)
    return sizes


def grow_partition(rows: int, cols: int, sizes: list[int]) -> np.ndarray | None:
    n_regions = len(sizes)
    labels = np.full((rows, cols), -1, dtype=int)
    cells = [(r, c) for r in range(rows) for c in range(cols)]
    seeds = random.sample(cells, n_regions)
    region_cells: list[set[tuple[int, int]]] = []
    for rid, (r, c) in enumerate(seeds):
        labels[r, c] = rid
        region_cells.append({(r, c)})

    remaining = {(r, c) for r, c in cells if labels[r, c] < 0}
    while remaining:
        moves: list[tuple[int, int, int]] = []
        for rid, pts in enumerate(region_cells):
            if len(pts) >= sizes[rid]:
                continue
            frontier: set[tuple[int, int]] = set()
            for r, c in pts:
                for dr, dc in DIR4:
                    nr, nc = r + dr, c + dc
                    if 0 <= nr < rows and 0 <= nc < cols and labels[nr, nc] < 0:
                        frontier.add((nr, nc))
            for nr, nc in frontier:
                moves.append((rid, nr, nc))
        if not moves:
            return None
        rid, nr, nc = random.choice(moves)
        labels[nr, nc] = rid
        region_cells[rid].add((nr, nc))
        remaining.remove((nr, nc))

    if any(len(region_cells[rid]) != sizes[rid] for rid in range(n_regions)):
        return None
    return labels


def render_template(labels: np.ndarray, bg: int = BG, wall: int = WALL, cell: int = CELL) -> np.ndarray:
    rows, cols = labels.shape
    h = rows * cell + rows + 1
    w = cols * cell + cols + 1
    grid = np.full((h, w), wall, dtype=int)
    for r in range(rows):
        for c in range(cols):
            r0 = 1 + r * (cell + 1)
            c0 = 1 + c * (cell + 1)
            grid[r0 : r0 + cell, c0 : c0 + cell] = bg
            if c + 1 < cols and labels[r, c] == labels[r, c + 1]:
                grid[r0 : r0 + cell, c0 + cell] = bg
            if r + 1 < rows and labels[r, c] == labels[r + 1, c]:
                grid[r0 + cell, c0 : c0 + cell] = bg
    return grid


def valid_partition(labels: np.ndarray) -> bool:
    keys: set[tuple[tuple[int, int], bytes]] = set()
    rows, cols = labels.shape
    for rid in range(int(labels.max()) + 1):
        pts = np.argwhere(labels == rid)
        if pts.size == 0:
            return False
        b = bbox_coords(pts)
        rel = pts - np.array([b[0], b[1]])
        mask = mask_from_coords(rel, (b[2] - b[0] + 1, b[3] - b[1] + 1))
        if has_hole(mask):
            return False
        key = shape_key(mask)
        if key in keys:
            return False
        keys.add(key)
        if mask.shape == (rows, cols):
            return False
    return True


def make_partition() -> np.ndarray | None:
    for _ in range(300):
        rows = random.randint(4, 6)
        cols = random.randint(4, 5)
        total = rows * cols
        max_regions = min(7, total // 2)
        min_regions = 4
        if max_regions < min_regions:
            continue
        n_regions = random.randint(min_regions, max_regions)
        sizes = random_region_sizes(total, n_regions)
        labels = grow_partition(rows, cols, sizes)
        if labels is None:
            continue
        if valid_partition(labels):
            return labels
    return None


def choose_samples(region_infos: list[dict], template_w: int) -> list[int] | None:
    max_sample_w = 30 - template_w - 2
    placeable = [
        idx
        for idx, info in enumerate(region_infos)
        if info["mask"].shape[1] <= max_sample_w and info["mask"].shape[0] <= 12
    ]
    if not placeable:
        return None
    max_count = min(len(placeable), len(region_infos) - 1)
    if max_count < 1:
        return None
    count = random.randint(1, max_count)
    return random.sample(placeable, count)


def place_samples(
    patches: list[np.ndarray],
    template_h: int,
    template_w: int,
) -> tuple[np.ndarray, int, int, list[tuple[int, int]]] | None:
    max_ph = max(int(p.shape[0]) for p in patches)
    max_pw = max(int(p.shape[1]) for p in patches)
    gap = random.randint(2, 4)
    panel_w_max = 30 - template_w - gap
    panel_w_min = max(max_pw + 1, min(8, panel_w_max))
    if panel_w_min > panel_w_max:
        return None
    panel_w = random.randint(panel_w_min, panel_w_max)
    height_min = max(template_h, max_ph + 2, 16)
    height_max = min(30, max(template_h + 6, 22))
    if height_min > height_max:
        return None
    height = random.randint(height_min, height_max)
    template_r0 = random.randint(0, height - template_h)
    template_c0 = 0
    panel_x0 = template_c0 + template_w + gap
    width = panel_x0 + panel_w
    occupied = np.zeros((height, width), dtype=bool)
    occupied[max(0, template_r0 - 1) : min(height, template_r0 + template_h + 1), :panel_x0] = True
    placements: list[tuple[int, int]] = []
    for patch in patches:
        ph, pw = map(int, patch.shape)
        placed = False
        for _ in range(250):
            r0 = random.randint(0, height - ph)
            c0 = random.randint(panel_x0, width - pw)
            rr0 = max(0, r0 - 1)
            cc0 = max(0, c0 - 1)
            rr1 = min(height, r0 + ph + 1)
            cc1 = min(width, c0 + pw + 1)
            if occupied[rr0:rr1, cc0:cc1].any():
                continue
            occupied[rr0:rr1, cc0:cc1] = True
            placements.append((r0, c0))
            placed = True
            break
        if not placed:
            return None

    grid = np.full((height, width), BG, dtype=int)
    return grid, template_r0, template_c0, placements


def gen_attempt() -> np.ndarray | None:
    labels = make_partition()
    if labels is None:
        return None

    template = render_template(labels)
    expected = template.copy()
    regions = extract_regions(template, BG)
    if len(regions) < 4:
        return None
    if len({info["key"] for info in regions}) != len(regions):
        return None

    sample_ids = choose_samples(regions, template.shape[1])
    if not sample_ids:
        return None

    sample_colors = random.sample(COLORS, len(sample_ids))
    patches: list[np.ndarray] = []
    for idx, color in zip(sample_ids, sample_colors):
        mask = regions[idx]["mask"]
        patch = np.full(mask.shape, BG, dtype=int)
        patch[mask] = int(color)
        patches.append(patch)
        coords = regions[idx]["coords"]
        expected[coords[:, 0], coords[:, 1]] = int(color)

    placed = place_samples(patches, template.shape[0], template.shape[1])
    if placed is None:
        return None
    grid, template_r0, template_c0, placements = placed
    th, tw = map(int, template.shape)
    grid[template_r0 : template_r0 + th, template_c0 : template_c0 + tw] = template
    for patch, (r0, c0) in zip(patches, placements):
        ph, pw = map(int, patch.shape)
        mask = patch != BG
        grid[r0 : r0 + ph, c0 : c0 + pw][mask] = patch[mask]

    if np.array_equal(main(grid), expected):
        return grid
    return None


def generate_input() -> np.ndarray:
    for _ in range(GEN_RETRIES):
        grid = gen_attempt()
        if grid is not None:
            return grid
    raise RuntimeError("failed to generate valid input")
