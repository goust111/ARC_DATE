import random
from collections import deque

import numpy as np


SEPARATOR_COLOR = 8
RULER_COLOR = 9
ARC_MAX_SIDE = 30
GENERATION_ATTEMPTS = 400
MAX_REGION_SIDE = 14


def get_bg(grid: np.ndarray) -> int:
    values, counts = np.unique(grid, return_counts=True)
    return int(values[np.argmax(counts)])


def bbox_from_coords(coords: list[tuple[int, int]]) -> tuple[int, int, int, int]:
    rows = [r for r, _ in coords]
    cols = [c for _, c in coords]
    return min(rows), min(cols), max(rows), max(cols)


def find_components(
    grid: np.ndarray,
    *,
    excluded: set[int],
) -> list[dict]:
    h, w = map(int, grid.shape)
    seen = np.zeros((h, w), dtype=bool)
    components: list[dict] = []

    for r in range(h):
        for c in range(w):
            color = int(grid[r, c])
            if seen[r, c] or color in excluded:
                continue

            q = deque([(r, c)])
            seen[r, c] = True
            coords: list[tuple[int, int]] = []
            while q:
                cr, cc = q.popleft()
                coords.append((cr, cc))
                for nr, nc in ((cr - 1, cc), (cr + 1, cc), (cr, cc - 1), (cr, cc + 1)):
                    if 0 <= nr < h and 0 <= nc < w and not seen[nr, nc] and int(grid[nr, nc]) == color:
                        seen[nr, nc] = True
                        q.append((nr, nc))

            r0, c0, r1, c1 = bbox_from_coords(coords)
            components.append(
                {
                    "color": color,
                    "coords": coords,
                    "bbox": (r0, c0, r1, c1),
                    "area": len(coords),
                }
            )
    return components


def paste(dst: np.ndarray, patch: np.ndarray, top: int, left: int) -> None:
    h, w = map(int, patch.shape)
    dst[top : top + h, left : left + w] = patch


def scale_patch(patch: np.ndarray, factor: int) -> np.ndarray:
    return np.repeat(np.repeat(patch, factor, axis=0), factor, axis=1)


def assemble_output(parts: dict[str, np.ndarray], bg: int) -> np.ndarray:
    top_h = max(int(parts["tl"].shape[0]), int(parts["tr"].shape[0]))
    bottom_h = max(int(parts["bl"].shape[0]), int(parts["br"].shape[0]))
    left_w = max(int(parts["tl"].shape[1]), int(parts["bl"].shape[1]))
    right_w = max(int(parts["tr"].shape[1]), int(parts["br"].shape[1]))

    out = np.full((top_h + 1 + bottom_h, left_w + 1 + right_w), int(bg), dtype=int)
    paste(out, parts["tl"], 0, 0)
    paste(out, parts["tr"], 0, left_w + 1)
    paste(out, parts["bl"], top_h + 1, 0)
    paste(out, parts["br"], top_h + 1, left_w + 1)
    return out


def extract_region(region: np.ndarray, bg: int, separator_color: int) -> tuple[int, np.ndarray]:
    components = find_components(region, excluded={int(bg), int(separator_color)})
    assert len(components) == 2, "each sub-region must contain one ruler and one pattern"

    h, w = map(int, region.shape)
    ruler = None
    pattern = None
    for comp in components:
        r0, c0, r1, c1 = comp["bbox"]
        comp_h = r1 - r0 + 1
        comp_w = c1 - c0 + 1
        touches_edge = r0 == 0 or c0 == 0 or r1 == h - 1 or c1 == w - 1
        is_line = (comp_h == 1 or comp_w == 1) and comp["area"] == max(comp_h, comp_w)
        if touches_edge and is_line and 1 <= comp["area"] <= 4:
            ruler = comp
        else:
            pattern = comp

    assert ruler is not None and pattern is not None, "region must have a unique ruler and pattern"

    r0, c0, r1, c1 = pattern["bbox"]
    assert r1 > r0 and c1 > c0, "pattern bbox must be at least 2x2"
    patch = np.full((r1 - r0 + 1, c1 - c0 + 1), int(bg), dtype=int)
    for pr, pc in pattern["coords"]:
        patch[pr - r0, pc - c0] = int(pattern["color"])
    return int(ruler["area"]), patch


def main(grid: np.ndarray) -> np.ndarray:
    """
    Read the ruler length in each quadrant and enlarge the shared sprite accordingly.

    Input structure:
    - The background is the most common color.
    - One full horizontal separator row and one full vertical separator column of the same color split the grid into four rectangular sub-regions.
    - Each sub-region contains exactly two non-background objects: a small monochrome 4-connected pattern and a one-cell-thick ruler bar touching the region edge.

    Rule:
    - Identify the separator cross and crop the four regions around it.
    - In each region, measure the ruler bar length L and crop the pattern to its bbox.
    - Scale the cropped pattern by L with block replication, then place the four scaled results in top-left, top-right, bottom-left, bottom-right order with a one-cell background gap between columns and rows.

    Key insight:
    - The ruler direction is irrelevant noise; only its visible length encodes the integer scale factor for the matching quadrant's pattern.
    """
    g = np.asarray(grid, dtype=int)
    assert g.ndim == 2, "expected a 2D grid"
    h, w = map(int, g.shape)
    bg = get_bg(g)

    row_candidates = [(r, int(g[r, 0])) for r in range(h) if np.all(g[r, :] == g[r, 0]) and int(g[r, 0]) != bg]
    col_candidates = [(c, int(g[0, c])) for c in range(w) if np.all(g[:, c] == g[0, c]) and int(g[0, c]) != bg]
    matches = [(r, c, color_r) for r, color_r in row_candidates for c, color_c in col_candidates if color_r == color_c]
    assert len(matches) == 1, "expected a unique separator cross"
    sep_r, sep_c, sep_color = matches[0]

    regions = {
        "tl": g[:sep_r, :sep_c],
        "tr": g[:sep_r, sep_c + 1 :],
        "bl": g[sep_r + 1 :, :sep_c],
        "br": g[sep_r + 1 :, sep_c + 1 :],
    }
    assert all(region.size > 0 for region in regions.values()), "all four regions must be non-empty"

    scaled_parts: dict[str, np.ndarray] = {}
    for key, region in regions.items():
        factor, patch = extract_region(region, bg, sep_color)
        scaled_parts[key] = scale_patch(patch, factor)

    return assemble_output(scaled_parts, bg)


def neighbors4(r: int, c: int, h: int, w: int):
    if r > 0:
        yield r - 1, c
    if r + 1 < h:
        yield r + 1, c
    if c > 0:
        yield r, c - 1
    if c + 1 < w:
        yield r, c + 1


def sample_pattern_mask(height: int, width: int) -> np.ndarray:
    min_area = max(height + width - 1, (height * width + 1) // 2)
    target_area = random.randint(min_area, height * width)

    mask = np.zeros((height, width), dtype=bool)
    steps = ["D"] * (height - 1) + ["R"] * (width - 1)
    random.shuffle(steps)

    r = c = 0
    mask[r, c] = True
    for step in steps:
        if step == "D":
            r += 1
        else:
            c += 1
        mask[r, c] = True

    frontier: set[tuple[int, int]] = set()

    def add_frontier(rr: int, cc: int) -> None:
        for nr, nc in neighbors4(rr, cc, height, width):
            if not mask[nr, nc]:
                frontier.add((nr, nc))

    for rr, cc in np.argwhere(mask):
        add_frontier(int(rr), int(cc))

    while int(mask.sum()) < target_area:
        if not frontier:
            break
        cell = random.choice(tuple(frontier))
        frontier.remove(cell)
        rr, cc = cell
        mask[rr, cc] = True
        add_frontier(rr, cc)

    if random.random() < 0.5:
        mask = np.flipud(mask)
    if random.random() < 0.5:
        mask = np.fliplr(mask)
    return mask


def make_pattern_patch(mask: np.ndarray, color: int, bg: int) -> np.ndarray:
    patch = np.full(mask.shape, int(bg), dtype=int)
    patch[mask] = int(color)
    return patch


def place_ruler(region: np.ndarray, length: int) -> None:
    h, w = map(int, region.shape)
    if random.random() < 0.5:
        row = 0 if random.random() < 0.5 else h - 1
        col = random.randint(0, w - length)
        region[row, col : col + length] = RULER_COLOR
    else:
        col = 0 if random.random() < 0.5 else w - 1
        row = random.randint(0, h - length)
        region[row : row + length, col] = RULER_COLOR


def build_region(region_h: int, region_w: int, patch: np.ndarray, scale: int, bg: int) -> np.ndarray:
    out = np.full((region_h, region_w), int(bg), dtype=int)
    patch_h, patch_w = map(int, patch.shape)
    top = random.randint(1, region_h - patch_h - 2)
    left = random.randint(1, region_w - patch_w - 2)
    paste(out, patch, top, left)
    place_ruler(out, scale)
    return out


def sample_case() -> tuple[np.ndarray, np.ndarray] | None:
    bg = random.choice([c for c in range(10) if c not in {SEPARATOR_COLOR, RULER_COLOR}])

    pattern_h = random.randint(2, 4)
    pattern_w = random.randint(2, 4)
    pattern_mask = sample_pattern_mask(pattern_h, pattern_w)

    scales = {key: random.randint(1, 4) for key in ("tl", "tr", "bl", "br")}
    scaled_heights = {key: pattern_h * value for key, value in scales.items()}
    scaled_widths = {key: pattern_w * value for key, value in scales.items()}
    output_h = max(scaled_heights["tl"], scaled_heights["tr"]) + 1 + max(scaled_heights["bl"], scaled_heights["br"])
    output_w = max(scaled_widths["tl"], scaled_widths["bl"]) + 1 + max(scaled_widths["tr"], scaled_widths["br"])
    if output_h > ARC_MAX_SIDE or output_w > ARC_MAX_SIDE:
        return None

    min_region_h = max(pattern_h + 3, 5)
    min_region_w = max(pattern_w + 3, 5)
    if min_region_h > MAX_REGION_SIDE or min_region_w > MAX_REGION_SIDE:
        return None

    top_h = random.randint(min_region_h, MAX_REGION_SIDE)
    bottom_h = random.randint(min_region_h, MAX_REGION_SIDE)
    left_w = random.randint(min_region_w, MAX_REGION_SIDE)
    right_w = random.randint(min_region_w, MAX_REGION_SIDE)
    grid_h = top_h + 1 + bottom_h
    grid_w = left_w + 1 + right_w
    if grid_h > ARC_MAX_SIDE or grid_w > ARC_MAX_SIDE:
        return None

    available = [c for c in range(10) if c not in {bg, SEPARATOR_COLOR, RULER_COLOR}]
    palette_size = random.randint(2, min(4, len(available)))
    palette = random.sample(available, palette_size)
    colors = {key: random.choice(palette) for key in ("tl", "tr", "bl", "br")}
    if len(set(colors.values())) == 1:
        replacement_keys = [key for key in colors if colors[key] == colors["tl"]]
        if len(palette) > 1:
            colors[random.choice(replacement_keys)] = random.choice([c for c in palette if c != colors["tl"]])

    patches = {key: make_pattern_patch(pattern_mask, color, bg) for key, color in colors.items()}

    tl = build_region(top_h, left_w, patches["tl"], scales["tl"], bg)
    tr = build_region(top_h, right_w, patches["tr"], scales["tr"], bg)
    bl = build_region(bottom_h, left_w, patches["bl"], scales["bl"], bg)
    br = build_region(bottom_h, right_w, patches["br"], scales["br"], bg)

    grid = np.full((grid_h, grid_w), int(bg), dtype=int)
    grid[top_h, :] = SEPARATOR_COLOR
    grid[:, left_w] = SEPARATOR_COLOR
    paste(grid, tl, 0, 0)
    paste(grid, tr, 0, left_w + 1)
    paste(grid, bl, top_h + 1, 0)
    paste(grid, br, top_h + 1, left_w + 1)

    counts = np.bincount(grid.ravel(), minlength=10)
    if int(np.argmax(counts)) != bg:
        return None

    scaled_parts = {key: scale_patch(patches[key], scales[key]) for key in patches}
    expected = assemble_output(scaled_parts, bg)
    if max(expected.shape) > ARC_MAX_SIDE:
        return None

    try:
        if not np.array_equal(main(grid), expected):
            return None
    except Exception:
        return None
    return grid, expected


def generate_input() -> np.ndarray:
    for _ in range(GENERATION_ATTEMPTS):
        sample = sample_case()
        if sample is not None:
            grid, _expected = sample
            return grid
    raise RuntimeError("failed to generate a valid input for puzzle 4")
