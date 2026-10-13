import random

import numpy as np


BG = 0
GREY = 5
PALETTE = (1, 2, 3, 4, 6, 7, 8, 9)
QUADRANTS = ("tl", "tr", "br", "bl")
ROTATIONS = {"tl": 0, "tr": 1, "br": 2, "bl": 3}

GRID_H_RANGE = (24, 32)
GRID_W_RANGE = (24, 32)
FENCE_COUNT_RANGE = (2, 4)
TEMPLATE_SIZES = (3, 4)
INTERIOR_EXTRA = (0, 2, 4)
PLACEMENT_GAP = 1
PLACEMENT_RETRIES = 200
TEMPLATE_RETRIES = 200
GENERATION_ATTEMPTS = 400


def _neighbors4(r: int, c: int) -> tuple[tuple[int, int], ...]:
    return ((r - 1, c), (r + 1, c), (r, c - 1), (r, c + 1))


def _bbox(coords: list[tuple[int, int]]) -> tuple[int, int, int, int]:
    rows = [r for r, _ in coords]
    cols = [c for _, c in coords]
    return min(rows), min(cols), max(rows), max(cols)


def _components(mask: np.ndarray) -> list[list[tuple[int, int]]]:
    h, w = mask.shape
    seen = np.zeros((h, w), dtype=bool)
    comps: list[list[tuple[int, int]]] = []
    for r in range(h):
        for c in range(w):
            if not mask[r, c] or seen[r, c]:
                continue
            stack = [(r, c)]
            seen[r, c] = True
            comp: list[tuple[int, int]] = []
            while stack:
                rr, cc = stack.pop()
                comp.append((int(rr), int(cc)))
                for nr, nc in _neighbors4(int(rr), int(cc)):
                    if 0 <= nr < h and 0 <= nc < w and mask[nr, nc] and not seen[nr, nc]:
                        seen[nr, nc] = True
                        stack.append((nr, nc))
            comps.append(comp)
    return comps


def _background_color(grid: np.ndarray) -> int:
    vals, counts = np.unique(grid, return_counts=True)
    return int(vals[np.argmax(counts)])


def _border_mask(height: int, width: int) -> np.ndarray:
    border = np.zeros((height, width), dtype=bool)
    border[0, :] = True
    border[-1, :] = True
    border[:, 0] = True
    border[:, -1] = True
    return border


def _find_fences(grid: np.ndarray, bg: int) -> list[dict]:
    fences: list[dict] = []
    for color in np.unique(grid):
        color = int(color)
        if color == bg:
            continue
        for comp in _components(grid == color):
            r0, c0, r1, c1 = _bbox(comp)
            height = r1 - r0 + 1
            width = c1 - c0 + 1
            if height < 5 or width < 5:
                continue
            if len(comp) != 2 * height + 2 * width - 4:
                continue

            patch = grid[r0 : r1 + 1, c0 : c1 + 1]
            border = _border_mask(height, width)
            if not np.all(patch[border] == color):
                continue
            if np.any(patch[~border] == color):
                continue

            interior = patch[1:-1, 1:-1]
            seed_pixels = np.argwhere(interior != bg)
            if len(seed_pixels) != 1:
                continue
            sr, sc = map(int, seed_pixels[0])
            seed = (r0 + 1 + sr, c0 + 1 + sc)
            fences.append(
                {
                    "border_color": color,
                    "bbox": (r0, c0, r1, c1),
                    "seed": seed,
                }
            )

    assert fences, "expected at least one rectangular fence"
    fences.sort(key=lambda item: item["bbox"])
    return fences


def _mark_fence_cells(mask: np.ndarray, fence: dict) -> None:
    r0, c0, r1, c1 = fence["bbox"]
    mask[r0, c0 : c1 + 1] = True
    mask[r1, c0 : c1 + 1] = True
    mask[r0 : r1 + 1, c0] = True
    mask[r0 : r1 + 1, c1] = True
    sr, sc = fence["seed"]
    mask[sr, sc] = True


def _extract_template(grid: np.ndarray, bg: int, fences: list[dict]) -> np.ndarray:
    used = np.zeros(grid.shape, dtype=bool)
    for fence in fences:
        _mark_fence_cells(used, fence)

    template_cells = np.argwhere((grid != bg) & (~used))
    assert len(template_cells) > 0, "template must exist outside all fences"
    r0, c0 = np.min(template_cells, axis=0)
    r1, c1 = np.max(template_cells, axis=0)
    patch = np.asarray(grid[r0 : r1 + 1, c0 : c1 + 1], dtype=int).copy()
    assert patch.shape[0] == patch.shape[1] and patch.shape[0] in TEMPLATE_SIZES
    assert len(set(map(int, patch[patch != bg].ravel()))) == 1, "template should use one foreground color"
    return patch


def _quadrant_from_seed(fence: dict) -> str:
    r0, c0, r1, c1 = fence["bbox"]
    sr, sc = fence["seed"]
    inner_h = r1 - r0 - 1
    inner_w = c1 - c0 - 1
    rel_r = sr - (r0 + 1)
    rel_c = sc - (c0 + 1)
    top = rel_r < inner_h / 2.0
    left = rel_c < inner_w / 2.0
    if top and left:
        return "tl"
    if top and not left:
        return "tr"
    if not top and not left:
        return "br"
    return "bl"


def _anchor_top_left(seed: tuple[int, int], quadrant: str, shape: tuple[int, int]) -> tuple[int, int]:
    sr, sc = seed
    height, width = shape
    if quadrant == "tl":
        return sr, sc
    if quadrant == "tr":
        return sr, sc - width + 1
    if quadrant == "br":
        return sr - height + 1, sc - width + 1
    return sr - height + 1, sc


def _stamp_pattern(
    out: np.ndarray,
    template: np.ndarray,
    seed: tuple[int, int],
    quadrant: str,
    color: int,
    bg: int,
    fence_bbox: tuple[int, int, int, int],
) -> None:
    rotated = np.rot90(template, -ROTATIONS[quadrant])
    top, left = _anchor_top_left(seed, quadrant, rotated.shape)
    height, width = rotated.shape
    r0, c0, r1, c1 = fence_bbox
    assert top >= r0 + 1 and left >= c0 + 1
    assert top + height - 1 <= r1 - 1 and left + width - 1 <= c1 - 1
    region = out[top : top + height, left : left + width]
    region[rotated != bg] = int(color)


def main(grid: np.ndarray) -> np.ndarray:
    """
    Rotate a corner-anchored template according to each seed pixel's quadrant,
    recolor it with the seed color, and stamp it inside the matching fence.
    """
    g = np.asarray(grid, dtype=int)
    bg = _background_color(g)
    fences = _find_fences(g, bg)
    template = _extract_template(g, bg, fences)

    out = g.copy()
    for fence in fences:
        seed = fence["seed"]
        quadrant = _quadrant_from_seed(fence)
        seed_color = int(g[seed])
        _stamp_pattern(out, template, seed, quadrant, seed_color, bg, fence["bbox"])
    return out


def _add_frontier(mask: np.ndarray, frontier: set[tuple[int, int]], r: int, c: int) -> None:
    for nr, nc in _neighbors4(r, c):
        if 0 <= nr < mask.shape[0] and 0 <= nc < mask.shape[1] and not mask[nr, nc]:
            frontier.add((int(nr), int(nc)))


def _random_template() -> np.ndarray:
    size = random.choice(TEMPLATE_SIZES)
    for _ in range(TEMPLATE_RETRIES):
        target = random.randint(size + 1, size * size - 2)
        mask = np.zeros((size, size), dtype=bool)
        mask[0, 0] = True
        frontier: set[tuple[int, int]] = set()
        _add_frontier(mask, frontier, 0, 0)

        while int(mask.sum()) < target and frontier:
            r, c = random.choice(tuple(frontier))
            frontier.remove((r, c))
            mask[r, c] = True
            _add_frontier(mask, frontier, r, c)

        if int(mask.sum()) != target:
            continue
        if not np.any(mask[-1, :]) or not np.any(mask[:, -1]):
            continue
        if len({np.rot90(mask, k).tobytes() for k in range(4)}) != 4:
            continue
        patch = np.full((size, size), BG, dtype=int)
        patch[mask] = GREY
        return patch
    raise RuntimeError("failed to generate a non-symmetric template")


def _corner_bbox(height: int, width: int, size: int, corner: str) -> tuple[int, int, int, int]:
    if corner == "tl":
        return 0, 0, size - 1, size - 1
    if corner == "tr":
        return 0, width - size, size - 1, width - 1
    if corner == "br":
        return height - size, width - size, height - 1, width - 1
    return height - size, 0, height - 1, size - 1


def _mark_box(mask: np.ndarray, box: tuple[int, int, int, int], pad: int) -> None:
    r0, c0, r1, c1 = box
    rr0 = max(0, r0 - pad)
    cc0 = max(0, c0 - pad)
    rr1 = min(mask.shape[0] - 1, r1 + pad)
    cc1 = min(mask.shape[1] - 1, c1 + pad)
    mask[rr0 : rr1 + 1, cc0 : cc1 + 1] = True


def _box_is_free(mask: np.ndarray, box: tuple[int, int, int, int], pad: int) -> bool:
    r0, c0, r1, c1 = box
    rr0 = max(0, r0 - pad)
    cc0 = max(0, c0 - pad)
    rr1 = min(mask.shape[0] - 1, r1 + pad)
    cc1 = min(mask.shape[1] - 1, c1 + pad)
    return not bool(mask[rr0 : rr1 + 1, cc0 : cc1 + 1].any())


def _seed_position(inner_h: int, inner_w: int, quadrant: str) -> tuple[int, int]:
    if quadrant in {"tl", "tr"}:
        row_range = range(0, inner_h // 2)
    else:
        row_range = range(inner_h // 2, inner_h)
    if quadrant in {"tl", "bl"}:
        col_range = range(0, inner_w // 2)
    else:
        col_range = range(inner_w // 2, inner_w)
    return random.choice(tuple(row_range)), random.choice(tuple(col_range))


def _draw_fence(grid: np.ndarray, bbox: tuple[int, int, int, int], border_color: int) -> None:
    r0, c0, r1, c1 = bbox
    grid[r0, c0 : c1 + 1] = border_color
    grid[r1, c0 : c1 + 1] = border_color
    grid[r0 : r1 + 1, c0] = border_color
    grid[r0 : r1 + 1, c1] = border_color


def _expected_output(inp: np.ndarray, template: np.ndarray, fences: list[dict]) -> np.ndarray:
    out = inp.copy()
    for fence in fences:
        _stamp_pattern(
            out,
            template,
            fence["seed"],
            fence["quadrant"],
            fence["seed_color"],
            BG,
            fence["bbox"],
        )
    return out


def _build_sample() -> np.ndarray | None:
    height = random.randint(*GRID_H_RANGE)
    width = random.randint(*GRID_W_RANGE)
    template = _random_template()
    template_size = template.shape[0]
    grid = np.full((height, width), BG, dtype=int)

    corner = random.choice(QUADRANTS)
    tr0, tc0, tr1, tc1 = _corner_bbox(height, width, template_size, corner)
    grid[tr0 : tr1 + 1, tc0 : tc1 + 1] = template

    occupied = np.zeros((height, width), dtype=bool)
    _mark_box(occupied, (tr0, tc0, tr1, tc1), PLACEMENT_GAP)

    fence_count = random.randint(*FENCE_COUNT_RANGE)
    colors = random.sample(PALETTE, 2 * fence_count)
    border_colors = colors[:fence_count]
    seed_colors = colors[fence_count:]
    quadrants = random.sample(QUADRANTS, fence_count)

    specs: list[dict] = []
    for idx in range(fence_count):
        inner_h = 2 * template_size + random.choice(INTERIOR_EXTRA)
        inner_w = 2 * template_size + random.choice(INTERIOR_EXTRA)
        specs.append(
            {
                "border_color": border_colors[idx],
                "seed_color": seed_colors[idx],
                "quadrant": quadrants[idx],
                "inner_h": inner_h,
                "inner_w": inner_w,
                "outer_area": (inner_h + 2) * (inner_w + 2),
            }
        )
    specs.sort(key=lambda item: item["outer_area"], reverse=True)

    placed: list[dict] = []
    for spec in specs:
        outer_h = spec["inner_h"] + 2
        outer_w = spec["inner_w"] + 2
        if height - outer_h < 2 or width - outer_w < 2:
            return None
        for _ in range(PLACEMENT_RETRIES):
            r0 = random.randint(1, height - outer_h - 1)
            c0 = random.randint(1, width - outer_w - 1)
            bbox = (r0, c0, r0 + outer_h - 1, c0 + outer_w - 1)
            if not _box_is_free(occupied, bbox, PLACEMENT_GAP):
                continue

            rel_seed_r, rel_seed_c = _seed_position(spec["inner_h"], spec["inner_w"], spec["quadrant"])
            seed = (r0 + 1 + rel_seed_r, c0 + 1 + rel_seed_c)
            placed.append(
                {
                    "bbox": bbox,
                    "border_color": spec["border_color"],
                    "seed_color": spec["seed_color"],
                    "quadrant": spec["quadrant"],
                    "seed": seed,
                }
            )
            _mark_box(occupied, bbox, PLACEMENT_GAP)
            break
        else:
            return None

    for fence in placed:
        _draw_fence(grid, fence["bbox"], fence["border_color"])
        sr, sc = fence["seed"]
        grid[sr, sc] = fence["seed_color"]

    expected = _expected_output(grid, template, placed)
    produced = main(grid)
    if not np.array_equal(produced, expected):
        return None
    if np.array_equal(grid, expected):
        return None
    return grid


def generate_input() -> np.ndarray:
    for _ in range(GENERATION_ATTEMPTS):
        sample = _build_sample()
        if sample is not None:
            return sample
    raise RuntimeError("failed to generate a valid input")
