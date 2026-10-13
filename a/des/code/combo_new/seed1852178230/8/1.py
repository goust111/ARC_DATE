import random
from typing import Dict, Iterable, List, Sequence, Tuple

import numpy as np


BG = 0
FENCE = 5
COLOR_POOL = [1, 2, 3, 4, 6, 7, 8, 9]

PATTERN_LIBRARY: Tuple[Tuple[Tuple[int, int], ...], ...] = (
    ((0, 0), (1, 0), (2, 0), (2, 1)),
    ((0, 0), (0, 1), (1, 1), (2, 1)),
    ((0, 0), (1, 0), (1, 1), (1, 2)),
    ((0, 0), (0, 1), (1, 1), (2, 1), (2, 2)),
    ((0, 0), (1, 0), (1, 1), (2, 1), (2, 2)),
    ((0, 0), (1, 0), (2, 0), (1, 1), (1, 2)),
    ((0, 0), (0, 1), (0, 2), (1, 0), (2, 0)),
)

MAX_GEN_RETRIES = 300


def _components(mask: np.ndarray) -> List[np.ndarray]:
    h, w = mask.shape
    seen = np.zeros((h, w), dtype=bool)
    comps: List[np.ndarray] = []
    for r in range(h):
        for c in range(w):
            if not mask[r, c] or seen[r, c]:
                continue
            stack = [(r, c)]
            seen[r, c] = True
            pts = []
            while stack:
                cr, cc = stack.pop()
                pts.append((cr, cc))
                for nr, nc in ((cr - 1, cc), (cr + 1, cc), (cr, cc - 1), (cr, cc + 1)):
                    if 0 <= nr < h and 0 <= nc < w and mask[nr, nc] and not seen[nr, nc]:
                        seen[nr, nc] = True
                        stack.append((nr, nc))
            comps.append(np.array(pts, dtype=int))
    return comps


def _bbox(coords: np.ndarray) -> Tuple[int, int, int, int]:
    return (
        int(coords[:, 0].min()),
        int(coords[:, 1].min()),
        int(coords[:, 0].max()),
        int(coords[:, 1].max()),
    )


def _normalize_coords(coords: Iterable[Tuple[int, int]]) -> Tuple[Tuple[int, int], ...]:
    pts = [(int(r), int(c)) for r, c in coords]
    min_r = min(r for r, _ in pts)
    min_c = min(c for _, c in pts)
    return tuple(sorted((r - min_r, c - min_c) for r, c in pts))


def _shape_size(coords: Sequence[Tuple[int, int]]) -> Tuple[int, int]:
    return (
        1 + max(int(r) for r, _ in coords),
        1 + max(int(c) for _, c in coords),
    )


def _rotate_coords(coords: Sequence[Tuple[int, int]], turns_cw: int) -> Tuple[Tuple[int, int], ...]:
    norm = _normalize_coords(coords)
    h, w = _shape_size(norm)
    mask = np.zeros((h, w), dtype=bool)
    for r, c in norm:
        mask[r, c] = True
    rot = np.rot90(mask, -int(turns_cw) % 4)
    return _normalize_coords(map(tuple, np.argwhere(rot)))


def _find_fences(grid: np.ndarray) -> List[Tuple[int, int, int, int]]:
    fences: List[Tuple[int, int, int, int]] = []
    for comp in _components(grid == FENCE):
        r0, c0, r1, c1 = _bbox(comp)
        if r1 - r0 < 2 or c1 - c0 < 2:
            continue
        sub = grid[r0 : r1 + 1, c0 : c1 + 1]
        border = np.zeros_like(sub, dtype=bool)
        border[0, :] = True
        border[-1, :] = True
        border[:, 0] = True
        border[:, -1] = True
        if np.all(sub[border] == FENCE) and np.all(sub[~border] != FENCE):
            if int(border.sum()) == len(comp):
                fences.append((r0, c0, r1, c1))
    fences.sort()
    return fences


def _extract_templates(
    grid: np.ndarray,
    fences: Sequence[Tuple[int, int, int, int]],
) -> Dict[int, Tuple[Tuple[int, int], ...]]:
    outside = np.ones(grid.shape, dtype=bool)
    for r0, c0, r1, c1 in fences:
        outside[r0 + 1 : r1, c0 + 1 : c1] = False

    tmpl_mask = (grid != BG) & (grid != FENCE) & outside
    templates: Dict[int, Tuple[Tuple[int, int], ...]] = {}
    for comp in _components(tmpl_mask):
        colors = {int(grid[r, c]) for r, c in comp}
        assert len(colors) == 1, "template component must be monochrome"
        color = next(iter(colors))
        rel = _normalize_coords(comp)
        assert color not in templates, f"duplicate template color {color}"
        templates[color] = rel
    assert templates, "at least one template is required"
    return templates


def _quadrant_of_seed(
    seed_r: int,
    seed_c: int,
    fence: Tuple[int, int, int, int],
) -> int:
    r0, c0, r1, c1 = fence
    inner_h = r1 - r0 - 1
    inner_w = c1 - c0 - 1
    rel_r = seed_r - (r0 + 1)
    rel_c = seed_c - (c0 + 1)
    top = rel_r * 2 < inner_h
    left = rel_c * 2 < inner_w
    if top and left:
        return 0
    if top and not left:
        return 1
    if not top and not left:
        return 2
    return 3


def _paint_rotated_template(
    out: np.ndarray,
    color: int,
    coords: Sequence[Tuple[int, int]],
    fence: Tuple[int, int, int, int],
    quadrant: int,
) -> None:
    r0, c0, r1, c1 = fence
    inner_top, inner_left = r0 + 1, c0 + 1
    inner_bottom, inner_right = r1 - 1, c1 - 1

    oriented = _rotate_coords(coords, quadrant)
    h, w = _shape_size(oriented)

    if quadrant == 0:
        base_r, base_c = inner_top, inner_left
    elif quadrant == 1:
        base_r, base_c = inner_top, inner_right - (w - 1)
    elif quadrant == 2:
        base_r, base_c = inner_bottom - (h - 1), inner_right - (w - 1)
    else:
        base_r, base_c = inner_bottom - (h - 1), inner_left

    for dr, dc in oriented:
        rr = base_r + dr
        cc = base_c + dc
        assert inner_top <= rr <= inner_bottom and inner_left <= cc <= inner_right
        out[rr, cc] = color


def _solve(grid: np.ndarray) -> np.ndarray:
    g = np.asarray(grid, dtype=int)
    fences = _find_fences(g)
    assert fences, "no grey rectangular fence found"
    templates = _extract_templates(g, fences)

    out = g.copy()
    for fence in fences:
        r0, c0, r1, c1 = fence
        inner = g[r0 + 1 : r1, c0 + 1 : c1]
        seeds = np.argwhere((inner != BG) & (inner != FENCE))
        assert len(seeds) == 1, "each fence interior must contain exactly one seed"
        sr, sc = map(int, seeds[0])
        color = int(inner[sr, sc])
        assert color in templates, f"seed color {color} has no template"
        abs_r = r0 + 1 + sr
        abs_c = c0 + 1 + sc
        quadrant = _quadrant_of_seed(abs_r, abs_c, fence)
        _paint_rotated_template(out, color, templates[color], fence, quadrant)
    return out


def main(grid: np.ndarray) -> np.ndarray:
    """
    Fill each fenced interior with the color-matched corner template, rotated by the seed's quadrant.

    Input structure:
    - The grid has a black background, several grey hollow rectangles, and a small gallery of colored template sprites in one corner area.
    - Each grey rectangle is a one-pixel-thick closed fence with exactly one non-black, non-grey seed pixel inside.
    - Each template sprite is monochrome, separated from the others by black background, and its color matches one or more fence seeds.

    Rule:
    - Extract every outside-fence template as coordinates relative to its own bounding box.
    - For each fence, read the unique seed color and use it to select the matching template.
    - Determine which interior quadrant contains the seed: top-left, top-right, bottom-right, or bottom-left.
    - Rotate the chosen template by 0, 90, 180, or 270 degrees clockwise for those four quadrants, then align the rotated template to the corresponding interior corner of the fence.
    - Keep the grey fence borders and the template gallery unchanged.

    Key insight:
    - Color identifies which sprite to copy, while seed position identifies how to orient it, so the final pattern needs both a color lookup and a spatial rotation lookup.
    """
    return _solve(np.asarray(grid, dtype=int))


def _gallery_dims(templates: Sequence[Tuple[Tuple[int, int], ...]]) -> Tuple[int, int, int, int]:
    n = len(templates)
    rows = 1 if n <= 2 else 2
    cols = (n + rows - 1) // rows
    max_h = max(_shape_size(t)[0] for t in templates)
    max_w = max(_shape_size(t)[1] for t in templates)
    cell_h = max_h + 1
    cell_w = max_w + 1
    return rows, cols, cell_h, cell_w


def _paint_gallery(
    grid: np.ndarray,
    colors: Sequence[int],
    templates: Sequence[Tuple[Tuple[int, int], ...]],
    corner: str,
) -> Tuple[int, int, int, int]:
    rows, cols, cell_h, cell_w = _gallery_dims(templates)
    gallery_h = rows * cell_h + (rows - 1)
    gallery_w = cols * cell_w + (cols - 1)
    margin = random.randint(0, 1)
    if corner.startswith("T"):
        top = margin
    else:
        top = grid.shape[0] - gallery_h - margin
    if corner.endswith("L"):
        left = margin
    else:
        left = grid.shape[1] - gallery_w - margin

    for idx, (color, coords) in enumerate(zip(colors, templates)):
        rr = idx // cols
        cc = idx % cols
        h, w = _shape_size(coords)
        cell_top = top + rr * (cell_h + 1)
        cell_left = left + cc * (cell_w + 1)
        off_r = random.randint(0, cell_h - h)
        off_c = random.randint(0, cell_w - w)
        for dr, dc in coords:
            grid[cell_top + off_r + dr, cell_left + off_c + dc] = color
    return top, left, top + gallery_h - 1, left + gallery_w - 1


def _mark_reserved(
    reserved: np.ndarray,
    r0: int,
    c0: int,
    r1: int,
    c1: int,
    pad: int,
) -> None:
    rr0 = max(0, r0 - pad)
    cc0 = max(0, c0 - pad)
    rr1 = min(reserved.shape[0] - 1, r1 + pad)
    cc1 = min(reserved.shape[1] - 1, c1 + pad)
    reserved[rr0 : rr1 + 1, cc0 : cc1 + 1] = True


def _free_positions(
    reserved: np.ndarray,
    h: int,
    w: int,
    border: int,
) -> List[Tuple[int, int]]:
    H, W = reserved.shape
    out: List[Tuple[int, int]] = []
    for r in range(border, H - h - border + 1):
        for c in range(border, W - w - border + 1):
            if not reserved[r : r + h, c : c + w].any():
                out.append((r, c))
    return out


def _seed_position(inner_h: int, inner_w: int, quadrant: int) -> Tuple[int, int]:
    row_mid = inner_h // 2
    col_mid = inner_w // 2
    if quadrant == 0:
        rows = range(0, row_mid)
        cols = range(0, col_mid)
    elif quadrant == 1:
        rows = range(0, row_mid)
        cols = range(col_mid, inner_w)
    elif quadrant == 2:
        rows = range(row_mid, inner_h)
        cols = range(col_mid, inner_w)
    else:
        rows = range(row_mid, inner_h)
        cols = range(0, col_mid)
    return random.choice(list(rows)), random.choice(list(cols))


def _build_expected(
    grid: np.ndarray,
    fences: Sequence[dict],
    template_map: Dict[int, Tuple[Tuple[int, int], ...]],
) -> np.ndarray:
    out = grid.copy()
    for fence in fences:
        _paint_rotated_template(
            out,
            fence["color"],
            template_map[fence["color"]],
            fence["bbox"],
            fence["quadrant"],
        )
    return out


def _gen_attempt() -> np.ndarray | None:
    n_templates = random.randint(2, 4)
    colors = random.sample(COLOR_POOL, n_templates)
    templates = random.sample(list(PATTERN_LIBRARY), n_templates)
    corner = random.choice(["TL", "TR", "BL", "BR"])

    n_fences = random.randint(max(4, n_templates), 8)
    grid_h = random.randint(22 if n_fences <= 5 else 24, 30)
    grid_w = random.randint(22 if n_fences <= 5 else 24, 30)
    grid = np.zeros((grid_h, grid_w), dtype=int)

    gallery_box = _paint_gallery(grid, colors, templates, corner)
    reserved = np.zeros_like(grid, dtype=bool)
    _mark_reserved(reserved, *gallery_box, pad=1)

    quadrants = ([0, 1, 2, 3] * ((n_fences + 3) // 4))[:n_fences]
    random.shuffle(quadrants)

    fence_colors = list(colors)
    while len(fence_colors) < n_fences:
        fence_colors.append(random.choice(colors))
    random.shuffle(fence_colors)

    fence_specs = []
    size_hi = 6 if n_fences <= 5 else 5
    for quadrant, color in zip(quadrants, fence_colors):
        inner_h = random.randint(4, size_hi)
        inner_w = random.randint(4, size_hi)
        outer_h = inner_h + 2
        outer_w = inner_w + 2
        positions = _free_positions(reserved, outer_h, outer_w, border=1)
        if not positions:
            return None
        r0, c0 = random.choice(positions)
        r1 = r0 + outer_h - 1
        c1 = c0 + outer_w - 1
        grid[r0, c0 : c1 + 1] = FENCE
        grid[r1, c0 : c1 + 1] = FENCE
        grid[r0 : r1 + 1, c0] = FENCE
        grid[r0 : r1 + 1, c1] = FENCE
        sr, sc = _seed_position(inner_h, inner_w, quadrant)
        grid[r0 + 1 + sr, c0 + 1 + sc] = color
        fence_specs.append(
            {
                "bbox": (r0, c0, r1, c1),
                "quadrant": quadrant,
                "color": color,
            }
        )
        _mark_reserved(reserved, r0, c0, r1, c1, pad=1)

    template_map = {color: _normalize_coords(coords) for color, coords in zip(colors, templates)}
    expected = _build_expected(grid, fence_specs, template_map)

    try:
        produced = main(grid)
    except Exception:
        return None
    if not np.array_equal(produced, expected):
        return None
    return grid


def generate_input() -> np.ndarray:
    for _ in range(MAX_GEN_RETRIES):
        grid = _gen_attempt()
        if grid is not None:
            return grid
    raise RuntimeError("failed to generate valid input")
