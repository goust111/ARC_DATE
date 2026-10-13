import random
from typing import Iterable

import numpy as np


BG = 0
FRAME_COLOR = 5
COLOR_POOL = (1, 2, 3, 4, 6, 7, 8, 9)
GEN_RETRIES = 500


def _most_common_color(grid: np.ndarray) -> int:
    vals, counts = np.unique(grid, return_counts=True)
    return int(vals[np.argmax(counts)])


def _bbox(coords: np.ndarray) -> tuple[int, int, int, int]:
    rows = coords[:, 0]
    cols = coords[:, 1]
    return int(rows.min()), int(cols.min()), int(rows.max()), int(cols.max())


def _roll_mask(mask: np.ndarray, shift: tuple[int, int]) -> np.ndarray:
    return np.roll(mask, shift=shift, axis=(0, 1))


def _expand_mask(mask: np.ndarray, scale_r: int, scale_c: int) -> np.ndarray:
    return np.kron(mask.astype(bool), np.ones((scale_r, scale_c), dtype=bool))


def _find_frame(grid: np.ndarray, bg: int) -> tuple[int, tuple[int, int, int, int]]:
    best: tuple[int, int, tuple[int, int, int, int]] | None = None
    for color in np.unique(grid):
        color = int(color)
        if color == bg:
            continue
        coords = np.argwhere(grid == color)
        if coords.size == 0:
            continue
        r0, c0, r1, c1 = _bbox(coords)
        h = r1 - r0 + 1
        w = c1 - c0 + 1
        if h < 3 or w < 3:
            continue
        border = np.zeros((h, w), dtype=bool)
        border[0, :] = True
        border[-1, :] = True
        border[:, 0] = True
        border[:, -1] = True
        sub = grid[r0 : r1 + 1, c0 : c1 + 1]
        if not np.all(sub[border] == color):
            continue
        if np.any(sub[~border] == color):
            continue
        if np.count_nonzero(sub == color) != int(border.sum()):
            continue
        if not np.any(sub[1:-1, 1:-1] != bg):
            continue
        area = h * w
        if best is None or area > best[0]:
            best = (area, color, (r0, c0, r1, c1))
    assert best is not None, "expected one rectangular frame"
    return best[1], best[2]


def _extract_templates(
    grid: np.ndarray, bg: int, frame_bbox: tuple[int, int, int, int]
) -> tuple[list[dict], dict[int, tuple[int, int]]]:
    r0, c0, r1, c1 = frame_bbox
    interior = grid[r0 + 1 : r1, c0 + 1 : c1]
    seed_colors = [int(color) for color in np.unique(interior) if int(color) != bg]
    assert seed_colors, "expected colored seeds inside the frame"

    outside_rect = np.ones(grid.shape, dtype=bool)
    outside_rect[r0 : r1 + 1, c0 : c1 + 1] = False

    templates: list[dict] = []
    seeds: dict[int, tuple[int, int]] = {}
    for color in seed_colors:
        seed_coords = np.argwhere(interior == color)
        assert len(seed_coords) == 1, "each template color must have exactly one seed"
        seeds[color] = (int(seed_coords[0, 0]), int(seed_coords[0, 1]))

        outer_coords = np.argwhere((grid == color) & outside_rect)
        assert outer_coords.size > 0, "each seed color must have an outer template"
        tr0, tc0, tr1, tc1 = _bbox(outer_coords)
        mask = np.zeros((tr1 - tr0 + 1, tc1 - tc0 + 1), dtype=bool)
        mask[outer_coords[:, 0] - tr0, outer_coords[:, 1] - tc0] = True
        templates.append({"color": color, "bbox": (tr0, tc0, tr1, tc1), "mask": mask})

    shapes = {tuple(t["mask"].shape) for t in templates}
    assert len(shapes) == 1, "all outer templates must share one size"
    templates.sort(key=lambda t: (t["bbox"][0], t["bbox"][1]))
    return templates, seeds


def _resolve_alignment(mask: np.ndarray, seed_cell: tuple[int, int]) -> np.ndarray:
    variants: dict[bytes, np.ndarray] = {}
    for r, c in np.argwhere(mask):
        shifted = _roll_mask(mask, (seed_cell[0] - int(r), seed_cell[1] - int(c)))
        variants.setdefault(shifted.tobytes(), shifted)
    assert variants, "template mask must contain at least one colored cell"
    assert len(variants) == 1, "seed does not determine a unique translated template"
    return next(iter(variants.values()))


def _solve(grid: np.ndarray) -> np.ndarray:
    g = np.asarray(grid, dtype=int)
    bg = _most_common_color(g)
    frame_color, frame_bbox = _find_frame(g, bg)
    templates, seeds = _extract_templates(g, bg, frame_bbox)

    r0, c0, r1, c1 = frame_bbox
    inner_h = r1 - r0 - 1
    inner_w = c1 - c0 - 1
    template_h, template_w = map(int, templates[0]["mask"].shape)
    assert inner_h % template_h == 0 and inner_w % template_w == 0
    scale_r = inner_h // template_h
    scale_c = inner_w // template_w

    layers: list[tuple[int, np.ndarray]] = []
    for info in templates:
        color = int(info["color"])
        seed_r, seed_c = seeds[color]
        coarse_seed = (seed_r // scale_r, seed_c // scale_c)
        aligned = _resolve_alignment(info["mask"], coarse_seed)
        layers.append((color, aligned))

    filled = np.full((inner_h, inner_w), bg, dtype=int)
    for color, coarse_mask in reversed(layers):
        expanded = _expand_mask(coarse_mask, scale_r, scale_c)
        filled[expanded] = color

    out = g.copy()
    out[r0 + 1 : r1, c0 + 1 : c1] = filled
    return out


def main(grid: np.ndarray) -> np.ndarray:
    """
    Overlay the matching outer templates inside the grey frame by using each seed to choose the template phase.

    Input structure:
    - The background is black, and one grey one-cell-thick rectangle forms the only frame.
    - Inside the frame are several single-pixel seeds of distinct colors.
    - Outside the frame, each seed color appears again as one monochrome template; all templates share the same bounding-box size.

    Rule:
    - Find the outer template whose color matches each seed.
    - Treat the frame interior as a uniform block-scaled version of the template grid: every template cell expands to an equal-sized block, and the seed's block location selects the translated phase of that scaled template.
    - Expand every matched template to the frame size, align it so the seed lies in one colored block, and overlay the aligned layers inside the frame.
    - When layers overlap, the template whose outer copy is higher in the input grid overrides lower templates; leave the frame and the outer templates unchanged.

    Key insight:
    - A seed encodes two things at once: its color selects which outer template to use, and its position inside the frame selects the translated phase of that template after block scaling.
    """
    return _solve(grid)


def _canonical_roll(mask: np.ndarray) -> bytes:
    h, w = mask.shape
    best: bytes | None = None
    for dr in range(h):
        for dc in range(w):
            rolled = _roll_mask(mask, (dr, dc)).tobytes()
            if best is None or rolled < best:
                best = rolled
    assert best is not None
    return best


def _is_unambiguous_template(mask: np.ndarray) -> bool:
    if not np.any(mask) or np.all(mask):
        return False
    variants = set()
    for r, c in np.argwhere(mask):
        variants.add(_roll_mask(mask, (-int(r), -int(c))).tobytes())
        if len(variants) > 1:
            return False
    return True


def _uses_full_bbox(mask: np.ndarray) -> bool:
    return bool(mask[0, :].any() and mask[-1, :].any() and mask[:, 0].any() and mask[:, -1].any())


def _candidate_masks(h: int, w: int) -> list[np.ndarray]:
    rr, cc = np.indices((h, w))
    raw: list[np.ndarray] = []

    dot = np.zeros((h, w), dtype=bool)
    dot[0, 0] = True
    raw.append(dot)

    for r in range(h):
        mask = np.zeros((h, w), dtype=bool)
        mask[r, :] = True
        raw.append(mask)

    for c in range(w):
        mask = np.zeros((h, w), dtype=bool)
        mask[:, c] = True
        raw.append(mask)

    for mod in (2, 3):
        raw.append((rr % mod) == 0)
        raw.append((cc % mod) == 0)
        raw.append(((rr + cc) % mod) == 0)
        raw.append(((rr - cc) % mod) == 0)
        raw.append(((rr % mod) == 0) & ((cc % mod) == 0))

    if h == w:
        raw.append(np.eye(h, dtype=bool))
        raw.append(np.fliplr(np.eye(h, dtype=bool)))

    seen: set[bytes] = set()
    masks: list[np.ndarray] = []
    for mask in raw:
        if not _is_unambiguous_template(mask):
            continue
        if not _uses_full_bbox(mask):
            continue
        density = float(mask.mean())
        if density < 0.08 or density > 0.7:
            continue
        key = _canonical_roll(mask)
        if key in seen:
            continue
        seen.add(key)
        masks.append(mask.copy())
    return masks


def _render_templates(
    grid: np.ndarray, placements: Iterable[tuple[int, int, int, np.ndarray]]
) -> None:
    for top, left, color, mask in placements:
        h, w = mask.shape
        sub = grid[top : top + h, left : left + w]
        sub[mask] = color


def _composite_visible_counts(
    layers: list[tuple[int, np.ndarray]], scale_r: int, scale_c: int
) -> tuple[np.ndarray, dict[int, int], int]:
    h, w = layers[0][1].shape
    coverage = np.zeros((h, w), dtype=int)
    for _color, coarse in layers:
        coverage += coarse.astype(int)
    overlap = int(np.count_nonzero(coverage > 1))

    inner = np.full((h * scale_r, w * scale_c), BG, dtype=int)
    for color, coarse in reversed(layers):
        inner[_expand_mask(coarse, scale_r, scale_c)] = color

    visible: dict[int, int] = {}
    for color, _coarse in layers:
        visible[color] = int(np.count_nonzero(inner == color))
    return inner, visible, overlap


def _choose_seed_positions(
    rng: random.Random, layers: list[tuple[int, np.ndarray]], scale_r: int, scale_c: int
) -> dict[int, tuple[int, int]]:
    used_cells: set[tuple[int, int]] = set()
    seeds: dict[int, tuple[int, int]] = {}
    for color, coarse in layers:
        cells = [tuple(map(int, cell)) for cell in np.argwhere(coarse)]
        rng.shuffle(cells)
        cells = [cell for cell in cells if cell not in used_cells]
        if not cells:
            raise ValueError("not enough distinct colored blocks for seeds")
        cr, cc = cells[0]
        used_cells.add((cr, cc))
        sr = cr * scale_r + rng.randrange(scale_r)
        sc = cc * scale_c + rng.randrange(scale_c)
        seeds[color] = (sr, sc)
    return seeds


def _gen_attempt(rng: random.Random) -> np.ndarray | None:
    template_h = rng.randint(2, 5)
    template_w = rng.randint(2, 5)
    scale_r = rng.randint(1, 4)
    scale_c = rng.randint(1, 4)
    if scale_r == 1 and scale_c == 1:
        scale_r = rng.randint(2, 4)
    inner_h = template_h * scale_r
    inner_w = template_w * scale_c
    frame_h = inner_h + 2
    frame_w = inner_w + 2

    layer_count = rng.randint(2, 4)
    masks = _candidate_masks(template_h, template_w)
    if len(masks) < layer_count:
        return None
    rng.shuffle(masks)
    masks = masks[:layer_count]

    colors = list(rng.sample(COLOR_POOL, layer_count))
    coarse_layers: list[tuple[int, np.ndarray]] = []
    for color, mask in zip(colors, masks):
        true_cells = np.argwhere(mask)
        anchor_r, anchor_c = map(int, true_cells[rng.randrange(len(true_cells))])
        target_r = rng.randrange(template_h)
        target_c = rng.randrange(template_w)
        aligned = _roll_mask(mask, (target_r - anchor_r, target_c - anchor_c))
        coarse_layers.append((color, aligned))

    inner, visible, overlap = _composite_visible_counts(coarse_layers, scale_r, scale_c)
    min_visible = max(2, scale_r * scale_c)
    if overlap == 0:
        return None
    if any(count < min_visible for count in visible.values()):
        return None
    if len({int(v) for v in np.unique(inner) if int(v) != BG}) < 2:
        return None

    try:
        seeds = _choose_seed_positions(rng, coarse_layers, scale_r, scale_c)
    except ValueError:
        return None

    gap_rows = [rng.randint(1, 2) for _ in range(layer_count - 1)]
    strip_h = template_h * layer_count + sum(gap_rows)
    strip_w = template_w
    side_gap = rng.randint(2, 4)

    min_grid_h = max(strip_h, frame_h) + 2
    min_grid_w = strip_w + side_gap + frame_w + 2
    if min_grid_h > 30 or min_grid_w > 30:
        return None
    grid_h = rng.randint(min_grid_h, min(30, min_grid_h + 4))
    grid_w = rng.randint(min_grid_w, min(30, min_grid_w + 4))

    frame_top = rng.randint(1, grid_h - frame_h - 1)
    strip_top = rng.randint(1, grid_h - strip_h - 1)
    side = rng.choice(("left", "right"))
    if side == "left":
        max_strip_left = grid_w - (strip_w + side_gap + frame_w) - 1
        if max_strip_left < 1:
            return None
        strip_left = rng.randint(1, max_strip_left)
        frame_left = strip_left + strip_w + side_gap
    else:
        max_frame_left = grid_w - (frame_w + side_gap + strip_w) - 1
        if max_frame_left < 1:
            return None
        frame_left = rng.randint(1, max_frame_left)
        strip_left = frame_left + frame_w + side_gap

    grid = np.full((grid_h, grid_w), BG, dtype=int)

    frame_bottom = frame_top + frame_h - 1
    frame_right = frame_left + frame_w - 1
    grid[frame_top, frame_left : frame_right + 1] = FRAME_COLOR
    grid[frame_bottom, frame_left : frame_right + 1] = FRAME_COLOR
    grid[frame_top : frame_bottom + 1, frame_left] = FRAME_COLOR
    grid[frame_top : frame_bottom + 1, frame_right] = FRAME_COLOR

    placements: list[tuple[int, int, int, np.ndarray]] = []
    row = strip_top
    for idx, (color, coarse) in enumerate(coarse_layers):
        placements.append((row, strip_left, color, coarse))
        row += template_h
        if idx < len(gap_rows):
            row += gap_rows[idx]
    _render_templates(grid, placements)

    for color, (sr, sc) in seeds.items():
        grid[frame_top + 1 + sr, frame_left + 1 + sc] = color

    expected = grid.copy()
    expected[frame_top + 1 : frame_bottom, frame_left + 1 : frame_right] = inner
    solved = main(grid)
    if not np.array_equal(solved, expected):
        return None
    return grid


def generate_input() -> np.ndarray:
    rng = random.Random()
    for _ in range(GEN_RETRIES):
        grid = _gen_attempt(rng)
        if grid is not None:
            return grid
    raise RuntimeError("failed to generate a valid puzzle instance")
