from __future__ import annotations

import numpy as np

from arc_utils import bbox_coords, find_components, get_bg, paste


def _find_frame(g: np.ndarray, bg: int) -> tuple[int, tuple[int, int, int, int]]:
    candidates: list[tuple[int, int, tuple[int, int, int, int]]] = []
    for comp in find_components(g, bg=bg, connectivity=4, monochromatic=True):
        color = int(comp["color"])
        r0, c0, r1, c1 = map(int, comp["bbox"])
        h = r1 - r0 + 1
        w = c1 - c0 + 1
        if h < 3 or w < 3:
            continue

        border = np.zeros((h, w), dtype=bool)
        border[0, :] = True
        border[-1, :] = True
        border[:, 0] = True
        border[:, -1] = True

        sub = g[r0 : r1 + 1, c0 : c1 + 1]
        if int(comp["area"]) == int(border.sum()) and np.array_equal(sub == color, border):
            candidates.append((int(comp["area"]), color, (r0, c0, r1, c1)))

    assert candidates, "expected one rectangular frame"
    max_area = max(area for area, _color, _bbox in candidates)
    largest = [candidate for candidate in candidates if candidate[0] == max_area]
    assert len(largest) == 1, "rectangular frame must be structurally unique"
    _area, color, frame_bbox = largest[0]
    return color, frame_bbox


def _templates_above_frame(g: np.ndarray, bg: int, frame_r0: int) -> list[np.ndarray]:
    top = g[:frame_r0]
    templates: list[np.ndarray] = []
    for color in np.unique(top):
        color = int(color)
        if color == bg:
            continue
        coords = np.argwhere(top == color)
        r0, c0, r1, c1 = bbox_coords(coords)
        templates.append(top[r0 : r1 + 1, c0 : c1 + 1] == color)

    assert templates, "expected upper template masks"
    return templates


def main(grid: np.ndarray) -> np.ndarray:
    """
    Complete the framed area by expanding the unique upper template that contains the existing seed pixels.

    Input structure:
    - A teal background contains small colored template masks above a larger rectangular frame.
    - The frame encloses one fill color arranged as a partial block-scaled copy of one template.
    - The frame interior dimensions are an integer square scale of the matching template bbox.

    Rule:
    - Crop to the rectangular frame.
    - Identify the unique upper template whose block-scaled mask covers every existing fill pixel inside the frame.
    - Reset the frame interior to background and paint the whole scaled mask with the fill color.

    Key insight:
    The lower colored pixels are seed cells that reveal which upper template should be completed inside the frame.
    """
    g = np.asarray(grid, dtype=int)
    bg = int(get_bg(g))
    frame_color, (r0, c0, r1, c1) = _find_frame(g, bg)

    out = g[r0 : r1 + 1, c0 : c1 + 1].copy()
    interior = out[1:-1, 1:-1]
    fill_colors = set(map(int, np.unique(interior))) - {bg, frame_color}
    assert len(fill_colors) == 1, "frame interior must contain exactly one fill color"
    fill_color = fill_colors.pop()

    seed = interior == fill_color
    assert np.any(seed), "frame must contain seed pixels"
    ih, iw = seed.shape

    matches: list[np.ndarray] = []
    for template in _templates_above_frame(g, bg, r0):
        th, tw = template.shape
        if ih % th != 0 or iw % tw != 0:
            continue
        scale_h = ih // th
        scale_w = iw // tw
        if scale_h != scale_w:
            continue

        scaled = np.kron(template, np.ones((scale_h, scale_w), dtype=bool))
        if np.all(~seed | scaled):
            matches.append(scaled)

    assert len(matches) == 1, "seed pixels must select exactly one scaled template"
    completed = matches[0]
    assert completed.shape == interior.shape

    interior[:, :] = bg
    interior[completed] = fill_color
    return out

# --------------- generator (att2) ---------------

GEN_BG = 8
GEN_GRID_SHAPE = (27, 21)
GEN_TEMPLATE_SHAPES = [(3, 3), (3, 4), (4, 3)]
GEN_TEMPLATE_COUNT_RANGE = (2, 6)
GEN_TEMPLATE_ROW = 1
GEN_TEMPLATE_LEFT_MIN = 1
GEN_TEMPLATE_GAP = 1
GEN_MAX_TEMPLATE_SPAN = 19
GEN_DENSITY_MIN_NUM = 1
GEN_DENSITY_MIN_DEN = 2
GEN_DENSITY_MAX_NUM = 5
GEN_DENSITY_MAX_DEN = 6
GEN_SCALE_RANGE = (2, 5)
GEN_SEED_COUNTS = (2, 3)
GEN_SEED_FRACTION_RANGE = (0.20, 0.60)
GEN_FRAME_ROW_GAP = 2
GEN_FRAME_LEFT_MIN = 1
GEN_ATTEMPTS = 300
GEN_PALETTE = [0, 1, 2, 3, 4, 5, 6, 7, 9]

def _gen_has_full_bbox(mask: np.ndarray) -> bool:
    return bool(mask[0].any() and mask[-1].any() and mask[:, 0].any() and mask[:, -1].any())


def _gen_valid_masks(th: int, tw: int) -> list[np.ndarray]:
    n = int(th * tw)
    lo = (GEN_DENSITY_MIN_NUM * n + GEN_DENSITY_MIN_DEN - 1) // GEN_DENSITY_MIN_DEN
    hi = (GEN_DENSITY_MAX_NUM * n) // GEN_DENSITY_MAX_DEN
    masks: list[np.ndarray] = []
    for bits in range(1 << n):
        count = int(bits.bit_count())
        if count < lo or count > hi:
            continue
        flat = np.array([(bits >> k) & 1 for k in range(n)], dtype=bool)
        mask = flat.reshape(th, tw)
        if _gen_has_full_bbox(mask):
            masks.append(mask)
    return masks


def _gen_colored_patch(mask: np.ndarray, color: int, bg: int) -> np.ndarray:
    patch = np.full(mask.shape, bg, dtype=int)
    patch[mask] = int(color)
    return patch


def _gen_draw_frame(shape: tuple[int, int], bg: int, frame_color: int) -> np.ndarray:
    patch = np.full(shape, bg, dtype=int)
    patch[0, :] = frame_color
    patch[-1, :] = frame_color
    patch[:, 0] = frame_color
    patch[:, -1] = frame_color
    return patch


def _gen_frame_patch(
    target_mask: np.ndarray,
    seed_cells: np.ndarray,
    scale: int,
    bg: int,
    frame_color: int,
    fill_color: int,
) -> tuple[np.ndarray, np.ndarray]:
    th, tw = map(int, target_mask.shape)
    scaled = np.kron(target_mask, np.ones((scale, scale), dtype=bool))
    out = _gen_draw_frame((th * scale + 2, tw * scale + 2), bg, frame_color)
    out[1:-1, 1:-1][scaled] = fill_color

    frame = _gen_draw_frame(out.shape, bg, frame_color)
    for r, c in seed_cells:
        rr = int(r) * scale
        cc = int(c) * scale
        frame[1 + rr : 1 + rr + scale, 1 + cc : 1 + cc + scale] = fill_color
    return frame, out


def _gen_attempt() -> np.ndarray | None:
    rng = np.random.default_rng()
    bg = GEN_BG
    count = int(rng.integers(*GEN_TEMPLATE_COUNT_RANGE))
    shape_choices = [
        shape for shape in GEN_TEMPLATE_SHAPES if count * shape[1] + count - 1 <= GEN_MAX_TEMPLATE_SPAN
    ]
    th, tw = shape_choices[int(rng.integers(0, len(shape_choices)))]
    scale = int(rng.integers(*GEN_SCALE_RANGE))

    valid_masks = _gen_valid_masks(th, tw)
    if len(valid_masks) < count:
        return None

    target = valid_masks[int(rng.integers(0, len(valid_masks)))]
    true_cells = np.argwhere(target)
    target_area = int(len(true_cells))
    min_seed_frac, max_seed_frac = GEN_SEED_FRACTION_RANGE
    seed_counts = [k for k in GEN_SEED_COUNTS if min_seed_frac <= k / target_area <= max_seed_frac]
    if not seed_counts:
        return None
    seed_count = int(seed_counts[int(rng.integers(0, len(seed_counts)))])
    seed_idx = rng.choice(target_area, size=seed_count, replace=False)
    seed_cells = true_cells[seed_idx]

    covers_seed = np.array([bool(mask[seed_cells[:, 0], seed_cells[:, 1]].all()) for mask in valid_masks])
    distractor_pool = [mask for mask, covers in zip(valid_masks, covers_seed) if not covers]
    if len(distractor_pool) < count - 1:
        return None
    distractor_idx = rng.choice(len(distractor_pool), size=count - 1, replace=False)
    distractors = [distractor_pool[int(i)] for i in distractor_idx]

    masks = distractors + [target]
    rng.shuffle(masks)

    colors = list(GEN_PALETTE)
    rng.shuffle(colors)
    frame_color = int(colors[0])
    fill_color = int(colors[1])
    template_colors = [int(c) for c in colors[2 : 2 + count]]

    frame, expected = _gen_frame_patch(target, seed_cells, scale, bg, frame_color, fill_color)
    grid_h, grid_w = GEN_GRID_SHAPE
    grid = np.full(GEN_GRID_SHAPE, bg, dtype=int)

    total_template_w = count * tw + count - 1
    template_c0 = int(rng.integers(GEN_TEMPLATE_LEFT_MIN, grid_w - total_template_w))
    c = template_c0
    for mask, color in zip(masks, template_colors):
        paste(grid, _gen_colored_patch(mask, color, bg), GEN_TEMPLATE_ROW, c, transparent=bg)
        c += tw + GEN_TEMPLATE_GAP

    fh, fw = map(int, frame.shape)
    min_r0 = th + GEN_FRAME_ROW_GAP
    max_r0 = grid_h - fh
    if min_r0 > max_r0:
        return None
    frame_r0 = int(rng.integers(min_r0, max_r0 + 1))
    frame_c0 = int(rng.integers(GEN_FRAME_LEFT_MIN, grid_w - fw))
    paste(grid, frame, frame_r0, frame_c0)

    try:
        if not np.array_equal(main(grid), expected):
            return None
    except Exception:
        return None
    return grid


def generate_input() -> np.ndarray:
    for _ in range(GEN_ATTEMPTS):
        grid = _gen_attempt()
        if grid is not None:
            return grid
    raise RuntimeError("failed to generate valid 8a6d367c input")
