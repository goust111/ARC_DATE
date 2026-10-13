from __future__ import annotations

import numpy as np


BG_COLOR = 7
FRAME_COLOR = 5
SEED_COLOR = 2
TEMPLATE_COLORS = (0, 1, 3, 4, 6, 8, 9)

TEMPLATE_COUNT_RANGE = (2, 6)
TEMPLATE_SIZE_RANGE = (3, 6)
SCALE_RANGE = (2, 5)
GEN_RETRIES = 400
MAX_GRID_DIM = 30


def get_bg(grid: np.ndarray) -> int:
    values, counts = np.unique(grid, return_counts=True)
    return int(values[np.argmax(counts)])


def bbox_coords(coords: np.ndarray) -> tuple[int, int, int, int]:
    return (
        int(np.min(coords[:, 0])),
        int(np.min(coords[:, 1])),
        int(np.max(coords[:, 0])),
        int(np.max(coords[:, 1])),
    )


def components_mask(mask: np.ndarray) -> list[np.ndarray]:
    h, w = mask.shape
    seen = np.zeros_like(mask, dtype=bool)
    components: list[np.ndarray] = []
    for r in range(h):
        for c in range(w):
            if not mask[r, c] or seen[r, c]:
                continue
            stack = [(r, c)]
            seen[r, c] = True
            points: list[tuple[int, int]] = []
            while stack:
                cr, cc = stack.pop()
                points.append((cr, cc))
                for nr, nc in ((cr - 1, cc), (cr + 1, cc), (cr, cc - 1), (cr, cc + 1)):
                    if 0 <= nr < h and 0 <= nc < w and mask[nr, nc] and not seen[nr, nc]:
                        seen[nr, nc] = True
                        stack.append((nr, nc))
            components.append(np.array(points, dtype=int))
    return components


def _find_frame(grid: np.ndarray, bg: int) -> tuple[int, tuple[int, int, int, int]]:
    candidates: list[tuple[int, int, tuple[int, int, int, int]]] = []
    for color in np.unique(grid):
        color = int(color)
        if color == bg:
            continue
        for coords in components_mask(grid == color):
            r0, c0, r1, c1 = bbox_coords(coords)
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
            if int(coords.shape[0]) == int(border.sum()) and np.array_equal(sub == color, border):
                candidates.append((int(border.sum()), color, (r0, c0, r1, c1)))

    assert candidates, "expected a rectangular frame"
    largest_area = max(area for area, _color, _bbox in candidates)
    largest = [candidate for candidate in candidates if candidate[0] == largest_area]
    assert len(largest) == 1, "frame must be structurally unique"
    _area, color, bbox = largest[0]
    return color, bbox


def _extract_templates_above_frame(
    grid: np.ndarray, bg: int, frame_color: int, frame_r0: int
) -> list[np.ndarray]:
    top = grid[:frame_r0]
    templates: list[np.ndarray] = []
    for color in np.unique(top):
        color = int(color)
        if color == bg or color == frame_color:
            continue
        for coords in components_mask(top == color):
            r0, c0, r1, c1 = bbox_coords(coords)
            mask = np.zeros((r1 - r0 + 1, c1 - c0 + 1), dtype=bool)
            mask[coords[:, 0] - r0, coords[:, 1] - c0] = True
            templates.append(mask)
    assert templates, "expected templates above the frame"
    return templates


def _scale_mask(mask: np.ndarray, scale_h: int, scale_w: int) -> np.ndarray:
    return np.kron(mask, np.ones((scale_h, scale_w), dtype=bool))


def main(grid: np.ndarray) -> np.ndarray:
    """
    Pick the upper template whose scaled mask contains all seeds and complete it inside the frame.
    """
    g = np.asarray(grid, dtype=int)
    bg = get_bg(g)
    frame_color, (r0, c0, r1, c1) = _find_frame(g, bg)

    out = g.copy()
    interior = out[r0 + 1 : r1, c0 + 1 : c1]
    assert interior.size, "frame must have a non-empty interior"

    fill_colors = set(map(int, np.unique(interior))) - {bg, frame_color}
    assert len(fill_colors) == 1, "frame interior must contain exactly one seed color"
    fill_color = fill_colors.pop()

    seeds = interior == fill_color
    assert np.any(seeds), "frame interior must contain seed pixels"
    ih, iw = seeds.shape

    matches: list[np.ndarray] = []
    for template in _extract_templates_above_frame(g, bg, frame_color, r0):
        th, tw = template.shape
        if ih % th != 0 or iw % tw != 0:
            continue
        scaled = _scale_mask(template, ih // th, iw // tw)
        if np.all(~seeds | scaled):
            matches.append(scaled)

    assert len(matches) == 1, "seed pixels must select exactly one template"
    interior[:, :] = bg
    interior[matches[0]] = fill_color
    return out


def _touches_full_bbox(mask: np.ndarray) -> bool:
    return bool(mask[0].any() and mask[-1].any() and mask[:, 0].any() and mask[:, -1].any())


def _is_border_rectangle(mask: np.ndarray) -> bool:
    border = np.zeros_like(mask, dtype=bool)
    border[0, :] = True
    border[-1, :] = True
    border[:, 0] = True
    border[:, -1] = True
    return np.array_equal(mask, border)


def _mask_signature(mask: np.ndarray) -> tuple[int, int, tuple[int, ...]]:
    return mask.shape[0], mask.shape[1], tuple(int(v) for v in mask.astype(int).ravel())


def _random_connected_mask(rng: np.random.Generator, h: int, w: int) -> np.ndarray:
    area = h * w
    min_cells = max(4, int(np.ceil(area * 0.35)))
    max_cells = min(area - 1, int(np.floor(area * 0.72)))
    for _ in range(200):
        target_cells = int(rng.integers(min_cells, max_cells + 1))
        mask = np.zeros((h, w), dtype=bool)
        start = (int(rng.integers(0, h)), int(rng.integers(0, w)))
        mask[start] = True
        frontier: set[tuple[int, int]] = set()

        def add_frontier(r: int, c: int) -> None:
            for nr, nc in ((r - 1, c), (r + 1, c), (r, c - 1), (r, c + 1)):
                if 0 <= nr < h and 0 <= nc < w and not mask[nr, nc]:
                    frontier.add((nr, nc))

        add_frontier(*start)
        while int(mask.sum()) < target_cells and frontier:
            cell = tuple(frontier.pop()) if len(frontier) == 1 else list(frontier)[int(rng.integers(0, len(frontier)))]
            frontier.discard(cell)
            mask[cell] = True
            add_frontier(*cell)

        if int(mask.sum()) != target_cells:
            continue
        if not _touches_full_bbox(mask):
            continue
        if _is_border_rectangle(mask):
            continue
        return mask
    raise RuntimeError("failed to generate a connected template")


def _layout_templates(
    rng: np.random.Generator, masks: list[np.ndarray]
) -> tuple[dict[int, tuple[int, int]], int, int]:
    count = len(masks)
    order = list(map(int, rng.permutation(count)))
    use_two_rows = count >= 4 or (count >= 3 and bool(rng.integers(0, 2)))
    if use_two_rows:
        split = int(rng.integers(1, count))
        rows = [order[:split], order[split:]]
        rows = [row for row in rows if row]
    else:
        rows = [order]

    placements: dict[int, tuple[int, int]] = {}
    top_h = 0
    top_w = 0
    row_gap = int(rng.integers(1, 3))

    for row_index, row in enumerate(rows):
        row_h = max(int(masks[idx].shape[0]) for idx in row)
        x = 0
        item_positions: list[tuple[int, int]] = []
        for idx in row:
            if item_positions:
                x += int(rng.integers(1, 4))
            item_positions.append((idx, x))
            x += int(masks[idx].shape[1])
        for idx, c0 in item_positions:
            h = int(masks[idx].shape[0])
            y_off = int(rng.integers(0, row_h - h + 1))
            placements[idx] = (top_h + y_off, c0)
        top_w = max(top_w, x)
        top_h += row_h
        if row_index != len(rows) - 1:
            top_h += row_gap

    return placements, top_h, top_w


def _choose_seed_positions(
    rng: np.random.Generator, target_scaled: np.ndarray, distractors: list[np.ndarray]
) -> np.ndarray | None:
    target_cells = [tuple(map(int, cell)) for cell in np.argwhere(target_scaled)]
    area = len(target_cells)
    if area < 2:
        return None

    chosen: list[tuple[int, int]] = []
    remaining = set(range(len(distractors)))
    while remaining:
        best_cell: tuple[int, int] | None = None
        best_cover: set[int] = set()
        for cell in target_cells:
            cover = {idx for idx in remaining if not bool(distractors[idx][cell])}
            if len(cover) > len(best_cover):
                best_cover = cover
                best_cell = cell
        if best_cell is None or not best_cover:
            return None
        chosen.append(best_cell)
        remaining -= best_cover

    if len(chosen) > max(5, area // 3):
        return None

    min_seed_count = max(2, len(chosen))
    max_seed_count = min(area - 1, max(min_seed_count, int(np.ceil(area * 0.45))))
    if max_seed_count < min_seed_count:
        return None

    seed_count = int(rng.integers(min_seed_count, max_seed_count + 1))
    selected = set(chosen)
    available = [cell for cell in target_cells if cell not in selected]
    if seed_count > len(selected):
        extra_indices = rng.choice(len(available), size=seed_count - len(selected), replace=False)
        for idx in np.atleast_1d(extra_indices):
            selected.add(available[int(idx)])
    return np.array(sorted(selected), dtype=int)


def _paint_mask(grid: np.ndarray, mask: np.ndarray, color: int, top: int, left: int) -> None:
    h, w = mask.shape
    sub = grid[top : top + h, left : left + w]
    sub[mask] = color


def _build_expected(
    grid: np.ndarray, frame_r0: int, frame_c0: int, scaled_mask: np.ndarray
) -> np.ndarray:
    out = grid.copy()
    ih, iw = scaled_mask.shape
    interior = out[frame_r0 + 1 : frame_r0 + 1 + ih, frame_c0 + 1 : frame_c0 + 1 + iw]
    interior[:, :] = BG_COLOR
    interior[scaled_mask] = SEED_COLOR
    return out


def _gen_attempt() -> np.ndarray | None:
    rng = np.random.default_rng()
    count = int(rng.integers(*TEMPLATE_COUNT_RANGE))

    target_h = int(rng.integers(*TEMPLATE_SIZE_RANGE))
    target_w = int(rng.integers(*TEMPLATE_SIZE_RANGE))
    target_mask = _random_connected_mask(rng, target_h, target_w)

    same_bbox_count = 0
    if count >= 2 and rng.random() < 0.7:
        same_bbox_count = 1
    if count >= 4 and rng.random() < 0.3:
        same_bbox_count += 1
    same_bbox_count = min(same_bbox_count, count - 1)

    templates = [{"mask": target_mask.copy(), "target": True}]
    used = {_mask_signature(target_mask)}

    while len(templates) < 1 + same_bbox_count:
        mask = _random_connected_mask(rng, target_h, target_w)
        sig = _mask_signature(mask)
        if sig in used:
            continue
        used.add(sig)
        templates.append({"mask": mask, "target": False})

    while len(templates) < count:
        h = int(rng.integers(*TEMPLATE_SIZE_RANGE))
        w = int(rng.integers(*TEMPLATE_SIZE_RANGE))
        mask = _random_connected_mask(rng, h, w)
        sig = _mask_signature(mask)
        if sig in used:
            continue
        used.add(sig)
        templates.append({"mask": mask, "target": False})

    order = list(map(int, rng.permutation(count)))
    templates = [templates[idx] for idx in order]
    target_index = next(i for i, template in enumerate(templates) if template["target"])

    if bool(rng.integers(0, 2)):
        scale_h = scale_w = int(rng.integers(*SCALE_RANGE))
    else:
        scale_h = int(rng.integers(*SCALE_RANGE))
        scale_w = int(rng.integers(*SCALE_RANGE))
        if scale_h == scale_w:
            scale_w = 2 + (scale_w % 3)

    target_scaled = _scale_mask(templates[target_index]["mask"], scale_h, scale_w)
    ih, iw = target_scaled.shape
    frame_h = ih + 2
    frame_w = iw + 2

    scaled_candidates: list[np.ndarray | None] = []
    for template in templates:
        mask = template["mask"]
        th, tw = mask.shape
        if ih % th != 0 or iw % tw != 0:
            scaled_candidates.append(None)
            continue
        scaled_candidates.append(_scale_mask(mask, ih // th, iw // tw))

    distractors = [
        scaled
        for idx, scaled in enumerate(scaled_candidates)
        if idx != target_index and scaled is not None
    ]
    if any(np.array_equal(target_scaled, scaled) for scaled in distractors):
        return None

    seeds = _choose_seed_positions(rng, target_scaled, distractors)
    if seeds is None:
        return None

    masks = [template["mask"] for template in templates]
    placements, top_h, top_w = _layout_templates(rng, masks)
    if max(top_w, frame_w) + 5 > MAX_GRID_DIM:
        return None

    top_margin = int(rng.integers(1, 4))
    gap_to_frame = int(rng.integers(2, 5))
    frame_r0 = top_margin + top_h + gap_to_frame
    grid_h = frame_r0 + frame_h + int(rng.integers(1, 4))
    if grid_h > MAX_GRID_DIM:
        return None

    grid_w = max(top_w, frame_w) + int(rng.integers(4, 8))
    if grid_w > MAX_GRID_DIM:
        return None

    top_shift_c = int(rng.integers(1, grid_w - top_w))
    frame_c0 = int(rng.integers(1, grid_w - frame_w))

    grid = np.full((grid_h, grid_w), BG_COLOR, dtype=int)
    palette = list(TEMPLATE_COLORS)
    rng.shuffle(palette)
    colors = palette[:count]

    for idx, mask in enumerate(masks):
        local_r, local_c = placements[idx]
        _paint_mask(grid, mask, int(colors[idx]), top_margin + local_r, top_shift_c + local_c)

    grid[frame_r0, frame_c0 : frame_c0 + frame_w] = FRAME_COLOR
    grid[frame_r0 + frame_h - 1, frame_c0 : frame_c0 + frame_w] = FRAME_COLOR
    grid[frame_r0 : frame_r0 + frame_h, frame_c0] = FRAME_COLOR
    grid[frame_r0 : frame_r0 + frame_h, frame_c0 + frame_w - 1] = FRAME_COLOR
    for r, c in seeds:
        grid[frame_r0 + 1 + int(r), frame_c0 + 1 + int(c)] = SEED_COLOR

    expected = _build_expected(grid, frame_r0, frame_c0, target_scaled)
    try:
        actual = main(grid)
    except Exception:
        return None
    if not np.array_equal(actual, expected):
        return None
    return grid


def generate_input() -> np.ndarray:
    for _ in range(GEN_RETRIES):
        grid = _gen_attempt()
        if grid is not None:
            return grid
    raise RuntimeError("failed to generate a valid input")
