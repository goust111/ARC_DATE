import random
from collections import defaultdict

import numpy as np


LEGEND_STRIP_WIDTH = 3
GRAY_COLOR = 5
MAX_GENERATION_ATTEMPTS = 400
_OFFSETS4 = ((1, 0), (-1, 0), (0, 1), (0, -1))


def _get_bg(grid: np.ndarray) -> int:
    values, counts = np.unique(grid, return_counts=True)
    return int(values[np.argmax(counts)])


def _find_components(grid: np.ndarray, bg: int) -> list[dict]:
    h, w = map(int, grid.shape)
    seen = np.zeros((h, w), dtype=bool)
    components: list[dict] = []

    for r in range(h):
        for c in range(w):
            if seen[r, c] or int(grid[r, c]) == int(bg):
                continue
            color = int(grid[r, c])
            stack = [(r, c)]
            seen[r, c] = True
            coords: list[tuple[int, int]] = []

            while stack:
                cr, cc = stack.pop()
                coords.append((int(cr), int(cc)))
                for dr, dc in _OFFSETS4:
                    nr = int(cr + dr)
                    nc = int(cc + dc)
                    if not (0 <= nr < h and 0 <= nc < w):
                        continue
                    if seen[nr, nc] or int(grid[nr, nc]) != color:
                        continue
                    seen[nr, nc] = True
                    stack.append((nr, nc))

            rows = [rc[0] for rc in coords]
            cols = [rc[1] for rc in coords]
            components.append(
                {
                    "color": color,
                    "coords": np.asarray(coords, dtype=int),
                    "bbox": (min(rows), min(cols), max(rows), max(cols)),
                    "area": len(coords),
                }
            )

    return components


def _bbox_shape(bbox: tuple[int, int, int, int]) -> tuple[int, int]:
    r0, c0, r1, c1 = bbox
    return int(r1 - r0 + 1), int(c1 - c0 + 1)


def _component_patch(comp: dict, bg: int) -> np.ndarray:
    r0, c0, r1, c1 = comp["bbox"]
    patch = np.full((r1 - r0 + 1, c1 - c0 + 1), int(bg), dtype=int)
    for r, c in comp["coords"]:
        patch[int(r - r0), int(c - c0)] = int(comp["color"])
    return patch


def _legend_side(components: list[dict], width: int) -> str:
    def edge_candidates(side: str) -> list[dict]:
        out = []
        for comp in components:
            h, w = _bbox_shape(comp["bbox"])
            if h > 3 or w > 3:
                continue
            r0, c0, r1, c1 = comp["bbox"]
            if side == "left" and c1 < LEGEND_STRIP_WIDTH:
                out.append(comp)
            if side == "right" and c0 >= width - LEGEND_STRIP_WIDTH:
                out.append(comp)
        return out

    left = edge_candidates("left")
    right = edge_candidates("right")
    left_valid = len(left) >= 2 and len({int(comp["color"]) for comp in left}) >= 2
    right_valid = len(right) >= 2 and len({int(comp["color"]) for comp in right}) >= 2
    assert left_valid ^ right_valid, "expected exactly one legend strip side"
    return "left" if left_valid else "right"


def _paste(dst: np.ndarray, patch: np.ndarray, top: int, left: int, bg: int) -> None:
    ph, pw = map(int, patch.shape)
    assert 0 <= top <= dst.shape[0] - ph
    assert 0 <= left <= dst.shape[1] - pw
    mask = patch != int(bg)
    target = dst[top : top + ph, left : left + pw]
    assert np.all(target[mask] == int(bg))
    target[mask] = patch[mask]


def _aligned_origin(
    obj_bbox: tuple[int, int, int, int], patch_shape: tuple[int, int]
) -> tuple[int, int]:
    r0, c0, r1, c1 = obj_bbox
    ph, pw = patch_shape
    top_num = int(r0 + r1 - (ph - 1))
    left_num = int(c0 + c1 - (pw - 1))
    assert top_num % 2 == 0 and left_num % 2 == 0, "center alignment must land on cells"
    return top_num // 2, left_num // 2


def main(grid: np.ndarray) -> np.ndarray:
    """
    Replace each gray object by the legend template with the same pixel count, then remove the legend strip.
    """
    g = np.asarray(grid, dtype=int)
    assert g.ndim == 2
    height, width = map(int, g.shape)
    assert width > LEGEND_STRIP_WIDTH

    bg = _get_bg(g)
    components = _find_components(g, bg)
    assert components, "expected non-background components"

    side = _legend_side(components, width)
    legend_cols = slice(0, LEGEND_STRIP_WIDTH) if side == "left" else slice(width - LEGEND_STRIP_WIDTH, width)
    core = g[:, LEGEND_STRIP_WIDTH:] if side == "left" else g[:, : width - LEGEND_STRIP_WIDTH]

    legend_grid = g[:, legend_cols]
    legend_bg = _get_bg(legend_grid)
    assert legend_bg == bg
    legend_components = _find_components(legend_grid, bg)
    assert len(legend_components) >= 2

    templates: dict[int, np.ndarray] = {}
    for comp in legend_components:
        h, w = _bbox_shape(comp["bbox"])
        assert h <= 3 and w <= 3
        area = int(comp["area"])
        assert area not in templates, "legend areas must be unique"
        templates[area] = _component_patch(comp, bg)

    core_colors = [int(color) for color in np.unique(core) if int(color) != int(bg)]
    assert len(core_colors) == 1, "core should contain only gray objects"
    object_color = core_colors[0]
    core_components = _find_components(core, bg)
    assert core_components and all(int(comp["color"]) == object_color for comp in core_components)

    out = np.full(core.shape, int(bg), dtype=int)
    for comp in core_components:
        area = int(comp["area"])
        assert area in templates, "each gray object must match a legend area"
        patch = templates[area]
        top, left = _aligned_origin(comp["bbox"], patch.shape)
        _paste(out, patch, top, left, bg)

    return out


def _mask_connected(mask: np.ndarray) -> bool:
    coords = np.argwhere(mask)
    if len(coords) == 0:
        return False

    start = tuple(map(int, coords[0]))
    stack = [start]
    seen = {start}

    while stack:
        r, c = stack.pop()
        for dr, dc in _OFFSETS4:
            nr = int(r + dr)
            nc = int(c + dc)
            if not (0 <= nr < 3 and 0 <= nc < 3) or not bool(mask[nr, nc]):
                continue
            nxt = (nr, nc)
            if nxt not in seen:
                seen.add(nxt)
                stack.append(nxt)

    return len(seen) == int(mask.sum())


def _build_shape_bank() -> dict[int, list[np.ndarray]]:
    bank: dict[int, list[np.ndarray]] = defaultdict(list)
    for bits in range(1, 1 << 9):
        flat = np.array([(bits >> i) & 1 for i in range(9)], dtype=bool)
        mask = flat.reshape(3, 3)
        if not _mask_connected(mask):
            continue
        rows, cols = np.where(mask)
        if rows.min() != 0 or rows.max() != 2 or cols.min() != 0 or cols.max() != 2:
            continue
        bank[int(mask.sum())].append(mask.copy())
    return dict(bank)


_SHAPE_BANK = _build_shape_bank()
_USABLE_AREAS = [area for area, masks in _SHAPE_BANK.items() if int(area) >= 4 and len(masks) >= 2]
assert len(_USABLE_AREAS) >= 4


def _sample_rows(height: int, count: int) -> list[int] | None:
    candidates = list(range(height - 2))
    random.shuffle(candidates)
    rows: list[int] = []
    for row in candidates:
        if all(abs(int(row) - int(prev)) >= 4 for prev in rows):
            rows.append(int(row))
            if len(rows) == count:
                return sorted(rows)
    return None


def _sample_slots(height: int, width: int, count: int) -> list[tuple[int, int]] | None:
    candidates = [(r, c) for r in range(height - 2) for c in range(width - 2)]
    random.shuffle(candidates)
    slots: list[tuple[int, int]] = []
    for row, col in candidates:
        if all(abs(int(row) - int(pr)) > 3 or abs(int(col) - int(pc)) > 3 for pr, pc in slots):
            slots.append((int(row), int(col)))
            if len(slots) == count:
                return slots
    return None


def _choose_object_mask(area: int, template_mask: np.ndarray) -> np.ndarray:
    masks = _SHAPE_BANK[area]
    alternatives = [mask for mask in masks if not np.array_equal(mask, template_mask)]
    pool = alternatives if alternatives else masks
    return random.choice(pool).copy()


def _gen_attempt() -> np.ndarray | None:
    n_templates = random.randint(2, 4)
    template_areas = random.sample(_USABLE_AREAS, n_templates)
    height_min = max(11, 4 * n_templates - 1)
    grid_h = random.randint(height_min, 18)
    core_w = random.randint(11, 18)
    n_objects = random.randint(n_templates, min(n_templates + 3, 6))

    legend_rows = _sample_rows(grid_h, n_templates)
    object_slots = _sample_slots(grid_h, core_w, n_objects)
    if legend_rows is None or object_slots is None:
        return None

    bg = random.choice([color for color in range(10) if color != GRAY_COLOR])
    legend_colors = random.sample(
        [color for color in range(10) if color not in {bg, GRAY_COLOR}],
        n_templates,
    )
    side = random.choice(("left", "right"))

    grid = np.full((grid_h, core_w + LEGEND_STRIP_WIDTH), int(bg), dtype=int)
    template_masks: dict[int, np.ndarray] = {}
    template_colors: dict[int, int] = {}

    legend_c0 = 0 if side == "left" else core_w
    for area, color, row in zip(template_areas, legend_colors, legend_rows):
        mask = random.choice(_SHAPE_BANK[area]).copy()
        template_masks[area] = mask
        template_colors[area] = int(color)
        patch = grid[row : row + 3, legend_c0 : legend_c0 + 3]
        patch[mask] = int(color)

    object_areas = list(template_areas)
    while len(object_areas) < n_objects:
        object_areas.append(random.choice(template_areas))
    random.shuffle(object_areas)

    expected = np.full((grid_h, core_w), int(bg), dtype=int)
    core_c0 = LEGEND_STRIP_WIDTH if side == "left" else 0
    for (row, col), area in zip(object_slots, object_areas):
        obj_mask = _choose_object_mask(area, template_masks[area])
        in_patch = grid[row : row + 3, core_c0 + col : core_c0 + col + 3]
        if np.any(in_patch != int(bg)):
            return None
        in_patch[obj_mask] = GRAY_COLOR

        out_patch = expected[row : row + 3, col : col + 3]
        out_patch[template_masks[area]] = int(template_colors[area])

    if np.count_nonzero(grid == GRAY_COLOR) <= 0:
        return None
    if _get_bg(grid) != bg:
        return None

    try:
        produced = main(grid)
    except AssertionError:
        return None
    if not np.array_equal(produced, expected):
        return None
    return grid


def generate_input() -> np.ndarray:
    for _ in range(MAX_GENERATION_ATTEMPTS):
        grid = _gen_attempt()
        if grid is not None:
            return grid
    raise RuntimeError("failed to generate a valid sample")
