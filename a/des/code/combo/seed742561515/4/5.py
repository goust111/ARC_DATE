from __future__ import annotations

import numpy as np


BLACK = 0
BLUE = 1
RED = 2
GREEN = 3
YELLOW = 4
GREY = 5
PINK = 6
ORANGE = 7
CYAN = 8
MAROON = 9

DIR4 = ((1, 0), (-1, 0), (0, 1), (0, -1))
TRANSFORM_NAMES = (
    "id",
    "rot90",
    "rot180",
    "rot270",
    "flip_h",
    "flip_v",
    "diag_main",
    "diag_anti",
)
TRANSFORM_BLOCKS = (
    ("id", "rot90", "rot180"),
    ("rot270", "flip_h", "flip_v"),
    ("diag_main", "diag_anti"),
)
BORDER_COLOR_BY_TRANSFORM = {
    "id": RED,
    "rot90": BLUE,
    "rot180": BLUE,
    "rot270": BLUE,
    "flip_h": GREEN,
    "flip_v": GREEN,
    "diag_main": ORANGE,
    "diag_anti": ORANGE,
}

TOP_COLOR_POOL = (BLUE, RED, GREEN, YELLOW, PINK, ORANGE, CYAN)
MIN_OBJECTS = 2
MAX_OBJECTS = 4
MIN_AREA = 4
MAX_AREA = 8
MAX_SHAPE_DIM = 4
MAX_SHAPE_ATTEMPTS = 160
MAX_GENERATION_ATTEMPTS = 400

TOP_SIDE_PAD = 1
TOP_GAP = 1
BOTTOM_SIDE_PAD = 1
BOTTOM_GAP = 2
MIN_WIDTH = 12
MAX_WIDTH = 30
MAX_HEIGHT = 30

_RNG = np.random.default_rng()
_SUCCESSFUL_GENERATIONS = 0


def _mask_key(mask: np.ndarray) -> tuple[int, int, bytes]:
    m = np.asarray(mask, dtype=bool)
    return int(m.shape[0]), int(m.shape[1]), m.astype(np.uint8).tobytes()


def _apply_transform(mask: np.ndarray, name: str) -> np.ndarray:
    if name == "id":
        return mask.copy()
    if name == "rot90":
        return np.rot90(mask, -1).copy()
    if name == "rot180":
        return np.rot90(mask, 2).copy()
    if name == "rot270":
        return np.rot90(mask, 1).copy()
    if name == "flip_h":
        return np.flipud(mask).copy()
    if name == "flip_v":
        return np.fliplr(mask).copy()
    if name == "diag_main":
        return mask.T.copy()
    if name == "diag_anti":
        return np.rot90(mask.T, 2).copy()
    raise AssertionError(f"unknown transform {name}")


def _transform_variants(mask: np.ndarray) -> dict[str, np.ndarray]:
    return {name: _apply_transform(mask, name) for name in TRANSFORM_NAMES}


def _canonical_key(mask: np.ndarray) -> tuple[int, int, bytes]:
    return min(_mask_key(variant) for variant in _transform_variants(mask).values())


def _is_fully_asymmetric(mask: np.ndarray) -> bool:
    return len({_mask_key(variant) for variant in _transform_variants(mask).values()}) == 8


def _cells_to_mask(cells: set[tuple[int, int]]) -> np.ndarray:
    rows = [r for r, _c in cells]
    cols = [c for _r, c in cells]
    r0, r1 = min(rows), max(rows)
    c0, c1 = min(cols), max(cols)
    mask = np.zeros((r1 - r0 + 1, c1 - c0 + 1), dtype=bool)
    for r, c in cells:
        mask[r - r0, c - c0] = True
    return mask


def _find_separator_row(grid: np.ndarray) -> int:
    rows = [r for r in range(grid.shape[0]) if np.all(grid[r, :] == MAROON)]
    assert len(rows) == 1, "expected exactly one maroon separator row"
    sep = rows[0]
    assert 0 < sep < grid.shape[0] - 1
    return int(sep)


def _find_components(mask: np.ndarray) -> list[np.ndarray]:
    h, w = mask.shape
    seen = np.zeros((h, w), dtype=bool)
    components: list[np.ndarray] = []

    for r in range(h):
        for c in range(w):
            if not mask[r, c] or seen[r, c]:
                continue
            stack = [(r, c)]
            seen[r, c] = True
            coords: list[tuple[int, int]] = []
            while stack:
                rr, cc = stack.pop()
                coords.append((rr, cc))
                for dr, dc in DIR4:
                    nr = rr + dr
                    nc = cc + dc
                    if 0 <= nr < h and 0 <= nc < w and mask[nr, nc] and not seen[nr, nc]:
                        seen[nr, nc] = True
                        stack.append((nr, nc))
            components.append(np.array(coords, dtype=int))

    return components


def _extract_object(panel: np.ndarray, coords: np.ndarray, row_offset: int) -> dict:
    r0 = int(coords[:, 0].min())
    c0 = int(coords[:, 1].min())
    r1 = int(coords[:, 0].max())
    c1 = int(coords[:, 1].max())
    rel = coords - np.array([r0, c0], dtype=int)
    mask = np.zeros((r1 - r0 + 1, c1 - c0 + 1), dtype=bool)
    mask[rel[:, 0], rel[:, 1]] = True
    colors = {int(panel[r, c]) for r, c in coords}
    assert len(colors) == 1
    color = colors.pop()
    abs_coords = coords.copy()
    abs_coords[:, 0] += int(row_offset)
    return {
        "color": int(color),
        "mask": mask,
        "bbox": (r0 + row_offset, c0, r1 + row_offset, c1),
        "coords": abs_coords,
        "shape_key": _canonical_key(mask),
    }


def _extract_objects(panel: np.ndarray, row_offset: int, color: int | None = None) -> list[dict]:
    if color is None:
        object_mask = (panel != BLACK) & (panel != MAROON)
    else:
        object_mask = panel == int(color)
    objects = [_extract_object(panel, coords, row_offset) for coords in _find_components(object_mask)]
    assert objects
    return objects


def _match_objects(top_objects: list[dict], bottom_objects: list[dict]) -> list[tuple[int, str]]:
    candidates: list[list[tuple[int, str]]] = []
    for bottom in bottom_objects:
        matches: list[tuple[int, str]] = []
        for top_index, top in enumerate(top_objects):
            for name, variant in _transform_variants(top["mask"]).items():
                if variant.shape == bottom["mask"].shape and np.array_equal(variant, bottom["mask"]):
                    matches.append((top_index, name))
        assert matches, "each shadow must match some template"
        candidates.append(matches)

    order = sorted(range(len(bottom_objects)), key=lambda idx: len(candidates[idx]))
    assignment: list[tuple[int, str] | None] = [None] * len(bottom_objects)
    used: set[int] = set()

    def _search(pos: int) -> bool:
        if pos == len(order):
            return True
        bottom_index = order[pos]
        for top_index, name in sorted(
            candidates[bottom_index],
            key=lambda item: (item[0], TRANSFORM_NAMES.index(item[1])),
        ):
            if top_index in used:
                continue
            used.add(top_index)
            assignment[bottom_index] = (top_index, name)
            if _search(pos + 1):
                return True
            assignment[bottom_index] = None
            used.remove(top_index)
        return False

    assert _search(0), "failed to find a one-to-one template/shadow matching"
    resolved = [item for item in assignment if item is not None]
    assert len(resolved) == len(top_objects) == len(bottom_objects)
    assert len({top_index for top_index, _name in resolved}) == len(top_objects)
    return resolved


def _draw_border(out: np.ndarray, bbox: tuple[int, int, int, int], color: int, sep: int) -> None:
    r0, c0, r1, c1 = bbox
    br0, bc0, br1, bc1 = r0 - 1, c0 - 1, r1 + 1, c1 + 1
    assert sep < br0 <= br1 < out.shape[0]
    assert 0 <= bc0 <= bc1 < out.shape[1]

    ring = np.zeros_like(out, dtype=bool)
    ring[br0, bc0 : bc1 + 1] = True
    ring[br1, bc0 : bc1 + 1] = True
    ring[br0 : br1 + 1, bc0] = True
    ring[br0 : br1 + 1, bc1] = True
    assert np.all(out[ring] == BLACK), "border must be drawn on empty background"
    out[ring] = int(color)


def main(grid: np.ndarray) -> np.ndarray:
    """
    Recolor each lower grey shadow with its matched upper template color and
    add a border whose color encodes the shadow's dihedral transform class.
    """
    g = np.asarray(grid, dtype=int)
    assert g.ndim == 2 and g.size > 0

    sep = _find_separator_row(g)
    top_panel = g[:sep, :]
    bottom_panel = g[sep + 1 :, :]
    assert top_panel.size > 0 and bottom_panel.size > 0

    top_objects = _extract_objects(top_panel, row_offset=0)
    bottom_objects = _extract_objects(bottom_panel, row_offset=sep + 1, color=GREY)
    assert MIN_OBJECTS <= len(top_objects) <= MAX_OBJECTS
    assert len(top_objects) == len(bottom_objects)
    assert len({obj["color"] for obj in top_objects}) == len(top_objects)
    assert len({obj["shape_key"] for obj in top_objects}) == len(top_objects), "top shapes must be unique up to symmetry"

    matches = _match_objects(top_objects, bottom_objects)

    out = g.copy()
    for bottom_index, (top_index, transform_name) in enumerate(matches):
        bottom = bottom_objects[bottom_index]
        top = top_objects[top_index]
        out[bottom["coords"][:, 0], bottom["coords"][:, 1]] = int(top["color"])
        _draw_border(out, bottom["bbox"], BORDER_COLOR_BY_TRANSFORM[transform_name], sep)

    return out


def _random_mask(area: int) -> np.ndarray | None:
    for _ in range(MAX_SHAPE_ATTEMPTS):
        cells = {(3, 3)}
        while len(cells) < area:
            frontier: set[tuple[int, int]] = set()
            for r, c in cells:
                for dr, dc in DIR4:
                    cand = (r + dr, c + dc)
                    if cand in cells:
                        continue
                    trial = set(cells)
                    trial.add(cand)
                    rows = [rr for rr, _cc in trial]
                    cols = [cc for _rr, cc in trial]
                    if max(rows) - min(rows) + 1 <= MAX_SHAPE_DIM and max(cols) - min(cols) + 1 <= MAX_SHAPE_DIM:
                        frontier.add(cand)
            if not frontier:
                break
            frontier_list = list(frontier)
            pick = frontier_list[int(_RNG.integers(0, len(frontier_list)))]
            cells.add(pick)

        if len(cells) != area:
            continue
        mask = _cells_to_mask(cells)
        if mask.shape[0] <= 1 or mask.shape[1] <= 1:
            continue
        if not _is_fully_asymmetric(mask):
            continue
        return mask
    return None


def _sample_base_masks(count: int) -> list[np.ndarray] | None:
    masks: list[np.ndarray] = []
    seen: set[tuple[int, int, bytes]] = set()
    while len(masks) < count and len(seen) < 200:
        area = int(_RNG.integers(MIN_AREA, MAX_AREA + 1))
        mask = _random_mask(area)
        if mask is None:
            continue
        key = _canonical_key(mask)
        if key in seen:
            continue
        seen.add(key)
        masks.append(mask)
    if len(masks) != count:
        return None
    return masks


def _assign_positions(widths: list[int], total_width: int, gap_min: int, side_pad: int) -> list[int] | None:
    need = sum(widths) + gap_min * (len(widths) - 1) + 2 * side_pad
    if need > total_width:
        return None
    slack = total_width - need
    extras = _RNG.multinomial(slack, np.ones(len(widths) + 1) / (len(widths) + 1))
    cols: list[int] = []
    col = side_pad + int(extras[0])
    for idx, width in enumerate(widths):
        cols.append(int(col))
        col += int(width)
        if idx < len(widths) - 1:
            col += gap_min + int(extras[idx + 1])
    return cols


def _sample_rows(heights: list[int], panel_height: int, side_pad: int) -> list[int] | None:
    rows: list[int] = []
    for height in heights:
        low = side_pad
        high = panel_height - side_pad - int(height)
        if low > high:
            return None
        rows.append(int(_RNG.integers(low, high + 1)))
    return rows


def _paint_mask(grid: np.ndarray, mask: np.ndarray, r0: int, c0: int, color: int) -> None:
    h, w = mask.shape
    patch = grid[r0 : r0 + h, c0 : c0 + w]
    assert patch.shape == mask.shape
    assert np.all(patch[mask] == BLACK)
    patch[mask] = int(color)


def _sample_transform_plan(phase: int) -> list[str]:
    mandatory = list(TRANSFORM_BLOCKS[phase])
    count = int(_RNG.integers(len(mandatory), MAX_OBJECTS + 1))
    if count > len(mandatory):
        pool = [name for name in TRANSFORM_NAMES if name not in mandatory]
        picks = _RNG.choice(pool, size=count - len(mandatory), replace=False)
        mandatory.extend(str(name) for name in picks.tolist())
    _RNG.shuffle(mandatory)
    return mandatory


def _build_candidate_grid() -> np.ndarray | None:
    phase = _SUCCESSFUL_GENERATIONS % len(TRANSFORM_BLOCKS)
    transforms = _sample_transform_plan(phase)
    count = len(transforms)

    base_masks = _sample_base_masks(count)
    if base_masks is None:
        return None

    colors = _RNG.choice(TOP_COLOR_POOL, size=count, replace=False).astype(int).tolist()
    transformed_masks = [_apply_transform(base_masks[idx], transforms[idx]) for idx in range(count)]

    top_order = _RNG.permutation(count).tolist()
    bottom_order = _RNG.permutation(count).tolist()

    top_widths = [int(base_masks[idx].shape[1]) for idx in top_order]
    bottom_widths = [int(transformed_masks[idx].shape[1]) for idx in bottom_order]
    min_width = max(
        MIN_WIDTH,
        sum(top_widths) + TOP_GAP * (count - 1) + 2 * TOP_SIDE_PAD,
        sum(bottom_widths) + BOTTOM_GAP * (count - 1) + 2 * BOTTOM_SIDE_PAD,
    )
    if min_width > MAX_WIDTH:
        return None
    width = int(_RNG.integers(min_width, min(MAX_WIDTH, min_width + 6) + 1))

    top_height = max(max(int(mask.shape[0]) for mask in base_masks) + 2 * TOP_SIDE_PAD, 6)
    top_height += int(_RNG.integers(0, 5))
    bottom_height = max(max(int(mask.shape[0]) for mask in transformed_masks) + 2 * BOTTOM_SIDE_PAD, 6)
    bottom_height += int(_RNG.integers(0, 5))
    total_height = top_height + 1 + bottom_height
    if total_height > MAX_HEIGHT:
        return None

    top_cols = _assign_positions(top_widths, width, TOP_GAP, TOP_SIDE_PAD)
    bottom_cols = _assign_positions(bottom_widths, width, BOTTOM_GAP, BOTTOM_SIDE_PAD)
    if top_cols is None or bottom_cols is None:
        return None

    top_rows = _sample_rows([int(base_masks[idx].shape[0]) for idx in top_order], top_height, TOP_SIDE_PAD)
    bottom_rows = _sample_rows(
        [int(transformed_masks[idx].shape[0]) for idx in bottom_order],
        bottom_height,
        BOTTOM_SIDE_PAD,
    )
    if top_rows is None or bottom_rows is None:
        return None

    grid = np.full((total_height, width), BLACK, dtype=int)
    grid[top_height, :] = MAROON

    for pos, obj_index in enumerate(top_order):
        _paint_mask(grid, base_masks[obj_index], top_rows[pos], top_cols[pos], colors[obj_index])

    bottom_offset = top_height + 1
    for pos, obj_index in enumerate(bottom_order):
        _paint_mask(
            grid,
            transformed_masks[obj_index],
            bottom_offset + bottom_rows[pos],
            bottom_cols[pos],
            GREY,
        )

    return grid


def generate_input() -> np.ndarray:
    global _SUCCESSFUL_GENERATIONS

    for _ in range(MAX_GENERATION_ATTEMPTS):
        grid = _build_candidate_grid()
        if grid is None:
            continue
        try:
            _ = main(grid)
        except AssertionError:
            continue
        _SUCCESSFUL_GENERATIONS += 1
        return grid
    raise RuntimeError("failed to generate a valid input grid")
