from __future__ import annotations

import random

import numpy as np


BACKGROUND_COLORS = {1, 3}
OCCLUDER_COLORS = (2, 4, 6, 7)
FRAME_COLOR = 5
TEMPLATE_COLOR = 9

MATCH_DIMS = ((2, 3), (3, 2))
DISTRACTOR_DIMS = ((2, 4), (4, 2), (3, 3), (2, 5), (5, 2), (3, 4), (4, 3))
FRAME_CORNERS = ("tl", "tr", "bl", "br")
MAX_GENERATION_ATTEMPTS = 400


def _bbox_coords(coords: np.ndarray) -> tuple[int, int, int, int]:
    r0, c0 = coords.min(axis=0)
    r1, c1 = coords.max(axis=0)
    return int(r0), int(c0), int(r1), int(c1)


def _components_of_color(grid: np.ndarray, color: int) -> list[np.ndarray]:
    mask = grid == int(color)
    h, w = grid.shape
    seen = np.zeros((h, w), dtype=bool)
    comps: list[np.ndarray] = []
    for r in range(h):
        for c in range(w):
            if not mask[r, c] or seen[r, c]:
                continue
            stack = [(r, c)]
            seen[r, c] = True
            cells: list[tuple[int, int]] = []
            while stack:
                rr, cc = stack.pop()
                cells.append((rr, cc))
                for dr, dc in ((-1, 0), (1, 0), (0, -1), (0, 1)):
                    nr, nc = rr + dr, cc + dc
                    if 0 <= nr < h and 0 <= nc < w and mask[nr, nc] and not seen[nr, nc]:
                        seen[nr, nc] = True
                        stack.append((nr, nc))
            comps.append(np.asarray(cells, dtype=int))
    return comps


def _is_solid_rectangle(coords: np.ndarray) -> bool:
    r0, c0, r1, c1 = _bbox_coords(coords)
    return int(coords.shape[0]) == (r1 - r0 + 1) * (c1 - c0 + 1)


def _find_frame(grid: np.ndarray) -> tuple[int, int, int, int]:
    frames = []
    for coords in _components_of_color(grid, FRAME_COLOR):
        r0, c0, r1, c1 = _bbox_coords(coords)
        patch = grid[r0 : r1 + 1, c0 : c1 + 1]
        expected = np.zeros_like(patch, dtype=bool)
        expected[-1, :] = True
        expected[:, -1] = True
        if np.array_equal(patch == FRAME_COLOR, expected):
            frames.append((r0, c0, r1, c1))
    assert len(frames) == 1, "expected exactly one grey L-frame"
    return frames[0]


def _extract_template_dims(grid: np.ndarray, frame_bbox: tuple[int, int, int, int]) -> tuple[int, int]:
    r0, c0, r1, c1 = frame_bbox
    inner = grid[r0:r1, c0:c1]
    coords = np.argwhere(inner == TEMPLATE_COLOR)
    assert coords.size > 0, "template must exist inside the frame"
    bbox = _bbox_coords(coords)
    tr0, tc0, tr1, tc1 = bbox
    patch = inner[tr0 : tr1 + 1, tc0 : tc1 + 1]
    assert np.all(patch == TEMPLATE_COLOR), "template must be a solid rectangle"
    return int(tr1 - tr0 + 1), int(tc1 - tc0 + 1)


def _find_occluders(grid: np.ndarray) -> list[tuple[int, int, int, int, int]]:
    occluders: list[tuple[int, int, int, int, int]] = []
    for color in OCCLUDER_COLORS:
        for coords in _components_of_color(grid, color):
            assert _is_solid_rectangle(coords), "occluders must be solid rectangles"
            r0, c0, r1, c1 = _bbox_coords(coords)
            occluders.append((int(color), r0, c0, r1, c1))
    assert len(occluders) >= 3, "expected multiple rectangular occluders"
    return occluders


def _period_mapping(
    grid: np.ndarray, visible: np.ndarray, row_period: int, col_period: int
) -> dict[tuple[int, int], int] | None:
    values: dict[tuple[int, int], int] = {}
    h, w = grid.shape
    for r in range(h):
        for c in range(w):
            if not visible[r, c]:
                continue
            key = (r % row_period, c % col_period)
            value = int(grid[r, c])
            old = values.get(key)
            if old is not None and old != value:
                return None
            values[key] = value
    return values


def _infer_periodic_fill(grid: np.ndarray) -> tuple[int, int, dict[tuple[int, int], int]]:
    visible = np.isin(grid, list(BACKGROUND_COLORS))
    assert visible.any(), "need visible background pixels"
    candidates: list[tuple[int, int, dict[tuple[int, int], int]]] = []
    for row_period in (1, 2, 3):
        for col_period in (1, 2, 3):
            mapping = _period_mapping(grid, visible, row_period, col_period)
            if mapping is not None:
                candidates.append((row_period, col_period, mapping))
    assert candidates, "visible pixels must determine a periodic background"
    min_area = min(rp * cp for rp, cp, _ in candidates)
    best = [(rp, cp, m) for rp, cp, m in candidates if rp * cp == min_area]
    first_rp, first_cp, first_map = best[0]
    for rp, cp, mapping in best[1:]:
        assert rp == first_rp and cp == first_cp and mapping == first_map, "minimal periods must agree"
    return first_rp, first_cp, first_map


def main(grid: np.ndarray) -> np.ndarray:
    """
    Selectively restore only the rectangular occluder whose shape matches the template inside the grey L-frame.

    Input structure:
    - The whole grid is a blue/green periodic background built from a small 2x2 or 3x3 tile.
    - Three or four solid rectangular occluders of colors red/yellow/magenta/orange cover parts of that background.
    - A grey L-frame, with its vertical leg on the right and horizontal leg on the bottom of its bbox, encloses a maroon rectangular template in its upper-left interior.

    Rule:
    - Find the grey L-frame and read the maroon template rectangle's bbox height and width.
    - Among all solid colored occluder rectangles, identify the unique one with the same height and width.
    - Infer the smallest row/column periodic background from the still-visible blue/green pixels.
    - Repaint only the matched occluder's rectangle with the inferred periodic colors; keep every other occluder, the frame, and the template unchanged.

    Key insight:
    - The frame does not mark something to transform directly; it provides a shape selector that chooses which occluder should be healed, and only that region gets the periodic background restored.
    """
    g = np.asarray(grid, dtype=int)
    assert g.ndim == 2

    frame_bbox = _find_frame(g)
    template_dims = _extract_template_dims(g, frame_bbox)

    occluders = _find_occluders(g)
    matches = [rect for rect in occluders if (rect[3] - rect[1] + 1, rect[4] - rect[2] + 1) == template_dims]
    assert len(matches) == 1, "template must match exactly one occluder size"
    color, r0, c0, r1, c1 = matches[0]
    _ = color

    row_period, col_period, mapping = _infer_periodic_fill(g)
    out = g.copy()
    for r in range(r0, r1 + 1):
        for c in range(c0, c1 + 1):
            key = (r % row_period, c % col_period)
            assert key in mapping, "every target residue must be visible somewhere else"
            out[r, c] = mapping[key]
    return out


def _tile_grid(tile: np.ndarray, shape: tuple[int, int]) -> np.ndarray:
    h, w = shape
    th, tw = tile.shape
    rows = np.arange(h)[:, None] % th
    cols = np.arange(w)[None, :] % tw
    return tile[rows, cols].astype(int, copy=True)


def _minimal_period(tile: np.ndarray) -> tuple[int, int]:
    h, w = tile.shape
    for rp in range(1, h + 1):
        if np.array_equal(tile, tile[np.arange(h) % rp][:, np.arange(w)]):
            row_period = rp
            break
    else:
        row_period = h
    for cp in range(1, w + 1):
        if np.array_equal(tile, tile[:, np.arange(w) % cp]):
            col_period = cp
            break
    else:
        col_period = w
    return row_period, col_period


def _random_tile(size: int) -> np.ndarray:
    for _ in range(200):
        tile = np.random.choice([1, 3], size=(size, size))
        if len(set(map(int, tile.flat))) != 2:
            continue
        if _minimal_period(tile) != (size, size):
            continue
        return tile.astype(int)
    raise RuntimeError("failed to generate a non-degenerate periodic tile")


def _frame_bbox_for_corner(corner: str, grid_shape: tuple[int, int], frame_shape: tuple[int, int]) -> tuple[int, int, int, int]:
    h, w = grid_shape
    fh, fw = frame_shape
    if corner == "tl":
        r0, c0 = 0, 0
    elif corner == "tr":
        r0, c0 = 0, w - fw
    elif corner == "bl":
        r0, c0 = h - fh, 0
    else:
        r0, c0 = h - fh, w - fw
    return r0, c0, r0 + fh - 1, c0 + fw - 1


def _paint_frame_and_template(
    grid: np.ndarray,
    frame_bbox: tuple[int, int, int, int],
    template_dims: tuple[int, int],
) -> None:
    r0, c0, r1, c1 = frame_bbox
    grid[r1, c0 : c1 + 1] = FRAME_COLOR
    grid[r0 : r1 + 1, c1] = FRAME_COLOR
    inner_h = r1 - r0
    inner_w = c1 - c0
    th, tw = template_dims
    tr = r0 + random.randint(0, inner_h - th)
    tc = c0 + random.randint(0, inner_w - tw)
    grid[tr : tr + th, tc : tc + tw] = TEMPLATE_COLOR


def _can_place(
    occupied: np.ndarray, top: int, left: int, height: int, width: int, padding: int
) -> bool:
    h, w = occupied.shape
    if top < 0 or left < 0 or top + height > h or left + width > w:
        return False
    r0 = max(0, top - padding)
    c0 = max(0, left - padding)
    r1 = min(h, top + height + padding)
    c1 = min(w, left + width + padding)
    return not bool(occupied[r0:r1, c0:c1].any())


def _place_rectangles(
    grid_shape: tuple[int, int],
    reserved_bbox: tuple[int, int, int, int],
    dims: list[tuple[int, int]],
) -> list[tuple[int, int, int, int]] | None:
    h, w = grid_shape
    occupied = np.zeros((h, w), dtype=bool)
    rr0, rc0, rr1, rc1 = reserved_bbox
    occupied[rr0 : rr1 + 1, rc0 : rc1 + 1] = True
    placed: list[tuple[int, int, int, int]] = []
    for rect_h, rect_w in dims:
        candidates = []
        for r0 in range(h - rect_h + 1):
            for c0 in range(w - rect_w + 1):
                if _can_place(occupied, r0, c0, rect_h, rect_w, padding=1):
                    candidates.append((r0, c0))
        if not candidates:
            return None
        r0, c0 = random.choice(candidates)
        occupied[r0 : r0 + rect_h, c0 : c0 + rect_w] = True
        placed.append((r0, c0, r0 + rect_h - 1, c0 + rect_w - 1))
    return placed


def _all_residues_visible(blocked: np.ndarray, period: int) -> bool:
    for rr in range(period):
        for cc in range(period):
            if not np.any(~blocked[rr::period, cc::period]):
                return False
    return True


def _expected_output(
    base: np.ndarray, inp: np.ndarray, target_bbox: tuple[int, int, int, int]
) -> np.ndarray:
    out = inp.copy()
    r0, c0, r1, c1 = target_bbox
    out[r0 : r1 + 1, c0 : c1 + 1] = base[r0 : r1 + 1, c0 : c1 + 1]
    return out


def _attempt_generate() -> np.ndarray | None:
    height = random.randint(17, 22)
    width = random.randint(17, 22)
    tile_size = random.choice((2, 3))
    tile = _random_tile(tile_size)
    base = _tile_grid(tile, (height, width))

    template_dims = random.choice(MATCH_DIMS)
    occluder_count = random.randint(3, 4)
    dims = [template_dims, (template_dims[1], template_dims[0])]
    extra_pool = [d for d in DISTRACTOR_DIMS if d not in dims]
    dims.extend(random.sample(extra_pool, occluder_count - len(dims)))
    random.shuffle(dims)
    target_index = dims.index(template_dims)

    inner_h = template_dims[0] + random.randint(1, 3)
    inner_w = template_dims[1] + random.randint(1, 3)
    frame_shape = (inner_h + 1, inner_w + 1)
    corner = random.choice(FRAME_CORNERS)
    frame_bbox = _frame_bbox_for_corner(corner, (height, width), frame_shape)

    rects = _place_rectangles((height, width), frame_bbox, dims)
    if rects is None:
        return None

    inp = base.copy()
    occluder_colors = random.sample(list(OCCLUDER_COLORS), occluder_count)
    for color, (r0, c0, r1, c1) in zip(occluder_colors, rects):
        inp[r0 : r1 + 1, c0 : c1 + 1] = int(color)
    _paint_frame_and_template(inp, frame_bbox, template_dims)

    blocked = ~np.isin(inp, list(BACKGROUND_COLORS))
    if not _all_residues_visible(blocked, tile_size):
        return None

    target_bbox = rects[target_index]
    expected = _expected_output(base, inp, target_bbox)

    other_restored = inp.copy()
    for idx, bbox in enumerate(rects):
        if idx == target_index:
            continue
        r0, c0, r1, c1 = bbox
        other_restored[r0 : r1 + 1, c0 : c1 + 1] = base[r0 : r1 + 1, c0 : c1 + 1]
    if np.array_equal(other_restored, expected):
        return None

    try:
        actual = main(inp)
    except Exception:
        return None
    if not np.array_equal(actual, expected):
        return None

    recovered_mask = actual != inp
    if recovered_mask.sum() != template_dims[0] * template_dims[1]:
        return None
    return inp


def generate_input() -> np.ndarray:
    for _ in range(MAX_GENERATION_ATTEMPTS):
        grid = _attempt_generate()
        if grid is not None:
            return grid
    raise RuntimeError("failed to generate a valid puzzle instance")
