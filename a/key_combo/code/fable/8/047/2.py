from __future__ import annotations

import random

import numpy as np


BG = 0
SPRITE_SIZE = 3
HEADER_COLOR_POOL = list(range(1, 10))
ROW_COUNT_CHOICES = [2, 3, 4]
COL_COUNT_CHOICES = [2, 3, 4]
GRID_MIN = 20
GRID_MAX = 28
SPRITE_GAP = 1
MAX_ATTEMPTS = 400
NEIGHBORS4 = [(-1, 0), (1, 0), (0, -1), (0, 1)]
NEIGHBORS8 = [
    (-1, -1),
    (-1, 0),
    (-1, 1),
    (0, -1),
    (0, 1),
    (1, -1),
    (1, 0),
    (1, 1),
]


def _get_bg(grid: np.ndarray) -> int:
    values, counts = np.unique(grid, return_counts=True)
    return int(values[np.argmax(counts)])


def _bbox(coords: np.ndarray) -> tuple[int, int, int, int]:
    r0, c0 = coords.min(axis=0)
    r1, c1 = coords.max(axis=0)
    return int(r0), int(c0), int(r1), int(c1)


def _components(mask: np.ndarray, connectivity: int = 4) -> list[np.ndarray]:
    h, w = mask.shape
    seen = np.zeros((h, w), dtype=bool)
    neighbors = NEIGHBORS4 if int(connectivity) == 4 else NEIGHBORS8
    comps: list[np.ndarray] = []

    for r in range(h):
        for c in range(w):
            if not mask[r, c] or seen[r, c]:
                continue
            stack = [(r, c)]
            seen[r, c] = True
            coords = []
            while stack:
                rr, cc = stack.pop()
                coords.append((rr, cc))
                for dr, dc in neighbors:
                    nr, nc = rr + dr, cc + dc
                    if 0 <= nr < h and 0 <= nc < w and mask[nr, nc] and not seen[nr, nc]:
                        seen[nr, nc] = True
                        stack.append((nr, nc))
            comps.append(np.array(coords, dtype=int))
    return comps


def _sprite_mask_from_coords(coords: np.ndarray) -> np.ndarray:
    r0, c0, r1, c1 = _bbox(coords)
    assert r1 - r0 + 1 == SPRITE_SIZE and c1 - c0 + 1 == SPRITE_SIZE
    mask = np.zeros((SPRITE_SIZE, SPRITE_SIZE), dtype=bool)
    mask[coords[:, 0] - r0, coords[:, 1] - c0] = True
    return mask


def _stamp_box(center_r: int, center_c: int) -> tuple[int, int, int, int]:
    return int(center_r - 1), int(center_c - 1), int(center_r + 1), int(center_c + 1)


def _boxes_have_gap(a: tuple[int, int, int, int], b: tuple[int, int, int, int], gap: int = SPRITE_GAP) -> bool:
    ar0, ac0, ar1, ac1 = a
    br0, bc0, br1, bc1 = b
    g = int(gap)
    return ar1 + g < br0 or br1 + g < ar0 or ac1 + g < bc0 or bc1 + g < ac0


def _enumerate_sprite_masks() -> list[np.ndarray]:
    masks = []
    for bits in range(1, 1 << (SPRITE_SIZE * SPRITE_SIZE)):
        mask = np.array([(bits >> i) & 1 for i in range(9)], dtype=bool).reshape(SPRITE_SIZE, SPRITE_SIZE)
        area = int(mask.sum())
        if area < 4 or area > 7:
            continue
        if not mask[1, 1]:
            continue
        if not (mask[0, :].any() and mask[2, :].any() and mask[:, 0].any() and mask[:, 2].any()):
            continue
        if len(_components(mask, connectivity=8)) != 1:
            continue
        masks.append(mask)
    return masks


ALL_SPRITE_MASKS = _enumerate_sprite_masks()
MASKS_BY_AREA: dict[int, list[np.ndarray]] = {}
for _mask in ALL_SPRITE_MASKS:
    MASKS_BY_AREA.setdefault(int(_mask.sum()), []).append(_mask)


def _parse_scene(
    grid: np.ndarray,
) -> tuple[int, list[tuple[int, int]], list[tuple[int, int]], dict[int, np.ndarray], dict[int, np.ndarray]]:
    g = np.asarray(grid, dtype=int)
    assert g.ndim == 2 and g.size > 0
    h, w = g.shape
    bg = _get_bg(g)

    assert g[0, 0] == bg
    row_headers = [(int(r), int(g[r, 0])) for r in range(1, h) if int(g[r, 0]) != bg]
    col_headers = [(int(c), int(g[0, c])) for c in range(1, w) if int(g[0, c]) != bg]
    assert row_headers and col_headers

    header_lookup: dict[int, tuple[int, int]] = {}
    for r, color in row_headers:
        assert color not in header_lookup
        header_lookup[color] = (int(r), 0)
    for c, color in col_headers:
        assert color not in header_lookup
        header_lookup[color] = (0, int(c))

    non_bg_colors = {int(color) for color in np.unique(g) if int(color) != bg}
    assert non_bg_colors == set(header_lookup), "every non-background color must be a unique header color"

    selected_rows = {r for r, _color in row_headers}
    selected_cols = {c for c, _color in col_headers}
    sprite_masks: dict[int, np.ndarray] = {}
    sprite_coords: dict[int, np.ndarray] = {}

    for color, header_coord in header_lookup.items():
        comps = _components(g == color, connectivity=8)
        assert len(comps) == 2, "each header color must appear as one edge marker and one sprite"

        header_comp = None
        sprite_comp = None
        for comp in comps:
            if comp.shape[0] == 1 and tuple(map(int, comp[0])) == header_coord:
                header_comp = comp
            else:
                assert sprite_comp is None
                sprite_comp = comp

        assert header_comp is not None and sprite_comp is not None
        r0, c0, r1, c1 = _bbox(sprite_comp)
        assert r0 >= 2 and c0 >= 2
        assert r1 < h and c1 < w
        assert set(range(r0, r1 + 1)).isdisjoint(selected_rows)
        assert set(range(c0, c1 + 1)).isdisjoint(selected_cols)
        mask = _sprite_mask_from_coords(sprite_comp)
        sprite_masks[color] = mask
        sprite_coords[color] = sprite_comp

    mask_keys = [mask.tobytes() for mask in sprite_masks.values()]
    assert len(mask_keys) == len(set(mask_keys)), "dictionary shapes must all be distinct"
    return bg, row_headers, col_headers, sprite_masks, sprite_coords


def main(grid: np.ndarray) -> np.ndarray:
    """Stamp each row/column intersection with the row-selected sprite shape recolored to the column-selected color.

    Input structure:
    - The background is black, with row headers as isolated colored pixels on the
      left edge and column headers as isolated colored pixels on the top edge.
    - Every header color also appears exactly once as a detached same-color sprite
      somewhere inside the grid; each sprite has a 3x3 bounding box, and all such
      shapes are distinct.
    - The detached sprites stay away from the selected rows and columns, and the
      chosen row/column intersections are spaced so their 3x3 stamps do not overlap.

    Rule:
    - Each left-edge header selects its row, and each top-edge header selects its column.
    - For every selected row/column intersection, take the sprite shape belonging
      to the row-header color, recolor that whole shape to the column-header color,
      and stamp it centered on the intersection.
    - Keep all header pixels, erase all detached dictionary sprites, and leave the
      remaining background unchanged.

    Key insight:
    - The detached sprites form a shape dictionary indexed by color: row headers
      choose which shape to use, while column headers choose the output color.
    """
    g = np.asarray(grid, dtype=int)
    bg, row_headers, col_headers, sprite_masks, sprite_coords = _parse_scene(g)
    out = g.copy()

    for coords in sprite_coords.values():
        out[coords[:, 0], coords[:, 1]] = bg

    h, w = g.shape
    for r, row_color in row_headers:
        mask = sprite_masks[row_color]
        for c, col_color in col_headers:
            r0, c0, r1, c1 = _stamp_box(r, c)
            assert 0 <= r0 and r1 < h and 0 <= c0 and c1 < w
            out[r0 : r1 + 1, c0 : c1 + 1][mask] = col_color

    return out


def _sample_axis_positions(size: int, count: int) -> list[int] | None:
    candidates = list(range(2, size - 1))
    for _ in range(200):
        chosen = sorted(random.sample(candidates, count))
        if all(b - a >= SPRITE_SIZE for a, b in zip(chosen, chosen[1:])):
            return chosen
    return None


def _sample_sprite_masks(total_count: int, row_count: int) -> list[np.ndarray] | None:
    shared_areas = [area for area, masks in MASKS_BY_AREA.items() if len(masks) >= 2]
    for _ in range(200):
        chosen: list[np.ndarray] = []
        same_area = random.choice(shared_areas)
        chosen.extend(random.sample(MASKS_BY_AREA[same_area], 2))
        used = {mask.tobytes() for mask in chosen}

        pool = ALL_SPRITE_MASKS[:]
        random.shuffle(pool)
        for mask in pool:
            key = mask.tobytes()
            if key in used:
                continue
            chosen.append(mask)
            used.add(key)
            if len(chosen) == total_count:
                break

        if len(chosen) != total_count:
            continue
        row_areas = [int(mask.sum()) for mask in chosen[:row_count]]
        if len(row_areas) == len(set(row_areas)):
            continue
        return [mask.copy() for mask in chosen]
    return None


def _place_sprites(
    h: int,
    w: int,
    count: int,
    selected_rows: set[int],
    selected_cols: set[int],
) -> list[tuple[int, int]] | None:
    candidates = []
    for r in range(2, h - 2):
        row_band = {r, r + 1, r + 2}
        if not row_band.isdisjoint(selected_rows):
            continue
        for c in range(2, w - 2):
            col_band = {c, c + 1, c + 2}
            if not col_band.isdisjoint(selected_cols):
                continue
            candidates.append((int(r), int(c)))

    if len(candidates) < count:
        return None

    for _ in range(50):
        order = candidates[:]
        random.shuffle(order)
        chosen: list[tuple[int, int]] = []
        chosen_boxes: list[tuple[int, int, int, int]] = []
        for r, c in order:
            box = (r, c, r + 2, c + 2)
            if all(_boxes_have_gap(box, other, gap=SPRITE_GAP) for other in chosen_boxes):
                chosen.append((r, c))
                chosen_boxes.append(box)
                if len(chosen) == count:
                    return chosen
    return None


def _render_sprite(mask: np.ndarray, color: int) -> np.ndarray:
    sprite = np.zeros((SPRITE_SIZE, SPRITE_SIZE), dtype=int)
    sprite[mask] = int(color)
    return sprite


def _render_input(
    h: int,
    w: int,
    row_headers: list[tuple[int, int]],
    col_headers: list[tuple[int, int]],
    masks_by_color: dict[int, np.ndarray],
    sprite_positions: dict[int, tuple[int, int]],
) -> np.ndarray:
    grid = np.full((h, w), BG, dtype=int)
    for r, color in row_headers:
        grid[r, 0] = color
    for c, color in col_headers:
        grid[0, c] = color
    for color, (r0, c0) in sprite_positions.items():
        sprite = _render_sprite(masks_by_color[color], color)
        grid[r0 : r0 + SPRITE_SIZE, c0 : c0 + SPRITE_SIZE][sprite != BG] = sprite[sprite != BG]
    return grid


def _expected_output(
    inp: np.ndarray,
    row_headers: list[tuple[int, int]],
    col_headers: list[tuple[int, int]],
    masks_by_color: dict[int, np.ndarray],
    sprite_positions: dict[int, tuple[int, int]],
) -> np.ndarray:
    out = inp.copy()
    for color, (r0, c0) in sprite_positions.items():
        out[r0 : r0 + SPRITE_SIZE, c0 : c0 + SPRITE_SIZE][masks_by_color[color]] = BG
    for r, row_color in row_headers:
        mask = masks_by_color[row_color]
        for c, col_color in col_headers:
            out[r - 1 : r + 2, c - 1 : c + 2][mask] = col_color
    return out


def _validate_semantics(
    inp: np.ndarray,
    out: np.ndarray,
    row_headers: list[tuple[int, int]],
    col_headers: list[tuple[int, int]],
    masks_by_color: dict[int, np.ndarray],
    sprite_positions: dict[int, tuple[int, int]],
) -> bool:
    if len(row_headers) < 2 or len(col_headers) < 2:
        return False
    if inp.shape != out.shape:
        return False

    row_positions = [r for r, _color in row_headers]
    col_positions = [c for c, _color in col_headers]
    if any(b - a < SPRITE_SIZE for a, b in zip(sorted(row_positions), sorted(row_positions)[1:])):
        return False
    if any(b - a < SPRITE_SIZE for a, b in zip(sorted(col_positions), sorted(col_positions)[1:])):
        return False

    row_shapes = [masks_by_color[color] for _r, color in row_headers]
    row_areas = [int(mask.sum()) for mask in row_shapes]
    if len(row_areas) == len(set(row_areas)):
        return False

    shape_keys = [mask.tobytes() for mask in masks_by_color.values()]
    if len(shape_keys) != len(set(shape_keys)):
        return False

    stamp_boxes = [_stamp_box(r, c) for r, _row_color in row_headers for c, _col_color in col_headers]
    for i, box_a in enumerate(stamp_boxes):
        ar0, ac0, _ar1, _ac1 = box_a
        if ar0 <= 0 or ac0 <= 0:
            return False
        for box_b in stamp_boxes[i + 1 :]:
            if not _boxes_have_gap(box_a, box_b, gap=0):
                return False

    expected = _expected_output(inp, row_headers, col_headers, masks_by_color, sprite_positions)
    if not np.array_equal(out, expected):
        return False

    probe_row = row_headers[0][0]
    first_color = col_headers[0][1]
    second_color = col_headers[1][1]
    probe_mask = masks_by_color[row_headers[0][1]]
    patch_a = out[probe_row - 1 : probe_row + 2, col_headers[0][0] - 1 : col_headers[0][0] + 2]
    patch_b = out[probe_row - 1 : probe_row + 2, col_headers[1][0] - 1 : col_headers[1][0] + 2]
    if not (np.array_equal(patch_a == first_color, probe_mask) and np.array_equal(patch_b == second_color, probe_mask)):
        return False

    probe_col = col_headers[0][0]
    first_shape = masks_by_color[row_headers[0][1]]
    second_shape = masks_by_color[row_headers[1][1]]
    patch_c = out[row_headers[0][0] - 1 : row_headers[0][0] + 2, probe_col - 1 : probe_col + 2]
    patch_d = out[row_headers[1][0] - 1 : row_headers[1][0] + 2, probe_col - 1 : probe_col + 2]
    if not (np.array_equal(patch_c == first_color, first_shape) and np.array_equal(patch_d == first_color, second_shape)):
        return False

    return True


def _gen_attempt() -> np.ndarray | None:
    row_count = random.choice(ROW_COUNT_CHOICES)
    col_count = random.choice(COL_COUNT_CHOICES)
    total_colors = row_count + col_count
    h = random.randint(max(GRID_MIN, 3 * row_count + 8), GRID_MAX)
    w = random.randint(max(GRID_MIN, 3 * col_count + 8), GRID_MAX)

    row_positions = _sample_axis_positions(h, row_count)
    col_positions = _sample_axis_positions(w, col_count)
    if row_positions is None or col_positions is None:
        return None

    colors = random.sample(HEADER_COLOR_POOL, total_colors)
    row_colors = colors[:row_count]
    col_colors = colors[row_count:]
    masks = _sample_sprite_masks(total_colors, row_count)
    if masks is None:
        return None

    row_headers = list(zip(row_positions, row_colors))
    col_headers = list(zip(col_positions, col_colors))
    masks_by_color = {color: mask for color, mask in zip(colors, masks)}
    selected_rows = set(row_positions)
    selected_cols = set(col_positions)

    sprite_anchors = _place_sprites(h, w, total_colors, selected_rows, selected_cols)
    if sprite_anchors is None:
        return None
    sprite_positions = {color: anchor for color, anchor in zip(colors, sprite_anchors)}

    inp = _render_input(h, w, row_headers, col_headers, masks_by_color, sprite_positions)
    expected = _expected_output(inp, row_headers, col_headers, masks_by_color, sprite_positions)

    try:
        actual = main(inp)
    except Exception:
        return None
    if not np.array_equal(actual, expected):
        return None
    if not _validate_semantics(inp, actual, row_headers, col_headers, masks_by_color, sprite_positions):
        return None
    return inp


def generate_input() -> np.ndarray:
    for _ in range(MAX_ATTEMPTS):
        grid = _gen_attempt()
        if grid is not None:
            return grid
    raise RuntimeError("failed to generate a valid input for fable/8/047/2")
