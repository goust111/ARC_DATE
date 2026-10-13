from __future__ import annotations

import random
import sys
from pathlib import Path

import numpy as np

try:
    _ARC_UTILS_DIR = Path("/home/lixinyu/snap/new/solvers")
    if _ARC_UTILS_DIR.exists():
        sys.path.append(str(_ARC_UTILS_DIR))
    from arc_utils import bbox, colors_present
except Exception:
    def bbox(mask: np.ndarray):
        coords = np.argwhere(mask)
        if coords.size == 0:
            return None
        r0, c0 = coords.min(axis=0)
        r1, c1 = coords.max(axis=0)
        return int(r0), int(c0), int(r1), int(c1)

    def colors_present(grid: np.ndarray, *, exclude=None) -> list[int]:
        cols = set(map(int, np.unique(grid)))
        if exclude is not None:
            cols -= set(map(int, exclude))
        return sorted(cols)


SIDES = ("top", "right", "bottom", "left")
SIDE_TO_TURNS = {"top": 0, "right": 1, "bottom": 2, "left": 3}
PALETTE = (1, 2, 3, 4, 6, 7, 8, 9)
TEMPLATE_DIMS = ((3, 5), (5, 3), (4, 6), (6, 4))
GRID_H_RANGE = (14, 20)
GRID_W_RANGE = (14, 20)
WINDOW_MARGIN = 2
WINDOW_GAP = 1
MAX_ATTEMPTS = 400


def _center_clip(mask: np.ndarray, out_shape: tuple[int, int]) -> np.ndarray:
    out_h, out_w = map(int, out_shape)
    in_h, in_w = map(int, mask.shape)
    out = np.zeros((out_h, out_w), dtype=bool)
    r0 = (out_h - in_h) // 2
    c0 = (out_w - in_w) // 2

    dst_r0 = max(0, r0)
    dst_c0 = max(0, c0)
    src_r0 = max(0, -r0)
    src_c0 = max(0, -c0)

    dst_r1 = min(out_h, r0 + in_h)
    dst_c1 = min(out_w, c0 + in_w)
    src_r1 = src_r0 + (dst_r1 - dst_r0)
    src_c1 = src_c0 + (dst_c1 - dst_c0)

    if dst_r0 < dst_r1 and dst_c0 < dst_c1:
        out[dst_r0:dst_r1, dst_c0:dst_c1] = mask[src_r0:src_r1, src_c0:src_c1]
    return out


def _rotate_mask_in_box(mask: np.ndarray, turns_cw: int) -> np.ndarray:
    turns = int(turns_cw) % 4
    rotated = np.rot90(mask.astype(bool), -turns)
    return _center_clip(rotated, mask.shape)


def _marker_side(coord: tuple[int, int], shape: tuple[int, int]) -> str:
    r, c = map(int, coord)
    h, w = map(int, shape)
    on_top = r == 0
    on_bottom = r == h - 1
    on_left = c == 0
    on_right = c == w - 1
    assert on_top + on_bottom + on_left + on_right == 1, "marker must be on exactly one side"
    if on_top:
        return "top"
    if on_right:
        return "right"
    if on_bottom:
        return "bottom"
    return "left"


def _extract_color_data(grid: np.ndarray) -> list[tuple[int, tuple[int, int], np.ndarray, str]]:
    h, w = grid.shape
    boundary = np.zeros_like(grid, dtype=bool)
    boundary[0, :] = True
    boundary[-1, :] = True
    boundary[:, 0] = True
    boundary[:, -1] = True

    data = []
    for color in colors_present(grid, exclude={0}):
        marker_locs = np.argwhere((grid == color) & boundary)
        assert len(marker_locs) == 1, "each color must have exactly one boundary marker"
        side = _marker_side((int(marker_locs[0, 0]), int(marker_locs[0, 1])), (h, w))

        local_mask = (grid == color) & ~boundary
        box = bbox(local_mask)
        assert box is not None, "each color must have an interior partial sprite"
        r0, c0, r1, c1 = box
        crop = local_mask[r0 : r1 + 1, c0 : c1 + 1]
        data.append((int(color), (int(r0), int(c0)), crop.astype(bool), side))
    assert len(data) == 3, "expected exactly three colored sprite families"
    return data


def main(grid: np.ndarray) -> np.ndarray:
    """
    Complete aligned partial sprites, then rotate each finished sprite from its boundary marker.

    Input structure:
    - A black background contains exactly three non-black colors.
    - For each color, interior pixels form one partial copy of a common sprite mask, all aligned by their top-left origin.
    - Each color also has exactly one same-colored marker pixel on the outer boundary, with at most one marker per side.

    Rule:
    - Remove boundary markers from the object extraction step and crop each color's interior pixels to its bounding box.
    - Top-left align the three cropped partial masks and take their pixelwise union to recover the full sprite template.
    - For each color, place that full template back at the partial sprite's origin, rotate it by the angle encoded by the marker side (top/right/bottom/left = 0/90/180/270 clockwise), center the rotated sprite in the original template box, and clip anything outside that box.
    - Return the grid with only these rotated completed sprites on black background.

    Key insight:
    - The boundary markers do not belong to the sprites; they only encode rotation, while the full sprite shape must first be reconstructed by unioning complementary partial copies across colors.
    """
    g = np.asarray(grid, dtype=int)
    assert g.ndim == 2, "expected a 2D grid"

    color_data = _extract_color_data(g)
    template_h = max(mask.shape[0] for _, _, mask, _ in color_data)
    template_w = max(mask.shape[1] for _, _, mask, _ in color_data)

    template = np.zeros((template_h, template_w), dtype=bool)
    for _, _, mask, _ in color_data:
        h, w = mask.shape
        template[:h, :w] |= mask
    assert np.any(template), "recovered template must be nonempty"

    out = np.zeros_like(g)
    for color, (r0, c0), _mask, side in color_data:
        rotated = _rotate_mask_in_box(template, SIDE_TO_TURNS[side])
        out[r0 : r0 + template_h, c0 : c0 + template_w][rotated] = color
    return out


def _path(a: tuple[int, int], b: tuple[int, int]) -> list[tuple[int, int]]:
    r, c = a
    tr, tc = b
    pts = [(r, c)]
    while (r, c) != (tr, tc):
        options = []
        if r != tr:
            options.append((r + (1 if tr > r else -1), c))
        if c != tc:
            options.append((r, c + (1 if tc > c else -1)))
        r, c = random.choice(options)
        pts.append((r, c))
    return pts


def _neighbors4(r: int, c: int, h: int, w: int) -> list[tuple[int, int]]:
    out = []
    for dr, dc in ((-1, 0), (1, 0), (0, -1), (0, 1)):
        nr = r + dr
        nc = c + dc
        if 0 <= nr < h and 0 <= nc < w:
            out.append((nr, nc))
    return out


def _rotation_signatures(mask: np.ndarray) -> list[bytes]:
    return [_rotate_mask_in_box(mask, k).tobytes() for k in range(4)]


def _generate_template(shape: tuple[int, int]) -> np.ndarray | None:
    h, w = map(int, shape)
    area = h * w
    min_on = max(6, int(np.ceil(0.4 * area)))
    max_on = max(min_on, int(np.floor(0.75 * area)))

    for _ in range(120):
        mask = np.zeros((h, w), dtype=bool)
        center = (h // 2, w // 2)
        anchors = [
            (0, random.randrange(w)),
            (random.randrange(h), 0),
            (h - 1, random.randrange(w)),
            (random.randrange(h), w - 1),
        ]
        random.shuffle(anchors)
        for anchor in anchors:
            for rr, cc in _path(center, anchor):
                mask[rr, cc] = True

        if int(mask.sum()) > max_on:
            continue

        target = random.randint(max(min_on, int(mask.sum())), max_on)
        while int(mask.sum()) < target:
            frontier = {
                nbr
                for rr, cc in np.argwhere(mask)
                for nbr in _neighbors4(int(rr), int(cc), h, w)
                if not mask[nbr]
            }
            if not frontier:
                break
            rr, cc = random.choice(tuple(frontier))
            mask[rr, cc] = True

        if not (mask[0].any() and mask[-1].any() and mask[:, 0].any() and mask[:, -1].any()):
            continue
        if len(set(_rotation_signatures(mask))) != 4:
            continue
        return mask
    return None


def _generate_partials(template: np.ndarray) -> list[np.ndarray] | None:
    h, w = template.shape
    coords = [tuple(map(int, rc)) for rc in np.argwhere(template)]
    area = len(coords)
    top_cells = [p for p in coords if p[0] == 0]
    left_cells = [p for p in coords if p[1] == 0]
    bottom_cells = [p for p in coords if p[0] == h - 1]
    right_cells = [p for p in coords if p[1] == w - 1]
    patterns = [
        ({0}, 1),
        ({1}, 1),
        ({2}, 1),
        ({0, 1}, 3),
        ({0, 2}, 3),
        ({1, 2}, 3),
        ({0, 1, 2}, 1),
    ]

    for _ in range(160):
        protected = [set() for _ in range(3)]
        for idx in range(3):
            protected[idx].add(random.choice(top_cells))
            protected[idx].add(random.choice(left_cells))

        bottom_owner = random.randrange(3)
        right_owner = random.randrange(3)
        protected[bottom_owner].add(random.choice(bottom_cells))
        protected[right_owner].add(random.choice(right_cells))

        shrink_h = random.choice([i for i in range(3) if i != bottom_owner])
        shrink_w = random.choice([i for i in range(3) if i != right_owner])

        keepers: dict[tuple[int, int], set[int]] = {}
        weighted_patterns = [pattern for pattern, weight in patterns for _ in range(weight)]
        for coord in coords:
            must_keep = {idx for idx in range(3) if coord in protected[idx]}
            if must_keep:
                subset = set(must_keep)
                for idx in range(3):
                    if idx not in subset and random.random() < 0.2:
                        subset.add(idx)
            else:
                subset = set(random.choice(weighted_patterns))
            keepers[coord] = subset

        for coord in bottom_cells:
            if shrink_h in keepers[coord] and coord not in protected[shrink_h]:
                if keepers[coord] == {shrink_h}:
                    keepers[coord].add(random.choice([i for i in range(3) if i != shrink_h]))
                keepers[coord].discard(shrink_h)
        for coord in right_cells:
            if shrink_w in keepers[coord] and coord not in protected[shrink_w]:
                if keepers[coord] == {shrink_w}:
                    keepers[coord].add(random.choice([i for i in range(3) if i != shrink_w]))
                keepers[coord].discard(shrink_w)

        masks = [np.zeros_like(template, dtype=bool) for _ in range(3)]
        for coord, subset in keepers.items():
            for idx in subset:
                masks[idx][coord] = True

        if not np.array_equal(np.logical_or.reduce(masks), template):
            continue

        boxes = [bbox(mask) for mask in masks]
        if any(box is None for box in boxes):
            continue

        valid = True
        for box, mask in zip(boxes, masks):
            r0, c0, r1, c1 = box
            if (r0, c0) != (0, 0):
                valid = False
                break
            missing = area - int(mask.sum())
            if missing < 2 or missing > max(4, int(np.floor(0.45 * area))):
                valid = False
                break
            if not mask[0].any() or not mask[:, 0].any():
                valid = False
                break
            if int(mask.sum()) < max(6, int(np.ceil(0.5 * area))):
                valid = False
                break
            if not template[: r1 + 1, : c1 + 1].shape == mask[: r1 + 1, : c1 + 1].shape:
                valid = False
                break
        if not valid:
            continue

        shape_set = {(box[2] - box[0] + 1, box[3] - box[1] + 1) for box in boxes}
        if len(shape_set) == 1 and next(iter(shape_set)) == template.shape:
            continue

        if len({mask.tobytes() for mask in masks}) < 2:
            continue
        return masks
    return None


def _rects_separated(
    a: tuple[int, int], b: tuple[int, int], h: int, w: int, gap: int
) -> bool:
    ar, ac = a
    br, bc = b
    return (
        ar + h + gap <= br
        or br + h + gap <= ar
        or ac + w + gap <= bc
        or bc + w + gap <= ac
    )


def _place_windows(grid_h: int, grid_w: int, h: int, w: int) -> list[tuple[int, int]] | None:
    rows = range(WINDOW_MARGIN, grid_h - h - WINDOW_MARGIN + 1)
    cols = range(WINDOW_MARGIN, grid_w - w - WINDOW_MARGIN + 1)
    candidates = [(r, c) for r in rows for c in cols]
    random.shuffle(candidates)
    placed: list[tuple[int, int]] = []
    for r, c in candidates:
        if all(_rects_separated((r, c), other, h, w, WINDOW_GAP) for other in placed):
            placed.append((r, c))
            if len(placed) == 3:
                return placed
    return None


def _marker_coord(side: str, shape: tuple[int, int]) -> tuple[int, int]:
    h, w = map(int, shape)
    if side == "top":
        return 0, random.randint(1, w - 2)
    if side == "right":
        return random.randint(1, h - 2), w - 1
    if side == "bottom":
        return h - 1, random.randint(1, w - 2)
    return random.randint(1, h - 2), 0


def _render_objects(
    shape: tuple[int, int],
    colors: list[int],
    origins: list[tuple[int, int]],
    masks: list[np.ndarray],
) -> np.ndarray:
    out = np.zeros(shape, dtype=int)
    box_h, box_w = masks[0].shape
    for color, (r0, c0), mask in zip(colors, origins, masks):
        out[r0 : r0 + box_h, c0 : c0 + box_w][mask] = color
    return out


def _semantic_checks(
    inp: np.ndarray,
    expected: np.ndarray,
    colors: list[int],
    origins: list[tuple[int, int]],
    template: np.ndarray,
    partials: list[np.ndarray],
    sides_by_color: dict[int, str],
) -> bool:
    if not np.array_equal(main(inp), expected):
        return False
    if np.any(expected[0, :]) or np.any(expected[-1, :]) or np.any(expected[:, 0]) or np.any(expected[:, -1]):
        return False

    no_rotation_masks = [template.copy() for _ in colors]
    no_rotation = _render_objects(inp.shape, colors, origins, no_rotation_masks)
    if np.array_equal(no_rotation, expected):
        return False

    rotated_partials = [
        _rotate_mask_in_box(partial, SIDE_TO_TURNS[sides_by_color[color]])
        for color, partial in zip(colors, partials)
    ]
    if np.array_equal(_render_objects(inp.shape, colors, origins, rotated_partials), expected):
        return False

    for color in colors:
        marker_count = int(
            np.count_nonzero(
                ((inp == color) & ((np.indices(inp.shape)[0] == 0) | (np.indices(inp.shape)[0] == inp.shape[0] - 1) | (np.indices(inp.shape)[1] == 0) | (np.indices(inp.shape)[1] == inp.shape[1] - 1)))
            )
        )
        if marker_count != 1:
            return False
    return True


def _attempt_generate() -> np.ndarray | None:
    template_shape = random.choice(TEMPLATE_DIMS)
    template = _generate_template(template_shape)
    if template is None:
        return None

    partials = _generate_partials(template)
    if partials is None:
        return None

    box_h, box_w = template.shape
    min_h = max(GRID_H_RANGE[0], box_h + 2 * WINDOW_MARGIN + 1)
    min_w = max(GRID_W_RANGE[0], box_w + 2 * WINDOW_MARGIN + 1)
    if min_h > GRID_H_RANGE[1] or min_w > GRID_W_RANGE[1]:
        return None

    grid_h = random.randint(min_h, GRID_H_RANGE[1])
    grid_w = random.randint(min_w, GRID_W_RANGE[1])
    origins = _place_windows(grid_h, grid_w, box_h, box_w)
    if origins is None:
        return None

    colors = random.sample(PALETTE, 3)
    used_sides = random.sample(SIDES, 3)
    sides_by_color = {color: side for color, side in zip(colors, used_sides)}

    inp = _render_objects((grid_h, grid_w), colors, origins, partials)
    for color in colors:
        mr, mc = _marker_coord(sides_by_color[color], inp.shape)
        if inp[mr, mc] != 0:
            return None
        inp[mr, mc] = color

    expected_masks = [_rotate_mask_in_box(template, SIDE_TO_TURNS[sides_by_color[color]]) for color in colors]
    expected = _render_objects(inp.shape, colors, origins, expected_masks)

    if not _semantic_checks(inp, expected, colors, origins, template, partials, sides_by_color):
        return None
    return inp


def generate_input() -> np.ndarray:
    for _ in range(MAX_ATTEMPTS):
        grid = _attempt_generate()
        if grid is not None:
            return grid
    raise RuntimeError("failed to generate a valid input for puzzle 6")
