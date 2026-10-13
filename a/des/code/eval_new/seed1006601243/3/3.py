from __future__ import annotations

from collections import deque

import numpy as np


BLACK = 0
GREY = 5
PIECE_PALETTE = (1, 2, 3, 4, 6, 7, 8, 9)
GEN_ATTEMPTS = 400
MASK_ATTEMPTS = 160
RNG = np.random.default_rng()


def _neighbors4(r: int, c: int, h: int, w: int):
    if r > 0:
        yield r - 1, c
    if r + 1 < h:
        yield r + 1, c
    if c > 0:
        yield r, c - 1
    if c + 1 < w:
        yield r, c + 1


def _coords_bbox(coords: np.ndarray) -> tuple[int, int, int, int]:
    rows = coords[:, 0]
    cols = coords[:, 1]
    return int(rows.min()), int(cols.min()), int(rows.max()), int(cols.max())


def _components(grid: np.ndarray, bg: int) -> list[dict[str, object]]:
    h, w = grid.shape
    seen = np.zeros((h, w), dtype=bool)
    comps: list[dict[str, object]] = []

    for r in range(h):
        for c in range(w):
            color = int(grid[r, c])
            if color == bg or seen[r, c]:
                continue
            seen[r, c] = True
            queue = deque([(r, c)])
            cells: list[tuple[int, int]] = []
            while queue:
                rr, cc = queue.popleft()
                cells.append((int(rr), int(cc)))
                for nr, nc in _neighbors4(int(rr), int(cc), h, w):
                    if not seen[nr, nc] and int(grid[nr, nc]) == color:
                        seen[nr, nc] = True
                        queue.append((int(nr), int(nc)))

            coords = np.array(cells, dtype=int)
            comps.append(
                {
                    "color": color,
                    "coords": coords,
                    "bbox": _coords_bbox(coords),
                    "size": int(coords.shape[0]),
                }
            )
    return comps


def _is_frame_component(grid: np.ndarray, comp: dict[str, object], bg: int) -> bool:
    color = int(comp["color"])
    r0, c0, r1, c1 = comp["bbox"]
    h = int(r1 - r0 + 1)
    w = int(c1 - c0 + 1)
    if h < 3 or w < 3:
        return False

    sub = grid[r0 : r1 + 1, c0 : c1 + 1]
    if not np.all(sub[0, :] == color):
        return False
    if not np.all(sub[-1, :] == color):
        return False
    if not np.all(sub[:, 0] == color):
        return False
    if not np.all(sub[:, -1] == color):
        return False
    if np.any(sub[1:-1, 1:-1] == color):
        return False

    perimeter = 2 * (h + w) - 4
    if int(comp["size"]) != perimeter:
        return False

    inner = sub[1:-1, 1:-1]
    return bool(np.any((inner != bg) & (inner != color)))


def _find_target_frame(grid: np.ndarray, comps: list[dict[str, object]], bg: int) -> dict[str, object]:
    candidates = []
    for comp in comps:
        if not _is_frame_component(grid, comp, bg):
            continue
        r0, c0, r1, c1 = comp["bbox"]
        area = int((r1 - r0 + 1) * (c1 - c0 + 1))
        candidates.append((int(c0 + c1), -area, comp))

    assert candidates, "expected one framed target rectangle"
    candidates.sort(key=lambda item: (item[0], item[1]))
    return candidates[0][2]


def main(grid: np.ndarray) -> np.ndarray:
    """
    Crop the grey target frame and restack the scattered same-sized fragments by descending area.

    Input structure:
    - The background is black.
    - The left side contains a one-pixel grey rectangular frame whose interior is black except for isolated colored marker pixels.
    - The right side contains several disconnected monochrome fragments, one color per fragment, and every fragment bbox matches the frame interior size.

    Rule:
    - Detect the grey target frame and clear its interior markers.
    - For each fragment, translate it so that its bounding box aligns exactly with the frame interior.
    - Overlay the aligned fragments from largest area to smallest area, with later smaller fragments covering earlier larger ones.
    - Return only the cropped target frame together with its stacked interior result.

    Key insight:
    - The marker dots only confirm the common alignment; the actual layer order comes from comparing fragment areas, because all fragments already share the frame interior bbox.
    """
    g = np.asarray(grid, dtype=int)
    assert g.ndim == 2 and g.size > 0, "expected a non-empty 2D grid"

    bg = BLACK if np.any(g == BLACK) else int(np.bincount(g.reshape(-1)).argmax())
    comps = _components(g, bg)
    frame = _find_target_frame(g, comps, bg)
    frame_color = int(frame["color"])
    fr0, fc0, fr1, fc1 = frame["bbox"]

    inner_h = int(fr1 - fr0 - 1)
    inner_w = int(fc1 - fc0 - 1)
    assert inner_h > 0 and inner_w > 0

    pieces: list[dict[str, object]] = []
    for comp in comps:
        if comp is frame:
            continue
        cr0, cc0, cr1, cc1 = comp["bbox"]
        if int(comp["size"]) == 1 and fr0 < cr0 < fr1 and fc0 < cc0 < fc1:
            continue
        if int(comp["color"]) == frame_color:
            continue
        if (int(cr1 - cr0 + 1), int(cc1 - cc0 + 1)) != (inner_h, inner_w):
            continue
        pieces.append(comp)

    assert pieces, "expected at least one fragment"
    assert len({int(comp["color"]) for comp in pieces}) == len(pieces), "fragments must have unique colors"

    out = np.full((int(fr1 - fr0 + 1), int(fc1 - fc0 + 1)), bg, dtype=int)
    out[0, :] = frame_color
    out[-1, :] = frame_color
    out[:, 0] = frame_color
    out[:, -1] = frame_color
    interior = out[1:-1, 1:-1]

    pieces.sort(key=lambda comp: (-int(comp["size"]), int(comp["bbox"][0]), int(comp["bbox"][1])))
    for comp in pieces:
        pr0, pc0, _pr1, _pc1 = comp["bbox"]
        coords = comp["coords"]
        rr = coords[:, 0] - int(pr0)
        cc = coords[:, 1] - int(pc0)
        interior[rr, cc] = int(comp["color"])

    return out


def _manhattan_path(a: tuple[int, int], b: tuple[int, int], vertical_first: bool) -> list[tuple[int, int]]:
    ar, ac = a
    br, bc = b
    path: list[tuple[int, int]] = []

    if vertical_first:
        step_r = 1 if br >= ar else -1
        for rr in range(ar, br + step_r, step_r):
            path.append((int(rr), int(ac)))
        step_c = 1 if bc >= ac else -1
        for cc in range(ac + step_c, bc + step_c, step_c):
            path.append((int(br), int(cc)))
    else:
        step_c = 1 if bc >= ac else -1
        for cc in range(ac, bc + step_c, step_c):
            path.append((int(ar), int(cc)))
        step_r = 1 if br >= ar else -1
        for rr in range(ar + step_r, br + step_r, step_r):
            path.append((int(rr), int(bc)))
    return path


def _touches_all_sides(mask: np.ndarray) -> bool:
    return bool(mask[0, :].any() and mask[-1, :].any() and mask[:, 0].any() and mask[:, -1].any())


def _is_perimeter_mask(mask: np.ndarray) -> bool:
    expected = np.zeros_like(mask, dtype=bool)
    expected[0, :] = True
    expected[-1, :] = True
    expected[:, 0] = True
    expected[:, -1] = True
    return bool(np.array_equal(mask, expected))


def _grow_mask(
    inner_h: int,
    inner_w: int,
    shared_cell: tuple[int, int],
    target_area: int,
    rng: np.random.Generator,
) -> np.ndarray | None:
    cells = {shared_cell}
    edge_points = [
        (0, int(rng.integers(0, inner_w))),
        (inner_h - 1, int(rng.integers(0, inner_w))),
        (int(rng.integers(0, inner_h)), 0),
        (int(rng.integers(0, inner_h)), inner_w - 1),
    ]
    for point in edge_points:
        for cell in _manhattan_path(shared_cell, point, bool(rng.integers(0, 2))):
            cells.add(cell)
    if len(cells) > target_area:
        return None

    while len(cells) < target_area:
        frontier = []
        frontier_set = set()
        for r, c in cells:
            for nr, nc in _neighbors4(int(r), int(c), inner_h, inner_w):
                if (nr, nc) in cells or (nr, nc) in frontier_set:
                    continue
                frontier.append((int(nr), int(nc)))
                frontier_set.add((int(nr), int(nc)))
        if not frontier:
            return None
        cells.add(frontier[int(rng.integers(0, len(frontier)))])

    mask = np.zeros((inner_h, inner_w), dtype=bool)
    for r, c in cells:
        mask[int(r), int(c)] = True

    if not _touches_all_sides(mask):
        return None
    if mask.all() or _is_perimeter_mask(mask):
        return None
    return mask


def _sample_mask(inner_h: int, inner_w: int, shared_cell: tuple[int, int], rng: np.random.Generator) -> np.ndarray | None:
    area = inner_h * inner_w
    min_target = max(inner_h + inner_w + 1, int(np.ceil(area * 0.38)))
    max_target = min(area - 1, int(np.floor(area * 0.78)))
    if min_target > max_target:
        return None

    for _ in range(MASK_ATTEMPTS):
        target_area = int(rng.integers(min_target, max_target + 1))
        mask = _grow_mask(inner_h, inner_w, shared_cell, target_area, rng)
        if mask is not None:
            return mask
    return None


def _generate_masks(inner_h: int, inner_w: int, count: int, rng: np.random.Generator):
    for _ in range(MASK_ATTEMPTS):
        shared_cell = (int(rng.integers(1, inner_h - 1)), int(rng.integers(1, inner_w - 1)))
        masks = []
        for _piece in range(count):
            mask = _sample_mask(inner_h, inner_w, shared_cell, rng)
            if mask is None:
                break
            masks.append(mask)
        if len(masks) != count:
            continue

        sizes = [int(mask.sum()) for mask in masks]
        if len(set(sizes)) != count:
            continue

        unique_sets = []
        for idx, mask in enumerate(masks):
            others = np.zeros_like(mask, dtype=bool)
            for jdx, other in enumerate(masks):
                if idx != jdx:
                    others |= other
            unique = np.argwhere(mask & ~others)
            if unique.size == 0:
                break
            unique_sets.append(unique)
        else:
            return masks, unique_sets
    return None


def _choose_marker_cell(unique_coords: np.ndarray, inner_h: int, inner_w: int, rng: np.random.Generator) -> tuple[int, int]:
    candidates = [
        (int(r), int(c))
        for r, c in unique_coords
        if 0 < int(r) < inner_h - 1 and 0 < int(c) < inner_w - 1
    ]
    if not candidates:
        candidates = [(int(r), int(c)) for r, c in unique_coords]
    return candidates[int(rng.integers(0, len(candidates)))]


def _render_expected(inner_h: int, inner_w: int, piece_colors: list[int], masks: list[np.ndarray]) -> np.ndarray:
    out = np.full((inner_h + 2, inner_w + 2), BLACK, dtype=int)
    out[0, :] = GREY
    out[-1, :] = GREY
    out[:, 0] = GREY
    out[:, -1] = GREY
    interior = out[1:-1, 1:-1]

    order = sorted(range(len(masks)), key=lambda idx: -int(masks[idx].sum()))
    for idx in order:
        interior[masks[idx]] = int(piece_colors[idx])
    return out


def _build_slot_positions(
    piece_top: int,
    piece_left: int,
    inner_h: int,
    inner_w: int,
    rows: int,
    cols: int,
    gap_y: int,
    gap_x: int,
) -> list[tuple[int, int]]:
    slots = []
    for rr in range(rows):
        for cc in range(cols):
            top = piece_top + rr * (inner_h + gap_y)
            left = piece_left + cc * (inner_w + gap_x)
            slots.append((int(top), int(left)))
    return slots


def _gen_attempt() -> np.ndarray | None:
    rng = RNG
    piece_count = int(rng.integers(2, 5))
    inner_h = int(rng.integers(4, 8))
    inner_w = int(rng.integers(4, 8))
    frame_h = inner_h + 2
    frame_w = inner_w + 2

    mask_bundle = _generate_masks(inner_h, inner_w, piece_count, rng)
    if mask_bundle is None:
        return None
    masks, unique_sets = mask_bundle

    gap_x = int(rng.integers(1, 3))
    gap_y = int(rng.integers(1, 3))
    feasible_cols = []
    for cols in (1, 2):
        rows = (piece_count + cols - 1) // cols
        layout_w = cols * inner_w + (cols - 1) * gap_x
        layout_h = rows * inner_h + (rows - 1) * gap_y
        min_w = max(2 * frame_w, 2 * layout_w - 1, frame_w + layout_w)
        min_h = max(frame_h, layout_h)
        if min_w <= 30 and min_h <= 30:
            feasible_cols.append((cols, rows, layout_w, layout_h, min_w, min_h))
    if not feasible_cols:
        return None

    cols, rows, layout_w, layout_h, min_w, min_h = feasible_cols[int(rng.integers(0, len(feasible_cols)))]
    grid_w = int(rng.integers(min_w, 31))
    grid_h = int(rng.integers(min_h, 31))
    mid = grid_w // 2

    left_room = mid - frame_w
    right_room = grid_w - mid - layout_w
    if left_room < 0 or right_room < 0:
        return None

    frame_top = int(rng.integers(0, grid_h - frame_h + 1))
    frame_left = int(rng.integers(0, left_room + 1))
    piece_top = int(rng.integers(0, grid_h - layout_h + 1))
    piece_left = mid + int(rng.integers(0, right_room + 1))

    colors = list(map(int, rng.choice(PIECE_PALETTE, size=piece_count, replace=False)))
    slot_positions = _build_slot_positions(piece_top, piece_left, inner_h, inner_w, rows, cols, gap_y, gap_x)
    rng.shuffle(slot_positions)
    slot_positions = slot_positions[:piece_count]

    marker_cells = [
        _choose_marker_cell(unique_sets[idx], inner_h, inner_w, rng)
        for idx in range(piece_count)
    ]

    grid = np.zeros((grid_h, grid_w), dtype=int)
    grid[frame_top, frame_left : frame_left + frame_w] = GREY
    grid[frame_top + frame_h - 1, frame_left : frame_left + frame_w] = GREY
    grid[frame_top : frame_top + frame_h, frame_left] = GREY
    grid[frame_top : frame_top + frame_h, frame_left + frame_w - 1] = GREY

    for color, marker in zip(colors, marker_cells):
        mr, mc = marker
        grid[frame_top + 1 + mr, frame_left + 1 + mc] = int(color)

    for idx, (top, left) in enumerate(slot_positions):
        patch = grid[top : top + inner_h, left : left + inner_w]
        patch[masks[idx]] = int(colors[idx])

    expected = _render_expected(inner_h, inner_w, colors, masks)
    if not np.array_equal(main(grid), expected):
        return None
    return grid


def generate_input() -> np.ndarray:
    for _ in range(GEN_ATTEMPTS):
        grid = _gen_attempt()
        if grid is not None:
            return grid
    raise RuntimeError("failed to generate a valid input")
