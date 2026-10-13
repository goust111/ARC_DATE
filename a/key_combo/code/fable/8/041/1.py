from __future__ import annotations

from collections import Counter, deque
import random

import numpy as np


BLACK = 0
YELLOW = 4
GREY = 5
PALETTE = [1, 2, 3, 6, 7, 8, 9]


def _get_bg(grid: np.ndarray) -> int:
    return int(Counter(map(int, grid.ravel())).most_common(1)[0][0])


def _bbox_coords(coords: np.ndarray) -> tuple[int, int, int, int]:
    r0, c0 = coords.min(axis=0)
    r1, c1 = coords.max(axis=0)
    return int(r0), int(c0), int(r1), int(c1)


def _bbox(mask: np.ndarray) -> tuple[int, int, int, int] | None:
    coords = np.argwhere(mask)
    if coords.size == 0:
        return None
    return _bbox_coords(coords)


def _crop_bbox(grid: np.ndarray, box: tuple[int, int, int, int]) -> np.ndarray:
    r0, c0, r1, c1 = box
    return grid[r0 : r1 + 1, c0 : c1 + 1].copy()


def _neighbors4(r: int, c: int) -> tuple[tuple[int, int], ...]:
    return ((r - 1, c), (r + 1, c), (r, c - 1), (r, c + 1))


def _in_bounds(shape: tuple[int, int], r: int, c: int) -> bool:
    h, w = shape
    return 0 <= r < h and 0 <= c < w


def _components_of_mask(mask: np.ndarray) -> list[np.ndarray]:
    h, w = mask.shape
    seen = np.zeros((h, w), dtype=bool)
    out: list[np.ndarray] = []
    for r in range(h):
        for c in range(w):
            if not mask[r, c] or seen[r, c]:
                continue
            q = deque([(r, c)])
            seen[r, c] = True
            coords = []
            while q:
                cr, cc = q.popleft()
                coords.append((cr, cc))
                for nr, nc in _neighbors4(cr, cc):
                    if _in_bounds(mask.shape, nr, nc) and mask[nr, nc] and not seen[nr, nc]:
                        seen[nr, nc] = True
                        q.append((nr, nc))
            out.append(np.array(coords, dtype=int))
    return out


def _find_components(
    grid: np.ndarray,
    *,
    bg: int,
    colors: list[int] | None = None,
) -> list[dict]:
    if colors is None:
        colors = sorted(set(map(int, grid.ravel())) - {int(bg)})
    comps: list[dict] = []
    for color in colors:
        mask = grid == int(color)
        for coords in _components_of_mask(mask):
            comps.append(
                {
                    "color": int(color),
                    "coords": coords,
                    "bbox": _bbox_coords(coords),
                    "area": int(coords.shape[0]),
                }
            )
    return comps


def _paste(dst: np.ndarray, patch: np.ndarray, r0: int, c0: int, *, transparent: int | None = None) -> None:
    h, w = patch.shape
    view = dst[r0 : r0 + h, c0 : c0 + w]
    if transparent is None:
        view[:] = patch
    else:
        mask = patch != int(transparent)
        view[mask] = patch[mask]


def _frame_perimeter_mask(height: int, width: int) -> np.ndarray:
    mask = np.zeros((height, width), dtype=bool)
    mask[0, :] = True
    mask[-1, :] = True
    mask[:, 0] = True
    mask[:, -1] = True
    return mask


def _find_frame(grid: np.ndarray, bg: int) -> tuple[int, int, int, int]:
    candidates = []
    for comp in _find_components(grid, bg=bg, colors=[GREY]):
        r0, c0, r1, c1 = comp["bbox"]
        h = r1 - r0 + 1
        w = c1 - c0 + 1
        if h < 3 or w < 3:
            continue
        patch = _crop_bbox(grid, comp["bbox"])
        mask = patch == GREY
        if np.array_equal(mask, _frame_perimeter_mask(h, w)):
            candidates.append(comp["bbox"])
    assert len(candidates) == 1, "expected exactly one grey rectangular frame"
    return candidates[0]


def _piece_info_from_component(grid: np.ndarray, comp: dict) -> dict:
    box = comp["bbox"]
    patch = _crop_bbox(grid, box)
    bg = _get_bg(grid)
    colors = sorted(set(map(int, patch.ravel())) - {bg})
    assert len(colors) == 1
    return {
        "color": colors[0],
        "bbox": box,
        "patch": patch,
        "mask": patch != bg,
    }


def _mask_overlap(mask_a: np.ndarray, pos_a: tuple[int, int], mask_b: np.ndarray, pos_b: tuple[int, int]) -> bool:
    ra, ca = pos_a
    rb, cb = pos_b
    ha, wa = mask_a.shape
    hb, wb = mask_b.shape
    r0 = max(ra, rb)
    c0 = max(ca, cb)
    r1 = min(ra + ha, rb + hb)
    c1 = min(ca + wa, cb + wb)
    if r0 >= r1 or c0 >= c1:
        return False
    a = mask_a[r0 - ra : r1 - ra, c0 - ca : c1 - ca]
    b = mask_b[r0 - rb : r1 - rb, c0 - cb : c1 - cb]
    return bool(np.any(a & b))


def _enumerate_packings(interior_shape: tuple[int, int], pieces: list[dict]) -> list[tuple[tuple[int, int], tuple[int, int]]]:
    (h0, w0), (h1, w1) = pieces[0]["mask"].shape, pieces[1]["mask"].shape
    H, W = interior_shape
    sols = []
    for r0 in range(H - h0 + 1):
        for c0 in range(W - w0 + 1):
            for r1 in range(H - h1 + 1):
                for c1 in range(W - w1 + 1):
                    if not _mask_overlap(pieces[0]["mask"], (r0, c0), pieces[1]["mask"], (r1, c1)):
                        sols.append(((r0, c0), (r1, c1)))
    return sols


def _intended_positions_from_masks(mask_a: np.ndarray, mask_b: np.ndarray) -> tuple[tuple[int, int], tuple[int, int], dict, dict]:
    box_a = _bbox(mask_a)
    box_b = _bbox(mask_b)
    assert box_a is not None and box_b is not None
    ra, ca, _, _ = box_a
    rb, cb, _, _ = box_b
    sprite_a = _crop_bbox(mask_a.astype(int), box_a).astype(bool)
    sprite_b = _crop_bbox(mask_b.astype(int), box_b).astype(bool)
    piece_a = {"mask": sprite_a}
    piece_b = {"mask": sprite_b}
    return (ra, ca), (rb, cb), piece_a, piece_b


def _solve_positions(interior_shape: tuple[int, int], pieces: list[dict]) -> tuple[tuple[int, int], tuple[int, int]]:
    sols = _enumerate_packings(interior_shape, pieces)
    assert len(sols) == 1, "packing inside the frame must be unique"
    return sols[0]


def main(grid: np.ndarray) -> np.ndarray:
    """
    Fit the two external puzzle pieces into the only non-overlapping translation inside the grey frame, then color the leftover interior cells yellow.

    Input structure:
    - The grid has a black background with one grey hollow rectangular frame.
    - The frame interior is empty in the input.
    - Exactly two separate monochromatic puzzle pieces of different colors lie outside the frame, and yellow does not appear.

    Rule:
    - Translate the two pieces into the frame interior without rotating or flipping them.
    - Use the unique placement where both pieces stay inside the frame and do not overlap.
    - Paint the uncovered interior cells yellow, erase the pieces from their original locations, and keep the grey frame unchanged.

    Key insight:
    - The yellow region is not given directly: it is the negative space left only after finding the unique legal joint assembly of the two pieces.
    """
    g = np.asarray(grid, dtype=int)
    assert g.ndim == 2
    bg = _get_bg(g)
    assert bg == BLACK, "expected black background"
    assert not np.any(g == YELLOW), "yellow is output-only"

    frame = _find_frame(g, bg)
    r0, c0, r1, c1 = frame
    assert np.all(g[r0 + 1 : r1, c0 + 1 : c1] == bg), "frame interior must be empty"

    piece_colors = [c for c in sorted(set(map(int, g.ravel()))) if c not in (bg, GREY)]
    assert len(piece_colors) == 2, "expected exactly two piece colors"
    piece_comps = _find_components(g, bg=bg, colors=piece_colors)
    assert len(piece_comps) == 2, "expected exactly two external pieces"

    pieces = []
    for comp in sorted(piece_comps, key=lambda item: item["color"]):
        rr, cc = comp["coords"].T
        assert not np.any((r0 <= rr) & (rr <= r1) & (c0 <= cc) & (cc <= c1)), "pieces must start outside the frame"
        pieces.append(_piece_info_from_component(g, comp))

    interior_shape = (r1 - r0 - 1, c1 - c0 - 1)
    positions = _solve_positions(interior_shape, pieces)

    out = g.copy()
    out[(out != bg) & (out != GREY)] = bg
    out[r0 + 1 : r1, c0 + 1 : c1] = YELLOW
    for piece, (pr, pc) in zip(pieces, positions):
        _paste(out, piece["patch"], r0 + 1 + pr, c0 + 1 + pc, transparent=bg)
    return out


# --------------- generator ---------------

GEN_INTERIOR_H = (6, 8)
GEN_INTERIOR_W = (6, 9)
GEN_HOLE_COMPONENTS = (1, 3)
GEN_HOLE_CELLS_PER_COMPONENT = (1, 3)
GEN_GRID_H = (16, 26)
GEN_GRID_W = (16, 26)
GEN_RETRIES = 600


def _mask_connected(mask: np.ndarray) -> bool:
    coords = np.argwhere(mask)
    return coords.size > 0 and len(_components_of_mask(mask)) == 1


def _touch_sides(mask: np.ndarray) -> set[str]:
    h, w = mask.shape
    sides: set[str] = set()
    if np.any(mask[0, :]):
        sides.add("top")
    if np.any(mask[-1, :]):
        sides.add("bottom")
    if np.any(mask[:, 0]):
        sides.add("left")
    if np.any(mask[:, -1]):
        sides.add("right")
    return sides


def _make_union_mask(height: int, width: int) -> np.ndarray:
    mask = np.ones((height, width), dtype=bool)
    interior_cells = [(r, c) for r in range(1, height - 1) for c in range(1, width - 1)]
    random.shuffle(interior_cells)

    n_components = random.randint(*GEN_HOLE_COMPONENTS)
    used: set[tuple[int, int]] = set()
    for _ in range(n_components):
        seeds = [cell for cell in interior_cells if cell not in used]
        if not seeds:
            break
        start = random.choice(seeds)
        target = random.randint(*GEN_HOLE_CELLS_PER_COMPONENT)
        comp = {start}
        frontier = [start]
        while frontier and len(comp) < target:
            cr, cc = random.choice(frontier)
            options = []
            for nr, nc in _neighbors4(cr, cc):
                if 1 <= nr < height - 1 and 1 <= nc < width - 1 and (nr, nc) not in used and (nr, nc) not in comp:
                    options.append((nr, nc))
            if not options:
                frontier.remove((cr, cc))
                continue
            nxt = random.choice(options)
            comp.add(nxt)
            frontier.append(nxt)
        for cell in comp:
            used.add(cell)
            mask[cell] = False
    assert np.any(~mask)
    return mask


def _opposite_corners(height: int, width: int) -> tuple[tuple[int, int], tuple[int, int]]:
    if random.random() < 0.5:
        return (0, 0), (height - 1, width - 1)
    return (0, width - 1), (height - 1, 0)


def _grow_partition(union: np.ndarray) -> tuple[np.ndarray, np.ndarray] | None:
    height, width = union.shape
    seed_a, seed_b = _opposite_corners(height, width)
    a_mask = union.copy()
    b_mask = np.zeros_like(union, dtype=bool)
    b_mask[seed_b] = True
    a_mask[seed_b] = False

    total = int(np.count_nonzero(union))
    target_b = random.randint(max(6, total // 3), min(total - 6, (2 * total) // 3))

    for _ in range(total * 4):
        if int(np.count_nonzero(b_mask)) >= target_b:
            break
        candidates = []
        for r, c in np.argwhere(a_mask):
            cell = (int(r), int(c))
            if cell == seed_a:
                continue
            if not any(_in_bounds(union.shape, nr, nc) and b_mask[nr, nc] for nr, nc in _neighbors4(*cell)):
                continue
            test_a = a_mask.copy()
            test_a[cell] = False
            if _mask_connected(test_a):
                candidates.append(cell)
        if not candidates:
            return None
        choice = random.choice(candidates)
        a_mask[choice] = False
        b_mask[choice] = True

    if int(np.count_nonzero(b_mask)) < target_b:
        return None
    if not _mask_connected(a_mask):
        return None
    if not _mask_connected(b_mask):
        return None
    if np.any(a_mask & b_mask):
        return None
    return a_mask, b_mask


def _mask_to_patch(mask: np.ndarray, color: int) -> tuple[np.ndarray, tuple[int, int]]:
    box = _bbox(mask)
    assert box is not None
    patch = np.full((box[2] - box[0] + 1, box[3] - box[1] + 1), BLACK, dtype=int)
    sub = _crop_bbox(mask.astype(int), box).astype(bool)
    patch[sub] = int(color)
    return patch, (box[0], box[1])


def _bbox_intersects(a: tuple[int, int, int, int], b: tuple[int, int, int, int]) -> bool:
    return not (a[2] < b[0] or b[2] < a[0] or a[3] < b[1] or b[3] < a[1])


def _expand_bbox(box: tuple[int, int, int, int], pad: int, shape: tuple[int, int]) -> tuple[int, int, int, int]:
    h, w = shape
    return (
        max(0, box[0] - pad),
        max(0, box[1] - pad),
        min(h - 1, box[2] + pad),
        min(w - 1, box[3] + pad),
    )


def _place_external_pieces(
    grid: np.ndarray,
    frame_box: tuple[int, int, int, int],
    patches: list[np.ndarray],
) -> bool:
    forbidden_boxes = [_expand_bbox(frame_box, 1, grid.shape)]
    placed_boxes: list[tuple[int, int, int, int]] = []
    order = list(range(len(patches)))
    random.shuffle(order)

    for idx in order:
        patch = patches[idx]
        ph, pw = patch.shape
        candidates = []
        for r in range(1, grid.shape[0] - ph):
            for c in range(1, grid.shape[1] - pw):
                box = (r, c, r + ph - 1, c + pw - 1)
                padded = _expand_bbox(box, 1, grid.shape)
                if any(_bbox_intersects(padded, other) for other in forbidden_boxes + placed_boxes):
                    continue
                if np.any(grid[r : r + ph, c : c + pw] != BLACK):
                    continue
                candidates.append((r, c, box))
        if not candidates:
            return False
        r, c, box = random.choice(candidates)
        _paste(grid, patch, r, c, transparent=BLACK)
        placed_boxes.append(box)
    return True


def _build_expected(
    canvas_shape: tuple[int, int],
    frame_box: tuple[int, int, int, int],
    pieces: list[np.ndarray],
    positions: list[tuple[int, int]],
) -> np.ndarray:
    out = np.full(canvas_shape, BLACK, dtype=int)
    r0, c0, r1, c1 = frame_box
    out[r0, c0 : c1 + 1] = GREY
    out[r1, c0 : c1 + 1] = GREY
    out[r0 : r1 + 1, c0] = GREY
    out[r0 : r1 + 1, c1] = GREY
    out[r0 + 1 : r1, c0 + 1 : c1] = YELLOW
    for patch, (pr, pc) in zip(pieces, positions):
        _paste(out, patch, r0 + 1 + pr, c0 + 1 + pc, transparent=BLACK)
    return out


def _instance_passes_checks(inp: np.ndarray, expected: np.ndarray, piece_masks: list[np.ndarray], intended: list[tuple[int, int]]) -> bool:
    if not np.array_equal(main(inp), expected):
        return False

    frame = _find_frame(inp, BLACK)
    interior_shape = (frame[2] - frame[0] - 1, frame[3] - frame[1] - 1)
    piece_infos = [{"mask": mask} for mask in piece_masks]
    sols = _enumerate_packings(interior_shape, piece_infos)
    if sols != [(tuple(intended[0]), tuple(intended[1]))]:
        return False

    if not np.any(expected == YELLOW):
        return False

    for mask in piece_masks:
        h, w = mask.shape
        placements = (interior_shape[0] - h + 1) * (interior_shape[1] - w + 1)
        if placements < 2:
            return False

    output_colors = sorted(set(map(int, expected.ravel())) - {BLACK, GREY, YELLOW})
    input_colors = sorted(set(map(int, inp.ravel())) - {BLACK, GREY})
    if len(input_colors) != 2 or input_colors != output_colors:
        return False
    return True


def _gen_attempt() -> np.ndarray | None:
    inner_h = random.randint(*GEN_INTERIOR_H)
    inner_w = random.randint(*GEN_INTERIOR_W)
    union = _make_union_mask(inner_h, inner_w)

    partition = _grow_partition(union)
    if partition is None:
        return None
    mask_a, mask_b = partition

    if not np.any(~(mask_a | mask_b)):
        return None

    piece_a_sides = _touch_sides(mask_a)
    piece_b_sides = _touch_sides(mask_b)
    if len(piece_a_sides) < 2 or len(piece_b_sides) < 2:
        return None

    color_a, color_b = random.sample(PALETTE, 2)
    patch_a, pos_a = _mask_to_patch(mask_a, color_a)
    patch_b, pos_b = _mask_to_patch(mask_b, color_b)
    piece_masks = [patch_a != BLACK, patch_b != BLACK]
    intended = [pos_a, pos_b]

    sols = _enumerate_packings((inner_h, inner_w), [{"mask": piece_masks[0]}, {"mask": piece_masks[1]}])
    if sols != [(tuple(pos_a), tuple(pos_b))]:
        return None

    outer_h, outer_w = inner_h + 2, inner_w + 2
    min_grid_h = max(GEN_GRID_H[0], outer_h + 6, max(patch_a.shape[0], patch_b.shape[0]) + 4)
    min_grid_w = max(GEN_GRID_W[0], outer_w + 6, max(patch_a.shape[1], patch_b.shape[1]) + 4)
    if min_grid_h > GEN_GRID_H[1] or min_grid_w > GEN_GRID_W[1]:
        return None

    grid_h = random.randint(min_grid_h, GEN_GRID_H[1])
    grid_w = random.randint(min_grid_w, GEN_GRID_W[1])
    inp = np.full((grid_h, grid_w), BLACK, dtype=int)

    frame_r = random.randint(1, grid_h - outer_h - 1)
    frame_c = random.randint(1, grid_w - outer_w - 1)
    frame_box = (frame_r, frame_c, frame_r + outer_h - 1, frame_c + outer_w - 1)

    inp[frame_box[0], frame_box[1] : frame_box[3] + 1] = GREY
    inp[frame_box[2], frame_box[1] : frame_box[3] + 1] = GREY
    inp[frame_box[0] : frame_box[2] + 1, frame_box[1]] = GREY
    inp[frame_box[0] : frame_box[2] + 1, frame_box[3]] = GREY

    patches = [patch_a, patch_b]
    if not _place_external_pieces(inp, frame_box, patches):
        return None

    expected = _build_expected(inp.shape, frame_box, patches, intended)
    if not _instance_passes_checks(inp, expected, piece_masks, intended):
        return None
    return inp


def generate_input() -> np.ndarray:
    for _ in range(GEN_RETRIES):
        grid = _gen_attempt()
        if grid is not None:
            return grid
    raise RuntimeError("failed to generate a valid unique-assembly frame puzzle")
