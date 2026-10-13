from __future__ import annotations

import random

import numpy as np


BG = 0
FRAME_COLOR = 5
PALETTE = (1, 2, 3, 4, 6, 7, 8, 9)
QUADRANTS = ("ul", "ur", "ll", "lr")
NEIGHBORS4 = ((-1, 0), (1, 0), (0, -1), (0, 1))

SHAPE_RETRIES = 200
SCENE_RETRIES = 400
RECT_RETRIES = 300


def _bbox_from_coords(coords: np.ndarray) -> tuple[int, int, int, int]:
    return (
        int(coords[:, 0].min()),
        int(coords[:, 1].min()),
        int(coords[:, 0].max()),
        int(coords[:, 1].max()),
    )


def _crop_mask(mask: np.ndarray) -> np.ndarray:
    coords = np.argwhere(mask)
    assert coords.size > 0
    r0, c0, r1, c1 = _bbox_from_coords(coords)
    return mask[r0 : r1 + 1, c0 : c1 + 1].astype(bool)


def _is_connected(mask: np.ndarray) -> bool:
    coords = np.argwhere(mask)
    if coords.size == 0:
        return False
    h, w = mask.shape
    start = tuple(map(int, coords[0]))
    seen = {start}
    stack = [start]
    while stack:
        r, c = stack.pop()
        for dr, dc in NEIGHBORS4:
            nr, nc = r + dr, c + dc
            if 0 <= nr < h and 0 <= nc < w and mask[nr, nc] and (nr, nc) not in seen:
                seen.add((nr, nc))
                stack.append((nr, nc))
    return len(seen) == int(mask.sum())


def _canonical_signature(mask: np.ndarray) -> bytes:
    base = _crop_mask(mask)
    variants = []
    for rot in range(4):
        turned = np.rot90(base, rot)
        variants.append(turned.tobytes() + bytes(turned.shape))
        flipped = np.fliplr(turned)
        variants.append(flipped.tobytes() + bytes(flipped.shape))
    return min(variants)


def _transform_patch(patch: np.ndarray, quadrant: str) -> np.ndarray:
    if quadrant == "ul":
        return patch.copy()
    if quadrant == "ur":
        return np.fliplr(patch)
    if quadrant == "ll":
        return np.flipud(patch)
    if quadrant == "lr":
        return np.rot90(patch, 2)
    raise AssertionError(f"unknown quadrant: {quadrant}")


def _transform_mask(mask: np.ndarray, quadrant: str) -> np.ndarray:
    return _transform_patch(mask.astype(int), quadrant).astype(bool)


def _quadrant_frontier_coords(mask_or_patch: np.ndarray, quadrant: str) -> np.ndarray:
    occ = np.asarray(mask_or_patch) != BG
    h, w = occ.shape
    rr, cc = np.indices((h, w))
    if quadrant == "ul":
        edge = (rr == 0) | (cc == 0)
    elif quadrant == "ur":
        edge = (rr == 0) | (cc == w - 1)
    elif quadrant == "ll":
        edge = (rr == h - 1) | (cc == 0)
    elif quadrant == "lr":
        edge = (rr == h - 1) | (cc == w - 1)
    else:
        raise AssertionError(f"unknown quadrant: {quadrant}")
    coords = np.argwhere(occ & edge)
    assert coords.size > 0
    return coords.astype(int)


def _anchor_coord(coords: np.ndarray, quadrant: str) -> tuple[int, int]:
    rows = coords[:, 0]
    cols = coords[:, 1]
    if quadrant == "ul":
        row = int(rows.min())
        col = int(cols[rows == row].min())
        return row, col
    if quadrant == "ur":
        row = int(rows.min())
        col = int(cols[rows == row].max())
        return row, col
    if quadrant == "ll":
        row = int(rows.max())
        col = int(cols[rows == row].min())
        return row, col
    if quadrant == "lr":
        row = int(rows.max())
        col = int(cols[rows == row].max())
        return row, col
    raise AssertionError(f"unknown quadrant: {quadrant}")


def _cell_in_quadrant(cell: tuple[int, int], shape: tuple[int, int], quadrant: str) -> bool:
    r, c = cell
    h, w = shape
    top = r < (h / 2.0)
    left = c < (w / 2.0)
    if quadrant == "ul":
        return top and left
    if quadrant == "ur":
        return top and not left
    if quadrant == "ll":
        return (not top) and left
    if quadrant == "lr":
        return (not top) and (not left)
    raise AssertionError(f"unknown quadrant: {quadrant}")


def _seed_quadrant(seed_local: np.ndarray, inner_shape: tuple[int, int]) -> str:
    rows = seed_local[:, 0]
    cols = seed_local[:, 1]
    h, w = inner_shape
    top_all = bool(np.all(rows < (h / 2.0)))
    bottom_all = bool(np.all(rows >= (h / 2.0)))
    left_all = bool(np.all(cols < (w / 2.0)))
    right_all = bool(np.all(cols >= (w / 2.0)))

    if (top_all or bottom_all) and (left_all or right_all):
        top = top_all
        left = left_all
    else:
        top = float(rows.mean()) < (h / 2.0)
        left = float(cols.mean()) < (w / 2.0)

    if top and left:
        return "ul"
    if top and not left:
        return "ur"
    if (not top) and left:
        return "ll"
    return "lr"


def _boxes_overlap(
    a: tuple[int, int, int, int], b: tuple[int, int, int, int], pad: int = 0
) -> bool:
    ar0, ac0, ar1, ac1 = a
    br0, bc0, br1, bc1 = b
    ar0 -= pad
    ac0 -= pad
    ar1 += pad
    ac1 += pad
    br0 -= pad
    bc0 -= pad
    br1 += pad
    bc1 += pad
    return not (ar1 < br0 or br1 < ar0 or ac1 < bc0 or bc1 < ac0)


def _extract_monochrome_components(grid: np.ndarray, ignore: set[int]) -> list[dict]:
    h, w = grid.shape
    visited = np.zeros((h, w), dtype=bool)
    comps: list[dict] = []
    for r in range(h):
        for c in range(w):
            color = int(grid[r, c])
            if color in ignore or visited[r, c]:
                continue
            visited[r, c] = True
            stack = [(r, c)]
            coords: list[tuple[int, int]] = []
            while stack:
                rr, cc = stack.pop()
                coords.append((rr, cc))
                for dr, dc in NEIGHBORS4:
                    nr, nc = rr + dr, cc + dc
                    if (
                        0 <= nr < h
                        and 0 <= nc < w
                        and not visited[nr, nc]
                        and int(grid[nr, nc]) == color
                    ):
                        visited[nr, nc] = True
                        stack.append((nr, nc))
            arr = np.asarray(coords, dtype=int)
            comps.append(
                {
                    "color": color,
                    "coords": arr,
                    "bbox": _bbox_from_coords(arr),
                    "area": int(arr.shape[0]),
                }
            )
    return comps


def _component_patch(comp: dict) -> np.ndarray:
    color = int(comp["color"])
    r0, c0, r1, c1 = comp["bbox"]
    patch = np.zeros((r1 - r0 + 1, c1 - c0 + 1), dtype=int)
    coords = comp["coords"]
    patch[coords[:, 0] - r0, coords[:, 1] - c0] = color
    return patch


def _find_frame(grid: np.ndarray) -> tuple[int, tuple[int, int, int, int]]:
    candidates: list[tuple[int, int, tuple[int, int, int, int]]] = []
    for color in np.unique(grid):
        color = int(color)
        if color == BG:
            continue
        coords = np.argwhere(grid == color)
        r0, c0, r1, c1 = _bbox_from_coords(coords)
        h = r1 - r0 + 1
        w = c1 - c0 + 1
        if h != w or h < 5:
            continue
        sub = grid[r0 : r1 + 1, c0 : c1 + 1] == color
        ring = np.zeros_like(sub, dtype=bool)
        ring[0, :] = True
        ring[-1, :] = True
        ring[:, 0] = True
        ring[:, -1] = True
        if np.array_equal(sub, ring):
            candidates.append((h * w, color, (r0, c0, r1, c1)))
    assert len(candidates) == 1
    _area, color, bbox = candidates[0]
    return color, bbox


def main(grid: np.ndarray) -> np.ndarray:
    """Find the color-matched outer template, orient it by the seed quadrant, and stamp it into the frame.

    Input structure:
    - The background is black.
    - One grey square border encloses an empty inner square; inside it sits one color's seed pixel set in a single quadrant.
    - Outside the frame are several separated monochrome template sprites, each color occupying exactly one outer component.

    Rule:
    - Match the seed color to the outer template of the same color.
    - Determine the seed quadrant inside the frame: upper-left keeps the template, upper-right mirrors it horizontally, lower-left mirrors it vertically, and lower-right rotates it by 180 degrees.
    - Use the seed pixels as the oriented template's corner-frontier anchor and paint the full oriented template into the frame while leaving the border and outer templates unchanged.

    Key insight:
    - The seed color chooses which template to use, while the seed position independently chooses which reflection or rotation of that template must be inserted.
    """
    g = np.asarray(grid, dtype=int)
    assert g.ndim == 2

    frame_color, (fr0, fc0, fr1, fc1) = _find_frame(g)
    inner = g[fr0 + 1 : fr1, fc0 + 1 : fc1]
    seed_mask = (inner != BG) & (inner != frame_color)
    seed_local = np.argwhere(seed_mask)
    assert seed_local.size > 0
    seed_colors = {int(x) for x in inner[seed_mask].ravel()}
    assert len(seed_colors) == 1
    seed_color = seed_colors.pop()
    quadrant = _seed_quadrant(seed_local, inner.shape)
    seed_global = seed_local + np.array([fr0 + 1, fc0 + 1], dtype=int)

    comps = _extract_monochrome_components(g, ignore={BG, frame_color})
    template_candidates = [
        comp
        for comp in comps
        if int(comp["color"]) == seed_color and not _boxes_overlap(comp["bbox"], (fr0, fc0, fr1, fc1))
    ]
    assert len(template_candidates) == 1
    template = _component_patch(template_candidates[0])
    transformed = _transform_patch(template, quadrant)

    template_frontier = _quadrant_frontier_coords(transformed, quadrant)
    template_anchor = _anchor_coord(template_frontier, quadrant)
    seed_anchor = _anchor_coord(seed_global, quadrant)
    top = int(seed_anchor[0] - template_anchor[0])
    left = int(seed_anchor[1] - template_anchor[1])

    shifted_frontier = {
        (int(r + top), int(c + left)) for r, c in template_frontier.tolist()
    }
    observed_seeds = {tuple(map(int, rc)) for rc in seed_global.tolist()}
    assert observed_seeds <= shifted_frontier

    out = g.copy()
    paint = np.argwhere(transformed != BG)
    for rr, cc in paint:
        gr = int(top + rr)
        gc = int(left + cc)
        assert fr0 < gr < fr1 and fc0 < gc < fc1
        out[gr, gc] = seed_color
    return out


def _random_shape_mask() -> np.ndarray | None:
    for _ in range(SHAPE_RETRIES):
        h = random.randint(3, 5)
        w = random.randint(3, 5)
        fill_p = random.uniform(0.35, 0.7)
        mask = np.zeros((h, w), dtype=bool)
        for r in range(h):
            for c in range(w):
                if random.random() < fill_p:
                    mask[r, c] = True
        if int(mask.sum()) < max(4, (h + w) // 2):
            continue
        if int(mask.sum()) >= h * w:
            continue
        if not np.any(mask[0, :]) or not np.any(mask[-1, :]):
            continue
        if not np.any(mask[:, 0]) or not np.any(mask[:, -1]):
            continue
        if not _is_connected(mask):
            continue
        variants = {
            _transform_mask(mask, "ul").tobytes() + bytes(_transform_mask(mask, "ul").shape),
            _transform_mask(mask, "ur").tobytes() + bytes(_transform_mask(mask, "ur").shape),
            _transform_mask(mask, "ll").tobytes() + bytes(_transform_mask(mask, "ll").shape),
            _transform_mask(mask, "lr").tobytes() + bytes(_transform_mask(mask, "lr").shape),
        }
        if len(variants) != 4:
            continue
        return mask
    return None


def _choose_frame_and_seed(
    mask: np.ndarray, quadrant: str
) -> tuple[int, tuple[int, int], np.ndarray, np.ndarray] | None:
    oriented = _transform_mask(mask, quadrant)
    ph, pw = oriented.shape
    frontier = _quadrant_frontier_coords(oriented, quadrant)
    anchor = _anchor_coord(frontier, quadrant)

    frame_sides = list(range(max(8, max(ph, pw) + 4), 16))
    random.shuffle(frame_sides)
    for side in frame_sides:
        inner = side - 2
        if ph > inner or pw > inner:
            continue
        placements = []
        for top in range(inner - ph + 1):
            for left in range(inner - pw + 1):
                anchor_cell = (top + anchor[0], left + anchor[1])
                if not _cell_in_quadrant(anchor_cell, (inner, inner), quadrant):
                    continue
                eligible = []
                for rr, cc in frontier.tolist():
                    cell = (int(top + rr), int(left + cc))
                    if _cell_in_quadrant(cell, (inner, inner), quadrant):
                        eligible.append(cell)
                if not eligible:
                    continue
                placements.append((top, left, eligible))
        if not placements:
            continue

        multi = [p for p in placements if len(p[2]) >= 2]
        if multi and random.random() < 0.5:
            top, left, eligible = random.choice(multi)
        else:
            top, left, eligible = random.choice(placements)

        anchor_cell = (top + anchor[0], left + anchor[1])
        extras = [cell for cell in eligible if cell != anchor_cell]
        if extras:
            extra_count = random.randint(0, len(extras))
            chosen = [anchor_cell] + random.sample(extras, extra_count)
        else:
            chosen = [anchor_cell]
        chosen.sort()
        return side, (top, left), np.asarray(chosen, dtype=int), oriented
    return None


def _rand_coord(limit: int, size: int) -> int:
    slack = limit - size
    if slack <= 0:
        return 0
    low = 1 if slack >= 2 else 0
    high = slack - 1 if slack >= 2 else slack
    return random.randint(low, high)


def _place_rect(
    canvas_shape: tuple[int, int],
    rect_shape: tuple[int, int],
    reserved: list[tuple[int, int, int, int]],
    pad: int = 1,
) -> tuple[int, int] | None:
    h, w = canvas_shape
    rh, rw = rect_shape
    for _ in range(RECT_RETRIES):
        top = _rand_coord(h, rh)
        left = _rand_coord(w, rw)
        bbox = (top, left, top + rh - 1, left + rw - 1)
        if any(_boxes_overlap(bbox, other, pad=pad) for other in reserved):
            continue
        return top, left
    return None


def _render_patch(canvas: np.ndarray, patch: np.ndarray, top: int, left: int) -> None:
    mask = patch != BG
    h, w = patch.shape
    region = canvas[top : top + h, left : left + w]
    region[mask] = patch[mask]


def _build_expected_output(
    grid: np.ndarray,
    frame_bbox: tuple[int, int, int, int],
    oriented_patch: np.ndarray,
    inner_top_left: tuple[int, int],
) -> np.ndarray:
    out = grid.copy()
    fr0, fc0, _fr1, _fc1 = frame_bbox
    inner_r0 = fr0 + 1 + inner_top_left[0]
    inner_c0 = fc0 + 1 + inner_top_left[1]
    _render_patch(out, oriented_patch, inner_r0, inner_c0)
    return out


def _generate_attempt() -> np.ndarray | None:
    template_count = random.randint(3, 5)
    colors = random.sample(PALETTE, template_count)
    quadrant = random.choice(QUADRANTS)
    chosen_index = random.randrange(template_count)

    masks: list[np.ndarray] = []
    signatures: set[bytes] = set()
    for _ in range(template_count):
        for _ in range(SHAPE_RETRIES):
            mask = _random_shape_mask()
            if mask is None:
                continue
            sig = _canonical_signature(mask)
            if sig in signatures:
                continue
            signatures.add(sig)
            masks.append(mask)
            break
        else:
            return None

    chosen_mask = masks[chosen_index]
    frame_seed = _choose_frame_and_seed(chosen_mask, quadrant)
    if frame_seed is None:
        return None
    frame_side, inner_top_left, seed_local_coords, oriented_patch = frame_seed

    rects = [(frame_side, frame_side)] + [mask.shape for mask in masks]
    min_side = max(
        18,
        frame_side + 4,
        max(h for h, _ in rects) + 4,
        max(w for _, w in rects) + 4,
        int(np.ceil(np.sqrt(frame_side * frame_side + sum(h * w for h, w in rects) + 60))),
    )
    if min_side > 30:
        return None
    height = random.randint(min_side, 30)
    width = random.randint(min_side, 30)

    reserved: list[tuple[int, int, int, int]] = []
    frame_pos = _place_rect((height, width), (frame_side, frame_side), reserved, pad=1)
    if frame_pos is None:
        return None
    frame_top, frame_left = frame_pos
    frame_bbox = (frame_top, frame_left, frame_top + frame_side - 1, frame_left + frame_side - 1)
    reserved.append(frame_bbox)

    template_positions: list[tuple[int, int]] = [(-1, -1)] * template_count
    order = list(range(template_count))
    order.sort(key=lambda idx: int(masks[idx].sum()), reverse=True)
    for idx in order:
        pos = _place_rect((height, width), masks[idx].shape, reserved, pad=1)
        if pos is None:
            return None
        top, left = pos
        template_positions[idx] = (top, left)
        h, w = masks[idx].shape
        reserved.append((top, left, top + h - 1, left + w - 1))

    grid = np.zeros((height, width), dtype=int)
    grid[frame_top, frame_left : frame_left + frame_side] = FRAME_COLOR
    grid[frame_top + frame_side - 1, frame_left : frame_left + frame_side] = FRAME_COLOR
    grid[frame_top : frame_top + frame_side, frame_left] = FRAME_COLOR
    grid[frame_top : frame_top + frame_side, frame_left + frame_side - 1] = FRAME_COLOR

    for idx, mask in enumerate(masks):
        color = colors[idx]
        patch = np.zeros(mask.shape, dtype=int)
        patch[mask] = color
        top, left = template_positions[idx]
        _render_patch(grid, patch, top, left)

    seed_color = colors[chosen_index]
    inner_origin = np.array([frame_top + 1, frame_left + 1], dtype=int)
    for rr, cc in seed_local_coords.tolist():
        gr, gc = (inner_origin + np.array([rr, cc], dtype=int)).tolist()
        grid[int(gr), int(gc)] = seed_color

    expected = _build_expected_output(grid, frame_bbox, oriented_patch.astype(int) * seed_color, inner_top_left)
    try:
        actual = main(grid)
    except Exception:
        return None
    if not np.array_equal(actual, expected):
        return None
    return grid


def generate_input() -> np.ndarray:
    for _ in range(SCENE_RETRIES):
        grid = _generate_attempt()
        if grid is not None:
            return grid
    raise RuntimeError("failed to generate a valid input")
