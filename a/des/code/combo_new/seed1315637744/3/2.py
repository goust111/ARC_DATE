import random

import numpy as np


BG = 0
GENERATION_ATTEMPTS = 400
FRAME_RETRIES = 120
BLOCK_RETRIES = 220
COLOR_POOL = tuple(range(1, 10))
BLOCK_SHAPES = (
    (1, 1),
    (1, 2),
    (2, 1),
    (1, 3),
    (3, 1),
    (2, 2),
    (2, 3),
    (3, 2),
)


def _components(mask: np.ndarray) -> list[np.ndarray]:
    h, w = mask.shape
    seen = np.zeros_like(mask, dtype=bool)
    comps = []
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
                for dr, dc in ((-1, 0), (1, 0), (0, -1), (0, 1)):
                    nr, nc = rr + dr, cc + dc
                    if 0 <= nr < h and 0 <= nc < w and mask[nr, nc] and not seen[nr, nc]:
                        seen[nr, nc] = True
                        stack.append((nr, nc))
            comps.append(np.array(coords, dtype=int))
    return comps


def _bbox(coords: np.ndarray) -> tuple[int, int, int, int]:
    return (
        int(coords[:, 0].min()),
        int(coords[:, 1].min()),
        int(coords[:, 0].max()),
        int(coords[:, 1].max()),
    )


def _shift_bbox(bbox: tuple[int, int, int, int], dr: int, dc: int) -> tuple[int, int, int, int]:
    r0, c0, r1, c1 = bbox
    return r0 + dr, c0 + dc, r1 + dr, c1 + dc


def _shape_from_bbox(bbox: tuple[int, int, int, int]) -> tuple[int, int]:
    r0, c0, r1, c1 = bbox
    return r1 - r0 + 1, c1 - c0 + 1


def _strictly_contains(outer: tuple[int, int, int, int], inner: tuple[int, int, int, int]) -> bool:
    or0, oc0, or1, oc1 = outer
    ir0, ic0, ir1, ic1 = inner
    return or0 < ir0 and oc0 < ic0 and ir1 < or1 and ic1 < oc1


def _read_palette(row: np.ndarray) -> list[int]:
    colors = []
    c = 0
    while c < row.shape[0]:
        if int(row[c]) == BG:
            c += 1
            continue
        color = int(row[c])
        start = c
        while c < row.shape[0] and int(row[c]) == color:
            c += 1
        assert c - start == 2, "palette entries must be 2-pixel runs"
        colors.append(color)
        if c < row.shape[0]:
            assert int(row[c]) == BG, "palette entries must be separated by background"
    assert colors, "palette must contain at least one color"
    assert len(colors) == len(set(colors)), "palette colors must be unique"
    return colors


def _body_color(body: np.ndarray) -> int:
    colors = [int(v) for v in np.unique(body) if int(v) != BG]
    assert len(colors) == 1, "body must contain exactly one foreground color"
    return colors[0]


def _is_frame(coords: np.ndarray, bbox: tuple[int, int, int, int]) -> bool:
    r0, c0, r1, c1 = bbox
    h, w = _shape_from_bbox(bbox)
    if h < 3 or w < 3:
        return False
    if coords.shape[0] != 2 * (h + w) - 4:
        return False
    for r, c in coords:
        if r not in (r0, r1) and c not in (c0, c1):
            return False
    return True


def _is_small_block(coords: np.ndarray, bbox: tuple[int, int, int, int]) -> bool:
    h, w = _shape_from_bbox(bbox)
    return coords.shape[0] == h * w and h * w <= 6


def _extract_objects(body: np.ndarray) -> tuple[int, list[dict]]:
    fg = _body_color(body)
    objects = []
    for coords in _components(body == fg):
        bbox = _bbox(coords)
        if _is_frame(coords, bbox):
            kind = "frame"
        elif _is_small_block(coords, bbox):
            kind = "block"
        else:
            raise AssertionError("unexpected component shape")
        objects.append({"coords": coords, "bbox": bbox, "kind": kind})
    return fg, objects


def _depth_for_bbox(bbox: tuple[int, int, int, int], frame_bboxes: list[tuple[int, int, int, int]]) -> int:
    return sum(1 for frame_bbox in frame_bboxes if _strictly_contains(frame_bbox, bbox))


def _draw_frame(grid: np.ndarray, bbox: tuple[int, int, int, int], color: int) -> None:
    r0, c0, r1, c1 = bbox
    grid[r0, c0 : c1 + 1] = color
    grid[r1, c0 : c1 + 1] = color
    grid[r0 : r1 + 1, c0] = color
    grid[r0 : r1 + 1, c1] = color


def _draw_block(grid: np.ndarray, bbox: tuple[int, int, int, int], color: int) -> None:
    r0, c0, r1, c1 = bbox
    grid[r0 : r1 + 1, c0 : c1 + 1] = color


def _mark_frame(mask: np.ndarray, bbox: tuple[int, int, int, int]) -> None:
    r0, c0, r1, c1 = bbox
    mask[r0, c0 : c1 + 1] = True
    mask[r1, c0 : c1 + 1] = True
    mask[r0 : r1 + 1, c0] = True
    mask[r0 : r1 + 1, c1] = True


def _mark_block(mask: np.ndarray, bbox: tuple[int, int, int, int]) -> None:
    r0, c0, r1, c1 = bbox
    mask[r0 : r1 + 1, c0 : c1 + 1] = True


def _occupied_near_bbox(mask: np.ndarray, bbox: tuple[int, int, int, int], pad: int = 1) -> bool:
    h, w = mask.shape
    r0, c0, r1, c1 = bbox
    rr0 = max(0, r0 - pad)
    cc0 = max(0, c0 - pad)
    rr1 = min(h - 1, r1 + pad)
    cc1 = min(w - 1, c1 + pad)
    return bool(mask[rr0 : rr1 + 1, cc0 : cc1 + 1].any())


def _build_chain(
    rng: random.Random, depth: int, max_outer_h: int, max_outer_w: int
) -> dict | None:
    min_outer_h = 4 * depth + 1
    min_outer_w = 4 * depth + 1
    if max_outer_h < min_outer_h or max_outer_w < min_outer_w:
        return None

    outer_h = min_outer_h + rng.randint(0, min(4, max_outer_h - min_outer_h))
    outer_w = min_outer_w + rng.randint(0, min(6, max_outer_w - min_outer_w))

    frames = []
    current = (0, 0, outer_h - 1, outer_w - 1)
    for layer in range(depth):
        frames.append(current)
        if layer == depth - 1:
            break
        remaining_frames = depth - layer - 1
        cur_h, cur_w = _shape_from_bbox(current)
        child_min_h = 4 * remaining_frames + 1
        child_min_w = 4 * remaining_frames + 1
        child_h_max = cur_h - 4
        child_w_max = cur_w - 4
        if child_h_max < child_min_h or child_w_max < child_min_w:
            return None
        child_h = child_min_h + rng.randint(0, min(4, child_h_max - child_min_h))
        child_w = child_min_w + rng.randint(0, min(5, child_w_max - child_min_w))
        top_margin_max = cur_h - child_h - 2
        left_margin_max = cur_w - child_w - 2
        if top_margin_max < 2 or left_margin_max < 2:
            return None
        child_r0 = current[0] + rng.randint(2, top_margin_max)
        child_c0 = current[1] + rng.randint(2, left_margin_max)
        current = (child_r0, child_c0, child_r0 + child_h - 1, child_c0 + child_w - 1)

    innermost = frames[-1]
    inner_h, inner_w = _shape_from_bbox(innermost)
    fits = [(bh, bw) for bh, bw in BLOCK_SHAPES if bh <= inner_h - 4 and bw <= inner_w - 4]
    if not fits:
        return None
    block_h, block_w = rng.choice(fits)
    block_top_offset_max = inner_h - block_h - 2
    block_left_offset_max = inner_w - block_w - 2
    if block_top_offset_max < 2 or block_left_offset_max < 2:
        return None
    block_r0 = innermost[0] + rng.randint(2, block_top_offset_max)
    block_c0 = innermost[1] + rng.randint(2, block_left_offset_max)
    block = (block_r0, block_c0, block_r0 + block_h - 1, block_c0 + block_w - 1)
    return {"frames": frames, "block": block}


def _place_chain(
    grid: np.ndarray,
    occupied: np.ndarray,
    chain: dict,
    top: int,
    left: int,
    color: int,
) -> tuple[list[tuple[int, int, int, int]], tuple[int, int, int, int]]:
    abs_frames = [_shift_bbox(bbox, top, left) for bbox in chain["frames"]]
    abs_block = _shift_bbox(chain["block"], top, left)
    for bbox in abs_frames:
        _draw_frame(grid, bbox, color)
        _mark_frame(occupied, bbox)
    _draw_block(grid, abs_block, color)
    _mark_block(occupied, abs_block)
    return abs_frames, abs_block


def _choose_palette_and_fg(rng: random.Random, palette_len: int) -> tuple[list[int], int]:
    colors = list(COLOR_POOL)
    rng.shuffle(colors)
    palette = colors[:palette_len]
    fg = colors[palette_len]
    return palette, fg


def _place_extra_chain(
    rng: random.Random,
    grid: np.ndarray,
    occupied: np.ndarray,
    frame_bboxes: list[tuple[int, int, int, int]],
    fg: int,
) -> None:
    body_h, body_w = grid.shape
    max_depth = min(3, max(1, (min(body_h, body_w) - 1) // 4))
    depth = rng.randint(1, max_depth)
    max_outer_h = min(body_h - 2, 4 * depth + 7)
    max_outer_w = min(body_w - 2, 4 * depth + 9)
    chain = _build_chain(rng, depth, max_outer_h, max_outer_w)
    if chain is None:
        return

    outer_h, outer_w = _shape_from_bbox(chain["frames"][0])
    if body_h - outer_h < 2 or body_w - outer_w < 2:
        return

    for _ in range(FRAME_RETRIES):
        top = rng.randint(1, body_h - outer_h - 1)
        left = rng.randint(1, body_w - outer_w - 1)
        outer_bbox = _shift_bbox(chain["frames"][0], top, left)
        if _depth_for_bbox(outer_bbox, frame_bboxes) != 0:
            continue
        if _occupied_near_bbox(occupied, outer_bbox):
            continue
        new_frames, _ = _place_chain(grid, occupied, chain, top, left, fg)
        frame_bboxes.extend(new_frames)
        return


def _place_block_at_depth(
    rng: random.Random,
    grid: np.ndarray,
    occupied: np.ndarray,
    frame_bboxes: list[tuple[int, int, int, int]],
    fg: int,
    target_depth: int,
) -> bool:
    body_h, body_w = grid.shape
    candidates = [shape for shape in BLOCK_SHAPES if shape[0] <= body_h and shape[1] <= body_w]
    if not candidates:
        return False
    for _ in range(BLOCK_RETRIES):
        block_h, block_w = rng.choice(candidates)
        r0 = rng.randint(0, body_h - block_h)
        c0 = rng.randint(0, body_w - block_w)
        bbox = (r0, c0, r0 + block_h - 1, c0 + block_w - 1)
        if _depth_for_bbox(bbox, frame_bboxes) != target_depth:
            continue
        if _occupied_near_bbox(occupied, bbox):
            continue
        _draw_block(grid, bbox, fg)
        _mark_block(occupied, bbox)
        return True
    return False


def _make_grid(rng: random.Random) -> np.ndarray | None:
    max_depth = rng.randint(1, 4)
    palette_len = max_depth + 1
    palette, fg = _choose_palette_and_fg(rng, palette_len)

    body_h_min = max(10, 4 * max_depth + 4)
    body_w_min = max(12, 4 * max_depth + 6, 3 * palette_len + 1)
    body_h = rng.randint(body_h_min, min(29, body_h_min + 6))
    body_w = rng.randint(body_w_min, min(30, body_w_min + 8))

    grid = np.full((body_h + 1, body_w), BG, dtype=int)
    occupied = np.zeros((body_h, body_w), dtype=bool)

    main_chain = _build_chain(rng, max_depth, body_h - 2, body_w - 2)
    if main_chain is None:
        return None
    outer_h, outer_w = _shape_from_bbox(main_chain["frames"][0])
    if body_h - outer_h < 2 or body_w - outer_w < 2:
        return None

    top = rng.randint(1, body_h - outer_h - 1)
    left = rng.randint(1, body_w - outer_w - 1)
    frame_bboxes, _ = _place_chain(grid[:-1], occupied, main_chain, top, left, fg)

    extra_chain_count = rng.randint(0, 2)
    for _ in range(extra_chain_count):
        _place_extra_chain(rng, grid[:-1], occupied, frame_bboxes, fg)

    depth_targets = [0, max_depth]
    for _ in range(rng.randint(1, 4)):
        depth_targets.append(rng.randint(0, max_depth))
    rng.shuffle(depth_targets)
    for target_depth in depth_targets:
        _place_block_at_depth(rng, grid[:-1], occupied, frame_bboxes, fg, target_depth)

    palette_row = np.full(body_w, BG, dtype=int)
    palette_width = 3 * palette_len - 1
    start = rng.randint(0, body_w - palette_width)
    cursor = start
    for i, color in enumerate(palette):
        palette_row[cursor : cursor + 2] = color
        cursor += 2
        if i != palette_len - 1:
            cursor += 1
    grid[-1] = palette_row

    return grid


def _validate_generated(grid: np.ndarray) -> None:
    assert isinstance(grid, np.ndarray)
    assert grid.ndim == 2
    assert max(grid.shape) <= 30
    palette = _read_palette(grid[-1])
    body = grid[:-1]
    _fg, objects = _extract_objects(body)
    assert any(obj["kind"] == "frame" for obj in objects)
    assert any(obj["kind"] == "block" for obj in objects)
    frame_bboxes = [obj["bbox"] for obj in objects if obj["kind"] == "frame"]
    depths = [_depth_for_bbox(obj["bbox"], frame_bboxes) for obj in objects]
    assert max(depths) < len(palette)
    _ = main(grid)


def main(grid: np.ndarray) -> np.ndarray:
    """
    Recolor nested monochrome frames and small blocks according to the bottom palette order.

    Input structure:
    - Black background with a palette row on the bottom edge.
    - The palette row contains several distinct 2-pixel color blocks ordered left to right, with 1-pixel black gaps between them.
    - Above the palette, all foreground objects share one color and consist of 1-pixel-thick closed rectangular frames plus small filled rectangles.

    Rule:
    - Treat each frame or filled block as one object.
    - The depth of an object is the number of rectangular frames that strictly contain its bounding box.
    - Recolor every object with the palette color at the matching depth index, while leaving the palette row unchanged.

    Key insight:
    - The input color alone is uninformative; the output color comes from containment depth, so the task is to recover the frame nesting tree before recoloring.
    """
    g = np.asarray(grid, dtype=int)
    out = g.copy()
    palette = _read_palette(g[-1])
    body = g[:-1]
    fg, objects = _extract_objects(body)

    frame_bboxes = [obj["bbox"] for obj in objects if obj["kind"] == "frame"]
    for obj in objects:
        depth = _depth_for_bbox(obj["bbox"], frame_bboxes)
        assert depth < len(palette), "palette must cover all object depths"
        color = palette[depth]
        for r, c in obj["coords"]:
            out[r, c] = color

    assert fg != BG
    return out


def generate_input() -> np.ndarray:
    rng = random.Random()
    for _ in range(GENERATION_ATTEMPTS):
        grid = _make_grid(rng)
        if grid is None:
            continue
        try:
            _validate_generated(grid)
        except AssertionError:
            continue
        return grid
    raise RuntimeError(f"generate_input failed after {GENERATION_ATTEMPTS} attempts")
