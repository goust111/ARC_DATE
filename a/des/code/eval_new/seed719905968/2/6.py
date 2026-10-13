from collections import deque
import numpy as np


BG = 0
FENCE_COLOR = 5
MAX_GENERATION_ATTEMPTS = 500
CANVAS_SIZE = 30
SPRITE_PADDING = 1

SPRITE_SHAPES = [
    np.array([[1]], dtype=int),
    np.array([[1, 1]], dtype=int),
    np.array([[1], [1]], dtype=int),
    np.array([[1, 1], [1, 1]], dtype=int),
    np.array([[1, 1], [0, 1]], dtype=int),
    np.array([[1, 1], [1, 0]], dtype=int),
    np.array([[1, 0], [1, 1]], dtype=int),
    np.array([[0, 1], [1, 1]], dtype=int),
    np.array([[1, 1, 1]], dtype=int),
    np.array([[1], [1], [1]], dtype=int),
]


def _bbox(coords: np.ndarray) -> tuple[int, int, int, int]:
    r0, c0 = coords.min(axis=0)
    r1, c1 = coords.max(axis=0)
    return int(r0), int(c0), int(r1), int(c1)


def _bbox_h(bbox: tuple[int, int, int, int]) -> int:
    return int(bbox[2] - bbox[0] + 1)


def _bbox_w(bbox: tuple[int, int, int, int]) -> int:
    return int(bbox[3] - bbox[1] + 1)


def _find_color_components(grid: np.ndarray, bg: int = BG) -> list[dict]:
    h, w = grid.shape
    seen = np.zeros((h, w), dtype=bool)
    components: list[dict] = []
    comp_id = 0
    for r in range(h):
        for c in range(w):
            color = int(grid[r, c])
            if color == bg or seen[r, c]:
                continue
            q = deque([(r, c)])
            seen[r, c] = True
            coords: list[tuple[int, int]] = []
            while q:
                rr, cc = q.popleft()
                coords.append((rr, cc))
                for dr, dc in ((1, 0), (-1, 0), (0, 1), (0, -1)):
                    nr = rr + dr
                    nc = cc + dc
                    if not (0 <= nr < h and 0 <= nc < w):
                        continue
                    if seen[nr, nc] or int(grid[nr, nc]) != color:
                        continue
                    seen[nr, nc] = True
                    q.append((nr, nc))
            arr = np.array(coords, dtype=int)
            components.append(
                {
                    "id": comp_id,
                    "color": color,
                    "coords": arr,
                    "bbox": _bbox(arr),
                    "area": int(arr.shape[0]),
                }
            )
            comp_id += 1
    return components


def _is_rectangular_frame(grid: np.ndarray, comp: dict) -> bool:
    r0, c0, r1, c1 = comp["bbox"]
    if _bbox_h(comp["bbox"]) < 3 or _bbox_w(comp["bbox"]) < 3:
        return False
    color = int(comp["color"])
    patch = grid[r0 : r1 + 1, c0 : c1 + 1]
    border = np.zeros_like(patch, dtype=bool)
    border[0, :] = True
    border[-1, :] = True
    border[:, 0] = True
    border[:, -1] = True
    if not np.all(patch[border] == color):
        return False
    if np.any(patch[~border] == color):
        return False
    return int(comp["area"]) == int(border.sum())


def _inside_bbox(inner: tuple[int, int, int, int], outer: tuple[int, int, int, int]) -> bool:
    return (
        int(outer[0]) <= int(inner[0]) <= int(inner[2]) <= int(outer[2])
        and int(outer[1]) <= int(inner[1]) <= int(inner[3]) <= int(outer[3])
    )


def _overlap_1d(a0: int, a1: int, b0: int, b1: int) -> bool:
    return max(int(a0), int(b0)) <= min(int(a1), int(b1))


def _adjacent_side(
    comp_bbox: tuple[int, int, int, int], frame_bbox: tuple[int, int, int, int]
) -> str | None:
    r0, c0, r1, c1 = frame_bbox
    cr0, cc0, cr1, cc1 = comp_bbox
    sides: list[str] = []
    if cr1 == r0 - 1 and _overlap_1d(cc0, cc1, c0, c1):
        sides.append("top")
    if cr0 == r1 + 1 and _overlap_1d(cc0, cc1, c0, c1):
        sides.append("bottom")
    if cc1 == c0 - 1 and _overlap_1d(cr0, cr1, r0, r1):
        sides.append("left")
    if cc0 == c1 + 1 and _overlap_1d(cr0, cr1, r0, r1):
        sides.append("right")
    if len(sides) == 1:
        return sides[0]
    return None


def _parse_inner_blocks(grid: np.ndarray, frame_bbox: tuple[int, int, int, int], comps: list[dict]) -> dict:
    r0, c0, r1, c1 = frame_bbox
    inner_bbox = (r0 + 1, c0 + 1, r1 - 1, c1 - 1)
    blocks = [comp for comp in comps if _inside_bbox(comp["bbox"], inner_bbox)]
    assert 2 <= len(blocks) <= 4

    colors = {int(block["color"]) for block in blocks}
    assert len(colors) == 1
    block_color = int(next(iter(colors)))

    sizes = set()
    mask = np.zeros_like(grid, dtype=bool)
    for block in blocks:
        br0, bc0, br1, bc1 = block["bbox"]
        bh = _bbox_h(block["bbox"])
        bw = _bbox_w(block["bbox"])
        assert bh == bw
        patch = grid[br0 : br1 + 1, bc0 : bc1 + 1]
        assert np.all(patch == int(block["color"]))
        sizes.add(bh)
        mask[block["coords"][:, 0], block["coords"][:, 1]] = True
    assert len(sizes) == 1
    size = int(next(iter(sizes)))

    inner = grid[inner_bbox[0] : inner_bbox[2] + 1, inner_bbox[1] : inner_bbox[3] + 1]
    inner_mask = mask[inner_bbox[0] : inner_bbox[2] + 1, inner_bbox[1] : inner_bbox[3] + 1]
    assert not np.any((inner != BG) & ~inner_mask)

    row_spans = {(int(block["bbox"][0]), int(block["bbox"][2])) for block in blocks}
    col_spans = {(int(block["bbox"][1]), int(block["bbox"][3])) for block in blocks}

    if len(row_spans) == 1 and len(col_spans) == len(blocks):
        orientation = "horizontal"
        blocks = sorted(blocks, key=lambda block: int(block["bbox"][1]))
        starts = [int(block["bbox"][1]) for block in blocks]
    elif len(col_spans) == 1 and len(row_spans) == len(blocks):
        orientation = "vertical"
        blocks = sorted(blocks, key=lambda block: int(block["bbox"][0]))
        starts = [int(block["bbox"][0]) for block in blocks]
    else:
        raise AssertionError("inner blocks must align on one axis")

    step = int(size + 1)
    for a, b in zip(starts, starts[1:]):
        assert int(b - a) == step

    return {
        "blocks": blocks,
        "block_color": block_color,
        "count": len(blocks),
        "orientation": orientation,
        "size": size,
    }


def _infer_copy_sign(orientation: str, side: str, sprite_bbox: tuple[int, int, int, int], blocks: list[dict]) -> int:
    if orientation == "horizontal":
        if side == "left":
            return -1
        if side == "right":
            return 1
        sprite_center2 = int(sprite_bbox[1] + sprite_bbox[3])
        left_center2 = int(blocks[0]["bbox"][1] + blocks[0]["bbox"][3])
        right_center2 = int(blocks[-1]["bbox"][1] + blocks[-1]["bbox"][3])
        return 1 if abs(sprite_center2 - left_center2) <= abs(sprite_center2 - right_center2) else -1

    if side == "top":
        return -1
    if side == "bottom":
        return 1
    sprite_center2 = int(sprite_bbox[0] + sprite_bbox[2])
    top_center2 = int(blocks[0]["bbox"][0] + blocks[0]["bbox"][2])
    bottom_center2 = int(blocks[-1]["bbox"][0] + blocks[-1]["bbox"][2])
    return 1 if abs(sprite_center2 - top_center2) <= abs(sprite_center2 - bottom_center2) else -1


def _paint_shifted_component(
    out: np.ndarray, coords: np.ndarray, dr: int, dc: int, color: int
) -> None:
    rows = coords[:, 0] + int(dr)
    cols = coords[:, 1] + int(dc)
    out[rows, cols] = int(color)


def main(grid: np.ndarray) -> np.ndarray:
    """
    Copy each exterior sprite along the axis encoded by the blocks inside its fence.

    Input structure:
    - The background is black.
    - Several hollow rectangular fences appear as one-pixel-thick monochrome frames.
    - Inside each fence are 2-4 equal solid squares of one color, spaced by one black
      cell and aligned either horizontally or vertically.
    - Exactly one small sprite sits flush against the outside of each fence on one side.

    Rule:
    - For every fence, detect the inner squares' orientation, count N, and color.
    - Find the attached exterior sprite and its anchor side.
    - Starting from the original sprite, place N total copies along the inner-square axis
      with a stride of the sprite's own width or height plus one cell.
    - Keep the original sprite color unchanged, but recolor every added copy to the
      inner-square color. Leave the fence and inner squares unchanged.

    Key insight:
    - The interior squares encode an axis and repetition count, while the attached sprite
      provides the start point whose side and endpoint position resolve where the exterior
      repetition should extend.
    """
    g = np.array(grid, dtype=int)
    components = _find_color_components(g, bg=BG)
    frames = [comp for comp in components if _is_rectangular_frame(g, comp)]
    frames.sort(key=lambda comp: comp["bbox"])
    frame_ids = {int(frame["id"]) for frame in frames}

    inner_owner: dict[int, int | None] = {}
    for comp in components:
        owner = None
        for idx, frame in enumerate(frames):
            fr0, fc0, fr1, fc1 = frame["bbox"]
            inner_bbox = (fr0 + 1, fc0 + 1, fr1 - 1, fc1 - 1)
            if _inside_bbox(comp["bbox"], inner_bbox):
                owner = idx
                break
        inner_owner[int(comp["id"])] = owner

    out = g.copy()
    for idx, frame in enumerate(frames):
        frame_bbox = frame["bbox"]
        inner_comps = [comp for comp in components if inner_owner[int(comp["id"])] == idx]
        info = _parse_inner_blocks(g, frame_bbox, inner_comps)
        blocks = info["blocks"]
        block_color = int(info["block_color"])
        count = int(info["count"])
        orientation = str(info["orientation"])

        sprite_candidates: list[tuple[dict, str]] = []
        for comp in components:
            comp_id = int(comp["id"])
            if comp_id in frame_ids:
                continue
            if inner_owner[comp_id] is not None:
                continue
            side = _adjacent_side(comp["bbox"], frame_bbox)
            if side is not None:
                sprite_candidates.append((comp, side))
        assert len(sprite_candidates) == 1
        sprite, side = sprite_candidates[0]
        sign = _infer_copy_sign(orientation, side, sprite["bbox"], blocks)

        stride = _bbox_w(sprite["bbox"]) + 1 if orientation == "horizontal" else _bbox_h(sprite["bbox"]) + 1
        dr = 0
        dc = 0
        if orientation == "horizontal":
            dc = int(sign * stride)
        else:
            dr = int(sign * stride)

        for k in range(1, count):
            _paint_shifted_component(out, sprite["coords"], dr * k, dc * k, block_color)

    return out


def _aligned_start(center2: int, span: int) -> int:
    return (int(center2) - int(span) + 1) // 2


def _paste_mask(grid: np.ndarray, mask: np.ndarray, top: int, left: int, color: int) -> None:
    rows, cols = np.nonzero(mask)
    grid[rows + int(top), cols + int(left)] = int(color)


def _build_object(
    rng: np.random.Generator,
    inner_color: int,
    sprite_color: int,
    block_size_max: int,
    margin_max: int,
) -> np.ndarray | None:
    count = int(rng.integers(2, 5))
    orientation = str(rng.choice(["horizontal", "vertical"]))
    block_size = int(rng.integers(1, block_size_max + 1))
    top_margin = int(rng.integers(0, margin_max + 1))
    bottom_margin = int(rng.integers(0, margin_max + 1))
    left_margin = int(rng.integers(0, margin_max + 1))
    right_margin = int(rng.integers(0, margin_max + 1))

    if orientation == "horizontal":
        inner_h = top_margin + block_size + bottom_margin
        inner_w = left_margin + count * block_size + (count - 1) + right_margin
        block_row0 = 1 + top_margin
        block_bboxes = []
        for i in range(count):
            c0 = 1 + left_margin + i * (block_size + 1)
            block_bboxes.append((block_row0, c0, block_row0 + block_size - 1, c0 + block_size - 1))
    else:
        inner_h = top_margin + count * block_size + (count - 1) + bottom_margin
        inner_w = left_margin + block_size + right_margin
        block_col0 = 1 + left_margin
        block_bboxes = []
        for i in range(count):
            r0 = 1 + top_margin + i * (block_size + 1)
            block_bboxes.append((r0, block_col0, r0 + block_size - 1, block_col0 + block_size - 1))

    fence_h = inner_h + 2
    fence_w = inner_w + 2
    sprite_mask = SPRITE_SHAPES[int(rng.integers(0, len(SPRITE_SHAPES)))]
    sh, sw = sprite_mask.shape
    side = str(rng.choice(["top", "bottom", "left", "right"]))

    if orientation == "horizontal":
        row_center2 = int(block_bboxes[0][0] + block_bboxes[0][2])
        if side == "left":
            sign = -1
            sprite_r0 = _aligned_start(row_center2, sh)
            sprite_c0 = -sw
        elif side == "right":
            sign = 1
            sprite_r0 = _aligned_start(row_center2, sh)
            sprite_c0 = fence_w
        else:
            start_from_left = bool(rng.integers(0, 2) == 0)
            target = block_bboxes[0] if start_from_left else block_bboxes[-1]
            sign = 1 if start_from_left else -1
            sprite_r0 = -sh if side == "top" else fence_h
            sprite_c0 = _aligned_start(int(target[1] + target[3]), sw)
    else:
        col_center2 = int(block_bboxes[0][1] + block_bboxes[0][3])
        if side == "top":
            sign = -1
            sprite_r0 = -sh
            sprite_c0 = _aligned_start(col_center2, sw)
        elif side == "bottom":
            sign = 1
            sprite_r0 = fence_h
            sprite_c0 = _aligned_start(col_center2, sw)
        else:
            start_from_top = bool(rng.integers(0, 2) == 0)
            target = block_bboxes[0] if start_from_top else block_bboxes[-1]
            sign = 1 if start_from_top else -1
            sprite_r0 = _aligned_start(int(target[0] + target[2]), sh)
            sprite_c0 = -sw if side == "left" else fence_w

    sprite_r1 = sprite_r0 + sh - 1
    sprite_c1 = sprite_c0 + sw - 1
    stride_r = sh + 1 if orientation == "vertical" else 0
    stride_c = sw + 1 if orientation == "horizontal" else 0
    stride_r *= sign
    stride_c *= sign

    sprite_boxes = []
    for k in range(count):
        sprite_boxes.append(
            (
                int(sprite_r0 + k * stride_r),
                int(sprite_c0 + k * stride_c),
                int(sprite_r1 + k * stride_r),
                int(sprite_c1 + k * stride_c),
            )
        )

    fence_box = (0, 0, fence_h - 1, fence_w - 1)
    min_r = min([fence_box[0]] + [box[0] for box in sprite_boxes])
    min_c = min([fence_box[1]] + [box[1] for box in sprite_boxes])
    max_r = max([fence_box[2]] + [box[2] for box in sprite_boxes])
    max_c = max([fence_box[3]] + [box[3] for box in sprite_boxes])
    shift_r = -min_r
    shift_c = -min_c
    patch_h = max_r - min_r + 1
    patch_w = max_c - min_c + 1
    if patch_h > CANVAS_SIZE or patch_w > CANVAS_SIZE:
        return None

    patch = np.zeros((patch_h, patch_w), dtype=int)
    fr0, fc0, fr1, fc1 = (
        int(fence_box[0] + shift_r),
        int(fence_box[1] + shift_c),
        int(fence_box[2] + shift_r),
        int(fence_box[3] + shift_c),
    )
    patch[fr0, fc0 : fc1 + 1] = FENCE_COLOR
    patch[fr1, fc0 : fc1 + 1] = FENCE_COLOR
    patch[fr0 : fr1 + 1, fc0] = FENCE_COLOR
    patch[fr0 : fr1 + 1, fc1] = FENCE_COLOR

    for br0, bc0, br1, bc1 in block_bboxes:
        patch[br0 + shift_r : br1 + shift_r + 1, bc0 + shift_c : bc1 + shift_c + 1] = inner_color

    _paste_mask(patch, sprite_mask, sprite_r0 + shift_r, sprite_c0 + shift_c, sprite_color)
    return patch


def _place_patch(
    canvas: np.ndarray, occupied: np.ndarray, patch: np.ndarray, rng: np.random.Generator
) -> tuple[int, int, int, int] | None:
    ph, pw = patch.shape
    positions: list[tuple[int, int]] = []
    for r in range(CANVAS_SIZE - ph + 1):
        for c in range(CANVAS_SIZE - pw + 1):
            rr0 = max(0, r - SPRITE_PADDING)
            cc0 = max(0, c - SPRITE_PADDING)
            rr1 = min(CANVAS_SIZE, r + ph + SPRITE_PADDING)
            cc1 = min(CANVAS_SIZE, c + pw + SPRITE_PADDING)
            if not occupied[rr0:rr1, cc0:cc1].any():
                positions.append((r, c))
    if not positions:
        return None
    r, c = positions[int(rng.integers(0, len(positions)))]
    mask = patch != BG
    canvas[r : r + ph, c : c + pw][mask] = patch[mask]
    rr0 = max(0, r - SPRITE_PADDING)
    cc0 = max(0, c - SPRITE_PADDING)
    rr1 = min(CANVAS_SIZE, r + ph + SPRITE_PADDING)
    cc1 = min(CANVAS_SIZE, c + pw + SPRITE_PADDING)
    occupied[rr0:rr1, cc0:cc1] = True
    return int(r), int(c), int(r + ph - 1), int(c + pw - 1)


def _gen_attempt(rng: np.random.Generator) -> np.ndarray | None:
    num_objects = int(rng.choice([1, 2, 3], p=[0.25, 0.5, 0.25]))
    palette = [color for color in range(1, 10) if color != FENCE_COLOR]
    rng.shuffle(palette)
    if len(palette) < 2 * num_objects:
        return None

    objects: list[np.ndarray] = []
    for i in range(num_objects):
        if num_objects == 1:
            block_size_max = 3
            margin_max = 2
        elif num_objects == 2:
            block_size_max = 3 if bool(rng.integers(0, 2)) else 2
            margin_max = 2
        else:
            block_size_max = 2
            margin_max = 1

        patch = _build_object(
            rng,
            inner_color=int(palette[2 * i]),
            sprite_color=int(palette[2 * i + 1]),
            block_size_max=block_size_max,
            margin_max=margin_max,
        )
        if patch is None:
            return None
        objects.append(patch)

    objects.sort(key=lambda patch: patch.shape[0] * patch.shape[1], reverse=True)
    canvas = np.zeros((CANVAS_SIZE, CANVAS_SIZE), dtype=int)
    occupied = np.zeros((CANVAS_SIZE, CANVAS_SIZE), dtype=bool)
    placements: list[tuple[int, int, int, int]] = []
    for patch in objects:
        placed = _place_patch(canvas, occupied, patch, rng)
        if placed is None:
            return None
        placements.append(placed)

    min_r = min(box[0] for box in placements)
    min_c = min(box[1] for box in placements)
    max_r = max(box[2] for box in placements)
    max_c = max(box[3] for box in placements)
    extra_top = int(rng.integers(0, min(4, min_r + 1)))
    extra_left = int(rng.integers(0, min(4, min_c + 1)))
    extra_bottom = int(rng.integers(0, min(4, CANVAS_SIZE - max_r)))
    extra_right = int(rng.integers(0, min(4, CANVAS_SIZE - max_c)))
    crop = canvas[
        min_r - extra_top : max_r + extra_bottom + 1,
        min_c - extra_left : max_c + extra_right + 1,
    ].copy()

    out = main(crop)
    if crop.ndim != 2 or out.ndim != 2:
        return None
    if max(crop.shape) > 30 or max(out.shape) > 30:
        return None
    if np.array_equal(crop, out):
        return None
    return crop


def generate_input() -> np.ndarray:
    rng = np.random.default_rng()
    for _ in range(MAX_GENERATION_ATTEMPTS):
        grid = _gen_attempt(rng)
        if grid is not None:
            return grid
    raise RuntimeError(f"failed to generate a valid input after {MAX_GENERATION_ATTEMPTS} attempts")
