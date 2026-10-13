from __future__ import annotations

import random

import numpy as np


def _most_common_color(grid: np.ndarray) -> int:
    values, counts = np.unique(grid, return_counts=True)
    return int(values[np.argmax(counts)])


def _components(mask: np.ndarray) -> list[np.ndarray]:
    h, w = mask.shape
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
                cr, cc = stack.pop()
                cells.append((cr, cc))
                for dr, dc in ((-1, 0), (1, 0), (0, -1), (0, 1)):
                    nr, nc = cr + dr, cc + dc
                    if 0 <= nr < h and 0 <= nc < w and mask[nr, nc] and not seen[nr, nc]:
                        seen[nr, nc] = True
                        stack.append((nr, nc))
            comps.append(np.array(cells, dtype=int))
    return comps


def _bbox(coords: np.ndarray) -> tuple[int, int, int, int]:
    r0, c0 = coords.min(axis=0)
    r1, c1 = coords.max(axis=0)
    return int(r0), int(c0), int(r1), int(c1)


def _coords_to_mask(coords: np.ndarray) -> np.ndarray:
    r0, c0, r1, c1 = _bbox(coords)
    out = np.zeros((r1 - r0 + 1, c1 - c0 + 1), dtype=bool)
    out[coords[:, 0] - r0, coords[:, 1] - c0] = True
    return out


def _crop_mask(mask: np.ndarray) -> np.ndarray:
    coords = np.argwhere(mask)
    assert len(coords) > 0
    r0, c0, r1, c1 = _bbox(coords)
    return mask[r0 : r1 + 1, c0 : c1 + 1]


def _mask_signature(mask: np.ndarray) -> tuple[int, int, bytes]:
    tight = _crop_mask(mask)
    return int(tight.shape[0]), int(tight.shape[1]), tight.tobytes()


def _dilate8(mask: np.ndarray) -> np.ndarray:
    h, w = mask.shape
    out = np.zeros((h + 2, w + 2), dtype=bool)
    for dr in range(3):
        for dc in range(3):
            out[dr : dr + h, dc : dc + w] |= mask
    return out


def _extract_frame(
    grid: np.ndarray, bg: int, coords: np.ndarray
) -> dict[str, object] | None:
    r0, c0, r1, c1 = _bbox(coords)
    patch = grid[r0 : r1 + 1, c0 : c1 + 1]
    h, w = patch.shape
    if h < 5 or w < 5:
        return None

    boundary = np.zeros((h, w), dtype=bool)
    boundary[0, :] = True
    boundary[-1, :] = True
    boundary[:, 0] = True
    boundary[:, -1] = True
    boundary_vals = patch[boundary]
    if np.any(boundary_vals == bg):
        return None
    border_colors = {int(v) for v in np.unique(boundary_vals)}
    if len(border_colors) != 1:
        return None
    border_color = next(iter(border_colors))

    hole_mask = patch == bg
    hole_comps = _components(hole_mask)
    if len(hole_comps) != 1:
        return None
    hole_coords = hole_comps[0]
    hr0, hc0, hr1, hc1 = _bbox(hole_coords)
    if hr0 == 0 or hc0 == 0 or hr1 == h - 1 or hc1 == w - 1:
        return None

    inside = patch[1:-1, 1:-1]
    inside_hole = hole_mask[1:-1, 1:-1]
    if not np.any(~inside_hole):
        return None
    fill_colors = {int(v) for v in np.unique(inside[~inside_hole])}
    if len(fill_colors) != 1:
        return None
    fill_color = next(iter(fill_colors))
    if fill_color == border_color or fill_color == bg:
        return None

    tight_hole = np.zeros((hr1 - hr0 + 1, hc1 - hc0 + 1), dtype=bool)
    tight_hole[hole_coords[:, 0] - hr0, hole_coords[:, 1] - hc0] = True
    return {
        "bbox": (r0, c0, r1, c1),
        "coords": coords,
        "border_color": border_color,
        "fill_color": fill_color,
        "hole_bbox": (r0 + hr0, c0 + hc0, r0 + hr1, c0 + hc1),
        "hole_mask": tight_hole,
    }


def _extract_key(grid: np.ndarray, bg: int, coords: np.ndarray) -> dict[str, object]:
    r0, c0, r1, c1 = _bbox(coords)
    patch = grid[r0 : r1 + 1, c0 : c1 + 1]
    mask = np.zeros((r1 - r0 + 1, c1 - c0 + 1), dtype=bool)
    mask[coords[:, 0] - r0, coords[:, 1] - c0] = True
    colors = {int(v) for v in np.unique(patch[mask])}
    assert len(colors) == 1, "every key must be monochrome"
    return {
        "bbox": (r0, c0, r1, c1),
        "coords": coords,
        "mask": mask,
        "color": next(iter(colors)),
    }


def _solve_assignment(
    frames: list[dict[str, object]], keys: list[dict[str, object]]
) -> list[tuple[int, np.ndarray]]:
    candidates: list[list[tuple[int, np.ndarray]]] = []
    for frame in frames:
        hole_mask = frame["hole_mask"]
        frame_cands: list[tuple[int, np.ndarray]] = []
        for key_idx, key in enumerate(keys):
            seen_rotations: set[tuple[int, int, bytes]] = set()
            for rot in range(4):
                rot_mask = np.rot90(key["mask"], rot)
                rot_sig = _mask_signature(rot_mask)
                if rot_sig in seen_rotations:
                    continue
                seen_rotations.add(rot_sig)
                if np.array_equal(_dilate8(rot_mask), hole_mask):
                    frame_cands.append((key_idx, rot_mask))
        assert frame_cands, "every frame must match at least one key"
        candidates.append(frame_cands)

    order = sorted(range(len(frames)), key=lambda idx: (len(candidates[idx]), idx))
    assignment: list[tuple[int, np.ndarray] | None] = [None] * len(frames)
    used: set[int] = set()

    def dfs(pos: int) -> bool:
        if pos == len(order):
            return True
        frame_idx = order[pos]
        for key_idx, rot_mask in candidates[frame_idx]:
            if key_idx in used:
                continue
            used.add(key_idx)
            assignment[frame_idx] = (key_idx, rot_mask)
            if dfs(pos + 1):
                return True
            assignment[frame_idx] = None
            used.remove(key_idx)
        return False

    assert dfs(0), "frames and keys must admit a complete matching"
    return [item for item in assignment if item is not None]


def main(grid: np.ndarray) -> np.ndarray:
    """
    Match each rectangular frame's background keyhole to the correct loose key, rotate the key into the hole, and outline the remaining gap with the frame border color.

    Input structure:
    - The background is the dominant color.
    - Several multicolor rectangular frame components appear; each has a one-pixel-thick border in a unique color, a uniform interior fill color, and exactly one connected background hole inside.
    - Separate monochrome key sprites of a shared color lie elsewhere on the grid, one per frame.

    Rule:
    - Detect each frame and extract the tight mask of its internal background hole.
    - For each loose key sprite, try the four right-angle rotations and find the unique frame whose hole equals that rotated key after a one-pixel 8-neighborhood expansion.
    - Erase the original loose keys, then place each matched rotated key into its frame so the expanded hole becomes the key plus a one-cell gap layer.
    - Fill every hole cell not covered by the inserted key with that frame's border color, leaving the rest of the grid unchanged.

    Key insight:
    - The hole is not the key itself: it is the key dilated outward by one pixel, so solving the puzzle requires reversing that hidden expansion before translating the key into the frame.
    """
    g = np.asarray(grid, dtype=int)
    assert g.ndim == 2
    bg = _most_common_color(g)

    comps = _components(g != bg)
    assert comps, "expected at least one foreground component"

    frames: list[dict[str, object]] = []
    keys: list[dict[str, object]] = []
    for coords in comps:
        frame = _extract_frame(g, bg, coords)
        if frame is not None:
            frames.append(frame)
        else:
            keys.append(_extract_key(g, bg, coords))

    assert frames, "expected at least one frame"
    assert keys, "expected at least one key"
    assert len(frames) == len(keys), "frame and key counts must match"

    key_colors = {int(key["color"]) for key in keys}
    assert len(key_colors) == 1, "all keys must share one color"
    key_color = next(iter(key_colors))

    assignment = _solve_assignment(frames, keys)

    out = g.copy()
    for key in keys:
        out[key["coords"][:, 0], key["coords"][:, 1]] = bg

    for frame, (key_idx, rot_mask) in zip(frames, assignment):
        del key_idx
        hr0, hc0, hr1, hc1 = frame["hole_bbox"]
        hole_mask = frame["hole_mask"]
        border_color = int(frame["border_color"])
        sub = out[hr0 : hr1 + 1, hc0 : hc1 + 1]
        sub[hole_mask] = border_color
        kh, kw = rot_mask.shape
        inner = sub[1 : 1 + kh, 1 : 1 + kw]
        inner[rot_mask] = key_color

    return out


GRID_MIN_SIDE = 18
GRID_MAX_SIDE = 30
OBJECT_COUNT_MIN = 2
OBJECT_COUNT_MAX = 4
SHAPE_AREA_MIN = 3
SHAPE_AREA_MAX = 6
SHAPE_BBOX_MAX = 5
FRAME_MARGIN_MAX = 2
PLACE_GAP = 1
GENERATION_ATTEMPTS = 500


def _grow_shape(area: int) -> np.ndarray:
    cells = {(0, 0)}
    while len(cells) < area:
        frontier: set[tuple[int, int]] = set()
        for r, c in cells:
            frontier.update({(r - 1, c), (r + 1, c), (r, c - 1), (r, c + 1)})
        frontier.difference_update(cells)
        cells.add(random.choice(tuple(frontier)))
    coords = np.array(sorted(cells), dtype=int)
    return _coords_to_mask(coords)


def _sample_base_shape(used_hole_sigs: set[tuple[int, int, bytes]]) -> np.ndarray | None:
    for _ in range(200):
        area = random.randint(SHAPE_AREA_MIN, SHAPE_AREA_MAX)
        mask = _grow_shape(area)
        if max(mask.shape) > SHAPE_BBOX_MAX or min(mask.shape) < 2:
            continue
        rot_sigs = [_mask_signature(np.rot90(mask, rot)) for rot in range(4)]
        if len(set(rot_sigs)) != 4:
            continue
        hole_sigs = [_mask_signature(_dilate8(np.rot90(mask, rot))) for rot in range(4)]
        if len(set(hole_sigs)) != 4:
            continue
        if any(sig in used_hole_sigs for sig in hole_sigs):
            continue
        used_hole_sigs.update(hole_sigs)
        return mask
    return None


def _build_frame_patch(
    hole_mask: np.ndarray, bg: int, border_color: int, fill_color: int
) -> tuple[np.ndarray, tuple[int, int]]:
    hh, hw = hole_mask.shape
    top = random.randint(0, FRAME_MARGIN_MAX)
    bottom = random.randint(0, FRAME_MARGIN_MAX)
    left = random.randint(0, FRAME_MARGIN_MAX)
    right = random.randint(0, FRAME_MARGIN_MAX)
    if top == bottom == left == right == 0:
        side = random.randrange(4)
        if side == 0:
            top = 1
        elif side == 1:
            bottom = 1
        elif side == 2:
            left = 1
        else:
            right = 1
    inner_h = hh + top + bottom
    inner_w = hw + left + right
    patch = np.full((inner_h + 2, inner_w + 2), fill_color, dtype=int)
    patch[0, :] = border_color
    patch[-1, :] = border_color
    patch[:, 0] = border_color
    patch[:, -1] = border_color
    hole_r = 1 + top
    hole_c = 1 + left
    sub = patch[hole_r : hole_r + hh, hole_c : hole_c + hw]
    sub[hole_mask] = bg
    return patch, (hole_r, hole_c)


def _key_patch(mask: np.ndarray, key_color: int, bg: int) -> np.ndarray:
    return np.where(mask, key_color, bg).astype(int)


def _can_place(
    reserved: np.ndarray, top: int, left: int, h: int, w: int, pad: int
) -> bool:
    grid_h, grid_w = reserved.shape
    r0 = max(0, top - pad)
    c0 = max(0, left - pad)
    r1 = min(grid_h, top + h + pad)
    c1 = min(grid_w, left + w + pad)
    return not np.any(reserved[r0:r1, c0:c1])


def _mark_reserved(
    reserved: np.ndarray, top: int, left: int, h: int, w: int, pad: int
) -> None:
    grid_h, grid_w = reserved.shape
    r0 = max(0, top - pad)
    c0 = max(0, left - pad)
    r1 = min(grid_h, top + h + pad)
    c1 = min(grid_w, left + w + pad)
    reserved[r0:r1, c0:c1] = True


def _place_objects(
    bg: int, objects: list[dict[str, object]]
) -> tuple[np.ndarray, dict[int, tuple[int, int]]] | None:
    max_h = max(int(obj["patch"].shape[0]) for obj in objects)
    max_w = max(int(obj["patch"].shape[1]) for obj in objects)
    bbox_area = sum((int(obj["patch"].shape[0]) + 2) * (int(obj["patch"].shape[1]) + 2) for obj in objects)

    for _ in range(80):
        min_h = max(GRID_MIN_SIDE, max_h + 2)
        min_w = max(GRID_MIN_SIDE, max_w + 2)
        grid_h = random.randint(min_h, GRID_MAX_SIDE)
        needed_w = max(min_w, (bbox_area + grid_h - 1) // grid_h)
        if needed_w > GRID_MAX_SIDE:
            continue
        grid_w = random.randint(needed_w, GRID_MAX_SIDE)

        canvas = np.full((grid_h, grid_w), bg, dtype=int)
        reserved = np.zeros((grid_h, grid_w), dtype=bool)
        placements: dict[int, tuple[int, int]] = {}
        success = True

        for obj in sorted(objects, key=lambda item: int(np.prod(item["patch"].shape)), reverse=True):
            patch = obj["patch"]
            h, w = patch.shape
            candidates: list[tuple[int, int]] = []
            for top in range(1, grid_h - h):
                for left in range(1, grid_w - w):
                    if _can_place(reserved, top, left, h, w, PLACE_GAP):
                        candidates.append((top, left))
            if not candidates:
                success = False
                break
            top, left = random.choice(candidates)
            canvas[top : top + h, left : left + w] = patch
            _mark_reserved(reserved, top, left, h, w, PLACE_GAP)
            placements[int(obj["id"])] = (top, left)

        if not success:
            continue
        if _most_common_color(canvas) != bg:
            continue
        return canvas, placements
    return None


def _build_expected(
    canvas: np.ndarray,
    bg: int,
    key_color: int,
    frames: list[dict[str, object]],
    keys: list[dict[str, object]],
    placements: dict[int, tuple[int, int]],
) -> np.ndarray:
    out = canvas.copy()
    for key in keys:
        top, left = placements[int(key["id"])]
        mask = key["mask"]
        sub = out[top : top + mask.shape[0], left : left + mask.shape[1]]
        sub[mask] = bg

    for frame in frames:
        top, left = placements[int(frame["id"])]
        patch = frame["patch"]
        hole_r, hole_c = frame["hole_offset"]
        hole_mask = frame["hole_mask"]
        border_color = int(frame["border_color"])
        key_mask = frame["output_key_mask"]
        hh, hw = hole_mask.shape
        sub = out[top + hole_r : top + hole_r + hh, left + hole_c : left + hole_c + hw]
        sub[hole_mask] = border_color
        inner = sub[1 : 1 + key_mask.shape[0], 1 : 1 + key_mask.shape[1]]
        inner[key_mask] = key_color
        assert patch.shape[0] >= hole_r + hh and patch.shape[1] >= hole_c + hw

    return out


def _gen_attempt() -> np.ndarray | None:
    count = random.randint(OBJECT_COUNT_MIN, OBJECT_COUNT_MAX)
    palette = list(range(10))
    random.shuffle(palette)
    bg = palette.pop()
    key_color = palette.pop()
    border_colors = [palette.pop() for _ in range(count)]
    fill_colors = [palette.pop() for _ in range(count)]

    used_hole_sigs: set[tuple[int, int, bytes]] = set()
    base_shapes: list[np.ndarray] = []
    for _ in range(count):
        shape = _sample_base_shape(used_hole_sigs)
        if shape is None:
            return None
        base_shapes.append(shape)

    frame_specs: list[dict[str, object]] = []
    key_specs: list[dict[str, object]] = []
    objects: list[dict[str, object]] = []
    object_id = 0

    for idx, base in enumerate(base_shapes):
        input_mask = np.rot90(base, random.randint(0, 3))
        output_mask = np.rot90(base, random.randint(0, 3))
        hole_mask = _dilate8(output_mask)
        frame_patch, hole_offset = _build_frame_patch(
            hole_mask, bg, border_colors[idx], fill_colors[idx]
        )
        key_patch = _key_patch(input_mask, key_color, bg)

        frame_spec = {
            "id": object_id,
            "patch": frame_patch,
            "hole_offset": hole_offset,
            "hole_mask": hole_mask,
            "border_color": border_colors[idx],
            "output_key_mask": output_mask,
        }
        frame_specs.append(frame_spec)
        objects.append(frame_spec)
        object_id += 1

        key_spec = {
            "id": object_id,
            "patch": key_patch,
            "mask": input_mask,
        }
        key_specs.append(key_spec)
        objects.append(key_spec)
        object_id += 1

    placed = _place_objects(bg, objects)
    if placed is None:
        return None
    canvas, placements = placed

    expected = _build_expected(canvas, bg, key_color, frame_specs, key_specs, placements)
    try:
        produced = main(canvas)
    except Exception:
        return None
    if not np.array_equal(produced, expected):
        return None
    return canvas


def generate_input() -> np.ndarray:
    for _ in range(GENERATION_ATTEMPTS):
        grid = _gen_attempt()
        if grid is not None:
            return grid
    raise RuntimeError("failed to generate a valid input")
