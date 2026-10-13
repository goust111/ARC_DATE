import random
from collections import Counter

import numpy as np


BACKGROUND = 0
MAX_GEN_ATTEMPTS = 400
COPY_COUNT_RANGE = (4, 5)
SPRITE_H_RANGE = (4, 6)
SPRITE_W_RANGE = (4, 6)
CANVAS_H_RANGE = (18, 30)
CANVAS_W_RANGE = (18, 30)
FRAME_PAD = 1
FRAME_INNER_GAP = 1
PLACEMENT_MARGIN = 1
CORNERS = ("tl", "tr", "bl", "br")
PALETTE = [1, 2, 3, 4, 5, 6, 7, 8, 9]


def _dominant_color(grid: np.ndarray) -> int:
    values = np.asarray(grid, dtype=int).ravel()
    return int(Counter(map(int, values)).most_common(1)[0][0])


def _bbox_from_coords(coords: np.ndarray) -> tuple[int, int, int, int]:
    r0, c0 = coords.min(axis=0)
    r1, c1 = coords.max(axis=0)
    return int(r0), int(c0), int(r1), int(c1)


def _bbox(mask: np.ndarray) -> tuple[int, int, int, int] | None:
    coords = np.argwhere(mask)
    if coords.size == 0:
        return None
    return _bbox_from_coords(coords)


def _crop(grid: np.ndarray, bbox: tuple[int, int, int, int]) -> np.ndarray:
    r0, c0, r1, c1 = bbox
    return np.asarray(grid[r0 : r1 + 1, c0 : c1 + 1], dtype=int)


def _components(mask: np.ndarray, connectivity: int = 4) -> list[np.ndarray]:
    H, W = mask.shape
    seen = np.zeros_like(mask, dtype=bool)
    if connectivity == 4:
        steps = [(-1, 0), (1, 0), (0, -1), (0, 1)]
    else:
        steps = [(dr, dc) for dr in (-1, 0, 1) for dc in (-1, 0, 1) if (dr, dc) != (0, 0)]
    out = []
    for r in range(H):
        for c in range(W):
            if seen[r, c] or not bool(mask[r, c]):
                continue
            stack = [(r, c)]
            seen[r, c] = True
            coords = []
            while stack:
                rr, cc = stack.pop()
                coords.append((rr, cc))
                for dr, dc in steps:
                    nr, nc = rr + dr, cc + dc
                    if 0 <= nr < H and 0 <= nc < W and not seen[nr, nc] and bool(mask[nr, nc]):
                        seen[nr, nc] = True
                        stack.append((nr, nc))
            out.append(np.array(coords, dtype=int))
    return out


def _transform(arr: np.ndarray, tid: int) -> np.ndarray:
    tid = int(tid)
    if tid == 0:
        return np.array(arr, copy=True)
    if tid == 1:
        return np.rot90(arr, 1)
    if tid == 2:
        return np.rot90(arr, 2)
    if tid == 3:
        return np.rot90(arr, 3)
    flipped = np.fliplr(arr)
    if tid == 4:
        return flipped
    if tid == 5:
        return np.rot90(flipped, 1)
    if tid == 6:
        return np.rot90(flipped, 2)
    if tid == 7:
        return np.rot90(flipped, 3)
    raise ValueError(f"unknown transform id: {tid}")


def _all_transforms(arr: np.ndarray) -> list[np.ndarray]:
    return [_transform(arr, tid) for tid in range(8)]


def _rect_ring_mask(h: int, w: int) -> np.ndarray:
    mask = np.zeros((h, w), dtype=bool)
    mask[0, :] = True
    mask[-1, :] = True
    mask[:, 0] = True
    mask[:, -1] = True
    return mask


def _touches_corner(bbox: tuple[int, int, int, int], shape: tuple[int, int], corner: str) -> bool:
    r0, c0, r1, c1 = bbox
    H, W = shape
    if corner == "tl":
        return r0 == 0 and c0 == 0
    if corner == "tr":
        return r0 == 0 and c1 == W - 1
    if corner == "bl":
        return r1 == H - 1 and c0 == 0
    if corner == "br":
        return r1 == H - 1 and c1 == W - 1
    raise ValueError(corner)


def _detect_stage_frame(grid: np.ndarray, bg: int) -> tuple[int, tuple[int, int, int, int], str]:
    H, W = grid.shape
    candidates = []
    for corner, (cr, cc) in zip(CORNERS, ((0, 0), (0, W - 1), (H - 1, 0), (H - 1, W - 1))):
        color = int(grid[cr, cc])
        if color == bg:
            continue
        comp_mask = grid == color
        comp = None
        for coords in _components(comp_mask, connectivity=4):
            if any(int(r) == cr and int(c) == cc for r, c in coords):
                comp = coords
                break
        if comp is None:
            continue
        bbox = _bbox_from_coords(comp)
        patch = np.zeros((bbox[2] - bbox[0] + 1, bbox[3] - bbox[1] + 1), dtype=bool)
        local = comp - np.array([[bbox[0], bbox[1]]], dtype=int)
        patch[local[:, 0], local[:, 1]] = True
        if patch.shape[0] < 5 or patch.shape[1] < 5:
            continue
        if not _touches_corner(bbox, grid.shape, corner):
            continue
        if np.array_equal(patch, _rect_ring_mask(*patch.shape)):
            candidates.append((color, bbox, corner))
    assert len(candidates) == 1, "expected exactly one corner frame"
    return candidates[0]


def _extract_patches(grid: np.ndarray, bg: int, frame_color: int, frame_bbox: tuple[int, int, int, int]) -> tuple[list[np.ndarray], int]:
    mask = (grid != bg) & (grid != frame_color)
    patches = []
    stage_idx = None
    for coords in _components(mask, connectivity=4):
        bbox = _bbox_from_coords(coords)
        patch = _crop(grid, bbox)
        assert not np.any(patch == bg), "copy components must fill their bbox"
        patches.append(patch)
        if frame_bbox[0] < bbox[0] and bbox[2] < frame_bbox[2] and frame_bbox[1] < bbox[1] and bbox[3] < frame_bbox[3]:
            stage_idx = len(patches) - 1
    assert stage_idx is not None, "stage copy not found"
    assert len(patches) >= 3, "need several copy fragments"
    return patches, stage_idx


def _collect_unique_outputs(patches: list[np.ndarray], stage_idx: int) -> list[np.ndarray]:
    target_shape = patches[stage_idx].shape
    palette = sorted({int(v) for patch in patches for v in np.unique(patch)})
    outputs = {}

    for occluder in palette:
        stage = patches[stage_idx]
        stage_visible = stage != occluder
        if not stage_visible.any() or bool(stage_visible.all()):
            continue

        candidate_colors = sorted([c for c in palette if c != occluder])
        if len(candidate_colors) != 2:
            continue

        options = {}
        valid = int(occluder in stage)
        if not valid:
            continue

        for idx, patch in enumerate(patches):
            if occluder not in patch:
                valid = False
                break
            if idx == stage_idx:
                options[idx] = [np.array(stage, copy=True)]
                continue
            seen = set()
            opts = []
            for arr in _all_transforms(patch):
                if arr.shape != target_shape:
                    continue
                vis = arr != occluder
                if not vis.any() or bool(vis.all()):
                    continue
                if np.any(stage_visible & vis & (arr != stage)):
                    continue
                key = arr.tobytes()
                if key in seen:
                    continue
                seen.add(key)
                opts.append(arr)
            if not opts:
                valid = False
                break
            options[idx] = opts
        if not valid:
            continue

        known = np.full(target_shape, -1, dtype=int)
        known[stage_visible] = stage[stage_visible]
        order = sorted([idx for idx in range(len(patches)) if idx != stage_idx], key=lambda idx: len(options[idx]))
        local_outputs = {}

        def dfs(pos: int, current: np.ndarray) -> None:
            if len(local_outputs) > 1:
                return
            if pos == len(order):
                if np.any(current == -1):
                    return
                if sorted(map(int, np.unique(current))) != candidate_colors:
                    return
                local_outputs[current.tobytes()] = np.array(current, copy=True)
                return
            idx = order[pos]
            for arr in options[idx]:
                vis = arr != occluder
                clash = (current != -1) & vis & (current != arr)
                if np.any(clash):
                    continue
                nxt = np.array(current, copy=True)
                nxt[vis] = arr[vis]
                dfs(pos + 1, nxt)

        dfs(0, known)
        outputs.update(local_outputs)

    return list(outputs.values())


def main(grid: np.ndarray) -> np.ndarray:
    """
    Reconstruct the full two-color sprite from complementary occluded copies and output it in the stage copy's orientation.

    Input structure:
    - A dominant background contains several full-rectangle copies of one two-color sprite.
    - Each copy is rotated or mirrored, and some of its cells are replaced by one shared occluder color; no copy is complete.
    - One corner holds a single-color rectangular frame, and inside that frame sits another occluded copy whose orientation is the output reference.

    Rule:
    - Detect the corner frame and use the copy inside it as the target orientation.
    - Crop every copy, enumerate the eight rotation/flip transforms, and align them by requiring all non-occluded visible cells to agree in the stage copy's coordinate system.
    - Merge the complementary visible cells to recover the full two-color sprite, then return exactly that reconstructed sprite in the stage orientation.

    Key insight:
    - The answer never appears fully in the input; it is only recoverable by matching differently occluded rotated/mirrored copies, and the stage frame chooses which recovered orientation to output.
    """
    g = np.asarray(grid, dtype=int)
    bg = _dominant_color(g)
    frame_color, frame_bbox, _corner = _detect_stage_frame(g, bg)
    patches, stage_idx = _extract_patches(g, bg, frame_color, frame_bbox)
    outputs = _collect_unique_outputs(patches, stage_idx)
    assert len(outputs) == 1, "reconstruction must be unique"
    return outputs[0]


def _random_sprite(h: int, w: int, colors: tuple[int, int]) -> np.ndarray | None:
    a, b = map(int, colors)
    total = h * w
    for _ in range(120):
        mask = np.random.rand(h, w) < random.uniform(0.35, 0.65)
        for _ in range(2):
            next_mask = mask.copy()
            for r in range(h):
                for c in range(w):
                    r0, r1 = max(0, r - 1), min(h, r + 2)
                    c0, c1 = max(0, c - 1), min(w, c + 2)
                    neighborhood = mask[r0:r1, c0:c1]
                    next_mask[r, c] = int(neighborhood.sum()) >= (neighborhood.size // 2 + 1)
            mask = next_mask
        count = int(mask.sum())
        if count < 4 or total - count < 4:
            continue
        sprite = np.where(mask, a, b).astype(int)
        if any(np.array_equal(sprite, _transform(sprite, tid)) for tid in range(1, 8)):
            continue
        if len({tuple(row.tolist()) for row in sprite}) < 3:
            continue
        if len({tuple(col.tolist()) for col in sprite.T}) < 3:
            continue
        return sprite
    return None


def _random_anchor_masks(h: int, w: int, count: int) -> list[np.ndarray] | None:
    anchors = []
    occupied = np.zeros((h, w), dtype=bool)
    for _ in range(count):
        for _ in range(80):
            ah = random.randint(1, min(2, h))
            aw = random.randint(1, min(2, w))
            r0 = random.randint(0, h - ah)
            c0 = random.randint(0, w - aw)
            if occupied[r0 : r0 + ah, c0 : c0 + aw].any():
                continue
            mask = np.zeros((h, w), dtype=bool)
            mask[r0 : r0 + ah, c0 : c0 + aw] = True
            anchors.append(mask)
            occupied |= mask
            break
        else:
            return None
    return anchors


def _add_random_rect(mask: np.ndarray) -> None:
    h, w = mask.shape
    rh = random.randint(1, max(1, h - 1))
    rw = random.randint(1, max(1, w - 1))
    r0 = random.randint(0, h - rh)
    c0 = random.randint(0, w - rw)
    mask[r0 : r0 + rh, c0 : c0 + rw] = True


def _random_hidden_masks(sprite: np.ndarray, copy_count: int) -> list[np.ndarray] | None:
    h, w = sprite.shape
    anchors = _random_anchor_masks(h, w, copy_count)
    if anchors is None:
        return None

    for _ in range(160):
        hidden = [np.zeros((h, w), dtype=bool) for _ in range(copy_count)]
        for i in range(copy_count):
            for j in range(copy_count):
                if i != j:
                    hidden[j] |= anchors[i]
        for j in range(copy_count):
            extra_count = random.randint(1, 3)
            for _ in range(extra_count):
                _add_random_rect(hidden[j])

        all_hidden = np.logical_and.reduce(hidden)
        if all_hidden.any():
            coords = np.argwhere(all_hidden)
            random.shuffle(coords)
            for r, c in coords:
                hidden[random.randrange(copy_count)][int(r), int(c)] = False

        if any(not mask.any() or bool(mask.all()) for mask in hidden):
            continue
        if len({mask.tobytes() for mask in hidden}) != copy_count:
            continue

        visible = [~mask for mask in hidden]
        if not np.logical_or.reduce(visible).all():
            continue
        necessary = []
        for i in range(copy_count):
            others = [visible[j] for j in range(copy_count) if j != i]
            necessary.append(not np.logical_or.reduce(others).all())
        if not all(necessary):
            continue

        color_values = sorted(map(int, np.unique(sprite)))
        good_colors = True
        for vis in visible:
            counts = [int(np.sum((sprite == color) & vis)) for color in color_values]
            if min(counts) < 2:
                good_colors = False
                break
        if not good_colors:
            continue

        coverage = [int(vis.sum()) for vis in visible]
        total = h * w
        if min(coverage) < max(4, total // 3):
            continue
        if max(coverage) > total - 2:
            continue
        return hidden
    return None


def _render_copy(sprite: np.ndarray, hidden_mask: np.ndarray, transform_id: int, occluder: int) -> np.ndarray:
    patch = _transform(sprite, transform_id)
    local_hidden = _transform(hidden_mask.astype(int), transform_id).astype(bool)
    out = np.array(patch, copy=True)
    out[local_hidden] = int(occluder)
    return out


def _random_transforms(copy_count: int) -> list[int]:
    for _ in range(120):
        transforms = random.sample(range(8), copy_count)
        if len(set(transforms)) != copy_count:
            continue
        if not any(t < 4 for t in transforms):
            continue
        if not any(t >= 4 for t in transforms):
            continue
        return transforms
    raise RuntimeError("failed to choose transforms")


def _corner_frame_bbox(H: int, W: int, outer_h: int, outer_w: int, corner: str) -> tuple[int, int, int, int]:
    if corner == "tl":
        return 0, 0, outer_h - 1, outer_w - 1
    if corner == "tr":
        return 0, W - outer_w, outer_h - 1, W - 1
    if corner == "bl":
        return H - outer_h, 0, H - 1, outer_w - 1
    if corner == "br":
        return H - outer_h, W - outer_w, H - 1, W - 1
    raise ValueError(corner)


def _boxes_too_close(a: tuple[int, int, int, int], b: tuple[int, int, int, int], margin: int) -> bool:
    return not (
        a[2] + margin < b[0]
        or b[2] + margin < a[0]
        or a[3] + margin < b[1]
        or b[3] + margin < a[1]
    )


def _try_place_box(H: int, W: int, h: int, w: int, used: list[tuple[int, int, int, int]]) -> tuple[int, int, int, int] | None:
    for _ in range(160):
        r0 = random.randint(PLACEMENT_MARGIN, H - h - PLACEMENT_MARGIN)
        c0 = random.randint(PLACEMENT_MARGIN, W - w - PLACEMENT_MARGIN)
        box = (r0, c0, r0 + h - 1, c0 + w - 1)
        if any(_boxes_too_close(box, other, PLACEMENT_MARGIN) for other in used):
            continue
        return box
    return None


def _paint_rect_patch(grid: np.ndarray, bbox: tuple[int, int, int, int], patch: np.ndarray) -> None:
    r0, c0, r1, c1 = bbox
    assert patch.shape == (r1 - r0 + 1, c1 - c0 + 1)
    grid[r0 : r1 + 1, c0 : c1 + 1] = patch


def _paint_frame(grid: np.ndarray, bbox: tuple[int, int, int, int], color: int) -> None:
    r0, c0, r1, c1 = bbox
    grid[r0, c0 : c1 + 1] = color
    grid[r1, c0 : c1 + 1] = color
    grid[r0 : r1 + 1, c0] = color
    grid[r0 : r1 + 1, c1] = color


def _validate_instance(
    grid: np.ndarray,
    expected: np.ndarray,
    sprite: np.ndarray,
    hidden_masks: list[np.ndarray],
    transforms: list[int],
    stage_idx: int,
) -> bool:
    try:
        solved = main(grid)
        bg = _dominant_color(grid)
        frame_color, frame_bbox, _ = _detect_stage_frame(grid, bg)
        patches, parsed_stage_idx = _extract_patches(grid, bg, frame_color, frame_bbox)
        outputs = _collect_unique_outputs(patches, parsed_stage_idx)
    except AssertionError:
        return False
    if not np.array_equal(solved, expected):
        return False
    if len(outputs) != 1 or not np.array_equal(outputs[0], expected):
        return False
    visible = [~mask for mask in hidden_masks]
    if len(set(transforms)) != len(transforms):
        return False
    if any(np.array_equal(expected, _transform(sprite, t)) for i, t in enumerate(transforms) if i != stage_idx):
        return False
    if not any(t < 4 for t in transforms) or not any(t >= 4 for t in transforms):
        return False
    for i in range(len(visible)):
        others = [visible[j] for j in range(len(visible)) if j != i]
        if np.logical_or.reduce(others).all():
            return False
    return True


def _gen_attempt() -> np.ndarray | None:
    H = random.randint(*CANVAS_H_RANGE)
    W = random.randint(*CANVAS_W_RANGE)
    copy_count = random.randint(*COPY_COUNT_RANGE)
    colors = random.sample(PALETTE, 4)
    sprite_colors = (colors[0], colors[1])
    occluder = int(colors[2])
    frame_color = int(colors[3])

    sprite_h = random.randint(*SPRITE_H_RANGE)
    sprite_w = random.randint(*SPRITE_W_RANGE)
    if sprite_h == sprite_w:
        return None
    sprite = _random_sprite(sprite_h, sprite_w, sprite_colors)
    if sprite is None:
        return None

    hidden_masks = _random_hidden_masks(sprite, copy_count)
    if hidden_masks is None:
        return None

    transforms = _random_transforms(copy_count)
    stage_idx = random.randrange(copy_count)
    rendered_patches = [_render_copy(sprite, hidden_masks[idx], transforms[idx], occluder) for idx in range(copy_count)]
    stage_transform = transforms[stage_idx]
    stage_patch = rendered_patches[stage_idx]
    expected = _transform(sprite, stage_transform)
    patch_outputs = _collect_unique_outputs(rendered_patches, stage_idx)
    if len(patch_outputs) != 1 or not np.array_equal(patch_outputs[0], expected):
        return None

    outer_h = stage_patch.shape[0] + 2 * (FRAME_PAD + FRAME_INNER_GAP)
    outer_w = stage_patch.shape[1] + 2 * (FRAME_PAD + FRAME_INNER_GAP)
    if outer_h > H or outer_w > W:
        return None

    corner = random.choice(CORNERS)
    frame_bbox = _corner_frame_bbox(H, W, outer_h, outer_w, corner)
    used_boxes = [frame_bbox]
    grid = np.full((H, W), BACKGROUND, dtype=int)
    _paint_frame(grid, frame_bbox, frame_color)

    frame_r0, frame_c0, _, _ = frame_bbox
    copy_r0 = frame_r0 + FRAME_PAD + FRAME_INNER_GAP
    copy_c0 = frame_c0 + FRAME_PAD + FRAME_INNER_GAP
    if corner == "tr":
        copy_c0 = frame_bbox[3] - FRAME_PAD - FRAME_INNER_GAP - stage_patch.shape[1] + 1
    elif corner == "bl":
        copy_r0 = frame_bbox[2] - FRAME_PAD - FRAME_INNER_GAP - stage_patch.shape[0] + 1
    elif corner == "br":
        copy_r0 = frame_bbox[2] - FRAME_PAD - FRAME_INNER_GAP - stage_patch.shape[0] + 1
        copy_c0 = frame_bbox[3] - FRAME_PAD - FRAME_INNER_GAP - stage_patch.shape[1] + 1
    stage_copy_bbox = (
        copy_r0,
        copy_c0,
        copy_r0 + stage_patch.shape[0] - 1,
        copy_c0 + stage_patch.shape[1] - 1,
    )
    _paint_rect_patch(grid, stage_copy_bbox, stage_patch)

    for idx in range(copy_count):
        if idx == stage_idx:
            continue
        patch = rendered_patches[idx]
        box = _try_place_box(H, W, patch.shape[0], patch.shape[1], used_boxes)
        if box is None:
            return None
        used_boxes.append(box)
        _paint_rect_patch(grid, box, patch)

    if np.max(grid.shape) > 30:
        return None
    if len({tuple(np.unique(patch).tolist()) for patch in [stage_patch]}) < 1:
        return None
    if not _validate_instance(grid, expected, sprite, hidden_masks, transforms, stage_idx):
        return None
    return grid


def generate_input() -> np.ndarray:
    for _ in range(MAX_GEN_ATTEMPTS):
        grid = _gen_attempt()
        if grid is not None:
            return grid
    raise RuntimeError("failed to generate a valid residual-fragment puzzle")
