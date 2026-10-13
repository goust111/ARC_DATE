from __future__ import annotations

from collections import Counter
import random
import numpy as np


GENERATE_RETRIES = 400
MAX_FRAME_SIDE = 10
MIN_FRAME_SIDE = 6
MAX_CANVAS_SIDE = 30


def _get_bg(grid: np.ndarray) -> int:
    values, counts = np.unique(grid, return_counts=True)
    return int(values[np.argmax(counts)])


def _components(mask: np.ndarray) -> list[np.ndarray]:
    mask = np.asarray(mask, dtype=bool)
    h, w = mask.shape
    seen = np.zeros((h, w), dtype=bool)
    out: list[np.ndarray] = []
    for r in range(h):
        for c in range(w):
            if not mask[r, c] or seen[r, c]:
                continue
            stack = [(r, c)]
            seen[r, c] = True
            coords: list[tuple[int, int]] = []
            while stack:
                rr, cc = stack.pop()
                coords.append((rr, cc))
                for dr, dc in ((1, 0), (-1, 0), (0, 1), (0, -1)):
                    nr, nc = rr + dr, cc + dc
                    if 0 <= nr < h and 0 <= nc < w and mask[nr, nc] and not seen[nr, nc]:
                        seen[nr, nc] = True
                        stack.append((nr, nc))
            out.append(np.array(coords, dtype=int))
    return out


def _bbox(coords: np.ndarray) -> tuple[int, int, int, int]:
    r0, c0 = coords.min(axis=0)
    r1, c1 = coords.max(axis=0)
    return int(r0), int(c0), int(r1), int(c1)


def _crop_mask(mask: np.ndarray) -> np.ndarray:
    mask = np.asarray(mask, dtype=bool)
    rows = np.any(mask, axis=1)
    cols = np.any(mask, axis=0)
    if not rows.any() or not cols.any():
        return np.zeros((0, 0), dtype=bool)
    return mask[np.ix_(rows, cols)]


def _mask_signature(mask: np.ndarray) -> tuple[tuple[int, int], tuple[int, ...]]:
    base = _crop_mask(mask).astype(np.uint8)
    variants: list[tuple[tuple[int, int], tuple[int, ...]]] = []
    for k in range(4):
        rot = np.rot90(base, k)
        for arr in (rot, np.fliplr(rot)):
            arr = _crop_mask(arr)
            variants.append((tuple(map(int, arr.shape)), tuple(int(x) for x in arr.ravel())))
    return min(variants)


def _embed_mask(mask: np.ndarray, h: int, w: int, r0: int, c0: int) -> np.ndarray:
    out = np.zeros((h, w), dtype=bool)
    hh, ww = mask.shape
    out[r0 : r0 + hh, c0 : c0 + ww] = mask
    return out


def _dilate8(mask: np.ndarray) -> np.ndarray:
    h, w = mask.shape
    out = mask.copy()
    for dr in (-1, 0, 1):
        for dc in (-1, 0, 1):
            if dr == 0 and dc == 0:
                continue
            src_r0 = max(0, -dr)
            src_r1 = h - max(0, dr)
            src_c0 = max(0, -dc)
            src_c1 = w - max(0, dc)
            dst_r0 = max(0, dr)
            dst_r1 = dst_r0 + (src_r1 - src_r0)
            dst_c0 = max(0, dc)
            dst_c1 = dst_c0 + (src_c1 - src_c0)
            out[dst_r0:dst_r1, dst_c0:dst_c1] |= mask[src_r0:src_r1, src_c0:src_c1]
    return out


def _extract_holes(patch: np.ndarray, bg: int) -> list[dict]:
    holes: list[dict] = []
    h, w = patch.shape
    for coords in _components(patch == int(bg)):
        if any(r == 0 or c == 0 or r == h - 1 or c == w - 1 for r, c in coords):
            continue
        r0, c0, r1, c1 = _bbox(coords)
        local = np.zeros((r1 - r0 + 1, c1 - c0 + 1), dtype=bool)
        local[coords[:, 0] - r0, coords[:, 1] - c0] = True
        holes.append(
            {
                "coords": coords.copy(),
                "bbox": (r0, c0, r1, c1),
                "mask": local,
                "sig": _mask_signature(local),
            }
        )
    holes.sort(key=lambda item: (item["bbox"][0], item["bbox"][1], item["bbox"][2], item["bbox"][3]))
    return holes


def _find_frame_and_sprite_components(grid: np.ndarray, bg: int) -> tuple[list[dict], list[dict]]:
    g = np.asarray(grid, dtype=int)
    frames: list[dict] = []
    sprites: list[dict] = []
    for color in sorted(int(c) for c in np.unique(g) if int(c) != int(bg)):
        for coords in _components(g == int(color)):
            r0, c0, r1, c1 = _bbox(coords)
            patch = g[r0 : r1 + 1, c0 : c1 + 1]
            local_mask = np.zeros_like(patch, dtype=bool)
            local_mask[coords[:, 0] - r0, coords[:, 1] - c0] = True
            same_color_only = np.array_equal(local_mask, patch == int(color))
            holes = _extract_holes(patch, bg) if same_color_only else []
            is_frame = (
                same_color_only
                and patch.shape[0] >= 3
                and patch.shape[1] >= 3
                and len(holes) >= 1
                and np.all(patch[0, :] == int(color))
                and np.all(patch[-1, :] == int(color))
                and np.all(patch[:, 0] == int(color))
                and np.all(patch[:, -1] == int(color))
            )
            info = {
                "color": int(color),
                "bbox": (r0, c0, r1, c1),
                "mask": local_mask,
                "coords": coords,
                "patch": patch,
            }
            if is_frame:
                info["holes"] = holes
                frames.append(info)
            else:
                info["sig"] = _mask_signature(local_mask)
                sprites.append(info)
    frames.sort(key=lambda item: item["bbox"])
    sprites.sort(key=lambda item: (item["bbox"][0], item["bbox"][1], item["color"]))
    return frames, sprites


def _apply_frame_rule(frame_patch: np.ndarray, frame_color: int, holes: list[dict], fill_colors: list[int]) -> np.ndarray:
    out = np.array(frame_patch, dtype=int, copy=True)
    for hole, color in zip(holes, fill_colors):
        rr = hole["coords"][:, 0]
        cc = hole["coords"][:, 1]
        out[rr, cc] = int(color)
    thick = len(holes)
    out[:thick, :] = int(frame_color)
    out[-thick:, :] = int(frame_color)
    out[:, :thick] = int(frame_color)
    out[:, -thick:] = int(frame_color)
    return out


def main(grid: np.ndarray) -> np.ndarray:
    g = np.asarray(grid, dtype=int)
    bg = _get_bg(g)
    frames, sprites = _find_frame_and_sprite_components(g, bg)
    assert frames, "expected at least one frame"

    sprite_colors: dict[tuple[tuple[int, int], tuple[int, ...]], int] = {}
    for sprite in sprites:
        sig = sprite["sig"]
        color = int(sprite["color"])
        if sig in sprite_colors:
            assert sprite_colors[sig] == color, "ambiguous sprite colors for the same shape"
        else:
            sprite_colors[sig] = color

    out = np.full_like(g, int(bg))
    bbox_r0 = min(frame["bbox"][0] for frame in frames)
    bbox_c0 = min(frame["bbox"][1] for frame in frames)
    bbox_r1 = max(frame["bbox"][2] for frame in frames)
    bbox_c1 = max(frame["bbox"][3] for frame in frames)

    for frame in frames:
        fills: list[int] = []
        for hole in frame["holes"]:
            assert hole["sig"] in sprite_colors, "missing sprite for frame hole"
            fills.append(int(sprite_colors[hole["sig"]]))
        transformed = _apply_frame_rule(frame["patch"], int(frame["color"]), frame["holes"], fills)
        r0, c0, r1, c1 = frame["bbox"]
        out[r0 : r1 + 1, c0 : c1 + 1] = transformed

    return out[bbox_r0 : bbox_r1 + 1, bbox_c0 : bbox_c1 + 1]


def _random_connected_shape(rng: random.Random) -> np.ndarray:
    for _ in range(200):
        h = rng.randint(2, 4)
        w = rng.randint(2, 4)
        area = rng.randint(2, min(h * w, 6))
        cells = {(rng.randrange(h), rng.randrange(w))}
        while len(cells) < area:
            base_r, base_c = rng.choice(tuple(cells))
            nbrs = []
            for dr, dc in ((1, 0), (-1, 0), (0, 1), (0, -1)):
                nr, nc = base_r + dr, base_c + dc
                if 0 <= nr < h and 0 <= nc < w and (nr, nc) not in cells:
                    nbrs.append((nr, nc))
            if not nbrs:
                break
            cells.add(rng.choice(nbrs))
        if len(cells) != area:
            continue
        mask = np.zeros((h, w), dtype=bool)
        for r, c in cells:
            mask[r, c] = True
        mask = _crop_mask(mask)
        if mask.sum() < 2:
            continue
        if mask.shape == (1, 1):
            continue
        inner_voids = [
            comp
            for comp in _components(~mask)
            if not any(r == 0 or c == 0 or r == mask.shape[0] - 1 or c == mask.shape[1] - 1 for r, c in comp)
        ]
        if inner_voids:
            continue
        return mask
    raise RuntimeError("failed to create connected shape")


def _random_shape_transform(mask: np.ndarray, rng: random.Random) -> np.ndarray:
    out = np.array(mask, copy=True)
    if rng.random() < 0.5:
        out = np.fliplr(out)
    return np.rot90(out, rng.randrange(4))


def _place_masks(
    board_h: int,
    board_w: int,
    masks: list[np.ndarray],
    rng: random.Random,
    blocked: np.ndarray | None = None,
    border_gap: int = 0,
    edge_bias_first: bool = False,
) -> list[tuple[int, int]] | None:
    if blocked is None:
        blocked = np.zeros((board_h, board_w), dtype=bool)
    order = sorted(range(len(masks)), key=lambda i: int(masks[i].sum()), reverse=True)
    placements: dict[int, tuple[int, int]] = {}

    def rec(idx: int, cur_blocked: np.ndarray) -> bool:
        if idx == len(order):
            return True
        hole_idx = order[idx]
        mask = masks[hole_idx]
        hh, ww = mask.shape
        candidates = [
            (r, c)
            for r in range(border_gap, board_h - hh - border_gap + 1)
            for c in range(border_gap, board_w - ww - border_gap + 1)
        ]
        rng.shuffle(candidates)
        if edge_bias_first and idx == 0:
            candidates.sort(
                key=lambda rc: not (
                    rc[0] == border_gap
                    or rc[1] == border_gap
                    or rc[0] + hh == board_h - border_gap
                    or rc[1] + ww == board_w - border_gap
                )
            )
        for r0, c0 in candidates:
            full = _embed_mask(mask, board_h, board_w, r0, c0)
            if np.any(full & cur_blocked):
                continue
            placements[hole_idx] = (r0, c0)
            if rec(idx + 1, cur_blocked | _dilate8(full)):
                return True
            placements.pop(hole_idx, None)
        return False

    if not rec(0, blocked.copy()):
        return None
    return [placements[i] for i in range(len(masks))]


def _build_frame_patch(bg: int, frame_color: int, hole_masks: list[np.ndarray], rng: random.Random) -> tuple[np.ndarray, list[dict]]:
    n_holes = len(hole_masks)
    areas = [int(mask.sum()) for mask in hole_masks]
    max_h = max(int(mask.shape[0]) for mask in hole_masks)
    max_w = max(int(mask.shape[1]) for mask in hole_masks)
    min_side = max(
        MIN_FRAME_SIDE,
        max_h + 2,
        max_w + 2,
        2 * n_holes + 1,
        int(np.ceil(np.sqrt(sum(areas) * 2.2))) + 2,
    )
    candidates: list[tuple[int, int]] = []
    for h in range(min_side, MAX_FRAME_SIDE + 1):
        for w in range(min_side, MAX_FRAME_SIDE + 1):
            candidates.append((h, w))
    rng.shuffle(candidates)
    for h, w in candidates:
        placements = _place_masks(
            h,
            w,
            hole_masks,
            rng,
            border_gap=1,
            edge_bias_first=n_holes > 1 and rng.random() < 0.65,
        )
        if placements is None:
            continue
        patch = np.full((h, w), int(frame_color), dtype=int)
        holes: list[dict] = []
        for mask, (r0, c0) in zip(hole_masks, placements):
            full = _embed_mask(mask, h, w, r0, c0)
            patch[full] = int(bg)
            coords = np.argwhere(full)
            rr0, cc0, rr1, cc1 = _bbox(coords)
            holes.append(
                {
                    "coords": coords,
                    "bbox": (rr0, cc0, rr1, cc1),
                    "mask": mask.copy(),
                    "sig": _mask_signature(mask),
                }
            )
        holes.sort(key=lambda item: (item["bbox"][0], item["bbox"][1], item["bbox"][2], item["bbox"][3]))
        return patch, holes
    raise RuntimeError("failed to place holes inside frame")


def _arrange_frames(frame_patches: list[np.ndarray], rng: random.Random) -> tuple[list[tuple[int, int]], int, int]:
    if len(frame_patches) == 1:
        patch = frame_patches[0]
        return [(0, 0)], int(patch.shape[0]), int(patch.shape[1])

    orientation = rng.choice(["horizontal", "vertical"])
    gap = rng.randint(1, 3)
    if orientation == "horizontal":
        union_h = max(int(p.shape[0]) for p in frame_patches)
        xs: list[int] = []
        x = 0
        for patch in frame_patches:
            xs.append(x)
            x += int(patch.shape[1]) + gap
        union_w = x - gap
        placements = [(rng.randint(0, union_h - int(p.shape[0])), xs[i]) for i, p in enumerate(frame_patches)]
        return placements, union_h, union_w

    union_w = max(int(p.shape[1]) for p in frame_patches)
    ys: list[int] = []
    y = 0
    for patch in frame_patches:
        ys.append(y)
        y += int(patch.shape[0]) + gap
    union_h = y - gap
    placements = [(ys[i], rng.randint(0, union_w - int(p.shape[1]))) for i, p in enumerate(frame_patches)]
    return placements, union_h, union_w


def _paint_mask(canvas: np.ndarray, mask: np.ndarray, color: int, r0: int, c0: int) -> None:
    hh, ww = mask.shape
    patch = canvas[r0 : r0 + hh, c0 : c0 + ww]
    patch[mask] = int(color)


def _place_sprites(
    canvas_h: int,
    canvas_w: int,
    union_bbox: tuple[int, int, int, int],
    sprite_masks: list[np.ndarray],
    rng: random.Random,
) -> list[tuple[int, int]] | None:
    blocked = np.zeros((canvas_h, canvas_w), dtype=bool)
    ur0, uc0, ur1, uc1 = union_bbox
    blocked[max(0, ur0 - 1) : min(canvas_h, ur1 + 2), max(0, uc0 - 1) : min(canvas_w, uc1 + 2)] = True
    return _place_masks(canvas_h, canvas_w, sprite_masks, rng, blocked=blocked, border_gap=0)


def _expected_output(
    bg: int,
    frame_patches: list[np.ndarray],
    frame_infos: list[list[dict]],
    fill_colors: list[list[int]],
    placements: list[tuple[int, int]],
    union_h: int,
    union_w: int,
) -> np.ndarray:
    out = np.full((union_h, union_w), int(bg), dtype=int)
    for patch, holes, colors, (r0, c0) in zip(frame_patches, frame_infos, fill_colors, placements):
        transformed = _apply_frame_rule(patch, int(patch[0, 0]), holes, colors)
        hh, ww = transformed.shape
        out[r0 : r0 + hh, c0 : c0 + ww] = transformed
    return out


def _gen_attempt() -> np.ndarray | None:
    rng = random.Random()
    bg = rng.randrange(10)
    frame_count = rng.choice([1, 2])
    hole_counts = [rng.randint(1, 3) for _ in range(frame_count)]
    total_holes = sum(hole_counts)

    usable_colors = [c for c in range(10) if c != bg]
    rng.shuffle(usable_colors)
    needed = frame_count + total_holes
    if len(usable_colors) < needed:
        return None
    frame_colors = usable_colors[:frame_count]
    sprite_colors_flat = usable_colors[frame_count:needed]

    used_shapes: set[tuple[tuple[int, int], tuple[int, ...]]] = set()
    all_hole_masks: list[np.ndarray] = []
    for _ in range(total_holes):
        for _retry in range(200):
            mask = _random_connected_shape(rng)
            sig = _mask_signature(mask)
            if sig not in used_shapes:
                used_shapes.add(sig)
                all_hole_masks.append(mask)
                break
        else:
            return None

    frame_patches: list[np.ndarray] = []
    frame_infos: list[list[dict]] = []
    fill_colors: list[list[int]] = []
    sprite_masks: list[np.ndarray] = []
    sprite_colors: list[int] = []

    cursor = 0
    for frame_idx in range(frame_count):
        n_holes = hole_counts[frame_idx]
        hole_masks = all_hole_masks[cursor : cursor + n_holes]
        hole_fill_colors = sprite_colors_flat[cursor : cursor + n_holes]
        cursor += n_holes
        try:
            patch, holes = _build_frame_patch(bg, frame_colors[frame_idx], hole_masks, rng)
        except RuntimeError:
            return None
        frame_patches.append(patch)
        frame_infos.append(holes)
        fill_colors.append([int(c) for c in hole_fill_colors])
        for mask, color in zip(hole_masks, hole_fill_colors):
            sprite_masks.append(_random_shape_transform(mask, rng))
            sprite_colors.append(int(color))

    frame_positions, union_h, union_w = _arrange_frames(frame_patches, rng)
    if union_h > MAX_CANVAS_SIDE - 4 or union_w > MAX_CANVAS_SIDE - 4:
        return None

    expected = _expected_output(bg, frame_patches, frame_infos, fill_colors, frame_positions, union_h, union_w)

    margin_choices = list(range(2, 6))
    rng.shuffle(margin_choices)
    for top in margin_choices:
        for bottom in margin_choices:
            for left in margin_choices:
                for right in margin_choices:
                    canvas_h = union_h + top + bottom
                    canvas_w = union_w + left + right
                    if canvas_h > MAX_CANVAS_SIDE or canvas_w > MAX_CANVAS_SIDE:
                        continue
                    union_bbox = (top, left, top + union_h - 1, left + union_w - 1)
                    sprite_positions = _place_sprites(canvas_h, canvas_w, union_bbox, sprite_masks, rng)
                    if sprite_positions is None:
                        continue
                    canvas = np.full((canvas_h, canvas_w), int(bg), dtype=int)
                    for patch, (r0, c0) in zip(frame_patches, frame_positions):
                        hh, ww = patch.shape
                        canvas[top + r0 : top + r0 + hh, left + c0 : left + c0 + ww] = patch
                    for mask, color, (r0, c0) in zip(sprite_masks, sprite_colors, sprite_positions):
                        _paint_mask(canvas, mask, color, r0, c0)
                    try:
                        actual = main(canvas)
                    except AssertionError:
                        continue
                    if np.array_equal(actual, expected):
                        return canvas
    return None


def generate_input() -> np.ndarray:
    for _ in range(GENERATE_RETRIES):
        grid = _gen_attempt()
        if grid is not None:
            return grid
    raise RuntimeError("generate_input: failed to build a valid input")
