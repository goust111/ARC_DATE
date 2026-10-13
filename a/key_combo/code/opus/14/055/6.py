from __future__ import annotations

from collections import deque
from itertools import permutations

import numpy as np


BG = 0
FRAME_COLORS = (5, 7, 8)
SPRITE_COLORS = (2, 1, 3)
RNG = np.random.default_rng()
MAX_ATTEMPTS = 400


def _bbox(mask: np.ndarray) -> tuple[int, int, int, int]:
    rows, cols = np.nonzero(mask)
    assert len(rows) > 0, "expected a non-empty mask"
    return int(rows.min()), int(cols.min()), int(rows.max()), int(cols.max())


def _neighbors4(r: int, c: int) -> tuple[tuple[int, int], ...]:
    return ((r - 1, c), (r + 1, c), (r, c - 1), (r, c + 1))


def _component_masks(mask: np.ndarray) -> list[np.ndarray]:
    h, w = mask.shape
    seen = np.zeros((h, w), dtype=bool)
    comps: list[np.ndarray] = []
    for sr, sc in np.argwhere(mask):
        sr = int(sr)
        sc = int(sc)
        if seen[sr, sc]:
            continue
        comp = np.zeros((h, w), dtype=bool)
        stack = [(sr, sc)]
        seen[sr, sc] = True
        while stack:
            r, c = stack.pop()
            comp[r, c] = True
            for nr, nc in _neighbors4(r, c):
                if 0 <= nr < h and 0 <= nc < w and mask[nr, nc] and not seen[nr, nc]:
                    seen[nr, nc] = True
                    stack.append((nr, nc))
        comps.append(comp)
    return comps


def _is_rect_frame(mask: np.ndarray) -> bool:
    if not mask.any():
        return False
    r0, c0, r1, c1 = _bbox(mask)
    if r1 - r0 < 2 or c1 - c0 < 2:
        return False
    expected = np.zeros_like(mask, dtype=bool)
    expected[r0, c0 : c1 + 1] = True
    expected[r1, c0 : c1 + 1] = True
    expected[r0 : r1 + 1, c0] = True
    expected[r0 : r1 + 1, c1] = True
    return np.array_equal(mask, expected)


def _scaled_index(i: int, src_len: int, dst_len: int) -> int:
    assert src_len >= 1 and dst_len >= 1
    if src_len == 1:
        return 0 if dst_len == 1 else dst_len // 2
    return int(round(int(i) * (dst_len - 1) / (src_len - 1)))


def _draw_segment(mask: np.ndarray, a: tuple[int, int], b: tuple[int, int]) -> None:
    (r0, c0), (r1, c1) = a, b
    if r0 == r1:
        lo, hi = sorted((c0, c1))
        mask[r0, lo : hi + 1] = True
        return
    if c0 == c1:
        lo, hi = sorted((r0, r1))
        mask[lo : hi + 1, c0] = True
        return
    raise AssertionError("adjacent source pixels must map to horizontal or vertical segments")


def _fill_enclosed(stroke: np.ndarray) -> np.ndarray:
    h, w = stroke.shape
    open_space = ~stroke
    padded = np.pad(open_space, 1, constant_values=True)
    seen = np.zeros_like(padded, dtype=bool)
    q: deque[tuple[int, int]] = deque()
    ph, pw = padded.shape

    for r in range(ph):
        for c in (0, pw - 1):
            if padded[r, c] and not seen[r, c]:
                seen[r, c] = True
                q.append((r, c))
    for c in range(pw):
        for r in (0, ph - 1):
            if padded[r, c] and not seen[r, c]:
                seen[r, c] = True
                q.append((r, c))

    while q:
        r, c = q.popleft()
        for nr, nc in _neighbors4(r, c):
            if 0 <= nr < ph and 0 <= nc < pw and padded[nr, nc] and not seen[nr, nc]:
                seen[nr, nc] = True
                q.append((nr, nc))

    holes = open_space & ~seen[1 : h + 1, 1 : w + 1]
    return stroke | holes


def _stretch_sprite(sprite: np.ndarray, target_shape: tuple[int, int]) -> np.ndarray:
    src_h, src_w = sprite.shape
    dst_h, dst_w = target_shape
    assert sprite.any(), "expected a non-empty sprite"
    assert dst_h >= 1 and dst_w >= 1, "frame interior must be non-empty"

    stroke = np.zeros((dst_h, dst_w), dtype=bool)
    coords = np.argwhere(sprite)
    mapped: dict[tuple[int, int], tuple[int, int]] = {}
    for sr, sc in coords:
        sr = int(sr)
        sc = int(sc)
        tr = _scaled_index(sr, src_h, dst_h)
        tc = _scaled_index(sc, src_w, dst_w)
        mapped[(sr, sc)] = (tr, tc)
        stroke[tr, tc] = True

    for sr, sc in coords:
        sr = int(sr)
        sc = int(sc)
        for nr, nc in ((sr + 1, sc), (sr, sc + 1)):
            if (nr, nc) in mapped:
                _draw_segment(stroke, mapped[(sr, sc)], mapped[(nr, nc)])

    return _fill_enclosed(stroke)


def _extract_objects(grid: np.ndarray) -> tuple[int, list[dict], list[dict]]:
    g = np.asarray(grid, dtype=int)
    colors, counts = np.unique(g, return_counts=True)
    bg = int(colors[np.argmax(counts)])

    frames: list[dict] = []
    sprites: list[dict] = []
    for color in sorted(int(c) for c in colors if int(c) != bg):
        mask = g == color
        comps = _component_masks(mask)
        assert len(comps) == 1, "each object color must form one component"
        comp = comps[0]
        bbox = _bbox(comp)
        if _is_rect_frame(comp):
            r0, c0, r1, c1 = bbox
            frames.append(
                {
                    "color": color,
                    "bbox": bbox,
                    "interior_area": (r1 - r0 - 1) * (c1 - c0 - 1),
                }
            )
        else:
            sr0, sc0, sr1, sc1 = bbox
            cropped = comp[sr0 : sr1 + 1, sc0 : sc1 + 1]
            sprites.append(
                {
                    "color": color,
                    "bbox": bbox,
                    "count": int(comp.sum()),
                    "mask": cropped,
                    "cells": comp,
                }
            )

    assert len(frames) == 3, "expected three rectangular frames"
    assert len(sprites) == 3, "expected three sprites"
    frame_areas = [frame["interior_area"] for frame in frames]
    sprite_counts = [sprite["count"] for sprite in sprites]
    assert len(set(frame_areas)) == 3, "frame interior areas must be distinct"
    assert len(set(sprite_counts)) == 3, "sprite pixel counts must be distinct"

    outer_bbox = max(frames, key=lambda item: item["interior_area"])["bbox"]
    or0, oc0, or1, oc1 = outer_bbox
    for sprite in sprites:
        comp = sprite["cells"]
        assert not np.any(comp[or0 : or1 + 1, oc0 : oc1 + 1]), "sprites must lie outside all frames"

    return bg, frames, sprites


def _apply_pairing(
    frames_only: np.ndarray,
    frames_sorted: list[dict],
    sprites_sorted: list[dict],
    *,
    bg: int,
) -> np.ndarray:
    out = np.array(frames_only, dtype=int, copy=True)
    for frame, sprite in zip(frames_sorted, sprites_sorted):
        r0, c0, r1, c1 = frame["bbox"]
        target = out[r0 + 1 : r1, c0 + 1 : c1]
        stretched = _stretch_sprite(sprite["mask"], target.shape)
        target[(target == bg) & stretched] = int(sprite["color"])
    return out


def main(grid: np.ndarray) -> np.ndarray:
    """
    Pair each outer sprite with a nested hollow frame by separate area rankings, then stretch the sprite into its matched frame interior.

    Input structure:
    - Black is the background color.
    - Three one-pixel-thick hollow rectangular frames of distinct colors are nested concentrically with gaps between them.
    - Three small 4-connected sprites of distinct colors lie completely outside the outermost frame, and each sprite color appears in exactly one component.

    Rule:
    - Identify the three frames and rank them by interior area from smallest to largest.
    - Identify the three sprites and rank them by foreground-pixel count from smallest to largest.
    - Pair equal ranks. For each pair, clear the original sprite, map the sprite bbox to the matched frame interior bbox, connect mapped positions of 4-neighbor sprite pixels with straight horizontal or vertical segments, then flood-fill enclosed regions of that stretched stroke with the sprite color.
    - Keep all frame pixels unchanged while painting the stretched result inside the matched frame.

    Key insight:
    - The correct destination is determined by two independent rankings: sprite size and frame interior area. Color and spatial proximity are distractors, while the sprite shape itself controls the stretched fill pattern.
    """
    g = np.asarray(grid, dtype=int)
    bg, frames, sprites = _extract_objects(g)
    frames_sorted = sorted(frames, key=lambda item: item["interior_area"])
    sprites_sorted = sorted(sprites, key=lambda item: item["count"])

    out = g.copy()
    for sprite in sprites:
        out[sprite["cells"]] = bg
    return _apply_pairing(out, frames_sorted, sprites_sorted, bg=bg)


def _draw_frame(grid: np.ndarray, bbox: tuple[int, int, int, int], color: int) -> None:
    r0, c0, r1, c1 = bbox
    grid[r0, c0 : c1 + 1] = color
    grid[r1, c0 : c1 + 1] = color
    grid[r0 : r1 + 1, c0] = color
    grid[r0 : r1 + 1, c1] = color


def _trim(mask: np.ndarray) -> np.ndarray:
    r0, c0, r1, c1 = _bbox(mask)
    return mask[r0 : r1 + 1, c0 : c1 + 1].copy()


def _random_transform(mask: np.ndarray) -> np.ndarray:
    out = np.array(mask, dtype=bool, copy=True)
    k = int(RNG.integers(0, 4))
    out = np.rot90(out, k)
    if int(RNG.integers(0, 2)):
        out = np.fliplr(out)
    if int(RNG.integers(0, 2)):
        out = np.flipud(out)
    return _trim(out)


def _sprite_mask_for_rank(rank: int) -> np.ndarray:
    if rank == 0:
        return np.array([[1, 1], [1, 1]], dtype=bool)
    if rank == 1:
        return _random_transform(np.array([[1, 0, 0], [1, 0, 0], [1, 1, 1]], dtype=bool))
    return _random_transform(np.array([[1, 1, 1, 1], [1, 0, 0, 1]], dtype=bool))


def _expanded_overlap(
    bbox_a: tuple[int, int, int, int],
    bbox_b: tuple[int, int, int, int],
    margin: int = 1,
) -> bool:
    ar0, ac0, ar1, ac1 = bbox_a
    br0, bc0, br1, bc1 = bbox_b
    return not (
        ar1 + margin < br0
        or br1 + margin < ar0
        or ac1 + margin < bc0
        or bc1 + margin < ac0
    )


def _place_sprites(
    shape: tuple[int, int],
    outer_bbox: tuple[int, int, int, int],
    placed: list[tuple[int, int, int, int]],
    grid_shape: tuple[int, int],
) -> tuple[int, int] | None:
    h, w = shape
    gh, gw = grid_shape
    or0, oc0, or1, oc1 = outer_bbox

    regions: list[tuple[str, tuple[int, int], tuple[int, int]]] = []
    if or0 - h - 1 >= 0:
        regions.append(("top", (0, or0 - h - 1), (0, gw - w)))
    if or1 + h + 1 < gh:
        regions.append(("bottom", (or1 + 2, gh - h), (0, gw - w)))
    if oc0 - w - 1 >= 0:
        regions.append(("left", (0, gh - h), (0, oc0 - w - 1)))
    if oc1 + w + 1 < gw:
        regions.append(("right", (0, gh - h), (oc1 + 2, gw - w)))
    if not regions:
        return None

    for _ in range(250):
        _name, (rlo, rhi), (clo, chi) = regions[int(RNG.integers(0, len(regions)))]
        if rlo > rhi or clo > chi:
            continue
        r0 = int(RNG.integers(rlo, rhi + 1))
        c0 = int(RNG.integers(clo, chi + 1))
        bbox = (r0, c0, r0 + h - 1, c0 + w - 1)
        if any(_expanded_overlap(bbox, other, margin=1) for other in placed):
            continue
        if _expanded_overlap(bbox, outer_bbox, margin=1):
            continue
        return r0, c0
    return None


def _add_sprite(grid: np.ndarray, mask: np.ndarray, top: int, left: int, color: int) -> None:
    h, w = mask.shape
    view = grid[top : top + h, left : left + w]
    assert np.all(view[mask] == BG), "sprite placement must land on background"
    view[mask] = color


def _build_expected(frames_only: np.ndarray, frames: list[dict], sprites: list[dict]) -> np.ndarray:
    ordered_frames = sorted(frames, key=lambda item: item["interior_area"])
    ordered_sprites = sorted(sprites, key=lambda item: item["count"])
    return _apply_pairing(frames_only, ordered_frames, ordered_sprites, bg=BG)


def _placement_cells(frame: dict, sprite: dict) -> np.ndarray:
    r0, c0, r1, c1 = frame["bbox"]
    stretched = _stretch_sprite(sprite["mask"], (r1 - r0 - 1, c1 - c0 - 1))
    placed = np.zeros((r1 - r0 + 1, c1 - c0 + 1), dtype=bool)
    placed[1:-1, 1:-1] = stretched
    return placed


def _validate_instance(inp: np.ndarray, expected: np.ndarray, frames: list[dict], sprites: list[dict]) -> bool:
    if not np.array_equal(main(inp), expected):
        return False

    parsed_bg, parsed_frames, parsed_sprites = _extract_objects(inp)
    if parsed_bg != BG:
        return False
    if sorted(frame["interior_area"] for frame in parsed_frames) != sorted(frame["interior_area"] for frame in frames):
        return False
    if sorted(sprite["count"] for sprite in parsed_sprites) != sorted(sprite["count"] for sprite in sprites):
        return False

    frames_only = inp.copy()
    for sprite in parsed_sprites:
        frames_only[sprite["cells"]] = BG

    true_frames = sorted(frames, key=lambda item: item["interior_area"])
    true_sprites = sorted(sprites, key=lambda item: item["count"])
    uniques = set()
    for perm in permutations(range(3)):
        candidate = _apply_pairing(
            frames_only,
            true_frames,
            [true_sprites[idx] for idx in perm],
            bg=BG,
        )
        uniques.add(candidate.tobytes())
    if len(uniques) != 6:
        return False

    smaller_bboxes = [frame["bbox"] for frame in true_frames]
    for idx, (frame, sprite) in enumerate(zip(true_frames, true_sprites)):
        if idx == 0:
            continue
        local = _placement_cells(frame, sprite)
        fr0, fc0, _fr1, _fc1 = frame["bbox"]
        hit = np.zeros_like(inp, dtype=bool)
        hit[fr0 : fr0 + local.shape[0], fc0 : fc0 + local.shape[1]] = local
        for sr0, sc0, sr1, sc1 in smaller_bboxes[:idx]:
            if np.any(hit[sr0 : sr1 + 1, sc0 : sc1 + 1]):
                return False

    for sprite in sprites:
        color = int(sprite["color"])
        if np.any(expected[sprite["cells"]] == color):
            return False

    return True


def _gen_attempt() -> np.ndarray | None:
    gh = int(RNG.integers(20, 28))
    gw = int(RNG.integers(20, 28))

    inner_h = int(RNG.integers(6, 9))
    inner_w = int(RNG.integers(6, 9))
    gap_im_r = int(RNG.integers(3, 6))
    gap_im_c = int(RNG.integers(3, 6))
    gap_mo_r = int(RNG.integers(3, 6))
    gap_mo_c = int(RNG.integers(3, 6))

    mid_h = inner_h + 2 * gap_im_r
    mid_w = inner_w + 2 * gap_im_c
    outer_h = mid_h + 2 * gap_mo_r
    outer_w = mid_w + 2 * gap_mo_c
    if outer_h > gh - 2 or outer_w > gw - 2:
        return None

    outer_r0 = int(RNG.integers(1, gh - outer_h))
    outer_c0 = int(RNG.integers(1, gw - outer_w))
    outer_r1 = outer_r0 + outer_h - 1
    outer_c1 = outer_c0 + outer_w - 1
    mid_r0, mid_c0 = outer_r0 + gap_mo_r, outer_c0 + gap_mo_c
    mid_r1, mid_c1 = outer_r1 - gap_mo_r, outer_c1 - gap_mo_c
    inner_r0, inner_c0 = mid_r0 + gap_im_r, mid_c0 + gap_im_c
    inner_r1, inner_c1 = mid_r1 - gap_im_r, mid_c1 - gap_im_c

    frame_bboxes = [
        (inner_r0, inner_c0, inner_r1, inner_c1),
        (mid_r0, mid_c0, mid_r1, mid_c1),
        (outer_r0, outer_c0, outer_r1, outer_c1),
    ]
    frame_color_order = list(RNG.permutation(FRAME_COLORS))
    grid = np.full((gh, gw), BG, dtype=int)
    frames: list[dict] = []
    for color, bbox in zip(frame_color_order, frame_bboxes):
        _draw_frame(grid, bbox, int(color))
        r0, c0, r1, c1 = bbox
        frames.append(
            {
                "color": int(color),
                "bbox": bbox,
                "interior_area": (r1 - r0 - 1) * (c1 - c0 - 1),
            }
        )

    sprite_color_order = list(RNG.permutation(SPRITE_COLORS))
    sprite_counts = [4, 5, 6]
    sprite_masks = [_sprite_mask_for_rank(idx) for idx in range(3)]

    placed_boxes: list[tuple[int, int, int, int]] = [frame_bboxes[-1]]
    sprites: list[dict] = []
    input_grid = grid.copy()
    for color, count, mask in zip(sprite_color_order, sprite_counts, sprite_masks):
        pos = _place_sprites(mask.shape, frame_bboxes[-1], placed_boxes, (gh, gw))
        if pos is None:
            return None
        top, left = pos
        _add_sprite(input_grid, mask, top, left, int(color))
        bbox = (top, left, top + mask.shape[0] - 1, left + mask.shape[1] - 1)
        placed_boxes.append(bbox)
        sprite_cells = np.zeros((gh, gw), dtype=bool)
        sprite_cells[top : top + mask.shape[0], left : left + mask.shape[1]] = mask
        sprites.append(
            {
                "color": int(color),
                "count": int(count),
                "bbox": bbox,
                "mask": mask.copy(),
                "cells": sprite_cells,
            }
        )

    expected = _build_expected(grid, frames, sprites)
    if not _validate_instance(input_grid, expected, frames, sprites):
        return None
    return input_grid


def generate_input() -> np.ndarray:
    for _ in range(MAX_ATTEMPTS):
        grid = _gen_attempt()
        if grid is not None:
            return grid
    raise RuntimeError(f"failed to generate a valid input after {MAX_ATTEMPTS} attempts")
