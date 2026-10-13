from __future__ import annotations

from collections import deque
import random

import numpy as np


BG = 0
YELLOW = 4
GREY = 5
OUTER_FRAME = 9
FRAME_COLORS = (1, 2, 3, 6, 7, 8)
GEN_ATTEMPTS = 600


def _bbox(mask: np.ndarray) -> tuple[int, int, int, int] | None:
    coords = np.argwhere(mask)
    if coords.size == 0:
        return None
    r0, c0 = coords.min(axis=0)
    r1, c1 = coords.max(axis=0)
    return int(r0), int(c0), int(r1), int(c1)


def _components(mask: np.ndarray) -> list[np.ndarray]:
    h, w = mask.shape
    seen = np.zeros_like(mask, dtype=bool)
    out: list[np.ndarray] = []
    for r in range(h):
        for c in range(w):
            if not mask[r, c] or seen[r, c]:
                continue
            q = deque([(r, c)])
            seen[r, c] = True
            coords: list[tuple[int, int]] = []
            while q:
                rr, cc = q.popleft()
                coords.append((rr, cc))
                for nr, nc in ((rr - 1, cc), (rr + 1, cc), (rr, cc - 1), (rr, cc + 1)):
                    if 0 <= nr < h and 0 <= nc < w and mask[nr, nc] and not seen[nr, nc]:
                        seen[nr, nc] = True
                        q.append((nr, nc))
            out.append(np.array(coords, dtype=int))
    return out


def _frame_mask_from_bbox(shape: tuple[int, int], bbox: tuple[int, int, int, int]) -> np.ndarray:
    r0, c0, r1, c1 = bbox
    mask = np.zeros(shape, dtype=bool)
    mask[r0, c0 : c1 + 1] = True
    mask[r1, c0 : c1 + 1] = True
    mask[r0 : r1 + 1, c0] = True
    mask[r0 : r1 + 1, c1] = True
    return mask


def _find_outer_frame_bbox(grid: np.ndarray) -> tuple[int, int, int, int]:
    mask = grid == OUTER_FRAME
    bbox = _bbox(mask)
    assert bbox is not None, "missing outer frame"
    r0, c0, r1, c1 = bbox
    assert r1 - r0 + 1 >= 3 and c1 - c0 + 1 >= 3, "outer frame too small"
    expected = _frame_mask_from_bbox(grid.shape, bbox)
    assert np.array_equal(mask, expected), "outer frame must be a unique complete rectangle"
    return bbox


def _detect_inner_frames(crop: np.ndarray) -> list[dict]:
    colors = [int(c) for c in np.unique(crop) if int(c) not in (BG, YELLOW, GREY, OUTER_FRAME)]
    frames: list[dict] = []
    for color in colors:
        for coords in _components(crop == color):
            rr = coords[:, 0]
            cc = coords[:, 1]
            bbox = (int(rr.min()), int(cc.min()), int(rr.max()), int(cc.max()))
            r0, c0, r1, c1 = bbox
            if r1 - r0 + 1 < 3 or c1 - c0 + 1 < 3:
                continue
            local = np.zeros((r1 - r0 + 1, c1 - c0 + 1), dtype=bool)
            local[rr - r0, cc - c0] = True
            expected = _frame_mask_from_bbox(local.shape, (0, 0, local.shape[0] - 1, local.shape[1] - 1))
            if not np.array_equal(local, expected):
                continue
            center = ((r0 + r1) // 2, (c0 + c1) // 2)
            marker_count = int(np.count_nonzero(crop[r0 + 1 : r1, c0 + 1 : c1] == YELLOW))
            frames.append(
                {
                    "color": color,
                    "bbox": bbox,
                    "center": center,
                    "count": marker_count,
                    "mask": _frame_mask_from_bbox(crop.shape, bbox),
                }
            )
    assert 3 <= len(frames) <= 4, "expected 3-4 inner hollow frames"
    counts = [frame["count"] for frame in frames]
    assert all(1 <= count <= 5 for count in counts), "marker counts must be 1..5"
    assert len(counts) == len(set(counts)), "marker counts must be distinct"
    return frames


def _protected_mask(crop: np.ndarray, frames: list[dict]) -> np.ndarray:
    protected = (crop == OUTER_FRAME) | (crop == YELLOW)
    for frame in frames:
        protected |= frame["mask"]
    return protected


def _ordered_frames(frames: list[dict], mode: str) -> list[dict]:
    if mode == "asc":
        return sorted(frames, key=lambda frame: frame["count"])
    if mode == "desc":
        return sorted(frames, key=lambda frame: frame["count"], reverse=True)
    if mode == "scan":
        return sorted(frames, key=lambda frame: frame["bbox"])
    raise ValueError(f"unknown order mode: {mode}")


def _render_from_crop(
    crop: np.ndarray,
    frames: list[dict],
    *,
    order_mode: str = "asc",
    alternate: bool = True,
    overwrite: bool = True,
    use_marker_count: bool = True,
) -> np.ndarray:
    out = crop.copy()
    protected = _protected_mask(crop, frames)
    for frame in _ordered_frames(frames, order_mode):
        center_r, center_c = frame["center"]
        length = frame["count"] if use_marker_count else 1
        for offset in range(length):
            rr = center_r + offset
            if rr >= crop.shape[0]:
                break
            if protected[rr, center_c]:
                continue
            if not overwrite and out[rr, center_c] != crop[rr, center_c]:
                continue
            if alternate:
                out[rr, center_c] = frame["color"] if offset % 2 == 0 else GREY
            else:
                out[rr, center_c] = frame["color"]
    return out


def main(grid: np.ndarray) -> np.ndarray:
    """Crop to the unique maroon frame, count markers in each inner hollow frame, then draw sorted alternating downward rays.

    Input structure:
    - The grid has black background, one unique complete rectangular frame in color 9, and 3-4 inner colored hollow rectangles.
    - Yellow marker pixels lie inside those inner rectangles, with distinct marker counts per frame.
    - Pixels outside the color-9 frame are distractors and must be ignored.

    Rule:
    - Find the unique complete color-9 rectangular frame and crop the grid to that bounding box, keeping the frame itself.
    - Inside the crop, count the yellow markers inside each colored hollow rectangle, then sort the rectangles by that count ascending.
    - Process frames in that sorted order: from each frame center, paint a downward ray of exactly that many offsets, using the frame color on even offsets and grey on odd offsets.
    - Later rays overwrite earlier rays, but rays never replace the inner frame borders, the outer color-9 border, or the yellow markers.

    Key insight:
    - The marker count has three linked roles at once: it decides processing order, ray length, and therefore overwrite priority when vertical rays collide.
    """
    g = np.asarray(grid, dtype=int)
    assert g.ndim == 2, "expected a 2D grid"
    r0, c0, r1, c1 = _find_outer_frame_bbox(g)
    crop = g[r0 : r1 + 1, c0 : c1 + 1].copy()
    frames = _detect_inner_frames(crop)
    return _render_from_crop(crop, frames, order_mode="asc", alternate=True, overwrite=True, use_marker_count=True)


def _valid_dims_for_count(count: int, *, force_height: int | None = None) -> list[tuple[int, int]]:
    dims: list[tuple[int, int]] = []
    heights = [force_height] if force_height is not None else [3, 5, 7]
    for h in heights:
        for w in (3, 5, 7):
            if h is None:
                continue
            if (h - 2) * (w - 2) >= count:
                dims.append((h, w))
    return dims


def _bbox_with_gap_ok(new_box: tuple[int, int, int, int], boxes: list[tuple[int, int, int, int]], gap: int) -> bool:
    nr0, nc0, nr1, nc1 = new_box
    for r0, c0, r1, c1 in boxes:
        if not (nr1 + gap < r0 or r1 + gap < nr0 or nc1 + gap < c0 or c1 + gap < nc0):
            return False
    return True


def _sample_counts(n_frames: int) -> tuple[int, int, list[int]]:
    upper = 5
    lower = random.choice((1, 2, 3))
    pool = [1, 2, 3, 4]
    pool.remove(lower)
    others = random.sample(pool, n_frames - 2)
    return upper, lower, others


def _place_scene_in_crop(crop_h: int, crop_w: int, n_frames: int) -> list[dict] | None:
    upper_count, lower_count, other_counts = _sample_counts(n_frames)
    colors = random.sample(FRAME_COLORS, n_frames)

    upper = {
        "color": colors[0],
        "count": upper_count,
        "size": (3, 7),
    }
    lower_width = random.choice([w for _h, w in _valid_dims_for_count(lower_count, force_height=3)])
    lower = {
        "color": colors[1],
        "count": lower_count,
        "size": (3, lower_width),
    }
    others = []
    for color, count in zip(colors[2:], other_counts):
        dims = _valid_dims_for_count(count)
        others.append({"color": color, "count": count, "size": random.choice(dims)})

    center_min = 1 + max(upper["size"][1] // 2, lower["size"][1] // 2)
    center_max = crop_w - 2 - max(upper["size"][1] // 2, lower["size"][1] // 2)
    if center_min > center_max:
        return None
    shared_c = random.randint(center_min, center_max)

    upper_top_max = crop_h - 7 - lower_count
    if upper_top_max < 1:
        return None
    upper_top = random.randint(1, upper_top_max)
    lower_top = upper_top + 4

    upper_h, upper_w = upper["size"]
    lower_h, lower_w = lower["size"]
    upper_box = (upper_top, shared_c - upper_w // 2, upper_top + upper_h - 1, shared_c + upper_w // 2)
    lower_box = (lower_top, shared_c - lower_w // 2, lower_top + lower_h - 1, shared_c + lower_w // 2)
    if upper_box[1] < 1 or upper_box[3] > crop_w - 2 or lower_box[1] < 1 or lower_box[3] > crop_w - 2:
        return None

    frames = []
    for spec, bbox in ((upper, upper_box), (lower, lower_box)):
        r0, c0, r1, c1 = bbox
        center = ((r0 + r1) // 2, (c0 + c1) // 2)
        frames.append(
            {
                "color": spec["color"],
                "count": spec["count"],
                "bbox": bbox,
                "center": center,
            }
        )

    placed_boxes = [upper_box, lower_box]
    for spec in others:
        h, w = spec["size"]
        placed = False
        for _ in range(250):
            r0 = random.randint(1, crop_h - h - 1)
            c0 = random.randint(1, crop_w - w - 1)
            box = (r0, c0, r0 + h - 1, c0 + w - 1)
            center = ((box[0] + box[2]) // 2, (box[1] + box[3]) // 2)
            if center[1] == shared_c:
                continue
            if center[0] + spec["count"] >= crop_h - 1:
                continue
            if not _bbox_with_gap_ok(box, placed_boxes, gap=1):
                continue
            placed_boxes.append(box)
            frames.append(
                {
                    "color": spec["color"],
                    "count": spec["count"],
                    "bbox": box,
                    "center": center,
                }
            )
            placed = True
            break
        if not placed:
            return None

    for frame in frames:
        if frame["center"][0] + frame["count"] >= crop_h - 1:
            return None
    return frames


def _fill_crop(crop_h: int, crop_w: int, frames: list[dict]) -> np.ndarray:
    crop = np.full((crop_h, crop_w), BG, dtype=int)
    crop[0, :] = OUTER_FRAME
    crop[-1, :] = OUTER_FRAME
    crop[:, 0] = OUTER_FRAME
    crop[:, -1] = OUTER_FRAME
    for frame in frames:
        r0, c0, r1, c1 = frame["bbox"]
        crop[r0, c0 : c1 + 1] = frame["color"]
        crop[r1, c0 : c1 + 1] = frame["color"]
        crop[r0 : r1 + 1, c0] = frame["color"]
        crop[r0 : r1 + 1, c1] = frame["color"]
        interior = [(r, c) for r in range(r0 + 1, r1) for c in range(c0 + 1, c1)]
        center_r, center_c = frame["center"]
        safe = [(r, c) for r, c in interior if not (c == center_c and center_r <= r < center_r + frame["count"])]
        choices = safe if len(safe) >= frame["count"] else interior
        for r, c in random.sample(choices, frame["count"]):
            crop[r, c] = YELLOW
    return crop


def _embed_crop(crop: np.ndarray) -> np.ndarray:
    crop_h, crop_w = crop.shape
    full_h = random.randint(crop_h + 2, min(30, crop_h + 6))
    full_w = random.randint(crop_w + 2, min(30, crop_w + 6))
    top = random.randint(1, full_h - crop_h - 1)
    left = random.randint(1, full_w - crop_w - 1)
    grid = np.full((full_h, full_w), BG, dtype=int)
    grid[top : top + crop_h, left : left + crop_w] = crop

    outside = [(r, c) for r in range(full_h) for c in range(full_w) if not (top <= r < top + crop_h and left <= c < left + crop_w)]
    random.shuffle(outside)
    for r, c in outside[: random.randint(1, min(10, len(outside)))]:
        grid[r, c] = random.choice((1, 2, 3, 4, 5, 6, 7, 8))
    return grid


def _scene_semantics_ok(grid: np.ndarray) -> bool:
    try:
        r0, c0, r1, c1 = _find_outer_frame_bbox(grid)
        crop = grid[r0 : r1 + 1, c0 : c1 + 1].copy()
        frames = _detect_inner_frames(crop)
        actual = _render_from_crop(crop, frames, order_mode="asc", alternate=True, overwrite=True, use_marker_count=True)
        if not np.array_equal(main(grid), actual):
            return False
        if np.array_equal(actual, crop):
            return False
        if np.array_equal(actual, _render_from_crop(crop, frames, order_mode="desc", alternate=True, overwrite=True, use_marker_count=True)):
            return False
        if np.array_equal(actual, _render_from_crop(crop, frames, order_mode="asc", alternate=True, overwrite=False, use_marker_count=True)):
            return False
        if np.array_equal(actual, _render_from_crop(crop, frames, order_mode="asc", alternate=False, overwrite=True, use_marker_count=True)):
            return False
        if np.array_equal(actual, _render_from_crop(crop, frames, order_mode="asc", alternate=True, overwrite=True, use_marker_count=False)):
            return False
        changed = actual != crop
        if int(np.count_nonzero(changed)) < 3:
            return False
        if not np.any(actual[changed] == GREY):
            return False
        if not np.any((actual != crop) & (actual != GREY)):
            return False
        return True
    except Exception:
        return False


def _gen_attempt() -> np.ndarray | None:
    crop_h = random.randint(16, 22)
    crop_w = random.randint(16, 22)
    n_frames = random.choice((3, 4))
    frames = _place_scene_in_crop(crop_h, crop_w, n_frames)
    if frames is None:
        return None
    crop = _fill_crop(crop_h, crop_w, frames)
    grid = _embed_crop(crop)
    if not _scene_semantics_ok(grid):
        return None
    return grid


def generate_input() -> np.ndarray:
    for _ in range(GEN_ATTEMPTS):
        grid = _gen_attempt()
        if grid is not None:
            return grid
    raise RuntimeError("failed to generate a valid input for puzzle 9")
