from __future__ import annotations

from collections import deque
import random

import numpy as np


GRAY = 5
SIDES = ("top", "bottom", "left", "right")
DIRS4 = ((-1, 0), (1, 0), (0, -1), (0, 1))
GENERATE_RETRIES = 300


def _get_bg(grid: np.ndarray) -> int:
    colors, counts = np.unique(grid, return_counts=True)
    return int(colors[np.argmax(counts)])


def _components_by_color(grid: np.ndarray, bg: int) -> list[dict]:
    h, w = grid.shape
    seen = np.zeros((h, w), dtype=bool)
    comps: list[dict] = []

    for r in range(h):
        for c in range(w):
            if seen[r, c] or int(grid[r, c]) == bg:
                continue

            color = int(grid[r, c])
            q = deque([(r, c)])
            seen[r, c] = True
            coords: list[tuple[int, int]] = []

            while q:
                rr, cc = q.popleft()
                coords.append((rr, cc))
                for dr, dc in DIRS4:
                    nr, nc = rr + dr, cc + dc
                    if not (0 <= nr < h and 0 <= nc < w):
                        continue
                    if seen[nr, nc] or int(grid[nr, nc]) != color:
                        continue
                    seen[nr, nc] = True
                    q.append((nr, nc))

            arr = np.array(coords, dtype=int)
            r0, c0 = arr.min(axis=0)
            r1, c1 = arr.max(axis=0)
            mask = np.zeros((r1 - r0 + 1, c1 - c0 + 1), dtype=bool)
            mask[arr[:, 0] - r0, arr[:, 1] - c0] = True
            comps.append(
                {
                    "color": color,
                    "coords": arr,
                    "bbox": (int(r0), int(c0), int(r1), int(c1)),
                    "mask": mask,
                }
            )

    return comps


def _detect_frame(mask: np.ndarray) -> dict | None:
    h, w = mask.shape
    if h < 4 or w < 4:
        return None

    border = np.zeros_like(mask, dtype=bool)
    border[0, :] = True
    border[-1, :] = True
    border[:, 0] = True
    border[:, -1] = True

    if np.any(mask & ~border):
        return None

    missing = border & ~mask
    if not missing.any():
        return None

    side_positions = {
        "top": np.flatnonzero(missing[0]),
        "bottom": np.flatnonzero(missing[-1]),
        "left": np.flatnonzero(missing[:, 0]),
        "right": np.flatnonzero(missing[:, -1]),
    }
    active = [side for side, pos in side_positions.items() if pos.size]
    if len(active) != 1:
        return None

    side = active[0]
    pos = side_positions[side]
    limit = w if side in ("top", "bottom") else h
    if pos.size not in (2, 3):
        return None
    if pos[0] == 0 or pos[-1] == limit - 1:
        return None
    if np.any(np.diff(pos) != 1):
        return None

    expected = border.copy()
    if side == "top":
        expected[0, pos[0] : pos[-1] + 1] = False
    elif side == "bottom":
        expected[-1, pos[0] : pos[-1] + 1] = False
    elif side == "left":
        expected[pos[0] : pos[-1] + 1, 0] = False
    else:
        expected[pos[0] : pos[-1] + 1, -1] = False

    if not np.array_equal(mask, expected):
        return None

    return {
        "side": side,
        "gap_start": int(pos[0]),
        "gap_len": int(pos.size),
    }


def _contact_face(mask: np.ndarray, side: str) -> tuple[int, int]:
    if side == "top":
        pos = np.flatnonzero(mask[-1])
    elif side == "bottom":
        pos = np.flatnonzero(mask[0])
    elif side == "left":
        pos = np.flatnonzero(mask[:, -1])
    else:
        pos = np.flatnonzero(mask[:, 0])

    if pos.size == 0 or np.any(np.diff(pos) != 1):
        raise AssertionError("blob contact face must be contiguous")
    return int(pos[0]), int(pos.size)


def _find_ray_blob_id(
    grid: np.ndarray,
    bg: int,
    frame_id: int,
    frame: dict,
    coord_to_comp: dict[tuple[int, int], int],
    frame_ids: set[int],
) -> int:
    h, w = grid.shape
    r0, c0, r1, c1 = frame["component"]["bbox"]
    gap_start = int(frame["gap_start"])
    gap_len = int(frame["gap_len"])
    side = frame["side"]

    if side == "top":
        cols = range(c0 + gap_start, c0 + gap_start + gap_len)
        for rr in range(r0 - 1, -1, -1):
            ids = {coord_to_comp[(rr, cc)] for cc in cols if int(grid[rr, cc]) != bg}
            ids.discard(frame_id)
            if ids:
                if len(ids) != 1:
                    raise AssertionError("ray hit multiple objects")
                blob_id = next(iter(ids))
                if blob_id in frame_ids:
                    raise AssertionError("ray must hit a blob, not another frame")
                return int(blob_id)
    elif side == "bottom":
        cols = range(c0 + gap_start, c0 + gap_start + gap_len)
        for rr in range(r1 + 1, h):
            ids = {coord_to_comp[(rr, cc)] for cc in cols if int(grid[rr, cc]) != bg}
            ids.discard(frame_id)
            if ids:
                if len(ids) != 1:
                    raise AssertionError("ray hit multiple objects")
                blob_id = next(iter(ids))
                if blob_id in frame_ids:
                    raise AssertionError("ray must hit a blob, not another frame")
                return int(blob_id)
    elif side == "left":
        rows = range(r0 + gap_start, r0 + gap_start + gap_len)
        for cc in range(c0 - 1, -1, -1):
            ids = {coord_to_comp[(rr, cc)] for rr in rows if int(grid[rr, cc]) != bg}
            ids.discard(frame_id)
            if ids:
                if len(ids) != 1:
                    raise AssertionError("ray hit multiple objects")
                blob_id = next(iter(ids))
                if blob_id in frame_ids:
                    raise AssertionError("ray must hit a blob, not another frame")
                return int(blob_id)
    else:
        rows = range(r0 + gap_start, r0 + gap_start + gap_len)
        for cc in range(c1 + 1, w):
            ids = {coord_to_comp[(rr, cc)] for rr in rows if int(grid[rr, cc]) != bg}
            ids.discard(frame_id)
            if ids:
                if len(ids) != 1:
                    raise AssertionError("ray hit multiple objects")
                blob_id = next(iter(ids))
                if blob_id in frame_ids:
                    raise AssertionError("ray must hit a blob, not another frame")
                return int(blob_id)

    raise AssertionError("every frame gap must see a blob on its outward ray")


def _analyze_scene(grid: np.ndarray) -> tuple[int, list[dict]]:
    bg = _get_bg(grid)
    comps = _components_by_color(grid, bg)
    if not comps:
        raise AssertionError("expected frames and blobs")

    coord_to_comp: dict[tuple[int, int], int] = {}
    for idx, comp in enumerate(comps):
        for r, c in comp["coords"]:
            coord_to_comp[(int(r), int(c))] = int(idx)

    frames: list[dict] = []
    frame_ids: set[int] = set()
    for idx, comp in enumerate(comps):
        info = _detect_frame(comp["mask"])
        if info is None:
            continue
        info = dict(info)
        info["id"] = int(idx)
        info["component"] = comp
        frames.append(info)
        frame_ids.add(int(idx))

    if not frames:
        raise AssertionError("expected at least one frame")

    scene: list[dict] = []
    used_blobs: set[int] = set()
    for frame in sorted(frames, key=lambda item: item["component"]["bbox"][:2]):
        blob_id = _find_ray_blob_id(grid, bg, frame["id"], frame, coord_to_comp, frame_ids)
        if blob_id in used_blobs:
            raise AssertionError("a blob cannot serve multiple frames")

        blob = comps[blob_id]
        face_offset, face_len = _contact_face(blob["mask"], frame["side"])
        if face_len != int(frame["gap_len"]):
            raise AssertionError("blob face length must match the gap")

        fr0, fc0, fr1, fc1 = frame["component"]["bbox"]
        fh = fr1 - fr0 + 1
        fw = fc1 - fc0 + 1
        bh, bw = blob["mask"].shape
        gap_start = int(frame["gap_start"])

        if frame["side"] == "top":
            if blob["bbox"][2] >= fr0:
                raise AssertionError("top blob must sit above the frame")
            rel_c = gap_start - face_offset
            if not (1 <= rel_c and rel_c + bw <= fw - 1 and bh <= fh - 2):
                raise AssertionError("mirrored top blob must fit inside the frame")
        elif frame["side"] == "bottom":
            if blob["bbox"][0] <= fr1:
                raise AssertionError("bottom blob must sit below the frame")
            rel_c = gap_start - face_offset
            if not (1 <= rel_c and rel_c + bw <= fw - 1 and bh <= fh - 2):
                raise AssertionError("mirrored bottom blob must fit inside the frame")
        elif frame["side"] == "left":
            if blob["bbox"][3] >= fc0:
                raise AssertionError("left blob must sit left of the frame")
            rel_r = gap_start - face_offset
            if not (1 <= rel_r and rel_r + bh <= fh - 1 and bw <= fw - 2):
                raise AssertionError("mirrored left blob must fit inside the frame")
        else:
            if blob["bbox"][1] <= fc1:
                raise AssertionError("right blob must sit right of the frame")
            rel_r = gap_start - face_offset
            if not (1 <= rel_r and rel_r + bh <= fh - 1 and bw <= fw - 2):
                raise AssertionError("mirrored right blob must fit inside the frame")

        scene.append(
            {
                "frame": frame,
                "blob": blob,
                "face_offset": int(face_offset),
            }
        )
        used_blobs.add(blob_id)

    return bg, scene


def main(grid: np.ndarray) -> np.ndarray:
    """
    Mirror the ray-selected outside blob into each gapped frame and recolor the frame to that blob's color.

    Input structure:
    - The background is the dominant color.
    - Several monochrome hollow rectangular frames sit on the background; each frame has exactly one 2-3 cell gap on one side of its 1-cell border.
    - Outside the frames are separate monochrome solid blobs, with each relevant blob lying on the outward normal ray of one frame gap.

    Rule:
    - Detect which side of each frame contains the contiguous gap.
    - Follow the outward ray through that gap and pick the nearest blob hit by the ray.
    - Mirror the blob across the open side, place the mirrored shape just inside the frame against the gap, recolor the whole frame border to the blob's color, and erase the original blob back to background.
    - Leave all other background cells unchanged.

    Key insight: The missing side segment determines the frame-blob pairing, and the transferred interior shape is a mirrored copy across that side rather than a direct translation.
    """
    g = np.asarray(grid, dtype=int)
    bg, scene = _analyze_scene(g)
    out = g.copy()

    for item in scene:
        frame = item["frame"]
        blob = item["blob"]
        color = int(blob["color"])
        face_offset = int(item["face_offset"])
        fr0, fc0, fr1, fc1 = frame["component"]["bbox"]
        gap_start = int(frame["gap_start"])
        side = frame["side"]

        out[blob["coords"][:, 0], blob["coords"][:, 1]] = bg
        out[frame["component"]["coords"][:, 0], frame["component"]["coords"][:, 1]] = color

        if side in ("top", "bottom"):
            mirrored = np.flipud(blob["mask"])
            dest_c = fc0 + gap_start - face_offset
            dest_r = fr0 + 1 if side == "top" else fr1 - mirrored.shape[0]
        else:
            mirrored = np.fliplr(blob["mask"])
            dest_r = fr0 + gap_start - face_offset
            dest_c = fc0 + 1 if side == "left" else fc1 - mirrored.shape[1]

        rows, cols = np.where(mirrored)
        out[dest_r + rows, dest_c + cols] = color

    return out


def _grow_intervals(span: int, depth: int, near_interval: tuple[int, int]) -> list[tuple[int, int]]:
    intervals = [near_interval]
    for _ in range(depth - 1):
        prev_left, prev_right = intervals[-1]
        left = max(0, prev_left + random.choice((-1, 0, 1)))
        right = min(span - 1, prev_right + random.choice((-1, 0, 1)))
        left = min(left, prev_right)
        right = max(right, prev_left)
        if left > right:
            left, right = prev_left, prev_right
        intervals.append((int(left), int(right)))
    return intervals


def _mask_from_row_intervals(intervals: list[tuple[int, int]], width: int) -> np.ndarray:
    mask = np.zeros((len(intervals), width), dtype=bool)
    for r, (left, right) in enumerate(intervals):
        mask[r, left : right + 1] = True
    return mask


def _mask_from_col_intervals(intervals: list[tuple[int, int]], height: int) -> np.ndarray:
    mask = np.zeros((height, len(intervals)), dtype=bool)
    for c, (top, bottom) in enumerate(intervals):
        mask[top : bottom + 1, c] = True
    return mask


def _frame_mask(height: int, width: int, side: str, gap_start: int, gap_len: int) -> np.ndarray:
    mask = np.zeros((height, width), dtype=bool)
    mask[0, :] = True
    mask[-1, :] = True
    mask[:, 0] = True
    mask[:, -1] = True
    if side == "top":
        mask[0, gap_start : gap_start + gap_len] = False
    elif side == "bottom":
        mask[-1, gap_start : gap_start + gap_len] = False
    elif side == "left":
        mask[gap_start : gap_start + gap_len, 0] = False
    else:
        mask[gap_start : gap_start + gap_len, -1] = False
    return mask


def _make_spec(side: str, color: int) -> dict | None:
    gap_len = random.choice((2, 3))

    if side in ("top", "bottom"):
        frame_h = random.randint(5, 8)
        frame_w = random.randint(max(6, gap_len + 4), 10)
        gap_start = random.randint(1, frame_w - gap_len - 1)
        inner_h = frame_h - 2
        inner_w = frame_w - 2

        width_options: list[tuple[int, int, int]] = []
        for blob_w in range(gap_len, min(inner_w, gap_len + 2) + 1):
            low = max(0, gap_start + blob_w - frame_w + 1)
            high = min(blob_w - gap_len, gap_start - 1)
            if low <= high:
                width_options.append((blob_w, low, high))
        if not width_options:
            return None

        blob_w, low, high = random.choice(width_options)
        face_offset = random.randint(low, high)
        max_depth = max(2, inner_h - 1)
        blob_h = random.randint(2, max_depth)
        near = (face_offset, face_offset + gap_len - 1)
        intervals = _grow_intervals(blob_w, blob_h, near)
        rows = list(reversed(intervals)) if side == "top" else intervals
        blob_mask = _mask_from_row_intervals(rows, blob_w)
    else:
        frame_h = random.randint(max(6, gap_len + 4), 10)
        frame_w = random.randint(5, 8)
        gap_start = random.randint(1, frame_h - gap_len - 1)
        inner_h = frame_h - 2
        inner_w = frame_w - 2

        height_options: list[tuple[int, int, int]] = []
        for blob_h in range(gap_len, min(inner_h, gap_len + 2) + 1):
            low = max(0, gap_start + blob_h - frame_h + 1)
            high = min(blob_h - gap_len, gap_start - 1)
            if low <= high:
                height_options.append((blob_h, low, high))
        if not height_options:
            return None

        blob_h, low, high = random.choice(height_options)
        face_offset = random.randint(low, high)
        max_depth = max(2, inner_w - 1)
        blob_w = random.randint(2, max_depth)
        near = (face_offset, face_offset + gap_len - 1)
        intervals = _grow_intervals(blob_h, blob_w, near)
        cols = list(reversed(intervals)) if side == "left" else intervals
        blob_mask = _mask_from_col_intervals(cols, blob_h)

    frame_mask = _frame_mask(frame_h, frame_w, side, gap_start, gap_len)
    return {
        "side": side,
        "color": int(color),
        "gap_len": int(gap_len),
        "gap_start": int(gap_start),
        "face_offset": int(face_offset),
        "gap_dist": random.randint(1, 3),
        "frame_mask": frame_mask,
        "blob_mask": blob_mask,
    }


def _placement_ranges(height: int, width: int, spec: dict, margin: int = 1) -> tuple[int, int, int, int] | None:
    frame_h, frame_w = spec["frame_mask"].shape
    blob_h, blob_w = spec["blob_mask"].shape
    gap_start = int(spec["gap_start"])
    face_offset = int(spec["face_offset"])
    gap_dist = int(spec["gap_dist"])
    side = spec["side"]

    if side == "top":
        frame_r_lo = margin + gap_dist + blob_h
        frame_r_hi = height - margin - frame_h
        frame_c_lo = max(margin, margin - gap_start + face_offset)
        frame_c_hi = min(width - margin - frame_w, width - margin - blob_w - gap_start + face_offset)
    elif side == "bottom":
        frame_r_lo = margin
        frame_r_hi = height - margin - frame_h - gap_dist - blob_h
        frame_c_lo = max(margin, margin - gap_start + face_offset)
        frame_c_hi = min(width - margin - frame_w, width - margin - blob_w - gap_start + face_offset)
    elif side == "left":
        frame_r_lo = max(margin, margin - gap_start + face_offset)
        frame_r_hi = min(height - margin - frame_h, height - margin - blob_h - gap_start + face_offset)
        frame_c_lo = margin + gap_dist + blob_w
        frame_c_hi = width - margin - frame_w
    else:
        frame_r_lo = max(margin, margin - gap_start + face_offset)
        frame_r_hi = min(height - margin - frame_h, height - margin - blob_h - gap_start + face_offset)
        frame_c_lo = margin
        frame_c_hi = width - margin - frame_w - gap_dist - blob_w

    if frame_r_lo > frame_r_hi or frame_c_lo > frame_c_hi:
        return None
    return (int(frame_r_lo), int(frame_r_hi), int(frame_c_lo), int(frame_c_hi))


def _locate_blob(frame_r: int, frame_c: int, spec: dict) -> tuple[int, int]:
    blob_h, blob_w = spec["blob_mask"].shape
    gap_start = int(spec["gap_start"])
    face_offset = int(spec["face_offset"])
    gap_dist = int(spec["gap_dist"])
    frame_h, frame_w = spec["frame_mask"].shape
    side = spec["side"]

    if side == "top":
        blob_r = frame_r - gap_dist - blob_h
        blob_c = frame_c + gap_start - face_offset
    elif side == "bottom":
        blob_r = frame_r + frame_h + gap_dist
        blob_c = frame_c + gap_start - face_offset
    elif side == "left":
        blob_r = frame_r + gap_start - face_offset
        blob_c = frame_c - gap_dist - blob_w
    else:
        blob_r = frame_r + gap_start - face_offset
        blob_c = frame_c + frame_w + gap_dist
    return (int(blob_r), int(blob_c))


def _composite_bbox(frame_r: int, frame_c: int, spec: dict) -> tuple[int, int, int, int]:
    frame_h, frame_w = spec["frame_mask"].shape
    blob_h, blob_w = spec["blob_mask"].shape
    blob_r, blob_c = _locate_blob(frame_r, frame_c, spec)
    min_r = min(frame_r, blob_r)
    min_c = min(frame_c, blob_c)
    max_r = max(frame_r + frame_h - 1, blob_r + blob_h - 1)
    max_c = max(frame_c + frame_w - 1, blob_c + blob_w - 1)
    return (int(min_r), int(min_c), int(max_r), int(max_c))


def _try_place(height: int, width: int, occupied: np.ndarray, spec: dict) -> tuple[int, int, int, int] | None:
    ranges = _placement_ranges(height, width, spec)
    if ranges is None:
        return None

    r_lo, r_hi, c_lo, c_hi = ranges
    for _ in range(120):
        frame_r = random.randint(r_lo, r_hi)
        frame_c = random.randint(c_lo, c_hi)
        blob_r, blob_c = _locate_blob(frame_r, frame_c, spec)
        box_r0, box_c0, box_r1, box_c1 = _composite_bbox(frame_r, frame_c, spec)
        pad_r0 = max(0, box_r0 - 1)
        pad_c0 = max(0, box_c0 - 1)
        pad_r1 = min(height - 1, box_r1 + 1)
        pad_c1 = min(width - 1, box_c1 + 1)
        if occupied[pad_r0 : pad_r1 + 1, pad_c0 : pad_c1 + 1].any():
            continue
        occupied[pad_r0 : pad_r1 + 1, pad_c0 : pad_c1 + 1] = True
        return (int(frame_r), int(frame_c), int(blob_r), int(blob_c))

    return None


def _draw_mask(grid: np.ndarray, mask: np.ndarray, top: int, left: int, color: int) -> None:
    rows, cols = np.where(mask)
    grid[top + rows, left + cols] = int(color)


def _gen_attempt() -> np.ndarray | None:
    n_frames = random.choice((2, 3, 4))
    min_dim = 16 + 2 * n_frames
    height = random.randint(min_dim, 30)
    width = random.randint(min_dim, 30)

    palette = [color for color in range(10) if color != GRAY]
    bg = random.choice(palette)
    blob_colors = random.sample([color for color in palette if color != bg], n_frames)
    chosen_sides = random.sample(SIDES, len(SIDES))[:n_frames]

    specs: list[dict] = []
    used_shapes: set[tuple[tuple[int, int], bytes]] = set()
    for side, color in zip(chosen_sides, blob_colors):
        for _ in range(50):
            spec = _make_spec(side, color)
            if spec is None:
                continue
            key = (spec["blob_mask"].shape, spec["blob_mask"].tobytes())
            if key in used_shapes:
                continue
            used_shapes.add(key)
            specs.append(spec)
            break
        else:
            return None

    grid = np.full((height, width), int(bg), dtype=int)
    occupied = np.zeros((height, width), dtype=bool)
    placements: list[tuple[dict, tuple[int, int, int, int]]] = []

    for spec in specs:
        placed = _try_place(height, width, occupied, spec)
        if placed is None:
            return None
        placements.append((spec, placed))

    for spec, (frame_r, frame_c, blob_r, blob_c) in placements:
        _draw_mask(grid, spec["frame_mask"], frame_r, frame_c, GRAY)
        _draw_mask(grid, spec["blob_mask"], blob_r, blob_c, spec["color"])

    if int(np.count_nonzero(grid == bg)) <= grid.size // 2:
        return None

    try:
        _, scene = _analyze_scene(grid)
        out = main(grid)
    except Exception:
        return None

    if len(scene) != n_frames:
        return None
    if out.shape != grid.shape or not np.any(out != grid):
        return None
    if max(grid.shape) > 30 or max(out.shape) > 30:
        return None

    return grid


def generate_input() -> np.ndarray:
    for _ in range(GENERATE_RETRIES):
        grid = _gen_attempt()
        if grid is not None:
            return grid
    raise RuntimeError(f"failed to generate valid input after {GENERATE_RETRIES} attempts")
