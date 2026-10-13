import random
from collections import deque

import numpy as np


BG = 0
DIR_TO_VEC = {
    "up": (-1, 0),
    "down": (1, 0),
    "left": (0, -1),
    "right": (0, 1),
}
VEC_TO_DIR = {v: k for k, v in DIR_TO_VEC.items()}
DIRS4 = ((1, 0), (-1, 0), (0, 1), (0, -1))


def _neighbors4(r: int, c: int, h: int, w: int):
    for dr, dc in DIRS4:
        nr = r + dr
        nc = c + dc
        if 0 <= nr < h and 0 <= nc < w:
            yield nr, nc


def _components(mask: np.ndarray) -> list[np.ndarray]:
    h, w = mask.shape
    seen = np.zeros((h, w), dtype=bool)
    parts: list[np.ndarray] = []
    for r in range(h):
        for c in range(w):
            if not mask[r, c] or seen[r, c]:
                continue
            q = deque([(r, c)])
            seen[r, c] = True
            pts: list[tuple[int, int]] = []
            while q:
                cr, cc = q.popleft()
                pts.append((cr, cc))
                for nr, nc in _neighbors4(cr, cc, h, w):
                    if mask[nr, nc] and not seen[nr, nc]:
                        seen[nr, nc] = True
                        q.append((nr, nc))
            parts.append(np.asarray(pts, dtype=int))
    return parts


def _bbox(coords: np.ndarray) -> tuple[int, int, int, int]:
    return (
        int(coords[:, 0].min()),
        int(coords[:, 1].min()),
        int(coords[:, 0].max()),
        int(coords[:, 1].max()),
    )


def _is_rect_frame(mask: np.ndarray) -> bool:
    coords = np.argwhere(mask)
    if len(coords) == 0:
        return False
    r0, c0, r1, c1 = _bbox(coords)
    h = r1 - r0 + 1
    w = c1 - c0 + 1
    if h < 3 or w < 3:
        return False
    cropped = mask[r0 : r1 + 1, c0 : c1 + 1]
    expected = np.zeros_like(cropped, dtype=bool)
    expected[0, :] = True
    expected[-1, :] = True
    expected[:, 0] = True
    expected[:, -1] = True
    return bool(np.array_equal(cropped, expected))


def _find_frames(grid: np.ndarray) -> list[tuple[int, int, int, int, int]]:
    frames: list[tuple[int, int, int, int, int]] = []
    for color in sorted(int(v) for v in np.unique(grid) if int(v) != BG):
        for comp in _components(grid == color):
            comp_mask = np.zeros_like(grid, dtype=bool)
            comp_mask[comp[:, 0], comp[:, 1]] = True
            if _is_rect_frame(comp_mask):
                r0, c0, r1, c1 = _bbox(comp)
                frames.append((r0, c0, r1, c1, color))
    frames.sort()
    return frames


def _infer_growth_vector(coords: np.ndarray) -> tuple[int, int]:
    r0, c0, r1, c1 = _bbox(coords)
    center_r = (r0 + r1) / 2.0
    center_c = (c0 + c1) / 2.0
    centroid_r = float(coords[:, 0].mean())
    centroid_c = float(coords[:, 1].mean())
    off_r = centroid_r - center_r
    off_c = centroid_c - center_c
    if abs(off_c) > abs(off_r):
        return (0, 1) if off_c < 0 else (0, -1)
    if abs(off_r) > abs(off_c):
        return (1, 0) if off_r < 0 else (-1, 0)

    left = int(np.sum(coords[:, 1] < center_c))
    right = int(np.sum(coords[:, 1] > center_c))
    up = int(np.sum(coords[:, 0] < center_r))
    down = int(np.sum(coords[:, 0] > center_r))
    hdiff = left - right
    vdiff = up - down
    if abs(hdiff) > abs(vdiff):
        return (0, 1) if hdiff > 0 else (0, -1)
    if abs(vdiff) > abs(hdiff):
        return (1, 0) if vdiff > 0 else (-1, 0)

    if (c1 - c0) >= (r1 - r0):
        return (0, 1) if centroid_c <= center_c else (0, -1)
    return (1, 0) if centroid_r <= center_r else (-1, 0)


def _paint_growth(
    out: np.ndarray,
    frame_bbox: tuple[int, int, int, int],
    frame_color: int,
    object_coords: np.ndarray,
    growth_vec: tuple[int, int],
) -> None:
    r0, c0, r1, c1 = frame_bbox
    dr, dc = growth_vec
    if dc == 1:
        for row in sorted(set(int(v) for v in object_coords[:, 0])):
            cols = object_coords[object_coords[:, 0] == row][:, 1]
            start = int(cols.max()) + 1
            end = c1 - 1
            if start <= end:
                out[row, start : end + 1] = frame_color
        return
    if dc == -1:
        for row in sorted(set(int(v) for v in object_coords[:, 0])):
            cols = object_coords[object_coords[:, 0] == row][:, 1]
            start = c0 + 1
            end = int(cols.min()) - 1
            if start <= end:
                out[row, start : end + 1] = frame_color
        return
    if dr == 1:
        for col in sorted(set(int(v) for v in object_coords[:, 1])):
            rows = object_coords[object_coords[:, 1] == col][:, 0]
            start = int(rows.max()) + 1
            end = r1 - 1
            if start <= end:
                out[start : end + 1, col] = frame_color
        return
    for col in sorted(set(int(v) for v in object_coords[:, 1])):
        rows = object_coords[object_coords[:, 1] == col][:, 0]
        start = r0 + 1
        end = int(rows.min()) - 1
        if start <= end:
            out[start : end + 1, col] = frame_color


def main(grid: np.ndarray) -> np.ndarray:
    """
    For each colored rectangular frame, infer the inner object's dense side from
    its centroid shift, then grow the object's silhouette in the opposite
    cardinal direction using the frame color until the inner wall is reached.
    """
    g = np.asarray(grid, dtype=int)
    out = g.copy()
    frames = _find_frames(g)
    for r0, c0, r1, c1, frame_color in frames:
        if r1 - r0 < 2 or c1 - c0 < 2:
            continue
        interior = g[r0 + 1 : r1, c0 + 1 : c1]
        objects = _components(interior != BG)
        if not objects:
            continue
        obj = max(objects, key=len)
        obj_abs = obj + np.array([r0 + 1, c0 + 1], dtype=int)
        growth_vec = _infer_growth_vector(obj_abs)
        _paint_growth(out, (r0, c0, r1, c1), frame_color, obj_abs, growth_vec)
    return out


def _shape_matches_direction(mask: np.ndarray, grow_dir: str) -> bool:
    coords = np.argwhere(mask)
    if len(coords) == 0:
        return False
    return VEC_TO_DIR[_infer_growth_vector(coords)] == grow_dir


def _can_remove(mask: np.ndarray, r: int, c: int) -> bool:
    if not mask[r, c]:
        return False
    test = mask.copy()
    test[r, c] = False
    if not np.any(test):
        return False
    return len(_components(test)) == 1


def _make_biased_shape(grow_dir: str, rng: random.Random) -> np.ndarray | None:
    for _ in range(200):
        if grow_dir in ("left", "right"):
            h = rng.randint(3, 6)
            w = rng.randint(4, 7)
            dense_w = rng.randint(2, w - 2)
            mask = np.zeros((h, w), dtype=bool)
            center_rows = [h // 2] if h % 2 == 1 else [h // 2 - 1, h // 2]
            bridge_rows = center_rows[:]
            if h >= 5 and rng.random() < 0.35:
                candidate = center_rows[0] - 1
                if 0 <= candidate < h:
                    bridge_rows.append(candidate)
            bridge_rows = sorted(set(bridge_rows))
            if grow_dir == "right":
                mask[:, :dense_w] = True
                for rr in bridge_rows:
                    mask[rr, dense_w:w] = True
            else:
                mask[:, w - dense_w : w] = True
                for rr in bridge_rows:
                    mask[rr, : w - dense_w] = True
            if rng.random() < 0.5:
                rr = rng.choice(bridge_rows)
                if grow_dir == "right":
                    cc = rng.randint(dense_w, w - 1)
                    if rr + 1 < h:
                        mask[rr + 1, cc] = True
                    elif rr - 1 >= 0:
                        mask[rr - 1, cc] = True
                else:
                    cc = rng.randint(0, w - dense_w - 1)
                    if rr + 1 < h:
                        mask[rr + 1, cc] = True
                    elif rr - 1 >= 0:
                        mask[rr - 1, cc] = True
            candidates = []
            if grow_dir == "right":
                for rr in range(h):
                    for cc in range(min(dense_w, w - 1)):
                        if cc == dense_w - 1 and rr in bridge_rows:
                            continue
                        candidates.append((rr, cc))
            else:
                for rr in range(h):
                    for cc in range(max(1, w - dense_w), w):
                        if cc == w - dense_w and rr in bridge_rows:
                            continue
                        candidates.append((rr, cc))
        else:
            h = rng.randint(4, 7)
            w = rng.randint(3, 6)
            dense_h = rng.randint(2, h - 2)
            mask = np.zeros((h, w), dtype=bool)
            center_cols = [w // 2] if w % 2 == 1 else [w // 2 - 1, w // 2]
            bridge_cols = center_cols[:]
            if w >= 5 and rng.random() < 0.35:
                candidate = center_cols[0] - 1
                if 0 <= candidate < w:
                    bridge_cols.append(candidate)
            bridge_cols = sorted(set(bridge_cols))
            if grow_dir == "down":
                mask[:dense_h, :] = True
                for cc in bridge_cols:
                    mask[dense_h:h, cc] = True
            else:
                mask[h - dense_h : h, :] = True
                for cc in bridge_cols:
                    mask[: h - dense_h, cc] = True
            if rng.random() < 0.5:
                cc = rng.choice(bridge_cols)
                if grow_dir == "down":
                    rr = rng.randint(dense_h, h - 1)
                    if cc + 1 < w:
                        mask[rr, cc + 1] = True
                    elif cc - 1 >= 0:
                        mask[rr, cc - 1] = True
                else:
                    rr = rng.randint(0, h - dense_h - 1)
                    if cc + 1 < w:
                        mask[rr, cc + 1] = True
                    elif cc - 1 >= 0:
                        mask[rr, cc - 1] = True
            candidates = []
            if grow_dir == "down":
                for rr in range(min(dense_h, h - 1)):
                    for cc in range(w):
                        if rr == dense_h - 1 and cc in bridge_cols:
                            continue
                        candidates.append((rr, cc))
            else:
                for rr in range(max(1, h - dense_h), h):
                    for cc in range(w):
                        if rr == h - dense_h and cc in bridge_cols:
                            continue
                        candidates.append((rr, cc))

        rng.shuffle(candidates)
        removals = rng.randint(0, min(2, len(candidates)))
        removed = 0
        for rr, cc in candidates:
            if removed >= removals:
                break
            if _can_remove(mask, rr, cc):
                mask[rr, cc] = False
                removed += 1

        if len(_components(mask)) != 1:
            continue
        if not _shape_matches_direction(mask, grow_dir):
            continue
        return mask
    return None


def _find_positions(occupied: np.ndarray, ph: int, pw: int, padding: int) -> list[tuple[int, int]]:
    h, w = occupied.shape
    positions: list[tuple[int, int]] = []
    for r in range(h - ph + 1):
        for c in range(w - pw + 1):
            rs = max(0, r - padding)
            re = min(h, r + ph + padding)
            cs = max(0, c - padding)
            ce = min(w, c + pw + padding)
            if not occupied[rs:re, cs:ce].any():
                positions.append((r, c))
    return positions


def _render_expected(
    inp: np.ndarray,
    specs: list[dict[str, object]],
) -> np.ndarray:
    out = inp.copy()
    for spec in specs:
        _paint_growth(
            out,
            spec["frame_bbox"],
            int(spec["frame_color"]),
            np.asarray(spec["object_coords"], dtype=int),
            DIR_TO_VEC[str(spec["grow_dir"])],
        )
    return out


def _gen_attempt(rng: random.Random) -> np.ndarray | None:
    n_frames = rng.randint(1, 4)
    min_side = 18 + max(0, n_frames - 2) * 3
    h = rng.randint(min_side, 30)
    w = rng.randint(min_side, 30)
    grid = np.zeros((h, w), dtype=int)
    occupied = np.zeros((h, w), dtype=bool)

    colors = list(range(1, 10))
    rng.shuffle(colors)
    if 2 * n_frames > len(colors):
        return None
    wall_colors = colors[:n_frames]
    obj_colors = colors[n_frames : 2 * n_frames]

    dirs = ["up", "down", "left", "right"]
    rng.shuffle(dirs)
    grow_dirs = [dirs[i] for i in range(min(n_frames, 4))]
    while len(grow_dirs) < n_frames:
        grow_dirs.append(rng.choice(dirs))
    rng.shuffle(grow_dirs)

    specs: list[dict[str, object]] = []
    for idx in range(n_frames):
        grow_dir = grow_dirs[idx]
        shape = _make_biased_shape(grow_dir, rng)
        if shape is None:
            return None
        oh, ow = shape.shape
        top_gap = rng.randint(1, 3)
        bottom_gap = rng.randint(1, 3)
        left_gap = rng.randint(1, 3)
        right_gap = rng.randint(1, 3)
        if grow_dir == "up":
            top_gap = rng.randint(2, 5)
        elif grow_dir == "down":
            bottom_gap = rng.randint(2, 5)
        elif grow_dir == "left":
            left_gap = rng.randint(2, 5)
        else:
            right_gap = rng.randint(2, 5)
        inner_h = top_gap + oh + bottom_gap
        inner_w = left_gap + ow + right_gap
        outer_h = inner_h + 2
        outer_w = inner_w + 2
        if outer_h >= h or outer_w >= w:
            return None

        positions = _find_positions(occupied, outer_h, outer_w, padding=2)
        if not positions:
            return None
        fr0, fc0 = rng.choice(positions)
        fr1 = fr0 + outer_h - 1
        fc1 = fc0 + outer_w - 1

        wall_color = wall_colors[idx]
        obj_color = obj_colors[idx]
        grid[fr0, fc0 : fc1 + 1] = wall_color
        grid[fr1, fc0 : fc1 + 1] = wall_color
        grid[fr0 : fr1 + 1, fc0] = wall_color
        grid[fr0 : fr1 + 1, fc1] = wall_color

        obj_r0 = fr0 + 1 + top_gap
        obj_c0 = fc0 + 1 + left_gap
        patch = np.where(shape, obj_color, BG)
        target = grid[obj_r0 : obj_r0 + oh, obj_c0 : obj_c0 + ow]
        if np.any((patch != BG) & (target != BG)):
            return None
        target[shape] = obj_color

        obj_coords = np.argwhere(shape) + np.array([obj_r0, obj_c0], dtype=int)
        if VEC_TO_DIR[_infer_growth_vector(obj_coords)] != grow_dir:
            return None
        if np.array_equal(_render_expected(grid, [{"frame_bbox": (fr0, fc0, fr1, fc1), "frame_color": wall_color, "object_coords": obj_coords, "grow_dir": grow_dir}]), grid):
            return None

        occupied[max(0, fr0 - 2) : min(h, fr1 + 3), max(0, fc0 - 2) : min(w, fc1 + 3)] = True
        specs.append(
            {
                "frame_bbox": (fr0, fc0, fr1, fc1),
                "frame_color": wall_color,
                "object_coords": obj_coords,
                "grow_dir": grow_dir,
            }
        )

    expected = _render_expected(grid, specs)
    try:
        got = main(grid)
    except Exception:
        return None
    if np.array_equal(got, grid):
        return None
    if not np.array_equal(got, expected):
        return None
    return grid


def generate_input() -> np.ndarray:
    rng = random.Random()
    for _ in range(400):
        result = _gen_attempt(rng)
        if result is not None:
            return result
    raise RuntimeError("generate_input: failed after retries")
