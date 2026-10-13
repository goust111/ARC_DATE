import random
from collections import Counter, deque

import numpy as np


DIRS = {
    "up": (-1, 0),
    "down": (1, 0),
    "left": (0, -1),
    "right": (0, 1),
}
OPPOSITE = {"up": "down", "down": "up", "left": "right", "right": "left"}
COLORS = tuple(range(10))
BG = 0
GEN_RETRIES = 400


def _neighbors4(r: int, c: int):
    yield r - 1, c
    yield r + 1, c
    yield r, c - 1
    yield r, c + 1


def _components(mask: np.ndarray):
    h, w = mask.shape
    seen = np.zeros_like(mask, dtype=bool)
    comps = []
    for r in range(h):
        for c in range(w):
            if not mask[r, c] or seen[r, c]:
                continue
            q = deque([(r, c)])
            seen[r, c] = True
            pts = []
            while q:
                cr, cc = q.popleft()
                pts.append((cr, cc))
                for nr, nc in _neighbors4(cr, cc):
                    if 0 <= nr < h and 0 <= nc < w and mask[nr, nc] and not seen[nr, nc]:
                        seen[nr, nc] = True
                        q.append((nr, nc))
            comps.append(pts)
    return comps


def _bbox(points):
    rows = [r for r, _ in points]
    cols = [c for _, c in points]
    return min(rows), min(cols), max(rows), max(cols)


def _majority_color(grid: np.ndarray) -> int:
    vals, counts = np.unique(grid, return_counts=True)
    return int(vals[np.argmax(counts)])


def _gap_info(border_mask: np.ndarray):
    h, w = border_mask.shape
    expected = np.zeros_like(border_mask, dtype=bool)
    expected[0, :] = True
    expected[-1, :] = True
    expected[:, 0] = True
    expected[:, -1] = True
    missing = np.argwhere(expected & ~border_mask)
    if missing.shape != (2, 2):
        return None
    missing = [tuple(map(int, x)) for x in missing]
    if all(r == 0 for r, _ in missing) and abs(missing[0][1] - missing[1][1]) == 1:
        return "up"
    if all(r == h - 1 for r, _ in missing) and abs(missing[0][1] - missing[1][1]) == 1:
        return "down"
    if all(c == 0 for _, c in missing) and abs(missing[0][0] - missing[1][0]) == 1:
        return "left"
    if all(c == w - 1 for _, c in missing) and abs(missing[0][0] - missing[1][0]) == 1:
        return "right"
    return None


def _resize_mask(mask: np.ndarray, shape: tuple[int, int]) -> np.ndarray:
    th, tw = shape
    sh, sw = mask.shape
    out = np.zeros(shape, dtype=bool)
    for r in range(th):
        sr = min(sh - 1, (r * sh) // th)
        for c in range(tw):
            sc = min(sw - 1, (c * sw) // tw)
            out[r, c] = bool(mask[sr, sc])
    return out


def _detect_frames(grid: np.ndarray):
    bg = _majority_color(grid)
    frames = []
    for color in sorted(set(map(int, np.unique(grid))) - {bg}):
        for pts in _components(grid == color):
            r0, c0, r1, c1 = _bbox(pts)
            if r1 - r0 + 1 < 5 or c1 - c0 + 1 < 5:
                continue
            patch = grid[r0 : r1 + 1, c0 : c1 + 1]
            border_mask = patch == color
            opening = _gap_info(border_mask)
            if opening is None:
                continue
            expected = np.zeros_like(border_mask, dtype=bool)
            expected[0, :] = True
            expected[-1, :] = True
            expected[:, 0] = True
            expected[:, -1] = True
            if np.any(border_mask & ~expected):
                continue
            interior = patch[1:-1, 1:-1]
            inside_colors = sorted(set(map(int, np.unique(interior))) - {bg})
            if len(inside_colors) != 1:
                continue
            obj_color = inside_colors[0]
            obj_mask = interior == obj_color
            comps = _components(obj_mask)
            if len(comps) != 1:
                continue
            cr0, cc0, cr1, cc1 = _bbox(comps[0])
            sprite = obj_mask[cr0 : cr1 + 1, cc0 : cc1 + 1]
            frames.append(
                {
                    "color": color,
                    "bbox": (r0, c0, r1, c1),
                    "opening": opening,
                    "inner_shape": interior.shape,
                    "obj_color": obj_color,
                    "obj_mask": sprite,
                }
            )
    assert frames, "expected at least one valid frame"
    covered = np.zeros_like(grid, dtype=bool)
    for frame in frames:
        r0, c0, r1, c1 = frame["bbox"]
        covered[r0 : r1 + 1, c0 : c1 + 1] |= grid[r0 : r1 + 1, c0 : c1 + 1] != bg
    assert np.array_equal(covered, grid != bg), "all non-background pixels must belong to frames or their objects"
    return frames


def _find_target(frame, frames):
    r0, c0, r1, c1 = frame["bbox"]
    side = frame["opening"]
    best = None
    best_dist = None
    for other in frames:
        if other is frame or other["opening"] != OPPOSITE[side]:
            continue
        or0, oc0, or1, oc1 = other["bbox"]
        if side == "right":
            if oc0 <= c1:
                continue
            if max(r0, or0) > min(r1, or1):
                continue
            dist = oc0 - c1
        elif side == "left":
            if oc1 >= c0:
                continue
            if max(r0, or0) > min(r1, or1):
                continue
            dist = c0 - oc1
        elif side == "down":
            if or0 <= r1:
                continue
            if max(c0, oc0) > min(c1, oc1):
                continue
            dist = or0 - r1
        else:
            if or1 >= r0:
                continue
            if max(c0, oc0) > min(c1, oc1):
                continue
            dist = r0 - or1
        if best_dist is None or dist < best_dist:
            best_dist = dist
            best = other
    assert best is not None, "every frame must have a facing target"
    return best


def main(grid: np.ndarray) -> np.ndarray:
    g = np.asarray(grid, dtype=int)
    assert g.ndim == 2
    bg = _majority_color(g)
    frames = _detect_frames(g)
    targets = {id(frame): _find_target(frame, frames) for frame in frames}
    for frame in frames:
        assert targets[id(targets[id(frame)])] is frame, "facing relation must be symmetric"

    out = g.copy()
    for frame in frames:
        r0, c0, r1, c1 = frame["bbox"]
        out[r0 + 1 : r1, c0 + 1 : c1] = bg

    for frame in frames:
        incoming = targets[id(frame)]
        r0, c0, r1, c1 = frame["bbox"]
        inner_shape = (r1 - r0 - 1, c1 - c0 - 1)
        scaled = _resize_mask(incoming["obj_mask"], inner_shape)
        out[r0 + 1 : r1, c0 + 1 : c1][scaled] = incoming["obj_color"]
    return out


def _make_object_mask(rng: random.Random, h: int, w: int) -> np.ndarray:
    kind = rng.choice(["rect", "plus", "diag", "lshape", "blob"])
    mask = np.zeros((h, w), dtype=bool)
    if kind == "rect":
        rh = rng.randint(1, h)
        rw = rng.randint(1, w)
        r0 = rng.randint(0, h - rh)
        c0 = rng.randint(0, w - rw)
        mask[r0 : r0 + rh, c0 : c0 + rw] = True
    elif kind == "plus":
        cr = rng.randrange(h)
        cc = rng.randrange(w)
        mask[cr, :] = True
        mask[:, cc] = True
    elif kind == "diag":
        if rng.random() < 0.5:
            for r in range(h):
                c = min(w - 1, (r * w) // h)
                mask[r, c] = True
        else:
            for r in range(h):
                c = min(w - 1, ((h - 1 - r) * w) // h)
                mask[r, c] = True
    elif kind == "lshape":
        mask[:, rng.randrange(w)] = True
        mask[rng.randrange(h), :] = True
    else:
        sr = rng.randrange(h)
        sc = rng.randrange(w)
        mask[sr, sc] = True
        target = rng.randint(max(2, (h * w) // 5), max(2, (2 * h * w) // 3))
        pts = {(sr, sc)}
        for _ in range(target * 4):
            if len(pts) >= target:
                break
            pr, pc = rng.choice(tuple(pts))
            cand = [(nr, nc) for nr, nc in _neighbors4(pr, pc) if 0 <= nr < h and 0 <= nc < w]
            if cand:
                pts.add(rng.choice(cand))
        for r, c in pts:
            mask[r, c] = True
    comps = _components(mask)
    if len(comps) > 1:
        largest = max(comps, key=len)
        clean = np.zeros_like(mask)
        for r, c in largest:
            clean[r, c] = True
        mask = clean
    if not mask.any():
        mask[rng.randrange(h), rng.randrange(w)] = True
    return mask


def _draw_frame(grid: np.ndarray, top: int, left: int, oh: int, ow: int, color: int, opening: str):
    grid[top, left : left + ow] = color
    grid[top + oh - 1, left : left + ow] = color
    grid[top : top + oh, left] = color
    grid[top : top + oh, left + ow - 1] = color
    if opening in ("up", "down"):
        start = left + ow // 2 - 1
        row = top if opening == "up" else top + oh - 1
        grid[row, start : start + 2] = BG
    else:
        start = top + oh // 2 - 1
        col = left if opening == "left" else left + ow - 1
        grid[start : start + 2, col] = BG


def _paint_object(grid: np.ndarray, top: int, left: int, mask: np.ndarray, color: int):
    h, w = mask.shape
    grid[top : top + h, left : left + w][mask] = color


def _sample_pair(rng: random.Random, horizontal: bool):
    if horizontal:
        oh1, ow1 = rng.randint(5, 8), rng.randint(5, 8)
        oh2, ow2 = rng.randint(5, 8), rng.randint(5, 8)
        oh = max(oh1, oh2)
        oh1 = oh2 = oh
        gap = rng.randint(2, 5)
        frames = [
            {"top": 0, "left": 0, "oh": oh1, "ow": ow1, "opening": "right"},
            {"top": 0, "left": ow1 + gap, "oh": oh2, "ow": ow2, "opening": "left"},
        ]
    else:
        oh1, ow1 = rng.randint(5, 8), rng.randint(5, 8)
        oh2, ow2 = rng.randint(5, 8), rng.randint(5, 8)
        ow = max(ow1, ow2)
        ow1 = ow2 = ow
        gap = rng.randint(2, 5)
        frames = [
            {"top": 0, "left": 0, "oh": oh1, "ow": ow1, "opening": "down"},
            {"top": oh1 + gap, "left": 0, "oh": oh2, "ow": ow2, "opening": "up"},
        ]
    return frames


def _try_generate(rng: random.Random):
    h = rng.randint(22, 34)
    w = rng.randint(22, 34)
    grid = np.full((h, w), BG, dtype=int)
    occupancy = np.zeros((h, w), dtype=bool)
    pair_count = rng.randint(1, 3)
    orientations = [rng.choice([True, False]) for _ in range(pair_count)]
    if pair_count > 1 and len(set(orientations)) == 1 and rng.random() < 0.5:
        orientations[0] = not orientations[0]

    pair_frames = []
    for horizontal in orientations:
        rel_frames = _sample_pair(rng, horizontal)
        pair_h = max(f["top"] + f["oh"] for f in rel_frames)
        pair_w = max(f["left"] + f["ow"] for f in rel_frames)
        if pair_h > h or pair_w > w:
            return None
        placed = False
        for _ in range(80):
            top = rng.randint(0, h - pair_h)
            left = rng.randint(0, w - pair_w)
            if occupancy[max(0, top - 2) : min(h, top + pair_h + 2), max(0, left - 2) : min(w, left + pair_w + 2)].any():
                continue
            colors = [c for c in COLORS if c != BG]
            rng.shuffle(colors)
            if len(colors) < 4:
                return None
            frame_colors = colors[:2]
            obj_colors = colors[2:4]
            for fc, oc, spec in zip(frame_colors, obj_colors, rel_frames):
                ft = top + spec["top"]
                fl = left + spec["left"]
                _draw_frame(grid, ft, fl, spec["oh"], spec["ow"], fc, spec["opening"])
                inner_h = spec["oh"] - 2
                inner_w = spec["ow"] - 2
                obj_mask = _make_object_mask(rng, inner_h, inner_w)
                _paint_object(grid, ft + 1, fl + 1, obj_mask, oc)
                pair_frames.append((ft, fl, spec["oh"], spec["ow"]))
            occupancy[top : top + pair_h, left : left + pair_w] = True
            placed = True
            break
        if not placed:
            return None

    if len(pair_frames) == 1 and rng.random() < 0.4:
        return None

    detected = _detect_frames(grid)
    if len(detected) != 2 * pair_count:
        return None
    targets = {id(frame): _find_target(frame, detected) for frame in detected}
    for frame in detected:
        if targets[id(targets[id(frame)])] is not frame:
            return None
    return grid


def generate_input() -> np.ndarray:
    rng = random.Random()
    for _ in range(GEN_RETRIES):
        grid = _try_generate(rng)
        if grid is not None:
            return grid
    raise RuntimeError("failed to generate a valid grid")
