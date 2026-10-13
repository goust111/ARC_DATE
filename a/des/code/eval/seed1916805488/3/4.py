import random
from collections import deque
from typing import List

import numpy as np


BG = 0
FRAME = 7
KEY_COLORS = [1, 2, 3, 4, 5, 6, 8, 9]
DIR4 = [(-1, 0), (1, 0), (0, -1), (0, 1)]
MAX_GENERATION_ATTEMPTS = 300


def _components(mask: np.ndarray) -> List[list[tuple[int, int]]]:
    h, w = mask.shape
    seen = np.zeros((h, w), dtype=bool)
    comps: List[list[tuple[int, int]]] = []
    for r in range(h):
        for c in range(w):
            if not mask[r, c] or seen[r, c]:
                continue
            q = deque([(r, c)])
            seen[r, c] = True
            pts: list[tuple[int, int]] = []
            while q:
                rr, cc = q.popleft()
                pts.append((rr, cc))
                for dr, dc in DIR4:
                    nr, nc = rr + dr, cc + dc
                    if 0 <= nr < h and 0 <= nc < w and mask[nr, nc] and not seen[nr, nc]:
                        seen[nr, nc] = True
                        q.append((nr, nc))
            comps.append(pts)
    return comps


def _crop_mask(mask: np.ndarray) -> np.ndarray:
    pts = np.argwhere(mask)
    r0, c0 = pts.min(axis=0)
    r1, c1 = pts.max(axis=0)
    return mask[r0 : r1 + 1, c0 : c1 + 1].copy()


def _shape_variants(mask: np.ndarray) -> list[np.ndarray]:
    variants: list[np.ndarray] = []
    seen = set()
    for k in range(4):
        rot = np.rot90(mask, -k)
        key = (rot.shape, rot.tobytes())
        if key not in seen:
            seen.add(key)
            variants.append(rot)
    return variants


def _shape_signature(mask: np.ndarray) -> tuple:
    variants = _shape_variants(mask)
    keys = sorted((v.shape[0], v.shape[1], v.tobytes()) for v in variants)
    return keys[0]


def _generate_shape_pool() -> list[np.ndarray]:
    pool: list[np.ndarray] = []
    seen_classes = set()
    for h in (1, 2):
        for w in (2, 3, 4):
            for bits in range(1, 1 << (h * w)):
                mask = np.zeros((h, w), dtype=bool)
                for idx in range(h * w):
                    if (bits >> idx) & 1:
                        mask[idx // w, idx % w] = True
                if mask.sum() < 2:
                    continue
                if not mask[-1].any():
                    continue
                if len(_components(mask)) != 1:
                    continue
                variants = _shape_variants(mask)
                if len(variants) != 4:
                    continue
                sig = _shape_signature(mask)
                if sig in seen_classes:
                    continue
                seen_classes.add(sig)
                pool.append(mask.copy())
    pool.sort(key=lambda m: (m.shape[0] * m.shape[1], int(m.sum()), m.shape, m.tobytes()))
    return pool


SHAPE_POOL = _generate_shape_pool()


def _draw_frame_with_lockhole(
    canvas: np.ndarray,
    top: int,
    left: int,
    outer_h: int,
    outer_w: int,
    lockhole: np.ndarray,
    chamber_left: int,
) -> None:
    bottom = top + outer_h - 1
    right = left + outer_w - 1
    canvas[top, left : right + 1] = FRAME
    canvas[bottom, left : right + 1] = FRAME
    canvas[top : bottom + 1, left] = FRAME
    canvas[top : bottom + 1, right] = FRAME

    h, w = lockhole.shape
    notch_left = left + 1 + chamber_left + 1
    chamber_abs_left = left + 1 + chamber_left
    cap_row = bottom - h
    canvas[cap_row : bottom + 1, chamber_abs_left : chamber_abs_left + w + 2] = FRAME
    for rr in range(h):
        for cc in range(w):
            if lockhole[rr, cc]:
                canvas[bottom - h + 1 + rr, notch_left + cc] = BG


def _find_frames(grid: np.ndarray) -> list[dict]:
    comps = _components(grid == FRAME)
    frames = []
    for pts in comps:
        rs = [r for r, _ in pts]
        cs = [c for _, c in pts]
        r0, r1 = min(rs), max(rs)
        c0, c1 = min(cs), max(cs)
        if r1 - r0 + 1 < 5 or c1 - c0 + 1 < 5:
            continue
        if not np.all(grid[r0, c0 : c1 + 1] == FRAME):
            continue
        if not np.all(grid[r0 : r1 + 1, c0] == FRAME):
            continue
        if not np.all(grid[r0 : r1 + 1, c1] == FRAME):
            continue
        sub = grid[r0 : r1 + 1, c0 : c1 + 1]
        bg_comps = _components(sub == BG)
        lockholes = []
        for comp in bg_comps:
            if not any(rr == sub.shape[0] - 1 for rr, _ in comp):
                continue
            if any(cc == 0 or cc == sub.shape[1] - 1 for _, cc in comp):
                continue
            mask = np.zeros(sub.shape, dtype=bool)
            for rr, cc in comp:
                mask[rr, cc] = True
            mask = _crop_mask(mask)
            if mask.shape[0] > 2 or mask.shape[1] > 4 or mask.sum() < 2:
                continue
            lockholes.append(mask)
        if len(lockholes) != 1:
            continue
        frames.append({"bbox": (r0, c0, r1, c1), "lockhole": lockholes[0]})
    frames.sort(key=lambda d: d["bbox"])
    return frames


def _find_keys(grid: np.ndarray) -> list[dict]:
    comps = _components((grid != BG) & (grid != FRAME))
    keys = []
    for pts in comps:
        colors = {int(grid[r, c]) for r, c in pts}
        if len(colors) != 1:
            continue
        color = next(iter(colors))
        rs = [r for r, _ in pts]
        cs = [c for _, c in pts]
        r0, r1 = min(rs), max(rs)
        c0, c1 = min(cs), max(cs)
        mask = np.zeros((r1 - r0 + 1, c1 - c0 + 1), dtype=bool)
        for r, c in pts:
            mask[r - r0, c - c0] = True
        keys.append({"color": color, "mask": mask, "bbox": (r0, c0, r1, c1)})
    keys.sort(key=lambda d: d["bbox"])
    return keys


def _fill_pattern(out: np.ndarray, bbox: tuple[int, int, int, int], color: int, rotations: int) -> None:
    r0, c0, r1, c1 = bbox
    for r in range(r0 + 1, r1):
        for c in range(c0 + 1, c1):
            rr = r - (r0 + 1)
            cc = c - (c0 + 1)
            paint = False
            if rotations == 0:
                paint = True
            elif rotations == 1:
                paint = cc % 2 == 1
            elif rotations == 2:
                paint = rr % 2 == 1
            else:
                paint = (rr + cc) % 2 == 1
            out[r, c] = color if paint else BG


def main(grid: np.ndarray) -> np.ndarray:
    g = np.asarray(grid, dtype=int)
    frames = _find_frames(g)
    keys = _find_keys(g)
    assert frames, "expected at least one orange frame"
    assert len(frames) == len(keys), "frame/key counts must match"

    matches: list[tuple[int, int, int]] = []
    used_frames = set()
    for ki, key in enumerate(keys):
        options = []
        for fi, frame in enumerate(frames):
            target = frame["lockhole"]
            for rotations in range(4):
                if np.array_equal(np.rot90(key["mask"], -rotations), target):
                    options.append((fi, rotations))
        assert len(options) == 1, "every key must have exactly one match"
        fi, rotations = options[0]
        assert fi not in used_frames, "matches must be one-to-one"
        used_frames.add(fi)
        matches.append((ki, fi, rotations))

    out = np.full_like(g, BG)
    for frame in frames:
        r0, c0, r1, c1 = frame["bbox"]
        out[r0, c0 : c1 + 1] = FRAME
        out[r1, c0 : c1 + 1] = FRAME
        out[r0 : r1 + 1, c0] = FRAME
        out[r0 : r1 + 1, c1] = FRAME

    for ki, fi, rotations in matches:
        _fill_pattern(out, frames[fi]["bbox"], keys[ki]["color"], rotations)

    return out


def _boxes_overlap(a: tuple[int, int, int, int], b: tuple[int, int, int, int], gap: int = 0) -> bool:
    ar0, ac0, ar1, ac1 = a
    br0, bc0, br1, bc1 = b
    return not (ar1 + gap < br0 or br1 + gap < ar0 or ac1 + gap < bc0 or bc1 + gap < ac0)


def _random_shape_selection(rng: random.Random, count: int) -> list[np.ndarray]:
    shapes = SHAPE_POOL.copy()
    rng.shuffle(shapes)
    chosen: list[np.ndarray] = []
    used = set()
    for shape in shapes:
        sig = _shape_signature(shape)
        if sig in used:
            continue
        chosen.append(shape.copy())
        used.add(sig)
        if len(chosen) == count:
            return chosen
    raise RuntimeError("insufficient distinct shapes")


def _gen_attempt(rng: random.Random) -> np.ndarray | None:
    n = rng.randint(2, 4)
    shapes = _random_shape_selection(rng, n)
    colors = rng.sample(KEY_COLORS, n)

    frame_specs = []
    for shape, color in zip(shapes, colors):
        sh, sw = shape.shape
        inner_h = rng.randint(max(sh + 3, 5), max(sh + 4, 6))
        inner_w = rng.randint(max(sw + 3, 6), max(sw + 5, 8))
        outer_h = inner_h + 2
        outer_w = inner_w + 2
        chamber_w = sw + 2
        max_lock_left = inner_w - chamber_w
        if max_lock_left < 0:
            return None
        if max_lock_left >= 2:
            lock_left = rng.randint(1, max_lock_left)
        else:
            lock_left = rng.randint(0, max_lock_left)
        rotations = rng.randrange(4)
        key_mask = np.rot90(shape, rotations)
        frame_specs.append(
            {
                "lockhole": shape.copy(),
                "color": color,
                "outer_h": outer_h,
                "outer_w": outer_w,
                "lock_left": lock_left,
                "rotations": rotations,
                "key_mask": key_mask,
            }
        )

    min_h = max(22, sum(spec["outer_h"] for spec in frame_specs[: min(3, n)]) // 2 + 10)
    min_w = max(22, sum(spec["outer_w"] for spec in frame_specs[: min(3, n)]) // 2 + 10)
    h = rng.randint(min_h, 31)
    w = rng.randint(min_w, 31)
    grid = np.full((h, w), BG, dtype=int)

    frame_boxes: list[tuple[int, int, int, int]] = []
    for spec in frame_specs:
        placed = False
        for _ in range(120):
            top = rng.randint(0, h - spec["outer_h"])
            left = rng.randint(0, w - spec["outer_w"])
            box = (top, left, top + spec["outer_h"] - 1, left + spec["outer_w"] - 1)
            if any(_boxes_overlap(box, old, gap=2) for old in frame_boxes):
                continue
            _draw_frame_with_lockhole(grid, top, left, spec["outer_h"], spec["outer_w"], spec["lockhole"], spec["lock_left"])
            spec["frame_box"] = box
            frame_boxes.append(box)
            placed = True
            break
        if not placed:
            return None

    key_boxes: list[tuple[int, int, int, int]] = frame_boxes.copy()
    for spec in frame_specs:
        km = spec["key_mask"]
        kh, kw = km.shape
        placed = False
        for _ in range(200):
            top = rng.randint(0, h - kh)
            left = rng.randint(0, w - kw)
            box = (top, left, top + kh - 1, left + kw - 1)
            if any(_boxes_overlap(box, old, gap=1) for old in key_boxes):
                continue
            for rr in range(kh):
                for cc in range(kw):
                    if km[rr, cc]:
                        grid[top + rr, left + cc] = spec["color"]
            key_boxes.append(box)
            placed = True
            break
        if not placed:
            return None

    try:
        out = main(grid)
    except AssertionError:
        return None

    # Reject degenerate cases where two distinct frames would get identical fill appearance.
    sigs = []
    for spec in frame_specs:
        sigs.append((spec["rotations"], spec["color"]))
    if len(set(sigs)) != len(sigs):
        return None
    if out.shape != grid.shape:
        return None
    return grid


def generate_input() -> np.ndarray:
    rng = random.Random()
    for _ in range(MAX_GENERATION_ATTEMPTS):
        grid = _gen_attempt(rng)
        if grid is not None:
            return grid
    raise RuntimeError(f"failed to generate input after {MAX_GENERATION_ATTEMPTS} attempts")
