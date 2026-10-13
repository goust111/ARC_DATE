import random
from collections import deque

import numpy as np


BLACK = 0
RED = 2
GREEN = 3
SIDES = ("top", "right", "bottom", "left")
BODY_COLORS = (1, 4, 5, 6, 7, 8, 9)
STAMP_SIZES = (3, 4, 5)
GEN_RETRIES = 400


def _components(mask: np.ndarray) -> list[np.ndarray]:
    seen = np.zeros(mask.shape, dtype=bool)
    comps: list[list[tuple[int, int]]] = []
    h, w = mask.shape
    for sr, sc in np.argwhere(mask):
        sr, sc = int(sr), int(sc)
        if seen[sr, sc]:
            continue
        q: deque[tuple[int, int]] = deque([(sr, sc)])
        seen[sr, sc] = True
        pts: list[tuple[int, int]] = []
        while q:
            r, c = q.popleft()
            pts.append((r, c))
            for dr, dc in ((-1, 0), (1, 0), (0, -1), (0, 1)):
                nr, nc = r + dr, c + dc
                if 0 <= nr < h and 0 <= nc < w and mask[nr, nc] and not seen[nr, nc]:
                    seen[nr, nc] = True
                    q.append((nr, nc))
        comps.append(pts)
    return [np.array(comp, dtype=int) for comp in comps]


def _bbox(coords: np.ndarray) -> tuple[int, int, int, int]:
    return (
        int(coords[:, 0].min()),
        int(coords[:, 1].min()),
        int(coords[:, 0].max()),
        int(coords[:, 1].max()),
    )


def _edge_coords(r: int, c: int, h: int, w: int, side: str) -> list[tuple[int, int]]:
    if side == "top":
        return [(r, cc) for cc in range(c, c + w)]
    if side == "bottom":
        return [(r + h - 1, cc) for cc in range(c, c + w)]
    if side == "left":
        return [(rr, c) for rr in range(r, r + h)]
    if side == "right":
        return [(rr, c + w - 1) for rr in range(r, r + h)]
    raise ValueError(side)


def _red_segment_side(grid: np.ndarray, r: int, c: int, h: int, w: int) -> str | None:
    red = {(int(rr), int(cc)) for rr, cc in np.argwhere(grid[r : r + h, c : c + w] == RED)}
    red = {(rr + r, cc + c) for rr, cc in red}
    if not red:
        return None

    candidates: list[str] = []
    for side in SIDES:
        edge = _edge_coords(r, c, h, w, side)
        edge_set = set(edge)
        if not red.issubset(edge_set):
            continue
        idxs = [i for i, pt in enumerate(edge) if pt in red]
        if idxs and max(idxs) - min(idxs) + 1 == len(idxs):
            candidates.append(side)
    return candidates[0] if len(candidates) == 1 else None


def _find_frames(grid: np.ndarray) -> list[dict]:
    h, w = grid.shape
    frames: list[dict] = []
    seen_masks: set[tuple[tuple[int, int], ...]] = set()
    for inner_h in range(3, 6):
        for inner_w in range(3, 6):
            fh, fw = inner_h + 2, inner_w + 2
            for r in range(h - fh + 1):
                for c in range(w - fw + 1):
                    patch = grid[r : r + fh, c : c + fw]
                    if not np.all(patch[1:-1, 1:-1] == BLACK):
                        continue

                    border = np.ones((fh, fw), dtype=bool)
                    border[1:-1, 1:-1] = False
                    vals = set(map(int, patch[border]))
                    if not vals.issubset({GREEN, RED}) or RED not in vals:
                        continue

                    side = _red_segment_side(grid, r, c, fh, fw)
                    if side is None:
                        continue
                    coords = tuple(sorted((rr, cc) for rr, cc in np.argwhere(border)))
                    abs_coords = tuple((rr + r, cc + c) for rr, cc in coords)
                    if abs_coords in seen_masks:
                        continue
                    seen_masks.add(abs_coords)
                    frames.append(
                        {
                            "outer": (r, c, fh, fw),
                            "inner": (r + 1, c + 1, inner_h, inner_w),
                            "shape": (inner_h, inner_w),
                            "side": side,
                            "mask": abs_coords,
                        }
                    )
    frames.sort(key=lambda item: item["outer"])
    return frames


def _stamp_red_side(stamp: np.ndarray) -> str | None:
    checks = {
        "top": np.all(stamp[0, :] == RED),
        "right": np.all(stamp[:, -1] == RED),
        "bottom": np.all(stamp[-1, :] == RED),
        "left": np.all(stamp[:, 0] == RED),
    }
    sides = [side for side, ok in checks.items() if bool(ok)]
    return sides[0] if len(sides) == 1 else None


def _rotations_to_side(source_side: str, target_side: str) -> int:
    source_idx = SIDES.index(source_side)
    target_idx = SIDES.index(target_side)
    return (source_idx - target_idx) % 4


def _find_stamps(grid: np.ndarray, frame_mask: np.ndarray) -> list[dict]:
    stamps: list[dict] = []
    for coords in _components((grid != BLACK) & (~frame_mask)):
        r0, c0, r1, c1 = _bbox(coords)
        h, w = r1 - r0 + 1, c1 - c0 + 1
        if not (3 <= h <= 5 and 3 <= w <= 5):
            continue
        if len(coords) != h * w:
            continue
        crop = grid[r0 : r1 + 1, c0 : c1 + 1]
        if not np.all(crop != BLACK):
            continue
        side = _stamp_red_side(crop)
        if side is None:
            continue
        stamps.append({"bbox": (r0, c0, h, w), "shape": (h, w), "side": side, "array": crop.copy()})
    stamps.sort(key=lambda item: item["bbox"])
    return stamps


def main(grid: np.ndarray) -> np.ndarray:
    """Rotate each stamp to the matching canvas marker side and paste it inside the canvas frame.

    Input structure:
    - The background is black.
    - Small filled 3x3 to 5x5 stamp templates are multicolored rectangles; exactly one full edge of each stamp is red.
    - Larger one-pixel hollow canvas frames have green borders, black interiors, and exactly one continuous red segment on one frame side.
    - Every canvas interior size uniquely matches one stamp size.

    Rule:
    - Pair each stamp with the canvas whose interior has the same size.
    - Read the frame side containing the red segment.
    - Rotate the stamp by a quarter-turn amount that moves its all-red edge to that same side.
    - Paint the rotated stamp into the canvas interior, keep the frame border unchanged, and erase the stamp's original position to black.

    Key insight:
    - The stamp's full red edge and the canvas frame's red segment are matching orientation anchors; size links each stamp to its target canvas.
    """
    g = np.asarray(grid, dtype=int)
    assert g.ndim == 2

    frames = _find_frames(g)
    assert frames

    frame_mask = np.zeros(g.shape, dtype=bool)
    for frame in frames:
        for r, c in frame["mask"]:
            frame_mask[r, c] = True

    stamps = _find_stamps(g, frame_mask)
    assert len(stamps) == len(frames)

    out = g.copy()
    unused = set(range(len(stamps)))
    for frame in frames:
        candidates: list[tuple[int, int, np.ndarray]] = []
        for idx in sorted(unused):
            stamp = stamps[idx]
            if stamp["shape"] != frame["shape"]:
                continue
            k = _rotations_to_side(stamp["side"], frame["side"])
            rotated = np.rot90(stamp["array"], k)
            if rotated.shape == frame["shape"]:
                candidates.append((idx, k, rotated))

        if not candidates:
            for idx in sorted(unused):
                stamp = stamps[idx]
                k = _rotations_to_side(stamp["side"], frame["side"])
                rotated = np.rot90(stamp["array"], k)
                if rotated.shape == frame["shape"]:
                    candidates.append((idx, k, rotated))

        assert len(candidates) == 1
        idx, _k, rotated = candidates[0]
        unused.remove(idx)

        sr, sc, sh, sw = stamps[idx]["bbox"]
        out[sr : sr + sh, sc : sc + sw] = BLACK
        ir, ic, ih, iw = frame["inner"]
        out[ir : ir + ih, ic : ic + iw] = rotated

    return out


def _paint_frame(grid: np.ndarray, r: int, c: int, size: int, side: str) -> None:
    fh = fw = size + 2
    grid[r, c : c + fw] = GREEN
    grid[r + fh - 1, c : c + fw] = GREEN
    grid[r : r + fh, c] = GREEN
    grid[r : r + fh, c + fw - 1] = GREEN

    length = random.randint(1, size)
    start = random.randint(1, size - length + 1)
    if side == "top":
        grid[r, c + start : c + start + length] = RED
    elif side == "bottom":
        grid[r + fh - 1, c + start : c + start + length] = RED
    elif side == "left":
        grid[r + start : r + start + length, c] = RED
    elif side == "right":
        grid[r + start : r + start + length, c + fw - 1] = RED
    else:
        raise ValueError(side)


def _make_stamp(size: int, red_side: str) -> np.ndarray:
    colors = random.sample(BODY_COLORS, random.randint(2, 4))
    stamp = np.empty((size, size), dtype=int)
    for r in range(size):
        for c in range(size):
            stamp[r, c] = colors[(r * size + c + random.randrange(len(colors))) % len(colors)]

    if red_side == "top":
        stamp[0, :] = RED
    elif red_side == "bottom":
        stamp[-1, :] = RED
    elif red_side == "left":
        stamp[:, 0] = RED
    elif red_side == "right":
        stamp[:, -1] = RED
    else:
        raise ValueError(red_side)

    # Keep all non-anchor cells away from red so the full red edge is the only orientation marker.
    for r in range(size):
        for c in range(size):
            on_anchor = (
                (red_side == "top" and r == 0)
                or (red_side == "bottom" and r == size - 1)
                or (red_side == "left" and c == 0)
                or (red_side == "right" and c == size - 1)
            )
            if not on_anchor and stamp[r, c] == RED:
                stamp[r, c] = colors[0]
    return stamp


def _free_location(occupied: np.ndarray, ph: int, pw: int, padding: int = 1) -> tuple[int, int] | None:
    h, w = occupied.shape
    locs: list[tuple[int, int]] = []
    for r in range(h - ph + 1):
        for c in range(w - pw + 1):
            r0, r1 = max(0, r - padding), min(h, r + ph + padding)
            c0, c1 = max(0, c - padding), min(w, c + pw + padding)
            if not occupied[r0:r1, c0:c1].any():
                locs.append((r, c))
    return random.choice(locs) if locs else None


def _reserve(occupied: np.ndarray, r: int, c: int, h: int, w: int, padding: int = 1) -> None:
    r0, r1 = max(0, r - padding), min(occupied.shape[0], r + h + padding)
    c0, c1 = max(0, c - padding), min(occupied.shape[1], c + w + padding)
    occupied[r0:r1, c0:c1] = True


def _expected_output(inp: np.ndarray, placements: list[dict]) -> np.ndarray:
    out = inp.copy()
    for item in placements:
        sr, sc = item["stamp_pos"]
        size = item["size"]
        out[sr : sr + size, sc : sc + size] = BLACK
    for item in placements:
        fr, fc = item["frame_pos"]
        out[fr + 1 : fr + 1 + item["size"], fc + 1 : fc + 1 + item["size"]] = item["rotated"]
    return out


def _gen_attempt() -> np.ndarray | None:
    pair_count = random.randint(1, 3)
    sizes = random.sample(STAMP_SIZES, pair_count)
    height = random.randint(16 + 2 * pair_count, 30)
    width = random.randint(16 + 2 * pair_count, 30)

    grid = np.full((height, width), BLACK, dtype=int)
    occupied = np.zeros((height, width), dtype=bool)
    placements: list[dict] = []

    jobs = []
    for size in sizes:
        source_side = random.choice(SIDES)
        target_side = random.choice(SIDES)
        stamp = _make_stamp(size, source_side)
        k = _rotations_to_side(source_side, target_side)
        rotated = np.rot90(stamp, k)
        jobs.append((size, source_side, target_side, stamp, rotated))
    random.shuffle(jobs)

    for size, source_side, target_side, stamp, rotated in jobs:
        loc = _free_location(occupied, size, size, padding=1)
        if loc is None:
            return None
        sr, sc = loc
        grid[sr : sr + size, sc : sc + size] = stamp
        _reserve(occupied, sr, sc, size, size, padding=1)

        frame_size = size + 2
        loc = _free_location(occupied, frame_size, frame_size, padding=1)
        if loc is None:
            return None
        fr, fc = loc
        _paint_frame(grid, fr, fc, size, target_side)
        _reserve(occupied, fr, fc, frame_size, frame_size, padding=1)

        placements.append(
            {
                "size": size,
                "source_side": source_side,
                "target_side": target_side,
                "stamp_pos": (sr, sc),
                "frame_pos": (fr, fc),
                "rotated": rotated,
            }
        )

    expected = _expected_output(grid, placements)
    try:
        actual = main(grid)
    except Exception:
        return None
    if not np.array_equal(actual, expected):
        return None
    return grid


def generate_input() -> np.ndarray:
    for _ in range(GEN_RETRIES):
        grid = _gen_attempt()
        if grid is not None:
            return grid
    raise RuntimeError("failed to generate valid stamp-canvas input")
