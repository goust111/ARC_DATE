import random
from collections import Counter, deque

import numpy as np


BG = 4
SHADOW = 5
FRAME_COLORS = [1, 2, 3, 6, 7, 8, 9]


def _component_coords(mask: np.ndarray) -> list[list[tuple[int, int]]]:
    h, w = mask.shape
    seen = np.zeros((h, w), dtype=bool)
    comps: list[list[tuple[int, int]]] = []
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
                for dr, dc in ((1, 0), (-1, 0), (0, 1), (0, -1)):
                    nr, nc = rr + dr, cc + dc
                    if 0 <= nr < h and 0 <= nc < w and mask[nr, nc] and not seen[nr, nc]:
                        seen[nr, nc] = True
                        q.append((nr, nc))
            comps.append(coords)
    return comps


def _border_coords(r0: int, c0: int, side: int) -> list[tuple[int, int]]:
    pts: list[tuple[int, int]] = []
    r1, c1 = r0 + side - 1, c0 + side - 1
    for c in range(c0, c1 + 1):
        pts.append((r0, c))
        pts.append((r1, c))
    for r in range(r0 + 1, r1):
        pts.append((r, c0))
        pts.append((r, c1))
    return pts


def _detect_frames(grid: np.ndarray) -> list[dict]:
    frames: list[dict] = []
    h, w = grid.shape
    for color in sorted(set(map(int, grid.ravel())) - {BG, SHADOW}):
        for coords in _component_coords(grid == color):
            rows = [r for r, _ in coords]
            cols = [c for _, c in coords]
            r0, r1 = min(rows), max(rows)
            c0, c1 = min(cols), max(cols)
            side = r1 - r0 + 1
            if side < 4 or side != c1 - c0 + 1:
                continue
            border = _border_coords(r0, c0, side)
            if len(coords) != len(border):
                continue
            patch = grid[r0 : r1 + 1, c0 : c1 + 1]
            if not np.all(patch[0, :] == color) or not np.all(patch[-1, :] == color):
                continue
            if not np.all(patch[:, 0] == color) or not np.all(patch[:, -1] == color):
                continue
            if np.any(patch[1:-1, 1:-1] == color):
                continue
            frames.append(
                {
                    "color": color,
                    "r0": r0,
                    "c0": c0,
                    "side": side,
                    "border": border,
                }
            )
    return frames


def _score_vector(grid: np.ndarray, frames: list[dict], dr: int, dc: int) -> tuple[int, int]:
    h, w = grid.shape
    supporting = 0
    matches = 0
    for frame in frames:
        count = 0
        for r, c in frame["border"]:
            rr, cc = r + dr, c + dc
            if 0 <= rr < h and 0 <= cc < w and int(grid[rr, cc]) == SHADOW:
                count += 1
        if count > 0:
            supporting += 1
            matches += count
    return supporting, matches


def _infer_vector(grid: np.ndarray, frames: list[dict]) -> tuple[int, int]:
    h, w = grid.shape
    best_vec: tuple[int, int] | None = None
    best_score: tuple[int, int, int] | None = None
    for k in range(1, max(h, w)):
        for sr in (-1, 1):
            for sc in (-1, 1):
                dr, dc = sr * k, sc * k
                supporting, matches = _score_vector(grid, frames, dr, dc)
                score = (supporting, matches, -k)
                if best_score is None or score > best_score:
                    best_score = score
                    best_vec = (dr, dc)
    assert best_vec is not None and best_score is not None and best_score[0] > 0
    return best_vec


def main(grid: np.ndarray) -> np.ndarray:
    g = np.array(grid, dtype=int)
    out = g.copy()
    frames = _detect_frames(g)
    if not frames:
        return out

    dr, dc = _infer_vector(g, frames)
    h, w = g.shape
    for frame in frames:
        existing = 0
        translated: list[tuple[int, int]] = []
        for r, c in frame["border"]:
            rr, cc = r + dr, c + dc
            if 0 <= rr < h and 0 <= cc < w:
                translated.append((rr, cc))
                if int(g[rr, cc]) == SHADOW:
                    existing += 1
        if existing > 0:
            continue
        for rr, cc in translated:
            if int(out[rr, cc]) == BG:
                out[rr, cc] = SHADOW
    return out


def _rects_overlap(a: tuple[int, int, int, int], b: tuple[int, int, int, int], pad: int = 0) -> bool:
    ar0, ac0, ar1, ac1 = a
    br0, bc0, br1, bc1 = b
    return not (ar1 + pad < br0 or br1 + pad < ar0 or ac1 + pad < bc0 or bc1 + pad < ac0)


def _shadow_free_count(frame: dict, frames: list[dict], dr: int, dc: int, h: int, w: int) -> int:
    occupied = set()
    for other in frames:
        occupied.update(other["border"])
    count = 0
    for r, c in frame["border"]:
        rr, cc = r + dr, c + dc
        if 0 <= rr < h and 0 <= cc < w and (rr, cc) not in occupied:
            count += 1
    return count


def _draw_frame(grid: np.ndarray, frame: dict) -> None:
    for r, c in frame["border"]:
        grid[r, c] = frame["color"]


def _frame_bbox(frame: dict) -> tuple[int, int, int, int]:
    return (frame["r0"], frame["c0"], frame["r0"] + frame["side"] - 1, frame["c0"] + frame["side"] - 1)


def _make_frame(color: int, side: int, r0: int, c0: int) -> dict:
    return {"color": color, "r0": r0, "c0": c0, "side": side, "border": _border_coords(r0, c0, side)}


def _can_place(frame: dict, frames: list[dict], forbidden: set[tuple[int, int]] | None = None) -> bool:
    if any(_rects_overlap(_frame_bbox(frame), _frame_bbox(other), pad=1) for other in frames):
        return False
    if forbidden is not None and any(pt in forbidden for pt in frame["border"]):
        return False
    return True


def _gen_attempt() -> np.ndarray | None:
    h = random.randint(18, 26)
    w = random.randint(18, 26)
    k = random.randint(1, 4)
    dr = random.choice([-k, k])
    dc = random.choice([-k, k])
    n_frames = random.randint(3, 5)
    want_occlusion = random.random() < 0.5

    frames: list[dict] = []
    colors = random.sample(FRAME_COLORS, n_frames)
    reserved_demo_shadow: set[tuple[int, int]] = set()

    def sample_frame(color: int, side_lo: int = 4, side_hi: int = 7) -> dict | None:
        for _ in range(250):
            side = random.randint(side_lo, side_hi)
            r0_min = max(0, -dr)
            c0_min = max(0, -dc)
            r0_max = h - side - max(0, dr)
            c0_max = w - side - max(0, dc)
            if r0_min > r0_max or c0_min > c0_max:
                return None
            frame = _make_frame(color, side, random.randint(r0_min, r0_max), random.randint(c0_min, c0_max))
            if _can_place(frame, frames, reserved_demo_shadow):
                return frame
        return None

    missing_frame = sample_frame(colors[0])
    if missing_frame is None:
        return None
    frames.append(missing_frame)

    if want_occlusion:
        shadow_pts = {(r + dr, c + dc) for r, c in missing_frame["border"]}
        blocker = None
        for _ in range(400):
            side = random.randint(4, 7)
            r0_min = max(0, missing_frame["r0"] + dr - side + 1)
            c0_min = max(0, missing_frame["c0"] + dc - side + 1)
            r0_max = min(h - side, missing_frame["r0"] + dr + missing_frame["side"] - 1)
            c0_max = min(w - side, missing_frame["c0"] + dc + missing_frame["side"] - 1)
            if r0_min > r0_max or c0_min > c0_max:
                continue
            cand = _make_frame(colors[1], side, random.randint(r0_min, r0_max), random.randint(c0_min, c0_max))
            overlap = len(shadow_pts & set(cand["border"]))
            if 2 <= overlap < len(missing_frame["border"]) - 2 and _can_place(cand, frames, reserved_demo_shadow):
                blocker = cand
                break
        if blocker is None:
            return None
        frames.append(blocker)

    demo_color_index = 2 if want_occlusion else 1
    demo_frame = None
    for _ in range(300):
        cand = sample_frame(colors[demo_color_index])
        if cand is None:
            continue
        if _shadow_free_count(cand, frames + [cand], dr, dc, h, w) != len(cand["border"]):
            continue
        demo_frame = cand
        break
    if demo_frame is None:
        return None
    frames.append(demo_frame)
    reserved_demo_shadow.update((r + dr, c + dc) for r, c in demo_frame["border"])

    for color in colors[len(frames) :]:
        extra = sample_frame(color)
        if extra is None:
            return None
        frames.append(extra)

    frame_infos = [(frame, _shadow_free_count(frame, frames, dr, dc, h, w)) for frame in frames]
    demos = [demo_frame]
    for frame, free_count in frame_infos:
        if frame is demo_frame:
            continue
        if free_count >= max(4, len(frame["border"]) // 3) and random.random() < 0.4:
            demos.append(frame)
    demo_set = {id(frame) for frame in demos}
    missing_frames = [frame for frame, _ in frame_infos if id(frame) not in demo_set]
    if not missing_frames:
        return None
    if want_occlusion:
        missing_free = [_shadow_free_count(frame, frames, dr, dc, h, w) for frame in missing_frames]
        if not any(0 < free < len(frame["border"]) for frame, free in zip(missing_frames, missing_free)):
            return None

    grid = np.full((h, w), BG, dtype=int)
    for frame in frames:
        _draw_frame(grid, frame)
    for frame in frames:
        if id(frame) not in demo_set:
            continue
        for r, c in frame["border"]:
            rr, cc = r + dr, c + dc
            if 0 <= rr < h and 0 <= cc < w and int(grid[rr, cc]) == BG:
                grid[rr, cc] = SHADOW

    if len(set(map(int, grid.ravel()))) < 4:
        return None
    return grid


def generate_input() -> np.ndarray:
    for _ in range(400):
        grid = _gen_attempt()
        if grid is not None:
            return grid
    raise RuntimeError("generate_input failed after 400 attempts")
