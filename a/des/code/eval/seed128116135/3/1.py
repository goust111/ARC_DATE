import random
from collections import deque

import numpy as np


BG = 0
FRAME = 8
CONTENT = 2
MIRROR = 3

DIRS = {
    "up": (-1, 0),
    "down": (1, 0),
    "left": (0, -1),
    "right": (0, 1),
}

GEN_RETRIES = 400
PLACE_RETRIES = 160
FRAME_MARGIN = 1
RESERVE_PAD = 1


def _bg_color(grid: np.ndarray) -> int:
    values, counts = np.unique(np.asarray(grid, dtype=int), return_counts=True)
    return int(values[np.argmax(counts)])


def _find_components(grid: np.ndarray, bg: int) -> list[dict]:
    g = np.asarray(grid, dtype=int)
    h, w = g.shape
    seen = np.zeros((h, w), dtype=bool)
    comps = []

    for r in range(h):
        for c in range(w):
            if seen[r, c] or int(g[r, c]) == int(bg):
                continue
            color = int(g[r, c])
            q = deque([(r, c)])
            seen[r, c] = True
            coords = []
            while q:
                rr, cc = q.popleft()
                coords.append((rr, cc))
                for dr, dc in ((-1, 0), (1, 0), (0, -1), (0, 1)):
                    nr, nc = rr + dr, cc + dc
                    if 0 <= nr < h and 0 <= nc < w and not seen[nr, nc] and int(g[nr, nc]) == color:
                        seen[nr, nc] = True
                        q.append((nr, nc))
            arr = np.asarray(coords, dtype=int)
            r0, c0 = arr.min(axis=0)
            r1, c1 = arr.max(axis=0)
            comps.append(
                {
                    "color": color,
                    "coords": arr,
                    "bbox": (int(r0), int(c0), int(r1), int(c1)),
                    "area": int(len(arr)),
                }
            )
    return comps


def _gap_run(side: np.ndarray) -> tuple[int, int] | None:
    if side.size < 4 or not bool(side[0]) or not bool(side[-1]):
        return None
    missing = np.flatnonzero(~side)
    if missing.size < 2:
        return None
    if missing[0] == 0 or missing[-1] == side.size - 1:
        return None
    if int(missing[-1] - missing[0] + 1) != int(missing.size):
        return None
    return int(missing[0]), int(missing[-1])


def _frame_info(grid: np.ndarray, comp: dict, bg: int) -> dict | None:
    g = np.asarray(grid, dtype=int)
    h, w = g.shape
    r0, c0, r1, c1 = comp["bbox"]
    color = int(comp["color"])
    if r0 == 0 or c0 == 0 or r1 == h - 1 or c1 == w - 1:
        return None

    patch = g[r0 : r1 + 1, c0 : c1 + 1]
    ph, pw = patch.shape
    if ph < 5 or pw < 5:
        return None

    color_mask = patch == color
    border = np.zeros_like(color_mask, dtype=bool)
    border[0, :] = True
    border[-1, :] = True
    border[:, 0] = True
    border[:, -1] = True
    if np.any(color_mask & ~border):
        return None

    sides = {
        "up": color_mask[0, :],
        "down": color_mask[-1, :],
        "left": color_mask[:, 0],
        "right": color_mask[:, -1],
    }
    full = {name: bool(np.all(side)) for name, side in sides.items()}
    gaps = {name: _gap_run(side) for name, side in sides.items()}
    gap_dirs = [name for name, run in gaps.items() if run is not None]
    if len(gap_dirs) != 1:
        return None
    gap_dir = gap_dirs[0]
    if any(not full[name] for name in sides if name != gap_dir):
        return None
    return {
        "color": color,
        "bbox": (r0, c0, r1, c1),
        "direction": gap_dir,
        "gap": gaps[gap_dir],
    }


def _extract_payload(grid: np.ndarray, frame: dict, bg: int) -> dict | None:
    g = np.asarray(grid, dtype=int)
    r0, c0, r1, c1 = frame["bbox"]
    frame_color = int(frame["color"])
    inner = g[r0 + 1 : r1, c0 + 1 : c1]
    mask = (inner != int(bg)) & (inner != frame_color)
    if not bool(mask.any()):
        return None
    coords = np.argwhere(mask)
    pr0, pc0 = coords.min(axis=0)
    pr1, pc1 = coords.max(axis=0)
    patch = np.full((int(pr1 - pr0 + 1), int(pc1 - pc0 + 1)), int(bg), dtype=int)
    for rr, cc in coords:
        patch[int(rr - pr0), int(cc - pc0)] = int(inner[rr, cc])
    abs_r0 = int(r0 + 1 + pr0)
    abs_c0 = int(c0 + 1 + pc0)
    return {
        "bbox": (abs_r0, abs_c0, abs_r0 + patch.shape[0] - 1, abs_c0 + patch.shape[1] - 1),
        "patch": patch,
        "coords": np.argwhere(patch != int(bg)),
    }


def _patch_fits(base: np.ndarray, patch: np.ndarray, r0: int, c0: int, bg: int) -> bool:
    h, w = patch.shape
    if r0 < 0 or c0 < 0 or r0 + h > base.shape[0] or c0 + w > base.shape[1]:
        return False
    coords = np.argwhere(patch != int(bg))
    for rr, cc in coords:
        if int(base[r0 + rr, c0 + cc]) != int(bg):
            return False
    return True


def _mirror_patch(payload_patch: np.ndarray, direction: str, bg: int) -> np.ndarray:
    if direction in ("left", "right"):
        mirrored = np.fliplr(payload_patch)
    else:
        mirrored = np.flipud(payload_patch)
    out = np.full_like(mirrored, int(bg))
    out[mirrored != int(bg)] = MIRROR
    return out


def _mirror_origin(r0: int, c0: int, patch: np.ndarray, direction: str) -> tuple[int, int]:
    h, w = patch.shape
    if direction == "right":
        return r0, c0 + w
    if direction == "left":
        return r0, c0 - w
    if direction == "down":
        return r0 + h, c0
    return r0 - h, c0


def _find_endpoint(base: np.ndarray, payload: dict, direction: str, bg: int) -> tuple[int, int]:
    start_r0, start_c0, _r1, _c1 = payload["bbox"]
    patch = payload["patch"]
    dr, dc = DIRS[direction]
    mirror = _mirror_patch(patch, direction, bg)
    best = None
    step = 0

    while True:
        r0 = int(start_r0 + dr * step)
        c0 = int(start_c0 + dc * step)
        if not _patch_fits(base, patch, r0, c0, bg):
            break
        mr0, mc0 = _mirror_origin(r0, c0, patch, direction)
        if _patch_fits(base, mirror, mr0, mc0, bg):
            best = (r0, c0)
        step += 1

    assert best is not None, "expected a valid endpoint with room for the mirror"
    return best


def _paint_patch(grid: np.ndarray, patch: np.ndarray, r0: int, c0: int, bg: int) -> None:
    coords = np.argwhere(patch != int(bg))
    for rr, cc in coords:
        grid[r0 + rr, c0 + cc] = int(patch[rr, cc])


def main(grid: np.ndarray) -> np.ndarray:
    """
    Move each inner payload out through its frame gap, then append a directional mirror copy.
    """
    g = np.asarray(grid, dtype=int)
    bg = _bg_color(g)

    frames = []
    for comp in _find_components(g, bg):
        info = _frame_info(g, comp, bg)
        if info is not None:
            payload = _extract_payload(g, info, bg)
            if payload is not None:
                info["payload"] = payload
                frames.append(info)

    if not frames:
        return g.copy()

    out = g.copy()
    for frame in frames:
        payload = frame["payload"]
        pr0, pc0, pr1, pc1 = payload["bbox"]
        out[pr0 : pr1 + 1, pc0 : pc1 + 1][payload["patch"] != int(bg)] = int(bg)

    base = out.copy()
    for frame in frames:
        direction = frame["direction"]
        payload = frame["payload"]
        end_r0, end_c0 = _find_endpoint(base, payload, direction, bg)
        _paint_patch(out, payload["patch"], end_r0, end_c0, bg)
        mirror = _mirror_patch(payload["patch"], direction, bg)
        mr0, mc0 = _mirror_origin(end_r0, end_c0, payload["patch"], direction)
        _paint_patch(out, mirror, mr0, mc0, bg)
    return out


def _crop_bool(mask: np.ndarray) -> np.ndarray:
    coords = np.argwhere(mask)
    r0, c0 = coords.min(axis=0)
    r1, c1 = coords.max(axis=0)
    return mask[int(r0) : int(r1) + 1, int(c0) : int(c1) + 1]


def _random_shape() -> np.ndarray:
    for _ in range(80):
        h = random.randint(2, 4)
        w = random.randint(2, 4)
        area = random.randint(max(3, (h * w) // 2), h * w)
        cells = {(random.randrange(h), random.randrange(w))}
        while len(cells) < area:
            frontier = []
            for r, c in cells:
                for dr, dc in ((-1, 0), (1, 0), (0, -1), (0, 1)):
                    nr, nc = r + dr, c + dc
                    if 0 <= nr < h and 0 <= nc < w and (nr, nc) not in cells:
                        frontier.append((nr, nc))
            cells.add(random.choice(frontier))
        mask = np.zeros((h, w), dtype=bool)
        for r, c in cells:
            mask[r, c] = True
        mask = _crop_bool(mask)
        if mask.shape[0] >= 2 and mask.shape[1] >= 2:
            return mask
    raise RuntimeError("failed to generate a payload shape")


def _build_frame_patch(height: int, width: int, direction: str, gap_start: int, gap_len: int) -> np.ndarray:
    patch = np.full((height, width), BG, dtype=int)
    patch[0, :] = FRAME
    patch[-1, :] = FRAME
    patch[:, 0] = FRAME
    patch[:, -1] = FRAME
    if direction == "up":
        patch[0, gap_start : gap_start + gap_len] = BG
    elif direction == "down":
        patch[-1, gap_start : gap_start + gap_len] = BG
    elif direction == "left":
        patch[gap_start : gap_start + gap_len, 0] = BG
    else:
        patch[gap_start : gap_start + gap_len, -1] = BG
    return patch


def _rect_clear(mask: np.ndarray, rect: tuple[int, int, int, int], pad: int) -> bool:
    r0, r1, c0, c1 = rect
    rr0 = max(0, r0 - pad)
    rr1 = min(mask.shape[0] - 1, r1 + pad)
    cc0 = max(0, c0 - pad)
    cc1 = min(mask.shape[1] - 1, c1 + pad)
    return not bool(mask[rr0 : rr1 + 1, cc0 : cc1 + 1].any())


def _mark_rect(mask: np.ndarray, rect: tuple[int, int, int, int], pad: int) -> None:
    r0, r1, c0, c1 = rect
    rr0 = max(0, r0 - pad)
    rr1 = min(mask.shape[0] - 1, r1 + pad)
    cc0 = max(0, c0 - pad)
    cc1 = min(mask.shape[1] - 1, c1 + pad)
    mask[rr0 : rr1 + 1, cc0 : cc1 + 1] = True


def _paint_rect(grid: np.ndarray, rect: tuple[int, int, int, int], color: int) -> None:
    r0, r1, c0, c1 = rect
    grid[r0 : r1 + 1, c0 : c1 + 1] = int(color)


def _try_place_object(grid: np.ndarray, reserved: np.ndarray, direction: str, blocker: bool) -> bool:
    h, w = grid.shape
    shape = _random_shape()
    sh, sw = shape.shape

    for _ in range(PLACE_RETRIES):
        inner_h = sh + random.randint(2, 4)
        inner_w = sw + random.randint(2, 5)
        inner_r = random.randint(1, inner_h - sh - 1)
        inner_c = random.randint(1, inner_w - sw - 1)
        fh, fw = inner_h + 2, inner_w + 2
        thickness = random.randint(1, 3) if blocker else 0

        frame_rect = None
        blocker_rect = None
        corridor_rect = None
        payload_r0 = payload_c0 = None

        if direction in ("left", "right"):
            if h - fh - FRAME_MARGIN < FRAME_MARGIN:
                continue
            frame_r0 = random.randint(FRAME_MARGIN, h - fh - FRAME_MARGIN)
            end_r0 = frame_r0 + 1 + inner_r
            if direction == "right":
                end_c0 = w - thickness - 2 * sw if blocker else w - 2 * sw
                max_frame_c0 = min(w - fw - FRAME_MARGIN, end_c0 - inner_c - 2)
                if max_frame_c0 < FRAME_MARGIN:
                    continue
                frame_c0 = random.randint(FRAME_MARGIN, max_frame_c0)
                start_c0 = frame_c0 + 1 + inner_c
                blocker_rect = (end_r0, end_r0 + sh - 1, w - thickness, w - 1) if blocker else None
                corridor_rect = (end_r0, end_r0 + sh - 1, start_c0, end_c0 + 2 * sw - 1)
            else:
                end_c0 = thickness + sw if blocker else sw
                min_frame_c0 = max(FRAME_MARGIN, end_c0 - inner_c)
                max_frame_c0 = w - fw - FRAME_MARGIN
                if min_frame_c0 > max_frame_c0:
                    continue
                frame_c0 = random.randint(min_frame_c0, max_frame_c0)
                start_c0 = frame_c0 + 1 + inner_c
                blocker_rect = (end_r0, end_r0 + sh - 1, 0, thickness - 1) if blocker else None
                corridor_rect = (end_r0, end_r0 + sh - 1, end_c0 - sw, start_c0 + sw - 1)

            frame_rect = (frame_r0, frame_r0 + fh - 1, frame_c0, frame_c0 + fw - 1)
            payload_r0 = frame_r0 + 1 + inner_r
            payload_c0 = frame_c0 + 1 + inner_c
            gap_start = 1 + inner_r
            gap_len = sh
        else:
            if w - fw - FRAME_MARGIN < FRAME_MARGIN:
                continue
            frame_c0 = random.randint(FRAME_MARGIN, w - fw - FRAME_MARGIN)
            end_c0 = frame_c0 + 1 + inner_c
            if direction == "down":
                end_r0 = h - thickness - 2 * sh if blocker else h - 2 * sh
                max_frame_r0 = min(h - fh - FRAME_MARGIN, end_r0 - inner_r - 2)
                if max_frame_r0 < FRAME_MARGIN:
                    continue
                frame_r0 = random.randint(FRAME_MARGIN, max_frame_r0)
                start_r0 = frame_r0 + 1 + inner_r
                blocker_rect = (h - thickness, h - 1, end_c0, end_c0 + sw - 1) if blocker else None
                corridor_rect = (start_r0, end_r0 + 2 * sh - 1, end_c0, end_c0 + sw - 1)
            else:
                end_r0 = thickness + sh if blocker else sh
                min_frame_r0 = max(FRAME_MARGIN, end_r0 - inner_r)
                max_frame_r0 = h - fh - FRAME_MARGIN
                if min_frame_r0 > max_frame_r0:
                    continue
                frame_r0 = random.randint(min_frame_r0, max_frame_r0)
                start_r0 = frame_r0 + 1 + inner_r
                blocker_rect = (0, thickness - 1, end_c0, end_c0 + sw - 1) if blocker else None
                corridor_rect = (end_r0 - sh, start_r0 + sh - 1, end_c0, end_c0 + sw - 1)

            frame_rect = (frame_r0, frame_r0 + fh - 1, frame_c0, frame_c0 + fw - 1)
            payload_r0 = frame_r0 + 1 + inner_r
            payload_c0 = frame_c0 + 1 + inner_c
            gap_start = 1 + inner_c
            gap_len = sw

        rects = [frame_rect, corridor_rect]
        if blocker_rect is not None:
            rects.append(blocker_rect)
        if any(not _rect_clear(reserved, rect, RESERVE_PAD) for rect in rects):
            continue

        frame_patch = _build_frame_patch(fh, fw, direction, gap_start, gap_len)
        _paint_patch(grid, frame_patch, frame_rect[0], frame_rect[2], BG)
        content_patch = np.full((sh, sw), BG, dtype=int)
        content_patch[shape] = CONTENT
        _paint_patch(grid, content_patch, payload_r0, payload_c0, BG)
        if blocker_rect is not None:
            _paint_rect(grid, blocker_rect, FRAME)

        for rect in rects:
            _mark_rect(reserved, rect, RESERVE_PAD)
        return True

    return False


def _gen_attempt() -> np.ndarray | None:
    height = random.randint(22, 30)
    width = random.randint(22, 30)
    grid = np.full((height, width), BG, dtype=int)
    reserved = np.zeros((height, width), dtype=bool)

    n_objects = random.randint(1, 4)
    directions = [random.choice(["up", "down", "left", "right"]) for _ in range(n_objects)]
    blockers = [bool(random.getrandbits(1)) for _ in range(n_objects)]

    for direction, blocker in zip(directions, blockers):
        if not _try_place_object(grid, reserved, direction, blocker):
            return None

    try:
        out = main(grid)
    except AssertionError:
        return None

    if np.array_equal(grid, out):
        return None
    if not bool(np.any(out == MIRROR)):
        return None
    return grid


def generate_input() -> np.ndarray:
    for _ in range(GEN_RETRIES):
        grid = _gen_attempt()
        if grid is not None:
            return grid
    raise RuntimeError("failed to generate a valid input")
