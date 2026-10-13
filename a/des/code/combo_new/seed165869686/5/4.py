import random
from collections import deque

import numpy as np


DIRS = ("up", "down", "left", "right")
COLOR_POOL = tuple(range(1, 10))
GEN_RETRIES = 400


def _background_color(grid: np.ndarray) -> int:
    values, counts = np.unique(grid, return_counts=True)
    return int(values[np.argmax(counts)])


def _color_components(grid: np.ndarray, color: int) -> list[dict[str, object]]:
    mask = grid == color
    h, w = grid.shape
    seen = np.zeros((h, w), dtype=bool)
    comps: list[dict[str, object]] = []

    for r in range(h):
        for c in range(w):
            if not mask[r, c] or seen[r, c]:
                continue
            q = deque([(r, c)])
            seen[r, c] = True
            coords: list[tuple[int, int]] = []
            while q:
                cr, cc = q.popleft()
                coords.append((cr, cc))
                for nr, nc in ((cr - 1, cc), (cr + 1, cc), (cr, cc - 1), (cr, cc + 1)):
                    if 0 <= nr < h and 0 <= nc < w and mask[nr, nc] and not seen[nr, nc]:
                        seen[nr, nc] = True
                        q.append((nr, nc))

            rows = [p[0] for p in coords]
            cols = [p[1] for p in coords]
            r0, r1 = min(rows), max(rows)
            c0, c1 = min(cols), max(cols)
            crop = mask[r0 : r1 + 1, c0 : c1 + 1]
            comps.append(
                {
                    "color": int(color),
                    "coords": coords,
                    "bbox": (r0, c0, r1, c1),
                    "mask": crop,
                }
            )
    return comps


def _u_frame_direction(mask: np.ndarray) -> str | None:
    h, w = mask.shape
    if h < 3 or w < 3:
        return None
    if np.any(mask[1:-1, 1:-1]):
        return None

    top = mask[0, :]
    bottom = mask[-1, :]
    left = mask[:, 0]
    right = mask[:, -1]
    candidates: list[str] = []

    if bottom.all() and left.all() and right.all() and not top[1:-1].any():
        candidates.append("up")
    if top.all() and left.all() and right.all() and not bottom[1:-1].any():
        candidates.append("down")
    if top.all() and bottom.all() and right.all() and not left[1:-1].any():
        candidates.append("left")
    if top.all() and bottom.all() and left.all() and not right[1:-1].any():
        candidates.append("right")

    if len(candidates) != 1:
        return None
    return candidates[0]


def _find_u_frames(grid: np.ndarray, bg: int) -> list[dict[str, object]]:
    frames: list[dict[str, object]] = []
    for color in np.unique(grid):
        color = int(color)
        if color == bg:
            continue
        for comp in _color_components(grid, color):
            direction = _u_frame_direction(comp["mask"])
            if direction is None:
                continue
            frame = dict(comp)
            frame["direction"] = direction
            frames.append(frame)
    frames.sort(key=lambda comp: tuple(comp["bbox"]))  # deterministic order
    return frames


def _pattern_bbox(mask: np.ndarray) -> tuple[int, int, int, int]:
    pts = np.argwhere(mask)
    r0, c0 = pts.min(axis=0)
    r1, c1 = pts.max(axis=0)
    return int(r0), int(c0), int(r1), int(c1)


def main(grid: np.ndarray) -> np.ndarray:
    """
    Identify each U-shaped frame, mirror its interior pattern across the axis perpendicular to the opening, and eject it outside the opening.

    Input structure:
    - The grid has a black background and contains several monochrome U-shaped rectangular frames, each with exactly one open side.
    - Every frame uses a unique color and encloses a smaller non-background pattern separated from the frame by at least one black cell.
    - Openings may face up, down, left, or right.

    Rule:
    - Detect the missing side of each U-shaped frame to determine its opening direction.
    - Extract the non-background pixels inside the frame, mirror them in their tight bounding box, and use a left-right flip for left/right openings or an up-down flip for up/down openings.
    - Remove the original interior pattern and translate the mirrored pattern straight through the opening so its nearest edge sits immediately outside the frame.
    - Keep every U-shaped frame unchanged.

    Key insight:
    - The opening direction controls two independent operations at once: which mirror axis to use and which outward translation direction to apply.
    """
    g = np.asarray(grid, dtype=int)
    bg = _background_color(g)
    out = g.copy()

    for frame in _find_u_frames(g, bg):
        r0, c0, r1, c1 = frame["bbox"]
        frame_color = int(frame["color"])
        direction = str(frame["direction"])

        inner = g[r0 + 1 : r1, c0 + 1 : c1]
        inner_mask = (inner != bg) & (inner != frame_color)
        if not inner_mask.any():
            continue

        ir0, ic0, ir1, ic1 = _pattern_bbox(inner_mask)
        pr0, pc0 = r0 + 1 + ir0, c0 + 1 + ic0
        pr1, pc1 = r0 + 1 + ir1, c0 + 1 + ic1

        patch = g[pr0 : pr1 + 1, pc0 : pc1 + 1].copy()
        patch_mask = (patch != bg) & (patch != frame_color)

        if direction in ("left", "right"):
            patch = np.fliplr(patch)
            patch_mask = np.fliplr(patch_mask)
        else:
            patch = np.flipud(patch)
            patch_mask = np.flipud(patch_mask)

        src = out[pr0 : pr1 + 1, pc0 : pc1 + 1]
        src[(g[pr0 : pr1 + 1, pc0 : pc1 + 1] != bg) & (g[pr0 : pr1 + 1, pc0 : pc1 + 1] != frame_color)] = bg

        if direction == "up":
            nr0, nc0 = r0 - patch.shape[0], pc0
        elif direction == "down":
            nr0, nc0 = r1 + 1, pc0
        elif direction == "left":
            nr0, nc0 = pr0, c0 - patch.shape[1]
        else:
            nr0, nc0 = pr0, c1 + 1

        nr1 = nr0 + patch.shape[0]
        nc1 = nc0 + patch.shape[1]
        assert 0 <= nr0 and nr1 <= g.shape[0] and 0 <= nc0 and nc1 <= g.shape[1]

        dest = out[nr0:nr1, nc0:nc1]
        assert np.all(dest[patch_mask] == bg)
        dest[patch_mask] = patch[patch_mask]

    return out


def _u_mask(height: int, width: int, direction: str) -> np.ndarray:
    mask = np.zeros((height, width), dtype=bool)
    if direction == "up":
        mask[:, 0] = True
        mask[:, -1] = True
        mask[-1, :] = True
    elif direction == "down":
        mask[:, 0] = True
        mask[:, -1] = True
        mask[0, :] = True
    elif direction == "left":
        mask[0, :] = True
        mask[-1, :] = True
        mask[:, -1] = True
    else:
        mask[0, :] = True
        mask[-1, :] = True
        mask[:, 0] = True
    return mask


def _crop_mask(mask: np.ndarray) -> np.ndarray:
    pts = np.argwhere(mask)
    r0, c0 = pts.min(axis=0)
    r1, c1 = pts.max(axis=0)
    return mask[r0 : r1 + 1, c0 : c1 + 1]


def _connected(mask: np.ndarray) -> bool:
    pts = np.argwhere(mask)
    if len(pts) == 0:
        return False
    start = tuple(int(x) for x in pts[0])
    q = deque([start])
    seen = {start}
    while q:
        r, c = q.popleft()
        for nr, nc in ((r - 1, c), (r + 1, c), (r, c - 1), (r, c + 1)):
            if 0 <= nr < mask.shape[0] and 0 <= nc < mask.shape[1] and mask[nr, nc]:
                nxt = (nr, nc)
                if nxt not in seen:
                    seen.add(nxt)
                    q.append(nxt)
    return len(seen) == int(mask.sum())


def _random_pattern(rng: random.Random, direction: str) -> np.ndarray:
    for _ in range(200):
        h = rng.randint(2, 4)
        w = rng.randint(2, 4)
        count = rng.randint(2, min(h * w - 1, 6))
        mask = np.zeros((h, w), dtype=bool)
        start = (rng.randrange(h), rng.randrange(w))
        mask[start] = True
        frontier = {start}
        while int(mask.sum()) < count:
            base = rng.choice(tuple(frontier))
            neighbors = []
            for nr, nc in ((base[0] - 1, base[1]), (base[0] + 1, base[1]), (base[0], base[1] - 1), (base[0], base[1] + 1)):
                if 0 <= nr < h and 0 <= nc < w and not mask[nr, nc]:
                    neighbors.append((nr, nc))
            if not neighbors:
                frontier.discard(base)
                if not frontier:
                    break
                continue
            chosen = rng.choice(neighbors)
            mask[chosen] = True
            frontier.add(chosen)

        if int(mask.sum()) < 2 or not _connected(mask):
            continue
        mask = _crop_mask(mask)
        if direction in ("left", "right"):
            if np.array_equal(mask, np.fliplr(mask)):
                continue
        else:
            if np.array_equal(mask, np.flipud(mask)):
                continue
        if np.array_equal(mask, np.fliplr(mask)) and np.array_equal(mask, np.flipud(mask)):
            continue
        return mask
    raise RuntimeError("failed to build asymmetric pattern")


def _destination_bbox(
    frame_bbox: tuple[int, int, int, int],
    pattern_bbox: tuple[int, int, int, int],
    direction: str,
) -> tuple[int, int, int, int]:
    r0, c0, r1, c1 = frame_bbox
    pr0, pc0, pr1, pc1 = pattern_bbox
    ph = pr1 - pr0 + 1
    pw = pc1 - pc0 + 1
    if direction == "up":
        return (r0 - ph, pc0, r0 - 1, pc1)
    if direction == "down":
        return (r1 + 1, pc0, r1 + ph, pc1)
    if direction == "left":
        return (pr0, c0 - pw, pr1, c0 - 1)
    return (pr0, c1 + 1, pr1, c1 + pw)


def _mark_forbidden(occupied: np.ndarray, bbox: tuple[int, int, int, int], pad: int = 1) -> None:
    r0, c0, r1, c1 = bbox
    rr0 = max(0, r0 - pad)
    cc0 = max(0, c0 - pad)
    rr1 = min(occupied.shape[0] - 1, r1 + pad)
    cc1 = min(occupied.shape[1] - 1, c1 + pad)
    occupied[rr0 : rr1 + 1, cc0 : cc1 + 1] = True


def _free(occupied: np.ndarray, bbox: tuple[int, int, int, int], pad: int = 1) -> bool:
    r0, c0, r1, c1 = bbox
    rr0 = max(0, r0 - pad)
    cc0 = max(0, c0 - pad)
    rr1 = min(occupied.shape[0] - 1, r1 + pad)
    cc1 = min(occupied.shape[1] - 1, c1 + pad)
    return not occupied[rr0 : rr1 + 1, cc0 : cc1 + 1].any()


def _bbox_union(a: tuple[int, int, int, int], b: tuple[int, int, int, int]) -> tuple[int, int, int, int]:
    return (min(a[0], b[0]), min(a[1], b[1]), max(a[2], b[2]), max(a[3], b[3]))


def _draw_object(
    grid: np.ndarray,
    frame_color: int,
    pattern_color: int,
    direction: str,
    frame_bbox: tuple[int, int, int, int],
    pattern_offset: tuple[int, int],
    pattern_mask: np.ndarray,
) -> None:
    r0, c0, r1, c1 = frame_bbox
    h = r1 - r0 + 1
    w = c1 - c0 + 1
    frame = _u_mask(h, w, direction)
    grid[r0 : r1 + 1, c0 : c1 + 1][frame] = frame_color

    pr, pc = pattern_offset
    ph, pw = pattern_mask.shape
    patch = grid[r0 + pr : r0 + pr + ph, c0 + pc : c0 + pc + pw]
    patch[pattern_mask] = pattern_color


def _gen_attempt(rng: random.Random) -> np.ndarray | None:
    height = rng.randint(18, 30)
    width = rng.randint(18, 30)
    grid = np.zeros((height, width), dtype=int)
    occupied = np.zeros((height, width), dtype=bool)

    max_objects = min(4, 8 // 2)
    object_count = rng.randint(1, max_objects)
    colors = list(COLOR_POOL)
    rng.shuffle(colors)
    needed = object_count * 2
    if needed > len(colors):
        return None
    frame_colors = colors[:object_count]
    pattern_colors = colors[object_count:needed]

    dirs = list(DIRS)
    rng.shuffle(dirs)
    directions = [dirs[i % 4] for i in range(object_count)]
    while len(directions) < object_count:
        directions.append(rng.choice(DIRS))
    rng.shuffle(directions)

    for idx in range(object_count):
        frame_color = frame_colors[idx]
        pattern_color = pattern_colors[idx]
        direction = directions[idx]
        pattern_mask = _random_pattern(rng, direction)
        ph, pw = pattern_mask.shape

        placed = False
        for _ in range(160):
            frame_h = rng.randint(max(6, ph + 4), min(11, ph + 7))
            frame_w = rng.randint(max(6, pw + 4), min(11, pw + 7))
            max_pr = frame_h - ph - 2
            max_pc = frame_w - pw - 2
            if max_pr < 2 or max_pc < 2:
                continue
            pattern_r = rng.randint(2, max_pr)
            pattern_c = rng.randint(2, max_pc)

            if direction == "up":
                r_lo, r_hi = ph + 1, height - frame_h - 2
                c_lo, c_hi = 1, width - frame_w - 2
            elif direction == "down":
                r_lo, r_hi = 1, height - frame_h - ph - 2
                c_lo, c_hi = 1, width - frame_w - 2
            elif direction == "left":
                r_lo, r_hi = 1, height - frame_h - 2
                c_lo, c_hi = pw + 1, width - frame_w - 2
            else:
                r_lo, r_hi = 1, height - frame_h - 2
                c_lo, c_hi = 1, width - frame_w - pw - 2

            if r_lo > r_hi or c_lo > c_hi:
                continue

            r0 = rng.randint(r_lo, r_hi)
            c0 = rng.randint(c_lo, c_hi)
            frame_bbox = (r0, c0, r0 + frame_h - 1, c0 + frame_w - 1)
            pattern_bbox = (
                r0 + pattern_r,
                c0 + pattern_c,
                r0 + pattern_r + ph - 1,
                c0 + pattern_c + pw - 1,
            )
            dest_bbox = _destination_bbox(frame_bbox, pattern_bbox, direction)
            union_bbox = _bbox_union(frame_bbox, dest_bbox)

            if not _free(occupied, union_bbox, pad=1):
                continue

            _draw_object(
                grid,
                frame_color=frame_color,
                pattern_color=pattern_color,
                direction=direction,
                frame_bbox=frame_bbox,
                pattern_offset=(pattern_r, pattern_c),
                pattern_mask=pattern_mask,
            )
            _mark_forbidden(occupied, union_bbox, pad=1)
            placed = True
            break

        if not placed:
            return None

    frames = _find_u_frames(grid, 0)
    if len(frames) != object_count:
        return None
    out = main(grid)
    if np.array_equal(out, grid):
        return None
    if out.shape != grid.shape or max(out.shape) > 30:
        return None
    return grid


def generate_input() -> np.ndarray:
    for _ in range(GEN_RETRIES):
        rng = random.Random()
        grid = _gen_attempt(rng)
        if grid is not None:
            return grid
    raise RuntimeError("failed to generate a valid U-frame puzzle")
