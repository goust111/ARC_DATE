import random
from collections import deque

import numpy as np


BG = 0
SHADOW = 5
PALETTE = (1, 2, 3, 4, 6, 7, 8, 9)
DIRS = {
    "up": (-1, 0),
    "down": (1, 0),
    "left": (0, -1),
    "right": (0, 1),
}
GEN_ATTEMPTS = 400
ITEM_ATTEMPTS = 120


def _in_bounds(h: int, w: int, r: int, c: int) -> bool:
    return 0 <= r < h and 0 <= c < w


def _monochrome_components(grid: np.ndarray) -> list[dict[str, object]]:
    h, w = grid.shape
    seen = np.zeros((h, w), dtype=bool)
    comps: list[dict[str, object]] = []
    for r in range(h):
        for c in range(w):
            color = int(grid[r, c])
            if color in (BG, SHADOW) or seen[r, c]:
                continue
            q = deque([(r, c)])
            seen[r, c] = True
            coords: list[tuple[int, int]] = []
            while q:
                rr, cc = q.popleft()
                coords.append((rr, cc))
                for dr, dc in DIRS.values():
                    nr, nc = rr + dr, cc + dc
                    if _in_bounds(h, w, nr, nc) and not seen[nr, nc] and int(grid[nr, nc]) == color:
                        seen[nr, nc] = True
                        q.append((nr, nc))
            comps.append({"color": color, "coords": coords})
    return comps


def _shadow_component(shadow_mask: np.ndarray, seed: tuple[int, int]) -> set[tuple[int, int]]:
    h, w = shadow_mask.shape
    q = deque([seed])
    seen = {seed}
    while q:
        r, c = q.popleft()
        for dr, dc in DIRS.values():
            nr, nc = r + dr, c + dc
            if _in_bounds(h, w, nr, nc) and shadow_mask[nr, nc] and (nr, nc) not in seen:
                seen.add((nr, nc))
                q.append((nr, nc))
    return seen


def _infer_shadow(grid: np.ndarray, coords: list[tuple[int, int]]) -> tuple[str, int]:
    h, w = grid.shape
    shadow_mask = grid == SHADOW
    candidates: list[tuple[str, int]] = []

    for name, (dr, dc) in DIRS.items():
        surface: list[tuple[int, int]] = []
        lengths: list[int] = []

        for r, c in coords:
            nr, nc = r + dr, c + dc
            if _in_bounds(h, w, nr, nc) and shadow_mask[nr, nc]:
                surface.append((r, c))
                run = 0
                rr, cc = nr, nc
                while _in_bounds(h, w, rr, cc) and shadow_mask[rr, cc]:
                    run += 1
                    rr += dr
                    cc += dc
                lengths.append(run)

        if not surface or len(set(lengths)) != 1:
            continue

        shadow_len = lengths[0]
        expected: set[tuple[int, int]] = set()
        valid = True
        for r, c in surface:
            for step in range(1, shadow_len + 1):
                rr = r + dr * step
                cc = c + dc * step
                if not _in_bounds(h, w, rr, cc) or not shadow_mask[rr, cc]:
                    valid = False
                    break
                expected.add((rr, cc))
            if not valid:
                break
            rr = r + dr * (shadow_len + 1)
            cc = c + dc * (shadow_len + 1)
            if _in_bounds(h, w, rr, cc) and shadow_mask[rr, cc]:
                valid = False
                break
        if not valid:
            continue

        actual = _shadow_component(shadow_mask, next(iter(expected)))
        if actual != expected:
            continue

        candidates.append((name, shadow_len))

    assert len(candidates) == 1, "expected one unique shadow direction per obstacle"
    return candidates[0]


def main(grid: np.ndarray) -> np.ndarray:
    """
    Reverse grey shadows into colored light rays.

    Each colored obstacle has a grey horizontal or vertical shadow on one side.
    The shadow length is reused as the distance from the obstacle to the source
    side, so the output removes grey and extends the obstacle color by the same
    number of cells in the opposite direction.
    """
    g = np.asarray(grid, dtype=int)
    assert g.ndim == 2, "expected a 2D grid"
    h, w = g.shape

    out = g.copy()
    out[out == SHADOW] = BG

    for comp in _monochrome_components(g):
        color = int(comp["color"])
        coords = list(comp["coords"])
        shadow_dir, shadow_len = _infer_shadow(g, coords)
        dr, dc = DIRS[shadow_dir]

        for r, c in coords:
            for step in range(1, shadow_len + 1):
                rr = r - dr * step
                cc = c - dc * step
                assert _in_bounds(h, w, rr, cc), "inferred source ray left the grid"
                out[rr, cc] = color

    return out


def _layout_cells(
    shadow_dir: str, length: int, shadow_len: int, anchor_r: int, anchor_c: int
) -> dict[str, object]:
    if shadow_dir == "right":
        obj = {(anchor_r + i, anchor_c) for i in range(length)}
        shadow = {(anchor_r + i, anchor_c + step) for i in range(length) for step in range(1, shadow_len + 1)}
        rays = {(anchor_r + i, anchor_c - step) for i in range(length) for step in range(1, shadow_len + 1)}
    elif shadow_dir == "left":
        obj = {(anchor_r + i, anchor_c) for i in range(length)}
        shadow = {(anchor_r + i, anchor_c - step) for i in range(length) for step in range(1, shadow_len + 1)}
        rays = {(anchor_r + i, anchor_c + step) for i in range(length) for step in range(1, shadow_len + 1)}
    elif shadow_dir == "down":
        obj = {(anchor_r, anchor_c + i) for i in range(length)}
        shadow = {(anchor_r + step, anchor_c + i) for i in range(length) for step in range(1, shadow_len + 1)}
        rays = {(anchor_r - step, anchor_c + i) for i in range(length) for step in range(1, shadow_len + 1)}
    else:
        obj = {(anchor_r, anchor_c + i) for i in range(length)}
        shadow = {(anchor_r - step, anchor_c + i) for i in range(length) for step in range(1, shadow_len + 1)}
        rays = {(anchor_r + step, anchor_c + i) for i in range(length) for step in range(1, shadow_len + 1)}

    full = obj | shadow | rays
    rows = [r for r, _ in full]
    cols = [c for _, c in full]
    bbox = (min(rows), min(cols), max(rows), max(cols))
    return {"obj": obj, "shadow": shadow, "rays": rays, "bbox": bbox}


def _mark_blocked(blocked: np.ndarray, bbox: tuple[int, int, int, int], margin: int = 1) -> None:
    h, w = blocked.shape
    r0, c0, r1, c1 = bbox
    r0 = max(0, r0 - margin)
    c0 = max(0, c0 - margin)
    r1 = min(h - 1, r1 + margin)
    c1 = min(w - 1, c1 + margin)
    blocked[r0 : r1 + 1, c0 : c1 + 1] = True


def _place_one(
    h: int,
    w: int,
    blocked: np.ndarray,
    shadow_dir: str,
    length: int,
    shadow_len: int,
) -> dict[str, object] | None:
    if shadow_dir in ("left", "right"):
        min_r, max_r = 0, h - length
        min_c, max_c = shadow_len, w - shadow_len - 1
    else:
        min_r, max_r = shadow_len, h - shadow_len - 1
        min_c, max_c = 0, w - length

    if min_r > max_r or min_c > max_c:
        return None

    for _ in range(ITEM_ATTEMPTS):
        anchor_r = random.randint(min_r, max_r)
        anchor_c = random.randint(min_c, max_c)
        item = _layout_cells(shadow_dir, length, shadow_len, anchor_r, anchor_c)
        full = item["obj"] | item["shadow"] | item["rays"]
        if any(blocked[r, c] for r, c in full):
            continue
        return item
    return None


def _gen_attempt() -> np.ndarray | None:
    h = random.randint(11, 18)
    w = random.randint(11, 18)
    area = h * w
    max_objects = 3 if area < 180 else 4 if area < 250 else 5
    count = random.randint(2, max_objects)
    colors = random.sample(PALETTE, count)

    grid = np.full((h, w), BG, dtype=int)
    blocked = np.zeros((h, w), dtype=bool)
    directions = [random.choice(tuple(DIRS)) for _ in range(count)]

    # Reshuffle once in a while so small canvases are less likely to fail early.
    random.shuffle(directions)

    for idx in range(count):
        shadow_dir = directions[idx]
        length = random.randint(1, 3)
        max_shadow = min(5, (w - 1) // 2) if shadow_dir in ("left", "right") else min(5, (h - 1) // 2)
        if max_shadow < 1:
            return None
        shadow_len = random.randint(1, max_shadow)

        item = _place_one(h, w, blocked, shadow_dir, length, shadow_len)
        if item is None:
            return None

        color = colors[idx]
        for r, c in item["obj"]:
            grid[r, c] = color
        for r, c in item["shadow"]:
            grid[r, c] = SHADOW
        _mark_blocked(blocked, item["bbox"], margin=1)

    # Generator-side validation keeps the random scene aligned with main().
    _ = main(grid)
    return grid


def generate_input() -> np.ndarray:
    for _ in range(GEN_ATTEMPTS):
        grid = _gen_attempt()
        if grid is not None:
            return grid
    raise RuntimeError("failed to generate a valid shadow-ray scene")
