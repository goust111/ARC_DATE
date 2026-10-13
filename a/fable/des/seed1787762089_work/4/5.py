from collections import deque

import numpy as np


DIR4 = [(-1, 0), (1, 0), (0, -1), (0, 1)]
ROTATE_DIR_CCW = {"up": "left", "left": "down", "down": "right", "right": "up"}
FRAME_SIZE = 7
FRAME_GAP = {"up": (0, 3), "left": (3, 0), "right": (3, 6)}
FRAME_ANCHOR = {"up": (1, 3), "left": (3, 1), "right": (3, 5)}
GEN_RETRIES = 400
BG_COLOR = 0
COLOR_POOL = np.array([1, 2, 3, 4, 5, 6, 7, 8, 9], dtype=int)
_RNG = np.random.default_rng()


def _mask(rows: list[str]) -> np.ndarray:
    return np.array([[ch == "1" for ch in row] for row in rows], dtype=bool)


SPRITE_LIBRARY = []
for rows in [
    ["10", "11", "11"],
    ["010", "111", "110"],
    ["010", "111", "111"],
    ["010", "111", "110", "110"],
    ["010", "111", "101", "111"],
    ["010", "111", "111", "111"],
]:
    mask = _mask(rows)
    tip_col = int(np.flatnonzero(mask[0])[0])
    SPRITE_LIBRARY.append(
        {
            "mask": mask,
            "tip": (0, tip_col),
            "dir": "up",
            "area": int(mask.sum()),
        }
    )


def _get_bg(grid: np.ndarray) -> int:
    return int(np.argmax(np.bincount(np.asarray(grid, dtype=int).ravel())))


def _find_components(grid: np.ndarray, bg: int) -> list[dict]:
    h, w = map(int, grid.shape)
    seen = np.zeros((h, w), dtype=bool)
    comps = []
    for r in range(h):
        for c in range(w):
            color = int(grid[r, c])
            if color == bg or seen[r, c]:
                continue
            seen[r, c] = True
            q = deque([(r, c)])
            coords = []
            while q:
                rr, cc = q.popleft()
                coords.append((rr, cc))
                for dr, dc in DIR4:
                    nr, nc = rr + dr, cc + dc
                    if 0 <= nr < h and 0 <= nc < w and not seen[nr, nc] and int(grid[nr, nc]) == color:
                        seen[nr, nc] = True
                        q.append((nr, nc))
            pts = np.array(coords, dtype=int)
            r0, c0 = pts.min(axis=0)
            r1, c1 = pts.max(axis=0)
            local = np.zeros((int(r1 - r0 + 1), int(c1 - c0 + 1)), dtype=bool)
            for rr, cc in coords:
                local[rr - r0, cc - c0] = True
            comps.append(
                {
                    "color": color,
                    "coords": coords,
                    "bbox": (int(r0), int(c0), int(r1), int(c1)),
                    "mask": local,
                    "area": int(len(coords)),
                }
            )
    return comps


def _frame_gap_direction(mask: np.ndarray) -> str | None:
    if mask.shape != (FRAME_SIZE, FRAME_SIZE):
        return None
    if np.any(mask[1:-1, 1:-1]):
        return None
    boundary = np.zeros_like(mask, dtype=bool)
    boundary[0, :] = True
    boundary[-1, :] = True
    boundary[:, 0] = True
    boundary[:, -1] = True
    for direction, (gr, gc) in FRAME_GAP.items():
        expected = boundary.copy()
        expected[gr, gc] = False
        if np.array_equal(mask, expected):
            return direction
    return None


def _is_connected(mask: np.ndarray) -> bool:
    pts = np.argwhere(mask)
    if len(pts) == 0:
        return False
    h, w = map(int, mask.shape)
    start = tuple(map(int, pts[0]))
    seen = {start}
    q = deque([start])
    while q:
        r, c = q.popleft()
        for dr, dc in DIR4:
            nr, nc = r + dr, c + dc
            if 0 <= nr < h and 0 <= nc < w and mask[nr, nc] and (nr, nc) not in seen:
                seen.add((nr, nc))
                q.append((nr, nc))
    return len(seen) == int(mask.sum())


def _extract_tentacle(mask: np.ndarray) -> tuple[tuple[int, int], str]:
    candidates = []
    h, w = map(int, mask.shape)
    for r, c in np.argwhere(mask):
        r = int(r)
        c = int(c)
        degree = 0
        for dr, dc in DIR4:
            nr, nc = r + dr, c + dc
            if 0 <= nr < h and 0 <= nc < w and mask[nr, nc]:
                degree += 1
        if degree != 1:
            continue
        remainder = mask.copy()
        remainder[r, c] = False
        if not _is_connected(remainder):
            continue
        rest = np.argwhere(remainder)
        r0, c0 = rest.min(axis=0)
        r1, c1 = rest.max(axis=0)
        if int(r1 - r0 + 1) > 3 or int(c1 - c0 + 1) > 3:
            continue
        directions = []
        if r < r0:
            directions.append("up")
        if r > r1:
            directions.append("down")
        if c < c0:
            directions.append("left")
        if c > c1:
            directions.append("right")
        if len(directions) == 1:
            candidates.append(((r, c), directions[0]))
    assert len(candidates) == 1, "expected one unambiguous tentacle tip"
    return candidates[0]


def _rotate_mask_point_ccw(mask: np.ndarray, point: tuple[int, int]) -> tuple[np.ndarray, tuple[int, int]]:
    h, w = map(int, mask.shape)
    r, c = point
    return np.rot90(mask, 1), (w - 1 - c, r)


def _rotate_to_direction(
    mask: np.ndarray,
    point: tuple[int, int],
    direction: str,
    target: str,
) -> tuple[np.ndarray, tuple[int, int]]:
    rotated_mask = mask.copy()
    rotated_point = point
    rotated_dir = direction
    for _ in range(4):
        if rotated_dir == target:
            return rotated_mask, rotated_point
        rotated_mask, rotated_point = _rotate_mask_point_ccw(rotated_mask, rotated_point)
        rotated_dir = ROTATE_DIR_CCW[rotated_dir]
    raise AssertionError("failed to rotate sprite into target direction")


def _stamp_frame(out: np.ndarray, bbox: tuple[int, int, int, int], color: int, gap_dir: str, bg: int) -> None:
    r0, c0, r1, c1 = bbox
    out[r0, c0 : c1 + 1] = color
    out[r1, c0 : c1 + 1] = color
    out[r0 : r1 + 1, c0] = color
    out[r0 : r1 + 1, c1] = color
    gr, gc = FRAME_GAP[gap_dir]
    out[r0 + gr, c0 + gc] = bg


def _compose_output(shape: tuple[int, int], bg: int, frames: list[dict], sprites: list[dict]) -> np.ndarray:
    assert len(frames) == len(sprites)
    ordered_frames = sorted(frames, key=lambda frame: (frame["bbox"][1], frame["bbox"][0]))
    ordered_sprites = sorted(sprites, key=lambda sprite: sprite["area"])
    assert len({sprite["area"] for sprite in ordered_sprites}) == len(ordered_sprites)

    out = np.full(shape, bg, dtype=int)
    for frame in ordered_frames:
        _stamp_frame(out, frame["bbox"], int(frame["color"]), frame["gap_dir"], bg)

    for frame, sprite in zip(ordered_frames, ordered_sprites):
        mask, tip = _rotate_to_direction(sprite["mask"], sprite["tip"], sprite["dir"], frame["gap_dir"])
        anchor_r, anchor_c = FRAME_ANCHOR[frame["gap_dir"]]
        r0, c0, r1, c1 = frame["bbox"]
        top = r0 + anchor_r - tip[0]
        left = c0 + anchor_c - tip[1]
        for rr, cc in np.argwhere(mask):
            rr = int(rr)
            cc = int(cc)
            gr = top + rr
            gc = left + cc
            assert r0 + 1 <= gr <= r1 - 1 and c0 + 1 <= gc <= c1 - 1, "sprite must stay inside frame"
            out[gr, gc] = int(frame["color"])
    return out


def main(grid: np.ndarray) -> np.ndarray:
    g = np.asarray(grid, dtype=int)
    assert g.ndim == 2
    bg = _get_bg(g)

    frames = []
    sprites = []
    for comp in _find_components(g, bg):
        gap_dir = _frame_gap_direction(comp["mask"])
        if gap_dir is not None:
            frames.append({"bbox": comp["bbox"], "color": comp["color"], "gap_dir": gap_dir})
            continue
        tip, direction = _extract_tentacle(comp["mask"])
        sprites.append({"mask": comp["mask"], "tip": tip, "dir": direction, "area": comp["area"]})

    assert 3 <= len(frames) <= 4
    assert len(frames) == len(sprites)
    return _compose_output(g.shape, bg, frames, sprites)


def _random_derangement(rng: np.random.Generator, items: list[int]) -> list[int]:
    base = np.arange(len(items))
    while True:
        perm = rng.permutation(len(items))
        if np.all(perm != base):
            return [items[int(i)] for i in perm]


def _random_orientation(
    rng: np.random.Generator, spec: dict
) -> tuple[np.ndarray, tuple[int, int], str]:
    mask = spec["mask"].copy()
    tip = spec["tip"]
    direction = spec["dir"]
    turns = int(rng.integers(4))
    for _ in range(turns):
        mask, tip = _rotate_mask_point_ccw(mask, tip)
        direction = ROTATE_DIR_CCW[direction]
    return mask, tip, direction


def _rects_intersect(a: tuple[int, int, int, int], b: tuple[int, int, int, int]) -> bool:
    return not (a[2] < b[0] or b[2] < a[0] or a[3] < b[1] or b[3] < a[1])


def _place_sprites(
    rng: np.random.Generator,
    grid_shape: tuple[int, int],
    frame_top: int,
    sprites: list[dict],
) -> list[dict] | None:
    h, w = grid_shape
    max_bottom = frame_top - 2
    placed = [None] * len(sprites)
    padded_rects = []
    order = list(rng.permutation(len(sprites)))
    for idx in order:
        sprite = sprites[idx]
        sh, sw = map(int, sprite["mask"].shape)
        row_lo = 1
        row_hi = max_bottom - sh + 1
        col_lo = 1
        col_hi = w - sw - 1
        if row_lo > row_hi or col_lo > col_hi:
            return None
        success = False
        for _ in range(200):
            top = int(rng.integers(row_lo, row_hi + 1))
            left = int(rng.integers(col_lo, col_hi + 1))
            padded = (top - 1, left - 1, top + sh, left + sw)
            if any(_rects_intersect(padded, other) for other in padded_rects):
                continue
            placed[idx] = {**sprite, "top": top, "left": left}
            padded_rects.append(padded)
            success = True
            break
        if not success:
            return None
    return placed


def _build_input(
    shape: tuple[int, int],
    bg: int,
    frames: list[dict],
    placed_sprites: list[dict],
) -> np.ndarray:
    grid = np.full(shape, bg, dtype=int)
    for frame in frames:
        _stamp_frame(grid, frame["bbox"], int(frame["color"]), frame["gap_dir"], bg)
    for sprite in placed_sprites:
        top = int(sprite["top"])
        left = int(sprite["left"])
        for rr, cc in np.argwhere(sprite["mask"]):
            grid[top + int(rr), left + int(cc)] = int(sprite["color"])
    return grid


def _gen_attempt() -> np.ndarray | None:
    rng = _RNG
    count = int(rng.choice([3, 4]))
    frame_gap = int(rng.integers(1, 4))
    frame_band_w = count * FRAME_SIZE + (count - 1) * frame_gap
    width = int(rng.integers(frame_band_w + 3, frame_band_w + 10))
    height = int(rng.integers(20, 29))
    frame_top = height - FRAME_SIZE - 1
    if frame_top < 9:
        return None

    left_margin = int(rng.integers(1, width - frame_band_w))
    frame_colors = [int(v) for v in rng.choice(COLOR_POOL, size=count, replace=False)]
    frame_dirs = [str(v) for v in rng.choice(["up", "left", "right"], size=count, replace=True)]
    frames = []
    for i in range(count):
        c0 = left_margin + i * (FRAME_SIZE + frame_gap)
        r0 = frame_top
        frames.append(
            {
                "bbox": (r0, c0, r0 + FRAME_SIZE - 1, c0 + FRAME_SIZE - 1),
                "color": frame_colors[i],
                "gap_dir": frame_dirs[i],
            }
        )

    chosen = [SPRITE_LIBRARY[int(i)] for i in rng.choice(len(SPRITE_LIBRARY), size=count, replace=False)]
    chosen.sort(key=lambda spec: spec["area"])
    sprite_colors = _random_derangement(rng, frame_colors)
    sprites = []
    for spec, color in zip(chosen, sprite_colors):
        mask, tip, direction = _random_orientation(rng, spec)
        sprites.append(
            {
                "mask": mask,
                "tip": tip,
                "dir": direction,
                "area": int(spec["area"]),
                "color": int(color),
            }
        )

    placed = _place_sprites(rng, (height, width), frame_top, sprites)
    if placed is None:
        return None

    inp = _build_input((height, width), BG_COLOR, frames, placed)
    expected = _compose_output(inp.shape, BG_COLOR, frames, placed)
    try:
        actual = main(inp)
    except AssertionError:
        return None
    if not np.array_equal(actual, expected):
        return None
    return inp


def generate_input() -> np.ndarray:
    for _ in range(GEN_RETRIES):
        grid = _gen_attempt()
        if grid is not None:
            return grid
    raise RuntimeError("failed to generate a valid sample")
