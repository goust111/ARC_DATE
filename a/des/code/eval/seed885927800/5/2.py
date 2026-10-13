from __future__ import annotations

import random
from collections import Counter

import numpy as np


BG = 8
RED = 2
PALETTE = [0, 1, 3, 4, 5, 6, 7, 9]
DIRECTIONS = ("up", "down", "left", "right")
GEN_ATTEMPTS = 400


def _get_bg(grid: np.ndarray) -> int:
    return int(Counter(map(int, grid.ravel())).most_common(1)[0][0])


def _neighbors(
    r: int, c: int, h: int, w: int, connectivity: int
) -> list[tuple[int, int]]:
    out: list[tuple[int, int]] = []
    offsets = [(1, 0), (-1, 0), (0, 1), (0, -1)]
    if connectivity == 8:
        offsets += [(1, 1), (1, -1), (-1, 1), (-1, -1)]
    for dr, dc in offsets:
        nr = r + dr
        nc = c + dc
        if 0 <= nr < h and 0 <= nc < w:
            out.append((nr, nc))
    return out


def _components(
    mask: np.ndarray, connectivity: int = 4
) -> list[list[tuple[int, int]]]:
    h, w = mask.shape
    seen = np.zeros((h, w), dtype=bool)
    comps: list[list[tuple[int, int]]] = []
    for r in range(h):
        for c in range(w):
            if not bool(mask[r, c]) or bool(seen[r, c]):
                continue
            stack = [(r, c)]
            seen[r, c] = True
            coords: list[tuple[int, int]] = []
            while stack:
                cr, cc = stack.pop()
                coords.append((cr, cc))
                for nr, nc in _neighbors(cr, cc, h, w, connectivity):
                    if bool(mask[nr, nc]) and not bool(seen[nr, nc]):
                        seen[nr, nc] = True
                        stack.append((nr, nc))
            comps.append(coords)
    return comps


def _bbox(coords: list[tuple[int, int]]) -> tuple[int, int, int, int]:
    rows = [r for r, _ in coords]
    cols = [c for _, c in coords]
    return min(rows), min(cols), max(rows), max(cols)


def _component_patch(
    coords: list[tuple[int, int]], bbox: tuple[int, int, int, int]
) -> np.ndarray:
    r0, c0, r1, c1 = bbox
    patch = np.zeros((r1 - r0 + 1, c1 - c0 + 1), dtype=bool)
    for r, c in coords:
        patch[r - r0, c - c0] = True
    return patch


def _is_connected(mask: np.ndarray, connectivity: int) -> bool:
    if not bool(mask.any()):
        return False
    return len(_components(mask, connectivity)) == 1


def _rotation_matches(key_mask: np.ndarray, lock_mask: np.ndarray) -> list[np.ndarray]:
    matches: list[np.ndarray] = []
    for k in range(4):
        rotated = np.rot90(key_mask, k)
        if rotated.shape == lock_mask.shape and np.array_equal(~rotated, lock_mask):
            matches.append(rotated)
    return matches


def _is_frame_component(coords: list[tuple[int, int]]) -> bool:
    r0, c0, r1, c1 = _bbox(coords)
    h = r1 - r0 + 1
    w = c1 - c0 + 1
    if h < 4 or w < 4:
        return False
    if len(coords) != 2 * h + 2 * w - 4:
        return False
    coord_set = set(coords)
    for r in range(r0, r1 + 1):
        for c in range(c0, c1 + 1):
            on_border = r in (r0, r1) or c in (c0, c1)
            if on_border != ((r, c) in coord_set):
                return False
    return True


def _extract_key(
    grid: np.ndarray, bbox: tuple[int, int, int, int], bg: int
) -> tuple[int, np.ndarray]:
    r0, c0, r1, c1 = bbox
    inner = grid[r0 + 1 : r1, c0 + 1 : c1]
    colors = sorted({int(v) for v in np.unique(inner) if int(v) != bg})
    assert len(colors) == 1, "frame interior must contain exactly one key color"
    key_color = int(colors[0])
    key_mask = inner == key_color
    comps = _components(key_mask, connectivity=8)
    assert len(comps) == 1, "key must be one connected component"
    kbbox = _bbox(comps[0])
    cropped = _component_patch(comps[0], kbbox)
    return key_color, cropped


def _find_frames(
    grid: np.ndarray, bg: int
) -> tuple[list[dict[str, object]], np.ndarray]:
    frames: list[dict[str, object]] = []
    cover = np.zeros_like(grid, dtype=bool)
    colors = sorted({int(v) for v in np.unique(grid) if int(v) not in (bg, RED)})
    for color in colors:
        for coords in _components(grid == color, connectivity=4):
            if not _is_frame_component(coords):
                continue
            bbox = _bbox(coords)
            try:
                key_color, key_mask = _extract_key(grid, bbox, bg)
            except AssertionError:
                continue
            frames.append(
                {
                    "border_color": color,
                    "bbox": bbox,
                    "key_color": key_color,
                    "key_mask": key_mask,
                }
            )
            r0, c0, r1, c1 = bbox
            cover[r0 : r1 + 1, c0 : c1 + 1] = True
    assert frames, "expected at least one frame"
    return frames, cover


def _find_marker_side(
    grid: np.ndarray, bbox: tuple[int, int, int, int]
) -> tuple[str, tuple[int, int]]:
    h, w = grid.shape
    r0, c0, r1, c1 = bbox
    hits: list[tuple[str, tuple[int, int]]] = []
    if r0 > 0:
        for c in range(c0, c1 + 1):
            if int(grid[r0 - 1, c]) == RED:
                hits.append(("up", (r0 - 1, c)))
    if r1 + 1 < h:
        for c in range(c0, c1 + 1):
            if int(grid[r1 + 1, c]) == RED:
                hits.append(("down", (r1 + 1, c)))
    if c0 > 0:
        for r in range(r0, r1 + 1):
            if int(grid[r, c0 - 1]) == RED:
                hits.append(("left", (r, c0 - 1)))
    if c1 + 1 < w:
        for r in range(r0, r1 + 1):
            if int(grid[r, c1 + 1]) == RED:
                hits.append(("right", (r, c1 + 1)))
    assert len(hits) == 1, "each lockhole must have exactly one red side marker"
    return hits[0]


def main(grid: np.ndarray) -> np.ndarray:
    """
    Pair each external lockhole with the frame whose border has the same color.
    Rotate the frame's key until it is the rectangle-complement of the lockhole,
    then emit key-colored rays away from the red marker side using row/column
    key counts as ray lengths.
    """
    g = np.asarray(grid, dtype=int)
    assert g.ndim == 2
    bg = _get_bg(g)

    frames, cover = _find_frames(g, bg)
    frame_by_color: dict[int, dict[str, object]] = {}
    for frame in frames:
        border_color = int(frame["border_color"])
        assert border_color not in frame_by_color, "frame border colors must be unique"
        frame_by_color[border_color] = frame

    out = g.copy()
    out[out == RED] = bg

    lockholes: list[dict[str, object]] = []
    colors = sorted({int(v) for v in np.unique(g) if int(v) not in (bg, RED)})
    for color in colors:
        comp_mask = (g == color) & (~cover)
        for coords in _components(comp_mask, connectivity=8):
            bbox = _bbox(coords)
            side, _marker = _find_marker_side(g, bbox)
            lockholes.append(
                {
                    "color": color,
                    "bbox": bbox,
                    "mask": _component_patch(coords, bbox),
                    "side": side,
                }
            )

    assert lockholes, "expected at least one lockhole"

    for lockhole in lockholes:
        color = int(lockhole["color"])
        assert color in frame_by_color, "lockhole color must match a frame border"
        frame = frame_by_color[color]
        key_color = int(frame["key_color"])
        key_mask = np.asarray(frame["key_mask"], dtype=bool)
        lock_mask = np.asarray(lockhole["mask"], dtype=bool)
        matches = _rotation_matches(key_mask, lock_mask)
        assert matches, "expected a complementary key rotation"
        aligned_key = matches[0]

        r0, c0, r1, c1 = lockhole["bbox"]
        side = str(lockhole["side"])
        if side in ("left", "right"):
            lengths = aligned_key.sum(axis=1)
            for offset, raw_len in enumerate(lengths):
                length = int(raw_len)
                row = r0 + offset
                for step in range(1, length + 1):
                    col = c0 - step if side == "left" else c1 + step
                    if not (0 <= col < out.shape[1]):
                        break
                    if int(out[row, col]) != bg:
                        break
                    out[row, col] = key_color
        else:
            lengths = aligned_key.sum(axis=0)
            for offset, raw_len in enumerate(lengths):
                length = int(raw_len)
                col = c0 + offset
                for step in range(1, length + 1):
                    row = r0 - step if side == "up" else r1 + step
                    if not (0 <= row < out.shape[0]):
                        break
                    if int(out[row, col]) != bg:
                        break
                    out[row, col] = key_color

    return out


def _random_key_mask(rng: random.Random) -> np.ndarray:
    for _ in range(2000):
        h = rng.randint(3, 5)
        w = rng.randint(3, 5)
        total = h * w
        p = rng.uniform(0.35, 0.65)
        mask = np.array(
            [[rng.random() < p for _ in range(w)] for _ in range(h)], dtype=bool
        )
        area = int(mask.sum())
        if area < 3 or area > total - 3:
            continue

        if not _is_connected(mask, connectivity=8) or not _is_connected(
            ~mask, connectivity=8
        ):
            continue
        if np.any(mask.sum(axis=1) == 0) or np.any(mask.sum(axis=0) == 0):
            continue
        if np.any((~mask).sum(axis=1) == 0) or np.any((~mask).sum(axis=0) == 0):
            continue
        return mask
    raise RuntimeError("failed to generate a valid key mask")


def _paint_frame(
    grid: np.ndarray,
    blocked: np.ndarray,
    key_mask: np.ndarray,
    border_color: int,
    key_color: int,
    rng: random.Random,
) -> bool:
    h, w = key_mask.shape
    outer_h = h + 2
    outer_w = w + 2
    gh, gw = grid.shape
    for _ in range(300):
        r0 = rng.randint(1, gh - outer_h - 1)
        c0 = rng.randint(1, gw - outer_w - 1)
        if blocked[r0 - 1 : r0 + outer_h + 1, c0 - 1 : c0 + outer_w + 1].any():
            continue
        r1 = r0 + outer_h - 1
        c1 = c0 + outer_w - 1
        grid[r0, c0 : c1 + 1] = border_color
        grid[r1, c0 : c1 + 1] = border_color
        grid[r0 : r1 + 1, c0] = border_color
        grid[r0 : r1 + 1, c1] = border_color
        inner = grid[r0 + 1 : r1, c0 + 1 : c1]
        inner[key_mask] = key_color
        blocked[r0 - 1 : r0 + outer_h + 1, c0 - 1 : c0 + outer_w + 1] = True
        return True
    return False


def _lockhole_reserve_bbox(
    r0: int, c0: int, h: int, w: int, side: str, max_len: int
) -> tuple[int, int, int, int]:
    r1 = r0 + h - 1
    c1 = c0 + w - 1
    if side == "right":
        return r0 - 1, c0 - 1, r1 + 1, c1 + max_len + 1
    if side == "left":
        return r0 - 1, c0 - max_len - 1, r1 + 1, c1 + 1
    if side == "down":
        return r0 - 1, c0 - 1, r1 + max_len + 1, c1 + 1
    return r0 - max_len - 1, c0 - 1, r1 + 1, c1 + 1


def _paint_lockhole(
    grid: np.ndarray,
    blocked: np.ndarray,
    lock_mask: np.ndarray,
    color: int,
    side: str,
    max_len: int,
    rng: random.Random,
) -> bool:
    h, w = lock_mask.shape
    gh, gw = grid.shape
    for _ in range(400):
        r0 = rng.randint(1, gh - h - 1)
        c0 = rng.randint(1, gw - w - 1)
        rr0, cc0, rr1, cc1 = _lockhole_reserve_bbox(r0, c0, h, w, side, max_len)
        if rr0 < 0 or cc0 < 0 or rr1 >= gh or cc1 >= gw:
            continue
        if blocked[rr0 : rr1 + 1, cc0 : cc1 + 1].any():
            continue
        patch = grid[r0 : r0 + h, c0 : c0 + w]
        patch[lock_mask] = color
        r1 = r0 + h - 1
        c1 = c0 + w - 1
        if side == "right":
            marker = (r0 + rng.randrange(h), c1 + 1)
        elif side == "left":
            marker = (r0 + rng.randrange(h), c0 - 1)
        elif side == "down":
            marker = (r1 + 1, c0 + rng.randrange(w))
        else:
            marker = (r0 - 1, c0 + rng.randrange(w))
        grid[marker] = RED
        blocked[rr0 : rr1 + 1, cc0 : cc1 + 1] = True
        return True
    return False


def _gen_attempt(rng: random.Random) -> np.ndarray | None:
    n_pairs = rng.randint(2, 4)
    min_side = 14 + 3 * n_pairs
    height = rng.randint(min_side, min_side + 8)
    width = rng.randint(min_side, min_side + 10)
    grid = np.full((height, width), BG, dtype=int)
    blocked = np.zeros((height, width), dtype=bool)

    colors = rng.sample(PALETTE, 2 * n_pairs)
    border_colors = colors[:n_pairs]
    key_colors = colors[n_pairs:]
    directions = rng.sample(list(DIRECTIONS), n_pairs)
    rotations = rng.sample([0, 1, 2, 3], n_pairs)

    specs: list[dict[str, object]] = []
    for border_color, key_color, side, rot_k in zip(
        border_colors, key_colors, directions, rotations
    ):
        for _ in range(200):
            key_mask = _random_key_mask(rng)
            rotated = np.rot90(key_mask, rot_k)
            lock_mask = ~rotated
            if len(_rotation_matches(key_mask, lock_mask)) != 1:
                continue
            max_len = int(
                rotated.sum(axis=1 if side in ("left", "right") else 0).max()
            )
            if max_len <= 0:
                continue
            specs.append(
                {
                    "border_color": border_color,
                    "key_color": key_color,
                    "key_mask": key_mask,
                    "lock_mask": lock_mask,
                    "side": side,
                    "max_len": max_len,
                }
            )
            break
        else:
            return None

    rng.shuffle(specs)
    for spec in specs:
        ok = _paint_frame(
            grid,
            blocked,
            np.asarray(spec["key_mask"], dtype=bool),
            int(spec["border_color"]),
            int(spec["key_color"]),
            rng,
        )
        if not ok:
            return None

    rng.shuffle(specs)
    for spec in specs:
        ok = _paint_lockhole(
            grid,
            blocked,
            np.asarray(spec["lock_mask"], dtype=bool),
            int(spec["border_color"]),
            str(spec["side"]),
            int(spec["max_len"]),
            rng,
        )
        if not ok:
            return None

    try:
        out = main(grid)
    except AssertionError:
        return None
    if np.array_equal(out, grid):
        return None
    return grid


def generate_input() -> np.ndarray:
    rng = random.Random()
    for _ in range(GEN_ATTEMPTS):
        grid = _gen_attempt(rng)
        if grid is not None:
            return grid
    raise RuntimeError("failed to generate a valid input")
