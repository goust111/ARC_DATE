from __future__ import annotations

from collections import deque
import random

import numpy as np


BG = 0
ROW_GAP = 1
COL_GAP = 1
MAX_GENERATION_ATTEMPTS = 400
MAX_OBJECT_BUILD_ATTEMPTS = 200
MAX_PLACEMENT_ATTEMPTS = 80
COLORS = tuple(range(1, 10))


def _neighbors4(r: int, c: int) -> tuple[tuple[int, int], ...]:
    return ((r - 1, c), (r + 1, c), (r, c - 1), (r, c + 1))


def _components(mask: np.ndarray) -> list[list[tuple[int, int]]]:
    h, w = mask.shape
    seen = np.zeros((h, w), dtype=bool)
    comps: list[list[tuple[int, int]]] = []
    for r in range(h):
        for c in range(w):
            if not mask[r, c] or seen[r, c]:
                continue
            seen[r, c] = True
            q: deque[tuple[int, int]] = deque([(r, c)])
            pts: list[tuple[int, int]] = []
            while q:
                rr, cc = q.popleft()
                pts.append((rr, cc))
                for nr, nc in _neighbors4(rr, cc):
                    if 0 <= nr < h and 0 <= nc < w and mask[nr, nc] and not seen[nr, nc]:
                        seen[nr, nc] = True
                        q.append((nr, nc))
            comps.append(pts)
    return comps


def _bbox(points: list[tuple[int, int]]) -> tuple[int, int, int, int]:
    rows = [r for r, _ in points]
    cols = [c for _, c in points]
    return min(rows), min(cols), max(rows), max(cols)


def _crop_mask(mask: np.ndarray) -> np.ndarray:
    pts = np.argwhere(mask)
    if pts.size == 0:
        return np.zeros((1, 1), dtype=bool)
    r0, c0 = pts.min(axis=0)
    r1, c1 = pts.max(axis=0)
    return mask[int(r0) : int(r1) + 1, int(c0) : int(c1) + 1]


def _is_connected(mask: np.ndarray) -> bool:
    return np.count_nonzero(mask) > 0 and len(_components(mask)) == 1


def _count_holes(mask: np.ndarray) -> int:
    h, w = mask.shape
    holes = 0
    for comp in _components(~mask):
        touches_edge = any(r == 0 or c == 0 or r == h - 1 or c == w - 1 for r, c in comp)
        if not touches_edge:
            holes += 1
    return holes


def _extract_objects(grid: np.ndarray) -> list[dict[str, object]]:
    g = np.asarray(grid, dtype=int)
    objects: list[dict[str, object]] = []
    for color in sorted(int(v) for v in np.unique(g) if int(v) != BG):
        for pts in _components(g == color):
            r0, c0, r1, c1 = _bbox(pts)
            sprite = np.zeros((r1 - r0 + 1, c1 - c0 + 1), dtype=int)
            mask = np.zeros_like(sprite, dtype=bool)
            for r, c in pts:
                sprite[r - r0, c - c0] = color
                mask[r - r0, c - c0] = True
            objects.append(
                {
                    "color": color,
                    "bbox": (r0, c0, r1, c1),
                    "sprite": sprite,
                    "area": int(mask.sum()),
                    "holes": _count_holes(mask),
                }
            )
    return objects


def _pack_objects(objects: list[dict[str, object]]) -> np.ndarray:
    if not objects:
        return np.zeros((1, 1), dtype=int)

    grouped: dict[int, list[dict[str, object]]] = {}
    for obj in objects:
        grouped.setdefault(int(obj["holes"]), []).append(obj)

    rows: list[list[dict[str, object]]] = []
    for holes in sorted(grouped):
        row_objs = sorted(
            grouped[holes],
            key=lambda obj: (
                int(obj["area"]),
                int(obj["bbox"][1]),
                int(obj["bbox"][0]),
            ),
        )
        rows.append(row_objs)

    row_heights = [max(int(obj["sprite"].shape[0]) for obj in row) for row in rows]
    row_widths = [
        sum(int(obj["sprite"].shape[1]) for obj in row) + COL_GAP * (len(row) - 1)
        for row in rows
    ]
    out_h = sum(row_heights) + ROW_GAP * (len(rows) - 1)
    out_w = max(row_widths)

    out = np.zeros((out_h, out_w), dtype=int)
    cur_r = 0
    for row_idx, row in enumerate(rows):
        cur_c = 0
        for obj in row:
            sprite = np.asarray(obj["sprite"], dtype=int)
            sh, sw = sprite.shape
            target = out[cur_r : cur_r + sh, cur_c : cur_c + sw]
            mask = sprite != BG
            target[mask] = sprite[mask]
            cur_c += sw + COL_GAP
        cur_r += row_heights[row_idx]
        if row_idx + 1 < len(rows):
            cur_r += ROW_GAP

    return out


def main(grid: np.ndarray) -> np.ndarray:
    """
    Group objects by their number of enclosed black holes, then pack each group.
    """
    g = np.asarray(grid, dtype=int)
    assert g.ndim == 2
    objects = _extract_objects(g)
    return _pack_objects(objects)


def _random_transform(mask: np.ndarray, rng: random.Random) -> np.ndarray:
    out = np.array(mask, dtype=bool)
    out = np.rot90(out, rng.randrange(4))
    if rng.random() < 0.5:
        out = np.fliplr(out)
    if rng.random() < 0.5:
        out = np.flipud(out)
    return np.array(out, dtype=bool)


def _random_polyomino_zero(rng: random.Random) -> np.ndarray | None:
    h = rng.randint(3, 6)
    w = rng.randint(3, 6)
    area = rng.randint(4, min(h * w - 1, 10))
    start = (rng.randrange(h), rng.randrange(w))
    cells = {start}
    for _ in range(area - 1):
        frontier = set()
        for r, c in cells:
            for nr, nc in _neighbors4(r, c):
                if 0 <= nr < h and 0 <= nc < w and (nr, nc) not in cells:
                    frontier.add((nr, nc))
        if not frontier:
            return None
        cells.add(rng.choice(tuple(frontier)))
    mask = np.zeros((h, w), dtype=bool)
    for r, c in cells:
        mask[r, c] = True
    mask = _crop_mask(mask)
    if not _is_connected(mask) or _count_holes(mask) != 0:
        return None
    return mask


def _open_frame_zero(rng: random.Random) -> np.ndarray | None:
    h = rng.randint(4, 7)
    w = rng.randint(4, 7)
    mask = np.zeros((h, w), dtype=bool)
    mask[0, :] = True
    mask[-1, :] = True
    mask[:, 0] = True
    mask[:, -1] = True
    side = rng.choice(("top", "bottom", "left", "right"))
    if side in ("top", "bottom"):
        span = rng.randint(1, max(1, w - 2))
        start = rng.randint(1, w - span - 1)
        row = 0 if side == "top" else h - 1
        mask[row, start : start + span] = False
    else:
        span = rng.randint(1, max(1, h - 2))
        start = rng.randint(1, h - span - 1)
        col = 0 if side == "left" else w - 1
        mask[start : start + span, col] = False
    if not _is_connected(mask) or _count_holes(mask) != 0:
        return None
    return mask


def _carve_edge_notches(mask: np.ndarray, rng: random.Random) -> np.ndarray:
    out = np.array(mask, dtype=bool)
    notch_count = rng.randint(0, 3)
    for _ in range(notch_count):
        h, w = out.shape
        if h <= 2 or w <= 2:
            break
        side = rng.choice(("top", "bottom", "left", "right"))
        if side in ("top", "bottom"):
            depth = rng.randint(1, min(2, h - 1))
            span = rng.randint(1, max(1, w - 2))
            start = rng.randint(0, w - span)
            row_slice = slice(0, depth) if side == "top" else slice(h - depth, h)
            out[row_slice, start : start + span] = False
        else:
            depth = rng.randint(1, min(2, w - 1))
            span = rng.randint(1, max(1, h - 2))
            start = rng.randint(0, h - span)
            col_slice = slice(0, depth) if side == "left" else slice(w - depth, w)
            out[start : start + span, col_slice] = False
    return _crop_mask(out)


def _solid_with_holes(target_holes: int, rng: random.Random) -> np.ndarray | None:
    if target_holes == 0:
        outer_h = rng.randint(3, 7)
        outer_w = rng.randint(3, 7)
    elif target_holes == 1:
        outer_h = rng.randint(5, 7)
        outer_w = rng.randint(5, 7)
    else:
        outer_h = rng.randint(6, 8)
        outer_w = rng.randint(6, 8)

    mask = np.ones((outer_h, outer_w), dtype=bool)
    blocked = np.zeros_like(mask, dtype=bool)

    for _ in range(target_holes):
        placed = False
        for _ in range(80):
            max_h = max(1, min(3, outer_h - 2))
            max_w = max(1, min(3, outer_w - 2))
            hole_h = rng.randint(1, max_h)
            hole_w = rng.randint(1, max_w)
            if outer_h - hole_h - 1 < 1 or outer_w - hole_w - 1 < 1:
                continue
            r0 = rng.randint(1, outer_h - hole_h - 1)
            c0 = rng.randint(1, outer_w - hole_w - 1)
            rr0 = max(0, r0 - 1)
            cc0 = max(0, c0 - 1)
            rr1 = min(outer_h, r0 + hole_h + 1)
            cc1 = min(outer_w, c0 + hole_w + 1)
            if blocked[rr0:rr1, cc0:cc1].any():
                continue
            mask[r0 : r0 + hole_h, c0 : c0 + hole_w] = False
            blocked[rr0:rr1, cc0:cc1] = True
            placed = True
            break
        if not placed:
            return None

    mask = _carve_edge_notches(mask, rng)
    mask = _random_transform(mask, rng)
    mask = _crop_mask(mask)
    if np.count_nonzero(mask) < 4:
        return None
    if not _is_connected(mask):
        return None
    if _count_holes(mask) != target_holes:
        return None
    return mask


def _random_mask(target_holes: int, rng: random.Random) -> np.ndarray:
    builders = []
    if target_holes == 0:
        builders = (
            _random_polyomino_zero,
            _open_frame_zero,
            lambda rr: _solid_with_holes(0, rr),
        )
    else:
        builders = (lambda rr: _solid_with_holes(target_holes, rr),)

    for _ in range(MAX_OBJECT_BUILD_ATTEMPTS):
        mask = rng.choice(builders)(rng)
        if mask is None:
            continue
        if _count_holes(mask) == target_holes and _is_connected(mask):
            return mask
    raise RuntimeError(f"failed to build object with {target_holes} holes")


def _mask_to_sprite(mask: np.ndarray, color: int) -> np.ndarray:
    sprite = np.zeros(mask.shape, dtype=int)
    sprite[mask] = color
    return sprite


def _place_objects(records: list[dict[str, object]], rng: random.Random) -> np.ndarray | None:
    max_h = max(int(rec["sprite"].shape[0]) for rec in records)
    max_w = max(int(rec["sprite"].shape[1]) for rec in records)
    count = len(records)

    min_height = max(18, max_h + 4, 8 + count * 3)
    min_width = max(18, max_w + 4, 10 + count * 4)
    if min_height > 30 or min_width > 30:
        return None

    height = rng.randint(min_height, 30)
    width = rng.randint(min_width, 30)

    grid = np.zeros((height, width), dtype=int)
    occupied = np.zeros((height, width), dtype=bool)

    order = records[:]
    rng.shuffle(order)
    for rec in order:
        sprite = np.asarray(rec["sprite"], dtype=int)
        sh, sw = sprite.shape
        placed = False
        positions = [(r, c) for r in range(1, height - sh) for c in range(1, width - sw)]
        rng.shuffle(positions)
        for r0, c0 in positions:
            rr0 = r0 - 1
            cc0 = c0 - 1
            rr1 = r0 + sh + 1
            cc1 = c0 + sw + 1
            if occupied[rr0:rr1, cc0:cc1].any():
                continue
            mask = sprite != BG
            grid[r0 : r0 + sh, c0 : c0 + sw][mask] = sprite[mask]
            occupied[rr0:rr1, cc0:cc1] = True
            rec["bbox"] = (r0, c0, r0 + sh - 1, c0 + sw - 1)
            placed = True
            break
        if not placed:
            return None

    return grid


def _make_records(rng: random.Random) -> list[dict[str, object]]:
    counts = {0: rng.randint(1, 2), 1: rng.randint(1, 2), 2: rng.randint(1, 2)}
    if all(count == 1 for count in counts.values()):
        counts[rng.choice((0, 1, 2))] = 2

    total = sum(counts.values())
    colors = rng.sample(COLORS, total)
    color_idx = 0
    records: list[dict[str, object]] = []
    for holes in (0, 1, 2):
        for _ in range(counts[holes]):
            mask = _random_mask(holes, rng)
            color = colors[color_idx]
            color_idx += 1
            sprite = _mask_to_sprite(mask, color)
            records.append(
                {
                    "color": color,
                    "holes": holes,
                    "area": int(mask.sum()),
                    "sprite": sprite,
                    "bbox": (0, 0, sprite.shape[0] - 1, sprite.shape[1] - 1),
                }
            )
    return records


def _gen_attempt(rng: random.Random) -> np.ndarray | None:
    records = _make_records(rng)
    grid = _place_objects(records, rng)
    if grid is None:
        return None

    expected = _pack_objects(records)
    try:
        actual = main(grid)
    except Exception:
        return None
    if not np.array_equal(actual, expected):
        return None
    if np.array_equal(actual, grid):
        return None
    return grid


def generate_input() -> np.ndarray:
    rng = random.Random()
    for _ in range(MAX_GENERATION_ATTEMPTS):
        grid = _gen_attempt(rng)
        if grid is not None:
            return grid
    raise RuntimeError("failed to generate a valid input")
