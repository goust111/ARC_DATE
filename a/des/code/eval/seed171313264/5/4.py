from __future__ import annotations

from collections import deque
import random

import numpy as np


BG = 0
WALL = 7

LR_KIND = "lr"
UD_KIND = "ud"
ROT_KIND = "rot"
NONE_KIND = "none"
KINDS = (LR_KIND, UD_KIND, ROT_KIND, NONE_KIND)

GEN_ATTEMPTS = 500
LAYOUT_CHOICES = ((1, 2), (2, 1), (1, 3), (3, 1), (2, 2), (2, 3), (3, 2))
ROOM_OUTER_RANGE = (7, 9)
SEPARATED_GAP_RANGE = (1, 3)
MARGIN_RANGE = (1, 4)
MAX_CANVAS = 30
PATTERN_COLORS = (1, 2, 3, 4, 5, 6, 8, 9)


def _neighbors4(r: int, c: int) -> tuple[tuple[int, int], ...]:
    return ((r - 1, c), (r + 1, c), (r, c - 1), (r, c + 1))


def _components(mask: np.ndarray) -> list[list[tuple[int, int]]]:
    h, w = mask.shape
    seen = np.zeros_like(mask, dtype=bool)
    comps: list[list[tuple[int, int]]] = []
    for r in range(h):
        for c in range(w):
            if not mask[r, c] or seen[r, c]:
                continue
            seen[r, c] = True
            queue: deque[tuple[int, int]] = deque([(r, c)])
            comp: list[tuple[int, int]] = []
            while queue:
                cr, cc = queue.popleft()
                comp.append((int(cr), int(cc)))
                for nr, nc in _neighbors4(cr, cc):
                    if 0 <= nr < h and 0 <= nc < w and mask[nr, nc] and not seen[nr, nc]:
                        seen[nr, nc] = True
                        queue.append((int(nr), int(nc)))
            comps.append(comp)
    return comps


def _flood_from_border(passable: np.ndarray) -> np.ndarray:
    h, w = passable.shape
    seen = np.zeros_like(passable, dtype=bool)
    queue: deque[tuple[int, int]] = deque()
    for r in range(h):
        for c in (0, w - 1):
            if passable[r, c] and not seen[r, c]:
                seen[r, c] = True
                queue.append((int(r), int(c)))
    for c in range(w):
        for r in (0, h - 1):
            if passable[r, c] and not seen[r, c]:
                seen[r, c] = True
                queue.append((int(r), int(c)))
    while queue:
        r, c = queue.popleft()
        for nr, nc in _neighbors4(r, c):
            if 0 <= nr < h and 0 <= nc < w and passable[nr, nc] and not seen[nr, nc]:
                seen[nr, nc] = True
                queue.append((int(nr), int(nc)))
    return seen


def _bbox(points: list[tuple[int, int]]) -> tuple[int, int, int, int]:
    rows = [r for r, _ in points]
    cols = [c for _, c in points]
    return min(rows), min(cols), max(rows), max(cols)


def _mask_bbox(mask: np.ndarray) -> tuple[int, int, int, int]:
    rows = np.where(mask.any(axis=1))[0]
    cols = np.where(mask.any(axis=0))[0]
    assert rows.size > 0 and cols.size > 0, "pattern must be nonempty"
    return int(rows[0]), int(cols[0]), int(rows[-1]), int(cols[-1])


def _trim_mask(mask: np.ndarray) -> np.ndarray:
    r0, c0, r1, c1 = _mask_bbox(mask)
    return mask[r0 : r1 + 1, c0 : c1 + 1]


def _extract_rooms(grid: np.ndarray) -> list[dict]:
    g = np.asarray(grid, dtype=int)
    enclosed = (g != WALL) & ~_flood_from_border(g != WALL)
    rooms: list[dict] = []
    for comp in _components(enclosed):
        r0, c0, r1, c1 = _bbox(comp)
        mask = np.zeros_like(g, dtype=bool)
        for r, c in comp:
            mask[r, c] = True
        assert bool(np.all(mask[r0 : r1 + 1, c0 : c1 + 1])), "room interior must be rectangular"

        interior = g[r0 : r1 + 1, c0 : c1 + 1]
        colors = sorted(set(map(int, interior.ravel())) - {BG})
        assert len(colors) == 1, "each room must contain exactly one non-black pattern color"
        color = int(colors[0])
        pattern_mask = interior == color
        pr0, pc0, pr1, pc1 = _mask_bbox(pattern_mask)
        pattern = np.full((pr1 - pr0 + 1, pc1 - pc0 + 1), BG, dtype=int)
        pattern[pattern_mask[pr0 : pr1 + 1, pc0 : pc1 + 1]] = color
        rooms.append(
            {
                "bbox": (int(r0), int(c0), int(r1), int(c1)),
                "pattern": pattern,
            }
        )
    assert rooms, "expected at least one room"
    return rooms


def _symmetry_flags(mask: np.ndarray) -> tuple[bool, bool, bool]:
    lr = bool(np.array_equal(mask, np.fliplr(mask)))
    ud = bool(np.array_equal(mask, np.flipud(mask)))
    rot = bool(np.array_equal(mask, np.rot90(mask, 2)))
    return lr, ud, rot


def _transform_pattern(pattern: np.ndarray) -> np.ndarray:
    mask = pattern != BG
    assert bool(mask.any())
    lr, ud, rot = _symmetry_flags(mask)
    if lr:
        return np.flipud(pattern)
    if ud:
        return np.fliplr(pattern)
    if rot:
        return np.rot90(pattern, -1)
    return pattern.copy()


def _pack_axis(
    bboxes: list[tuple[int, int, int, int]],
    sizes: list[int],
    start_idx: int,
    end_idx: int,
) -> list[int]:
    n = len(bboxes)
    starts = [int(box[start_idx]) for box in bboxes]
    ends = [int(box[end_idx]) for box in bboxes]
    order = sorted(range(n), key=lambda idx: (starts[idx], ends[idx], idx))
    pos = [0] * n
    for j in order:
        best = 0
        for i in order:
            if ends[i] < starts[j]:
                best = max(best, pos[i] + int(sizes[i]) + 1)
        pos[j] = int(best)
    return pos


def main(grid: np.ndarray) -> np.ndarray:
    """
    Remove the orange room borders, transform each room's monochrome motif by
    its symmetry type, and pack the transformed motifs so the left/right and
    above/below relations match the original room layout.
    """
    g = np.asarray(grid, dtype=int)
    rooms = _extract_rooms(g)
    patterns = [_transform_pattern(room["pattern"]) for room in rooms]
    bboxes = [room["bbox"] for room in rooms]
    xs = _pack_axis(bboxes, [pattern.shape[1] for pattern in patterns], 1, 3)
    ys = _pack_axis(bboxes, [pattern.shape[0] for pattern in patterns], 0, 2)

    out_h = max(y + pattern.shape[0] for y, pattern in zip(ys, patterns))
    out_w = max(x + pattern.shape[1] for x, pattern in zip(xs, patterns))
    out = np.full((out_h, out_w), BG, dtype=int)

    for pattern, top, left in zip(patterns, ys, xs):
        h, w = pattern.shape
        view = out[top : top + h, left : left + w]
        mask = pattern != BG
        assert not bool(np.any(mask & (view != BG))), "packed motifs must not overlap"
        view[mask] = pattern[mask]
    return out


def _is_connected(mask: np.ndarray) -> bool:
    pts = np.argwhere(mask)
    if pts.size == 0:
        return False
    start = (int(pts[0, 0]), int(pts[0, 1]))
    queue: deque[tuple[int, int]] = deque([start])
    seen = {start}
    target = {tuple(map(int, pt)) for pt in pts}
    while queue:
        r, c = queue.popleft()
        for nr, nc in _neighbors4(r, c):
            if (nr, nc) in target and (nr, nc) not in seen:
                seen.add((int(nr), int(nc)))
                queue.append((int(nr), int(nc)))
    return len(seen) == len(target)


def _pattern_kind(mask: np.ndarray) -> str | None:
    lr, ud, rot = _symmetry_flags(mask)
    if lr and not ud:
        return LR_KIND
    if ud and not lr:
        return UD_KIND
    if rot and not lr and not ud:
        return ROT_KIND
    if not lr and not ud and not rot:
        return NONE_KIND
    return None


def _usable_mask(mask: np.ndarray) -> bool:
    count = int(mask.sum())
    if count < 4 or count >= int(mask.size):
        return False
    rows = np.where(mask.any(axis=1))[0]
    cols = np.where(mask.any(axis=0))[0]
    if rows.size < 2 or cols.size < 2:
        return False
    return _is_connected(mask)


def _random_bool(h: int, w: int) -> np.ndarray:
    prob = random.uniform(0.30, 0.62)
    return np.array([[random.random() < prob for _ in range(w)] for _ in range(h)], dtype=bool)


def _make_lr_mask(h: int, w: int) -> np.ndarray:
    base = _random_bool(h, w // 2 + (w % 2))
    left = base[:, : w // 2]
    center = base[:, w // 2 : w // 2 + (w % 2)]
    return np.concatenate([left, center, np.fliplr(left)], axis=1)


def _make_ud_mask(h: int, w: int) -> np.ndarray:
    base = _random_bool(h // 2 + (h % 2), w)
    top = base[: h // 2, :]
    center = base[h // 2 : h // 2 + (h % 2), :]
    return np.concatenate([top, center, np.flipud(top)], axis=0)


def _make_rot_mask(h: int, w: int) -> np.ndarray:
    mask = np.zeros((h, w), dtype=bool)
    seen: set[tuple[int, int]] = set()
    prob = random.uniform(0.28, 0.60)
    for r in range(h):
        for c in range(w):
            if (r, c) in seen:
                continue
            rr, cc = h - 1 - r, w - 1 - c
            value = random.random() < prob
            mask[r, c] = value
            mask[rr, cc] = value
            seen.add((r, c))
            seen.add((rr, cc))
    return mask


def _make_none_mask(h: int, w: int) -> np.ndarray:
    return _random_bool(h, w)


def _sample_mask(kind: str, max_h: int, max_w: int) -> np.ndarray:
    assert max_h >= 3 and max_w >= 3
    for _ in range(400):
        h = random.randint(3, min(5, max_h))
        w = random.randint(3, min(5, max_w))
        if kind == LR_KIND:
            mask = _make_lr_mask(h, w)
        elif kind == UD_KIND:
            mask = _make_ud_mask(h, w)
        elif kind == ROT_KIND:
            mask = _make_rot_mask(h, w)
        else:
            mask = _make_none_mask(h, w)
        if not bool(mask.any()):
            continue
        mask = _trim_mask(mask)
        mh, mw = mask.shape
        if mh < 3 or mw < 3 or mh > max_h or mw > max_w:
            continue
        if _pattern_kind(mask) == kind and _usable_mask(mask):
            return mask
    raise RuntimeError(f"failed to sample a {kind} pattern")


def _build_room(height: int, width: int, color: int, kind: str) -> np.ndarray:
    interior_h = int(height - 2)
    interior_w = int(width - 2)
    mask = _sample_mask(kind, interior_h, interior_w)
    mh, mw = mask.shape
    top = random.randint(0, interior_h - mh)
    left = random.randint(0, interior_w - mw)

    room = np.full((height, width), BG, dtype=int)
    room[0, :] = WALL
    room[-1, :] = WALL
    room[:, 0] = WALL
    room[:, -1] = WALL
    room[1 + top : 1 + top + mh, 1 + left : 1 + left + mw][mask] = int(color)
    return room


def _paste_nonzero(dst: np.ndarray, src: np.ndarray, top: int, left: int) -> bool:
    h, w = src.shape
    view = dst[top : top + h, left : left + w]
    src_mask = src != BG
    overlap = src_mask & (view != BG)
    if bool(np.any(view[overlap] != src[overlap])):
        return False
    view[src_mask] = src[src_mask]
    return True


def _offsets(lengths: list[int], gaps: list[int]) -> list[int]:
    offsets = [0]
    for size, gap in zip(lengths[:-1], gaps):
        offsets.append(offsets[-1] + size - 1 if gap == -1 else offsets[-1] + size + gap)
    return offsets


def _choose_kinds(n_rooms: int) -> list[str]:
    kinds: list[str] = []
    while len(kinds) < n_rooms:
        kinds.extend(random.sample(list(KINDS), len(KINDS)))
    chosen = kinds[:n_rooms]
    random.shuffle(chosen)
    return chosen


def _gen_attempt() -> np.ndarray | None:
    rows, cols = random.choice(LAYOUT_CHOICES)
    room_count = int(rows * cols)
    row_heights = [random.randint(*ROOM_OUTER_RANGE) for _ in range(rows)]
    col_widths = [random.randint(*ROOM_OUTER_RANGE) for _ in range(cols)]
    row_gaps = [-1 if random.random() < 0.5 else random.randint(*SEPARATED_GAP_RANGE) for _ in range(rows - 1)]
    col_gaps = [-1 if random.random() < 0.5 else random.randint(*SEPARATED_GAP_RANGE) for _ in range(cols - 1)]

    row_offsets = _offsets(row_heights, row_gaps)
    col_offsets = _offsets(col_widths, col_gaps)
    patch_h = int(row_offsets[-1] + row_heights[-1])
    patch_w = int(col_offsets[-1] + col_widths[-1])
    if patch_h > MAX_CANVAS - 2 or patch_w > MAX_CANVAS - 2:
        return None

    colors = random.sample(list(PATTERN_COLORS), room_count)
    kinds = _choose_kinds(room_count)
    patch = np.full((patch_h, patch_w), BG, dtype=int)

    idx = 0
    for r in range(rows):
        for c in range(cols):
            room = _build_room(row_heights[r], col_widths[c], colors[idx], kinds[idx])
            if not _paste_nonzero(patch, room, row_offsets[r], col_offsets[c]):
                return None
            idx += 1

    top = random.randint(*MARGIN_RANGE)
    left = random.randint(*MARGIN_RANGE)
    bottom = random.randint(*MARGIN_RANGE)
    right = random.randint(*MARGIN_RANGE)
    canvas_h = patch_h + top + bottom
    canvas_w = patch_w + left + right
    if canvas_h > MAX_CANVAS or canvas_w > MAX_CANVAS:
        return None

    canvas = np.full((canvas_h, canvas_w), BG, dtype=int)
    if not _paste_nonzero(canvas, patch, top, left):
        return None

    try:
        rooms = _extract_rooms(canvas)
        out = main(canvas)
    except Exception:
        return None

    if len(rooms) != room_count:
        return None
    if np.array_equal(out, canvas):
        return None
    return canvas


def generate_input() -> np.ndarray:
    for _ in range(GEN_ATTEMPTS):
        candidate = _gen_attempt()
        if candidate is not None:
            return candidate
    raise RuntimeError("failed to generate a valid room-symmetry puzzle")
