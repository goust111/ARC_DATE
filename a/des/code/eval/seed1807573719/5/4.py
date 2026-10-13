from __future__ import annotations

from collections import deque
import random

import numpy as np


BLACK = 0
GRAY = 5
AZURE = 8
PALETTE = [1, 2, 3, 4, 6, 7, 9]
GEN_RETRIES = 400


def _components(mask: np.ndarray, connectivity: int = 4) -> list[list[tuple[int, int]]]:
    h, w = mask.shape
    seen = np.zeros((h, w), dtype=bool)
    if connectivity == 8:
        steps = [
            (-1, -1),
            (-1, 0),
            (-1, 1),
            (0, -1),
            (0, 1),
            (1, -1),
            (1, 0),
            (1, 1),
        ]
    else:
        steps = [(-1, 0), (1, 0), (0, -1), (0, 1)]

    comps: list[list[tuple[int, int]]] = []
    for r in range(h):
        for c in range(w):
            if seen[r, c] or not mask[r, c]:
                continue
            seen[r, c] = True
            q = deque([(r, c)])
            pts: list[tuple[int, int]] = []
            while q:
                cr, cc = q.popleft()
                pts.append((cr, cc))
                for dr, dc in steps:
                    nr, nc = cr + dr, cc + dc
                    if 0 <= nr < h and 0 <= nc < w and mask[nr, nc] and not seen[nr, nc]:
                        seen[nr, nc] = True
                        q.append((nr, nc))
            comps.append(pts)
    return comps


def _bbox(points: list[tuple[int, int]]) -> tuple[int, int, int, int]:
    rs = [r for r, _ in points]
    cs = [c for _, c in points]
    return min(rs), min(cs), max(rs), max(cs)


def _rect_perimeter_cells(r0: int, c0: int, r1: int, c1: int) -> set[tuple[int, int]]:
    cells: set[tuple[int, int]] = set()
    for c in range(c0, c1 + 1):
        cells.add((r0, c))
        cells.add((r1, c))
    for r in range(r0, r1 + 1):
        cells.add((r, c0))
        cells.add((r, c1))
    return cells


def _find_rectangles(g: np.ndarray) -> list[tuple[int, int, int, int]]:
    rects: list[tuple[int, int, int, int]] = []
    for pts in _components(g == GRAY, connectivity=4):
        r0, c0, r1, c1 = _bbox(pts)
        if r1 - r0 < 2 or c1 - c0 < 2:
            continue
        if set(pts) == _rect_perimeter_cells(r0, c0, r1, c1):
            rects.append((r0, c0, r1, c1))
    return rects


def _inside_bbox(r: int, c: int, box: tuple[int, int, int, int]) -> bool:
    r0, c0, r1, c1 = box
    return r0 <= r <= r1 and c0 <= c <= c1


def _extract_key_mask(points: list[tuple[int, int]]) -> np.ndarray:
    r0, c0, r1, c1 = _bbox(points)
    mask = np.zeros((r1 - r0 + 1, c1 - c0 + 1), dtype=bool)
    for r, c in points:
        mask[r - r0, c - c0] = True
    return mask


def _find_azure_line(g: np.ndarray) -> int:
    rows = [r for r in range(g.shape[0]) if np.all(g[r] == AZURE)]
    assert rows, "expected a full-width azure line"
    return max(rows)


def _point_neighbors4(r: int, c: int, n: int) -> list[tuple[int, int]]:
    out: list[tuple[int, int]] = []
    for dr, dc in [(-1, 0), (1, 0), (0, -1), (0, 1)]:
        nr, nc = r + dr, c + dc
        if 0 <= nr < n and 0 <= nc < n:
            out.append((nr, nc))
    return out


def main(grid: np.ndarray) -> np.ndarray:
    g = np.asarray(grid, dtype=int)
    h, w = g.shape
    line_row = _find_azure_line(g)

    rects = _find_rectangles(g)
    slots: list[tuple[int, int, int, int]] = []
    frames: list[tuple[int, int, int, int]] = []
    for box in rects:
        r0, c0, r1, c1 = box
        interior = g[r0 + 1 : r1, c0 + 1 : c1]
        if r1 + 1 == line_row and interior.size > 0 and np.all(interior == BLACK):
            slots.append(box)
        else:
            frames.append(box)

    slots.sort(key=lambda b: b[1])
    frames.sort(key=lambda b: b[1])
    assert slots and len(slots) == len(frames), "frames and output slots must match"

    n = slots[0][2] - slots[0][0] - 1
    assert n == slots[0][3] - slots[0][1] - 1, "slot interior must be square"
    for r0, c0, r1, c1 in slots:
        assert r1 - r0 - 1 == n and c1 - c0 - 1 == n, "all slots must share one size"

    marker_positions: dict[tuple[int, int], int] = {}
    for r0, c0, r1, c1 in frames:
        assert r1 + 1 < h and c1 + 1 < w, "frame marker must be in-bounds"
        marker = int(g[r1 + 1, c1 + 1])
        assert marker not in (BLACK, GRAY, AZURE), "frame marker must be colored"
        marker_positions[(r1 + 1, c1 + 1)] = marker

    nonfixed = (~np.isin(g, [BLACK, GRAY, AZURE])).astype(bool)
    key_masks: dict[int, np.ndarray] = {}
    for pts in _components(nonfixed, connectivity=4):
        colors = {int(g[r, c]) for r, c in pts}
        assert len(colors) == 1, "each movable key must be monochrome"
        color = colors.pop()
        if len(pts) == 1 and pts[0] in marker_positions:
            continue
        if any(_inside_bbox(r, c, box) for r, c in pts for box in rects):
            continue
        assert color not in key_masks, "key colors must be unique"
        mask = _extract_key_mask(pts)
        key_masks[color] = mask

    assert len(key_masks) == len(frames), "must detect one key per frame"

    out = g.copy()
    for idx, frame in enumerate(frames):
        r0, c0, r1, c1 = frame
        interior = g[r0 + 1 : r1, c0 + 1 : c1]
        assert interior.shape == (n, n), "frame and slot sizes must agree"

        fill_colors = set(map(int, np.unique(interior))) - {BLACK}
        assert len(fill_colors) == 1, "each frame must expose exactly one key color"
        fill_color = fill_colors.pop()
        assert fill_color in key_masks, "frame color must match one key"
        key_mask = key_masks[fill_color]
        assert key_mask.shape == interior.shape, "key bbox must match frame interior"

        visible_fill = interior == fill_color
        assert not np.any(visible_fill & key_mask), "visible fill cannot overlap key space"
        hidden_pattern = (interior == BLACK) & (~key_mask)

        sr0, sc0, sr1, sc1 = slots[idx]
        marker_color = marker_positions[(r1 + 1, c1 + 1)]
        slot = out[sr0 + 1 : sr1, sc0 + 1 : sc1]
        slot[:] = BLACK
        slot[hidden_pattern] = marker_color

    return out


def _random_path(
    rng: random.Random, start: tuple[int, int], end: tuple[int, int]
) -> list[tuple[int, int]]:
    r, c = start
    tr, tc = end
    path = [(r, c)]
    while (r, c) != (tr, tc):
        steps: list[tuple[int, int]] = []
        if r < tr:
            steps.append((1, 0))
        elif r > tr:
            steps.append((-1, 0))
        if c < tc:
            steps.append((0, 1))
        elif c > tc:
            steps.append((0, -1))
        dr, dc = rng.choice(steps)
        r += dr
        c += dc
        path.append((r, c))
    return path


def _touches_all_sides(mask: np.ndarray) -> bool:
    return (
        np.any(mask[0, :])
        and np.any(mask[-1, :])
        and np.any(mask[:, 0])
        and np.any(mask[:, -1])
    )


def _random_key_mask(rng: random.Random, n: int) -> np.ndarray:
    min_target = max(2 * n, (n * n) // 3)
    max_target = min(n * n - 2, (2 * n * n) // 3 + 1)
    for _ in range(300):
        anchors = [
            (0, rng.randrange(n)),
            (n - 1, rng.randrange(n)),
            (rng.randrange(n), 0),
            (rng.randrange(n), n - 1),
        ]
        cells: set[tuple[int, int]] = {anchors[0]}
        for anchor in anchors[1:]:
            start = rng.choice(tuple(cells))
            cells.update(_random_path(rng, start, anchor))

        target = rng.randint(min_target, max_target)
        while len(cells) < target:
            frontier = []
            for r, c in cells:
                for nr, nc in _point_neighbors4(r, c, n):
                    if (nr, nc) not in cells:
                        frontier.append((nr, nc))
            if not frontier:
                break
            cells.add(rng.choice(frontier))

        mask = np.zeros((n, n), dtype=bool)
        for r, c in cells:
            mask[r, c] = True

        if not _touches_all_sides(mask):
            continue
        if mask.sum() < min_target or mask.sum() > max_target:
            continue
        if (~mask).sum() < 2:
            continue
        if len(_components(mask, connectivity=4)) != 1:
            continue
        return mask
    raise RuntimeError("failed to generate a key mask")


def _random_hidden_pattern(rng: random.Random, available: np.ndarray) -> np.ndarray:
    coords = [tuple(x) for x in np.argwhere(available)]
    assert len(coords) >= 2
    limit = min(len(coords) - 1, max(1, len(coords) // 2))
    target = rng.randint(1, limit)

    if rng.random() < 0.5:
        chosen = {rng.choice(coords)}
        while len(chosen) < target:
            frontier = []
            for r, c in chosen:
                for nr, nc in _point_neighbors4(r, c, available.shape[0]):
                    if available[nr, nc] and (nr, nc) not in chosen:
                        frontier.append((nr, nc))
            if not frontier:
                break
            chosen.add(rng.choice(frontier))
    else:
        chosen = set(rng.sample(coords, target))

    mask = np.zeros_like(available, dtype=bool)
    for r, c in chosen:
        mask[r, c] = True
    if mask.sum() == 0 or mask.sum() >= available.sum():
        return _random_hidden_pattern(rng, available)
    return mask


def _inflate(box: tuple[int, int, int, int], h: int, w: int, pad: int = 1) -> tuple[int, int, int, int]:
    r0, c0, r1, c1 = box
    return max(0, r0 - pad), max(0, c0 - pad), min(h - 1, r1 + pad), min(w - 1, c1 + pad)


def _mark_box(mask: np.ndarray, box: tuple[int, int, int, int]) -> None:
    r0, c0, r1, c1 = box
    mask[r0 : r1 + 1, c0 : c1 + 1] = True


def _draw_rect_border(g: np.ndarray, box: tuple[int, int, int, int], color: int) -> None:
    r0, c0, r1, c1 = box
    g[r0, c0 : c1 + 1] = color
    g[r1, c0 : c1 + 1] = color
    g[r0 : r1 + 1, c0] = color
    g[r0 : r1 + 1, c1] = color


def _gen_attempt(rng: random.Random) -> np.ndarray | None:
    n = rng.choice([4, 5, 6])
    count = rng.choice([2, 3])
    key_colors = rng.sample(PALETTE, count)
    marker_pool = [c for c in PALETTE if c not in key_colors]
    marker_colors = rng.sample(marker_pool, count)

    outer = n + 2
    frame_left = rng.randint(1, 3)
    slot_left = rng.randint(1, 3)
    frame_gaps = [rng.randint(2, 5) for _ in range(count - 1)]
    slot_gaps = [rng.randint(2, 4) for _ in range(count - 1)]

    frame_cols = [frame_left]
    slot_cols = [slot_left]
    for gap in frame_gaps:
        frame_cols.append(frame_cols[-1] + outer + gap)
    for gap in slot_gaps:
        slot_cols.append(slot_cols[-1] + outer + gap)

    min_w = max(frame_cols[-1] + outer + 3, slot_cols[-1] + outer + 3, count * (n + 4) + 6)
    w = min_w + rng.randint(2, n + 6)

    frame_top = rng.randint(1, 3)
    middle_band = 2 * n + rng.randint(4, 8)
    h = frame_top + outer + middle_band + outer + 1
    slot_top = h - 1 - outer
    if slot_top - (frame_top + outer) < n + 3:
        return None

    g = np.full((h, w), BLACK, dtype=int)
    occupied = np.zeros((h, w), dtype=bool)

    slots: list[tuple[int, int, int, int]] = []
    frames: list[tuple[int, int, int, int]] = []
    pairs: list[dict[str, object]] = []

    for idx in range(count):
        frame_box = (frame_top, frame_cols[idx], frame_top + outer - 1, frame_cols[idx] + outer - 1)
        slot_box = (slot_top, slot_cols[idx], slot_top + outer - 1, slot_cols[idx] + outer - 1)
        marker_pos = (frame_box[2] + 1, frame_box[3] + 1)
        if marker_pos[0] >= h or marker_pos[1] >= w:
            return None

        if occupied[_inflate(frame_box, h, w, 1)[0] : _inflate(frame_box, h, w, 1)[2] + 1, _inflate(frame_box, h, w, 1)[1] : _inflate(frame_box, h, w, 1)[3] + 1].any():
            return None
        if occupied[_inflate(slot_box, h, w, 1)[0] : _inflate(slot_box, h, w, 1)[2] + 1, _inflate(slot_box, h, w, 1)[1] : _inflate(slot_box, h, w, 1)[3] + 1].any():
            return None

        key_mask = _random_key_mask(rng, n)
        hidden_mask = _random_hidden_pattern(rng, ~key_mask)
        visible_mask = (~key_mask) & (~hidden_mask)
        if not np.any(visible_mask):
            return None

        _draw_rect_border(g, frame_box, GRAY)
        _draw_rect_border(g, slot_box, GRAY)
        g[marker_pos] = marker_colors[idx]
        g[frame_box[0] + 1 : frame_box[2], frame_box[1] + 1 : frame_box[3]][visible_mask] = key_colors[idx]

        _mark_box(occupied, _inflate(frame_box, h, w, 1))
        _mark_box(occupied, _inflate(slot_box, h, w, 1))
        _mark_box(occupied, _inflate((marker_pos[0], marker_pos[1], marker_pos[0], marker_pos[1]), h, w, 1))

        slots.append(slot_box)
        frames.append(frame_box)
        pairs.append(
            {
                "key_color": key_colors[idx],
                "marker_color": marker_colors[idx],
                "key_mask": key_mask,
                "hidden_mask": hidden_mask,
                "slot_box": slot_box,
            }
        )

    g[-1, :] = AZURE

    key_row_min = frame_top + outer + 1
    key_row_max = slot_top - n - 1
    if key_row_max < key_row_min:
        return None

    for pair in pairs:
        placed = False
        for _ in range(200):
            r0 = rng.randint(key_row_min, key_row_max)
            c0 = rng.randint(1, w - n - 2)
            box = (r0, c0, r0 + n - 1, c0 + n - 1)
            br0, bc0, br1, bc1 = _inflate(box, h, w, 1)
            if occupied[br0 : br1 + 1, bc0 : bc1 + 1].any():
                continue
            key_mask = pair["key_mask"]
            assert isinstance(key_mask, np.ndarray)
            sub = g[r0 : r0 + n, c0 : c0 + n]
            sub[key_mask] = int(pair["key_color"])
            _mark_box(occupied, (br0, bc0, br1, bc1))
            placed = True
            break
        if not placed:
            return None

    out = main(g)
    for pair in pairs:
        hidden_mask = pair["hidden_mask"]
        slot_box = pair["slot_box"]
        assert isinstance(hidden_mask, np.ndarray)
        assert isinstance(slot_box, tuple)
        sr0, sc0, sr1, sc1 = slot_box
        slot = out[sr0 + 1 : sr1, sc0 + 1 : sc1]
        expected = np.full((n, n), BLACK, dtype=int)
        expected[hidden_mask] = int(pair["marker_color"])
        if not np.array_equal(slot, expected):
            return None

    return g


def generate_input() -> np.ndarray:
    rng = random.Random()
    for _ in range(GEN_RETRIES):
        grid = _gen_attempt(rng)
        if grid is not None:
            return grid
    raise RuntimeError("generate_input: exhausted retries")
