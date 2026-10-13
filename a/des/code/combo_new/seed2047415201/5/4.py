from __future__ import annotations

from collections import defaultdict, deque
import random

import numpy as np


AZURE = 8
MAROON = 9
OBJECT_COLORS = tuple(range(8))


def _neighbors4(r: int, c: int):
    yield r - 1, c
    yield r + 1, c
    yield r, c - 1
    yield r, c + 1


def _bbox(coords: list[tuple[int, int]]) -> tuple[int, int, int, int]:
    rows = [r for r, _ in coords]
    cols = [c for _, c in coords]
    return min(rows), min(cols), max(rows), max(cols)


def _is_connected(mask: np.ndarray) -> bool:
    pts = np.argwhere(mask)
    if pts.size == 0:
        return False
    h, w = mask.shape
    start = tuple(map(int, pts[0]))
    seen = {start}
    q: deque[tuple[int, int]] = deque([start])
    while q:
        r, c = q.popleft()
        for nr, nc in _neighbors4(r, c):
            if 0 <= nr < h and 0 <= nc < w and bool(mask[nr, nc]) and (nr, nc) not in seen:
                seen.add((nr, nc))
                q.append((nr, nc))
    return len(seen) == int(np.count_nonzero(mask))


def _count_holes(mask: np.ndarray) -> int:
    h, w = mask.shape
    empty = ~mask
    seen = np.zeros((h, w), dtype=bool)
    holes = 0
    for r in range(h):
        for c in range(w):
            if not bool(empty[r, c]) or bool(seen[r, c]):
                continue
            touches_edge = False
            q: deque[tuple[int, int]] = deque([(r, c)])
            seen[r, c] = True
            while q:
                cr, cc = q.popleft()
                if cr == 0 or cc == 0 or cr == h - 1 or cc == w - 1:
                    touches_edge = True
                for nr, nc in _neighbors4(cr, cc):
                    if 0 <= nr < h and 0 <= nc < w and bool(empty[nr, nc]) and not bool(seen[nr, nc]):
                        seen[nr, nc] = True
                        q.append((nr, nc))
            if not touches_edge:
                holes += 1
    return holes


def _marker_row_and_cols(g: np.ndarray) -> tuple[int, list[int]]:
    candidate_rows = [r for r in range(g.shape[0]) if int(np.count_nonzero(g[r] == MAROON)) >= 5]
    assert candidate_rows, "expected a bottom row with five Maroon dock markers"
    row = max(candidate_rows)
    cols = [int(c) for c in np.where(g[row] == MAROON)[0]]
    assert len(cols) == 5, "expected exactly five Maroon dock markers"
    return int(row), cols


def _objects(g: np.ndarray) -> list[dict[str, object]]:
    h, w = g.shape
    seen = np.zeros((h, w), dtype=bool)
    objects: list[dict[str, object]] = []

    for r in range(h):
        for c in range(w):
            color = int(g[r, c])
            if color in (AZURE, MAROON) or bool(seen[r, c]):
                continue
            coords: list[tuple[int, int]] = []
            q: deque[tuple[int, int]] = deque([(r, c)])
            seen[r, c] = True
            while q:
                cr, cc = q.popleft()
                coords.append((cr, cc))
                for nr, nc in _neighbors4(cr, cc):
                    if (
                        0 <= nr < h
                        and 0 <= nc < w
                        and not bool(seen[nr, nc])
                        and int(g[nr, nc]) == color
                    ):
                        seen[nr, nc] = True
                        q.append((nr, nc))

            r0, c0, r1, c1 = _bbox(coords)
            mask = np.zeros((r1 - r0 + 1, c1 - c0 + 1), dtype=bool)
            for rr, cc in coords:
                mask[rr - r0, cc - c0] = True
            sprite = np.full(mask.shape, AZURE, dtype=int)
            sprite[mask] = color
            objects.append(
                {
                    "row": int(r0),
                    "col": int(c0),
                    "holes": int(_count_holes(mask)),
                    "sprite": sprite,
                }
            )
    return objects


def main(grid: np.ndarray) -> np.ndarray:
    """
    Dock each object above the marker whose index matches its number of holes.

    Input structure:
    - The background is Azure.
    - Several same-color 4-connected foreground objects appear above a bottom row of five Maroon marker pixels.
    - Each object may enclose 0 to 4 separate Azure holes inside its bounding box.

    Rule:
    - Count the enclosed Azure holes of every foreground object.
    - Clear the object's original location.
    - Move it above the marker at that hole-count index, centered on the marker with one Azure row above the marker row.
    - Objects with the same hole count are ordered by their original left-to-right positions and stacked upward with one Azure spacer row.

    Key insight: The object's hidden structural property, its hole count, selects the numbered dock position supplied by the bottom markers.
    """
    g = np.asarray(grid, dtype=int)
    marker_row, marker_cols = _marker_row_and_cols(g)
    out = np.full(g.shape, AZURE, dtype=int)
    out[marker_row, marker_cols] = MAROON

    groups: dict[int, list[dict[str, object]]] = defaultdict(list)
    for obj in _objects(g):
        holes = int(obj["holes"])
        if 0 <= holes < len(marker_cols):
            groups[holes].append(obj)

    for holes, objs in groups.items():
        objs.sort(key=lambda obj: (int(obj["col"]), int(obj["row"])))
        bottom = int(marker_row - 2)
        for obj in objs:
            sprite = np.asarray(obj["sprite"], dtype=int)
            sh, sw = sprite.shape
            top = int(bottom - sh + 1)
            left = int(marker_cols[holes] - sw // 2)
            rr, cc = np.where(sprite != AZURE)
            for dr, dc in zip(rr, cc):
                tr = top + int(dr)
                tc = left + int(dc)
                if 0 <= tr < out.shape[0] and 0 <= tc < out.shape[1]:
                    out[tr, tc] = int(sprite[int(dr), int(dc)])
            bottom = int(top - 2)

    return out


def _valid_mask(mask: np.ndarray, holes: int) -> bool:
    if not _is_connected(mask):
        return False
    if not mask[0].any() or not mask[-1].any() or not mask[:, 0].any() or not mask[:, -1].any():
        return False
    return _count_holes(mask) == holes


def _randomize_orientation(mask: np.ndarray, rng: random.Random) -> np.ndarray:
    out = np.rot90(mask, rng.randrange(4))
    if rng.random() < 0.5:
        out = np.fliplr(out)
    if rng.random() < 0.5:
        out = np.flipud(out)
    return np.array(out, dtype=bool)


def _make_mask_with_holes(holes: int, rng: random.Random) -> np.ndarray:
    if holes == 0:
        for _ in range(100):
            h, w = rng.choice([(3, 3), (3, 5), (5, 3), (5, 5)])
            mask = np.ones((h, w), dtype=bool)
            boundary = (
                [(0, c) for c in range(w)]
                + [(h - 1, c) for c in range(w)]
                + [(r, 0) for r in range(1, h - 1)]
                + [(r, w - 1) for r in range(1, h - 1)]
            )
            rng.shuffle(boundary)
            for r, c in boundary:
                if rng.random() >= 0.35:
                    continue
                trial = mask.copy()
                trial[r, c] = False
                if _valid_mask(trial, 0):
                    mask = trial
            mask = _randomize_orientation(mask, rng)
            if _valid_mask(mask, 0):
                return mask
        return np.ones((3, 3), dtype=bool)

    choices = {
        1: [(1, 1), (1, 2), (2, 1), (2, 2)],
        2: [(1, 2), (2, 1), (2, 2)],
        3: [(2, 2)],
        4: [(2, 2)],
    }
    for _ in range(100):
        hole_rows, hole_cols = rng.choice(choices[holes])
        mask = np.ones((2 * hole_rows + 1, 2 * hole_cols + 1), dtype=bool)
        candidates = [(2 * r + 1, 2 * c + 1) for r in range(hole_rows) for c in range(hole_cols)]
        for r, c in rng.sample(candidates, holes):
            mask[r, c] = False

        corners = [(0, 0), (0, mask.shape[1] - 1), (mask.shape[0] - 1, 0), (mask.shape[0] - 1, mask.shape[1] - 1)]
        rng.shuffle(corners)
        for r, c in corners:
            if rng.random() >= 0.3:
                continue
            trial = mask.copy()
            trial[r, c] = False
            if _valid_mask(trial, holes):
                mask = trial

        mask = _randomize_orientation(mask, rng)
        if _valid_mask(mask, holes):
            return mask

    hole_rows, hole_cols = (2, 2) if holes > 2 else ((1, holes) if holes > 1 else (1, 1))
    mask = np.ones((2 * hole_rows + 1, 2 * hole_cols + 1), dtype=bool)
    for r, c in [(2 * r + 1, 2 * c + 1) for r in range(hole_rows) for c in range(hole_cols)][:holes]:
        mask[r, c] = False
    return mask


def _make_sprite(holes: int, color: int, rng: random.Random) -> np.ndarray:
    mask = _make_mask_with_holes(holes, rng)
    sprite = np.full(mask.shape, AZURE, dtype=int)
    sprite[mask] = int(color)
    return sprite


def _can_place(g: np.ndarray, sprite: np.ndarray, r0: int, c0: int) -> bool:
    sh, sw = sprite.shape
    h, w = g.shape
    if r0 < 1 or c0 < 1 or r0 + sh >= h - 3 or c0 + sw >= w - 1:
        return False
    area = g[r0 - 1 : r0 + sh + 1, c0 - 1 : c0 + sw + 1]
    return bool(np.all(area == AZURE))


def _paste_sprite(g: np.ndarray, sprite: np.ndarray, r0: int, c0: int) -> None:
    rr, cc = np.where(sprite != AZURE)
    for dr, dc in zip(rr, cc):
        g[r0 + int(dr), c0 + int(dc)] = int(sprite[int(dr), int(dc)])


def _gen_attempt(rng: random.Random) -> np.ndarray | None:
    height = rng.randint(25, 30)
    width = rng.randint(29, 30)
    spacing = 6
    left_max = width - 3 - spacing * 4
    if left_max < 2:
        return None
    first_marker = rng.randint(2, left_max)
    marker_cols = [first_marker + spacing * i for i in range(5)]

    grid = np.full((height, width), AZURE, dtype=int)
    grid[height - 1, marker_cols] = MAROON

    hole_counts = [0, 1, 2, 3, 4]
    hole_counts.extend(rng.randrange(5) for _ in range(rng.randint(0, 3)))
    rng.shuffle(hole_counts)
    colors = rng.sample(list(OBJECT_COLORS), len(hole_counts))

    sprites = [_make_sprite(holes, color, rng) for holes, color in zip(hole_counts, colors)]
    for sprite in sprites:
        placed = False
        sh, sw = sprite.shape
        max_r = height - sh - 4
        max_c = width - sw - 2
        if max_r < 1 or max_c < 1:
            return None
        for _ in range(500):
            r0 = rng.randint(1, max_r)
            c0 = rng.randint(1, max_c)
            if _can_place(grid, sprite, r0, c0):
                _paste_sprite(grid, sprite, r0, c0)
                placed = True
                break
        if not placed:
            return None

    seen_holes = sorted(int(obj["holes"]) for obj in _objects(grid))
    if seen_holes != sorted(hole_counts):
        return None
    out = main(grid)
    if out.ndim != 2 or max(out.shape) > 30:
        return None
    return grid


def generate_input() -> np.ndarray:
    rng = random.Random()
    for _ in range(300):
        grid = _gen_attempt(rng)
        if grid is not None:
            return grid
    raise RuntimeError("generate_input failed after 300 attempts")
