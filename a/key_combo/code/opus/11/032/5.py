from __future__ import annotations

import random

import numpy as np


BG = 0
BASE = 5
MARKER_COLORS = (1, 2, 3, 4)
GEN_ATTEMPTS = 500


def _neighbors4(r: int, c: int):
    yield r - 1, c
    yield r + 1, c
    yield r, c - 1
    yield r, c + 1


def _components4(mask: np.ndarray) -> list[np.ndarray]:
    h, w = mask.shape
    seen = np.zeros((h, w), dtype=bool)
    out: list[np.ndarray] = []
    for r in range(h):
        for c in range(w):
            if not mask[r, c] or seen[r, c]:
                continue
            stack = [(r, c)]
            seen[r, c] = True
            coords = []
            while stack:
                rr, cc = stack.pop()
                coords.append((rr, cc))
                for nr, nc in _neighbors4(rr, cc):
                    if 0 <= nr < h and 0 <= nc < w and mask[nr, nc] and not seen[nr, nc]:
                        seen[nr, nc] = True
                        stack.append((nr, nc))
            out.append(np.array(coords, dtype=int))
    return out


def _bbox(coords: np.ndarray) -> tuple[int, int, int, int]:
    r0, c0 = coords.min(axis=0)
    r1, c1 = coords.max(axis=0)
    return int(r0), int(c0), int(r1), int(c1)


def _extract_rectangles(grid: np.ndarray) -> list[dict]:
    g = np.asarray(grid, dtype=int)
    comps = _components4(g != BG)
    assert 3 <= len(comps) <= 4, "expected 3-4 grey-based rectangles"

    rects = []
    for coords in comps:
        r0, c0, r1, c1 = _bbox(coords)
        patch = g[r0 : r1 + 1, c0 : c1 + 1].copy()
        assert np.all(patch != BG), "rectangle bbox must be fully filled"
        assert np.any(patch == BASE), "each rectangle needs a grey base"
        marker_mask = patch != BASE
        marker_count = int(marker_mask.sum())
        assert marker_count >= 1
        marker_colors = {int(v) for v in patch[marker_mask]}
        assert marker_colors <= set(MARKER_COLORS), "unexpected marker color"
        rects.append(
            {
                "top_left": (int(r0), int(c0)),
                "dist": int(r0 + c0),
                "patch": patch,
                "marker_count": marker_count,
            }
        )

    shapes = {tuple(rect["patch"].shape) for rect in rects}
    assert len(shapes) == 1, "all rectangles must have the same size"
    return rects


def _compose(rects: list[dict], *, carve_conflicts: bool = True, any_overlap_conflict: bool = False) -> np.ndarray:
    h, w = rects[0]["patch"].shape
    out = np.full((h, w), BASE, dtype=int)
    history: list[list[set[int]]] = [[set() for _ in range(w)] for _ in range(h)]
    writes = np.zeros((h, w), dtype=int)
    order = sorted(rects, key=lambda rect: (rect["marker_count"], rect["dist"]))

    for rect in order:
        patch = rect["patch"]
        for r, c in np.argwhere(patch != BASE):
            rr, cc = int(r), int(c)
            color = int(patch[rr, cc])
            out[rr, cc] = color
            history[rr][cc].add(color)
            writes[rr, cc] += 1

    if carve_conflicts:
        for r in range(h):
            for c in range(w):
                if any_overlap_conflict:
                    if writes[r, c] >= 2:
                        out[r, c] = BG
                elif len(history[r][c]) >= 2:
                    out[r, c] = BG
    return out


def main(grid: np.ndarray) -> np.ndarray:
    """
    Overlay same-sized grey rectangles after sorting them by marker count, then punch black holes at multi-color conflicts.

    Input structure:
    - The grid has a black background and 3-4 separated filled rectangles.
    - Every rectangle has the same height and width, with grey as its base color.
    - Each rectangle contains a small set of non-grey marker pixels chosen from blue, red, green, and yellow.

    Rule:
    - Detect every filled rectangle and count its non-grey marker pixels.
    - Sort rectangles by marker count ascending; equal counts are ordered by the Manhattan distance from the rectangle's top-left corner to the grid origin.
    - Start a grey output canvas of the shared rectangle size, then overlay rectangles in that order, treating grey as transparent and marker pixels as opaque.
    - For each output position, record every distinct marker color that ever writes there; after all overlays, any position written by two or more different colors becomes black.

    Key insight:
    The separated rectangles are aligned layers of one template, so the task is to merge them by relative coordinates while distinguishing harmless same-color overlaps from true multi-color conflicts.
    """
    rects = _extract_rectangles(np.asarray(grid, dtype=int))
    return _compose(rects)


def _sample_counts(rect_count: int) -> list[int]:
    if rect_count == 3:
        while True:
            counts = random.sample([2, 3, 4], 3)
            if counts != [2, 3, 4]:
                return counts
    patterns = [
        [4, 4, 2, 3],
        [2, 4, 4, 3],
        [3, 2, 4, 4],
        [2, 2, 3, 4],
        [4, 2, 2, 3],
    ]
    return list(random.choice(patterns))


def _build_marker_maps(rect_count: int, counts: list[int], h: int, w: int) -> list[dict[tuple[int, int], int]] | None:
    coords = [(r, c) for r in range(h) for c in range(w)]
    for _ in range(250):
        maps: list[dict[tuple[int, int], int]] = [dict() for _ in range(rect_count)]
        conflict_positions = random.sample(coords, 2)
        same_overlap = random.choice([p for p in coords if p not in conflict_positions])

        participants = random.sample(range(rect_count), random.randint(2, min(3, rect_count)))
        same_color = random.choice(MARKER_COLORS)
        for idx in participants:
            maps[idx][same_overlap] = same_color

        ok = True
        for pos in conflict_positions:
            participants = random.sample(range(rect_count), random.randint(2, min(3, rect_count)))
            picked = list(random.sample(MARKER_COLORS, 2))
            while len(picked) < len(participants):
                picked.append(random.choice(MARKER_COLORS))
            random.shuffle(picked)
            for idx, color in zip(participants, picked):
                if pos in maps[idx]:
                    ok = False
                    break
                maps[idx][pos] = int(color)
            if not ok:
                break
        if not ok:
            continue

        for idx in range(rect_count):
            if not maps[idx]:
                pos = random.choice(coords)
                maps[idx][pos] = random.choice(MARKER_COLORS)

        for idx in range(rect_count):
            while len(maps[idx]) < counts[idx]:
                overlap_candidates = [
                    pos
                    for pos in coords
                    if pos not in maps[idx] and any(pos in maps[j] for j in range(rect_count) if j != idx)
                ]
                fresh_candidates = [
                    pos
                    for pos in coords
                    if pos not in maps[idx] and all(pos not in maps[j] for j in range(rect_count) if j != idx)
                ]
                if overlap_candidates and random.random() < 0.55:
                    pos = random.choice(overlap_candidates)
                    existing = [maps[j][pos] for j in range(rect_count) if j != idx and pos in maps[j]]
                    if random.random() < 0.35:
                        color = random.choice(existing)
                    else:
                        diff = [c for c in MARKER_COLORS if c not in set(existing)]
                        color = random.choice(diff or MARKER_COLORS)
                else:
                    pool = fresh_candidates or [pos for pos in coords if pos not in maps[idx]]
                    if not pool:
                        ok = False
                        break
                    pos = random.choice(pool)
                    color = random.choice(MARKER_COLORS)
                maps[idx][pos] = int(color)
            if not ok:
                break
        if not ok:
            continue

        pos_to_colors: dict[tuple[int, int], list[int]] = {}
        for mapping in maps:
            for pos, color in mapping.items():
                pos_to_colors.setdefault(pos, []).append(int(color))

        multi_color = [pos for pos, colors in pos_to_colors.items() if len(set(colors)) >= 2]
        same_color_overlap_positions = [
            pos for pos, colors in pos_to_colors.items() if len(colors) >= 2 and len(set(colors)) == 1
        ]
        if len(multi_color) < 2:
            continue
        if not same_color_overlap_positions:
            continue
        if len(multi_color) == len(pos_to_colors):
            continue
        return maps
    return None


def _make_patch(h: int, w: int, mapping: dict[tuple[int, int], int]) -> np.ndarray:
    patch = np.full((h, w), BASE, dtype=int)
    for (r, c), color in mapping.items():
        patch[int(r), int(c)] = int(color)
    return patch


def _build_layout(rect_count: int, h: int, w: int, counts: list[int]) -> tuple[list[int], list[int], int, int] | None:
    top_margin = random.randint(1, 2)
    bottom_margin = random.randint(1, 2)
    gaps = [random.randint(1, 3) for _ in range(rect_count - 1)]
    ys = [top_margin]
    for gap in gaps:
        ys.append(ys[-1] + h + gap)
    grid_h = ys[-1] + h + bottom_margin
    if grid_h > 30:
        return None

    base_width = random.randint(max(12, w + 6), min(22, 30))
    xs = [random.randint(1, base_width - w - 1) for _ in range(rect_count)]

    duplicates = {}
    for idx, count in enumerate(counts):
        duplicates.setdefault(count, []).append(idx)
    tie_groups = [group for group in duplicates.values() if len(group) >= 2]
    if tie_groups:
        group = min(tie_groups, key=lambda g: max(g) - min(g))
        upper, lower = sorted(group[:2])
        xs[upper] = base_width - w - 1
        xs[lower] = 1
        if ys[upper] + xs[upper] <= ys[lower] + xs[lower]:
            return None

    return ys, xs, grid_h, base_width


def _conflict_stats(maps: list[dict[tuple[int, int], int]]) -> tuple[int, int]:
    pos_to_colors: dict[tuple[int, int], list[int]] = {}
    for mapping in maps:
        for pos, color in mapping.items():
            pos_to_colors.setdefault(pos, []).append(int(color))
    multi_color = sum(1 for colors in pos_to_colors.values() if len(set(colors)) >= 2)
    same_color_overlap = sum(1 for colors in pos_to_colors.values() if len(colors) >= 2 and len(set(colors)) == 1)
    return multi_color, same_color_overlap


def _validate_generated(grid: np.ndarray, counts: list[int], ys: list[int], xs: list[int], maps: list[dict[tuple[int, int], int]]) -> bool:
    try:
        rects = _extract_rectangles(grid)
        expected = _compose(rects)
        actual = main(grid)
    except Exception:
        return False

    if not np.array_equal(actual, expected):
        return False
    if not np.any(expected == BG):
        return False
    if not np.any(np.isin(expected, MARKER_COLORS)):
        return False
    if any(np.array_equal(expected, rect["patch"]) for rect in rects):
        return False

    no_holes = _compose(rects, carve_conflicts=False)
    if np.array_equal(no_holes, expected):
        return False
    any_overlap_black = _compose(rects, carve_conflicts=True, any_overlap_conflict=True)
    if np.array_equal(any_overlap_black, expected):
        return False

    multi_color, same_color_overlap = _conflict_stats(maps)
    if multi_color < 2 or same_color_overlap < 1:
        return False

    if len(counts) == 4:
        duplicate_count = next(count for count in set(counts) if counts.count(count) >= 2)
        dup_ids = [i for i, count in enumerate(counts) if count == duplicate_count][:2]
        upper, lower = sorted(dup_ids)
        if not (ys[upper] < ys[lower] and ys[upper] + xs[upper] > ys[lower] + xs[lower]):
            return False

    return True


def _gen_attempt() -> np.ndarray | None:
    rect_count = 3 if random.random() < 0.58 else 4
    h = random.randint(4, 6)
    w = random.randint(4, 6)
    counts = _sample_counts(rect_count)

    marker_maps = _build_marker_maps(rect_count, counts, h, w)
    if marker_maps is None:
        return None

    layout = _build_layout(rect_count, h, w, counts)
    if layout is None:
        return None
    ys, xs, grid_h, grid_w = layout

    grid = np.full((grid_h, grid_w), BG, dtype=int)
    for idx in range(rect_count):
        patch = _make_patch(h, w, marker_maps[idx])
        r0, c0 = ys[idx], xs[idx]
        grid[r0 : r0 + h, c0 : c0 + w] = patch

    if not _validate_generated(grid, counts, ys, xs, marker_maps):
        return None
    return grid


def generate_input() -> np.ndarray:
    for _ in range(GEN_ATTEMPTS):
        grid = _gen_attempt()
        if grid is not None:
            return grid
    raise RuntimeError("failed to generate a valid input for puzzle 5")
