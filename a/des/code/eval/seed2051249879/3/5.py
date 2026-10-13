from __future__ import annotations

from collections import Counter
import random

import numpy as np


GREEN = 3
RED = 2
BLUE = 1
YELLOW = 4

BG_CHOICES = (0, 5, 6, 7, 8, 9)
BBOX_MIN, BBOX_MAX = 4, 7
OBJECT_COUNT_MIN, OBJECT_COUNT_MAX = 3, 5
GRID_H_MIN, GRID_H_MAX = 18, 24
WIDTH_SLACK_MIN, WIDTH_SLACK_MAX = 4, 12
BOTTOM_CLEARANCE = 4
GEN_ATTEMPTS = 400

_NEIGHBORS4 = ((1, 0), (-1, 0), (0, 1), (0, -1))
_NEIGHBORS8 = (
    (1, 0),
    (-1, 0),
    (0, 1),
    (0, -1),
    (1, 1),
    (1, -1),
    (-1, 1),
    (-1, -1),
)


def _get_bg(grid: np.ndarray) -> int:
    return int(Counter(map(int, grid.ravel())).most_common(1)[0][0])


def _components(mask: np.ndarray, connectivity: int = 8) -> list[np.ndarray]:
    h, w = mask.shape
    seen = np.zeros((h, w), dtype=bool)
    neighbors = _NEIGHBORS8 if connectivity == 8 else _NEIGHBORS4
    comps: list[np.ndarray] = []

    for r in range(h):
        for c in range(w):
            if not mask[r, c] or seen[r, c]:
                continue
            stack = [(r, c)]
            seen[r, c] = True
            pts: list[tuple[int, int]] = []
            while stack:
                cr, cc = stack.pop()
                pts.append((cr, cc))
                for dr, dc in neighbors:
                    nr, nc = cr + dr, cc + dc
                    if 0 <= nr < h and 0 <= nc < w and mask[nr, nc] and not seen[nr, nc]:
                        seen[nr, nc] = True
                        stack.append((nr, nc))
            comps.append(np.array(pts, dtype=int))
    return comps


def _bbox_from_coords(coords: np.ndarray) -> tuple[int, int, int, int]:
    rows = coords[:, 0]
    cols = coords[:, 1]
    return int(rows.min()), int(cols.min()), int(rows.max()), int(cols.max())


def _mask_from_coords(coords: np.ndarray) -> tuple[np.ndarray, tuple[int, int, int, int]]:
    r0, c0, r1, c1 = _bbox_from_coords(coords)
    mask = np.zeros((r1 - r0 + 1, c1 - c0 + 1), dtype=bool)
    mask[coords[:, 0] - r0, coords[:, 1] - c0] = True
    return mask, (r0, c0, r1, c1)


def _has_rot180(mask: np.ndarray) -> bool:
    return bool(np.array_equal(mask, mask[::-1, ::-1]))


def _has_h_mirror(mask: np.ndarray) -> bool:
    return bool(np.array_equal(mask, mask[::-1, :]))


def _has_v_mirror(mask: np.ndarray) -> bool:
    return bool(np.array_equal(mask, mask[:, ::-1]))


def _classify_symmetry(mask: np.ndarray) -> str:
    if _has_rot180(mask):
        return "rot"
    if _has_h_mirror(mask) or _has_v_mirror(mask):
        return "mirror"
    return "none"


def _paint(
    canvas: np.ndarray,
    mask: np.ndarray,
    top: int,
    left: int,
    color: int,
    allowed_under: set[int] | None = None,
) -> None:
    allowed_under = set() if allowed_under is None else allowed_under
    h, w = mask.shape
    sub = canvas[top : top + h, left : left + w]
    assert sub.shape == mask.shape
    covered = {int(v) for v in np.unique(sub[mask])}
    assert covered <= allowed_under | {int(color), _get_bg(canvas)}
    sub[mask] = int(color)


def main(grid: np.ndarray) -> np.ndarray:
    """Classify green objects by symmetry, then recolor, move, or delete them."""
    g = np.asarray(grid, dtype=int)
    assert g.ndim == 2
    h, w = g.shape
    bg = _get_bg(g)

    counts = Counter(map(int, g.ravel()))
    singleton_colors = [color for color, count in counts.items() if color != bg and count == 1]
    assert len(singleton_colors) == 3, "expected exactly three singleton legend markers"

    markers: list[tuple[int, int, int]] = []
    for color in singleton_colors:
        r, c = np.argwhere(g == color)[0]
        markers.append((int(r), int(c), int(color)))
    markers.sort(key=lambda item: (item[1], item[0]))
    rot_color = markers[0][2]
    mirror_color = markers[1][2]
    marker_colors = {color for _r, _c, color in markers}

    object_colors = [color for color in counts if color != bg and color not in marker_colors]
    assert len(object_colors) == 1, "expected a single green object color"
    object_color = int(object_colors[0])

    comps = _components(g == object_color, connectivity=8)
    assert comps, "expected at least one object component"

    out = np.full_like(g, bg)
    for r, c, color in markers:
        out[r, c] = color

    comps.sort(key=lambda pts: _bbox_from_coords(pts))
    for coords in comps:
        mask, (r0, c0, _r1, _c1) = _mask_from_coords(coords)
        kind = _classify_symmetry(mask)
        if kind == "rot":
            _paint(out, mask, r0, c0, rot_color)
            continue
        if kind == "mirror":
            dst_r0 = h - mask.shape[0]
            _paint(out, mask, dst_r0, c0, mirror_color, allowed_under=marker_colors)
            continue

    return out


def _is_connected(mask: np.ndarray) -> bool:
    return len(_components(mask, connectivity=4)) == 1


def _touches_all_sides(mask: np.ndarray) -> bool:
    return bool(mask[0].any() and mask[-1].any() and mask[:, 0].any() and mask[:, -1].any())


def _orbit_cells(h: int, w: int, kind: str) -> list[tuple[tuple[int, int], ...]]:
    seen: set[tuple[int, int]] = set()
    groups: list[tuple[tuple[int, int], ...]] = []
    for r in range(h):
        for c in range(w):
            if (r, c) in seen:
                continue
            if kind == "rot":
                orbit = {(r, c), (h - 1 - r, w - 1 - c)}
            elif kind == "h":
                orbit = {(r, c), (h - 1 - r, c)}
            elif kind == "v":
                orbit = {(r, c), (r, w - 1 - c)}
            else:
                raise ValueError(kind)
            seen.update(orbit)
            groups.append(tuple(sorted(orbit)))
    return groups


def _singletons(h: int, w: int) -> list[tuple[tuple[int, int], ...]]:
    return [((r, c),) for r in range(h) for c in range(w)]


def _carve_full_mask(
    h: int,
    w: int,
    groups: list[tuple[tuple[int, int], ...]],
    keep_ratio: tuple[float, float],
) -> tuple[np.ndarray, int]:
    total = h * w
    lo = max(h + w - 1, int(np.ceil(total * keep_ratio[0])))
    hi = max(lo, int(np.floor(total * keep_ratio[1])))
    target = random.randint(lo, hi)

    mask = np.ones((h, w), dtype=bool)
    removed = 0
    while mask.sum() > target:
        valid: list[tuple[tuple[tuple[int, int], ...], np.ndarray]] = []
        for group in groups:
            if mask.sum() - len(group) < target:
                continue
            if not all(mask[r, c] for r, c in group):
                continue
            trial = mask.copy()
            for r, c in group:
                trial[r, c] = False
            if not _touches_all_sides(trial):
                continue
            if not _is_connected(trial):
                continue
            valid.append((group, trial))
        if not valid:
            break
        group, trial = random.choice(valid)
        mask = trial
        removed += len(group)
    return mask, removed


def _random_transform(mask: np.ndarray) -> np.ndarray:
    op = random.randrange(6)
    if op < 4:
        return np.rot90(mask, op).copy()
    if op == 4:
        return np.fliplr(mask).copy()
    return np.flipud(mask).copy()


def _make_shape(kind: str) -> np.ndarray:
    for _ in range(200):
        h = random.randint(BBOX_MIN, BBOX_MAX)
        w = random.randint(BBOX_MIN, BBOX_MAX)
        if kind == "rot":
            groups = _orbit_cells(h, w, "rot")
            mask, removed = _carve_full_mask(h, w, groups, (0.42, 0.76))
            need_removed = 1
        elif kind == "mirror":
            groups = _orbit_cells(h, w, random.choice(("h", "v")))
            mask, removed = _carve_full_mask(h, w, groups, (0.38, 0.72))
            need_removed = 2
        elif kind == "none":
            groups = _singletons(h, w)
            mask, removed = _carve_full_mask(h, w, groups, (0.38, 0.70))
            need_removed = 3
        else:
            raise ValueError(kind)

        if removed < need_removed:
            continue
        mask = _random_transform(mask)
        if _classify_symmetry(mask) == kind:
            return mask
    raise RuntimeError(f"failed to generate {kind} shape")


def _balanced_classes(n: int) -> list[str]:
    order = ["rot", "mirror", "none"]
    counts = {kind: 1 for kind in order}
    classes = order.copy()
    while len(classes) < n:
        min_count = min(counts.values())
        choices = [kind for kind, count in counts.items() if count == min_count]
        chosen = random.choice(choices)
        classes.append(chosen)
        counts[chosen] += 1
    random.shuffle(classes)
    return classes


def _layout_columns(widths: list[int]) -> tuple[int, list[int]]:
    gaps = [1] * (len(widths) + 1)
    slack = random.randint(WIDTH_SLACK_MIN, WIDTH_SLACK_MAX)
    for _ in range(slack):
        gaps[random.randrange(len(gaps))] += 1

    cols: list[int] = []
    cur = gaps[0]
    for idx, width in enumerate(widths):
        cols.append(cur)
        cur += width + gaps[idx + 1]
    total_width = sum(widths) + sum(gaps)
    assert cur == total_width
    return total_width, cols


def _choose_marker_cols(width: int, cover_candidates: list[int]) -> list[int]:
    required: list[int] = []
    if cover_candidates and random.random() < 0.6:
        want = 1 + int(len(cover_candidates) > 1 and random.random() < 0.35)
        random.shuffle(cover_candidates)
        for col in cover_candidates:
            if all(abs(col - prev) >= 2 for prev in required):
                required.append(col)
            if len(required) == want:
                break

    base_cols = list(range(1, width - 1)) if width >= 5 else list(range(width))
    for _ in range(300):
        picked = required.copy()
        pool = [c for c in base_cols if c not in picked]
        picked.extend(random.sample(pool, 3 - len(picked)))
        picked.sort()
        if picked[1] - picked[0] >= 2 and picked[2] - picked[1] >= 2:
            return picked
    raise RuntimeError("failed to place markers")


def _gen_attempt() -> np.ndarray | None:
    bg = random.choice(BG_CHOICES)
    n_objects = random.randint(OBJECT_COUNT_MIN, OBJECT_COUNT_MAX)
    classes = _balanced_classes(n_objects)
    masks = [_make_shape(kind) for kind in classes]

    widths = [mask.shape[1] for mask in masks]
    heights = [mask.shape[0] for mask in masks]
    width, cols = _layout_columns(widths)

    min_h = max(GRID_H_MIN, max(heights) + BOTTOM_CLEARANCE + 1)
    max_h = max(min_h, GRID_H_MAX)
    height = random.randint(min_h, max_h)

    tops = [random.randint(1, height - obj_h - BOTTOM_CLEARANCE) for obj_h in heights]
    grid = np.full((height, width), bg, dtype=int)

    moving_bottom_cols: list[int] = []
    for kind, mask, top, left in zip(classes, masks, tops, cols):
        h, w = mask.shape
        sub = grid[top : top + h, left : left + w]
        if np.any(sub[mask] != bg):
            return None
        sub[mask] = GREEN
        if kind == "mirror":
            moving_bottom_cols.extend(left + int(c) for c in np.flatnonzero(mask[-1]))

    marker_cols = _choose_marker_cols(width, sorted(set(moving_bottom_cols)))
    marker_row = height - 1
    for col, color in zip(marker_cols, (RED, BLUE, YELLOW)):
        if grid[marker_row, col] != bg:
            return None
        grid[marker_row, col] = color

    if _get_bg(grid) != bg:
        return None

    try:
        out = main(grid)
    except AssertionError:
        return None

    if np.any(out == GREEN):
        return None
    if int(np.count_nonzero(out == RED)) <= 1:
        return None
    if int(np.count_nonzero(out == BLUE)) <= 1:
        return None
    if len(_components(grid == GREEN, connectivity=8)) != n_objects:
        return None
    return grid


def generate_input() -> np.ndarray:
    for _ in range(GEN_ATTEMPTS):
        grid = _gen_attempt()
        if grid is not None:
            return grid
    raise RuntimeError("failed to generate valid input")
