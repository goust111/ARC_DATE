from collections import Counter, deque
import random

import numpy as np


ARC_MAX = 30
GREY = 5
MAROON = 9
GEN_RETRIES = 400


def _neighbors4(r: int, c: int):
    yield r - 1, c
    yield r + 1, c
    yield r, c - 1
    yield r, c + 1


def _most_common_color(g: np.ndarray) -> int:
    return int(Counter(map(int, g.ravel())).most_common(1)[0][0])


def _components_mask(mask: np.ndarray) -> list[np.ndarray]:
    h, w = map(int, mask.shape)
    seen = np.zeros((h, w), dtype=bool)
    comps: list[np.ndarray] = []
    for r in range(h):
        for c in range(w):
            if seen[r, c] or not bool(mask[r, c]):
                continue
            q = deque([(r, c)])
            seen[r, c] = True
            pts: list[tuple[int, int]] = []
            while q:
                rr, cc = q.popleft()
                pts.append((int(rr), int(cc)))
                for nr, nc in _neighbors4(rr, cc):
                    if 0 <= nr < h and 0 <= nc < w and not seen[nr, nc] and bool(mask[nr, nc]):
                        seen[nr, nc] = True
                        q.append((int(nr), int(nc)))
            comps.append(np.array(pts, dtype=int))
    return comps


def _bbox(coords: np.ndarray) -> tuple[int, int, int, int]:
    r0, c0 = coords.min(axis=0)
    r1, c1 = coords.max(axis=0)
    return int(r0), int(c0), int(r1), int(c1)


def _trim_mask(mask: np.ndarray) -> np.ndarray:
    pts = np.argwhere(mask)
    r0, c0, r1, c1 = _bbox(pts)
    return mask[r0 : r1 + 1, c0 : c1 + 1].copy()


def _hole_count(mask: np.ndarray) -> int:
    m = np.pad(np.asarray(mask, dtype=bool), 1, constant_values=False)
    h, w = map(int, m.shape)
    bg = ~m
    seen = np.zeros((h, w), dtype=bool)

    q = deque([(0, 0)])
    seen[0, 0] = True
    while q:
        r, c = q.popleft()
        for nr, nc in _neighbors4(r, c):
            if 0 <= nr < h and 0 <= nc < w and bg[nr, nc] and not seen[nr, nc]:
                seen[nr, nc] = True
                q.append((int(nr), int(nc)))

    holes = 0
    for r in range(h):
        for c in range(w):
            if not bg[r, c] or seen[r, c]:
                continue
            holes += 1
            q = deque([(r, c)])
            seen[r, c] = True
            while q:
                rr, cc = q.popleft()
                for nr, nc in _neighbors4(rr, cc):
                    if 0 <= nr < h and 0 <= nc < w and bg[nr, nc] and not seen[nr, nc]:
                        seen[nr, nc] = True
                        q.append((int(nr), int(nc)))
    return int(holes)


def _find_separator(g: np.ndarray, bg: int) -> tuple[int, int]:
    candidates: list[tuple[int, int]] = []
    for r in range(int(g.shape[0])):
        vals = np.unique(g[r])
        if vals.size == 1 and int(vals[0]) != int(bg):
            candidates.append((int(r), int(vals[0])))
    assert len(candidates) == 1, "expected one full non-background separator row"
    return candidates[0]


def _object_record(g: np.ndarray, coords: np.ndarray) -> dict:
    r0, c0, r1, c1 = _bbox(coords)
    h, w = int(r1 - r0 + 1), int(c1 - c0 + 1)
    mask = np.zeros((h, w), dtype=bool)
    rel: list[tuple[int, int, int]] = []
    for r, c in coords:
        rr, cc = int(r - r0), int(c - c0)
        mask[rr, cc] = True
        rel.append((rr, cc, int(g[int(r), int(c)])))
    center = (h // 2, w // 2)
    assert mask[center], "object center must be an object pixel"
    return {
        "bbox": (r0, c0, r1, c1),
        "shape": (h, w),
        "center": center,
        "rel": rel,
        "holes": _hole_count(mask),
    }


def _anchors_below(g: np.ndarray, sep_r: int, bg: int, n: int) -> list[tuple[int, int]]:
    coords: list[tuple[int, int]] = []
    colors: set[int] = set()
    for r in range(int(sep_r) + 1, int(g.shape[0])):
        for c in range(int(g.shape[1])):
            v = int(g[r, c])
            if v != int(bg):
                coords.append((int(r), int(c)))
                colors.add(v)

    assert len(coords) == n, "anchor count must match object count"
    assert len(colors) == 1, "anchors must share one color"
    assert len({r for r, _c in coords}) == 1, "anchors must lie on one row"
    coords = sorted(coords, key=lambda p: p[1])
    if len(coords) >= 3:
        gaps = [coords[i + 1][1] - coords[i][1] for i in range(len(coords) - 1)]
        assert len(set(gaps)) == 1, "anchors must be evenly spaced"
    return coords


def main(grid: np.ndarray) -> np.ndarray:
    """
    Sort hollow objects by their internal hole count and move them onto lower anchors.

    Input structure:
    - The most common color is the background.
    - A complete horizontal non-background row separates the upper object area
      from the lower anchor area.
    - The upper area contains several separated 4-connected colored sprites;
      each sprite has a unique number of enclosed background holes.
    - The lower area contains the same number of single-pixel anchors, all on
      one row and evenly spaced from left to right.

    Rule:
    - Count the enclosed background components inside each upper sprite.
    - Sort the sprites by that hole count from fewest to most.
    - Clear the upper area and the anchor area to background while preserving
      the separator row.
    - Paste the sorted sprites unchanged so each sprite's bounding-box center is
      aligned to the corresponding lower anchor from left to right.

    Key insight:
    The visible object color and outside silhouette are distractors; the hidden
    ordering key is the number of fully enclosed background holes inside each
    sprite.
    """
    g = np.asarray(grid, dtype=int)
    assert g.ndim == 2, "expected a 2D grid"
    h, w = map(int, g.shape)
    bg = _most_common_color(g)
    sep_r, _sep_color = _find_separator(g, bg)

    top_mask = g[:sep_r, :] != int(bg)
    comps = _components_mask(top_mask)
    assert comps, "expected upper objects"
    objects = [_object_record(g, coords) for coords in comps]
    hole_values = [int(obj["holes"]) for obj in objects]
    assert len(set(hole_values)) == len(hole_values), "object hole counts must be unique"

    anchors = _anchors_below(g, sep_r, bg, len(objects))
    ordered = sorted(objects, key=lambda obj: int(obj["holes"]))

    out = np.full((h, w), int(bg), dtype=int)
    out[sep_r, :] = g[sep_r, :]

    for obj, (ar, ac) in zip(ordered, anchors):
        cr, cc = map(int, obj["center"])
        top = int(ar - cr)
        left = int(ac - cc)
        oh, ow = map(int, obj["shape"])
        assert int(sep_r) < top and top + oh <= h, "placed object must fit below separator"
        assert 0 <= left and left + ow <= w, "placed object must fit horizontally"
        for rr, cc2, color in obj["rel"]:
            tr, tc = int(top + rr), int(left + cc2)
            assert int(out[tr, tc]) == int(bg), "placed objects must not overlap"
            out[tr, tc] = int(color)

    return out


def _add_ring(mask: np.ndarray, r: int, c: int) -> None:
    for dr in (-1, 0, 1):
        for dc in (-1, 0, 1):
            if dr == 0 and dc == 0:
                continue
            mask[int(r + dr), int(c + dc)] = True


def _connected(mask: np.ndarray) -> bool:
    comps = _components_mask(mask)
    return len(comps) == 1


def _base_zero_hole_shape() -> np.ndarray:
    for _ in range(100):
        h = random.choice([3, 5])
        w = random.choice([3, 5])
        m = np.zeros((h, w), dtype=bool)
        cr, cc = h // 2, w // 2
        m[cr, cc] = True
        target = random.randint(5, max(5, min(h * w - 2, 12)))
        frontier = {(cr, cc)}
        while int(m.sum()) < target:
            candidates = []
            for r, c in frontier:
                for nr, nc in _neighbors4(r, c):
                    if 0 <= nr < h and 0 <= nc < w and not m[nr, nc]:
                        candidates.append((int(nr), int(nc)))
            if not candidates:
                break
            r, c = random.choice(candidates)
            m[r, c] = True
            frontier.add((int(r), int(c)))

        m = _trim_mask(m)
        if (
            int(m.shape[0]) % 2 == 1
            and int(m.shape[1]) % 2 == 1
            and bool(m[int(m.shape[0]) // 2, int(m.shape[1]) // 2])
            and _hole_count(m) == 0
            and _connected(m)
        ):
            return m
    return np.array([[False, True, False], [True, True, True], [False, True, False]], dtype=bool)


def _base_holed_shape(holes: int) -> np.ndarray:
    assert holes >= 1
    if holes == 1:
        m = np.zeros((3, 5), dtype=bool)
        _add_ring(m, 1, 1)
        m[1, 3:5] = True
        return m
    if holes == 2:
        m = np.zeros((3, 7), dtype=bool)
        _add_ring(m, 1, 1)
        _add_ring(m, 1, 5)
        m[1, 3] = True
        return m

    m = np.zeros((7, 7), dtype=bool)
    centers = [(1, 1), (1, 5), (5, 1), (5, 5)]
    if holes == 3:
        centers.remove(random.choice(centers))
    for r, c in centers:
        _add_ring(m, r, c)
    m[3, :] = True
    m[:, 3] = True
    return m


def _mutate_shape(mask: np.ndarray, holes: int) -> np.ndarray:
    m = mask.copy()
    for _ in range(random.randint(0, 4)):
        candidates: list[tuple[int, int]] = []
        for r in range(int(m.shape[0])):
            for c in range(int(m.shape[1])):
                if bool(m[r, c]):
                    continue
                if any(
                    0 <= nr < int(m.shape[0]) and 0 <= nc < int(m.shape[1]) and bool(m[nr, nc])
                    for nr, nc in _neighbors4(r, c)
                ):
                    candidates.append((int(r), int(c)))
        random.shuffle(candidates)
        for r, c in candidates:
            trial = m.copy()
            trial[r, c] = True
            if _hole_count(trial) == holes and _connected(trial):
                m = trial
                break
    return m


def _shape_for_holes(holes: int) -> np.ndarray:
    for _ in range(100):
        if holes == 0:
            m = _base_zero_hole_shape()
        else:
            m = _base_holed_shape(holes)

        k = random.randrange(4)
        m = np.rot90(m, k)
        if random.random() < 0.5:
            m = np.fliplr(m)
        if random.random() < 0.5:
            m = np.flipud(m)
        m = _mutate_shape(np.asarray(m, dtype=bool), int(holes))
        m = _trim_mask(m)
        cr, cc = int(m.shape[0]) // 2, int(m.shape[1]) // 2
        if (
            int(m.shape[0]) % 2 == 1
            and int(m.shape[1]) % 2 == 1
            and bool(m[cr, cc])
            and _hole_count(m) == int(holes)
            and _connected(m)
        ):
            return m
    raise AssertionError("could not build requested shape")


def _bottom_layout(items: list[dict]) -> tuple[int, int, int, list[int]] | None:
    ordered = sorted(items, key=lambda item: int(item["holes"]))
    left_ext = [int(item["mask"].shape[1]) // 2 for item in ordered]
    right_ext = [int(item["mask"].shape[1]) - 1 - left_ext[i] for i, item in enumerate(ordered)]
    top_ext = [int(item["mask"].shape[0]) // 2 for item in ordered]
    bottom_ext = [int(item["mask"].shape[0]) - 1 - top_ext[i] for i, item in enumerate(ordered)]

    spacing = 2
    for i in range(len(ordered) - 1):
        spacing = max(spacing, int(right_ext[i] + left_ext[i + 1] + 2))
    spacing += random.randint(0, 2)

    min_w = int(left_ext[0] + spacing * (len(ordered) - 1) + right_ext[-1] + 2)
    if min_w > ARC_MAX:
        return None
    width = random.randint(min_w, ARC_MAX)
    extra = int(width - min_w)
    first_anchor = int(1 + left_ext[0] + random.randint(0, extra))
    anchors = [int(first_anchor + spacing * i) for i in range(len(ordered))]

    min_lower_h = int(max(top_ext) + max(bottom_ext) + 3)
    if min_lower_h > 12:
        return None
    lower_h = random.randint(min_lower_h, min(13, ARC_MAX - 9))
    min_anchor = int(max(top_ext) + 1)
    max_anchor = int(lower_h - max(bottom_ext) - 2)
    if min_anchor > max_anchor:
        return None
    anchor_rel_r = random.randint(min_anchor, max_anchor)
    return width, lower_h, anchor_rel_r, anchors


def _can_place(mask: np.ndarray, occupied: np.ndarray, r0: int, c0: int) -> bool:
    h, w = map(int, mask.shape)
    oh, ow = map(int, occupied.shape)
    if r0 < 0 or c0 < 0 or r0 + h > oh or c0 + w > ow:
        return False
    for rr, cc in np.argwhere(mask):
        r, c = int(r0 + rr), int(c0 + cc)
        for dr in (-1, 0, 1):
            for dc in (-1, 0, 1):
                nr, nc = int(r + dr), int(c + dc)
                if 0 <= nr < oh and 0 <= nc < ow and bool(occupied[nr, nc]):
                    return False
    return True


def _place_upper_objects(grid: np.ndarray, items: list[dict], top_h: int, width: int) -> bool:
    occupied = np.zeros((top_h, width), dtype=bool)
    order = items[:]
    random.shuffle(order)
    for item in order:
        mask = item["mask"]
        mh, mw = map(int, mask.shape)
        if mh > top_h or mw > width:
            return False
        placed = False
        for _ in range(200):
            r0 = random.randint(0, top_h - mh)
            c0 = random.randint(0, width - mw)
            if not _can_place(mask, occupied, r0, c0):
                continue
            pts = np.argwhere(mask)
            for rr, cc in pts:
                r, c = int(r0 + rr), int(c0 + cc)
                grid[r, c] = int(item["color"])
                occupied[r, c] = True
            placed = True
            break
        if not placed:
            return False
    return True


def _gen_attempt() -> np.ndarray | None:
    bg = random.choice([0, 1, 2, 3, 7])
    count = random.choice([3, 4, 5])
    hole_counts = [0] + random.sample([1, 2, 3, 4], count - 1)
    items = [{"holes": int(k), "mask": _shape_for_holes(int(k))} for k in hole_counts]

    colors = [c for c in range(10) if c not in {int(bg), GREY, MAROON}]
    if len(colors) < count:
        return None
    for item, color in zip(items, random.sample(colors, count)):
        item["color"] = int(color)

    layout = _bottom_layout(items)
    if layout is None:
        return None
    width, lower_h, anchor_rel_r, anchor_cols = layout

    max_obj_h = max(int(item["mask"].shape[0]) for item in items)
    top_min = max(8, int(max_obj_h + 2))
    top_max = int(ARC_MAX - lower_h - 1)
    if top_min > top_max:
        return None
    top_h = random.randint(top_min, top_max)
    height = int(top_h + 1 + lower_h)
    if height > ARC_MAX or width > ARC_MAX:
        return None

    grid = np.full((height, width), int(bg), dtype=int)
    grid[top_h, :] = GREY
    anchor_r = int(top_h + 1 + anchor_rel_r)
    for c in anchor_cols:
        grid[anchor_r, int(c)] = MAROON

    if not _place_upper_objects(grid, items, top_h, width):
        return None

    try:
        out = main(grid)
    except AssertionError:
        return None
    if out.ndim != 2 or max(out.shape) > ARC_MAX:
        return None
    return grid


def generate_input() -> np.ndarray:
    for _ in range(GEN_RETRIES):
        grid = _gen_attempt()
        if grid is not None:
            return grid
    raise RuntimeError("failed to generate a valid input")
