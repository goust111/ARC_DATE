from __future__ import annotations

import random
from collections import deque

import numpy as np


BG = 0
RECT_COLOR = 1
CHAIN_RED = 2
OUTPUT_RECT = 3
MARKER = 9


Coord = tuple[int, int]
BBox = tuple[int, int, int, int]


def _bbox(coords: np.ndarray) -> BBox:
    r0, c0 = coords.min(axis=0)
    r1, c1 = coords.max(axis=0)
    return int(r0), int(c0), int(r1), int(c1)


def _components(mask: np.ndarray) -> list[np.ndarray]:
    h, w = mask.shape
    seen = np.zeros_like(mask, dtype=bool)
    comps: list[np.ndarray] = []
    for r in range(h):
        for c in range(w):
            if not mask[r, c] or seen[r, c]:
                continue
            q = deque([(r, c)])
            seen[r, c] = True
            pts: list[Coord] = []
            while q:
                rr, cc = q.popleft()
                pts.append((rr, cc))
                for nr, nc in ((rr - 1, cc), (rr + 1, cc), (rr, cc - 1), (rr, cc + 1)):
                    if 0 <= nr < h and 0 <= nc < w and mask[nr, nc] and not seen[nr, nc]:
                        seen[nr, nc] = True
                        q.append((nr, nc))
            comps.append(np.array(pts, dtype=int))
    return comps


def _find_rectangle_and_chain(
    grid: np.ndarray,
) -> tuple[BBox, dict[int, tuple[int, int]], tuple[int, int, int]]:
    blue = _components(grid == RECT_COLOR)
    rects = []
    starts = []
    for comp in blue:
        box = _bbox(comp)
        r0, c0, r1, c1 = box
        h = r1 - r0 + 1
        w = c1 - c0 + 1
        if int(comp.shape[0]) != h * w:
            continue
        if h >= 2 and w >= 2:
            rects.append(box)
        elif w == 1:
            starts.append((c0, h, box))
    assert len(rects) == 1
    assert len(starts) == 1
    rect = rects[0]

    reds = _components(grid == CHAIN_RED)
    red_bars: dict[int, tuple[int, int]] = {}
    for comp in reds:
        box = _bbox(comp)
        r0, c0, r1, c1 = box
        h = r1 - r0 + 1
        w = c1 - c0 + 1
        assert w == 1 and int(comp.shape[0]) == h
        red_bars[c0] = (h, r1)

    start_col, start_h, start_box = starts[0]
    _sr0, _sc0, start_bottom, _sc1 = start_box
    return rect, red_bars, (start_col, start_h, start_bottom)


def _chain_length(grid: np.ndarray) -> int:
    _rect, red_bars, start = _find_rectangle_and_chain(grid)
    col, height, bottom = start
    length = 1
    while True:
        next_col = col + height
        nxt = red_bars.get(next_col)
        if nxt is None:
            break
        next_h, next_bottom = nxt
        assert next_bottom == bottom
        col, height = next_col, next_h
        length += 1
    return length


def _marker_distance(marker: Coord, rect: BBox) -> tuple[int, Coord]:
    r, c = marker
    r0, c0, r1, c1 = rect
    row_aligned = r0 <= r <= r1
    col_aligned = c0 <= c <= c1
    assert row_aligned ^ col_aligned
    if row_aligned:
        if c < c0:
            return c0 - c, (r, c0 - 1)
        return c - c1, (r, c1 + 1)
    if r < r0:
        return r0 - r, (r0 - 1, c)
    return r - r1, (r1 + 1, c)


def main(grid: np.ndarray) -> np.ndarray:
    """Use the bottom-left linked bar count to gate which aligned maroon markers slide to the rectangle.

    Input structure:
    - The background is black.
    - One solid blue rectangle sits near the center, and several isolated maroon marker pixels lie outside it.
    - Each marker is aligned with the rectangle by row or by column.
    - A bottom-aligned chain of one-pixel-wide vertical bars sits in the lower-left: the first bar is blue and each later bar is red.

    Rule:
    - Detect the bar chain by starting from the blue vertical bar and repeatedly finding the next red bar whose column equals the current bar column plus the current bar height.
    - Let L be the total number of bars in that chain, including the blue start bar.
    - For each maroon marker, measure its distance to the nearest aligned edge of the blue rectangle.
    - Keep only the markers whose distance equals L, slide each of them straight toward the rectangle, and stop in the exterior cell directly adjacent to the touched side.
    - Erase all other maroon markers, recolor the blue rectangle green, and leave the chain unchanged.

    Key insight:
    - The answer depends on combining two independent geometric analyses: a chain-length count in one corner and a distance test between the central rectangle and the aligned markers.
    """
    g = np.asarray(grid, dtype=int)
    rect, _red_bars, _start = _find_rectangle_and_chain(g)
    chain_len = _chain_length(g)
    out = g.copy()

    r0, c0, r1, c1 = rect
    out[r0 : r1 + 1, c0 : c1 + 1] = OUTPUT_RECT
    out[g == MARKER] = BG

    markers = [tuple(map(int, rc)) for rc in np.argwhere(g == MARKER)]
    assert markers
    moved: set[Coord] = set()
    selected = 0
    for marker in markers:
        dist, dst = _marker_distance(marker, rect)
        if dist != chain_len:
            continue
        selected += 1
        assert dst not in moved
        moved.add(dst)
        out[dst] = MARKER
    assert 1 <= selected <= 2
    return out


def _render_case(spec: dict) -> tuple[np.ndarray, np.ndarray]:
    h = spec["h"]
    w = spec["w"]
    rect = spec["rect"]
    chain = spec["chain"]
    markers = spec["markers"]

    grid = np.zeros((h, w), dtype=int)
    r0, c0, r1, c1 = rect
    grid[r0 : r1 + 1, c0 : c1 + 1] = RECT_COLOR

    bottom = chain["bottom"]
    col = chain["start_col"]
    for i, height in enumerate(chain["heights"]):
        top = bottom - height + 1
        grid[top : bottom + 1, col] = RECT_COLOR if i == 0 else CHAIN_RED
        if i + 1 < len(chain["heights"]):
            col += height

    for marker in markers:
        grid[marker["pos"]] = MARKER

    out = main(grid)
    return grid, out


def _candidate_markers(rect: BBox, shape: tuple[int, int]) -> list[dict]:
    h, w = shape
    r0, c0, r1, c1 = rect
    items: list[dict] = []
    for r in range(r0, r1 + 1):
        for dist in range(2, min(6, c0) + 1):
            items.append({"side": "left", "dist": dist, "pos": (r, c0 - dist)})
        for dist in range(2, min(6, w - 1 - c1) + 1):
            items.append({"side": "right", "dist": dist, "pos": (r, c1 + dist)})
    for c in range(c0, c1 + 1):
        for dist in range(2, min(6, r0) + 1):
            items.append({"side": "top", "dist": dist, "pos": (r0 - dist, c)})
        for dist in range(2, min(6, h - 1 - r1) + 1):
            items.append({"side": "bottom", "dist": dist, "pos": (r1 + dist, c)})
    return items


def _well_spaced(chosen: list[dict], cand: dict) -> bool:
    r, c = cand["pos"]
    for item in chosen:
        rr, cc = item["pos"]
        if max(abs(r - rr), abs(c - cc)) <= 1:
            return False
    return True


def _choose_markers(rect: BBox, shape: tuple[int, int], chain_len: int) -> list[dict] | None:
    candidates = _candidate_markers(rect, shape)
    by_dist: dict[int, list[dict]] = {}
    for cand in candidates:
        by_dist.setdefault(cand["dist"], []).append(cand)
    if chain_len not in by_dist or 2 not in by_dist or 6 not in by_dist:
        return None

    selected_count = random.choice([1, 2])
    if len(by_dist[chain_len]) < selected_count:
        return None

    total = random.randint(4, 6)
    others_needed = total - selected_count
    if others_needed < 2:
        return None

    low_distances = [d for d in sorted(by_dist) if d < chain_len]
    high_distances = [d for d in sorted(by_dist) if d > chain_len]
    if not low_distances or not high_distances:
        return None

    chosen: list[dict] = []
    pool = by_dist[chain_len].copy()
    random.shuffle(pool)
    for cand in pool:
        if _well_spaced(chosen, cand):
            chosen.append({**cand, "selected": True})
            if len(chosen) == selected_count:
                break
    if len(chosen) != selected_count:
        return None
    if selected_count == 2 and len({item["side"] for item in chosen}) == 1:
        return None

    required_dists = [random.choice(low_distances), random.choice(high_distances)]
    remaining = others_needed - 2
    other_distance_pool = [d for d in sorted(by_dist) if d != chain_len]
    required_dists.extend(random.choices(other_distance_pool, k=remaining))
    random.shuffle(required_dists)

    for dist in required_dists:
        options = by_dist[dist].copy()
        random.shuffle(options)
        placed = False
        for cand in options:
            if cand["pos"] in {item["pos"] for item in chosen}:
                continue
            if _well_spaced(chosen, cand):
                chosen.append({**cand, "selected": False})
                placed = True
                break
        if not placed:
            return None

    if len(chosen) != total:
        return None

    selected_dists = {item["dist"] for item in chosen if item["selected"]}
    if selected_dists != {chain_len}:
        return None
    dists = [item["dist"] for item in chosen]
    if min(dists) >= chain_len or max(dists) <= chain_len:
        return None
    return chosen


def _choose_chain(rect: BBox, shape: tuple[int, int]) -> dict | None:
    h, _w = shape
    _r0, c0, r1, _c1 = rect
    bottom = h - 2
    max_height = bottom - r1 - 2
    if max_height < 2:
        return None

    chain_len = random.choice([3, 4, 5])
    for _ in range(200):
        heights = [random.randint(2, min(4, max_height)) for _ in range(chain_len)]
        width = 1 + sum(heights[:-1])
        max_start = c0 - width - 2
        if max_start < 1:
            continue
        start_col = random.randint(1, max_start)
        return {
            "heights": heights,
            "bottom": bottom,
            "start_col": start_col,
            "length": chain_len,
        }
    return None


def _make_spec() -> dict | None:
    h = random.randint(16, 22)
    w = random.randint(19, 26)
    rect_h = random.randint(3, 5)
    rect_w = random.randint(3, 5)

    rect_r0_min = 4
    rect_r0_max = h - rect_h - 7
    rect_c0_min = 9
    rect_c0_max = w - rect_w - 7
    if rect_r0_min > rect_r0_max or rect_c0_min > rect_c0_max:
        return None

    rect_r0 = random.randint(rect_r0_min, rect_r0_max)
    rect_c0 = random.randint(rect_c0_min, rect_c0_max)
    rect = (rect_r0, rect_c0, rect_r0 + rect_h - 1, rect_c0 + rect_w - 1)

    chain = _choose_chain(rect, (h, w))
    if chain is None:
        return None
    markers = _choose_markers(rect, (h, w), chain["length"])
    if markers is None:
        return None

    spec = {"h": h, "w": w, "rect": rect, "chain": chain, "markers": markers}
    if not _validate_spec(spec):
        return None
    return spec


def _expected_from_spec(spec: dict) -> np.ndarray:
    h = spec["h"]
    w = spec["w"]
    out = np.zeros((h, w), dtype=int)
    r0, c0, r1, c1 = spec["rect"]
    out[r0 : r1 + 1, c0 : c1 + 1] = OUTPUT_RECT

    bottom = spec["chain"]["bottom"]
    col = spec["chain"]["start_col"]
    for i, height in enumerate(spec["chain"]["heights"]):
        top = bottom - height + 1
        out[top : bottom + 1, col] = RECT_COLOR if i == 0 else CHAIN_RED
        if i + 1 < len(spec["chain"]["heights"]):
            col += height

    rect = spec["rect"]
    for marker in spec["markers"]:
        if not marker["selected"]:
            continue
        _dist, dst = _marker_distance(marker["pos"], rect)
        out[dst] = MARKER
    return out


def _validate_spec(spec: dict) -> bool:
    grid, out = _render_case(spec)
    expected = _expected_from_spec(spec)
    if not np.array_equal(out, expected):
        return False

    if _chain_length(grid) != spec["chain"]["length"]:
        return False

    rect = spec["rect"]
    markers = [tuple(item["pos"]) for item in spec["markers"]]
    if len(markers) != len(set(markers)):
        return False

    selected = [item for item in spec["markers"] if item["selected"]]
    unselected = [item for item in spec["markers"] if not item["selected"]]
    if not (1 <= len(selected) <= 2):
        return False
    if len(unselected) < 2:
        return False

    dists = [item["dist"] for item in spec["markers"]]
    if min(dists) >= spec["chain"]["length"] or max(dists) <= spec["chain"]["length"]:
        return False

    wrong_nearest = {item["pos"] for item in spec["markers"] if item["dist"] == min(dists)}
    wrong_farthest = {item["pos"] for item in spec["markers"] if item["dist"] == max(dists)}
    right = {item["pos"] for item in selected}
    if right == wrong_nearest or right == wrong_farthest:
        return False

    for marker in spec["markers"]:
        dist, dst = _marker_distance(marker["pos"], rect)
        if dist != marker["dist"]:
            return False
        r, c = marker["pos"]
        rr, cc = dst
        if marker["selected"]:
            if max(abs(r - rr), abs(c - cc)) <= 0:
                return False
        if not (0 <= rr < spec["h"] and 0 <= cc < spec["w"]):
            return False

    return True


def generate_input() -> np.ndarray:
    for _ in range(500):
        spec = _make_spec()
        if spec is not None:
            return _render_case(spec)[0]
    raise RuntimeError("failed to generate a valid puzzle instance")
