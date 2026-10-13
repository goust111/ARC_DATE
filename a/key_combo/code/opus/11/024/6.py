import random
from collections import Counter, deque

import numpy as np


BLACK = 0
BLUE = 1
RED = 2
GREEN = 3
YELLOW = 4
GREY = 5
PALETTE = (BLUE, RED, GREEN, YELLOW)

GEN_ATTEMPTS = 400
SELF_CHECKS = 12

_N4 = [(-1, 0), (1, 0), (0, -1), (0, 1)]
_N8 = [(dr, dc) for dr in (-1, 0, 1) for dc in (-1, 0, 1) if dr or dc]

_SHAPES = [
    ((0, 0), (0, 1)),
    ((0, 0), (1, 0)),
    ((0, 0), (0, 1), (0, 2)),
    ((0, 0), (1, 0), (2, 0)),
    ((0, 0), (0, 1), (1, 0)),
    ((0, 0), (0, 1), (1, 1)),
    ((0, 0), (1, 0), (1, 1)),
    ((0, 1), (1, 0), (1, 1)),
    ((0, 0), (0, 1), (1, 0), (1, 1)),
    ((0, 0), (0, 1), (0, 2), (0, 3)),
    ((0, 0), (1, 0), (2, 0), (3, 0)),
    ((0, 0), (1, 0), (2, 0), (2, 1)),
    ((0, 1), (1, 1), (2, 1), (2, 0)),
    ((0, 0), (0, 1), (0, 2), (1, 0)),
    ((0, 0), (0, 1), (0, 2), (1, 2)),
    ((0, 1), (1, 0), (1, 1), (1, 2)),
    ((0, 0), (0, 1), (1, 1), (2, 1)),
]


def get_bg(grid: np.ndarray) -> int:
    vals, counts = np.unique(np.asarray(grid, dtype=int), return_counts=True)
    return int(vals[np.argmax(counts)])


def _bbox(coords: np.ndarray) -> tuple[int, int, int, int]:
    r0, c0 = coords.min(axis=0)
    r1, c1 = coords.max(axis=0)
    return int(r0), int(c0), int(r1), int(c1)


def _components_of_color(grid: np.ndarray, color: int) -> list[np.ndarray]:
    g = np.asarray(grid, dtype=int)
    seen = np.zeros(g.shape, dtype=bool)
    comps: list[np.ndarray] = []
    for r, c in np.argwhere(g == int(color)):
        r = int(r)
        c = int(c)
        if seen[r, c]:
            continue
        q = deque([(r, c)])
        seen[r, c] = True
        pts = []
        while q:
            rr, cc = q.popleft()
            pts.append((rr, cc))
            for dr, dc in _N4:
                nr, nc = rr + dr, cc + dc
                if 0 <= nr < g.shape[0] and 0 <= nc < g.shape[1] and not seen[nr, nc] and g[nr, nc] == color:
                    seen[nr, nc] = True
                    q.append((nr, nc))
        comps.append(np.array(pts, dtype=int))
    return comps


def _find_gray_rectangle(grid: np.ndarray) -> tuple[int, int, int, int]:
    rects = []
    for comp in _components_of_color(grid, GREY):
        r0, c0, r1, c1 = _bbox(comp)
        h = r1 - r0 + 1
        w = c1 - c0 + 1
        if comp.shape[0] == h * w and h % 2 == 0 and w % 2 == 0:
            rects.append((r0, c0, r1, c1))
    assert len(rects) == 1, "expected exactly one solid even-sized gray rectangle"
    return rects[0]


def _region_name(cy: float, cx: float, center_r: float, center_c: float) -> str:
    assert cy != center_r and cx != center_c, "object centroid may not lie on a split axis"
    if cy < center_r and cx < center_c:
        return "tl"
    if cy < center_r and cx > center_c:
        return "tr"
    if cy > center_r and cx < center_c:
        return "bl"
    return "br"


def _analyze(grid: np.ndarray) -> dict:
    g = np.asarray(grid, dtype=int)
    assert g.ndim == 2
    bg = get_bg(g)
    assert bg == BLACK

    r0, c0, r1, c1 = _find_gray_rectangle(g)
    center_r = (r0 + r1) / 2.0
    center_c = (c0 + c1) / 2.0
    region_counts = {name: Counter() for name in ("tl", "tr", "bl", "br")}
    components = []

    for color in np.unique(g):
        color = int(color)
        if color in (bg, GREY):
            continue
        for comp in _components_of_color(g, color):
            rr0, cc0, rr1, cc1 = _bbox(comp)
            assert rr1 < r0 or rr0 > r1 or cc1 < c0 or cc0 > c1, "outside objects may not overlap rectangle"
            centroid = comp.mean(axis=0)
            region = _region_name(float(centroid[0]), float(centroid[1]), center_r, center_c)
            region_counts[region][color] += 1
            components.append({"color": color, "coords": comp, "region": region})

    winners = {}
    for name, counts in region_counts.items():
        assert counts, f"missing objects in region {name}"
        ordered = counts.most_common()
        assert len(ordered) >= 2, f"region {name} must contain at least two colors"
        assert ordered[0][1] > ordered[1][1], f"region {name} must have a unique most-common color"
        winners[name] = int(ordered[0][0])
    assert len(set(winners.values())) == 4, "the four quadrant winner colors must be distinct"

    return {
        "bbox": (r0, c0, r1, c1),
        "center": (center_r, center_c),
        "counts": region_counts,
        "winners": winners,
        "components": components,
    }


def main(grid: np.ndarray) -> np.ndarray:
    """
    Recolor the gray rectangle quadrants from object-count winners in the matching outer diagonal regions.

    Input structure:
    - The background is black.
    - One solid gray rectangle of even height and width sits near the center.
    - Outside the rectangle, small 4-connected monochrome objects appear in the four diagonal regions around the rectangle.

    Rule:
    - Find the gray rectangle and split it into four equal quadrants.
    - Group every non-gray foreground object by the position of its centroid relative to the rectangle center: top-left, top-right, bottom-left, or bottom-right.
    - In each region, count how many separate objects of each color appear, take the unique most frequent color, and fill the corresponding rectangle quadrant with that color.
    - Keep every outside object unchanged.

    Key insight:
    - The quadrant colors are not given by local marker pixels; they come from counting whole objects in the matching outer diagonal sector, so both spatial classification and per-color component counting are necessary.
    """
    g = np.asarray(grid, dtype=int)
    info = _analyze(g)
    r0, c0, r1, c1 = info["bbox"]
    winners = info["winners"]
    r_mid = r0 + (r1 - r0 + 1) // 2
    c_mid = c0 + (c1 - c0 + 1) // 2

    out = g.copy()
    out[r0:r_mid, c0:c_mid] = winners["tl"]
    out[r0:r_mid, c_mid : c1 + 1] = winners["tr"]
    out[r_mid : r1 + 1, c0:c_mid] = winners["bl"]
    out[r_mid : r1 + 1, c_mid : c1 + 1] = winners["br"]
    return out


def _choose_region_counts(rng: random.Random, winner: int) -> Counter:
    n_colors = rng.choice([2, 3, 4])
    others = [color for color in PALETTE if color != winner]
    chosen = [winner] + rng.sample(others, n_colors - 1)
    rng.shuffle(chosen)

    win_count = rng.choice([2, 3])
    counts = Counter({winner: win_count})
    for color in chosen:
        if color == winner:
            continue
        counts[color] = rng.randint(1, win_count - 1)
    return counts


def _quadrant_bounds(region: str, shape: tuple[int, int], rect_bbox: tuple[int, int, int, int]) -> tuple[int, int, int, int]:
    h, w = shape
    r0, c0, r1, c1 = rect_bbox
    r_mid = r0 + (r1 - r0 + 1) // 2
    c_mid = c0 + (c1 - c0 + 1) // 2
    if region == "tl":
        return 0, r_mid - 1, 0, c_mid - 1
    if region == "tr":
        return 0, r_mid - 1, c_mid, w - 1
    if region == "bl":
        return r_mid, h - 1, 0, c_mid - 1
    return r_mid, h - 1, c_mid, w - 1


def _place_object(
    grid: np.ndarray,
    reserved: np.ndarray,
    region: str,
    color: int,
    rect_bbox: tuple[int, int, int, int],
    rng: random.Random,
) -> bool:
    h, w = grid.shape
    rr0, rr1, cc0, cc1 = _quadrant_bounds(region, grid.shape, rect_bbox)
    shape = rng.choice(_SHAPES)
    max_r = max(r for r, _ in shape)
    max_c = max(c for _, c in shape)

    anchors = [(r, c) for r in range(rr0, rr1 - max_r + 1) for c in range(cc0, cc1 - max_c + 1)]
    rng.shuffle(anchors)

    r0, c0, r1, c1 = rect_bbox
    center_r = (r0 + r1) / 2.0
    center_c = (c0 + c1) / 2.0
    for ar, ac in anchors:
        coords = [(ar + dr, ac + dc) for dr, dc in shape]
        if any(not (0 <= r < h and 0 <= c < w) for r, c in coords):
            continue
        if any(grid[r, c] != BLACK or reserved[r, c] for r, c in coords):
            continue
        if any(r0 <= r <= r1 and c0 <= c <= c1 for r, c in coords):
            continue
        centroid = np.mean(np.array(coords, dtype=float), axis=0)
        if _region_name(float(centroid[0]), float(centroid[1]), center_r, center_c) != region:
            continue

        for r, c in coords:
            grid[r, c] = color
        for r, c in coords:
            for dr, dc in _N8 + [(0, 0)]:
                nr, nc = r + dr, c + dc
                if 0 <= nr < h and 0 <= nc < w:
                    reserved[nr, nc] = True
        return True
    return False


def _validate_input(grid: np.ndarray) -> None:
    info = _analyze(grid)
    r0, c0, r1, c1 = info["bbox"]
    h = r1 - r0 + 1
    w = c1 - c0 + 1
    assert h in (6, 8) and w in (6, 8)

    for comp in info["components"]:
        assert 2 <= comp["coords"].shape[0] <= 4

    for name, counts in info["counts"].items():
        assert 2 <= len(counts) <= 4
        for value in counts.values():
            assert 1 <= value <= 3
        ordered = counts.most_common()
        assert ordered[0][1] > ordered[1][1]

    out = main(grid)
    outside_in = grid.copy()
    outside_in[r0 : r1 + 1, c0 : c1 + 1] = BLACK
    outside_out = out.copy()
    outside_out[r0 : r1 + 1, c0 : c1 + 1] = BLACK
    assert np.array_equal(outside_in, outside_out), "outside objects must remain unchanged"

    r_mid = r0 + h // 2
    c_mid = c0 + w // 2
    assert np.all(out[r0:r_mid, c0:c_mid] == info["winners"]["tl"])
    assert np.all(out[r0:r_mid, c_mid : c1 + 1] == info["winners"]["tr"])
    assert np.all(out[r_mid : r1 + 1, c0:c_mid] == info["winners"]["bl"])
    assert np.all(out[r_mid : r1 + 1, c_mid : c1 + 1] == info["winners"]["br"])


def _gen_attempt(rng: random.Random) -> np.ndarray | None:
    h = rng.randint(22, 30)
    w = rng.randint(22, 30)
    rect_h = rng.choice([6, 8])
    rect_w = rng.choice([6, 8])

    base_r0 = h // 2 - rect_h // 2
    base_c0 = w // 2 - rect_w // 2
    r0 = min(max(base_r0 + rng.randint(-1, 1), 4), h - rect_h - 4)
    c0 = min(max(base_c0 + rng.randint(-1, 1), 4), w - rect_w - 4)
    r1 = r0 + rect_h - 1
    c1 = c0 + rect_w - 1

    grid = np.zeros((h, w), dtype=int)
    grid[r0 : r1 + 1, c0 : c1 + 1] = GREY
    reserved = np.zeros_like(grid, dtype=bool)
    reserved[max(0, r0 - 1) : min(h, r1 + 2), max(0, c0 - 1) : min(w, c1 + 2)] = True

    winners = list(PALETTE)
    rng.shuffle(winners)
    region_order = ["tl", "tr", "bl", "br"]
    region_counts = {name: _choose_region_counts(rng, winner) for name, winner in zip(region_order, winners)}

    expanded_objects = []
    for region in region_order:
        items = []
        for color, count in region_counts[region].items():
            items.extend([int(color)] * int(count))
        rng.shuffle(items)
        expanded_objects.append((region, items))

    for region, items in expanded_objects:
        for color in items:
            if not _place_object(grid, reserved, region, color, (r0, c0, r1, c1), rng):
                return None

    try:
        _validate_input(grid)
    except AssertionError:
        return None
    return grid


def generate_input() -> np.ndarray:
    rng = random.Random()
    for _ in range(GEN_ATTEMPTS):
        grid = _gen_attempt(rng)
        if grid is not None:
            break
    else:
        raise RuntimeError("failed to generate a valid puzzle-6 input")

    for _ in range(SELF_CHECKS):
        _validate_input(grid)
    return grid
