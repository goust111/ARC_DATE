import random
from collections import defaultdict

import numpy as np


COLOR_MAP = {
    1: 6,
    2: 8,
    3: 9,
    4: 7,
}
SOURCE_COLORS = tuple(COLOR_MAP.keys())
BACKGROUND_CHOICES = (0, 5)

# --------------- shared helpers ---------------


def _get_bg(grid: np.ndarray) -> int:
    values, counts = np.unique(grid, return_counts=True)
    return int(values[np.argmax(counts)])


def _components_of_color(grid: np.ndarray, color: int) -> list[dict]:
    h, w = grid.shape
    seen = np.zeros((h, w), dtype=bool)
    comps: list[dict] = []
    mask = grid == int(color)

    for r, c in np.argwhere(mask):
        r = int(r)
        c = int(c)
        if seen[r, c]:
            continue
        stack = [(r, c)]
        seen[r, c] = True
        coords = []
        while stack:
            cr, cc = stack.pop()
            coords.append((int(cr), int(cc)))
            for dr, dc in ((1, 0), (-1, 0), (0, 1), (0, -1)):
                nr = int(cr + dr)
                nc = int(cc + dc)
                if 0 <= nr < h and 0 <= nc < w and mask[nr, nc] and not seen[nr, nc]:
                    seen[nr, nc] = True
                    stack.append((nr, nc))

        arr = np.array(coords, dtype=int)
        r0, c0 = arr.min(axis=0)
        r1, c1 = arr.max(axis=0)
        comps.append(
            {
                "color": int(color),
                "coords": arr,
                "bbox": (int(r0), int(c0), int(r1), int(c1)),
                "area": int(arr.shape[0]),
            }
        )

    return comps


def _parse_anchors(grid: np.ndarray) -> tuple[int, dict[int, list[dict]], list[dict]]:
    g = np.asarray(grid, dtype=int)
    bg = _get_bg(g)
    by_color: dict[int, list[dict]] = defaultdict(list)
    anchors: list[dict] = []

    for color in sorted(int(c) for c in np.unique(g) if int(c) != int(bg)):
        comps = _components_of_color(g, color)
        assert len(comps) == 2, "each foreground color must form exactly two anchors"
        for comp in comps:
            r0, c0, r1, c1 = comp["bbox"]
            side = int(r1 - r0 + 1)
            assert side == int(c1 - c0 + 1), "anchors must have square bboxes"
            assert side in (2, 3), "anchors must be 2x2 or 3x3"
            assert int(comp["area"]) == side * side, "anchors must be solid squares"
            block = g[r0 : r1 + 1, c0 : c1 + 1]
            assert np.all(block == int(color)), "anchor bbox must be monochrome"
            anchor = {
                "color": int(color),
                "bbox": (int(r0), int(c0), int(r1), int(c1)),
                "size": int(side),
            }
            by_color[int(color)].append(anchor)
            anchors.append(anchor)

    assert anchors, "expected at least one anchor pair"
    return int(bg), by_color, anchors


def _center(anchor: dict) -> tuple[int, int]:
    r0, c0, r1, c1 = anchor["bbox"]
    return int((r0 + r1) // 2), int((c0 + c1) // 2)


def _reading_key(anchor: dict) -> tuple[int, int, int]:
    r0, c0, _r1, _c1 = anchor["bbox"]
    return int(r0), int(c0), int(anchor["size"])


def _outside_point(anchor: dict, side: str) -> tuple[int, int]:
    r0, c0, r1, c1 = anchor["bbox"]
    # Raster midpoint: for 2x2 anchors the midpoint is floored to one cell.
    mr = int((r0 + r1) // 2)
    mc = int((c0 + c1) // 2)
    if side == "left":
        return mr, int(c0 - 1)
    if side == "right":
        return mr, int(c1 + 1)
    if side == "top":
        return int(r0 - 1), mc
    if side == "bottom":
        return int(r1 + 1), mc
    raise ValueError(f"unknown side {side}")


def _line_cells(a: tuple[int, int], b: tuple[int, int]) -> list[tuple[int, int]]:
    ar, ac = a
    br, bc = b
    if ar == br:
        step = 1 if bc >= ac else -1
        return [(int(ar), int(c)) for c in range(int(ac), int(bc) + step, step)]
    if ac == bc:
        step = 1 if br >= ar else -1
        return [(int(r), int(ac)) for r in range(int(ar), int(br) + step, step)]
    raise ValueError("line must be axis-aligned")


def _role_and_order(a: dict, b: dict) -> tuple[dict, dict, str]:
    if int(a["size"]) != int(b["size"]):
        start, target = (a, b) if int(a["size"]) < int(b["size"]) else (b, a)
        return start, target, "HV"
    start, target = sorted((a, b), key=_reading_key)
    return start, target, "VH"


def _build_segments(start: dict, target: dict, order: str) -> tuple[list[tuple[int, int]], list[tuple[int, int]]]:
    sr, sc = _center(start)
    tr, tc = _center(target)
    assert sr != tr and sc != tc, "anchor pairs must differ in both row and column"

    if order == "HV":
        start_side = "right" if tc > sc else "left"
        target_side = "top" if tr > sr else "bottom"
        start_pt = _outside_point(start, start_side)
        end_pt = _outside_point(target, target_side)
        turn = (int(start_pt[0]), int(end_pt[1]))
        first = _line_cells(start_pt, turn)[:-1]
        second = _line_cells(turn, end_pt)
    else:
        assert order == "VH"
        start_side = "bottom" if tr > sr else "top"
        target_side = "left" if tc > sc else "right"
        start_pt = _outside_point(start, start_side)
        end_pt = _outside_point(target, target_side)
        turn = (int(end_pt[0]), int(start_pt[1]))
        first = _line_cells(start_pt, turn)[:-1]
        second = _line_cells(turn, end_pt)

    assert len(first) >= 2, "first bridge segment must be visible"
    assert len(second) >= 2, "second bridge segment must be visible"
    return first, second


def _anchor_mask(shape: tuple[int, int], anchors: list[dict]) -> np.ndarray:
    mask = np.zeros(shape, dtype=bool)
    for anchor in anchors:
        r0, c0, r1, c1 = anchor["bbox"]
        mask[r0 : r1 + 1, c0 : c1 + 1] = True
    return mask


def _solve(grid: np.ndarray) -> np.ndarray:
    g = np.asarray(grid, dtype=int)
    bg, by_color, anchors = _parse_anchors(g)
    occupied = _anchor_mask(g.shape, anchors)
    out = g.copy()

    for color in sorted(by_color):
        a, b = by_color[color]
        start, target, order = _role_and_order(a, b)
        first, second = _build_segments(start, target, order)
        mapped = int(COLOR_MAP[int(color)])

        for r, c in first + second:
            assert 0 <= r < g.shape[0] and 0 <= c < g.shape[1], "bridge must stay in bounds"
            assert not occupied[r, c], "bridge must stay outside every anchor"

        for r, c in first:
            out[r, c] = int(color)
        for r, c in second:
            out[r, c] = mapped

    assert _get_bg(out) == bg, "background should remain the dominant color"
    return out


def main(grid: np.ndarray) -> np.ndarray:
    """
    Pair same-colored solid 2x2/3x3 anchors and join each pair with a two-color L bridge.

    - Different sizes: start at the smaller anchor and go horizontal, then vertical.
    - Equal sizes: use reading order to choose the start anchor, then go vertical, then horizontal.
    - The first segment uses the anchor color and the second uses the task's fixed mapping color.
    """
    return _solve(grid)


# --------------- generator ---------------


GEN_ATTEMPTS = 400
PAIR_ATTEMPTS = 200
PAIR_COUNT_CHOICES = (2, 3)
GRID_HEIGHT_RANGE = (18, 28)
GRID_WIDTH_RANGE = (18, 28)


def _expanded_bbox(bbox: tuple[int, int, int, int], pad: int, shape: tuple[int, int]) -> tuple[int, int, int, int]:
    h, w = shape
    r0, c0, r1, c1 = bbox
    return (
        max(0, int(r0 - pad)),
        max(0, int(c0 - pad)),
        min(h - 1, int(r1 + pad)),
        min(w - 1, int(c1 + pad)),
    )


def _bbox_clear(mask: np.ndarray, bbox: tuple[int, int, int, int], pad: int = 0) -> bool:
    r0, c0, r1, c1 = _expanded_bbox(bbox, pad, mask.shape)
    return not bool(mask[r0 : r1 + 1, c0 : c1 + 1].any())


def _paint_bbox(mask: np.ndarray, bbox: tuple[int, int, int, int]) -> None:
    r0, c0, r1, c1 = bbox
    mask[r0 : r1 + 1, c0 : c1 + 1] = True


def _random_anchor(h: int, w: int, size: int, color: int) -> dict:
    r0 = random.randint(1, int(h - size - 2))
    c0 = random.randint(1, int(w - size - 2))
    return {
        "color": int(color),
        "bbox": (int(r0), int(c0), int(r0 + size - 1), int(c0 + size - 1)),
        "size": int(size),
    }


def _sample_pair_modes(pair_count: int) -> list[str]:
    modes = ["same", "diff"]
    while len(modes) < pair_count:
        modes.append(random.choice(["same", "diff"]))
    random.shuffle(modes)
    return modes


def _sample_sizes(mode: str) -> tuple[int, int]:
    if mode == "same":
        size = random.choice((2, 3))
        return int(size), int(size)
    return 2, 3


def _place_pair(
    h: int,
    w: int,
    color: int,
    mode: str,
    reserved: np.ndarray,
) -> tuple[dict, dict, list[tuple[int, int]], list[tuple[int, int]]] | None:
    size_a, size_b = _sample_sizes(mode)

    for _ in range(PAIR_ATTEMPTS):
        a = _random_anchor(h, w, size_a, color)
        if not _bbox_clear(reserved, a["bbox"], pad=1):
            continue

        b = _random_anchor(h, w, size_b, color)
        if not _bbox_clear(reserved, b["bbox"], pad=1):
            continue
        if not _bbox_clear(_anchor_mask((h, w), [a]), b["bbox"], pad=1):
            continue

        start, target, order = _role_and_order(a, b)
        try:
            first, second = _build_segments(start, target, order)
        except (AssertionError, ValueError):
            continue

        current_anchors = _anchor_mask((h, w), [a, b])
        path_cells = first + second
        if len(path_cells) != len(set(path_cells)):
            continue

        blocked = False
        for r, c in path_cells:
            if not (0 <= r < h and 0 <= c < w):
                blocked = True
                break
            if current_anchors[r, c] or reserved[r, c]:
                blocked = True
                break
        if blocked:
            continue

        return a, b, first, second

    return None


def _gen_attempt() -> np.ndarray | None:
    pair_count = random.choice(PAIR_COUNT_CHOICES)
    h = random.randint(*GRID_HEIGHT_RANGE)
    w = random.randint(*GRID_WIDTH_RANGE)
    bg = random.choice(BACKGROUND_CHOICES)
    colors = random.sample(SOURCE_COLORS, pair_count)
    modes = _sample_pair_modes(pair_count)

    grid = np.full((h, w), int(bg), dtype=int)
    reserved = np.zeros((h, w), dtype=bool)

    for color, mode in zip(colors, modes):
        placed = _place_pair(h, w, int(color), mode, reserved)
        if placed is None:
            return None
        a, b, first, second = placed
        for anchor in (a, b):
            r0, c0, r1, c1 = anchor["bbox"]
            grid[r0 : r1 + 1, c0 : c1 + 1] = int(color)
            _paint_bbox(reserved, anchor["bbox"])
        for r, c in first + second:
            reserved[r, c] = True

    # Ensure the generated sample satisfies the solver's assumptions.
    _ = _solve(grid)
    return grid


def generate_input() -> np.ndarray:
    for _ in range(GEN_ATTEMPTS):
        grid = _gen_attempt()
        if grid is not None:
            return grid
    raise RuntimeError("failed to generate a valid ARC sample")
