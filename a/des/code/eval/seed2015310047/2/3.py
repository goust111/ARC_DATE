from __future__ import annotations

from itertools import combinations

import numpy as np


def _background_color(grid: np.ndarray) -> int:
    colors, counts = np.unique(grid, return_counts=True)
    return int(colors[int(np.argmax(counts))])


def _components(mask: np.ndarray) -> list[np.ndarray]:
    h, w = mask.shape
    seen = np.zeros((h, w), dtype=bool)
    comps: list[np.ndarray] = []
    for r in range(h):
        for c in range(w):
            if not mask[r, c] or seen[r, c]:
                continue
            stack = [(int(r), int(c))]
            seen[r, c] = True
            coords: list[tuple[int, int]] = []
            while stack:
                rr, cc = stack.pop()
                coords.append((int(rr), int(cc)))
                for dr, dc in ((1, 0), (-1, 0), (0, 1), (0, -1)):
                    nr = rr + dr
                    nc = cc + dc
                    if 0 <= nr < h and 0 <= nc < w and mask[nr, nc] and not seen[nr, nc]:
                        seen[nr, nc] = True
                        stack.append((int(nr), int(nc)))
            comps.append(np.array(coords, dtype=int))
    return comps


def _frame_from_points(points: set[tuple[int, int]]) -> dict | None:
    if len(points) < 8:
        return None

    rows = [r for r, _c in points]
    cols = [c for _r, c in points]
    r0 = min(rows)
    c0 = min(cols)
    r1 = max(rows)
    c1 = max(cols)
    h = int(r1 - r0 + 1)
    w = int(c1 - c0 + 1)
    if h < 3 or w < 3:
        return None

    perimeter = int(2 * (h + w) - 4)
    if len(points) != perimeter:
        return None

    for rr, cc in points:
        if rr not in (r0, r1) and cc not in (c0, c1):
            return None

    for cc in range(c0, c1 + 1):
        if (r0, cc) not in points or (r1, cc) not in points:
            return None
    for rr in range(r0, r1 + 1):
        if (rr, c0) not in points or (rr, c1) not in points:
            return None

    return {
        "bbox": (int(r0), int(c0), int(r1), int(c1)),
        "height": h,
        "width": w,
    }


def _extract_frame_component(comp: np.ndarray) -> dict:
    pts = {tuple(map(int, rc)) for rc in comp}
    candidates = []
    for marker in pts:
        frame = _frame_from_points(pts - {marker})
        if frame is None:
            continue
        r0, c0, r1, c1 = map(int, frame["bbox"])
        mr, mc = map(int, marker)
        if mr != r0 - 1:
            continue
        if not (c0 <= mc < c1):
            continue
        if (r0, mc) not in pts:
            continue
        candidates.append((frame, (mr, mc)))

    assert len(candidates) == 1, "each frame color must contain one ring plus one self-marker"
    frame, marker = candidates[0]
    frame["marker"] = marker
    return frame


def _detect_items(grid: np.ndarray) -> tuple[int, list[dict]]:
    bg = _background_color(grid)
    items: list[dict] = []
    for color in np.unique(grid):
        color = int(color)
        if color == bg:
            continue
        comps = _components(grid == color)
        assert comps, "non-background color must appear at least once"
        largest = max(comps, key=lambda comp: int(comp.shape[0]))
        frame = _extract_frame_component(largest)
        mr, mc = map(int, frame["marker"])
        assert mc + 1 < grid.shape[1], "pointer cell must exist"
        next_color = int(grid[mr, mc + 1])
        item = {
            "color": color,
            "next_color": next_color,
            "bbox": frame["bbox"],
            "height": int(frame["height"]),
            "width": int(frame["width"]),
        }
        items.append(item)

    by_color = {item["color"]: item for item in items}
    assert len(by_color) == len(items), "frame colors must be unique"
    return bg, items


def _decode_chain(items: list[dict], bg: int) -> list[dict]:
    by_color = {int(item["color"]): item for item in items}
    pointed = {int(item["next_color"]) for item in items if int(item["next_color"]) != bg}
    starts = [color for color in by_color if color not in pointed]
    assert len(starts) == 1, "marker links must define exactly one chain head"

    order: list[dict] = []
    seen: set[int] = set()
    current = int(starts[0])
    while current != bg:
        assert current in by_color, "pointer must target an existing frame color or background"
        assert current not in seen, "marker chain must be acyclic"
        item = by_color[current]
        order.append(item)
        seen.add(current)
        current = int(item["next_color"])

    assert len(order) == len(items), "marker chain must include every frame"
    return order


def _paint_frame(out: np.ndarray, color: int, top: int, left: int, height: int, width: int) -> None:
    bottom = int(top + height - 1)
    right = int(left + width - 1)
    out[top, left : right + 1] = int(color)
    out[bottom, left : right + 1] = int(color)
    out[top : bottom + 1, left] = int(color)
    out[top : bottom + 1, right] = int(color)


def main(grid: np.ndarray) -> np.ndarray:
    """
    Assemble the scattered rectangular rings into a concentric stack using the
    color-link chain encoded by the dots above each ring.
    """
    g = np.asarray(grid, dtype=int)
    assert g.ndim == 2, "expected a 2D grid"

    bg, items = _detect_items(g)
    chain = _decode_chain(items, bg)
    outer = chain[-1]
    out_h = int(outer["height"])
    out_w = int(outer["width"])
    out = np.full((out_h, out_w), int(bg), dtype=int)

    for item in chain:
        h = int(item["height"])
        w = int(item["width"])
        assert (out_h - h) % 2 == 0 and (out_w - w) % 2 == 0, "rings must share a grid-aligned center"
        top = int((out_h - h) // 2)
        left = int((out_w - w) // 2)
        _paint_frame(out, int(item["color"]), top, left, h, w)

    return out


# --------------- generator ---------------

_RNG = np.random.default_rng()

GRID_HEIGHT_RANGE = (20, 30)
GRID_WIDTH_RANGE = (20, 30)
FRAME_COUNT_RANGE = (3, 5)
OUTER_HEIGHT_RANGE = (11, 19)
OUTER_WIDTH_RANGE = (11, 19)
INNER_HEIGHT_RANGE = (5, 17)
INNER_WIDTH_RANGE = (5, 17)
PLACEMENT_RETRIES = 80
GENERATION_RETRIES = 400


def _randint(lo: int, hi: int) -> int:
    return int(_RNG.integers(int(lo), int(hi) + 1))


def _choice(seq):
    return seq[_randint(0, len(seq) - 1)]


def _sample_without_replacement(seq, k: int):
    idx = _RNG.choice(len(seq), size=int(k), replace=False)
    return [seq[int(i)] for i in idx]


def _all_crossing_free(dims: list[tuple[int, int]]) -> bool:
    sorted_dims = sorted((int(h), int(w)) for h, w in dims)
    widths = [w for _h, w in sorted_dims]
    return widths == sorted(widths)


def _has_crossing_pair(dims: list[tuple[int, int]]) -> bool:
    for (h1, w1), (h2, w2) in combinations(dims, 2):
        if (h1 < h2 and w1 > w2) or (h1 > h2 and w1 < w2):
            return True
    return False


def _sample_dimensions() -> list[tuple[int, int]] | None:
    count = _randint(*FRAME_COUNT_RANGE)
    same_h_parity = _randint(0, 1)
    same_w_parity = _randint(0, 1)
    require_crossing = bool(_randint(0, 1))

    outer_candidates = [
        (h, w)
        for h in range(OUTER_HEIGHT_RANGE[0], OUTER_HEIGHT_RANGE[1] + 1)
        for w in range(OUTER_WIDTH_RANGE[0], OUTER_WIDTH_RANGE[1] + 1)
        if h % 2 == same_h_parity and w % 2 == same_w_parity
    ]
    if not outer_candidates:
        return None
    outer_h, outer_w = _choice(outer_candidates)

    inner_candidates = [
        (h, w)
        for h in range(INNER_HEIGHT_RANGE[0], min(INNER_HEIGHT_RANGE[1], outer_h - 2) + 1)
        for w in range(INNER_WIDTH_RANGE[0], min(INNER_WIDTH_RANGE[1], outer_w - 2) + 1)
        if h % 2 == same_h_parity and w % 2 == same_w_parity
    ]
    inner_candidates = [(h, w) for h, w in inner_candidates if (h, w) != (outer_h, outer_w)]
    if len(inner_candidates) < count - 1:
        return None

    for _ in range(GENERATION_RETRIES):
        inner = _sample_without_replacement(inner_candidates, count - 1)
        areas = [int(h * w) for h, w in inner]
        if len(set(areas)) != len(areas):
            continue
        if require_crossing:
            if not _has_crossing_pair(inner):
                continue
        else:
            if not _all_crossing_free(inner):
                continue
        outer_area = int(outer_h * outer_w)
        if any(area >= outer_area for area in areas):
            continue
        return [(int(h), int(w)) for h, w in inner] + [(int(outer_h), int(outer_w))]
    return None


def _marker_column(width: int) -> int:
    return _randint(0, int(width) - 2)


def _build_patch(height: int, width: int, color: int, next_color: int, bg: int) -> np.ndarray:
    patch = np.full((int(height) + 1, int(width)), int(bg), dtype=int)
    patch[1, :] = int(color)
    patch[-1, :] = int(color)
    patch[1:, 0] = int(color)
    patch[1:, -1] = int(color)
    mc = _marker_column(width)
    patch[0, mc] = int(color)
    patch[0, mc + 1] = int(next_color)
    return patch


def _place_patches(patches: list[np.ndarray], bg: int) -> np.ndarray | None:
    max_h = max(int(patch.shape[0]) for patch in patches)
    max_w = max(int(patch.shape[1]) for patch in patches)
    grid_h = _randint(max(GRID_HEIGHT_RANGE[0], max_h + 2), GRID_HEIGHT_RANGE[1])
    grid_w = _randint(max(GRID_WIDTH_RANGE[0], max_w + 2), GRID_WIDTH_RANGE[1])
    grid = np.full((grid_h, grid_w), int(bg), dtype=int)
    reserved = np.zeros((grid_h, grid_w), dtype=bool)

    order = list(range(len(patches)))
    _RNG.shuffle(order)
    for idx in order:
        patch = patches[idx]
        ph, pw = map(int, patch.shape)
        placed = False
        for _ in range(PLACEMENT_RETRIES):
            top = _randint(0, grid_h - ph)
            left = _randint(0, grid_w - pw)
            r0 = max(0, top - 1)
            c0 = max(0, left - 1)
            r1 = min(grid_h, top + ph + 1)
            c1 = min(grid_w, left + pw + 1)
            if reserved[r0:r1, c0:c1].any():
                continue
            region = grid[top : top + ph, left : left + pw]
            mask = patch != int(bg)
            region[mask] = patch[mask]
            reserved[top : top + ph, left : left + pw] = True
            placed = True
            break
        if not placed:
            return None

    return grid


def _build_case() -> np.ndarray | None:
    dims = _sample_dimensions()
    if dims is None:
        return None

    count = len(dims)
    outer = max(dims, key=lambda hw: (int(hw[0]), int(hw[1]), int(hw[0] * hw[1])))
    inner = [tuple(map(int, hw)) for hw in dims if tuple(map(int, hw)) != tuple(map(int, outer))]
    assert len(inner) == count - 1

    bg = _randint(0, 9)
    colors = [c for c in range(10) if c != bg]
    if len(colors) < count:
        return None
    _RNG.shuffle(colors)
    frame_colors = [int(c) for c in colors[:count]]

    items = [
        {
            "height": int(h),
            "width": int(w),
            "area": int(h * w),
            "color": int(color),
        }
        for (h, w), color in zip(inner + [outer], frame_colors)
    ]

    outer_item = next(item for item in items if (item["height"], item["width"]) == outer)
    non_outer = [item for item in items if item is not outer_item]
    start_item = min(non_outer, key=lambda item: int(item["area"]))
    middle = [item for item in non_outer if item is not start_item]

    if len(middle) > 1:
        _RNG.shuffle(middle)
        if tuple(item["area"] for item in middle) == tuple(sorted(item["area"] for item in middle)):
            middle = middle[1:] + middle[:1]

    chain = [start_item] + middle + [outer_item]
    for item, nxt in zip(chain, chain[1:]):
        item["next_color"] = int(nxt["color"])
    chain[-1]["next_color"] = int(bg)

    patches = [
        _build_patch(
            int(item["height"]),
            int(item["width"]),
            int(item["color"]),
            int(item["next_color"]),
            int(bg),
        )
        for item in items
    ]

    grid = _place_patches(patches, bg)
    if grid is None:
        return None

    colors, counts = np.unique(grid, return_counts=True)
    if int(colors[int(np.argmax(counts))]) != bg:
        return None

    decoded_bg, decoded_items = _detect_items(grid)
    decoded_chain = _decode_chain(decoded_items, decoded_bg)
    if decoded_bg != bg:
        return None
    if [int(item["color"]) for item in decoded_chain] != [int(item["color"]) for item in chain]:
        return None

    out = main(grid)
    if out.shape != (int(outer_item["height"]), int(outer_item["width"])):
        return None
    return grid


def generate_input() -> np.ndarray:
    for _ in range(GENERATION_RETRIES):
        grid = _build_case()
        if grid is not None:
            return grid
    raise RuntimeError("failed to generate a valid puzzle instance")
