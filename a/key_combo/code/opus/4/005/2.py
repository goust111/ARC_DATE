import random
from collections import deque
from importlib.machinery import SourceFileLoader
from pathlib import Path

import numpy as np

_BLACK = 0
_GREEN = 3
_MARKER_COLORS = (3, 5, 8)
_PAIR_MAP = {8: 2, 3: 1, 5: 4}
_GEN_RETRIES = 400
_GEN_COUNTER = 0


def _load_arc_utils():
    path = Path("/home/lixinyu/snap/new/solvers/arc_utils.py")
    if not path.exists():
        return None
    try:
        return SourceFileLoader("_arc_utils_005_2", str(path)).load_module()
    except Exception:
        return None


_ARC_UTILS = _load_arc_utils()


if _ARC_UTILS is not None:
    neighbors4 = _ARC_UTILS.neighbors4
else:
    def neighbors4(r: int, c: int):
        yield r - 1, c
        yield r + 1, c
        yield r, c - 1
        yield r, c + 1


def _in_bounds(shape: tuple[int, int], r: int, c: int) -> bool:
    h, w = map(int, shape)
    return 0 <= int(r) < h and 0 <= int(c) < w


def _connected_components(mask: np.ndarray) -> list[dict]:
    mask = np.asarray(mask, dtype=bool)
    h, w = map(int, mask.shape)
    seen = np.zeros((h, w), dtype=bool)
    comps: list[dict] = []
    for r in range(h):
        for c in range(w):
            if not mask[r, c] or seen[r, c]:
                continue
            q = deque([(int(r), int(c))])
            seen[r, c] = True
            coords: list[tuple[int, int]] = []
            touches_border = False
            while q:
                rr, cc = q.popleft()
                coords.append((int(rr), int(cc)))
                if rr == 0 or rr == h - 1 or cc == 0 or cc == w - 1:
                    touches_border = True
                for nr, nc in neighbors4(rr, cc):
                    if _in_bounds(mask.shape, nr, nc) and mask[nr, nc] and not seen[nr, nc]:
                        seen[nr, nc] = True
                        q.append((int(nr), int(nc)))
            arr = np.array(coords, dtype=int)
            comps.append({"coords": arr, "touches_border": touches_border})
    return comps


def _mask_from_coords(shape: tuple[int, int], coords: np.ndarray) -> np.ndarray:
    out = np.zeros(shape, dtype=bool)
    out[coords[:, 0], coords[:, 1]] = True
    return out


def _flood_exterior(blocked: np.ndarray) -> np.ndarray:
    blocked = np.asarray(blocked, dtype=bool)
    h, w = map(int, blocked.shape)
    outside = np.zeros((h, w), dtype=bool)
    q: deque[tuple[int, int]] = deque()

    def seed(r: int, c: int) -> None:
        if not blocked[r, c] and not outside[r, c]:
            outside[r, c] = True
            q.append((int(r), int(c)))

    for r in range(h):
        seed(r, 0)
        seed(r, w - 1)
    for c in range(w):
        seed(0, c)
        seed(h - 1, c)

    while q:
        r, c = q.popleft()
        for nr, nc in neighbors4(r, c):
            if _in_bounds(blocked.shape, nr, nc) and not blocked[nr, nc] and not outside[nr, nc]:
                outside[nr, nc] = True
                q.append((int(nr), int(nc)))
    return outside


def _analyze_input(grid: np.ndarray) -> dict:
    g = np.asarray(grid, dtype=int)
    assert g.ndim == 2

    boundary_mask = np.zeros(g.shape, dtype=bool)
    boundaries: list[dict] = []
    for comp in _connected_components(g == _GREEN):
        comp_mask = _mask_from_coords(g.shape, comp["coords"])
        enclosed = ~_flood_exterior(comp_mask)
        interior = enclosed & ~comp_mask
        if interior.any() and not comp["touches_border"]:
            boundary_mask |= comp_mask
            boundaries.append({"coords": comp["coords"], "mask": comp_mask, "enclosed": enclosed})

    regions: list[dict] = []
    for comp in _connected_components(~boundary_mask):
        coords = comp["coords"]
        r0, c0 = map(int, coords[0])
        depth = sum(1 for boundary in boundaries if boundary["enclosed"][r0, c0])
        markers: list[tuple[int, int, int]] = []
        for r, c in coords:
            color = int(g[r, c])
            if color != _BLACK:
                markers.append((int(r), int(c), color))
        regions.append(
            {
                "coords": coords,
                "touches_border": comp["touches_border"],
                "depth": int(depth),
                "markers": markers,
            }
        )

    return {"boundary_mask": boundary_mask, "boundaries": boundaries, "regions": regions}


def main(grid: np.ndarray) -> np.ndarray:
    """
    Fill only leaf enclosed regions, with the fill color gated by nesting-depth parity.

    Input structure:
    - Black background contains several 4-connected green closed boundary rings, with some rings nested inside others.
    - Each leaf enclosed region contains exactly one isolated marker pixel of color cyan, green, or gray.
    - Non-leaf enclosed regions contain no markers; green boundary pixels stay separate from the marker even when the marker is green.

    Rule:
    - Identify the true green boundary loops, then split the non-boundary cells into connected regions.
    - For each non-border region containing a marker, compute how many green enclosed regions contain it; that count is the nesting depth.
    - Odd depths use the paired color map cyan->red, green->blue, gray->yellow; even depths keep the marker color itself.
    - Fill the entire marked leaf region, including the marker pixel, with that chosen color; keep non-leaf black areas and all green boundaries unchanged.

    Key insight:
    - The marker color alone is insufficient: the same marker color can produce different outputs depending on how many enclosing green loops surround its region.
    """
    g = np.asarray(grid, dtype=int)
    info = _analyze_input(g)
    out = g.copy()

    for region in info["regions"]:
        if region["touches_border"] or not region["markers"]:
            continue
        assert len(region["markers"]) == 1, "expected one marker per leaf region"
        marker_color = int(region["markers"][0][2])
        assert marker_color in _PAIR_MAP, "unexpected marker color"
        fill_color = _PAIR_MAP[marker_color] if region["depth"] % 2 == 1 else marker_color
        coords = region["coords"]
        out[coords[:, 0], coords[:, 1]] = fill_color

    return out


def _draw_rect_ring(grid: np.ndarray, top: int, left: int, height: int, width: int) -> None:
    bottom = top + height - 1
    right = left + width - 1
    grid[top, left : right + 1] = _GREEN
    grid[bottom, left : right + 1] = _GREEN
    grid[top : bottom + 1, left] = _GREEN
    grid[top : bottom + 1, right] = _GREEN


def _leaf_size(depth: int) -> tuple[int, int]:
    hi = {1: 8, 2: 7, 3: 6}[int(depth)]
    return random.randint(5, hi), random.randint(5, hi)


def _sample_specs() -> list[dict]:
    global _GEN_COUNTER
    idx = _GEN_COUNTER
    _GEN_COUNTER += 1

    chain_count = 3 if idx < 3 or random.random() < 0.7 else 2
    shared_color = [8, 3, 5][idx % 3]
    other_colors = [color for color in _MARKER_COLORS if color != shared_color]

    depth_to_color: dict[int, int] = {2: shared_color}
    if chain_count == 3:
        odd_shared = 1 if idx % 2 == 0 else 3
        odd_other = 3 if odd_shared == 1 else 1
        depth_to_color[odd_shared] = shared_color
        depth_to_color[odd_other] = other_colors[(idx // 2) % len(other_colors)]
        depths = [1, 2, 3]
    else:
        odd_depth = 1 if (idx // 3) % 2 == 0 else 3
        depth_to_color[odd_depth] = shared_color
        depths = [2, odd_depth]

    specs = []
    for depth in depths:
        leaf_h, leaf_w = _leaf_size(depth)
        specs.append(
            {
                "depth": int(depth),
                "marker_color": int(depth_to_color[depth]),
                "leaf_h": int(leaf_h),
                "leaf_w": int(leaf_w),
                "outer_h": int(leaf_h + 4 * (depth - 1)),
                "outer_w": int(leaf_w + 4 * (depth - 1)),
            }
        )
    random.shuffle(specs)
    return specs


def _sample_shape(specs: list[dict]) -> tuple[int, int]:
    max_h = max(spec["outer_h"] for spec in specs)
    max_w = max(spec["outer_w"] for spec in specs)
    total_area = sum(spec["outer_h"] * spec["outer_w"] for spec in specs)
    base = max(max_h + 2, max_w + 2, int(np.ceil(np.sqrt(total_area))) + 4)
    h = random.randint(base, min(30, base + 4))
    w = random.randint(base, min(30, base + 4))
    return int(h), int(w)


def _place_specs(shape: tuple[int, int], specs: list[dict]) -> list[dict] | None:
    h, w = map(int, shape)
    used = np.zeros((h, w), dtype=bool)
    placed: list[dict] = []

    random.shuffle(specs)
    order = sorted(specs, key=lambda spec: spec["outer_h"] * spec["outer_w"], reverse=True)
    for original in order:
        spec = dict(original)
        max_top = h - spec["outer_h"] - 1
        max_left = w - spec["outer_w"] - 1
        if max_top < 1 or max_left < 1:
            return None

        choices: list[tuple[int, int]] = []
        for _ in range(300):
            top = random.randint(1, max_top)
            left = random.randint(1, max_left)
            r0 = max(0, top - 1)
            r1 = min(h, top + spec["outer_h"] + 1)
            c0 = max(0, left - 1)
            c1 = min(w, left + spec["outer_w"] + 1)
            if used[r0:r1, c0:c1].any():
                continue
            choices.append((int(top), int(left)))
            if len(choices) >= 20:
                break
        if not choices:
            return None

        top, left = random.choice(choices)
        spec["top"] = int(top)
        spec["left"] = int(left)
        placed.append(spec)
        used[max(0, top - 1) : min(h, top + spec["outer_h"] + 1), max(0, left - 1) : min(w, left + spec["outer_w"] + 1)] = True

    return placed


def _paint_chain(grid: np.ndarray, spec: dict) -> None:
    top = int(spec["top"])
    left = int(spec["left"])
    depth = int(spec["depth"])
    for level in range(depth):
        ring_top = top + 2 * level
        ring_left = left + 2 * level
        ring_h = spec["outer_h"] - 4 * level
        ring_w = spec["outer_w"] - 4 * level
        _draw_rect_ring(grid, ring_top, ring_left, int(ring_h), int(ring_w))

    inner_top = top + 2 * (depth - 1)
    inner_left = left + 2 * (depth - 1)
    marker_r = random.randint(inner_top + 2, inner_top + spec["leaf_h"] - 3)
    marker_c = random.randint(inner_left + 2, inner_left + spec["leaf_w"] - 3)
    grid[marker_r, marker_c] = int(spec["marker_color"])
    spec["marker"] = (int(marker_r), int(marker_c))


def _validate_example(inp: np.ndarray, specs: list[dict]) -> None:
    info = _analyze_input(inp)
    boundary_mask = info["boundary_mask"]
    assert np.all(inp[boundary_mask] == _GREEN)

    inner_regions = [region for region in info["regions"] if not region["touches_border"]]
    leaf_regions = [region for region in inner_regions if region["markers"]]
    non_leaf_regions = [region for region in inner_regions if not region["markers"]]

    assert len(leaf_regions) == len(specs), "every chain should contribute one leaf region"
    assert non_leaf_regions, "need non-leaf regions so the keep-black branch is visible"
    assert {region["depth"] % 2 for region in leaf_regions} == {0, 1}, "need both odd and even depth leaves"

    leaf_by_marker = {}
    for region in leaf_regions:
        assert len(region["markers"]) == 1
        marker = region["markers"][0]
        assert marker[2] in _MARKER_COLORS
        leaf_by_marker[(marker[0], marker[1])] = region

    expected_markers = {tuple(spec["marker"]) for spec in specs}
    assert set(leaf_by_marker) == expected_markers
    for spec in specs:
        region = leaf_by_marker[tuple(spec["marker"])]
        assert region["depth"] == spec["depth"]
        assert region["markers"][0][2] == spec["marker_color"]

    grouped: dict[int, list[dict]] = {}
    for region in leaf_regions:
        grouped.setdefault(region["markers"][0][2], []).append(region)

    witnessed_shared_color = False
    for color, regions in grouped.items():
        parities = {region["depth"] % 2 for region in regions}
        fills = {_PAIR_MAP[color] if region["depth"] % 2 == 1 else color for region in regions}
        if parities == {0, 1} and len(fills) == 2:
            witnessed_shared_color = True
            break
    assert witnessed_shared_color, "same marker color must appear at odd and even depths"

    for region in non_leaf_regions:
        coords = region["coords"]
        assert np.all(inp[coords[:, 0], coords[:, 1]] == _BLACK)

    out = main(inp)
    assert out.shape == inp.shape
    assert np.array_equal(out[boundary_mask], inp[boundary_mask])
    assert not np.array_equal(out, inp)

    for region in inner_regions:
        coords = region["coords"]
        rr, cc = coords[:, 0], coords[:, 1]
        if region["markers"]:
            marker_color = region["markers"][0][2]
            fill = _PAIR_MAP[marker_color] if region["depth"] % 2 == 1 else marker_color
            assert np.all(out[rr, cc] == fill)
        else:
            assert np.array_equal(out[rr, cc], inp[rr, cc])


def _gen_attempt() -> np.ndarray | None:
    specs = _sample_specs()
    shape = _sample_shape(specs)
    placed = _place_specs(shape, specs)
    if placed is None:
        return None

    grid = np.zeros(shape, dtype=int)
    for spec in placed:
        _paint_chain(grid, spec)

    _validate_example(grid, placed)
    return grid


def generate_input() -> np.ndarray:
    for _ in range(_GEN_RETRIES):
        grid = _gen_attempt()
        if grid is not None:
            return grid
    raise RuntimeError("failed to generate a valid nested-depth color-mapping task")
