from __future__ import annotations

import random

import numpy as np


DIRS = [(-1, 0), (0, 1), (1, 0), (0, -1)]
SIDE_NAMES = ["top", "right", "bottom", "left"]
SIDE_TO_CW = {"top": 0, "right": 1, "bottom": 2, "left": 3}
BG = 0

RAW_SHAPES = [
    [[1, 0], [1, 1]],
    [[1, 0], [1, 0], [1, 1]],
    [[1, 1, 1], [0, 1, 0]],
    [[1, 1, 1], [1, 0, 0]],
    [[1, 0, 0], [1, 1, 1]],
    [[1, 1, 0], [1, 0, 0], [1, 1, 1]],
    [[1, 0, 1], [1, 1, 1]],
    [[1, 1, 0], [0, 1, 0], [0, 1, 1]],
    [[1, 0, 0], [1, 1, 0], [0, 1, 1]],
]


def _components(mask: np.ndarray) -> list[list[tuple[int, int]]]:
    h, w = mask.shape
    seen = np.zeros((h, w), dtype=bool)
    comps: list[list[tuple[int, int]]] = []
    for r in range(h):
        for c in range(w):
            if not mask[r, c] or seen[r, c]:
                continue
            stack = [(int(r), int(c))]
            seen[r, c] = True
            pts: list[tuple[int, int]] = []
            while stack:
                rr, cc = stack.pop()
                pts.append((rr, cc))
                for dr, dc in DIRS:
                    nr = rr + dr
                    nc = cc + dc
                    if 0 <= nr < h and 0 <= nc < w and mask[nr, nc] and not seen[nr, nc]:
                        seen[nr, nc] = True
                        stack.append((nr, nc))
            comps.append(pts)
    return comps


def _bbox(coords: list[tuple[int, int]] | tuple[tuple[int, int], ...]) -> tuple[int, int, int, int]:
    rs = [r for r, _ in coords]
    cs = [c for _, c in coords]
    return min(rs), min(cs), max(rs), max(cs)


def _coords_to_mask(coords: list[tuple[int, int]] | tuple[tuple[int, int], ...]) -> np.ndarray:
    r0, c0, r1, c1 = _bbox(coords)
    out = np.zeros((r1 - r0 + 1, c1 - c0 + 1), dtype=bool)
    for r, c in coords:
        out[r - r0, c - c0] = True
    return out


def _rotate_mask_cw(mask: np.ndarray, k: int) -> np.ndarray:
    return np.rot90(mask, -(k % 4))


def _unique_rotations(mask: np.ndarray) -> list[np.ndarray]:
    seen: set[tuple[tuple[int, int], bytes]] = set()
    out: list[np.ndarray] = []
    for k in range(4):
        rot = _rotate_mask_cw(mask, k)
        key = (rot.shape, rot.tobytes())
        if key not in seen:
            seen.add(key)
            out.append(rot)
    return out


def _shape_library() -> list[np.ndarray]:
    shapes: list[np.ndarray] = []
    seen: set[tuple[tuple[int, int], bytes]] = set()
    for raw in RAW_SHAPES:
        mask = np.array(raw, dtype=bool)
        if len(_unique_rotations(mask)) != 4:
            continue
        key = min((rot.shape, rot.tobytes()) for rot in _unique_rotations(mask))
        if key in seen:
            continue
        seen.add(key)
        shapes.append(mask)
    assert shapes, "shape library must be non-empty"
    return shapes


SHAPE_LIBRARY = _shape_library()


def _holes_for_component(grid: np.ndarray, coords: list[tuple[int, int]]) -> list[dict]:
    r0, c0, r1, c1 = _bbox(coords)
    sub = grid[r0 : r1 + 1, c0 : c1 + 1]
    holes: list[dict] = []
    for comp in _components(sub == BG):
        if any(
            rr == 0 or cc == 0 or rr == sub.shape[0] - 1 or cc == sub.shape[1] - 1
            for rr, cc in comp
        ):
            continue
        hr0, hc0, hr1, hc1 = _bbox(comp)
        mask = np.zeros((hr1 - hr0 + 1, hc1 - hc0 + 1), dtype=bool)
        for rr, cc in comp:
            mask[rr - hr0, cc - hc0] = True
        holes.append({"r0": r0 + hr0, "c0": c0 + hc0, "mask": mask})
    return holes


def _marker_side(
    marker_pos: tuple[int, int], region_cells: set[tuple[int, int]], region_color: int
) -> tuple[str, int]:
    r, c = marker_pos
    adj_dirs = []
    for idx, (dr, dc) in enumerate(DIRS):
        if (r + dr, c + dc) in region_cells:
            adj_dirs.append(idx)

    if len(adj_dirs) == 1:
        side_idx = (adj_dirs[0] + 2) % 4
        return SIDE_NAMES[side_idx], BG
    if len(adj_dirs) == 3:
        side_idx = next(idx for idx in range(4) if idx not in adj_dirs)
        return SIDE_NAMES[side_idx], int(region_color)

    rs = [rr for rr, _ in region_cells]
    cs = [cc for _, cc in region_cells]
    r0, c0, r1, c1 = min(rs), min(cs), max(rs), max(cs)
    if r < r0:
        return "top", BG
    if r > r1:
        return "bottom", BG
    if c < c0:
        return "left", BG
    if c > c1:
        return "right", BG
    if r == r0:
        return "top", int(region_color)
    if r == r1:
        return "bottom", int(region_color)
    if c == c0:
        return "left", int(region_color)
    if c == c1:
        return "right", int(region_color)
    raise AssertionError("failed to infer marker side")


def main(grid: np.ndarray) -> np.ndarray:
    """
    Fill each hole in a large monochrome region with its matching outside sprite,
    after rotating that sprite according to a one-pixel edge marker.
    """
    g = np.asarray(grid, dtype=int)
    assert g.ndim == 2
    out = g.copy()

    colors = [int(v) for v in np.unique(g) if int(v) != BG]
    comps_by_color: dict[int, list[dict]] = {}
    singletons: list[dict] = []

    for color in colors:
        comps: list[dict] = []
        for pts in _components(g == color):
            comp = {
                "coords": pts,
                "size": int(len(pts)),
                "mask": _coords_to_mask(pts),
            }
            comps.append(comp)
            if comp["size"] == 1:
                singletons.append({"color": int(color), "pos": pts[0]})
        comps_by_color[int(color)] = comps

    for color, comps in comps_by_color.items():
        region_comp = None
        hole_info = None
        for comp in comps:
            holes = _holes_for_component(g, comp["coords"])
            if len(holes) == 1:
                assert region_comp is None, "region color should have a unique holed component"
                region_comp = comp
                hole_info = holes[0]
        if region_comp is None:
            continue

        region_cells = set(region_comp["coords"])
        matching_markers = [
            marker
            for marker in singletons
            if marker["color"] != color
            and any((marker["pos"][0] + dr, marker["pos"][1] + dc) in region_cells for dr, dc in DIRS)
        ]
        assert len(matching_markers) == 1, "each region should have exactly one adjacent marker"
        marker = matching_markers[0]
        side, restore_value = _marker_side(marker["pos"], region_cells, color)
        rot_cw = SIDE_TO_CW[side]

        sprite_comp = None
        for comp in comps:
            if comp is region_comp:
                continue
            if np.array_equal(_rotate_mask_cw(comp["mask"], rot_cw), hole_info["mask"]):
                assert sprite_comp is None, "sprite match should be unique"
                sprite_comp = comp

        assert sprite_comp is not None, "expected a same-color outside sprite matching the hole"

        for r, c in sprite_comp["coords"]:
            out[r, c] = BG

        mr, mc = marker["pos"]
        out[mr, mc] = int(restore_value)

        patch = out[
            hole_info["r0"] : hole_info["r0"] + hole_info["mask"].shape[0],
            hole_info["c0"] : hole_info["c0"] + hole_info["mask"].shape[1],
        ]
        patch[hole_info["mask"]] = int(color)

    return out


def _paste_patch(dst: np.ndarray, patch: np.ndarray, r0: int, c0: int) -> None:
    h, w = patch.shape
    sub = dst[r0 : r0 + h, c0 : c0 + w]
    assert sub.shape == patch.shape
    mask = patch != BG
    assert not np.any(sub[mask] != BG)
    sub[mask] = patch[mask]


def _reserve_ok(occupied: np.ndarray, shape: tuple[int, int], r0: int, c0: int, pad: int) -> bool:
    h, w = shape
    H, W = occupied.shape
    if r0 < 0 or c0 < 0 or r0 + h > H or c0 + w > W:
        return False
    rr0 = max(0, r0 - pad)
    cc0 = max(0, c0 - pad)
    rr1 = min(H, r0 + h + pad)
    cc1 = min(W, c0 + w + pad)
    return not bool(np.any(occupied[rr0:rr1, cc0:cc1]))


def _reserve_bbox(occupied: np.ndarray, shape: tuple[int, int], r0: int, c0: int) -> None:
    h, w = shape
    occupied[r0 : r0 + h, c0 : c0 + w] = True


def _add_tab(mask: np.ndarray, side: str) -> tuple[np.ndarray, tuple[int, int]]:
    h, w = mask.shape
    if side in ("top", "bottom"):
        tab_h = random.randint(1, 2)
        tab_w = random.randint(2, max(2, w - 2))
        start_c = random.randint(1, w - tab_w - 1)
        if side == "top":
            out = np.zeros((h + tab_h, w), dtype=bool)
            out[tab_h:, :] = mask
            out[:tab_h, start_c : start_c + tab_w] = True
            return out, (tab_h, 0)
        out = np.zeros((h + tab_h, w), dtype=bool)
        out[:h, :] = mask
        out[h:, start_c : start_c + tab_w] = True
        return out, (0, 0)

    tab_w = random.randint(1, 2)
    tab_h = random.randint(2, max(2, h - 2))
    start_r = random.randint(1, h - tab_h - 1)
    if side == "left":
        out = np.zeros((h, w + tab_w), dtype=bool)
        out[:, tab_w:] = mask
        out[start_r : start_r + tab_h, :tab_w] = True
        return out, (0, tab_w)
    out = np.zeros((h, w + tab_w), dtype=bool)
    out[:, :w] = mask
    out[start_r : start_r + tab_h, w:] = True
    return out, (0, 0)


def _build_object(
    color: int,
    marker_color: int,
    side: str,
    marker_outside: bool,
) -> tuple[np.ndarray, np.ndarray, np.ndarray]:
    hole_mask = random.choice(SHAPE_LIBRARY)
    hole_mask = _rotate_mask_cw(hole_mask, random.randint(0, 3))
    hh, hw = hole_mask.shape

    top = random.randint(1, 3)
    bottom = random.randint(1, 3)
    left = random.randint(1, 3)
    right = random.randint(1, 3)
    core_h = hh + top + bottom
    core_w = hw + left + right
    solid = np.ones((core_h, core_w), dtype=bool)
    core_origin = (0, 0)

    other_sides = [name for name in SIDE_NAMES if name != side]
    if random.random() < 0.6:
        solid, shift = _add_tab(solid, random.choice(other_sides))
        core_origin = (core_origin[0] + shift[0], core_origin[1] + shift[1])

    input_region = solid.copy()
    hole_r = core_origin[0] + top
    hole_c = core_origin[1] + left
    input_region[hole_r : hole_r + hh, hole_c : hole_c + hw][hole_mask] = False

    solid = np.pad(solid, 1)
    input_region = np.pad(input_region, 1)
    core_origin = (core_origin[0] + 1, core_origin[1] + 1)

    if side == "top":
        edge_c = random.randint(core_origin[1] + 1, core_origin[1] + core_w - 2)
        marker_pos = (core_origin[0] - 1, edge_c) if marker_outside else (core_origin[0], edge_c)
    elif side == "right":
        edge_r = random.randint(core_origin[0] + 1, core_origin[0] + core_h - 2)
        marker_pos = (edge_r, core_origin[1] + core_w) if marker_outside else (edge_r, core_origin[1] + core_w - 1)
    elif side == "bottom":
        edge_c = random.randint(core_origin[1] + 1, core_origin[1] + core_w - 2)
        marker_pos = (core_origin[0] + core_h, edge_c) if marker_outside else (core_origin[0] + core_h - 1, edge_c)
    else:
        edge_r = random.randint(core_origin[0] + 1, core_origin[0] + core_h - 2)
        marker_pos = (edge_r, core_origin[1] - 1) if marker_outside else (edge_r, core_origin[1])

    input_patch = np.zeros(input_region.shape, dtype=int)
    input_patch[input_region] = int(color)
    input_patch[marker_pos] = int(marker_color)

    solved_patch = np.zeros(solid.shape, dtype=int)
    solved_patch[solid] = int(color)

    sprite_mask = np.rot90(hole_mask, SIDE_TO_CW[side])
    sprite_patch = np.zeros(sprite_mask.shape, dtype=int)
    sprite_patch[sprite_mask] = int(color)

    return input_patch, solved_patch, sprite_patch


def _sample_directions(n: int) -> list[str]:
    if n == 4:
        dirs = SIDE_NAMES[:]
        random.shuffle(dirs)
        return dirs
    return random.sample(SIDE_NAMES, n)


def _attempt_generate() -> tuple[np.ndarray, np.ndarray] | None:
    h = random.randint(22, 30)
    w = random.randint(22, 30)
    grid = np.zeros((h, w), dtype=int)
    expected = np.zeros((h, w), dtype=int)
    occupied = np.zeros((h, w), dtype=bool)

    n_objects = random.randint(2, 4)
    palette = list(range(1, 10))
    random.shuffle(palette)
    colors = palette[:n_objects]
    marker_colors = palette[n_objects : 2 * n_objects]
    sides = _sample_directions(n_objects)

    region_specs = []
    sprite_specs = []
    for idx in range(n_objects):
        marker_outside = bool(random.randint(0, 1))
        region_patch, solved_patch, sprite_patch = _build_object(
            colors[idx],
            marker_colors[idx],
            sides[idx],
            marker_outside,
        )
        region_specs.append((region_patch, solved_patch))
        sprite_specs.append(sprite_patch)

    region_specs.sort(key=lambda item: item[0].shape[0] * item[0].shape[1], reverse=True)
    sprite_specs.sort(key=lambda patch: patch.shape[0] * patch.shape[1], reverse=True)

    for region_patch, solved_patch in region_specs:
        placed = False
        for _ in range(200):
            r0 = random.randint(0, h - region_patch.shape[0])
            c0 = random.randint(0, w - region_patch.shape[1])
            if not _reserve_ok(occupied, region_patch.shape, r0, c0, pad=1):
                continue
            _paste_patch(grid, region_patch, r0, c0)
            _paste_patch(expected, solved_patch, r0, c0)
            _reserve_bbox(occupied, region_patch.shape, r0, c0)
            placed = True
            break
        if not placed:
            return None

    for sprite_patch in sprite_specs:
        placed = False
        for _ in range(200):
            r0 = random.randint(0, h - sprite_patch.shape[0])
            c0 = random.randint(0, w - sprite_patch.shape[1])
            if not _reserve_ok(occupied, sprite_patch.shape, r0, c0, pad=1):
                continue
            _paste_patch(grid, sprite_patch, r0, c0)
            _reserve_bbox(occupied, sprite_patch.shape, r0, c0)
            placed = True
            break
        if not placed:
            return None

    try:
        actual = main(grid)
    except AssertionError:
        return None
    if not np.array_equal(actual, expected):
        return None
    return grid, expected


def generate_input() -> np.ndarray:
    for _ in range(400):
        built = _attempt_generate()
        if built is not None:
            return built[0]
    raise RuntimeError("failed to generate a valid input")
