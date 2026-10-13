import sys
from pathlib import Path

import numpy as np

_UTILS = Path("/home/lixinyu/snap/new/solvers")
if str(_UTILS) not in sys.path:
    sys.path.append(str(_UTILS))

from arc_utils import Color, bbox_coords, bbox_h, bbox_w, colors_present, components_mask, get_bg, paste


_RNG = np.random.default_rng()
_PALETTE = [Color.BLUE, Color.GREEN, Color.YELLOW, Color.PINK, Color.ORANGE, Color.TEAL, Color.MAROON]
_MAX_ATTEMPTS = 300

_HOLED_TEMPLATES = {
    "hole1": [(0, 0), (0, 1), (1, 0), (1, 1)],
    "hole2": [(0, 0), (0, 1), (0, 2), (1, 0), (1, 1), (1, 2)],
    "hole3": [(0, 0), (0, 1), (0, 2), (1, 2), (2, 2), (2, 1), (1, 1), (1, 0)],
    "hole9": [(0, 0), (0, 1), (0, 2), (1, 0), (1, 2), (2, 0), (2, 1), (2, 2)],
}

_NONHOLE_TEMPLATES = {
    "open5": [(0, 0), (0, 1), (0, 2), (1, 0), (2, 0), (2, 1), (2, 2)],
    "zig6": [(0, 0), (0, 1), (1, 1), (1, 2), (2, 2), (2, 3)],
    "bend5": [(0, 0), (0, 1), (0, 2), (1, 2), (2, 2)],
}


def _coords_to_mask(coords: np.ndarray) -> tuple[np.ndarray, tuple[int, int, int, int]]:
    pts = np.asarray(coords, dtype=int)
    box = bbox_coords(pts)
    r0, c0, r1, c1 = box
    mask = np.zeros((r1 - r0 + 1, c1 - c0 + 1), dtype=bool)
    mask[pts[:, 0] - r0, pts[:, 1] - c0] = True
    return mask, box


def _densify_sparse_coords(coords: np.ndarray) -> np.ndarray:
    pts = {tuple(map(int, rc)) for rc in np.asarray(coords, dtype=int).tolist()}
    dense = set(pts)
    for r, c in pts:
        if (r, c + 2) in pts:
            dense.add((r, c + 1))
        if (r + 2, c) in pts:
            dense.add((r + 1, c))
    return np.array(sorted(dense), dtype=int)


def _hole_areas(mask: np.ndarray) -> list[int]:
    bg_mask = ~mask
    holes = []
    for comp in components_mask(bg_mask, connectivity=4):
        if (
            np.any(comp[:, 0] == 0)
            or np.any(comp[:, 0] == mask.shape[0] - 1)
            or np.any(comp[:, 1] == 0)
            or np.any(comp[:, 1] == mask.shape[1] - 1)
        ):
            continue
        holes.append(int(comp.shape[0]))
    return sorted(holes)


def _sprite_info(color: int, sparse_coords: np.ndarray) -> dict:
    sparse_coords = np.asarray(sparse_coords, dtype=int)
    sparse_box = bbox_coords(sparse_coords)
    dense_coords = _densify_sparse_coords(sparse_coords)
    dense_mask, dense_box = _coords_to_mask(dense_coords)
    hole_areas = _hole_areas(dense_mask)
    return {
        "color": int(color),
        "sparse_coords": sparse_coords,
        "source_bbox": sparse_box,
        "dense_coords": dense_coords,
        "dense_bbox": dense_box,
        "mask": dense_mask,
        "hole_areas": hole_areas,
        "hole_area": int(sum(hole_areas)),
        "dense_area": int(np.count_nonzero(dense_mask)),
        "bbox_area": int(dense_mask.shape[0] * dense_mask.shape[1]),
    }


def _analyze_grid(grid: np.ndarray) -> dict:
    g = np.asarray(grid, dtype=int)
    bg = get_bg(g)
    red_coords = np.argwhere(g == Color.RED)
    assert red_coords.size > 0, "expected a red sparse frame"
    frame_bbox = bbox_coords(red_coords)
    horizontal = bbox_w(frame_bbox) > bbox_h(frame_bbox)
    sprite_colors = [c for c in colors_present(g, exclude={bg}) if c not in {Color.GREY, Color.RED}]
    sprites = []
    for color in sprite_colors:
        coords = np.argwhere(g == int(color))
        assert coords.size > 0
        sprites.append(_sprite_info(int(color), coords))
    assert sprites, "expected external sprites"
    return {
        "bg": int(bg),
        "frame_bbox": frame_bbox,
        "horizontal": bool(horizontal),
        "sprites": sprites,
    }


def _draw_dense_frame(out: np.ndarray, frame_bbox: tuple[int, int, int, int]) -> None:
    r0, c0, r1, c1 = frame_bbox
    out[r0, c0 : c1 + 1] = Color.RED
    out[r1, c0 : c1 + 1] = Color.RED
    out[r0 : r1 + 1, c0] = Color.RED
    out[r0 : r1 + 1, c1] = Color.RED


def _place_inside_frame(
    out: np.ndarray, sprites: list[dict], frame_bbox: tuple[int, int, int, int], horizontal: bool, bg: int
) -> None:
    r0, c0, r1, c1 = frame_bbox
    inner_h = r1 - r0 - 1
    inner_w = c1 - c0 - 1
    if horizontal:
        total = sum(sprite["mask"].shape[1] for sprite in sprites)
        remaining = inner_w - total
        assert remaining > 0 and remaining % (len(sprites) + 1) == 0
        gap = remaining // (len(sprites) + 1)
        cur_c = c0 + 1 + gap
        for sprite in sprites:
            h, w = sprite["mask"].shape
            cur_r = r0 + 1 + (inner_h - h) // 2
            patch = np.full((h, w), int(bg), dtype=int)
            patch[sprite["mask"]] = int(sprite["color"])
            paste(out, patch, cur_r, cur_c, mask=sprite["mask"], overwrite=True, clip=False)
            cur_c += w + gap
    else:
        total = sum(sprite["mask"].shape[0] for sprite in sprites)
        remaining = inner_h - total
        assert remaining > 0 and remaining % (len(sprites) + 1) == 0
        gap = remaining // (len(sprites) + 1)
        cur_r = r0 + 1 + gap
        for sprite in sprites:
            h, w = sprite["mask"].shape
            cur_c = c0 + 1 + (inner_w - w) // 2
            patch = np.full((h, w), int(bg), dtype=int)
            patch[sprite["mask"]] = int(sprite["color"])
            paste(out, patch, cur_r, cur_c, mask=sprite["mask"], overwrite=True, clip=False)
            cur_r += h + gap


def main(grid: np.ndarray) -> np.ndarray:
    """Densify the sparse sprites, keep only the holed ones, and sort them into the frame by hole area.

    Input structure:
    - The background is black, with a grey two-step lattice on even-even cells.
    - Red pixels form one large sparse rectangular frame on that lattice.
    - Outside the frame are 3-5 differently-colored sparse monochrome sprites whose pixels also lie on the lattice.

    Rule:
    - Densify the red frame and each sprite by filling the midpoint between same-color pixels that are two cells apart in one row or one column.
    - For every densified sprite, detect enclosed black 4-connected holes inside its own bounding box and sum their areas.
    - Discard sprites with no enclosed hole; sort the remaining sprites by total hole area ascending.
    - Clear the frame interior and every source sprite bounding box, keep the outside decoration, redraw the dense red frame, and place the kept densified sprites inside it with equal long-axis gaps and centered short-axis alignment.

    Key insight:
    - The sparse lattice hides topology: only after densification do some sprites become closed boundaries whose hole areas can be measured and used for ordering.
    """
    g = np.asarray(grid, dtype=int)
    info = _analyze_grid(g)
    bg = info["bg"]
    frame_bbox = info["frame_bbox"]
    selected = [sprite for sprite in info["sprites"] if sprite["hole_area"] > 0]
    assert len(selected) >= 1, "expected at least one holed sprite"
    selected.sort(key=lambda sprite: (sprite["hole_area"], sprite["color"]))

    out = g.copy()
    r0, c0, r1, c1 = frame_bbox
    out[r0 + 1 : r1, c0 + 1 : c1] = bg
    out[g == Color.RED] = bg
    for sprite in info["sprites"]:
        sr0, sc0, sr1, sc1 = sprite["source_bbox"]
        out[sr0 : sr1 + 1, sc0 : sc1 + 1] = bg

    _draw_dense_frame(out, frame_bbox)
    _place_inside_frame(out, selected, frame_bbox, info["horizontal"], bg)
    return out


def _template_to_actual(coarse_coords: list[tuple[int, int]], rot: int, flip: bool) -> np.ndarray:
    arr = np.array(coarse_coords, dtype=int)
    mask, _ = _coords_to_mask(arr)
    if flip:
        mask = np.fliplr(mask)
    mask = np.rot90(mask, int(rot))
    coords = np.argwhere(mask)
    return coords * 2


def _build_template(name: str, color: int, holed: bool) -> dict:
    coarse = _HOLED_TEMPLATES[name] if holed else _NONHOLE_TEMPLATES[name]
    coords = _template_to_actual(
        coarse,
        rot=int(_RNG.integers(0, 4)),
        flip=bool(_RNG.integers(0, 2)),
    )
    info = _sprite_info(int(color), coords)
    info["name"] = name
    info["holed"] = bool(holed)
    return info


def _odd_gap(min_gap: int = 1, max_gap: int = 3) -> int:
    options = [gap for gap in range(min_gap, max_gap + 1) if gap % 2 == 1]
    return int(options[int(_RNG.integers(0, len(options)))])


def _even_value(max_value: int) -> int:
    assert max_value >= 0
    return int(2 * _RNG.integers(0, max_value // 2 + 1))


def _pack_band(sprites: list[dict], axis: str) -> dict:
    if not sprites:
        return {"items": [], "span": 0, "starts": []}
    lengths = [sprite["mask"].shape[1] if axis == "x" else sprite["mask"].shape[0] for sprite in sprites]
    gap = _odd_gap(1, 3)
    span = int(sum(lengths) + gap * (len(lengths) - 1))
    starts = [0]
    for length in lengths[:-1]:
        starts.append(starts[-1] + length + gap)
    return {"items": sprites, "span": span, "starts": starts}


def _position_band(grid: np.ndarray, band: dict, fixed: int, axis: str) -> None:
    if not band["items"]:
        return
    total = band["span"]
    if axis == "x":
        offset = _even_value(grid.shape[1] - total)
        for sprite, start in zip(band["items"], band["starts"], strict=True):
            coords = sprite["sparse_coords"] + np.array([fixed, offset + start], dtype=int)
            grid[coords[:, 0], coords[:, 1]] = int(sprite["color"])
    else:
        offset = _even_value(grid.shape[0] - total)
        for sprite, start in zip(band["items"], band["starts"], strict=True):
            coords = sprite["sparse_coords"] + np.array([offset + start, fixed], dtype=int)
            grid[coords[:, 0], coords[:, 1]] = int(sprite["color"])


def _holed_specs() -> list[dict]:
    choice = int(_RNG.integers(0, 3))
    names = ["hole3", "hole9"] if choice == 0 else (["hole1", "hole3", "hole9"] if choice == 1 else ["hole2", "hole3", "hole9"])
    colors = _RNG.choice(_PALETTE, size=len(names), replace=False).tolist()
    return [_build_template(name, color, holed=True) for name, color in zip(names, colors, strict=True)]


def _nonholed_specs(used_colors: set[int]) -> list[dict]:
    count = int(_RNG.integers(1, 3))
    names = _RNG.choice(list(_NONHOLE_TEMPLATES), size=count, replace=False).tolist()
    remaining = [color for color in _PALETTE if color not in used_colors]
    colors = _RNG.choice(remaining, size=count, replace=False).tolist()
    return [_build_template(name, color, holed=False) for name, color in zip(names, colors, strict=True)]


def _frame_dims(holed: list[dict], all_sprites: list[dict], horizontal: bool) -> tuple[int, int]:
    gap = _odd_gap(1, 3)
    center_pad = int(_RNG.integers(1, 3))
    if horizontal:
        inner_w = int(sum(sprite["mask"].shape[1] for sprite in holed) + gap * (len(holed) + 1))
        inner_h = int(max(max(sprite["mask"].shape[0] for sprite in holed) + 2 * center_pad, max(sprite["mask"].shape[0] for sprite in all_sprites)))
    else:
        inner_h = int(sum(sprite["mask"].shape[0] for sprite in holed) + gap * (len(holed) + 1))
        inner_w = int(max(max(sprite["mask"].shape[1] for sprite in holed) + 2 * center_pad, max(sprite["mask"].shape[1] for sprite in all_sprites)))
    frame_h = inner_h + 2
    frame_w = inner_w + 2
    assert frame_h % 2 == 1 and frame_w % 2 == 1
    assert (frame_w > frame_h) if horizontal else (frame_h >= frame_w)
    return frame_h, frame_w


def _band_assignment(holed: list[dict], nonholed: list[dict]) -> tuple[list[dict], list[dict]]:
    hole_order = sorted(holed, key=lambda sprite: sprite["hole_area"])
    perm = hole_order.copy()
    while [sprite["color"] for sprite in perm] == [sprite["color"] for sprite in hole_order]:
        perm = list(_RNG.permutation(hole_order))
    main_band = list(perm)
    other_band = list(nonholed)
    if nonholed:
        insert_sprite = other_band.pop(0)
        insert_at = int(_RNG.integers(0, len(main_band) + 1))
        main_band.insert(insert_at, insert_sprite)
    return main_band, other_band


def _paint_sparse_frame(grid: np.ndarray, frame_bbox: tuple[int, int, int, int]) -> None:
    r0, c0, r1, c1 = frame_bbox
    grid[r0, c0 : c1 + 1 : 2] = Color.RED
    grid[r1, c0 : c1 + 1 : 2] = Color.RED
    grid[r0 : r1 + 1 : 2, c0] = Color.RED
    grid[r0 : r1 + 1 : 2, c1] = Color.RED


def _generate_attempt() -> np.ndarray | None:
    holed = _holed_specs()
    used_colors = {sprite["color"] for sprite in holed}
    nonholed = _nonholed_specs(used_colors)
    all_sprites = holed + nonholed
    horizontal = bool(_RNG.integers(0, 2))
    frame_h, frame_w = _frame_dims(holed, all_sprites, horizontal)
    main_band_items, other_band_items = _band_assignment(holed, nonholed)

    if horizontal:
        main_band = _pack_band(main_band_items, "x")
        other_band = _pack_band(other_band_items, "x")
        top_is_main = bool(_RNG.integers(0, 2))
        top_band = main_band if top_is_main else other_band
        bottom_band = other_band if top_is_main else main_band
        top_h = max((sprite["mask"].shape[0] for sprite in top_band["items"]), default=0)
        bottom_h = max((sprite["mask"].shape[0] for sprite in bottom_band["items"]), default=0)
        frame_r0 = (top_h + 1 + _even_value(2)) if top_band["items"] else _even_value(2)
        bottom_r0 = frame_r0 + frame_h + (1 + _even_value(2))
        total_h = bottom_r0 + bottom_h if bottom_band["items"] else frame_r0 + frame_h + _even_value(2)
        width_need = max(frame_w, top_band["span"], bottom_band["span"])
        total_w = width_need + _even_value(6)
        if total_h > 30 or total_w > 30:
            return None
        grid = np.zeros((total_h, total_w), dtype=int)
        grid[::2, ::2] = Color.GREY
        frame_c0 = _even_value(total_w - frame_w)
        frame_bbox = (frame_r0, frame_c0, frame_r0 + frame_h - 1, frame_c0 + frame_w - 1)
        _paint_sparse_frame(grid, frame_bbox)
        if top_band["items"]:
            _position_band(grid, top_band, 0, "x")
        if bottom_band["items"]:
            _position_band(grid, bottom_band, bottom_r0, "x")
    else:
        main_band = _pack_band(main_band_items, "y")
        other_band = _pack_band(other_band_items, "y")
        left_is_main = bool(_RNG.integers(0, 2))
        left_band = main_band if left_is_main else other_band
        right_band = other_band if left_is_main else main_band
        left_w = max((sprite["mask"].shape[1] for sprite in left_band["items"]), default=0)
        right_w = max((sprite["mask"].shape[1] for sprite in right_band["items"]), default=0)
        frame_c0 = (left_w + 1 + _even_value(2)) if left_band["items"] else _even_value(2)
        right_c0 = frame_c0 + frame_w + (1 + _even_value(2))
        total_w = right_c0 + right_w if right_band["items"] else frame_c0 + frame_w + _even_value(2)
        height_need = max(frame_h, left_band["span"], right_band["span"])
        total_h = height_need + _even_value(6)
        if total_h > 30 or total_w > 30:
            return None
        grid = np.zeros((total_h, total_w), dtype=int)
        grid[::2, ::2] = Color.GREY
        frame_r0 = _even_value(total_h - frame_h)
        frame_bbox = (frame_r0, frame_c0, frame_r0 + frame_h - 1, frame_c0 + frame_w - 1)
        _paint_sparse_frame(grid, frame_bbox)
        if left_band["items"]:
            _position_band(grid, left_band, 0, "y")
        if right_band["items"]:
            _position_band(grid, right_band, right_c0, "y")

    return grid


def _validate_generated(grid: np.ndarray) -> bool:
    info = _analyze_grid(grid)
    selected = [sprite for sprite in info["sprites"] if sprite["hole_area"] > 0]
    rejected = [sprite for sprite in info["sprites"] if sprite["hole_area"] == 0]
    if len(selected) < 2 or len(rejected) < 1:
        return False
    hole_areas = [sprite["hole_area"] for sprite in selected]
    if len(set(hole_areas)) != len(hole_areas):
        return False

    hole_order = [sprite["color"] for sprite in sorted(selected, key=lambda sprite: sprite["hole_area"])]
    axis_key = 1 if info["horizontal"] else 0
    source_order = [sprite["color"] for sprite in sorted(selected, key=lambda sprite: sprite["source_bbox"][axis_key])]
    if source_order == hole_order:
        return False

    dense_keys = [sprite["dense_area"] for sprite in selected]
    if len(set(dense_keys)) == len(dense_keys):
        dense_order = [sprite["color"] for sprite in sorted(selected, key=lambda sprite: sprite["dense_area"])]
        if dense_order == hole_order:
            return False

    bbox_keys = [sprite["bbox_area"] for sprite in selected]
    if len(set(bbox_keys)) == len(bbox_keys):
        bbox_order = [sprite["color"] for sprite in sorted(selected, key=lambda sprite: sprite["bbox_area"])]
        if bbox_order == hole_order:
            return False

    frame_r0, frame_c0, frame_r1, frame_c1 = info["frame_bbox"]
    inner_h = frame_r1 - frame_r0 - 1
    inner_w = frame_c1 - frame_c0 - 1
    if any(sprite["mask"].shape[0] > inner_h or sprite["mask"].shape[1] > inner_w for sprite in info["sprites"]):
        return False

    out = main(grid)
    if out.ndim != 2 or max(out.shape) > 30:
        return False
    if np.array_equal(out, grid):
        return False
    return True


def generate_input() -> np.ndarray:
    for _ in range(_MAX_ATTEMPTS):
        grid = _generate_attempt()
        if grid is not None and _validate_generated(grid):
            return grid
    raise RuntimeError("generate_input failed")
