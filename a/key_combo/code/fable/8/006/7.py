from __future__ import annotations

from importlib.machinery import SourceFileLoader
import random

import numpy as np


_ARC = SourceFileLoader(
    "_arc_utils_shared",
    "/home/lixinyu/snap/new/solvers/arc_utils.py",
).load_module()

bbox_coords = _ARC.bbox_coords
components_mask = _ARC.components_mask
crop_bbox = _ARC.crop_bbox
get_bg = _ARC.get_bg
paste = _ARC.paste


def _bbox_center(box: tuple[int, int, int, int]) -> tuple[int, int]:
    r0, c0, r1, c1 = map(int, box)
    return (r0 + r1) // 2, (c0 + c1) // 2


def _legend_info(grid: np.ndarray, bg: int) -> tuple[np.ndarray, list[tuple[int, int]], tuple[int, int, int, int]]:
    mask = grid != bg
    candidates: list[tuple[np.ndarray, tuple[int, int, int, int], str]] = []
    for coords in components_mask(mask, connectivity=4):
        if not any(
            r == 0 or c == 0 or r == grid.shape[0] - 1 or c == grid.shape[1] - 1
            for r, c in coords
        ):
            continue
        box = bbox_coords(coords)
        r0, c0, r1, c1 = box
        h = r1 - r0 + 1
        w = c1 - c0 + 1
        side = None
        if r0 == 0 or r1 == grid.shape[0] - 1:
            if h == 2 and w >= 2:
                side = "horizontal"
        elif c0 == 0 or c1 == grid.shape[1] - 1:
            if w == 2 and h >= 2:
                side = "vertical"
        if side is None:
            continue
        sub = crop_bbox(grid, box)
        if np.any(sub == bg):
            continue
        candidates.append((coords, box, side))

    assert len(candidates) == 1, "expected exactly one edge legend"
    coords, box, side = candidates[0]
    legend_mask = np.zeros(grid.shape, dtype=bool)
    legend_mask[coords[:, 0], coords[:, 1]] = True

    sub = crop_bbox(grid, box)
    pairs: list[tuple[int, int]] = []
    if side == "horizontal":
        for c in range(sub.shape[1]):
            pairs.append((int(sub[0, c]), int(sub[1, c])))
    else:
        for r in range(sub.shape[0]):
            pairs.append((int(sub[r, 0]), int(sub[r, 1])))

    lefts = [a for a, _b in pairs]
    assert len(lefts) == len(set(lefts)), "legend source colors must be unique"
    return legend_mask, pairs, box


def _object_records(grid: np.ndarray, bg: int, legend_mask: np.ndarray) -> dict[int, dict]:
    colors = [int(c) for c in np.unique(grid) if int(c) != bg]
    records: dict[int, dict] = {}
    for color in colors:
        comps = components_mask(grid == color, connectivity=4)
        outside = []
        for coords in comps:
            inside_legend = bool(np.all(legend_mask[coords[:, 0], coords[:, 1]]))
            if not inside_legend:
                outside.append(coords)
        if not outside:
            continue
        assert len(outside) == 1, f"color {color} must have exactly one object"
        coords = outside[0]
        box = bbox_coords(coords)
        r0, c0, r1, c1 = box
        assert (r1 - r0 + 1) % 2 == 1 and (c1 - c0 + 1) % 2 == 1, "object bbox must be odd-sized"
        sprite = crop_bbox(grid == color, box)
        assert sprite.any(), "object sprite must be non-empty"
        center = _bbox_center(box)
        rel = np.argwhere(sprite) - np.array([sprite.shape[0] // 2, sprite.shape[1] // 2], dtype=int)
        records[color] = {
            "coords": coords,
            "bbox": box,
            "center": center,
            "sprite": sprite,
            "rel": rel.astype(int),
        }
    return records


def _paint_relative_shape(
    out: np.ndarray,
    color: int,
    center: tuple[int, int],
    rel_coords: np.ndarray,
    protected: np.ndarray,
) -> None:
    claimed: set[tuple[int, int]] = set()
    cr, cc = map(int, center)
    for dr, dc in rel_coords:
        rr = cr + int(dr)
        cc2 = cc + int(dc)
        assert 0 <= rr < out.shape[0] and 0 <= cc2 < out.shape[1], "redrawn shape out of bounds"
        assert not protected[rr, cc2], "redrawn shape may not overlap legend or preserved object"
        key = (rr, cc2)
        assert key not in claimed, "duplicate target cell"
        claimed.add(key)
    for rr, cc2 in claimed:
        out[rr, cc2] = int(color)


def main(grid: np.ndarray) -> np.ndarray:
    """
    Replace each legend-selected sprite with another sprite's shape while keeping its own color and center.

    Input structure:
    - The background is the dominant color.
    - Inside the grid are several pairwise color-distinct solid sprite components with pairwise distinct odd-sized bounding boxes.
    - One fully occupied 2×N or N×2 multicolor legend lies on a grid edge and lists color pairs.

    Rule:
    - Read each legend pair (a, b): erase the object of color a, then redraw color a using the original shape of color b.
    - Anchor the transferred shape at the center of a's original bounding box, using the input snapshot for every donor shape even if b is also remapped.
    - Keep every object whose color never appears on the left side unchanged, and preserve the legend exactly.

    Key insight:
    - The edge legend is not a fill palette; its right entries donate shapes, and all remappings must be computed simultaneously from the old scene.
    """
    g = np.asarray(grid, dtype=int)
    assert g.ndim == 2, "expected a 2D grid"

    bg = int(get_bg(g))
    legend_mask, pairs, _legend_box = _legend_info(g, bg)
    objects = _object_records(g, bg, legend_mask)
    mapping = {a: b for a, b in pairs}

    assert all(a in objects and b in objects for a, b in pairs), "legend colors must correspond to scene objects"

    out = g.copy()
    for a in mapping:
        coords = objects[a]["coords"]
        out[coords[:, 0], coords[:, 1]] = bg

    protected = (out != bg)
    for a, b in pairs:
        donor = objects[b]
        target = objects[a]
        _paint_relative_shape(out, a, target["center"], donor["rel"], protected)
        protected = (out != bg)

    return out


def _shape(rows: list[str]) -> np.ndarray:
    return np.array([[ch == "1" for ch in row] for row in rows], dtype=bool)


_SHAPE_LIBRARY = [
    _shape(["010", "111", "010"]),
    _shape(["111", "010", "011"]),
    _shape(["110", "011", "010"]),
    _shape(["010", "111", "001"]),
    _shape(["00100", "01110", "11111", "01110", "00100"]),
    _shape(["11111", "00100", "00100", "00100", "00100"]),
    _shape(["11000", "11100", "01110", "00111", "00011"]),
    _shape(["10101", "11111", "00100", "00100", "00100"]),
    _shape(["10000", "11000", "11100", "01110", "00111"]),
    _shape(["00100", "01110", "11111", "00110", "00110"]),
]

_GRID_MIN = 22
_GRID_MAX = 30
_OBJECT_COUNT_RANGE = (4, 6)
_MARGIN = 1
_PLACEMENT_GAP = 1
_GEN_RETRIES = 400


def _orient_shape(mask: np.ndarray) -> np.ndarray:
    out = np.array(mask, copy=True)
    if random.random() < 0.5:
        out = np.fliplr(out)
    k = random.randrange(4)
    out = np.rot90(out, k)
    return np.array(out, copy=True)


def _select_shapes(count: int) -> list[np.ndarray] | None:
    for _ in range(200):
        bases = random.sample(_SHAPE_LIBRARY, count)
        shapes = [_orient_shape(base) for base in bases]
        if any(len(components_mask(shape, connectivity=4)) != 1 for shape in shapes):
            continue
        keys = {(tuple(shape.shape), shape.tobytes()) for shape in shapes}
        if len(keys) == count:
            return shapes
    return None


def _build_mapping(colors: list[int], shapes_by_color: dict[int, np.ndarray]) -> dict[int, int] | None:
    mapped_count = random.randint(2, len(colors) - 1)
    mapped = random.sample(colors, mapped_count)
    chain_len = 2 if mapped_count == 2 else random.choice([2, 3])
    chain = random.sample(mapped, chain_len)
    mapping: dict[int, int] = {}
    for idx, color in enumerate(chain):
        mapping[color] = chain[(idx + 1) % chain_len]
    for color in mapped:
        if color in mapping:
            continue
        donors = [other for other in colors if other != color]
        random.shuffle(donors)
        mapping[color] = donors[0]

    if len(mapping) >= len(colors):
        return None
    if not any(donor in mapping for donor in mapping.values()):
        return None
    if not any(shapes_by_color[a].shape != shapes_by_color[b].shape for a, b in mapping.items()):
        return None
    return mapping


def _dilate(mask: np.ndarray, radius: int) -> np.ndarray:
    out = np.array(mask, copy=True)
    coords = np.argwhere(mask)
    for r, c in coords:
        for dr in range(-radius, radius + 1):
            for dc in range(-radius, radius + 1):
                rr = int(r) + dr
                cc = int(c) + dc
                if 0 <= rr < mask.shape[0] and 0 <= cc < mask.shape[1]:
                    out[rr, cc] = True
    return out


def _abs_coords(mask: np.ndarray, r0: int, c0: int) -> list[tuple[int, int]]:
    return [(int(r0 + r), int(c0 + c)) for r, c in np.argwhere(mask)]


def _place_objects(
    shapes_by_color: dict[int, np.ndarray],
    mapping: dict[int, int],
    grid_shape: tuple[int, int],
    reserved: np.ndarray,
) -> dict[int, dict] | None:
    input_forbidden = _dilate(reserved, _PLACEMENT_GAP)
    output_forbidden = _dilate(reserved, _PLACEMENT_GAP)
    placements: dict[int, dict] = {}

    def output_shape(color: int) -> np.ndarray:
        donor = mapping.get(color, color)
        return shapes_by_color[donor]

    order = sorted(
        shapes_by_color,
        key=lambda color: int(shapes_by_color[color].sum() + output_shape(color).sum()),
        reverse=True,
    )

    for color in order:
        in_shape = shapes_by_color[color]
        out_shape = output_shape(color)
        in_h, in_w = map(int, in_shape.shape)
        out_h, out_w = map(int, out_shape.shape)
        candidates: list[tuple[int, int]] = []

        for r0 in range(_MARGIN, grid_shape[0] - in_h - _MARGIN + 1):
            for c0 in range(_MARGIN, grid_shape[1] - in_w - _MARGIN + 1):
                center_r = r0 + in_h // 2
                center_c = c0 + in_w // 2
                out_r0 = center_r - out_h // 2
                out_c0 = center_c - out_w // 2
                if out_r0 < _MARGIN or out_c0 < _MARGIN:
                    continue
                if out_r0 + out_h > grid_shape[0] - _MARGIN or out_c0 + out_w > grid_shape[1] - _MARGIN:
                    continue

                in_coords = _abs_coords(in_shape, r0, c0)
                out_coords = _abs_coords(out_shape, out_r0, out_c0)
                if any(input_forbidden[r, c] for r, c in in_coords):
                    continue
                if any(output_forbidden[r, c] for r, c in out_coords):
                    continue
                candidates.append((r0, c0))

        random.shuffle(candidates)
        if not candidates:
            return None

        placed = False
        for r0, c0 in candidates:
            center_r = r0 + in_h // 2
            center_c = c0 + in_w // 2
            out_r0 = center_r - out_h // 2
            out_c0 = center_c - out_w // 2
            in_patch = np.zeros(grid_shape, dtype=bool)
            out_patch = np.zeros(grid_shape, dtype=bool)
            for r, c in _abs_coords(in_shape, r0, c0):
                in_patch[r, c] = True
            for r, c in _abs_coords(out_shape, out_r0, out_c0):
                out_patch[r, c] = True
            if np.any(input_forbidden & in_patch) or np.any(output_forbidden & out_patch):
                continue
            placements[color] = {
                "input_shape": in_shape,
                "output_shape": out_shape,
                "r0": int(r0),
                "c0": int(c0),
                "out_r0": int(out_r0),
                "out_c0": int(out_c0),
            }
            input_forbidden |= _dilate(in_patch, _PLACEMENT_GAP)
            output_forbidden |= _dilate(out_patch, _PLACEMENT_GAP)
            placed = True
            break
        if not placed:
            return None

    return placements


def _legend_start(grid_shape: tuple[int, int], pair_count: int, side: str) -> int | None:
    h, w = grid_shape
    if side in {"top", "bottom"}:
        if w < pair_count + 2:
            return None
        return random.randint(1, w - pair_count - 1)
    if h < pair_count + 2:
        return None
    return random.randint(1, h - pair_count - 1)


def _place_legend(
    grid: np.ndarray,
    pairs: list[tuple[int, int]],
    side: str,
    start: int,
) -> np.ndarray:
    legend = np.zeros_like(grid, dtype=bool)
    pair_count = len(pairs)
    h, w = grid.shape
    if side in {"top", "bottom"}:
        c0 = int(start)
        r0 = 0 if side == "top" else h - 2
        for idx, (a, b) in enumerate(pairs):
            grid[r0, c0 + idx] = int(a)
            grid[r0 + 1, c0 + idx] = int(b)
            legend[r0 : r0 + 2, c0 + idx] = True
    else:
        r0 = int(start)
        c0 = 0 if side == "left" else w - 2
        for idx, (a, b) in enumerate(pairs):
            grid[r0 + idx, c0] = int(a)
            grid[r0 + idx, c0 + 1] = int(b)
            legend[r0 + idx, c0 : c0 + 2] = True
    return legend


def _render_input(
    bg: int,
    grid_shape: tuple[int, int],
    legend_pairs: list[tuple[int, int]],
    side: str,
    legend_start: int,
    placements: dict[int, dict],
) -> tuple[np.ndarray, np.ndarray]:
    grid = np.full(grid_shape, int(bg), dtype=int)
    legend_mask = _place_legend(grid, legend_pairs, side, legend_start)
    for color, info in placements.items():
        patch = np.full(info["input_shape"].shape, int(bg), dtype=int)
        patch[info["input_shape"]] = int(color)
        paste(grid, patch, info["r0"], info["c0"], transparent=bg)
    return grid, legend_mask


def _expected_output(
    grid: np.ndarray,
    bg: int,
    legend_mask: np.ndarray,
    mapping: dict[int, int],
    placements: dict[int, dict],
) -> np.ndarray:
    out = np.array(grid, copy=True)
    for color in mapping:
        patch = np.full(placements[color]["input_shape"].shape, int(bg), dtype=int)
        patch[placements[color]["input_shape"]] = int(bg)
        paste(out, patch, placements[color]["r0"], placements[color]["c0"])

    protected = (out != bg) | legend_mask
    for color, donor in mapping.items():
        info = placements[color]
        out_shape = placements[donor]["input_shape"]
        center_r = info["r0"] + info["input_shape"].shape[0] // 2
        center_c = info["c0"] + info["input_shape"].shape[1] // 2
        out_r0 = center_r - out_shape.shape[0] // 2
        out_c0 = center_c - out_shape.shape[1] // 2
        for rr, cc in _abs_coords(out_shape, out_r0, out_c0):
            assert not protected[rr, cc], "expected output collision"
            out[rr, cc] = int(color)
            protected[rr, cc] = True
    return out


def _validate_instance(
    grid: np.ndarray,
    bg: int,
    legend_pairs: list[tuple[int, int]],
    legend_mask: np.ndarray,
    mapping: dict[int, int],
    placements: dict[int, dict],
) -> bool:
    parsed_legend_mask, parsed_pairs, _ = _legend_info(grid, bg)
    if not np.array_equal(parsed_legend_mask, legend_mask):
        return False
    if parsed_pairs != legend_pairs:
        return False

    expected = _expected_output(grid, bg, legend_mask, mapping, placements)
    actual = main(grid)
    if not np.array_equal(expected, actual):
        return False
    if np.array_equal(grid, actual):
        return False
    if len(mapping) >= len(placements):
        return False
    if not any(donor in mapping for donor in mapping.values()):
        return False
    if not any(
        placements[a]["input_shape"].shape != placements[b]["input_shape"].shape
        for a, b in mapping.items()
    ):
        return False

    objects = _object_records(grid, bg, legend_mask)
    if set(objects) != set(placements):
        return False
    return True


def _gen_attempt() -> np.ndarray | None:
    h = random.randint(_GRID_MIN, _GRID_MAX)
    w = random.randint(_GRID_MIN, _GRID_MAX)
    bg = random.randint(0, 9)

    object_count = random.randint(*_OBJECT_COUNT_RANGE)
    colors = random.sample([c for c in range(10) if c != bg], object_count)
    shapes = _select_shapes(object_count)
    if shapes is None:
        return None
    shapes_by_color = {color: shape for color, shape in zip(colors, shapes, strict=False)}

    mapping = _build_mapping(colors, shapes_by_color)
    if mapping is None:
        return None

    side = random.choice(["top", "bottom", "left", "right"])
    legend_pairs = [(a, mapping[a]) for a in random.sample(list(mapping), len(mapping))]
    legend_start = _legend_start((h, w), len(legend_pairs), side)
    if legend_start is None:
        return None
    reserved_grid = np.full((h, w), int(bg), dtype=int)
    legend_mask = _place_legend(reserved_grid, legend_pairs, side, legend_start)

    placements = _place_objects(shapes_by_color, mapping, (h, w), legend_mask)
    if placements is None:
        return None

    grid, legend_mask2 = _render_input(bg, (h, w), legend_pairs, side, legend_start, placements)
    if not np.array_equal(legend_mask, legend_mask2):
        return None

    if not _validate_instance(grid, bg, legend_pairs, legend_mask, mapping, placements):
        return None
    return grid


def generate_input() -> np.ndarray:
    for _ in range(_GEN_RETRIES):
        grid = _gen_attempt()
        if grid is not None:
            return grid
    raise RuntimeError("failed to generate a valid input")
