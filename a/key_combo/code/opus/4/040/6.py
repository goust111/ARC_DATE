from __future__ import annotations

import random
import sys
from itertools import combinations
from pathlib import Path

import numpy as np


_ARC_UTILS_DIR = Path("/home/lixinyu/snap/new/solvers")
if _ARC_UTILS_DIR.exists():
    sys.path.append(str(_ARC_UTILS_DIR))

from arc_utils import Color, bbox_coords, crop_bbox, paste


Coord = tuple[int, int]
BBox = tuple[int, int, int, int]


FRAME_COLORS = (Color.BLUE, Color.RED, Color.GREEN)
MAX_GENERATION_ATTEMPTS = 400
CANVAS_MIN = 22
CANVAS_MAX = 30


def _bbox_h(box: BBox) -> int:
    return int(box[2] - box[0] + 1)


def _bbox_w(box: BBox) -> int:
    return int(box[3] - box[1] + 1)


def _boundary_cells(box: BBox) -> list[Coord]:
    r0, c0, r1, c1 = box
    cells = [(r0, c) for c in range(c0, c1 + 1)]
    cells += [(r1, c) for c in range(c0, c1 + 1)]
    cells += [(r, c0) for r in range(r0 + 1, r1)]
    cells += [(r, c1) for r in range(r0 + 1, r1)]
    return cells


def _detect_side(markers: list[Coord], box: BBox) -> str:
    (r0, c0), (r1, c1) = markers
    br0, bc0, br1, bc1 = box
    if r0 == r1 == br0:
        return "top"
    if r0 == r1 == br1:
        return "bottom"
    if c0 == c1 == bc0:
        return "left"
    if c0 == c1 == bc1:
        return "right"
    raise AssertionError("markers must lie on one side of the frame")


def _extract_frames(grid: np.ndarray) -> list[dict]:
    g = np.asarray(grid, dtype=int)
    colors = [int(c) for c in np.unique(g) if int(c) not in {Color.BLACK, Color.YELLOW, Color.GREY}]
    assert set(colors) == set(FRAME_COLORS), "expected exactly three frame colors"

    frames: list[dict] = []
    for color in colors:
        coords = np.argwhere(g == color)
        assert coords.size > 0
        box = bbox_coords(coords)
        r0, c0, r1, c1 = box
        assert r1 - r0 >= 2 and c1 - c0 >= 2

        boundary = _boundary_cells(box)
        boundary_vals = [int(g[r, c]) for r, c in boundary]
        assert all(v in {color, Color.YELLOW} for v in boundary_vals)
        markers = [(r, c) for r, c in boundary if int(g[r, c]) == Color.YELLOW]
        assert len(markers) == 2
        side = _detect_side(markers, box)

        interior_box = (r0 + 1, c0 + 1, r1 - 1, c1 - 1)
        interior = crop_bbox(g, interior_box)
        assert set(map(int, np.unique(interior))).issubset({Color.BLACK, Color.GREY})

        if side in {"top", "bottom"}:
            cols = sorted(c for _r, c in markers)
            assert cols[1] - cols[0] >= 2
            crop_box = (r0 + 1, cols[0] + 1, r1 - 1, cols[1] - 1)
        else:
            rows = sorted(r for r, _c in markers)
            assert rows[1] - rows[0] >= 2
            crop_box = (rows[0] + 1, c0 + 1, rows[1] - 1, c1 - 1)

        crop = crop_bbox(g, crop_box)
        local_crop_box = (
            crop_box[0] - interior_box[0],
            crop_box[1] - interior_box[1],
            crop_box[2] - interior_box[0],
            crop_box[3] - interior_box[1],
        )

        frames.append(
            {
                "color": color,
                "box": box,
                "interior_box": interior_box,
                "interior": interior,
                "interior_shape": interior.shape,
                "interior_area": int(interior.size),
                "markers": markers,
                "side": side,
                "crop_box": crop_box,
                "crop_local_box": local_crop_box,
                "crop": crop,
                "crop_area": int(crop.size),
            }
        )

    assert len(frames) == 3
    return frames


def _repeat_to_shape(tile: np.ndarray, shape: tuple[int, int]) -> np.ndarray:
    h, w = map(int, shape)
    th, tw = tile.shape
    reps = ((h + th - 1) // th, (w + tw - 1) // tw)
    return np.tile(tile, reps)[:h, :w]


def _overlay_template(base: np.ndarray, template: np.ndarray) -> np.ndarray:
    out = np.array(base, copy=True)
    tiled = _repeat_to_shape(template, base.shape)
    out[tiled == Color.GREY] = Color.GREY
    return out


def main(grid: np.ndarray) -> np.ndarray:
    """
    Crop each marked frame by its marker span, then tile the smallest crop transparently across the larger crops.

    Input structure:
    - A black background contains exactly three non-overlapping hollow rectangular frames in blue, red, and green.
    - Each frame border has exactly two yellow marker pixels on one side, and the frame interior contains only black and gray pixels.
    - The two markers define either a horizontal span across columns or a vertical span across rows inside that frame.

    Rule:
    - For each frame, use the marker side to crop one interior subgrid: horizontal-side markers keep all interior rows and only the columns strictly between the markers; vertical-side markers keep all interior columns and only the rows strictly between the markers.
    - Sort the three cropped subgrids by area; keep the smallest one unchanged as the template.
    - For each larger crop, periodically repeat the template from that crop's upper-left corner and write only the template's gray pixels back into the crop, leaving template-black positions unchanged.

    Key insight:
    - The frame size does not determine the template; the decisive information is the marker-defined crop span, which must be extracted before the area ranking and transparent tiling can be resolved.
    """
    g = np.asarray(grid, dtype=int)
    assert g.ndim == 2 and g.size > 0

    frames = _extract_frames(g)
    areas = [frame["crop_area"] for frame in frames]
    assert len(set(areas)) == 3
    frames.sort(key=lambda frame: frame["crop_area"])
    template = frames[0]["crop"]

    out = g.copy()
    for frame in frames[1:]:
        filled = _overlay_template(frame["crop"], template)
        paste(out, filled, frame["crop_box"][0], frame["crop_box"][1])
    return out


def _valid_template_shapes() -> list[tuple[int, int]]:
    out: list[tuple[int, int]] = []
    for h in range(2, 5):
        for w in range(2, 5):
            larger = []
            for kh in range(1, 8 // h + 1):
                for kw in range(1, 8 // w + 1):
                    if (kh, kw) == (1, 1):
                        continue
                    shape = (h * kh, w * kw)
                    if _allowed_orientations(shape, require_outside=True):
                        larger.append(shape)
            if len({shape[0] * shape[1] for shape in larger}) < 2:
                continue
            if _allowed_orientations((h, w), require_outside=False):
                out.append((h, w))
    return out


def _allowed_orientations(crop_shape: tuple[int, int], *, require_outside: bool) -> list[str]:
    ch, cw = crop_shape
    options: list[str] = []
    if 4 <= ch <= 8:
        min_iw = max(4, cw + (1 if require_outside else 0))
        if min_iw <= 8:
            options.append("H")
    if 4 <= cw <= 8:
        min_ih = max(4, ch + (1 if require_outside else 0))
        if min_ih <= 8:
            options.append("V")
    return options


TEMPLATE_SHAPES = _valid_template_shapes()


def _pick_crop_shapes() -> list[tuple[int, int]]:
    template = random.choice(TEMPLATE_SHAPES)
    th, tw = template
    larger = []
    for kh in range(1, 8 // th + 1):
        for kw in range(1, 8 // tw + 1):
            if (kh, kw) == (1, 1):
                continue
            shape = (th * kh, tw * kw)
            if _allowed_orientations(shape, require_outside=True):
                larger.append(shape)

    pairs = [
        pair
        for pair in combinations(larger, 2)
        if pair[0][0] * pair[0][1] != pair[1][0] * pair[1][1]
    ]
    assert pairs
    first, second = random.choice(pairs)
    return [template, first, second]


def _build_spec(crop_shape: tuple[int, int], *, require_outside: bool) -> dict:
    ch, cw = crop_shape
    orientation = random.choice(_allowed_orientations(crop_shape, require_outside=require_outside))
    side = random.choice(["top", "bottom"] if orientation == "H" else ["left", "right"])

    if orientation == "H":
        ih = ch
        iw = random.randint(max(4, cw + (1 if require_outside else 0)), 8)
        extra = iw - cw
        left_margin = random.randint(0, extra)
        right_margin = extra - left_margin
        marker_a = left_margin
        marker_b = marker_a + cw + 1
        crop_local_box = (0, left_margin, ih - 1, left_margin + cw - 1)
        side_margins = (left_margin, right_margin)
    else:
        iw = cw
        ih = random.randint(max(4, ch + (1 if require_outside else 0)), 8)
        extra = ih - ch
        top_margin = random.randint(0, extra)
        bottom_margin = extra - top_margin
        marker_a = top_margin
        marker_b = marker_a + ch + 1
        crop_local_box = (top_margin, 0, top_margin + ch - 1, iw - 1)
        side_margins = (top_margin, bottom_margin)

    return {
        "crop_shape": crop_shape,
        "crop_area": int(ch * cw),
        "orientation": orientation,
        "side": side,
        "interior_shape": (ih, iw),
        "interior_area": int(ih * iw),
        "outer_shape": (ih + 2, iw + 2),
        "marker_pair": (marker_a, marker_b),
        "crop_local_box": crop_local_box,
        "side_margins": side_margins,
        "require_outside": require_outside,
    }


def _crop_mask(shape: tuple[int, int], crop_local_box: BBox) -> np.ndarray:
    mask = np.zeros(shape, dtype=bool)
    r0, c0, r1, c1 = crop_local_box
    mask[r0 : r1 + 1, c0 : c1 + 1] = True
    return mask


def _random_cells(cells: list[Coord], count: int) -> list[Coord]:
    count = min(count, len(cells))
    return random.sample(cells, count) if count > 0 else []


def _make_template_pattern(shape: tuple[int, int]) -> np.ndarray:
    h, w = shape
    area = h * w
    min_gray = 1 if area == 2 else 2
    gray_count = random.randint(min_gray, min(4, area - 1))
    cells = [(r, c) for r in range(h) for c in range(w)]
    pattern = np.zeros((h, w), dtype=int)
    for r, c in random.sample(cells, gray_count):
        pattern[r, c] = Color.GREY
    assert np.any(pattern == Color.GREY) and np.any(pattern == Color.BLACK)
    return pattern


def _build_interior(spec: dict, template_pattern: np.ndarray) -> np.ndarray:
    ih, iw = spec["interior_shape"]
    crop_local_box = spec["crop_local_box"]
    crop_mask = _crop_mask((ih, iw), crop_local_box)
    interior = np.zeros((ih, iw), dtype=int)
    ch, cw = spec["crop_shape"]
    cr0, cc0, cr1, cc1 = crop_local_box

    if not spec["require_outside"]:
        paste(interior, template_pattern, cr0, cc0)
        outside = list(map(tuple, np.argwhere(~crop_mask)))
        extra = random.randint(0, min(1, max(0, 6 - int(np.sum(interior == Color.GREY)))))
        for r, c in _random_cells(outside, extra):
            interior[r, c] = Color.GREY
        return interior

    tiled = _repeat_to_shape(template_pattern, (ch, cw))
    zero_phase = [tuple(map(int, rc)) for rc in np.argwhere(tiled == Color.BLACK)]
    one_phase = [tuple(map(int, rc)) for rc in np.argwhere(tiled == Color.GREY)]
    assert zero_phase and one_phase

    survivor_r, survivor_c = random.choice(zero_phase)
    interior[cr0 + survivor_r, cc0 + survivor_c] = Color.GREY

    outside = list(map(tuple, np.argwhere(~crop_mask)))
    assert outside
    outside_r, outside_c = random.choice(outside)
    interior[outside_r, outside_c] = Color.GREY

    spare_inside = [
        (cr0 + r, cc0 + c)
        for r, c in zero_phase
        if (cr0 + r, cc0 + c) != (cr0 + survivor_r, cc0 + survivor_c)
    ]
    if spare_inside and random.random() < 0.5:
        rr, cc = random.choice(spare_inside)
        interior[rr, cc] = Color.GREY

    if len(outside) > 1 and random.random() < 0.5:
        rr, cc = random.choice([cell for cell in outside if cell != (outside_r, outside_c)])
        interior[rr, cc] = Color.GREY

    assert 1 <= int(np.sum(interior == Color.GREY)) <= 6
    return interior


def _paint_frame(grid: np.ndarray, spec: dict) -> None:
    top = spec["top"]
    left = spec["left"]
    frame = np.full(spec["outer_shape"], spec["frame_color"], dtype=int)
    frame[1:-1, 1:-1] = Color.BLACK

    side = spec["side"]
    a, b = spec["marker_pair"]
    if side == "top":
        frame[0, a] = Color.YELLOW
        frame[0, b] = Color.YELLOW
    elif side == "bottom":
        frame[-1, a] = Color.YELLOW
        frame[-1, b] = Color.YELLOW
    elif side == "left":
        frame[a, 0] = Color.YELLOW
        frame[b, 0] = Color.YELLOW
    else:
        frame[a, -1] = Color.YELLOW
        frame[b, -1] = Color.YELLOW

    paste(grid, frame, top, left)
    paste(grid, spec["interior"], top + 1, left + 1, transparent=Color.BLACK)


def _place_specs(specs: list[dict]) -> tuple[int, int] | None:
    order = list(range(len(specs)))
    random.shuffle(order)
    split = random.choice([1, 2])
    row_sets = [order[:split], order[split:]]
    gap_x = random.randint(2, 5)
    gap_y = random.randint(2, 5)

    row_heights = [max(specs[i]["outer_shape"][0] for i in row) for row in row_sets]
    row_widths = [
        sum(specs[i]["outer_shape"][1] for i in row) + gap_x * (len(row) - 1)
        for row in row_sets
    ]
    need_h = sum(row_heights) + gap_y
    need_w = max(row_widths)
    if need_h > CANVAS_MAX or need_w > CANVAS_MAX:
        return None

    canvas_h = random.randint(max(CANVAS_MIN, need_h), CANVAS_MAX)
    canvas_w = random.randint(max(CANVAS_MIN, need_w), CANVAS_MAX)
    row_top = random.randint(0, canvas_h - need_h)

    current_r = row_top
    for row, row_h, row_w in zip(row_sets, row_heights, row_widths):
        row_left = random.randint(0, canvas_w - row_w)
        current_c = row_left
        for idx in row:
            spec = specs[idx]
            oh, ow = spec["outer_shape"]
            spec["top"] = current_r + random.randint(0, row_h - oh)
            spec["left"] = current_c
            current_c += ow + gap_x
        current_r += row_h + gap_y

    return canvas_h, canvas_w


def _validate_sample(grid: np.ndarray) -> None:
    frames = _extract_frames(grid)
    crop_areas = [frame["crop_area"] for frame in frames]
    assert len(set(crop_areas)) == 3

    frames.sort(key=lambda frame: frame["crop_area"])
    template = frames[0]
    assert template["interior_area"] > min(frame["interior_area"] for frame in frames[1:])

    template_crop = template["crop"]
    th, tw = template_crop.shape
    for frame in frames[1:]:
        ch, cw = frame["crop"].shape
        assert ch % th == 0 and cw % tw == 0

    out = main(grid)
    assert np.array_equal(crop_bbox(out, template["interior_box"]), crop_bbox(grid, template["interior_box"]))

    for frame in frames:
        interior = crop_bbox(grid, frame["interior_box"])
        gray_count = int(np.sum(interior == Color.GREY))
        assert 1 <= gray_count <= 6

    for frame in frames[1:]:
        before_crop = frame["crop"]
        after_crop = crop_bbox(out, frame["crop_box"])
        expected_crop = _overlay_template(before_crop, template_crop)
        assert np.array_equal(after_crop, expected_crop)
        assert np.any(after_crop != before_crop)

        before_interior = crop_bbox(grid, frame["interior_box"])
        after_interior = crop_bbox(out, frame["interior_box"])
        local_mask = np.ones(before_interior.shape, dtype=bool)
        r0, c0, r1, c1 = frame["crop_local_box"]
        local_mask[r0 : r1 + 1, c0 : c1 + 1] = False
        assert np.any(local_mask)
        assert np.array_equal(before_interior[local_mask], after_interior[local_mask])
        assert np.any(before_interior[local_mask] == Color.GREY)

        tiled = _repeat_to_shape(template_crop, before_crop.shape)
        assert np.any((before_crop == Color.GREY) & (tiled == Color.BLACK))
        assert np.any((before_crop == Color.BLACK) & (tiled == Color.GREY))


def _attempt_generate() -> np.ndarray | None:
    crop_shapes = _pick_crop_shapes()
    specs = [_build_spec(shape, require_outside=(i != 0)) for i, shape in enumerate(crop_shapes)]
    if specs[0]["interior_area"] <= min(spec["interior_area"] for spec in specs[1:]):
        return None

    colors = list(FRAME_COLORS)
    random.shuffle(colors)
    for spec, color in zip(specs, colors):
        spec["frame_color"] = color

    template_pattern = _make_template_pattern(specs[0]["crop_shape"])
    for spec in specs:
        spec["interior"] = _build_interior(spec, template_pattern)

    canvas_shape = _place_specs(specs)
    if canvas_shape is None:
        return None

    grid = np.zeros(canvas_shape, dtype=int)
    for spec in specs:
        _paint_frame(grid, spec)

    try:
        _validate_sample(grid)
    except Exception:
        return None
    return grid


def generate_input() -> np.ndarray:
    for _ in range(MAX_GENERATION_ATTEMPTS):
        grid = _attempt_generate()
        if grid is not None:
            return grid
    raise RuntimeError("failed to generate a valid input")
