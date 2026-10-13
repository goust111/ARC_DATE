from __future__ import annotations

from collections import defaultdict
from pathlib import Path
import sys

import numpy as np

_UTILS_DIR = Path("/home/lixinyu/snap/new/solvers")
if str(_UTILS_DIR) not in sys.path:
    sys.path.append(str(_UTILS_DIR))

from arc_utils import Color, crop_bbox, find_components, get_bg, paste, scale_sprite


_RNG = np.random.default_rng()
_STAMP_SHAPES = ((5, 3), (7, 3), (3, 5), (7, 5))
_PAIR_COUNT_BY_WIDTH = {3: (2, 3), 5: (2, 3)}
_MAX_GENERATION_ATTEMPTS = 400
_PLACEMENT_PADDING = 1
_PLACEMENT_MARGIN = 1
_PALETTE = tuple(range(1, 10))


def _ordered_pair(coords: tuple[tuple[int, int], tuple[int, int]]) -> tuple[tuple[int, int], tuple[int, int]]:
    a, b = coords
    if a[0] == b[0]:
        return tuple(sorted((a, b), key=lambda p: p[1]))
    return tuple(sorted((a, b), key=lambda p: p[0]))


def _pair_axis(coords: tuple[tuple[int, int], tuple[int, int]]) -> str:
    (r1, c1), (r2, c2) = coords
    assert (r1 == r2) ^ (c1 == c2), "beacon pair must align on exactly one axis"
    return "H" if r1 == r2 else "V"


def _pair_span(coords: tuple[tuple[int, int], tuple[int, int]]) -> int:
    (r1, c1), (r2, c2) = coords
    return abs(c2 - c1) + 1 if r1 == r2 else abs(r2 - r1) + 1


def _boxes_separated(a: tuple[int, int, int, int], b: tuple[int, int, int, int], pad: int = 0) -> bool:
    return (
        a[2] + pad < b[0]
        or b[2] + pad < a[0]
        or a[3] + pad < b[1]
        or b[3] + pad < a[1]
    )


def _make_stamp_mask(rng: np.random.Generator) -> np.ndarray:
    h, w = _STAMP_SHAPES[int(rng.integers(len(_STAMP_SHAPES)))]
    mask = np.zeros((h, w), dtype=bool)
    mid_r = h // 2
    mid_c = w // 2

    # The center cross guarantees connectivity and makes the beacon endpoints occupied.
    mask[mid_r, :] = True
    mask[:, mid_c] = True

    candidates: list[tuple[int, int]] = []
    for r in range(h):
        for c in range(w):
            mate = (h - 1 - r, w - 1 - c)
            if (r, c) >= mate:
                continue
            if r == mid_r or c == mid_c or mate[0] == mid_r or mate[1] == mid_c:
                continue
            candidates.append((r, c))

    if candidates:
        extra_target = int(rng.integers(1, min(4, len(candidates)) + 1))
        chosen = 0
        while chosen < extra_target:
            frontier = []
            for r, c in candidates:
                mate_r, mate_c = h - 1 - r, w - 1 - c
                if mask[r, c] or mask[mate_r, mate_c]:
                    continue
                if _touches_mask(mask, r, c) and _touches_mask(mask, mate_r, mate_c):
                    frontier.append((r, c))
            if not frontier:
                break
            pick_r, pick_c = frontier[int(rng.integers(len(frontier)))]
            mask[pick_r, pick_c] = True
            mask[h - 1 - pick_r, w - 1 - pick_c] = True
            chosen += 1

    assert np.array_equal(mask, np.rot90(mask, 2))
    assert not np.array_equal(mask, np.rot90(mask, 1))
    comps = find_components(mask.astype(int), bg=0, connectivity=4, monochromatic=True, colors=[1], return_format="dict")
    assert len(comps) == 1
    return mask


def _touches_mask(mask: np.ndarray, r: int, c: int) -> bool:
    return (
        (r > 0 and mask[r - 1, c])
        or (r + 1 < mask.shape[0] and mask[r + 1, c])
        or (c > 0 and mask[r, c - 1])
        or (c + 1 < mask.shape[1] and mask[r, c + 1])
    )


def _recolor_scaled(sprite: np.ndarray, scale: int, color: int, bg: int) -> np.ndarray:
    scaled = scale_sprite(sprite, scale)
    out = np.full(scaled.shape, int(bg), dtype=int)
    out[scaled != bg] = int(color)
    return out


def _parse_scene(
    grid: np.ndarray,
) -> tuple[int, int, np.ndarray, dict[int, tuple[tuple[int, int], tuple[int, int]]]]:
    g = np.asarray(grid, dtype=int)
    assert g.ndim == 2, "expected a 2D grid"
    bg = int(get_bg(g))

    by_color: dict[int, list[dict]] = defaultdict(list)
    for comp in find_components(g, bg=bg, connectivity=4, monochromatic=True, return_format="dict"):
        by_color[int(comp["color"])].append(comp)

    stamp_candidates: list[dict] = []
    beacon_pairs: dict[int, tuple[tuple[int, int], tuple[int, int]]] = {}
    for color, comps in by_color.items():
        areas = [int(comp["area"]) for comp in comps]
        if len(comps) == 1 and areas[0] > 1:
            stamp_candidates.append(comps[0])
            continue

        assert len(comps) == 2, "each beacon color must appear exactly twice"
        coords: list[tuple[int, int]] = []
        for comp in comps:
            assert int(comp["area"]) == 1, "beacons must be single pixels"
            coords.append(tuple(map(int, comp["coords"][0])))
        ordered = _ordered_pair((coords[0], coords[1]))
        axis = _pair_axis(ordered)
        span = _pair_span(ordered)
        gap = span - 1
        assert gap % 2 == 0 and gap >= 2, "beacon gap must be a positive even number"
        beacon_pairs[int(color)] = ordered
        assert axis in {"H", "V"}

    assert len(stamp_candidates) == 1, "expected exactly one multi-cell stamp sprite"
    stamp_color = int(stamp_candidates[0]["color"])
    stamp = crop_bbox(g, stamp_candidates[0]["bbox"])
    stamp_mask = stamp != bg
    assert np.array_equal(stamp_mask, np.rot90(stamp_mask, 2)), "stamp must have 180-degree symmetry"
    assert stamp.shape[0] % 2 == 1 and stamp.shape[1] % 2 == 1, "stamp dimensions must be odd"
    return bg, stamp_color, stamp, beacon_pairs


def main(grid: np.ndarray) -> np.ndarray:
    """
    Stamp the unique sprite at every same-color beacon pair using the pair's midpoint, axis, color, and scale.

    Input structure:
    - A black background contains one connected monochrome stamp sprite and several same-color beacon pairs.
    - Each beacon is a single pixel; each beacon color appears exactly twice and differs from the stamp color.
    - Every pair is aligned on one row or one column, and the distance between the two beacons leaves an integer midpoint.

    Rule:
    - Keep the original stamp in place.
    - Erase each beacon pair, then use that pair's color for a new copy of the stamp.
    - Horizontal pairs use the stamp as-is; vertical pairs use the stamp rotated by 90 degrees.
    - Uniformly scale the chosen orientation so its span along the pair axis matches the inclusive beacon span, then center the recolored copy at the pair midpoint.

    Key insight:
    - Each beacon pair jointly encodes four things at once: where the copy goes, which way it points, which color it gets, and the integer scale factor.
    """
    g = np.asarray(grid, dtype=int)
    bg, _stamp_color, stamp, beacon_pairs = _parse_scene(g)

    out = g.copy()
    for color in sorted(beacon_pairs):
        for r, c in beacon_pairs[color]:
            out[r, c] = bg

    for color in sorted(beacon_pairs):
        coords = beacon_pairs[color]
        axis = _pair_axis(coords)
        oriented = stamp if axis == "H" else np.rot90(stamp, 1)
        pair_span = _pair_span(coords)
        axis_span = oriented.shape[1] if axis == "H" else oriented.shape[0]
        assert pair_span % axis_span == 0, "beacon span must be an integer multiple of the stamp span"
        scale = pair_span // axis_span
        copy = _recolor_scaled(oriented, scale, color, bg)

        mid_r = (coords[0][0] + coords[1][0]) // 2
        mid_c = (coords[0][1] + coords[1][1]) // 2
        r0 = mid_r - copy.shape[0] // 2
        c0 = mid_c - copy.shape[1] // 2
        r1 = r0 + copy.shape[0]
        c1 = c0 + copy.shape[1]
        assert 0 <= r0 < r1 <= g.shape[0] and 0 <= c0 < c1 <= g.shape[1], "stamped copy must stay in-bounds"

        target = out[r0:r1, c0:c1]
        fill = copy != bg
        assert np.all(target[fill] == bg), "stamped copies must not overlap the original stamp or each other"
        paste(out, copy, r0, c0, transparent=bg)

    return out


def _sample_positions(
    rng: np.random.Generator,
    canvas_shape: tuple[int, int],
    item_shapes: list[tuple[int, int]],
) -> list[tuple[int, int]] | None:
    H, W = canvas_shape
    order = sorted(range(len(item_shapes)), key=lambda i: item_shapes[i][0] * item_shapes[i][1], reverse=True)
    positions: list[tuple[int, int] | None] = [None] * len(item_shapes)
    boxes: list[tuple[int, int, int, int]] = []

    for idx in order:
        h, w = item_shapes[idx]
        r_min = _PLACEMENT_MARGIN
        c_min = _PLACEMENT_MARGIN
        r_max = H - h - _PLACEMENT_MARGIN
        c_max = W - w - _PLACEMENT_MARGIN
        if r_max < r_min or c_max < c_min:
            return None

        candidates = [(r, c) for r in range(r_min, r_max + 1) for c in range(c_min, c_max + 1)]
        rng.shuffle(candidates)
        for r, c in candidates:
            box = (r, c, r + h - 1, c + w - 1)
            if all(_boxes_separated(box, other, pad=_PLACEMENT_PADDING) for other in boxes):
                positions[idx] = (r, c)
                boxes.append(box)
                break
        else:
            return None

    return [pos for pos in positions if pos is not None]


def _validate_sample(
    inp: np.ndarray,
    out: np.ndarray,
    stamp_box: tuple[int, int, int, int],
    pair_specs: list[dict],
) -> None:
    bg, stamp_color, stamp, beacon_pairs = _parse_scene(inp)
    assert not np.array_equal(stamp != bg, np.rot90(stamp != bg, 1)), "rotation branch must matter"
    assert len(pair_specs) == len(beacon_pairs) >= 2

    seen_axes = set()
    seen_big_scale = False
    for spec in pair_specs:
        color = int(spec["color"])
        assert color in beacon_pairs
        coords = beacon_pairs[color]
        assert coords == spec["beacons"]
        seen_axes.add(spec["axis"])
        seen_big_scale |= int(spec["scale"]) > 1

        oriented = stamp if spec["axis"] == "H" else np.rot90(stamp, 1)
        expected = _recolor_scaled(oriented, int(spec["scale"]), color, bg)
        expected_span = oriented.shape[1] if spec["axis"] == "H" else oriented.shape[0]
        assert _pair_span(coords) == expected_span * int(spec["scale"])

        comps = find_components(out, bg=bg, connectivity=4, monochromatic=True, colors=[color], return_format="dict")
        assert len(comps) == 1, "each stamped color must become one connected object"
        assert tuple(map(int, comps[0]["bbox"])) == spec["box"]
        assert np.array_equal(crop_bbox(out, spec["box"]), expected)

    assert seen_axes == {"H", "V"}, "generator must expose both orientation branches"
    assert seen_big_scale, "generator must expose a non-trivial scale factor"

    stamp_comps = find_components(out, bg=bg, connectivity=4, monochromatic=True, colors=[stamp_color], return_format="dict")
    assert len(stamp_comps) == 1
    assert tuple(map(int, stamp_comps[0]["bbox"])) == stamp_box
    assert np.array_equal(main(inp), out)


def _gen_attempt() -> np.ndarray | None:
    rng = _RNG
    bg = Color.BLACK

    stamp_mask = _make_stamp_mask(rng)
    stamp_h, stamp_w = map(int, stamp_mask.shape)

    pair_count_lo, pair_count_hi = _PAIR_COUNT_BY_WIDTH[stamp_w]
    pair_count = int(rng.integers(pair_count_lo, pair_count_hi + 1))
    axes = ["H", "V"] + [("H" if int(rng.integers(2)) == 0 else "V") for _ in range(pair_count - 2)]
    rng.shuffle(axes)

    scales = [1] * pair_count
    forced_large = int(rng.integers(pair_count))
    scales[forced_large] = 3
    for i in range(pair_count):
        if i == forced_large:
            continue
        scales[i] = 3 if rng.random() < 0.35 else 1

    palette = list(rng.choice(np.array(_PALETTE, dtype=int), size=pair_count + 1, replace=False))
    stamp_color = int(palette[0])
    pair_colors = [int(x) for x in palette[1:]]

    stamp = np.full(stamp_mask.shape, bg, dtype=int)
    stamp[stamp_mask] = stamp_color

    pair_specs: list[dict] = []
    item_arrays = [stamp]
    item_shapes = [tuple(map(int, stamp.shape))]
    for axis, scale, color in zip(axes, scales, pair_colors):
        oriented = stamp if axis == "H" else np.rot90(stamp, 1)
        copy = _recolor_scaled(oriented, int(scale), color, bg)
        item_arrays.append(copy)
        item_shapes.append(tuple(map(int, copy.shape)))
        pair_specs.append({"axis": axis, "scale": int(scale), "color": color, "array": copy})

    max_h = max(shape[0] for shape in item_shapes)
    max_w = max(shape[1] for shape in item_shapes)
    total_area = sum(shape[0] * shape[1] for shape in item_shapes)
    base_side = max(22, int(np.ceil(np.sqrt(total_area * 1.7))), max_h + 2, max_w + 2)
    if base_side > 30:
        return None

    H = int(rng.integers(base_side, 31))
    W = int(rng.integers(base_side, 31))
    positions = _sample_positions(rng, (H, W), item_shapes)
    if positions is None:
        return None

    inp = np.full((H, W), bg, dtype=int)
    out = np.full((H, W), bg, dtype=int)

    stamp_r, stamp_c = positions[0]
    paste(inp, stamp, stamp_r, stamp_c, transparent=bg)
    paste(out, stamp, stamp_r, stamp_c, transparent=bg)
    stamp_box = (stamp_r, stamp_c, stamp_r + stamp.shape[0] - 1, stamp_c + stamp.shape[1] - 1)

    for spec, (r0, c0) in zip(pair_specs, positions[1:]):
        copy = spec["array"]
        paste(out, copy, r0, c0, transparent=bg)
        h, w = copy.shape
        center_r = r0 + h // 2
        center_c = c0 + w // 2
        if spec["axis"] == "H":
            beacons = ((center_r, c0), (center_r, c0 + w - 1))
        else:
            beacons = ((r0, center_c), (r0 + h - 1, center_c))

        ordered = _ordered_pair(beacons)
        rel_a = (ordered[0][0] - r0, ordered[0][1] - c0)
        rel_b = (ordered[1][0] - r0, ordered[1][1] - c0)
        assert copy[rel_a] == spec["color"] and copy[rel_b] == spec["color"]
        inp[ordered[0]] = spec["color"]
        inp[ordered[1]] = spec["color"]
        spec["beacons"] = ordered
        spec["box"] = (r0, c0, r0 + h - 1, c0 + w - 1)

    _validate_sample(inp, out, stamp_box, pair_specs)
    return inp


def generate_input() -> np.ndarray:
    for _ in range(_MAX_GENERATION_ATTEMPTS):
        grid = _gen_attempt()
        if grid is not None:
            return grid
    raise RuntimeError("failed to generate a valid beacon-pair stamping input")
