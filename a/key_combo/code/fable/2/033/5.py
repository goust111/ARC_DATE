from __future__ import annotations

from pathlib import Path
import random
import sys

import numpy as np

_UTILS_DIR = Path("/home/lixinyu/snap/new/solvers")
if str(_UTILS_DIR) not in sys.path:
    sys.path.append(str(_UTILS_DIR))

from arc_utils import bbox_coords, components_mask, find_components, get_bg


BBox = tuple[int, int, int, int]

GEN_RETRIES = 500
PALETTE = tuple(range(1, 10))
CHANNEL_COUNT_OPTIONS = (3, 4)
CHANNEL_HEIGHT_MIN = 2
CHANNEL_HEIGHT_MAX = 4
TEMPLATE_WIDTH_MIN = 8
TEMPLATE_WIDTH_MAX = 12
DEPTH_MIN = 2
DEPTH_MAX = 6
SOLID_BAND_MIN = 1
SOLID_BAND_MAX = 2
OUTER_MARGIN_MIN = 1
OUTER_MARGIN_MAX = 4
APPROACH_MARGIN_MIN = 7
APPROACH_MARGIN_MAX = 11
FAR_MARGIN_MIN = 2
FAR_MARGIN_MAX = 5
SPRITE_WIDTH_MAX = 5


def _template_component(grid: np.ndarray, bg: int) -> dict:
    comps = find_components(grid, bg=bg, connectivity=8, monochromatic=True, return_format="dict")
    assert comps, "expected at least one foreground component"
    comps = sorted(comps, key=lambda comp: int(comp["area"]), reverse=True)
    if len(comps) > 1:
        assert int(comps[0]["area"]) > int(comps[1]["area"]), "template must be the unique largest component"
    return comps[0]


def _channel_side(template: np.ndarray, base: int) -> str:
    left_open = bool(np.any(template[:, 0] != int(base)))
    right_open = bool(np.any(template[:, -1] != int(base)))
    assert left_open ^ right_open, "expected channels on exactly one vertical side"
    return "left" if left_open else "right"


def _row_depths(template: np.ndarray, base: int, side: str) -> list[int]:
    heights: list[int] = []
    h, w = template.shape
    for r in range(h):
        depth = 0
        if side == "left":
            while depth < w and int(template[r, depth]) != int(base):
                depth += 1
            assert np.all(template[r, depth:] == int(base)), "left-facing channel rows must be rectangular prefixes"
        else:
            while depth < w and int(template[r, w - 1 - depth]) != int(base):
                depth += 1
            assert np.all(template[r, : w - depth] == int(base)), "right-facing channel rows must be rectangular suffixes"
        heights.append(int(depth))
    return heights


def _extract_channels(
    grid: np.ndarray,
    bg: int,
    base: int,
    bbox: BBox,
    side: str,
) -> list[dict]:
    r0, c0, r1, c1 = bbox
    template = grid[r0 : r1 + 1, c0 : c1 + 1]
    depths = _row_depths(template, base, side)
    channels: list[dict] = []
    rr = 0
    while rr < len(depths):
        depth = int(depths[rr])
        if depth == 0:
            rr += 1
            continue
        start = rr
        while rr < len(depths) and int(depths[rr]) == depth:
            rr += 1
        end = rr - 1
        if side == "left":
            region = template[start : end + 1, :depth]
            deep_col = int(c0 + depth - 1)
        else:
            region = template[start : end + 1, template.shape[1] - depth :]
            deep_col = int(c1 - depth + 1)
        dye_cells = np.argwhere((region != int(bg)) & (region != int(base)))
        assert dye_cells.shape[0] == 1, "each channel must contain exactly one dye cell"
        local_dye_r = int(dye_cells[0, 0])
        dye_r = int(r0 + start + local_dye_r)
        dye_color = int(region[tuple(dye_cells[0])])
        channels.append(
            {
                "r0": int(r0 + start),
                "r1": int(r0 + end),
                "depth": int(depth),
                "dye_r": int(dye_r),
                "dye_c": int(deep_col),
                "dye_color": int(dye_color),
            }
        )
    assert channels, "expected at least one channel"
    return channels


def _extract_sprites(grid: np.ndarray, bg: int, base: int, bbox: BBox, side: str) -> list[dict]:
    r0, c0, r1, c1 = bbox
    mask = (grid != int(bg)) & (grid != int(base))
    comps = components_mask(mask, connectivity=4)
    sprites: list[dict] = []
    dye_components = 0
    for coords in comps:
        br0, bc0, br1, bc1 = bbox_coords(coords)
        inside_template = br0 >= r0 and br1 <= r1 and bc0 >= c0 and bc1 <= c1
        if inside_template:
            assert coords.shape[0] == 1, "only dye dots may live inside the template"
            dye_components += 1
            continue
        h = int(br1 - br0 + 1)
        w = int(bc1 - bc0 + 1)
        color = int(grid[int(coords[0, 0]), int(coords[0, 1])])
        assert coords.shape[0] == h * w, "sprites must be solid rectangles"
        assert np.all(grid[br0 : br1 + 1, bc0 : bc1 + 1] == color), "sprite rectangle must be monochrome"
        if side == "left":
            assert bc1 < c0, "all sprites must approach from the channel side"
        else:
            assert bc0 > c1, "all sprites must approach from the channel side"
        sprites.append(
            {
                "r0": int(br0),
                "r1": int(br1),
                "c0": int(bc0),
                "c1": int(bc1),
                "h": int(h),
                "w": int(w),
                "color": int(color),
            }
        )
    assert sprites, "expected at least one outside sprite"
    sprites.sort(key=lambda sprite: (int(sprite["r0"]), int(sprite["c0"])))
    assert dye_components >= 1, "expected dye dots inside the template"
    return sprites


def _parse_scene(grid: np.ndarray) -> dict:
    g = np.asarray(grid, dtype=int)
    assert g.ndim == 2
    bg = int(get_bg(g))
    template_comp = _template_component(g, bg)
    base = int(template_comp["color"])
    bbox = bbox_coords(np.argwhere(g == int(base)))
    template = g[bbox[0] : bbox[2] + 1, bbox[1] : bbox[3] + 1]
    side = _channel_side(template, base)
    channels = _extract_channels(g, bg, base, bbox, side)
    sprites = _extract_sprites(g, bg, base, bbox, side)
    return {
        "grid": g,
        "bg": int(bg),
        "base": int(base),
        "bbox": tuple(map(int, bbox)),
        "side": side,
        "channels": channels,
        "sprites": sprites,
    }


def _place_channel_sprite(out: np.ndarray, bbox: BBox, side: str, channel: dict, sprite: dict, color: int) -> None:
    _r0, c0, _r1, c1 = bbox
    h = int(sprite["h"])
    w = int(sprite["w"])
    rr0 = int(sprite["r0"])
    if side == "left":
        cc0 = int(c0 + int(channel["depth"]) - w)
    else:
        cc0 = int(c1 - int(channel["depth"]) + 1)
    out[rr0 : rr0 + h, cc0 : cc0 + w] = int(color)


def _place_flush_sprite(out: np.ndarray, bbox: BBox, side: str, sprite: dict) -> None:
    _r0, c0, _r1, c1 = bbox
    h = int(sprite["h"])
    w = int(sprite["w"])
    rr0 = int(sprite["r0"])
    if side == "left":
        cc0 = int(c0 - w)
    else:
        cc0 = int(c1 + 1)
    out[rr0 : rr0 + h, cc0 : cc0 + w] = int(sprite["color"])


def main(grid: np.ndarray) -> np.ndarray:
    """
    Slide each outside rectangle into a side-open template only when its full row span exactly matches one slot, then recolor it from the slot-bottom dye.

    Input structure:
    - A dominant background surrounds one large monochrome template rectangle.
    - The template has several rectangular channels cut in from exactly one vertical side; each channel contains one colored dye pixel at its deepest cell.
    - Several solid monochrome rectangles sit outside that same side and are aligned to slide horizontally toward the template.

    Rule:
    - For each outside rectangle, compare its complete row span with the channel openings on the approached template side.
    - A rectangle enters only when its row span matches one whole channel exactly; it slides to the channel bottom, keeps its own width, and is recolored to that channel's dye color, covering the dye pixel.
    - Any rectangle whose span is shorter, taller, or vertically offset from every full opening stops flush against the outer template face and keeps its original color; the template and all unused channels otherwise stay unchanged.

    Key insight:
    - The decisive key is exact opening height across the rectangle's entire cross-section: nearby recesses are bait, while the final color comes only from the single dye cell buried at the slot bottom.
    """
    scene = _parse_scene(np.asarray(grid, dtype=int))
    g = scene["grid"]
    bg = int(scene["bg"])
    bbox = scene["bbox"]
    side = scene["side"]
    channels = scene["channels"]
    sprites = scene["sprites"]

    channel_by_rows = {
        (int(channel["r0"]), int(channel["r1"])): channel
        for channel in channels
    }

    out = g.copy()
    for sprite in sprites:
        out[int(sprite["r0"]) : int(sprite["r1"]) + 1, int(sprite["c0"]) : int(sprite["c1"]) + 1] = int(bg)

    for sprite in sprites:
        key = (int(sprite["r0"]), int(sprite["r1"]))
        channel = channel_by_rows.get(key)
        if channel is None:
            _place_flush_sprite(out, bbox, side, sprite)
            continue
        assert int(sprite["w"]) <= int(channel["depth"]), "matching sprite must fit inside channel depth"
        _place_channel_sprite(out, bbox, side, channel, sprite, int(channel["dye_color"]))

    return out


def _build_channel_rows(
    channel_heights: list[int],
    top_solid: int,
    gap_sizes: list[int],
) -> tuple[int, list[tuple[int, int]], list[tuple[int, int]]]:
    rows: list[tuple[int, int]] = []
    solids: list[tuple[int, int]] = []
    cur = int(top_solid)
    if top_solid > 0:
        solids.append((0, int(top_solid - 1)))
    for idx, height in enumerate(channel_heights):
        r0 = int(cur)
        r1 = int(cur + height - 1)
        rows.append((r0, r1))
        cur = int(r1 + 1)
        if idx < len(gap_sizes):
            gap = int(gap_sizes[idx])
            solids.append((cur, int(cur + gap - 1)))
            cur += gap
    return int(cur), rows, solids


def _render_input_from_spec(spec: dict) -> np.ndarray:
    h = int(spec["grid_h"])
    w = int(spec["grid_w"])
    bg = int(spec["bg"])
    base = int(spec["base"])
    tr0, tc0, tr1, tc1 = spec["bbox"]
    side = spec["side"]

    grid = np.full((h, w), bg, dtype=int)
    grid[tr0 : tr1 + 1, tc0 : tc1 + 1] = int(base)

    for channel in spec["channels"]:
        r0 = int(channel["r0"])
        r1 = int(channel["r1"])
        depth = int(channel["depth"])
        if side == "left":
            grid[r0 : r1 + 1, tc0 : tc0 + depth] = int(bg)
            dye_c = int(tc0 + depth - 1)
        else:
            grid[r0 : r1 + 1, tc1 - depth + 1 : tc1 + 1] = int(bg)
            dye_c = int(tc1 - depth + 1)
        grid[int(channel["dye_r"]), dye_c] = int(channel["dye_color"])

    for sprite in spec["sprites"]:
        grid[int(sprite["r0"]) : int(sprite["r1"]) + 1, int(sprite["c0"]) : int(sprite["c0"]) + int(sprite["w"])] = int(sprite["color"])

    return grid


def _render_output_from_spec(
    spec: dict,
    *,
    allow_subrange: bool,
    sink_to_bottom: bool,
    recolor: bool,
) -> np.ndarray:
    grid = _render_input_from_spec(spec)
    bg = int(spec["bg"])
    bbox = spec["bbox"]
    side = spec["side"]
    channels = spec["channels"]

    out = grid.copy()
    for sprite in spec["sprites"]:
        out[int(sprite["r0"]) : int(sprite["r1"]) + 1, int(sprite["c0"]) : int(sprite["c0"]) + int(sprite["w"])] = int(bg)

    for sprite in spec["sprites"]:
        enters = sprite["kind"] == "exact" or (allow_subrange and sprite["kind"] == "subrange")
        if not enters:
            _place_flush_sprite(out, bbox, side, sprite)
            continue

        channel = channels[int(sprite["channel_idx"])]
        paint = int(channel["dye_color"]) if recolor else int(sprite["color"])
        if sink_to_bottom:
            _place_channel_sprite(out, bbox, side, channel, sprite, paint)
        else:
            tr0, tc0, tr1, tc1 = bbox
            rr0 = int(sprite["r0"])
            hh = int(sprite["h"])
            ww = int(sprite["w"])
            if side == "left":
                cc0 = int(tc0)
            else:
                cc0 = int(tc1 - ww + 1)
            out[rr0 : rr0 + hh, cc0 : cc0 + ww] = paint

    return out


def _sprite_start_col(side: str, bbox: BBox, width: int, gap: int) -> int:
    _r0, c0, _r1, c1 = bbox
    if side == "left":
        return int(c0 - gap - width)
    return int(c1 + 1 + gap)


def _make_spec() -> dict | None:
    n_channels = random.choice(CHANNEL_COUNT_OPTIONS)
    heights = [random.randint(CHANNEL_HEIGHT_MIN, CHANNEL_HEIGHT_MAX) for _ in range(n_channels)]
    top_solid = random.randint(SOLID_BAND_MIN, SOLID_BAND_MAX)
    bottom_solid = random.randint(SOLID_BAND_MIN, SOLID_BAND_MAX)
    gap_sizes = [random.randint(SOLID_BAND_MIN, SOLID_BAND_MAX) for _ in range(n_channels - 1)]
    used_height, row_ranges, solid_runs = _build_channel_rows(heights, top_solid, gap_sizes)
    template_h = int(used_height + bottom_solid)
    if not (10 <= template_h <= 18):
        return None

    template_w = random.randint(TEMPLATE_WIDTH_MIN, TEMPLATE_WIDTH_MAX)
    max_depth = min(DEPTH_MAX, template_w - 3)
    if max_depth < DEPTH_MIN:
        return None

    depths = [random.randint(DEPTH_MIN, max_depth) for _ in range(n_channels)]
    if max(depths) < 3 and max_depth >= 3:
        depths[random.randrange(n_channels)] = random.randint(3, max_depth)

    approach_margin = random.randint(max_depth + 2, APPROACH_MARGIN_MAX)
    far_margin = random.randint(FAR_MARGIN_MIN, FAR_MARGIN_MAX)
    grid_w = int(approach_margin + template_w + far_margin)
    if grid_w > 30:
        return None

    outer_top = random.randint(OUTER_MARGIN_MIN, OUTER_MARGIN_MAX)
    outer_bottom = random.randint(OUTER_MARGIN_MIN, OUTER_MARGIN_MAX)
    grid_h = int(outer_top + template_h + outer_bottom)
    if grid_h > 30:
        return None

    side = random.choice(("left", "right"))
    if side == "left":
        bbox = (
            int(outer_top),
            int(approach_margin),
            int(outer_top + template_h - 1),
            int(approach_margin + template_w - 1),
        )
    else:
        bbox = (
            int(outer_top),
            int(far_margin),
            int(outer_top + template_h - 1),
            int(far_margin + template_w - 1),
        )

    tr0, tc0, _tr1, _tc1 = bbox
    channel_specs: list[dict] = []
    for (local_r0, local_r1), depth in zip(row_ranges, depths):
        channel_specs.append(
            {
                "r0": int(tr0 + local_r0),
                "r1": int(tr0 + local_r1),
                "depth": int(depth),
            }
        )

    solid_global = [(int(tr0 + a), int(tr0 + b)) for a, b in solid_runs]
    solid_global.append((int(tr0 + used_height), int(tr0 + template_h - 1)))

    if any(r0 > r1 for r0, r1 in solid_global):
        return None

    unused_channel_indices = list(range(n_channels))
    random.shuffle(unused_channel_indices)
    used_channels = sorted(unused_channel_indices[:2])
    bait_channel_idx = next(idx for idx in range(n_channels) if idx not in used_channels)

    sprite_count = 4
    color_count = 1 + n_channels + sprite_count
    colors = random.sample(PALETTE, color_count)
    base = int(colors[0])
    dye_colors = list(map(int, colors[1 : 1 + n_channels]))
    sprite_colors = list(map(int, colors[1 + n_channels :]))

    for channel, dye_color in zip(channel_specs, dye_colors):
        channel["dye_color"] = int(dye_color)
        channel["dye_r"] = random.randint(int(channel["r0"]), int(channel["r1"]))

    exact_idx, short_idx = used_channels
    if random.random() < 0.5:
        exact_idx, short_idx = short_idx, exact_idx

    sprites: list[dict] = []

    for idx, channel_idx in enumerate((exact_idx, short_idx)):
        channel = channel_specs[channel_idx]
        width = int(channel["depth"])
        kind = "exact"
        if channel_idx == short_idx:
            width = random.randint(1, int(channel["depth"]) - 1)
        gap_room = int(approach_margin - width)
        if gap_room < 1:
            return None
        gap = random.randint(1, min(4, gap_room))
        start_col = _sprite_start_col(side, bbox, width, gap)
        sprites.append(
            {
                "kind": kind,
                "channel_idx": int(channel_idx),
                "r0": int(channel["r0"]),
                "r1": int(channel["r1"]),
                "h": int(channel["r1"] - channel["r0"] + 1),
                "w": int(width),
                "c0": int(start_col),
                "color": int(sprite_colors[idx]),
            }
        )

    bait_channel = channel_specs[bait_channel_idx]
    bait_h = random.randint(1, int(bait_channel["r1"] - bait_channel["r0"]))
    if bait_h == int(bait_channel["r1"] - bait_channel["r0"] + 1):
        bait_h -= 1
    if bait_h < 1:
        return None
    bait_r0 = random.randint(int(bait_channel["r0"]), int(bait_channel["r1"] - bait_h + 1))
    bait_width = random.randint(1, min(int(bait_channel["depth"]), SPRITE_WIDTH_MAX))
    bait_gap_room = int(approach_margin - bait_width)
    if bait_gap_room < 1:
        return None
    bait_gap = random.randint(1, min(4, bait_gap_room))
    sprites.append(
        {
            "kind": "subrange",
            "channel_idx": int(bait_channel_idx),
            "r0": int(bait_r0),
            "r1": int(bait_r0 + bait_h - 1),
            "h": int(bait_h),
            "w": int(bait_width),
            "c0": int(_sprite_start_col(side, bbox, bait_width, bait_gap)),
            "color": int(sprite_colors[2]),
        }
    )

    solid_run = random.choice(solid_global)
    solid_h = random.randint(1, int(solid_run[1] - solid_run[0] + 1))
    solid_r0 = random.randint(int(solid_run[0]), int(solid_run[1] - solid_h + 1))
    wall_width = random.randint(1, min(SPRITE_WIDTH_MAX, approach_margin - 1))
    wall_gap = random.randint(1, min(4, int(approach_margin - wall_width)))
    sprites.append(
        {
            "kind": "wall",
            "channel_idx": -1,
            "r0": int(solid_r0),
            "r1": int(solid_r0 + solid_h - 1),
            "h": int(solid_h),
            "w": int(wall_width),
            "c0": int(_sprite_start_col(side, bbox, wall_width, wall_gap)),
            "color": int(sprite_colors[3]),
        }
    )

    seen_rows: list[tuple[int, int]] = []
    for sprite in sprites:
        row_pair = (int(sprite["r0"]), int(sprite["r1"]))
        if any(not (row_pair[1] < a or row_pair[0] > b) for a, b in seen_rows):
            return None
        seen_rows.append(row_pair)

    spec = {
        "grid_h": int(grid_h),
        "grid_w": int(grid_w),
        "bg": 0,
        "base": int(base),
        "bbox": bbox,
        "side": side,
        "channels": channel_specs,
        "sprites": sprites,
    }
    return spec


def _spec_matches_parse(spec: dict, parsed: dict) -> bool:
    if spec["bbox"] != parsed["bbox"] or spec["side"] != parsed["side"] or spec["base"] != parsed["base"]:
        return False
    parsed_channels = [
        (
            int(channel["r0"]),
            int(channel["r1"]),
            int(channel["depth"]),
            int(channel["dye_r"]),
            int(channel["dye_color"]),
        )
        for channel in parsed["channels"]
    ]
    spec_channels = [
        (
            int(channel["r0"]),
            int(channel["r1"]),
            int(channel["depth"]),
            int(channel["dye_r"]),
            int(channel["dye_color"]),
        )
        for channel in spec["channels"]
    ]
    if parsed_channels != spec_channels:
        return False

    parsed_sprites = [
        (
            int(sprite["r0"]),
            int(sprite["r1"]),
            int(sprite["w"]),
            int(sprite["color"]),
        )
        for sprite in parsed["sprites"]
    ]
    spec_sprites = sorted(
        (
            int(sprite["r0"]),
            int(sprite["r1"]),
            int(sprite["w"]),
            int(sprite["color"]),
        )
        for sprite in spec["sprites"]
    )
    return parsed_sprites == spec_sprites


def _passes_semantic_checks(spec: dict) -> bool:
    inp = _render_input_from_spec(spec)
    expected = _render_output_from_spec(spec, allow_subrange=False, sink_to_bottom=True, recolor=True)
    if np.array_equal(inp, expected):
        return False

    loose = _render_output_from_spec(spec, allow_subrange=True, sink_to_bottom=True, recolor=True)
    surface = _render_output_from_spec(spec, allow_subrange=False, sink_to_bottom=False, recolor=True)
    keep_color = _render_output_from_spec(spec, allow_subrange=False, sink_to_bottom=True, recolor=False)
    if np.array_equal(loose, expected):
        return False
    if np.array_equal(surface, expected):
        return False
    if np.array_equal(keep_color, expected):
        return False

    try:
        parsed = _parse_scene(inp)
        actual = main(inp)
    except AssertionError:
        return False

    if not np.array_equal(actual, expected):
        return False
    if not _spec_matches_parse(spec, parsed):
        return False

    exact_sprites = [sprite for sprite in spec["sprites"] if sprite["kind"] == "exact"]
    subrange_sprites = [sprite for sprite in spec["sprites"] if sprite["kind"] == "subrange"]
    wall_sprites = [sprite for sprite in spec["sprites"] if sprite["kind"] == "wall"]
    if len(exact_sprites) != 2 or len(subrange_sprites) != 1 or len(wall_sprites) != 1:
        return False

    exact_depths = [int(spec["channels"][int(sprite["channel_idx"])]["depth"]) for sprite in exact_sprites]
    exact_widths = [int(sprite["w"]) for sprite in exact_sprites]
    if not any(width == depth for width, depth in zip(exact_widths, exact_depths)):
        return False
    if not any(width < depth for width, depth in zip(exact_widths, exact_depths)):
        return False

    bait = subrange_sprites[0]
    bait_channel = spec["channels"][int(bait["channel_idx"])]
    if not (int(bait_channel["r0"]) <= int(bait["r0"]) and int(bait["r1"]) <= int(bait_channel["r1"])):
        return False
    if (int(bait["r0"]), int(bait["r1"])) == (int(bait_channel["r0"]), int(bait_channel["r1"])):
        return False

    wall = wall_sprites[0]
    for rr in range(int(wall["r0"]), int(wall["r1"]) + 1):
        for channel in spec["channels"]:
            if int(channel["r0"]) <= rr <= int(channel["r1"]):
                return False

    return True


def _gen_attempt() -> np.ndarray | None:
    spec = _make_spec()
    if spec is None:
        return None
    if not _passes_semantic_checks(spec):
        return None
    return _render_input_from_spec(spec)


def generate_input() -> np.ndarray:
    for _ in range(GEN_RETRIES):
        grid = _gen_attempt()
        if grid is not None:
            return grid
    raise RuntimeError("failed to generate a valid slot-dye input")
