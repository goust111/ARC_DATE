from __future__ import annotations

import random
import sys
from pathlib import Path

import numpy as np

_ARC_UTILS_DIR = Path("/home/lixinyu/snap/new/solvers")
if str(_ARC_UTILS_DIR) not in sys.path:
    sys.path.insert(0, str(_ARC_UTILS_DIR))

from arc_utils import bbox, crop_bbox, find_components, get_bg, paste, scale_sprite


BG = 0
LEGEND_BORDER = 5
LEGEND_INK = 6
FRAME_BORDER = 8
DECODE_COLOR_POOL = (1, 2, 3, 4, 7, 9)

BASE_SYMBOLS = (
    np.array([[0, 0, 1], [0, 0, 1], [1, 1, 1]], dtype=int),
    np.array([[0, 0, 1], [0, 1, 1], [1, 1, 0]], dtype=int),
    np.array([[0, 0, 1], [1, 1, 1], [0, 0, 1]], dtype=int),
    np.array([[0, 0, 1], [1, 1, 1], [0, 1, 0]], dtype=int),
    np.array([[0, 1, 1], [0, 1, 0], [1, 1, 0]], dtype=int),
)

GRID_MIN = 24
GRID_MAX = 30
SYMBOL_COUNT_CHOICES = (3, 4)
SCALE_VALUES = (1, 1, 2, 2, 2, 3)
PAD_VALUES = (0, 0, 1, 1, 2)
MAX_ATTEMPTS = 120
GEOMETRY_ATTEMPTS = 120
LAYOUT_ATTEMPTS = 10


def _outline_mask(shape: tuple[int, int], color: int) -> np.ndarray:
    h, w = shape
    patch = np.full((h, w), BG, dtype=int)
    patch[0, :] = color
    patch[-1, :] = color
    patch[:, 0] = color
    patch[:, -1] = color
    return patch


def _is_rect_outline(component: dict) -> bool:
    r0, c0, r1, c1 = component["bbox"]
    h = int(r1 - r0 + 1)
    w = int(c1 - c0 + 1)
    if h < 2 or w < 2:
        return False
    expected = 2 * (h + w) - 4
    if int(component["area"]) != expected:
        return False
    coords = {(int(r), int(c)) for r, c in component["coords"]}
    for r in range(r0, r1 + 1):
        for c in range(c0, c1 + 1):
            on_border = r in (r0, r1) or c in (c0, c1)
            if on_border != ((r, c) in coords):
                return False
    return True


def _touches_corner(box: tuple[int, int, int, int], shape: tuple[int, int]) -> bool:
    r0, c0, r1, c1 = box
    h, w = shape
    return (
        (r0 == 0 and c0 == 0)
        or (r0 == 0 and c1 == w - 1)
        or (r1 == h - 1 and c0 == 0)
        or (r1 == h - 1 and c1 == w - 1)
    )


def _extract_legend(grid: np.ndarray, panel_box: tuple[int, int, int, int]) -> list[tuple[np.ndarray, int]]:
    pr0, pc0, pr1, pc1 = panel_box
    inner = grid[pr0 + 1 : pr1, pc0 + 1 : pc1]
    inner_h, inner_w = inner.shape
    assert inner_w == 5, "legend panel must store a 3x3 mask, one gap column, and one dot column"
    assert inner_h >= 3 and (inner_h + 1) % 4 == 0, "legend entries must be stacked with blank separators"

    entries: list[tuple[np.ndarray, int]] = []
    for idx in range((inner_h + 1) // 4):
        r0 = idx * 4
        symbol_window = inner[r0 : r0 + 3, 0:3]
        gap_col = inner[r0 : r0 + 3, 3]
        dot_column = inner[r0 : r0 + 3, 4]
        dot_color = int(dot_column[1])

        assert symbol_window.shape == (3, 3)
        assert np.all(gap_col == BG)
        assert dot_color not in (BG, LEGEND_BORDER, LEGEND_INK, FRAME_BORDER)
        assert int(dot_column[0]) == BG and int(dot_column[2]) == BG

        symbol_mask = symbol_window != BG
        assert np.any(symbol_mask), "legend symbol must not be empty"
        assert np.all(np.any(symbol_mask, axis=0)) and np.all(np.any(symbol_mask, axis=1)), "legend symbol must use its full 3x3 bbox"
        entries.append((symbol_mask, dot_color))

        if idx + 1 < (inner_h + 1) // 4:
            assert np.all(inner[r0 + 3, :] == BG), "legend entries must be separated by one blank row"

    masks = [mask for mask, _ in entries]
    for i, left in enumerate(masks):
        for j, right in enumerate(masks):
            if i >= j:
                continue
            assert not any(np.array_equal(np.rot90(left, k), right) for k in range(4)), "legend symbols must be unique up to rotation"
    return entries


def _downsample_symbol(mask: np.ndarray) -> np.ndarray:
    h, w = mask.shape
    assert h == w and h % 3 == 0, "cropped scaled symbol must be square with side divisible by three"
    k = h // 3
    small = np.zeros((3, 3), dtype=bool)
    for br in range(3):
        for bc in range(3):
            block = mask[br * k : (br + 1) * k, bc * k : (bc + 1) * k]
            val = bool(block[0, 0])
            assert np.all(block == val), "each scale block must be uniform"
            small[br, bc] = val
    return small


def main(grid: np.ndarray) -> np.ndarray:
    """
    Decode a corner legend of 3x3 masks, match each framed scaled-rotated copy, and fill the frame interior with the legend color.

    Input structure:
    - One corner contains a bordered legend panel with vertically stacked 3x3 one-color symbol masks, each paired with a single adjacent decode-color dot.
    - Elsewhere, several hollow rectangular frames contain one monochrome copy of a legend mask after integer scaling and a quarter-turn rotation, sometimes padded by background inside the frame.
    - The interior symbol colors are decoys chosen from other legend decode colors.

    Rule:
    - Read every legend entry as a binary 3x3 mask paired with its dot color.
    - For each non-legend frame, crop the bbox of the interior symbol, recover the scale from bbox_side / 3, downsample each k-by-k block back to a 3x3 binary mask, and compare all four rotations against the legend masks until exactly one match remains.
    - Fill the entire interior of that frame with the matched legend entry's decode color, leaving the legend panel and all frame borders unchanged.

    Key insight:
    - Size, orientation, and current color are all unreliable; only the crop -> normalize -> rotate -> match pipeline recovers the symbol identity from the in-grid legend.
    """
    g = np.asarray(grid, dtype=int)
    assert g.ndim == 2, "expected a 2D grid"

    bg = int(get_bg(g))
    components = find_components(g, bg=bg, connectivity=4, monochromatic=True, return_format="dict")

    panel_candidates = [
        comp
        for comp in components
        if _is_rect_outline(comp) and _touches_corner(tuple(comp["bbox"]), g.shape)
    ]
    assert len(panel_candidates) == 1, "expected exactly one corner legend panel"
    panel_box = tuple(panel_candidates[0]["bbox"])
    legend_entries = _extract_legend(g, panel_box)

    frame_candidates = [
        comp
        for comp in components
        if _is_rect_outline(comp) and tuple(comp["bbox"]) != panel_box
    ]
    assert frame_candidates, "expected at least one framed symbol"

    legend_masks = [mask for mask, _ in legend_entries]
    legend_colors = [int(color) for _, color in legend_entries]

    out = g.copy()
    for frame in frame_candidates:
        r0, c0, r1, c1 = frame["bbox"]
        assert r1 - r0 >= 2 and c1 - c0 >= 2, "frame must have a nonempty interior"
        inner = g[r0 + 1 : r1, c0 + 1 : c1]
        symbol_box = bbox(inner != bg)
        assert symbol_box is not None, "frame interior must contain a symbol"
        symbol_crop = crop_bbox(inner, symbol_box)
        small = _downsample_symbol(symbol_crop != bg)

        matches = []
        for idx, legend_mask in enumerate(legend_masks):
            if any(np.array_equal(np.rot90(small, rot), legend_mask) for rot in range(4)):
                matches.append(idx)
        assert len(matches) == 1, "normalized symbol must match exactly one legend entry"

        out[r0 + 1 : r1, c0 + 1 : c1] = legend_colors[matches[0]]

    return out


def _random_derangement(n: int) -> list[int]:
    items = list(range(n))
    while True:
        perm = items[:]
        random.shuffle(perm)
        if all(i != perm[i] for i in items):
            return perm


def _panel_outer_shape(symbol_count: int) -> tuple[int, int]:
    return 4 * symbol_count + 1, 7


def _panel_box(shape: tuple[int, int], symbol_count: int, corner: str) -> tuple[int, int, int, int]:
    panel_h, panel_w = _panel_outer_shape(symbol_count)
    h, w = shape
    if corner == "tl":
        r0, c0 = 0, 0
    elif corner == "tr":
        r0, c0 = 0, w - panel_w
    elif corner == "bl":
        r0, c0 = h - panel_h, 0
    else:
        r0, c0 = h - panel_h, w - panel_w
    return int(r0), int(c0), int(r0 + panel_h - 1), int(c0 + panel_w - 1)


def _place_panel(grid: np.ndarray, symbols: list[np.ndarray], decode_colors: list[int], corner: str) -> tuple[int, int, int, int]:
    panel_h, panel_w = _panel_outer_shape(len(symbols))
    r0, c0, r1, c1 = _panel_box(grid.shape, len(symbols), corner)
    paste(grid, _outline_mask((panel_h, panel_w), LEGEND_BORDER), r0, c0)
    for idx, (symbol, decode_color) in enumerate(zip(symbols, decode_colors)):
        er = r0 + 1 + idx * 4
        ec = c0 + 1
        patch = np.where(symbol, LEGEND_INK, BG).astype(int)
        paste(grid, patch, er, ec, transparent=BG)
        grid[er + 1, ec + 4] = int(decode_color)
    return int(r0), int(c0), int(r1), int(c1)


def _boxes_overlap(a: tuple[int, int, int, int], b: tuple[int, int, int, int], margin: int = 0) -> bool:
    ar0, ac0, ar1, ac1 = a
    br0, bc0, br1, bc1 = b
    return not (ar1 + margin < br0 or br1 + margin < ar0 or ac1 + margin < bc0 or bc1 + margin < ac0)


def _frame_outer_shape(scale: int, pads: tuple[int, int, int, int]) -> tuple[int, int]:
    top, bottom, left, right = pads
    return int(3 * scale + top + bottom + 2), int(3 * scale + left + right + 2)


def _expanded_area(shape: tuple[int, int]) -> int:
    h, w = shape
    return int((h + 2) * (w + 2))


def _frame_patch(symbol: np.ndarray, scale: int, rotation: int, trap_color: int, pads: tuple[int, int, int, int]) -> np.ndarray:
    scaled = scale_sprite(np.rot90(symbol, rotation).astype(int), scale)
    symbol_patch = np.where(scaled, trap_color, BG).astype(int)
    top, bottom, left, right = pads
    inner_h = symbol_patch.shape[0] + top + bottom
    inner_w = symbol_patch.shape[1] + left + right
    outer = _outline_mask((inner_h + 2, inner_w + 2), FRAME_BORDER)
    outer[1 + top : 1 + top + symbol_patch.shape[0], 1 + left : 1 + left + symbol_patch.shape[1]] = symbol_patch
    return outer


def _sample_symbols(symbol_count: int) -> list[np.ndarray]:
    symbols = [BASE_SYMBOLS[idx].copy() for idx in random.sample(range(len(BASE_SYMBOLS)), symbol_count)]
    random.shuffle(symbols)
    return symbols


def _sample_specs(symbol_count: int, decode_colors: list[int]) -> list[dict] | None:
    perm = _random_derangement(symbol_count)
    repeated_symbol = random.randrange(symbol_count)

    base_specs = [
        {
            "symbol_idx": symbol_idx,
            "trap_color": int(decode_colors[perm[symbol_idx]]),
        }
        for symbol_idx in range(symbol_count)
    ]

    extra_choices = [
        color
        for color in decode_colors
        if color != decode_colors[repeated_symbol] and color != base_specs[repeated_symbol]["trap_color"]
    ]
    if not extra_choices:
        extra_choices = [color for color in decode_colors if color != decode_colors[repeated_symbol]]
    base_specs.append({"symbol_idx": repeated_symbol, "trap_color": int(random.choice(extra_choices))})

    duplicate_indices = [idx for idx, spec in enumerate(base_specs) if spec["symbol_idx"] == repeated_symbol]
    assert len(duplicate_indices) == 2

    frame_count = len(base_specs)
    max_scale3 = 1 if frame_count == 5 else 2
    max_large_frames = 2 if frame_count == 5 else 3
    area_budget = 380 if frame_count == 5 else 440

    for _ in range(GEOMETRY_ATTEMPTS):
        specs = []
        for spec in base_specs:
            pads = tuple(int(random.choice(PAD_VALUES)) for _ in range(4))
            specs.append(
                {
                    **spec,
                    "scale": int(random.choice(SCALE_VALUES)),
                    "rotation": int(random.randrange(4)),
                    "pads": pads,
                }
            )

        if not any(spec["rotation"] != 0 for spec in specs):
            specs[random.randrange(frame_count)]["rotation"] = int(random.choice((1, 2, 3)))
        if not any(spec["scale"] > 1 for spec in specs):
            specs[random.randrange(frame_count)]["scale"] = 2
        if not any(sum(spec["pads"]) > 0 for spec in specs):
            idx = random.randrange(frame_count)
            pads = list(specs[idx]["pads"])
            pads[random.randrange(4)] = 1
            specs[idx]["pads"] = tuple(pads)

        left_idx, right_idx = duplicate_indices
        if (
            specs[left_idx]["scale"] == specs[right_idx]["scale"]
            and specs[left_idx]["rotation"] == specs[right_idx]["rotation"]
            and specs[left_idx]["pads"] == specs[right_idx]["pads"]
        ):
            specs[right_idx]["rotation"] = (int(specs[right_idx]["rotation"]) + 1) % 4

        patch_shapes = [_frame_outer_shape(spec["scale"], spec["pads"]) for spec in specs]
        if max(max(shape) for shape in patch_shapes) > 13:
            continue
        if sum(spec["scale"] == 3 for spec in specs) > max_scale3:
            continue
        if sum(max(shape) >= 12 for shape in patch_shapes) > max_large_frames:
            continue
        if sum(_expanded_area(shape) for shape in patch_shapes) > area_budget:
            continue

        for spec, patch_shape in zip(specs, patch_shapes):
            spec["patch_shape"] = patch_shape
        random.shuffle(specs)
        return specs

    return None


def _candidate_grid_shapes(panel_shape: tuple[int, int], patch_shapes: list[tuple[int, int]]) -> list[tuple[int, int]]:
    panel_h, panel_w = panel_shape
    max_ph = max(ph for ph, _ in patch_shapes)
    max_pw = max(pw for _, pw in patch_shapes)
    min_h = max(GRID_MIN, panel_h + 2, max_ph + 2)
    min_w = max(GRID_MIN, panel_w + 2, max_pw + 2)
    needed_area = (panel_h + 2) * (panel_w + 2) + sum(_expanded_area(shape) for shape in patch_shapes) + 16

    shapes = [
        (h, w)
        for h in range(min_h, GRID_MAX + 1)
        for w in range(min_w, GRID_MAX + 1)
        if h * w >= needed_area
    ]
    if not shapes:
        return [(GRID_MAX, GRID_MAX)]
    return sorted(shapes, key=lambda shape: (shape[0] * shape[1], random.random()))


def _layout_regions(
    shape: tuple[int, int],
    panel_box: tuple[int, int, int, int],
) -> list[tuple[int, int, int, int]]:
    h, w = shape
    pr0, pc0, pr1, pc1 = panel_box

    if pr0 == 0 and pc0 == 0:
        regions = [
            (1, pc1 + 2, pr1 - 1, w - 2),
            (pr1 + 2, 1, h - 2, w - 2),
        ]
    elif pr0 == 0 and pc1 == w - 1:
        regions = [
            (1, 1, pr1 - 1, pc0 - 2),
            (pr1 + 2, 1, h - 2, w - 2),
        ]
    elif pr1 == h - 1 and pc0 == 0:
        regions = [
            (1, 1, pr0 - 2, w - 2),
            (pr0 + 1, pc1 + 2, h - 2, w - 2),
        ]
    else:
        regions = [
            (1, 1, pr0 - 2, w - 2),
            (pr0 + 1, 1, h - 2, pc0 - 2),
        ]

    return [region for region in regions if region[0] <= region[2] and region[1] <= region[3]]


def _shelf_pack_region(
    region: tuple[int, int, int, int],
    items: list[tuple[int, tuple[int, int]]],
    mode: str,
) -> dict[int, tuple[int, int, int, int]] | None:
    if not items:
        return {}

    r0, c0, r1, c1 = region
    region_h = int(r1 - r0 + 1)
    region_w = int(c1 - c0 + 1)

    def order_key(item: tuple[int, tuple[int, int]]) -> tuple[float, ...]:
        _idx, (ph, pw) = item
        area = ph * pw
        if mode == "height":
            return (-ph, -pw, -area, random.random())
        if mode == "width":
            return (-pw, -ph, -area, random.random())
        return (-area, -max(ph, pw), -min(ph, pw), random.random())

    ordered = sorted(items, key=order_key)
    if any(ph > region_h or pw > region_w for _, (ph, pw) in ordered):
        return None

    shelves: list[dict] = []
    current_items: list[tuple[int, tuple[int, int]]] = []
    current_used_w = 0
    current_h = 0

    for item in ordered:
        _idx, (ph, pw) = item
        needed_w = pw if not current_items else pw + 1
        if current_items and current_used_w + needed_w > region_w:
            shelves.append({"items": current_items, "used_w": current_used_w, "height": current_h})
            current_items = []
            current_used_w = 0
            current_h = 0

        needed_w = pw if not current_items else pw + 1
        if current_used_w + needed_w > region_w:
            return None
        current_items.append(item)
        current_used_w += needed_w
        current_h = max(current_h, ph)

    if current_items:
        shelves.append({"items": current_items, "used_w": current_used_w, "height": current_h})

    total_h = sum(int(shelf["height"]) for shelf in shelves) + max(0, len(shelves) - 1)
    if total_h > region_h:
        return None

    y_offset = int(random.randint(0, region_h - total_h))
    placements: dict[int, tuple[int, int, int, int]] = {}
    row = int(r0 + y_offset)
    for shelf in shelves:
        shelf_items = shelf["items"]
        shelf_used_w = int(shelf["used_w"])
        shelf_h = int(shelf["height"])
        x_offset = int(random.randint(0, region_w - shelf_used_w))
        col = int(c0 + x_offset)
        for idx, (ph, pw) in shelf_items:
            placements[int(idx)] = (row, col, row + ph - 1, col + pw - 1)
            col += int(pw) + 1
        row += shelf_h + 1

    return placements


def _place_frames(
    shape: tuple[int, int],
    patch_shapes: list[tuple[int, int]],
    panel_box: tuple[int, int, int, int],
) -> list[tuple[int, int, int, int]] | None:
    regions = _layout_regions(shape, panel_box)
    if len(regions) != 2:
        return None

    region_a, region_b = regions
    assignments = list(range(1 << len(patch_shapes)))
    random.shuffle(assignments)

    modes = ["area", "height", "width"]
    random.shuffle(modes)

    for mask in assignments:
        left_items = []
        right_items = []
        for idx, patch_shape in enumerate(patch_shapes):
            target = right_items if (mask >> idx) & 1 else left_items
            target.append((idx, patch_shape))

        for mode_a in modes:
            placed_a = _shelf_pack_region(region_a, left_items, mode_a)
            if placed_a is None:
                continue
            for mode_b in modes:
                placed_b = _shelf_pack_region(region_b, right_items, mode_b)
                if placed_b is None:
                    continue

                placements = [None] * len(patch_shapes)
                for idx, box in {**placed_a, **placed_b}.items():
                    placements[idx] = box
                if all(box is not None for box in placements):
                    return [box for box in placements if box is not None]

    return None


def _semantic_assertions(grid: np.ndarray, metadata: dict) -> None:
    out = main(grid)
    assert out.shape == grid.shape
    assert not np.array_equal(out, grid), "output must differ from input"

    pr0, pc0, pr1, pc1 = metadata["panel_box"]
    assert np.array_equal(out[pr0 : pr1 + 1, pc0 : pc1 + 1], grid[pr0 : pr1 + 1, pc0 : pc1 + 1]), "legend panel must remain unchanged"

    frames = metadata["frames"]
    assert len({int(frame["symbol_idx"]) for frame in frames}) >= 2, "scene must contain at least two symbol identities"
    assert any(int(frame["rotation"]) != 0 for frame in frames), "scene must include a rotated symbol"
    assert any(int(frame["scale"]) > 1 for frame in frames), "scene must include a scaled symbol"
    assert any(sum(frame["pads"]) > 0 for frame in frames), "scene must require cropping away interior padding"

    by_symbol: dict[int, set[int]] = {}
    by_trap_color: dict[int, set[int]] = {}
    for frame in frames:
        symbol_idx = int(frame["symbol_idx"])
        trap_color = int(frame["trap_color"])
        decode_color = int(metadata["decode_colors"][symbol_idx])
        assert trap_color != decode_color, "trap color must be wrong for its symbol"
        by_symbol.setdefault(symbol_idx, set()).add(trap_color)
        by_trap_color.setdefault(trap_color, set()).add(decode_color)

        r0, c0, r1, c1 = frame["box"]
        fill = out[r0 + 1 : r1, c0 + 1 : c1]
        assert np.all(fill == decode_color), "frame interior must be recolored by decoded symbol identity"

    assert any(len(colors) >= 2 for colors in by_symbol.values()), "some symbol must appear in two different wrong colors"
    assert any(len(targets) >= 2 for targets in by_trap_color.values()) or any(
        len(colors) >= 2 for colors in by_symbol.values()
    ), "current color alone must not determine the output color"


def _generate_scene() -> tuple[np.ndarray, dict] | None:
    symbol_count = int(random.choice(SYMBOL_COUNT_CHOICES))
    symbols = _sample_symbols(symbol_count)
    decode_colors = list(random.sample(DECODE_COLOR_POOL, symbol_count))
    specs = _sample_specs(symbol_count, decode_colors)
    if specs is None:
        return None

    patch_shapes = [tuple(spec["patch_shape"]) for spec in specs]
    panel_shape = _panel_outer_shape(symbol_count)
    grid_shapes = _candidate_grid_shapes(panel_shape, patch_shapes)
    selected_shapes = grid_shapes[: max(1, LAYOUT_ATTEMPTS - 1)]
    if (GRID_MAX, GRID_MAX) not in selected_shapes:
        selected_shapes.append((GRID_MAX, GRID_MAX))

    for shape in selected_shapes:
        for corner in random.sample(("tl", "tr", "bl", "br"), 4):
            panel_box = _panel_box(shape, symbol_count, corner)
            placements = _place_frames(shape, patch_shapes, panel_box)
            if placements is None:
                continue

            grid = np.full(shape, BG, dtype=int)
            placed_panel_box = _place_panel(grid, symbols, decode_colors, corner)
            assert placed_panel_box == panel_box

            placed_frames: list[dict] = []
            for spec, box in zip(specs, placements):
                patch = _frame_patch(symbols[spec["symbol_idx"]], spec["scale"], spec["rotation"], spec["trap_color"], spec["pads"])
                assert patch.shape == tuple(spec["patch_shape"])
                paste(grid, patch, box[0], box[1])
                placed_frames.append(
                    {
                        "symbol_idx": int(spec["symbol_idx"]),
                        "trap_color": int(spec["trap_color"]),
                        "scale": int(spec["scale"]),
                        "rotation": int(spec["rotation"]),
                        "pads": tuple(int(v) for v in spec["pads"]),
                        "box": box,
                    }
                )

            metadata = {
                "decode_colors": [int(color) for color in decode_colors],
                "panel_box": panel_box,
                "frames": placed_frames,
            }
            try:
                _semantic_assertions(grid, metadata)
            except AssertionError:
                continue
            return grid, metadata

    return None


def generate_input() -> np.ndarray:
    for _ in range(MAX_ATTEMPTS):
        scene = _generate_scene()
        if scene is not None:
            return scene[0]
    raise RuntimeError("failed to generate a valid legend-decoding scene")
