from __future__ import annotations

import random
import sys
from pathlib import Path

import numpy as np

_UTILS_DIR = Path("/home/lixinyu/snap/new/solvers")
if str(_UTILS_DIR) not in sys.path:
    sys.path.append(str(_UTILS_DIR))

from arc_utils import bbox_coords, components_mask, draw_line, find_components, get_bg, paste


def _hole_regions(coords: np.ndarray) -> list[np.ndarray]:
    """Return enclosed non-object regions for one component as global (r,c) coords."""
    assert coords.ndim == 2 and coords.shape[1] == 2 and coords.shape[0] > 0

    r0, c0, r1, c1 = bbox_coords(coords)
    mask = np.zeros((r1 - r0 + 1, c1 - c0 + 1), dtype=bool)
    mask[coords[:, 0] - r0, coords[:, 1] - c0] = True

    padded = np.pad(mask, 1, constant_values=False)
    ph, pw = padded.shape
    holes: list[np.ndarray] = []
    for bg_coords in components_mask(~padded, connectivity=4):
        rr = bg_coords[:, 0]
        cc = bg_coords[:, 1]
        touches_outer = np.any((rr == 0) | (rr == ph - 1) | (cc == 0) | (cc == pw - 1))
        if touches_outer:
            continue
        global_coords = np.empty_like(bg_coords)
        global_coords[:, 0] = bg_coords[:, 0] + r0 - 1
        global_coords[:, 1] = bg_coords[:, 1] + c0 - 1
        holes.append(global_coords)
    return holes


def _analyze_objects(grid: np.ndarray) -> list[dict]:
    bg = int(get_bg(grid))
    comps = find_components(grid, bg=bg, connectivity=4, monochromatic=True, return_format="dict")
    objects: list[dict] = []
    for comp in comps:
        coords = comp["coords"]
        holes = _hole_regions(coords)
        if not holes:
            continue

        marker_color = None
        marker_pos = None
        for hole in holes:
            vals = grid[hole[:, 0], hole[:, 1]]
            hits = hole[vals != bg]
            if hits.shape[0] == 0:
                continue
            assert hits.shape[0] == 1, "each hole can contain at most one marker pixel"
            assert marker_pos is None, "each object can own at most one marker"
            rr, cc = map(int, hits[0])
            marker_pos = (rr, cc)
            marker_color = int(grid[rr, cc])

        bbox = tuple(map(int, comp["bbox"]))
        objects.append(
            {
                "color": int(comp["color"]),
                "coords": coords,
                "bbox": bbox,
                "hole_count": len(holes),
                "marker_color": marker_color,
                "marker_pos": marker_pos,
                "center_r": (bbox[0] + bbox[2]) / 2.0,
                "center_c": (bbox[1] + bbox[3]) / 2.0,
            }
        )

    assert objects, "expected at least one enclosed object"
    return objects


def _segments_from_objects(objects: list[dict]) -> list[dict]:
    by_holes: dict[int, list[dict]] = {}
    for obj in objects:
        by_holes.setdefault(int(obj["hole_count"]), []).append(obj)

    segments: list[dict] = []
    for hole_count, pair in by_holes.items():
        assert len(pair) == 2, "each hole count must belong to exactly one pair"
        markers = [obj for obj in pair if obj["marker_color"] is not None]
        assert len(markers) == 1, "each pair must have exactly one marker-bearing object"
        marker_color = int(markers[0]["marker_color"])

        same_row = pair[0]["center_r"] == pair[1]["center_r"]
        same_col = pair[0]["center_c"] == pair[1]["center_c"]
        assert same_row != same_col, "paired bbox centers must share exactly one axis"

        if same_row:
            row = int(round(pair[0]["center_r"]))
            left, right = sorted(pair, key=lambda obj: obj["center_c"])
            left_cols = left["coords"][left["coords"][:, 0] == row, 1]
            right_cols = right["coords"][right["coords"][:, 0] == row, 1]
            assert left_cols.size > 0 and right_cols.size > 0
            c0 = int(left_cols.max()) + 1
            c1 = int(right_cols.min()) - 1
            assert c0 <= c1, "paired objects must have background corridor between them"
            segments.append(
                {
                    "orient": "h",
                    "hole_count": int(hole_count),
                    "color": marker_color,
                    "r0": row,
                    "c0": c0,
                    "r1": row,
                    "c1": c1,
                }
            )
        else:
            col = int(round(pair[0]["center_c"]))
            top, bottom = sorted(pair, key=lambda obj: obj["center_r"])
            top_rows = top["coords"][top["coords"][:, 1] == col, 0]
            bottom_rows = bottom["coords"][bottom["coords"][:, 1] == col, 0]
            assert top_rows.size > 0 and bottom_rows.size > 0
            r0 = int(top_rows.max()) + 1
            r1 = int(bottom_rows.min()) - 1
            assert r0 <= r1, "paired objects must have background corridor between them"
            segments.append(
                {
                    "orient": "v",
                    "hole_count": int(hole_count),
                    "color": marker_color,
                    "r0": r0,
                    "c0": col,
                    "r1": r1,
                    "c1": col,
                }
            )
    return segments


def main(grid: np.ndarray) -> np.ndarray:
    """
    Connect each pair of closed objects that shares the same hole count using the marker color hidden inside one member of the pair.

    Input structure
    - The grid has a black background, several disconnected colored closed objects, and a few single-pixel markers inside some enclosed holes.
    - Every closed object is a single 4-connected monochrome component that encloses one or more hole regions.
    - Each hole count appears on exactly two objects, and the two bbox centers of that pair lie on the same row or on the same column.

    Rule
    - Count holes per object and pair objects by equal hole count, ignoring the objects' visible colors.
    - For each pair, find the unique marker pixel inside either member's holes and draw a one-cell-thick straight segment on the shared row or column between the two outlines using that marker color.
    - If a horizontal and a vertical segment cross, keep the color of the pair with the larger hole count at the crossing; preserve every object pixel and every marker pixel.

    Key insight
    - Topology chooses both the partner and the crossing priority: visible object color is only a decoy, while the marker only supplies paint after the hole-count pairing is known.
    """
    g = np.asarray(grid, dtype=int)
    assert g.ndim == 2

    out = g.copy()
    objects = _analyze_objects(g)
    segments = _segments_from_objects(objects)

    for seg in sorted(segments, key=lambda item: int(item["hole_count"])):
        draw_line(
            out,
            int(seg["r0"]),
            int(seg["c0"]),
            end_r=int(seg["r1"]),
            end_c=int(seg["c1"]),
            color=int(seg["color"]),
        )
    return out


# --------------- generator ---------------

BG = 0
GRID_MIN = 20
GRID_MAX = 29
PAIR_COUNT_CHOICES = (2, 3)
GEN_ATTEMPTS = 400
OBJECT_COLORS = (1, 2, 3, 5, 6, 7, 8, 9)
MARKER_COLORS = (1, 2, 3, 4, 5, 6, 7, 8, 9)
CELL_SIZES = (1, 3)
LAYOUTS_BY_HOLES = {
    1: ((1, 1), (1, 2), (2, 1), (2, 2)),
    2: ((1, 2), (2, 1), (2, 2)),
    3: ((1, 3), (3, 1), (2, 2)),
    4: ((2, 2),),
}


def _bbox_for(top: int, left: int, sprite: np.ndarray) -> tuple[int, int, int, int]:
    h, w = map(int, sprite.shape)
    return int(top), int(left), int(top + h - 1), int(left + w - 1)


def _boxes_clear(a: tuple[int, int, int, int], b: tuple[int, int, int, int], pad: int = 1) -> bool:
    return (
        a[2] + pad < b[0]
        or b[2] + pad < a[0]
        or a[3] + pad < b[1]
        or b[3] + pad < a[1]
    )


def _render_layout(
    rows: int,
    cols: int,
    row_heights: tuple[int, ...],
    col_widths: tuple[int, ...],
    selected: tuple[int, ...],
    obj_color: int,
    marker_color: int | None,
) -> np.ndarray | None:
    h = sum(row_heights) + rows + 1
    w = sum(col_widths) + cols + 1
    sprite = np.full((h, w), int(obj_color), dtype=int)

    row_starts = []
    cur = 1
    for size in row_heights:
        row_starts.append(cur)
        cur += size + 1
    col_starts = []
    cur = 1
    for size in col_widths:
        col_starts.append(cur)
        cur += size + 1

    hole_boxes: list[tuple[int, int, int, int]] = []
    for idx in selected:
        rr = idx // cols
        cc = idx % cols
        r0 = row_starts[rr]
        c0 = col_starts[cc]
        hh = row_heights[rr]
        ww = col_widths[cc]
        sprite[r0 : r0 + hh, c0 : c0 + ww] = BG
        hole_boxes.append((r0, c0, hh, ww))

    if marker_color is not None:
        roomy = [box for box in hole_boxes if box[2] * box[3] >= 2]
        if not roomy:
            return None
        r0, c0, hh, ww = random.choice(roomy)
        choices = [(r, c) for r in range(r0, r0 + hh) for c in range(c0, c0 + ww)]
        mr, mc = random.choice(choices)
        sprite[mr, mc] = int(marker_color)

    fg_coords = np.argwhere(sprite == int(obj_color))
    holes = _hole_regions(fg_coords)
    if len(holes) != len(selected):
        return None
    if marker_color is not None:
        hole_vals = [sprite[hole[:, 0], hole[:, 1]] for hole in holes]
        if sum(np.count_nonzero(vals != BG) for vals in hole_vals) != 1:
            return None
    return sprite


def _make_object(hole_count: int, obj_color: int, marker_color: int | None) -> np.ndarray | None:
    for _ in range(120):
        rows, cols = random.choice(LAYOUTS_BY_HOLES[int(hole_count)])
        row_heights = tuple(random.choice(CELL_SIZES) for _ in range(rows))
        col_widths = tuple(random.choice(CELL_SIZES) for _ in range(cols))
        selected = tuple(sorted(random.sample(range(rows * cols), k=int(hole_count))))
        sprite = _render_layout(rows, cols, row_heights, col_widths, selected, obj_color, marker_color)
        if sprite is None:
            continue
        if sprite.shape[0] % 2 == 1 and sprite.shape[1] % 2 == 1:
            return sprite
    return None


def _draw_expected(inp: np.ndarray, segments: list[dict]) -> np.ndarray:
    out = inp.copy()
    for seg in sorted(segments, key=lambda item: int(item["hole_count"])):
        draw_line(
            out,
            int(seg["r0"]),
            int(seg["c0"]),
            end_r=int(seg["r1"]),
            end_c=int(seg["c1"]),
            color=int(seg["color"]),
        )
    return out


def _build_pair_segments(pair_defs: list[dict]) -> list[dict]:
    segments: list[dict] = []
    for pair in pair_defs:
        a = pair["placements"][0]
        b = pair["placements"][1]
        hole_count = int(pair["hole_count"])
        marker_color = int(pair["marker_color"])
        if pair["orientation"] == "h":
            left, right = sorted((a, b), key=lambda item: item["left"])
            row = int(left["top"] + left["sprite"].shape[0] // 2)
            c0 = int(left["left"] + left["sprite"].shape[1])
            c1 = int(right["left"] - 1)
            segments.append(
                {"orient": "h", "hole_count": hole_count, "color": marker_color, "r0": row, "c0": c0, "r1": row, "c1": c1}
            )
        else:
            top, bottom = sorted((a, b), key=lambda item: item["top"])
            col = int(top["left"] + top["sprite"].shape[1] // 2)
            r0 = int(top["top"] + top["sprite"].shape[0])
            r1 = int(bottom["top"] - 1)
            segments.append(
                {"orient": "v", "hole_count": hole_count, "color": marker_color, "r0": r0, "c0": col, "r1": r1, "c1": col}
            )
    return segments


def _crossings(segments: list[dict]) -> list[tuple[tuple[int, int], dict, dict]]:
    found: list[tuple[tuple[int, int], dict, dict]] = []
    for i, first in enumerate(segments):
        for second in segments[i + 1 :]:
            if first["orient"] == second["orient"]:
                continue
            h_seg = first if first["orient"] == "h" else second
            v_seg = second if first["orient"] == "h" else first
            if h_seg["c0"] <= v_seg["c0"] <= h_seg["c1"] and v_seg["r0"] <= h_seg["r0"] <= v_seg["r1"]:
                found.append(((int(h_seg["r0"]), int(v_seg["c0"])), h_seg, v_seg))
    return found


def _validate_scene(inp: np.ndarray, pair_defs: list[dict], segments: list[dict]) -> bool:
    if max(inp.shape) > 30:
        return False

    expected = _draw_expected(inp, segments)
    actual = main(inp)
    if not np.array_equal(actual, expected):
        return False

    objects = _analyze_objects(inp)
    by_holes: dict[int, list[dict]] = {}
    for obj in objects:
        by_holes.setdefault(int(obj["hole_count"]), []).append(obj)
    if sorted(by_holes) != sorted(int(pair["hole_count"]) for pair in pair_defs):
        return False
    if any(len(items) != 2 for items in by_holes.values()):
        return False
    if sum(obj["marker_color"] is not None for obj in objects) != len(pair_defs):
        return False

    object_colors = []
    for pair in pair_defs:
        for placed in pair["placements"]:
            object_colors.append((int(pair["hole_count"]), int(placed["color"])))
    shared_color_across_counts = False
    for i, (holes_i, color_i) in enumerate(object_colors):
        for holes_j, color_j in object_colors[i + 1 :]:
            if color_i == color_j and holes_i != holes_j:
                shared_color_across_counts = True
                break
        if shared_color_across_counts:
            break
    if not shared_color_across_counts:
        return False

    crossings = _crossings(segments)
    if not crossings:
        return False
    for (rr, cc), seg_a, seg_b in crossings:
        winner = seg_a if int(seg_a["hole_count"]) > int(seg_b["hole_count"]) else seg_b
        if int(expected[rr, cc]) != int(winner["color"]):
            return False

    for seg in segments:
        if seg["orient"] == "h":
            if np.any(inp[int(seg["r0"]), int(seg["c0"]) : int(seg["c1"]) + 1] != BG):
                return False
        else:
            if np.any(inp[int(seg["r0"]) : int(seg["r1"]) + 1, int(seg["c0"])] != BG):
                return False

    return not np.array_equal(inp, expected)


def _sample_pair_specs(pair_count: int) -> list[dict]:
    hole_counts = sorted(random.sample((1, 2, 3, 4), k=pair_count))
    object_palette_size = max(1, pair_count - 1)
    object_palette = random.sample(list(OBJECT_COLORS), k=object_palette_size)
    marker_palette = random.sample([c for c in MARKER_COLORS if c not in object_palette], k=pair_count)

    pair_defs: list[dict] = []
    for idx, hole_count in enumerate(hole_counts):
        marker_color = int(marker_palette[idx])
        marker_side = random.randint(0, 1)
        placements = []
        for side in range(2):
            obj_color = int(object_palette[(idx + side) % len(object_palette)])
            marker = marker_color if side == marker_side else None
            sprite = _make_object(hole_count, obj_color, marker)
            if sprite is None:
                return []
            placements.append({"sprite": sprite, "color": obj_color, "marker_color": marker})
        pair_defs.append(
            {
                "hole_count": int(hole_count),
                "marker_color": marker_color,
                "orientation": None,
                "placements": placements,
            }
        )
    return pair_defs


def _place_cross_pair_objects(
    pair_defs: list[dict], grid_h: int, grid_w: int
) -> tuple[list[tuple[int, int, int, int]], list[dict]] | None:
    horizontal = pair_defs[0]
    vertical = pair_defs[1]
    horizontal["orientation"] = "h"
    vertical["orientation"] = "v"

    left_sprite = horizontal["placements"][0]["sprite"]
    right_sprite = horizontal["placements"][1]["sprite"]
    top_sprite = vertical["placements"][0]["sprite"]
    bottom_sprite = vertical["placements"][1]["sprite"]

    h_half = max(left_sprite.shape[0] // 2, right_sprite.shape[0] // 2)
    v_half = max(top_sprite.shape[1] // 2, bottom_sprite.shape[1] // 2)
    row_min = int(max(h_half + 2, top_sprite.shape[0] + 2))
    row_max = int(min(grid_h - h_half - 3, grid_h - bottom_sprite.shape[0] - 3))
    col_min = int(max(v_half + 2, left_sprite.shape[1] + 2))
    col_max = int(min(grid_w - v_half - 3, grid_w - right_sprite.shape[1] - 3))
    if row_min > row_max or col_min > col_max:
        return None

    axis_r = random.randint(row_min, row_max)
    axis_c = random.randint(col_min, col_max)

    def sample_horizontal(sprite: np.ndarray, side: str) -> tuple[int, int] | None:
        top = axis_r - sprite.shape[0] // 2
        if side == "left":
            lo = 1
            hi = axis_c - sprite.shape[1] - 1
        else:
            lo = axis_c + 1
            hi = grid_w - sprite.shape[1] - 2
        if lo > hi:
            return None
        left = random.randint(lo, hi)
        return top, left

    def sample_vertical(sprite: np.ndarray, side: str) -> tuple[int, int] | None:
        left = axis_c - sprite.shape[1] // 2
        if side == "top":
            lo = 1
            hi = axis_r - sprite.shape[0] - 1
        else:
            lo = axis_r + 1
            hi = grid_h - sprite.shape[0] - 2
        if lo > hi:
            return None
        top = random.randint(lo, hi)
        return top, left

    left_pos = sample_horizontal(left_sprite, "left")
    right_pos = sample_horizontal(right_sprite, "right")
    top_pos = sample_vertical(top_sprite, "top")
    bottom_pos = sample_vertical(bottom_sprite, "bottom")
    if None in (left_pos, right_pos, top_pos, bottom_pos):
        return None

    placed_boxes: list[tuple[int, int, int, int]] = []
    samples = [
        (horizontal["placements"][0], left_pos),
        (horizontal["placements"][1], right_pos),
        (vertical["placements"][0], top_pos),
        (vertical["placements"][1], bottom_pos),
    ]
    for placed, (top, left) in samples:
        placed["top"] = int(top)
        placed["left"] = int(left)
        box = _bbox_for(int(top), int(left), placed["sprite"])
        if any(not _boxes_clear(box, other, pad=1) for other in placed_boxes):
            return None
        placed_boxes.append(box)
    return placed_boxes, pair_defs


def _extra_pair_candidates(
    pair_def: dict,
    grid_h: int,
    grid_w: int,
    used_boxes: list[tuple[int, int, int, int]],
    cross_row: int,
    cross_col: int,
) -> list[dict]:
    candidates: list[dict] = []
    sprite_a = pair_def["placements"][0]["sprite"]
    sprite_b = pair_def["placements"][1]["sprite"]

    for orientation in ("h", "v"):
        if orientation == "h":
            max_half = max(sprite_a.shape[0] // 2, sprite_b.shape[0] // 2)
            for axis_r in range(max_half + 1, grid_h - max_half - 1):
                if axis_r == cross_row:
                    continue
                if axis_r > cross_row - 3 and axis_r < cross_row + 3:
                    continue
                top_a = axis_r - sprite_a.shape[0] // 2
                top_b = axis_r - sprite_b.shape[0] // 2
                for left_a in range(1, grid_w - sprite_a.shape[1] - 1):
                    box_a = _bbox_for(top_a, left_a, sprite_a)
                    if any(not _boxes_clear(box_a, other, pad=1) for other in used_boxes):
                        continue
                    for left_b in range(left_a + sprite_a.shape[1] + 2, grid_w - sprite_b.shape[1] - 1):
                        if left_a <= cross_col <= left_b + sprite_b.shape[1] - 1:
                            continue
                        box_b = _bbox_for(top_b, left_b, sprite_b)
                        if not _boxes_clear(box_a, box_b, pad=1):
                            continue
                        if any(not _boxes_clear(box_b, other, pad=1) for other in used_boxes):
                            continue
                        candidates.append(
                            {
                                "orientation": "h",
                                "positions": ((top_a, left_a), (top_b, left_b)),
                            }
                        )
        else:
            max_half = max(sprite_a.shape[1] // 2, sprite_b.shape[1] // 2)
            for axis_c in range(max_half + 1, grid_w - max_half - 1):
                if axis_c == cross_col:
                    continue
                if axis_c > cross_col - 3 and axis_c < cross_col + 3:
                    continue
                left_a = axis_c - sprite_a.shape[1] // 2
                left_b = axis_c - sprite_b.shape[1] // 2
                for top_a in range(1, grid_h - sprite_a.shape[0] - 1):
                    box_a = _bbox_for(top_a, left_a, sprite_a)
                    if any(not _boxes_clear(box_a, other, pad=1) for other in used_boxes):
                        continue
                    for top_b in range(top_a + sprite_a.shape[0] + 2, grid_h - sprite_b.shape[0] - 1):
                        if top_a <= cross_row <= top_b + sprite_b.shape[0] - 1:
                            continue
                        box_b = _bbox_for(top_b, left_b, sprite_b)
                        if not _boxes_clear(box_a, box_b, pad=1):
                            continue
                        if any(not _boxes_clear(box_b, other, pad=1) for other in used_boxes):
                            continue
                        candidates.append(
                            {
                                "orientation": "v",
                                "positions": ((top_a, left_a), (top_b, left_b)),
                            }
                        )
    return candidates


def _gen_attempt() -> np.ndarray | None:
    pair_count = random.choice(PAIR_COUNT_CHOICES)
    pair_defs = _sample_pair_specs(pair_count)
    if not pair_defs:
        return None

    if random.random() < 0.5:
        pair_defs[0], pair_defs[1] = pair_defs[1], pair_defs[0]
    grid_h = random.randint(GRID_MIN, GRID_MAX)
    grid_w = random.randint(GRID_MIN, GRID_MAX)

    placed = _place_cross_pair_objects(pair_defs, grid_h, grid_w)
    if placed is None:
        return None
    used_boxes, pair_defs = placed

    cross_row = int(pair_defs[0]["placements"][0]["top"] + pair_defs[0]["placements"][0]["sprite"].shape[0] // 2)
    cross_col = int(pair_defs[1]["placements"][0]["left"] + pair_defs[1]["placements"][0]["sprite"].shape[1] // 2)

    if pair_count == 3:
        pair_defs[2]["orientation"] = None
        candidates = _extra_pair_candidates(pair_defs[2], grid_h, grid_w, used_boxes, cross_row, cross_col)
        if not candidates:
            return None
        choice = random.choice(candidates)
        pair_defs[2]["orientation"] = choice["orientation"]
        for placed_obj, (top, left) in zip(pair_defs[2]["placements"], choice["positions"]):
            placed_obj["top"] = int(top)
            placed_obj["left"] = int(left)
            used_boxes.append(_bbox_for(int(top), int(left), placed_obj["sprite"]))

    grid = np.full((grid_h, grid_w), BG, dtype=int)
    for pair in pair_defs:
        for placed_obj in pair["placements"]:
            paste(grid, placed_obj["sprite"], int(placed_obj["top"]), int(placed_obj["left"]), transparent=BG)

    segments = _build_pair_segments(pair_defs)
    return grid if _validate_scene(grid, pair_defs, segments) else None


def generate_input() -> np.ndarray:
    for _ in range(GEN_ATTEMPTS):
        grid = _gen_attempt()
        if grid is not None:
            return grid
    raise RuntimeError("failed to generate a valid topology-pairing input")
