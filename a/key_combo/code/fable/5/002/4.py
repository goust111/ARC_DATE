from __future__ import annotations

from collections import Counter, defaultdict
import random
import sys
from typing import Iterable

import numpy as np


_UTILS_DIR = "/home/lixinyu/snap/new/solvers"
if _UTILS_DIR not in sys.path:
    sys.path.append(_UTILS_DIR)

try:
    from arc_utils import bbox_coords, colors_present, find_components, get_bg
except Exception:  # pragma: no cover - fallback for standalone loading.
    def get_bg(grid: np.ndarray) -> int:
        return int(Counter(map(int, grid.reshape(-1))).most_common(1)[0][0])

    def colors_present(grid: np.ndarray, *, exclude: Iterable[int] | None = None) -> list[int]:
        colors = set(map(int, grid.reshape(-1)))
        if exclude is not None:
            colors -= set(map(int, exclude))
        return sorted(colors)

    def bbox_coords(coords: np.ndarray) -> tuple[int, int, int, int]:
        r0, c0 = coords.min(axis=0)
        r1, c1 = coords.max(axis=0)
        return int(r0), int(c0), int(r1), int(c1)

    def _label_components(mask: np.ndarray, connectivity: int) -> list[np.ndarray]:
        H, W = mask.shape
        seen = np.zeros_like(mask, dtype=bool)
        if connectivity == 8:
            dirs = [(dr, dc) for dr in (-1, 0, 1) for dc in (-1, 0, 1) if (dr, dc) != (0, 0)]
        else:
            dirs = [(-1, 0), (1, 0), (0, -1), (0, 1)]
        comps = []
        for r in range(H):
            for c in range(W):
                if not mask[r, c] or seen[r, c]:
                    continue
                stack = [(r, c)]
                seen[r, c] = True
                pts = []
                while stack:
                    rr, cc = stack.pop()
                    pts.append((rr, cc))
                    for dr, dc in dirs:
                        nr, nc = rr + dr, cc + dc
                        if 0 <= nr < H and 0 <= nc < W and mask[nr, nc] and not seen[nr, nc]:
                            seen[nr, nc] = True
                            stack.append((nr, nc))
                comps.append(np.array(pts, dtype=int))
        return comps

    def find_components(
        grid: np.ndarray,
        *,
        bg: int | None = None,
        connectivity: int = 4,
        monochromatic: bool = True,
        colors: list[int] | tuple[int, ...] | None = None,
        min_area: int = 1,
        return_format: str = "dict",
    ) -> list[dict]:
        if bg is None:
            bg = get_bg(grid)
        if colors is None:
            scan_colors = colors_present(grid, exclude={bg})
        else:
            scan_colors = [int(c) for c in colors if int(c) != int(bg)]
        out = []
        if monochromatic:
            for color in scan_colors:
                for coords in _label_components(grid == int(color), connectivity):
                    if coords.shape[0] >= min_area:
                        out.append(
                            {
                                "color": int(color),
                                "coords": coords,
                                "bbox": bbox_coords(coords),
                                "area": int(coords.shape[0]),
                            }
                        )
            return out
        mask = (grid != int(bg)) if colors is None else ((grid != int(bg)) & np.isin(grid, scan_colors))
        for coords in _label_components(mask, connectivity):
            if coords.shape[0] >= min_area:
                out.append({"color": None, "coords": coords, "bbox": bbox_coords(coords), "area": int(coords.shape[0])})
        return out


Coord = tuple[int, int]
BBox = tuple[int, int, int, int]

N4 = ((-1, 0), (1, 0), (0, -1), (0, 1))
ARC_COLORS = tuple(range(10))


def _rect_perimeter(box: BBox) -> list[Coord]:
    r0, c0, r1, c1 = map(int, box)
    cells: list[Coord] = []
    for c in range(c0, c1 + 1):
        cells.append((r0, c))
        cells.append((r1, c))
    for r in range(r0 + 1, r1):
        cells.append((r, c0))
        cells.append((r, c1))
    return cells


def _rect_perimeter_set(box: BBox) -> set[Coord]:
    return set(_rect_perimeter(box))


def _inner_cells(box: BBox) -> list[Coord]:
    r0, c0, r1, c1 = map(int, box)
    return [(r, c) for r in range(r0 + 1, r1) for c in range(c0 + 1, c1)]


def _inner_area(box: BBox) -> int:
    r0, c0, r1, c1 = map(int, box)
    return int((r1 - r0 - 1) * (c1 - c0 - 1))


def _bbox_overlap_with_gap(a: BBox, b: BBox, gap: int) -> bool:
    return (
        a[0] - gap <= b[2] + gap
        and a[2] + gap >= b[0] - gap
        and a[1] - gap <= b[3] + gap
        and a[3] + gap >= b[1] - gap
    )


def _coords_box(coords: Iterable[Coord]) -> BBox:
    pts = list(coords)
    return min(r for r, _c in pts), min(c for _r, c in pts), max(r for r, _c in pts), max(c for _r, c in pts)


def _is_visible_frame_component(coords: np.ndarray) -> tuple[bool, BBox]:
    box = bbox_coords(coords)
    r0, c0, r1, c1 = box
    if r1 - r0 < 2 or c1 - c0 < 2:
        return False, box
    pts = {(int(r), int(c)) for r, c in coords}
    perim = _rect_perimeter_set(box)
    if not pts.issubset(perim):
        return False, box
    side_counts = [
        sum((r0, c) in pts for c in range(c0, c1 + 1)),
        sum((r1, c) in pts for c in range(c0, c1 + 1)),
        sum((r, c0) in pts for r in range(r0, r1 + 1)),
        sum((r, c1) in pts for r in range(r0, r1 + 1)),
    ]
    if min(side_counts) < 2:
        return False, box
    corners = {(r0, c0), (r0, c1), (r1, c0), (r1, c1)}
    if not corners.issubset(pts):
        return False, box
    return True, box


def _find_frames(g: np.ndarray, bg: int) -> tuple[int, list[dict]]:
    by_color: dict[int, list[dict]] = defaultdict(list)
    for color in colors_present(g, exclude={bg}):
        for comp in find_components(g, bg=bg, connectivity=4, monochromatic=True, colors=[int(color)]):
            ok, box = _is_visible_frame_component(comp["coords"])
            if ok:
                by_color[int(color)].append(
                    {
                        "color": int(color),
                        "box": box,
                        "area": _inner_area(box),
                        "coords": comp["coords"],
                    }
                )

    groups = [(color, frames) for color, frames in by_color.items() if len(frames) >= 2]
    if not groups and by_color:
        groups = list(by_color.items())
    assert len(groups) == 1, "expected one shared frame color"
    frame_color, frames = groups[0]
    frames = sorted(frames, key=lambda f: (f["box"][0], f["box"][1]))
    assert len(frames) >= 2, "expected several rectangular frames"
    return int(frame_color), frames


def _inside_or_on_box(p: Coord, box: BBox) -> bool:
    r, c = p
    r0, c0, r1, c1 = box
    return r0 <= r <= r1 and c0 <= c <= c1


def _is_outside_all_frames(p: Coord, frames: list[dict]) -> bool:
    return all(not _inside_or_on_box(p, f["box"]) for f in frames)


def _find_scatter_groups(g: np.ndarray, bg: int, frame_color: int, frames: list[dict]) -> dict[int, np.ndarray]:
    groups: dict[int, np.ndarray] = {}
    for color in colors_present(g, exclude={bg, frame_color}):
        comps = find_components(g, bg=bg, connectivity=4, monochromatic=True, colors=[int(color)])
        if not comps:
            continue
        if not all(int(comp["area"]) == 1 for comp in comps):
            continue
        coords = np.argwhere(g == int(color))
        pts = [(int(r), int(c)) for r, c in coords]
        if all(_is_outside_all_frames(p, frames) for p in pts):
            groups[int(color)] = coords
    return groups


def _component_plug_motion(coords: np.ndarray, box: BBox) -> tuple[int, int] | None:
    r0, c0, r1, c1 = map(int, box)
    pts = [(int(r), int(c)) for r, c in coords]
    has_inside = any(r0 < r < r1 and c0 < c < c1 for r, c in pts)
    if not has_inside:
        return None

    outside_sides: list[tuple[int, int]] = []
    if any(r < r0 for r, _c in pts):
        outside_sides.append((-1, 0))
    if any(r > r1 for r, _c in pts):
        outside_sides.append((1, 0))
    if any(c < c0 for _r, c in pts):
        outside_sides.append((0, -1))
    if any(c > c1 for _r, c in pts):
        outside_sides.append((0, 1))
    if len(outside_sides) != 1:
        return None

    outward = outside_sides[0]
    rs = [r for r, _c in pts]
    cs = [c for _r, c in pts]
    if outward == (-1, 0):
        return int(r0 - 1 - max(rs)), 0
    if outward == (1, 0):
        return int(r1 + 1 - min(rs)), 0
    if outward == (0, -1):
        return 0, int(c0 - 1 - max(cs))
    return 0, int(c1 + 1 - min(cs))


def _plug_components(g: np.ndarray, bg: int, frame_color: int) -> list[dict]:
    comps: list[dict] = []
    for color in colors_present(g, exclude={bg, frame_color}):
        comps.extend(find_components(g, bg=bg, connectivity=4, monochromatic=True, colors=[int(color)], min_area=2))
    return comps


def main(grid: np.ndarray) -> np.ndarray:
    """
    Fill the uniquely area-matched rectangle cavities and eject the plugs that intrude into those cavities.

    Input structure:
    - A dominant background contains several same-colored one-pixel hollow
      rectangular frames; some frame-edge cells can be hidden by plug blocks.
    - Solid non-frame plug components cross exactly one side of some frames and
      occupy one or more cells of the complete interior cavity.
    - Other non-background colors appear only as isolated scatter pixels outside
      all frames, with one color per scatter group.

    Rule:
    - Recover each full rectangular cavity from the frame border, counting every
      interior cell even when a plug currently occupies it.
    - For each scatter color, erase its scattered pixels and fill the only frame
      whose complete cavity area equals the scatter count with that scatter color.
    - For each plug intruding into a filled frame, translate the entire plug
      opposite its entry side until it is just outside the frame; unmatched frames
      and their plugs are left exactly as they were.

    Key insight:
    - The match uses the full hidden cavity capacity, not the currently empty
      cells; filling the cavity is what forces the covering plug out.
    """
    g = np.asarray(grid, dtype=int)
    assert g.ndim == 2, "expected a 2D grid"
    H, W = map(int, g.shape)
    bg = int(get_bg(g))

    frame_color, frames = _find_frames(g, bg)
    scatter_groups = _find_scatter_groups(g, bg, frame_color, frames)

    frames_by_area: dict[int, list[int]] = defaultdict(list)
    for i, frame in enumerate(frames):
        frames_by_area[int(frame["area"])].append(i)

    matched: dict[int, int] = {}
    matched_scatter_colors: set[int] = set()
    for color, coords in scatter_groups.items():
        count = int(coords.shape[0])
        target_indices = frames_by_area.get(count, [])
        if len(target_indices) != 1:
            continue
        idx = int(target_indices[0])
        assert idx not in matched, "two scatter groups cannot target one frame"
        matched[idx] = int(color)
        matched_scatter_colors.add(int(color))

    if not matched:
        return g.copy()

    all_plugs = _plug_components(g, bg, frame_color)
    plugs_by_frame: dict[int, list[dict]] = defaultdict(list)
    for frame_idx, frame in enumerate(frames):
        for comp in all_plugs:
            motion = _component_plug_motion(comp["coords"], frame["box"])
            if motion is None:
                continue
            plugs_by_frame[frame_idx].append(
                {
                    "color": int(comp["color"]),
                    "coords": comp["coords"],
                    "motion": motion,
                }
            )

    out = g.copy()

    for color in matched_scatter_colors:
        out[g == int(color)] = int(bg)

    for frame_idx, fill_color in matched.items():
        frame = frames[int(frame_idx)]
        box = frame["box"]

        for plug in plugs_by_frame.get(int(frame_idx), []):
            coords = plug["coords"]
            out[coords[:, 0], coords[:, 1]] = int(bg)

        for r, c in _rect_perimeter(box):
            out[int(r), int(c)] = int(frame_color)
        for r, c in _inner_cells(box):
            out[int(r), int(c)] = int(fill_color)

        for plug in plugs_by_frame.get(int(frame_idx), []):
            dr, dc = plug["motion"]
            for r, c in plug["coords"]:
                rr = int(r) + int(dr)
                cc = int(c) + int(dc)
                assert 0 <= rr < H and 0 <= cc < W, "ejected plug must stay in bounds"
                out[rr, cc] = int(plug["color"])

    return out


# ---------------- generator ----------------

GEN_ATTEMPTS = 600
GRID_SIZE_RANGE = (24, 30)
INNER_H_RANGE = (2, 5)
INNER_W_RANGE = (2, 6)
MAX_PLUG_SPAN = 3
MAX_PLUG_DEPTH = 2
PLUGIN_OUTSIDE_DEPTHS = (1, 2)
FRAME_GAP = 1
SCATTER_GAP = 1


def _dim_options() -> list[tuple[int, int, int]]:
    opts = []
    for h in range(INNER_H_RANGE[0], INNER_H_RANGE[1] + 1):
        for w in range(INNER_W_RANGE[0], INNER_W_RANGE[1] + 1):
            opts.append((h, w, h * w))
    return opts


DIM_OPTIONS = _dim_options()


def _plug_choices(ih: int, iw: int, intrusion: int | None = None) -> list[dict]:
    choices = []
    sides = ("top", "bottom", "left", "right")
    for side in sides:
        span_limit = iw if side in ("top", "bottom") else ih
        depth_limit = ih if side in ("top", "bottom") else iw
        for span in range(1, min(MAX_PLUG_SPAN, span_limit) + 1):
            for depth in range(1, min(MAX_PLUG_DEPTH, depth_limit) + 1):
                if intrusion is not None and span * depth != int(intrusion):
                    continue
                offsets = range(0, span_limit - span + 1)
                for offset in offsets:
                    for outside in PLUGIN_OUTSIDE_DEPTHS:
                        choices.append(
                            {
                                "side": side,
                                "span": span,
                                "depth": depth,
                                "offset": offset,
                                "outside": outside,
                                "intrusion": span * depth,
                            }
                        )
    return choices


def _random_plug(ih: int, iw: int, intrusion: int | None = None) -> dict | None:
    choices = _plug_choices(ih, iw, intrusion=intrusion)
    if not choices:
        return None
    return dict(random.choice(choices))


def _plug_coords(box: BBox, plug: dict) -> list[Coord]:
    r0, c0, r1, c1 = map(int, box)
    side = plug["side"]
    span = int(plug["span"])
    depth = int(plug["depth"])
    offset = int(plug["offset"])
    outside = int(plug["outside"])
    cells: list[Coord] = []

    if side == "top":
        cc0 = c0 + 1 + offset
        for r in range(r0 - outside, r0 + depth + 1):
            for c in range(cc0, cc0 + span):
                cells.append((r, c))
    elif side == "bottom":
        cc0 = c0 + 1 + offset
        for r in range(r1 - depth, r1 + outside + 1):
            for c in range(cc0, cc0 + span):
                cells.append((r, c))
    elif side == "left":
        rr0 = r0 + 1 + offset
        for r in range(rr0, rr0 + span):
            for c in range(c0 - outside, c0 + depth + 1):
                cells.append((r, c))
    else:
        rr0 = r0 + 1 + offset
        for r in range(rr0, rr0 + span):
            for c in range(c1 - depth, c1 + outside + 1):
                cells.append((r, c))
    return cells


def _plug_motion_from_spec(box: BBox, plug: dict) -> tuple[int, int]:
    side = plug["side"]
    depth = int(plug["depth"])
    if side == "top":
        return -depth - 1, 0
    if side == "bottom":
        return depth + 1, 0
    if side == "left":
        return 0, -depth - 1
    return 0, depth + 1


def _moved_plug_coords(box: BBox, plug: dict) -> list[Coord]:
    dr, dc = _plug_motion_from_spec(box, plug)
    return [(r + dr, c + dc) for r, c in _plug_coords(box, plug)]


def _spec_margin(spec: dict) -> tuple[int, int, int, int]:
    plug = spec.get("plug")
    if plug is None:
        return 0, 0, 0, 0
    depth = int(plug["depth"])
    outside = int(plug["outside"])
    top = bottom = left = right = 0
    side = plug["side"]
    needed = outside + depth + 1
    if side == "top":
        top = needed
    elif side == "bottom":
        bottom = needed
    elif side == "left":
        left = needed
    else:
        right = needed
    return top, bottom, left, right


def _choose_specs() -> list[dict] | None:
    by_area: dict[int, list[tuple[int, int]]] = defaultdict(list)
    for ih, iw, area in DIM_OPTIONS:
        by_area[area].append((ih, iw))

    traps = []
    for area in sorted(by_area):
        decoys = []
        for decoy_area in sorted(by_area):
            intrusion = decoy_area - area
            if intrusion <= 0 or intrusion > MAX_PLUG_SPAN * MAX_PLUG_DEPTH:
                continue
            dims = [(ih, iw) for ih, iw in by_area[decoy_area] if _plug_choices(ih, iw, intrusion=intrusion)]
            if dims:
                decoys.append((decoy_area, dims, intrusion))
        if len({d[0] for d in decoys}) >= 2:
            traps.append((area, decoys))
    if not traps:
        return None

    trap_area, decoys = random.choice(traps)
    decoy_by_area: dict[int, list[tuple[int, int, int]]] = defaultdict(list)
    for decoy_area, dims, intrusion in decoys:
        for ih, iw in dims:
            decoy_by_area[decoy_area].append((ih, iw, intrusion))
    decoy_areas = random.sample(list(decoy_by_area), 2)

    used_areas = {trap_area, *decoy_areas}
    other_match_areas = [area for area in by_area if area not in used_areas]
    if not other_match_areas:
        return None
    other_match_area = random.choice(other_match_areas)
    used_areas.add(other_match_area)

    specs: list[dict] = []
    tih, tiw = random.choice(by_area[trap_area])
    trap_plug = _random_plug(tih, tiw)
    if trap_plug is None:
        return None
    specs.append({"inner_h": tih, "inner_w": tiw, "area": trap_area, "matched": True, "plug": trap_plug, "trap": True})

    mih, miw = random.choice(by_area[other_match_area])
    maybe_plug = _random_plug(mih, miw) if random.random() < 0.55 else None
    specs.append({"inner_h": mih, "inner_w": miw, "area": other_match_area, "matched": True, "plug": maybe_plug})

    for decoy_area in decoy_areas:
        dih, diw, intrusion = random.choice(decoy_by_area[decoy_area])
        plug = _random_plug(dih, diw, intrusion=intrusion)
        if plug is None:
            return None
        specs.append({"inner_h": dih, "inner_w": diw, "area": decoy_area, "matched": False, "plug": plug, "remaining_equals": trap_area})

    if random.random() < 0.45:
        extra_areas = [area for area in by_area if area not in used_areas]
        if extra_areas:
            area = random.choice(extra_areas)
            ih, iw = random.choice(by_area[area])
            plug = _random_plug(ih, iw) if random.random() < 0.35 else None
            specs.append({"inner_h": ih, "inner_w": iw, "area": area, "matched": False, "plug": plug})

    random.shuffle(specs)
    return specs


def _footprint_for_box(box: BBox, spec: dict) -> set[Coord]:
    r0, c0, r1, c1 = box
    cells = {(r, c) for r in range(r0, r1 + 1) for c in range(c0, c1 + 1)}
    if spec.get("plug") is not None:
        cells.update(_plug_coords(box, spec["plug"]))
        cells.update(_moved_plug_coords(box, spec["plug"]))
    return cells


def _expanded_cells(cells: Iterable[Coord], shape: tuple[int, int], gap: int) -> set[Coord]:
    H, W = shape
    out: set[Coord] = set()
    for r, c in cells:
        for dr in range(-gap, gap + 1):
            for dc in range(-gap, gap + 1):
                rr, cc = r + dr, c + dc
                if 0 <= rr < H and 0 <= cc < W:
                    out.add((rr, cc))
    return out


def _place_specs(shape: tuple[int, int], specs: list[dict]) -> list[dict] | None:
    H, W = shape
    occupied: set[Coord] = set()
    placed: list[dict] = []
    specs_order = sorted(specs, key=lambda s: (s["inner_h"] + 2) * (s["inner_w"] + 2), reverse=True)

    for spec in specs_order:
        oh = int(spec["inner_h"]) + 2
        ow = int(spec["inner_w"]) + 2
        top, bottom, left, right = _spec_margin(spec)
        candidates = []
        for r0 in range(top, H - bottom - oh + 1):
            for c0 in range(left, W - right - ow + 1):
                box = (r0, c0, r0 + oh - 1, c0 + ow - 1)
                footprint = _footprint_for_box(box, spec)
                if any(not (0 <= r < H and 0 <= c < W) for r, c in footprint):
                    continue
                if occupied & _expanded_cells(footprint, shape, FRAME_GAP):
                    continue
                candidates.append((box, footprint))
        if not candidates:
            return None
        box, footprint = random.choice(candidates)
        spec2 = dict(spec)
        spec2["box"] = box
        placed.append(spec2)
        occupied.update(_expanded_cells(footprint, shape, FRAME_GAP))
    return placed


def _paint_cells(grid: np.ndarray, cells: Iterable[Coord], color: int) -> None:
    for r, c in cells:
        grid[int(r), int(c)] = int(color)


def _draw_frame(grid: np.ndarray, box: BBox, color: int) -> None:
    _paint_cells(grid, _rect_perimeter(box), color)


def _draw_plug(grid: np.ndarray, box: BBox, plug: dict, color: int) -> None:
    _paint_cells(grid, _plug_coords(box, plug), color)


def _candidate_scatter_cells(grid: np.ndarray, reserved: set[Coord], color: int, bg: int) -> list[Coord]:
    H, W = grid.shape
    cells = []
    for r in range(H):
        for c in range(W):
            if int(grid[r, c]) != int(bg):
                continue
            if (r, c) in reserved:
                continue
            clear = True
            for dr in range(-SCATTER_GAP, SCATTER_GAP + 1):
                for dc in range(-SCATTER_GAP, SCATTER_GAP + 1):
                    rr, cc = r + dr, c + dc
                    if 0 <= rr < H and 0 <= cc < W:
                        if int(grid[rr, cc]) == int(color) or (rr, cc) in reserved:
                            clear = False
                            break
                if not clear:
                    break
            if clear:
                cells.append((r, c))
    return cells


def _place_scatter_group(grid: np.ndarray, reserved: set[Coord], color: int, count: int, bg: int) -> list[Coord] | None:
    chosen: list[Coord] = []
    for _ in range(int(count)):
        candidates = _candidate_scatter_cells(grid, reserved, int(color), int(bg))
        if not candidates:
            return None
        cell = random.choice(candidates)
        grid[cell] = int(color)
        chosen.append(cell)
        reserved.update(_expanded_cells([cell], grid.shape, 0))
    return chosen


def _build_pair(shape: tuple[int, int], specs: list[dict], bg: int, frame_color: int, scatter_colors: list[int], plug_colors: list[int]) -> tuple[np.ndarray, np.ndarray, dict] | None:
    grid = np.full(shape, int(bg), dtype=int)
    expected = np.full(shape, int(bg), dtype=int)

    plugged_specs = [spec for spec in specs if spec.get("plug") is not None]
    plug_color_by_id = {id(spec): plug_colors[i] for i, spec in enumerate(plugged_specs)}
    scatter_color_by_id = {id(spec): scatter_colors[i] for i, spec in enumerate([s for s in specs if s["matched"]])}

    output_reserved: set[Coord] = set()
    for spec in specs:
        box = spec["box"]
        _draw_frame(grid, box, frame_color)
        _draw_frame(expected, box, frame_color)
        output_reserved.update(_footprint_for_box(box, spec))

    for spec in plugged_specs:
        color = plug_color_by_id[id(spec)]
        _draw_plug(grid, spec["box"], spec["plug"], color)
        _draw_plug(expected, spec["box"], spec["plug"], color)

    scatter_reserved = _expanded_cells(output_reserved, shape, 1)
    scatter_by_color: dict[int, list[Coord]] = {}
    for spec in [s for s in specs if s["matched"]]:
        color = scatter_color_by_id[id(spec)]
        cells = _place_scatter_group(grid, scatter_reserved, color, int(spec["area"]), bg)
        if cells is None:
            return None
        scatter_by_color[color] = cells

    expected = grid.copy()
    for color, cells in scatter_by_color.items():
        _paint_cells(expected, cells, bg)

    for spec in specs:
        if not spec["matched"]:
            continue
        fill_color = scatter_color_by_id[id(spec)]
        box = spec["box"]
        if spec.get("plug") is not None:
            _paint_cells(expected, _plug_coords(box, spec["plug"]), bg)
        _draw_frame(expected, box, frame_color)
        _paint_cells(expected, _inner_cells(box), fill_color)
        if spec.get("plug") is not None:
            plug_color = plug_color_by_id[id(spec)]
            _paint_cells(expected, _moved_plug_coords(box, spec["plug"]), plug_color)

    metadata = {
        "specs": specs,
        "scatter_by_color": scatter_by_color,
        "trap_area": next(spec["area"] for spec in specs if spec.get("trap")),
    }
    return grid, expected, metadata


def _remaining_empty_counts(grid: np.ndarray, specs: list[dict], bg: int) -> list[int]:
    counts = []
    for spec in specs:
        cells = _inner_cells(spec["box"])
        counts.append(sum(int(grid[r, c]) == int(bg) for r, c in cells))
    return counts


def _semantic_checks(inp: np.ndarray, out: np.ndarray, expected: np.ndarray, metadata: dict, bg: int) -> bool:
    if inp.shape != out.shape or max(inp.shape) > 30:
        return False
    if not np.array_equal(out, expected):
        return False

    specs = metadata["specs"]
    matched = [spec for spec in specs if spec["matched"]]
    unmatched = [spec for spec in specs if not spec["matched"]]
    if len(matched) < 2 or len(unmatched) < 2:
        return False
    if not any(spec.get("plug") is not None for spec in matched):
        return False
    if not any(spec.get("plug") is not None for spec in unmatched):
        return False

    trap_area = int(metadata["trap_area"])
    remaining = _remaining_empty_counts(inp, specs, bg)
    if sum(1 for x in remaining if int(x) == trap_area) < 2:
        return False

    full_areas = [int(spec["area"]) for spec in specs]
    scatter_counts = sorted(len(cells) for cells in metadata["scatter_by_color"].values())
    if len(full_areas) != len(set(full_areas)):
        return False
    if not all(full_areas.count(count) == 1 for count in scatter_counts):
        return False
    if any(np.array_equal(inp, out) for _ in [0]):
        return False
    return True


def _gen_attempt() -> np.ndarray | None:
    H = random.randint(*GRID_SIZE_RANGE)
    W = random.randint(*GRID_SIZE_RANGE)
    specs = _choose_specs()
    if specs is None:
        return None
    placed = _place_specs((H, W), specs)
    if placed is None:
        return None

    plugged_count = sum(1 for spec in placed if spec.get("plug") is not None)
    matched_count = sum(1 for spec in placed if spec["matched"])
    needed_colors = 2 + plugged_count + matched_count
    if needed_colors > len(ARC_COLORS):
        return None
    palette = random.sample(list(ARC_COLORS), needed_colors)
    bg = palette[0]
    frame_color = palette[1]
    scatter_colors = palette[2 : 2 + matched_count]
    plug_colors = palette[2 + matched_count :]

    pair = _build_pair((H, W), placed, bg, frame_color, scatter_colors, plug_colors)
    if pair is None:
        return None
    inp, expected, metadata = pair
    if int(get_bg(inp)) != int(bg):
        return None
    try:
        actual = main(inp)
    except Exception:
        return None
    if not _semantic_checks(inp, actual, expected, metadata, bg):
        return None
    return inp


def generate_input() -> np.ndarray:
    for _ in range(GEN_ATTEMPTS):
        grid = _gen_attempt()
        if grid is not None:
            return grid
    raise RuntimeError(f"failed to generate valid input after {GEN_ATTEMPTS} attempts")
