from __future__ import annotations

from collections import Counter, deque
import random

import numpy as np


BG = 0
PALETTE = tuple(range(1, 10))

# Generator layout.
GEN_ATTEMPTS = 400
GRID_H_RANGE = (24, 28)
LEGEND_ZONE_W = 5
PLAY_LEFT = LEGEND_ZONE_W + 2
LANE_GAP = 2
LANE_W_RANGE = (3, 4)
PLATFORM_COUNT = 5
PLATFORM_START_MIN = 4
PLATFORM_START_MAX = 6
PLATFORM_ROW_STEP = 4
OBJECT_H_RANGE = (2, 3)
MAX_LIFT = 20


def _get_bg(grid: np.ndarray) -> int:
    return int(Counter(map(int, grid.ravel())).most_common(1)[0][0])


def _bbox(coords: list[tuple[int, int]]) -> tuple[int, int, int, int]:
    rows = [r for r, _ in coords]
    cols = [c for _, c in coords]
    return min(rows), min(cols), max(rows), max(cols)


def _find_components(grid: np.ndarray, bg: int) -> list[dict]:
    h, w = grid.shape
    seen = np.zeros((h, w), dtype=bool)
    comps = []
    for r in range(h):
        for c in range(w):
            color = int(grid[r, c])
            if color == bg or seen[r, c]:
                continue
            queue = deque([(r, c)])
            seen[r, c] = True
            coords: list[tuple[int, int]] = []
            while queue:
                rr, cc = queue.popleft()
                coords.append((int(rr), int(cc)))
                for dr, dc in ((-1, 0), (1, 0), (0, -1), (0, 1)):
                    nr, nc = rr + dr, cc + dc
                    if 0 <= nr < h and 0 <= nc < w and not seen[nr, nc] and int(grid[nr, nc]) == color:
                        seen[nr, nc] = True
                        queue.append((nr, nc))
            comps.append(
                {
                    "color": color,
                    "coords": coords,
                    "area": len(coords),
                    "bbox": _bbox(coords),
                }
            )
    return comps


def _is_legend(comp: dict) -> bool:
    r0, c0, r1, c1 = comp["bbox"]
    return int(comp["area"]) == 2 and r0 == r1 and c1 == c0 + 1


def _is_platform(comp: dict) -> bool:
    r0, c0, r1, c1 = comp["bbox"]
    return r0 == r1 and int(comp["area"]) == (c1 - c0 + 1) and (c1 - c0 + 1) >= 3


def _legend_pairs(grid: np.ndarray, bg: int) -> list[tuple[int, int, int]]:
    h, w = grid.shape
    used: set[tuple[int, int]] = set()
    pairs = []
    for r in range(h):
        for c in range(w - 1):
            if (r, c) in used or (r, c + 1) in used:
                continue
            left = int(grid[r, c])
            right = int(grid[r, c + 1])
            if left == bg or right == bg or left == right:
                continue
            pair_cells = {(r, c), (r, c + 1)}
            isolated = True
            for rr, cc in pair_cells:
                for dr in (-1, 0, 1):
                    for dc in (-1, 0, 1):
                        nr, nc = rr + dr, cc + dc
                        if not (0 <= nr < h and 0 <= nc < w):
                            continue
                        if (nr, nc) in pair_cells:
                            continue
                        if int(grid[nr, nc]) != bg:
                            isolated = False
                            break
                    if not isolated:
                        break
                if not isolated:
                    break
            if isolated:
                used |= pair_cells
                pairs.append((r, c, c + 1))
    return pairs


def _parse_scene(grid: np.ndarray) -> tuple[int, dict[int, set[int]], list[dict], list[dict]]:
    g = np.asarray(grid, dtype=int)
    assert g.ndim == 2
    bg = _get_bg(g)

    legends: dict[int, set[int]] = {}
    legend_cells: set[tuple[int, int]] = set()
    for r, c0, c1 in _legend_pairs(g, bg):
        platform_color = int(g[r, c0])
        object_color = int(g[r, c1])
        legends.setdefault(object_color, set()).add(platform_color)
        legend_cells.add((r, c0))
        legend_cells.add((r, c1))

    scene = g.copy()
    for r, c in legend_cells:
        scene[r, c] = bg

    comps = _find_components(scene, bg)
    assert comps, "expected at least one non-background component outside legends"

    platforms: list[dict] = []
    objects: list[dict] = []

    for comp in comps:
        if _is_platform(comp):
            r0, c0, _, c1 = comp["bbox"]
            platforms.append(
                {
                    "color": int(comp["color"]),
                    "row": int(r0),
                    "c0": int(c0),
                    "c1": int(c1),
                }
            )
        else:
            objects.append(comp)

    assert legends, "expected at least one legend domino"
    assert platforms, "expected at least one platform"
    assert objects, "expected at least one falling object"

    platform_colors = [p["color"] for p in platforms]
    assert len(platform_colors) == len(set(platform_colors)), "platform colors must be unique"

    spans = []
    for obj in objects:
        r0, c0, r1, c1 = obj["bbox"]
        assert r1 > r0, "objects must not degenerate into one-row platform-like bars"
        for s0, s1 in spans:
            assert c1 < s0 or c0 > s1, "object horizontal projections must not overlap"
        spans.append((c0, c1))

    platforms.sort(key=lambda p: (p["row"], p["c0"]))
    objects.sort(key=lambda comp: comp["bbox"][1])
    return bg, legends, platforms, objects


def _candidate_platform_shifts(obj: dict, platforms: list[dict]) -> list[tuple[int, dict]]:
    coords = np.array(obj["coords"], dtype=int)
    cols = coords[:, 1]
    candidates: list[tuple[int, dict]] = []
    for platform in platforms:
        supported = (cols >= platform["c0"]) & (cols <= platform["c1"])
        if not np.any(supported):
            continue
        support_bottom = int(coords[supported, 0].max())
        shift = int(platform["row"] - 1 - support_bottom)
        if shift >= 0:
            candidates.append((shift, platform))
    return candidates


def _landing_shift(obj: dict, platforms: list[dict], catch_colors: set[int], height: int, *, strategy: str = "true") -> int:
    coords = np.array(obj["coords"], dtype=int)
    bottom_shift = int(height - 1 - coords[:, 0].max())
    assert bottom_shift >= 0

    candidates = _candidate_platform_shifts(obj, platforms)
    catchable = [(shift, platform) for shift, platform in candidates if platform["color"] in catch_colors]

    if strategy == "nearest_any":
        return min([bottom_shift] + [shift for shift, _ in candidates])
    if strategy == "lowest_match":
        return bottom_shift if not catchable else max(shift for shift, _ in catchable)
    if strategy == "true":
        return bottom_shift if not catchable else min(shift for shift, _ in catchable)
    raise ValueError(f"unknown strategy: {strategy}")


def _render_with_strategy(grid: np.ndarray, strategy: str) -> np.ndarray:
    g = np.asarray(grid, dtype=int)
    h, _ = g.shape
    bg, legends, platforms, objects = _parse_scene(g)
    out = g.copy()

    for obj in objects:
        for r, c in obj["coords"]:
            out[r, c] = bg

    for obj in objects:
        color = int(obj["color"])
        shift = _landing_shift(obj, platforms, legends.get(color, set()), h, strategy=strategy)
        for r, c in obj["coords"]:
            nr = int(r + shift)
            nc = int(c)
            assert out[nr, nc] == bg, "objects must land in empty cells"
            out[nr, nc] = color
    return out


def main(grid: np.ndarray) -> np.ndarray:
    """
    Drop each suspended object through irrelevant platforms until the legend-selected platform catches it.

    Input structure:
    - The background is black; several distinct-color horizontal platform segments sit at different heights.
    - Several single-color 4-connected falling objects hang above the platform field, and their horizontal projections do not overlap.
    - Isolated horizontal two-pixel legend dominos encode keys: left color = platform color, right color = object color.

    Rule:
    - Keep every legend domino and every platform fixed in place.
    - Each object falls straight downward.
    - When a falling object reaches a platform, it stops only if a legend domino contains that platform color on the left and that object's color on the right; otherwise it passes through and keeps falling.
    - If no catchable platform lies on its vertical path, the object lands on the bottom row.

    Key insight:
    - The tiny dominos are not moving pieces; they are a global platform-color/object-color permission table that decides which visible platforms are solid for each falling object.
    """
    return _render_with_strategy(np.asarray(grid, dtype=int), "true")


def _make_object_mask(height: int, width: int) -> np.ndarray:
    mask = np.zeros((height, width), dtype=bool)
    mask[-1, :] = True
    prev_lo, prev_hi = 0, width - 1
    for r in range(height - 2, -1, -1):
        length = random.randint(1, width)
        choices = []
        for lo in range(0, width - length + 1):
            hi = lo + length - 1
            if not (hi < prev_lo or lo > prev_hi):
                choices.append((lo, hi))
        lo, hi = random.choice(choices)
        mask[r, lo : hi + 1] = True
        prev_lo, prev_hi = lo, hi
    return mask


def _stamp(canvas: np.ndarray, mask: np.ndarray, r0: int, c0: int, color: int) -> None:
    h, w = mask.shape
    patch = canvas[r0 : r0 + h, c0 : c0 + w]
    assert patch.shape == mask.shape
    assert not np.any((patch != BG) & mask), "stamp collision"
    patch[mask] = int(color)


def _platform_rows(height: int) -> list[int]:
    max_start = min(PLATFORM_START_MAX, height - 4 - PLATFORM_ROW_STEP * (PLATFORM_COUNT - 1))
    if max_start < PLATFORM_START_MIN:
        raise ValueError("grid too short for five spaced platform rows")
    start = random.randint(PLATFORM_START_MIN, max_start)
    return [start + PLATFORM_ROW_STEP * i for i in range(PLATFORM_COUNT)]


def _legend_positions(height: int, count: int) -> list[tuple[int, int]]:
    rows = random.sample(list(range(1, height - 1, 2)), count)
    return [(int(r), random.randint(1, LEGEND_ZONE_W - 2)) for r in sorted(rows)]


def _scene_spec() -> dict | None:
    modes = ["top_multi", "lower", "bottom_platform"]
    if random.random() < 0.5:
        modes.append("empty_fall")
    random.shuffle(modes)

    lane_widths = [random.randint(*LANE_W_RANGE) for _ in modes]
    lane_specs = []
    x = PLAY_LEFT
    for mode, lane_w in zip(modes, lane_widths):
        lane_specs.append({"mode": mode, "c0": x, "c1": x + lane_w - 1, "lane_w": lane_w})
        x += lane_w + LANE_GAP
    width = lane_specs[-1]["c1"] + 2
    if width > 30:
        return None

    height = random.randint(*GRID_H_RANGE)
    rows = _platform_rows(height)
    row_perm = random.sample(rows, len(rows))
    row_groups = {
        "top_multi": sorted(row_perm[:2]),
        "lower": sorted(row_perm[2:4]),
        "bottom_platform": [row_perm[4]],
    }

    colors = random.sample(PALETTE, PLATFORM_COUNT + len(lane_specs))
    platform_colors = colors[:PLATFORM_COUNT]
    object_colors = colors[PLATFORM_COUNT:]
    color_cursor = 0

    for lane, obj_color in zip(lane_specs, object_colors):
        lane["object_color"] = obj_color
        height_obj = random.randint(*OBJECT_H_RANGE)
        width_obj = random.randint(2, lane["lane_w"])
        mask = _make_object_mask(height_obj, width_obj)
        lane["mask"] = mask
        lane["obj_c0"] = random.randint(lane["c0"], lane["c1"] - width_obj + 1)

        mode = lane["mode"]
        if mode == "empty_fall":
            lane["platforms"] = []
            lane["legend_colors"] = []
            lane["settled_r0"] = height - height_obj
            continue

        lane_rows = row_groups[mode]
        lane_platforms = []
        for row in lane_rows:
            lane_platforms.append(
                {
                    "row": int(row),
                    "c0": int(lane["c0"]),
                    "c1": int(lane["c1"]),
                    "color": int(platform_colors[color_cursor]),
                }
            )
            color_cursor += 1
        lane_platforms.sort(key=lambda p: p["row"])
        lane["platforms"] = lane_platforms

        if mode == "top_multi":
            lane["legend_colors"] = [p["color"] for p in lane_platforms]
            lane["settled_r0"] = lane_platforms[0]["row"] - height_obj
        elif mode == "lower":
            lane["legend_colors"] = [lane_platforms[-1]["color"]]
            lane["settled_r0"] = lane_platforms[-1]["row"] - height_obj
        else:
            lane["legend_colors"] = []
            lane["settled_r0"] = height - height_obj

    assert color_cursor == PLATFORM_COUNT

    for lane in lane_specs:
        max_lift = min(MAX_LIFT, int(lane["settled_r0"]))
        if lane["platforms"]:
            top_platform_row = min(platform["row"] for platform in lane["platforms"])
            settled_bottom = int(lane["settled_r0"] + lane["mask"].shape[0] - 1)
            min_lift = settled_bottom - top_platform_row + 1
        else:
            min_lift = 1
        if max_lift < max(1, min_lift):
            return None
        lane["lift"] = random.randint(max(1, min_lift), max_lift)
        lane["input_r0"] = int(lane["settled_r0"] - lane["lift"])

    legends = []
    for lane in lane_specs:
        for platform_color in lane["legend_colors"]:
            legends.append((int(platform_color), int(lane["object_color"])))
    if len(legends) < 3:
        return None
    return {"height": height, "width": width, "lanes": lane_specs, "legends": legends}


def _build_pair(spec: dict) -> tuple[np.ndarray, np.ndarray]:
    height, width = spec["height"], spec["width"]
    inp = np.zeros((height, width), dtype=int)
    out = np.zeros((height, width), dtype=int)

    all_platforms = []
    for lane in spec["lanes"]:
        for platform in lane["platforms"]:
            r = int(platform["row"])
            c0 = int(platform["c0"])
            c1 = int(platform["c1"])
            color = int(platform["color"])
            inp[r, c0 : c1 + 1] = color
            out[r, c0 : c1 + 1] = color
            all_platforms.append(platform)

    legend_positions = _legend_positions(height, len(spec["legends"]))
    for (platform_color, object_color), (r, c) in zip(spec["legends"], legend_positions):
        inp[r, c] = platform_color
        inp[r, c + 1] = object_color
        out[r, c] = platform_color
        out[r, c + 1] = object_color

    for lane in spec["lanes"]:
        color = int(lane["object_color"])
        mask = lane["mask"]
        _stamp(inp, mask, int(lane["input_r0"]), int(lane["obj_c0"]), color)
        _stamp(out, mask, int(lane["settled_r0"]), int(lane["obj_c0"]), color)

    return inp, out


def _validate_pair(inp: np.ndarray, out: np.ndarray, spec: dict) -> bool:
    try:
        bg, legends, platforms, objects = _parse_scene(inp)
    except Exception:
        return False

    if bg != BG:
        return False
    if len(platforms) != PLATFORM_COUNT:
        return False
    if len(objects) != len(spec["lanes"]):
        return False
    if not {"top_multi", "lower", "bottom_platform"}.issubset({lane["mode"] for lane in spec["lanes"]}):
        return False

    for lane in spec["lanes"]:
        if lane["lift"] < 1:
            return False
        if lane["mode"] == "top_multi" and len(legends.get(lane["object_color"], set())) != 2:
            return False
        if lane["mode"] == "lower":
            catch = legends.get(lane["object_color"], set())
            if catch != {lane["platforms"][-1]["color"]}:
                return False
        if lane["mode"] in {"bottom_platform", "empty_fall"} and legends.get(lane["object_color"], set()):
            return False

    try:
        solved = main(inp)
        nearest_any = _render_with_strategy(inp, "nearest_any")
        lowest_match = _render_with_strategy(inp, "lowest_match")
    except Exception:
        return False

    if not np.array_equal(solved, out):
        return False
    if np.array_equal(inp, out):
        return False
    if np.array_equal(nearest_any, out):
        return False
    if np.array_equal(lowest_match, out):
        return False
    return True


def _generate_attempt() -> np.ndarray | None:
    spec = _scene_spec()
    if spec is None:
        return None
    inp, out = _build_pair(spec)
    if not _validate_pair(inp, out, spec):
        return None
    return inp


def generate_input() -> np.ndarray:
    for _ in range(GEN_ATTEMPTS):
        grid = _generate_attempt()
        if grid is not None:
            return grid
    raise RuntimeError("failed to generate a valid selective-platform input")
