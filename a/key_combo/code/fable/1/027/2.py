from __future__ import annotations

from collections import Counter, deque
import random

import numpy as np


BG = 0
SIDES = ("top", "bottom", "left", "right")
MAX_GENERATION_ATTEMPTS = 400


def _get_bg(grid: np.ndarray) -> int:
    return int(Counter(map(int, grid.ravel())).most_common(1)[0][0])


def _neighbors4(r: int, c: int):
    yield r - 1, c
    yield r + 1, c
    yield r, c - 1
    yield r, c + 1


def _bbox_from_coords(coords: list[tuple[int, int]]) -> tuple[int, int, int, int]:
    rows = [r for r, _ in coords]
    cols = [c for _, c in coords]
    return min(rows), min(cols), max(rows), max(cols)


def _mono_components(grid: np.ndarray, bg: int) -> list[dict]:
    h, w = grid.shape
    seen = np.zeros((h, w), dtype=bool)
    comps: list[dict] = []
    for r in range(h):
        for c in range(w):
            color = int(grid[r, c])
            if color == bg or seen[r, c]:
                continue
            seen[r, c] = True
            q = deque([(r, c)])
            cells = [(r, c)]
            while q:
                rr, cc = q.popleft()
                for nr, nc in _neighbors4(rr, cc):
                    if 0 <= nr < h and 0 <= nc < w and not seen[nr, nc] and int(grid[nr, nc]) == color:
                        seen[nr, nc] = True
                        q.append((nr, nc))
                        cells.append((nr, nc))
            comps.append(
                {
                    "color": color,
                    "coords": np.array(cells, dtype=int),
                    "bbox": _bbox_from_coords(cells),
                    "area": len(cells),
                }
            )
    return comps


def _is_square_ring(grid: np.ndarray, comp: dict) -> bool:
    r0, c0, r1, c1 = map(int, comp["bbox"])
    side_h = r1 - r0 + 1
    side_w = c1 - c0 + 1
    if side_h != side_w or side_h < 3:
        return False
    if int(comp["area"]) != 4 * side_h - 4:
        return False
    patch = grid[r0 : r1 + 1, c0 : c1 + 1] == int(comp["color"])
    border = np.zeros((side_h, side_w), dtype=bool)
    border[0, :] = True
    border[-1, :] = True
    border[:, 0] = True
    border[:, -1] = True
    return bool(np.array_equal(patch, border))


def _paint_square_frame(grid: np.ndarray, bbox: tuple[int, int, int, int], color: int) -> None:
    r0, c0, r1, c1 = map(int, bbox)
    grid[r0, c0 : c1 + 1] = int(color)
    grid[r1, c0 : c1 + 1] = int(color)
    grid[r0 : r1 + 1, c0] = int(color)
    grid[r0 : r1 + 1, c1] = int(color)


def _core_mask_from_bbox(shape: tuple[int, int], bbox: tuple[int, int, int, int]) -> np.ndarray:
    r0, c0, r1, c1 = map(int, bbox)
    mask = np.zeros(shape, dtype=bool)
    mask[r0 + 1 : r1, c0 + 1 : c1] = True
    return mask


def _parse_puzzle(grid: np.ndarray) -> dict:
    g = np.asarray(grid, dtype=int)
    assert g.ndim == 2
    bg = _get_bg(g)
    comps = _mono_components(g, bg)

    frames = [comp for comp in comps if _is_square_ring(g, comp)]
    assert frames, "expected at least one square frame"
    frames.sort(key=lambda comp: int(comp["bbox"][2] - comp["bbox"][0]))

    center_key = None
    for comp in frames:
        r0, c0, r1, c1 = map(int, comp["bbox"])
        key = (r0 + r1, c0 + c1)
        if center_key is None:
            center_key = key
        assert key == center_key, "frames must be concentric"

    for i in range(len(frames) - 1):
        inner = frames[i]["bbox"]
        outer = frames[i + 1]["bbox"]
        assert int(outer[0]) <= int(inner[0]) - 2
        assert int(outer[1]) <= int(inner[1]) - 2
        assert int(outer[2]) >= int(inner[2]) + 2
        assert int(outer[3]) >= int(inner[3]) + 2

    frame_colors = {int(comp["color"]) for comp in frames}
    frame_by_color = {int(comp["color"]): comp for comp in frames}

    frame_mask = np.zeros(g.shape, dtype=bool)
    for comp in frames:
        for r, c in comp["coords"]:
            frame_mask[int(r), int(c)] = True

    inner_bbox = tuple(map(int, frames[0]["bbox"]))
    core_mask = _core_mask_from_bbox(g.shape, inner_bbox)
    assert core_mask.any(), "innermost frame must have a non-empty interior"
    core_values = g[core_mask]
    water_color = int(core_values[0])
    assert water_color != bg and water_color not in frame_colors
    assert np.all(core_values == water_color), "innermost interior must be a solid water color"

    outer_bbox = tuple(map(int, frames[-1]["bbox"]))
    marker_mask = (g != bg) & ~frame_mask & ~core_mask
    markers = []
    for r, c in np.argwhere(marker_mask):
        rr = int(r)
        cc = int(c)
        color = int(g[rr, cc])
        assert color in frame_by_color, "each marker must match a frame color"
        same_color_neighbors = 0
        for nr, nc in _neighbors4(rr, cc):
            if 0 <= nr < g.shape[0] and 0 <= nc < g.shape[1] and int(g[nr, nc]) == color:
                same_color_neighbors += 1
        assert same_color_neighbors == 0, "markers must be singleton pixels"
        r0, c0, r1, c1 = outer_bbox
        outside_vert = rr < r0 or rr > r1
        outside_horz = cc < c0 or cc > c1
        assert outside_vert ^ outside_horz, "marker must align by row or by column"
        markers.append({"pos": (rr, cc), "color": color})

    return {
        "bg": bg,
        "frames": [
            {"color": int(comp["color"]), "bbox": tuple(map(int, comp["bbox"]))}
            for comp in frames
        ],
        "frame_by_color": frame_by_color,
        "frame_mask": frame_mask,
        "outer_bbox": outer_bbox,
        "core_mask": core_mask,
        "water_color": water_color,
        "markers": markers,
    }


def _marker_gap(
    marker_pos: tuple[int, int],
    frame_bbox: tuple[int, int, int, int],
    outer_bbox: tuple[int, int, int, int],
) -> tuple[int, int]:
    r, c = map(int, marker_pos)
    fr0, fc0, fr1, fc1 = map(int, frame_bbox)
    or0, oc0, or1, oc1 = map(int, outer_bbox)
    candidates: list[tuple[int, int]] = []
    if r < or0 and oc0 <= c <= oc1 and fc0 <= c <= fc1:
        candidates.append((fr0, c))
    if r > or1 and oc0 <= c <= oc1 and fc0 <= c <= fc1:
        candidates.append((fr1, c))
    if c < oc0 and or0 <= r <= or1 and fr0 <= r <= fr1:
        candidates.append((r, fc0))
    if c > oc1 and or0 <= r <= or1 and fr0 <= r <= fr1:
        candidates.append((r, fc1))
    assert len(candidates) == 1, "marker must point to exactly one frame-side position"
    return candidates[0]


def _flood_from_core(wall_mask: np.ndarray, core_mask: np.ndarray) -> np.ndarray:
    h, w = wall_mask.shape
    visited = np.zeros((h, w), dtype=bool)
    q = deque()
    for r, c in np.argwhere(core_mask):
        rr = int(r)
        cc = int(c)
        visited[rr, cc] = True
        q.append((rr, cc))

    while q:
        r, c = q.popleft()
        for nr, nc in _neighbors4(r, c):
            if 0 <= nr < h and 0 <= nc < w and not visited[nr, nc] and not wall_mask[nr, nc]:
                visited[nr, nc] = True
                q.append((nr, nc))
    return visited


def _solve_from_parsed(grid: np.ndarray, parsed: dict) -> np.ndarray:
    g = np.asarray(grid, dtype=int)
    out = g.copy()
    bg = int(parsed["bg"])
    water_color = int(parsed["water_color"])
    outer_bbox = tuple(map(int, parsed["outer_bbox"]))

    wall_mask = parsed["frame_mask"].copy()
    frame_lookup = {
        int(frame["color"]): tuple(map(int, frame["bbox"]))
        for frame in parsed["frames"]
    }

    for marker in parsed["markers"]:
        color = int(marker["color"])
        pos = tuple(map(int, marker["pos"]))
        gap = _marker_gap(pos, frame_lookup[color], outer_bbox)
        out[pos] = bg
        out[gap] = bg
        wall_mask[gap] = False

    reached = _flood_from_core(wall_mask, parsed["core_mask"])
    out[reached] = water_color
    return out


def main(grid: np.ndarray) -> np.ndarray:
    """
    Open color-matched gaps in concentric square frames, then let the central water flood through every reachable corridor.

    Input structure:
    - The grid has a background plus one concentric stack of single-pixel-thick square frames with distinct colors.
    - The innermost frame encloses a solid square block of one water color.
    - Outside the outermost frame sit singleton markers whose colors match frame layers and whose rows or columns point straight at the frame stack.

    Rule:
    - Each outside marker ignores other frame colors and punches exactly one gap in the frame layer of the same color at the aligned side position, then the marker disappears.
    - After all gaps are opened, treat every remaining frame pixel as a wall and flood from the central water block through 4-neighbor open cells.
    - Recolor every reached corridor cell, reached background cell, and opened gap to the water color; untouched frame walls stay unchanged.

    Key insight:
    - Marker targeting is determined by color, not by the first frame hit, so the water can advance only through the consecutive chain of frame layers that actually received at least one opening.
    """
    parsed = _parse_puzzle(np.asarray(grid, dtype=int))
    return _solve_from_parsed(np.asarray(grid, dtype=int), parsed)


def _sample_open_flags(layer_count: int) -> tuple[list[bool], str]:
    mode = random.choices(["spill", "blocked", "prefix"], weights=[3, 5, 2], k=1)[0]
    if mode == "spill":
        return [True] * layer_count, mode

    if mode == "blocked":
        break_idx = random.randrange(layer_count - 1)
        flags = [True] * break_idx + [False]
        tail = [random.random() < 0.65 for _ in range(layer_count - break_idx - 1)]
        if tail and not any(tail):
            tail[random.randrange(len(tail))] = True
        if tail and random.random() < 0.7:
            tail[-1] = True
        return flags + tail, mode

    prefix_len = random.randint(1, layer_count - 1)
    return [True] * prefix_len + [False] * (layer_count - prefix_len), mode


def _consecutive_open_prefix(flags: list[bool]) -> int:
    count = 0
    for flag in flags:
        if not flag:
            break
        count += 1
    return count


def _choose_gap_on_side(
    frame_bbox: tuple[int, int, int, int],
    side: str,
) -> tuple[int, int]:
    r0, c0, r1, c1 = map(int, frame_bbox)
    if side == "top":
        return r0, random.randint(c0 + 1, c1 - 1)
    if side == "bottom":
        return r1, random.randint(c0 + 1, c1 - 1)
    if side == "left":
        return random.randint(r0 + 1, r1 - 1), c0
    if side == "right":
        return random.randint(r0 + 1, r1 - 1), c1
    raise ValueError("unknown side")


def _marker_pos_for_gap(
    outer_side: int,
    gap: tuple[int, int],
    side: str,
    distance: int,
) -> tuple[int, int]:
    r, c = map(int, gap)
    if side == "top":
        return -distance, c
    if side == "bottom":
        return outer_side - 1 + distance, c
    if side == "left":
        return r, -distance
    if side == "right":
        return r, outer_side - 1 + distance
    raise ValueError("unknown side")


def _render_input_from_plan(plan: dict) -> np.ndarray:
    grid = np.full(tuple(plan["shape"]), int(plan["bg"]), dtype=int)
    for frame in plan["frames"]:
        _paint_square_frame(grid, tuple(frame["bbox"]), int(frame["color"]))
    core_mask = _core_mask_from_bbox(grid.shape, tuple(plan["frames"][0]["bbox"]))
    grid[core_mask] = int(plan["water_color"])
    for marker in plan["markers"]:
        r, c = map(int, marker["pos"])
        grid[r, c] = int(marker["color"])
    return grid


def _render_output_from_plan(plan: dict) -> np.ndarray:
    grid = _render_input_from_plan(plan)
    parsed = {
        "bg": int(plan["bg"]),
        "frames": plan["frames"],
        "outer_bbox": tuple(plan["frames"][-1]["bbox"]),
        "frame_mask": np.zeros(tuple(plan["shape"]), dtype=bool),
        "core_mask": _core_mask_from_bbox(tuple(plan["shape"]), tuple(plan["frames"][0]["bbox"])),
        "water_color": int(plan["water_color"]),
        "markers": [{"pos": tuple(m["pos"]), "color": int(m["color"])} for m in plan["markers"]],
    }
    for frame in plan["frames"]:
        r0, c0, r1, c1 = map(int, frame["bbox"])
        parsed["frame_mask"][r0, c0 : c1 + 1] = True
        parsed["frame_mask"][r1, c0 : c1 + 1] = True
        parsed["frame_mask"][r0 : r1 + 1, c0] = True
        parsed["frame_mask"][r0 : r1 + 1, c1] = True
    return _solve_from_parsed(grid, parsed)


def _build_plan() -> dict | None:
    layer_count = random.randint(2, 5)
    inner_options = [side for side in range(5, 14, 2) if side + 4 * (layer_count - 1) <= 21]
    if not inner_options:
        return None
    inner_side = random.choice(inner_options)
    outer_side = inner_side + 4 * (layer_count - 1)

    palette = random.sample(list(range(1, 10)), layer_count + 1)
    frame_colors = palette[:layer_count]
    water_color = palette[-1]

    local_frames = []
    for idx in range(layer_count):
        side = inner_side + 4 * idx
        inset = 2 * (layer_count - 1 - idx)
        local_frames.append({"color": frame_colors[idx], "bbox": (inset, inset, inset + side - 1, inset + side - 1)})

    open_flags, mode = _sample_open_flags(layer_count)
    reachable_layers = _consecutive_open_prefix(open_flags)

    markers: list[dict] = []
    used_positions: set[tuple[int, int]] = set()
    for idx, frame in enumerate(local_frames):
        if not open_flags[idx]:
            continue
        marker_count = 1 + int(random.random() < 0.4)
        sides = random.sample(SIDES, marker_count)
        for side in sides:
            for _ in range(20):
                gap = _choose_gap_on_side(tuple(frame["bbox"]), side)
                distance = random.randint(2, 4)
                pos = _marker_pos_for_gap(outer_side, gap, side, distance)
                if pos in used_positions:
                    continue
                used_positions.add(pos)
                markers.append(
                    {
                        "pos": pos,
                        "color": int(frame["color"]),
                        "gap": gap,
                        "layer": idx,
                        "side": side,
                    }
                )
                break
            else:
                return None

    if not markers:
        return None

    all_points = []
    for frame in local_frames:
        r0, c0, r1, c1 = map(int, frame["bbox"])
        all_points.extend([(r0, c0), (r1, c1)])
    all_points.extend(tuple(marker["pos"]) for marker in markers)
    min_r = min(r for r, _ in all_points)
    max_r = max(r for r, _ in all_points)
    min_c = min(c for _, c in all_points)
    max_c = max(c for _, c in all_points)

    base_h = max_r - min_r + 1
    base_w = max_c - min_c + 1
    if base_h > 30 or base_w > 30:
        return None

    extra_h = random.randint(0, min(2, 30 - base_h))
    extra_w = random.randint(0, min(2, 30 - base_w))
    top_extra = random.randint(0, extra_h)
    left_extra = random.randint(0, extra_w)
    row_shift = -min_r + top_extra
    col_shift = -min_c + left_extra
    shape = (base_h + extra_h, base_w + extra_w)

    frames = []
    for frame in local_frames:
        r0, c0, r1, c1 = map(int, frame["bbox"])
        frames.append(
            {
                "color": int(frame["color"]),
                "bbox": (r0 + row_shift, c0 + col_shift, r1 + row_shift, c1 + col_shift),
            }
        )
    shifted_markers = []
    for marker in markers:
        r, c = map(int, marker["pos"])
        gr, gc = map(int, marker["gap"])
        shifted_markers.append(
            {
                "pos": (r + row_shift, c + col_shift),
                "color": int(marker["color"]),
                "gap": (gr + row_shift, gc + col_shift),
                "layer": int(marker["layer"]),
                "side": marker["side"],
            }
        )

    return {
        "shape": shape,
        "bg": BG,
        "water_color": int(water_color),
        "frames": frames,
        "markers": shifted_markers,
        "open_flags": open_flags,
        "mode": mode,
        "reachable_layers": reachable_layers,
    }


def _plan_structure_matches(plan: dict, parsed: dict) -> bool:
    plan_frames = [(int(frame["color"]), tuple(frame["bbox"])) for frame in plan["frames"]]
    parsed_frames = [(int(frame["color"]), tuple(frame["bbox"])) for frame in parsed["frames"]]
    if plan_frames != parsed_frames:
        return False
    plan_markers = sorted((tuple(marker["pos"]), int(marker["color"])) for marker in plan["markers"])
    parsed_markers = sorted((tuple(marker["pos"]), int(marker["color"])) for marker in parsed["markers"])
    if plan_markers != parsed_markers:
        return False
    return int(parsed["water_color"]) == int(plan["water_color"])


def _gen_attempt() -> tuple[np.ndarray, dict] | None:
    plan = _build_plan()
    if plan is None:
        return None
    grid = _render_input_from_plan(plan)
    expected = _render_output_from_plan(plan)

    try:
        parsed = _parse_puzzle(grid)
    except Exception:
        return None
    if not _plan_structure_matches(plan, parsed):
        return None

    actual = main(grid)
    if not np.array_equal(actual, expected):
        return None
    if np.array_equal(grid, expected):
        return None

    outer_r0, outer_c0, outer_r1, outer_c1 = map(int, plan["frames"][-1]["bbox"])
    outside_mask = np.ones(grid.shape, dtype=bool)
    outside_mask[outer_r0 : outer_r1 + 1, outer_c0 : outer_c1 + 1] = False
    if plan["reachable_layers"] == len(plan["frames"]):
        if not np.any(actual[outside_mask] == int(plan["water_color"])):
            return None
    else:
        if np.any(actual[outside_mask] == int(plan["water_color"])) and plan["reachable_layers"] < len(plan["frames"]):
            return None

    if plan["mode"] == "blocked":
        first_closed = plan["reachable_layers"]
        later_open = any(plan["open_flags"][first_closed + 1 :]) if first_closed < len(plan["frames"]) - 1 else False
        if not later_open:
            return None

    return grid, plan


def generate_input() -> np.ndarray:
    for _ in range(MAX_GENERATION_ATTEMPTS):
        attempt = _gen_attempt()
        if attempt is not None:
            return attempt[0]
    raise RuntimeError("failed to generate a valid opening-water puzzle")
