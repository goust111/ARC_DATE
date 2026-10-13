from __future__ import annotations

from collections import Counter, deque
from pathlib import Path
import random
import sys

import numpy as np

ARC_UTILS_DIR = Path("/home/lixinyu/snap/new/solvers")
if str(ARC_UTILS_DIR) not in sys.path:
    sys.path.append(str(ARC_UTILS_DIR))

from arc_utils import crop_bbox, find_components, get_bg, in_bounds, neighbors4


BG_COLOR = 0
FRAME_COLOR = 5
DOT_COLORS = (1, 2, 3, 4, 6, 7, 8, 9)
GEN_GRID_MIN = 20
GEN_GRID_MAX = 28
GEN_FRAME_COUNT_CHOICES = (2, 3)
GEN_FRAME_SIDE_MIN = 5
GEN_FRAME_SIDE_MAX = 8
GEN_DOT_COUNT_MIN = 3
GEN_DOT_COUNT_MAX = 5
GEN_ATTEMPTS = 500


def _rects_overlap(a: tuple[int, int, int, int], b: tuple[int, int, int, int]) -> bool:
    return not (a[2] < b[0] or b[2] < a[0] or a[3] < b[1] or b[3] < a[1])


def _full_perimeter(h: int, w: int) -> set[tuple[int, int]]:
    cells = set()
    for c in range(w):
        cells.add((0, c))
        cells.add((h - 1, c))
    for r in range(h):
        cells.add((r, 0))
        cells.add((r, w - 1))
    return cells


def _frame_from_component(component: dict) -> dict | None:
    r0, c0, r1, c1 = map(int, component["bbox"])
    h = r1 - r0 + 1
    w = c1 - c0 + 1
    if h < 3 or w < 3:
        return None

    rel_coords = {(int(r) - r0, int(c) - c0) for r, c in component["coords"]}
    perimeter = _full_perimeter(h, w)
    if not rel_coords <= perimeter:
        return None
    missing = list(perimeter - rel_coords)
    if len(missing) != 1:
        return None
    if len(rel_coords) != len(perimeter) - 1:
        return None

    gap_r, gap_c = missing[0]
    on_h = gap_r in (0, h - 1)
    on_v = gap_c in (0, w - 1)
    if on_h == on_v:
        return None

    if gap_r == 0:
        side = "top"
        out_dir = (-1, 0)
    elif gap_r == h - 1:
        side = "bottom"
        out_dir = (1, 0)
    elif gap_c == 0:
        side = "left"
        out_dir = (0, -1)
    else:
        side = "right"
        out_dir = (0, 1)

    return {
        "bbox": (r0, c0, r1, c1),
        "gap": (r0 + gap_r, c0 + gap_c),
        "gap_rel": (gap_r, gap_c),
        "side": side,
        "out_dir": out_dir,
        "color": int(component["color"]),
        "coords": component["coords"],
    }


def _bfs_inside_distances(shape: tuple[int, int], gap_rel: tuple[int, int]) -> np.ndarray:
    h, w = map(int, shape)
    blocked = np.zeros((h, w), dtype=bool)
    blocked[0, :] = True
    blocked[-1, :] = True
    blocked[:, 0] = True
    blocked[:, -1] = True
    blocked[gap_rel] = False

    dist = np.full((h, w), -1, dtype=int)
    queue = deque([gap_rel])
    dist[gap_rel] = 0
    while queue:
        r, c = queue.popleft()
        for nr, nc in neighbors4(r, c):
            if not in_bounds((h, w), nr, nc):
                continue
            if blocked[nr, nc] or dist[nr, nc] != -1:
                continue
            dist[nr, nc] = dist[r, c] + 1
            queue.append((nr, nc))
    return dist


def _parse_frames_and_dots(grid: np.ndarray) -> list[dict]:
    g = np.asarray(grid, dtype=int)
    bg = get_bg(g)
    components = find_components(g, bg=bg, connectivity=4, monochromatic=True)

    frames = []
    frame_mask = np.zeros_like(g, dtype=bool)
    for component in components:
        frame = _frame_from_component(component)
        if frame is None:
            continue
        frames.append(frame)
        frame_mask[component["coords"][:, 0], component["coords"][:, 1]] = True

    assert frames, "expected at least one hollow rectangle frame"

    non_frame = np.argwhere((g != bg) & ~frame_mask)
    dots_by_frame: list[list[tuple[int, int, int]]] = [[] for _ in frames]
    for rr_raw, cc_raw in non_frame:
        rr = int(rr_raw)
        cc = int(cc_raw)
        owners = []
        for idx, frame in enumerate(frames):
            r0, c0, r1, c1 = frame["bbox"]
            if r0 < rr < r1 and c0 < cc < c1:
                owners.append(idx)
        assert len(owners) == 1, "every non-frame cell must belong to exactly one frame interior"
        dots_by_frame[owners[0]].append((rr, cc, int(g[rr, cc])))

    parsed = []
    for frame, dots in zip(frames, dots_by_frame):
        r0, c0, r1, c1 = frame["bbox"]
        assert dots, "each frame must contain at least one interior dot"
        colors = [color for _r, _c, color in dots]
        assert len(set(colors)) == len(colors), "interior dot colors must be distinct"

        dist = _bfs_inside_distances((r1 - r0 + 1, c1 - c0 + 1), frame["gap_rel"])
        ordered = []
        for rr, cc, color in dots:
            rel = (rr - r0, cc - c0)
            dot_dist = int(dist[rel])
            assert dot_dist > 0, "every dot must be reachable from the door inside the box"
            ordered.append((dot_dist, rr, cc, color))
        distances = [item[0] for item in ordered]
        assert len(set(distances)) == len(distances), "door distances must be unique within a frame"

        parsed.append(
            {
                **frame,
                "dots": dots,
                "ordered": sorted(ordered),
            }
        )

    covered = frame_mask.copy()
    for frame in parsed:
        for rr, cc, _color in frame["dots"]:
            covered[rr, cc] = True
    assert np.array_equal(covered, g != bg), "scene contains unsupported foreground pixels"
    return parsed


def main(grid: np.ndarray) -> np.ndarray:
    """
    Empty each hollow rectangle and queue its interior dots outside the gap ordered by inside-the-box door distance.

    Input structure:
    - The background contains one or more axis-aligned hollow rectangular frames, each drawn by a one-pixel border with exactly one missing border cell acting as a door.
    - Each frame encloses several single-pixel dots of distinct colors, and dots from different frames stay separated.
    - Within any one frame, the dots have pairwise distinct 4-neighbor shortest-path distances to that frame's door when walking only inside the box and through the gap.

    Rule:
    - For every frame, compute each interior dot's shortest 4-neighbor path length to the door cell without crossing wall cells.
    - Clear the frame interior back to background while keeping the border and the gap unchanged.
    - Starting from the cell immediately outside the door, place the dot colors outward along the normal direction of that side, ordered from nearest-to-door to farthest.

    Key insight:
    - The missing border cell is both the exit and the direction marker, while the original dot positions matter only through their unique in-box geodesic distance order to that door.
    """
    g = np.asarray(grid, dtype=int)
    assert g.ndim == 2, "expected a 2D grid"

    bg = get_bg(g)
    frames = _parse_frames_and_dots(g)

    out = g.copy()
    for frame in frames:
        r0, c0, r1, c1 = frame["bbox"]
        gap_r, gap_c = frame["gap"]
        dr, dc = frame["out_dir"]

        out[r0 + 1 : r1, c0 + 1 : c1] = bg

        rr = gap_r + dr
        cc = gap_c + dc
        for _dist, _dot_r, _dot_c, color in frame["ordered"]:
            assert in_bounds(out.shape, rr, cc), "queue stripe leaves the grid"
            assert int(out[rr, cc]) == bg, "queue stripe must not collide with another object"
            out[rr, cc] = int(color)
            rr += dr
            cc += dc

    return out


def _distance_to_gap(pos: tuple[int, int], gap_rel: tuple[int, int]) -> int:
    return abs(int(pos[0]) - int(gap_rel[0])) + abs(int(pos[1]) - int(gap_rel[1]))


def _side_distance(pos: tuple[int, int], h: int, w: int, side: str) -> int:
    r, c = map(int, pos)
    if side == "top":
        return r
    if side == "bottom":
        return h - 1 - r
    if side == "left":
        return c
    if side == "right":
        return w - 1 - c
    raise AssertionError("unknown side")


def _naive_side_order(positions: list[tuple[int, int]], h: int, w: int, side: str) -> list[tuple[int, int]]:
    if side in ("top", "bottom"):
        return sorted(positions, key=lambda p: (_side_distance(p, h, w, side), p[1], p[0]))
    return sorted(positions, key=lambda p: (_side_distance(p, h, w, side), p[0], p[1]))


def _choose_dot_positions(
    h: int,
    w: int,
    side: str,
    gap_rel: tuple[int, int],
    count: int,
    require_trap: bool,
) -> list[tuple[int, int]] | None:
    pool = [(r, c) for r in range(1, h - 1) for c in range(1, w - 1)]
    if len(pool) < count:
        return None

    for _ in range(400):
        positions = random.sample(pool, count)
        dists = [_distance_to_gap(pos, gap_rel) for pos in positions]
        if len(set(dists)) != count:
            continue
        if not any(pos[0] != gap_rel[0] and pos[1] != gap_rel[1] for pos in positions):
            continue

        if require_trap:
            bands = Counter(pos[0] for pos in positions) if side in ("top", "bottom") else Counter(pos[1] for pos in positions)
            if max(bands.values()) < 2:
                continue
            true_order = sorted(positions, key=lambda pos: (_distance_to_gap(pos, gap_rel), pos[0], pos[1]))
            if _naive_side_order(positions, h, w, side) == true_order:
                continue

        return positions
    return None


def _reserve_bbox_for_spec(spec: dict, r0: int, c0: int) -> tuple[int, int, int, int]:
    h = int(spec["h"])
    w = int(spec["w"])
    count = int(spec["count"])
    side = spec["side"]
    if side == "top":
        return (r0 - count - 1, c0 - 1, r0 + h, c0 + w)
    if side == "bottom":
        return (r0 - 1, c0 - 1, r0 + h + count, c0 + w)
    if side == "left":
        return (r0 - 1, c0 - count - 1, r0 + h, c0 + w)
    if side == "right":
        return (r0 - 1, c0 - 1, r0 + h, c0 + w + count)
    raise AssertionError("unknown side")


def _place_specs(specs: list[dict], h: int, w: int) -> list[dict] | None:
    placed: list[dict] = []

    def reserve_area(spec: dict) -> int:
        reserve = _reserve_bbox_for_spec(spec, 0, 0)
        return (reserve[2] - reserve[0] + 1) * (reserve[3] - reserve[1] + 1)

    for spec in sorted(specs, key=reserve_area, reverse=True):
        fh = int(spec["h"])
        fw = int(spec["w"])
        count = int(spec["count"])
        side = spec["side"]

        if side == "top":
            row_lo, row_hi = count + 1, h - fh - 1
            col_lo, col_hi = 1, w - fw - 1
        elif side == "bottom":
            row_lo, row_hi = 1, h - fh - count - 1
            col_lo, col_hi = 1, w - fw - 1
        elif side == "left":
            row_lo, row_hi = 1, h - fh - 1
            col_lo, col_hi = count + 1, w - fw - 1
        else:
            row_lo, row_hi = 1, h - fh - 1
            col_lo, col_hi = 1, w - fw - count - 1

        if row_lo > row_hi or col_lo > col_hi:
            return None

        found = None
        for _ in range(250):
            r0 = random.randint(row_lo, row_hi)
            c0 = random.randint(col_lo, col_hi)
            reserve = _reserve_bbox_for_spec(spec, r0, c0)
            if any(_rects_overlap(reserve, other["reserve"]) for other in placed):
                continue

            gap_rel = spec["gap_rel"]
            found = {
                **spec,
                "bbox": (r0, c0, r0 + fh - 1, c0 + fw - 1),
                "gap": (r0 + gap_rel[0], c0 + gap_rel[1]),
                "reserve": reserve,
            }
            break
        if found is None:
            return None
        placed.append(found)

    return placed


def _render_input(specs: list[dict], shape: tuple[int, int]) -> np.ndarray:
    grid = np.full(shape, BG_COLOR, dtype=int)
    for spec in specs:
        r0, c0, r1, c1 = spec["bbox"]
        gap_r, gap_c = spec["gap"]
        grid[r0, c0 : c1 + 1] = FRAME_COLOR
        grid[r1, c0 : c1 + 1] = FRAME_COLOR
        grid[r0 : r1 + 1, c0] = FRAME_COLOR
        grid[r0 : r1 + 1, c1] = FRAME_COLOR
        grid[gap_r, gap_c] = BG_COLOR

        for (rr, cc), color in zip(spec["dot_positions"], spec["colors"]):
            grid[r0 + rr, c0 + cc] = int(color)
    return grid


def _expected_output(specs: list[dict], inp: np.ndarray) -> np.ndarray:
    out = inp.copy()
    for spec in specs:
        r0, c0, r1, c1 = spec["bbox"]
        gap_r, gap_c = spec["gap"]
        dr, dc = spec["out_dir"]
        out[r0 + 1 : r1, c0 + 1 : c1] = BG_COLOR

        ordered = sorted(
            zip(spec["distances"], spec["colors"]),
            key=lambda item: item[0],
        )
        rr = gap_r + dr
        cc = gap_c + dc
        for _dist, color in ordered:
            if not in_bounds(out.shape, rr, cc):
                raise AssertionError("generated stripe left the grid")
            if int(out[rr, cc]) != BG_COLOR:
                raise AssertionError("generated stripe collided with another object")
            out[rr, cc] = int(color)
            rr += dr
            cc += dc
    return out


def _sample_spec(side: str, require_trap: bool) -> dict | None:
    h = random.randint(GEN_FRAME_SIDE_MIN, GEN_FRAME_SIDE_MAX)
    w = random.randint(GEN_FRAME_SIDE_MIN, GEN_FRAME_SIDE_MAX)
    count_lo = max(GEN_DOT_COUNT_MIN, 4 if require_trap else GEN_DOT_COUNT_MIN)
    count_hi = min(GEN_DOT_COUNT_MAX, (h - 2) * (w - 2))
    if count_lo > count_hi:
        return None
    count = random.randint(count_lo, count_hi)

    if side in ("top", "bottom"):
        gap_rel = (0, random.randint(1, w - 2)) if side == "top" else (h - 1, random.randint(1, w - 2))
    else:
        gap_rel = (random.randint(1, h - 2), 0) if side == "left" else (random.randint(1, h - 2), w - 1)

    positions = _choose_dot_positions(h, w, side, gap_rel, count, require_trap)
    if positions is None:
        return None

    colors = random.sample(DOT_COLORS, count)
    distances = [_distance_to_gap(pos, gap_rel) for pos in positions]
    out_dir = {
        "top": (-1, 0),
        "bottom": (1, 0),
        "left": (0, -1),
        "right": (0, 1),
    }[side]
    return {
        "h": h,
        "w": w,
        "count": count,
        "side": side,
        "gap_rel": gap_rel,
        "dot_positions": positions,
        "colors": colors,
        "distances": distances,
        "out_dir": out_dir,
        "trap": require_trap,
    }


def _scene_semantics_ok(specs: list[dict], inp: np.ndarray, out: np.ndarray) -> bool:
    if not np.array_equal(main(inp), out):
        return False

    if not any(spec["trap"] for spec in specs):
        return False

    for spec in specs:
        if len(set(spec["distances"])) != len(spec["distances"]):
            return False
        if not any(pos[0] != spec["gap_rel"][0] and pos[1] != spec["gap_rel"][1] for pos in spec["dot_positions"]):
            return False

        true_order = sorted(spec["dot_positions"], key=lambda pos: (_distance_to_gap(pos, spec["gap_rel"]), pos[0], pos[1]))
        if spec["trap"] and _naive_side_order(spec["dot_positions"], spec["h"], spec["w"], spec["side"]) == true_order:
            return False

    parsed = _parse_frames_and_dots(inp)
    if len(parsed) != len(specs):
        return False
    return True


def _gen_attempt() -> np.ndarray | None:
    h = random.randint(GEN_GRID_MIN, GEN_GRID_MAX)
    w = random.randint(GEN_GRID_MIN, GEN_GRID_MAX)
    frame_count = random.choice(GEN_FRAME_COUNT_CHOICES)
    sides = random.sample(("top", "bottom", "left", "right"), frame_count)
    trap_idx = random.randrange(frame_count)

    specs: list[dict] = []
    for idx, side in enumerate(sides):
        spec = _sample_spec(side, require_trap=idx == trap_idx)
        if spec is None:
            return None
        specs.append(spec)

    placed = _place_specs(specs, h, w)
    if placed is None:
        return None

    inp = _render_input(placed, (h, w))
    out = _expected_output(placed, inp)
    if not _scene_semantics_ok(placed, inp, out):
        return None
    return inp


def generate_input() -> np.ndarray:
    for _ in range(GEN_ATTEMPTS):
        grid = _gen_attempt()
        if grid is not None:
            return grid
    raise RuntimeError("failed to generate a valid queue-by-door-distance input")
