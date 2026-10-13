from __future__ import annotations

from collections import deque
from importlib.machinery import SourceFileLoader
from pathlib import Path
import random
import sys

import numpy as np

_UTILS_DIR = Path("/home/lixinyu/snap/new/solvers")
if str(_UTILS_DIR) not in sys.path:
    sys.path.append(str(_UTILS_DIR))

from arc_utils import get_bg, label_mask


BG = 0
BLUE = 1
RED = 2
GREEN = 3
YELLOW = 4
GREY = 5
BODY_COLORS = (6, 7, 8, 9)
MARKER_COLORS = (BLUE, RED, GREEN, YELLOW)
GEN_ATTEMPTS = 500


def _neighbors4(r: int, c: int) -> list[tuple[int, int]]:
    return [(r - 1, c), (r + 1, c), (r, c - 1), (r, c + 1)]


def _neighbors8(r: int, c: int) -> list[tuple[int, int]]:
    out = []
    for dr in (-1, 0, 1):
        for dc in (-1, 0, 1):
            if dr == 0 and dc == 0:
                continue
            out.append((r + dr, c + dc))
    return out


def _largest_enclosed_component_bbox(mask: np.ndarray, connectivity: int = 4) -> tuple[int, int, int, int]:
    lab, n = label_mask(mask.astype(bool), connectivity=connectivity)
    H, W = mask.shape
    best = None
    best_area = -1
    for cid in range(1, n + 1):
        coords = np.argwhere(lab == cid)
        if coords.size == 0:
            continue
        rs = coords[:, 0]
        cs = coords[:, 1]
        if (rs == 0).any() or (rs == H - 1).any() or (cs == 0).any() or (cs == W - 1).any():
            continue
        area = int(coords.shape[0])
        if area > best_area:
            r0, c0 = coords.min(axis=0)
            r1, c1 = coords.max(axis=0)
            best = (int(r0), int(c0), int(r1), int(c1))
            best_area = area
    assert best is not None, "expected one enclosed black frame interior"
    return best


def _parse_frame(grid: np.ndarray) -> dict:
    inner = _largest_enclosed_component_bbox(grid == BG, connectivity=4)
    ir0, ic0, ir1, ic1 = inner
    fr0, fc0, fr1, fc1 = ir0 - 1, ic0 - 1, ir1 + 1, ic1 + 1
    assert fr0 >= 0 and fc0 >= 0 and fr1 < grid.shape[0] and fc1 < grid.shape[1]

    top_cols: list[int] = []
    top_colors: dict[int, int] = {}
    for c in range(ic0, ic1 + 1):
        val = int(grid[fr0, c])
        if val in MARKER_COLORS:
            top_cols.append(int(c))
            top_colors[int(c)] = val
        else:
            assert val == GREY

    row_by_color: dict[int, int] = {}
    color_by_row: dict[int, int] = {}
    for r in range(ir0, ir1 + 1):
        val = int(grid[r, fc0])
        if val in MARKER_COLORS:
            assert val not in row_by_color
            row_by_color[val] = int(r)
            color_by_row[int(r)] = val
        else:
            assert val == GREY

    assert len(top_cols) >= 2
    assert len(row_by_color) >= 2
    assert np.all(grid[fr0, fc0 + 1 : fc1 + 1] != BG)
    assert np.all(grid[fr1, fc0:fc1 + 1] == GREY)
    assert np.all(grid[fr0 + 1 : fr1 + 1, fc1] == GREY)

    return {
        "inner": inner,
        "frame": (fr0, fc0, fr1, fc1),
        "top_cols": top_cols,
        "top_colors": top_colors,
        "row_by_color": row_by_color,
        "color_by_row": color_by_row,
    }


def _extract_sprites(grid: np.ndarray, frame: tuple[int, int, int, int]) -> list[dict]:
    fr0, fc0, fr1, fc1 = frame
    mask = (grid != BG) & (grid != GREY)
    lab, n = label_mask(mask, connectivity=4)
    sprites: list[dict] = []
    for cid in range(1, n + 1):
        coords = np.argwhere(lab == cid)
        if coords.size == 0:
            continue
        area = int(coords.shape[0])
        coords_list = [(int(r), int(c)) for r, c in coords]
        touches_frame = any(fr0 <= r <= fr1 and fc0 <= c <= fc1 for r, c in coords_list)
        if touches_frame:
            continue
        cmax = max(c for _r, c in coords_list)
        if cmax >= fc0:
            continue
        assert 3 <= area <= 5

        vals, counts = np.unique(grid[coords[:, 0], coords[:, 1]], return_counts=True)
        colors = [int(v) for v in vals]
        counts = [int(v) for v in counts]
        assert len(colors) == 2
        assert 1 in counts

        marker_color = colors[counts.index(1)]
        body_color = colors[1 - counts.index(1)]
        marker_coord = None
        body_offsets: list[tuple[int, int]] = []
        for r, c in coords_list:
            val = int(grid[r, c])
            if val == marker_color:
                assert marker_coord is None
                marker_coord = (int(r), int(c))
            else:
                assert val == body_color
        assert marker_coord is not None
        mr, mc = marker_coord
        for r, c in coords_list:
            if (r, c) != marker_coord:
                body_offsets.append((int(r - mr), int(c - mc)))

        sprites.append(
            {
                "coords": coords_list,
                "coord_set": set(coords_list),
                "marker": marker_coord,
                "marker_color": int(marker_color),
                "body_color": int(body_color),
                "body_offsets": sorted(body_offsets),
            }
        )

    assert 2 <= len(sprites) <= 4
    return sprites


def _extract_paths(grid: np.ndarray, frame: tuple[int, int, int, int]) -> list[set[tuple[int, int]]]:
    fr0, fc0, fr1, fc1 = frame
    mask = grid == GREY
    mask[fr0, fc0:fc1 + 1] = False
    mask[fr1, fc0:fc1 + 1] = False
    mask[fr0:fr1 + 1, fc0] = False
    mask[fr0:fr1 + 1, fc1] = False
    lab, n = label_mask(mask, connectivity=8)
    paths: list[set[tuple[int, int]]] = []
    for cid in range(1, n + 1):
        coords = np.argwhere(lab == cid)
        if coords.size == 0:
            continue
        paths.append({(int(r), int(c)) for r, c in coords})
    return paths


def _touches_path(sprite: dict, path: set[tuple[int, int]]) -> bool:
    for r, c in path:
        for nr, nc in _neighbors8(r, c):
            if (nr, nc) in sprite["coord_set"]:
                return True
    return False


def _sprite_connectivity_data(grid: np.ndarray, frame_info: dict, sprites: list[dict]) -> list[dict]:
    _fr0, fc0, _fr1, _fc1 = frame_info["frame"]
    color_by_row = frame_info["color_by_row"]
    paths = _extract_paths(grid, frame_info["frame"])
    data: list[dict] = []

    for sprite in sprites:
        touching = [path for path in paths if _touches_path(sprite, path)]
        assert len(touching) <= 1
        arrival_row = None
        if touching:
            rows = sorted({int(r) for r, c in touching[0] if c == fc0 - 1 and int(r) in color_by_row})
            if len(rows) == 1:
                arrival_row = rows[0]
                assert color_by_row[arrival_row] == sprite["marker_color"]
        item = dict(sprite)
        item["arrival_row"] = arrival_row
        data.append(item)
    return data


def _path_touches_coords(path: list[tuple[int, int]] | set[tuple[int, int]], coords: set[tuple[int, int]]) -> bool:
    for r, c in path:
        for nr, nc in _neighbors8(r, c):
            if (nr, nc) in coords:
                return True
    return False


def main(grid: np.ndarray) -> np.ndarray:
    """
    Stamp each connected sprite into the frame using path-arrival rows and top-edge marker columns.

    Input structure:
    - The background is black.
    - A grey rectangular frame sits on the right; its interior is black, its top edge has colored marker pixels, and its left edge has colored row markers.
    - On the left are 2-4 small 4-connected two-color sprites; each sprite has one singleton marker pixel and a monochrome body.
    - Grey single-pixel paths run from sprites toward the frame's left side; some sprites may have a broken path that never reaches the frame.

    Rule:
    - Find the frame by its enclosed black interior, then read every colored top-edge marker column and every colored left-edge marker row.
    - For each sprite, follow its adjacent grey path; if the path reaches the cell immediately left of the frame at row R, that sprite's stamping row is R.
    - Stamp only the sprite body, anchored by the sprite marker pixel, at every intersection of row R with every top-marker column; stamp sprites from smaller arrival row to larger arrival row so later ones overwrite earlier ones.
    - Erase all original sprites, all grey path pixels, and all colored frame-edge markers; keep the grey frame body and the stamped body pixels. Sprites whose path never reaches the frame are erased without stamping.

    Key insight:
    - The sprite marker color identifies which left-edge label matches, but the decisive signal is whether the grey path actually reaches that frame row; the row target comes from the path endpoint, not from the sprite's original placement.
    """
    g = np.asarray(grid, dtype=int)
    assert g.ndim == 2
    assert get_bg(g) == BG

    frame_info = _parse_frame(g)
    sprites = _extract_sprites(g, frame_info["frame"])
    sprite_data = _sprite_connectivity_data(g, frame_info, sprites)

    out = g.copy()
    fr0, fc0, fr1, fc1 = frame_info["frame"]
    for path in _extract_paths(g, frame_info["frame"]):
        for r, c in path:
            out[r, c] = BG
    for sprite in sprite_data:
        for r, c in sprite["coords"]:
            out[r, c] = BG
    for c in frame_info["top_cols"]:
        out[fr0, c] = BG
    for r in frame_info["color_by_row"]:
        out[r, fc0] = BG

    movers = [sprite for sprite in sprite_data if sprite["arrival_row"] is not None]
    movers.sort(key=lambda sprite: int(sprite["arrival_row"]))
    for sprite in movers:
        body_color = int(sprite["body_color"])
        row = int(sprite["arrival_row"])
        for col in frame_info["top_cols"]:
            for dr, dc in sprite["body_offsets"]:
                rr = int(row + dr)
                cc = int(col + dc)
                assert fr0 < rr < fr1 and fc0 < cc < fc1
                out[rr, cc] = body_color

    return out


def _random_shape_offsets() -> list[tuple[int, int]]:
    for _ in range(80):
        total = random.randint(3, 5)
        coords = {(0, 0)}
        while len(coords) < total:
            r, c = random.choice(tuple(coords))
            nr, nc = random.choice(_neighbors4(r, c))
            if -2 <= nr <= 2 and -2 <= nc <= 2:
                coords.add((nr, nc))

        transforms = [
            lambda r, c: (r, c),
            lambda r, c: (r, -c),
            lambda r, c: (-r, c),
            lambda r, c: (-r, -c),
            lambda r, c: (c, r),
            lambda r, c: (c, -r),
            lambda r, c: (-c, r),
            lambda r, c: (-c, -r),
        ]
        tf = random.choice(transforms)
        mapped = {tf(r, c) for r, c in coords}
        if (0, 0) not in mapped:
            continue
        body_offsets = sorted((int(r), int(c)) for r, c in mapped if (r, c) != (0, 0))
        rows = [0] + [r for r, _c in body_offsets]
        cols = [0] + [c for _r, c in body_offsets]
        if max(rows) - min(rows) <= 3 and max(cols) - min(cols) <= 3:
            return body_offsets
    raise RuntimeError("failed to build a sprite shape")


def _shape_to_patch(body_offsets: list[tuple[int, int]], body_color: int, marker_color: int) -> tuple[np.ndarray, tuple[int, int]]:
    coords = [(0, 0)] + list(body_offsets)
    rows = [r for r, _c in coords]
    cols = [c for _r, c in coords]
    min_r, max_r = min(rows), max(rows)
    min_c, max_c = min(cols), max(cols)
    patch = np.zeros((max_r - min_r + 1, max_c - min_c + 1), dtype=int)
    marker_pos = (-min_r, -min_c)
    patch[marker_pos] = int(marker_color)
    for dr, dc in body_offsets:
        patch[marker_pos[0] + dr, marker_pos[1] + dc] = int(body_color)
    return patch, (int(marker_pos[0]), int(marker_pos[1]))


def _bbox_fits_with_padding(occupied: np.ndarray, r0: int, c0: int, h: int, w: int, padding: int = 1) -> bool:
    H, W = occupied.shape
    rr0 = max(0, r0 - padding)
    cc0 = max(0, c0 - padding)
    rr1 = min(H, r0 + h + padding)
    cc1 = min(W, c0 + w + padding)
    return not occupied[rr0:rr1, cc0:cc1].any()


def _choose_top_cols(ic0: int, ic1: int, connected_specs: list[dict]) -> list[int] | None:
    left_margin = 0
    right_margin = 0
    for spec in connected_specs:
        dcs = [dc for _dr, dc in spec["body_offsets"]]
        left_margin = max(left_margin, max(0, -min([0] + dcs)))
        right_margin = max(right_margin, max([0] + dcs))
    candidates = list(range(ic0 + left_margin, ic1 - right_margin + 1))
    if len(candidates) < 2:
        return None
    n_top = random.randint(2, min(4, len(candidates)))
    for _ in range(120):
        cols = sorted(random.sample(candidates, n_top))
        if len(cols) >= 2 and max(cols) - min(cols) >= 2:
            return cols
    return None


def _choose_target_rows(ir0: int, ir1: int, connected_specs: list[dict]) -> list[int] | None:
    sorted_specs = sorted(connected_specs, key=lambda spec: spec["marker_abs"][0])
    feasible_rows: list[set[int]] = []
    for spec in sorted_specs:
        drs = [dr for dr, _dc in spec["body_offsets"]]
        top_margin = max(0, -min([0] + drs))
        bottom_margin = max([0] + drs)
        rows = set(range(ir0 + top_margin, ir1 - bottom_margin + 1))
        if not rows:
            return None
        feasible_rows.append(rows)

    min_spacing = 2

    def search(idx: int, prev_row: int | None) -> list[int] | None:
        if idx == len(sorted_specs):
            return []

        candidates = sorted(
            row
            for row in feasible_rows[idx]
            if prev_row is None or row - prev_row >= min_spacing
        )
        random.shuffle(candidates)
        for row in candidates:
            tail = search(idx + 1, row)
            if tail is not None:
                return [int(row)] + tail
        return None

    for _ in range(80):
        rows = search(0, None)
        if rows is None:
            return None
        if any(rows[i] != sorted_specs[i]["marker_abs"][0] for i in range(len(rows))):
            return rows
    return None


def _stamped_cells(spec: dict, row: int, cols: list[int]) -> set[tuple[int, int]]:
    painted = set()
    for col in cols:
        for dr, dc in spec["body_offsets"]:
            painted.add((int(row + dr), int(col + dc)))
    return painted


def _has_output_overlap(connected_specs: list[dict], rows: list[int], cols: list[int]) -> bool:
    sorted_specs = sorted(connected_specs, key=lambda spec: spec["marker_abs"][0])
    painted: dict[tuple[int, int], int] = {}
    overlap = False
    for spec, row in zip(sorted_specs, rows):
        for cell in _stamped_cells(spec, row, cols):
            if cell in painted and painted[cell] != spec["body_color"]:
                overlap = True
            painted[cell] = spec["body_color"]
    return overlap


def _expand8(coords: set[tuple[int, int]], H: int, W: int) -> set[tuple[int, int]]:
    expanded = set()
    for r, c in coords:
        for nr in range(r - 1, r + 2):
            for nc in range(c - 1, c + 2):
                if 0 <= nr < H and 0 <= nc < W:
                    expanded.add((int(nr), int(nc)))
    return expanded


def _candidate_starts(grid: np.ndarray, marker_abs: tuple[int, int], forbidden: set[tuple[int, int]]) -> list[tuple[int, int]]:
    mr, mc = marker_abs
    starts = []
    for nr, nc in _neighbors8(mr, mc):
        if 0 <= nr < grid.shape[0] and 0 <= nc < grid.shape[1] and int(grid[nr, nc]) == BG and (nr, nc) not in forbidden:
            starts.append((int(nr), int(nc)))
    starts.sort(key=lambda rc: (abs(rc[1] - mc), abs(rc[0] - mr)))
    return starts


def _bfs_path(
    grid: np.ndarray,
    start: tuple[int, int],
    goal: tuple[int, int],
    forbidden: set[tuple[int, int]],
) -> list[tuple[int, int]] | None:
    H, W = grid.shape
    dq = deque([start])
    prev = {start: None}
    while dq:
        cur = dq.popleft()
        if cur == goal:
            break
        nbrs = sorted(
            _neighbors8(cur[0], cur[1]),
            key=lambda rc: (abs(rc[0] - goal[0]) + abs(rc[1] - goal[1]), abs(rc[0] - cur[0]) + abs(rc[1] - cur[1])),
        )
        for nr, nc in nbrs:
            nxt = (int(nr), int(nc))
            if not (0 <= nr < H and 0 <= nc < W):
                continue
            if nxt in prev:
                continue
            if nxt != goal and int(grid[nr, nc]) != BG:
                continue
            if nxt != goal and nxt in forbidden:
                continue
            prev[nxt] = cur
            dq.append(nxt)
    if goal not in prev:
        return None
    out = []
    cur = goal
    while cur is not None:
        out.append(cur)
        cur = prev[cur]
    out.reverse()
    return out


def _route_to_goal(
    grid: np.ndarray,
    sprite: dict,
    goal: tuple[int, int],
    other_sprites: list[dict],
    used_paths: set[tuple[int, int]],
    fc0: int,
) -> list[tuple[int, int]] | None:
    H, W = grid.shape
    own_coords = set(sprite["coords"])
    own_expanded = _expand8(own_coords, H, W)
    other_forbidden: set[tuple[int, int]] = set()
    for other in other_sprites:
        other_forbidden |= _expand8(set(other["coords"]), H, W)
    blocked_by_paths = _expand8(used_paths, H, W)
    if goal in other_forbidden or goal in blocked_by_paths:
        return None
    forbidden = other_forbidden | blocked_by_paths | {cell for cell in own_expanded if cell not in _neighbors8(*sprite["marker_abs"])}
    forbidden |= {(r, fc0 - 1) for r in range(H) if (r, fc0 - 1) != goal}

    best = None
    starts = _candidate_starts(grid, sprite["marker_abs"], other_forbidden | _expand8(used_paths, H, W))
    starts.sort(key=lambda rc: abs(rc[0] - goal[0]) + abs(rc[1] - goal[1]))
    for start in starts:
        path = _bfs_path(grid, start, goal, forbidden)
        if path is None:
            continue
        if any(any((nr, nc) in own_coords for nr, nc in _neighbors8(r, c)) for r, c in path[1:]):
            continue
        if not _path_touches_coords(path[:1], own_coords):
            continue
        if any(_path_touches_coords(path, set(other["coords"])) for other in other_sprites):
            continue
        if best is None or len(path) < len(best):
            best = path
    return best


def _route_fragment(
    grid: np.ndarray,
    sprite: dict,
    other_sprites: list[dict],
    used_paths: set[tuple[int, int]],
    fc0: int,
) -> list[tuple[int, int]] | None:
    H, W = grid.shape
    own_coords = set(sprite["coords"])
    other_forbidden: set[tuple[int, int]] = set()
    for other in other_sprites:
        other_forbidden |= _expand8(set(other["coords"]), H, W)

    starts = _candidate_starts(grid, sprite["marker_abs"], other_forbidden | _expand8(used_paths, H, W))
    if not starts:
        return None

    candidate_goals = []
    for r in range(H):
        for c in range(0, max(1, fc0 - 3)):
            if int(grid[r, c]) != BG:
                continue
            if abs(r - sprite["marker_abs"][0]) + abs(c - sprite["marker_abs"][1]) < 3:
                continue
            candidate_goals.append((int(r), int(c)))
    random.shuffle(candidate_goals)

    for start in starts:
        for goal in candidate_goals[:80]:
            path = _route_to_goal(grid, sprite, goal, other_sprites, used_paths, fc0)
            if path is None:
                continue
            if any(c == fc0 - 1 for _r, c in path):
                continue
            return path
    return None


def _render_expected(spec: dict) -> np.ndarray:
    out = spec["grid"].copy()
    fr0, fc0, _fr1, _fc1 = spec["frame"]

    for sprite in spec["sprites"]:
        for r, c in sprite["coords"]:
            out[r, c] = BG
    for cell in spec["all_path_cells"]:
        out[cell] = BG
    for c in spec["top_cols"]:
        out[fr0, c] = BG
    for row in spec["row_by_color"].values():
        out[row, fc0] = BG

    connected = [sprite for sprite in spec["sprites"] if sprite["connected"]]
    connected.sort(key=lambda sprite: int(sprite["target_row"]))
    for sprite in connected:
        for col in spec["top_cols"]:
            for dr, dc in sprite["body_offsets"]:
                out[int(sprite["target_row"] + dr), int(col + dc)] = int(sprite["body_color"])
    return out


def _build_attempt() -> dict | None:
    total_sprites = random.randint(2, 4)
    disconnected_count = 1 if total_sprites >= 3 and random.random() < 0.45 else 0
    connected_count = total_sprites - disconnected_count
    if connected_count < 2:
        disconnected_count = 0
        connected_count = total_sprites

    H = random.randint(18, 24)
    left_width = random.randint(7, 10)
    frame_h = random.randint(8, min(12, H - 4))
    frame_w = random.randint(8, 11)
    W = left_width + 1 + frame_w + 1
    if W > 30:
        return None

    fr0 = random.randint(1, H - frame_h - 2)
    fc0 = left_width + 1
    fr1 = fr0 + frame_h - 1
    fc1 = fc0 + frame_w - 1
    ir0, ic0, ir1, ic1 = fr0 + 1, fc0 + 1, fr1 - 1, fc1 - 1

    sprite_shapes = [_random_shape_offsets() for _ in range(total_sprites)]
    body_colors = random.sample(BODY_COLORS, total_sprites)
    sprite_specs = []
    for idx, offsets in enumerate(sprite_shapes):
        patch, marker_local = _shape_to_patch(offsets, body_colors[idx], BLUE)
        sprite_specs.append(
            {
                "body_offsets": offsets,
                "body_color": int(body_colors[idx]),
                "patch_shape": patch.shape,
                "marker_local": marker_local,
            }
        )

    connected_specs = sprite_specs[:connected_count]
    top_cols = _choose_top_cols(ic0, ic1, connected_specs)
    if top_cols is None:
        return None

    grid = np.zeros((H, W), dtype=int)
    grid[fr0, fc0:fc1 + 1] = GREY
    grid[fr1, fc0:fc1 + 1] = GREY
    grid[fr0:fr1 + 1, fc0] = GREY
    grid[fr0:fr1 + 1, fc1] = GREY

    occupied = np.zeros((H, W), dtype=bool)
    placed_specs = []
    for spec in sorted(sprite_specs, key=lambda item: item["patch_shape"][0] * item["patch_shape"][1], reverse=True):
        h, w = spec["patch_shape"]
        placed = False
        for _ in range(250):
            r0 = random.randint(0, H - h)
            c0 = random.randint(0, fc0 - 2 - w)
            if c0 + w > fc0 - 1:
                continue
            if not _bbox_fits_with_padding(occupied, r0, c0, h, w, padding=1):
                continue
            coords = []
            marker_abs = (int(r0 + spec["marker_local"][0]), int(c0 + spec["marker_local"][1]))
            for dr, dc in [(0, 0)] + spec["body_offsets"]:
                coords.append((int(marker_abs[0] + dr), int(marker_abs[1] + dc)))
            for r, c in coords:
                occupied[r, c] = True
            placed_specs.append(
                {
                    "body_offsets": list(spec["body_offsets"]),
                    "body_color": int(spec["body_color"]),
                    "coords": coords,
                    "marker_abs": marker_abs,
                    "connected": False,
                }
            )
            placed = True
            break
        if not placed:
            return None

    placed_specs.sort(key=lambda spec: (spec["marker_abs"][0], spec["marker_abs"][1]))
    connected_specs = placed_specs[:connected_count]
    disconnected_specs = placed_specs[connected_count:]

    target_rows = _choose_target_rows(ir0, ir1, connected_specs)
    if target_rows is None:
        return None
    if not _has_output_overlap(connected_specs, target_rows, top_cols):
        return None

    connected_marker_colors = random.sample(MARKER_COLORS, connected_count)
    row_by_color = {int(color): int(row) for color, row in zip(connected_marker_colors, target_rows)}

    top_colors = [random.choice(MARKER_COLORS) for _ in top_cols]
    if len(top_cols) >= 2 and len(set(top_colors)) == 1:
        top_colors[-1] = random.choice([c for c in MARKER_COLORS if c != top_colors[0]])

    for spec, color, row in zip(connected_specs, connected_marker_colors, target_rows):
        spec["connected"] = True
        spec["marker_color"] = int(color)
        spec["target_row"] = int(row)

    for spec in disconnected_specs:
        if random.random() < 0.7:
            spec["marker_color"] = int(random.choice(connected_marker_colors))
        else:
            spec["marker_color"] = int(random.choice(MARKER_COLORS))
        spec["target_row"] = None

    for spec in placed_specs:
        marker_r, marker_c = spec["marker_abs"]
        grid[marker_r, marker_c] = int(spec["marker_color"])
        for dr, dc in spec["body_offsets"]:
            grid[marker_r + dr, marker_c + dc] = int(spec["body_color"])

    for col, color in zip(top_cols, top_colors):
        grid[fr0, col] = int(color)
    for color, row in row_by_color.items():
        grid[row, fc0] = int(color)

    all_path_cells: set[tuple[int, int]] = set()
    route_order = sorted([spec for spec in placed_specs if spec["connected"]], key=lambda spec: int(spec["target_row"]))
    for spec in route_order:
        others = [other for other in placed_specs if other is not spec]
        goal = (int(spec["target_row"]), int(fc0 - 1))
        path = _route_to_goal(grid, spec, goal, others, all_path_cells, fc0)
        if path is None:
            return None
        spec["path"] = path
        for cell in path:
            grid[cell] = GREY
        all_path_cells.update(path)

    for spec in disconnected_specs:
        others = [other for other in placed_specs if other is not spec]
        path = _route_fragment(grid, spec, others, all_path_cells, fc0)
        if path is None:
            return None
        spec["path"] = path
        for cell in path:
            grid[cell] = GREY
        all_path_cells.update(path)

    moved = [spec for spec in connected_specs if int(spec["target_row"]) != int(spec["marker_abs"][0])]
    if not moved:
        return None

    vals, counts = np.unique(grid, return_counts=True)
    if int(vals[np.argmax(counts)]) != BG:
        return None

    return {
        "grid": grid,
        "frame": (fr0, fc0, fr1, fc1),
        "inner": (ir0, ic0, ir1, ic1),
        "top_cols": top_cols,
        "row_by_color": row_by_color,
        "sprites": placed_specs,
        "all_path_cells": all_path_cells,
        "has_disconnected": bool(disconnected_specs),
    }


def _semantic_validate(spec: dict) -> bool:
    try:
        frame = _parse_frame(spec["grid"])
        parsed_sprites = _extract_sprites(spec["grid"], frame["frame"])
        sprite_data = _sprite_connectivity_data(spec["grid"], frame, parsed_sprites)
    except AssertionError:
        return False
    connected_rows = sorted(int(sprite["arrival_row"]) for sprite in sprite_data if sprite["arrival_row"] is not None)
    expected_rows = sorted(int(spec_sprite["target_row"]) for spec_sprite in spec["sprites"] if spec_sprite["connected"])
    if connected_rows != expected_rows:
        return False
    if len(frame["top_cols"]) != len(spec["top_cols"]):
        return False
    if not any(
        int(spec_sprite["target_row"]) != int(spec_sprite["marker_abs"][0])
        for spec_sprite in spec["sprites"]
        if spec_sprite["connected"]
    ):
        return False
    if spec["has_disconnected"] and all(sprite["arrival_row"] is not None for sprite in sprite_data):
        return False
    return _has_output_overlap(
        [sprite for sprite in spec["sprites"] if sprite["connected"]],
        [int(sprite["target_row"]) for sprite in sorted([s for s in spec["sprites"] if s["connected"]], key=lambda s: s["marker_abs"][0])],
        spec["top_cols"],
    )


def generate_input() -> np.ndarray:
    for _ in range(GEN_ATTEMPTS):
        spec = _build_attempt()
        if spec is None:
            continue
        expected = _render_expected(spec)
        if not _semantic_validate(spec):
            continue
        try:
            actual = main(spec["grid"])
        except AssertionError:
            continue
        if np.array_equal(actual, expected) and not np.array_equal(actual, spec["grid"]):
            return spec["grid"]
    raise RuntimeError(f"failed to generate a valid puzzle after {GEN_ATTEMPTS} attempts")
