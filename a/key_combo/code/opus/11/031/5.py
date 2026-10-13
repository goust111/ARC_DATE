from __future__ import annotations

from collections import Counter, deque
import random
from typing import Iterable

import numpy as np


BG = 0
ANCHOR_COLORS = (1, 2, 3)
BODY_COLORS = (4, 6)
ORANGE = 7


def _neighbors4(r: int, c: int) -> list[tuple[int, int]]:
    return [(r - 1, c), (r + 1, c), (r, c - 1), (r, c + 1)]


def _get_bg(grid: np.ndarray) -> int:
    return int(Counter(map(int, grid.reshape(-1))).most_common(1)[0][0])


def _bbox(coords: list[tuple[int, int]]) -> tuple[int, int, int, int]:
    rows = [r for r, _ in coords]
    cols = [c for _, c in coords]
    return min(rows), min(cols), max(rows), max(cols)


def _component_from_mask(mask: np.ndarray) -> list[list[tuple[int, int]]]:
    h, w = mask.shape
    seen = np.zeros_like(mask, dtype=bool)
    comps: list[list[tuple[int, int]]] = []
    for r in range(h):
        for c in range(w):
            if not mask[r, c] or seen[r, c]:
                continue
            q = deque([(r, c)])
            seen[r, c] = True
            comp: list[tuple[int, int]] = []
            while q:
                rr, cc = q.popleft()
                comp.append((rr, cc))
                for nr, nc in _neighbors4(rr, cc):
                    if 0 <= nr < h and 0 <= nc < w and mask[nr, nc] and not seen[nr, nc]:
                        seen[nr, nc] = True
                        q.append((nr, nc))
            comps.append(comp)
    return comps


def _crop_patch(grid: np.ndarray, coords: list[tuple[int, int]], bg: int) -> tuple[np.ndarray, tuple[int, int, int, int]]:
    r0, c0, r1, c1 = _bbox(coords)
    patch = np.full((r1 - r0 + 1, c1 - c0 + 1), int(bg), dtype=int)
    for r, c in coords:
        patch[r - r0, c - c0] = int(grid[r, c])
    return patch, (r0, c0, r1, c1)


def _feasible_mask(traversable: np.ndarray, h: int, w: int) -> np.ndarray:
    rows, cols = traversable.shape
    out = np.zeros((rows - h + 1, cols - w + 1), dtype=bool)
    for r in range(rows - h + 1):
        for c in range(cols - w + 1):
            out[r, c] = bool(traversable[r : r + h, c : c + w].all())
    return out


def _reachable(feasible: np.ndarray, start: tuple[int, int], goal: tuple[int, int]) -> bool:
    if start == goal:
        return True
    rows, cols = feasible.shape
    if not (0 <= start[0] < rows and 0 <= start[1] < cols and feasible[start]):
        return False
    if not (0 <= goal[0] < rows and 0 <= goal[1] < cols and feasible[goal]):
        return False
    seen = np.zeros_like(feasible, dtype=bool)
    q = deque([start])
    seen[start] = True
    while q:
        r, c = q.popleft()
        for nr, nc in _neighbors4(r, c):
            if 0 <= nr < rows and 0 <= nc < cols and feasible[nr, nc] and not seen[nr, nc]:
                if (nr, nc) == goal:
                    return True
                seen[nr, nc] = True
                q.append((nr, nc))
    return False


def _entry_candidates(feasible: np.ndarray, start: tuple[int, int]) -> list[tuple[int, int]]:
    rows, cols = feasible.shape
    out = []
    for dr, dc in [(-1, 0), (1, 0), (0, -1), (0, 1)]:
        for step in range(1, max(rows, cols) + 1):
            rr = start[0] + dr * step
            cc = start[1] + dc * step
            if not (0 <= rr < rows and 0 <= cc < cols):
                break
            if feasible[rr, cc]:
                out.append((rr, cc))
                break
    return out


def _solve(grid: np.ndarray, *, return_meta: bool = False):
    g = np.asarray(grid, dtype=int)
    rows, cols = g.shape
    bg = _get_bg(g)

    anchors: dict[int, tuple[int, int]] = {}
    for r in range(rows):
        for c in range(cols):
            if r not in (0, rows - 1) and c not in (0, cols - 1):
                continue
            color = int(g[r, c])
            if color in (int(bg), ORANGE):
                continue
            if color in anchors:
                continue
            if any(0 <= nr < rows and 0 <= nc < cols and int(g[nr, nc]) == ORANGE for nr, nc in _neighbors4(r, c)):
                anchors[color] = (r, c)

    orange_mask = g == ORANGE
    channel_coords = _component_from_mask(orange_mask)
    channels = [set(comp) for comp in channel_coords]
    anchor_to_channel: dict[int, int] = {}
    for color, (r, c) in anchors.items():
        hits = {
            idx
            for nr, nc in _neighbors4(r, c)
            if 0 <= nr < rows and 0 <= nc < cols and orange_mask[nr, nc]
            for idx, comp in enumerate(channels)
            if (nr, nc) in comp
        }
        if len(hits) == 1:
            anchor_to_channel[color] = next(iter(hits))

    sprite_mask = (g != bg) & (g != ORANGE)
    sprite_mask[0, :] = False
    sprite_mask[-1, :] = False
    sprite_mask[:, 0] = False
    sprite_mask[:, -1] = False
    sprite_components = _component_from_mask(sprite_mask)

    moves = []
    meta = []
    for comp in sprite_components:
        patch, box = _crop_patch(g, comp, bg)
        r0, c0, r1, c1 = box
        h = r1 - r0 + 1
        w = c1 - c0 + 1
        colors = Counter(int(g[r, c]) for r, c in comp)
        marker_colors = [color for color in colors if color in anchors]
        if len(marker_colors) != 1 or colors[marker_colors[0]] != 1:
            meta.append({"bbox": box, "moved": False, "reason": "marker"})
            continue
        marker_color = marker_colors[0]
        if marker_color not in anchor_to_channel:
            meta.append({"bbox": box, "moved": False, "reason": "anchor"})
            continue
        marker_abs = next((r, c) for r, c in comp if int(g[r, c]) == marker_color)
        off_r = marker_abs[0] - r0
        off_c = marker_abs[1] - c0
        anchor_r, anchor_c = anchors[marker_color]
        dst_r0 = anchor_r - off_r
        dst_c0 = anchor_c - off_c

        traversable = np.zeros_like(g, dtype=bool)
        for rr, cc in channels[anchor_to_channel[marker_color]]:
            traversable[rr, cc] = True
        traversable[anchor_r, anchor_c] = True

        if h > rows or w > cols:
            meta.append({"bbox": box, "moved": False, "reason": "size"})
            continue

        feasible = _feasible_mask(traversable, h, w)
        start = (r0, c0)
        entries = _entry_candidates(feasible, start)
        dst_ok = 0 <= dst_r0 < feasible.shape[0] and 0 <= dst_c0 < feasible.shape[1] and feasible[dst_r0, dst_c0]
        can_move = len(entries) == 1 and dst_ok and _reachable(feasible, entries[0], (dst_r0, dst_c0))
        meta.append(
            {
                "bbox": box,
                "moved": can_move,
                "entry_candidates": entries,
                "marker_color": marker_color,
                "dest": (dst_r0, dst_c0),
            }
        )
        if can_move:
            moves.append((comp, patch, box, (dst_r0, dst_c0)))

    out = g.copy()
    for comp, _patch, _box, _dst in moves:
        for r, c in comp:
            out[r, c] = int(bg)
    for _comp, patch, _box, (dst_r0, dst_c0) in moves:
        for rr in range(patch.shape[0]):
            for cc in range(patch.shape[1]):
                val = int(patch[rr, cc])
                if val != int(bg):
                    out[dst_r0 + rr, dst_c0 + cc] = val

    if return_meta:
        return out, meta
    return out


def main(grid: np.ndarray) -> np.ndarray:
    """
    Slide each sprite through its color-matched channel only if its bounding box fits every bottleneck.

    Input structure:
    - Black background contains 2-3 disjoint orange winding channels.
    - Each channel has one colored border anchor at its border end and one interior opening near a multi-color sprite.
    - Each sprite is a small 4-connected rectangle of body color with one marker pixel whose color matches exactly one anchor.

    Rule:
    - For each sprite, select the channel whose border anchor has the same color as the sprite's marker pixel.
    - Treat the sprite's full bounding box as the moving footprint; the sprite may enter only one step through its adjacent opening and may continue only through channel placements where that whole bounding box fits.
    - If the anchor-aligned destination inside the matched channel is reachable, move the entire sprite there so its marker pixel covers the anchor and clear the original sprite cells to background.
    - If the matched channel has a bottleneck too narrow for the sprite's bounding box, leave that sprite unchanged.

    Key insight:
    - Color chooses which channel matters, but width decides whether movement is allowed at all: the matched anchor is necessary yet insufficient without bounding-box passability along the whole corridor.
    """
    return _solve(grid)


def _sign(x: int) -> int:
    return 0 if x == 0 else (1 if x > 0 else -1)


def _compress_waypoints(points: Iterable[tuple[int, int]]) -> list[tuple[int, int]]:
    out: list[tuple[int, int]] = []
    for p in points:
        if not out or out[-1] != p:
            out.append(p)
    changed = True
    while changed and len(out) >= 3:
        changed = False
        nxt = [out[0]]
        for i in range(1, len(out) - 1):
            a = nxt[-1]
            b = out[i]
            c = out[i + 1]
            if (a[0] == b[0] == c[0]) or (a[1] == b[1] == c[1]):
                changed = True
                continue
            nxt.append(b)
        nxt.append(out[-1])
        out = nxt
    return out


def _path_positions(waypoints: list[tuple[int, int]]) -> tuple[list[tuple[int, int]], list[dict]]:
    pts: list[tuple[int, int]] = [waypoints[0]]
    segments: list[dict] = []
    for a, b in zip(waypoints, waypoints[1:]):
        r0, c0 = a
        r1, c1 = b
        if r0 != r1 and c0 != c1:
            raise ValueError("waypoints must be axis-aligned")
        if r0 == r1:
            step = _sign(c1 - c0)
            seg_pts = [(r0, c) for c in range(c0, c1 + step, step)]
            orientation = "h"
        else:
            step = _sign(r1 - r0)
            seg_pts = [(r, c0) for r in range(r0, r1 + step, step)]
            orientation = "v"
        pts.extend(seg_pts[1:])
        segments.append({"a": a, "b": b, "orientation": orientation})
    return pts, segments


def _paint_rect(mask: np.ndarray, r0: int, c0: int, h: int, w: int, value: bool = True) -> None:
    mask[r0 : r0 + h, c0 : c0 + w] = value


def _render_channel(path: list[tuple[int, int]], h: int, w: int, shape: tuple[int, int]) -> np.ndarray:
    mask = np.zeros(shape, dtype=bool)
    for r0, c0 in path:
        _paint_rect(mask, r0, c0, h, w, True)
    return mask


def _segment_span(seg: dict, h: int, w: int) -> tuple[int, int, int, int]:
    (r0, c0), (r1, c1) = seg["a"], seg["b"]
    if seg["orientation"] == "h":
        left = min(c0, c1)
        right = max(c0, c1) + w - 1
        return r0, r0 + h - 1, left, right
    top = min(r0, r1)
    bottom = max(r0, r1) + h - 1
    return top, bottom, c0, c0 + w - 1


def _widen_segment(mask: np.ndarray, seg: dict, h: int, w: int, zone: tuple[int, int, int, int]) -> bool:
    zr0, zc0, zr1, zc1 = zone
    sr0, sr1, sc0, sc1 = _segment_span(seg, h, w)
    if seg["orientation"] == "h":
        choices = []
        if sr0 - 1 >= zr0:
            choices.append(sr0 - 1)
        if sr1 + 1 <= zr1:
            choices.append(sr1 + 1)
        if not choices:
            return False
        rr = random.choice(choices)
        mask[rr, sc0 : sc1 + 1] = True
        return True
    choices = []
    if sc0 - 1 >= zc0:
        choices.append(sc0 - 1)
    if sc1 + 1 <= zc1:
        choices.append(sc1 + 1)
    if not choices:
        return False
    cc = random.choice(choices)
    mask[sr0 : sr1 + 1, cc] = True
    return True


def _carve_bottleneck(mask: np.ndarray, seg: dict, h: int, w: int) -> bool:
    sr0, sr1, sc0, sc1 = _segment_span(seg, h, w)
    if seg["orientation"] == "h":
        if h < 3 or sc1 - sc0 + 1 < w + 2:
            return False
        row = random.choice([sr0, sr1])
        lo = sc0 + 1
        hi = sc1 - 1
        if lo > hi:
            return False
        length = random.randint(1, min(4, hi - lo + 1))
        start = random.randint(lo, hi - length + 1)
        mask[row, start : start + length] = False
        return True
    if w < 3 or sr1 - sr0 + 1 < h + 2:
        return False
    col = random.choice([sc0, sc1])
    lo = sr0 + 1
    hi = sr1 - 1
    if lo > hi:
        return False
    length = random.randint(1, min(4, hi - lo + 1))
    start = random.randint(lo, hi - length + 1)
    mask[start : start + length, col] = False
    return True


def _side_setup(side: str, grid_h: int, grid_w: int, h: int, w: int) -> tuple[tuple[int, int, int, int], tuple[int, int], tuple[int, int], tuple[int, int], tuple[int, int]]:
    mid_r = grid_h // 2
    mid_c = grid_w // 2
    if side == "top":
        zone = (0, 2, mid_r, mid_c - 3)
        entry = (zone[2], random.randint(max(zone[1] + 1, zone[3] - 2), zone[3]))
        dest = (0, random.randint(zone[1] + 1, zone[3]))
        start = (entry[0] + h, entry[1])
        outward = (1, 0)
        return zone, entry, dest, start, outward
    if side == "right":
        zone = (2, mid_c + 1, mid_r - 2, grid_w - w)
        entry = (random.randint(max(zone[0], zone[2] - 2), zone[2]), zone[1])
        dest = (random.randint(zone[0], zone[2] - 1), grid_w - w)
        start = (entry[0], entry[1] - w)
        outward = (0, -1)
        return zone, entry, dest, start, outward
    if side == "bottom":
        zone = (mid_r + 1, mid_c + 2, grid_h - h, grid_w - 4)
        entry = (zone[0], random.randint(zone[1], min(zone[3] - 1, zone[1] + 2)))
        dest = (grid_h - h, random.randint(zone[1], zone[3] - 1))
        start = (entry[0] - h, entry[1])
        outward = (-1, 0)
        return zone, entry, dest, start, outward
    if side == "left":
        zone = (mid_r + 2, 0, grid_h - h - 1, mid_c - 3)
        entry = (random.randint(zone[0], min(zone[2], zone[0] + 2)), zone[3])
        dest = (random.randint(zone[0], zone[2]), 0)
        start = (entry[0], entry[1] + w)
        outward = (0, 1)
        return zone, entry, dest, start, outward
    raise ValueError("bad side")


def _build_waypoints(side: str, zone: tuple[int, int, int, int], entry: tuple[int, int], dest: tuple[int, int]) -> list[tuple[int, int]]:
    zr0, zc0, zr1, zc1 = zone
    sr, sc = entry
    dr, dc = dest
    if side == "top":
        mid_r = random.randint(max(1, zr0 + 1), max(1, sr - 2))
        bend_c = random.randint(zc0 + 1, zc1)
        pts = [(sr, sc), (mid_r, sc), (mid_r, bend_c), (0, bend_c), (0, dc)]
    elif side == "right":
        mid_c = random.randint(min(sc + 2, zc1), max(sc + 2, zc1))
        bend_r = random.randint(zr0, zr1 - 1)
        pts = [(sr, sc), (sr, mid_c), (bend_r, mid_c), (bend_r, zc1), (dr, zc1)]
    elif side == "bottom":
        mid_r = random.randint(sr + 2, zr1)
        bend_c = random.randint(zc0, zc1 - 1)
        pts = [(sr, sc), (mid_r, sc), (mid_r, bend_c), (zr1, bend_c), (zr1, dc)]
    else:
        mid_c = random.randint(zc0, max(zc0, sc - 2))
        bend_r = random.randint(zr0, zr1)
        pts = [(sr, sc), (sr, mid_c), (bend_r, mid_c), (bend_r, 0), (dr, 0)]
    return _compress_waypoints(pts)


def _marker_offset(side: str, h: int, w: int) -> tuple[int, int]:
    if side == "top":
        return 0, random.randint(0, w - 1)
    if side == "bottom":
        return h - 1, random.randint(0, w - 1)
    if side == "left":
        return random.randint(0, h - 1), 0
    return random.randint(0, h - 1), w - 1


def _sprite_patch(h: int, w: int, body_color: int, marker_color: int, marker_off: tuple[int, int]) -> np.ndarray:
    patch = np.full((h, w), int(body_color), dtype=int)
    patch[marker_off] = int(marker_color)
    return patch


def _paint_patch(grid: np.ndarray, patch: np.ndarray, top_left: tuple[int, int]) -> None:
    r0, c0 = top_left
    for r in range(patch.shape[0]):
        for c in range(patch.shape[1]):
            val = int(patch[r, c])
            if val != BG:
                grid[r0 + r, c0 + c] = val


def _channel_and_sprite_specs(num_channels: int) -> list[dict]:
    templates = {
        2: [
            ["top", "right"],
            ["right", "bottom"],
            ["bottom", "left"],
            ["left", "top"],
        ],
        3: [
            ["top", "right", "bottom"],
            ["right", "bottom", "left"],
            ["bottom", "left", "top"],
            ["left", "top", "right"],
        ],
    }
    sides = random.choice(templates[num_channels])
    colors = random.sample(list(ANCHOR_COLORS), len(sides))
    blocked_count = random.randint(1, len(sides) - 1)
    blocked_indices = set(random.sample(range(len(sides)), blocked_count))
    specs = []
    for i, side in enumerate(sides):
        blocked = i in blocked_indices
        if blocked:
            if random.random() < 0.5:
                h, w = 3, random.choice([2, 3])
                block_on = "h"
            else:
                h, w = random.choice([2, 3]), 3
                block_on = "v"
        else:
            h, w = random.choice([(2, 2), (2, 3), (3, 2), (3, 3)])
            block_on = None
        specs.append(
            {
                "side": side,
                "anchor_color": colors[i],
                "body_color": BODY_COLORS[i % len(BODY_COLORS)],
                "h": h,
                "w": w,
                "blocked": blocked,
                "block_on": block_on,
            }
        )
    return specs


def _naive_color_only(grid: np.ndarray) -> np.ndarray:
    g = np.asarray(grid, dtype=int).copy()
    bg = _get_bg(g)
    rows, cols = g.shape
    anchors: dict[int, tuple[int, int]] = {}
    for r in range(rows):
        for c in range(cols):
            if r not in (0, rows - 1) and c not in (0, cols - 1):
                continue
            val = int(g[r, c])
            if val in (bg, ORANGE):
                continue
            if val not in anchors:
                anchors[val] = (r, c)

    sprite_mask = (g != bg) & (g != ORANGE)
    sprite_mask[0, :] = False
    sprite_mask[-1, :] = False
    sprite_mask[:, 0] = False
    sprite_mask[:, -1] = False
    comps = _component_from_mask(sprite_mask)
    moves = []
    for comp in comps:
        patch, box = _crop_patch(g, comp, bg)
        r0, c0, _r1, _c1 = box
        colors = Counter(int(g[r, c]) for r, c in comp)
        marker_colors = [color for color in colors if color in anchors and colors[color] == 1]
        if len(marker_colors) != 1:
            continue
        marker = marker_colors[0]
        mr, mc = next((r, c) for r, c in comp if int(g[r, c]) == marker)
        off = (mr - r0, mc - c0)
        ar, ac = anchors[marker]
        moves.append((comp, patch, (ar - off[0], ac - off[1])))
    out = g.copy()
    for comp, _patch, _dst in moves:
        for r, c in comp:
            out[r, c] = bg
    for _comp, patch, (r0, c0) in moves:
        if 0 <= r0 <= rows - patch.shape[0] and 0 <= c0 <= cols - patch.shape[1]:
            _paint_patch(out, patch, (r0, c0))
    return out


def _build_candidate() -> np.ndarray | None:
    grid_h = random.randint(22, 28)
    grid_w = random.randint(22, 28)
    num_channels = random.randint(2, 3)
    specs = _channel_and_sprite_specs(num_channels)

    canvas = np.zeros((grid_h, grid_w), dtype=int)
    occupied = np.zeros_like(canvas, dtype=bool)
    sprite_starts = []
    expected_blocked = 0

    for spec in specs:
        side = spec["side"]
        h = spec["h"]
        w = spec["w"]
        zone, entry, dest_tl, start_tl, _outward = _side_setup(side, grid_h, grid_w, h, w)
        marker_off = _marker_offset(side, h, w)
        anchor = (dest_tl[0] + marker_off[0], dest_tl[1] + marker_off[1])
        if side == "top" and anchor[0] != 0:
            return None
        if side == "right" and anchor[1] != grid_w - 1:
            return None
        if side == "bottom" and anchor[0] != grid_h - 1:
            return None
        if side == "left" and anchor[1] != 0:
            return None

        patch = _sprite_patch(h, w, spec["body_color"], spec["anchor_color"], marker_off)
        waypoints = _build_waypoints(side, zone, entry, dest_tl)
        path, segments = _path_positions(waypoints)
        mask = _render_channel(path, h, w, canvas.shape)

        widenable = [seg for seg in segments if (seg["orientation"] == "h" and h < 4) or (seg["orientation"] == "v" and w < 4)]
        if widenable:
            _widen_segment(mask, random.choice(widenable), h, w, zone)

        if spec["blocked"]:
            candidates = [seg for seg in segments if seg["orientation"] == spec["block_on"]]
            candidates = [seg for seg in candidates if seg != segments[0] and seg != segments[-1]]
            if not candidates:
                candidates = [seg for seg in segments if seg["orientation"] == spec["block_on"]]
            if not candidates or not _carve_bottleneck(mask, random.choice(candidates), h, w):
                return None
            expected_blocked += 1

        if not (0 <= start_tl[0] <= grid_h - h and 0 <= start_tl[1] <= grid_w - w):
            return None
        start_rect = np.zeros_like(canvas, dtype=bool)
        _paint_rect(start_rect, start_tl[0], start_tl[1], h, w, True)
        if (start_rect & mask).any():
            return None
        if (start_rect & occupied).any():
            return None
        if (mask & occupied).any():
            return None
        if anchor[0] not in (0, grid_h - 1) and anchor[1] not in (0, grid_w - 1):
            return None

        for nr, nc in _neighbors4(*anchor):
            if 0 <= nr < grid_h and 0 <= nc < grid_w and (nr, nc) != anchor and mask[nr, nc]:
                break
        else:
            return None

        canvas[mask] = ORANGE
        canvas[anchor] = spec["anchor_color"]
        occupied |= mask
        occupied[anchor] = True
        sprite_starts.append((patch, start_tl))

    for patch, start_tl in sprite_starts:
        r0, c0 = start_tl
        if (canvas[r0 : r0 + patch.shape[0], c0 : c0 + patch.shape[1]] != BG).any():
            return None
        _paint_patch(canvas, patch, start_tl)

    out, meta = _solve(canvas, return_meta=True)
    moved = sum(1 for item in meta if item.get("moved"))
    stayed = len(meta) - moved
    if len(meta) != len(specs):
        return None
    if moved < 1 or stayed < 1:
        return None
    if sum(1 for item in meta if not item.get("moved")) < expected_blocked:
        return None
    if any(len(item.get("entry_candidates", [])) != 1 for item in meta):
        return None
    if np.array_equal(out, canvas):
        return None
    if np.array_equal(_naive_color_only(canvas), out):
        return None

    return canvas


def generate_input() -> np.ndarray:
    for _ in range(400):
        grid = _build_candidate()
        if grid is not None:
            return grid
    raise RuntimeError("failed to generate a valid sample")
