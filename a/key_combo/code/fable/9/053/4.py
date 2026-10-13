from __future__ import annotations

import random
from collections import Counter

import numpy as np


BG = 0
FRAME = 1
MARKER_PALETTE = [2, 3, 4, 5, 6, 7, 8, 9]
MAX_ATTEMPTS = 500
ALL_DIRS = ["N", "S", "W", "E", "NW", "NE", "SW", "SE"]


def _get_bg(grid: np.ndarray) -> int:
    return int(Counter(map(int, grid.ravel())).most_common(1)[0][0])


def _bbox_coords(coords: list[tuple[int, int]]) -> tuple[int, int, int, int]:
    rows = [r for r, _ in coords]
    cols = [c for _, c in coords]
    return min(rows), min(cols), max(rows), max(cols)


def _components_by_color(grid: np.ndarray, bg: int) -> list[dict]:
    h, w = grid.shape
    seen = np.zeros((h, w), dtype=bool)
    out: list[dict] = []
    for r in range(h):
        for c in range(w):
            if seen[r, c] or int(grid[r, c]) == bg:
                continue
            color = int(grid[r, c])
            stack = [(r, c)]
            seen[r, c] = True
            coords: list[tuple[int, int]] = []
            while stack:
                rr, cc = stack.pop()
                coords.append((rr, cc))
                for nr, nc in ((rr - 1, cc), (rr + 1, cc), (rr, cc - 1), (rr, cc + 1)):
                    if 0 <= nr < h and 0 <= nc < w and not seen[nr, nc] and int(grid[nr, nc]) == color:
                        seen[nr, nc] = True
                        stack.append((nr, nc))
            out.append(
                {
                    "color": color,
                    "coords": coords,
                    "bbox": _bbox_coords(coords),
                    "area": len(coords),
                }
            )
    return out


def _is_hollow_frame(grid: np.ndarray, comp: dict) -> bool:
    r0, c0, r1, c1 = comp["bbox"]
    if r1 - r0 < 2 or c1 - c0 < 2:
        return False
    expected_area = 2 * ((r1 - r0 + 1) + (c1 - c0 + 1)) - 4
    if comp["area"] != expected_area:
        return False
    color = int(comp["color"])
    sub = grid[r0 : r1 + 1, c0 : c1 + 1]
    expected = np.zeros_like(sub, dtype=bool)
    expected[0, :] = True
    expected[-1, :] = True
    expected[:, 0] = True
    expected[:, -1] = True
    return bool(np.array_equal(sub == color, expected))


def _dist_to_frame(point: tuple[int, int], frame_bbox: tuple[int, int, int, int]) -> int:
    r, c = point
    r0, c0, r1, c1 = frame_bbox
    dr = 0 if r0 <= r <= r1 else (r0 - r if r < r0 else r - r1)
    dc = 0 if c0 <= c <= c1 else (c0 - c if c < c0 else c - c1)
    return max(dr, dc)


def _ring_mask(shape: tuple[int, int], frame_bbox: tuple[int, int, int, int], k: int) -> np.ndarray:
    r0, c0, r1, c1 = frame_bbox
    rr0, cc0 = r0 + k, c0 + k
    rr1, cc1 = r1 - k, c1 - k
    mask = np.zeros(shape, dtype=bool)
    if rr0 > rr1 or cc0 > cc1:
        return mask
    mask[rr0 : rr1 + 1, cc0 : cc1 + 1] = True
    if rr0 + 1 <= rr1 - 1 and cc0 + 1 <= cc1 - 1:
        mask[rr0 + 1 : rr1, cc0 + 1 : cc1] = False
    return mask


def _max_layer(frame_bbox: tuple[int, int, int, int]) -> int:
    r0, c0, r1, c1 = frame_bbox
    return min((r1 - r0) // 2, (c1 - c0) // 2)


def _draw_frame(grid: np.ndarray, bbox: tuple[int, int, int, int], color: int) -> None:
    r0, c0, r1, c1 = bbox
    grid[r0, c0 : c1 + 1] = color
    grid[r1, c0 : c1 + 1] = color
    grid[r0 : r1 + 1, c0] = color
    grid[r0 : r1 + 1, c1] = color


def _candidate_positions(
    shape: tuple[int, int], frame_bbox: tuple[int, int, int, int], k: int, direction: str | None
) -> list[tuple[int, int]]:
    h, w = shape
    r0, c0, r1, c1 = frame_bbox
    pts: list[tuple[int, int]] = []
    for r in range(max(0, r0 - k), min(h - 1, r1 + k) + 1):
        for c in range(max(0, c0 - k), min(w - 1, c1 + k) + 1):
            if r0 <= r <= r1 and c0 <= c <= c1:
                continue
            if _dist_to_frame((r, c), frame_bbox) != k:
                continue
            if direction == "N" and not (r == r0 - k and c0 <= c <= c1):
                continue
            if direction == "S" and not (r == r1 + k and c0 <= c <= c1):
                continue
            if direction == "W" and not (c == c0 - k and r0 <= r <= r1):
                continue
            if direction == "E" and not (c == c1 + k and r0 <= r <= r1):
                continue
            if direction == "NW" and not (r < r0 and c < c0):
                continue
            if direction == "NE" and not (r < r0 and c > c1):
                continue
            if direction == "SW" and not (r > r1 and c < c0):
                continue
            if direction == "SE" and not (r > r1 and c > c1):
                continue
            pts.append((r, c))
    return pts


def _solve_with_scene(grid: np.ndarray) -> tuple[list[tuple[int, int, int, int]], list[dict], int]:
    g = np.asarray(grid, dtype=int)
    assert g.ndim == 2, "expected a 2D grid"
    bg = _get_bg(g)
    comps = _components_by_color(g, bg)
    frames = [comp["bbox"] for comp in comps if _is_hollow_frame(g, comp)]
    assert 1 <= len(frames) <= 2, "expected one or two hollow rectangular frames"
    frames.sort()

    frame_boxes = set(frames)
    markers: list[dict] = []
    seen_layer_by_frame: set[tuple[int, int]] = set()
    for comp in comps:
        if comp["bbox"] in frame_boxes and _is_hollow_frame(g, comp):
            continue
        assert comp["area"] == 1, "all non-frame foreground objects must be singleton markers"
        r, c = comp["coords"][0]
        dists = [_dist_to_frame((r, c), frame_bbox) for frame_bbox in frames]
        best = min(dists)
        assert best >= 1, "markers must be outside the frames"
        assert dists.count(best) == 1, "each marker must have a unique nearest frame"
        frame_idx = dists.index(best)
        frame_bbox = frames[frame_idx]
        assert _ring_mask(g.shape, frame_bbox, best).any(), "marker distance must refer to an existing inner layer"
        key = (frame_idx, best)
        assert key not in seen_layer_by_frame, "at most one marker per frame-distance pair"
        seen_layer_by_frame.add(key)
        markers.append(
            {
                "pos": (int(r), int(c)),
                "color": int(comp["color"]),
                "frame_idx": frame_idx,
                "k": int(best),
            }
        )

    assert markers, "expected at least one marker"
    return frames, markers, bg


def main(grid: np.ndarray) -> np.ndarray:
    """
    Color the indicated inner rectangular rings of each hollow frame according to marker-to-frame Chebyshev distance.

    Input structure:
    - A mostly empty grid contains one or two one-pixel-thick hollow rectangular frames.
    - Outside the frames sit isolated singleton marker pixels in various colors.
    - The same frame can have markers at several different outside distances, with some distances intentionally missing.

    Rule:
    - For each marker, find the unique nearest frame using Chebyshev distance to the frame border.
    - If that distance is k, paint the k-th inner rectangular ring of that frame with the marker's color, counting inward from the frame border.
    - Leave unmentioned inner rings as background, and preserve the original frames, markers, and outer background.

    Key insight:
    - Marker position around a frame is a decoy; only the nearest-frame assignment and the marker's Chebyshev distance determine which inner ring gets colored.
    """
    g = np.asarray(grid, dtype=int)
    frames, markers, _bg = _solve_with_scene(g)
    out = g.copy()
    for marker in markers:
        ring = _ring_mask(g.shape, frames[marker["frame_idx"]], marker["k"])
        out[ring] = int(marker["color"])
    return out


def _choose_layer_subset(rng: random.Random, usable_max: int, force_gap: bool) -> list[int]:
    assert usable_max >= 3
    for _ in range(100):
        count = rng.randint(2, min(3, usable_max))
        layers = sorted(rng.sample(range(1, usable_max + 1), count))
        has_gap = layers[-1] - layers[0] + 1 > len(layers)
        if force_gap and not has_gap:
            continue
        if not force_gap and rng.random() < 0.3 and has_gap:
            return layers
        if not force_gap:
            return layers
        return layers
    raise RuntimeError("failed to choose layer subset")


def _build_layout(rng: random.Random) -> dict | None:
    n_frames = 1 if rng.random() < 0.45 else 2
    frame_specs = []
    force_gap_frame = rng.randrange(n_frames)
    for idx in range(n_frames):
        if n_frames == 1:
            fh = rng.randint(7, 11)
            fw = rng.randint(7, 11)
            usable_cap = 4
        else:
            fh = rng.randint(7, 9)
            fw = rng.randint(7, 9)
            usable_cap = 3
        bbox = (0, 0, fh - 1, fw - 1)
        usable_max = min(_max_layer(bbox), usable_cap)
        layers = _choose_layer_subset(rng, usable_max, force_gap=idx == force_gap_frame)
        frame_specs.append({"size": (fh, fw), "layers": layers})

    scene_max_k = max(max(spec["layers"]) for spec in frame_specs)
    margin_min = scene_max_k
    margin_max = scene_max_k + 2

    if n_frames == 1:
        fh, fw = frame_specs[0]["size"]
        top = rng.randint(margin_min, margin_max)
        left = rng.randint(margin_min, margin_max)
        bottom = rng.randint(margin_min, margin_max)
        right = rng.randint(margin_min, margin_max)
        h = top + fh + bottom
        w = left + fw + right
        if max(h, w) > 30:
            return None
        frame_specs[0]["bbox"] = (top, left, top + fh - 1, left + fw - 1)
        return {"shape": (h, w), "frames": frame_specs, "orientation": None}

    orientation = rng.choice(["H", "V"])
    gap = rng.randint(2 * scene_max_k, 2 * scene_max_k + 2)
    if orientation == "H":
        fh0, fw0 = frame_specs[0]["size"]
        fh1, fw1 = frame_specs[1]["size"]
        top = rng.randint(margin_min, margin_max)
        left = rng.randint(margin_min, margin_max)
        bottom = rng.randint(margin_min, margin_max)
        right = rng.randint(margin_min, margin_max)
        h = top + max(fh0, fh1) + bottom
        w = left + fw0 + gap + fw1 + right
        if max(h, w) > 30:
            return None
        frame_specs[0]["bbox"] = (top, left, top + fh0 - 1, left + fw0 - 1)
        c1 = left + fw0 + gap
        frame_specs[1]["bbox"] = (top, c1, top + fh1 - 1, c1 + fw1 - 1)
    else:
        fh0, fw0 = frame_specs[0]["size"]
        fh1, fw1 = frame_specs[1]["size"]
        top = rng.randint(margin_min, margin_max)
        left = rng.randint(margin_min, margin_max)
        bottom = rng.randint(margin_min, margin_max)
        right = rng.randint(margin_min, margin_max)
        h = top + fh0 + gap + fh1 + bottom
        w = left + max(fw0, fw1) + right
        if max(h, w) > 30:
            return None
        frame_specs[0]["bbox"] = (top, left, top + fh0 - 1, left + fw0 - 1)
        r1 = top + fh0 + gap
        frame_specs[1]["bbox"] = (r1, left, r1 + fh1 - 1, left + fw1 - 1)
    return {"shape": (h, w), "frames": frame_specs, "orientation": orientation}


def _place_markers(rng: random.Random, layout: dict) -> list[dict] | None:
    frames = [spec["bbox"] for spec in layout["frames"]]
    total_markers = sum(len(spec["layers"]) for spec in layout["frames"])
    colors = rng.sample(MARKER_PALETTE, total_markers)
    color_iter = iter(colors)
    markers: list[dict] = []
    used_positions: list[tuple[int, int]] = []
    corridor_dirs: dict[int, str] = {}
    corridor_seen: set[int] = set()
    if layout["orientation"] == "H":
        target = rng.randrange(2)
        corridor_dirs = {target: "E" if target == 0 else "W"}
    elif layout["orientation"] == "V":
        target = rng.randrange(2)
        corridor_dirs = {target: "S" if target == 0 else "N"}

    tasks: list[tuple[int, int, int, bool]] = []
    for frame_idx, spec in enumerate(layout["frames"]):
        corridor_k = max(spec["layers"]) if corridor_dirs else None
        for k in spec["layers"]:
            tasks.append((frame_idx, k, next(color_iter), corridor_k == k))
    tasks.sort(key=lambda item: (not item[3], -item[1], item[0]))

    for frame_idx, k, color, want_corridor in tasks:
        preferred = corridor_dirs.get(frame_idx) if want_corridor and frame_idx not in corridor_seen else None
        direction_order = ALL_DIRS[:]
        rng.shuffle(direction_order)
        if preferred is not None:
            direction_order = [preferred] + [d for d in direction_order if d != preferred] + [None]
        else:
            direction_order = direction_order + [None]

        placed = None
        used_dir = None
        for direction in direction_order:
            candidates = _candidate_positions(layout["shape"], frames[frame_idx], k, direction)
            rng.shuffle(candidates)
            for pos in candidates:
                if any(max(abs(pos[0] - pr), abs(pos[1] - pc)) <= 1 for pr, pc in used_positions):
                    continue
                dists = [_dist_to_frame(pos, bbox) for bbox in frames]
                best = min(dists)
                if dists[frame_idx] != best or dists.count(best) != 1:
                    continue
                placed = pos
                used_dir = direction
                break
            if placed is not None:
                break

        if placed is None:
            return None
        used_positions.append(placed)
        if preferred is not None and used_dir == preferred:
            corridor_seen.add(frame_idx)
        markers.append(
            {
                "pos": placed,
                "color": color,
                "frame_idx": frame_idx,
                "k": k,
            }
        )

    if corridor_dirs and corridor_seen != set(corridor_dirs):
        return None
    return markers


def _render_input(shape: tuple[int, int], frame_specs: list[dict], markers: list[dict]) -> np.ndarray:
    grid = np.full(shape, BG, dtype=int)
    for spec in frame_specs:
        _draw_frame(grid, spec["bbox"], FRAME)
    for marker in markers:
        r, c = marker["pos"]
        grid[r, c] = int(marker["color"])
    return grid


def _expected_output(inp: np.ndarray, frames: list[dict], markers: list[dict]) -> np.ndarray:
    out = inp.copy()
    for marker in markers:
        ring = _ring_mask(inp.shape, frames[marker["frame_idx"]]["bbox"], marker["k"])
        out[ring] = int(marker["color"])
    return out


def _validate_generated_scene(inp: np.ndarray, out: np.ndarray, layout: dict, markers: list[dict]) -> bool:
    if inp.ndim != 2 or max(inp.shape) > 30 or out.ndim != 2 or max(out.shape) > 30:
        return False
    if _get_bg(inp) != BG:
        return False

    forced_gap = False
    for spec in layout["frames"]:
        layers = sorted(spec["layers"])
        if layers[-1] - layers[0] + 1 > len(layers):
            forced_gap = True
        interior = inp[spec["bbox"][0] + 1 : spec["bbox"][2], spec["bbox"][1] + 1 : spec["bbox"][3]]
        if not np.all(interior == BG):
            return False
    if not forced_gap:
        return False

    parsed_frames, parsed_markers, bg = _solve_with_scene(inp)
    if bg != BG or len(parsed_frames) != len(layout["frames"]) or len(parsed_markers) != len(markers):
        return False

    seen_layers: dict[int, set[int]] = {}
    for marker in markers:
        seen_layers.setdefault(marker["frame_idx"], set()).add(marker["k"])
        d = _dist_to_frame(marker["pos"], layout["frames"][marker["frame_idx"]]["bbox"])
        if d != marker["k"]:
            return False
        all_dists = [_dist_to_frame(marker["pos"], spec["bbox"]) for spec in layout["frames"]]
        if all_dists.count(min(all_dists)) != 1:
            return False
    for frame_idx, layers in seen_layers.items():
        if len(layers) != len(layout["frames"][frame_idx]["layers"]):
            return False

    for frame_idx, spec in enumerate(layout["frames"]):
        chosen = {marker["k"]: marker["color"] for marker in markers if marker["frame_idx"] == frame_idx}
        max_layer = _max_layer(spec["bbox"])
        for k in range(1, max_layer + 1):
            ring = _ring_mask(inp.shape, spec["bbox"], k)
            ring_vals = set(map(int, out[ring]))
            if k in chosen:
                if ring_vals != {chosen[k]}:
                    return False
            else:
                if ring_vals != {BG}:
                    return False

    try:
        solved = main(inp)
    except Exception:
        return False
    return bool(np.array_equal(solved, out))


def _gen_attempt(rng: random.Random) -> np.ndarray | None:
    layout = _build_layout(rng)
    if layout is None:
        return None
    markers = _place_markers(rng, layout)
    if markers is None:
        return None
    inp = _render_input(layout["shape"], layout["frames"], markers)
    out = _expected_output(inp, layout["frames"], markers)
    if not _validate_generated_scene(inp, out, layout, markers):
        return None
    return inp


def generate_input() -> np.ndarray:
    rng = random.Random()
    for _ in range(MAX_ATTEMPTS):
        grid = _gen_attempt(rng)
        if grid is not None:
            return grid
    raise RuntimeError("failed to generate a valid distance-ring input")
