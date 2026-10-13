from __future__ import annotations

from collections import Counter
from itertools import combinations

import numpy as np


BG = 0
FRAME = 5
MARKER_COLORS = (1, 2, 3, 4, 7)
MAX_GEN_ATTEMPTS = 500


def _neighbors4(r: int, c: int):
    yield r - 1, c
    yield r + 1, c
    yield r, c - 1
    yield r, c + 1


def _bbox(coords: list[tuple[int, int]]) -> tuple[int, int, int, int]:
    rows = [r for r, _ in coords]
    cols = [c for _, c in coords]
    return min(rows), min(cols), max(rows), max(cols)


def _find_components_of_color(grid: np.ndarray, color: int) -> list[list[tuple[int, int]]]:
    h, w = grid.shape
    seen = np.zeros((h, w), dtype=bool)
    comps: list[list[tuple[int, int]]] = []
    for r in range(h):
        for c in range(w):
            if seen[r, c] or int(grid[r, c]) != color:
                continue
            stack = [(r, c)]
            seen[r, c] = True
            comp: list[tuple[int, int]] = []
            while stack:
                rr, cc = stack.pop()
                comp.append((rr, cc))
                for nr, nc in _neighbors4(rr, cc):
                    if 0 <= nr < h and 0 <= nc < w and not seen[nr, nc] and int(grid[nr, nc]) == color:
                        seen[nr, nc] = True
                        stack.append((nr, nc))
            comps.append(comp)
    return comps


def _frame_dict(coords: list[tuple[int, int]]) -> dict | None:
    r0, c0, r1, c1 = _bbox(coords)
    height = r1 - r0 + 1
    width = c1 - c0 + 1
    if not (3 <= height <= 6 and 3 <= width <= 6):
        return None
    border = {
        (r0, c) for c in range(c0, c1 + 1)
    } | {
        (r1, c) for c in range(c0, c1 + 1)
    } | {
        (r, c0) for r in range(r0, r1 + 1)
    } | {
        (r, c1) for r in range(r0, r1 + 1)
    }
    coord_set = set(coords)
    if coord_set != border:
        return None
    return {
        "bbox": (r0, c0, r1, c1),
        "border": border,
        "height": height,
        "width": width,
    }


def _find_frames(grid: np.ndarray) -> list[dict]:
    frames = []
    for comp in _find_components_of_color(grid, FRAME):
        frame = _frame_dict(comp)
        if frame is None:
            continue
        frames.append(frame)
    frames.sort(key=lambda frame: frame["bbox"])
    return frames


def _find_markers(grid: np.ndarray) -> list[dict]:
    h, w = grid.shape
    markers = []
    for c in range(1, w - 1):
        if int(grid[0, c]) not in (BG, FRAME):
            markers.append({"start": (0, c), "color": int(grid[0, c])})
        if int(grid[h - 1, c]) not in (BG, FRAME):
            markers.append({"start": (h - 1, c), "color": int(grid[h - 1, c])})
    for r in range(1, h - 1):
        if int(grid[r, 0]) not in (BG, FRAME):
            markers.append({"start": (r, 0), "color": int(grid[r, 0])})
        if int(grid[r, w - 1]) not in (BG, FRAME):
            markers.append({"start": (r, w - 1), "color": int(grid[r, w - 1])})
    markers.sort(key=lambda marker: marker["start"])
    return markers


def _ray_direction(shape: tuple[int, int], start: tuple[int, int]) -> tuple[int, int, str]:
    h, w = shape
    r, c = start
    if r == 0:
        return 1, 0, "top"
    if r == h - 1:
        return -1, 0, "bottom"
    if c == 0:
        return 0, 1, "left"
    if c == w - 1:
        return 0, -1, "right"
    raise ValueError("marker must lie on the boundary")


def _trace_ray(grid: np.ndarray, start: tuple[int, int], frame_at: dict[tuple[int, int], int]) -> dict | None:
    h, w = grid.shape
    dr, dc, side = _ray_direction(grid.shape, start)
    r, c = start
    rr, cc = r + dr, c + dc
    path: list[tuple[int, int]] = []
    while 0 <= rr < h and 0 <= cc < w:
        if int(grid[rr, cc]) == FRAME:
            frame_idx = frame_at.get((rr, cc))
            if frame_idx is None:
                return None
            if path:
                endpoint = path[-1]
            else:
                endpoint = start
            return {
                "start": start,
                "side": side,
                "direction": (dr, dc),
                "hit_cell": (rr, cc),
                "frame_idx": frame_idx,
                "distance": len(path),
                "endpoint": endpoint,
                "path": tuple(path),
            }
        path.append((rr, cc))
        rr += dr
        cc += dc
    return None


def _analyze_grid(grid: np.ndarray) -> dict:
    g = np.asarray(grid, dtype=int)
    if g.ndim != 2:
        raise AssertionError("expected a 2D grid")

    frames = _find_frames(g)
    if not (2 <= len(frames) <= 4):
        raise AssertionError("expected 2 to 4 hollow frames")

    frame_at: dict[tuple[int, int], int] = {}
    for idx, frame in enumerate(frames):
        for cell in frame["border"]:
            frame_at[cell] = idx

    markers = _find_markers(g)
    if not (3 <= len(markers) <= 5):
        raise AssertionError("expected 3 to 5 boundary markers")

    hits_by_frame: list[list[dict]] = [[] for _ in frames]
    paths: list[set[tuple[int, int]]] = []
    for marker in markers:
        start = marker["start"]
        hit = _trace_ray(g, start, frame_at)
        if hit is None:
            raise AssertionError("every marker must hit a frame")
        hit["color"] = marker["color"]
        hits_by_frame[hit["frame_idx"]].append(hit)
        paths.append(set(hit["path"]))

    winners: dict[int, dict] = {}
    for idx, hits in enumerate(hits_by_frame):
        if not hits:
            continue
        ranked = sorted(hits, key=lambda hit: (hit["distance"], hit["start"]))
        if len(ranked) >= 2 and ranked[0]["distance"] == ranked[1]["distance"]:
            raise AssertionError("frame competitions must have a unique shortest ray")
        winners[idx] = ranked[0]

    return {
        "grid": g,
        "frames": frames,
        "markers": markers,
        "hits_by_frame": hits_by_frame,
        "winners": winners,
        "paths": paths,
    }


def _transform_from_analysis(info: dict) -> np.ndarray:
    grid = info["grid"]
    out = grid.copy()

    for idx, winner in info["winners"].items():
        r0, c0, r1, c1 = info["frames"][idx]["bbox"]
        out[r0 + 1 : r1, c0 + 1 : c1] = int(winner["color"])

    for hits in info["hits_by_frame"]:
        for hit in hits:
            for r, c in hit["path"]:
                out[r, c] = BG

    for marker in info["markers"]:
        r, c = marker["start"]
        out[r, c] = BG

    return out


def main(grid: np.ndarray) -> np.ndarray:
    """
    Fill each hit hollow frame with the color of its shortest competing boundary ray.

    Input structure:
    - The grid has a black background with 2 to 4 grey hollow rectangular frames.
    - Distinct colored marker pixels sit on non-corner boundary cells and point inward by their boundary side.
    - Each marker's inward row or column reaches at least one frame border before leaving the canvas.

    Rule:
    - Cast one orthogonal ray from every boundary marker toward the grid interior.
    - Stop the ray immediately before the first grey frame-border pixel that it encounters, and record the hit frame plus the ray distance.
    - For each frame hit by at least one ray, fill its interior with the color of the unique shortest ray that hit it; keep the grey border unchanged.
    - Erase all boundary markers and any ray-path trace cells back to black.

    Key insight:
    - The marker-to-frame distance is not just reachability information: when several rays hit the same frame, the shortest one uniquely wins the frame's fill color.
    """
    return _transform_from_analysis(_analyze_grid(grid))


def _boxes_too_close(a: tuple[int, int, int, int], b: tuple[int, int, int, int], pad: int = 1) -> bool:
    return not (
        a[2] + pad < b[0]
        or b[2] + pad < a[0]
        or a[3] + pad < b[1]
        or b[3] + pad < a[1]
    )


def _draw_frame(grid: np.ndarray, bbox: tuple[int, int, int, int]) -> None:
    r0, c0, r1, c1 = bbox
    grid[r0, c0 : c1 + 1] = FRAME
    grid[r1, c0 : c1 + 1] = FRAME
    grid[r0 : r1 + 1, c0] = FRAME
    grid[r0 : r1 + 1, c1] = FRAME


def _sample_frames(rng: np.random.Generator) -> np.ndarray | None:
    n_frames = int(rng.integers(2, 5))
    h = int(rng.integers(max(15, 10 + n_frames), 21))
    w = int(rng.integers(15, 21))
    frames: list[tuple[int, int, int, int]] = []

    for _ in range(n_frames):
        placed = False
        for _ in range(300):
            fh = int(rng.integers(3, 7))
            fw = int(rng.integers(3, 7))
            if fh >= h - 1 or fw >= w - 1:
                continue
            r0 = int(rng.integers(1, h - fh))
            c0 = int(rng.integers(1, w - fw))
            box = (r0, c0, r0 + fh - 1, c0 + fw - 1)
            if any(_boxes_too_close(box, other, pad=1) for other in frames):
                continue
            frames.append(box)
            placed = True
            break
        if not placed:
            return None

    grid = np.zeros((h, w), dtype=int)
    for box in frames:
        _draw_frame(grid, box)
    return grid


def _enumerate_candidates(grid: np.ndarray) -> list[dict]:
    frames = _find_frames(grid)
    candidates = []
    frame_at = {}
    for idx, frame in enumerate(frames):
        for cell in frame["border"]:
            frame_at[cell] = idx
    h, w = grid.shape
    starts = (
        [(0, c) for c in range(1, w - 1)]
        + [(h - 1, c) for c in range(1, w - 1)]
        + [(r, 0) for r in range(1, h - 1)]
        + [(r, w - 1) for r in range(1, h - 1)]
    )
    for start in starts:
        hit = _trace_ray(grid, start, frame_at)
        if hit is not None and hit["distance"] >= 1:
            candidates.append(hit)
    return candidates


def _paths_cross(a: dict, b: dict) -> bool:
    return bool(set(a["path"]) & set(b["path"]))


def _selection_signature(selected: list[dict], n_frames: int) -> tuple[bool, bool, bool]:
    counts = Counter(hit["frame_idx"] for hit in selected)
    has_multi = any(count >= 2 for count in counts.values())
    has_single = any(count == 1 for count in counts.values())
    has_unhit = len(counts) < n_frames
    return has_multi, has_single, has_unhit


def _selection_has_unique_winners(selected: list[dict]) -> bool:
    by_frame: dict[int, list[int]] = {}
    for hit in selected:
        by_frame.setdefault(hit["frame_idx"], []).append(int(hit["distance"]))
    for distances in by_frame.values():
        distances.sort()
        if len(distances) >= 2 and distances[0] == distances[1]:
            return False
    return True


def _pick_marker_set(candidates: list[dict], n_frames: int, rng: np.random.Generator) -> list[dict] | None:
    if len(candidates) < 3:
        return None

    if n_frames == 2:
        scenario = ["multi", rng.choice(["single", "unhit"]).item()]
    else:
        scenario_options = [
            ["multi", "single"],
            ["multi", "unhit"],
            ["multi", "single", "unhit"],
        ]
        scenario = scenario_options[int(rng.integers(0, len(scenario_options)))]

    target_count = int(rng.integers(3, min(5, len(MARKER_COLORS), len(candidates)) + 1))
    pair_options = []
    for a, b in combinations(candidates, 2):
        if a["frame_idx"] != b["frame_idx"]:
            continue
        if a["distance"] == b["distance"]:
            continue
        if _paths_cross(a, b):
            continue
        pair_options.append((a, b))
    if not pair_options:
        return None

    rng.shuffle(pair_options)
    for pair in pair_options:
        for _ in range(60):
            selected = [pair[0], pair[1]]
            used_starts = {pair[0]["start"], pair[1]["start"]}
            used_path = set(pair[0]["path"]) | set(pair[1]["path"])

            remaining = [cand for cand in candidates if cand["start"] not in used_starts]
            rng.shuffle(remaining)
            while remaining and len(selected) < target_count:
                counts = Counter(hit["frame_idx"] for hit in selected)
                need_single = "single" in scenario and not any(v == 1 for v in counts.values())
                preferred = [
                    cand
                    for cand in remaining
                    if not (set(cand["path"]) & used_path)
                    and (not need_single or counts[cand["frame_idx"]] == 0)
                ]
                fallback = [cand for cand in remaining if not (set(cand["path"]) & used_path)]
                pool = preferred or fallback
                if not pool:
                    break
                choice = pool[int(rng.integers(0, len(pool)))]
                selected.append(choice)
                used_starts.add(choice["start"])
                used_path |= set(choice["path"])
                remaining = [cand for cand in remaining if cand["start"] not in used_starts]

            if len(selected) < 3 or not _selection_has_unique_winners(selected):
                continue

            has_multi, has_single, has_unhit = _selection_signature(selected, n_frames)
            if not has_multi:
                continue
            if "single" in scenario and not has_single:
                continue
            if "unhit" in scenario and not has_unhit:
                continue
            return selected
    return None


def _install_markers(grid: np.ndarray, selected: list[dict], rng: np.random.Generator) -> np.ndarray:
    out = grid.copy()
    colors = list(rng.choice(np.array(MARKER_COLORS), size=len(selected), replace=False))
    rng.shuffle(colors)
    for hit, color in zip(selected, colors):
        r, c = hit["start"]
        out[r, c] = int(color)
    return out


def _validate_generated_input(grid: np.ndarray) -> dict:
    if not isinstance(grid, np.ndarray):
        raise AssertionError("input must be a numpy array")
    if grid.ndim != 2 or max(grid.shape) > 30:
        raise AssertionError("grid must be <= 30x30")

    info = _analyze_grid(grid)
    frames = info["frames"]
    markers = info["markers"]

    for frame in frames:
        r0, c0, r1, c1 = frame["bbox"]
        if not np.all(grid[r0 + 1 : r1, c0 + 1 : c1] == BG):
            raise AssertionError("input frame interiors must be empty")

    marker_colors = [marker["color"] for marker in markers]
    if len(marker_colors) != len(set(marker_colors)):
        raise AssertionError("marker colors must be distinct")
    if any(color not in MARKER_COLORS for color in marker_colors):
        raise AssertionError("marker colors must come from the allowed palette")

    nonzero = {(int(r), int(c)) for r, c in np.argwhere(grid != BG)}
    frame_cells = set()
    for frame in frames:
        frame_cells |= frame["border"]
    marker_cells = {marker["start"] for marker in markers}
    if nonzero != frame_cells | marker_cells:
        raise AssertionError("input may only contain frame borders and boundary markers")

    for a, b in combinations((frame["bbox"] for frame in frames), 2):
        if _boxes_too_close(a, b, pad=1):
            raise AssertionError("frames must not overlap or touch")

    path_sets = [set(hit["path"]) for hits in info["hits_by_frame"] for hit in hits]
    for i in range(len(path_sets)):
        for j in range(i + 1, len(path_sets)):
            if path_sets[i] & path_sets[j]:
                raise AssertionError("pre-hit ray paths must not cross")

    hit_counts = [len(hits) for hits in info["hits_by_frame"]]
    if max(hit_counts) < 2:
        raise AssertionError("at least one frame must be hit by multiple rays")

    return info


def _validate_output_semantics(inp: np.ndarray, out: np.ndarray, info: dict) -> None:
    expected = _transform_from_analysis(info)
    if not np.array_equal(out, expected):
        raise AssertionError("solver output does not match the expected transformation")

    for marker in info["markers"]:
        if int(out[marker["start"]]) != BG:
            raise AssertionError("markers must be erased in the output")

    for idx, frame in enumerate(info["frames"]):
        r0, c0, r1, c1 = frame["bbox"]
        border_coords = frame["border"]
        if any(int(out[r, c]) != FRAME for r, c in border_coords):
            raise AssertionError("frame borders must stay grey")
        interior = out[r0 + 1 : r1, c0 + 1 : c1]
        if idx in info["winners"]:
            winner_color = int(info["winners"][idx]["color"])
            if not np.all(interior == winner_color):
                raise AssertionError("hit frame interior must be uniformly filled")
        else:
            if not np.all(interior == BG):
                raise AssertionError("unhit frame must stay hollow")


def _gen_attempt(rng: np.random.Generator) -> np.ndarray | None:
    blank = _sample_frames(rng)
    if blank is None:
        return None

    candidates = _enumerate_candidates(blank)
    selected = _pick_marker_set(candidates, len(_find_frames(blank)), rng)
    if selected is None:
        return None

    grid = _install_markers(blank, selected, rng)
    info = _validate_generated_input(grid)
    out = main(grid)
    _validate_output_semantics(grid, out, info)
    return grid


def generate_input() -> np.ndarray:
    rng = np.random.default_rng()
    for _ in range(MAX_GEN_ATTEMPTS):
        grid = _gen_attempt(rng)
        if grid is not None:
            return grid
    raise RuntimeError("failed to generate a valid ray-frame competition grid")
