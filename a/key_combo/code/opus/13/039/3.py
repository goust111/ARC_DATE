from __future__ import annotations

from collections import Counter, deque
import hashlib

import numpy as np


BG = 0
GREY = 5
CORNERS = ("TL", "TR", "BR", "BL")
NEXT_CORNER = {"TL": "TR", "TR": "BR", "BR": "BL", "BL": "TL"}
CORNER_COLORS = (1, 2, 3, 4)
MAX_GEN_ATTEMPTS = 800


_CORNER_TEMPLATES = (
    {(0, 0), (0, 1), (1, 0), (1, 1)},
    {(0, 0), (0, 1), (0, 2), (1, 0), (1, 1)},
    {(0, 0), (1, 0), (2, 0), (1, 1), (2, 1)},
    {(0, 0), (0, 1), (1, 0), (2, 0), (2, 1)},
    {(0, 0), (0, 1), (0, 2), (1, 0), (2, 0), (2, 1)},
    {(0, 0), (0, 1), (1, 0), (1, 1), (2, 0), (2, 1)},
    {(0, 0), (0, 1), (0, 2), (1, 0), (1, 1), (2, 0), (2, 1)},
    {(0, 0), (0, 1), (0, 2), (1, 0), (1, 2), (2, 0), (2, 1), (2, 2)},
)


def _get_bg(grid: np.ndarray) -> int:
    counts = Counter(map(int, grid.ravel()))
    return int(counts.most_common(1)[0][0])


def _neighbors(r: int, c: int, connectivity: int) -> tuple[tuple[int, int], ...]:
    if connectivity == 4:
        return ((r - 1, c), (r + 1, c), (r, c - 1), (r, c + 1))
    return tuple(
        (r + dr, c + dc)
        for dr in (-1, 0, 1)
        for dc in (-1, 0, 1)
        if (dr, dc) != (0, 0)
    )


def _components_of_color(grid: np.ndarray, color: int, connectivity: int) -> list[np.ndarray]:
    h, w = grid.shape
    seen = np.zeros((h, w), dtype=bool)
    comps: list[np.ndarray] = []
    for r in range(h):
        for c in range(w):
            if seen[r, c] or int(grid[r, c]) != color:
                continue
            q: deque[tuple[int, int]] = deque([(r, c)])
            seen[r, c] = True
            cells: list[tuple[int, int]] = []
            while q:
                rr, cc = q.popleft()
                cells.append((rr, cc))
                for nr, nc in _neighbors(rr, cc, connectivity):
                    if 0 <= nr < h and 0 <= nc < w and not seen[nr, nc] and int(grid[nr, nc]) == color:
                        seen[nr, nc] = True
                        q.append((nr, nc))
            comps.append(np.array(cells, dtype=int))
    return comps


def _bbox_coords(coords: np.ndarray) -> tuple[int, int, int, int]:
    r0, c0 = coords.min(axis=0)
    r1, c1 = coords.max(axis=0)
    return int(r0), int(c0), int(r1), int(c1)


def _corner_from_bbox(box: tuple[int, int, int, int], h: int, w: int) -> str | None:
    r0, c0, r1, c1 = box
    if r0 == 0 and c0 == 0:
        return "TL"
    if r0 == 0 and c1 == w - 1:
        return "TR"
    if r1 == h - 1 and c1 == w - 1:
        return "BR"
    if r1 == h - 1 and c0 == 0:
        return "BL"
    return None


def _corner_anchor(corner: str, h: int, w: int, obj_h: int, obj_w: int) -> tuple[int, int]:
    if corner == "TL":
        return 0, 0
    if corner == "TR":
        return 0, w - obj_w
    if corner == "BR":
        return h - obj_h, w - obj_w
    assert corner == "BL"
    return h - obj_h, 0


def _normalize(coords: set[tuple[int, int]] | list[tuple[int, int]]) -> list[tuple[int, int]]:
    rows = [r for r, _ in coords]
    cols = [c for _, c in coords]
    r0 = min(rows)
    c0 = min(cols)
    return sorted((r - r0, c - c0) for r, c in coords)


def _is_diagonal_segment(coords: np.ndarray) -> bool:
    cells = [tuple(map(int, rc)) for rc in coords]
    if len(cells) not in (2, 3):
        return False
    norm = _normalize(cells)
    length = len(cells)
    return norm == [(i, i) for i in range(length)] or norm == [(i, length - 1 - i) for i in range(length)]


def _segment_endpoints(coords: np.ndarray) -> list[tuple[int, int]]:
    cells = [tuple(map(int, rc)) for rc in coords]
    pts = set(cells)
    ends = []
    for r, c in cells:
        deg = sum((nr, nc) in pts for nr, nc in _neighbors(r, c, 8))
        if deg == 1:
            ends.append((r, c))
    if len(cells) == 2 and len(ends) == 2:
        return sorted(ends)
    assert len(ends) == 2, "expected a simple diagonal segment with two endpoints"
    return sorted(ends)


def _ray_direction_from_center(r: int, c: int, shape: tuple[int, int]) -> tuple[int, int]:
    h, w = shape
    dr = -1 if 2 * r < h - 1 else 1
    dc = -1 if 2 * c < w - 1 else 1
    return dr, dc


def _ray_path(endpoint: tuple[int, int], obstacle: np.ndarray) -> list[tuple[int, int]]:
    h, w = obstacle.shape
    r, c = endpoint
    dr, dc = _ray_direction_from_center(r, c, obstacle.shape)
    path: list[tuple[int, int]] = []
    while True:
        r += dr
        c += dc
        if not (0 <= r < h and 0 <= c < w):
            break
        if int(obstacle[r, c]) != BG:
            break
        path.append((r, c))
    return path


def _find_corner_objects(grid: np.ndarray, bg: int) -> dict[str, dict]:
    h, w = grid.shape
    by_corner: dict[str, dict] = {}
    for color in sorted(set(map(int, np.unique(grid))) - {bg, GREY}):
        for coords in _components_of_color(grid, color, connectivity=4):
            box = _bbox_coords(coords)
            corner = _corner_from_bbox(box, h, w)
            if corner is None:
                continue
            assert corner not in by_corner, "expected at most one corner object per corner"
            by_corner[corner] = {"color": int(color), "coords": coords, "bbox": box}
    return by_corner


def _find_segments(grid: np.ndarray) -> list[np.ndarray]:
    segments = _components_of_color(grid, GREY, connectivity=8)
    assert 1 <= len(segments) <= 2, "expected 1 or 2 grey diagonal segments"
    for coords in segments:
        assert _is_diagonal_segment(coords), "grey objects must be short diagonal segments"
    return segments


def _trace_all_rays(grid: np.ndarray, segments: list[np.ndarray]) -> tuple[set[tuple[int, int]], list[list[tuple[int, int]]]]:
    obstacle = np.asarray(grid, dtype=int)
    ray_pixels: set[tuple[int, int]] = set()
    all_paths: list[list[tuple[int, int]]] = []
    for seg in segments:
        for endpoint in _segment_endpoints(seg):
            path = _ray_path(endpoint, obstacle)
            all_paths.append(path)
            ray_pixels.update(path)
    return ray_pixels, all_paths


def _blocked_bands(ray_pixels: set[tuple[int, int]], shape: tuple[int, int]) -> set[str]:
    h, w = shape
    blocked: set[str] = set()
    for r, c in ray_pixels:
        if r == 0:
            blocked.add("TL")
        if c == w - 1:
            blocked.add("TR")
        if r == h - 1:
            blocked.add("BR")
        if c == 0:
            blocked.add("BL")
    return blocked


def _solve(
    grid: np.ndarray,
    *,
    honor_barriers: bool = True,
    do_copies: bool = True,
) -> np.ndarray:
    g = np.asarray(grid, dtype=int)
    assert g.ndim == 2, "expected a 2D grid"
    bg = _get_bg(g)
    corner_objects = _find_corner_objects(g, bg)
    assert 2 <= len(corner_objects) <= 4, "expected 2 to 4 corner objects"
    segments = _find_segments(g)

    ray_pixels, _ = _trace_all_rays(g, segments)
    blocked = _blocked_bands(ray_pixels, g.shape) if honor_barriers else set()

    out = g.copy()
    for r, c in ray_pixels:
        out[r, c] = GREY

    if not do_copies:
        return out

    h, w = g.shape
    for src_corner in CORNERS:
        comp = corner_objects.get(src_corner)
        if comp is None:
            continue
        if src_corner in blocked:
            continue
        dst_corner = NEXT_CORNER[src_corner]
        r0, c0, r1, c1 = comp["bbox"]
        obj_h = int(r1 - r0 + 1)
        obj_w = int(c1 - c0 + 1)
        dst_r0, dst_c0 = _corner_anchor(dst_corner, h, w, obj_h, obj_w)
        color = int(comp["color"])
        for rr, cc in comp["coords"]:
            out[int(dst_r0 + rr - r0), int(dst_c0 + cc - c0)] = color
    return out


def main(grid: np.ndarray) -> np.ndarray:
    """
    Extend grey endpoint rays to form edge barriers, then copy each corner object clockwise only when its edge path is clear.

    Input structure:
    - The background is black.
    - Two to four monochromatic 4-connected objects sit in true grid corners, each touching the two borders of that corner.
    - One or two short grey diagonal segments lie near the border but away from the corner objects.

    Rule:
    - Keep the input unchanged as the starting state.
    - From each endpoint of every grey diagonal segment, emit one diagonal ray in the direction away from the grid center; paint every background cell on that ray grey until the next step would hit a non-background cell or leave the grid.
    - The four clockwise edge-path bands are: top row for TL->TR, right column for TR->BR, bottom row for BR->BL, and left column for BL->TL.
    - Process corner objects in TL, TR, BR, BL order. If a source corner's band contains any grey ray pixel, do not copy that object. Otherwise stamp the source object's mask and color into the next clockwise corner by aligning its bbox to the destination corner, overwriting whatever is there.

    Key insight:
    - The grey rays are not decorative output marks: they are logical gates, because only the corner-to-corner copies whose border band stays ray-free are allowed to happen.
    """
    return _solve(grid, honor_barriers=True, do_copies=True)


def _transform_template(template: set[tuple[int, int]], corner: str) -> tuple[list[tuple[int, int]], int, int]:
    h = max(r for r, _ in template) + 1
    w = max(c for _, c in template) + 1
    coords = list(template)
    if corner in ("TR", "BR"):
        coords = [(r, w - 1 - c) for r, c in coords]
    if corner in ("BL", "BR"):
        coords = [(h - 1 - r, c) for r, c in coords]
    return _normalize(coords), h, w


def _stamp_cells(grid: np.ndarray, cells: list[tuple[int, int]], color: int) -> None:
    for r, c in cells:
        grid[r, c] = color


def _corner_cells(corner: str, template: set[tuple[int, int]], shape: tuple[int, int]) -> list[tuple[int, int]]:
    h, w = shape
    coords, obj_h, obj_w = _transform_template(template, corner)
    r0, c0 = _corner_anchor(corner, h, w, obj_h, obj_w)
    return [(r0 + r, c0 + c) for r, c in coords]


def _corner_zone(shape: tuple[int, int], pad: int = 4) -> set[tuple[int, int]]:
    h, w = shape
    cells: set[tuple[int, int]] = set()
    for r in range(h):
        for c in range(w):
            if (r < pad and c < pad) or (r < pad and c >= w - pad) or (r >= h - pad and c >= w - pad) or (r >= h - pad and c < pad):
                cells.add((r, c))
    return cells


def _segment_cells(r0: int, c0: int, length: int, slope: str) -> list[tuple[int, int]]:
    if slope == "\\":
        return [(r0 + i, c0 + i) for i in range(length)]
    return [(r0 + i, c0 + length - 1 - i) for i in range(length)]


def _edge_distance(shape: tuple[int, int], cell: tuple[int, int]) -> int:
    h, w = shape
    r, c = cell
    return min(r, c, h - 1 - r, w - 1 - c)


def _band_cells_by_corner(shape: tuple[int, int]) -> dict[str, list[tuple[int, int]]]:
    h, w = shape
    return {
        "TL": [(0, c) for c in range(w)],
        "TR": [(r, w - 1) for r in range(h)],
        "BR": [(h - 1, c) for c in range(w)],
        "BL": [(r, 0) for r in range(h)],
    }


def _enumerate_segment_candidates(base: np.ndarray) -> list[dict]:
    h, w = base.shape
    forbidden = _corner_zone(base.shape, pad=4)
    candidates: list[dict] = []
    for length in (2, 3):
        for slope in ("\\", "/"):
            row_max = h - length - 1
            col_values = range(1, w - length)
            for r0 in range(1, row_max + 1):
                for c0 in col_values:
                    cells = _segment_cells(r0, c0, length, slope)
                    if any(cell in forbidden for cell in cells):
                        continue
                    if any(_edge_distance(base.shape, cell) > 3 for cell in cells):
                        continue
                    if any(int(base[r, c]) != BG for r, c in cells):
                        continue
                    obstacle = base.copy()
                    _stamp_cells(obstacle, cells, GREY)
                    seg = np.array(cells, dtype=int)
                    endpoints = _segment_endpoints(seg)
                    paths = [_ray_path(endpoint, obstacle) for endpoint in endpoints]
                    if any(len(path) == 0 for path in paths):
                        continue
                    ray_pixels = set(cell for path in paths for cell in path)
                    bands = _blocked_bands(ray_pixels, base.shape)
                    if not bands or len(bands) == 4:
                        continue
                    bad_band_hit = False
                    for r, c in ray_pixels:
                        if r == 0 and not (2 <= c <= w - 3):
                            bad_band_hit = True
                        if r == h - 1 and not (2 <= c <= w - 3):
                            bad_band_hit = True
                        if c == 0 and not (2 <= r <= h - 3):
                            bad_band_hit = True
                        if c == w - 1 and not (2 <= r <= h - 3):
                            bad_band_hit = True
                    if bad_band_hit:
                        continue
                    candidates.append({"cells": cells, "bands": bands, "ray_pixels": ray_pixels})
    return candidates


def _segments_separated(a: list[tuple[int, int]], b: list[tuple[int, int]]) -> bool:
    for r1, c1 in a:
        for r2, c2 in b:
            if max(abs(r1 - r2), abs(c1 - c2)) <= 1:
                return False
    return True


def _combo_rays(base: np.ndarray, segments: list[list[tuple[int, int]]]) -> tuple[set[tuple[int, int]], list[list[tuple[int, int]]]]:
    obstacle = base.copy()
    for cells in segments:
        _stamp_cells(obstacle, cells, GREY)
    ray_pixels: set[tuple[int, int]] = set()
    paths: list[list[tuple[int, int]]] = []
    for cells in segments:
        seg = np.array(cells, dtype=int)
        for endpoint in _segment_endpoints(seg):
            path = _ray_path(endpoint, obstacle)
            paths.append(path)
            ray_pixels.update(path)
    return ray_pixels, paths


def _corner_count_after_copy(source_corners: set[str], blocked: set[str]) -> int:
    return sum(corner not in blocked for corner in source_corners)


def _remove_segments(grid: np.ndarray) -> np.ndarray:
    out = np.asarray(grid, dtype=int).copy()
    out[out == GREY] = BG
    return out


def _copy_only_without_segments(grid: np.ndarray) -> np.ndarray:
    g = _remove_segments(grid)
    bg = _get_bg(g)
    corner_objects = _find_corner_objects(g, bg)
    out = g.copy()
    h, w = g.shape
    for src_corner in CORNERS:
        comp = corner_objects.get(src_corner)
        if comp is None:
            continue
        dst_corner = NEXT_CORNER[src_corner]
        r0, c0, r1, c1 = comp["bbox"]
        obj_h = int(r1 - r0 + 1)
        obj_w = int(c1 - c0 + 1)
        dst_r0, dst_c0 = _corner_anchor(dst_corner, h, w, obj_h, obj_w)
        color = int(comp["color"])
        for rr, cc in comp["coords"]:
            out[int(dst_r0 + rr - r0), int(dst_c0 + cc - c0)] = color
    return out


def _valid_input(grid: np.ndarray) -> bool:
    try:
        g = np.asarray(grid, dtype=int)
        if g.ndim != 2 or max(g.shape) > 30:
            return False
        bg = _get_bg(g)
        if bg != BG:
            return False
        corner_objects = _find_corner_objects(g, bg)
        if not (2 <= len(corner_objects) <= 4):
            return False
        if len({int(comp["color"]) for comp in corner_objects.values()}) != len(corner_objects):
            return False
        segments = _find_segments(g)
        if not (1 <= len(segments) <= 2):
            return False
        ray_pixels, paths = _trace_all_rays(g, segments)
        if not ray_pixels or any(len(path) == 0 for path in paths):
            return False
        blocked = _blocked_bands(ray_pixels, g.shape)
        occupied = set(corner_objects)
        if not (occupied & blocked):
            return False
        if not (occupied - blocked):
            return False
        if _corner_count_after_copy(occupied, blocked) == 0:
            return False
        out = _solve(g, honor_barriers=True, do_copies=True)
        if np.array_equal(out, g):
            return False
        if np.array_equal(out, _solve(g, honor_barriers=False, do_copies=True)):
            return False
        if np.array_equal(out, _solve(g, honor_barriers=True, do_copies=False)):
            return False
        if np.array_equal(out, _copy_only_without_segments(g)):
            return False
        return True
    except Exception:
        return False


def _attempt_generate(rng: np.random.Generator) -> np.ndarray | None:
    h = int(rng.integers(13, 19))
    w = int(rng.integers(13, 19))
    grid = np.zeros((h, w), dtype=int)

    corner_count = int(rng.integers(2, 5))
    chosen_corners = list(rng.choice(np.array(CORNERS), size=corner_count, replace=False))
    chosen_colors = list(rng.choice(np.array(CORNER_COLORS), size=corner_count, replace=False))
    rng.shuffle(chosen_corners)

    for corner, color in zip(chosen_corners, chosen_colors):
        template = _CORNER_TEMPLATES[int(rng.integers(len(_CORNER_TEMPLATES)))]
        cells = _corner_cells(corner, template, grid.shape)
        _stamp_cells(grid, cells, int(color))

    candidates = _enumerate_segment_candidates(grid)
    if not candidates:
        return None

    rng.shuffle(candidates)
    desired_counts = [1, 2] if float(rng.random()) < 0.5 else [2, 1]
    occupied = set(chosen_corners)
    chosen_segments: list[list[tuple[int, int]]] | None = None

    for count in desired_counts:
        if count == 1:
            for cand in candidates:
                blocked = set(cand["bands"])
                if not (occupied & blocked) or not (occupied - blocked):
                    continue
                chosen_segments = [cand["cells"]]
                break
        else:
            for i, first in enumerate(candidates):
                for second in candidates[i + 1 :]:
                    if not _segments_separated(first["cells"], second["cells"]):
                        continue
                    ray_pixels, paths = _combo_rays(grid, [first["cells"], second["cells"]])
                    if not ray_pixels or any(len(path) == 0 for path in paths):
                        continue
                    if len(ray_pixels) != sum(len(path) for path in paths):
                        continue
                    blocked = _blocked_bands(ray_pixels, grid.shape)
                    if not (occupied & blocked) or not (occupied - blocked):
                        continue
                    chosen_segments = [first["cells"], second["cells"]]
                    break
                if chosen_segments is not None:
                    break
        if chosen_segments is not None:
            break

    if chosen_segments is None:
        return None

    for cells in chosen_segments:
        _stamp_cells(grid, cells, GREY)

    if not _valid_input(grid):
        return None
    return grid


def generate_input() -> np.ndarray:
    rng = np.random.default_rng()
    for _ in range(MAX_GEN_ATTEMPTS):
        grid = _attempt_generate(rng)
        if grid is not None:
            return grid
    raise RuntimeError("failed to generate a valid input for puzzle 039/3")


if __name__ == "__main__":
    sigs = set()
    outs = set()
    for _ in range(20):
        inp = generate_input()
        out = main(inp)
        sigs.add(hashlib.md5(inp.tobytes() + str(inp.shape).encode()).hexdigest())
        outs.add(hashlib.md5(out.tobytes() + str(out.shape).encode()).hexdigest())
    print("input_variants", len(sigs))
    print("output_variants", len(outs))
