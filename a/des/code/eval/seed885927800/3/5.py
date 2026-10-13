from __future__ import annotations

from collections import deque
from typing import Iterable

import numpy as np


BG = 0
FRAME_COLOR = 5
STAMP_COLORS = (1, 2, 3, 4, 6, 7, 8, 9)
GEN_RETRY_LIMIT = 300


def _bbox(coords: np.ndarray) -> tuple[int, int, int, int]:
    rows = coords[:, 0]
    cols = coords[:, 1]
    return int(rows.min()), int(cols.min()), int(rows.max()), int(cols.max())


def _components(mask: np.ndarray) -> list[np.ndarray]:
    h, w = mask.shape
    seen = np.zeros_like(mask, dtype=bool)
    comps: list[np.ndarray] = []
    for r in range(h):
        for c in range(w):
            if not mask[r, c] or seen[r, c]:
                continue
            q = deque([(r, c)])
            seen[r, c] = True
            cells: list[tuple[int, int]] = []
            while q:
                rr, cc = q.popleft()
                cells.append((rr, cc))
                for dr, dc in ((1, 0), (-1, 0), (0, 1), (0, -1)):
                    nr = rr + dr
                    nc = cc + dc
                    if 0 <= nr < h and 0 <= nc < w and mask[nr, nc] and not seen[nr, nc]:
                        seen[nr, nc] = True
                        q.append((nr, nc))
            comps.append(np.array(cells, dtype=int))
    return comps


def _find_canvas(grid: np.ndarray) -> tuple[int, int, int, int, int]:
    g = np.asarray(grid, dtype=int)
    candidates: list[tuple[int, int, int, int, int]] = []
    for color in sorted(set(map(int, g.flatten())) - {BG}):
        coords = np.argwhere(g == color)
        if coords.size == 0:
            continue
        r0, c0, r1, c1 = _bbox(coords)
        h = int(r1 - r0 + 1)
        w = int(c1 - c0 + 1)
        if h < 5 or w < 5:
            continue
        border_count = 2 * (h + w) - 4
        if int(coords.shape[0]) != border_count:
            continue
        if not np.all(g[r0, c0 : c1 + 1] == color):
            continue
        if not np.all(g[r1, c0 : c1 + 1] == color):
            continue
        if not np.all(g[r0 : r1 + 1, c0] == color):
            continue
        if not np.all(g[r0 : r1 + 1, c1] == color):
            continue
        candidates.append((h * w, int(color), int(r0), int(c0), int(r1), int(c1)))
    assert candidates, "expected a framed canvas"
    candidates.sort()
    _area, color, r0, c0, r1, c1 = candidates[-1]
    return int(r0), int(c0), int(r1), int(c1), int(color)


def _extract_stamps(grid: np.ndarray, top_limit: int, frame_color: int) -> list[dict]:
    upper = np.asarray(grid[:top_limit, :], dtype=int)
    colors = sorted(set(map(int, upper.flatten())) - {BG, int(frame_color)})
    stamps: list[dict] = []
    for color in colors:
        coords = np.argwhere(upper == color)
        assert coords.size > 0
        r0, c0, r1, c1 = _bbox(coords)
        patch = upper[r0 : r1 + 1, c0 : c1 + 1]
        mask = patch == color
        assert bool(mask.any())
        stamps.append(
            {
                "color": int(color),
                "area": int(mask.sum()),
                "mask": mask,
                "center": (mask.shape[0] // 2, mask.shape[1] // 2),
                "bbox": (int(r0), int(c0), int(r1), int(c1)),
            }
        )
    assert len(stamps) >= 2
    return stamps


def _stamp(out: np.ndarray, color: int, mask: np.ndarray, center: tuple[int, int], point: tuple[int, int]) -> None:
    pr, pc = map(int, point)
    cr, cc = map(int, center)
    top = int(pr - cr)
    left = int(pc - cc)
    rows, cols = np.nonzero(mask)
    for rr, cc2 in zip(rows.tolist(), cols.tolist()):
        dr = int(top + rr)
        dc = int(left + cc2)
        if 0 <= dr < out.shape[0] and 0 <= dc < out.shape[1]:
            out[dr, dc] = int(color)


def _solve(grid: np.ndarray) -> np.ndarray:
    g = np.asarray(grid, dtype=int)
    assert g.ndim == 2 and g.size > 0

    r0, c0, r1, c1, frame_color = _find_canvas(g)
    interior = g[r0 + 1 : r1, c0 + 1 : c1].copy()
    stamps = _extract_stamps(g, r0, frame_color)
    stamps.sort(key=lambda item: (int(item["area"]), int(item["bbox"][1]), int(item["bbox"][0])))

    for stamp in stamps:
        color = int(stamp["color"])
        points = [tuple(map(int, rc)) for rc in np.argwhere(interior == color)]
        points.sort(key=lambda rc: (int(rc[1]), int(rc[0])))
        for point in points:
            _stamp(interior, color, stamp["mask"], stamp["center"], point)
    return interior


def main(grid: np.ndarray) -> np.ndarray:
    """
    Stamp same-color templates from the upper region onto the framed lower canvas.

    Input structure
    - Black background.
    - Several monochrome connected stamp shapes above.
    - A large framed canvas below with singleton colored points inside.

    Rule
    - Each colored point inside the canvas selects the upper stamp of the same color.
    - Process stamps from smaller area to larger area.
    - Center-stamp every selected template onto the canvas, clipping at the edges.

    Output
    - The framed canvas interior after all overlays.
    """
    return _solve(grid)


def _path_to(center: tuple[int, int], target: tuple[int, int], rng: np.random.Generator) -> set[tuple[int, int]]:
    r, c = map(int, center)
    tr, tc = map(int, target)
    cells = {(r, c)}
    while (r, c) != (tr, tc):
        choices = []
        if r != tr:
            choices.append((r + (1 if tr > r else -1), c))
        if c != tc:
            choices.append((r, c + (1 if tc > c else -1)))
        r, c = choices[int(rng.integers(len(choices)))]
        cells.add((int(r), int(c)))
    return cells


def _random_skeleton(h: int, w: int, rng: np.random.Generator) -> set[tuple[int, int]]:
    center = (h // 2, w // 2)
    targets = (
        (0, int(rng.integers(w))),
        (h - 1, int(rng.integers(w))),
        (int(rng.integers(h)), 0),
        (int(rng.integers(h)), w - 1),
    )
    cells = {center}
    for target in targets:
        cells |= _path_to(center, target, rng)
    return cells


def _grow_mask(h: int, w: int, cells: set[tuple[int, int]], target_area: int, rng: np.random.Generator) -> np.ndarray:
    cells = set(cells)
    while len(cells) < target_area:
        frontier = []
        for r, c in cells:
            for dr, dc in ((1, 0), (-1, 0), (0, 1), (0, -1)):
                nr = r + dr
                nc = c + dc
                if 0 <= nr < h and 0 <= nc < w and (nr, nc) not in cells:
                    frontier.append((nr, nc))
        assert frontier
        cells.add(frontier[int(rng.integers(len(frontier)))])
    mask = np.zeros((h, w), dtype=bool)
    for r, c in cells:
        mask[int(r), int(c)] = True
    rows, cols = np.nonzero(mask)
    assert rows.min() == 0 and rows.max() == h - 1
    assert cols.min() == 0 and cols.max() == w - 1
    return mask


def _mask_signature(mask: np.ndarray) -> tuple[int, int, bytes]:
    return int(mask.shape[0]), int(mask.shape[1]), mask.astype(np.uint8).tobytes()


def _make_stamps(count: int, rng: np.random.Generator) -> list[dict] | None:
    dims_options = ((3, 3), (3, 5), (5, 3), (5, 5))
    stamps: list[dict] = []
    used_areas: set[int] = set()
    used_shapes: set[tuple[int, int, bytes]] = set()
    colors = rng.choice(np.array(STAMP_COLORS, dtype=int), size=count, replace=False).tolist()
    for color in colors:
        made = None
        for _ in range(80):
            h, w = dims_options[int(rng.integers(len(dims_options)))]
            skeleton = _random_skeleton(h, w, rng)
            min_area = len(skeleton)
            extra_cap = 3 if h * w <= 9 else 8
            max_area = min(h * w, min_area + extra_cap)
            choices = [area for area in range(min_area, max_area + 1) if area not in used_areas]
            if not choices:
                continue
            area = int(choices[int(rng.integers(len(choices)))])
            mask = _grow_mask(h, w, skeleton, area, rng)
            sig = _mask_signature(mask)
            if sig in used_shapes:
                continue
            made = {
                "color": int(color),
                "mask": mask,
                "area": int(mask.sum()),
                "center": (h // 2, w // 2),
            }
            used_areas.add(int(mask.sum()))
            used_shapes.add(sig)
            break
        if made is None:
            return None
        stamps.append(made)
    stamps.sort(key=lambda item: int(item["area"]))
    return stamps


def _footprint(stamp: dict, point: tuple[int, int], canvas_shape: tuple[int, int]) -> tuple[set[tuple[int, int]], bool]:
    h, w = map(int, canvas_shape)
    pr, pc = map(int, point)
    cr, cc = map(int, stamp["center"])
    rows, cols = np.nonzero(stamp["mask"])
    visible: set[tuple[int, int]] = set()
    clipped = False
    for rr, cc2 in zip(rows.tolist(), cols.tolist()):
        dr = int(pr - cr + rr)
        dc = int(pc - cc + cc2)
        if 0 <= dr < h and 0 <= dc < w:
            visible.add((int(dr), int(dc)))
        else:
            clipped = True
    return visible, clipped


def _find_overlap_pair(
    stamps: list[dict],
    canvas_shape: tuple[int, int],
    occupied: set[tuple[int, int]],
    rng: np.random.Generator,
) -> tuple[tuple[int, int], tuple[int, int], int, int] | None:
    if len(stamps) < 2:
        return None
    idx_a, idx_b = sorted(rng.choice(len(stamps), size=2, replace=False).tolist())
    stamp_a = stamps[idx_a]
    stamp_b = stamps[idx_b]
    h, w = map(int, canvas_shape)
    for _ in range(250):
        pa = (int(rng.integers(h)), int(rng.integers(w)))
        if pa in occupied:
            continue
        for dr in range(-2, 3):
            for dc in range(-2, 3):
                pb = (int(pa[0] + dr), int(pa[1] + dc))
                if pb == pa or pb in occupied:
                    continue
                if not (0 <= pb[0] < h and 0 <= pb[1] < w):
                    continue
                foot_a, _ = _footprint(stamp_a, pa, canvas_shape)
                foot_b, _ = _footprint(stamp_b, pb, canvas_shape)
                if foot_a & foot_b:
                    return pa, pb, idx_a, idx_b
    return None


def _find_clipped_point(
    stamp: dict,
    canvas_shape: tuple[int, int],
    occupied: set[tuple[int, int]],
    rng: np.random.Generator,
) -> tuple[int, int] | None:
    h, w = map(int, canvas_shape)
    for _ in range(250):
        side = int(rng.integers(4))
        if side == 0:
            point = (int(rng.integers(stamp["center"][0] + 1)), int(rng.integers(w)))
        elif side == 1:
            point = (int(h - 1 - rng.integers(stamp["mask"].shape[0] - stamp["center"][0])), int(rng.integers(w)))
        elif side == 2:
            point = (int(rng.integers(h)), int(rng.integers(stamp["center"][1] + 1)))
        else:
            point = (int(rng.integers(h)), int(w - 1 - rng.integers(stamp["mask"].shape[1] - stamp["center"][1])))
        if point in occupied:
            continue
        _visible, clipped = _footprint(stamp, point, canvas_shape)
        if clipped:
            return point
    return None


def _random_free_point(
    canvas_shape: tuple[int, int],
    occupied: set[tuple[int, int]],
    rng: np.random.Generator,
) -> tuple[int, int] | None:
    h, w = map(int, canvas_shape)
    for _ in range(250):
        point = (int(rng.integers(h)), int(rng.integers(w)))
        if point not in occupied:
            return point
    return None


def _place_stamps_on_top(
    stamps: list[dict],
    grid_shape: tuple[int, int],
    frame_top: int,
    rng: np.random.Generator,
) -> list[tuple[int, int]] | None:
    grid_h, grid_w = map(int, grid_shape)
    assert frame_top < grid_h
    order = sorted(range(len(stamps)), key=lambda idx: int(stamps[idx]["mask"].size), reverse=True)
    rects: list[tuple[int, int, int, int]] = []
    positions: list[tuple[int, int] | None] = [None] * len(stamps)
    for idx in order:
        mask = stamps[idx]["mask"]
        h, w = map(int, mask.shape)
        max_r = frame_top - h - 1
        if max_r < 1 or grid_w - w - 1 < 1:
            return None
        placed = None
        for _ in range(250):
            r = int(rng.integers(1, max_r + 1))
            c = int(rng.integers(1, grid_w - w))
            box = (r - 1, c - 1, r + h, c + w)
            if any(not (box[2] < rr0 or rr1 < box[0] or box[3] < cc0 or cc1 < box[1]) for rr0, cc0, rr1, cc1 in rects):
                continue
            placed = (r, c)
            rects.append((r, c, r + h - 1, c + w - 1))
            break
        if placed is None:
            return None
        positions[idx] = placed
    return [pos for pos in positions if pos is not None]


def _build_input(rng: np.random.Generator) -> np.ndarray | None:
    stamp_count = int(rng.integers(2, 5))
    stamps = _make_stamps(stamp_count, rng)
    if stamps is None:
        return None

    canvas_h = int(rng.integers(8, 13))
    canvas_w = int(rng.integers(10, 17))
    outer_h = canvas_h + 2
    outer_w = canvas_w + 2

    top_band = int(rng.integers(8, 13))
    bottom_pad = int(rng.integers(1, 4))
    stamp_width_need = sum(int(stamp["mask"].shape[1]) for stamp in stamps) + 2 * len(stamps) + 2
    grid_w = max(outer_w + int(rng.integers(2, 7)), stamp_width_need + int(rng.integers(0, 4)))
    grid_h = top_band + outer_h + bottom_pad
    frame_top = top_band
    frame_left = int(rng.integers(1, grid_w - outer_w))

    stamp_positions = _place_stamps_on_top(stamps, (grid_h, grid_w), frame_top, rng)
    if stamp_positions is None:
        return None

    counts = {idx: 1 for idx in range(len(stamps))}
    want_overlap = bool(int(rng.integers(2)))
    want_clip = bool(int(rng.integers(2)))
    overlap_pair = None
    clip_idx = None
    if want_overlap:
        overlap_pair = tuple(sorted(rng.choice(len(stamps), size=2, replace=False).tolist()))
        counts[overlap_pair[0]] += 1
        counts[overlap_pair[1]] += 1
    if want_clip:
        clip_idx = int(rng.integers(len(stamps)))
        counts[clip_idx] += 1
    for idx in range(len(stamps)):
        if rng.random() < 0.4 and counts[idx] < 3:
            counts[idx] += 1

    occupied_points: set[tuple[int, int]] = set()
    markers: dict[int, list[tuple[int, int]]] = {idx: [] for idx in range(len(stamps))}

    if overlap_pair is not None:
        found = _find_overlap_pair(stamps, (canvas_h, canvas_w), occupied_points, rng)
        if found is None:
            return None
        pa, pb, idx_a, idx_b = found
        markers[idx_a].append(pa)
        markers[idx_b].append(pb)
        occupied_points.add(pa)
        occupied_points.add(pb)

    if clip_idx is not None:
        point = _find_clipped_point(stamps[clip_idx], (canvas_h, canvas_w), occupied_points, rng)
        if point is None:
            return None
        markers[clip_idx].append(point)
        occupied_points.add(point)

    for idx, need in counts.items():
        while len(markers[idx]) < need:
            point = _random_free_point((canvas_h, canvas_w), occupied_points, rng)
            if point is None:
                return None
            markers[idx].append(point)
            occupied_points.add(point)

    canvas = np.full((canvas_h, canvas_w), BG, dtype=int)
    for idx, pts in markers.items():
        color = int(stamps[idx]["color"])
        for r, c in pts:
            canvas[int(r), int(c)] = color

    grid = np.full((grid_h, grid_w), BG, dtype=int)
    grid[frame_top : frame_top + outer_h, frame_left : frame_left + outer_w] = BG
    grid[frame_top, frame_left : frame_left + outer_w] = FRAME_COLOR
    grid[frame_top + outer_h - 1, frame_left : frame_left + outer_w] = FRAME_COLOR
    grid[frame_top : frame_top + outer_h, frame_left] = FRAME_COLOR
    grid[frame_top : frame_top + outer_h, frame_left + outer_w - 1] = FRAME_COLOR
    grid[frame_top + 1 : frame_top + outer_h - 1, frame_left + 1 : frame_left + outer_w - 1] = canvas

    for stamp, (sr, sc) in zip(stamps, stamp_positions):
        rows, cols = np.nonzero(stamp["mask"])
        for rr, cc in zip(rows.tolist(), cols.tolist()):
            grid[int(sr + rr), int(sc + cc)] = int(stamp["color"])

    try:
        out = _solve(grid)
    except AssertionError:
        return None

    if out.shape != (canvas_h, canvas_w):
        return None
    if np.count_nonzero(out) <= np.count_nonzero(canvas):
        return None

    if want_overlap:
        overlap_ok = False
        for i in range(len(stamps)):
            for j in range(i + 1, len(stamps)):
                for pa in markers[i]:
                    for pb in markers[j]:
                        foot_a, _ = _footprint(stamps[i], pa, (canvas_h, canvas_w))
                        foot_b, _ = _footprint(stamps[j], pb, (canvas_h, canvas_w))
                        if foot_a & foot_b:
                            overlap_ok = True
                            break
                    if overlap_ok:
                        break
                if overlap_ok:
                    break
            if overlap_ok:
                break
        if not overlap_ok:
            return None

    if want_clip:
        clip_ok = False
        for idx, pts in markers.items():
            for point in pts:
                _visible, clipped = _footprint(stamps[idx], point, (canvas_h, canvas_w))
                if clipped:
                    clip_ok = True
                    break
            if clip_ok:
                break
        if not clip_ok:
            return None

    return grid


def generate_input() -> np.ndarray:
    rng = np.random.default_rng()
    for _ in range(GEN_RETRY_LIMIT):
        grid = _build_input(rng)
        if grid is not None:
            return grid
    raise RuntimeError("failed to generate a valid input")
