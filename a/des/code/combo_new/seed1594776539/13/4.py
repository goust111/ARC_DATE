from __future__ import annotations

from collections import defaultdict, deque
import random

import numpy as np


BG = 0
PIPE = 5
FRAME = 7
SOURCE_COLORS = (1, 2, 3, 4, 6, 8, 9)
SIDES = ("top", "bottom", "left", "right")
MAX_GENERATION_ATTEMPTS = 300


def main(grid: np.ndarray) -> np.ndarray:
    """
    Route each source color through its grey pipe into the contacted orange container.

    Input structure:
    - The background is black.
    - Several small monochrome 2x2 or 3x3 filled squares act as color sources.
    - One or more orange one-cell-thick rectangular frames act as empty containers.
    - Grey one-cell-wide orthogonal pipe polylines connect each source to one container edge; some pipes may cross on shared grey pixels.

    Rule:
    - Identify every source square, every orange rectangular frame, and the grey pipe network.
    - For each source, trace its pipe from the unique source-touching grey pixel to the contacted container edge, continuing straight through crossings and turning only at true corners.
    - A container with one incoming pipe is filled entirely with that source color.
    - A container with multiple incoming pipes is partitioned by the sides where the pipes arrive: top and bottom entries fill upper and lower halves, left and right entries fill left and right halves, and multiple entries on the same side split that side's region according to their contact positions.

    Key insight:
    - The grey network must be read as geometric paths rather than a single connected blob: at crossings the intended pipe keeps its direction, and the container fill is determined by entry side and edge position, not by color mixing.
    """
    g = np.asarray(grid, dtype=int)
    assert g.ndim == 2

    bg = _background_color(g)
    frame_color, containers = _find_containers(g, bg)
    pipe_color = _find_pipe_color(g, bg, frame_color)
    sources = _find_sources(g, bg, frame_color, pipe_color)

    pipe_set = {tuple(map(int, coord)) for coord in np.argwhere(g == pipe_color)}
    container_hits: dict[int, list[tuple[str, int, int]]] = defaultdict(list)

    for source in sources:
        start, anchor = _source_pipe_start(source, pipe_set, g.shape)
        hit = _trace_to_container(start, anchor, pipe_set, containers)
        container_hits[hit[0]].append((hit[1], hit[2], source["color"]))

    out = g.copy()
    for idx, container in enumerate(containers):
        entries = container_hits.get(idx, [])
        if entries:
            _fill_container(out, container["bbox"], entries)

    return out


def _background_color(grid: np.ndarray) -> int:
    values, counts = np.unique(grid, return_counts=True)
    return int(values[np.argmax(counts)])


def _neighbors4(r: int, c: int, h: int, w: int) -> list[tuple[int, int]]:
    out = []
    if r > 0:
        out.append((r - 1, c))
    if r + 1 < h:
        out.append((r + 1, c))
    if c > 0:
        out.append((r, c - 1))
    if c + 1 < w:
        out.append((r, c + 1))
    return out


def _components(mask: np.ndarray) -> list[list[tuple[int, int]]]:
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
                for nr, nc in _neighbors4(rr, cc, h, w):
                    if mask[nr, nc] and not seen[nr, nc]:
                        seen[nr, nc] = True
                        q.append((nr, nc))
            comps.append(comp)
    return comps


def _bbox(coords: list[tuple[int, int]]) -> tuple[int, int, int, int]:
    rows = [r for r, _ in coords]
    cols = [c for _, c in coords]
    return min(rows), min(cols), max(rows), max(cols)


def _is_frame_component(mask: np.ndarray, coords: list[tuple[int, int]]) -> bool:
    r0, c0, r1, c1 = _bbox(coords)
    h = r1 - r0 + 1
    w = c1 - c0 + 1
    if h < 3 or w < 3:
        return False
    if len(coords) != 2 * h + 2 * w - 4:
        return False
    sub = mask[r0 : r1 + 1, c0 : c1 + 1]
    if not np.all(sub[0, :]) or not np.all(sub[-1, :]) or not np.all(sub[:, 0]) or not np.all(sub[:, -1]):
        return False
    return not np.any(sub[1:-1, 1:-1])


def _is_source_component(coords: list[tuple[int, int]]) -> bool:
    r0, c0, r1, c1 = _bbox(coords)
    h = r1 - r0 + 1
    w = c1 - c0 + 1
    return h == w and h in (2, 3) and len(coords) == h * w


def _find_containers(grid: np.ndarray, bg: int) -> tuple[int, list[dict[str, tuple[int, int, int, int]]]]:
    candidates = []
    for color in sorted(int(v) for v in np.unique(grid) if int(v) != bg):
        mask = grid == color
        comps = _components(mask)
        frame_boxes = []
        for comp in comps:
            if _is_frame_component(mask, comp):
                frame_boxes.append(_bbox(comp))
        if frame_boxes and len(frame_boxes) == len(comps):
            candidates.append((color, sorted(frame_boxes)))
    assert len(candidates) == 1, "expected a unique frame color"
    frame_color, boxes = candidates[0]
    containers = [{"bbox": box} for box in boxes]
    return frame_color, containers


def _find_pipe_color(grid: np.ndarray, bg: int, frame_color: int) -> int:
    remaining = [int(v) for v in np.unique(grid) if int(v) not in {bg, frame_color}]
    line_like = []
    for color in remaining:
        comps = _components(grid == color)
        if comps and not all(_is_source_component(comp) for comp in comps):
            line_like.append(color)
    assert len(line_like) == 1, "expected a unique pipe color"
    return line_like[0]


def _find_sources(grid: np.ndarray, bg: int, frame_color: int, pipe_color: int) -> list[dict[str, object]]:
    sources = []
    for color in sorted(int(v) for v in np.unique(grid) if int(v) not in {bg, frame_color, pipe_color}):
        comps = _components(grid == color)
        assert len(comps) == 1 and _is_source_component(comps[0]), "every source must be one small filled square"
        coords = comps[0]
        sources.append({"color": color, "coords": coords, "bbox": _bbox(coords)})
    assert sources, "expected at least one source"
    return sources


def _source_pipe_start(source: dict[str, object], pipe_set: set[tuple[int, int]], shape: tuple[int, int]) -> tuple[tuple[int, int], tuple[int, int]]:
    h, w = shape
    coords = source["coords"]
    coord_set = set(coords)
    starts = []
    for r, c in coords:
        for nr, nc in _neighbors4(r, c, h, w):
            if (nr, nc) in pipe_set:
                starts.append(((nr, nc), (r, c)))
    unique = {(start, anchor) for start, anchor in starts}
    assert len({start for start, _ in unique}) == 1, "each source must touch exactly one pipe cell"
    start, anchor = next(iter(unique))
    assert anchor in coord_set
    return start, anchor


def _container_contact(cell: tuple[int, int], containers: list[dict[str, tuple[int, int, int, int]]]) -> tuple[int, str, int] | None:
    r, c = cell
    for idx, container in enumerate(containers):
        r0, c0, r1, c1 = container["bbox"]
        if r == r0 - 1 and c0 + 1 <= c <= c1 - 1:
            return idx, "top", c
        if r == r1 + 1 and c0 + 1 <= c <= c1 - 1:
            return idx, "bottom", c
        if c == c0 - 1 and r0 + 1 <= r <= r1 - 1:
            return idx, "left", r
        if c == c1 + 1 and r0 + 1 <= r <= r1 - 1:
            return idx, "right", r
    return None


def _trace_to_container(
    start: tuple[int, int],
    anchor: tuple[int, int],
    pipe_set: set[tuple[int, int]],
    containers: list[dict[str, tuple[int, int, int, int]]],
) -> tuple[int, str, int]:
    cur = start
    prev = anchor
    seen = set()
    for _ in range(len(pipe_set) + 5):
        hit = _container_contact(cur, containers)
        if hit is not None:
            return hit
        if (cur, prev) in seen:
            break
        seen.add((cur, prev))
        dr = cur[0] - prev[0]
        dc = cur[1] - prev[1]
        nxts = []
        for cand in ((cur[0] - 1, cur[1]), (cur[0] + 1, cur[1]), (cur[0], cur[1] - 1), (cur[0], cur[1] + 1)):
            if cand in pipe_set and cand != prev:
                nxts.append(cand)
        assert nxts, "pipe terminated before reaching a container"
        straight = (cur[0] + dr, cur[1] + dc)
        if straight in nxts:
            nxt = straight
        else:
            assert len(nxts) == 1, "ambiguous non-geometric pipe junction"
            nxt = nxts[0]
        prev, cur = cur, nxt
    raise AssertionError("failed to trace pipe to a container")


def _strip_starts(length: int, positions: list[int]) -> list[int]:
    starts = [0]
    for a, b in zip(positions, positions[1:]):
        starts.append((a + b + 1) // 2)
    starts.append(length)
    return starts


def _fill_column_strips(region: np.ndarray, entries: list[tuple[str, int, int]], c0: int) -> None:
    width = region.shape[1]
    items = sorted((pos - (c0 + 1), color) for _, pos, color in entries)
    xs = [x for x, _ in items]
    starts = _strip_starts(width, xs)
    for i, (_, color) in enumerate(items):
        region[:, starts[i] : starts[i + 1]] = color


def _fill_row_strips(region: np.ndarray, entries: list[tuple[str, int, int]], r0: int) -> None:
    height = region.shape[0]
    items = sorted((pos - (r0 + 1), color) for _, pos, color in entries)
    ys = [y for y, _ in items]
    starts = _strip_starts(height, ys)
    for i, (_, color) in enumerate(items):
        region[starts[i] : starts[i + 1], :] = color


def _fill_nearest(region: np.ndarray, entries: list[tuple[str, int, int]], bbox: tuple[int, int, int, int]) -> None:
    r0, c0, _, _ = bbox
    h, w = region.shape
    converted = []
    for side, pos, color in entries:
        if side in {"top", "bottom"}:
            converted.append((side, pos - (c0 + 1), color))
        else:
            converted.append((side, pos - (r0 + 1), color))
    for rr in range(h):
        for cc in range(w):
            best_score = None
            best_color = None
            for side, pos, color in converted:
                if side == "top":
                    score = rr + abs(cc - pos)
                elif side == "bottom":
                    score = (h - 1 - rr) + abs(cc - pos)
                elif side == "left":
                    score = cc + abs(rr - pos)
                else:
                    score = (w - 1 - cc) + abs(rr - pos)
                key = (score, color)
                if best_score is None or key < best_score:
                    best_score = key
                    best_color = color
            region[rr, cc] = best_color


def _fill_container(grid: np.ndarray, bbox: tuple[int, int, int, int], entries: list[tuple[str, int, int]]) -> None:
    r0, c0, r1, c1 = bbox
    region = grid[r0 + 1 : r1, c0 + 1 : c1]
    if len(entries) == 1:
        region[:, :] = entries[0][2]
        return

    groups: dict[str, list[tuple[str, int, int]]] = defaultdict(list)
    for entry in entries:
        groups[entry[0]].append(entry)

    sides = set(groups)
    if sides <= {"top", "bottom"}:
        if groups["top"] and groups["bottom"]:
            split = (region.shape[0] + 1) // 2
            _fill_column_strips(region[:split, :], groups["top"], c0)
            _fill_column_strips(region[split:, :], groups["bottom"], c0)
        elif groups["top"]:
            _fill_column_strips(region, groups["top"], c0)
        else:
            _fill_column_strips(region, groups["bottom"], c0)
        return

    if sides <= {"left", "right"}:
        if groups["left"] and groups["right"]:
            split = (region.shape[1] + 1) // 2
            _fill_row_strips(region[:, :split], groups["left"], r0)
            _fill_row_strips(region[:, split:], groups["right"], r0)
        elif groups["left"]:
            _fill_row_strips(region, groups["left"], r0)
        else:
            _fill_row_strips(region, groups["right"], r0)
        return

    _fill_nearest(region, entries, bbox)


def _draw_frame(grid: np.ndarray, bbox: tuple[int, int, int, int], color: int) -> None:
    r0, c0, r1, c1 = bbox
    grid[r0, c0 : c1 + 1] = color
    grid[r1, c0 : c1 + 1] = color
    grid[r0 : r1 + 1, c0] = color
    grid[r0 : r1 + 1, c1] = color


def _touches_non_bg(grid: np.ndarray, bbox: tuple[int, int, int, int], pad: int = 0) -> bool:
    h, w = grid.shape
    r0, c0, r1, c1 = bbox
    rr0 = max(0, r0 - pad)
    cc0 = max(0, c0 - pad)
    rr1 = min(h - 1, r1 + pad)
    cc1 = min(w - 1, c1 + pad)
    return np.any(grid[rr0 : rr1 + 1, cc0 : cc1 + 1] != BG)


def _boxes_close(a: tuple[int, int, int, int], b: tuple[int, int, int, int], pad: int) -> bool:
    ar0, ac0, ar1, ac1 = a
    br0, bc0, br1, bc1 = b
    return not (ar1 + pad < br0 or br1 + pad < ar0 or ac1 + pad < bc0 or bc1 + pad < ac0)


def _entry_border_and_pipe(bbox: tuple[int, int, int, int], side: str, pos: int) -> tuple[tuple[int, int], tuple[int, int]]:
    r0, c0, r1, c1 = bbox
    if side == "top":
        return (r0, pos), (r0 - 1, pos)
    if side == "bottom":
        return (r1, pos), (r1 + 1, pos)
    if side == "left":
        return (pos, c0), (pos, c0 - 1)
    return (pos, c1), (pos, c1 + 1)


def _segment(a: tuple[int, int], b: tuple[int, int]) -> list[tuple[int, int]]:
    if a == b:
        return [a]
    if a[0] == b[0]:
        step = 1 if b[1] > a[1] else -1
        return [(a[0], c) for c in range(a[1], b[1] + step, step)]
    if a[1] == b[1]:
        step = 1 if b[0] > a[0] else -1
        return [(r, a[1]) for r in range(a[0], b[0] + step, step)]
    raise ValueError("segment endpoints must be axis-aligned")


def _polyline(points: list[tuple[int, int]]) -> list[tuple[int, int]]:
    path: list[tuple[int, int]] = []
    for a, b in zip(points, points[1:]):
        seg = _segment(a, b)
        if path:
            seg = seg[1:]
        path.extend(seg)
    return path


def _path_cell_orientation(path: list[tuple[int, int]], idx: int) -> str | None:
    prev = path[idx - 1] if idx > 0 else None
    nxt = path[idx + 1] if idx + 1 < len(path) else None
    dirs = []
    if prev is not None:
        dirs.append((path[idx][0] - prev[0], path[idx][1] - prev[1]))
    if nxt is not None:
        dirs.append((nxt[0] - path[idx][0], nxt[1] - path[idx][1]))
    if len(dirs) != 2:
        return None
    if dirs[0][0] == dirs[1][0] == 0 and dirs[0][1] == -dirs[1][1]:
        return "h"
    if dirs[0][1] == dirs[1][1] == 0 and dirs[0][0] == -dirs[1][0]:
        return "v"
    return None


def _existing_pipe_orientation(pipe_set: set[tuple[int, int]], cell: tuple[int, int]) -> str | None:
    r, c = cell
    neighbors = [n for n in ((r - 1, c), (r + 1, c), (r, c - 1), (r, c + 1)) if n in pipe_set]
    if len(neighbors) != 2:
        return None
    if neighbors[0][0] == neighbors[1][0] == r:
        return "h"
    if neighbors[0][1] == neighbors[1][1] == c:
        return "v"
    return None


def _validate_path(
    grid: np.ndarray,
    path: list[tuple[int, int]],
    pipe_set: set[tuple[int, int]],
    source_cells: set[tuple[int, int]],
    target_border: tuple[int, int],
) -> bool:
    h, w = grid.shape
    path_set = set(path)
    if len(path_set) != len(path):
        return False
    for idx, cell in enumerate(path):
        r, c = cell
        if not (0 <= r < h and 0 <= c < w):
            return False
        prev = path[idx - 1] if idx > 0 else None
        nxt = path[idx + 1] if idx + 1 < len(path) else None
        existing = cell in pipe_set

        if grid[r, c] != BG and not existing:
            return False

        for nr, nc in _neighbors4(r, c, h, w):
            if (nr, nc) in pipe_set:
                if not existing and (nr, nc) not in {prev, nxt}:
                    return False
            elif grid[nr, nc] != BG:
                if idx == 0 and (nr, nc) in source_cells:
                    continue
                if idx == len(path) - 1 and (nr, nc) == target_border:
                    continue
                return False

        if existing:
            if prev in pipe_set or nxt in pipe_set:
                return False
            old_orientation = _existing_pipe_orientation(pipe_set, cell)
            new_orientation = _path_cell_orientation(path, idx)
            if old_orientation is None or new_orientation is None or old_orientation == new_orientation:
                return False
        else:
            for nb in (prev, nxt):
                if nb is not None and nb in pipe_set and nb not in path_set:
                    return False

    return True


def _source_bbox_and_path(
    rng: random.Random,
    grid: np.ndarray,
    container_bbox: tuple[int, int, int, int],
    side: str,
    pos: int,
    pipe_set: set[tuple[int, int]],
) -> tuple[tuple[int, int, int, int], list[tuple[int, int]]] | None:
    h, w = grid.shape
    r0, c0, r1, c1 = container_bbox
    border_cell, end = _entry_border_and_pipe(container_bbox, side, pos)

    size = rng.choice((2, 3))
    attach_offset = 1 if size == 3 else rng.randrange(2)

    for _ in range(80):
        if side == "top":
            attach_col = rng.randint(max(1, c0 - 4), min(w - 2, c1 + 4))
            row_lo = max(size, r0 - 8)
            row_hi = r0 - 4
            if row_lo > row_hi:
                continue
            sr1 = rng.randint(row_lo, row_hi)
            sr0 = sr1 - size + 1
            sc0 = attach_col - attach_offset
            sc1 = sc0 + size - 1
            if sc0 < 0 or sc1 >= w:
                continue
            bbox = (sr0, sc0, sr1, sc1)
            if _touches_non_bg(grid, bbox, pad=1):
                continue
            start = (sr1 + 1, attach_col)
            source_cells = {(rr, cc) for rr in range(sr0, sr1 + 1) for cc in range(sc0, sc1 + 1)}
            if start[1] == end[1]:
                path = _segment(start, end)
            else:
                lane_candidates = list(range(start[0] + 1, end[0]))
                rng.shuffle(lane_candidates)
                path = []
                for lane_r in lane_candidates:
                    candidate = _polyline([start, (lane_r, attach_col), (lane_r, end[1]), end])
                    if _validate_path(grid, candidate, pipe_set, source_cells, border_cell):
                        path = candidate
                        break
                if not path:
                    continue
            if _validate_path(grid, path, pipe_set, source_cells, border_cell):
                return bbox, path

        elif side == "bottom":
            attach_col = rng.randint(max(1, c0 - 4), min(w - 2, c1 + 4))
            row_lo = r1 + 4
            row_hi = min(h - size - 1, r1 + 8)
            if row_lo > row_hi:
                continue
            sr0 = rng.randint(row_lo, row_hi)
            sr1 = sr0 + size - 1
            sc0 = attach_col - attach_offset
            sc1 = sc0 + size - 1
            if sc0 < 0 or sc1 >= w or sr1 >= h:
                continue
            bbox = (sr0, sc0, sr1, sc1)
            if _touches_non_bg(grid, bbox, pad=1):
                continue
            start = (sr0 - 1, attach_col)
            source_cells = {(rr, cc) for rr in range(sr0, sr1 + 1) for cc in range(sc0, sc1 + 1)}
            if start[1] == end[1]:
                path = _segment(start, end)
            else:
                lane_candidates = list(range(end[0] + 1, start[0]))
                rng.shuffle(lane_candidates)
                path = []
                for lane_r in lane_candidates:
                    candidate = _polyline([start, (lane_r, attach_col), (lane_r, end[1]), end])
                    if _validate_path(grid, candidate, pipe_set, source_cells, border_cell):
                        path = candidate
                        break
                if not path:
                    continue
            if _validate_path(grid, path, pipe_set, source_cells, border_cell):
                return bbox, path

        elif side == "left":
            attach_row = rng.randint(max(1, r0 - 4), min(h - 2, r1 + 4))
            col_lo = max(size, c0 - 8)
            col_hi = c0 - 4
            if col_lo > col_hi:
                continue
            sc1 = rng.randint(col_lo, col_hi)
            sc0 = sc1 - size + 1
            sr0 = attach_row - attach_offset
            sr1 = sr0 + size - 1
            if sr0 < 0 or sr1 >= h:
                continue
            bbox = (sr0, sc0, sr1, sc1)
            if _touches_non_bg(grid, bbox, pad=1):
                continue
            start = (attach_row, sc1 + 1)
            source_cells = {(rr, cc) for rr in range(sr0, sr1 + 1) for cc in range(sc0, sc1 + 1)}
            if start[0] == end[0]:
                path = _segment(start, end)
            else:
                lane_candidates = list(range(start[1] + 1, end[1]))
                rng.shuffle(lane_candidates)
                path = []
                for lane_c in lane_candidates:
                    candidate = _polyline([start, (attach_row, lane_c), (end[0], lane_c), end])
                    if _validate_path(grid, candidate, pipe_set, source_cells, border_cell):
                        path = candidate
                        break
                if not path:
                    continue
            if _validate_path(grid, path, pipe_set, source_cells, border_cell):
                return bbox, path

        else:
            attach_row = rng.randint(max(1, r0 - 4), min(h - 2, r1 + 4))
            col_lo = c1 + 4
            col_hi = min(w - size - 1, c1 + 8)
            if col_lo > col_hi:
                continue
            sc0 = rng.randint(col_lo, col_hi)
            sc1 = sc0 + size - 1
            sr0 = attach_row - attach_offset
            sr1 = sr0 + size - 1
            if sr0 < 0 or sr1 >= h or sc1 >= w:
                continue
            bbox = (sr0, sc0, sr1, sc1)
            if _touches_non_bg(grid, bbox, pad=1):
                continue
            start = (attach_row, sc0 - 1)
            source_cells = {(rr, cc) for rr in range(sr0, sr1 + 1) for cc in range(sc0, sc1 + 1)}
            if start[0] == end[0]:
                path = _segment(start, end)
            else:
                lane_candidates = list(range(end[1] + 1, start[1]))
                rng.shuffle(lane_candidates)
                path = []
                for lane_c in lane_candidates:
                    candidate = _polyline([start, (attach_row, lane_c), (end[0], lane_c), end])
                    if _validate_path(grid, candidate, pipe_set, source_cells, border_cell):
                        path = candidate
                        break
                if not path:
                    continue
            if _validate_path(grid, path, pipe_set, source_cells, border_cell):
                return bbox, path

    return None


def _container_spec(
    rng: random.Random,
    h: int,
    w: int,
    kind: str,
    existing: list[tuple[int, int, int, int]],
) -> dict[str, object] | None:
    for _ in range(120):
        if kind == "single":
            side = rng.choice(SIDES)
            outer_h = rng.randint(5, 8)
            outer_w = rng.randint(5, 9)
        elif kind == "opposite":
            vertical = rng.choice((True, False))
            side = "tb" if vertical else "lr"
            outer_h = rng.randint(6, 9) if vertical else rng.randint(5, 8)
            outer_w = rng.randint(5, 9) if vertical else rng.randint(6, 9)
        else:
            side = rng.choice(SIDES)
            if side in {"top", "bottom"}:
                outer_h = rng.randint(5, 8)
                outer_w = rng.randint(6, 10)
            else:
                outer_h = rng.randint(6, 10)
                outer_w = rng.randint(5, 8)

        top_margin = 6 if side in {"top", "tb"} else 3
        bottom_margin = 6 if side in {"bottom", "tb"} else 3
        left_margin = 6 if side in {"left", "lr"} else 3
        right_margin = 6 if side in {"right", "lr"} else 3

        if h - top_margin - bottom_margin < outer_h or w - left_margin - right_margin < outer_w:
            continue

        r0 = rng.randint(top_margin, h - bottom_margin - outer_h)
        c0 = rng.randint(left_margin, w - right_margin - outer_w)
        bbox = (r0, c0, r0 + outer_h - 1, c0 + outer_w - 1)

        if any(_boxes_close(bbox, other, pad=4) for other in existing):
            continue

        positions = {}
        if kind == "single":
            if side in {"top", "bottom"}:
                positions[side] = [rng.randint(c0 + 1, c0 + outer_w - 2)]
            else:
                positions[side] = [rng.randint(r0 + 1, r0 + outer_h - 2)]
        elif kind == "opposite":
            if side == "tb":
                positions["top"] = [rng.randint(c0 + 1, c0 + outer_w - 2)]
                positions["bottom"] = [rng.randint(c0 + 1, c0 + outer_w - 2)]
            else:
                positions["left"] = [rng.randint(r0 + 1, r0 + outer_h - 2)]
                positions["right"] = [rng.randint(r0 + 1, r0 + outer_h - 2)]
        else:
            if side in {"top", "bottom"}:
                choices = list(range(c0 + 1, c0 + outer_w - 1))
            else:
                choices = list(range(r0 + 1, r0 + outer_h - 1))
            if len(choices) < 2:
                continue
            rng.shuffle(choices)
            picks = sorted(choices[:2])
            if len(choices) >= 4 and abs(picks[0] - picks[1]) < 2:
                continue
            positions[side] = picks

        return {"bbox": bbox, "kind": kind, "positions": positions}
    return None


def _gen_attempt(rng: random.Random) -> np.ndarray | None:
    h = rng.randint(20, 29)
    w = rng.randint(20, 29)
    grid = np.full((h, w), BG, dtype=int)

    container_count = 3 if rng.random() < 0.55 else 2
    type_pool = ["single", "opposite", "same"]
    rng.shuffle(type_pool)
    kinds = type_pool[:container_count]

    container_specs = []
    used_boxes: list[tuple[int, int, int, int]] = []
    for kind in kinds:
        spec = _container_spec(rng, h, w, kind, used_boxes)
        if spec is None:
            return None
        container_specs.append(spec)
        used_boxes.append(spec["bbox"])

    total_sources = sum(sum(len(v) for v in spec["positions"].values()) for spec in container_specs)
    if total_sources > len(SOURCE_COLORS):
        return None
    colors = list(SOURCE_COLORS)
    rng.shuffle(colors)

    for spec in container_specs:
        _draw_frame(grid, spec["bbox"], FRAME)

    pipe_set: set[tuple[int, int]] = set()
    color_idx = 0
    for spec in container_specs:
        bbox = spec["bbox"]
        entries = []
        for side in SIDES:
            for pos in spec["positions"].get(side, []):
                entries.append((side, pos))
        rng.shuffle(entries)
        for side, pos in entries:
            placed = _source_bbox_and_path(rng, grid, bbox, side, pos, pipe_set)
            if placed is None:
                return None
            source_bbox, path = placed
            color = colors[color_idx]
            color_idx += 1
            sr0, sc0, sr1, sc1 = source_bbox
            grid[sr0 : sr1 + 1, sc0 : sc1 + 1] = color
            for cell in path:
                grid[cell] = PIPE
            pipe_set.update(path)

    return grid


def generate_input() -> np.ndarray:
    rng = random.Random()
    for _ in range(MAX_GENERATION_ATTEMPTS):
        grid = _gen_attempt(rng)
        if grid is None:
            continue
        try:
            out = main(grid)
        except Exception:
            continue
        if out.shape == grid.shape and max(grid.shape) <= 30 and max(out.shape) <= 30:
            return grid
    raise RuntimeError("failed to generate a valid pipe-container puzzle")
