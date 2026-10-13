from __future__ import annotations

from collections import deque
import random

import numpy as np


BLACK = 0
YELLOW = 4
GREY = 5
TRACK_COLORS = (1, 2, 3, 6, 7, 8, 9)

GRID_MIN = 14
GRID_MAX = 24
MAX_TRACKS = 4
MAX_TRACK_LEN = 11
MAX_RELAYS = 5
GENERATION_RETRIES = 400
PLACEMENT_RETRIES = 120


def _neighbors4(r: int, c: int) -> tuple[tuple[int, int], ...]:
    return ((r - 1, c), (r + 1, c), (r, c - 1), (r, c + 1))


def _components(mask: np.ndarray) -> list[list[tuple[int, int]]]:
    h, w = mask.shape
    seen = np.zeros((h, w), dtype=bool)
    comps: list[list[tuple[int, int]]] = []
    for r in range(h):
        for c in range(w):
            if seen[r, c] or not bool(mask[r, c]):
                continue
            seen[r, c] = True
            q = deque([(r, c)])
            comp: list[tuple[int, int]] = []
            while q:
                rr, cc = q.popleft()
                comp.append((rr, cc))
                for nr, nc in _neighbors4(rr, cc):
                    if 0 <= nr < h and 0 <= nc < w and bool(mask[nr, nc]) and not bool(seen[nr, nc]):
                        seen[nr, nc] = True
                        q.append((nr, nc))
            comps.append(comp)
    return comps


def _bbox(coords: list[tuple[int, int]]) -> tuple[int, int, int, int]:
    rows = [r for r, _ in coords]
    cols = [c for _, c in coords]
    return min(rows), min(cols), max(rows), max(cols)


def _mask_connected(mask: np.ndarray) -> bool:
    pts = np.argwhere(mask)
    if pts.size == 0:
        return False
    seen = np.zeros_like(mask, dtype=bool)
    start = tuple(map(int, pts[0]))
    q = deque([start])
    seen[start] = True
    count = 0
    while q:
        r, c = q.popleft()
        count += 1
        for nr, nc in _neighbors4(r, c):
            if 0 <= nr < mask.shape[0] and 0 <= nc < mask.shape[1] and bool(mask[nr, nc]) and not bool(seen[nr, nc]):
                seen[nr, nc] = True
                q.append((nr, nc))
    return count == int(mask.sum())


def _is_asymmetric(mask: np.ndarray) -> bool:
    return (
        not np.array_equal(mask, np.fliplr(mask))
        and not np.array_equal(mask, np.flipud(mask))
        and not np.array_equal(mask, np.rot90(mask, 2))
    )


def _pattern_catalog() -> dict[int, list[np.ndarray]]:
    catalog: dict[int, list[np.ndarray]] = {2: [], 3: []}
    for size in (2, 3):
        total = size * size
        for bits in range(1, 1 << total):
            mask = np.zeros((size, size), dtype=bool)
            for idx in range(total):
                if bits & (1 << idx):
                    mask[idx // size, idx % size] = True
            if not np.all(mask.any(axis=0)) or not np.all(mask.any(axis=1)):
                continue
            if not _mask_connected(mask):
                continue
            if not _is_asymmetric(mask):
                continue
            catalog[size].append(mask)
    return catalog


def _contact_ok(mask: np.ndarray, edge: str, anchor_idx: int) -> bool:
    if edge == "top":
        return int(mask[0, :].sum()) == 1 and bool(mask[0, anchor_idx])
    if edge == "bottom":
        return int(mask[-1, :].sum()) == 1 and bool(mask[-1, anchor_idx])
    if edge == "left":
        return int(mask[:, 0].sum()) == 1 and bool(mask[anchor_idx, 0])
    if edge == "right":
        return int(mask[:, -1].sum()) == 1 and bool(mask[anchor_idx, -1])
    raise ValueError(edge)


_PATTERNS = _pattern_catalog()
_PLACEMENT_PATTERNS: dict[tuple[int, str, int], list[np.ndarray]] = {}
for _size, _masks in _PATTERNS.items():
    for _edge in ("top", "bottom", "left", "right"):
        for _anchor in range(_size):
            key = (_size, _edge, _anchor)
            _PLACEMENT_PATTERNS[key] = [mask.copy() for mask in _masks if _contact_ok(mask, _edge, _anchor)]
            assert _PLACEMENT_PATTERNS[key], f"missing patterns for {key}"


def _track_components(grid: np.ndarray) -> list[dict[str, object]]:
    track_mask = (grid != BLACK) & (grid != GREY)
    tracks: list[dict[str, object]] = []
    for coords in _components(track_mask):
        rows = sorted({r for r, _ in coords})
        cols = sorted({c for _, c in coords})
        colors = {int(grid[r, c]) for r, c in coords}
        rail_colors = colors - {YELLOW}
        assert len(rail_colors) == 1, "each track must have exactly one non-yellow rail color"
        rail_color = rail_colors.pop()

        if len(rows) == 1:
            row = rows[0]
            c0, c1 = cols[0], cols[-1]
            assert len(coords) == c1 - c0 + 1, "horizontal track must be gapless"
            orientation = "h"
            endpoints = ((row, c0), (row, c1))
        else:
            assert len(cols) == 1, "track must be straight"
            col = cols[0]
            r0, r1 = rows[0], rows[-1]
            assert len(coords) == r1 - r0 + 1, "vertical track must be gapless"
            orientation = "v"
            endpoints = ((r0, col), (r1, col))

        tracks.append(
            {
                "coords": coords,
                "coord_set": set(coords),
                "orientation": orientation,
                "endpoints": endpoints,
                "color": int(rail_color),
                "relay_count": sum(int(grid[r, c] == YELLOW) for r, c in coords),
            }
        )
    return tracks


def main(grid: np.ndarray) -> np.ndarray:
    """
    Move each gray pattern block along its adjacent rail by the number of yellow relay cells,
    then recolor the block to the rail color while leaving the rails unchanged.
    """
    g = np.asarray(grid, dtype=int)
    h, w = g.shape

    grey_comps = _components(g == GREY)
    grey_index = np.full((h, w), -1, dtype=int)
    for idx, coords in enumerate(grey_comps):
        for r, c in coords:
            grey_index[r, c] = idx

    tracks = _track_components(g)
    assert tracks, "expected at least one track"
    assert grey_comps, "expected at least one gray pattern block"

    moves: list[tuple[int, int, int, int, int]] = []
    used_grey: set[int] = set()

    for track in tracks:
        endpoints = track["endpoints"]
        endpoint_hits: list[tuple[int, int]] = []
        for endpoint_idx, (er, ec) in enumerate(endpoints):
            neighbor_ids = {
                int(grey_index[nr, nc])
                for nr, nc in _neighbors4(int(er), int(ec))
                if 0 <= nr < h and 0 <= nc < w and int(grey_index[nr, nc]) != -1
            }
            assert len(neighbor_ids) <= 1, "an endpoint cannot touch multiple gray blocks"
            if neighbor_ids:
                endpoint_hits.append((endpoint_idx, next(iter(neighbor_ids))))

        assert len(endpoint_hits) == 1, "each track must have one gray block attached to exactly one endpoint"
        endpoint_idx, grey_id = endpoint_hits[0]
        assert grey_id not in used_grey, "gray blocks must match tracks one-to-one"
        used_grey.add(grey_id)

        touched_track_cells = set()
        for r, c in grey_comps[grey_id]:
            for nr, nc in _neighbors4(r, c):
                if (nr, nc) in track["coord_set"]:
                    touched_track_cells.add((nr, nc))
        assert touched_track_cells == {tuple(map(int, endpoints[endpoint_idx]))}, "gray block must touch only the matched endpoint"

        if track["orientation"] == "h":
            dr, dc = (0, 1) if endpoint_idx == 0 else (0, -1)
        else:
            dr, dc = (1, 0) if endpoint_idx == 0 else (-1, 0)

        moves.append(
            (
                int(grey_id),
                int(dr),
                int(dc),
                int(track["relay_count"]),
                int(track["color"]),
            )
        )

    assert len(used_grey) == len(grey_comps), "every gray block must belong to exactly one track"

    out = g.copy()
    out[out == GREY] = BLACK
    for grey_id, dr, dc, dist, color in moves:
        for r, c in grey_comps[grey_id]:
            rr = int(r + dr * dist)
            cc = int(c + dc * dist)
            assert 0 <= rr < h and 0 <= cc < w, "translated block must stay in bounds"
            assert int(out[rr, cc]) == BLACK, "translated block must land on empty background"
            out[rr, cc] = int(color)

    return out


def _dilate(mask: np.ndarray) -> np.ndarray:
    h, w = mask.shape
    out = mask.copy()
    for dr in (-1, 0, 1):
        for dc in (-1, 0, 1):
            if dr == 0 and dc == 0:
                continue
            src_r0 = max(0, -dr)
            src_r1 = min(h, h - dr)
            src_c0 = max(0, -dc)
            src_c1 = min(w, w - dc)
            dst_r0 = src_r0 + dr
            dst_r1 = src_r1 + dr
            dst_c0 = src_c0 + dc
            dst_c1 = src_c1 + dc
            out[dst_r0:dst_r1, dst_c0:dst_c1] |= mask[src_r0:src_r1, src_c0:src_c1]
    return out


def _rect_cells(r0: int, c0: int, size: int) -> list[tuple[int, int]]:
    return [(r, c) for r in range(r0, r0 + size) for c in range(c0, c0 + size)]


def _paint_scene(grid: np.ndarray, objects: list[dict[str, object]]) -> None:
    for obj in objects:
        track_cells: list[tuple[int, int]] = obj["track_cells"]  # type: ignore[assignment]
        station_cells: set[tuple[int, int]] = obj["station_cells"]  # type: ignore[assignment]
        color = int(obj["color"])
        for r, c in track_cells:
            grid[r, c] = color
        for r, c in station_cells:
            grid[r, c] = YELLOW
        for r, c in obj["start_cells"]:  # type: ignore[assignment]
            grid[r, c] = GREY


def _sample_object(h: int, w: int, blocked: np.ndarray, color: int) -> dict[str, object] | None:
    orientation = random.choice(("h", "v"))
    size = random.choice((2, 3))

    axis_limit = w if orientation == "h" else h
    max_len = min(MAX_TRACK_LEN, axis_limit)
    if max_len < 2 * size:
        return None
    length = random.randint(2 * size, max_len)

    relay_hi = min(MAX_RELAYS, length - size)
    if relay_hi < size:
        return None
    relay_count = random.randint(size, relay_hi)

    endpoint_at_min = bool(random.getrandbits(1))
    attach_positive = bool(random.getrandbits(1))

    if orientation == "h":
        row_lo = size if not attach_positive else 0
        row_hi = h - 1 if not attach_positive else h - size - 1
        if row_lo > row_hi:
            return None
        row = random.randint(row_lo, row_hi)
        col0 = random.randint(0, w - length)
        col1 = col0 + length - 1

        block_r0 = row - size if not attach_positive else row + 1
        block_c0 = col0 if endpoint_at_min else col1 - size + 1
        dest_c0 = block_c0 + relay_count if endpoint_at_min else block_c0 - relay_count

        if not (0 <= dest_c0 and dest_c0 + size <= w):
            return None

        contact_edge = "bottom" if not attach_positive else "top"
        anchor_idx = 0 if endpoint_at_min else size - 1
        pattern = random.choice(_PLACEMENT_PATTERNS[(size, contact_edge, anchor_idx)]).copy()

        track_cells = [(row, c) for c in range(col0, col1 + 1)]
        offsets = random.sample(range(1, length - 1), relay_count)
        station_cells = {(row, col0 + offset) for offset in offsets}
        start_cells = [(block_r0 + dr, block_c0 + dc) for dr, dc in np.argwhere(pattern)]
        dest_cells = [(block_r0 + dr, dest_c0 + dc) for dr, dc in np.argwhere(pattern)]
        reserved = set(track_cells)
        reserved.update(_rect_cells(block_r0, block_c0, size))
        reserved.update(_rect_cells(block_r0, dest_c0, size))
    else:
        col_lo = size if not attach_positive else 0
        col_hi = w - 1 if not attach_positive else w - size - 1
        if col_lo > col_hi:
            return None
        col = random.randint(col_lo, col_hi)
        row0 = random.randint(0, h - length)
        row1 = row0 + length - 1

        block_r0 = row0 if endpoint_at_min else row1 - size + 1
        block_c0 = col - size if not attach_positive else col + 1
        dest_r0 = block_r0 + relay_count if endpoint_at_min else block_r0 - relay_count

        if not (0 <= dest_r0 and dest_r0 + size <= h):
            return None

        contact_edge = "right" if not attach_positive else "left"
        anchor_idx = 0 if endpoint_at_min else size - 1
        pattern = random.choice(_PLACEMENT_PATTERNS[(size, contact_edge, anchor_idx)]).copy()

        track_cells = [(r, col) for r in range(row0, row1 + 1)]
        offsets = random.sample(range(1, length - 1), relay_count)
        station_cells = {(row0 + offset, col) for offset in offsets}
        start_cells = [(block_r0 + dr, block_c0 + dc) for dr, dc in np.argwhere(pattern)]
        dest_cells = [(dest_r0 + dr, block_c0 + dc) for dr, dc in np.argwhere(pattern)]
        reserved = set(track_cells)
        reserved.update(_rect_cells(block_r0, block_c0, size))
        reserved.update(_rect_cells(dest_r0, block_c0, size))

    candidate = np.zeros((h, w), dtype=bool)
    for r, c in reserved:
        candidate[r, c] = True
    if bool((candidate & blocked).any()):
        return None

    return {
        "orientation": orientation,
        "color": int(color),
        "relay_count": int(relay_count),
        "track_cells": track_cells,
        "station_cells": station_cells,
        "start_cells": start_cells,
        "dest_cells": dest_cells,
        "reserved": reserved,
    }


def _gen_attempt() -> np.ndarray | None:
    h = random.randint(GRID_MIN, GRID_MAX)
    w = random.randint(GRID_MIN, GRID_MAX)
    grid = np.full((h, w), BLACK, dtype=int)

    max_tracks = min(MAX_TRACKS, len(TRACK_COLORS))
    track_count = random.randint(1, max_tracks)
    colors = random.sample(TRACK_COLORS, track_count)

    blocked = np.zeros((h, w), dtype=bool)
    objects: list[dict[str, object]] = []

    for color in colors:
        placed = None
        for _ in range(PLACEMENT_RETRIES):
            candidate = _sample_object(h, w, blocked, color)
            if candidate is None:
                continue
            reserved_mask = np.zeros((h, w), dtype=bool)
            for r, c in candidate["reserved"]:  # type: ignore[index]
                reserved_mask[r, c] = True
            blocked |= _dilate(reserved_mask)
            objects.append(candidate)
            placed = candidate
            break
        if placed is None:
            return None

    _paint_scene(grid, objects)

    try:
        out = main(grid)
    except Exception:
        return None

    if np.array_equal(grid, out):
        return None
    return grid


def generate_input() -> np.ndarray:
    for _ in range(GENERATION_RETRIES):
        grid = _gen_attempt()
        if grid is not None:
            return grid
    raise RuntimeError("failed to generate a valid rail-translation puzzle")
