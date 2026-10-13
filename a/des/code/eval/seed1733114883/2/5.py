from __future__ import annotations

from collections import deque
import random

import numpy as np


BG = 0
WIRE = 5
RECEIVER = 9
SIGNAL_COLORS = (1, 2, 3, 4, 6, 7, 8)
DIR4 = [(-1, 0), (1, 0), (0, -1), (0, 1)]
MAX_RETRIES = 400


def _in_bounds(shape: tuple[int, int], r: int, c: int) -> bool:
    return 0 <= r < shape[0] and 0 <= c < shape[1]


def _component_cells(mask: np.ndarray) -> list[list[tuple[int, int]]]:
    h, w = mask.shape
    seen = np.zeros((h, w), dtype=bool)
    comps: list[list[tuple[int, int]]] = []
    for r in range(h):
        for c in range(w):
            if not mask[r, c] or seen[r, c]:
                continue
            seen[r, c] = True
            q = deque([(r, c)])
            pts: list[tuple[int, int]] = []
            while q:
                rr, cc = q.popleft()
                pts.append((rr, cc))
                for dr, dc in DIR4:
                    nr, nc = rr + dr, cc + dc
                    if _in_bounds(mask.shape, nr, nc) and mask[nr, nc] and not seen[nr, nc]:
                        seen[nr, nc] = True
                        q.append((nr, nc))
            comps.append(pts)
    return comps


def _bbox(points: list[tuple[int, int]]) -> tuple[int, int, int, int]:
    rows = [r for r, _ in points]
    cols = [c for _, c in points]
    return min(rows), min(cols), max(rows), max(cols)


def _find_legend(grid: np.ndarray) -> tuple[dict[tuple[int, int], int], np.ndarray]:
    h, w = grid.shape
    candidates: list[tuple[int, dict[tuple[int, int], int], np.ndarray]] = []
    for r0 in (0, h - 2):
        if r0 < 0 or r0 + 1 >= h:
            continue
        top = grid[r0]
        cols = np.flatnonzero(top != BG)
        if cols.size == 0:
            continue
        c0, c1 = int(cols.min()), int(cols.max())
        width = c1 - c0 + 1
        if width < 2 or width % 2 != 0:
            continue
        if np.any(grid[r0, :c0] != BG) or np.any(grid[r0, c1 + 1 :] != BG):
            continue
        if np.any(grid[r0 + 1, :c0] != BG) or np.any(grid[r0 + 1, c1 + 1 :] != BG):
            continue
        if np.any(grid[r0, c0 : c1 + 1] == BG):
            continue

        mapping: dict[tuple[int, int], int] = {}
        ok = True
        for c in range(c0, c1 + 1, 2):
            input_color = int(grid[r0, c])
            filter_color = int(grid[r0, c + 1])
            output_color = int(grid[r0 + 1, c])
            filler = int(grid[r0 + 1, c + 1])
            if BG in (input_color, filter_color, output_color) or filler != BG:
                ok = False
                break
            key = (input_color, filter_color)
            if key in mapping and mapping[key] != output_color:
                ok = False
                break
            mapping[key] = output_color
        if not ok or not mapping:
            continue

        mask = np.zeros_like(grid, dtype=bool)
        mask[r0 : r0 + 2, c0 : c1 + 1] = True
        candidates.append((len(mapping), mapping, mask))

    if not candidates:
        raise AssertionError("legend not found")

    candidates.sort(key=lambda item: item[0], reverse=True)
    return candidates[0][1], candidates[0][2]


def _find_sources(grid: np.ndarray, legend_mask: np.ndarray) -> list[dict[str, object]]:
    mask = (grid != BG) & (grid != WIRE) & (grid != RECEIVER) & (~legend_mask)
    sources: list[dict[str, object]] = []
    for pts in _component_cells(mask):
        if len(pts) != 9:
            continue
        r0, c0, r1, c1 = _bbox(pts)
        if (r1 - r0, c1 - c0) != (2, 2):
            continue
        color = int(grid[r0, c0])
        patch = grid[r0 : r1 + 1, c0 : c1 + 1]
        if np.all(patch == color):
            sources.append(
                {
                    "color": color,
                    "bbox": (r0, c0, r1, c1),
                    "cells": set(pts),
                }
            )
    if not sources:
        raise AssertionError("sources not found")
    return sources


def _find_receivers(grid: np.ndarray) -> list[dict[str, object]]:
    receivers: list[dict[str, object]] = []
    for pts in _component_cells(grid == RECEIVER):
        if len(pts) != 8:
            continue
        r0, c0, r1, c1 = _bbox(pts)
        if (r1 - r0, c1 - c0) != (2, 2):
            continue
        patch = grid[r0 : r1 + 1, c0 : c1 + 1]
        expected = np.full((3, 3), RECEIVER, dtype=int)
        expected[1, 1] = BG
        if np.array_equal(patch, expected):
            receivers.append(
                {
                    "center": (r0 + 1, c0 + 1),
                    "border": set(pts),
                }
            )
    if not receivers:
        raise AssertionError("receivers not found")
    return receivers


def _source_exit(
    grid: np.ndarray,
    source_cells: set[tuple[int, int]],
    legend_mask: np.ndarray,
) -> tuple[tuple[int, int], tuple[int, int]]:
    exits: dict[tuple[int, int], tuple[int, int]] = {}
    for r, c in source_cells:
        for dr, dc in DIR4:
            nr, nc = r + dr, c + dc
            if not _in_bounds(grid.shape, nr, nc) or (nr, nc) in source_cells:
                continue
            if legend_mask[nr, nc] or int(grid[nr, nc]) == BG:
                continue
            if int(grid[nr, nc]) == RECEIVER:
                continue
            exits[(nr, nc)] = (dr, dc)
    if len(exits) != 1:
        raise AssertionError("each source must have exactly one wire exit")
    (cell, direction), = exits.items()
    return cell, direction


def main(grid: np.ndarray) -> np.ndarray:
    """
    Propagate each 3x3 source color through its grey wire, apply legend-driven
    color changes at filter pixels, fill the receiver center, and restore
    filters back to grey wire cells.
    """
    g = np.asarray(grid, dtype=int)
    mapping, legend_mask = _find_legend(g)
    sources = _find_sources(g, legend_mask)
    receivers = _find_receivers(g)

    border_to_center: dict[tuple[int, int], tuple[int, int]] = {}
    for receiver in receivers:
        center = receiver["center"]
        for cell in receiver["border"]:
            border_to_center[cell] = center

    out = g.copy()
    seen_centers: set[tuple[int, int]] = set()

    for source in sources:
        current = int(source["color"])
        pos, direction = _source_exit(g, source["cells"], legend_mask)
        dr, dc = direction
        filter_cells: list[tuple[int, int]] = []

        while True:
            r, c = pos
            value = int(g[r, c])
            if value == WIRE:
                pass
            else:
                key = (current, value)
                if key not in mapping:
                    raise AssertionError("missing legend mapping for filter")
                current = mapping[key]
                filter_cells.append(pos)

            nr, nc = r + dr, c + dc
            if not _in_bounds(g.shape, nr, nc):
                raise AssertionError("wire left the grid before reaching receiver")

            if (nr, nc) in border_to_center:
                center = border_to_center[(nr, nc)]
                seen_centers.add(center)
                if out[center] != BG:
                    raise AssertionError("receiver center must be blank in input")
                out[center] = current
                for fr, fc in filter_cells:
                    out[fr, fc] = WIRE
                break

            next_value = int(g[nr, nc])
            if next_value in (BG, RECEIVER) or legend_mask[nr, nc]:
                raise AssertionError("wire path is broken or ambiguous")
            pos = (nr, nc)

    if len(seen_centers) != len(receivers):
        raise AssertionError("every receiver must be reached exactly once")

    return out


def _reserve_legend_buffer(blocked: np.ndarray, legend_side: str) -> None:
    blocked[:, 0] = True
    blocked[:, -1] = True
    if legend_side == "top":
        blocked[:3, :] = True
    else:
        blocked[-3:, :] = True


def _mark_blocked(blocked: np.ndarray, cells: list[tuple[int, int]]) -> None:
    for r, c in cells:
        for nr in range(r - 1, r + 2):
            for nc in range(c - 1, c + 2):
                if _in_bounds(blocked.shape, nr, nc):
                    blocked[nr, nc] = True


def _pair_cells(tl_r: int, tl_c: int) -> list[tuple[int, int]]:
    return [(tl_r + dr, tl_c + dc) for dr in range(3) for dc in range(3)]


def _receiver_frame_cells(tl_r: int, tl_c: int) -> list[tuple[int, int]]:
    cells = _pair_cells(tl_r, tl_c)
    return [cell for cell in cells if cell != (tl_r + 1, tl_c + 1)]


def _choose_filter_indices(length: int, count: int, rng: random.Random) -> list[int]:
    if count == 0:
        return []
    if length < 2 * count + 1:
        raise AssertionError("wire too short for requested filters")
    indices: list[int] = []
    start = 1
    for placed in range(count):
        remaining = count - placed - 1
        hi = length - 2 - 2 * remaining
        idx = rng.randint(start, hi)
        indices.append(idx)
        start = idx + 2
    return indices


def _attempt_place_pair(
    blocked: np.ndarray,
    orientation: str,
    filter_count: int,
    rng: random.Random,
) -> dict[str, object] | None:
    h, w = blocked.shape
    min_wire_len = 1 if filter_count == 0 else 2 * filter_count + 1
    for _ in range(250):
        if orientation == "h":
            max_wire_len = w - 8
            if max_wire_len < min_wire_len:
                return None
            wire_len = rng.randint(min_wire_len, min(max_wire_len, min_wire_len + 8))
            total_width = wire_len + 6
            if total_width > w - 2:
                continue
            top_r = rng.randint(1, h - 4)
            left_c = rng.randint(1, w - total_width - 1)
            right_c = left_c + 3 + wire_len
            attach = rng.randint(0, 2)
            wire_row = top_r + attach
            left_bbox = _pair_cells(top_r, left_c)
            right_bbox = _pair_cells(top_r, right_c)
            wire_cells = [(wire_row, left_c + 3 + i) for i in range(wire_len)]
            all_cells = left_bbox + right_bbox + wire_cells
            if any(blocked[r, c] for r, c in all_cells):
                continue
            source_on_left = rng.random() < 0.5
            if source_on_left:
                source_tl = (top_r, left_c)
                receiver_tl = (top_r, right_c)
                path = wire_cells[:]
            else:
                source_tl = (top_r, right_c)
                receiver_tl = (top_r, left_c)
                path = list(reversed(wire_cells))
        else:
            max_wire_len = h - 8
            if max_wire_len < min_wire_len:
                return None
            wire_len = rng.randint(min_wire_len, min(max_wire_len, min_wire_len + 8))
            total_height = wire_len + 6
            if total_height > h - 2:
                continue
            top_r = rng.randint(1, h - total_height - 1)
            left_c = rng.randint(1, w - 4)
            bottom_r = top_r + 3 + wire_len
            attach = rng.randint(0, 2)
            wire_col = left_c + attach
            top_bbox = _pair_cells(top_r, left_c)
            bottom_bbox = _pair_cells(bottom_r, left_c)
            wire_cells = [(top_r + 3 + i, wire_col) for i in range(wire_len)]
            all_cells = top_bbox + bottom_bbox + wire_cells
            if any(blocked[r, c] for r, c in all_cells):
                continue
            source_on_top = rng.random() < 0.5
            if source_on_top:
                source_tl = (top_r, left_c)
                receiver_tl = (bottom_r, left_c)
                path = wire_cells[:]
            else:
                source_tl = (bottom_r, left_c)
                receiver_tl = (top_r, left_c)
                path = list(reversed(wire_cells))

        return {
            "orientation": orientation,
            "filter_count": filter_count,
            "wire_path": path,
            "all_cells": all_cells,
            "source_tl": source_tl,
            "receiver_tl": receiver_tl,
        }
    return None


def _expected_output(grid: np.ndarray, pairs: list[dict[str, object]]) -> np.ndarray:
    out = grid.copy()
    for pair in pairs:
        rr, rc = pair["receiver_tl"]
        out[rr + 1, rc + 1] = int(pair["final_color"])
        for idx in pair["filter_indices"]:
            fr, fc = pair["wire_path"][idx]
            out[fr, fc] = WIRE
    return out


def _gen_attempt(rng: random.Random) -> np.ndarray | None:
    pair_count = rng.choice([2, 3, 4])
    orientations = [rng.choice(["h", "v"]) for _ in range(pair_count)]
    if pair_count >= 2 and len(set(orientations)) == 1 and rng.random() < 0.7:
        flip = rng.randrange(pair_count)
        orientations[flip] = "h" if orientations[flip] == "v" else "v"

    filter_counts = [rng.choice([0, 1, 2, 3]) for _ in range(pair_count)]
    if sum(filter_counts) == 0:
        filter_counts[rng.randrange(pair_count)] = 1
    if all(count < 2 for count in filter_counts) and rng.random() < 0.5:
        filter_counts[rng.randrange(pair_count)] = 2

    h = rng.randint(18, 28)
    w = rng.randint(20, 30)
    legend_side = rng.choice(["top", "bottom"])
    blocked = np.zeros((h, w), dtype=bool)
    _reserve_legend_buffer(blocked, legend_side)

    specs = list(zip(orientations, filter_counts))
    specs.sort(key=lambda item: item[1], reverse=True)

    pairs: list[dict[str, object]] = []
    for orientation, filter_count in specs:
        pair = _attempt_place_pair(blocked, orientation, filter_count, rng)
        if pair is None:
            return None
        _mark_blocked(blocked, pair["all_cells"])
        pairs.append(pair)

    source_colors = rng.sample(list(SIGNAL_COLORS), pair_count)
    mapping: dict[tuple[int, int], int] = {}
    total_filters = 0

    for pair, source_color in zip(pairs, source_colors):
        path = pair["wire_path"]
        filter_count = int(pair["filter_count"])
        if len(path) < 1 or len(path) < (2 * filter_count + 1 if filter_count else 1):
            return None
        filter_indices = _choose_filter_indices(len(path), filter_count, rng)
        current = source_color
        filter_colors: list[int] = []
        for _ in filter_indices:
            filter_color = rng.choice(SIGNAL_COLORS)
            key = (current, filter_color)
            if key not in mapping:
                outputs = [color for color in SIGNAL_COLORS if color != current]
                mapping[key] = rng.choice(outputs)
            current = mapping[key]
            filter_colors.append(filter_color)
        pair["source_color"] = source_color
        pair["filter_indices"] = filter_indices
        pair["filter_colors"] = filter_colors
        pair["final_color"] = current
        total_filters += len(filter_indices)

    if total_filters == 0 or not mapping:
        return None

    legend_entries = list(mapping.items())
    rng.shuffle(legend_entries)
    legend_width = 2 * len(legend_entries)
    if legend_width > w:
        return None
    legend_c0 = rng.randint(0, w - legend_width)
    legend_r0 = 0 if legend_side == "top" else h - 2

    grid = np.zeros((h, w), dtype=int)

    for pair in pairs:
        sr, sc = pair["source_tl"]
        rr, rc = pair["receiver_tl"]
        grid[sr : sr + 3, sc : sc + 3] = int(pair["source_color"])
        for br, bc in _receiver_frame_cells(rr, rc):
            grid[br, bc] = RECEIVER
        for wr, wc in pair["wire_path"]:
            grid[wr, wc] = WIRE
        for idx, filter_color in zip(pair["filter_indices"], pair["filter_colors"]):
            fr, fc = pair["wire_path"][idx]
            grid[fr, fc] = filter_color

    for i, ((input_color, filter_color), output_color) in enumerate(legend_entries):
        c = legend_c0 + 2 * i
        grid[legend_r0, c] = input_color
        grid[legend_r0, c + 1] = filter_color
        grid[legend_r0 + 1, c] = output_color
        grid[legend_r0 + 1, c + 1] = BG

    expected = _expected_output(grid, pairs)
    if not np.array_equal(main(grid), expected):
        return None
    return grid


def generate_input() -> np.ndarray:
    rng = random.Random()
    for _ in range(MAX_RETRIES):
        grid = _gen_attempt(rng)
        if grid is not None:
            return grid
    raise RuntimeError(f"failed to generate valid input after {MAX_RETRIES} attempts")
