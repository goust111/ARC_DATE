from __future__ import annotations

from collections import defaultdict, deque
import random

import numpy as np


BG = 0
FRAME = 5
ALL_COLORS = tuple(color for color in range(10) if color not in (BG, FRAME))
SIDES = ("top", "right", "bottom", "left")
SIDE_TO_DIR = {
    "top": (-1, 0),
    "right": (0, 1),
    "bottom": (1, 0),
    "left": (0, -1),
}

GEN_RETRIES = 500
FRAME_COUNT_RANGE = (2, 5)
FRAME_SIDE_RANGE = (6, 10)
MAX_OUTPUT_SIDE = 32
MAX_PATH_EXTRA = 8
PATH_TURN_OPTIONS = (0, 1)


def _neighbors4(r: int, c: int) -> tuple[tuple[int, int], ...]:
    return ((r - 1, c), (r + 1, c), (r, c - 1), (r, c + 1))


def _components(mask: np.ndarray) -> list[np.ndarray]:
    h, w = map(int, mask.shape)
    seen = np.zeros((h, w), dtype=bool)
    comps: list[np.ndarray] = []

    for r in range(h):
        for c in range(w):
            if not mask[r, c] or seen[r, c]:
                continue

            queue = deque([(r, c)])
            seen[r, c] = True
            cells: list[tuple[int, int]] = []

            while queue:
                rr, cc = queue.popleft()
                cells.append((rr, cc))
                for nr, nc in _neighbors4(rr, cc):
                    if 0 <= nr < h and 0 <= nc < w and mask[nr, nc] and not seen[nr, nc]:
                        seen[nr, nc] = True
                        queue.append((nr, nc))

            comps.append(np.array(cells, dtype=int))

    return comps


def _bbox(cells: np.ndarray) -> tuple[int, int, int, int]:
    return (
        int(cells[:, 0].min()),
        int(cells[:, 1].min()),
        int(cells[:, 0].max()),
        int(cells[:, 1].max()),
    )


def _bbox_gap(a: tuple[int, int, int, int], b: tuple[int, int, int, int]) -> int:
    def gap_1d(x0: int, x1: int, y0: int, y1: int) -> int:
        if y1 < x0:
            return int(x0 - y1 - 1)
        if x1 < y0:
            return int(y0 - x1 - 1)
        return 0

    return int(gap_1d(a[0], a[2], b[0], b[2]) + gap_1d(a[1], a[3], b[1], b[3]))


def _rotate_dir(direction: tuple[int, int], turns: int) -> tuple[int, int]:
    dr, dc = direction
    turns %= 4
    for _ in range(turns):
        dr, dc = -dc, dr
    return int(dr), int(dc)


def _rotate_points(
    points: list[tuple[int, int]], height: int, width: int, turns: int
) -> tuple[list[tuple[int, int]], int, int]:
    turns %= 4
    if turns == 0:
        return [(int(r), int(c)) for r, c in points], int(height), int(width)
    if turns == 1:
        return [(int(width - 1 - c), int(r)) for r, c in points], int(width), int(height)
    if turns == 2:
        return [(int(height - 1 - r), int(width - 1 - c)) for r, c in points], int(height), int(width)
    return [(int(c), int(height - 1 - r)) for r, c in points], int(width), int(height)


def _rotation_to_match(
    direction: tuple[int, int], target: tuple[int, int]
) -> int:
    for turns in range(4):
        if _rotate_dir(direction, turns) == target:
            return int(turns)
    raise AssertionError("direction match failed")


def _turn_count(path: list[tuple[int, int]]) -> int:
    turns = 0
    for i in range(1, len(path) - 1):
        dr1 = int(path[i][0] - path[i - 1][0])
        dc1 = int(path[i][1] - path[i - 1][1])
        dr2 = int(path[i + 1][0] - path[i][0])
        dc2 = int(path[i + 1][1] - path[i][1])
        if (dr1, dc1) != (dr2, dc2):
            turns += 1
    return int(turns)


def _is_simple_segment(path: list[tuple[int, int]]) -> bool:
    if len(path) != len(set(path)):
        return False
    cells = set(path)
    degrees = [sum(neigh in cells for neigh in _neighbors4(*cell)) for cell in cells]
    return max(degrees, default=0) <= 2 and sum(degree == 1 for degree in degrees) == 2


def _order_path(
    path_cells: set[tuple[int, int]],
    start: tuple[int, int],
    end: tuple[int, int],
) -> list[tuple[int, int]]:
    ordered = [start]
    seen = {start}
    prev: tuple[int, int] | None = None
    cur = start

    while cur != end:
        choices = []
        for nxt in _neighbors4(*cur):
            if nxt not in path_cells or nxt == prev:
                continue
            choices.append(nxt)
        assert len(choices) == 1, "path must be a simple non-branching segment"
        prev, cur = cur, choices[0]
        assert cur not in seen, "path must not self-intersect"
        seen.add(cur)
        ordered.append(cur)

    assert seen == path_cells, "ordered path must cover the whole segment"
    return ordered


def _gap_border_cell(side: str, position: int, frame_h: int, frame_w: int) -> tuple[int, int]:
    if side == "top":
        return 0, int(position)
    if side == "bottom":
        return int(frame_h - 1), int(position)
    if side == "left":
        return int(position), 0
    return int(position), int(frame_w - 1)


def _gap_anchor(side: str, position: int, interior_h: int, interior_w: int) -> tuple[int, int]:
    if side == "top":
        return 0, int(position - 1)
    if side == "bottom":
        return int(interior_h - 1), int(position - 1)
    if side == "left":
        return int(position - 1), 0
    return int(position - 1), int(interior_w - 1)


def _frame_info_from_path(
    path: list[tuple[int, int]],
    path_color: int,
    gaps: list[dict],
    free_endpoint: tuple[int, int] | None,
    bbox_key: tuple[int, int],
) -> dict:
    rs = [r for r, _ in path]
    cs = [c for _, c in path]
    r0, c0 = min(rs), min(cs)
    r1, c1 = max(rs), max(cs)
    translated_path = [(int(r - r0), int(c - c0)) for r, c in path]
    path_set = set(translated_path)
    gap_info = []

    for gap in gaps:
        ar, ac = gap["anchor"]
        gap_info.append(
            {
                "marker": gap["marker"],
                "dir": tuple(int(x) for x in gap["dir"]),
                "endpoint": (int(ar - r0), int(ac - c0)),
            }
        )

    free_endpoints: list[tuple[int, int]] = []
    if free_endpoint is not None:
        free_endpoints.append((int(free_endpoint[0] - r0), int(free_endpoint[1] - c0)))

    marker_to_gap = {
        int(gap["marker"]): idx
        for idx, gap in enumerate(gap_info)
        if gap["marker"] is not None
    }

    return {
        "path_color": int(path_color),
        "path_h": int(r1 - r0 + 1),
        "path_w": int(c1 - c0 + 1),
        "path_cells": translated_path,
        "path_set": path_set,
        "gaps": gap_info,
        "free_endpoints": free_endpoints,
        "marker_to_gap": marker_to_gap,
        "bbox_key": tuple(int(x) for x in bbox_key),
    }


def _parse_frames(grid: np.ndarray) -> list[dict]:
    g = np.asarray(grid, dtype=int)
    h, w = map(int, g.shape)
    frames: list[dict] = []
    grey_components = _components(g == FRAME)
    comp_boxes = [_bbox(comp) for comp in grey_components]
    parent = list(range(len(grey_components)))

    def find(index: int) -> int:
        while parent[index] != index:
            parent[index] = parent[parent[index]]
            index = parent[index]
        return int(index)

    def union(a: int, b: int) -> None:
        ra, rb = find(a), find(b)
        if ra != rb:
            parent[rb] = ra

    for i in range(len(grey_components)):
        for j in range(i + 1, len(grey_components)):
            if _bbox_gap(comp_boxes[i], comp_boxes[j]) <= 1:
                union(i, j)

    groups: dict[int, list[np.ndarray]] = defaultdict(list)
    for index, comp in enumerate(grey_components):
        groups[find(index)].append(comp)

    for parts in groups.values():
        merged = np.vstack(parts)
        comp = merged
        r0, c0, r1, c1 = _bbox(comp)
        assert r1 - r0 >= 2 and c1 - c0 >= 2, "frame must enclose an interior"

        gaps = []
        border_positions: list[tuple[int, int]] = []
        for c in range(c0, c1 + 1):
            border_positions.append((r0, c))
            border_positions.append((r1, c))
        for r in range(r0 + 1, r1):
            border_positions.append((r, c0))
            border_positions.append((r, c1))

        for rr, cc in border_positions:
            if int(g[rr, cc]) == FRAME:
                continue
            if (rr in (r0, r1)) and (cc in (c0, c1)):
                raise AssertionError("gap cannot occupy a frame corner")
            if rr == r0:
                side = "top"
            elif rr == r1:
                side = "bottom"
            elif cc == c0:
                side = "left"
            else:
                side = "right"
            dr, dc = SIDE_TO_DIR[side]
            mr, mc = int(rr + dr), int(cc + dc)
            marker = None
            if 0 <= mr < h and 0 <= mc < w:
                value = int(g[mr, mc])
                if value not in (BG, FRAME):
                    marker = int(value)
            gaps.append({"cell": (int(rr), int(cc)), "dir": (int(dr), int(dc)), "marker": marker})

        assert 1 <= len(gaps) <= 2, "each frame must have one or two gaps"

        inner = g[r0 + 1 : r1, c0 + 1 : c1]
        path_mask = (inner != BG) & (inner != FRAME)
        path_points = np.argwhere(path_mask)
        assert path_points.size > 0, "frame must contain a colored path"
        path_colors = {int(inner[r, c]) for r, c in path_points}
        assert len(path_colors) == 1, "each frame must contain exactly one path color"
        path_color = int(next(iter(path_colors)))

        global_path = {(int(r0 + 1 + r), int(c0 + 1 + c)) for r, c in path_points}
        degrees = {}
        endpoints: list[tuple[int, int]] = []
        for cell in global_path:
            deg = sum(neigh in global_path for neigh in _neighbors4(*cell))
            degrees[cell] = int(deg)
            if deg <= 1:
                endpoints.append(cell)
        assert len(endpoints) == 2, "path must have exactly two endpoints"

        path_box = _bbox(np.array(sorted(global_path), dtype=int))
        pr0, pc0, _pr1, _pc1 = path_box
        translated_path = {(int(r - pr0), int(c - pc0)) for r, c in global_path}
        gap_info = []
        gap_endpoints: set[tuple[int, int]] = set()

        for gap in gaps:
            dr, dc = gap["dir"]
            rr, cc = gap["cell"]
            anchor = (int(rr - dr), int(cc - dc))
            assert anchor in global_path, "each gap must touch a path endpoint"
            assert degrees[anchor] <= 1, "only path endpoints may touch a gap"
            local_anchor = (int(anchor[0] - pr0), int(anchor[1] - pc0))
            gap_endpoints.add(local_anchor)
            gap_info.append(
                {
                    "marker": gap["marker"],
                    "dir": tuple(int(x) for x in gap["dir"]),
                    "endpoint": local_anchor,
                }
            )

        free_endpoints = [
            (int(r - pr0), int(c - pc0))
            for r, c in endpoints
            if (int(r - pr0), int(c - pc0)) not in gap_endpoints
        ]
        marker_to_gap = {
            int(gap["marker"]): idx
            for idx, gap in enumerate(gap_info)
            if gap["marker"] is not None
        }

        frames.append(
            {
                "path_color": int(path_color),
                "path_h": int(path_box[2] - path_box[0] + 1),
                "path_w": int(path_box[3] - path_box[1] + 1),
                "path_cells": sorted(translated_path),
                "path_set": translated_path,
                "gaps": gap_info,
                "free_endpoints": free_endpoints,
                "marker_to_gap": marker_to_gap,
                "bbox_key": (int(r0), int(c0)),
            }
        )

    assert frames, "at least one grey frame is required"
    return frames


def _choose_start_frame(frames: list[dict], adjacency: dict[int, list[dict]]) -> int:
    if len(frames) == 1:
        return 0
    ends = [idx for idx in range(len(frames)) if len(adjacency[idx]) == 1]
    assert len(ends) == 2, "marker pairings must form one open chain"

    single_gap = [idx for idx in ends if len(frames[idx]["gaps"]) == 1]
    if len(single_gap) == 1:
        return int(single_gap[0])

    free_starts = [idx for idx in ends if len(frames[idx]["free_endpoints"]) == 1]
    if len(free_starts) == 1:
        return int(free_starts[0])

    return int(min(ends, key=lambda idx: frames[idx]["bbox_key"]))


def _resolve_frame_path(
    frame: dict,
    prev_color: int | None,
    next_color: int | None,
) -> tuple[list[tuple[int, int]], tuple[int, int] | None, tuple[int, int] | None]:
    def endpoint_from_gap(index: int) -> tuple[int, int]:
        return tuple(int(x) for x in frame["gaps"][index]["endpoint"])

    if prev_color is None:
        if frame["free_endpoints"]:
            start_kind = ("free", tuple(int(x) for x in frame["free_endpoints"][0]))
        else:
            unmarked = [idx for idx, gap in enumerate(frame["gaps"]) if gap["marker"] is None]
            if unmarked:
                start_kind = ("gap", int(unmarked[0]))
            else:
                start_kind = ("gap", int(min(range(len(frame["gaps"])), key=lambda idx: frame["gaps"][idx]["endpoint"])))
    else:
        start_kind = ("gap", int(frame["marker_to_gap"][int(prev_color)]))

    if next_color is None:
        unmarked = [idx for idx, gap in enumerate(frame["gaps"]) if gap["marker"] is None]
        remaining_unmarked = [idx for idx in unmarked if start_kind != ("gap", idx)]
        if remaining_unmarked:
            end_kind = ("gap", int(remaining_unmarked[0]))
        elif frame["free_endpoints"]:
            end_kind = ("free", tuple(int(x) for x in frame["free_endpoints"][0]))
        else:
            remaining = [idx for idx in range(len(frame["gaps"])) if start_kind != ("gap", idx)]
            assert remaining, "final frame needs a terminal endpoint"
            end_kind = ("gap", int(remaining[0]))
    else:
        end_kind = ("gap", int(frame["marker_to_gap"][int(next_color)]))

    if start_kind[0] == "free":
        start_cell = tuple(int(x) for x in start_kind[1])
        entry_dir = None
    else:
        start_cell = endpoint_from_gap(int(start_kind[1]))
        entry_dir = tuple(int(x) for x in frame["gaps"][int(start_kind[1])]["dir"])

    if end_kind[0] == "free":
        end_cell = tuple(int(x) for x in end_kind[1])
        exit_dir = None
    else:
        end_cell = endpoint_from_gap(int(end_kind[1]))
        exit_dir = tuple(int(x) for x in frame["gaps"][int(end_kind[1])]["dir"])

    ordered = _order_path(set(frame["path_set"]), start_cell, end_cell)
    return ordered, entry_dir, exit_dir


def _paint_cell(paint: dict[tuple[int, int], int], cell: tuple[int, int], color: int) -> None:
    assert cell not in paint, "assembled path must not overlap itself"
    paint[cell] = int(color)


def _assemble_from_frames(frames: list[dict]) -> np.ndarray:
    color_to_gaps: dict[int, list[tuple[int, int]]] = defaultdict(list)
    for frame_index, frame in enumerate(frames):
        for gap_index, gap in enumerate(frame["gaps"]):
            marker = gap["marker"]
            if marker is not None:
                color_to_gaps[int(marker)].append((int(frame_index), int(gap_index)))

    adjacency: dict[int, list[dict]] = {idx: [] for idx in range(len(frames))}
    for color, items in color_to_gaps.items():
        assert len(items) == 2, "each marker color must appear on exactly two gaps"
        (a_frame, a_gap), (b_frame, b_gap) = items
        adjacency[a_frame].append({"to": int(b_frame), "color": int(color), "gap": int(a_gap)})
        adjacency[b_frame].append({"to": int(a_frame), "color": int(color), "gap": int(b_gap)})

    for edges in adjacency.values():
        assert len(edges) <= 2, "chain frames cannot branch"

    start = _choose_start_frame(frames, adjacency)
    sequence = []
    connector_colors: list[int] = []
    visited: set[int] = set()
    prev_frame: int | None = None
    prev_color: int | None = None
    cur = int(start)

    while True:
        assert cur not in visited, "marker graph must be acyclic"
        visited.add(cur)
        next_edges = [edge for edge in adjacency[cur] if edge["to"] != prev_frame]
        assert len(next_edges) <= 1, "chain traversal must have at most one unused exit"
        next_edge = next_edges[0] if next_edges else None
        next_color = None if next_edge is None else int(next_edge["color"])

        ordered, entry_dir, exit_dir = _resolve_frame_path(frames[cur], prev_color, next_color)
        sequence.append(
            {
                "path_color": int(frames[cur]["path_color"]),
                "ordered": ordered,
                "path_h": int(frames[cur]["path_h"]),
                "path_w": int(frames[cur]["path_w"]),
                "entry_dir": entry_dir,
                "exit_dir": exit_dir,
            }
        )

        if next_edge is None:
            break

        connector_colors.append(int(next_color))
        prev_frame, cur, prev_color = cur, int(next_edge["to"]), int(next_color)

    assert len(visited) == len(frames), "all frames must belong to the same chain"

    paint: dict[tuple[int, int], int] = {}
    first = sequence[0]
    for cell in first["ordered"]:
        _paint_cell(paint, tuple(int(x) for x in cell), int(first["path_color"]))

    prev_end = tuple(int(x) for x in first["ordered"][-1])
    prev_exit = first["exit_dir"]

    for idx in range(1, len(sequence)):
        segment = sequence[idx]
        assert prev_exit is not None, "non-final segment must exit through a gap"
        assert segment["entry_dir"] is not None, "non-initial segment must enter through a gap"

        turns = _rotation_to_match(segment["entry_dir"], (-prev_exit[0], -prev_exit[1]))
        rotated_cells, _new_h, _new_w = _rotate_points(
            list(segment["ordered"]),
            int(segment["path_h"]),
            int(segment["path_w"]),
            turns,
        )
        rotated_exit = None
        if segment["exit_dir"] is not None:
            rotated_exit = _rotate_dir(segment["exit_dir"], turns)

        connector = (int(prev_end[0] + prev_exit[0]), int(prev_end[1] + prev_exit[1]))
        target_start = (int(prev_end[0] + 2 * prev_exit[0]), int(prev_end[1] + 2 * prev_exit[1]))
        dr = int(target_start[0] - rotated_cells[0][0])
        dc = int(target_start[1] - rotated_cells[0][1])
        translated = [(int(r + dr), int(c + dc)) for r, c in rotated_cells]

        _paint_cell(paint, connector, int(connector_colors[idx - 1]))
        for cell in translated:
            _paint_cell(paint, cell, int(segment["path_color"]))

        prev_end = translated[-1]
        prev_exit = rotated_exit

    min_r = min(r for r, _ in paint)
    min_c = min(c for _, c in paint)
    max_r = max(r for r, _ in paint)
    max_c = max(c for _, c in paint)

    out = np.full((max_r - min_r + 1, max_c - min_c + 1), BG, dtype=int)
    for (r, c), color in paint.items():
        out[int(r - min_r), int(c - min_c)] = int(color)
    return out


def main(grid: np.ndarray) -> np.ndarray:
    """
    Link framed path segments by paired gap-marker colors and output the assembled path.
    """
    g = np.asarray(grid, dtype=int)
    frames = _parse_frames(g)
    return _assemble_from_frames(frames)


def _find_exact_path(
    height: int,
    width: int,
    start: tuple[int, int],
    end: tuple[int, int],
    target_len: int,
    rng: random.Random,
) -> list[tuple[int, int]] | None:
    path = [tuple(int(x) for x in start)]
    used = {tuple(int(x) for x in start)}

    def search(cur: tuple[int, int]) -> bool:
        if len(path) == int(target_len):
            return cur == end

        remain = int(target_len - len(path))
        dist = abs(cur[0] - end[0]) + abs(cur[1] - end[1])
        if dist > remain:
            return False
        if (remain - dist) % 2 != 0:
            return False

        neighbors = []
        for nr, nc in _neighbors4(*cur):
            nxt = (int(nr), int(nc))
            if not (0 <= nr < height and 0 <= nc < width):
                continue
            if nxt in used:
                continue
            if nxt == end and len(path) != int(target_len) - 1:
                continue
            used_neighbors = sum(neigh in used for neigh in _neighbors4(*nxt))
            if used_neighbors != 1:
                continue
            neighbors.append(nxt)

        rng.shuffle(neighbors)
        neighbors.sort(key=lambda cell: abs(cell[0] - end[0]) + abs(cell[1] - end[1]))

        for nxt in neighbors:
            used.add(nxt)
            path.append(nxt)
            if search(nxt):
                return True
            path.pop()
            used.remove(nxt)

        return False

    return path if search(start) else None


def _random_path(
    height: int,
    width: int,
    start: tuple[int, int],
    end: tuple[int, int],
    min_turns: int,
    rng: random.Random,
) -> list[tuple[int, int]] | None:
    shortest = abs(start[0] - end[0]) + abs(start[1] - end[1]) + 1
    max_len = min(int(height * width), int(shortest + MAX_PATH_EXTRA))
    candidates = [length for length in range(shortest, max_len + 1) if (length - shortest) % 2 == 0]
    rng.shuffle(candidates)

    for target_len in candidates:
        for _ in range(60):
            path = _find_exact_path(height, width, start, end, target_len, rng)
            if path is not None and _is_simple_segment(path) and _turn_count(path) >= min_turns:
                return path
    return None


def _random_gap(side: str, frame_h: int, frame_w: int, rng: random.Random) -> dict:
    if side in ("top", "bottom"):
        position = rng.randint(1, frame_w - 2)
    else:
        position = rng.randint(1, frame_h - 2)
    return {
        "side": side,
        "position": int(position),
        "dir": SIDE_TO_DIR[side],
        "anchor": _gap_anchor(side, int(position), frame_h - 2, frame_w - 2),
    }


def _build_frame(
    role: str,
    path_color: int,
    incoming_marker: int | None,
    outgoing_marker: int | None,
    frame_index: int,
    rng: random.Random,
) -> dict | None:
    for _ in range(120):
        frame_h = rng.randint(*FRAME_SIDE_RANGE)
        frame_w = rng.randint(*FRAME_SIDE_RANGE)
        inner_h, inner_w = int(frame_h - 2), int(frame_w - 2)
        sides = list(SIDES)
        rng.shuffle(sides)
        min_turns = rng.choice(PATH_TURN_OPTIONS)

        if role == "start":
            gap = _random_gap(sides[0], frame_h, frame_w, rng)
            gap["marker"] = int(outgoing_marker)
            interior_cells = [
                (r, c)
                for r in range(1, inner_h - 1)
                for c in range(1, inner_w - 1)
                if (r, c) != gap["anchor"]
            ]
            if not interior_cells:
                continue
            free_endpoint = rng.choice(interior_cells)
            path = _random_path(inner_h, inner_w, free_endpoint, gap["anchor"], min_turns, rng)
            if path is None:
                continue
            gaps = [gap]

        elif role == "internal":
            gap_a = _random_gap(sides[0], frame_h, frame_w, rng)
            gap_b = _random_gap(sides[1], frame_h, frame_w, rng)
            gap_a["marker"] = int(incoming_marker)
            gap_b["marker"] = int(outgoing_marker)
            path = _random_path(inner_h, inner_w, gap_a["anchor"], gap_b["anchor"], min_turns, rng)
            if path is None:
                continue
            gaps = [gap_a, gap_b]
            free_endpoint = None

        else:
            gap_a = _random_gap(sides[0], frame_h, frame_w, rng)
            gap_b = _random_gap(sides[1], frame_h, frame_w, rng)
            gap_a["marker"] = int(incoming_marker)
            gap_b["marker"] = None
            path = _random_path(inner_h, inner_w, gap_a["anchor"], gap_b["anchor"], min_turns, rng)
            if path is None:
                continue
            gaps = [gap_a, gap_b]
            free_endpoint = None

        patch = np.full((frame_h + 2, frame_w + 2), BG, dtype=int)
        patch[1 : 1 + frame_h, 1] = FRAME
        patch[1 : 1 + frame_h, frame_w] = FRAME
        patch[1, 1 : 1 + frame_w] = FRAME
        patch[frame_h, 1 : 1 + frame_w] = FRAME

        for gap in gaps:
            gr, gc = _gap_border_cell(gap["side"], int(gap["position"]), frame_h, frame_w)
            patch[int(1 + gr), int(1 + gc)] = BG
            if gap["marker"] is not None:
                dr, dc = gap["dir"]
                patch[int(1 + gr + dr), int(1 + gc + dc)] = int(gap["marker"])

        for r, c in path:
            patch[int(2 + r), int(2 + c)] = int(path_color)

        frame_info = _frame_info_from_path(
            path=path,
            path_color=int(path_color),
            gaps=gaps,
            free_endpoint=free_endpoint,
            bbox_key=(int(frame_index), 0),
        )
        return {"patch": patch, "frame_info": frame_info}

    return None


def _split_nonempty(items: list[int], groups: int, rng: random.Random) -> list[list[int]]:
    if groups == 1:
        return [items]
    cuts = sorted(rng.sample(range(1, len(items)), groups - 1))
    out = []
    start = 0
    for cut in cuts + [len(items)]:
        out.append(items[start:cut])
        start = cut
    return out


def _layout_patches(patches: list[np.ndarray], rng: random.Random) -> np.ndarray:
    order = list(range(len(patches)))
    rng.shuffle(order)

    if len(patches) >= 4:
        row_count = rng.randint(2, min(3, len(patches)))
    else:
        row_count = rng.randint(1, min(2, len(patches)))
    rows = _split_nonempty(order, row_count, rng)

    row_specs = []
    max_row_width = 0
    total_height = 0
    row_gaps = [rng.randint(1, 3) for _ in range(max(0, row_count - 1))]

    for row in rows:
        gaps = [rng.randint(1, 3) for _ in range(max(0, len(row) - 1))]
        row_width = sum(int(patches[idx].shape[1]) for idx in row) + sum(gaps)
        row_height = max(int(patches[idx].shape[0]) for idx in row)
        row_specs.append((row, gaps, row_width, row_height))
        max_row_width = max(max_row_width, int(row_width))
        total_height += int(row_height)

    total_height += sum(row_gaps)
    left = rng.randint(1, 3)
    right = rng.randint(1, 3)
    top = rng.randint(1, 3)
    bottom = rng.randint(1, 3)
    slack_w = rng.randint(0, 4)

    grid_h = int(top + total_height + bottom)
    grid_w = int(left + max_row_width + right + slack_w)
    canvas = np.full((grid_h, grid_w), BG, dtype=int)

    y = int(top)
    for row_index, (row, gaps, row_width, row_height) in enumerate(row_specs):
        x = int(left + rng.randint(0, max_row_width - row_width + slack_w))
        for item_index, patch_index in enumerate(row):
            patch = patches[patch_index]
            ph, pw = map(int, patch.shape)
            y_offset = int(y + rng.randint(0, row_height - ph))
            mask = patch != BG
            canvas[y_offset : y_offset + ph, x : x + pw][mask] = patch[mask]
            x += pw
            if item_index < len(gaps):
                x += gaps[item_index]
        y += row_height
        if row_index < len(row_gaps):
            y += row_gaps[row_index]

    return canvas


def _gen_attempt(rng: random.Random) -> np.ndarray | None:
    n_frames = rng.randint(*FRAME_COUNT_RANGE)
    connector_count = int(n_frames - 1)
    path_palette_size = rng.randint(2, min(3, len(ALL_COLORS) - connector_count))
    path_palette = rng.sample(list(ALL_COLORS), path_palette_size)
    marker_pool = [color for color in ALL_COLORS if color not in path_palette]
    marker_colors = rng.sample(marker_pool, connector_count)
    path_colors = [rng.choice(path_palette) for _ in range(n_frames)]

    built_frames = []
    for index in range(n_frames):
        if index == 0:
            frame = _build_frame(
                role="start",
                path_color=path_colors[index],
                incoming_marker=None,
                outgoing_marker=marker_colors[index],
                frame_index=index,
                rng=rng,
            )
        elif index == n_frames - 1:
            frame = _build_frame(
                role="end",
                path_color=path_colors[index],
                incoming_marker=marker_colors[index - 1],
                outgoing_marker=None,
                frame_index=index,
                rng=rng,
            )
        else:
            frame = _build_frame(
                role="internal",
                path_color=path_colors[index],
                incoming_marker=marker_colors[index - 1],
                outgoing_marker=marker_colors[index],
                frame_index=index,
                rng=rng,
            )
        if frame is None:
            return None
        built_frames.append(frame)

    frame_infos = [frame["frame_info"] for frame in built_frames]
    try:
        expected = _assemble_from_frames(frame_infos)
    except AssertionError:
        return None

    if expected.shape[0] > MAX_OUTPUT_SIDE or expected.shape[1] > MAX_OUTPUT_SIDE:
        return None

    grid = _layout_patches([frame["patch"] for frame in built_frames], rng)
    if int(np.count_nonzero(grid == BG)) <= int(np.count_nonzero(grid != BG)):
        return None

    try:
        produced = main(grid)
    except AssertionError:
        return None

    if not np.array_equal(produced, expected):
        return None

    return grid


def generate_input() -> np.ndarray:
    rng = random.Random()
    for _ in range(GEN_RETRIES):
        grid = _gen_attempt(rng)
        if grid is not None:
            return grid
    raise RuntimeError("generate_input: exhausted attempts")
