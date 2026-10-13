from __future__ import annotations

from collections import deque
import random

import numpy as np


BLACK = 0
BLUE = 1
RED = 2
YELLOW = 4
GREY = 5
TEAL = 8

GEN_ATTEMPTS = 500
ODD_SIZES_SMALL = (5, 7)
ODD_SIZES_MEDIUM = (5, 7, 9)
ODD_SIZES_ALL = (5, 7, 9, 11)


def _neighbors4(r: int, c: int):
    yield r - 1, c
    yield r + 1, c
    yield r, c - 1
    yield r, c + 1


def _components_mask(mask: np.ndarray) -> list[np.ndarray]:
    h, w = mask.shape
    seen = np.zeros((h, w), dtype=bool)
    comps: list[np.ndarray] = []
    for r in range(h):
        for c in range(w):
            if not mask[r, c] or seen[r, c]:
                continue
            q = deque([(r, c)])
            seen[r, c] = True
            coords = []
            while q:
                rr, cc = q.popleft()
                coords.append((rr, cc))
                for nr, nc in _neighbors4(rr, cc):
                    if 0 <= nr < h and 0 <= nc < w and mask[nr, nc] and not seen[nr, nc]:
                        seen[nr, nc] = True
                        q.append((nr, nc))
            comps.append(np.array(coords, dtype=int))
    return comps


def _bbox_coords(coords: np.ndarray) -> tuple[int, int, int, int]:
    r0, c0 = coords.min(axis=0)
    r1, c1 = coords.max(axis=0)
    return int(r0), int(c0), int(r1), int(c1)


def _frame_cells(bbox: tuple[int, int, int, int]) -> set[tuple[int, int]]:
    r0, c0, r1, c1 = bbox
    cells = set()
    for c in range(c0, c1 + 1):
        cells.add((r0, c))
        cells.add((r1, c))
    for r in range(r0, r1 + 1):
        cells.add((r, c0))
        cells.add((r, c1))
    return cells


def _frame_from_center(center: tuple[int, int], size: int) -> dict:
    cr, cc = center
    half = size // 2
    bbox = (cr - half, cc - half, cr + half, cc + half)
    return {
        "center": (int(cr), int(cc)),
        "size": int(size),
        "bbox": bbox,
    }


def _bridge_cells(frame_a: dict, frame_b: dict) -> list[tuple[int, int]]:
    (ra, ca), (rb, cb) = frame_a["center"], frame_b["center"]
    a0, a1, a2, a3 = frame_a["bbox"]
    b0, b1, b2, b3 = frame_b["bbox"]
    if ra == rb:
        if ca > cb:
            frame_a, frame_b = frame_b, frame_a
            (ra, ca), (rb, cb) = frame_a["center"], frame_b["center"]
            a0, a1, a2, a3 = frame_a["bbox"]
            b0, b1, b2, b3 = frame_b["bbox"]
        row = ra
        return [(row, c) for c in range(a3 + 1, b1)]
    if ca == cb:
        if ra > rb:
            frame_a, frame_b = frame_b, frame_a
            (ra, ca), (rb, cb) = frame_a["center"], frame_b["center"]
            a0, a1, a2, a3 = frame_a["bbox"]
            b0, b1, b2, b3 = frame_b["bbox"]
        col = ca
        return [(r, col) for r in range(a2 + 1, b0)]
    raise ValueError("bridged frames must share a row or column center")


def _extract_frames(grid: np.ndarray) -> tuple[list[dict], np.ndarray]:
    g = np.asarray(grid, dtype=int)
    frames = []
    frame_map = np.full(g.shape, -1, dtype=int)
    for coords in _components_mask(g == GREY):
        r0, c0, r1, c1 = _bbox_coords(coords)
        h = r1 - r0 + 1
        w = c1 - c0 + 1
        if h != w or h < 5:
            raise AssertionError("every grey component must be a square frame")
        sub = g[r0 : r1 + 1, c0 : c1 + 1] == GREY
        perimeter = np.zeros((h, w), dtype=bool)
        perimeter[0, :] = True
        perimeter[-1, :] = True
        perimeter[:, 0] = True
        perimeter[:, -1] = True
        if not np.array_equal(sub, perimeter):
            raise AssertionError("grey component must be a 1-pixel-thick hollow frame")
        idx = len(frames)
        frame_map[r0 : r1 + 1, c0 : c1 + 1][perimeter] = idx
        frames.append(
            {
                "id": idx,
                "bbox": (r0, c0, r1, c1),
                "size": h,
                "center": ((r0 + r1) // 2, (c0 + c1) // 2),
            }
        )
    if not frames:
        raise AssertionError("expected at least one grey frame")
    return frames, frame_map


def _bridge_endpoint_frame_ids(cell: tuple[int, int], frame_map: np.ndarray) -> set[int]:
    h, w = frame_map.shape
    out = set()
    r, c = cell
    for nr, nc in _neighbors4(r, c):
        if 0 <= nr < h and 0 <= nc < w and frame_map[nr, nc] >= 0:
            out.add(int(frame_map[nr, nc]))
    return out


def _extract_bridges(grid: np.ndarray, frame_map: np.ndarray) -> list[dict]:
    g = np.asarray(grid, dtype=int)
    bridges = []
    for coords in _components_mask(g == TEAL):
        rows = sorted(set(int(v) for v in coords[:, 0]))
        cols = sorted(set(int(v) for v in coords[:, 1]))
        if (len(rows) == 1) == (len(cols) == 1):
            raise AssertionError("each bridge must be a non-empty straight segment")
        if len(rows) == 1:
            row = rows[0]
            c0 = int(coords[:, 1].min())
            c1 = int(coords[:, 1].max())
            if coords.shape[0] != c1 - c0 + 1:
                raise AssertionError("horizontal bridge must be contiguous")
            endpoints = [(row, c0), (row, c1)]
        else:
            col = cols[0]
            r0 = int(coords[:, 0].min())
            r1 = int(coords[:, 0].max())
            if coords.shape[0] != r1 - r0 + 1:
                raise AssertionError("vertical bridge must be contiguous")
            endpoints = [(r0, col), (r1, col)]

        left_ids = _bridge_endpoint_frame_ids(endpoints[0], frame_map)
        right_ids = _bridge_endpoint_frame_ids(endpoints[1], frame_map)
        if len(left_ids) != 1 or len(right_ids) != 1:
            raise AssertionError("each bridge endpoint must touch exactly one frame")
        a = next(iter(left_ids))
        b = next(iter(right_ids))
        if a == b:
            raise AssertionError("bridge cannot connect a frame to itself")
        bridges.append({"coords": coords, "edge": tuple(sorted((a, b)))})
    return bridges


def _component_sizes(frame_count: int, edges: list[tuple[int, int]]) -> dict[int, int]:
    graph = {idx: set() for idx in range(frame_count)}
    for a, b in edges:
        graph[a].add(b)
        graph[b].add(a)

    sizes: dict[int, int] = {}
    seen = set()
    for start in range(frame_count):
        if start in seen:
            continue
        stack = [start]
        comp = []
        seen.add(start)
        while stack:
            node = stack.pop()
            comp.append(node)
            for nxt in graph[node]:
                if nxt not in seen:
                    seen.add(nxt)
                    stack.append(nxt)
        size = len(comp)
        for node in comp:
            sizes[node] = size
    return sizes


def _analyze_scene(grid: np.ndarray) -> dict:
    frames, frame_map = _extract_frames(grid)
    bridges = _extract_bridges(grid, frame_map)
    edges = sorted({bridge["edge"] for bridge in bridges})
    comp_sizes = _component_sizes(len(frames), edges)
    return {
        "frames": frames,
        "frame_map": frame_map,
        "bridges": bridges,
        "edges": edges,
        "component_sizes": comp_sizes,
    }


def _fill_concentric(out: np.ndarray, bbox: tuple[int, int, int, int]) -> None:
    r0, c0, r1, c1 = bbox
    for r in range(r0 + 1, r1):
        for c in range(c0 + 1, c1):
            dist = min(r - (r0 + 1), (r1 - 1) - r, c - (c0 + 1), (c1 - 1) - c)
            out[r, c] = RED if dist % 2 == 0 else BLUE


def _fill_solid(out: np.ndarray, bbox: tuple[int, int, int, int], color: int) -> None:
    r0, c0, r1, c1 = bbox
    out[r0 + 1 : r1, c0 + 1 : c1] = int(color)


def main(grid: np.ndarray) -> np.ndarray:
    """
    Use the cyan bridge network to decide which grey square frames get concentric filling.

    Input structure:
    - The grid has a black background, 4-6 grey one-pixel-thick hollow square frames, and some straight cyan bridge segments.
    - Each bridge is a horizontal or vertical line outside the frames and touches one frame at each endpoint.
    - Frames do not overlap, and the bridge network encodes which frames belong to the same connected component.

    Rule:
    - Treat every grey frame as a graph node and every cyan bridge as an undirected edge between the two endpoint-touching frames.
    - Keep every grey border unchanged, erase every cyan bridge to black, and inspect the size of each frame's connected component.
    - If a frame belongs to a component of size at least 2, fill its strict interior by alternating red then blue concentric Manhattan layers starting from the inner wall.
    - If a frame is isolated, fill its strict interior uniformly yellow.

    Key insight:
    - The cyan bridges disappear in the output; their only purpose is to encode graph connectivity, which alone determines whether a frame gets concentric red/blue layers or a solid yellow interior.
    """
    g = np.asarray(grid, dtype=int)
    scene = _analyze_scene(g)

    out = g.copy()
    out[out == TEAL] = BLACK
    for frame in scene["frames"]:
        bbox = frame["bbox"]
        if scene["component_sizes"][frame["id"]] >= 2:
            _fill_concentric(out, bbox)
        else:
            _fill_solid(out, bbox, YELLOW)
    return out


def _sequence_for_group(kind: str) -> list[str]:
    if kind == "chain_h3":
        return ["R", "R"]
    if kind == "chain_v3":
        return ["D", "D"]
    if kind == "corner3":
        return random.choice((["R", "D"], ["D", "R"]))
    if kind == "chain_h4":
        return ["R", "R", "R"]
    if kind == "chain_v4":
        return ["D", "D", "D"]
    if kind == "corner4":
        return random.choice((["R", "D", "R"], ["D", "R", "D"]))
    if kind == "pair_h":
        return ["R"]
    if kind == "pair_v":
        return ["D"]
    if kind == "solo":
        return []
    raise ValueError(f"unknown group kind: {kind}")


def _sizes_for_group(frame_count: int) -> tuple[int, ...]:
    if frame_count >= 4:
        pool = ODD_SIZES_SMALL
    elif frame_count == 3:
        pool = ODD_SIZES_MEDIUM
    elif frame_count == 2:
        pool = ODD_SIZES_MEDIUM
    else:
        pool = ODD_SIZES_ALL
    return tuple(int(random.choice(pool)) for _ in range(frame_count))


def _gaps_for_group(frame_count: int) -> tuple[int, ...]:
    if frame_count >= 4:
        return tuple(int(random.randint(2, 3)) for _ in range(frame_count - 1))
    return tuple(int(random.randint(2, 4)) for _ in range(frame_count - 1))


def _build_group(kind: str) -> dict:
    moves = _sequence_for_group(kind)
    frame_count = len(moves) + 1
    sizes = _sizes_for_group(frame_count)
    gaps = _gaps_for_group(frame_count)

    centers = [(sizes[0] // 2, sizes[0] // 2)]
    for i, move in enumerate(moves):
        prev_r, prev_c = centers[-1]
        prev_half = sizes[i] // 2
        next_half = sizes[i + 1] // 2
        delta = prev_half + gaps[i] + next_half + 1
        if move == "R":
            centers.append((prev_r, prev_c + delta))
        elif move == "D":
            centers.append((prev_r + delta, prev_c))
        else:
            raise ValueError("unsupported move")

    frames = [_frame_from_center(center, size) for center, size in zip(centers, sizes)]
    edges = [(i, i + 1) for i in range(frame_count - 1)]
    cells = set()
    for frame in frames:
        cells.update(_frame_cells(frame["bbox"]))
    for a, b in edges:
        bridge = _bridge_cells(frames[a], frames[b])
        if not bridge:
            raise AssertionError("bridge must have at least one pixel")
        cells.update(bridge)

    min_r = min(r for r, _ in cells)
    min_c = min(c for _, c in cells)
    norm_frames = []
    for frame in frames:
        r0, c0, r1, c1 = frame["bbox"]
        cr, cc = frame["center"]
        norm_frames.append(
            {
                "center": (cr - min_r, cc - min_c),
                "size": frame["size"],
                "bbox": (r0 - min_r, c0 - min_c, r1 - min_r, c1 - min_c),
            }
        )
    norm_cells = {(r - min_r, c - min_c) for r, c in cells}
    max_r = max(r for r, _ in norm_cells)
    max_c = max(c for _, c in norm_cells)
    return {
        "kind": kind,
        "frames": norm_frames,
        "edges": edges,
        "cells": norm_cells,
        "shape": (max_r + 1, max_c + 1),
    }


def _bbox_gap(b1: tuple[int, int, int, int], b2: tuple[int, int, int, int]) -> tuple[int, int]:
    r0, c0, r1, c1 = b1
    rr0, cc0, rr1, cc1 = b2
    row_gap = max(rr0 - r1 - 1, r0 - rr1 - 1, 0)
    col_gap = max(cc0 - c1 - 1, c0 - cc1 - 1, 0)
    return row_gap, col_gap


def _place_groups(h: int, w: int, groups: list[dict]) -> tuple[list[dict], list[tuple[int, int]]] | None:
    placed_groups = []
    placed_frame_bboxes: list[tuple[int, int, int, int]] = []
    occupied = set()
    halo = set()

    for group in sorted(groups, key=lambda g: g["shape"][0] * g["shape"][1], reverse=True):
        gh, gw = group["shape"]
        if gh > h or gw > w:
            return None
        positions = [(r0, c0) for r0 in range(h - gh + 1) for c0 in range(w - gw + 1)]
        random.shuffle(positions)
        chosen = None
        for r_shift, c_shift in positions:
            shifted_cells = {(r + r_shift, c + c_shift) for r, c in group["cells"]}
            if any(cell in halo for cell in shifted_cells):
                continue
            shifted_bboxes = []
            ok = True
            for frame in group["frames"]:
                r0, c0, r1, c1 = frame["bbox"]
                shifted = (r0 + r_shift, c0 + c_shift, r1 + r_shift, c1 + c_shift)
                for other in placed_frame_bboxes:
                    row_gap, col_gap = _bbox_gap(shifted, other)
                    if row_gap < 2 and col_gap < 2:
                        ok = False
                        break
                if not ok:
                    break
                shifted_bboxes.append(shifted)
            if ok:
                chosen = (r_shift, c_shift, shifted_cells, shifted_bboxes)
                break
        if chosen is None:
            return None

        r_shift, c_shift, shifted_cells, shifted_bboxes = chosen
        translated_frames = []
        for frame in group["frames"]:
            r0, c0, r1, c1 = frame["bbox"]
            cr, cc = frame["center"]
            translated_frames.append(
                {
                    "center": (cr + r_shift, cc + c_shift),
                    "size": frame["size"],
                    "bbox": (r0 + r_shift, c0 + c_shift, r1 + r_shift, c1 + c_shift),
                }
            )
        placed_groups.append({"frames": translated_frames, "edges": group["edges"]})
        occupied.update(shifted_cells)
        for r, c in shifted_cells:
            for dr in (-1, 0, 1):
                for dc in (-1, 0, 1):
                    nr = r + dr
                    nc = c + dc
                    if 0 <= nr < h and 0 <= nc < w:
                        halo.add((nr, nc))
        placed_frame_bboxes.extend(shifted_bboxes)

    global_edges = []
    offset = 0
    all_groups = []
    for placed in placed_groups:
        all_groups.append(placed)
        for a, b in placed["edges"]:
            global_edges.append((a + offset, b + offset))
        offset += len(placed["frames"])
    return all_groups, global_edges


def _paint_group(grid: np.ndarray, frames: list[dict], edges: list[tuple[int, int]]) -> None:
    for frame in frames:
        r0, c0, r1, c1 = frame["bbox"]
        grid[r0, c0 : c1 + 1] = GREY
        grid[r1, c0 : c1 + 1] = GREY
        grid[r0 : r1 + 1, c0] = GREY
        grid[r0 : r1 + 1, c1] = GREY
    for a, b in edges:
        for r, c in _bridge_cells(frames[a], frames[b]):
            grid[r, c] = TEAL


def _group_plan(total_frames: int) -> list[str]:
    main_size = random.choice((3, 4))
    if main_size >= total_frames:
        main_size = total_frames - 1
    main_size = max(main_size, 3)

    if main_size == 3:
        kinds = ["chain_h3", "chain_v3", "corner3"]
    else:
        kinds = ["chain_h4", "chain_v4", "corner4"]
    plan = [random.choice(kinds)]

    remaining = total_frames - main_size
    if remaining >= 2 and random.random() < 0.45:
        plan.append(random.choice(["pair_h", "pair_v"]))
        remaining -= 2
    plan.extend(["solo"] * remaining)
    tail = plan[1:]
    random.shuffle(tail)
    plan[1:] = tail
    return plan


def _validate_generated_scene(inp: np.ndarray, out: np.ndarray) -> None:
    scene = _analyze_scene(inp)
    sizes = list(scene["component_sizes"].values())
    if max(sizes) < 3:
        raise AssertionError("generator must include a component with at least three frames")
    if 1 not in sizes:
        raise AssertionError("generator must include an isolated frame")
    if not scene["bridges"]:
        raise AssertionError("generator must include at least one bridge")

    if np.any(out == TEAL):
        raise AssertionError("bridges must be erased in the output")

    for frame in scene["frames"]:
        bbox = frame["bbox"]
        r0, c0, r1, c1 = bbox
        if not np.all(out[r0, c0 : c1 + 1] == GREY):
            raise AssertionError("top border changed")
        if not np.all(out[r1, c0 : c1 + 1] == GREY):
            raise AssertionError("bottom border changed")
        if not np.all(out[r0 : r1 + 1, c0] == GREY):
            raise AssertionError("left border changed")
        if not np.all(out[r0 : r1 + 1, c1] == GREY):
            raise AssertionError("right border changed")

        interior = out[r0 + 1 : r1, c0 + 1 : c1]
        comp_size = scene["component_sizes"][frame["id"]]
        if comp_size == 1:
            if not np.all(interior == YELLOW):
                raise AssertionError("isolated frame interior must be solid yellow")
        else:
            for r in range(r0 + 1, r1):
                for c in range(c0 + 1, c1):
                    dist = min(r - (r0 + 1), (r1 - 1) - r, c - (c0 + 1), (c1 - 1) - c)
                    expected = RED if dist % 2 == 0 else BLUE
                    if int(out[r, c]) != expected:
                        raise AssertionError("connected frame interior must follow concentric parity")


def _attempt_generate() -> np.ndarray | None:
    h = random.randint(22, 29)
    w = random.randint(22, 29)
    total_frames = random.randint(4, 6)
    plan = _group_plan(total_frames)
    groups = [_build_group(kind) for kind in plan]

    placed = _place_groups(h, w, groups)
    if placed is None:
        return None

    placed_groups, _ = placed
    grid = np.full((h, w), BLACK, dtype=int)
    for group in placed_groups:
        _paint_group(grid, group["frames"], group["edges"])

    try:
        out = main(grid)
        _validate_generated_scene(grid, out)
    except AssertionError:
        return None
    return grid


def generate_input() -> np.ndarray:
    for _ in range(GEN_ATTEMPTS):
        grid = _attempt_generate()
        if grid is not None:
            return grid
    raise RuntimeError("failed to generate a valid bridge-driven concentric-fill task")
