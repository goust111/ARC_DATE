from __future__ import annotations

import sys
from pathlib import Path

import numpy as np


def _ensure_arc_utils_importable() -> None:
    here = Path(__file__).resolve()
    for parent in [here.parent, *here.parents]:
        candidate = parent / "new" / "solvers"
        if candidate.is_dir():
            path = str(candidate)
            if path not in sys.path:
                sys.path.append(path)
            return
    fallback = Path("/home/lixinyu/snap/new/solvers")
    if fallback.is_dir():
        path = str(fallback)
        if path not in sys.path:
            sys.path.append(path)


_ensure_arc_utils_importable()

from arc_utils import Color, components_mask, find_components, get_bg, in_bounds, neighbors4


GREY = int(Color.GREY)
BG = int(Color.BLACK)
PALETTE = [int(c) for c in Color.NOT_BLACK if int(c) != GREY]
ORTHO_DIRS = [(-1, 0), (1, 0), (0, -1), (0, 1)]
MAX_GENERATION_ATTEMPTS = 400


def _add(p: tuple[int, int], d: tuple[int, int], k: int = 1) -> tuple[int, int]:
    return int(p[0] + d[0] * k), int(p[1] + d[1] * k)


def _frame_mask(shape: tuple[int, int], r0: int, c0: int, side: int) -> np.ndarray:
    mask = np.zeros(shape, dtype=bool)
    r1 = r0 + side - 1
    c1 = c0 + side - 1
    mask[r0, c0 : c1 + 1] = True
    mask[r1, c0 : c1 + 1] = True
    mask[r0 : r1 + 1, c0] = True
    mask[r0 : r1 + 1, c1] = True
    return mask


def _render_nested(colors: list[int]) -> np.ndarray:
    side = 2 * len(colors) - 1
    patch = np.full((side, side), int(colors[0]), dtype=int)
    for layer, color in enumerate(colors[1:], start=1):
        patch[layer : side - layer, layer : side - layer] = int(color)
    return patch


def _render_output(
    shape: tuple[int, int],
    frame_bbox: tuple[int, int, int, int],
    ordered_colors: list[int],
    bg: int = BG,
) -> np.ndarray:
    out = np.full(shape, int(bg), dtype=int)
    r0, c0, r1, c1 = frame_bbox
    out[r0, c0 : c1 + 1] = GREY
    out[r1, c0 : c1 + 1] = GREY
    out[r0 : r1 + 1, c0] = GREY
    out[r0 : r1 + 1, c1] = GREY
    inner = _render_nested(ordered_colors)
    out[r0 + 1 : r1, c0 + 1 : c1] = inner
    return out


def _blob_components(grid: np.ndarray, bg: int) -> list[dict]:
    return [
        comp
        for comp in find_components(grid, bg=bg, connectivity=4, monochromatic=True)
        if int(comp["color"]) not in {int(bg), GREY}
    ]


def _candidate_wires(
    grid: np.ndarray,
    frame_mask: np.ndarray,
    blob_cell_to_id: dict[tuple[int, int], int],
) -> list[dict]:
    frame_set = set(map(tuple, np.argwhere(frame_mask)))
    wire_mask = (grid == GREY) & ~frame_mask
    wires: list[dict] = []
    for coords in components_mask(wire_mask, connectivity=4):
        comp_set = {tuple(map(int, cell)) for cell in coords}
        touches_frame: set[tuple[int, int]] = set()
        blob_ids: set[int] = set()
        for r, c in comp_set:
            for nr, nc in neighbors4(r, c):
                nbr = (int(nr), int(nc))
                if nbr in frame_set:
                    touches_frame.add(nbr)
                if nbr in blob_cell_to_id:
                    blob_ids.add(blob_cell_to_id[nbr])
        if touches_frame and len(blob_ids) == 1:
            blob_id = next(iter(blob_ids))
            wires.append(
                {
                    "blob_id": int(blob_id),
                    "length": int(coords.shape[0]),
                    "touches_frame": touches_frame,
                    "coords": coords,
                }
            )
    return wires


def _parse_instance(grid: np.ndarray) -> dict:
    g = np.asarray(grid, dtype=int)
    bg = int(get_bg(g))
    blobs = _blob_components(g, bg)
    assert blobs, "expected at least one colored blob"
    n = len(blobs)
    frame_side = 2 * n + 1
    H, W = g.shape
    assert frame_side <= min(H, W), "frame size implied by blob count does not fit the grid"

    blob_cell_to_id: dict[tuple[int, int], int] = {}
    for idx, blob in enumerate(blobs):
        for r, c in blob["coords"]:
            blob_cell_to_id[(int(r), int(c))] = int(idx)

    candidates: list[dict] = []
    for r0 in range(H - frame_side + 1):
        for c0 in range(W - frame_side + 1):
            sub = g[r0 : r0 + frame_side, c0 : c0 + frame_side]
            if not (
                np.all(sub[0, :] == GREY)
                and np.all(sub[-1, :] == GREY)
                and np.all(sub[:, 0] == GREY)
                and np.all(sub[:, -1] == GREY)
            ):
                continue
            if np.any(sub[1:-1, 1:-1] != bg):
                continue

            mask = _frame_mask(g.shape, r0, c0, frame_side)
            wires = _candidate_wires(g, mask, blob_cell_to_id)
            if len(wires) != n:
                continue
            if {wire["blob_id"] for wire in wires} != set(range(n)):
                continue
            lengths = [int(wire["length"]) for wire in wires]
            if len(lengths) != len(set(lengths)):
                continue
            if int(np.count_nonzero((g == GREY) & ~mask)) != sum(lengths):
                continue

            center_offset = abs((r0 + frame_side / 2) - (H / 2)) + abs((c0 + frame_side / 2) - (W / 2))
            candidates.append(
                {
                    "frame_bbox": (r0, c0, r0 + frame_side - 1, c0 + frame_side - 1),
                    "frame_mask": mask,
                    "wires": wires,
                    "center_offset": float(center_offset),
                    "blobs": blobs,
                    "bg": bg,
                }
            )

    assert candidates, "failed to locate the central grey frame"
    candidates.sort(key=lambda item: (item["center_offset"], item["frame_bbox"]))
    return candidates[0]


def main(input_grid: np.ndarray) -> np.ndarray:
    """
    Reorder blob colors by wire length and repaint them as concentric square rings inside the central frame.

    Input structure:
    - The background is black, and a grey hollow square frame sits near the center with an empty interior.
    - Outside the frame are several monochromatic connected blobs, each attached to the frame by a one-cell-wide grey wire.
    - The frame's interior side length is exactly `2 * blob_count - 1`, while blob areas and wire lengths are intentionally unrelated.

    Rule:
    - Measure each wire by counting its grey pixels outside the frame.
    - Sort the blob colors from longest wire to shortest wire.
    - Erase every outer blob and wire, keep the grey frame, and fill the frame interior with concentric square rings whose outer-to-inner order follows that wire-length ranking.

    Key insight:
    - The ordering key is the actual routed wire length, not blob size and not straight-line proximity to the frame.
    """
    parsed = _parse_instance(np.asarray(input_grid, dtype=int))
    ordered_wires = sorted(parsed["wires"], key=lambda wire: int(wire["length"]), reverse=True)
    ordered_colors = [int(parsed["blobs"][wire["blob_id"]]["color"]) for wire in ordered_wires]
    return _render_output(input_grid.shape, parsed["frame_bbox"], ordered_colors, parsed["bg"])


def _attachment_candidates(frame_bbox: tuple[int, int, int, int]) -> list[dict]:
    r0, c0, r1, c1 = frame_bbox
    candidates: list[dict] = []
    for c in range(c0 + 1, c1, 2):
        candidates.append({"attach": (r0, c), "outward": (-1, 0), "tangents": ((0, -1), (0, 1))})
        candidates.append({"attach": (r1, c), "outward": (1, 0), "tangents": ((0, -1), (0, 1))})
    for r in range(r0 + 1, r1, 2):
        candidates.append({"attach": (r, c0), "outward": (0, -1), "tangents": ((-1, 0), (1, 0))})
        candidates.append({"attach": (r, c1), "outward": (0, 1), "tangents": ((-1, 0), (1, 0))})
    return candidates


def _path_clear(
    path: list[tuple[int, int]],
    shape: tuple[int, int],
    occupied: set[tuple[int, int]],
    attachment: tuple[int, int],
) -> bool:
    path_set = set(path)
    if len(path_set) != len(path):
        return False
    for idx, cell in enumerate(path):
        if not in_bounds(shape, cell[0], cell[1]):
            return False
        if cell in occupied:
            return False
        prev_cell = path[idx - 1] if idx > 0 else None
        next_cell = path[idx + 1] if idx + 1 < len(path) else None
        allowed_occ = {attachment} if idx == 0 else set()
        for nr, nc in neighbors4(*cell):
            nbr = (int(nr), int(nc))
            if nbr in path_set and nbr not in {prev_cell, next_cell}:
                return False
            if nbr in occupied and nbr not in allowed_occ:
                return False
    return True


def _blob_cell_clear(
    cell: tuple[int, int],
    shape: tuple[int, int],
    occupied: set[tuple[int, int]],
    current: set[tuple[int, int]],
    allowed_occ: set[tuple[int, int]],
) -> bool:
    if not in_bounds(shape, cell[0], cell[1]):
        return False
    if cell in occupied or cell in current:
        return False
    for nr, nc in neighbors4(*cell):
        nbr = (int(nr), int(nc))
        if nbr in current:
            continue
        if nbr in occupied and nbr not in allowed_occ:
            return False
    return True


def _grow_blob(
    endpoint: tuple[int, int],
    back_dir: tuple[int, int],
    area: int,
    shape: tuple[int, int],
    occupied: set[tuple[int, int]],
    rng: np.random.Generator,
) -> set[tuple[int, int]] | None:
    try_dirs = list(ORTHO_DIRS)
    rng.shuffle(try_dirs)
    banned_anchor = (-back_dir[0], -back_dir[1])
    for direction in try_dirs:
        if direction == banned_anchor:
            continue
        anchor = _add(endpoint, direction)
        if not _blob_cell_clear(anchor, shape, occupied, set(), {endpoint}):
            continue
        blob = {anchor}
        while len(blob) < area:
            frontier: list[tuple[int, int]] = []
            for cell in blob:
                for nr, nc in neighbors4(*cell):
                    nxt = (int(nr), int(nc))
                    if nxt in frontier:
                        continue
                    if _blob_cell_clear(nxt, shape, occupied, blob, set()):
                        frontier.append(nxt)
            if not frontier:
                break
            scores = []
            for cand in frontier:
                contact = sum((nbr in blob) for nbr in neighbors4(*cand))
                spread = abs(cand[0] - anchor[0]) + abs(cand[1] - anchor[1])
                scores.append((contact, -spread, float(rng.random()), cand))
            scores.sort(reverse=True)
            blob.add(scores[0][3])
        if len(blob) == area:
            return blob
    return None


def _wire_direct_distance(blob_cells: set[tuple[int, int]], frame_cells: set[tuple[int, int]]) -> int:
    return min(abs(br - fr) + abs(bc - fc) for br, bc in blob_cells for fr, fc in frame_cells)


def _perm_not_matching_sorted_rank(n: int, rng: np.random.Generator) -> list[int]:
    base = list(range(n))
    for _ in range(50):
        perm = list(rng.permutation(n))
        if any(perm[i] != base[i] for i in range(n)):
            return perm
    return list(reversed(base))


def _make_case(rng: np.random.Generator) -> tuple[np.ndarray, dict] | None:
    n = int(rng.integers(2, 6))
    frame_side = 2 * n + 1
    min_h = max(18, frame_side + 8)
    min_w = max(18, frame_side + 8)
    H = int(rng.integers(min_h, 31))
    W = int(rng.integers(min_w, 31))

    r0_center = H // 2 - frame_side // 2
    c0_center = W // 2 - frame_side // 2
    r0 = int(np.clip(r0_center + int(rng.integers(-2, 3)), 2, H - frame_side - 2))
    c0 = int(np.clip(c0_center + int(rng.integers(-2, 3)), 2, W - frame_side - 2))
    frame_bbox = (r0, c0, r0 + frame_side - 1, c0 + frame_side - 1)

    grid = np.full((H, W), BG, dtype=int)
    frame_mask = _frame_mask((H, W), r0, c0, frame_side)
    grid[frame_mask] = GREY

    attachments = _attachment_candidates(frame_bbox)
    rng.shuffle(attachments)
    selected = attachments[:n]
    frame_cells = set(map(tuple, np.argwhere(frame_mask)))
    occupied = set(frame_cells)
    routes: list[dict] = []
    used_lengths: set[int] = set()

    for spec in selected:
        options: list[dict] = []
        for len1 in range(2, 7):
            for tangent in spec["tangents"]:
                for len2 in range(0, 6):
                    for len3 in range(0, 4):
                        if len2 == 0 and len3 == 0 and rng.random() < 0.75:
                            continue
                        attach = spec["attach"]
                        outward = spec["outward"]
                        seg1 = [_add(attach, outward, k) for k in range(1, len1 + 1)]
                        if not seg1:
                            continue
                        turn_start = seg1[-1]
                        seg2 = [_add(turn_start, tangent, k) for k in range(1, len2 + 1)]
                        seg3_start = seg2[-1] if seg2 else turn_start
                        seg3 = [_add(seg3_start, outward, k) for k in range(1, len3 + 1)]
                        path = seg1 + seg2 + seg3
                        if len(path) in used_lengths:
                            continue
                        if not _path_clear(path, grid.shape, occupied, attach):
                            continue
                        last_dir = tangent if seg2 else outward
                        options.append(
                            {
                                "attach": attach,
                                "outward": outward,
                                "path": path,
                                "length": len(path),
                                "last_dir": last_dir,
                                "bent": bool(seg2),
                            }
                        )
        if not options:
            return None
        choice = options[int(rng.integers(0, len(options)))]
        routes.append(choice)
        used_lengths.add(int(choice["length"]))
        occupied.update(choice["path"])
        for cell in choice["path"]:
            grid[cell] = GREY

    if not any(route["bent"] for route in routes):
        return None

    wire_order = sorted(range(n), key=lambda idx: int(routes[idx]["length"]), reverse=True)
    colors = [int(color) for color in rng.choice(PALETTE, size=n, replace=False)]
    sorted_areas = sorted(rng.choice(np.arange(2, 9), size=n, replace=False).tolist(), reverse=True)
    area_perm = _perm_not_matching_sorted_rank(n, rng)
    areas_by_route = [0] * n
    for rank, route_idx in enumerate(wire_order):
        areas_by_route[route_idx] = int(sorted_areas[area_perm[rank]])

    objects_meta: list[dict] = []
    for idx in rng.permutation(n):
        route = routes[int(idx)]
        blob = _grow_blob(route["path"][-1], route["last_dir"], areas_by_route[int(idx)], grid.shape, occupied, rng)
        if blob is None:
            return None
        color = colors[int(idx)]
        for cell in blob:
            grid[cell] = int(color)
        occupied.update(blob)
        objects_meta.append(
            {
                "color": int(color),
                "wire_length": int(route["length"]),
                "blob_cells": set(blob),
                "area": int(len(blob)),
            }
        )

    distances = [_wire_direct_distance(meta["blob_cells"], frame_cells) for meta in objects_meta]
    dist_order = sorted(range(n), key=lambda idx: distances[idx], reverse=True)
    area_order = sorted(range(n), key=lambda idx: objects_meta[idx]["area"], reverse=True)
    wire_order_meta = sorted(range(n), key=lambda idx: objects_meta[idx]["wire_length"], reverse=True)
    if area_order == wire_order_meta or dist_order == wire_order_meta:
        return None

    ordered_colors = [objects_meta[idx]["color"] for idx in wire_order_meta]
    expected = _render_output(grid.shape, frame_bbox, ordered_colors, BG)

    parsed = _parse_instance(grid)
    parsed_pairs = sorted(
        [(int(parsed["blobs"][wire["blob_id"]]["color"]), int(wire["length"])) for wire in parsed["wires"]],
        key=lambda item: item[1],
        reverse=True,
    )
    expected_pairs = sorted(
        [(int(meta["color"]), int(meta["wire_length"])) for meta in objects_meta],
        key=lambda item: item[1],
        reverse=True,
    )
    if parsed_pairs != expected_pairs:
        return None
    if not np.array_equal(main(grid), expected):
        return None

    meta = {
        "frame_bbox": frame_bbox,
        "objects": objects_meta,
        "expected": expected,
    }
    return grid, meta


def generate_input() -> np.ndarray:
    rng = np.random.default_rng()
    for _ in range(MAX_GENERATION_ATTEMPTS):
        case = _make_case(rng)
        if case is not None:
            return case[0]
    raise RuntimeError("failed to generate a valid non-degenerate ARC instance")
