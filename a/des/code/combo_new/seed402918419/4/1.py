import random
from collections import Counter
from itertools import combinations

import numpy as np


BG_COLOR = 0
FRAME_COLOR = 5
COLOR_POOL = [1, 2, 3, 4, 6, 7, 8, 9]
SIDES = ("top", "bottom", "left", "right")

BASE_SHAPES = {
    "hook_left": np.array([[1, 1, 1], [1, 0, 0]], dtype=bool),
    "tee": np.array([[1, 1, 1], [0, 1, 0]], dtype=bool),
    "hook_right": np.array([[1, 1, 1], [0, 0, 1]], dtype=bool),
}


def get_bg(grid: np.ndarray) -> int:
    vals, counts = np.unique(grid, return_counts=True)
    return int(vals[np.argmax(counts)])


def _components(mask: np.ndarray) -> list[np.ndarray]:
    h, w = mask.shape
    seen = np.zeros((h, w), dtype=bool)
    comps = []
    for r in range(h):
        for c in range(w):
            if not mask[r, c] or seen[r, c]:
                continue
            stack = [(r, c)]
            seen[r, c] = True
            pts = []
            while stack:
                rr, cc = stack.pop()
                pts.append((rr, cc))
                for dr, dc in ((1, 0), (-1, 0), (0, 1), (0, -1)):
                    nr = rr + dr
                    nc = cc + dc
                    if 0 <= nr < h and 0 <= nc < w and mask[nr, nc] and not seen[nr, nc]:
                        seen[nr, nc] = True
                        stack.append((nr, nc))
            comps.append(np.array(pts, dtype=int))
    return comps


def _bbox(coords: np.ndarray) -> tuple[int, int, int, int]:
    return (
        int(coords[:, 0].min()),
        int(coords[:, 1].min()),
        int(coords[:, 0].max()),
        int(coords[:, 1].max()),
    )


def _dock_width(mask: np.ndarray, side: str) -> int | None:
    coords = np.argwhere(mask)
    if side == "top":
        edge_r = int(coords[:, 0].max())
        edge = sorted(int(c) for r, c in coords if int(r) == edge_r)
    elif side == "bottom":
        edge_r = int(coords[:, 0].min())
        edge = sorted(int(c) for r, c in coords if int(r) == edge_r)
    elif side == "left":
        edge_c = int(coords[:, 1].max())
        edge = sorted(int(r) for r, c in coords if int(c) == edge_c)
    elif side == "right":
        edge_c = int(coords[:, 1].min())
        edge = sorted(int(r) for r, c in coords if int(c) == edge_c)
    else:
        raise ValueError(side)

    if not edge:
        return None
    width = int(edge[-1] - edge[0] + 1)
    if len(edge) != width:
        return None
    return width


def _build_shape_options() -> dict[str, list[dict]]:
    by_side = {side: [] for side in SIDES}
    for name, base in BASE_SHAPES.items():
        for side in SIDES:
            widths: dict[int, list[int]] = {}
            for k in range(4):
                width = _dock_width(np.rot90(base, k), side)
                if width is not None:
                    widths.setdefault(int(width), []).append(int(k))
            for width, rotations in widths.items():
                if len(rotations) == 1:
                    by_side[side].append(
                        {
                            "name": name,
                            "base_mask": base.copy(),
                            "dock_rotation": int(rotations[0]),
                            "gap_width": int(width),
                        }
                    )
    return by_side


SHAPE_OPTIONS = _build_shape_options()


def _analyze_frame(coords: np.ndarray, color: int) -> dict | None:
    r0, c0, r1, c1 = _bbox(coords)
    h = int(r1 - r0 + 1)
    w = int(c1 - c0 + 1)
    if h < 5 or w < 5:
        return None

    rel = np.zeros((h, w), dtype=bool)
    rel[coords[:, 0] - r0, coords[:, 1] - c0] = True
    if rel[1:-1, 1:-1].any():
        return None
    if not (rel[0, 0] and rel[0, -1] and rel[-1, 0] and rel[-1, -1]):
        return None

    candidates = []
    checks = {
        "top": rel[0, 1:-1],
        "bottom": rel[-1, 1:-1],
        "left": rel[1:-1, 0],
        "right": rel[1:-1, -1],
    }
    for side, arr in checks.items():
        missing = [i + 1 for i, v in enumerate(arr.tolist()) if not v]
        if not missing:
            continue
        if missing[-1] - missing[0] + 1 != len(missing):
            return None
        candidates.append((side, int(missing[0]), int(len(missing))))

    if len(candidates) != 1:
        return None

    side, gap_start, gap_width = candidates[0]
    if int(coords.shape[0]) != 2 * (h + w) - 4 - gap_width:
        return None

    if side == "top":
        gap_cells = [(r0, c0 + gap_start + i) for i in range(gap_width)]
    elif side == "bottom":
        gap_cells = [(r1, c0 + gap_start + i) for i in range(gap_width)]
    elif side == "left":
        gap_cells = [(r0 + gap_start + i, c0) for i in range(gap_width)]
    else:
        gap_cells = [(r0 + gap_start + i, c1) for i in range(gap_width)]

    return {
        "color": int(color),
        "bbox": (int(r0), int(c0), int(r1), int(c1)),
        "side": side,
        "gap_start": int(gap_start),
        "gap_width": int(gap_width),
        "gap_cells": tuple((int(r), int(c)) for r, c in gap_cells),
        "coords": tuple((int(r), int(c)) for r, c in coords.tolist()),
    }


def _find_frames(grid: np.ndarray, bg: int) -> list[dict]:
    frames = []
    for color in sorted(int(c) for c in np.unique(grid) if int(c) != int(bg)):
        for coords in _components(grid == int(color)):
            frame = _analyze_frame(coords, int(color))
            if frame is not None:
                frames.append(frame)
    frames.sort(key=lambda item: item["bbox"])
    return frames


def _extract_piece(grid: np.ndarray, coords: np.ndarray, bg: int) -> dict:
    r0, c0, r1, c1 = _bbox(coords)
    h = int(r1 - r0 + 1)
    w = int(c1 - c0 + 1)
    mask = np.zeros((h, w), dtype=bool)
    patch = np.full((h, w), int(bg), dtype=int)
    for r, c in coords.tolist():
        rr = int(r - r0)
        cc = int(c - c0)
        mask[rr, cc] = True
        patch[rr, cc] = int(grid[int(r), int(c)])
    colors = frozenset(int(grid[int(r), int(c)]) for r, c in coords.tolist())
    return {
        "bbox": (int(r0), int(c0), int(r1), int(c1)),
        "mask": mask,
        "patch": patch,
        "coords": tuple((int(r), int(c)) for r, c in coords.tolist()),
        "color_set": colors,
        "area": int(coords.shape[0]),
    }


def _find_candidate_pieces(grid: np.ndarray, frames: list[dict], bg: int) -> tuple[list[dict], set[tuple[int, int]]]:
    h, w = grid.shape
    frame_mask = np.zeros((h, w), dtype=bool)
    for frame in frames:
        for r, c in frame["coords"]:
            frame_mask[r, c] = True

    marker_positions: set[tuple[int, int]] = set()
    for frame in frames:
        r0, c0, r1, c1 = frame["bbox"]
        markers = []
        for r in range(r0 + 1, r1):
            for c in range(c0 + 1, c1):
                if frame_mask[r, c]:
                    continue
                v = int(grid[r, c])
                if v != int(bg):
                    markers.append((r, c))
                    marker_positions.add((r, c))
        frame["marker_positions"] = tuple(markers)
        frame["marker_colors"] = frozenset(int(grid[r, c]) for r, c in markers)

    candidate_mask = (grid != int(bg)) & (~frame_mask)
    for r, c in marker_positions:
        candidate_mask[r, c] = False

    pieces = [_extract_piece(grid, coords, bg) for coords in _components(candidate_mask)]
    pieces.sort(key=lambda item: (item["bbox"], item["area"]))
    return pieces, marker_positions


def _find_docking(frame: dict, piece: dict, occupied: np.ndarray, shape: tuple[int, int]) -> list[dict]:
    h, w = shape
    r0, c0, r1, c1 = frame["bbox"]
    gap_set = set(frame["gap_cells"])
    gap_width = int(frame["gap_width"])
    gap_r0 = min(r for r, _ in gap_set)
    gap_c0 = min(c for _, c in gap_set)

    placements = []
    for k in range(4):
        patch = np.rot90(piece["patch"], k).copy()
        mask = np.rot90(piece["mask"], k).copy()
        coords = [(int(r), int(c)) for r, c in np.argwhere(mask)]
        if frame["side"] == "top":
            edge_r = max(r for r, _ in coords)
            edge = sorted(c for r, c in coords if r == edge_r)
            if len(edge) != gap_width or edge[-1] - edge[0] + 1 != gap_width:
                continue
            dr = int(r0 - edge_r)
            dc = int(gap_c0 - edge[0])
            edge_abs = {(dr + edge_r, dc + c) for c in edge}
        elif frame["side"] == "bottom":
            edge_r = min(r for r, _ in coords)
            edge = sorted(c for r, c in coords if r == edge_r)
            if len(edge) != gap_width or edge[-1] - edge[0] + 1 != gap_width:
                continue
            dr = int(r1 - edge_r)
            dc = int(gap_c0 - edge[0])
            edge_abs = {(dr + edge_r, dc + c) for c in edge}
        elif frame["side"] == "left":
            edge_c = max(c for _, c in coords)
            edge = sorted(r for r, c in coords if c == edge_c)
            if len(edge) != gap_width or edge[-1] - edge[0] + 1 != gap_width:
                continue
            dr = int(gap_r0 - edge[0])
            dc = int(c0 - edge_c)
            edge_abs = {(dr + r, dc + edge_c) for r in edge}
        else:
            edge_c = min(c for _, c in coords)
            edge = sorted(r for r, c in coords if c == edge_c)
            if len(edge) != gap_width or edge[-1] - edge[0] + 1 != gap_width:
                continue
            dr = int(gap_r0 - edge[0])
            dc = int(c1 - edge_c)
            edge_abs = {(dr + r, dc + edge_c) for r in edge}

        if edge_abs != gap_set:
            continue

        translated = [(dr + r, dc + c) for r, c in coords]
        if any(not (0 <= rr < h and 0 <= cc < w) for rr, cc in translated):
            continue
        if any(
            r0 <= rr <= r1 and c0 <= cc <= c1 and (rr, cc) not in gap_set
            for rr, cc in translated
        ):
            continue
        if any(occupied[rr, cc] for rr, cc in translated):
            continue

        placements.append({"rotation": int(k), "patch": patch, "mask": mask, "origin": (int(dr), int(dc))})
    return placements


def main(grid: np.ndarray) -> np.ndarray:
    """
    Match each frame's interior color key to one external puzzle piece and dock that piece into the frame's side gap.

    Input structure:
    - The background is black, and several monochrome rectangular frames sit on it with one contiguous opening on exactly one side.
    - Each frame encloses a few single-pixel colored markers that lie strictly inside the frame's bounding box.
    - Elsewhere on the grid are several small multicolor 4-connected puzzle pieces; each piece has a unique color set.

    Rule:
    - For each frame, collect the set of marker colors inside it.
    - Find the unique external puzzle piece whose color set matches that marker set exactly.
    - Detect which side of the frame has the opening, rotate the matched piece by 0/90/180/270 degrees, and place it so one edge of the piece exactly fills the opening while the rest of the piece remains outside the frame.
    - Remove all interior markers and erase every external candidate piece from its original position; keep only the frames and the docked matched pieces.

    Key insight:
    - The interior marker colors identify which piece belongs to each frame, while the frame opening's side and width determine the only rotation that can physically dock that piece.
    """
    g = np.asarray(grid, dtype=int)
    h, w = g.shape
    bg = get_bg(g)

    frames = _find_frames(g, bg)
    assert frames, "expected at least one frame"
    pieces, _marker_positions = _find_candidate_pieces(g, frames, bg)

    by_color_set: dict[frozenset[int], list[dict]] = {}
    for piece in pieces:
        by_color_set.setdefault(piece["color_set"], []).append(piece)

    out = np.full((h, w), int(bg), dtype=int)
    occupied = np.zeros((h, w), dtype=bool)
    for frame in frames:
        for r, c in frame["coords"]:
            out[r, c] = int(frame["color"])
            occupied[r, c] = True

    used_piece_ids = set()
    for frame in frames:
        matches = by_color_set.get(frame["marker_colors"], [])
        assert len(matches) == 1, "each frame must match exactly one piece by color set"
        piece = matches[0]
        used_piece_ids.add(id(piece))
        placements = _find_docking(frame, piece, occupied, (h, w))
        assert len(placements) == 1, "each matched piece must have a unique valid docking rotation"
        placement = placements[0]
        dr, dc = placement["origin"]
        for rr, cc in np.argwhere(placement["mask"]):
            r = int(dr + rr)
            c = int(dc + cc)
            out[r, c] = int(placement["patch"][rr, cc])
            occupied[r, c] = True

    return out


def _make_frame_coords(frame_h: int, frame_w: int, side: str, gap_start: int, gap_width: int) -> tuple[list[tuple[int, int]], list[tuple[int, int]]]:
    coords = []
    for c in range(frame_w):
        if not (side == "top" and gap_start <= c < gap_start + gap_width):
            coords.append((0, c))
        if not (side == "bottom" and gap_start <= c < gap_start + gap_width):
            coords.append((frame_h - 1, c))
    for r in range(1, frame_h - 1):
        if not (side == "left" and gap_start <= r < gap_start + gap_width):
            coords.append((r, 0))
        if not (side == "right" and gap_start <= r < gap_start + gap_width):
            coords.append((r, frame_w - 1))

    if side == "top":
        gap = [(0, gap_start + i) for i in range(gap_width)]
    elif side == "bottom":
        gap = [(frame_h - 1, gap_start + i) for i in range(gap_width)]
    elif side == "left":
        gap = [(gap_start + i, 0) for i in range(gap_width)]
    else:
        gap = [(gap_start + i, frame_w - 1) for i in range(gap_width)]
    return coords, gap


def _local_docked_coords(
    mask: np.ndarray, side: str, gap_start: int, frame_h: int, frame_w: int
) -> tuple[list[tuple[int, int]], tuple[int, int]]:
    coords = [(int(r), int(c)) for r, c in np.argwhere(mask)]
    if side == "top":
        edge_r = max(r for r, _ in coords)
        edge = sorted(c for r, c in coords if r == edge_r)
        dr = -edge_r
        dc = int(gap_start - edge[0])
    elif side == "bottom":
        edge_r = min(r for r, _ in coords)
        edge = sorted(c for r, c in coords if r == edge_r)
        dr = int(frame_h - 1 - edge_r)
        dc = int(gap_start - edge[0])
    elif side == "left":
        edge_c = max(c for _, c in coords)
        edge = sorted(r for r, c in coords if c == edge_c)
        dr = int(gap_start - edge[0])
        dc = -edge_c
    else:
        edge_c = min(c for _, c in coords)
        edge = sorted(r for r, c in coords if c == edge_c)
        dr = int(gap_start - edge[0])
        dc = int(frame_w - 1 - edge_c)
    return [(int(dr + r), int(dc + c)) for r, c in coords], (int(dr), int(dc))


def _expand_bbox(bbox: tuple[int, int, int, int], pad: int, shape: tuple[int, int]) -> tuple[int, int, int, int] | None:
    h, w = shape
    r0 = max(0, int(bbox[0] - pad))
    c0 = max(0, int(bbox[1] - pad))
    r1 = min(h - 1, int(bbox[2] + pad))
    c1 = min(w - 1, int(bbox[3] + pad))
    if r0 > r1 or c0 > c1:
        return None
    return (r0, c0, r1, c1)


def _rect_free(blocked: np.ndarray, bbox: tuple[int, int, int, int]) -> bool:
    r0, c0, r1, c1 = bbox
    return not blocked[r0 : r1 + 1, c0 : c1 + 1].any()


def _mark_rect(blocked: np.ndarray, bbox: tuple[int, int, int, int]) -> None:
    r0, c0, r1, c1 = bbox
    blocked[r0 : r1 + 1, c0 : c1 + 1] = True


def _stamp_patch(grid: np.ndarray, patch: np.ndarray, top: int, left: int, bg: int) -> None:
    for rr, cc in np.argwhere(patch != int(bg)):
        grid[int(top + rr), int(left + cc)] = int(patch[rr, cc])


def _build_colored_base(mask: np.ndarray, colors: tuple[int, ...], rng: random.Random) -> np.ndarray:
    patch = np.full(mask.shape, BG_COLOR, dtype=int)
    coords = [(int(r), int(c)) for r, c in np.argwhere(mask)]
    rng.shuffle(coords)
    color_list = list(colors)
    rng.shuffle(color_list)
    for (r, c), color in zip(coords, color_list):
        patch[r, c] = int(color)
    return patch


def _sample_free_piece_location(
    blocked: np.ndarray,
    patch: np.ndarray,
    rng: random.Random,
) -> tuple[int, int] | None:
    h, w = blocked.shape
    ph, pw = patch.shape
    positions = []
    for top in range(0, h - ph + 1):
        for left in range(0, w - pw + 1):
            ok = True
            cells = np.argwhere(patch != BG_COLOR)
            for rr, cc in cells:
                r = int(top + rr)
                c = int(left + cc)
                if blocked[r, c]:
                    ok = False
                    break
            if not ok:
                continue
            bbox = (int(top), int(left), int(top + ph - 1), int(left + pw - 1))
            pad_box = _expand_bbox(bbox, 1, blocked.shape)
            if pad_box is None or not _rect_free(blocked, pad_box):
                continue
            positions.append((int(top), int(left)))
    if not positions:
        return None
    return rng.choice(positions)


def _build_expected(shape: tuple[int, int], frame_specs: list[dict]) -> np.ndarray:
    h, w = shape
    out = np.full((h, w), BG_COLOR, dtype=int)
    for spec in frame_specs:
        for rr, cc in spec["frame_coords_global"]:
            out[rr, cc] = FRAME_COLOR
        patch = np.rot90(spec["base_patch"], spec["dock_rotation"]).copy()
        dr, dc = spec["dock_origin_global"]
        for rr, cc in np.argwhere(patch != BG_COLOR):
            out[int(dr + rr), int(dc + cc)] = int(patch[rr, cc])
    return out


def _gen_attempt(rng: random.Random) -> np.ndarray | None:
    h = rng.randint(20, 30)
    w = rng.randint(20, 30)
    frame_count = rng.randint(1, 3)
    distractor_count = rng.randint(1, 3)
    total_pieces = frame_count + distractor_count

    color_sets = rng.sample(list(combinations(COLOR_POOL, 4)), total_pieces)
    blocked_output = np.zeros((h, w), dtype=bool)
    frame_specs = []

    for idx in range(frame_count):
        placed = False
        for _ in range(300):
            side = rng.choice(SIDES)
            option = rng.choice(SHAPE_OPTIONS[side])
            gap_width = int(option["gap_width"])
            min_h = max(5, gap_width + 3 if side in ("left", "right") else 5)
            min_w = max(5, gap_width + 3 if side in ("top", "bottom") else 5)
            frame_h = rng.randint(min_h, 9)
            frame_w = rng.randint(min_w, 9)
            if (frame_h - 2) * (frame_w - 2) < 4:
                continue

            if side in ("top", "bottom"):
                gap_start = rng.randint(1, frame_w - gap_width - 1)
            else:
                gap_start = rng.randint(1, frame_h - gap_width - 1)

            frame_coords_local, gap_cells_local = _make_frame_coords(frame_h, frame_w, side, gap_start, gap_width)
            dock_mask = np.rot90(option["base_mask"], option["dock_rotation"]).copy()
            dock_coords_local, dock_origin_local = _local_docked_coords(
                dock_mask, side, gap_start, frame_h, frame_w
            )
            combined = frame_coords_local + dock_coords_local

            min_r = min(r for r, _ in combined)
            min_c = min(c for _, c in combined)
            max_r = max(r for r, _ in combined)
            max_c = max(c for _, c in combined)
            obj_h = int(max_r - min_r + 1)
            obj_w = int(max_c - min_c + 1)
            if obj_h + 2 > h or obj_w + 2 > w:
                continue

            top = rng.randint(1, h - obj_h - 1)
            left = rng.randint(1, w - obj_w - 1)
            shift_r = int(top - min_r)
            shift_c = int(left - min_c)

            combined_global = [(shift_r + r, shift_c + c) for r, c in combined]
            bbox_global = (
                min(r for r, _ in combined_global),
                min(c for _, c in combined_global),
                max(r for r, _ in combined_global),
                max(c for _, c in combined_global),
            )
            pad_box = _expand_bbox(bbox_global, 1, (h, w))
            if pad_box is None or not _rect_free(blocked_output, pad_box):
                continue

            frame_coords_global = [(shift_r + r, shift_c + c) for r, c in frame_coords_local]
            gap_cells_global = [(shift_r + r, shift_c + c) for r, c in gap_cells_local]
            frame_r0 = min(r for r, _ in frame_coords_global + gap_cells_global)
            frame_c0 = min(c for _, c in frame_coords_global + gap_cells_global)
            frame_r1 = max(r for r, _ in frame_coords_global + gap_cells_global)
            frame_c1 = max(c for _, c in frame_coords_global + gap_cells_global)
            interior_cells = [
                (r, c)
                for r in range(frame_r0 + 1, frame_r1)
                for c in range(frame_c0 + 1, frame_c1)
            ]
            if len(interior_cells) < 4:
                continue

            _mark_rect(blocked_output, pad_box)
            frame_specs.append(
                {
                    "side": side,
                    "gap_width": gap_width,
                    "gap_start": gap_start,
                    "frame_h": frame_h,
                    "frame_w": frame_w,
                    "bbox": (frame_r0, frame_c0, frame_r1, frame_c1),
                    "frame_coords_global": frame_coords_global,
                    "gap_cells_global": gap_cells_global,
                    "dock_coords_local": dock_coords_local,
                    "dock_origin_global": (
                        int(shift_r + dock_origin_local[0]),
                        int(shift_c + dock_origin_local[1]),
                    ),
                    "base_mask": option["base_mask"].copy(),
                    "dock_rotation": int(option["dock_rotation"]),
                    "color_set": tuple(int(v) for v in color_sets[idx]),
                }
            )
            placed = True
            break
        if not placed:
            return None

    grid = np.full((h, w), BG_COLOR, dtype=int)
    blocked_input = np.zeros((h, w), dtype=bool)
    for spec in frame_specs:
        for r, c in spec["frame_coords_global"]:
            grid[r, c] = FRAME_COLOR
        box = _expand_bbox(spec["bbox"], 1, (h, w))
        if box is not None:
            _mark_rect(blocked_input, box)
        markers = rng.sample(
            [
                (r, c)
                for r in range(spec["bbox"][0] + 1, spec["bbox"][2])
                for c in range(spec["bbox"][1] + 1, spec["bbox"][3])
            ],
            4,
        )
        marker_colors = list(spec["color_set"])
        rng.shuffle(marker_colors)
        for (r, c), color in zip(markers, marker_colors):
            grid[r, c] = int(color)
            blocked_input[r, c] = True

    piece_specs = []
    for spec in frame_specs:
        base_patch = _build_colored_base(spec["base_mask"], spec["color_set"], rng)
        spec["base_patch"] = base_patch
        piece_specs.append({"base_patch": base_patch, "kind": "matched"})

    for idx in range(distractor_count):
        color_set = tuple(int(v) for v in color_sets[frame_count + idx])
        shape_name = rng.choice(list(BASE_SHAPES))
        base_mask = BASE_SHAPES[shape_name]
        base_patch = _build_colored_base(base_mask, color_set, rng)
        piece_specs.append({"base_patch": base_patch, "kind": "distractor"})

    rng.shuffle(piece_specs)
    for piece_spec in piece_specs:
        patch = np.rot90(piece_spec["base_patch"], rng.randint(0, 3)).copy()
        loc = _sample_free_piece_location(blocked_input, patch, rng)
        if loc is None:
            return None
        top, left = loc
        _stamp_patch(grid, patch, top, left, BG_COLOR)
        bbox = (int(top), int(left), int(top + patch.shape[0] - 1), int(left + patch.shape[1] - 1))
        pad_box = _expand_bbox(bbox, 1, (h, w))
        if pad_box is None:
            return None
        _mark_rect(blocked_input, pad_box)

    expected = _build_expected((h, w), frame_specs)
    try:
        solved = main(grid)
    except AssertionError:
        return None
    if not np.array_equal(solved, expected):
        return None
    return grid


def generate_input() -> np.ndarray:
    rng = random.Random()
    for _ in range(500):
        grid = _gen_attempt(rng)
        if grid is not None:
            return grid
    raise RuntimeError("failed to generate a valid input")
