from __future__ import annotations

import random
from dataclasses import dataclass

import numpy as np


DIRS = {
    "U": (-1, 0),
    "D": (1, 0),
    "L": (0, -1),
    "R": (0, 1),
}

COLOR_POOL = [1, 2, 3, 4, 5, 6, 7, 8, 9]
GRID_SIZE_RANGE = (24, 29)
GEN_ATTEMPTS = 2000


@dataclass(frozen=True)
class Frame:
    color: int
    bbox: tuple[int, int, int, int]
    direction: str
    depth: int = 0


def _bbox(coords: np.ndarray) -> tuple[int, int, int, int]:
    r0, c0 = coords.min(axis=0)
    r1, c1 = coords.max(axis=0)
    return int(r0), int(c0), int(r1), int(c1)


def _components(mask: np.ndarray) -> list[np.ndarray]:
    h, w = mask.shape
    seen = np.zeros_like(mask, dtype=bool)
    comps: list[np.ndarray] = []
    for r in range(h):
        for c in range(w):
            if not mask[r, c] or seen[r, c]:
                continue
            stack = [(r, c)]
            seen[r, c] = True
            pts: list[tuple[int, int]] = []
            while stack:
                rr, cc = stack.pop()
                pts.append((rr, cc))
                for nr, nc in ((rr - 1, cc), (rr + 1, cc), (rr, cc - 1), (rr, cc + 1)):
                    if 0 <= nr < h and 0 <= nc < w and mask[nr, nc] and not seen[nr, nc]:
                        seen[nr, nc] = True
                        stack.append((nr, nc))
            comps.append(np.asarray(pts, dtype=int))
    return comps


def _ring_cells(bbox: tuple[int, int, int, int]) -> set[tuple[int, int]]:
    r0, c0, r1, c1 = bbox
    cells: set[tuple[int, int]] = set()
    for c in range(c0, c1 + 1):
        cells.add((r0, c))
        cells.add((r1, c))
    for r in range(r0 + 1, r1):
        cells.add((r, c0))
        cells.add((r, c1))
    return cells


def _tail_cells(bbox: tuple[int, int, int, int], direction: str) -> list[tuple[int, int]]:
    r0, c0, r1, c1 = bbox
    mr = (r0 + r1) // 2
    mc = (c0 + c1) // 2
    if direction == "U":
        return [(r0 - 1, mc), (r0 - 2, mc)]
    if direction == "D":
        return [(r1 + 1, mc), (r1 + 2, mc)]
    if direction == "L":
        return [(mr, c0 - 1), (mr, c0 - 2)]
    if direction == "R":
        return [(mr, c1 + 1), (mr, c1 + 2)]
    raise ValueError(direction)


def _shift_bbox(bbox: tuple[int, int, int, int], dr: int, dc: int) -> tuple[int, int, int, int]:
    r0, c0, r1, c1 = bbox
    return r0 + dr, c0 + dc, r1 + dr, c1 + dc


def _detect_frame(coords: np.ndarray, color: int) -> Frame | None:
    r0, c0, r1, c1 = _bbox(coords)
    local = coords - np.array([r0, c0], dtype=int)
    h = r1 - r0 + 1
    w = c1 - c0 + 1
    cells = {tuple(map(int, rc)) for rc in local}
    row_counts = [sum(1 for rr, _cc in cells if rr == row) for row in range(h)]
    col_counts = [sum(1 for _rr, cc in cells if cc == col) for col in range(w)]

    candidates: list[tuple[str, tuple[int, int, int, int], list[tuple[int, int]]]] = []
    if h >= 5 and row_counts[0] == 1 and row_counts[1] == 1:
        body = (2, 0, h - 1, w - 1)
        mc = (w - 1) // 2
        candidates.append(("U", body, [(1, mc), (0, mc)]))
    if h >= 5 and row_counts[h - 1] == 1 and row_counts[h - 2] == 1:
        body = (0, 0, h - 3, w - 1)
        mc = (w - 1) // 2
        candidates.append(("D", body, [(h - 2, mc), (h - 1, mc)]))
    if w >= 5 and col_counts[0] == 1 and col_counts[1] == 1:
        body = (0, 2, h - 1, w - 1)
        mr = (h - 1) // 2
        candidates.append(("L", body, [(mr, 1), (mr, 0)]))
    if w >= 5 and col_counts[w - 1] == 1 and col_counts[w - 2] == 1:
        body = (0, 0, h - 1, w - 3)
        mr = (h - 1) // 2
        candidates.append(("R", body, [(mr, w - 2), (mr, w - 1)]))

    for direction, body, tail in candidates:
        br0, bc0, br1, bc1 = body
        bh = br1 - br0 + 1
        bw = bc1 - bc0 + 1
        if bh < 5 or bw < 5 or bh % 2 == 0 or bw % 2 == 0:
            continue
        expected = _ring_cells(body) | set(tail)
        if cells == expected:
            return Frame(
                color=color,
                bbox=(r0 + br0, c0 + bc0, r0 + br1, c0 + bc1),
                direction=direction,
            )
    return None


def _parse_frames(grid: np.ndarray) -> list[Frame]:
    g = np.asarray(grid, dtype=int)
    frames: list[Frame] = []
    for color in sorted(int(v) for v in np.unique(g) if int(v) != 0):
        for comp in _components(g == color):
            frame = _detect_frame(comp, color)
            if frame is None:
                raise ValueError("found non-frame component")
            frames.append(frame)

    depths: list[int] = []
    for frame in frames:
        r0, c0, r1, c1 = frame.bbox
        containers = 0
        for other in frames:
            if other is frame:
                continue
            R0, C0, R1, C1 = other.bbox
            if R0 < r0 and C0 < c0 and r1 < R1 and c1 < C1:
                containers += 1
        depths.append(containers + 1)
    return [Frame(f.color, f.bbox, f.direction, depth) for f, depth in zip(frames, depths)]


def _draw_output(shape: tuple[int, int], frames: list[Frame]) -> np.ndarray:
    out = np.zeros(shape, dtype=int)
    ordered = sorted(frames, key=lambda frame: (frame.depth, frame.color, frame.bbox))
    h, w = shape
    for frame in ordered:
        dr, dc = DIRS[frame.direction]
        shift = frame.depth
        for r, c in _ring_cells(_shift_bbox(frame.bbox, dr * shift, dc * shift)):
            if 0 <= r < h and 0 <= c < w:
                out[r, c] = frame.color
    return out


def main(grid: np.ndarray) -> np.ndarray:
    """
    Slide nested rectangular frames by distances encoded by containment depth.

    Input structure:
    - The background is black.
    - Each foreground component is a one-pixel-thick rectangular frame with a two-pixel tail of the same color attached at the midpoint of one side.
    - Frames of the same color form 2-3 strictly nested layers, and separate color groups do not overlap.

    Rule:
    - Detect every rectangular frame and ignore its two-pixel tail except for the tail direction.
    - Compute each frame's nesting depth from strict rectangle containment: outermost depth 1, its child depth 2, and so on.
    - Translate each frame by exactly its depth along its own tail direction, then redraw only the translated frame borders, clipping any out-of-bounds cells.
    - Draw shallower frames first and deeper frames afterward.

    Key insight:
    - Tail length is constant and carries only direction; the hidden displacement magnitude comes entirely from the frame's topological depth in the nesting tree.
    """
    frames = _parse_frames(np.asarray(grid, dtype=int))
    return _draw_output(np.asarray(grid, dtype=int).shape, frames)


def _sample_odd(rng: random.Random, lo: int, hi: int) -> int:
    choices = [v for v in range(lo, hi + 1) if v % 2 == 1]
    return rng.choice(choices)


def _grow_parent(
    child_bbox: tuple[int, int, int, int],
    child_direction: str,
    rng: random.Random,
) -> tuple[int, int, int, int]:
    mins = {"top": 2, "bottom": 2, "left": 2, "right": 2}
    if child_direction == "U":
        mins["top"] = 4
    elif child_direction == "D":
        mins["bottom"] = 4
    elif child_direction == "L":
        mins["left"] = 4
    else:
        mins["right"] = 4

    top = mins["top"] + 2 * rng.randint(0, 1)
    bottom = mins["bottom"] + 2 * rng.randint(0, 1)
    left = mins["left"] + 2 * rng.randint(0, 1)
    right = mins["right"] + 2 * rng.randint(0, 1)
    r0, c0, r1, c1 = child_bbox
    return r0 - top, c0 - left, r1 + bottom, c1 + right


def _render_group_local(
    color: int,
    directions: list[str],
    innermost_hw: tuple[int, int],
    rng: random.Random,
) -> tuple[list[Frame], tuple[int, int], list[set[tuple[int, int]]]]:
    layers = len(directions)
    h, w = innermost_hw
    bboxes: list[tuple[int, int, int, int]] = [None] * layers  # type: ignore[assignment]
    bboxes[-1] = (0, 0, h - 1, w - 1)
    for idx in range(layers - 2, -1, -1):
        bboxes[idx] = _grow_parent(bboxes[idx + 1], directions[idx + 1], rng)

    outer_r0 = min(b[0] for b in bboxes)
    outer_c0 = min(b[1] for b in bboxes)
    shifted = [
        (r0 - outer_r0, c0 - outer_c0, r1 - outer_r0, c1 - outer_c0)
        for (r0, c0, r1, c1) in bboxes
    ]
    frames = [Frame(color, bbox, direction, idx + 1) for idx, (bbox, direction) in enumerate(zip(shifted, directions))]

    cells_per_frame: list[set[tuple[int, int]]] = []
    all_cells: list[tuple[int, int]] = []
    for frame in frames:
        cells = _ring_cells(frame.bbox) | set(_tail_cells(frame.bbox, frame.direction))
        cells_per_frame.append(cells)
        all_cells.extend(cells)

    min_r = min(r for r, _c in all_cells)
    min_c = min(c for _r, c in all_cells)
    if min_r < 0 or min_c < 0:
        shift_r = -min_r
        shift_c = -min_c
        frames = [
            Frame(color, _shift_bbox(frame.bbox, shift_r, shift_c), frame.direction, frame.depth)
            for frame in frames
        ]
        cells_per_frame = [
            {(r + shift_r, c + shift_c) for (r, c) in cells}
            for cells in cells_per_frame
        ]

    all_shifted = [rc for cells in cells_per_frame for rc in cells]
    height = max(r for r, _c in all_shifted) + 1
    width = max(c for _r, c in all_shifted) + 1
    return frames, (height, width), cells_per_frame


def _make_group(rng: random.Random, color: int, layers: int) -> tuple[list[Frame], tuple[int, int], list[set[tuple[int, int]]]]:
    for _ in range(180):
        directions = [rng.choice(list(DIRS)) for _ in range(layers)]
        if len(set(directions)) == 1 and rng.random() < 0.4:
            continue
        inner_hi = 5 if layers == 3 else 7
        inner_h = _sample_odd(rng, 5, inner_hi)
        inner_w = _sample_odd(rng, 5, inner_hi)
        frames, shape, cells = _render_group_local(color, directions, (inner_h, inner_w), rng)
        outer_h = frames[0].bbox[2] - frames[0].bbox[0] + 1
        outer_w = frames[0].bbox[3] - frames[0].bbox[1] + 1
        if layers == 2 and max(outer_h, outer_w) > 11:
            continue
        if layers == 3 and max(outer_h, outer_w) > 15:
            continue
        local = np.zeros(shape, dtype=int)
        for layer_cells in cells:
            for r, c in layer_cells:
                if local[r, c] != 0:
                    break
                local[r, c] = color
            else:
                continue
            break
        else:
            try:
                parsed = _parse_frames(local)
            except ValueError:
                continue
            if len(parsed) == layers:
                return frames, shape, cells
    raise RuntimeError("failed to build group")


def _place_groups(
    rng: random.Random,
    grid_shape: tuple[int, int],
    groups: list[tuple[list[Frame], tuple[int, int], list[set[tuple[int, int]]]]],
    want_clip: bool = False,
) -> tuple[np.ndarray, list[Frame]]:
    h, w = grid_shape
    placed_boxes: list[tuple[int, int, int, int]] = []
    placed_frames: list[Frame] = []
    grid = np.zeros((h, w), dtype=int)
    need_clip = want_clip

    for frames, (gh, gw), cells_per_frame in groups:
        candidates = [(r, c) for r in range(h - gh + 1) for c in range(w - gw + 1)]
        rng.shuffle(candidates)
        placed = False
        clip_pick = None
        normal_pick = None
        for top, left in candidates:
            box = (top, left, top + gh - 1, left + gw - 1)
            ok = True
            for r0, c0, r1, c1 in placed_boxes:
                if not (box[2] + 3 < r0 or r1 + 3 < box[0] or box[3] + 3 < c0 or c1 + 3 < box[1]):
                    ok = False
                    break
            if not ok:
                continue

            translated_frames = [
                Frame(frame.color, _shift_bbox(frame.bbox, top, left), frame.direction, frame.depth)
                for frame in frames
            ]
            translated_cells = [
                {(r + top, c + left) for (r, c) in cells}
                for cells in cells_per_frame
            ]
            if any(grid[r, c] != 0 for cells in translated_cells for (r, c) in cells):
                continue
            pick = (box, translated_frames, translated_cells)
            if need_clip and _has_clipping((h, w), translated_frames):
                clip_pick = pick
                break
            if normal_pick is None:
                normal_pick = pick
            if not need_clip:
                break
        chosen = clip_pick if clip_pick is not None else normal_pick
        if chosen is None:
            raise RuntimeError("failed to place groups")
        box, translated_frames, translated_cells = chosen
        if need_clip and clip_pick is not None:
            need_clip = False
        for frame_cells, frame in zip(translated_cells, translated_frames):
            for r, c in frame_cells:
                grid[r, c] = frame.color
            placed_frames.append(frame)
        placed_boxes.append(box)
        placed = True
        if not placed:
            raise RuntimeError("failed to place groups")
    if need_clip:
        raise RuntimeError("failed to force clipped placement")
    return grid, placed_frames


def _has_clipping(shape: tuple[int, int], frames: list[Frame]) -> bool:
    h, w = shape
    for frame in frames:
        dr, dc = DIRS[frame.direction]
        shift = frame.depth
        moved = _ring_cells(_shift_bbox(frame.bbox, dr * shift, dc * shift))
        if any(not (0 <= r < h and 0 <= c < w) for r, c in moved):
            return True
    return False


def _different_group_overlap(shape: tuple[int, int], frames: list[Frame]) -> bool:
    h, w = shape
    seen = np.zeros((h, w), dtype=int)
    for frame in sorted(frames, key=lambda x: x.depth):
        dr, dc = DIRS[frame.direction]
        moved = _ring_cells(_shift_bbox(frame.bbox, dr * frame.depth, dc * frame.depth))
        for r, c in moved:
            if not (0 <= r < h and 0 <= c < w):
                continue
            if seen[r, c] not in (0, frame.color):
                return True
            seen[r, c] = frame.color
    return False


def _self_overlap(frames: list[Frame]) -> bool:
    per_color: dict[int, set[tuple[int, int]]] = {}
    for frame in frames:
        dr, dc = DIRS[frame.direction]
        moved = _ring_cells(_shift_bbox(frame.bbox, dr * frame.depth, dc * frame.depth))
        bucket = per_color.setdefault(frame.color, set())
        if bucket & moved:
            return True
        bucket |= moved
    return False


def _layers_profile(rng: random.Random, n_groups: int) -> list[int]:
    if n_groups == 2:
        if rng.random() < 0.55:
            profile = [2, 3]
            rng.shuffle(profile)
            return profile
        return [2, 2] if rng.random() < 0.9 else [3, 3]
    if rng.random() < 0.75:
        return [2, 2, 2]
    else:
        profile = [2, 2, 3]
        rng.shuffle(profile)
        return profile


def generate_input() -> np.ndarray:
    for _ in range(GEN_ATTEMPTS):
        rng = random.Random()
        h = rng.randint(*GRID_SIZE_RANGE)
        w = rng.randint(*GRID_SIZE_RANGE)
        n_groups = rng.choices([2, 3], weights=[0.8, 0.2])[0]
        colors = rng.sample(COLOR_POOL, n_groups)
        groups = []
        bad_group = False
        for color, layers in zip(colors, _layers_profile(rng, n_groups)):
            try:
                groups.append(_make_group(rng, color, layers))
            except RuntimeError:
                bad_group = True
                break
        if bad_group:
            continue

        try:
            grid, placed_frames = _place_groups(rng, (h, w), groups, want_clip=False)
        except RuntimeError:
            continue

        if _different_group_overlap((h, w), placed_frames):
            continue
        if _self_overlap(placed_frames):
            continue

        try:
            parsed = _parse_frames(grid)
        except ValueError:
            continue
        parsed_signature = sorted((f.color, f.bbox, f.direction, f.depth) for f in parsed)
        planned_signature = sorted((f.color, f.bbox, f.direction, f.depth) for f in placed_frames)
        if parsed_signature != planned_signature:
            continue

        expected = _draw_output((h, w), placed_frames)
        actual = main(grid)
        if not np.array_equal(actual, expected):
            continue
        if len(parsed) < 4:
            continue
        return grid
    raise RuntimeError("failed to generate valid nested-frame input")
