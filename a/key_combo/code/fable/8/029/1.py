from __future__ import annotations

import random

import numpy as np


class Color:
    BLACK = 0
    BLUE = 1
    TEAL = 8


SIDE_TO_VEC = {
    "top": (-1, 0),
    "bottom": (1, 0),
    "left": (0, -1),
    "right": (0, 1),
}
OPPOSITE_SIDE = {
    "top": "bottom",
    "bottom": "top",
    "left": "right",
    "right": "left",
}
SIDES = ("top", "bottom", "left", "right")

GRID_MIN = 18
GRID_MAX = 30
CHAIN_LEN_RANGE = (2, 5)
FRAME_H_RANGE = (4, 7)
FRAME_W_RANGE = (4, 7)
SEGMENT_LEN_RANGE = (1, 8)
EXTRA_FRAME_RANGE = (0, 2)
TERMINAL_WALL_PROB = 0.65
MAX_GENERATE_ATTEMPTS = 500


def _in_bounds(shape: tuple[int, int], r: int, c: int) -> bool:
    return 0 <= int(r) < int(shape[0]) and 0 <= int(c) < int(shape[1])


def _neighbors4(r: int, c: int) -> list[tuple[int, int]]:
    return [(r - 1, c), (r + 1, c), (r, c - 1), (r, c + 1)]


def _bbox_coords(coords: list[tuple[int, int]]) -> tuple[int, int, int, int]:
    rows = [r for r, _ in coords]
    cols = [c for _, c in coords]
    return min(rows), min(cols), max(rows), max(cols)


def _components_mask(mask: np.ndarray) -> list[list[tuple[int, int]]]:
    h, w = mask.shape
    seen = np.zeros((h, w), dtype=bool)
    comps: list[list[tuple[int, int]]] = []
    for r in range(h):
        for c in range(w):
            if not mask[r, c] or seen[r, c]:
                continue
            stack = [(r, c)]
            seen[r, c] = True
            comp: list[tuple[int, int]] = []
            while stack:
                rr, cc = stack.pop()
                comp.append((rr, cc))
                for nr, nc in _neighbors4(rr, cc):
                    if _in_bounds(mask.shape, nr, nc) and mask[nr, nc] and not seen[nr, nc]:
                        seen[nr, nc] = True
                        stack.append((nr, nc))
            comps.append(comp)
    return comps


def _border_cells(bbox: tuple[int, int, int, int]) -> list[tuple[int, int]]:
    r0, c0, r1, c1 = bbox
    cells: list[tuple[int, int]] = []
    for c in range(c0, c1 + 1):
        cells.append((r0, c))
        cells.append((r1, c))
    for r in range(r0 + 1, r1):
        cells.append((r, c0))
        cells.append((r, c1))
    return cells


def _gap_side(gap: tuple[int, int], bbox: tuple[int, int, int, int]) -> str:
    r, c = gap
    r0, c0, r1, c1 = bbox
    if r == r0:
        return "top"
    if r == r1:
        return "bottom"
    if c == c0:
        return "left"
    if c == c1:
        return "right"
    raise AssertionError("gap must lie on the bbox border")


def _fill_interior(grid: np.ndarray, bbox: tuple[int, int, int, int], color: int) -> None:
    r0, c0, r1, c1 = bbox
    if r1 - r0 >= 2 and c1 - c0 >= 2:
        grid[r0 + 1 : r1, c0 + 1 : c1] = int(color)


def _frame_gap_coord(bbox: tuple[int, int, int, int], side: str, offset: int) -> tuple[int, int]:
    r0, c0, r1, c1 = bbox
    if side == "top":
        return r0, c0 + offset
    if side == "bottom":
        return r1, c0 + offset
    if side == "left":
        return r0 + offset, c0
    if side == "right":
        return r0 + offset, c1
    raise AssertionError("unknown side")


def _bbox_from_border_cell(
    border_cell: tuple[int, int],
    side: str,
    height: int,
    width: int,
    offset: int,
) -> tuple[int, int, int, int]:
    r, c = border_cell
    if side == "top":
        return r, c - offset, r + height - 1, c - offset + width - 1
    if side == "bottom":
        return r - height + 1, c - offset, r, c - offset + width - 1
    if side == "left":
        return r - offset, c, r - offset + height - 1, c + width - 1
    if side == "right":
        return r - offset, c - width + 1, r - offset + height - 1, c
    raise AssertionError("unknown side")


def _valid_offsets(side: str, height: int, width: int) -> list[int]:
    limit = width - 2 if side in ("top", "bottom") else height - 2
    return list(range(1, limit + 1))


def _bbox_in_bounds(bbox: tuple[int, int, int, int], shape: tuple[int, int]) -> bool:
    r0, c0, r1, c1 = bbox
    return 0 <= r0 <= r1 < shape[0] and 0 <= c0 <= c1 < shape[1]


def _boxes_overlap(a: tuple[int, int, int, int], b: tuple[int, int, int, int], pad: int = 0) -> bool:
    ar0, ac0, ar1, ac1 = a
    br0, bc0, br1, bc1 = b
    return not (ar1 + pad < br0 or br1 + pad < ar0 or ac1 + pad < bc0 or bc1 + pad < ac0)


def _cell_in_box(cell: tuple[int, int], bbox: tuple[int, int, int, int]) -> bool:
    r, c = cell
    r0, c0, r1, c1 = bbox
    return r0 <= r <= r1 and c0 <= c <= c1


def _segment_cells(
    start: tuple[int, int],
    direction: tuple[int, int],
    steps: int,
    *,
    include_start: bool = True,
    include_end: bool = True,
) -> list[tuple[int, int]]:
    dr, dc = direction
    begin = 0 if include_start else 1
    end = steps if include_end else steps - 1
    return [(start[0] + dr * k, start[1] + dc * k) for k in range(begin, end + 1)]


def _interior_values(grid: np.ndarray, bbox: tuple[int, int, int, int]) -> np.ndarray:
    r0, c0, r1, c1 = bbox
    if r1 - r0 < 2 or c1 - c0 < 2:
        return np.zeros((0, 0), dtype=int)
    return grid[r0 + 1 : r1, c0 + 1 : c1]


def _frame_is_connected(bbox: tuple[int, int, int, int], gaps: tuple[tuple[int, int], ...]) -> bool:
    if len(gaps) <= 1:
        return True

    r0, c0, r1, c1 = bbox
    side_a = _gap_side(gaps[0], bbox)
    side_b = _gap_side(gaps[1], bbox)
    if side_b == OPPOSITE_SIDE[side_a]:
        return True

    gap_by_side = {side_a: gaps[0], side_b: gaps[1]}
    top = gap_by_side.get("top")
    bottom = gap_by_side.get("bottom")
    left = gap_by_side.get("left")
    right = gap_by_side.get("right")

    if top is not None and left is not None and top[1] == c0 + 1 and left[0] == r0 + 1:
        return False
    if top is not None and right is not None and top[1] == c1 - 1 and right[0] == r0 + 1:
        return False
    if bottom is not None and left is not None and bottom[1] == c0 + 1 and left[0] == r1 - 1:
        return False
    if bottom is not None and right is not None and bottom[1] == c1 - 1 and right[0] == r1 - 1:
        return False
    return True


def _analyze_grid(grid: np.ndarray) -> dict:
    g = np.asarray(grid, dtype=int)
    if g.ndim != 2:
        raise AssertionError("expected a 2D grid")

    blue_cells = [(int(r), int(c)) for r, c in np.argwhere(g == Color.BLUE)]
    if not blue_cells:
        raise AssertionError("expected at least one blue frame")

    blue_set = set(blue_cells)
    rows = sorted({r for r, _ in blue_cells})
    cols = sorted({c for _, c in blue_cells})

    def candidate_from_bbox(bbox: tuple[int, int, int, int]) -> dict | None:
        r0, c0, r1, c1 = bbox
        if r1 - r0 < 2 or c1 - c0 < 2:
            return None
        corners = {(r0, c0), (r0, c1), (r1, c0), (r1, c1)}
        if any(g[corner] != Color.BLUE for corner in corners):
            return None

        border = _border_cells(bbox)
        missing = []
        blue_on_border = set()
        for cell in border:
            value = int(g[cell])
            if value == Color.BLUE:
                blue_on_border.add(cell)
            else:
                if cell in corners:
                    return None
                if value != Color.BLACK:
                    return None
                missing.append(cell)

        if len(missing) not in (1, 2):
            return None
        gap_sides = {gap: _gap_side(gap, bbox) for gap in missing}
        if len(set(gap_sides.values())) != len(missing):
            return None

        interior = _interior_values(g, bbox)
        if np.any(interior == Color.BLUE):
            return None
        if len(missing) == 1:
            if not np.all(interior == Color.TEAL):
                return None
        elif interior.size and not np.all(interior == Color.BLACK):
            return None

        return {
            "bbox": bbox,
            "gaps": tuple(missing),
            "gap_sides": gap_sides,
            "interior": interior.copy(),
            "blue_cells": frozenset(blue_on_border),
        }

    candidates_by_bbox: dict[tuple[int, int, int, int], dict] = {}
    for r0 in rows:
        for r1 in rows:
            if r1 - r0 < 2:
                continue
            for c0 in cols:
                if (r0, c0) not in blue_set or (r1, c0) not in blue_set:
                    continue
                for c1 in cols:
                    if c1 - c0 < 2:
                        continue
                    if (r0, c1) not in blue_set or (r1, c1) not in blue_set:
                        continue
                    bbox = (r0, c0, r1, c1)
                    if bbox in candidates_by_bbox:
                        continue
                    candidate = candidate_from_bbox(bbox)
                    if candidate is not None:
                        candidates_by_bbox[bbox] = candidate

    candidates = list(candidates_by_bbox.values())
    if not candidates:
        raise AssertionError("could not identify any valid frame rectangles")

    cover_map: dict[tuple[int, int], list[int]] = {cell: [] for cell in blue_set}
    for idx, candidate in enumerate(candidates):
        for cell in candidate["blue_cells"]:
            cover_map[cell].append(idx)
    if any(not cover_map[cell] for cell in blue_set):
        raise AssertionError("some blue pixels are not covered by any frame candidate")

    def select_frames(uncovered: set[tuple[int, int]], chosen: list[int]) -> list[int] | None:
        if not uncovered:
            return chosen
        pivot = min(uncovered, key=lambda cell: len(cover_map[cell]))
        for idx in sorted(cover_map[pivot], key=lambda i: len(candidates[i]["blue_cells"]), reverse=True):
            cand = candidates[idx]
            if not cand["blue_cells"].issubset(uncovered):
                continue
            if any(_boxes_overlap(cand["bbox"], candidates[j]["bbox"], pad=0) for j in chosen):
                continue
            result = select_frames(uncovered - set(cand["blue_cells"]), chosen + [idx])
            if result is not None:
                return result
        return None

    selected = select_frames(set(blue_set), [])
    if selected is None:
        raise AssertionError("failed to assign blue pixels to non-overlapping frames")

    frames = [candidates[idx] for idx in selected]
    gap_map: dict[tuple[int, int], int] = {}
    for idx, frame in enumerate(frames):
        for gap in frame["gaps"]:
            if gap in gap_map:
                raise AssertionError("gap cells must be unique")
            gap_map[gap] = idx

    source_ids = [
        idx
        for idx, frame in enumerate(frames)
        if len(frame["gaps"]) == 1 and np.all(frame["interior"] == Color.TEAL)
    ]
    if len(source_ids) != 1:
        raise AssertionError("expected exactly one teal-filled source frame with one gap")

    source_id = source_ids[0]
    for idx, frame in enumerate(frames):
        if idx == source_id:
            continue
        if len(frame["gaps"]) != 2:
            raise AssertionError("non-source frames must have exactly two gaps")
        if frame["interior"].size and not np.all(frame["interior"] == Color.BLACK):
            raise AssertionError("non-source frame interiors must start black")

    return {"frames": frames, "source_id": source_id, "gap_map": gap_map}


def _solve(input_grid: np.ndarray, *, return_trace: bool = False):
    g = np.asarray(input_grid, dtype=int)
    info = _analyze_grid(g)
    frames = info["frames"]
    gap_map = info["gap_map"]

    out = g.copy()
    current_id = info["source_id"]
    visited_ids = {current_id}
    visited_bboxes = [frames[current_id]["bbox"]]
    _fill_interior(out, frames[current_id]["bbox"], Color.TEAL)
    current_gap = frames[current_id]["gaps"][0]
    segments = []

    for _ in range(len(frames) + 1):
        side = frames[current_id]["gap_sides"][current_gap]
        dr, dc = SIDE_TO_VEC[side]
        painted: list[tuple[int, int]] = []
        r, c = current_gap

        while _in_bounds(g.shape, r, c):
            target_id = None
            if (r, c) in gap_map and gap_map[(r, c)] != current_id:
                candidate = gap_map[(r, c)]
                entry_side = frames[candidate]["gap_sides"][(r, c)]
                if SIDE_TO_VEC[entry_side] == (-dr, -dc):
                    target_id = candidate

            if target_id is not None:
                if target_id in visited_ids:
                    raise AssertionError("chain revisits a frame")
                out[r, c] = Color.TEAL
                painted.append((r, c))
                _fill_interior(out, frames[target_id]["bbox"], Color.TEAL)
                visited_ids.add(target_id)
                visited_bboxes.append(frames[target_id]["bbox"])
                segments.append(
                    {
                        "stop": "frame",
                        "painted": tuple(painted),
                        "target_bbox": frames[target_id]["bbox"],
                    }
                )
                current_id = target_id
                current_gap = next(gap for gap in frames[target_id]["gaps"] if gap != (r, c))
                break

            if g[r, c] == Color.BLUE:
                segments.append({"stop": "wall", "painted": tuple(painted), "blocked_cell": (r, c)})
                return (out, {"visited_bboxes": visited_bboxes, "segments": segments}) if return_trace else out

            out[r, c] = Color.TEAL
            painted.append((r, c))
            r += dr
            c += dc
        else:
            segments.append({"stop": "boundary", "painted": tuple(painted)})
            return (out, {"visited_bboxes": visited_bboxes, "segments": segments}) if return_trace else out

    raise AssertionError("chain length exceeded the number of frames")


def main(input_grid: np.ndarray) -> np.ndarray:
    """
    Light hollow rectangle frames by following a single ray chain through aligned border gaps.

    Input structure:
    - Black background with several blue hollow rectangular frames whose borders are one pixel thick.
    - Exactly one source frame has a single non-corner gap and its full interior is already teal.
    - Every other frame has exactly two non-corner gaps on different sides; some untouched frames may align with a ray only at a solid wall cell.

    Rule:
    - Start from the source frame and fire a teal ray outward through its unique gap, perpendicular to that side.
    - The ray paints every traversed black cell teal, including the gap cell itself, and stops if it reaches the grid edge or a blue wall pixel.
    - If the ray hits another frame exactly at one of that frame's gaps, fill that frame's interior teal and continue from its other gap.
    - Repeat this one-in one-out propagation until the final ray leaves the grid or is blocked by a frame wall; frames never hit by the chain stay hollow.

    Key insight:
    - The missing border cells are doors rather than defects: only a door hit from the correct outside direction lets the ray enter and continue the domino chain.
    """
    return _solve(input_grid, return_trace=False)


def _draw_frame(grid: np.ndarray, bbox: tuple[int, int, int, int], gaps: tuple[tuple[int, int], ...]) -> None:
    gap_set = set(gaps)
    for cell in _border_cells(bbox):
        if cell not in gap_set:
            grid[cell] = Color.BLUE


def _render_input(
    shape: tuple[int, int],
    frames: list[dict],
    chain: list[dict],
) -> np.ndarray:
    grid = np.zeros(shape, dtype=int)
    for frame in frames:
        _draw_frame(grid, frame["bbox"], frame["gaps"])
    _fill_interior(grid, chain[0]["bbox"], Color.TEAL)
    return grid


def _render_expected_output(
    input_grid: np.ndarray,
    chain: list[dict],
    segments: list[dict],
) -> np.ndarray:
    out = np.array(input_grid, copy=True)
    for frame in chain:
        _fill_interior(out, frame["bbox"], Color.TEAL)
    for segment in segments:
        for r, c in segment["painted"]:
            out[r, c] = Color.TEAL
    return out


def _can_place_bbox(
    bbox: tuple[int, int, int, int],
    shape: tuple[int, int],
    frames: list[dict],
    reserved_cells: set[tuple[int, int]],
) -> bool:
    if not _bbox_in_bounds(bbox, shape):
        return False
    for frame in frames:
        if _boxes_overlap(bbox, frame["bbox"], pad=1):
            return False
    for cell in reserved_cells:
        if _cell_in_box(cell, bbox):
            return False
    return True


def _segment_clear(
    start: tuple[int, int],
    direction: tuple[int, int],
    steps: int,
    frames: list[dict],
    reserved_cells: set[tuple[int, int]],
) -> bool:
    if steps < 1:
        return False
    for cell in _segment_cells(start, direction, steps, include_start=False, include_end=False):
        if cell in reserved_cells:
            return False
        for frame in frames:
            if _cell_in_box(cell, frame["bbox"]):
                return False
    return True


def _sample_dims() -> tuple[int, int]:
    return random.randint(*FRAME_H_RANGE), random.randint(*FRAME_W_RANGE)


def _shuffle(seq):
    seq = list(seq)
    random.shuffle(seq)
    return seq


def _candidate_exit_sides(entry_side: str) -> list[str]:
    adjacent = [side for side in SIDES if side not in (entry_side, OPPOSITE_SIDE[entry_side])]
    opposite = [OPPOSITE_SIDE[entry_side]]
    pool = adjacent + opposite if random.random() < 0.75 else opposite + adjacent
    return _shuffle(pool)


def _add_extra_frames(shape: tuple[int, int], frames: list[dict], reserved_cells: set[tuple[int, int]]) -> None:
    target = random.randint(*EXTRA_FRAME_RANGE)
    for _ in range(target * 25):
        if target <= 0:
            return
        h, w = _sample_dims()
        r0 = random.randint(0, shape[0] - h)
        c0 = random.randint(0, shape[1] - w)
        bbox = (r0, c0, r0 + h - 1, c0 + w - 1)
        if not _can_place_bbox(bbox, shape, frames, reserved_cells):
            continue
        side1 = random.choice(SIDES)
        side2 = random.choice([side for side in SIDES if side != side1])
        gap1 = _frame_gap_coord(bbox, side1, random.choice(_valid_offsets(side1, h, w)))
        gap2 = _frame_gap_coord(bbox, side2, random.choice(_valid_offsets(side2, h, w)))
        frames.append({"bbox": bbox, "gaps": (gap1, gap2), "role": "extra"})
        target -= 1


def _build_instance() -> tuple[np.ndarray, np.ndarray, dict] | None:
    shape = (random.randint(GRID_MIN, GRID_MAX), random.randint(GRID_MIN, GRID_MAX))
    chain_len = random.randint(*CHAIN_LEN_RANGE)
    terminal_mode = "wall" if random.random() < TERMINAL_WALL_PROB else "boundary"
    chain: list[dict] = []
    frames: list[dict] = []
    segments: list[dict] = []
    reserved_cells: set[tuple[int, int]] = set()
    budget = [5000]

    def spend(cost: int = 1) -> bool:
        budget[0] -= cost
        return budget[0] >= 0

    def place_source() -> bool:
        for _ in range(80):
            if not spend():
                return False
            h, w = _sample_dims()
            r0 = random.randint(0, shape[0] - h)
            c0 = random.randint(0, shape[1] - w)
            bbox = (r0, c0, r0 + h - 1, c0 + w - 1)
            exit_side = random.choice(SIDES)
            gap = _frame_gap_coord(bbox, exit_side, random.choice(_valid_offsets(exit_side, h, w)))
            frame = {"bbox": bbox, "gaps": (gap,), "role": "source", "exit_side": exit_side, "exit_gap": gap}
            chain.append(frame)
            frames.append({"bbox": bbox, "gaps": (gap,), "role": "source"})
            return True
        return False

    def extend_chain(index: int) -> bool:
        current = chain[-1]
        if index == chain_len:
            return finish_chain()

        exit_gap = current["exit_gap"]
        exit_side = current["exit_side"]
        direction = SIDE_TO_VEC[exit_side]
        entry_side = OPPOSITE_SIDE[exit_side]

        lengths = _shuffle(range(SEGMENT_LEN_RANGE[0], SEGMENT_LEN_RANGE[1] + 1))
        for steps in lengths:
            if not spend():
                return False
            if not _segment_clear(exit_gap, direction, steps, frames, reserved_cells):
                continue
            entry_cell = (exit_gap[0] + direction[0] * steps, exit_gap[1] + direction[1] * steps)
            for _ in range(24):
                if not spend():
                    return False
                h, w = _sample_dims()
                entry_offsets = _shuffle(_valid_offsets(entry_side, h, w))
                for entry_offset in entry_offsets:
                    if not spend():
                        return False
                    bbox = _bbox_from_border_cell(entry_cell, entry_side, h, w, entry_offset)
                    if not _can_place_bbox(bbox, shape, frames, reserved_cells):
                        continue
                    for exit_side2 in _candidate_exit_sides(entry_side):
                        exit_offsets = _shuffle(_valid_offsets(exit_side2, h, w))
                        for exit_offset in exit_offsets:
                            if not spend():
                                return False
                            exit_gap2 = _frame_gap_coord(bbox, exit_side2, exit_offset)
                            segment = {
                                "painted": tuple(_segment_cells(exit_gap, direction, steps)),
                                "stop": "frame",
                                "target_bbox": bbox,
                            }
                            reserved_added = set(segment["painted"])
                            chain.append(
                                {
                                    "bbox": bbox,
                                    "gaps": (entry_cell, exit_gap2),
                                    "role": "chain",
                                    "entry_gap": entry_cell,
                                    "exit_gap": exit_gap2,
                                    "exit_side": exit_side2,
                                }
                            )
                            frames.append({"bbox": bbox, "gaps": (entry_cell, exit_gap2), "role": "chain"})
                            segments.append(segment)
                            reserved_cells.update(reserved_added)
                            if extend_chain(index + 1):
                                return True
                            reserved_cells.difference_update(reserved_added)
                            segments.pop()
                            frames.pop()
                            chain.pop()
        return False

    def finish_chain() -> bool:
        last = chain[-1]
        entry_gap = last.get("entry_gap")
        entry_side = None if entry_gap is None else _gap_side(entry_gap, last["bbox"])
        h = last["bbox"][2] - last["bbox"][0] + 1
        w = last["bbox"][3] - last["bbox"][1] + 1

        candidate_sides = [side for side in SIDES if side != entry_side]
        candidate_sides = _shuffle(candidate_sides)
        for exit_side in candidate_sides:
            if not spend():
                return False
            for exit_offset in _shuffle(_valid_offsets(exit_side, h, w)):
                if not spend():
                    return False
                exit_gap = _frame_gap_coord(last["bbox"], exit_side, exit_offset)
                if entry_gap is not None and exit_gap == entry_gap:
                    continue
                direction = SIDE_TO_VEC[exit_side]
                if terminal_mode == "boundary":
                    painted = []
                    rr, cc = exit_gap
                    ok = True
                    while _in_bounds(shape, rr, cc):
                        if (rr, cc) != exit_gap:
                            if (rr, cc) in reserved_cells:
                                ok = False
                                break
                            for frame in frames:
                                if frame["bbox"] != last["bbox"] and _cell_in_box((rr, cc), frame["bbox"]):
                                    ok = False
                                    break
                            if not ok:
                                break
                        painted.append((rr, cc))
                        rr += direction[0]
                        cc += direction[1]
                    if ok:
                        last["gaps"] = (entry_gap, exit_gap) if entry_gap is not None else (exit_gap,)
                        last["exit_gap"] = exit_gap
                        last["exit_side"] = exit_side
                        frames[-1]["gaps"] = last["gaps"]
                        segment = {"painted": tuple(painted), "stop": "boundary"}
                        segments.append(segment)
                        reserved_cells.update(painted)
                        return True
                else:
                    lengths = _shuffle(range(SEGMENT_LEN_RANGE[0], SEGMENT_LEN_RANGE[1] + 1))
                    hit_side = OPPOSITE_SIDE[exit_side]
                    for steps in lengths:
                        if not spend():
                            return False
                        if not _segment_clear(exit_gap, direction, steps, frames, reserved_cells):
                            continue
                        hit_cell = (exit_gap[0] + direction[0] * steps, exit_gap[1] + direction[1] * steps)
                        for _ in range(20):
                            if not spend():
                                return False
                            dh, dw = _sample_dims()
                            for hit_offset in _shuffle(_valid_offsets(hit_side, dh, dw)):
                                if not spend():
                                    return False
                                decoy_bbox = _bbox_from_border_cell(hit_cell, hit_side, dh, dw, hit_offset)
                                if not _can_place_bbox(decoy_bbox, shape, frames, reserved_cells):
                                    continue
                                gap_side1 = random.choice(SIDES)
                                gap_side2 = random.choice([side for side in SIDES if side != gap_side1])
                                gaps = []
                                for gap_side in (gap_side1, gap_side2):
                                    offsets = _shuffle(_valid_offsets(gap_side, dh, dw))
                                    chosen_gap = None
                                    for off in offsets:
                                        gap = _frame_gap_coord(decoy_bbox, gap_side, off)
                                        if gap != hit_cell:
                                            chosen_gap = gap
                                            break
                                    if chosen_gap is None:
                                        break
                                    gaps.append(chosen_gap)
                                if len(gaps) != 2:
                                    continue

                                last["gaps"] = (entry_gap, exit_gap) if entry_gap is not None else (exit_gap,)
                                last["exit_gap"] = exit_gap
                                last["exit_side"] = exit_side
                                frames[-1]["gaps"] = last["gaps"]
                                frames.append({"bbox": decoy_bbox, "gaps": tuple(gaps), "role": "decoy"})
                                segment = {
                                    "painted": tuple(_segment_cells(exit_gap, direction, steps, include_end=False)),
                                    "stop": "wall",
                                    "blocked_cell": hit_cell,
                                    "blocker_bbox": decoy_bbox,
                                }
                                segments.append(segment)
                                reserved_cells.update(segment["painted"])
                                return True
        return False

    if not place_source():
        return None
    if not extend_chain(1):
        return None

    _add_extra_frames(shape, frames, reserved_cells)
    input_grid = _render_input(shape, frames, chain)
    expected = _render_expected_output(input_grid, chain, segments)
    actual, trace = _solve(input_grid, return_trace=True)

    if not np.array_equal(actual, expected):
        return None
    if trace["visited_bboxes"] != [frame["bbox"] for frame in chain]:
        return None
    if len(trace["segments"]) != len(segments):
        return None
    for actual_seg, expected_seg in zip(trace["segments"], segments):
        if actual_seg["stop"] != expected_seg["stop"]:
            return None
        if tuple(actual_seg["painted"]) != tuple(expected_seg["painted"]):
            return None
        if actual_seg["stop"] == "frame" and actual_seg["target_bbox"] != expected_seg["target_bbox"]:
            return None
        if actual_seg["stop"] == "wall" and actual_seg["blocked_cell"] != expected_seg["blocked_cell"]:
            return None
    if len(chain) < 2:
        return None

    meta = {"chain": chain, "frames": frames, "segments": segments, "shape": shape}
    return input_grid, expected, meta


def generate_input() -> np.ndarray:
    for _ in range(MAX_GENERATE_ATTEMPTS):
        built = _build_instance()
        if built is not None:
            return built[0]
    raise RuntimeError("failed to generate a valid chained-frame ARC input")
